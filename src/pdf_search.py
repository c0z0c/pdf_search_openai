"""
VectorStore MVP - 파이프라인 기반 RAG 시스템
Author: 14_3팀_김명환
Description: 2단계 검색을 활용한 고급 RAG 시스템
"""

import os
# OpenMP 중복 로드 경고 무시 (FAISS 사용 시 필요)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path

# 스크립트 실행 시 현재 디렉토리를 sys.path에 추가
SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# helper 모듈 import 전에 기본 라이브러리 import
import pickle
import hashlib
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable, Any

# pandas, numpy 등 기본 라이브러리
import pandas as pd
from tqdm import tqdm

# LangChain 관련
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import pymupdf4llm
import fitz
import re

# helper 모듈 import (try-except로 안전하게)
try:
    from . import helper_utils as hu
    from .helper_utils import *
    from . import helper_c0z0c_dev as helper
    from .helper_c0z0c_dev import *
except ImportError:
    try:
        import helper_utils as hu
        from helper_utils import *
        import helper_c0z0c_dev as helper
        from helper_c0z0c_dev import *
    except ImportError as e:
        logging.warning(f"helper 모듈 로드 실패: {e}")
        # 기본값 설정
        IS_COLAB = False
        RUN_PROCESS = {'심플데이타': False}
    
    def drive_root():
        """기본 drive_root 함수"""
        return str(Path.home())
    
    def timestamp(format_type='yymmdd_HHMMSS'):
        """기본 timestamp 함수"""
        from datetime import datetime
        return datetime.now().strftime('%y%m%d_%H%M%S')

# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 콘솔 핸들러 추가 (스크립트 실행 시 로그 출력)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# 0. OpenAI API 키 설정
openai_api_key = None
if IS_COLAB:
    from google.colab import userdata
    openai_api_key = userdata.get('OPENAI_API_KEY')
else:
    from dotenv import load_dotenv
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key:
    openai_api_key = openai_api_key.strip()
    os.environ["OPENAI_API_KEY"] = openai_api_key
    logger.debug(f"OPENAI_API_KEY [{openai_api_key[:4]}****{openai_api_key[-4:]}] 환경변수 설정 완료")
    logger.info("OPENAI_API_KEY 설정")
else:
    logger.warning("openai_api_key가 설정되지 않아 OpenAI 로그인 생략됨")

# 1. 데이터 구조 정의
@dataclass
class ChunkMetadata:
    """청크 메타데이터"""
    file_name: str  # 파일명만
    page: int       # 페이지 번호
    chunk_type: str  # 'original' or 'summary'
    chunk_index: int # 청크 인덱스
    original_chunk_index: Optional[int] = None  # 요약본의 경우 원본 청크 인덱스


@dataclass
class SearchResult:
    """검색 결과"""
    content: str # 청크 내용
    file_name: str # 파일명만
    page: int # 페이지 번호
    score: float # 유사도
    chunk_type: str # 청크 유형

# 2.0 파일 해시 관리 클래스

class FileHashManager:
    """파일 해시 관리 클래스"""

    def __init__(self, hash_algorithm: str = "sha256"):
        self.hash_algorithm = hash_algorithm

    def calculate_file_hash(self, file_path: str) -> str:
        """
        파일의 해시 값을 계산합니다.

        Args:
            file_path (str): 파일 경로

        Returns:
            str: 파일 해시 값
        """
        hash_func = hashlib.new(self.hash_algorithm)
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hash_func.update(chunk)
        return hash_func.hexdigest()

    def is_file_modified(self, file_path: str, stored_hash: str) -> bool:
        """
        파일이 수정되었는지 확인합니다.

        Args:
            file_path (str): 파일 경로
            stored_hash (str): 저장된 해시 값

        Returns:
            bool: 파일이 수정되었으면 True, 그렇지 않으면 False
        """
        current_hash = self.calculate_file_hash(file_path)
        return current_hash != stored_hash

# 2. 문서 처리 파이프라인
class DocumentProcessingPipeline:
    """PDF → Markdown → 청킹 파이프라인 (콜백 지원)"""

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Args:
            chunk_size: 청크 최대 크기
            chunk_overlap: 청크 간 중복 크기
            progress_callback: PDF 변환 진행 상황 콜백 함수
                호출 시 전달되는 딕셔너리:
                {
                    'file_name': str,
                    'current_page': int,
                    'total_pages': int,
                    'page_content_length': int,
                    'status': str  # 'processing', 'empty', 'failed'
                }
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.progress_callback = progress_callback
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            length_function=len,
        )
        self.hash_manager = FileHashManager()
        self.processed_files = {}
        
    def is_file_already_processed(self, file_path: str) -> bool:
        """
        파일이 이미 처리되었는지 확인합니다.

        Args:
            file_path (str): 파일 경로

        Returns:
            bool: 파일이 이미 처리되었으면 True, 그렇지 않으면 False
        """
        file_hash = self.hash_manager.calculate_file_hash(file_path)
        if file_hash in self.processed_files:
            logger.debug(f"파일이 이미 처리되었습니다: {file_path}")
            return True
        self.processed_files[file_hash] = file_path
        return False
        
    def save_markdown_from_pdf(self, pdf_path: str, pages_data: List[Dict]) -> None:
        """
        PDF 페이지별 데이터를 마크다운으로 저장합니다.

        Args:
            pdf_path (str): PDF 파일 경로
            pages_data (List[Dict]): 페이지별 {'page_num', 'content'} 딕셔너리 리스트
        """
        md_path = Path(pdf_path).with_suffix('.md')
        
        with open(md_path, 'w', encoding='utf-8') as md_file:
            for page_data in pages_data:
                page_num = page_data['page_num']
                content = page_data['content']
                md_file.write(f"\n\n--- 페이지 {page_num} ---\n\n")
                md_file.write(content)
        
        logger.debug(f"Markdown 파일 저장 완료: {md_path}")

    def markdown_with_progress(self, pdf_path: str) -> List[Dict]:
        """
        PDF를 페이지별로 Markdown 변환 (진행 상황 콜백 지원)
        
        Args:
            pdf_path (str): PDF 파일 경로
        
        Returns:
            List[Dict]: [{'page_num': int, 'content': str}, ...]
        """        
        file_name = Path(pdf_path).name
        
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)

        if RUN_PROCESS['심플데이타']:
            total_pages = min(total_pages, 10)

        pages_data = []
        with tqdm(total=total_pages, desc="PDF to Markdown", unit="page") as pbar:
            pbar_end_str = ""
            progress_callback_data = {
                            'file_name': file_name,
                            'current_page': 1,
                            'total_pages': 1,
                            'page_content_length': 0,
                            'status': "",
                            'error': ""
                        }
            for page_num in range(total_pages):
                try:
                    markdown = pymupdf4llm.to_markdown(
                        doc=pdf_path,
                        pages=[page_num]
                    )
                    
                    # 전처리
                    markdown = self.clean_markdown_text(markdown)
                    
                    # 상태 판단
                    if not markdown.strip():
                        status = 'empty'
                        pbar_end_str = f"empty"
                        markdown = f"[빈 페이지]"
                    else:
                        status = 'processing'
                        pbar_end_str = f"len={len(markdown)}"
                    
                    pages_data.append({
                        'page_num': page_num + 1,
                        'content': markdown
                    })
                    
                    progress_callback_data.update({
                        'file_name': file_name,
                        'current_page': page_num + 1,
                        'total_pages': total_pages,
                        'page_content_length': len(markdown),
                        'status': status,
                        'error': ""
                    })

                except Exception as e:
                    status = 'failed'
                    pbar_end_str = f"err={e}"
                    logger.warning(pbar_end_str)
                    
                    pages_data.append({
                        'page_num': page_num + 1,
                        'content': "[변환 실패]"
                    })
                    
                    progress_callback_data.update({
                        'file_name': file_name,
                        'current_page': page_num + 1,
                        'total_pages': total_pages,
                        'page_content_length': 0,
                        'status': status,
                        'error': str(e)
                    })
                finally:
                    pbar.set_postfix_str(pbar_end_str)
                    pbar.update(1)
                    # 콜백 호출
                    if self.progress_callback:
                        self.progress_callback(progress_callback_data)
        
        return pages_data

    def pdf_to_markdown(self, pdf_path: str) -> Tuple[List[Dict], Dict]:
        """
        PDF를 페이지별 Markdown으로 변환

        Args:
            pdf_path: PDF 파일 경로

        Returns:
            (pages_data, metadata)
            - pages_data: [{'page_num': int, 'content': str}, ...]
            - metadata: 파일 정보
        """
        logger.debug(f"PDF 변환 중: {Path(pdf_path).name}")

        # 페이지별 변환
        pages_data = self.markdown_with_progress(pdf_path)
        
        # Markdown 파일 저장
        self.save_markdown_from_pdf(pdf_path, pages_data)

        # 파일 해시 계산
        file_hash = self.hash_manager.calculate_file_hash(pdf_path)
        
        # 메타데이터 추출
        metadata = {
            'file_name': Path(pdf_path).name,
            'file_path': pdf_path,
            'file_hash': file_hash,
            'page_count': len(pages_data)
        }

        return pages_data, metadata

    def clean_markdown_text(self, text: str) -> str:
        """
        Markdown 텍스트 전처리
        
        Args:
            text: 원본 텍스트
            
        Returns:
            str: 전처리된 텍스트
        """
        import re
        
        # 연속 공백 → 단일 공백
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 연속 개행(3개 이상) → 2개
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 각 줄 앞뒤 공백 제거
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()

    def chunk_document(
        self, 
        pages_data: List[Dict],
        file_name: str,
        file_path: str,
        file_hash: str
    ) -> List[Document]:
        """
        페이지별 데이터를 청킹 (페이지 번호 보존)

        Args:
            pages_data: [{'page_num': int, 'content': str}, ...]
            file_name: 파일명
            file_path: 파일 경로
            file_hash: 파일 해시 값

        Returns:
            청크된 Document 리스트
        """
        logger.debug(f"문서 청킹 중: {file_name}")

        chunks = []
        chunk_index = 0

        for page_data in pages_data:
            page_num = page_data['page_num']
            page_content = page_data['content']
            
            # 빈 페이지 스킵
            if not page_content.strip() or page_content == "[빈 페이지]":
                continue

            # 페이지 청킹
            page_chunks = self.text_splitter.create_documents(
                texts=[page_content],
                metadatas=[{
                    'file_name': file_name,
                    'file_path': file_path,
                    'file_hash': file_hash,
                    'page': page_num,  # 정확한 PDF 페이지 번호
                    'chunk_type': 'original',
                    'chunk_index': chunk_index,
                }]
            )

            for chunk in page_chunks:
                chunk.metadata['chunk_index'] = chunk_index
                chunk_index += 1
                chunks.append(chunk)

        logger.debug(f"총 {len(chunks)}개 청크 생성")
        return chunks

    def process_pdf(self, pdf_path: str) -> List[Document]:
        """
        PDF 파일 전체 처리 파이프라인

        Args:
            pdf_path: PDF 파일 경로

        Returns:
            청크된 Document 리스트
        """
        pages_data, metadata = self.pdf_to_markdown(pdf_path)
        
        chunks = self.chunk_document(
            pages_data,
            metadata['file_name'],
            metadata['file_path'],
            metadata['file_hash'],
        )
        
        return chunks
    
    def info(self, chunks: List[Document]) -> pd.DataFrame:
        """
        청크 정보를 DataFrame으로 출력

        Args:
            chunks: 생성된 청크 리스트

        Returns:
            pandas.DataFrame: 청크 정보 요약
        """
        chunk_data = [
            {
                "파일명": chunk.metadata['file_name'],
                "페이지": chunk.metadata['page'],
                "청크 인덱스": chunk.metadata['chunk_index'],
                "청크 길이": len(chunk.page_content),
                "청크 유형": chunk.metadata.get('chunk_type', 'unknown'),
            }
            for chunk in chunks
        ]

        df = pd.DataFrame(chunk_data)
        print(f"총 {len(chunks)}개의 청크 정보가 생성되었습니다.")
        return df

# 3. 요약 파이프라인
class SummaryPipeline:
    """문서 요약 파이프라인 (원본의 20% 크기, 콜백 지원)"""

    def __init__(
        self,
        llm,
        summary_ratio: float = 0.2,
        summary_overlap_ratio: float = 0.1,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Args:
            llm: LangChain LLM 인스턴스
            summary_ratio: 요약 비율 (기본값: 0.2 = 20%)
            summary_overlap_ratio: 요약 오버랩 비율 (미사용)
            progress_callback: 요약 진행 상황 콜백 함수
                호출 시 전달되는 딕셔너리:
                {
                    'current_chunk': int,
                    'total_chunks': int,
                    'file_name': str,
                    'page': int,
                    'original_length': int,
                    'summary_length': int,
                    'compression_ratio': float,
                    'status': str,  # 'processing', 'completed', 'failed'
                    'error': str (optional)
                }
        """
        self.llm = llm
        self.summary_ratio = summary_ratio
        self.summary_overlap_ratio = summary_overlap_ratio
        self.progress_callback = progress_callback

    def summarize_chunk(self, chunk: Document, min_length: int = 100) -> str:
        """
        단일 청크 요약

        Args:
            chunk: Document 청크

        Returns:
            요약된 텍스트
        """
        original_length = len(chunk.page_content)
        
        # 짧은 청크는 원본 반환
        if original_length <= min_length:
            return chunk.page_content

        target_length = int(len(chunk.page_content) * self.summary_ratio)

        prompt = f"""다음 텍스트를 약 {target_length}자 정도로 핵심 내용만 요약해주세요.
중요한 키워드와 주요 개념을 유지하면서 간결하게 작성해주세요.

원문:
{chunk.page_content}

요약:"""

        summary = self.llm.invoke(prompt).content
        return summary

    def create_summary_chunks(self, original_chunks: List[Document]) -> List[Document]:
        """
        원본 청크들의 요약본 생성 (진행 상황 콜백 지원)

        Args:
            original_chunks: 원본 청크 리스트

        Returns:
            요약된 청크 리스트
        """
        total_chunks = len(original_chunks)
        logger.debug(f"{total_chunks}개 청크 요약 중...")

        summary_chunks = []

        with tqdm(total=total_chunks, desc="청크 요약", unit="chunk") as pbar:
            pbar_end_str = ""
            for idx, chunk in enumerate(original_chunks):
                current_chunk = idx + 1
                file_name = chunk.metadata.get('file_name', 'unknown')
                page = chunk.metadata.get('page', 0)
                original_length = len(chunk.page_content)
                
                # 진행 상황 딕셔너리 초기화
                progress_info = {
                    'current_chunk': current_chunk,
                    'total_chunks': total_chunks,
                    'file_name': file_name,
                    'page': page,
                    'original_length': original_length,
                    'summary_length': 0,
                    'compression_ratio': 0.0,
                    'status': 'processing',
                    'error': ''
                }

                try:
                    # 청크 요약
                    summary_text = self.summarize_chunk(chunk)
                    summary_length = len(summary_text)
                    compression_ratio = summary_length / original_length if original_length > 0 else 0.0
                    
                    # 상태 업데이트
                    progress_info.update({
                        'summary_length': summary_length,
                        'compression_ratio': compression_ratio,
                        'status': 'completed'
                    })

                    # 요약 청크 생성
                    summary_chunk = Document(
                        page_content=summary_text,
                        metadata={
                            'file_name': file_name,
                            'file_path': chunk.metadata.get('file_path', ''),
                            'file_hash': chunk.metadata.get('file_hash', ''),
                            'page': page,
                            'chunk_type': 'summary',
                            'chunk_index': len(summary_chunks),
                            'original_chunk_index': chunk.metadata['chunk_index']
                        }
                    )

                    summary_chunks.append(summary_chunk)
                    
                    # tqdm 포스트픽스 업데이트
                    if len(file_name) > 5:
                        pbar_end_str = f"{file_name[:5]}..."
                    else:
                        pbar_end_str = f"{file_name}"
                    pbar_end_str += f"p.{page} | "
                    pbar_end_str += f"{original_length}→{summary_length}자 "
                    pbar_end_str += f"({compression_ratio:.1%})"

                except Exception as e:
                    status = 'failed'
                    error_msg = str(e)
                    pbar_end_str = f"failed: {error_msg[:30]}"
                    
                    progress_info.update({
                        'status': status,
                        'error': error_msg
                    })
                    
                finally:
                    # 콜백 호출
                    pbar.set_postfix_str(pbar_end_str)
                    pbar.update(1)
                    
                    if self.progress_callback:
                        self.progress_callback(progress_info)

        logger.debug(f"요약 완료: {len(summary_chunks)}개 청크")
        return summary_chunks

# 4. 2단계 검색 파이프라인
class TwoStageSearchPipeline:
    """2단계 검색 파이프라인"""

    def __init__(
        self,
        summary_vectorstore: FAISS,
        original_vectorstore: FAISS,
        similarity_threshold: float = 0.75,  # 수정: 0.5 → 0.75 (더 엄격하게)
        top_k_summary: int = 5,
        top_k_final: int = 2,
        score_gap_threshold: float = 0.15
    ):
        self.summary_vectorstore = summary_vectorstore
        self.original_vectorstore = original_vectorstore
        self.similarity_threshold = similarity_threshold
        self.top_k_summary = top_k_summary
        self.top_k_final = top_k_final
        self.score_gap_threshold = score_gap_threshold

    def _distance_to_similarity(self, distance: float) -> float:
        """
        거리 → 유사도 변환
        
        FAISS L2 거리 특성:
        - 0에 가까울수록: 매우 유사
        - 1.0 이상: 관련성 낮음
        
        변환 전략:
        - distance < 0.5 → 0.8 이상 (우수)
        - distance = 1.0 → 0.5 (보통)
        - distance > 2.0 → 0.33 이하 (낮음)
        """
        return 1.0 / (1.0 + distance)

    def search(self, query: str) -> List[SearchResult]:
        """
        2단계 검색 실행

        Args:
            query: 검색 쿼리

        Returns:
            최종 검색 결과 리스트
        """
        logger.debug(f"검색 쿼리: '{query}'")

        # 1단계: 요약문 검색
        logger.debug("  [1단계] 요약문 검색 중...")
        summary_results = self.summary_vectorstore.similarity_search_with_score(
            query, k=self.top_k_summary
        )
        
        # 거리 → 유사도 변환
        summary_results_normalized = [
            (doc, self._distance_to_similarity(score))
            for doc, score in summary_results
        ]

        # 유사도 체크 (높을수록 좋음)
        if not summary_results_normalized or summary_results_normalized[0][1] < self.similarity_threshold:
            logger.warning(f"   요약문 유사도 낮음 (score: {summary_results_normalized[0][1] if summary_results_normalized else 0:.3f})")
            logger.debug("  [2단계] 원본 문서 직접 검색...")
            final_results = self._search_original_directly(query)
            return final_results

        # 2단계 진행
        for idx, (doc, score) in enumerate(summary_results_normalized, 1):
            logger.debug(f"    [{idx}] 파일명: {doc.metadata['file_name']}, "
                        f"페이지: {doc.metadata['page']}, "
                        f"청크 인덱스: {doc.metadata['chunk_index']}, "
                        f"유사도: {score:.3f}")
        
        logger.debug("  [2단계] 원본 문서 검색 중...")

        # 요약문에 해당하는 원본 청크 인덱스 추출
        original_indices = [
            doc.metadata.get('original_chunk_index')
            for doc, score in summary_results_normalized
        ]

        # 원본 문서에서 해당 청크들 검색
        final_results = self._search_original_by_indices(query, original_indices)

        # 최종 결과 필터링
        return self._filter_final_results(final_results)

    def _search_original_directly(self, query: str) -> List[SearchResult]:
        """원본 문서 직접 검색"""
        results = self.original_vectorstore.similarity_search_with_score(
            query, k=self.top_k_final
        )

        return [
            SearchResult(
                content=doc.page_content,
                file_name=doc.metadata['file_name'],
                page=doc.metadata['page'],
                score=self._distance_to_similarity(float(score)),
                chunk_type=doc.metadata['chunk_type']
            )
            for doc, score in results
        ]

    def _search_original_by_indices(
        self,
        query: str,
        target_indices: List[int]
    ) -> List[SearchResult]:
        """특정 인덱스의 원본 청크들 중에서 검색"""

        # 모든 원본 문서 가져오기
        all_docs = self.original_vectorstore.similarity_search_with_score(query, k=100)

        # 타겟 인덱스에 해당하는 문서만 필터링
        filtered_results = [
            (doc, self._distance_to_similarity(score))
            for doc, score in all_docs
            if doc.metadata['chunk_index'] in target_indices
        ]

        # 상위 k개 선택
        filtered_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)[:self.top_k_final * 2]

        # 검색 결과 로깅
        logger.debug(f"  원본 문서 검색 완료: {len(filtered_results)}개 발견")
        for idx, (doc, score) in enumerate(filtered_results, 1):
            logger.debug(f"    [{idx}] 파일명: {doc.metadata['file_name']}, "
                        f"페이지: {doc.metadata['page']}, "
                        f"청크 인덱스: {doc.metadata['chunk_index']}, "
                        f"유사도: {score:.3f}")

        return [
            SearchResult(
                content=doc.page_content,
                file_name=doc.metadata['file_name'],
                page=doc.metadata['page'],
                score=float(score),
                chunk_type=doc.metadata['chunk_type']
            )
            for doc, score in filtered_results
        ]

    def _filter_final_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        최종 결과 필터링
        - 유사도 좋은 상위 2개 선정
        - 1등과 2등 차이가 크면 1등만 선택
        """
        if not results:
            return []

        # 유사도 기준 정렬 (높은 순)
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)

        # 최상위 결과
        top_result = sorted_results[0]

        # 2등이 있고, 1등과의 차이가 작으면 2개 반환
        if len(sorted_results) > 1:
            second_result = sorted_results[1]
            score_gap = top_result.score - second_result.score

            logger.debug(f"  1등 유사도: {top_result.score:.3f}")
            logger.debug(f"  2등 유사도: {second_result.score:.3f}")
            logger.debug(f"  점수 차이: {score_gap:.3f}")

            if score_gap <= self.score_gap_threshold:
                logger.debug(f"  최종 선택: 상위 2개")
                return sorted_results[:2]
            else:
                logger.debug(f"  최종 선택: 1등만 (점수 차이 큼)")
                return [top_result]

        return [top_result]

# 5. DB 관리 파이프라인
class VectorStoreManager:
    """VectorStore 관리 클래스 - 저장/로드/수정/삭제"""

    def __init__(self, db_path: str = "./vector_db"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.embeddings = OpenAIEmbeddings()

    def save(
        self,
        summary_vectorstore: FAISS,
        original_vectorstore: FAISS,
        name: str = "default"
    ):
        """
        VectorStore 저장

        Args:
            summary_vectorstore: 요약문 벡터스토어
            original_vectorstore: 원본 벡터스토어
            name: DB 이름
        """
        logger.debug(f"VectorStore 저장 중: {name}")

        summary_path = self.db_path / f"{name}_summary"
        original_path = self.db_path / f"{name}_original"

        summary_vectorstore.save_local(str(summary_path))
        original_vectorstore.save_local(str(original_path))

        logger.debug(f"저장 완료: {self.db_path}")

    def load(self, name: str = "default") -> Tuple[FAISS, FAISS]:
        """
        VectorStore 로드

        Args:
            name: DB 이름

        Returns:
            (summary_vectorstore, original_vectorstore)
        """
        logger.debug(f"VectorStore 로드 중: {name}")

        summary_path = self.db_path / f"{name}_summary"
        original_path = self.db_path / f"{name}_original"

        if not summary_path.exists() or not original_path.exists():
            logger.warning(f"VectorStore '{name}' 미존재 — 빈 VectorStore 생성")
            # 더미 문서로 빈 FAISS 생성 (최소 1개 문서 필요)
            from langchain_core.documents import Document
            
            dummy_doc = Document(
                page_content="[초기화용 더미 문서]",
                metadata={
                    'file_name': '__init__',
                    'page': 0,
                    'chunk_type': 'dummy',
                    'chunk_index': 0
                }
            )
            summary_vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
            original_vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
            # 즉시 저장 (다음 로드 시 사용)
            summary_vectorstore.save_local(str(summary_path))
            original_vectorstore.save_local(str(original_path))
            
            logger.debug(f"빈 VectorStore 생성 및 저장 완료: {name}")
            return summary_vectorstore, original_vectorstore

        summary_vectorstore = FAISS.load_local(
            str(summary_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        original_vectorstore = FAISS.load_local(
            str(original_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        logger.debug(f"로드 완료")
        return summary_vectorstore, original_vectorstore

    def delete(self, name: str = "default"):
        """VectorStore 삭제"""
        logger.debug(f"VectorStore 삭제 중: {name}")

        summary_path = self.db_path / f"{name}_summary"
        original_path = self.db_path / f"{name}_original"

        if summary_path.exists():
            import shutil
            shutil.rmtree(summary_path)
        if original_path.exists():
            import shutil
            shutil.rmtree(original_path)

        logger.debug(f"삭제 완료")

    def list_stores(self) -> List[str]:
        """저장된 VectorStore 목록"""
        stores = set()
        for path in self.db_path.glob("*_summary"):
            name = path.name.replace("_summary", "")
            stores.add(name)

        return sorted(list(stores))


# 6. 통합 VectorStore 클래스
class VectorStore:
    """
    VectorStore - 통합 클래스
    
    Features:
    - PDF → Markdown 변환 (진행 상황 콜백)
    - 600자 청킹 (오버랩 100)
    - 20% 요약 생성 (진행 상황 콜백)
    - 2단계 검색 (요약문 → 원본)
    - DB 저장/로드/관리
    """

    def __init__(
        self,
        llm,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        db_path: str = "./vector_db",
        embedding_batch_size: int = 100,
        pdf_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        summary_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Args:
            llm: LangChain LLM 인스턴스
            chunk_size: 청크 최대 크기
            chunk_overlap: 청크 오버랩 크기
            db_path: 벡터 DB 저장 경로
            embedding_batch_size: 임베딩 배치 크기
            pdf_progress_callback: PDF 변환 진행 콜백
            summary_progress_callback: 요약 진행 콜백
        """
        self.llm = llm
        self.embeddings = OpenAIEmbeddings(chunk_size=embedding_batch_size)

        # 파이프라인 초기화 (콜백 등록)
        self.doc_pipeline = DocumentProcessingPipeline(
            chunk_size, 
            chunk_overlap,
            progress_callback=pdf_progress_callback
        )
        self.summary_pipeline = SummaryPipeline(
            llm,
            progress_callback=summary_progress_callback
        )
        self.db_manager = VectorStoreManager(db_path)

        # VectorStore
        self.summary_vectorstore: Optional[FAISS] = None
        self.original_vectorstore: Optional[FAISS] = None
        self.search_pipeline: Optional[TwoStageSearchPipeline] = None

    def add_documents(self, pdf_paths: List[str], batch_size: int = 100) -> None:
        """
        PDF 문서 추가 (해시 기반 중복 제거 + 더미 문서 자동 삭제)
        
        Args:
            pdf_paths: PDF 파일 경로 리스트
            batch_size: 임베딩 배치 크기
        """
        logger.debug("문서 추가 시작")
        
        # 더미 문서 검증 및 제거
        self._remove_dummy_if_exists()
        
        all_original_chunks = []
        all_summary_chunks = []
        
        for pdf_path in pdf_paths:
            # 중복 체크
            if self.doc_pipeline.is_file_already_processed(pdf_path):
                logger.debug(f"이미 처리된 파일 스킵: {Path(pdf_path).name}")
                continue
            
            logger.debug(f"처리 중: {pdf_path}")
            
            original_chunks = self.doc_pipeline.process_pdf(pdf_path)
            all_original_chunks.extend(original_chunks)
            
            summary_chunks = self.summary_pipeline.create_summary_chunks(original_chunks)
            all_summary_chunks.extend(summary_chunks)

        if not all_original_chunks:
            logger.warning("추가할 새 문서가 없습니다. (모두 중복)")
            return

        # 3. VectorStore 생성
        logger.debug("VectorStore 생성 중...")

        if self.original_vectorstore is None:
            # 초기 생성은 첫 배치만
            first_batch = all_original_chunks[:batch_size]
            self.original_vectorstore = FAISS.from_documents(first_batch, self.embeddings)
            
            first_summary_batch = all_summary_chunks[:batch_size]
            self.summary_vectorstore = FAISS.from_documents(first_summary_batch, self.embeddings)
            
            # 나머지 배치 추가
            for i in range(batch_size, len(all_original_chunks), batch_size):
                batch = all_original_chunks[i:i+batch_size]
                self.original_vectorstore.add_documents(batch)
                
            for i in range(batch_size, len(all_summary_chunks), batch_size):
                batch = all_summary_chunks[i:i+batch_size]
                self.summary_vectorstore.add_documents(batch)
        else:
            # 기존 VectorStore에 배치로 추가
            for i in range(0, len(all_original_chunks), batch_size):
                batch = all_original_chunks[i:i+batch_size]
                self.original_vectorstore.add_documents(batch)
                logger.debug(f"원본 배치 {i//batch_size + 1} 추가 완료")
                
            for i in range(0, len(all_summary_chunks), batch_size):
                batch = all_summary_chunks[i:i+batch_size]
                self.summary_vectorstore.add_documents(batch)
                logger.debug(f"요약 배치 {i//batch_size + 1} 추가 완료")

        # 4. 검색 파이프라인 초기화
        self.search_pipeline = TwoStageSearchPipeline(
            self.summary_vectorstore,
            self.original_vectorstore
        )

        logger.debug("문서 추가 완료!")
        logger.debug(f"  - 원본 청크: {len(all_original_chunks)}개")
        logger.debug(f"  - 요약 청크: {len(all_summary_chunks)}개")


    def _remove_dummy_if_exists(self) -> None:
        """
        더미 문서 검증 및 제거 (DB 크기 1일 때만)
        
        작동 조건:
        - original_vectorstore 크기가 정확히 1
        - 해당 문서의 file_name이 '__init__'
        - chunk_type이 'dummy'
        
        제거 방법:
        - 기존 DB 삭제 후 빈 VectorStore로 재초기화
        """
        if self.original_vectorstore is None:
            return
        
        # docstore 크기 확인
        doc_count = len(self.original_vectorstore.docstore._dict)
        
        if doc_count != 1:
            logger.debug(f"더미 검사 스킵: DB 크기 {doc_count}개")
            return
        
        # 유일한 문서 추출
        first_doc = list(self.original_vectorstore.docstore._dict.values())[0]
        
        # 더미 문서 검증
        if (first_doc.metadata.get('file_name') == '__init__' and 
            first_doc.metadata.get('chunk_type') == 'dummy'):
            
            logger.info("더미 문서 감지 — 제거 후 재초기화")
            
            # VectorStore 초기화
            self.original_vectorstore = None
            self.summary_vectorstore = None
            self.search_pipeline = None
            
            logger.debug("더미 제거 완료: VectorStore 초기화됨")
        else:
            logger.debug("더미 문서 아님 — 제거 생략")

    def search(self, query: str) -> List[Dict]:
        """
        검색 실행

        Args:
            query: 검색 쿼리

        Returns:
            검색 결과 (파일명, 페이지, 내용 포함)
        """
        if self.search_pipeline is None:
            raise ValueError("VectorStore가 비어있습니다. 먼저 문서를 추가하세요.")

        results = self.search_pipeline.search(query)

        # 결과 포맷팅
        formatted_results = []
        for idx, result in enumerate(results, 1):
            formatted_results.append({
                'rank': idx,
                'file_name': result.file_name,
                'page': result.page,
                'content': result.content,
                'score': result.score,
                'chunk_type': result.chunk_type
            })

        return formatted_results

    def save(self, name: str = "default"):
        """VectorStore 저장"""
        if self.summary_vectorstore is None or self.original_vectorstore is None:
            raise ValueError("저장할 VectorStore가 없습니다.")

        self.db_manager.save(
            self.summary_vectorstore,
            self.original_vectorstore,
            name
        )

    def load(self, name: str = "default"):
        """VectorStore 로드 및 기존 파일 해시 동기화"""
        self.summary_vectorstore, self.original_vectorstore = self.db_manager.load(name)

        # 검색 파이프라인 초기화
        self.search_pipeline = TwoStageSearchPipeline(
            self.summary_vectorstore,
            self.original_vectorstore
        )
        
        self._sync_processed_files_from_db()    
        logger.debug(f"로드 완료: {len(self.doc_pipeline.processed_files)}개 파일 해시 동기화됨")
        
    def _sync_processed_files_from_db(self) -> None:
        """
        DB에 저장된 파일 해시 정보를 doc_pipeline.processed_files에 동기화
        """
        if self.original_vectorstore is None:
            return
        
        all_docs = list(self.original_vectorstore.docstore._dict.values())
        
        for doc in all_docs:
            file_hash = doc.metadata.get('file_hash')
            file_path = doc.metadata.get('file_name')  # 파일명만 저장됨
            
            if file_hash and file_path:
                # 해시를 키로, 파일명을 값으로 저장 (경로 복원 불가하므로 파일명만)
                self.doc_pipeline.processed_files[file_hash] = file_path
        
        logger.debug(f"DB 동기화: {len(self.doc_pipeline.processed_files)}개 파일 해시")        

    def delete(self, name: str = "default"):
        """VectorStore 삭제"""
        self.db_manager.delete(name)

    def list_stores(self) -> List[str]:
        """저장된 VectorStore 목록"""
        return self.db_manager.list_stores()

    def get_rag_context(self, query: str) -> str:
        """
        RAG용 컨텍스트 생성

        Args:
            query: 검색 쿼리

        Returns:
            포맷팅된 컨텍스트 문자열
        """
        results = self.search(query)

        context_parts = []
        for result in results:
            context_parts.append(
                f"[출처: {result['file_name']}, 페이지 {result['page']}, 유사도 {result['score']:.3f}]\n"
                f"{result['content']}\n"
            )

        return "\n\n".join(context_parts)
    
    def generate_answer(self, query: str, context: Optional[str] = None) -> str:
        """
        RAG 기반 답변 생성
        
        Args:
            query (str): 사용자 질의
            context (Optional[str]): 검색된 컨텍스트 (None이면 자동 검색)
        
        Returns:
            str: 답변 문자열 (질의/답변 라벨 제외)
        """
        # context 자동 생성 (옵션)
        if context is None:
            logger.debug("context가 제공되지 않아 자동 검색 실행")
            context = self.get_rag_context(query)
        
        # context 유효성 검사
        if not context or context.strip() == "":
            return "죄송합니다. 제공된 자료에서 관련 정보를 찾을 수 없습니다."
        
        # RAG 프롬프트 템플릿
        prompt = f"""다음 컨텍스트를 참고하여 질문에 답변해주세요.
    답변은 컨텍스트 내용에 기반해야 하며, 출처 정보(파일명, 페이지, 유사도)를 포함해주세요.
    답변만 작성하고 '질의:', '답변:' 등의 라벨은 붙이지 마세요.

    컨텍스트:
    {context}

    질문: {query}

    답변:"""
        
        # LLM 호출
        try:
            response = self.llm.invoke(prompt)
            answer = response.content.strip()
        except Exception as e:
            logger.error(f"LLM 호출 실패: {e}")
            answer = "답변 생성 중 오류가 발생했습니다."
        
        # 답변만 반환
        return answer
    
    def get_sample(
        self,
        file_name: str,
        chunk_index: int
    ) -> Optional[Dict[str, Any]]:
        """
        특정 파일의 특정 청크 정보 조회
        
        Args:
            file_name (str): 파일명 (예: "sample_korean.pdf")
            chunk_index (int): 청크 인덱스 (0부터 시작)
        
        Returns:
            Dict[str, Any]: 청크 정보 딕셔너리
            {
                'file_name': str,
                'chunk_index': int,
                'page': int,
                'original_text': str,
                'original_length': int,
                'summary_text': str,
                'summary_length': int,
                'compression_ratio': float,
                'file_hash': str,
                'metadata': Dict
            }
            
            조회 실패 시 None 반환
        
        Examples:
            >>> info = vector_store.get_sample("sample.pdf", 5)
            >>> print(f"원본: {info['original_length']}자")
            >>> print(f"요약: {info['summary_length']}자")
        """
        if self.original_vectorstore is None or self.summary_vectorstore is None:
            logger.warning("VectorStore가 초기화되지 않았습니다.")
            return None
        
        # 원본 청크 검색
        original_doc = None
        for doc_id, doc in self.original_vectorstore.docstore._dict.items():
            if (doc.metadata.get('file_name') == file_name and 
                doc.metadata.get('chunk_index') == chunk_index):
                original_doc = doc
                break
        
        if original_doc is None:
            logger.warning(
                f"청크를 찾을 수 없습니다: {file_name}, chunk_index={chunk_index}"
            )
            return None
        
        # 요약 청크 검색
        summary_doc = None
        for doc_id, doc in self.summary_vectorstore.docstore._dict.items():
            if (doc.metadata.get('file_name') == file_name and 
                doc.metadata.get('original_chunk_index') == chunk_index):
                summary_doc = doc
                break
        
        # 결과 구성
        original_text = original_doc.page_content
        original_length = len(original_text)
        
        summary_text = summary_doc.page_content if summary_doc else "[요약 없음]"
        summary_length = len(summary_text) if summary_doc else 0
        
        compression_ratio = (
            summary_length / original_length 
            if original_length > 0 else 0.0
        )
        
        result = {
            'file_name': file_name,
            'chunk_index': chunk_index,
            'page': original_doc.metadata.get('page', 0),
            'original_text': original_text,
            'original_length': original_length,
            'summary_text': summary_text,
            'summary_length': summary_length,
            'compression_ratio': compression_ratio,
            'file_hash': original_doc.metadata.get('file_hash', 'N/A'),
            'metadata': original_doc.metadata
        }
        
        logger.debug(
            f"청크 조회 완료: {file_name} [청크 {chunk_index}] "
            f"원본 {original_length}자 → 요약 {summary_length}자 "
            f"({compression_ratio:.1%})"
        )
        
        return result
    
    def get_metadata_info(self) -> pd.DataFrame:
        """
        VectorStore에 저장된 문서 메타데이터 조회
        
        Returns:
            pd.DataFrame: 파일별 메타데이터 정보
        """
        if self.original_vectorstore is None:
            logger.warning("VectorStore가 비어있습니다.")
            return pd.DataFrame()
        
        # FAISS docstore에서 모든 문서 추출
        all_docs = list(self.original_vectorstore.docstore._dict.values())
        
        # 파일별 통계 집계
        file_stats = {}
        for doc in all_docs:
            file_name = doc.metadata.get('file_name', 'unknown')
            
            if file_name not in file_stats:
                file_stats[file_name] = {
                    'file_name': file_name,
                    'file_hash': doc.metadata.get('file_hash', 'N/A'),
                    'total_pages': set(),
                    'chunk_count': 0,
                    'chunk_type': doc.metadata.get('chunk_type', 'unknown')
                }
            
            file_stats[file_name]['total_pages'].add(doc.metadata.get('page', 0))
            file_stats[file_name]['chunk_count'] += 1
        
        # DataFrame 변환
        result = []
        for stats in file_stats.values():
            result.append({
                '파일명': stats['file_name'],
                '페이지수': len(stats['total_pages']),
                '청크개수': stats['chunk_count'],
                '청크유형': stats['chunk_type'],
                '파일해시': stats['file_hash'][:16] + '...'  # 앞 16자만 표시
            })
        
        df = pd.DataFrame(result)
        logger.debug(f"총 {len(df)}개 파일 메타데이터 조회 완료")
        return df
    
    def print_sample(self, file_name: str, chunk_index: int) -> None:
        """
        청크 정보를 보기 좋게 출력
        
        Args:
            file_name (str): 파일명
            chunk_index (int): 청크 인덱스
        
        Examples:
            >>> vector_store.print_sample("sample.pdf", 3)
        """
        info = self.get_sample(file_name, chunk_index)
        
        if info is None:
            logger.info("청크를 찾을 수 없습니다.")
            return
        
        logger.info("." * 80)
        logger.info(f"파일명: {info['file_name']}")
        logger.info(f"청크 인덱스: {info['chunk_index']} | 페이지: {info['page']}")
        logger.info(f"압축률: {info['compression_ratio']:.1%} "
            f"({info['original_length']}자 → {info['summary_length']}자)")
        logger.info("." * 80)
        
        logger.info("[원본 텍스트]")
        logger.info(info['original_text'][:500])
        if info['original_length'] > 500:
            logger.info(f"... (총 {info['original_length']}자)")

        logger.info("." * 80)
        logger.info("[요약 텍스트]")
        logger.info(info['summary_text'][:300])
        if info['summary_length'] > 300:
            logger.info(f"... (총 {info['summary_length']}자)")

        logger.info("." * 80)

# 7. 사용
if __name__ == "__main__":
    def simple_progress_callback(info: Dict[str, Any]) -> None:
        """간단한 진행 상황 출력"""
        progress_message = (
            f"[{info['file_name']}] "
            f"{info['current_page']}/{info['total_pages']} "
            f"({info['status']}) "
            f"{info['page_content_length']}자"
        )
        print(f"{progress_message}")
            
    def summary_progress_callback(info: Dict[str, Any]) -> None:
        """요약 진행 상황 출력"""
        progress_message = ""
        if info['status'] == 'completed':
            progress_message =(
                f"VectorStore "
                f"[{info['file_name']}] "
                f"청크 {info['current_chunk']}/{info['total_chunks']} | "
                f"페이지 {info['page']} | "
                f"압축률 {info['compression_ratio']:.1%} "
                f"({info['original_length']}→{info['summary_length']}자)"
            )
        elif info['status'] == 'failed':
            progress_message =(
                f"VectorStore "
                f"[{info['file_name']}] "
                f"청크 {info['current_chunk']} 요약 실패: {info['error']}"
            )
        print(f"{progress_message}")

    def test_main():
        from langchain_openai import ChatOpenAI

        data_path = str(Path(drive_root()) / 'data')
        logger.info(f'데이터 경로 설정 완료: {data_path}')

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        db_path = str(Path(data_path) / r'mission14_vectorstore_db')
        vector_store = VectorStore(llm=llm, chunk_size=600, chunk_overlap=100, db_path=db_path)
        vector_store.load("my_knowledge_base")
        pdf_files = []
        pdf_files.append(str(Path(data_path) / r'2024년+원천징수의무자를+위한+연말정산+신고안내.pdf'))
        vector_store.add_documents(pdf_files)
        vector_store.save("my_knowledge_base")

        metadata_info = vector_store.get_metadata_info()
        metadata_info.head()

        # query = "RAG의 핵심 원리는 무엇인가요?"
        querys = [
            "원천징수는 무엇인가요?",        # ✓ 문서에 존재
            "HPGP는 무엇인가?",              # ✗ 문서에 없음 → 필터링되어야 함
            "월세공제는 무엇인가요?",        # ✓ 문서에 존재
            "블록체인 기술이란?",            # ✗ 문서에 없음 → 필터링되어야 함
        ]
        
        for query in querys:
            results = vector_store.search(query)

            # 결과 출력
            print("."*60)
            print("검색 결과")
            print("."*60)
            for result in results:
                print(f"[{result['rank']}위] {result['file_name']} (p.{result['page']}) - 유사도: {result['score']:.3f}")
                print(f"{result['content'][:50]}... len={len(result['content'])}")
                print("."*30)

            # RAG 컨텍스트 생성
            context = vector_store.get_rag_context(query)
            print("."*60)
            print("RAG 컨텍스트")
            print("."*60)
            print(f"{context[:50]}... len={len(context)}")

            # RAG 답변 생성
            result = vector_store.generate_answer(query, context=context)
            print("-"*60)
            print(result)
            print("="*60)

    logger.setLevel(logging.INFO)
    test_main()
