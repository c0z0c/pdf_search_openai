"""
VectorStore MVP - 파이프라인 기반 RAG 시스템
Author: 14_3팀_김명환
Description: 2단계 검색을 활용한 고급 RAG 시스템
"""

import os
import pickle
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import importlib
from urllib.request import urlretrieve

import helper_utils as hu
from helper_utils import *

import helper_c0z0c_dev as helper
from helper_c0z0c_dev import *

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import pymupdf4llm
import pandas as pd
from tqdm import tqdm
import pytz

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
    os.environ["OPENAI_API_KEY"] = openai_api_key # LangChain
    logger.debug(f"OPENAI_API_KEY [{openai_api_key[:4]}****{openai_api_key[-4:]}] 환경변수 설정 완료")
    logger.info("OPENAI_API_KEY 설정")
else:
    logger.warning("openai_api_key가 설정되지 않아 OpenAI 로그인 생략됨")

# 프로젝트 루트 기준 경로 설정
if IS_COLAB:
    data_path = str(Path(drive_root()) / 'data')
else:
    # 로컬 환경: src 폴더 기준 상위 폴더의 data
    project_root = Path(__file__).parent.parent
    data_path = str(project_root / 'data')
    
logger.info(f'데이터 경로 설정 완료: {data_path}')

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

import hashlib

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
    """PDF → Markdown → 청킹 파이프라인"""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            length_function=len,
        )
        self.hash_manager = FileHashManager()
        self.processed_files = {}  # 파일 해시 저장소
        
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
        
    def save_markdown_from_pdf(self, pdf_path: str, markdown_text: str) -> None:
        """
        PDF 파일명을 기반으로 마크다운 문서를 저장합니다.

        Args:
            pdf_path (str): PDF 파일 경로
            markdown_text (str): 변환된 마크다운 텍스트
        """
        # PDF 파일명을 기반으로 .md 확장자로 변경
        md_path = Path(pdf_path).with_suffix('.md')
        
        # 마크다운 파일 저장
        with open(md_path, 'w', encoding='utf-8') as md_file:
            md_file.write(markdown_text)
        
        logger.debug(f"Markdown 파일 저장 완료: {md_path}")        

    def markdown_with_progress(self, pdf_path: str) -> str:
        """
        PDF를 Markdown으로 변환하면서 진행 상황을 표시합니다.
        10페이지 이상은 병렬 처리, 미만은 순차 처리

        Args:
            pdf_path (str): PDF 파일 경로

        Returns:
            str: 변환된 Markdown 텍스트
        """
        import fitz
        #from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # PDF 페이지 수 확인
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)

        markdown_pages = []
        with tqdm(total=total_pages, desc="PDF to Markdown", unit="page") as pbar:
            for page_num in range(total_pages):
                try:
                    markdown = pymupdf4llm.to_markdown(
                        doc=pdf_path,
                        pages=[page_num]
                    )
                    markdown_pages.append((page_num, markdown))
                except Exception as e:
                    logger.warning(f"페이지 {page_num} 순차 변환 실패: {e}")
                    markdown_pages.append((page_num, ""))
                finally:
                    pbar.update(1)
        
        markdown_pages.sort(key=lambda x: x[0])
        return "\n\n".join([md for _, md in markdown_pages if md])

    def pdf_to_markdown(self, pdf_path: str) -> Tuple[str, Dict]:
        """
        PDF를 Markdown으로 변환

        Args:
            pdf_path: PDF 파일 경로

        Returns:
            (markdown_text, metadata)
        """
        logger.debug(f"PDF 변환 중: {Path(pdf_path).name}")

        # pymupdf4llm을 사용하여 PDF를 Markdown으로 변환
        #markdown_text = pymupdf4llm.to_markdown(pdf_path)
        markdown_text = self.markdown_with_progress(pdf_path)
        
        # 전처리 추가
        markdown_text = self.clean_markdown_text(markdown_text)

        self.save_markdown_from_pdf(pdf_path, markdown_text)

        # 파일 해시 계산
        file_hash = self.hash_manager.calculate_file_hash(pdf_path)
        # 메타데이터 추출
        metadata = {
            'file_name': Path(pdf_path).name,
            'file_path': pdf_path,
            'file_hash': file_hash,
            'page_count': markdown_text.count('---')  # 페이지 구분자 기준
        }

        return markdown_text, metadata

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
        markdown_text: str, 
        file_name: str,
        file_path: str,
        file_hash: str  # 추가
    ) -> List[Document]:
        """
        Markdown 문서를 청킹

        Args:
            markdown_text: Markdown 텍스트
            file_name: 파일명
            file_hash: 파일 해시 값

        Returns:
            청크된 Document 리스트
        """
        logger.debug(f"문서 청킹 중: {file_name}")

        pages = markdown_text.split('-----')
        chunks = []
        chunk_index = 0

        for page_num, page_content in enumerate(pages, 1):
            if not page_content.strip():
                continue

            page_chunks = self.text_splitter.create_documents(
                texts=[page_content],
                metadatas=[{
                    'file_name': file_name,
                    'file_path': file_path,
                    'file_hash': file_hash,
                    'page': page_num,
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
        markdown_text, metadata = self.pdf_to_markdown(pdf_path)
        
        # file_hash 전달 추가
        chunks = self.chunk_document(
            markdown_text, 
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
                "파일 경로": chunk.metadata.get('file_path', 'unknown'),
                "파일 해시": chunk.metadata.get('file_hash', 'unknown'),
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
    """문서 요약 파이프라인 (원본의 20% 크기)"""

    def __init__(self, llm, summary_ratio: float = 0.2, summary_overlap_ratio: float = 0.1):
        self.llm = llm
        self.summary_ratio = summary_ratio
        self.summary_overlap_ratio = summary_overlap_ratio

    def summarize_chunk(self, chunk: Document) -> str:
        """
        단일 청크 요약

        Args:
            chunk: Document 청크

        Returns:
            요약된 텍스트
        """
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
        원본 청크들의 요약본 생성

        Args:
            original_chunks: 원본 청크 리스트

        Returns:
            요약된 청크 리스트
        """
        logger.debug(f"{len(original_chunks)}개 청크 요약 중...")

        summary_chunks = []

        for idx, chunk in enumerate(original_chunks):
            logger.debug(f"  요약 진행: {idx + 1}/{len(original_chunks)}")

            # 청크 요약
            summary_text = self.summarize_chunk(chunk)
            
            logger.debug(f"원본 : {chunk}")
            logger.debug(f"요약 : {summary_text}")

            # 요약 청크 생성
            summary_chunk = Document(
                page_content=summary_text,
                metadata={
                    'file_name': chunk.metadata['file_name'],
                    'file_path': chunk.metadata['file_path'],
                    'file_hash': chunk.metadata['file_hash'],
                    'page': chunk.metadata['page'],
                    'chunk_type': 'summary',
                    'chunk_index': len(summary_chunks),
                    'original_chunk_index': chunk.metadata['chunk_index']
                }
            )

            summary_chunks.append(summary_chunk)

        logger.debug(f"요약 완료: {len(summary_chunks)}개 청크")
        return summary_chunks

# 4. 2단계 검색 파이프라인

class TwoStageSearchPipeline:
    """2단계 검색 파이프라인"""

    def __init__(
        self,
        summary_vectorstore: FAISS,
        original_vectorstore: FAISS,
        similarity_threshold: float = 0.5,  # 유사도 기준 (0.5 = 중간)
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
        """거리 → 유사도 변환 (0~1, 높을수록 유사)"""
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
            raise FileNotFoundError(f"VectorStore '{name}' not found")

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
    MVP VectorStore - 통합 클래스

    Features:
    - PDF → Markdown 변환
    - 600자 청킹 (오버랩 100)
    - 20% 요약 생성
    - 2단계 검색 (요약문 → 원본)
    - DB 저장/로드/관리
    """

    def __init__(
        self,
        llm,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        db_path: Optional[str] = None
    ):
        self.llm = llm
        self.embeddings = OpenAIEmbeddings()

        # db_path가 None이면 프로젝트 구조에 맞는 기본 경로 사용
        if db_path is None:
            if IS_COLAB:
                db_path = str(Path(drive_root()) / 'data' / 'vectorstore')
            else:
                # 로컬: src 기준 상위의 data/vectorstore
                project_root = Path(__file__).parent.parent
                db_path = str(project_root / 'data' / 'vectorstore')

        # 파이프라인 초기화
        self.doc_pipeline = DocumentProcessingPipeline(chunk_size, chunk_overlap)
        self.summary_pipeline = SummaryPipeline(llm)
        self.db_manager = VectorStoreManager(db_path)

        # VectorStore
        self.summary_vectorstore: Optional[FAISS] = None
        self.original_vectorstore: Optional[FAISS] = None
        self.search_pipeline: Optional[TwoStageSearchPipeline] = None

    def add_documents(self, pdf_paths: List[str]) -> None:
        """
        PDF 문서 추가 (해시 기반 중복 제거)
        
        Args:
            pdf_paths: PDF 파일 경로 리스트
        """
        logger.debug("문서 추가 시작")
        
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
            self.original_vectorstore = FAISS.from_documents(
                all_original_chunks, self.embeddings
            )
            self.summary_vectorstore = FAISS.from_documents(
                all_summary_chunks, self.embeddings
            )
        else:
            # 기존 VectorStore에 추가
            self.original_vectorstore.add_documents(all_original_chunks)
            self.summary_vectorstore.add_documents(all_summary_chunks)

        # 4. 검색 파이프라인 초기화
        self.search_pipeline = TwoStageSearchPipeline(
            self.summary_vectorstore,
            self.original_vectorstore
        )

        logger.debug("문서 추가 완료!")
        logger.debug(f"  - 원본 청크: {len(all_original_chunks)}개")
        logger.debug(f"  - 요약 청크: {len(all_summary_chunks)}개")

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
            str: 포맷팅된 질의/답변 문자열
        """
        # context 자동 생성 (옵션)
        if context is None:
            logger.debug("context가 제공되지 않아 자동 검색 실행")
            context = self.get_rag_context(query)
        
        # context 유효성 검사
        if not context or context.strip() == "":
            answer = "죄송합니다. 제공된 자료에서 관련 정보를 찾을 수 없습니다."
            return f"질의: {query}\n답변:\n{answer}"
        
        # RAG 프롬프트 템플릿
        prompt = f"""다음 컨텍스트를 참고하여 질문에 답변해주세요.
답변은 컨텍스트 내용에 기반해야 하며, 출처 정보(파일명, 페이지, 유사도)를 포함해주세요.

컨텍스트:
{context}

질문: {query}

답변: """
        
        # LLM 호출
        try:
            response = self.llm.invoke(prompt)
            answer = response.content.strip()
        except Exception as e:
            logger.error(f"LLM 호출 실패: {e}")
            answer = "답변 생성 중 오류가 발생했습니다."
        
        # 포맷팅된 결과 반환
        return f"질의: {query}\n답변: {answer}"
    
    
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

# 7. 사용
if __name__ == "__main__":
    from langchain_openai import ChatOpenAI

    # LLM 초기화
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # VectorStore 생성 (db_path=None으로 자동 경로 설정)
    vector_store = VectorStore(
        llm=llm,
        chunk_size=600,
        chunk_overlap=100,
        db_path=None  # 자동으로 data/vectorstore 경로 사용
    )
    
    logger.info(f"VectorStore DB 경로: {vector_store.db_manager.db_path}")
    
    # 기존 벡터스토어 로드 시도
    try:
        vector_store.load("my_knowledge_base")
        metadata_info = vector_store.get_metadata_info()
        print("\n=== 기존 벡터스토어 메타데이터 ===")
        print(metadata_info)
    except FileNotFoundError:
        logger.info("기존 벡터스토어가 없습니다. 새로 생성합니다.")

    # 문서 추가 (예시)
    # data_path는 전역변수로 이미 설정됨
    from pathlib import Path
    pdf_path = Path(data_path) / 'sample_korean.pdf'
    
    if pdf_path.exists():
        pdf_files = [str(pdf_path)]
        vector_store.add_documents(pdf_files)
        
        # 저장
        vector_store.save("my_knowledge_base")
        logger.info("벡터스토어 저장 완료")
    else:
        logger.warning(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        logger.info("예제 PDF를 data/ 폴더에 추가하거나 기존 벡터스토어를 사용하세요.")

    # 검색 예시
    if vector_store.search_pipeline:
        query = "RAG의 핵심 원리는 무엇인가요?"
        
        # RAG 컨텍스트 생성
        context = vector_store.get_rag_context(query)
        
        # RAG 답변 생성
        result = vector_store.generate_answer(query, context=context)
        print("\n" + "="*60)
        print(result)
        print("="*60 + "\n")
    else:
        logger.warning("검색 파이프라인이 초기화되지 않았습니다. 먼저 문서를 추가하세요.")
