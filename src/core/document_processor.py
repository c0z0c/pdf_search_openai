"""
문서 처리 파이프라인 - PDF → Markdown → 청킹
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable, Any

import pymupdf4llm
import fitz
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ..config.config import Config
from ..utils.logging_config import get_logger
from .file_hash_manager import FileHashManager

logger = get_logger(__name__)


class DocumentProcessingPipeline:
    """
    PDF → Markdown → 청킹 파이프라인 (콜백 지원)
    
    Attributes:
        chunk_size: 청크 최대 크기
        chunk_overlap: 청크 간 중복 크기
        progress_callback: PDF 변환 진행 상황 콜백 함수
        text_splitter: RecursiveCharacterTextSplitter 인스턴스
        hash_manager: FileHashManager 인스턴스
        processed_files: 처리된 파일의 해시 딕셔너리
    """

    def __init__(
        self,
        chunk_size: int = Config.CHUNK_SIZE,
        chunk_overlap: int = Config.CHUNK_OVERLAP,
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
                    'status': str,  # 'processing', 'empty', 'failed'
                    'error': str
                }
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.progress_callback = progress_callback
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=Config.CHUNK_SEPARATORS,
            length_function=len,
        )
        self.hash_manager = FileHashManager(Config.HASH_ALGORITHM)
        self.processed_files: Dict[str, str] = {}
        
    def is_file_already_processed(self, file_path: str) -> bool:
        """
        파일이 이미 처리되었는지 확인합니다.

        Args:
            file_path: 파일 경로

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
            pdf_path: PDF 파일 경로
            pages_data: 페이지별 {'page_num', 'content'} 딕셔너리 리스트
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
            pdf_path: PDF 파일 경로
        
        Returns:
            List[Dict]: [{'page_num': int, 'content': str}, ...]
        """        
        file_name = Path(pdf_path).name
        
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)

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
        pages_data, metadata = self.pdf_to_markdown(pdf_path)
        
        chunks = self.chunk_document(
            pages_data,
            metadata['file_name'],
            metadata['file_path'],
            metadata['file_hash'],
        )
        
        return chunks
