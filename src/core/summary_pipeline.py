"""
요약 파이프라인 - 문서 요약 생성
"""

from typing import List, Optional, Callable, Dict, Any

from tqdm import tqdm
from langchain_core.documents import Document

from ..config.config import Config
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class SummaryPipeline:
    """
    문서 요약 파이프라인 (원본의 20% 크기, 콜백 지원)
    
    Attributes:
        llm: LangChain LLM 인스턴스
        summary_ratio: 요약 비율
        summary_overlap_ratio: 요약 오버랩 비율
        progress_callback: 요약 진행 상황 콜백 함수
    """

    def __init__(
        self,
        llm,
        summary_ratio: float = Config.SUMMARY_RATIO,
        summary_overlap_ratio: float = Config.SUMMARY_OVERLAP_RATIO,
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

    def summarize_chunk(
        self, 
        chunk: Document, 
        min_length: int = Config.SUMMARY_MIN_LENGTH
    ) -> str:
        """
        단일 청크 요약

        Args:
            chunk: Document 청크
            min_length: 최소 길이 (이하면 원본 반환)

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
