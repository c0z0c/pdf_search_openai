"""
데이터 모델 정의
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ChunkMetadata:
    """
    청크 메타데이터
    
    Attributes:
        file_name: 파일명만
        page: 페이지 번호
        chunk_type: 'original' or 'summary'
        chunk_index: 청크 인덱스
        original_chunk_index: 요약본의 경우 원본 청크 인덱스
    """
    file_name: str
    page: int
    chunk_type: str
    chunk_index: int
    original_chunk_index: Optional[int] = None


@dataclass
class SearchResult:
    """
    검색 결과
    
    Attributes:
        content: 청크 내용
        file_name: 파일명만
        page: 페이지 번호
        score: 유사도 점수
        chunk_type: 청크 유형 ('original' or 'summary')
    """
    content: str
    file_name: str
    page: int
    score: float
    chunk_type: str
