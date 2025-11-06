"""
설정 모듈 - 환경변수 및 기본값 중앙화
"""

import os
from pathlib import Path
from typing import Optional

# OpenMP 중복 로드 경고 무시 (FAISS 사용 시 필요)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Config:
    """애플리케이션 설정 클래스"""
    
    # OpenAI API
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_TEMPERATURE: float = 0.0
    
    # 청킹 설정
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100
    CHUNK_SEPARATORS: list = ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
    
    # 요약 설정
    SUMMARY_RATIO: float = 0.2
    SUMMARY_OVERLAP_RATIO: float = 0.1
    SUMMARY_MIN_LENGTH: int = 100
    
    # 검색 설정
    SIMILARITY_THRESHOLD: float = 0.75
    TOP_K_SUMMARY: int = 5
    TOP_K_FINAL: int = 2
    SCORE_GAP_THRESHOLD: float = 0.15
    
    # 임베딩 설정
    EMBEDDING_BATCH_SIZE: int = 100
    
    # 경로 설정
    DEFAULT_DB_PATH: str = "./vector_db"
    DEFAULT_DATA_PATH: str = "./data"
    
    # 해시 설정
    HASH_ALGORITHM: str = "sha256"
    
    @classmethod
    def init_openai_key(cls, is_colab: bool = False) -> bool:
        """
        OpenAI API 키 초기화
        
        Args:
            is_colab: Colab 환경 여부
            
        Returns:
            bool: 초기화 성공 여부
        """
        if is_colab:
            try:
                from google.colab import userdata
                cls.OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
            except ImportError:
                return False
        else:
            from dotenv import load_dotenv
            load_dotenv()
            cls.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        if cls.OPENAI_API_KEY:
            cls.OPENAI_API_KEY = cls.OPENAI_API_KEY.strip()
            os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
            return True
        
        return False
    
    @classmethod
    def get_db_path(cls, custom_path: Optional[str] = None) -> Path:
        """
        VectorStore DB 경로 반환
        
        Args:
            custom_path: 사용자 지정 경로
            
        Returns:
            Path: DB 경로
        """
        path = custom_path or cls.DEFAULT_DB_PATH
        return Path(path)
    
    @classmethod
    def get_data_path(cls, custom_path: Optional[str] = None) -> Path:
        """
        데이터 경로 반환
        
        Args:
            custom_path: 사용자 지정 경로
            
        Returns:
            Path: 데이터 경로
        """
        path = custom_path or cls.DEFAULT_DATA_PATH
        return Path(path)
