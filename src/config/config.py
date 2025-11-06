"""
설정 모듈 - 환경변수 및 기본값 중앙화
"""

import os
from pathlib import Path
from typing import Optional

# OpenMP 중복 로드 경고 무시 (FAISS 사용 시 필요)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Config:
    """
    애플리케이션 설정 클래스
    
    이 클래스는 애플리케이션에서 사용되는 다양한 설정 값을 중앙에서 관리합니다.
    환경 변수, 기본값, 경로, API 키 초기화, 데이터베이스 경로 등을 포함합니다.
    """

    # OpenAI API 설정
    OPENAI_API_KEY: Optional[str] = None  # OpenAI API 키 (환경 변수에서 로드)
    OPENAI_MODEL: str = "gpt-4o-mini"  # 기본 OpenAI 모델 이름
    OPENAI_TEMPERATURE: float = 0.0  # 생성 온도 (0.0은 결정론적 응답)

    # 청킹(Chunking) 설정
    CHUNK_SIZE: int = 600  # 청크 크기 (문서 분할 시 사용)
    CHUNK_OVERLAP: int = 100  # 청크 간 중첩 크기
    CHUNK_SEPARATORS: list = ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]  # 청크 분리자

    # 요약 설정
    SUMMARY_RATIO: float = 0.2  # 요약 비율 (원문 대비)
    SUMMARY_OVERLAP_RATIO: float = 0.1  # 요약 간 중첩 비율
    SUMMARY_MIN_LENGTH: int = 100  # 요약 최소 길이 (문자 수)

    # 검색 설정
    SIMILARITY_THRESHOLD: float = 0.75  # 유사도 임계값 (검색 결과 필터링)
    TOP_K_SUMMARY: int = 5  # 요약 단계에서 상위 K개 선택
    TOP_K_FINAL: int = 2  # 최종 단계에서 상위 K개 선택
    SCORE_GAP_THRESHOLD: float = 0.15  # 점수 차이 임계값 (결과 필터링)

    # 임베딩 설정
    EMBEDDING_BATCH_SIZE: int = 100  # 임베딩 배치 크기 (병렬 처리 최적화)

    # 경로 설정
    DEFAULT_DB_PATH: str = "./vector_db"  # 기본 벡터 데이터베이스 경로
    DEFAULT_DATA_PATH: str = "./data"  # 기본 데이터 경로

    # 해시 설정
    HASH_ALGORITHM: str = "sha256"  # 기본 해시 알고리즘 (파일 무결성 확인 등)

    @classmethod
    def init_openai_key(cls, is_colab: bool = False) -> bool:
        """
        OpenAI API 키 초기화
        
        Args:
            is_colab (bool): Colab 환경 여부 (True일 경우 Colab에서 키를 로드)
            
        Returns:
            bool: 초기화 성공 여부 (True: 성공, False: 실패)
        """
        if is_colab:
            try:
                from google.colab import userdata
                cls.OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')  # Colab에서 키 로드
            except ImportError:
                return False
        else:
            from dotenv import load_dotenv
            load_dotenv()  # .env 파일에서 환경 변수 로드
            cls.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 키 로드
        
        if cls.OPENAI_API_KEY:
            cls.OPENAI_API_KEY = cls.OPENAI_API_KEY.strip()  # 키 공백 제거
            os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY  # 환경 변수에 설정
            return True
        
        return False

    @classmethod
    def set_openai_api_key(cls, is_colab: bool = False) -> None:
        """
        OpenAI API 키를 설정하고 로깅 처리
        
        Args:
            is_colab (bool): Colab 환경 여부 (True일 경우 Colab에서 키를 로드)
        """
        from ..utils.logging_config import get_logger
        logger = get_logger(__name__)
        
        openai_api_key = None
        if is_colab:
            try:
                from google.colab import userdata
                openai_api_key = userdata.get('OPENAI_API_KEY')
            except ImportError:
                pass
        else:
            from dotenv import load_dotenv
            load_dotenv()
            openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if openai_api_key:
            openai_api_key = openai_api_key.strip()
            os.environ["OPENAI_API_KEY"] = openai_api_key
            cls.OPENAI_API_KEY = openai_api_key
            logger.debug(f"OPENAI_API_KEY [{openai_api_key[:4]}****{openai_api_key[-4:]}] 환경변수 설정 완료")
            logger.info("OPENAI_API_KEY 설정")
        else:
            logger.warning("openai_api_key가 설정되지 않아 OpenAI 로그인 생략됨")

    @classmethod
    def get_db_path(cls, custom_path: Optional[str] = None) -> Path:
        """
        VectorStore DB 경로 반환
        
        Args:
            custom_path (Optional[str]): 사용자 지정 경로 (기본값: None)
            
        Returns:
            Path: DB 경로 (사용자 지정 경로가 없으면 기본 경로 반환)
        """
        path = custom_path or cls.DEFAULT_DB_PATH
        return Path(path)

    @classmethod
    def get_data_path(cls, custom_path: Optional[str] = None) -> Path:
        """
        데이터 경로 반환
        
        Args:
            custom_path (Optional[str]): 사용자 지정 경로 (기본값: None)
            
        Returns:
            Path: 데이터 경로 (사용자 지정 경로가 없으면 기본 경로 반환)
        """
        path = custom_path or cls.DEFAULT_DATA_PATH
        return Path(path)