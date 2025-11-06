"""
VectorStore - 파이프라인 기반 RAG 시스템 (레거시 호환)
Author: 14_3팀_김명환
Description: 2단계 검색을 활용한 고급 RAG 시스템

Warning: 이 모듈은 하위 호환성을 위해 유지됩니다.
새로운 코드에서는 다음과 같이 import하세요:
    from src.vectorstore import VectorStore
    from src.config import Config
    from src.models.data_models import ChunkMetadata, SearchResult
"""

import os
import sys
from pathlib import Path

# OpenMP 중복 로드 경고 무시
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 스크립트 실행 시 현재 디렉토리를 sys.path에 추가
SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# helper 모듈 import
from .utils import helper_utils as hu
from .utils.helper_utils import *
from .utils import helper_c0z0c_dev as helper
from .utils.helper_c0z0c_dev import *

from .config.config import Config
from .models.data_models import ChunkMetadata, SearchResult
from .core import (
    FileHashManager,
    DocumentProcessingPipeline,
    SummaryPipeline,
    TwoStageSearchPipeline,
    VectorStoreManager,
)
from .vectorstore import VectorStore
from .utils.logging_config import setup_logger, get_logger

# 로거 설정
logger = get_logger(__name__)

# OpenAI API 키 설정
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

# Config를 통한 초기화도 시도
if not openai_api_key:
    Config.init_openai_key(is_colab=IS_COLAB)

# 하위 호환성을 위한 re-export
__all__ = [
    # 설정
    'Config',
    
    # 데이터 모델
    'ChunkMetadata',
    'SearchResult',
    
    # 핵심 클래스
    'FileHashManager',
    'DocumentProcessingPipeline',
    'SummaryPipeline',
    'TwoStageSearchPipeline',
    'VectorStoreManager',
    'VectorStore',
    
    # 유틸리티
    'setup_logger',
    'get_logger',
    'logger',
    
    # helper 함수
    'drive_root',
    'timestamp',
]

# 테스트 함수
def test_main():
    """테스트 함수"""
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
    print(metadata_info.head())

    querys = [
        "원천징수는 무엇인가요?",
        "HPGP는 무엇인가?",
        "월세공제는 무엇인가요?",
        "블록체인 기술이란?",
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


if __name__ == "__main__":
    import logging
    logger.setLevel(logging.INFO)
    test_main()
