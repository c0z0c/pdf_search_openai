"""
VectorStore 관리자 - 저장/로드/삭제
"""

from pathlib import Path
from typing import List, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ..config.config import Config
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    """
    VectorStore 관리 클래스 - 저장/로드/수정/삭제
    
    Attributes:
        db_path: VectorStore 저장 경로
        embeddings: OpenAIEmbeddings 인스턴스
    """

    def __init__(self, db_path: str = Config.DEFAULT_DB_PATH):
        """
        Args:
            db_path: VectorStore 저장 경로
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.embeddings = OpenAIEmbeddings()

    def save(
        self,
        summary_vectorstore: FAISS,
        original_vectorstore: FAISS,
        name: str = "default"
    ) -> None:
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

    def delete(self, name: str = "default") -> None:
        """
        VectorStore 삭제
        
        Args:
            name: DB 이름
        """
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
        """
        저장된 VectorStore 목록
        
        Returns:
            List[str]: VectorStore 이름 리스트
        """
        stores = set()
        for path in self.db_path.glob("*_summary"):
            name = path.name.replace("_summary", "")
            stores.add(name)

        return sorted(list(stores))
