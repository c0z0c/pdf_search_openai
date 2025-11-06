"""
VectorStore - 통합 RAG 시스템 인터페이스
"""

from pathlib import Path
from typing import List, Dict, Optional, Callable, Any

import pandas as pd
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from .config.config import Config
from .models.data_models import SearchResult
from .core.document_processor import DocumentProcessingPipeline
from .core.summary_pipeline import SummaryPipeline
from .core.search_pipeline import TwoStageSearchPipeline
from .core.vectorstore_manager import VectorStoreManager
from .utils.logging_config import get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    VectorStore - 통합 RAG 시스템 인터페이스
    
    Features:
    - PDF → Markdown 변환 (진행 상황 콜백)
    - 600자 청킹 (오버랩 100)
    - 20% 요약 생성 (진행 상황 콜백)
    - 2단계 검색 (요약문 → 원본)
    - DB 저장/로드/관리
    
    Attributes:
        llm: LangChain LLM 인스턴스
        embeddings: OpenAIEmbeddings 인스턴스
        doc_pipeline: DocumentProcessingPipeline 인스턴스
        summary_pipeline: SummaryPipeline 인스턴스
        db_manager: VectorStoreManager 인스턴스
        summary_vectorstore: 요약문 벡터스토어
        original_vectorstore: 원본 벡터스토어
        search_pipeline: TwoStageSearchPipeline 인스턴스
    """

    def __init__(
        self,
        llm,
        chunk_size: int = Config.CHUNK_SIZE,
        chunk_overlap: int = Config.CHUNK_OVERLAP,
        db_path: str = Config.DEFAULT_DB_PATH,
        embedding_batch_size: int = Config.EMBEDDING_BATCH_SIZE,
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

        # VectorStore 생성
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

        # 검색 파이프라인 초기화
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

    def save(self, name: str = "default") -> None:
        """
        VectorStore 저장
        
        Args:
            name: DB 이름
        """
        if self.summary_vectorstore is None or self.original_vectorstore is None:
            raise ValueError("저장할 VectorStore가 없습니다.")

        self.db_manager.save(
            self.summary_vectorstore,
            self.original_vectorstore,
            name
        )

    def load(self, name: str = "default") -> None:
        """
        VectorStore 로드 및 기존 파일 해시 동기화
        
        Args:
            name: DB 이름
        """
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
            file_path = doc.metadata.get('file_name')
            
            if file_hash and file_path:
                self.doc_pipeline.processed_files[file_hash] = file_path
        
        logger.debug(f"DB 동기화: {len(self.doc_pipeline.processed_files)}개 파일 해시")

    def delete(self, name: str = "default") -> None:
        """
        VectorStore 삭제
        
        Args:
            name: DB 이름
        """
        self.db_manager.delete(name)

    def list_stores(self) -> List[str]:
        """
        저장된 VectorStore 목록
        
        Returns:
            List[str]: VectorStore 이름 리스트
        """
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
            query: 사용자 질의
            context: 검색된 컨텍스트 (None이면 자동 검색)
        
        Returns:
            str: 답변 문자열 (질의/답변 라벨 제외)
        """
        # context 자동 생성
        if context is None:
            logger.debug("context가 제공되지 않아 자동 검색 실행")
            context = self.get_rag_context(query)
        
        # context 유효성 검사
        if not context or context.strip() == "":
            return "죄송합니다. 제공된 자료에서 관련 정보를 찾을 수 없습니다."
        
        # RAG 프롬프트 템플릿
        prompt = f"""다음 컨텍스트를 참고하여 질문에 답변해주세요.
답변은 컨텍스트 내용에 기반해야 하며, 출처 정보(파일명, 페이지, 유사도)를 답변 다음 줄에 포함해주세요.
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
        
        return answer
    
    def get_sample(
        self,
        file_name: str,
        chunk_index: int
    ) -> Optional[Dict[str, Any]]:
        """
        특정 파일의 특정 청크 정보 조회
        
        Args:
            file_name: 파일명
            chunk_index: 청크 인덱스 (0부터 시작)
        
        Returns:
            Dict[str, Any]: 청크 정보 딕셔너리 또는 None
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
                '파일해시': stats['file_hash'][:16] + '...'
            })
        
        df = pd.DataFrame(result)
        logger.debug(f"총 {len(df)}개 파일 메타데이터 조회 완료")
        return df
    
    def delete_by_file_name(self, file_name: str) -> bool:
        """
        특정 파일의 모든 청크를 VectorStore에서 삭제
        
        Args:
            file_name: 삭제할 파일명
            
        Returns:
            bool: 삭제 성공 여부
        """
        if self.original_vectorstore is None or self.summary_vectorstore is None:
            logger.warning("VectorStore가 비어있습니다.")
            return False
        
        try:
            # Original vectorstore에서 삭제
            orig_docstore = self.original_vectorstore.docstore._dict
            orig_index_to_id = self.original_vectorstore.index_to_docstore_id
            
            # 삭제할 인덱스 수집
            indices_to_delete = []
            for idx, doc_id in orig_index_to_id.items():
                doc = orig_docstore.get(doc_id)
                if doc and doc.metadata.get('file_name') == file_name:
                    indices_to_delete.append(idx)
            
            if not indices_to_delete:
                logger.warning(f"'{file_name}' 파일을 찾을 수 없습니다.")
                return False
            
            # FAISS에서 인덱스 삭제
            self.original_vectorstore.index.remove_ids(
                np.array(indices_to_delete, dtype=np.int64)
            )
            
            # docstore에서 문서 삭제
            for idx in indices_to_delete:
                doc_id = orig_index_to_id[idx]
                if doc_id in orig_docstore:
                    del orig_docstore[doc_id]
                del orig_index_to_id[idx]
            
            # 인덱스 재정렬
            new_index_to_id = {new_idx: doc_id 
                               for new_idx, (_, doc_id) in enumerate(orig_index_to_id.items())}
            self.original_vectorstore.index_to_docstore_id = new_index_to_id
            
            logger.debug(f"Original vectorstore: {len(indices_to_delete)}개 청크 삭제")
            
            # Summary vectorstore에서 삭제
            summ_docstore = self.summary_vectorstore.docstore._dict
            summ_index_to_id = self.summary_vectorstore.index_to_docstore_id
            
            summ_indices_to_delete = []
            for idx, doc_id in summ_index_to_id.items():
                doc = summ_docstore.get(doc_id)
                if doc and doc.metadata.get('file_name') == file_name:
                    summ_indices_to_delete.append(idx)
            
            if summ_indices_to_delete:
                self.summary_vectorstore.index.remove_ids(
                    np.array(summ_indices_to_delete, dtype=np.int64)
                )
                
                for idx in summ_indices_to_delete:
                    doc_id = summ_index_to_id[idx]
                    if doc_id in summ_docstore:
                        del summ_docstore[doc_id]
                    del summ_index_to_id[idx]
                
                new_summ_index_to_id = {new_idx: doc_id 
                                        for new_idx, (_, doc_id) in enumerate(summ_index_to_id.items())}
                self.summary_vectorstore.index_to_docstore_id = new_summ_index_to_id
                
                logger.debug(f"Summary vectorstore: {len(summ_indices_to_delete)}개 청크 삭제")
            
            # processed_files에서 제거
            hashes_to_remove = [
                hash_key for hash_key, fname in self.doc_pipeline.processed_files.items()
                if fname == file_name
            ]
            for hash_key in hashes_to_remove:
                del self.doc_pipeline.processed_files[hash_key]
                logger.debug(f"processed_files에서 해시 제거: {hash_key[:16]}...")
            
            logger.info(f"✓ '{file_name}' 삭제 완료 (원본: {len(indices_to_delete)}, 요약: {len(summ_indices_to_delete)})")
            return True
            
        except Exception as e:
            logger.error(f"파일 삭제 실패: {str(e)}")
            return False
    
    def print_sample(self, file_name: str, chunk_index: int) -> None:
        """
        청크 정보를 보기 좋게 출력
        
        Args:
            file_name: 파일명
            chunk_index: 청크 인덱스
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
