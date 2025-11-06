"""
2단계 검색 파이프라인 - 요약문 → 원본 검색
"""

from typing import List

from langchain_community.vectorstores import FAISS

from ..config.config import Config
from ..models.data_models import SearchResult
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class TwoStageSearchPipeline:
    """
    2단계 검색 파이프라인
    
    Attributes:
        summary_vectorstore: 요약문 벡터스토어
        original_vectorstore: 원본 벡터스토어
        similarity_threshold: 유사도 임계값
        top_k_summary: 요약문 검색 상위 K개
        top_k_final: 최종 검색 상위 K개
        score_gap_threshold: 점수 차이 임계값
    """

    def __init__(
        self,
        summary_vectorstore: FAISS,
        original_vectorstore: FAISS,
        similarity_threshold: float = Config.SIMILARITY_THRESHOLD,
        top_k_summary: int = Config.TOP_K_SUMMARY,
        top_k_final: int = Config.TOP_K_FINAL,
        score_gap_threshold: float = Config.SCORE_GAP_THRESHOLD
    ):
        """
        Args:
            summary_vectorstore: 요약문 벡터스토어
            original_vectorstore: 원본 벡터스토어
            similarity_threshold: 유사도 임계값
            top_k_summary: 요약문 검색 상위 K개
            top_k_final: 최종 검색 상위 K개
            score_gap_threshold: 점수 차이 임계값
        """
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
        
        Args:
            distance: FAISS L2 거리
            
        Returns:
            float: 유사도 점수 (0~1)
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
            if doc.metadata.get('original_chunk_index') is not None
        ]

        # 원본 문서에서 해당 청크들 검색
        final_results = self._search_original_by_indices(query, original_indices)

        # 최종 결과 필터링
        return self._filter_final_results(final_results)

    def _search_original_directly(self, query: str) -> List[SearchResult]:
        """
        원본 문서 직접 검색
        
        Args:
            query: 검색 쿼리
            
        Returns:
            List[SearchResult]: 검색 결과 리스트
        """
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
        """
        특정 인덱스의 원본 청크들 중에서 검색
        
        Args:
            query: 검색 쿼리
            target_indices: 원본 청크 인덱스 리스트
            
        Returns:
            List[SearchResult]: 검색 결과 리스트
        """
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
        
        Args:
            results: 검색 결과 리스트
            
        Returns:
            List[SearchResult]: 필터링된 검색 결과
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
