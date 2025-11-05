"""
Streamlit RAG 검색 애플리케이션
Author: 14_3팀_김명환
Description: PDF 기반 RAG 시스템 웹 인터페이스
"""

import os
import sys
from pathlib import Path
from typing import Optional, List
import streamlit as st
from langchain_openai import ChatOpenAI

# src 모듈 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from pdf_search import VectorStore

# 페이지 설정
st.set_page_config(
    page_title="PDF RAG 검색",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
    .source-info {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def load_api_key() -> Optional[str]:
    """환경에 맞춰 OpenAI API 키를 로드합니다.
    
    Returns:
        str: OpenAI API 키 또는 None
    """
    # Streamlit Cloud (secrets.toml)
    try:
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            return st.secrets['OPENAI_API_KEY']
    except Exception:
        pass
    
    # 로컬 환경 (.env)
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    return api_key


def init_vector_store(api_key: str) -> VectorStore:
    """VectorStore 초기화
    
    Args:
        api_key: OpenAI API 키
        
    Returns:
        VectorStore: 초기화된 벡터스토어
    """
    os.environ['OPENAI_API_KEY'] = api_key
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 자동 경로 설정 (data/vectorstore)
    vector_store = VectorStore(
        llm=llm,
        chunk_size=600,
        chunk_overlap=100,
        db_path=None  # 자동 경로
    )
    
    return vector_store


def display_search_results(results: List[dict]) -> None:
    """검색 결과를 표시합니다.
    
    Args:
        results: 검색 결과 리스트
    """
    if not results:
        st.warning("검색 결과가 없습니다.")
        return
    
    for idx, result in enumerate(results, 1):
        with st.container():
            st.markdown(f"""
            <div class="result-box">
                <div class="source-info">
                    <b>출처 {idx}:</b> {result['file_name']} | 
                    <b>페이지:</b> {result['page']} | 
                    <b>유사도:</b> {result['score']:.3f}
                </div>
                <div>{result['content']}</div>
            </div>
            """, unsafe_allow_html=True)


def main():
    """메인 애플리케이션"""
    
    # 헤더
    st.markdown('<p class="main-header">🔍 PDF RAG 검색 시스템</p>', unsafe_allow_html=True)
    
    # API 키 로드
    api_key = load_api_key()
    
    if not api_key:
        st.error("⚠️ OpenAI API 키가 설정되지 않았습니다.")
        st.info("""
        **로컬 환경**: `.env` 파일에 `OPENAI_API_KEY=your_key` 추가  
        **Streamlit Cloud**: Secrets에 `OPENAI_API_KEY` 추가
        """)
        st.stop()
    
    # 세션 상태 초기화
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    if 'db_loaded' not in st.session_state:
        st.session_state.db_loaded = False
    
    # 사이드바 - 벡터스토어 관리
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 벡터스토어 로드
        st.subheader("1. 벡터스토어 로드")
        db_name = st.text_input("DB 이름", value="my_knowledge_base")
        
        if st.button("🔄 벡터스토어 로드", use_container_width=True):
            with st.spinner("벡터스토어 로딩 중..."):
                try:
                    vector_store = init_vector_store(api_key)
                    vector_store.load(db_name)
                    st.session_state.vector_store = vector_store
                    st.session_state.db_loaded = True
                    st.success(f"✅ '{db_name}' 로드 완료")
                    
                    # 메타데이터 표시
                    metadata_df = vector_store.get_metadata_info()
                    st.dataframe(metadata_df, use_container_width=True)
                    
                except FileNotFoundError:
                    st.error(f"❌ '{db_name}' 벡터스토어를 찾을 수 없습니다.")
                except Exception as e:
                    st.error(f"❌ 로드 실패: {str(e)}")
        
        st.divider()
        
        # PDF 업로드
        st.subheader("2. PDF 업로드 (선택)")
        uploaded_files = st.file_uploader(
            "PDF 파일 선택",
            type=['pdf'],
            accept_multiple_files=True,
            help="새로운 PDF를 추가하려면 파일을 선택하세요"
        )
        
        if uploaded_files and st.button("📤 PDF 추가 및 저장", use_container_width=True):
            with st.spinner("PDF 처리 중..."):
                try:
                    # 임시 파일로 저장
                    temp_dir = project_root / 'data' / 'temp'
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    
                    pdf_paths = []
                    for uploaded_file in uploaded_files:
                        temp_path = temp_dir / uploaded_file.name
                        with open(temp_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        pdf_paths.append(str(temp_path))
                    
                    # VectorStore 초기화 (없으면)
                    if st.session_state.vector_store is None:
                        st.session_state.vector_store = init_vector_store(api_key)
                    
                    # 문서 추가
                    st.session_state.vector_store.add_documents(pdf_paths)
                    
                    # 저장
                    st.session_state.vector_store.save(db_name)
                    st.session_state.db_loaded = True
                    
                    st.success(f"✅ {len(uploaded_files)}개 파일 추가 완료")
                    
                    # 임시 파일 삭제
                    for temp_path in pdf_paths:
                        Path(temp_path).unlink(missing_ok=True)
                    
                except Exception as e:
                    st.error(f"❌ PDF 처리 실패: {str(e)}")
        
        st.divider()
        
        # 정보
        st.subheader("ℹ️ 시스템 정보")
        st.caption(f"**모델**: gpt-4o-mini")
        st.caption(f"**청크 크기**: 600자")
        st.caption(f"**오버랩**: 100자")
    
    # 메인 영역 - 검색
    if not st.session_state.db_loaded:
        st.info("👈 사이드바에서 벡터스토어를 로드하거나 PDF를 업로드하세요.")
        st.stop()
    
    # 검색 탭
    tab1, tab2 = st.tabs(["🔍 검색", "💬 RAG 답변"])
    
    with tab1:
        st.subheader("검색 쿼리")
        query = st.text_input(
            "질문을 입력하세요",
            placeholder="예: RAG의 핵심 원리는 무엇인가요?",
            key="search_query"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            search_button = st.button("🔍 검색", use_container_width=True)
        
        if search_button and query:
            with st.spinner("검색 중..."):
                try:
                    results = st.session_state.vector_store.search(query)
                    
                    st.success(f"✅ {len(results)}개 결과 발견")
                    
                    # 결과 표시
                    display_search_results(results)
                    
                except Exception as e:
                    st.error(f"❌ 검색 실패: {str(e)}")
    
    with tab2:
        st.subheader("RAG 기반 답변 생성")
        query_rag = st.text_area(
            "질문을 입력하세요",
            placeholder="예: RAG 시스템의 장점과 단점을 설명해주세요.",
            height=100,
            key="rag_query"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            generate_button = st.button("💬 답변 생성", use_container_width=True)
        
        if generate_button and query_rag:
            with st.spinner("답변 생성 중..."):
                try:
                    # 컨텍스트 검색
                    context = st.session_state.vector_store.get_rag_context(query_rag)
                    
                    # 답변 생성
                    answer = st.session_state.vector_store.generate_answer(query_rag, context=context)
                    
                    # 답변 표시
                    st.markdown("### 📝 답변")
                    st.markdown(answer)
                    
                    # 컨텍스트 표시 (확장 가능)
                    with st.expander("📚 참조된 문서 컨텍스트"):
                        st.text(context)
                    
                except Exception as e:
                    st.error(f"❌ 답변 생성 실패: {str(e)}")
    
    # 푸터
    st.divider()
    st.caption("💡 2단계 검색 파이프라인 (요약문 → 원본) | 청크 크기: 600자 | 오버랩: 100자")


if __name__ == "__main__":
    main()
