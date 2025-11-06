"""
Streamlit RAG 검색 애플리케이션
Author: 14_3팀_김명환
Description: PDF 기반 RAG 시스템 웹 인터페이스
"""

import os
import warnings
import logging

# Streamlit secrets 경로 설정 (경고 방지)
os.environ['STREAMLIT_SECRETS_PATH'] = ''

# Python 경고 필터링
warnings.filterwarnings('ignore', message='.*st.cache is deprecated.*')
warnings.filterwarnings('ignore', message='.*torch.classes.*')

# Streamlit 로깅 레벨 조정 (secrets 메시지 숨김)
logging.getLogger('streamlit').setLevel(logging.ERROR)

import streamlit as st

st.set_page_config(
    page_title="PDF RAG 검색",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

import sys
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any
from langchain_openai import ChatOpenAI
import extra_streamlit_components as stx

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pdf_search import VectorStore

# CookieManager 초기화
cookie_manager = stx.CookieManager()


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
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


def load_api_key_from_env() -> Optional[str]:
    """환경에서 OpenAI API 키를 로드합니다."""
    # try:
    #     if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
    #         return st.secrets['OPENAI_API_KEY']
    # except Exception:
    #     pass

    from dotenv import load_dotenv
    load_dotenv()
    return os.getenv('OPENAI_API_KEY')


def save_api_key_to_cookie(api_key: str) -> None:
    """API 키를 쿠키에 저장합니다."""
    cookie_manager.set('openai_api_key', api_key, expires_at=None)


def load_api_key_from_cookie() -> Optional[str]:
    """쿠키에서 API 키를 로드합니다."""
    return cookie_manager.get('openai_api_key')


def get_api_key() -> Optional[str]:
    """우선순위: 쿠키 > 환경변수"""
    cookie_key = load_api_key_from_cookie()
    if cookie_key:
        return cookie_key
    return load_api_key_from_env()


def display_api_key_input() -> Optional[str]:
    """API 키 입력 UI를 표시하고 저장합니다."""
    st.warning("OpenAI API 키를 입력하세요")
    
    st.markdown("""
    ### API 키 발급 방법
    1. [OpenAI Platform](https://platform.openai.com/api-keys) 접속
    2. 로그인 후 'Create new secret key' 클릭
    3. 생성된 키 복사 (sk-로 시작)
    4. 아래에 붙여넣기
    
    키는 암호화되어 브라우저 쿠키에 저장됩니다.
    """)
    
    with st.form("api_key_form"):
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-proj-...",
            help="API 키는 sk-로 시작합니다"
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            submit = st.form_submit_button("저장 및 시작", use_container_width=True)
        
        if submit and api_key_input:
            if not api_key_input.startswith('sk-'):
                st.error("유효하지 않은 API 키 형식입니다 (sk-로 시작해야 함)")
                return None
            
            if len(api_key_input) < 20:
                st.error("API 키가 너무 짧습니다")
                return None
            
            save_api_key_to_cookie(api_key_input)
            st.success("API 키 저장 완료")
            st.info("페이지를 새로고침합니다")
            st.rerun()
    
    return None


def create_pdf_progress_callback() -> tuple:
    """PDF 변환용 progress callback"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def callback(info: Dict[str, Any]) -> None:
        current = info['current_page']
        total = info['total_pages']
        progress = current / total if total > 0 else 0
        
        progress_bar.progress(progress)
        
        status_msg = f"PDF 변환: {info['file_name']} ({current}/{total}) - {info['status']}"
        if info['page_content_length'] > 0:
            status_msg += f" | {info['page_content_length']}자"
        if info.get('error'):
            status_msg += f" | 오류: {info['error'][:30]}"
        
        status_text.text(status_msg)
    
    return callback, progress_bar, status_text


def create_summary_progress_callback() -> tuple:
    """요약 생성용 progress callback"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def callback(info: Dict[str, Any]) -> None:
        current = info['current_chunk']
        total = info['total_chunks']
        progress = current / total if total > 0 else 0
        
        progress_bar.progress(progress)
        
        if info['status'] == 'completed':
            status_msg = (
                f"요약 생성: {info['file_name']} ({current}/{total}) | "
                f"페이지 {info['page']} | "
                f"압축률 {info['compression_ratio']:.1%} "
                f"({info['original_length']}→{info['summary_length']}자)"
            )
        elif info['status'] == 'failed':
            status_msg = f"요약 실패: {info.get('error', 'Unknown error')[:50]}"
        else:
            status_msg = f"요약 생성 중... ({current}/{total})"
        
        status_text.text(status_msg)
    
    return callback, progress_bar, status_text


def init_vector_store(
    api_key: str,
    db_path: Optional[str] = None,
    pdf_callback: Optional[Callable] = None,
    summary_callback: Optional[Callable] = None
) -> VectorStore:
    """VectorStore 초기화 (callback 동적 등록)"""
    os.environ['OPENAI_API_KEY'] = api_key
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    if db_path is None:
        db_path = str(project_root / 'data' / 'vectorstore_db')
    
    print('.' * 80)
    print(f"db_path={db_path}")
    
    vector_store = VectorStore(
        llm=llm,
        chunk_size=600,
        chunk_overlap=100,
        db_path=db_path,
        pdf_progress_callback=pdf_callback,
        summary_progress_callback=summary_callback
    )
    
    return vector_store


def display_search_results(results: List[dict]) -> None:
    """검색 결과 표시"""
    if not results:
        st.warning("검색 결과가 없습니다")
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
    
    st.markdown('<p class="main-header">PDF RAG 검색 시스템</p>', unsafe_allow_html=True)
    
    api_key = get_api_key()
    
    with st.sidebar:
        st.header("설정")
        
        if api_key:
            st.success("API 키 로드됨")
            if st.button("API 키 변경", use_container_width=True):
                cookie_manager.delete('openai_api_key')
                st.rerun()
        else:
            display_api_key_input()
            st.stop()
        
        st.divider()
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    if 'db_loaded' not in st.session_state:
        st.session_state.db_loaded = False
    
    with st.sidebar:
        st.subheader("1. 벡터스토어 로드")
        db_name = st.text_input("DB 이름", value="my_knowledge_base")
        
        if st.button("벡터스토어 로드", use_container_width=True):
            with st.spinner("벡터스토어 로딩 중..."):
                try:
                    vector_store = init_vector_store(api_key)
                    vector_store.load(db_name)
                    st.session_state.vector_store = vector_store
                    st.session_state.db_loaded = True
                    st.success(f"'{db_name}' 로드 완료")
                    
                    metadata_df = vector_store.get_metadata_info()
                    if not metadata_df.empty:
                        st.dataframe(metadata_df, use_container_width=True)
                    else:
                        st.info("빈 벡터스토어입니다")
                    
                except FileNotFoundError:
                    st.error(f"'{db_name}' 벡터스토어를 찾을 수 없습니다")
                except Exception as e:
                    st.error(f"로드 실패: {str(e)}")
        
        st.divider()
        
        st.subheader("2. PDF 업로드 (선택)")
        uploaded_files = st.file_uploader(
            "PDF 파일 선택",
            type=['pdf'],
            accept_multiple_files=True,
            help="새로운 PDF를 추가하려면 파일을 선택하세요"
        )
        
        if uploaded_files and st.button("PDF 추가 및 저장", use_container_width=True):
            try:
                temp_dir = project_root / 'data' / 'temp'
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                pdf_paths = []
                for uploaded_file in uploaded_files:
                    temp_path = temp_dir / uploaded_file.name
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    pdf_paths.append(str(temp_path))
                
                pdf_callback, pdf_progress, pdf_status = create_pdf_progress_callback()
                summary_callback, summary_progress, summary_status = create_summary_progress_callback()
                
                vector_store = init_vector_store(
                    api_key,
                    pdf_callback=pdf_callback,
                    summary_callback=summary_callback
                )
                
                if st.session_state.db_loaded:
                    try:
                        vector_store.load(db_name)
                    except FileNotFoundError:
                        pass
                
                vector_store.add_documents(pdf_paths)
                
                pdf_progress.empty()
                pdf_status.empty()
                summary_progress.empty()
                summary_status.empty()
                
                with st.spinner("벡터스토어 저장 중..."):
                    vector_store.save(db_name)
                    st.session_state.vector_store = vector_store
                    st.session_state.db_loaded = True
                
                st.success(f"{len(uploaded_files)}개 파일 추가 완료")
                
                for temp_path in pdf_paths:
                    Path(temp_path).unlink(missing_ok=True)
                
            except Exception as e:
                st.error(f"PDF 처리 실패: {str(e)}")
        
        st.divider()
        
        st.subheader("시스템 정보")
        st.caption("**모델**: gpt-4o-mini")
        st.caption("**청크 크기**: 600자")
        st.caption("**오버랩**: 100자")
        st.caption("**요약 비율**: 20%")
    
    if not st.session_state.db_loaded:
        st.info("사이드바에서 벡터스토어를 로드하거나 PDF를 업로드하세요")
        st.stop()
    
    tab1, tab2 = st.tabs(["검색", "RAG 답변"])
    
    with tab1:
        st.subheader("검색 쿼리")
        query = st.text_input(
            "질문을 입력하세요",
            placeholder="예: RAG의 핵심 원리는 무엇인가요?",
            key="search_query"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            search_button = st.button("검색", use_container_width=True)
        
        if search_button and query:
            with st.spinner("검색 중..."):
                try:
                    results = st.session_state.vector_store.search(query)
                    
                    st.success(f"{len(results)}개 결과 발견")
                    display_search_results(results)
                    
                except Exception as e:
                    st.error(f"검색 실패: {str(e)}")
    
    with tab2:
        st.subheader("RAG 기반 답변 생성")
        query_rag = st.text_area(
            "질문을 입력하세요",
            placeholder="예: RAG 시스템의 장점과 단점을 설명해주세요",
            height=100,
            key="rag_query"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            generate_button = st.button("답변 생성", use_container_width=True)
        
        if generate_button and query_rag:
            with st.spinner("답변 생성 중..."):
                try:
                    context = st.session_state.vector_store.get_rag_context(query_rag)
                    
                    answer = st.session_state.vector_store.generate_answer(query_rag, context=context)
                    
                    st.markdown("### 답변")
                    st.markdown(answer)
                    
                    with st.expander("참조된 문서 컨텍스트"):
                        st.text(context)
                    
                except Exception as e:
                    st.error(f"답변 생성 실패: {str(e)}")
    
    st.divider()
    st.caption("2단계 검색 파이프라인 (요약문 → 원본) | 청크 크기: 600자 | 오버랩: 100자")


if __name__ == "__main__":
    main()