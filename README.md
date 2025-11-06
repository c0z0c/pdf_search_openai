# PDF Search with OpenAI

[RAG 시스템 구현 관련 동영상 바로가기](https://youtu.be/Iv18gd7ouDA)  
[PDF RAG 검색 시스템 바로가기](https://pdfsearchopenai-5ckqof7mjy3gvnqxipnltt.streamlit.app/)

PDF 문서 기반 RAG (Retrieval-Augmented Generation) 시스템

## 개요

2단계 검색 파이프라인을 활용한 고급 RAG 시스템입니다.
- PDF를 Markdown으로 변환하여 청킹
- 원본 청크의 20% 크기로 요약본 생성
- 요약문 검색 → 원본 문서 검색 (2단계)
- FAISS 벡터스토어 기반 유사도 검색
- Streamlit 웹 인터페이스 제공

## 주요 기능

### 코어 기능
- **문서 처리**: PDF → Markdown 변환 (pymupdf4llm)
- **지능형 청킹**: 600자 청크 + 100자 오버랩
- **자동 요약**: LLM 기반 20% 요약 생성
- **2단계 검색**: 요약문으로 후보 선정 → 원본 문서 정밀 검색
- **해시 기반 중복 제거**: 동일 파일 재처리 방지
- **벡터스토어 관리**: 저장/로드/삭제/목록 조회

### 웹 인터페이스 (Streamlit)
- **실시간 진행 상황 표시**: PDF 변환 및 요약 생성 프로세스 모니터링
- **DB 관리**: 파일별 선택 삭제, 전체 초기화
- **메타데이터 조회**: 파일명, 페이지수, 청크개수 확인
- **모듈 강제 리로드**: VectorStore 클래스 동적 재로딩
- **API 키 관리**: 세션 기반 안전한 키 저장

## 설치

### 요구사항
- Python 3.8+
- OpenAI API Key

### 패키지 설치

```bash
pip install -r requirements.txt
```

### 환경변수 설정

`.env` 파일 생성:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## 사용법

### Streamlit 웹 애플리케이션

#### 로컬 실행
```bash
# 의존성 설치
pip install -r requirements.txt

# .env 파일 생성
echo "OPENAI_API_KEY=your-api-key-here" > .env

# 애플리케이션 실행
streamlit run app_main.py
```

브라우저에서 `http://localhost:8501` 자동 실행

#### 주요 기능
1. **API 키 입력**: 웹 UI에서 직접 입력 또는 .env 파일 사용
2. **벡터스토어 로드**: DB 이름 입력 후 로드 버튼 클릭
3. **PDF 업로드**: 새 문서 추가 시 진행 상황 실시간 표시
4. **DB 관리**: 
   - 파일별 선택 삭제 (멀티 선택 지원)
   - DB 전체 초기화
   - 메타데이터 조회
5. **검색/RAG 답변**: 
   - 검색: 유사도 기반 문서 검색
   - RAG 답변: LLM 기반 자연어 답변 생성

### Python API 사용

#### 기본 사용

```python
from langchain_openai import ChatOpenAI
from src.pdf_search import VectorStore

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# VectorStore 생성 (progress callback 옵션)
def pdf_callback(info):
    print(f"PDF 변환: {info['file_name']} ({info['current_page']}/{info['total_pages']})")

def summary_callback(info):
    if info['status'] == 'completed':
        print(f"요약 완료: 압축률 {info['compression_ratio']:.1%}")

vector_store = VectorStore(
    llm=llm,
    chunk_size=600,
    chunk_overlap=100,
    db_path="./data/vectorstore_db",
    pdf_progress_callback=pdf_callback,
    summary_progress_callback=summary_callback
)

# 문서 추가
pdf_files = ["path/to/document.pdf"]
vector_store.add_documents(pdf_files)

# 저장
vector_store.save("my_knowledge_base")

# 검색
query = "RAG의 핵심 원리는 무엇인가요?"
context = vector_store.get_rag_context(query)

# 답변 생성
answer = vector_store.generate_answer(query, context=context)
print(answer)
```

#### 벡터스토어 관리

```python
# 기존 벡터스토어 로드
vector_store.load("my_knowledge_base")

# 메타데이터 조회
metadata_info = vector_store.get_metadata_info()
print(metadata_info)

# 특정 파일 삭제
vector_store.delete_by_file_name("sample.pdf")
vector_store.save("my_knowledge_base")

# 특정 청크 조회
chunk_info = vector_store.get_sample("sample.pdf", chunk_index=5)
print(f"원본: {chunk_info['original_length']}자")
print(f"요약: {chunk_info['summary_length']}자")
```

## 프로젝트 구조

```
pdf_search_openai/
├── README.md                     # 프로젝트 설명
├── STREAMLIT_GUIDE.md            # Streamlit 앱 사용 가이드
├── requirements.txt              # Python 패키지 의존성
├── app_main.py                   # Streamlit 웹 애플리케이션
├── .env                          # 환경변수 (로컬, gitignore)
├── .gitignore                    # Git 제외 파일 목록
├── src/
│   ├── __init__.py
│   ├── pdf_search.py            # RAG 시스템 코어
│   ├── helper_utils.py          # 유틸리티 함수
│   └── helper_c0z0c_dev.py      # 개발 헬퍼 함수
├── docs/                         # 문서
│   ├── pd_search.md
│   ├── deployment_fix.md
│   └── path_improvements.md
├── data/                         # 데이터 파일
│   ├── temp/                    # 임시 파일 (업로드된 PDF)
│   └── vectorstore_db/          # VectorStore DB
│       ├── my_knowledge_base_original/
│       │   └── index.faiss
│       └── my_knowledge_base_summary/
│           └── index.faiss
└── examples/                     # 예제 스크립트
    └── basic_usage.py
```

## 아키텍처

### 클래스 구조

1. **FileHashManager**: 파일 해시 계산 및 중복 검증
2. **DocumentProcessingPipeline**: PDF → Markdown → 청킹 (progress callback 지원)
3. **SummaryPipeline**: 청크 요약 (20% 크기, progress callback 지원)
4. **TwoStageSearchPipeline**: 2단계 검색 (요약문 → 원본)
5. **VectorStoreManager**: DB 저장/로드/관리
6. **VectorStore**: 통합 인터페이스

### 처리 흐름

```
PDF 파일
  ↓
[파일 해시 확인] ← 중복 제거
  ↓
[Markdown 변환] → [progress callback]
  ↓
[청킹 (600자)]
  ↓
[요약 생성 (20%)] → [progress callback]
  ↓
[벡터 임베딩]
  ↓
[FAISS 저장]
  ↓
[2단계 검색]
  ① 요약문 검색 (top_k=5)
  ② 원본 문서 검색 (top_k=2)
  ↓
[RAG 답변 생성]
```

### 주요 개선 사항

#### 1. Progress Callback 시스템
- PDF 변환 진행 상황 실시간 전달
- 요약 생성 진행률 및 압축률 모니터링
- Streamlit UI와 통합하여 사용자 경험 향상

#### 2. 더미 문서 자동 관리
- 빈 VectorStore 생성 시 더미 문서 자동 생성
- 실제 문서 추가 시 더미 자동 제거
- DB 크기 최적화

#### 3. 파일별 삭제 기능
- `delete_by_file_name()`: 특정 파일의 모든 청크 삭제
- FAISS 인덱스 및 docstore 동기화
- processed_files 해시 테이블 업데이트

#### 4. 모듈 동적 리로딩
- `force_reload_modules()`: sys.modules 캐시 삭제
- VectorStore 클래스 핫 리로드
- 세션 상태 초기화

## 설정

### VectorStore 초기화

```python
vector_store = VectorStore(
    llm=llm,
    chunk_size=600,                    # 청크 크기
    chunk_overlap=100,                 # 오버랩 크기
    db_path="./data/vectorstore_db",
    embedding_batch_size=100,          # 임베딩 배치 크기
    pdf_progress_callback=None,        # PDF 변환 콜백 (선택)
    summary_progress_callback=None     # 요약 생성 콜백 (선택)
)
```

### 검색 파라미터

`TwoStageSearchPipeline`에서 파라미터 조정:
- `similarity_threshold`: 유사도 임계값 (기본: 0.75)
- `top_k_summary`: 요약문 검색 개수 (기본: 5)
- `top_k_final`: 최종 결과 개수 (기본: 2)
- `score_gap_threshold`: 1등-2등 점수 차이 임계값 (기본: 0.15)

### 요약 설정

`SummaryPipeline`에서 파라미터 조정:
- `summary_ratio`: 요약 비율 (기본: 0.2 = 20%)
- `min_length`: 최소 요약 길이 (기본: 100자)

## API 메서드

### VectorStore 주요 메서드

| 메서드 | 설명 | 반환 |
|--------|------|------|
| `add_documents(pdf_paths)` | PDF 문서 추가 및 임베딩 | None |
| `save(name)` | 벡터스토어 저장 | None |
| `load(name)` | 벡터스토어 로드 | None |
| `delete(name)` | 벡터스토어 삭제 | None |
| `search(query)` | 2단계 검색 실행 | List[Dict] |
| `get_rag_context(query)` | RAG용 컨텍스트 생성 | str |
| `generate_answer(query, context)` | LLM 답변 생성 | str |
| `get_metadata_info()` | 파일별 메타데이터 조회 | pd.DataFrame |
| `delete_by_file_name(file_name)` | 특정 파일 삭제 | bool |
| `get_sample(file_name, chunk_index)` | 특정 청크 조회 | Dict |
| `print_sample(file_name, chunk_index)` | 청크 정보 출력 | None |

## 문제 해결

### API 키 오류
- 로컬: `.env` 파일 확인
- Streamlit Cloud: Settings > Secrets 확인

### VectorStore 로드 실패
- `data/vectorstore_db` 디렉토리 존재 확인
- DB 이름이 정확한지 확인 (예: `my_knowledge_base`)

### 모듈 메서드 누락 오류
- Streamlit 앱에서 "🔄 모듈 강제 리로드" 버튼 클릭
- 또는 `force_reload_modules()` 호출

### PDF 업로드 실패
- 파일 형식이 PDF인지 확인
- 파일 크기 제한 확인 (Streamlit Cloud: 200MB)

## 기술 스택

- **LLM**: gpt-4o-mini
- **임베딩**: text-embedding-ada-002 (OpenAI)
- **벡터스토어**: FAISS
- **PDF 처리**: pymupdf4llm, PyMuPDF
- **텍스트 분할**: LangChain RecursiveCharacterTextSplitter
- **웹 프레임워크**: Streamlit

## 라이선스

MIT License

## 참고 문서

- [RAG 시스템 구현 동영상](https://youtu.be/Iv18gd7ouDA)
- [PDF RAG 검색 시스템 (데모)](https://pdfsearchopenai-5ckqof7mjy3gvnqxipnltt.streamlit.app/)
- [Streamlit 사용 가이드](./STREAMLIT_GUIDE.md)
- [상세 문서](./docs/pd_search.md)

## 문의

Author: 14_3팀_김명환
