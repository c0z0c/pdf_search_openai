# PDF Search with OpenAI

PDF 문서 기반 RAG (Retrieval-Augmented Generation) 시스템

## 개요

2단계 검색 파이프라인을 활용한 고급 RAG 시스템입니다.
- PDF를 Markdown으로 변환하여 청킹
- 원본 청크의 20% 크기로 요약본 생성
- 요약문 검색 → 원본 문서 검색 (2단계)
- FAISS 벡터스토어 기반 유사도 검색

## 주요 기능

- **문서 처리**: PDF → Markdown 변환 (pymupdf4llm)
- **지능형 청킹**: 600자 청크 + 100자 오버랩
- **자동 요약**: LLM 기반 20% 요약 생성
- **2단계 검색**: 요약문으로 후보 선정 → 원본 문서 정밀 검색
- **해시 기반 중복 제거**: 동일 파일 재처리 방지
- **벡터스토어 관리**: 저장/로드/삭제/목록 조회

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

### 기본 사용

```python
from langchain_openai import ChatOpenAI
from src.pdf_search import VectorStore

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# VectorStore 생성
vector_store = VectorStore(
    llm=llm,
    chunk_size=600,
    chunk_overlap=100,
    db_path="./data/vectorstore"
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

### 벡터스토어 로드

```python
# 기존 벡터스토어 로드
vector_store.load("my_knowledge_base")

# 메타데이터 조회
metadata_info = vector_store.get_metadata_info()
print(metadata_info)
```

## 프로젝트 구조

```
pdf_search_openai/
├── README.md                     # 프로젝트 설명
├── requirements.txt              # Python 패키지 의존성
├── .env.example                  # 환경변수 예제
├── .gitignore                    # Git 제외 파일 목록
├── src/
│   └── pdf_search.py            # 메인 코드
├── docs/                         # 문서
│   └── pd_search.md
├── data/                         # 데이터 파일
│   └── vectorstore/             # VectorStore DB
│       ├── original/
│       └── summary/
└── examples/                     # 예제 스크립트 (향후)
```

## 아키텍처

### 클래스 구조

1. **DocumentProcessingPipeline**: PDF → Markdown → 청킹
2. **SummaryPipeline**: 청크 요약 (20% 크기)
3. **TwoStageSearchPipeline**: 2단계 검색 (요약문 → 원본)
4. **VectorStoreManager**: DB 저장/로드/관리
5. **VectorStore**: 통합 인터페이스

### 처리 흐름

```
PDF 파일
  ↓
[Markdown 변환]
  ↓
[청킹 (600자)]
  ↓
[요약 생성 (20%)]
  ↓
[벡터 임베딩]
  ↓
[FAISS 저장]
  ↓
[2단계 검색]
  ↓
[RAG 답변 생성]
```

## 설정

### 청킹 설정

```python
vector_store = VectorStore(
    llm=llm,
    chunk_size=600,          # 청크 크기
    chunk_overlap=100,       # 오버랩 크기
    db_path="./data/vectorstore"
)
```

### 검색 설정

`TwoStageSearchPipeline`에서 파라미터 조정:
- `similarity_threshold`: 유사도 임계값 (기본: 0.5)
- `top_k_summary`: 요약문 검색 개수 (기본: 5)
- `top_k_final`: 최종 결과 개수 (기본: 2)
- `score_gap_threshold`: 1등-2등 점수 차이 임계값 (기본: 0.15)

## 라이선스

MIT License

## 문의

Author: 14_3팀_김명환
