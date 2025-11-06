# Streamlit RAG 검색 애플리케이션 사용 가이드

[RAG 시스템 구현 관련 동영상 바로가기](https://youtu.be/Iv18gd7ouDA)  
[PDF RAG 검색 시스템 바로가기](https://pdfsearchopenai-5ckqof7mjy3gvnqxipnltt.streamlit.app/)

## 1. 로컬 환경 실행

### 1.1 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt

# .env 파일 생성 (방법 1: 직접 생성)
echo "OPENAI_API_KEY=your-api-key-here" > .env

# .env 파일 생성 (방법 2: 텍스트 에디터)
# .env 파일에 다음 내용 추가:
# OPENAI_API_KEY=sk-proj-...
```

### 1.2 애플리케이션 실행

```bash
# 기본 실행
streamlit run app_main.py

# conda 환경에서 실행
conda run -n openai-notebook streamlit run app_main.py

# 또는
conda activate openai-notebook
streamlit run app_main.py
```

브라우저에서 `http://localhost:8501` 자동 실행됩니다.

## 2. Streamlit Cloud 배포

### 2.1 GitHub 저장소 준비

```bash
git add .
git commit -m "Add Streamlit app"
git push origin main
```

### 2.2 Streamlit Cloud 배포

1. [Streamlit Cloud](https://streamlit.io/cloud) 접속
2. **New app** 클릭
3. GitHub 저장소 선택
4. **Main file path**: `app_main.py`
5. **Advanced settings** > **Secrets** 추가:
   ```toml
   OPENAI_API_KEY = "your-openai-api-key-here"
   ```
6. **Deploy** 클릭

## 3. 기능 설명

### 3.1 API 키 설정

**방법 1: 웹 UI 입력**
1. 애플리케이션 실행 시 API 키 입력 화면 표시
2. [OpenAI Platform](https://platform.openai.com/api-keys)에서 API 키 발급
3. 키 입력 후 "저장 및 시작" 버튼 클릭
4. 세션 동안 유지됨 (브라우저 닫으면 재입력 필요)

**방법 2: .env 파일 사용**
1. `.env` 파일에 `OPENAI_API_KEY=sk-...` 추가
2. 애플리케이션 자동 로드

**API 키 변경**
- 사이드바 "API 키 변경" 버튼 클릭
- 세션 및 환경 변수 초기화 후 재입력

### 3.2 벡터스토어 로드

1. 사이드바 "1. 벡터스토어 로드" 섹션
2. DB 이름 입력 (기본값: `my_knowledge_base`)
3. **벡터스토어 로드** 버튼 클릭
4. 메타데이터 테이블 확인:
   - 파일명
   - 페이지수
   - 청크개수
   - 청크유형
   - 파일해시

### 3.3 DB 관리

**파일별 삭제**
1. "2. DB 관리" 섹션에서 현재 DB 파일 목록 확인
2. "삭제할 파일 선택" 멀티셀렉트에서 파일 선택
3. **선택 파일 삭제** 버튼 클릭
4. 자동으로 VectorStore 저장됨

**DB 전체 초기화**
1. **DB 전체 초기화** 버튼 클릭
2. 경고 메시지 확인 후 진행
3. 모든 문서가 삭제되고 빈 VectorStore 생성

### 3.4 PDF 업로드 (선택)

1. "3. PDF 업로드 (선택)" 섹션
2. 파일 선택 (여러 파일 동시 선택 가능)
3. **PDF 추가 및 저장** 버튼 클릭
4. 진행 상황 실시간 표시:
   - **PDF 변환**: 페이지별 변환 진행률
   - **요약 생성**: 청크별 요약 및 압축률
5. 완료 후 자동으로 VectorStore 저장
6. 임시 파일 자동 삭제

### 3.5 검색 기능

**검색 탭**
1. 질문 입력 (예: "RAG의 핵심 원리는 무엇인가요?")
2. **검색** 버튼 클릭
3. 결과 확인:
   - 출처 순위
   - 파일명 및 페이지
   - 유사도 점수
   - 문서 내용

**RAG 답변 탭**
1. 자연어 질문 입력 (예: "연말 정산 때 비거주자가 주의할 점을 알려 줘")
2. **답변 생성** 버튼 클릭
3. LLM 기반 답변 확인
4. "참조된 문서 컨텍스트" 확장 메뉴로 출처 확인

### 3.6 모듈 강제 리로드

**사용 시점**
- VectorStore 메서드 누락 오류 발생 시
- `delete_by_file_name` 등의 메서드가 인식되지 않을 때

**사용 방법**
1. 사이드바 하단 "🔄 모듈 강제 리로드" 버튼 클릭
2. sys.modules 캐시 삭제 및 세션 초기화
3. 벡터스토어 재로드 필요

## 4. 디렉토리 구조

```
pdf_search_openai/
├── app_main.py              # Streamlit 앱
├── src/
│   ├── pdf_search.py        # RAG 시스템 코어
│   ├── helper_utils.py
│   └── helper_c0z0c_dev.py
├── data/
│   ├── vectorstore/         # 벡터스토어 저장 경로
│   │   ├── my_knowledge_base_summary/
│   │   └── my_knowledge_base_original/
│   └── temp/                # 임시 파일
├── .streamlit/
│   └── secrets.toml.example # Secrets 예시
├── requirements.txt
└── .env                     # 로컬 API 키 (gitignore)
```

## 5. 주요 특징

### 5.1 코어 기능
- **2단계 검색 파이프라인**: 요약문 → 원본 순차 검색
- **지능형 유사도 필터링**: 임계값 0.75 (관련 없는 문서 자동 제외)
- **청크 최적화**: 600자 청크, 100자 오버랩
- **20% 요약 생성**: LLM 기반 자동 요약

### 5.2 사용자 경험
- **실시간 진행 상황**: PDF 변환 및 요약 생성 프로세스 모니터링
- **멀티 파일 처리**: 여러 PDF 동시 업로드 및 삭제
- **메타데이터 조회**: 파일별 페이지/청크 정보 실시간 표시
- **에러 처리**: 빈 페이지, 변환 실패 등 안전 처리

### 5.3 기술적 특징
- **환경 자동 감지**: 로컬/.env ↔ Streamlit Cloud/secrets
- **해시 기반 중복 제거**: 동일 파일 재처리 방지
- **더미 문서 자동 관리**: 빈 VectorStore 초기화 시 더미 생성 및 자동 제거
- **모듈 핫 리로드**: sys.modules 캐시 삭제 및 동적 재로딩
- **세션 상태 관리**: API 키 및 VectorStore 상태 보존

## 6. 문제 해결

### 6.1 API 키 오류

**증상**: "OpenAI API 키를 입력하세요" 경고

**해결 방법**:
- 로컬: `.env` 파일 확인 (`OPENAI_API_KEY=sk-...`)
- Streamlit Cloud: Settings > Secrets 확인
- API 키 형식 확인 (sk-로 시작, 최소 20자)

### 6.2 벡터스토어 로드 실패

**증상**: "'{db_name}' 벡터스토어를 찾을 수 없습니다"

**해결 방법**:
- `data/vectorstore_db` 디렉토리 존재 확인
- DB 이름이 정확한지 확인 (예: `my_knowledge_base`)
- 초기 로드 시 빈 VectorStore 자동 생성됨

### 6.3 VectorStore 메서드 누락 오류

**증상**: "⚠️ VectorStore 메서드가 누락되었습니다"

**해결 방법**:
1. 사이드바 "🔄 모듈 강제 리로드" 버튼 클릭
2. 벡터스토어 재로드
3. 문제 지속 시 애플리케이션 재시작

### 6.4 PDF 업로드 실패

**증상**: "PDF 처리 실패" 오류

**해결 방법**:
- 파일 형식이 PDF인지 확인
- 파일 크기 제한 확인 (Streamlit Cloud: 200MB)
- 손상된 PDF 파일 여부 확인
- 임시 디렉토리 쓰기 권한 확인

### 6.5 검색 결과 없음

**증상**: "검색 결과가 없습니다"

**원인**:
- 질문과 문서의 유사도가 임계값(0.75) 미만
- VectorStore가 비어있음

**해결 방법**:
- 질문을 더 구체적으로 수정
- 관련 PDF 문서 추가
- 메타데이터 확인하여 DB에 문서가 있는지 확인

### 6.6 요약 생성 실패

**증상**: 특정 청크 요약 실패 메시지

**해결 방법**:
- OpenAI API 할당량 확인
- 네트워크 연결 상태 확인
- 실패한 청크는 원본 그대로 저장됨 (시스템 계속 동작)

## 7. 시스템 사양 및 설정

### 7.1 기본 설정

| 항목 | 값 | 설명 |
|------|-----|------|
| **LLM 모델** | gpt-4o-mini | 답변 생성 및 요약 |
| **임베딩 모델** | text-embedding-ada-002 | 벡터 변환 |
| **벡터스토어** | FAISS | 유사도 검색 |
| **청크 크기** | 600자 | 텍스트 분할 단위 |
| **청크 오버랩** | 100자 | 문맥 유지를 위한 중복 |
| **요약 비율** | 20% | 원본 대비 요약 크기 |
| **임베딩 배치** | 100 | 배치 처리 크기 |

### 7.2 검색 파라미터

| 항목 | 값 | 설명 |
|------|-----|------|
| **유사도 임계값** | 0.75 | 관련 없는 문서 필터링 |
| **요약문 검색 개수** | 5 | 1단계 후보 개수 |
| **최종 결과 개수** | 2 | 2단계 최종 선택 |
| **점수 차이 임계값** | 0.15 | 1등-2등 차이 기준 |

### 7.3 성능 최적화

**중복 제거**
- SHA-256 해시 기반 파일 중복 검증
- 동일 파일 재처리 방지

**배치 처리**
- 임베딩 생성 시 100개 단위 배치
- 메모리 효율적 처리

**더미 문서 관리**
- 빈 VectorStore 초기화 시 더미 생성
- 실제 문서 추가 시 자동 제거

## 8. 고급 사용법

### 8.1 Python API 연동

Streamlit 앱의 백엔드를 직접 사용:

```python
from src.pdf_search import VectorStore
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

vector_store = VectorStore(
    llm=llm,
    chunk_size=600,
    chunk_overlap=100,
    db_path="./data/vectorstore_db"
)

# 기존 DB 로드
vector_store.load("my_knowledge_base")

# 검색
results = vector_store.search("원천징수란?")

# 답변 생성
context = vector_store.get_rag_context("원천징수란?")
answer = vector_store.generate_answer("원천징수란?", context=context)
```

### 8.2 Progress Callback 커스터마이징

```python
def custom_pdf_callback(info):
    print(f"[PDF] {info['file_name']} - {info['current_page']}/{info['total_pages']}")
    if info['status'] == 'failed':
        print(f"  오류: {info['error']}")

def custom_summary_callback(info):
    if info['status'] == 'completed':
        print(f"[요약] 압축률: {info['compression_ratio']:.1%}")

vector_store = VectorStore(
    llm=llm,
    pdf_progress_callback=custom_pdf_callback,
    summary_progress_callback=custom_summary_callback
)
```

### 8.3 메타데이터 분석

```python
# 전체 메타데이터 조회
metadata_df = vector_store.get_metadata_info()
print(metadata_df)

# 특정 청크 상세 조회
chunk_info = vector_store.get_sample("sample.pdf", chunk_index=5)
print(f"원본: {chunk_info['original_length']}자")
print(f"요약: {chunk_info['summary_length']}자")
print(f"압축률: {chunk_info['compression_ratio']:.1%}")

# 시각화 출력
vector_store.print_sample("sample.pdf", 5)
```

## 9. 배포 및 운영

### 9.1 로컬 개발
- `.env` 파일로 API 키 관리
- `data/` 디렉토리에 VectorStore 저장
- Git에서 `.env` 및 `data/` 제외

### 9.2 Streamlit Cloud 배포
- GitHub 저장소 연결
- Settings > Secrets에 `OPENAI_API_KEY` 추가
- 자동 빌드 및 배포

### 9.3 모니터링
- Streamlit Cloud 대시보드에서 로그 확인
- 오류 발생 시 자동 재시작
- 사용량 및 성능 메트릭 제공

## 10. 참고 자료

- [RAG 시스템 구현 동영상](https://youtu.be/Iv18gd7ouDA)
- [PDF RAG 검색 시스템 (데모)](https://pdfsearchopenai-5ckqof7mjy3gvnqxipnltt.streamlit.app/)
- [프로젝트 README](./README.md)
- [상세 문서](./docs/pd_search.md)
