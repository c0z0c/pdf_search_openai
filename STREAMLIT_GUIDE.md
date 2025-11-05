# 📖 Streamlit RAG 검색 애플리케이션 사용 가이드

## 1. 로컬 환경 실행

### 1.1 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt

# .env 파일 생성
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### 1.2 애플리케이션 실행

```bash
# streamlit run app_main.py
conda run -n openai-notebook streamlit run app_main.py
# conda activate openai-notebook && streamlit run app_main.py
```

브라우저에서 `http://localhost:8501` 자동 실행

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

### 3.1 벡터스토어 로드

- 사이드바에서 DB 이름 입력 (기본값: `my_knowledge_base`)
- **벡터스토어 로드** 버튼 클릭
- 메타데이터 테이블 확인

### 3.2 PDF 업로드 (선택)

- 새로운 PDF를 추가할 경우 파일 업로드
- **PDF 추가 및 저장** 버튼 클릭
- 자동으로 벡터스토어에 저장

### 3.3 검색 기능

**🔍 검색 탭**:
- 키워드 기반 검색
- 결과: 파일명, 페이지, 유사도 점수 표시

**💬 RAG 답변 탭**:
- 자연어 질문 입력
- LLM 기반 답변 생성
- 참조 문서 컨텍스트 확인 가능

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

- **2단계 검색 파이프라인**: 요약문 → 원본 순차 검색
- **환경 자동 감지**: 로컬/.env ↔ Streamlit Cloud/secrets
- **청크 최적화**: 600자 청크, 100자 오버랩
- **실시간 메타데이터**: 파일별 페이지/청크 정보 표시

## 6. 문제 해결

### API 키 오류
- 로컬: `.env` 파일 확인
- Streamlit Cloud: Settings > Secrets 확인

### 벡터스토어 로드 실패
- `data/vectorstore` 디렉토리 존재 확인
- DB 이름이 정확한지 확인

### PDF 업로드 실패
- 파일 형식이 PDF인지 확인
- 파일 크기 제한 확인 (Streamlit Cloud: 200MB)

## 7. 시스템 사양

- **모델**: gpt-4o-mini
- **임베딩**: text-embedding-ada-002
- **벡터스토어**: FAISS
- **청크 크기**: 600자
- **오버랩**: 100자
