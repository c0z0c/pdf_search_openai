# Streamlit Cloud 배포 수정사항

## 수정 날짜
2025-11-06

## 문제
- ModuleNotFoundError: helper_utils를 찾을 수 없음
- Streamlit Cloud 배포 환경에서 모듈 경로 인식 실패

## 해결 방법

### 1. `src/__init__.py` 생성
- `src` 디렉토리를 Python 패키지로 만듦
- `VectorStore` 클래스를 패키지 레벨에서 export

### 2. `src/pdf_search.py` 수정
- 상대 import 우선 시도 (`.helper_utils`)
- 절대 import fallback 추가
- 로컬/배포 환경 모두 호환

### 3. `app_main.py` 수정
- `from src.pdf_search import VectorStore`로 명시적 import
- `sys.path`에 프로젝트 루트 추가

### 4. `requirements.txt` 수정
- `streamlit-cookies-manager` → `extra-streamlit-components` 변경
- 코드에서 실제 사용하는 패키지와 일치

## 배포 체크리스트
- [x] `src/__init__.py` 생성
- [x] 상대 import 수정
- [x] requirements.txt 정확성 확인
- [ ] Git push 후 Streamlit Cloud 재배포
- [ ] 배포 로그 확인

## 테스트 방법
```bash
# 로컬 테스트
python -c "from src.pdf_search import VectorStore; print('성공')"

# Streamlit 실행
streamlit run app_main.py
```

## 주의사항
- `helper_utils.py`, `helper_c0z0c_dev.py`는 `src/` 디렉토리에 유지
- 상대 import가 실패하면 자동으로 절대 import 시도
- Git에 `src/__init__.py` 포함 필수
