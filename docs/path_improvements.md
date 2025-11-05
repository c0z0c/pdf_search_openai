# 경로 수정 및 개선 사항

## 주요 변경사항

### 1. **모듈 Import 개선**

**Before:**
```python
from urllib.request import urlretrieve
urlretrieve("...", "helper_utils.py")
import helper_utils as hu
from helper_utils import *
```

**After:**
```python
import os
import pickle
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path  # 명시적 import 추가
import pandas as pd       # 추가
from tqdm import tqdm     # 추가
import pytz              # 추가

# Helper 파일 다운로드 (Colab 환경)
try:
    urlretrieve("...", "helper_utils.py")
    import helper_utils as hu
    from helper_utils import *
except Exception as e:
    # 로컬 환경에서는 helper 없이 작동
    IS_COLAB = False
    def drive_root():
        return "/content/drive/MyDrive"
```

**개선점:**
- `pathlib.Path`를 명시적으로 import하여 환경 독립성 확보
- 필수 패키지 `pandas`, `tqdm`, `pytz` 추가
- helper 파일이 없는 로컬 환경에서도 작동하도록 try-except 처리

---

### 2. **데이터 경로 자동 설정**

**Before:**
```python
data_path = str(Path(drive_root()) / 'data')
logger.info(f'데이터 경로 설정 완료: {data_path}')
```

**After:**
```python
# 프로젝트 루트 기준 경로 설정
if IS_COLAB:
    data_path = str(Path(drive_root()) / 'data')
else:
    # 로컬 환경: src 폴더 기준 상위 폴더의 data
    project_root = Path(__file__).parent.parent
    data_path = str(project_root / 'data')
    
logger.info(f'데이터 경로 설정 완료: {data_path}')
```

**개선점:**
- Colab과 로컬 환경을 자동 구분
- 로컬에서는 프로젝트 루트 기준 상대 경로 사용 (`src/` 기준 `../data/`)

---

### 3. **VectorStore 기본 DB 경로 자동 설정**

**Before:**
```python
def __init__(
    self,
    llm,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    db_path: str = "./vector_db"  # 하드코딩된 상대 경로
):
```

**After:**
```python
def __init__(
    self,
    llm,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    db_path: Optional[str] = None  # None으로 자동 경로 사용
):
    # db_path가 None이면 프로젝트 구조에 맞는 기본 경로 사용
    if db_path is None:
        if IS_COLAB:
            db_path = str(Path(drive_root()) / 'data' / 'vectorstore')
        else:
            # 로컬: src 기준 상위의 data/vectorstore
            project_root = Path(__file__).parent.parent
            db_path = str(project_root / 'data' / 'vectorstore')
```

**개선점:**
- `db_path=None`으로 자동 경로 설정 가능
- 표준 프로젝트 구조 `data/vectorstore/` 사용
- Colab과 로컬 환경 모두 지원

---

### 4. **__main__ 블록 간소화**

**Before:**
```python
db_path = str(Path(data_path) / r'mission14_vectorstore_db')
vector_store = VectorStore(llm=llm, chunk_size=600, chunk_overlap=100, db_path=db_path)
vector_store.load("my_knowledge_base")
```

**After:**
```python
# db_path=None으로 자동 경로 설정
vector_store = VectorStore(
    llm=llm,
    chunk_size=600,
    chunk_overlap=100,
    db_path=None  # 자동으로 data/vectorstore 경로 사용
)

logger.info(f"VectorStore DB 경로: {vector_store.db_manager.db_path}")

# 기존 벡터스토어 로드 시도
try:
    vector_store.load("my_knowledge_base")
    metadata_info = vector_store.get_metadata_info()
    print("\n=== 기존 벡터스토어 메타데이터 ===")
    print(metadata_info)
except FileNotFoundError:
    logger.info("기존 벡터스토어가 없습니다. 새로 생성합니다.")
```

**개선점:**
- 명시적 경로 지정 불필요
- 에러 핸들링 강화
- 로깅 개선

---

## 프로젝트 구조

```
pdf_search_openai/
├── src/
│   └── pdf_search.py           # 경로 자동 설정 적용
├── data/
│   └── vectorstore/            # 표준 DB 경로
│       ├── my_knowledge_base_original/
│       │   ├── index.faiss
│       │   └── index.pkl
│       └── my_knowledge_base_summary/
│           ├── index.faiss
│           └── index.pkl
├── examples/
│   └── basic_usage.py          # 사용 예제
└── README.md
```

---

## 사용법

### 기본 사용 (자동 경로)

```python
from langchain_openai import ChatOpenAI
from src.pdf_search import VectorStore

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# db_path 생략 시 자동으로 data/vectorstore 사용
vector_store = VectorStore(llm=llm)

# 문서 추가
vector_store.add_documents(["data/sample.pdf"])
vector_store.save("my_knowledge_base")
```

### 커스텀 경로 지정

```python
# 원하는 경로 직접 지정 가능
vector_store = VectorStore(
    llm=llm,
    db_path="./custom_db_path"
)
```

---

## 장점

1. **환경 독립성**: Colab/로컬 자동 구분
2. **상대 경로 자동 설정**: 프로젝트 어디서든 실행 가능
3. **표준 구조 준수**: `data/vectorstore/` 경로 사용
4. **유연성 유지**: 필요시 커스텀 경로 지정 가능
5. **에러 핸들링**: 파일 없음, 경로 오류 등 처리

---

## 테스트

### 로컬 환경에서 테스트

```bash
cd pdf_search_openai
python src/pdf_search.py
```

**예상 출력:**
```
INFO:__main__:데이터 경로 설정 완료: D:\GoogleDrive\pdf_search_openai\data
INFO:__main__:VectorStore DB 경로: D:\GoogleDrive\pdf_search_openai\data\vectorstore
```

### Colab 환경에서 테스트

```python
# Colab에서 실행시
!wget https://raw.githubusercontent.com/.../pdf_search.py
from pdf_search import VectorStore

# 자동으로 /content/drive/MyDrive/data/vectorstore 사용
```

---

## 마이그레이션

### 기존 코드 수정 방법

**기존:**
```python
db_path = "./data/vectorstore_db"
vector_store = VectorStore(llm=llm, db_path=db_path)
```

**권장:**
```python
# db_path 생략 (자동 경로 사용)
vector_store = VectorStore(llm=llm)
```

**또는 명시적 지정:**
```python
from pathlib import Path

project_root = Path(__file__).parent.parent
db_path = str(project_root / 'data' / 'vectorstore')
vector_store = VectorStore(llm=llm, db_path=db_path)
```

---

## 주의사항

1. **기존 DB 파일**: `data/vectorstore_db/` → `data/vectorstore/`로 이동 필요
2. **환경변수**: `.env` 파일에 `OPENAI_API_KEY` 설정 필수
3. **의존성**: `requirements.txt` 설치 필수

```bash
pip install -r requirements.txt
```
