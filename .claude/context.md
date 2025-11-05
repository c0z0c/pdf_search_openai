# 프로젝트: 14_3팀 RAG 시스템 (김명환)

당신은 AI 엔지니어 입니다.

---
# 가장 중요
- 인터넷에서 최신 정보를 검색하여 반영해 주세요.
- 문서의 내용은 항상 새로 읽어서 사용.
- 이모지 금지, 불필요한 장황함 금지.

---

## 코딩 스타일

### 필수 규칙
- 모든 함수에 타입 힌트 사용
- Google 스타일 Docstring (Args, Returns 포함)
- 함수/변수: `snake_case`, 클래스: `PascalCase`, 상수: `UPPER_CASE`
- 에러 핸들링: 항상 try-except 블록 사용
- 복잡한 로직에는 주석 필수
- PEP 8 + Black + isort.
- PEP 484, PEP 257 준수.

### 코드 구조 원칙
- 파이프라인은 단일 책임 원칙 준수
- 재사용 가능한 모듈로 분리
- 설정 값은 상수로 외부화

---

## 작성 규칙

### 항상 포함할 것
1. 에러 핸들링 및 의미 있는 에러 메시지
2. 진행 상황 출력 (print 또는 logging)
3. 메타데이터 검증 (파일명, 페이지, 인덱스)
4. 검색 유사도 임계값 체크

### 절대 하지 말 것
- API 키 하드코딩 (환경 변수 사용)
- 전체 파일 경로 저장 (파일명만 저장)
- 에러 발생 시 무시 (명확한 에러 처리)
- 매직 넘버 사용 (상수로 정의)

---

## 자주 사용하는 패턴

### 파일명만 추출
```python
from pathlib import Path
file_name = Path(pdf_path).name
```

# 상호작용 프로토콜(설명 후 확인 → 코딩)
1) 요구 요약: 목표/제약/산출물. 
2) 접근 전략 설명: 선택지·트레이드오프, 변경 파일, 성능/재현성 영향.  
3) 확인 질문: “이 전략으로 코딩 진행할까요? (예/아니오/수정)”
4) 예 혹은 승인 후 코딩: MVP 최소 실행 예제, 경로/의존성/헬퍼 호출, 타입힌트, Docstring 포함.
5) 20라인 내의 짧은 단문 일 경우 확인 질문 없이 바로 내용 진행

# 기본 스택(우선 import)
- DL: torch, torchvision, torchaudio, transformers, datasets, accelerate, peft, sentencepiece
- Data: numpy, pandas, scikit-learn, scipy
- Viz: matplotlib, seaborn, plotly(선택)
- Exp: wandb, tqdm, pytz, rich(선택)
- Dev: jupyter, ipykernel, python-dotenv, pathlib(표준), json/pickle/yaml(선택)
- Lint/Type: black, isort, ruff/flake8, mypy(선택)
- TrainingArguments: transformers.TrainingArguments
- Tokenizers: transformers.PreTrainedTokenizerFast
- HuggingFace Datasets: datasets.Dataset

# 헬퍼 모듈 사용 규칙
- 중복 구현 금지: 동일 기능은 헬퍼로 해결.
- helper_utils.py 파일을 읽어서 선언된 함수를 사용하여 기능을 구현.
- helper_c0z0c_dev.py 파일을 읽어서 선언된 함수를 사용하여 기능을 구현.
- 경로/저장 로직: get_path_modeling(), save_model_dict() 패턴 준수.
- Colab/로컬 분기: drive_root() 사용.
```python
# 헬퍼 로드(필수 스니펫)
from urllib.request import urlretrieve; urlretrieve("https://raw.githubusercontent.com/c0z0c/jupyter_hangul/refs/heads/beta/helper_utils.py", "helper_utils.py")
import importlib, helper_utils as hu
importlib.reload(hu); from helper_utils import *
```

# 전역 및 초기화(노트북 공통)
```python
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_device_cpu = torch.device('cpu')
_kst = pytz.timezone('Asia/Seoul')
```

# 주요 컴포넌트
## 학습시
**WandB 설정**:
```python
wandb.init(
    entity="c0z0c-dev-home",
    project="xxx",  # 프로젝트명
    name=f"{timestamp(yymmdd_HHMMSS)}_{model_name}"
    job_type="training",
    tags=["training"],
)
```

# 모델 평가시
**WandB 설정**
```python
wandb.init(
    entity="c0z0c-dev-home",
    project="sprint_mission_10",
    name=f"EVAL_{model_name}_{timestamp}",
    job_type="evaluation",
    tags=["evaluation"],
    reinit=True  # 필수
)
```

## Model Classes
**설계 원칙**:
- 단일 책임: Dataset/Model/Pipeline 분리
- 헬퍼 통합: `get_path_modeling()`, `save_model_dict()` 사용
- 타입 힌트 필수
- 상속 활용: Base → 특화 클래스

# Documentation (GitHub Pages)
- 목차: `1.` `1.1.` `1.1.1`
- Mermaid: 노드 라벨 큰따옴표 `A["노드"]`
- 수식: `$$` 블록 우선
- 용어: 한영 병기 (normalization, 노멀라이제이션)
- 이모지 금지
