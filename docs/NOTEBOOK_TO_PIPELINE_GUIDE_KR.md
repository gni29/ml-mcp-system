# Jupyter 노트북을 ML 파이프라인으로 변환 가이드

탐색적 Jupyter 노트북을 프로덕션 준비된 ML 파이프라인으로 자동 변환합니다.

---

## 🎯 개요

**노트북-파이프라인** 변환기는 Jupyter 노트북을 분석하여 다음을 추출합니다:
- 데이터 로딩 코드
- 전처리 단계
- 특징 엔지니어링
- 모델 학습
- 평가 메트릭
- 예측 로직

그리고 다음을 생성합니다:
- ✅ 구조화된 Python 파이프라인 스크립트
- ✅ 설정 파일 (JSON)
- ✅ 테스트 파일 (선택사항)
- ✅ CLI 인터페이스
- ✅ 사용하기 쉬운 MLPipeline 클래스

---

## 🚀 빠른 시작

### Python으로 직접 사용

```python
from python.ml.pipeline.notebook_to_pipeline import NotebookToPipeline

# 변환기 초기화
transformer = NotebookToPipeline('analysis.ipynb', framework='sklearn')

# 노트북 파싱
parse_result = transformer.parse_notebook()
print(f"{parse_result['total_cells']}개의 코드 셀 발견")
print(f"감지된 프레임워크: {parse_result['framework']}")

# 파이프라인 생성
files = transformer.generate_pipeline(
    output_path='ml_pipeline.py',
    include_tests=True,
    include_config=True
)

print(f"생성됨: {files}")

# 요약 출력
print(transformer.generate_summary())
```

### CLI 사용

```bash
# 기본 사용법
python -m python.ml.pipeline.notebook_to_pipeline \
  --notebook analysis.ipynb \
  --output ml_pipeline.py

# 테스트 및 요약 포함
python -m python.ml.pipeline.notebook_to_pipeline \
  --notebook analysis.ipynb \
  --output ml_pipeline.py \
  --include-tests \
  --summary

# 프레임워크 지정
python -m python.ml.pipeline.notebook_to_pipeline \
  --notebook pytorch_model.ipynb \
  --output pytorch_pipeline.py \
  --framework pytorch
```

### MCP 도구 사용

```javascript
await mcp.call('notebook_to_pipeline', {
  notebook_path: 'experiments/analysis.ipynb',
  output_path: 'pipelines/ml_pipeline.py',
  framework: 'sklearn',
  include_tests: true,
  include_config: true,
  show_summary: true
});
```

---

## 📋 생성되는 파일

### 1. 메인 파이프라인 파일 (`ml_pipeline.py`)

```python
"""
ML 파이프라인
Jupyter 노트북에서 생성: analysis.ipynb
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 설정
CONFIG = {
    'data': {'input_path': 'data/input', 'test_size': 0.2},
    'preprocessing': {'handle_missing': True, 'scale_features': True},
    'model': {'save_path': 'models'}
}

def load_data(config):
    """데이터 로드 및 준비"""
    # 노트북의 데이터 로딩 코드가 여기에 들어갑니다
    pass

def preprocess_data(X, y, config):
    """특징 전처리"""
    # 노트북의 전처리 코드가 여기에 들어갑니다
    pass

def train_model(X_train, y_train, config):
    """ML 모델 학습"""
    # 노트북의 학습 코드가 여기에 들어갑니다
    pass

def evaluate_model(model, X_test, y_test):
    """모델 성능 평가"""
    # 노트북의 평가 코드가 여기에 들어갑니다
    pass

class MLPipeline:
    """완전한 ML 파이프라인"""

    def __init__(self, config=None):
        self.config = config or CONFIG
        self.model = None

    def fit(self, X, y):
        """완전한 파이프라인 학습"""
        X_processed, y_processed = preprocess_data(X, y, self.config)
        self.model = train_model(X_processed, y_processed, self.config)
        return self

    def predict(self, X):
        """예측 수행"""
        X_processed, _ = preprocess_data(X, None, self.config)
        return self.model.predict(X_processed)

    def save(self, path):
        """파이프라인 저장"""
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        """파이프라인 로드"""
        import joblib
        return joblib.load(path)

# CLI 인터페이스
if __name__ == '__main__':
    import argparse
    # 완전한 CLI 구현...
```

### 2. 설정 파일 (`ml_pipeline_config.json`)

```json
{
  "notebook": "analysis.ipynb",
  "generated": "2025-10-01T10:30:00",
  "framework": "sklearn",
  "dependencies": ["pandas", "numpy", "scikit-learn"],
  "components": {
    "data_loading": true,
    "preprocessing": true,
    "feature_engineering": true,
    "model_training": true,
    "model_evaluation": true
  },
  "pipeline_config": {
    "data": {
      "input_path": "data/input.csv",
      "test_size": 0.2,
      "random_state": 42
    },
    "preprocessing": {
      "handle_missing": true,
      "scale_features": true
    },
    "model": {
      "save_path": "models/model.pkl"
    }
  }
}
```

### 3. 테스트 파일 (`test_ml_pipeline.py`)

```python
"""ML 파이프라인 테스트"""

import unittest
import pandas as pd
import numpy as np
from ml_pipeline import MLPipeline

class TestMLPipeline(unittest.TestCase):
    """ML 파이프라인 테스트"""

    def setUp(self):
        self.pipeline = MLPipeline()
        # 샘플 데이터 생성
        self.X_train = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        self.y_train = pd.Series(np.random.randint(0, 2, 100))

    def test_pipeline_fit(self):
        """파이프라인 학습 테스트"""
        self.pipeline.fit(self.X_train, self.y_train)
        self.assertIsNotNone(self.pipeline.model)

    def test_pipeline_predict(self):
        """파이프라인 예측 테스트"""
        self.pipeline.fit(self.X_train, self.y_train)
        predictions = self.pipeline.predict(self.X_train)
        self.assertEqual(len(predictions), len(self.X_train))

if __name__ == '__main__':
    unittest.main()
```

---

## 💡 사용 예제

### 예제 1: 모델 학습 및 저장

```bash
# 모델 학습
python ml_pipeline.py --train data/train.csv --model-path model.pkl

# 커스텀 설정 사용
python ml_pipeline.py --train data/train.csv --config my_config.json --model-path model.pkl
```

### 예제 2: 예측 수행

```bash
# 예측 수행
python ml_pipeline.py --predict data/new_data.csv --model-path model.pkl
```

### 예제 3: 학습 및 평가

```bash
# 한 번에 학습 및 평가
python ml_pipeline.py \
  --train data/train.csv \
  --test data/test.csv \
  --model-path model.pkl
```

### 예제 4: 라이브러리로 사용

```python
from ml_pipeline import MLPipeline
import pandas as pd

# 데이터 로드
X_train = pd.read_csv('data/train.csv')
y_train = X_train['target']
X_train = X_train.drop('target', axis=1)

# 파이프라인 생성 및 학습
pipeline = MLPipeline()
pipeline.fit(X_train, y_train)

# 모델 저장
pipeline.save('models/my_model.pkl')

# 나중에: 로드 및 예측
pipeline = MLPipeline.load('models/my_model.pkl')
X_new = pd.read_csv('data/new_data.csv')
predictions = pipeline.predict(X_new)
```

---

## 🔍 추출되는 내용

변환기는 자동으로 다음을 식별하고 추출합니다:

### 1. 데이터 로딩
```python
# 다음과 같은 패턴 감지:
df = pd.read_csv('data.csv')
df = pd.read_excel('data.xlsx')
data = np.load('data.npy')
```

### 2. 전처리
```python
# 다음과 같은 패턴 감지:
df.fillna(df.mean())
df.dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 3. 특징 엔지니어링
```python
# 다음과 같은 패턴 감지:
selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)
```

### 4. 모델 학습
```python
# 다음과 같은 패턴 감지:
model = RandomForestClassifier()
model.fit(X_train, y_train)
model = XGBClassifier()
model.fit(X_train, y_train)
```

### 5. 평가
```python
# 다음과 같은 패턴 감지:
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
confusion_matrix(y_test, y_pred)
```

---

## ⚙️ 설정 옵션

### 프레임워크 감지

변환기는 사용된 ML 프레임워크를 자동 감지합니다:
- **sklearn**: scikit-learn 모델
- **pytorch**: PyTorch 모델
- **tensorflow**: TensorFlow/Keras 모델
- **xgboost**: XGBoost 모델
- **lightgbm**: LightGBM 모델

수동으로 지정할 수도 있습니다:
```python
transformer = NotebookToPipeline('notebook.ipynb', framework='sklearn')
```

### 생성 옵션

```python
files = transformer.generate_pipeline(
    output_path='pipeline.py',       # 출력 파일 경로
    include_tests=True,               # 테스트 파일 생성
    include_config=True               # 설정 파일 생성
)
```

---

## 📊 변환 요약

변환 후 상세한 요약을 받을 수 있습니다:

```
노트북-파이프라인 변환 요약
==================================================

노트북: analysis.ipynb
프레임워크: sklearn
총 셀 수: 25

추출된 컴포넌트:
  - Imports: 5
  - 데이터 로딩: 2
  - 전처리: 4
  - 특징 엔지니어링: 2
  - 모델 학습: 3
  - 모델 평가: 2
  - 예측: 1
  - 시각화: 4
  - 유틸리티 함수: 2

종속성: numpy, pandas, scikit-learn, matplotlib

생성된 파이프라인 구조:
  ✓ 데이터 로딩 함수
  ✓ 전처리 함수
  ✓ 특징 엔지니어링 함수
  ✓ 학습 함수
  ✓ 평가 함수
  ✓ 예측 함수
  ✓ 파이프라인 클래스
  ✓ CLI 인터페이스
```

---

## 🎯 모범 사례

### 1. 노트북 구성

최상의 결과를 위해 명확한 섹션으로 노트북을 구성하세요:
- 데이터 로딩
- 탐색적 데이터 분석 (EDA)
- 전처리
- 특징 엔지니어링
- 모델 학습
- 평가
- 예측

### 2. 설명적인 변수명 사용

```python
# 좋음
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 피하기
a, b, c, d = train_test_split(X, y)
```

### 3. 함수를 모듈화하여 유지

```python
# 좋음 - 추출하기 쉬움
def preprocess_data(df):
    df = df.fillna(df.mean())
    return df

# 피하기 - 추출하기 어려움
df = df.fillna(df.mean())  # 노트북 전체에 흩어짐
```

### 4. 생성된 코드 검토

항상 생성된 파이프라인을 검토하고 테스트하세요:
```bash
# 테스트 실행
python test_ml_pipeline.py

# 샘플 데이터로 테스트
python ml_pipeline.py --train sample_data.csv --test sample_test.csv
```

---

## 🔧 고급 사용법

### 커스텀 파이프라인 템플릿

변환기를 확장하여 커스텀 템플릿을 사용할 수 있습니다:

```python
class CustomNotebookToPipeline(NotebookToPipeline):
    def _generate_pipeline_class(self):
        # 커스텀 파이프라인 클래스 생성
        return custom_code
```

### MLOps와 통합

MLflow와 결합하여 실험 추적:

```python
from python.ml.mlops.mlflow_tracker import MLflowTracker
from ml_pipeline import MLPipeline

tracker = MLflowTracker(experiment_name='my_experiment')

with tracker.start_run():
    pipeline = MLPipeline()
    pipeline.fit(X_train, y_train)

    # 메트릭 로깅
    tracker.log_metrics({'accuracy': 0.95})

    # 모델 로깅
    tracker.log_model(pipeline.model, 'model')
```

### 배포

생성된 파이프라인 배포:

```python
from python.ml.deployment.model_server import ModelServer

# 파이프라인 저장
pipeline.save('models/pipeline.pkl')

# API를 통해 서빙
server = ModelServer()
server.register_model('my_pipeline', 'models/pipeline.pkl', 'classifier')
server.start()
```

---

## 🐛 문제 해결

### 문제: Import 누락
**해결책**: 노트북의 첫 번째 셀에 필요한 import 추가

### 문제: 복잡한 전처리가 추출되지 않음
**해결책**: 노트북에서 전처리를 함수로 리팩토링

### 문제: 프레임워크가 감지되지 않음
**해결책**: 프레임워크를 명시적으로 지정:
```python
transformer = NotebookToPipeline('notebook.ipynb', framework='sklearn')
```

### 문제: 생성된 코드가 실행되지 않음
**해결책**:
1. 생성된 코드 검토
2. 모든 종속성이 나열되어 있는지 확인
3. 먼저 샘플 데이터로 테스트
4. 필요에 따라 생성된 코드 수정

---

## 📚 예제

`examples/notebook_to_pipeline/`에서 완전한 예제를 참조하세요:
- `example_sklearn.ipynb` - scikit-learn 분류
- `example_pytorch.ipynb` - PyTorch 딥러닝
- `example_timeseries.ipynb` - 시계열 예측
- `example_nlp.ipynb` - NLP 파이프라인

---

## 🚀 다음 단계

파이프라인을 생성한 후:

1. **검토**: 생성된 코드 검토
2. **테스트**: 샘플 데이터로 테스트
3. **커스터마이즈**: 필요에 따라 설정 조정
4. **통합**: MLOps 도구와 통합 (MLflow, 모델 서빙)
5. **배포**: 프로덕션에 배포

---

## 🔄 전체 워크플로우 예제

```python
# 1단계: 노트북 변환
from python.ml.pipeline.notebook_to_pipeline import NotebookToPipeline

transformer = NotebookToPipeline('customer_churn.ipynb')
transformer.parse_notebook()
files = transformer.generate_pipeline(
    output_path='churn_pipeline.py',
    include_tests=True,
    include_config=True
)

# 2단계: 생성된 파이프라인으로 학습
from churn_pipeline import MLPipeline
import pandas as pd

X_train = pd.read_csv('data/train.csv')
y_train = X_train['churn']
X_train = X_train.drop('churn', axis=1)

pipeline = MLPipeline()
pipeline.fit(X_train, y_train)
pipeline.save('models/churn_model.pkl')

# 3단계: MLflow로 추적
from python.ml.mlops.mlflow_tracker import MLflowTracker

tracker = MLflowTracker(experiment_name='churn_prediction')
with tracker.start_run():
    tracker.log_params({'model_type': 'RandomForest'})
    tracker.log_metrics({'accuracy': 0.92})
    tracker.log_model(pipeline.model, 'churn_model')

# 4단계: 프로덕션에 배포
from python.ml.deployment.model_server import ModelServer

server = ModelServer(port=8000)
server.register_model(
    model_name='churn_predictor',
    model_path='models/churn_model.pkl',
    model_type='classifier'
)
server.start()

# 5단계: 모니터링 설정
from python.ml.mlops.model_monitor import ModelMonitor

monitor = ModelMonitor(
    model_name='churn_predictor',
    monitoring_window=10000,
    drift_threshold=0.1
)

# 프로덕션에서 예측 로깅
for customer in production_data:
    prediction = pipeline.predict([customer])
    monitor.log_prediction(
        input_data=customer,
        prediction=prediction,
        latency_ms=25
    )

# 드리프트 확인
drift_report = monitor.check_drift(X_train, production_data)
if drift_report['drift_detected']:
    print("⚠️ 데이터 드리프트 감지! 재학습 필요")
```

---

## 📖 관련 문서

- **PHASE_10_USAGE_GUIDE_KR.md**: MLOps 도구 사용 가이드
- **SYSTEM_OVERVIEW_KR.md**: 전체 시스템 개요
- **API_REFERENCE_KR.md**: API 레퍼런스
- **TECHNICAL_SUMMARY_KR.md**: 기술 요약

---

*작성일: 2025년 10월 1일*
*버전: 1.0.0*
*ML MCP System v2.0.0의 일부*
