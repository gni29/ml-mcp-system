# ML MCP 시스템 - 사용 예제 및 워크플로우

## 📖 개요

이 문서는 ML MCP 시스템의 포괄적인 예제와 실제 워크플로우를 제공합니다. 각 예제는 샘플 데이터, 단계별 지침 및 예상 출력을 포함합니다.

## 🏁 빠른 시작 가이드

### 필수 조건
```bash
# 1. 의존성 설치
npm install
pip install -r python/requirements.txt

# 2. MCP 서버 시작
npm run mcp:analysis
npm run mcp:ml
npm run mcp:visualization
```

### 첫 번째 분석 (5분)
```bash
# 1. 샘플 데이터 사용
cp data/sample_data.csv my_data.csv

# 2. 기본 통계
echo '{"data_file": "my_data.csv"}' | python python/analyzers/basic/descriptive_stats.py

# 3. 시각화 생성
echo '{"data_file": "my_data.csv", "columns": ["age", "income"]}' | python python/visualization/2d/scatter.py
```

## 📊 실제 워크플로우

### 1. 고객 데이터 분석 워크플로우

**시나리오**: 고객 인구통계 및 구매 행동 분석

**샘플 데이터 구조**:
```csv
customer_id,age,income,education,spending_score,region
1001,25,35000,Bachelor,85,North
1002,45,75000,Master,45,South
1003,35,50000,Bachelor,75,East
...
```

#### 단계 1: 초기 데이터 탐색
```bash
# 기본 통계
echo '{
  "data_file": "customer_data.csv",
  "output_dir": "customer_analysis/step1_basics"
}' | python python/analyzers/basic/descriptive_stats.py
```

**예상 출력**:
```json
{
  "success": true,
  "statistics": {
    "age": {"mean": 38.5, "std": 12.3, "min": 18, "max": 70},
    "income": {"mean": 52500, "std": 18200, "min": 20000, "max": 150000},
    "spending_score": {"mean": 60.2, "std": 25.8, "min": 1, "max": 99}
  }
}
```

#### 단계 2: 결측 데이터 분석
```bash
echo '{
  "data_file": "customer_data.csv",
  "output_dir": "customer_analysis/step2_missing"
}' | python python/analyzers/basic/missing_data.py
```

#### 단계 3: 상관관계 분석
```bash
echo '{
  "data_file": "customer_data.csv",
  "columns": ["age", "income", "spending_score"],
  "output_dir": "customer_analysis/step3_correlation"
}' | python python/analyzers/basic/correlation.py
```

#### 단계 4: 고객 세분화 (클러스터링)
```bash
echo '{
  "data_file": "customer_data.csv",
  "feature_columns": ["age", "income", "spending_score"],
  "algorithms": ["kmeans", "hierarchical"],
  "n_clusters": 4,
  "output_dir": "customer_analysis/step4_segmentation"
}' | python python/ml/unsupervised/clustering.py
```

#### 단계 5: 시각화 대시보드
```bash
# 산점도 매트릭스
echo '{
  "data_file": "customer_data.csv",
  "x_column": "age",
  "y_column": "income",
  "color_column": "region",
  "size_column": "spending_score",
  "plot_types": ["2d", "matrix", "correlations"],
  "output_dir": "customer_analysis/step5_visuals"
}' | python python/visualizations/scatter_plots.py

# 인터랙티브 대시보드
echo '{
  "data_file": "customer_data.csv",
  "numeric_columns": ["age", "income", "spending_score"],
  "categorical_columns": ["education", "region"],
  "plot_types": ["plotly_dashboard", "plotly_3d"],
  "output_dir": "customer_analysis/step5_visuals"
}' | python python/visualizations/interactive_plots.py
```

### 2. 매출 예측 워크플로우

**시나리오**: 과거 데이터를 기반으로 월별 매출 예측

**샘플 데이터 구조**:
```csv
date,sales,marketing_spend,season,promotions,competitor_price
2023-01-01,125000,15000,Winter,2,99.99
2023-02-01,135000,18000,Winter,1,95.99
2023-03-01,142000,16000,Spring,3,98.99
...
```

#### 완전한 워크플로우 스크립트:
```bash
#!/bin/bash
# 매출 예측 완전 워크플로우

ANALYSIS_DIR="sales_forecast_$(date +%Y%m%d_%H%M%S)"
DATA_FILE="sales_data.csv"

echo "🚀 매출 예측 분석 시작..."

# 단계 1: 데이터 품질 검사
echo "📊 단계 1: 데이터 품질 분석..."
echo '{
  "data_file": "'$DATA_FILE'",
  "output_dir": "'$ANALYSIS_DIR'/01_data_quality"
}' | python python/analyzers/basic/missing_data.py

# 단계 2: 시계열 분석
echo "📈 단계 2: 시계열 시각화..."
echo '{
  "data_file": "'$DATA_FILE'",
  "date_column": "date",
  "value_columns": ["sales", "marketing_spend"],
  "plot_types": ["line", "seasonal_decompose", "rolling_stats"],
  "output_dir": "'$ANALYSIS_DIR'/02_timeseries"
}' | python python/visualizations/time_series_plots.py

# 단계 3: 특성 공학
echo "🔧 단계 3: 특성 공학..."
echo '{
  "data_file": "'$DATA_FILE'",
  "target_column": "sales",
  "transformations": ["log", "polynomial"],
  "output_dir": "'$ANALYSIS_DIR'/03_features"
}' | python python/ml/preprocessing/feature_engineering.py

# 단계 4: 예측 모델
echo "🔮 단계 4: 예측 모델 구축..."
echo '{
  "data_file": "'$ANALYSIS_DIR'/03_features/engineered_data.csv",
  "date_column": "date",
  "value_column": "sales",
  "forecast_periods": 12,
  "models": ["arima", "exponential_smoothing", "linear_trend"],
  "output_dir": "'$ANALYSIS_DIR'/04_forecasting"
}' | python python/ml/time_series/forecasting.py

# 단계 5: 모델 평가
echo "📊 단계 5: 모델 평가..."
echo '{
  "model_files": ["'$ANALYSIS_DIR'/04_forecasting/arima_model.joblib",
                  "'$ANALYSIS_DIR'/04_forecasting/exp_smoothing_model.joblib"],
  "test_data_file": "'$DATA_FILE'",
  "target_column": "sales",
  "output_dir": "'$ANALYSIS_DIR'/05_evaluation"
}' | python python/ml/evaluation/model_evaluation.py

echo "✅ 분석 완료! 결과 위치: $ANALYSIS_DIR"
```

### 3. 의료 진단 분류

**시나리오**: 의료 진단을 위한 분류 모델 구축

**샘플 데이터 구조**:
```csv
patient_id,age,gender,symptom1,symptom2,test_result1,test_result2,diagnosis
P001,45,M,1,0,15.2,Normal,Disease_A
P002,32,F,0,1,12.8,Abnormal,Healthy
P003,67,M,1,1,18.5,Abnormal,Disease_B
...
```

#### 고급 분류 워크플로우:
```bash
# 1. 포괄적 데이터 분석
echo '{
  "data_file": "medical_data.csv",
  "output_dir": "medical_analysis/exploratory"
}' | python python/analyzers/basic/descriptive_stats.py

# 2. 진단별 분포 분석
echo '{
  "data_file": "medical_data.csv",
  "categorical_columns": ["diagnosis"],
  "numeric_columns": ["age", "test_result1"],
  "plot_types": ["box", "violin", "strip"],
  "output_dir": "medical_analysis/distributions"
}' | python python/visualizations/categorical_plots.py

# 3. 특성 선택 및 공학
echo '{
  "data_file": "medical_data.csv",
  "target_column": "diagnosis",
  "techniques": ["selection", "scaling", "interaction"],
  "selection_method": "recursive",
  "output_dir": "medical_analysis/features"
}' | python python/ml/preprocessing/advanced_feature_engineering.py

# 4. 다중 알고리즘 분류
echo '{
  "data_file": "medical_analysis/features/engineered_data.csv",
  "target_column": "diagnosis",
  "algorithms": ["logistic", "random_forest", "svm", "gradient_boosting"],
  "cross_validation": 10,
  "test_size": 0.2,
  "output_dir": "medical_analysis/models"
}' | python python/ml/supervised/classification/classification_trainer.py

# 5. 통계적 검증
echo '{
  "data_file": "medical_analysis/features/engineered_data.csv",
  "numeric_columns": ["age", "test_result1"],
  "target_column": "diagnosis",
  "plot_types": ["distribution", "qq", "confidence"],
  "output_dir": "medical_analysis/statistics"
}' | python python/visualizations/statistical_plots.py
```

### 4. 금융 리스크 평가

**시나리오**: 고객 금융 데이터를 사용한 대출 채무불이행 위험 평가

**샘플 데이터 구조**:
```csv
loan_id,age,income,credit_score,loan_amount,employment_years,debt_ratio,default
L001,28,45000,650,25000,3,0.35,0
L002,45,85000,750,50000,12,0.25,0
L003,35,35000,580,30000,2,0.65,1
...
```

#### 리스크 평가 파이프라인:
```python
# risk_assessment_pipeline.py
import subprocess
import json
from datetime import datetime

def run_analysis(params, script_path):
    """분석 도구 실행을 위한 헬퍼 함수"""
    process = subprocess.run(
        ['python', script_path],
        input=json.dumps(params),
        text=True,
        capture_output=True
    )
    return json.loads(process.stdout)

def financial_risk_pipeline(data_file):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = f"risk_assessment_{timestamp}"

    # 단계 1: 이상치 탐지 (사기 지표)
    print("🔍 이상치 및 이상 현상 탐지...")
    outlier_params = {
        "data_file": data_file,
        "methods": ["iqr", "isolation_forest", "lof"],
        "contamination": 0.05,
        "output_dir": f"{output_base}/01_outliers"
    }
    outliers = run_analysis(outlier_params, "python/analyzers/advanced/outlier_detection.py")

    # 단계 2: 상관관계 분석
    print("📊 특성 상관관계 분석...")
    corr_params = {
        "data_file": data_file,
        "method": "spearman",  # 금융 데이터에 더 적합
        "min_correlation": 0.3,
        "output_dir": f"{output_base}/02_correlations"
    }
    correlations = run_analysis(corr_params, "python/analyzers/basic/correlation.py")

    # 단계 3: 리스크 팩터 공학
    print("🔧 리스크 팩터 공학...")
    feature_params = {
        "data_file": data_file,
        "target_column": "default",
        "transformations": ["log", "interaction"],
        "output_dir": f"{output_base}/03_features"
    }
    features = run_analysis(feature_params, "python/ml/preprocessing/feature_engineering.py")

    # 단계 4: 불균형 데이터 처리와 모델 훈련
    print("🤖 리스크 모델 훈련...")
    model_params = {
        "data_file": f"{output_base}/03_features/engineered_data.csv",
        "target_column": "default",
        "algorithms": ["logistic", "random_forest", "gradient_boosting"],
        "cross_validation": 5,
        "class_weight": "balanced",  # 불균형 클래스 처리
        "output_dir": f"{output_base}/04_models"
    }
    models = run_analysis(model_params, "python/ml/supervised/classification/classification_trainer.py")

    # 단계 5: 리스크 시각화
    print("📈 리스크 시각화 생성...")
    viz_params = {
        "data_file": data_file,
        "numeric_columns": ["age", "income", "credit_score", "debt_ratio"],
        "categorical_columns": ["default"],
        "plot_types": ["plotly_scatter", "plotly_heatmap", "plotly_dashboard"],
        "output_dir": f"{output_base}/05_visualizations"
    }
    visualizations = run_analysis(viz_params, "python/visualizations/interactive_plots.py")

    print(f"✅ 리스크 평가 완료! 결과 위치: {output_base}")
    return {
        "analysis_id": output_base,
        "outliers_detected": len(outliers.get("consensus_outliers", [])),
        "best_model": models.get("best_model", {}),
        "key_risk_factors": correlations.get("significant_correlations", [])[:5]
    }

# 사용법
if __name__ == "__main__":
    result = financial_risk_pipeline("loan_data.csv")
    print(json.dumps(result, indent=2))
```

## 🎯 특화 사용 사례

### A. 부동산 가격 예측

```bash
# 완전한 부동산 분석
echo '{
  "data_file": "real_estate.csv",
  "target_column": "price",
  "feature_columns": ["sqft", "bedrooms", "bathrooms", "age", "location_score"],
  "algorithms": ["linear", "ridge", "random_forest"],
  "output_dir": "real_estate_analysis"
}' | python python/ml/supervised/regression/regression_trainer.py
```

### B. 마케팅 캠페인 최적화

```bash
# A/B 테스트 분석
echo '{
  "data_file": "campaign_data.csv",
  "categorical_columns": ["campaign_type", "customer_segment"],
  "numeric_columns": ["conversion_rate", "cost_per_click", "revenue"],
  "plot_types": ["bar", "box", "heatmap"],
  "output_dir": "campaign_analysis"
}' | python python/visualizations/categorical_plots.py
```

### C. 품질 관리 분석

```bash
# 제조 품질 분석
echo '{
  "data_file": "production_data.csv",
  "columns": ["temperature", "pressure", "speed", "quality_score"],
  "methods": ["iqr", "zscore"],
  "output_dir": "quality_control"
}' | python python/analyzers/advanced/outlier_detection.py
```

## 📋 모범 사례 및 팁

### 1. 데이터 준비 체크리스트
```bash
# 분석 전에 항상 확인:
echo '{
  "data_file": "your_data.csv"
}' | python python/analyzers/basic/missing_data.py

# 정리 및 검증
echo '{
  "data_file": "your_data.csv",
  "strategy": "analyze"
}' | python python/analyzers/basic/missing_data.py
```

### 2. 점진적 분석 전략

**1단계: 탐색** (5-10분)
- 기본 통계
- 결측 데이터 분석
- 분포 플롯

**2단계: 조사** (15-30분)
- 상관관계 분석
- 이상치 탐지
- 범주형 분석

**3단계: 모델링** (30-60분)
- 특성 공학
- 모델 훈련
- 모델 평가

**4단계: 검증** (15-30분)
- 통계적 테스트
- 잔차 분석
- 교차 검증

### 3. 출력 조직화

```
project_analysis_YYYYMMDD_HHMMSS/
├── 01_exploration/
│   ├── basic_stats.json
│   ├── distributions.png
│   └── missing_data.json
├── 02_investigation/
│   ├── correlations.png
│   ├── outliers.json
│   └── categorical_analysis.png
├── 03_modeling/
│   ├── features/
│   ├── models/
│   └── evaluation/
├── 04_validation/
│   ├── statistical_tests.json
│   └── residual_plots.png
└── final_report.html
```

### 4. 산업별 일반적인 워크플로우

#### 의료 분석
```bash
# 환자 결과 예측
1. missing_data.py → 환자 기록 정리
2. correlation.py → 증상 관계 찾기
3. classification.py → 결과 예측
4. statistical_plots.py → 결과 검증
```

#### 금융 서비스
```bash
# 신용 리스크 평가
1. outlier_detection.py → 사기 탐지
2. feature_engineering.py → 리스크 팩터
3. classification.py → 채무불이행 예측
4. interactive_plots.py → 리스크 대시보드
```

#### 리테일 분석
```bash
# 고객 세분화
1. descriptive_stats.py → 고객 인구통계
2. clustering.py → 고객 세분화
3. scatter_plots.py → 세그먼트 시각화
4. forecasting.py → 수요 예측
```

#### 제조업
```bash
# 품질 관리
1. distribution.py → 프로세스 안정성
2. outlier_detection.py → 결함 탐지
3. time_series_plots.py → 트렌드 분석
4. regression.py → 품질 예측
```

## 🔧 통합 예제

### Jupyter Notebook과 함께
```python
import subprocess
import json
import pandas as pd

def run_mcp_tool(tool_script, params):
    """Jupyter에서 MCP 도구 실행"""
    result = subprocess.run(
        ['python', tool_script],
        input=json.dumps(params),
        text=True,
        capture_output=True
    )
    return json.loads(result.stdout)

# 사용 예제
params = {
    "data_file": "data.csv",
    "columns": ["feature1", "feature2"]
}
result = run_mcp_tool("python/analyzers/basic/correlation.py", params)
print(result)
```

### R 통합
```r
# MCP 도구용 R 래퍼
library(jsonlite)

run_mcp_analysis <- function(script_path, params) {
  params_json <- toJSON(params, auto_unbox = TRUE)
  result <- system2("python",
                   args = script_path,
                   input = params_json,
                   stdout = TRUE)
  fromJSON(result)
}

# 사용법
params <- list(data_file = "data.csv", columns = c("x", "y"))
result <- run_mcp_analysis("python/analyzers/basic/correlation.py", params)
```

### 웹 애플리케이션과 함께
```javascript
// Node.js 웹 서비스 통합
const { spawn } = require('child_process');

async function runMCPAnalysis(scriptPath, params) {
  return new Promise((resolve, reject) => {
    const python = spawn('python', [scriptPath]);

    python.stdin.write(JSON.stringify(params));
    python.stdin.end();

    let result = '';
    python.stdout.on('data', (data) => {
      result += data;
    });

    python.on('close', (code) => {
      if (code === 0) {
        resolve(JSON.parse(result));
      } else {
        reject(new Error('분석 실패'));
      }
    });
  });
}

// Express.js에서 사용
app.post('/api/analyze', async (req, res) => {
  try {
    const result = await runMCPAnalysis(
      'python/analyzers/basic/descriptive_stats.py',
      req.body
    );
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
```

## 🚨 문제 해결 가이드

### 일반적인 문제 및 해결책

**1. "모듈을 찾을 수 없음" 오류**
```bash
# 해결책: Python 경로 확인
export PYTHONPATH="${PYTHONPATH}:$(pwd)/python"
pip install -r python/requirements.txt
```

**2. "파일을 찾을 수 없음" 오류**
```bash
# 해결책: 절대 경로 사용
echo '{
  "data_file": "/full/path/to/data.csv"
}' | python python/analyzers/basic/descriptive_stats.py
```

**3. 대용량 데이터셋의 메모리 오류**
```bash
# 해결책: 샘플링 사용
echo '{
  "data_file": "large_data.csv",
  "sample_size": 10000
}' | python python/analyzers/basic/descriptive_stats.py
```

**4. 권한 오류**
```bash
# 해결책: 디렉토리 권한 확인
chmod 755 output_directory
mkdir -p output_directory
```

## 📈 성능 최적화

### 대용량 데이터셋의 경우
1. **청킹 사용**: 더 작은 배치로 데이터 처리
2. **먼저 샘플링**: 탐색을 위해 대표 샘플 사용
3. **메모리 최적화**: 파일을 닫고 변수를 정리
4. **병렬 처리**: 가능한 경우 다중 코어 사용

### 프로덕션 사용
1. **결과 캐시**: 중간 결과 저장
2. **입력 검증**: 처리 전 데이터 품질 확인
3. **리소스 모니터링**: 메모리 및 CPU 사용량 추적
4. **로그 작업**: 디버깅을 위한 상세 로그 유지

---

*더 많은 예제와 고급 사용 패턴은 [개발자 가이드](DEVELOPER_GUIDE_KR.md) 및 [API 레퍼런스](API_REFERENCE_KR.md)를 참조하세요.*