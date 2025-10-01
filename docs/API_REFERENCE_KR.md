# ML MCP 시스템 - API 레퍼런스

## 📖 개요

ML MCP 시스템은 포괄적인 데이터 분석, 머신러닝, 시각화를 위한 21개의 강력한 도구를 3개 모듈에 걸쳐 제공합니다. 이 문서는 모든 사용 가능한 도구의 완전한 API 명세를 제공합니다.

## 🏗️ 시스템 아키텍처

```
ML MCP 시스템
├── ml-mcp-analysis/     # 통계 분석 (5개 도구)
├── ml-mcp-ml/          # 머신러닝 (8개 도구)
├── ml-mcp-visualization/ # 데이터 시각화 (8개 도구)
└── ml-mcp-shared/      # 공통 유틸리티
```

## 📊 모듈: 분석 도구

### 1. 기본 통계 (`basic_stats`)

**설명**: 포괄적인 기술통계 분석

**매개변수**:
```json
{
  "data_file": "string (필수) - 데이터 파일 경로 (CSV/Excel)",
  "columns": "array[string] (선택) - 분석할 특정 컬럼들",
  "output_dir": "string (선택, 기본값: 'results') - 출력 디렉토리"
}
```

**지원 입력 형식**: CSV, Excel (.xlsx, .xls)

**출력**:
```json
{
  "success": true,
  "statistics": {
    "컬럼명": {
      "count": "number - 유효값 개수",
      "mean": "number - 산술평균",
      "std": "number - 표준편차",
      "min": "number - 최솟값",
      "25%": "number - 25백분위수",
      "50%": "number - 중위수",
      "75%": "number - 75백분위수",
      "max": "number - 최댓값",
      "skewness": "number - 비대칭도",
      "kurtosis": "number - 첨도"
    }
  },
  "data_info": {
    "shape": "[행수, 열수]",
    "memory_usage": "string - 메모리 사용량 정보",
    "dtypes": "object - 컬럼별 데이터 타입"
  }
}
```

**오류 코드**:
- `FILE_NOT_FOUND`: 데이터 파일이 존재하지 않음
- `INVALID_FORMAT`: 지원하지 않는 파일 형식
- `NO_NUMERIC_COLUMNS`: 분석할 수치형 컬럼이 없음

### 2. 상관관계 분석 (`correlation`)

**설명**: Pearson, Spearman, Kendall 상관관계 분석

**매개변수**:
```json
{
  "data_file": "string (필수)",
  "columns": "array[string] (선택) - 상관분석할 컬럼들",
  "method": "string (선택, 기본값: 'pearson') - pearson|spearman|kendall",
  "min_correlation": "number (선택, 기본값: 0.0) - 최소 상관계수 임계값",
  "output_dir": "string (선택)"
}
```

**출력**:
```json
{
  "success": true,
  "correlation_matrix": "object - 전체 상관관계 행렬",
  "significant_correlations": [
    {
      "variable1": "string",
      "variable2": "string",
      "correlation": "number",
      "p_value": "number",
      "interpretation": "string - 약함|보통|강함"
    }
  ],
  "method_used": "string",
  "generated_files": ["correlation_heatmap.png"]
}
```

### 3. 분포 분석 (`distribution`)

**설명**: 통계적 분포 분석 및 시각화

**매개변수**:
```json
{
  "data_file": "string (필수)",
  "columns": "array[string] (선택)",
  "plot_types": "array[string] (선택) - histogram|kde|qq|box",
  "bins": "number (선택, 기본값: 30) - 히스토그램 구간 수",
  "output_dir": "string (선택)"
}
```

**출력**:
```json
{
  "success": true,
  "distributions": {
    "컬럼명": {
      "normality_test": {
        "shapiro_wilk": {"statistic": "number", "p_value": "number"},
        "jarque_bera": {"statistic": "number", "p_value": "number"}
      },
      "distribution_params": {
        "mean": "number",
        "std": "number",
        "skewness": "number",
        "kurtosis": "number"
      }
    }
  },
  "generated_files": ["distribution_plots.png"]
}
```

### 4. 결측 데이터 분석 (`missing_data`)

**설명**: 포괄적인 결측 데이터 패턴 분석

**매개변수**:
```json
{
  "data_file": "string (필수)",
  "strategy": "string (선택, 기본값: 'analyze') - analyze|impute",
  "impute_method": "string (선택) - mean|median|mode|forward_fill|backward_fill",
  "output_dir": "string (선택)"
}
```

**출력**:
```json
{
  "success": true,
  "missing_summary": {
    "total_missing": "number",
    "missing_percentage": "number",
    "columns_with_missing": "array[string]"
  },
  "missing_patterns": [
    {
      "pattern": "string - 결측값의 이진 패턴",
      "count": "number - 이 패턴을 가진 행 수",
      "percentage": "number"
    }
  ],
  "recommendations": {
    "action": "string - 전략 권장사항",
    "reasoning": "string - 이 전략을 권장하는 이유"
  }
}
```

### 5. 이상치 탐지 (`outlier_detection`)

**설명**: 다중 방법 이상치 탐지 및 분석

**매개변수**:
```json
{
  "data_file": "string (필수)",
  "columns": "array[string] (선택)",
  "methods": "array[string] (선택) - iqr|zscore|isolation_forest|lof",
  "contamination": "number (선택, 기본값: 0.1) - 예상 이상치 비율",
  "output_dir": "string (선택)"
}
```

**출력**:
```json
{
  "success": true,
  "outliers_detected": {
    "방법명": {
      "outlier_indices": "array[number] - 이상치 행 인덱스",
      "outlier_count": "number",
      "outlier_percentage": "number",
      "threshold_values": "object - 방법별 임계값"
    }
  },
  "consensus_outliers": "array[number] - 여러 방법에서 탐지된 이상치",
  "generated_files": ["outlier_analysis.png"]
}
```

## 🤖 모듈: 머신러닝 도구

### 1. 특성 공학 (`feature_engineering`)

**설명**: 자동화된 특성 공학 및 변환

**매개변수**:
```json
{
  "data_file": "string (필수)",
  "target_column": "string (선택) - 지도학습을 위한 목표 변수",
  "feature_types": "object (선택) - 컬럼 타입 지정",
  "transformations": "array[string] (선택) - log|sqrt|polynomial|interaction",
  "output_dir": "string (선택)"
}
```

**출력**:
```json
{
  "success": true,
  "engineered_features": {
    "new_features": "array[string] - 생성된 특성명",
    "transformation_log": "array[object] - 적용된 변환 기록",
    "feature_importance": "object - 특성 중요도 점수"
  },
  "data_shape": {
    "original": "[행수, 열수]",
    "engineered": "[행수, 열수]"
  },
  "saved_artifacts": {
    "engineered_data": "string - 파일 경로",
    "transformer_pipeline": "string - Sklearn 파이프라인 파일"
  }
}
```

### 2. 분류 훈련 (`classification`)

**설명**: 다중 알고리즘 분류 모델 훈련 및 평가

**매개변수**:
```json
{
  "data_file": "string (필수)",
  "target_column": "string (필수) - 분류 목표값",
  "feature_columns": "array[string] (선택) - 사용할 특성들",
  "algorithms": "array[string] (선택) - logistic|random_forest|svm|gradient_boosting",
  "test_size": "number (선택, 기본값: 0.2) - 훈련/테스트 분할 비율",
  "cross_validation": "number (선택, 기본값: 5) - 교차검증 폴드 수",
  "output_dir": "string (선택)"
}
```

**출력**:
```json
{
  "success": true,
  "models_trained": {
    "알고리즘명": {
      "accuracy": "number",
      "precision": "number",
      "recall": "number",
      "f1_score": "number",
      "roc_auc": "number",
      "confusion_matrix": "array[array[number]]",
      "feature_importance": "object",
      "model_file": "string - 저장된 모델 경로"
    }
  },
  "best_model": {
    "algorithm": "string",
    "score": "number",
    "hyperparameters": "object"
  }
}
```

### 3. 회귀 훈련 (`regression`)

**설명**: 다중 알고리즘 회귀 모델 훈련 및 평가

**매개변수**:
```json
{
  "data_file": "string (필수)",
  "target_column": "string (필수) - 회귀 목표값",
  "feature_columns": "array[string] (선택)",
  "algorithms": "array[string] (선택) - linear|ridge|lasso|random_forest|gradient_boosting",
  "test_size": "number (선택, 기본값: 0.2)",
  "cross_validation": "number (선택, 기본값: 5)",
  "output_dir": "string (선택)"
}
```

**출력**:
```json
{
  "success": true,
  "models_trained": {
    "알고리즘명": {
      "r2_score": "number - R-제곱 계수",
      "mean_squared_error": "number",
      "mean_absolute_error": "number",
      "root_mean_squared_error": "number",
      "feature_importance": "object",
      "model_file": "string"
    }
  },
  "residual_analysis": {
    "residual_std": "number",
    "residual_mean": "number",
    "normality_test": "object"
  }
}
```

### 4. 클러스터링 분석 (`clustering`)

**설명**: 비지도 클러스터링과 다중 알고리즘

**매개변수**:
```json
{
  "data_file": "string (필수)",
  "feature_columns": "array[string] (선택)",
  "algorithms": "array[string] (선택) - kmeans|hierarchical|dbscan|gaussian_mixture",
  "n_clusters": "number (선택, 기본값: 3) - 해당 알고리즘용",
  "output_dir": "string (선택)"
}
```

**출력**:
```json
{
  "success": true,
  "clustering_results": {
    "알고리즘명": {
      "cluster_labels": "array[number] - 클러스터 할당",
      "n_clusters_found": "number",
      "silhouette_score": "number",
      "calinski_harabasz_score": "number",
      "davies_bouldin_score": "number"
    }
  },
  "cluster_analysis": {
    "cluster_centers": "array[array[number]]",
    "cluster_sizes": "array[number]",
    "optimal_clusters": "number - 권장 클러스터 수"
  }
}
```

### 5. 시계열 예측 (`forecasting`)

**설명**: 시계열 분석 및 예측

**매개변수**:
```json
{
  "data_file": "string (필수)",
  "date_column": "string (필수) - 날짜/시간 컬럼",
  "value_column": "string (필수) - 예측할 값",
  "forecast_periods": "number (선택, 기본값: 30) - 예측 기간",
  "models": "array[string] (선택) - arima|exponential_smoothing|linear_trend",
  "seasonal": "boolean (선택, 기본값: true) - 계절성 포함",
  "output_dir": "string (선택)"
}
```

**출력**:
```json
{
  "success": true,
  "forecasting_results": {
    "모델명": {
      "forecast_values": "array[number]",
      "confidence_intervals": {
        "lower": "array[number]",
        "upper": "array[number]"
      },
      "model_metrics": {
        "aic": "number",
        "bic": "number",
        "mape": "number",
        "rmse": "number"
      }
    }
  },
  "time_series_analysis": {
    "trend": "string - 증가|감소|안정",
    "seasonality_detected": "boolean",
    "seasonal_period": "number"
  }
}
```

### 6. PCA 분석 (`pca`)

**설명**: 주성분 분석을 통한 차원 축소

**매개변수**:
```json
{
  "data_file": "string (필수)",
  "feature_columns": "array[string] (선택)",
  "n_components": "number (선택) - 컴포넌트 수",
  "variance_threshold": "number (선택, 기본값: 0.95) - 유지할 분산 비율",
  "output_dir": "string (선택)"
}
```

**출력**:
```json
{
  "success": true,
  "pca_results": {
    "n_components": "number - 선택된 컴포넌트 수",
    "explained_variance_ratio": "array[number] - 컴포넌트별 분산 비율",
    "cumulative_variance": "array[number]",
    "component_loadings": "array[array[number]]",
    "transformed_data_file": "string"
  },
  "dimensionality_reduction": {
    "original_dimensions": "number",
    "reduced_dimensions": "number",
    "variance_retained": "number"
  }
}
```

### 7. 고급 특성 공학 (`advanced_feature_engineering`)

**설명**: 정교한 특성 공학 기법

**매개변수**:
```json
{
  "data_file": "string (필수)",
  "target_column": "string (선택)",
  "techniques": "array[string] (선택) - polynomial|interaction|selection|scaling",
  "polynomial_degree": "number (선택, 기본값: 2)",
  "selection_method": "string (선택) - univariate|recursive|lasso",
  "output_dir": "string (선택)"
}
```

### 8. 모델 평가 (`model_evaluation`)

**설명**: 포괄적인 모델 평가 및 비교

**매개변수**:
```json
{
  "model_files": "array[string] (필수) - 저장된 모델 경로들",
  "test_data_file": "string (필수)",
  "target_column": "string (필수)",
  "evaluation_metrics": "array[string] (선택) - 커스텀 메트릭",
  "output_dir": "string (선택)"
}
```

## 📊 모듈: 시각화 도구

### 1. 산점도 (`scatter_plots`)

**설명**: 다차원 산점도 분석

**매개변수**:
```json
{
  "data_file": "string (필수)",
  "x_column": "string (필수) - X축 변수",
  "y_column": "string (필수) - Y축 변수",
  "color_column": "string (선택) - 범주형 색상 구분",
  "size_column": "string (선택) - 점 크기 변수",
  "plot_types": "array[string] (선택) - 2d|3d|matrix|outliers|correlations",
  "output_dir": "string (선택)"
}
```

**출력**:
```json
{
  "success": true,
  "generated_files": "array[string] - 생성된 플롯 파일들",
  "scatter_analysis": {
    "correlation_coefficient": "number",
    "trend_line": {
      "slope": "number",
      "intercept": "number",
      "r_squared": "number"
    },
    "outliers_detected": "array[number] - 이상치 인덱스"
  }
}
```

### 2. 시계열 플롯 (`time_series_plots`)

**설명**: 포괄적인 시계열 시각화

**매개변수**:
```json
{
  "data_file": "string (필수)",
  "date_column": "string (필수)",
  "value_columns": "array[string] (필수)",
  "plot_types": "array[string] (선택) - line|area|seasonal_decompose|rolling_stats|autocorrelation",
  "rolling_window": "number (선택, 기본값: 30)",
  "output_dir": "string (선택)"
}
```

### 3. 범주형 플롯 (`categorical_plots`)

**설명**: 범주형 데이터 시각화 스위트

**매개변수**:
```json
{
  "data_file": "string (필수)",
  "categorical_columns": "array[string] (필수)",
  "numeric_columns": "array[string] (선택)",
  "plot_types": "array[string] (선택) - bar|pie|box|violin|heatmap|strip",
  "output_dir": "string (선택)"
}
```

### 4. 통계적 플롯 (`statistical_plots`)

**설명**: 통계 분석 시각화

**매개변수**:
```json
{
  "data_file": "string (필수)",
  "numeric_columns": "array[string] (필수)",
  "target_column": "string (선택) - 회귀 분석용",
  "plot_types": "array[string] (선택) - distribution|qq|residual|probability|confidence",
  "output_dir": "string (선택)"
}
```

### 5. 인터랙티브 플롯 (`interactive_plots`)

**설명**: 웹 기반 인터랙티브 시각화

**매개변수**:
```json
{
  "data_file": "string (필수)",
  "numeric_columns": "array[string] (선택)",
  "categorical_columns": "array[string] (선택)",
  "plot_types": "array[string] (선택) - plotly_scatter|plotly_timeseries|plotly_3d|plotly_heatmap|bokeh_scatter",
  "output_dir": "string (선택)"
}
```

**출력**: 웹 보기용 JavaScript가 포함된 인터랙티브 HTML 파일

### 6. 분포 플롯 (`distribution_plots`)

**설명**: 통계적 분포 시각화

**매개변수**:
```json
{
  "data_file": "string (필수)",
  "columns": "array[string] (선택)",
  "plot_types": "array[string] (선택) - histogram|kde|qq|box|violin",
  "output_dir": "string (선택)"
}
```

### 7. 히트맵 (`heatmaps`)

**설명**: 상관관계 및 데이터 히트맵

**매개변수**:
```json
{
  "data_file": "string (필수)",
  "columns": "array[string] (선택)",
  "correlation_method": "string (선택, 기본값: 'pearson')",
  "annot": "boolean (선택, 기본값: true)",
  "output_dir": "string (선택)"
}
```

### 8. 고급 플롯 (`advanced_plots`)

**설명**: 복합 다중 패널 시각화

**매개변수**:
```json
{
  "data_file": "string (필수)",
  "analysis_type": "string (필수) - regression|classification|clustering|comparison",
  "columns": "array[string] (선택)",
  "output_dir": "string (선택)"
}
```

## 🔧 공통 매개변수

### 파일 형식 지원
- **CSV**: 다양한 구분자를 가진 `.csv` 파일
- **Excel**: `.xlsx`, `.xls` 파일 (모든 시트 지원)
- **Parquet**: `.parquet` 파일 (고성능)
- **JSON**: `.json` 파일 (구조화된 데이터)

### 출력 디렉토리 구조
```
output_dir/
├── data/           # 처리된 데이터셋
├── models/         # 훈련된 모델 파일
├── plots/          # 생성된 시각화
├── reports/        # 분석 보고서
└── metadata.json   # 실행 메타데이터
```

### 오류 처리

모든 도구는 표준화된 오류 응답을 반환합니다:
```json
{
  "success": false,
  "error": "string - 사용자 친화적 오류 메시지",
  "error_type": "string - 오류 카테고리",
  "error_code": "string - 특정 오류 코드",
  "suggestions": "array[string] - 잠재적 해결책"
}
```

### 성능 가이드라인

| 데이터셋 크기 | 메모리 요구사항 | 처리 시간 | 권장사항 |
|-------------|----------------|----------|---------|
| < 1MB | 50MB | < 1초 | 모든 도구 사용 가능 |
| 1-100MB | 500MB | 1-30초 | 탐색용 샘플링 사용 |
| 100MB-1GB | 2GB | 30초-5분 | 청킹 고려 |
| > 1GB | 8GB+ | 5분+ | 분산 처리 사용 |

## 🚨 속도 제한 및 리소스 관리

- **동시 요청**: 클라이언트당 최대 5개
- **메모리 제한**: 작업당 4GB
- **파일 크기 제한**: 파일당 2GB
- **처리 시간 제한**: 작업당 10분

## 📝 모범 사례

### 데이터 준비
1. **데이터 정리**: 결측값을 적절히 제거하거나 처리
2. **일관된 타입**: 컬럼 전체에서 일관된 데이터 타입 보장
3. **적절한 크기**: 탐색용으로는 초기 데이터셋을 100MB 이하로 유지
4. **데이터 백업**: 항상 원본 데이터 백업 유지

### 도구 선택
1. **간단하게 시작**: 기본 통계 및 분포 분석부터 시작
2. **반복적 접근**: 결과를 사용하여 다음 분석 단계 안내
3. **검증**: 여러 도구로 결과 교차 확인
4. **문서화**: 재현성을 위해 분석 매개변수 저장

### 출력 관리
1. **정리**: 설명적인 출력 디렉토리 이름 사용
2. **버전 관리**: 분석 실행에 타임스탬프 포함
3. **아카이브**: 완료된 분석을 향후 참조용으로 아카이브
4. **공유**: 협업을 위해 생성된 보고서 사용

---

*추가 지원, 예제 및 튜토리얼은 [사용 예제](USAGE_EXAMPLES_KR.md) 및 [개발자 가이드](DEVELOPER_GUIDE_KR.md)를 참조하세요.*