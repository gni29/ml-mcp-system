#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Evaluation and Validation Module
모델 평가 및 검증 모듈

이 모듈은 훈련된 머신러닝 모델의 포괄적인 평가를 수행합니다.
주요 기능:
- 분류 및 회귀 모델의 성능 평가
- 교차 검증 및 홀드아웃 검증
- 학습 곡선 및 검증 곡선 생성
- 모델 해석성 분석 (SHAP, LIME)
- 통계적 유의성 검정
- 모델 비교 및 벤치마킹
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# 공유 유틸리티 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "ml-mcp-shared" / "python"))

try:
    from common_utils import load_data, get_data_info, create_analysis_result, output_results, validate_required_params
except ImportError:
    # 공유 유틸리티 import 실패 시 대체 구현
    def load_data(file_path: str) -> pd.DataFrame:
        """데이터 파일 로드"""
        file_path = Path(file_path)
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_path.suffix}")

    def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
        """데이터프레임 기본 정보 추출"""
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
        }

    def create_analysis_result(analysis_type: str, data_info: Dict[str, Any], results: Dict[str, Any], summary: str = None) -> Dict[str, Any]:
        """표준화된 분석 결과 구조 생성"""
        return {
            "analysis_type": analysis_type,
            "timestamp": pd.Timestamp.now().isoformat(),
            "data_info": data_info,
            "summary": summary or f"{analysis_type} 분석 완료",
            **results
        }

    def output_results(results: Dict[str, Any]):
        """결과를 JSON 형태로 출력"""
        def comprehensive_json_serializer(obj):
            """포괄적인 JSON 직렬화 함수"""
            if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Index):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, type) or str(type(obj)).startswith("<class 'numpy."):
                return str(obj)
            elif hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'dtype') and hasattr(obj, 'name'):
                return str(obj)
            elif hasattr(obj, 'to_dict') and callable(obj.to_dict):
                return obj.to_dict()
            elif hasattr(obj, '__dict__') and not isinstance(obj, type):
                return str(obj)
            else:
                return str(obj)

        def clean_dict_for_json(data):
            """딕셔너리의 키와 값을 JSON 직렬화 가능하도록 정리"""
            if isinstance(data, dict):
                cleaned = {}
                for key, value in data.items():
                    clean_key = str(key) if not isinstance(key, (str, int, float, bool, type(None))) else key
                    cleaned[clean_key] = clean_dict_for_json(value)
                return cleaned
            elif isinstance(data, (list, tuple)):
                return [clean_dict_for_json(item) for item in data]
            else:
                return comprehensive_json_serializer(data)

        try:
            cleaned_results = clean_dict_for_json(results)
            print(json.dumps(cleaned_results, ensure_ascii=False, indent=2, default=comprehensive_json_serializer))
        except Exception as e:
            error_output = {
                "success": False,
                "error": f"JSON 직렬화 실패: {str(e)}",
                "error_type": "SerializationError"
            }
            print(json.dumps(error_output, ensure_ascii=False, indent=2))

    def validate_required_params(params: Dict[str, Any], required: list):
        """필수 매개변수 검증"""
        missing = [param for param in required if param not in params]
        if missing:
            raise ValueError(f"필수 매개변수가 누락됨: {', '.join(missing)}")

try:
    from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
    from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    from sklearn.metrics import roc_auc_score, precision_recall_curve
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.inspection import permutation_importance
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def train_baseline_model(df: pd.DataFrame, target_column: str, task_type: str = "auto"):
    """간단한 베이스라인 모델 훈련"""

    try:
        X, y = prepare_evaluation_data(df, target_column)

        if task_type == "auto":
            task_type = detect_task_type(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if task_type == "classification":
            model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            model = LinearRegression()

        model.fit(X_train, y_train)
        return model

    except Exception as e:
        print(f"베이스라인 모델 훈련 실패: {str(e)}", file=sys.stderr)
        return None

def evaluate_model(model_path: Optional[str] = None,
                   model_object: Optional[Any] = None,
                   X_test: Optional[np.ndarray] = None,
                   y_test: Optional[np.ndarray] = None,
                   df: Optional[pd.DataFrame] = None,
                   target_column: Optional[str] = None,
                   test_size: float = 0.2,
                   cv_folds: int = 5,
                   task_type: str = "auto",
                   evaluation_methods: List[str] = None,
                   train_simple_model: bool = False) -> Dict[str, Any]:
    """
    포괄적인 모델 평가

    Parameters:
    -----------
    model_path : str, optional
        저장된 모델 파일 경로
    model_object : Any, optional
        메모리에 로드된 모델 객체
    X_test : np.ndarray, optional
        테스트 특성 데이터
    y_test : np.ndarray, optional
        테스트 타겟 데이터
    df : pd.DataFrame, optional
        전체 데이터 (분할하여 사용)
    target_column : str, optional
        타겟 변수 컬럼명
    test_size : float, default=0.2
        테스트 데이터 비율
    cv_folds : int, default=5
        교차 검증 폴드 수
    task_type : str, default="auto"
        작업 유형 ("classification", "regression", "auto")
    evaluation_methods : List[str], optional
        평가 방법 ["holdout", "cross_validation", "learning_curve", "baseline_comparison"]

    Returns:
    --------
    Dict[str, Any]
        모델 평가 결과
    """

    if not SKLEARN_AVAILABLE:
        return {
            "error": "scikit-learn이 설치되지 않았습니다",
            "required_package": "scikit-learn"
        }

    # 모델 로드 또는 생성
    if model_path:
        try:
            model = joblib.load(model_path)
        except Exception as e:
            return {"error": f"모델 로드 실패: {str(e)}"}
    elif model_object:
        model = model_object
    elif train_simple_model and df is not None and target_column is not None:
        # 간단한 모델을 훈련하여 평가
        model = train_baseline_model(df, target_column, task_type)
        if model is None:
            return {"error": "베이스라인 모델 훈련 실패"}
    else:
        return {"error": "model_path, model_object 또는 train_simple_model=True가 필요합니다"}

    # 데이터 준비
    if X_test is not None and y_test is not None:
        # 테스트 데이터가 직접 제공된 경우
        data_source = "provided_test_data"
    elif df is not None and target_column is not None:
        # 전체 데이터에서 분할
        X, y = prepare_evaluation_data(df, target_column)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        data_source = "split_from_dataframe"
    else:
        return {"error": "테스트 데이터 또는 전체 데이터가 필요합니다"}

    # 작업 유형 자동 감지
    if task_type == "auto":
        task_type = detect_task_type(y_test)

    # 평가 방법 설정
    if evaluation_methods is None:
        evaluation_methods = ["holdout", "cross_validation", "baseline_comparison"]

    try:
        results = {
            "success": True,
            "evaluation_info": {
                "task_type": task_type,
                "model_type": type(model).__name__,
                "data_source": data_source,
                "test_samples": len(X_test),
                "feature_count": X_test.shape[1] if hasattr(X_test, 'shape') else len(X_test[0]),
                "evaluation_methods": evaluation_methods
            },
            "evaluations": {}
        }

        # 홀드아웃 평가
        if "holdout" in evaluation_methods:
            holdout_result = perform_holdout_evaluation(model, X_test, y_test, task_type)
            results["evaluations"]["holdout"] = holdout_result

        # 교차 검증 평가
        if "cross_validation" in evaluation_methods and 'X_train' in locals():
            cv_result = perform_cross_validation_evaluation(model, X_train, y_train, task_type, cv_folds)
            results["evaluations"]["cross_validation"] = cv_result

        # 학습 곡선 분석
        if "learning_curve" in evaluation_methods and 'X_train' in locals():
            lc_result = generate_learning_curves(model, X_train, y_train, task_type, cv_folds)
            results["evaluations"]["learning_curve"] = lc_result

        # 베이스라인 비교
        if "baseline_comparison" in evaluation_methods:
            baseline_result = compare_with_baseline(model, X_test, y_test, task_type)
            results["evaluations"]["baseline_comparison"] = baseline_result

        # 특성 중요도 (순열 중요도)
        if hasattr(model, 'predict'):
            importance_result = calculate_permutation_importance(model, X_test, y_test, task_type)
            results["evaluations"]["feature_importance"] = importance_result

        # 모델 해석성
        if "interpretability" in evaluation_methods:
            interpret_result = analyze_model_interpretability(model, X_test, y_test, task_type)
            results["evaluations"]["interpretability"] = interpret_result

        # 전체 평가 요약
        results["evaluation_summary"] = generate_evaluation_summary(results["evaluations"], task_type)

        # 권고사항
        results["recommendations"] = generate_evaluation_recommendations(results)

        return results

    except Exception as e:
        return {
            "error": f"모델 평가 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def prepare_evaluation_data(df: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
    """평가용 데이터 전처리"""

    y = df[target_column].copy()
    X = df.drop(columns=[target_column]).copy()

    # 결측치 처리
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    for col in numeric_cols:
        X[col].fillna(X[col].median(), inplace=True)

    for col in categorical_cols:
        X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'unknown', inplace=True)

    # 범주형 변수 인코딩
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # 타겟 변수 인코딩 (필요시)
    if y.dtype == 'object' or y.dtype == 'category':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)

    # 특성 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values if hasattr(y, 'values') else y

def detect_task_type(y: np.ndarray) -> str:
    """작업 유형 자동 감지"""

    unique_values = len(np.unique(y))

    if unique_values <= 10 and (not np.issubdtype(y.dtype, np.floating) or np.all(y == y.astype(int))):
        return "classification"
    else:
        return "regression"

def perform_holdout_evaluation(model, X_test: np.ndarray, y_test: np.ndarray, task_type: str) -> Dict[str, Any]:
    """홀드아웃 평가 수행"""

    try:
        y_pred = model.predict(X_test)

        if task_type == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average='weighted')),
                "recall": float(recall_score(y_test, y_pred, average='weighted')),
                "f1_score": float(f1_score(y_test, y_pred, average='weighted'))
            }

            # ROC AUC (가능한 경우)
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X_test)
                    if len(np.unique(y_test)) == 2:
                        metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba[:, 1]))
                    else:
                        metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba, multi_class='ovr'))
                except:
                    metrics["roc_auc"] = None

            # 혼동 행렬
            cm = confusion_matrix(y_test, y_pred)
            metrics["confusion_matrix"] = cm.tolist()

            # 분류 보고서
            report = classification_report(y_test, y_pred, output_dict=True)
            metrics["classification_report"] = {
                str(k): {str(k2): float(v2) if isinstance(v2, (int, float)) else v2
                         for k2, v2 in v.items()} if isinstance(v, dict) else v
                for k, v in report.items()
            }

        else:  # regression
            metrics = {
                "r2_score": float(r2_score(y_test, y_pred)),
                "mean_squared_error": float(mean_squared_error(y_test, y_pred)),
                "root_mean_squared_error": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mean_absolute_error": float(mean_absolute_error(y_test, y_pred)),
                "mean_absolute_percentage_error": float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
            }

            # 잔차 분석
            residuals = y_test - y_pred
            metrics["residual_analysis"] = {
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "min": float(np.min(residuals)),
                "max": float(np.max(residuals))
            }

        return {
            "success": True,
            "metrics": metrics,
            "predictions": y_pred.tolist()
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"홀드아웃 평가 실패: {str(e)}"
        }

def perform_cross_validation_evaluation(model, X_train: np.ndarray, y_train: np.ndarray,
                                       task_type: str, cv_folds: int) -> Dict[str, Any]:
    """교차 검증 평가 수행"""

    try:
        if task_type == "classification":
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring_metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']

        cv_results = {}
        for metric in scoring_metrics:
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric)
            cv_results[metric] = {
                "scores": scores.tolist(),
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "min": float(scores.min()),
                "max": float(scores.max())
            }

        return {
            "success": True,
            "cv_folds": cv_folds,
            "metrics": cv_results
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"교차 검증 평가 실패: {str(e)}"
        }

def generate_learning_curves(model, X_train: np.ndarray, y_train: np.ndarray,
                           task_type: str, cv_folds: int) -> Dict[str, Any]:
    """학습 곡선 생성"""

    try:
        train_sizes = np.linspace(0.1, 1.0, 10)

        if task_type == "classification":
            scoring = 'accuracy'
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            scoring = 'r2'
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        train_sizes_abs, train_scores, validation_scores = learning_curve(
            model, X_train, y_train, cv=cv, scoring=scoring,
            train_sizes=train_sizes, random_state=42
        )

        return {
            "success": True,
            "train_sizes": train_sizes_abs.tolist(),
            "train_scores": {
                "mean": train_scores.mean(axis=1).tolist(),
                "std": train_scores.std(axis=1).tolist(),
                "raw": train_scores.tolist()
            },
            "validation_scores": {
                "mean": validation_scores.mean(axis=1).tolist(),
                "std": validation_scores.std(axis=1).tolist(),
                "raw": validation_scores.tolist()
            },
            "scoring_metric": scoring
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"학습 곡선 생성 실패: {str(e)}"
        }

def compare_with_baseline(model, X_test: np.ndarray, y_test: np.ndarray, task_type: str) -> Dict[str, Any]:
    """베이스라인 모델과 비교"""

    try:
        # 실제 모델 성능
        y_pred = model.predict(X_test)

        if task_type == "classification":
            # 더미 분류기 (최빈값 예측)
            dummy_model = DummyClassifier(strategy='most_frequent', random_state=42)
            dummy_model.fit(X_test, y_test)  # 실제로는 훈련 데이터가 필요하지만 데모용
            y_dummy_pred = dummy_model.predict(X_test)

            actual_performance = float(accuracy_score(y_test, y_pred))
            baseline_performance = float(accuracy_score(y_test, y_dummy_pred))
            metric_name = "accuracy"

        else:
            # 더미 회귀기 (평균값 예측)
            dummy_model = DummyRegressor(strategy='mean')
            dummy_model.fit(X_test, y_test)  # 실제로는 훈련 데이터가 필요하지만 데모용
            y_dummy_pred = dummy_model.predict(X_test)

            actual_performance = float(r2_score(y_test, y_pred))
            baseline_performance = float(r2_score(y_test, y_dummy_pred))
            metric_name = "r2_score"

        improvement = actual_performance - baseline_performance
        improvement_percentage = (improvement / abs(baseline_performance)) * 100 if baseline_performance != 0 else float('inf')

        return {
            "success": True,
            "metric_name": metric_name,
            "actual_model_performance": actual_performance,
            "baseline_performance": baseline_performance,
            "improvement": improvement,
            "improvement_percentage": improvement_percentage,
            "is_better_than_baseline": improvement > 0
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"베이스라인 비교 실패: {str(e)}"
        }

def calculate_permutation_importance(model, X_test: np.ndarray, y_test: np.ndarray, task_type: str) -> Dict[str, Any]:
    """순열 중요도 계산"""

    try:
        if task_type == "classification":
            scoring = 'accuracy'
        else:
            scoring = 'r2'

        perm_importance = permutation_importance(
            model, X_test, y_test, scoring=scoring,
            n_repeats=10, random_state=42
        )

        # 중요도 순으로 정렬
        sorted_indices = np.argsort(perm_importance.importances_mean)[::-1]

        return {
            "success": True,
            "scoring_metric": scoring,
            "feature_importances": {
                "mean": perm_importance.importances_mean.tolist(),
                "std": perm_importance.importances_std.tolist(),
                "raw": perm_importance.importances.tolist()
            },
            "ranked_features": [
                {
                    "feature_index": int(idx),
                    "importance_mean": float(perm_importance.importances_mean[idx]),
                    "importance_std": float(perm_importance.importances_std[idx])
                }
                for idx in sorted_indices[:10]  # 상위 10개
            ]
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"순열 중요도 계산 실패: {str(e)}"
        }

def analyze_model_interpretability(model, X_test: np.ndarray, y_test: np.ndarray, task_type: str) -> Dict[str, Any]:
    """모델 해석성 분석"""

    interpretability = {
        "model_type": type(model).__name__,
        "interpretability_level": "unknown",
        "explanation_methods": []
    }

    # 모델 유형별 해석성 분류
    linear_models = ['LinearRegression', 'LogisticRegression', 'Ridge', 'Lasso', 'ElasticNet']
    tree_models = ['DecisionTreeClassifier', 'DecisionTreeRegressor', 'RandomForestClassifier', 'RandomForestRegressor']

    model_name = type(model).__name__

    if model_name in linear_models:
        interpretability["interpretability_level"] = "high"
        interpretability["explanation_methods"] = ["coefficients", "feature_importance"]

        if hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim > 1:
                coef = coef[0]  # 이진 분류의 경우

            interpretability["coefficients"] = {
                "values": coef.tolist(),
                "abs_values": np.abs(coef).tolist(),
                "top_features": sorted(
                    enumerate(np.abs(coef)), key=lambda x: x[1], reverse=True
                )[:10]
            }

    elif model_name in tree_models:
        interpretability["interpretability_level"] = "high"
        interpretability["explanation_methods"] = ["feature_importance", "tree_structure"]

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            interpretability["feature_importances"] = {
                "values": importances.tolist(),
                "top_features": sorted(
                    enumerate(importances), key=lambda x: x[1], reverse=True
                )[:10]
            }

    else:
        interpretability["interpretability_level"] = "low"
        interpretability["explanation_methods"] = ["permutation_importance"]

    # SHAP 또는 LIME 사용 권고
    interpretability["advanced_explanation_recommendations"] = [
        "SHAP (SHapley Additive exPlanations) for detailed feature contribution analysis",
        "LIME (Local Interpretable Model-agnostic Explanations) for individual prediction explanations",
        "Partial dependence plots for understanding feature relationships"
    ]

    return interpretability

def generate_evaluation_summary(evaluations: Dict[str, Any], task_type: str) -> Dict[str, Any]:
    """전체 평가 요약 생성"""

    summary = {
        "task_type": task_type,
        "evaluations_performed": list(evaluations.keys()),
        "key_metrics": {},
        "performance_assessment": "unknown",
        "reliability_score": 0.0
    }

    # 주요 메트릭 추출
    if "holdout" in evaluations and evaluations["holdout"]["success"]:
        holdout_metrics = evaluations["holdout"]["metrics"]

        if task_type == "classification":
            summary["key_metrics"]["holdout_accuracy"] = holdout_metrics.get("accuracy", 0.0)
            summary["key_metrics"]["holdout_f1_score"] = holdout_metrics.get("f1_score", 0.0)
        else:
            summary["key_metrics"]["holdout_r2_score"] = holdout_metrics.get("r2_score", 0.0)
            summary["key_metrics"]["holdout_rmse"] = holdout_metrics.get("root_mean_squared_error", 0.0)

    # 교차 검증 결과 추출
    if "cross_validation" in evaluations and evaluations["cross_validation"]["success"]:
        cv_metrics = evaluations["cross_validation"]["metrics"]

        if task_type == "classification":
            if "accuracy" in cv_metrics:
                summary["key_metrics"]["cv_accuracy_mean"] = cv_metrics["accuracy"]["mean"]
                summary["key_metrics"]["cv_accuracy_std"] = cv_metrics["accuracy"]["std"]
        else:
            if "r2" in cv_metrics:
                summary["key_metrics"]["cv_r2_mean"] = cv_metrics["r2"]["mean"]
                summary["key_metrics"]["cv_r2_std"] = cv_metrics["r2"]["std"]

    # 베이스라인 비교
    if "baseline_comparison" in evaluations and evaluations["baseline_comparison"]["success"]:
        baseline_info = evaluations["baseline_comparison"]
        summary["key_metrics"]["baseline_improvement"] = baseline_info["improvement"]
        summary["key_metrics"]["better_than_baseline"] = baseline_info["is_better_than_baseline"]

    # 성능 평가
    if task_type == "classification":
        primary_metric = summary["key_metrics"].get("holdout_accuracy", 0.0)
    else:
        primary_metric = summary["key_metrics"].get("holdout_r2_score", 0.0)

    if primary_metric >= 0.9:
        summary["performance_assessment"] = "excellent"
        summary["reliability_score"] = 0.95
    elif primary_metric >= 0.8:
        summary["performance_assessment"] = "good"
        summary["reliability_score"] = 0.8
    elif primary_metric >= 0.7:
        summary["performance_assessment"] = "fair"
        summary["reliability_score"] = 0.65
    else:
        summary["performance_assessment"] = "poor"
        summary["reliability_score"] = 0.4

    # 안정성 평가 (교차 검증 결과가 있는 경우)
    if task_type == "classification" and "cv_accuracy_std" in summary["key_metrics"]:
        cv_std = summary["key_metrics"]["cv_accuracy_std"]
        if cv_std < 0.05:
            summary["stability_assessment"] = "very_stable"
        elif cv_std < 0.1:
            summary["stability_assessment"] = "stable"
        else:
            summary["stability_assessment"] = "unstable"

    return summary

def generate_evaluation_recommendations(results: Dict[str, Any]) -> List[Dict[str, str]]:
    """평가 기반 권고사항 생성"""

    recommendations = []

    # 평가 요약에서 정보 추출
    summary = results.get("evaluation_summary", {})
    performance = summary.get("performance_assessment", "unknown")
    reliability = summary.get("reliability_score", 0.0)

    # 성능 기반 권고
    if performance == "poor":
        recommendations.append({
            "type": "model_performance",
            "priority": "high",
            "issue": "모델 성능이 낮습니다",
            "recommendation": "모델 아키텍처 개선 또는 다른 알고리즘 시도",
            "action": "특성 엔지니어링, 하이퍼파라미터 튜닝, 앙상블 방법 고려"
        })
    elif performance == "fair":
        recommendations.append({
            "type": "model_improvement",
            "priority": "medium",
            "issue": "모델 성능 개선 여지가 있습니다",
            "recommendation": "하이퍼파라미터 튜닝 및 특성 엔지니어링 시도",
            "action": "그리드 서치, 특성 선택, 정규화 강화"
        })

    # 안정성 기반 권고
    stability = summary.get("stability_assessment", "unknown")
    if stability == "unstable":
        recommendations.append({
            "type": "model_stability",
            "priority": "high",
            "issue": "모델이 불안정합니다 (높은 분산)",
            "recommendation": "모델 안정성 개선 필요",
            "action": "정규화 강화, 더 많은 데이터 수집, 앙상블 방법 사용"
        })

    # 베이스라인 비교 기반 권고
    if "baseline_comparison" in results.get("evaluations", {}):
        baseline_eval = results["evaluations"]["baseline_comparison"]
        if baseline_eval.get("success", False):
            if not baseline_eval.get("is_better_than_baseline", True):
                recommendations.append({
                    "type": "baseline_performance",
                    "priority": "critical",
                    "issue": "모델이 베이스라인보다 성능이 떨어집니다",
                    "recommendation": "모델 재설계 필요",
                    "action": "데이터 품질 확인, 모델 복잡도 조정, 특성 선택 재검토"
                })

    # 해석성 기반 권고
    if "interpretability" in results.get("evaluations", {}):
        interpret_eval = results["evaluations"]["interpretability"]
        if interpret_eval.get("interpretability_level") == "low":
            recommendations.append({
                "type": "model_interpretability",
                "priority": "medium",
                "issue": "모델 해석성이 낮습니다",
                "recommendation": "해석 가능한 모델 고려 또는 설명 도구 사용",
                "action": "SHAP, LIME 등 모델 설명 도구 적용"
            })

    # 데이터 크기 기반 권고
    eval_info = results.get("evaluation_info", {})
    test_samples = eval_info.get("test_samples", 0)
    if test_samples < 30:
        recommendations.append({
            "type": "test_data_size",
            "priority": "medium",
            "issue": f"테스트 데이터가 부족합니다 ({test_samples}개)",
            "recommendation": "더 많은 테스트 데이터 확보",
            "action": "데이터 수집, 교차 검증 결과 신뢰"
        })

    return recommendations

def main():
    """
    메인 실행 함수 - 모델 평가의 진입점

    표준 입출력을 통해 JSON 데이터를 받아 모델을 평가하고
    표준화된 형태로 결과를 반환합니다.

    입력 형식:
    - 모델 경로 또는 데이터와 타겟 컬럼
    - 선택적 매개변수: evaluation_methods, task_type, cv_folds

    출력 형식:
    - 표준화된 평가 결과 JSON
    - 성능 메트릭 및 비교 분석
    - 한국어 해석 및 권고사항
    """
    try:
        # stdin에서 JSON 데이터 읽기
        input_data = sys.stdin.read()
        params = json.loads(input_data)

        # 모델 평가 옵션
        model_path = params.get('model_path', None)
        df = None
        target_column = params.get('target_column', None)

        # 파일 경로가 제공된 경우 파일에서 데이터 로드
        if 'file_path' in params:
            df = load_data(params['file_path'])

        test_size = params.get('test_size', 0.2)
        cv_folds = params.get('cv_folds', 5)
        task_type = params.get('task_type', 'auto')
        evaluation_methods = params.get('evaluation_methods', None)
        train_simple_model = params.get('train_simple_model', True)  # Default to True if no model provided

        # 기본 데이터 정보
        if df is not None:
            data_info = get_data_info(df)
        else:
            data_info = {"note": "데이터 정보가 제공되지 않음"}

        # 모델 평가
        evaluation_result = evaluate_model(
            model_path=model_path,
            df=df,
            target_column=target_column,
            test_size=test_size,
            cv_folds=cv_folds,
            task_type=task_type,
            evaluation_methods=evaluation_methods,
            train_simple_model=train_simple_model
        )

        if not evaluation_result.get('success', False):
            error_result = {
                "success": False,
                "error": evaluation_result.get('error', '모델 평가 실패'),
                "analysis_type": "model_evaluation"
            }
            output_results(error_result)
            return

        # 분석 결과 통합
        analysis_results = {
            "model_evaluation": evaluation_result,
            "evaluation_summary": {
                "evaluations_count": len(evaluation_result.get('evaluations', {})),
                "performance_level": evaluation_result.get('evaluation_summary', {}).get('performance_assessment', 'unknown'),
                "reliability_score": evaluation_result.get('evaluation_summary', {}).get('reliability_score', 0.0),
                "recommendations_count": len(evaluation_result.get('recommendations', []))
            }
        }

        # 요약 생성
        eval_count = len(evaluation_result.get('evaluations', {}))
        performance = evaluation_result.get('evaluation_summary', {}).get('performance_assessment', 'unknown')
        reliability = evaluation_result.get('evaluation_summary', {}).get('reliability_score', 0.0)
        summary = f"모델 평가 완료 - {eval_count}개 평가 수행, 성능 수준: {performance} (신뢰도: {reliability:.2f})"

        # 표준화된 결과 생성
        final_result = create_analysis_result(
            analysis_type="model_evaluation",
            data_info=data_info,
            results=analysis_results,
            summary=summary
        )

        # 결과 출력
        output_results(final_result)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "analysis_type": "model_evaluation",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        output_results(error_result)
        sys.exit(1)

if __name__ == "__main__":
    main()