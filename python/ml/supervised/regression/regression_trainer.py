#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression Model Training Module
회귀 모델 훈련 모듈

이 모듈은 다양한 회귀 알고리즘을 훈련하고 평가합니다.
주요 기능:
- 선형 회귀, 랜덤 포레스트, SVR, XGBoost 회귀
- 교차 검증 및 하이퍼파라미터 튜닝
- 모델 성능 평가 및 비교 (R², MAE, MSE, RMSE)
- 특성 중요도 및 회귀 계수 분석
- 잔차 분석 및 모델 진단
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
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent / "ml-mcp-shared" / "python"))

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
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.metrics import median_absolute_error
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def train_regression_models(df: pd.DataFrame, target_column: str,
                           algorithms: List[str] = None,
                           test_size: float = 0.2,
                           cv_folds: int = 5,
                           tune_hyperparameters: bool = True,
                           model_save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    포괄적인 회귀 모델 훈련 및 평가

    Parameters:
    -----------
    df : pd.DataFrame
        훈련용 데이터프레임
    target_column : str
        타겟 변수 컬럼명
    algorithms : List[str], optional
        사용할 알고리즘 목록 ['linear', 'ridge', 'lasso', 'random_forest', 'svr', 'xgboost']
    test_size : float, default=0.2
        테스트 데이터 비율
    cv_folds : int, default=5
        교차 검증 폴드 수
    tune_hyperparameters : bool, default=True
        하이퍼파라미터 튜닝 여부
    model_save_path : str, optional
        모델 저장 경로

    Returns:
    --------
    Dict[str, Any]
        훈련 결과 및 모델 성능
    """

    if not SKLEARN_AVAILABLE:
        return {
            "error": "scikit-learn이 설치되지 않았습니다",
            "required_package": "scikit-learn"
        }

    if target_column not in df.columns:
        return {
            "error": f"타겟 컬럼 '{target_column}'이 데이터프레임에 없습니다",
            "available_columns": df.columns.tolist()
        }

    # 타겟 변수가 수치형인지 확인
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        return {
            "error": f"타겟 변수 '{target_column}'이 수치형이 아닙니다",
            "target_dtype": str(df[target_column].dtype)
        }

    if algorithms is None:
        algorithms = ['linear', 'ridge', 'random_forest']
        if XGBOOST_AVAILABLE:
            algorithms.append('xgboost')

    try:
        results = {
            "success": True,
            "training_info": {
                "target_column": target_column,
                "algorithms_used": algorithms,
                "test_size": test_size,
                "cv_folds": cv_folds,
                "hyperparameter_tuning": tune_hyperparameters
            },
            "models": {},
            "performance_comparison": {},
            "best_model": None
        }

        # 데이터 전처리
        X, y, preprocessing_info = preprocess_regression_data(df, target_column)
        results["preprocessing_info"] = preprocessing_info

        # 훈련/테스트 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        results["data_split"] = {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_count": X_train.shape[1],
            "target_statistics": {
                "mean": float(np.mean(y)),
                "std": float(np.std(y)),
                "min": float(np.min(y)),
                "max": float(np.max(y)),
                "median": float(np.median(y))
            }
        }

        # 각 알고리즘별 훈련
        trained_models = {}
        performance_scores = {}

        for algorithm in algorithms:
            print(f"훈련 중: {algorithm}", file=sys.stderr)

            model_result = train_single_regression_model(
                X_train, X_test, y_train, y_test,
                algorithm, cv_folds, tune_hyperparameters
            )

            if model_result["success"]:
                trained_models[algorithm] = model_result["model"]
                results["models"][algorithm] = model_result["results"]
                performance_scores[algorithm] = model_result["results"]["test_performance"]["r2_score"]

        # 최고 성능 모델 선택 (R² 기준)
        if performance_scores:
            best_algorithm = max(performance_scores, key=performance_scores.get)
            results["best_model"] = {
                "algorithm": best_algorithm,
                "r2_score": performance_scores[best_algorithm],
                "model_details": results["models"][best_algorithm]
            }

            # 모델 저장
            if model_save_path:
                save_result = save_regression_model(
                    trained_models[best_algorithm],
                    model_save_path,
                    best_algorithm,
                    preprocessing_info
                )
                results["model_save_info"] = save_result

        # 성능 비교
        results["performance_comparison"] = generate_regression_performance_comparison(results["models"])

        # 잔차 분석
        if trained_models and best_algorithm in trained_models:
            best_model = trained_models[best_algorithm]
            y_pred = best_model.predict(X_test)
            results["residual_analysis"] = perform_residual_analysis(y_test, y_pred)

        # 권고사항
        results["recommendations"] = generate_regression_recommendations(results)

        return results

    except Exception as e:
        return {
            "error": f"회귀 모델 훈련 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def preprocess_regression_data(df: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """회귀 데이터 전처리"""

    preprocessing_info = {
        "original_shape": df.shape,
        "missing_values_handled": False,
        "categorical_encoded": False,
        "features_scaled": False,
        "outliers_detected": False
    }

    # 타겟 변수 분리
    y = df[target_column].copy()
    X = df.drop(columns=[target_column]).copy()

    # 결측치 처리
    if X.isnull().sum().sum() > 0:
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns

        # 수치형: 중위값으로 대체
        for col in numeric_cols:
            X[col].fillna(X[col].median(), inplace=True)

        # 범주형: 최빈값으로 대체
        for col in categorical_cols:
            X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'unknown', inplace=True)

        preprocessing_info["missing_values_handled"] = True

    # 범주형 변수 인코딩
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_columns:
        encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

        preprocessing_info["categorical_encoded"] = True
        preprocessing_info["encoders"] = {col: enc.classes_.tolist() for col, enc in encoders.items()}

    # 이상치 탐지 (IQR 방법)
    outlier_info = detect_outliers(X, y)
    preprocessing_info["outliers_detected"] = True
    preprocessing_info["outlier_info"] = outlier_info

    # 특성 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    preprocessing_info["features_scaled"] = True
    preprocessing_info["scaler_info"] = {
        "feature_names": X.columns.tolist(),
        "n_features": X_scaled.shape[1]
    }

    return X_scaled, y.values, preprocessing_info

def detect_outliers(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """이상치 탐지"""

    outlier_info = {
        "feature_outliers": {},
        "target_outliers": {},
        "total_outlier_rows": 0
    }

    try:
        # 특성별 이상치 (IQR 방법)
        for col in X.select_dtypes(include=[np.number]).columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
            outlier_info["feature_outliers"][col] = {
                "count": int(outliers),
                "percentage": float(outliers / len(X) * 100),
                "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)}
            }

        # 타겟 변수 이상치
        Q1_y = y.quantile(0.25)
        Q3_y = y.quantile(0.75)
        IQR_y = Q3_y - Q1_y
        lower_bound_y = Q1_y - 1.5 * IQR_y
        upper_bound_y = Q3_y + 1.5 * IQR_y

        target_outliers = ((y < lower_bound_y) | (y > upper_bound_y)).sum()
        outlier_info["target_outliers"] = {
            "count": int(target_outliers),
            "percentage": float(target_outliers / len(y) * 100),
            "bounds": {"lower": float(lower_bound_y), "upper": float(upper_bound_y)}
        }

    except Exception as e:
        outlier_info["error"] = str(e)

    return outlier_info

def train_single_regression_model(X_train: np.ndarray, X_test: np.ndarray,
                                 y_train: np.ndarray, y_test: np.ndarray,
                                 algorithm: str, cv_folds: int,
                                 tune_hyperparameters: bool) -> Dict[str, Any]:
    """단일 회귀 모델 훈련"""

    try:
        # 모델 및 하이퍼파라미터 설정
        model_configs = get_regression_model_configurations(algorithm)

        if not model_configs["available"]:
            return {
                "success": False,
                "error": model_configs["error"]
            }

        model = model_configs["model"]
        param_grid = model_configs["param_grid"] if tune_hyperparameters else {}

        # 하이퍼파라미터 튜닝
        if tune_hyperparameters and param_grid:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                model, param_grid, cv=cv, scoring='r2', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            tuning_info = {
                "best_params": grid_search.best_params_,
                "best_cv_score": float(grid_search.best_score_),
                "param_grid": param_grid
            }
        else:
            model.fit(X_train, y_train)
            best_model = model
            tuning_info = {"hyperparameter_tuning": False}

        # 교차 검증 점수
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring='r2')

        # 예측 및 성능 평가
        y_pred = best_model.predict(X_test)

        # 성능 메트릭 계산
        performance = calculate_regression_metrics(y_test, y_pred)

        # 특성 중요도 (가능한 경우)
        feature_importance = get_regression_feature_importance(best_model, X_train.shape[1])

        return {
            "success": True,
            "model": best_model,
            "results": {
                "algorithm": algorithm,
                "tuning_info": tuning_info,
                "cross_validation": {
                    "cv_scores": cv_scores.tolist(),
                    "mean_cv_score": float(cv_scores.mean()),
                    "std_cv_score": float(cv_scores.std())
                },
                "test_performance": performance,
                "feature_importance": feature_importance,
                "predictions": y_pred.tolist()
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"{algorithm} 회귀 훈련 실패: {str(e)}"
        }

def get_regression_model_configurations(algorithm: str) -> Dict[str, Any]:
    """회귀 모델 설정 반환"""

    configs = {
        "linear": {
            "available": True,
            "model": LinearRegression(),
            "param_grid": {}  # 선형 회귀는 하이퍼파라미터가 없음
        },
        "ridge": {
            "available": True,
            "model": Ridge(random_state=42),
            "param_grid": {
                'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            }
        },
        "lasso": {
            "available": True,
            "model": Lasso(random_state=42),
            "param_grid": {
                'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            }
        },
        "elastic_net": {
            "available": True,
            "model": ElasticNet(random_state=42),
            "param_grid": {
                'alpha': [0.001, 0.01, 0.1, 1, 10],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        },
        "random_forest": {
            "available": True,
            "model": RandomForestRegressor(random_state=42),
            "param_grid": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        "svr": {
            "available": True,
            "model": SVR(),
            "param_grid": {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
    }

    if XGBOOST_AVAILABLE:
        configs["xgboost"] = {
            "available": True,
            "model": xgb.XGBRegressor(random_state=42),
            "param_grid": {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
    else:
        configs["xgboost"] = {
            "available": False,
            "error": "XGBoost가 설치되지 않았습니다"
        }

    return configs.get(algorithm, {
        "available": False,
        "error": f"지원하지 않는 알고리즘: {algorithm}"
    })

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """회귀 성능 메트릭 계산"""

    metrics = {
        "r2_score": float(r2_score(y_true, y_pred)),
        "mean_squared_error": float(mean_squared_error(y_true, y_pred)),
        "root_mean_squared_error": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mean_absolute_error": float(mean_absolute_error(y_true, y_pred)),
        "median_absolute_error": float(median_absolute_error(y_true, y_pred))
    }

    # 추가 메트릭
    metrics["mean_absolute_percentage_error"] = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    metrics["explained_variance"] = float(1 - np.var(y_true - y_pred) / np.var(y_true))

    return metrics

def get_regression_feature_importance(model, n_features: int) -> Optional[Dict[str, Any]]:
    """회귀 모델의 특성 중요도 추출"""

    try:
        if hasattr(model, 'feature_importances_'):
            # 트리 기반 모델
            importances = model.feature_importances_
            return {
                "type": "feature_importances",
                "values": importances.tolist(),
                "top_features": sorted(
                    enumerate(importances),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }
        elif hasattr(model, 'coef_'):
            # 선형 모델
            coef = model.coef_
            abs_coef = np.abs(coef)
            return {
                "type": "coefficients",
                "values": coef.tolist(),
                "abs_values": abs_coef.tolist(),
                "top_features": sorted(
                    enumerate(abs_coef),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }
    except:
        pass

    return None

def perform_residual_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """잔차 분석 수행"""

    residuals = y_true - y_pred

    analysis = {
        "residual_statistics": {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
            "median": float(np.median(residuals))
        },
        "residual_distribution": {
            "skewness": float(residuals.std() / residuals.mean() if residuals.mean() != 0 else 0),
            "normality_test": "Shapiro-Wilk test recommended for detailed analysis"
        },
        "outlier_residuals": []
    }

    # 잔차 이상치 (|잔차| > 2*표준편차)
    threshold = 2 * np.std(residuals)
    outlier_indices = np.where(np.abs(residuals) > threshold)[0]

    analysis["outlier_residuals"] = {
        "count": len(outlier_indices),
        "indices": outlier_indices.tolist(),
        "threshold": float(threshold)
    }

    # 잔차 패턴 분석
    analysis["pattern_analysis"] = {
        "heteroscedasticity_warning": bool(np.std(residuals[:len(residuals)//2]) / np.std(residuals[len(residuals)//2:]) > 1.5),
        "autocorrelation_warning": "Use Durbin-Watson test for time series data"
    }

    return analysis

def generate_regression_performance_comparison(models: Dict[str, Any]) -> Dict[str, Any]:
    """회귀 모델 성능 비교 분석"""

    if not models:
        return {"error": "비교할 모델이 없습니다"}

    comparison = {
        "metrics_comparison": {},
        "ranking": {},
        "statistical_analysis": {}
    }

    # 메트릭별 비교
    metrics = ['r2_score', 'mean_squared_error', 'mean_absolute_error', 'root_mean_squared_error']
    for metric in metrics:
        comparison["metrics_comparison"][metric] = {
            str(algo): float(results["test_performance"][metric])
            for algo, results in models.items()
            if metric in results["test_performance"]
        }

    # 순위 매기기 (R²는 높을수록, 에러는 낮을수록 좋음)
    for metric in metrics:
        if metric in comparison["metrics_comparison"]:
            reverse = metric == 'r2_score'  # R²만 내림차순
            sorted_models = sorted(
                comparison["metrics_comparison"][metric].items(),
                key=lambda x: x[1], reverse=reverse
            )
            comparison["ranking"][metric] = [{"algorithm": algo, "score": score}
                                           for algo, score in sorted_models]

    # 통계 분석
    for metric in metrics:
        if metric in comparison["metrics_comparison"]:
            scores = list(comparison["metrics_comparison"][metric].values())
            comparison["statistical_analysis"][metric] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "range": float(np.max(scores) - np.min(scores))
            }

    return comparison

def generate_regression_recommendations(results: Dict[str, Any]) -> List[Dict[str, str]]:
    """회귀 훈련 권고사항 생성"""

    recommendations = []

    # 데이터 크기 기반 권고
    if "data_split" in results:
        train_size = results["data_split"]["train_samples"]
        feature_count = results["data_split"]["feature_count"]

        if train_size < 30:
            recommendations.append({
                "type": "data_size",
                "priority": "high",
                "issue": "훈련 데이터가 매우 부족합니다",
                "recommendation": f"현재 {train_size}개 샘플, 최소 30개 이상 권장",
                "action": "더 많은 데이터 수집 필요"
            })
        elif train_size < feature_count * 5:
            recommendations.append({
                "type": "sample_to_feature_ratio",
                "priority": "medium",
                "issue": f"샘플 수가 특성 수 대비 부족합니다 ({train_size} 샘플, {feature_count} 특성)",
                "recommendation": "샘플:특성 비율 최소 5:1 권장",
                "action": "데이터 추가 수집 또는 특성 선택"
            })

    # 모델 성능 기반 권고
    if "best_model" in results and results["best_model"]:
        best_r2 = results["best_model"]["r2_score"]
        if best_r2 < 0.3:
            recommendations.append({
                "type": "model_performance",
                "priority": "high",
                "issue": f"최고 모델 R² 점수가 낮습니다 ({best_r2:.3f})",
                "recommendation": "모델 성능 개선 필요",
                "action": "특성 엔지니어링, 다른 알고리즘, 데이터 품질 검토"
            })
        elif best_r2 > 0.95:
            recommendations.append({
                "type": "overfitting_risk",
                "priority": "medium",
                "issue": f"매우 높은 R² 점수 ({best_r2:.3f})",
                "recommendation": "과적합 가능성 검토",
                "action": "교차 검증 결과 확인, 정규화 강화"
            })

    # 이상치 기반 권고
    if "preprocessing_info" in results and "outlier_info" in results["preprocessing_info"]:
        outlier_info = results["preprocessing_info"]["outlier_info"]
        if "target_outliers" in outlier_info:
            target_outlier_pct = outlier_info["target_outliers"].get("percentage", 0)
            if target_outlier_pct > 10:
                recommendations.append({
                    "type": "target_outliers",
                    "priority": "medium",
                    "issue": f"타겟 변수에 많은 이상치 ({target_outlier_pct:.1f}%)",
                    "recommendation": "이상치 처리 고려",
                    "action": "이상치 제거, 변환, 또는 로버스트 모델 사용"
                })

    # 잔차 분석 기반 권고
    if "residual_analysis" in results:
        residual_analysis = results["residual_analysis"]
        if residual_analysis.get("pattern_analysis", {}).get("heteroscedasticity_warning", False):
            recommendations.append({
                "type": "heteroscedasticity",
                "priority": "medium",
                "issue": "잔차의 이분산성 탐지",
                "recommendation": "이분산성 문제 해결 필요",
                "action": "변수 변환, 가중 회귀, 또는 로버스트 표준오차 사용"
            })

    return recommendations

def save_regression_model(model, save_path: str, algorithm: str, preprocessing_info: Dict[str, Any]) -> Dict[str, Any]:
    """회귀 모델 저장"""

    try:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 모델 저장
        model_file = save_path / f"{algorithm}_regression_model.joblib"
        joblib.dump(model, model_file)

        # 전처리 정보 저장
        preprocessing_file = save_path / f"{algorithm}_regression_preprocessing.json"
        with open(preprocessing_file, 'w', encoding='utf-8') as f:
            json.dump(preprocessing_info, f, ensure_ascii=False, indent=2, default=str)

        return {
            "success": True,
            "model_file": str(model_file),
            "preprocessing_file": str(preprocessing_file),
            "algorithm": algorithm
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"모델 저장 실패: {str(e)}"
        }

def main():
    """
    메인 실행 함수 - 회귀 모델 훈련의 진입점

    표준 입출력을 통해 JSON 데이터를 받아 회귀 모델을 훈련하고
    표준화된 형태로 결과를 반환합니다.

    입력 형식:
    - JSON을 통한 데이터 또는 파일 경로
    - 필수 매개변수: target_column
    - 선택적 매개변수: algorithms, test_size, cv_folds, tune_hyperparameters

    출력 형식:
    - 표준화된 분석 결과 JSON
    - 모델 성능 및 비교 분석
    - 잔차 분석 및 진단
    - 한국어 해석 및 권고사항
    """
    try:
        # stdin에서 JSON 데이터 읽기
        input_data = sys.stdin.read()
        params = json.loads(input_data)

        # 필수 매개변수 검증
        validate_required_params(params, ['target_column'])

        # 파일 경로가 제공된 경우 파일에서 데이터 로드
        if 'file_path' in params:
            df = load_data(params['file_path'])
        else:
            # JSON 데이터에서 직접 DataFrame 생성
            if 'data' in params:
                df = pd.DataFrame(params['data'])
            else:
                df = pd.DataFrame(params)

        # 회귀 훈련 옵션
        target_column = params['target_column']
        algorithms = params.get('algorithms', None)
        test_size = params.get('test_size', 0.2)
        cv_folds = params.get('cv_folds', 5)
        tune_hyperparameters = params.get('tune_hyperparameters', True)
        model_save_path = params.get('model_save_path', None)

        # 데이터 기본 정보 추출
        data_info = get_data_info(df)

        # 회귀 모델 훈련
        training_result = train_regression_models(
            df, target_column, algorithms, test_size, cv_folds,
            tune_hyperparameters, model_save_path
        )

        if not training_result.get('success', False):
            error_result = {
                "success": False,
                "error": training_result.get('error', '회귀 모델 훈련 실패'),
                "analysis_type": "regression_training"
            }
            output_results(error_result)
            return

        # 분석 결과 통합
        analysis_results = {
            "regression_training": training_result,
            "training_summary": {
                "algorithms_trained": len(training_result.get('models', {})),
                "best_algorithm": training_result.get('best_model', {}).get('algorithm', 'None'),
                "best_r2_score": training_result.get('best_model', {}).get('r2_score', 0.0),
                "recommendations_count": len(training_result.get('recommendations', []))
            }
        }

        # 요약 생성
        models_count = len(training_result.get('models', {}))
        best_algo = training_result.get('best_model', {}).get('algorithm', 'None')
        best_r2 = training_result.get('best_model', {}).get('r2_score', 0.0)
        summary = f"회귀 모델 훈련 완료 - {models_count}개 모델 훈련, 최고 성능: {best_algo} (R²={best_r2:.3f})"

        # 표준화된 결과 생성
        final_result = create_analysis_result(
            analysis_type="regression_training",
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
            "analysis_type": "regression_training",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        output_results(error_result)
        sys.exit(1)

if __name__ == "__main__":
    main()