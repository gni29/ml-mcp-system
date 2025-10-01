#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification Model Training Module
분류 모델 훈련 모듈

이 모듈은 다양한 분류 알고리즘을 훈련하고 평가합니다.
주요 기능:
- 로지스틱 회귀, 랜덤 포레스트, SVM, XGBoost 분류기
- 교차 검증 및 하이퍼파라미터 튜닝
- 모델 성능 평가 및 비교
- 특성 중요도 분석
- 모델 저장 및 로드
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
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def train_classification_models(df: pd.DataFrame, target_column: str,
                               algorithms: List[str] = None,
                               test_size: float = 0.2,
                               cv_folds: int = 5,
                               tune_hyperparameters: bool = True,
                               model_save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    포괄적인 분류 모델 훈련 및 평가

    Parameters:
    -----------
    df : pd.DataFrame
        훈련용 데이터프레임
    target_column : str
        타겟 변수 컬럼명
    algorithms : List[str], optional
        사용할 알고리즘 목록 ['logistic', 'random_forest', 'svm', 'xgboost']
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

    if algorithms is None:
        algorithms = ['logistic', 'random_forest', 'svm']
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
        X, y, preprocessing_info = preprocess_data(df, target_column)
        results["preprocessing_info"] = preprocessing_info

        # 훈련/테스트 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        results["data_split"] = {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_count": X_train.shape[1],
            "class_distribution": {
                str(k): int(v) for k, v in pd.Series(y).value_counts().to_dict().items()
            }
        }

        # 각 알고리즘별 훈련
        trained_models = {}
        performance_scores = {}

        for algorithm in algorithms:
            print(f"훈련 중: {algorithm}", file=sys.stderr)

            model_result = train_single_model(
                X_train, X_test, y_train, y_test,
                algorithm, cv_folds, tune_hyperparameters
            )

            if model_result["success"]:
                trained_models[algorithm] = model_result["model"]
                results["models"][algorithm] = model_result["results"]
                performance_scores[algorithm] = model_result["results"]["test_performance"]["accuracy"]

        # 최고 성능 모델 선택
        if performance_scores:
            best_algorithm = max(performance_scores, key=performance_scores.get)
            results["best_model"] = {
                "algorithm": best_algorithm,
                "accuracy": performance_scores[best_algorithm],
                "model_details": results["models"][best_algorithm]
            }

            # 모델 저장
            if model_save_path:
                save_result = save_model(
                    trained_models[best_algorithm],
                    model_save_path,
                    best_algorithm,
                    preprocessing_info
                )
                results["model_save_info"] = save_result

        # 성능 비교
        results["performance_comparison"] = generate_performance_comparison(results["models"])

        # 권고사항
        results["recommendations"] = generate_training_recommendations(results)

        return results

    except Exception as e:
        return {
            "error": f"모델 훈련 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def preprocess_data(df: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """데이터 전처리"""

    preprocessing_info = {
        "original_shape": df.shape,
        "missing_values_handled": False,
        "categorical_encoded": False,
        "features_scaled": False,
        "target_encoded": False
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

    # 특성 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    preprocessing_info["features_scaled"] = True
    preprocessing_info["scaler_info"] = {
        "feature_names": X.columns.tolist(),
        "n_features": X_scaled.shape[1]
    }

    # 타겟 변수 인코딩 (필요시)
    if y.dtype == 'object' or y.dtype == 'category':
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        preprocessing_info["target_encoded"] = True
        preprocessing_info["target_classes"] = target_encoder.classes_.tolist()
    else:
        y_encoded = y.values
        preprocessing_info["target_classes"] = sorted(y.unique().tolist())

    return X_scaled, y_encoded, preprocessing_info

def train_single_model(X_train: np.ndarray, X_test: np.ndarray,
                      y_train: np.ndarray, y_test: np.ndarray,
                      algorithm: str, cv_folds: int,
                      tune_hyperparameters: bool) -> Dict[str, Any]:
    """단일 모델 훈련"""

    try:
        # 모델 및 하이퍼파라미터 설정
        model_configs = get_model_configurations(algorithm)

        if not model_configs["available"]:
            return {
                "success": False,
                "error": model_configs["error"]
            }

        model = model_configs["model"]
        param_grid = model_configs["param_grid"] if tune_hyperparameters else {}

        # 하이퍼파라미터 튜닝
        if tune_hyperparameters and param_grid:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1
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
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds)

        # 예측 및 성능 평가
        y_pred = best_model.predict(X_test)
        y_pred_proba = None
        if hasattr(best_model, "predict_proba"):
            y_pred_proba = best_model.predict_proba(X_test)

        # 성능 메트릭 계산
        performance = calculate_performance_metrics(y_test, y_pred, y_pred_proba)

        # 특성 중요도 (가능한 경우)
        feature_importance = get_feature_importance(best_model, X_train.shape[1])

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
                "feature_importance": feature_importance
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"{algorithm} 훈련 실패: {str(e)}"
        }

def get_model_configurations(algorithm: str) -> Dict[str, Any]:
    """모델 설정 반환"""

    configs = {
        "logistic": {
            "available": True,
            "model": LogisticRegression(random_state=42, max_iter=1000),
            "param_grid": {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        },
        "random_forest": {
            "available": True,
            "model": RandomForestClassifier(random_state=42),
            "param_grid": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        "svm": {
            "available": True,
            "model": SVC(random_state=42, probability=True),
            "param_grid": {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
        }
    }

    if XGBOOST_AVAILABLE:
        configs["xgboost"] = {
            "available": True,
            "model": xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
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

def calculate_performance_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """성능 메트릭 계산"""

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average='weighted')),
        "recall": float(recall_score(y_true, y_pred, average='weighted')),
        "f1_score": float(f1_score(y_true, y_pred, average='weighted'))
    }

    # ROC AUC (이진 분류 또는 확률 예측 가능한 경우)
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:  # 이진 분류
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
            else:  # 다중 분류
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba, multi_class='ovr'))
        except:
            metrics["roc_auc"] = None

    # 분류 보고서
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics["classification_report"] = {
        str(k): {str(k2): float(v2) if isinstance(v2, (int, float)) else v2
                 for k2, v2 in v.items()} if isinstance(v, dict) else v
        for k, v in report.items()
    }

    # 혼동 행렬
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics

def get_feature_importance(model, n_features: int) -> Optional[Dict[str, Any]]:
    """특성 중요도 추출"""

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
            if model.coef_.ndim == 1:
                coef = model.coef_
            else:
                coef = model.coef_[0]  # 이진 분류의 경우

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

def generate_performance_comparison(models: Dict[str, Any]) -> Dict[str, Any]:
    """모델 성능 비교 분석"""

    if not models:
        return {"error": "비교할 모델이 없습니다"}

    comparison = {
        "metrics_comparison": {},
        "ranking": {},
        "statistical_analysis": {}
    }

    # 메트릭별 비교
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    for metric in metrics:
        comparison["metrics_comparison"][metric] = {
            str(algo): float(results["test_performance"][metric])
            for algo, results in models.items()
            if metric in results["test_performance"]
        }

    # 순위 매기기
    for metric in metrics:
        if metric in comparison["metrics_comparison"]:
            sorted_models = sorted(
                comparison["metrics_comparison"][metric].items(),
                key=lambda x: x[1], reverse=True
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

def generate_training_recommendations(results: Dict[str, Any]) -> List[Dict[str, str]]:
    """훈련 권고사항 생성"""

    recommendations = []

    # 데이터 크기 기반 권고
    if "data_split" in results:
        train_size = results["data_split"]["train_samples"]
        if train_size < 100:
            recommendations.append({
                "type": "data_size",
                "priority": "high",
                "issue": "훈련 데이터가 부족합니다",
                "recommendation": f"현재 {train_size}개 샘플, 최소 100개 이상 권장",
                "action": "더 많은 데이터 수집 필요"
            })

    # 클래스 불균형 확인
    if "data_split" in results and "class_distribution" in results["data_split"]:
        class_counts = results["data_split"]["class_distribution"]
        if len(class_counts) > 1:
            min_class = min(class_counts.values())
            max_class = max(class_counts.values())
            imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')

            if imbalance_ratio > 3:
                recommendations.append({
                    "type": "class_imbalance",
                    "priority": "medium",
                    "issue": f"클래스 불균형 발견 (비율: {imbalance_ratio:.1f}:1)",
                    "recommendation": "클래스 균형 기법 적용",
                    "action": "SMOTE, 가중치 조정, 또는 언더샘플링 고려"
                })

    # 모델 성능 기반 권고
    if "best_model" in results and results["best_model"]:
        best_accuracy = results["best_model"]["accuracy"]
        if best_accuracy < 0.7:
            recommendations.append({
                "type": "model_performance",
                "priority": "high",
                "issue": f"최고 모델 정확도가 낮습니다 ({best_accuracy:.3f})",
                "recommendation": "모델 개선 필요",
                "action": "특성 엔지니어링, 더 복잡한 모델, 앙상블 기법 고려"
            })
        elif best_accuracy > 0.95:
            recommendations.append({
                "type": "overfitting_risk",
                "priority": "medium",
                "issue": f"매우 높은 정확도 ({best_accuracy:.3f})",
                "recommendation": "과적합 가능성 검토",
                "action": "교차 검증 결과 확인, 정규화 강화"
            })

    # 특성 수 기반 권고
    if "preprocessing_info" in results and "scaler_info" in results["preprocessing_info"]:
        n_features = results["preprocessing_info"]["scaler_info"]["n_features"]
        if "data_split" in results:
            n_samples = results["data_split"]["train_samples"]
            if n_features > n_samples / 10:
                recommendations.append({
                    "type": "curse_of_dimensionality",
                    "priority": "medium",
                    "issue": f"특성 수가 샘플 대비 많습니다 ({n_features} 특성, {n_samples} 샘플)",
                    "recommendation": "차원 축소 고려",
                    "action": "PCA, 특성 선택, 정규화 강화"
                })

    return recommendations

def save_model(model, save_path: str, algorithm: str, preprocessing_info: Dict[str, Any]) -> Dict[str, Any]:
    """모델 저장"""

    try:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 모델 저장
        model_file = save_path / f"{algorithm}_model.joblib"
        joblib.dump(model, model_file)

        # 전처리 정보 저장
        preprocessing_file = save_path / f"{algorithm}_preprocessing.json"
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
    메인 실행 함수 - 분류 모델 훈련의 진입점

    표준 입출력을 통해 JSON 데이터를 받아 분류 모델을 훈련하고
    표준화된 형태로 결과를 반환합니다.

    입력 형식:
    - JSON을 통한 데이터 또는 파일 경로
    - 필수 매개변수: target_column
    - 선택적 매개변수: algorithms, test_size, cv_folds, tune_hyperparameters

    출력 형식:
    - 표준화된 분석 결과 JSON
    - 모델 성능 및 비교 분석
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

        # 분류 훈련 옵션
        target_column = params['target_column']
        algorithms = params.get('algorithms', None)
        test_size = params.get('test_size', 0.2)
        cv_folds = params.get('cv_folds', 5)
        tune_hyperparameters = params.get('tune_hyperparameters', True)
        model_save_path = params.get('model_save_path', None)

        # 데이터 기본 정보 추출
        data_info = get_data_info(df)

        # 분류 모델 훈련
        training_result = train_classification_models(
            df, target_column, algorithms, test_size, cv_folds,
            tune_hyperparameters, model_save_path
        )

        if not training_result.get('success', False):
            error_result = {
                "success": False,
                "error": training_result.get('error', '분류 모델 훈련 실패'),
                "analysis_type": "classification_training"
            }
            output_results(error_result)
            return

        # 분석 결과 통합
        analysis_results = {
            "classification_training": training_result,
            "training_summary": {
                "algorithms_trained": len(training_result.get('models', {})),
                "best_algorithm": training_result.get('best_model', {}).get('algorithm', 'None'),
                "best_accuracy": training_result.get('best_model', {}).get('accuracy', 0.0),
                "recommendations_count": len(training_result.get('recommendations', []))
            }
        }

        # 요약 생성
        models_count = len(training_result.get('models', {}))
        best_algo = training_result.get('best_model', {}).get('algorithm', 'None')
        best_acc = training_result.get('best_model', {}).get('accuracy', 0.0)
        summary = f"분류 모델 훈련 완료 - {models_count}개 모델 훈련, 최고 성능: {best_algo} ({best_acc:.3f})"

        # 표준화된 결과 생성
        final_result = create_analysis_result(
            analysis_type="classification_training",
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
            "analysis_type": "classification_training",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        output_results(error_result)
        sys.exit(1)

if __name__ == "__main__":
    main()