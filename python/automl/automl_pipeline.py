#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoML Pipeline
자동 머신러닝 파이프라인

이 모듈은 자동화된 머신러닝 파이프라인을 제공합니다.
주요 기능:
- 자동 알고리즘 선택
- 하이퍼파라미터 자동 튜닝
- 특성 엔지니어링 자동화
- 모델 앙상블
- 성능 최적화
- 전체 파이프라인 자동화
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
sys.path.append(str(Path(__file__).parent.parent.parent / "ml-mcp-shared" / "python"))

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
    from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
    from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel
    from sklearn.ensemble import VotingClassifier, VotingRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
    from sklearn.svm import SVC, SVR
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
    from sklearn.pipeline import Pipeline
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

class AutoMLPipeline:
    """자동 머신러닝 파이프라인"""

    def __init__(self):
        self.best_model = None
        self.preprocessing_pipeline = None
        self.feature_selector = None
        self.search_results = {}
        self.ensemble_model = None

    def run_automl(self, df: pd.DataFrame, target_column: str, task_type: str = "auto",
                   time_budget: int = 300, optimization_metric: str = "auto",
                   include_ensembles: bool = True, max_models: int = 10) -> Dict[str, Any]:
        """
        AutoML 파이프라인 실행

        Parameters:
        -----------
        df : pd.DataFrame
            훈련용 데이터프레임
        target_column : str
            타겟 변수 컬럼명
        task_type : str, default="auto"
            작업 유형 ("classification", "regression", "auto")
        time_budget : int, default=300
            최적화 시간 예산 (초)
        optimization_metric : str, default="auto"
            최적화 메트릭
        include_ensembles : bool, default=True
            앙상블 모델 포함 여부
        max_models : int, default=10
            최대 시도할 모델 수

        Returns:
        --------
        Dict[str, Any]
            AutoML 결과
        """

        if not SKLEARN_AVAILABLE:
            return {
                "error": "scikit-learn이 설치되지 않았습니다",
                "required_package": "scikit-learn"
            }

        try:
            results = {
                "success": True,
                "automl_config": {
                    "task_type": task_type,
                    "time_budget": time_budget,
                    "optimization_metric": optimization_metric,
                    "include_ensembles": include_ensembles,
                    "max_models": max_models
                },
                "pipeline_steps": [],
                "model_results": {},
                "best_model": {},
                "performance_comparison": {},
                "feature_engineering": {}
            }

            # 1. 데이터 분석 및 작업 유형 결정
            print("🔍 데이터 분석 중...", file=sys.stderr)
            task_type = self.determine_task_type(df, target_column) if task_type == "auto" else task_type
            results["automl_config"]["determined_task_type"] = task_type
            results["pipeline_steps"].append("데이터 분석 및 작업 유형 결정")

            # 2. 데이터 전처리
            print("🔧 데이터 전처리 중...", file=sys.stderr)
            X_processed, y_processed, preprocessing_info = self.preprocess_data(df, target_column, task_type)
            results["preprocessing_info"] = preprocessing_info
            results["pipeline_steps"].append("데이터 전처리")

            # 3. 특성 엔지니어링
            print("⚙️ 특성 엔지니어링 중...", file=sys.stderr)
            X_engineered, feature_engineering_info = self.perform_feature_engineering(
                X_processed, y_processed, task_type
            )
            results["feature_engineering"] = feature_engineering_info
            results["pipeline_steps"].append("특성 엔지니어링")

            # 4. 훈련/테스트 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X_engineered, y_processed, test_size=0.2, random_state=42,
                stratify=y_processed if task_type == "classification" else None
            )

            # 5. 모델 선택 및 하이퍼파라미터 최적화
            print("🤖 모델 최적화 중...", file=sys.stderr)
            model_results = self.optimize_models(
                X_train, X_test, y_train, y_test, task_type,
                time_budget, optimization_metric, max_models
            )
            results["model_results"] = model_results
            results["pipeline_steps"].append("모델 최적화")

            # 6. 앙상블 모델 생성
            if include_ensembles and len(model_results) >= 2:
                print("🎭 앙상블 모델 생성 중...", file=sys.stderr)
                ensemble_result = self.create_ensemble(
                    model_results, X_train, X_test, y_train, y_test, task_type
                )
                results["ensemble_result"] = ensemble_result
                results["pipeline_steps"].append("앙상블 모델 생성")

            # 7. 최고 성능 모델 선택
            best_model_info = self.select_best_model(results.get("model_results", {}),
                                                   results.get("ensemble_result", {}), task_type)
            results["best_model"] = best_model_info
            results["pipeline_steps"].append("최고 성능 모델 선택")

            # 8. 모델 해석 및 인사이트
            print("📊 모델 해석 중...", file=sys.stderr)
            interpretation_results = self.interpret_model(
                best_model_info, X_engineered.columns.tolist() if hasattr(X_engineered, 'columns') else None
            )
            results["model_interpretation"] = interpretation_results
            results["pipeline_steps"].append("모델 해석")

            # 9. 권고사항 생성
            recommendations = self.generate_automl_recommendations(results)
            results["recommendations"] = recommendations

            return results

        except Exception as e:
            return {
                "success": False,
                "error": f"AutoML 파이프라인 실패: {str(e)}",
                "error_type": type(e).__name__
            }

    def determine_task_type(self, df: pd.DataFrame, target_column: str) -> str:
        """작업 유형 자동 결정"""

        target_series = df[target_column]
        unique_values = target_series.nunique()
        total_values = len(target_series)

        # 수치형이지만 카테고리가 적은 경우
        if pd.api.types.is_numeric_dtype(target_series):
            if unique_values <= 10 and unique_values < total_values * 0.1:
                return "classification"
            else:
                return "regression"
        else:
            return "classification"

    def preprocess_data(self, df: pd.DataFrame, target_column: str, task_type: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """데이터 전처리"""

        preprocessing_info = {
            "steps_performed": [],
            "missing_values": {},
            "categorical_encoding": {},
            "scaling": {},
            "feature_count": {"original": 0, "final": 0}
        }

        # 타겟 변수 분리
        y = df[target_column].copy()
        X = df.drop(columns=[target_column]).copy()
        preprocessing_info["feature_count"]["original"] = X.shape[1]

        # 결측값 처리
        missing_info = self.handle_missing_values(X)
        preprocessing_info["missing_values"] = missing_info
        if missing_info["values_imputed"] > 0:
            preprocessing_info["steps_performed"].append("결측값 처리")

        # 범주형 변수 인코딩
        encoding_info = self.encode_categorical_variables(X)
        preprocessing_info["categorical_encoding"] = encoding_info
        if encoding_info["columns_encoded"] > 0:
            preprocessing_info["steps_performed"].append("범주형 변수 인코딩")

        # 타겟 변수 처리
        if task_type == "classification" and y.dtype == 'object':
            le = LabelEncoder()
            y_processed = le.fit_transform(y)
            preprocessing_info["target_encoding"] = {
                "classes": le.classes_.tolist(),
                "encoded": True
            }
        else:
            y_processed = y.values

        # 특성 스케일링
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        preprocessing_info["scaling"] = {
            "method": "StandardScaler",
            "applied": True
        }
        preprocessing_info["steps_performed"].append("특성 스케일링")

        preprocessing_info["feature_count"]["final"] = X_scaled.shape[1]

        return X_scaled, y_processed, preprocessing_info

    def handle_missing_values(self, X: pd.DataFrame) -> Dict[str, Any]:
        """결측값 처리"""

        missing_info = {
            "original_missing": X.isnull().sum().sum(),
            "values_imputed": 0,
            "strategies": {}
        }

        if missing_info["original_missing"] > 0:
            for col in X.columns:
                if X[col].isnull().sum() > 0:
                    if pd.api.types.is_numeric_dtype(X[col]):
                        # 수치형: 중위값
                        median_val = X[col].median()
                        X[col].fillna(median_val, inplace=True)
                        missing_info["strategies"][col] = f"중위값 대체 ({median_val})"
                    else:
                        # 범주형: 최빈값
                        mode_val = X[col].mode()[0] if len(X[col].mode()) > 0 else 'unknown'
                        X[col].fillna(mode_val, inplace=True)
                        missing_info["strategies"][col] = f"최빈값 대체 ({mode_val})"

            missing_info["values_imputed"] = missing_info["original_missing"]

        return missing_info

    def encode_categorical_variables(self, X: pd.DataFrame) -> Dict[str, Any]:
        """범주형 변수 인코딩"""

        encoding_info = {
            "columns_encoded": 0,
            "encoding_methods": {},
            "new_features_created": 0
        }

        categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in categorical_columns:
            unique_count = X[col].nunique()

            if unique_count <= 5:
                # One-hot 인코딩
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = X.drop(columns=[col])
                X = pd.concat([X, dummies], axis=1)
                encoding_info["encoding_methods"][col] = f"One-hot 인코딩 ({unique_count-1}개 특성 생성)"
                encoding_info["new_features_created"] += unique_count - 1
            else:
                # Label 인코딩
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoding_info["encoding_methods"][col] = "Label 인코딩"

            encoding_info["columns_encoded"] += 1

        return encoding_info

    def perform_feature_engineering(self, X: np.ndarray, y: np.ndarray, task_type: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """특성 엔지니어링"""

        feature_eng_info = {
            "techniques_applied": [],
            "original_features": X.shape[1],
            "final_features": 0,
            "feature_selection": {}
        }

        X_engineered = X.copy()

        # 특성 선택
        if X.shape[1] > 20:  # 특성이 많은 경우에만 적용
            selector = SelectKBest(k=min(15, X.shape[1]))
            X_engineered = selector.fit_transform(X_engineered, y)

            selected_features = selector.get_support(indices=True)
            feature_eng_info["feature_selection"] = {
                "method": "SelectKBest",
                "features_selected": len(selected_features),
                "selected_indices": selected_features.tolist()
            }
            feature_eng_info["techniques_applied"].append("특성 선택")

        feature_eng_info["final_features"] = X_engineered.shape[1]

        return X_engineered, feature_eng_info

    def optimize_models(self, X_train, X_test, y_train, y_test, task_type: str,
                       time_budget: int, optimization_metric: str, max_models: int) -> Dict[str, Any]:
        """모델 최적화"""

        # 모델 후보군 정의
        if task_type == "classification":
            models = {
                "random_forest": RandomForestClassifier(random_state=42),
                "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
                "gradient_boosting": GradientBoostingClassifier(random_state=42),
                "svm": SVC(random_state=42, probability=True),
                "knn": KNeighborsClassifier()
            }
            scoring_metric = "accuracy" if optimization_metric == "auto" else optimization_metric
        else:
            models = {
                "random_forest": RandomForestRegressor(random_state=42),
                "linear_regression": LinearRegression(),
                "gradient_boosting": GradientBoostingRegressor(random_state=42),
                "ridge": Ridge(random_state=42),
                "knn": KNeighborsRegressor()
            }
            scoring_metric = "r2" if optimization_metric == "auto" else optimization_metric

        # 하이퍼파라미터 그리드
        param_grids = self.get_hyperparameter_grids()

        model_results = {}
        time_per_model = time_budget // min(max_models, len(models))

        for model_name, model in list(models.items())[:max_models]:
            print(f"  - {model_name} 최적화 중...", file=sys.stderr)

            try:
                # 빠른 하이퍼파라미터 검색
                param_grid = param_grids.get(model_name, {})

                if param_grid:
                    # RandomizedSearch로 빠른 최적화
                    search = RandomizedSearchCV(
                        model, param_grid, n_iter=min(20, len(param_grid)),
                        cv=3, scoring=scoring_metric, random_state=42,
                        n_jobs=-1
                    )
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                else:
                    # 기본 모델 사용
                    best_model = model
                    best_model.fit(X_train, y_train)

                # 성능 평가
                y_pred = best_model.predict(X_test)

                if task_type == "classification":
                    score = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    metrics = {"accuracy": score, "f1_score": f1}
                else:
                    score = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    metrics = {"r2_score": score, "mse": mse}

                model_results[model_name] = {
                    "model": best_model,
                    "score": float(score),
                    "metrics": {k: float(v) for k, v in metrics.items()},
                    "hyperparameters": best_model.get_params() if hasattr(best_model, 'get_params') else {}
                }

            except Exception as e:
                print(f"    {model_name} 최적화 실패: {str(e)}", file=sys.stderr)
                continue

        return model_results

    def get_hyperparameter_grids(self) -> Dict[str, Dict]:
        """하이퍼파라미터 그리드 정의"""

        return {
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10]
            },
            "logistic_regression": {
                "C": [0.1, 1, 10],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "lbfgs"]
            },
            "gradient_boosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            },
            "svm": {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"]
            },
            "ridge": {
                "alpha": [0.1, 1, 10, 100]
            },
            "knn": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"]
            }
        }

    def create_ensemble(self, model_results: Dict, X_train, X_test, y_train, y_test, task_type: str) -> Dict[str, Any]:
        """앙상블 모델 생성"""

        if len(model_results) < 2:
            return {"error": "앙상블을 위한 충분한 모델이 없습니다"}

        try:
            # 상위 성능 모델들 선택
            sorted_models = sorted(model_results.items(), key=lambda x: x[1]["score"], reverse=True)
            top_models = sorted_models[:min(3, len(sorted_models))]

            models_for_ensemble = [(name, info["model"]) for name, info in top_models]

            if task_type == "classification":
                ensemble = VotingClassifier(models_for_ensemble, voting='soft')
            else:
                ensemble = VotingRegressor(models_for_ensemble)

            ensemble.fit(X_train, y_train)
            y_pred = ensemble.predict(X_test)

            if task_type == "classification":
                score = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                metrics = {"accuracy": score, "f1_score": f1}
            else:
                score = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                metrics = {"r2_score": score, "mse": mse}

            return {
                "ensemble_model": ensemble,
                "component_models": [name for name, _ in models_for_ensemble],
                "score": float(score),
                "metrics": {k: float(v) for k, v in metrics.items()},
                "improvement_over_best_single": float(score - max([info["score"] for info in model_results.values()]))
            }

        except Exception as e:
            return {"error": f"앙상블 생성 실패: {str(e)}"}

    def select_best_model(self, model_results: Dict, ensemble_result: Dict, task_type: str) -> Dict[str, Any]:
        """최고 성능 모델 선택"""

        all_results = model_results.copy()
        if "ensemble_model" in ensemble_result:
            all_results["ensemble"] = {
                "score": ensemble_result["score"],
                "metrics": ensemble_result["metrics"],
                "model": ensemble_result["ensemble_model"]
            }

        if not all_results:
            return {"error": "선택할 모델이 없습니다"}

        best_model_name = max(all_results.keys(), key=lambda x: all_results[x]["score"])
        best_model_info = all_results[best_model_name]

        return {
            "best_model_name": best_model_name,
            "best_score": best_model_info["score"],
            "best_metrics": best_model_info["metrics"],
            "model_object": best_model_info["model"],
            "is_ensemble": best_model_name == "ensemble"
        }

    def interpret_model(self, best_model_info: Dict, feature_names: Optional[List[str]]) -> Dict[str, Any]:
        """모델 해석"""

        interpretation = {
            "model_type": type(best_model_info["model_object"]).__name__,
            "interpretability_level": "unknown",
            "feature_importance": None,
            "model_complexity": "unknown"
        }

        try:
            model = best_model_info["model_object"]

            # 특성 중요도 추출
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                interpretation["feature_importance"] = {
                    "values": importances.tolist(),
                    "feature_names": feature_names[:len(importances)] if feature_names else None
                }
                interpretation["interpretability_level"] = "high"

            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if coef.ndim > 1:
                    coef = coef[0]
                interpretation["feature_importance"] = {
                    "values": coef.tolist(),
                    "feature_names": feature_names[:len(coef)] if feature_names else None
                }
                interpretation["interpretability_level"] = "high"

            else:
                interpretation["interpretability_level"] = "low"

            # 모델 복잡도
            if hasattr(model, 'n_estimators'):
                interpretation["model_complexity"] = f"앙상블 ({model.n_estimators} 추정기)"
            elif hasattr(model, 'support_vectors_'):
                interpretation["model_complexity"] = f"SVM ({len(model.support_vectors_)} 서포트 벡터)"
            else:
                interpretation["model_complexity"] = "단순 모델"

        except Exception as e:
            interpretation["error"] = str(e)

        return interpretation

    def generate_automl_recommendations(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
        """AutoML 권고사항 생성"""

        recommendations = []

        # 최고 모델 성능 평가
        if "best_model" in results and "best_score" in results["best_model"]:
            best_score = results["best_model"]["best_score"]

            if best_score < 0.7:
                recommendations.append({
                    "type": "performance",
                    "priority": "high",
                    "issue": f"모델 성능이 낮습니다 ({best_score:.3f})",
                    "recommendation": "데이터 품질 향상 또는 더 많은 데이터 수집 필요",
                    "action": "특성 엔지니어링 강화, 데이터 전처리 개선"
                })

        # 특성 수 대비 데이터 수
        if "preprocessing_info" in results and "feature_count" in results["preprocessing_info"]:
            feature_count = results["preprocessing_info"]["feature_count"]["final"]

            recommendations.append({
                "type": "feature_optimization",
                "priority": "medium",
                "issue": f"특성 수: {feature_count}개",
                "recommendation": "특성 선택 또는 차원 축소 고려",
                "action": "PCA, 특성 중요도 기반 선택"
            })

        # 앙상블 효과
        if "ensemble_result" in results and "improvement_over_best_single" in results["ensemble_result"]:
            improvement = results["ensemble_result"]["improvement_over_best_single"]

            if improvement > 0.01:
                recommendations.append({
                    "type": "ensemble_success",
                    "priority": "info",
                    "issue": f"앙상블로 {improvement:.3f} 성능 향상",
                    "recommendation": "앙상블 모델 사용 권장",
                    "action": "프로덕션에서 앙상블 모델 배포"
                })

        return recommendations

def main():
    """메인 실행 함수"""
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

        # AutoML 옵션
        target_column = params['target_column']
        task_type = params.get('task_type', 'auto')
        time_budget = params.get('time_budget', 300)
        optimization_metric = params.get('optimization_metric', 'auto')
        include_ensembles = params.get('include_ensembles', True)
        max_models = params.get('max_models', 10)

        # 데이터 기본 정보 추출
        data_info = get_data_info(df)

        # AutoML 실행
        automl_pipeline = AutoMLPipeline()
        automl_result = automl_pipeline.run_automl(
            df, target_column, task_type, time_budget,
            optimization_metric, include_ensembles, max_models
        )

        if not automl_result.get('success', False):
            error_result = {
                "success": False,
                "error": automl_result.get('error', 'AutoML 파이프라인 실패'),
                "analysis_type": "automl_pipeline"
            }
            output_results(error_result)
            return

        # 분석 결과 통합
        analysis_results = {
            "automl_pipeline": automl_result,
            "automl_summary": {
                "pipeline_steps": len(automl_result.get('pipeline_steps', [])),
                "models_trained": len(automl_result.get('model_results', {})),
                "best_model": automl_result.get('best_model', {}).get('best_model_name', 'None'),
                "best_score": automl_result.get('best_model', {}).get('best_score', 0.0),
                "recommendations_count": len(automl_result.get('recommendations', []))
            }
        }

        # 요약 생성
        steps_count = len(automl_result.get('pipeline_steps', []))
        models_count = len(automl_result.get('model_results', {}))
        best_model = automl_result.get('best_model', {}).get('best_model_name', 'None')
        best_score = automl_result.get('best_model', {}).get('best_score', 0.0)
        summary = f"AutoML 파이프라인 완료 - {steps_count}단계 수행, {models_count}개 모델 훈련, 최고 성능: {best_model} ({best_score:.3f})"

        # 표준화된 결과 생성
        final_result = create_analysis_result(
            analysis_type="automl_pipeline",
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
            "analysis_type": "automl_pipeline",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        output_results(error_result)
        sys.exit(1)

if __name__ == "__main__":
    main()