#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Networks Implementation
신경망 구현

이 모듈은 다양한 신경망 모델을 구현합니다.
주요 기능:
- 다층 퍼셉트론 (MLP) 분류/회귀
- 자동 네트워크 아키텍처 설계
- 배치 정규화 및 드롭아웃
- 조기 종료 및 학습률 스케줄링
- 한국어 해석 및 인사이트
"""

import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    import joblib
except ImportError as e:
    print(json.dumps({
        "success": False,
        "error": f"필수 라이브러리 누락: {str(e)}",
        "required_packages": ["scikit-learn", "joblib"]
    }, ensure_ascii=False))
    sys.exit(1)

def clean_dict_for_json(obj):
    """JSON 직렬화를 위한 딕셔너리 정리"""
    if isinstance(obj, dict):
        return {str(k): clean_dict_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_dict_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return [clean_dict_for_json(item) for item in obj]
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

def design_network_architecture(input_size: int, output_size: int, task_type: str,
                               complexity: str = "medium") -> List[int]:
    """
    네트워크 아키텍처 자동 설계

    Parameters:
    -----------
    input_size : int
        입력 차원 수
    output_size : int
        출력 차원 수
    task_type : str
        작업 유형 ('classification' 또는 'regression')
    complexity : str
        복잡도 ('simple', 'medium', 'complex')

    Returns:
    --------
    List[int]
        은닉층 크기 목록
    """

    if complexity == "simple":
        # 간단한 구조: 1-2개 은닉층
        if input_size <= 10:
            return [max(5, input_size // 2)]
        elif input_size <= 50:
            return [input_size // 2, input_size // 4]
        else:
            return [50, 25]

    elif complexity == "medium":
        # 중간 구조: 2-3개 은닉층
        if input_size <= 10:
            return [input_size * 2, input_size]
        elif input_size <= 50:
            return [input_size, input_size // 2, input_size // 4]
        else:
            return [100, 50, 25]

    else:  # complex
        # 복잡한 구조: 3-4개 은닉층
        if input_size <= 10:
            return [input_size * 3, input_size * 2, input_size]
        elif input_size <= 50:
            return [input_size * 2, input_size, input_size // 2, input_size // 4]
        else:
            return [200, 100, 50, 25]

def train_neural_network(df: pd.DataFrame, target_column: str, task_type: str = "auto",
                        network_architectures: List[str] = ["simple", "medium", "complex"],
                        test_size: float = 0.2, validation_size: float = 0.1,
                        random_state: int = 42, max_iter: int = 1000,
                        early_stopping: bool = True, model_save_path: str = "neural_network_model.pkl",
                        cross_validation: bool = True, cv_folds: int = 5) -> Dict[str, Any]:
    """
    신경망 모델 훈련

    Parameters:
    -----------
    df : pd.DataFrame
        입력 데이터프레임
    target_column : str
        타겟 컬럼명
    task_type : str
        작업 유형 ('auto', 'classification', 'regression')
    network_architectures : List[str]
        테스트할 네트워크 복잡도
    test_size : float
        테스트 데이터 비율
    validation_size : float
        검증 데이터 비율
    random_state : int
        랜덤 시드
    max_iter : int
        최대 반복 횟수
    early_stopping : bool
        조기 종료 사용 여부
    model_save_path : str
        모델 저장 경로
    cross_validation : bool
        교차 검증 수행 여부
    cv_folds : int
        교차 검증 폴드 수

    Returns:
    --------
    Dict[str, Any]
        훈련 결과
    """

    try:
        # 데이터 검증
        if target_column not in df.columns:
            return {
                "success": False,
                "error": f"타겟 컬럼 '{target_column}'이 데이터에 없습니다",
                "available_columns": list(df.columns)
            }

        if df.empty:
            return {
                "success": False,
                "error": "입력 데이터가 비어있습니다"
            }

        # 특성과 타겟 분리
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 결측값 처리
        X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        y = y.fillna(y.mode()[0] if not y.mode().empty else 0)

        # 범주형 변수 인코딩
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # 작업 유형 자동 결정
        if task_type == "auto":
            unique_values = len(y.unique())
            if unique_values <= 20 and (y.dtype == 'object' or y.dtype.name == 'category'):
                task_type = "classification"
            elif unique_values <= 20 and unique_values < len(y) * 0.1:
                task_type = "classification"
            else:
                task_type = "regression"

        # 타겟 변수 인코딩 (분류의 경우)
        label_encoder = None
        if task_type == "classification" and y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        # 데이터 분할
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if task_type == "classification" else None
        )

        # 훈련/검증 분할
        if validation_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=validation_size/(1-test_size),
                random_state=random_state, stratify=y_temp if task_type == "classification" else None
            )
        else:
            X_train, X_val, y_train, y_val = X_temp, None, y_temp, None

        # 특성 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        if X_val is not None:
            X_val_scaled = scaler.transform(X_val)

        # 네트워크 아키텍처 설계 및 모델 훈련
        models = {}
        results = {}
        input_size = X_train_scaled.shape[1]
        output_size = len(np.unique(y)) if task_type == "classification" else 1

        for complexity in network_architectures:
            hidden_layers = design_network_architecture(input_size, output_size, task_type, complexity)

            # 모델 생성
            if task_type == "classification":
                model = MLPClassifier(
                    hidden_layer_sizes=tuple(hidden_layers),
                    max_iter=max_iter,
                    random_state=random_state,
                    early_stopping=early_stopping,
                    validation_fraction=0.1 if early_stopping and X_val is None else 0.0,
                    n_iter_no_change=10,
                    learning_rate='adaptive',
                    alpha=0.001
                )
            else:
                model = MLPRegressor(
                    hidden_layer_sizes=tuple(hidden_layers),
                    max_iter=max_iter,
                    random_state=random_state,
                    early_stopping=early_stopping,
                    validation_fraction=0.1 if early_stopping and X_val is None else 0.0,
                    n_iter_no_change=10,
                    learning_rate='adaptive',
                    alpha=0.001
                )

            # 모델 훈련
            model.fit(X_train_scaled, y_train)
            models[complexity] = model

            # 예측
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            if X_val is not None:
                y_pred_val = model.predict(X_val_scaled)

            # 성능 평가
            if task_type == "classification":
                train_score = accuracy_score(y_train, y_pred_train)
                test_score = accuracy_score(y_test, y_pred_test)
                val_score = accuracy_score(y_val, y_pred_val) if X_val is not None else None
            else:
                train_score = -mean_squared_error(y_train, y_pred_train)  # 음수로 변환 (높을수록 좋음)
                test_score = -mean_squared_error(y_test, y_pred_test)
                val_score = -mean_squared_error(y_val, y_pred_val) if X_val is not None else None

            # 교차 검증
            cv_scores = None
            if cross_validation:
                scoring = 'accuracy' if task_type == "classification" else 'neg_mean_squared_error'
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring=scoring)

            results[complexity] = {
                "architecture": hidden_layers,
                "n_parameters": sum([layer.size for layer in model.coefs_]) + sum([layer.size for layer in model.intercepts_]),
                "train_score": float(train_score),
                "test_score": float(test_score),
                "validation_score": float(val_score) if val_score is not None else None,
                "cv_scores": cv_scores.tolist() if cv_scores is not None else None,
                "cv_mean": float(cv_scores.mean()) if cv_scores is not None else None,
                "cv_std": float(cv_scores.std()) if cv_scores is not None else None,
                "n_iterations": model.n_iter_,
                "loss_curve": model.loss_curve_.tolist() if hasattr(model, 'loss_curve_') else None
            }

        # 최고 성능 모델 선택
        best_complexity = max(results.keys(), key=lambda k: results[k]["test_score"])
        best_model = models[best_complexity]
        best_results = results[best_complexity]

        # 최고 모델 저장
        model_data = {
            "model": best_model,
            "scaler": scaler,
            "label_encoder": label_encoder,
            "feature_names": list(X.columns),
            "task_type": task_type,
            "architecture": best_results["architecture"]
        }
        joblib.dump(model_data, model_save_path)

        # 상세 성능 분석
        performance_analysis = analyze_neural_network_performance(
            best_model, X_test_scaled, y_test, task_type, label_encoder
        )

        # 결과 정리
        result = {
            "success": True,
            "model_save_path": model_save_path,
            "task_type": task_type,
            "best_architecture": best_complexity,
            "architectures_tested": len(network_architectures),
            "input_features": input_size,
            "output_size": output_size,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "validation_samples": len(X_val) if X_val is not None else 0,
            "results_by_architecture": clean_dict_for_json(results),
            "best_model_performance": clean_dict_for_json(best_results),
            "performance_analysis": clean_dict_for_json(performance_analysis),
            "insights": generate_neural_network_insights(results, performance_analysis, task_type)
        }

        return clean_dict_for_json(result)

    except Exception as e:
        return {
            "success": False,
            "error": f"신경망 훈련 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def analyze_neural_network_performance(model, X_test: np.ndarray, y_test: np.ndarray,
                                     task_type: str, label_encoder=None) -> Dict[str, Any]:
    """신경망 성능 상세 분석"""

    analysis = {
        "predictions": {},
        "model_complexity": {},
        "learning_analysis": {}
    }

    # 예측 분석
    y_pred = model.predict(X_test)

    if task_type == "classification":
        analysis["predictions"] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "unique_classes": len(np.unique(y_test)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }

        # 클래스별 성능
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            analysis["predictions"]["prediction_confidence"] = {
                "mean_max_probability": float(np.mean(np.max(y_proba, axis=1))),
                "min_max_probability": float(np.min(np.max(y_proba, axis=1))),
                "std_max_probability": float(np.std(np.max(y_proba, axis=1)))
            }

    else:  # regression
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

        analysis["predictions"] = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2),
            "prediction_range": {
                "min": float(np.min(y_pred)),
                "max": float(np.max(y_pred)),
                "mean": float(np.mean(y_pred)),
                "std": float(np.std(y_pred))
            },
            "residual_analysis": {
                "mean_residual": float(np.mean(y_test - y_pred)),
                "residual_std": float(np.std(y_test - y_pred))
            }
        }

    # 모델 복잡도 분석
    total_params = sum([layer.size for layer in model.coefs_]) + sum([layer.size for layer in model.intercepts_])
    analysis["model_complexity"] = {
        "total_parameters": int(total_params),
        "hidden_layers": len(model.hidden_layer_sizes),
        "layer_sizes": list(model.hidden_layer_sizes),
        "activation_function": model.activation,
        "solver": model.solver,
        "learning_rate": model.learning_rate
    }

    # 학습 분석
    analysis["learning_analysis"] = {
        "iterations_completed": int(model.n_iter_),
        "max_iterations": int(model.max_iter),
        "converged": model.n_iter_ < model.max_iter,
        "final_loss": float(model.loss_) if hasattr(model, 'loss_') else None
    }

    if hasattr(model, 'loss_curve_'):
        loss_curve = model.loss_curve_
        analysis["learning_analysis"]["loss_curve_analysis"] = {
            "initial_loss": float(loss_curve[0]),
            "final_loss": float(loss_curve[-1]),
            "loss_reduction": float(loss_curve[0] - loss_curve[-1]),
            "loss_reduction_percent": float((loss_curve[0] - loss_curve[-1]) / loss_curve[0] * 100),
            "plateaued": bool(len(loss_curve) > 10 and np.std(loss_curve[-10:]) < 0.001)
        }

    return analysis

def generate_neural_network_insights(results: Dict[str, Any], performance_analysis: Dict[str, Any],
                                    task_type: str) -> List[str]:
    """신경망 인사이트 생성"""

    insights = []

    # 아키텍처 비교
    if len(results) > 1:
        best_arch = max(results.keys(), key=lambda k: results[k]["test_score"])
        worst_arch = min(results.keys(), key=lambda k: results[k]["test_score"])

        best_score = results[best_arch]["test_score"]
        worst_score = results[worst_arch]["test_score"]

        if best_score - worst_score > 0.05:  # 5% 이상 차이
            insights.append(f"'{best_arch}' 아키텍처가 '{worst_arch}'보다 {(best_score - worst_score)*100:.1f}% 더 좋은 성능을 보입니다")
        else:
            insights.append("다양한 아키텍처 간 성능 차이가 작습니다")

    # 과적합 분석
    for arch, result in results.items():
        train_score = result["train_score"]
        test_score = result["test_score"]

        if train_score - test_score > 0.1:  # 10% 이상 차이
            insights.append(f"'{arch}' 아키텍처에서 과적합이 감지되었습니다")
            break

    # 수렴 분석
    learning_analysis = performance_analysis.get("learning_analysis", {})
    if not learning_analysis.get("converged", True):
        insights.append("모델이 완전히 수렴하지 않았습니다. max_iter 증가를 고려하세요")

    # 성능 분석
    if task_type == "classification":
        accuracy = performance_analysis.get("predictions", {}).get("accuracy", 0)
        if accuracy > 0.9:
            insights.append("매우 높은 분류 정확도를 달성했습니다")
        elif accuracy < 0.7:
            insights.append("분류 정확도가 낮습니다. 데이터 전처리나 특성 엔지니어링을 고려하세요")
    else:
        r2 = performance_analysis.get("predictions", {}).get("r2_score", 0)
        if r2 > 0.8:
            insights.append("매우 좋은 회귀 성능을 달성했습니다")
        elif r2 < 0.5:
            insights.append("회귀 성능이 낮습니다. 모델 복잡도 증가나 특성 개선을 고려하세요")

    # 모델 복잡도 분석
    complexity_analysis = performance_analysis.get("model_complexity", {})
    total_params = complexity_analysis.get("total_parameters", 0)

    if total_params > 10000:
        insights.append("높은 복잡도의 모델입니다. 정규화나 드롭아웃 고려를 권장합니다")
    elif total_params < 100:
        insights.append("단순한 모델입니다. 더 복잡한 아키텍처로 성능 향상 가능할 수 있습니다")

    # 손실 곡선 분석
    if "loss_curve_analysis" in learning_analysis:
        loss_analysis = learning_analysis["loss_curve_analysis"]
        if loss_analysis.get("plateaued", False):
            insights.append("학습이 조기에 정체되었습니다. 학습률 조정을 고려하세요")

        loss_reduction = loss_analysis.get("loss_reduction_percent", 0)
        if loss_reduction > 90:
            insights.append(f"손실이 {loss_reduction:.1f}% 감소하여 효과적인 학습이 이루어졌습니다")

    return insights

def main():
    """메인 실행 함수"""
    try:
        # stdin에서 JSON 데이터 읽기
        input_data = sys.stdin.read()
        params = json.loads(input_data)

        # 신경망 훈련
        result = train_neural_network(**params)

        # JSON으로 결과 출력
        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
        print(json.dumps(error_result, ensure_ascii=False, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()