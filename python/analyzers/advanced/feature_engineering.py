#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Engineering Module
특성 엔지니어링 모듈

이 모듈은 데이터의 특성을 변환하고 새로운 특성을 생성합니다.
주요 기능:
- 스케일링 및 정규화 (StandardScaler, MinMaxScaler, RobustScaler)
- 범주형 변수 인코딩 (One-hot, Label, Ordinal encoding)
- 다항식 특성 및 상호작용 항목 생성
- 차원 축소 (PCA, SelectKBest)
- 특성 선택 및 중요도 분석
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
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
            elif isinstance(obj, type) or str(type(obj)).startswith("<class 'numpy."):  # Handle numpy dtypes
                return str(obj)
            elif hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            elif hasattr(obj, 'dtype') and hasattr(obj, 'name'):  # numpy dtypes
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
                    # 키를 문자열로 변환
                    clean_key = str(key) if not isinstance(key, (str, int, float, bool, type(None))) else key
                    # 값을 재귀적으로 정리
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
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression, chi2
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def perform_feature_engineering(df: pd.DataFrame, operations: List[str],
                               target_column: Optional[str] = None) -> Dict[str, Any]:
    """
    포괄적인 특성 엔지니어링 수행

    Parameters:
    -----------
    df : pd.DataFrame
        변환할 데이터프레임
    operations : List[str]
        수행할 작업들 ['scaling', 'encoding', 'polynomial', 'selection', 'pca']
    target_column : str, optional
        타겟 변수 컬럼명 (특성 선택 시 필요)

    Returns:
    --------
    Dict[str, Any]
        특성 엔지니어링 결과
    """

    if not SKLEARN_AVAILABLE:
        return {
            "error": "scikit-learn이 설치되지 않았습니다",
            "required_package": "scikit-learn"
        }

    if df.empty:
        return {
            "error": "데이터프레임이 비어있습니다",
            "data_shape": df.shape
        }

    try:
        results = {
            "success": True,
            "original_shape": df.shape,
            "operations_performed": [],
            "transformations": {}
        }

        # 작업용 데이터프레임 복사
        df_transformed = df.copy()

        # 타겟 변수 분리 (있는 경우)
        y = None
        if target_column and target_column in df.columns:
            y = df[target_column].copy()
            df_transformed = df_transformed.drop(columns=[target_column])

        # 1. 스케일링
        if 'scaling' in operations:
            scaling_result = apply_scaling(df_transformed)
            results['transformations']['scaling'] = scaling_result
            results['operations_performed'].append('scaling')

        # 2. 인코딩
        if 'encoding' in operations:
            encoding_result = apply_encoding(df_transformed)
            results['transformations']['encoding'] = encoding_result
            results['operations_performed'].append('encoding')

            # 인코딩된 데이터프레임으로 업데이트
            if 'encoded_dataframe' in encoding_result:
                df_transformed = encoding_result['encoded_dataframe']

        # 3. 다항식 특성
        if 'polynomial' in operations:
            polynomial_result = apply_polynomial_features(df_transformed)
            results['transformations']['polynomial'] = polynomial_result
            results['operations_performed'].append('polynomial')

        # 4. 특성 선택
        if 'selection' in operations and target_column and y is not None:
            selection_result = apply_feature_selection(df_transformed, y, target_column)
            results['transformations']['selection'] = selection_result
            results['operations_performed'].append('selection')

        # 5. PCA
        if 'pca' in operations:
            pca_result = apply_pca(df_transformed)
            results['transformations']['pca'] = pca_result
            results['operations_performed'].append('pca')

        # 변환 요약
        results['transformation_summary'] = generate_transformation_summary(df, df_transformed, results)

        # 권고사항
        results['recommendations'] = generate_engineering_recommendations(results, df)

        return results

    except Exception as e:
        return {
            "error": f"특성 엔지니어링 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def apply_scaling(df: pd.DataFrame) -> Dict[str, Any]:
    """수치형 변수 스케일링 적용"""

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_columns:
        return {"error": "스케일링할 수치형 컬럼이 없습니다"}

    try:
        scaling_results = {
            "numeric_columns": numeric_columns,
            "scaling_methods": {}
        }

        # 원본 통계
        original_stats = {
            "mean": {str(k): float(v) for k, v in df[numeric_columns].mean().to_dict().items()},
            "std": {str(k): float(v) for k, v in df[numeric_columns].std().to_dict().items()},
            "min": {str(k): float(v) for k, v in df[numeric_columns].min().to_dict().items()},
            "max": {str(k): float(v) for k, v in df[numeric_columns].max().to_dict().items()}
        }

        # StandardScaler
        scaler_standard = StandardScaler()
        X_standard = scaler_standard.fit_transform(df[numeric_columns])
        df_standard = pd.DataFrame(X_standard, columns=[f"{col}_standard" for col in numeric_columns])

        scaling_results["scaling_methods"]["standard"] = {
            "description": "평균 0, 분산 1로 정규화",
            "new_columns": df_standard.columns.tolist(),
            "statistics": {
                "mean": {str(k): float(v) for k, v in df_standard.mean().to_dict().items()},
                "std": {str(k): float(v) for k, v in df_standard.std().to_dict().items()}
            }
        }

        # MinMaxScaler
        scaler_minmax = MinMaxScaler()
        X_minmax = scaler_minmax.fit_transform(df[numeric_columns])
        df_minmax = pd.DataFrame(X_minmax, columns=[f"{col}_minmax" for col in numeric_columns])

        scaling_results["scaling_methods"]["minmax"] = {
            "description": "0-1 범위로 정규화",
            "new_columns": df_minmax.columns.tolist(),
            "statistics": {
                "min": {str(k): float(v) for k, v in df_minmax.min().to_dict().items()},
                "max": {str(k): float(v) for k, v in df_minmax.max().to_dict().items()}
            }
        }

        # RobustScaler
        scaler_robust = RobustScaler()
        X_robust = scaler_robust.fit_transform(df[numeric_columns])
        df_robust = pd.DataFrame(X_robust, columns=[f"{col}_robust" for col in numeric_columns])

        scaling_results["scaling_methods"]["robust"] = {
            "description": "중위값과 IQR을 사용한 정규화 (이상치에 강함)",
            "new_columns": df_robust.columns.tolist(),
            "statistics": {
                "median": {str(k): float(v) for k, v in df_robust.median().to_dict().items()},
                "iqr": {str(k): float(v) for k, v in (df_robust.quantile(0.75) - df_robust.quantile(0.25)).to_dict().items()}
            }
        }

        scaling_results["original_statistics"] = original_stats
        scaling_results["scaling_comparison"] = compare_scaling_methods(df[numeric_columns],
                                                                       df_standard, df_minmax, df_robust)

        return scaling_results

    except Exception as e:
        return {"error": f"스케일링 실패: {str(e)}"}

def apply_encoding(df: pd.DataFrame) -> Dict[str, Any]:
    """범주형 변수 인코딩 적용"""

    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not categorical_columns:
        return {"error": "인코딩할 범주형 컬럼이 없습니다"}

    try:
        encoding_results = {
            "categorical_columns": categorical_columns,
            "encoding_methods": {},
            "encoded_dataframe": df.copy()
        }

        df_encoded = df.copy()

        for col in categorical_columns:
            # 기본 정보
            unique_values = df[col].unique()
            n_unique = len(unique_values)

            col_results = {
                "original_unique_count": n_unique,
                "unique_values": unique_values.tolist()[:10],  # 처음 10개만
                "encoding_applied": []
            }

            # Label Encoding
            try:
                le = LabelEncoder()
                encoded_values = le.fit_transform(df[col].fillna('missing'))
                df_encoded[f"{col}_label"] = encoded_values

                col_results["encoding_applied"].append({
                    "method": "label_encoding",
                    "new_column": f"{col}_label",
                    "description": "정수로 인코딩 (0, 1, 2, ...)",
                    "mapping": dict(zip(le.classes_, le.transform(le.classes_)))
                })
            except Exception as e:
                col_results["label_encoding_error"] = str(e)

            # One-Hot Encoding (카테고리 수가 너무 많지 않은 경우)
            if n_unique <= 10:
                try:
                    dummies = pd.get_dummies(df[col], prefix=col, prefix_sep='_')
                    for dummy_col in dummies.columns:
                        df_encoded[dummy_col] = dummies[dummy_col]

                    col_results["encoding_applied"].append({
                        "method": "one_hot_encoding",
                        "new_columns": dummies.columns.tolist(),
                        "description": f"{n_unique}개의 더미 변수 생성"
                    })
                except Exception as e:
                    col_results["one_hot_encoding_error"] = str(e)
            else:
                col_results["one_hot_skipped"] = f"카테고리가 너무 많음 ({n_unique}개)"

            encoding_results["encoding_methods"][col] = col_results

        # 인코딩된 DataFrame 저장
        encoding_results["encoded_dataframe"] = df_encoded
        encoding_results["final_shape"] = df_encoded.shape
        encoding_results["new_columns_count"] = len(df_encoded.columns) - len(df.columns)

        return encoding_results

    except Exception as e:
        return {"error": f"인코딩 실패: {str(e)}"}

def apply_polynomial_features(df: pd.DataFrame, degree: int = 2, max_features: int = 100) -> Dict[str, Any]:
    """다항식 특성 생성"""

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_columns) < 2:
        return {"error": "다항식 특성 생성을 위해 최소 2개의 수치형 컬럼이 필요합니다"}

    try:
        # 메모리 효율성을 위해 컬럼 수 제한
        selected_cols = numeric_columns[:5] if len(numeric_columns) > 5 else numeric_columns

        poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)
        X_poly = poly.fit_transform(df[selected_cols])

        # 특성명 생성
        feature_names = poly.get_feature_names_out(selected_cols)

        # 특성이 너무 많으면 제한
        if len(feature_names) > max_features:
            feature_names = feature_names[:max_features]
            X_poly = X_poly[:, :max_features]

        df_poly = pd.DataFrame(X_poly, columns=feature_names)

        polynomial_results = {
            "original_features": selected_cols,
            "degree": degree,
            "original_feature_count": len(selected_cols),
            "polynomial_feature_count": len(feature_names),
            "new_features": feature_names.tolist(),
            "feature_types": {
                "interaction_features": [name for name in feature_names if ' ' in name and '^' not in name],
                "polynomial_features": [name for name in feature_names if '^' in name],
                "original_features": [name for name in feature_names if name in selected_cols]
            },
            "statistics": {
                "mean": {str(k): float(v) for k, v in df_poly.mean().to_dict().items()},
                "std": {str(k): float(v) for k, v in df_poly.std().to_dict().items()}
            }
        }

        return polynomial_results

    except Exception as e:
        return {"error": f"다항식 특성 생성 실패: {str(e)}"}

def apply_feature_selection(df: pd.DataFrame, y: pd.Series, target_column: str) -> Dict[str, Any]:
    """특성 선택 수행"""

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_columns) < 2:
        return {"error": "특성 선택을 위해 최소 2개의 수치형 컬럼이 필요합니다"}

    try:
        selection_results = {
            "original_features": numeric_columns,
            "target_column": target_column,
            "selection_methods": {}
        }

        # 타겟 변수 타입 결정
        is_classification = y.dtype == 'object' or y.nunique() < 10

        if is_classification:
            # 분류용 특성 선택
            if y.dtype == 'object':
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
            else:
                y_encoded = y.values

            # SelectKBest with chi2 (양수 값만)
            if all(df[numeric_columns].min() >= 0):
                try:
                    selector_chi2 = SelectKBest(chi2, k=min(5, len(numeric_columns)))
                    X_selected = selector_chi2.fit_transform(df[numeric_columns], y_encoded)

                    selected_features = [numeric_columns[i] for i in selector_chi2.get_support(indices=True)]
                    scores = selector_chi2.scores_

                    selection_results["selection_methods"]["chi2"] = {
                        "selected_features": selected_features,
                        "feature_scores": {str(k): float(v) for k, v in zip(numeric_columns, scores.tolist())},
                        "description": "카이제곱 검정 기반 특성 선택"
                    }
                except Exception as e:
                    selection_results["selection_methods"]["chi2"] = {"error": str(e)}

            # SelectKBest with f_classif
            try:
                selector_f = SelectKBest(f_classif, k=min(5, len(numeric_columns)))
                X_selected = selector_f.fit_transform(df[numeric_columns], y_encoded)

                selected_features = [numeric_columns[i] for i in selector_f.get_support(indices=True)]
                scores = selector_f.scores_

                selection_results["selection_methods"]["f_classif"] = {
                    "selected_features": selected_features,
                    "feature_scores": {str(k): float(v) for k, v in zip(numeric_columns, scores.tolist())},
                    "description": "F-통계량 기반 특성 선택 (분류)"
                }
            except Exception as e:
                selection_results["selection_methods"]["f_classif"] = {"error": str(e)}

            # Random Forest Feature Importance
            try:
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(df[numeric_columns], y_encoded)

                importances = rf.feature_importances_
                indices = np.argsort(importances)[::-1]

                selection_results["selection_methods"]["random_forest"] = {
                    "feature_importances": {str(k): float(v) for k, v in zip(numeric_columns, importances.tolist())},
                    "ranked_features": [numeric_columns[i] for i in indices],
                    "top_5_features": [numeric_columns[i] for i in indices[:5]],
                    "description": "Random Forest 특성 중요도"
                }
            except Exception as e:
                selection_results["selection_methods"]["random_forest"] = {"error": str(e)}

        else:
            # 회귀용 특성 선택
            # SelectKBest with f_regression
            try:
                selector_f = SelectKBest(f_regression, k=min(5, len(numeric_columns)))
                X_selected = selector_f.fit_transform(df[numeric_columns], y)

                selected_features = [numeric_columns[i] for i in selector_f.get_support(indices=True)]
                scores = selector_f.scores_

                selection_results["selection_methods"]["f_regression"] = {
                    "selected_features": selected_features,
                    "feature_scores": {str(k): float(v) for k, v in zip(numeric_columns, scores.tolist())},
                    "description": "F-통계량 기반 특성 선택 (회귀)"
                }
            except Exception as e:
                selection_results["selection_methods"]["f_regression"] = {"error": str(e)}

            # Random Forest Feature Importance
            try:
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(df[numeric_columns], y)

                importances = rf.feature_importances_
                indices = np.argsort(importances)[::-1]

                selection_results["selection_methods"]["random_forest"] = {
                    "feature_importances": {str(k): float(v) for k, v in zip(numeric_columns, importances.tolist())},
                    "ranked_features": [numeric_columns[i] for i in indices],
                    "top_5_features": [numeric_columns[i] for i in indices[:5]],
                    "description": "Random Forest 특성 중요도"
                }
            except Exception as e:
                selection_results["selection_methods"]["random_forest"] = {"error": str(e)}

        return selection_results

    except Exception as e:
        return {"error": f"특성 선택 실패: {str(e)}"}

def apply_pca(df: pd.DataFrame, n_components: int = None) -> Dict[str, Any]:
    """PCA 차원 축소 적용"""

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_columns) < 2:
        return {"error": "PCA를 위해 최소 2개의 수치형 컬럼이 필요합니다"}

    try:
        if n_components is None:
            n_components = min(5, len(numeric_columns))

        # 표준화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[numeric_columns])

        # PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # PCA 컴포넌트 DataFrame
        pca_columns = [f"PC{i+1}" for i in range(n_components)]
        df_pca = pd.DataFrame(X_pca, columns=pca_columns)

        pca_results = {
            "original_features": numeric_columns,
            "n_components": n_components,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "components": pca.components_.tolist(),
            "component_names": pca_columns,
            "feature_loadings": {
                f"PC{i+1}": {str(k): float(v) for k, v in zip(numeric_columns, pca.components_[i])}
                for i in range(n_components)
            },
            "statistics": {
                "mean": {str(k): float(v) for k, v in df_pca.mean().to_dict().items()},
                "std": {str(k): float(v) for k, v in df_pca.std().to_dict().items()}
            }
        }

        # 컴포넌트 해석
        component_interpretations = []
        for i in range(n_components):
            loadings = pca.components_[i]
            # 절댓값이 큰 특성들 찾기
            sorted_indices = np.argsort(np.abs(loadings))[::-1]
            top_features = [(numeric_columns[idx], loadings[idx]) for idx in sorted_indices[:3]]

            component_interpretations.append({
                "component": f"PC{i+1}",
                "variance_explained": float(pca.explained_variance_ratio_[i]),
                "top_contributing_features": [
                    {"feature": feat, "loading": float(loading)}
                    for feat, loading in top_features
                ]
            })

        pca_results["component_interpretations"] = component_interpretations

        return pca_results

    except Exception as e:
        return {"error": f"PCA 실패: {str(e)}"}

def compare_scaling_methods(original: pd.DataFrame, standard: pd.DataFrame,
                           minmax: pd.DataFrame, robust: pd.DataFrame) -> Dict[str, Any]:
    """스케일링 방법들 비교"""

    comparison = {
        "variance_comparison": {
            "original": float(original.var().mean()),
            "standard": float(standard.var().mean()),
            "minmax": float(minmax.var().mean()),
            "robust": float(robust.var().mean())
        },
        "range_comparison": {
            "original": float((original.max() - original.min()).mean()),
            "standard": float((standard.max() - standard.min()).mean()),
            "minmax": float((minmax.max() - minmax.min()).mean()),
            "robust": float((robust.max() - robust.min()).mean())
        },
        "recommendations": []
    }

    # 권고사항 생성
    if original.var().max() / original.var().min() > 100:
        comparison["recommendations"].append({
            "issue": "변수별 분산 차이가 큽니다",
            "recommendation": "StandardScaler 또는 RobustScaler 사용 권고"
        })

    if (original.max() - original.min()).max() > 1000:
        comparison["recommendations"].append({
            "issue": "변수별 범위 차이가 큽니다",
            "recommendation": "MinMaxScaler 사용 권고"
        })

    return comparison

def generate_transformation_summary(original_df: pd.DataFrame, transformed_df: pd.DataFrame,
                                   results: Dict[str, Any]) -> Dict[str, Any]:
    """변환 요약 생성"""

    summary = {
        "shape_change": {
            "original": original_df.shape,
            "transformed": transformed_df.shape,
            "columns_added": transformed_df.shape[1] - original_df.shape[1]
        },
        "operations_summary": {},
        "data_quality_metrics": {}
    }

    # 작업별 요약
    for operation in results.get('operations_performed', []):
        if operation in results.get('transformations', {}):
            operation_data = results['transformations'][operation]

            if operation == 'scaling':
                summary["operations_summary"]["scaling"] = {
                    "methods_applied": len(operation_data.get('scaling_methods', {})),
                    "columns_scaled": len(operation_data.get('numeric_columns', []))
                }
            elif operation == 'encoding':
                summary["operations_summary"]["encoding"] = {
                    "columns_encoded": len(operation_data.get('categorical_columns', [])),
                    "new_columns_created": operation_data.get('new_columns_count', 0)
                }
            elif operation == 'polynomial':
                summary["operations_summary"]["polynomial"] = {
                    "degree": operation_data.get('degree', 0),
                    "features_created": operation_data.get('polynomial_feature_count', 0)
                }

    # 데이터 품질 메트릭
    summary["data_quality_metrics"] = {
        "missing_values_original": int(original_df.isnull().sum().sum()),
        "missing_values_transformed": int(transformed_df.isnull().sum().sum()),
        "data_types_original": {str(k): int(v) for k, v in original_df.dtypes.value_counts().to_dict().items()},
        "data_types_transformed": {str(k): int(v) for k, v in transformed_df.dtypes.value_counts().to_dict().items()}
    }

    return summary

def generate_engineering_recommendations(results: Dict[str, Any], original_df: pd.DataFrame) -> List[Dict[str, str]]:
    """특성 엔지니어링 권고사항 생성"""

    recommendations = []

    # 데이터 특성 기반 권고
    numeric_cols = len(original_df.select_dtypes(include=[np.number]).columns)
    categorical_cols = len(original_df.select_dtypes(include=['object', 'category']).columns)

    if numeric_cols > 10:
        recommendations.append({
            "type": "dimensionality_reduction",
            "action": "차원 축소 기법 적용을 고려하세요",
            "reason": f"수치형 변수가 많습니다 ({numeric_cols}개)",
            "methods": "PCA, SelectKBest 등"
        })

    if categorical_cols > 5:
        recommendations.append({
            "type": "encoding_strategy",
            "action": "범주형 변수 인코딩 전략을 신중히 선택하세요",
            "reason": f"범주형 변수가 많습니다 ({categorical_cols}개)",
            "methods": "Target encoding, Frequency encoding 고려"
        })

    # 변환 결과 기반 권고
    if 'transformations' in results:
        if 'polynomial' in results['transformations']:
            poly_features = results['transformations']['polynomial'].get('polynomial_feature_count', 0)
            if poly_features > 50:
                recommendations.append({
                    "type": "feature_selection",
                    "action": "다항식 특성이 너무 많습니다. 특성 선택을 수행하세요",
                    "reason": f"{poly_features}개의 다항식 특성 생성",
                    "methods": "정규화, 특성 중요도 기반 선택"
                })

    if original_df.isnull().sum().sum() > 0:
        recommendations.append({
            "type": "missing_data",
            "action": "결측치 처리를 먼저 수행하세요",
            "reason": f"{original_df.isnull().sum().sum()}개의 결측치 존재",
            "methods": "대체, 제거, 또는 결측치 표시 변수 생성"
        })

    return recommendations

def main():
    """
    메인 실행 함수 - 특성 엔지니어링의 진입점

    표준 입출력을 통해 JSON 데이터를 받아 특성 엔지니어링을 수행하고
    표준화된 형태로 결과를 반환합니다.

    입력 형식:
    - JSON을 통한 데이터 또는 파일 경로
    - 선택적 매개변수: file_path, operations, target_column

    출력 형식:
    - 표준화된 분석 결과 JSON
    - 성공/실패 상태 포함
    - 한국어 해석 및 권고사항
    """
    try:
        # stdin에서 JSON 데이터 읽기
        input_data = sys.stdin.read()
        params = json.loads(input_data)

        # 파일 경로가 제공된 경우 파일에서 데이터 로드
        if 'file_path' in params:
            df = load_data(params['file_path'])
        else:
            # JSON 데이터에서 직접 DataFrame 생성
            if 'data' in params:
                df = pd.DataFrame(params['data'])
            else:
                df = pd.DataFrame(params)

        # 특성 엔지니어링 옵션
        operations = params.get('operations', ['scaling', 'encoding'])
        target_column = params.get('target_column', None)

        # 데이터 기본 정보 추출
        data_info = get_data_info(df)

        # 특성 엔지니어링 수행
        engineering_result = perform_feature_engineering(df, operations, target_column)

        if not engineering_result.get('success', False):
            error_result = {
                "success": False,
                "error": engineering_result.get('error', '특성 엔지니어링 실패'),
                "analysis_type": "feature_engineering"
            }
            output_results(error_result)
            return

        # 분석 결과 통합
        analysis_results = {
            "feature_engineering": engineering_result,
            "engineering_summary": {
                "operations_performed": engineering_result.get('operations_performed', []),
                "original_shape": engineering_result.get('original_shape', [0, 0]),
                "transformations_count": len(engineering_result.get('transformations', {})),
                "success_rate": "100%" if engineering_result.get('success') else "부분적"
            }
        }

        # 요약 생성
        operations_count = len(engineering_result.get('operations_performed', []))
        transformations_count = len(engineering_result.get('transformations', {}))
        summary = f"특성 엔지니어링 완료 - {operations_count}개 작업 수행, {transformations_count}개 변환 적용"

        # 표준화된 결과 생성
        final_result = create_analysis_result(
            analysis_type="feature_engineering",
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
            "analysis_type": "feature_engineering",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        output_results(error_result)
        sys.exit(1)

if __name__ == "__main__":
    main()