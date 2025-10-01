#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Principal Component Analysis (PCA) Module
주성분 분석(PCA) 모듈

이 모듈은 고차원 데이터의 차원을 축소하고 주요 패턴을 추출합니다.
주요 기능:
- 주성분 분석을 통한 차원 축소
- 설명 분산비와 누적 분산비 계산
- 주성분 해석 및 특성 기여도 분석
- 최적 성분 수 결정 및 시각화 권고
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
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
            elif isinstance(obj, type):  # Handle numpy dtypes
                return str(obj)
            elif hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            elif hasattr(obj, 'to_dict') and callable(obj.to_dict):
                return obj.to_dict()
            else:
                return str(obj)

        print(json.dumps(results, ensure_ascii=False, indent=2, default=comprehensive_json_serializer))

    def validate_required_params(params: Dict[str, Any], required: list):
        """필수 매개변수 검증"""
        missing = [param for param in required if param not in params]
        if missing:
            raise ValueError(f"필수 매개변수가 누락됨: {', '.join(missing)}")

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def perform_pca_analysis(df: pd.DataFrame, n_components: Optional[int] = None,
                        standardize: bool = True) -> Dict[str, Any]:
    """
    주성분 분석 수행

    Parameters:
    -----------
    df : pd.DataFrame
        분석할 데이터프레임
    n_components : int, optional
        추출할 주성분 개수 (기본값: min(n_features, n_samples, 10))
    standardize : bool, default=True
        데이터 표준화 여부

    Returns:
    --------
    Dict[str, Any]
        PCA 분석 결과
    """

    if not SKLEARN_AVAILABLE:
        return {
            "error": "scikit-learn이 설치되지 않았습니다",
            "required_package": "scikit-learn"
        }

    # 수치형 컬럼만 선택
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return {
            "error": "분석 가능한 수치형 컬럼이 없습니다",
            "available_columns": list(df.columns),
            "column_types": df.dtypes.astype(str).to_dict()
        }

    if len(numeric_df.columns) < 2:
        return {
            "error": "PCA를 위해 최소 2개의 수치형 컬럼이 필요합니다",
            "available_numeric_columns": len(numeric_df.columns)
        }

    if len(numeric_df) < 2:
        return {
            "error": "PCA를 위해 최소 2개의 데이터 포인트가 필요합니다",
            "data_points": len(numeric_df)
        }

    try:
        # 결측치 처리
        if numeric_df.isnull().any().any():
            numeric_df = numeric_df.fillna(numeric_df.mean())

        results = {
            "success": True,
            "original_features": numeric_df.columns.tolist(),
            "data_points": len(numeric_df),
            "original_dimensions": len(numeric_df.columns),
            "standardized": standardize
        }

        # 최적 컴포넌트 수 결정
        if n_components is None:
            n_components = min(len(numeric_df.columns), len(numeric_df), 10)
        else:
            n_components = min(n_components, len(numeric_df.columns), len(numeric_df))

        # 데이터 전처리
        if standardize:
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(numeric_df)
            results["preprocessing"] = {
                "method": "StandardScaler",
                "original_means": numeric_df.mean().to_dict(),
                "original_stds": numeric_df.std().to_dict()
            }
        else:
            X_processed = numeric_df.values
            results["preprocessing"] = {"method": "None"}

        # PCA 수행
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_processed)

        # 주성분 DataFrame 생성
        component_names = [f"PC{i+1}" for i in range(n_components)]
        df_pca = pd.DataFrame(X_pca, columns=component_names, index=numeric_df.index)

        # 기본 PCA 결과
        results.update({
            "n_components": n_components,
            "explained_variance": pca.explained_variance_.tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "singular_values": pca.singular_values_.tolist(),
            "components": pca.components_.tolist(),
            "component_names": component_names
        })

        # 주성분 해석
        component_interpretations = analyze_components(pca, numeric_df.columns, n_components)
        results["component_interpretations"] = component_interpretations

        # 특성 기여도 분석
        feature_contributions = analyze_feature_contributions(pca, numeric_df.columns, n_components)
        results["feature_contributions"] = feature_contributions

        # 변환된 데이터 통계
        results["transformed_data_stats"] = {
            "means": df_pca.mean().to_dict(),
            "stds": df_pca.std().to_dict(),
            "ranges": (df_pca.max() - df_pca.min()).to_dict()
        }

        # 최적 컴포넌트 수 권고
        optimal_components = recommend_optimal_components(pca.explained_variance_ratio_)
        results["optimal_components_recommendation"] = optimal_components

        # 차원 축소 효과 분석
        dimensionality_analysis = analyze_dimensionality_reduction(
            len(numeric_df.columns), n_components, pca.explained_variance_ratio_
        )
        results["dimensionality_analysis"] = dimensionality_analysis

        # 데이터 품질 평가
        data_quality = evaluate_pca_quality(pca, X_processed, numeric_df)
        results["data_quality_assessment"] = data_quality

        return results

    except Exception as e:
        return {
            "error": f"PCA 분석 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def analyze_components(pca, feature_names: List[str], n_components: int) -> List[Dict[str, Any]]:
    """주성분별 해석 분석"""

    interpretations = []

    for i in range(n_components):
        component_loadings = pca.components_[i]

        # 절댓값 기준 정렬
        sorted_indices = np.argsort(np.abs(component_loadings))[::-1]

        # 상위 기여 특성들
        top_features = []
        for idx in sorted_indices[:min(5, len(feature_names))]:
            feature_name = feature_names[idx]
            loading = component_loadings[idx]
            contribution_pct = (loading**2) / np.sum(component_loadings**2) * 100

            top_features.append({
                "feature": feature_name,
                "loading": float(loading),
                "contribution_percentage": float(contribution_pct),
                "direction": "positive" if loading > 0 else "negative"
            })

        # 주성분 특성 분류
        positive_features = [f["feature"] for f in top_features if f["direction"] == "positive"][:3]
        negative_features = [f["feature"] for f in top_features if f["direction"] == "negative"][:3]

        # 해석 생성
        interpretation = generate_component_interpretation(positive_features, negative_features, i+1)

        interpretations.append({
            "component": f"PC{i+1}",
            "variance_explained": float(pca.explained_variance_ratio_[i]),
            "variance_explained_pct": float(pca.explained_variance_ratio_[i] * 100),
            "top_contributing_features": top_features,
            "interpretation": interpretation,
            "dominant_features": {
                "positive": positive_features,
                "negative": negative_features
            }
        })

    return interpretations

def analyze_feature_contributions(pca, feature_names: List[str], n_components: int) -> Dict[str, Any]:
    """특성별 기여도 분석"""

    contributions = {}

    for i, feature in enumerate(feature_names):
        feature_contributions = []
        total_contribution = 0

        for j in range(n_components):
            loading = pca.components_[j, i]
            variance_weight = pca.explained_variance_ratio_[j]
            weighted_contribution = (loading**2) * variance_weight

            feature_contributions.append({
                "component": f"PC{j+1}",
                "loading": float(loading),
                "variance_weighted_contribution": float(weighted_contribution)
            })

            total_contribution += weighted_contribution

        contributions[feature] = {
            "total_contribution": float(total_contribution),
            "component_contributions": feature_contributions,
            "importance_rank": 0  # Will be filled later
        }

    # 중요도 순위 매기기
    sorted_features = sorted(contributions.keys(),
                           key=lambda x: contributions[x]["total_contribution"],
                           reverse=True)

    for rank, feature in enumerate(sorted_features, 1):
        contributions[feature]["importance_rank"] = rank

    return {
        "feature_contributions": contributions,
        "feature_importance_ranking": sorted_features,
        "most_important_features": sorted_features[:5],
        "least_important_features": sorted_features[-3:]
    }

def recommend_optimal_components(explained_variance_ratio: np.ndarray) -> Dict[str, Any]:
    """최적 주성분 수 권고"""

    cumulative_variance = np.cumsum(explained_variance_ratio)

    # 다양한 기준으로 최적 수 계산
    criteria = {}

    # 80% 분산 설명 기준
    idx_80 = np.argmax(cumulative_variance >= 0.8) + 1
    criteria["variance_80_percent"] = int(idx_80)

    # 90% 분산 설명 기준
    idx_90 = np.argmax(cumulative_variance >= 0.9) + 1
    criteria["variance_90_percent"] = int(idx_90)

    # Kaiser 기준 (고유값 > 1, 표준화된 경우)
    eigenvalues = explained_variance_ratio * len(explained_variance_ratio)
    kaiser_components = int(np.sum(eigenvalues > 1))
    criteria["kaiser_criterion"] = max(1, kaiser_components)

    # Elbow method (분산 감소율 기준)
    if len(explained_variance_ratio) > 2:
        variance_diff = np.diff(explained_variance_ratio)
        variance_diff_2 = np.diff(variance_diff)
        if len(variance_diff_2) > 0:
            elbow_idx = np.argmax(variance_diff_2) + 2
            criteria["elbow_method"] = int(min(elbow_idx, len(explained_variance_ratio)))

    # 평균 기준으로 권고
    valid_criteria = [v for v in criteria.values() if v <= len(explained_variance_ratio)]
    if valid_criteria:
        recommended = int(np.median(valid_criteria))
    else:
        recommended = min(3, len(explained_variance_ratio))

    return {
        "recommended_components": recommended,
        "criteria_analysis": criteria,
        "variance_at_recommended": float(cumulative_variance[recommended-1]) if recommended <= len(cumulative_variance) else 1.0,
        "reasoning": f"{recommended}개 주성분으로 {cumulative_variance[recommended-1]*100:.1f}%의 분산 설명 가능" if recommended <= len(cumulative_variance) else "데이터 차원이 낮아 모든 컴포넌트 사용 권고"
    }

def analyze_dimensionality_reduction(original_dims: int, reduced_dims: int,
                                   explained_variance_ratio: np.ndarray) -> Dict[str, Any]:
    """차원 축소 효과 분석"""

    reduction_ratio = reduced_dims / original_dims
    total_variance_retained = np.sum(explained_variance_ratio)
    information_loss = 1 - total_variance_retained

    # 효율성 평가
    efficiency_score = total_variance_retained / reduction_ratio

    # 차원 축소 품질 평가
    if total_variance_retained > 0.9:
        quality_grade = "우수"
    elif total_variance_retained > 0.8:
        quality_grade = "양호"
    elif total_variance_retained > 0.7:
        quality_grade = "보통"
    else:
        quality_grade = "개선 필요"

    return {
        "original_dimensions": original_dims,
        "reduced_dimensions": reduced_dims,
        "reduction_ratio": float(reduction_ratio),
        "dimensions_removed": original_dims - reduced_dims,
        "variance_retained": float(total_variance_retained),
        "information_loss": float(information_loss),
        "efficiency_score": float(efficiency_score),
        "quality_grade": quality_grade,
        "compression_benefit": f"{original_dims}차원을 {reduced_dims}차원으로 축소하여 {(1-reduction_ratio)*100:.1f}% 차원 감소"
    }

def evaluate_pca_quality(pca, X_processed: np.ndarray, original_df: pd.DataFrame) -> Dict[str, Any]:
    """PCA 품질 평가"""

    quality_metrics = {}

    # 설명 분산 분포 분석
    explained_var_ratio = pca.explained_variance_ratio_
    quality_metrics["variance_distribution"] = {
        "first_component_dominance": float(explained_var_ratio[0]),
        "top_3_components_share": float(np.sum(explained_var_ratio[:min(3, len(explained_var_ratio))])),
        "variance_concentration": "높음" if explained_var_ratio[0] > 0.5 else "보통" if explained_var_ratio[0] > 0.3 else "낮음"
    }

    # 데이터 적합성 평가
    n_samples, n_features = X_processed.shape

    # Kaiser-Meyer-Olkin (KMO) 근사치
    correlation_matrix = np.corrcoef(X_processed.T)
    if n_features > 1:
        try:
            partial_corr_sum = np.sum(correlation_matrix**2) - n_features  # 대각선 제외
            total_corr_sum = np.sum(correlation_matrix**2)
            kmo_approx = partial_corr_sum / total_corr_sum if total_corr_sum > 0 else 0

            quality_metrics["data_suitability"] = {
                "kmo_approximation": float(kmo_approx),
                "suitability_assessment": "우수" if kmo_approx > 0.8 else "양호" if kmo_approx > 0.6 else "보통"
            }
        except:
            quality_metrics["data_suitability"] = {"assessment": "계산 불가"}

    # 샘플 크기 적절성
    sample_adequacy = n_samples / n_features if n_features > 0 else 0
    quality_metrics["sample_adequacy"] = {
        "samples_per_feature": float(sample_adequacy),
        "adequacy_level": "우수" if sample_adequacy >= 10 else "충분" if sample_adequacy >= 5 else "부족"
    }

    # 결측치 및 데이터 품질
    quality_metrics["data_quality"] = {
        "missing_values_handled": int(original_df.isnull().sum().sum()),
        "data_completeness": float((original_df.size - original_df.isnull().sum().sum()) / original_df.size)
    }

    return quality_metrics

def generate_component_interpretation(positive_features: List[str],
                                    negative_features: List[str],
                                    component_num: int) -> str:
    """주성분 해석 텍스트 생성"""

    interpretation_parts = [f"PC{component_num}는 "]

    if positive_features:
        pos_str = ", ".join(positive_features)
        interpretation_parts.append(f"{pos_str}에 양의 기여를 하는 ")

    if negative_features:
        neg_str = ", ".join(negative_features)
        if positive_features:
            interpretation_parts.append(f"반면 {neg_str}에는 음의 기여를 하는 ")
        else:
            interpretation_parts.append(f"{neg_str}에 음의 기여를 하는 ")

    interpretation_parts.append("차원을 나타냅니다.")

    return "".join(interpretation_parts)

def main():
    """
    메인 실행 함수 - PCA 분석의 진입점

    표준 입출력을 통해 JSON 데이터를 받아 PCA 분석을 수행하고
    표준화된 형태로 결과를 반환합니다.

    입력 형식:
    - JSON을 통한 데이터 또는 파일 경로
    - 선택적 매개변수: file_path, n_components, standardize

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

        # PCA 옵션
        n_components = params.get('n_components', None)
        standardize = params.get('standardize', True)

        # 데이터 기본 정보 추출
        data_info = get_data_info(df)

        # PCA 분석 수행
        pca_result = perform_pca_analysis(df, n_components, standardize)

        if not pca_result.get('success', False):
            error_result = {
                "success": False,
                "error": pca_result.get('error', 'PCA 분석 실패'),
                "analysis_type": "pca_analysis"
            }
            output_results(error_result)
            return

        # 분석 결과 통합
        analysis_results = {
            "pca_analysis": pca_result,
            "pca_summary": {
                "original_dimensions": pca_result.get('original_dimensions', 0),
                "reduced_dimensions": pca_result.get('n_components', 0),
                "variance_retained": f"{sum(pca_result.get('explained_variance_ratio', []))*100:.1f}%",
                "recommended_components": pca_result.get('optimal_components_recommendation', {}).get('recommended_components', 0),
                "quality_grade": pca_result.get('dimensionality_analysis', {}).get('quality_grade', '알 수 없음')
            }
        }

        # 요약 생성
        original_dims = pca_result.get('original_dimensions', 0)
        reduced_dims = pca_result.get('n_components', 0)
        variance_retained = sum(pca_result.get('explained_variance_ratio', [])) * 100
        summary = f"PCA 분석 완료 - {original_dims}차원을 {reduced_dims}차원으로 축소, {variance_retained:.1f}% 분산 보존"

        # 표준화된 결과 생성
        final_result = create_analysis_result(
            analysis_type="pca_analysis",
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
            "analysis_type": "pca_analysis",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        output_results(error_result)
        sys.exit(1)

if __name__ == "__main__":
    main()