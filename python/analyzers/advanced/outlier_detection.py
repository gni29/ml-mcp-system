#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outlier Detection Module
이상치 탐지 모듈

이 모듈은 다양한 방법으로 데이터의 이상치를 탐지하고 분석합니다.
주요 기능:
- Isolation Forest, Local Outlier Factor, Statistical methods
- 다중 방법론을 통한 합의 기반 이상치 탐지
- 이상치 점수 및 확률 계산
- 이상치 제거 및 처리 권고사항 제공
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
        print(json.dumps(results, ensure_ascii=False, indent=2, default=str))

    def validate_required_params(params: Dict[str, Any], required: list):
        """필수 매개변수 검증"""
        missing = [param for param in required if param not in params]
        if missing:
            raise ValueError(f"필수 매개변수가 누락됨: {', '.join(missing)}")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def detect_outliers_comprehensive(df: pd.DataFrame, methods: List[str] = None,
                                contamination: float = 0.1) -> Dict[str, Any]:
    """
    종합적인 이상치 탐지 수행

    Parameters:
    -----------
    df : pd.DataFrame
        분석할 데이터프레임
    methods : List[str], optional
        사용할 이상치 탐지 방법들 ['isolation_forest', 'lof', 'statistical', 'iqr', 'dbscan']
    contamination : float, default=0.1
        예상되는 이상치 비율

    Returns:
    --------
    Dict[str, Any]
        이상치 탐지 결과
    """

    if methods is None:
        methods = ['isolation_forest', 'lof', 'statistical', 'iqr']

    # 수치형 컬럼만 선택
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return {
            "error": "분석 가능한 수치형 컬럼이 없습니다",
            "available_columns": list(df.columns),
            "column_types": df.dtypes.astype(str).to_dict()
        }

    if len(numeric_df) < 5:
        return {
            "error": "이상치 탐지를 위한 데이터가 부족합니다 (최소 5개 필요)",
            "data_size": len(numeric_df)
        }

    try:
        results = {
            "success": True,
            "data_points": len(df),
            "features_analyzed": list(numeric_df.columns),
            "contamination_rate": contamination,
            "methods_used": methods
        }

        # 데이터 전처리
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_df)

        outlier_scores = {}
        outlier_labels = {}

        # 1. Isolation Forest
        if 'isolation_forest' in methods and SKLEARN_AVAILABLE:
            try:
                iso_forest = IsolationForest(contamination=contamination, random_state=42)
                iso_labels = iso_forest.fit_predict(X_scaled)
                iso_scores = iso_forest.score_samples(X_scaled)

                outlier_labels['isolation_forest'] = (iso_labels == -1).astype(int).tolist()
                outlier_scores['isolation_forest'] = iso_scores.tolist()

                results['isolation_forest'] = {
                    "outliers_detected": int(np.sum(iso_labels == -1)),
                    "outlier_percentage": float(np.mean(iso_labels == -1) * 100),
                    "outlier_indices": np.where(iso_labels == -1)[0].tolist()
                }
            except Exception as e:
                results['isolation_forest'] = {"error": f"Isolation Forest 실행 실패: {str(e)}"}

        # 2. Local Outlier Factor
        if 'lof' in methods and SKLEARN_AVAILABLE:
            try:
                n_neighbors = min(20, len(X_scaled) - 1)
                if n_neighbors >= 1:
                    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
                    lof_labels = lof.fit_predict(X_scaled)
                    lof_scores = lof.negative_outlier_factor_

                    outlier_labels['lof'] = (lof_labels == -1).astype(int).tolist()
                    outlier_scores['lof'] = lof_scores.tolist()

                    results['lof'] = {
                        "outliers_detected": int(np.sum(lof_labels == -1)),
                        "outlier_percentage": float(np.mean(lof_labels == -1) * 100),
                        "outlier_indices": np.where(lof_labels == -1)[0].tolist(),
                        "n_neighbors": n_neighbors
                    }
            except Exception as e:
                results['lof'] = {"error": f"LOF 실행 실패: {str(e)}"}

        # 3. Statistical method (Z-score)
        if 'statistical' in methods:
            try:
                z_scores = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std())
                max_z_scores = z_scores.max(axis=1)
                statistical_outliers = max_z_scores > 3.0

                outlier_labels['statistical'] = statistical_outliers.astype(int).tolist()
                outlier_scores['statistical'] = max_z_scores.tolist()

                results['statistical'] = {
                    "outliers_detected": int(np.sum(statistical_outliers)),
                    "outlier_percentage": float(np.mean(statistical_outliers) * 100),
                    "outlier_indices": np.where(statistical_outliers)[0].tolist(),
                    "threshold": 3.0,
                    "method": "z_score"
                }
            except Exception as e:
                results['statistical'] = {"error": f"Statistical method 실행 실패: {str(e)}"}

        # 4. IQR method
        if 'iqr' in methods:
            try:
                Q1 = numeric_df.quantile(0.25)
                Q3 = numeric_df.quantile(0.75)
                IQR = Q3 - Q1

                # 각 변수별 이상치 탐지
                iqr_outliers = pd.Series([False] * len(numeric_df), index=numeric_df.index)
                for col in numeric_df.columns:
                    if IQR[col] > 0:  # IQR이 0이 아닌 경우만
                        lower_bound = Q1[col] - 1.5 * IQR[col]
                        upper_bound = Q3[col] + 1.5 * IQR[col]
                        col_outliers = (numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)
                        iqr_outliers = iqr_outliers | col_outliers

                outlier_labels['iqr'] = iqr_outliers.astype(int).tolist()

                results['iqr'] = {
                    "outliers_detected": int(np.sum(iqr_outliers)),
                    "outlier_percentage": float(np.mean(iqr_outliers) * 100),
                    "outlier_indices": np.where(iqr_outliers)[0].tolist(),
                    "method": "interquartile_range"
                }
            except Exception as e:
                results['iqr'] = {"error": f"IQR method 실행 실패: {str(e)}"}

        # 5. DBSCAN-based outlier detection
        if 'dbscan' in methods and SKLEARN_AVAILABLE:
            try:
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                cluster_labels = dbscan.fit_predict(X_scaled)
                dbscan_outliers = cluster_labels == -1

                outlier_labels['dbscan'] = dbscan_outliers.astype(int).tolist()

                results['dbscan'] = {
                    "outliers_detected": int(np.sum(dbscan_outliers)),
                    "outlier_percentage": float(np.mean(dbscan_outliers) * 100),
                    "outlier_indices": np.where(dbscan_outliers)[0].tolist(),
                    "eps": 0.5,
                    "min_samples": 5
                }
            except Exception as e:
                results['dbscan'] = {"error": f"DBSCAN 실행 실패: {str(e)}"}

        # 합의 기반 이상치 탐지
        if len(outlier_labels) > 1:
            consensus_analysis = perform_consensus_analysis(outlier_labels, outlier_scores)
            results['consensus_analysis'] = consensus_analysis

        # 이상치 통계 요약
        if outlier_labels:
            outlier_summary = generate_outlier_summary(numeric_df, outlier_labels)
            results['outlier_summary'] = outlier_summary

        # 처리 권고사항
        recommendations = generate_outlier_recommendations(results)
        results['recommendations'] = recommendations

        return results

    except Exception as e:
        return {
            "error": f"이상치 탐지 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def perform_consensus_analysis(outlier_labels: Dict[str, List],
                             outlier_scores: Dict[str, List]) -> Dict[str, Any]:
    """다중 방법론 합의 분석"""

    n_points = len(list(outlier_labels.values())[0])
    n_methods = len(outlier_labels)

    # 각 데이터 포인트별 이상치 투표 수 계산
    consensus_votes = np.zeros(n_points)
    for method_labels in outlier_labels.values():
        consensus_votes += np.array(method_labels)

    # 합의 기준 설정
    majority_threshold = n_methods // 2 + 1
    unanimous_threshold = n_methods

    # 합의 기반 이상치 분류
    majority_outliers = consensus_votes >= majority_threshold
    unanimous_outliers = consensus_votes >= unanimous_threshold

    consensus_analysis = {
        "total_methods": n_methods,
        "majority_threshold": majority_threshold,
        "unanimous_threshold": unanimous_threshold,
        "majority_consensus": {
            "outliers_detected": int(np.sum(majority_outliers)),
            "outlier_percentage": float(np.mean(majority_outliers) * 100),
            "outlier_indices": np.where(majority_outliers)[0].tolist()
        },
        "unanimous_consensus": {
            "outliers_detected": int(np.sum(unanimous_outliers)),
            "outlier_percentage": float(np.mean(unanimous_outliers) * 100),
            "outlier_indices": np.where(unanimous_outliers)[0].tolist()
        },
        "vote_distribution": {
            str(i): int(np.sum(consensus_votes == i))
            for i in range(n_methods + 1)
        }
    }

    return consensus_analysis

def generate_outlier_summary(numeric_df: pd.DataFrame,
                           outlier_labels: Dict[str, List]) -> Dict[str, Any]:
    """이상치 통계 요약 생성"""

    summary = {
        "method_comparison": {},
        "feature_impact": {}
    }

    # 방법론별 비교
    for method, labels in outlier_labels.items():
        outlier_mask = np.array(labels, dtype=bool)
        if outlier_mask.any():
            outlier_data = numeric_df[outlier_mask]
            normal_data = numeric_df[~outlier_mask]

            summary["method_comparison"][method] = {
                "outlier_count": int(outlier_mask.sum()),
                "outlier_percentage": float(outlier_mask.mean() * 100),
                "outlier_mean": outlier_data.mean().to_dict(),
                "normal_mean": normal_data.mean().to_dict()
            }

    # 특성별 영향도 분석
    for col in numeric_df.columns:
        col_impact = {}
        for method, labels in outlier_labels.items():
            outlier_mask = np.array(labels, dtype=bool)
            if outlier_mask.any():
                outlier_values = numeric_df.loc[outlier_mask, col]
                normal_values = numeric_df.loc[~outlier_mask, col]

                if len(normal_values) > 0 and normal_values.std() > 0:
                    effect_size = abs(outlier_values.mean() - normal_values.mean()) / normal_values.std()
                    col_impact[method] = {
                        "effect_size": float(effect_size),
                        "outlier_mean": float(outlier_values.mean()),
                        "normal_mean": float(normal_values.mean())
                    }

        summary["feature_impact"][col] = col_impact

    return summary

def generate_outlier_recommendations(results: Dict[str, Any]) -> List[Dict[str, str]]:
    """이상치 처리 권고사항 생성"""

    recommendations = []

    # 합의 분석 기반 권고
    if 'consensus_analysis' in results:
        majority = results['consensus_analysis']['majority_consensus']
        unanimous = results['consensus_analysis']['unanimous_consensus']

        if unanimous['outliers_detected'] > 0:
            recommendations.append({
                "type": "high_confidence",
                "action": f"{unanimous['outliers_detected']}개의 확실한 이상치 제거를 권고합니다",
                "reason": "모든 탐지 방법에서 이상치로 판정",
                "priority": "high"
            })

        if majority['outliers_detected'] > unanimous['outliers_detected']:
            additional = majority['outliers_detected'] - unanimous['outliers_detected']
            recommendations.append({
                "type": "moderate_confidence",
                "action": f"{additional}개의 추가 이상치에 대한 세부 검토가 필요합니다",
                "reason": "일부 방법에서만 이상치로 판정",
                "priority": "medium"
            })

    # 방법론별 권고
    method_results = {k: v for k, v in results.items()
                     if k in ['isolation_forest', 'lof', 'statistical', 'iqr', 'dbscan']
                     and isinstance(v, dict) and 'outliers_detected' in v}

    if method_results:
        outlier_percentages = [v['outlier_percentage'] for v in method_results.values()]
        avg_percentage = np.mean(outlier_percentages)

        if avg_percentage > 20:
            recommendations.append({
                "type": "high_contamination",
                "action": "이상치 비율이 매우 높습니다. 데이터 수집 과정을 점검하세요",
                "reason": f"평균 이상치 비율: {avg_percentage:.1f}%",
                "priority": "high"
            })
        elif avg_percentage > 10:
            recommendations.append({
                "type": "moderate_contamination",
                "action": "이상치 비율이 높습니다. 도메인 전문가와 검토하세요",
                "reason": f"평균 이상치 비율: {avg_percentage:.1f}%",
                "priority": "medium"
            })
        else:
            recommendations.append({
                "type": "normal_contamination",
                "action": "정상적인 이상치 수준입니다. 분석 목적에 따라 처리하세요",
                "reason": f"평균 이상치 비율: {avg_percentage:.1f}%",
                "priority": "low"
            })

    return recommendations

def main():
    """
    메인 실행 함수 - 이상치 탐지의 진입점

    표준 입출력을 통해 JSON 데이터를 받아 이상치 탐지를 수행하고
    표준화된 형태로 결과를 반환합니다.

    입력 형식:
    - JSON을 통한 데이터 또는 파일 경로
    - 선택적 매개변수: file_path, methods, contamination

    출력 형식:
    - 표준화된 분석 결과 JSON
    - 성공/실패 상태 포함
    - 한국어 해석 및 처리 권고사항
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

        # 이상치 탐지 옵션
        methods = params.get('methods', ['isolation_forest', 'lof', 'statistical', 'iqr'])
        contamination = params.get('contamination', 0.1)

        # 데이터 기본 정보 추출
        data_info = get_data_info(df)

        # 이상치 탐지 수행
        outlier_result = detect_outliers_comprehensive(df, methods, contamination)

        if not outlier_result.get('success', False):
            error_result = {
                "success": False,
                "error": outlier_result.get('error', '이상치 탐지 실패'),
                "analysis_type": "outlier_detection"
            }
            output_results(error_result)
            return

        # 분석 결과 통합
        analysis_results = {
            "outlier_detection": outlier_result,
            "detection_summary": {
                "methods_used": outlier_result.get('methods_used', []),
                "features_analyzed": outlier_result.get('features_analyzed', []),
                "total_outliers_consensus": outlier_result.get('consensus_analysis', {}).get('majority_consensus', {}).get('outliers_detected', 0),
                "contamination_assessment": "높음" if outlier_result.get('consensus_analysis', {}).get('majority_consensus', {}).get('outlier_percentage', 0) > 10 else "보통"
            }
        }

        # 요약 생성
        consensus_outliers = outlier_result.get('consensus_analysis', {}).get('majority_consensus', {}).get('outliers_detected', 0)
        methods_count = len(outlier_result.get('methods_used', []))
        summary = f"이상치 탐지 완료 - {methods_count}개 방법으로 {consensus_outliers}개 이상치 발견"

        # 표준화된 결과 생성
        final_result = create_analysis_result(
            analysis_type="outlier_detection",
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
            "analysis_type": "outlier_detection",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        output_results(error_result)
        sys.exit(1)

if __name__ == "__main__":
    main()