#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clustering Analysis Module
클러스터링 분석 모듈

이 모듈은 다양한 클러스터링 알고리즘을 사용하여 데이터의 패턴을 발견합니다.
주요 기능:
- K-Means, DBSCAN, Hierarchical Clustering, Gaussian Mixture Model
- 최적 클러스터 수 자동 결정 (실루엣 분석)
- 클러스터 품질 평가 및 해석
- 차원 축소를 통한 클러스터 시각화
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
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def perform_clustering_analysis(df: pd.DataFrame, algorithm: str = 'kmeans',
                              n_clusters: int = 3, auto_determine: bool = True) -> Dict[str, Any]:
    """
    클러스터링 분석 수행

    Parameters:
    -----------
    df : pd.DataFrame
        분석할 데이터프레임
    algorithm : str, default='kmeans'
        사용할 클러스터링 알고리즘 ('kmeans', 'dbscan', 'hierarchical', 'gmm')
    n_clusters : int, default=3
        클러스터 개수 (dbscan 제외)
    auto_determine : bool, default=True
        최적 클러스터 수 자동 결정 여부

    Returns:
    --------
    Dict[str, Any]
        클러스터링 분석 결과
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

    if len(numeric_df) < 3:
        return {
            "error": "클러스터링을 위한 데이터가 부족합니다 (최소 3개 필요)",
            "data_size": len(numeric_df)
        }

    try:
        # 데이터 전처리 (표준화)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_df)

        results = {
            "success": True,
            "algorithm": algorithm,
            "data_points": len(df),
            "features_used": list(numeric_df.columns),
            "preprocessing": "StandardScaler applied"
        }

        # 최적 클러스터 수 결정 (K-Means, GMM만 해당)
        if auto_determine and algorithm in ['kmeans', 'gmm']:
            optimal_k = find_optimal_clusters(X_scaled, max_k=min(10, len(X_scaled)//2))
            n_clusters = optimal_k
            results["optimal_clusters_analysis"] = {
                "method": "silhouette_analysis",
                "optimal_k": optimal_k
            }

        # 클러스터링 수행
        if algorithm == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(X_scaled)
            results["cluster_centers"] = clusterer.cluster_centers_.tolist()
            results["inertia"] = float(clusterer.inertia_)

        elif algorithm == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = clusterer.fit_predict(X_scaled)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            results["n_noise_points"] = int(list(cluster_labels).count(-1))
            results["eps"] = 0.5
            results["min_samples"] = 5

        elif algorithm == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(X_scaled)

        elif algorithm == 'gmm':
            clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(X_scaled)
            results["convergence"] = clusterer.converged_
            results["n_iter"] = int(clusterer.n_iter_)
            results["log_likelihood"] = float(clusterer.lower_bound_)

        else:
            return {"error": f"지원하지 않는 알고리즘: {algorithm}"}

        results["n_clusters"] = int(n_clusters)
        results["cluster_labels"] = cluster_labels.tolist()

        # 클러스터 품질 평가
        if len(set(cluster_labels)) > 1:  # 클러스터가 1개 이상일 때만
            if len(set(cluster_labels)) < len(X_scaled):  # 모든 점이 다른 클러스터가 아닐 때
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)
                davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)

                results["quality_metrics"] = {
                    "silhouette_score": float(silhouette_avg),
                    "calinski_harabasz_score": float(calinski_harabasz),
                    "davies_bouldin_score": float(davies_bouldin)
                }

        # 클러스터별 통계
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            if cluster_mask.any():
                cluster_data = numeric_df[cluster_mask]
                cluster_stats[f"cluster_{i}"] = {
                    "size": int(cluster_mask.sum()),
                    "percentage": float(cluster_mask.mean() * 100),
                    "mean_values": cluster_data.mean().to_dict()
                }

        results["cluster_statistics"] = cluster_stats

        # 차원 축소 (시각화용)
        if X_scaled.shape[1] > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            results["pca_for_visualization"] = {
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "pca_coordinates": X_pca.tolist()
            }

        return results

    except Exception as e:
        return {
            "error": f"클러스터링 분석 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def find_optimal_clusters(X, max_k=10):
    """실루엣 분석을 통한 최적 클러스터 수 찾기"""
    silhouette_scores = []
    k_range = range(2, min(max_k + 1, len(X)))

    for k in k_range:
        if k >= len(X):
            break

        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        except:
            continue

    if silhouette_scores:
        optimal_idx = np.argmax(silhouette_scores)
        return list(k_range)[optimal_idx]
    else:
        return 3  # 기본값

def main():
    """
    메인 실행 함수 - 클러스터링 분석의 진입점

    표준 입출력을 통해 JSON 데이터를 받아 클러스터링 분석을 수행하고
    표준화된 형태로 결과를 반환합니다.

    입력 형식:
    - JSON을 통한 데이터 또는 파일 경로
    - 선택적 매개변수: file_path, algorithm, n_clusters, auto_determine

    출력 형식:
    - 표준화된 분석 결과 JSON
    - 성공/실패 상태 포함
    - 한국어 해석 및 클러스터 품질 평가
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

        # 클러스터링 옵션
        algorithm = params.get('algorithm', 'kmeans')
        n_clusters = params.get('n_clusters', 3)
        auto_determine = params.get('auto_determine', True)

        # 데이터 기본 정보 추출
        data_info = get_data_info(df)

        # 클러스터링 분석 수행
        clustering_result = perform_clustering_analysis(df, algorithm, n_clusters, auto_determine)

        if not clustering_result.get('success', False):
            error_result = {
                "success": False,
                "error": clustering_result.get('error', '클러스터링 분석 실패'),
                "analysis_type": "clustering_analysis"
            }
            output_results(error_result)
            return

        # 분석 결과 통합
        analysis_results = {
            "clustering_analysis": clustering_result,
            "clustering_summary": {
                "algorithm_used": clustering_result.get('algorithm', algorithm),
                "n_clusters_found": clustering_result.get('n_clusters', 0),
                "data_points_clustered": clustering_result.get('data_points', 0),
                "quality_assessment": "우수" if clustering_result.get('quality_metrics', {}).get('silhouette_score', 0) > 0.5 else "보통"
            }
        }

        # 요약 생성
        n_clusters_found = clustering_result.get('n_clusters', 0)
        algorithm_used = clustering_result.get('algorithm', algorithm)
        summary = f"클러스터링 분석 완료 - {algorithm_used} 알고리즘으로 {n_clusters_found}개 클러스터 발견"

        # 표준화된 결과 생성
        final_result = create_analysis_result(
            analysis_type="clustering_analysis",
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
            "analysis_type": "clustering_analysis",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        output_results(error_result)
        sys.exit(1)

if __name__ == "__main__":
    main()