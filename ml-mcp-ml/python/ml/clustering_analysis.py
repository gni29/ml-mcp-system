#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clustering Analysis for ML MCP
ML MCP용 클러스터링 분석 스크립트
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('default')

# Add shared utilities to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "ml-mcp-shared" / "python"))

try:
    from common_utils import load_data, get_data_info, create_analysis_result, output_results, validate_required_params
except ImportError:
    # Fallback implementations
    def load_data(file_path: str) -> pd.DataFrame:
        file_path = Path(file_path)
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_path.suffix}")

    def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
        }

    def create_analysis_result(analysis_type: str, data_info: Dict[str, Any], results: Dict[str, Any], summary: str = None) -> Dict[str, Any]:
        return {
            "analysis_type": analysis_type,
            "data_info": data_info,
            "summary": summary or f"{analysis_type} 완료",
            **results
        }

    def output_results(results: Dict[str, Any]):
        print(json.dumps(results, ensure_ascii=False, indent=2))

    def validate_required_params(params: Dict[str, Any], required: list):
        missing = [param for param in required if param not in params]
        if missing:
            raise ValueError(f"필수 매개변수가 누락됨: {', '.join(missing)}")


def preprocess_clustering_data(df: pd.DataFrame) -> tuple:
    """
    Preprocess data for clustering
    클러스터링을 위한 데이터 전처리
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        raise ValueError("클러스터링을 위한 수치형 컬럼이 없습니다")

    X = df[numeric_cols].copy()

    # Handle missing values
    X = X.dropna()

    if len(X) == 0:
        raise ValueError("결측치 제거 후 데이터가 없습니다")

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, X, scaler, list(numeric_cols)


def determine_optimal_clusters(X: np.ndarray, max_clusters: int = 10) -> Dict[str, Any]:
    """
    Determine optimal number of clusters using multiple methods
    여러 방법을 사용하여 최적 클러스터 수 결정
    """
    if len(X) < 4:
        return {"optimal_k": 2, "method": "default", "scores": {}}

    max_k = min(max_clusters, len(X) - 1)
    k_range = range(2, max_k + 1)

    # Elbow method (inertia)
    inertias = []
    silhouette_scores = []
    calinski_scores = []
    davies_bouldin_scores = []

    for k in k_range:
        try:
            # K-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            # Calculate metrics
            inertias.append(kmeans.inertia_)

            if len(np.unique(labels)) > 1:  # Need at least 2 clusters for silhouette
                silhouette_avg = silhouette_score(X, labels)
                silhouette_scores.append(silhouette_avg)

                calinski_score = calinski_harabasz_score(X, labels)
                calinski_scores.append(calinski_score)

                db_score = davies_bouldin_score(X, labels)
                davies_bouldin_scores.append(db_score)
            else:
                silhouette_scores.append(0)
                calinski_scores.append(0)
                davies_bouldin_scores.append(float('inf'))

        except Exception as e:
            print(f"경고: k={k}에서 클러스터링 실패: {e}")
            continue

    # Find optimal k using different methods
    optimal_methods = {}

    # Elbow method (look for the "elbow" in inertia)
    if len(inertias) >= 2:
        # Simple elbow detection: find maximum second derivative
        if len(inertias) >= 3:
            second_derivatives = []
            for i in range(1, len(inertias) - 1):
                second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
                second_derivatives.append(second_deriv)

            if second_derivatives:
                elbow_idx = np.argmax(second_derivatives)
                optimal_methods['elbow'] = k_range[elbow_idx + 1]

    # Silhouette method (maximum silhouette score)
    if silhouette_scores:
        max_silhouette_idx = np.argmax(silhouette_scores)
        optimal_methods['silhouette'] = k_range[max_silhouette_idx]

    # Calinski-Harabasz method (maximum score)
    if calinski_scores:
        max_calinski_idx = np.argmax(calinski_scores)
        optimal_methods['calinski_harabasz'] = k_range[max_calinski_idx]

    # Davies-Bouldin method (minimum score)
    if davies_bouldin_scores:
        min_db_idx = np.argmin(davies_bouldin_scores)
        optimal_methods['davies_bouldin'] = k_range[min_db_idx]

    # Choose the most commonly suggested k
    if optimal_methods:
        k_suggestions = list(optimal_methods.values())
        optimal_k = max(set(k_suggestions), key=k_suggestions.count)
    else:
        optimal_k = 3  # Default

    scores_dict = {
        'k_range': list(k_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'calinski_scores': calinski_scores,
        'davies_bouldin_scores': davies_bouldin_scores
    }

    return {
        'optimal_k': optimal_k,
        'method_suggestions': optimal_methods,
        'scores': scores_dict
    }


def perform_clustering(X: np.ndarray, algorithm: str, n_clusters: int) -> tuple:
    """
    Perform clustering with specified algorithm
    지정된 알고리즘으로 클러스터링 수행
    """
    if algorithm == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(X)
        cluster_centers = clusterer.cluster_centers_
        inertia = clusterer.inertia_

    elif algorithm == 'hierarchical':
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clusterer.fit_predict(X)
        cluster_centers = None
        inertia = None

    elif algorithm == 'dbscan':
        # For DBSCAN, estimate eps parameter
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=min(10, len(X)))
        neighbors_fit = neighbors.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)
        distances = np.sort(distances[:, -1])
        eps = np.percentile(distances, 90)  # Use 90th percentile as eps

        clusterer = DBSCAN(eps=eps, min_samples=max(2, len(X) // 50))
        labels = clusterer.fit_predict(X)
        cluster_centers = None
        inertia = None

        # DBSCAN can produce noise points (label -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    elif algorithm == 'gaussian_mixture':
        clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
        clusterer.fit(X)
        labels = clusterer.predict(X)
        cluster_centers = clusterer.means_
        inertia = None

    else:
        raise ValueError(f"지원하지 않는 클러스터링 알고리즘: {algorithm}")

    return labels, clusterer, cluster_centers, inertia, n_clusters


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate clustering quality
    클러스터링 품질 평가
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters < 2:
        return {
            'n_clusters': n_clusters,
            'silhouette_score': None,
            'calinski_harabasz_score': None,
            'davies_bouldin_score': None,
            'note': '클러스터가 2개 미만이어서 평가 지표를 계산할 수 없습니다'
        }

    # Calculate clustering metrics
    metrics = {
        'n_clusters': n_clusters,
        'silhouette_score': float(silhouette_score(X, labels)),
        'calinski_harabasz_score': float(calinski_harabasz_score(X, labels)),
        'davies_bouldin_score': float(davies_bouldin_score(X, labels))
    }

    # Cluster size analysis
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique_labels.astype(str), counts.astype(int)))

    # Remove noise cluster if exists (label -1)
    if '-1' in cluster_sizes:
        noise_points = cluster_sizes.pop('-1')
        metrics['noise_points'] = noise_points
    else:
        metrics['noise_points'] = 0

    metrics['cluster_sizes'] = cluster_sizes

    # Calculate balance
    if cluster_sizes:
        size_values = list(cluster_sizes.values())
        metrics['cluster_balance'] = {
            'min_size': min(size_values),
            'max_size': max(size_values),
            'size_ratio': max(size_values) / min(size_values) if min(size_values) > 0 else float('inf'),
            'balance_score': 1 - (np.std(size_values) / np.mean(size_values)) if np.mean(size_values) > 0 else 0
        }

    return metrics


def generate_clustering_visualizations(X: np.ndarray, labels: np.ndarray,
                                     original_data: pd.DataFrame,
                                     feature_names: List[str]) -> List[str]:
    """
    Generate clustering visualizations
    클러스터링 시각화 생성
    """
    plot_files = []

    try:
        # 2D PCA visualization
        if X.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X)

            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter)
            plt.xlabel(f'PC1 (explained variance: {pca.explained_variance_ratio_[0]:.2%})')
            plt.ylabel(f'PC2 (explained variance: {pca.explained_variance_ratio_[1]:.2%})')
            plt.title('Clustering Results - PCA Visualization')
            plt.grid(True, alpha=0.3)

            plot_file = 'clustering_pca_2d.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(plot_file)

        # Feature pairwise plots (if not too many features)
        if len(feature_names) <= 4:
            fig, axes = plt.subplots(len(feature_names), len(feature_names),
                                   figsize=(12, 12))

            for i, feature1 in enumerate(feature_names):
                for j, feature2 in enumerate(feature_names):
                    ax = axes[i, j] if len(feature_names) > 1 else axes

                    if i == j:
                        # Histogram on diagonal
                        for cluster in np.unique(labels):
                            if cluster != -1:  # Skip noise points
                                mask = labels == cluster
                                ax.hist(original_data[feature1][mask], alpha=0.6,
                                       label=f'Cluster {cluster}', bins=15)
                        ax.set_xlabel(feature1)
                        ax.set_ylabel('Frequency')
                        ax.legend()
                    else:
                        # Scatter plot
                        scatter = ax.scatter(original_data[feature1], original_data[feature2],
                                           c=labels, cmap='viridis', alpha=0.6, s=20)
                        ax.set_xlabel(feature1)
                        ax.set_ylabel(feature2)

            plt.suptitle('Clustering Results - Feature Space')
            plt.tight_layout()

            plot_file = 'clustering_feature_space.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(plot_file)

        # Cluster size distribution
        plt.figure(figsize=(8, 6))
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Filter out noise points
        mask = unique_labels != -1
        cluster_labels = unique_labels[mask]
        cluster_counts = counts[mask]

        plt.bar(range(len(cluster_labels)), cluster_counts,
                color=plt.cm.viridis(np.linspace(0, 1, len(cluster_labels))))
        plt.xlabel('Cluster')
        plt.ylabel('Number of Points')
        plt.title('Cluster Size Distribution')
        plt.xticks(range(len(cluster_labels)), [f'Cluster {i}' for i in cluster_labels])
        plt.grid(True, alpha=0.3)

        plot_file = 'cluster_size_distribution.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot_file)

    except Exception as e:
        print(f"경고: 시각화 생성 중 오류 발생: {e}")

    return plot_files


def perform_clustering_analysis(data_file: str, algorithm: str = 'kmeans', n_clusters: int = 3,
                               auto_determine_clusters: bool = True,
                               include_visualization: bool = True) -> Dict[str, Any]:
    """
    Perform comprehensive clustering analysis
    포괄적인 클러스터링 분석 수행
    """
    # Load and preprocess data
    df = load_data(data_file)
    X_scaled, X_original, scaler, feature_names = preprocess_clustering_data(df)

    # Determine optimal number of clusters if requested
    final_n_clusters = n_clusters
    optimization_results = None

    if auto_determine_clusters and algorithm in ['kmeans', 'hierarchical', 'gaussian_mixture']:
        optimization_results = determine_optimal_clusters(X_scaled)
        final_n_clusters = optimization_results['optimal_k']

    # Perform clustering
    labels, clusterer, cluster_centers, inertia, actual_n_clusters = perform_clustering(
        X_scaled, algorithm, final_n_clusters
    )

    # Evaluate clustering
    evaluation_metrics = evaluate_clustering(X_scaled, labels)

    # Generate cluster profiles
    cluster_profiles = generate_cluster_profiles(X_original, labels, feature_names)

    # Generate visualizations if requested
    plot_files = []
    if include_visualization:
        plot_files = generate_clustering_visualizations(
            X_scaled, labels, X_original, feature_names
        )

    # Save clustering results
    output_data = X_original.copy()
    output_data['cluster'] = labels
    output_file = f"clustering_results_{algorithm}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_data.to_csv(output_file, index=False)

    # Save clustering model
    model_file = f"clustering_model_{algorithm}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.joblib"
    model_info = {
        'clusterer': clusterer,
        'scaler': scaler,
        'feature_names': feature_names,
        'algorithm': algorithm,
        'n_clusters': actual_n_clusters,
        'labels': labels
    }
    joblib.dump(model_info, model_file)

    # Prepare results
    results = {
        'algorithm': algorithm,
        'requested_clusters': n_clusters,
        'final_clusters': actual_n_clusters,
        'auto_determine_clusters': auto_determine_clusters,
        'data_points': len(X_scaled),
        'features_used': feature_names,
        'output_file': output_file,
        'model_file': model_file,

        # Performance metrics
        'silhouette_score': evaluation_metrics.get('silhouette_score'),
        'calinski_harabasz_score': evaluation_metrics.get('calinski_harabasz_score'),
        'davies_bouldin_score': evaluation_metrics.get('davies_bouldin_score'),
        'inertia': inertia,

        # Cluster analysis
        'cluster_sizes': evaluation_metrics.get('cluster_sizes', {}),
        'cluster_profiles': cluster_profiles,
        'evaluation_metrics': evaluation_metrics,
        'plot_files': plot_files
    }

    # Add optimization results if performed
    if optimization_results:
        results['optimization_results'] = optimization_results

    return results


def generate_cluster_profiles(X: pd.DataFrame, labels: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
    """
    Generate detailed profiles for each cluster
    각 클러스터에 대한 상세 프로필 생성
    """
    profiles = {}

    unique_labels = np.unique(labels)

    for cluster_id in unique_labels:
        if cluster_id == -1:  # Skip noise points
            continue

        mask = labels == cluster_id
        cluster_data = X[mask]

        profile = {
            'size': int(np.sum(mask)),
            'percentage': float(np.sum(mask) / len(labels) * 100),
            'feature_statistics': {}
        }

        # Calculate statistics for each feature
        for feature in feature_names:
            feature_data = cluster_data[feature]
            profile['feature_statistics'][feature] = {
                'mean': float(feature_data.mean()),
                'median': float(feature_data.median()),
                'std': float(feature_data.std()),
                'min': float(feature_data.min()),
                'max': float(feature_data.max()),
                'q25': float(feature_data.quantile(0.25)),
                'q75': float(feature_data.quantile(0.75))
            }

        profiles[f'cluster_{cluster_id}'] = profile

    return profiles


def main():
    """메인 실행 함수"""
    try:
        # Read input from stdin
        input_data = sys.stdin.read()
        options = json.loads(input_data)

        # Validate required parameters
        validate_required_params(options, ['data_file'])

        # Extract parameters
        data_file = options['data_file']
        algorithm = options.get('algorithm', 'kmeans')
        n_clusters = options.get('n_clusters', 3)
        auto_determine_clusters = options.get('auto_determine_clusters', True)
        include_visualization = options.get('include_visualization', True)

        # Perform clustering analysis
        results = perform_clustering_analysis(
            data_file=data_file,
            algorithm=algorithm,
            n_clusters=n_clusters,
            auto_determine_clusters=auto_determine_clusters,
            include_visualization=include_visualization
        )

        # Get data info for final result
        df = load_data(data_file)
        data_info = get_data_info(df)

        # Create final result
        final_result = create_analysis_result(
            analysis_type="clustering_analysis",
            data_info=data_info,
            results=results,
            summary=f"{algorithm} 클러스터링 완료 - {results['final_clusters']}개 클러스터, 실루엣 점수: {results['silhouette_score']:.3f}"
        )

        # Output results
        output_results(final_result)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "analysis_type": "clustering_analysis"
        }
        output_results(error_result)
        sys.exit(1)


if __name__ == "__main__":
    main()