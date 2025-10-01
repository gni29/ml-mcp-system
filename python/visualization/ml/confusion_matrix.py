#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Confusion Matrix Visualization
혼동 행렬 시각화

이 모듈은 분류 모델의 성능을 평가하는 혼동 행렬을 생성합니다.
주요 기능:
- 혼동 행렬 히트맵 생성
- 정확도, 정밀도, 재현율 계산
- 클래스별 성능 분석
- 오분류 패턴 분석
- 한국어 성능 해석
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Malgun Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def create_confusion_matrix(y_true: List, y_pred: List, labels: Optional[List[str]] = None,
                           title: str = "Confusion Matrix", output_path: str = "confusion_matrix.png",
                           figsize: tuple = (8, 6), normalize: Optional[str] = None) -> Dict[str, Any]:
    """
    혼동 행렬 시각화 생성

    Parameters:
    -----------
    y_true : List
        실제 라벨
    y_pred : List
        예측 라벨
    labels : List[str], optional
        클래스 라벨 이름
    title : str, default="Confusion Matrix"
        차트 제목
    output_path : str, default="confusion_matrix.png"
        출력 파일 경로
    figsize : tuple, default=(8, 6)
        그림 크기
    normalize : str, optional
        정규화 방식 ('true', 'pred', 'all', None)

    Returns:
    --------
    Dict[str, Any]
        혼동 행렬 결과 및 성능 분석
    """

    try:
        # 데이터 타입 변환
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if len(y_true) != len(y_pred):
            return {
                "success": False,
                "error": "실제 라벨과 예측 라벨의 길이가 다릅니다",
                "y_true_length": len(y_true),
                "y_pred_length": len(y_pred)
            }

        if len(y_true) == 0:
            return {
                "success": False,
                "error": "데이터가 비어있습니다"
            }

        # 고유 클래스 추출
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(unique_classes)

        # 라벨이 제공되지 않은 경우 기본값 사용
        if labels is None:
            labels = [str(cls) for cls in unique_classes]
        elif len(labels) != n_classes:
            labels = [str(cls) for cls in unique_classes]

        # 혼동 행렬 계산
        cm = confusion_matrix(y_true, y_pred, labels=unique_classes)

        # 정규화 적용
        if normalize == 'true':
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_to_plot = cm_normalized
            fmt = '.2f'
            cmap = 'Blues'
        elif normalize == 'pred':
            cm_normalized = cm.astype('float') / cm.sum(axis=0)
            cm_to_plot = cm_normalized
            fmt = '.2f'
            cmap = 'Blues'
        elif normalize == 'all':
            cm_normalized = cm.astype('float') / cm.sum()
            cm_to_plot = cm_normalized
            fmt = '.2f'
            cmap = 'Blues'
        else:
            cm_to_plot = cm
            fmt = 'd'
            cmap = 'Blues'

        # 그래프 생성
        fig, ax = plt.subplots(figsize=figsize)

        # 히트맵 생성
        sns.heatmap(cm_to_plot, annot=True, fmt=fmt, cmap=cmap,
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': '값 (정규화됨)' if normalize else '개수'},
                    ax=ax)

        # 제목 및 라벨 설정
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('예측 라벨 (Predicted Label)', fontsize=12)
        ax.set_ylabel('실제 라벨 (True Label)', fontsize=12)

        # 레이아웃 최적화
        plt.tight_layout()

        # 파일 저장
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 성능 메트릭 계산
        performance_metrics = calculate_performance_metrics(y_true, y_pred, unique_classes, labels)

        # 오분류 분석
        misclassification_analysis = analyze_misclassifications(cm, labels)

        # 클래스별 분석
        class_analysis = analyze_class_performance(y_true, y_pred, unique_classes, labels)

        return {
            "success": True,
            "output_path": output_path,
            "chart_type": "confusion_matrix",
            "confusion_matrix": cm.tolist(),
            "normalized_confusion_matrix": cm_normalized.tolist() if normalize else None,
            "class_labels": labels,
            "performance_metrics": performance_metrics,
            "misclassification_analysis": misclassification_analysis,
            "class_analysis": class_analysis,
            "insights": generate_confusion_matrix_insights(performance_metrics, misclassification_analysis, class_analysis)
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"혼동 행렬 생성 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def calculate_performance_metrics(y_true, y_pred, unique_classes, labels) -> Dict[str, Any]:
    """성능 메트릭 계산"""

    metrics = {}

    # 전체 성능 메트릭
    metrics["overall"] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    }

    # 분류 보고서
    try:
        class_report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
        metrics["classification_report"] = {
            str(k): {str(k2): float(v2) if isinstance(v2, (int, float)) else v2
                     for k2, v2 in v.items()} if isinstance(v, dict) else v
            for k, v in class_report.items()
        }
    except Exception as e:
        metrics["classification_report_error"] = str(e)

    return metrics

def analyze_misclassifications(cm, labels) -> Dict[str, Any]:
    """오분류 패턴 분석"""

    analysis = {
        "total_predictions": int(cm.sum()),
        "correct_predictions": int(np.diag(cm).sum()),
        "misclassifications": int(cm.sum() - np.diag(cm).sum()),
        "misclassification_rate": float((cm.sum() - np.diag(cm).sum()) / cm.sum()),
        "most_confused_pairs": [],
        "class_confusion_details": {}
    }

    # 가장 혼동이 많은 클래스 쌍 찾기
    confusion_pairs = []
    n_classes = len(labels)

    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    "true_class": labels[i],
                    "predicted_class": labels[j],
                    "count": int(cm[i, j]),
                    "percentage": float(cm[i, j] / cm.sum() * 100)
                })

    # 혼동이 많은 순으로 정렬
    confusion_pairs.sort(key=lambda x: x["count"], reverse=True)
    analysis["most_confused_pairs"] = confusion_pairs[:5]  # 상위 5개

    # 클래스별 혼동 상세 분석
    for i, label in enumerate(labels):
        true_positives = int(cm[i, i])
        false_positives = int(cm[:, i].sum() - cm[i, i])
        false_negatives = int(cm[i, :].sum() - cm[i, i])
        true_negatives = int(cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i])

        analysis["class_confusion_details"][label] = {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "sensitivity": float(true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else 0,
            "specificity": float(true_negatives / (true_negatives + false_positives)) if (true_negatives + false_positives) > 0 else 0
        }

    return analysis

def analyze_class_performance(y_true, y_pred, unique_classes, labels) -> Dict[str, Any]:
    """클래스별 성능 분석"""

    analysis = {
        "class_distribution": {},
        "class_performance_ranking": [],
        "performance_variability": {}
    }

    # 클래스별 분포
    true_counts = pd.Series(y_true).value_counts()
    pred_counts = pd.Series(y_pred).value_counts()

    for i, (cls, label) in enumerate(zip(unique_classes, labels)):
        true_count = true_counts.get(cls, 0)
        pred_count = pred_counts.get(cls, 0)

        analysis["class_distribution"][label] = {
            "true_count": int(true_count),
            "predicted_count": int(pred_count),
            "true_percentage": float(true_count / len(y_true) * 100),
            "predicted_percentage": float(pred_count / len(y_pred) * 100) if len(y_pred) > 0 else 0
        }

    # 클래스별 F1 점수로 성능 순위
    try:
        f1_scores = f1_score(y_true, y_pred, labels=unique_classes, average=None, zero_division=0)
        class_performance = [(labels[i], float(score)) for i, score in enumerate(f1_scores)]
        class_performance.sort(key=lambda x: x[1], reverse=True)
        analysis["class_performance_ranking"] = [
            {"class": cls, "f1_score": score} for cls, score in class_performance
        ]

        # 성능 변동성
        analysis["performance_variability"] = {
            "f1_mean": float(np.mean(f1_scores)),
            "f1_std": float(np.std(f1_scores)),
            "f1_range": float(np.max(f1_scores) - np.min(f1_scores)),
            "coefficient_of_variation": float(np.std(f1_scores) / np.mean(f1_scores)) if np.mean(f1_scores) > 0 else 0
        }
    except Exception as e:
        analysis["performance_ranking_error"] = str(e)

    return analysis

def generate_confusion_matrix_insights(performance_metrics, misclassification_analysis, class_analysis) -> List[str]:
    """혼동 행렬 인사이트 생성"""

    insights = []

    # 전체 성능 평가
    overall = performance_metrics.get("overall", {})
    accuracy = overall.get("accuracy", 0)

    if accuracy >= 0.9:
        insights.append(f"매우 우수한 분류 성능을 보입니다 (정확도: {accuracy:.1%})")
    elif accuracy >= 0.8:
        insights.append(f"좋은 분류 성능을 보입니다 (정확도: {accuracy:.1%})")
    elif accuracy >= 0.7:
        insights.append(f"보통 수준의 분류 성능을 보입니다 (정확도: {accuracy:.1%})")
    else:
        insights.append(f"분류 성능 개선이 필요합니다 (정확도: {accuracy:.1%})")

    # 오분류 패턴 분석
    misclass_rate = misclassification_analysis.get("misclassification_rate", 0)
    if misclass_rate > 0.2:
        insights.append(f"오분류율이 높습니다 ({misclass_rate:.1%})")

    # 가장 혼동되는 클래스 쌍
    confused_pairs = misclassification_analysis.get("most_confused_pairs", [])
    if confused_pairs:
        top_confusion = confused_pairs[0]
        insights.append(f"'{top_confusion['true_class']}'와 '{top_confusion['predicted_class']}' 간 혼동이 가장 많습니다 ({top_confusion['count']}회)")

    # 클래스별 성능 편차
    performance_var = class_analysis.get("performance_variability", {})
    cv = performance_var.get("coefficient_of_variation", 0)
    if cv > 0.3:
        insights.append("클래스별 성능 편차가 큽니다")
        # 성능이 가장 좋은/나쁜 클래스
        ranking = class_analysis.get("class_performance_ranking", [])
        if len(ranking) >= 2:
            best_class = ranking[0]
            worst_class = ranking[-1]
            insights.append(f"성능이 가장 좋은 클래스: '{best_class['class']}' (F1: {best_class['f1_score']:.3f})")
            insights.append(f"성능이 가장 낮은 클래스: '{worst_class['class']}' (F1: {worst_class['f1_score']:.3f})")

    # 클래스 불균형 분석
    distribution = class_analysis.get("class_distribution", {})
    if distribution:
        percentages = [info["true_percentage"] for info in distribution.values()]
        if max(percentages) / min(percentages) > 3:
            insights.append("클래스 불균형이 존재합니다")

    return insights

def main():
    """메인 실행 함수"""
    try:
        # stdin에서 JSON 데이터 읽기
        input_data = sys.stdin.read()
        params = json.loads(input_data)

        # 혼동 행렬 생성
        result = create_confusion_matrix(**params)

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