import { Logger } from '../../utils/logger.js';

export class ResultFormatter {
  constructor() {
    this.logger = new Logger();
  }

  formatAnalysisResult(result, analysisType) {
    try {
      switch (analysisType) {
        case 'descriptive_stats':
          return this.formatDescriptiveStats(result);
        case 'correlation':
          return this.formatCorrelationResult(result);
        case 'clustering':
          return this.formatClusteringResult(result);
        case 'pca':
          return this.formatPCAResult(result);
        case 'ml_model':
          return this.formatMLModelResult(result);
        case 'visualization':
          return this.formatVisualizationResult(result);
        default:
          return this.formatGenericResult(result);
      }
    } catch (error) {
      this.logger.error('결과 포맷팅 실패:', error);
      return this.formatErrorResult(error);
    }
  }

  formatDescriptiveStats(result) {
    const { statistics, summary } = result;
    
    let formattedText = '📊 **기본 통계 분석 결과**\n\n';
    
    if (statistics) {
      formattedText += '### 주요 통계량\n';
      for (const [column, stats] of Object.entries(statistics)) {
        formattedText += `\n**${column}:**\n`;
        formattedText += `- 평균: ${this.formatNumber(stats.mean)}\n`;
        formattedText += `- 표준편차: ${this.formatNumber(stats.std)}\n`;
        formattedText += `- 최솟값: ${this.formatNumber(stats.min)}\n`;
        formattedText += `- 최댓값: ${this.formatNumber(stats.max)}\n`;
        formattedText += `- 중앙값: ${this.formatNumber(stats.median)}\n`;
      }
    }

    if (summary) {
      formattedText += '\n### 요약\n';
      formattedText += summary;
    }

    return {
      content: [{
        type: 'text',
        text: formattedText
      }],
      metadata: {
        analysisType: 'descriptive_stats',
        timestamp: new Date().toISOString()
      }
    };
  }

  formatCorrelationResult(result) {
    const { correlation_matrix, strong_correlations, summary } = result;
    
    let formattedText = '🔗 **상관관계 분석 결과**\n\n';
    
    if (strong_correlations && strong_correlations.length > 0) {
      formattedText += '### 강한 상관관계 (|r| > 0.7)\n';
      for (const corr of strong_correlations) {
        const emoji = corr.correlation > 0 ? '📈' : '📉';
        formattedText += `${emoji} ${corr.var1} ↔ ${corr.var2}: ${this.formatNumber(corr.correlation)}\n`;
      }
    }

    if (correlation_matrix) {
      formattedText += '\n### 상관관계 매트릭스\n';
      formattedText += '*(히트맵으로 시각화를 권장합니다)*\n';
    }

    if (summary) {
      formattedText += '\n### 분석 요약\n';
      formattedText += summary;
    }

    return {
      content: [{
        type: 'text',
        text: formattedText
      }],
      metadata: {
        analysisType: 'correlation',
        timestamp: new Date().toISOString()
      }
    };
  }

  formatClusteringResult(result) {
    const { cluster_labels, cluster_centers, metrics, summary } = result;
    
    let formattedText = '🎯 **클러스터링 분석 결과**\n\n';
    
    if (cluster_labels) {
      const clusterCounts = this.countClusters(cluster_labels);
      formattedText += '### 클러스터 분포\n';
      for (const [cluster, count] of Object.entries(clusterCounts)) {
        formattedText += `- 클러스터 ${cluster}: ${count}개 데이터 포인트\n`;
      }
    }

    if (metrics) {
      formattedText += '\n### 클러스터링 품질 지표\n';
      if (metrics.silhouette_score) {
        formattedText += `- 실루엣 점수: ${this.formatNumber(metrics.silhouette_score)}\n`;
      }
      if (metrics.calinski_harabasz_score) {
        formattedText += `- Calinski-Harabasz 점수: ${this.formatNumber(metrics.calinski_harabasz_score)}\n`;
      }
    }

    if (summary) {
      formattedText += '\n### 분석 요약\n';
      formattedText += summary;
    }

    return {
      content: [{
        type: 'text',
        text: formattedText
      }],
      metadata: {
        analysisType: 'clustering',
        timestamp: new Date().toISOString()
      }
    };
  }

  formatPCAResult(result) {
    const { explained_variance, n_components, summary } = result;
    
    let formattedText = '🔍 **주성분 분석(PCA) 결과**\n\n';
    
    if (n_components) {
      formattedText += `### 주성분 개수: ${n_components}\n\n`;
    }

    if (explained_variance) {
      formattedText += '### 설명 분산\n';
      explained_variance.forEach((variance, index) => {
        formattedText += `- PC${index + 1}: ${this.formatPercentage(variance)}\n`;
      });
      
      const cumulative = explained_variance.reduce((acc, curr, index) => {
        acc.push((acc[index - 1] || 0) + curr);
        return acc;
      }, []);
      
      formattedText += '\n### 누적 설명 분산\n';
      cumulative.forEach((variance, index) => {
        formattedText += `- PC1~PC${index + 1}: ${this.formatPercentage(variance)}\n`;
      });
    }

    if (summary) {
      formattedText += '\n### 분석 요약\n';
      formattedText += summary;
    }

    return {
      content: [{
        type: 'text',
        text: formattedText
      }],
      metadata: {
        analysisType: 'pca',
        timestamp: new Date().toISOString()
      }
    };
  }

  formatMLModelResult(result) {
    const { model_type, metrics, feature_importance, summary } = result;
    
    let formattedText = `🤖 **${model_type} 모델 결과**\n\n`;
    
    if (metrics) {
      formattedText += '### 모델 성능 지표\n';
      for (const [metric, value] of Object.entries(metrics)) {
        const emoji = this.getMetricEmoji(metric);
        formattedText += `${emoji} ${this.formatMetricName(metric)}: ${this.formatNumber(value)}\n`;
      }
    }

    if (feature_importance && feature_importance.length > 0) {
      formattedText += '\n### 중요 피처 (상위 10개)\n';
      feature_importance.slice(0, 10).forEach((item, index) => {
        formattedText += `${index + 1}. ${item.feature}: ${this.formatNumber(item.importance)}\n`;
      });
    }

    if (summary) {
      formattedText += '\n### 모델 요약\n';
      formattedText += summary;
    }

    return {
      content: [{
        type: 'text',
        text: formattedText
      }],
      metadata: {
        analysisType: 'ml_model',
        modelType: model_type,
        timestamp: new Date().toISOString()
      }
    };
  }

  formatVisualizationResult(result) {
    const { chart_path, chart_type, data_summary } = result;
    
    let formattedText = `📈 **${chart_type} 시각화 완료**\n\n`;
    
    if (chart_path) {
      formattedText += `### 생성된 차트\n`;
      formattedText += `파일 경로: ${chart_path}\n\n`;
    }

    if (data_summary) {
      formattedText += '### 데이터 요약\n';
      formattedText += `- 데이터 포인트: ${data_summary.total_points}개\n`;
      if (data_summary.columns) {
        formattedText += `- 사용된 컬럼: ${data_summary.columns.join(', ')}\n`;
      }
    }

    return {
      content: [{
        type: 'text',
        text: formattedText
      }],
      metadata: {
        analysisType: 'visualization',
        chartType: chart_type,
        chartPath: chart_path,
        timestamp: new Date().toISOString()
      }
    };
  }

  formatGenericResult(result) {
    let formattedText = '📋 **분석 결과**\n\n';
    
    if (typeof result === 'object') {
      for (const [key, value] of Object.entries(result)) {
        formattedText += `**${key}:** ${this.formatValue(value)}\n`;
      }
    } else {
      formattedText += String(result);
    }

    return {
      content: [{
        type: 'text',
        text: formattedText
      }],
      metadata: {
        analysisType: 'generic',
        timestamp: new Date().toISOString()
      }
    };
  }

  formatErrorResult(error) {
    return {
      content: [{
        type: 'text',
        text: `❌ **오류 발생**\n\n${error.message}`
      }],
      isError: true,
      metadata: {
        error: error.message,
        timestamp: new Date().toISOString()
      }
    };
  }

  // 유틸리티 메서드들
  formatNumber(value) {
    if (typeof value !== 'number') return String(value);
    
    if (Math.abs(value) < 0.001) {
      return value.toExponential(2);
    } else if (Math.abs(value) < 1) {
      return value.toFixed(4);
    } else if (Math.abs(value) < 1000) {
      return value.toFixed(2);
    } else {
      return value.toLocaleString();
    }
  }

  formatPercentage(value) {
    return `${(value * 100).toFixed(1)}%`;
  }

  formatValue(value) {
    if (typeof value === 'number') {
      return this.formatNumber(value);
    } else if (Array.isArray(value)) {
      return value.length > 5 ? `[${value.slice(0, 5).join(', ')}, ...]` : `[${value.join(', ')}]`;
    } else if (typeof value === 'object' && value !== null) {
      return JSON.stringify(value, null, 2);
    } else {
      return String(value);
    }
  }

  countClusters(labels) {
    const counts = {};
    labels.forEach(label => {
      counts[label] = (counts[label] || 0) + 1;
    });
    return counts;
  }

  getMetricEmoji(metric) {
    const emojiMap = {
      'accuracy': '🎯',
      'precision': '🔍',
      'recall': '📊',
      'f1_score': '⚖️',
      'roc_auc': '📈',
      'mse': '📉',
      'rmse': '📏',
      'mae': '📐',
      'r2_score': '📊'
    };
    return emojiMap[metric] || '📋';
  }

  formatMetricName(metric) {
    const nameMap = {
      'accuracy': '정확도',
      'precision': '정밀도',
      'recall': '재현율',
      'f1_score': 'F1 점수',
      'roc_auc': 'ROC AUC',
      'mse': '평균제곱오차',
      'rmse': '평균제곱근오차',
      'mae': '평균절대오차',
      'r2_score': 'R² 점수'
    };
    return nameMap[metric] || metric;
  }

  // 복합 결과 포맷팅
  formatWorkflowResult(workflowResult) {
    const { workflowName, steps, finalResult, executionTime } = workflowResult;
    
    let formattedText = `🔄 **워크플로우 실행 완료: ${workflowName}**\n\n`;
    formattedText += `⏱️ 총 실행 시간: ${Math.round(executionTime / 1000)}초\n\n`;
    
    formattedText += '### 실행된 단계\n';
    steps.forEach((step, index) => {
      const status = step.success ? '✅' : '❌';
      formattedText += `${status} ${index + 1}. ${step.type} - ${step.method}\n`;
    });
    
    if (finalResult && finalResult.summary) {
      formattedText += '\n### 최종 결과\n';
      formattedText += finalResult.summary;
    }

    return {
      content: [{
        type: 'text',
        text: formattedText
      }],
      metadata: {
        analysisType: 'workflow',
        workflowName,
        executionTime,
        timestamp: new Date().toISOString()
      }
    };
  }

  // 비교 결과 포맷팅
  formatComparisonResult(comparisonResult) {
    const { comparisonType, results, winner } = comparisonResult;
    
    let formattedText = `🏆 **${comparisonType} 비교 결과**\n\n`;
    
    if (winner) {
      formattedText += `### 최고 성능 모델: ${winner.name}\n`;
      formattedText += `성능 점수: ${this.formatNumber(winner.score)}\n\n`;
    }

    formattedText += '### 전체 결과\n';
    results.forEach((result, index) => {
      const rank = index + 1;
      const medal = rank === 1 ? '🥇' : rank === 2 ? '🥈' : rank === 3 ? '🥉' : '📍';
      formattedText += `${medal} ${rank}. ${result.name}: ${this.formatNumber(result.score)}\n`;
    });

    return {
      content: [{
        type: 'text',
        text: formattedText
      }],
      metadata: {
        analysisType: 'comparison',
        comparisonType,
        timestamp: new Date().toISOString()
      }
    };
  }
}stdout.on('data', (data) => {
        stdout += data.toString();
      });

      process.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      const timeoutId = setTimeout(() => {
        process.kill('SIGTERM');
        reject(new Error(`Python 실행 시간 초과: ${timeout}ms`));
      }, timeout);

      process.
