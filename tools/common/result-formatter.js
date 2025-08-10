// tools/common/result-formatter.js
import { Logger } from '../../utils/logger.js';

export class ResultFormatter {
  constructor() {
    this.logger = new Logger();
    this.formatters = this.initializeFormatters();
  }

  initializeFormatters() {
    return {
      'descriptive_stats': this.formatDescriptiveStats.bind(this),
      'correlation': this.formatCorrelationResult.bind(this),
      'distribution': this.formatDistributionResult.bind(this),
      'clustering': this.formatClusteringResult.bind(this),
      'pca': this.formatPCAResult.bind(this),
      'ml_model': this.formatMLModelResult.bind(this),
      'classification': this.formatClassificationResult.bind(this),
      'regression': this.formatRegressionResult.bind(this),
      'visualization': this.formatVisualizationResult.bind(this),
      'timeseries': this.formatTimeSeriesResult.bind(this),
      'outlier_detection': this.formatOutlierResult.bind(this),
      'feature_engineering': this.formatFeatureEngineeringResult.bind(this),
      'workflow': this.formatWorkflowResult.bind(this),
      'comparison': this.formatComparisonResult.bind(this),
      'generic': this.formatGenericResult.bind(this),
      'error': this.formatErrorResult.bind(this)
    };
  }

  formatAnalysisResult(result, analysisType) {
    try {
      this.logger.debug(`결과 포맷팅 시작: ${analysisType}`);
      
      // 포맷터 함수 찾기
      const formatter = this.formatters[analysisType] || this.formatters['generic'];
      
      // 결과 포맷팅
      const formattedResult = formatter(result);
      
      // 메타데이터 추가
      if (!formattedResult.metadata) {
        formattedResult.metadata = {};
      }
      formattedResult.metadata.analysisType = analysisType;
      formattedResult.metadata.timestamp = new Date().toISOString();
      
      this.logger.debug(`결과 포맷팅 완료: ${analysisType}`);
      return formattedResult;
      
    } catch (error) {
      this.logger.error('결과 포맷팅 실패:', error);
      return this.formatErrorResult(error);
    }
  }

  formatDescriptiveStats(result) {
    const { statistics, summary, data_info } = result;
    
    let formattedText = '📊 **기본 통계 분석 결과**\n\n';
    
    // 데이터 기본 정보
    if (data_info) {
      formattedText += '### 데이터 정보\n';
      formattedText += `- 데이터 크기: ${data_info.shape ? data_info.shape.join(' × ') : 'Unknown'}\n`;
      formattedText += `- 컬럼 수: ${data_info.columns ? data_info.columns.length : 'Unknown'}\n`;
      if (data_info.null_counts) {
        const totalNulls = Object.values(data_info.null_counts).reduce((sum, count) => sum + count, 0);
        formattedText += `- 결측값: ${totalNulls}개\n`;
      }
      formattedText += '\n';
    }
    
    // 통계량
    if (statistics) {
      formattedText += '### 주요 통계량\n';
      for (const [column, stats] of Object.entries(statistics)) {
        formattedText += `\n**${column}:**\n`;
        formattedText += `- 평균: ${this.formatNumber(stats.mean)}\n`;
        formattedText += `- 표준편차: ${this.formatNumber(stats.std)}\n`;
        formattedText += `- 최솟값: ${this.formatNumber(stats.min)}\n`;
        formattedText += `- 최댓값: ${this.formatNumber(stats.max)}\n`;
        formattedText += `- 중앙값: ${this.formatNumber(stats.median)}\n`;
        
        if (stats.q25 !== undefined && stats.q75 !== undefined) {
          formattedText += `- 1사분위수: ${this.formatNumber(stats.q25)}\n`;
          formattedText += `- 3사분위수: ${this.formatNumber(stats.q75)}\n`;
        }
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
      analysisType: 'descriptive_stats'
    };
  }

  formatCorrelationResult(result) {
    const { correlation_matrix, strong_correlations, summary, correlation_pairs } = result;
    
    let formattedText = '🔗 **상관관계 분석 결과**\n\n';
    
    if (strong_correlations && strong_correlations.length > 0) {
      formattedText += '### 강한 상관관계 (|r| > 0.7)\n';
      for (const corr of strong_correlations) {
        const emoji = corr.correlation > 0 ? '📈' : '📉';
        const strength = Math.abs(corr.correlation) > 0.9 ? '매우 강함' : '강함';
        formattedText += `${emoji} **${corr.var1}** ↔ **${corr.var2}**: ${this.formatNumber(corr.correlation)} (${strength})\n`;
      }
      formattedText += '\n';
    }

    if (correlation_pairs && correlation_pairs.length > 0) {
      formattedText += '### 모든 상관관계 (상위 10개)\n';
      correlation_pairs.slice(0, 10).forEach((pair, index) => {
        const emoji = pair.correlation > 0 ? '📈' : '📉';
        formattedText += `${index + 1}. ${emoji} ${pair.var1} ↔ ${pair.var2}: ${this.formatNumber(pair.correlation)}\n`;
      });
      formattedText += '\n';
    }

    if (correlation_matrix) {
      const matrixSize = Object.keys(correlation_matrix).length;
      formattedText += `### 상관관계 매트릭스 (${matrixSize}×${matrixSize})\n`;
      formattedText += '*(히트맵으로 시각화를 권장합니다)*\n\n';
    }

    if (summary) {
      formattedText += '### 분석 요약\n';
      formattedText += summary;
    }

    return {
      content: [{
        type: 'text',
        text: formattedText
      }],
      analysisType: 'correlation'
    };
  }

  formatDistributionResult(result) {
    const { distribution_stats, normality_tests, summary } = result;
    
    let formattedText = '📈 **분포 분석 결과**\n\n';
    
    if (distribution_stats) {
      formattedText += '### 분포 통계\n';
      for (const [column, stats] of Object.entries(distribution_stats)) {
        formattedText += `\n**${column}:**\n`;
        if (stats.skewness !== undefined) {
          const skewDirection = stats.skewness > 0 ? '우편향' : stats.skewness < 0 ? '좌편향' : '대칭';
          formattedText += `- 왜도: ${this.formatNumber(stats.skewness)} (${skewDirection})\n`;
        }
        if (stats.kurtosis !== undefined) {
          formattedText += `- 첨도: ${this.formatNumber(stats.kurtosis)}\n`;
        }
      }
    }

    if (normality_tests) {
      formattedText += '\n### 정규성 검정\n';
      for (const [column, test] of Object.entries(normality_tests)) {
        const isNormal = test.p_value > 0.05;
        const emoji = isNormal ? '✅' : '❌';
        formattedText += `${emoji} **${column}**: p-value = ${this.formatNumber(test.p_value)} (${isNormal ? '정규분포' : '비정규분포'})\n`;
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
      analysisType: 'distribution'
    };
  }

  formatClusteringResult(result) {
    const { cluster_labels, cluster_centers, metrics, n_clusters, algorithm, summary } = result;
    
    let formattedText = `🎯 **클러스터링 분석 결과 (${algorithm || 'Unknown'})**\n\n`;
    
    if (cluster_labels) {
      const clusterCounts = this.countClusters(cluster_labels);
      const totalPoints = cluster_labels.length;
      
      formattedText += `### 클러스터 분포 (총 ${totalPoints}개 데이터 포인트)\n`;
      Object.entries(clusterCounts).sort((a, b) => b[1] - a[1]).forEach(([cluster, count]) => {
        const percentage = ((count / totalPoints) * 100).toFixed(1);
        formattedText += `- **클러스터 ${cluster}**: ${count}개 (${percentage}%)\n`;
      });
      formattedText += '\n';
    }

    if (metrics) {
      formattedText += '### 클러스터링 품질 지표\n';
      if (metrics.silhouette_score !== undefined) {
        const quality = metrics.silhouette_score > 0.7 ? '우수' : 
                       metrics.silhouette_score > 0.5 ? '양호' : 
                       metrics.silhouette_score > 0.25 ? '보통' : '개선 필요';
        formattedText += `- **실루엣 점수**: ${this.formatNumber(metrics.silhouette_score)} (${quality})\n`;
      }
      if (metrics.calinski_harabasz_score) {
        formattedText += `- **Calinski-Harabasz 점수**: ${this.formatNumber(metrics.calinski_harabasz_score)}\n`;
      }
      if (metrics.davies_bouldin_score) {
        formattedText += `- **Davies-Bouldin 점수**: ${this.formatNumber(metrics.davies_bouldin_score)} (낮을수록 좋음)\n`;
      }
      formattedText += '\n';
    }

    if (summary) {
      formattedText += '### 분석 요약\n';
      formattedText += summary;
    }

    return {
      content: [{
        type: 'text',
        text: formattedText
      }],
      analysisType: 'clustering'
    };
  }

  formatPCAResult(result) {
    const { explained_variance, explained_variance_ratio, n_components, loadings, summary } = result;
    
    let formattedText = '🔍 **주성분 분석(PCA) 결과**\n\n';
    
    if (n_components) {
      formattedText += `### 추출된 주성분: ${n_components}개\n\n`;
    }

    if (explained_variance_ratio) {
      formattedText += '### 설명 분산 비율\n';
      explained_variance_ratio.forEach((ratio, index) => {
        formattedText += `- **PC${index + 1}**: ${this.formatPercentage(ratio)}\n`;
      });
      
      // 누적 설명 분산
      const cumulative = explained_variance_ratio.reduce((acc, curr, index) => {
        acc.push((acc[index - 1] || 0) + curr);
        return acc;
      }, []);
      
      formattedText += '\n### 누적 설명 분산\n';
      cumulative.forEach((variance, index) => {
        formattedText += `- **PC1~PC${index + 1}**: ${this.formatPercentage(variance)}\n`;
      });
      formattedText += '\n';
    }

    if (loadings) {
      formattedText += '### 주요 로딩 (상위 5개 변수)\n';
      Object.entries(loadings).forEach(([pc, loading]) => {
        formattedText += `\n**${pc}:**\n`;
        const sortedLoading = Object.entries(loading)
          .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
          .slice(0, 5);
        
        sortedLoading.forEach(([variable, value]) => {
          formattedText += `  - ${variable}: ${this.formatNumber(value)}\n`;
        });
      });
      formattedText += '\n';
    }

    if (summary) {
      formattedText += '### 분석 요약\n';
      formattedText += summary;
    }

    return {
      content: [{
        type: 'text',
        text: formattedText
      }],
      analysisType: 'pca'
    };
  }

  formatMLModelResult(result) {
    const { model_type, metrics, feature_importance, model_info, summary } = result;
    
    let formattedText = `🤖 **${model_type || 'Machine Learning'} 모델 결과**\n\n`;
    
    // 모델 정보
    if (model_info) {
      formattedText += '### 모델 정보\n';
      if (model_info.algorithm) formattedText += `- **알고리즘**: ${model_info.algorithm}\n`;
      if (model_info.training_samples) formattedText += `- **훈련 샘플**: ${model_info.training_samples}개\n`;
      if (model_info.test_samples) formattedText += `- **테스트 샘플**: ${model_info.test_samples}개\n`;
      if (model_info.features) formattedText += `- **특성 수**: ${model_info.features}개\n`;
      formattedText += '\n';
    }
    
    // 성능 지표
    if (metrics) {
      formattedText += '### 모델 성능 지표\n';
      for (const [metric, value] of Object.entries(metrics)) {
        const emoji = this.getMetricEmoji(metric);
        const interpretation = this.interpretMetric(metric, value);
        formattedText += `${emoji} **${this.formatMetricName(metric)}**: ${this.formatNumber(value)}`;
        if (interpretation) formattedText += ` (${interpretation})`;
        formattedText += '\n';
      }
      formattedText += '\n';
    }

    // 특성 중요도
    if (feature_importance && feature_importance.length > 0) {
      formattedText += '### 특성 중요도 (상위 10개)\n';
      feature_importance.slice(0, 10).forEach((item, index) => {
        const bar = this.createProgressBar(item.importance, 0, Math.max(...feature_importance.map(f => f.importance)));
        formattedText += `${index + 1}. **${item.feature}**: ${this.formatNumber(item.importance)} ${bar}\n`;
      });
      formattedText += '\n';
    }

    if (summary) {
      formattedText += '### 모델 요약\n';
      formattedText += summary;
    }

    return {
      content: [{
        type: 'text',
        text: formattedText
      }],
      analysisType: 'ml_model'
    };
  }

  formatClassificationResult(result) {
    const { accuracy, precision, recall, f1_score, confusion_matrix, classification_report, summary } = result;
    
    let formattedText = '🎯 **분류 모델 결과**\n\n';
    
    // 주요 지표
    formattedText += '### 주요 성능 지표\n';
    if (accuracy !== undefined) formattedText += `🎯 **정확도**: ${this.formatPercentage(accuracy)}\n`;
    if (precision !== undefined) formattedText += `🔍 **정밀도**: ${this.formatPercentage(precision)}\n`;
    if (recall !== undefined) formattedText += `📊 **재현율**: ${this.formatPercentage(recall)}\n`;
    if (f1_score !== undefined) formattedText += `⚖️ **F1 점수**: ${this.formatPercentage(f1_score)}\n`;
    formattedText += '\n';

    // 혼동 행렬
    if (confusion_matrix) {
      formattedText += '### 혼동 행렬\n';
      formattedText += '```\n';
      confusion_matrix.forEach(row => {
        formattedText += row.map(val => String(val).padStart(6)).join(' ') + '\n';
      });
      formattedText += '```\n\n';
    }

    if (summary) {
      formattedText += '### 분석 요약\n';
      formattedText += summary;
    }

    return {
      content: [{
        type: 'text',
        text: formattedText
      }],
      analysisType: 'classification'
    };
  }

  formatRegressionResult(result) {
    const { r2_score, mse, rmse, mae, residuals_stats, summary } = result;
    
    let formattedText = '📈 **회귀 모델 결과**\n\n';
    
    // 성능 지표
    formattedText += '### 회귀 성능 지표\n';
    if (r2_score !== undefined) {
      const r2_quality = r2_score > 0.9 ? '우수' : r2_score > 0.7 ? '양호' : r2_score > 0.5 ? '보통' : '개선 필요';
      formattedText += `📊 **R² 점수**: ${this.formatNumber(r2_score)} (${r2_quality})\n`;
    }
    if (mse !== undefined) formattedText += `📉 **평균제곱오차 (MSE)**: ${this.formatNumber(mse)}\n`;
    if (rmse !== undefined) formattedText += `📏 **평균제곱근오차 (RMSE)**: ${this.formatNumber(rmse)}\n`;
    if (mae !== undefined) formattedText += `📐 **평균절대오차 (MAE)**: ${this.formatNumber(mae)}\n`;
    formattedText += '\n';

    // 잔차 통계
    if (residuals_stats) {
      formattedText += '### 잔차 분석\n';
      if (residuals_stats.mean !== undefined) formattedText += `- **잔차 평균**: ${this.formatNumber(residuals_stats.mean)}\n`;
      if (residuals_stats.std !== undefined) formattedText += `- **잔차 표준편차**: ${this.formatNumber(residuals_stats.std)}\n`;
      formattedText += '\n';
    }

    if (summary) {
      formattedText += '### 분석 요약\n';
      formattedText += summary;
    }

    return {
      content: [{
        type: 'text',
        text: formattedText
      }],
      analysisType: 'regression'
    };
  }

  formatVisualizationResult(result) {
    const { chart_path, chart_type, data_summary, insights } = result;
    
    let formattedText = `📈 **시각화 완료: ${chart_type || 'Chart'}**\n\n`;
    
    if (chart_path) {
      formattedText += `### 생성된 차트\n`;
      formattedText += `📁 **파일 경로**: \`${chart_path}\`\n\n`;
    }

    if (data_summary) {
      formattedText += '### 데이터 요약\n';
      if (data_summary.total_points) formattedText += `- **데이터 포인트**: ${data_summary.total_points.toLocaleString()}개\n`;
      if (data_summary.columns) formattedText += `- **사용된 컬럼**: ${data_summary.columns.join(', ')}\n`;
      if (data_summary.date_range) formattedText += `- **날짜 범위**: ${data_summary.date_range}\n`;
      formattedText += '\n';
    }

    if (insights && insights.length > 0) {
      formattedText += '### 시각화 인사이트\n';
      insights.forEach((insight, index) => {
        formattedText += `${index + 1}. ${insight}\n`;
      });
    }

    return {
      content: [{
        type: 'text',
        text: formattedText
      }],
      analysisType: 'visualization'
    };
  }

  formatTimeSeriesResult(result) {
    const { trend, seasonality, forecast, metrics, summary } = result;
    
    let formattedText = '📊 **시계열 분석 결과**\n\n';
    
    if (trend) {
      formattedText += '### 트렌드 분석\n';
      if (trend.direction) formattedText += `- **트렌드 방향**: ${trend.direction}\n`;
      if (trend.slope) formattedText += `- **기울기**: ${this.formatNumber(trend.slope)}\n`;
      if (trend.strength) formattedText += `- **트렌드 강도**: ${this.formatNumber(trend.strength)}\n`;
      formattedText += '\n';
    }

    if (seasonality) {
      formattedText += '### 계절성 분석\n';
      if (seasonality.detected) formattedText += `- **계절성 감지**: ${seasonality.detected ? '✅ 있음' : '❌ 없음'}\n`;
      if (seasonality.period) formattedText += `- **주기**: ${seasonality.period}\n`;
      if (seasonality.strength) formattedText += `- **계절성 강도**: ${this.formatNumber(seasonality.strength)}\n`;
      formattedText += '\n';
    }

    if (forecast) {
      formattedText += '### 예측 결과\n';
      if (forecast.periods) formattedText += `- **예측 기간**: ${forecast.periods}개 시점\n`;
      if (forecast.confidence_interval) formattedText += `- **신뢰구간**: ${forecast.confidence_interval}%\n`;
      formattedText += '\n';
    }

    if (metrics) {
      formattedText += '### 예측 성능\n';
      Object.entries(metrics).forEach(([metric, value]) => {
        formattedText += `- **${this.formatMetricName(metric)}**: ${this.formatNumber(value)}\n`;
      });
      formattedText += '\n';
    }

    if (summary) {
      formattedText += '### 분석 요약\n';
      formattedText += summary;
    }

    return {
      content: [{
        type: 'text',
        text: formattedText
      }],
      analysisType: 'timeseries'
    };
  }

  formatOutlierResult(result) {
    const { outlier_indices, outlier_scores, method, threshold, summary } = result;
    
    let formattedText = `🔍 **이상치 탐지 결과 (${method || 'Unknown method'})**\n\n`;
    
    if (outlier_indices) {
      const totalPoints = outlier_indices.total || 'Unknown';
      const outlierCount = outlier_indices.outliers ? outlier_indices.outliers.length : 0;
      const outlierRate = totalPoints !== 'Unknown' ? ((outlierCount / totalPoints) * 100).toFixed(2) : 'Unknown';
      
      formattedText += '### 이상치 탐지 결과\n';
      formattedText += `- **전체 데이터**: ${totalPoints}개\n`;
      formattedText += `- **이상치**: ${outlierCount}개 (${outlierRate}%)\n`;
      formattedText += `- **정상 데이터**: ${totalPoints - outlierCount}개\n`;
      
      if (threshold) {
        formattedText += `- **임계값**: ${this.formatNumber(threshold)}\n`;
      }
      formattedText += '\n';
    }

    if (outlier_scores) {
      formattedText += '### 이상치 점수 통계\n';
      if (outlier_scores.min !== undefined) formattedText += `- **최소값**: ${this.formatNumber(outlier_scores.min)}\n`;
      if (outlier_scores.max !== undefined) formattedText += `- **최댓값**: ${this.formatNumber(outlier_scores.max)}\n`;
      if (outlier_scores.mean !== undefined) formattedText += `- **평균**: ${this.formatNumber(outlier_scores.mean)}\n`;
      formattedText += '\n';
    }

    if (summary) {
      formattedText += '### 분석 요약\n';
      formattedText += summary;
    }

    return {
      content: [{
        type: 'text',
        text: formattedText
      }],
      analysisType: 'outlier_detection'
    };
  }

  formatFeatureEngineeringResult(result) {
    const { new_features, original_features, feature_types, summary } = result;
    
    let formattedText = '🔧 **특성 엔지니어링 결과**\n\n';
    
    if (original_features && new_features) {
      formattedText += '### 특성 변화\n';
      formattedText += `- **원본 특성**: ${original_features}개\n`;
      formattedText += `- **새로운 특성**: ${new_features}개\n`;
      formattedText += `- **총 특성**: ${original_features + new_features}개\n\n`;
    }

    if (feature_types) {
      formattedText += '### 생성된 특성 유형\n';
      Object.entries(feature_types).forEach(([type, count]) => {
        formattedText += `- **${type}**: ${count}개\n`;
      });
      formattedText += '\n';
    }

    if (summary) {
      formattedText += '### 분석 요약\n';
      formattedText += summary;
    }

    return {
      content: [{
        type: 'text',
        text: formattedText
      }],
      analysisType: 'feature_engineering'
    };
  }

  formatWorkflowResult(workflowResult) {
    const { workflowName, steps, finalResult, executionTime, summary } = workflowResult;
    
    let formattedText = `🔄 **워크플로우 실행 완료: ${workflowName}**\n\n`;
    
    // 실행 정보
    formattedText += '### 실행 정보\n';
    formattedText += `⏱️ **총 실행 시간**: ${Math.round(executionTime / 1000)}초\n`;
    formattedText += `📋 **실행된 단계**: ${steps.length}개\n`;
    
    const successCount = steps.filter(step => step.success).length;
    const successRate = ((successCount / steps.length) * 100).toFixed(1);
    formattedText += `✅ **성공률**: ${successRate}%\n\n`;
    
    // 단계별 결과
    formattedText += '### 실행된 단계\n';
    steps.forEach((step, index) => {
      const status = step.success ? '✅' : '❌';
      const duration = step.executionTime ? ` (${Math.round(step.executionTime)}ms)` : '';
      formattedText += `${status} **${index + 1}.** ${step.type} - ${step.method}${duration}\n`;
    });
    formattedText += '\n';
    
    // 최종 결과
if (finalResult) {
  if (finalResult.summary) {
    formattedText += '### 최종 결과\n';
    formattedText += finalResult.summary + '\n\n';
  }
  
  if (finalResult.artifacts && finalResult.artifacts.length > 0) {
    formattedText += '### 생성된 파일\n';
    finalResult.artifacts.forEach(artifact => {
      const emoji = artifact.type === 'visualization' ? '📊' : 
                       artifact.type === 'model' ? '🤖' : 
                       artifact.type === 'report' ? '📄' : '📁';
          formattedText += `${emoji} **${artifact.name}**: \`${artifact.path}\`\n`;
        });
        formattedText += '\n';
      }
    }

    if (summary) {
      formattedText += '### 워크플로우 요약\n';
      formattedText += summary;
    }

    return {
      content: [{
        type: 'text',
        text: formattedText
      }],
      analysisType: 'workflow'
    };
  }

  formatComparisonResult(result) {
    const { comparison_type, models, best_model, metrics_comparison, summary } = result;
    
    let formattedText = `🏆 **${comparison_type || '모델'} 비교 결과**\n\n`;
    
    if (best_model) {
      formattedText += `### 최고 성능 모델\n`;
      formattedText += `🥇 **${best_model.name}**\n`;
      if (best_model.score !== undefined) {
        formattedText += `- **점수**: ${this.formatNumber(best_model.score)}\n`;
      }
      if (best_model.metrics) {
        Object.entries(best_model.metrics).forEach(([metric, value]) => {
          const emoji = this.getMetricEmoji(metric);
          formattedText += `- ${emoji} **${this.formatMetricName(metric)}**: ${this.formatNumber(value)}\n`;
        });
      }
      formattedText += '\n';
    }

    if (models && models.length > 0) {
      formattedText += '### 전체 모델 순위\n';
      models.forEach((model, index) => {
        const rank = index + 1;
        const medal = rank === 1 ? '🥇' : rank === 2 ? '🥈' : rank === 3 ? '🥉' : '📍';
        formattedText += `${medal} **${rank}. ${model.name}**\n`;
        
        if (model.score !== undefined) {
          formattedText += `   점수: ${this.formatNumber(model.score)}\n`;
        }
        
        if (model.key_metrics) {
          Object.entries(model.key_metrics).slice(0, 3).forEach(([metric, value]) => {
            formattedText += `   ${this.formatMetricName(metric)}: ${this.formatNumber(value)}\n`;
          });
        }
        formattedText += '\n';
      });
    }

    if (metrics_comparison) {
      formattedText += '### 지표별 비교\n';
      Object.entries(metrics_comparison).forEach(([metric, values]) => {
        const emoji = this.getMetricEmoji(metric);
        formattedText += `${emoji} **${this.formatMetricName(metric)}**:\n`;
        
        const sortedValues = Object.entries(values)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 3);
        
        sortedValues.forEach(([modelName, value], index) => {
          const rank = index + 1;
          formattedText += `   ${rank}. ${modelName}: ${this.formatNumber(value)}\n`;
        });
        formattedText += '\n';
      });
    }

    if (summary) {
      formattedText += '### 비교 요약\n';
      formattedText += summary;
    }

    return {
      content: [{
        type: 'text',
        text: formattedText
      }],
      analysisType: 'comparison'
    };
  }

  formatGenericResult(result) {
    let formattedText = '📋 **분석 결과**\n\n';
    
    if (typeof result === 'object' && result !== null) {
      if (result.summary) {
        formattedText += '### 요약\n';
        formattedText += result.summary + '\n\n';
      }
      
      // 기타 속성들 표시
      const otherProps = Object.entries(result).filter(([key]) => key !== 'summary');
      if (otherProps.length > 0) {
        formattedText += '### 세부 정보\n';
        otherProps.forEach(([key, value]) => {
          formattedText += `**${key}**: ${this.formatValue(value)}\n`;
        });
      }
    } else {
      formattedText += String(result);
    }

    return {
      content: [{
        type: 'text',
        text: formattedText
      }],
      analysisType: 'generic'
    };
  }

  formatErrorResult(error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    
    return {
      content: [{
        type: 'text',
        text: `❌ **오류 발생**\n\n\`\`\`\n${errorMessage}\n\`\`\``
      }],
      isError: true,
      analysisType: 'error'
    };
  }

  // 유틸리티 메서드들
  formatNumber(value) {
    if (typeof value !== 'number' || isNaN(value)) {
      return String(value);
    }
    
    if (Math.abs(value) < 0.001 && value !== 0) {
      return value.toExponential(2);
    } else if (Math.abs(value) < 1) {
      return value.toFixed(4);
    } else if (Math.abs(value) < 1000) {
      return value.toFixed(2);
    } else {
      return value.toLocaleString('ko-KR');
    }
  }

  formatPercentage(value) {
    if (typeof value !== 'number' || isNaN(value)) {
      return String(value);
    }
    return `${(value * 100).toFixed(1)}%`;
  }

  formatValue(value) {
    if (typeof value === 'number') {
      return this.formatNumber(value);
    } else if (Array.isArray(value)) {
      if (value.length === 0) return '[]';
      if (value.length > 5) {
        return `[${value.slice(0, 5).map(v => this.formatValue(v)).join(', ')}, ... (총 ${value.length}개)]`;
      }
      return `[${value.map(v => this.formatValue(v)).join(', ')}]`;
    } else if (typeof value === 'object' && value !== null) {
      const entries = Object.entries(value);
      if (entries.length === 0) return '{}';
      if (entries.length > 3) {
        const preview = entries.slice(0, 3)
          .map(([k, v]) => `${k}: ${this.formatValue(v)}`)
          .join(', ');
        return `{${preview}, ... (총 ${entries.length}개 속성)}`;
      }
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
      'auc': '📈',
      'mse': '📉',
      'rmse': '📏',
      'mae': '📐',
      'r2_score': '📊',
      'r2': '📊',
      'silhouette_score': '🎯',
      'calinski_harabasz_score': '📊',
      'davies_bouldin_score': '📉'
    };
    return emojiMap[metric.toLowerCase()] || '📋';
  }

  formatMetricName(metric) {
    const nameMap = {
      'accuracy': '정확도',
      'precision': '정밀도',
      'recall': '재현율',
      'f1_score': 'F1 점수',
      'roc_auc': 'ROC AUC',
      'auc': 'AUC',
      'mse': '평균제곱오차',
      'rmse': '평균제곱근오차',
      'mae': '평균절대오차',
      'r2_score': 'R² 점수',
      'r2': 'R² 점수',
      'silhouette_score': '실루엣 점수',
      'calinski_harabasz_score': 'Calinski-Harabasz 점수',
      'davies_bouldin_score': 'Davies-Bouldin 점수'
    };
    return nameMap[metric.toLowerCase()] || metric;
  }

  interpretMetric(metric, value) {
    if (typeof value !== 'number' || isNaN(value)) return null;
    
    const metricLower = metric.toLowerCase();
    
    switch (metricLower) {
      case 'accuracy':
      case 'precision':
      case 'recall':
      case 'f1_score':
      case 'roc_auc':
      case 'auc':
        if (value > 0.9) return '우수';
        if (value > 0.8) return '양호';
        if (value > 0.7) return '보통';
        if (value > 0.6) return '개선 필요';
        return '부족';
      
      case 'r2_score':
      case 'r2':
        if (value > 0.9) return '우수';
        if (value > 0.7) return '양호';
        if (value > 0.5) return '보통';
        if (value > 0.3) return '개선 필요';
        return '부족';
      
      case 'silhouette_score':
        if (value > 0.7) return '우수';
        if (value > 0.5) return '양호';
        if (value > 0.25) return '보통';
        return '개선 필요';
      
      case 'davies_bouldin_score':
        if (value < 0.5) return '우수';
        if (value < 1.0) return '양호';
        if (value < 1.5) return '보통';
        return '개선 필요';
      
      default:
        return null;
    }
  }

  createProgressBar(value, min = 0, max = 1, length = 20) {
    const normalized = Math.max(0, Math.min(1, (value - min) / (max - min)));
    const filled = Math.round(normalized * length);
    const empty = length - filled;
    return '▓'.repeat(filled) + '░'.repeat(empty);
  }
}
