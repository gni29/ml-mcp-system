// tools/ml/evaluator.js - 머신러닝 모델 평가 도구
import { PythonExecutor } from '../common/python-executor.js';
import { ResultFormatter } from '../common/result-formatter.js';
import { Logger } from '../../utils/logger.js';
import { ConfigLoader } from '../../utils/config-loader.js';
import { FileManager } from '../common/file-manager.js';

export class MLEvaluator {
  constructor() {
    this.pythonExecutor = new PythonExecutor();
    this.resultFormatter = new ResultFormatter();
    this.logger = new Logger();
    this.configLoader = new ConfigLoader();
    this.fileManager = new FileManager();
    this.evaluationHistory = [];
    
    this.initializeEvaluator();
  }

  async initializeEvaluator() {
    try {
      await this.configLoader.initialize();
      this.logger.info('MLEvaluator 초기화 완료');
    } catch (error) {
      this.logger.error('MLEvaluator 초기화 실패:', error);
    }
  }

  async evaluateClassification(modelPath, testData, options = {}) {
    const {
      target_column,
      metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'],
      class_names = null,
      plot_confusion_matrix = true,
      plot_roc_curve = true,
      plot_precision_recall = true,
      cross_validation = false,
      cv_folds = 5
    } = options;

    try {
      this.logger.info('분류 모델 평가 시작');

      if (!target_column) {
        throw new Error('target_column이 필요합니다.');
      }

      const scriptPath = 'python/ml/evaluation/classification_evaluation.py';
      const params = {
        model_path: modelPath,
        test_data_path: typeof testData === 'string' ? testData : null,
        test_data_json: typeof testData === 'object' ? JSON.stringify(testData) : null,
        target_column,
        metrics: metrics.join(','),
        class_names: class_names ? class_names.join(',') : null,
        plot_confusion_matrix,
        plot_roc_curve,
        plot_precision_recall,
        cross_validation,
        cv_folds
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const evaluationResult = JSON.parse(result.output);
        this.saveEvaluationHistory('classification', modelPath, evaluationResult);
        return this.resultFormatter.formatAnalysisResult(evaluationResult, 'model_evaluation');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('분류 모델 평가 실패:', error);
      throw error;
    }
  }

  async evaluateRegression(modelPath, testData, options = {}) {
    const {
      target_column,
      metrics = ['mse', 'rmse', 'mae', 'r2_score', 'mape'],
      plot_predictions = true,
      plot_residuals = true,
      plot_feature_importance = true,
      cross_validation = false,
      cv_folds = 5
    } = options;

    try {
      this.logger.info('회귀 모델 평가 시작');

      if (!target_column) {
        throw new Error('target_column이 필요합니다.');
      }

      const scriptPath = 'python/ml/evaluation/regression_evaluation.py';
      const params = {
        model_path: modelPath,
        test_data_path: typeof testData === 'string' ? testData : null,
        test_data_json: typeof testData === 'object' ? JSON.stringify(testData) : null,
        target_column,
        metrics: metrics.join(','),
        plot_predictions,
        plot_residuals,
        plot_feature_importance,
        cross_validation,
        cv_folds
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const evaluationResult = JSON.parse(result.output);
        this.saveEvaluationHistory('regression', modelPath, evaluationResult);
        return this.resultFormatter.formatAnalysisResult(evaluationResult, 'model_evaluation');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('회귀 모델 평가 실패:', error);
      throw error;
    }
  }

  async evaluateClustering(modelPath, data, options = {}) {
    const {
      metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score'],
      plot_clusters = true,
      plot_silhouette = true,
      true_labels = null
    } = options;

    try {
      this.logger.info('클러스터링 모델 평가 시작');

      const scriptPath = 'python/ml/evaluation/clustering_evaluation.py';
      const params = {
        model_path: modelPath,
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        metrics: metrics.join(','),
        plot_clusters,
        plot_silhouette,
        true_labels: true_labels ? JSON.stringify(true_labels) : null
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const evaluationResult = JSON.parse(result.output);
        this.saveEvaluationHistory('clustering', modelPath, evaluationResult);
        return this.resultFormatter.formatAnalysisResult(evaluationResult, 'model_evaluation');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('클러스터링 모델 평가 실패:', error);
      throw error;
    }
  }

  async crossValidateModel(modelPath, data, options = {}) {
    const {
      target_column,
      cv_folds = 5,
      cv_strategy = 'kfold',
      scoring = 'accuracy',
      shuffle = true,
      random_state = 42,
      plot_cv_scores = true
    } = options;

    try {
      this.logger.info('교차 검증 시작');

      const scriptPath = 'python/ml/evaluation/cross_validation.py';
      const params = {
        model_path: modelPath,
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        target_column,
        cv_folds,
        cv_strategy,
        scoring,
        shuffle,
        random_state,
        plot_cv_scores
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const cvResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(cvResult, 'cross_validation');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('교차 검증 실패:', error);
      throw error;
    }
  }

  async compareModels(modelPaths, testData, options = {}) {
    const {
      target_column,
      evaluation_metrics = ['accuracy', 'precision', 'recall', 'f1_score'],
      model_names = null,
      plot_comparison = true,
      statistical_test = true
    } = options;

    try {
      this.logger.info(`모델 비교 시작: ${modelPaths.length}개 모델`);

      const scriptPath = 'python/ml/evaluation/model_comparison.py';
      const params = {
        model_paths: modelPaths.join(','),
        test_data_path: typeof testData === 'string' ? testData : null,
        test_data_json: typeof testData === 'object' ? JSON.stringify(testData) : null,
        target_column,
        evaluation_metrics: evaluation_metrics.join(','),
        model_names: model_names ? model_names.join(',') : null,
        plot_comparison,
        statistical_test
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const comparisonResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(comparisonResult, 'model_comparison');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('모델 비교 실패:', error);
      throw error;
    }
  }

  async evaluateFeatureImportance(modelPath, data, options = {}) {
    const {
      method = 'default',
      feature_names = null,
      plot_importance = true,
      top_n_features = 20
    } = options;

    try {
      this.logger.info('피처 중요도 평가 시작');

      const scriptPath = 'python/ml/evaluation/feature_importance.py';
      const params = {
        model_path: modelPath,
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        method,
        feature_names: feature_names ? feature_names.join(',') : null,
        plot_importance,
        top_n_features
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const importanceResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(importanceResult, 'feature_importance');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('피처 중요도 평가 실패:', error);
      throw error;
    }
  }

  async evaluateModelRobustness(modelPath, data, options = {}) {
    const {
      target_column,
      noise_levels = [0.01, 0.05, 0.1, 0.2],
      perturbation_methods = ['gaussian', 'uniform'],
      n_iterations = 10,
      plot_robustness = true
    } = options;

    try {
      this.logger.info('모델 강건성 평가 시작');

      const scriptPath = 'python/ml/evaluation/robustness_evaluation.py';
      const params = {
        model_path: modelPath,
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        target_column,
        noise_levels: noise_levels.join(','),
        perturbation_methods: perturbation_methods.join(','),
        n_iterations,
        plot_robustness
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const robustnessResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(robustnessResult, 'robustness_evaluation');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('모델 강건성 평가 실패:', error);
      throw error;
    }
  }

  async evaluateBias(modelPath, data, options = {}) {
    const {
      target_column,
      protected_attributes,
      fairness_metrics = ['demographic_parity', 'equalized_odds', 'calibration'],
      plot_bias_analysis = true
    } = options;

    try {
      this.logger.info('모델 편향성 평가 시작');

      if (!protected_attributes || protected_attributes.length === 0) {
        throw new Error('protected_attributes가 필요합니다.');
      }

      const scriptPath = 'python/ml/evaluation/bias_evaluation.py';
      const params = {
        model_path: modelPath,
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        target_column,
        protected_attributes: protected_attributes.join(','),
        fairness_metrics: fairness_metrics.join(','),
        plot_bias_analysis
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const biasResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(biasResult, 'bias_evaluation');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('모델 편향성 평가 실패:', error);
      throw error;
    }
  }

  async evaluatePerformanceOverTime(modelPath, timeSeriesData, options = {}) {
    const {
      target_column,
      time_column,
      window_size = 30,
      metrics = ['accuracy', 'precision', 'recall'],
      plot_performance_drift = true
    } = options;

    try {
      this.logger.info('시간에 따른 성능 평가 시작');

      const scriptPath = 'python/ml/evaluation/performance_over_time.py';
      const params = {
        model_path: modelPath,
        data_path: typeof timeSeriesData === 'string' ? timeSeriesData : null,
        data_json: typeof timeSeriesData === 'object' ? JSON.stringify(timeSeriesData) : null,
        target_column,
        time_column,
        window_size,
        metrics: metrics.join(','),
        plot_performance_drift
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const performanceResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(performanceResult, 'performance_over_time');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('시간에 따른 성능 평가 실패:', error);
      throw error;
    }
  }

  async comprehensiveEvaluation(modelPath, testData, options = {}) {
    const {
      target_column,
      model_type = 'classification',
      include_cross_validation = true,
      include_feature_importance = true,
      include_robustness = false,
      include_bias = false,
      protected_attributes = null,
      generate_report = true
    } = options;

    try {
      this.logger.info('종합 모델 평가 시작');

      const results = {
        evaluation_type: 'comprehensive',
        timestamp: new Date().toISOString(),
        model_path: modelPath,
        results: {},
        execution_info: {
          started_at: new Date().toISOString(),
          evaluations_requested: []
        }
      };

      // 기본 평가
      if (model_type === 'classification') {
        results.execution_info.evaluations_requested.push('classification_evaluation');
        try {
          results.results.classification_evaluation = await this.evaluateClassification(
            modelPath, testData, { target_column, ...options }
          );
        } catch (error) {
          this.logger.warn('분류 평가 실패:', error);
          results.results.classification_evaluation = { error: true, message: error.message };
        }
      } else if (model_type === 'regression') {
        results.execution_info.evaluations_requested.push('regression_evaluation');
        try {
          results.results.regression_evaluation = await this.evaluateRegression(
            modelPath, testData, { target_column, ...options }
          );
        } catch (error) {
          this.logger.warn('회귀 평가 실패:', error);
          results.results.regression_evaluation = { error: true, message: error.message };
        }
      }

      // 교차 검증
      if (include_cross_validation) {
        results.execution_info.evaluations_requested.push('cross_validation');
        try {
          results.results.cross_validation = await this.crossValidateModel(
            modelPath, testData, { target_column, ...options }
          );
        } catch (error) {
          this.logger.warn('교차 검증 실패:', error);
          results.results.cross_validation = { error: true, message: error.message };
        }
      }

      // 피처 중요도
      if (include_feature_importance) {
        results.execution_info.evaluations_requested.push('feature_importance');
        try {
          results.results.feature_importance = await this.evaluateFeatureImportance(
            modelPath, testData, options
          );
        } catch (error) {
          this.logger.warn('피처 중요도 평가 실패:', error);
          results.results.feature_importance = { error: true, message: error.message };
        }
      }

      // 강건성 평가
      if (include_robustness) {
        results.execution_info.evaluations_requested.push('robustness_evaluation');
        try {
          results.results.robustness_evaluation = await this.evaluateModelRobustness(
            modelPath, testData, { target_column, ...options }
          );
        } catch (error) {
          this.logger.warn('강건성 평가 실패:', error);
          results.results.robustness_evaluation = { error: true, message: error.message };
        }
      }

      // 편향성 평가
      if (include_bias && protected_attributes) {
        results.execution_info.evaluations_requested.push('bias_evaluation');
        try {
          results.results.bias_evaluation = await this.evaluateBias(
            modelPath, testData, { target_column, protected_attributes, ...options }
          );
        } catch (error) {
          this.logger.warn('편향성 평가 실패:', error);
          results.results.bias_evaluation = { error: true, message: error.message };
        }
      }

      // 실행 시간 기록
      results.execution_info.completed_at = new Date().toISOString();
      results.execution_info.total_duration_ms = Date.now() - new Date(results.execution_info.started_at).getTime();

      // 평가 요약 생성
      if (generate_report) {
        results.summary = this.generateEvaluationSummary(results.results);
      }

      this.logger.info('종합 모델 평가 완료');
      return this.resultFormatter.formatAnalysisResult(results, 'comprehensive_evaluation');

    } catch (error) {
      this.logger.error('종합 모델 평가 실패:', error);
      throw error;
    }
  }

  generateEvaluationSummary(evaluationResults) {
    const summary = {
      successful_evaluations: [],
      failed_evaluations: [],
      key_metrics: {},
      recommendations: []
    };

    // 성공/실패 평가 분류
    Object.entries(evaluationResults).forEach(([evaluationType, result]) => {
      if (result.error) {
        summary.failed_evaluations.push(evaluationType);
      } else {
        summary.successful_evaluations.push(evaluationType);
        
        // 주요 메트릭 추출
        if (result.results && result.results.metrics) {
          summary.key_metrics[evaluationType] = result.results.metrics;
        }
      }
    });

    // 권장사항 생성
    if (summary.successful_evaluations.includes('classification_evaluation')) {
      const metrics = summary.key_metrics.classification_evaluation;
      if (metrics && metrics.accuracy < 0.8) {
        summary.recommendations.push('모델 성능이 낮습니다. 데이터 추가 수집이나 모델 개선을 고려해보세요.');
      }
    }

    if (summary.successful_evaluations.includes('cross_validation')) {
      summary.recommendations.push('교차 검증 결과를 바탕으로 모델의 일반화 성능을 확인하세요.');
    }

    if (summary.successful_evaluations.includes('feature_importance')) {
      summary.recommendations.push('중요한 피처들을 확인하여 도메인 지식과 일치하는지 검토해보세요.');
    }

    return summary;
  }

  saveEvaluationHistory(evaluationType, modelPath, evaluationResult) {
    try {
      const historyEntry = {
        timestamp: new Date().toISOString(),
        evaluation_type: evaluationType,
        model_path: modelPath,
        metrics: evaluationResult.results?.metrics || {},
        execution_time: evaluationResult.metadata?.execution_time || null
      };

      this.evaluationHistory.push(historyEntry);

      // 히스토리 크기 제한 (최근 500개만 유지)
      if (this.evaluationHistory.length > 500) {
        this.evaluationHistory = this.evaluationHistory.slice(-250);
      }

      this.logger.debug('평가 히스토리 저장 완료');
    } catch (error) {
      this.logger.warn('평가 히스토리 저장 실패:', error);
    }
  }

  getEvaluationHistory(limit = 50) {
    return this.evaluationHistory.slice(-limit);
  }

  getAvailableEvaluationMethods() {
    return {
      classification: 'Classification Evaluation - 분류 모델 평가',
      regression: 'Regression Evaluation - 회귀 모델 평가',
      clustering: 'Clustering Evaluation - 클러스터링 평가',
      cross_validation: 'Cross Validation - 교차 검증',
      model_comparison: 'Model Comparison - 모델 비교',
      feature_importance: 'Feature Importance - 피처 중요도',
      robustness: 'Robustness Evaluation - 강건성 평가',
      bias: 'Bias Evaluation - 편향성 평가',
      performance_over_time: 'Performance Over Time - 시간에 따른 성능'
    };
  }

  getPerformanceMetrics() {
    const totalEvaluations = this.evaluationHistory.length;
    const recentEvaluations = this.evaluationHistory.slice(-10);
    
    const avgExecutionTime = recentEvaluations.length > 0 
      ? recentEvaluations.reduce((sum, e) => sum + (e.execution_time || 0), 0) / recentEvaluations.length 
      : 0;

    const evaluationTypes = [...new Set(this.evaluationHistory.map(e => e.evaluation_type))];

    return {
      total_evaluations: totalEvaluations,
      unique_evaluation_types: evaluationTypes.length,
      average_execution_time: avgExecutionTime,
      most_common_evaluation: this.getMostCommonEvaluationType()
    };
  }

  getMostCommonEvaluationType() {
    if (this.evaluationHistory.length === 0) return 'none';
    
    const typeCounts = {};
    this.evaluationHistory.forEach(e => {
      typeCounts[e.evaluation_type] = (typeCounts[e.evaluation_type] || 0) + 1;
    });

    return Object.keys(typeCounts).reduce((a, b) => typeCounts[a] > typeCounts[b] ? a : b);
  }
}