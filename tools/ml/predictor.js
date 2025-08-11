// tools/ml/predictor.js - 머신러닝 예측 및 추론 인터페이스
import { Logger } from '../../utils/logger.js';
import { PythonExecutor } from '../common/python-executor.js';
import { ResultFormatter } from '../common/result-formatter.js';

export class MLPredictor {
  constructor() {
    this.logger = new Logger();
    this.pythonExecutor = new PythonExecutor();
    this.resultFormatter = new ResultFormatter();
    this.predictionHistory = [];
    this.loadedModels = new Map();
  }

  async initialize() {
    try {
      await this.pythonExecutor.initialize();
      this.logger.info('MLPredictor 초기화 완료');
    } catch (error) {
      this.logger.error('MLPredictor 초기화 실패:', error);
      throw error;
    }
  }

  async loadModel(modelPath, modelType = 'auto') {
    try {
      this.logger.info('모델 로드 시작', { modelPath, modelType });

      const scriptPath = 'python/ml/model_loader.py';
      const args = {
        model_path: modelPath,
        model_type: modelType,
        load_preprocessor: true
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 60000
      });

      if (result.success) {
        const loadResult = JSON.parse(result.output);
        this.loadedModels.set(modelPath, {
          model_info: loadResult,
          loaded_at: new Date().toISOString()
        });
        this.logger.info('모델 로드 완료', { modelPath });
        return loadResult;
      } else {
        throw new Error(`모델 로드 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('모델 로드 실패:', error);
      throw error;
    }
  }

  async predict(data, modelPath, options = {}) {
    try {
      this.logger.info('예측 시작', { modelPath });

      const {
        return_probabilities = false,
        batch_size = 1000,
        confidence_interval = false,
        feature_importance = false,
        explain_predictions = false
      } = options;

      const scriptPath = 'python/ml/prediction.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        model_path: modelPath,
        return_probabilities,
        batch_size,
        confidence_interval,
        feature_importance,
        explain_predictions
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 300000 // 5분
      });

      if (result.success) {
        const predictionResult = JSON.parse(result.output);
        this.recordPredictionHistory(modelPath, options, predictionResult);
        return this.resultFormatter.formatAnalysisResult(predictionResult, 'ml_prediction');
      } else {
        throw new Error(`예측 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('예측 실패:', error);
      throw error;
    }
  }

  async predictSingle(inputData, modelPath, options = {}) {
    try {
      this.logger.info('단일 예측 시작', { modelPath });

      const {
        return_probabilities = true,
        explain_prediction = true,
        confidence_score = true
      } = options;

      const scriptPath = 'python/ml/single_prediction.py';
      const args = {
        input_data: inputData,
        model_path: modelPath,
        return_probabilities,
        explain_prediction,
        confidence_score
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 30000
      });

      if (result.success) {
        const singlePredResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(singlePredResult, 'single_prediction');
      } else {
        throw new Error(`단일 예측 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('단일 예측 실패:', error);
      throw error;
    }
  }

  async predictBatch(dataList, modelPath, options = {}) {
    try {
      this.logger.info('배치 예측 시작', { 
        modelPath, 
        batchSize: dataList.length 
      });

      const {
        batch_size = 100,
        parallel_processing = false,
        return_probabilities = false,
        progress_callback = null
      } = options;

      const scriptPath = 'python/ml/batch_prediction.py';
      const args = {
        data_list: dataList,
        model_path: modelPath,
        batch_size,
        parallel_processing,
        return_probabilities
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 600000 // 10분
      });

      if (result.success) {
        const batchResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(batchResult, 'batch_prediction');
      } else {
        throw new Error(`배치 예측 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('배치 예측 실패:', error);
      throw error;
    }
  }

  async predictWithUncertainty(data, modelPath, options = {}) {
    try {
      this.logger.info('불확실성을 포함한 예측 시작', { modelPath });

      const {
        method = 'bootstrap',
        n_samples = 100,
        confidence_level = 0.95,
        return_intervals = true
      } = options;

      const scriptPath = 'python/ml/uncertainty_prediction.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        model_path: modelPath,
        method,
        n_samples,
        confidence_level,
        return_intervals
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 600000 // 10분
      });

      if (result.success) {
        const uncertaintyResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(uncertaintyResult, 'uncertainty_prediction');
      } else {
        throw new Error(`불확실성 예측 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('불확실성 예측 실패:', error);
      throw error;
    }
  }

  async predictTimeSeries(data, modelPath, options = {}) {
    try {
      this.logger.info('시계열 예측 시작', { modelPath });

      const {
        forecast_horizon = 30,
        confidence_intervals = true,
        seasonal_decomposition = false,
        return_components = false
      } = options;

      const scriptPath = 'python/ml/timeseries_prediction.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        model_path: modelPath,
        forecast_horizon,
        confidence_intervals,
        seasonal_decomposition,
        return_components
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 300000 // 5분
      });

      if (result.success) {
        const timeseriesResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(timeseriesResult, 'timeseries_prediction');
      } else {
        throw new Error(`시계열 예측 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('시계열 예측 실패:', error);
      throw error;
    }
  }

  async explainPredictions(data, modelPath, options = {}) {
    try {
      this.logger.info('예측 설명 시작', { modelPath });

      const {
        explanation_method = 'shap',
        sample_size = 100,
        feature_names = null,
        visualize = true
      } = options;

      const scriptPath = 'python/ml/prediction_explanation.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        model_path: modelPath,
        explanation_method,
        sample_size,
        feature_names,
        visualize
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 600000 // 10분
      });

      if (result.success) {
        const explanationResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(explanationResult, 'prediction_explanation');
      } else {
        throw new Error(`예측 설명 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('예측 설명 실패:', error);
      throw error;
    }
  }

  async validatePredictions(predictions, actualValues, modelType = 'auto') {
    try {
      this.logger.info('예측 검증 시작', { modelType });

      const scriptPath = 'python/ml/prediction_validation.py';
      const args = {
        predictions: predictions,
        actual_values: actualValues,
        model_type: modelType,
        compute_metrics: true,
        visualize_results: true
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 120000
      });

      if (result.success) {
        const validationResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(validationResult, 'prediction_validation');
      } else {
        throw new Error(`예측 검증 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('예측 검증 실패:', error);
      throw error;
    }
  }

  async monitorPredictions(predictionData, options = {}) {
    try {
      this.logger.info('예측 모니터링 시작');

      const {
        drift_detection = true,
        performance_tracking = true,
        alert_thresholds = {},
        save_results = true
      } = options;

      const scriptPath = 'python/ml/prediction_monitoring.py';
      const args = {
        prediction_data: predictionData,
        drift_detection,
        performance_tracking,
        alert_thresholds,
        save_results
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 180000
      });

      if (result.success) {
        const monitoringResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(monitoringResult, 'prediction_monitoring');
      } else {
        throw new Error(`예측 모니터링 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('예측 모니터링 실패:', error);
      throw error;
    }
  }

  async compareModels(data, modelPaths, options = {}) {
    try {
      this.logger.info('모델 비교 시작', { modelCount: modelPaths.length });

      const {
        metrics = ['accuracy', 'precision', 'recall', 'f1'],
        cross_validation = true,
        cv_folds = 5,
        visualize_comparison = true
      } = options;

      const scriptPath = 'python/ml/model_comparison.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        model_paths: modelPaths,
        metrics,
        cross_validation,
        cv_folds,
        visualize_comparison
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 900000 // 15분
      });

      if (result.success) {
        const comparisonResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(comparisonResult, 'model_comparison');
      } else {
        throw new Error(`모델 비교 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('모델 비교 실패:', error);
      throw error;
    }
  }

  async predictWithEnsemble(data, modelPaths, options = {}) {
    try {
      this.logger.info('앙상블 예측 시작', { modelCount: modelPaths.length });

      const {
        ensemble_method = 'voting',
        weights = null,
        return_individual_predictions = false,
        confidence_scoring = true
      } = options;

      const scriptPath = 'python/ml/ensemble_prediction.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        model_paths: modelPaths,
        ensemble_method,
        weights,
        return_individual_predictions,
        confidence_scoring
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 600000 // 10분
      });

      if (result.success) {
        const ensembleResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(ensembleResult, 'ensemble_prediction');
      } else {
        throw new Error(`앙상블 예측 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('앙상블 예측 실패:', error);
      throw error;
    }
  }

  recordPredictionHistory(modelPath, options, result) {
    const record = {
      timestamp: new Date().toISOString(),
      model_path: modelPath,
      options,
      success: !result.error,
      prediction_count: result.prediction_count || null,
      prediction_time: result.prediction_time || null
    };

    this.predictionHistory.push(record);
    
    // 히스토리 크기 제한 (최대 100개)
    if (this.predictionHistory.length > 100) {
      this.predictionHistory = this.predictionHistory.slice(-50);
    }
  }

  // 모델 정보 조회 메서드들
  async getModelInfo(modelPath) {
    try {
      const scriptPath = 'python/ml/model_info.py';
      const args = { model_path: modelPath };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 30000
      });

      if (result.success) {
        return JSON.parse(result.output);
      } else {
        throw new Error(`모델 정보 조회 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('모델 정보 조회 실패:', error);
      throw error;
    }
  }

  async listAvailableModels(modelsDirectory = './models') {
    try {
      const scriptPath = 'python/ml/list_models.py';
      const args = { models_directory: modelsDirectory };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 30000
      });

      if (result.success) {
        return JSON.parse(result.output);
      } else {
        throw new Error(`모델 목록 조회 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('모델 목록 조회 실패:', error);
      throw error;
    }
  }

  async validateModelCompatibility(data, modelPath) {
    try {
      const scriptPath = 'python/ml/model_compatibility.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        model_path: modelPath
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 60000
      });

      if (result.success) {
        return JSON.parse(result.output);
      } else {
        throw new Error(`모델 호환성 검사 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('모델 호환성 검사 실패:', error);
      throw error;
    }
  }

  // 유틸리티 메서드들
  getPredictionHistory(limit = 10) {
    return this.predictionHistory.slice(-limit);
  }

  getLoadedModels() {
    return Array.from(this.loadedModels.entries()).map(([path, info]) => ({
      model_path: path,
      ...info
    }));
  }

  unloadModel(modelPath) {
    const removed = this.loadedModels.delete(modelPath);
    if (removed) {
      this.logger.info('모델 언로드 완료', { modelPath });
    }
    return removed;
  }

  unloadAllModels() {
    const count = this.loadedModels.size;
    this.loadedModels.clear();
    this.logger.info('모든 모델 언로드 완료', { count });
    return count;
  }

  getSupportedPredictionTypes() {
    return [
      'single_prediction',
      'batch_prediction', 
      'streaming_prediction',
      'timeseries_prediction',
      'uncertainty_prediction',
      'ensemble_prediction'
    ];
  }

  getSupportedExplanationMethods() {
    return ['shap', 'lime', 'permutation_importance', 'partial_dependence'];
  }

  getSupportedEnsembleMethods() {
    return ['voting', 'averaging', 'weighted_average', 'stacking', 'blending'];
  }

  // 예측 성능 통계
  getPredictionStats() {
    if (this.predictionHistory.length === 0) {
      return {
        total_predictions: 0,
        success_rate: 0,
        average_prediction_time: 0,
        most_used_models: []
      };
    }

    const successful = this.predictionHistory.filter(p => p.success);
    const modelUsage = {};
    let totalTime = 0;
    let timeCount = 0;

    this.predictionHistory.forEach(pred => {
      // 모델 사용 빈도 계산
      modelUsage[pred.model_path] = (modelUsage[pred.model_path] || 0) + 1;
      
      // 평균 예측 시간 계산
      if (pred.prediction_time) {
        totalTime += pred.prediction_time;
        timeCount++;
      }
    });

    const mostUsedModels = Object.entries(modelUsage)
      .map(([path, count]) => ({ model_path: path, usage_count: count }))
      .sort((a, b) => b.usage_count - a.usage_count)
      .slice(0, 5);

    return {
      total_predictions: this.predictionHistory.length,
      successful_predictions: successful.length,
      success_rate: (successful.length / this.predictionHistory.length) * 100,
      average_prediction_time: timeCount > 0 ? totalTime / timeCount : 0,
      most_used_models: mostUsedModels,
      loaded_models_count: this.loadedModels.size
    };
  }

  async getPredictorStatus() {
    return {
      python_executor_status: await this.pythonExecutor.getExecutionStats(),
      prediction_history_count: this.predictionHistory.length,
      loaded_models_count: this.loadedModels.size,
      prediction_stats: this.getPredictionStats(),
      supported_features: {
        prediction_types: this.getSupportedPredictionTypes(),
        explanation_methods: this.getSupportedExplanationMethods(),
        ensemble_methods: this.getSupportedEnsembleMethods()
      }
    };
  }

  async cleanup() {
    this.unloadAllModels();
    await this.pythonExecutor.shutdown();
    this.logger.info('MLPredictor 정리 완료');
  }
}