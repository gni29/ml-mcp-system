// tools/ml/predictor.js - 머신러닝 예측 도구
import { PythonExecutor } from '../common/python-executor.js';
import { ResultFormatter } from '../common/result-formatter.js';
import { Logger } from '../../utils/logger.js';
import { ConfigLoader } from '../../utils/config-loader.js';
import { FileManager } from '../common/file-manager.js';

export class MLPredictor {
  constructor() {
    this.pythonExecutor = new PythonExecutor();
    this.resultFormatter = new ResultFormatter();
    this.logger = new Logger();
    this.configLoader = new ConfigLoader();
    this.fileManager = new FileManager();
    this.modelCache = new Map();
    this.predictionHistory = [];
    
    this.initializePredictor();
  }

  async initializePredictor() {
    try {
      await this.configLoader.initialize();
      this.logger.info('MLPredictor 초기화 완료');
    } catch (error) {
      this.logger.error('MLPredictor 초기화 실패:', error);
    }
  }

  async loadModel(modelPath, modelType = 'sklearn') {
    try {
      this.logger.info(`모델 로딩 시작: ${modelPath}`);

      // 캐시 확인
      if (this.modelCache.has(modelPath)) {
        this.logger.info('캐시된 모델 사용');
        return this.modelCache.get(modelPath);
      }

      const scriptPath = 'python/ml/model_loader.py';
      const params = {
        model_path: modelPath,
        model_type: modelType
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const modelInfo = JSON.parse(result.output);
        
        // 모델 정보를 캐시에 저장
        this.modelCache.set(modelPath, {
          ...modelInfo,
          loaded_at: new Date().toISOString(),
          model_path: modelPath
        });

        this.logger.info('모델 로딩 완료');
        return modelInfo;
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('모델 로딩 실패:', error);
      throw error;
    }
  }

  async predict(modelPath, data, options = {}) {
    const {
      output_probabilities = false,
      batch_size = 1000,
      include_confidence = true,
      preprocessing_pipeline = null,
      feature_columns = null,
      return_explanations = false
    } = options;

    try {
      this.logger.info('예측 시작');

      // 모델 로딩
      const modelInfo = await this.loadModel(modelPath);

      const scriptPath = 'python/ml/prediction.py';
      const params = {
        model_path: modelPath,
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        output_probabilities,
        batch_size,
        include_confidence,
        preprocessing_pipeline,
        feature_columns: feature_columns ? feature_columns.join(',') : null,
        return_explanations
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const predictionResult = JSON.parse(result.output);
        
        // 예측 히스토리 저장
        this.savePredictionHistory(modelPath, predictionResult);

        return this.resultFormatter.formatAnalysisResult(predictionResult, 'ml_prediction');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('예측 실패:', error);
      throw error;
    }
  }

  async batchPredict(modelPath, dataFiles, options = {}) {
    const {
      output_dir = './results/predictions',
      parallel_processing = true,
      max_workers = 4,
      combine_results = true
    } = options;

    try {
      this.logger.info(`배치 예측 시작: ${dataFiles.length}개 파일`);

      // 출력 디렉토리 생성
      await this.fileManager.createDirectory(output_dir);

      const scriptPath = 'python/ml/batch_prediction.py';
      const params = {
        model_path: modelPath,
        data_files: dataFiles.join(','),
        output_dir,
        parallel_processing,
        max_workers,
        combine_results,
        ...options
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const batchResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(batchResult, 'batch_prediction');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('배치 예측 실패:', error);
      throw error;
    }
  }

  async predictWithExplanation(modelPath, data, options = {}) {
    const {
      explanation_method = 'shap',
      feature_names = null,
      num_features = 10,
      plot_explanations = true
    } = options;

    try {
      this.logger.info('설명 가능한 예측 시작');

      const scriptPath = 'python/ml/explainable_prediction.py';
      const params = {
        model_path: modelPath,
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        explanation_method,
        feature_names: feature_names ? feature_names.join(',') : null,
        num_features,
        plot_explanations
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const explainableResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(explainableResult, 'explainable_prediction');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('설명 가능한 예측 실패:', error);
      throw error;
    }
  }

  async realTimePrediction(modelPath, dataStream, options = {}) {
    const {
      buffer_size = 100,
      prediction_interval = 1000, // ms
      alert_threshold = null,
      save_predictions = true,
      output_file = null
    } = options;

    try {
      this.logger.info('실시간 예측 시작');

      const scriptPath = 'python/ml/realtime_prediction.py';
      const params = {
        model_path: modelPath,
        data_stream: JSON.stringify(dataStream),
        buffer_size,
        prediction_interval,
        alert_threshold,
        save_predictions,
        output_file
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const realtimeResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(realtimeResult, 'realtime_prediction');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('실시간 예측 실패:', error);
      throw error;
    }
  }

  async ensemblePredict(modelPaths, data, options = {}) {
    const {
      ensemble_method = 'voting',
      weights = null,
      output_individual_predictions = false,
      confidence_threshold = 0.5
    } = options;

    try {
      this.logger.info(`앙상블 예측 시작: ${modelPaths.length}개 모델`);

      const scriptPath = 'python/ml/ensemble_prediction.py';
      const params = {
        model_paths: modelPaths.join(','),
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        ensemble_method,
        weights: weights ? weights.join(',') : null,
        output_individual_predictions,
        confidence_threshold
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const ensembleResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(ensembleResult, 'ensemble_prediction');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('앙상블 예측 실패:', error);
      throw error;
    }
  }

  async calibrateModel(modelPath, calibrationData, options = {}) {
    const {
      calibration_method = 'platt',
      cv_folds = 5,
      output_path = null
    } = options;

    try {
      this.logger.info('모델 캘리브레이션 시작');

      const scriptPath = 'python/ml/model_calibration.py';
      const params = {
        model_path: modelPath,
        calibration_data_path: typeof calibrationData === 'string' ? calibrationData : null,
        calibration_data_json: typeof calibrationData === 'object' ? JSON.stringify(calibrationData) : null,
        calibration_method,
        cv_folds,
        output_path
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const calibrationResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(calibrationResult, 'model_calibration');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('모델 캘리브레이션 실패:', error);
      throw error;
    }
  }

  async predictUncertainty(modelPath, data, options = {}) {
    const {
      uncertainty_method = 'bootstrap',
      n_bootstrap = 100,
      confidence_level = 0.95,
      plot_uncertainty = true
    } = options;

    try {
      this.logger.info('불확실성 예측 시작');

      const scriptPath = 'python/ml/uncertainty_prediction.py';
      const params = {
        model_path: modelPath,
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        uncertainty_method,
        n_bootstrap,
        confidence_level,
        plot_uncertainty
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const uncertaintyResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(uncertaintyResult, 'uncertainty_prediction');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('불확실성 예측 실패:', error);
      throw error;
    }
  }

  async comparePredictions(predictions, groundTruth, options = {}) {
    const {
      metrics = ['accuracy', 'precision', 'recall', 'f1'],
      plot_comparison = true,
      detailed_analysis = true
    } = options;

    try {
      this.logger.info('예측 결과 비교 시작');

      const scriptPath = 'python/ml/prediction_comparison.py';
      const params = {
        predictions: JSON.stringify(predictions),
        ground_truth: JSON.stringify(groundTruth),
        metrics: metrics.join(','),
        plot_comparison,
        detailed_analysis
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const comparisonResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(comparisonResult, 'prediction_comparison');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('예측 결과 비교 실패:', error);
      throw error;
    }
  }

  savePredictionHistory(modelPath, predictionResult) {
    try {
      const historyEntry = {
        timestamp: new Date().toISOString(),
        model_path: modelPath,
        prediction_count: predictionResult.results?.predictions?.length || 0,
        prediction_type: predictionResult.results?.prediction_type || 'unknown',
        accuracy: predictionResult.results?.confidence || null,
        execution_time: predictionResult.metadata?.execution_time || null
      };

      this.predictionHistory.push(historyEntry);

      // 히스토리 크기 제한 (최근 1000개만 유지)
      if (this.predictionHistory.length > 1000) {
        this.predictionHistory = this.predictionHistory.slice(-500);
      }

      this.logger.debug('예측 히스토리 저장 완료');
    } catch (error) {
      this.logger.warn('예측 히스토리 저장 실패:', error);
    }
  }

  getLoadedModels() {
    return Array.from(this.modelCache.keys()).map(modelPath => ({
      model_path: modelPath,
      ...this.modelCache.get(modelPath)
    }));
  }

  getPredictionHistory(limit = 50) {
    return this.predictionHistory.slice(-limit);
  }

  clearModelCache() {
    this.modelCache.clear();
    this.logger.info('모델 캐시 클리어 완료');
  }

  getAvailablePredictionMethods() {
    return {
      predict: 'Standard Prediction - 기본 예측',
      batch_predict: 'Batch Prediction - 배치 예측',
      predict_with_explanation: 'Explainable Prediction - 설명 가능한 예측',
      realtime_prediction: 'Real-time Prediction - 실시간 예측',
      ensemble_predict: 'Ensemble Prediction - 앙상블 예측',
      predict_uncertainty: 'Uncertainty Prediction - 불확실성 예측'
    };
  }

  getPerformanceMetrics() {
    const totalPredictions = this.predictionHistory.length;
    const recentPredictions = this.predictionHistory.slice(-10);
    
    const avgExecutionTime = recentPredictions.length > 0 
      ? recentPredictions.reduce((sum, p) => sum + (p.execution_time || 0), 0) / recentPredictions.length 
      : 0;

    const avgConfidence = recentPredictions.length > 0 
      ? recentPredictions.reduce((sum, p) => sum + (p.accuracy || 0), 0) / recentPredictions.length 
      : 0;

    return {
      total_predictions: totalPredictions,
      loaded_models: this.modelCache.size,
      average_execution_time: avgExecutionTime,
      average_confidence: avgConfidence,
      cache_hit_rate: this.calculateCacheHitRate()
    };
  }

  calculateCacheHitRate() {
    // 캐시 히트율 계산 로직 (실제 구현 시 더 정교하게)
    return this.modelCache.size > 0 ? 0.8 : 0; // 임시값
  }
}