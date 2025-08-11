// tools/ml/trainer.js - 머신러닝 모델 훈련 인터페이스
import { Logger } from '../../utils/logger.js';
import { PythonExecutor } from '../common/python-executor.js';
import { ResultFormatter } from '../common/result-formatter.js';

export class MLTrainer {
  constructor() {
    this.logger = new Logger();
    this.pythonExecutor = new PythonExecutor();
    this.resultFormatter = new ResultFormatter();
    this.trainingHistory = [];
    this.supportedModels = {
      supervised: ['classification', 'regression', 'ensemble'],
      unsupervised: ['kmeans', 'hierarchical', 'dbscan'],
      deep_learning: ['neural_network', 'cnn', 'rnn']
    };
  }

  async initialize() {
    try {
      await this.pythonExecutor.initialize();
      this.logger.info('MLTrainer 초기화 완료');
    } catch (error) {
      this.logger.error('MLTrainer 초기화 실패:', error);
      throw error;
    }
  }

  async trainModel(data, modelConfig) {
    try {
      this.logger.info('모델 훈련 시작', { modelType: modelConfig.model_type });

      const {
        model_type,
        model_category = 'supervised',
        target_column = null,
        algorithm = 'auto',
        parameters = {},
        validation_split = 0.2,
        cross_validation = false,
        save_model = true
      } = modelConfig;

      // 모델 타입 검증
      this.validateModelConfig(modelConfig);

      // 적절한 Python 스크립트 선택
      const scriptPath = this.getTrainingScriptPath(model_category, model_type);
      
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        model_type,
        model_category,
        target_column,
        algorithm,
        parameters,
        validation_split,
        cross_validation,
        save_model
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 600000 // 10분
      });

      if (result.success) {
        const trainingResult = JSON.parse(result.output);
        this.recordTrainingHistory(modelConfig, trainingResult);
        return this.resultFormatter.formatAnalysisResult(trainingResult, 'model_training');
      } else {
        throw new Error(`모델 훈련 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('모델 훈련 실패:', error);
      throw error;
    }
  }

  async trainClassificationModel(data, targetColumn, options = {}) {
    try {
      this.logger.info('분류 모델 훈련 시작');

      const {
        algorithm = 'random_forest',
        test_size = 0.2,
        random_state = 42,
        cross_validation = true,
        hyperparameter_tuning = false,
        feature_selection = false
      } = options;

      const scriptPath = 'python/ml/supervised/classification.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        target_column: targetColumn,
        algorithm,
        test_size,
        random_state,
        cross_validation,
        hyperparameter_tuning,
        feature_selection
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 600000
      });

      if (result.success) {
        const classificationResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(classificationResult, 'classification_model');
      } else {
        throw new Error(`분류 모델 훈련 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('분류 모델 훈련 실패:', error);
      throw error;
    }
  }

  async trainRegressionModel(data, targetColumn, options = {}) {
    try {
      this.logger.info('회귀 모델 훈련 시작');

      const {
        algorithm = 'random_forest',
        test_size = 0.2,
        random_state = 42,
        cross_validation = true,
        hyperparameter_tuning = false,
        feature_selection = false
      } = options;

      const scriptPath = 'python/ml/supervised/regression.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        target_column: targetColumn,
        algorithm,
        test_size,
        random_state,
        cross_validation,
        hyperparameter_tuning,
        feature_selection
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 600000
      });

      if (result.success) {
        const regressionResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(regressionResult, 'regression_model');
      } else {
        throw new Error(`회귀 모델 훈련 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('회귀 모델 훈련 실패:', error);
      throw error;
    }
  }

  async trainEnsembleModel(data, targetColumn, options = {}) {
    try {
      this.logger.info('앙상블 모델 훈련 시작');

      const {
        ensemble_type = 'voting',
        base_estimators = ['random_forest', 'gradient_boosting', 'svm'],
        test_size = 0.2,
        random_state = 42,
        cross_validation = true
      } = options;

      const scriptPath = 'python/ml/supervised/ensemble.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        target_column: targetColumn,
        ensemble_type,
        base_estimators,
        test_size,
        random_state,
        cross_validation
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 900000 // 15분
      });

      if (result.success) {
        const ensembleResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(ensembleResult, 'ensemble_model');
      } else {
        throw new Error(`앙상블 모델 훈련 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('앙상블 모델 훈련 실패:', error);
      throw error;
    }
  }

  async trainClusteringModel(data, options = {}) {
    try {
      this.logger.info('클러스터링 모델 훈련 시작');

      const {
        algorithm = 'kmeans',
        n_clusters = 'auto',
        features = null,
        preprocessing = true,
        evaluation_metrics = true
      } = options;

      const scriptPath = this.getClusteringScriptPath(algorithm);
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        algorithm,
        n_clusters,
        features,
        preprocessing,
        evaluation_metrics
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 300000 // 5분
      });

      if (result.success) {
        const clusteringResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(clusteringResult, 'clustering_model');
      } else {
        throw new Error(`클러스터링 모델 훈련 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('클러스터링 모델 훈련 실패:', error);
      throw error;
    }
  }

  async trainNeuralNetwork(data, targetColumn, options = {}) {
    try {
      this.logger.info('신경망 모델 훈련 시작');

      const {
        network_type = 'feedforward',
        hidden_layers = [128, 64, 32],
        activation = 'relu',
        optimizer = 'adam',
        epochs = 100,
        batch_size = 32,
        validation_split = 0.2,
        early_stopping = true
      } = options;

      const scriptPath = 'python/ml/deep_learning/neural_network.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        target_column: targetColumn,
        network_type,
        hidden_layers,
        activation,
        optimizer,
        epochs,
        batch_size,
        validation_split,
        early_stopping
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 1800000 // 30분
      });

      if (result.success) {
        const neuralNetResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(neuralNetResult, 'neural_network_model');
      } else {
        throw new Error(`신경망 모델 훈련 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('신경망 모델 훈련 실패:', error);
      throw error;
    }
  }

  async trainCNN(data, targetColumn, options = {}) {
    try {
      this.logger.info('CNN 모델 훈련 시작');

      const {
        input_shape = null,
        num_classes = null,
        conv_layers = [32, 64, 128],
        kernel_size = 3,
        pool_size = 2,
        dropout_rate = 0.2,
        epochs = 50,
        batch_size = 32
      } = options;

      const scriptPath = 'python/ml/deep_learning/cnn.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        target_column: targetColumn,
        input_shape,
        num_classes,
        conv_layers,
        kernel_size,
        pool_size,
        dropout_rate,
        epochs,
        batch_size
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 3600000 // 1시간
      });

      if (result.success) {
        const cnnResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(cnnResult, 'cnn_model');
      } else {
        throw new Error(`CNN 모델 훈련 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('CNN 모델 훈련 실패:', error);
      throw error;
    }
  }

  async trainRNN(data, targetColumn, options = {}) {
    try {
      this.logger.info('RNN 모델 훈련 시작');

      const {
        rnn_type = 'lstm',
        sequence_length = 10,
        hidden_units = 128,
        num_layers = 2,
        dropout_rate = 0.2,
        epochs = 100,
        batch_size = 32
      } = options;

      const scriptPath = 'python/ml/deep_learning/rnn.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        target_column: targetColumn,
        rnn_type,
        sequence_length,
        hidden_units,
        num_layers,
        dropout_rate,
        epochs,
        batch_size
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 3600000 // 1시간
      });

      if (result.success) {
        const rnnResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(rnnResult, 'rnn_model');
      } else {
        throw new Error(`RNN 모델 훈련 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('RNN 모델 훈련 실패:', error);
      throw error;
    }
  }

  validateModelConfig(modelConfig) {
    const { model_type, model_category, target_column } = modelConfig;

    if (!model_category || !this.supportedModels[model_category]) {
      throw new Error(`지원하지 않는 모델 카테고리: ${model_category}`);
    }

    if (!model_type || !this.supportedModels[model_category].includes(model_type)) {
      throw new Error(`지원하지 않는 모델 타입: ${model_type} (카테고리: ${model_category})`);
    }

    if (model_category === 'supervised' && !target_column) {
      throw new Error('지도학습 모델은 타겟 컬럼이 필요합니다.');
    }
  }

  getTrainingScriptPath(category, type) {
    const scriptMap = {
      supervised: {
        classification: 'python/ml/supervised/classification.py',
        regression: 'python/ml/supervised/regression.py',
        ensemble: 'python/ml/supervised/ensemble.py'
      },
      unsupervised: {
        kmeans: 'python/ml/unsupervised/kmeans.py',
        hierarchical: 'python/ml/unsupervised/hierarchical.py',
        dbscan: 'python/ml/unsupervised/dbscan.py'
      },
      deep_learning: {
        neural_network: 'python/ml/deep_learning/neural_network.py',
        cnn: 'python/ml/deep_learning/cnn.py',
        rnn: 'python/ml/deep_learning/rnn.py'
      }
    };

    return scriptMap[category]?.[type] || null;
  }

  getClusteringScriptPath(algorithm) {
    const algorithmMap = {
      kmeans: 'python/ml/unsupervised/kmeans.py',
      hierarchical: 'python/ml/unsupervised/hierarchical.py',
      dbscan: 'python/ml/unsupervised/dbscan.py'
    };

    return algorithmMap[algorithm] || 'python/ml/unsupervised/kmeans.py';
  }

  recordTrainingHistory(modelConfig, result) {
    const record = {
      timestamp: new Date().toISOString(),
      model_config: modelConfig,
      success: !result.error,
      training_time: result.training_time || null,
      model_id: result.model_id || null,
      performance: result.performance || null
    };

    this.trainingHistory.push(record);
    
    // 히스토리 크기 제한 (최대 100개)
    if (this.trainingHistory.length > 100) {
      this.trainingHistory = this.trainingHistory.slice(-50);
    }
  }

  // 유틸리티 메서드들
  getTrainingHistory(limit = 10) {
    return this.trainingHistory.slice(-limit);
  }

  getSupportedModels() {
    return this.supportedModels;
  }

  getSupportedAlgorithms(category, type) {
    const algorithms = {
      supervised: {
        classification: ['random_forest', 'gradient_boosting', 'svm', 'logistic_regression', 'naive_bayes', 'knn'],
        regression: ['random_forest', 'gradient_boosting', 'linear_regression', 'ridge', 'lasso', 'svr'],
        ensemble: ['voting', 'bagging', 'boosting', 'stacking']
      },
      unsupervised: {
        kmeans: ['kmeans', 'kmeans++', 'mini_batch_kmeans'],
        hierarchical: ['ward', 'complete', 'average', 'single'],
        dbscan: ['dbscan', 'optics', 'hdbscan']
      },
      deep_learning: {
        neural_network: ['feedforward', 'autoencoder', 'deep_neural_network'],
        cnn: ['lenet', 'alexnet', 'vgg', 'resnet', 'custom'],
        rnn: ['vanilla_rnn', 'lstm', 'gru', 'bidirectional_lstm']
      }
    };

    return algorithms[category]?.[type] || [];
  }

  async getTrainingStatus() {
    return {
      python_executor_status: await this.pythonExecutor.getExecutionStats(),
      training_history_count: this.trainingHistory.length,
      supported_models: this.supportedModels,
      last_training: this.trainingHistory.length > 0 ? 
        this.trainingHistory[this.trainingHistory.length - 1] : null
    };
  }

  async cleanup() {
    await this.pythonExecutor.shutdown();
    this.logger.info('MLTrainer 정리 완료');
  }
}