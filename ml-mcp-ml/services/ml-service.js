/**
 * Machine Learning Service for ML MCP
 * ML MCP용 머신러닝 서비스 - 고급 ML 모델링과 예측에 특화
 */

import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import { BaseService } from 'ml-mcp-shared/utils/base-service';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class MachineLearningService extends BaseService {
  constructor(logger) {
    super('machine-learning-service', 'ml', '1.0.0');
    this.logger = logger;
    this.capabilities = ['tools'];
    this.modelCache = new Map(); // Cache for trained models
  }

  /**
   * Initialize the ML service
   */
  async initialize() {
    try {
      this.logger.info('🤖 머신러닝 서비스 초기화 중');

      // Test ML environment
      await this.testMLEnvironment();

      await super.initialize();
      this.logger.info('✅ 머신러닝 서비스 초기화 완료');

    } catch (error) {
      this.logger.error('❌ 머신러닝 서비스 초기화 실패:', error);
      throw error;
    }
  }

  /**
   * Test ML environment
   */
  async testMLEnvironment() {
    return new Promise((resolve, reject) => {
      const pythonProcess = spawn('python', ['-c',
        'import sklearn, pandas, numpy, joblib; print("ML environment OK")'
      ]);

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error('머신러닝 라이브러리 (scikit-learn, pandas, numpy, joblib)를 사용할 수 없습니다'));
        }
      });

      pythonProcess.on('error', (error) => {
        reject(new Error(`Python ML 환경 테스트 오류: ${error.message}`));
      });
    });
  }

  /**
   * Get available ML tools
   */
  async getTools() {
    return [
      {
        name: 'train_classifier',
        description: '분류 모델을 훈련합니다 (Random Forest, SVM, Logistic Regression 등)',
        inputSchema: {
          type: 'object',
          properties: {
            data_file: {
              type: 'string',
              description: '훈련 데이터 파일 경로'
            },
            target_column: {
              type: 'string',
              description: '타겟(레이블) 컬럼명'
            },
            model_type: {
              type: 'string',
              description: '모델 유형',
              enum: ['random_forest', 'svm', 'logistic_regression', 'gradient_boosting', 'neural_network'],
              default: 'random_forest'
            },
            test_size: {
              type: 'number',
              description: '테스트 데이터 비율',
              default: 0.2,
              minimum: 0.1,
              maximum: 0.5
            },
            cross_validation: {
              type: 'boolean',
              description: '교차 검증 수행 여부',
              default: true
            },
            save_model: {
              type: 'boolean',
              description: '모델 저장 여부',
              default: true
            }
          },
          required: ['data_file', 'target_column']
        }
      },
      {
        name: 'train_regressor',
        description: '회귀 모델을 훈련합니다 (Linear Regression, Random Forest, SVR 등)',
        inputSchema: {
          type: 'object',
          properties: {
            data_file: {
              type: 'string',
              description: '훈련 데이터 파일 경로'
            },
            target_column: {
              type: 'string',
              description: '타겟(예측 대상) 컬럼명'
            },
            model_type: {
              type: 'string',
              description: '회귀 모델 유형',
              enum: ['linear_regression', 'random_forest', 'svr', 'gradient_boosting', 'neural_network'],
              default: 'random_forest'
            },
            test_size: {
              type: 'number',
              description: '테스트 데이터 비율',
              default: 0.2,
              minimum: 0.1,
              maximum: 0.5
            },
            cross_validation: {
              type: 'boolean',
              description: '교차 검증 수행 여부',
              default: true
            },
            save_model: {
              type: 'boolean',
              description: '모델 저장 여부',
              default: true
            }
          },
          required: ['data_file', 'target_column']
        }
      },
      {
        name: 'hyperparameter_tuning',
        description: '하이퍼파라미터 튜닝을 통해 최적의 모델을 찾습니다',
        inputSchema: {
          type: 'object',
          properties: {
            data_file: {
              type: 'string',
              description: '훈련 데이터 파일 경로'
            },
            target_column: {
              type: 'string',
              description: '타겟 컬럼명'
            },
            model_type: {
              type: 'string',
              description: '튜닝할 모델 유형',
              enum: ['random_forest', 'svm', 'gradient_boosting', 'neural_network'],
              default: 'random_forest'
            },
            task_type: {
              type: 'string',
              description: '작업 유형',
              enum: ['classification', 'regression'],
              default: 'classification'
            },
            search_method: {
              type: 'string',
              description: '탐색 방법',
              enum: ['grid_search', 'random_search', 'bayesian_optimization'],
              default: 'grid_search'
            },
            cv_folds: {
              type: 'number',
              description: '교차 검증 폴드 수',
              default: 5,
              minimum: 3,
              maximum: 10
            }
          },
          required: ['data_file', 'target_column', 'task_type']
        }
      },
      {
        name: 'feature_engineering',
        description: '특성 공학을 통해 새로운 피처를 생성하고 기존 피처를 변환합니다',
        inputSchema: {
          type: 'object',
          properties: {
            data_file: {
              type: 'string',
              description: '입력 데이터 파일 경로'
            },
            operations: {
              type: 'array',
              description: '수행할 특성 공학 작업들',
              items: {
                type: 'string',
                enum: ['scaling', 'normalization', 'encoding', 'polynomial_features', 'feature_selection', 'pca', 'interaction_features']
              },
              default: ['scaling', 'encoding']
            },
            target_column: {
              type: 'string',
              description: '타겟 컬럼명 (특성 선택 시 필요)'
            },
            output_file: {
              type: 'string',
              description: '변환된 데이터 저장 경로',
              default: 'processed_data.csv'
            }
          },
          required: ['data_file']
        }
      },
      {
        name: 'model_evaluation',
        description: '훈련된 모델의 성능을 평가하고 상세한 분석을 제공합니다',
        inputSchema: {
          type: 'object',
          properties: {
            model_file: {
              type: 'string',
              description: '평가할 모델 파일 경로'
            },
            test_data_file: {
              type: 'string',
              description: '테스트 데이터 파일 경로'
            },
            target_column: {
              type: 'string',
              description: '타겟 컬럼명'
            },
            task_type: {
              type: 'string',
              description: '작업 유형',
              enum: ['classification', 'regression'],
              default: 'classification'
            },
            generate_plots: {
              type: 'boolean',
              description: '시각화 플롯 생성 여부',
              default: true
            }
          },
          required: ['model_file', 'test_data_file', 'target_column', 'task_type']
        }
      },
      {
        name: 'make_predictions',
        description: '훈련된 모델을 사용하여 새로운 데이터에 대한 예측을 수행합니다',
        inputSchema: {
          type: 'object',
          properties: {
            model_file: {
              type: 'string',
              description: '예측에 사용할 모델 파일 경로'
            },
            input_data_file: {
              type: 'string',
              description: '예측할 데이터 파일 경로'
            },
            output_file: {
              type: 'string',
              description: '예측 결과 저장 경로',
              default: 'predictions.csv'
            },
            include_probabilities: {
              type: 'boolean',
              description: '분류 확률 포함 여부 (분류 모델인 경우)',
              default: false
            }
          },
          required: ['model_file', 'input_data_file']
        }
      },
      {
        name: 'clustering_analysis',
        description: '비지도 학습을 통한 클러스터링 분석을 수행합니다',
        inputSchema: {
          type: 'object',
          properties: {
            data_file: {
              type: 'string',
              description: '클러스터링할 데이터 파일 경로'
            },
            algorithm: {
              type: 'string',
              description: '클러스터링 알고리즘',
              enum: ['kmeans', 'hierarchical', 'dbscan', 'gaussian_mixture'],
              default: 'kmeans'
            },
            n_clusters: {
              type: 'number',
              description: '클러스터 수 (K-means, Hierarchical인 경우)',
              default: 3,
              minimum: 2,
              maximum: 20
            },
            auto_determine_clusters: {
              type: 'boolean',
              description: '최적 클러스터 수 자동 결정',
              default: true
            },
            include_visualization: {
              type: 'boolean',
              description: '클러스터링 시각화 포함',
              default: true
            }
          },
          required: ['data_file']
        }
      },
      {
        name: 'time_series_forecasting',
        description: '시계열 데이터에 대한 예측 모델을 구축합니다',
        inputSchema: {
          type: 'object',
          properties: {
            data_file: {
              type: 'string',
              description: '시계열 데이터 파일 경로'
            },
            date_column: {
              type: 'string',
              description: '날짜/시간 컬럼명'
            },
            value_column: {
              type: 'string',
              description: '예측할 값의 컬럼명'
            },
            forecast_periods: {
              type: 'number',
              description: '예측할 기간 수',
              default: 30,
              minimum: 1,
              maximum: 365
            },
            model_type: {
              type: 'string',
              description: '시계열 모델 유형',
              enum: ['arima', 'lstm', 'prophet', 'exponential_smoothing'],
              default: 'arima'
            },
            include_seasonality: {
              type: 'boolean',
              description: '계절성 고려 여부',
              default: true
            }
          },
          required: ['data_file', 'date_column', 'value_column']
        }
      }
    ];
  }

  /**
   * Execute ML tool
   */
  async executeTool(toolName, args) {
    if (!this.isInitialized) {
      throw new Error('머신러닝 서비스가 초기화되지 않았습니다');
    }

    this.logger.info(`ML 도구 실행 중: ${toolName}`, args);

    switch (toolName) {
      case 'train_classifier':
        return await this.handleTrainClassifier(args);
      case 'train_regressor':
        return await this.handleTrainRegressor(args);
      case 'hyperparameter_tuning':
        return await this.handleHyperparameterTuning(args);
      case 'feature_engineering':
        return await this.handleFeatureEngineering(args);
      case 'model_evaluation':
        return await this.handleModelEvaluation(args);
      case 'make_predictions':
        return await this.handleMakePredictions(args);
      case 'clustering_analysis':
        return await this.handleClusteringAnalysis(args);
      case 'time_series_forecasting':
        return await this.handleTimeSeriesForecasting(args);
      default:
        throw new Error(`알 수 없는 ML 도구: ${toolName}`);
    }
  }

  /**
   * Handle classifier training
   */
  async handleTrainClassifier(args) {
    const { data_file, target_column, model_type = 'random_forest', test_size = 0.2, cross_validation = true, save_model = true } = args;

    try {
      const result = await this.runMLScript('train_classifier', {
        file_path: data_file,
        target_column,
        model_type,
        test_size,
        cross_validation,
        save_model
      });

      return {
        content: [{
          type: 'text',
          text: `**분류 모델 훈련 완료**\n\n` +
                `**모델 유형:** ${model_type}\n` +
                `**데이터:** ${data_file}\n` +
                `**타겟:** ${target_column}\n\n` +
                `**성능 지표:**\n` +
                `• 정확도: ${result.accuracy || 'N/A'}\n` +
                `• 정밀도: ${result.precision || 'N/A'}\n` +
                `• 재현율: ${result.recall || 'N/A'}\n` +
                `• F1 점수: ${result.f1_score || 'N/A'}\n\n` +
                `${cross_validation ? `**교차 검증 점수:** ${result.cv_scores?.join(', ') || 'N/A'}\n\n` : ''}` +
                `분류 모델 훈련이 완료되었습니다.`
        }]
      };
    } catch (error) {
      throw new Error(`분류 모델 훈련 실패: ${error.message}`);
    }
  }

  /**
   * Handle regressor training
   */
  async handleTrainRegressor(args) {
    const { data_file, target_column, model_type = 'random_forest', test_size = 0.2, cross_validation = true, save_model = true } = args;

    try {
      const result = await this.runMLScript('train_regressor', {
        file_path: data_file,
        target_column,
        model_type,
        test_size,
        cross_validation,
        save_model
      });

      return {
        content: [{
          type: 'text',
          text: `**회귀 모델 훈련 완료**\n\n` +
                `**모델 유형:** ${model_type}\n` +
                `**데이터:** ${data_file}\n` +
                `**타겟:** ${target_column}\n\n` +
                `**성능 지표:**\n` +
                `• R² 점수: ${result.r2_score || 'N/A'}\n` +
                `• MAE: ${result.mae || 'N/A'}\n` +
                `• MSE: ${result.mse || 'N/A'}\n` +
                `• RMSE: ${result.rmse || 'N/A'}\n\n` +
                `${cross_validation ? `**교차 검증 점수:** ${result.cv_scores?.join(', ') || 'N/A'}\n\n` : ''}` +
                `회귀 모델 훈련이 완료되었습니다.`
        }]
      };
    } catch (error) {
      throw new Error(`회귀 모델 훈련 실패: ${error.message}`);
    }
  }

  /**
   * Handle hyperparameter tuning
   */
  async handleHyperparameterTuning(args) {
    const { data_file, target_column, model_type = 'random_forest', task_type = 'classification', search_method = 'grid_search', cv_folds = 5 } = args;

    try {
      const result = await this.runMLScript('hyperparameter_tuning', {
        file_path: data_file,
        target_column,
        model_type,
        task_type,
        search_method,
        cv_folds
      });

      return {
        content: [{
          type: 'text',
          text: `**하이퍼파라미터 튜닝 완료**\n\n` +
                `**모델:** ${model_type}\n` +
                `**탐색 방법:** ${search_method}\n` +
                `**작업 유형:** ${task_type}\n\n` +
                `**최적 파라미터:**\n` +
                `${Object.entries(result.best_params || {}).map(([k, v]) => `• ${k}: ${v}`).join('\n')}\n\n` +
                `**최고 성능:** ${result.best_score || 'N/A'}\n\n` +
                `하이퍼파라미터 튜닝이 완료되었습니다.`
        }]
      };
    } catch (error) {
      throw new Error(`하이퍼파라미터 튜닝 실패: ${error.message}`);
    }
  }

  /**
   * Handle feature engineering
   */
  async handleFeatureEngineering(args) {
    const { data_file, operations = ['scaling', 'encoding'], target_column, output_file = 'processed_data.csv' } = args;

    try {
      const result = await this.runMLScript('feature_engineering', {
        file_path: data_file,
        operations,
        target_column
      });

      return {
        content: [{
          type: 'text',
          text: `**특성 공학 완료**\\n\\n` +
                `**입력 데이터:** ${data_file}\\n` +
                `**수행 작업:** ${operations.join(', ')}\\n` +
                `**출력 파일:** ${output_file}\\n\\n` +
                `**변환 결과:**\\n` +
                `• 원본 특성 수: ${result.original_features || 'N/A'}\\n` +
                `• 변환 후 특성 수: ${result.processed_features || 'N/A'}\\n` +
                `• 데이터 크기: ${result.data_shape || 'N/A'}\\n\\n` +
                `특성 공학이 완료되었습니다.`
        }]
      };
    } catch (error) {
      throw new Error(`특성 공학 실패: ${error.message}`);
    }
  }

  /**
   * Handle model evaluation
   */
  async handleModelEvaluation(args) {
    const { model_file, test_data_file, target_column, task_type = 'classification', generate_plots = true } = args;

    try {
      const result = await this.runMLScript('model_evaluation', {
        model_file,
        test_data_file,
        target_column,
        task_type,
        generate_plots
      });

      const metricsText = task_type === 'classification'
        ? `• 정확도: ${result.accuracy || 'N/A'}\n• 정밀도: ${result.precision || 'N/A'}\n• 재현율: ${result.recall || 'N/A'}\n• F1 점수: ${result.f1_score || 'N/A'}`
        : `• R² 점수: ${result.r2_score || 'N/A'}\n• MAE: ${result.mae || 'N/A'}\n• MSE: ${result.mse || 'N/A'}\n• RMSE: ${result.rmse || 'N/A'}`;

      return {
        content: [{
          type: 'text',
          text: `**모델 평가 완료**\n\n` +
                `**모델 파일:** ${model_file}\n` +
                `**테스트 데이터:** ${test_data_file}\n` +
                `**작업 유형:** ${task_type}\n\n` +
                `**성능 지표:**\n` +
                `${metricsText}\n\n` +
                `${generate_plots ? '시각화 플롯이 생성되었습니다.\n\n' : ''}` +
                `모델 평가가 완료되었습니다.`
        }]
      };
    } catch (error) {
      throw new Error(`모델 평가 실패: ${error.message}`);
    }
  }

  /**
   * Handle predictions
   */
  async handleMakePredictions(args) {
    const { model_file, input_data_file, output_file = 'predictions.csv', include_probabilities = false } = args;

    try {
      const result = await this.runMLScript('make_predictions', {
        model_file,
        input_data_file,
        output_file,
        include_probabilities
      });

      return {
        content: [{
          type: 'text',
          text: `**예측 완료**\n\n` +
                `**모델:** ${model_file}\n` +
                `**입력 데이터:** ${input_data_file}\n` +
                `**출력 파일:** ${output_file}\n\n` +
                `**예측 결과:**\n` +
                `• 예측된 샘플 수: ${result.predictions_count || 'N/A'}\n` +
                `• 고유 예측값 수: ${result.unique_predictions || 'N/A'}\n` +
                `${include_probabilities ? `• 확률 정보 포함: 예\n` : ''}\n` +
                `예측이 완료되었습니다.`
        }]
      };
    } catch (error) {
      throw new Error(`예측 실패: ${error.message}`);
    }
  }

  /**
   * Handle clustering analysis
   */
  async handleClusteringAnalysis(args) {
    const { data_file, algorithm = 'kmeans', n_clusters = 3, auto_determine_clusters = true, include_visualization = true } = args;

    try {
      const result = await this.runMLScript('clustering', {
        file_path: data_file,
        algorithm,
        n_clusters,
        auto_determine: auto_determine_clusters
      });

      return {
        content: [{
          type: 'text',
          text: `**클러스터링 분석 완료**\\n\\n` +
                `**알고리즘:** ${algorithm}\\n` +
                `**데이터:** ${data_file}\\n` +
                `**클러스터 수:** ${result.final_clusters || n_clusters}\\n\\n` +
                `**클러스터링 결과:**\\n` +
                `• 실루엣 점수: ${result.silhouette_score || 'N/A'}\\n` +
                `• 관성(Inertia): ${result.inertia || 'N/A'}\\n` +
                `• 클러스터 크기: ${result.cluster_sizes?.join(', ') || 'N/A'}\\n\\n` +
                `${auto_determine_clusters ? `최적 클러스터 수가 자동으로 결정되었습니다.\\n` : ''}` +
                `${include_visualization ? `클러스터링 시각화가 생성되었습니다.\\n` : ''}\\n` +
                `클러스터링 분석이 완료되었습니다.`
        }]
      };
    } catch (error) {
      throw new Error(`클러스터링 분석 실패: ${error.message}`);
    }
  }

  /**
   * Handle time series forecasting
   */
  async handleTimeSeriesForecasting(args) {
    const { data_file, date_column, value_column, forecast_periods = 30, model_type = 'arima', include_seasonality = true } = args;

    try {
      const result = await this.runMLScript('forecasting', {
        file_path: data_file,
        date_column,
        value_column,
        forecast_periods,
        model_type
      });

      return {
        content: [{
          type: 'text',
          text: `**시계열 예측 완료**\\n\\n` +
                `**모델:** ${model_type}\\n` +
                `**데이터:** ${data_file}\\n` +
                `**예측 기간:** ${forecast_periods}\\n\\n` +
                `**예측 성능:**\\n` +
                `• MAE: ${result.mae || 'N/A'}\\n` +
                `• MAPE: ${result.mape || 'N/A'}\\n` +
                `• RMSE: ${result.rmse || 'N/A'}\\n\\n` +
                `**모델 정보:**\\n` +
                `• 훈련 기간: ${result.train_period || 'N/A'}\\n` +
                `• 계절성: ${include_seasonality ? '포함' : '미포함'}\\n\\n` +
                `시계열 예측이 완료되었습니다.`
        }]
      };
    } catch (error) {
      throw new Error(`시계열 예측 실패: ${error.message}`);
    }
  }

  /**
   * Get script path based on script name
   */
  getScriptPath(scriptName) {
    // ML MCP scripts are in python/ml/ directory
    const mlScripts = [
      'train_classifier', 'train_regressor', 'hyperparameter_tuning',
      'feature_engineering', 'model_evaluation', 'make_predictions',
      'clustering_analysis', 'time_series_forecasting'
    ];

    if (mlScripts.includes(scriptName)) {
      return path.join(__dirname, '..', 'python', 'ml', `${scriptName}.py`);
    }

    // Fallback to analyzers directory for other scripts
    const advancedScripts = ['clustering', 'outlier_detection', 'pca'];
    const timeseriesScripts = ['forecasting', 'seasonality', 'trend_analysis'];

    let subdir = 'basic';
    if (advancedScripts.includes(scriptName)) {
      subdir = 'advanced';
    } else if (timeseriesScripts.includes(scriptName)) {
      subdir = 'timeseries';
    }

    return path.join(__dirname, '..', '..', 'python', 'analyzers', subdir, `${scriptName}.py`);
  }

  /**
   * Run ML Python script
   */
  async runMLScript(scriptName, options = {}) {
    const scriptPath = this.getScriptPath(scriptName);

    return new Promise((resolve, reject) => {
      const process = spawn('python', [scriptPath]);

      let stdout = '';
      let stderr = '';

      process.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      process.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      // Send options via stdin
      process.stdin.write(JSON.stringify(options));
      process.stdin.end();

      process.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdout);
            resolve(result);
          } catch (e) {
            resolve({ output: stdout, raw: true });
          }
        } else {
          reject(new Error(`ML 스크립트 실패 (exit code: ${code})\\n${stderr}`));
        }
      });

      process.on('error', (error) => {
        reject(new Error(`ML 프로세스 오류: ${error.message}`));
      });

      // Set timeout (5 minutes for complex ML operations)
      setTimeout(() => {
        process.kill('SIGKILL');
        reject(new Error('ML 스크립트 실행 시간 초과 (5분)'));
      }, 300000);
    });
  }

  /**
   * Get service status with ML-specific metrics
   */
  getStatus() {
    const baseStatus = super.getStatus();
    return {
      ...baseStatus,
      toolCount: 8,
      focus: 'machine_learning',
      features: ['classification', 'regression', 'clustering', 'time_series', 'hyperparameter_tuning', 'feature_engineering'],
      modelCache: {
        size: this.modelCache.size,
        keys: Array.from(this.modelCache.keys())
      }
    };
  }

  /**
   * Clear model cache
   */
  clearModelCache() {
    this.modelCache.clear();
    this.logger.info('모델 캐시가 정리되었습니다');
  }
}

export default MachineLearningService;