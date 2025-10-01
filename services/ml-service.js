/**
 * ML Service - Handles machine learning operations
 * Provides tools for model training, prediction, and ML workflows
 */

import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class MLService {
  constructor(logger) {
    this.logger = logger;
    this.name = 'ml-service';
    this.type = 'machine_learning';
    this.version = '2.0.0';
    this.capabilities = ['tools'];
    this.isInitialized = false;
    this.models = new Map();
  }

  /**
   * Initialize the ML service
   */
  async initialize() {
    try {
      this.logger.info('ML 서비스 초기화 중');

      // Test ML environment
      await this.testMLEnvironment();

      this.isInitialized = true;
      this.logger.info('ML 서비스 초기화 완료');

    } catch (error) {
      this.logger.error('ML 서비스 초기화 실패:', error);
      throw error;
    }
  }

  /**
   * Test ML environment
   */
  async testMLEnvironment() {
    return new Promise((resolve, reject) => {
      const pythonProcess = spawn('python', ['-c', 'import sklearn, pandas, numpy; print("ML environment OK")']);

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error('ML libraries (sklearn, pandas, numpy) are not available'));
        }
      });

      pythonProcess.on('error', (error) => {
        reject(new Error(`ML environment test error: ${error.message}`));
      });
    });
  }

  /**
   * Get available tools
   */
  async getTools() {
    return [
      {
        name: 'train_model',
        description: '머신러닝 모델을 훈련합니다.',
        inputSchema: {
          type: 'object',
          properties: {
            query: {
              type: 'string',
              description: '모델 훈련 요청 내용'
            },
            model_type: {
              type: 'string',
              description: '모델 유형',
              enum: ['regression', 'classification', 'clustering', 'auto'],
              default: 'auto'
            },
            file_path: {
              type: 'string',
              description: '훈련 데이터 파일 경로'
            },
            target_column: {
              type: 'string',
              description: '타겟 변수 열 이름 (지도학습용)'
            },
            test_size: {
              type: 'number',
              description: '테스트 세트 비율',
              default: 0.2,
              minimum: 0.1,
              maximum: 0.5
            }
          },
          required: ['query']
        }
      },
      {
        name: 'predict',
        description: '훈련된 모델로 예측을 수행합니다.',
        inputSchema: {
          type: 'object',
          properties: {
            model_id: {
              type: 'string',
              description: '사용할 모델 ID'
            },
            data: {
              type: 'object',
              description: '예측할 데이터'
            },
            file_path: {
              type: 'string',
              description: '예측할 데이터 파일 경로'
            }
          },
          required: ['model_id']
        }
      },
      {
        name: 'evaluate_model',
        description: '모델 성능을 평가합니다.',
        inputSchema: {
          type: 'object',
          properties: {
            model_id: {
              type: 'string',
              description: '평가할 모델 ID'
            },
            test_data_path: {
              type: 'string',
              description: '테스트 데이터 파일 경로'
            },
            metrics: {
              type: 'array',
              description: '사용할 평가 지표',
              items: {
                type: 'string',
                enum: ['accuracy', 'precision', 'recall', 'f1', 'rmse', 'mae', 'r2']
              },
              default: ['accuracy']
            }
          },
          required: ['model_id']
        }
      },
      {
        name: 'clustering',
        description: '클러스터링 분석을 수행합니다.',
        inputSchema: {
          type: 'object',
          properties: {
            file_path: {
              type: 'string',
              description: '분석할 데이터 파일 경로'
            },
            algorithm: {
              type: 'string',
              description: '클러스터링 알고리즘',
              enum: ['kmeans', 'dbscan', 'hierarchical'],
              default: 'kmeans'
            },
            n_clusters: {
              type: 'integer',
              description: '클러스터 수 (k-means용)',
              default: 3,
              minimum: 2,
              maximum: 20
            },
            features: {
              type: 'array',
              description: '사용할 특성 열 이름',
              items: {
                type: 'string'
              }
            }
          },
          required: ['file_path']
        }
      },
      {
        name: 'feature_engineering',
        description: '특성 공학을 수행합니다.',
        inputSchema: {
          type: 'object',
          properties: {
            file_path: {
              type: 'string',
              description: '처리할 데이터 파일 경로'
            },
            operations: {
              type: 'array',
              description: '수행할 작업들',
              items: {
                type: 'string',
                enum: ['scaling', 'encoding', 'feature_selection', 'pca', 'polynomial']
              },
              default: ['scaling', 'encoding']
            },
            output_path: {
              type: 'string',
              description: '처리된 데이터 저장 경로'
            }
          },
          required: ['file_path']
        }
      },
      {
        name: 'list_models',
        description: '저장된 모델 목록을 조회합니다.',
        inputSchema: {
          type: 'object',
          properties: {},
          additionalProperties: false
        }
      }
    ];
  }

  /**
   * Execute a tool
   */
  async executeTool(toolName, args) {
    if (!this.isInitialized) {
      throw new Error('ML service not initialized');
    }

    this.logger.info(`ML 도구 실행 중: ${toolName}`, args);

    switch (toolName) {
      case 'train_model':
        return await this.handleTrainModel(args);
      case 'predict':
        return await this.handlePredict(args);
      case 'evaluate_model':
        return await this.handleEvaluateModel(args);
      case 'clustering':
        return await this.handleClustering(args);
      case 'feature_engineering':
        return await this.handleFeatureEngineering(args);
      case 'list_models':
        return await this.handleListModels(args);
      default:
        throw new Error(`Unknown ML tool: ${toolName}`);
    }
  }

  /**
   * Handle model training
   */
  async handleTrainModel(args) {
    const { query, model_type = 'auto', file_path, target_column, test_size = 0.2 } = args;

    try {
      let targetFile = file_path;
      if (!targetFile) {
        targetFile = await this.autoDetectDataFile();
      }

      if (!targetFile) {
        return {
          content: [{
            type: 'text',
            text: `**모델 훈련 요청: ${query}**\n\n` +
                  `훈련 데이터 파일을 지정해주세요. \`file_path\` 파라미터를 사용하거나 ` +
                  `데이터 파일을 프로젝트 디렉토리에 배치하세요.`
          }]
        };
      }

      const result = await this.runMLScript('train', {
        data: targetFile,
        model_type,
        target: target_column,
        test_size
      });

      // Store model information
      if (result.model_id) {
        this.models.set(result.model_id, {
          id: result.model_id,
          type: model_type,
          created: new Date().toISOString(),
          data_path: targetFile,
          target_column,
          performance: result.performance
        });
      }

      return {
        content: [{
          type: 'text',
          text: `**모델 훈련 완료**\n\n` +
                `**요청:** ${query}\n` +
                `**모델 유형:** ${model_type}\n` +
                `**데이터:** ${targetFile}\n` +
                `**타겟 변수:** ${target_column || 'auto-detected'}\n\n` +
                `**결과:**\n${JSON.stringify(result, null, 2)}`
        }]
      };

    } catch (error) {
      throw new Error(`Model training failed: ${error.message}`);
    }
  }

  /**
   * Handle prediction
   */
  async handlePredict(args) {
    const { model_id, data, file_path } = args;

    try {
      if (!this.models.has(model_id)) {
        throw new Error(`Model ${model_id} not found`);
      }

      const result = await this.runMLScript('predict', {
        model_id,
        data: data || file_path
      });

      return {
        content: [{
          type: 'text',
          text: `**예측 완료**\n\n` +
                `**모델 ID:** ${model_id}\n` +
                `**입력 데이터:** ${file_path || 'JSON data'}\n\n` +
                `**예측 결과:**\n${JSON.stringify(result, null, 2)}`
        }]
      };

    } catch (error) {
      throw new Error(`Prediction failed: ${error.message}`);
    }
  }

  /**
   * Handle model evaluation
   */
  async handleEvaluateModel(args) {
    const { model_id, test_data_path, metrics = ['accuracy'] } = args;

    try {
      if (!this.models.has(model_id)) {
        throw new Error(`Model ${model_id} not found`);
      }

      const result = await this.runMLScript('evaluate', {
        model_id,
        test_data: test_data_path,
        metrics
      });

      return {
        content: [{
          type: 'text',
          text: `**모델 평가 완료**\n\n` +
                `**모델 ID:** ${model_id}\n` +
                `**테스트 데이터:** ${test_data_path || 'validation set'}\n` +
                `**평가 지표:** ${metrics.join(', ')}\n\n` +
                `**평가 결과:**\n${JSON.stringify(result, null, 2)}`
        }]
      };

    } catch (error) {
      throw new Error(`Model evaluation failed: ${error.message}`);
    }
  }

  /**
   * Handle clustering
   */
  async handleClustering(args) {
    const { file_path, algorithm = 'kmeans', n_clusters = 3, features } = args;

    try {
      const result = await this.runMLScript('clustering', {
        data: file_path,
        algorithm,
        n_clusters,
        features
      });

      return {
        content: [{
          type: 'text',
          text: `**클러스터링 분석 완료**\n\n` +
                `**알고리즘:** ${algorithm}\n` +
                `**클러스터 수:** ${n_clusters}\n` +
                `**데이터:** ${file_path}\n\n` +
                `**결과:**\n${JSON.stringify(result, null, 2)}`
        }]
      };

    } catch (error) {
      throw new Error(`Clustering failed: ${error.message}`);
    }
  }

  /**
   * Handle feature engineering
   */
  async handleFeatureEngineering(args) {
    const { file_path, operations = ['scaling', 'encoding'], output_path } = args;

    try {
      const result = await this.runMLScript('feature_engineering', {
        data: file_path,
        operations,
        output: output_path
      });

      return {
        content: [{
          type: 'text',
          text: `**특성 공학 완료**\n\n` +
                `**원본 데이터:** ${file_path}\n` +
                `**수행 작업:** ${operations.join(', ')}\n` +
                `**출력 경로:** ${output_path || 'auto-generated'}\n\n` +
                `**결과:**\n${JSON.stringify(result, null, 2)}`
        }]
      };

    } catch (error) {
      throw new Error(`Feature engineering failed: ${error.message}`);
    }
  }

  /**
   * Handle list models
   */
  async handleListModels(args) {
    const modelList = Array.from(this.models.values());

    return {
      content: [{
        type: 'text',
        text: `**저장된 모델 목록**\n\n` +
              `**총 ${modelList.length}개의 모델이 저장되어 있습니다.**\n\n` +
              modelList.map(model =>
                `• **${model.id}** (${model.type})\n` +
                `  생성일: ${model.created}\n` +
                `  데이터: ${model.data_path}\n` +
                `  타겟: ${model.target_column || 'N/A'}`
              ).join('\n\n')
      }]
    };
  }

  /**
   * Run ML Python script
   */
  async runMLScript(operation, options = {}) {
    const scriptPath = path.join(__dirname, '..', 'python', 'ml', 'ml_runner.py');
    const args = [scriptPath, operation];

    // Add options as JSON
    const optionsJson = JSON.stringify(options);

    return new Promise((resolve, reject) => {
      const process = spawn('python', args);

      let stdout = '';
      let stderr = '';

      process.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      process.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      // Send options via stdin
      process.stdin.write(optionsJson);
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
          reject(new Error(`ML script failed (exit code: ${code})\n${stderr}`));
        }
      });

      process.on('error', (error) => {
        reject(new Error(`ML process error: ${error.message}`));
      });

      // Set timeout
      setTimeout(() => {
        process.kill('SIGKILL');
        reject(new Error('ML script execution timeout'));
      }, 600000); // 10 minutes for ML operations
    });
  }

  /**
   * Auto-detect data files in the project
   */
  async autoDetectDataFile() {
    const fs = await import('fs/promises');
    const dataExtensions = ['.csv', '.json', '.xlsx', '.xls', '.parquet'];
    const searchDirs = ['data', 'datasets', '.'];

    for (const dir of searchDirs) {
      try {
        const files = await fs.readdir(dir);
        for (const file of files) {
          const ext = path.extname(file).toLowerCase();
          if (dataExtensions.includes(ext)) {
            return path.join(dir, file);
          }
        }
      } catch (error) {
        // Directory doesn't exist, continue
      }
    }

    return null;
  }

  /**
   * Health check
   */
  isHealthy() {
    return this.isInitialized;
  }

  /**
   * Get service status
   */
  getStatus() {
    return {
      name: this.name,
      type: this.type,
      version: this.version,
      initialized: this.isInitialized,
      healthy: this.isHealthy(),
      toolCount: 6,
      modelCount: this.models.size,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Cleanup
   */
  async cleanup() {
    this.logger.info('ML 서비스 정리 중');
    this.models.clear();
    this.isInitialized = false;
  }
}

export default MLService;