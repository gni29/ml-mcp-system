/**
 * Interpretability Service for Model Explainability MCP
 * 모델 해석 서비스 - SHAP, 특징 중요도, 설명 가능한 AI
 */

import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import { BaseService } from 'ml-mcp-shared/utils/base-service.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class InterpretabilityService extends BaseService {
  constructor(logger) {
    super('interpretability-service', 'interpretability', '1.0.0');
    this.logger = logger;
    this.capabilities = ['tools'];
  }

  /**
   * Initialize the interpretability service
   */
  async initialize() {
    try {
      this.logger.info('🔍 모델 해석 서비스 초기화 중');

      // Test interpretability environment
      await this.testInterpretabilityEnvironment();

      await super.initialize();
      this.logger.info('✅ 모델 해석 서비스 초기화 완료');

    } catch (error) {
      this.logger.error('❌ 모델 해석 서비스 초기화 실패:', error);
      throw error;
    }
  }

  /**
   * Test interpretability environment
   */
  async testInterpretabilityEnvironment() {
    return new Promise((resolve, reject) => {
      const pythonProcess = spawn('python', ['-c',
        'import sklearn, pandas, numpy; print("Interpretability environment OK")'
      ]);

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error('해석 라이브러리를 사용할 수 없습니다'));
        }
      });

      pythonProcess.on('error', (error) => {
        reject(new Error(`Python 환경 테스트 오류: ${error.message}`));
      });
    });
  }

  /**
   * Get available interpretability tools
   */
  async getTools() {
    return [
      {
        name: 'analyze_feature_importance',
        description: '모델의 특징 중요도를 분석합니다 (Tree, Permutation, Coefficient 방식)',
        inputSchema: {
          type: 'object',
          properties: {
            model_file: {
              type: 'string',
              description: '훈련된 모델 파일 경로 (.pkl 또는 .joblib)'
            },
            data_file: {
              type: 'string',
              description: '데이터 파일 경로 (CSV)'
            },
            target_column: {
              type: 'string',
              description: '타겟 컬럼명'
            },
            method: {
              type: 'string',
              description: '중요도 계산 방법',
              enum: ['auto', 'tree', 'permutation', 'coefficient'],
              default: 'auto'
            },
            top_n: {
              type: 'number',
              description: '상위 N개 특징 표시',
              default: 10,
              minimum: 1,
              maximum: 50
            }
          },
          required: ['model_file', 'data_file']
        }
      },
      {
        name: 'explain_with_shap',
        description: 'SHAP를 사용하여 모델 예측을 설명합니다 (전역 및 지역 설명)',
        inputSchema: {
          type: 'object',
          properties: {
            model_file: {
              type: 'string',
              description: '훈련된 모델 파일 경로'
            },
            data_file: {
              type: 'string',
              description: '데이터 파일 경로 (CSV)'
            },
            target_column: {
              type: 'string',
              description: '타겟 컬럼명'
            },
            explain_type: {
              type: 'string',
              description: '설명 유형',
              enum: ['global', 'local', 'both'],
              default: 'both'
            },
            sample_size: {
              type: 'number',
              description: '분석할 샘플 수',
              default: 100,
              minimum: 10,
              maximum: 1000
            }
          },
          required: ['model_file', 'data_file']
        }
      },
      {
        name: 'plot_partial_dependence',
        description: '특징의 부분 의존성 플롯을 생성합니다 (PDP)',
        inputSchema: {
          type: 'object',
          properties: {
            model_file: {
              type: 'string',
              description: '훈련된 모델 파일 경로'
            },
            data_file: {
              type: 'string',
              description: '데이터 파일 경로 (CSV)'
            },
            target_column: {
              type: 'string',
              description: '타겟 컬럼명'
            },
            features: {
              type: 'array',
              description: '분석할 특징 이름 목록',
              items: {
                type: 'string'
              }
            },
            output_path: {
              type: 'string',
              description: '출력 파일 경로',
              default: 'partial_dependence.png'
            }
          },
          required: ['model_file', 'data_file', 'features']
        }
      },
      {
        name: 'detect_feature_interactions',
        description: '특징 간 상호작용을 감지합니다',
        inputSchema: {
          type: 'object',
          properties: {
            model_file: {
              type: 'string',
              description: '훈련된 모델 파일 경로'
            },
            data_file: {
              type: 'string',
              description: '데이터 파일 경로 (CSV)'
            },
            target_column: {
              type: 'string',
              description: '타겟 컬럼명'
            },
            top_n: {
              type: 'number',
              description: '상위 N개 상호작용 표시',
              default: 10
            }
          },
          required: ['model_file', 'data_file']
        }
      }
    ];
  }

  /**
   * Execute interpretability tool
   */
  async executeTool(toolName, args) {
    const pythonScriptsPath = path.join(__dirname, '..', '..', 'python', 'ml', 'interpretability');

    let scriptPath;
    let scriptArgs = [];

    switch (toolName) {
      case 'analyze_feature_importance':
        scriptPath = path.join(pythonScriptsPath, 'feature_importance.py');
        scriptArgs = [
          'analyze',
          '--model', args.model_file,
          '--data', args.data_file,
          '--method', args.method || 'auto',
          '--top-n', String(args.top_n || 10)
        ];
        if (args.target_column) {
          scriptArgs.push('--target', args.target_column);
        }
        break;

      case 'explain_with_shap':
        scriptPath = path.join(pythonScriptsPath, 'shap_explainer.py');
        scriptArgs = [
          'explain',
          '--model', args.model_file,
          '--data', args.data_file,
          '--type', args.explain_type || 'both',
          '--samples', String(args.sample_size || 100)
        ];
        if (args.target_column) {
          scriptArgs.push('--target', args.target_column);
        }
        break;

      case 'plot_partial_dependence':
        scriptPath = path.join(pythonScriptsPath, 'feature_importance.py');
        scriptArgs = [
          'pdp',
          '--model', args.model_file,
          '--data', args.data_file,
          '--features', args.features.join(','),
          '--output', args.output_path || 'partial_dependence.png'
        ];
        if (args.target_column) {
          scriptArgs.push('--target', args.target_column);
        }
        break;

      case 'detect_feature_interactions':
        scriptPath = path.join(pythonScriptsPath, 'feature_importance.py');
        scriptArgs = [
          'interactions',
          '--model', args.model_file,
          '--data', args.data_file,
          '--top-n', String(args.top_n || 10)
        ];
        if (args.target_column) {
          scriptArgs.push('--target', args.target_column);
        }
        break;

      default:
        throw new Error(`알 수 없는 도구: ${toolName}`);
    }

    return await this.runPythonScript(scriptPath, scriptArgs, toolName);
  }

  /**
   * Run Python script
   */
  async runPythonScript(scriptPath, args, toolName) {
    return new Promise((resolve, reject) => {
      this.logger.info(`Python 스크립트 실행: ${scriptPath}`, { args });

      const pythonProcess = spawn('python', [scriptPath, ...args], {
        env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
      });

      let output = '';
      let errorOutput = '';

      pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          this.logger.error(`Python 스크립트 실패 (code ${code}):`, errorOutput);
          reject(new Error(`스크립트 실행 실패: ${errorOutput}`));
          return;
        }

        try {
          const jsonMatch = output.match(/\{[\s\S]*\}/);
          if (!jsonMatch) {
            this.logger.error('JSON 출력을 찾을 수 없음:', output);
            reject(new Error('유효한 JSON 출력을 찾을 수 없습니다'));
            return;
          }

          const result = JSON.parse(jsonMatch[0]);

          // Format result for MCP
          const formatted = this.formatResult(toolName, result);
          resolve(formatted);

        } catch (error) {
          this.logger.error('JSON 파싱 실패:', error);
          reject(new Error(`결과 파싱 실패: ${error.message}`));
        }
      });

      pythonProcess.on('error', (error) => {
        this.logger.error('Python 프로세스 오류:', error);
        reject(new Error(`Python 실행 오류: ${error.message}`));
      });
    });
  }

  /**
   * Format result for MCP response
   */
  formatResult(toolName, result) {
    let text = `**모델 해석 결과: ${toolName}**\n\n`;

    if (result.error) {
      text += `**오류:** ${result.error}\n`;
      return {
        content: [{ type: 'text', text }],
        isError: true
      };
    }

    switch (toolName) {
      case 'analyze_feature_importance':
        text += `**방법:** ${result.method}\n\n`;
        text += `**상위 특징:**\n`;
        if (result.top_features) {
          result.top_features.forEach((feat, idx) => {
            const importance = feat.importance_mean !== undefined ? feat.importance_mean : feat.importance;
            text += `${idx + 1}. **${feat.feature}**: ${importance.toFixed(4)}\n`;
          });
        }
        if (result.summary) {
          text += `\n**요약:**\n`;
          text += `• 가장 중요한 특징: ${result.summary.most_important}\n`;
          text += `• 평균 중요도: ${result.summary.mean_importance?.toFixed(4) || 'N/A'}\n`;
        }
        break;

      case 'explain_with_shap':
        text += `**SHAP 설명**\n\n`;
        if (result.global_importance) {
          text += `**전역 중요도 (상위 5개):**\n`;
          result.global_importance.top_features.slice(0, 5).forEach((feat, idx) => {
            text += `${idx + 1}. **${feat.feature}**: ${feat.mean_abs_shap.toFixed(4)}\n`;
          });
        }
        if (result.instance_explanation) {
          text += `\n**개별 예측 설명 (상위 3개):**\n`;
          result.instance_explanation.feature_contributions.slice(0, 3).forEach((contrib, idx) => {
            text += `${idx + 1}. **${contrib.feature}**: ${contrib.shap_value.toFixed(4)} (${contrib.impact})\n`;
          });
        }
        break;

      case 'plot_partial_dependence':
        text += `부분 의존성 플롯이 생성되었습니다.\n`;
        text += `**특징:** ${result.features?.join(', ')}\n`;
        text += `**출력 파일:** ${result.output_path}\n`;
        break;

      case 'detect_feature_interactions':
        text += `**발견된 상호작용:**\n\n`;
        if (result.interactions) {
          result.interactions.forEach((inter, idx) => {
            text += `${idx + 1}. **${inter.feature_1}** ↔ **${inter.feature_2}**\n`;
            text += `   상관계수: ${inter.correlation.toFixed(3)} (${inter.interaction_strength})\n\n`;
          });
        }
        if (result.summary) {
          text += `**요약:** ${result.summary.total_interactions_found}개 상호작용 발견\n`;
        }
        break;

      default:
        text += JSON.stringify(result, null, 2);
    }

    return {
      content: [
        {
          type: 'text',
          text
        }
      ]
    };
  }
}