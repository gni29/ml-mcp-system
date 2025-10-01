#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Python 스크립트 실행 헬퍼 함수
function executePythonScript(scriptPath, params) {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [scriptPath], {
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(stdout);
          resolve(result);
        } catch (error) {
          reject(new Error(`JSON 파싱 실패: ${error.message}`));
        }
      } else {
        reject(new Error(`Python 스크립트 실행 실패 (코드: ${code}): ${stderr}`));
      }
    });

    pythonProcess.on('error', (error) => {
      reject(new Error(`프로세스 실행 실패: ${error.message}`));
    });

    // 파라미터 전송
    pythonProcess.stdin.write(JSON.stringify(params));
    pythonProcess.stdin.end();
  });
}

const server = new Server(
  {
    name: 'ml-automl-mcp',
    version: '1.0.0',
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// 도구 목록 정의
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      // AutoML 도구
      {
        name: 'run_automl_pipeline',
        description: 'AutoML 파이프라인 실행 - 자동 머신러닝 모델 훈련 및 최적화',
        inputSchema: {
          type: 'object',
          properties: {
            df: {
              type: 'object',
              description: '입력 데이터프레임 (JSON 형태)'
            },
            target_column: {
              type: 'string',
              description: '타겟 컬럼명'
            },
            task_type: {
              type: 'string',
              enum: ['auto', 'classification', 'regression'],
              default: 'auto',
              description: '작업 유형'
            },
            time_budget: {
              type: 'number',
              default: 300,
              description: '시간 예산 (초)'
            },
            optimization_metric: {
              type: 'string',
              default: 'auto',
              description: '최적화 메트릭'
            },
            ensemble_size: {
              type: 'number',
              default: 5,
              description: '앙상블 크기'
            }
          },
          required: ['df', 'target_column']
        }
      },

      // 신경망 도구
      {
        name: 'train_neural_network',
        description: '신경망 모델 훈련 - 다층 퍼셉트론 분류/회귀 모델',
        inputSchema: {
          type: 'object',
          properties: {
            df: {
              type: 'object',
              description: '입력 데이터프레임 (JSON 형태)'
            },
            target_column: {
              type: 'string',
              description: '타겟 컬럼명'
            },
            task_type: {
              type: 'string',
              enum: ['auto', 'classification', 'regression'],
              default: 'auto',
              description: '작업 유형'
            },
            network_architectures: {
              type: 'array',
              items: { type: 'string', enum: ['simple', 'medium', 'complex'] },
              default: ['simple', 'medium', 'complex'],
              description: '테스트할 네트워크 아키텍처'
            },
            test_size: {
              type: 'number',
              default: 0.2,
              description: '테스트 데이터 비율'
            },
            max_iter: {
              type: 'number',
              default: 1000,
              description: '최대 반복 횟수'
            },
            early_stopping: {
              type: 'boolean',
              default: true,
              description: '조기 종료 사용 여부'
            }
          },
          required: ['df', 'target_column']
        }
      },

      // 시계열 예측 도구
      {
        name: 'advanced_time_series_forecast',
        description: '고급 시계열 예측 - ARIMA, Prophet, LSTM 앙상블',
        inputSchema: {
          type: 'object',
          properties: {
            df: {
              type: 'object',
              description: '입력 데이터프레임 (JSON 형태)'
            },
            date_column: {
              type: 'string',
              description: '날짜 컬럼명'
            },
            value_column: {
              type: 'string',
              description: '예측할 값 컬럼명'
            },
            forecast_periods: {
              type: 'number',
              default: 30,
              description: '예측 기간'
            },
            models: {
              type: 'array',
              items: { type: 'string', enum: ['arima', 'prophet', 'lstm'] },
              default: ['arima', 'prophet', 'lstm'],
              description: '사용할 모델 목록'
            },
            train_ratio: {
              type: 'number',
              default: 0.8,
              description: '훈련 데이터 비율'
            },
            ensemble_method: {
              type: 'string',
              enum: ['average', 'median', 'weighted'],
              default: 'weighted',
              description: '앙상블 방법'
            }
          },
          required: ['df', 'date_column', 'value_column']
        }
      },

      // NLP 감정 분석 도구
      {
        name: 'analyze_sentiment',
        description: '감정 분석 - 텍스트 감정 분류 (VADER, TextBlob)',
        inputSchema: {
          type: 'object',
          properties: {
            texts: {
              type: 'array',
              items: { type: 'string' },
              description: '분석할 텍스트 목록'
            },
            method: {
              type: 'string',
              enum: ['auto', 'vader', 'textblob'],
              default: 'auto',
              description: '분석 방법'
            }
          },
          required: ['texts']
        }
      },

      // NLP 텍스트 분류 도구
      {
        name: 'train_text_classifier',
        description: '텍스트 분류 모델 훈련 - 다양한 알고리즘으로 텍스트 분류',
        inputSchema: {
          type: 'object',
          properties: {
            df: {
              type: 'object',
              description: '입력 데이터프레임 (JSON 형태)'
            },
            text_column: {
              type: 'string',
              description: '텍스트 컬럼명'
            },
            label_column: {
              type: 'string',
              description: '라벨 컬럼명'
            },
            algorithms: {
              type: 'array',
              items: { type: 'string', enum: ['logistic', 'naive_bayes', 'svm', 'random_forest'] },
              default: ['logistic', 'naive_bayes', 'svm', 'random_forest'],
              description: '사용할 알고리즘 목록'
            },
            test_size: {
              type: 'number',
              default: 0.2,
              description: '테스트 데이터 비율'
            },
            max_features: {
              type: 'number',
              default: 5000,
              description: '최대 특성 수'
            }
          },
          required: ['df', 'text_column', 'label_column']
        }
      },

      // NLP 키워드 추출 도구
      {
        name: 'extract_keywords',
        description: '키워드 추출 - TF-IDF 기반 텍스트 키워드 추출',
        inputSchema: {
          type: 'object',
          properties: {
            texts: {
              type: 'array',
              items: { type: 'string' },
              description: '텍스트 목록'
            },
            max_features: {
              type: 'number',
              default: 20,
              description: '최대 키워드 수'
            },
            method: {
              type: 'string',
              enum: ['tfidf', 'count'],
              default: 'tfidf',
              description: '추출 방법'
            },
            ngram_range: {
              type: 'array',
              items: { type: 'number' },
              default: [1, 2],
              description: 'n-gram 범위 [최소, 최대]'
            }
          },
          required: ['texts']
        }
      },

      // 언어 감지 도구
      {
        name: 'detect_language',
        description: '언어 감지 - 텍스트의 언어 자동 감지',
        inputSchema: {
          type: 'object',
          properties: {
            text: {
              type: 'string',
              description: '언어를 감지할 텍스트'
            }
          },
          required: ['text']
        }
      },

      // 하이퍼파라미터 최적화 도구
      {
        name: 'optimize_hyperparameters',
        description: '하이퍼파라미터 최적화 - 베이지안 최적화를 통한 모델 튜닝',
        inputSchema: {
          type: 'object',
          properties: {
            df: {
              type: 'object',
              description: '입력 데이터프레임 (JSON 형태)'
            },
            target_column: {
              type: 'string',
              description: '타겟 컬럼명'
            },
            algorithm: {
              type: 'string',
              enum: ['random_forest', 'xgboost', 'svm', 'neural_network'],
              description: '최적화할 알고리즘'
            },
            optimization_trials: {
              type: 'number',
              default: 100,
              description: '최적화 시도 횟수'
            },
            cv_folds: {
              type: 'number',
              default: 5,
              description: '교차 검증 폴드 수'
            }
          },
          required: ['df', 'target_column', 'algorithm']
        }
      },

      // 모델 해석 도구
      {
        name: 'explain_model',
        description: '모델 해석 - SHAP, 특성 중요도를 통한 모델 설명',
        inputSchema: {
          type: 'object',
          properties: {
            model_path: {
              type: 'string',
              description: '모델 파일 경로'
            },
            df: {
              type: 'object',
              description: '설명할 데이터프레임 (JSON 형태)'
            },
            explanation_type: {
              type: 'string',
              enum: ['feature_importance', 'shap', 'permutation'],
              default: 'feature_importance',
              description: '설명 방법'
            },
            sample_size: {
              type: 'number',
              default: 100,
              description: 'SHAP 분석용 샘플 크기'
            }
          },
          required: ['model_path', 'df']
        }
      },

      // 앙상블 모델 도구
      {
        name: 'create_ensemble_model',
        description: '앙상블 모델 생성 - 여러 모델을 결합한 앙상블',
        inputSchema: {
          type: 'object',
          properties: {
            df: {
              type: 'object',
              description: '입력 데이터프레임 (JSON 형태)'
            },
            target_column: {
              type: 'string',
              description: '타겟 컬럼명'
            },
            base_models: {
              type: 'array',
              items: { type: 'string' },
              default: ['random_forest', 'xgboost', 'logistic_regression'],
              description: '기본 모델 목록'
            },
            ensemble_method: {
              type: 'string',
              enum: ['voting', 'stacking', 'bagging'],
              default: 'voting',
              description: '앙상블 방법'
            },
            cv_folds: {
              type: 'number',
              default: 5,
              description: '교차 검증 폴드 수'
            }
          },
          required: ['df', 'target_column']
        }
      }
    ]
  };
});

// 도구 실행 핸들러
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    let scriptPath;
    let result;

    switch (name) {
      case 'run_automl_pipeline':
        scriptPath = path.join(__dirname, '../../python/automl/automl_pipeline.py');
        result = await executePythonScript(scriptPath, args);
        break;

      case 'train_neural_network':
        scriptPath = path.join(__dirname, '../../python/ml/deep_learning/neural_networks.py');
        result = await executePythonScript(scriptPath, args);
        break;

      case 'advanced_time_series_forecast':
        scriptPath = path.join(__dirname, '../../python/ml/time_series/advanced_forecasting.py');
        result = await executePythonScript(scriptPath, args);
        break;

      case 'analyze_sentiment':
        scriptPath = path.join(__dirname, '../../python/ml/nlp/text_analysis.py');
        result = await executePythonScript(scriptPath, { function: 'analyze_sentiment', ...args });
        break;

      case 'train_text_classifier':
        scriptPath = path.join(__dirname, '../../python/ml/nlp/text_analysis.py');
        result = await executePythonScript(scriptPath, { function: 'train_text_classifier', ...args });
        break;

      case 'extract_keywords':
        scriptPath = path.join(__dirname, '../../python/ml/nlp/text_analysis.py');
        result = await executePythonScript(scriptPath, { function: 'extract_keywords', ...args });
        break;

      case 'detect_language':
        scriptPath = path.join(__dirname, '../../python/ml/nlp/text_analysis.py');
        result = await executePythonScript(scriptPath, { function: 'detect_language', ...args });
        break;

      case 'optimize_hyperparameters':
        scriptPath = path.join(__dirname, '../../python/automl/hyperparameter_optimizer.py');
        result = await executePythonScript(scriptPath, args);
        break;

      case 'explain_model':
        scriptPath = path.join(__dirname, '../../python/automl/model_explainer.py');
        result = await executePythonScript(scriptPath, args);
        break;

      case 'create_ensemble_model':
        scriptPath = path.join(__dirname, '../../python/automl/ensemble_builder.py');
        result = await executePythonScript(scriptPath, args);
        break;

      default:
        throw new Error(`알 수 없는 도구: ${name}`);
    }

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(result, null, 2)
        }
      ]
    };

  } catch (error) {
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            success: false,
            error: error.message,
            tool: name
          }, null, 2)
        }
      ],
      isError: true
    };
  }
});

// 서버 시작
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('ML AutoML MCP 서버가 시작되었습니다');
}

main().catch((error) => {
  console.error('서버 시작 실패:', error);
  process.exit(1);
});