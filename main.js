#!/usr/bin/env node

// main.js - ML MCP 서버 메인 파일 (Qwen/Llama 지원)
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { MainProcessor } from './core/main-processor.js';
import { Logger } from './utils/logger.js';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class MLMCPServer {
  constructor() {
    this.server = new Server(
      {
        name: 'ml-mcp-system',
        version: '1.0.0'
      },
      {
        capabilities: {
          tools: {},
          resources: {},
          prompts: {}
        }
      }
    );
    
    this.logger = new Logger();
    this.processor = new MainProcessor();
    this.isInitialized = false;
    
    this.setupToolHandlers();
    this.setupResourceHandlers();
    this.setupPromptHandlers();
  }

  async initialize() {
    try {
      this.logger.info('ML MCP 서버 초기화 시작');
      
      // 메인 프로세서 초기화
      await this.processor.initialize();
      
      this.isInitialized = true;
      this.logger.info('ML MCP 서버 초기화 완료');
    } catch (error) {
      this.logger.error('ML MCP 서버 초기화 실패:', error);
      throw error;
    }
  }

  setupToolHandlers() {
    // 도구 목록 반환
    this.server.setRequestHandler('tools/list', async () => {
      return {
        tools: [
          // 동적 분석 도구들
          {
            name: 'python_runner',
            description: 'Python ML 모듈 통합 실행기를 사용하여 분석을 수행합니다. 기본 분석, 고급 분석, 배치 처리를 지원합니다.',
            inputSchema: {
              type: 'object',
              properties: {
                command: {
                  type: 'string',
                  enum: ['basic', 'advanced', 'batch', 'validate', 'list'],
                  description: '실행할 명령'
                },
                data_path: {
                  type: 'string',
                  description: '데이터 파일 또는 디렉토리 경로'
                },
                analysis_type: {
                  type: 'string',
                  description: '고급 분석 유형 (clustering, pca, outlier_detection, feature_engineering)'
                },
                output_dir: {
                  type: 'string',
                  description: '출력 디렉토리',
                  default: 'results'
                }
              },
              required: ['command']
            }
          },

          {
            name: 'dynamic_analysis',
            description: '사용자 요청에 맞는 Python 분석 모듈을 자동으로 찾아서 실행합니다. 키워드나 자연어로 분석을 요청하면 최적의 모듈을 선택하여 실행합니다.',
            inputSchema: {
              type: 'object',
              properties: {
                query: {
                  type: 'string',
                  description: '분석 요청 (예: "상관관계 분석", "클러스터링", "회귀분석", "데이터 시각화")'
                },
                data: {
                  type: 'object',
                  description: '분석할 데이터 (선택사항). 제공하지 않으면 자동으로 데이터 파일을 감지합니다.',
                  default: null
                },
                options: {
                  type: 'object',
                  description: '실행 옵션',
                  properties: {
                    timeout: {
                      type: 'number',
                      description: '실행 제한 시간 (밀리초)',
                      default: 300000
                    },
                    auto_detect_files: {
                      type: 'boolean',
                      description: '데이터 파일 자동 감지 여부',
                      default: true
                    },
                    moduleOptions: {
                      type: 'object',
                      description: '모듈별 특정 옵션'
                    }
                  },
                  default: {}
                }
              },
              required: ['query']
            }
          },

          {
            name: 'search_modules',
            description: '사용 가능한 Python 분석 모듈을 검색합니다. 키워드로 관련 모듈을 찾거나 카테고리별로 필터링할 수 있습니다.',
            inputSchema: {
              type: 'object',
              properties: {
                query: {
                  type: 'string',
                  description: '검색할 키워드 (예: "correlation", "clustering", "시각화")',
                  default: ''
                },
                category: {
                  type: 'string',
                  description: '카테고리 필터',
                  enum: ['analysis', 'ml', 'visualization', 'data', 'utils', 'custom'],
                  default: null
                },
                limit: {
                  type: 'number',
                  description: '결과 개수 제한',
                  default: 10,
                  minimum: 1,
                  maximum: 50
                }
              }
            }
          },

          {
            name: 'refresh_modules',
            description: 'Python 모듈을 다시 스캔하여 새로운 모듈을 발견합니다. 새로운 .py 파일을 추가한 후 이 명령을 실행하세요.',
            inputSchema: {
              type: 'object',
              properties: {},
              additionalProperties: false
            }
          },

          {
            name: 'module_stats',
            description: '모듈 시스템의 통계 및 현황을 조회합니다. 전체 모듈 수, 카테고리별 분포, 실행 통계 등을 확인할 수 있습니다.',
            inputSchema: {
              type: 'object',
              properties: {},
              additionalProperties: false
            }
          },

          {
            name: 'test_module',
            description: '특정 모듈의 실행을 테스트합니다. 모듈이 올바르게 작동하는지 확인할 때 사용합니다.',
            inputSchema: {
              type: 'object',
              properties: {
                moduleId: {
                  type: 'string',
                  description: '테스트할 모듈 ID (예: "analysis.basic.correlation", "ml.supervised.regression")'
                },
                testData: {
                  type: 'object',
                  description: '테스트용 데이터 (선택사항)',
                  default: null
                }
              },
              required: ['moduleId']
            }
          },

          {
            name: 'module_details',
            description: '특정 모듈의 상세 정보를 조회합니다. 모듈의 함수, 의존성, 사용 통계 등을 확인할 수 있습니다.',
            inputSchema: {
              type: 'object',
              properties: {
                moduleId: {
                  type: 'string',
                  description: '조회할 모듈 ID'
                }
              },
              required: ['moduleId']
            }
          },

          {
            name: 'validate_modules',
            description: '모든 모듈의 유효성을 검증합니다. 시스템 전체의 모듈 상태를 확인할 때 사용합니다.',
            inputSchema: {
              type: 'object',
              properties: {},
              additionalProperties: false
            }
          },

          // 기존 도구들
          {
            name: 'analyze_data',
            description: '데이터 파일을 분석하고 기본 통계 정보를 제공합니다.',
            inputSchema: {
              type: 'object',
              properties: {
                query: {
                  type: 'string',
                  description: '분석 요청 내용'
                },
                file_path: {
                  type: 'string',
                  description: '분석할 파일 경로 (선택사항)'
                },
                auto_detect_files: {
                  type: 'boolean',
                  description: '파일 자동 감지 여부',
                  default: true
                }
              },
              required: ['query']
            }
          },

          {
            name: 'visualize_data',
            description: '데이터를 시각화하여 차트나 그래프를 생성합니다.',
            inputSchema: {
              type: 'object',
              properties: {
                query: {
                  type: 'string',
                  description: '시각화 요청 내용'
                },
                chart_type: {
                  type: 'string',
                  description: '차트 유형 (선택사항)',
                  enum: ['bar', 'line', 'scatter', 'histogram', 'heatmap', 'auto']
                },
                auto_detect_files: {
                  type: 'boolean',
                  description: '파일 자동 감지 여부',
                  default: true
                }
              },
              required: ['query']
            }
          },

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
                  description: '모델 유형 (선택사항)',
                  enum: ['regression', 'classification', 'clustering', 'auto']
                },
                auto_detect_files: {
                  type: 'boolean',
                  description: '파일 자동 감지 여부',
                  default: true
                }
              },
              required: ['query']
            }
          },

          {
            name: 'system_status',
            description: '시스템 상태를 확인합니다.',
            inputSchema: {
              type: 'object',
              properties: {},
              additionalProperties: false
            }
          },

          {
            name: 'mode_switch',
            description: '작업 모드를 전환합니다.',
            inputSchema: {
              type: 'object',
              properties: {
                mode: {
                  type: 'string',
                  description: '전환할 모드',
                  enum: ['general', 'ml', 'data_analysis', 'visualization']
                }
              },
              required: ['mode']
            }
          },

          {
            name: 'general_query',
            description: '일반적인 질문이나 요청을 처리합니다.',
            inputSchema: {
              type: 'object',
              properties: {
                query: {
                  type: 'string',
                  description: '질문이나 요청 내용'
                }
              },
              required: ['query']
            }
          },

          // Phase 10: MLOps & Deployment Tools
          {
            name: 'mlops_experiment_track',
            description: 'MLflow를 사용하여 ML 실험을 추적하고 모델을 등록합니다. 파라미터, 메트릭, 아티팩트를 로깅합니다.',
            inputSchema: {
              type: 'object',
              properties: {
                action: {
                  type: 'string',
                  enum: ['start_run', 'log_params', 'log_metrics', 'log_model', 'register_model', 'list_runs', 'compare_runs', 'get_best_run'],
                  description: '실행할 MLflow 작업'
                },
                experiment_name: {
                  type: 'string',
                  description: '실험 이름',
                  default: 'default'
                },
                run_name: {
                  type: 'string',
                  description: '실행 이름'
                },
                params: {
                  type: 'object',
                  description: '로깅할 파라미터'
                },
                metrics: {
                  type: 'object',
                  description: '로깅할 메트릭'
                },
                model_path: {
                  type: 'string',
                  description: '모델 파일 경로'
                },
                model_name: {
                  type: 'string',
                  description: '등록할 모델 이름'
                },
                stage: {
                  type: 'string',
                  enum: ['Staging', 'Production', 'Archived'],
                  description: '모델 스테이지'
                },
                metric: {
                  type: 'string',
                  description: '비교 기준 메트릭'
                },
                run_ids: {
                  type: 'array',
                  items: { type: 'string' },
                  description: '비교할 run ID 목록'
                }
              },
              required: ['action']
            }
          },

          {
            name: 'mlops_model_serve',
            description: '훈련된 모델을 REST API로 서빙합니다. 모델 등록, 예측, 배치 예측을 지원합니다.',
            inputSchema: {
              type: 'object',
              properties: {
                action: {
                  type: 'string',
                  enum: ['register', 'predict', 'batch_predict', 'list_models', 'model_info', 'unregister', 'start_server', 'health_check'],
                  description: '실행할 작업'
                },
                model_name: {
                  type: 'string',
                  description: '모델 이름'
                },
                model_path: {
                  type: 'string',
                  description: '모델 파일 경로'
                },
                model_type: {
                  type: 'string',
                  enum: ['classifier', 'regressor', 'forecaster', 'nlp'],
                  description: '모델 유형'
                },
                features: {
                  type: 'array',
                  description: '예측할 피처 데이터'
                },
                port: {
                  type: 'number',
                  description: '서버 포트',
                  default: 8000
                }
              },
              required: ['action']
            }
          },

          {
            name: 'mlops_model_monitor',
            description: '프로덕션 모델의 성능을 모니터링하고 데이터 드리프트를 감지합니다.',
            inputSchema: {
              type: 'object',
              properties: {
                action: {
                  type: 'string',
                  enum: ['log_prediction', 'check_drift', 'get_metrics', 'generate_report'],
                  description: '실행할 모니터링 작업'
                },
                model_name: {
                  type: 'string',
                  description: '모델 이름'
                },
                reference_data_path: {
                  type: 'string',
                  description: '참조 데이터 경로 (훈련 데이터)'
                },
                current_data_path: {
                  type: 'string',
                  description: '현재 데이터 경로 (프로덕션 데이터)'
                },
                period: {
                  type: 'string',
                  enum: ['1h', '24h', '7d', 'all'],
                  description: '메트릭 조회 기간',
                  default: '24h'
                },
                output_path: {
                  type: 'string',
                  description: '리포트 출력 경로'
                }
              },
              required: ['action', 'model_name']
            }
          },

          {
            name: 'nlp_topic_modeling',
            description: '문서 컬렉션에서 주제를 발견합니다. LDA, NMF, BERTopic을 지원합니다.',
            inputSchema: {
              type: 'object',
              properties: {
                data_path: {
                  type: 'string',
                  description: '문서 데이터 CSV 파일 경로'
                },
                text_column: {
                  type: 'string',
                  description: '텍스트가 포함된 컬럼명',
                  default: 'text'
                },
                method: {
                  type: 'string',
                  enum: ['lda', 'nmf', 'bertopic'],
                  description: '주제 모델링 방법',
                  default: 'lda'
                },
                n_topics: {
                  type: 'number',
                  description: '주제 수',
                  default: 10
                },
                visualize: {
                  type: 'boolean',
                  description: '시각화 생성 여부',
                  default: true
                },
                output_dir: {
                  type: 'string',
                  description: '출력 디렉토리',
                  default: 'results'
                }
              },
              required: ['data_path']
            }
          },

          {
            name: 'nlp_entity_extraction',
            description: '텍스트에서 명명된 개체(인명, 지명, 조직명 등)를 추출합니다.',
            inputSchema: {
              type: 'object',
              properties: {
                data_path: {
                  type: 'string',
                  description: '텍스트 데이터 CSV 파일 경로'
                },
                text_column: {
                  type: 'string',
                  description: '텍스트가 포함된 컬럼명',
                  default: 'text'
                },
                model: {
                  type: 'string',
                  description: 'SpaCy 모델 이름',
                  default: 'en_core_web_sm'
                },
                backend: {
                  type: 'string',
                  enum: ['spacy', 'transformers'],
                  description: 'NER 백엔드',
                  default: 'spacy'
                },
                entity_types: {
                  type: 'array',
                  items: { type: 'string' },
                  description: '추출할 엔티티 유형 (예: PERSON, ORG, GPE)'
                },
                visualize: {
                  type: 'boolean',
                  description: 'HTML 시각화 생성 여부',
                  default: true
                },
                output_dir: {
                  type: 'string',
                  description: '출력 디렉토리',
                  default: 'results'
                }
              },
              required: ['data_path']
            }
          },

          {
            name: 'nlp_document_similarity',
            description: '문서 간 유사도를 계산하고 유사 문서를 찾습니다. 중복 탐지, 시맨틱 검색을 지원합니다.',
            inputSchema: {
              type: 'object',
              properties: {
                action: {
                  type: 'string',
                  enum: ['find_similar', 'find_duplicates', 'semantic_search', 'cluster'],
                  description: '실행할 작업'
                },
                data_path: {
                  type: 'string',
                  description: '문서 데이터 CSV 파일 경로'
                },
                text_column: {
                  type: 'string',
                  description: '텍스트가 포함된 컬럼명',
                  default: 'text'
                },
                method: {
                  type: 'string',
                  enum: ['tfidf', 'bert'],
                  description: '유사도 계산 방법',
                  default: 'tfidf'
                },
                query: {
                  type: 'string',
                  description: '검색 쿼리'
                },
                queries: {
                  type: 'array',
                  items: { type: 'string' },
                  description: '다중 검색 쿼리'
                },
                top_k: {
                  type: 'number',
                  description: '반환할 유사 문서 수',
                  default: 5
                },
                threshold: {
                  type: 'number',
                  description: '중복 판정 임계값 (0-1)',
                  default: 0.85
                },
                n_clusters: {
                  type: 'number',
                  description: '클러스터 수',
                  default: 5
                },
                visualize: {
                  type: 'boolean',
                  description: '시각화 생성 여부',
                  default: true
                },
                output_dir: {
                  type: 'string',
                  description: '출력 디렉토리',
                  default: 'results'
                }
              },
              required: ['action', 'data_path']
            }
          },

          {
            name: 'api_gateway_manage',
            description: 'ML API Gateway를 관리합니다. API 서버 시작, 상태 확인, 모델 관리를 수행합니다.',
            inputSchema: {
              type: 'object',
              properties: {
                action: {
                  type: 'string',
                  enum: ['start_server', 'health_check', 'list_endpoints', 'server_status'],
                  description: '실행할 작업'
                },
                port: {
                  type: 'number',
                  description: 'API 서버 포트',
                  default: 8080
                },
                enable_auth: {
                  type: 'boolean',
                  description: '인증 활성화 여부',
                  default: false
                },
                rate_limit: {
                  type: 'number',
                  description: '분당 요청 제한',
                  default: 100
                }
              },
              required: ['action']
            }
          },

          {
            name: 'notebook_to_pipeline',
            description: 'Jupyter 노트북을 프로덕션 ML 파이프라인으로 변환합니다. 탐색적 코드를 구조화된 파이프라인으로 자동 변환합니다.',
            inputSchema: {
              type: 'object',
              properties: {
                notebook_path: {
                  type: 'string',
                  description: 'Jupyter 노트북 파일 경로 (.ipynb)'
                },
                output_path: {
                  type: 'string',
                  description: '출력 파이프라인 파일 경로 (.py)'
                },
                framework: {
                  type: 'string',
                  enum: ['auto', 'sklearn', 'pytorch', 'tensorflow', 'xgboost'],
                  description: 'ML 프레임워크',
                  default: 'auto'
                },
                include_tests: {
                  type: 'boolean',
                  description: '테스트 파일 생성 여부',
                  default: false
                },
                include_config: {
                  type: 'boolean',
                  description: '설정 파일 생성 여부',
                  default: true
                },
                show_summary: {
                  type: 'boolean',
                  description: '변환 요약 표시 여부',
                  default: true
                }
              },
              required: ['notebook_path', 'output_path']
            }
          }
        ]
      };
    });

    // 도구 실행 핸들러
    this.server.setRequestHandler('tools/call', async (request) => {
      const { name, arguments: args } = request.params;
      
      if (!this.isInitialized) {
        throw new Error('서버가 아직 초기화되지 않았습니다.');
      }

      return await this.callTool(name, args);
    });
  }

  async callTool(name, args) {
    try {
      this.logger.info(`도구 실행: ${name}`, args);

      switch (name) {
        // Python Runner 도구
        case 'python_runner':
          return await this.handlePythonRunner(args);

        // 동적 분석 도구들
        case 'dynamic_analysis':
          return await this.processor.handleDynamicAnalysis(args);

        case 'search_modules':
          return await this.processor.handleModuleSearch(args);

        case 'refresh_modules':
          return await this.processor.handleModuleRefresh(args);

        case 'module_stats':
          return await this.processor.handleModuleStats(args);

        case 'test_module':
          return await this.processor.handleModuleTest(args);

        case 'module_details':
          return await this.processor.handleModuleDetails(args);

        case 'validate_modules':
          return await this.processor.handleModuleValidation(args);

        // 기존 도구들
        case 'analyze_data':
          return await this.processor.handleDataAnalysis(args);

        case 'visualize_data':
          return await this.processor.handleDataVisualization(args);

        case 'train_model':
          return await this.processor.handleModelTraining(args);

        case 'system_status':
          return await this.processor.handleSystemStatus(args);

        case 'mode_switch':
          return await this.processor.handleModeSwitch(args);

        case 'general_query':
          return await this.processor.handleGenericTask(args);

        // Phase 10: MLOps & NLP tools
        case 'mlops_experiment_track':
          return await this.handleMLflowTracking(args);

        case 'mlops_model_serve':
          return await this.handleModelServing(args);

        case 'mlops_model_monitor':
          return await this.handleModelMonitoring(args);

        case 'nlp_topic_modeling':
          return await this.handleTopicModeling(args);

        case 'nlp_entity_extraction':
          return await this.handleEntityExtraction(args);

        case 'nlp_document_similarity':
          return await this.handleDocumentSimilarity(args);

        case 'api_gateway_manage':
          return await this.handleAPIGateway(args);

        case 'notebook_to_pipeline':
          return await this.handleNotebookToPipeline(args);

        default:
          throw new Error(`알 수 없는 도구: ${name}`);
      }
    } catch (error) {
      this.logger.error(`도구 실행 실패 (${name}):`, error);
      return {
        content: [{
          type: 'text',
          text: `❌ **도구 실행 오류**\n\n` +
                `**도구:** ${name}\n` +
                `**오류:** ${error.message}\n\n` +
                `🔍 **해결 방법:**\n` +
                `   • 인자 형식이 올바른지 확인하세요\n` +
                `   • "모듈 통계" 명령으로 시스템 상태를 확인하세요\n` +
                `   • 문제가 지속되면 "모듈 새로고침"을 시도하세요\n\n` +
                `📋 **제공된 인자:**\n` +
                `\`\`\`json\n${JSON.stringify(args, null, 2)}\n\`\`\``
        }]
      };
    }
  }

  setupResourceHandlers() {
    // 리소스 목록 반환
    this.server.setRequestHandler('resources/list', async () => {
      return {
        resources: [
          {
            uri: 'analysis://modules',
            name: '분석 모듈 목록',
            description: '사용 가능한 모든 분석 모듈의 목록',
            mimeType: 'application/json'
          },
          {
            uri: 'analysis://stats',
            name: '시스템 통계',
            description: '모듈 시스템의 현재 통계 정보',
            mimeType: 'application/json'
          },
          {
            uri: 'analysis://history',
            name: '실행 기록',
            description: '최근 분석 실행 기록',
            mimeType: 'application/json'
          }
        ]
      };
    });

    // 리소스 읽기 핸들러
    this.server.setRequestHandler('resources/read', async (request) => {
      const { uri } = request.params;

      switch (uri) {
        case 'analysis://modules':
          return await this.getModulesResource();
        case 'analysis://stats':
          return await this.getStatsResource();
        case 'analysis://history':
          return await this.getHistoryResource();
        default:
          throw new Error(`알 수 없는 리소스: ${uri}`);
      }
    });
  }

  async getModulesResource() {
    try {
      const modules = await this.processor.dynamicLoader.getAvailableModules();
      return {
        contents: [{
          uri: 'analysis://modules',
          mimeType: 'application/json',
          text: JSON.stringify(modules, null, 2)
        }]
      };
    } catch (error) {
      this.logger.error('모듈 리소스 조회 실패:', error);
      throw error;
    }
  }

  // Phase 10: MLOps Tool Handlers
  async handleMLflowTracking(args) {
    const { action, experiment_name = 'default', ...options } = args;
    const modulePath = path.join(__dirname, 'python', 'ml', 'mlops', 'mlflow_tracker.py');

    return await this.executePythonModule(modulePath, action, {
      experiment: experiment_name,
      ...options
    }, 'MLflow 실험 추적');
  }

  async handleModelServing(args) {
    const { action, ...options } = args;
    const modulePath = path.join(__dirname, 'python', 'ml', 'deployment', 'model_server.py');

    return await this.executePythonModule(modulePath, action, options, '모델 서빙');
  }

  async handleModelMonitoring(args) {
    const { action, model_name, ...options } = args;
    const modulePath = path.join(__dirname, 'python', 'ml', 'mlops', 'model_monitor.py');

    return await this.executePythonModule(modulePath, action, {
      model_name,
      ...options
    }, '모델 모니터링');
  }

  async handleTopicModeling(args) {
    const { data_path, text_column = 'text', method = 'lda', n_topics = 10, ...options } = args;
    const modulePath = path.join(__dirname, 'python', 'ml', 'nlp', 'topic_modeling.py');

    const pythonArgs = [
      modulePath,
      '--input', data_path,
      '--column', text_column,
      '--method', method,
      '--n-topics', String(n_topics)
    ];

    if (options.visualize) pythonArgs.push('--visualize');
    if (options.output_dir) pythonArgs.push('--output', options.output_dir);

    return await this.executePythonScript(pythonArgs, '주제 모델링');
  }

  async handleEntityExtraction(args) {
    const { data_path, text_column = 'text', model = 'en_core_web_sm', backend = 'spacy', ...options } = args;
    const modulePath = path.join(__dirname, 'python', 'ml', 'nlp', 'ner_extractor.py');

    const pythonArgs = [
      modulePath,
      '--input', data_path,
      '--column', text_column,
      '--model', model,
      '--backend', backend
    ];

    if (options.entity_types) {
      pythonArgs.push('--entity-types', ...options.entity_types);
    }
    if (options.visualize) pythonArgs.push('--visualize');
    if (options.output_dir) pythonArgs.push('--output', options.output_dir);

    return await this.executePythonScript(pythonArgs, '개체명 인식');
  }

  async handleDocumentSimilarity(args) {
    const { action, data_path, text_column = 'text', method = 'tfidf', ...options } = args;
    const modulePath = path.join(__dirname, 'python', 'ml', 'nlp', 'document_similarity.py');

    const pythonArgs = [
      modulePath,
      '--action', action,
      '--input', data_path,
      '--column', text_column,
      '--method', method
    ];

    if (options.query) pythonArgs.push('--query', options.query);
    if (options.queries) pythonArgs.push('--queries', ...options.queries);
    if (options.top_k) pythonArgs.push('--top-k', String(options.top_k));
    if (options.threshold) pythonArgs.push('--threshold', String(options.threshold));
    if (options.n_clusters) pythonArgs.push('--n-clusters', String(options.n_clusters));
    if (options.visualize) pythonArgs.push('--visualize');
    if (options.output_dir) pythonArgs.push('--output', options.output_dir);

    return await this.executePythonScript(pythonArgs, '문서 유사도');
  }

  async handleAPIGateway(args) {
    const { action, port = 8080, enable_auth = false, rate_limit = 100 } = args;
    const modulePath = path.join(__dirname, 'python', 'ml', 'api', 'gateway.py');

    const pythonArgs = [
      modulePath,
      '--action', action,
      '--port', String(port),
      '--rate-limit', String(rate_limit)
    ];

    if (enable_auth) pythonArgs.push('--auth-enabled');

    return await this.executePythonScript(pythonArgs, 'API Gateway');
  }

  async handleNotebookToPipeline(args) {
    const {
      notebook_path,
      output_path,
      framework = 'auto',
      include_tests = false,
      include_config = true,
      show_summary = true
    } = args;

    const modulePath = path.join(__dirname, 'python', 'ml', 'pipeline', 'notebook_to_pipeline.py');

    const pythonArgs = [
      modulePath,
      '--notebook', notebook_path,
      '--output', output_path,
      '--framework', framework
    ];

    if (include_tests) pythonArgs.push('--include-tests');
    if (include_config) pythonArgs.push('--include-config');
    if (show_summary) pythonArgs.push('--summary');

    return await this.executePythonScript(pythonArgs, 'Notebook to Pipeline 변환');
  }

  // Generic Python execution helper
  async executePythonModule(modulePath, command, options, taskName) {
    const pythonArgs = [modulePath, command];

    // Add options as command line arguments
    for (const [key, value] of Object.entries(options)) {
      if (value !== undefined && value !== null) {
        pythonArgs.push(`--${key.replace(/_/g, '-')}`);
        if (typeof value !== 'boolean') {
          pythonArgs.push(String(value));
        }
      }
    }

    return await this.executePythonScript(pythonArgs, taskName);
  }

  async executePythonScript(pythonArgs, taskName) {
    this.logger.info(`${taskName} 실행 중:`, pythonArgs);

    return new Promise((resolve, reject) => {
      const process = spawn('python', pythonArgs, {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: __dirname
      });

      let stdout = '';
      let stderr = '';

      process.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      process.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      process.on('close', (code) => {
        if (code === 0) {
          let result;
          try {
            result = JSON.parse(stdout);
          } catch (e) {
            result = { output: stdout, raw: true };
          }

          resolve({
            content: [{
              type: 'text',
              text: `✅ **${taskName} 완료**\n\n${JSON.stringify(result, null, 2)}`
            }]
          });
        } else {
          resolve({
            content: [{
              type: 'text',
              text: `❌ **${taskName} 실패**\n\n**오류 코드:** ${code}\n\n**오류 메시지:**\n\`\`\`\n${stderr || stdout}\n\`\`\``
            }],
            isError: true
          });
        }
      });

      process.on('error', (error) => {
        reject(new Error(`${taskName} 프로세스 오류: ${error.message}`));
      });

      setTimeout(() => {
        process.kill('SIGKILL');
        reject(new Error(`${taskName} 실행 타임아웃`));
      }, 300000); // 5 minutes
    });
  }

  async handlePythonRunner(args) {
    try {
      const { command, data_path, analysis_type, output_dir = 'results' } = args;
      
      if (!command) {
        throw new Error('command 파라미터가 필요합니다');
      }

      const runnerPath = path.join(__dirname, 'scripts', 'python_runner.py');
      const pythonArgs = [runnerPath, command];

      if (data_path) {
        pythonArgs.push('--data', data_path);
      }
      if (analysis_type && command === 'advanced') {
        pythonArgs.push('--type', analysis_type);
      }
      if (output_dir) {
        pythonArgs.push('--output', output_dir);
      }

      this.logger.info('Python Runner 실행 중:', pythonArgs);

      return new Promise((resolve, reject) => {
        const process = spawn('python', pythonArgs, {
          stdio: ['pipe', 'pipe', 'pipe'],
          cwd: __dirname
        });

        let stdout = '';
        let stderr = '';

        process.stdout.on('data', (data) => {
          stdout += data.toString();
        });

        process.stderr.on('data', (data) => {
          stderr += data.toString();
        });

        process.on('close', (code) => {
          if (code === 0) {
            let result;
            try {
              // JSON 출력 파싱 시도
              result = JSON.parse(stdout);
            } catch (e) {
              // JSON이 아닌 경우 텍스트로 처리
              result = { output: stdout, raw: true };
            }

            resolve({
              content: [{
                type: 'text',
                text: `Python Runner 실행 완료:\n\n${JSON.stringify(result, null, 2)}`
              }]
            });
          } else {
            reject(new Error(`Python Runner 실행 실패 (exit code: ${code})\nstderr: ${stderr}\nstdout: ${stdout}`));
          }
        });

        process.on('error', (error) => {
          reject(new Error(`Python Runner 프로세스 오류: ${error.message}`));
        });

        // 타임아웃 설정 (5분)
        setTimeout(() => {
          process.kill('SIGKILL');
          reject(new Error('Python Runner 실행 타임아웃'));
        }, 300000);
      });

    } catch (error) {
      this.logger.error('Python Runner 실행 실패:', error);
      return {
        content: [{
          type: 'text',
          text: `Python Runner 실행 중 오류가 발생했습니다: ${error.message}`
        }],
        isError: true
      };
    }
  }

  async getStatsResource() {
    try {
      const stats = this.processor.dynamicLoader.getModuleStats();
      return {
        contents: [{
          uri: 'analysis://stats',
          mimeType: 'application/json',
          text: JSON.stringify(stats, null, 2)
        }]
      };
    } catch (error) {
      this.logger.error('통계 리소스 조회 실패:', error);
      throw error;
    }
  }

  async getHistoryResource() {
    try {
      const history = this.processor.dynamicLoader.getExecutionHistory(20);
      return {
        contents: [{
          uri: 'analysis://history',
          mimeType: 'application/json',
          text: JSON.stringify(history, null, 2)
        }]
      };
    } catch (error) {
      this.logger.error('히스토리 리소스 조회 실패:', error);
      throw error;
    }
  }

  setupPromptHandlers() {
    // 프롬프트 목록 반환
    this.server.setRequestHandler('prompts/list', async () => {
      return {
        prompts: [
          {
            name: 'analysis_guide',
            description: '데이터 분석 가이드 프롬프트',
            arguments: [
              {
                name: 'data_type',
                description: '데이터 유형',
                required: false
              }
            ]
          },
          {
            name: 'module_creation',
            description: '새 모듈 생성 가이드',
            arguments: [
              {
                name: 'analysis_type',
                description: '분석 유형',
                required: true
              }
            ]
          }
        ]
      };
    });

    // 프롬프트 가져오기 핸들러
    this.server.setRequestHandler('prompts/get', async (request) => {
      const { name, arguments: args } = request.params;

      switch (name) {
        case 'analysis_guide':
          return await this.getAnalysisGuidePrompt(args);
        case 'module_creation':
          return await this.getModuleCreationPrompt(args);
        default:
          throw new Error(`알 수 없는 프롬프트: ${name}`);
      }
    });
  }

  async getAnalysisGuidePrompt(args) {
    const dataType = args?.data_type || '일반';
    
    return {
      description: `${dataType} 데이터 분석 가이드`,
      messages: [
        {
          role: 'user',
          content: {
            type: 'text',
            text: `${dataType} 데이터를 분석하는 방법에 대해 단계별로 안내해주세요. 사용 가능한 분석 모듈과 추천 분석 방법을 포함해주세요.`
          }
        }
      ]
    };
  }

  async getModuleCreationPrompt(args) {
    const analysisType = args?.analysis_type || '기본';
    
    return {
      description: `${analysisType} 분석 모듈 생성 가이드`,
      messages: [
        {
          role: 'user',
          content: {
            type: 'text',
            text: `${analysisType} 분석을 위한 새 Python 모듈을 만드는 방법을 알려주세요. 파일 구조, 필수 함수, 그리고 예제 코드를 포함해주세요.`
          }
        }
      ]
    };
  }

  async run() {
    try {
      // 서버 초기화
      await this.initialize();

      // Transport 설정
      const transport = new StdioServerTransport();
      
      // 서버 실행
      await this.server.connect(transport);
      
      this.logger.info('ML MCP 서버가 성공적으로 시작되었습니다');
      
      // 프로세스 종료 핸들러
      process.on('SIGINT', async () => {
        this.logger.info('서버 종료 중...');
        await this.cleanup();
        process.exit(0);
      });

      process.on('SIGTERM', async () => {
        this.logger.info('서버 종료 중...');
        await this.cleanup();
        process.exit(0);
      });

    } catch (error) {
      this.logger.error('서버 실행 실패:', error);
      process.exit(1);
    }
  }

  async cleanup() {
    try {
      this.logger.info('서버 정리 작업 시작');
      
      // 필요한 정리 작업 수행
      if (this.processor) {
        await this.processor.cleanup?.();
      }
      
      this.logger.info('서버 정리 작업 완료');
    } catch (error) {
      this.logger.error('서버 정리 작업 실패:', error);
    }
  }
}

// 메인 실행
async function main() {
  const server = new MLMCPServer();
  await server.run();
}

// 에러 핸들링
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  process.exit(1);
});

// 스크립트가 직접 실행될 때만 메인 함수 호출
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
}

export { MLMCPServer };