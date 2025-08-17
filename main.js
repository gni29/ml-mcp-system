#!/usr/bin/env node

// main.js - ML MCP 서버 메인 파일
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { MainProcessor } from './core/main-processor.js';
import { Logger } from './utils/logger.js';

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