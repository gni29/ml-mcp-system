#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { fileURLToPath } from 'url';
import path from 'path';
import fs from 'fs/promises';
import { Logger } from './utils/logger.js';
import { ModelManager } from './core/model-manager.js';
import { PipelineManager } from './core/pipeline-manager.js';
import { MemoryManager } from './core/memory-manager.js';
import { QueryAnalyzer } from './parsers/query-analyzer.js';
import { IntentParser } from './parsers/intent-parser.js';
import { WorkflowBuilder } from './parsers/workflow-builder.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class MLMCPServer {
  constructor() {
    this.server = new Server(
      {
        name: 'ml-mcp-server',
        version: '1.0.0'
      },
      {
        capabilities: {
          tools: {
            listChanged: true,
            supportsStreaming: false
          },
          resources: {
            subscribe: false,
            listChanged: false
          },
          prompts: {
            listChanged: false
          }
        }
      }
    );

    // 핵심 컴포넌트 초기화
    this.logger = new Logger();
    this.modelManager = new ModelManager();
    this.pipelineManager = new PipelineManager();
    this.memoryManager = new MemoryManager();
    this.queryAnalyzer = new QueryAnalyzer();
    this.intentParser = new IntentParser();
    this.workflowBuilder = new WorkflowBuilder();

    // 세션 관리
    this.activeSessions = new Map();
    this.currentMode = 'general';

    // 도구 등록
    this.setupTools();
  }

  setupTools() {
    // 사용자 쿼리 처리 도구
    this.server.setRequestHandler('tools/call', async (request) => {
      const { name, arguments: args } = request.params;
      
      try {
        this.logger.info(`도구 호출: ${name}`, args);
        
        switch (name) {
          case 'process_user_query':
            return await this.processUserQuery(args);
          case 'analyze_data':
            return await this.analyzeData(args);
          case 'train_model':
            return await this.trainModel(args);
          case 'visualize_data':
            return await this.visualizeData(args);
          case 'get_system_status':
            return await this.getSystemStatus(args);
          case 'change_mode':
            return await this.changeMode(args);
          default:
            throw new Error(`Unknown tool: ${name}`);
        }
      } catch (error) {
        this.logger.error(`도구 실행 실패 [${name}]:`, error);
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify({
                type: 'error',
                message: error.message,
                tool: name
              })
            }
          ],
          isError: true
        };
      }
    });

    // 도구 목록 제공
    this.server.setRequestHandler('tools/list', async () => {
      return {
        tools: this.getAvailableTools()
      };
    });
  }

  getAvailableTools() {
    const baseTools = [
      {
        name: 'process_user_query',
        description: '사용자 쿼리를 분석하고 적절한 작업을 수행합니다',
        inputSchema: {
          type: 'object',
          properties: {
            query: {
              type: 'string',
              description: '사용자의 자연어 요청'
            },
            session_id: {
              type: 'string',
              description: '세션 ID'
            },
            conversation_history: {
              type: 'array',
              description: '대화 기록'
            }
          },
          required: ['query']
        }
      },
      {
        name: 'analyze_data',
        description: '데이터 분석을 수행합니다',
        inputSchema: {
          type: 'object',
          properties: {
            query: {
              type: 'string',
              description: '분석 요청 내용'
            },
            file_path: {
              type: 'string',
              description: '분석할 데이터 파일 경로'
            },
            analysis_type: {
              type: 'string',
              enum: ['basic', 'advanced', 'correlation', 'distribution', 'auto'],
              description: '분석 유형',
              default: 'auto'
            },
            columns: {
              type: 'array',
              items: { type: 'string' },
              description: '분석할 컬럼 목록'
            }
          },
          required: ['query']
        }
      },
      {
        name: 'train_model',
        description: '머신러닝 모델을 훈련합니다',
        inputSchema: {
          type: 'object',
          properties: {
            query: {
              type: 'string',
              description: '모델 훈련 요청'
            },
            file_path: {
              type: 'string',
              description: '훈련 데이터 파일 경로'
            },
            target_column: {
              type: 'string',
              description: '타겟 변수 컬럼명'
            },
            model_type: {
              type: 'string',
              enum: ['classification', 'regression', 'clustering', 'auto'],
              description: '모델 유형',
              default: 'auto'
            }
          },
          required: ['query']
        }
      },
      {
        name: 'visualize_data',
        description: '데이터 시각화를 수행합니다',
        inputSchema: {
          type: 'object',
          properties: {
            query: {
              type: 'string',
              description: '시각화 요청 내용'
            },
            file_path: {
              type: 'string',
              description: '시각화할 데이터 파일 경로'
            },
            chart_type: {
              type: 'string',
              enum: ['auto', 'scatter', 'line', 'bar', 'histogram', 'heatmap', 'boxplot'],
              description: '차트 유형',
              default: 'auto'
            },
            x_column: {
              type: 'string',
              description: 'X축 컬럼명'
            },
            y_column: {
              type: 'string',
              description: 'Y축 컬럼명'
            }
          },
          required: ['query']
        }
      },
      {
        name: 'get_system_status',
        description: '시스템 상태를 확인합니다',
        inputSchema: {
          type: 'object',
          properties: {}
        }
      },
      {
        name: 'change_mode',
        description: '작업 모드를 변경합니다',
        inputSchema: {
          type: 'object',
          properties: {
            mode: {
              type: 'string',
              enum: ['general', 'ml', 'deep_learning', 'nlp', 'computer_vision'],
              description: '변경할 모드'
            }
          },
          required: ['mode']
        }
      }
    ];

    return baseTools;
  }

  async processUserQuery(args) {
    try {
      const { query, session_id, conversation_history = [] } = args;
      
      this.logger.info(`사용자 쿼리 처리: ${query}`);
      
      // 세션 관리
      if (session_id && !this.activeSessions.has(session_id)) {
        this.activeSessions.set(session_id, {
          id: session_id,
          startTime: new Date(),
          messageCount: 0,
          context: {}
        });
      }

      // 1. 쿼리 분석
      const queryAnalysis = await this.queryAnalyzer.analyzeQuery(query);
      
      // 2. 의도 파악
      const intentAnalysis = await this.intentParser.parseIntent(query, {
        mode: this.currentMode,
        history: conversation_history.slice(-5)
      });

      // 3. 파일 자동 감지 (필요한 경우)
      const availableFiles = await this.detectDataFiles();
      
      // 4. 적절한 작업 결정 및 실행
      const result = await this.executeBasedOnIntent(intentAnalysis, queryAnalysis, availableFiles, args);
      
      // 5. 세션 업데이트
      if (session_id) {
        const session = this.activeSessions.get(session_id);
        session.messageCount++;
        session.lastActivity = new Date();
      }

      return result;

    } catch (error) {
      this.logger.error('사용자 쿼리 처리 실패:', error);
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              type: 'error',
              message: `요청 처리 중 오류가 발생했습니다: ${error.message}`,
              suggestion: '다시 시도하거나 다른 방식으로 질문해보세요.'
            })
          }
        ],
        isError: true
      };
    }
  }

  async executeBasedOnIntent(intentAnalysis, queryAnalysis, availableFiles, originalArgs) {
    const { intent, confidence, complexity } = intentAnalysis;
    
    // 의도에 따른 작업 분기
    switch (intent) {
      case 'analyze':
        return await this.handleAnalysisRequest(intentAnalysis, queryAnalysis, availableFiles);
      
      case 'visualize':
        return await this.handleVisualizationRequest(intentAnalysis, queryAnalysis, availableFiles);
      
      case 'train':
        return await this.handleTrainingRequest(intentAnalysis, queryAnalysis, availableFiles);
      
      case 'system':
        return await this.handleSystemRequest(intentAnalysis, originalArgs);
      
      case 'help':
        return await this.handleHelpRequest(intentAnalysis);
      
      case 'general':
      default:
        return await this.handleGeneralRequest(intentAnalysis, queryAnalysis);
    }
  }

  async handleAnalysisRequest(intentAnalysis, queryAnalysis, availableFiles) {
    try {
      // 분석할 파일 결정
      const targetFile = this.selectTargetFile(queryAnalysis, availableFiles);
      
      if (!targetFile) {
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify({
                type: 'error',
                message: '분석할 데이터 파일을 찾을 수 없습니다.',
                suggestion: '파일명을 명시하거나 현재 디렉토리에 데이터 파일(.csv, .xlsx 등)을 추가해주세요.',
                available_files: availableFiles
              })
            }
          ]
        };
      }

      // 워크플로우 생성
      const workflow = await this.workflowBuilder.buildWorkflow(intentAnalysis, {
        ...queryAnalysis,
        target_file: targetFile
      });

      // 파이프라인 실행
      const result = await this.pipelineManager.executeWorkflow(
        workflow, 
        intentAnalysis.session_id || 'default',
        intentAnalysis.original_query
      );

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              type: 'analysis_result',
              summary: result.finalResult?.summary || '분석이 완료되었습니다.',
              file_analyzed: targetFile,
              workflow_name: workflow.workflow.name,
              execution_time: result.executionTime,
              results: result.finalResult,
              files_created: this.extractCreatedFiles(result)
            })
          }
        ]
      };

    } catch (error) {
      this.logger.error('분석 요청 처리 실패:', error);
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              type: 'error',
              message: `분석 중 오류가 발생했습니다: ${error.message}`,
              suggestion: '파일 형식이나 데이터 구조를 확인해주세요.'
            })
          }
        ],
        isError: true
      };
    }
  }

  async handleVisualizationRequest(intentAnalysis, queryAnalysis, availableFiles) {
    try {
      const targetFile = this.selectTargetFile(queryAnalysis, availableFiles);
      
      if (!targetFile) {
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify({
                type: 'error',
                message: '시각화할 데이터 파일을 찾을 수 없습니다.',
                available_files: availableFiles
              })
            }
          ]
        };
      }

      // 시각화 워크플로우 생성
      const workflow = await this.workflowBuilder.buildVisualizationWorkflow(intentAnalysis, {
        ...queryAnalysis,
        target_file: targetFile
      });

      // 실행
      const result = await this.pipelineManager.executeWorkflow(
        workflow,
        intentAnalysis.session_id || 'default',
        intentAnalysis.original_query
      );

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              type: 'visualization_result',
              summary: '시각화가 생성되었습니다.',
              file_analyzed: targetFile,
              charts_created: this.extractVisualizationFiles(result),
              insights: this.extractInsights(result),
              workflow_name: workflow.workflow.name
            })
          }
        ]
      };

    } catch (error) {
      this.logger.error('시각화 요청 처리 실패:', error);
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              type: 'error',
              message: `시각화 중 오류가 발생했습니다: ${error.message}`
            })
          }
        ],
        isError: true
      };
    }
  }

  async handleTrainingRequest(intentAnalysis, queryAnalysis, availableFiles) {
    try {
      const targetFile = this.selectTargetFile(queryAnalysis, availableFiles);
      
      if (!targetFile) {
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify({
                type: 'error',
                message: '훈련할 데이터 파일을 찾을 수 없습니다.',
                available_files: availableFiles
              })
            }
          ]
        };
      }

      // 머신러닝 워크플로우 생성
      const workflow = await this.workflowBuilder.buildMLWorkflow(intentAnalysis, {
        ...queryAnalysis,
        target_file: targetFile
      });

      // 실행
      const result = await this.pipelineManager.executeWorkflow(
        workflow,
        intentAnalysis.session_id || 'default',
        intentAnalysis.original_query
      );

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              type: 'training_result',
              summary: '모델 훈련이 완료되었습니다.',
              file_used: targetFile,
              model_performance: this.extractModelPerformance(result),
              model_saved: this.extractModelPath(result),
              recommendations: this.extractRecommendations(result)
            })
          }
        ]
      };

    } catch (error) {
      this.logger.error('훈련 요청 처리 실패:', error);
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              type: 'error',
              message: `모델 훈련 중 오류가 발생했습니다: ${error.message}`
            })
          }
        ],
        isError: true
      };
    }
  }

  async handleSystemRequest(intentAnalysis, originalArgs) {
    try {
      const systemStatus = await this.getSystemStatus();
      return systemStatus;
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              type: 'error',
              message: `시스템 상태 확인 실패: ${error.message}`
            })
          }
        ],
        isError: true
      };
    }
  }

  async handleHelpRequest(intentAnalysis) {
    const helpContent = `
🤖 ML 분석 도우미 도움말

📊 **데이터 분석:**
• "data.csv 파일을 분석해주세요"
• "기본 통계를 보여주세요"
• "상관관계 분석을 해주세요"

📈 **시각화:**
• "히스토그램을 그려주세요"
• "산점도를 만들어주세요"
• "상관관계 히트맵을 보여주세요"

🤖 **머신러닝:**
• "예측 모델을 만들어주세요"
• "클러스터링을 해주세요"
• "분류 모델을 훈련시켜주세요"

⚙️ **시스템:**
• "상태 확인해주세요"
• "모드를 변경해주세요"

💡 **팁:**
• 현재 디렉토리의 파일들을 자동으로 감지합니다
• 자연어로 편하게 요청하세요
• 구체적인 컬럼명이나 파일명을 지정할 수 있습니다
`;

    return {
      content: [
        {
          type: 'text',
          text: helpContent
        }
      ]
    };
  }

  async handleGeneralRequest(intentAnalysis, queryAnalysis) {
    // 일반적인 대화나 질문 처리
    const response = `안녕하세요! ML 분석 도우미입니다.

다음과 같은 작업을 도와드릴 수 있습니다:
• 데이터 분석 및 통계
• 데이터 시각화
• 머신러닝 모델 훈련
• 시스템 상태 확인

구체적인 작업을 요청해주시면 도와드리겠습니다.
예: "data.csv 파일을 분석해주세요" 또는 "히스토그램을 그려주세요"`;

    return {
      content: [
        {
          type: 'text',
          text: response
        }
      ]
    };
  }

  async analyzeData(args) {
    // 직접 분석 도구 호출을 위한 별도 메서드
    return await this.handleAnalysisRequest(
      { intent: 'analyze', ...args },
      await this.queryAnalyzer.analyzeQuery(args.query || ''),
      await this.detectDataFiles()
    );
  }

  async trainModel(args) {
    // 직접 훈련 도구 호출을 위한 별도 메서드
    return await this.handleTrainingRequest(
      { intent: 'train', ...args },
      await this.queryAnalyzer.analyzeQuery(args.query || ''),
      await this.detectDataFiles()
    );
  }

  async visualizeData(args) {
    // 직접 시각화 도구 호출을 위한 별도 메서드
    return await this.handleVisualizationRequest(
      { intent: 'visualize', ...args },
      await this.queryAnalyzer.analyzeQuery(args.query || ''),
      await this.detectDataFiles()
    );
  }

  async getSystemStatus(args = {}) {
    try {
      // 메모리 상태
      const memoryStatus = await this.memoryManager.getCurrentMemoryUsage();
      
      // 모델 상태
      const modelStatus = this.modelManager.getLoadedModels();
      
      // 세션 정보
      const sessionInfo = {
        active_sessions: this.activeSessions.size,
        current_mode: this.currentMode
      };

      // 사용 가능한 파일들
      const availableFiles = await this.detectDataFiles();

      const statusText = `📊 시스템 상태 보고서

🤖 **모델 상태:**
- 로드된 모델: ${Object.keys(modelStatus).length}개
- 현재 모드: ${this.currentMode}

💾 **메모리 사용량:**
- 총 사용량: ${Math.round(memoryStatus.totalMB)}MB
- 사용률: ${Math.round(memoryStatus.usagePercent)}%

🔗 **세션 정보:**
- 활성 세션: ${sessionInfo.active_sessions}개

📁 **사용 가능한 파일:**
${availableFiles.length > 0 ? 
  availableFiles.map(f => `- ${f}`).join('\n') : 
  '- 감지된 데이터 파일 없음'}

✅ **시스템 상태:** 정상 작동`;

      return {
        content: [
          {
            type: 'text',
            text: statusText
          }
        ]
      };

    } catch (error) {
      this.logger.error('시스템 상태 확인 실패:', error);
      return {
        content: [
          {
            type: 'text',
            text: `❌ 시스템 상태 확인 중 오류가 발생했습니다: ${error.message}`
          }
        ],
        isError: true
      };
    }
  }

  async changeMode(args) {
    try {
      const { mode } = args;
      
      if (!mode) {
        return {
          content: [
            {
              type: 'text',
              text: '변경할 모드를 지정해주세요. (general, ml, deep_learning, nlp, computer_vision)'
            }
          ]
        };
      }

      const validModes = ['general', 'ml', 'deep_learning', 'nlp', 'computer_vision'];
      if (!validModes.includes(mode)) {
        return {
          content: [
            {
              type: 'text',
              text: `유효하지 않은 모드입니다. 사용 가능한 모드: ${validModes.join(', ')}`
            }
          ]
        };
      }

      this.currentMode = mode;
      
      return {
        content: [
          {
            type: 'text',
            text: `🔄 작업 모드가 '${mode}'로 변경되었습니다.`
          }
        ]
      };

    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `모드 변경 실패: ${error.message}`
          }
        ],
        isError: true
      };
    }
  }

  async detectDataFiles() {
    try {
      const files = await fs.readdir('./');
      const dataFiles = files.filter(file =>
        file.endsWith('.csv') ||
        file.endsWith('.xlsx') ||
        file.endsWith('.json') ||
        file.endsWith('.txt') ||
        file.endsWith('.parquet')
      );
      
      return dataFiles;
    } catch (error) {
      this.logger.error('파일 감지 실패:', error);
      return [];
    }
  }

  selectTargetFile(queryAnalysis, availableFiles) {
    // 명시적으로 지정된 파일이 있는지 확인
    if (queryAnalysis.resolved_references?.files?.length > 0) {
      const specifiedFile = queryAnalysis.resolved_references.files[0];
      if (availableFiles.includes(specifiedFile.name || specifiedFile)) {
        return specifiedFile.name || specifiedFile;
      }
    }

    // 쿼리에서 파일명 추출 시도
    const filePattern = /([a-zA-Z0-9_-]+\.(csv|xlsx|json|txt|parquet))/gi;
    const matches = queryAnalysis.original_query?.match(filePattern);
    if (matches) {
      const mentionedFile = matches[0];
      if (availableFiles.includes(mentionedFile)) {
        return mentionedFile;
      }
    }

    // 기본적으로 첫 번째 CSV 파일 선택
    const csvFiles = availableFiles.filter(f => f.endsWith('.csv'));
    if (csvFiles.length > 0) {
      return csvFiles[0];
    }

    // CSV가 없으면 다른 데이터 파일 선택
    if (availableFiles.length > 0) {
      return availableFiles[0];
    }

    return null;
  }

  extractCreatedFiles(result) {
    const files = [];
    
    if (result.finalResult?.outputs) {
      Object.values(result.finalResult.outputs).forEach(output => {
        if (output.result?.output_file) {
          files.push(output.result.output_file);
        }
        if (output.result?.chart_path) {
          files.push(output.result.chart_path);
        }
      });
    }

    return files;
  }

  extractVisualizationFiles(result) {
    const charts = [];
    
    if (result.finalResult?.visualizations) {
      result.finalResult.visualizations.forEach(viz => {
        if (viz.filePath) {
          charts.push({
            type: viz.chartType,
            file: viz.filePath,
            description: viz.description
          });
        }
      });
    }

    return charts;
  }

  extractInsights(result) {
    const insights = [];
    
    if (result.finalResult?.recommendations) {
      insights.push(...result.finalResult.recommendations);
    }

    return insights;
  }

  extractModelPerformance(result) {
    if (result.finalResult?.statistics) {
      return result.finalResult.statistics;
    }
    return null;
  }

  extractModelPath(result) {
    const files = this.extractCreatedFiles(result);
    const modelFile = files.find(f => f.includes('model') || f.endsWith('.pkl') || f.endsWith('.joblib'));
    return modelFile || null;
  }

  extractRecommendations(result) {
    return result.finalResult?.recommendations || [];
  }

  async initialize() {
    try {
      this.logger.info('ML MCP 서버 초기화 시작...');
      
      // 필요한 디렉토리 생성
      await this.createDirectories();
      
      // 핵심 컴포넌트 초기화
      await this.modelManager.initialize();
      await this.pipelineManager.initialize();
      await this.memoryManager.initialize();
      
      this.logger.info('✅ 모든 컴포넌트 초기화 완료');
      
    } catch (error) {
      this.logger.error('서버 초기화 실패:', error);
      throw error;
    }
  }

  async createDirectories() {
    const directories = [
      './results',
      './temp', 
      './logs',
      './data',
      './uploads'
    ];

    for (const dir of directories) {
      try {
        await fs.mkdir(dir, { recursive: true });
      } catch (error) {
        if (error.code !== 'EEXIST') {
          throw error;
        }
      }
    }
  }

  async run() {
    try {
      // 초기화
      await this.initialize();
      
      // MCP 서버 시작
      const transport = new StdioServerTransport();
      await this.server.connect(transport);
      
      this.logger.info('🚀 ML MCP 서버가 시작되었습니다.');
      
      // 정리 작업을 위한 시그널 핸들러
      process.on('SIGINT', async () => {
        this.logger.info('서버 종료 신호 수신...');
        await this.cleanup();
        process.exit(0);
      });

      process.on('SIGTERM', async () => {
        this.logger.info('서버 종료 신호 수신...');
        await this.cleanup();
        process.exit(0);
      });
      
    } catch (error) {
      this.logger.error('서버 시작 실패:', error);
      process.exit(1);
    }
  }

  async cleanup() {
    try {
      this.logger.info('서버 정리 작업 시작...');
      
      // 모델 정리
      if (this.modelManager) {
        await this.modelManager.cleanup();
      }
      
      // 메모리 정리
      if (this.memoryManager) {
        await this.memoryManager.cleanup();
      }
      
      // 활성 세션 정리
      this.activeSessions.clear();
      
      this.logger.info('✅ 서버 정리 완료');
      
    } catch (error) {
      this.logger.error('정리 작업 중 오류:', error);
    }
  }
}

// 서버 시작
async function main() {
  const server = new MLMCPServer();
  await server.run();
}

// 에러 핸들링
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection:', reason);
  process.exit(1);
});

process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  process.exit(1);
});

// 실행
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}

export default MLMCPServer;