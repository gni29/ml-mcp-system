#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';

import { ModelManager } from './core/model-manager.js';
import { Router } from './core/router.js';
import { MainProcessor } from './core/main-processor.js';
import { ContextTracker } from './core/context-tracker.js';
import { Logger } from './utils/logger.js';
import fs from 'fs/promises';
import path from 'path';

class MLMCPServer {
  constructor() {
    this.server = new Server(
      {
        name: 'ml-mcp-high-performance',
        version: '1.0.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.logger = new Logger();
    this.modelManager = new ModelManager();
    this.router = new Router(this.modelManager);
    this.processor = new MainProcessor(this.modelManager);
    this.contextTracker = new ContextTracker();

    this.setupHandlers();
  }

  setupHandlers() {
    // 도구 목록 제공
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: await this.getAvailableTools()
      };
    });

    // 도구 실행 처리
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;
      
      try {
        // 컨텍스트 업데이트
        this.contextTracker.updateContext(name, args);
        
        // 특별한 도구 처리
        if (name === 'general_query') {
          return await this.handleGeneralQuery(args);
        }
        
        // 라우팅 결정
        const routingDecision = await this.router.route(name, args);
        
        // 작업 실행
        const result = await this.executeTask(routingDecision, args);
        
        // 결과 반환
        return result;
        
      } catch (error) {
        this.logger.error('Task execution failed:', error);
        return {
          content: [
            {
              type: 'text',
              text: `오류가 발생했습니다: ${error.message}`
            }
          ],
          isError: true
        };
      }
    });
  }

  async getAvailableTools() {
    const currentMode = this.contextTracker.getCurrentMode();
    const baseTools = [
      {
        name: 'general_query',
        description: '자연어 질문 및 명령을 처리합니다',
        inputSchema: {
          type: 'object',
          properties: {
            query: {
              type: 'string',
              description: '사용자의 질문이나 명령'
            }
          },
          required: ['query']
        }
      },
      {
        name: 'mode_switch',
        description: '작업 모드를 전환합니다 (general/ml/coding)',
        inputSchema: {
          type: 'object',
          properties: {
            mode: {
              type: 'string',
              enum: ['general', 'ml', 'coding'],
              description: '전환할 모드'
            }
          },
          required: ['mode']
        }
      },
      {
        name: 'system_status',
        description: '시스템 및 모델 상태를 확인합니다',
        inputSchema: {
          type: 'object',
          properties: {},
          required: []
        }
      }
    ];

    // 모드에 따른 추가 도구들
    if (currentMode === 'ml') {
      baseTools.push(
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
                enum: ['basic', 'advanced', 'full'],
                description: '분석 수준',
                default: 'basic'
              },
              auto_detect_files: {
                type: 'boolean',
                description: '파일 자동 감지 여부',
                default: false
              }
            },
            required: ['query']
          }
        },
        {
          name: 'train_model',
          description: 'ML 모델을 훈련합니다',
          inputSchema: {
            type: 'object',
            properties: {
              query: {
                type: 'string',
                description: '훈련 요청 내용'
              },
              data_path: {
                type: 'string',
                description: '훈련 데이터 경로'
              },
              target_column: {
                type: 'string',
                description: '타겟 변수 컬럼명'
              },
              model_type: {
                type: 'string',
                enum: ['classification', 'regression', 'auto'],
                description: '모델 유형',
                default: 'auto'
              },
              auto_detect_files: {
                type: 'boolean',
                description: '파일 자동 감지 여부',
                default: false
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
                enum: ['auto', 'scatter', 'line', 'bar', 'histogram', 'heatmap'],
                description: '차트 유형',
                default: 'auto'
              },
              auto_detect_files: {
                type: 'boolean',
                description: '파일 자동 감지 여부',
                default: false
              }
            },
            required: ['query']
          }
        }
      );
    }

    return baseTools;
  }

  async handleGeneralQuery(args) {
    const { query } = args;
    
    // AI 모델을 사용해 사용자 의도 분석
    const analysisPrompt = `사용자가 "${query}"라고 말했습니다.

이것이 다음 중 어떤 유형의 요청인지 분석해주세요:
1. 데이터 분석 요청
2. 모델 훈련 요청  
3. 시각화 요청
4. 시스템 상태 확인
5. 모드 변경 요청
6. 일반 대화

분석 결과를 다음 형식으로 응답해주세요:
{
  "intent": "요청 유형",
  "confidence": 0.0-1.0,
  "suggested_action": "권장 행동",
  "requires_files": true/false,
  "response": "사용자에게 보여줄 응답"
}`;

    try {
      const response = await this.modelManager.queryModel('router', analysisPrompt, {
        temperature: 0.1,
        max_tokens: 500
      });

      // JSON 응답 파싱
      let analysis;
      try {
        const jsonMatch = response.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          analysis = JSON.parse(jsonMatch[0]);
        }
      } catch (parseError) {
        this.logger.warn('의도 분석 응답 파싱 실패:', parseError);
      }

      // 분석 결과에 따른 처리
      if (analysis) {
        // 파일이 필요한 경우 자동 감지
        if (analysis.requires_files) {
          const detectedFiles = await this.detectDataFiles();
          if (detectedFiles.length > 0) {
            analysis.response += `\n\n📁 감지된 데이터 파일:\n${detectedFiles.map(f => `- ${f}`).join('\n')}`;
          }
        }

        return {
          content: [
            {
              type: 'text',
              text: analysis.response
            }
          ],
          metadata: {
            intent: analysis.intent,
            confidence: analysis.confidence,
            suggested_action: analysis.suggested_action
          }
        };
      }

      // 파싱 실패 시 기본 응답
      return {
        content: [
          {
            type: 'text',
            text: response
          }
        ]
      };

    } catch (error) {
      this.logger.error('일반 쿼리 처리 실패:', error);
      return {
        content: [
          {
            type: 'text',
            text: '죄송합니다. 요청을 처리하는 중에 오류가 발생했습니다. 다시 시도해주세요.'
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
        file.endsWith('.txt')
      );
      
      return dataFiles;
    } catch (error) {
      this.logger.error('파일 감지 실패:', error);
      return [];
    }
  }

  async executeTask(routingDecision, args) {
    const { taskType, model, tools } = routingDecision;

    switch (taskType) {
      case 'simple':
        return await this.router.handleSimpleTask(args);
      
      case 'complex':
        return await this.processor.handleComplexTask(args, tools);
      
      case 'system':
        return await this.handleSystemTask(args);
      
      default:
        throw new Error(`Unknown task type: ${taskType}`);
    }
  }

  async handleSystemTask(args) {
    const { query, mode } = args;
    
    // 시스템 상태 확인
    if (query && (query.includes('상태') || query.includes('status'))) {
      return await this.getSystemStatus();
    }
    
    // 모드 변경
    if (mode) {
      await this.contextTracker.setMode(mode);
      return {
        content: [
          {
            type: 'text',
            text: `🔄 모드가 '${mode}'로 변경되었습니다.`
          }
        ]
      };
    }
    
    return {
      content: [
        {
          type: 'text',
          text: '시스템 작업이 완료되었습니다.'
        }
      ]
    };
  }

  async getSystemStatus() {
    try {
      const modelStatus = await this.modelManager.getModelStatus();
      const contextStats = this.contextTracker.getUsageStats();
      
      const statusText = `📊 시스템 상태:

🤖 모델 상태: ${Object.keys(modelStatus).length}개 모델 실행 중
🔄 현재 모드: ${contextStats.currentMode}
📈 처리된 작업: ${contextStats.totalEntries}개
💾 활성 세션: ${contextStats.activeSessions}개

모델 세부 정보:
${Object.entries(modelStatus).map(([type, info]) => 
  `- ${type}: ${info.name} (마지막 사용: ${new Date(info.lastUsed).toLocaleString()})`
).join('\n')}`;

      return {
        content: [
          {
            type: 'text',
            text: statusText
          }
        ]
      };
    } catch (error) {
      this.logger.error('시스템 상태 조회 실패:', error);
      return {
        content: [
          {
            type: 'text',
            text: '시스템 상태를 확인하는 중 오류가 발생했습니다.'
          }
        ],
        isError: true
      };
    }
  }

  async run() {
    try {
      // 모델 매니저 초기화
      await this.modelManager.initialize();
      
      // 컨텍스트 트래커 초기화
      await this.contextTracker.initialize();
      
      // MCP 서버 시작
      const transport = new StdioServerTransport();
      await this.server.connect(transport);
      
      this.logger.info('ML MCP 서버가 시작되었습니다.');
      
    } catch (error) {
      this.logger.error('서버 시작 실패:', error);
      process.exit(1);
    }
  }
}

// 서버 시작
const server = new MLMCPServer();
server.run().catch(console.error);
