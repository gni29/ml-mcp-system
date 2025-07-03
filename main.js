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

    if (currentMode === 'ml') {
      baseTools.push(
        {
          name: 'analyze_data',
          description: '데이터 분석을 수행합니다',
          inputSchema: {
            type: 'object',
            properties: {
              file_path: {
                type: 'string',
                description: '분석할 데이터 파일 경로'
              },
              analysis_type: {
                type: 'string',
                enum: ['basic', 'advanced', 'full'],
                description: '분석 수준',
                default: 'basic'
              }
            },
            required: ['file_path']
          }
        },
        {
          name: 'train_model',
          description: 'ML 모델을 훈련합니다',
          inputSchema: {
            type: 'object',
            properties: {
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
              }
            },
            required: ['data_path', 'target_column']
          }
        }
      );
    }

    return baseTools;
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
    // 시스템 관련 작업 처리
    return {
      content: [
        {
          type: 'text',
          text: '시스템 작업이 완료되었습니다.'
        }
      ]
    };
  }

  async run() {
    try {
      // 모델 매니저 초기화
      await this.modelManager.initialize();
      
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
