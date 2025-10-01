#!/usr/bin/env node
/**
 * Model Interpretability MCP Server
 * 모델 해석 MCP 서버 - SHAP, 특징 중요도, 설명 가능한 AI
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import { Logger } from 'ml-mcp-shared/utils/logger.js';
import { InterpretabilityService } from './services/interpretability-service.js';

class InterpretabilityMCPServer {
  constructor() {
    this.logger = new Logger('interpretability-mcp');
    this.server = new Server(
      {
        name: 'ml-mcp-interpretability',
        version: '1.0.0',
        description: 'Model Interpretability MCP - 설명 가능한 AI 및 모델 해석'
      },
      {
        capabilities: {
          tools: {}
        }
      }
    );

    this.interpretabilityService = new InterpretabilityService(this.logger);
    this.setupHandlers();
  }

  setupHandlers() {
    // List available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      try {
        const tools = await this.interpretabilityService.getTools();
        this.logger.info(`해석 도구 목록 요청 - ${tools.length}개 도구 반환`);
        return { tools };
      } catch (error) {
        this.logger.error('해석 도구 목록 조회 실패:', error);
        return { tools: [] };
      }
    });

    // Execute tool requests
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        this.logger.info(`해석 도구 실행 요청: ${name}`, { args });

        // Execute the tool through the service
        const result = await this.interpretabilityService.executeTool(name, args);

        this.logger.info(`해석 도구 실행 완료: ${name}`);
        return result;

      } catch (error) {
        this.logger.error(`해석 도구 실행 실패 [${name}]:`, error);

        return {
          content: [{
            type: 'text',
            text: `**모델 해석 작업 실패**\n\n` +
                  `**도구:** ${name}\n` +
                  `**오류:** ${error.message}\n\n` +
                  `모델 해석 도구 실행 중 오류가 발생했습니다. 모델 파일과 데이터를 확인해주세요.\n\n` +
                  `**일반적인 해결 방법:**\n` +
                  `• 훈련된 모델 파일 경로 확인\n` +
                  `• 데이터 파일 형식 확인\n` +
                  `• SHAP 라이브러리 설치: pip install shap\n` +
                  `• 충분한 메모리 확인 (SHAP 계산 시)`
          }],
          isError: true
        };
      }
    });

    // Handle server errors
    this.server.onerror = (error) => {
      this.logger.error('해석 MCP 서버 오류:', error);
    };

    process.on('SIGINT', async () => {
      this.logger.info('해석 서버 종료 신호 수신 중...');
      await this.cleanup();
      process.exit(0);
    });

    process.on('SIGTERM', async () => {
      this.logger.info('해석 서버 종료 신호 수신 중...');
      await this.cleanup();
      process.exit(0);
    });
  }

  async initialize() {
    try {
      this.logger.info('🔍 모델 해석 MCP 서버 초기화 중...');

      await this.interpretabilityService.initialize();

      const transport = new StdioServerTransport();
      await this.server.connect(transport);

      this.logger.info('✅ 모델 해석 MCP 서버 준비 완료');
      this.logger.info('📊 지원 기능: SHAP 설명, 특징 중요도, 부분 의존성');

    } catch (error) {
      this.logger.error('❌ 모델 해석 MCP 서버 초기화 실패:', error);
      process.exit(1);
    }
  }

  async cleanup() {
    try {
      this.logger.info('모델 해석 서버 정리 중...');
      await this.server.close();
      this.logger.info('모델 해석 서버 정리 완료');
    } catch (error) {
      this.logger.error('모델 해석 서버 정리 중 오류:', error);
    }
  }
}

// Start the server
const server = new InterpretabilityMCPServer();
server.initialize().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});