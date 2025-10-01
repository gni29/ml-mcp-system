#!/usr/bin/env node
/**
 * Lightweight Data Analysis MCP Server
 * 경량 데이터 분석 MCP 서버
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import { Logger } from 'ml-mcp-shared/utils/logger.js';
import { LightweightAnalysisService } from './services/analysis-service.js';

class LightweightAnalysisMCPServer {
  constructor() {
    this.logger = new Logger('lightweight-analysis-mcp');
    this.server = new Server(
      {
        name: 'ml-mcp-analysis',
        version: '1.0.0',
        description: 'Lightweight Data Analysis MCP - 기본 통계 분석 및 데이터 탐색'
      },
      {
        capabilities: {
          tools: {}
        }
      }
    );

    this.analysisService = new LightweightAnalysisService(this.logger);
    this.setupHandlers();
  }

  setupHandlers() {
    // List available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      try {
        const tools = await this.analysisService.getTools();
        this.logger.info(`도구 목록 요청 - ${tools.length}개 도구 반환`);
        return { tools };
      } catch (error) {
        this.logger.error('도구 목록 조회 실패:', error);
        return { tools: [] };
      }
    });

    // Execute tool requests
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        this.logger.info(`도구 실행 요청: ${name}`, { args });

        // Execute the tool through the analysis service
        const result = await this.analysisService.executeTool(name, args);

        this.logger.info(`도구 실행 완료: ${name}`);
        return result;

      } catch (error) {
        this.logger.error(`도구 실행 실패 [${name}]:`, error);

        return {
          content: [{
            type: 'text',
            text: `**분석 실패**\n\n` +
                  `**도구:** ${name}\n` +
                  `**오류:** ${error.message}\n\n` +
                  `분석 도구 실행 중 오류가 발생했습니다. 입력 데이터와 파일 경로를 확인해주세요.`
          }],
          isError: true
        };
      }
    });

    // Handle server errors
    this.server.onerror = (error) => {
      this.logger.error('MCP 서버 오류:', error);
    };

    process.on('SIGINT', async () => {
      this.logger.info('서버 종료 신호 수신 중...');
      await this.cleanup();
      process.exit(0);
    });

    process.on('SIGTERM', async () => {
      this.logger.info('서버 종료 신호 수신 중...');
      await this.cleanup();
      process.exit(0);
    });
  }

  async initialize() {
    try {
      this.logger.info('🚀 경량 데이터 분석 MCP 서버 초기화 중...');

      // Initialize the analysis service
      await this.analysisService.initialize();

      this.logger.info('✅ 경량 데이터 분석 MCP 서버 초기화 완료');
      this.logger.info(`📊 사용 가능한 분석 도구: ${(await this.analysisService.getTools()).length}개`);

    } catch (error) {
      this.logger.error('❌ 서버 초기화 실패:', error);
      throw error;
    }
  }

  async run() {
    try {
      await this.initialize();

      const transport = new StdioServerTransport();
      await this.server.connect(transport);

      this.logger.info('🔄 경량 데이터 분석 MCP 서버가 실행 중입니다...');
      this.logger.info('💡 지원하는 분석: 기본 통계, 상관관계, 결측치 분석, 데이터 품질 평가');

    } catch (error) {
      this.logger.error('서버 실행 실패:', error);
      process.exit(1);
    }
  }

  async cleanup() {
    try {
      this.logger.info('서버 정리 작업 수행 중...');

      if (this.analysisService) {
        await this.analysisService.cleanup();
      }

      this.logger.info('서버 정리 작업 완료');
    } catch (error) {
      this.logger.error('서버 정리 작업 실패:', error);
    }
  }

  // Get server status (for monitoring)
  getStatus() {
    return {
      server: {
        name: 'ml-mcp-analysis',
        version: '1.0.0',
        status: 'running',
        uptime: process.uptime(),
        memory: process.memoryUsage()
      },
      service: this.analysisService.getStatus()
    };
  }
}

// Create and run the server
const server = new LightweightAnalysisMCPServer();

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
  console.error('처리되지 않은 Promise 거부:', reason);
  process.exit(1);
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('처리되지 않은 예외:', error);
  process.exit(1);
});

// Run the server
server.run().catch((error) => {
  console.error('서버 시작 실패:', error);
  process.exit(1);
});