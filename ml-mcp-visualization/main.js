#!/usr/bin/env node
/**
 * Visualization MCP Server
 * 시각화 MCP 서버 - 고급 데이터 시각화 및 차트 생성에 특화
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import { Logger } from 'ml-mcp-shared/utils/logger.js';
import { VisualizationService } from './services/visualization-service.js';

class VisualizationMCPServer {
  constructor() {
    this.logger = new Logger('visualization-mcp');
    this.server = new Server(
      {
        name: 'ml-mcp-visualization',
        version: '1.0.0',
        description: 'Visualization MCP - 고급 데이터 시각화 및 차트 생성 시스템'
      },
      {
        capabilities: {
          tools: {}
        }
      }
    );

    this.visualizationService = new VisualizationService(this.logger);
    this.setupHandlers();
  }

  setupHandlers() {
    // List available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      try {
        const tools = await this.visualizationService.getTools();
        this.logger.info(`시각화 도구 목록 요청 - ${tools.length}개 도구 반환`);
        return { tools };
      } catch (error) {
        this.logger.error('시각화 도구 목록 조회 실패:', error);
        return { tools: [] };
      }
    });

    // Execute tool requests
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        this.logger.info(`시각화 도구 실행 요청: ${name}`, { args });

        // Execute the tool through the visualization service
        const result = await this.visualizationService.executeTool(name, args);

        this.logger.info(`시각화 도구 실행 완료: ${name}`);
        return result;

      } catch (error) {
        this.logger.error(`시각화 도구 실행 실패 [${name}]:`, error);

        return {
          content: [{
            type: 'text',
            text: `**시각화 생성 실패**\n\n` +
                  `**도구:** ${name}\n` +
                  `**오류:** ${error.message}\n\n` +
                  `시각화 도구 실행 중 오류가 발생했습니다. 데이터 파일과 컬럼명을 확인해주세요.\n\n` +
                  `**일반적인 해결 방법:**\n` +
                  `• 데이터 파일 경로와 형식 확인\n` +
                  `• 컬럼명 정확성 확인\n` +
                  `• 출력 디렉토리 쓰기 권한 확인\n` +
                  `• Python 시각화 라이브러리 설치 확인\n` +
                  `• 데이터에 시각화 가능한 값이 있는지 확인`
          }],
          isError: true
        };
      }
    });

    // Handle server errors
    this.server.onerror = (error) => {
      this.logger.error('시각화 MCP 서버 오류:', error);
    };

    process.on('SIGINT', async () => {
      this.logger.info('시각화 서버 종료 신호 수신 중...');
      await this.cleanup();
      process.exit(0);
    });

    process.on('SIGTERM', async () => {
      this.logger.info('시각화 서버 종료 신호 수신 중...');
      await this.cleanup();
      process.exit(0);
    });
  }

  async initialize() {
    try {
      this.logger.info('📊 시각화 MCP 서버 초기화 중...');

      // Initialize the visualization service
      await this.visualizationService.initialize();

      this.logger.info('✅ 시각화 MCP 서버 초기화 완료');
      this.logger.info(`🎨 사용 가능한 시각화 도구: ${(await this.visualizationService.getTools()).length}개`);
      this.logger.info('📈 지원 차트: 분포도, 상관관계, 산점도, 시계열, 범주형, 통계적, 인터랙티브, 대시보드');

    } catch (error) {
      this.logger.error('❌ 시각화 서버 초기화 실패:', error);
      throw error;
    }
  }

  async run() {
    try {
      await this.initialize();

      const transport = new StdioServerTransport();
      await this.server.connect(transport);

      this.logger.info('🔄 시각화 MCP 서버가 실행 중입니다...');
      this.logger.info('🎯 고급 시각화: 정적/인터랙티브 차트, 통계 플롯, 종합 대시보드');
      this.logger.info('🛠️ 지원 라이브러리: Matplotlib, Seaborn, Plotly, Bokeh');

    } catch (error) {
      this.logger.error('시각화 서버 실행 실패:', error);
      process.exit(1);
    }
  }

  async cleanup() {
    try {
      this.logger.info('시각화 서버 정리 작업 수행 중...');

      if (this.visualizationService) {
        // Clear output cache
        this.visualizationService.clearOutputCache();
        await this.visualizationService.cleanup();
      }

      this.logger.info('시각화 서버 정리 작업 완료');
    } catch (error) {
      this.logger.error('시각화 서버 정리 작업 실패:', error);
    }
  }

  // Get server status (for monitoring)
  getStatus() {
    return {
      server: {
        name: 'ml-mcp-visualization',
        version: '1.0.0',
        status: 'running',
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        focus: 'data_visualization'
      },
      service: this.visualizationService.getStatus(),
      capabilities: {
        static_plots: ['histogram', 'boxplot', 'violin', 'density', 'heatmap', 'scatter', 'line', 'bar'],
        statistical_plots: ['regression', 'residual', 'qq', 'joint', 'pair'],
        time_series_plots: ['line', 'area', 'seasonal_decompose', 'rolling_stats', 'autocorrelation'],
        interactive_plots: ['3d_scatter', 'surface', 'interactive_heatmap', 'parallel_coordinates', 'sankey'],
        dashboard_types: ['overview', 'statistical', 'exploratory', 'custom'],
        output_formats: ['png', 'pdf', 'svg', 'html', 'json']
      }
    };
  }
}

// Create and run the server
const server = new VisualizationMCPServer();

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

// Memory monitoring for visualization operations
const logMemoryUsage = () => {
  const usage = process.memoryUsage();
  if (usage.heapUsed > 300 * 1024 * 1024) { // 300MB threshold for visualization
    console.warn(`높은 메모리 사용량 감지: ${Math.round(usage.heapUsed / 1024 / 1024)}MB`);
  }
};

// Log memory usage every 3 minutes (visualization can be memory intensive)
setInterval(logMemoryUsage, 3 * 60 * 1000);

// Cleanup temporary files periodically
const cleanupTempFiles = () => {
  // This would implement cleanup of temporary visualization files
  // Implementation would depend on specific file organization
};

// Cleanup every hour
setInterval(cleanupTempFiles, 60 * 60 * 1000);

// Run the server
server.run().catch((error) => {
  console.error('시각화 서버 시작 실패:', error);
  process.exit(1);
});