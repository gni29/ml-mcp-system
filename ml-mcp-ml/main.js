#!/usr/bin/env node
/**
 * Machine Learning MCP Server
 * 머신러닝 MCP 서버 - 고급 ML 모델링과 예측에 특화
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import { Logger } from 'ml-mcp-shared/utils/logger.js';
import { MachineLearningService } from './services/ml-service.js';

class MachineLearningMCPServer {
  constructor() {
    this.logger = new Logger('machine-learning-mcp');
    this.server = new Server(
      {
        name: 'ml-mcp-ml',
        version: '1.0.0',
        description: 'Machine Learning MCP - 고급 머신러닝 모델링 및 예측 시스템'
      },
      {
        capabilities: {
          tools: {}
        }
      }
    );

    this.mlService = new MachineLearningService(this.logger);
    this.setupHandlers();
  }

  setupHandlers() {
    // List available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      try {
        const tools = await this.mlService.getTools();
        this.logger.info(`ML 도구 목록 요청 - ${tools.length}개 도구 반환`);
        return { tools };
      } catch (error) {
        this.logger.error('ML 도구 목록 조회 실패:', error);
        return { tools: [] };
      }
    });

    // Execute tool requests
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        this.logger.info(`ML 도구 실행 요청: ${name}`, { args });

        // Execute the tool through the ML service
        const result = await this.mlService.executeTool(name, args);

        this.logger.info(`ML 도구 실행 완료: ${name}`);
        return result;

      } catch (error) {
        this.logger.error(`ML 도구 실행 실패 [${name}]:`, error);

        return {
          content: [{
            type: 'text',
            text: `**머신러닝 작업 실패**\n\n` +
                  `**도구:** ${name}\n` +
                  `**오류:** ${error.message}\n\n` +
                  `머신러닝 도구 실행 중 오류가 발생했습니다. 데이터 형식과 파라미터를 확인해주세요.\n\n` +
                  `**일반적인 해결 방법:**\n` +
                  `• 데이터 파일 경로와 형식 확인\n` +
                  `• 타겟 컬럼명 정확성 확인\n` +
                  `• 필수 Python 라이브러리 설치 확인\n` +
                  `• 데이터에 충분한 샘플 수 확인`
          }],
          isError: true
        };
      }
    });

    // Handle server errors
    this.server.onerror = (error) => {
      this.logger.error('ML MCP 서버 오류:', error);
    };

    process.on('SIGINT', async () => {
      this.logger.info('ML 서버 종료 신호 수신 중...');
      await this.cleanup();
      process.exit(0);
    });

    process.on('SIGTERM', async () => {
      this.logger.info('ML 서버 종료 신호 수신 중...');
      await this.cleanup();
      process.exit(0);
    });
  }

  async initialize() {
    try {
      this.logger.info('🤖 머신러닝 MCP 서버 초기화 중...');

      // Initialize the ML service
      await this.mlService.initialize();

      this.logger.info('✅ 머신러닝 MCP 서버 초기화 완료');
      this.logger.info(`🧠 사용 가능한 ML 도구: ${(await this.mlService.getTools()).length}개`);
      this.logger.info('🎯 지원 모델: 분류, 회귀, 클러스터링, 시계열 예측');

    } catch (error) {
      this.logger.error('❌ ML 서버 초기화 실패:', error);
      throw error;
    }
  }

  async run() {
    try {
      await this.initialize();

      const transport = new StdioServerTransport();
      await this.server.connect(transport);

      this.logger.info('🔄 머신러닝 MCP 서버가 실행 중입니다...');
      this.logger.info('🚀 고급 ML 모델링: 분류, 회귀, 클러스터링, 하이퍼파라미터 튜닝, 특성 공학');
      this.logger.info('📊 지원 알고리즘: RandomForest, SVM, 신경망, Gradient Boosting, ARIMA');

    } catch (error) {
      this.logger.error('ML 서버 실행 실패:', error);
      process.exit(1);
    }
  }

  async cleanup() {
    try {
      this.logger.info('ML 서버 정리 작업 수행 중...');

      if (this.mlService) {
        // Clear model cache
        this.mlService.clearModelCache();
        await this.mlService.cleanup();
      }

      this.logger.info('ML 서버 정리 작업 완료');
    } catch (error) {
      this.logger.error('ML 서버 정리 작업 실패:', error);
    }
  }

  // Get server status (for monitoring)
  getStatus() {
    return {
      server: {
        name: 'ml-mcp-ml',
        version: '1.0.0',
        status: 'running',
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        focus: 'advanced_machine_learning'
      },
      service: this.mlService.getStatus(),
      capabilities: {
        classification: ['random_forest', 'svm', 'logistic_regression', 'gradient_boosting', 'neural_network'],
        regression: ['linear_regression', 'random_forest', 'svr', 'gradient_boosting', 'neural_network'],
        clustering: ['kmeans', 'hierarchical', 'dbscan', 'gaussian_mixture'],
        time_series: ['arima', 'lstm', 'prophet', 'exponential_smoothing'],
        optimization: ['grid_search', 'random_search', 'bayesian_optimization'],
        feature_engineering: ['scaling', 'encoding', 'pca', 'feature_selection', 'polynomial_features']
      }
    };
  }
}

// Create and run the server
const server = new MachineLearningMCPServer();

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

// Performance monitoring
const logMemoryUsage = () => {
  const usage = process.memoryUsage();
  if (usage.heapUsed > 500 * 1024 * 1024) { // 500MB threshold
    console.warn(`높은 메모리 사용량 감지: ${Math.round(usage.heapUsed / 1024 / 1024)}MB`);
  }
};

// Log memory usage every 5 minutes
setInterval(logMemoryUsage, 5 * 60 * 1000);

// Run the server
server.run().catch((error) => {
  console.error('ML 서버 시작 실패:', error);
  process.exit(1);
});