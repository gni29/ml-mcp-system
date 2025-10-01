#!/usr/bin/env node

/**
 * Modular ML MCP Server - Clean architecture with service-based design
 * Uses the new MCP core framework with separate services for different functionalities
 */

import { BaseMCPServer } from './mcp-core/server/base-server.js';
import { Logger } from './utils/logger.js';

// Import services
import { AnalysisService } from './services/analysis-service.js';
import { MLService } from './services/ml-service.js';
import { VisualizationService } from './services/visualization-service.js';

// Import existing processors for compatibility
import { MainProcessor } from './core/main-processor.js';

class ModularMLMCPServer {
  constructor() {
    this.logger = new Logger();
    this.baseMCPServer = null;
    this.services = new Map();
    this.mainProcessor = null; // For backward compatibility
  }

  /**
   * Initialize the modular MCP server
   */
  async initialize() {
    try {
      this.logger.info('🚀 모듈형 ML MCP 서버 초기화 중');

      // Initialize base MCP server
      await this.initializeBaseMCPServer();

      // Initialize services
      await this.initializeServices();

      // Initialize legacy processor for compatibility
      await this.initializeLegacyProcessor();

      // Register services with the base server
      await this.registerServices();

      this.logger.info('✅ 모듈형 ML MCP 서버 초기화 완료');

    } catch (error) {
      this.logger.error('❌ 모듈형 ML MCP 서버 초기화 실패:', error);
      throw error;
    }
  }

  /**
   * Initialize the base MCP server
   */
  async initializeBaseMCPServer() {
    this.logger.info('기본 MCP 서버 초기화 중');

    this.baseMCPServer = new BaseMCPServer({
      name: 'ml-mcp-system-modular',
      version: '2.0.0',
      capabilities: {
        tools: {},
        resources: {},
        prompts: {}
      }
    });

    await this.baseMCPServer.initialize();
    this.logger.info('기본 MCP 서버 초기화 완료');
  }

  /**
   * Initialize all services
   */
  async initializeServices() {
    this.logger.info('서비스 초기화 중');

    // Analysis Service
    const analysisService = new AnalysisService(this.logger);
    await analysisService.initialize();
    this.services.set('analysis', analysisService);

    // ML Service
    const mlService = new MLService(this.logger);
    await mlService.initialize();
    this.services.set('ml', mlService);

    // Visualization Service
    const visualizationService = new VisualizationService(this.logger);
    await visualizationService.initialize();
    this.services.set('visualization', visualizationService);

    this.logger.info(`${this.services.size}개 서비스 초기화 완료`);
  }

  /**
   * Initialize legacy processor for backward compatibility
   */
  async initializeLegacyProcessor() {
    this.logger.info('호환성을 위한 레거시 프로세서 초기화 중');

    this.mainProcessor = new MainProcessor();
    await this.mainProcessor.initialize?.();

    this.logger.info('레거시 프로세서 초기화 완료');
  }

  /**
   * Register all services with the base MCP server
   */
  async registerServices() {
    this.logger.info('MCP 서버에 서비스 등록 중');

    const serviceRegistrations = [];

    for (const [name, service] of this.services) {
      const metadata = {
        type: service.type,
        version: service.version,
        capabilities: service.capabilities,
        description: `${name} service for ML MCP system`,
        toolCount: (await service.getTools()).length
      };

      serviceRegistrations.push({ name, instance: service, metadata });
    }

    // Register legacy processor as a service
    if (this.mainProcessor) {
      serviceRegistrations.push({
        name: 'legacy-processor',
        instance: this.mainProcessor,
        metadata: {
          type: 'legacy',
          version: '1.0.0',
          capabilities: ['tools'],
          description: 'Legacy processor for backward compatibility'
        }
      });
    }

    // Register all services
    const servicesConfig = {};
    for (const { name, instance, metadata } of serviceRegistrations) {
      servicesConfig[name] = { instance, metadata };
    }

    await this.baseMCPServer.registerServices(servicesConfig);

    this.logger.info(`${serviceRegistrations.length}개 서비스 등록 완료`);
  }

  /**
   * Start the modular MCP server
   */
  async start() {
    try {
      this.logger.info('🚀 모듈형 ML MCP 서버 시작 중');

      // Start the base MCP server
      await this.baseMCPServer.start('stdio');

      this.logger.info('✅ 모듈형 ML MCP 서버 시작 완료');
      this.logServerStatus();

    } catch (error) {
      this.logger.error('❌ 모듈형 ML MCP 서버 시작 실패:', error);
      throw error;
    }
  }

  /**
   * Log server status for debugging
   */
  logServerStatus() {
    const status = this.baseMCPServer.getStatus();

    this.logger.info('📊 서버 상태:', {
      name: status.name,
      version: status.version,
      serviceCount: status.services.serviceCount,
      uptime: `${Math.round(status.uptime)}s`,
      memory: `${Math.round(status.memory.heapUsed / 1024 / 1024)}MB`
    });

    // Log service details
    for (const [name, service] of this.services) {
      const serviceStatus = service.getStatus();
      this.logger.info(`🔧 서비스 [${name}]:`, {
        type: serviceStatus.type,
        version: serviceStatus.version,
        healthy: serviceStatus.healthy,
        toolCount: serviceStatus.toolCount
      });
    }
  }

  /**
   * Get comprehensive server status
   */
  getStatus() {
    const baseStatus = this.baseMCPServer.getStatus();
    const serviceStatuses = {};

    for (const [name, service] of this.services) {
      serviceStatuses[name] = service.getStatus();
    }

    return {
      ...baseStatus,
      architecture: 'modular',
      services: serviceStatuses,
      totalTools: Object.values(serviceStatuses).reduce((sum, s) => sum + (s.toolCount || 0), 0)
    };
  }

  /**
   * Health check
   */
  async healthCheck() {
    const baseHealth = await this.baseMCPServer.healthCheck();
    const serviceHealth = {};

    for (const [name, service] of this.services) {
      serviceHealth[name] = service.isHealthy();
    }

    const allServicesHealthy = Object.values(serviceHealth).every(healthy => healthy);

    return {
      ...baseHealth,
      services: serviceHealth,
      allServicesHealthy,
      status: baseHealth.status === 'healthy' && allServicesHealthy ? 'healthy' : 'unhealthy'
    };
  }

  /**
   * Graceful shutdown
   */
  async shutdown(signal = 'SHUTDOWN') {
    try {
      this.logger.info(`🛑 모듈형 ML MCP 서버 종료 중 (${signal})`);

      // Cleanup services
      for (const [name, service] of this.services) {
        try {
          await service.cleanup?.();
          this.logger.info(`✅ 서비스 '${name}' 정리 완료`);
        } catch (error) {
          this.logger.error(`❌ 서비스 '${name}' 정리 실패:`, error);
        }
      }

      // Cleanup legacy processor
      if (this.mainProcessor) {
        await this.mainProcessor.cleanup?.();
      }

      // Shutdown base server
      await this.baseMCPServer.shutdown(signal);

      this.logger.info('✅ 모듈형 ML MCP 서버 종료 완료');

    } catch (error) {
      this.logger.error('❌ 종료 중 오류 발생:', error);
    }
  }
}

/**
 * Main execution function
 */
async function main() {
  const server = new ModularMLMCPServer();

  try {
    // Initialize and start server
    await server.initialize();
    await server.start();

    // Setup graceful shutdown handlers
    const shutdownHandler = (signal) => {
      console.log(`\n${signal} 신호 수신, 안전하게 종료 중...`);
      server.shutdown(signal);
    };

    process.on('SIGINT', () => shutdownHandler('SIGINT'));
    process.on('SIGTERM', () => shutdownHandler('SIGTERM'));

    // Handle uncaught exceptions
    process.on('uncaughtException', (error) => {
      console.error('처리되지 않은 예외:', error);
      server.shutdown('UNCAUGHT_EXCEPTION');
    });

    process.on('unhandledRejection', (reason, promise) => {
      console.error('처리되지 않은 Promise 거부:', promise, '이유:', reason);
      server.shutdown('UNHANDLED_REJECTION');
    });

  } catch (error) {
    console.error('❌ 서버 시작 중 치명적 오류:', error);
    process.exit(1);
  }
}

// Export for testing or programmatic use
export { ModularMLMCPServer };

// Run if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    console.error('❌ 치명적 오류:', error);
    process.exit(1);
  });
}