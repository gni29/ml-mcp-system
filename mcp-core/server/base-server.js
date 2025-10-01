#!/usr/bin/env node

/**
 * Base MCP Server - Core server foundation
 * Provides the basic MCP server setup with modular architecture
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { Logger } from '../../utils/logger.js';
import { ToolHandler } from '../handlers/tool-handler.js';
import { ResourceHandler } from '../handlers/resource-handler.js';
import { PromptHandler } from '../handlers/prompt-handler.js';
import { ServiceRegistry } from '../registry/service-registry.js';
import { ErrorHandler } from '../middleware/error-handler.js';
import { ValidationMiddleware } from '../middleware/validation.js';

export class BaseMCPServer {
  constructor(config = {}) {
    this.config = {
      name: 'ml-mcp-system',
      version: '2.0.0',
      capabilities: {
        tools: {},
        resources: {},
        prompts: {}
      },
      ...config
    };

    this.logger = new Logger();
    this.server = null;
    this.transport = null;
    this.isInitialized = false;

    // Core components
    this.serviceRegistry = new ServiceRegistry();
    this.errorHandler = new ErrorHandler(this.logger);
    this.validation = new ValidationMiddleware();

    // Handlers
    this.toolHandler = null;
    this.resourceHandler = null;
    this.promptHandler = null;
  }

  /**
   * Initialize the MCP server and all components
   */
  async initialize() {
    try {
      this.logger.info('기본 MCP 서버 초기화 중', {
        name: this.config.name,
        version: this.config.version
      });

      // Create MCP server instance
      this.server = new Server(
        {
          name: this.config.name,
          version: this.config.version
        },
        {
          capabilities: this.config.capabilities
        }
      );

      // Initialize service registry
      await this.serviceRegistry.initialize();

      // Initialize handlers
      await this.initializeHandlers();

      // Setup server request handlers
      this.setupServerHandlers();

      // Setup error handling
      this.setupErrorHandling();

      this.isInitialized = true;
      this.logger.info('기본 MCP 서버 초기화 완료');

    } catch (error) {
      this.logger.error('기본 MCP 서버 초기화 실패:', error);
      throw error;
    }
  }

  /**
   * Initialize all request handlers
   */
  async initializeHandlers() {
    this.logger.info('요청 핸들러 초기화 중');

    // Initialize tool handler
    this.toolHandler = new ToolHandler(this.serviceRegistry, this.logger);
    await this.toolHandler.initialize();

    // Initialize resource handler
    this.resourceHandler = new ResourceHandler(this.serviceRegistry, this.logger);
    await this.resourceHandler.initialize();

    // Initialize prompt handler
    this.promptHandler = new PromptHandler(this.serviceRegistry, this.logger);
    await this.promptHandler.initialize();

    this.logger.info('모든 요청 핸들러 초기화 완료');
  }

  /**
   * Setup MCP server request handlers
   */
  setupServerHandlers() {
    this.logger.info('서버 요청 핸들러 설정 중');

    // Tools
    this.server.setRequestHandler('tools/list', async (request) => {
      return await this.handleRequest('tools/list', request,
        () => this.toolHandler.listTools(request)
      );
    });

    this.server.setRequestHandler('tools/call', async (request) => {
      return await this.handleRequest('tools/call', request,
        () => this.toolHandler.callTool(request)
      );
    });

    // Resources
    this.server.setRequestHandler('resources/list', async (request) => {
      return await this.handleRequest('resources/list', request,
        () => this.resourceHandler.listResources(request)
      );
    });

    this.server.setRequestHandler('resources/read', async (request) => {
      return await this.handleRequest('resources/read', request,
        () => this.resourceHandler.readResource(request)
      );
    });

    // Prompts
    this.server.setRequestHandler('prompts/list', async (request) => {
      return await this.handleRequest('prompts/list', request,
        () => this.promptHandler.listPrompts(request)
      );
    });

    this.server.setRequestHandler('prompts/get', async (request) => {
      return await this.handleRequest('prompts/get', request,
        () => this.promptHandler.getPrompt(request)
      );
    });

    this.logger.info('서버 요청 핸들러 설정 완료');
  }

  /**
   * Generic request handler with middleware pipeline
   */
  async handleRequest(handlerName, request, handlerFunction) {
    try {
      // Validation middleware
      await this.validation.validateRequest(handlerName, request);

      // Execute handler
      const result = await handlerFunction();

      // Return result
      return result;

    } catch (error) {
      // Error handling middleware
      return this.errorHandler.handleError(handlerName, error, request);
    }
  }

  /**
   * Setup global error handling
   */
  setupErrorHandling() {
    // Unhandled rejections
    process.on('unhandledRejection', (reason, promise) => {
      this.logger.error('처리되지 않은 Promise 거부:', { reason, promise });
      this.errorHandler.handleUnhandledRejection(reason, promise);
    });

    // Uncaught exceptions
    process.on('uncaughtException', (error) => {
      this.logger.error('처리되지 않은 예외:', error);
      this.errorHandler.handleUncaughtException(error);
    });

    // Graceful shutdown
    process.on('SIGINT', () => this.shutdown('SIGINT'));
    process.on('SIGTERM', () => this.shutdown('SIGTERM'));
  }

  /**
   * Register a service with the service registry
   */
  async registerService(serviceName, serviceInstance, metadata = {}) {
    if (!this.isInitialized) {
      throw new Error('Server must be initialized before registering services');
    }

    return await this.serviceRegistry.register(serviceName, serviceInstance, metadata);
  }

  /**
   * Register multiple services at once
   */
  async registerServices(services) {
    const registrationPromises = Object.entries(services).map(
      ([name, { instance, metadata }]) =>
        this.registerService(name, instance, metadata)
    );

    return await Promise.all(registrationPromises);
  }

  /**
   * Get a registered service
   */
  getService(serviceName) {
    return this.serviceRegistry.get(serviceName);
  }

  /**
   * Start the MCP server
   */
  async start(transportType = 'stdio') {
    if (!this.isInitialized) {
      throw new Error('Server must be initialized before starting');
    }

    try {
      this.logger.info('MCP 서버 시작 중', { transport: transportType });

      // Create transport
      switch (transportType) {
        case 'stdio':
          this.transport = new StdioServerTransport();
          break;
        default:
          throw new Error(`Unsupported transport type: ${transportType}`);
      }

      // Connect server to transport
      await this.server.connect(this.transport);

      this.logger.info('MCP 서버 시작 완료', {
        name: this.config.name,
        version: this.config.version,
        transport: transportType
      });

      return true;

    } catch (error) {
      this.logger.error('MCP 서버 시작 실패:', error);
      throw error;
    }
  }

  /**
   * Graceful shutdown
   */
  async shutdown(signal = 'SHUTDOWN') {
    try {
      this.logger.info(`MCP 서버 종료 중 (${signal})`);

      // Cleanup handlers
      if (this.toolHandler) {
        await this.toolHandler.cleanup?.();
      }
      if (this.resourceHandler) {
        await this.resourceHandler.cleanup?.();
      }
      if (this.promptHandler) {
        await this.promptHandler.cleanup?.();
      }

      // Cleanup service registry
      await this.serviceRegistry.cleanup?.();

      // Close transport
      if (this.transport) {
        await this.transport.close?.();
      }

      this.logger.info('MCP 서버 종료 완료');

    } catch (error) {
      this.logger.error('종료 중 오류 발생:', error);
    } finally {
      process.exit(0);
    }
  }

  /**
   * Get server status
   */
  getStatus() {
    return {
      name: this.config.name,
      version: this.config.version,
      initialized: this.isInitialized,
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      services: this.serviceRegistry.getStatus(),
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Health check
   */
  async healthCheck() {
    const status = this.getStatus();

    const healthChecks = {
      server: this.isInitialized,
      serviceRegistry: this.serviceRegistry.isHealthy(),
      handlers: {
        tools: this.toolHandler?.isHealthy() ?? false,
        resources: this.resourceHandler?.isHealthy() ?? false,
        prompts: this.promptHandler?.isHealthy() ?? false
      }
    };

    const isHealthy = Object.values(healthChecks).every(check =>
      typeof check === 'boolean' ? check : Object.values(check).every(Boolean)
    );

    return {
      status: isHealthy ? 'healthy' : 'unhealthy',
      checks: healthChecks,
      ...status
    };
  }
}

export default BaseMCPServer;