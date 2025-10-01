/**
 * Tool Handler - Manages MCP tool requests
 * Handles tool listing and execution through the service registry
 */

export class ToolHandler {
  constructor(serviceRegistry, logger) {
    this.serviceRegistry = serviceRegistry;
    this.logger = logger;
    this.toolServices = new Map();
    this.isInitialized = false;
  }

  /**
   * Initialize the tool handler
   */
  async initialize() {
    try {
      this.logger.info('도구 핸들러 초기화 중');

      // Discover and register tool services
      await this.discoverToolServices();

      this.isInitialized = true;
      this.logger.info('도구 핸들러 초기화 완료');

    } catch (error) {
      this.logger.error('도구 핸들러 초기화 실패:', error);
      throw error;
    }
  }

  /**
   * Discover tool services from the service registry
   */
  async discoverToolServices() {
    // Find services that provide tools
    const toolProviders = this.serviceRegistry.findByCapability('tools');

    this.logger.info(`${toolProviders.length}개 도구 제공 서비스 발견`);

    for (const { name, service, metadata } of toolProviders) {
      try {
        // Get tools from the service
        const tools = await this.getToolsFromService(service);

        if (tools && tools.length > 0) {
          this.toolServices.set(name, {
            service,
            tools,
            metadata
          });

          this.logger.info(`서비스 ${name}에서 ${tools.length}개 도구 등록`);
        }

      } catch (error) {
        this.logger.error(`서비스 ${name}에서 도구 가져오기 실패:`, error);
      }
    }
  }

  /**
   * Get tools from a service
   */
  async getToolsFromService(service) {
    if (typeof service.getTools === 'function') {
      return await service.getTools();
    }

    if (typeof service.listTools === 'function') {
      return await service.listTools();
    }

    if (service.tools && Array.isArray(service.tools)) {
      return service.tools;
    }

    return [];
  }

  /**
   * List all available tools
   */
  async listTools(request) {
    if (!this.isInitialized) {
      throw new Error('Tool handler not initialized');
    }

    try {
      const allTools = [];

      // Collect tools from all registered services
      for (const [serviceName, { tools, metadata }] of this.toolServices) {
        for (const tool of tools) {
          allTools.push({
            ...tool,
            _service: serviceName,
            _serviceType: metadata.type,
            _serviceVersion: metadata.version
          });
        }
      }

      this.logger.info(`${this.toolServices.size}개 서비스에서 ${allTools.length}개 도구 목록 반환`);

      return {
        tools: allTools
      };

    } catch (error) {
      this.logger.error('도구 목록 가져오기 실패:', error);
      throw error;
    }
  }

  /**
   * Call a specific tool
   */
  async callTool(request) {
    if (!this.isInitialized) {
      throw new Error('Tool handler not initialized');
    }

    try {
      const { name: toolName, arguments: toolArgs } = request.params;

      if (!toolName) {
        throw new Error('Tool name is required');
      }

      this.logger.info(`도구 호출 중: ${toolName}`, { args: toolArgs });

      // Find the service that provides this tool
      const { service, serviceName } = await this.findToolService(toolName);

      if (!service) {
        throw new Error(`Tool '${toolName}' not found`);
      }

      // Execute the tool
      const result = await this.executeToolOnService(service, toolName, toolArgs);

      this.logger.info(`도구 '${toolName}' 실행 완료`);

      return result;

    } catch (error) {
      this.logger.error(`도구 '${request.params.name}' 호출 실패:`, error);
      throw error;
    }
  }

  /**
   * Find the service that provides a specific tool
   */
  async findToolService(toolName) {
    for (const [serviceName, { service, tools }] of this.toolServices) {
      const tool = tools.find(t => t.name === toolName);
      if (tool) {
        return { service, serviceName, tool };
      }
    }

    return { service: null, serviceName: null, tool: null };
  }

  /**
   * Execute a tool on a specific service
   */
  async executeToolOnService(service, toolName, toolArgs) {
    // Try different method names for tool execution
    const executeMethods = [
      'executeTool',
      'callTool',
      'runTool',
      'invoke',
      'execute'
    ];

    for (const methodName of executeMethods) {
      if (typeof service[methodName] === 'function') {
        return await service[methodName](toolName, toolArgs);
      }
    }

    // Try dynamic method based on tool name
    const toolMethodName = this.getToolMethodName(toolName);
    if (typeof service[toolMethodName] === 'function') {
      return await service[toolMethodName](toolArgs);
    }

    throw new Error(`Service does not support tool execution for: ${toolName}`);
  }

  /**
   * Convert tool name to method name
   */
  getToolMethodName(toolName) {
    // Convert tool_name to toolName or handleToolName
    const camelCase = toolName.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
    return `handle${camelCase.charAt(0).toUpperCase() + camelCase.slice(1)}`;
  }

  /**
   * Refresh tool discovery
   */
  async refresh() {
    this.logger.info('도구 서비스 새로고침 중');

    this.toolServices.clear();
    await this.discoverToolServices();

    this.logger.info('도구 서비스 새로고침 완료');

    return {
      success: true,
      serviceCount: this.toolServices.size,
      toolCount: this.getTotalToolCount()
    };
  }

  /**
   * Get total number of tools
   */
  getTotalToolCount() {
    let count = 0;
    for (const { tools } of this.toolServices.values()) {
      count += tools.length;
    }
    return count;
  }

  /**
   * Get tool statistics
   */
  getStats() {
    const stats = {
      serviceCount: this.toolServices.size,
      toolCount: this.getTotalToolCount(),
      services: []
    };

    for (const [serviceName, { tools, metadata }] of this.toolServices) {
      stats.services.push({
        name: serviceName,
        type: metadata.type,
        version: metadata.version,
        toolCount: tools.length,
        tools: tools.map(t => t.name)
      });
    }

    return stats;
  }

  /**
   * Validate tool request
   */
  validateToolRequest(toolName, toolArgs, toolSchema) {
    if (!toolSchema || !toolSchema.inputSchema) {
      return { valid: true };
    }

    const schema = toolSchema.inputSchema;

    // Basic validation - can be extended with a JSON schema validator
    if (schema.required) {
      for (const requiredField of schema.required) {
        if (!(requiredField in toolArgs)) {
          return {
            valid: false,
            error: `Missing required parameter: ${requiredField}`
          };
        }
      }
    }

    return { valid: true };
  }

  /**
   * Health check
   */
  isHealthy() {
    return this.isInitialized && this.serviceRegistry.isHealthy();
  }

  /**
   * Get tool handler status
   */
  getStatus() {
    return {
      initialized: this.isInitialized,
      serviceCount: this.toolServices.size,
      toolCount: this.getTotalToolCount(),
      healthy: this.isHealthy(),
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Cleanup
   */
  async cleanup() {
    this.logger.info('도구 핸들러 정리 중');

    this.toolServices.clear();
    this.isInitialized = false;

    this.logger.info('도구 핸들러 정리 완료');
  }
}

export default ToolHandler;