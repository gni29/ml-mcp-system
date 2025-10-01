/**
 * Base Service Class for ML MCP Services
 * 모든 MCP 서비스의 기본 클래스
 */

export class BaseService {
  constructor(name, type, version = '1.0.0') {
    this.name = name;
    this.type = type;
    this.version = version;
    this.capabilities = ['tools'];
    this.isInitialized = false;
    this.logger = null;
  }

  /**
   * Initialize the service
   */
  async initialize() {
    this.isInitialized = true;
    if (this.logger) {
      this.logger.info(`${this.name} 서비스 초기화 완료`);
    }
  }

  /**
   * Health check
   */
  isHealthy() {
    return this.isInitialized;
  }

  /**
   * Get service status
   */
  getStatus() {
    return {
      name: this.name,
      type: this.type,
      version: this.version,
      initialized: this.isInitialized,
      healthy: this.isHealthy(),
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Cleanup resources
   */
  async cleanup() {
    if (this.logger) {
      this.logger.info(`${this.name} 서비스 정리 중`);
    }
    this.isInitialized = false;
  }

  /**
   * Abstract method - must be implemented by subclasses
   */
  async getTools() {
    throw new Error('getTools() must be implemented by subclass');
  }

  /**
   * Abstract method - must be implemented by subclasses
   */
  async executeTool(toolName, args) {
    throw new Error('executeTool() must be implemented by subclass');
  }
}

export default BaseService;