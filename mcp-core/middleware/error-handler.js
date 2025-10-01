/**
 * Error Handler Middleware - Centralized error handling for MCP requests
 * Provides consistent error responses and logging across all handlers
 */

export class ErrorHandler {
  constructor(logger) {
    this.logger = logger;
    this.errorCounts = new Map();
    this.lastErrors = [];
    this.maxLastErrors = 100;
  }

  /**
   * Handle MCP request errors
   */
  handleError(handlerName, error, request = null) {
    // Record error
    this.recordError(handlerName, error);

    // Log error with context
    this.logError(handlerName, error, request);

    // Generate user-friendly error response
    return this.generateErrorResponse(handlerName, error, request);
  }

  /**
   * Record error statistics
   */
  recordError(handlerName, error) {
    const errorKey = `${handlerName}:${error.constructor.name}`;

    if (!this.errorCounts.has(errorKey)) {
      this.errorCounts.set(errorKey, 0);
    }

    this.errorCounts.set(errorKey, this.errorCounts.get(errorKey) + 1);

    // Store recent errors
    this.lastErrors.unshift({
      timestamp: new Date().toISOString(),
      handler: handlerName,
      errorType: error.constructor.name,
      message: error.message,
      stack: error.stack
    });

    // Limit stored errors
    if (this.lastErrors.length > this.maxLastErrors) {
      this.lastErrors = this.lastErrors.slice(0, this.maxLastErrors);
    }
  }

  /**
   * Log error with context
   */
  logError(handlerName, error, request = null) {
    const logContext = {
      handler: handlerName,
      errorType: error.constructor.name,
      message: error.message
    };

    if (request) {
      logContext.requestParams = request.params;
      logContext.requestMethod = request.method;
    }

    this.logger.error(`Error in ${handlerName}:`, logContext);

    // Log stack trace for debugging
    if (error.stack) {
      this.logger.debug('Error stack trace:', error.stack);
    }
  }

  /**
   * Generate user-friendly error response
   */
  generateErrorResponse(handlerName, error, request = null) {
    const errorType = this.categorizeError(error);
    const userMessage = this.getUserMessage(errorType, error);
    const troubleshooting = this.getTroubleshootingTips(errorType, handlerName);

    return {
      content: [{
        type: 'text',
        text: this.formatErrorMessage(userMessage, troubleshooting, handlerName, error)
      }],
      isError: true,
      _meta: {
        errorType,
        handler: handlerName,
        timestamp: new Date().toISOString(),
        requestId: this.generateRequestId()
      }
    };
  }

  /**
   * Categorize error types
   */
  categorizeError(error) {
    const errorMessage = error.message.toLowerCase();

    if (error.name === 'ValidationError' || errorMessage.includes('validation')) {
      return 'validation';
    }

    if (errorMessage.includes('not found') || errorMessage.includes('missing')) {
      return 'not_found';
    }

    if (errorMessage.includes('timeout') || errorMessage.includes('timed out')) {
      return 'timeout';
    }

    if (errorMessage.includes('permission') || errorMessage.includes('unauthorized')) {
      return 'permission';
    }

    if (errorMessage.includes('network') || errorMessage.includes('connection')) {
      return 'network';
    }

    if (errorMessage.includes('syntax') || errorMessage.includes('invalid')) {
      return 'syntax';
    }

    if (errorMessage.includes('dependency') || errorMessage.includes('module')) {
      return 'dependency';
    }

    return 'internal';
  }

  /**
   * Get user-friendly error message
   */
  getUserMessage(errorType, error) {
    const messages = {
      validation: '입력 값이 유효하지 않습니다. 요청 형식을 확인해주세요.',
      not_found: '요청한 리소스를 찾을 수 없습니다.',
      timeout: '요청 처리 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.',
      permission: '권한이 없거나 인증에 실패했습니다.',
      network: '네트워크 연결에 문제가 있습니다.',
      syntax: '요청 구문에 오류가 있습니다.',
      dependency: '필요한 모듈이나 서비스가 사용할 수 없습니다.',
      internal: '내부 서버 오류가 발생했습니다.'
    };

    return messages[errorType] || messages.internal;
  }

  /**
   * Get troubleshooting tips
   */
  getTroubleshootingTips(errorType, handlerName) {
    const tips = {
      validation: [
        '요청 파라미터의 형식과 필수 필드를 확인하세요',
        '데이터 타입이 올바른지 확인하세요',
        'API 문서를 참조하여 올바른 형식으로 요청하세요'
      ],
      not_found: [
        '요청한 도구, 리소스 또는 프롬프트 이름을 확인하세요',
        '\'list\' 명령으로 사용 가능한 항목을 확인하세요',
        '서비스가 올바르게 등록되었는지 확인하세요'
      ],
      timeout: [
        '잠시 후 다시 시도하세요',
        '더 작은 데이터셋으로 테스트해보세요',
        '시스템 상태를 확인하세요'
      ],
      permission: [
        '인증 정보를 확인하세요',
        '필요한 권한이 있는지 확인하세요',
        '관리자에게 문의하세요'
      ],
      network: [
        '인터넷 연결을 확인하세요',
        '방화벽 설정을 확인하세요',
        '잠시 후 다시 시도하세요'
      ],
      syntax: [
        '요청 형식을 다시 확인하세요',
        'JSON 구문이 올바른지 확인하세요',
        'API 문서의 예제를 참조하세요'
      ],
      dependency: [
        '필요한 Python 모듈이 설치되었는지 확인하세요',
        '시스템 유효성 검사를 실행하세요',
        '모듈을 새로고침하세요'
      ],
      internal: [
        '시스템 상태를 확인하세요',
        '로그를 확인하여 자세한 정보를 얻으세요',
        '문제가 지속되면 관리자에게 문의하세요'
      ]
    };

    return tips[errorType] || tips.internal;
  }

  /**
   * Format complete error message
   */
  formatErrorMessage(userMessage, troubleshooting, handlerName, error) {
    const requestId = this.generateRequestId();

    return `❌ **오류 발생**

**문제:** ${userMessage}

**위치:** ${handlerName}

🔧 **해결 방법:**
${troubleshooting.map(tip => `   • ${tip}`).join('\n')}

📋 **기술 정보:**
   • 오류 유형: ${error.constructor.name}
   • 요청 ID: ${requestId}
   • 시간: ${new Date().toLocaleString('ko-KR')}

💡 **추가 도움이 필요하시면:**
   • \`system_status\` 명령으로 시스템 상태를 확인하세요
   • \`validate_modules\` 명령으로 모듈 상태를 확인하세요
   • 로그를 확인하여 자세한 정보를 얻으세요`;
  }

  /**
   * Generate unique request ID
   */
  generateRequestId() {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Handle unhandled promise rejections
   */
  handleUnhandledRejection(reason, promise) {
    this.logger.error('Unhandled Promise Rejection:', {
      reason: reason?.message || reason,
      stack: reason?.stack
    });

    // Don't exit process, just log
    // In production, you might want to implement more sophisticated handling
  }

  /**
   * Handle uncaught exceptions
   */
  handleUncaughtException(error) {
    this.logger.error('Uncaught Exception:', {
      message: error.message,
      stack: error.stack
    });

    // Graceful shutdown
    setTimeout(() => {
      process.exit(1);
    }, 1000);
  }

  /**
   * Get error statistics
   */
  getErrorStats() {
    const stats = {
      totalErrors: this.lastErrors.length,
      errorsByType: {},
      errorsByHandler: {},
      recentErrors: this.lastErrors.slice(0, 10)
    };

    // Group errors by type and handler
    for (const [key, count] of this.errorCounts) {
      const [handler, errorType] = key.split(':');

      if (!stats.errorsByHandler[handler]) {
        stats.errorsByHandler[handler] = 0;
      }
      stats.errorsByHandler[handler] += count;

      if (!stats.errorsByType[errorType]) {
        stats.errorsByType[errorType] = 0;
      }
      stats.errorsByType[errorType] += count;
    }

    return stats;
  }

  /**
   * Clear error history
   */
  clearErrorHistory() {
    this.errorCounts.clear();
    this.lastErrors = [];

    this.logger.info('Error history cleared');

    return {
      success: true,
      message: 'Error history has been cleared',
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Set error tracking configuration
   */
  configure(options = {}) {
    if (options.maxLastErrors) {
      this.maxLastErrors = options.maxLastErrors;
    }

    return {
      success: true,
      configuration: {
        maxLastErrors: this.maxLastErrors
      }
    };
  }

  /**
   * Get error handler status
   */
  getStatus() {
    return {
      errorCounts: Object.fromEntries(this.errorCounts),
      totalErrors: this.lastErrors.length,
      maxLastErrors: this.maxLastErrors,
      healthy: true,
      timestamp: new Date().toISOString()
    };
  }
}

export default ErrorHandler;