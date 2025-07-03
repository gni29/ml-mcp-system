// utils/error-handler.js
import { Logger } from './logger.js';

export class ErrorHandler {
  constructor() {
    this.logger = new Logger();
  }

  handleError(error, context = {}) {
    const errorInfo = {
      message: error.message,
      stack: error.stack,
      context,
      timestamp: new Date().toISOString()
    };

    this.logger.error('오류 발생:', errorInfo);

    // 오류 유형별 처리
    if (error.code === 'ECONNREFUSED') {
      return this.handleConnectionError(error);
    } else if (error.code === 'TIMEOUT') {
      return this.handleTimeoutError(error);
    } else if (error.name === 'ValidationError') {
      return this.handleValidationError(error);
    } else {
      return this.handleGenericError(error);
    }
  }

  handleConnectionError(error) {
    return {
      content: [{
        type: 'text',
        text: '서비스에 연결할 수 없습니다. Ollama 서비스가 실행 중인지 확인해주세요.'
      }],
      isError: true,
      errorType: 'connection'
    };
  }

  handleTimeoutError(error) {
    return {
      content: [{
        type: 'text',
        text: '요청 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.'
      }],
      isError: true,
      errorType: 'timeout'
    };
  }

  handleValidationError(error) {
    return {
      content: [{
        type: 'text',
        text: `입력값이 올바르지 않습니다: ${error.message}`
      }],
      isError: true,
      errorType: 'validation'
    };
  }

  handleGenericError(error) {
    return {
      content: [{
        type: 'text',
        text: `오류가 발생했습니다: ${error.message}`
      }],
      isError: true,
      errorType: 'generic'
    };
  }
}
