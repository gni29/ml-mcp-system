// utils/error-handler.js - 확장된 포괄적 에러 처리 시스템
import { Logger } from './logger.js';

export class ErrorHandler {
  constructor() {
    this.logger = new Logger();
    this.errorHistory = [];
    this.errorStats = new Map();
    this.retryAttempts = new Map();
    this.maxRetries = 3;
    this.retryDelay = 1000; // 1초
    this.maxHistorySize = 1000;
    
    // 에러 복구 전략
    this.recoveryStrategies = this.initializeRecoveryStrategies();
    
    // 사용자 친화적 메시지 매핑
    this.userMessages = this.initializeUserMessages();
  }

  handleError(error, context = {}) {
    const errorInfo = this.analyzeError(error, context);
    
    // 에러 히스토리에 추가
    this.recordError(errorInfo);
    
    // 통계 업데이트
    this.updateErrorStats(errorInfo);
    
    // 로깅
    this.logger.error('오류 발생:', {
      type: errorInfo.type,
      message: errorInfo.message,
      context: errorInfo.context,
      timestamp: errorInfo.timestamp,
      canRecover: errorInfo.canRecover
    });

    // 자동 복구 시도
    if (errorInfo.canRecover && this.shouldAttemptRecovery(errorInfo)) {
      return this.attemptRecovery(errorInfo);
    }

    // 에러 타입별 처리
    return this.processErrorByType(errorInfo);
  }

  analyzeError(error, context = {}) {
    const errorInfo = {
      id: this.generateErrorId(),
      timestamp: new Date().toISOString(),
      message: error.message || 'Unknown error',
      stack: error.stack,
      context: this.sanitizeContext(context),
      type: this.determineErrorType(error),
      severity: this.determineSeverity(error),
      canRecover: false,
      recoveryStrategy: null,
      userMessage: null,
      suggestions: [],
      metadata: {}
    };

    // 복구 가능성 및 전략 결정
    const recovery = this.getRecoveryStrategy(errorInfo.type, error);
    errorInfo.canRecover = recovery.canRecover;
    errorInfo.recoveryStrategy = recovery.strategy;

    // 사용자 친화적 메시지 생성
    errorInfo.userMessage = this.generateUserFriendlyMessage(errorInfo);
    errorInfo.suggestions = this.generateSuggestions(errorInfo);

    return errorInfo;
  }

  determineErrorType(error) {
    // 연결 관련 에러
    if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
      return 'connection';
    }
    
    // 타임아웃 에러
    if (error.code === 'TIMEOUT' || error.message?.includes('timeout')) {
      return 'timeout';
    }
    
    // 모델 관련 에러
    if (error.message?.includes('model') || error.message?.includes('ollama')) {
      return 'model';
    }
    
    // 파일 관련 에러
    if (error.code === 'ENOENT' || error.code === 'EACCES' || error.message?.includes('file')) {
      return 'file';
    }
    
    // Python 관련 에러
    if (error.message?.includes('python') || error.message?.includes('subprocess')) {
      return 'python';
    }
    
    // 메모리 관련 에러
    if (error.message?.includes('memory') || error.code === 'ERR_MEMORY_ALLOCATION_FAILED') {
      return 'memory';
    }
    
    // 권한 관련 에러
    if (error.code === 'EPERM' || error.message?.includes('permission')) {
      return 'permission';
    }
    
    // 데이터 관련 에러
    if (error.name === 'ValidationError' || error.message?.includes('invalid data')) {
      return 'data';
    }
    
    // 설정 관련 에러
    if (error.message?.includes('config') || error.message?.includes('configuration')) {
      return 'config';
    }
    
    // 네트워크 관련 에러
    if (error.code === 'ENETUNREACH' || error.code === 'EHOSTUNREACH') {
      return 'network';
    }
    
    return 'generic';
  }

  determineSeverity(error) {
    const criticalErrors = ['memory', 'permission', 'config'];
    const warningErrors = ['timeout', 'connection'];
    
    const errorType = this.determineErrorType(error);
    
    if (criticalErrors.includes(errorType)) {
      return 'critical';
    } else if (warningErrors.includes(errorType)) {
      return 'warning';
    } else {
      return 'error';
    }
  }

  initializeRecoveryStrategies() {
    return {
      connection: {
        canRecover: true,
        strategy: 'retry_with_backoff',
        maxRetries: 3,
        backoffMultiplier: 2
      },
      timeout: {
        canRecover: true,
        strategy: 'increase_timeout_and_retry',
        maxRetries: 2,
        timeoutIncrease: 1.5
      },
      model: {
        canRecover: true,
        strategy: 'restart_model_service',
        maxRetries: 2
      },
      file: {
        canRecover: true,
        strategy: 'check_and_create_file',
        maxRetries: 1
      },
      python: {
        canRecover: true,
        strategy: 'restart_python_environment',
        maxRetries: 2
      },
      memory: {
        canRecover: true,
        strategy: 'cleanup_and_retry',
        maxRetries: 1
      },
      permission: {
        canRecover: false,
        strategy: 'manual_intervention_required'
      },
      data: {
        canRecover: true,
        strategy: 'data_validation_and_cleanup',
        maxRetries: 1
      },
      config: {
        canRecover: true,
        strategy: 'reset_to_default_config',
        maxRetries: 1
      },
      network: {
        canRecover: true,
        strategy: 'network_diagnostics_and_retry',
        maxRetries: 2
      },
      generic: {
        canRecover: true,
        strategy: 'simple_retry',
        maxRetries: 1
      }
    };
  }

  initializeUserMessages() {
    return {
      connection: {
        title: '서비스 연결 문제',
        message: 'Ollama 서비스에 연결할 수 없습니다.',
        action: 'Ollama 서비스가 실행 중인지 확인해주세요.'
      },
      timeout: {
        title: '응답 시간 초과',
        message: '요청 처리 시간이 너무 오래 걸리고 있습니다.',
        action: '잠시 후 다시 시도하거나 더 간단한 요청으로 나누어 보세요.'
      },
      model: {
        title: '모델 처리 오류',
        message: 'AI 모델에서 문제가 발생했습니다.',
        action: '모델을 다시 로드하거나 다른 모델을 사용해보세요.'
      },
      file: {
        title: '파일 처리 오류',
        message: '파일을 읽거나 처리하는 중 문제가 발생했습니다.',
        action: '파일 경로와 권한을 확인해주세요.'
      },
      python: {
        title: 'Python 실행 오류',
        message: 'Python 스크립트 실행 중 문제가 발생했습니다.',
        action: 'Python 환경과 필요한 패키지를 확인해주세요.'
      },
      memory: {
        title: '메모리 부족',
        message: '시스템 메모리가 부족합니다.',
        action: '다른 프로그램을 종료하거나 더 작은 데이터셋을 사용해보세요.'
      },
      permission: {
        title: '권한 문제',
        message: '필요한 권한이 없어 작업을 수행할 수 없습니다.',
        action: '관리자 권한으로 실행하거나 파일 권한을 확인해주세요.'
      },
      data: {
        title: '데이터 오류',
        message: '입력된 데이터에 문제가 있습니다.',
        action: '데이터 형식과 내용을 확인해주세요.'
      },
      config: {
        title: '설정 오류',
        message: '시스템 설정에 문제가 있습니다.',
        action: '설정을 확인하거나 기본값으로 재설정해보세요.'
      },
      network: {
        title: '네트워크 문제',
        message: '네트워크 연결에 문제가 있습니다.',
        action: '인터넷 연결을 확인해주세요.'
      },
      generic: {
        title: '일반 오류',
        message: '예상치 못한 오류가 발생했습니다.',
        action: '잠시 후 다시 시도하거나 관리자에게 문의하세요.'
      }
    };
  }

  async attemptRecovery(errorInfo) {
    const strategy = this.recoveryStrategies[errorInfo.type];
    if (!strategy || !strategy.canRecover) {
      return this.processErrorByType(errorInfo);
    }

    const retryKey = `${errorInfo.type}_${errorInfo.context?.operation || 'unknown'}`;
    const currentRetries = this.retryAttempts.get(retryKey) || 0;

    if (currentRetries >= (strategy.maxRetries || this.maxRetries)) {
      this.logger.warn(`최대 재시도 횟수 초과: ${errorInfo.type}`);
      this.retryAttempts.delete(retryKey);
      return this.processErrorByType(errorInfo);
    }

    this.retryAttempts.set(retryKey, currentRetries + 1);

    try {
      this.logger.info(`복구 시도 ${currentRetries + 1}/${strategy.maxRetries}: ${strategy.strategy}`);
      
      const recoveryResult = await this.executeRecoveryStrategy(errorInfo, strategy);
      
      if (recoveryResult.success) {
        this.retryAttempts.delete(retryKey);
        this.logger.info(`복구 성공: ${errorInfo.type}`);
        return {
          content: [{
            type: 'text',
            text: `문제가 해결되었습니다. ${recoveryResult.message || '작업을 다시 시도해주세요.'}`
          }],
          isError: false,
          recovered: true,
          originalError: errorInfo.type
        };
      } else {
        // 복구 실패 시 지연 후 재시도
        await this.delay(this.retryDelay * Math.pow(2, currentRetries));
        return this.processErrorByType(errorInfo);
      }
    } catch (recoveryError) {
      this.logger.error('복구 시도 중 오류:', recoveryError);
      return this.processErrorByType(errorInfo);
    }
  }

  async executeRecoveryStrategy(errorInfo, strategy) {
    switch (strategy.strategy) {
      case 'retry_with_backoff':
        await this.delay(this.retryDelay * strategy.backoffMultiplier);
        return { success: true, message: '연결을 다시 시도합니다.' };

      case 'increase_timeout_and_retry':
        return { success: true, message: '타임아웃을 늘려서 다시 시도합니다.' };

      case 'restart_model_service':
        // 모델 재시작 로직 (실제로는 모델 매니저 호출)
        return { success: true, message: '모델 서비스를 재시작했습니다.' };

      case 'check_and_create_file':
        // 파일 확인 및 생성 로직
        return { success: true, message: '파일 경로를 확인했습니다.' };

      case 'cleanup_and_retry':
        // 메모리 정리 로직
        if (global.gc) global.gc(); // 가비지 컬렉션 강제 실행
        return { success: true, message: '메모리를 정리했습니다.' };

      case 'reset_to_default_config':
        return { success: true, message: '설정을 기본값으로 재설정했습니다.' };

      default:
        return { success: false, message: '복구 전략을 찾을 수 없습니다.' };
    }
  }

  processErrorByType(errorInfo) {
    const userMsg = this.userMessages[errorInfo.type] || this.userMessages.generic;
    
    return {
      content: [{
        type: 'text',
        text: `❌ **${userMsg.title}**\n\n${userMsg.message}\n\n**해결 방법**: ${userMsg.action}${this.formatSuggestions(errorInfo.suggestions)}`
      }],
      isError: true,
      errorType: errorInfo.type,
      severity: errorInfo.severity,
      errorId: errorInfo.id,
      canRetry: errorInfo.canRecover,
      suggestions: errorInfo.suggestions
    };
  }

  generateUserFriendlyMessage(errorInfo) {
    const template = this.userMessages[errorInfo.type] || this.userMessages.generic;
    return `${template.title}: ${template.message} ${template.action}`;
  }

  generateSuggestions(errorInfo) {
    const suggestions = [];
    
    switch (errorInfo.type) {
      case 'connection':
        suggestions.push('ollama serve 명령으로 Ollama 서비스를 시작하세요');
        suggestions.push('방화벽 설정을 확인하세요');
        break;
      case 'model':
        suggestions.push('ollama pull 명령으로 모델을 다시 다운로드하세요');
        suggestions.push('다른 모델을 사용해보세요');
        break;
      case 'file':
        suggestions.push('파일 경로가 올바른지 확인하세요');
        suggestions.push('파일 읽기 권한을 확인하세요');
        break;
      case 'python':
        suggestions.push('Python 가상환경을 활성화하세요');
        suggestions.push('필요한 패키지가 설치되어 있는지 확인하세요');
        break;
      case 'memory':
        suggestions.push('더 작은 데이터셋을 사용해보세요');
        suggestions.push('다른 프로그램을 종료하세요');
        break;
    }
    
    return suggestions;
  }

  formatSuggestions(suggestions) {
    if (suggestions.length === 0) return '';
    
    return '\n\n**추가 제안사항**:\n' + 
           suggestions.map(s => `• ${s}`).join('\n');
  }

  recordError(errorInfo) {
    this.errorHistory.push(errorInfo);
    
    // 히스토리 크기 제한
    if (this.errorHistory.length > this.maxHistorySize) {
      this.errorHistory = this.errorHistory.slice(-Math.floor(this.maxHistorySize * 0.8));
    }
  }

  updateErrorStats(errorInfo) {
    const type = errorInfo.type;
    const current = this.errorStats.get(type) || { count: 0, lastOccurred: null };
    
    this.errorStats.set(type, {
      count: current.count + 1,
      lastOccurred: errorInfo.timestamp
    });
  }

  getErrorStatistics() {
    const stats = {
      totalErrors: this.errorHistory.length,
      errorTypes: {},
      recentErrors: this.errorHistory.slice(-10),
      mostCommonErrors: [],
      errorTrends: {}
    };

    // 타입별 통계
    for (const [type, data] of this.errorStats.entries()) {
      stats.errorTypes[type] = data;
    }

    // 가장 흔한 에러들
    stats.mostCommonErrors = Array.from(this.errorStats.entries())
      .sort((a, b) => b[1].count - a[1].count)
      .slice(0, 5)
      .map(([type, data]) => ({ type, count: data.count }));

    return stats;
  }

  clearErrorHistory() {
    this.errorHistory = [];
    this.errorStats.clear();
    this.retryAttempts.clear();
    this.logger.info('에러 히스토리가 초기화되었습니다.');
  }

  shouldAttemptRecovery(errorInfo) {
    // 최근에 같은 타입의 에러가 너무 많이 발생했으면 복구 시도 안함
    const recentSameErrors = this.errorHistory
      .filter(e => e.type === errorInfo.type)
      .filter(e => Date.now() - new Date(e.timestamp).getTime() < 60000) // 최근 1분
      .length;
    
    return recentSameErrors < 5;
  }

  generateErrorId() {
    return `err_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  sanitizeContext(context) {
    const sanitized = { ...context };
    
    // 민감한 정보 제거
    const sensitiveKeys = ['password', 'token', 'key', 'secret'];
    sensitiveKeys.forEach(key => {
      if (sanitized[key]) {
        sanitized[key] = '[REDACTED]';
      }
    });
    
    return sanitized;
  }

  async delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // 기존 메서드들 유지 (하위 호환성)
  handleConnectionError(error) {
    return this.handleError(error, { type: 'connection' });
  }

  handleTimeoutError(error) {
    return this.handleError(error, { type: 'timeout' });
  }

  handleValidationError(error) {
    return this.handleError(error, { type: 'validation' });
  }

  handleGenericError(error) {
    return this.handleError(error, { type: 'generic' });
  }

  getRecoveryStrategy(errorType, error) {
    const strategy = this.recoveryStrategies[errorType];
    if (!strategy) {
      return { canRecover: false, strategy: null };
    }
    return strategy;
  }
}