// utils/error-handler.js
import { Logger } from './logger.js';

export class ErrorHandler {
  constructor() {
    this.logger = new Logger();
    this.errorCounts = new Map();
    this.errorHistory = [];
    this.maxHistorySize = 1000;
    this.alertThresholds = {
      critical: 5,    // 5분 내 동일 에러 5회 시 critical
      warning: 3,     // 5분 내 동일 에러 3회 시 warning
      timeWindow: 300000  // 5분 (밀리초)
    };
    
    this.setupGlobalErrorHandlers();
  }

  setupGlobalErrorHandlers() {
    // 처리되지 않은 Promise 거부 처리
    process.on('unhandledRejection', (reason, promise) => {
      this.handleUnhandledRejection(reason, promise);
    });

    // 처리되지 않은 예외 처리
    process.on('uncaughtException', (error) => {
      this.handleUncaughtException(error);
    });

    // 경고 처리
    process.on('warning', (warning) => {
      this.handleWarning(warning);
    });
  }

  handleError(error, context = {}) {
    try {
      // 에러 정규화
      const normalizedError = this.normalizeError(error);
      
      // 에러 분류
      const errorClassification = this.classifyError(normalizedError);
      
      // 컨텍스트 정보 추가
      const enrichedContext = this.enrichContext(context, normalizedError);
      
      // 에러 로깅
      this.logError(normalizedError, errorClassification, enrichedContext);
      
      // 에러 추적
      this.trackError(normalizedError, errorClassification);
      
      // 에러 대응 결정
      const response = this.determineResponse(normalizedError, errorClassification, enrichedContext);
      
      // 알림 확인
      this.checkAlertThresholds(normalizedError);
      
      return response;

    } catch (handlingError) {
      // 에러 처리 중 에러 발생 시 기본 로깅
      console.error('에러 처리 중 오류 발생:', handlingError);
      console.error('원본 에러:', error);
      
      return {
        handled: false,
        action: 'log_only',
        message: '에러 처리 중 문제가 발생했습니다.',
        shouldRetry: false,
        shouldTerminate: false
      };
    }
  }

  normalizeError(error) {
    if (error instanceof Error) {
      return {
        name: error.name,
        message: error.message,
        stack: error.stack,
        code: error.code,
        syscall: error.syscall,
        errno: error.errno,
        path: error.path,
        timestamp: new Date().toISOString(),
        type: 'Error'
      };
    }

    if (typeof error === 'string') {
      return {
        name: 'StringError',
        message: error,
        stack: new Error().stack,
        timestamp: new Date().toISOString(),
        type: 'String'
      };
    }

    if (typeof error === 'object' && error !== null) {
      return {
        name: error.name || 'UnknownError',
        message: error.message || JSON.stringify(error),
        stack: error.stack || new Error().stack,
        timestamp: new Date().toISOString(),
        type: 'Object',
        ...error
      };
    }

    return {
      name: 'UnknownError',
      message: String(error),
      stack: new Error().stack,
      timestamp: new Date().toISOString(),
      type: typeof error
    };
  }

  classifyError(error) {
    const classification = {
      category: 'unknown',
      severity: 'medium',
      source: 'application',
      recoverable: true,
      userFacing: false
    };

    // 이름 기반 분류
    switch (error.name) {
      case 'TypeError':
      case 'ReferenceError':
      case 'SyntaxError':
        classification.category = 'programming';
        classification.severity = 'high';
        classification.recoverable = false;
        break;

      case 'ENOENT':
      case 'EACCES':
      case 'EPERM':
        classification.category = 'filesystem';
        classification.severity = 'medium';
        classification.source = 'system';
        break;

      case 'ECONNREFUSED':
      case 'ENOTFOUND':
      case 'ETIMEDOUT':
        classification.category = 'network';
        classification.severity = 'medium';
        classification.source = 'external';
        break;

      case 'ValidationError':
        classification.category = 'validation';
        classification.severity = 'low';
        classification.userFacing = true;
        break;

      case 'AuthenticationError':
      case 'AuthorizationError':
        classification.category = 'security';
        classification.severity = 'high';
        classification.userFacing = true;
        break;
    }

    // 메시지 기반 분류
    const message = error.message.toLowerCase();
    
    if (message.includes('python') || message.includes('pip')) {
      classification.category = 'python';
      classification.source = 'python';
    }

    if (message.includes('memory') || message.includes('ram')) {
      classification.category = 'resource';
      classification.severity = 'high';
      classification.source = 'system';
    }

    if (message.includes('timeout')) {
      classification.category = 'timeout';
      classification.severity = 'medium';
      classification.recoverable = true;
    }

    if (message.includes('model') || message.includes('ollama')) {
      classification.category = 'ml_model';
      classification.source = 'external';
    }

    return classification;
  }

  enrichContext(context, error) {
    return {
      ...context,
      timestamp: new Date().toISOString(),
      process: {
        pid: process.pid,
        memory: process.memoryUsage(),
        uptime: process.uptime(),
        cwd: process.cwd()
      },
      environment: {
        node_version: process.version,
        platform: process.platform,
        arch: process.arch
      },
      error_id: this.generateErrorId(error)
    };
  }

  generateErrorId(error) {
    const content = `${error.name}-${error.message}-${Date.now()}`;
    return Buffer.from(content).toString('base64').substring(0, 16);
  }

  logError(error, classification, context) {
    const logData = {
      error_id: context.error_id,
      error: {
        name: error.name,
        message: error.message,
        stack: error.stack
      },
      classification,
      context: {
        ...context,
        process: undefined // 로그에서는 프로세스 정보 제외
      }
    };

    switch (classification.severity) {
      case 'low':
        this.logger.warn('Low severity error:', logData);
        break;
      case 'medium':
        this.logger.error('Medium severity error:', logData);
        break;
      case 'high':
        this.logger.error('High severity error:', logData);
        break;
      case 'critical':
        this.logger.error('CRITICAL ERROR:', logData);
        break;
      default:
        this.logger.error('Unknown severity error:', logData);
    }
  }

  trackError(error, classification) {
    // 에러 히스토리에 추가
    const errorRecord = {
      timestamp: new Date(),
      name: error.name,
      message: error.message,
      classification,
      id: this.generateErrorId(error)
    };

    this.errorHistory.unshift(errorRecord);

    // 히스토리 크기 제한
    if (this.errorHistory.length > this.maxHistorySize) {
      this.errorHistory = this.errorHistory.slice(0, this.maxHistorySize);
    }

    // 에러 카운트 추적
    const errorKey = `${error.name}:${error.message}`;
    const now = Date.now();
    
    if (!this.errorCounts.has(errorKey)) {
      this.errorCounts.set(errorKey, []);
    }
    
    const errorTimes = this.errorCounts.get(errorKey);
    errorTimes.push(now);
    
    // 시간 윈도우 밖의 에러는 제거
    const cutoff = now - this.alertThresholds.timeWindow;
    this.errorCounts.set(errorKey, errorTimes.filter(time => time > cutoff));
  }

  determineResponse(error, classification, context) {
    const response = {
      handled: true,
      action: 'log_only',
      message: '에러가 처리되었습니다.',
      shouldRetry: false,
      shouldTerminate: false,
      userMessage: null,
      suggestedActions: []
    };

    // 심각도별 대응
    switch (classification.severity) {
      case 'critical':
        response.action = 'terminate';
        response.shouldTerminate = true;
        response.message = '심각한 에러로 인해 시스템을 종료합니다.';
        break;

      case 'high':
        if (!classification.recoverable) {
          response.action = 'restart_component';
          response.message = '복구 불가능한 에러가 발생했습니다.';
        } else {
          response.action = 'retry_with_fallback';
          response.shouldRetry = true;
          response.message = '에러 복구를 시도합니다.';
        }
        break;

      case 'medium':
        response.action = 'retry';
        response.shouldRetry = classification.recoverable;
        response.message = '일시적인 에러가 발생했습니다.';
        break;

      case 'low':
        response.action = 'log_only';
        response.message = '경미한 에러가 기록되었습니다.';
        break;
    }

    // 카테고리별 특별 처리
    switch (classification.category) {
      case 'network':
        response.shouldRetry = true;
        response.suggestedActions.push('네트워크 연결 확인');
        response.userMessage = '네트워크 연결에 문제가 있습니다. 잠시 후 다시 시도해주세요.';
        break;

      case 'filesystem':
        response.suggestedActions.push('파일 권한 확인', '디스크 공간 확인');
        response.userMessage = '파일 시스템 오류가 발생했습니다.';
        break;

      case 'python':
        response.suggestedActions.push('Python 환경 확인', '패키지 재설치');
        response.userMessage = 'Python 실행 중 오류가 발생했습니다.';
        break;

      case 'ml_model':
        response.suggestedActions.push('모델 서비스 상태 확인', 'Ollama 재시작');
        response.userMessage = 'AI 모델 서비스에 문제가 있습니다.';
        break;

      case 'validation':
        response.userMessage = '입력 데이터에 문제가 있습니다. 데이터를 확인해주세요.';
        break;

      case 'security':
        response.action = 'security_alert';
        response.userMessage = '보안 관련 문제가 발생했습니다.';
        break;
    }

    // 사용자 대면 에러 처리
    if (classification.userFacing && !response.userMessage) {
      response.userMessage = this.generateUserFriendlyMessage(error, classification);
    }

    return response;
  }

  generateUserFriendlyMessage(error, classification) {
    const messages = {
      network: '네트워크 연결에 문제가 있습니다. 잠시 후 다시 시도해주세요.',
      filesystem: '파일 접근에 문제가 있습니다. 파일 경로와 권한을 확인해주세요.',
      python: 'Python 처리 중 오류가 발생했습니다. 데이터 형식을 확인해주세요.',
      validation: '입력된 데이터에 문제가 있습니다. 데이터를 다시 확인해주세요.',
      ml_model: 'AI 모델 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.',
      timeout: '처리 시간이 초과되었습니다. 더 작은 데이터로 시도해주세요.',
      resource: '시스템 리소스가 부족합니다. 잠시 후 다시 시도해주세요.',
      security: '보안 정책에 의해 요청이 차단되었습니다.',
      programming: '시스템 내부 오류가 발생했습니다. 관리자에게 문의해주세요.'
    };

    return messages[classification.category] || '알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해주세요.';
  }

  checkAlertThresholds(error) {
    const errorKey = `${error.name}:${error.message}`;
    const errorTimes = this.errorCounts.get(errorKey) || [];
    
    if (errorTimes.length >= this.alertThresholds.critical) {
      this.triggerAlert('critical', error, errorTimes.length);
    } else if (errorTimes.length >= this.alertThresholds.warning) {
      this.triggerAlert('warning', error, errorTimes.length);
    }
  }

  triggerAlert(level, error, count) {
    const alert = {
      level,
      error: {
        name: error.name,
        message: error.message
      },
      count,
      timeWindow: this.alertThresholds.timeWindow / 1000 / 60, // 분 단위
      timestamp: new Date().toISOString()
    };

    this.logger.error(`ALERT [${level.toUpperCase()}]: 반복적인 에러 감지`, alert);

    // 실제 구현에서는 여기에 알림 시스템 (이메일, Slack 등) 연동
    if (level === 'critical') {
      console.error(`🚨 CRITICAL ALERT: ${error.name} 에러가 ${count}회 반복 발생`);
    } else {
      console.warn(`⚠️  WARNING: ${error.name} 에러가 ${count}회 발생`);
    }
  }

  handleUnhandledRejection(reason, promise) {
    const error = {
      name: 'UnhandledPromiseRejection',
      message: reason ? reason.toString() : 'Unknown rejection reason',
      stack: reason && reason.stack ? reason.stack : new Error().stack,
      promise: promise
    };

    const response = this.handleError(error, {
      source: 'unhandled_rejection',
      critical: true
    });

    if (response.shouldTerminate) {
      console.error('처리되지 않은 Promise 거부로 인해 프로세스를 종료합니다.');
      process.exit(1);
    }
  }

  handleUncaughtException(error) {
    const response = this.handleError(error, {
      source: 'uncaught_exception',
      critical: true
    });

    // 처리되지 않은 예외는 항상 프로세스 종료
    console.error('처리되지 않은 예외로 인해 프로세스를 종료합니다.');
    
    // 정리 작업 수행
    this.performCleanup();
    
    process.exit(1);
  }

  handleWarning(warning) {
    this.logger.warn('Node.js 경고:', {
      name: warning.name,
      message: warning.message,
      stack: warning.stack
    });
  }

  performCleanup() {
    try {
      // 임시 파일 정리, 연결 종료 등
      this.logger.info('시스템 정리 작업 수행 중...');
      
      // 에러 히스토리 저장 (구현 필요)
      this.saveErrorHistory();
      
    } catch (cleanupError) {
      console.error('정리 작업 중 오류:', cleanupError);
    }
  }

  async saveErrorHistory() {
    try {
      // 에러 히스토리를 파일로 저장 (구현 예시)
      const fs = await import('fs/promises');
      const path = await import('path');
      
      const historyFile = path.join('./logs', 'error-history.json');
      const historyData = {
        timestamp: new Date().toISOString(),
        errors: this.errorHistory.slice(0, 100), // 최근 100개만 저장
        summary: this.getErrorSummary()
      };
      
      await fs.writeFile(historyFile, JSON.stringify(historyData, null, 2));
      
    } catch (error) {
      console.error('에러 히스토리 저장 실패:', error);
    }
  }

  getErrorSummary() {
    const summary = {
      total_errors: this.errorHistory.length,
      error_types: {},
      recent_errors: this.errorHistory.slice(0, 10)
    };

    // 에러 타입별 집계
    for (const error of this.errorHistory) {
      const key = error.name;
      summary.error_types[key] = (summary.error_types[key] || 0) + 1;
    }

    return summary;
  }

  getErrorStats() {
    const now = Date.now();
    const oneHour = 60 * 60 * 1000;
    const oneDay = 24 * oneHour;

    const recentErrors = this.errorHistory.filter(
      error => (now - error.timestamp.getTime()) < oneHour
    );

    const todayErrors = this.errorHistory.filter(
      error => (now - error.timestamp.getTime()) < oneDay
    );

    return {
      total_errors: this.errorHistory.length,
      errors_last_hour: recentErrors.length,
      errors_today: todayErrors.length,
      most_common_errors: this.getMostCommonErrors(),
      active_alerts: this.getActiveAlerts(),
      error_rate_trend: this.calculateErrorRateTrend()
    };
  }

  getMostCommonErrors() {
    const errorCounts = {};
    
    for (const error of this.errorHistory) {
      const key = `${error.name}: ${error.message}`;
      errorCounts[key] = (errorCounts[key] || 0) + 1;
    }

    return Object.entries(errorCounts)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 5)
      .map(([error, count]) => ({ error, count }));
  }

  getActiveAlerts() {
    const now = Date.now();
    const alerts = [];

    for (const [errorKey, times] of this.errorCounts) {
      const recentTimes = times.filter(
        time => (now - time) < this.alertThresholds.timeWindow
      );

      if (recentTimes.length >= this.alertThresholds.warning) {
        alerts.push({
          error: errorKey,
          count: recentTimes.length,
          level: recentTimes.length >= this.alertThresholds.critical ? 'critical' : 'warning'
        });
      }
    }

    return alerts;
  }

  calculateErrorRateTrend() {
    const now = Date.now();
    const intervals = [
      { name: '지난_시간', duration: 60 * 60 * 1000 },
      { name: '지난_4시간', duration: 4 * 60 * 60 * 1000 },
      { name: '지난_24시간', duration: 24 * 60 * 60 * 1000 }
    ];

    const trend = {};

    for (const interval of intervals) {
      const cutoff = now - interval.duration;
      const errorCount = this.errorHistory.filter(
        error => error.timestamp.getTime() > cutoff
      ).length;

      trend[interval.name] = {
        count: errorCount,
        rate: errorCount / (interval.duration / (60 * 60 * 1000)) // 시간당 에러 수
      };
    }

    return trend;
  }

  clearErrorHistory() {
    this.errorHistory = [];
    this.errorCounts.clear();
    this.logger.info('에러 히스토리가 초기화되었습니다.');
  }

  // 특정 에러 타입 무시 설정
  setIgnorePattern(pattern) {
    if (!this.ignorePatterns) {
      this.ignorePatterns = [];
    }
    this.ignorePatterns.push(pattern);
    this.logger.info(`에러 무시 패턴 추가: ${pattern}`);
  }

  shouldIgnoreError(error) {
    if (!this.ignorePatterns) return false;

    const errorString = `${error.name}: ${error.message}`;
    return this.ignorePatterns.some(pattern => {
      if (typeof pattern === 'string') {
        return errorString.includes(pattern);
      }
      if (pattern instanceof RegExp) {
        return pattern.test(errorString);
      }
      return false;
    });
  }

  // 커스텀 에러 핸들러 등록
  registerCustomHandler(errorType, handler) {
    if (!this.customHandlers) {
      this.customHandlers = new Map();
    }
    this.customHandlers.set(errorType, handler);
    this.logger.info(`커스텀 에러 핸들러 등록: ${errorType}`);
  }

  async executeCustomHandler(error, classification, context) {
    if (!this.customHandlers) return null;

    const handler = this.customHandlers.get(error.name) || 
                   this.customHandlers.get(classification.category);

    if (handler && typeof handler === 'function') {
      try {
        return await handler(error, classification, context);
      } catch (handlerError) {
        this.logger.error('커스텀 핸들러 실행 실패:', handlerError);
        return null;
      }
    }

    return null;
  }

  // 에러 복구 시도
  async attemptRecovery(error, classification, context) {
    const recoveryStrategies = {
      network: async () => {
        // 네트워크 재연결 시도
        await this.wait(1000);
        return { success: true, message: '네트워크 재연결 시도' };
      },
      
      python: async () => {
        // Python 프로세스 재시작
        return { success: false, message: 'Python 자동 복구 미구현' };
      },
      
      ml_model: async () => {
        // 모델 서비스 재시작
        return { success: false, message: '모델 서비스 자동 복구 미구현' };
      },

      filesystem: async () => {
        // 디렉토리 생성 시도
        try {
          const fs = await import('fs/promises');
          if (error.code === 'ENOENT' && error.path) {
            const path = await import('path');
            await fs.mkdir(path.dirname(error.path), { recursive: true });
            return { success: true, message: '디렉토리 생성 완료' };
          }
        } catch (recoveryError) {
          return { success: false, message: `복구 실패: ${recoveryError.message}` };
        }
        return { success: false, message: '복구 방법을 찾을 수 없음' };
      }
    };

    const strategy = recoveryStrategies[classification.category];
    if (strategy) {
      this.logger.info(`에러 복구 시도: ${classification.category}`);
      return await strategy();
    }

    return { success: false, message: '복구 전략 없음' };
  }

  wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // 에러 리포트 생성
  generateErrorReport(timeRange = '24h') {
    const now = Date.now();
    const timeRanges = {
      '1h': 60 * 60 * 1000,
      '24h': 24 * 60 * 60 * 1000,
      '7d': 7 * 24 * 60 * 60 * 1000,
      '30d': 30 * 24 * 60 * 60 * 1000
    };

    const duration = timeRanges[timeRange] || timeRanges['24h'];
    const cutoff = now - duration;

    const relevantErrors = this.errorHistory.filter(
      error => error.timestamp.getTime() > cutoff
    );

    const report = {
      timeRange,
      period: {
        start: new Date(cutoff).toISOString(),
        end: new Date(now).toISOString()
      },
      summary: {
        totalErrors: relevantErrors.length,
        uniqueErrors: new Set(relevantErrors.map(e => e.name)).size,
        severity: this.calculateSeverityDistribution(relevantErrors),
        categories: this.calculateCategoryDistribution(relevantErrors)
      },
      topErrors: this.getTopErrorsInPeriod(relevantErrors),
      timeline: this.generateErrorTimeline(relevantErrors),
      recommendations: this.generateRecommendations(relevantErrors)
    };

    return report;
  }

  calculateSeverityDistribution(errors) {
    const distribution = { low: 0, medium: 0, high: 0, critical: 0 };
    
    for (const error of errors) {
      const severity = error.classification?.severity || 'medium';
      distribution[severity] = (distribution[severity] || 0) + 1;
    }

    return distribution;
  }

  calculateCategoryDistribution(errors) {
    const distribution = {};
    
    for (const error of errors) {
      const category = error.classification?.category || 'unknown';
      distribution[category] = (distribution[category] || 0) + 1;
    }

    return distribution;
  }

  getTopErrorsInPeriod(errors) {
    const errorCounts = {};
    
    for (const error of errors) {
      const key = `${error.name}: ${error.message}`;
      if (!errorCounts[key]) {
        errorCounts[key] = {
          name: error.name,
          message: error.message,
          count: 0,
          classification: error.classification
        };
      }
      errorCounts[key].count++;
    }

    return Object.values(errorCounts)
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);
  }

  generateErrorTimeline(errors) {
    const timeline = {};
    
    for (const error of errors) {
      const hour = new Date(error.timestamp).toISOString().substring(0, 13);
      timeline[hour] = (timeline[hour] || 0) + 1;
    }

    return timeline;
  }

  generateRecommendations(errors) {
    const recommendations = [];
    const categories = this.calculateCategoryDistribution(errors);

    if (categories.network > 5) {
      recommendations.push('네트워크 연결 안정성을 확인해보세요.');
    }

    if (categories.python > 3) {
      recommendations.push('Python 환경과 패키지 설정을 점검해보세요.');
    }

    if (categories.resource > 2) {
      recommendations.push('시스템 리소스 사용량을 모니터링하고 최적화해보세요.');
    }

    if (errors.length > 50) {
      recommendations.push('에러 발생률이 높습니다. 시스템 전반적인 점검이 필요합니다.');
    }

    return recommendations;
  }
}