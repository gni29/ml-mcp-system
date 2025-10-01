// utils/logger.js
import winston from 'winston';
import fs from 'fs/promises';
import path from 'path';

export class Logger {
  constructor(options = {}) {
    this.logLevel = options.level || process.env.LOG_LEVEL || 'info';
    this.logDir = options.logDir || './data/logs';
    this.maxFileSize = options.maxFileSize || 10 * 1024 * 1024; // 10MB
    this.maxFiles = options.maxFiles || 5;
    this.enableConsole = options.enableConsole !== false;
    
    this.setupLogger();
    this.createLogDirectories();
  }

  async createLogDirectories() {
    try {
      await fs.mkdir(this.logDir, { recursive: true });
    } catch (error) {
      console.error('로그 디렉토리 생성 실패:', error);
    }
  }

  setupLogger() {
    // 커스텀 포맷 생성
    const customFormat = winston.format.combine(
      winston.format.timestamp({
        format: 'YYYY-MM-DD HH:mm:ss'
      }),
      winston.format.errors({ stack: true }),
      winston.format.printf(({ timestamp, level, message, stack, ...meta }) => {
        let logMessage = `${timestamp} [${level.toUpperCase()}]: ${message}`;
        
        // 메타데이터가 있으면 추가
        if (Object.keys(meta).length > 0) {
          logMessage += ` ${JSON.stringify(meta)}`;
        }
        
        // 스택 트레이스가 있으면 추가
        if (stack) {
          logMessage += `\nStack: ${stack}`;
        }
        
        return logMessage;
      })
    );

    // 콘솔용 컬러 포맷
    const consoleFormat = winston.format.combine(
      winston.format.colorize(),
      winston.format.timestamp({
        format: 'HH:mm:ss'
      }),
      winston.format.printf(({ timestamp, level, message, ...meta }) => {
        let logMessage = `${timestamp} ${level}: ${message}`;
        
        // 메타데이터가 있으면 간단히 표시
        if (Object.keys(meta).length > 0) {
          const metaKeys = Object.keys(meta);
          if (metaKeys.length === 1 && typeof meta[metaKeys[0]] === 'string') {
            logMessage += ` (${meta[metaKeys[0]]})`;
          } else {
            logMessage += ` ${JSON.stringify(meta, null, 0)}`;
          }
        }
        
        return logMessage;
      })
    );

    // 전송자 설정
    const transports = [];

    // 콘솔 전송자
    if (this.enableConsole) {
      transports.push(
        new winston.transports.Console({
          level: this.logLevel,
          format: consoleFormat,
          handleExceptions: true,
          handleRejections: true
        })
      );
    }

    // 파일 전송자들
    transports.push(
      // 에러 전용 로그
      new winston.transports.File({
        filename: path.join(this.logDir, 'error.log'),
        level: 'error',
        format: customFormat,
        maxsize: this.maxFileSize,
        maxFiles: this.maxFiles,
        tailable: true
      }),
      
      // 모든 로그
      new winston.transports.File({
        filename: path.join(this.logDir, 'combined.log'),
        level: this.logLevel,
        format: customFormat,
        maxsize: this.maxFileSize,
        maxFiles: this.maxFiles,
        tailable: true
      }),
      
      // 디버그 전용 로그 (debug 레벨일 때만)
      new winston.transports.File({
        filename: path.join(this.logDir, 'debug.log'),
        level: 'debug',
        format: customFormat,
        maxsize: this.maxFileSize,
        maxFiles: 3,
        tailable: true
      })
    );

    // 로거 생성
    this.logger = winston.createLogger({
      level: this.logLevel,
      format: customFormat,
      transports: transports,
      exitOnError: false,
      
      // 예외 처리
      exceptionHandlers: [
        new winston.transports.File({
          filename: path.join(this.logDir, 'exceptions.log'),
          format: customFormat
        })
      ],
      
      // Promise rejection 처리
      rejectionHandlers: [
        new winston.transports.File({
          filename: path.join(this.logDir, 'rejections.log'),
          format: customFormat
        })
      ]
    });

    // 개발 환경에서 더 자세한 로깅
    if (process.env.NODE_ENV === 'development') {
      this.logger.level = 'debug';
    }
  }

  // 기본 로깅 메서드들
  info(message, meta = {}) {
    this.logger.info(message, this.sanitizeMeta(meta));
  }

  error(message, meta = {}) {
    // 에러 객체인 경우 특별 처리
    if (message instanceof Error) {
      const error = message;
      this.logger.error(error.message, {
        ...this.sanitizeMeta(meta),
        stack: error.stack,
        errorName: error.name,
        errorCode: error.code
      });
    } else {
      this.logger.error(message, this.sanitizeMeta(meta));
    }
  }

  warn(message, meta = {}) {
    this.logger.warn(message, this.sanitizeMeta(meta));
  }

  debug(message, meta = {}) {
    this.logger.debug(message, this.sanitizeMeta(meta));
  }

  verbose(message, meta = {}) {
    this.logger.verbose(message, this.sanitizeMeta(meta));
  }

  silly(message, meta = {}) {
    this.logger.silly(message, this.sanitizeMeta(meta));
  }

  // 특별한 로깅 메서드들
  api(method, url, status, responseTime, meta = {}) {
    this.info(`API ${method} ${url}`, {
      method,
      url,
      status,
      responseTime: `${responseTime}ms`,
      ...meta
    });
  }

  performance(operation, startTime, meta = {}) {
    const duration = Date.now() - startTime;
    this.info(`Performance: ${operation}`, {
      operation,
      duration: `${duration}ms`,
      ...meta
    });
  }

  memory(label = 'Memory Usage') {
    if (process.memoryUsage) {
      const usage = process.memoryUsage();
      this.debug(label, {
        rss: `${Math.round(usage.rss / 1024 / 1024)}MB`,
        heapTotal: `${Math.round(usage.heapTotal / 1024 / 1024)}MB`,
        heapUsed: `${Math.round(usage.heapUsed / 1024 / 1024)}MB`,
        external: `${Math.round(usage.external / 1024 / 1024)}MB`
      });
    }
  }

  system(event, details = {}) {
    this.info(`System: ${event}`, {
      event,
      timestamp: new Date().toISOString(),
      ...details
    });
  }

  security(event, details = {}) {
    this.warn(`Security: ${event}`, {
      event,
      severity: 'medium',
      timestamp: new Date().toISOString(),
      ...details
    });
  }

  model(modelName, action, details = {}) {
    this.info(`Model [${modelName}]: ${action}`, {
      modelName,
      action,
      ...details
    });
  }

  pipeline(pipelineName, step, details = {}) {
    this.info(`Pipeline [${pipelineName}]: ${step}`, {
      pipelineName,
      step,
      ...details
    });
  }

  workflow(workflowId, status, details = {}) {
    this.info(`Workflow [${workflowId}]: ${status}`, {
      workflowId,
      status,
      ...details
    });
  }

  user(userId, action, details = {}) {
    this.info(`User [${userId}]: ${action}`, {
      userId,
      action,
      privacy: 'masked',
      ...details
    });
  }

  // 메타데이터 정리 및 보안
  sanitizeMeta(meta) {
    if (!meta || typeof meta !== 'object') {
      return {};
    }

    const sanitized = { ...meta };
    
    // 민감한 정보 마스킹
    const sensitiveKeys = ['password', 'token', 'key', 'secret', 'auth', 'credential'];
    
    Object.keys(sanitized).forEach(key => {
      const lowerKey = key.toLowerCase();
      
      if (sensitiveKeys.some(sensitive => lowerKey.includes(sensitive))) {
        sanitized[key] = '[MASKED]';
      }
      
      // 순환 참조 방지
      if (typeof sanitized[key] === 'object' && sanitized[key] !== null) {
        try {
          JSON.stringify(sanitized[key]);
        } catch (error) {
          sanitized[key] = '[Circular Reference]';
        }
      }
    });

    return sanitized;
  }

  // 로그 레벨 동적 변경
  setLevel(level) {
    const validLevels = ['error', 'warn', 'info', 'verbose', 'debug', 'silly'];
    
    if (validLevels.includes(level)) {
      this.logger.level = level;
      this.logLevel = level;
      this.info(`로그 레벨 변경: ${level}`);
    } else {
      this.warn(`유효하지 않은 로그 레벨: ${level}. 사용 가능: ${validLevels.join(', ')}`);
    }
  }

  getLevel() {
    return this.logger.level;
  }

  // 로그 스트림 제어
  enableConsoleLogging() {
    this.enableConsole = true;
    this.setupLogger();
    this.info('콘솔 로깅 활성화');
  }

  disableConsoleLogging() {
    this.enableConsole = false;
    this.setupLogger();
    // 파일에만 로그 (콘솔 비활성화되었으므로 직접 파일에 씀)
    this.logger.info('콘솔 로깅 비활성화');
  }

  // 로그 파일 관리
  async getLogFiles() {
    try {
      const files = await fs.readdir(this.logDir);
      const logFiles = [];
      
      for (const file of files) {
        if (file.endsWith('.log')) {
          const filePath = path.join(this.logDir, file);
          const stats = await fs.stat(filePath);
          
          logFiles.push({
            name: file,
            path: filePath,
            size: stats.size,
            modified: stats.mtime,
            sizeFormatted: this.formatFileSize(stats.size)
          });
        }
      }
      
      return logFiles.sort((a, b) => b.modified - a.modified);
    } catch (error) {
      this.error('로그 파일 목록 조회 실패:', error);
      return [];
    }
  }

  async readLogFile(filename, lines = 100) {
    try {
      const filePath = path.join(this.logDir, filename);
      const content = await fs.readFile(filePath, 'utf-8');
      const allLines = content.split('\n');
      
      // 마지막 N줄 반환
      const lastLines = allLines.slice(-lines).filter(line => line.trim());
      
      return {
        filename,
        totalLines: allLines.length,
        returnedLines: lastLines.length,
        content: lastLines
      };
    } catch (error) {
      this.error(`로그 파일 읽기 실패 [${filename}]:`, error);
      throw error;
    }
  }

  async clearLogFile(filename) {
    try {
      const filePath = path.join(this.logDir, filename);
      await fs.writeFile(filePath, '');
      this.info(`로그 파일 초기화: ${filename}`);
      return true;
    } catch (error) {
      this.error(`로그 파일 초기화 실패 [${filename}]:`, error);
      return false;
    }
  }

  async deleteOldLogs(daysOld = 7) {
    try {
      const files = await this.getLogFiles();
      const cutoffDate = new Date();
      cutoffDate.setDate(cutoffDate.getDate() - daysOld);
      
      let deletedCount = 0;
      
      for (const file of files) {
        if (file.modified < cutoffDate) {
          await fs.unlink(file.path);
          deletedCount++;
          this.info(`오래된 로그 파일 삭제: ${file.name}`);
        }
      }
      
      if (deletedCount > 0) {
        this.info(`총 ${deletedCount}개의 오래된 로그 파일을 삭제했습니다.`);
      }
      
      return deletedCount;
    } catch (error) {
      this.error('오래된 로그 파일 삭제 실패:', error);
      return 0;
    }
  }

  // 로그 통계
  async getLogStats() {
    try {
      const files = await this.getLogFiles();
      let totalSize = 0;
      let totalLines = 0;
      
      for (const file of files) {
        totalSize += file.size;
        
        // 라인 수 계산 (샘플링)
        try {
          const content = await fs.readFile(file.path, 'utf-8');
          totalLines += content.split('\n').length;
        } catch (error) {
          // 파일 읽기 실패 시 무시
        }
      }
      
      return {
        fileCount: files.length,
        totalSize,
        totalSizeFormatted: this.formatFileSize(totalSize),
        totalLines,
        files: files.map(f => ({
          name: f.name,
          size: f.sizeFormatted,
          modified: f.modified.toISOString()
        }))
      };
    } catch (error) {
      this.error('로그 통계 생성 실패:', error);
      return {
        fileCount: 0,
        totalSize: 0,
        totalSizeFormatted: '0 Bytes',
        totalLines: 0,
        files: []
      };
    }
  }

  // 실시간 로그 모니터링
  createLogStream(filename = 'combined.log') {
    const filePath = path.join(this.logDir, filename);
    
    return new Promise((resolve, reject) => {
      import('fs').then(({ createReadStream }) => {
        const stream = createReadStream(filePath, {
          encoding: 'utf-8',
          flags: 'r'
        });
        
        stream.on('error', reject);
        stream.on('ready', () => resolve(stream));
      });
    });
  }

  // 구조화된 로깅을 위한 헬퍼
  logStructured(level, event, data = {}) {
    const structuredLog = {
      event,
      timestamp: new Date().toISOString(),
      data: this.sanitizeMeta(data)
    };
    
    this[level](`[${event}]`, structuredLog);
  }

  // 성능 측정 도구
  createTimer(label) {
    const startTime = Date.now();
    
    return {
      end: (meta = {}) => {
        const duration = Date.now() - startTime;
        this.performance(label, startTime, {
          duration: `${duration}ms`,
          ...meta
        });
        return duration;
      }
    };
  }

  // 배치 로깅 (여러 로그를 한번에)
  batch(logs) {
    if (!Array.isArray(logs)) {
      this.warn('배치 로그는 배열이어야 합니다.');
      return;
    }

    logs.forEach(log => {
      const { level, message, meta } = log;
      if (this[level] && typeof this[level] === 'function') {
        this[level](message, meta);
      } else {
        this.info(message, meta);
      }
    });
  }

  // 조건부 로깅
  logIf(condition, level, message, meta = {}) {
    if (condition) {
      this[level](message, meta);
    }
  }

  // 주기적 로깅 (디버깅용)
  startPeriodicLog(interval = 60000, label = 'Periodic Check') {
    const timer = setInterval(() => {
      this.memory(`${label} - Memory`);
      this.debug(`${label} - Heartbeat`);
    }, interval);

    return {
      stop: () => {
        clearInterval(timer);
        this.debug(`주기적 로깅 중지: ${label}`);
      }
    };
  }

  // 유틸리티 메서드들
  formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  // 로거 종료 (graceful shutdown)
  async close() {
    return new Promise((resolve) => {
      this.info('로거 종료 중...');
      
      this.logger.on('close', () => {
        resolve();
      });
      
      this.logger.close();
    });
  }

  // 헬스체크
  async healthCheck() {
    try {
      // 로그 디렉토리 접근 확인
      await fs.access(this.logDir);
      
      // 테스트 로그 작성
      const testMessage = `Health check at ${new Date().toISOString()}`;
      this.debug(testMessage);
      
      // 로그 파일 상태 확인
      const stats = await this.getLogStats();
      
      return {
        status: 'healthy',
        logDir: this.logDir,
        level: this.logLevel,
        enableConsole: this.enableConsole,
        stats
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        error: error.message,
        logDir: this.logDir
      };
    }
  }
}

// 기본 인스턴스 생성 및 export
export const logger = new Logger();

// 전역 에러 핸들러 설정
if (typeof process !== 'undefined') {
  process.on('uncaughtException', (error) => {
    logger.error('Uncaught Exception:', {
      message: error.message,
      stack: error.stack,
      fatal: true
    });
    
    // 1초 후 프로세스 종료 (로그 flush 대기)
    setTimeout(() => {
      process.exit(1);
    }, 1000);
  });

  process.on('unhandledRejection', (reason, promise) => {
    logger.error('Unhandled Promise Rejection:', {
      reason: reason?.message || reason,
      stack: reason?.stack,
      promise: promise.toString()
    });
  });

  process.on('SIGTERM', async () => {
    logger.info('SIGTERM 수신, 로거 종료 중...');
    await logger.close();
    process.exit(0);
  });

  process.on('SIGINT', async () => {
    logger.info('SIGINT 수신, 로거 종료 중...');
    await logger.close();
    process.exit(0);
  });
}

export default Logger;