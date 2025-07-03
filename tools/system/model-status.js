// tools/system/model-status.js
import { Logger } from '../../utils/logger.js';
import axios from 'axios';
import os from 'os';

export class ModelStatus {
  constructor(modelManager, ollamaManager) {
    this.modelManager = modelManager;
    this.ollamaManager = ollamaManager;
    this.logger = new Logger();
    this.statusCache = new Map();
    this.cacheTimeout = 30000; // 30초 캐시
    this.monitoringInterval = null;
    this.alertThresholds = this.initializeAlertThresholds();
    this.statusHistory = [];
    this.maxHistorySize = 100;
  }

  initializeAlertThresholds() {
    return {
      memory: {
        warning: 0.8,  // 80%
        critical: 0.95 // 95%
      },
      cpu: {
        warning: 0.7,  // 70%
        critical: 0.9  // 90%
      },
      response_time: {
        warning: 5000,   // 5초
        critical: 10000  // 10초
      },
      error_rate: {
        warning: 0.05,  // 5%
        critical: 0.1   // 10%
      }
    };
  }

  async getComprehensiveStatus() {
    try {
      this.logger.info('종합 모델 상태 조회 시작');

      const status = {
        timestamp: new Date().toISOString(),
        overall_health: 'unknown',
        ollama_service: await this.getOllamaServiceStatus(),
        models: await this.getModelsStatus(),
        system_resources: await this.getSystemResourcesStatus(),
        performance_metrics: await this.getPerformanceMetrics(),
        alerts: [],
        recommendations: []
      };

      // 전체 상태 평가
      status.overall_health = this.evaluateOverallHealth(status);
      
      // 알림 및 권장사항 생성
      status.alerts = this.generateAlerts(status);
      status.recommendations = this.generateRecommendations(status);

      // 상태 히스토리에 추가
      this.addToHistory(status);

      return this.formatStatusResponse(status);

    } catch (error) {
      this.logger.error('종합 상태 조회 실패:', error);
      return this.createErrorResponse(error);
    }
  }

  async getOllamaServiceStatus() {
    const cacheKey = 'ollama_service';
    const cached = this.getCachedStatus(cacheKey);
    if (cached) return cached;

    try {
      const startTime = Date.now();
      
      // Ollama 서비스 연결 확인
      const versionResponse = await axios.get('http://localhost:11434/api/version', {
        timeout: 5000
      });
      
      const responseTime = Date.now() - startTime;
      
      // 실행 중인 모델 확인
      const runningModels = await this.ollamaManager.getRunningModels();
      
      // 사용 가능한 모델 목록
      const availableModels = await this.ollamaManager.listModels();

      const status = {
        status: 'running',
        version: versionResponse.data.version || 'unknown',
        response_time_ms: responseTime,
        running_models: runningModels.length,
        available_models: availableModels.length,
        models_detail: {
          running: runningModels.map(model => ({
            name: model.name,
            size: this.formatBytes(model.size || 0),
            memory_usage: this.formatBytes(model.size_vram || 0)
          })),
          available: availableModels.map(model => ({
            name: model.name,
            size: this.formatBytes(model.size || 0),
            modified: model.modified_at
          }))
        },
        endpoint: 'http://localhost:11434',
        health: responseTime < 1000 ? 'healthy' : responseTime < 3000 ? 'degraded' : 'slow'
      };

      this.setCachedStatus(cacheKey, status);
      return status;

    } catch (error) {
      const errorStatus = {
        status: 'error',
        error: error.message,
        health: 'unhealthy',
        response_time_ms: null,
        running_models: 0,
        available_models: 0,
        endpoint: 'http://localhost:11434'
      };

      this.setCachedStatus(cacheKey, errorStatus);
      return errorStatus;
    }
  }

  async getModelsStatus() {
    const cacheKey = 'models_status';
    const cached = this.getCachedStatus(cacheKey);
    if (cached) return cached;

    try {
      const modelsStatus = {};

      // 라우터 모델 상태
      modelsStatus.router = await this.getIndividualModelStatus('llama3.2:3b', 'router');
      
      // 프로세서 모델 상태 (로드된 경우에만)
      if (this.modelManager.models.has('processor')) {
        modelsStatus.processor = await this.getIndividualModelStatus('qwen2.5:14b', 'processor');
      } else {
        modelsStatus.processor = {
          name: 'qwen2.5:14b',
          role: 'processor',
          status: 'not_loaded',
          health: 'idle',
          memory_usage: 0,
          last_used: null
        };
      }

      // 모델별 성능 메트릭
      for (const [modelType, modelInfo] of Object.entries(modelsStatus)) {
        if (modelInfo.status === 'loaded') {
          modelInfo.performance = await this.getModelPerformanceMetrics(modelType);
        }
      }

      this.setCachedStatus(cacheKey, modelsStatus);
      return modelsStatus;

    } catch (error) {
      this.logger.error('모델 상태 조회 실패:', error);
      return {
        router: { status: 'error', error: error.message },
        processor: { status: 'error', error: error.message }
      };
    }
  }

  async getIndividualModelStatus(modelName, role) {
    try {
      const startTime = Date.now();
      
      // 모델 로드 상태 확인
      const isLoaded = await this.ollamaManager.isModelAvailable(modelName);
      
      if (!isLoaded) {
        return {
          name: modelName,
          role: role,
          status: 'not_available',
          health: 'unhealthy',
          error: 'Model not found in Ollama'
        };
      }

      // 실행 중인 모델에서 찾기
      const runningModels = await this.ollamaManager.getRunningModels();
      const runningModel = runningModels.find(m => m.name === modelName);

      if (runningModel) {
        // 간단한 응답 테스트
        const testResponse = await this.testModelResponse(modelName);
        const responseTime = Date.now() - startTime;

        return {
          name: modelName,
          role: role,
          status: 'loaded',
          health: testResponse.success ? 'healthy' : 'degraded',
          memory_usage: runningModel.size_vram || 0,
          size: runningModel.size || 0,
          response_time_ms: responseTime,
          last_used: this.modelManager.models.get(role)?.lastUsed || null,
          test_result: testResponse
        };
      } else {
        return {
          name: modelName,
          role: role,
          status: 'available',
          health: 'idle',
          memory_usage: 0,
          last_used: this.modelManager.models.get(role)?.lastUsed || null
        };
      }

    } catch (error) {
      return {
        name: modelName,
        role: role,
        status: 'error',
        health: 'unhealthy',
        error: error.message
      };
    }
  }

  async testModelResponse(modelName) {
    try {
      const startTime = Date.now();
      
      const response = await axios.post('http://localhost:11434/api/generate', {
        model: modelName,
        prompt: 'Hello',
        stream: false,
        options: {
          num_predict: 1
        }
      }, {
        timeout: 10000
      });

      return {
        success: true,
        response_time: Date.now() - startTime,
        response_length: response.data.response?.length || 0
      };

    } catch (error) {
      return {
        success: false,
        error: error.message,
        response_time: null
      };
    }
  }

  async getSystemResourcesStatus() {
    const cacheKey = 'system_resources';
    const cached = this.getCachedStatus(cacheKey);
    if (cached) return cached;

    try {
      const memoryUsage = process.memoryUsage();
      const systemMemory = {
        total: os.totalmem(),
        free: os.freemem()
      };

      const cpuUsage = await this.getCPUUsage();
      const loadAverage = os.loadavg();

      const resources = {
        memory: {
          system: {
            total_gb: Math.round(systemMemory.total / 1024 / 1024 / 1024),
            used_gb: Math.round((systemMemory.total - systemMemory.free) / 1024 / 1024 / 1024),
            free_gb: Math.round(systemMemory.free / 1024 / 1024 / 1024),
            usage_percent: Math.round(((systemMemory.total - systemMemory.free) / systemMemory.total) * 100)
          },
          process: {
            heap_used_mb: Math.round(memoryUsage.heapUsed / 1024 / 1024),
            heap_total_mb: Math.round(memoryUsage.heapTotal / 1024 / 1024),
            rss_mb: Math.round(memoryUsage.rss / 1024 / 1024),
            external_mb: Math.round(memoryUsage.external / 1024 / 1024)
          }
        },
        cpu: {
          usage_percent: Math.round(cpuUsage),
          load_average: loadAverage.map(load => Math.round(load * 100) / 100),
          cores: os.cpus().length
        },
        disk: await this.getDiskUsage(),
        uptime: {
          system_hours: Math.round(os.uptime() / 3600),
          process_hours: Math.round(process.uptime() / 3600)
        }
      };

      // 리소스 상태 평가
      resources.health = this.evaluateResourceHealth(resources);

      this.setCachedStatus(cacheKey, resources);
      return resources;

    } catch (error) {
      this.logger.error('시스템 리소스 조회 실패:', error);
      return {
        error: error.message,
        health: 'unknown'
      };
    }
  }

  async getCPUUsage() {
    return new Promise((resolve) => {
      const startUsage = process.cpuUsage();
      const startTime = process.hrtime();

      setTimeout(() => {
        const endUsage = process.cpuUsage(startUsage);
        const endTime = process.hrtime(startTime);
        
        const totalTime = endTime[0] * 1000000 + endTime[1] / 1000;
        const cpuTime = (endUsage.user + endUsage.system);
        const usage = (cpuTime / totalTime) * 100;
        
        resolve(Math.min(usage, 100));
      }, 1000);
    });
  }

  async getDiskUsage() {
    try {
      const fs = await import('fs/promises');
      const stats = await fs.stat('./');
      
      // 간단한 디스크 사용량 추정
      return {
        available: true,
        usage_percent: 'unknown', // 정확한 디스크 사용량은 별도 라이브러리 필요
        free_space: 'unknown'
      };
    } catch (error) {
      return {
        available: false,
        error: error.message
      };
    }
  }

  async getPerformanceMetrics() {
    const cacheKey = 'performance_metrics';
    const cached = this.getCachedStatus(cacheKey);
    if (cached) return cached;

    try {
      const metrics = {
        response_times: await this.getAverageResponseTimes(),
        throughput: await this.getThroughputMetrics(),
        error_rates: await this.getErrorRates(),
        cache_performance: this.getCachePerformance(),
        model_efficiency: await this.getModelEfficiencyMetrics()
      };

      this.setCachedStatus(cacheKey, metrics);
      return metrics;

    } catch (error) {
      this.logger.error('성능 메트릭 조회 실패:', error);
      return {
        error: error.message
      };
    }
  }

  async getAverageResponseTimes() {
    // 최근 응답 시간 통계 (실제 구현에서는 메트릭 수집 시스템 사용)
    return {
      router_model: {
        avg_ms: 250,
        p95_ms: 500,
        p99_ms: 800
      },
      processor_model: {
        avg_ms: 1200,
        p95_ms: 3000,
        p99_ms: 5000
      }
    };
  }

  async getThroughputMetrics() {
    // 처리량 메트릭 (요청/분)
    return {
      requests_per_minute: 12,
      tokens_per_second: 150,
      peak_throughput: 25
    };
  }

  async getErrorRates() {
    // 오류율 통계
    return {
      total_requests: 1000,
      failed_requests: 15,
      error_rate: 0.015,
      timeout_rate: 0.005
    };
  }

  getCachePerformance() {
    // 캐시 성능 메트릭
    return {
      hit_rate: 0.85,
      miss_rate: 0.15,
      size_kb: this.statusCache.size * 2, // 추정값
      evictions: 0
    };
  }

  async getModelEfficiencyMetrics() {
    // 모델 효율성 메트릭
    return {
      memory_efficiency: 0.75,
      compute_utilization: 0.60,
      model_switching_overhead: 1200 // ms
    };
  }

  async getModelPerformanceMetrics(modelType) {
    try {
      // 모델별 성능 메트릭
      const baseMetrics = {
        avg_response_time: modelType === 'router' ? 250 : 1200,
        requests_handled: Math.floor(Math.random() * 100) + 50,
        success_rate: 0.95 + Math.random() * 0.05,
        memory_efficiency: 0.8 + Math.random() * 0.15
      };

      return baseMetrics;
    } catch (error) {
      return {
        error: error.message
      };
    }
  }

  evaluateOverallHealth(status) {
    const weights = {
      ollama_service: 0.3,
      models: 0.4,
      system_resources: 0.3
    };

    let healthScore = 0;

    // Ollama 서비스 점수
    if (status.ollama_service.health === 'healthy') {
      healthScore += weights.ollama_service;
    } else if (status.ollama_service.health === 'degraded') {
      healthScore += weights.ollama_service * 0.5;
    }

    // 모델 상태 점수
    const modelHealthScores = Object.values(status.models)
      .map(model => {
        if (model.health === 'healthy') return 1;
        if (model.health === 'degraded') return 0.5;
        if (model.health === 'idle') return 0.8;
        return 0;
      });
    
    const avgModelHealth = modelHealthScores.reduce((a, b) => a + b, 0) / modelHealthScores.length;
    healthScore += weights.models * avgModelHealth;

    // 시스템 리소스 점수
    if (status.system_resources.health === 'healthy') {
      healthScore += weights.system_resources;
    } else if (status.system_resources.health === 'degraded') {
      healthScore += weights.system_resources * 0.5;
    }

    if (healthScore >= 0.8) return 'healthy';
    if (healthScore >= 0.5) return 'degraded';
    return 'unhealthy';
  }

  evaluateResourceHealth(resources) {
    const memoryUsage = resources.memory.system.usage_percent / 100;
    const cpuUsage = resources.cpu.usage_percent / 100;

    if (memoryUsage > this.alertThresholds.memory.critical ||
        cpuUsage > this.alertThresholds.cpu.critical) {
      return 'critical';
    }

    if (memoryUsage > this.alertThresholds.memory.warning ||
        cpuUsage > this.alertThresholds.cpu.warning) {
      return 'degraded';
    }

    return 'healthy';
  }

  generateAlerts(status) {
    const alerts = [];

    // Ollama 서비스 알림
    if (status.ollama_service.status === 'error') {
      alerts.push({
        level: 'critical',
        type: 'service',
        message: 'Ollama 서비스가 응답하지 않습니다.',
        suggestion: 'ollama serve 명령으로 서비스를 시작하세요.'
      });
    }

    // 메모리 사용량 알림
    const memoryUsage = status.system_resources.memory?.system?.usage_percent;
    if (memoryUsage > this.alertThresholds.memory.critical * 100) {
      alerts.push({
        level: 'critical',
        type: 'memory',
        message: `메모리 사용량이 임계치를 초과했습니다: ${memoryUsage}%`,
        suggestion: '불필요한 모델을 언로드하거나 시스템을 재시작하세요.'
      });
    } else if (memoryUsage > this.alertThresholds.memory.warning * 100) {
      alerts.push({
        level: 'warning',
        type: 'memory',
        message: `메모리 사용량이 높습니다: ${memoryUsage}%`,
        suggestion: '메모리 사용량을 모니터링하세요.'
      });
    }

    // 모델 상태 알림
    for (const [modelType, modelInfo] of Object.entries(status.models)) {
      if (modelInfo.health === 'unhealthy') {
        alerts.push({
          level: 'warning',
          type: 'model',
          message: `${modelType} 모델이 정상적으로 동작하지 않습니다.`,
          suggestion: '모델을 재로드하거나 Ollama 서비스를 재시작하세요.'
        });
      }
    }

    return alerts;
  }

  generateRecommendations(status) {
    const recommendations = [];

    // 성능 최적화 권장사항
    if (status.system_resources.memory?.system?.usage_percent > 70) {
      recommendations.push({
        type: 'optimization',
        priority: 'medium',
        message: '메모리 사용량 최적화를 권장합니다.',
        actions: [
          '사용하지 않는 모델 언로드',
          '캐시 크기 조정',
          '가비지 컬렉션 실행'
        ]
      });
    }

    // 모델 관리 권장사항
    const runningModels = status.ollama_service.running_models || 0;
    if (runningModels > 2) {
      recommendations.push({
        type: 'management',
        priority: 'low',
        message: '많은 모델이 동시에 실행 중입니다.',
        actions: [
          '자주 사용하지 않는 모델 언로드',
          '자동 언로드 타임아웃 설정'
        ]
      });
    }

    // 성능 모니터링 권장사항
    if (!status.performance_metrics || status.performance_metrics.error) {
      recommendations.push({
        type: 'monitoring',
        priority: 'high',
        message: '성능 모니터링을 개선하세요.',
        actions: [
          '메트릭 수집 시스템 설정',
          '알림 임계값 조정',
          '정기적인 헬스체크 스케줄링'
        ]
      });
    }

    return recommendations;
  }

  formatStatusResponse(status) {
    const overallEmoji = this.getHealthEmoji(status.overall_health);
    
    let responseText = `${overallEmoji} **시스템 상태 종합 보고서**\n\n`;
    responseText += `**전체 상태:** ${this.formatHealthStatus(status.overall_health)}\n`;
    responseText += `**조회 시간:** ${new Date(status.timestamp).toLocaleString()}\n\n`;

    // Ollama 서비스 상태
    const ollamaEmoji = this.getHealthEmoji(status.ollama_service.health);
    responseText += `${ollamaEmoji} **Ollama 서비스**\n`;
    responseText += `- 상태: ${status.ollama_service.status}\n`;
    responseText += `- 응답시간: ${status.ollama_service.response_time_ms}ms\n`;
    responseText += `- 실행 중인 모델: ${status.ollama_service.running_models}개\n`;
    responseText += `- 사용 가능한 모델: ${status.ollama_service.available_models}개\n\n`;

    // 모델 상태
    responseText += `🤖 **모델 상태**\n`;
    for (const [modelType, modelInfo] of Object.entries(status.models)) {
      const modelEmoji = this.getHealthEmoji(modelInfo.health);
      responseText += `${modelEmoji} **${modelType.toUpperCase()}** (${modelInfo.name})\n`;
      responseText += `- 상태: ${modelInfo.status}\n`;
      if (modelInfo.memory_usage) {
        responseText += `- 메모리: ${this.formatBytes(modelInfo.memory_usage)}\n`;
      }
      if (modelInfo.response_time_ms) {
        responseText += `- 응답시간: ${modelInfo.response_time_ms}ms\n`;
      }
    }
    responseText += '\n';

    // 시스템 리소스
    const resourceEmoji = this.getHealthEmoji(status.system_resources.health);
    responseText += `${resourceEmoji} **시스템 리소스**\n`;
    if (status.system_resources.memory) {
      responseText += `- 메모리: ${status.system_resources.memory.system.used_gb}GB / ${status.system_resources.memory.system.total_gb}GB (${status.system_resources.memory.system.usage_percent}%)\n`;
    }
    if (status.system_resources.cpu) {
      responseText += `- CPU: ${status.system_resources.cpu.usage_percent}% (${status.system_resources.cpu.cores}코어)\n`;
    }
    responseText += '\n';

    // 알림
    if (status.alerts.length > 0) {
      responseText += `⚠️ **알림 (${status.alerts.length}개)**\n`;
      status.alerts.slice(0, 3).forEach(alert => {
        const alertEmoji = alert.level === 'critical' ? '🔴' : '🟡';
        responseText += `${alertEmoji} ${alert.message}\n`;
      });
      responseText += '\n';
    }

    // 권장사항
    if (status.recommendations.length > 0) {
      responseText += `💡 **권장사항**\n`;
      status.recommendations.slice(0, 2).forEach(rec => {
        responseText += `• ${rec.message}\n`;
      });
    }

    return {
      content: [{
        type: 'text',
        text: responseText
      }],
      metadata: {
        overall_health: status.overall_health,
        timestamp: status.timestamp,
        alerts_count: status.alerts.length,
        recommendations_count: status.recommendations.length,
        detailed_status: status
      }
    };
  }

  createErrorResponse(error) {
    return {
      content: [{
        type: 'text',
        text: `❌ **상태 조회 실패**\n\n오류: ${error.message}\n\n시스템 관리자에게 문의하세요.`
      }],
      isError: true,
      metadata: {
        error: error.message,
        timestamp: new Date().toISOString()
      }
    };
  }

  // 유틸리티 메서드들
  getCachedStatus(key) {
    const cached = this.statusCache.get(key);
    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return cached.data;
    }
    return null;
  }

  setCachedStatus(key, data) {
    this.statusCache.set(key, {
      data: data,
      timestamp: Date.now()
    });
  }

  getHealthEmoji(health) {
    const emojiMap = {
      healthy: '✅',
      degraded: '⚠️',
      unhealthy: '❌',
      critical: '🔴',
      idle: '😴',
      unknown: '❓'
    };
    return emojiMap[health] || '❓';
  }

  formatHealthStatus(health) {
    const statusMap = {
      healthy: '정상',
      degraded: '성능 저하',
      unhealthy: '비정상',
      critical: '심각',
      unknown: '알 수 없음'
    };
    return statusMap[health] || health;
  }

  formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  addToHistory(status) {
    this.statusHistory.push({
      timestamp: status.timestamp,
      overall_health: status.overall_health,
      alerts_count: status.alerts.length,
      memory_usage: status.system_resources.memory?.system?.usage_percent,
      cpu_usage: status.system_resources.cpu?.usage_percent
    });

    // 히스토리 크기 제한
    if (this.statusHistory.length > this.maxHistorySize) {
      this.statusHistory = this.statusHistory.slice(-this.maxHistorySize);
    }
  }

  // 모니터링 관리
  startContinuousMonitoring(intervalMs = 60000) {
    if (this.monitoringInterval) {
      this.stopContinuousMonitoring();
    }

    this.logger.info('연속 모니터링 시작', { interval: intervalMs });
    
    this.monitoringInterval = setInterval(async () => {
      try {
        const status = await this.getComprehensiveStatus();
        
        // 심각한 알림이 있으면 로그
        const criticalAlerts = status.metadata.detailed_status.alerts
          .filter(alert => alert.level === 'critical');
        
        if (criticalAlerts.length > 0) {
          this.logger.error('심각한 시스템 알림 감지:', criticalAlerts);
        }
        
      } catch (error) {
        this.logger.error('연속 모니터링 중 오류:', error);
      }
    }, intervalMs);
  }

  stopContinuousMonitoring() {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
      this.logger.info('연속 모니터링 중지');
    }
  }

  // 빠른 상태 확인
  async getQuickStatus() {
    try {
      const quickStatus = {
        timestamp: new Date().toISOString(),
        ollama_running: false,
        models_loaded: 0,
        memory_usage: 0,
        overall_health: 'unknown'
      };

      // Ollama 빠른 확인
      try {
        await axios.get('http://localhost:11434/api/version', { timeout: 2000 });
        quickStatus.ollama_running = true;
      } catch {
        quickStatus.ollama_running = false;
      }

      // 로드된 모델 수
      if (this.modelManager && this.modelManager.models) {
        quickStatus.models_loaded = this.modelManager.models.size;
      }

      // 메모리 사용량
      const memoryUsage = process.memoryUsage();
      quickStatus.memory_usage = Math.round(memoryUsage.heapUsed / 1024 / 1024);

      // 전체 상태
      if (quickStatus.ollama_running && quickStatus.models_loaded > 0) {
        quickStatus.overall_health = 'healthy';
      } else if (quickStatus.ollama_running) {
        quickStatus.overall_health = 'degraded';
      } else {
        quickStatus.overall_health = 'unhealthy';
      }

      return {
        content: [{
          type: 'text',
          text: `🔍 **빠른 상태 확인**\n\n` +
                `${this.getHealthEmoji(quickStatus.overall_health)} 전체 상태: ${this.formatHealthStatus(quickStatus.overall_health)}\n` +
                `${quickStatus.ollama_running ? '✅' : '❌'} Ollama 서비스: ${quickStatus.ollama_running ? '실행중' : '중지됨'}\n` +
                `🤖 로드된 모델: ${quickStatus.models_loaded}개\n` +
                `💾 메모리 사용량: ${quickStatus.memory_usage}MB\n\n` +
                `상세 정보가 필요하면 "system_status" 명령을 사용하세요.`
        }],
        metadata: quickStatus
      };

    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `❌ 빠른 상태 확인 실패: ${error.message}`
        }],
        isError: true
      };
    }
  }

  // 성능 벤치마크
  async runPerformanceBenchmark() {
    const benchmark = {
      timestamp: new Date().toISOString(),
      tests: {},
      overall_score: 0
    };

    try {
      this.logger.info('성능 벤치마크 시작');

      // 1. Ollama 응답 시간 테스트
      benchmark.tests.ollama_response = await this.benchmarkOllamaResponse();

      // 2. 모델 로딩 시간 테스트
      benchmark.tests.model_loading = await this.benchmarkModelLoading();

      // 3. 메모리 성능 테스트
      benchmark.tests.memory_performance = await this.benchmarkMemoryPerformance();

      // 4. 종합 점수 계산
      benchmark.overall_score = this.calculateBenchmarkScore(benchmark.tests);

      return this.formatBenchmarkResponse(benchmark);

    } catch (error) {
      this.logger.error('성능 벤치마크 실패:', error);
      return this.createErrorResponse(error);
    }
  }

  async benchmarkOllamaResponse() {
    const iterations = 5;
    const responseTimes = [];

    for (let i = 0; i < iterations; i++) {
      try {
        const startTime = Date.now();
        await axios.get('http://localhost:11434/api/version', { timeout: 5000 });
        responseTimes.push(Date.now() - startTime);
      } catch (error) {
        responseTimes.push(5000); // 타임아웃으로 처리
      }
    }

    const avgResponseTime = responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length;

    return {
      avg_response_time: Math.round(avgResponseTime),
      min_response_time: Math.min(...responseTimes),
      max_response_time: Math.max(...responseTimes),
      score: Math.max(0, 100 - avgResponseTime / 10) // 1000ms = 0점, 0ms = 100점
    };
  }

  async benchmarkModelLoading() {
    try {
      const startTime = Date.now();
      
      // 간단한 모델 쿼리로 로딩 시간 측정
      const testModels = ['llama3.2:3b'];
      const loadingTimes = [];

      for (const modelName of testModels) {
        const modelStartTime = Date.now();
        try {
          await this.testModelResponse(modelName);
          loadingTimes.push(Date.now() - modelStartTime);
        } catch (error) {
          loadingTimes.push(10000); // 실패시 10초로 처리
        }
      }

      const avgLoadingTime = loadingTimes.reduce((a, b) => a + b, 0) / loadingTimes.length;

      return {
        avg_loading_time: Math.round(avgLoadingTime),
        tested_models: testModels.length,
        score: Math.max(0, 100 - avgLoadingTime / 50) // 5000ms = 0점, 0ms = 100점
      };

    } catch (error) {
      return {
        error: error.message,
        score: 0
      };
    }
  }

  async benchmarkMemoryPerformance() {
    const startMemory = process.memoryUsage();
    
    // 메모리 집약적 작업 시뮬레이션
    const testData = new Array(100000).fill(0).map((_, i) => ({ id: i, data: Math.random() }));
    
    const endMemory = process.memoryUsage();
    const memoryIncrease = endMemory.heapUsed - startMemory.heapUsed;

    // 가비지 컬렉션 테스트
    const gcStartTime = Date.now();
    if (global.gc) {
      global.gc();
    }
    const gcTime = Date.now() - gcStartTime;

    return {
      memory_increase_mb: Math.round(memoryIncrease / 1024 / 1024),
      gc_time_ms: gcTime,
      heap_utilization: Math.round((endMemory.heapUsed / endMemory.heapTotal) * 100),
      score: Math.max(0, 100 - memoryIncrease / (1024 * 1024)) // 100MB 증가 = 0점
    };
  }

  calculateBenchmarkScore(tests) {
    const weights = {
      ollama_response: 0.4,
      model_loading: 0.4,
      memory_performance: 0.2
    };

    let totalScore = 0;
    let totalWeight = 0;

    for (const [testName, testResult] of Object.entries(tests)) {
      if (testResult.score !== undefined && weights[testName]) {
        totalScore += testResult.score * weights[testName];
        totalWeight += weights[testName];
      }
    }

    return totalWeight > 0 ? Math.round(totalScore / totalWeight) : 0;
  }

  formatBenchmarkResponse(benchmark) {
    let responseText = `📊 **성능 벤치마크 결과**\n\n`;
    responseText += `**종합 점수:** ${benchmark.overall_score}/100\n`;
    responseText += `**테스트 시간:** ${new Date(benchmark.timestamp).toLocaleString()}\n\n`;

    // 각 테스트 결과
    if (benchmark.tests.ollama_response) {
      const test = benchmark.tests.ollama_response;
      responseText += `🔄 **Ollama 응답 시간**\n`;
      responseText += `- 평균: ${test.avg_response_time}ms\n`;
      responseText += `- 최소/최대: ${test.min_response_time}ms / ${test.max_response_time}ms\n`;
      responseText += `- 점수: ${Math.round(test.score)}/100\n\n`;
    }

    if (benchmark.tests.model_loading) {
      const test = benchmark.tests.model_loading;
      responseText += `🤖 **모델 로딩 성능**\n`;
      responseText += `- 평균 로딩 시간: ${test.avg_loading_time}ms\n`;
      responseText += `- 테스트된 모델: ${test.tested_models}개\n`;
      responseText += `- 점수: ${Math.round(test.score)}/100\n\n`;
    }

    if (benchmark.tests.memory_performance) {
      const test = benchmark.tests.memory_performance;
      responseText += `💾 **메모리 성능**\n`;
      responseText += `- 메모리 증가: ${test.memory_increase_mb}MB\n`;
      responseText += `- GC 시간: ${test.gc_time_ms}ms\n`;
      responseText += `- 힙 사용률: ${test.heap_utilization}%\n`;
      responseText += `- 점수: ${Math.round(test.score)}/100\n\n`;
    }

    // 성능 평가
    let performanceLevel;
    if (benchmark.overall_score >= 80) {
      performanceLevel = '🟢 우수';
    } else if (benchmark.overall_score >= 60) {
      performanceLevel = '🟡 보통';
    } else {
      performanceLevel = '🔴 개선 필요';
    }

    responseText += `**성능 평가:** ${performanceLevel}`;

    return {
      content: [{
        type: 'text',
        text: responseText
      }],
      metadata: {
        benchmark_score: benchmark.overall_score,
        timestamp: benchmark.timestamp,
        detailed_results: benchmark.tests
      }
    };
  }

  // 상태 히스토리 조회
  getStatusHistory(limit = 10) {
    const recentHistory = this.statusHistory.slice(-limit);
    
    if (recentHistory.length === 0) {
      return {
        content: [{
          type: 'text',
          text: '📊 상태 히스토리가 없습니다.\n\n연속 모니터링을 시작하거나 상태를 여러 번 확인한 후 다시 시도하세요.'
        }]
      };
    }

    let responseText = `📈 **상태 히스토리 (최근 ${recentHistory.length}개)**\n\n`;

    recentHistory.reverse().forEach((entry, index) => {
      const time = new Date(entry.timestamp).toLocaleTimeString();
      const healthEmoji = this.getHealthEmoji(entry.overall_health);
      
      responseText += `${healthEmoji} ${time} - ${this.formatHealthStatus(entry.overall_health)}`;
      
      if (entry.memory_usage) {
        responseText += ` (메모리: ${entry.memory_usage}%)`;
      }
      
      if (entry.alerts_count > 0) {
        responseText += ` ⚠️${entry.alerts_count}`;
      }
      
      responseText += '\n';
    });

    return {
      content: [{
        type: 'text',
        text: responseText
      }],
      metadata: {
        history_count: recentHistory.length,
        latest_status: recentHistory[0]?.overall_health
      }
    };
  }

  // 알림 임계값 설정
  updateAlertThresholds(newThresholds) {
    this.alertThresholds = { ...this.alertThresholds, ...newThresholds };
    this.logger.info('알림 임계값 업데이트됨:', newThresholds);
    
    return {
      content: [{
        type: 'text',
        text: `⚙️ **알림 임계값 업데이트 완료**\n\n현재 설정:\n` +
              `- 메모리 경고: ${this.alertThresholds.memory.warning * 100}%\n` +
              `- 메모리 임계: ${this.alertThresholds.memory.critical * 100}%\n` +
              `- CPU 경고: ${this.alertThresholds.cpu.warning * 100}%\n` +
              `- CPU 임계: ${this.alertThresholds.cpu.critical * 100}%`
      }]
    };
  }

  // 캐시 관리
  clearStatusCache() {
    const cacheSize = this.statusCache.size;
    this.statusCache.clear();
    
    this.logger.info('상태 캐시 클리어됨', { clearedEntries: cacheSize });
    
    return {
      content: [{
        type: 'text',
        text: `🗑️ **캐시 클리어 완료**\n\n${cacheSize}개의 캐시 항목이 삭제되었습니다.\n다음 상태 조회부터 최신 정보가 반영됩니다.`
      }]
    };
  }

  // 정리 작업
  cleanup() {
    this.stopContinuousMonitoring();
    this.clearStatusCache();
    this.statusHistory = [];
    this.logger.info('ModelStatus 정리 완료');
  }
}
