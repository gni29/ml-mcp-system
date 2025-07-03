// core/memory-manager.js
import { Logger } from '../utils/logger.js';
import { CacheManager } from '../utils/cache-manager.js';
import fs from 'fs/promises';
import os from 'os';

export class MemoryManager {
  constructor() {
    this.logger = new Logger();
    this.cache = new CacheManager();
    this.memoryThresholds = null;
    this.monitoringInterval = null;
    this.isMonitoring = false;
    this.memoryStats = {
      peak: 0,
      current: 0,
      warnings: 0,
      cleanups: 0
    };
  }

  async initialize() {
    try {
      await this.loadMemoryThresholds();
      this.startMemoryMonitoring();
      this.logger.info('MemoryManager 초기화 완료');
    } catch (error) {
      this.logger.error('MemoryManager 초기화 실패:', error);
      throw error;
    }
  }

  async loadMemoryThresholds() {
    try {
      const thresholdsData = await fs.readFile('./config/memory-thresholds.json', 'utf-8');
      this.memoryThresholds = JSON.parse(thresholdsData);
    } catch (error) {
      this.logger.warn('메모리 임계값 로드 실패, 기본값 사용:', error);
      this.memoryThresholds = this.getDefaultThresholds();
    }
  }

  getDefaultThresholds() {
    return {
      router: {
        maxMemoryMB: 6000,
        warningThresholdMB: 5000,
        autoUnload: false
      },
      processor: {
        maxMemoryMB: 28000,
        warningThresholdMB: 25000,
        autoUnload: true,
        autoUnloadTimeoutMs: 600000
      },
      system: {
        maxTotalMemoryMB: 32000,
        emergencyThresholdMB: 30000
      }
    };
  }

  startMemoryMonitoring() {
    if (this.isMonitoring) return;
    
    this.isMonitoring = true;
    this.monitoringInterval = setInterval(() => {
      this.checkMemoryUsage();
    }, 10000); // 10초마다 체크

    this.logger.info('메모리 모니터링 시작');
  }

  stopMemoryMonitoring() {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
    this.isMonitoring = false;
    this.logger.info('메모리 모니터링 중지');
  }

  async checkMemoryUsage() {
    try {
      const memoryUsage = this.getCurrentMemoryUsage();
      this.memoryStats.current = memoryUsage.totalMB;
      
      if (memoryUsage.totalMB > this.memoryStats.peak) {
        this.memoryStats.peak = memoryUsage.totalMB;
      }

      // 경고 임계값 체크
      if (memoryUsage.totalMB > this.memoryThresholds.system.emergencyThresholdMB) {
        this.memoryStats.warnings++;
        await this.handleEmergencyCleanup();
      } else if (memoryUsage.totalMB > this.memoryThresholds.system.maxTotalMemoryMB * 0.8) {
        await this.handleWarningLevel();
      }

      // 상세 로깅 (디버그 모드에서만)
      if (process.env.LOG_LEVEL === 'debug') {
        this.logger.debug('메모리 사용량:', memoryUsage);
      }

    } catch (error) {
      this.logger.error('메모리 체크 실패:', error);
    }
  }

  getCurrentMemoryUsage() {
    const processMemory = process.memoryUsage();
    const systemMemory = {
      total: os.totalmem(),
      free: os.freemem()
    };

    const processUsageMB = {
      rss: Math.round(processMemory.rss / 1024 / 1024),
      heapTotal: Math.round(processMemory.heapTotal / 1024 / 1024),
      heapUsed: Math.round(processMemory.heapUsed / 1024 / 1024),
      external: Math.round(processMemory.external / 1024 / 1024)
    };

    const systemUsageMB = {
      total: Math.round(systemMemory.total / 1024 / 1024),
      free: Math.round(systemMemory.free / 1024 / 1024),
      used: Math.round((systemMemory.total - systemMemory.free) / 1024 / 1024)
    };

    return {
      process: processUsageMB,
      system: systemUsageMB,
      totalMB: processUsageMB.rss,
      heapUtilization: processUsageMB.heapUsed / processUsageMB.heapTotal,
      systemUtilization: systemUsageMB.used / systemUsageMB.total
    };
  }

  async handleEmergencyCleanup() {
    this.logger.warn('긴급 메모리 정리 시작');
    this.memoryStats.cleanups++;

    try {
      // 1. 가비지 컬렉션 강제 실행
      if (global.gc) {
        global.gc();
      }

      // 2. 캐시 정리
      this.cache.flush();

      // 3. 사용하지 않는 모델 언로드 요청
      await this.requestModelUnload();

      // 4. 메모리 사용량 재확인
      const afterCleanup = this.getCurrentMemoryUsage();
      this.logger.info('긴급 정리 완료', {
        memoryReduced: this.memoryStats.current - afterCleanup.totalMB
      });

    } catch (error) {
      this.logger.error('긴급 메모리 정리 실패:', error);
    }
  }

  async handleWarningLevel() {
    this.logger.warn('메모리 사용량 경고 수준');

    try {
      // 소프트 정리 실행
      await this.performSoftCleanup();
    } catch (error) {
      this.logger.error('경고 수준 정리 실패:', error);
    }
  }

  async performSoftCleanup() {
    // 1. 오래된 캐시 항목 정리
    this.cleanupOldCacheEntries();

    // 2. 결과 스토어 정리 요청
    await this.requestResultStoreCleanup();

    // 3. 임시 파일 정리
    await this.cleanupTemporaryFiles();
  }

  cleanupOldCacheEntries() {
    const stats = this.cache.getStats();
    const targetReduction = Math.max(1, Math.floor(stats.keys * 0.2)); // 20% 정리
    
    // 오래된 항목들 제거 (구현은 CacheManager에서)
    this.logger.debug(`캐시 정리: ${targetReduction}개 항목 제거 예정`);
  }

  async requestResultStoreCleanup() {
    // ResultStore에 정리 요청 (이벤트 기반으로 구현)
    process.emit('memory-pressure', { level: 'warning' });
  }

  async cleanupTemporaryFiles() {
    try {
      const tempDirs = ['./temp', './uploads'];
      
      for (const dir of tempDirs) {
        try {
          const files = await fs.readdir(dir);
          const now = Date.now();
          
          for (const file of files) {
            const filePath = `${dir}/${file}`;
            const stat = await fs.stat(filePath);
            
            // 1시간 이상 된 임시 파일 삭제
            if (now - stat.mtime.getTime() > 3600000) {
              await fs.unlink(filePath);
              this.logger.debug(`임시 파일 삭제: ${filePath}`);
            }
          }
        } catch (error) {
          // 디렉토리가 없는 경우 무시
        }
      }
    } catch (error) {
      this.logger.warn('임시 파일 정리 실패:', error);
    }
  }

  async requestModelUnload() {
    // ModelManager에 모델 언로드 요청
    process.emit('memory-pressure', {
      level: 'emergency',
      action: 'unload_models'
    });
  }

  // 메모리 예약 및 해제
  reserveMemory(componentName, estimatedMB) {
    const current = this.getCurrentMemoryUsage();
    const projected = current.totalMB + estimatedMB;
    
    if (projected > this.memoryThresholds.system.maxTotalMemoryMB) {
      throw new Error(`메모리 부족: 현재 ${current.totalMB}MB, 예상 ${projected}MB`);
    }
    
    this.logger.debug(`메모리 예약: ${componentName} - ${estimatedMB}MB`);
    return true;
  }

  releaseMemory(componentName, actualMB) {
    this.logger.debug(`메모리 해제: ${componentName} - ${actualMB}MB`);
    
    // 가비지 컬렉션 힌트
    if (actualMB > 1000 && global.gc) {
      setTimeout(() => global.gc(), 1000);
    }
  }

  // 메모리 사용량 분석
  analyzeMemoryUsage() {
    const current = this.getCurrentMemoryUsage();
    const cacheStats = this.cache.getStats();
    
    return {
      current: current,
      statistics: this.memoryStats,
      cache: cacheStats,
      thresholds: this.memoryThresholds,
      recommendations: this.generateRecommendations(current)
    };
  }

  generateRecommendations(memoryUsage) {
    const recommendations = [];
    
    if (memoryUsage.heapUtilization > 0.9) {
      recommendations.push('힙 메모리 사용률이 높습니다. 캐시 크기를 줄이는 것을 고려하세요.');
    }
    
    if (memoryUsage.systemUtilization > 0.8) {
      recommendations.push('시스템 메모리 사용률이 높습니다. 다른 프로세스를 종료하는 것을 고려하세요.');
    }
    
    if (this.memoryStats.warnings > 10) {
      recommendations.push('메모리 경고가 빈번합니다. 메모리 임계값을 조정하거나 시스템을 업그레이드하세요.');
    }
    
    return recommendations;
  }

  // 메모리 프로파일링
  startProfiling() {
    this.profilingData = {
      startTime: Date.now(),
      samples: [],
      interval: setInterval(() => {
        this.profilingData.samples.push({
          timestamp: Date.now(),
          usage: this.getCurrentMemoryUsage()
        });
      }, 1000)
    };
    
    this.logger.info('메모리 프로파일링 시작');
  }

  stopProfiling() {
    if (this.profilingData) {
      clearInterval(this.profilingData.interval);
      
      const duration = Date.now() - this.profilingData.startTime;
      const report = this.generateProfilingReport(this.profilingData.samples, duration);
      
      this.logger.info('메모리 프로파일링 완료', report);
      this.profilingData = null;
      
      return report;
    }
    
    return null;
  }

  generateProfilingReport(samples, duration) {
    if (samples.length === 0) return null;
    
    const memoryValues = samples.map(s => s.usage.totalMB);
    const heapValues = samples.map(s => s.usage.process.heapUsed);
    
    return {
      duration: duration,
      samples: samples.length,
      memory: {
        min: Math.min(...memoryValues),
        max: Math.max(...memoryValues),
        avg: memoryValues.reduce((a, b) => a + b, 0) / memoryValues.length,
        final: memoryValues[memoryValues.length - 1]
      },
      heap: {
        min: Math.min(...heapValues),
        max: Math.max(...heapValues),
        avg: heapValues.reduce((a, b) => a + b, 0) / heapValues.length,
        final: heapValues[heapValues.length - 1]
      },
      trend: this.calculateTrend(memoryValues)
    };
  }

  calculateTrend(values) {
    if (values.length < 2) return 'insufficient_data';
    
    const firstHalf = values.slice(0, Math.floor(values.length / 2));
    const secondHalf = values.slice(Math.floor(values.length / 2));
    
    const firstAvg = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
    const secondAvg = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;
    
    const difference = secondAvg - firstAvg;
    const threshold = firstAvg * 0.1; // 10% 임계값
    
    if (Math.abs(difference) < threshold) return 'stable';
    return difference > 0 ? 'increasing' : 'decreasing';
  }

  // 이벤트 리스너 설정
  setupEventListeners() {
    process.on('memory-pressure', (data) => {
      this.handleMemoryPressureEvent(data);
    });
    
    process.on('beforeExit', () => {
      this.stopMemoryMonitoring();
    });
  }

  handleMemoryPressureEvent(data) {
    this.logger.info('메모리 압박 이벤트 수신:', data);
    
    switch (data.level) {
      case 'warning':
        this.performSoftCleanup();
        break;
      case 'emergency':
        this.handleEmergencyCleanup();
        break;
    }
  }

  // 메모리 최적화 제안
  optimizeMemory() {
    const analysis = this.analyzeMemoryUsage();
    const optimizations = [];
    
    // 캐시 최적화
    if (analysis.cache.vsize > 100 * 1024 * 1024) { // 100MB 이상
      optimizations.push({
        type: 'cache',
        action: 'reduce_cache_size',
        impact: 'medium',
        description: '캐시 크기 축소'
      });
    }
    
    // 힙 최적화
    if (analysis.current.heapUtilization > 0.8) {
      optimizations.push({
        type: 'heap',
        action: 'force_gc',
        impact: 'low',
        description: '가비지 컬렉션 강제 실행'
      });
    }
    
    // 모델 최적화
    if (analysis.current.totalMB > this.memoryThresholds.system.maxTotalMemoryMB * 0.7) {
      optimizations.push({
        type: 'models',
        action: 'unload_unused_models',
        impact: 'high',
        description: '사용하지 않는 모델 언로드'
      });
    }
    
    return optimizations;
  }

  // 자동 최적화 실행
  async executeOptimizations(optimizations) {
    for (const opt of optimizations) {
      try {
        switch (opt.action) {
          case 'reduce_cache_size':
            this.cache.flush();
            break;
          case 'force_gc':
            if (global.gc) global.gc();
            break;
          case 'unload_unused_models':
            await this.requestModelUnload();
            break;
        }
        
        this.logger.info(`최적화 실행: ${opt.description}`);
      } catch (error) {
        this.logger.error(`최적화 실행 실패: ${opt.description}`, error);
      }
    }
  }

  // 통계 및 상태 조회
  getMemoryStatistics() {
    return {
      current: this.getCurrentMemoryUsage(),
      statistics: this.memoryStats,
      thresholds: this.memoryThresholds,
      monitoring: {
        isActive: this.isMonitoring,
        intervalMs: this.monitoringInterval ? 10000 : null
      }
    };
  }

  // 설정 업데이트
  updateThresholds(newThresholds) {
    this.memoryThresholds = { ...this.memoryThresholds, ...newThresholds };
    this.logger.info('메모리 임계값 업데이트됨');
  }

  // 정리
  destroy() {
    this.stopMemoryMonitoring();
    if (this.profilingData) {
      this.stopProfiling();
    }
    this.logger.info('MemoryManager 정리 완료');
  }
}
