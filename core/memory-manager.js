// core/memory-manager.js - 완성된 버전
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
    this.profilingData = null;
    this.memoryStats = {
      peak: 0,
      current: 0,
      warnings: 0,
      cleanups: 0,
      allocations: 0,
      deallocations: 0
    };
    this.memoryReservations = new Map();
    this.cleanupHistory = [];
    this.lastCleanupTime = 0;
  }

  async initialize() {
    try {
      await this.loadMemoryThresholds();
      this.setupEventListeners();
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
        autoUnload: false,
        autoUnloadTimeoutMs: 300000
      },
      processor: {
        maxMemoryMB: 28000,
        warningThresholdMB: 25000,
        autoUnload: true,
        autoUnloadTimeoutMs: 600000
      },
      system: {
        maxTotalMemoryMB: 32000,
        emergencyThresholdMB: 30000,
        warningThresholdMB: 25000,
        criticalThresholdMB: 28000
      },
      cache: {
        maxSizeMB: 1000,
        warningThresholdMB: 800,
        cleanupPercentage: 0.3
      },
      cleanup: {
        minIntervalMs: 60000, // 최소 1분 간격
        maxTempFileAge: 3600000, // 1시간
        maxLogFileAge: 86400000 // 24시간
      }
    };
  }

  setupEventListeners() {
    process.on('memory-pressure', (data) => {
      this.handleMemoryPressureEvent(data);
    });
    
    process.on('beforeExit', () => {
      this.stopMemoryMonitoring();
    });

    // 메모리 관련 이벤트 리스너
    process.on('warning', (warning) => {
      if (warning.name === 'MaxListenersExceededWarning' ||
          warning.name === 'DeprecationWarning') {
        this.logger.warn('Node.js 경고:', warning.message);
      }
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
      case 'critical':
        this.handleCriticalCleanup();
        break;
    }
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

      // 임계값 체크
      if (memoryUsage.totalMB > this.memoryThresholds.system.emergencyThresholdMB) {
        this.memoryStats.warnings++;
        await this.handleEmergencyCleanup();
      } else if (memoryUsage.totalMB > this.memoryThresholds.system.criticalThresholdMB) {
        await this.handleCriticalCleanup();
      } else if (memoryUsage.totalMB > this.memoryThresholds.system.warningThresholdMB) {
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
      external: Math.round(processMemory.external / 1024 / 1024),
      arrayBuffers: Math.round(processMemory.arrayBuffers / 1024 / 1024)
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
      systemUtilization: systemUsageMB.used / systemUsageMB.total,
      timestamp: Date.now()
    };
  }

  async handleEmergencyCleanup() {
    this.logger.warn('긴급 메모리 정리 시작');
    this.memoryStats.cleanups++;

    const cleanupStart = Date.now();
    const beforeCleanup = this.getCurrentMemoryUsage();

    try {
      // 1. 가비지 컬렉션 강제 실행
      if (global.gc) {
        global.gc();
        this.logger.debug('가비지 컬렉션 실행 완료');
      }

      // 2. 캐시 완전 정리
      await this.performCacheCleanup(1.0); // 100% 정리

      // 3. 사용하지 않는 모델 언로드 요청
      await this.requestModelUnload();

      // 4. 임시 파일 정리
      await this.cleanupTemporaryFiles();

      // 5. 메모리 예약 정리
      this.cleanupExpiredReservations();

      // 6. 메모리 사용량 재확인
      const afterCleanup = this.getCurrentMemoryUsage();
      const memoryFreed = beforeCleanup.totalMB - afterCleanup.totalMB;
      const cleanupTime = Date.now() - cleanupStart;

      this.recordCleanupResult('emergency', memoryFreed, cleanupTime);

      this.logger.info('긴급 정리 완료', {
        memoryFreed: `${memoryFreed}MB`,
        cleanupTime: `${cleanupTime}ms`,
        currentUsage: `${afterCleanup.totalMB}MB`
      });

    } catch (error) {
      this.logger.error('긴급 메모리 정리 실패:', error);
    }
  }

  async handleCriticalCleanup() {
    this.logger.warn('심각한 메모리 사용량 - 중요 정리 시작');
    
    const cleanupStart = Date.now();
    const beforeCleanup = this.getCurrentMemoryUsage();

    try {
      // 1. 부분 가비지 컬렉션
      if (global.gc) {
        global.gc();
      }

      // 2. 캐시 부분 정리 (70%)
      await this.performCacheCleanup(0.7);

      // 3. 오래된 결과 정리
      await this.requestResultStoreCleanup();

      // 4. 임시 파일 정리
      await this.cleanupTemporaryFiles();

      const afterCleanup = this.getCurrentMemoryUsage();
      const memoryFreed = beforeCleanup.totalMB - afterCleanup.totalMB;
      const cleanupTime = Date.now() - cleanupStart;

      this.recordCleanupResult('critical', memoryFreed, cleanupTime);

      this.logger.info('중요 정리 완료', {
        memoryFreed: `${memoryFreed}MB`,
        cleanupTime: `${cleanupTime}ms`
      });

    } catch (error) {
      this.logger.error('중요 메모리 정리 실패:', error);
    }
  }

  async handleWarningLevel() {
    this.logger.warn('메모리 사용량 경고 수준');

    try {
      await this.performSoftCleanup();
    } catch (error) {
      this.logger.error('경고 수준 정리 실패:', error);
    }
  }

  async performSoftCleanup() {
    // 소프트 정리는 너무 자주 실행되지 않도록 제한
    const now = Date.now();
    if (now - this.lastCleanupTime < this.memoryThresholds.cleanup.minIntervalMs) {
      return;
    }

    this.lastCleanupTime = now;
    const cleanupStart = Date.now();
    const beforeCleanup = this.getCurrentMemoryUsage();

    try {
      // 1. 오래된 캐시 항목 정리
      await this.cleanupOldCacheEntries();

      // 2. 결과 스토어 정리 요청
      await this.requestResultStoreCleanup();

      // 3. 임시 파일 정리
      await this.cleanupTemporaryFiles();

      // 4. 로그 파일 정리
      await this.cleanupLogFiles();

      // 5. 메모리 예약 정리
      this.cleanupExpiredReservations();

      const afterCleanup = this.getCurrentMemoryUsage();
      const memoryFreed = beforeCleanup.totalMB - afterCleanup.totalMB;
      const cleanupTime = Date.now() - cleanupStart;

      this.recordCleanupResult('soft', memoryFreed, cleanupTime);

      this.logger.debug('소프트 정리 완료', {
        memoryFreed: `${memoryFreed}MB`,
        cleanupTime: `${cleanupTime}ms`
      });

    } catch (error) {
      this.logger.error('소프트 정리 실패:', error);
    }
  }

  async cleanupOldCacheEntries() {
    try {
      const cacheStats = this.cache.getStats();
      
      // 캐시 크기가 임계값을 초과하는 경우에만 정리
      if (cacheStats.vsize > this.memoryThresholds.cache.warningThresholdMB * 1024 * 1024) {
        const cleanupPercentage = this.memoryThresholds.cache.cleanupPercentage;
        const targetReduction = Math.max(1, Math.floor(cacheStats.keys * cleanupPercentage));
        
        // CacheManager에 정리 요청
        await this.performCacheCleanup(cleanupPercentage);
        
        this.logger.debug(`캐시 정리: 예상 ${targetReduction}개 항목 제거`);
      }
    } catch (error) {
      this.logger.error('캐시 정리 실패:', error);
    }
  }

  async performCacheCleanup(percentage) {
    try {
      const beforeStats = this.cache.getStats();
      
      // 캐시 전체 정리 또는 부분 정리
      if (percentage >= 1.0) {
        this.cache.flush();
      } else {
        // 부분 정리 - 가장 오래된 항목부터 제거
        const targetRemoval = Math.floor(beforeStats.keys * percentage);
        
        // CacheManager에 부분 정리 기능이 있다면 사용
        if (this.cache.cleanup) {
          await this.cache.cleanup(targetRemoval);
        } else {
          // 기본 구현: 전체 정리
          this.cache.flush();
        }
      }
      
      const afterStats = this.cache.getStats();
      const itemsRemoved = beforeStats.keys - afterStats.keys;
      
      this.logger.debug(`캐시 정리 완료: ${itemsRemoved}개 항목 제거`);
      
    } catch (error) {
      this.logger.error('캐시 정리 실패:', error);
    }
  }

  async requestResultStoreCleanup() {
    try {
      // ResultStore에 정리 요청 (이벤트 기반)
      process.emit('memory-pressure', {
        level: 'warning',
        source: 'memory-manager',
        action: 'cleanup_results'
      });
      
      // 직접 결과 정리 (결과 디렉토리가 있다면)
      await this.cleanupResultsDirectory();
      
    } catch (error) {
      this.logger.error('결과 스토어 정리 요청 실패:', error);
    }
  }

  async cleanupResultsDirectory() {
    try {
      const resultsDir = './results';
      const files = await fs.readdir(resultsDir);
      const now = Date.now();
      let deletedCount = 0;
      
      for (const file of files) {
        const filePath = `${resultsDir}/${file}`;
        const stat = await fs.stat(filePath);
        
        // 24시간 이상 된 결과 파일 삭제
        if (now - stat.mtime.getTime() > 86400000) {
          await fs.unlink(filePath);
          deletedCount++;
        }
      }
      
      if (deletedCount > 0) {
        this.logger.debug(`결과 파일 정리: ${deletedCount}개 파일 삭제`);
      }
      
    } catch (error) {
      if (error.code !== 'ENOENT') {
        this.logger.error('결과 디렉토리 정리 실패:', error);
      }
    }
  }

  async cleanupTemporaryFiles() {
    try {
      const tempDirs = ['./temp', './uploads'];
      const maxAge = this.memoryThresholds.cleanup.maxTempFileAge;
      const now = Date.now();
      let totalDeleted = 0;
      
      for (const dir of tempDirs) {
        try {
          const files = await fs.readdir(dir);
          
          for (const file of files) {
            const filePath = `${dir}/${file}`;
            const stat = await fs.stat(filePath);
            
            // 지정된 시간 이상 된 임시 파일 삭제
            if (now - stat.mtime.getTime() > maxAge) {
              await fs.unlink(filePath);
              totalDeleted++;
              this.logger.debug(`임시 파일 삭제: ${filePath}`);
            }
          }
        } catch (error) {
          if (error.code !== 'ENOENT') {
            this.logger.warn(`임시 디렉토리 정리 실패: ${dir}`, error);
          }
        }
      }
      
      if (totalDeleted > 0) {
        this.logger.debug(`임시 파일 정리 완료: ${totalDeleted}개 파일 삭제`);
      }
      
    } catch (error) {
      this.logger.error('임시 파일 정리 실패:', error);
    }
  }

  async cleanupLogFiles() {
    try {
      const logDir = './data/logs';
      const maxAge = this.memoryThresholds.cleanup.maxLogFileAge;
      const now = Date.now();
      let totalDeleted = 0;
      
      const files = await fs.readdir(logDir);
      
      for (const file of files) {
        if (file.endsWith('.log') && !file.includes('current')) {
          const filePath = `${logDir}/${file}`;
          const stat = await fs.stat(filePath);
          
          // 지정된 시간 이상 된 로그 파일 삭제
          if (now - stat.mtime.getTime() > maxAge) {
            await fs.unlink(filePath);
            totalDeleted++;
          }
        }
      }
      
      if (totalDeleted > 0) {
        this.logger.debug(`로그 파일 정리 완료: ${totalDeleted}개 파일 삭제`);
      }
      
    } catch (error) {
      if (error.code !== 'ENOENT') {
        this.logger.error('로그 파일 정리 실패:', error);
      }
    }
  }

  cleanupExpiredReservations() {
    const now = Date.now();
    let expiredCount = 0;
    
    for (const [componentName, reservation] of this.memoryReservations) {
      // 1시간 이상 된 예약 정리
      if (now - reservation.timestamp > 3600000) {
        this.memoryReservations.delete(componentName);
        expiredCount++;
      }
    }
    
    if (expiredCount > 0) {
      this.logger.debug(`만료된 메모리 예약 정리: ${expiredCount}개`);
    }
  }

  recordCleanupResult(type, memoryFreed, cleanupTime) {
    const result = {
      type,
      memoryFreed,
      cleanupTime,
      timestamp: Date.now()
    };
    
    this.cleanupHistory.push(result);
    
    // 히스토리 크기 제한 (최대 100개)
    if (this.cleanupHistory.length > 100) {
      this.cleanupHistory = this.cleanupHistory.slice(-50);
    }
  }

  async requestModelUnload() {
    try {
      // ModelManager에 모델 언로드 요청
      process.emit('memory-pressure', {
        level: 'emergency',
        source: 'memory-manager',
        action: 'unload_models'
      });
      
      this.logger.debug('모델 언로드 요청 발송');
      
    } catch (error) {
      this.logger.error('모델 언로드 요청 실패:', error);
    }
  }

  // 메모리 예약 및 해제
  reserveMemory(componentName, estimatedMB) {
    const current = this.getCurrentMemoryUsage();
    const totalReserved = Array.from(this.memoryReservations.values())
      .reduce((sum, res) => sum + res.amount, 0);
    const projected = current.totalMB + estimatedMB + totalReserved;
    
    if (projected > this.memoryThresholds.system.maxTotalMemoryMB) {
      throw new Error(
        `메모리 부족: 현재 ${current.totalMB}MB, 예약된 ${totalReserved}MB, 요청 ${estimatedMB}MB, 예상 ${projected}MB`
      );
    }
    
    this.memoryReservations.set(componentName, {
      amount: estimatedMB,
      timestamp: Date.now()
    });
    
    this.memoryStats.allocations++;
    this.logger.debug(`메모리 예약: ${componentName} - ${estimatedMB}MB`);
    
    return true;
  }

  releaseMemory(componentName, actualMB) {
    if (this.memoryReservations.has(componentName)) {
      this.memoryReservations.delete(componentName);
      this.memoryStats.deallocations++;
    }
    
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
      reservations: Array.from(this.memoryReservations.entries()).map(([name, res]) => ({
        component: name,
        amount: res.amount,
        timestamp: res.timestamp
      })),
      recommendations: this.generateRecommendations(current),
      cleanupHistory: this.cleanupHistory.slice(-10) // 최근 10개
    };
  }

  generateRecommendations(memoryUsage) {
    const recommendations = [];
    
    if (memoryUsage.heapUtilization > 0.9) {
      recommendations.push({
        type: 'heap',
        priority: 'high',
        message: '힙 메모리 사용률이 높습니다. 캐시 크기를 줄이는 것을 고려하세요.',
        action: 'reduce_cache_size'
      });
    }
    
    if (memoryUsage.systemUtilization > 0.8) {
      recommendations.push({
        type: 'system',
        priority: 'medium',
        message: '시스템 메모리 사용률이 높습니다. 다른 프로세스를 종료하는 것을 고려하세요.',
        action: 'close_other_processes'
      });
    }
    
    if (this.memoryStats.warnings > 10) {
      recommendations.push({
        type: 'monitoring',
        priority: 'high',
        message: '메모리 경고가 빈번합니다. 메모리 임계값을 조정하거나 시스템을 업그레이드하세요.',
        action: 'adjust_thresholds'
      });
    }
    
    // 캐시 사용량 체크
    const cacheStats = this.cache.getStats();
    if (cacheStats.vsize > this.memoryThresholds.cache.warningThresholdMB * 1024 * 1024) {
      recommendations.push({
        type: 'cache',
        priority: 'medium',
        message: '캐시 사용량이 높습니다. 캐시 정리를 실행하세요.',
        action: 'cleanup_cache'
      });
    }
    
    return recommendations;
  }

  // 메모리 프로파일링
  startProfiling() {
    if (this.profilingData) {
      this.logger.warn('프로파일링이 이미 실행 중입니다.');
      return;
    }
    
    this.profilingData = {
      startTime: Date.now(),
      samples: [],
      interval: setInterval(() => {
        const usage = this.getCurrentMemoryUsage();
        this.profilingData.samples.push({
          timestamp: Date.now(),
          usage: usage
        });
        
        // 최대 1000개 샘플로 제한
        if (this.profilingData.samples.length > 1000) {
          this.profilingData.samples = this.profilingData.samples.slice(-500);
        }
      }, 1000)
    };
    
    this.logger.info('메모리 프로파일링 시작');
  }

  stopProfiling() {
    if (!this.profilingData) {
      this.logger.warn('실행 중인 프로파일링이 없습니다.');
      return null;
    }
    
    clearInterval(this.profilingData.interval);
    
    const duration = Date.now() - this.profilingData.startTime;
    const report = this.generateProfilingReport(this.profilingData.samples, duration);
    
    this.logger.info('메모리 프로파일링 완료', {
      duration: `${duration}ms`,
      samples: this.profilingData.samples.length
    });
    
    this.profilingData = null;
    return report;
  }

  generateProfilingReport(samples, duration) {
    if (samples.length === 0) {
      return {
        error: '프로파일링 데이터가 없습니다.',
        duration: duration
      };
    }
    
    const memoryValues = samples.map(s => s.usage.totalMB);
    const heapValues = samples.map(s => s.usage.process.heapUsed);
    const systemValues = samples.map(s => s.usage.system.used);
    
    const report = {
      duration: duration,
      samples: samples.length,
      sampling_interval: duration / samples.length,
      memory: {
        min: Math.min(...memoryValues),
        max: Math.max(...memoryValues),
        avg: Math.round(memoryValues.reduce((a, b) => a + b, 0) / memoryValues.length),
        final: memoryValues[memoryValues.length - 1],
        peak: Math.max(...memoryValues),
        variance: this.calculateVariance(memoryValues)
      },
      heap: {
        min: Math.min(...heapValues),
        max: Math.max(...heapValues),
        avg: Math.round(heapValues.reduce((a, b) => a + b, 0) / heapValues.length),
        final: heapValues[heapValues.length - 1],
        peak: Math.max(...heapValues),
        variance: this.calculateVariance(heapValues)
      },
      system: {
        min: Math.min(...systemValues),
        max: Math.max(...systemValues),
        avg: Math.round(systemValues.reduce((a, b) => a + b, 0) / systemValues.length),
        final: systemValues[systemValues.length - 1],
        peak: Math.max(...systemValues),
        variance: this.calculateVariance(systemValues)
      },
      trend: this.calculateTrend(memoryValues),
      stability: this.calculateStability(memoryValues),
      memory_leaks: this.detectMemoryLeaks(memoryValues)
    };
    
    return report;
  }

  calculateVariance(values) {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / values.length;
    return Math.round(variance);
  }

  calculateTrend(values) {
    if (values.length < 2) return 'insufficient_data';
    
    const firstHalf = values.slice(0, Math.floor(values.length / 2));
    const secondHalf = values.slice(Math.floor(values.length / 2));
    
    const firstAvg = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
    const secondAvg = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;
    
    const difference = secondAvg - firstAvg;
    const threshold = firstAvg * 0.1; // 10% 임계값
    
    if (Math.abs(difference) < threshold) {
      return 'stable';
    }
    
    return difference > 0 ? 'increasing' : 'decreasing';
  }

  calculateStability(values) {
    if (values.length < 10) return 'insufficient_data';
    
    const variance = this.calculateVariance(values);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const coefficientOfVariation = (Math.sqrt(variance) / mean) * 100;
    
    if (coefficientOfVariation < 5) return 'very_stable';
    if (coefficientOfVariation < 15) return 'stable';
    if (coefficientOfVariation < 30) return 'moderately_stable';
    if (coefficientOfVariation < 50) return 'unstable';
    return 'very_unstable';
  }

  detectMemoryLeaks(values) {
    if (values.length < 20) return { detected: false, reason: 'insufficient_data' };
    
    // 지속적인 메모리 증가 패턴 확인
    const segments = 5;
    const segmentSize = Math.floor(values.length / segments);
    const segmentAverages = [];
    
    for (let i = 0; i < segments; i++) {
      const start = i * segmentSize;
      const end = start + segmentSize;
      const segment = values.slice(start, end);
      const avg = segment.reduce((a, b) => a + b, 0) / segment.length;
      segmentAverages.push(avg);
    }
    
    // 연속적인 증가 패턴 확인
    let consistentIncrease = true;
    for (let i = 1; i < segmentAverages.length; i++) {
      if (segmentAverages[i] <= segmentAverages[i - 1]) {
        consistentIncrease = false;
        break;
      }
    }
    
    if (consistentIncrease) {
      const totalIncrease = segmentAverages[segmentAverages.length - 1] - segmentAverages[0];
      const increasePercentage = (totalIncrease / segmentAverages[0]) * 100;
      
      if (increasePercentage > 20) {
        return {
          detected: true,
          severity: increasePercentage > 50 ? 'high' : 'medium',
          increase_percentage: Math.round(increasePercentage),
          recommendation: '메모리 사용량이 지속적으로 증가하고 있습니다. 메모리 누수를 확인하세요.'
        };
      }
    }
    
    return { detected: false, reason: 'no_consistent_pattern' };
  }

  // 메모리 최적화 제안
  optimizeMemory() {
    const analysis = this.analyzeMemoryUsage();
    const optimizations = [];
    
    // 캐시 최적화
    if (analysis.cache.vsize > this.memoryThresholds.cache.warningThresholdMB * 1024 * 1024) {
      optimizations.push({
        type: 'cache',
        action: 'reduce_cache_size',
        impact: 'medium',
        description: '캐시 크기 축소',
        estimatedSavings: Math.round(analysis.cache.vsize / 1024 / 1024 * 0.3) + 'MB'
      });
    }
    
    // 힙 최적화
    if (analysis.current.heapUtilization > 0.8) {
      optimizations.push({
        type: 'heap',
        action: 'force_gc',
        impact: 'low',
        description: '가비지 컬렉션 강제 실행',
        estimatedSavings: Math.round(analysis.current.process.heapTotal * 0.1) + 'MB'
      });
    }
    
    // 모델 최적화
    if (analysis.current.totalMB > this.memoryThresholds.system.maxTotalMemoryMB * 0.7) {
      optimizations.push({
        type: 'models',
        action: 'unload_unused_models',
        impact: 'high',
        description: '사용하지 않는 모델 언로드',
        estimatedSavings: '2000-8000MB'
      });
    }
    
    // 임시 파일 최적화
    optimizations.push({
      type: 'files',
      action: 'cleanup_temp_files',
      impact: 'low',
      description: '임시 파일 정리',
      estimatedSavings: '10-100MB'
    });
    
    // 메모리 예약 최적화
    if (this.memoryReservations.size > 0) {
      optimizations.push({
        type: 'reservations',
        action: 'cleanup_expired_reservations',
        impact: 'low',
        description: '만료된 메모리 예약 정리',
        estimatedSavings: Array.from(this.memoryReservations.values())
          .reduce((sum, res) => sum + res.amount, 0) + 'MB'
      });
    }
    
    return optimizations;
  }

  // 자동 최적화 실행
  async executeOptimizations(optimizations) {
    const results = [];
    
    for (const opt of optimizations) {
      try {
        const beforeMemory = this.getCurrentMemoryUsage();
        let success = false;
        
        switch (opt.action) {
          case 'reduce_cache_size':
            await this.performCacheCleanup(0.3);
            success = true;
            break;
          case 'force_gc':
            if (global.gc) {
              global.gc();
              success = true;
            }
            break;
          case 'unload_unused_models':
            await this.requestModelUnload();
            success = true;
            break;
          case 'cleanup_temp_files':
            await this.cleanupTemporaryFiles();
            success = true;
            break;
          case 'cleanup_expired_reservations':
            this.cleanupExpiredReservations();
            success = true;
            break;
          default:
            success = false;
        }
        
        const afterMemory = this.getCurrentMemoryUsage();
        const actualSavings = beforeMemory.totalMB - afterMemory.totalMB;
        
        results.push({
          optimization: opt,
          success: success,
          actualSavings: actualSavings + 'MB',
          executionTime: Date.now()
        });
        
        this.logger.info(`최적화 실행: ${opt.description} - ${success ? '성공' : '실패'}`);
        
      } catch (error) {
        results.push({
          optimization: opt,
          success: false,
          error: error.message,
          executionTime: Date.now()
        });
        
        this.logger.error(`최적화 실행 실패: ${opt.description}`, error);
      }
    }
    
    return results;
  }

  // 자동 최적화 (전체)
  async performAutoOptimization() {
    this.logger.info('자동 메모리 최적화 시작');
    
    const optimizations = this.optimizeMemory();
    const results = await this.executeOptimizations(optimizations);
    
    const successCount = results.filter(r => r.success).length;
    const totalSavings = results.reduce((sum, r) => {
      const savings = parseInt(r.actualSavings) || 0;
      return sum + savings;
    }, 0);
    
    this.logger.info('자동 메모리 최적화 완료', {
      totalOptimizations: optimizations.length,
      successfulOptimizations: successCount,
      totalMemorySaved: totalSavings + 'MB'
    });
    
    return {
      optimizations: optimizations.length,
      successful: successCount,
      totalSavings: totalSavings + 'MB',
      results: results
    };
  }

  // 통계 및 상태 조회
  getMemoryStatistics() {
    const current = this.getCurrentMemoryUsage();
    
    return {
      current: current,
      statistics: {
        ...this.memoryStats,
        uptime: process.uptime(),
        monitoring_active: this.isMonitoring
      },
      thresholds: this.memoryThresholds,
      reservations: {
        active: this.memoryReservations.size,
        total_reserved: Array.from(this.memoryReservations.values())
          .reduce((sum, res) => sum + res.amount, 0)
      },
      cache: this.cache.getStats(),
      monitoring: {
        isActive: this.isMonitoring,
        intervalMs: this.monitoringInterval ? 10000 : null
      },
      profiling: {
        isActive: this.profilingData !== null,
        samples: this.profilingData ? this.profilingData.samples.length : 0
      }
    };
  }

  getDetailedReport() {
    const stats = this.getMemoryStatistics();
    const analysis = this.analyzeMemoryUsage();
    const optimizations = this.optimizeMemory();
    
    return {
      timestamp: new Date().toISOString(),
      summary: {
        current_usage: stats.current.totalMB + 'MB',
        peak_usage: stats.statistics.peak + 'MB',
        heap_utilization: Math.round(stats.current.heapUtilization * 100) + '%',
        system_utilization: Math.round(stats.current.systemUtilization * 100) + '%',
        warnings: stats.statistics.warnings,
        cleanups: stats.statistics.cleanups
      },
      detailed_stats: stats,
      analysis: analysis,
      optimizations: optimizations,
      health_status: this.getHealthStatus(stats.current),
      recommendations: analysis.recommendations
    };
  }

  getHealthStatus(currentUsage) {
    const memoryPercent = (currentUsage.totalMB / this.memoryThresholds.system.maxTotalMemoryMB) * 100;
    const heapPercent = currentUsage.heapUtilization * 100;
    
    if (memoryPercent > 95 || heapPercent > 95) {
      return { status: 'critical', message: '메모리 사용량이 매우 높습니다.' };
    } else if (memoryPercent > 85 || heapPercent > 85) {
      return { status: 'warning', message: '메모리 사용량이 높습니다.' };
    } else if (memoryPercent > 70 || heapPercent > 70) {
      return { status: 'caution', message: '메모리 사용량을 주의깊게 모니터링하세요.' };
    } else {
      return { status: 'healthy', message: '메모리 사용량이 정상입니다.' };
    }
  }

  // 설정 업데이트
  updateThresholds(newThresholds) {
    this.memoryThresholds = { ...this.memoryThresholds, ...newThresholds };
    this.logger.info('메모리 임계값 업데이트됨', newThresholds);
  }

  // 메모리 알림 설정
  setMemoryAlerts(config) {
    this.alertConfig = {
      warningThreshold: config.warningThreshold || 0.8,
      criticalThreshold: config.criticalThreshold || 0.9,
      emailNotifications: config.emailNotifications || false,
      webhookUrl: config.webhookUrl || null,
      ...config
    };
    
    this.logger.info('메모리 알림 설정 업데이트됨');
  }

  // 메모리 사용량 예측
  predictMemoryUsage(minutes = 30) {
    if (!this.profilingData || this.profilingData.samples.length < 10) {
      return {
        error: '예측에 필요한 데이터가 부족합니다.',
        recommendation: '메모리 프로파일링을 실행하여 데이터를 수집하세요.'
      };
    }
    
    const samples = this.profilingData.samples.slice(-20); // 최근 20개 샘플
    const values = samples.map(s => s.usage.totalMB);
    const trend = this.calculateTrend(values);
    
    let prediction = values[values.length - 1];
    let confidence = 'low';
    
    if (trend === 'increasing') {
      const recentGrowth = values[values.length - 1] - values[0];
      const growthRate = recentGrowth / values.length;
      const futureGrowth = growthRate * (minutes / (samples.length * 1000 / 60000)); // 분당 증가율
      
      prediction = values[values.length - 1] + futureGrowth;
      confidence = 'medium';
    } else if (trend === 'decreasing') {
      const recentDecrease = values[0] - values[values.length - 1];
      const decreaseRate = recentDecrease / values.length;
      const futureDecrease = decreaseRate * (minutes / (samples.length * 1000 / 60000));
      
      prediction = Math.max(0, values[values.length - 1] - futureDecrease);
      confidence = 'medium';
    } else {
      confidence = 'high';
    }
    
    return {
      current_usage: values[values.length - 1] + 'MB',
      predicted_usage: Math.round(prediction) + 'MB',
      prediction_time: minutes + ' minutes',
      trend: trend,
      confidence: confidence,
      warning: prediction > this.memoryThresholds.system.warningThresholdMB,
      critical: prediction > this.memoryThresholds.system.criticalThresholdMB
    };
  }

  // 정리 및 종료
  async destroy() {
    this.logger.info('MemoryManager 정리 시작');
    
    this.stopMemoryMonitoring();
    
    if (this.profilingData) {
      this.stopProfiling();
    }
    
    // 최종 메모리 상태 로그
    const finalStats = this.getMemoryStatistics();
    this.logger.info('최종 메모리 상태', {
      current_usage: finalStats.current.totalMB + 'MB',
      peak_usage: finalStats.statistics.peak + 'MB',
      total_warnings: finalStats.statistics.warnings,
      total_cleanups: finalStats.statistics.cleanups
    });
    
    this.logger.info('MemoryManager 정리 완료');
  }
}
