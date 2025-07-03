// utils/cache-manager.js
import NodeCache from 'node-cache';
import { Logger } from './logger.js';

export class CacheManager {
  constructor() {
    this.cache = new NodeCache({
      stdTTL: 3600, // 기본 1시간
      checkperiod: 600, // 10분마다 만료된 키 체크
      useClones: false
    });
    
    this.logger = new Logger();
    this.setupEventListeners();
  }

  setupEventListeners() {
    this.cache.on('set', (key, value) => {
      this.logger.debug(`캐시 저장: ${key}`);
    });

    this.cache.on('del', (key, value) => {
      this.logger.debug(`캐시 삭제: ${key}`);
    });

    this.cache.on('expired', (key, value) => {
      this.logger.debug(`캐시 만료: ${key}`);
    });
  }

  set(key, value, ttl = undefined) {
    return this.cache.set(key, value, ttl);
  }

  get(key) {
    return this.cache.get(key);
  }

  del(key) {
    return this.cache.del(key);
  }

  has(key) {
    return this.cache.has(key);
  }

  // 모델 응답 캐싱
  cacheModelResponse(modelName, prompt, response, ttl = 3600) {
    const key = this.generateCacheKey(modelName, prompt);
    this.set(key, response, ttl);
  }

  getCachedModelResponse(modelName, prompt) {
    const key = this.generateCacheKey(modelName, prompt);
    return this.get(key);
  }

  generateCacheKey(modelName, prompt) {
    // 간단한 해시 생성
    const hash = prompt.split('').reduce((a, b) => {
      a = ((a << 5) - a) + b.charCodeAt(0);
      return a & a;
    }, 0);
    
    return `${modelName}_${Math.abs(hash)}`;
  }

  // 통계
  getStats() {
    return {
      keys: this.cache.getStats().keys,
      hits: this.cache.getStats().hits,
      misses: this.cache.getStats().misses,
      ksize: this.cache.getStats().ksize,
      vsize: this.cache.getStats().vsize
    };
  }

  // 캐시 정리
  flush() {
    this.cache.flushAll();
    this.logger.info('모든 캐시 삭제됨');
  }
}
