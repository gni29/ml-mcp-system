// core/model-manager.js
import { Logger } from '../utils/logger.js';
import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';

export class ModelManager {
  constructor() {
    this.logger = new Logger();
    this.ollamaEndpoint = process.env.OLLAMA_ENDPOINT || 'http://localhost:11434';
    this.models = new Map();
    this.modelConfigs = new Map();
    this.loadedModels = new Set();
    this.lastUsed = new Map();
    this.isInitialized = false;
    
    // 기본 모델 설정
    this.defaultModels = {
      router: process.env.ROUTER_MODEL || 'llama3.2:3b',
      processor: process.env.PROCESSOR_MODEL || 'qwen2.5:14b'
    };
    
    // 메모리 관리
    this.memoryThresholds = {
      router: 6000, // MB
      processor: 28000 // MB
    };
    
    // 자동 언로드 타이머
    this.unloadTimers = new Map();
    this.autoUnloadTimeout = parseInt(process.env.AUTO_UNLOAD_TIMEOUT || '600000'); // 10분
    
    // 메모리 압박 이벤트 리스너
    this.setupMemoryPressureHandlers();
  }

  setupMemoryPressureHandlers() {
    process.on('memory-pressure', async (event) => {
      if (event.source !== 'memory-manager') return;
      
      this.logger.warn('메모리 압박 이벤트 수신:', event);
      
      switch (event.action) {
        case 'unload_models':
          await this.unloadUnusedModels();
          break;
        case 'emergency_cleanup':
          await this.emergencyCleanup();
          break;
      }
    });
  }

  async initialize() {
    try {
      this.logger.info('모델 매니저 초기화 시작...');
      
      // Ollama 서비스 연결 확인
      await this.checkOllamaConnection();
      
      // 모델 설정 파일 로드
      await this.loadModelConfigs();
      
      // 사용 가능한 모델 목록 확인
      await this.checkAvailableModels();
      
      // 라우터 모델 미리 로드 (가장 자주 사용됨)
      await this.preloadRouterModel();
      
      this.isInitialized = true;
      this.logger.info('✅ 모델 매니저 초기화 완료');
      
    } catch (error) {
      this.logger.error('모델 매니저 초기화 실패:', error);
      throw error;
    }
  }

  async checkOllamaConnection() {
    try {
      const response = await axios.get(`${this.ollamaEndpoint}/api/version`, {
        timeout: 5000
      });
      
      this.logger.info(`Ollama 연결 확인: 버전 ${response.data.version || 'Unknown'}`);
      return true;
      
    } catch (error) {
      throw new Error(`Ollama 서비스에 연결할 수 없습니다. 'ollama serve' 명령으로 시작해주세요.`);
    }
  }

  async loadModelConfigs() {
    try {
      const configPath = './models/model-configs.json';
      
      try {
        const configData = await fs.readFile(configPath, 'utf-8');
        const configs = JSON.parse(configData);
        
        Object.entries(configs).forEach(([modelType, config]) => {
          this.modelConfigs.set(modelType, config);
        });
        
        this.logger.info(`모델 설정 로드 완료: ${this.modelConfigs.size}개 설정`);
        
      } catch (fileError) {
        // 기본 설정 생성
        await this.createDefaultModelConfigs(configPath);
      }
      
    } catch (error) {
      this.logger.error('모델 설정 로드 실패:', error);
      throw error;
    }
  }

  async createDefaultModelConfigs(configPath) {
    const defaultConfigs = {
      router: {
        name: this.defaultModels.router,
        description: '빠른 응답을 위한 라우터 모델',
        maxMemoryMB: 6000,
        temperature: 0.7,
        maxTokens: 500,
        autoUnload: false
      },
      processor: {
        name: this.defaultModels.processor,
        description: '복잡한 작업을 위한 프로세서 모델',
        maxMemoryMB: 28000,
        temperature: 0.3,
        maxTokens: 2000,
        autoUnload: true
      }
    };

    try {
      // 디렉토리 생성
      await fs.mkdir('./models', { recursive: true });
      
      // 기본 설정 파일 생성
      await fs.writeFile(configPath, JSON.stringify(defaultConfigs, null, 2));
      
      // 설정을 메모리에 로드
      Object.entries(defaultConfigs).forEach(([modelType, config]) => {
        this.modelConfigs.set(modelType, config);
      });
      
      this.logger.info('기본 모델 설정 생성 완료');
      
    } catch (error) {
      this.logger.error('기본 설정 생성 실패:', error);
    }
  }

  async checkAvailableModels() {
    try {
      const response = await axios.get(`${this.ollamaEndpoint}/api/tags`);
      const installedModels = response.data.models || [];
      
      this.logger.info(`설치된 모델: ${installedModels.length}개`);
      
      // 필요한 모델이 설치되어 있는지 확인
      const requiredModels = [this.defaultModels.router, this.defaultModels.processor];
      const missingModels = [];
      
      for (const modelName of requiredModels) {
        const isInstalled = installedModels.some(model => 
          model.name === modelName || model.name.startsWith(modelName.split(':')[0])
        );
        
        if (!isInstalled) {
          missingModels.push(modelName);
        }
      }
      
      if (missingModels.length > 0) {
        this.logger.warn(`누락된 모델: ${missingModels.join(', ')}`);
        this.logger.warn('다음 명령으로 설치하세요:');
        missingModels.forEach(model => {
          this.logger.warn(`  ollama pull ${model}`);
        });
      }
      
      return installedModels;
      
    } catch (error) {
      this.logger.error('모델 목록 확인 실패:', error);
      throw error;
    }
  }

  async preloadRouterModel() {
    try {
      await this.loadModel('router');
      this.logger.info('라우터 모델 사전 로드 완료');
    } catch (error) {
      this.logger.warn('라우터 모델 사전 로드 실패:', error.message);
    }
  }

  async loadModel(modelType) {
    try {
      if (this.loadedModels.has(modelType)) {
        this.updateLastUsed(modelType);
        return true;
      }

      const config = this.modelConfigs.get(modelType);
      if (!config) {
        throw new Error(`모델 설정을 찾을 수 없습니다: ${modelType}`);
      }

      this.logger.info(`모델 로딩 시작: ${config.name}`);

      // 모델 로드 요청
      const loadResponse = await axios.post(`${this.ollamaEndpoint}/api/chat`, {
        model: config.name,
        messages: [
          {
            role: 'user',
            content: 'Hello'
          }
        ],
        stream: false,
        options: {
          temperature: config.temperature || 0.7,
          num_predict: 1
        }
      }, {
        timeout: 30000
      });

      if (loadResponse.data) {
        this.loadedModels.add(modelType);
        this.updateLastUsed(modelType);
        this.setupAutoUnload(modelType, config);
        
        this.logger.info(`✅ 모델 로드 완료: ${config.name}`);
        return true;
      }

    } catch (error) {
      this.logger.error(`모델 로드 실패 [${modelType}]:`, error.message);
      throw error;
    }
  }

  async unloadModel(modelType) {
    try {
      if (!this.loadedModels.has(modelType)) {
        return true;
      }

      const config = this.modelConfigs.get(modelType);
      if (!config) {
        return false;
      }

      // Ollama에서 모델 언로드 (실제로는 메모리에서 제거되지 않을 수 있음)
      this.loadedModels.delete(modelType);
      this.lastUsed.delete(modelType);
      
      // 자동 언로드 타이머 정리
      if (this.unloadTimers.has(modelType)) {
        clearTimeout(this.unloadTimers.get(modelType));
        this.unloadTimers.delete(modelType);
      }

      this.logger.info(`모델 언로드: ${config.name}`);
      return true;

    } catch (error) {
      this.logger.error(`모델 언로드 실패 [${modelType}]:`, error);
      return false;
    }
  }

  async queryModel(modelType, prompt, options = {}) {
    try {
      // 모델 로드 확인
      await this.loadModel(modelType);
      
      const config = this.modelConfigs.get(modelType);
      if (!config) {
        throw new Error(`모델 설정을 찾을 수 없습니다: ${modelType}`);
      }

      const requestOptions = {
        temperature: options.temperature || config.temperature || 0.7,
        num_predict: options.max_tokens || config.maxTokens || 500,
        top_p: options.top_p || 0.9,
        repeat_penalty: options.repeat_penalty || 1.1
      };

      this.logger.debug(`모델 쿼리 시작 [${modelType}]:`, {
        promptLength: prompt.length,
        options: requestOptions
      });

      const startTime = Date.now();

      const response = await axios.post(`${this.ollamaEndpoint}/api/chat`, {
        model: config.name,
        messages: [
          {
            role: 'user',
            content: prompt
          }
        ],
        stream: false,
        options: requestOptions
      }, {
        timeout: 60000 // 1분 타임아웃
      });

      const responseTime = Date.now() - startTime;
      this.updateLastUsed(modelType);

      if (response.data && response.data.message) {
        const result = response.data.message.content;
        
        this.logger.debug(`모델 응답 완료 [${modelType}]:`, {
          responseTime: `${responseTime}ms`,
          responseLength: result.length
        });

        return result;
      } else {
        throw new Error('모델에서 유효한 응답을 받지 못했습니다.');
      }

    } catch (error) {
      this.logger.error(`모델 쿼리 실패 [${modelType}]:`, error.message);
      throw error;
    }
  }

  async loadRouterModel() {
    return await this.loadModel('router');
  }

  async loadProcessorModel() {
    return await this.loadModel('processor');
  }

  async queryRouterModel(prompt, options = {}) {
    return await this.queryModel('router', prompt, options);
  }

  async queryProcessorModel(prompt, options = {}) {
    return await this.queryModel('processor', prompt, options);
  }

  updateLastUsed(modelType) {
    this.lastUsed.set(modelType, Date.now());
    
    // 자동 언로드 타이머 재설정
    const config = this.modelConfigs.get(modelType);
    if (config && config.autoUnload) {
      this.setupAutoUnload(modelType, config);
    }
  }

  setupAutoUnload(modelType, config) {
    // 기존 타이머 정리
    if (this.unloadTimers.has(modelType)) {
      clearTimeout(this.unloadTimers.get(modelType));
    }

    // 자동 언로드가 비활성화된 경우
    if (!config.autoUnload) {
      return;
    }

    // 새 타이머 설정
    const timer = setTimeout(async () => {
      const lastUsedTime = this.lastUsed.get(modelType) || 0;
      const timeSinceLastUse = Date.now() - lastUsedTime;

      // 최근에 사용되었다면 타이머 재설정
      if (timeSinceLastUse < this.autoUnloadTimeout) {
        this.setupAutoUnload(modelType, config);
        return;
      }

      // 모델 언로드
      await this.unloadModel(modelType);
      this.logger.info(`자동 언로드: ${config.name} (비활성 시간: ${Math.round(timeSinceLastUse / 1000)}초)`);
      
    }, this.autoUnloadTimeout);

    this.unloadTimers.set(modelType, timer);
  }

  async unloadUnusedModels() {
    const results = [];
    const currentTime = Date.now();

    for (const modelType of this.loadedModels) {
      const lastUsedTime = this.lastUsed.get(modelType) || 0;
      const timeSinceLastUse = currentTime - lastUsedTime;

      // 5분 이상 사용되지 않은 모델 언로드
      if (timeSinceLastUse > 300000) {
        const success = await this.unloadModel(modelType);
        results.push({
          modelType,
          success,
          timeSinceLastUse: Math.round(timeSinceLastUse / 1000)
        });
      }
    }

    if (results.length > 0) {
      this.logger.info('사용되지 않는 모델 언로드 완료:', results);
    }

    return results;
  }

  async emergencyCleanup() {
    this.logger.warn('긴급 모델 정리 시작...');
    
    const results = [];
    
    // 프로세서 모델부터 언로드 (더 많은 메모리 사용)
    if (this.loadedModels.has('processor')) {
      const success = await this.unloadModel('processor');
      results.push({ model: 'processor', success });
    }

    // 필요시 라우터 모델도 언로드
    if (this.loadedModels.size > 1 && this.loadedModels.has('router')) {
      const success = await this.unloadModel('router');
      results.push({ model: 'router', success });
    }

    this.logger.warn('긴급 정리 완료:', results);
    return results;
  }

  getLoadedModels() {
    const result = {};
    
    for (const modelType of this.loadedModels) {
      const config = this.modelConfigs.get(modelType);
      const lastUsedTime = this.lastUsed.get(modelType);
      
      result[modelType] = {
        name: config?.name || 'Unknown',
        lastUsed: lastUsedTime,
        config: config
      };
    }
    
    return result;
  }

  getModelStatus() {
    return {
      initialized: this.isInitialized,
      loaded_models: Array.from(this.loadedModels),
      model_count: this.loadedModels.size,
      available_configs: Array.from(this.modelConfigs.keys()),
      last_used: Object.fromEntries(this.lastUsed),
      auto_unload_timeout: this.autoUnloadTimeout
    };
  }

  async getMemoryUsage() {
    // Ollama API를 통한 메모리 사용량 조회 (가능한 경우)
    try {
      const response = await axios.get(`${this.ollamaEndpoint}/api/ps`);
      const runningModels = response.data.models || [];
      
      let totalMemory = 0;
      const modelMemory = {};
      
      runningModels.forEach(model => {
        const memory = model.size_vram || model.size || 0;
        totalMemory += memory;
        modelMemory[model.name] = memory;
      });
      
      return {
        total_memory_bytes: totalMemory,
        models: modelMemory,
        running_count: runningModels.length
      };
      
    } catch (error) {
      this.logger.debug('메모리 사용량 조회 실패:', error.message);
      return {
        total_memory_bytes: 0,
        models: {},
        running_count: this.loadedModels.size
      };
    }
  }

  async validateModel(modelName) {
    try {
      const response = await axios.post(`${this.ollamaEndpoint}/api/show`, {
        name: modelName
      });
      
      return response.data ? true : false;
      
    } catch (error) {
      return false;
    }
  }

  async listAvailableModels() {
    try {
      const response = await axios.get(`${this.ollamaEndpoint}/api/tags`);
      return response.data.models || [];
    } catch (error) {
      this.logger.error('모델 목록 조회 실패:', error);
      return [];
    }
  }

  async pullModel(modelName) {
    try {
      this.logger.info(`모델 다운로드 시작: ${modelName}`);
      
      const response = await axios.post(`${this.ollamaEndpoint}/api/pull`, {
        name: modelName
      }, {
        timeout: 300000 // 5분 타임아웃
      });
      
      this.logger.info(`모델 다운로드 완료: ${modelName}`);
      return true;
      
    } catch (error) {
      this.logger.error(`모델 다운로드 실패 [${modelName}]:`, error.message);
      return false;
    }
  }

  async cleanup() {
    try {
      this.logger.info('모델 매니저 정리 시작...');
      
      // 모든 타이머 정리
      for (const timer of this.unloadTimers.values()) {
        clearTimeout(timer);
      }
      this.unloadTimers.clear();
      
      // 모든 모델 언로드
      const unloadPromises = Array.from(this.loadedModels).map(modelType => 
        this.unloadModel(modelType)
      );
      await Promise.allSettled(unloadPromises);
      
      // 상태 초기화
      this.loadedModels.clear();
      this.lastUsed.clear();
      this.isInitialized = false;
      
      this.logger.info('✅ 모델 매니저 정리 완료');
      
    } catch (error) {
      this.logger.error('모델 매니저 정리 실패:', error);
    }
  }

  // 개발/디버깅용 메서드들
  async debugInfo() {
    const info = {
      initialized: this.isInitialized,
      loaded_models: this.getLoadedModels(),
      model_configs: Object.fromEntries(this.modelConfigs),
      memory_usage: await this.getMemoryUsage(),
      auto_unload_timers: this.unloadTimers.size,
      ollama_endpoint: this.ollamaEndpoint
    };
    
    return info;
  }

  async testConnection() {
    try {
      await this.checkOllamaConnection();
      return { success: true, message: 'Ollama 연결 성공' };
    } catch (error) {
      return { success: false, message: error.message };
    }
  }

  async testModel(modelType) {
    try {
      const testPrompt = 'Hello, please respond with "Test successful"';
      const response = await this.queryModel(modelType, testPrompt);
      
      return { 
        success: true, 
        model: modelType,
        response: response.substring(0, 100) + (response.length > 100 ? '...' : '')
      };
    } catch (error) {
      return { 
        success: false, 
        model: modelType, 
        error: error.message 
      };
    }
  }
}