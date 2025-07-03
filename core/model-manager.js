import axios from 'axios';
import { Logger } from '../utils/logger.js';
import fs from 'fs/promises';

export class ModelManager {
  constructor() {
    this.logger = new Logger();
    this.models = new Map();
    this.config = null;
    this.ollamaEndpoint = 'http://localhost:11434';
  }

  async initialize() {
    try {
      // 설정 로드
      await this.loadConfig();
      
      // Ollama 서비스 확인
      await this.checkOllamaService();
      
      // 기본 모델 로드
      await this.loadRouterModel();
      
      this.logger.info('모델 매니저 초기화 완료');
      
    } catch (error) {
      this.logger.error('모델 매니저 초기화 실패:', error);
      throw error;
    }
  }

  async loadConfig() {
    const configPath = './models/model-configs.json';
    const configData = await fs.readFile(configPath, 'utf-8');
    this.config = JSON.parse(configData);
  }

  async checkOllamaService() {
    try {
      const response = await axios.get(`${this.ollamaEndpoint}/api/version`);
      this.logger.info('Ollama 서비스 연결 확인:', response.data);
    } catch (error) {
      throw new Error('Ollama 서비스에 연결할 수 없습니다. ollama serve가 실행 중인지 확인하세요.');
    }
  }

  async loadRouterModel() {
    const routerConfig = this.config['llama-router'];
    
    try {
      // 모델 로드 확인
      await this.ensureModelLoaded(routerConfig.model);
      
      this.models.set('router', {
        name: routerConfig.model,
        config: routerConfig,
        status: 'loaded',
        lastUsed: Date.now()
      });
      
      this.logger.info(`라우터 모델 로드됨: ${routerConfig.model}`);
      
    } catch (error) {
      this.logger.error('라우터 모델 로드 실패:', error);
      throw error;
    }
  }

  async loadProcessorModel() {
    const processorConfig = this.config['qwen-processor'];
    
    try {
      await this.ensureModelLoaded(processorConfig.model);
      
      this.models.set('processor', {
        name: processorConfig.model,
        config: processorConfig,
        status: 'loaded',
        lastUsed: Date.now()
      });
      
      this.logger.info(`프로세서 모델 로드됨: ${processorConfig.model}`);
      
    } catch (error) {
      this.logger.error('프로세서 모델 로드 실패:', error);
      throw error;
    }
  }

  async ensureModelLoaded(modelName) {
    try {
      // 모델 목록 확인
      const response = await axios.get(`${this.ollamaEndpoint}/api/tags`);
      const availableModels = response.data.models.map(m => m.name);
      
      if (!availableModels.includes(modelName)) {
        throw new Error(`모델 ${modelName}이 설치되지 않았습니다. 'ollama pull ${modelName}' 실행하세요.`);
      }
      
    } catch (error) {
      if (error.response?.status === 404) {
        throw new Error(`모델 ${modelName}을 찾을 수 없습니다.`);
      }
      throw error;
    }
  }

  async queryModel(modelType, prompt, options = {}) {
    const model = this.models.get(modelType);
    if (!model) {
      throw new Error(`모델 ${modelType}이 로드되지 않았습니다.`);
    }

    try {
      const response = await axios.post(`${this.ollamaEndpoint}/api/generate`, {
        model: model.name,
        prompt: prompt,
        stream: false,
        ...model.config,
        ...options
      });

      // 사용 시간 업데이트
      model.lastUsed = Date.now();
      
      return response.data.response;
      
    } catch (error) {
      this.logger.error(`모델 쿼리 실패 (${modelType}):`, error);
      throw error;
    }
  }

  async getModelStatus() {
    const status = {};
    
    for (const [type, model] of this.models) {
      status[type] = {
        name: model.name,
        status: model.status,
        lastUsed: new Date(model.lastUsed).toISOString(),
        memoryUsage: await this.getModelMemoryUsage(model.name)
      };
    }
    
    return status;
  }

  async getModelMemoryUsage(modelName) {
    try {
      const response = await axios.get(`${this.ollamaEndpoint}/api/ps`);
      const runningModels = response.data.models || [];
      
      const modelInfo = runningModels.find(m => m.name === modelName);
      return modelInfo ? modelInfo.size_vram || 0 : 0;
      
    } catch (error) {
      return 0;
    }
  }

  async optimizeMemory() {
    const now = Date.now();
    const maxIdleTime = 10 * 60 * 1000; // 10분

    for (const [type, model] of this.models) {
      if (type !== 'router' && (now - model.lastUsed) > maxIdleTime) {
        await this.unloadModel(type);
      }
    }
  }

  async unloadModel(modelType) {
    if (this.models.has(modelType)) {
      this.models.delete(modelType);
      this.logger.info(`모델 언로드됨: ${modelType}`);
    }
  }
}
