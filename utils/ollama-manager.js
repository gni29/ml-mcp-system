// utils/ollama-manager.js
import axios from 'axios';
import { Logger } from './logger.js';

export class OllamaManager {
  constructor(endpoint = 'http://localhost:11434') {
    this.endpoint = endpoint;
    this.logger = new Logger();
  }

  async checkConnection() {
    try {
      const response = await axios.get(`${this.endpoint}/api/version`, {
        timeout: 5000
      });
      this.logger.info('Ollama 연결 확인됨:', response.data);
      return true;
    } catch (error) {
      this.logger.error('Ollama 연결 실패:', error.message);
      return false;
    }
  }

  async listModels() {
    try {
      const response = await axios.get(`${this.endpoint}/api/tags`);
      return response.data.models || [];
    } catch (error) {
      this.logger.error('모델 목록 조회 실패:', error);
      return [];
    }
  }

  async isModelAvailable(modelName) {
    const models = await this.listModels();
    return models.some(model => model.name === modelName);
  }

  async pullModel(modelName) {
    try {
      this.logger.info(`모델 다운로드 시작: ${modelName}`);
      
      const response = await axios.post(`${this.endpoint}/api/pull`, {
        name: modelName
      }, {
        timeout: 600000 // 10분 타임아웃
      });

      this.logger.info(`모델 다운로드 완료: ${modelName}`);
      return true;
    } catch (error) {
      this.logger.error(`모델 다운로드 실패: ${modelName}`, error);
      return false;
    }
  }

  async getRunningModels() {
    try {
      const response = await axios.get(`${this.endpoint}/api/ps`);
      return response.data.models || [];
    } catch (error) {
      this.logger.error('실행 중인 모델 조회 실패:', error);
      return [];
    }
  }

  async generateResponse(modelName, prompt, options = {}) {
    try {
      const response = await axios.post(`${this.endpoint}/api/generate`, {
        model: modelName,
        prompt: prompt,
        stream: false,
        ...options
      }, {
        timeout: 60000 // 1분 타임아웃
      });

      return response.data.response;
    } catch (error) {
      this.logger.error(`응답 생성 실패: ${modelName}`, error);
      throw error;
    }
  }

  async warmupModel(modelName) {
    try {
      this.logger.info(`모델 워밍업: ${modelName}`);
      await this.generateResponse(modelName, "Hello", { max_tokens: 10 });
      this.logger.info(`모델 워밍업 완료: ${modelName}`);
    } catch (error) {
      this.logger.warn(`모델 워밍업 실패: ${modelName}`, error);
    }
  }
}
