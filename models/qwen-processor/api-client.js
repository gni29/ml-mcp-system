// models/qwen-processor/api-client.js
import axios from 'axios';
import { Logger } from '../../utils/logger.js';

export class QwenApiClient {
  constructor() {
    this.logger = new Logger();
    this.ollamaEndpoint = 'http://localhost:11434';
    this.modelName = 'qwen2.5:14b';
    this.timeout = 30000; // 30초 타임아웃
    this.maxRetries = 3;
    this.retryDelay = 1000; // 1초
  }

  async initialize(config) {
    try {
      if (config && config.endpoint) {
        this.ollamaEndpoint = config.endpoint;
      }
      if (config && config.model) {
        this.modelName = config.model;
      }
      if (config && config.timeout) {
        this.timeout = config.timeout;
      }
      
      this.logger.info('Qwen API 클라이언트 초기화 완료');
    } catch (error) {
      this.logger.error('Qwen API 클라이언트 초기화 실패:', error);
      throw error;
    }
  }

  async checkConnection() {
    try {
      const response = await axios.get(`${this.ollamaEndpoint}/api/version`, {
        timeout: 5000
      });
      return {
        connected: true,
        version: response.data.version,
        endpoint: this.ollamaEndpoint
      };
    } catch (error) {
      this.logger.error('Ollama 연결 확인 실패:', error);
      return {
        connected: false,
        error: error.message,
        endpoint: this.ollamaEndpoint
      };
    }
  }

  async getAvailableModels() {
    try {
      const response = await axios.get(`${this.ollamaEndpoint}/api/tags`);
      return response.data.models.map(model => ({
        name: model.name,
        size: model.size,
        modified_at: model.modified_at
      }));
    } catch (error) {
      this.logger.error('모델 목록 조회 실패:', error);
      throw error;
    }
  }

  async isModelAvailable(modelName = this.modelName) {
    try {
      const models = await this.getAvailableModels();
      return models.some(model => model.name === modelName);
    } catch (error) {
      this.logger.error('모델 가용성 확인 실패:', error);
      return false;
    }
  }

  async generateResponse(prompt, options = {}) {
    const requestOptions = {
      model: options.model || this.modelName,
      prompt: prompt,
      stream: false,
      options: {
        temperature: options.temperature || 0.3,
        max_tokens: options.max_tokens || 2048,
        top_p: options.top_p || 0.9,
        top_k: options.top_k || 40,
        repeat_penalty: options.repeat_penalty || 1.1,
        num_ctx: options.num_ctx || 8192,
        num_thread: options.num_thread || 8
      }
    };

    try {
      const response = await this.makeRequestWithRetry(
        'POST',
        `${this.ollamaEndpoint}/api/generate`,
        requestOptions,
        this.timeout
      );

      if (response.data && response.data.response) {
        return {
          success: true,
          response: response.data.response,
          done: response.data.done,
          total_duration: response.data.total_duration,
          load_duration: response.data.load_duration,
          prompt_eval_count: response.data.prompt_eval_count,
          prompt_eval_duration: response.data.prompt_eval_duration,
          eval_count: response.data.eval_count,
          eval_duration: response.data.eval_duration
        };
      } else {
        throw new Error('응답 데이터가 유효하지 않습니다.');
      }
    } catch (error) {
      this.logger.error('Qwen 응답 생성 실패:', error);
      return {
        success: false,
        error: error.message,
        response: null
      };
    }
  }

  async generateStreamResponse(prompt, options = {}) {
    const requestOptions = {
      model: options.model || this.modelName,
      prompt: prompt,
      stream: true,
      options: {
        temperature: options.temperature || 0.3,
        max_tokens: options.max_tokens || 2048,
        top_p: options.top_p || 0.9,
        top_k: options.top_k || 40,
        repeat_penalty: options.repeat_penalty || 1.1,
        num_ctx: options.num_ctx || 8192,
        num_thread: options.num_thread || 8
      }
    };

    try {
      const response = await axios.post(
        `${this.ollamaEndpoint}/api/generate`,
        requestOptions,
        {
          timeout: this.timeout,
          responseType: 'stream'
        }
      );

      return response.data;
    } catch (error) {
      this.logger.error('Qwen 스트림 응답 생성 실패:', error);
      throw error;
    }
  }

  async makeRequestWithRetry(method, url, data, timeout = this.timeout) {
    let lastError;
    
    for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
      try {
        const config = {
          method: method,
          url: url,
          timeout: timeout,
          headers: {
            'Content-Type': 'application/json'
          }
        };

        if (data) {
          config.data = data;
        }

        const response = await axios(config);
        return response;
      } catch (error) {
        lastError = error;
        
        if (attempt < this.maxRetries) {
          this.logger.warn(`API 요청 실패 (${attempt}/${this.maxRetries}), 재시도 중...`);
          await this.delay(this.retryDelay * attempt);
        }
      }
    }

    throw lastError;
  }

  async testConnection() {
    try {
      const connectionTest = await this.checkConnection();
      if (!connectionTest.connected) {
        return {
          success: false,
          error: 'Ollama 서비스에 연결할 수 없습니다.',
          details: connectionTest.error
        };
      }

      const modelAvailable = await this.isModelAvailable();
      if (!modelAvailable) {
        return {
          success: false,
          error: `모델 ${this.modelName}이 설치되지 않았습니다.`,
          details: 'ollama pull 명령어로 모델을 설치하세요.'
        };
      }

      // 간단한 테스트 요청
      const testResponse = await this.generateResponse('Hello', {
        max_tokens: 10,
        temperature: 0.1
      });

      if (testResponse.success) {
        return {
          success: true,
          message: 'Qwen API 클라이언트 테스트 성공',
          response_time: testResponse.total_duration,
          model: this.modelName
        };
      } else {
        return {
          success: false,
          error: 'API 테스트 실패',
          details: testResponse.error
        };
      }
    } catch (error) {
      this.logger.error('연결 테스트 실패:', error);
      return {
        success: false,
        error: '연결 테스트 중 오류 발생',
        details: error.message
      };
    }
  }

  async delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async getModelInfo(modelName = this.modelName) {
    try {
      const response = await axios.post(`${this.ollamaEndpoint}/api/show`, {
        name: modelName
      });
      
      return {
        name: response.data.name,
        size: response.data.size,
        digest: response.data.digest,
        modified_at: response.data.modified_at,
        template: response.data.template,
        parameters: response.data.parameters,
        modelfile: response.data.modelfile
      };
    } catch (error) {
      this.logger.error('모델 정보 조회 실패:', error);
      throw error;
    }
  }

  async unloadModel(modelName = this.modelName) {
    try {
      const response = await axios.post(`${this.ollamaEndpoint}/api/generate`, {
        model: modelName,
        keep_alive: 0
      });
      
      this.logger.info(`모델 언로드 완료: ${modelName}`);
      return true;
    } catch (error) {
      this.logger.error('모델 언로드 실패:', error);
      return false;
    }
  }

  getEndpoint() {
    return this.ollamaEndpoint;
  }

  getModelName() {
    return this.modelName;
  }

  setTimeout(timeout) {
    this.timeout = timeout;
  }

  setMaxRetries(maxRetries) {
    this.maxRetries = maxRetries;
  }
}
