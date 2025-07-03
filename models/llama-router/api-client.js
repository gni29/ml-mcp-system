// models/llama-router/api-client.js
import axios from 'axios';
import { Logger } from '../../utils/logger.js';
import { CacheManager } from '../../utils/cache-manager.js';
import fs from 'fs/promises';

export class LlamaRouterClient {
  constructor() {
    this.logger = new Logger();
    this.cache = new CacheManager();
    this.config = null;
    this.endpoint = 'http://localhost:11434';
    this.modelName = 'llama3.2:3b';
    this.isLoaded = false;
    this.lastUsed = null;
    this.requestCount = 0;
    this.errorCount = 0;
    this.responseTimeHistory = [];
    this.maxHistorySize = 100;
  }

  async initialize() {
    try {
      await this.loadConfig();
      await this.ensureModelAvailable();
      await this.warmupModel();
      
      this.logger.info('LlamaRouterClient 초기화 완료', {
        model: this.modelName,
        endpoint: this.endpoint
      });
      
    } catch (error) {
      this.logger.error('LlamaRouterClient 초기화 실패:', error);
      throw error;
    }
  }

  async loadConfig() {
    try {
      const configData = await fs.readFile('./models/llama-router/config.json', 'utf-8');
      this.config = JSON.parse(configData);
      
      this.endpoint = this.config.endpoint || this.endpoint;
      this.modelName = this.config.model || this.modelName;
      
    } catch (error) {
      this.logger.warn('라우터 설정 로드 실패, 기본값 사용:', error);
      this.config = this.getDefaultConfig();
    }
  }

  getDefaultConfig() {
    return {
      model: 'llama3.2:3b',
      endpoint: 'http://localhost:11434',
      temperature: 0.1,
      max_tokens: 512,
      timeout: 30000,
      retry_attempts: 3,
      cache_ttl: 300000 // 5분
    };
  }

  async ensureModelAvailable() {
    try {
      // 모델 목록 확인
      const response = await axios.get(`${this.endpoint}/api/tags`, {
        timeout: 5000
      });
      
      const availableModels = response.data.models || [];
      const modelExists = availableModels.some(model => model.name === this.modelName);
      
      if (!modelExists) {
        throw new Error(`모델 ${this.modelName}이 Ollama에 설치되지 않았습니다. 'ollama pull ${this.modelName}' 명령으로 설치하세요.`);
      }
      
      this.logger.info('라우터 모델 확인 완료', { model: this.modelName });
      
    } catch (error) {
      if (error.code === 'ECONNREFUSED') {
        throw new Error('Ollama 서비스가 실행되지 않고 있습니다. "ollama serve" 명령으로 시작하세요.');
      }
      throw error;
    }
  }

  async warmupModel() {
    try {
      this.logger.info('라우터 모델 워밍업 시작');
      
      const warmupPrompt = "Hello";
      const startTime = Date.now();
      
      await this.generateResponse(warmupPrompt, {
        max_tokens: 5,
        temperature: 0.1,
        cache: false // 워밍업은 캐시하지 않음
      });
      
      const warmupTime = Date.now() - startTime;
      this.isLoaded = true;
      
      this.logger.info('라우터 모델 워밍업 완료', {
        warmupTime: `${warmupTime}ms`
      });
      
    } catch (error) {
      this.logger.warn('라우터 모델 워밍업 실패:', error);
      // 워밍업 실패해도 계속 진행
    }
  }

  async generateResponse(prompt, options = {}) {
    const startTime = Date.now();
    
    try {
      // 옵션 설정
      const requestOptions = {
        ...this.config,
        ...options,
        prompt: prompt,
        model: this.modelName,
        stream: false
      };

      // 캐시 확인
      if (options.cache !== false) {
        const cacheKey = this.generateCacheKey(prompt, requestOptions);
        const cachedResponse = this.cache.get(cacheKey);
        
        if (cachedResponse) {
          this.logger.debug('캐시된 응답 반환', { prompt: prompt.substring(0, 50) });
          return cachedResponse;
        }
      }

      // 재시도 로직
      let lastError;
      const maxRetries = options.retry_attempts || this.config.retry_attempts || 3;
      
      for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
          const response = await this.makeRequest(requestOptions, attempt);
          
          // 성공적인 응답 처리
          const result = this.processResponse(response, startTime);
          
          // 캐시 저장
          if (options.cache !== false && result.response) {
            const cacheKey = this.generateCacheKey(prompt, requestOptions);
            const cacheTtl = options.cache_ttl || this.config.cache_ttl || 300000;
            this.cache.set(cacheKey, result.response, cacheTtl);
          }
          
          return result.response;
          
        } catch (error) {
          lastError = error;
          
          if (attempt < maxRetries) {
            const delay = this.calculateRetryDelay(attempt);
            this.logger.warn(`라우터 요청 실패 (${attempt}/${maxRetries}), ${delay}ms 후 재시도:`, error.message);
            await this.sleep(delay);
          }
        }
      }
      
      throw lastError;
      
    } catch (error) {
      this.handleError(error, prompt, startTime);
      throw error;
    }
  }

  async makeRequest(requestOptions, attempt) {
    const timeout = requestOptions.timeout || 30000;
    
    const response = await axios.post(`${this.endpoint}/api/generate`, requestOptions, {
      timeout: timeout,
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'ML-MCP-Router/1.0'
      }
    });

    if (!response.data || typeof response.data.response !== 'string') {
      throw new Error('Invalid response format from Ollama');
    }

    return response;
  }

  processResponse(response, startTime) {
    const endTime = Date.now();
    const responseTime = endTime - startTime;
    
    // 통계 업데이트
    this.updateStatistics(responseTime, true);
    
    // 응답 데이터 추출
    const result = {
      response: response.data.response.trim(),
      metadata: {
        model: response.data.model || this.modelName,
        total_duration: response.data.total_duration,
        load_duration: response.data.load_duration,
        prompt_eval_count: response.data.prompt_eval_count,
        prompt_eval_duration: response.data.prompt_eval_duration,
        eval_count: response.data.eval_count,
        eval_duration: response.data.eval_duration,
        response_time_ms: responseTime,
        timestamp: new Date().toISOString()
      }
    };

    this.lastUsed = new Date();
    
    this.logger.debug('라우터 응답 생성 완료', {
      responseTime: `${responseTime}ms`,
      responseLength: result.response.length,
      evalCount: result.metadata.eval_count
    });

    return result;
  }

  handleError(error, prompt, startTime) {
    const responseTime = Date.now() - startTime;
    
    // 통계 업데이트
    this.updateStatistics(responseTime, false);
    
    // 에러 타입별 처리
    let errorType = 'unknown';
    let userMessage = '라우터 모델 요청 중 오류가 발생했습니다.';
    
    if (error.code === 'ECONNREFUSED') {
      errorType = 'connection_refused';
      userMessage = 'Ollama 서비스에 연결할 수 없습니다. 서비스가 실행 중인지 확인하세요.';
    } else if (error.code === 'ENOTFOUND') {
      errorType = 'host_not_found';
      userMessage = 'Ollama 서버를 찾을 수 없습니다. 주소를 확인하세요.';
    } else if (error.code === 'ETIMEDOUT' || error.message.includes('timeout')) {
      errorType = 'timeout';
      userMessage = '요청 시간이 초과되었습니다. 다시 시도해주세요.';
    } else if (error.response?.status === 404) {
      errorType = 'model_not_found';
      userMessage = `모델 ${this.modelName}을 찾을 수 없습니다.`;
    } else if (error.response?.status === 500) {
      errorType = 'server_error';
      userMessage = 'Ollama 서버 내부 오류가 발생했습니다.';
    }

    this.logger.error('라우터 모델 오류', {
      errorType,
      errorMessage: error.message,
      prompt: prompt.substring(0, 100),
      responseTime: `${responseTime}ms`,
      endpoint: this.endpoint
    });

    // 사용자 친화적 오류 메시지로 변경
    error.userMessage = userMessage;
    error.errorType = errorType;
  }

  updateStatistics(responseTime, success) {
    this.requestCount++;
    
    if (!success) {
      this.errorCount++;
    }
    
    // 응답 시간 히스토리 관리
    this.responseTimeHistory.push({
      timestamp: Date.now(),
      responseTime: responseTime,
      success: success
    });
    
    if (this.responseTimeHistory.length > this.maxHistorySize) {
      this.responseTimeHistory = this.responseTimeHistory.slice(-this.maxHistorySize);
    }
  }

  calculateRetryDelay(attempt) {
    // 지수 백오프: 1초, 2초, 4초...
    return Math.min(1000 * Math.pow(2, attempt - 1), 10000);
  }

  generateCacheKey(prompt, options) {
    const keyData = {
      prompt: prompt,
      temperature: options.temperature,
      max_tokens: options.max_tokens,
      model: this.modelName
    };
    
    // 간단한 해시 생성
    const keyString = JSON.stringify(keyData);
    let hash = 0;
    for (let i = 0; i < keyString.length; i++) {
      const char = keyString.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // 32bit 정수로 변환
    }
    
    return `router_${Math.abs(hash)}`;
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // 의도 분석 전용 메서드
  async analyzeIntent(userQuery) {
    const prompt = `사용자 쿼리를 분석하여 의도를 파악해주세요:

쿼리: "${userQuery}"

다음 JSON 형식으로 응답해주세요:
{
  "intent": "주요 의도 (analyze/visualize/train/predict/help/general)",
  "confidence": 0.0-1.0,
  "complexity": 0.0-1.0,
  "requires_data": true/false,
  "requires_training": true/false,
  "requires_visualization": true/false,
  "suggested_mode": "general/ml/coding",
  "keywords": ["키워드1", "키워드2"],
  "reasoning": "분석 근거"
}

분석 기준:
- intent: 사용자가 원하는 주요 작업
- confidence: 의도 파악의 확신도
- complexity: 작업의 복잡도 (0=매우 간단, 1=매우 복잡)
- 나머지는 해당 작업에 필요한 요소들`;

    try {
      const response = await this.generateResponse(prompt, {
        temperature: 0.1,
        max_tokens: 400,
        cache: true
      });

      return this.parseIntentResponse(response);
      
    } catch (error) {
      this.logger.error('의도 분석 실패:', error);
      return this.getDefaultIntent();
    }
  }

  parseIntentResponse(response) {
    try {
      // JSON 추출
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        
        // 기본값으로 검증 및 보완
        return {
          intent: parsed.intent || 'general',
          confidence: Math.max(0, Math.min(1, parsed.confidence || 0.5)),
          complexity: Math.max(0, Math.min(1, parsed.complexity || 0.3)),
          requires_data: parsed.requires_data !== undefined ? parsed.requires_data : false,
          requires_training: parsed.requires_training !== undefined ? parsed.requires_training : false,
          requires_visualization: parsed.requires_visualization !== undefined ? parsed.requires_visualization : false,
          suggested_mode: parsed.suggested_mode || 'general',
          keywords: Array.isArray(parsed.keywords) ? parsed.keywords : [],
          reasoning: parsed.reasoning || '분석 근거 없음'
        };
      }
    } catch (error) {
      this.logger.warn('의도 분석 응답 파싱 실패:', error);
    }

    return this.getDefaultIntent();
  }

  getDefaultIntent() {
    return {
      intent: 'general',
      confidence: 0.3,
      complexity: 0.3,
      requires_data: false,
      requires_training: false,
      requires_visualization: false,
      suggested_mode: 'general',
      keywords: [],
      reasoning: '의도 파악 실패, 기본값 사용'
    };
  }

  // 라우팅 결정 메서드
  async makeRoutingDecision(userQuery, context = {}) {
    const prompt = `사용자 요청을 분석하여 적절한 처리 방법을 결정해주세요:

요청: "${userQuery}"
현재 모드: ${context.currentMode || 'general'}
사용 가능한 도구: ${context.availableTools ? context.availableTools.join(', ') : '없음'}

다음 JSON 형식으로 응답해주세요:
{
  "task_type": "simple/complex/system",
  "recommended_model": "router/processor",
  "required_tools": ["tool1", "tool2"],
  "estimated_time": "seconds",
  "confidence": 0.0-1.0,
  "reasoning": "결정 근거"
}

결정 기준:
- simple: 라우터 모델로 처리 가능한 간단한 작업
- complex: 프로세서 모델이 필요한 복잡한 작업  
- system: 시스템 관리 작업
- estimated_time: 예상 처리 시간 (초)`;

    try {
      const response = await this.generateResponse(prompt, {
        temperature: 0.1,
        max_tokens: 300,
        cache: true
      });

      return this.parseRoutingResponse(response);
      
    } catch (error) {
      this.logger.error('라우팅 결정 실패:', error);
      return this.getDefaultRouting();
    }
  }

  parseRoutingResponse(response) {
    try {
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        
        return {
          task_type: parsed.task_type || 'simple',
          recommended_model: parsed.recommended_model || 'router',
          required_tools: Array.isArray(parsed.required_tools) ? parsed.required_tools : [],
          estimated_time: parsed.estimated_time || '30',
          confidence: Math.max(0, Math.min(1, parsed.confidence || 0.5)),
          reasoning: parsed.reasoning || '라우팅 근거 없음'
        };
      }
    } catch (error) {
      this.logger.warn('라우팅 응답 파싱 실패:', error);
    }

    return this.getDefaultRouting();
  }

  getDefaultRouting() {
    return {
      task_type: 'simple',
      recommended_model: 'router',
      required_tools: ['general_query'],
      estimated_time: '30',
      confidence: 0.3,
      reasoning: '라우팅 결정 실패, 기본값 사용'
    };
  }

  // 간단한 질문 응답 메서드
  async handleSimpleQuery(userQuery, context = {}) {
    const prompt = `다음 질문에 간단명료하게 답변해주세요:

질문: "${userQuery}"

답변 조건:
- 한국어로 답변
- 3-5 문장으로 요약
- 필요시 구체적인 예시 포함
- 전문용어는 쉽게 설명`;

    try {
      const response = await this.generateResponse(prompt, {
        temperature: 0.3,
        max_tokens: 400,
        cache: true
      });

      return {
        response: response,
        handled_by: 'router',
        response_type: 'simple_answer',
        timestamp: new Date().toISOString()
      };
      
    } catch (error) {
      this.logger.error('간단한 질문 처리 실패:', error);
      throw error;
    }
  }

  // 상태 및 통계 조회
  getStatus() {
    const recentResponses = this.responseTimeHistory.slice(-10);
    const avgResponseTime = recentResponses.length > 0
      ? recentResponses.reduce((sum, r) => sum + r.responseTime, 0) / recentResponses.length
      : 0;

    const successRate = this.requestCount > 0
      ? ((this.requestCount - this.errorCount) / this.requestCount) * 100
      : 0;

    return {
      model: this.modelName,
      endpoint: this.endpoint,
      is_loaded: this.isLoaded,
      last_used: this.lastUsed,
      statistics: {
        total_requests: this.requestCount,
        error_count: this.errorCount,
        success_rate: Math.round(successRate * 100) / 100,
        avg_response_time_ms: Math.round(avgResponseTime),
        cache_hit_rate: this.cache.getStats().hits / (this.cache.getStats().hits + this.cache.getStats().misses) * 100 || 0
      },
      config: {
        temperature: this.config.temperature,
        max_tokens: this.config.max_tokens,
        timeout: this.config.timeout,
        retry_attempts: this.config.retry_attempts
      },
      health: this.evaluateHealth()
    };
  }

  evaluateHealth() {
    if (!this.isLoaded) return 'not_loaded';
    
    const recentFailures = this.responseTimeHistory
      .slice(-10)
      .filter(r => !r.success).length;
    
    if (recentFailures >= 5) return 'unhealthy';
    if (recentFailures >= 2) return 'degraded';
    
    const avgResponseTime = this.responseTimeHistory.slice(-5)
      .reduce((sum, r) => sum + r.responseTime, 0) / Math.max(1, this.responseTimeHistory.slice(-5).length);
    
    if (avgResponseTime > 5000) return 'slow';
    if (avgResponseTime > 2000) return 'degraded';
    
    return 'healthy';
  }

  // 캐시 관리
  clearCache() {
    this.cache.flush();
    this.logger.info('라우터 모델 캐시 클리어됨');
  }

  getCacheStats() {
    return this.cache.getStats();
  }

  // 설정 업데이트
  updateConfig(newConfig) {
    this.config = { ...this.config, ...newConfig };
    this.logger.info('라우터 설정 업데이트됨:', newConfig);
  }

  // 정리 작업
  cleanup() {
    this.clearCache();
    this.responseTimeHistory = [];
    this.requestCount = 0;
    this.errorCount = 0;
    this.logger.info('LlamaRouterClient 정리 완료');
  }
}
