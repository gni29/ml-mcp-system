// models/llama-router/prompts.js
import { Logger } from '../../utils/logger.js';

export class LlamaRouterPrompts {
  constructor() {
    this.logger = new Logger();
    this.templates = this.initializeTemplates();
    this.responseFormats = this.initializeResponseFormats();
    this.examples = this.initializeExamples();
  }

  initializeTemplates() {
    return {
      // 의도 분석 프롬프트
      intent_analysis: {
        system: `당신은 사용자의 요청을 분석하여 의도를 파악하는 AI 어시스턴트입니다.
사용자의 질문이나 요청을 정확히 분석하여 다음 정보를 JSON 형식으로 제공해주세요.`,
        
        user_template: `사용자 요청: "{query}"

다음 항목들을 분석해주세요:

1. intent (주요 의도):
   - "analyze": 데이터 분석 요청
   - "visualize": 시각화/차트 생성 요청  
   - "train": 모델 훈련 요청
   - "predict": 예측/추론 요청
   - "help": 도움말/설명 요청
   - "system": 시스템 관리 요청
   - "general": 일반적인 대화

2. confidence (확신도): 0.0-1.0 (1.0이 가장 확실함)

3. complexity (복잡도): 0.0-1.0 
   - 0.0-0.3: 간단 (기본 통계, 단순 질문)
   - 0.4-0.6: 중간 (상관관계, 기본 ML)  
   - 0.7-1.0: 복잡 (딥러닝, 고급 분석)

4. requires_data: 데이터 파일이 필요한지 (true/false)
5. requires_training: 모델 훈련이 필요한지 (true/false)
6. requires_visualization: 시각화가 필요한지 (true/false)
7. suggested_mode: 권장 모드 ("general"/"ml"/"coding"/"deep_learning")
8. keywords: 핵심 키워드 배열
9. reasoning: 분석 근거 (한 문장)

JSON 형식으로만 응답하세요.`,

        fallback: `사용자 요청을 분석할 수 없습니다. 기본 설정을 사용합니다.`
      },

      // 라우팅 결정 프롬프트
      routing_decision: {
        system: `당신은 사용자 요청을 적절한 처리 모델로 라우팅하는 AI 어시스턴트입니다.
요청의 복잡도와 필요한 리소스를 분석하여 최적의 처리 방법을 결정해주세요.`,

        user_template: `사용자 요청: "{query}"
현재 모드: {current_mode}
사용 가능한 도구: {available_tools}
시스템 상태: {system_status}

다음 정보를 JSON 형식으로 제공해주세요:

1. task_type (작업 유형):
   - "simple": 라우터 모델로 처리 가능한 간단한 작업
   - "complex": 프로세서 모델이 필요한 복잡한 작업
   - "system": 시스템 관리 관련 작업
   - "hybrid": 여러 단계가 필요한 복합 작업

2. recommended_model: 권장 모델 ("router"/"processor"/"both")

3. required_tools: 필요한 도구들의 배열

4. estimated_time: 예상 처리 시간 (초 단위의 숫자)

5. confidence: 라우팅 결정의 확신도 (0.0-1.0)

6. resource_requirements:
   - memory_intensive: 메모리 집약적인지 (true/false)
   - cpu_intensive: CPU 집약적인지 (true/false)
   - gpu_required: GPU가 필요한지 (true/false)

7. reasoning: 라우팅 결정 근거 (한 문장)

JSON 형식으로만 응답하세요.`,

        fallback: `라우팅 결정을 할 수 없습니다. 기본 라우터 모델로 처리합니다.`
      },

      // 간단한 질문 답변 프롬프트
      simple_qa: {
        system: `당신은 친근하고 도움이 되는 AI 어시스턴트입니다.
사용자의 질문에 정확하고 이해하기 쉽게 답변해주세요.`,

        user_template: `질문: "{query}"

답변 가이드라인:
- 한국어로 답변
- 3-5 문장으로 간결하게 설명
- 전문용어는 쉽게 풀어서 설명
- 필요시 구체적인 예시 포함
- 추가 질문이 있으면 언제든지 물어보라고 안내

답변:`,

        fallback: `죄송합니다. 해당 질문에 대한 답변을 제공할 수 없습니다. 다른 방식으로 질문해주시거나, 더 구체적인 정보를 제공해주세요.`
      },

      // 모드 전환 분석 프롬프트
      mode_suggestion: {
        system: `당신은 사용자의 요청에 따라 최적의 작업 모드를 추천하는 AI 어시스턴트입니다.`,

        user_template: `사용자 요청: "{query}"
현재 모드: {current_mode}

사용 가능한 모드:
- general: 일반적인 대화 및 간단한 작업
- ml: 데이터 분석, 머신러닝, 시각화
- coding: 코드 생성, 리뷰, 디버깅
- deep_learning: 딥러닝 모델 개발 및 훈련

다음 정보를 JSON 형식으로 제공해주세요:

1. should_switch: 모드 전환이 필요한지 (true/false)
2. recommended_mode: 권장 모드
3. confidence: 추천 확신도 (0.0-1.0)
4. reasons: 모드 전환 이유들의 배열
5. benefits: 모드 전환 시 얻을 수 있는 이점들
6. switch_message: 사용자에게 보여줄 전환 메시지

JSON 형식으로만 응답하세요.`,

        fallback: `모드 분석을 할 수 없습니다. 현재 모드를 유지합니다.`
      },

      // 도움말 생성 프롬프트
      help_generation: {
        system: `당신은 사용자가 시스템을 효과적으로 사용할 수 있도록 도움을 주는 AI 어시스턴트입니다.`,

        user_template: `도움말 요청: "{query}"
현재 모드: {current_mode}
사용 가능한 기능: {available_features}

다음 형태로 도움말을 제공해주세요:

1. 요청과 관련된 주요 기능들
2. 각 기능의 간단한 설명
3. 사용 예시 (구체적인 명령어나 질문 형태)
4. 추가로 도움이 될 수 있는 관련 기능들

친근하고 이해하기 쉬운 언어로 설명해주세요.`,

        fallback: `일반적인 도움말을 제공해드리겠습니다. 구체적인 질문이 있으시면 언제든지 말씀해주세요.`
      },

      // 오류 설명 프롬프트
      error_explanation: {
        system: `당신은 시스템 오류를 사용자가 이해하기 쉽게 설명하는 AI 어시스턴트입니다.`,

        user_template: `오류 정보:
- 오류 타입: {error_type}
- 오류 메시지: {error_message}
- 발생 상황: {context}

다음 형태로 설명해주세요:

1. 무엇이 잘못되었는지 간단히 설명
2. 왜 이런 오류가 발생했는지 가능한 원인
3. 해결 방법 제안 (구체적인 단계)
4. 앞으로 이런 오류를 피하는 방법

기술적인 용어는 피하고 일반 사용자가 이해할 수 있는 언어로 설명해주세요.`,

        fallback: `시스템에서 오류가 발생했습니다. 잠시 후 다시 시도해주시거나, 시스템 관리자에게 문의해주세요.`
      }
    };
  }

  initializeResponseFormats() {
    return {
      intent_analysis: {
        intent: "string",
        confidence: "number",
        complexity: "number",
        requires_data: "boolean",
        requires_training: "boolean",
        requires_visualization: "boolean",
        suggested_mode: "string",
        keywords: "array",
        reasoning: "string"
      },

      routing_decision: {
        task_type: "string",
        recommended_model: "string",
        required_tools: "array",
        estimated_time: "number",
        confidence: "number",
        resource_requirements: {
          memory_intensive: "boolean",
          cpu_intensive: "boolean",
          gpu_required: "boolean"
        },
        reasoning: "string"
      },

      mode_suggestion: {
        should_switch: "boolean",
        recommended_mode: "string",
        confidence: "number",
        reasons: "array",
        benefits: "array",
        switch_message: "string"
      }
    };
  }

  initializeExamples() {
    return {
      intent_analysis: [
        {
          query: "이 데이터를 분석해줘",
          expected: {
            intent: "analyze",
            confidence: 0.9,
            complexity: 0.4,
            requires_data: true,
            requires_training: false,
            requires_visualization: true,
            suggested_mode: "ml",
            keywords: ["데이터", "분석"],
            reasoning: "명확한 데이터 분석 요청"
          }
        },
        {
          query: "안녕하세요",
          expected: {
            intent: "general",
            confidence: 0.95,
            complexity: 0.1,
            requires_data: false,
            requires_training: false,
            requires_visualization: false,
            suggested_mode: "general",
            keywords: ["인사"],
            reasoning: "일반적인 인사 표현"
          }
        },
        {
          query: "딥러닝 모델을 훈련하고 싶어요",
          expected: {
            intent: "train",
            confidence: 0.9,
            complexity: 0.8,
            requires_data: true,
            requires_training: true,
            requires_visualization: false,
            suggested_mode: "deep_learning",
            keywords: ["딥러닝", "모델", "훈련"],
            reasoning: "딥러닝 모델 훈련 요청"
          }
        }
      ],

      routing_decision: [
        {
          query: "안녕하세요, 도움이 필요해요",
          expected: {
            task_type: "simple",
            recommended_model: "router",
            required_tools: ["general_query"],
            estimated_time: 5,
            confidence: 0.9,
            resource_requirements: {
              memory_intensive: false,
              cpu_intensive: false,
              gpu_required: false
            },
            reasoning: "간단한 일반 대화"
          }
        },
        {
          query: "복잡한 시계열 예측 모델을 만들어주세요",
          expected: {
            task_type: "complex",
            recommended_model: "processor",
            required_tools: ["train_model", "analyze_data", "visualize_data"],
            estimated_time: 300,
            confidence: 0.8,
            resource_requirements: {
              memory_intensive: true,
              cpu_intensive: true,
              gpu_required: false
            },
            reasoning: "복잡한 ML 모델 훈련 필요"
          }
        }
      ]
    };
  }

  // 프롬프트 생성 메인 메서드
  generatePrompt(type, data = {}) {
    try {
      const template = this.templates[type];
      if (!template) {
        throw new Error(`Unknown prompt type: ${type}`);
      }

      let prompt = '';
      
      // 시스템 메시지가 있으면 추가
      if (template.system) {
        prompt += template.system + '\n\n';
      }

      // 사용자 템플릿 처리
      if (template.user_template) {
        prompt += this.processTemplate(template.user_template, data);
      }

      this.logger.debug('프롬프트 생성 완료', {
        type: type,
        length: prompt.length,
        hasData: Object.keys(data).length > 0
      });

      return prompt;

    } catch (error) {
      this.logger.error('프롬프트 생성 실패:', error);
      return this.templates[type]?.fallback || "프롬프트를 생성할 수 없습니다.";
    }
  }

  // 템플릿 변수 치환
  processTemplate(template, data) {
    let processed = template;

    // 기본값 설정
    const defaults = {
      query: '',
      current_mode: 'general',
      available_tools: '없음',
      system_status: '정상',
      available_features: '기본 기능',
      error_type: '알 수 없음',
      error_message: '오류 정보 없음',
      context: '상황 정보 없음'
    };

    const mergedData = { ...defaults, ...data };

    // 배열 타입 데이터 처리
    if (Array.isArray(mergedData.available_tools)) {
      mergedData.available_tools = mergedData.available_tools.join(', ') || '없음';
    }

    if (Array.isArray(mergedData.available_features)) {
      mergedData.available_features = mergedData.available_features.join(', ') || '기본 기능';
    }

    // 변수 치환
    for (const [key, value] of Object.entries(mergedData)) {
      const placeholder = `{${key}}`;
      if (processed.includes(placeholder)) {
        processed = processed.replace(new RegExp(placeholder.replace(/[{}]/g, '\\$&'), 'g'), String(value));
      }
    }

    return processed;
  }

  // 의도 분석 프롬프트 생성
  createIntentAnalysisPrompt(query, context = {}) {
    return this.generatePrompt('intent_analysis', {
      query: query,
      current_mode: context.currentMode,
      available_tools: context.availableTools
    });
  }

  // 라우팅 결정 프롬프트 생성
  createRoutingDecisionPrompt(query, context = {}) {
    return this.generatePrompt('routing_decision', {
      query: query,
      current_mode: context.currentMode,
      available_tools: context.availableTools,
      system_status: context.systemStatus || '정상'
    });
  }

  // 간단한 질문 답변 프롬프트 생성
  createSimpleQAPrompt(query) {
    return this.generatePrompt('simple_qa', {
      query: query
    });
  }

  // 모드 전환 제안 프롬프트 생성
  createModeSuggestionPrompt(query, currentMode) {
    return this.generatePrompt('mode_suggestion', {
      query: query,
      current_mode: currentMode
    });
  }

  // 도움말 생성 프롬프트 생성
  createHelpPrompt(query, context = {}) {
    return this.generatePrompt('help_generation', {
      query: query,
      current_mode: context.currentMode,
      available_features: context.availableFeatures
    });
  }

  // 오류 설명 프롬프트 생성
  createErrorExplanationPrompt(errorInfo) {
    return this.generatePrompt('error_explanation', {
      error_type: errorInfo.type,
      error_message: errorInfo.message,
      context: errorInfo.context
    });
  }

  // 맞춤형 프롬프트 생성 (고급 사용자용)
  createCustomPrompt(systemMessage, userTemplate, data = {}) {
    try {
      let prompt = '';
      
      if (systemMessage) {
        prompt += systemMessage + '\n\n';
      }

      if (userTemplate) {
        prompt += this.processTemplate(userTemplate, data);
      }

      return prompt;

    } catch (error) {
      this.logger.error('맞춤형 프롬프트 생성 실패:', error);
      return "맞춤형 프롬프트를 생성할 수 없습니다.";
    }
  }

  // 응답 형식 검증
  validateResponse(type, response) {
    try {
      const expectedFormat = this.responseFormats[type];
      if (!expectedFormat) {
        return { valid: false, error: `Unknown response type: ${type}` };
      }

      // JSON 파싱 시도
      let parsed;
      try {
        if (typeof response === 'string') {
          const jsonMatch = response.match(/\{[\s\S]*\}/);
          if (jsonMatch) {
            parsed = JSON.parse(jsonMatch[0]);
          } else {
            throw new Error('No JSON found in response');
          }
        } else {
          parsed = response;
        }
      } catch (parseError) {
        return { valid: false, error: 'Invalid JSON format' };
      }

      // 필드 검증
      const validation = this.validateFields(parsed, expectedFormat);
      
      return {
        valid: validation.valid,
        error: validation.error,
        data: parsed
      };

    } catch (error) {
      return { valid: false, error: error.message };
    }
  }

  validateFields(data, format, path = '') {
    for (const [field, expectedType] of Object.entries(format)) {
      const fieldPath = path ? `${path}.${field}` : field;
      
      if (!(field in data)) {
        return { valid: false, error: `Missing field: ${fieldPath}` };
      }

      const value = data[field];
      
      if (typeof expectedType === 'object') {
        // 중첩 객체 검증
        if (typeof value !== 'object' || value === null) {
          return { valid: false, error: `Field ${fieldPath} should be an object` };
        }
        
        const nestedValidation = this.validateFields(value, expectedType, fieldPath);
        if (!nestedValidation.valid) {
          return nestedValidation;
        }
      } else {
        // 기본 타입 검증
        if (!this.isValidType(value, expectedType)) {
          return { valid: false, error: `Field ${fieldPath} has wrong type. Expected: ${expectedType}` };
        }
      }
    }

    return { valid: true };
  }

  isValidType(value, expectedType) {
    switch (expectedType) {
      case 'string':
        return typeof value === 'string';
      case 'number':
        return typeof value === 'number' && !isNaN(value);
      case 'boolean':
        return typeof value === 'boolean';
      case 'array':
        return Array.isArray(value);
      default:
        return false;
    }
  }

  // 예시 데이터 조회
  getExamples(type) {
    return this.examples[type] || [];
  }

  // 프롬프트 테스트
  testPrompt(type, testData) {
    try {
      const prompt = this.generatePrompt(type, testData);
      const examples = this.getExamples(type);
      
      return {
        prompt: prompt,
        examples: examples,
        length: prompt.length,
        hasExamples: examples.length > 0
      };

    } catch (error) {
      return {
        error: error.message,
        prompt: null,
        examples: []
      };
    }
  }

  // 프롬프트 통계
  getPromptStats() {
    const stats = {
      total_templates: Object.keys(this.templates).length,
      total_formats: Object.keys(this.responseFormats).length,
      total_examples: Object.values(this.examples).reduce((sum, ex) => sum + ex.length, 0),
      template_types: Object.keys(this.templates),
      avg_template_length: 0
    };

    // 평균 템플릿 길이 계산
    const lengths = Object.values(this.templates).map(t =>
      (t.system || '').length + (t.user_template || '').length
    );
    
    stats.avg_template_length = lengths.length > 0
      ? Math.round(lengths.reduce((sum, len) => sum + len, 0) / lengths.length)
      : 0;

    return stats;
  }

  // 프롬프트 캐시 (성능 최적화)
  cachePrompt(type, data, prompt) {
    const cacheKey = `${type}_${JSON.stringify(data)}`;
    // 실제 구현에서는 LRU 캐시 등을 사용할 수 있음
    this.logger.debug('프롬프트 캐시됨', { type, cacheKey: cacheKey.substring(0, 50) });
  }

  // 설정 업데이트
  updateTemplate(type, newTemplate) {
    if (this.templates[type]) {
      this.templates[type] = { ...this.templates[type], ...newTemplate };
      this.logger.info('프롬프트 템플릿 업데이트됨', { type });
      return true;
    }
    return false;
  }

  // 새 템플릿 추가
  addTemplate(type, template) {
    this.templates[type] = template;
    this.logger.info('새 프롬프트 템플릿 추가됨', { type });
  }
}
