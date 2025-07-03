import { Logger } from '../utils/logger.js';

export class Router {
  constructor(modelManager) {
    this.modelManager = modelManager;
    this.logger = new Logger();
    this.routingRules = null;
    this.loadRoutingRules();
  }

  async loadRoutingRules() {
    try {
      const fs = await import('fs/promises');
      const rulesData = await fs.readFile('./config/routing-rules.json', 'utf-8');
      this.routingRules = JSON.parse(rulesData);
    } catch (error) {
      this.logger.warn('라우팅 규칙 로드 실패, 기본값 사용:', error);
      this.routingRules = this.getDefaultRules();
    }
  }

  getDefaultRules() {
    return {
      simple_queries: {
        keywords: ['안녕', '도움말', '상태', '모드'],
        maxComplexity: 0.3
      },
      complex_analysis: {
        keywords: ['분석', '모델', '훈련', '예측', '시각화'],
        minComplexity: 0.6
      }
    };
  }

  async route(toolName, args) {
    try {
      // 1. 기본 시스템 도구 확인
      if (this.isSystemTool(toolName)) {
        return {
          taskType: 'system',
          model: 'router',
          tools: [toolName]
        };
      }

      // 2. 라우터 모델로 의도 파악
      const intent = await this.analyzeIntent(toolName, args);
      
      // 3. 복잡도 기반 라우팅 결정
      const routingDecision = this.makeRoutingDecision(intent);
      
      this.logger.info('라우팅 결정:', routingDecision);
      
      return routingDecision;
      
    } catch (error) {
      this.logger.error('라우팅 실패:', error);
      // 기본값으로 폴백
      return {
        taskType: 'simple',
        model: 'router',
        tools: [toolName]
      };
    }
  }

  isSystemTool(toolName) {
    const systemTools = ['mode_switch', 'system_status', 'health_check'];
    return systemTools.includes(toolName);
  }

  async analyzeIntent(toolName, args) {
    const prompt = this.buildIntentAnalysisPrompt(toolName, args);
    
    const response = await this.modelManager.queryModel('router', prompt, {
      temperature: 0.1,
      max_tokens: 256
    });

    return this.parseIntentResponse(response);
  }

  buildIntentAnalysisPrompt(toolName, args) {
    return `사용자가 요청한 작업을 분석해주세요:

도구명: ${toolName}
인수: ${JSON.stringify(args, null, 2)}

다음 형식으로 응답해주세요:
{
  "intent": "사용자의 의도",
  "complexity": 0.0-1.0,
  "dataSize": "small/medium/large",
  "requiresProcessing": true/false,
  "suggestedModel": "router/processor"
}`;
  }

  parseIntentResponse(response) {
    try {
      // JSON 응답 파싱 시도
      const match = response.match(/\{[\s\S]*\}/);
      if (match) {
        return JSON.parse(match[0]);
      }
    } catch (error) {
      this.logger.warn('의도 분석 응답 파싱 실패:', error);
    }

    // 파싱 실패시 기본값 반환
    return {
      intent: "unknown",
      complexity: 0.5,
      dataSize: "medium",
      requiresProcessing: true,
      suggestedModel: "processor"
    };
  }

  makeRoutingDecision(intent) {
    const { complexity, requiresProcessing, suggestedModel } = intent;

    // 간단한 작업
    if (complexity < 0.4 && !requiresProcessing) {
      return {
        taskType: 'simple',
        model: 'router',
        tools: ['basic_response']
      };
    }

    // 복잡한 작업
    if (complexity >= 0.6 || requiresProcessing) {
      return {
        taskType: 'complex',
        model: 'processor',
        tools: ['python_executor', 'result_formatter']
      };
    }

    // 중간 복잡도 - 라우터로 시작, 필요시 프로세서 호출
    return {
      taskType: 'hybrid',
      model: 'router',
      fallbackModel: 'processor',
      tools: ['adaptive_processor']
    };
  }

  async handleSimpleTask(args) {
    const prompt = `사용자 요청에 간단히 응답해주세요: ${JSON.stringify(args)}`;
    
    const response = await this.modelManager.queryModel('router', prompt, {
      temperature: 0.3,
      max_tokens: 512
    });

    return {
      content: [
        {
          type: 'text',
          text: response
        }
      ]
    };
  }
}
