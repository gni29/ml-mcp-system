// core/router.js - 완전한 지능형 라우터 시스템
import { Logger } from '../utils/logger.js';
import { ConfigLoader } from '../utils/config-loader.js';

export class Router {
  constructor(modelManager) {
    this.modelManager = modelManager;
    this.logger = new Logger();
    this.configLoader = new ConfigLoader();
    this.routingRules = null;
    this.routingHistory = [];
    this.performanceMetrics = new Map();
    this.adaptiveLearning = {
      enabled: true,
      learningRate: 0.1,
      confidenceThreshold: 0.7
    };
  }

  async initialize() {
    try {
      // 설정 로더 초기화
      await this.configLoader.initialize();
      
      // 라우팅 규칙 로드
      await this.loadRoutingRules();
      
      // 성능 지표 초기화
      this.initializePerformanceMetrics();
      
      this.logger.info('Router 초기화 완료');
    } catch (error) {
      this.logger.error('Router 초기화 실패:', error);
      throw error;
    }
  }

  async loadRoutingRules() {
    try {
      this.routingRules = this.configLoader.getConfig('routing-rules');
      
      if (!this.routingRules) {
        this.logger.warn('라우팅 규칙을 찾을 수 없음, 기본값 사용');
        this.routingRules = this.getDefaultRules();
      }
      
      this.logger.info('라우팅 규칙 로드 완료');
    } catch (error) {
      this.logger.error('라우팅 규칙 로드 실패:', error);
      this.routingRules = this.getDefaultRules();
    }
  }

  getDefaultRules() {
    return {
      intent_patterns: {
        data_analysis: [
          'analyze', 'analysis', 'explore', 'examine', 'investigate',
          'statistics', 'stats', 'summary', 'describe', '분석', '탐색', '조사'
        ],
        visualization: [
          'plot', 'chart', 'graph', 'visualize', 'draw', 'show',
          'histogram', 'scatter', 'heatmap', '시각화', '차트', '그래프'
        ],
        machine_learning: [
          'train', 'model', 'predict', 'classification', 'regression',
          'clustering', 'ml', 'machine learning', '훈련', '모델', '예측'
        ],
        data_processing: [
          'clean', 'preprocess', 'transform', 'encode', 'scale',
          'normalize', 'feature engineering', '전처리', '정규화', '변환'
        ]
      },
      complexity_thresholds: {
        simple: { 
          maxTokens: 500, 
          useRouter: true,
          keywords: ['hello', 'help', 'status', '안녕', '도움말', '상태']
        },
        medium: { 
          maxTokens: 1000, 
          useRouter: false,
          keywords: ['basic analysis', 'simple chart', '기본 분석', '간단한']
        },
        complex: { 
          maxTokens: 2000, 
          useRouter: false,
          keywords: ['deep learning', 'pipeline', '딥러닝', '파이프라인']
        }
      },
      mode_switching: {
        auto_switch: true,
        confidence_threshold: 0.7,
        switch_delay_ms: 1000
      }
    };
  }

  async route(toolName, args) {
    const startTime = Date.now();
    
    try {
      this.logger.info(`라우팅 요청: ${toolName}`, { args });

      // 1. 시스템 도구 우선 처리
      if (this.isSystemTool(toolName)) {
        const decision = {
          taskType: 'system',
          model: 'router',
          tools: [toolName],
          reasoning: 'System tool detected'
        };
        
        this.recordRoutingDecision(toolName, decision, Date.now() - startTime);
        return decision;
      }

      // 2. 쿼리 분석 및 복잡도 계산
      const queryAnalysis = await this.analyzeQuery(toolName, args);
      
      // 3. 의도 파악
      const intentAnalysis = await this.analyzeIntent(queryAnalysis);
      
      // 4. 라우팅 결정
      const routingDecision = await this.makeRoutingDecision(queryAnalysis, intentAnalysis);
      
      // 5. 적응형 학습 적용
      if (this.adaptiveLearning.enabled) {
        this.applyAdaptiveLearning(routingDecision, queryAnalysis);
      }
      
      // 6. 성능 기록
      const executionTime = Date.now() - startTime;
      this.recordRoutingDecision(toolName, routingDecision, executionTime);
      
      this.logger.info('라우팅 결정 완료:', routingDecision);
      return routingDecision;
      
    } catch (error) {
      this.logger.error('라우팅 실패:', error);
      
      // 안전한 폴백 전략
      const fallbackDecision = this.createFallbackDecision(toolName, error);
      this.recordRoutingDecision(toolName, fallbackDecision, Date.now() - startTime, error);
      
      return fallbackDecision;
    }
  }

  async analyzeQuery(toolName, args) {
    const analysis = {
      toolName,
      args,
      queryText: this.extractQueryText(args),
      hasDataFiles: false,
      estimatedDataSize: 'unknown',
      requiredOperations: [],
      estimatedComplexity: 0,
      resourceRequirements: this.estimateResourceRequirements(args)
    };

    // 쿼리 텍스트 분석
    if (analysis.queryText) {
      analysis.wordCount = analysis.queryText.split(' ').length;
      analysis.hasDataFiles = this.detectDataFiles(analysis.queryText);
      analysis.requiredOperations = this.extractOperations(analysis.queryText);
      analysis.estimatedComplexity = this.calculateComplexity(analysis);
    }

    // 인수 분석
    analysis.hasFileArguments = this.hasFileArguments(args);
    analysis.hasModelArguments = this.hasModelArguments(args);
    analysis.hasVisualizationRequest = this.hasVisualizationRequest(args);

    return analysis;
  }

  extractQueryText(args) {
    // 다양한 인수 형태에서 쿼리 텍스트 추출
    const possibleKeys = ['query', 'prompt', 'request', 'message', 'text'];
    
    for (const key of possibleKeys) {
      if (args[key] && typeof args[key] === 'string') {
        return args[key];
      }
    }

    // 객체 전체를 문자열로 변환
    return JSON.stringify(args);
  }

  calculateComplexity(analysis) {
    let complexity = 0;

    // 기본 복잡도
    if (analysis.wordCount) {
      complexity += Math.min(analysis.wordCount * 0.01, 0.3);
    }

    // 연산 복잡도
    const operationComplexity = {
      'basic_stats': 0.1,
      'correlation': 0.15,
      'visualization': 0.2,
      'clustering': 0.4,
      'classification': 0.5,
      'regression': 0.5,
      'deep_learning': 0.8,
      'pipeline': 0.6,
      'feature_engineering': 0.3
    };

    for (const operation of analysis.requiredOperations) {
      complexity += operationComplexity[operation] || 0.2;
    }

    // 데이터 크기 고려
    if (analysis.hasDataFiles) {
      complexity += 0.1;
    }

    // 파일 인수 고려
    if (analysis.hasFileArguments) {
      complexity += 0.15;
    }

    // 모델 관련 작업
    if (analysis.hasModelArguments) {
      complexity += 0.3;
    }

    return Math.min(complexity, 1.0);
  }

  extractOperations(queryText) {
    const operations = [];
    const lowerQuery = queryText.toLowerCase();

    const operationPatterns = {
      'basic_stats': ['통계', 'statistics', 'describe', 'summary', '요약'],
      'correlation': ['상관관계', 'correlation', 'relationship', '관계'],
      'visualization': ['시각화', 'plot', 'chart', 'graph', '차트', '그래프'],
      'clustering': ['클러스터', 'cluster', 'grouping', '그룹화'],
      'classification': ['분류', 'classification', 'classify', '예측'],
      'regression': ['회귀', 'regression', 'predict', '예측'],
      'deep_learning': ['딥러닝', 'deep learning', 'neural network', '신경망'],
      'pipeline': ['파이프라인', 'pipeline', 'workflow', '워크플로우'],
      'feature_engineering': ['피처', 'feature', 'engineering', '특성']
    };

    for (const [operation, patterns] of Object.entries(operationPatterns)) {
      if (patterns.some(pattern => lowerQuery.includes(pattern))) {
        operations.push(operation);
      }
    }

    return operations;
  }

  async analyzeIntent(queryAnalysis) {
    const prompt = this.buildIntentAnalysisPrompt(queryAnalysis);
    
    try {
      const response = await this.modelManager.queryModel('router', prompt, {
        temperature: 0.1,
        max_tokens: 300
      });

      const intent = this.parseIntentResponse(response);
      
      // 규칙 기반 검증 및 보강
      return this.enhanceIntentWithRules(intent, queryAnalysis);
      
    } catch (error) {
      this.logger.error('의도 분석 실패:', error);
      return this.createFallbackIntent(queryAnalysis);
    }
  }

  buildIntentAnalysisPrompt(queryAnalysis) {
    return `사용자 요청을 분석해주세요:

도구: ${queryAnalysis.toolName}
쿼리: ${queryAnalysis.queryText}
복잡도: ${queryAnalysis.estimatedComplexity}
필요 연산: ${queryAnalysis.requiredOperations.join(', ')}

다음 JSON 형식으로 응답해주세요:
{
  "primary_intent": "main_goal",
  "confidence": 0.0-1.0,
  "complexity_level": "simple|medium|complex",
  "data_intensive": true/false,
  "requires_ml": true/false,
  "requires_visualization": true/false,
  "suggested_model": "router|processor",
  "reasoning": "결정 이유"
}

분류 기준:
- simple: 기본 질의, 도움말, 상태 확인
- medium: 기본 분석, 단순 시각화
- complex: ML 모델링, 복잡한 분석, 파이프라인`;
  }

  parseIntentResponse(response) {
    try {
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        
        // 유효성 검증
        if (parsed.primary_intent && parsed.confidence !== undefined) {
          return parsed;
        }
      }
    } catch (error) {
      this.logger.warn('의도 분석 응답 파싱 실패:', error);
    }

    return null;
  }

  enhanceIntentWithRules(intent, queryAnalysis) {
    if (!intent) {
      return this.createFallbackIntent(queryAnalysis);
    }

    // 규칙 기반 복잡도 검증
    const ruleBasedComplexity = this.getRuleBasedComplexity(queryAnalysis);
    
    // AI와 규칙 기반 결과 결합
    const enhancedIntent = {
      ...intent,
      rule_based_complexity: ruleBasedComplexity,
      final_complexity: this.combineComplexityScores(
        queryAnalysis.estimatedComplexity,
        ruleBasedComplexity,
        intent.confidence
      )
    };

    return enhancedIntent;
  }

  getRuleBasedComplexity(queryAnalysis) {
    const rules = this.routingRules.complexity_thresholds;
    
    // 키워드 기반 복잡도 판단
    for (const [level, config] of Object.entries(rules)) {
      if (config.keywords) {
        const hasKeywords = config.keywords.some(keyword =>
          queryAnalysis.queryText.toLowerCase().includes(keyword.toLowerCase())
        );
        
        if (hasKeywords) {
          return level;
        }
      }
    }

    // 연산 기반 복잡도 판단
    if (queryAnalysis.requiredOperations.includes('deep_learning')) {
      return 'complex';
    } else if (queryAnalysis.requiredOperations.includes('clustering') || 
               queryAnalysis.requiredOperations.includes('classification')) {
      return 'medium';
    } else {
      return 'simple';
    }
  }

  combineComplexityScores(numerical, categorical, confidence) {
    const categoryToNumber = {
      'simple': 0.2,
      'medium': 0.5,
      'complex': 0.8
    };

    const categoricalScore = categoryToNumber[categorical] || 0.5;
    
    // 신뢰도에 따른 가중 평균
    const combined = (numerical * confidence) + (categoricalScore * (1 - confidence));
    
    return Math.max(0, Math.min(1, combined));
  }

  async makeRoutingDecision(queryAnalysis, intentAnalysis) {
    const decision = {
      taskType: this.determineTaskType(intentAnalysis),
      model: this.selectModel(intentAnalysis, queryAnalysis),
      tools: this.selectTools(intentAnalysis, queryAnalysis),
      confidence: intentAnalysis.confidence,
      reasoning: this.generateReasoning(intentAnalysis, queryAnalysis),
      resourceEstimate: queryAnalysis.resourceRequirements,
      fallbackModel: null,
      adaptiveOptions: {}
    };

    // 폴백 모델 설정
    if (decision.model === 'router' && intentAnalysis.final_complexity > 0.4) {
      decision.fallbackModel = 'processor';
    }

    // 적응형 옵션 설정
    if (this.adaptiveLearning.enabled) {
      decision.adaptiveOptions = this.generateAdaptiveOptions(intentAnalysis);
    }

    return decision;
  }

  determineTaskType(intentAnalysis) {
    switch (intentAnalysis.complexity_level) {
      case 'simple':
        return 'simple_query';
      case 'medium':
        return intentAnalysis.requires_ml ? 'ml_analysis' : 'data_analysis';
      case 'complex':
        return intentAnalysis.requires_ml ? 'advanced_ml' : 'complex_analysis';
      default:
        return 'general';
    }
  }

  selectModel(intentAnalysis, queryAnalysis) {
    // 명시적인 모델 제안이 있으면 우선 사용
    if (intentAnalysis.suggested_model && 
        intentAnalysis.confidence > this.adaptiveLearning.confidenceThreshold) {
      return intentAnalysis.suggested_model;
    }

    // 복잡도 기반 결정
    if (intentAnalysis.final_complexity < 0.3) {
      return 'router';
    } else if (intentAnalysis.final_complexity > 0.6) {
      return 'processor';
    } else {
      // 중간 복잡도 - 추가 조건 확인
      if (intentAnalysis.data_intensive || intentAnalysis.requires_ml) {
        return 'processor';
      } else {
        return 'router';
      }
    }
  }

  selectTools(intentAnalysis, queryAnalysis) {
    const tools = [];

    // 기본 도구 선택
    if (intentAnalysis.data_intensive) {
      tools.push('data_loader', 'data_validator');
    }

    if (intentAnalysis.requires_ml) {
      tools.push('ml_trainer', 'model_evaluator');
    }

    if (intentAnalysis.requires_visualization) {
      tools.push('chart_generator', 'plot_manager');
    }

    // 파이썬 실행기 필요 여부
    if (queryAnalysis.requiredOperations.length > 0 && 
        !queryAnalysis.requiredOperations.every(op => op === 'basic_stats')) {
      tools.push('python_executor');
    }

    // 결과 포매터는 항상 포함
    tools.push('result_formatter');

    return tools.length > 0 ? tools : ['basic_response'];
  }

  generateReasoning(intentAnalysis, queryAnalysis) {
    const reasons = [];

    reasons.push(`복잡도: ${intentAnalysis.final_complexity.toFixed(2)}`);
    reasons.push(`의도: ${intentAnalysis.primary_intent}`);
    reasons.push(`신뢰도: ${intentAnalysis.confidence.toFixed(2)}`);

    if (intentAnalysis.data_intensive) {
      reasons.push('데이터 집약적 작업');
    }

    if (intentAnalysis.requires_ml) {
      reasons.push('머신러닝 필요');
    }

    if (queryAnalysis.requiredOperations.length > 0) {
      reasons.push(`연산: ${queryAnalysis.requiredOperations.join(', ')}`);
    }

    return reasons.join(', ');
  }

  // 시스템 도구 확인
  isSystemTool(toolName) {
    const systemTools = [
      'get_system_status', 'change_mode', 'health_check',
      'system_status', 'mode_switch', 'config_update'
    ];
    return systemTools.includes(toolName);
  }

  // 리소스 요구사항 추정
  estimateResourceRequirements(args) {
    return {
      memory_mb: 500,
      cpu_cores: 1,
      gpu_required: false,
      estimated_time: 30
    };
  }

  // 파일 관련 검사 메서드들
  detectDataFiles(queryText) {
    const fileExtensions = ['.csv', '.xlsx', '.json', '.parquet', '.h5'];
    return fileExtensions.some(ext => queryText.toLowerCase().includes(ext));
  }

  hasFileArguments(args) {
    const fileKeys = ['file_path', 'filename', 'data_file', 'input_file'];
    return fileKeys.some(key => args[key]);
  }

  hasModelArguments(args) {
    const modelKeys = ['model_type', 'algorithm', 'model_name'];
    return modelKeys.some(key => args[key]);
  }

  hasVisualizationRequest(args) {
    const vizKeys = ['chart_type', 'plot_type', 'visualization'];
    const vizValues = ['chart', 'plot', 'graph', 'visual'];
    
    return vizKeys.some(key => args[key]) ||
           Object.values(args).some(value => 
             typeof value === 'string' && 
             vizValues.some(viz => value.toLowerCase().includes(viz))
           );
  }

  // 성능 및 학습 관련 메서드들
  initializePerformanceMetrics() {
    this.performanceMetrics.set('total_requests', 0);
    this.performanceMetrics.set('successful_routes', 0);
    this.performanceMetrics.set('average_response_time', 0);
    this.performanceMetrics.set('model_accuracy', new Map());
  }

  recordRoutingDecision(toolName, decision, executionTime, error = null) {
    const record = {
      timestamp: new Date().toISOString(),
      toolName,
      decision,
      executionTime,
      success: !error,
      error: error?.message
    };

    this.routingHistory.push(record);

    // 히스토리 크기 제한
    if (this.routingHistory.length > 1000) {
      this.routingHistory = this.routingHistory.slice(-500);
    }

    // 성능 지표 업데이트
    this.updatePerformanceMetrics(record);
  }

  updatePerformanceMetrics(record) {
    const total = this.performanceMetrics.get('total_requests') + 1;
    this.performanceMetrics.set('total_requests', total);

    if (record.success) {
      this.performanceMetrics.set('successful_routes',
        this.performanceMetrics.get('successful_routes') + 1);
    }

    // 평균 응답 시간 업데이트
    const currentAvg = this.performanceMetrics.get('average_response_time');
    const newAvg = (currentAvg * (total - 1) + record.executionTime) / total;
    this.performanceMetrics.set('average_response_time', newAvg);
  }

  applyAdaptiveLearning(decision, queryAnalysis) {
    // 실제 구현에서는 더 복잡한 학습 알고리즘 사용
    // 여기서는 간단한 예시만 제공
    
    if (decision.confidence < this.adaptiveLearning.confidenceThreshold) {
      this.logger.info('낮은 신뢰도로 인한 적응형 학습 적용');
      // 학습 로직 구현
    }
  }

  generateAdaptiveOptions(intentAnalysis) {
    return {
      enable_fallback: intentAnalysis.confidence < 0.6,
      monitor_performance: true,
      collect_feedback: true
    };
  }

  createFallbackDecision(toolName, error) {
    return {
      taskType: 'fallback',
      model: 'router',
      tools: ['error_handler', 'basic_response'],
      confidence: 0.1,
      reasoning: `폴백 모드: ${error.message}`,
      error: true,
      fallbackModel: null
    };
  }

  createFallbackIntent(queryAnalysis) {
    return {
      primary_intent: 'general_query',
      confidence: 0.3,
      complexity_level: 'medium',
      data_intensive: queryAnalysis.hasDataFiles,
      requires_ml: false,
      requires_visualization: false,
      suggested_model: 'processor',
      reasoning: 'Fallback intent due to analysis failure',
      final_complexity: Math.max(0.3, queryAnalysis.estimatedComplexity)
    };
  }

  // 간단한 작업 처리
  async handleSimpleTask(args) {
    const prompt = `사용자 요청에 간단히 응답해주세요: ${JSON.stringify(args)}`;
    
    try {
      const response = await this.modelManager.queryModel('router', prompt, {
        temperature: 0.3,
        max_tokens: 512
      });

      return {
        content: [{
          type: 'text',
          text: response
        }],
        model_used: 'router',
        execution_time: Date.now()
      };
    } catch (error) {
      this.logger.error('간단한 작업 처리 실패:', error);
      throw error;
    }
  }

  // 통계 및 상태 정보
  getRoutingStatistics() {
    const totalRequests = this.performanceMetrics.get('total_requests');
    const successfulRoutes = this.performanceMetrics.get('successful_routes');
    
    return {
      total_requests: totalRequests,
      successful_routes: successfulRoutes,
      success_rate: totalRequests > 0 ? (successfulRoutes / totalRequests) : 0,
      average_response_time: this.performanceMetrics.get('average_response_time'),
      recent_history: this.routingHistory.slice(-10),
      adaptive_learning: this.adaptiveLearning
    };
  }

  // 설정 업데이트
  async updateRoutingRules(newRules) {
    try {
      this.routingRules = { ...this.routingRules, ...newRules };
      await this.configLoader.updateConfig('routing-rules', this.routingRules);
      this.logger.info('라우팅 규칙 업데이트 완료');
    } catch (error) {
      this.logger.error('라우팅 규칙 업데이트 실패:', error);
      throw error;
    }
  }
}