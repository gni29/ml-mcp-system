// core/smart-router.js
import { Logger } from '../utils/logger.js';
import fs from 'fs/promises';

export class SmartRouter {
  constructor(modelManager) {
    this.modelManager = modelManager;
    this.logger = new Logger();
    this.analysisMethods = null;
    this.pipelineTemplates = null;
    this.deepLearningConfig = null;
    this.visualizationConfig = null;
  }

  async initialize() {
    try {
      // 설정 파일들 로드
      await this.loadConfigurations();
      this.logger.info('SmartRouter 초기화 완료');
    } catch (error) {
      this.logger.error('SmartRouter 초기화 실패:', error);
      throw error;
    }
  }

  async loadConfigurations() {
    try {
      // 분석 방법 설정 로드
      const analysisData = await fs.readFile('./config/analysis-methods.json', 'utf-8');
      this.analysisMethods = JSON.parse(analysisData);

      // 파이프라인 템플릿 로드
      const pipelineData = await fs.readFile('./config/pipeline-templates.json', 'utf-8');
      this.pipelineTemplates = JSON.parse(pipelineData);

      // 딥러닝 설정 로드
      const dlData = await fs.readFile('./config/deep-learning-config.json', 'utf-8');
      this.deepLearningConfig = JSON.parse(dlData);

      // 시각화 설정 로드
      const vizData = await fs.readFile('./config/visualization-config.json', 'utf-8');
      this.visualizationConfig = JSON.parse(vizData);

    } catch (error) {
      this.logger.warn('설정 파일 로드 실패, 기본값 사용:', error);
      this.useDefaultConfigurations();
    }
  }

  useDefaultConfigurations() {
    this.analysisMethods = { basic: {}, advanced: {}, ml_traditional: {} };
    this.pipelineTemplates = { common_workflows: {}, ml_workflows: {} };
    this.deepLearningConfig = { computer_vision: {}, nlp: {} };
    this.visualizationConfig = { chart_types: {} };
  }

  async analyzeComplexQuery(query) {
    try {
      // AI 모델을 사용한 의도 분석
      const intentAnalysis = await this.performIntentAnalysis(query);
      
      // 복잡한 쿼리인지 판단
      if (intentAnalysis.complexity >= 0.7 || intentAnalysis.requiresMultipleSteps) {
        return this.buildWorkflow(intentAnalysis);
      } else {
        return this.buildSimpleTask(intentAnalysis);
      }
    } catch (error) {
      this.logger.error('복잡한 쿼리 분석 실패:', error);
      return this.buildFallbackTask(query);
    }
  }

  async performIntentAnalysis(query) {
    const prompt = `다음 사용자 쿼리를 분석해주세요:
"${query}"

다음 JSON 형식으로 응답해주세요:
{
  "intent": "primary_intent",
  "secondary_intents": ["intent1", "intent2"],
  "complexity": 0.0-1.0,
  "requiresMultipleSteps": true/false,
  "suggestedWorkflow": "workflow_name",
  "dataTypes": ["numeric", "categorical", "text", "image"],
  "analysisTypes": ["basic", "advanced", "ml", "dl", "visualization"],
  "keywords": ["keyword1", "keyword2"],
  "parameters": {
    "param1": "value1"
  }
}

분석 기준:
- 복잡도 0.0-0.3: 단순한 작업 (기본 통계, 단일 차트)
- 복잡도 0.4-0.6: 중간 작업 (상관관계, 단일 ML 모델)
- 복잡도 0.7-1.0: 복잡한 작업 (파이프라인, 딥러닝, 여러 단계)`;

    try {
      const response = await this.modelManager.queryModel('router', prompt, {
        temperature: 0.1,
        max_tokens: 800
      });

      return this.parseIntentResponse(response);
    } catch (error) {
      this.logger.error('의도 분석 실패:', error);
      return this.getDefaultIntent(query);
    }
  }

  parseIntentResponse(response) {
    try {
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]);
      }
    } catch (error) {
      this.logger.warn('의도 분석 응답 파싱 실패:', error);
    }

    return this.getDefaultIntent('');
  }

  getDefaultIntent(query) {
    const lowerQuery = query.toLowerCase();
    
    return {
      intent: this.detectBasicIntent(lowerQuery),
      secondary_intents: [],
      complexity: 0.3,
      requiresMultipleSteps: false,
      suggestedWorkflow: 'basic_analysis',
      dataTypes: ['numeric'],
      analysisTypes: ['basic'],
      keywords: this.extractKeywords(lowerQuery),
      parameters: {}
    };
  }

  detectBasicIntent(query) {
    if (query.includes('분석') || query.includes('analyze')) return 'analyze';
    if (query.includes('시각화') || query.includes('차트') || query.includes('그래프')) return 'visualize';
    if (query.includes('모델') || query.includes('훈련') || query.includes('학습')) return 'train';
    if (query.includes('예측') || query.includes('predict')) return 'predict';
    if (query.includes('클러스터') || query.includes('cluster')) return 'cluster';
    if (query.includes('pca') || query.includes('주성분')) return 'dimensionality_reduction';
    return 'analyze';
  }

  extractKeywords(query) {
    const keywords = [];
    const keywordMap = {
      'pca': ['pca', '주성분', '차원축소'],
      'clustering': ['클러스터', 'cluster', '군집'],
      'correlation': ['상관관계', '상관', 'correlation'],
      'regression': ['회귀', 'regression', '예측'],
      'classification': ['분류', 'classification', '클래스'],
      'visualization': ['시각화', '차트', '그래프', 'plot', 'chart'],
      'deep_learning': ['딥러닝', '신경망', 'neural', 'cnn', 'rnn', 'transformer']
    };

    for (const [key, terms] of Object.entries(keywordMap)) {
      if (terms.some(term => query.includes(term))) {
        keywords.push(key);
      }
    }

    return keywords;
  }

  buildWorkflow(intentAnalysis) {
    const { intent, suggestedWorkflow, keywords, parameters } = intentAnalysis;

    // 미리 정의된 워크플로우 찾기
    const workflow = this.findMatchingWorkflow(suggestedWorkflow, keywords);
    
    if (workflow) {
      return {
        taskType: 'workflow',
        workflow: workflow,
        parameters: parameters,
        estimated_time: this.estimateWorkflowTime(workflow)
      };
    }

    // 동적 워크플로우 생성
    return this.buildDynamicWorkflow(intentAnalysis);
  }

  findMatchingWorkflow(suggestedWorkflow, keywords) {
    // 정확한 워크플로우 이름 매칭
    for (const [category, workflows] of Object.entries(this.pipelineTemplates)) {
      if (workflows[suggestedWorkflow]) {
        return workflows[suggestedWorkflow];
      }
    }

    // 키워드 기반 매칭
    const workflowMap = {
      'pca': 'pca_visualization',
      'clustering': 'clustering_analysis',
      'correlation': 'correlation_feature_engineering',
      'model_comparison': 'model_comparison'
    };

    for (const keyword of keywords) {
      if (workflowMap[keyword]) {
        const workflowName = workflowMap[keyword];
        for (const [category, workflows] of Object.entries(this.pipelineTemplates)) {
          if (workflows[workflowName]) {
            return workflows[workflowName];
          }
        }
      }
    }

    return null;
  }

  buildDynamicWorkflow(intentAnalysis) {
    const { intent, keywords, analysisTypes, parameters } = intentAnalysis;
    const steps = [];

    // 분석 유형별 단계 추가
    if (analysisTypes.includes('basic')) {
      steps.push({
        type: 'basic',
        method: 'descriptive_stats',
        params: {}
      });
    }

    if (keywords.includes('correlation')) {
      steps.push({
        type: 'basic',
        method: 'correlation',
        params: {}
      });
    }

    if (keywords.includes('pca')) {
      steps.push({
        type: 'advanced',
        method: 'pca',
        params: { n_components: parameters.n_components || 3 }
      });
    }

    if (keywords.includes('clustering')) {
      steps.push({
        type: 'ml_traditional',
        method: 'unsupervised.clustering.kmeans',
        params: { n_clusters: parameters.n_clusters || 'auto' }
      });
    }

    if (analysisTypes.includes('visualization') || keywords.includes('visualization')) {
      steps.push({
        type: 'visualization',
        method: this.selectVisualizationMethod(keywords),
        params: {}
      });
    }

    return {
      taskType: 'workflow',
      workflow: {
        name: 'dynamic_workflow',
        description: '동적 생성된 워크플로우',
        steps: steps
      },
      parameters: parameters,
      estimated_time: this.estimateWorkflowTime({ steps })
    };
  }

  selectVisualizationMethod(keywords) {
    if (keywords.includes('pca')) return '3d.scatter_3d';
    if (keywords.includes('clustering')) return '2d.scatter';
    if (keywords.includes('correlation')) return '2d.heatmap';
    return '2d.scatter';
  }

  buildSimpleTask(intentAnalysis) {
    const { intent, keywords, parameters } = intentAnalysis;

    // 단순 작업 매핑
    const taskMap = {
      'analyze': { type: 'basic', method: 'descriptive_stats' },
      'visualize': { type: 'visualization', method: '2d.scatter' },
      'cluster': { type: 'ml_traditional', method: 'unsupervised.clustering.kmeans' },
      'dimensionality_reduction': { type: 'advanced', method: 'pca' }
    };

    const task = taskMap[intent] || taskMap['analyze'];

    return {
      taskType: 'single',
      task: {
        type: task.type,
        method: task.method,
        params: parameters
      },
      estimated_time: 30 // 30초 예상
    };
  }

  buildFallbackTask(query) {
    return {
      taskType: 'single',
      task: {
        type: 'basic',
        method: 'descriptive_stats',
        params: {}
      },
      estimated_time: 30
    };
  }

  estimateWorkflowTime(workflow) {
    if (!workflow.steps) return 60;
    
    const timeMap = {
      'basic': 15,
      'advanced': 30,
      'ml_traditional': 60,
      'deep_learning': 300,
      'visualization': 10
    };

    return workflow.steps.reduce((total, step) => {
      return total + (timeMap[step.type] || 30);
    }, 0);
  }

  getMethodConfig(type, method) {
    const parts = method.split('.');
    let config = this.analysisMethods[type];
    
    for (const part of parts) {
      if (config && config[part]) {
        config = config[part];
      } else {
        return null;
      }
    }
    
    return config;
  }

  getVisualizationConfig(method) {
    const parts = method.split('.');
    let config = this.visualizationConfig.chart_types;
    
    for (const part of parts) {
      if (config && config[part]) {
        config = config[part];
      } else {
        return null;
      }
    }
    
    return config;
  }

  getDeepLearningConfig(domain, task) {
    return this.deepLearningConfig[domain] && this.deepLearningConfig[domain][task];
  }
}
