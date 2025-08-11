// core/router.js
import { Logger } from '../utils/logger.js';
import { ConfigLoader } from '../utils/config-loader.js';

export class Router {
  constructor(modelManager) {
    this.modelManager = modelManager;
    this.logger = new Logger();
    this.configLoader = new ConfigLoader();
    this.routingRules = null;
    this.analysisMethodsConfig = null;
    this.pipelineTemplatesConfig = null;
    this.toolMappings = new Map();
    this.complexityThresholds = {
      simple: 0.3,
      medium: 0.6,
      complex: 0.8
    };
  }

  async initialize() {
    try {
      await this.loadConfigurations();
      this.setupToolMappings();
      this.logger.info('Router 초기화 완료');
    } catch (error) {
      this.logger.error('Router 초기화 실패:', error);
      this.useDefaultConfigurations();
    }
  }

  async loadConfigurations() {
    try {
      // 라우팅 규칙 로드
      this.routingRules = await this.configLoader.loadConfig('routing-rules.json');
      
      // 분석 방법 설정 로드
      this.analysisMethodsConfig = await this.configLoader.loadConfig('analysis-methods.json');
      
      // 파이프라인 템플릿 로드
      this.pipelineTemplatesConfig = await this.configLoader.loadConfig('pipeline-templates.json');
      
      this.logger.info('라우팅 설정 로드 완료');
    } catch (error) {
      this.logger.warn('설정 파일 로드 실패, 기본값 사용:', error);
      this.useDefaultConfigurations();
    }
  }

  useDefaultConfigurations() {
    this.routingRules = {
      simple_queries: {
        keywords: ['안녕', '도움말', '상태', '모드', '종료', '버전'],
        maxComplexity: 0.3,
        model: 'router',
        tools: ['system']
      },
      data_operations: {
        keywords: ['로드', '불러오기', '데이터', '파일', '읽기', 'load', 'read'],
        complexity: [0.2, 0.6],
        model: 'router',
        tools: ['data-loader', 'data-validator']
      },
      basic_analysis: {
        keywords: ['분석', '통계', '요약', '기술통계', '상관관계', 'analysis', 'stats'],
        complexity: [0.3, 0.7],
        model: 'router',
        tools: ['basic-analyzer', 'data-loader']
      },
      advanced_analysis: {
        keywords: ['고급분석', '클러스터링', 'pca', '주성분', '이상치'],
        complexity: [0.6, 0.9],
        model: 'processor',
        tools: ['advanced-analyzer', 'basic-analyzer']
      },
      ml_operations: {
        keywords: ['모델', '훈련', '예측', '머신러닝', '딥러닝', 'training', 'prediction'],
        minComplexity: 0.7,
        model: 'processor',
        tools: ['trainer', 'predictor', 'evaluator']
      },
      visualization: {
        keywords: ['시각화', '차트', '그래프', '플롯', 'plot', 'chart', 'visualization'],
        complexity: [0.4, 0.8],
        model: 'router',
        tools: ['chart-generator', 'plot-manager']
      }
    };

    this.analysisMethodsConfig = {
      basic: ['descriptive_stats', 'correlation', 'distribution'],
      advanced: ['pca', 'clustering', 'outlier_detection'],
      ml: ['classification', 'regression', 'ensemble']
    };

    this.pipelineTemplatesConfig = {
      data_exploration: {
        steps: ['load_data', 'validate_data', 'basic_analysis', 'visualization']
      },
      ml_workflow: {
        steps: ['load_data', 'preprocess', 'train_model', 'evaluate', 'predict']
      }
    };
  }

  setupToolMappings() {
    // 도구별 매핑 설정
    this.toolMappings.set('data-loader', {
      path: 'tools/data/data-loader.js',
      methods: ['loadCSV', 'loadExcel', 'loadJSON', 'loadParquet', 'loadHDF5'],
      complexity: 0.2
    });

    this.toolMappings.set('data-validator', {
      path: 'tools/data/data-validator.js',
      methods: ['validateData', 'checkDataTypes', 'findMissingValues'],
      complexity: 0.3
    });

    this.toolMappings.set('basic-analyzer', {
      path: 'tools/analysis/basic-analyzer.js',
      methods: ['descriptiveStats', 'correlation', 'distribution'],
      complexity: 0.4
    });

    this.toolMappings.set('advanced-analyzer', {
      path: 'tools/analysis/advanced-analyzer.js',
      methods: ['pca', 'clustering', 'outlierDetection'],
      complexity: 0.7
    });

    this.toolMappings.set('time-series-analyzer', {
      path: 'tools/analysis/time-series-analyzer.js',
      methods: ['trendAnalysis', 'seasonalityAnalysis', 'forecasting'],
      complexity: 0.6
    });

    this.toolMappings.set('trainer', {
      path: 'tools/ml/trainer.js',
      methods: ['trainClassification', 'trainRegression', 'trainClustering'],
      complexity: 0.8
    });

    this.toolMappings.set('predictor', {
      path: 'tools/ml/predictor.js',
      methods: ['predict', 'batchPredict', 'realTimePredict'],
      complexity: 0.7
    });

    this.toolMappings.set('evaluator', {
      path: 'tools/ml/evaluator.js',
      methods: ['evaluateModel', 'crossValidation', 'hyperparameterTuning'],
      complexity: 0.8
    });

    this.toolMappings.set('chart-generator', {
      path: 'tools/visualization/chart-generator.js',
      methods: ['generateChart', 'createPlot', 'interactivePlot'],
      complexity: 0.5
    });

    this.toolMappings.set('plot-manager', {
      path: 'tools/visualization/plot-manager.js',
      methods: ['managePlots', 'savePlot', 'exportPlot'],
      complexity: 0.4
    });
  }

  async route(toolName, args) {
    try {
      // 1. 시스템 도구 확인
      if (this.isSystemTool(toolName)) {
        return {
          taskType: 'system',
          model: 'router',
          tools: [toolName],
          complexity: 0.1,
          priority: 'high'
        };
      }

      // 2. 쿼리 복잡도 분석
      const complexity = await this.analyzeComplexity(toolName, args);
      
      // 3. 의도 파악
      const intent = await this.analyzeIntent(toolName, args);
      
      // 4. 적절한 모델 결정
      const selectedModel = this.selectModel(intent, complexity);
      
      // 5. 필요한 도구들 결정
      const requiredTools = this.selectTools(intent, toolName, args);
      
      // 6. 실행 계획 생성
      const executionPlan = this.createExecutionPlan(intent, requiredTools, complexity);

      return {
        taskType: intent.category,
        model: selectedModel,
        tools: requiredTools,
        complexity: complexity,
        priority: this.determinePriority(complexity, intent),
        executionPlan: executionPlan,
        estimatedTime: this.estimateExecutionTime(requiredTools, complexity),
        resourceRequirements: this.calculateResourceRequirements(requiredTools, complexity)
      };

    } catch (error) {
      this.logger.error('라우팅 실패:', error);
      return this.createFallbackRoute(toolName, args);
    }
  }

  async analyzeComplexity(toolName, args) {
    let complexity = 0.3; // 기본값

    try {
      // 도구 자체의 복잡도
      const toolMapping = this.toolMappings.get(toolName);
      if (toolMapping) {
        complexity = toolMapping.complexity;
      }

      // 인자의 복잡도 분석
      if (args) {
        // 데이터 크기 고려
        if (args.dataSize) {
          if (args.dataSize > 1000000) complexity += 0.3; // 1M 이상
          else if (args.dataSize > 100000) complexity += 0.2; // 100K 이상
          else if (args.dataSize > 10000) complexity += 0.1; // 10K 이상
        }

        // 파라미터 개수 고려
        const paramCount = Object.keys(args).length;
        if (paramCount > 10) complexity += 0.2;
        else if (paramCount > 5) complexity += 0.1;

        // 특정 복잡한 옵션들 고려
        if (args.deepLearning) complexity += 0.3;
        if (args.crossValidation) complexity += 0.2;
        if (args.hyperparameterTuning) complexity += 0.3;
        if (args.ensembleMethods) complexity += 0.2;
      }

      return Math.min(complexity, 1.0); // 최대값 1.0으로 제한
    } catch (error) {
      this.logger.warn('복잡도 분석 실패, 기본값 사용:', error);
      return 0.5;
    }
  }

  async analyzeIntent(toolName, args) {
    try {
      // 도구 이름 기반 의도 파악
      for (const [category, config] of Object.entries(this.routingRules)) {
        if (config.keywords) {
          const isMatch = config.keywords.some(keyword => 
            toolName.toLowerCase().includes(keyword.toLowerCase()) ||
            (args && JSON.stringify(args).toLowerCase().includes(keyword.toLowerCase()))
          );
          
          if (isMatch) {
            return {
              category: category,
              confidence: 0.8,
              keywords: config.keywords.filter(k => 
                toolName.toLowerCase().includes(k.toLowerCase())
              )
            };
          }
        }
      }

      // 도구 매핑 기반 의도 파악
      if (this.toolMappings.has(toolName)) {
        const toolMapping = this.toolMappings.get(toolName);
        const category = this.categorizeByPath(toolMapping.path);
        
        return {
          category: category,
          confidence: 0.7,
          toolBased: true
        };
      }

      // 기본 의도
      return {
        category: 'general',
        confidence: 0.3,
        fallback: true
      };

    } catch (error) {
      this.logger.warn('의도 분석 실패:', error);
      return { category: 'general', confidence: 0.1, error: true };
    }
  }

  categorizeByPath(path) {
    if (path.includes('/data/')) return 'data_operations';
    if (path.includes('/analysis/')) return 'basic_analysis';
    if (path.includes('/ml/')) return 'ml_operations';
    if (path.includes('/visualization/')) return 'visualization';
    return 'general';
  }

  selectModel(intent, complexity) {
    try {
      // 복잡도 기반 모델 선택
      if (complexity >= this.complexityThresholds.complex) {
        return 'processor'; // 복잡한 작업은 프로세서 모델
      }

      // 의도 기반 모델 선택
      const categoryConfig = this.routingRules[intent.category];
      if (categoryConfig && categoryConfig.model) {
        return categoryConfig.model;
      }

      // 기본적으로 복잡도에 따라 결정
      if (complexity >= this.complexityThresholds.medium) {
        return 'processor';
      } else {
        return 'router';
      }

    } catch (error) {
      this.logger.warn('모델 선택 실패, 기본값 사용:', error);
      return 'router';
    }
  }

  selectTools(intent, primaryTool, args) {
    try {
      const tools = [primaryTool];

      // 의도 기반 추가 도구 선택
      const categoryConfig = this.routingRules[intent.category];
      if (categoryConfig && categoryConfig.tools) {
        categoryConfig.tools.forEach(tool => {
          if (!tools.includes(tool)) {
            tools.push(tool);
          }
        });
      }

      // 의존성 기반 도구 추가
      const dependencies = this.getDependencies(primaryTool, args);
      dependencies.forEach(dep => {
        if (!tools.includes(dep)) {
          tools.push(dep);
        }
      });

      return tools;

    } catch (error) {
      this.logger.warn('도구 선택 실패:', error);
      return [primaryTool];
    }
  }

  getDependencies(toolName, args) {
    const dependencies = [];

    // 데이터가 필요한 경우 data-loader 추가
    if (this.requiresData(toolName) && args && !args.data) {
      dependencies.push('data-loader');
    }

    // 데이터 검증이 필요한 경우
    if (this.requiresValidation(toolName)) {
      dependencies.push('data-validator');
    }

    // 시각화가 필요한 경우
    if (this.requiresVisualization(toolName, args)) {
      dependencies.push('chart-generator');
    }

    return dependencies;
  }

  requiresData(toolName) {
    const dataRequiredTools = ['basic-analyzer', 'advanced-analyzer', 'trainer', 'predictor'];
    return dataRequiredTools.includes(toolName);
  }

  requiresValidation(toolName) {
    const validationRequiredTools = ['trainer', 'advanced-analyzer'];
    return validationRequiredTools.includes(toolName);
  }

  requiresVisualization(toolName, args) {
    if (args && args.visualization === false) return false;
    
    const vizTools = ['basic-analyzer', 'advanced-analyzer', 'time-series-analyzer'];
    return vizTools.includes(toolName);
  }

  createExecutionPlan(intent, tools, complexity) {
    const plan = {
      steps: [],
      parallel: [],
      sequential: []
    };

    try {
      // 순차 실행이 필요한 단계들
      const sequentialTools = ['data-loader', 'data-validator'];
      const parallelTools = ['chart-generator', 'plot-manager'];

      tools.forEach(tool => {
        if (sequentialTools.includes(tool)) {
          plan.sequential.push({
            tool: tool,
            order: sequentialTools.indexOf(tool),
            critical: true
          });
        } else if (parallelTools.includes(tool)) {
          plan.parallel.push({
            tool: tool,
            critical: false
          });
        } else {
          plan.steps.push({
            tool: tool,
            dependencies: this.getDependencies(tool),
            critical: true
          });
        }
      });

      // 실행 순서 정렬
      plan.sequential.sort((a, b) => a.order - b.order);

      return plan;

    } catch (error) {
      this.logger.warn('실행 계획 생성 실패:', error);
      return {
        steps: tools.map(tool => ({ tool, critical: true })),
        parallel: [],
        sequential: []
      };
    }
  }

  determinePriority(complexity, intent) {
    if (intent.category === 'system') return 'high';
    if (complexity >= this.complexityThresholds.complex) return 'high';
    if (complexity >= this.complexityThresholds.medium) return 'medium';
    return 'low';
  }

  estimateExecutionTime(tools, complexity) {
    const baseTime = 1000; // 1초
    const complexityMultiplier = 1 + complexity * 2;
    const toolMultiplier = tools.length * 0.5;
    
    return Math.round(baseTime * complexityMultiplier * toolMultiplier);
  }

  calculateResourceRequirements(tools, complexity) {
    const baseMemory = 100; // 100MB
    const baseCPU = 1;

    const memoryRequirement = baseMemory + (complexity * 500) + (tools.length * 50);
    const cpuRequirement = baseCPU + Math.floor(complexity * 2);

    return {
      memory_mb: Math.round(memoryRequirement),
      cpu_cores: Math.min(cpuRequirement, 8), // 최대 8코어
      gpu_required: complexity >= 0.8 && tools.some(t => t.includes('deep-learning')),
      disk_space_mb: 50 + (tools.length * 10)
    };
  }

  isSystemTool(toolName) {
    const systemTools = [
      'help', 'status', 'version', 'exit', 'quit',
      'list-tools', 'clear', 'reset', 'debug'
    ];
    return systemTools.includes(toolName.toLowerCase());
  }

  createFallbackRoute(toolName, args) {
    return {
      taskType: 'fallback',
      model: 'router',
      tools: [toolName],
      complexity: 0.3,
      priority: 'medium',
      executionPlan: {
        steps: [{ tool: toolName, critical: true }],
        parallel: [],
        sequential: []
      },
      estimatedTime: 2000,
      resourceRequirements: {
        memory_mb: 200,
        cpu_cores: 1,
        gpu_required: false,
        disk_space_mb: 20
      },
      fallback: true
    };
  }

  // 라우팅 통계 및 성능 모니터링
  getRoutingStats() {
    return {
      totalRoutes: this.routingStats?.total || 0,
      successfulRoutes: this.routingStats?.successful || 0,
      failedRoutes: this.routingStats?.failed || 0,
      averageComplexity: this.routingStats?.avgComplexity || 0,
      mostUsedModel: this.routingStats?.mostUsedModel || 'router',
      mostUsedTools: this.routingStats?.mostUsedTools || []
    };
  }

  // 설정 리로드
  async reloadConfigurations() {
    try {
      await this.loadConfigurations();
      this.setupToolMappings();
      this.logger.info('라우팅 설정 리로드 완료');
      return true;
    } catch (error) {
      this.logger.error('설정 리로드 실패:', error);
      return false;
    }
  }
}