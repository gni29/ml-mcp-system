// parsers/workflow-builder.js
import { Logger } from '../utils/logger.js';
import fs from 'fs/promises';

export class WorkflowBuilder {
  constructor() {
    this.logger = new Logger();
    this.workflowTemplates = null;
    this.stepDependencies = this.initializeStepDependencies();
    this.stepCompatibility = this.initializeStepCompatibility();
  }

  async initialize() {
    try {
      await this.loadWorkflowTemplates();
      this.logger.info('WorkflowBuilder 초기화 완료');
    } catch (error) {
      this.logger.error('WorkflowBuilder 초기화 실패:', error);
      throw error;
    }
  }

  async loadWorkflowTemplates() {
    try {
      const templatesData = await fs.readFile('./config/pipeline-templates.json', 'utf-8');
      this.workflowTemplates = JSON.parse(templatesData);
    } catch (error) {
      this.logger.warn('워크플로우 템플릿 로드 실패:', error);
      this.workflowTemplates = { 
        common_workflows: {
          basic_analysis: {
            name: 'basic_analysis',
            description: '기본 데이터 분석',
            steps: [
              { type: 'basic', method: 'descriptive_stats', params: {} },
              { type: 'basic', method: 'correlation', params: {} },
              { type: 'visualization', method: '2d.scatter', params: {} }
            ]
          },
          comprehensive_eda: {
            name: 'comprehensive_eda',
            description: '포괄적 탐색적 데이터 분석',
            steps: [
              { type: 'basic', method: 'descriptive_stats', params: {} },
              { type: 'basic', method: 'correlation', params: {} },
              { type: 'basic', method: 'distribution', params: {} },
              { type: 'advanced', method: 'outlier_detection', params: {} },
              { type: 'visualization', method: 'heatmap', params: {} }
            ]
          }
        },
        ml_workflows: {
          classification: {
            name: 'classification_pipeline',
            description: '분류 모델 파이프라인',
            steps: [
              { type: 'basic', method: 'descriptive_stats', params: {} },
              { type: 'advanced', method: 'feature_engineering', params: {} },
              { type: 'ml_traditional', method: 'classification', params: {} },
              { type: 'visualization', method: 'confusion_matrix', params: {} }
            ]
          },
          regression: {
            name: 'regression_pipeline',
            description: '회귀 모델 파이프라인',
            steps: [
              { type: 'basic', method: 'descriptive_stats', params: {} },
              { type: 'basic', method: 'correlation', params: {} },
              { type: 'ml_traditional', method: 'regression', params: {} },
              { type: 'visualization', method: 'scatter', params: {} }
            ]
          }
        }
      };
    }
  }

  initializeStepDependencies() {
    return {
      'basic.descriptive_stats': {
        requires: ['data_loading'],
        provides: ['statistics', 'summary_stats', 'data_overview']
      },
      'basic.correlation': {
        requires: ['basic.descriptive_stats'],
        provides: ['correlation_matrix', 'correlation_results']
      },
      'basic.distribution': {
        requires: ['data_loading'],
        provides: ['distribution_stats', 'normality_tests']
      },
      'advanced.feature_engineering': {
        requires: ['basic.correlation'],
        provides: ['engineered_features', 'feature_importance']
      },
      'advanced.pca': {
        requires: ['basic.descriptive_stats'],
        provides: ['pca_components', 'explained_variance', 'transformed_data']
      },
      'advanced.clustering': {
        requires: ['data_preprocessing'],
        provides: ['cluster_labels', 'cluster_centers', 'cluster_metrics']
      },
      'advanced.outlier_detection': {
        requires: ['basic.descriptive_stats'],
        provides: ['outlier_indices', 'cleaned_data']
      },
      'ml_traditional.classification': {
        requires: ['feature_engineering', 'data_preprocessing'],
        provides: ['trained_model', 'predictions', 'classification_metrics']
      },
      'ml_traditional.regression': {
        requires: ['feature_engineering', 'data_preprocessing'],
        provides: ['trained_model', 'predictions', 'regression_metrics']
      },
      'deep_learning.cnn': {
        requires: ['image_preprocessing'],
        provides: ['trained_model', 'feature_maps', 'predictions']
      },
      'deep_learning.rnn': {
        requires: ['sequence_preprocessing'],
        provides: ['trained_model', 'sequence_predictions']
      },
      'visualization.heatmap': {
        requires: ['correlation_matrix'],
        provides: ['correlation_plot']
      },
      'visualization.scatter': {
        requires: ['data_loading'],
        provides: ['scatter_plot']
      },
      'timeseries.trend_analysis': {
        requires: ['time_series_data'],
        provides: ['trend_components', 'seasonality']
      },
      'timeseries.forecasting': {
        requires: ['trend_analysis'],
        provides: ['forecast_results', 'prediction_intervals']
      }
    };
  }

  initializeStepCompatibility() {
    return {
      'basic': {
        data_types: ['tabular', 'numerical'],
        incompatible_with: [],
        resource_requirements: { memory: 'low', cpu: 'low', gpu: false }
      },
      'advanced': {
        data_types: ['tabular', 'numerical'],
        incompatible_with: [],
        resource_requirements: { memory: 'medium', cpu: 'medium', gpu: false }
      },
      'ml_traditional': {
        data_types: ['tabular', 'numerical'],
        incompatible_with: ['deep_learning'],
        resource_requirements: { memory: 'medium', cpu: 'high', gpu: false }
      },
      'deep_learning': {
        data_types: ['image', 'text', 'tabular'],
        incompatible_with: [],
        resource_requirements: { memory: 'high', cpu: 'high', gpu: true }
      },
      'visualization': {
        data_types: ['all'],
        incompatible_with: [],
        resource_requirements: { memory: 'low', cpu: 'low', gpu: false }
      },
      'timeseries': {
        data_types: ['time_series'],
        incompatible_with: ['ml_traditional'],
        resource_requirements: { memory: 'medium', cpu: 'medium', gpu: false }
      },
      'preprocessing': {
        data_types: ['all'],
        incompatible_with: [],
        resource_requirements: { memory: 'low', cpu: 'low', gpu: false }
      },
      'postprocessing': {
        data_types: ['all'],
        incompatible_with: [],
        resource_requirements: { memory: 'low', cpu: 'low', gpu: false }
      }
    };
  }

  async buildWorkflow(intentAnalysis, queryAnalysis = {}) {
    try {
      this.logger.debug('워크플로우 구축 시작', { intentAnalysis, queryAnalysis });

      // 1. 템플릿 기반 워크플로우 검색
      const templateMatches = this.findMatchingTemplates(intentAnalysis, queryAnalysis);
      
      let workflow;
      let executionPlan;

      if (templateMatches.length > 0) {
        // 가장 적합한 템플릿 선택 및 커스터마이징
        const bestTemplate = templateMatches[0].template;
        workflow = this.customizeTemplate(bestTemplate, intentAnalysis, queryAnalysis);
        this.logger.debug(`템플릿 기반 워크플로우 선택: ${bestTemplate.name}`);
      } else {
        // 동적 워크플로우 생성
        workflow = await this.buildDynamicWorkflow(intentAnalysis, queryAnalysis);
        this.logger.debug('동적 워크플로우 생성');
      }

      // 2. 워크플로우 최적화
      workflow = this.optimizeWorkflow(workflow);

      // 3. 실행 계획 생성
      executionPlan = this.generateExecutionPlan(workflow);

      // 4. 워크플로우 검증
      const validation = this.validateWorkflow(workflow);
      if (!validation.isValid) {
        this.logger.warn('워크플로우 검증 실패:', validation.errors);
        return this.getFallbackWorkflow();
      }

      const result = {
        workflow,
        execution_plan: executionPlan,
        estimated_time: this.estimateExecutionTime(workflow),
        resource_requirements: this.calculateResourceRequirements(workflow),
        validation: validation,
        optimizations: this.suggestOptimizations(workflow)
      };

      this.logger.info('워크플로우 구축 완료', {
        steps: workflow.steps.length,
        estimated_time: result.estimated_time
      });

      return result;

    } catch (error) {
      this.logger.error('워크플로우 구축 실패:', error);
      return this.getFallbackWorkflow();
    }
  }

  findMatchingTemplates(intentAnalysis, queryAnalysis) {
    const matches = [];
    const { suggested_methods, ai_analysis } = intentAnalysis;
    const { data_requirements = {}, parameters = {} } = queryAnalysis;

    for (const [category, workflows] of Object.entries(this.workflowTemplates)) {
      for (const [name, workflow] of Object.entries(workflows)) {
        let score = 0;
        
        // 제안된 메서드와의 매칭 점수
        for (const method of suggested_methods) {
          const [type, methodName] = method.split('.');
          if (workflow.steps.some(step => step.type === type && 
              (step.method === methodName || !methodName))) {
            score += 2;
          }
        }

        // 데이터 타입 매칭
        if (data_requirements.type && 
            this.isWorkflowCompatibleWithDataType(workflow, data_requirements.type)) {
          score += 1;
        }

        // 복잡도 매칭
        if (ai_analysis && ai_analysis.complexity) {
          const workflowComplexity = this.calculateWorkflowComplexity(workflow);
          const complexityDiff = Math.abs(workflowComplexity - ai_analysis.complexity);
          if (complexityDiff < 0.3) {
            score += 1;
          }
        }

        // 파라미터 매칭
        if (this.hasMatchingParameters(workflow, parameters)) {
          score += 1;
        }

        if (score > 0) {
          matches.push({
            template: workflow,
            score: score,
            name: name,
            category: category
          });
        }
      }
    }

    // 점수 순으로 정렬
    return matches.sort((a, b) => b.score - a.score);
  }

  customizeTemplate(template, intentAnalysis, queryAnalysis) {
    const customizedWorkflow = JSON.parse(JSON.stringify(template)); // 깊은 복사
    
    // 파라미터 커스터마이징
    this.customizeParameters(customizedWorkflow, intentAnalysis, queryAnalysis);
    
    // 단계 추가/제거
    this.adjustSteps(customizedWorkflow, intentAnalysis, queryAnalysis);
    
    // 메타데이터 추가
    customizedWorkflow.customized = true;
    customizedWorkflow.original_template = template.name;
    customizedWorkflow.customization_timestamp = new Date().toISOString();
    
    return customizedWorkflow;
  }

  customizeParameters(workflow, intentAnalysis, queryAnalysis) {
    const { parameters } = queryAnalysis;
    
    for (const step of workflow.steps) {
      if (step.params) {
        // 의도 분석에서 추출된 파라미터 적용
        if (intentAnalysis.parameters) {
          Object.assign(step.params, intentAnalysis.parameters);
        }

        // 쿼리 분석에서 추출된 파라미터 적용
        if (parameters) {
          // 타입별 파라미터 매핑
          if (step.type === 'advanced' && step.method === 'clustering') {
            if (parameters.n_clusters) step.params.n_clusters = parameters.n_clusters;
          }
          if (step.type === 'advanced' && step.method === 'pca') {
            if (parameters.n_components) step.params.n_components = parameters.n_components;
          }
          if (step.type === 'ml_traditional') {
            if (parameters.test_size) step.params.test_size = parameters.test_size;
            if (parameters.random_state) step.params.random_state = parameters.random_state;
          }
        }
      }
    }
  }

  adjustSteps(workflow, intentAnalysis, queryAnalysis) {
    const { ai_analysis } = intentAnalysis;

    // 시각화 요구사항에 따른 단계 추가
    if (ai_analysis && ai_analysis.requires_visualization) {
      const hasVisualization = workflow.steps.some(step => step.type === 'visualization');
      if (!hasVisualization) {
        const vizMethod = this.selectVisualizationMethod(intentAnalysis, queryAnalysis);
        workflow.steps.push({
          type: 'visualization',
          method: vizMethod,
          params: this.extractVisualizationParameters(queryAnalysis)
        });
      }
    }

    // 데이터 전처리 요구사항에 따른 단계 추가
    const { data_requirements } = queryAnalysis;
    if (data_requirements && data_requirements.preprocessing === 'extensive') {
      const hasPreprocessing = workflow.steps.some(step => step.type === 'preprocessing');
      if (!hasPreprocessing) {
        workflow.steps.unshift({
          type: 'preprocessing',
          method: 'comprehensive',
          params: {}
        });
      }
    }
  }

  async buildDynamicWorkflow(intentAnalysis, queryAnalysis) {
    const workflow = {
      name: 'dynamic_workflow',
      description: '동적 생성된 워크플로우',
      steps: [],
      metadata: {
        created_at: new Date().toISOString(),
        intent_analysis: intentAnalysis,
        query_analysis: queryAnalysis
      }
    };

    // 1. 데이터 전처리 단계 추가
    const preprocessingSteps = this.buildPreprocessingSteps(queryAnalysis);
    workflow.steps.push(...preprocessingSteps);

    // 2. 주요 분석 단계 추가
    const analysisSteps = this.buildAnalysisSteps(intentAnalysis, queryAnalysis);
    workflow.steps.push(...analysisSteps);

    // 3. 시각화 단계 추가
    const visualizationSteps = this.buildVisualizationSteps(intentAnalysis, queryAnalysis);
    workflow.steps.push(...visualizationSteps);

    // 4. 후처리 단계 추가
    const postprocessingSteps = this.buildPostprocessingSteps(intentAnalysis, queryAnalysis);
    workflow.steps.push(...postprocessingSteps);

    return workflow;
  }

  buildPreprocessingSteps(queryAnalysis) {
    const steps = [];
    const { data_requirements = {} } = queryAnalysis;

    // 데이터 로딩 단계
    steps.push({
      type: 'data',
      method: 'load',
      params: {
        file_type: data_requirements.file_type || 'csv',
        data_type: data_requirements.type || 'tabular'
      }
    });

    // 전처리 단계들
    if (data_requirements.preprocessing === 'extensive') {
      steps.push({
        type: 'preprocessing',
        method: 'handle_missing_values',
        params: { strategy: 'auto' }
      });

      steps.push({
        type: 'advanced',
        method: 'outlier_detection',
        params: { method: 'iqr' }
      });

      steps.push({
        type: 'preprocessing',
        method: 'normalize',
        params: { method: 'standard' }
      });
    } else if (data_requirements.preprocessing === 'moderate') {
      steps.push({
        type: 'preprocessing',
        method: 'handle_missing_values',
        params: { strategy: 'simple' }
      });
    }

    return steps;
  }

  buildAnalysisSteps(intentAnalysis, queryAnalysis) {
    const steps = [];
    const { suggested_methods, primary_analysis } = intentAnalysis;

    // 기본 통계 분석 (딥러닝이 아닌 경우에만)
    if (!suggested_methods.some(method => method.startsWith('deep_learning'))) {
      steps.push({
        type: 'basic',
        method: 'descriptive_stats',
        params: {}
      });
    }

    // 제안된 메서드들 추가
    for (const method of suggested_methods) {
      const [type, ...methodParts] = method.split('.');
      const methodName = methodParts.join('.');

      if (methodName) {
        steps.push({
          type: type,
          method: methodName,
          params: this.extractMethodParameters(method, intentAnalysis, queryAnalysis)
        });
      }
    }

    // 주요 분석이 아직 포함되지 않았다면 추가
    const primaryMethod = `${primary_analysis.category}.${primary_analysis.type}`;
    if (!suggested_methods.includes(primaryMethod)) {
      steps.push({
        type: primary_analysis.category,
        method: primary_analysis.type,
        params: {}
      });
    }

    return steps;
  }

  buildVisualizationSteps(intentAnalysis, queryAnalysis) {
    const steps = [];
    const { ai_analysis } = intentAnalysis;

    if (ai_analysis && ai_analysis.requires_visualization) {
      // 자동 시각화 선택
      const vizMethod = this.selectVisualizationMethod(intentAnalysis, queryAnalysis);
      
      steps.push({
        type: 'visualization',
        method: vizMethod,
        params: this.extractVisualizationParameters(queryAnalysis)
      });
    }

    return steps;
  }

  buildPostprocessingSteps(intentAnalysis, queryAnalysis) {
    const steps = [];

    // 결과 요약 단계
    steps.push({
      type: 'postprocessing',
      method: 'summarize_results',
      params: {}
    });

    // 리포트 생성 단계 (복잡한 분석인 경우)
    if (intentAnalysis.complexity > 0.5) {
      steps.push({
        type: 'postprocessing',
        method: 'generate_report',
        params: {
          format: 'comprehensive',
          include_charts: true
        }
      });
    }

    return steps;
  }

  optimizeWorkflow(workflow) {
    const optimized = JSON.parse(JSON.stringify(workflow));

    // 1. 중복 단계 제거
    optimized.steps = this.removeDuplicateSteps(optimized.steps);

    // 2. 단계 순서 최적화
    optimized.steps = this.optimizeStepOrder(optimized.steps);

    // 3. 종속성 검증
    this.validateDependencies(optimized.steps);

    // 4. 병렬 실행 가능한 단계 식별
    optimized.parallel_groups = this.identifyParallelSteps(optimized.steps);

    return optimized;
  }

  removeDuplicateSteps(steps) {
    const seen = new Set();
    return steps.filter(step => {
      const key = `${step.type}.${step.method}`;
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  }

  optimizeStepOrder(steps) {
    // 종속성을 고려한 위상 정렬
    const dependencyGraph = new Map();
    const inDegree = new Map();

    // 그래프 초기화
    steps.forEach((step, index) => {
      const stepKey = `${step.type}.${step.method}`;
      dependencyGraph.set(index, []);
      inDegree.set(index, 0);
    });

    // 종속성 관계 구축
    steps.forEach((step, index) => {
      const stepKey = `${step.type}.${step.method}`;
      const dependencies = this.stepDependencies[stepKey];
      
      if (dependencies && dependencies.requires) {
        for (const requirement of dependencies.requires) {
          // 이전 단계에서 requirement를 제공하는 단계 찾기
          for (let i = 0; i < index; i++) {
            const prevStep = steps[i];
            const prevStepKey = `${prevStep.type}.${prevStep.method}`;
            const prevDependencies = this.stepDependencies[prevStepKey];
            
            if (prevDependencies && prevDependencies.provides && 
                prevDependencies.provides.includes(requirement)) {
              dependencyGraph.get(i).push(index);
              inDegree.set(index, inDegree.get(index) + 1);
            }
          }
        }
      }
    });

    // 위상 정렬 수행
    const result = [];
    const queue = [];

    // 진입 차수가 0인 노드들을 큐에 추가
    for (const [node, degree] of inDegree) {
      if (degree === 0) {
        queue.push(node);
      }
    }

    while (queue.length > 0) {
      const current = queue.shift();
      result.push(steps[current]);

      // 인접 노드들의 진입 차수 감소
      for (const neighbor of dependencyGraph.get(current)) {
        inDegree.set(neighbor, inDegree.get(neighbor) - 1);
        if (inDegree.get(neighbor) === 0) {
          queue.push(neighbor);
        }
      }
    }

    return result.length === steps.length ? result : steps; // 순환 의존성이 있는 경우 원본 반환
  }

  validateDependencies(steps) {
    const issues = [];
    const availableProvisions = new Set(['data_loading']); // 기본적으로 사용 가능한 것들

    steps.forEach((step, index) => {
      const stepKey = `${step.type}.${step.method}`;
      const dependencies = this.stepDependencies[stepKey];
      
      if (dependencies) {
        const { requires, provides } = dependencies;
        
        // 필요한 의존성 확인
        if (requires) {
          const missingDependencies = requires.filter(req => !availableProvisions.has(req));
          if (missingDependencies.length > 0) {
            issues.push({
              step: stepKey,
              position: index,
              missing_dependencies: missingDependencies
            });
          }
        }
        
        // 제공하는 것들을 사용 가능한 목록에 추가
        if (provides) {
          provides.forEach(provision => availableProvisions.add(provision));
        }
      }
    });

    return issues;
  }

  identifyParallelSteps(steps) {
    const parallelGroups = [];
    const processed = new Set();

    for (let i = 0; i < steps.length; i++) {
      if (processed.has(i)) continue;

      const currentStep = steps[i];
      const group = [i];

      // 현재 단계와 병렬 실행 가능한 단계들 찾기
      for (let j = i + 1; j < steps.length; j++) {
        if (processed.has(j)) continue;

        const nextStep = steps[j];
        if (this.canRunInParallel(currentStep, nextStep, steps.slice(0, j))) {
          group.push(j);
        }
      }

      if (group.length > 1) {
        parallelGroups.push(group);
        group.forEach(index => processed.add(index));
      } else {
        processed.add(i);
      }
    }

    return parallelGroups;
  }

  canRunInParallel(step1, step2, previousSteps) {
    const step1Key = `${step1.type}.${step1.method}`;
    const step2Key = `${step2.type}.${step2.method}`;
    
    const deps1 = this.stepDependencies[step1Key];
    const deps2 = this.stepDependencies[step2Key];

    // 서로의 출력에 의존하지 않는지 확인
    if (deps1 && deps2) {
      const provides1 = deps1.provides || [];
      const provides2 = deps2.provides || [];
      const requires1 = deps1.requires || [];
      const requires2 = deps2.requires || [];

      // step2가 step1의 출력에 의존하거나 그 반대인 경우 병렬 실행 불가
      if (requires2.some(req => provides1.includes(req)) ||
          requires1.some(req => provides2.includes(req))) {
        return false;
      }
    }

    // 호환성 확인
    const compat1 = this.stepCompatibility[step1.type];
    const compat2 = this.stepCompatibility[step2.type];

    if (compat1 && compat2) {
      return !compat1.incompatible_with.includes(step2.type) &&
             !compat2.incompatible_with.includes(step1.type);
    }

    return true;
  }

  generateExecutionPlan(workflow) {
    const plan = {
      total_steps: workflow.steps.length,
      execution_order: [],
      parallel_groups: workflow.parallel_groups || [],
      resource_allocation: {},
      checkpoints: []
    };

    // 실행 순서 계획
    for (let i = 0; i < workflow.steps.length; i++) {
      const step = workflow.steps[i];
      
      // 병렬 그룹 확인
      const parallelGroup = workflow.parallel_groups?.find(group => group.includes(i));
      
      plan.execution_order.push({
        step_index: i,
        step: step,
        parallel_group: parallelGroup ? parallelGroup.indexOf(i) : null,
        estimated_time: this.estimateStepTime(step),
        resource_requirements: this.calculateStepResources(step)
      });
    }

    // 체크포인트 설정
    const checkpointIndices = this.identifyCheckpoints(workflow.steps);
    plan.checkpoints = checkpointIndices.map(index => ({
      step_index: index,
      description: `체크포인트 ${index + 1}`
    }));

    return plan;
  }

  estimateExecutionTime(workflow) {
    let totalTime = 0;
    const parallelGroups = workflow.parallel_groups || [];

    for (let i = 0; i < workflow.steps.length; i++) {
      const step = workflow.steps[i];
      const stepTime = this.estimateStepTime(step);

      // 병렬 그룹에 속한 경우 최대 시간만 추가
      const parallelGroup = parallelGroups.find(group => group.includes(i));
      if (parallelGroup) {
        if (i === parallelGroup[0]) { // 그룹의 첫 번째 단계인 경우
          const groupTimes = parallelGroup.map(index => 
            this.estimateStepTime(workflow.steps[index])
          );
          totalTime += Math.max(...groupTimes);
        }
        // 그룹의 다른 단계들은 시간 추가하지 않음
      } else {
        totalTime += stepTime;
      }
    }

    return totalTime;
  }

  estimateStepTime(step) {
    const baseTimeMap = {
      'basic': 15,
      'advanced': 45,
      'ml_traditional': 120,
      'deep_learning': 600,
      'visualization': 10,
      'timeseries': 60,
      'preprocessing': 30,
      'postprocessing': 15
    };

    const baseTime = baseTimeMap[step.type] || 30;

    // 파라미터에 따른 시간 조정
    let timeMultiplier = 1;
    if (step.params) {
      if (step.params.n_clusters && step.params.n_clusters > 5) {
        timeMultiplier *= 1.5;
      }
      if (step.params.n_components && step.params.n_components > 10) {
        timeMultiplier *= 1.3;
      }
      if (step.params.epochs && step.params.epochs > 50) {
        timeMultiplier *= 2;
      }
    }

    return Math.round(baseTime * timeMultiplier);
  }

  calculateResourceRequirements(workflow) {
    let maxMemory = 0;
    let maxCpu = 0;
    let requiresGpu = false;
    let totalDiskSpace = 0;

    for (const step of workflow.steps) {
      const stepResources = this.calculateStepResources(step);
      maxMemory = Math.max(maxMemory, stepResources.memory_mb);
      maxCpu = Math.max(maxCpu, stepResources.cpu_cores);
      requiresGpu = requiresGpu || stepResources.gpu_required;
      totalDiskSpace += stepResources.disk_space_mb;
    }

    return {
      memory_mb: maxMemory,
      cpu_cores: maxCpu,
      gpu_required: requiresGpu,
      disk_space_mb: totalDiskSpace,
      network_required: workflow.steps.some(step => 
        step.type === 'deep_learning' || step.method === 'download_data'
      )
    };
  }

  calculateStepResources(step) {
    const baseResources = {
      memory_mb: 100,
      cpu_cores: 1,
      gpu_required: false,
      disk_space_mb: 10
    };

    // 타입별 기본 리소스
    const typeResourceMap = {
      'basic': { memory_mb: 100, cpu_cores: 1, gpu_required: false, disk_space_mb: 10 },
      'advanced': { memory_mb: 500, cpu_cores: 2, gpu_required: false, disk_space_mb: 50 },
      'ml_traditional': { memory_mb: 1000, cpu_cores: 4, gpu_required: false, disk_space_mb: 100 },
      'deep_learning': { memory_mb: 4000, cpu_cores: 8, gpu_required: true, disk_space_mb: 500 },
      'visualization': { memory_mb: 200, cpu_cores: 1, gpu_required: false, disk_space_mb: 20 },
      'timeseries': { memory_mb: 300, cpu_cores: 2, gpu_required: false, disk_space_mb: 30 },
      'preprocessing': { memory_mb: 200, cpu_cores: 2, gpu_required: false, disk_space_mb: 20 },
      'postprocessing': { memory_mb: 100, cpu_cores: 1, gpu_required: false, disk_space_mb: 10 }
    };

    const resources = typeResourceMap[step.type] || baseResources;

    // 파라미터에 따른 리소스 조정
    if (step.params) {
      if (step.params.n_estimators && step.params.n_estimators > 100) {
        resources.memory_mb *= 1.5;
        resources.cpu_cores = Math.min(resources.cpu_cores + 1, 8);
      }
      if (step.params.batch_size && step.params.batch_size > 32) {
        resources.memory_mb *= 1.3;
      }
      if (step.params.epochs && step.params.epochs > 100) {
        resources.memory_mb *= 2;
        resources.disk_space_mb *= 1.5;
      }
    }

    return resources;
  }

  identifyCheckpoints(steps) {
    const checkpoints = [];
    
    // 중요한 단계 후에 체크포인트 설정
    steps.forEach((step, index) => {
      if (step.type === 'ml_traditional' || 
          step.type === 'deep_learning' || 
          (step.type === 'advanced' && step.method === 'feature_engineering')) {
        checkpoints.push(index);
      }
    });

    // 최소한 중간에 하나의 체크포인트
    if (checkpoints.length === 0 && steps.length > 3) {
      checkpoints.push(Math.floor(steps.length / 2));
    }

    return checkpoints;
  }

  // 유틸리티 메서드들
  calculateWorkflowComplexity(workflow) {
    const complexityScores = {
      'basic': 0.2,
      'advanced': 0.5,
      'ml_traditional': 0.7,
      'deep_learning': 0.9,
      'visualization': 0.1,
      'timeseries': 0.6,
      'preprocessing': 0.3,
      'postprocessing': 0.2
    };

    let totalComplexity = 0;
    for (const step of workflow.steps) {
      totalComplexity += complexityScores[step.type] || 0.3;
    }

    return Math.min(totalComplexity / workflow.steps.length, 1.0);
  }

  isWorkflowCompatibleWithDataType(workflow, dataType) {
    for (const step of workflow.steps) {
      const compatibility = this.stepCompatibility[step.type];
      if (compatibility && 
          !compatibility.data_types.includes(dataType) && 
          !compatibility.data_types.includes('all')) {
        return false;
      }
    }
    return true;
  }

  hasMatchingParameters(workflow, parameters) {
    if (!parameters || Object.keys(parameters).length === 0) return false;

    return workflow.steps.some(step => {
      if (!step.params) return false;
      
      return Object.keys(step.params).some(param =>
        parameters[param] !== undefined
      );
    });
  }

  selectVisualizationMethod(intentAnalysis, queryAnalysis) {
    const { suggested_methods, keywords } = intentAnalysis;
    const { data_requirements = {} } = queryAnalysis;

    // 키워드 기반 시각화 선택
    if (keywords && keywords.analysis) {
      for (const analysis of keywords.analysis) {
        if (analysis.category === 'visualization') {
          return analysis.type;
        }
      }
    }

    // 분석 타입에 따른 자동 선택
    if (suggested_methods.includes('basic.correlation')) {
      return 'heatmap';
    }
    if (suggested_methods.includes('advanced.clustering')) {
      return 'scatter';
    }
    if (suggested_methods.includes('ml_traditional.classification')) {
      return 'confusion_matrix';
    }
    if (suggested_methods.includes('ml_traditional.regression')) {
      return 'scatter';
    }

    // 데이터 타입에 따른 기본 선택
    switch (data_requirements.type) {
      case 'time_series':
        return 'line';
      case 'categorical':
        return 'bar';
      default:
        return 'scatter';
    }
  }

  extractVisualizationParameters(queryAnalysis) {
    const { parameters = {} } = queryAnalysis;
    
    return {
      figsize: [10, 8],
      title: parameters.title || '',
      x_column: parameters.x_column || null,
      y_column: parameters.y_column || null,
      color_column: parameters.color_column || null
    };
  }

  extractMethodParameters(method, intentAnalysis, queryAnalysis) {
    const [type, methodName] = method.split('.');
    const { parameters = {} } = queryAnalysis;
    const intentParams = intentAnalysis.parameters || {};

    const methodParams = {};

    // 타입별 기본 파라미터 설정
    switch (type) {
      case 'advanced':
        if (methodName === 'clustering') {
          methodParams.n_clusters = parameters.n_clusters || intentParams.n_clusters || 3;
          methodParams.algorithm = parameters.algorithm || 'kmeans';
        } else if (methodName === 'pca') {
          methodParams.n_components = parameters.n_components || intentParams.n_components || 2;
        } else if (methodName === 'outlier_detection') {
          methodParams.method = parameters.outlier_method || 'iqr';
          methodParams.threshold = parameters.threshold || 1.5;
        }
        break;

      case 'ml_traditional':
        methodParams.test_size = parameters.test_size || intentParams.test_size || 0.2;
        methodParams.random_state = parameters.random_state || 42;
        methodParams.cross_validation = parameters.cross_validation || true;
        
        if (methodName === 'classification') {
          methodParams.algorithm = parameters.algorithm || 'random_forest';
        } else if (methodName === 'regression') {
          methodParams.algorithm = parameters.algorithm || 'linear';
        }
        break;

      case 'deep_learning':
        methodParams.epochs = parameters.epochs || 50;
        methodParams.batch_size = parameters.batch_size || 32;
        methodParams.learning_rate = parameters.learning_rate || 0.001;
        break;

      case 'visualization':
        Object.assign(methodParams, this.extractVisualizationParameters(queryAnalysis));
        break;
    }

    return methodParams;
  }

  findDuplicateSteps(steps) {
    const duplicates = [];
    const seen = new Map();

    steps.forEach((step, index) => {
      const key = `${step.type}.${step.method}`;
      if (seen.has(key)) {
        duplicates.push({
          original_index: seen.get(key),
          duplicate_index: index,
          step: key
        });
      } else {
        seen.set(key, index);
      }
    });

    return duplicates;
  }

  getFallbackWorkflow() {
    return {
      workflow: {
        name: 'fallback_workflow',
        description: '기본 분석 워크플로우',
        steps: [
          {
            type: 'basic',
            method: 'descriptive_stats',
            params: {}
          },
          {
            type: 'basic',
            method: 'correlation',
            params: {}
          },
          {
            type: 'visualization',
            method: 'scatter',
            params: {}
          }
        ]
      },
      execution_plan: {
        total_steps: 3,
        execution_order: [
          {
            step_index: 0,
            step: { type: 'basic', method: 'descriptive_stats', params: {} },
            parallel_group: null,
            estimated_time: 15,
            resource_requirements: { memory_mb: 100, cpu_cores: 1, gpu_required: false, disk_space_mb: 10 }
          },
          {
            step_index: 1,
            step: { type: 'basic', method: 'correlation', params: {} },
            parallel_group: null,
            estimated_time: 15,
            resource_requirements: { memory_mb: 100, cpu_cores: 1, gpu_required: false, disk_space_mb: 10 }
          },
          {
            step_index: 2,
            step: { type: 'visualization', method: 'scatter', params: {} },
            parallel_group: null,
            estimated_time: 10,
            resource_requirements: { memory_mb: 200, cpu_cores: 1, gpu_required: false, disk_space_mb: 20 }
          }
        ],
        parallel_groups: [],
        resource_allocation: {},
        checkpoints: [
          {
            step_index: 1,
            description: '중간 체크포인트'
          }
        ]
      },
      estimated_time: 40,
      resource_requirements: {
        memory_mb: 200,
        cpu_cores: 1,
        gpu_required: false,
        disk_space_mb: 40,
        network_required: false
      }
    };
  }

  // 워크플로우 검증 메서드
  validateWorkflow(workflow) {
    const validationResults = {
      isValid: true,
      errors: [],
      warnings: [],
      suggestions: []
    };

    // 기본 구조 검증
    if (!workflow.steps || workflow.steps.length === 0) {
      validationResults.errors.push('워크플로우에 단계가 없습니다.');
      validationResults.isValid = false;
    }

    // 각 단계 검증
    for (let i = 0; i < workflow.steps.length; i++) {
      const step = workflow.steps[i];
      
      if (!step.type || !step.method) {
        validationResults.errors.push(`단계 ${i + 1}: type과 method가 필요합니다.`);
        validationResults.isValid = false;
      }

      // 호환성 검증
      const compatibility = this.stepCompatibility[step.type];
      if (compatibility) {
        const previousSteps = workflow.steps.slice(0, i);
        for (const prevStep of previousSteps) {
          if (compatibility.incompatible_with.includes(prevStep.type)) {
            validationResults.warnings.push(
              `단계 ${i + 1} (${step.type})은 단계 ${previousSteps.indexOf(prevStep) + 1} (${prevStep.type})과 호환되지 않을 수 있습니다.`
            );
          }
        }
      }
    }

    // 종속성 검증
    const dependencyIssues = this.validateDependencies(workflow.steps);
    if (dependencyIssues.length > 0) {
      validationResults.warnings.push(...dependencyIssues.map(issue =>
        `단계 ${issue.position + 1} (${issue.step})의 종속성이 충족되지 않았습니다: ${issue.missing_dependencies.join(', ')}`
      ));
    }

    // 리소스 요구사항 검증
    const resourceReq = this.calculateResourceRequirements(workflow);
    if (resourceReq.memory_mb > 16000) {
      validationResults.warnings.push('높은 메모리 사용량이 예상됩니다. 시스템 리소스를 확인하세요.');
    }

    if (resourceReq.gpu_required) {
      validationResults.suggestions.push('GPU가 필요한 작업입니다. GPU 사용 가능 여부를 확인하세요.');
    }

    return validationResults;
  }

  // 워크플로우 최적화 제안
  suggestOptimizations(workflow) {
    const suggestions = [];

    // 병렬 처리 제안
    const parallelGroups = this.identifyParallelSteps(workflow.steps);
    if (parallelGroups.length > 0) {
      suggestions.push({
        type: 'parallelization',
        message: `${parallelGroups.length}개의 병렬 처리 그룹을 식별했습니다. 실행 시간을 단축할 수 있습니다.`,
        groups: parallelGroups,
        estimated_time_saving: this.calculateParallelTimeSaving(workflow, parallelGroups)
      });
    }

    // 중복 단계 제거 제안
    const duplicates = this.findDuplicateSteps(workflow.steps);
    if (duplicates.length > 0) {
      suggestions.push({
        type: 'deduplication',
        message: '중복된 단계를 제거하여 효율성을 높일 수 있습니다.',
        duplicates: duplicates
      });
    }

    // 시각화 최적화 제안
    const vizSteps = workflow.steps.filter(step => step.type === 'visualization');
    if (vizSteps.length > 3) {
      suggestions.push({
        type: 'visualization_optimization',
        message: '많은 시각화 단계가 있습니다. 대시보드 형태로 통합하는 것을 고려하세요.',
        current_count: vizSteps.length,
        suggested_approach: 'dashboard'
      });
    }

    // 리소스 최적화 제안
    const resourceReq = this.calculateResourceRequirements(workflow);
    if (resourceReq.memory_mb > 8000) {
      suggestions.push({
        type: 'resource_optimization',
        message: '메모리 사용량이 높습니다. 배치 크기를 줄이거나 데이터를 청크 단위로 처리하는 것을 고려하세요.',
        current_memory: resourceReq.memory_mb,
        optimization_techniques: ['batch_processing', 'data_chunking', 'memory_mapping']
      });
    }

    // 단계 순서 최적화 제안
    const orderOptimization = this.analyzeStepOrder(workflow.steps);
    if (orderOptimization.canOptimize) {
      suggestions.push({
        type: 'step_order_optimization',
        message: '단계 순서를 최적화하여 중간 결과를 더 효율적으로 활용할 수 있습니다.',
        current_order: orderOptimization.current,
        suggested_order: orderOptimization.suggested
      });
    }

    return suggestions;
  }

  calculateParallelTimeSaving(workflow, parallelGroups) {
    let totalSaving = 0;
    
    for (const group of parallelGroups) {
      const groupTimes = group.map(index => 
        this.estimateStepTime(workflow.steps[index])
      );
      const sequentialTime = groupTimes.reduce((sum, time) => sum + time, 0);
      const parallelTime = Math.max(...groupTimes);
      totalSaving += sequentialTime - parallelTime;
    }
    
    return totalSaving;
  }

  analyzeStepOrder(steps) {
    // 간단한 휴리스틱: 데이터 전처리 -> 분석 -> 시각화 -> 후처리 순서 확인
    const typeOrder = ['preprocessing', 'basic', 'advanced', 'ml_traditional', 'deep_learning', 'visualization', 'postprocessing'];
    
    let canOptimize = false;
    const currentOrder = steps.map(step => step.type);
    const suggestedOrder = [];

    // 현재 순서가 권장 순서와 다른지 확인
    let lastTypeIndex = -1;
    for (const step of steps) {
      const typeIndex = typeOrder.indexOf(step.type);
      if (typeIndex < lastTypeIndex) {
        canOptimize = true;
        break;
      }
      lastTypeIndex = typeIndex;
    }

    if (canOptimize) {
      // 타입별로 그룹화하여 권장 순서 생성
      const typeGroups = {};
      steps.forEach((step, index) => {
        if (!typeGroups[step.type]) {
          typeGroups[step.type] = [];
        }
        typeGroups[step.type].push(index);
      });

      for (const type of typeOrder) {
        if (typeGroups[type]) {
          suggestedOrder.push(...typeGroups[type]);
        }
      }
    }

    return {
      canOptimize,
      current: currentOrder,
      suggested: canOptimize ? suggestedOrder : null
    };
  }

  // 워크플로우 내보내기/가져오기
  exportWorkflow(workflow, format = 'json') {
    switch (format) {
      case 'json':
        return JSON.stringify(workflow, null, 2);
      case 'yaml':
        // YAML 변환 로직 (실제로는 yaml 라이브러리 필요)
        return this.convertToYaml(workflow);
      default:
        throw new Error(`지원하지 않는 형식: ${format}`);
    }
  }

  convertToYaml(workflow) {
    // 간단한 YAML 변환 (실제로는 yaml 라이브러리 사용 권장)
    let yaml = `name: ${workflow.name}\n`;
    yaml += `description: "${workflow.description}"\n`;
    yaml += `steps:\n`;
    
    workflow.steps.forEach((step, index) => {
      yaml += `  - type: ${step.type}\n`;
      yaml += `    method: ${step.method}\n`;
      if (step.params && Object.keys(step.params).length > 0) {
        yaml += `    params:\n`;
        Object.entries(step.params).forEach(([key, value]) => {
          yaml += `      ${key}: ${JSON.stringify(value)}\n`;
        });
      }
    });
    
    return yaml;
  }

  async importWorkflow(workflowData, format = 'json') {
    let workflow;
    
    switch (format) {
      case 'json':
        workflow = typeof workflowData === 'string' ? 
          JSON.parse(workflowData) : workflowData;
        break;
      case 'yaml':
        // YAML 파싱 로직 (실제로는 yaml 라이브러리 필요)
        workflow = this.parseYaml(workflowData);
        break;
      default:
        throw new Error(`지원하지 않는 형식: ${format}`);
    }

    // 가져온 워크플로우 검증
    const validation = this.validateWorkflow(workflow);
    if (!validation.isValid) {
      throw new Error(`워크플로우 검증 실패: ${validation.errors.join(', ')}`);
    }

    return workflow;
  }

  parseYaml(yamlString) {
    // 간단한 YAML 파서 (실제로는 yaml 라이브러리 사용 권장)
    // 이는 매우 기본적인 구현입니다
    const lines = yamlString.split('\n');
    const workflow = { steps: [] };
    
    let currentStep = null;
    let inParams = false;
    
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith('name:')) {
        workflow.name = trimmed.split(':')[1].trim();
      } else if (trimmed.startsWith('description:')) {
        workflow.description = trimmed.split(':')[1].trim().replace(/"/g, '');
      } else if (trimmed.startsWith('- type:')) {
        if (currentStep) {
          workflow.steps.push(currentStep);
        }
        currentStep = { type: trimmed.split(':')[1].trim(), params: {} };
        inParams = false;
      } else if (trimmed.startsWith('method:') && currentStep) {
        currentStep.method = trimmed.split(':')[1].trim();
      } else if (trimmed === 'params:') {
        inParams = true;
      } else if (inParams && trimmed.includes(':')) {
        const [key, value] = trimmed.split(':');
        try {
          currentStep.params[key.trim()] = JSON.parse(value.trim());
        } catch {
          currentStep.params[key.trim()] = value.trim();
        }
      }
    }
    
    if (currentStep) {
      workflow.steps.push(currentStep);
    }
    
    return workflow;
  }

  // 통계 및 모니터링
  getWorkflowStatistics(workflow) {
    const stats = {
      total_steps: workflow.steps.length,
      step_types: {},
      estimated_resources: this.calculateResourceRequirements(workflow),
      estimated_time: this.estimateExecutionTime(workflow),
      complexity_score: this.calculateWorkflowComplexity(workflow)
    };

    // 단계 타입별 통계
    workflow.steps.forEach(step => {
      stats.step_types[step.type] = (stats.step_types[step.type] || 0) + 1;
    });

    return stats;
  }

  // 워크플로우 템플릿 관리
  async saveWorkflowAsTemplate(workflow, templateName, category = 'custom') {
    try {
      if (!this.workflowTemplates[category]) {
        this.workflowTemplates[category] = {};
      }

      const template = {
        ...workflow,
        name: templateName,
        template: true,
        created_at: new Date().toISOString(),
        usage_count: 0
      };

      this.workflowTemplates[category][templateName] = template;

      // 템플릿 파일에 저장 (실제 구현에서는 파일 시스템에 저장)
      await this.saveTemplatesToFile();

      this.logger.info(`워크플로우 템플릿 저장 완료: ${templateName}`);
      return template;

    } catch (error) {
      this.logger.error('워크플로우 템플릿 저장 실패:', error);
      throw error;
    }
  }

  async saveTemplatesToFile() {
    try {
      const templatesJson = JSON.stringify(this.workflowTemplates, null, 2);
      await fs.writeFile('./config/pipeline-templates.json', templatesJson, 'utf-8');
    } catch (error) {
      this.logger.warn('템플릿 파일 저장 실패:', error);
    }
  }

  listAvailableTemplates() {
    const templates = [];
    
    for (const [category, workflows] of Object.entries(this.workflowTemplates)) {
      for (const [name, workflow] of Object.entries(workflows)) {
        templates.push({
          name: name,
          category: category,
          description: workflow.description,
          steps: workflow.steps.length,
          complexity: this.calculateWorkflowComplexity(workflow),
          usage_count: workflow.usage_count || 0
        });
      }
    }
    
    return templates.sort((a, b) => b.usage_count - a.usage_count);
  }

  // 디버깅 및 로깅
  debugWorkflow(workflow) {
    const debug = {
      workflow_info: {
        name: workflow.name,
        steps: workflow.steps.length,
        has_parallel_groups: !!workflow.parallel_groups
      },
      step_analysis: workflow.steps.map((step, index) => ({
        index: index,
        type: step.type,
        method: step.method,
        has_params: !!step.params && Object.keys(step.params).length > 0,
        estimated_time: this.estimateStepTime(step),
        resource_requirements: this.calculateStepResources(step)
      })),
      dependencies: this.validateDependencies(workflow.steps),
      optimization_opportunities: this.suggestOptimizations(workflow)
    };

    this.logger.debug('워크플로우 디버그 정보:', debug);
    return debug;
  }
}