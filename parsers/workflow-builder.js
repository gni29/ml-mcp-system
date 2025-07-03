'visualization.3d.scatter_3d': {
        requires: ['pca_components', 'transformed_data'],
        provides: ['3d_plot', 'dimensional_insights']
      },
      'visualization.2d.heatmap': {
        requires: ['correlation_matrix'],
        provides: ['heatmap_plot', 'correlation_insights']
      },
      'deep_learning.computer_vision.image_classification': {
        requires: ['image_data', 'preprocessed_images'],
        provides: ['trained_model', 'classification_results', 'model_checkpoints']
      },
      'deep_learning.nlp.text_classification': {
        requires: ['text_data', 'tokenized_text'],
        provides: ['trained_model', 'text_predictions', 'model_checkpoints']
      }
    };
  }

  initializeStepCompatibility() {
    return {
      'basic': {
        compatible_with: ['basic', 'advanced', 'visualization'],
        incompatible_with: ['deep_learning'],
        data_types: ['tabular', 'numeric']
      },
      'advanced': {
        compatible_with: ['basic', 'advanced', 'ml_traditional', 'visualization'],
        incompatible_with: [],
        data_types: ['tabular', 'numeric']
      },
      'ml_traditional': {
        compatible_with: ['basic', 'advanced', 'visualization'],
        incompatible_with: ['deep_learning'],
        data_types: ['tabular', 'numeric', 'categorical']
      },
      'deep_learning': {
        compatible_with: ['visualization'],
        incompatible_with: ['basic', 'advanced', 'ml_traditional'],
        data_types: ['image', 'text', 'audio', 'video']
      },
      'visualization': {
        compatible_with: ['basic', 'advanced', 'ml_traditional', 'deep_learning'],
        incompatible_with: [],
        data_types: ['all']
      }
    };
  }

  async buildWorkflow(intentAnalysis, queryAnalysis) {
    try {
      this.logger.info('워크플로우 구성 시작');

      // 1. 기존 템플릿 매칭 시도
      const templateWorkflow = this.findMatchingTemplate(intentAnalysis, queryAnalysis);
      if (templateWorkflow) {
        return this.customizeTemplate(templateWorkflow, intentAnalysis, queryAnalysis);
      }

      // 2. 동적 워크플로우 생성
      const dynamicWorkflow = await this.buildDynamicWorkflow(intentAnalysis, queryAnalysis);
      
      // 3. 워크플로우 검증 및 최적화
      const optimizedWorkflow = this.optimizeWorkflow(dynamicWorkflow);
      
      // 4. 실행 계획 생성
      const executionPlan = this.generateExecutionPlan(optimizedWorkflow);

      return {
        workflow: optimizedWorkflow,
        execution_plan: executionPlan,
        estimated_time: this.estimateExecutionTime(optimizedWorkflow),
        resource_requirements: this.calculateResourceRequirements(optimizedWorkflow)
      };

    } catch (error) {
      this.logger.error('워크플로우 구성 실패:', error);
      return this.buildFallbackWorkflow(intentAnalysis, queryAnalysis);
    }
  }

  findMatchingTemplate(intentAnalysis, queryAnalysis) {
    const { keywords, suggested_methods } = intentAnalysis;
    
    // 키워드 기반 템플릿 매칭
    const keywordMatches = this.matchTemplatesByKeywords(keywords);
    
    // 메서드 기반 템플릿 매칭
    const methodMatches = this.matchTemplatesByMethods(suggested_methods);
    
    // 쿼리 분석 기반 템플릿 매칭
    const queryMatches = this.matchTemplatesByQuery(queryAnalysis);
    
    // 가장 적합한 템플릿 선택
    const allMatches = [...keywordMatches, ...methodMatches, ...queryMatches];
    if (allMatches.length === 0) return null;
    
    // 매칭 스코어 기반 정렬
    const sortedMatches = allMatches.sort((a, b) => b.score - a.score);
    
    return sortedMatches[0].template;
  }

  matchTemplatesByKeywords(keywords) {
    const matches = [];
    
    for (const [category, workflows] of Object.entries(this.workflowTemplates)) {
      for (const [name, workflow] of Object.entries(workflows)) {
        let score = 0;
        
        // 워크플로우 단계와 키워드 매칭
        for (const step of workflow.steps) {
          const stepKeywords = this.getStepKeywords(step);
          
          if (keywords.analysis) {
            for (const analysis of keywords.analysis) {
              if (stepKeywords.includes(analysis.type)) {
                score += 2;
              }
              if (stepKeywords.includes(analysis.category)) {
                score += 1;
              }
            }
          }
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
    
    return matches;
  }

  matchTemplatesByMethods(methods) {
    const matches = [];
    
    for (const [category, workflows] of Object.entries(this.workflowTemplates)) {
      for (const [name, workflow] of Object.entries(workflows)) {
        let score = 0;
        
        for (const step of workflow.steps) {
          const stepMethod = `${step.type}.${step.method}`;
          
          if (methods.includes(stepMethod)) {
            score += 3; // 정확한 메서드 매칭은 높은 점수
          } else if (methods.some(method => method.startsWith(step.type))) {
            score += 1; // 타입 매칭은 낮은 점수
          }
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
    
    return matches;
  }

  matchTemplatesByQuery(queryAnalysis) {
    const matches = [];
    const { data_requirements, parameters } = queryAnalysis;
    
    for (const [category, workflows] of Object.entries(this.workflowTemplates)) {
      for (const [name, workflow] of Object.entries(workflows)) {
        let score = 0;
        
        // 데이터 타입 매칭
        if (this.isWorkflowCompatibleWithDataType(workflow, data_requirements.data_type)) {
          score += 2;
        }
        
        // 파라미터 매칭
        if (this.hasMatchingParameters(workflow, parameters)) {
          score += 1;
        }
        
        // 전처리 요구사항 매칭
        if (this.matchesPreprocessingRequirements(workflow, data_requirements.preprocessing)) {
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
    
    return matches;
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
    const { data_requirements } = queryAnalysis;

    // 데이터 로딩 단계
    steps.push({
      type: 'data',
      method: 'load',
      params: {
        file_type: data_requirements.file_type,
        data_type: data_requirements.data_type
      }
    });

    // 전처리 단계들
    if (data_requirements.preprocessing.includes('missing_value_handling')) {
      steps.push({
        type: 'preprocessing',
        method: 'handle_missing_values',
        params: { strategy: 'auto' }
      });
    }

    if (data_requirements.preprocessing.includes('outlier_detection')) {
      steps.push({
        type: 'advanced',
        method: 'outlier_detection',
        params: { method: 'iqr' }
      });
    }

    if (data_requirements.preprocessing.includes('normalization')) {
      steps.push({
        type: 'preprocessing',
        method: 'normalize',
        params: { method: 'standard' }
      });
    }

    return steps;
  }

  buildAnalysisSteps(intentAnalysis, queryAnalysis) {
    const steps = [];
    const { suggested_methods, primary_analysis } = intentAnalysis;

    // 기본 통계 분석 (거의 항상 포함)
    if (!suggested_methods.includes('deep_learning')) {
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

      steps.push({
        type: type,
        method: methodName,
        params: this.extractMethodParameters(method, intentAnalysis, queryAnalysis)
      });
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

    // 리포트 생성 단계
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
    const uniqueSteps = [];

    for (const step of steps) {
      const stepKey = `${step.type}.${step.method}`;
      if (!seen.has(stepKey)) {
        seen.add(stepKey);
        uniqueSteps.push(step);
      }
    }

    return uniqueSteps;
  }

  optimizeStepOrder(steps) {
    const orderedSteps = [];
    const remaining = [...steps];

    // 종속성 기반 정렬
    while (remaining.length > 0) {
      let added = false;

      for (let i = 0; i < remaining.length; i++) {
        const step = remaining[i];
        const stepKey = `${step.type}.${step.method}`;
        const dependencies = this.stepDependencies[stepKey];

        if (!dependencies || this.areDependenciesSatisfied(dependencies.requires, orderedSteps)) {
          orderedSteps.push(step);
          remaining.splice(i, 1);
          added = true;
          break;
        }
      }

      if (!added) {
        // 순환 종속성이나 해결할 수 없는 종속성이 있는 경우
        this.logger.warn('종속성 해결 실패, 남은 단계들을 순서대로 추가');
        orderedSteps.push(...remaining);
        break;
      }
    }

    return orderedSteps;
  }

  areDependenciesSatisfied(requires, completedSteps) {
    if (!requires || requires.length === 0) return true;

    const completedStepKeys = completedSteps.map(step => `${step.type}.${step.method}`);
    const completedProvides = completedSteps.flatMap(step => {
      const stepKey = `${step.type}.${step.method}`;
      const stepDep = this.stepDependencies[stepKey];
      return stepDep ? stepDep.provides || [] : [];
    });

    return requires.every(req =>
      completedStepKeys.includes(req) || completedProvides.includes(req)
    );
  }

  validateDependencies(steps) {
    const issues = [];

    for (let i = 0; i < steps.length; i++) {
      const step = steps[i];
      const stepKey = `${step.type}.${step.method}`;
      const dependencies = this.stepDependencies[stepKey];

      if (dependencies && dependencies.requires) {
        const previousSteps = steps.slice(0, i);
        if (!this.areDependenciesSatisfied(dependencies.requires, previousSteps)) {
          issues.push({
            step: stepKey,
            position: i,
            missing_dependencies: dependencies.requires
          });
        }
      }
    }

    if (issues.length > 0) {
      this.logger.warn('종속성 검증 실패:', issues);
    }

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
    let currentGroup = 0;
    for (let i = 0; i < workflow.steps.length; i++) {
      const step = workflow.steps[i];
      
      // 병렬 그룹 확인
      const parallelGroup = workflow.parallel_groups?.find(group => group.includes(i));
      
      plan.execution_order.push({
        step_index: i,
        step: step,
        parallel_group: parallelGroup ? currentGroup : null,
        estimated_time: this.estimateStepTime(step),
        resource_requirements: this.calculateStepResources(step)
      });

      if (parallelGroup && i === Math.max(...parallelGroup)) {
        currentGroup++;
      }
    }

    // 체크포인트 설정
    const checkpointInterval = Math.max(1, Math.floor(workflow.steps.length / 3));
    for (let i = checkpointInterval; i < workflow.steps.length; i += checkpointInterval) {
      plan.checkpoints.push({
        step_index: i,
        description: `중간 체크포인트 ${Math.floor(i / checkpointInterval)}`
      });
    }

    return plan;
  }

  estimateExecutionTime(workflow) {
    const stepTimes = {
      'basic': 15,
      'advanced': 30,
      'ml_traditional': 60,
      'deep_learning': 300,
      'visualization': 10,
      'preprocessing': 20,
      'postprocessing': 10
    };

    let totalTime = 0;
    let parallelTime = 0;
    const parallelGroups = workflow.parallel_groups || [];

    for (let i = 0; i < workflow.steps.length; i++) {
      const step = workflow.steps[i];
      const stepTime = stepTimes[step.type] || 30;

      // 병렬 그룹에 속하는지 확인
      const parallelGroup = parallelGroups.find(group => group.includes(i));
      
      if (parallelGroup) {
        // 병렬 그룹 내에서 가장 긴 시간
        const groupTimes = parallelGroup.map(idx => {
          const groupStep = workflow.steps[idx];
          return stepTimes[groupStep.type] || 30;
        });
        const maxGroupTime = Math.max(...groupTimes);
        
        if (parallelTime === 0) {
          parallelTime = maxGroupTime;
        }
        
        // 그룹의 마지막 단계인 경우 병렬 시간을 총 시간에 추가
        if (i === Math.max(...parallelGroup)) {
          totalTime += parallelTime;
          parallelTime = 0;
        }
      } else {
        totalTime += stepTime;
      }
    }

    return totalTime;
  }

  calculateResourceRequirements(workflow) {
    const requirements = {
      memory_mb: 0,
      cpu_cores: 1,
      gpu_required: false,
      disk_space_mb: 100,
      network_required: false
    };

    for (const step of workflow.steps) {
      const stepRequirements = this.calculateStepResources(step);
      
      requirements.memory_mb = Math.max(requirements.memory_mb, stepRequirements.memory_mb);
      requirements.cpu_cores = Math.max(requirements.cpu_cores, stepRequirements.cpu_cores);
      requirements.gpu_required = requirements.gpu_required || stepRequirements.gpu_required;
      requirements.disk_space_mb += stepRequirements.disk_space_mb;
      requirements.network_required = requirements.network_required || stepRequirements.network_required;
    }

    return requirements;
  }

  estimateStepTime(step) {
    const baseTimes = {
      'basic': 15,
      'advanced': 30,
      'ml_traditional': 60,
      'deep_learning': 300,
      'visualization': 10,
      'preprocessing': 20,
      'postprocessing': 10
    };

    let baseTime = baseTimes[step.type] || 30;

    // 파라미터에 따른 시간 조정
    if (step.params) {
      if (step.params.epochs) {
        baseTime *= Math.log10(step.params.epochs + 1);
      }
      if (step.params.n_clusters && step.params.n_clusters > 10) {
        baseTime *= 1.5;
      }
    }

    return Math.round(baseTime);
  }

  calculateStepResources(step) {
    const baseResources = {
      'basic': { memory_mb: 100, cpu_cores: 1, gpu_required: false, disk_space_mb: 10 },
      'advanced': { memory_mb: 500, cpu_cores: 2, gpu_required: false, disk_space_mb: 50 },
      'ml_traditional': { memory_mb: 1000, cpu_cores: 4, gpu_required: false, disk_space_mb: 100 },
      'deep_learning': { memory_mb: 4000, cpu_cores: 8, gpu_required: true, disk_space_mb: 500 },
      'visualization': { memory_mb: 200, cpu_cores: 1, gpu_required: false, disk_space_mb: 20 },
      'preprocessing': { memory_mb: 300, cpu_cores: 2, gpu_required: false, disk_space_mb: 30 },
      'postprocessing': { memory_mb: 100, cpu_cores: 1, gpu_required: false, disk_space_mb: 10 }
    };

    return baseResources[step.type] || baseResources['basic'];
  }

  // 유틸리티 메서드들
  getStepKeywords(step) {
    const keywordMap = {
      'basic': ['basic', 'stats', 'descriptive', 'correlation', 'distribution'],
      'advanced': ['advanced', 'pca', 'clustering', 'outlier', 'feature'],
      'ml_traditional': ['ml', 'machine', 'learning', 'regression', 'classification'],
      'deep_learning': ['deep', 'neural', 'cnn', 'rnn', 'transformer'],
      'visualization': ['plot', 'chart', 'graph', 'visual', 'scatter', 'heatmap']
    };

    return keywordMap[step.type] || [];
  }

  isWorkflowCompatibleWithDataType(workflow, dataType) {
    for (const step of workflow.steps) {
      const compatibility = this.stepCompatibility[step.type];
      if (compatibility && !compatibility.data_types.includes(dataType) && !compatibility.data_types.includes('all')) {
        return false;
      }
    }
    return true;
  }

  hasMatchingParameters(workflow, parameters) {
    return workflow.steps.some(step => {
      if (!step.params) return false;
      
      return Object.keys(step.params).some(param =>
        parameters.model[param] !== undefined ||
        parameters.visualization[param] !== undefined ||
        parameters.data[param] !== undefined
      );
    });
  }

  matchesPreprocessingRequirements(workflow, preprocessing) {
    if (!preprocessing || preprocessing.length === 0) return true;
    
    return preprocessing.some(req =>
      workflow.steps.some(step =>
        step.method && step.method.includes(req.replace('_', ''))
      )
    );
  }

  customizeParameters(workflow, intentAnalysis, queryAnalysis) {
    const { parameters } = queryAnalysis;
    
    for (const step of workflow.steps) {
      if (step.params) {
        // 모델 파라미터 적용
        Object.assign(step.params, parameters.model);
        
        // 시각화 파라미터 적용
        if (step.type === 'visualization') {
          Object.assign(step.params, parameters.visualization);
        }
        
        // 데이터 파라미터 적용
        if (step.type === 'preprocessing' || step.type === 'data') {
          Object.assign(step.params, parameters.data);
        }
      }
    }
  }

  adjustSteps(workflow, intentAnalysis, queryAnalysis) {
    // 복잡도에 따른 단계 조정
    if (intentAnalysis.complexity < 0.3) {
      // 간단한 분석만 유지
      workflow.steps = workflow.steps.filter(step =>
        step.type === 'basic' || step.type === 'visualization'
      );
    } else if (intentAnalysis.complexity > 0.8) {
      // 추가 분석 단계 삽입
      const additionalSteps = this.getAdditionalStepsForComplexAnalysis();
      workflow.steps.push(...additionalSteps);
    }
  }

  getAdditionalStepsForComplexAnalysis() {
    return [
      {
        type: 'advanced',
        method: 'feature_importance',
        params: {}
      },
      {
        type: 'postprocessing',
        method: 'sensitivity_analysis',
        params: {}
      }
    ];
  }

  extractMethodParameters(method, intentAnalysis, queryAnalysis) {
    const { parameters } = queryAnalysis;
    const baseParams = {};

    // 메서드별 기본 파라미터 설정
    if (method.includes('clustering')) {
      baseParams.n_clusters = parameters.model.n_clusters || 'auto';
    }
    
    if (method.includes('pca')) {
      baseParams.n_components = parameters.model.n_components || 3;
    }
    
    if (method.includes('regression') || method.includes('classification')) {
      baseParams.test_size = parameters.model.test_size || 0.2;
      baseParams.random_state = parameters.model.random_state || 42;
    }

    return baseParams;
  }

  selectVisualizationMethod(intentAnalysis, queryAnalysis) {
    const { keywords } = intentAnalysis;
    
    // PCA 결과가 있으면 3D 시각화
    if (keywords.analysis?.some(a => a.type === 'pca')) {
      return '3d.scatter_3d';
    }
    
    // 상관관계 분석이 있으면 히트맵
    if (keywords.analysis?.some(a => a.type === 'correlation')) {
      return '2d.heatmap';
    }
    
    // 기본적으로 산점도
    return '2d.scatter';
  }

  extractVisualizationParameters(queryAnalysis) {
    const { parameters } = queryAnalysis;
    return parameters.visualization || {};
  }

  buildFallbackWorkflow(intentAnalysis, queryAnalysis) {
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
            type: 'visualization',
            method: '2d.scatter',
            params: {}
          }
        ]
      },
      execution_plan: {
        total_steps: 2,
        execution_order: [],
        parallel_groups: [],
        resource_allocation: {},
        checkpoints: []
      },
      estimated_time: 30,
      resource_requirements: {// parsers/workflow-builder.js
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
      this.workflowTemplates = { common_workflows: {}, ml_workflows: {} };
    }
  }

  initializeStepDependencies() {
    return {
      'basic.correlation': {
        requires: ['basic.descriptive_stats'],
        provides: ['correlation_matrix', 'correlation_results']
      },
      'advanced.feature_engineering': {
        requires: ['basic.correlation'],
        provides: ['engineered_features', 'feature_importance']
      },
      'advanced.pca': {
        requires: ['basic.descriptive_stats'],
        provides: ['pca_components', 'explained_variance', 'transformed_data']
      },
      'ml_traditional.supervised.regression': {
        requires: ['data_preprocessing'],
        provides: ['trained_model', 'predictions', 'model_metrics']
      },
      'ml_traditional.supervised.classification': {
        requires: ['data_preprocessing'],
        provides: ['trained_model', 'predictions', 'classification_report']
      },
      'ml_traditional.unsupervised.clustering': {
        requires: ['data_preprocessing'],
        provides: ['cluster_labels', 'cluster_centers', 'cluster_metrics']
      },
      'visualization.2d.scatter': {
        requires: ['numeric_data'],
        provides: ['scatter_plot', 'plot_insights']
      },
      '
        memory_mb: 200,
           cpu_cores: 1,
           gpu_required: false,
           disk_space_mb: 20,
           network_required: false
         }
       };
     }
   }
