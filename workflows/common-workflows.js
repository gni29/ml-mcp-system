// workflows/common-workflows.js - 완성된 버전
import { Logger } from '../utils/logger.js';

export class CommonWorkflows {
  constructor() {
    this.logger = new Logger();
    this.workflowTemplates = this.initializeCommonWorkflows();
    this.stepKeywords = this.initializeStepKeywords();
    this.resourceProfiles = this.initializeResourceProfiles();
  }

  initializeCommonWorkflows() {
    return {
      data_exploration: {
        name: 'data_exploration',
        description: '기본 데이터 탐색 및 분석',
        category: 'common',
        steps: [
          {
            type: 'data_loading',
            method: 'load_dataset',
            params: {
              file_path: '{file_path}',
              format: '{format}'
            },
            outputs: ['dataset', 'metadata'],
            requires: []
          },
          {
            type: 'basic',
            method: 'descriptive_stats',
            params: {},
            outputs: ['statistics', 'summary'],
            requires: ['dataset']
          },
          {
            type: 'basic',
            method: 'missing_values_analysis',
            params: {},
            outputs: ['missing_data_report'],
            requires: ['dataset']
          },
          {
            type: 'basic',
            method: 'data_types_analysis',
            params: {},
            outputs: ['data_types_report'],
            requires: ['dataset']
          },
          {
            type: 'visualization',
            method: 'distribution_plots',
            params: {
              plot_type: 'histogram',
              columns: 'numeric'
            },
            outputs: ['distribution_plots'],
            requires: ['dataset']
          },
          {
            type: 'visualization',
            method: 'correlation_heatmap',
            params: {
              method: 'pearson'
            },
            outputs: ['correlation_heatmap'],
            requires: ['dataset']
          }
        ],
        estimated_time: 120,
        resource_requirements: {
          memory_mb: 500,
          cpu_cores: 1,
          gpu_required: false
        }
      },

      correlation_analysis: {
        name: 'correlation_analysis',
        description: '상관관계 분석 및 시각화',
        category: 'common',
        steps: [
          {
            type: 'basic',
            method: 'correlation',
            params: {
              method: 'pearson'
            },
            outputs: ['correlation_matrix'],
            requires: ['dataset']
          },
          {
            type: 'basic',
            method: 'correlation_significance',
            params: {
              alpha: 0.05
            },
            outputs: ['significance_test'],
            requires: ['correlation_matrix']
          },
          {
            type: 'visualization',
            method: 'correlation_heatmap',
            params: {
              annotate: true,
              color_scheme: 'coolwarm'
            },
            outputs: ['correlation_heatmap'],
            requires: ['correlation_matrix']
          },
          {
            type: 'visualization',
            method: 'correlation_network',
            params: {
              threshold: 0.7
            },
            outputs: ['correlation_network'],
            requires: ['correlation_matrix']
          }
        ],
        estimated_time: 60,
        resource_requirements: {
          memory_mb: 300,
          cpu_cores: 1,
          gpu_required: false
        }
      },

      outlier_detection: {
        name: 'outlier_detection',
        description: '이상치 탐지 및 분석',
        category: 'common',
        steps: [
          {
            type: 'advanced',
            method: 'outlier_detection',
            params: {
              method: 'isolation_forest',
              contamination: 0.1
            },
            outputs: ['outlier_scores', 'outlier_labels'],
            requires: ['dataset']
          },
          {
            type: 'advanced',
            method: 'outlier_analysis',
            params: {
              methods: ['iqr', 'zscore', 'isolation_forest']
            },
            outputs: ['outlier_comparison'],
            requires: ['dataset']
          },
          {
            type: 'visualization',
            method: 'outlier_plots',
            params: {
              plot_types: ['boxplot', 'scatter', 'histogram']
            },
            outputs: ['outlier_visualizations'],
            requires: ['outlier_scores', 'outlier_labels']
          }
        ],
        estimated_time: 90,
        resource_requirements: {
          memory_mb: 400,
          cpu_cores: 1,
          gpu_required: false
        }
      },

      time_series_basic: {
        name: 'time_series_basic',
        description: '기본 시계열 분석',
        category: 'time_series',
        steps: [
          {
            type: 'timeseries',
            method: 'time_series_decomposition',
            params: {
              model: 'additive',
              period: 'auto'
            },
            outputs: ['decomposition_results'],
            requires: ['dataset']
          },
          {
            type: 'timeseries',
            method: 'stationarity_test',
            params: {
              test: 'adf'
            },
            outputs: ['stationarity_results'],
            requires: ['dataset']
          },
          {
            type: 'timeseries',
            method: 'seasonality_analysis',
            params: {
              max_period: 365
            },
            outputs: ['seasonality_results'],
            requires: ['dataset']
          },
          {
            type: 'visualization',
            method: 'time_series_plots',
            params: {
              plot_types: ['line', 'seasonal', 'residual']
            },
            outputs: ['time_series_visualizations'],
            requires: ['decomposition_results']
          }
        ],
        estimated_time: 150,
        resource_requirements: {
          memory_mb: 600,
          cpu_cores: 1,
          gpu_required: false
        }
      },

      feature_engineering: {
        name: 'feature_engineering',
        description: '피처 엔지니어링 및 선택',
        category: 'advanced',
        steps: [
          {
            type: 'advanced',
            method: 'feature_scaling',
            params: {
              method: 'standard',
              columns: 'numeric'
            },
            outputs: ['scaled_features'],
            requires: ['dataset']
          },
          {
            type: 'advanced',
            method: 'feature_encoding',
            params: {
              categorical_method: 'onehot',
              handle_unknown: 'ignore'
            },
            outputs: ['encoded_features'],
            requires: ['dataset']
          },
          {
            type: 'advanced',
            method: 'feature_selection',
            params: {
              method: 'selectkbest',
              k: 10
            },
            outputs: ['selected_features'],
            requires: ['encoded_features']
          },
          {
            type: 'advanced',
            method: 'feature_importance',
            params: {
              method: 'mutual_info'
            },
            outputs: ['feature_importance'],
            requires: ['selected_features']
          }
        ],
        estimated_time: 120,
        resource_requirements: {
          memory_mb: 800,
          cpu_cores: 2,
          gpu_required: false
        }
      },

      quick_analysis: {
        name: 'quick_analysis',
        description: '빠른 데이터 분석',
        category: 'common',
        steps: [
          {
            type: 'basic',
            method: 'descriptive_stats',
            params: {},
            outputs: ['statistics'],
            requires: ['dataset']
          },
          {
            type: 'basic',
            method: 'correlation',
            params: {
              method: 'pearson'
            },
            outputs: ['correlation_matrix'],
            requires: ['dataset']
          },
          {
            type: 'visualization',
            method: 'summary_dashboard',
            params: {
              charts: ['histogram', 'correlation', 'boxplot']
            },
            outputs: ['summary_dashboard'],
            requires: ['statistics', 'correlation_matrix']
          }
        ],
        estimated_time: 45,
        resource_requirements: {
          memory_mb: 200,
          cpu_cores: 1,
          gpu_required: false
        }
      },

      data_quality_assessment: {
        name: 'data_quality_assessment',
        description: '데이터 품질 평가',
        category: 'common',
        steps: [
          {
            type: 'basic',
            method: 'missing_values_analysis',
            params: {},
            outputs: ['missing_analysis'],
            requires: ['dataset']
          },
          {
            type: 'basic',
            method: 'duplicate_analysis',
            params: {},
            outputs: ['duplicate_analysis'],
            requires: ['dataset']
          },
          {
            type: 'basic',
            method: 'data_consistency_check',
            params: {},
            outputs: ['consistency_report'],
            requires: ['dataset']
          },
          {
            type: 'advanced',
            method: 'data_profiling',
            params: {
              include_correlations: true
            },
            outputs: ['data_profile'],
            requires: ['dataset']
          },
          {
            type: 'visualization',
            method: 'quality_dashboard',
            params: {},
            outputs: ['quality_dashboard'],
            requires: ['missing_analysis', 'duplicate_analysis', 'consistency_report']
          }
        ],
        estimated_time: 90,
        resource_requirements: {
          memory_mb: 400,
          cpu_cores: 1,
          gpu_required: false
        }
      }
    };
  }

  initializeStepKeywords() {
    return {
      'data_loading': ['load', 'import', 'read', 'file', 'dataset'],
      'basic.descriptive_stats': ['statistics', 'summary', 'describe', 'mean', 'std'],
      'basic.correlation': ['correlation', 'relationship', 'association'],
      'basic.missing_values_analysis': ['missing', 'null', 'na', 'incomplete'],
      'basic.data_types_analysis': ['types', 'dtype', 'schema', 'format'],
      'visualization.distribution_plots': ['histogram', 'distribution', 'frequency'],
      'visualization.correlation_heatmap': ['heatmap', 'correlation', 'matrix'],
      'visualization.correlation_network': ['network', 'graph', 'connections'],
      'advanced.outlier_detection': ['outlier', 'anomaly', 'unusual', 'extreme'],
      'advanced.feature_engineering': ['feature', 'transform', 'encode', 'scale'],
      'timeseries.time_series_decomposition': ['decomposition', 'trend', 'seasonal'],
      'timeseries.stationarity_test': ['stationarity', 'stationary', 'adf', 'kpss']
    };
  }

  initializeResourceProfiles() {
    return {
      'light': {
        memory_mb: 200,
        cpu_cores: 1,
        gpu_required: false,
        disk_space_mb: 50
      },
      'medium': {
        memory_mb: 500,
        cpu_cores: 1,
        gpu_required: false,
        disk_space_mb: 100
      },
      'heavy': {
        memory_mb: 1000,
        cpu_cores: 2,
        gpu_required: false,
        disk_space_mb: 200
      },
      'gpu_required': {
        memory_mb: 2000,
        cpu_cores: 4,
        gpu_required: true,
        disk_space_mb: 500
      }
    };
  }

  getWorkflow(workflowName) {
    return this.workflowTemplates[workflowName] || null;
  }

  getAllWorkflows() {
    return this.workflowTemplates;
  }

  getWorkflowsByCategory(category) {
    const workflows = {};
    for (const [name, workflow] of Object.entries(this.workflowTemplates)) {
      if (workflow.category === category) {
        workflows[name] = workflow;
      }
    }
    return workflows;
  }

  getWorkflowsByKeywords(keywords) {
    const matches = [];
    
    for (const [name, workflow] of Object.entries(this.workflowTemplates)) {
      let score = 0;
      
      // 워크플로우 이름과 설명에서 키워드 매칭
      for (const keyword of keywords) {
        if (workflow.name.includes(keyword.toLowerCase()) ||
            workflow.description.toLowerCase().includes(keyword.toLowerCase())) {
          score += 2;
        }
      }
      
      // 단계별 키워드 매칭
      for (const step of workflow.steps) {
        const stepKey = `${step.type}.${step.method}`;
        const stepKeywords = this.stepKeywords[stepKey] || [];
        
        for (const keyword of keywords) {
          if (stepKeywords.includes(keyword.toLowerCase())) {
            score += 1;
          }
        }
      }
      
      if (score > 0) {
        matches.push({
          name,
          workflow,
          score
        });
      }
    }
    
    return matches.sort((a, b) => b.score - a.score);
  }

  getStepKeywords(step) {
    const stepKey = `${step.type}.${step.method}`;
    return this.stepKeywords[stepKey] || [];
  }

  customizeWorkflow(workflowName, customizations) {
    const baseWorkflow = this.getWorkflow(workflowName);
    if (!baseWorkflow) {
      throw new Error(`워크플로우 '${workflowName}'을 찾을 수 없습니다.`);
    }

    const customizedWorkflow = JSON.parse(JSON.stringify(baseWorkflow));
    
    // 파라미터 커스터마이징
    if (customizations.parameters) {
      customizedWorkflow.steps.forEach(step => {
        const stepKey = `${step.type}.${step.method}`;
        if (customizations.parameters[stepKey]) {
          step.params = {
            ...step.params,
            ...customizations.parameters[stepKey]
          };
        }
      });
    }

    // 단계 추가
    if (customizations.additionalSteps) {
      for (const additionalStep of customizations.additionalSteps) {
        const insertIndex = additionalStep.insertAfter !== undefined ?
          additionalStep.insertAfter + 1 : customizedWorkflow.steps.length;
        
        customizedWorkflow.steps.splice(insertIndex, 0, additionalStep.step);
      }
    }

    // 단계 제거
    if (customizations.removeSteps) {
      customizedWorkflow.steps = customizedWorkflow.steps.filter(
        step => !customizations.removeSteps.includes(`${step.type}.${step.method}`)
      );
    }

    // 워크플로우 메타데이터 업데이트
    if (customizations.name) {
      customizedWorkflow.name = customizations.name;
    }
    if (customizations.description) {
      customizedWorkflow.description = customizations.description;
    }

    // 시간 및 리소스 요구사항 재계산
    customizedWorkflow.estimated_time = this.estimateExecutionTime(customizedWorkflow);
    customizedWorkflow.resource_requirements = this.calculateResourceRequirements(customizedWorkflow);

    return customizedWorkflow;
  }

  validateWorkflow(workflow) {
    const validationResult = {
      valid: true,
      errors: [],
      warnings: [],
      suggestions: []
    };

    // 기본 구조 검증
    if (!workflow.steps || !Array.isArray(workflow.steps) || workflow.steps.length === 0) {
      validationResult.valid = false;
      validationResult.errors.push('워크플로우에 유효한 단계가 없습니다.');
      return validationResult;
    }

    // 각 단계 검증
    workflow.steps.forEach((step, index) => {
      // 필수 필드 검증
      if (!step.type || !step.method) {
        validationResult.valid = false;
        validationResult.errors.push(`단계 ${index + 1}: type과 method가 필요합니다.`);
      }

      // 파라미터 검증
      if (!step.params) {
        validationResult.warnings.push(`단계 ${index + 1}: 파라미터가 정의되지 않았습니다.`);
        step.params = {};
      }

      // 출력 검증
      if (!step.outputs || step.outputs.length === 0) {
        validationResult.warnings.push(`단계 ${index + 1}: 출력이 정의되지 않았습니다.`);
      }

      // 의존성 검증
      if (step.requires && step.requires.length > 0) {
        for (const requirement of step.requires) {
          const isAvailable = this.isOutputAvailable(workflow.steps, index, requirement);
          if (!isAvailable) {
            validationResult.valid = false;
            validationResult.errors.push(
              `단계 ${index + 1}: 필요한 입력 '${requirement}'를 이전 단계에서 찾을 수 없습니다.`
            );
          }
        }
      }
    });

    // 순환 의존성 검증
    const circularDependency = this.detectCircularDependencies(workflow);
    if (circularDependency) {
      validationResult.valid = false;
      validationResult.errors.push('순환 의존성이 감지되었습니다.');
    }

    // 성능 최적화 제안
    const optimizations = this.suggestOptimizations(workflow);
    validationResult.suggestions.push(...optimizations);

    return validationResult;
  }

  isOutputAvailable(steps, currentIndex, requiredOutput) {
    for (let i = 0; i < currentIndex; i++) {
      if (steps[i].outputs && steps[i].outputs.includes(requiredOutput)) {
        return true;
      }
    }
    return false;
  }

  detectCircularDependencies(workflow) {
    // 단순한 순환 의존성 검사
    const visited = new Set();
    const recursionStack = new Set();
    
    const hasCircularDependency = (stepIndex) => {
      if (recursionStack.has(stepIndex)) {
        return true;
      }
      
      if (visited.has(stepIndex)) {
        return false;
      }
      
      visited.add(stepIndex);
      recursionStack.add(stepIndex);
      
      const step = workflow.steps[stepIndex];
      if (step.requires) {
        for (const requirement of step.requires) {
          // 해당 요구사항을 제공하는 단계 찾기
          const providerIndex = workflow.steps.findIndex(s =>
            s.outputs && s.outputs.includes(requirement)
          );
          
          if (providerIndex !== -1 && hasCircularDependency(providerIndex)) {
            return true;
          }
        }
      }
      
      recursionStack.delete(stepIndex);
      return false;
    };
    
    for (let i = 0; i < workflow.steps.length; i++) {
      if (hasCircularDependency(i)) {
        return true;
      }
    }
    
    return false;
  }

  suggestOptimizations(workflow) {
    const suggestions = [];
    
    // 병렬 실행 가능한 단계 찾기
    const parallelGroups = this.findParallelExecutionGroups(workflow);
    if (parallelGroups.length > 0) {
      suggestions.push(`${parallelGroups.length}개의 병렬 실행 그룹을 만들어 성능을 개선할 수 있습니다.`);
    }
    
    // 중복 계산 찾기
    const duplicateCalculations = this.findDuplicateCalculations(workflow);
    if (duplicateCalculations.length > 0) {
      suggestions.push('중복 계산을 캐싱하여 성능을 개선할 수 있습니다.');
    }
    
    // 리소스 집약적인 단계 경고
    const heavySteps = workflow.steps.filter(step =>
      this.getStepResourceProfile(step).memory_mb > 1000
    );
    if (heavySteps.length > 0) {
      suggestions.push(`${heavySteps.length}개의 리소스 집약적인 단계가 있습니다. 메모리 사용량을 고려하세요.`);
    }
    
    return suggestions;
  }

  findParallelExecutionGroups(workflow) {
    const groups = [];
    const processed = new Set();
    
    for (let i = 0; i < workflow.steps.length; i++) {
      if (processed.has(i)) continue;
      
      const parallelSteps = [i];
      const currentStep = workflow.steps[i];
      
      // 같은 입력을 사용하는 다른 단계들 찾기
      for (let j = i + 1; j < workflow.steps.length; j++) {
        if (processed.has(j)) continue;
        
        const otherStep = workflow.steps[j];
        if (this.canExecuteInParallel(currentStep, otherStep)) {
          parallelSteps.push(j);
          processed.add(j);
        }
      }
      
      if (parallelSteps.length > 1) {
        groups.push(parallelSteps);
      }
      
      processed.add(i);
    }
    
    return groups;
  }

  canExecuteInParallel(step1, step2) {
    // 같은 입력을 사용하고 서로 의존하지 않는 경우
    const step1Requires = step1.requires || [];
    const step2Requires = step2.requires || [];
    const step1Outputs = step1.outputs || [];
    const step2Outputs = step2.outputs || [];
    
    // 서로의 출력을 입력으로 사용하지 않는 경우
    const step1UsesStep2Output = step1Requires.some(req => step2Outputs.includes(req));
    const step2UsesStep1Output = step2Requires.some(req => step1Outputs.includes(req));
    
    return !step1UsesStep2Output && !step2UsesStep1Output;
  }

  findDuplicateCalculations(workflow) {
    const calculations = new Map();
    const duplicates = [];
    
    workflow.steps.forEach((step, index) => {
      const key = `${step.type}.${step.method}:${JSON.stringify(step.params)}`;
      
      if (calculations.has(key)) {
        duplicates.push({
          original: calculations.get(key),
          duplicate: index,
          step: step
        });
      } else {
        calculations.set(key, index);
      }
    });
    
    return duplicates;
  }

  estimateExecutionTime(workflow) {
    if (workflow.estimated_time) {
      return workflow.estimated_time;
    }

    const stepTimes = {
      'data_loading': 30,
      'basic': 20,
      'advanced': 40,
      'visualization': 25,
      'timeseries': 35,
      'ml_traditional': 60,
      'deep_learning': 180
    };

    let totalTime = 0;
    workflow.steps.forEach(step => {
      const baseTime = stepTimes[step.type] || 30;
      
      // 파라미터에 따른 시간 조정
      let timeMultiplier = 1;
      if (step.params) {
        // 데이터 크기에 따른 조정
        if (step.params.sample_size && step.params.sample_size > 10000) {
          timeMultiplier *= 1.5;
        }
        
        // 복잡한 알고리즘에 따른 조정
        if (step.params.method === 'isolation_forest' ||
            step.params.method === 'neural_network') {
          timeMultiplier *= 2;
        }
      }
      
      totalTime += baseTime * timeMultiplier;
    });

    return Math.round(totalTime);
  }

  calculateResourceRequirements(workflow) {
    let maxMemory = 0;
    let maxCpuCores = 1;
    let gpuRequired = false;
    let totalDiskSpace = 0;

    workflow.steps.forEach(step => {
      const stepProfile = this.getStepResourceProfile(step);
      
      maxMemory = Math.max(maxMemory, stepProfile.memory_mb);
      maxCpuCores = Math.max(maxCpuCores, stepProfile.cpu_cores);
      gpuRequired = gpuRequired || stepProfile.gpu_required;
      totalDiskSpace += stepProfile.disk_space_mb || 0;
    });

    return {
      memory_mb: maxMemory,
      cpu_cores: maxCpuCores,
      gpu_required: gpuRequired,
      disk_space_mb: totalDiskSpace,
      network_required: workflow.steps.some(step =>
        step.type === 'data_loading' && step.params && step.params.url
      )
    };
  }

  getStepResourceProfile(step) {
    const baseProfiles = {
      'data_loading': { memory_mb: 200, cpu_cores: 1, gpu_required: false, disk_space_mb: 50 },
      'basic': { memory_mb: 150, cpu_cores: 1, gpu_required: false, disk_space_mb: 20 },
      'advanced': { memory_mb: 500, cpu_cores: 2, gpu_required: false, disk_space_mb: 100 },
      'visualization': { memory_mb: 300, cpu_cores: 1, gpu_required: false, disk_space_mb: 50 },
      'timeseries': { memory_mb: 400, cpu_cores: 1, gpu_required: false, disk_space_mb: 30 },
      'ml_traditional': { memory_mb: 800, cpu_cores: 2, gpu_required: false, disk_space_mb: 200 },
      'deep_learning': { memory_mb: 2000, cpu_cores: 4, gpu_required: true, disk_space_mb: 500 }
    };

    const baseProfile = baseProfiles[step.type] || baseProfiles['basic'];
    
    // 파라미터에 따른 조정
    let memoryMultiplier = 1;
    if (step.params) {
      if (step.params.sample_size && step.params.sample_size > 100000) {
        memoryMultiplier *= 2;
      }
      
      if (step.method === 'neural_network' || step.method === 'deep_learning') {
        memoryMultiplier *= 3;
      }
    }

    return {
      ...baseProfile,
      memory_mb: Math.round(baseProfile.memory_mb * memoryMultiplier)
    };
  }

  generateExecutionPlan(workflow) {
    const plan = {
      total_steps: workflow.steps.length,
      estimated_time: this.estimateExecutionTime(workflow),
      execution_order: [],
      dependencies: {},
      parallel_groups: this.findParallelExecutionGroups(workflow),
      resource_allocation: this.calculateResourceRequirements(workflow),
      checkpoints: this.generateCheckpoints(workflow)
    };

    // 실행 순서 생성
    workflow.steps.forEach((step, index) => {
      const stepProfile = this.getStepResourceProfile(step);
      
      plan.execution_order.push({
        step_index: index,
        step: step,
        parallel_group: this.findParallelGroup(index, plan.parallel_groups),
        estimated_time: this.estimateStepTime(step),
        resource_requirements: stepProfile
      });
    });

    // 종속성 분석
    this.analyzeDependencies(workflow, plan);

    return plan;
  }

  findParallelGroup(stepIndex, parallelGroups) {
    for (let i = 0; i < parallelGroups.length; i++) {
      if (parallelGroups[i].includes(stepIndex)) {
        return i;
      }
    }
    return null;
  }

  generateCheckpoints(workflow) {
    const checkpoints = [];
    
    // 데이터 로딩 후 체크포인트
    const dataLoadingIndex = workflow.steps.findIndex(step => step.type === 'data_loading');
    if (dataLoadingIndex !== -1) {
      checkpoints.push({
        step_index: dataLoadingIndex,
        name: 'data_loaded',
        description: '데이터 로딩 완료'
      });
    }
    
    // 기본 분석 후 체크포인트
    const basicAnalysisIndex = workflow.steps.findIndex(step =>
      step.type === 'basic' && step.method === 'descriptive_stats'
    );
    if (basicAnalysisIndex !== -1) {
      checkpoints.push({
        step_index: basicAnalysisIndex,
        name: 'basic_analysis_complete',
        description: '기본 분석 완료'
      });
    }
    
    // 중간 지점 체크포인트
    const midPoint = Math.floor(workflow.steps.length / 2);
    if (midPoint > 0) {
      checkpoints.push({
        step_index: midPoint,
        name: 'midpoint_checkpoint',
        description: '중간 체크포인트'
      });
    }
    
    return checkpoints;
  }

  estimateStepTime(step) {
    const baseTimes = {
      'data_loading': 30,
      'basic': 20,
      'advanced': 40,
      'visualization': 25,
      'timeseries': 35,
      'ml_traditional': 60,
      'deep_learning': 180
    };

    const baseTime = baseTimes[step.type] || 30;
    
    // 메서드별 시간 조정
    const methodMultipliers = {
      'isolation_forest': 1.5,
      'neural_network': 2.0,
      'deep_learning': 3.0,
      'feature_selection': 1.2,
      'correlation_network': 1.3,
      'time_series_decomposition': 1.4
    };
    
    const multiplier = methodMultipliers[step.method] || 1.0;
    return Math.round(baseTime * multiplier);
  }

  analyzeDependencies(workflow, plan) {
    const outputs = new Set();
    
    workflow.steps.forEach((step, index) => {
      const dependencies = [];
      
      // 입력 요구사항 확인
      if (step.requires && step.requires.length > 0) {
        step.requires.forEach(requirement => {
          if (!outputs.has(requirement)) {
            dependencies.push(requirement);
          }
        });
      }
      
      plan.dependencies[index] = dependencies;
      
      // 출력 추가
      if (step.outputs && step.outputs.length > 0) {
        step.outputs.forEach(output => outputs.add(output));
      }
    });
  }

  // 워크플로우 내보내기/가져오기
  exportWorkflow(workflowName, format = 'json') {
    const workflow = this.getWorkflow(workflowName);
    if (!workflow) {
      throw new Error(`워크플로우 '${workflowName}'을 찾을 수 없습니다.`);
    }

    switch (format) {
      case 'json':
        return JSON.stringify(workflow, null, 2);
      case 'yaml':
        return this.convertToYAML(workflow);
      case 'summary':
        return this.generateWorkflowSummary(workflow);
      default:
        return workflow;
    }
  }

  convertToYAML(workflow) {
    // 간단한 YAML 변환 (실제 구현에서는 yaml 라이브러리 사용)
    let yaml = `name: ${workflow.name}\n`;
    yaml += `description: ${workflow.description}\n`;
    yaml += `category: ${workflow.category}\n`;
    yaml += `steps:\n`;
    
    workflow.steps.forEach((step, index) => {
      yaml += `  - type: ${step.type}\n`;
      yaml += `    method: ${step.method}\n`;
      yaml += `    params:\n`;
      
      for (const [key, value] of Object.entries(step.params || {})) {
        yaml += `      ${key}: ${value}\n`;
      }
      
      if (step.outputs && step.outputs.length > 0) {
        yaml += `    outputs:\n`;
        step.outputs.forEach(output => {
          yaml += `      - ${output}\n`;
        });
      }
      
      if (step.requires && step.requires.length > 0) {
        yaml += `    requires:\n`;
        step.requires.forEach(requirement => {
          yaml += `      - ${requirement}\n`;
        });
      }
      
      yaml += '\n';
    });
    
    return yaml;
  }

  generateWorkflowSummary(workflow) {
    let summary = `# ${workflow.name}\n\n`;
    summary += `**설명**: ${workflow.description}\n`;
    summary += `**카테고리**: ${workflow.category}\n`;
    summary += `**예상 실행 시간**: ${workflow.estimated_time || this.estimateExecutionTime(workflow)}초\n`;
    summary += `**단계 수**: ${workflow.steps.length}\n\n`;
    
    summary += `## 실행 단계\n\n`;
    workflow.steps.forEach((step, index) => {
      summary += `${index + 1}. **${step.type}.${step.method}**\n`;
      summary += `   - 예상 시간: ${this.estimateStepTime(step)}초\n`;
      
      if (step.outputs && step.outputs.length > 0) {
        summary += `   - 출력: ${step.outputs.join(', ')}\n`;
      }
      
      if (step.requires && step.requires.length > 0) {
        summary += `   - 필요 입력: ${step.requires.join(', ')}\n`;
      }
      
      summary += '\n';
    });
    
    const resourceReq = this.calculateResourceRequirements(workflow);
    summary += `## 리소스 요구사항\n\n`;
    summary += `- **메모리**: ${resourceReq.memory_mb}MB\n`;
    summary += `- **CPU 코어**: ${resourceReq.cpu_cores}개\n`;
    summary += `- **GPU 필요**: ${resourceReq.gpu_required ? '예' : '아니오'}\n`;
    summary += `- **디스크 공간**: ${resourceReq.disk_space_mb}MB\n`;
    
    return summary;
  }

  importWorkflow(workflowData, format = 'json') {
    let workflow;
    
    try {
      switch (format) {
        case 'json':
          workflow = typeof workflowData === 'string' ?
            JSON.parse(workflowData) : workflowData;
          break;
        case 'yaml':
          workflow = this.parseYAML(workflowData);
          break;
        default:
          workflow = workflowData;
      }

      const validation = this.validateWorkflow(workflow);
      if (!validation.valid) {
        throw new Error(`워크플로우 검증 실패: ${validation.errors.join(', ')}`);
      }

      // 워크플로우 정규화
      workflow = this.normalizeWorkflow(workflow);

      return workflow;
    } catch (error) {
      this.logger.error('워크플로우 임포트 실패:', error);
      throw error;
    }
  }

  parseYAML(yamlString) {
    // 간단한 YAML 파싱 (실제 구현에서는 yaml 라이브러리 사용)
    const lines = yamlString.split('\n');
    const workflow = {
      steps: []
    };
    
    let currentStep = null;
    let currentSection = null;
    
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) continue;
      
      if (trimmed.startsWith('name:')) {
        workflow.name = trimmed.split(':')[1].trim();
      } else if (trimmed.startsWith('description:')) {
        workflow.description = trimmed.split(':')[1].trim();
      } else if (trimmed.startsWith('category:')) {
        workflow.category = trimmed.split(':')[1].trim();
      } else if (trimmed.startsWith('- type:')) {
        if (currentStep) {
          workflow.steps.push(currentStep);
        }
        currentStep = {
          type: trimmed.split(':')[1].trim(),
          params: {}
        };
      } else if (trimmed.startsWith('method:') && currentStep) {
        currentStep.method = trimmed.split(':')[1].trim();
      }
      // 더 복잡한 파싱 로직은 yaml 라이브러리 사용
    }
    
    if (currentStep) {
      workflow.steps.push(currentStep);
    }
    
    return workflow;
  }

  normalizeWorkflow(workflow) {
    // 워크플로우 정규화
    const normalized = {
      name: workflow.name || 'unnamed_workflow',
      description: workflow.description || 'No description',
      category: workflow.category || 'custom',
      steps: [],
      estimated_time: null,
      resource_requirements: null
    };

    // 단계 정규화
    workflow.steps.forEach(step => {
      const normalizedStep = {
        type: step.type,
        method: step.method,
        params: step.params || {},
        outputs: step.outputs || [],
        requires: step.requires || []
      };
      
      normalized.steps.push(normalizedStep);
    });

    // 시간 및 리소스 요구사항 계산
    normalized.estimated_time = this.estimateExecutionTime(normalized);
    normalized.resource_requirements = this.calculateResourceRequirements(normalized);

    return normalized;
  }

  // 워크플로우 추천 시스템
  recommendWorkflow(userQuery, dataContext = {}) {
    const recommendations = [];
    
    // 키워드 기반 추천
    const keywords = this.extractKeywords(userQuery);
    const keywordMatches = this.getWorkflowsByKeywords(keywords);
    
    // 데이터 컨텍스트 기반 추천
    const contextMatches = this.getWorkflowsByDataContext(dataContext);
    
    // 모든 매칭 결과 합치기
    const allMatches = [...keywordMatches, ...contextMatches];
    
    // 중복 제거 및 점수 합산
    const scoreMap = new Map();
    allMatches.forEach(match => {
      const existing = scoreMap.get(match.name);
      if (existing) {
        existing.score += match.score;
      } else {
        scoreMap.set(match.name, match);
      }
    });
    
    // 점수순 정렬
    const sortedMatches = Array.from(scoreMap.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, 5); // 상위 5개
    
    return sortedMatches.map(match => ({
      name: match.name,
      workflow: match.workflow,
      score: match.score,
      reason: this.generateRecommendationReason(match.workflow, userQuery, dataContext)
    }));
  }

  extractKeywords(query) {
    const keywords = [];
    const commonWords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'];
    
    const words = query.toLowerCase()
      .replace(/[^\w\s]/g, '')
      .split(/\s+/)
      .filter(word => word.length > 2 && !commonWords.includes(word));
    
    return words;
  }

  getWorkflowsByDataContext(dataContext) {
    const matches = [];
    
    for (const [name, workflow] of Object.entries(this.workflowTemplates)) {
      let score = 0;
      
      // 데이터 타입 기반 매칭
      if (dataContext.hasTimeSeries && workflow.category === 'time_series') {
        score += 3;
      }
      
      if (dataContext.hasNumericalData && workflow.steps.some(s => s.type === 'basic')) {
        score += 2;
      }
      
      if (dataContext.hasCategoricalData && workflow.steps.some(s => s.method === 'feature_encoding')) {
        score += 2;
      }
      
      // 데이터 크기 기반 매칭
      if (dataContext.isLargeDataset && workflow.name.includes('quick')) {
        score -= 1; // 큰 데이터셋에는 빠른 분석 비추천
      }
      
      if (dataContext.isSmallDataset && workflow.estimated_time > 180) {
        score -= 1; // 작은 데이터셋에는 긴 분석 비추천
      }
      
      if (score > 0) {
        matches.push({
          name,
          workflow,
          score
        });
      }
    }
    
    return matches;
  }

  generateRecommendationReason(workflow, userQuery, dataContext) {
    const reasons = [];
    
    // 키워드 매칭 이유
    const keywords = this.extractKeywords(userQuery);
    const matchedKeywords = [];
    
    for (const step of workflow.steps) {
      const stepKeywords = this.getStepKeywords(step);
      for (const keyword of keywords) {
        if (stepKeywords.includes(keyword)) {
          matchedKeywords.push(keyword);
        }
      }
    }
    
    if (matchedKeywords.length > 0) {
      reasons.push(`키워드 매칭: ${matchedKeywords.join(', ')}`);
    }
    
    // 데이터 컨텍스트 이유
    if (dataContext.hasTimeSeries && workflow.category === 'time_series') {
      reasons.push('시계열 데이터에 적합');
    }
    
    if (dataContext.isQuickAnalysis && workflow.name.includes('quick')) {
      reasons.push('빠른 분석에 적합');
    }
    
    // 기본 이유
    if (reasons.length === 0) {
      reasons.push(`${workflow.category} 카테고리의 일반적인 워크플로우`);
    }
    
    return reasons.join(', ');
  }

  // 워크플로우 통계
  getWorkflowStatistics() {
    const stats = {
      total_workflows: Object.keys(this.workflowTemplates).length,
      categories: {},
      avg_steps: 0,
      avg_execution_time: 0,
      step_type_distribution: {},
      resource_requirements: {
        light: 0,
        medium: 0,
        heavy: 0,
        gpu_required: 0
      }
    };

    let totalSteps = 0;
    let totalTime = 0;

    for (const [name, workflow] of Object.entries(this.workflowTemplates)) {
      // 카테고리 통계
      stats.categories[workflow.category] = (stats.categories[workflow.category] || 0) + 1;
      
      // 단계 수 및 시간 통계
      totalSteps += workflow.steps.length;
      totalTime += workflow.estimated_time || this.estimateExecutionTime(workflow);
      
      // 단계 타입 분포
      workflow.steps.forEach(step => {
        stats.step_type_distribution[step.type] = (stats.step_type_distribution[step.type] || 0) + 1;
      });
      
      // 리소스 요구사항 분포
      const resourceReq = this.calculateResourceRequirements(workflow);
      if (resourceReq.gpu_required) {
        stats.resource_requirements.gpu_required++;
      } else if (resourceReq.memory_mb > 1000) {
        stats.resource_requirements.heavy++;
      } else if (resourceReq.memory_mb > 500) {
        stats.resource_requirements.medium++;
      } else {
        stats.resource_requirements.light++;
      }
    }

    stats.avg_steps = Math.round(totalSteps / stats.total_workflows);
    stats.avg_execution_time = Math.round(totalTime / stats.total_workflows);

    return stats;
  }

  // 워크플로우 복제 및 변형
  cloneWorkflow(workflowName, newName, modifications = {}) {
    const originalWorkflow = this.getWorkflow(workflowName);
    if (!originalWorkflow) {
      throw new Error(`워크플로우 '${workflowName}'을 찾을 수 없습니다.`);
    }

    const clonedWorkflow = JSON.parse(JSON.stringify(originalWorkflow));
    clonedWorkflow.name = newName;
    clonedWorkflow.description = modifications.description ||
      `${originalWorkflow.description} (복제됨)`;

    // 수정사항 적용
    if (modifications.additionalSteps) {
      clonedWorkflow.steps.push(...modifications.additionalSteps);
    }

    if (modifications.removeSteps) {
      clonedWorkflow.steps = clonedWorkflow.steps.filter(step =>
        !modifications.removeSteps.includes(`${step.type}.${step.method}`)
      );
    }

    if (modifications.parameterChanges) {
      clonedWorkflow.steps.forEach(step => {
        const stepKey = `${step.type}.${step.method}`;
        if (modifications.parameterChanges[stepKey]) {
          step.params = {
            ...step.params,
            ...modifications.parameterChanges[stepKey]
          };
        }
      });
    }

    // 시간 및 리소스 요구사항 재계산
    clonedWorkflow.estimated_time = this.estimateExecutionTime(clonedWorkflow);
    clonedWorkflow.resource_requirements = this.calculateResourceRequirements(clonedWorkflow);

    return clonedWorkflow;
  }

  // 워크플로우 템플릿 추가
  addWorkflowTemplate(workflow) {
    const validation = this.validateWorkflow(workflow);
    if (!validation.valid) {
      throw new Error(`워크플로우 검증 실패: ${validation.errors.join(', ')}`);
    }

    const normalizedWorkflow = this.normalizeWorkflow(workflow);
    this.workflowTemplates[normalizedWorkflow.name] = normalizedWorkflow;
    
    this.logger.info(`워크플로우 템플릿 추가됨: ${normalizedWorkflow.name}`);
    return normalizedWorkflow;
  }

  // 워크플로우 템플릿 제거
  removeWorkflowTemplate(workflowName) {
    if (!this.workflowTemplates[workflowName]) {
      throw new Error(`워크플로우 '${workflowName}'을 찾을 수 없습니다.`);
    }

    delete this.workflowTemplates[workflowName];
    this.logger.info(`워크플로우 템플릿 제거됨: ${workflowName}`);
  }
}
