// workflows/common.js
import { Logger } from '../utils/logger.js';

export class CommonWorkflows {
  constructor() {
    this.logger = new Logger();
    this.workflowTemplates = this.initializeCommonWorkflows();
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
            outputs: ['dataset', 'metadata']
          },
          {
            type: 'basic',
            method: 'descriptive_stats',
            params: {},
            outputs: ['statistics', 'summary']
          },
          {
            type: 'basic',
            method: 'missing_values_analysis',
            params: {},
            outputs: ['missing_data_report']
          },
          {
            type: 'basic',
            method: 'data_types_analysis',
            params: {},
            outputs: ['data_types_report']
          },
          {
            type: 'visualization',
            method: 'distribution_plots',
            params: {
              plot_type: 'histogram',
              columns: 'numeric'
            },
            outputs: ['distribution_plots']
          },
          {
            type: 'visualization',
            method: 'correlation_heatmap',
            params: {
              method: 'pearson'
            },
            outputs: ['correlation_heatmap']
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
            outputs: ['correlation_matrix']
          },
          {
            type: 'basic',
            method: 'correlation',
            params: {
              method: 'spearman'
            },
            outputs: ['spearman_correlation']
          },
          {
            type: 'visualization',
            method: 'correlation_heatmap',
            params: {
              annotation: true,
              colormap: 'coolwarm'
            },
            outputs: ['correlation_heatmap']
          },
          {
            type: 'visualization',
            method: 'scatter_matrix',
            params: {
              alpha: 0.6
            },
            outputs: ['scatter_matrix']
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
            type: 'basic',
            method: 'outlier_detection',
            params: {
              method: 'iqr',
              threshold: 1.5
            },
            outputs: ['outlier_indices_iqr']
          },
          {
            type: 'basic',
            method: 'outlier_detection',
            params: {
              method: 'zscore',
              threshold: 3
            },
            outputs: ['outlier_indices_zscore']
          },
          {
            type: 'basic',
            method: 'outlier_detection',
            params: {
              method: 'isolation_forest',
              contamination: 0.1
            },
            outputs: ['outlier_indices_isolation']
          },
          {
            type: 'visualization',
            method: 'box_plots',
            params: {
              columns: 'numeric'
            },
            outputs: ['box_plots']
          },
          {
            type: 'visualization',
            method: 'outlier_scatter',
            params: {
              method: 'all'
            },
            outputs: ['outlier_scatter_plots']
          }
        ],
        estimated_time: 90,
        resource_requirements: {
          memory_mb: 400,
          cpu_cores: 1,
          gpu_required: false
        }
      },

      feature_engineering: {
        name: 'feature_engineering',
        description: '특성 공학 및 변환',
        category: 'common',
        steps: [
          {
            type: 'advanced',
            method: 'feature_scaling',
            params: {
              method: 'standard',
              columns: 'numeric'
            },
            outputs: ['scaled_features']
          },
          {
            type: 'advanced',
            method: 'feature_encoding',
            params: {
              method: 'one_hot',
              columns: 'categorical'
            },
            outputs: ['encoded_features']
          },
          {
            type: 'advanced',
            method: 'feature_selection',
            params: {
              method: 'variance_threshold',
              threshold: 0.01
            },
            outputs: ['selected_features']
          },
          {
            type: 'advanced',
            method: 'feature_creation',
            params: {
              method: 'polynomial',
              degree: 2
            },
            outputs: ['new_features']
          },
          {
            type: 'visualization',
            method: 'feature_importance',
            params: {
              method: 'mutual_info'
            },
            outputs: ['feature_importance_plot']
          }
        ],
        estimated_time: 150,
        resource_requirements: {
          memory_mb: 800,
          cpu_cores: 2,
          gpu_required: false
        }
      },

      time_series_analysis: {
        name: 'time_series_analysis',
        description: '시계열 데이터 분석',
        category: 'common',
        steps: [
          {
            type: 'timeseries',
            method: 'trend_analysis',
            params: {
              method: 'linear'
            },
            outputs: ['trend_components']
          },
          {
            type: 'timeseries',
            method: 'seasonality_analysis',
            params: {
              method: 'decompose'
            },
            outputs: ['seasonal_components']
          },
          {
            type: 'timeseries',
            method: 'stationarity_test',
            params: {
              method: 'adf'
            },
            outputs: ['stationarity_results']
          },
          {
            type: 'timeseries',
            method: 'autocorrelation',
            params: {
              lags: 40
            },
            outputs: ['autocorrelation_plot']
          },
          {
            type: 'visualization',
            method: 'time_series_plot',
            params: {
              show_trend: true,
              show_seasonal: true
            },
            outputs: ['time_series_plots']
          }
        ],
        estimated_time: 120,
        resource_requirements: {
          memory_mb: 600,
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
            params: {
              show_patterns: true
            },
            outputs: ['missing_data_report']
          },
          {
            type: 'basic',
            method: 'duplicate_detection',
            params: {
              subset: null
            },
            outputs: ['duplicate_report']
          },
          {
            type: 'basic',
            method: 'data_consistency_check',
            params: {
              rules: 'default'
            },
            outputs: ['consistency_report']
          },
          {
            type: 'basic',
            method: 'data_completeness_check',
            params: {
              threshold: 0.8
            },
            outputs: ['completeness_report']
          },
          {
            type: 'visualization',
            method: 'data_quality_dashboard',
            params: {
              include_recommendations: true
            },
            outputs: ['quality_dashboard']
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

  customizeWorkflow(workflowName, customizations) {
    const baseWorkflow = this.getWorkflow(workflowName);
    if (!baseWorkflow) {
      throw new Error(`워크플로우 '${workflowName}'을 찾을 수 없습니다.`);
    }

    const customizedWorkflow = JSON.parse(JSON.stringify(baseWorkflow));
    
    // 파라미터 커스터마이징
    if (customizations.parameters) {
      customizedWorkflow.steps.forEach(step => {
        if (customizations.parameters[step.method]) {
          step.params = {
            ...step.params,
            ...customizations.parameters[step.method]
          };
        }
      });
    }

    // 단계 추가/제거
    if (customizations.additionalSteps) {
      customizedWorkflow.steps.push(...customizations.additionalSteps);
    }

    if (customizations.removeSteps) {
      customizedWorkflow.steps = customizedWorkflow.steps.filter(
        step => !customizations.removeSteps.includes(step.method)
      );
    }

    // 워크플로우 메타데이터 업데이트
    if (customizations.name) {
      customizedWorkflow.name = customizations.name;
    }
    if (customizations.description) {
      customizedWorkflow.description = customizations.description;
    }

    return customizedWorkflow;
  }

  validateWorkflow(workflow) {
    const validationResult = {
      valid: true,
      errors: [],
      warnings: []
    };

    // 기본 구조 검증
    if (!workflow.steps || !Array.isArray(workflow.steps) || workflow.steps.length === 0) {
      validationResult.valid = false;
      validationResult.errors.push('워크플로우에 유효한 단계가 없습니다.');
      return validationResult;
    }

    // 각 단계 검증
    workflow.steps.forEach((step, index) => {
      if (!step.type || !step.method) {
        validationResult.valid = false;
        validationResult.errors.push(`단계 ${index + 1}: type과 method가 필요합니다.`);
      }

      if (!step.params) {
        validationResult.warnings.push(`단계 ${index + 1}: 파라미터가 정의되지 않았습니다.`);
      }

      if (!step.outputs || step.outputs.length === 0) {
        validationResult.warnings.push(`단계 ${index + 1}: 출력이 정의되지 않았습니다.`);
      }
    });

    return validationResult;
  }

  estimateExecutionTime(workflow) {
    if (workflow.estimated_time) {
      return workflow.estimated_time;
    }

    // 단계별 예상 시간 계산
    const stepTimes = {
      'data_loading': 30,
      'basic': 20,
      'advanced': 40,
      'visualization': 25,
      'timeseries': 35
    };

    let totalTime = 0;
    workflow.steps.forEach(step => {
      totalTime += stepTimes[step.type] || 30;
    });

    return totalTime;
  }

  generateExecutionPlan(workflow) {
    const plan = {
      total_steps: workflow.steps.length,
      estimated_time: this.estimateExecutionTime(workflow),
      execution_order: [],
      dependencies: {},
      parallel_groups: []
    };

    // 실행 순서 생성
    workflow.steps.forEach((step, index) => {
      plan.execution_order.push({
        step_index: index,
        step_name: `${step.type}.${step.method}`,
        params: step.params,
        outputs: step.outputs || [],
        estimated_time: this.estimateStepTime(step)
      });
    });

    // 종속성 분석
    this.analyzeDependencies(workflow, plan);

    return plan;
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

    return baseTimes[step.type] || 30;
  }

  analyzeDependencies(workflow, plan) {
    const outputs = new Set();
    
    workflow.steps.forEach((step, index) => {
      const dependencies = [];
      
      // 입력 요구사항 확인
      if (step.requires) {
        step.requires.forEach(requirement => {
          if (!outputs.has(requirement)) {
            dependencies.push(requirement);
          }
        });
      }
      
      plan.dependencies[index] = dependencies;
      
      // 출력 추가
      if (step.outputs) {
        step.outputs.forEach(output => outputs.add(output));
      }
    });
  }

  exportWorkflow(workflowName, format = 'json') {
    const workflow = this.getWorkflow(workflowName);
    if (!workflow) {
      throw new Error(`워크플로우 '${workflowName}'을 찾을 수 없습니다.`);
    }

    switch (format) {
      case 'json':
        return JSON.stringify(workflow, null, 2);
      case 'yaml':
        // YAML 변환 로직 (필요시 구현)
        return workflow;
      default:
        return workflow;
    }
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
          // YAML 파싱 로직 (필요시 구현)
          workflow = workflowData;
          break;
        default:
          workflow = workflowData;
      }

      const validation = this.validateWorkflow(workflow);
      if (!validation.valid) {
        throw new Error(`워크플로우 검증 실패: ${validation.errors.join(', ')}`);
      }

      return workflow;
    } catch (error) {
      this.logger.error('워크플로우 임포트 실패:', error);
      throw error;
    }
  }
}
