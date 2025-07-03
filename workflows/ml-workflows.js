// workflows/ml-workflows.js
import { Logger } from '../utils/logger.js';

export class MLWorkflows {
  constructor() {
    this.logger = new Logger();
    this.mlWorkflowTemplates = this.initializeMLWorkflows();
  }

  initializeMLWorkflows() {
    return {
      supervised_classification: {
        name: 'supervised_classification',
        description: '지도학습 분류 모델 훈련 및 평가',
        category: 'ml_supervised',
        steps: [
          {
            type: 'data_preprocessing',
            method: 'split_data',
            params: {
              test_size: 0.2,
              random_state: 42,
              stratify: true
            },
            outputs: ['X_train', 'X_test', 'y_train', 'y_test']
          },
          {
            type: 'advanced',
            method: 'feature_scaling',
            params: {
              method: 'standard',
              fit_on_train: true
            },
            outputs: ['X_train_scaled', 'X_test_scaled', 'scaler']
          },
          {
            type: 'ml_traditional',
            method: 'supervised.classification.random_forest',
            params: {
              n_estimators: 100,
              max_depth: null,
              random_state: 42
            },
            outputs: ['rf_model', 'rf_predictions', 'rf_probabilities']
          },
          {
            type: 'ml_traditional',
            method: 'supervised.classification.svm',
            params: {
              kernel: 'rbf',
              C: 1.0,
              random_state: 42
            },
            outputs: ['svm_model', 'svm_predictions']
          },
          {
            type: 'ml_traditional',
            method: 'supervised.classification.logistic_regression',
            params: {
              random_state: 42,
              max_iter: 1000
            },
            outputs: ['lr_model', 'lr_predictions', 'lr_probabilities']
          },
          {
            type: 'ml_evaluation',
            method: 'regression_metrics',
            params: {},
            outputs: ['regression_report', 'metrics_summary']
          },
          {
            type: 'visualization',
            method: 'regression_plots',
            params: {
              include_residuals: true
            },
            outputs: ['regression_plots']
          },
          {
            type: 'visualization',
            method: 'model_comparison',
            params: {
              metrics: ['r2_score', 'mse', 'mae']
            },
            outputs: ['model_comparison_chart']
          }
        ],
        estimated_time: 240,
        resource_requirements: {
          memory_mb: 800,
          cpu_cores: 2,
          gpu_required: false
        }
      },

      clustering_analysis: {
        name: 'clustering_analysis',
        description: '비지도학습 클러스터링 분석',
        category: 'ml_unsupervised',
        steps: [
          {
            type: 'advanced',
            method: 'feature_scaling',
            params: {
              method: 'standard'
            },
            outputs: ['scaled_data', 'scaler']
          },
          {
            type: 'ml_traditional',
            method: 'unsupervised.clustering.kmeans',
            params: {
              n_clusters: 'auto',
              random_state: 42
            },
            outputs: ['kmeans_model', 'kmeans_labels', 'kmeans_centers']
          },
          {
            type: 'ml_traditional',
            method: 'unsupervised.clustering.hierarchical',
            params: {
              n_clusters: 'auto',
              linkage: 'ward'
            },
            outputs: ['hierarchical_labels', 'dendrogram']
          },
          {
            type: 'ml_traditional',
            method: 'unsupervised.clustering.dbscan',
            params: {
              eps: 'auto',
              min_samples: 5
            },
            outputs: ['dbscan_labels']
          },
          {
            type: 'ml_evaluation',
            method: 'clustering_metrics',
            params: {
              metrics: ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
            },
            outputs: ['clustering_metrics']
          },
          {
            type: 'visualization',
            method: 'cluster_visualization',
            params: {
              method: 'pca',
              n_components: 2
            },
            outputs: ['cluster_plots']
          },
          {
            type: 'visualization',
            method: 'dendrogram_plot',
            params: {
              truncate_mode: 'level',
              p: 3
            },
            outputs: ['dendrogram_plot']
          }
        ],
        estimated_time: 180,
        resource_requirements: {
          memory_mb: 600,
          cpu_cores: 2,
          gpu_required: false
        }
      },

      dimensionality_reduction: {
        name: 'dimensionality_reduction',
        description: '차원 축소 및 시각화',
        category: 'ml_unsupervised',
        steps: [
          {
            type: 'advanced',
            method: 'feature_scaling',
            params: {
              method: 'standard'
            },
            outputs: ['scaled_data', 'scaler']
          },
          {
            type: 'advanced',
            method: 'pca',
            params: {
              n_components: 0.95,
              whiten: false
            },
            outputs: ['pca_model', 'pca_transformed', 'explained_variance']
          },
          {
            type: 'advanced',
            method: 'tsne',
            params: {
              n_components: 2,
              perplexity: 30,
              random_state: 42
            },
            outputs: ['tsne_transformed']
          },
          {
            type: 'advanced',
            method: 'umap',
            params: {
              n_components: 2,
              n_neighbors: 15,
              random_state: 42
            },
            outputs: ['umap_transformed']
          },
          {
            type: 'visualization',
            method: 'pca_analysis',
            params: {
              show_loadings: true,
              show_explained_variance: true
            },
            outputs: ['pca_plots']
          },
          {
            type: 'visualization',
            method: 'dimensionality_comparison',
            params: {
              methods: ['pca', 'tsne', 'umap']
            },
            outputs: ['dimension_reduction_comparison']
          }
        ],
        estimated_time: 150,
        resource_requirements: {
          memory_mb: 800,
          cpu_cores: 2,
          gpu_required: false
        }
      },

      hyperparameter_tuning: {
        name: 'hyperparameter_tuning',
        description: '하이퍼파라미터 최적화',
        category: 'ml_optimization',
        steps: [
          {
            type: 'data_preprocessing',
            method: 'split_data',
            params: {
              test_size: 0.2,
              random_state: 42
            },
            outputs: ['X_train', 'X_test', 'y_train', 'y_test']
          },
          {
            type: 'ml_optimization',
            method: 'grid_search',
            params: {
              model: 'random_forest',
              param_grid: {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
              },
              cv: 5,
              scoring: 'accuracy'
            },
            outputs: ['grid_search_results', 'best_params_grid']
          },
          {
            type: 'ml_optimization',
            method: 'random_search',
            params: {
              model: 'random_forest',
              param_distributions: {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30, 40],
                'min_samples_split': [2, 5, 10, 15]
              },
              n_iter: 20,
              cv: 5,
              scoring: 'accuracy',
              random_state: 42
            },
            outputs: ['random_search_results', 'best_params_random']
          },
          {
            type: 'ml_optimization',
            method: 'bayesian_optimization',
            params: {
              model: 'random_forest',
              n_calls: 30,
              random_state: 42
            },
            outputs: ['bayesian_results', 'best_params_bayesian']
          },
          {
            type: 'ml_evaluation',
            method: 'model_evaluation',
            params: {
              use_best_params: true
            },
            outputs: ['final_model', 'final_predictions', 'final_metrics']
          },
          {
            type: 'visualization',
            method: 'hyperparameter_analysis',
            params: {
              show_convergence: true
            },
            outputs: ['hyperparameter_plots']
          }
        ],
        estimated_time: 600,
        resource_requirements: {
          memory_mb: 1200,
          cpu_cores: 4,
          gpu_required: false
        }
      },

      ensemble_methods: {
        name: 'ensemble_methods',
        description: '앙상블 방법을 사용한 모델 결합',
        category: 'ml_ensemble',
        steps: [
          {
            type: 'data_preprocessing',
            method: 'split_data',
            params: {
              test_size: 0.2,
              random_state: 42
            },
            outputs: ['X_train', 'X_test', 'y_train', 'y_test']
          },
          {
            type: 'ml_ensemble',
            method: 'voting_classifier',
            params: {
              estimators: [
                'random_forest',
                'svm',
                'logistic_regression'
              ],
              voting: 'soft'
            },
            outputs: ['voting_model', 'voting_predictions']
          },
          {
            type: 'ml_ensemble',
            method: 'bagging_classifier',
            params: {
              base_estimator: 'decision_tree',
              n_estimators: 100,
              random_state: 42
            },
            outputs: ['bagging_model', 'bagging_predictions']
          },
          {
            type: 'ml_ensemble',
            method: 'adaboost_classifier',
            params: {
              n_estimators: 100,
              learning_rate: 1.0,
              random_state: 42
            },
            outputs: ['adaboost_model', 'adaboost_predictions']
          },
          {
            type: 'ml_ensemble',
            method: 'gradient_boosting_classifier',
            params: {
              n_estimators: 100,
              learning_rate: 0.1,
              random_state: 42
            },
            outputs: ['gb_model', 'gb_predictions']
          },
          {
            type: 'ml_evaluation',
            method: 'ensemble_comparison',
            params: {
              metrics: ['accuracy', 'precision', 'recall', 'f1_score']
            },
            outputs: ['ensemble_metrics']
          },
          {
            type: 'visualization',
            method: 'ensemble_performance',
            params: {
              show_individual_models: true
            },
            outputs: ['ensemble_performance_plots']
          }
        ],
        estimated_time: 420,
        resource_requirements: {
          memory_mb: 1500,
          cpu_cores: 3,
          gpu_required: false
        }
      },

      feature_selection: {
        name: 'feature_selection',
        description: '특성 선택 및 중요도 분석',
        category: 'ml_feature_engineering',
        steps: [
          {
            type: 'advanced',
            method: 'feature_selection',
            params: {
              method: 'variance_threshold',
              threshold: 0.01
            },
            outputs: ['variance_selected_features']
          },
          {
            type: 'advanced',
            method: 'feature_selection',
            params: {
              method: 'univariate_selection',
              k: 10
            },
            outputs: ['univariate_selected_features']
          },
          {
            type: 'advanced',
            method: 'feature_selection',
            params: {
              method: 'recursive_feature_elimination',
              estimator: 'random_forest',
              n_features_to_select: 10
            },
            outputs: ['rfe_selected_features']
          },
          {
            type: 'advanced',
            method: 'feature_importance',
            params: {
              method: 'tree_based',
              estimator: 'random_forest'
            },
            outputs: ['tree_feature_importance']
          },
          {
            type: 'advanced',
            method: 'feature_importance',
            params: {
              method: 'permutation',
              estimator: 'random_forest'
            },
            outputs: ['permutation_feature_importance']
          },
          {
            type: 'visualization',
            method: 'feature_importance_plots',
            params: {
              top_n: 20
            },
            outputs: ['feature_importance_plots']
          },
          {
            type: 'visualization',
            method: 'feature_selection_comparison',
            params: {
              methods: ['variance', 'univariate', 'rfe']
            },
            outputs: ['feature_selection_comparison']
          }
        ],
        estimated_time: 200,
        resource_requirements: {
          memory_mb: 800,
          cpu_cores: 2,
          gpu_required: false
        }
      },

      model_validation: {
        name: 'model_validation',
        description: '모델 검증 및 교차 검증',
        category: 'ml_validation',
        steps: [
          {
            type: 'ml_validation',
            method: 'cross_validation',
            params: {
              cv: 5,
              scoring: ['accuracy', 'precision', 'recall', 'f1_score']
            },
            outputs: ['cv_scores', 'cv_metrics']
          },
          {
            type: 'ml_validation',
            method: 'stratified_k_fold',
            params: {
              n_splits: 5,
              random_state: 42
            },
            outputs: ['stratified_cv_scores']
          },
          {
            type: 'ml_validation',
            method: 'time_series_split',
            params: {
              n_splits: 5
            },
            outputs: ['time_series_cv_scores']
          },
          {
            type: 'ml_validation',
            method: 'bootstrap_validation',
            params: {
              n_bootstrap: 100,
              random_state: 42
            },
            outputs: ['bootstrap_scores']
          },
          {
            type: 'ml_evaluation',
            method: 'learning_curve',
            params: {
              train_sizes: [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
              cv: 5
            },
            outputs: ['learning_curve_data']
          },
          {
            type: 'ml_evaluation',
            method: 'validation_curve',
            params: {
              param_name: 'n_estimators',
              param_range: [10, 50, 100, 200, 500],
              cv: 5
            },
            outputs: ['validation_curve_data']
          },
          {
            type: 'visualization',
            method: 'validation_plots',
            params: {
              include_learning_curve: true,
              include_validation_curve: true
            },
            outputs: ['validation_plots']
          }
        ],
        estimated_time: 300,
        resource_requirements: {
          memory_mb: 1000,
          cpu_cores: 2,
          gpu_required: false
        }
      }
    };
  }

  getWorkflow(workflowName) {
    return this.mlWorkflowTemplates[workflowName] || null;
  }

  getAllWorkflows() {
    return this.mlWorkflowTemplates;
  }

  getWorkflowsByCategory(category) {
    const workflows = {};
    for (const [name, workflow] of Object.entries(this.mlWorkflowTemplates)) {
      if (workflow.category === category) {
        workflows[name] = workflow;
      }
    }
    return workflows;
  }

  getAvailableCategories() {
    const categories = new Set();
    for (const workflow of Object.values(this.mlWorkflowTemplates)) {
      categories.add(workflow.category);
    }
    return Array.from(categories);
  }

  customizeMLWorkflow(workflowName, customizations) {
    const baseWorkflow = this.getWorkflow(workflowName);
    if (!baseWorkflow) {
      throw new Error(`ML 워크플로우 '${workflowName}'을 찾을 수 없습니다.`);
    }

    const customizedWorkflow = JSON.parse(JSON.stringify(baseWorkflow));
    
    // 모델 파라미터 커스터마이징
    if (customizations.modelParams) {
      customizedWorkflow.steps.forEach(step => {
        if (step.type.startsWith('ml_') && customizations.modelParams[step.method]) {
          step.params = {
            ...step.params,
            ...customizations.modelParams[step.method]
          };
        }
      });
    }

    // 평가 메트릭 커스터마이징
    if (customizations.evaluationMetrics) {
      customizedWorkflow.steps.forEach(step => {
        if (step.type === 'ml_evaluation' && step.params.metrics) {
          step.params.metrics = customizations.evaluationMetrics;
        }
      });
    }

    // 교차 검증 설정
    if (customizations.crossValidation) {
      customizedWorkflow.steps.forEach(step => {
        if (step.params.cv) {
          step.params.cv = customizations.crossValidation.folds || step.params.cv;
        }
        if (step.params.scoring && customizations.crossValidation.scoring) {
          step.params.scoring = customizations.crossValidation.scoring;
        }
      });
    }

    return customizedWorkflow;
  }

  generateMLPipeline(problemType, dataInfo, requirements = {}) {
    const pipelines = {
      'classification': this.generateClassificationPipeline,
      'regression': this.generateRegressionPipeline,
      'clustering': this.generateClusteringPipeline,
      'dimensionality_reduction': this.generateDimensionalityReductionPipeline
    };

    const generator = pipelines[problemType];
    if (!generator) {
      throw new Error(`지원하지 않는 문제 유형: ${problemType}`);
    }

    return generator.call(this, dataInfo, requirements);
  }

  generateClassificationPipeline(dataInfo, requirements) {
    const pipeline = {
      name: 'custom_classification_pipeline',
      description: '커스텀 분류 파이프라인',
      category: 'ml_custom',
      steps: []
    };

    // 데이터 전처리
    pipeline.steps.push({
      type: 'data_preprocessing',
      method: 'split_data',
      params: {
        test_size: requirements.testSize || 0.2,
        random_state: requirements.randomState || 42,
        stratify: dataInfo.isImbalanced ? true : false
      },
      outputs: ['X_train', 'X_test', 'y_train', 'y_test']
    });

    // 특성 스케일링
    if (dataInfo.hasNumericFeatures) {
      pipeline.steps.push({
        type: 'advanced',
        method: 'feature_scaling',
        params: {
          method: requirements.scalingMethod || 'standard'
        },
        outputs: ['X_train_scaled', 'X_test_scaled', 'scaler']
      });
    }

    // 모델 선택
    const models = requirements.models || ['random_forest', 'svm', 'logistic_regression'];
    models.forEach(model => {
      pipeline.steps.push({
        type: 'ml_traditional',
        method: `supervised.classification.${model}`,
        params: this.getDefaultModelParams(model),
        outputs: [`${model}_model`, `${model}_predictions`]
      });
    });

    // 평가
    pipeline.steps.push({
      type: 'ml_evaluation',
      method: 'classification_metrics',
      params: {
        average: 'weighted'
      },
      outputs: ['classification_report', 'confusion_matrix']
    });

    return pipeline;
  }

  generateRegressionPipeline(dataInfo, requirements) {
    const pipeline = {
      name: 'custom_regression_pipeline',
      description: '커스텀 회귀 파이프라인',
      category: 'ml_custom',
      steps: []
    };

    // 데이터 전처리
    pipeline.steps.push({
      type: 'data_preprocessing',
      method: 'split_data',
      params: {
        test_size: requirements.testSize || 0.2,
        random_state: requirements.randomState || 42
      },
      outputs: ['X_train', 'X_test', 'y_train', 'y_test']
    });

    // 모델 및 평가 단계 추가
    const models = requirements.models || ['linear_regression', 'random_forest', 'gradient_boosting'];
    models.forEach(model => {
      pipeline.steps.push({
        type: 'ml_traditional',
        method: `supervised.regression.${model}`,
        params: this.getDefaultModelParams(model),
        outputs: [`${model}_model`, `${model}_predictions`]
      });
    });

    pipeline.steps.push({
      type: 'ml_evaluation',
      method: 'regression_metrics',
      params: {},
      outputs: ['regression_report']
    });

    return pipeline;
  }

  generateClusteringPipeline(dataInfo, requirements) {
    const pipeline = {
      name: 'custom_clustering_pipeline',
      description: '커스텀 클러스터링 파이프라인',
      category: 'ml_custom',
      steps: []
    };

    // 특성 스케일링
    pipeline.steps.push({
      type: 'advanced',
      method: 'feature_scaling',
      params: {
        method: 'standard'
      },
      outputs: ['scaled_data', 'scaler']
    });

    // 클러스터링 방법
    const methods = requirements.methods || ['kmeans', 'hierarchical', 'dbscan'];
    methods.forEach(method => {
      pipeline.steps.push({
        type: 'ml_traditional',
        method: `unsupervised.clustering.${method}`,
        params: this.getDefaultModelParams(method),
        outputs: [`${method}_labels`]
      });
    });

    // 평가
    pipeline.steps.push({
      type: 'ml_evaluation',
      method: 'clustering_metrics',
      params: {},
      outputs: ['clustering_metrics']
    });

    return pipeline;
  }

  generateDimensionalityReductionPipeline(dataInfo, requirements) {
    const pipeline = {
      name: 'custom_dimensionality_reduction_pipeline',
      description: '커스텀 차원 축소 파이프라인',
      category: 'ml_custom',
      steps: []
    };

    // 특성 스케일링
    pipeline.steps.push({
      type: 'advanced',
      method: 'feature_scaling',
      params: {
        method: 'standard'
      },
      outputs: ['scaled_data', 'scaler']
    });

    // 차원 축소 방법
    const methods = requirements.methods || ['pca', 'tsne'];
    methods.forEach(method => {
      pipeline.steps.push({
        type: 'advanced',
        method: method,
        params: this.getDefaultModelParams(method),
        outputs: [`${method}_transformed`]
      });
    });

    // 시각화
    pipeline.steps.push({
      type: 'visualization',
      method: 'dimensionality_comparison',
      params: {
        methods: methods
      },
      outputs: ['dimension_reduction_plots']
    });

    return pipeline;
  }

  getDefaultModelParams(modelType) {
    const defaultParams = {
      // 분류 모델
      'random_forest': {
        n_estimators: 100,
        random_state: 42
      },
      'svm': {
        kernel: 'rbf',
        C: 1.0,
        random_state: 42
      },
      'logistic_regression': {
        random_state: 42,
        max_iter: 1000
      },
      
      // 회귀 모델
      'linear_regression': {},
      'gradient_boosting': {
        n_estimators: 100,
        learning_rate: 0.1,
        random_state: 42
      },
      
      // 클러스터링
      'kmeans': {
        n_clusters: 3,
        random_state: 42
      },
      'hierarchical': {
        n_clusters: 3,
        linkage: 'ward'
      },
      'dbscan': {
        eps: 0.5,
        min_samples: 5
      },
      
      // 차원 축소
      'pca': {
        n_components: 2
      },
      'tsne': {
        n_components: 2,
        random_state: 42
      }
    };

    return defaultParams[modelType] || {};
  }

  validateMLWorkflow(workflow) {
    const validationResult = {
      valid: true,
      errors: [],
      warnings: []
    };

    // ML 워크플로우 특화 검증
    const mlSteps = workflow.steps.filter(step => step.type.startsWith('ml_'));
    if (mlSteps.length === 0) {
      validationResult.warnings.push('ML 관련 단계가 없습니다.');
    }

    // 평가 단계 확인
    const evaluationSteps = workflow.steps.filter(step => step.type === 'ml_evaluation');
    if (evaluationSteps.length === 0) {
      validationResult.warnings.push('모델 평가 단계가 없습니다.');
    }

    // 데이터 분할 확인
    const splitSteps = workflow.steps.filter(step => step.method === 'split_data');
    if (splitSteps.length === 0) {
      validationResult.warnings.push('데이터 분할 단계가 없습니다.');
    }

    return validationResult;
  }

  exportMLWorkflow(workflowName, format = 'json') {
    const workflow = this.getWorkflow(workflowName);
    if (!workflow) {
      throw new Error(`ML 워크플로우 '${workflowName}'을 찾을 수 없습니다.`);
    }

    switch (format) {
      case 'json':
        return JSON.stringify(workflow, null, 2);
      case 'sklearn_pipeline':
        return this.convertToSklearnPipeline(workflow);
      default:
        return workflow;
    }
  }

  convertToSklearnPipeline(workflow) {
    // sklearn Pipeline 형태로 변환하는 로직
    // 실제 구현에서는 더 복잡한 변환 로직이 필요
    const pipelineSteps = workflow.steps
      .filter(step => step.type.startsWith('ml_') || step.type === 'advanced')
      .map(step => ({
        name: step.method,
        params: step.params
      }));

    return {
      pipeline_steps: pipelineSteps,
      original_workflow: workflow.name
    };
  }
}'classification_metrics',
            params: {
              average: 'weighted'
            },
            outputs: ['classification_report', 'confusion_matrix', 'metrics_summary']
          },
          {
            type: 'visualization',
            method: 'model_comparison',
            params: {
              metrics: ['accuracy', 'precision', 'recall', 'f1_score']
            },
            outputs: ['model_comparison_chart']
          },
          {
            type: 'visualization',
            method: 'confusion_matrix_heatmap',
            params: {
              normalize: true
            },
            outputs: ['confusion_matrix_plots']
          }
        ],
        estimated_time: 300,
        resource_requirements: {
          memory_mb: 1024,
          cpu_cores: 2,
          gpu_required: false
        }
      },

      supervised_regression: {
        name: 'supervised_regression',
        description: '지도학습 회귀 모델 훈련 및 평가',
        category: 'ml_supervised',
        steps: [
          {
            type: 'data_preprocessing',
            method: 'split_data',
            params: {
              test_size: 0.2,
              random_state: 42
            },
            outputs: ['X_train', 'X_test', 'y_train', 'y_test']
          },
          {
            type: 'advanced',
            method: 'feature_scaling',
            params: {
              method: 'standard'
            },
            outputs: ['X_train_scaled', 'X_test_scaled', 'scaler']
          },
          {
            type: 'ml_traditional',
            method: 'supervised.regression.linear_regression',
            params: {},
            outputs: ['linear_model', 'linear_predictions']
          },
          {
            type: 'ml_traditional',
            method: 'supervised.regression.random_forest',
            params: {
              n_estimators: 100,
              random_state: 42
            },
            outputs: ['rf_model', 'rf_predictions']
          },
          {
            type: 'ml_traditional',
            method: 'supervised.regression.gradient_boosting',
            params: {
              n_estimators: 100,
              learning_rate: 0.1,
              random_state: 42
            },
            outputs: ['gb_model', 'gb_predictions']
          },
          {
            type: 'ml_evaluation',
            method:
