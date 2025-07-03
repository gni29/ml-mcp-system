// workflows/viz-workflows.js
import { Logger } from '../utils/logger.js';

export class VizWorkflows {
  constructor() {
    this.logger = new Logger();
    this.visualizationWorkflows = this.initializeVisualizationWorkflows();
  }

  initializeVisualizationWorkflows() {
    return {
      exploratory_data_viz: {
        name: 'exploratory_data_viz',
        description: '탐색적 데이터 시각화 - 데이터 전체적인 패턴 파악',
        category: 'viz_exploratory',
        steps: [
          {
            type: 'visualization',
            method: 'data_overview',
            params: {
              show_info: true,
              show_head: 10,
              show_tail: 5
            },
            outputs: ['data_overview_table']
          },
          {
            type: 'visualization',
            method: 'distribution_plots',
            params: {
              plot_type: 'histogram',
              columns: 'numeric',
              bins: 30,
              kde: true
            },
            outputs: ['distribution_histograms']
          },
          {
            type: 'visualization',
            method: 'box_plots',
            params: {
              columns: 'numeric',
              show_outliers: true,
              orient: 'vertical'
            },
            outputs: ['box_plots']
          },
          {
            type: 'visualization',
            method: 'correlation_heatmap',
            params: {
              method: 'pearson',
              annot: true,
              cmap: 'coolwarm',
              figsize: [12, 10]
            },
            outputs: ['correlation_heatmap']
          },
          {
            type: 'visualization',
            method: 'scatter_matrix',
            params: {
              alpha: 0.6,
              figsize: [15, 15],
              diagonal: 'hist'
            },
            outputs: ['scatter_matrix']
          },
          {
            type: 'visualization',
            method: 'missing_data_matrix',
            params: {
              show_patterns: true,
              color_map: 'viridis'
            },
            outputs: ['missing_data_viz']
          }
        ],
        estimated_time: 180,
        resource_requirements: {
          memory_mb: 800,
          cpu_cores: 1,
          gpu_required: false
        }
      },

      statistical_viz: {
        name: 'statistical_viz',
        description: '통계적 시각화 - 통계 분석 결과 시각화',
        category: 'viz_statistical',
        steps: [
          {
            type: 'visualization',
            method: 'qq_plots',
            params: {
              columns: 'numeric',
              distribution: 'normal'
            },
            outputs: ['qq_plots']
          },
          {
            type: 'visualization',
            method: 'probability_plots',
            params: {
              columns: 'numeric',
              distributions: ['normal', 'lognormal', 'exponential']
            },
            outputs: ['probability_plots']
          },
          {
            type: 'visualization',
            method: 'confidence_intervals',
            params: {
              confidence_level: 0.95,
              method: 'bootstrap'
            },
            outputs: ['confidence_interval_plots']
          },
          {
            type: 'visualization',
            method: 'regression_plots',
            params: {
              show_confidence: true,
              show_prediction: true,
              residual_plots: true
            },
            outputs: ['regression_plots']
          },
          {
            type: 'visualization',
            method: 'hypothesis_test_viz',
            params: {
              test_type: 'auto',
              show_p_value: true,
              show_effect_size: true
            },
            outputs: ['hypothesis_test_plots']
          }
        ],
        estimated_time: 150,
        resource_requirements: {
          memory_mb: 600,
          cpu_cores: 1,
          gpu_required: false
        }
      },

      ml_model_viz: {
        name: 'ml_model_viz',
        description: '머신러닝 모델 시각화 - 모델 성능 및 결과 시각화',
        category: 'viz_ml',
        steps: [
          {
            type: 'visualization',
            method: 'feature_importance',
            params: {
              model_type: 'auto',
              top_n: 20,
              plot_type: 'bar'
            },
            outputs: ['feature_importance_plot']
          },
          {
            type: 'visualization',
            method: 'learning_curve',
            params: {
              cv: 5,
              train_sizes: [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
              scoring: 'accuracy'
            },
            outputs: ['learning_curve_plot']
          },
          {
            type: 'visualization',
            method: 'validation_curve',
            params: {
              param_name: 'auto',
              param_range: 'auto',
              cv: 5,
              scoring: 'accuracy'
            },
            outputs: ['validation_curve_plot']
          },
          {
            type: 'visualization',
            method: 'confusion_matrix',
            params: {
              normalize: true,
              cmap: 'Blues',
              show_values: true
            },
            outputs: ['confusion_matrix_plot']
          },
          {
            type: 'visualization',
            method: 'roc_curve',
            params: {
              multi_class: 'auto',
              show_auc: true
            },
            outputs: ['roc_curve_plot']
          },
          {
            type: 'visualization',
            method: 'precision_recall_curve',
            params: {
              show_ap_score: true
            },
            outputs: ['pr_curve_plot']
          },
          {
            type: 'visualization',
            method: 'model_comparison',
            params: {
              metrics: ['accuracy', 'precision', 'recall', 'f1_score'],
              plot_type: 'bar'
            },
            outputs: ['model_comparison_plot']
          }
        ],
        estimated_time: 200,
        resource_requirements: {
          memory_mb: 800,
          cpu_cores: 1,
          gpu_required: false
        }
      },

      clustering_viz: {
        name: 'clustering_viz',
        description: '클러스터링 시각화 - 클러스터링 결과 시각화',
        category: 'viz_clustering',
        steps: [
          {
            type: 'visualization',
            method: 'cluster_scatter',
            params: {
              method: 'pca',
              n_components: 2,
              show_centroids: true,
              color_palette: 'tab10'
            },
            outputs: ['cluster_scatter_2d']
          },
          {
            type: 'visualization',
            method: 'cluster_scatter_3d',
            params: {
              method: 'pca',
              n_components: 3,
              show_centroids: true,
              interactive: true
            },
            outputs: ['cluster_scatter_3d']
          },
          {
            type: 'visualization',
            method: 'dendrogram',
            params: {
              truncate_mode: 'level',
              p: 30,
              leaf_rotation: 90,
              leaf_font_size: 12
            },
            outputs: ['dendrogram_plot']
          },
          {
            type: 'visualization',
            method: 'cluster_heatmap',
            params: {
              standardize: true,
              cmap: 'RdYlBu_r',
              show_cluster_labels: true
            },
            outputs: ['cluster_heatmap']
          },
          {
            type: 'visualization',
            method: 'silhouette_analysis',
            params: {
              show_avg_score: true,
              show_individual_scores: true
            },
            outputs: ['silhouette_plot']
          },
          {
            type: 'visualization',
            method: 'elbow_method',
            params: {
              k_range: [2, 3, 4, 5, 6, 7, 8, 9, 10],
              metric: 'distortion'
            },
            outputs: ['elbow_plot']
          }
        ],
        estimated_time: 160,
        resource_requirements: {
          memory_mb: 700,
          cpu_cores: 1,
          gpu_required: false
        }
      },

      dimensionality_reduction_viz: {
        name: 'dimensionality_reduction_viz',
        description: '차원 축소 시각화 - PCA, t-SNE, UMAP 등 시각화',
        category: 'viz_dimensionality',
        steps: [
          {
            type: 'visualization',
            method: 'pca_analysis',
            params: {
              n_components: 10,
              show_explained_variance: true,
              show_cumulative_variance: true,
              show_loadings: true
            },
            outputs: ['pca_analysis_plots']
          },
          {
            type: 'visualization',
            method: 'pca_biplot',
            params: {
              n_components: 2,
              show_loadings: true,
              loading_scale: 1.0,
              alpha: 0.6
            },
            outputs: ['pca_biplot']
          },
          {
            type: 'visualization',
            method: 'tsne_plot',
            params: {
              n_components: 2,
              perplexity: 30,
              n_iter: 1000,
              random_state: 42
            },
            outputs: ['tsne_plot']
          },
          {
            type: 'visualization',
            method: 'umap_plot',
            params: {
              n_components: 2,
              n_neighbors: 15,
              min_dist: 0.1,
              random_state: 42
            },
            outputs: ['umap_plot']
          },
          {
            type: 'visualization',
            method: 'dimension_comparison',
            params: {
              methods: ['pca', 'tsne', 'umap'],
              n_components: 2,
              figsize: [15, 5]
            },
            outputs: ['dimension_comparison_plot']
          },
          {
            type: 'visualization',
            method: 'scree_plot',
            params: {
              n_components: 20,
              show_kaiser_rule: true
            },
            outputs: ['scree_plot']
          }
        ],
        estimated_time: 220,
        resource_requirements: {
          memory_mb: 1000,
          cpu_cores: 2,
          gpu_required: false
        }
      },

      time_series_viz: {
        name: 'time_series_viz',
        description: '시계열 데이터 시각화 - 시계열 패턴 및 분석 결과 시각화',
        category: 'viz_timeseries',
        steps: [
          {
            type: 'visualization',
            method: 'time_series_line',
            params: {
              show_trend: true,
              show_seasonal: true,
              figsize: [15, 8]
            },
            outputs: ['time_series_line_plot']
          },
          {
            type: 'visualization',
            method: 'seasonal_decomposition',
            params: {
              model: 'additive',
              period: 'auto',
              show_original: true
            },
            outputs: ['seasonal_decomposition_plot']
          },
          {
            type: 'visualization',
            method: 'autocorrelation_plots',
            params: {
              lags: 40,
              show_acf: true,
              show_pacf: true
            },
            outputs: ['autocorrelation_plots']
          },
          {
            type: 'visualization',
            method: 'rolling_statistics',
            params: {
              window: 12,
              show_mean: true,
              show_std: true,
              show_min_max: true
            },
            outputs: ['rolling_statistics_plot']
          },
          {
            type: 'visualization',
            method: 'lag_plots',
            params: {
              lags: [1, 2, 3, 4, 5, 6],
              figsize: [12, 8]
            },
            outputs: ['lag_plots']
          },
          {
            type: 'visualization',
            method: 'forecast_plot',
            params: {
              forecast_periods: 24,
              show_confidence_interval: true,
              confidence_level: 0.95
            },
            outputs: ['forecast_plot']
          }
        ],
        estimated_time: 180,
        resource_requirements: {
          memory_mb: 600,
          cpu_cores: 1,
          gpu_required: false
        }
      },

      interactive_dashboard: {
        name: 'interactive_dashboard',
        description: '인터랙티브 대시보드 - 동적 시각화 및 대시보드 생성',
        category: 'viz_interactive',
        steps: [
          {
            type: 'visualization',
            method: 'plotly_dashboard',
            params: {
              charts: ['scatter', 'histogram', 'box', 'heatmap'],
              interactive_filters: true,
              export_html: true
            },
            outputs: ['interactive_dashboard_html']
          },
          {
            type: 'visualization',
            method: 'bokeh_dashboard',
            params: {
              charts: ['scatter', 'line', 'bar'],
              widgets: ['slider', 'select', 'checkbox'],
              server_mode: false
            },
            outputs: ['bokeh_dashboard_html']
          },
          {
            type: 'visualization',
            method: 'streamlit_components',
            params: {
              components: ['metrics', 'charts', 'tables'],
              layout: 'sidebar',
              theme: 'light'
            },
            outputs: ['streamlit_components']
          },
          {
            type: 'visualization',
            method: 'data_table',
            params: {
              interactive: true,
              searchable: true,
              sortable: true,
              paginated: true
            },
            outputs: ['interactive_data_table']
          }
        ],
        estimated_time: 240,
        resource_requirements: {
          memory_mb: 1200,
          cpu_cores: 2,
          gpu_required: false
        }
      },

      publication_ready_viz: {
        name: 'publication_ready_viz',
        description: '출판용 고품질 시각화 - 논문/보고서용 시각화',
        category: 'viz_publication',
        steps: [
          {
            type: 'visualization',
            method: 'publication_scatter',
            params: {
              style: 'seaborn-whitegrid',
              dpi: 300,
              format: 'png',
              show_statistics: true,
              font_size: 12
            },
            outputs: ['publication_scatter_plot']
          },
          {
            type: 'visualization',
            method: 'publication_heatmap',
            params: {
              style: 'seaborn-white',
              dpi: 300,
              format: 'png',
              colorbar_label: 'auto',
              font_size: 12
            },
            outputs: ['publication_heatmap']
          },
          {
            type: 'visualization',
            method: 'publication_barplot',
            params: {
              style: 'seaborn-whitegrid',
              dpi: 300,
              format: 'png',
              show_values: true,
              font_size: 12
            },
            outputs: ['publication_barplot']
          },
          {
            type: 'visualization',
            method: 'figure_grid',
            params: {
              grid_size: [2, 2],
              shared_axes: true,
              dpi: 300,
              format: 'png'
            },
            outputs: ['publication_figure_grid']
          },
          {
            type: 'visualization',
            method: 'export_formats',
            params: {
              formats: ['png', 'pdf', 'svg', 'eps'],
              dpi: 300,
              bbox_inches: 'tight'
            },
            outputs: ['exported_figures']
          }
        ],
        estimated_time: 120,
        resource_requirements: {
          memory_mb: 800,
          cpu_cores: 1,
          gpu_required: false
        }
      },

      custom_viz_pipeline: {
        name: 'custom_viz_pipeline',
        description: '커스텀 시각화 파이프라인 - 사용자 정의 시각화 워크플로우',
        category: 'viz_custom',
        steps: [
          {
            type: 'visualization',
            method: 'custom_preprocessing',
            params: {
              operations: ['normalize', 'filter', 'aggregate'],
              custom_functions: []
            },
            outputs: ['preprocessed_data']
          },
          {
            type: 'visualization',
            method: 'custom_plot',
            params: {
              plot_type: 'auto',
              custom_style: {},
              annotations: []
            },
            outputs: ['custom_plot']
          },
          {
            type: 'visualization',
            method: 'layout_optimization',
            params: {
              optimize_for: 'readability',
              color_scheme: 'auto',
              font_optimization: true
            },
            outputs: ['optimized_layout']
          },
          {
            type: 'visualization',
            method: 'export_pipeline',
            params: {
              export_code: true,
              export_data: true,
              export_config: true
            },
            outputs: ['exportable_pipeline']
          }
        ],
        estimated_time: 100,
        resource_requirements: {
          memory_mb: 500,
          cpu_cores: 1,
          gpu_required: false
        }
      }
    };
  }

  getWorkflow(workflowName) {
    return this.visualizationWorkflows[workflowName] || null;
  }

  getAllWorkflows() {
    return this.visualizationWorkflows;
  }

  getWorkflowsByCategory(category) {
    const workflows = {};
    for (const [name, workflow] of Object.entries(this.visualizationWorkflows)) {
      if (workflow.category === category) {
        workflows[name] = workflow;
      }
    }
    return workflows;
  }

  getAvailableCategories() {
    const categories = new Set();
    for (const workflow of Object.values(this.visualizationWorkflows)) {
      categories.add(workflow.category);
    }
    return Array.from(categories);
  }

  customizeVisualizationWorkflow(workflowName, customizations) {
    const baseWorkflow = this.getWorkflow(workflowName);
    if (!baseWorkflow) {
      throw new Error(`시각화 워크플로우 '${workflowName}'을 찾을 수 없습니다.`);
    }

    const customizedWorkflow = JSON.parse(JSON.stringify(baseWorkflow));

    // 시각화 파라미터 커스터마이징
    if (customizations.visualParams) {
      customizedWorkflow.steps.forEach(step => {
        if (step.type === 'visualization' && customizations.visualParams[step.method]) {
          step.params = {
            ...step.params,
            ...customizations.visualParams[step.method]
          };
        }
      });
    }

    // 스타일 커스터마이징
    if (customizations.style) {
      customizedWorkflow.steps.forEach(step => {
        if (step.type === 'visualization') {
          step.params.style = customizations.style.theme || step.params.style;
          step.params.color_palette = customizations.style.color_palette || step.params.color_palette;
          step.params.figsize = customizations.style.figsize || step.params.figsize;
          step.params.dpi = customizations.style.dpi || step.params.dpi;
        }
      });
    }

    // 출력 형식 커스터마이징
    if (customizations.outputFormat) {
      customizedWorkflow.steps.forEach(step => {
        if (step.type === 'visualization') {
          step.params.format = customizations.outputFormat.format || step.params.format;
          step.params.interactive = customizations.outputFormat.interactive || step.params.interactive;
        }
      });
    }

    return customizedWorkflow;
  }

  generateVisualizationPipeline(vizType, dataInfo, requirements = {}) {
    const generators = {
      'exploratory': this.generateExploratoryVizPipeline,
      'statistical': this.generateStatisticalVizPipeline,
      'ml_model': this.generateMLModelVizPipeline,
      'clustering': this.generateClusteringVizPipeline,
      'time_series': this.generateTimeSeriesVizPipeline,
      'interactive': this.generateInteractiveVizPipeline
    };

    const generator = generators[vizType];
    if (!generator) {
      throw new Error(`지원하지 않는 시각화 유형: ${vizType}`);
    }

    return generator.call(this, dataInfo, requirements);
  }

  generateExploratoryVizPipeline(dataInfo, requirements) {
    const pipeline = {
      name: 'custom_exploratory_viz',
      description: '커스텀 탐색적 시각화 파이프라인',
      category: 'viz_custom',
      steps: []
    };

    // 데이터 개요
    pipeline.steps.push({
      type: 'visualization',
      method: 'data_overview',
      params: {
        show_info: true,
        show_head: requirements.preview_rows || 10
      },
      outputs: ['data_overview']
    });

    // 수치형 데이터 시각화
    if (dataInfo.hasNumericColumns) {
      pipeline.steps.push({
        type: 'visualization',
        method: 'distribution_plots',
        params: {
          plot_type: requirements.distrib_plot_type || 'histogram',
          columns: 'numeric',
          bins: requirements.bins || 30
        },
        outputs: ['distribution_plots']
      });

      pipeline.steps.push({
        type: 'visualization',
        method: 'correlation_heatmap',
        params: {
          method: requirements.corr_method || 'pearson',
          annot: true
        },
        outputs: ['correlation_heatmap']
      });
    }

    // 범주형 데이터 시각화
    if (dataInfo.hasCategoricalColumns) {
      pipeline.steps.push({
        type: 'visualization',
        method: 'categorical_plots',
        params: {
          plot_type: requirements.cat_plot_type || 'countplot',
          columns: 'categorical'
        },
        outputs: ['categorical_plots']
      });
    }

    return pipeline;
  }

  generateStatisticalVizPipeline(dataInfo, requirements) {
    const pipeline = {
      name: 'custom_statistical_viz',
      description: '커스텀 통계 시각화 파이프라인',
      category: 'viz_custom',
      steps: []
    };

    // 정규성 검정 시각화
    pipeline.steps.push({
      type: 'visualization',
      method: 'qq_plots',
      params: {
        columns: 'numeric',
        distribution: requirements.distribution || 'normal'
      },
      outputs: ['qq_plots']
    });

    // 회귀 분석 시각화
    if (requirements.regression) {
      pipeline.steps.push({
        type: 'visualization',
        method: 'regression_plots',
        params: {
          show_confidence: true,
          show_prediction: true,
          residual_plots: true
        },
        outputs: ['regression_plots']
      });
    }

    return pipeline;
  }

  generateMLModelVizPipeline(dataInfo, requirements) {
    const pipeline = {
      name: 'custom_ml_model_viz',
      description: '커스텀 ML 모델 시각화 파이프라인',
      category: 'viz_custom',
      steps: []
    };

    // 특성 중요도
    pipeline.steps.push({
      type: 'visualization',
      method: 'feature_importance',
      params: {
        model_type: requirements.model_type || 'auto',
        top_n: requirements.top_features || 20
      },
      outputs: ['feature_importance_plot']
    });

    // 모델 성능 시각화
    if (requirements.problem_type === 'classification') {
      pipeline.steps.push({
        type: 'visualization',
        method: 'confusion_matrix',
        params: {
          normalize: requirements.normalize_cm || true
        },
        outputs: ['confusion_matrix_plot']
      });

      pipeline.steps.push({
        type: 'visualization',
        method: 'roc_curve',
        params: {
          show_auc: true
        },
        outputs: ['roc_curve_plot']
      });
    }

    return pipeline;
  }

  generateClusteringVizPipeline(dataInfo, requirements) {
    const pipeline = {
      name: 'custom_clustering_viz',
      description: '커스텀 클러스터링 시각화 파이프라인',
      category: 'viz_custom',
      steps: []
    };

    // 클러스터 산점도
    pipeline.steps.push({
      type: 'visualization',
      method: 'cluster_scatter',
      params: {
        method: requirements.reduction_method || 'pca',
        n_components: requirements.n_components || 2,
        show_centroids: true
      },
      outputs: ['cluster_scatter_plot']
    });

    // 실루엣 분석
    pipeline.steps.push({
      type: 'visualization',
      method: 'silhouette_analysis',
      params: {
        show_avg_score: true
      },
      outputs: ['silhouette_plot']
    });

    // 엘보우 방법
    if (requirements.show_elbow) {
      pipeline.steps.push({
        type: 'visualization',
        method: 'elbow_method',
        params: {
          k_range: requirements.k_range || [2, 3, 4, 5, 6, 7, 8, 9, 10]
        },
        outputs: ['elbow_plot']
      });
    }

    return pipeline;
  }

  generateTimeSeriesVizPipeline(dataInfo, requirements) {
    const pipeline = {
      name: 'custom_time_series_viz',
      description: '커스텀 시계열 시각화 파이프라인',
      category: 'viz_custom',
      steps: []
    };

    // 시계열 라인 플롯
    pipeline.steps.push({
      type: 'visualization',
      method: 'time_series_line',
      params: {
        show_trend: requirements.show_trend || true,
        show_seasonal: requirements.show_seasonal || true
      },
      outputs: ['time_series_line_plot']
    });

    // 계절성 분해
    pipeline.steps.push({
      type: 'visualization',
      method: 'seasonal_decomposition',
      params: {
        model: requirements.decomposition_model || 'additive',
        period: requirements.period || 'auto'
      },
      outputs: ['seasonal_decomposition_plot']
    });

    // 자기상관 플롯
    pipeline.steps.push({
      type: 'visualization',
      method: 'autocorrelation_plots',
      params: {
        lags: requirements.lags || 40
      },
      outputs: ['autocorrelation_plots']
    });

    return pipeline;
  }

  generateInteractiveVizPipeline(dataInfo, requirements) {
    const pipeline = {
      name: 'custom_interactive_viz',
      description: '커스텀 인터랙티브 시각화 파이프라인',
      category: 'viz_custom',
      steps: []
    };

    // 인터랙티브 대시보드
    pipeline.steps.push({
      type: 'visualization',
      method: 'plotly_dashboard',
      params: {
        charts: requirements.charts || ['scatter', 'histogram', 'box'],
        interactive_filters: requirements.filters || true
      },
      outputs: ['interactive_dashboard']
    });

    // 데이터 테이블
    pipeline.steps.push({
      type: 'visualization',
      method: 'data_table',
      params: {
        interactive: true,
        searchable: true,
        sortable: true
      },
      outputs: ['interactive_data_table']
    });

    return pipeline;
  }

  getVisualizationMethods() {
    const methods = new Set();
    for (const workflow of Object.values(this.visualizationWorkflows)) {
      workflow.steps.forEach(step => {
        if (step.type === 'visualization') {
          methods.add(step.method);
        }
      });
    }
    return Array.from(methods);
  }

  getVisualizationCategories() {
    const categories = new Set();
    for (const workflow of Object.values(this.visualizationWorkflows)) {
      categories.add(workflow.category);
    }
    return Array.from(categories);
  }

  validateVisualizationWorkflow(workflow) {
    const validationResult = {
      valid: true,
      errors: [],
      warnings: []
    };

    // 시각화 워크플로우 특화 검증
    const vizSteps = workflow.steps.filter(step => step.type === 'visualization');
    if (vizSteps.length === 0) {
      validationResult.errors.push('시각화 단계가 없습니다.');
      validationResult.valid = false;
    }

    // 필수 파라미터 검증
    workflow.steps.forEach((step, index) => {
      if (step.type === 'visualization') {
        // 기본 파라미터 존재 여부 확인
        if (!step.params) {
          validationResult.warnings.push(`단계 ${index + 1}: 시각화 파라미터가 정의되지 않았습니다.`);
        }

        // 출력 형식 검증
        if (step.params.format && !['png', 'pdf', 'svg', 'html', 'json'].includes(step.params.format)) {
          validationResult.warnings.push(`단계 ${index + 1}: 지원하지 않는 출력 형식입니다.`);
        }

        // 인터랙티브 차트 검증
        if (step.method.includes('plotly') || step.method.includes('bokeh')) {
          if (!step.params.export_html) {
            validationResult.warnings.push(`단계 ${index + 1}: 인터랙티브 차트의 HTML 출력 설정을 확인하세요.`);
          }
        }

        // 고해상도 출력 검증
        if (step.method.includes('publication')) {
          if (!step.params.dpi || step.params.dpi < 300) {
            validationResult.warnings.push(`단계 ${index + 1}: 출판용 차트는 300 DPI 이상을 권장합니다.`);
          }
        }
      }
    });

    // 메모리 사용량 검증
    const estimatedMemory = this.estimateVisualizationMemory(workflow);
    if (estimatedMemory > 2000) {
      validationResult.warnings.push('높은 메모리 사용량이 예상됩니다. 대용량 데이터 처리 시 주의하세요.');
    }

    return validationResult;
  }

  estimateVisualizationMemory(workflow) {
    const memoryEstimates = {
      'data_overview': 50,
      'distribution_plots': 100,
      'box_plots': 80,
      'correlation_heatmap': 150,
      'scatter_matrix': 200,
      'missing_data_matrix': 100,
      'qq_plots': 120,
      'regression_plots': 150,
      'feature_importance': 80,
      'confusion_matrix': 100,
      'roc_curve': 120,
      'cluster_scatter': 150,
      'cluster_scatter_3d': 200,
      'dendrogram': 100,
      'pca_analysis': 180,
      'tsne_plot': 250,
      'umap_plot': 200,
      'time_series_line': 120,
      'seasonal_decomposition': 150,
      'autocorrelation_plots': 100,
      'plotly_dashboard': 300,
      'bokeh_dashboard': 250,
      'publication_scatter': 150,
      'interactive_dashboard': 400
    };

    let totalMemory = 0;
    workflow.steps.forEach(step => {
      if (step.type === 'visualization') {
        totalMemory += memoryEstimates[step.method] || 100;
      }
    });

    return totalMemory;
  }

  optimizeVisualizationWorkflow(workflow) {
    const optimized = JSON.parse(JSON.stringify(workflow));

    // 1. 중복 시각화 제거
    const uniqueSteps = new Map();
    optimized.steps = optimized.steps.filter(step => {
      if (step.type === 'visualization') {
        const key = `${step.method}_${JSON.stringify(step.params)}`;
        if (uniqueSteps.has(key)) {
          return false;
        }
        uniqueSteps.set(key, true);
      }
      return true;
    });

    // 2. 시각화 순서 최적화 (데이터 개요 -> 분포 -> 상관관계 -> 고급 시각화)
    const stepOrder = {
      'data_overview': 1,
      'distribution_plots': 2,
      'box_plots': 3,
      'correlation_heatmap': 4,
      'scatter_matrix': 5,
      'missing_data_matrix': 6,
      'qq_plots': 7,
      'regression_plots': 8,
      'feature_importance': 9,
      'confusion_matrix': 10,
      'roc_curve': 11,
      'cluster_scatter': 12,
      'dendrogram': 13,
      'pca_analysis': 14,
      'tsne_plot': 15,
      'time_series_line': 16,
      'seasonal_decomposition': 17,
      'plotly_dashboard': 18,
      'publication_scatter': 19
    };

    optimized.steps.sort((a, b) => {
      if (a.type === 'visualization' && b.type === 'visualization') {
        const orderA = stepOrder[a.method] || 99;
        const orderB = stepOrder[b.method] || 99;
        return orderA - orderB;
      }
      return 0;
    });

    // 3. 파라미터 최적화
    optimized.steps.forEach(step => {
      if (step.type === 'visualization') {
        // 기본 figsize 설정
        if (!step.params.figsize) {
          step.params.figsize = this.getOptimalFigsize(step.method);
        }

        // 기본 DPI 설정
        if (!step.params.dpi) {
          step.params.dpi = step.method.includes('publication') ? 300 : 150;
        }

        // 기본 색상 팔레트 설정
        if (!step.params.color_palette && !step.params.cmap) {
          step.params.color_palette = this.getOptimalColorPalette(step.method);
        }
      }
    });

    return optimized;
  }

  getOptimalFigsize(method) {
    const figsizes = {
      'data_overview': [12, 6],
      'distribution_plots': [15, 10],
      'box_plots': [12, 8],
      'correlation_heatmap': [12, 10],
      'scatter_matrix': [15, 15],
      'missing_data_matrix': [12, 8],
      'qq_plots': [12, 8],
      'regression_plots': [15, 10],
      'feature_importance': [12, 8],
      'confusion_matrix': [8, 6],
      'roc_curve': [8, 6],
      'cluster_scatter': [10, 8],
      'cluster_scatter_3d': [12, 10],
      'dendrogram': [15, 10],
      'pca_analysis': [15, 10],
      'tsne_plot': [10, 8],
      'umap_plot': [10, 8],
      'time_series_line': [15, 8],
      'seasonal_decomposition': [15, 12],
      'autocorrelation_plots': [12, 8],
      'publication_scatter': [8, 6],
      'publication_heatmap': [10, 8]
    };

    return figsizes[method] || [10, 6];
  }

  getOptimalColorPalette(method) {
    const palettes = {
      'distribution_plots': 'viridis',
      'box_plots': 'Set2',
      'correlation_heatmap': 'coolwarm',
      'scatter_matrix': 'viridis',
      'missing_data_matrix': 'viridis',
      'feature_importance': 'viridis',
      'confusion_matrix': 'Blues',
      'cluster_scatter': 'tab10',
      'cluster_heatmap': 'RdYlBu_r',
      'pca_analysis': 'viridis',
      'tsne_plot': 'tab10',
      'umap_plot': 'tab10',
      'time_series_line': 'tab10',
      'seasonal_decomposition': 'tab10',
      'publication_scatter': 'deep',
      'publication_heatmap': 'RdBu_r'
    };

    return palettes[method] || 'viridis';
  }

  exportVisualizationWorkflow(workflowName, format = 'json') {
    const workflow = this.getWorkflow(workflowName);
    if (!workflow) {
      throw new Error(`시각화 워크플로우 '${workflowName}'을 찾을 수 없습니다.`);
    }

    switch (format) {
      case 'json':
        return JSON.stringify(workflow, null, 2);
      case 'python':
        return this.convertToPythonScript(workflow);
      case 'r':
        return this.convertToRScript(workflow);
      case 'matplotlib':
        return this.convertToMatplotlibScript(workflow);
      default:
        return workflow;
    }
  }

  convertToPythonScript(workflow) {
    let script = `# ${workflow.name} - ${workflow.description}\n`;
    script += `import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\n`;
    script += `import seaborn as sns\nimport plotly.express as px\nimport plotly.graph_objects as go\n\n`;
    script += `# 데이터 로드\n# df = pd.read_csv('your_data.csv')\n\n`;

    workflow.steps.forEach((step, index) => {
      if (step.type === 'visualization') {
        script += `# 단계 ${index + 1}: ${step.method}\n`;
        script += this.generatePythonCode(step);
        script += `\n`;
      }
    });

    script += `# 모든 플롯 표시\nplt.show()\n`;
    return script;
  }

  convertToRScript(workflow) {
    let script = `# ${workflow.name} - ${workflow.description}\n`;
    script += `library(ggplot2)\nlibrary(dplyr)\nlibrary(corrplot)\nlibrary(plotly)\n\n`;
    script += `# 데이터 로드\n# df <- read.csv('your_data.csv')\n\n`;

    workflow.steps.forEach((step, index) => {
      if (step.type === 'visualization') {
        script += `# 단계 ${index + 1}: ${step.method}\n`;
        script += this.generateRCode(step);
        script += `\n`;
      }
    });

    return script;
  }

  convertToMatplotlibScript(workflow) {
    let script = `# ${workflow.name} - Matplotlib 전용\n`;
    script += `import matplotlib.pyplot as plt\nimport seaborn as sns\nimport pandas as pd\n\n`;
    script += `# 스타일 설정\nsns.set_style('whitegrid')\nplt.rcParams['figure.figsize'] = (12, 8)\n\n`;

    workflow.steps.forEach((step, index) => {
      if (step.type === 'visualization') {
        script += `# 단계 ${index + 1}: ${step.method}\n`;
        script += this.generateMatplotlibCode(step);
        script += `\n`;
      }
    });

    return script;
  }

  generatePythonCode(step) {
    const codeTemplates = {
      'data_overview': 'print(df.info())\nprint(df.describe())\nprint(df.head())',
      'distribution_plots': 'df.hist(bins=30, figsize=(15, 10))\nplt.tight_layout()',
      'box_plots': 'df.boxplot(figsize=(12, 8))\nplt.xticks(rotation=45)',
      'correlation_heatmap': 'corr = df.corr()\nsns.heatmap(corr, annot=True, cmap="coolwarm", center=0)',
      'scatter_matrix': 'pd.plotting.scatter_matrix(df, alpha=0.6, figsize=(15, 15))',
      'feature_importance': 'feature_importance.plot(kind="barh", figsize=(12, 8))',
      'confusion_matrix': 'sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")',
      'roc_curve': 'plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")\nplt.plot([0, 1], [0, 1], "k--")',
      'cluster_scatter': 'plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10")',
      'time_series_line': 'plt.plot(df.index, df.values)\nplt.title("Time Series")',
      'plotly_dashboard': 'fig = px.scatter_matrix(df)\nfig.show()'
    };

    return codeTemplates[step.method] || f'# {step.method} 코드를 여기에 추가하세요';
  }

  generateRCode(step) {
    const codeTemplates = {
      'data_overview': 'str(df)\nsummary(df)\nhead(df)',
      'distribution_plots': 'df %>% gather() %>% ggplot(aes(value)) + geom_histogram() + facet_wrap(~key)',
      'box_plots': 'df %>% gather() %>% ggplot(aes(key, value)) + geom_boxplot()',
      'correlation_heatmap': 'corrplot(cor(df), method="color", type="upper")',
      'scatter_matrix': 'pairs(df)',
      'time_series_line': 'ggplot(df, aes(x=date, y=value)) + geom_line()',
      'plotly_dashboard': 'plot_ly(df, x=~x, y=~y, type="scatter", mode="markers")'
    };

    return codeTemplates[step.method] || f'# {step.method} 코드를 여기에 추가하세요';
  }

  generateMatplotlibCode(step) {
    const codeTemplates = {
      'distribution_plots': 'fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))\ndf.hist(ax=axes)',
      'box_plots': 'df.boxplot()\nplt.xticks(rotation=45)',
      'correlation_heatmap': 'corr = df.corr()\nplt.imshow(corr, cmap="coolwarm")\nplt.colorbar()',
      'scatter_matrix': 'from pandas.plotting import scatter_matrix\nscatter_matrix(df, alpha=0.6, figsize=(15, 15))',
      'time_series_line': 'plt.plot(df.index, df.values)\nplt.title("Time Series")\nplt.xlabel("Time")\nplt.ylabel("Value")'
    };

    return codeTemplates[step.method] || f'# {step.method} matplotlib 코드를 여기에 추가하세요';
  }

  getVisualizationRecommendations(dataInfo) {
    const recommendations = [];

    // 데이터 타입에 따른 추천
    if (dataInfo.hasNumericColumns && dataInfo.hasCategoricalColumns) {
      recommendations.push({
        workflow: 'exploratory_data_viz',
        reason: '수치형과 범주형 데이터가 모두 있어 전체적인 탐색이 필요합니다.',
        priority: 'high'
      });
    }

    if (dataInfo.hasTimeSeriesData) {
      recommendations.push({
        workflow: 'time_series_viz',
        reason: '시계열 데이터가 감지되었습니다.',
        priority: 'high'
      });
    }

    if (dataInfo.hasHighDimensionalData) {
      recommendations.push({
        workflow: 'dimensionality_reduction_viz',
        reason: '고차원 데이터에 대한 차원 축소 시각화가 도움이 됩니다.',
        priority: 'medium'
      });
    }

    if (dataInfo.hasMLModelResults) {
      recommendations.push({
        workflow: 'ml_model_viz',
        reason: '머신러닝 모델 결과가 있어 성능 시각화가 필요합니다.',
        priority: 'high'
      });
    }

    if (dataInfo.hasClusteringResults) {
      recommendations.push({
        workflow: 'clustering_viz',
        reason: '클러스터링 결과가 있어 클러스터 시각화가 필요합니다.',
        priority: 'high'
      });
    }

    if (dataInfo.requiresPublication) {
      recommendations.push({
        workflow: 'publication_ready_viz',
        reason: '출판용 고품질 시각화가 필요합니다.',
        priority: 'medium'
      });
    }

    return recommendations.sort((a, b) => {
      const priorityOrder = { 'high': 3, 'medium': 2, 'low': 1 };
      return priorityOrder[b.priority] - priorityOrder[a.priority];
    });
  }

  createVisualizationSummary(workflowResults) {
    const summary = {
      totalVisualizations: 0,
      visualizationTypes: {},
      outputFiles: [],
      errors: [],
      recommendations: []
    };

    workflowResults.steps.forEach(step => {
      if (step.type === 'visualization') {
        summary.totalVisualizations++;
        
        if (!summary.visualizationTypes[step.method]) {
          summary.visualizationTypes[step.method] = 0;
        }
        summary.visualizationTypes[step.method]++;

        if (step.success && step.result && step.result.output_path) {
          summary.outputFiles.push({
            type: step.method,
            path: step.result.output_path,
            format: step.params.format || 'png'
          });
        }

        if (!step.success) {
          summary.errors.push({
            step: step.method,
            error: step.error
          });
        }
      }
    });

    // 추가 시각화 추천
    if (summary.totalVisualizations < 5) {
      summary.recommendations.push('더 많은 시각화를 통해 데이터 패턴을 파악해보세요.');
    }

    if (!summary.visualizationTypes['correlation_heatmap']) {
      summary.recommendations.push('상관관계 히트맵을 통해 변수 간 관계를 확인해보세요.');
    }

    if (!summary.visualizationTypes['distribution_plots']) {
      summary.recommendations.push('분포 플롯을 통해 데이터 분포를 확인해보세요.');
    }

    return summary;
  }
}
