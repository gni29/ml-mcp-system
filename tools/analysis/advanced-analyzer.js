// tools/analysis/advanced-analyzer.js - 고급 분석 도구
import { PythonExecutor } from '../common/python-executor.js';
import { ResultFormatter } from '../common/result-formatter.js';
import { Logger } from '../../utils/logger.js';
import { ConfigLoader } from '../../utils/config-loader.js';

export class AdvancedAnalyzer {
  constructor() {
    this.pythonExecutor = new PythonExecutor();
    this.resultFormatter = new ResultFormatter();
    this.logger = new Logger();
    this.configLoader = new ConfigLoader();
    this.analysisConfig = null;
    
    this.initializeAnalyzer();
  }

  async initializeAnalyzer() {
    try {
      this.analysisConfig = await this.configLoader.loadConfig('analysis-methods.json');
      this.logger.info('AdvancedAnalyzer 초기화 완료');
    } catch (error) {
      this.logger.error('AdvancedAnalyzer 초기화 실패:', error);
      this.analysisConfig = this.getDefaultConfig();
    }
  }

  getDefaultConfig() {
    return {
      advanced: {
        pca: {
          complexity: 0.6,
          estimated_time_ms: 5000,
          python_script: 'python/analysis/advanced/pca.py'
        },
        clustering: {
          complexity: 0.7,
          estimated_time_ms: 8000,
          python_script: 'python/analysis/advanced/clustering.py'
        },
        outlier_detection: {
          complexity: 0.5,
          estimated_time_ms: 4000,
          python_script: 'python/analysis/advanced/outlier_detection.py'
        },
        feature_engineering: {
          complexity: 0.6,
          estimated_time_ms: 6000,
          python_script: 'python/analysis/advanced/feature_engineering.py'
        }
      }
    };
  }

  async performPCA(data, options = {}) {
    const {
      n_components = 2,
      standardize = true,
      variance_threshold = 0.95,
      plot_results = true,
      include_loadings = true
    } = options;

    try {
      this.logger.info('PCA 분석 시작');

      const config = this.getAnalysisConfig('pca');
      const scriptPath = config.python_script || 'python/analysis/advanced/pca.py';

      const params = {
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        n_components,
        standardize,
        variance_threshold,
        plot_results,
        include_loadings
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const analysisResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(analysisResult, 'advanced_analysis');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('PCA 분석 실패:', error);
      throw error;
    }
  }

  async performClustering(data, options = {}) {
    const {
      algorithm = 'kmeans',
      n_clusters = 3,
      auto_optimize = true,
      max_clusters = 10,
      random_state = 42,
      plot_results = true,
      eps = 0.5,
      min_samples = 5
    } = options;

    try {
      this.logger.info(`클러스터링 분석 시작: ${algorithm}`);

      const config = this.getAnalysisConfig('clustering');
      const scriptPath = config.python_script || 'python/analysis/advanced/clustering.py';

      const params = {
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        algorithm,
        n_clusters,
        auto_optimize,
        max_clusters,
        random_state,
        plot_results,
        eps,
        min_samples
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const analysisResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(analysisResult, 'advanced_analysis');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('클러스터링 분석 실패:', error);
      throw error;
    }
  }

  async detectOutliers(data, options = {}) {
    const {
      method = 'iqr',
      threshold = 1.5,
      contamination = 0.1,
      plot_results = true,
      return_clean_data = true
    } = options;

    try {
      this.logger.info(`이상치 탐지 시작: ${method}`);

      const config = this.getAnalysisConfig('outlier_detection');
      const scriptPath = config.python_script || 'python/analysis/advanced/outlier_detection.py';

      const params = {
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        method,
        threshold,
        contamination,
        plot_results,
        return_clean_data
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const analysisResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(analysisResult, 'advanced_analysis');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('이상치 탐지 실패:', error);
      throw error;
    }
  }

  async performFeatureEngineering(data, options = {}) {
    const {
      operations = ['scaling', 'encoding'],
      scaling_method = 'standard',
      encoding_method = 'onehot',
      polynomial_degree = 2,
      interaction_features = false,
      target_column = null,
      remove_low_variance = true,
      variance_threshold = 0.01,
      correlation_threshold = 0.95
    } = options;

    try {
      this.logger.info('피처 엔지니어링 시작');

      const config = this.getAnalysisConfig('feature_engineering');
      const scriptPath = config.python_script || 'python/analysis/advanced/feature_engineering.py';

      const params = {
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        operations: operations.join(','),
        scaling_method,
        encoding_method,
        polynomial_degree,
        interaction_features,
        target_column,
        remove_low_variance,
        variance_threshold,
        correlation_threshold
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const analysisResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(analysisResult, 'advanced_analysis');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('피처 엔지니어링 실패:', error);
      throw error;
    }
  }

  async performDimensionalityReduction(data, options = {}) {
    const {
      method = 'pca',
      n_components = 2,
      perplexity = 30,
      n_neighbors = 15,
      min_dist = 0.1,
      random_state = 42,
      plot_results = true
    } = options;

    try {
      this.logger.info(`차원 축소 분석 시작: ${method}`);

      const config = this.getAnalysisConfig('pca'); // PCA 스크립트 재사용 또는 별도 스크립트
      let scriptPath;

      switch (method) {
        case 'pca':
          scriptPath = config.python_script || 'python/analysis/advanced/pca.py';
          break;
        case 'tsne':
          scriptPath = 'python/analysis/advanced/tsne.py';
          break;
        case 'umap':
          scriptPath = 'python/analysis/advanced/umap.py';
          break;
        default:
          scriptPath = config.python_script || 'python/analysis/advanced/pca.py';
      }

      const params = {
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        method,
        n_components,
        perplexity,
        n_neighbors,
        min_dist,
        random_state,
        plot_results
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const analysisResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(analysisResult, 'advanced_analysis');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('차원 축소 분석 실패:', error);
      throw error;
    }
  }

  async comprehensiveAdvancedAnalysis(data, options = {}) {
    const {
      include_pca = true,
      include_clustering = true,
      include_outlier_detection = true,
      include_feature_engineering = false,
      generate_summary = true
    } = options;

    try {
      this.logger.info('종합 고급 분석 시작');

      const results = {
        analysis_type: 'comprehensive_advanced',
        timestamp: new Date().toISOString(),
        results: {},
        execution_info: {
          started_at: new Date().toISOString(),
          analyses_requested: []
        }
      };

      // 요청된 분석 목록 기록
      if (include_pca) results.execution_info.analyses_requested.push('pca');
      if (include_clustering) results.execution_info.analyses_requested.push('clustering');
      if (include_outlier_detection) results.execution_info.analyses_requested.push('outlier_detection');
      if (include_feature_engineering) results.execution_info.analyses_requested.push('feature_engineering');

      // PCA 분석
      if (include_pca) {
        try {
          this.logger.info('PCA 분석 실행 중...');
          results.results.pca = await this.performPCA(data, options);
        } catch (error) {
          this.logger.warn('PCA 분석 실패:', error);
          results.results.pca = { 
            error: true, 
            message: error.message,
            analysis_type: 'pca'
          };
        }
      }

      // 클러스터링 분석
      if (include_clustering) {
        try {
          this.logger.info('클러스터링 분석 실행 중...');
          results.results.clustering = await this.performClustering(data, options);
        } catch (error) {
          this.logger.warn('클러스터링 분석 실패:', error);
          results.results.clustering = { 
            error: true, 
            message: error.message,
            analysis_type: 'clustering'
          };
        }
      }

      // 이상치 탐지
      if (include_outlier_detection) {
        try {
          this.logger.info('이상치 탐지 실행 중...');
          results.results.outlier_detection = await this.detectOutliers(data, options);
        } catch (error) {
          this.logger.warn('이상치 탐지 실패:', error);
          results.results.outlier_detection = { 
            error: true, 
            message: error.message,
            analysis_type: 'outlier_detection'
          };
        }
      }

      // 피처 엔지니어링
      if (include_feature_engineering) {
        try {
          this.logger.info('피처 엔지니어링 실행 중...');
          results.results.feature_engineering = await this.performFeatureEngineering(data, options);
        } catch (error) {
          this.logger.warn('피처 엔지니어링 실패:', error);
          results.results.feature_engineering = { 
            error: true, 
            message: error.message,
            analysis_type: 'feature_engineering'
          };
        }
      }

      // 실행 시간 기록
      results.execution_info.completed_at = new Date().toISOString();
      results.execution_info.total_duration_ms = Date.now() - new Date(results.execution_info.started_at).getTime();

      // 요약 생성
      if (generate_summary) {
        results.summary = this.generateAdvancedAnalysisSummary(results.results);
      }

      this.logger.info('종합 고급 분석 완료');
      return this.resultFormatter.formatAnalysisResult(results, 'comprehensive_advanced_analysis');

    } catch (error) {
      this.logger.error('종합 고급 분석 실패:', error);
      throw error;
    }
  }

  generateAdvancedAnalysisSummary(analysisResults) {
    const summary = {
      successful_analyses: [],
      failed_analyses: [],
      key_insights: [],
      recommendations: []
    };

    // 성공/실패 분석 분류
    Object.entries(analysisResults).forEach(([analysisType, result]) => {
      if (result.error) {
        summary.failed_analyses.push(analysisType);
      } else {
        summary.successful_analyses.push(analysisType);
        
        // 주요 인사이트 추출
        if (result.metadata?.summary) {
          summary.key_insights.push(`${analysisType}: ${result.metadata.summary}`);
        }
      }
    });

    // 권장사항 생성
    if (summary.successful_analyses.includes('pca')) {
      summary.recommendations.push('PCA 결과를 바탕으로 차원 축소된 데이터를 활용한 추가 분석을 고려해보세요.');
    }

    if (summary.successful_analyses.includes('clustering')) {
      summary.recommendations.push('클러스터링 결과를 바탕으로 그룹별 특성 분석을 수행해보세요.');
    }

    if (summary.successful_analyses.includes('outlier_detection')) {
      summary.recommendations.push('탐지된 이상치를 제거하거나 별도 분석하여 데이터 품질을 개선해보세요.');
    }

    return summary;
  }

  // 분석 설정 가져오기
  getAnalysisConfig(methodName) {
    if (this.analysisConfig?.advanced?.[methodName]) {
      return this.analysisConfig.advanced[methodName];
    }
    
    return this.getDefaultConfig().advanced[methodName] || {
      complexity: 0.5,
      estimated_time_ms: 5000,
      python_script: `python/analysis/advanced/${methodName}.py`
    };
  }

  // 분석 메서드 목록 반환
  getAvailableMethods() {
    return {
      pca: 'Principal Component Analysis - 주성분 분석',
      clustering: 'Clustering Analysis - 클러스터링 분석',
      outlier_detection: 'Outlier Detection - 이상치 탐지',
      feature_engineering: 'Feature Engineering - 피처 엔지니어링',
      dimensionality_reduction: 'Dimensionality Reduction - 차원 축소'
    };
  }

  // 성능 메트릭 반환
  getPerformanceMetrics() {
    return {
      total_analyses_performed: this.performanceMetrics?.get('total_analyses') || 0,
      average_execution_time: this.performanceMetrics?.get('avg_execution_time') || 0,
      success_rate: this.performanceMetrics?.get('success_rate') || 0,
      most_used_method: this.performanceMetrics?.get('most_used_method') || 'unknown'
    };
  }
}