// tools/analysis/advanced-analyzer.js - 고급 분석 인터페이스
import { Logger } from '../../utils/logger.js';
import { PythonExecutor } from '../common/python-executor.js';
import { ResultFormatter } from '../common/result-formatter.js';

export class AdvancedAnalyzer {
  constructor() {
    this.logger = new Logger();
    this.pythonExecutor = new PythonExecutor();
    this.resultFormatter = new ResultFormatter();
    this.analysisHistory = [];
    this.supportedAnalysisTypes = {
      dimensionality_reduction: ['pca', 'tsne', 'umap', 'factor_analysis'],
      clustering: ['kmeans', 'hierarchical', 'dbscan', 'gaussian_mixture'],
      outlier_detection: ['isolation_forest', 'local_outlier_factor', 'one_class_svm', 'elliptic_envelope'],
      feature_engineering: ['feature_selection', 'feature_scaling', 'feature_creation', 'encoding'],
      timeseries: ['trend_analysis', 'seasonality', 'forecasting', 'anomaly_detection'],
      statistical_tests: ['normality', 'stationarity', 'correlation', 'hypothesis_testing']
    };
  }

  async initialize() {
    try {
      await this.pythonExecutor.initialize();
      this.logger.info('AdvancedAnalyzer 초기화 완료');
    } catch (error) {
      this.logger.error('AdvancedAnalyzer 초기화 실패:', error);
      throw error;
    }
  }

  async performPCA(data, options = {}) {
    try {
      this.logger.info('PCA 분석 시작');

      const {
        n_components = 'auto',
        standardize = true,
        target_variance = 0.95,
        visualize = true,
        feature_names = null
      } = options;

      const scriptPath = 'python/analysis/advanced/pca.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        n_components,
        standardize,
        target_variance,
        visualize,
        feature_names
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 180000
      });

      if (result.success) {
        const pcaResult = JSON.parse(result.output);
        this.recordAnalysisHistory('pca', options, pcaResult);
        return this.resultFormatter.formatAnalysisResult(pcaResult, 'pca_analysis');
      } else {
        throw new Error(`PCA 분석 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('PCA 분석 실패:', error);
      throw error;
    }
  }

  async performClustering(data, options = {}) {
    try {
      this.logger.info('클러스터링 분석 시작');

      const {
        algorithm = 'kmeans',
        n_clusters = 'auto',
        features = null,
        preprocessing = true,
        evaluation = true,
        visualize = true
      } = options;

      const scriptPath = 'python/analysis/advanced/clustering.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        algorithm,
        n_clusters,
        features,
        preprocessing,
        evaluation,
        visualize
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 300000
      });

      if (result.success) {
        const clusteringResult = JSON.parse(result.output);
        this.recordAnalysisHistory('clustering', options, clusteringResult);
        return this.resultFormatter.formatAnalysisResult(clusteringResult, 'clustering_analysis');
      } else {
        throw new Error(`클러스터링 분석 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('클러스터링 분석 실패:', error);
      throw error;
    }
  }

  async detectOutliers(data, options = {}) {
    try {
      this.logger.info('이상치 탐지 시작');

      const {
        method = 'isolation_forest',
        contamination = 'auto',
        features = null,
        visualize = true,
        return_scores = true
      } = options;

      const scriptPath = 'python/analysis/advanced/outlier_detection.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        method,
        contamination,
        features,
        visualize,
        return_scores
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 180000
      });

      if (result.success) {
        const outlierResult = JSON.parse(result.output);
        this.recordAnalysisHistory('outlier_detection', options, outlierResult);
        return this.resultFormatter.formatAnalysisResult(outlierResult, 'outlier_detection');
      } else {
        throw new Error(`이상치 탐지 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('이상치 탐지 실패:', error);
      throw error;
    }
  }

  async performFeatureEngineering(data, options = {}) {
    try {
      this.logger.info('피처 엔지니어링 시작');

      const {
        operations = ['selection', 'scaling', 'encoding'],
        target_column = null,
        selection_method = 'variance_threshold',
        scaling_method = 'standard',
        encoding_method = 'label',
        create_features = false
      } = options;

      const scriptPath = 'python/analysis/advanced/feature_engineering.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        operations,
        target_column,
        selection_method,
        scaling_method,
        encoding_method,
        create_features
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 240000
      });

      if (result.success) {
        const featureResult = JSON.parse(result.output);
        this.recordAnalysisHistory('feature_engineering', options, featureResult);
        return this.resultFormatter.formatAnalysisResult(featureResult, 'feature_engineering');
      } else {
        throw new Error(`피처 엔지니어링 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('피처 엔지니어링 실패:', error);
      throw error;
    }
  }

  async analyzeTimeSeries(data, options = {}) {
    try {
      this.logger.info('시계열 분석 시작');

      const {
        time_column = null,
        value_column = null,
        analysis_type = 'comprehensive',
        decomposition = true,
        stationarity_test = true,
        forecasting = false,
        forecast_periods = 30
      } = options;

      const scriptPath = 'python/analysis/timeseries/trend_analysis.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        time_column,
        value_column,
        analysis_type,
        decomposition,
        stationarity_test,
        forecasting,
        forecast_periods
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 300000
      });

      if (result.success) {
        const timeseriesResult = JSON.parse(result.output);
        this.recordAnalysisHistory('timeseries', options, timeseriesResult);
        return this.resultFormatter.formatAnalysisResult(timeseriesResult, 'timeseries_analysis');
      } else {
        throw new Error(`시계열 분석 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('시계열 분석 실패:', error);
      throw error;
    }
  }

  async performSeasonalityAnalysis(data, options = {}) {
    try {
      this.logger.info('계절성 분석 시작');

      const {
        time_column = null,
        value_column = null,
        seasonal_periods = ['yearly', 'monthly', 'weekly'],
        decomposition_model = 'additive',
        visualize = true
      } = options;

      const scriptPath = 'python/analysis/timeseries/seasonality.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        time_column,
        value_column,
        seasonal_periods,
        decomposition_model,
        visualize
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 240000
      });

      if (result.success) {
        const seasonalityResult = JSON.parse(result.output);
        this.recordAnalysisHistory('seasonality', options, seasonalityResult);
        return this.resultFormatter.formatAnalysisResult(seasonalityResult, 'seasonality_analysis');
      } else {
        throw new Error(`계절성 분석 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('계절성 분석 실패:', error);
      throw error;
    }
  }

  async forecastTimeSeries(data, options = {}) {
    try {
      this.logger.info('시계열 예측 시작');

      const {
        time_column = null,
        value_column = null,
        forecast_periods = 30,
        model_type = 'auto',
        confidence_intervals = true,
        seasonal = true,
        exogenous_vars = null
      } = options;

      const scriptPath = 'python/analysis/timeseries/forecasting.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        time_column,
        value_column,
        forecast_periods,
        model_type,
        confidence_intervals,
        seasonal,
        exogenous_vars
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 600000 // 10분
      });

      if (result.success) {
        const forecastResult = JSON.parse(result.output);
        this.recordAnalysisHistory('forecasting', options, forecastResult);
        return this.resultFormatter.formatAnalysisResult(forecastResult, 'timeseries_forecast');
      } else {
        throw new Error(`시계열 예측 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('시계열 예측 실패:', error);
      throw error;
    }
  }

  async performStatisticalTests(data, options = {}) {
    try {
      this.logger.info('통계 검정 시작');

      const {
        test_types = ['normality', 'correlation', 'independence'],
        columns = null,
        significance_level = 0.05,
        multiple_testing_correction = true
      } = options;

      const scriptPath = 'python/analysis/statistical/hypothesis_tests.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        test_types,
        columns,
        significance_level,
        multiple_testing_correction
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 180000
      });

      if (result.success) {
        const testResult = JSON.parse(result.output);
        this.recordAnalysisHistory('statistical_tests', options, testResult);
        return this.resultFormatter.formatAnalysisResult(testResult, 'statistical_tests');
      } else {
        throw new Error(`통계 검정 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('통계 검정 실패:', error);
      throw error;
    }
  }

  async performDimensionalityReduction(data, options = {}) {
    try {
      this.logger.info('차원 축소 시작');

      const {
        method = 'pca',
        n_components = 2,
        standardize = true,
        visualize = true,
        return_transformed_data = true
      } = options;

      const scriptPath = 'python/analysis/advanced/dimensionality_reduction.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        method,
        n_components,
        standardize,
        visualize,
        return_transformed_data
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 300000
      });

      if (result.success) {
        const reductionResult = JSON.parse(result.output);
        this.recordAnalysisHistory('dimensionality_reduction', options, reductionResult);
        return this.resultFormatter.formatAnalysisResult(reductionResult, 'dimensionality_reduction');
      } else {
        throw new Error(`차원 축소 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('차원 축소 실패:', error);
      throw error;
    }
  }

  async analyzeAssociations(data, options = {}) {
    try {
      this.logger.info('연관성 분석 시작');

      const {
        min_support = 0.1,
        min_confidence = 0.5,
        min_lift = 1.0,
        max_length = 5,
        transaction_column = null
      } = options;

      const scriptPath = 'python/analysis/advanced/association_rules.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        min_support,
        min_confidence,
        min_lift,
        max_length,
        transaction_column
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 240000
      });

      if (result.success) {
        const associationResult = JSON.parse(result.output);
        this.recordAnalysisHistory('association_analysis', options, associationResult);
        return this.resultFormatter.formatAnalysisResult(associationResult, 'association_analysis');
      } else {
        throw new Error(`연관성 분석 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('연관성 분석 실패:', error);
      throw error;
    }
  }

  recordAnalysisHistory(analysisType, options, result) {
    const record = {
      timestamp: new Date().toISOString(),
      analysis_type: analysisType,
      options,
      success: !result.error,
      execution_time: result.execution_time || null
    };

    this.analysisHistory.push(record);
    
    // 히스토리 크기 제한 (최대 100개)
    if (this.analysisHistory.length > 100) {
      this.analysisHistory = this.analysisHistory.slice(-50);
    }
  }

  // 유틸리티 메서드들
  getAnalysisHistory(limit = 10) {
    return this.analysisHistory.slice(-limit);
  }

  getSupportedAnalysisTypes() {
    return this.supportedAnalysisTypes;
  }

  getAnalysisRecommendations(dataInfo) {
    const { numeric_columns, categorical_columns, row_count, has_datetime } = dataInfo;
    const recommendations = [];

    // 차원이 많은 경우 차원 축소 추천
    if (numeric_columns.length > 10) {
      recommendations.push({
        analysis: 'dimensionality_reduction',
        reason: '변수가 많아 차원 축소가 유용할 수 있습니다.',
        methods: ['pca', 'tsne']
      });
    }

    // 충분한 데이터가 있는 경우 클러스터링 추천
    if (row_count > 100 && numeric_columns.length >= 2) {
      recommendations.push({
        analysis: 'clustering',
        reason: '데이터의 패턴을 찾기 위한 클러스터링을 시도해보세요.',
        methods: ['kmeans', 'hierarchical']
      });
    }

    // 시계열 데이터인 경우
    if (has_datetime) {
      recommendations.push({
        analysis: 'timeseries',
        reason: '시간 정보가 있어 시계열 분석이 가능합니다.',
        methods: ['trend_analysis', 'seasonality', 'forecasting']
      });
    }

    // 이상치 탐지는 항상 유용
    if (numeric_columns.length > 0) {
      recommendations.push({
        analysis: 'outlier_detection',
        reason: '데이터 품질 향상을 위한 이상치 탐지를 권장합니다.',
        methods: ['isolation_forest', 'local_outlier_factor']
      });
    }

    return recommendations;
  }

  async getAdvancedAnalyzerStatus() {
    return {
      python_executor_status: await this.pythonExecutor.getExecutionStats(),
      analysis_history_count: this.analysisHistory.length,
      supported_analysis_types: this.supportedAnalysisTypes,
      last_analysis: this.analysisHistory.length > 0 ? 
        this.analysisHistory[this.analysisHistory.length - 1] : null
    };
  }

  async cleanup() {
    await this.pythonExecutor.shutdown();
    this.logger.info('AdvancedAnalyzer 정리 완료');
  }
}