// tools/analysis/time-series-analyzer.js - 시계열 분석 도구
import { PythonExecutor } from '../common/python-executor.js';
import { ResultFormatter } from '../common/result-formatter.js';
import { Logger } from '../../utils/logger.js';
import { ConfigLoader } from '../../utils/config-loader.js';

export class TimeSeriesAnalyzer {
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
      this.logger.info('TimeSeriesAnalyzer 초기화 완료');
    } catch (error) {
      this.logger.error('TimeSeriesAnalyzer 초기화 실패:', error);
      this.analysisConfig = this.getDefaultConfig();
    }
  }

  getDefaultConfig() {
    return {
      timeseries: {
        trend_analysis: {
          complexity: 0.4,
          estimated_time_ms: 3000,
          python_script: 'python/analysis/timeseries/trend_analysis.py'
        },
        seasonality: {
          complexity: 0.5,
          estimated_time_ms: 4000,
          python_script: 'python/analysis/timeseries/seasonality.py'
        },
        forecasting: {
          complexity: 0.7,
          estimated_time_ms: 8000,
          python_script: 'python/analysis/timeseries/forecasting.py'
        },
        anomaly_detection: {
          complexity: 0.6,
          estimated_time_ms: 6000,
          python_script: 'python/analysis/timeseries/anomaly_detection.py'
        },
        decomposition: {
          complexity: 0.4,
          estimated_time_ms: 3000,
          python_script: 'python/analysis/timeseries/decomposition.py'
        }
      }
    };
  }

  async analyzeTrend(data, options = {}) {
    const {
      date_column,
      value_column,
      method = 'linear',
      period = null,
      plot_results = true,
      confidence_level = 0.95
    } = options;

    try {
      this.logger.info('트렌드 분석 시작');

      if (!date_column || !value_column) {
        throw new Error('date_column과 value_column이 필요합니다.');
      }

      const config = this.getAnalysisConfig('trend_analysis');
      const scriptPath = config.python_script || 'python/analysis/timeseries/trend_analysis.py';

      const params = {
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        date_column,
        value_column,
        method,
        period,
        plot_results,
        confidence_level
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const analysisResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(analysisResult, 'timeseries_analysis');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('트렌드 분석 실패:', error);
      throw error;
    }
  }

  async analyzeSeasonality(data, options = {}) {
    const {
      date_column,
      value_column,
      model = 'additive',
      period = null,
      auto_detect_period = true,
      plot_results = true
    } = options;

    try {
      this.logger.info('계절성 분석 시작');

      if (!date_column || !value_column) {
        throw new Error('date_column과 value_column이 필요합니다.');
      }

      const config = this.getAnalysisConfig('seasonality');
      const scriptPath = config.python_script || 'python/analysis/timeseries/seasonality.py';

      const params = {
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        date_column,
        value_column,
        model,
        period,
        auto_detect_period,
        plot_results
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const analysisResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(analysisResult, 'timeseries_analysis');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('계절성 분석 실패:', error);
      throw error;
    }
  }

  async forecast(data, options = {}) {
    const {
      date_column,
      value_column,
      method = 'arima',
      periods = 12,
      confidence_level = 0.95,
      seasonal = true,
      auto_tune = true,
      plot_results = true,
      include_prediction_intervals = true
    } = options;

    try {
      this.logger.info(`시계열 예측 시작: ${method}`);

      if (!date_column || !value_column) {
        throw new Error('date_column과 value_column이 필요합니다.');
      }

      const config = this.getAnalysisConfig('forecasting');
      const scriptPath = config.python_script || 'python/analysis/timeseries/forecasting.py';

      const params = {
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        date_column,
        value_column,
        method,
        periods,
        confidence_level,
        seasonal,
        auto_tune,
        plot_results,
        include_prediction_intervals
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const analysisResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(analysisResult, 'timeseries_analysis');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('시계열 예측 실패:', error);
      throw error;
    }
  }

  async detectAnomalies(data, options = {}) {
    const {
      date_column,
      value_column,
      method = 'isolation_forest',
      contamination = 0.1,
      window_size = 10,
      threshold = 2,
      plot_results = true,
      return_clean_data = false
    } = options;

    try {
      this.logger.info(`시계열 이상 탐지 시작: ${method}`);

      if (!date_column || !value_column) {
        throw new Error('date_column과 value_column이 필요합니다.');
      }

      const config = this.getAnalysisConfig('anomaly_detection');
      const scriptPath = config.python_script || 'python/analysis/timeseries/anomaly_detection.py';

      const params = {
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        date_column,
        value_column,
        method,
        contamination,
        window_size,
        threshold,
        plot_results,
        return_clean_data
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const analysisResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(analysisResult, 'timeseries_analysis');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('시계열 이상 탐지 실패:', error);
      throw error;
    }
  }

  async decompose(data, options = {}) {
    const {
      date_column,
      value_column,
      model = 'additive',
      period = null,
      extrapolate_trend = 'freq',
      plot_results = true
    } = options;

    try {
      this.logger.info('시계열 분해 시작');

      if (!date_column || !value_column) {
        throw new Error('date_column과 value_column이 필요합니다.');
      }

      const config = this.getAnalysisConfig('decomposition');
      const scriptPath = config.python_script || 'python/analysis/timeseries/decomposition.py';

      const params = {
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        date_column,
        value_column,
        model,
        period,
        extrapolate_trend,
        plot_results
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const analysisResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(analysisResult, 'timeseries_analysis');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('시계열 분해 실패:', error);
      throw error;
    }
  }

  async analyzeStationarity(data, options = {}) {
    const {
      date_column,
      value_column,
      test_type = 'adf',
      max_lags = null,
      plot_results = true
    } = options;

    try {
      this.logger.info('정상성 검정 시작');

      if (!date_column || !value_column) {
        throw new Error('date_column과 value_column이 필요합니다.');
      }

      const scriptPath = 'python/analysis/timeseries/stationarity_test.py';

      const params = {
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        date_column,
        value_column,
        test_type,
        max_lags,
        plot_results
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const analysisResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(analysisResult, 'timeseries_analysis');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('정상성 검정 실패:', error);
      throw error;
    }
  }

  async autocorrelationAnalysis(data, options = {}) {
    const {
      date_column,
      value_column,
      max_lags = 40,
      alpha = 0.05,
      plot_results = true
    } = options;

    try {
      this.logger.info('자기상관 분석 시작');

      if (!date_column || !value_column) {
        throw new Error('date_column과 value_column이 필요합니다.');
      }

      const scriptPath = 'python/analysis/timeseries/autocorrelation.py';

      const params = {
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        date_column,
        value_column,
        max_lags,
        alpha,
        plot_results
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const analysisResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(analysisResult, 'timeseries_analysis');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('자기상관 분석 실패:', error);
      throw error;
    }
  }

  async comprehensiveTimeSeriesAnalysis(data, options = {}) {
    const {
      date_column,
      value_column,
      include_trend = true,
      include_seasonality = true,
      include_forecasting = true,
      include_anomaly_detection = true,
      include_decomposition = true,
      forecast_periods = 12,
      generate_summary = true
    } = options;

    try {
      this.logger.info('종합 시계열 분석 시작');

      if (!date_column || !value_column) {
        throw new Error('date_column과 value_column이 필요합니다.');
      }

      const results = {
        analysis_type: 'comprehensive_timeseries',
        timestamp: new Date().toISOString(),
        results: {},
        execution_info: {
          started_at: new Date().toISOString(),
          analyses_requested: []
        }
      };

      // 요청된 분석 목록 기록
      if (include_trend) results.execution_info.analyses_requested.push('trend_analysis');
      if (include_seasonality) results.execution_info.analyses_requested.push('seasonality');
      if (include_forecasting) results.execution_info.analyses_requested.push('forecasting');
      if (include_anomaly_detection) results.execution_info.analyses_requested.push('anomaly_detection');
      if (include_decomposition) results.execution_info.analyses_requested.push('decomposition');

      // 트렌드 분석
      if (include_trend) {
        try {
          this.logger.info('트렌드 분석 실행 중...');
          results.results.trend_analysis = await this.analyzeTrend(data, { 
            date_column, 
            value_column, 
            ...options 
          });
        } catch (error) {
          this.logger.warn('트렌드 분석 실패:', error);
          results.results.trend_analysis = { 
            error: true, 
            message: error.message,
            analysis_type: 'trend_analysis'
          };
        }
      }

      // 계절성 분석
      if (include_seasonality) {
        try {
          this.logger.info('계절성 분석 실행 중...');
          results.results.seasonality = await this.analyzeSeasonality(data, { 
            date_column, 
            value_column, 
            ...options 
          });
        } catch (error) {
          this.logger.warn('계절성 분석 실패:', error);
          results.results.seasonality = { 
            error: true, 
            message: error.message,
            analysis_type: 'seasonality'
          };
        }
      }

      // 시계열 분해
      if (include_decomposition) {
        try {
          this.logger.info('시계열 분해 실행 중...');
          results.results.decomposition = await this.decompose(data, { 
            date_column, 
            value_column, 
            ...options 
          });
        } catch (error) {
          this.logger.warn('시계열 분해 실패:', error);
          results.results.decomposition = { 
            error: true, 
            message: error.message,
            analysis_type: 'decomposition'
          };
        }
      }

      // 이상 탐지
      if (include_anomaly_detection) {
        try {
          this.logger.info('이상 탐지 실행 중...');
          results.results.anomaly_detection = await this.detectAnomalies(data, { 
            date_column, 
            value_column, 
            ...options 
          });
        } catch (error) {
          this.logger.warn('이상 탐지 실패:', error);
          results.results.anomaly_detection = { 
            error: true, 
            message: error.message,
            analysis_type: 'anomaly_detection'
          };
        }
      }

      // 예측
      if (include_forecasting) {
        try {
          this.logger.info('시계열 예측 실행 중...');
          results.results.forecasting = await this.forecast(data, { 
            date_column, 
            value_column, 
            periods: forecast_periods,
            ...options 
          });
        } catch (error) {
          this.logger.warn('시계열 예측 실패:', error);
          results.results.forecasting = { 
            error: true, 
            message: error.message,
            analysis_type: 'forecasting'
          };
        }
      }

      // 실행 시간 기록
      results.execution_info.completed_at = new Date().toISOString();
      results.execution_info.total_duration_ms = Date.now() - new Date(results.execution_info.started_at).getTime();

      // 요약 생성
      if (generate_summary) {
        results.summary = this.generateTimeSeriesSummary(results.results);
      }

      this.logger.info('종합 시계열 분석 완료');
      return this.resultFormatter.formatAnalysisResult(results, 'comprehensive_timeseries_analysis');

    } catch (error) {
      this.logger.error('종합 시계열 분석 실패:', error);
      throw error;
    }
  }

  generateTimeSeriesSummary(analysisResults) {
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
    if (summary.successful_analyses.includes('trend_analysis')) {
      summary.recommendations.push('트렌드 분석 결과를 바탕으로 장기 전략을 수립해보세요.');
    }

    if (summary.successful_analyses.includes('seasonality')) {
      summary.recommendations.push('계절성 패턴을 고려한 예측 모델을 구축해보세요.');
    }

    if (summary.successful_analyses.includes('anomaly_detection')) {
      summary.recommendations.push('탐지된 이상 데이터를 조사하여 특별한 사건이나 변화를 확인해보세요.');
    }

    if (summary.successful_analyses.includes('forecasting')) {
      summary.recommendations.push('예측 결과의 신뢰구간을 고려하여 리스크 관리 계획을 세워보세요.');
    }

    return summary;
  }

  // 분석 설정 가져오기
  getAnalysisConfig(methodName) {
    if (this.analysisConfig?.timeseries?.[methodName]) {
      return this.analysisConfig.timeseries[methodName];
    }
    
    return this.getDefaultConfig().timeseries[methodName] || {
      complexity: 0.5,
      estimated_time_ms: 5000,
      python_script: `python/analysis/timeseries/${methodName}.py`
    };
  }

  // 사용 가능한 예측 방법들
  getAvailableForecastingMethods() {
    return {
      arima: 'ARIMA - 자기회귀통합이동평균 모델',
      prophet: 'Prophet - Facebook의 시계열 예측 모델',
      exponential_smoothing: 'Exponential Smoothing - 지수 평활법',
      lstm: 'LSTM - 장단기 메모리 신경망',
      linear_regression: 'Linear Regression - 선형 회귀'
    };
  }

  // 사용 가능한 이상 탐지 방법들
  getAvailableAnomalyDetectionMethods() {
    return {
      isolation_forest: 'Isolation Forest - 고립 숲',
      statistical: 'Statistical - 통계적 방법 (Z-score, IQR)',
      lstm_autoencoder: 'LSTM Autoencoder - LSTM 오토인코더',
      prophet: 'Prophet - Prophet 기반 이상 탐지',
      moving_average: 'Moving Average - 이동 평균 기반'
    };
  }

  // 시계열 데이터 유효성 검사
  async validateTimeSeriesData(data, date_column, value_column) {
    try {
      const scriptPath = 'python/analysis/timeseries/validation.py';
      
      const params = {
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        date_column,
        value_column
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        return JSON.parse(result.output);
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('시계열 데이터 검증 실패:', error);
      return {
        is_valid: false,
        errors: [error.message],
        warnings: []
      };
    }
  }

  // 시계열 분석 방법 목록 반환
  getAvailableMethods() {
    return {
      trend_analysis: 'Trend Analysis - 트렌드 분석',
      seasonality: 'Seasonality Analysis - 계절성 분석',
      forecasting: 'Time Series Forecasting - 시계열 예측',
      anomaly_detection: 'Anomaly Detection - 이상 탐지',
      decomposition: 'Time Series Decomposition - 시계열 분해',
      stationarity_test: 'Stationarity Test - 정상성 검정',
      autocorrelation: 'Autocorrelation Analysis - 자기상관 분석'
    };
  }

  // 성능 메트릭 반환
  getPerformanceMetrics() {
    return {
      total_analyses_performed: this.performanceMetrics?.get('total_timeseries_analyses') || 0,
      average_execution_time: this.performanceMetrics?.get('avg_timeseries_execution_time') || 0,
      success_rate: this.performanceMetrics?.get('timeseries_success_rate') || 0,
      most_used_method: this.performanceMetrics?.get('most_used_timeseries_method') || 'unknown'
    };
  }
}