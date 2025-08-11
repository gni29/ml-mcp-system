// tools/visualization/chart-generator.js - 차트 및 시각화 생성 인터페이스
import { Logger } from '../../utils/logger.js';
import { PythonExecutor } from '../common/python-executor.js';
import { ResultFormatter } from '../common/result-formatter.js';

export class ChartGenerator {
  constructor() {
    this.logger = new Logger();
    this.pythonExecutor = new PythonExecutor();
    this.resultFormatter = new ResultFormatter();
    this.chartHistory = [];
    this.supportedChartTypes = {
      basic: ['bar', 'line', 'scatter', 'pie', 'histogram', 'box', 'violin'],
      statistical: ['correlation_heatmap', 'distribution', 'qq_plot', 'residual_plot'],
      ml: ['confusion_matrix', 'roc_curve', 'feature_importance', 'learning_curve'],
      advanced: ['3d_scatter', 'parallel_coordinates', 'radar_chart', 'treemap']
    };
  }

  async initialize() {
    try {
      await this.pythonExecutor.initialize();
      this.logger.info('ChartGenerator 초기화 완료');
    } catch (error) {
      this.logger.error('ChartGenerator 초기화 실패:', error);
      throw error;
    }
  }

  async generateChart(data, chartConfig) {
    try {
      this.logger.info('차트 생성 시작', { chartType: chartConfig.chart_type });

      const {
        chart_type,
        chart_category = 'basic',
        x_column = null,
        y_column = null,
        color_column = null,
        title = null,
        style = 'default',
        interactive = false,
        save_format = 'png'
      } = chartConfig;

      // 차트 타입 검증
      this.validateChartConfig(chartConfig);

      // 적절한 Python 스크립트 선택
      const scriptPath = this.getVisualizationScriptPath(chart_category, chart_type);
      
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        chart_type,
        chart_category,
        x_column,
        y_column,
        color_column,
        title,
        style,
        interactive,
        save_format
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 180000 // 3분
      });

      if (result.success) {
        const chartResult = JSON.parse(result.output);
        this.recordChartHistory(chartConfig, chartResult);
        return this.resultFormatter.formatAnalysisResult(chartResult, 'chart_generation');
      } else {
        throw new Error(`차트 생성 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('차트 생성 실패:', error);
      throw error;
    }
  }

  async generateBasicChart(data, chartType, options = {}) {
    try {
      this.logger.info('기본 차트 생성 시작', { chartType });

      const {
        x_column = null,
        y_column = null,
        color_column = null,
        title = null,
        figsize = [10, 6],
        dpi = 300,
        style = 'seaborn'
      } = options;

      const scriptPath = 'python/visualization/basic_charts.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        chart_type: chartType,
        x_column,
        y_column,
        color_column,
        title,
        figsize,
        dpi,
        style
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 120000
      });

      if (result.success) {
        const chartResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(chartResult, 'basic_chart');
      } else {
        throw new Error(`기본 차트 생성 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('기본 차트 생성 실패:', error);
      throw error;
    }
  }

  async generateStatisticalPlot(data, plotType, options = {}) {
    try {
      this.logger.info('통계 플롯 생성 시작', { plotType });

      const {
        x_column = null,
        y_column = null,
        hue_column = null,
        plot_params = {},
        save_format = 'png'
      } = options;

      const scriptPath = 'python/visualization/statistical_plots.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        plot_type: plotType,
        x_column,
        y_column,
        hue_column,
        plot_params,
        save_format
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 150000
      });

      if (result.success) {
        const plotResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(plotResult, 'statistical_plot');
      } else {
        throw new Error(`통계 플롯 생성 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('통계 플롯 생성 실패:', error);
      throw error;
    }
  }

  async generateMLVisualization(data, vizType, modelData = null, options = {}) {
    try {
      this.logger.info('ML 시각화 생성 시작', { vizType });

      const {
        model_type = null,
        predictions = null,
        feature_names = null,
        class_names = null,
        save_format = 'png'
      } = options;

      const scriptPath = 'python/visualization/ml_visualization.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        model_data: modelData,
        viz_type: vizType,
        model_type,
        predictions,
        feature_names,
        class_names,
        save_format
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 180000
      });

      if (result.success) {
        const vizResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(vizResult, 'ml_visualization');
      } else {
        throw new Error(`ML 시각화 생성 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('ML 시각화 생성 실패:', error);
      throw error;
    }
  }

  async generateInteractivePlot(data, plotType, options = {}) {
    try {
      this.logger.info('인터랙티브 플롯 생성 시작', { plotType });

      const {
        x_column = null,
        y_column = null,
        color_column = null,
        size_column = null,
        hover_data = null,
        title = null,
        theme = 'plotly'
      } = options;

      const scriptPath = 'python/visualization/interactive_plots.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        plot_type: plotType,
        x_column,
        y_column,
        color_column,
        size_column,
        hover_data,
        title,
        theme
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 180000
      });

      if (result.success) {
        const plotResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(plotResult, 'interactive_plot');
      } else {
        throw new Error(`인터랙티브 플롯 생성 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('인터랙티브 플롯 생성 실패:', error);
      throw error;
    }
  }

  async generateDashboard(data, dashboardConfig) {
    try {
      this.logger.info('대시보드 생성 시작');

      const {
        charts = [],
        layout = 'grid',
        title = 'Data Dashboard',
        theme = 'default',
        export_format = 'html'
      } = dashboardConfig;

      const scriptPath = 'python/visualization/dashboard.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        charts,
        layout,
        title,
        theme,
        export_format
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 300000 // 5분
      });

      if (result.success) {
        const dashboardResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(dashboardResult, 'dashboard');
      } else {
        throw new Error(`대시보드 생성 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('대시보드 생성 실패:', error);
      throw error;
    }
  }

  async generateCustomVisualization(data, customCode, options = {}) {
    try {
      this.logger.info('커스텀 시각화 생성 시작');

      const {
        imports = [],
        parameters = {},
        save_format = 'png'
      } = options;

      const scriptPath = 'python/visualization/custom_viz.py';
      const args = {
        data_source: typeof data === 'string' ? data : 'memory',
        data_content: typeof data === 'string' ? null : JSON.stringify(data),
        custom_code: customCode,
        imports,
        parameters,
        save_format
      };

      const result = await this.pythonExecutor.executeFile(scriptPath, {
        args: JSON.stringify(args),
        timeout: 240000 // 4분
      });

      if (result.success) {
        const customResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(customResult, 'custom_visualization');
      } else {
        throw new Error(`커스텀 시각화 생성 실패: ${result.error}`);
      }
    } catch (error) {
      this.logger.error('커스텀 시각화 생성 실패:', error);
      throw error;
    }
  }

  validateChartConfig(chartConfig) {
    const { chart_type, chart_category } = chartConfig;

    if (!chart_category || !this.supportedChartTypes[chart_category]) {
      throw new Error(`지원하지 않는 차트 카테고리: ${chart_category}`);
    }

    if (!chart_type || !this.supportedChartTypes[chart_category].includes(chart_type)) {
      throw new Error(`지원하지 않는 차트 타입: ${chart_type} (카테고리: ${chart_category})`);
    }
  }

  getVisualizationScriptPath(category, type) {
    const scriptMap = {
      basic: 'python/visualization/basic_charts.py',
      statistical: 'python/visualization/statistical_plots.py',
      ml: 'python/visualization/ml_visualization.py',
      advanced: 'python/visualization/advanced_plots.py'
    };

    return scriptMap[category] || 'python/visualization/basic_charts.py';
  }

  recordChartHistory(chartConfig, result) {
    const record = {
      timestamp: new Date().toISOString(),
      chart_config: chartConfig,
      success: !result.error,
      chart_files: result.chart_files || [],
      generation_time: result.generation_time || null
    };

    this.chartHistory.push(record);
    
    // 히스토리 크기 제한 (최대 100개)
    if (this.chartHistory.length > 100) {
      this.chartHistory = this.chartHistory.slice(-50);
    }
  }

  // 빠른 차트 생성 메서드들
  async createBarChart(data, xColumn, yColumn, options = {}) {
    return await this.generateBasicChart(data, 'bar', {
      x_column: xColumn,
      y_column: yColumn,
      ...options
    });
  }

  async createLineChart(data, xColumn, yColumn, options = {}) {
    return await this.generateBasicChart(data, 'line', {
      x_column: xColumn,
      y_column: yColumn,
      ...options
    });
  }

  async createScatterPlot(data, xColumn, yColumn, options = {}) {
    return await this.generateBasicChart(data, 'scatter', {
      x_column: xColumn,
      y_column: yColumn,
      ...options
    });
  }

  async createHistogram(data, column, options = {}) {
    return await this.generateBasicChart(data, 'histogram', {
      x_column: column,
      ...options
    });
  }

  async createBoxPlot(data, column, options = {}) {
    return await this.generateBasicChart(data, 'box', {
      y_column: column,
      ...options
    });
  }

  async createHeatmap(data, options = {}) {
    return await this.generateStatisticalPlot(data, 'heatmap', options);
  }

  async createPairPlot(data, options = {}) {
    return await this.generateStatisticalPlot(data, 'pairplot', options);
  }

  async createCorrelationMatrix(data, options = {}) {
    return await this.generateStatisticalPlot(data, 'correlation_matrix', options);
  }

  // 유틸리티 메서드들
  getChartHistory(limit = 10) {
    return this.chartHistory.slice(-limit);
  }

  getSupportedChartTypes() {
    return this.supportedChartTypes;
  }

  getSupportedStyles() {
    return ['default', 'seaborn', 'ggplot', 'bmh', 'fivethirtyeight', 'dark_background'];
  }

  getSupportedFormats() {
    return ['png', 'jpg', 'svg', 'pdf', 'html'];
  }

  getRecommendedChartType(dataInfo) {
    const { numeric_columns, categorical_columns, row_count } = dataInfo;
    
    const recommendations = [];

    // 단일 숫자형 변수
    if (numeric_columns.length === 1 && categorical_columns.length === 0) {
      recommendations.push('histogram', 'box');
    }
    
    // 두 숫자형 변수
    else if (numeric_columns.length === 2) {
      recommendations.push('scatter', 'line');
    }
    
    // 하나의 범주형, 하나의 숫자형
    else if (numeric_columns.length === 1 && categorical_columns.length === 1) {
      recommendations.push('bar', 'box');
    }
    
    // 여러 숫자형 변수
    else if (numeric_columns.length > 2) {
      recommendations.push('correlation_heatmap', 'pairplot');
    }
    
    // 시계열 데이터 (날짜 컬럼이 있는 경우)
    if (dataInfo.date_columns && dataInfo.date_columns.length > 0) {
      recommendations.push('line', 'time_series');
    }

    return recommendations;
  }

  async getChartStatus() {
    return {
      python_executor_status: await this.pythonExecutor.getExecutionStats(),
      chart_history_count: this.chartHistory.length,
      supported_chart_types: this.supportedChartTypes,
      last_chart: this.chartHistory.length > 0 ? 
        this.chartHistory[this.chartHistory.length - 1] : null
    };
  }

  async cleanup() {
    await this.pythonExecutor.shutdown();
    this.logger.info('ChartGenerator 정리 완료');
  }
}