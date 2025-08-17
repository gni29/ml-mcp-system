// tools/visualization/chart-generator.js - 차트 생성 도구
import { PythonExecutor } from '../common/python-executor.js';
import { ResultFormatter } from '../common/result-formatter.js';
import { Logger } from '../../utils/logger.js';
import { ConfigLoader } from '../../utils/config-loader.js';
import { FileManager } from '../common/file-manager.js';

export class ChartGenerator {
  constructor() {
    this.pythonExecutor = new PythonExecutor();
    this.resultFormatter = new ResultFormatter();
    this.logger = new Logger();
    this.configLoader = new ConfigLoader();
    this.fileManager = new FileManager();
    this.chartHistory = [];
    this.chartConfig = null;
    
    this.initializeGenerator();
  }

  async initializeGenerator() {
    try {
      this.chartConfig = await this.configLoader.loadConfig('visualization-config.json');
      this.logger.info('ChartGenerator 초기화 완료');
    } catch (error) {
      this.logger.error('ChartGenerator 초기화 실패:', error);
      this.chartConfig = this.getDefaultConfig();
    }
  }

  getDefaultConfig() {
    return {
      chart_types: {
        '2d': {
          scatter: { script: 'python/visualization/2d/scatter.py' },
          line: { script: 'python/visualization/2d/line.py' },
          bar: { script: 'python/visualization/2d/bar.py' },
          histogram: { script: 'python/visualization/2d/histogram.py' },
          boxplot: { script: 'python/visualization/2d/boxplot.py' }
        },
        '3d': {
          scatter_3d: { script: 'python/visualization/3d/scatter_3d.py' },
          surface: { script: 'python/visualization/3d/surface.py' }
        }
      },
      default_style: 'seaborn',
      output_formats: ['png', 'svg', 'html'],
      default_dpi: 300
    };
  }

  async createScatterPlot(data, options = {}) {
    const {
      x_column,
      y_column,
      color_column = null,
      size_column = null,
      title = null,
      xlabel = null,
      ylabel = null,
      style = 'seaborn',
      figsize = [10, 8],
      alpha = 0.7,
      marker_size = 50,
      color_palette = 'viridis',
      output_format = 'png'
    } = options;

    try {
      this.logger.info('산점도 생성 시작');

      if (!x_column || !y_column) {
        throw new Error('x_column과 y_column이 필요합니다.');
      }

      const scriptPath = this.getChartScript('2d', 'scatter');
      const params = {
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        x_column,
        y_column,
        color_column,
        size_column,
        title: title || `${x_column} vs ${y_column}`,
        xlabel: xlabel || x_column,
        ylabel: ylabel || y_column,
        style,
        figsize: figsize.join(','),
        alpha,
        marker_size,
        color_palette,
        output_format
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const chartResult = JSON.parse(result.output);
        this.saveChartHistory('scatter', chartResult);
        return this.resultFormatter.formatAnalysisResult(chartResult, 'visualization');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('산점도 생성 실패:', error);
      throw error;
    }
  }

  async createLinePlot(data, options = {}) {
    const {
      x_column,
      y_columns,
      title = null,
      xlabel = null,
      ylabel = null,
      style = 'seaborn',
      figsize = [12, 6],
      linewidth = 2,
      marker = 'o',
      color_palette = 'tab10',
      output_format = 'png'
    } = options;

    try {
      this.logger.info('선 그래프 생성 시작');

      if (!x_column || !y_columns || y_columns.length === 0) {
        throw new Error('x_column과 y_columns가 필요합니다.');
      }

      const scriptPath = this.getChartScript('2d', 'line');
      const params = {
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        x_column,
        y_columns: y_columns.join(','),
        title: title || `${y_columns.join(', ')} over ${x_column}`,
        xlabel: xlabel || x_column,
        ylabel: ylabel || 'Value',
        style,
        figsize: figsize.join(','),
        linewidth,
        marker,
        color_palette,
        output_format
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const chartResult = JSON.parse(result.output);
        this.saveChartHistory('line', chartResult);
        return this.resultFormatter.formatAnalysisResult(chartResult, 'visualization');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('선 그래프 생성 실패:', error);
      throw error;
    }
  }

  async createBarChart(data, options = {}) {
    const {
      x_column,
      y_column,
      orientation = 'vertical',
      title = null,
      xlabel = null,
      ylabel = null,
      style = 'seaborn',
      figsize = [10, 6],
      color = 'steelblue',
      output_format = 'png'
    } = options;

    try {
      this.logger.info('막대 그래프 생성 시작');

      if (!x_column || !y_column) {
        throw new Error('x_column과 y_column이 필요합니다.');
      }

      const scriptPath = this.getChartScript('2d', 'bar');
      const params = {
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        x_column,
        y_column,
        orientation,
        title: title || `${y_column} by ${x_column}`,
        xlabel: xlabel || x_column,
        ylabel: ylabel || y_column,
        style,
        figsize: figsize.join(','),
        color,
        output_format
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const chartResult = JSON.parse(result.output);
        this.saveChartHistory('bar', chartResult);
        return this.resultFormatter.formatAnalysisResult(chartResult, 'visualization');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('막대 그래프 생성 실패:', error);
      throw error;
    }
  }

  async createHistogram(data, options = {}) {
    const {
      column,
      bins = 30,
      title = null,
      xlabel = null,
      ylabel = 'Frequency',
      style = 'seaborn',
      figsize = [10, 6],
      color = 'skyblue',
      alpha = 0.7,
      kde = true,
      output_format = 'png'
    } = options;

    try {
      this.logger.info('히스토그램 생성 시작');

      if (!column) {
        throw new Error('column이 필요합니다.');
      }

      const scriptPath = this.getChartScript('2d', 'histogram');
      const params = {
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        column,
        bins,
        title: title || `Distribution of ${column}`,
        xlabel: xlabel || column,
        ylabel,
        style,
        figsize: figsize.join(','),
        color,
        alpha,
        kde,
        output_format
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const chartResult = JSON.parse(result.output);
        this.saveChartHistory('histogram', chartResult);
        return this.resultFormatter.formatAnalysisResult(chartResult, 'visualization');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('히스토그램 생성 실패:', error);
      throw error;
    }
  }

  async createBoxPlot(data, options = {}) {
    const {
      y_column,
      x_column = null,
      title = null,
      xlabel = null,
      ylabel = null,
      style = 'seaborn',
      figsize = [10, 6],
      output_format = 'png'
    } = options;

    try {
      this.logger.info('박스플롯 생성 시작');

      if (!y_column) {
        throw new Error('y_column이 필요합니다.');
      }

      const scriptPath = this.getChartScript('2d', 'boxplot');
      const params = {
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        y_column,
        x_column,
        title: title || `Box Plot of ${y_column}${x_column ? ` by ${x_column}` : ''}`,
        xlabel: xlabel || x_column || '',
        ylabel: ylabel || y_column,
        style,
        figsize: figsize.join(','),
        output_format
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const chartResult = JSON.parse(result.output);
        this.saveChartHistory('boxplot', chartResult);
        return this.resultFormatter.formatAnalysisResult(chartResult, 'visualization');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('박스플롯 생성 실패:', error);
      throw error;
    }
  }

  async createHeatmap(data, options = {}) {
    const {
      title = 'Heatmap',
      cmap = 'viridis',
      annot = true,
      fmt = '.2f',
      style = 'seaborn',
      figsize = [10, 8],
      output_format = 'png'
    } = options;

    try {
      this.logger.info('히트맵 생성 시작');

      const scriptPath = 'python/visualization/heatmap.py';
      const params = {
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        title,
        cmap,
        annot,
        fmt,
        style,
        figsize: figsize.join(','),
        output_format
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const chartResult = JSON.parse(result.output);
        this.saveChartHistory('heatmap', chartResult);
        return this.resultFormatter.formatAnalysisResult(chartResult, 'visualization');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('히트맵 생성 실패:', error);
      throw error;
    }
  }

  async create3DScatterPlot(data, options = {}) {
    const {
      x_column,
      y_column,
      z_column,
      color_column = null,
      title = null,
      xlabel = null,
      ylabel = null,
      zlabel = null,
      style = 'seaborn',
      figsize = [12, 9],
      alpha = 0.7,
      marker_size = 50,
      output_format = 'png'
    } = options;

    try {
      this.logger.info('3D 산점도 생성 시작');

      if (!x_column || !y_column || !z_column) {
        throw new Error('x_column, y_column, z_column이 모두 필요합니다.');
      }

      const scriptPath = this.getChartScript('3d', 'scatter_3d');
      const params = {
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        x_column,
        y_column,
        z_column,
        color_column,
        title: title || `3D Scatter: ${x_column}, ${y_column}, ${z_column}`,
        xlabel: xlabel || x_column,
        ylabel: ylabel || y_column,
        zlabel: zlabel || z_column,
        style,
        figsize: figsize.join(','),
        alpha,
        marker_size,
        output_format
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const chartResult = JSON.parse(result.output);
        this.saveChartHistory('scatter_3d', chartResult);
        return this.resultFormatter.formatAnalysisResult(chartResult, 'visualization');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('3D 산점도 생성 실패:', error);
      throw error;
    }
  }

  async createInteractivePlot(data, options = {}) {
    const {
      chart_type = 'scatter',
      x_column,
      y_column,
      color_column = null,
      title = null,
      library = 'plotly',
      output_format = 'html'
    } = options;

    try {
      this.logger.info('인터랙티브 플롯 생성 시작');

      const scriptPath = 'python/visualization/interactive_plots.py';
      const params = {
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        chart_type,
        x_column,
        y_column,
        color_column,
        title: title || `Interactive ${chart_type}`,
        library,
        output_format
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const chartResult = JSON.parse(result.output);
        this.saveChartHistory('interactive', chartResult);
        return this.resultFormatter.formatAnalysisResult(chartResult, 'visualization');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('인터랙티브 플롯 생성 실패:', error);
      throw error;
    }
  }

  async createMultipleCharts(data, chartConfigs, options = {}) {
    const {
      layout = 'grid',
      rows = 2,
      cols = 2,
      figsize = [16, 12],
      title = 'Multiple Charts',
      output_format = 'png'
    } = options;

    try {
      this.logger.info(`다중 차트 생성 시작: ${chartConfigs.length}개 차트`);

      const scriptPath = 'python/visualization/multiple_charts.py';
      const params = {
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        chart_configs: JSON.stringify(chartConfigs),
        layout,
        rows,
        cols,
        figsize: figsize.join(','),
        title,
        output_format
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const chartResult = JSON.parse(result.output);
        this.saveChartHistory('multiple', chartResult);
        return this.resultFormatter.formatAnalysisResult(chartResult, 'visualization');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('다중 차트 생성 실패:', error);
      throw error;
    }
  }

  async createStatisticalPlots(data, options = {}) {
    const {
      plot_types = ['distribution', 'correlation', 'regression'],
      columns = null,
      style = 'seaborn',
      figsize = [15, 10],
      output_format = 'png'
    } = options;

    try {
      this.logger.info('통계 플롯 생성 시작');

      const scriptPath = 'python/visualization/statistical_plots.py';
      const params = {
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        plot_types: plot_types.join(','),
        columns: columns ? columns.join(',') : null,
        style,
        figsize: figsize.join(','),
        output_format
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const chartResult = JSON.parse(result.output);
        this.saveChartHistory('statistical', chartResult);
        return this.resultFormatter.formatAnalysisResult(chartResult, 'visualization');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('통계 플롯 생성 실패:', error);
      throw error;
    }
  }

  async autoGenerateCharts(data, options = {}) {
    const {
      max_charts = 6,
      include_correlation = true,
      include_distribution = true,
      include_relationships = true,
      target_column = null
    } = options;

    try {
      this.logger.info('자동 차트 생성 시작');

      const scriptPath = 'python/visualization/auto_charts.py';
      const params = {
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        max_charts,
        include_correlation,
        include_distribution,
        include_relationships,
        target_column
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const chartResult = JSON.parse(result.output);
        this.saveChartHistory('auto_generated', chartResult);
        return this.resultFormatter.formatAnalysisResult(chartResult, 'visualization');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('자동 차트 생성 실패:', error);
      throw error;
    }
  }

  async createCustomChart(data, customCode, options = {}) {
    const {
      title = 'Custom Chart',
      figsize = [10, 6],
      style = 'seaborn',
      output_format = 'png'
    } = options;

    try {
      this.logger.info('커스텀 차트 생성 시작');

      const scriptPath = 'python/visualization/custom_chart.py';
      const params = {
        data_path: typeof data === 'string' ? data : null,
        data_json: typeof data === 'object' ? JSON.stringify(data) : null,
        custom_code: customCode,
        title,
        figsize: figsize.join(','),
        style,
        output_format
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const chartResult = JSON.parse(result.output);
        this.saveChartHistory('custom', chartResult);
        return this.resultFormatter.formatAnalysisResult(chartResult, 'visualization');
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('커스텀 차트 생성 실패:', error);
      throw error;
    }
  }

  getChartScript(category, chartType) {
    if (this.chartConfig?.chart_types?.[category]?.[chartType]?.script) {
      return this.chartConfig.chart_types[category][chartType].script;
    }
    
    return `python/visualization/${category}/${chartType}.py`;
  }

  saveChartHistory(chartType, chartResult) {
    try {
      const historyEntry = {
        timestamp: new Date().toISOString(),
        chart_type: chartType,
        output_path: chartResult.results?.chart_path || null,
        parameters: chartResult.parameters || {},
        execution_time: chartResult.metadata?.execution_time || null
      };

      this.chartHistory.push(historyEntry);

      // 히스토리 크기 제한 (최근 200개만 유지)
      if (this.chartHistory.length > 200) {
        this.chartHistory = this.chartHistory.slice(-100);
      }

      this.logger.debug('차트 히스토리 저장 완료');
    } catch (error) {
      this.logger.warn('차트 히스토리 저장 실패:', error);
    }
  }

  getChartHistory(limit = 50) {
    return this.chartHistory.slice(-limit);
  }

  getAvailableChartTypes() {
    return {
      '2d': {
        scatter: 'Scatter Plot - 산점도',
        line: 'Line Plot - 선 그래프',
        bar: 'Bar Chart - 막대 그래프',
        histogram: 'Histogram - 히스토그램',
        boxplot: 'Box Plot - 박스플롯',
        heatmap: 'Heatmap - 히트맵'
      },
      '3d': {
        scatter_3d: '3D Scatter Plot - 3D 산점도',
        surface: 'Surface Plot - 표면 그래프'
      },
      interactive: {
        plotly: 'Plotly Interactive Charts',
        bokeh: 'Bokeh Interactive Charts'
      },
      statistical: {
        distribution: 'Distribution Plots',
        correlation: 'Correlation Plots',
        regression: 'Regression Plots'
      }
    };
  }

  getAvailableStyles() {
    return [
      'seaborn',
      'ggplot',
      'fivethirtyeight',
      'bmh',
      'dark_background',
      'classic',
      'seaborn-whitegrid',
      'seaborn-darkgrid'
    ];
  }

  getAvailableOutputFormats() {
    return ['png', 'jpg', 'svg', 'pdf', 'html', 'json'];
  }

  getPerformanceMetrics() {
    const totalCharts = this.chartHistory.length;
    const recentCharts = this.chartHistory.slice(-10);
    
    const avgExecutionTime = recentCharts.length > 0 
      ? recentCharts.reduce((sum, c) => sum + (c.execution_time || 0), 0) / recentCharts.length 
      : 0;

    const chartTypes = [...new Set(this.chartHistory.map(c => c.chart_type))];

    return {
      total_charts_generated: totalCharts,
      unique_chart_types: chartTypes.length,
      average_execution_time: avgExecutionTime,
      most_common_chart_type: this.getMostCommonChartType()
    };
  }

  getMostCommonChartType() {
    if (this.chartHistory.length === 0) return 'none';
    
    const typeCounts = {};
    this.chartHistory.forEach(c => {
      typeCounts[c.chart_type] = (typeCounts[c.chart_type] || 0) + 1;
    });

    return Object.keys(typeCounts).reduce((a, b) => typeCounts[a] > typeCounts[b] ? a : b);
  }
}