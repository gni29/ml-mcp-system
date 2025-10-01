/**
 * Visualization Service - Handles data visualization operations
 * Provides tools for creating charts, plots, and interactive visualizations
 */

import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class VisualizationService {
  constructor(logger) {
    this.logger = logger;
    this.name = 'visualization-service';
    this.type = 'visualization';
    this.version = '2.0.0';
    this.capabilities = ['tools'];
    this.isInitialized = false;
    this.plotHistory = [];
  }

  /**
   * Initialize the visualization service
   */
  async initialize() {
    try {
      this.logger.info('시각화 서비스 초기화 중');

      // Test visualization environment
      await this.testVisualizationEnvironment();

      this.isInitialized = true;
      this.logger.info('시각화 서비스 초기화 완료');

    } catch (error) {
      this.logger.error('시각화 서비스 초기화 실패:', error);
      throw error;
    }
  }

  /**
   * Test visualization environment
   */
  async testVisualizationEnvironment() {
    return new Promise((resolve, reject) => {
      const pythonProcess = spawn('python', ['-c', 'import matplotlib, seaborn, plotly; print("Visualization environment OK")']);

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          resolve();
        } else {
          this.logger.warn('일부 시각화 라이브러리를 사용할 수 없을 수 있습니다');
          resolve(); // Don't fail initialization if optional libraries are missing
        }
      });

      pythonProcess.on('error', (error) => {
        reject(new Error(`Visualization environment test error: ${error.message}`));
      });
    });
  }

  /**
   * Get available tools
   */
  async getTools() {
    return [
      {
        name: 'visualize_data',
        description: '데이터를 시각화하여 차트나 그래프를 생성합니다.',
        inputSchema: {
          type: 'object',
          properties: {
            query: {
              type: 'string',
              description: '시각화 요청 내용'
            },
            file_path: {
              type: 'string',
              description: '시각화할 데이터 파일 경로'
            },
            chart_type: {
              type: 'string',
              description: '차트 유형',
              enum: ['bar', 'line', 'scatter', 'histogram', 'heatmap', 'box', 'violin', 'auto'],
              default: 'auto'
            },
            columns: {
              type: 'array',
              description: '시각화할 열 이름들',
              items: {
                type: 'string'
              }
            },
            interactive: {
              type: 'boolean',
              description: '인터랙티브 차트 생성 여부',
              default: false
            }
          },
          required: ['query']
        }
      },
      {
        name: 'create_dashboard',
        description: '여러 차트를 포함한 대시보드를 생성합니다.',
        inputSchema: {
          type: 'object',
          properties: {
            file_path: {
              type: 'string',
              description: '데이터 파일 경로'
            },
            dashboard_type: {
              type: 'string',
              description: '대시보드 유형',
              enum: ['overview', 'statistical', 'correlation', 'distribution'],
              default: 'overview'
            },
            output_format: {
              type: 'string',
              description: '출력 형식',
              enum: ['html', 'png', 'pdf'],
              default: 'html'
            },
            title: {
              type: 'string',
              description: '대시보드 제목'
            }
          },
          required: ['file_path']
        }
      },
      {
        name: 'correlation_heatmap',
        description: '상관관계 히트맵을 생성합니다.',
        inputSchema: {
          type: 'object',
          properties: {
            file_path: {
              type: 'string',
              description: '데이터 파일 경로'
            },
            method: {
              type: 'string',
              description: '상관관계 계산 방법',
              enum: ['pearson', 'spearman', 'kendall'],
              default: 'pearson'
            },
            figsize: {
              type: 'array',
              description: '그림 크기 [width, height]',
              items: {
                type: 'number'
              },
              default: [10, 8]
            }
          },
          required: ['file_path']
        }
      },
      {
        name: 'distribution_plots',
        description: '데이터 분포 플롯을 생성합니다.',
        inputSchema: {
          type: 'object',
          properties: {
            file_path: {
              type: 'string',
              description: '데이터 파일 경로'
            },
            columns: {
              type: 'array',
              description: '분포를 확인할 열 이름들',
              items: {
                type: 'string'
              }
            },
            plot_type: {
              type: 'string',
              description: '분포 플롯 유형',
              enum: ['histogram', 'density', 'box', 'violin', 'all'],
              default: 'histogram'
            }
          },
          required: ['file_path']
        }
      },
      {
        name: 'scatter_plot_matrix',
        description: '산점도 매트릭스를 생성합니다.',
        inputSchema: {
          type: 'object',
          properties: {
            file_path: {
              type: 'string',
              description: '데이터 파일 경로'
            },
            columns: {
              type: 'array',
              description: '포함할 열 이름들',
              items: {
                type: 'string'
              }
            },
            color_by: {
              type: 'string',
              description: '색상 구분할 열 이름'
            }
          },
          required: ['file_path']
        }
      },
      {
        name: 'time_series_plot',
        description: '시계열 데이터 플롯을 생성합니다.',
        inputSchema: {
          type: 'object',
          properties: {
            file_path: {
              type: 'string',
              description: '데이터 파일 경로'
            },
            time_column: {
              type: 'string',
              description: '시간 열 이름'
            },
            value_columns: {
              type: 'array',
              description: '값 열 이름들',
              items: {
                type: 'string'
              }
            },
            resample: {
              type: 'string',
              description: '리샘플링 주기 (D, W, M 등)'
            }
          },
          required: ['file_path', 'time_column']
        }
      }
    ];
  }

  /**
   * Execute a tool
   */
  async executeTool(toolName, args) {
    if (!this.isInitialized) {
      throw new Error('Visualization service not initialized');
    }

    this.logger.info(`시각화 도구 실행 중: ${toolName}`, args);

    switch (toolName) {
      case 'visualize_data':
        return await this.handleVisualizeData(args);
      case 'create_dashboard':
        return await this.handleCreateDashboard(args);
      case 'correlation_heatmap':
        return await this.handleCorrelationHeatmap(args);
      case 'distribution_plots':
        return await this.handleDistributionPlots(args);
      case 'scatter_plot_matrix':
        return await this.handleScatterPlotMatrix(args);
      case 'time_series_plot':
        return await this.handleTimeSeriesPlot(args);
      default:
        throw new Error(`Unknown visualization tool: ${toolName}`);
    }
  }

  /**
   * Handle general data visualization
   */
  async handleVisualizeData(args) {
    const { query, file_path, chart_type = 'auto', columns, interactive = false } = args;

    try {
      let targetFile = file_path;
      if (!targetFile) {
        targetFile = await this.autoDetectDataFile();
      }

      if (!targetFile) {
        return {
          content: [{
            type: 'text',
            text: `**시각화 요청: ${query}**\n\n` +
                  `시각화할 데이터 파일을 지정해주세요. \`file_path\` 파라미터를 사용하거나 ` +
                  `데이터 파일을 프로젝트 디렉토리에 배치하세요.`
          }]
        };
      }

      const result = await this.runVisualizationScript('general', {
        data: targetFile,
        chart_type,
        columns,
        interactive,
        query
      });

      // Store plot in history
      this.plotHistory.push({
        timestamp: new Date().toISOString(),
        query,
        file_path: targetFile,
        chart_type,
        result
      });

      return {
        content: [{
          type: 'text',
          text: `**시각화 완료**\n\n` +
                `**요청:** ${query}\n` +
                `**차트 유형:** ${chart_type}\n` +
                `**데이터:** ${targetFile}\n` +
                `**인터랙티브:** ${interactive ? '예' : '아니오'}\n\n` +
                `**결과:**\n${JSON.stringify(result, null, 2)}`
        }]
      };

    } catch (error) {
      throw new Error(`Data visualization failed: ${error.message}`);
    }
  }

  /**
   * Handle dashboard creation
   */
  async handleCreateDashboard(args) {
    const { file_path, dashboard_type = 'overview', output_format = 'html', title } = args;

    try {
      const result = await this.runVisualizationScript('dashboard', {
        data: file_path,
        type: dashboard_type,
        format: output_format,
        title
      });

      return {
        content: [{
          type: 'text',
          text: `**대시보드 생성 완료**\n\n` +
                `**유형:** ${dashboard_type}\n` +
                `**데이터:** ${file_path}\n` +
                `**출력 형식:** ${output_format}\n` +
                `**제목:** ${title || 'Auto-generated'}\n\n` +
                `**결과:**\n${JSON.stringify(result, null, 2)}`
        }]
      };

    } catch (error) {
      throw new Error(`Dashboard creation failed: ${error.message}`);
    }
  }

  /**
   * Handle correlation heatmap
   */
  async handleCorrelationHeatmap(args) {
    const { file_path, method = 'pearson', figsize = [10, 8] } = args;

    try {
      const result = await this.runVisualizationScript('heatmap', {
        data: file_path,
        method,
        figsize
      });

      return {
        content: [{
          type: 'text',
          text: `**상관관계 히트맵 생성 완료**\n\n` +
                `**방법:** ${method}\n` +
                `**데이터:** ${file_path}\n` +
                `**크기:** ${figsize.join(' x ')}\n\n` +
                `**결과:**\n${JSON.stringify(result, null, 2)}`
        }]
      };

    } catch (error) {
      throw new Error(`Correlation heatmap failed: ${error.message}`);
    }
  }

  /**
   * Handle distribution plots
   */
  async handleDistributionPlots(args) {
    const { file_path, columns, plot_type = 'histogram' } = args;

    try {
      const result = await this.runVisualizationScript('distribution', {
        data: file_path,
        columns,
        plot_type
      });

      return {
        content: [{
          type: 'text',
          text: `**분포 플롯 생성 완료**\n\n` +
                `**플롯 유형:** ${plot_type}\n` +
                `**데이터:** ${file_path}\n` +
                `**컬럼:** ${columns ? columns.join(', ') : 'auto-detected'}\n\n` +
                `**결과:**\n${JSON.stringify(result, null, 2)}`
        }]
      };

    } catch (error) {
      throw new Error(`Distribution plots failed: ${error.message}`);
    }
  }

  /**
   * Handle scatter plot matrix
   */
  async handleScatterPlotMatrix(args) {
    const { file_path, columns, color_by } = args;

    try {
      const result = await this.runVisualizationScript('scatter_matrix', {
        data: file_path,
        columns,
        color_by
      });

      return {
        content: [{
          type: 'text',
          text: `**산점도 매트릭스 생성 완료**\n\n` +
                `**데이터:** ${file_path}\n` +
                `**컬럼:** ${columns ? columns.join(', ') : 'auto-detected'}\n` +
                `**색상 구분:** ${color_by || 'none'}\n\n` +
                `**결과:**\n${JSON.stringify(result, null, 2)}`
        }]
      };

    } catch (error) {
      throw new Error(`Scatter plot matrix failed: ${error.message}`);
    }
  }

  /**
   * Handle time series plot
   */
  async handleTimeSeriesPlot(args) {
    const { file_path, time_column, value_columns, resample } = args;

    try {
      const result = await this.runVisualizationScript('timeseries', {
        data: file_path,
        time_column,
        value_columns,
        resample
      });

      return {
        content: [{
          type: 'text',
          text: `**시계열 플롯 생성 완료**\n\n` +
                `**데이터:** ${file_path}\n` +
                `**시간 컬럼:** ${time_column}\n` +
                `**값 컬럼:** ${value_columns ? value_columns.join(', ') : 'auto-detected'}\n` +
                `**리샘플링:** ${resample || 'none'}\n\n` +
                `**결과:**\n${JSON.stringify(result, null, 2)}`
        }]
      };

    } catch (error) {
      throw new Error(`Time series plot failed: ${error.message}`);
    }
  }

  /**
   * Run visualization Python script
   */
  async runVisualizationScript(operation, options = {}) {
    const scriptPath = path.join(__dirname, '..', 'python', 'visualization', 'auto_visualizer.py');
    const args = [scriptPath, operation];

    // Add options as JSON
    const optionsJson = JSON.stringify(options);

    return new Promise((resolve, reject) => {
      const process = spawn('python', args);

      let stdout = '';
      let stderr = '';

      process.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      process.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      // Send options via stdin
      process.stdin.write(optionsJson);
      process.stdin.end();

      process.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdout);
            resolve(result);
          } catch (e) {
            resolve({ output: stdout, raw: true });
          }
        } else {
          reject(new Error(`Visualization script failed (exit code: ${code})\n${stderr}`));
        }
      });

      process.on('error', (error) => {
        reject(new Error(`Visualization process error: ${error.message}`));
      });

      // Set timeout
      setTimeout(() => {
        process.kill('SIGKILL');
        reject(new Error('Visualization script execution timeout'));
      }, 300000); // 5 minutes
    });
  }

  /**
   * Auto-detect data files in the project
   */
  async autoDetectDataFile() {
    const fs = await import('fs/promises');
    const dataExtensions = ['.csv', '.json', '.xlsx', '.xls', '.parquet'];
    const searchDirs = ['data', 'datasets', '.'];

    for (const dir of searchDirs) {
      try {
        const files = await fs.readdir(dir);
        for (const file of files) {
          const ext = path.extname(file).toLowerCase();
          if (dataExtensions.includes(ext)) {
            return path.join(dir, file);
          }
        }
      } catch (error) {
        // Directory doesn't exist, continue
      }
    }

    return null;
  }

  /**
   * Get plot history
   */
  getPlotHistory(limit = 10) {
    return this.plotHistory.slice(-limit).reverse();
  }

  /**
   * Clear plot history
   */
  clearPlotHistory() {
    this.plotHistory = [];
    return { success: true, message: 'Plot history cleared' };
  }

  /**
   * Health check
   */
  isHealthy() {
    return this.isInitialized;
  }

  /**
   * Get service status
   */
  getStatus() {
    return {
      name: this.name,
      type: this.type,
      version: this.version,
      initialized: this.isInitialized,
      healthy: this.isHealthy(),
      toolCount: 6,
      plotHistoryCount: this.plotHistory.length,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Cleanup
   */
  async cleanup() {
    this.logger.info('시각화 서비스 정리 중');
    this.plotHistory = [];
    this.isInitialized = false;
  }
}

export default VisualizationService;