/**
 * Visualization Service for Visualization MCP
 * 시각화 MCP용 시각화 서비스 - 고급 차트 및 플롯 생성에 특화
 */

import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import { BaseService } from 'ml-mcp-shared/utils/base-service';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class VisualizationService extends BaseService {
  constructor(logger) {
    super('visualization-service', 'visualization', '1.0.0');
    this.logger = logger;
    this.capabilities = ['tools'];
    this.outputCache = new Map(); // Cache for generated visualizations
  }

  /**
   * Initialize the visualization service
   */
  async initialize() {
    try {
      this.logger.info('📊 시각화 서비스 초기화 중');

      // Test visualization environment
      await this.testVisualizationEnvironment();

      await super.initialize();
      this.logger.info('✅ 시각화 서비스 초기화 완료');

    } catch (error) {
      this.logger.error('❌ 시각화 서비스 초기화 실패:', error);
      throw error;
    }
  }

  /**
   * Test visualization environment
   */
  async testVisualizationEnvironment() {
    return new Promise((resolve, reject) => {
      const pythonProcess = spawn('python', ['-c',
        'import matplotlib, seaborn, pandas, numpy; print("Visualization environment OK")'
      ]);

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error('시각화 라이브러리 (matplotlib, seaborn, pandas, numpy)를 사용할 수 없습니다'));
        }
      });

      pythonProcess.on('error', (error) => {
        reject(new Error(`Python 시각화 환경 테스트 오류: ${error.message}`));
      });
    });
  }

  /**
   * Get available visualization tools
   */
  async getTools() {
    return [
      {
        name: 'create_distribution_plots',
        description: '데이터 분포를 시각화합니다 (히스토그램, 박스플롯, 바이올린 플롯 등)',
        inputSchema: {
          type: 'object',
          properties: {
            data_file: {
              type: 'string',
              description: '시각화할 데이터 파일 경로'
            },
            columns: {
              type: 'array',
              description: '시각화할 컬럼들 (미지정시 모든 수치형 컬럼)',
              items: { type: 'string' }
            },
            plot_types: {
              type: 'array',
              description: '생성할 플롯 유형들',
              items: {
                type: 'string',
                enum: ['histogram', 'boxplot', 'violin', 'density', 'qq']
              },
              default: ['histogram', 'boxplot']
            },
            output_dir: {
              type: 'string',
              description: '출력 디렉토리',
              default: 'visualizations'
            }
          },
          required: ['data_file']
        }
      },
      {
        name: 'create_correlation_heatmap',
        description: '상관관계 히트맵을 생성합니다',
        inputSchema: {
          type: 'object',
          properties: {
            data_file: {
              type: 'string',
              description: '데이터 파일 경로'
            },
            method: {
              type: 'string',
              description: '상관계수 계산 방법',
              enum: ['pearson', 'spearman', 'kendall'],
              default: 'pearson'
            },
            figure_size: {
              type: 'array',
              description: '그림 크기 [가로, 세로]',
              items: { type: 'number' },
              default: [12, 10]
            },
            color_scheme: {
              type: 'string',
              description: '색상 스키마',
              enum: ['coolwarm', 'viridis', 'plasma', 'RdYlBu'],
              default: 'coolwarm'
            },
            output_file: {
              type: 'string',
              description: '출력 파일명',
              default: 'correlation_heatmap.png'
            }
          },
          required: ['data_file']
        }
      },
      {
        name: 'create_scatter_plots',
        description: '산점도 매트릭스 및 개별 산점도를 생성합니다',
        inputSchema: {
          type: 'object',
          properties: {
            data_file: {
              type: 'string',
              description: '데이터 파일 경로'
            },
            x_column: {
              type: 'string',
              description: 'X축 컬럼 (개별 산점도용)'
            },
            y_column: {
              type: 'string',
              description: 'Y축 컬럼 (개별 산점도용)'
            },
            color_column: {
              type: 'string',
              description: '색상 구분 컬럼 (선택사항)'
            },
            create_matrix: {
              type: 'boolean',
              description: '산점도 매트릭스 생성 여부',
              default: true
            },
            add_trendline: {
              type: 'boolean',
              description: '추세선 추가 여부',
              default: true
            },
            output_dir: {
              type: 'string',
              description: '출력 디렉토리',
              default: 'visualizations'
            }
          },
          required: ['data_file']
        }
      },
      {
        name: 'create_time_series_plots',
        description: '시계열 데이터 시각화를 생성합니다',
        inputSchema: {
          type: 'object',
          properties: {
            data_file: {
              type: 'string',
              description: '시계열 데이터 파일 경로'
            },
            date_column: {
              type: 'string',
              description: '날짜/시간 컬럼명'
            },
            value_columns: {
              type: 'array',
              description: '시각화할 값 컬럼들',
              items: { type: 'string' }
            },
            plot_types: {
              type: 'array',
              description: '생성할 플롯 유형들',
              items: {
                type: 'string',
                enum: ['line', 'area', 'seasonal_decompose', 'rolling_stats', 'autocorrelation']
              },
              default: ['line', 'seasonal_decompose']
            },
            rolling_window: {
              type: 'number',
              description: '이동평균 윈도우 크기',
              default: 30
            },
            output_dir: {
              type: 'string',
              description: '출력 디렉토리',
              default: 'visualizations'
            }
          },
          required: ['data_file', 'date_column', 'value_columns']
        }
      },
      {
        name: 'create_categorical_plots',
        description: '범주형 데이터 시각화를 생성합니다',
        inputSchema: {
          type: 'object',
          properties: {
            data_file: {
              type: 'string',
              description: '데이터 파일 경로'
            },
            categorical_columns: {
              type: 'array',
              description: '범주형 컬럼들',
              items: { type: 'string' }
            },
            numeric_column: {
              type: 'string',
              description: '분석할 수치형 컬럼 (선택사항)'
            },
            plot_types: {
              type: 'array',
              description: '생성할 플롯 유형들',
              items: {
                type: 'string',
                enum: ['countplot', 'barplot', 'pieplot', 'treemap', 'sunburst']
              },
              default: ['countplot', 'barplot']
            },
            max_categories: {
              type: 'number',
              description: '표시할 최대 범주 수',
              default: 20
            },
            output_dir: {
              type: 'string',
              description: '출력 디렉토리',
              default: 'visualizations'
            }
          },
          required: ['data_file', 'categorical_columns']
        }
      },
      {
        name: 'create_statistical_plots',
        description: '통계적 시각화를 생성합니다 (회귀선, 신뢰구간 등)',
        inputSchema: {
          type: 'object',
          properties: {
            data_file: {
              type: 'string',
              description: '데이터 파일 경로'
            },
            x_column: {
              type: 'string',
              description: 'X축 변수'
            },
            y_column: {
              type: 'string',
              description: 'Y축 변수'
            },
            group_column: {
              type: 'string',
              description: '그룹 변수 (선택사항)'
            },
            plot_types: {
              type: 'array',
              description: '생성할 통계 플롯들',
              items: {
                type: 'string',
                enum: ['regplot', 'residplot', 'lmplot', 'jointplot', 'pairplot']
              },
              default: ['regplot', 'residplot']
            },
            confidence_interval: {
              type: 'number',
              description: '신뢰구간 수준',
              default: 95,
              minimum: 90,
              maximum: 99
            },
            output_dir: {
              type: 'string',
              description: '출력 디렉토리',
              default: 'visualizations'
            }
          },
          required: ['data_file', 'x_column', 'y_column']
        }
      },
      {
        name: 'create_interactive_plots',
        description: '인터랙티브 시각화를 생성합니다 (Plotly 기반)',
        inputSchema: {
          type: 'object',
          properties: {
            data_file: {
              type: 'string',
              description: '데이터 파일 경로'
            },
            plot_type: {
              type: 'string',
              description: '인터랙티브 플롯 유형',
              enum: ['scatter_3d', 'surface', 'heatmap_interactive', 'parallel_coordinates', 'sankey', 'treemap_interactive'],
              default: 'scatter_3d'
            },
            x_column: {
              type: 'string',
              description: 'X축 컬럼'
            },
            y_column: {
              type: 'string',
              description: 'Y축 컬럼'
            },
            z_column: {
              type: 'string',
              description: 'Z축 컬럼 (3D 플롯용)'
            },
            color_column: {
              type: 'string',
              description: '색상 컬럼'
            },
            output_file: {
              type: 'string',
              description: '출력 HTML 파일명',
              default: 'interactive_plot.html'
            }
          },
          required: ['data_file', 'plot_type']
        }
      },
      {
        name: 'create_dashboard',
        description: '종합 대시보드를 생성합니다',
        inputSchema: {
          type: 'object',
          properties: {
            data_file: {
              type: 'string',
              description: '데이터 파일 경로'
            },
            dashboard_type: {
              type: 'string',
              description: '대시보드 유형',
              enum: ['overview', 'statistical', 'exploratory', 'custom'],
              default: 'overview'
            },
            include_sections: {
              type: 'array',
              description: '포함할 섹션들',
              items: {
                type: 'string',
                enum: ['summary_stats', 'distributions', 'correlations', 'missing_data', 'outliers', 'time_series']
              },
              default: ['summary_stats', 'distributions', 'correlations']
            },
            target_column: {
              type: 'string',
              description: '타겟 변수 (분석 중심)'
            },
            output_file: {
              type: 'string',
              description: '출력 대시보드 파일명',
              default: 'data_dashboard.html'
            }
          },
          required: ['data_file']
        }
      }
    ];
  }

  /**
   * Execute visualization tool
   */
  async executeTool(toolName, args) {
    if (!this.isInitialized) {
      throw new Error('시각화 서비스가 초기화되지 않았습니다');
    }

    this.logger.info(`시각화 도구 실행 중: ${toolName}`, args);

    switch (toolName) {
      case 'create_distribution_plots':
        return await this.handleCreateDistributionPlots(args);
      case 'create_correlation_heatmap':
        return await this.handleCreateCorrelationHeatmap(args);
      case 'create_scatter_plots':
        return await this.handleCreateScatterPlots(args);
      case 'create_time_series_plots':
        return await this.handleCreateTimeSeriesPlots(args);
      case 'create_categorical_plots':
        return await this.handleCreateCategoricalPlots(args);
      case 'create_statistical_plots':
        return await this.handleCreateStatisticalPlots(args);
      case 'create_interactive_plots':
        return await this.handleCreateInteractivePlots(args);
      case 'create_dashboard':
        return await this.handleCreateDashboard(args);
      default:
        throw new Error(`알 수 없는 시각화 도구: ${toolName}`);
    }
  }

  /**
   * Handle distribution plots creation
   */
  async handleCreateDistributionPlots(args) {
    const { data_file, columns, plot_types = ['histogram', 'boxplot'], output_dir = 'visualizations' } = args;

    try {
      const result = await this.runVisualizationScript('distribution_plots', {
        data_file,
        columns,
        plot_types,
        output_dir
      });

      return {
        content: [{
          type: 'text',
          text: `**분포 시각화 생성 완료**\\n\\n` +
                `**데이터:** ${data_file}\\n` +
                `**생성된 플롯:** ${plot_types.join(', ')}\\n` +
                `**분석된 컬럼 수:** ${result.analyzed_columns || 'N/A'}\\n` +
                `**출력 디렉토리:** ${output_dir}\\n\\n` +
                `**생성된 파일들:**\\n` +
                `${result.generated_files?.map(f => `• ${f}`).join('\\n') || '파일 목록을 불러올 수 없습니다'}\\n\\n` +
                `분포 시각화가 완료되었습니다.`
        }]
      };
    } catch (error) {
      throw new Error(`분포 시각화 생성 실패: ${error.message}`);
    }
  }

  /**
   * Handle correlation heatmap creation
   */
  async handleCreateCorrelationHeatmap(args) {
    const { data_file, method = 'pearson', figure_size = [12, 10], color_scheme = 'coolwarm', output_file = 'correlation_heatmap.png' } = args;

    try {
      const result = await this.runVisualizationScript('correlation_heatmap', {
        data_file,
        method,
        figure_size,
        color_scheme,
        output_file
      });

      return {
        content: [{
          type: 'text',
          text: `**상관관계 히트맵 생성 완료**\\n\\n` +
                `**데이터:** ${data_file}\\n` +
                `**상관계수 방법:** ${method}\\n` +
                `**분석된 변수 수:** ${result.variable_count || 'N/A'}\\n` +
                `**출력 파일:** ${output_file}\\n\\n` +
                `**주요 상관관계:**\\n` +
                `• 최고 상관계수: ${result.max_correlation || 'N/A'}\\n` +
                `• 강한 상관관계 수: ${result.strong_correlations || 'N/A'}\\n\\n` +
                `상관관계 히트맵이 생성되었습니다.`
        }]
      };
    } catch (error) {
      throw new Error(`상관관계 히트맵 생성 실패: ${error.message}`);
    }
  }

  /**
   * Handle scatter plots creation
   */
  async handleCreateScatterPlots(args) {
    const { data_file, x_column, y_column, color_column, create_matrix = true, add_trendline = true, output_dir = 'visualizations' } = args;

    try {
      const result = await this.runVisualizationScript('scatter_plots', {
        data_file,
        x_column,
        y_column,
        color_column,
        create_matrix,
        add_trendline,
        output_dir
      });

      return {
        content: [{
          type: 'text',
          text: `**산점도 생성 완료**\\n\\n` +
                `**데이터:** ${data_file}\\n` +
                `${x_column && y_column ? `**축:** ${x_column} vs ${y_column}\\n` : ''}` +
                `**색상 구분:** ${color_column || '없음'}\\n` +
                `**매트릭스 생성:** ${create_matrix ? '예' : '아니오'}\\n` +
                `**추세선:** ${add_trendline ? '포함' : '미포함'}\\n\\n` +
                `**생성된 파일들:**\\n` +
                `${result.generated_files?.map(f => `• ${f}`).join('\\n') || '파일 목록을 불러올 수 없습니다'}\\n\\n` +
                `산점도 시각화가 완료되었습니다.`
        }]
      };
    } catch (error) {
      throw new Error(`산점도 생성 실패: ${error.message}`);
    }
  }

  /**
   * Handle time series plots creation
   */
  async handleCreateTimeSeriesPlots(args) {
    const { data_file, date_column, value_columns, plot_types = ['line', 'seasonal_decompose'], rolling_window = 30, output_dir = 'visualizations' } = args;

    try {
      const result = await this.runVisualizationScript('time_series_plots', {
        data_file,
        date_column,
        value_columns,
        plot_types,
        rolling_window,
        output_dir
      });

      return {
        content: [{
          type: 'text',
          text: `**시계열 시각화 생성 완료**\\n\\n` +
                `**데이터:** ${data_file}\\n` +
                `**날짜 컬럼:** ${date_column}\\n` +
                `**값 컬럼들:** ${value_columns.join(', ')}\\n` +
                `**플롯 유형:** ${plot_types.join(', ')}\\n` +
                `**이동평균 윈도우:** ${rolling_window}\\n\\n` +
                `**시계열 정보:**\\n` +
                `• 데이터 기간: ${result.date_range || 'N/A'}\\n` +
                `• 데이터 포인트 수: ${result.data_points || 'N/A'}\\n` +
                `• 주기성 탐지: ${result.seasonality_detected || 'N/A'}\\n\\n` +
                `시계열 시각화가 완료되었습니다.`
        }]
      };
    } catch (error) {
      throw new Error(`시계열 시각화 생성 실패: ${error.message}`);
    }
  }

  /**
   * Handle categorical plots creation
   */
  async handleCreateCategoricalPlots(args) {
    const { data_file, categorical_columns, numeric_column, plot_types = ['countplot', 'barplot'], max_categories = 20, output_dir = 'visualizations' } = args;

    try {
      const result = await this.runVisualizationScript('categorical_plots', {
        data_file,
        categorical_columns,
        numeric_column,
        plot_types,
        max_categories,
        output_dir
      });

      return {
        content: [{
          type: 'text',
          text: `**범주형 시각화 생성 완료**\\n\\n` +
                `**데이터:** ${data_file}\\n` +
                `**범주형 컬럼들:** ${categorical_columns.join(', ')}\\n` +
                `**수치형 컬럼:** ${numeric_column || '없음'}\\n` +
                `**플롯 유형:** ${plot_types.join(', ')}\\n\\n` +
                `**분석 결과:**\\n` +
                `• 분석된 범주 수: ${result.total_categories || 'N/A'}\\n` +
                `• 생성된 차트 수: ${result.chart_count || 'N/A'}\\n\\n` +
                `범주형 데이터 시각화가 완료되었습니다.`
        }]
      };
    } catch (error) {
      throw new Error(`범주형 시각화 생성 실패: ${error.message}`);
    }
  }

  /**
   * Handle statistical plots creation
   */
  async handleCreateStatisticalPlots(args) {
    const { data_file, x_column, y_column, group_column, plot_types = ['regplot', 'residplot'], confidence_interval = 95, output_dir = 'visualizations' } = args;

    try {
      const result = await this.runVisualizationScript('statistical_plots', {
        data_file,
        x_column,
        y_column,
        group_column,
        plot_types,
        confidence_interval,
        output_dir
      });

      return {
        content: [{
          type: 'text',
          text: `**통계적 시각화 생성 완료**\\n\\n` +
                `**데이터:** ${data_file}\\n` +
                `**변수:** ${x_column} vs ${y_column}\\n` +
                `**그룹 변수:** ${group_column || '없음'}\\n` +
                `**신뢰구간:** ${confidence_interval}%\\n\\n` +
                `**통계 정보:**\\n` +
                `• 상관계수: ${result.correlation || 'N/A'}\\n` +
                `• R² 값: ${result.r_squared || 'N/A'}\\n` +
                `• p-value: ${result.p_value || 'N/A'}\\n\\n` +
                `통계적 시각화가 완료되었습니다.`
        }]
      };
    } catch (error) {
      throw new Error(`통계적 시각화 생성 실패: ${error.message}`);
    }
  }

  /**
   * Handle interactive plots creation
   */
  async handleCreateInteractivePlots(args) {
    const { data_file, plot_type = 'scatter_3d', x_column, y_column, z_column, color_column, output_file = 'interactive_plot.html' } = args;

    try {
      const result = await this.runVisualizationScript('interactive_plots', {
        data_file,
        plot_type,
        x_column,
        y_column,
        z_column,
        color_column,
        output_file
      });

      return {
        content: [{
          type: 'text',
          text: `**인터랙티브 시각화 생성 완료**\\n\\n` +
                `**데이터:** ${data_file}\\n` +
                `**플롯 유형:** ${plot_type}\\n` +
                `**변수들:** ${[x_column, y_column, z_column].filter(Boolean).join(', ')}\\n` +
                `**색상 변수:** ${color_column || '없음'}\\n` +
                `**출력 파일:** ${output_file}\\n\\n` +
                `**인터랙티브 기능:**\\n` +
                `• 확대/축소, 회전 가능\\n` +
                `• 데이터 포인트 상세 정보 표시\\n` +
                `• 범례 및 필터링 지원\\n\\n` +
                `인터랙티브 시각화가 생성되었습니다.`
        }]
      };
    } catch (error) {
      throw new Error(`인터랙티브 시각화 생성 실패: ${error.message}`);
    }
  }

  /**
   * Handle dashboard creation
   */
  async handleCreateDashboard(args) {
    const { data_file, dashboard_type = 'overview', include_sections = ['summary_stats', 'distributions', 'correlations'], target_column, output_file = 'data_dashboard.html' } = args;

    try {
      const result = await this.runVisualizationScript('dashboard', {
        data_file,
        dashboard_type,
        include_sections,
        target_column,
        output_file
      });

      return {
        content: [{
          type: 'text',
          text: `**대시보드 생성 완료**\\n\\n` +
                `**데이터:** ${data_file}\\n` +
                `**대시보드 유형:** ${dashboard_type}\\n` +
                `**포함 섹션:** ${include_sections.join(', ')}\\n` +
                `**타겟 변수:** ${target_column || '없음'}\\n` +
                `**출력 파일:** ${output_file}\\n\\n` +
                `**대시보드 구성:**\\n` +
                `• 섹션 수: ${result.section_count || 'N/A'}\\n` +
                `• 차트 수: ${result.chart_count || 'N/A'}\\n` +
                `• 인터랙티브 요소: ${result.interactive_elements || 'N/A'}개\\n\\n` +
                `종합 대시보드가 생성되었습니다.`
        }]
      };
    } catch (error) {
      throw new Error(`대시보드 생성 실패: ${error.message}`);
    }
  }

  /**
   * Run visualization Python script
   */
  async runVisualizationScript(scriptName, options = {}) {
    const scriptPath = path.join(__dirname, '..', 'python', 'visualizations', `${scriptName}.py`);

    return new Promise((resolve, reject) => {
      const process = spawn('python', [scriptPath]);

      let stdout = '';
      let stderr = '';

      process.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      process.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      // Send options via stdin
      process.stdin.write(JSON.stringify(options));
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
          reject(new Error(`시각화 스크립트 실패 (exit code: ${code})\\n${stderr}`));
        }
      });

      process.on('error', (error) => {
        reject(new Error(`시각화 프로세스 오류: ${error.message}`));
      });

      // Set timeout (3 minutes for visualization operations)
      setTimeout(() => {
        process.kill('SIGKILL');
        reject(new Error('시각화 스크립트 실행 시간 초과 (3분)'));
      }, 180000);
    });
  }

  /**
   * Get service status with visualization-specific metrics
   */
  getStatus() {
    const baseStatus = super.getStatus();
    return {
      ...baseStatus,
      toolCount: 8,
      focus: 'data_visualization',
      features: ['distribution_plots', 'correlation_heatmaps', 'scatter_plots', 'time_series', 'categorical_plots', 'statistical_plots', 'interactive_plots', 'dashboards'],
      outputCache: {
        size: this.outputCache.size,
        keys: Array.from(this.outputCache.keys())
      }
    };
  }

  /**
   * Clear output cache
   */
  clearOutputCache() {
    this.outputCache.clear();
    this.logger.info('시각화 출력 캐시가 정리되었습니다');
  }
}

export default VisualizationService;