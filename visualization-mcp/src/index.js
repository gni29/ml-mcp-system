#!/usr/bin/env node

/**
 * ML MCP Visualization Server
 * 머신러닝 데이터 시각화 MCP 서버
 *
 * 이 서버는 다음과 같은 시각화 기능을 제공합니다:
 * - 기본 차트 생성 (선 그래프, 막대 그래프, 산점도, 히스토그램)
 * - 통계 플롯 (박스 플롯, 바이올린 플롯, 히트맵, 상관관계 매트릭스)
 * - 머신러닝 시각화 (혼동 행렬, ROC 곡선, 특성 중요도, 학습 곡선)
 * - 대화형 차트 (Plotly, Bokeh 기반)
 * - 3D 시각화
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from '@modelcontextprotocol/sdk/types.js';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Python 스크립트 경로 설정
const PYTHON_SCRIPTS_PATH = path.resolve(__dirname, '../../python/visualization');

class VisualizationMCPServer {
  constructor() {
    this.server = new Server(
      {
        name: 'ml-visualization-mcp',
        version: '1.0.0',
        description: '머신러닝 데이터 시각화를 위한 MCP 서버 - ML Data Visualization MCP Server'
      },
      {
        capabilities: {
          tools: {},
        },
      },
    );

    this.setupToolHandlers();
  }

  setupToolHandlers() {
    // 도구 목록 제공
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: [
          // 기본 차트 생성
          {
            name: 'create_line_chart',
            description: '선 그래프 생성 - Line chart generation with customizable styling',
            inputSchema: {
              type: 'object',
              properties: {
                data: {
                  type: 'object',
                  description: '차트 데이터 (JSON 형태) - Chart data in JSON format'
                },
                x_column: {
                  type: 'string',
                  description: 'X축 컬럼명 - X-axis column name'
                },
                y_column: {
                  type: 'string',
                  description: 'Y축 컬럼명 - Y-axis column name'
                },
                title: {
                  type: 'string',
                  description: '차트 제목 - Chart title',
                  default: ''
                },
                output_path: {
                  type: 'string',
                  description: '출력 파일 경로 - Output file path',
                  default: 'line_chart.png'
                }
              },
              required: ['data', 'x_column', 'y_column']
            }
          },
          {
            name: 'create_scatter_plot',
            description: '산점도 생성 - Scatter plot generation with correlation analysis',
            inputSchema: {
              type: 'object',
              properties: {
                data: {
                  type: 'object',
                  description: '차트 데이터 (JSON 형태) - Chart data in JSON format'
                },
                x_column: {
                  type: 'string',
                  description: 'X축 컬럼명 - X-axis column name'
                },
                y_column: {
                  type: 'string',
                  description: 'Y축 컬럼명 - Y-axis column name'
                },
                color_column: {
                  type: 'string',
                  description: '색상 구분 컬럼명 - Color grouping column name',
                  default: null
                },
                title: {
                  type: 'string',
                  description: '차트 제목 - Chart title',
                  default: ''
                },
                output_path: {
                  type: 'string',
                  description: '출력 파일 경로 - Output file path',
                  default: 'scatter_plot.png'
                }
              },
              required: ['data', 'x_column', 'y_column']
            }
          },
          {
            name: 'create_histogram',
            description: '히스토그램 생성 - Histogram generation with distribution analysis',
            inputSchema: {
              type: 'object',
              properties: {
                data: {
                  type: 'object',
                  description: '차트 데이터 (JSON 형태) - Chart data in JSON format'
                },
                column: {
                  type: 'string',
                  description: '히스토그램 생성할 컬럼명 - Column name for histogram'
                },
                bins: {
                  type: 'integer',
                  description: '구간 수 - Number of bins',
                  default: 30
                },
                title: {
                  type: 'string',
                  description: '차트 제목 - Chart title',
                  default: ''
                },
                output_path: {
                  type: 'string',
                  description: '출력 파일 경로 - Output file path',
                  default: 'histogram.png'
                }
              },
              required: ['data', 'column']
            }
          },
          {
            name: 'create_bar_chart',
            description: '막대 그래프 생성 - Bar chart generation for categorical data',
            inputSchema: {
              type: 'object',
              properties: {
                data: {
                  type: 'object',
                  description: '차트 데이터 (JSON 형태) - Chart data in JSON format'
                },
                x_column: {
                  type: 'string',
                  description: 'X축 컬럼명 (범주형) - X-axis column name (categorical)'
                },
                y_column: {
                  type: 'string',
                  description: 'Y축 컬럼명 (수치형) - Y-axis column name (numerical)'
                },
                title: {
                  type: 'string',
                  description: '차트 제목 - Chart title',
                  default: ''
                },
                output_path: {
                  type: 'string',
                  description: '출력 파일 경로 - Output file path',
                  default: 'bar_chart.png'
                }
              },
              required: ['data', 'x_column', 'y_column']
            }
          },

          // 통계 플롯
          {
            name: 'create_box_plot',
            description: '박스 플롯 생성 - Box plot for distribution analysis',
            inputSchema: {
              type: 'object',
              properties: {
                data: {
                  type: 'object',
                  description: '차트 데이터 (JSON 형태) - Chart data in JSON format'
                },
                x_column: {
                  type: 'string',
                  description: 'X축 컬럼명 (범주형) - X-axis column name (categorical)',
                  default: null
                },
                y_column: {
                  type: 'string',
                  description: 'Y축 컬럼명 (수치형) - Y-axis column name (numerical)'
                },
                title: {
                  type: 'string',
                  description: '차트 제목 - Chart title',
                  default: ''
                },
                output_path: {
                  type: 'string',
                  description: '출력 파일 경로 - Output file path',
                  default: 'box_plot.png'
                }
              },
              required: ['data', 'y_column']
            }
          },
          {
            name: 'create_correlation_heatmap',
            description: '상관관계 히트맵 생성 - Correlation heatmap for numerical features',
            inputSchema: {
              type: 'object',
              properties: {
                data: {
                  type: 'object',
                  description: '차트 데이터 (JSON 형태) - Chart data in JSON format'
                },
                columns: {
                  type: 'array',
                  items: { type: 'string' },
                  description: '분석할 컬럼 목록 - Columns to analyze',
                  default: null
                },
                method: {
                  type: 'string',
                  enum: ['pearson', 'kendall', 'spearman'],
                  description: '상관관계 계산 방법 - Correlation method',
                  default: 'pearson'
                },
                title: {
                  type: 'string',
                  description: '차트 제목 - Chart title',
                  default: ''
                },
                output_path: {
                  type: 'string',
                  description: '출력 파일 경로 - Output file path',
                  default: 'correlation_heatmap.png'
                }
              },
              required: ['data']
            }
          },

          // ML 시각화
          {
            name: 'create_confusion_matrix',
            description: '혼동 행렬 생성 - Confusion matrix visualization for classification results',
            inputSchema: {
              type: 'object',
              properties: {
                y_true: {
                  type: 'array',
                  description: '실제 값 - True labels'
                },
                y_pred: {
                  type: 'array',
                  description: '예측 값 - Predicted labels'
                },
                labels: {
                  type: 'array',
                  items: { type: 'string' },
                  description: '클래스 라벨 - Class labels',
                  default: null
                },
                title: {
                  type: 'string',
                  description: '차트 제목 - Chart title',
                  default: 'Confusion Matrix'
                },
                output_path: {
                  type: 'string',
                  description: '출력 파일 경로 - Output file path',
                  default: 'confusion_matrix.png'
                }
              },
              required: ['y_true', 'y_pred']
            }
          },
          {
            name: 'create_feature_importance_plot',
            description: '특성 중요도 플롯 생성 - Feature importance plot for ML models',
            inputSchema: {
              type: 'object',
              properties: {
                feature_names: {
                  type: 'array',
                  items: { type: 'string' },
                  description: '특성 이름 목록 - Feature names'
                },
                importance_values: {
                  type: 'array',
                  items: { type: 'number' },
                  description: '중요도 값 - Importance values'
                },
                top_k: {
                  type: 'integer',
                  description: '상위 몇 개 특성을 표시할지 - Number of top features to show',
                  default: 15
                },
                title: {
                  type: 'string',
                  description: '차트 제목 - Chart title',
                  default: 'Feature Importance'
                },
                output_path: {
                  type: 'string',
                  description: '출력 파일 경로 - Output file path',
                  default: 'feature_importance.png'
                }
              },
              required: ['feature_names', 'importance_values']
            }
          },
          {
            name: 'create_learning_curve',
            description: '학습 곡선 생성 - Learning curve visualization for model training analysis',
            inputSchema: {
              type: 'object',
              properties: {
                train_sizes: {
                  type: 'array',
                  items: { type: 'number' },
                  description: '훈련 데이터 크기 - Training data sizes'
                },
                train_scores: {
                  type: 'object',
                  description: '훈련 점수 (mean, std) - Training scores (mean, std)'
                },
                val_scores: {
                  type: 'object',
                  description: '검증 점수 (mean, std) - Validation scores (mean, std)'
                },
                metric_name: {
                  type: 'string',
                  description: '평가 지표명 - Metric name',
                  default: 'Score'
                },
                title: {
                  type: 'string',
                  description: '차트 제목 - Chart title',
                  default: 'Learning Curve'
                },
                output_path: {
                  type: 'string',
                  description: '출력 파일 경로 - Output file path',
                  default: 'learning_curve.png'
                }
              },
              required: ['train_sizes', 'train_scores', 'val_scores']
            }
          },

          // 자동 시각화
          {
            name: 'auto_visualize',
            description: '자동 시각화 - Automatic visualization based on data types',
            inputSchema: {
              type: 'object',
              properties: {
                data: {
                  type: 'object',
                  description: '시각화할 데이터 (JSON 형태) - Data to visualize in JSON format'
                },
                output_dir: {
                  type: 'string',
                  description: '출력 디렉토리 - Output directory',
                  default: 'auto_viz_results'
                },
                max_plots: {
                  type: 'integer',
                  description: '최대 생성할 플롯 수 - Maximum number of plots to generate',
                  default: 10
                }
              },
              required: ['data']
            }
          },

          // 대화형 차트
          {
            name: 'create_interactive_chart',
            description: '대화형 차트 생성 - Interactive chart generation using Plotly',
            inputSchema: {
              type: 'object',
              properties: {
                data: {
                  type: 'object',
                  description: '차트 데이터 (JSON 형태) - Chart data in JSON format'
                },
                chart_type: {
                  type: 'string',
                  enum: ['scatter', 'line', 'bar', 'box', 'histogram', '3d_scatter'],
                  description: '차트 유형 - Chart type'
                },
                x_column: {
                  type: 'string',
                  description: 'X축 컬럼명 - X-axis column name'
                },
                y_column: {
                  type: 'string',
                  description: 'Y축 컬럼명 - Y-axis column name'
                },
                z_column: {
                  type: 'string',
                  description: 'Z축 컬럼명 (3D용) - Z-axis column name (for 3D)',
                  default: null
                },
                color_column: {
                  type: 'string',
                  description: '색상 구분 컬럼명 - Color grouping column name',
                  default: null
                },
                title: {
                  type: 'string',
                  description: '차트 제목 - Chart title',
                  default: ''
                },
                output_path: {
                  type: 'string',
                  description: '출력 파일 경로 (.html) - Output file path (.html)',
                  default: 'interactive_chart.html'
                }
              },
              required: ['data', 'chart_type', 'x_column']
            }
          }
        ]
      };
    });

    // 도구 실행 처리
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        switch (name) {
          case 'create_line_chart':
            return await this.runPythonScript('2d/line.py', args);

          case 'create_scatter_plot':
            return await this.runPythonScript('2d/scatter_enhanced.py', args);

          case 'create_histogram':
            return await this.runPythonScript('2d/histogram.py', args);

          case 'create_bar_chart':
            return await this.runPythonScript('2d/bar_chart.py', args);

          case 'create_box_plot':
            return await this.runPythonScript('statistical/box_plot.py', args);

          case 'create_correlation_heatmap':
            return await this.runPythonScript('2d/heatmap.py', args);

          case 'create_confusion_matrix':
            return await this.runPythonScript('ml/confusion_matrix.py', args);

          case 'create_feature_importance_plot':
            return await this.runPythonScript('ml/feature_importance.py', args);

          case 'create_learning_curve':
            return await this.runPythonScript('ml/learning_curve.py', args);

          case 'auto_visualize':
            return await this.runPythonScript('auto_visualizer.py', args);

          case 'create_interactive_chart':
            return await this.runPythonScript('interactive/plotly_charts.py', args);

          default:
            throw new Error(`Unknown tool: ${name}`);
        }
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: `Error executing ${name}: ${error.message}\n\nStackTrace: ${error.stack}`
            }
          ],
          isError: true
        };
      }
    });
  }

  async runPythonScript(scriptPath, args) {
    return new Promise((resolve, reject) => {
      const fullScriptPath = path.join(PYTHON_SCRIPTS_PATH, scriptPath);
      const pythonProcess = spawn('python', [fullScriptPath], {
        cwd: path.dirname(fullScriptPath)
      });

      // 입력 데이터를 JSON 형태로 전송
      pythonProcess.stdin.write(JSON.stringify(args));
      pythonProcess.stdin.end();

      let stdout = '';
      let stderr = '';

      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          reject(new Error(`Python script failed with code ${code}\nStderr: ${stderr}\nStdout: ${stdout}`));
          return;
        }

        try {
          // JSON 응답 파싱
          const result = JSON.parse(stdout);

          resolve({
            content: [
              {
                type: 'text',
                text: JSON.stringify(result, null, 2)
              }
            ]
          });
        } catch (parseError) {
          // JSON 파싱 실패 시 텍스트로 응답
          resolve({
            content: [
              {
                type: 'text',
                text: stdout || 'Script completed successfully but returned no output'
              }
            ]
          });
        }
      });

      pythonProcess.on('error', (error) => {
        reject(new Error(`Failed to start Python process: ${error.message}`));
      });
    });
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('ML Visualization MCP Server running on stdio');
  }
}

const server = new VisualizationMCPServer();
server.run().catch(console.error);