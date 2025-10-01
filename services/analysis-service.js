/**
 * Analysis Service - Handles all data analysis operations
 * Provides tools for descriptive statistics, correlation analysis, and data exploration
 */

import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class AnalysisService {
  constructor(logger) {
    this.logger = logger;
    this.name = 'analysis-service';
    this.type = 'analysis';
    this.version = '2.0.0';
    this.capabilities = ['tools'];
    this.isInitialized = false;
  }

  /**
   * Initialize the analysis service
   */
  async initialize() {
    try {
      this.logger.info('분석 서비스 초기화 중');

      // Test Python availability
      await this.testPythonEnvironment();

      this.isInitialized = true;
      this.logger.info('분석 서비스 초기화 완료');

    } catch (error) {
      this.logger.error('분석 서비스 초기화 실패:', error);
      throw error;
    }
  }

  /**
   * Test Python environment
   */
  async testPythonEnvironment() {
    return new Promise((resolve, reject) => {
      const pythonProcess = spawn('python', ['--version']);

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error('Python is not available or not properly configured'));
        }
      });

      pythonProcess.on('error', (error) => {
        reject(new Error(`Python process error: ${error.message}`));
      });
    });
  }

  /**
   * Get available tools
   */
  async getTools() {
    return [
      {
        name: 'analyze_data',
        description: '데이터 파일을 분석하고 기본 통계 정보를 제공합니다.',
        inputSchema: {
          type: 'object',
          properties: {
            query: {
              type: 'string',
              description: '분석 요청 내용'
            },
            file_path: {
              type: 'string',
              description: '분석할 파일 경로 (선택사항)'
            },
            auto_detect_files: {
              type: 'boolean',
              description: '파일 자동 감지 여부',
              default: true
            }
          },
          required: ['query']
        }
      },
      {
        name: 'integrated_analysis',
        description: '통합 분석을 수행하여 HTML 리포트와 JSON 결과를 생성합니다.',
        inputSchema: {
          type: 'object',
          properties: {
            file_path: {
              type: 'string',
              description: '분석할 데이터 파일 경로'
            },
            output_dir: {
              type: 'string',
              description: '출력 디렉토리',
              default: 'results'
            },
            dataset_name: {
              type: 'string',
              description: '데이터셋 이름 (선택사항)'
            }
          },
          required: ['file_path']
        }
      },
      {
        name: 'correlation_analysis',
        description: '변수들 간의 상관관계를 분석합니다.',
        inputSchema: {
          type: 'object',
          properties: {
            file_path: {
              type: 'string',
              description: '분석할 데이터 파일 경로'
            },
            method: {
              type: 'string',
              description: '상관관계 분석 방법',
              enum: ['pearson', 'spearman', 'kendall'],
              default: 'pearson'
            }
          },
          required: ['file_path']
        }
      },
      {
        name: 'descriptive_statistics',
        description: '기술통계 분석을 수행합니다.',
        inputSchema: {
          type: 'object',
          properties: {
            file_path: {
              type: 'string',
              description: '분석할 데이터 파일 경로'
            },
            include_plots: {
              type: 'boolean',
              description: '분포 플롯 포함 여부',
              default: true
            }
          },
          required: ['file_path']
        }
      },
      {
        name: 'missing_data_analysis',
        description: '결측치 패턴을 분석하고 처리 방법을 제안합니다.',
        inputSchema: {
          type: 'object',
          properties: {
            file_path: {
              type: 'string',
              description: '분석할 데이터 파일 경로'
            },
            suggest_imputation: {
              type: 'boolean',
              description: '결측치 보완 방법 제안 여부',
              default: true
            }
          },
          required: ['file_path']
        }
      }
    ];
  }

  /**
   * Execute a tool
   */
  async executeTool(toolName, args) {
    if (!this.isInitialized) {
      throw new Error('Analysis service not initialized');
    }

    this.logger.info(`분석 도구 실행 중: ${toolName}`, args);

    switch (toolName) {
      case 'analyze_data':
        return await this.handleAnalyzeData(args);
      case 'integrated_analysis':
        return await this.handleIntegratedAnalysis(args);
      case 'correlation_analysis':
        return await this.handleCorrelationAnalysis(args);
      case 'descriptive_statistics':
        return await this.handleDescriptiveStatistics(args);
      case 'missing_data_analysis':
        return await this.handleMissingDataAnalysis(args);
      default:
        throw new Error(`Unknown analysis tool: ${toolName}`);
    }
  }

  /**
   * Handle general data analysis
   */
  async handleAnalyzeData(args) {
    const { query, file_path, auto_detect_files = true } = args;

    try {
      // If no file path provided and auto-detect is enabled, try to find data files
      let targetFile = file_path;
      if (!targetFile && auto_detect_files) {
        targetFile = await this.autoDetectDataFile();
      }

      if (!targetFile) {
        return {
          content: [{
            type: 'text',
            text: `**데이터 분석 요청: ${query}**\n\n` +
                  `분석할 데이터 파일을 지정해주세요. \`file_path\` 파라미터를 사용하거나 ` +
                  `데이터 파일을 프로젝트 디렉토리에 배치하세요.`
          }]
        };
      }

      // Perform basic analysis
      const result = await this.runPythonScript('basic', {
        data: targetFile,
        output: 'results'
      });

      return {
        content: [{
          type: 'text',
          text: `**데이터 분석 완료**\n\n` +
                `**요청:** ${query}\n` +
                `**데이터:** ${targetFile}\n\n` +
                `**결과:**\n${JSON.stringify(result, null, 2)}`
        }]
      };

    } catch (error) {
      throw new Error(`Data analysis failed: ${error.message}`);
    }
  }

  /**
   * Handle integrated analysis
   */
  async handleIntegratedAnalysis(args) {
    const { file_path, output_dir = 'results', dataset_name } = args;

    try {
      const result = await this.runPythonScript('integrated', {
        data: file_path,
        output: output_dir
      });

      return {
        content: [{
          type: 'text',
          text: `**통합 분석 완료**\n\n` +
                `**데이터셋:** ${dataset_name || path.basename(file_path)}\n` +
                `**출력 위치:** ${output_dir}\n\n` +
                `**생성된 파일:**\n` +
                `• HTML 리포트: enhanced_analysis_report_*.html\n` +
                `• JSON 결과: enhanced_analysis_results_*.json\n` +
                `• 시각화 이미지: plots/\n\n` +
                `분석이 완료되었습니다. 결과 파일을 확인하세요.`
        }]
      };

    } catch (error) {
      throw new Error(`Integrated analysis failed: ${error.message}`);
    }
  }

  /**
   * Handle correlation analysis
   */
  async handleCorrelationAnalysis(args) {
    const { file_path, method = 'pearson' } = args;

    try {
      const result = await this.runSpecificAnalysis('correlation', {
        data: file_path,
        method
      });

      return {
        content: [{
          type: 'text',
          text: `**상관관계 분석 완료**\n\n` +
                `**방법:** ${method}\n` +
                `**데이터:** ${file_path}\n\n` +
                `**결과:**\n${JSON.stringify(result, null, 2)}`
        }]
      };

    } catch (error) {
      throw new Error(`Correlation analysis failed: ${error.message}`);
    }
  }

  /**
   * Handle descriptive statistics
   */
  async handleDescriptiveStatistics(args) {
    const { file_path, include_plots = true } = args;

    try {
      const result = await this.runSpecificAnalysis('descriptive_stats', {
        data: file_path,
        plots: include_plots
      });

      return {
        content: [{
          type: 'text',
          text: `**기술통계 분석 완료**\n\n` +
                `**데이터:** ${file_path}\n` +
                `**플롯 포함:** ${include_plots ? '예' : '아니오'}\n\n` +
                `**결과:**\n${JSON.stringify(result, null, 2)}`
        }]
      };

    } catch (error) {
      throw new Error(`Descriptive statistics failed: ${error.message}`);
    }
  }

  /**
   * Handle missing data analysis
   */
  async handleMissingDataAnalysis(args) {
    const { file_path, suggest_imputation = true } = args;

    try {
      const result = await this.runSpecificAnalysis('missing_data', {
        data: file_path,
        imputation: suggest_imputation
      });

      return {
        content: [{
          type: 'text',
          text: `**결측치 분석 완료**\n\n` +
                `**데이터:** ${file_path}\n` +
                `**보완 방법 제안:** ${suggest_imputation ? '예' : '아니오'}\n\n` +
                `**결과:**\n${JSON.stringify(result, null, 2)}`
        }]
      };

    } catch (error) {
      throw new Error(`Missing data analysis failed: ${error.message}`);
    }
  }

  /**
   * Run Python script with arguments
   */
  async runPythonScript(command, options = {}) {
    const runnerPath = path.join(__dirname, '..', 'scripts', 'python_runner.py');
    const args = [runnerPath, command];

    // Add options as arguments
    if (options.data) {
      args.push('--data', options.data);
    }
    if (options.output) {
      args.push('--output', options.output);
    }
    if (options.type) {
      args.push('--type', options.type);
    }

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

      process.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdout);
            resolve(result);
          } catch (e) {
            resolve({ output: stdout, raw: true });
          }
        } else {
          reject(new Error(`Python script failed (exit code: ${code})\n${stderr}`));
        }
      });

      process.on('error', (error) => {
        reject(new Error(`Python process error: ${error.message}`));
      });

      // Set timeout
      setTimeout(() => {
        process.kill('SIGKILL');
        reject(new Error('Python script execution timeout'));
      }, 300000); // 5 minutes
    });
  }

  /**
   * Run specific analysis module
   */
  async runSpecificAnalysis(analysisType, options) {
    const modulePath = path.join(__dirname, '..', 'python', 'analyzers', 'basic', `${analysisType}.py`);

    return new Promise((resolve, reject) => {
      const process = spawn('python', [modulePath]);

      let stdout = '';
      let stderr = '';

      process.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      process.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      // Send data to stdin if needed
      if (options.data) {
        process.stdin.write(JSON.stringify(options));
        process.stdin.end();
      }

      process.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdout);
            resolve(result);
          } catch (e) {
            resolve({ output: stdout, raw: true });
          }
        } else {
          reject(new Error(`Analysis failed (exit code: ${code})\n${stderr}`));
        }
      });

      process.on('error', (error) => {
        reject(new Error(`Analysis process error: ${error.message}`));
      });
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
      toolCount: 5,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Cleanup
   */
  async cleanup() {
    this.logger.info('분석 서비스 정리 중');
    this.isInitialized = false;
  }
}

export default AnalysisService;