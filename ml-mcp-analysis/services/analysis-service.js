/**
 * Lightweight Analysis Service
 * 경량 분석 서비스 - 기본 통계 및 데이터 탐색에 특화
 */

import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import { BaseService } from 'ml-mcp-shared/utils/base-service';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class LightweightAnalysisService extends BaseService {
  constructor(logger) {
    super('lightweight-analysis-service', 'analysis', '1.0.0');
    this.logger = logger;
    this.capabilities = ['tools'];
  }

  /**
   * Initialize the analysis service
   */
  async initialize() {
    try {
      this.logger.info('경량 분석 서비스 초기화 중');

      // Test Python environment
      await this.testPythonEnvironment();

      await super.initialize();
      this.logger.info('경량 분석 서비스 초기화 완료');

    } catch (error) {
      this.logger.error('경량 분석 서비스 초기화 실패:', error);
      throw error;
    }
  }

  /**
   * Test Python environment
   */
  async testPythonEnvironment() {
    return new Promise((resolve, reject) => {
      const pythonProcess = spawn('python', ['-c', 'import pandas, numpy, scipy; print("Analysis environment OK")']);

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error('분석 라이브러리 (pandas, numpy, scipy)를 사용할 수 없습니다'));
        }
      });

      pythonProcess.on('error', (error) => {
        reject(new Error(`Python 환경 테스트 오류: ${error.message}`));
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
        description: '데이터의 기본 통계 정보와 개요를 제공합니다.',
        inputSchema: {
          type: 'object',
          properties: {
            query: {
              type: 'string',
              description: '분석 요청 내용'
            },
            file_path: {
              type: 'string',
              description: '분석할 파일 경로'
            },
            include_distribution: {
              type: 'boolean',
              description: '분포 분석 포함 여부',
              default: true
            }
          },
          required: ['query', 'file_path']
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
            columns: {
              type: 'array',
              description: '분석할 특정 컬럼들 (선택사항)',
              items: {
                type: 'string'
              }
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
            },
            threshold: {
              type: 'number',
              description: '강한 상관관계 임계값',
              default: 0.7,
              minimum: 0,
              maximum: 1
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
      },
      {
        name: 'data_quality_assessment',
        description: '데이터 품질을 종합적으로 평가합니다.',
        inputSchema: {
          type: 'object',
          properties: {
            file_path: {
              type: 'string',
              description: '분석할 데이터 파일 경로'
            },
            generate_report: {
              type: 'boolean',
              description: '상세 품질 보고서 생성 여부',
              default: false
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
      throw new Error('분석 서비스가 초기화되지 않았습니다');
    }

    this.logger.info(`분석 도구 실행 중: ${toolName}`, args);

    switch (toolName) {
      case 'analyze_data':
        return await this.handleAnalyzeData(args);
      case 'descriptive_statistics':
        return await this.handleDescriptiveStatistics(args);
      case 'correlation_analysis':
        return await this.handleCorrelationAnalysis(args);
      case 'missing_data_analysis':
        return await this.handleMissingDataAnalysis(args);
      case 'data_quality_assessment':
        return await this.handleDataQualityAssessment(args);
      default:
        throw new Error(`알 수 없는 분석 도구: ${toolName}`);
    }
  }

  /**
   * Handle general data analysis
   */
  async handleAnalyzeData(args) {
    const { query, file_path, include_distribution = true } = args;

    try {
      const result = await this.runAnalysisScript('distribution', {
        file_path,
        include_distribution,
        query
      });

      return {
        content: [{
          type: 'text',
          text: `**데이터 분석 완료**\\n\\n` +
                `**요청:** ${query}\\n` +
                `**데이터:** ${file_path}\\n\\n` +
                `**주요 결과:**\\n` +
                `• 데이터 크기: ${result.data_info?.shape?.[0] || 'N/A'}행 × ${result.data_info?.shape?.[1] || 'N/A'}열\\n` +
                `• 수치형 변수: ${result.data_info?.numeric_columns?.length || 0}개\\n` +
                `• 범주형 변수: ${result.data_info?.categorical_columns?.length || 0}개\\n` +
                `• 결측치: ${result.missing_summary?.total_missing || 0}개\\n\\n` +
                `분석이 완료되었습니다.`
        }]
      };
    } catch (error) {
      throw new Error(`데이터 분석 실패: ${error.message}`);
    }
  }

  /**
   * Handle descriptive statistics
   */
  async handleDescriptiveStatistics(args) {
    const { file_path, columns } = args;

    try {
      const result = await this.runAnalysisScript('descriptive_stats', {
        file_path,
        columns
      });

      return {
        content: [{
          type: 'text',
          text: `**기술통계 분석 완료**\\n\\n` +
                `**데이터:** ${file_path}\\n` +
                `**분석 변수:** ${columns ? columns.join(', ') : '모든 수치형 변수'}\\n\\n` +
                `**통계 요약:**\\n` +
                `• 평균값 범위: ${this.formatStatRange(result.summary_stats, 'mean')}\\n` +
                `• 표준편차 범위: ${this.formatStatRange(result.summary_stats, 'std')}\\n` +
                `• 변동계수가 높은 변수: ${result.high_variation_vars?.join(', ') || '없음'}\\n\\n` +
                `기술통계 분석이 완료되었습니다.`
        }]
      };
    } catch (error) {
      throw new Error(`기술통계 분석 실패: ${error.message}`);
    }
  }

  /**
   * Handle correlation analysis
   */
  async handleCorrelationAnalysis(args) {
    const { file_path, method = 'pearson', threshold = 0.7 } = args;

    try {
      const result = await this.runAnalysisScript('correlation', {
        file_path,
        method,
        threshold
      });

      return {
        content: [{
          type: 'text',
          text: `**상관관계 분석 완료**\\n\\n` +
                `**방법:** ${method}\\n` +
                `**임계값:** ${threshold}\\n` +
                `**데이터:** ${file_path}\\n\\n` +
                `**주요 결과:**\\n` +
                `• 강한 상관관계: ${result.strong_correlations?.length || 0}개\\n` +
                `• 분석 변수 수: ${result.analyzed_variables || 0}개\\n` +
                `• 최고 상관계수: ${result.max_correlation || 'N/A'}\\n\\n` +
                `상관관계 분석이 완료되었습니다.`
        }]
      };
    } catch (error) {
      throw new Error(`상관관계 분석 실패: ${error.message}`);
    }
  }

  /**
   * Handle missing data analysis
   */
  async handleMissingDataAnalysis(args) {
    const { file_path, suggest_imputation = true } = args;

    try {
      const result = await this.runAnalysisScript('missing_data', {
        file_path,
        suggest_imputation
      });

      return {
        content: [{
          type: 'text',
          text: `**결측치 분석 완료**\\n\\n` +
                `**데이터:** ${file_path}\\n\\n` +
                `**결측치 현황:**\\n` +
                `• 총 결측값: ${result.missing_summary?.total_missing || 0}개\\n` +
                `• 결측 비율: ${result.missing_summary?.missing_percentage || 0}%\\n` +
                `• 완전한 행: ${result.missing_summary?.complete_rows || 0}개\\n` +
                `• 결측이 있는 컬럼: ${result.columns_with_missing?.length || 0}개\\n\\n` +
                `${suggest_imputation ? '보완 방법 제안이 포함되었습니다.' : ''}\\n\\n` +
                `결측치 분석이 완료되었습니다.`
        }]
      };
    } catch (error) {
      throw new Error(`결측치 분석 실패: ${error.message}`);
    }
  }

  /**
   * Handle data quality assessment
   */
  async handleDataQualityAssessment(args) {
    const { file_path, generate_report = false } = args;

    try {
      const result = await this.runAnalysisScript('descriptive_stats', {
        file_path,
        generate_report
      });

      return {
        content: [{
          type: 'text',
          text: `**데이터 품질 평가 완료**\\n\\n` +
                `**데이터:** ${file_path}\\n\\n` +
                `**품질 점수:**\\n` +
                `• 완전성: ${result.quality_scores?.completeness || 0}%\\n` +
                `• 일관성: ${result.quality_scores?.consistency || 0}%\\n` +
                `• 정확성: ${result.quality_scores?.accuracy || 0}%\\n` +
                `• 전체 점수: ${result.overall_score || 0}%\\n\\n` +
                `**주요 이슈:**\\n` +
                `${result.quality_issues?.map(issue => `• ${issue}`).join('\\n') || '없음'}\\n\\n` +
                `데이터 품질 평가가 완료되었습니다.`
        }]
      };
    } catch (error) {
      throw new Error(`데이터 품질 평가 실패: ${error.message}`);
    }
  }

  /**
   * Run Python analysis script
   */
  async runAnalysisScript(scriptName, options = {}) {
    const scriptPath = path.join(__dirname, '..', '..', 'python', 'analyzers', 'basic', `${scriptName}.py`);

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
          reject(new Error(`분석 스크립트 실패 (exit code: ${code})\\n${stderr}`));
        }
      });

      process.on('error', (error) => {
        reject(new Error(`분석 프로세스 오류: ${error.message}`));
      });

      // Set timeout
      setTimeout(() => {
        process.kill('SIGKILL');
        reject(new Error('분석 스크립트 실행 시간 초과'));
      }, 60000); // 1 minute timeout for lightweight analysis
    });
  }

  /**
   * Helper method to format statistical ranges
   */
  formatStatRange(stats, statType) {
    if (!stats || !stats[statType]) return 'N/A';

    const values = Object.values(stats[statType]);
    const min = Math.min(...values);
    const max = Math.max(...values);

    return `${min.toFixed(2)} ~ ${max.toFixed(2)}`;
  }

  /**
   * Get service status with analysis-specific metrics
   */
  getStatus() {
    const baseStatus = super.getStatus();
    return {
      ...baseStatus,
      toolCount: 5,
      focus: 'lightweight_analysis',
      features: ['basic_statistics', 'correlation', 'missing_data', 'data_quality']
    };
  }
}

export default LightweightAnalysisService;