// core/pipeline-manager.js - 미완성 부분들 완성

  // executeWorkflow 메서드 완성
  async executeWorkflow(workflowData, sessionId, userQuery) {
    if (this.isExecuting) {
      throw new Error('다른 워크플로우가 실행 중입니다.');
    }

    this.isExecuting = true;
    this.currentSession = sessionId;

    try {
      this.logger.info('워크플로우 실행 시작', {
        sessionId,
        workflowName: workflowData.workflow.name
      });

      const startTime = Date.now();
      const results = {
        sessionId,
        workflowName: workflowData.workflow.name,
        userQuery,
        startTime: new Date().toISOString(),
        steps: [],
        intermediateResults: {},
        finalResult: null,
        status: 'running'
      };

      // 워크플로우 단계별 실행
      for (let i = 0; i < workflowData.workflow.steps.length; i++) {
        const step = workflowData.workflow.steps[i];
        const stepResult = await this.executeStep(step, results.intermediateResults, i + 1);
        
        results.steps.push(stepResult);
        
        // 중간 결과 저장
        if (stepResult.success) {
          results.intermediateResults[`step_${i + 1}`] = stepResult.result;
          results.intermediateResults[`${step.type}_${step.method}`] = stepResult.result;
        }
      }

      // 최종 결과 생성
      results.finalResult = await this.generateFinalResult(results);
      results.status = 'completed';
      results.endTime = new Date().toISOString();
      results.executionTime = Date.now() - startTime;

      // 결과 저장
      await this.saveWorkflowResults(results);

      this.logger.info('워크플로우 실행 완료', {
        sessionId,
        executionTime: results.executionTime
      });

      return results;

    } catch (error) {
      this.logger.error('워크플로우 실행 실패:', error);
      throw error;
    } finally {
      this.isExecuting = false;
      this.currentSession = null;
    }
  }

  // generateFinalResult 메서드 완성
  async generateFinalResult(results) {
    try {
      const finalResult = {
        summary: this.generateSummary(results),
        outputs: this.collectOutputs(results),
        visualizations: this.collectVisualizations(results),
        statistics: this.collectStatistics(results),
        recommendations: this.generateRecommendations(results),
        artifacts: this.collectArtifacts(results)
      };

      return finalResult;
    } catch (error) {
      this.logger.error('최종 결과 생성 실패:', error);
      return {
        error: error.message,
        partialResults: results.intermediateResults
      };
    }
  }

  generateSummary(results) {
    const { steps, executionTime, workflowName } = results;
    const successfulSteps = steps.filter(step => step.success).length;
    const failedSteps = steps.filter(step => !step.success).length;

    return {
      workflowName,
      totalSteps: steps.length,
      successfulSteps,
      failedSteps,
      executionTime: `${(executionTime / 1000).toFixed(2)}초`,
      successRate: `${((successfulSteps / steps.length) * 100).toFixed(1)}%`
    };
  }

  collectOutputs(results) {
    const outputs = {};
    
    results.steps.forEach((step, index) => {
      if (step.success && step.result) {
        outputs[`step_${index + 1}`] = {
          type: step.type,
          method: step.method,
          result: step.result,
          executionTime: step.executionTime
        };
      }
    });

    return outputs;
  }

  collectVisualizations(results) {
    const visualizations = [];
    
    results.steps.forEach((step, index) => {
      if (step.success && step.type === 'visualization' && step.result) {
        visualizations.push({
          stepNumber: index + 1,
          chartType: step.method,
          filePath: step.result.chart_path,
          description: step.result.description || `${step.method} 차트`
        });
      }
    });

    return visualizations;
  }

  collectStatistics(results) {
    const statistics = {};
    
    results.steps.forEach((step, index) => {
      if (step.success &&
          (step.type === 'basic' || step.type === 'advanced') &&
          step.result &&
          step.result.statistics) {
        statistics[`${step.type}_${step.method}`] = step.result.statistics;
      }
    });

    return statistics;
  }

  generateRecommendations(results) {
    const recommendations = [];
    
    // 데이터 품질 기반 권장사항
    const dataQuality = this.assessDataQuality(results);
    if (dataQuality.missingValues > 0.1) {
      recommendations.push('데이터에 누락값이 많습니다. 데이터 전처리를 고려해보세요.');
    }

    // 모델 성능 기반 권장사항
    const modelPerformance = this.assessModelPerformance(results);
    if (modelPerformance.accuracy < 0.8) {
      recommendations.push('모델 성능이 낮습니다. 하이퍼파라미터 튜닝을 고려해보세요.');
    }

    // 추가 분석 제안
    const hasCorrelation = results.steps.some(step =>
      step.type === 'basic' && step.method === 'correlation'
    );
    if (!hasCorrelation) {
      recommendations.push('변수 간 상관관계 분석을 추가해보세요.');
    }

    return recommendations;
  }

  collectArtifacts(results) {
    const artifacts = [];
    
    results.steps.forEach((step, index) => {
      if (step.success && step.result) {
        // 생성된 파일들 수집
        if (step.result.chart_path) {
          artifacts.push({
            type: 'visualization',
            name: `${step.method}_chart`,
            path: step.result.chart_path,
            stepNumber: index + 1
          });
        }
        
        if (step.result.model_path) {
          artifacts.push({
            type: 'model',
            name: `${step.method}_model`,
            path: step.result.model_path,
            stepNumber: index + 1
          });
        }
        
        if (step.result.report_path) {
          artifacts.push({
            type: 'report',
            name: `${step.method}_report`,
            path: step.result.report_path,
            stepNumber: index + 1
          });
        }
      }
    });

    return artifacts;
  }

  assessDataQuality(results) {
    // 데이터 품질 평가 로직
    let missingValues = 0;
    let dataShape = null;
    
    const dataLoadStep = results.steps.find(step =>
      step.type === 'data_loading' && step.success
    );
    
    if (dataLoadStep && dataLoadStep.result) {
      const stats = dataLoadStep.result.statistics;
      if (stats && stats.missing_percentage) {
        missingValues = stats.missing_percentage / 100;
      }
      dataShape = {
        rows: dataLoadStep.result.rowCount,
        columns: dataLoadStep.result.columnCount
      };
    }
    
    return {
      missingValues,
      dataShape,
      quality: missingValues < 0.05 ? 'high' : missingValues < 0.2 ? 'medium' : 'low'
    };
  }

  assessModelPerformance(results) {
    // 모델 성능 평가 로직
    let accuracy = null;
    let metrics = {};
    
    const mlSteps = results.steps.filter(step =>
      step.type === 'ml_traditional' && step.success
    );
    
    if (mlSteps.length > 0) {
      const lastMlStep = mlSteps[mlSteps.length - 1];
      if (lastMlStep.result && lastMlStep.result.metrics) {
        metrics = lastMlStep.result.metrics;
        accuracy = metrics.accuracy || metrics.r2_score || null;
      }
    }
    
    return {
      accuracy,
      metrics,
      performance: accuracy ? (accuracy > 0.8 ? 'high' : accuracy > 0.6 ? 'medium' : 'low') : 'unknown'
    };
  }

  // saveWorkflowResults 메서드 완성
  async saveWorkflowResults(results) {
    try {
      // 결과 저장소에 저장
      await this.resultStore.saveResults(results.sessionId, results);
      
      // 파일 시스템에도 저장
      const resultsDir = './results';
      await fs.mkdir(resultsDir, { recursive: true });
      
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const filename = `workflow_${results.sessionId}_${timestamp}.json`;
      const filepath = path.join(resultsDir, filename);
      
      await fs.writeFile(filepath, JSON.stringify(results, null, 2));
      
      this.logger.info('워크플로우 결과 저장 완료:', filepath);
    } catch (error) {
      this.logger.error('워크플로우 결과 저장 실패:', error);
      throw error;
    }
  }

  // generateVisualizationPythonCode 메서드 완성
  generateVisualizationPythonCode(vizConfig, params, intermediateResults) {
    const { script, chart_type, description } = vizConfig;
    
    return `
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 결과 디렉토리 설정
output_dir = './results'
os.makedirs(output_dir, exist_ok=True)

try:
    # 데이터 로드
    data_available = False
    df = None
    
    # 중간 결과에서 데이터 찾기
    ${JSON.stringify(intermediateResults)} # 중간 결과 데이터
    
    # 데이터 파일에서 로드
    if os.path.exists('temp/current_data.csv'):
        df = pd.read_csv('temp/current_data.csv')
        data_available = True
    elif os.path.exists('data/current_data.csv'):
        df = pd.read_csv('data/current_data.csv')
        data_available = True
    
    if not data_available:
        raise ValueError("시각화할 데이터를 찾을 수 없습니다.")
    
    # 차트 생성
    plt.figure(figsize=(12, 8))
    
    chart_type = '${chart_type}'
    if chart_type == '2d.scatter':
        # 산점도
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
            plt.scatter(df[x_col], df[y_col], alpha=0.6)
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f'{x_col} vs {y_col}')
        else:
            raise ValueError("산점도를 그리기 위한 숫자 컬럼이 부족합니다.")
    
    elif chart_type == '2d.line':
        # 선 그래프
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 1:
            y_col = numeric_cols[0]
            plt.plot(df.index, df[y_col], marker='o')
            plt.xlabel('Index')
            plt.ylabel(y_col)
            plt.title(f'{y_col} 추세')
        else:
            raise ValueError("선 그래프를 그리기 위한 숫자 컬럼이 부족합니다.")
    
    elif chart_type == '2d.bar':
        # 막대 그래프
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) >= 1:
            cat_col = categorical_cols[0]
            value_counts = df[cat_col].value_counts().head(10)
            plt.bar(range(len(value_counts)), value_counts.values)
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
            plt.xlabel(cat_col)
            plt.ylabel('Count')
            plt.title(f'{cat_col} 분포')
        else:
            raise ValueError("막대 그래프를 그리기 위한 범주형 컬럼이 부족합니다.")
    
    elif chart_type == '2d.histogram':
        # 히스토그램
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 1:
            col = numeric_cols[0]
            plt.hist(df[col].dropna(), bins=30, alpha=0.7)
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.title(f'{col} 분포')
        else:
            raise ValueError("히스토그램을 그리기 위한 숫자 컬럼이 부족합니다.")
    
    elif chart_type == '2d.heatmap':
        # 히트맵 (상관관계)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('상관관계 히트맵')
        else:
            raise ValueError("히트맵을 그리기 위한 숫자 컬럼이 부족합니다.")
    
    else:
        raise ValueError(f"지원하지 않는 차트 유형: {chart_type}")
    
    # 파일 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{chart_type.replace(".", "_")}_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 결과 반환
    result = {
        'success': True,
        'chart_type': chart_type,
        'chart_path': filepath,
        'description': '${description}',
        'data_summary': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns)
        }
    }
    
    print(json.dumps(result, ensure_ascii=False, default=str))
    
except Exception as e:
    error_result = {
        'success': False,
        'error': str(e),
        'error_type': type(e).__name__
    }
    print(json.dumps(error_result, ensure_ascii=False))
    raise
`;
  }

  // 결과 파싱 메서드들 완성
  parseAnalysisResult(result) {
    try {
      const parsed = JSON.parse(result.output);
      return {
        success: true,
        result: parsed,
        executionTime: result.executionTime || 0
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
        rawOutput: result.output
      };
    }
  }

  parseMLResult(result) {
    try {
      const parsed = JSON.parse(result.output);
      return {
        success: true,
        result: parsed,
        executionTime: result.executionTime || 0
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
        rawOutput: result.output
      };
    }
  }

  parseDeepLearningResult(result) {
    try {
      const parsed = JSON.parse(result.output);
      return {
        success: true,
        result: parsed,
        executionTime: result.executionTime || 0
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
        rawOutput: result.output
      };
    }
  }

  parseVisualizationResult(result) {
    try {
      const parsed = JSON.parse(result.output);
      return {
        success: true,
        result: parsed,
        executionTime: result.executionTime || 0
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
        rawOutput: result.output
      };
    }
  }

  // 워크플로우 실행 상태 확인
  getExecutionStatus() {
    return {
      isExecuting: this.isExecuting,
      currentSession: this.currentSession,
      startTime: this.currentSession ? new Date().toISOString() : null
    };
  }

  // 워크플로우 중단
  async cancelWorkflow() {
    if (!this.isExecuting) {
      throw new Error('실행 중인 워크플로우가 없습니다.');
    }

    try {
      // Python 프로세스 종료 시도
      // 실제 구현에서는 실행 중인 프로세스를 추적하고 종료해야 함
      
      this.isExecuting = false;
      this.currentSession = null;
      
      this.logger.info('워크플로우 실행이 취소되었습니다.');
      
      return {
        success: true,
        message: '워크플로우가 취소되었습니다.'
      };
    } catch (error) {
      this.logger.error('워크플로우 취소 실패:', error);
      throw error;
    }
  }
