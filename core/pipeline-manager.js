// core/pipeline-manager.js
import { Logger } from '../utils/logger.js';
import { PythonExecutor } from '../tools/common/python-executor.js';
import { ResultStore } from './result-store.js';
import path from 'path';
import fs from 'fs/promises';

export class PipelineManager {
  constructor(smartRouter) {
    this.smartRouter = smartRouter;
    this.logger = new Logger();
    this.pythonExecutor = new PythonExecutor();
    this.resultStore = new ResultStore();
    this.currentSession = null;
    this.isExecuting = false;
  }

  async initialize() {
    try {
      await this.pythonExecutor.initialize();
      this.logger.info('PipelineManager 초기화 완료');
    } catch (error) {
      this.logger.error('PipelineManager 초기화 실패:', error);
      throw error;
    }
  }

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

  async executeStep(step, intermediateResults, stepNumber) {
    const stepStartTime = Date.now();
    
    try {
      this.logger.info(`단계 ${stepNumber} 실행 시작`, {
        type: step.type,
        method: step.method
      });

      let result;
      
      switch (step.type) {
        case 'basic':
        case 'advanced':
        case 'timeseries':
          result = await this.executeAnalysisStep(step, intermediateResults);
          break;
        case 'ml_traditional':
          result = await this.executeMLStep(step, intermediateResults);
          break;
        case 'deep_learning':
          result = await this.executeDeepLearningStep(step, intermediateResults);
          break;
        case 'visualization':
          result = await this.executeVisualizationStep(step, intermediateResults);
          break;
        default:
          throw new Error(`알 수 없는 단계 타입: ${step.type}`);
      }

      const stepResult = {
        stepNumber,
        type: step.type,
        method: step.method,
        params: step.params,
        success: true,
        result: result,
        executionTime: Date.now() - stepStartTime,
        timestamp: new Date().toISOString()
      };

      this.logger.info(`단계 ${stepNumber} 실행 완료`, {
        executionTime: stepResult.executionTime
      });

      return stepResult;

    } catch (error) {
      this.logger.error(`단계 ${stepNumber} 실행 실패:`, error);
      
      return {
        stepNumber,
        type: step.type,
        method: step.method,
        params: step.params,
        success: false,
        error: error.message,
        executionTime: Date.now() - stepStartTime,
        timestamp: new Date().toISOString()
      };
    }
  }

  async executeAnalysisStep(step, intermediateResults) {
    const methodConfig = this.smartRouter.getMethodConfig(step.type, step.method);
    
    if (!methodConfig) {
      throw new Error(`설정을 찾을 수 없습니다: ${step.type}.${step.method}`);
    }

    // Python 스크립트 실행
    const pythonCode = this.generateAnalysisPythonCode(methodConfig, step.params, intermediateResults);
    const result = await this.pythonExecutor.execute(pythonCode);

    return this.parseAnalysisResult(result);
  }

  async executeMLStep(step, intermediateResults) {
    const methodConfig = this.smartRouter.getMethodConfig(step.type, step.method);
    
    if (!methodConfig) {
      throw new Error(`ML 설정을 찾을 수 없습니다: ${step.type}.${step.method}`);
    }

    // ML Python 스크립트 실행
    const pythonCode = this.generateMLPythonCode(methodConfig, step.params, intermediateResults);
    const result = await this.pythonExecutor.execute(pythonCode);

    return this.parseMLResult(result);
  }

  async executeDeepLearningStep(step, intermediateResults) {
    const [domain, task] = step.method.split('.');
    const dlConfig = this.smartRouter.getDeepLearningConfig(domain, task);
    
    if (!dlConfig) {
      throw new Error(`딥러닝 설정을 찾을 수 없습니다: ${domain}.${task}`);
    }

    // 딥러닝 모드 결정 (training vs inference)
    const mode = step.params.mode || 'training';
    const scriptPath = dlConfig[mode];

    if (!scriptPath) {
      throw new Error(`딥러닝 스크립트를 찾을 수 없습니다: ${mode}`);
    }

    // 딥러닝 Python 스크립트 실행
    const pythonCode = this.generateDeepLearningPythonCode(scriptPath, step.params, intermediateResults);
    const result = await this.pythonExecutor.execute(pythonCode, {
      timeout: 600000 // 10분 타임아웃
    });

    return this.parseDeepLearningResult(result);
  }

  async executeVisualizationStep(step, intermediateResults) {
    const vizConfig = this.smartRouter.getVisualizationConfig(step.method);
    
    if (!vizConfig) {
      throw new Error(`시각화 설정을 찾을 수 없습니다: ${step.method}`);
    }

    // 시각화 Python 스크립트 실행
    const pythonCode = this.generateVisualizationPythonCode(vizConfig, step.params, intermediateResults);
    const result = await this.pythonExecutor.execute(pythonCode);

    return this.parseVisualizationResult(result);
  }

  generateAnalysisPythonCode(methodConfig, params, intermediateResults) {
    const { script, class: className, method } = methodConfig;
    
    return `
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# 스크립트 경로 추가
sys.path.append('${path.dirname(script)}')

# 클래스 import
from ${path.basename(script, '.py')} import ${className}

# 데이터 로드
data_path = "${params.data_path || 'data/current_data.csv'}"
if Path(data_path).exists():
    df = pd.read_csv(data_path)
else:
    # 중간 결과에서 데이터 로드
    df = pd.read_csv('temp/intermediate_data.csv')

# 분석 실행
analyzer = ${className}()
result = analyzer.${method}(df, **${JSON.stringify(params)})

# 결과 저장
output = {
    'success': True,
    'result': result,
    'data_shape': df.shape,
    'columns': df.columns.tolist()
}

print(json.dumps(output, default=str))
    `;
  }

  generateMLPythonCode(methodConfig, params, intermediateResults) {
    const { script, class: className, method } = methodConfig;
    
    return `
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, regression_report

# 스크립트 경로 추가
sys.path.append('${path.dirname(script)}')

# 클래스 import
from ${path.basename(script, '.py')} import ${className}

# 데이터 로드
data_path = "${params.data_path || 'data/current_data.csv'}"
df = pd.read_csv(data_path)

# 타겟 변수 분리
target_column = "${params.target_column || 'target'}"
if target_column not in df.columns:
    # 자동으로 타겟 변수 찾기
    target_column = df.columns[-1]

X = df.drop(columns=[target_column])
y = df[target_column]

# 훈련/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 모델 훈련 및 평가
model = ${className}()
result = model.${method}(X_train, X_test, y_train, y_test, **${JSON.stringify(params)})

# 결과 저장
output = {
    'success': True,
    'result': result,
    'data_shape': df.shape,
    'features': X.columns.tolist(),
    'target': target_column
}

print(json.dumps(output, default=str))
    `;
  }

  generateDeepLearningPythonCode(scriptPath, params, intermediateResults) {
    return `
import sys
import json
import torch
from pathlib import Path

# 스크립트 경로 추가
sys.path.append('${path.dirname(scriptPath)}')

# 메인 실행 함수 import
from ${path.basename(scriptPath, '.py')} import main

# 설정 파라미터
config = ${JSON.stringify(params)}

# 딥러닝 실행
try:
    result = main(config)
    output = {
        'success': True,
        'result': result
    }
except Exception as e:
    output = {
        'success': False,
        'error': str(e)
    }

print(json.dumps(output, default=str))
    `;
  }

  generateVisualizationPythonCode(vizConfig, params, intermediateResults) {
    const { script, class: className, method } = vizConfig;
    
    return `
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 스크립트 경로 추가
sys.path.append('${path.dirname(script)}')

# 클래스 import
from ${path.basename(script, '.py')} import ${className}

# 데이터 로드
data_path = "${params.data_path || 'data/current_data.csv'}"
df = pd.read_csv(data_path)

# 시각화 생성
visualizer = ${className}()
chart_path = visualizer.${method}(df, **${JSON.stringify(params)})

# 결과 저장
output = {
    'success': True,
    'chart_path': chart_path,
    'data_shape': df.shape
}

print(json.dumps(output, default=str))
    `;
  }

  parseAnalysisResult(result) {
    try {
      const parsed = JSON.parse(result);
      return parsed;
    } catch (error) {
      return {
        success: false,
        error: '분석 결과 파싱 실패',
        raw_output: result
      };
    }
  }

  parseMLResult(result) {
    try {
      const parsed = JSON.parse(result);
      return parsed;
    } catch (error) {
      return {
        success: false,
        error: 'ML 결과 파싱 실패',
        raw_output: result
      };
    }
  }

  parseDeepLearningResult(result) {
    try {
      const parsed = JSON.parse(result);
      return parsed;
    } catch (error) {
      return {
        success: false,
        error: '딥러닝 결과 파싱 실패',
        raw_output: result
      };
    }
  }

  parseVisualizationResult(result) {
    try {
      const parsed = JSON.parse(result);
      return parsed;
    } catch (error) {
      return {
        success: false,
        error: '시각화 결과 파싱 실패',
        raw_output: result
      };
    }
  }

  async generateFinalResult(workflowResults) {
    const { steps, intermediateResults, workflowName } = workflowResults;
    
    // 성공한 단계들만 필터링
    const successfulSteps = steps.filter(step => step.success);
    
    if (successfulSteps.length === 0) {
      return {
        success: false,
        message: '모든 단계가 실패했습니다.',
        errors: steps.map(step => step.error).filter(Boolean)
      };
    }

    // 최종 결과 요약 생성
    const summary = {
      success: true,
      workflowName,
      completedSteps: successfulSteps.length,
      totalSteps: steps.length,
      results: {},
      visualizations: [],
      reports: []
    };

    // 각 단계별 결과 수집
    for (const step of successfulSteps) {
      if (step.type === 'visualization' && step.result.chart_path) {
        summary.visualizations.push({
          type: step.method,
          path: step.result.chart_path,
          description: `${step.method} 시각화`
        });
      } else {
        summary.results[`${step.type}_${step.method}`] = step.result;
      }
    }

    // 요약 리포트 생성
    summary.reports.push(await this.generateWorkflowReport(workflowResults));

    return summary;
  }

  async generateWorkflowReport(workflowResults) {
    const { workflowName, steps, executionTime } = workflowResults;
    
    let report = `# ${workflowName} 실행 보고서\n\n`;
    report += `실행 시간: ${executionTime}ms\n`;
    report += `완료된 단계: ${steps.filter(s => s.success).length}/${steps.length}\n\n`;
    
    report += `## 단계별 결과\n\n`;
    
    for (const step of steps) {
      report += `### ${step.stepNumber}. ${step.type} - ${step.method}\n`;
      report += `- 상태: ${step.success ? '성공' : '실패'}\n`;
      report += `- 실행 시간: ${step.executionTime}ms\n`;
      
      if (step.success) {
        report += `- 결과: 성공적으로 완료\n`;
      } else {
        report += `- 오류: ${step.error}\n`;
      }
      report += `\n`;
    }
    
    return report;
  }

  async saveWorkflowResults(results) {
    try {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const sessionDir = `./results/${results.sessionId}_${timestamp.split('T')[0]}`;
      
      await fs.mkdir(sessionDir, { recursive: true });
      
      // 워크플로우 결과 저장
      const resultFile = path.join(sessionDir, `workflow_result_${timestamp}.json`);
      await fs.writeFile(resultFile, JSON.stringify(results, null, 2));
      
      // 요약 리포트 저장
      if (results.finalResult && results.finalResult.reports) {
        const reportFile = path.join(sessionDir, `workflow_report_${timestamp}.md`);
        await fs.writeFile(reportFile, results.finalResult.reports[0]);
      }
      
      this.logger.info('워크플로우 결과 저장 완료', {
        sessionDir,
        resultFile
      });
      
    } catch (error) {
      this.logger.error('워크플로우 결과 저장 실패:', error);
    }
  }

  // 실행 중인 워크플로우 취소
  cancelCurrentWorkflow() {
    if (this.isExecuting) {
      this.logger.info('워크플로우 취소 요청');
      // Python 프로세스 종료 로직 추가 가능
      this.isExecuting = false;
      return true;
    }
    return false;
  }

  // 워크플로우 상태 확인
  getWorkflowStatus() {
    return {
      isExecuting: this.isExecuting,
      currentSession: this.currentSession,
      uptime: process.uptime()
    };
  }
}
