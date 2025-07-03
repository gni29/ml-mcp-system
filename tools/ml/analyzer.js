import { Logger } from '../../utils/logger.js';
import { PythonExecutor } from '../common/python-executor.js';
import { ResultFormatter } from '../common/result-formatter.js';

export class Analyzer {
  constructor() {
    this.logger = new Logger();
    this.pythonExecutor = new PythonExecutor();
    this.resultFormatter = new ResultFormatter();
  }

  async initialize() {
    await this.pythonExecutor.initialize();
    this.logger.info('Analyzer 초기화 완료');
  }

  async analyzeData(filePath, analysisType, parameters = {}) {
    try {
      this.logger.info(`데이터 분석 시작: ${analysisType}`);
      
      const pythonCode = this.generateAnalysisCode(filePath, analysisType, parameters);
      const result = await this.pythonExecutor.execute(pythonCode);
      
      if (result.success) {
        const parsedResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(parsedResult, analysisType);
      } else {
        throw new Error(result.error);
      }
    } catch (error) {
      this.logger.error('데이터 분석 실패:', error);
      throw error;
    }
  }

  generateAnalysisCode(filePath, analysisType, parameters) {
    const baseCode = `
import pandas as pd
import numpy as np
import json
from pathlib import Path

# 데이터 로드
df = pd.read_csv('${filePath}')
result = {}
`;

    switch (analysisType) {
      case 'descriptive_stats':
        return baseCode + `
# 기본 통계 분석
statistics = {}
for col in df.select_dtypes(include=[np.number]).columns:
    statistics[col] = {
        'mean': df[col].mean(),
        'std': df[col].std(),
        'min': df[col].min(),
        'max': df[col].max(),
        'median': df[col].median(),
        'q25': df[col].quantile(0.25),
        'q75': df[col].quantile(0.75)
    }

result['statistics'] = statistics
result['summary'] = f"총 {len(df)}개 행, {len(df.columns)}개 열의 데이터를 분석했습니다."
print(json.dumps(result, default=str))
`;

      case 'correlation':
        return baseCode + `
# 상관관계 분석
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

# 강한 상관관계 찾기
strong_correlations = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_value = correlation_matrix.iloc[i, j]
        if abs(corr_value) > 0.7:
            strong_correlations.append({
                'var1': correlation_matrix.columns[i],
                'var2': correlation_matrix.columns[j],
                'correlation': corr_value
            })

result['correlation_matrix'] = correlation_matrix.to_dict()
result['strong_correlations'] = strong_correlations
result['summary'] = f"{len(strong_correlations)}개의 강한 상관관계를 발견했습니다."
print(json.dumps(result, default=str))
`;

      default:
        throw new Error(`지원하지 않는 분석 타입: ${analysisType}`);
    }
  }
}
