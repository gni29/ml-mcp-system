import { Logger } from '../../utils/logger.js';
import { PythonExecutor } from '../common/python-executor.js';
import { ResultFormatter } from '../common/result-formatter.js';

export class Visualizer {
  constructor() {
    this.logger = new Logger();
    this.pythonExecutor = new PythonExecutor();
    this.resultFormatter = new ResultFormatter();
  }

  async initialize() {
    await this.pythonExecutor.initialize();
    this.logger.info('Visualizer 초기화 완료');
  }

  async createVisualization(filePath, chartType, parameters = {}) {
    try {
      this.logger.info(`시각화 생성 시작: ${chartType}`);
      
      const pythonCode = this.generateVisualizationCode(filePath, chartType, parameters);
      const result = await this.pythonExecutor.execute(pythonCode);
      
      if (result.success) {
        const parsedResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(parsedResult, 'visualization');
      } else {
        throw new Error(result.error);
      }
    } catch (error) {
      this.logger.error('시각화 생성 실패:', error);
      throw error;
    }
  }

  generateVisualizationCode(filePath, chartType, parameters) {
    const {
      x_column,
      y_column,
      color_column,
      title = '',
      figsize = [10, 8],
      output_dir = './results'
    } = parameters;

    const baseCode = `
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

# 출력 디렉토리 생성
os.makedirs('${output_dir}', exist_ok=True)

# 데이터 로드
df = pd.read_csv('${filePath}')

# 한글 폰트 설정 (시스템에 따라 조정 필요)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 그래프 크기 설정
plt.figure(figsize=(${figsize[0]}, ${figsize[1]}))
`;

    switch (chartType) {
      case 'scatter':
        return baseCode + `
# 산점도 생성
if '${color_column}' and '${color_column}' in df.columns:
    scatter = plt.scatter(df['${x_column}'], df['${y_column}'], 
                         c=df['${color_column}'], alpha=0.7, cmap='viridis')
    plt.colorbar(scatter, label='${color_column}')
else:
    plt.scatter(df['${x_column}'], df['${y_column}'], alpha=0.7)

plt.xlabel('${x_column}')
plt.ylabel('${y_column}')
plt.title('${title}' if '${title}' else f'{x_column} vs ${y_column}')
plt.grid(True, alpha=0.3)

# 파일 저장
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'scatter_{timestamp}.png'
filepath = os.path.join('${output_dir}', filename)
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()

result = {
    'chart_type': 'scatter',
    'chart_path': filepath,
    'data_summary': {
        'total_points': len(df),
        'columns': ['${x_column}', '${y_column}']
    }
}

print(json.dumps(result))
`;

      case 'line':
        return baseCode + `
# 선 그래프 생성
plt.plot(df['${x_column}'], df['${y_column}'], marker='o', linewidth=2, markersize=4)

plt.xlabel('${x_column}')
plt.ylabel('${y_column}')
plt.title('${title}' if '${title}' else f'{y_column} over {x_column}')
plt.grid(True, alpha=0.3)

# 파일 저장
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'line_{timestamp}.png'
filepath = os.path.join('${output_dir}', filename)
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()

result = {
    'chart_type': 'line',
    'chart_path': filepath,
    'data_summary': {
        'total_points': len(df),
        'columns': ['${x_column}', '${y_column}']
    }
}

print(json.dumps(result))
`;

      case 'heatmap':
        return baseCode + `
# 숫자형 컬럼만 선택하여 상관관계 매트릭스 생성
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()

# 히트맵 생성
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})

plt.title('${title}' if '${title}' else 'Correlation Heatmap')
plt.tight_layout()

# 파일 저장
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'heatmap_{timestamp}.png'
filepath = os.path.join('${output_dir}', filename)
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()

result = {
    'chart_type': 'heatmap',
    'chart_path': filepath,
    'data_summary': {
        'total_points': len(df),
        'columns': numeric_df.columns.tolist()
    }
}

print(json.dumps(result))
`;

      case 'histogram':
        return baseCode + `
# 히스토그램 생성
plt.hist(df['${x_column}'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')

plt.xlabel('${x_column}')
plt.ylabel('Frequency')
plt.title('${title}' if '${title}' else f'Distribution of {x_column}')
plt.grid(True, alpha=0.3)

# 파일 저장
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'histogram_{timestamp}.png'
filepath = os.path.join('${output_dir}', filename)
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()

result = {
    'chart_type': 'histogram',
    'chart_path': filepath,
    'data_summary': {
        'total_points': len(df),
        'columns': ['${x_column}']
    }
}

print(json.dumps(result))
`;

      case 'boxplot':
        return baseCode + `
# 박스플롯 생성
if '${color_column}' and '${color_column}' in df.columns:
    sns.boxplot(data=df, x='${color_column}', y='${y_column}')
else:
    plt.boxplot(df['${y_column}'])
    plt.ylabel('${y_column}')

plt.title('${title}' if '${title}' else f'Box Plot of {y_column}')
plt.grid(True, alpha=0.3)

# 파일 저장
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'boxplot_{timestamp}.png'
filepath = os.path.join('${output_dir}', filename)
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()

result = {
    'chart_type': 'boxplot',
    'chart_path': filepath,
    'data_summary': {
        'total_points': len(df),
        'columns': ['${y_column}']
    }
}

print(json.dumps(result))
`;

      default:
        throw new Error(`지원하지 않는 차트 타입: ${chartType}`);
    }
  }

  async createMultipleCharts(filePath, chartConfigs) {
    try {
      const results = [];
      
      for (const config of chartConfigs) {
        const result = await this.createVisualization(filePath, config.type, config.parameters);
        results.push(result);
      }
      
      return {
        content: [{
          type: 'text',
          text: `📊 **${chartConfigs.length}개 차트 생성 완료**\n\n` +
                results.map((r, i) => `${i + 1}. ${chartConfigs[i].type} 차트`).join('\n')
        }],
        charts: results,
        metadata: {
          analysisType: 'multiple_visualization',
          chartCount: chartConfigs.length,
          timestamp: new Date().toISOString()
        }
      };
    } catch (error) {
      this.logger.error('다중 차트 생성 실패:', error);
      throw error;
    }
  }
}
