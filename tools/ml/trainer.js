import { Logger } from '../../utils/logger.js';
import { PythonExecutor } from '../common/python-executor.js';
import { ResultFormatter } from '../common/result-formatter.js';

export class Trainer {
  constructor() {
    this.logger = new Logger();
    this.pythonExecutor = new PythonExecutor();
    this.resultFormatter = new ResultFormatter();
  }

  async initialize() {
    await this.pythonExecutor.initialize();
    this.logger.info('Trainer 초기화 완료');
  }

  async trainModel(filePath, modelType, targetColumn, parameters = {}) {
    try {
      this.logger.info(`모델 훈련 시작: ${modelType}`);
      
      const pythonCode = this.generateTrainingCode(filePath, modelType, targetColumn, parameters);
      const result = await this.pythonExecutor.execute(pythonCode, { timeout: 600000 }); // 10분
      
      if (result.success) {
        const parsedResult = JSON.parse(result.output);
        return this.resultFormatter.formatAnalysisResult(parsedResult, 'ml_model');
      } else {
        throw new Error(result.error);
      }
    } catch (error) {
      this.logger.error('모델 훈련 실패:', error);
      throw error;
    }
  }

  generateTrainingCode(filePath, modelType, targetColumn, parameters) {
    const {
      test_size = 0.2,
      random_state = 42,
      cross_validation = false,
      cv_folds = 5
    } = parameters;

    const baseCode = `
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드
df = pd.read_csv('${filePath}')

# 타겟 변수 분리
if '${targetColumn}' not in df.columns:
    raise ValueError(f"타겟 컬럼 '${targetColumn}'을 찾을 수 없습니다.")

X = df.drop(columns=['${targetColumn}'])
y = df['${targetColumn}']

# 범주형 변수 인코딩
categorical_columns = X.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# 타겟 변수가 범주형인 경우 인코딩
if y.dtype == 'object':
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    task_type = 'classification'
else:
    task_type = 'regression'

# 숫자형 변수 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 훈련/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=${test_size}, random_state=${random_state}
)
`;

    switch (modelType) {
      case 'linear_regression':
        return baseCode + `
from sklearn.linear_model import LinearRegression

# 모델 훈련
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
metrics = {
    'mse': mean_squared_error(y_test, y_pred),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
    'mae': mean_absolute_error(y_test, y_pred),
    'r2_score': r2_score(y_test, y_pred)
}

result = {
    'model_type': 'Linear Regression',
    'metrics': metrics,
    'predictions_sample': y_pred[:10].tolist(),
    'actual_sample': y_test[:10].tolist(),
    'summary': f"선형 회귀 모델 훈련 완료. R² 점수: {metrics['r2_score']:.4f}"
}

print(json.dumps(result, default=str))
`;

      case 'random_forest':
        return baseCode + `
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# 모델 선택
if task_type == 'classification':
    model = RandomForestClassifier(n_estimators=100, random_state=${random_state})
else:
    model = RandomForestRegressor(n_estimators=100, random_state=${random_state})

# 모델 훈련
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 피처 중요도
feature_importance = [
    {'feature': X.columns[i], 'importance': importance}
    for i, importance in enumerate(model.feature_importances_)
]
feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)

# 평가
if task_type == 'classification':
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
else:
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2_score': r2_score(y_test, y_pred)
    }

result = {
    'model_type': 'Random Forest',
    'task_type': task_type,
    'metrics': metrics,
    'feature_importance': feature_importance,
    'predictions_sample': y_pred[:10].tolist(),
    'actual_sample': y_test[:10].tolist(),
    'summary': f"랜덤 포레스트 모델 훈련 완료. 상위 3개 중요 피처: {', '.join([f['feature'] for f in feature_importance[:3]])}"
}

print(json.dumps(result, default=str))
`;

      case 'xgboost':
        return baseCode + `
try:
    import xgboost as xgb
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    raise ImportError("XGBoost가 설치되지 않았습니다. 'pip install xgboost'로 설치하세요.")

# 모델 선택
if task_type == 'classification':
    model = XGBClassifier(n_estimators=100, random_state=${random_state})
else:
    model = XGBRegressor(n_estimators=100, random_state=${random_state})

# 모델 훈련
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 피처 중요도
feature_importance = [
    {'feature': X.columns[i], 'importance': importance}
    for i, importance in enumerate(model.feature_importances_)
]
feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)

# 평가
if task_type == 'classification':
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
else:
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2_score': r2_score(y_test, y_pred)
    }

result = {
    'model_type': 'XGBoost',
    'task_type': task_type,
    'metrics': metrics,
    'feature_importance': feature_importance,
    'predictions_sample': y_pred[:10].tolist(),
    'actual_sample': y_test[:10].tolist(),
    'summary': f"XGBoost 모델 훈련 완료. 최고 성능 피처: {feature_importance[0]['feature']}"
}

print(json.dumps(result, default=str))
`;

      default:
        throw new Error(`지원하지 않는 모델 타입: ${modelType}`);
    }
  }

  async compareModels(filePath, targetColumn, modelTypes, parameters = {}) {
    try {
      this.logger.info('모델 비교 시작');
      
      const results = [];
      
      for (const modelType of modelTypes) {
        const result = await this.trainModel(filePath, modelType, targetColumn, parameters);
        results.push({
          modelType,
          result
        });
      }
      
      return this.formatComparisonResults(results);
    } catch (error) {
      this.logger.error('모델 비교 실패:', error);
      throw error;
    }
  }

  formatComparisonResults(results) {
    // 모델별 성능 점수 추출 및 순위 매기기
    const comparison = results.map(({ modelType, result }) => {
      const metrics = result.metadata ? result.result?.metrics : result.metrics;
      const mainMetric = metrics?.accuracy || metrics?.r2_score || 0;
      
      return {
        name: modelType,
        score: mainMetric,
        fullResult: result
      };
    }).sort((a, b) => b.score - a.score);

    return this.resultFormatter.formatComparisonResult({
      comparisonType: '모델 성능',
      results: comparison,
      winner: comparison[0]
    });
  }
}
