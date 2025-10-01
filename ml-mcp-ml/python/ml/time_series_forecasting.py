#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Forecasting for ML MCP
ML MCP용 시계열 예측 스크립트
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import joblib
import warnings
warnings.filterwarnings('ignore')

# Time series and ML libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Statistical libraries
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.seasonal import seasonal_decompose

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('default')

# Add shared utilities to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "ml-mcp-shared" / "python"))

try:
    from common_utils import load_data, get_data_info, create_analysis_result, output_results, validate_required_params
except ImportError:
    # Fallback implementations
    def load_data(file_path: str) -> pd.DataFrame:
        file_path = Path(file_path)
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_path.suffix}")

    def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
        }

    def create_analysis_result(analysis_type: str, data_info: Dict[str, Any], results: Dict[str, Any], summary: str = None) -> Dict[str, Any]:
        return {
            "analysis_type": analysis_type,
            "data_info": data_info,
            "summary": summary or f"{analysis_type} 완료",
            **results
        }

    def output_results(results: Dict[str, Any]):
        print(json.dumps(results, ensure_ascii=False, indent=2))

    def validate_required_params(params: Dict[str, Any], required: list):
        missing = [param for param in required if param not in params]
        if missing:
            raise ValueError(f"필수 매개변수가 누락됨: {', '.join(missing)}")


def preprocess_time_series_data(df: pd.DataFrame, date_column: str, value_column: str) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Preprocess time series data
    시계열 데이터 전처리
    """
    # Validate columns
    if date_column not in df.columns:
        raise ValueError(f"날짜 컬럼 '{date_column}'이 데이터에 없습니다")
    if value_column not in df.columns:
        raise ValueError(f"값 컬럼 '{value_column}'이 데이터에 없습니다")

    # Copy dataframe
    data = df.copy()

    # Convert date column to datetime
    try:
        data[date_column] = pd.to_datetime(data[date_column])
    except Exception as e:
        raise ValueError(f"날짜 컬럼 변환 실패: {e}")

    # Convert value column to numeric
    try:
        data[value_column] = pd.to_numeric(data[value_column], errors='coerce')
    except Exception as e:
        raise ValueError(f"값 컬럼 변환 실패: {e}")

    # Remove rows with missing values
    data = data.dropna(subset=[date_column, value_column])

    if len(data) == 0:
        raise ValueError("전처리 후 데이터가 없습니다")

    # Sort by date
    data = data.sort_values(date_column)

    # Set date as index
    data.set_index(date_column, inplace=True)

    # Create time series
    ts = data[value_column]

    return ts, data


def analyze_time_series_properties(ts: pd.Series) -> Dict[str, Any]:
    """
    Analyze time series properties
    시계열 속성 분석
    """
    analysis = {}

    # Basic statistics
    analysis['basic_stats'] = {
        'length': len(ts),
        'mean': float(ts.mean()),
        'std': float(ts.std()),
        'min': float(ts.min()),
        'max': float(ts.max()),
        'missing_values': int(ts.isnull().sum())
    }

    # Date range
    analysis['date_range'] = {
        'start_date': ts.index.min().isoformat(),
        'end_date': ts.index.max().isoformat(),
        'duration_days': (ts.index.max() - ts.index.min()).days
    }

    # Frequency analysis
    try:
        freq = pd.infer_freq(ts.index)
        analysis['frequency'] = freq if freq else 'irregular'
    except:
        analysis['frequency'] = 'irregular'

    # Trend analysis
    try:
        # Simple linear trend
        x = np.arange(len(ts))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, ts.values)

        analysis['trend'] = {
            'slope': float(slope),
            'r_squared': float(r_value**2),
            'p_value': float(p_value),
            'direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
        }
    except:
        analysis['trend'] = {'error': '트렌드 분석 실패'}

    # Stationarity test (Augmented Dickey-Fuller)
    try:
        from statsmodels.tsa.stattools import adfuller
        adf_result = adfuller(ts.dropna())

        analysis['stationarity'] = {
            'adf_statistic': float(adf_result[0]),
            'p_value': float(adf_result[1]),
            'is_stationary': adf_result[1] < 0.05,
            'critical_values': {k: float(v) for k, v in adf_result[4].items()}
        }
    except:
        analysis['stationarity'] = {'error': '정상성 검정 실패'}

    return analysis


def decompose_time_series(ts: pd.Series, include_seasonality: bool = True) -> Dict[str, Any]:
    """
    Decompose time series into components
    시계열을 구성 요소로 분해
    """
    decomposition_results = {}

    try:
        if len(ts) >= 24 and include_seasonality:  # Need sufficient data for seasonal decomposition
            # Try different periods
            periods_to_try = [12, 7, 30, 4]  # Monthly, weekly, daily, quarterly

            for period in periods_to_try:
                if len(ts) >= 2 * period:
                    try:
                        decomposition = seasonal_decompose(ts, model='additive', period=period)

                        decomposition_results[f'period_{period}'] = {
                            'trend_strength': float(1 - np.var(decomposition.resid.dropna()) / np.var(decomposition.trend.dropna() + decomposition.resid.dropna())),
                            'seasonal_strength': float(1 - np.var(decomposition.resid.dropna()) / np.var(decomposition.seasonal.dropna() + decomposition.resid.dropna())),
                            'period': period
                        }
                        break  # Use the first successful decomposition
                    except:
                        continue

        if not decomposition_results:
            decomposition_results['note'] = '시계열 분해를 위한 충분한 데이터가 없습니다'

    except Exception as e:
        decomposition_results['error'] = f'시계열 분해 실패: {str(e)}'

    return decomposition_results


def create_lstm_features(ts: pd.Series, lookback: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create features for LSTM-like models
    LSTM형 모델을 위한 특성 생성
    """
    data = ts.values
    X, y = [], []

    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])

    return np.array(X), np.array(y)


def train_arima_model(ts: pd.Series) -> Tuple[Any, Dict[str, Any]]:
    """
    Train ARIMA model with automatic parameter selection
    자동 파라미터 선택으로 ARIMA 모델 훈련
    """
    try:
        # Auto ARIMA (simplified parameter search)
        best_aic = float('inf')
        best_params = (1, 1, 1)
        best_model = None

        # Simple grid search for small parameter space
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        model = ARIMA(ts, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_params = (p, d, q)
                            best_model = fitted_model
                    except:
                        continue

        if best_model is None:
            # Fallback to simple ARIMA(1,1,1)
            model = ARIMA(ts, order=(1, 1, 1))
            best_model = model.fit()
            best_params = (1, 1, 1)

        model_info = {
            'type': 'ARIMA',
            'parameters': best_params,
            'aic': float(best_model.aic),
            'bic': float(best_model.bic),
            'log_likelihood': float(best_model.llf)
        }

        return best_model, model_info

    except Exception as e:
        raise ValueError(f"ARIMA 모델 훈련 실패: {str(e)}")


def train_exponential_smoothing_model(ts: pd.Series, include_seasonality: bool = True) -> Tuple[Any, Dict[str, Any]]:
    """
    Train Exponential Smoothing model
    지수평활법 모델 훈련
    """
    try:
        # Determine seasonal period
        seasonal_period = None
        if include_seasonality and len(ts) >= 24:
            # Try to detect seasonality
            seasonal_period = min(12, len(ts) // 2)

        # Fit ETS model
        if seasonal_period:
            model = ETSModel(ts, trend='add', seasonal='add', seasonal_periods=seasonal_period)
        else:
            model = ETSModel(ts, trend='add', seasonal=None)

        fitted_model = model.fit()

        model_info = {
            'type': 'Exponential Smoothing',
            'trend': 'additive',
            'seasonal': 'additive' if seasonal_period else None,
            'seasonal_period': seasonal_period,
            'aic': float(fitted_model.aic),
            'bic': float(fitted_model.bic)
        }

        return fitted_model, model_info

    except Exception as e:
        raise ValueError(f"지수평활법 모델 훈련 실패: {str(e)}")


def train_lstm_model(ts: pd.Series, lookback: int = 10) -> Tuple[Any, Dict[str, Any]]:
    """
    Train LSTM-like model using MLPRegressor
    MLPRegressor를 사용한 LSTM형 모델 훈련
    """
    try:
        # Create features
        X, y = create_lstm_features(ts, lookback)

        if len(X) < 10:
            raise ValueError("LSTM 모델 훈련을 위한 데이터가 부족합니다")

        # Scale the data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Train neural network
        model = MLPRegressor(
            hidden_layer_sizes=(50, 25),
            activation='relu',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2
        )

        model.fit(X_scaled, y_scaled)

        model_info = {
            'type': 'LSTM (MLP)',
            'lookback_period': lookback,
            'hidden_layers': (50, 25),
            'training_samples': len(X),
            'scalers': {'X': scaler_X, 'y': scaler_y}
        }

        return model, model_info

    except Exception as e:
        raise ValueError(f"LSTM 모델 훈련 실패: {str(e)}")


def make_forecasts(model: Any, model_info: Dict[str, Any], ts: pd.Series, forecast_periods: int) -> np.ndarray:
    """
    Make forecasts using trained model
    훈련된 모델로 예측 수행
    """
    model_type = model_info['type']

    if model_type == 'ARIMA':
        forecast = model.forecast(steps=forecast_periods)
        return forecast

    elif model_type == 'Exponential Smoothing':
        forecast = model.forecast(steps=forecast_periods)
        return forecast

    elif model_type == 'LSTM (MLP)':
        lookback = model_info['lookback_period']
        scaler_X = model_info['scalers']['X']
        scaler_y = model_info['scalers']['y']

        # Use last 'lookback' values to predict
        last_values = ts.values[-lookback:]
        forecasts = []

        for _ in range(forecast_periods):
            # Prepare input
            X_input = last_values.reshape(1, -1)
            X_scaled = scaler_X.transform(X_input)

            # Make prediction
            y_scaled = model.predict(X_scaled)
            y_pred = scaler_y.inverse_transform(y_scaled.reshape(-1, 1))[0, 0]

            forecasts.append(y_pred)

            # Update last_values for next prediction
            last_values = np.roll(last_values, -1)
            last_values[-1] = y_pred

        return np.array(forecasts)

    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")


def evaluate_forecast_accuracy(actual: pd.Series, predicted: np.ndarray) -> Dict[str, float]:
    """
    Evaluate forecast accuracy
    예측 정확도 평가
    """
    # Align actual and predicted data
    min_length = min(len(actual), len(predicted))
    actual_aligned = actual.values[-min_length:]
    predicted_aligned = predicted[:min_length]

    # Calculate metrics
    mae = mean_absolute_error(actual_aligned, predicted_aligned)
    mse = mean_squared_error(actual_aligned, predicted_aligned)
    rmse = np.sqrt(mse)

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual_aligned - predicted_aligned) / np.maximum(np.abs(actual_aligned), 1e-8))) * 100

    # R-squared
    r2 = r2_score(actual_aligned, predicted_aligned)

    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'mape': float(mape),
        'r2_score': float(r2)
    }


def generate_forecast_plots(ts: pd.Series, forecasts: np.ndarray, forecast_periods: int,
                          model_type: str) -> List[str]:
    """
    Generate forecast visualization plots
    예측 시각화 플롯 생성
    """
    plot_files = []

    try:
        # Create forecast dates
        last_date = ts.index[-1]
        freq = pd.infer_freq(ts.index) or 'D'
        forecast_dates = pd.date_range(start=last_date, periods=forecast_periods + 1, freq=freq)[1:]

        # Plot historical data and forecast
        plt.figure(figsize=(12, 8))

        # Plot historical data (last 50 points for clarity)
        history_to_show = min(50, len(ts))
        plt.plot(ts.index[-history_to_show:], ts.values[-history_to_show:],
                label='Historical Data', color='blue', linewidth=2)

        # Plot forecast
        plt.plot(forecast_dates, forecasts, label='Forecast',
                color='red', linewidth=2, linestyle='--')

        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(f'Time Series Forecast - {model_type}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        plot_file = f'forecast_{model_type.lower().replace(" ", "_")}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot_file)

    except Exception as e:
        print(f"경고: 플롯 생성 중 오류 발생: {e}")

    return plot_files


def perform_time_series_forecasting(data_file: str, date_column: str, value_column: str,
                                   forecast_periods: int = 30, model_type: str = 'arima',
                                   include_seasonality: bool = True) -> Dict[str, Any]:
    """
    Perform comprehensive time series forecasting
    포괄적인 시계열 예측 수행
    """
    # Load and preprocess data
    df = load_data(data_file)
    ts, processed_data = preprocess_time_series_data(df, date_column, value_column)

    # Analyze time series properties
    ts_properties = analyze_time_series_properties(ts)

    # Decompose time series
    decomposition_results = decompose_time_series(ts, include_seasonality)

    # Split data for evaluation (80-20 split)
    split_point = int(len(ts) * 0.8)
    train_ts = ts[:split_point]
    test_ts = ts[split_point:]

    # Train model based on specified type
    if model_type.lower() == 'arima':
        model, model_info = train_arima_model(train_ts)
    elif model_type.lower() == 'exponential_smoothing':
        model, model_info = train_exponential_smoothing_model(train_ts, include_seasonality)
    elif model_type.lower() == 'lstm':
        model, model_info = train_lstm_model(train_ts)
    else:
        # Default to ARIMA
        model, model_info = train_arima_model(train_ts)
        model_type = 'arima'

    # Make in-sample predictions for evaluation
    evaluation_metrics = {}
    if len(test_ts) > 0:
        test_forecasts = make_forecasts(model, model_info, train_ts, len(test_ts))
        evaluation_metrics = evaluate_forecast_accuracy(test_ts, test_forecasts)

    # Make future forecasts
    future_forecasts = make_forecasts(model, model_info, ts, forecast_periods)

    # Create forecast dates
    last_date = ts.index[-1]
    freq = pd.infer_freq(ts.index) or 'D'
    forecast_dates = pd.date_range(start=last_date, periods=forecast_periods + 1, freq=freq)[1:]

    # Generate plots
    plot_files = generate_forecast_plots(ts, future_forecasts, forecast_periods, model_info['type'])

    # Save results
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'forecast': future_forecasts
    })
    forecast_file = f"time_series_forecast_{model_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    forecast_df.to_csv(forecast_file, index=False)

    # Save model
    model_file = f"time_series_model_{model_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.joblib"
    joblib.dump({
        'model': model,
        'model_info': model_info,
        'ts_properties': ts_properties,
        'original_data': processed_data
    }, model_file)

    # Prepare results
    results = {
        'model_type': model_info['type'],
        'forecast_periods': forecast_periods,
        'data_points': len(ts),
        'train_period': f"{train_ts.index[0].date()} to {train_ts.index[-1].date()}",
        'forecast_file': forecast_file,
        'model_file': model_file,

        # Time series properties
        'time_series_properties': ts_properties,
        'decomposition_results': decomposition_results,

        # Model information
        'model_details': model_info,

        # Forecasts
        'forecasts': future_forecasts.tolist(),
        'forecast_dates': [d.isoformat() for d in forecast_dates],

        # Evaluation metrics
        'evaluation_metrics': evaluation_metrics,
        'plot_files': plot_files
    }

    return results


def main():
    """메인 실행 함수"""
    try:
        # Read input from stdin
        input_data = sys.stdin.read()
        options = json.loads(input_data)

        # Validate required parameters
        validate_required_params(options, ['data_file', 'date_column', 'value_column'])

        # Extract parameters
        data_file = options['data_file']
        date_column = options['date_column']
        value_column = options['value_column']
        forecast_periods = options.get('forecast_periods', 30)
        model_type = options.get('model_type', 'arima')
        include_seasonality = options.get('include_seasonality', True)

        # Perform time series forecasting
        results = perform_time_series_forecasting(
            data_file=data_file,
            date_column=date_column,
            value_column=value_column,
            forecast_periods=forecast_periods,
            model_type=model_type,
            include_seasonality=include_seasonality
        )

        # Get data info for final result
        df = load_data(data_file)
        data_info = get_data_info(df)

        # Create final result
        final_result = create_analysis_result(
            analysis_type="time_series_forecasting",
            data_info=data_info,
            results=results,
            summary=f"{results['model_type']} 시계열 예측 완료 - {forecast_periods}기간 예측"
        )

        # Output results
        output_results(final_result)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "analysis_type": "time_series_forecasting"
        }
        output_results(error_result)
        sys.exit(1)


if __name__ == "__main__":
    main()