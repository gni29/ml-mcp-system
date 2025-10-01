#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Time Series Forecasting
고급 시계열 예측

이 모듈은 고급 시계열 예측 모델을 구현합니다.
주요 기능:
- ARIMA 자동 모델 선택
- Prophet을 이용한 계절성 분해
- LSTM 신경망 예측
- 앙상블 예측
- 예측 불확실성 분석
- 한국어 해석 및 인사이트
"""

import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import MinMaxScaler
    import joblib
except ImportError as e:
    print(json.dumps({
        "success": False,
        "error": f"기본 라이브러리 누락: {str(e)}",
        "required_packages": ["scikit-learn", "joblib"]
    }, ensure_ascii=False))
    sys.exit(1)

# Optional dependencies with graceful fallback
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

def clean_dict_for_json(obj):
    """JSON 직렬화를 위한 딕셔너리 정리"""
    if isinstance(obj, dict):
        return {str(k): clean_dict_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_dict_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return [clean_dict_for_json(item) for item in obj]
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

def check_stationarity(ts: pd.Series) -> Dict[str, Any]:
    """
    시계열 정상성 검정

    Parameters:
    -----------
    ts : pd.Series
        시계열 데이터

    Returns:
    --------
    Dict[str, Any]
        정상성 검정 결과
    """
    try:
        if not ARIMA_AVAILABLE:
            return {"stationary": False, "method": "unavailable"}

        result = adfuller(ts.dropna())

        return {
            "stationary": result[1] < 0.05,  # p-value < 0.05면 정상성
            "adf_statistic": float(result[0]),
            "p_value": float(result[1]),
            "critical_values": {str(k): float(v) for k, v in result[4].items()},
            "method": "Augmented Dickey-Fuller"
        }
    except Exception as e:
        return {"stationary": False, "error": str(e), "method": "failed"}

def auto_arima_selection(ts: pd.Series, max_p: int = 5, max_q: int = 5, max_d: int = 2) -> Dict[str, Any]:
    """
    자동 ARIMA 모델 선택

    Parameters:
    -----------
    ts : pd.Series
        시계열 데이터
    max_p : int
        최대 AR 차수
    max_q : int
        최대 MA 차수
    max_d : int
        최대 차분 차수

    Returns:
    --------
    Dict[str, Any]
        최적 ARIMA 모델 정보
    """
    if not ARIMA_AVAILABLE:
        return {"error": "ARIMA 라이브러리가 설치되지 않았습니다"}

    best_aic = float('inf')
    best_order = None
    best_model = None
    models_tested = []

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(ts, order=(p, d, q))
                    fitted_model = model.fit()
                    aic = fitted_model.aic

                    models_tested.append({
                        "order": (p, d, q),
                        "aic": float(aic),
                        "bic": float(fitted_model.bic)
                    })

                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                        best_model = fitted_model

                except Exception:
                    continue

    if best_model is None:
        return {"error": "적절한 ARIMA 모델을 찾을 수 없습니다"}

    return {
        "best_order": best_order,
        "best_aic": float(best_aic),
        "best_bic": float(best_model.bic),
        "model": best_model,
        "models_tested": models_tested,
        "total_models_tested": len(models_tested)
    }

def create_lstm_model(lookback: int, features: int = 1) -> object:
    """
    LSTM 모델 생성

    Parameters:
    -----------
    lookback : int
        입력 시퀀스 길이
    features : int
        특성 수

    Returns:
    --------
    LSTM 모델
    """
    if not TENSORFLOW_AVAILABLE:
        return None

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, features)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_lstm_data(data: np.ndarray, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    LSTM용 데이터 준비

    Parameters:
    -----------
    data : np.ndarray
        시계열 데이터
    lookback : int
        입력 시퀀스 길이

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        X, y 데이터
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def advanced_time_series_forecast(df: pd.DataFrame, date_column: str, value_column: str,
                                forecast_periods: int = 30, models: List[str] = ["arima", "prophet", "lstm"],
                                train_ratio: float = 0.8, lookback_window: int = 60,
                                confidence_level: float = 0.95, ensemble_method: str = "average",
                                model_save_path: str = "time_series_models.pkl") -> Dict[str, Any]:
    """
    고급 시계열 예측

    Parameters:
    -----------
    df : pd.DataFrame
        입력 데이터프레임
    date_column : str
        날짜 컬럼명
    value_column : str
        예측할 값 컬럼명
    forecast_periods : int
        예측 기간
    models : List[str]
        사용할 모델 목록
    train_ratio : float
        훈련 데이터 비율
    lookback_window : int
        LSTM 입력 윈도우 크기
    confidence_level : float
        신뢰구간 수준
    ensemble_method : str
        앙상블 방법
    model_save_path : str
        모델 저장 경로

    Returns:
    --------
    Dict[str, Any]
        예측 결과
    """

    try:
        # 데이터 검증
        if date_column not in df.columns or value_column not in df.columns:
            return {
                "success": False,
                "error": f"필요한 컬럼이 없습니다: {date_column}, {value_column}",
                "available_columns": list(df.columns)
            }

        # 날짜 컬럼 변환
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)

        # 결측값 처리
        df[value_column] = df[value_column].fillna(method='ffill').fillna(method='bfill')

        # 시계열 데이터 준비
        ts = df.set_index(date_column)[value_column]

        # 훈련/테스트 분할
        train_size = int(len(ts) * train_ratio)
        train_ts = ts[:train_size]
        test_ts = ts[train_size:]

        # 모델별 결과 저장
        model_results = {}
        forecasts = {}
        model_objects = {}

        # 데이터 분석
        data_analysis = {
            "total_points": len(ts),
            "train_points": len(train_ts),
            "test_points": len(test_ts),
            "date_range": {
                "start": str(ts.index.min()),
                "end": str(ts.index.max())
            },
            "value_statistics": {
                "mean": float(ts.mean()),
                "std": float(ts.std()),
                "min": float(ts.min()),
                "max": float(ts.max())
            }
        }

        # 정상성 검정
        stationarity_test = check_stationarity(train_ts)
        data_analysis["stationarity"] = stationarity_test

        # ARIMA 모델
        if "arima" in models and ARIMA_AVAILABLE:
            try:
                arima_result = auto_arima_selection(train_ts)
                if "error" not in arima_result:
                    arima_model = arima_result["model"]
                    arima_forecast = arima_model.forecast(steps=len(test_ts))
                    arima_future = arima_model.forecast(steps=forecast_periods)

                    # 신뢰구간
                    forecast_ci = arima_model.get_forecast(steps=forecast_periods).conf_int()

                    model_results["arima"] = {
                        "order": arima_result["best_order"],
                        "aic": arima_result["best_aic"],
                        "bic": arima_result["best_bic"],
                        "test_mse": float(mean_squared_error(test_ts, arima_forecast)),
                        "test_mae": float(mean_absolute_error(test_ts, arima_forecast))
                    }

                    forecasts["arima"] = {
                        "values": arima_future.tolist(),
                        "confidence_intervals": {
                            "lower": forecast_ci.iloc[:, 0].tolist(),
                            "upper": forecast_ci.iloc[:, 1].tolist()
                        }
                    }

                    model_objects["arima"] = arima_model

            except Exception as e:
                model_results["arima"] = {"error": str(e)}

        # Prophet 모델
        if "prophet" in models and PROPHET_AVAILABLE:
            try:
                # Prophet용 데이터 준비
                prophet_df = train_ts.reset_index()
                prophet_df.columns = ['ds', 'y']

                # Prophet 모델 훈련
                prophet_model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    confidence_level=confidence_level
                )
                prophet_model.fit(prophet_df)

                # 테스트 예측
                test_dates = pd.DataFrame({'ds': test_ts.index})
                prophet_test = prophet_model.predict(test_dates)

                # 미래 예측
                future_dates = prophet_model.make_future_dataframe(periods=forecast_periods)
                prophet_forecast = prophet_model.predict(future_dates)

                # 미래 예측값만 추출
                future_forecast = prophet_forecast.tail(forecast_periods)

                model_results["prophet"] = {
                    "test_mse": float(mean_squared_error(test_ts, prophet_test['yhat'])),
                    "test_mae": float(mean_absolute_error(test_ts, prophet_test['yhat'])),
                    "seasonality_components": list(prophet_model.seasonalities.keys())
                }

                forecasts["prophet"] = {
                    "values": future_forecast['yhat'].tolist(),
                    "confidence_intervals": {
                        "lower": future_forecast['yhat_lower'].tolist(),
                        "upper": future_forecast['yhat_upper'].tolist()
                    }
                }

                model_objects["prophet"] = prophet_model

            except Exception as e:
                model_results["prophet"] = {"error": str(e)}

        # LSTM 모델
        if "lstm" in models and TENSORFLOW_AVAILABLE:
            try:
                # 데이터 스케일링
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(train_ts.values.reshape(-1, 1))

                # LSTM 데이터 준비
                X_train, y_train = prepare_lstm_data(scaled_data, lookback_window)

                if len(X_train) > 0:
                    # 모델 생성
                    lstm_model = create_lstm_model(lookback_window)

                    # 조기 종료 콜백
                    early_stop = EarlyStopping(monitor='loss', patience=10)

                    # 모델 훈련
                    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                    lstm_model.fit(X_train, y_train, epochs=50, batch_size=32,
                                 verbose=0, callbacks=[early_stop])

                    # 테스트 예측
                    test_scaled = scaler.transform(test_ts.values.reshape(-1, 1))
                    full_scaled = scaler.transform(ts.values.reshape(-1, 1))

                    # 테스트 데이터 예측
                    test_predictions = []
                    for i in range(len(test_ts)):
                        if train_size + i >= lookback_window:
                            input_seq = full_scaled[train_size + i - lookback_window:train_size + i]
                            input_seq = input_seq.reshape((1, lookback_window, 1))
                            pred = lstm_model.predict(input_seq, verbose=0)
                            test_predictions.append(pred[0][0])

                    if test_predictions:
                        test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1)).flatten()

                        # 미래 예측
                        future_predictions = []
                        last_sequence = full_scaled[-lookback_window:]

                        for _ in range(forecast_periods):
                            input_seq = last_sequence.reshape((1, lookback_window, 1))
                            pred = lstm_model.predict(input_seq, verbose=0)
                            future_predictions.append(pred[0][0])

                            # 시퀀스 업데이트
                            last_sequence = np.append(last_sequence[1:], pred[0][0]).reshape(-1, 1)

                        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

                        model_results["lstm"] = {
                            "test_mse": float(mean_squared_error(test_ts[:len(test_predictions)], test_predictions)),
                            "test_mae": float(mean_absolute_error(test_ts[:len(test_predictions)], test_predictions)),
                            "lookback_window": lookback_window,
                            "epochs_trained": len(lstm_model.history.history['loss']) if hasattr(lstm_model, 'history') else None
                        }

                        forecasts["lstm"] = {
                            "values": future_predictions.tolist(),
                            "confidence_intervals": None  # LSTM은 기본적으로 점 예측
                        }

                        model_objects["lstm"] = {
                            "model": lstm_model,
                            "scaler": scaler
                        }

            except Exception as e:
                model_results["lstm"] = {"error": str(e)}

        # 앙상블 예측
        ensemble_forecast = None
        if len(forecasts) > 1:
            ensemble_values = []
            for i in range(forecast_periods):
                values = []
                for model_name, forecast in forecasts.items():
                    if "error" not in model_results[model_name]:
                        values.append(forecast["values"][i])

                if values:
                    if ensemble_method == "average":
                        ensemble_values.append(np.mean(values))
                    elif ensemble_method == "median":
                        ensemble_values.append(np.median(values))
                    elif ensemble_method == "weighted":
                        # 가중평균 (테스트 성능 기반)
                        weights = []
                        for model_name in forecasts.keys():
                            if "error" not in model_results[model_name]:
                                mse = model_results[model_name].get("test_mse", float('inf'))
                                weights.append(1 / (1 + mse))  # MSE 역수 기반 가중치

                        if weights:
                            weights = np.array(weights) / np.sum(weights)
                            ensemble_values.append(np.average(values, weights=weights))

            if ensemble_values:
                ensemble_forecast = {
                    "values": ensemble_values,
                    "method": ensemble_method,
                    "models_used": list(forecasts.keys())
                }

        # 예측 날짜 생성
        last_date = ts.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')

        # 모델 저장
        save_data = {
            "models": model_objects,
            "data_info": {
                "date_column": date_column,
                "value_column": value_column,
                "last_date": str(last_date),
                "frequency": "D"
            }
        }
        joblib.dump(save_data, model_save_path)

        # 결과 정리
        result = {
            "success": True,
            "model_save_path": model_save_path,
            "data_analysis": clean_dict_for_json(data_analysis),
            "models_performance": clean_dict_for_json(model_results),
            "forecasts": clean_dict_for_json(forecasts),
            "ensemble_forecast": clean_dict_for_json(ensemble_forecast),
            "forecast_dates": [str(date) for date in forecast_dates],
            "forecast_periods": forecast_periods,
            "models_available": {
                "arima": ARIMA_AVAILABLE,
                "prophet": PROPHET_AVAILABLE,
                "lstm": TENSORFLOW_AVAILABLE
            },
            "insights": generate_forecasting_insights(model_results, data_analysis, forecasts)
        }

        return clean_dict_for_json(result)

    except Exception as e:
        return {
            "success": False,
            "error": f"시계열 예측 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def generate_forecasting_insights(model_results: Dict[str, Any], data_analysis: Dict[str, Any],
                                forecasts: Dict[str, Any]) -> List[str]:
    """시계열 예측 인사이트 생성"""

    insights = []

    # 데이터 특성 분석
    if "stationarity" in data_analysis:
        stationarity = data_analysis["stationarity"]
        if stationarity.get("stationary", False):
            insights.append("시계열 데이터가 정상성을 만족합니다")
        else:
            insights.append("시계열 데이터가 비정상적입니다. 차분이나 변환을 고려하세요")

    # 모델 성능 비교
    successful_models = {k: v for k, v in model_results.items() if "error" not in v}

    if len(successful_models) > 1:
        best_model = min(successful_models.keys(),
                        key=lambda k: successful_models[k].get("test_mse", float('inf')))
        best_mse = successful_models[best_model]["test_mse"]
        insights.append(f"'{best_model}' 모델이 가장 좋은 성능을 보입니다 (MSE: {best_mse:.4f})")

    elif len(successful_models) == 1:
        model_name = list(successful_models.keys())[0]
        insights.append(f"'{model_name}' 모델만 성공적으로 훈련되었습니다")

    # 개별 모델 인사이트
    if "arima" in successful_models:
        arima_order = model_results["arima"]["order"]
        insights.append(f"ARIMA{arima_order} 모델이 선택되었습니다")

    if "prophet" in successful_models:
        seasonality = model_results["prophet"].get("seasonality_components", [])
        if seasonality:
            insights.append(f"Prophet이 감지한 계절성: {', '.join(seasonality)}")

    if "lstm" in successful_models:
        lookback = model_results["lstm"]["lookback_window"]
        insights.append(f"LSTM이 {lookback}일의 과거 데이터를 사용합니다")

    # 예측 범위 분석
    if forecasts:
        for model_name, forecast in forecasts.items():
            if "error" not in model_results.get(model_name, {}):
                values = forecast["values"]
                if values:
                    forecast_mean = np.mean(values)
                    historical_mean = data_analysis["value_statistics"]["mean"]

                    diff_percent = abs(forecast_mean - historical_mean) / historical_mean * 100
                    if diff_percent > 20:
                        insights.append(f"{model_name} 예측값이 과거 평균과 {diff_percent:.1f}% 차이를 보입니다")

    # 신뢰구간 분석
    wide_intervals = []
    for model_name, forecast in forecasts.items():
        if forecast.get("confidence_intervals"):
            ci = forecast["confidence_intervals"]
            if ci["lower"] and ci["upper"]:
                avg_width = np.mean([u - l for u, l in zip(ci["upper"], ci["lower"])])
                historical_std = data_analysis["value_statistics"]["std"]

                if avg_width > 2 * historical_std:
                    wide_intervals.append(model_name)

    if wide_intervals:
        insights.append(f"{', '.join(wide_intervals)} 모델의 신뢰구간이 넓어 예측 불확실성이 높습니다")

    return insights

def main():
    """메인 실행 함수"""
    try:
        # stdin에서 JSON 데이터 읽기
        input_data = sys.stdin.read()
        params = json.loads(input_data)

        # 시계열 예측
        result = advanced_time_series_forecast(**params)

        # JSON으로 결과 출력
        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
        print(json.dumps(error_result, ensure_ascii=False, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()