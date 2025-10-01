#!/usr/bin/env python3
"""
Prophet Forecaster for ML MCP System
Facebook Prophet time series forecasting
"""

import pandas as pd
import numpy as np
import json
import sys
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Check Prophet availability
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ProphetForecaster:
    """Facebook Prophet time series forecaster"""

    def __init__(self, seasonality_mode: str = 'additive',
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0):
        """
        Initialize Prophet forecaster

        Args:
            seasonality_mode: 'additive' or 'multiplicative'
            changepoint_prior_scale: Flexibility of trend (default 0.05)
            seasonality_prior_scale: Flexibility of seasonality (default 10.0)
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet required. Install with: pip install prophet")

        self.model = None
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.history = None
        self.forecast = None

    def fit(self, df: pd.DataFrame, date_column: str = 'ds',
            value_column: str = 'y', **kwargs) -> Dict[str, Any]:
        """
        Fit Prophet model

        Args:
            df: DataFrame with time series data
            date_column: Name of date column (will be renamed to 'ds')
            value_column: Name of value column (will be renamed to 'y')
            **kwargs: Additional Prophet parameters

        Returns:
            Fitting results
        """
        try:
            # Prepare data
            prophet_df = df[[date_column, value_column]].copy()
            prophet_df.columns = ['ds', 'y']

            # Ensure datetime
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

            # Initialize model
            self.model = Prophet(
                seasonality_mode=self.seasonality_mode,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
                **kwargs
            )

            # Fit model
            self.model.fit(prophet_df)
            self.history = prophet_df

            # Get model parameters
            result = {
                'success': True,
                'model': 'prophet',
                'data_points': len(prophet_df),
                'date_range': {
                    'start': prophet_df['ds'].min().isoformat(),
                    'end': prophet_df['ds'].max().isoformat()
                },
                'parameters': {
                    'seasonality_mode': self.seasonality_mode,
                    'changepoint_prior_scale': self.changepoint_prior_scale,
                    'seasonality_prior_scale': self.seasonality_prior_scale
                },
                'changepoints': {
                    'n_changepoints': len(self.model.changepoints),
                    'changepoints': [cp.isoformat() for cp in self.model.changepoints]
                }
            }

            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }

    def predict(self, periods: int, freq: str = 'D',
                include_history: bool = False) -> Dict[str, Any]:
        """
        Make future predictions

        Args:
            periods: Number of periods to forecast
            freq: Frequency ('D'=daily, 'W'=weekly, 'M'=monthly, etc.)
            include_history: Include historical data in output

        Returns:
            Forecast results
        """
        if self.model is None:
            return {
                'success': False,
                'error': 'Model not fitted. Call fit() first.'
            }

        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=periods, freq=freq,
                                                      include_history=include_history)

            # Make prediction
            forecast = self.model.predict(future)
            self.forecast = forecast

            # Extract forecast values
            if include_history:
                forecast_data = forecast.tail(periods)
            else:
                forecast_data = forecast

            result = {
                'success': True,
                'periods': periods,
                'freq': freq,
                'forecast': {
                    'dates': forecast_data['ds'].dt.strftime('%Y-%m-%d').tolist(),
                    'yhat': forecast_data['yhat'].tolist(),
                    'yhat_lower': forecast_data['yhat_lower'].tolist(),
                    'yhat_upper': forecast_data['yhat_upper'].tolist(),
                    'trend': forecast_data['trend'].tolist()
                },
                'summary': {
                    'mean_forecast': float(forecast_data['yhat'].mean()),
                    'forecast_range': {
                        'min': float(forecast_data['yhat'].min()),
                        'max': float(forecast_data['yhat'].max())
                    },
                    'uncertainty': {
                        'mean_width': float((forecast_data['yhat_upper'] - forecast_data['yhat_lower']).mean())
                    }
                }
            }

            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }

    def plot_forecast(self, output_path: Optional[str] = None) -> str:
        """
        Plot forecast

        Args:
            output_path: Output file path

        Returns:
            Path to saved plot
        """
        if self.model is None or self.forecast is None:
            raise ValueError("Model not fitted or no forecast available")

        fig = self.model.plot(self.forecast)
        plt.tight_layout()

        if output_path is None:
            output_path = 'prophet_forecast.png'

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_components(self, output_path: Optional[str] = None) -> str:
        """
        Plot forecast components (trend, seasonality)

        Args:
            output_path: Output file path

        Returns:
            Path to saved plot
        """
        if self.model is None or self.forecast is None:
            raise ValueError("Model not fitted or no forecast available")

        fig = self.model.plot_components(self.forecast)
        plt.tight_layout()

        if output_path is None:
            output_path = 'prophet_components.png'

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def cross_validation(self, initial: str, period: str, horizon: str) -> Dict[str, Any]:
        """
        Perform cross-validation

        Args:
            initial: Initial training period (e.g., '365 days')
            period: Period between cutoff dates (e.g., '180 days')
            horizon: Forecast horizon (e.g., '30 days')

        Returns:
            Cross-validation results
        """
        if self.model is None:
            return {
                'success': False,
                'error': 'Model not fitted. Call fit() first.'
            }

        try:
            from prophet.diagnostics import cross_validation, performance_metrics

            # Perform cross-validation
            df_cv = cross_validation(self.model, initial=initial,
                                    period=period, horizon=horizon)

            # Calculate performance metrics
            df_metrics = performance_metrics(df_cv)

            result = {
                'success': True,
                'method': 'cross_validation',
                'parameters': {
                    'initial': initial,
                    'period': period,
                    'horizon': horizon
                },
                'metrics': {
                    'mae': float(df_metrics['mae'].mean()),
                    'mape': float(df_metrics['mape'].mean()),
                    'rmse': float(df_metrics['rmse'].mean()),
                    'coverage': float(df_metrics['coverage'].mean())
                },
                'n_folds': len(df_cv['cutoff'].unique()),
                'summary': {
                    'mean_absolute_error': float(df_metrics['mae'].mean()),
                    'mean_absolute_percentage_error': float(df_metrics['mape'].mean() * 100),
                    'root_mean_squared_error': float(df_metrics['rmse'].mean())
                }
            }

            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }

    def add_holidays(self, holidays_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Add custom holidays/events

        Args:
            holidays_df: DataFrame with columns 'holiday' and 'ds'

        Returns:
            Status
        """
        if self.model is not None:
            return {
                'success': False,
                'error': 'Cannot add holidays after fitting. Create new model.'
            }

        try:
            self.holidays = holidays_df
            return {
                'success': True,
                'holidays_added': len(holidays_df),
                'holidays': holidays_df.to_dict('records')
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def detect_anomalies(self, threshold: float = 0.95) -> Dict[str, Any]:
        """
        Detect anomalies in historical data

        Args:
            threshold: Probability threshold for anomaly detection

        Returns:
            Anomaly detection results
        """
        if self.forecast is None:
            return {
                'success': False,
                'error': 'No forecast available. Call predict() first.'
            }

        try:
            # Compare actual values with predictions
            forecast_hist = self.forecast[self.forecast['ds'].isin(self.history['ds'])]

            # Merge with actual values
            comparison = forecast_hist.merge(self.history, on='ds')

            # Find points outside prediction interval
            comparison['is_anomaly'] = (
                (comparison['y'] < comparison['yhat_lower']) |
                (comparison['y'] > comparison['yhat_upper'])
            )

            anomalies = comparison[comparison['is_anomaly']]

            result = {
                'success': True,
                'threshold': threshold,
                'total_points': len(comparison),
                'anomalies_detected': len(anomalies),
                'anomaly_rate': float(len(anomalies) / len(comparison)),
                'anomalies': [
                    {
                        'date': row['ds'].isoformat(),
                        'actual': float(row['y']),
                        'predicted': float(row['yhat']),
                        'lower_bound': float(row['yhat_lower']),
                        'upper_bound': float(row['yhat_upper']),
                        'deviation': float(abs(row['y'] - row['yhat']))
                    }
                    for _, row in anomalies.head(20).iterrows()
                ]
            }

            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }


def main():
    """CLI interface"""
    if len(sys.argv) < 2:
        print("Usage: python prophet_forecaster.py <action>")
        print("Actions: demo, check_availability")
        sys.exit(1)

    action = sys.argv[1]

    try:
        if action == 'check_availability':
            result = {
                'prophet_available': PROPHET_AVAILABLE,
                'install_command': 'pip install prophet' if not PROPHET_AVAILABLE else None
            }

        elif action == 'demo':
            if not PROPHET_AVAILABLE:
                result = {
                    'success': False,
                    'error': 'Prophet not installed',
                    'install_command': 'pip install prophet'
                }
            else:
                print("Prophet Forecaster Demo")
                print("=" * 50)

                # Generate synthetic time series
                np.random.seed(42)
                dates = pd.date_range('2023-01-01', periods=365, freq='D')
                trend = np.linspace(100, 150, 365)
                seasonality = 10 * np.sin(2 * np.pi * np.arange(365) / 365 * 4)
                noise = np.random.normal(0, 5, 365)
                values = trend + seasonality + noise

                df = pd.DataFrame({
                    'date': dates,
                    'sales': values
                })

                # Fit model
                forecaster = ProphetForecaster()
                fit_result = forecaster.fit(df, date_column='date', value_column='sales')
                print(f"\nModel fitted: {fit_result['data_points']} data points")

                # Make forecast
                forecast_result = forecaster.predict(periods=30, freq='D')
                print(f"\nForecast generated: {forecast_result['periods']} periods")
                print(f"Mean forecast: {forecast_result['summary']['mean_forecast']:.2f}")

                # Detect anomalies
                anomaly_result = forecaster.detect_anomalies()
                print(f"\nAnomalies detected: {anomaly_result['anomalies_detected']} "
                     f"({anomaly_result['anomaly_rate']*100:.1f}%)")

                result = {
                    'success': True,
                    'fit_result': fit_result,
                    'forecast_result': forecast_result,
                    'anomaly_result': anomaly_result
                }

        else:
            result = {'error': f'Unknown action: {action}'}

        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
        print(json.dumps(error_result, ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()