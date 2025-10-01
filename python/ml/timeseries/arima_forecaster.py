#!/usr/bin/env python3
"""
ARIMA Forecaster for ML MCP System
ARIMA/SARIMA time series forecasting with auto parameter selection
"""

import pandas as pd
import numpy as np
import json
import sys
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Check statsmodels availability
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Check pmdarima availability
try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ARIMAForecaster:
    """ARIMA/SARIMA time series forecaster"""

    def __init__(self, seasonal: bool = False, m: int = 12):
        """
        Initialize ARIMA forecaster

        Args:
            seasonal: Use SARIMA (seasonal ARIMA)
            m: Seasonal period (12 for monthly, 4 for quarterly, etc.)
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required. Install with: pip install statsmodels")

        self.model = None
        self.fitted_model = None
        self.seasonal = seasonal
        self.m = m
        self.order = None
        self.seasonal_order = None
        self.timeseries = None

    def check_stationarity(self, timeseries: pd.Series) -> Dict[str, Any]:
        """
        Test for stationarity using ADF and KPSS tests

        Args:
            timeseries: Time series data

        Returns:
            Stationarity test results
        """
        # Augmented Dickey-Fuller test
        adf_result = adfuller(timeseries, autolag='AIC')
        adf_stationary = adf_result[1] < 0.05  # p-value < 0.05 means stationary

        # KPSS test
        kpss_result = kpss(timeseries, regression='c', nlags='auto')
        kpss_stationary = kpss_result[1] >= 0.05  # p-value >= 0.05 means stationary

        result = {
            'adf_test': {
                'statistic': float(adf_result[0]),
                'p_value': float(adf_result[1]),
                'critical_values': {str(k): float(v) for k, v in adf_result[4].items()},
                'stationary': adf_stationary
            },
            'kpss_test': {
                'statistic': float(kpss_result[0]),
                'p_value': float(kpss_result[1]),
                'critical_values': {str(k): float(v) for k, v in kpss_result[3].items()},
                'stationary': kpss_stationary
            },
            'summary': {
                'is_stationary': adf_stationary and kpss_stationary,
                'recommendation': 'Differencing not needed' if (adf_stationary and kpss_stationary)
                                else 'Apply differencing to make series stationary'
            }
        }

        return result

    def auto_fit(self, timeseries: pd.Series, seasonal: bool = None,
                m: int = None, max_p: int = 5, max_q: int = 5,
                max_d: int = 2, **kwargs) -> Dict[str, Any]:
        """
        Automatically find best ARIMA parameters

        Args:
            timeseries: Time series data
            seasonal: Override seasonal setting
            m: Override seasonal period
            max_p: Maximum AR order
            max_q: Maximum MA order
            max_d: Maximum differencing order
            **kwargs: Additional auto_arima parameters

        Returns:
            Fitting results
        """
        if not PMDARIMA_AVAILABLE:
            return {
                'success': False,
                'error': 'pmdarima required for auto_fit. Install with: pip install pmdarima'
            }

        try:
            seasonal = seasonal if seasonal is not None else self.seasonal
            m = m if m is not None else self.m

            # Auto ARIMA
            auto_model = auto_arima(
                timeseries,
                seasonal=seasonal,
                m=m,
                max_p=max_p,
                max_q=max_q,
                max_d=max_d,
                start_p=1,
                start_q=1,
                suppress_warnings=True,
                stepwise=True,
                **kwargs
            )

            self.fitted_model = auto_model
            self.order = auto_model.order
            self.seasonal_order = auto_model.seasonal_order if seasonal else None
            self.timeseries = timeseries

            result = {
                'success': True,
                'method': 'auto_arima',
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'aic': float(auto_model.aic()),
                'bic': float(auto_model.bic()),
                'data_points': len(timeseries),
                'parameters': {
                    'p': self.order[0],
                    'd': self.order[1],
                    'q': self.order[2]
                }
            }

            if seasonal and self.seasonal_order:
                result['seasonal_parameters'] = {
                    'P': self.seasonal_order[0],
                    'D': self.seasonal_order[1],
                    'Q': self.seasonal_order[2],
                    'm': self.seasonal_order[3]
                }

            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }

    def fit(self, timeseries: pd.Series, order: Tuple[int, int, int],
            seasonal_order: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Fit ARIMA/SARIMA model with specified parameters

        Args:
            timeseries: Time series data
            order: (p, d, q) order
            seasonal_order: (P, D, Q, m) seasonal order

        Returns:
            Fitting results
        """
        try:
            if seasonal_order:
                # SARIMA
                model = SARIMAX(timeseries, order=order,
                               seasonal_order=seasonal_order,
                               enforce_stationarity=False,
                               enforce_invertibility=False)
            else:
                # ARIMA
                model = ARIMA(timeseries, order=order)

            self.fitted_model = model.fit()
            self.order = order
            self.seasonal_order = seasonal_order
            self.timeseries = timeseries

            result = {
                'success': True,
                'method': 'manual_fit',
                'model': 'SARIMA' if seasonal_order else 'ARIMA',
                'order': order,
                'seasonal_order': seasonal_order,
                'aic': float(self.fitted_model.aic),
                'bic': float(self.fitted_model.bic),
                'data_points': len(timeseries)
            }

            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }

    def predict(self, steps: int, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Make future predictions

        Args:
            steps: Number of steps to forecast
            alpha: Significance level for confidence intervals

        Returns:
            Forecast results
        """
        if self.fitted_model is None:
            return {
                'success': False,
                'error': 'Model not fitted. Call fit() or auto_fit() first.'
            }

        try:
            # Get forecast
            forecast_result = self.fitted_model.get_forecast(steps=steps)
            forecast_mean = forecast_result.predicted_mean
            forecast_ci = forecast_result.conf_int(alpha=alpha)

            # Create index for forecast
            if isinstance(self.timeseries.index, pd.DatetimeIndex):
                last_date = self.timeseries.index[-1]
                freq = self.timeseries.index.freq or pd.infer_freq(self.timeseries.index)
                forecast_index = pd.date_range(start=last_date, periods=steps+1, freq=freq)[1:]
            else:
                forecast_index = range(len(self.timeseries), len(self.timeseries) + steps)

            result = {
                'success': True,
                'steps': steps,
                'confidence_level': 1 - alpha,
                'forecast': {
                    'index': [str(idx) for idx in forecast_index],
                    'values': forecast_mean.tolist(),
                    'lower_bound': forecast_ci.iloc[:, 0].tolist(),
                    'upper_bound': forecast_ci.iloc[:, 1].tolist()
                },
                'summary': {
                    'mean_forecast': float(forecast_mean.mean()),
                    'forecast_range': {
                        'min': float(forecast_mean.min()),
                        'max': float(forecast_mean.max())
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

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get detailed model summary

        Returns:
            Model summary
        """
        if self.fitted_model is None:
            return {
                'success': False,
                'error': 'Model not fitted'
            }

        try:
            summary = self.fitted_model.summary()

            result = {
                'success': True,
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'aic': float(self.fitted_model.aic),
                'bic': float(self.fitted_model.bic),
                'log_likelihood': float(self.fitted_model.llf),
                'summary_text': str(summary)
            }

            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def plot_diagnostics(self, output_path: Optional[str] = None) -> str:
        """
        Plot model diagnostics

        Args:
            output_path: Output file path

        Returns:
            Path to saved plot
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted")

        fig = self.fitted_model.plot_diagnostics(figsize=(12, 8))
        plt.tight_layout()

        if output_path is None:
            output_path = 'arima_diagnostics.png'

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_forecast(self, steps: int, output_path: Optional[str] = None) -> str:
        """
        Plot forecast

        Args:
            steps: Number of steps to forecast
            output_path: Output file path

        Returns:
            Path to saved plot
        """
        forecast_result = self.predict(steps)

        if not forecast_result['success']:
            raise ValueError(f"Forecast failed: {forecast_result['error']}")

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot historical data
        ax.plot(self.timeseries.index, self.timeseries.values,
               label='Historical', color='blue')

        # Plot forecast
        forecast_data = forecast_result['forecast']
        forecast_index = pd.to_datetime(forecast_data['index']) if forecast_data['index'][0].count('-') > 1 else range(len(self.timeseries), len(self.timeseries) + steps)

        ax.plot(forecast_index, forecast_data['values'],
               label='Forecast', color='red', linestyle='--')

        # Plot confidence interval
        ax.fill_between(forecast_index,
                       forecast_data['lower_bound'],
                       forecast_data['upper_bound'],
                       alpha=0.3, color='red')

        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(f'ARIMA{self.order} Forecast ({steps} steps)')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if output_path is None:
            output_path = 'arima_forecast.png'

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_acf_pacf(self, lags: int = 40, output_path: Optional[str] = None) -> str:
        """
        Plot ACF and PACF

        Args:
            lags: Number of lags
            output_path: Output file path

        Returns:
            Path to saved plot
        """
        if self.timeseries is None:
            raise ValueError("No timeseries data available")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # ACF
        plot_acf(self.timeseries, lags=lags, ax=ax1)
        ax1.set_title('Autocorrelation Function (ACF)')

        # PACF
        plot_pacf(self.timeseries, lags=lags, ax=ax2)
        ax2.set_title('Partial Autocorrelation Function (PACF)')

        plt.tight_layout()

        if output_path is None:
            output_path = 'arima_acf_pacf.png'

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path


def main():
    """CLI interface"""
    if len(sys.argv) < 2:
        print("Usage: python arima_forecaster.py <action>")
        print("Actions: demo, check_availability")
        sys.exit(1)

    action = sys.argv[1]

    try:
        if action == 'check_availability':
            result = {
                'statsmodels_available': STATSMODELS_AVAILABLE,
                'pmdarima_available': PMDARIMA_AVAILABLE,
                'install_commands': {
                    'statsmodels': 'pip install statsmodels' if not STATSMODELS_AVAILABLE else None,
                    'pmdarima': 'pip install pmdarima' if not PMDARIMA_AVAILABLE else None
                }
            }

        elif action == 'demo':
            if not STATSMODELS_AVAILABLE:
                result = {
                    'success': False,
                    'error': 'statsmodels not installed',
                    'install_command': 'pip install statsmodels pmdarima'
                }
            else:
                print("ARIMA Forecaster Demo")
                print("=" * 50)

                # Generate synthetic time series
                np.random.seed(42)
                n = 200
                trend = np.linspace(0, 10, n)
                seasonal = 5 * np.sin(2 * np.pi * np.arange(n) / 12)
                noise = np.random.normal(0, 1, n)
                ts = pd.Series(trend + seasonal + noise)

                # Initialize forecaster
                forecaster = ARIMAForecaster(seasonal=False)

                # Check stationarity
                stationarity = forecaster.check_stationarity(ts)
                print(f"\nStationarity: {stationarity['summary']['is_stationary']}")
                print(f"ADF p-value: {stationarity['adf_test']['p_value']:.4f}")

                if PMDARIMA_AVAILABLE:
                    # Auto fit
                    fit_result = forecaster.auto_fit(ts)
                    print(f"\nBest order: ARIMA{fit_result['order']}")
                    print(f"AIC: {fit_result['aic']:.2f}")
                else:
                    # Manual fit
                    fit_result = forecaster.fit(ts, order=(1, 1, 1))
                    print(f"\nOrder: ARIMA{fit_result['order']}")
                    print(f"AIC: {fit_result['aic']:.2f}")

                # Forecast
                forecast_result = forecaster.predict(steps=12)
                print(f"\nForecast: {forecast_result['steps']} steps")
                print(f"Mean forecast: {forecast_result['summary']['mean_forecast']:.2f}")

                result = {
                    'success': True,
                    'stationarity': stationarity,
                    'fit_result': fit_result,
                    'forecast_result': forecast_result
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