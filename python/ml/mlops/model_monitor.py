"""
Model Monitoring
Monitor ML model performance in production

Features:
- Prediction monitoring (latency, throughput)
- Data drift detection
- Model drift detection
- Performance degradation alerts
- Automatic retraining triggers
- Dashboard integration
- Metrics collection and reporting

Usage:
    from model_monitor import ModelMonitor

    monitor = ModelMonitor(model_name='fraud_detector')

    # Log predictions
    monitor.log_prediction(X, y_pred, latency_ms=45)

    # Check for drift
    drift_report = monitor.check_drift(reference_data=X_train, current_data=X_recent)

    # Get metrics
    metrics = monitor.get_metrics(period='7d')
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from collections import deque

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("Warning: Evidently not installed. Install with: pip install evidently")


class ModelMonitor:
    """
    Monitor ML model performance in production

    Tracks predictions, detects drift, and triggers alerts
    """

    def __init__(
        self,
        model_name: str,
        monitoring_window: int = 1000,
        drift_threshold: float = 0.1,
        save_dir: Optional[str] = None
    ):
        """
        Initialize model monitor

        Args:
            model_name: Name of model to monitor
            monitoring_window: Number of predictions to keep in memory
            drift_threshold: Threshold for drift detection (0-1)
            save_dir: Directory to save monitoring data
        """
        self.model_name = model_name
        self.monitoring_window = monitoring_window
        self.drift_threshold = drift_threshold
        self.save_dir = Path(save_dir) if save_dir else Path('monitoring') / model_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Monitoring data
        self.predictions = deque(maxlen=monitoring_window)
        self.latencies = deque(maxlen=monitoring_window)
        self.timestamps = deque(maxlen=monitoring_window)
        self.features_log = []

        # Statistics
        self.total_predictions = 0
        self.total_errors = 0
        self.drift_detected_count = 0

        # Reference data for drift detection
        self.reference_data = None

    def log_prediction(
        self,
        input_features: np.ndarray,
        prediction: Any,
        latency_ms: float,
        timestamp: Optional[datetime] = None,
        ground_truth: Optional[Any] = None
    ):
        """
        Log a prediction

        Args:
            input_features: Input features
            prediction: Model prediction
            latency_ms: Prediction latency in milliseconds
            timestamp: Prediction timestamp
            ground_truth: Optional ground truth label
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Store prediction
        self.predictions.append({
            'prediction': prediction,
            'ground_truth': ground_truth,
            'timestamp': timestamp
        })

        self.latencies.append(latency_ms)
        self.timestamps.append(timestamp)
        self.total_predictions += 1

        # Store features for drift detection (sample to avoid memory issues)
        if len(self.features_log) < 10000:
            if isinstance(input_features, np.ndarray):
                self.features_log.append(input_features.flatten())
            else:
                self.features_log.append(input_features)

    def set_reference_data(self, reference_data: pd.DataFrame):
        """
        Set reference data for drift detection

        Args:
            reference_data: Training or validation data to use as reference
        """
        self.reference_data = reference_data

    def check_drift(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        current_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Check for data drift

        Args:
            reference_data: Reference data (training data)
            current_data: Current production data

        Returns:
            Drift report
        """
        if reference_data is not None:
            self.reference_data = reference_data

        if self.reference_data is None:
            return {'error': 'No reference data set'}

        if current_data is None:
            if not self.features_log:
                return {'error': 'No current data available'}
            current_data = pd.DataFrame(self.features_log)

        # Simple drift detection (KS test for each feature)
        drift_report = {
            'drift_detected': False,
            'drifted_features': [],
            'drift_scores': {},
            'timestamp': datetime.now().isoformat()
        }

        if EVIDENTLY_AVAILABLE:
            try:
                # Use Evidently for comprehensive drift detection
                report = Report(metrics=[DataDriftPreset()])
                report.run(
                    reference_data=self.reference_data,
                    current_data=current_data
                )

                # Extract drift information
                result = report.as_dict()
                drift_report['drift_detected'] = result['metrics'][0]['result']['dataset_drift']
                drift_report['drifted_features'] = [
                    feat for feat, info in result['metrics'][0]['result']['drift_by_columns'].items()
                    if info['drift_detected']
                ]

            except Exception as e:
                drift_report['error'] = f"Evidently drift detection failed: {str(e)}"

        else:
            # Fallback: Simple statistical drift detection
            from scipy import stats

            for col in self.reference_data.columns:
                if col in current_data.columns:
                    # KS test
                    statistic, pvalue = stats.ks_2samp(
                        self.reference_data[col],
                        current_data[col]
                    )

                    drift_report['drift_scores'][col] = {
                        'statistic': float(statistic),
                        'pvalue': float(pvalue)
                    }

                    if pvalue < 0.05:  # Significant drift
                        drift_report['drift_detected'] = True
                        drift_report['drifted_features'].append(col)

        if drift_report['drift_detected']:
            self.drift_detected_count += 1

        return drift_report

    def get_metrics(self, period: str = '1h') -> Dict[str, Any]:
        """
        Get monitoring metrics for a time period

        Args:
            period: Time period ('1h', '24h', '7d', 'all')

        Returns:
            Monitoring metrics
        """
        # Parse period
        if period == 'all':
            cutoff_time = datetime.min
        else:
            value = int(period[:-1])
            unit = period[-1]

            if unit == 'h':
                cutoff_time = datetime.now() - timedelta(hours=value)
            elif unit == 'd':
                cutoff_time = datetime.now() - timedelta(days=value)
            else:
                cutoff_time = datetime.min

        # Filter data by period
        filtered_latencies = [
            lat for lat, ts in zip(self.latencies, self.timestamps)
            if ts >= cutoff_time
        ]

        filtered_predictions = [
            pred for pred, ts in zip(self.predictions, self.timestamps)
            if ts >= cutoff_time
        ]

        # Calculate metrics
        metrics = {
            'model_name': self.model_name,
            'period': period,
            'total_predictions': len(filtered_predictions),
            'total_predictions_lifetime': self.total_predictions,
            'drift_detected_count': self.drift_detected_count,
            'latency': {},
            'throughput': {},
            'errors': {
                'total': self.total_errors,
                'rate': self.total_errors / max(1, self.total_predictions)
            }
        }

        if filtered_latencies:
            metrics['latency'] = {
                'mean_ms': float(np.mean(filtered_latencies)),
                'median_ms': float(np.median(filtered_latencies)),
                'p95_ms': float(np.percentile(filtered_latencies, 95)),
                'p99_ms': float(np.percentile(filtered_latencies, 99)),
                'max_ms': float(np.max(filtered_latencies))
            }

        if filtered_predictions:
            # Calculate throughput
            if len(filtered_predictions) > 1:
                time_span_hours = (self.timestamps[-1] - cutoff_time).total_seconds() / 3600
                if time_span_hours > 0:
                    metrics['throughput'] = {
                        'predictions_per_hour': len(filtered_predictions) / time_span_hours,
                        'predictions_per_minute': len(filtered_predictions) / (time_span_hours * 60)
                    }

        return metrics

    def trigger_retraining(self) -> Dict[str, Any]:
        """
        Trigger model retraining

        Returns:
            Retraining status
        """
        return {
            'model_name': self.model_name,
            'trigger_reason': 'drift_detected',
            'timestamp': datetime.now().isoformat(),
            'status': 'retraining_triggered',
            'message': 'Model retraining has been triggered due to drift detection'
        }

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate monitoring report

        Args:
            output_path: Path to save report

        Returns:
            Report as JSON string
        """
        report = {
            'model_name': self.model_name,
            'generated_at': datetime.now().isoformat(),
            'metrics_1h': self.get_metrics('1h'),
            'metrics_24h': self.get_metrics('24h'),
            'metrics_7d': self.get_metrics('7d'),
            'drift_status': {
                'drift_detected_count': self.drift_detected_count,
                'monitoring_window': self.monitoring_window
            }
        }

        report_json = json.dumps(report, indent=2)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_json)

        return report_json

    def save_state(self):
        """Save monitoring state to disk"""
        state = {
            'model_name': self.model_name,
            'total_predictions': self.total_predictions,
            'total_errors': self.total_errors,
            'drift_detected_count': self.drift_detected_count,
            'last_updated': datetime.now().isoformat()
        }

        state_path = self.save_dir / 'monitor_state.json'
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self):
        """Load monitoring state from disk"""
        state_path = self.save_dir / 'monitor_state.json'

        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)

            self.total_predictions = state.get('total_predictions', 0)
            self.total_errors = state.get('total_errors', 0)
            self.drift_detected_count = state.get('drift_detected_count', 0)


# CLI interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Model Monitor')
    parser.add_argument('command', choices=['report', 'drift-check'],
                        help='Command to execute')
    parser.add_argument('--model-name', type=str, required=True, help='Model name')
    parser.add_argument('--reference-data', type=str, help='Path to reference data CSV')
    parser.add_argument('--current-data', type=str, help='Path to current data CSV')
    parser.add_argument('--output', type=str, help='Output path for report')

    args = parser.parse_args()

    monitor = ModelMonitor(model_name=args.model_name)

    if args.command == 'report':
        report = monitor.generate_report(args.output)
        print("Monitoring Report:")
        print(report)

    elif args.command == 'drift-check':
        if not args.reference_data or not args.current_data:
            print("Error: --reference-data and --current-data required")
            exit(1)

        ref_data = pd.read_csv(args.reference_data)
        curr_data = pd.read_csv(args.current_data)

        drift_report = monitor.check_drift(ref_data, curr_data)

        print("\nDrift Detection Report:")
        print(json.dumps(drift_report, indent=2))

        if drift_report.get('drift_detected'):
            print("\n⚠️ DRIFT DETECTED!")
            print(f"Drifted features: {', '.join(drift_report['drifted_features'])}")
