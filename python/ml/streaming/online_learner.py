#!/usr/bin/env python3
"""
Online Learning Module for ML MCP System
Real-time model updates and streaming analytics
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import SGDClassifier, SGDRegressor, PassiveAggressiveClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error


class OnlineLearner:
    """Online/incremental learning for streaming data"""

    def __init__(self, task_type: str = 'classification', model_type: str = 'sgd'):
        """
        Initialize online learner

        Args:
            task_type: 'classification' or 'regression'
            model_type: 'sgd' or 'passive_aggressive'
        """
        self.task_type = task_type
        self.model_type = model_type
        self.model = self._initialize_model()
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.n_samples_seen = 0
        self.performance_history = []

    def _initialize_model(self):
        """Initialize incremental model"""
        if self.task_type == 'classification':
            if self.model_type == 'sgd':
                return SGDClassifier(
                    loss='log_loss',
                    learning_rate='optimal',
                    random_state=42
                )
            elif self.model_type == 'passive_aggressive':
                return PassiveAggressiveClassifier(random_state=42)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
        else:  # regression
            return SGDRegressor(
                loss='squared_error',
                learning_rate='optimal',
                random_state=42
            )

    def partial_fit(self, X: pd.DataFrame, y: pd.Series,
                   classes: Optional[List] = None) -> Dict[str, Any]:
        """
        Incremental learning on new batch

        Args:
            X: Feature batch
            y: Target batch
            classes: All possible classes (required for first batch in classification)

        Returns:
            Update statistics
        """
        if not self.is_fitted:
            # First batch - fit scaler
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            # Update scaler incrementally
            X_scaled = self.scaler.transform(X)

        # Partial fit model
        if self.task_type == 'classification' and classes is not None:
            self.model.partial_fit(X_scaled, y, classes=classes)
        else:
            self.model.partial_fit(X_scaled, y)

        self.n_samples_seen += len(X)

        # Calculate performance on this batch
        y_pred = self.model.predict(X_scaled)

        if self.task_type == 'classification':
            score = accuracy_score(y, y_pred)
            metric_name = 'accuracy'
        else:
            score = mean_squared_error(y, y_pred)
            metric_name = 'mse'

        self.performance_history.append(score)

        return {
            'batch_size': len(X),
            'total_samples_seen': self.n_samples_seen,
            'batch_performance': {
                metric_name: float(score)
            }
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.performance_history:
            return {'error': 'No performance history'}

        metric_name = 'accuracy' if self.task_type == 'classification' else 'mse'

        return {
            'total_samples': self.n_samples_seen,
            'num_updates': len(self.performance_history),
            'current_performance': {
                metric_name: float(self.performance_history[-1])
            },
            'average_performance': {
                metric_name: float(np.mean(self.performance_history))
            },
            'performance_trend': self._calculate_trend()
        }

    def _calculate_trend(self) -> str:
        """Calculate performance trend"""
        if len(self.performance_history) < 2:
            return 'insufficient_data'

        recent = self.performance_history[-10:]
        early = self.performance_history[:10]

        if self.task_type == 'classification':
            # For accuracy, higher is better
            if np.mean(recent) > np.mean(early):
                return 'improving'
            else:
                return 'degrading'
        else:
            # For MSE, lower is better
            if np.mean(recent) < np.mean(early):
                return 'improving'
            else:
                return 'degrading'


class StreamingAggregator:
    """Aggregate statistics from streaming data"""

    def __init__(self, window_size: Optional[int] = None):
        """
        Initialize streaming aggregator

        Args:
            window_size: Rolling window size (None = unbounded)
        """
        self.window_size = window_size
        self.data_buffer = deque(maxlen=window_size) if window_size else deque()
        self.count = 0
        self.running_mean = 0.0
        self.running_m2 = 0.0  # For variance calculation

    def update(self, value: float) -> Dict[str, Any]:
        """
        Update with new value using Welford's algorithm

        Args:
            value: New data point

        Returns:
            Current statistics
        """
        self.count += 1
        delta = value - self.running_mean
        self.running_mean += delta / self.count
        delta2 = value - self.running_mean
        self.running_m2 += delta * delta2

        # Add to buffer for percentiles
        self.data_buffer.append(value)

        return self.get_statistics()

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics"""
        if self.count == 0:
            return {'error': 'No data'}

        variance = self.running_m2 / self.count if self.count > 1 else 0
        std = np.sqrt(variance)

        stats = {
            'count': self.count,
            'mean': float(self.running_mean),
            'std': float(std),
            'variance': float(variance)
        }

        # Calculate percentiles from buffer
        if len(self.data_buffer) > 0:
            buffer_array = np.array(list(self.data_buffer))
            stats.update({
                'min': float(np.min(buffer_array)),
                'max': float(np.max(buffer_array)),
                'median': float(np.median(buffer_array)),
                'q25': float(np.percentile(buffer_array, 25)),
                'q75': float(np.percentile(buffer_array, 75))
            })

        return stats


class ChangeDetector:
    """Detect concept drift in streaming data"""

    def __init__(self, window_size: int = 100, threshold: float = 0.05):
        """
        Initialize change detector

        Args:
            window_size: Window for comparison
            threshold: Significance threshold
        """
        self.window_size = window_size
        self.threshold = threshold
        self.reference_window = deque(maxlen=window_size)
        self.current_window = deque(maxlen=window_size)
        self.drift_detected = False

    def update(self, value: float) -> Dict[str, Any]:
        """
        Update with new value and check for drift

        Args:
            value: New data point

        Returns:
            Drift detection results
        """
        if len(self.reference_window) < self.window_size:
            # Still filling reference window
            self.reference_window.append(value)
            return {
                'drift_detected': False,
                'status': 'initializing',
                'reference_size': len(self.reference_window)
            }

        # Add to current window
        self.current_window.append(value)

        if len(self.current_window) < self.window_size:
            return {
                'drift_detected': False,
                'status': 'collecting',
                'current_size': len(self.current_window)
            }

        # Perform statistical test
        from scipy.stats import ks_2samp

        ref_array = np.array(list(self.reference_window))
        curr_array = np.array(list(self.current_window))

        statistic, p_value = ks_2samp(ref_array, curr_array)

        self.drift_detected = p_value < self.threshold

        result = {
            'drift_detected': self.drift_detected,
            'p_value': float(p_value),
            'statistic': float(statistic),
            'threshold': self.threshold,
            'reference_mean': float(np.mean(ref_array)),
            'current_mean': float(np.mean(curr_array))
        }

        # If drift detected, update reference window
        if self.drift_detected:
            self.reference_window = deque(list(self.current_window), maxlen=self.window_size)
            self.current_window.clear()
            result['action'] = 'reference_window_updated'

        return result


class RealTimePredictor:
    """Real-time predictions with monitoring"""

    def __init__(self, model: Any):
        """
        Initialize real-time predictor

        Args:
            model: Trained model
        """
        self.model = model
        self.prediction_count = 0
        self.prediction_times = deque(maxlen=1000)
        self.confidence_scores = deque(maxlen=1000)

    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Make real-time predictions

        Args:
            X: Features

        Returns:
            Predictions with metadata
        """
        import time

        start_time = time.time()

        # Predictions
        predictions = self.model.predict(X)

        # Confidence scores (if available)
        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(X)
            confidence = np.max(probas, axis=1)
            self.confidence_scores.extend(confidence)
        else:
            confidence = None

        elapsed = time.time() - start_time
        self.prediction_times.append(elapsed)
        self.prediction_count += len(X)

        return {
            'predictions': predictions.tolist(),
            'confidence': confidence.tolist() if confidence is not None else None,
            'metadata': {
                'prediction_time_ms': elapsed * 1000,
                'samples_processed': len(X),
                'total_predictions': self.prediction_count,
                'avg_prediction_time_ms': np.mean(self.prediction_times) * 1000
            }
        }


def main():
    """CLI interface"""
    if len(sys.argv) < 2:
        print("Usage: python online_learner.py <mode>")
        print("Modes: demo, aggregator_demo, drift_demo")
        sys.exit(1)

    mode = sys.argv[1]

    try:
        if mode == 'demo':
            # Demo online learning
            np.random.seed(42)

            # Generate synthetic streaming data
            learner = OnlineLearner(task_type='classification', model_type='sgd')

            results = []
            for batch_num in range(5):
                X = pd.DataFrame(np.random.randn(100, 5))
                y = pd.Series(np.random.randint(0, 2, 100))

                update = learner.partial_fit(X, y, classes=[0, 1])
                results.append(update)

            summary = learner.get_performance_summary()

            result = {
                'updates': results,
                'summary': summary
            }

        elif mode == 'aggregator_demo':
            # Demo streaming aggregator
            aggregator = StreamingAggregator(window_size=100)

            np.random.seed(42)
            values = np.random.normal(100, 15, 200)

            updates = []
            for v in values[::20]:  # Show every 20th update
                stats = aggregator.update(v)
                updates.append(stats)

            result = {'updates': updates}

        elif mode == 'drift_demo':
            # Demo drift detection
            detector = ChangeDetector(window_size=50)

            np.random.seed(42)
            # Generate data with concept drift
            data_before = np.random.normal(100, 10, 100)
            data_after = np.random.normal(110, 10, 100)  # Drift!

            updates = []
            for v in np.concatenate([data_before, data_after]):
                detection = detector.update(v)
                if detection.get('drift_detected'):
                    updates.append(detection)

            result = {'drift_detections': updates}

        else:
            result = {'error': f'Unknown mode: {mode}'}

        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

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