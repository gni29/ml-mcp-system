#!/usr/bin/env python3
"""
Performance Monitor Module for ML MCP System
Tracks and logs performance metrics for operations
"""

import time
import psutil
import json
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime
from functools import wraps
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class PerformanceMonitor:
    """Monitor and track performance metrics"""

    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize performance monitor

        Args:
            log_file: Path to log file for storing metrics
        """
        self.log_file = log_file
        self.metrics = []
        self.current_operation = None
        self.start_time = None
        self.start_memory = None

    def start_monitoring(self, operation_name: str):
        """Start monitoring an operation"""
        self.current_operation = operation_name
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()

        print(f"Started: {operation_name}", file=sys.stderr)

    def stop_monitoring(self, additional_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Stop monitoring and record metrics

        Args:
            additional_info: Additional metrics to record

        Returns:
            Performance metrics
        """
        if self.current_operation is None:
            raise RuntimeError("No operation being monitored")

        end_time = time.time()
        end_memory = self._get_memory_usage()

        elapsed_time = end_time - self.start_time
        memory_delta = end_memory - self.start_memory

        metrics = {
            'operation': self.current_operation,
            'timestamp': datetime.now().isoformat(),
            'elapsed_time_seconds': round(elapsed_time, 3),
            'start_memory_mb': round(self.start_memory, 2),
            'end_memory_mb': round(end_memory, 2),
            'memory_delta_mb': round(memory_delta, 2),
            'cpu_percent': psutil.cpu_percent(interval=0.1)
        }

        if additional_info:
            metrics.update(additional_info)

        self.metrics.append(metrics)

        if self.log_file:
            self._save_metrics()

        print(f"Completed: {self.current_operation} in {elapsed_time:.2f}s", file=sys.stderr)

        # Reset state
        self.current_operation = None
        self.start_time = None
        self.start_memory = None

        return metrics

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def _save_metrics(self):
        """Save metrics to log file"""
        if not self.log_file:
            return

        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Append to existing file or create new
        mode = 'a' if log_path.exists() else 'w'
        with open(log_path, mode) as f:
            for metric in self.metrics:
                f.write(json.dumps(metric, default=str) + '\n')

        self.metrics = []  # Clear after saving

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all recorded metrics"""
        if not self.metrics:
            return {'message': 'No metrics recorded'}

        operations = {}
        for metric in self.metrics:
            op_name = metric['operation']
            if op_name not in operations:
                operations[op_name] = {
                    'count': 0,
                    'total_time': 0,
                    'total_memory': 0,
                    'times': [],
                    'memory_deltas': []
                }

            operations[op_name]['count'] += 1
            operations[op_name]['total_time'] += metric['elapsed_time_seconds']
            operations[op_name]['total_memory'] += metric['memory_delta_mb']
            operations[op_name]['times'].append(metric['elapsed_time_seconds'])
            operations[op_name]['memory_deltas'].append(metric['memory_delta_mb'])

        # Calculate statistics
        summary = {}
        for op_name, data in operations.items():
            times = data['times']
            memory = data['memory_deltas']

            summary[op_name] = {
                'count': data['count'],
                'avg_time_seconds': round(data['total_time'] / data['count'], 3),
                'min_time_seconds': round(min(times), 3),
                'max_time_seconds': round(max(times), 3),
                'avg_memory_mb': round(data['total_memory'] / data['count'], 2),
                'total_memory_mb': round(data['total_memory'], 2)
            }

        return summary


def timer(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        print(f"Starting {func.__name__}...", file=sys.stderr)

        try:
            result = func(*args, **kwargs)

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)

            elapsed = end_time - start_time
            memory_delta = end_memory - start_memory

            print(f"Completed {func.__name__} in {elapsed:.2f}s (Δ memory: {memory_delta:+.1f} MB)",
                  file=sys.stderr)

            return result

        except Exception as e:
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"Failed {func.__name__} after {elapsed:.2f}s: {str(e)}", file=sys.stderr)
            raise

    return wrapper


def profile_dataframe_operation(operation_name: str):
    """Decorator for profiling DataFrame operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = PerformanceMonitor()
            monitor.start_monitoring(operation_name)

            result = func(*args, **kwargs)

            # Try to get DataFrame metrics
            additional_info = {}
            if hasattr(result, 'shape'):
                additional_info['result_shape'] = result.shape
                additional_info['result_memory_mb'] = result.memory_usage(deep=True).sum() / (1024 * 1024)

            monitor.stop_monitoring(additional_info)

            return result

        return wrapper
    return decorator


class ProgressTracker:
    """Track progress of long-running operations"""

    def __init__(self, total_items: int, description: str = "Processing"):
        """
        Initialize progress tracker

        Args:
            total_items: Total number of items to process
            description: Description of the operation
        """
        self.total_items = total_items
        self.description = description
        self.current_item = 0
        self.start_time = time.time()
        self.last_update = 0

    def update(self, items: int = 1):
        """
        Update progress

        Args:
            items: Number of items processed
        """
        self.current_item += items
        current_time = time.time()

        # Update every 1 second or at completion
        if current_time - self.last_update >= 1.0 or self.current_item >= self.total_items:
            self._print_progress()
            self.last_update = current_time

    def _print_progress(self):
        """Print progress bar"""
        percent = (self.current_item / self.total_items) * 100
        elapsed = time.time() - self.start_time

        # Estimate time remaining
        if self.current_item > 0:
            rate = self.current_item / elapsed
            remaining = (self.total_items - self.current_item) / rate
            eta_str = f"ETA: {remaining:.0f}s"
        else:
            eta_str = "ETA: --"

        # Progress bar
        bar_length = 30
        filled = int(bar_length * self.current_item / self.total_items)
        bar = '█' * filled + '░' * (bar_length - filled)

        print(f"\r{self.description}: [{bar}] {percent:.1f}% ({self.current_item}/{self.total_items}) {eta_str}",
              end='', file=sys.stderr)

        if self.current_item >= self.total_items:
            print(f" - Done in {elapsed:.1f}s", file=sys.stderr)


def get_system_resources() -> Dict[str, Any]:
    """Get current system resource usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    return {
        'cpu': {
            'percent': cpu_percent,
            'count': psutil.cpu_count(),
            'count_logical': psutil.cpu_count(logical=True)
        },
        'memory': {
            'total_mb': memory.total / (1024 * 1024),
            'available_mb': memory.available / (1024 * 1024),
            'used_mb': memory.used / (1024 * 1024),
            'percent': memory.percent
        },
        'disk': {
            'total_gb': disk.total / (1024 * 1024 * 1024),
            'free_gb': disk.free / (1024 * 1024 * 1024),
            'used_gb': disk.used / (1024 * 1024 * 1024),
            'percent': disk.percent
        }
    }


def estimate_operation_time(data_size_mb: float, operation_type: str) -> Dict[str, Any]:
    """
    Estimate operation time based on data size and type

    Args:
        data_size_mb: Size of data in MB
        operation_type: Type of operation (basic, ml, visualization)

    Returns:
        Time estimate
    """
    # Rough estimates based on typical performance (adjust based on actual benchmarks)
    time_per_mb = {
        'basic': 0.1,      # 100ms per MB for basic stats
        'ml': 2.0,         # 2s per MB for ML training
        'visualization': 0.5,  # 500ms per MB for visualization
        'clustering': 3.0, # 3s per MB for clustering
        'pca': 1.5        # 1.5s per MB for PCA
    }

    base_time = time_per_mb.get(operation_type, 1.0)
    estimated_seconds = data_size_mb * base_time

    # Add overhead for small datasets
    if data_size_mb < 1:
        estimated_seconds += 1

    return {
        'estimated_seconds': round(estimated_seconds, 1),
        'estimated_minutes': round(estimated_seconds / 60, 1),
        'data_size_mb': data_size_mb,
        'operation_type': operation_type,
        'confidence': 'rough_estimate'
    }


def main():
    """CLI interface for performance monitoring"""
    if len(sys.argv) < 2:
        print("Usage: python performance_monitor.py <action>")
        print("Actions: resources, estimate <size_mb> <operation_type>")
        sys.exit(1)

    action = sys.argv[1]

    try:
        if action == 'resources':
            result = get_system_resources()
        elif action == 'estimate' and len(sys.argv) >= 4:
            size_mb = float(sys.argv[2])
            op_type = sys.argv[3]
            result = estimate_operation_time(size_mb, op_type)
        else:
            result = {'error': 'Invalid action or missing arguments'}

        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(json.dumps({'success': False, 'error': str(e)}, ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()