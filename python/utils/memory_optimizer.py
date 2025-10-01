#!/usr/bin/env python3
"""
Memory Optimizer Module for ML MCP System
Handles large dataset processing with memory efficiency
"""

import pandas as pd
import numpy as np
import gc
import psutil
import json
import sys
from typing import Dict, Any, List, Optional, Iterator
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class MemoryOptimizer:
    """Optimize memory usage for data processing"""

    def __init__(self, memory_limit_mb: int = 1024):
        """
        Initialize memory optimizer

        Args:
            memory_limit_mb: Maximum memory to use in MB
        """
        self.memory_limit_mb = memory_limit_mb
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
            'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / (1024 * 1024)
        }

    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame dtypes to reduce memory usage

        Args:
            df: Input DataFrame

        Returns:
            Optimized DataFrame with reduced memory footprint
        """
        initial_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)

        # Optimize numeric columns
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = self._optimize_int(df[col])

        for col in df.select_dtypes(include=['float']).columns:
            df[col] = self._optimize_float(df[col])

        # Optimize object columns
        for col in df.select_dtypes(include=['object']).columns:
            num_unique = df[col].nunique()
            num_total = len(df[col])

            # Convert to category if less than 50% unique values
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')

        final_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
        reduction = ((initial_memory - final_memory) / initial_memory) * 100

        print(f"Memory optimized: {initial_memory:.2f} MB -> {final_memory:.2f} MB ({reduction:.1f}% reduction)",
              file=sys.stderr)

        return df

    def _optimize_int(self, col: pd.Series) -> pd.Series:
        """Optimize integer column dtype"""
        col_min = col.min()
        col_max = col.max()

        if col_min >= 0:
            if col_max < 255:
                return col.astype(np.uint8)
            elif col_max < 65535:
                return col.astype(np.uint16)
            elif col_max < 4294967295:
                return col.astype(np.uint32)
            else:
                return col.astype(np.uint64)
        else:
            if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                return col.astype(np.int8)
            elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                return col.astype(np.int16)
            elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                return col.astype(np.int32)
            else:
                return col.astype(np.int64)

    def _optimize_float(self, col: pd.Series) -> pd.Series:
        """Optimize float column dtype"""
        col_min = col.min()
        col_max = col.max()

        if col_min > np.finfo(np.float16).min and col_max < np.finfo(np.float16).max:
            return col.astype(np.float32)  # float16 can be unstable
        elif col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
            return col.astype(np.float32)
        else:
            return col.astype(np.float64)

    def chunk_reader(self, file_path: str, chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
        """
        Read large files in chunks

        Args:
            file_path: Path to the data file
            chunk_size: Number of rows per chunk

        Yields:
            DataFrame chunks
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        if extension == '.csv':
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                yield self.optimize_dtypes(chunk)
        elif extension == '.json':
            # JSON doesn't support native chunking, read and split
            df = pd.read_json(file_path)
            for i in range(0, len(df), chunk_size):
                yield self.optimize_dtypes(df.iloc[i:i+chunk_size])
        else:
            raise ValueError(f"Chunked reading not supported for {extension}")

    def process_large_file(self, file_path: str,
                          process_func: callable,
                          chunk_size: int = 10000) -> List[Any]:
        """
        Process large file in chunks with memory management

        Args:
            file_path: Path to data file
            process_func: Function to apply to each chunk
            chunk_size: Rows per chunk

        Returns:
            List of results from processing each chunk
        """
        results = []

        for i, chunk in enumerate(self.chunk_reader(file_path, chunk_size)):
            print(f"Processing chunk {i+1}...", file=sys.stderr)

            # Process chunk
            result = process_func(chunk)
            results.append(result)

            # Force garbage collection
            del chunk
            gc.collect()

            # Check memory usage
            memory = self.get_memory_usage()
            if memory['rss_mb'] > self.memory_limit_mb * 0.9:
                print(f"Warning: High memory usage ({memory['rss_mb']:.1f} MB)", file=sys.stderr)
                gc.collect()

        return results

    def reduce_dataframe_size(self, df: pd.DataFrame,
                             max_rows: Optional[int] = None,
                             sample_fraction: Optional[float] = None) -> pd.DataFrame:
        """
        Reduce DataFrame size by sampling or limiting rows

        Args:
            df: Input DataFrame
            max_rows: Maximum number of rows to keep
            sample_fraction: Fraction of rows to sample (0-1)

        Returns:
            Reduced DataFrame
        """
        if sample_fraction is not None:
            df = df.sample(frac=sample_fraction, random_state=42)
            print(f"Sampled {len(df)} rows ({sample_fraction*100}%)", file=sys.stderr)

        if max_rows is not None and len(df) > max_rows:
            df = df.head(max_rows)
            print(f"Limited to {max_rows} rows", file=sys.stderr)

        return df

    def clear_memory(self):
        """Force garbage collection and memory cleanup"""
        gc.collect()
        memory = self.get_memory_usage()
        print(f"Memory after cleanup: {memory['rss_mb']:.1f} MB", file=sys.stderr)


class StreamProcessor:
    """Process data in streaming fashion for minimal memory usage"""

    def __init__(self):
        self.optimizer = MemoryOptimizer()

    def streaming_stats(self, file_path: str, chunk_size: int = 10000) -> Dict[str, Any]:
        """
        Calculate statistics using streaming algorithm

        Args:
            file_path: Path to data file
            chunk_size: Rows per chunk

        Returns:
            Statistical summary
        """
        n = 0
        means = {}
        m2s = {}  # For variance calculation (Welford's algorithm)
        mins = {}
        maxs = {}

        for chunk in self.optimizer.chunk_reader(file_path, chunk_size):
            numeric_cols = chunk.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                values = chunk[col].dropna()

                if col not in means:
                    means[col] = 0
                    m2s[col] = 0
                    mins[col] = values.min() if len(values) > 0 else float('inf')
                    maxs[col] = values.max() if len(values) > 0 else float('-inf')

                # Update statistics using Welford's algorithm
                for value in values:
                    n += 1
                    delta = value - means[col]
                    means[col] += delta / n
                    delta2 = value - means[col]
                    m2s[col] += delta * delta2

                    mins[col] = min(mins[col], value)
                    maxs[col] = max(maxs[col], value)

        # Calculate final statistics
        stats = {}
        for col in means.keys():
            variance = m2s[col] / (n - 1) if n > 1 else 0
            stats[col] = {
                'mean': means[col],
                'std': np.sqrt(variance),
                'min': mins[col],
                'max': maxs[col],
                'count': n
            }

        return {
            'streaming_statistics': stats,
            'total_rows_processed': n,
            'memory_efficient': True
        }


def analyze_memory_requirements(file_path: str) -> Dict[str, Any]:
    """
    Analyze file and estimate memory requirements

    Args:
        file_path: Path to data file

    Returns:
        Memory analysis report
    """
    file_path = Path(file_path)
    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    # Estimate memory requirements (typically 2-5x file size)
    estimated_memory_mb = file_size_mb * 3

    available_memory = psutil.virtual_memory().available / (1024 * 1024)

    recommendation = "direct_load"
    if estimated_memory_mb > available_memory * 0.5:
        recommendation = "chunked_processing"
    if estimated_memory_mb > available_memory * 0.8:
        recommendation = "streaming_only"

    return {
        'file_size_mb': round(file_size_mb, 2),
        'estimated_memory_mb': round(estimated_memory_mb, 2),
        'available_memory_mb': round(available_memory, 2),
        'recommendation': recommendation,
        'can_load_directly': estimated_memory_mb < available_memory * 0.5,
        'requires_chunking': estimated_memory_mb > available_memory * 0.5
    }


def main():
    """CLI interface for memory optimizer"""
    if len(sys.argv) < 2:
        print("Usage: python memory_optimizer.py <file_path> [action]")
        print("Actions: analyze, optimize, stats")
        sys.exit(1)

    file_path = sys.argv[1]
    action = sys.argv[2] if len(sys.argv) > 2 else 'analyze'

    try:
        if action == 'analyze':
            result = analyze_memory_requirements(file_path)
        elif action == 'optimize':
            optimizer = MemoryOptimizer()
            df = pd.read_csv(file_path)
            df_optimized = optimizer.optimize_dtypes(df)
            result = {
                'success': True,
                'original_memory_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'optimized_memory_mb': df_optimized.memory_usage(deep=True).sum() / (1024 * 1024)
            }
        elif action == 'stats':
            processor = StreamProcessor()
            result = processor.streaming_stats(file_path)
        else:
            result = {'error': f'Unknown action: {action}'}

        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    except Exception as e:
        print(json.dumps({'success': False, 'error': str(e)}, ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()