#!/usr/bin/env python3
"""
Parallel Processor Module for ML MCP System
Enables parallel processing for improved performance
"""

import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Callable, List, Any, Dict, Optional, Iterable
import json
import sys
from functools import partial
import warnings
warnings.filterwarnings('ignore')


class ParallelProcessor:
    """Parallel processing for data operations"""

    def __init__(self, n_jobs: Optional[int] = None, backend: str = 'multiprocessing'):
        """
        Initialize parallel processor

        Args:
            n_jobs: Number of parallel jobs (None = auto-detect)
            backend: 'multiprocessing' or 'threading'
        """
        self.n_jobs = n_jobs or max(1, cpu_count() - 1)
        self.backend = backend

    def map(self, func: Callable, items: Iterable, chunksize: int = 1) -> List[Any]:
        """
        Apply function to items in parallel

        Args:
            func: Function to apply
            items: Iterable of items to process
            chunksize: Items per worker

        Returns:
            List of results
        """
        items_list = list(items)

        if len(items_list) <= 1 or self.n_jobs == 1:
            # No benefit from parallelization
            return [func(item) for item in items_list]

        if self.backend == 'multiprocessing':
            with Pool(processes=self.n_jobs) as pool:
                results = pool.map(func, items_list, chunksize=chunksize)
        else:  # threading
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                results = list(executor.map(func, items_list, chunksize=chunksize))

        return results

    def map_async(self, func: Callable, items: Iterable,
                  callback: Optional[Callable] = None) -> List[Any]:
        """
        Apply function to items asynchronously

        Args:
            func: Function to apply
            items: Iterable of items
            callback: Optional callback for each result

        Returns:
            List of results
        """
        items_list = list(items)
        results = []

        if self.backend == 'multiprocessing':
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {executor.submit(func, item): i for i, item in enumerate(items_list)}

                for future in as_completed(futures):
                    result = future.result()
                    results.append((futures[future], result))
                    if callback:
                        callback(result)

            # Sort by original order
            results.sort(key=lambda x: x[0])
            return [r[1] for r in results]
        else:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {executor.submit(func, item): i for i, item in enumerate(items_list)}

                for future in as_completed(futures):
                    result = future.result()
                    results.append((futures[future], result))
                    if callback:
                        callback(result)

            results.sort(key=lambda x: x[0])
            return [r[1] for r in results]

    def apply_to_dataframe_columns(self, df: pd.DataFrame,
                                   func: Callable,
                                   columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Apply function to DataFrame columns in parallel

        Args:
            df: Input DataFrame
            func: Function to apply to each column
            columns: Columns to process (None = all)

        Returns:
            DataFrame with processed columns
        """
        columns = columns or df.columns.tolist()

        def process_column(col_name):
            return col_name, func(df[col_name])

        results = self.map(process_column, columns)

        # Reconstruct DataFrame
        result_df = df.copy()
        for col_name, processed_col in results:
            result_df[col_name] = processed_col

        return result_df

    def split_dataframe_process(self, df: pd.DataFrame,
                                func: Callable,
                                combine_func: Optional[Callable] = None) -> Any:
        """
        Split DataFrame, process in parallel, and combine

        Args:
            df: Input DataFrame
            func: Function to apply to each chunk
            combine_func: Function to combine results (default: pd.concat)

        Returns:
            Combined result
        """
        # Split DataFrame into chunks
        chunk_size = max(1, len(df) // self.n_jobs)
        chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

        print(f"Processing {len(chunks)} chunks in parallel...", file=sys.stderr)

        # Process chunks
        results = self.map(func, chunks)

        # Combine results
        if combine_func:
            return combine_func(results)
        else:
            # Default: concatenate if results are DataFrames
            if results and isinstance(results[0], pd.DataFrame):
                return pd.concat(results, ignore_index=True)
            return results


class BatchProcessor:
    """Process multiple files or datasets in batches"""

    def __init__(self, n_jobs: Optional[int] = None):
        """
        Initialize batch processor

        Args:
            n_jobs: Number of parallel jobs
        """
        self.processor = ParallelProcessor(n_jobs=n_jobs)

    def process_files(self, file_paths: List[str],
                     process_func: Callable[[str], Any]) -> List[Any]:
        """
        Process multiple files in parallel

        Args:
            file_paths: List of file paths
            process_func: Function to process each file

        Returns:
            List of results
        """
        print(f"Processing {len(file_paths)} files in parallel...", file=sys.stderr)
        return self.processor.map(process_func, file_paths)

    def process_with_progress(self, items: List[Any],
                             process_func: Callable) -> List[Any]:
        """
        Process items with progress tracking

        Args:
            items: Items to process
            process_func: Function to process each item

        Returns:
            List of results
        """
        total = len(items)
        results = []
        completed = 0

        def callback(result):
            nonlocal completed
            completed += 1
            percent = (completed / total) * 100
            print(f"\rProgress: {completed}/{total} ({percent:.1f}%)", end='', file=sys.stderr)

        results = self.processor.map_async(process_func, items, callback=callback)
        print()  # New line after progress
        return results


def parallel_apply(df: pd.DataFrame, func: Callable,
                  axis: int = 0, n_jobs: Optional[int] = None) -> pd.Series:
    """
    Parallel version of DataFrame.apply()

    Args:
        df: Input DataFrame
        func: Function to apply
        axis: 0 for columns, 1 for rows
        n_jobs: Number of parallel jobs

    Returns:
        Series with results
    """
    processor = ParallelProcessor(n_jobs=n_jobs)

    if axis == 0:
        # Apply to each column
        def process_col(col_name):
            return func(df[col_name])

        results = processor.map(process_col, df.columns)
        return pd.Series(results, index=df.columns)
    else:
        # Apply to each row
        chunk_size = max(1, len(df) // processor.n_jobs)
        chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

        def process_chunk(chunk):
            return chunk.apply(func, axis=1)

        results = processor.map(process_chunk, chunks)
        return pd.concat(results)


def parallel_groupby_apply(df: pd.DataFrame, groupby_col: str,
                          func: Callable, n_jobs: Optional[int] = None) -> pd.DataFrame:
    """
    Parallel version of groupby().apply()

    Args:
        df: Input DataFrame
        groupby_col: Column to group by
        func: Function to apply to each group
        n_jobs: Number of parallel jobs

    Returns:
        Combined results
    """
    processor = ParallelProcessor(n_jobs=n_jobs)

    groups = df.groupby(groupby_col)
    group_keys = list(groups.groups.keys())

    def process_group(key):
        group_df = groups.get_group(key)
        return func(group_df)

    results = processor.map(process_group, group_keys)

    # Combine results
    if results and isinstance(results[0], pd.DataFrame):
        return pd.concat(results, ignore_index=True)
    elif results and isinstance(results[0], pd.Series):
        return pd.concat(results)
    else:
        return pd.Series(results, index=group_keys)


def parallel_correlation(df: pd.DataFrame, n_jobs: Optional[int] = None) -> pd.DataFrame:
    """
    Compute correlation matrix in parallel

    Args:
        df: Input DataFrame
        n_jobs: Number of parallel jobs

    Returns:
        Correlation matrix
    """
    processor = ParallelProcessor(n_jobs=n_jobs)
    numeric_df = df.select_dtypes(include=[np.number])
    columns = numeric_df.columns.tolist()

    def compute_correlations(col1):
        correlations = {}
        for col2 in columns:
            correlations[col2] = numeric_df[col1].corr(numeric_df[col2])
        return col1, correlations

    results = processor.map(compute_correlations, columns)

    # Build correlation matrix
    corr_dict = {col: corr_vals for col, corr_vals in results}
    return pd.DataFrame(corr_dict).T


def parallel_feature_engineering(df: pd.DataFrame,
                                feature_funcs: List[Callable],
                                n_jobs: Optional[int] = None) -> pd.DataFrame:
    """
    Apply multiple feature engineering functions in parallel

    Args:
        df: Input DataFrame
        feature_funcs: List of functions that take df and return new columns
        n_jobs: Number of parallel jobs

    Returns:
        DataFrame with new features
    """
    processor = ParallelProcessor(n_jobs=n_jobs)

    def apply_func(func):
        return func(df)

    new_features = processor.map(apply_func, feature_funcs)

    # Combine with original DataFrame
    result_df = df.copy()
    for feature in new_features:
        if isinstance(feature, pd.DataFrame):
            result_df = pd.concat([result_df, feature], axis=1)
        elif isinstance(feature, pd.Series):
            result_df[feature.name] = feature

    return result_df


def optimize_parallel_jobs(data_size_mb: float, operation_complexity: str = 'medium') -> int:
    """
    Determine optimal number of parallel jobs

    Args:
        data_size_mb: Size of data in MB
        operation_complexity: 'low', 'medium', or 'high'

    Returns:
        Optimal number of jobs
    """
    max_jobs = cpu_count()

    # Complexity factors
    complexity_factors = {
        'low': 1.0,      # Simple operations benefit from max parallelization
        'medium': 0.75,  # Medium operations have some overhead
        'high': 0.5      # Complex operations have high overhead
    }

    factor = complexity_factors.get(operation_complexity, 0.75)

    # For small datasets, parallelization overhead may not be worth it
    if data_size_mb < 1:
        return 1
    elif data_size_mb < 10:
        return min(2, max_jobs)
    elif data_size_mb < 100:
        return max(1, int(max_jobs * factor * 0.5))
    else:
        return max(1, int(max_jobs * factor))


def main():
    """CLI interface for parallel processor"""
    if len(sys.argv) < 2:
        print("Usage: python parallel_processor.py <action>")
        print("Actions: info, optimize <size_mb> <complexity>")
        sys.exit(1)

    action = sys.argv[1]

    try:
        if action == 'info':
            result = {
                'cpu_count': cpu_count(),
                'recommended_jobs': max(1, cpu_count() - 1),
                'backends': ['multiprocessing', 'threading']
            }
        elif action == 'optimize' and len(sys.argv) >= 4:
            size_mb = float(sys.argv[2])
            complexity = sys.argv[3]
            optimal_jobs = optimize_parallel_jobs(size_mb, complexity)
            result = {
                'data_size_mb': size_mb,
                'complexity': complexity,
                'optimal_jobs': optimal_jobs,
                'max_jobs': cpu_count()
            }
        else:
            result = {'error': 'Invalid action or missing arguments'}

        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(json.dumps({'success': False, 'error': str(e)}, ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()