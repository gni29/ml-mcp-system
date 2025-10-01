#!/usr/bin/env python3
"""
Input Validator Module for ML MCP System
Validates and sanitizes inputs for security and reliability
"""

import pandas as pd
import numpy as np
import json
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import warnings
warnings.filterwarnings('ignore')


class InputValidator:
    """Validate and sanitize inputs"""

    # Security limits
    MAX_FILE_SIZE_MB = 1000  # 1GB max file size
    MAX_ROWS = 10_000_000    # 10M rows max
    MAX_COLUMNS = 10_000     # 10K columns max
    MAX_STRING_LENGTH = 100_000  # 100K characters max for strings
    MAX_PATH_LENGTH = 500    # Max path length

    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'.csv', '.json', '.xlsx', '.xls', '.parquet', '.h5', '.hdf5', '.tsv', '.txt'}

    # Dangerous path patterns
    DANGEROUS_PATTERNS = [
        r'\.\.',  # Path traversal
        r'~',     # Home directory
        r'\$',    # Environment variables
        r'\\\\',  # Network paths
        r'/etc',  # System directories
        r'/sys',
        r'/proc',
        r'C:\\Windows',
        r'C:\\Program Files'
    ]

    def __init__(self, strict_mode: bool = True):
        """
        Initialize validator

        Args:
            strict_mode: Enable strict validation rules
        """
        self.strict_mode = strict_mode
        self.validation_errors = []

    def validate_file_path(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate file path for security

        Args:
            file_path: Path to validate

        Returns:
            (is_valid, error_message)
        """
        # Check length
        if len(file_path) > self.MAX_PATH_LENGTH:
            return False, f"Path too long (max {self.MAX_PATH_LENGTH} characters)"

        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, file_path, re.IGNORECASE):
                return False, f"Dangerous path pattern detected: {pattern}"

        # Convert to Path object
        try:
            path = Path(file_path)
        except Exception as e:
            return False, f"Invalid path: {str(e)}"

        # Check if path exists
        if not path.exists():
            return False, f"File not found: {file_path}"

        # Check if it's a file (not directory)
        if not path.is_file():
            return False, f"Not a file: {file_path}"

        # Check file extension
        if path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            return False, f"Unsupported file type: {path.suffix}"

        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.MAX_FILE_SIZE_MB:
            return False, f"File too large: {file_size_mb:.1f} MB (max {self.MAX_FILE_SIZE_MB} MB)"

        return True, None

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame for safety and quality

        Args:
            df: DataFrame to validate

        Returns:
            (is_valid, list of warnings/errors)
        """
        issues = []

        # Check dimensions
        rows, cols = df.shape

        if rows > self.MAX_ROWS:
            issues.append(f"Too many rows: {rows} (max {self.MAX_ROWS})")
            if self.strict_mode:
                return False, issues

        if cols > self.MAX_COLUMNS:
            issues.append(f"Too many columns: {cols} (max {self.MAX_COLUMNS})")
            if self.strict_mode:
                return False, issues

        # Check for empty DataFrame
        if rows == 0:
            issues.append("DataFrame is empty")
            return False, issues

        if cols == 0:
            issues.append("DataFrame has no columns")
            return False, issues

        # Check for suspicious column names
        for col in df.columns:
            if not self._is_safe_string(str(col)):
                issues.append(f"Suspicious column name: {col}")

        # Check for extremely long strings
        for col in df.select_dtypes(include=['object']).columns:
            max_length = df[col].astype(str).str.len().max()
            if max_length > self.MAX_STRING_LENGTH:
                issues.append(f"Column '{col}' contains very long strings (max: {max_length})")

        # Check for all-null columns
        null_cols = df.columns[df.isnull().all()].tolist()
        if null_cols:
            issues.append(f"Columns with all null values: {', '.join(null_cols)}")

        # Check for duplicate column names
        if df.columns.duplicated().any():
            duplicates = df.columns[df.columns.duplicated()].tolist()
            issues.append(f"Duplicate column names: {', '.join(duplicates)}")
            if self.strict_mode:
                return False, issues

        return len(issues) == 0 or not self.strict_mode, issues

    def validate_parameter(self, param_name: str, param_value: Any,
                          param_type: type, allowed_values: Optional[List] = None,
                          min_value: Optional[float] = None,
                          max_value: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate a parameter value

        Args:
            param_name: Name of parameter
            param_value: Value to validate
            param_type: Expected type
            allowed_values: Optional list of allowed values
            min_value: Optional minimum value (for numbers)
            max_value: Optional maximum value (for numbers)

        Returns:
            (is_valid, error_message)
        """
        # Check type
        if not isinstance(param_value, param_type):
            return False, f"{param_name} must be {param_type.__name__}, got {type(param_value).__name__}"

        # Check allowed values
        if allowed_values is not None and param_value not in allowed_values:
            return False, f"{param_name} must be one of {allowed_values}, got {param_value}"

        # Check numeric range
        if isinstance(param_value, (int, float)):
            if min_value is not None and param_value < min_value:
                return False, f"{param_name} must be >= {min_value}, got {param_value}"
            if max_value is not None and param_value > max_value:
                return False, f"{param_name} must be <= {max_value}, got {param_value}"

        # Check string safety
        if isinstance(param_value, str):
            if not self._is_safe_string(param_value):
                return False, f"{param_name} contains suspicious characters"

        return True, None

    def validate_column_exists(self, df: pd.DataFrame, column: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that a column exists in DataFrame

        Args:
            df: DataFrame
            column: Column name

        Returns:
            (is_valid, error_message)
        """
        if column not in df.columns:
            available = ', '.join(df.columns[:10])  # Show first 10
            return False, f"Column '{column}' not found. Available: {available}"

        return True, None

    def validate_numeric_column(self, df: pd.DataFrame, column: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that a column is numeric

        Args:
            df: DataFrame
            column: Column name

        Returns:
            (is_valid, error_message)
        """
        exists, error = self.validate_column_exists(df, column)
        if not exists:
            return False, error

        if not pd.api.types.is_numeric_dtype(df[column]):
            return False, f"Column '{column}' must be numeric, got {df[column].dtype}"

        return True, None

    def sanitize_column_name(self, column_name: str) -> str:
        """
        Sanitize column name for safety

        Args:
            column_name: Original column name

        Returns:
            Sanitized column name
        """
        # Remove dangerous characters
        sanitized = re.sub(r'[^\w\s-]', '', str(column_name))

        # Replace spaces with underscores
        sanitized = re.sub(r'\s+', '_', sanitized)

        # Limit length
        if len(sanitized) > 100:
            sanitized = sanitized[:100]

        # Ensure not empty
        if not sanitized:
            sanitized = "column"

        return sanitized

    def sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitize DataFrame for safety

        Args:
            df: DataFrame to sanitize

        Returns:
            Sanitized DataFrame
        """
        df = df.copy()

        # Sanitize column names
        df.columns = [self.sanitize_column_name(col) for col in df.columns]

        # Handle duplicate column names
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            dup_indices = cols[cols == dup].index
            cols[dup_indices] = [f"{dup}_{i}" for i in range(len(dup_indices))]
        df.columns = cols

        # Remove all-null columns
        df = df.dropna(axis=1, how='all')

        # Limit string lengths
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str[:self.MAX_STRING_LENGTH]

        # Replace inf values
        df = df.replace([np.inf, -np.inf], np.nan)

        return df

    def _is_safe_string(self, s: str) -> bool:
        """Check if string is safe (no SQL injection, code injection, etc.)"""
        dangerous_patterns = [
            r';\s*DROP',  # SQL injection
            r';\s*DELETE',
            r';\s*UPDATE',
            r'<script',   # XSS
            r'javascript:',
            r'eval\(',    # Code injection
            r'exec\(',
            r'__import__',
            r'\.\./',     # Path traversal
            r'\.\.\\'
        ]

        s_upper = s.upper()
        for pattern in dangerous_patterns:
            if re.search(pattern, s_upper, re.IGNORECASE):
                return False

        return True

    def validate_json_input(self, json_str: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Validate and parse JSON input

        Args:
            json_str: JSON string

        Returns:
            (is_valid, parsed_data, error_message)
        """
        # Check length
        if len(json_str) > 1_000_000:  # 1MB max
            return False, None, "JSON too large (max 1MB)"

        # Try to parse
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return False, None, f"Invalid JSON: {str(e)}"

        # Check depth (prevent deeply nested JSON attacks)
        max_depth = self._get_json_depth(data)
        if max_depth > 50:
            return False, None, f"JSON too deeply nested (depth: {max_depth}, max: 50)"

        return True, data, None

    def _get_json_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Get maximum depth of JSON object"""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._get_json_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._get_json_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth


class SecurityChecker:
    """Additional security checks"""

    @staticmethod
    def check_file_permissions(file_path: str) -> Dict[str, Any]:
        """Check file permissions"""
        path = Path(file_path)

        import stat
        mode = path.stat().st_mode

        return {
            'readable': bool(mode & stat.S_IRUSR),
            'writable': bool(mode & stat.S_IWUSR),
            'executable': bool(mode & stat.S_IXUSR),
            'is_symlink': path.is_symlink()
        }

    @staticmethod
    def sanitize_output_path(output_path: str, base_dir: str = "results") -> str:
        """
        Sanitize output path to prevent writing to dangerous locations

        Args:
            output_path: Requested output path
            base_dir: Base directory for outputs

        Returns:
            Safe output path
        """
        # Convert to Path
        path = Path(output_path)

        # Get just the filename
        filename = path.name

        # Create safe path in base directory
        safe_path = Path(base_dir) / filename

        # Ensure base directory exists
        safe_path.parent.mkdir(parents=True, exist_ok=True)

        return str(safe_path)


def main():
    """CLI interface for input validation"""
    if len(sys.argv) < 3:
        print("Usage: python input_validator.py <action> <arg>")
        print("Actions: validate_path <path>, validate_csv <path>")
        sys.exit(1)

    action = sys.argv[1]
    arg = sys.argv[2]

    validator = InputValidator()

    try:
        if action == 'validate_path':
            is_valid, error = validator.validate_file_path(arg)
            result = {
                'valid': is_valid,
                'error': error,
                'path': arg
            }
        elif action == 'validate_csv':
            is_valid, error = validator.validate_file_path(arg)
            if not is_valid:
                result = {'valid': False, 'error': error}
            else:
                df = pd.read_csv(arg)
                df_valid, issues = validator.validate_dataframe(df)
                result = {
                    'valid': df_valid,
                    'issues': issues,
                    'shape': df.shape,
                    'path': arg
                }
        else:
            result = {'error': f'Unknown action: {action}'}

        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(json.dumps({'success': False, 'error': str(e)}, ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()