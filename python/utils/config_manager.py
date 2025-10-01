#!/usr/bin/env python3
"""
Configuration Manager Module for ML MCP System
Centralized configuration management for all modules
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class ConfigManager:
    """Manage system configuration"""

    DEFAULT_CONFIG = {
        # System settings
        'system': {
            'environment': 'production',  # production, development, testing
            'log_level': 'INFO',
            'debug_mode': False,
            'verbose': False
        },

        # Memory settings
        'memory': {
            'max_memory_mb': 1024,
            'enable_optimization': True,
            'chunk_size': 10000,
            'enable_streaming': False,
            'gc_threshold': 0.9  # Trigger GC at 90% memory usage
        },

        # Performance settings
        'performance': {
            'enable_caching': True,
            'cache_ttl_seconds': 3600,
            'enable_parallel': True,
            'n_jobs': None,  # None = auto-detect
            'parallel_backend': 'multiprocessing'
        },

        # Data validation settings
        'validation': {
            'strict_mode': True,
            'max_file_size_mb': 1000,
            'max_rows': 10_000_000,
            'max_columns': 10_000,
            'allowed_extensions': ['.csv', '.json', '.xlsx', '.xls', '.parquet', '.h5', '.hdf5', '.tsv', '.txt']
        },

        # Error handling settings
        'error_handling': {
            'enable_recovery': True,
            'max_retries': 3,
            'retry_delay_seconds': 1,
            'enable_graceful_degradation': True,
            'log_errors': True
        },

        # Output settings
        'output': {
            'format': 'json',  # json, csv, excel, html
            'pretty_print': True,
            'include_metadata': True,
            'save_intermediate': False,
            'output_dir': 'results',
            'temp_dir': 'temp'
        },

        # Visualization settings
        'visualization': {
            'default_style': 'seaborn',
            'figure_size': [10, 6],
            'dpi': 100,
            'color_palette': 'default',
            'save_format': 'png',
            'interactive': False
        },

        # ML model settings
        'ml': {
            'random_state': 42,
            'test_size': 0.2,
            'cv_folds': 5,
            'enable_model_caching': True,
            'model_cache_dir': 'temp/models',
            'max_training_time_seconds': 300
        },

        # Logging settings
        'logging': {
            'enable': True,
            'log_dir': 'logs',
            'log_file': 'ml-mcp.log',
            'max_log_size_mb': 100,
            'backup_count': 5
        }
    }

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager

        Args:
            config_file: Path to configuration file (JSON)
        """
        self.config_file = config_file
        self.config = self.DEFAULT_CONFIG.copy()

        # Load from file if provided
        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)

        # Override with environment variables
        self.load_from_env()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            key: Configuration key (e.g., 'memory.max_memory_mb')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation

        Args:
            key: Configuration key (e.g., 'memory.max_memory_mb')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config

        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the value
        config[keys[-1]] = value

    def load_from_file(self, file_path: str):
        """
        Load configuration from JSON file

        Args:
            file_path: Path to JSON configuration file
        """
        with open(file_path, 'r') as f:
            loaded_config = json.load(f)

        # Deep merge with default config
        self.config = self._deep_merge(self.config, loaded_config)

    def save_to_file(self, file_path: Optional[str] = None):
        """
        Save configuration to JSON file

        Args:
            file_path: Path to save configuration (uses self.config_file if None)
        """
        file_path = file_path or self.config_file

        if not file_path:
            raise ValueError("No file path specified")

        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def load_from_env(self):
        """Load configuration from environment variables"""
        # Environment variable format: MLMCP_SECTION_KEY
        # Example: MLMCP_MEMORY_MAX_MEMORY_MB=2048

        env_prefix = 'MLMCP_'

        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(env_prefix):].lower()

                # Convert underscores to dot notation
                # MEMORY_MAX_MEMORY_MB -> memory.max_memory_mb
                parts = config_key.split('_')

                # Try to find matching config key
                if len(parts) >= 2:
                    section = parts[0]
                    setting = '_'.join(parts[1:])
                    full_key = f"{section}.{setting}"

                    # Convert value to appropriate type
                    converted_value = self._convert_env_value(value)

                    # Set the value
                    self.set(full_key, converted_value)

    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type"""
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False

        # Number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # String
        return value

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge two dictionaries"""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section

        Args:
            section: Section name (e.g., 'memory', 'performance')

        Returns:
            Dictionary of section configuration
        """
        return self.config.get(section, {})

    def reset_to_default(self):
        """Reset configuration to default values"""
        self.config = self.DEFAULT_CONFIG.copy()

    def validate(self) -> tuple:
        """
        Validate configuration values

        Returns:
            (is_valid, list of errors)
        """
        errors = []

        # Validate memory settings
        max_memory = self.get('memory.max_memory_mb')
        if max_memory <= 0:
            errors.append("memory.max_memory_mb must be positive")

        # Validate chunk size
        chunk_size = self.get('memory.chunk_size')
        if chunk_size <= 0:
            errors.append("memory.chunk_size must be positive")

        # Validate file size limits
        max_file_size = self.get('validation.max_file_size_mb')
        if max_file_size <= 0:
            errors.append("validation.max_file_size_mb must be positive")

        # Validate test size
        test_size = self.get('ml.test_size')
        if not 0 < test_size < 1:
            errors.append("ml.test_size must be between 0 and 1")

        # Validate CV folds
        cv_folds = self.get('ml.cv_folds')
        if cv_folds < 2:
            errors.append("ml.cv_folds must be at least 2")

        return len(errors) == 0, errors

    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            'environment': self.get('system.environment'),
            'debug_mode': self.get('system.debug_mode'),
            'memory_limit_mb': self.get('memory.max_memory_mb'),
            'caching_enabled': self.get('performance.enable_caching'),
            'parallel_enabled': self.get('performance.enable_parallel'),
            'strict_validation': self.get('validation.strict_mode'),
            'error_recovery': self.get('error_handling.enable_recovery'),
            'output_dir': self.get('output.output_dir'),
            'config_file': self.config_file
        }


# Global configuration instance
_global_config = None


def get_config(config_file: Optional[str] = None) -> ConfigManager:
    """
    Get global configuration instance

    Args:
        config_file: Path to configuration file

    Returns:
        ConfigManager instance
    """
    global _global_config

    if _global_config is None:
        # Look for config file in standard locations
        if config_file is None:
            config_paths = [
                'config.json',
                'ml-mcp-config.json',
                os.path.expanduser('~/.ml-mcp/config.json'),
                '/etc/ml-mcp/config.json'
            ]

            for path in config_paths:
                if Path(path).exists():
                    config_file = path
                    break

        _global_config = ConfigManager(config_file)

    return _global_config


def create_default_config(file_path: str = 'config.json'):
    """
    Create a default configuration file

    Args:
        file_path: Path to save configuration
    """
    config = ConfigManager()
    config.save_to_file(file_path)
    print(f"Default configuration created: {file_path}")


def main():
    """CLI interface for configuration management"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python config_manager.py <action> [args]")
        print("Actions:")
        print("  show              - Show current configuration")
        print("  get <key>         - Get configuration value")
        print("  set <key> <value> - Set configuration value")
        print("  create [file]     - Create default config file")
        print("  validate          - Validate configuration")
        sys.exit(1)

    action = sys.argv[1]
    config = get_config()

    try:
        if action == 'show':
            print(json.dumps(config.config, indent=2))

        elif action == 'get' and len(sys.argv) >= 3:
            key = sys.argv[2]
            value = config.get(key)
            print(json.dumps({key: value}, indent=2))

        elif action == 'set' and len(sys.argv) >= 4:
            key = sys.argv[2]
            value = sys.argv[3]
            config.set(key, config._convert_env_value(value))
            print(f"Set {key} = {value}")

        elif action == 'create':
            file_path = sys.argv[2] if len(sys.argv) >= 3 else 'config.json'
            create_default_config(file_path)

        elif action == 'validate':
            is_valid, errors = config.validate()
            result = {
                'valid': is_valid,
                'errors': errors
            }
            print(json.dumps(result, indent=2))

        elif action == 'summary':
            print(json.dumps(config.get_summary(), indent=2))

        else:
            print(f"Unknown action: {action}")
            sys.exit(1)

    except Exception as e:
        print(json.dumps({'success': False, 'error': str(e)}, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()