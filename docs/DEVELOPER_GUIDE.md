# ML MCP System - Developer Guide

## 🏗️ Architecture Overview

The ML MCP System is built on a modular, plugin-based architecture that allows for easy extension and customization. This guide covers everything you need to know to develop, extend, and maintain the system.

### System Architecture

```
ML MCP System
├── ml-mcp-analysis/         # Statistical Analysis Module
│   ├── services/           # MCP service handlers
│   ├── python/            # Python analysis scripts
│   └── package.json       # Node.js MCP server
├── ml-mcp-ml/             # Machine Learning Module
│   ├── services/          # ML service handlers
│   ├── python/           # ML training scripts
│   └── package.json      # Node.js MCP server
├── ml-mcp-visualization/  # Visualization Module
│   ├── services/         # Visualization handlers
│   ├── python/          # Visualization scripts
│   └── package.json     # Node.js MCP server
├── ml-mcp-shared/        # Shared Utilities
│   ├── utils/           # Common utilities
│   ├── python/         # Shared Python modules
│   └── package.json    # Shared components
└── docs/               # Documentation
```

## 🔧 Development Environment Setup

### Prerequisites
- **Node.js** 18+ with npm
- **Python** 3.8+ with pip
- **Git** for version control
- **VS Code** (recommended) or preferred IDE

### Initial Setup
```bash
# 1. Clone the repository
git clone https://github.com/your-org/ml-mcp-system.git
cd ml-mcp-system

# 2. Install Node.js dependencies
npm install

# 3. Install Python dependencies
pip install -r python/requirements.txt

# 4. Install development dependencies
pip install -r python/requirements-dev.txt

# 5. Set up pre-commit hooks
pre-commit install

# 6. Run tests to verify setup
npm test
python -m pytest python/tests/
```

### Development Tools
```bash
# Code formatting
npm run lint
python -m black python/
python -m isort python/

# Type checking
python -m mypy python/

# Testing
npm run test:watch
python -m pytest python/tests/ --watch
```

## 🏛️ Module Architecture

### MCP Server Structure
Each module follows the standard MCP (Model Context Protocol) pattern:

```javascript
// services/[module]-service.js
import { BaseService } from '@ml-mcp-shared/utils/base-service';

export class AnalysisService extends BaseService {
  constructor() {
    super('ml-mcp-analysis');
    this.tools = [
      // Tool definitions
    ];
  }

  async handleToolCall(name, arguments_) {
    // Route to appropriate handler
  }
}
```

### Python Script Structure
All Python scripts follow a consistent pattern:

```python
#!/usr/bin/env python3
"""
Module Description
Brief description of what this module does.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add shared utilities to path
sys.path.append(str(Path(__file__).parent.parent / "ml-mcp-shared" / "python"))

from common_utils import load_data, output_results

def main_function(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main processing function

    Parameters:
    -----------
    params : Dict[str, Any]
        Input parameters from MCP call

    Returns:
    --------
    Dict[str, Any]
        Results dictionary
    """
    try:
        # Input validation
        data_file = params.get('data_file')
        if not data_file:
            raise ValueError("data_file parameter is required")

        # Load and process data
        df = load_data(data_file)

        # Perform analysis
        results = {
            "success": True,
            "data_shape": list(df.shape),
            # ... additional results
        }

        return results

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

def main():
    """Entry point for command-line execution"""
    try:
        input_data = sys.stdin.read()
        params = json.loads(input_data)
        result = main_function(params)
        output_results(result)
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
        output_results(error_result)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## 🔨 Adding New Tools

### Step 1: Create Python Script
Create your analysis script following the standard pattern:

```python
# python/analyzers/custom/my_new_analyzer.py
#!/usr/bin/env python3
"""
My New Analyzer
Custom analysis for specific domain requirements.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

# Standard imports
sys.path.append(str(Path(__file__).parent.parent.parent / "ml-mcp-shared" / "python"))
from common_utils import load_data, output_results

def analyze_custom_metrics(df: pd.DataFrame,
                          target_column: str,
                          custom_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform custom analysis

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_column : str
        Target variable for analysis
    custom_params : Dict[str, Any]
        Custom parameters specific to this analysis

    Returns:
    --------
    Dict[str, Any]
        Analysis results
    """

    # Input validation
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    # Perform your custom analysis
    results = {
        "metric_1": calculate_metric_1(df, target_column),
        "metric_2": calculate_metric_2(df, custom_params),
        "visualizations": create_custom_plots(df, target_column)
    }

    return results

def calculate_metric_1(df: pd.DataFrame, target_column: str) -> float:
    """Calculate custom metric 1"""
    # Your implementation here
    return df[target_column].mean()

def calculate_metric_2(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate custom metric 2"""
    # Your implementation here
    return {"value": 42, "interpretation": "meaningful result"}

def create_custom_plots(df: pd.DataFrame, target_column: str) -> List[str]:
    """Create domain-specific visualizations"""
    import matplotlib.pyplot as plt

    # Create your custom plots
    plt.figure(figsize=(10, 6))
    # ... plotting code ...

    output_file = "custom_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return [output_file]

def main_function(params: Dict[str, Any]) -> Dict[str, Any]:
    """Main execution function"""
    try:
        # Extract parameters
        data_file = params.get('data_file')
        target_column = params.get('target_column')
        custom_params = params.get('custom_params', {})

        # Validation
        if not data_file:
            raise ValueError("data_file parameter is required")
        if not target_column:
            raise ValueError("target_column parameter is required")

        # Load data
        df = load_data(data_file)

        # Perform analysis
        analysis_results = analyze_custom_metrics(df, target_column, custom_params)

        # Format results
        results = {
            "success": True,
            "analysis_type": "custom_metrics",
            "data_info": {
                "shape": list(df.shape),
                "columns": df.columns.tolist()
            },
            "results": analysis_results
        }

        return results

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

def main():
    """CLI entry point"""
    try:
        input_data = sys.stdin.read()
        params = json.loads(input_data)
        result = main_function(params)
        output_results(result)
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
        output_results(error_result)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Step 2: Register Tool in MCP Service
Add your tool to the appropriate service:

```javascript
// ml-mcp-analysis/services/analysis-service.js
export class AnalysisService extends BaseService {
  constructor() {
    super('ml-mcp-analysis');
    this.tools = [
      // ... existing tools ...
      {
        name: 'my_custom_analyzer',
        description: 'Custom analysis for domain-specific requirements',
        inputSchema: {
          type: 'object',
          properties: {
            data_file: {
              type: 'string',
              description: 'Path to the data file'
            },
            target_column: {
              type: 'string',
              description: 'Target variable for analysis'
            },
            custom_params: {
              type: 'object',
              description: 'Custom parameters for analysis',
              properties: {
                param1: { type: 'number' },
                param2: { type: 'string' }
              }
            }
          },
          required: ['data_file', 'target_column']
        }
      }
    ];
  }

  async handleMyCustomAnalyzer(args) {
    const scriptPath = path.join(__dirname, '../python/analyzers/custom/my_new_analyzer.py');
    return this.executePythonScript(scriptPath, args);
  }
}
```

### Step 3: Add Tests
Create comprehensive tests for your new tool:

```python
# python/tests/test_my_new_analyzer.py
import pytest
import pandas as pd
import json
import tempfile
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from analyzers.custom.my_new_analyzer import main_function, analyze_custom_metrics

class TestMyNewAnalyzer:

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'target': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        return data

    @pytest.fixture
    def temp_csv_file(self, sample_data):
        """Create temporary CSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            yield f.name
        Path(f.name).unlink()  # Cleanup

    def test_analyze_custom_metrics_basic(self, sample_data):
        """Test basic functionality of custom metrics"""
        result = analyze_custom_metrics(
            sample_data,
            'target',
            {'param1': 1.0, 'param2': 'test'}
        )

        assert 'metric_1' in result
        assert 'metric_2' in result
        assert result['metric_1'] == 3.3  # Mean of target column

    def test_main_function_success(self, temp_csv_file):
        """Test successful execution of main function"""
        params = {
            'data_file': temp_csv_file,
            'target_column': 'target',
            'custom_params': {'param1': 1.0}
        }

        result = main_function(params)

        assert result['success'] is True
        assert 'results' in result
        assert 'data_info' in result

    def test_main_function_missing_file(self):
        """Test error handling for missing file"""
        params = {
            'data_file': 'nonexistent.csv',
            'target_column': 'target'
        }

        result = main_function(params)

        assert result['success'] is False
        assert 'error' in result

    def test_main_function_missing_column(self, temp_csv_file):
        """Test error handling for missing column"""
        params = {
            'data_file': temp_csv_file,
            'target_column': 'nonexistent_column'
        }

        result = main_function(params)

        assert result['success'] is False
        assert 'error' in result

    def test_parameter_validation(self, temp_csv_file):
        """Test parameter validation"""
        # Missing required parameter
        params = {
            'data_file': temp_csv_file
            # Missing target_column
        }

        result = main_function(params)
        assert result['success'] is False

    @pytest.mark.parametrize("param_value,expected", [
        (1.0, "expected_result_1"),
        (2.0, "expected_result_2"),
        (3.0, "expected_result_3")
    ])
    def test_parameter_variations(self, sample_data, param_value, expected):
        """Test different parameter values"""
        result = analyze_custom_metrics(
            sample_data,
            'target',
            {'param1': param_value}
        )
        # Add your specific assertions based on expected behavior
        assert result is not None

# Integration tests
class TestMyNewAnalyzerIntegration:

    def test_cli_integration(self, temp_csv_file):
        """Test command-line interface"""
        import subprocess

        params = {
            'data_file': temp_csv_file,
            'target_column': 'target'
        }

        result = subprocess.run(
            ['python', 'analyzers/custom/my_new_analyzer.py'],
            input=json.dumps(params),
            text=True,
            capture_output=True,
            cwd=Path(__file__).parent.parent
        )

        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output['success'] is True

# Performance tests
class TestMyNewAnalyzerPerformance:

    def test_large_dataset_performance(self):
        """Test performance with larger datasets"""
        import time

        # Create larger dataset
        large_data = pd.DataFrame({
            'feature1': range(10000),
            'feature2': range(10000, 20000),
            'target': [i * 1.1 for i in range(10000)]
        })

        start_time = time.time()
        result = analyze_custom_metrics(large_data, 'target', {})
        end_time = time.time()

        # Should complete within reasonable time
        assert (end_time - start_time) < 5.0  # 5 seconds
        assert result is not None

if __name__ == "__main__":
    pytest.main([__file__])
```

### Step 4: Add Documentation
Update the API reference and add usage examples:

```markdown
<!-- Add to API_REFERENCE.md -->

### Custom Analyzer (`my_custom_analyzer`)

**Description**: Domain-specific analysis with custom metrics

**Parameters**:
```json
{
  "data_file": "string (required) - Path to data file",
  "target_column": "string (required) - Target variable",
  "custom_params": "object (optional) - Custom parameters"
}
```

**Output**:
```json
{
  "success": true,
  "results": {
    "metric_1": "number - Custom metric 1",
    "metric_2": "object - Custom metric 2 with interpretation"
  }
}
```
```

## 🧪 Testing Framework

### Testing Philosophy
- **Unit Tests**: Test individual functions in isolation
- **Integration Tests**: Test complete workflows
- **Performance Tests**: Ensure acceptable performance
- **Regression Tests**: Prevent breaking changes

### Running Tests
```bash
# Run all tests
npm test
python -m pytest

# Run specific test categories
python -m pytest python/tests/unit/
python -m pytest python/tests/integration/
python -m pytest python/tests/performance/

# Run with coverage
python -m pytest --cov=python/analyzers --cov-report=html

# Run with verbose output
python -m pytest -v --tb=short
```

### Test Data Management
```python
# python/tests/conftest.py
import pytest
import pandas as pd
import tempfile
from pathlib import Path

@pytest.fixture(scope="session")
def sample_datasets():
    """Create standard test datasets"""
    datasets = {}

    # Simple dataset
    datasets['simple'] = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [2, 4, 6, 8, 10],
        'category': ['A', 'B', 'A', 'B', 'A']
    })

    # Complex dataset
    datasets['complex'] = pd.DataFrame({
        'numeric1': np.random.normal(0, 1, 100),
        'numeric2': np.random.normal(5, 2, 100),
        'categorical1': np.random.choice(['X', 'Y', 'Z'], 100),
        'categorical2': np.random.choice(['Alpha', 'Beta'], 100),
        'target': np.random.choice([0, 1], 100)
    })

    return datasets

@pytest.fixture
def temp_data_files(sample_datasets):
    """Create temporary files for testing"""
    files = {}

    for name, df in sample_datasets.items():
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            files[name] = f.name

    yield files

    # Cleanup
    for filepath in files.values():
        Path(filepath).unlink()
```

## 🔍 Debugging and Profiling

### Debugging Tools
```python
# Add to your scripts for debugging
import logging
import pdb

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_function(df, params):
    """Example function with debugging"""
    logger.info(f"Processing dataframe with shape {df.shape}")
    logger.debug(f"Parameters: {params}")

    # Set breakpoint for interactive debugging
    # pdb.set_trace()

    try:
        result = process_data(df, params)
        logger.info("Processing completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error in processing: {e}")
        raise
```

### Performance Profiling
```python
# profiling_example.py
import cProfile
import pstats
from io import StringIO

def profile_function(func, *args, **kwargs):
    """Profile a function's performance"""
    pr = cProfile.Profile()
    pr.enable()

    result = func(*args, **kwargs)

    pr.disable()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()

    print(s.getvalue())
    return result

# Usage
# result = profile_function(my_analysis_function, df, params)
```

### Memory Profiling
```python
# memory_profiling.py
from memory_profiler import profile

@profile
def memory_intensive_function(df):
    """Function decorated for memory profiling"""
    # Your analysis code here
    result = df.groupby('category').agg({'value': 'mean'})
    return result

# Run with: python -m memory_profiler memory_profiling.py
```

## 📦 Packaging and Distribution

### Creating New Modules
```bash
# Create new module structure
mkdir ml-mcp-newmodule
cd ml-mcp-newmodule

# Initialize npm package
npm init -y

# Create standard structure
mkdir -p services python/analyzers tests

# Create package.json
cat > package.json << EOF
{
  "name": "ml-mcp-newmodule",
  "version": "1.0.0",
  "type": "module",
  "main": "index.js",
  "scripts": {
    "start": "node index.js",
    "test": "jest"
  },
  "dependencies": {
    "@modelcontextprotocol/sdk": "^0.4.0"
  }
}
EOF
```

### Version Management
```bash
# Update version
npm version patch  # 1.0.0 -> 1.0.1
npm version minor  # 1.0.1 -> 1.1.0
npm version major  # 1.1.0 -> 2.0.0

# Tag release
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

## 🔧 Configuration Management

### Environment Configuration
```javascript
// config/environment.js
export const config = {
  development: {
    logLevel: 'debug',
    maxMemoryUsage: '2GB',
    timeoutMs: 30000,
    pythonPath: 'python'
  },
  production: {
    logLevel: 'info',
    maxMemoryUsage: '8GB',
    timeoutMs: 300000,
    pythonPath: '/usr/bin/python3'
  },
  test: {
    logLevel: 'error',
    maxMemoryUsage: '1GB',
    timeoutMs: 10000,
    pythonPath: 'python'
  }
};

export function getConfig() {
  const env = process.env.NODE_ENV || 'development';
  return config[env];
}
```

### Python Configuration
```python
# python/config/settings.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Settings:
    """Application settings"""
    log_level: str = "INFO"
    max_file_size_mb: int = 100
    default_output_dir: str = "output"
    temp_dir: str = "/tmp"
    matplotlib_backend: str = "Agg"

    @classmethod
    def from_env(cls) -> 'Settings':
        """Load settings from environment variables"""
        return cls(
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            max_file_size_mb=int(os.getenv('MAX_FILE_SIZE_MB', '100')),
            default_output_dir=os.getenv('OUTPUT_DIR', 'output'),
            temp_dir=os.getenv('TEMP_DIR', '/tmp'),
            matplotlib_backend=os.getenv('MPL_BACKEND', 'Agg')
        )

# Global settings instance
settings = Settings.from_env()
```

## 🚀 Deployment Strategies

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./
COPY python/requirements*.txt ./python/

# Install dependencies
RUN npm install && \
    pip install -r python/requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 3000

# Start command
CMD ["npm", "start"]
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-mcp-analysis:
    build: ./ml-mcp-analysis
    ports:
      - "3001:3000"
    environment:
      - NODE_ENV=production
      - LOG_LEVEL=info
    volumes:
      - ./data:/app/data
      - ./output:/app/output

  ml-mcp-ml:
    build: ./ml-mcp-ml
    ports:
      - "3002:3000"
    environment:
      - NODE_ENV=production
    volumes:
      - ./data:/app/data
      - ./models:/app/models

  ml-mcp-visualization:
    build: ./ml-mcp-visualization
    ports:
      - "3003:3000"
    environment:
      - NODE_ENV=production
    volumes:
      - ./data:/app/data
      - ./visualizations:/app/visualizations
```

### CI/CD Pipeline
```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        node-version: [18.x, 20.x]
        python-version: [3.8, 3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3

    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: ${{ matrix.node-version }}

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        npm install
        pip install -r python/requirements.txt
        pip install -r python/requirements-dev.txt

    - name: Run linting
      run: |
        npm run lint
        python -m black --check python/
        python -m isort --check python/

    - name: Run tests
      run: |
        npm test
        python -m pytest python/tests/ --cov=python/analyzers

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v3

    - name: Build and push Docker images
      run: |
        docker build -t ml-mcp-system:latest .
        # Add your deployment steps here
```

## 📊 Monitoring and Observability

### Logging Configuration
```javascript
// utils/logger.js
import winston from 'winston';

const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'ml-mcp-system' },
  transports: [
    new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
    new winston.transports.File({ filename: 'logs/combined.log' }),
    new winston.transports.Console({
      format: winston.format.simple()
    })
  ]
});

export default logger;
```

### Performance Monitoring
```python
# python/utils/monitoring.py
import time
import psutil
import logging
from functools import wraps
from typing import Callable, Any

def monitor_performance(func: Callable) -> Callable:
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Record start metrics
        start_time = time.time()
        start_memory = psutil.virtual_memory().used

        try:
            result = func(*args, **kwargs)

            # Record end metrics
            end_time = time.time()
            end_memory = psutil.virtual_memory().used

            # Log performance metrics
            logging.info(f"Function {func.__name__} completed", extra={
                'execution_time': end_time - start_time,
                'memory_used_mb': (end_memory - start_memory) / 1024 / 1024,
                'success': True
            })

            return result

        except Exception as e:
            end_time = time.time()
            logging.error(f"Function {func.__name__} failed", extra={
                'execution_time': end_time - start_time,
                'error': str(e),
                'success': False
            })
            raise

    return wrapper

# Usage
@monitor_performance
def analyze_data(df, params):
    # Your analysis code
    pass
```

## 🔒 Security Best Practices

### Input Validation
```python
# python/utils/validation.py
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

class ValidationError(Exception):
    """Custom validation error"""
    pass

def validate_file_path(file_path: str, allowed_extensions: List[str] = None) -> Path:
    """Validate file path for security"""
    if not file_path:
        raise ValidationError("File path cannot be empty")

    path = Path(file_path)

    # Check for path traversal attacks
    if '..' in str(path) or str(path).startswith('/'):
        raise ValidationError("Invalid file path: path traversal detected")

    # Check file extension
    if allowed_extensions and path.suffix.lower() not in allowed_extensions:
        raise ValidationError(f"Invalid file extension. Allowed: {allowed_extensions}")

    # Check if file exists and is readable
    if not path.exists():
        raise ValidationError(f"File does not exist: {file_path}")

    if not os.access(path, os.R_OK):
        raise ValidationError(f"File is not readable: {file_path}")

    return path

def validate_parameters(params: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Validate parameters against schema"""
    validated = {}

    for key, config in schema.items():
        value = params.get(key)

        # Check required parameters
        if config.get('required', False) and value is None:
            raise ValidationError(f"Required parameter missing: {key}")

        # Type validation
        expected_type = config.get('type')
        if value is not None and expected_type:
            if not isinstance(value, expected_type):
                raise ValidationError(f"Parameter {key} must be {expected_type.__name__}")

        # Range validation for numbers
        if isinstance(value, (int, float)):
            min_val = config.get('min')
            max_val = config.get('max')
            if min_val is not None and value < min_val:
                raise ValidationError(f"Parameter {key} must be >= {min_val}")
            if max_val is not None and value > max_val:
                raise ValidationError(f"Parameter {key} must be <= {max_val}")

        validated[key] = value

    return validated
```

### Secure File Handling
```python
# python/utils/secure_file_handler.py
import tempfile
import shutil
import os
from pathlib import Path
from contextlib import contextmanager

@contextmanager
def secure_temp_directory():
    """Create a secure temporary directory"""
    temp_dir = tempfile.mkdtemp(prefix='ml_mcp_')
    try:
        # Set secure permissions
        os.chmod(temp_dir, 0o700)
        yield Path(temp_dir)
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent security issues"""
    # Remove dangerous characters
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    sanitized = ''.join(c for c in filename if c in safe_chars)

    # Prevent empty names
    if not sanitized:
        sanitized = "unnamed_file"

    # Limit length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]

    return sanitized
```

## 🔄 Contribution Guidelines

### Code Standards
1. **Python**: Follow PEP 8, use type hints, write docstrings
2. **JavaScript**: Use ES6+, consistent naming, JSDoc comments
3. **Testing**: Minimum 80% code coverage
4. **Documentation**: Update docs with any API changes

### Pull Request Process
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and add tests
4. Update documentation
5. Run full test suite
6. Submit pull request

### Code Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass and coverage is adequate
- [ ] Documentation is updated
- [ ] No security vulnerabilities introduced
- [ ] Performance impact considered
- [ ] Backward compatibility maintained

---

## 📚 Additional Resources

### Useful Links
- [Model Context Protocol Specification](https://spec.modelcontextprotocol.io/)
- [Python Type Hints Guide](https://docs.python.org/3/library/typing.html)
- [Jest Testing Framework](https://jestjs.io/)
- [pytest Documentation](https://docs.pytest.org/)

### Community
- GitHub Discussions for questions
- Issue tracker for bugs and feature requests
- Wiki for community-contributed examples

---

*This guide covers the essential aspects of developing with the ML MCP System. For specific implementation questions, please refer to the source code or create an issue in the repository.*