// utils/config-loader.js
import { Logger } from './logger.js';
import fs from 'fs/promises';
import path from 'path';

export class ConfigLoader {
  constructor() {
    this.logger = new Logger();
    this.configCache = new Map();
    this.watchedFiles = new Map();
    this.configDir = './config';
    this.defaultConfigs = this.initializeDefaultConfigs();
  }

  async initialize() {
    try {
      // 설정 디렉토리 생성
      await this.ensureConfigDirectory();
      
      // 기본 설정 파일들 생성 (없는 경우)
      await this.createMissingConfigFiles();
      
      // 모든 설정 파일 로드
      await this.loadAllConfigs();
      
      this.logger.info('ConfigLoader 초기화 완료');
    } catch (error) {
      this.logger.error('ConfigLoader 초기화 실패:', error);
      throw error;
    }
  }

  async ensureConfigDirectory() {
    try {
      await fs.mkdir(this.configDir, { recursive: true });
      this.logger.debug(`설정 디렉토리 확인: ${this.configDir}`);
    } catch (error) {
      this.logger.error('설정 디렉토리 생성 실패:', error);
      throw error;
    }
  }

  initializeDefaultConfigs() {
    return {
      'analysis-methods.json': {
        basic: {
          descriptive_stats: {
            enabled: true,
            include_percentiles: true,
            percentiles: [25, 50, 75, 90, 95, 99],
            include_skewness: true,
            include_kurtosis: true
          },
          correlation: {
            enabled: true,
            methods: ['pearson', 'spearman', 'kendall'],
            default_method: 'pearson',
            threshold: 0.5
          },
          missing_values: {
            enabled: true,
            show_patterns: true,
            threshold_percent: 5
          }
        },
        advanced: {
          outlier_detection: {
            enabled: true,
            methods: ['iqr', 'zscore', 'isolation_forest'],
            default_method: 'iqr',
            iqr_factor: 1.5,
            zscore_threshold: 3
          },
          feature_engineering: {
            enabled: true,
            auto_encoding: true,
            scaling_methods: ['standard', 'minmax', 'robust'],
            default_scaling: 'standard'
          },
          dimensionality_reduction: {
            enabled: true,
            methods: ['pca', 'tsne', 'umap'],
            default_method: 'pca',
            n_components: 2
          }
        }
      },

      'pipeline-templates.json': {
        data_exploration: {
          name: 'data_exploration',
          description: '기본 데이터 탐색 파이프라인',
          steps: [
            { type: 'data_loading', method: 'load_dataset' },
            { type: 'basic', method: 'descriptive_stats' },
            { type: 'basic', method: 'missing_values_analysis' },
            { type: 'visualization', method: 'distribution_plots' },
            { type: 'basic', method: 'correlation' },
            { type: 'visualization', method: 'correlation_heatmap' }
          ]
        },
        ml_pipeline: {
          name: 'ml_pipeline',
          description: '머신러닝 파이프라인',
          steps: [
            { type: 'data_loading', method: 'load_dataset' },
            { type: 'preprocessing', method: 'clean_data' },
            { type: 'advanced', method: 'feature_engineering' },
            { type: 'ml_traditional', method: 'train_model' },
            { type: 'ml_traditional', method: 'evaluate_model' },
            { type: 'visualization', method: 'model_performance' }
          ]
        }
      },

      'python-config.json': {
        environment: {
          python_path: 'python3',
          virtual_env: './python-env',
          use_virtual_env: true,
          timeout: 300000
        },
        packages: {
          required: [
            'pandas>=1.3.0',
            'numpy>=1.21.0',
            'scikit-learn>=1.0.0',
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0'
          ],
          optional: [
            'plotly>=5.0.0',
            'tensorflow>=2.8.0',
            'torch>=1.11.0'
          ]
        },
        execution: {
          max_memory_mb: 4000,
          max_execution_time: 300,
          temp_dir: './temp',
          cleanup_temp_files: true
        }
      },

      'models-config.json': {
        router: {
          name: 'llama3.2:3b',
          description: '빠른 라우팅을 위한 경량 모델',
          endpoint: 'http://localhost:11434',
          temperature: 0.7,
          max_tokens: 500,
          context_length: 2048,
          memory_limit_mb: 6000,
          auto_unload: false,
          keep_alive: 300000
        },
        processor: {
          name: 'qwen2.5:14b',
          description: '복잡한 작업을 위한 고성능 모델',
          endpoint: 'http://localhost:11434',
          temperature: 0.3,
          max_tokens: 2000,
          context_length: 8192,
          memory_limit_mb: 28000,
          auto_unload: true,
          keep_alive: 600000
        }
      },

      'visualization-config.json': {
        default_settings: {
          figure_size: [10, 8],
          dpi: 100,
          style: 'seaborn-v0_8',
          color_palette: 'Set1',
          font_size: 12,
          save_format: 'png'
        },
        chart_types: {
          scatter: { marker_size: 50, alpha: 0.7 },
          line: { line_width: 2, marker_size: 6 },
          bar: { width: 0.8, alpha: 0.8 },
          histogram: { bins: 30, alpha: 0.7 },
          heatmap: { annot: true, cmap: 'coolwarm' }
        },
        export: {
          formats: ['png', 'jpg', 'svg', 'pdf'],
          quality: 95,
          transparent: false,
          bbox_inches: 'tight'
        }
      },

      'routing-rules.json': {
        intent_patterns: {
          data_analysis: [
            'analyze', 'analysis', 'explore', 'examine', 'investigate',
            'statistics', 'stats', 'summary', 'describe'
          ],
          visualization: [
            'plot', 'chart', 'graph', 'visualize', 'draw', 'show',
            'histogram', 'scatter', 'heatmap'
          ],
          machine_learning: [
            'train', 'model', 'predict', 'classification', 'regression',
            'clustering', 'ml', 'machine learning'
          ],
          data_processing: [
            'clean', 'preprocess', 'transform', 'encode', 'scale',
            'normalize', 'feature engineering'
          ]
        },
        complexity_thresholds: {
          simple: { token_limit: 500, use_router: true },
          medium: { token_limit: 1000, use_router: false },
          complex: { token_limit: 2000, use_router: false }
        },
        mode_switching: {
          auto_switch: true,
          confidence_threshold: 0.7,
          switch_delay_ms: 1000
        }
      }
    };
  }

  async createMissingConfigFiles() {
    for (const [filename, defaultConfig] of Object.entries(this.defaultConfigs)) {
      const filePath = path.join(this.configDir, filename);
      
      try {
        await fs.access(filePath);
        this.logger.debug(`설정 파일 존재: ${filename}`);
      } catch (error) {
        // 파일이 없으면 생성
        await this.createConfigFile(filename, defaultConfig);
      }
    }
  }

  async createConfigFile(filename, config) {
    try {
      const filePath = path.join(this.configDir, filename);
      const configJson = JSON.stringify(config, null, 2);
      
      await fs.writeFile(filePath, configJson, 'utf-8');
      this.logger.info(`기본 설정 파일 생성: ${filename}`);
    } catch (error) {
      this.logger.error(`설정 파일 생성 실패 [${filename}]:`, error);
      throw error;
    }
  }

  async loadAllConfigs() {
    const configFiles = Object.keys(this.defaultConfigs);
    
    for (const filename of configFiles) {
      try {
        await this.loadConfig(filename);
      } catch (error) {
        this.logger.warn(`설정 파일 로드 실패 [${filename}]:`, error);
        // 기본 설정 사용
        const configName = filename.replace('.json', '');
        this.configCache.set(configName, this.defaultConfigs[filename]);
      }
    }
  }

  async loadConfig(filename) {
    try {
      const filePath = path.join(this.configDir, filename);
      const configData = await fs.readFile(filePath, 'utf-8');
      const config = JSON.parse(configData);
      
      const configName = filename.replace('.json', '');
      this.configCache.set(configName, config);
      
      this.logger.debug(`설정 로드 완료: ${configName}`);
      return config;
    } catch (error) {
      this.logger.error(`설정 로드 실패 [${filename}]:`, error);
      throw error;
    }
  }

  getConfig(configName) {
    if (this.configCache.has(configName)) {
      return this.configCache.get(configName);
    }
    
    this.logger.warn(`설정을 찾을 수 없음: ${configName}`);
    return null;
  }

  getConfigValue(configName, keyPath, defaultValue = null) {
    const config = this.getConfig(configName);
    if (!config) return defaultValue;
    
    const keys = keyPath.split('.');
    let current = config;
    
    for (const key of keys) {
      if (current && typeof current === 'object' && key in current) {
        current = current[key];
      } else {
        return defaultValue;
      }
    }
    
    return current;
  }

  async updateConfig(configName, updates) {
    try {
      const currentConfig = this.getConfig(configName) || {};
      const updatedConfig = this.deepMerge(currentConfig, updates);
      
      // 캐시 업데이트
      this.configCache.set(configName, updatedConfig);
      
      // 파일 업데이트
      const filename = `${configName}.json`;
      const filePath = path.join(this.configDir, filename);
      const configJson = JSON.stringify(updatedConfig, null, 2);
      
      await fs.writeFile(filePath, configJson, 'utf-8');
      
      this.logger.info(`설정 업데이트 완료: ${configName}`);
      return updatedConfig;
    } catch (error) {
      this.logger.error(`설정 업데이트 실패 [${configName}]:`, error);
      throw error;
    }
  }

  deepMerge(target, source) {
    const result = { ...target };
    
    for (const key in source) {
      if (source[key] !== null && typeof source[key] === 'object' && !Array.isArray(source[key])) {
        result[key] = this.deepMerge(result[key] || {}, source[key]);
      } else {
        result[key] = source[key];
      }
    }
    
    return result;
  }

  async reloadConfig(configName) {
    try {
      const filename = `${configName}.json`;
      await this.loadConfig(filename);
      this.logger.info(`설정 재로드 완료: ${configName}`);
    } catch (error) {
      this.logger.error(`설정 재로드 실패 [${configName}]:`, error);
      throw error;
    }
  }

  async reloadAllConfigs() {
    try {
      this.configCache.clear();
      await this.loadAllConfigs();
      this.logger.info('모든 설정 재로드 완료');
    } catch (error) {
      this.logger.error('설정 재로드 실패:', error);
      throw error;
    }
  }

  getAvailableConfigs() {
    return Array.from(this.configCache.keys());
  }

  async validateConfig(configName) {
    const config = this.getConfig(configName);
    if (!config) {
      return { valid: false, errors: [`설정을 찾을 수 없음: ${configName}`] };
    }

    const validation = { valid: true, errors: [], warnings: [] };

    // 설정별 검증 로직
    switch (configName) {
      case 'models-config':
        this.validateModelsConfig(config, validation);
        break;
      case 'python-config':
        this.validatePythonConfig(config, validation);
        break;
      case 'analysis-methods':
        this.validateAnalysisConfig(config, validation);
        break;
    }

    return validation;
  }

  validateModelsConfig(config, validation) {
    ['router', 'processor'].forEach(modelType => {
      if (!config[modelType]) {
        validation.errors.push(`${modelType} 모델 설정 누락`);
        validation.valid = false;
      } else {
        const model = config[modelType];
        if (!model.name) {
          validation.errors.push(`${modelType} 모델 이름 누락`);
          validation.valid = false;
        }
        if (!model.endpoint) {
          validation.errors.push(`${modelType} 모델 엔드포인트 누락`);
          validation.valid = false;
        }
      }
    });
  }

  validatePythonConfig(config, validation) {
    if (!config.environment) {
      validation.errors.push('Python 환경 설정 누락');
      validation.valid = false;
    }
    
    if (!config.packages || !config.packages.required) {
      validation.errors.push('필수 Python 패키지 목록 누락');
      validation.valid = false;
    }
  }

  validateAnalysisConfig(config, validation) {
    if (!config.basic) {
      validation.warnings.push('기본 분석 방법 설정 누락');
    }
    
    if (!config.advanced) {
      validation.warnings.push('고급 분석 방법 설정 누락');
    }
  }

  async exportConfig(configName, format = 'json') {
    const config = this.getConfig(configName);
    if (!config) {
      throw new Error(`설정을 찾을 수 없음: ${configName}`);
    }

    switch (format) {
      case 'json':
        return JSON.stringify(config, null, 2);
      case 'yaml':
        // 간단한 YAML 변환 (실제로는 yaml 라이브러리 사용 권장)
        return this.convertToYAML(config);
      default:
        throw new Error(`지원하지 않는 형식: ${format}`);
    }
  }

  convertToYAML(obj, indent = 0) {
    let yaml = '';
    const spaces = '  '.repeat(indent);
    
    for (const [key, value] of Object.entries(obj)) {
      if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
        yaml += `${spaces}${key}:\n`;
        yaml += this.convertToYAML(value, indent + 1);
      } else if (Array.isArray(value)) {
        yaml += `${spaces}${key}:\n`;
        value.forEach(item => {
          yaml += `${spaces}  - ${item}\n`;
        });
      } else {
        yaml += `${spaces}${key}: ${value}\n`;
      }
    }
    
    return yaml;
  }
}