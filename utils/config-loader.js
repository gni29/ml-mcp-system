// utils/config-loader.js
import { Logger } from './logger.js';
import fs from 'fs/promises';
import path from 'path';

export class ConfigLoader {
  constructor() {
    this.logger = new Logger();
    this.configDir = './config';
    this.cache = new Map();
    this.watchedFiles = new Map();
    this.enableCache = true;
    this.enableWatch = false;
    this.defaultConfigs = new Map();
    
    this.setupDefaultConfigs();
  }

  setupDefaultConfigs() {
    // 기본 라우팅 규칙
    this.defaultConfigs.set('routing-rules.json', {
      simple_queries: {
        keywords: ['안녕', '도움말', '상태', '모드', '종료', 'help', 'status'],
        maxComplexity: 0.3,
        model: 'router',
        tools: ['system']
      },
      data_operations: {
        keywords: ['로드', '불러오기', '데이터', '파일', 'load', 'read'],
        complexity: [0.2, 0.6],
        model: 'router',
        tools: ['data-loader', 'data-validator']
      },
      analysis: {
        keywords: ['분석', '통계', '상관관계', 'analysis', 'stats'],
        complexity: [0.3, 0.7],
        model: 'router',
        tools: ['basic-analyzer']
      },
      ml_operations: {
        keywords: ['모델', '훈련', '예측', '머신러닝', 'training', 'prediction'],
        minComplexity: 0.7,
        model: 'processor',
        tools: ['trainer', 'predictor']
      }
    });

    // 기본 Python 설정
    this.defaultConfigs.set('python-config.json', {
      python_executable: 'python3',
      virtual_env: './python-env',
      timeout: 30000,
      max_memory_mb: 2048,
      working_directory: './python',
      environment_variables: {
        PYTHONPATH: './python',
        MATPLOTLIB_BACKEND: 'Agg'
      },
      required_packages: [
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0'
      ]
    });

    // 기본 모델 설정
    this.defaultConfigs.set('models-config.json', {
      router: {
        model: 'llama3.2:3b',
        endpoint: 'http://localhost:11434',
        temperature: 0.1,
        max_tokens: 1024,
        role: 'routing'
      },
      processor: {
        model: 'qwen2.5:14b',
        endpoint: 'http://localhost:11434',
        temperature: 0.3,
        max_tokens: 2048,
        role: 'processing'
      }
    });

    // 기본 딥러닝 설정
    this.defaultConfigs.set('deep-learning-config.json', {
      frameworks: {
        tensorflow: {
          enabled: true,
          gpu_support: false,
          memory_growth: true
        },
        pytorch: {
          enabled: false,
          gpu_support: false
        }
      },
      default_parameters: {
        batch_size: 32,
        epochs: 100,
        learning_rate: 0.001,
        validation_split: 0.2
      },
      model_architectures: {
        simple_nn: {
          layers: [128, 64, 32],
          activation: 'relu',
          dropout: 0.3
        },
        cnn_basic: {
          conv_layers: [32, 64, 128],
          kernel_size: 3,
          pool_size: 2,
          dropout: 0.5
        }
      }
    });

    // 기본 시각화 설정
    this.defaultConfigs.set('visualization-config.json', {
      default_style: 'seaborn',
      color_palettes: {
        categorical: 'Set1',
        sequential: 'viridis',
        diverging: 'RdBu'
      },
      figure_size: [10, 6],
      dpi: 100,
      file_format: 'png',
      interactive: false,
      chart_types: {
        line: {
          default_params: { marker: 'o', linewidth: 2 }
        },
        bar: {
          default_params: { alpha: 0.8 }
        },
        scatter: {
          default_params: { alpha: 0.6, s: 50 }
        },
        heatmap: {
          default_params: { cmap: 'viridis', annot: true }
        }
      }
    });
  }

  async loadConfig(filename, options = {}) {
    const {
      useCache = this.enableCache,
      fallbackToDefault = true,
      validate = true,
      watch = this.enableWatch
    } = options;

    try {
      // 캐시 확인
      if (useCache && this.cache.has(filename)) {
        this.logger.debug(`캐시에서 설정 로드: ${filename}`);
        return this.cache.get(filename);
      }

      // 파일 경로 구성
      const filePath = path.join(this.configDir, filename);
      
      // 파일 존재 여부 확인
      try {
        await fs.access(filePath);
      } catch (error) {
        if (fallbackToDefault && this.defaultConfigs.has(filename)) {
          this.logger.warn(`설정 파일 없음, 기본값 사용: ${filename}`);
          return this.getDefaultConfig(filename);
        }
        throw new Error(`설정 파일을 찾을 수 없습니다: ${filename}`);
      }

      // 파일 읽기
      const configData = await fs.readFile(filePath, 'utf-8');
      let config;

      // JSON 파싱
      try {
        config = JSON.parse(configData);
      } catch (parseError) {
        this.logger.error(`설정 파일 파싱 실패: ${filename}`, parseError);
        if (fallbackToDefault && this.defaultConfigs.has(filename)) {
          this.logger.warn('기본 설정으로 대체합니다.');
          return this.getDefaultConfig(filename);
        }
        throw new Error(`설정 파일 파싱 실패: ${filename}`);
      }

      // 설정 검증
      if (validate) {
        const validationResult = this.validateConfig(filename, config);
        if (!validationResult.valid) {
          this.logger.warn(`설정 검증 실패: ${filename}`, validationResult.errors);
          // 검증 실패 시에도 설정을 사용하되 경고만 출력
        }
      }

      // 기본값과 병합
      if (fallbackToDefault && this.defaultConfigs.has(filename)) {
        config = this.mergeWithDefaults(filename, config);
      }

      // 캐시에 저장
      if (useCache) {
        this.cache.set(filename, config);
      }

      // 파일 감시 설정
      if (watch && !this.watchedFiles.has(filename)) {
        await this.watchConfigFile(filename, filePath);
      }

      this.logger.info(`설정 로드 완료: ${filename}`);
      return config;

    } catch (error) {
      this.logger.error(`설정 로드 실패: ${filename}`, error);
      
      // 최후의 수단으로 기본 설정 반환
      if (fallbackToDefault && this.defaultConfigs.has(filename)) {
        this.logger.warn('기본 설정으로 대체합니다.');
        return this.getDefaultConfig(filename);
      }
      
      throw error;
    }
  }

  getDefaultConfig(filename) {
    if (this.defaultConfigs.has(filename)) {
      return JSON.parse(JSON.stringify(this.defaultConfigs.get(filename)));
    }
    throw new Error(`기본 설정이 존재하지 않습니다: ${filename}`);
  }

  validateConfig(filename, config) {
    const result = {
      valid: true,
      errors: [],
      warnings: []
    };

    try {
      switch (filename) {
        case 'routing-rules.json':
          result = this.validateRoutingRules(config);
          break;
        case 'analysis-methods.json':
          result = this.validateAnalysisMethods(config);
          break;
        case 'pipeline-templates.json':
          result = this.validatePipelineTemplates(config);
          break;
        case 'python-config.json':
          result = this.validatePythonConfig(config);
          break;
        case 'models-config.json':
          result = this.validateModelsConfig(config);
          break;
        default:
          // 일반적인 JSON 구조 검증
          result = this.validateGenericConfig(config);
      }
    } catch (error) {
      result.valid = false;
      result.errors.push(`검증 중 오류 발생: ${error.message}`);
    }

    return result;
  }

  validateRoutingRules(config) {
    const result = { valid: true, errors: [], warnings: [] };

    if (!config || typeof config !== 'object') {
      result.valid = false;
      result.errors.push('라우팅 규칙은 객체여야 합니다.');
      return result;
    }

    for (const [ruleName, rule] of Object.entries(config)) {
      if (!rule.keywords || !Array.isArray(rule.keywords)) {
        result.warnings.push(`${ruleName}: keywords가 배열이 아닙니다.`);
      }

      if (rule.complexity && Array.isArray(rule.complexity)) {
        if (rule.complexity.length !== 2 || 
            rule.complexity[0] >= rule.complexity[1]) {
          result.errors.push(`${ruleName}: complexity 범위가 잘못되었습니다.`);
          result.valid = false;
        }
      }

      if (!rule.model) {
        result.warnings.push(`${ruleName}: model이 지정되지 않았습니다.`);
      }
    }

    return result;
  }

  validateAnalysisMethods(config) {
    const result = { valid: true, errors: [], warnings: [] };

    const requiredCategories = ['basic', 'advanced', 'timeseries', 'ml_traditional'];
    
    for (const category of requiredCategories) {
      if (!config[category]) {
        result.warnings.push(`${category} 카테고리가 없습니다.`);
      } else {
        for (const [methodName, method] of Object.entries(config[category])) {
          if (!method.python_script) {
            result.errors.push(`${category}.${methodName}: python_script가 필요합니다.`);
            result.valid = false;
          }

          if (method.complexity && (method.complexity < 0 || method.complexity > 1)) {
            result.errors.push(`${category}.${methodName}: complexity는 0-1 사이여야 합니다.`);
            result.valid = false;
          }
        }
      }
    }

    return result;
  }

  validatePipelineTemplates(config) {
    const result = { valid: true, errors: [], warnings: [] };

    const requiredSections = ['common_workflows', 'ml_workflows'];
    
    for (const section of requiredSections) {
      if (!config[section]) {
        result.warnings.push(`${section} 섹션이 없습니다.`);
        continue;
      }

      for (const [workflowName, workflow] of Object.entries(config[section])) {
        if (!workflow.steps || !Array.isArray(workflow.steps)) {
          result.errors.push(`${workflowName}: steps가 배열이 아닙니다.`);
          result.valid = false;
          continue;
        }

        for (const [stepIndex, step] of workflow.steps.entries()) {
          if (!step.type || !step.method) {
            result.errors.push(`${workflowName} 단계 ${stepIndex + 1}: type과 method가 필요합니다.`);
            result.valid = false;
          }

          if (!step.outputs || !Array.isArray(step.outputs)) {
            result.warnings.push(`${workflowName} 단계 ${stepIndex + 1}: outputs가 정의되지 않았습니다.`);
          }
        }
      }
    }

    return result;
  }

  validatePythonConfig(config) {
    const result = { valid: true, errors: [], warnings: [] };

    if (!config.python_executable) {
      result.errors.push('python_executable이 필요합니다.');
      result.valid = false;
    }

    if (config.timeout && (typeof config.timeout !== 'number' || config.timeout <= 0)) {
      result.errors.push('timeout은 양수여야 합니다.');
      result.valid = false;
    }

    if (config.max_memory_mb && (typeof config.max_memory_mb !== 'number' || config.max_memory_mb <= 0)) {
      result.errors.push('max_memory_mb는 양수여야 합니다.');
      result.valid = false;
    }

    if (!config.required_packages || !Array.isArray(config.required_packages)) {
      result.warnings.push('required_packages가 배열이 아닙니다.');
    }

    return result;
  }

  validateModelsConfig(config) {
    const result = { valid: true, errors: [], warnings: [] };

    const requiredModels = ['router', 'processor'];
    
    for (const modelName of requiredModels) {
      if (!config[modelName]) {
        result.errors.push(`${modelName} 모델 설정이 없습니다.`);
        result.valid = false;
        continue;
      }

      const model = config[modelName];
      
      if (!model.model) {
        result.errors.push(`${modelName}: model 이름이 필요합니다.`);
        result.valid = false;
      }

      if (!model.endpoint) {
        result.errors.push(`${modelName}: endpoint가 필요합니다.`);
        result.valid = false;
      }

      if (model.temperature && (model.temperature < 0 || model.temperature > 2)) {
        result.warnings.push(`${modelName}: temperature는 0-2 사이를 권장합니다.`);
      }
    }

    return result;
  }

  validateGenericConfig(config) {
    const result = { valid: true, errors: [], warnings: [] };

    if (!config || typeof config !== 'object') {
      result.valid = false;
      result.errors.push('설정은 객체여야 합니다.');
    }

    return result;
  }

  mergeWithDefaults(filename, config) {
    const defaultConfig = this.getDefaultConfig(filename);
    return this.deepMerge(defaultConfig, config);
  }

  deepMerge(target, source) {
    const result = { ...target };

    for (const key in source) {
      if (source.hasOwnProperty(key)) {
        if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
          result[key] = this.deepMerge(target[key] || {}, source[key]);
        } else {
          result[key] = source[key];
        }
      }
    }

    return result;
  }

  async watchConfigFile(filename, filePath) {
    try {
      const watcher = fs.watch(filePath, { encoding: 'utf8' }, async (eventType) => {
        if (eventType === 'change') {
          this.logger.info(`설정 파일 변경 감지: ${filename}`);
          
          // 캐시 무효화
          this.cache.delete(filename);
          
          // 설정 다시 로드
          try {
            await this.loadConfig(filename, { useCache: false, watch: false });
            this.logger.info(`설정 파일 다시 로드 완료: ${filename}`);
          } catch (error) {
            this.logger.error(`설정 파일 다시 로드 실패: ${filename}`, error);
          }
        }
      });

      this.watchedFiles.set(filename, watcher);
      this.logger.debug(`파일 감시 시작: ${filename}`);

    } catch (error) {
      this.logger.warn(`파일 감시 설정 실패: ${filename}`, error);
    }
  }

  async saveConfig(filename, config, options = {}) {
    const {
      backup = true,
      validate = true,
      updateCache = true
    } = options;

    try {
      const filePath = path.join(this.configDir, filename);
      
      // 설정 검증
      if (validate) {
        const validationResult = this.validateConfig(filename, config);
        if (!validationResult.valid) {
          throw new Error(`설정 검증 실패: ${validationResult.errors.join(', ')}`);
        }
      }

      // 백업 생성
      if (backup) {
        await this.createBackup(filename);
      }

      // 디렉토리 생성
      await fs.mkdir(this.configDir, { recursive: true });

      // 파일 쓰기
      const configJson = JSON.stringify(config, null, 2);
      await fs.writeFile(filePath, configJson, 'utf-8');

      // 캐시 업데이트
      if (updateCache) {
        this.cache.set(filename, config);
      }

      this.logger.info(`설정 저장 완료: ${filename}`);
      return true;

    } catch (error) {
      this.logger.error(`설정 저장 실패: ${filename}`, error);
      throw error;
    }
  }

  async createBackup(filename) {
    try {
      const filePath = path.join(this.configDir, filename);
      const backupDir = path.join(this.configDir, 'backups');
      
      // 백업 디렉토리 생성
      await fs.mkdir(backupDir, { recursive: true });

      // 파일 존재 여부 확인
      try {
        await fs.access(filePath);
      } catch {
        // 파일이 없으면 백업 불필요
        return;
      }

      // 백업 파일명 생성
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const backupFilename = `${filename}.${timestamp}.bak`;
      const backupPath = path.join(backupDir, backupFilename);

      // 파일 복사
      await fs.copyFile(filePath, backupPath);
      
      this.logger.debug(`백업 생성 완료: ${backupFilename}`);

      // 오래된 백업 정리 (최근 10개만 유지)
      await this.cleanupOldBackups(filename, backupDir);

    } catch (error) {
      this.logger.warn(`백업 생성 실패: ${filename}`, error);
    }
  }

  async cleanupOldBackups(filename, backupDir) {
    try {
      const files = await fs.readdir(backupDir);
      const backupFiles = files
        .filter(file => file.startsWith(filename) && file.endsWith('.bak'))
        .map(file => ({
          name: file,
          path: path.join(backupDir, file)
        }))
        .sort((a, b) => b.name.localeCompare(a.name)); // 최신 순 정렬

      // 10개 초과 시 오래된 것들 삭제
      if (backupFiles.length > 10) {
        const filesToDelete = backupFiles.slice(10);
        for (const file of filesToDelete) {
          await fs.unlink(file.path);
          this.logger.debug(`오래된 백업 삭제: ${file.name}`);
        }
      }

    } catch (error) {
      this.logger.warn('백업 정리 실패:', error);
    }
  }

  async listConfigs() {
    try {
      const files = await fs.readdir(this.configDir);
      const configFiles = files.filter(file => file.endsWith('.json'));
      
      const configs = [];
      for (const file of configFiles) {
        try {
          const filePath = path.join(this.configDir, file);
          const stats = await fs.stat(filePath);
          
          configs.push({
            filename: file,
            size: stats.size,
            modified: stats.mtime,
            cached: this.cache.has(file),
            watched: this.watchedFiles.has(file)
          });
        } catch (error) {
          this.logger.warn(`파일 정보 조회 실패: ${file}`, error);
        }
      }
      
      return configs;

    } catch (error) {
      this.logger.error('설정 목록 조회 실패:', error);
      return [];
    }
  }

  clearCache(filename = null) {
    if (filename) {
      this.cache.delete(filename);
      this.logger.debug(`캐시 삭제: ${filename}`);
    } else {
      this.cache.clear();
      this.logger.debug('전체 캐시 삭제');
    }
  }

  stopWatching(filename = null) {
    if (filename) {
      const watcher = this.watchedFiles.get(filename);
      if (watcher) {
        watcher.close();
        this.watchedFiles.delete(filename);
        this.logger.debug(`파일 감시 중지: ${filename}`);
      }
    } else {
      for (const [file, watcher] of this.watchedFiles) {
        watcher.close();
        this.logger.debug(`파일 감시 중지: ${file}`);
      }
      this.watchedFiles.clear();
    }
  }

  enableAutoWatch() {
    this.enableWatch = true;
    this.logger.info('자동 파일 감시 활성화');
  }

  disableAutoWatch() {
    this.enableWatch = false;
    this.stopWatching();
    this.logger.info('자동 파일 감시 비활성화');
  }

  getCacheStats() {
    return {
      cached_configs: this.cache.size,
      watched_files: this.watchedFiles.size,
      cache_enabled: this.enableCache,
      watch_enabled: this.enableWatch,
      cached_files: Array.from(this.cache.keys()),
      watched_files_list: Array.from(this.watchedFiles.keys())
    };
  }

  async initializeConfigDirectory() {
    try {
      // config 디렉토리 생성
      await fs.mkdir(this.configDir, { recursive: true });
      
      // 기본 설정 파일들 생성 (존재하지 않는 경우만)
      for (const [filename, defaultConfig] of this.defaultConfigs) {
        const filePath = path.join(this.configDir, filename);
        
        try {
          await fs.access(filePath);
          this.logger.debug(`설정 파일 이미 존재: ${filename}`);
        } catch {
          // 파일이 없으면 기본 설정으로 생성
          await this.saveConfig(filename, defaultConfig, { 
            backup: false, 
            validate: false 
          });
          this.logger.info(`기본 설정 파일 생성: ${filename}`);
        }
      }

      return true;

    } catch (error) {
      this.logger.error('설정 디렉토리 초기화 실패:', error);
      throw error;
    }
  }

  // 설정 템플릿 생성
  async createConfigTemplate(templateName, description = '') {
    const templates = {
      'custom-analysis.json': {
        name: '사용자 정의 분석',
        description: description || '사용자가 정의한 분석 방법',
        methods: {
          custom_method: {
            name: '사용자 정의 방법',
            description: '설명을 입력하세요',
            python_script: 'python/custom/custom_method.py',
            parameters: {
              param1: {
                type: 'string',
                default: 'default_value',
                description: '파라미터 설명'
              }
            },
            output_format: ['result'],
            complexity: 0.5,
            estimated_time_ms: 5000
          }
        }
      },
      'custom-workflow.json': {
        name: '사용자 정의 워크플로우',
        description: description || '사용자가 정의한 워크플로우',
        category: 'custom',
        complexity: 0.5,
        estimated_time: 300,
        steps: [
          {
            order: 1,
            type: 'data',
            method: 'load',
            description: '데이터 로드',
            params: {},
            outputs: ['data'],
            required: true
          }
        ]
      }
    };

    if (templates[templateName]) {
      await this.saveConfig(templateName, templates[templateName]);
      this.logger.info(`설정 템플릿 생성: ${templateName}`);
      return templates[templateName];
    } else {
      throw new Error(`알 수 없는 템플릿: ${templateName}`);
    }
  }
}