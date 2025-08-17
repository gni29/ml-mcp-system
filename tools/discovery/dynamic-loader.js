// tools/discovery/dynamic-loader.js
import { Logger } from '../../utils/logger.js';
import { PythonExecutor } from '../common/python-executor.js';
import { ModuleScanner } from './module-scanner.js';

export class DynamicLoader {
  constructor() {
    this.logger = new Logger();
    this.pythonExecutor = new PythonExecutor();
    this.moduleScanner = new ModuleScanner();
    this.executionHistory = [];
    this.maxHistorySize = 100;
    this.defaultTimeout = 300000; // 5분
  }

  async initialize() {
    try {
      this.logger.info('동적 로더 초기화 시작');
      
      // Python 실행기 초기화
      await this.pythonExecutor.initialize();
      
      // 모듈 스캐너 초기화  
      await this.moduleScanner.initialize();
      
      this.logger.info('동적 로더 초기화 완료');
    } catch (error) {
      this.logger.error('동적 로더 초기화 실패:', error);
      throw error;
    }
  }

  async findAndExecuteModule(query, data = null, options = {}) {
    const startTime = Date.now();
    
    try {
      this.logger.info(`동적 분석 요청: "${query}"`);
      
      // 1. 분석 의도 파싱
      const analysisIntent = this.parseAnalysisIntent(query);
      this.logger.debug('분석 의도:', analysisIntent);
      
      // 2. 최적 모듈 찾기
      const module = await this.findBestModule(query, analysisIntent, options);
      
      if (!module) {
        return await this.handleNoModuleFound(query, analysisIntent);
      }

      this.logger.info(`모듈 발견: ${module.displayName} (${module.id})`);

      // 3. 모듈 실행
      const result = await this.executeModule(module, data, options);

      // 4. 실행 기록 저장
      const executionTime = Date.now() - startTime;
      this.addToExecutionHistory(query, module, result, executionTime, true);

      return {
        success: true,
        module: this.formatModuleInfo(module),
        result: result,
        executionTime: executionTime,
        executedAt: new Date().toISOString()
      };

    } catch (error) {
      const executionTime = Date.now() - startTime;
      this.logger.error('동적 모듈 실행 실패:', error);
      
      // 실패 기록
      this.addToExecutionHistory(query, null, null, executionTime, false, error.message);
      
      throw error;
    }
  }

  parseAnalysisIntent(query) {
    const queryLower = query.toLowerCase();
    const intent = {
      category: null,
      subcategory: null,
      tags: [],
      analysisType: null,
      keywords: this.extractKeywords(queryLower),
      confidence: 0
    };

    // 분석 유형별 패턴 매칭
    const patterns = this.getAnalysisPatterns();
    
    for (const [type, pattern] of Object.entries(patterns)) {
      if (this.matchesPattern(queryLower, pattern)) {
        intent.category = pattern.category;
        intent.subcategory = pattern.subcategory;
        intent.tags.push(...pattern.tags);
        intent.analysisType = type;
        intent.confidence = pattern.confidence || 0.7;
        break;
      }
    }

    return intent;
  }

  getAnalysisPatterns() {
    return {
      // 기본 통계 분석
      'descriptive_stats': {
        keywords: ['통계', '기술통계', 'stats', 'descriptive', '요약', '평균', '표준편차'],
        category: 'analysis',
        subcategory: 'basic',
        tags: ['statistics', 'descriptive'],
        confidence: 0.9
      },
      
      'correlation': {
        keywords: ['상관관계', '상관', 'correlation', '연관성', '관계'],
        category: 'analysis',
        subcategory: 'basic',
        tags: ['correlation', 'relationship'],
        confidence: 0.95
      },
      
      'distribution': {
        keywords: ['분포', 'distribution', '히스토그램', '정규성', '분산'],
        category: 'analysis',
        subcategory: 'basic',
        tags: ['distribution', 'histogram'],
        confidence: 0.8
      },

      // 머신러닝
      'regression': {
        keywords: ['회귀', 'regression', '선형회귀', '예측', '추정'],
        category: 'ml',
        subcategory: 'supervised',
        tags: ['regression', 'prediction'],
        confidence: 0.9
      },
      
      'classification': {
        keywords: ['분류', 'classification', '범주', '클래스', '분류기'],
        category: 'ml',
        subcategory: 'supervised',
        tags: ['classification', 'categorization'],
        confidence: 0.9
      },
      
      'clustering': {
        keywords: ['클러스터', 'cluster', '군집', '그룹', '묶기'],
        category: 'ml',
        subcategory: 'unsupervised',
        tags: ['clustering', 'grouping'],
        confidence: 0.9
      },

      // 고급 분석
      'pca': {
        keywords: ['pca', '주성분', '차원축소', 'dimension', '압축'],
        category: 'analysis',
        subcategory: 'advanced',
        tags: ['pca', 'dimensionality'],
        confidence: 0.95
      },
      
      'outlier': {
        keywords: ['이상치', 'outlier', '특이값', '이상', '비정상'],
        category: 'analysis',
        subcategory: 'advanced',
        tags: ['outlier', 'anomaly'],
        confidence: 0.9
      },

      // 시각화
      'visualization': {
        keywords: ['시각화', '차트', '그래프', 'plot', 'chart', 'graph', '그림', '그리기'],
        category: 'visualization',
        subcategory: 'general',
        tags: ['visualization', 'plotting'],
        confidence: 0.8
      },

      // 시계열
      'timeseries': {
        keywords: ['시계열', 'timeseries', '시간', '트렌드', '계절성', 'forecast'],
        category: 'analysis',
        subcategory: 'timeseries',
        tags: ['timeseries', 'temporal'],
        confidence: 0.9
      },

      // 전처리
      'preprocessing': {
        keywords: ['전처리', 'preprocessing', '정제', '클리닝', '변환', 'clean'],
        category: 'data',
        subcategory: 'preprocessing',
        tags: ['preprocessing', 'cleaning'],
        confidence: 0.8
      }
    };
  }

  extractKeywords(query) {
    // 간단한 키워드 추출
    const words = query.split(/\s+/).filter(word => word.length > 1);
    return words.map(word => word.toLowerCase());
  }

  matchesPattern(query, pattern) {
    return pattern.keywords.some(keyword => query.includes(keyword));
  }

  async findBestModule(query, intent, options = {}) {
    // 1. 의도 기반 검색
    if (intent.category) {
      const intentBasedModule = this.moduleScanner.findBestMatch(query, {
        category: intent.category,
        subcategory: intent.subcategory,
        tags: intent.tags,
        executableOnly: true
      });
      
      if (intentBasedModule) {
        this.logger.debug('의도 기반 모듈 매칭 성공');
        return intentBasedModule;
      }
    }

    // 2. 전체 검색 (실행 가능한 모듈만)
    const generalModule = this.moduleScanner.findBestMatch(query, {
      executableOnly: true,
      fuzzy: true
    });
    
    if (generalModule) {
      this.logger.debug('일반 검색 모듈 매칭 성공');
      return generalModule;
    }

    // 3. 카테고리별 대안 검색
    if (intent.category) {
      const categoryModules = this.moduleScanner.getModulesByCategory(intent.category);
      const executableCategoryModules = categoryModules.filter(m => m.isExecutable);
      
      if (executableCategoryModules.length > 0) {
        this.logger.debug('카테고리 기반 대안 모듈 선택');
        return executableCategoryModules[0];
      }
    }

    return null;
  }

  async handleNoModuleFound(query, intent) {
    this.logger.warn(`모듈을 찾을 수 없음: ${query}`);
    
    // 대안 모듈 제안
    const suggestions = await this.generateSuggestions(query, intent);
    
    const error = new Error(`'${query}'와 관련된 분석 모듈을 찾을 수 없습니다.`);
    error.suggestions = suggestions;
    error.intent = intent;
    
    throw error;
  }

  async generateSuggestions(query, intent) {
    const suggestions = [];
    
    // 1. 유사한 모듈 찾기
    const similarModules = this.moduleScanner.searchModules(query, { fuzzy: true })
      .slice(0, 3)
      .map(module => ({
        id: module.id,
        name: module.displayName,
        category: module.category,
        description: module.description || '설명 없음',
        score: this.moduleScanner.calculateMatchScore(module, query)
      }));

    if (similarModules.length > 0) {
      suggestions.push({
        type: 'similar_modules',
        title: '유사한 모듈들',
        modules: similarModules
      });
    }

    // 2. 카테고리별 추천
    if (intent.category) {
      const categoryModules = this.moduleScanner.getModulesByCategory(intent.category)
        .filter(m => m.isExecutable)
        .slice(0, 3)
        .map(module => ({
          id: module.id,
          name: module.displayName,
          description: module.description || '설명 없음'
        }));

      if (categoryModules.length > 0) {
        suggestions.push({
          type: 'category_modules',
          title: `${intent.category} 카테고리 모듈들`,
          modules: categoryModules
        });
      }
    }

    // 3. 범용 분석 도구 추천
    const generalModules = this.moduleScanner.getModulesByCategory('analysis')
      .filter(m => m.isExecutable && m.subcategory === 'basic')
      .slice(0, 2)
      .map(module => ({
        id: module.id,
        name: module.displayName,
        description: module.description || '범용 분석 도구'
      }));

    if (generalModules.length > 0) {
      suggestions.push({
        type: 'general_tools',
        title: '범용 분석 도구',
        modules: generalModules
      });
    }

    return suggestions;
  }

  async executeModule(module, data = null, options = {}) {
    try {
      this.logger.info(`모듈 실행 시작: ${module.displayName}`);
      
      // 모듈 검증
      await this.validateModuleForExecution(module);

      // 실행 파라미터 준비
      const executionParams = this.prepareExecutionParams(module, data, options);

      // Python 스크립트 실행
      const result = await this.pythonExecutor.executeScript(module.path, executionParams);

      // 결과 처리
      const processedResult = this.processExecutionResult(result, module);
      
      this.logger.info(`모듈 실행 완료: ${module.displayName}`);
      return processedResult;

    } catch (error) {
      this.logger.error(`모듈 실행 실패: ${module.id}`, error);
      throw new Error(`모듈 '${module.displayName}' 실행 중 오류 발생: ${error.message}`);
    }
  }

  async validateModuleForExecution(module) {
    // 1. 파일 존재 확인
    try {
      await import('fs/promises').then(fs => fs.access(module.path));
    } catch {
      throw new Error(`모듈 파일을 찾을 수 없습니다: ${module.path}`);
    }

    // 2. 에러 상태 확인
    if (module.hasErrors) {
      this.logger.warn(`모듈에 오류가 있을 수 있습니다: ${module.errorMessage}`);
    }

    // 3. 실행 가능성 확인
    if (!module.isExecutable) {
      this.logger.warn(`모듈 ${module.id}는 표준 실행 함수가 없을 수 있습니다.`);
    }
  }

  prepareExecutionParams(module, data, options) {
    const params = {
      timeout: options.timeout || this.defaultTimeout,
      workingDir: options.workingDir || './temp',
      env: {
        MODULE_ID: module.id,
        MODULE_NAME: module.name,
        MODULE_CATEGORY: module.category,
        MODULE_SUBCATEGORY: module.subcategory,
        ...options.env
      },
      captureOutput: true
    };

    // 데이터 파라미터 추가
    if (data) {
      params.data = data;
    }

    // 모듈별 옵션 추가
    if (options.moduleOptions) {
      params.options = options.moduleOptions;
    }

    return params;
  }

  processExecutionResult(result, module) {
    // 결과 구조 검증
    if (!result) {
      throw new Error('모듈에서 결과를 반환하지 않았습니다.');
    }

    // 오류 처리
    if (result.error) {
      throw new Error(`모듈 실행 오류: ${result.error}`);
    }

    // 결과 메타데이터 추가
    return {
      ...result,
      module: {
        id: module.id,
        name: module.displayName,
        category: module.category,
        subcategory: module.subcategory,
        path: module.path
      },
      executedAt: new Date().toISOString()
    };
  }

  formatModuleInfo(module) {
    return {
      id: module.id,
      name: module.displayName,
      category: module.category,
      subcategory: module.subcategory,
      description: module.description,
      isExecutable: module.isExecutable,
      hasErrors: module.hasErrors,
      tags: module.tags
    };
  }

  addToExecutionHistory(query, module, result, executionTime, success, errorMessage = null) {
    this.executionHistory.unshift({
      query: query,
      module: module ? {
        id: module.id,
        name: module.displayName,
        path: module.path
      } : null,
      success: success,
      executionTime: executionTime,
      executedAt: new Date().toISOString(),
      errorMessage: errorMessage
    });

    // 히스토리 크기 제한
    if (this.executionHistory.length > this.maxHistorySize) {
      this.executionHistory = this.executionHistory.slice(0, this.maxHistorySize);
    }
  }

  // 모듈 정보 제공 메서드들

  async getAvailableModules(category = null, options = {}) {
    const {
      executableOnly = false,
      includeErrors = false
    } = options;

    let modules = category ? 
      this.moduleScanner.getModulesByCategory(category) : 
      this.moduleScanner.getAllModules();

    // 필터링
    if (executableOnly) {
      modules = modules.filter(m => m.isExecutable);
    }

    if (!includeErrors) {
      modules = modules.filter(m => !m.hasErrors);
    }

    return modules.map(module => this.formatModuleInfo(module));
  }

  async suggestModules(query, limit = 5) {
    const matches = this.moduleScanner.searchModules(query, { 
      fuzzy: true,
      executableOnly: true
    });
    
    return matches.slice(0, limit).map(module => ({
      id: module.id,
      name: module.displayName,
      description: module.description || '설명 없음',
      category: module.category,
      subcategory: module.subcategory,
      tags: module.tags,
      relevanceScore: this.moduleScanner.calculateMatchScore(module, query)
    }));
  }

  getExecutionHistory(limit = 10) {
    return this.executionHistory.slice(0, limit);
  }

  getModuleStats() {
    const stats = this.moduleScanner.getModuleStats();
    
    // 실행 히스토리 통계 추가
    const recentExecutions = this.executionHistory.slice(0, 20);
    const successfulExecutions = recentExecutions.filter(h => h.success).length;
    
    return {
      ...stats,
      execution: {
        totalInHistory: this.executionHistory.length,
        recent: recentExecutions.length,
        recentSuccessRate: recentExecutions.length > 0 ? 
          (successfulExecutions / recentExecutions.length * 100).toFixed(1) + '%' : 'N/A',
        averageExecutionTime: this.calculateAverageExecutionTime(recentExecutions)
      }
    };
  }

  calculateAverageExecutionTime(executions) {
    if (executions.length === 0) return 0;
    
    const totalTime = executions.reduce((sum, exec) => sum + (exec.executionTime || 0), 0);
    return Math.round(totalTime / executions.length);
  }

  // 개발자 및 관리 도구

  async refreshModules() {
    this.logger.info('모듈 새로고침 시작');
    return await this.moduleScanner.refreshModules();
  }

  async testModule(moduleId, testData = null, options = {}) {
    const module = this.moduleScanner.getModuleById(moduleId);
    
    if (!module) {
      throw new Error(`모듈을 찾을 수 없습니다: ${moduleId}`);
    }

    const testOptions = {
      ...options,
      timeout: options.timeout || 60000 // 테스트는 1분 제한
    };

    try {
      const result = await this.executeModule(module, testData, testOptions);
      
      return {
        success: true,
        moduleId: moduleId,
        moduleName: module.displayName,
        result: result,
        executionTime: result.executionTime || 0
      };
      
    } catch (error) {
      return {
        success: false,
        moduleId: moduleId,
        moduleName: module.displayName,
        error: error.message,
        errorType: error.name || 'UnknownError'
      };
    }
  }

  async validateAllModules() {
    const modules = this.moduleScanner.getAllModules();
    const results = [];

    for (const module of modules) {
      try {
        const testResult = await this.testModule(module.id, null, { timeout: 30000 });
        results.push(testResult);
      } catch (error) {
        results.push({
          success: false,
          moduleId: module.id,
          moduleName: module.displayName,
          error: error.message
        });
      }
    }

    return {
      total: results.length,
      successful: results.filter(r => r.success).length,
      failed: results.filter(r => !r.success).length,
      results: results
    };
  }

  async getModuleDetails(moduleId) {
    const module = this.moduleScanner.getModuleById(moduleId);
    
    if (!module) {
      throw new Error(`모듈을 찾을 수 없습니다: ${moduleId}`);
    }

    // 실행 히스토리에서 해당 모듈 찾기
    const moduleHistory = this.executionHistory.filter(h => 
      h.module && h.module.id === moduleId
    );

    return {
      ...module,
      usage: {
        totalExecutions: moduleHistory.length,
        successfulExecutions: moduleHistory.filter(h => h.success).length,
        lastExecuted: moduleHistory.length > 0 ? moduleHistory[0].executedAt : null,
        averageExecutionTime: this.calculateAverageExecutionTime(moduleHistory)
      }
    };
  }

  // 설정 및 상태 관리

  setDefaultTimeout(timeout) {
    this.defaultTimeout = timeout;
  }

  clearExecutionHistory() {
    this.executionHistory = [];
  }

  async getSystemHealth() {
    const stats = this.getModuleStats();
    const recentExecutions = this.executionHistory.slice(0, 10);
    
    return {
      moduleScanner: {
        status: 'healthy',
        totalModules: stats.total,
        executableModules: stats.executable,
        lastScan: stats.lastScan,
        isScanning: stats.isScanning
      },
      pythonExecutor: {
        status: 'healthy', // 실제로는 executor 상태 확인 필요
        available: true
      },
      execution: {
        recentExecutions: recentExecutions.length,
        recentSuccessRate: stats.execution.recentSuccessRate,
        averageExecutionTime: stats.execution.averageExecutionTime
      },
      timestamp: new Date().toISOString()
    };
  }
}