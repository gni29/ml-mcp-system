// tools/discovery/module-scanner.js
import fs from 'fs/promises';
import path from 'path';
import { Logger } from '../../utils/logger.js';

export class ModuleScanner {
  constructor() {
    this.logger = new Logger();
    this.pythonBasePath = './python';
    this.moduleCache = new Map();
    this.lastScanTime = null;
    this.scanInterval = 30000; // 30초마다 스캔
    this.isScanning = false;
  }

  async initialize() {
    try {
      this.logger.info('모듈 스캐너 초기화 시작');
      
      // 초기 스캔 실행
      await this.scanAllModules();
      
      // 주기적 스캔 시작
      this.startPeriodicScan();
      
      this.logger.info('모듈 스캐너 초기화 완료');
    } catch (error) {
      this.logger.error('모듈 스캐너 초기화 실패:', error);
      throw error;
    }
  }

  async scanAllModules() {
    if (this.isScanning) {
      this.logger.debug('이미 스캔 중입니다');
      return this.getLastScanResult();
    }

    try {
      this.isScanning = true;
      this.logger.info('Python 모듈 전체 스캔 시작');
      const startTime = Date.now();
      
      // 캐시 초기화
      this.moduleCache.clear();

      // Python 디렉토리 확인
      if (!await this.checkPythonDirectory()) {
        return this.createEmptyResult();
      }

      // 재귀적 스캔
      const modules = await this.scanDirectory(this.pythonBasePath);
      
      // 모듈 메타데이터 추출
      await this.extractAllMetadata(modules);

      // 스캔 결과 정리
      const scanTime = Date.now() - startTime;
      this.lastScanTime = Date.now();
      
      const result = {
        modules: Array.from(this.moduleCache.values()),
        count: modules.length,
        scanTime: scanTime,
        lastScan: this.lastScanTime,
        success: true
      };

      this.logger.info(`모듈 스캔 완료: ${modules.length}개 모듈 발견 (${scanTime}ms)`);
      return result;

    } catch (error) {
      this.logger.error('모듈 스캔 실패:', error);
      throw error;
    } finally {
      this.isScanning = false;
    }
  }

  async checkPythonDirectory() {
    try {
      await fs.access(this.pythonBasePath);
      const stat = await fs.stat(this.pythonBasePath);
      return stat.isDirectory();
    } catch {
      this.logger.warn(`Python 디렉토리가 존재하지 않습니다: ${this.pythonBasePath}`);
      return false;
    }
  }

  createEmptyResult() {
    return {
      modules: [],
      count: 0,
      scanTime: 0,
      lastScan: Date.now(),
      success: false,
      error: 'Python 디렉토리 없음'
    };
  }

  async scanDirectory(dirPath, basePath = this.pythonBasePath) {
    const modules = [];
    
    try {
      const entries = await fs.readdir(dirPath, { withFileTypes: true });
      
      for (const entry of entries) {
        const fullPath = path.join(dirPath, entry.name);
        
        // 제외할 디렉토리/파일 확인
        if (this.shouldSkipEntry(entry.name)) {
          continue;
        }
        
        if (entry.isDirectory()) {
          // 하위 디렉토리 재귀 탐색
          const subModules = await this.scanDirectory(fullPath, basePath);
          modules.push(...subModules);
        } else if (entry.isFile() && entry.name.endsWith('.py')) {
          // Python 파일 처리
          const module = await this.createModuleInfo(fullPath, basePath);
          if (module) {
            modules.push(module);
            this.moduleCache.set(module.id, module);
          }
        }
      }
      
    } catch (error) {
      this.logger.warn(`디렉토리 스캔 실패: ${dirPath}`, error);
    }
    
    return modules;
  }

  shouldSkipEntry(name) {
    const skipPatterns = [
      '__pycache__',
      '.git',
      '.pytest_cache',
      '__init__.py',
      'test_',
      '.pyc'
    ];
    
    return skipPatterns.some(pattern => 
      name.includes(pattern) || name.startsWith('.')
    );
  }

  async createModuleInfo(fullPath, basePath) {
    try {
      const stats = await fs.stat(fullPath);
      const relativePath = path.relative(basePath, fullPath);
      const moduleName = path.basename(fullPath, '.py');
      
      const moduleInfo = {
        id: this.generateModuleId(relativePath),
        name: moduleName,
        displayName: this.formatDisplayName(moduleName),
        category: this.determineCategory(relativePath),
        subcategory: this.determineSubcategory(relativePath),
        path: fullPath,
        relativePath: relativePath,
        lastModified: stats.mtime,
        size: stats.size,
        description: null,
        functions: [],
        dependencies: [],
        tags: this.generateTags(relativePath, moduleName),
        isExecutable: false,
        hasErrors: false,
        errorMessage: null
      };

      return moduleInfo;
    } catch (error) {
      this.logger.warn(`모듈 정보 생성 실패: ${fullPath}`, error);
      return null;
    }
  }

  generateModuleId(relativePath) {
    return relativePath
      .replace(/\.py$/, '')
      .replace(/[\/\\]/g, '.')
      .toLowerCase();
  }

  determineCategory(relativePath) {
    const pathParts = relativePath.toLowerCase().split(/[\/\\]/);
    
    const categoryMap = {
      'analysis': 'analysis',
      'ml': 'ml',
      'visualization': 'visualization',
      'data': 'data',
      'utils': 'utils',
      'models': 'models',
      'preprocessing': 'preprocessing',
      'postprocessing': 'postprocessing'
    };

    for (const part of pathParts) {
      if (categoryMap[part]) {
        return categoryMap[part];
      }
    }

    return 'custom';
  }

  determineSubcategory(relativePath) {
    const pathParts = relativePath.toLowerCase().split(/[\/\\]/);
    
    const subcategoryMap = {
      'basic': 'basic',
      'advanced': 'advanced',
      'supervised': 'supervised',
      'unsupervised': 'unsupervised',
      'deep_learning': 'deep_learning',
      'timeseries': 'timeseries',
      'nlp': 'nlp',
      'computer_vision': 'computer_vision',
      'statistical': 'statistical'
    };

    for (const part of pathParts) {
      if (subcategoryMap[part]) {
        return subcategoryMap[part];
      }
    }

    return 'general';
  }

  formatDisplayName(moduleName) {
    return moduleName
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  }

  generateTags(relativePath, moduleName) {
    const tags = new Set();
    const pathParts = relativePath.toLowerCase().split(/[\/\\]/);
    const nameParts = moduleName.toLowerCase().split('_');
    
    // 경로 기반 태그
    pathParts.forEach(part => {
      if (part && part !== 'python' && !part.endsWith('.py')) {
        tags.add(part);
      }
    });
    
    // 이름 기반 태그
    const keywordTags = {
      'stats': 'statistics',
      'stat': 'statistics',
      'correlation': 'correlation',
      'regression': 'regression',
      'classification': 'classification',
      'cluster': 'clustering',
      'plot': 'visualization',
      'chart': 'visualization',
      'graph': 'visualization',
      'predict': 'prediction',
      'forecast': 'forecasting',
      'model': 'modeling',
      'train': 'training',
      'test': 'testing',
      'valid': 'validation',
      'preprocess': 'preprocessing',
      'clean': 'cleaning',
      'transform': 'transformation'
    };

    nameParts.forEach(part => {
      if (keywordTags[part]) {
        tags.add(keywordTags[part]);
      }
      tags.add(part);
    });

    return Array.from(tags).filter(tag => tag.length > 1);
  }

  async extractAllMetadata(modules) {
    const promises = modules.map(module => this.extractModuleMetadata(module));
    await Promise.allSettled(promises);
  }

  async extractModuleMetadata(moduleInfo) {
    try {
      const content = await fs.readFile(moduleInfo.path, 'utf-8');
      
      // 모듈 docstring 추출
      moduleInfo.description = this.extractDocstring(content);
      
      // 함수 목록 추출
      moduleInfo.functions = this.extractFunctions(content);
      
      // 실행 가능성 확인
      moduleInfo.isExecutable = this.checkExecutability(moduleInfo.functions);
      
      // 의존성 추출
      moduleInfo.dependencies = this.extractDependencies(content);
      
      // 구문 검사
      await this.validateSyntax(moduleInfo);
      
    } catch (error) {
      this.logger.warn(`메타데이터 추출 실패: ${moduleInfo.path}`, error);
      moduleInfo.hasErrors = true;
      moduleInfo.errorMessage = error.message;
    }
  }

  extractDocstring(content) {
    // 모듈 레벨 docstring 찾기
    const patterns = [
      /^"""([\s\S]*?)"""/m,
      /^'''([\s\S]*?)'''/m,
      /^#\s*(.*?)$/m
    ];

    for (const pattern of patterns) {
      const match = content.match(pattern);
      if (match) {
        return match[1].trim();
      }
    }

    return null;
  }

  extractFunctions(content) {
    const functions = [];
    const functionPattern = /^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/gm;
    let match;

    while ((match = functionPattern.exec(content)) !== null) {
      functions.push(match[1]);
    }

    return functions;
  }

  checkExecutability(functions) {
    const executableFunctions = [
      'perform_analysis',
      'analyze',
      'execute',
      'run',
      'main',
      'process',
      'calculate',
      'compute'
    ];

    return functions.some(func => executableFunctions.includes(func));
  }

  extractDependencies(content) {
    const dependencies = new Set();
    const importPatterns = [
      /^import\s+([a-zA-Z_][a-zA-Z0-9_.]*)/gm,
      /^from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import/gm
    ];

    importPatterns.forEach(pattern => {
      let match;
      while ((match = pattern.exec(content)) !== null) {
        dependencies.add(match[1]);
      }
    });

    return Array.from(dependencies);
  }

  async validateSyntax(moduleInfo) {
    // 기본적인 Python 구문 검사
    try {
      const content = await fs.readFile(moduleInfo.path, 'utf-8');
      
      // 기본 구문 오류 확인
      const syntaxErrors = this.checkBasicSyntax(content);
      
      if (syntaxErrors.length > 0) {
        moduleInfo.hasErrors = true;
        moduleInfo.errorMessage = syntaxErrors.join('; ');
      }
      
    } catch (error) {
      moduleInfo.hasErrors = true;
      moduleInfo.errorMessage = `구문 검사 실패: ${error.message}`;
    }
  }

  checkBasicSyntax(content) {
    const errors = [];
    const lines = content.split('\n');
    
    lines.forEach((line, index) => {
      const trimmed = line.trim();
      
      // 기본적인 들여쓰기 검사
      if (trimmed.startsWith('def ') || trimmed.startsWith('class ')) {
        if (!trimmed.endsWith(':')) {
          errors.push(`라인 ${index + 1}: 함수/클래스 정의에 콜론 누락`);
        }
      }
      
      // 기본적인 괄호 균형 검사
      const openParens = (line.match(/\(/g) || []).length;
      const closeParens = (line.match(/\)/g) || []).length;
      if (openParens !== closeParens) {
        errors.push(`라인 ${index + 1}: 괄호 불균형`);
      }
    });
    
    return errors;
  }

  startPeriodicScan() {
    setInterval(async () => {
      try {
        await this.scanAllModules();
      } catch (error) {
        this.logger.error('주기적 스캔 실패:', error);
      }
    }, this.scanInterval);
  }

  // 검색 및 조회 메서드들

  searchModules(query, options = {}) {
    const {
      category = null,
      subcategory = null,
      tags = [],
      fuzzy = true,
      executableOnly = false
    } = options;

    const modules = Array.from(this.moduleCache.values());
    const queryLower = query.toLowerCase();
    
    return modules.filter(module => {
      // 에러가 있는 모듈 제외
      if (module.hasErrors && !options.includeErrors) {
        return false;
      }
      
      // 실행 가능한 모듈만
      if (executableOnly && !module.isExecutable) {
        return false;
      }
      
      // 카테고리 필터
      if (category && module.category !== category) {
        return false;
      }
      
      if (subcategory && module.subcategory !== subcategory) {
        return false;
      }
      
      // 태그 필터
      if (tags.length > 0 && !tags.some(tag => module.tags.includes(tag))) {
        return false;
      }
      
      // 텍스트 검색
      const searchFields = [
        module.name,
        module.displayName,
        module.description || '',
        ...module.functions,
        ...module.tags
      ].join(' ').toLowerCase();
      
      if (fuzzy) {
        return searchFields.includes(queryLower);
      } else {
        return searchFields.split(' ').some(field => field === queryLower);
      }
    });
  }

  findBestMatch(query, options = {}) {
    const matches = this.searchModules(query, options);
    
    if (matches.length === 0) {
      return null;
    }
    
    // 점수 기반 정렬
    const scoredMatches = matches.map(module => ({
      module,
      score: this.calculateMatchScore(module, query)
    }));
    
    scoredMatches.sort((a, b) => b.score - a.score);
    
    return scoredMatches[0].module;
  }

  calculateMatchScore(module, query) {
    let score = 0;
    const queryLower = query.toLowerCase();
    
    // 정확한 이름 매칭
    if (module.name.toLowerCase() === queryLower) {
      score += 100;
    } else if (module.name.toLowerCase().includes(queryLower)) {
      score += 60;
    }
    
    // 표시 이름 매칭
    if (module.displayName.toLowerCase().includes(queryLower)) {
      score += 40;
    }
    
    // 함수 이름 매칭
    const functionMatch = module.functions.some(func => 
      func.toLowerCase().includes(queryLower)
    );
    if (functionMatch) {
      score += 30;
    }
    
    // 설명 매칭
    if (module.description && module.description.toLowerCase().includes(queryLower)) {
      score += 20;
    }
    
    // 태그 매칭
    const tagMatch = module.tags.some(tag => tag.includes(queryLower));
    if (tagMatch) {
      score += 25;
    }
    
    // 보너스 점수
    if (module.isExecutable) score += 10;
    if (!module.hasErrors) score += 5;
    if (module.category !== 'custom') score += 5;
    
    return score;
  }

  // 조회 메서드들

  getModulesByCategory(category) {
    return Array.from(this.moduleCache.values())
      .filter(module => module.category === category && !module.hasErrors);
  }

  getModuleById(id) {
    return this.moduleCache.get(id);
  }

  getAllModules() {
    return Array.from(this.moduleCache.values());
  }

  getExecutableModules() {
    return Array.from(this.moduleCache.values())
      .filter(module => module.isExecutable && !module.hasErrors);
  }

  getModuleStats() {
    const modules = this.getAllModules();
    const stats = {
      total: modules.length,
      executable: modules.filter(m => m.isExecutable).length,
      withErrors: modules.filter(m => m.hasErrors).length,
      byCategory: {},
      bySubcategory: {},
      lastScan: this.lastScanTime,
      isScanning: this.isScanning
    };

    modules.forEach(module => {
      // 카테고리별 통계
      if (!stats.byCategory[module.category]) {
        stats.byCategory[module.category] = 0;
      }
      stats.byCategory[module.category]++;
      
      // 하위 카테고리별 통계
      if (!stats.bySubcategory[module.subcategory]) {
        stats.bySubcategory[module.subcategory] = 0;
      }
      stats.bySubcategory[module.subcategory]++;
    });

    return stats;
  }

  getLastScanResult() {
    return {
      modules: Array.from(this.moduleCache.values()),
      count: this.moduleCache.size,
      lastScan: this.lastScanTime,
      isScanning: this.isScanning
    };
  }

  // 유틸리티 메서드들

  async refreshModules() {
    return await this.scanAllModules();
  }

  clearCache() {
    this.moduleCache.clear();
    this.lastScanTime = null;
  }

  setScanInterval(interval) {
    this.scanInterval = interval;
  }
}