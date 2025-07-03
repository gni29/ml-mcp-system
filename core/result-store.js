// core/result-store.js
import { Logger } from '../utils/logger.js';
import fs from 'fs/promises';
import path from 'path';

export class ResultStore {
  constructor() {
    this.logger = new Logger();
    this.sessionResults = new Map();
    this.globalResults = new Map();
    this.maxSessionResults = 100; // 세션당 최대 결과 수
    this.maxGlobalResults = 1000; // 전체 최대 결과 수
  }

  async initialize() {
    try {
      await this.loadPersistentResults();
      this.logger.info('ResultStore 초기화 완료');
    } catch (error) {
      this.logger.warn('ResultStore 초기화 중 오류:', error);
    }
  }

  // 세션별 결과 저장
  storeSessionResult(sessionId, resultType, result) {
    const sessionKey = `session_${sessionId}`;
    
    if (!this.sessionResults.has(sessionKey)) {
      this.sessionResults.set(sessionKey, new Map());
    }
    
    const sessionData = this.sessionResults.get(sessionKey);
    const resultKey = `${resultType}_${Date.now()}`;
    
    const resultData = {
      key: resultKey,
      type: resultType,
      result: result,
      timestamp: new Date().toISOString(),
      sessionId: sessionId
    };
    
    sessionData.set(resultKey, resultData);
    
    // 세션 결과 수 제한
    if (sessionData.size > this.maxSessionResults) {
      const oldestKey = sessionData.keys().next().value;
      sessionData.delete(oldestKey);
    }
    
    this.logger.debug(`세션 결과 저장: ${sessionId} - ${resultType}`);
    return resultKey;
  }

  // 세션별 결과 조회
  getSessionResult(sessionId, resultKey) {
    const sessionKey = `session_${sessionId}`;
    const sessionData = this.sessionResults.get(sessionKey);
    
    if (!sessionData) {
      return null;
    }
    
    return sessionData.get(resultKey);
  }

  // 세션의 모든 결과 조회
  getSessionResults(sessionId) {
    const sessionKey = `session_${sessionId}`;
    const sessionData = this.sessionResults.get(sessionKey);
    
    if (!sessionData) {
      return [];
    }
    
    return Array.from(sessionData.values());
  }

  // 세션의 특정 타입 결과들 조회
  getSessionResultsByType(sessionId, resultType) {
    const sessionResults = this.getSessionResults(sessionId);
    return sessionResults.filter(result => result.type === resultType);
  }

  // 세션의 최신 결과 조회
  getLatestSessionResult(sessionId, resultType = null) {
    const sessionResults = this.getSessionResults(sessionId);
    
    let filteredResults = sessionResults;
    if (resultType) {
      filteredResults = sessionResults.filter(result => result.type === resultType);
    }
    
    if (filteredResults.length === 0) {
      return null;
    }
    
    // 타임스탬프 기준으로 정렬하여 최신 결과 반환
    return filteredResults.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))[0];
  }

  // 글로벌 결과 저장 (모든 세션에서 접근 가능)
  storeGlobalResult(key, result, metadata = {}) {
    const resultData = {
      key: key,
      result: result,
      metadata: metadata,
      timestamp: new Date().toISOString(),
      accessCount: 0
    };
    
    this.globalResults.set(key, resultData);
    
    // 글로벌 결과 수 제한
    if (this.globalResults.size > this.maxGlobalResults) {
      // 가장 오래된 결과 삭제
      const oldestKey = this.globalResults.keys().next().value;
      this.globalResults.delete(oldestKey);
    }
    
    this.logger.debug(`글로벌 결과 저장: ${key}`);
    return key;
  }

  // 글로벌 결과 조회
  getGlobalResult(key) {
    const result = this.globalResults.get(key);
    if (result) {
      result.accessCount++;
      result.lastAccessed = new Date().toISOString();
    }
    return result;
  }

  // 결과 간 연관관계 설정
  linkResults(sourceSessionId, sourceResultKey, targetSessionId, targetResultKey, relation = 'derived_from') {
    const sourceResult = this.getSessionResult(sourceSessionId, sourceResultKey);
    const targetResult = this.getSessionResult(targetSessionId, targetResultKey);
    
    if (!sourceResult || !targetResult) {
      throw new Error('연결할 결과를 찾을 수 없습니다.');
    }
    
    // 연관관계 정보 추가
    if (!sourceResult.relations) {
      sourceResult.relations = [];
    }
    
    sourceResult.relations.push({
      type: relation,
      targetSessionId: targetSessionId,
      targetResultKey: targetResultKey,
      timestamp: new Date().toISOString()
    });
    
    this.logger.debug(`결과 연관관계 설정: ${sourceSessionId}/${sourceResultKey} -> ${targetSessionId}/${targetResultKey}`);
  }

  // 워크플로우 결과 저장
  storeWorkflowResult(sessionId, workflowName, steps, finalResult) {
    const workflowResult = {
      workflowName: workflowName,
      steps: steps,
      finalResult: finalResult,
      timestamp: new Date().toISOString(),
      sessionId: sessionId
    };
    
    const resultKey = this.storeSessionResult(sessionId, 'workflow', workflowResult);
    
    // 각 단계별 결과도 개별 저장
    steps.forEach((step, index) => {
      if (step.success) {
        this.storeSessionResult(sessionId, `step_${index + 1}_${step.type}`, step.result);
      }
    });
    
    return resultKey;
  }

  // 중간 결과 저장 (파이프라인 단계 간 데이터 전달용)
  storeIntermediateResult(sessionId, stepId, result) {
    const intermediateKey = `intermediate_${stepId}`;
    return this.storeSessionResult(sessionId, intermediateKey, result);
  }

  // 중간 결과 조회
  getIntermediateResult(sessionId, stepId) {
    const intermediateKey = `intermediate_${stepId}`;
    const results = this.getSessionResultsByType(sessionId, intermediateKey);
    return results.length > 0 ? results[results.length - 1] : null;
  }

  // 결과 검색
  searchResults(query, sessionId = null, resultType = null) {
    const results = [];
    
    // 세션별 검색
    if (sessionId) {
      const sessionResults = this.getSessionResults(sessionId);
      results.push(...sessionResults);
    } else {
      // 모든 세션 검색
      for (const sessionData of this.sessionResults.values()) {
        results.push(...Array.from(sessionData.values()));
      }
    }
    
    // 타입별 필터링
    let filteredResults = results;
    if (resultType) {
      filteredResults = results.filter(result => result.type === resultType);
    }
    
    // 쿼리 검색
    if (query) {
      const queryLower = query.toLowerCase();
      filteredResults = filteredResults.filter(result => {
        const resultStr = JSON.stringify(result).toLowerCase();
        return resultStr.includes(queryLower);
      });
    }
    
    return filteredResults.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
  }

  // 세션 정리
  clearSession(sessionId) {
    const sessionKey = `session_${sessionId}`;
    const deleted = this.sessionResults.delete(sessionKey);
    
    if (deleted) {
      this.logger.info(`세션 결과 정리 완료: ${sessionId}`);
    }
    
    return deleted;
  }

  // 오래된 결과 정리
  cleanupOldResults(maxAgeHours = 24) {
    const cutoffTime = new Date(Date.now() - maxAgeHours * 60 * 60 * 1000);
    let cleanedCount = 0;
    
    // 세션 결과 정리
    for (const [sessionKey, sessionData] of this.sessionResults) {
      const keysToDelete = [];
      
      for (const [resultKey, result] of sessionData) {
        if (new Date(result.timestamp) < cutoffTime) {
          keysToDelete.push(resultKey);
        }
      }
      
      keysToDelete.forEach(key => {
        sessionData.delete(key);
        cleanedCount++;
      });
      
      // 빈 세션 제거
      if (sessionData.size === 0) {
        this.sessionResults.delete(sessionKey);
      }
    }
    
    // 글로벌 결과 정리
    const globalKeysToDelete = [];
    for (const [key, result] of this.globalResults) {
      if (new Date(result.timestamp) < cutoffTime) {
        globalKeysToDelete.push(key);
      }
    }
    
    globalKeysToDelete.forEach(key => {
      this.globalResults.delete(key);
      cleanedCount++;
    });
    
    this.logger.info(`오래된 결과 정리 완료: ${cleanedCount}개 결과 삭제`);
    return cleanedCount;
  }

  // 통계 정보 조회
  getStatistics() {
    const totalSessions = this.sessionResults.size;
    let totalResults = 0;
    let resultTypes = new Set();
    
    for (const sessionData of this.sessionResults.values()) {
      totalResults += sessionData.size;
      for (const result of sessionData.values()) {
        resultTypes.add(result.type);
      }
    }
    
    return {
      totalSessions,
      totalResults,
      globalResults: this.globalResults.size,
      resultTypes: Array.from(resultTypes),
      memoryUsage: this.getMemoryUsage()
    };
  }

  // 메모리 사용량 추정
  getMemoryUsage() {
    const sessionSize = JSON.stringify(Array.from(this.sessionResults.entries())).length;
    const globalSize = JSON.stringify(Array.from(this.globalResults.entries())).length;
    
    return {
      sessionResultsKB: Math.round(sessionSize / 1024),
      globalResultsKB: Math.round(globalSize / 1024),
      totalKB: Math.round((sessionSize + globalSize) / 1024)
    };
  }

  // 영구 저장소에 결과 저장
  async savePersistentResults() {
    try {
      const persistentDir = './data/persistent';
      await fs.mkdir(persistentDir, { recursive: true });
      
      // 중요한 결과들만 선별하여 저장
      const importantResults = this.getImportantResults();
      
      const persistentFile = path.join(persistentDir, 'important_results.json');
      await fs.writeFile(persistentFile, JSON.stringify(importantResults, null, 2));
      
      this.logger.info('중요한 결과 영구 저장 완료');
    } catch (error) {
      this.logger.error('영구 저장 실패:', error);
    }
  }

  // 영구 저장소에서 결과 로드
  async loadPersistentResults() {
    try {
      const persistentFile = './data/persistent/important_results.json';
      const data = await fs.readFile(persistentFile, 'utf-8');
      const importantResults = JSON.parse(data);
      
      // 글로벌 결과로 복원
      for (const result of importantResults) {
        this.globalResults.set(result.key, result);
      }
      
      this.logger.info('영구 저장된 결과 로드 완료');
    } catch (error) {
      // 파일이 없으면 무시
      if (error.code !== 'ENOENT') {
        this.logger.warn('영구 저장된 결과 로드 실패:', error);
      }
    }
  }

  // 중요한 결과 선별
  getImportantResults() {
    const important = [];
    
    // 접근 횟수가 많은 글로벌 결과
    for (const result of this.globalResults.values()) {
      if (result.accessCount > 5) {
        important.push(result);
      }
    }
    
    // 최근 워크플로우 결과
    const recentWorkflows = this.searchResults('', null, 'workflow')
      .slice(0, 10); // 최근 10개
    
    important.push(...recentWorkflows);
    
    return important;
  }

  // 결과 내보내기
  async exportResults(sessionId, format = 'json') {
    const results = sessionId ? this.getSessionResults(sessionId) : this.searchResults('');
    
    const exportData = {
      exportTime: new Date().toISOString(),
      sessionId: sessionId,
      totalResults: results.length,
      results: results
    };
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `results_export_${timestamp}.${format}`;
    const exportPath = path.join('./exports', filename);
    
    await fs.mkdir('./exports', { recursive: true });
    
    if (format === 'json') {
      await fs.writeFile(exportPath, JSON.stringify(exportData, null, 2));
    } else if (format === 'csv') {
      const csv = this.convertToCSV(results);
      await fs.writeFile(exportPath, csv);
    }
    
    this.logger.info(`결과 내보내기 완료: ${exportPath}`);
    return exportPath;
  }

  // CSV 변환
  convertToCSV(results) {
    if (results.length === 0) return '';
    
    const headers = ['timestamp', 'sessionId', 'type', 'key', 'result'];
    const rows = results.map(result => [
      result.timestamp,
      result.sessionId,
      result.type,
      result.key,
      JSON.stringify(result.result)
    ]);
    
    return [headers, ...rows].map(row => row.join(',')).join('\n');
  }

  // 결과 가져오기 (외부 파일에서)
  async importResults(filePath) {
    try {
      const data = await fs.readFile(filePath, 'utf-8');
      const importData = JSON.parse(data);
      
      let importedCount = 0;
      
      if (importData.results && Array.isArray(importData.results)) {
        for (const result of importData.results) {
          if (result.sessionId && result.key) {
            this.storeSessionResult(result.sessionId, result.type, result.result);
            importedCount++;
          }
        }
      }
      
      this.logger.info(`결과 가져오기 완료: ${importedCount}개 결과 추가`);
      return importedCount;
    } catch (error) {
      this.logger.error('결과 가져오기 실패:', error);
      throw error;
    }
  }

  // 결과 백업
  async backupResults() {
    const backupData = {
      backupTime: new Date().toISOString(),
      sessionResults: Array.from(this.sessionResults.entries()).map(([key, value]) => [
        key,
        Array.from(value.entries())
      ]),
      globalResults: Array.from(this.globalResults.entries()),
      statistics: this.getStatistics()
    };
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const backupPath = path.join('./backups', `results_backup_${timestamp}.json`);
    
    await fs.mkdir('./backups', { recursive: true });
    await fs.writeFile(backupPath, JSON.stringify(backupData, null, 2));
    
    this.logger.info(`결과 백업 완료: ${backupPath}`);
    return backupPath;
  }

  // 백업에서 복원
  async restoreFromBackup(backupPath) {
    try {
      const data = await fs.readFile(backupPath, 'utf-8');
      const backupData = JSON.parse(data);
      
      // 기존 데이터 초기화
      this.sessionResults.clear();
      this.globalResults.clear();
      
      // 세션 결과 복원
      if (backupData.sessionResults) {
        for (const [sessionKey, sessionEntries] of backupData.sessionResults) {
          const sessionMap = new Map(sessionEntries);
          this.sessionResults.set(sessionKey, sessionMap);
        }
      }
      
      // 글로벌 결과 복원
      if (backupData.globalResults) {
        for (const [key, result] of backupData.globalResults) {
          this.globalResults.set(key, result);
        }
      }
      
      this.logger.info('백업에서 복원 완료');
      return true;
    } catch (error) {
      this.logger.error('백업 복원 실패:', error);
      throw error;
    }
  }

  // 결과 유효성 검사
  validateResult(result) {
    if (!result || typeof result !== 'object') {
      return false;
    }
    
    const requiredFields = ['timestamp', 'sessionId', 'type', 'key'];
    return requiredFields.every(field => result.hasOwnProperty(field));
  }

  // 정리 작업 스케줄링
  scheduleCleanup() {
    // 1시간마다 정리 작업 실행
    setInterval(() => {
      this.cleanupOldResults(24); // 24시간 이상 된 결과 정리
    }, 60 * 60 * 1000);
    
    // 12시간마다 중요한 결과 저장
    setInterval(() => {
      this.savePersistentResults();
    }, 12 * 60 * 60 * 1000);
  }
}
