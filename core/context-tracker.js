// core/context-tracker.js
import { Logger } from '../utils/logger.js';
import fs from 'fs/promises';
import path from 'path';

export class ContextTracker {
  constructor() {
    this.logger = new Logger();
    this.currentMode = 'general';
    this.sessionData = new Map();
    this.contextHistory = [];
    this.stateFile = './data/state/current-mode.json';
  }

  async initialize() {
    try {
      await this.loadState();
      this.logger.info('컨텍스트 추적기 초기화 완료');
    } catch (error) {
      this.logger.warn('컨텍스트 상태 로드 실패, 기본값 사용:', error);
      this.currentMode = 'general';
    }
  }

  async loadState() {
    try {
      const stateData = await fs.readFile(this.stateFile, 'utf-8');
      const state = JSON.parse(stateData);
      this.currentMode = state.currentMode || 'general';
      this.contextHistory = state.contextHistory || [];
    } catch (error) {
      // 파일이 없으면 기본값 유지
      await this.saveState();
    }
  }

  async saveState() {
    try {
      const stateDir = path.dirname(this.stateFile);
      await fs.mkdir(stateDir, { recursive: true });
      
      const state = {
        currentMode: this.currentMode,
        contextHistory: this.contextHistory.slice(-100), // 최근 100개만 저장
        lastUpdated: new Date().toISOString()
      };
      
      await fs.writeFile(this.stateFile, JSON.stringify(state, null, 2));
    } catch (error) {
      this.logger.error('상태 저장 실패:', error);
    }
  }

  updateContext(toolName, args) {
    const contextEntry = {
      timestamp: new Date().toISOString(),
      toolName,
      args,
      mode: this.currentMode
    };

    this.contextHistory.push(contextEntry);
    
    // 메모리 관리를 위해 최근 1000개만 유지
    if (this.contextHistory.length > 1000) {
      this.contextHistory = this.contextHistory.slice(-1000);
    }

    // 주기적으로 상태 저장
    this.saveState();
  }

  getCurrentMode() {
    return this.currentMode;
  }

  async setMode(newMode) {
    if (this.currentMode !== newMode) {
      this.logger.info(`모드 변경: ${this.currentMode} → ${newMode}`);
      this.currentMode = newMode;
      
      this.updateContext('mode_change', {
        previousMode: this.currentMode,
        newMode: newMode
      });
      
      await this.saveState();
    }
  }

  getRecentContext(limit = 10) {
    return this.contextHistory.slice(-limit);
  }

  getContextByMode(mode, limit = 50) {
    return this.contextHistory
      .filter(entry => entry.mode === mode)
      .slice(-limit);
  }

  getSessionData(sessionId) {
    return this.sessionData.get(sessionId) || {};
  }

  setSessionData(sessionId, data) {
    this.sessionData.set(sessionId, {
      ...this.getSessionData(sessionId),
      ...data,
      lastUpdated: new Date().toISOString()
    });
  }

  clearSessionData(sessionId) {
    this.sessionData.delete(sessionId);
  }

  // 통계 및 분석
  getUsageStats() {
    const now = new Date();
    const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000);
    
    const recentEntries = this.contextHistory.filter(
      entry => new Date(entry.timestamp) > oneHourAgo
    );

    const toolUsage = {};
    const modeUsage = {};

    recentEntries.forEach(entry => {
      toolUsage[entry.toolName] = (toolUsage[entry.toolName] || 0) + 1;
      modeUsage[entry.mode] = (modeUsage[entry.mode] || 0) + 1;
    });

    return {
      totalEntries: recentEntries.length,
      toolUsage,
      modeUsage,
      currentMode: this.currentMode,
      activeSessions: this.sessionData.size
    };
  }
}
