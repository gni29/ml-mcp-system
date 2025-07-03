// tools/system/mode-switch.js
import { Logger } from '../../utils/logger.js';
import fs from 'fs/promises';

export class ModeSwitch {
  constructor(contextTracker, modelManager) {
    this.contextTracker = contextTracker;
    this.modelManager = modelManager;
    this.logger = new Logger();
    this.modes = null;
    this.currentMode = 'general';
    this.transitionInProgress = false;
    this.modeHistory = [];
    this.autoSwitchEnabled = true;
  }

  async initialize() {
    try {
      await this.loadModeConfigurations();
      this.currentMode = this.contextTracker.getCurrentMode();
      this.logger.info('ModeSwitch 초기화 완료', { currentMode: this.currentMode });
    } catch (error) {
      this.logger.error('ModeSwitch 초기화 실패:', error);
      throw error;
    }
  }

  async loadModeConfigurations() {
    try {
      const modesData = await fs.readFile('./config/modes.json', 'utf-8');
      this.modes = JSON.parse(modesData);
      
      // 자동 전환 설정 로드
      this.autoSwitchEnabled = this.modes.mode_switching?.auto_switch?.enabled ?? true;
      
    } catch (error) {
      this.logger.warn('모드 설정 로드 실패, 기본값 사용:', error);
      this.modes = this.getDefaultModes();
    }
  }

  getDefaultModes() {
    return {
      general: {
        name: "일반 모드",
        description: "기본적인 대화 및 간단한 작업 처리",
        active_model: "router",
        tools: ["general_query", "mode_switch", "system_status"],
        features: {
          natural_language_processing: true,
          basic_calculations: true,
          file_operations: true
        }
      },
      ml: {
        name: "머신러닝 모드",
        description: "데이터 분석, ML 모델 훈련 및 시각화",
        active_model: "processor",
        tools: ["analyze_data", "train_model", "visualize_data"],
        features: {
          data_analysis: true,
          machine_learning: true,
          data_visualization: true
        }
      }
    };
  }

  async switchMode(targetMode, options = {}) {
    if (this.transitionInProgress) {
      throw new Error('모드 전환이 이미 진행 중입니다.');
    }

    if (!this.isValidMode(targetMode)) {
      throw new Error(`유효하지 않은 모드: ${targetMode}`);
    }

    if (this.currentMode === targetMode) {
      this.logger.info('이미 해당 모드입니다.', { mode: targetMode });
      return this.createModeResponse(targetMode, false);
    }

    this.transitionInProgress = true;
    const previousMode = this.currentMode;

    try {
      this.logger.info('모드 전환 시작', {
        from: previousMode,
        to: targetMode,
        manual: options.manual || false
      });

      // 1. 전환 전 검증
      await this.validateModeTransition(previousMode, targetMode);

      // 2. 이전 모드 정리
      await this.cleanupPreviousMode(previousMode);

      // 3. 새 모드 준비
      await this.prepareNewMode(targetMode);

      // 4. 모드 전환 실행
      await this.executeModeSwitching(targetMode);

      // 5. 전환 후 검증
      await this.validateModeSwitched(targetMode);

      // 6. 히스토리 업데이트
      this.updateModeHistory(previousMode, targetMode);

      this.currentMode = targetMode;
      
      this.logger.info('모드 전환 완료', {
        from: previousMode,
        to: targetMode,
        duration: Date.now() - (options.startTime || Date.now())
      });

      return this.createModeResponse(targetMode, true);

    } catch (error) {
      this.logger.error('모드 전환 실패:', error);
      
      // 롤백 시도
      try {
        await this.rollbackModeSwitch(previousMode);
      } catch (rollbackError) {
        this.logger.error('모드 롤백 실패:', rollbackError);
      }

      throw error;
    } finally {
      this.transitionInProgress = false;
    }
  }

  async validateModeTransition(fromMode, toMode) {
    // 리소스 사용량 확인
    const targetModeConfig = this.modes[toMode];
    const resourceReq = targetModeConfig.resource_usage;

    if (resourceReq) {
      const systemMemory = process.memoryUsage();
      const availableMemory = systemMemory.heapTotal - systemMemory.heapUsed;

      if (resourceReq.memory_limit_mb * 1024 * 1024 > availableMemory * 2) {
        this.logger.warn('메모리 부족 경고', {
          required: resourceReq.memory_limit_mb,
          available: Math.round(availableMemory / 1024 / 1024)
        });
      }

      if (resourceReq.gpu_required && !this.isGPUAvailable()) {
        throw new Error(`모드 ${toMode}은 GPU가 필요하지만 사용할 수 없습니다.`);
      }
    }

    // 모델 가용성 확인
    if (targetModeConfig.active_model === 'processor') {
      const modelAvailable = await this.modelManager.isModelAvailable('qwen2.5:14b');
      if (!modelAvailable) {
        throw new Error('프로세서 모델이 사용할 수 없습니다.');
      }
    }
  }

  async cleanupPreviousMode(previousMode) {
    const previousModeConfig = this.modes[previousMode];
    
    if (previousModeConfig) {
      // 이전 모드의 리소스 정리
      if (previousModeConfig.auto_unload_timeout) {
        // 메모리 정리
        if (global.gc) {
          global.gc();
        }

        // 사용하지 않는 모델 언로드
        if (previousMode === 'ml' && this.modelManager.models.has('processor')) {
          setTimeout(() => {
            this.modelManager.optimizeMemory();
          }, previousModeConfig.auto_unload_timeout);
        }
      }
    }
  }

  async prepareNewMode(targetMode) {
    const targetModeConfig = this.modes[targetMode];
    
    // 필요한 모델 로드
    if (targetModeConfig.active_model === 'processor') {
      await this.modelManager.loadProcessorModel();
    }

    // 필요한 서비스 초기화
    if (targetMode === 'ml') {
      // ML 모드 전용 초기화
      await this.initializeMLServices();
    } else if (targetMode === 'deep_learning') {
      // 딥러닝 모드 전용 초기화
      await this.initializeDeepLearningServices();
    }
  }

  async executeModeSwitching(targetMode) {
    // 컨텍스트 트래커에 모드 변경 알림
    await this.contextTracker.setMode(targetMode);

    // 모드별 설정 적용
    const targetModeConfig = this.modes[targetMode];
    
    // 로깅 레벨 조정
    if (targetModeConfig.logging) {
      process.env.LOG_LEVEL = targetModeConfig.logging.level;
    }

    // 성능 모니터링 설정
    if (targetModeConfig.performance_monitoring) {
      this.enablePerformanceMonitoring();
    }
  }

  async validateModeSwitched(targetMode) {
    // 모드 전환이 성공적으로 완료되었는지 확인
    const currentContextMode = this.contextTracker.getCurrentMode();
    
    if (currentContextMode !== targetMode) {
      throw new Error(`모드 전환 검증 실패: 예상 ${targetMode}, 실제 ${currentContextMode}`);
    }

    // 필요한 서비스들이 정상 동작하는지 확인
    const targetModeConfig = this.modes[targetMode];
    
    if (targetModeConfig.active_model === 'processor') {
      const modelStatus = await this.modelManager.getModelStatus();
      if (!modelStatus.processor || modelStatus.processor.status !== 'loaded') {
        throw new Error('프로세서 모델이 정상적으로 로드되지 않았습니다.');
      }
    }
  }

  updateModeHistory(fromMode, toMode) {
    this.modeHistory.push({
      from: fromMode,
      to: toMode,
      timestamp: new Date().toISOString(),
      success: true
    });

    // 히스토리 크기 제한 (최근 50개)
    if (this.modeHistory.length > 50) {
      this.modeHistory = this.modeHistory.slice(-50);
    }
  }

  async rollbackModeSwitch(previousMode) {
    this.logger.info('모드 롤백 시도', { targetMode: previousMode });
    
    try {
      await this.contextTracker.setMode(previousMode);
      this.currentMode = previousMode;
      
      this.modeHistory.push({
        from: this.currentMode,
        to: previousMode,
        timestamp: new Date().toISOString(),
        success: false,
        rollback: true
      });
      
    } catch (error) {
      this.logger.error('롤백 실패:', error);
      throw error;
    }
  }

  createModeResponse(mode, switched) {
    const modeConfig = this.modes[mode];
    const emoji = this.getModeEmoji(mode);
    
    let responseText = switched
      ? `${emoji} **모드 전환 완료!**\n\n`
      : `${emoji} **현재 모드 상태**\n\n`;
    
    responseText += `**${modeConfig.name}**\n`;
    responseText += `${modeConfig.description}\n\n`;
    
    if (switched) {
      responseText += `### 🎯 활성화된 기능\n`;
      const features = Object.entries(modeConfig.features || {})
        .filter(([key, value]) => value)
        .map(([key]) => this.formatFeatureName(key));
      
      responseText += features.map(feature => `• ${feature}`).join('\n');
      responseText += `\n\n### 🔧 사용 가능한 도구\n`;
      responseText += modeConfig.tools.map(tool => `• ${tool}`).join('\n');
    }

    return {
      content: [{
        type: 'text',
        text: responseText
      }],
      metadata: {
        mode: mode,
        switched: switched,
        available_tools: modeConfig.tools,
        features: modeConfig.features,
        timestamp: new Date().toISOString()
      }
    };
  }

  async analyzeQueryForAutoSwitch(query) {
    if (!this.autoSwitchEnabled) {
      return null;
    }

    const normalizedQuery = query.toLowerCase();
    const autoSwitchConfig = this.modes.mode_switching?.auto_switch;
    
    if (!autoSwitchConfig || !autoSwitchConfig.keywords) {
      return null;
    }

    // 각 모드의 키워드 매칭
    for (const [mode, keywords] of Object.entries(autoSwitchConfig.keywords)) {
      const matchCount = keywords.filter(keyword =>
        normalizedQuery.includes(keyword.toLowerCase())
      ).length;

      const confidence = matchCount / keywords.length;
      
      if (confidence >= (autoSwitchConfig.confidence_threshold || 0.3)) {
        if (mode !== this.currentMode) {
          return {
            suggestedMode: mode,
            confidence: confidence,
            matchedKeywords: keywords.filter(keyword =>
              normalizedQuery.includes(keyword.toLowerCase())
            ),
            reason: `쿼리에서 ${mode} 모드 관련 키워드를 감지했습니다.`
          };
        }
      }
    }

    return null;
  }

  async handleAutoSwitch(autoSwitchResult) {
    try {
      this.logger.info('자동 모드 전환 실행', autoSwitchResult);
      
      const result = await this.switchMode(autoSwitchResult.suggestedMode, {
        manual: false,
        confidence: autoSwitchResult.confidence,
        reason: autoSwitchResult.reason
      });

      return {
        ...result,
        autoSwitched: true,
        suggestion: autoSwitchResult
      };

    } catch (error) {
      this.logger.error('자동 모드 전환 실패:', error);
      
      return {
        content: [{
          type: 'text',
          text: `자동 모드 전환을 시도했지만 실패했습니다. 수동으로 모드를 전환해주세요.\n오류: ${error.message}`
        }],
        autoSwitchFailed: true,
        error: error.message
      };
    }
  }

  // 유틸리티 메서드들
  isValidMode(mode) {
    return this.modes && this.modes.hasOwnProperty(mode);
  }

  getCurrentMode() {
    return this.currentMode;
  }

  getAvailableModes() {
    return Object.keys(this.modes || {});
  }

  getModeConfig(mode) {
    return this.modes[mode] || null;
  }

  getModeEmoji(mode) {
    const emojiMap = {
      general: '🤖',
      ml: '🧠',
      coding: '💻',
      deep_learning: '🚀'
    };
    return emojiMap[mode] || '⚙️';
  }

  formatFeatureName(featureName) {
    return featureName
      .replace(/_/g, ' ')
      .replace(/\b\w/g, char => char.toUpperCase());
  }

  isGPUAvailable() {
    // 간단한 GPU 사용 가능 여부 확인
    // 실제 구현에서는 nvidia-smi 등을 사용할 수 있음
    return process.env.CUDA_VISIBLE_DEVICES !== undefined ||
           process.env.GPU_AVAILABLE === 'true';
  }

  async initializeMLServices() {
    // ML 모드 전용 서비스 초기화
    this.logger.debug('ML 서비스 초기화 중...');
    
    // Python 환경 확인
    // 데이터 로더 초기화
    // 시각화 엔진 준비 등
  }

  async initializeDeepLearningServices() {
    // 딥러닝 모드 전용 서비스 초기화
    this.logger.debug('딥러닝 서비스 초기화 중...');
    
    // GPU 메모리 확인
    // 딥러닝 프레임워크 로드
    // 대용량 모델 준비 등
  }

  enablePerformanceMonitoring() {
    // 성능 모니터링 활성화
    this.logger.debug('성능 모니터링 활성화');
  }

  // 통계 및 상태 정보
  getModeStatistics() {
    const stats = {
      current_mode: this.currentMode,
      transition_in_progress: this.transitionInProgress,
      auto_switch_enabled: this.autoSwitchEnabled,
      mode_history: this.modeHistory.slice(-10), // 최근 10개
      available_modes: this.getAvailableModes()
    };

    return stats;
  }

  getModeUsageAnalytics() {
    const analytics = {
      total_switches: this.modeHistory.length,
      successful_switches: this.modeHistory.filter(h => h.success).length,
      mode_usage: {}
    };

    // 모드별 사용 통계
    for (const mode of this.getAvailableModes()) {
      const usage = this.modeHistory.filter(h => h.to === mode).length;
      analytics.mode_usage[mode] = usage;
    }

    return analytics;
  }

  // 설정 관리
  async updateAutoSwitchSettings(enabled, confidenceThreshold = 0.3) {
    this.autoSwitchEnabled = enabled;
    
    if (this.modes.mode_switching) {
      this.modes.mode_switching.auto_switch.enabled = enabled;
      this.modes.mode_switching.auto_switch.confidence_threshold = confidenceThreshold;
    }

    this.logger.info('자동 전환 설정 업데이트', {
      enabled,
      confidenceThreshold
    });
  }

  // 모드별 헬스체크
  async performModeHealthCheck(mode) {
    const modeConfig = this.modes[mode];
    const healthCheck = {
      mode: mode,
      status: 'unknown',
      checks: {},
      timestamp: new Date().toISOString()
    };

    try {
      // 기본 설정 확인
      healthCheck.checks.configuration = modeConfig ? 'ok' : 'missing';

      // 모델 상태 확인
      if (modeConfig.active_model) {
        const modelStatus = await this.modelManager.getModelStatus();
        healthCheck.checks.model = modelStatus[modeConfig.active_model] ? 'ok' : 'not_loaded';
      }

      // 리소스 요구사항 확인
      if (modeConfig.resource_usage) {
        const memoryUsage = process.memoryUsage();
        const memoryOk = memoryUsage.heapUsed < modeConfig.resource_usage.memory_limit_mb * 1024 * 1024;
        healthCheck.checks.memory = memoryOk ? 'ok' : 'insufficient';

        if (modeConfig.resource_usage.gpu_required) {
          healthCheck.checks.gpu = this.isGPUAvailable() ? 'ok' : 'not_available';
        }
      }

      // 전체 상태 결정
      const allChecks = Object.values(healthCheck.checks);
      if (allChecks.every(check => check === 'ok')) {
        healthCheck.status = 'healthy';
      } else if (allChecks.some(check => check === 'ok')) {
        healthCheck.status = 'degraded';
      } else {
        healthCheck.status = 'unhealthy';
      }

    } catch (error) {
      healthCheck.status = 'error';
      healthCheck.error = error.message;
    }

    return healthCheck;
  }
}
