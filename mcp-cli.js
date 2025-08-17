#!/usr/bin/env node

// mcp-cli.js - ML MCP 시스템 CLI 인터페이스
import readline from 'readline';
import chalk from 'chalk';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { spawn } from 'child_process';
import fs from 'fs/promises';
import path from 'path';

class MCPCLIClient {
  constructor() {
    this.client = null;
    this.transport = null;
    this.serverProcess = null;
    this.isConnected = false;
    this.currentMode = 'general';
    this.conversationHistory = [];
    this.availableTools = [];
    this.maxHistorySize = 50;
    
    // CLI 설정
    this.rl = null;
    this.isRunning = false;
    
    // 연결 설정
    this.serverPath = './main.js';
    this.connectionTimeout = 30000; // 30초
    this.maxRetries = 3;
  }

  async initialize() {
    try {
      console.log(chalk.cyan('🚀 ML MCP CLI 시스템 초기화 중...'));
      
      // MCP 서버 연결
      await this.connectToServer();
      
      // 사용 가능한 도구 로드
      await this.loadAvailableTools();
      
      // CLI 인터페이스 설정
      this.setupReadlineInterface();
      
      console.log(chalk.green('✅ 초기화 완료!\n'));
      
      return true;
    } catch (error) {
      console.error(chalk.red('❌ 초기화 실패:'), error.message);
      throw error;
    }
  }

  async connectToServer() {
    console.log(chalk.yellow('🔌 MCP 서버에 연결 중...'));
    
    let retries = 0;
    while (retries < this.maxRetries) {
      try {
        // 서버 프로세스 시작
        this.serverProcess = spawn('node', [this.serverPath], {
          stdio: ['pipe', 'pipe', 'pipe'],
          cwd: process.cwd()
        });

        // 서버 오류 처리
        this.serverProcess.stderr.on('data', (data) => {
          console.error(chalk.red('서버 오류:'), data.toString());
        });

        this.serverProcess.on('error', (error) => {
          console.error(chalk.red('서버 프로세스 오류:'), error);
        });

        // Transport 및 Client 설정
        this.transport = new StdioClientTransport(
          this.serverProcess.stdout,
          this.serverProcess.stdin
        );

        this.client = new Client(
          {
            name: 'ml-mcp-cli',
            version: '1.0.0'
          },
          {
            capabilities: {}
          }
        );

        // 연결 시도
        await Promise.race([
          this.client.connect(this.transport),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('연결 시간 초과')), this.connectionTimeout)
          )
        ]);

        this.isConnected = true;
        console.log(chalk.green('✅ MCP 서버 연결 성공'));
        return;

      } catch (error) {
        retries++;
        console.log(chalk.yellow(`⚠️ 연결 실패 (${retries}/${this.maxRetries}): ${error.message}`));
        
        if (this.serverProcess) {
          this.serverProcess.kill();
          this.serverProcess = null;
        }
        
        if (retries < this.maxRetries) {
          console.log(chalk.yellow(`🔄 ${2}초 후 재시도...`));
          await new Promise(resolve => setTimeout(resolve, 2000));
        }
      }
    }
    
    throw new Error('MCP 서버 연결에 실패했습니다.');
  }

  async loadAvailableTools() {
    try {
      console.log(chalk.yellow('🛠️ 사용 가능한 도구 로드 중...'));
      
      const response = await this.client.listTools();
      this.availableTools = response.tools || [];
      
      console.log(chalk.green(`✅ ${this.availableTools.length}개 도구 로드 완료`));
    } catch (error) {
      console.warn(chalk.yellow('⚠️ 도구 목록 로드 실패:', error.message));
      this.availableTools = [];
    }
  }

  setupReadlineInterface() {
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      prompt: this.getPrompt(),
      historySize: 100
    });

    // 자동완성 설정
    this.rl.setPrompt(this.getPrompt());
    
    // Ctrl+C 핸들러
    this.rl.on('SIGINT', () => {
      console.log(chalk.yellow('\n\n👋 안녕히 가세요!'));
      this.cleanup();
    });

    // 입력 핸들러
    this.rl.on('line', async (input) => {
      await this.handleUserInput(input.trim());
    });

    // 종료 핸들러
    this.rl.on('close', () => {
      this.cleanup();
    });
  }

  getPrompt() {
    const modeColors = {
      general: chalk.blue,
      ml: chalk.magenta,
      data_analysis: chalk.green,
      visualization: chalk.cyan
    };
    
    const coloredMode = modeColors[this.currentMode] || chalk.blue;
    return coloredMode(`ML[${this.currentMode}]> `);
  }

  async run() {
    try {
      await this.initialize();
      
      this.isRunning = true;
      this.showWelcomeMessage();
      this.rl.prompt();
      
      // 프로세스 종료 핸들러
      process.on('SIGTERM', () => this.cleanup());
      process.on('exit', () => this.cleanup());
      
    } catch (error) {
      console.error(chalk.red('❌ CLI 실행 실패:'), error.message);
      process.exit(1);
    }
  }

  showWelcomeMessage() {
    console.log(chalk.cyan('┌─────────────────────────────────────────────────────────┐'));
    console.log(chalk.cyan('│                   🧠 ML MCP 시스템                      │'));
    console.log(chalk.cyan('│              동적 AI 분석 플랫폼에 오신 것을 환영합니다!   │'));
    console.log(chalk.cyan('└─────────────────────────────────────────────────────────┘'));
    console.log();
    console.log(chalk.green('🎯 주요 기능:'));
    console.log(chalk.white('   • 자동 모듈 발견 및 실행'));
    console.log(chalk.white('   • 자연어 분석 요청'));
    console.log(chalk.white('   • 실시간 데이터 처리'));
    console.log(chalk.white('   • 동적 시각화 생성'));
    console.log();
    console.log(chalk.yellow('💡 사용 예시:'));
    console.log(chalk.gray('   • "상관관계 분석해줘"'));
    console.log(chalk.gray('   • "이 데이터로 클러스터링 해줘"'));
    console.log(chalk.gray('   • "차트 그려줘"'));
    console.log(chalk.gray('   • "모듈 검색 regression"'));
    console.log();
    console.log(chalk.cyan('📚 도움말: "도움말" 또는 "help" 입력'));
    console.log(chalk.cyan('🚪 종료: "exit", "quit" 또는 Ctrl+C'));
    console.log(chalk.gray('─'.repeat(60)));
    console.log();
  }

  async handleUserInput(input) {
    if (!input) {
      this.rl.prompt();
      return;
    }

    // 특수 명령어 처리
    if (this.handleSpecialCommands(input)) {
      this.rl.prompt();
      return;
    }

    try {
      // 사용자 입력 분석 및 도구 호출
      const toolCall = this.analyzeUserInput(input);
      
      if (!toolCall) {
        console.log(chalk.red('❌ 요청을 이해할 수 없습니다. 다시 시도해주세요.'));
        console.log(chalk.gray('💡 "도움말"을 입력하면 사용 가능한 명령어를 확인할 수 있습니다.'));
        this.rl.prompt();
        return;
      }
      
      console.log(chalk.yellow(`\n🔄 처리 중... (${toolCall.name})`));
      
      // MCP 서버에 도구 호출 요청
      const result = await this.callMCPTool(toolCall);
      
      // 결과 표시
      await this.displayResult(result);
      
      // 대화 기록에 추가
      this.addToHistory(input, result);
      
    } catch (error) {
      console.error(chalk.red('❌ 처리 중 오류 발생:'), error.message);
      
      // 연결 오류인 경우 재연결 시도
      if (error.message.includes('연결') || error.message.includes('transport')) {
        console.log(chalk.yellow('🔄 서버 재연결 시도 중...'));
        try {
          await this.reconnectToServer();
          console.log(chalk.green('✅ 재연결 성공'));
        } catch (reconnectError) {
          console.error(chalk.red('❌ 재연결 실패:'), reconnectError.message);
        }
      }
    }
    
    this.rl.prompt();
  }

  handleSpecialCommands(input) {
    const lowerInput = input.toLowerCase();
    
    // 종료 명령어
    if (['exit', 'quit', 'q', '종료', '나가기'].includes(lowerInput)) {
      console.log(chalk.yellow('\n👋 안녕히 가세요!'));
      this.cleanup();
      return true;
    }
    
    // 도움말
    if (['help', 'h', '도움말', '도움', 'ㅗ디ㅔ'].includes(lowerInput)) {
      this.showHelpMessage();
      return true;
    }
    
    // 도구 목록
    if (['tools', 'list', '도구', '목록', '기능'].includes(lowerInput)) {
      this.showAvailableTools();
      return true;
    }
    
    // 히스토리
    if (['history', 'hist', '히스토리', '기록'].includes(lowerInput)) {
      this.showConversationHistory();
      return true;
    }
    
    // 클리어
    if (['clear', 'cls', '지우기', '청소'].includes(lowerInput)) {
      console.clear();
      this.showWelcomeMessage();
      return true;
    }
    
    // 상태 확인
    if (['status', 'stat', '상태', '현황'].includes(lowerInput)) {
      this.showSystemStatus();
      return true;
    }
    
    return false;
  }

  analyzeUserInput(userInput) {
    try {
      const input = userInput.toLowerCase().trim();
      
      // 1. 동적 분석 요청 감지 (최우선)
      if (this.isDynamicAnalysisRequest(input, userInput)) {
        return {
          name: 'dynamic_analysis',
          arguments: {
            query: userInput,
            options: {
              auto_detect_files: true
            }
          }
        };
      }

      // 2. 모듈 관리 명령어들
      const moduleCommand = this.parseModuleCommand(input, userInput);
      if (moduleCommand) {
        return moduleCommand;
      }

      // 3. 시스템 관리 명령어들  
      const systemCommand = this.parseSystemCommand(input);
      if (systemCommand) {
        return systemCommand;
      }

      // 4. 기존 특정 분석 요청들
      const specificCommand = this.parseSpecificCommand(input, userInput);
      if (specificCommand) {
        return specificCommand;
      }

      // 5. 기본 폴백 - 동적 분석 시도
      return {
        name: 'dynamic_analysis',
        arguments: {
          query: userInput,
          options: {
            auto_detect_files: true
          }
        }
      };
      
    } catch (error) {
      console.error('입력 분석 실패:', error);
      return null;
    }
  }

  isDynamicAnalysisRequest(input, originalInput) {
    // 명시적 분석 키워드들
    const analysisKeywords = [
      // 통계 분석
      '통계', 'stats', 'statistics', '기술통계', 'descriptive',
      '상관관계', 'correlation', '상관', '연관성', '관계',
      '분포', 'distribution', '히스토그램', 'histogram',
      '빈도', 'frequency', '빈도수',
      
      // 머신러닝
      '회귀', 'regression', '선형회귀', 'linear',
      '분류', 'classification', '분류기', 'classifier',
      '클러스터', 'cluster', '군집', '그룹핑', 'clustering',
      'pca', '주성분', '차원축소', 'dimensionality',
      '이상치', 'outlier', '특이값', 'anomaly',
      '예측', 'prediction', 'predict', 'forecast',
      
      // 시각화
      '시각화', 'visualization', 'visualize',
      '차트', 'chart', '그래프', 'graph', 'plot',
      '히트맵', 'heatmap', '산점도', 'scatter',
      '막대그래프', 'bar', '선그래프', 'line',
      
      // 시계열
      '시계열', 'timeseries', '시간', 'temporal',
      '트렌드', 'trend', '계절성', 'seasonal',
      
      // 전처리
      '전처리', 'preprocessing', '정제', 'cleaning',
      '변환', 'transform', '정규화', 'normalize',
      
      // 기타 분석
      '감정분석', 'sentiment', '텍스트분석', 'text',
      '네트워크분석', 'network', '그래프분석'
    ];

    // 분석 동사들
    const analysisVerbs = [
      '분석', 'analyze', 'analysis',
      '실행', 'execute', 'run',
      '수행', 'perform', '진행',
      '계산', 'calculate', 'compute',
      '처리', 'process', '생성', 'generate',
      '만들', 'create', '그려', 'draw'
    ];

    // 키워드 + 동사 조합 확인
    const hasKeyword = analysisKeywords.some(keyword => input.includes(keyword));
    const hasVerb = analysisVerbs.some(verb => input.includes(verb));
    
    // 파일 확장자 언급
    const hasDataFile = /\.(csv|xlsx|json|parquet|h5|hdf5|txt|data)/.test(input);
    
    // "데이터" 관련 용어
    const hasDataTerms = ['데이터', 'data', '파일', 'file'].some(term => input.includes(term));

    // 분석 요청 패턴들
    const analysisPatterns = [
      /(.+)\s*(분석|analysis|analyze)/,
      /(.+)\s*(해줘|해주세요|실행|run)/,
      /(.+)\s*(그려|그리기|plot|chart)/,
      /(어떤|what|how)\s*(.+)/,
      /(찾아|find|search)\s*(.+)/,
    ];

    const matchesPattern = analysisPatterns.some(pattern => pattern.test(input));

    // 조건 조합으로 판단
    return (hasKeyword && hasVerb) || 
           (hasKeyword && hasDataTerms) ||
           (hasDataFile && hasVerb) ||
           matchesPattern;
  }

  parseModuleCommand(input, originalInput) {
    // 모듈 검색
    if (input.includes('모듈') && (input.includes('검색') || input.includes('찾') || input.includes('search'))) {
      const query = this.extractSearchQuery(originalInput);
      return {
        name: 'search_modules',
        arguments: {
          query: query,
          limit: 10
        }
      };
    }

    // 모듈 새로고침
    if (input.includes('모듈') && (input.includes('새로고침') || input.includes('refresh') || input.includes('스캔'))) {
      return {
        name: 'refresh_modules',
        arguments: {}
      };
    }

    // 모듈 통계
    if (input.includes('모듈') && (input.includes('통계') || input.includes('현황') || input.includes('stats'))) {
      return {
        name: 'module_stats',
        arguments: {}
      };
    }

    // 모듈 테스트
    if (input.includes('모듈') && (input.includes('테스트') || input.includes('test'))) {
      const moduleId = this.extractModuleId(originalInput);
      if (moduleId) {
        return {
          name: 'test_module',
          arguments: {
            moduleId: moduleId
          }
        };
      }
    }

    // 모듈 상세 정보
    if (input.includes('모듈') && (input.includes('정보') || input.includes('상세') || input.includes('details'))) {
      const moduleId = this.extractModuleId(originalInput);
      if (moduleId) {
        return {
          name: 'module_details',
          arguments: {
            moduleId: moduleId
          }
        };
      }
    }

    return null;
  }

  parseSystemCommand(input) {
    // 시스템 상태
    if (input.includes('시스템') && (input.includes('상태') || input.includes('status'))) {
      return {
        name: 'system_status',
        arguments: {}
      };
    }

    // 모드 전환
    if (input.includes('모드')) {
      let mode = 'general';
      if (input.includes('ml') || input.includes('머신러닝')) {
        mode = 'ml';
      } else if (input.includes('분석')) {
        mode = 'data_analysis';
      } else if (input.includes('시각화')) {
        mode = 'visualization';
      }
      
      return {
        name: 'mode_switch',
        arguments: {
          mode: mode
        }
      };
    }

    return null;
  }

  parseSpecificCommand(input, originalInput) {
    // 데이터 분석 요청 (구체적)
    if (input.includes('분석') && !this.isDynamicAnalysisRequest(input, originalInput)) {
      return {
        name: 'analyze_data',
        arguments: {
          query: originalInput,
          auto_detect_files: true
        }
      };
    }
    
    // 시각화 요청 (구체적)
    if ((input.includes('시각화') || input.includes('차트') || input.includes('그래프')) && 
        !this.isDynamicAnalysisRequest(input, originalInput)) {
      return {
        name: 'visualize_data',
        arguments: {
          query: originalInput,
          auto_detect_files: true
        }
      };
    }
    
    // 모델 훈련 요청 (구체적)
    if ((input.includes('모델') && input.includes('훈련')) || 
        (input.includes('학습')) && !this.isDynamicAnalysisRequest(input, originalInput)) {
      return {
        name: 'train_model',
        arguments: {
          query: originalInput,
          auto_detect_files: true
        }
      };
    }

    return null;
  }

  extractSearchQuery(userInput) {
    const patterns = [
      /모듈\s*검색\s*(.+)/i,
      /search\s*modules?\s*(.+)/i,
      /찾.*모듈\s*(.+)/i,
      /모듈.*찾.*\s*(.+)/i
    ];
    
    for (const pattern of patterns) {
      const match = userInput.match(pattern);
      if (match) {
        return match[1].trim();
      }
    }
    
    return '';
  }

  extractModuleId(userInput) {
    const patterns = [
      /모듈\s*(?:테스트|정보|상세)\s*([a-zA-Z_.]+)/i,
      /(?:test|details)\s*module\s*([a-zA-Z_.]+)/i,
      /([a-zA-Z_.]+)\s*모듈/i
    ];
    
    for (const pattern of patterns) {
      const match = userInput.match(pattern);
      if (match) {
        return match[1];
      }
    }
    
    return '';
  }

  async callMCPTool(toolCall) {
    try {
      if (!this.isConnected) {
        throw new Error('MCP 서버에 연결되지 않았습니다.');
      }
      
      const result = await this.client.callTool({
        name: toolCall.name,
        arguments: toolCall.arguments
      });

      return result;
      
    } catch (error) {
      throw new Error(`MCP 도구 호출 실패: ${error.message}`);
    }
  }

  async displayResult(result) {
    if (!result) {
      console.log(chalk.gray('결과를 표시할 수 없습니다.'));
      return;
    }

    if (result.error) {
      console.log(chalk.red('❌ 오류:'), result.error);
      return;
    }

    if (result.content && result.content[0]) {
      const content = result.content[0].text;
      
      // 결과 유형에 따른 포맷팅
      if (content.includes('🎯 동적 분석 완료')) {
        console.log(chalk.green('\n✨ 동적 분석 결과:'));
        console.log(this.formatDynamicResult(content));
      } else if (content.includes('🔍 **사용 가능한 분석 모듈**')) {
        console.log(chalk.cyan('\n📚 모듈 검색 결과:'));
        console.log(this.formatModuleList(content));
      } else if (content.includes('🔄 **모듈 새로고침 완료**')) {
        console.log(chalk.blue('\n🔄 모듈 새로고침:'));
        console.log(this.formatRefreshResult(content));
      } else if (content.includes('📊 **모듈 시스템 통계**')) {
        console.log(chalk.magenta('\n📊 시스템 통계:'));
        console.log(this.formatStatsResult(content));
      } else {
        // 기본 표시
        console.log(chalk.white('\n' + content));
      }
    }
    
    console.log(); // 빈 줄 추가
  }

  formatDynamicResult(content) {
    return content
      .replace(/\*\*(.*?)\*\*/g, chalk.bold('$1'))
      .replace(/🎯/g, chalk.green('🎯'))
      .replace(/📦/g, chalk.blue('📦'))
      .replace(/⏱️/g, chalk.yellow('⏱️'))
      .replace(/📊/g, chalk.cyan('📊'))
      .replace(/✅/g, chalk.green('✅'));
  }

  formatModuleList(content) {
    return content
      .replace(/\*\*(.*?)\*\*/g, chalk.cyan.bold('$1'))
      .replace(/📂/g, chalk.yellow('📂'))
      .replace(/📄/g, chalk.gray('📄'))
      .replace(/🏷️/g, chalk.magenta('🏷️'))
      .replace(/✅/g, chalk.green('✅'))
      .replace(/❌/g, chalk.red('❌'))
      .replace(/🆔/g, chalk.gray('🆔'));
  }

  formatRefreshResult(content) {
    return content
      .replace(/\*\*(.*?)\*\*/g, chalk.blue.bold('$1'))
      .replace(/📊/g, chalk.cyan('📊'))
      .replace(/📂/g, chalk.yellow('📂'))
      .replace(/🔄/g, chalk.blue('🔄'))
      .replace(/⚠️/g, chalk.yellow('⚠️'));
  }

  formatStatsResult(content) {
    return content
      .replace(/\*\*(.*?)\*\*/g, chalk.magenta.bold('$1'))
      .replace(/📈/g, chalk.green('📈'))
      .replace(/📂/g, chalk.yellow('📂'))
      .replace(/🔧/g, chalk.blue('🔧'))
      .replace(/⚡/g, chalk.yellow('⚡'))
      .replace(/📋/g, chalk.cyan('📋'))
      .replace(/🏥/g, chalk.red('🏥'));
  }

  showHelpMessage() {
    console.log(chalk.cyan('\n📚 ML MCP 시스템 도움말\n'));
    
    console.log(chalk.yellow('🎯 기본 사용법:'));
    console.log(chalk.white('   • 자연어로 분석 요청: "상관관계 분석해줘", "데이터 시각화해줘"'));
    console.log(chalk.white('   • 파일 지정: "sales.csv 파일로 회귀분석해줘"'));
    console.log(chalk.white('   • 모듈 검색: "모듈 검색 clustering"'));
    console.log();
    
    console.log(chalk.yellow('🔧 모듈 관리 명령어:'));
    console.log(chalk.gray('   • 모듈 검색 [키워드]     - 관련 모듈 찾기'));
    console.log(chalk.gray('   • 모듈 새로고침          - 새 모듈 스캔'));
    console.log(chalk.gray('   • 모듈 통계             - 모듈 현황 확인'));
    console.log(chalk.gray('   • 모듈 테스트 [ID]       - 모듈 실행 테스트'));
    console.log(chalk.gray('   • 모듈 정보 [ID]        - 모듈 상세 정보'));
    console.log();
    
    console.log(chalk.yellow('⚙️ 시스템 명령어:'));
    console.log(chalk.gray('   • 상태                  - 시스템 상태 확인'));
    console.log(chalk.gray('   • 도구                  - 사용 가능한 도구 목록'));
    console.log(chalk.gray('   • 기록                  - 대화 기록 보기'));
    console.log(chalk.gray('   • 지우기                - 화면 지우기'));
    console.log();
    
    console.log(chalk.yellow('📝 분석 예시:'));
    console.log(chalk.gray('   • "이 데이터의 상관관계를 분석해줘"'));
    console.log(chalk.gray('   • "클러스터링으로 그룹을 나눠줘"'));
    console.log(chalk.gray('   • "히스토그램 차트 그려줘"'));
    console.log(chalk.gray('   • "이상치를 탐지해줘"'));
    console.log(chalk.gray('   • "시계열 예측 모델 만들어줘"'));
    console.log();
    
    console.log(chalk.yellow('🚪 종료:'));
    console.log(chalk.gray('   • exit, quit, 종료 또는 Ctrl+C'));
    
    console.log(chalk.gray('─'.repeat(60)));
  }

  showAvailableTools() {
    console.log(chalk.cyan('\n🛠️ 사용 가능한 도구 목록\n'));
    
    if (this.availableTools.length === 0) {
      console.log(chalk.yellow('도구 정보를 불러오는 중...'));
      return;
    }

    // 도구를 카테고리별로 분류
    const categories = {
      '동적 분석': ['dynamic_analysis', 'search_modules', 'refresh_modules'],
      '모듈 관리': ['module_stats', 'test_module', 'module_details', 'validate_modules'],
      '데이터 분석': ['analyze_data', 'visualize_data', 'train_model'],
      '시스템': ['system_status', 'mode_switch', 'general_query']
    };

    for (const [category, toolNames] of Object.entries(categories)) {
      console.log(chalk.yellow(`📁 ${category}:`));
      
      toolNames.forEach(toolName => {
        const tool = this.availableTools.find(t => t.name === toolName);
        if (tool) {
          console.log(chalk.white(`   • ${tool.name}`));
          console.log(chalk.gray(`     ${tool.description}`));
        }
      });
      console.log();
    }
    
    console.log(chalk.gray('─'.repeat(50)));
  }

  showConversationHistory() {
    console.log(chalk.cyan('\n📜 대화 히스토리:\n'));
    
    if (this.conversationHistory.length === 0) {
      console.log(chalk.yellow('  아직 대화 기록이 없습니다.'));
      return;
    }

    const recentHistory = this.conversationHistory.slice(-10);
    recentHistory.forEach((entry, index) => {
      const time = new Date(entry.timestamp).toLocaleTimeString();
      console.log(chalk.blue(`[${time}] 사용자: `) + chalk.white(entry.input));
      
      if (entry.output) {
        const preview = this.extractPreview(entry.output);
        console.log(chalk.green(`[${time}] 시스템: `) + chalk.gray(preview));
      }
      
      console.log();
    });
    
    console.log(chalk.gray('─'.repeat(50)));
  }

  extractPreview(output) {
    if (typeof output === 'string') {
      return output.substring(0, 100) + (output.length > 100 ? '...' : '');
    }
    
    if (output.content && output.content[0] && output.content[0].text) {
      const text = output.content[0].text;
      return text.substring(0, 100) + (text.length > 100 ? '...' : '');
    }
    
    return 'Response received';
  }

  showSystemStatus() {
    console.log(chalk.cyan('\n🔍 시스템 상태\n'));
    
    console.log(chalk.yellow('📡 연결 상태:'));
    console.log(`   • MCP 서버: ${this.isConnected ? chalk.green('✅ 연결됨') : chalk.red('❌ 연결 끊김')}`);
    console.log(`   • 현재 모드: ${chalk.blue(this.currentMode)}`);
    console.log(`   • 사용 가능한 도구: ${chalk.cyan(this.availableTools.length)}개`);
    console.log();
    
    console.log(chalk.yellow('📊 사용 통계:'));
    console.log(`   • 대화 기록: ${chalk.cyan(this.conversationHistory.length)}개`);
    console.log(`   • 실행 중: ${this.isRunning ? chalk.green('Yes') : chalk.red('No')}`);
    console.log();
    
    if (this.serverProcess) {
      console.log(chalk.yellow('🖥️ 서버 프로세스:'));
      console.log(`   • PID: ${chalk.cyan(this.serverProcess.pid)}`);
      console.log(`   • 상태: ${this.serverProcess.killed ? chalk.red('종료됨') : chalk.green('실행 중')}`);
    }
    
    console.log(chalk.gray('─'.repeat(50)));
  }

  addToHistory(input, output) {
    this.conversationHistory.push({
      timestamp: new Date().toISOString(),
      input: input,
      output: output
    });

    // 히스토리 크기 제한
    if (this.conversationHistory.length > this.maxHistorySize) {
      this.conversationHistory = this.conversationHistory.slice(-this.maxHistorySize);
    }
  }

  async reconnectToServer() {
    console.log(chalk.yellow('🔄 MCP 서버 재연결 시도 중...'));
    
    // 기존 연결 정리
    if (this.serverProcess) {
      this.serverProcess.kill();
      this.serverProcess = null;
    }
    
    this.isConnected = false;
    
    // 재연결 시도
    await this.connectToServer();
    await this.loadAvailableTools();
    
    console.log(chalk.green('✅ 재연결 성공'));
  }

  cleanup() {
    try {
      console.log(chalk.yellow('\n🧹 정리 작업 중...'));
      
      this.isRunning = false;
      
      // Readline 인터페이스 정리
      if (this.rl) {
        this.rl.close();
        this.rl = null;
      }
      
      // MCP 클라이언트 연결 종료
      if (this.client && this.isConnected) {
        try {
          this.client.close();
        } catch (error) {
          // 조용히 무시
        }
        this.client = null;
        this.isConnected = false;
      }
      
      // 서버 프로세스 종료
      if (this.serverProcess && !this.serverProcess.killed) {
        this.serverProcess.kill('SIGTERM');
        
        // 강제 종료 타이머
        setTimeout(() => {
          if (this.serverProcess && !this.serverProcess.killed) {
            this.serverProcess.kill('SIGKILL');
          }
        }, 5000);
        
        this.serverProcess = null;
      }
      
      console.log(chalk.green('✅ 정리 완료'));
      
    } catch (error) {
      console.error(chalk.red('❌ 정리 중 오류:'), error.message);
    } finally {
      process.exit(0);
    }
  }

  // 유틸리티 메서드들
  
  setMode(mode) {
    this.currentMode = mode;
    this.rl?.setPrompt(this.getPrompt());
    console.log(chalk.green(`✅ 모드가 ${mode}로 변경되었습니다.`));
  }

  async checkServerHealth() {
    try {
      if (!this.isConnected) {
        return false;
      }
      
      // 간단한 도구 호출로 서버 상태 확인
      await this.client.listTools();
      return true;
      
    } catch (error) {
      console.warn(chalk.yellow('⚠️ 서버 상태 확인 실패:', error.message));
      return false;
    }
  }

  async autoSaveHistory() {
    try {
      const historyFile = path.join(process.cwd(), '.mcp_history.json');
      const historyData = {
        timestamp: new Date().toISOString(),
        history: this.conversationHistory
      };
      
      await fs.writeFile(historyFile, JSON.stringify(historyData, null, 2));
    } catch (error) {
      // 조용히 무시 (선택적 기능)
    }
  }

  async loadHistory() {
    try {
      const historyFile = path.join(process.cwd(), '.mcp_history.json');
      const data = await fs.readFile(historyFile, 'utf-8');
      const historyData = JSON.parse(data);
      
      if (historyData.history && Array.isArray(historyData.history)) {
        this.conversationHistory = historyData.history.slice(-this.maxHistorySize);
        console.log(chalk.gray(`📜 이전 대화 기록 ${this.conversationHistory.length}개를 불러왔습니다.`));
      }
    } catch (error) {
      // 파일이 없거나 읽기 실패 시 조용히 무시
    }
  }

  showProgress(message) {
    process.stdout.write(chalk.yellow(`${message}...`));
  }

  hideProgress() {
    process.stdout.write('\r\x1b[K'); // 현재 줄 지우기
  }

  formatFileSize(bytes) {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(1)}${units[unitIndex]}`;
  }

  truncateText(text, maxLength = 100) {
    if (text.length <= maxLength) {
      return text;
    }
    return text.substring(0, maxLength - 3) + '...';
  }
}

// 메인 실행 함수
async function main() {
  const cli = new MCPCLIClient();
  
  try {
    // 이전 히스토리 로드
    await cli.loadHistory();
    
    // CLI 실행
    await cli.run();
    
  } catch (error) {
    console.error(chalk.red('❌ CLI 실행 실패:'), error.message);
    console.log(chalk.yellow('\n🔧 문제 해결 방법:'));
    console.log(chalk.gray('   1. Ollama 서비스가 실행 중인지 확인하세요: ollama serve'));
    console.log(chalk.gray('   2. 필요한 모델이 설치되어 있는지 확인하세요'));
    console.log(chalk.gray('   3. Node.js 의존성이 설치되어 있는지 확인하세요: npm install'));
    console.log(chalk.gray('   4. Python 환경이 올바르게 설정되어 있는지 확인하세요'));
    process.exit(1);
  }
}

// 에러 핸들링
process.on('unhandledRejection', (reason, promise) => {
  console.error(chalk.red('Unhandled Rejection:'), reason);
});

process.on('uncaughtException', (error) => {
  console.error(chalk.red('Uncaught Exception:'), error);
  process.exit(1);
});

// 스크립트가 직접 실행될 때만 메인 함수 호출
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    console.error(chalk.red('Fatal error:'), error);
    process.exit(1);
  });
}

export { MCPCLIClient };