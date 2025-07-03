#!/usr/bin/env node

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import readline from 'readline';
import chalk from 'chalk';
import { fileURLToPath } from 'url';
import path from 'path';
import fs from 'fs/promises';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class MCPCLIClient {
  constructor() {
    this.client = null;
    this.transport = null;
    this.isConnected = false;
    this.availableTools = [];
    this.conversationHistory = [];
    
    // Readline 인터페이스 설정
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      prompt: chalk.blue('ML> ')
    });
  }

  async initialize() {
    try {
      console.log(chalk.cyan('🔬 MCP ML CLI 시작 중...'));
      
      // 필요한 디렉토리 생성
      await this.createDirectories();
      
      // StdioClientTransport를 사용하여 직접 서버 시작 및 연결
      await this.connectToServer();
      
      // 사용 가능한 도구 목록 가져오기
      await this.loadAvailableTools();
      
      console.log(chalk.green('✅ MCP 클라이언트 초기화 완료!'));
      this.showWelcomeMessage();
      
      // 대화 시작
      this.startConversation();
      
    } catch (error) {
      console.error(chalk.red('❌ 초기화 실패:'), error.message);
      console.error(chalk.red('스택 트레이스:'), error.stack);
      await this.cleanup();
      process.exit(1);
    }
  }

  async createDirectories() {
    const directories = [
      './results',
      './uploads',
      './temp',
      './logs',
      './data',
      './data/state',
      './data/cache',
      './data/logs'
    ];

    for (const dir of directories) {
      try {
        await fs.mkdir(dir, { recursive: true });
      } catch (error) {
        if (error.code !== 'EEXIST') {
          throw error;
        }
      }
    }
  }

  async connectToServer() {
    console.log(chalk.yellow('🔗 MCP 서버 시작 및 연결 중...'));
    
    try {
      const serverPath = path.join(__dirname, 'main.js');
      console.log(chalk.gray(`서버 경로: ${serverPath}`));
      
      // StdioClientTransport를 사용하여 서버 시작 및 연결
      this.transport = new StdioClientTransport({
        command: 'node',
        args: [serverPath],
        env: { ...process.env, NODE_ENV: 'production' }
      });

      // MCP 클라이언트 생성
      this.client = new Client(
        {
          name: 'ml-mcp-cli',
          version: '1.0.0'
        },
        {
          capabilities: {}
        }
      );

      // 서버에 연결
      await this.client.connect(this.transport);
      this.isConnected = true;
      
      console.log(chalk.green('✅ MCP 서버 연결 완료'));
      
    } catch (error) {
      console.error(chalk.red('연결 실패 상세:'), error.message);
      throw new Error(`MCP 서버 연결 실패: ${error.message}`);
    }
  }

  async loadAvailableTools() {
    try {
      const response = await this.client.listTools();
      this.availableTools = response.tools || [];
      console.log(chalk.cyan(`📋 사용 가능한 도구: ${this.availableTools.length}개`));
      
    } catch (error) {
      console.warn(chalk.yellow('⚠️ 도구 목록 로드 실패:'), error.message);
      this.availableTools = [];
    }
  }

  showWelcomeMessage() {
    console.log(chalk.cyan('\n🤖 ML 분석 도우미에 오신 것을 환영합니다!'));
    console.log(chalk.gray('자연어로 명령을 입력하면 AI가 이해하고 실행합니다.'));
    console.log(chalk.gray('도움말을 보려면 "도움말" 또는 "help"를 입력하세요.'));
    console.log(chalk.gray('종료하려면 "종료" 또는 "exit"를 입력하세요.'));
    console.log(chalk.gray('─'.repeat(50)));
  }

  startConversation() {
    this.rl.on('line', async (input) => {
      const userInput = input.trim();

      if (userInput === '') {
        this.rl.prompt();
        return;
      }

      // 종료 명령 처리
      if (this.isExitCommand(userInput)) {
        console.log(chalk.green('👋 안녕히 가세요!'));
        await this.cleanup();
        process.exit(0);
      }

      // 도움말 명령 처리
      if (this.isHelpCommand(userInput)) {
        this.showHelp();
        this.rl.prompt();
        return;
      }

      // 도구 목록 명령 처리
      if (this.isToolsListCommand(userInput)) {
        this.showAvailableTools();
        this.rl.prompt();
        return;
      }

      // 히스토리 명령 처리
      if (this.isHistoryCommand(userInput)) {
        this.showConversationHistory();
        this.rl.prompt();
        return;
      }

      // 사용자 입력 처리
      await this.processUserInput(userInput);
      this.rl.prompt();
    });

    this.rl.on('close', async () => {
      console.log(chalk.yellow('\n프로그램을 종료합니다...'));
      await this.cleanup();
      process.exit(0);
    });

    // 첫 번째 프롬프트 표시
    this.rl.prompt();
  }

  isExitCommand(input) {
    const exitCommands = ['exit', 'quit', 'bye', '종료', '나가기', '그만'];
    return exitCommands.includes(input.toLowerCase());
  }

  isHelpCommand(input) {
    const helpCommands = ['help', 'h', '도움말', '도움', 'usage'];
    return helpCommands.includes(input.toLowerCase());
  }

  isToolsListCommand(input) {
    const toolsCommands = ['tools', 'list', '도구', '도구 목록', '기능'];
    return toolsCommands.includes(input.toLowerCase());
  }

  isHistoryCommand(input) {
    const historyCommands = ['history', 'hist', '히스토리', '기록'];
    return historyCommands.includes(input.toLowerCase());
  }

  showHelp() {
    console.log(chalk.cyan('\n📖 사용 가능한 명령어:'));
    console.log(chalk.white('  📊 데이터 분석: ') + chalk.gray('"data.csv 파일을 분석해줘"'));
    console.log(chalk.white('  🤖 모델 훈련: ') + chalk.gray('"이 데이터로 예측 모델을 만들어줘"'));
    console.log(chalk.white('  📈 시각화: ') + chalk.gray('"차트를 그려줘", "시각화해줘"'));
    console.log(chalk.white('  🔄 모드 변경: ') + chalk.gray('"ML 모드로 변경해줘"'));
    console.log(chalk.white('  📋 상태 확인: ') + chalk.gray('"상태 확인해줘"'));
    console.log(chalk.white('  🔧 도구 목록: ') + chalk.gray('"도구 목록"'));
    console.log(chalk.white('  📜 대화 기록: ') + chalk.gray('"히스토리"'));
    console.log(chalk.white('  🚪 종료: ') + chalk.gray('"종료" 또는 "exit"'));
    console.log(chalk.gray('─'.repeat(50)));
  }

  showAvailableTools() {
    console.log(chalk.cyan('\n🔧 사용 가능한 도구:'));
    
    if (this.availableTools.length === 0) {
      console.log(chalk.yellow('  도구를 불러올 수 없습니다.'));
      return;
    }

    this.availableTools.forEach((tool, index) => {
      console.log(chalk.white(`  ${index + 1}. ${tool.name}`));
      console.log(chalk.gray(`     ${tool.description}`));
    });
    
    console.log(chalk.gray('─'.repeat(50)));
  }

  showConversationHistory() {
    console.log(chalk.cyan('\n📜 대화 히스토리:'));
    
    if (this.conversationHistory.length === 0) {
      console.log(chalk.yellow('  아직 대화 기록이 없습니다.'));
      return;
    }

    this.conversationHistory.slice(-10).forEach((entry, index) => {
      const time = new Date(entry.timestamp).toLocaleTimeString();
      console.log(chalk.blue(`[${time}] 사용자: `) + chalk.white(entry.input));
      
      if (entry.output) {
        const preview = entry.output.substring(0, 100);
        console.log(chalk.green(`[${time}] 시스템: `) + chalk.gray(preview + (entry.output.length > 100 ? '...' : '')));
      }
      
      console.log();
    });
    
    console.log(chalk.gray('─'.repeat(50)));
  }

  async processUserInput(userInput) {
    const startTime = Date.now();
    
    try {
      console.log(chalk.yellow('\n🔄 처리 중...'));
      
      // 사용자 입력을 분석하여 적절한 도구 선택
      const toolCall = this.analyzeUserInput(userInput);
      
      if (!toolCall) {
        console.log(chalk.red('❌ 요청을 이해할 수 없습니다. 다시 시도해주세요.'));
        console.log(chalk.gray('💡 "도움말"을 입력하면 사용 가능한 명령어를 확인할 수 있습니다.'));
        return;
      }
      
      console.log(chalk.cyan(`🔧 도구 실행: ${toolCall.name}`));
      
      // MCP 서버에 도구 호출 요청
      const result = await this.callMCPTool(toolCall);
      
      // 결과 표시
      await this.displayResult(result);
      
      // 대화 기록에 추가
      this.addToHistory(userInput, result, Date.now() - startTime);
      
    } catch (error) {
      console.error(chalk.red('❌ 처리 중 오류 발생:'), error.message);
    }
  }

  analyzeUserInput(userInput) {
    try {
      // 간단한 키워드 기반 분석
      const input = userInput.toLowerCase();
      
      // 데이터 분석 요청
      if (input.includes('분석') || input.includes('analyze') ||
          input.includes('통계') || input.includes('살펴') ||
          input.includes('조사') || input.includes('탐색')) {
        return {
          name: 'analyze_data',
          arguments: {
            query: userInput,
            auto_detect_files: true
          }
        };
      }
      
      // 시각화 요청
      if (input.includes('시각화') || input.includes('차트') ||
          input.includes('그래프') || input.includes('plot') ||
          input.includes('visualize') || input.includes('그려')) {
        return {
          name: 'visualize_data',
          arguments: {
            query: userInput,
            auto_detect_files: true
          }
        };
      }
      
      // 모델 훈련 요청
      if (input.includes('모델') || input.includes('훈련') ||
          input.includes('학습') || input.includes('train') ||
          input.includes('예측') || input.includes('predict')) {
        return {
          name: 'train_model',
          arguments: {
            query: userInput,
            auto_detect_files: true
          }
        };
      }
      
      // 시스템 상태 확인
      if (input.includes('상태') || input.includes('status') ||
          input.includes('건강') || input.includes('health') ||
          input.includes('모니터') || input.includes('시스템')) {
        return {
          name: 'system_status',
          arguments: {}
        };
      }
      
      // 모드 변경
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
      
      // 파일 관련 요청
      if (input.includes('파일') || input.includes('file') ||
          input.includes('데이터') || input.includes('csv') ||
          input.includes('excel') || input.includes('json')) {
        return {
          name: 'analyze_data',
          arguments: {
            query: userInput,
            auto_detect_files: true
          }
        };
      }
      
      // 기본적으로 일반 쿼리로 처리
      return {
        name: 'general_query',
        arguments: {
          query: userInput
        }
      };
      
    } catch (error) {
      console.error('입력 분석 실패:', error);
      return null;
    }
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

    if (result.isError) {
      console.log(chalk.red('❌ 오류 발생:'));
    } else {
      console.log(chalk.green('✅ 처리 완료:'));
    }

    if (result.content && Array.isArray(result.content)) {
      result.content.forEach(item => {
        if (item.type === 'text') {
          console.log(chalk.white(item.text));
        } else if (item.type === 'image') {
          console.log(chalk.cyan(`🖼️ 이미지: ${item.url || '이미지 생성됨'}`));
        } else if (item.type === 'json') {
          console.log(chalk.gray(JSON.stringify(item.data, null, 2)));
        }
      });
    } else {
      console.log(chalk.gray('결과 내용을 표시할 수 없습니다.'));
    }
  }

  addToHistory(input, output, duration) {
    const entry = {
      timestamp: Date.now(),
      input: input,
      output: output?.content?.[0]?.text || '응답 없음',
      duration: duration
    };

    this.conversationHistory.push(entry);
    
    // 최대 100개의 기록만 유지
    if (this.conversationHistory.length > 100) {
      this.conversationHistory.shift();
    }
  }

  async cleanup() {
    try {
      if (this.client && this.isConnected) {
        await this.client.close();
      }
      
      if (this.transport) {
        await this.transport.close();
      }
      
      if (this.rl) {
        this.rl.close();
      }
      
    } catch (error) {
      console.error('정리 중 오류:', error.message);
    }
  }
}

// 메인 실행 함수
async function main() {
  const client = new MCPCLIClient();
  
  // 종료 시그널 처리
  process.on('SIGINT', async () => {
    console.log(chalk.yellow('\n👋 MCP CLI를 종료합니다...'));
    await client.cleanup();
    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    console.log(chalk.yellow('\n👋 MCP CLI를 종료합니다...'));
    await client.cleanup();
    process.exit(0);
  });

  await client.initialize();
}

main().catch(console.error);
