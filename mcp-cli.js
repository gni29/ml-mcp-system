#!/usr/bin/env node

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { spawn } from 'child_process';
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
    this.serverProcess = null;
    this.isConnected = false;
    this.currentSession = null;
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
      
      // MCP 서버 시작
      await this.startMCPServer();
      
      // MCP 클라이언트 연결
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

    async startMCPServer() {
      return new Promise((resolve, reject) => {
        console.log(chalk.yellow('🔧 MCP 서버 시작 중...'));
        
        const serverPath = path.join(__dirname, 'main.js');
        console.log(chalk.gray(`서버 경로: ${serverPath}`));
        
        // MCP 서버 프로세스 시작 - stdio 설정 수정
        this.serverProcess = spawn('node', [serverPath], {
          stdio: ['pipe', 'pipe', 'pipe'], // stdin, stdout, stderr를 모두 pipe로 설정
          cwd: __dirname,
          env: process.env
        });

        let hasStarted = false;
        
        // 10초 타임아웃 설정
        const timeout = setTimeout(() => {
          if (!hasStarted) {
            reject(new Error('MCP 서버 시작 타임아웃'));
          }
        }, 10000);

        // 서버 출력 모니터링
        this.serverProcess.stdout.on('data', (data) => {
          const output = data.toString();
          console.log(chalk.gray('서버 출력:'), output.trim());
          
          if (output.includes('ML MCP 서버가 시작되었습니다') ||
              output.includes('서버가 시작되었습니다') ||
              output.includes('Server started')) {
            hasStarted = true;
            clearTimeout(timeout);
            console.log(chalk.green('✅ MCP 서버 시작 완료'));
            // 서버가 완전히 준비될 때까지 잠시 대기
            setTimeout(resolve, 2000);
          }
        });

        // 에러 처리
        this.serverProcess.stderr.on('data', (data) => {
          const errorMsg = data.toString();
          console.error(chalk.red('서버 오류:'), errorMsg.trim());
          
          if (errorMsg.includes('Error:') || errorMsg.includes('ERROR')) {
            clearTimeout(timeout);
            reject(new Error(`MCP 서버 시작 실패: ${errorMsg}`));
          }
        });

        // 프로세스 종료 처리
        this.serverProcess.on('close', (code) => {
          if (code !== 0 && !hasStarted) {
            clearTimeout(timeout);
            reject(new Error(`MCP 서버 프로세스가 종료되었습니다. 종료 코드: ${code}`));
          }
        });

        // 프로세스 오류 처리
        this.serverProcess.on('error', (error) => {
          clearTimeout(timeout);
          reject(new Error(`MCP 서버 프로세스 오류: ${error.message}`));
        });
      });
    }

    async connectToServer() {
      console.log(chalk.yellow('🔗 MCP 서버에 연결 중...'));
      
      try {
        // 서버 프로세스가 실행 중인지 확인
        if (!this.serverProcess || this.serverProcess.killed) {
          throw new Error('MCP 서버 프로세스가 실행되지 않고 있습니다.');
        }

        // stdout과 stdin이 제대로 설정되었는지 확인
        if (!this.serverProcess.stdout || !this.serverProcess.stdin) {
          throw new Error('MCP 서버 프로세스의 stdio가 올바르게 설정되지 않았습니다.');
        }

        // StdioClientTransport 생성 - 올바른 파라미터로 설정
        this.transport = new StdioClientTransport({
          reader: this.serverProcess.stdout,
          writer: this.serverProcess.stdin
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

        // 연결 시도
        await this.client.connect(this.transport);
        this.isConnected = true;
        
        console.log(chalk.green('✅ MCP 서버 연결 완료'));
        
      } catch (error) {
        console.error(chalk.red('연결 실패 상세:'), error.message);
        console.error(chalk.red('스택 트레이스:'), error.stack);
        
        // 추가 디버깅 정보
        if (this.serverProcess) {
          console.log(chalk.gray('서버 프로세스 상태:'), {
            killed: this.serverProcess.killed,
            exitCode: this.serverProcess.exitCode,
            pid: this.serverProcess.pid
          });
        }
        
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
    console.log(chalk.yellow('💡 도움말: "도움말" 또는 "help"'));
    console.log(chalk.yellow('🔧 사용 가능한 도구: "도구 목록"'));
    console.log(chalk.yellow('🚪 종료: "종료" 또는 "exit"'));
    console.log(chalk.gray('─'.repeat(50)));
    
    // 사용 예시
    console.log(chalk.blue('📖 사용 예시:'));
    console.log(chalk.white('  • "data.csv 파일을 분석해주세요"'));
    console.log(chalk.white('  • "이 데이터로 예측 모델을 만들어주세요"'));
    console.log(chalk.white('  • "시각화 차트를 그려주세요"'));
    console.log(chalk.white('  • "시스템 상태를 확인해주세요"'));
    console.log(chalk.gray('─'.repeat(50)));
  }

  startConversation() {
    this.currentSession = `session_${Date.now()}`;
    this.rl.prompt();
    
    this.rl.on('line', async (input) => {
      const userInput = input.trim();
      
      if (!userInput) {
        this.rl.prompt();
        return;
      }
      
      // 종료 명령 처리
      if (this.isExitCommand(userInput)) {
        await this.shutdown();
        return;
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
      
      // 대화 히스토리 명령 처리
      if (this.isHistoryCommand(userInput)) {
        this.showConversationHistory();
        this.rl.prompt();
        return;
      }
      
      // 일반 명령 처리
      await this.processUserInput(userInput);
      this.rl.prompt();
    });
    
    this.rl.on('close', async () => {
      await this.shutdown();
    });
  }

  isExitCommand(input) {
    const exitCommands = ['exit', 'quit', '종료', 'bye', 'goodbye', 'q'];
    return exitCommands.includes(input.toLowerCase());
  }

  isHelpCommand(input) {
    const helpCommands = ['help', 'h', '도움말', '도움', '?'];
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
      
      // 연결 문제인 경우 재연결 시도
      if (error.message.includes('연결') || error.message.includes('connection')) {
        console.log(chalk.yellow('🔄 서버 재연결 시도 중...'));
        try {
          await this.reconnectToServer();
          console.log(chalk.green('✅ 서버 재연결 성공'));
        } catch (reconnectError) {
          console.error(chalk.red('❌ 재연결 실패:'), reconnectError.message);
        }
      }
    }
  }

  async reconnectToServer() {
    if (this.isConnected) {
      await this.client.close();
      this.isConnected = false;
    }
    
    // 잠시 대기 후 재연결
    await new Promise(resolve => setTimeout(resolve, 2000));
    await this.connectToServer();
    await this.loadAvailableTools();
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
    
    console.log(chalk.green('\n🤖 응답:'));
    
    if (result.content) {
      for (const content of result.content) {
        if (content.type === 'text') {
          console.log(chalk.white(content.text));
        } else if (content.type === 'image') {
          console.log(chalk.cyan(`🖼️ 이미지: ${content.source || 'image'}`));
        } else if (content.type === 'resource') {
          console.log(chalk.cyan(`📄 리소스: ${content.resource.uri}`));
        } else {
          console.log(chalk.gray(`📄 ${content.type}: ${JSON.stringify(content, null, 2)}`));
        }
      }
    } else if (result.result) {
      // 결과가 result 프로퍼티에 있는 경우
      console.log(chalk.white(JSON.stringify(result.result, null, 2)));
    } else {
      console.log(chalk.white(JSON.stringify(result, null, 2)));
    }
    
    // 에러 표시
    if (result.isError) {
      console.log(chalk.red('\n⚠️ 처리 중 오류가 발생했습니다.'));
    }
    
    // 결과 저장
    if (result.shouldSave !== false) {
      await this.saveResult(result);
    }
  }

  async saveResult(result) {
    try {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const sessionDir = path.join('./results', `${this.currentSession}_${timestamp.split('T')[0]}`);
      
      await fs.mkdir(sessionDir, { recursive: true });
      
      const resultFile = path.join(sessionDir, `result_${timestamp}.json`);
      await fs.writeFile(resultFile, JSON.stringify(result, null, 2));
      
      console.log(chalk.cyan(`💾 결과 저장됨: ${resultFile}`));
      
    } catch (error) {
      console.warn(chalk.yellow('⚠️ 결과 저장 실패:'), error.message);
    }
  }

  addToHistory(input, output, executionTime) {
    const entry = {
      timestamp: Date.now(),
      input: input,
      output: output ? JSON.stringify(output) : null,
      executionTime: executionTime
    };
    
    this.conversationHistory.push(entry);
    
    // 히스토리 크기 제한 (최대 100개)
    if (this.conversationHistory.length > 100) {
      this.conversationHistory = this.conversationHistory.slice(-50);
    }
  }

  async shutdown() {
    console.log(chalk.cyan('\n👋 MCP CLI를 종료합니다...'));
    
    try {
      // 대화 히스토리 저장
      await this.saveConversationHistory();
      
      // MCP 클라이언트 연결 해제
      if (this.client && this.isConnected) {
        await this.client.close();
        this.isConnected = false;
      }
      
      // 서버 프로세스 종료
      if (this.serverProcess && !this.serverProcess.killed) {
        console.log(chalk.yellow('🔧 MCP 서버 종료 중...'));
        
        // 정상 종료 시그널 전송
        this.serverProcess.kill('SIGTERM');
        
        // 강제 종료 대기
        await new Promise((resolve) => {
          const timeout = setTimeout(() => {
            if (this.serverProcess && !this.serverProcess.killed) {
              console.log(chalk.yellow('⚠️ 서버 강제 종료'));
              this.serverProcess.kill('SIGKILL');
            }
            resolve();
          }, 5000);
          
          this.serverProcess.on('exit', () => {
            clearTimeout(timeout);
            resolve();
          });
        });
      }
      
      console.log(chalk.green('✅ 정상적으로 종료되었습니다.'));
      
    } catch (error) {
      console.error(chalk.red('종료 중 오류:'), error.message);
    }
    
    this.rl.close();
    process.exit(0);
  }

  async saveConversationHistory() {
    try {
      if (this.conversationHistory.length > 0) {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const historyFile = path.join('./logs', `conversation_${this.currentSession}_${timestamp}.json`);
        
        const historyData = {
          session: this.currentSession,
          startTime: this.conversationHistory[0]?.timestamp,
          endTime: Date.now(),
          totalEntries: this.conversationHistory.length,
          history: this.conversationHistory
        };
        
        await fs.writeFile(historyFile, JSON.stringify(historyData, null, 2));
        console.log(chalk.cyan(`💾 대화 기록 저장됨: ${historyFile}`));
      }
    } catch (error) {
      console.warn(chalk.yellow('⚠️ 대화 기록 저장 실패:'), error.message);
    }
  }

  async cleanup() {
    await this.shutdown();
  }
}

// 메인 실행
async function main() {
  const cli = new MCPCLIClient();
  
  // 시그널 핸들러
  process.on('SIGINT', async () => {
    console.log(chalk.yellow('\n🔄 종료 시그널 수신...'));
    await cli.cleanup();
  });
  
  process.on('SIGTERM', async () => {
    console.log(chalk.yellow('\n🔄 종료 시그널 수신...'));
    await cli.cleanup();
  });
  
  await cli.initialize();
}

// 에러 핸들링
process.on('unhandledRejection', (reason, promise) => {
  console.error(chalk.red('Unhandled Rejection:'), reason);
  process.exit(1);
});

process.on('uncaughtException', (error) => {
  console.error(chalk.red('Uncaught Exception:'), error);
  process.exit(1);
});

main().catch(console.error);
