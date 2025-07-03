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
      await this.cleanup();
      process.exit(1);
    }
  }

  async createDirectories() {
    const directories = [
      './results',
      './uploads',
      './temp',
      './logs'
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
      
      // MCP 서버 프로세스 시작
      this.serverProcess = spawn('node', [serverPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, MCP_MODE: 'server' }
      });

      // 서버 시작 대기
      let outputData = '';
      
      const timeout = setTimeout(() => {
        reject(new Error('MCP 서버 시작 시간 초과'));
      }, 30000);

      this.serverProcess.stdout.on('data', (data) => {
        outputData += data.toString();
        
        // 서버 시작 완료 메시지 확인
        if (outputData.includes('ML MCP 서버가 시작되었습니다')) {
          clearTimeout(timeout);
          console.log(chalk.green('✅ MCP 서버 시작 완료'));
          resolve();
        }
      });

      this.serverProcess.stderr.on('data', (data) => {
        const errorMsg = data.toString();
        if (errorMsg.includes('error') || errorMsg.includes('Error')) {
          clearTimeout(timeout);
          reject(new Error(`MCP 서버 오류: ${errorMsg}`));
        }
      });

      this.serverProcess.on('error', (error) => {
        clearTimeout(timeout);
        reject(new Error(`MCP 서버 프로세스 오류: ${error.message}`));
      });

      this.serverProcess.on('exit', (code) => {
        if (code !== 0) {
          clearTimeout(timeout);
          reject(new Error(`MCP 서버가 예상치 못하게 종료됨 (코드: ${code})`));
        }
      });
    });
  }

  async connectToServer() {
    console.log(chalk.yellow('🔗 MCP 서버에 연결 중...'));
    
    // StdioClientTransport 생성
    this.transport = new StdioClientTransport({
      stdin: this.serverProcess.stdin,
      stdout: this.serverProcess.stdout
    });

    // MCP 클라이언트 생성
    this.client = new Client(
      {
        name: 'ml-mcp-cli',
        version: '1.0.0'
      },
      {
        capabilities: {
          roots: {
            listChanged: true
          }
        }
      }
    );

    // 서버에 연결
    await this.client.connect(this.transport);
    this.isConnected = true;
    
    console.log(chalk.green('✅ MCP 서버 연결 완료'));
  }

  async loadAvailableTools() {
    try {
      const response = await this.client.request(
        {
          method: 'tools/list'
        },
        {
          method: 'tools/list',
          params: {}
        }
      );

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
      
      // 일반 명령 처리
      await this.processUserInput(userInput);
      this.rl.prompt();
    });
    
    this.rl.on('close', async () => {
      await this.shutdown();
    });
  }

  isExitCommand(input) {
    const exitCommands = ['exit', 'quit', '종료', 'bye', 'goodbye'];
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

  showHelp() {
    console.log(chalk.cyan('\n📖 사용 가능한 명령어:'));
    console.log(chalk.white('  📊 데이터 분석: ') + chalk.gray('"data.csv 파일을 분석해줘"'));
    console.log(chalk.white('  🤖 모델 훈련: ') + chalk.gray('"이 데이터로 예측 모델을 만들어줘"'));
    console.log(chalk.white('  📈 시각화: ') + chalk.gray('"차트를 그려줘", "시각화해줘"'));
    console.log(chalk.white('  🔄 모드 변경: ') + chalk.gray('"ML 모드로 변경해줘"'));
    console.log(chalk.white('  📋 상태 확인: ') + chalk.gray('"상태 확인해줘"'));
    console.log(chalk.white('  🔧 도구 목록: ') + chalk.gray('"도구 목록"'));
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

  async processUserInput(userInput) {
    try {
      console.log(chalk.yellow('\n🔄 처리 중...'));
      
      // 사용자 입력을 분석하여 적절한 도구 선택
      const toolCall = await this.analyzeUserInput(userInput);
      
      if (!toolCall) {
        console.log(chalk.red('❌ 요청을 이해할 수 없습니다. 다시 시도해주세요.'));
        return;
      }
      
      // MCP 서버에 도구 호출 요청
      const result = await this.callMCPTool(toolCall);
      
      // 결과 표시
      await this.displayResult(result);
      
    } catch (error) {
      console.error(chalk.red('❌ 처리 중 오류 발생:'), error.message);
    }
  }

  async analyzeUserInput(userInput) {
    try {
      // 간단한 키워드 기반 분석 (나중에 AI 모델로 대체 가능)
      const input = userInput.toLowerCase();
      
      if (input.includes('분석') || input.includes('analyze')) {
        return {
          name: 'analyze_data',
          arguments: {
            query: userInput,
            auto_detect_files: true
          }
        };
      }
      
      if (input.includes('시각화') || input.includes('차트') || input.includes('그래프')) {
        return {
          name: 'visualize_data',
          arguments: {
            query: userInput,
            auto_detect_files: true
          }
        };
      }
      
      if (input.includes('모델') || input.includes('훈련') || input.includes('학습')) {
        return {
          name: 'train_model',
          arguments: {
            query: userInput,
            auto_detect_files: true
          }
        };
      }
      
      if (input.includes('상태') || input.includes('status')) {
        return {
          name: 'system_status',
          arguments: {}
        };
      }
      
      if (input.includes('모드')) {
        return {
          name: 'mode_switch',
          arguments: {
            mode: input.includes('ml') ? 'ml' : 'general'
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
      const response = await this.client.request(
        {
          method: 'tools/call',
          params: {
            name: toolCall.name,
            arguments: toolCall.arguments
          }
        },
        {
          method: 'tools/call',
          params: {
            name: toolCall.name,
            arguments: toolCall.arguments
          }
        }
      );

      return response;
      
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
        }
      }
    } else {
      console.log(chalk.white(JSON.stringify(result, null, 2)));
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

  async shutdown() {
    console.log(chalk.cyan('\n👋 MCP CLI를 종료합니다...'));
    
    try {
      // MCP 클라이언트 연결 해제
      if (this.client && this.isConnected) {
        await this.client.close();
      }
      
      // 서버 프로세스 종료
      if (this.serverProcess && !this.serverProcess.killed) {
        this.serverProcess.kill('SIGTERM');
        
        // 강제 종료 대기
        await new Promise((resolve) => {
          const timeout = setTimeout(() => {
            this.serverProcess.kill('SIGKILL');
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

  async cleanup() {
    await this.shutdown();
  }
}

// 메인 실행
async function main() {
  const cli = new MCPCLIClient();
  
  // 시그널 핸들러
  process.on('SIGINT', async () => {
    await cli.cleanup();
  });
  
  process.on('SIGTERM', async () => {
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
