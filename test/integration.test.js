#!/usr/bin/env node

// test/integration.test.js
import { spawn } from 'child_process';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';
import chalk from 'chalk';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class IntegrationTest {
  constructor() {
    this.ollamaEndpoint = 'http://localhost:11434';
    this.serverProcess = null;
    this.client = null;
    this.transport = null;
    this.testResults = [];
    this.startTime = Date.now();
    this.requiredModels = ['llama3.2:3b', 'qwen2.5:14b'];
  }

  async run() {
    try {
      console.log(chalk.cyan('🧪 ML MCP 시스템 통합 테스트 시작'));
      console.log(chalk.gray('=' * 60));

      // 1. 환경 검증
      await this.validateEnvironment();

      // 2. 서비스 상태 확인
      await this.checkServices();

      // 3. MCP 서버 시작
      await this.startMCPServer();

      // 4. 클라이언트 연결
      await this.connectClient();

      // 5. 기본 기능 테스트
      await this.runBasicTests();

      // 6. 모델 테스트
      await this.runModelTests();

      // 7. 도구 테스트
      await this.runToolTests();

      // 8. 성능 테스트
      await this.runPerformanceTests();

      // 9. 결과 보고서 생성
      await this.generateReport();

      // 10. 정리
      await this.cleanup();

    } catch (error) {
      console.error(chalk.red('❌ 통합 테스트 실패:'), error.message);
      await this.cleanup();
      process.exit(1);
    }
  }

  async validateEnvironment() {
    console.log(chalk.yellow('🔍 환경 검증 중...'));
    
    // Node.js 버전 확인
    const nodeVersion = process.version;
    console.log(chalk.blue(`Node.js 버전: ${nodeVersion}`));
    
    const majorVersion = parseInt(nodeVersion.slice(1).split('.')[0]);
    if (majorVersion < 18) {
      throw new Error('Node.js 18 이상이 필요합니다.');
    }

    // 필요한 파일 존재 확인
    const requiredFiles = [
      '../main.js',
      '../mcp-cli.js',
      '../package.json',
      '../models/model-configs.json'
    ];

    for (const file of requiredFiles) {
      const filePath = path.join(__dirname, file);
      try {
        await fs.access(filePath);
        console.log(chalk.green(`✅ ${file} 존재`));
      } catch (error) {
        console.log(chalk.red(`❌ ${file} 누락`));
        throw new Error(`필수 파일 누락: ${file}`);
      }
    }

    // 필요한 디렉토리 생성
    const requiredDirs = ['../results', '../temp', '../logs', '../data'];
    for (const dir of requiredDirs) {
      const dirPath = path.join(__dirname, dir);
      try {
        await fs.mkdir(dirPath, { recursive: true });
      } catch (error) {
        // 이미 존재하면 무시
      }
    }

    console.log(chalk.green('✅ 환경 검증 완료'));
  }

  async checkServices() {
    console.log(chalk.yellow('🔗 서비스 상태 확인 중...'));

    // Ollama 서비스 확인
    try {
      const response = await axios.get(`${this.ollamaEndpoint}/api/version`, {
        timeout: 5000
      });
      
      console.log(chalk.green('✅ Ollama 서비스 실행 중'));
      console.log(chalk.gray(`   버전: ${response.data.version || 'Unknown'}`));
      
      this.recordTest('ollama_service', true, 'Ollama 서비스 연결 성공');

    } catch (error) {
      console.log(chalk.red('❌ Ollama 서비스 연결 실패'));
      this.recordTest('ollama_service', false, error.message);
      throw new Error('Ollama 서비스를 먼저 시작해주세요: ollama serve');
    }

    // 필요한 모델 확인
    try {
      const response = await axios.get(`${this.ollamaEndpoint}/api/tags`);
      const installedModels = response.data.models?.map(m => m.name) || [];
      
      console.log(chalk.blue(`설치된 모델: ${installedModels.length}개`));
      
      let missingModels = [];
      for (const model of this.requiredModels) {
        if (installedModels.includes(model)) {
          console.log(chalk.green(`✅ ${model} - 설치됨`));
        } else {
          console.log(chalk.red(`❌ ${model} - 누락`));
          missingModels.push(model);
        }
      }

      if (missingModels.length > 0) {
        console.log(chalk.yellow('⚠️ 누락된 모델이 있습니다:'));
        missingModels.forEach(model => {
          console.log(chalk.white(`   ollama pull ${model}`));
        });
        console.log(chalk.gray('또는 npm run models 명령으로 자동 설치'));
      }

      this.recordTest('models_check', missingModels.length === 0,
        `${this.requiredModels.length - missingModels.length}/${this.requiredModels.length} 모델 사용 가능`);

    } catch (error) {
      this.recordTest('models_check', false, error.message);
    }
  }

  async startMCPServer() {
    console.log(chalk.yellow('🚀 MCP 서버 시작 중...'));

    return new Promise((resolve, reject) => {
      const serverPath = path.join(__dirname, '../main.js');
      
      this.serverProcess = spawn('node', [serverPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, NODE_ENV: 'test' }
      });

      let hasStarted = false;
      const timeout = setTimeout(() => {
        if (!hasStarted) {
          reject(new Error('MCP 서버 시작 시간 초과'));
        }
      }, 30000);

      this.serverProcess.stdout.on('data', (data) => {
        const output = data.toString();
        
        if (output.includes('서버가 시작되었습니다') ||
            output.includes('ML MCP 서버가 시작되었습니다')) {
          hasStarted = true;
          clearTimeout(timeout);
          console.log(chalk.green('✅ MCP 서버 시작 완료'));
          this.recordTest('mcp_server_start', true, 'MCP 서버 시작 성공');
          setTimeout(resolve, 2000); // 서버 완전 로드 대기
        }
      });

      this.serverProcess.stderr.on('data', (data) => {
        const errorMsg = data.toString();
        if (errorMsg.includes('Error:')) {
          clearTimeout(timeout);
          reject(new Error(`MCP 서버 시작 실패: ${errorMsg}`));
        }
      });

      this.serverProcess.on('error', (error) => {
        clearTimeout(timeout);
        reject(new Error(`MCP 서버 프로세스 오류: ${error.message}`));
      });
    });
  }

  async connectClient() {
    console.log(chalk.yellow('🔗 MCP 클라이언트 연결 중...'));

    try {
      this.transport = new StdioClientTransport({
        reader: this.serverProcess.stdout,
        writer: this.serverProcess.stdin
      });

      this.client = new Client(
        {
          name: 'integration-test-client',
          version: '1.0.0'
        },
        {
          capabilities: {}
        }
      );

      await this.client.connect(this.transport);
      
      console.log(chalk.green('✅ MCP 클라이언트 연결 완료'));
      this.recordTest('mcp_client_connect', true, 'MCP 클라이언트 연결 성공');

    } catch (error) {
      console.log(chalk.red('❌ MCP 클라이언트 연결 실패'));
      this.recordTest('mcp_client_connect', false, error.message);
      throw error;
    }
  }

  async runBasicTests() {
    console.log(chalk.yellow('🧪 기본 기능 테스트 중...'));

    // 도구 목록 조회 테스트
    try {
      const toolsResponse = await this.client.listTools();
      const tools = toolsResponse.tools || [];
      
      console.log(chalk.green(`✅ 도구 목록 조회 성공: ${tools.length}개`));
      this.recordTest('list_tools', true, `${tools.length}개 도구 발견`);

      // 주요 도구 확인
      const expectedTools = ['general_query', 'system_status', 'analyze_data'];
      const foundTools = tools.map(t => t.name);
      
      for (const tool of expectedTools) {
        if (foundTools.includes(tool)) {
          console.log(chalk.green(`✅ ${tool} 도구 존재`));
        } else {
          console.log(chalk.yellow(`⚠️ ${tool} 도구 누락`));
        }
      }

    } catch (error) {
      console.log(chalk.red('❌ 도구 목록 조회 실패'));
      this.recordTest('list_tools', false, error.message);
    }

    // 기본 응답 테스트
    try {
      const response = await this.client.callTool({
        name: 'general_query',
        arguments: {
          query: 'Hello, this is a test'
        }
      });

      if (response && response.content) {
        console.log(chalk.green('✅ 기본 응답 테스트 성공'));
        this.recordTest('basic_response', true, '기본 응답 생성 성공');
      } else {
        console.log(chalk.red('❌ 기본 응답 테스트 실패'));
        this.recordTest('basic_response', false, '응답 내용 없음');
      }

    } catch (error) {
      console.log(chalk.red('❌ 기본 응답 테스트 실패'));
      this.recordTest('basic_response', false, error.message);
    }
  }

  async runModelTests() {
    console.log(chalk.yellow('🤖 모델 테스트 중...'));

    // 라우터 모델 테스트
    try {
      const startTime = Date.now();
      
      const response = await this.client.callTool({
        name: 'general_query',
        arguments: {
          query: '간단한 질문입니다. 짧게 답해주세요.'
        }
      });

      const responseTime = Date.now() - startTime;
      
      if (response && response.content) {
        console.log(chalk.green(`✅ 라우터 모델 테스트 성공 (${responseTime}ms)`));
        this.recordTest('router_model', true, `응답 시간: ${responseTime}ms`);
      } else {
        console.log(chalk.red('❌ 라우터 모델 테스트 실패'));
        this.recordTest('router_model', false, '응답 없음');
      }

    } catch (error) {
      console.log(chalk.red('❌ 라우터 모델 테스트 실패'));
      this.recordTest('router_model', false, error.message);
    }

    // 시스템 상태 테스트
    try {
      const response = await this.client.callTool({
        name: 'system_status',
        arguments: {}
      });

      if (response && response.content) {
        console.log(chalk.green('✅ 시스템 상태 테스트 성공'));
        this.recordTest('system_status', true, '시스템 상태 조회 성공');
      } else {
        console.log(chalk.red('❌ 시스템 상태 테스트 실패'));
        this.recordTest('system_status', false, '상태 정보 없음');
      }

    } catch (error) {
      console.log(chalk.red('❌ 시스템 상태 테스트 실패'));
      this.recordTest('system_status', false, error.message);
    }
  }

  async runToolTests() {
    console.log(chalk.yellow('🔧 도구 테스트 중...'));

    // 테스트 데이터 생성
    await this.createTestData();

    // 데이터 분석 도구 테스트
    try {
      const response = await this.client.callTool({
        name: 'analyze_data',
        arguments: {
          query: 'test_data.csv 파일을 분석해주세요',
          auto_detect_files: true
        }
      });

      if (response && response.content) {
        console.log(chalk.green('✅ 데이터 분석 도구 테스트 성공'));
        this.recordTest('analyze_data_tool', true, '데이터 분석 성공');
      } else {
        console.log(chalk.yellow('⚠️ 데이터 분석 도구 테스트 부분 성공'));
        this.recordTest('analyze_data_tool', false, '분석 결과 없음');
      }

    } catch (error) {
      console.log(chalk.red('❌ 데이터 분석 도구 테스트 실패'));
      this.recordTest('analyze_data_tool', false, error.message);
    }

    // 모드 변경 테스트
    try {
      const response = await this.client.callTool({
        name: 'mode_switch',
        arguments: {
          mode: 'ml'
        }
      });

      if (response && response.content) {
        console.log(chalk.green('✅ 모드 변경 테스트 성공'));
        this.recordTest('mode_switch', true, 'ML 모드 변경 성공');
      } else {
        console.log(chalk.red('❌ 모드 변경 테스트 실패'));
        this.recordTest('mode_switch', false, '모드 변경 응답 없음');
      }

    } catch (error) {
      console.log(chalk.red('❌ 모드 변경 테스트 실패'));
      this.recordTest('mode_switch', false, error.message);
    }
  }

  async runPerformanceTests() {
    console.log(chalk.yellow('⚡ 성능 테스트 중...'));

    // 동시 요청 테스트
    try {
      const concurrentRequests = 3;
      const startTime = Date.now();
      
      const promises = Array.from({ length: concurrentRequests }, (_, i) =>
        this.client.callTool({
          name: 'general_query',
          arguments: {
            query: `동시 요청 테스트 ${i + 1}`
          }
        })
      );

      const results = await Promise.all(promises);
      const totalTime = Date.now() - startTime;
      const avgTime = totalTime / concurrentRequests;

      console.log(chalk.green(`✅ 동시 요청 테스트 성공 (평균: ${avgTime.toFixed(2)}ms)`));
      this.recordTest('concurrent_requests', true, `${concurrentRequests}개 요청, 평균 ${avgTime.toFixed(2)}ms`);

    } catch (error) {
      console.log(chalk.red('❌ 동시 요청 테스트 실패'));
      this.recordTest('concurrent_requests', false, error.message);
    }

    // 메모리 사용량 테스트
    try {
      const memoryUsage = process.memoryUsage();
      const heapUsedMB = Math.round(memoryUsage.heapUsed / 1024 / 1024);
      const rssUsedMB = Math.round(memoryUsage.rss / 1024 / 1024);

      console.log(chalk.blue(`메모리 사용량: Heap ${heapUsedMB}MB, RSS ${rssUsedMB}MB`));
      this.recordTest('memory_usage', true, `Heap: ${heapUsedMB}MB, RSS: ${rssUsedMB}MB`);

    } catch (error) {
      this.recordTest('memory_usage', false, error.message);
    }
  }

  async createTestData() {
    try {
      const testData = `name,age,city
John,25,New York
Jane,30,Los Angeles
Bob,22,Chicago
Alice,28,Houston
Charlie,35,Phoenix`;

      await fs.writeFile('./test_data.csv', testData);
      console.log(chalk.gray('테스트 데이터 생성 완료'));

    } catch (error) {
      console.log(chalk.yellow('테스트 데이터 생성 실패'));
    }
  }

  recordTest(testName, success, details) {
    this.testResults.push({
      name: testName,
      success: success,
      details: details,
      timestamp: new Date().toISOString()
    });
  }

  async generateReport() {
    console.log(chalk.yellow('📊 테스트 결과 보고서 생성 중...'));

    const totalTests = this.testResults.length;
    const passedTests = this.testResults.filter(t => t.success).length;
    const failedTests = totalTests - passedTests;
    const successRate = ((passedTests / totalTests) * 100).toFixed(1);
    const totalTime = Date.now() - this.startTime;

    // 콘솔 출력
    console.log(chalk.green('\n🎉 테스트 완료!'));
    console.log(chalk.gray('=' * 60));
    console.log(chalk.cyan('📈 테스트 결과 요약:'));
    console.log(chalk.white(`   총 테스트: ${totalTests}개`));
    console.log(chalk.green(`   성공: ${passedTests}개`));
    console.log(chalk.red(`   실패: ${failedTests}개`));
    console.log(chalk.blue(`   성공률: ${successRate}%`));
    console.log(chalk.gray(`   실행 시간: ${(totalTime / 1000).toFixed(2)}초`));

    // 실패한 테스트 상세 정보
    if (failedTests > 0) {
      console.log(chalk.red('\n❌ 실패한 테스트:'));
      this.testResults
        .filter(t => !t.success)
        .forEach(test => {
          console.log(chalk.white(`   • ${test.name}: ${test.details}`));
        });
    }

    // 성공한 테스트
    if (passedTests > 0) {
      console.log(chalk.green('\n✅ 성공한 테스트:'));
      this.testResults
        .filter(t => t.success)
        .forEach(test => {
          console.log(chalk.white(`   • ${test.name}: ${test.details}`));
        });
    }

    // 보고서 파일 생성
    try {
      const report = {
        summary: {
          total_tests: totalTests,
          passed_tests: passedTests,
          failed_tests: failedTests,
          success_rate: successRate + '%',
          execution_time: (totalTime / 1000).toFixed(2) + 's',
          timestamp: new Date().toISOString()
        },
        details: this.testResults,
        environment: {
          node_version: process.version,
          platform: process.platform,
          arch: process.arch
        }
      };

      await fs.mkdir('./test_results', { recursive: true });
      const reportPath = `./test_results/integration_test_${Date.now()}.json`;
      await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
      
      console.log(chalk.cyan(`\n📄 상세 보고서: ${reportPath}`));

    } catch (error) {
      console.log(chalk.yellow('보고서 파일 생성 실패:', error.message));
    }

    // 권장사항
    console.log(chalk.blue('\n💡 권장사항:'));
    if (failedTests === 0) {
      console.log(chalk.white('   🎉 모든 테스트가 성공했습니다!'));
      console.log(chalk.white('   시스템이 정상적으로 작동합니다.'));
    } else {
      console.log(chalk.white('   • 실패한 테스트를 확인하고 문제를 해결하세요.'));
      console.log(chalk.white('   • Ollama 서비스와 모델 상태를 확인하세요.'));
      console.log(chalk.white('   • 로그 파일을 확인하여 상세한 오류 정보를 확인하세요.'));
    }
  }

  async cleanup() {
    console.log(chalk.yellow('🧹 정리 중...'));

    // 클라이언트 연결 해제
    try {
      if (this.client) {
        await this.client.close();
      }
    } catch (error) {
      console.log(chalk.yellow('클라이언트 정리 실패'));
    }

    // 서버 프로세스 종료
    try {
      if (this.serverProcess && !this.serverProcess.killed) {
        this.serverProcess.kill('SIGTERM');
        
        // 강제 종료 대기
        await new Promise((resolve) => {
          const timeout = setTimeout(() => {
            if (this.serverProcess && !this.serverProcess.killed) {
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
    } catch (error) {
      console.log(chalk.yellow('서버 프로세스 정리 실패'));
    }

    // 테스트 파일 정리
    try {
      await fs.unlink('./test_data.csv');
    } catch (error) {
      // 파일이 없으면 무시
    }

    console.log(chalk.green('✅ 정리 완료'));
  }
}

// 메인 실행
async function main() {
  const test = new IntegrationTest();
  await test.run();
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

// SIGINT 핸들러 (Ctrl+C)
process.on('SIGINT', async () => {
  console.log(chalk.yellow('\n테스트 중단됨'));
  process.exit(0);
});

main().catch(console.error);
