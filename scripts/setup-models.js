#!/usr/bin/env node

// scripts/setup-models.js
import { spawn } from 'child_process';
import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';
import chalk from 'chalk';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class ModelSetup {
  constructor() {
    this.ollamaEndpoint = 'http://localhost:11434';
    this.models = [
      {
        name: 'llama3.2:3b',
        role: 'router',
        description: '빠른 의도 파악 및 라우팅 결정',
        estimatedSize: '~2GB',
        memoryUsage: '~6GB',
        priority: 1
      },
      {
        name: 'qwen2.5:14b',
        role: 'processor',
        description: '복잡한 코드 생성 및 분석 작업',
        estimatedSize: '~8GB',
        memoryUsage: '~28GB',
        priority: 2
      }
    ];
    this.installedModels = [];
    this.failedModels = [];
  }

  async run() {
    try {
      console.log(chalk.cyan('🤖 ML MCP 모델 자동 설치 시작'));
      console.log(chalk.gray('=' * 50));

      // 1. 시스템 요구사항 확인
      await this.checkSystemRequirements();

      // 2. Ollama 서비스 확인
      await this.checkOllamaService();

      // 3. 현재 설치된 모델 확인
      await this.checkExistingModels();

      // 4. 모델 설치
      await this.installModels();

      // 5. 설치 완료 후 테스트
      await this.testInstalledModels();

      // 6. 설정 파일 생성/업데이트
      await this.updateConfigFiles();

      // 7. 완료 메시지
      this.showCompletionMessage();

    } catch (error) {
      console.error(chalk.red('❌ 모델 설치 실패:'), error.message);
      process.exit(1);
    }
  }

  async checkSystemRequirements() {
    console.log(chalk.yellow('📋 시스템 요구사항 확인 중...'));

    // 디스크 공간 확인
    try {
      const stats = await fs.stat('./');
      console.log(chalk.green('✅ 디스크 접근 가능'));
    } catch (error) {
      throw new Error('디스크 접근 권한이 없습니다.');
    }

    // 메모리 확인
    const totalMemory = process.memoryUsage().heapTotal;
    const totalMemoryGB = Math.round(totalMemory / 1024 / 1024 / 1024);
    
    console.log(chalk.blue(`💾 사용 가능한 메모리: ~${totalMemoryGB}GB`));
    
    if (totalMemoryGB < 8) {
      console.log(chalk.yellow('⚠️ 메모리가 부족할 수 있습니다. 최소 8GB 권장'));
    }

    // 필요한 디스크 공간 안내
    const totalSize = this.models.reduce((sum, model) => {
      return sum + parseInt(model.estimatedSize.replace('~', '').replace('GB', ''));
    }, 0);

    console.log(chalk.blue(`💿 필요한 디스크 공간: ~${totalSize}GB`));
    console.log(chalk.green('✅ 시스템 요구사항 확인 완료'));
  }

  async checkOllamaService() {
    console.log(chalk.yellow('🔍 Ollama 서비스 확인 중...'));

    try {
      const response = await axios.get(`${this.ollamaEndpoint}/api/version`, {
        timeout: 5000
      });

      console.log(chalk.green('✅ Ollama 서비스 실행 중'));
      console.log(chalk.gray(`   버전: ${response.data.version || 'Unknown'}`));

    } catch (error) {
      if (error.code === 'ECONNREFUSED') {
        console.log(chalk.red('❌ Ollama 서비스가 실행되지 않고 있습니다.'));
        console.log(chalk.yellow('다음 명령어로 Ollama를 시작하세요:'));
        console.log(chalk.white('  ollama serve'));
        throw new Error('Ollama 서비스를 먼저 시작해주세요.');
      }
      throw error;
    }
  }

  async checkExistingModels() {
    console.log(chalk.yellow('📦 설치된 모델 확인 중...'));

    try {
      const response = await axios.get(`${this.ollamaEndpoint}/api/tags`);
      const installedModels = response.data.models || [];
      const installedNames = installedModels.map(m => m.name);

      console.log(chalk.blue(`현재 설치된 모델: ${installedNames.length}개`));

      for (const model of this.models) {
        if (installedNames.includes(model.name)) {
          console.log(chalk.green(`✅ ${model.name} - 이미 설치됨`));
          this.installedModels.push(model.name);
        } else {
          console.log(chalk.yellow(`⏳ ${model.name} - 설치 필요`));
        }
      }

    } catch (error) {
      console.error(chalk.red('모델 목록 확인 실패:'), error.message);
      throw error;
    }
  }

  async installModels() {
    const modelsToInstall = this.models.filter(model =>
      !this.installedModels.includes(model.name)
    );

    if (modelsToInstall.length === 0) {
      console.log(chalk.green('✅ 모든 필요한 모델이 이미 설치되어 있습니다.'));
      return;
    }

    console.log(chalk.yellow(`🔄 ${modelsToInstall.length}개 모델 설치 시작...`));

    for (const model of modelsToInstall) {
      try {
        console.log(chalk.cyan(`\n📥 ${model.name} 설치 중...`));
        console.log(chalk.gray(`   설명: ${model.description}`));
        console.log(chalk.gray(`   예상 크기: ${model.estimatedSize}`));
        console.log(chalk.gray(`   메모리 사용량: ${model.memoryUsage}`));

        await this.installSingleModel(model);
        
        console.log(chalk.green(`✅ ${model.name} 설치 완료`));
        this.installedModels.push(model.name);

      } catch (error) {
        console.error(chalk.red(`❌ ${model.name} 설치 실패:`, error.message));
        this.failedModels.push({
          name: model.name,
          error: error.message
        });
      }
    }
  }

  async installSingleModel(model) {
    return new Promise((resolve, reject) => {
      const pullProcess = spawn('ollama', ['pull', model.name], {
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let output = '';
      let hasProgress = false;
      let lastProgress = '';

      pullProcess.stdout.on('data', (data) => {
        const text = data.toString();
        output += text;

        // 진행률 표시
        const progressMatch = text.match(/(\d+)%/);
        if (progressMatch) {
          const progress = progressMatch[1];
          if (progress !== lastProgress) {
            process.stdout.write(`\r${chalk.blue(`   진행률: ${progress}%`)}`);
            lastProgress = progress;
            hasProgress = true;
          }
        }

        // 상태 메시지 표시
        if (text.includes('pulling')) {
          process.stdout.write(`\r${chalk.yellow('   다운로드 중...')}`);
        } else if (text.includes('verifying')) {
          process.stdout.write(`\r${chalk.yellow('   검증 중...')}`);
        } else if (text.includes('writing')) {
          process.stdout.write(`\r${chalk.yellow('   파일 생성 중...')}`);
        } else if (text.includes('success')) {
          process.stdout.write(`\r${chalk.green('   다운로드 완료!')}`);
        }
      });

      pullProcess.stderr.on('data', (data) => {
        const errorText = data.toString();
        if (!errorText.includes('progress')) {
          console.error(chalk.red(`   오류: ${errorText.trim()}`));
        }
      });

      pullProcess.on('close', (code) => {
        if (hasProgress) {
          console.log(); // 새 줄 추가
        }

        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`모델 설치 실패 (종료 코드: ${code})`));
        }
      });

      pullProcess.on('error', (error) => {
        reject(new Error(`프로세스 오류: ${error.message}`));
      });
    });
  }

  async testInstalledModels() {
    console.log(chalk.yellow('\n🧪 설치된 모델 테스트 중...'));

    for (const modelName of this.installedModels) {
      try {
        console.log(chalk.blue(`테스트 중: ${modelName}`));
        
        const testResult = await this.testSingleModel(modelName);
        
        if (testResult.success) {
          console.log(chalk.green(`✅ ${modelName} 테스트 성공`));
          console.log(chalk.gray(`   응답 시간: ${testResult.responseTime}ms`));
        } else {
          console.log(chalk.red(`❌ ${modelName} 테스트 실패`));
        }

      } catch (error) {
        console.log(chalk.red(`❌ ${modelName} 테스트 오류:`, error.message));
      }
    }
  }

  async testSingleModel(modelName) {
    const startTime = Date.now();
    
    try {
      const response = await axios.post(`${this.ollamaEndpoint}/api/generate`, {
        model: modelName,
        prompt: 'Hello, respond with just "OK"',
        stream: false
      }, {
        timeout: 30000
      });

      const responseTime = Date.now() - startTime;
      
      return {
        success: true,
        responseTime: responseTime,
        response: response.data.response
      };

    } catch (error) {
      return {
        success: false,
        error: error.message,
        responseTime: Date.now() - startTime
      };
    }
  }

  async updateConfigFiles() {
    console.log(chalk.yellow('⚙️ 설정 파일 업데이트 중...'));

    try {
      // 모델 설정 파일 생성
      const modelConfig = {
        'llama-router': {
          model: 'llama3.2:3b',
          endpoint: this.ollamaEndpoint,
          temperature: 0.1,
          max_tokens: 512,
          role: 'routing',
          description: '빠른 의도 파악 및 라우팅 결정',
          memory_limit: '6GB',
          auto_unload: false,
          optimization: {
            context_length: 4096,
            batch_size: 1,
            num_threads: 4
          }
        },
        'qwen-processor': {
          model: 'qwen2.5:14b',
          endpoint: this.ollamaEndpoint,
          temperature: 0.3,
          max_tokens: 2048,
          role: 'processing',
          description: '복잡한 코드 생성 및 분석 작업',
          memory_limit: '28GB',
          auto_unload: true,
          auto_unload_timeout: 600000,
          optimization: {
            context_length: 8192,
            batch_size: 1,
            num_threads: 8
          }
        }
      };

      // models 디렉토리 생성
      await fs.mkdir('./models', { recursive: true });

      // 설정 파일 저장
      await fs.writeFile(
        './models/model-configs.json',
        JSON.stringify(modelConfig, null, 2)
      );

      console.log(chalk.green('✅ 설정 파일 생성 완료'));

    } catch (error) {
      console.error(chalk.red('설정 파일 생성 실패:'), error.message);
    }
  }

  showCompletionMessage() {
    console.log(chalk.green('\n🎉 모델 설치 완료!'));
    console.log(chalk.gray('=' * 50));

    // 설치 요약
    console.log(chalk.cyan('📊 설치 요약:'));
    console.log(chalk.white(`   성공: ${this.installedModels.length}개 모델`));
    console.log(chalk.white(`   실패: ${this.failedModels.length}개 모델`));

    // 설치된 모델 목록
    if (this.installedModels.length > 0) {
      console.log(chalk.green('\n✅ 설치된 모델:'));
      this.installedModels.forEach(model => {
        const modelInfo = this.models.find(m => m.name === model);
        console.log(chalk.white(`   • ${model} (${modelInfo?.role || 'unknown'})`));
      });
    }

    // 실패한 모델 목록
    if (this.failedModels.length > 0) {
      console.log(chalk.red('\n❌ 실패한 모델:'));
      this.failedModels.forEach(model => {
        console.log(chalk.white(`   • ${model.name}: ${model.error}`));
      });
    }

    // 다음 단계 안내
    console.log(chalk.yellow('\n🚀 다음 단계:'));
    console.log(chalk.white('   1. npm run cli 또는 node mcp-cli.js 실행'));
    console.log(chalk.white('   2. "안녕하세요" 명령으로 테스트'));
    console.log(chalk.white('   3. "상태 확인해주세요" 명령으로 시스템 상태 확인'));

    // 사용 가능한 명령어
    console.log(chalk.blue('\n💡 사용 가능한 명령어:'));
    console.log(chalk.white('   • npm run cli    - MCP CLI 실행'));
    console.log(chalk.white('   • npm run start  - MCP 서버 실행'));
    console.log(chalk.white('   • npm run test   - 통합 테스트'));

    // 문제 해결 정보
    if (this.failedModels.length > 0) {
      console.log(chalk.yellow('\n🔧 문제 해결:'));
      console.log(chalk.white('   • 실패한 모델은 수동으로 설치 가능: ollama pull <model-name>'));
      console.log(chalk.white('   • 디스크 공간 부족 시 불필요한 파일 삭제'));
      console.log(chalk.white('   • 메모리 부족 시 다른 프로그램 종료'));
    }

    console.log(chalk.gray('\n모델 설치 스크립트 완료'));
  }
}

// 메인 실행
async function main() {
  const setup = new ModelSetup();
  await setup.run();
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
process.on('SIGINT', () => {
  console.log(chalk.yellow('\n설치 중단됨'));
  process.exit(0);
});

main().catch(console.error);
