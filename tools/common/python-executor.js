// tools/common/python-executor.js
import { spawn } from 'child_process';
import { Logger } from '../../utils/logger.js';
import fs from 'fs/promises';
import path from 'path';

export class PythonExecutor {
  constructor() {
    this.logger = new Logger();
    this.pythonPath = 'python3';
    this.venvPath = './python-env';
    this.tempDir = './temp';
    this.maxExecutionTime = 300000; // 5분
    this.executionStats = {
      totalExecutions: 0,
      successfulExecutions: 0,
      failedExecutions: 0,
      averageExecutionTime: 0
    };
  }

  async initialize() {
    try {
      await this.setupEnvironment();
      await this.installDependencies();
      await this.validatePythonEnvironment();
      this.logger.info('Python 실행기 초기화 완료');
    } catch (error) {
      this.logger.error('Python 실행기 초기화 실패:', error);
      throw error;
    }
  }

  async setupEnvironment() {
    try {
      // 임시 디렉토리 생성
      await fs.mkdir(this.tempDir, { recursive: true });
      
      // 가상환경 확인 및 설정
      try {
        await fs.access(path.join(this.venvPath, 'bin', 'python'));
        this.pythonPath = path.join(this.venvPath, 'bin', 'python');
        this.logger.info('가상환경 사용:', this.pythonPath);
      } catch {
        // Windows 환경 확인
        try {
          await fs.access(path.join(this.venvPath, 'Scripts', 'python.exe'));
          this.pythonPath = path.join(this.venvPath, 'Scripts', 'python.exe');
          this.logger.info('Windows 가상환경 사용:', this.pythonPath);
        } catch {
          this.logger.info('가상환경을 찾을 수 없습니다. 시스템 Python 사용');
        }
      }
    } catch (error) {
      this.logger.error('환경 설정 실패:', error);
      throw error;
    }
  }

  async installDependencies() {
    const requiredPackages = [
      'pandas>=1.3.0',
      'numpy>=1.21.0',
      'scikit-learn>=1.0.0',
      'matplotlib>=3.5.0',
      'seaborn>=0.11.0',
      'plotly>=5.0.0',
      'openpyxl>=3.0.0',
      'xlrd>=2.0.0',
      'pyarrow>=5.0.0',
      'h5py>=3.0.0',
      'xgboost>=1.5.0',
      'lightgbm>=3.3.0',
      'torch>=1.12.0',
      'transformers>=4.20.0',
      'datasets>=2.0.0'
    ];

    for (const pkg of requiredPackages) {
      try {
        await this.runCommand(`${this.pythonPath} -m pip install ${pkg}`, { timeout: 60000 });
        this.logger.debug(`패키지 설치 완료: ${pkg}`);
      } catch (error) {
        this.logger.warn(`패키지 설치 실패: ${pkg}`, error);
      }
    }
  }

  async execute(code, options = {}) {
    const {
      timeout = this.maxExecutionTime,
      workingDir = this.tempDir,
      env = {},
      captureOutput = true,
      streamOutput = false
    } = options;

    const startTime = Date.now();
    this.executionStats.totalExecutions++;

    try {
      // 임시 스크립트 파일 생성
      const scriptPath = await this.createTempScript(code);
      
      // Python 실행
      const result = await this.runPythonScript(scriptPath, {
        timeout,
        workingDir,
        env,
        captureOutput,
        streamOutput
      });
      
      // 임시 파일 정리
      await this.cleanupTempFile(scriptPath);
      
      // 통계 업데이트
      const executionTime = Date.now() - startTime;
      this.updateExecutionStats(executionTime, true);
      
      return result;
    } catch (error) {
      this.updateExecutionStats(Date.now() - startTime, false);
      this.logger.error('Python 코드 실행 실패:', error);
      throw error;
    }
  }

  async createTempScript(code) {
    const timestamp = Date.now();
    const random = Math.random().toString(36).substring(2, 8);
    const scriptPath = path.join(this.tempDir, `script_${timestamp}_${random}.py`);
    
    const wrappedCode = `
import sys
import traceback
import json
import warnings
import os
from pathlib import Path

# 경고 무시
warnings.filterwarnings('ignore')

# 작업 디렉토리 설정
os.chdir('${this.tempDir}')

# 결과 저장을 위한 함수
def save_result(result, filename='result.json'):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        print(f"결과 저장 실패: {e}")

# 플롯 저장을 위한 함수
def save_plot(filename='plot.png'):
    try:
        import matplotlib.pyplot as plt
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename
    except Exception as e:
        print(f"플롯 저장 실패: {e}")
        return None

try:
${code.split('\n').map(line => '    ' + line).join('\n')}
except Exception as e:
    error_info = {
        'error': str(e),
        'type': type(e).__name__,
        'traceback': traceback.format_exc(),
        'line_number': traceback.extract_tb(e.__traceback__)[-1].lineno if e.__traceback__ else None
    }
    print(json.dumps(error_info, ensure_ascii=False, indent=2))
    sys.exit(1)
`;

    await fs.writeFile(scriptPath, wrappedCode, 'utf-8');
    return scriptPath;
  }

  async runPythonScript(scriptPath, options = {}) {
    const {
      timeout = this.maxExecutionTime,
      workingDir = this.tempDir,
      env = {},
      captureOutput = true,
      streamOutput = false
    } = options;

    return new Promise((resolve, reject) => {
      const process = spawn(this.pythonPath, [scriptPath], {
        cwd: workingDir,
        env: { ...process.env, ...env },
        stdio: streamOutput ? 'inherit' : ['pipe', 'pipe', 'pipe']
      });

      let stdout = '';
      let stderr = '';

      if (captureOutput && !streamOutput) {
        process.stdout.on('data', (data) => {
          stdout += data.toString();
        });

        process.stderr.on('data', (data) => {
          stderr += data.toString();
        });
      }

      // 타임아웃 설정
      const timeoutId = setTimeout(() => {
        process.kill('SIGTERM');
        reject(new Error(`Python 실행 시간 초과: ${timeout}ms`));
      }, timeout);

      process.on('close', (code) => {
        clearTimeout(timeoutId);
        
        if (code === 0) {
          resolve({
            success: true,
            output: stdout,
            error: stderr,
            exitCode: code
          });
        } else {
          try {
            // JSON 형태의 에러 정보 파싱 시도
            const errorInfo = JSON.parse(stdout);
            reject(new Error(`Python 실행 오류: ${errorInfo.error}\n위치: 라인 ${errorInfo.line_number}`));
          } catch {
            reject(new Error(`Python 실행 실패 (코드: ${code})\n출력: ${stdout}\n에러: ${stderr}`));
          }
        }
      });

      process.on('error', (error) => {
        clearTimeout(timeoutId);
        reject(new Error(`Python 프로세스 오류: ${error.message}`));
      });
    });
  }

  async runCommand(command, options = {}) {
    const { timeout = 30000, workingDir = this.tempDir } = options;

    return new Promise((resolve, reject) => {
      const process = spawn('sh', ['-c', command], {
        cwd: workingDir,
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let stdout = '';
      let stderr = '';

      process.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      process.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      const timeoutId = setTimeout(() => {
        process.kill('SIGTERM');
        reject(new Error(`명령 실행 시간 초과: ${timeout}ms`));
      }, timeout);

      process.on('close', (code) => {
        clearTimeout(timeoutId);
        
        if (code === 0) {
          resolve(stdout);
        } else {
          reject(new Error(`명령 실행 실패 (코드: ${code}): ${stderr}`));
        }
      });

      process.on('error', (error) => {
        clearTimeout(timeoutId);
        reject(new Error(`명령 프로세스 오류: ${error.message}`));
      });
    });
  }

  async cleanupTempFile(filePath) {
    try {
      await fs.unlink(filePath);
      this.logger.debug(`임시 파일 삭제: ${filePath}`);
    } catch (error) {
      this.logger.warn('임시 파일 삭제 실패:', error);
    }
  }

  async validatePythonEnvironment() {
    try {
      const result = await this.runCommand(`${this.pythonPath} --version`);
      this.logger.info('Python 버전:', result.trim());
      
      // 기본 라이브러리 테스트
      const testCode = `
import sys
print(f"Python 버전: {sys.version}")
print(f"실행 경로: {sys.executable}")

# 주요 라이브러리 테스트
libraries = ['pandas', 'numpy', 'sklearn', 'matplotlib']
for lib in libraries:
    try:
        __import__(lib)
        print(f"✓ {lib} 사용 가능")
    except ImportError:
        print(f"✗ {lib} 사용 불가")
`;
      
      await this.execute(testCode);
      return true;
    } catch (error) {
      this.logger.error('Python 환경 검증 실패:', error);
      return false;
    }
  }

  updateExecutionStats(executionTime, success) {
    if (success) {
      this.executionStats.successfulExecutions++;
    } else {
      this.executionStats.failedExecutions++;
    }
    
    // 평균 실행 시간 계산
    const totalTime = this.executionStats.averageExecutionTime * (this.executionStats.totalExecutions - 1);
    this.executionStats.averageExecutionTime = (totalTime + executionTime) / this.executionStats.totalExecutions;
  }

  getExecutionStats() {
    return {
      ...this.executionStats,
      successRate: (this.executionStats.successfulExecutions / this.executionStats.totalExecutions * 100).toFixed(2) + '%',
      pythonPath: this.pythonPath,
      venvPath: this.venvPath,
      tempDir: this.tempDir,
      maxExecutionTime: this.maxExecutionTime
    };
  }

  async executeFile(filePath, options = {}) {
    try {
      const code = await fs.readFile(filePath, 'utf-8');
      return await this.execute(code, options);
    } catch (error) {
      this.logger.error('파일 실행 실패:', error);
      throw error;
    }
  }

  async executeWithDataFrame(code, dataPath, options = {}) {
    const enhancedCode = `
import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('${dataPath}')
print(f"데이터 형태: {df.shape}")
print(f"컬럼: {df.columns.tolist()}")

${code}
`;
    
    return await this.execute(enhancedCode, options);
  }

  async cleanupTempFiles() {
    try {
      const files = await fs.readdir(this.tempDir);
      const tempFiles = files.filter(file => file.startsWith('script_'));
      
      for (const file of tempFiles) {
        await fs.unlink(path.join(this.tempDir, file));
      }
      
      this.logger.info(`임시 파일 정리 완료: ${tempFiles.length}개 파일 삭제`);
    } catch (error) {
      this.logger.warn('임시 파일 정리 실패:', error);
    }
  }

  async shutdown() {
    await this.cleanupTempFiles();
    this.logger.info('Python 실행기 종료');
  }
}
