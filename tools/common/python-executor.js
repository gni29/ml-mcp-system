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
  }

  async initialize() {
    try {
      await this.setupEnvironment();
      await this.installDependencies();
      this.logger.info('Python 실행기 초기화 완료');
    } catch (error) {
      this.logger.error('Python 실행기 초기화 실패:', error);
      throw error;
    }
  }

  async setupEnvironment() {
    try {
      await fs.mkdir(this.tempDir, { recursive: true });
      
      // 가상환경 확인
      try {
        await fs.access(path.join(this.venvPath, 'bin', 'python'));
        this.pythonPath = path.join(this.venvPath, 'bin', 'python');
      } catch {
        this.logger.info('가상환경을 찾을 수 없습니다. 시스템 Python 사용');
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
      'seaborn>=0.11.0'
    ];

    for (const pkg of requiredPackages) {
      try {
        await this.runCommand(`${this.pythonPath} -m pip install ${pkg}`, { timeout: 30000 });
      } catch (error) {
        this.logger.warn(`패키지 설치 실패: ${pkg}`, error);
      }
    }
  }

  async execute(code, options = {}) {
    const {
      timeout = this.maxExecutionTime,
      workingDir = this.tempDir,
      env = {}
    } = options;

    try {
      // 임시 스크립트 파일 생성
      const scriptPath = await this.createTempScript(code);
      
      // Python 실행
      const result = await this.runPythonScript(scriptPath, {
        timeout,
        workingDir,
        env
      });
      
      // 임시 파일 정리
      await this.cleanupTempFile(scriptPath);
      
      return result;
    } catch (error) {
      this.logger.error('Python 코드 실행 실패:', error);
      throw error;
    }
  }

  async createTempScript(code) {
    const timestamp = Date.now();
    const scriptPath = path.join(this.tempDir, `script_${timestamp}.py`);
    
    const wrappedCode = `
import sys
import traceback
import json
import warnings
warnings.filterwarnings('ignore')

try:
${code.split('\n').map(line => '    ' + line).join('\n')}
except Exception as e:
    error_info = {
        'error': str(e),
        'type': type(e).__name__,
        'traceback': traceback.format_exc()
    }
    print(json.dumps(error_info))
    sys.exit(1)
`;

    await fs.writeFile(scriptPath, wrappedCode);
    return scriptPath;
  }

  async runPythonScript(scriptPath, options = {}) {
    const {
      timeout = this.maxExecutionTime,
      workingDir = this.tempDir,
      env = {}
    } = options;

    return new Promise((resolve, reject) => {
      const process = spawn(this.pythonPath, [scriptPath], {
        cwd: workingDir,
        env: { ...process.env, ...env },
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let stdout = '';
      let stderr = '';

      process.on('close', (code) => {
        clearTimeout(timeoutId);
        
        if (code === 0) {
          resolve({
            success: true,
            output: stdout,
            error: stderr
          });
        } else {
          try {
            const errorInfo = JSON.parse(stdout);
            reject(new Error(`Python 실행 오류: ${errorInfo.error}`));
          } catch {
            reject(new Error(`Python 실행 실패 (코드: ${code}): ${stderr}`));
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
    const { timeout = 30000 } = options;

    return new Promise((resolve, reject) => {
      const process = spawn('sh', ['-c', command], {
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
    });
  }

  async cleanupTempFile(filePath) {
    try {
      await fs.unlink(filePath);
    } catch (error) {
      this.logger.warn('임시 파일 삭제 실패:', error);
    }
  }

  async validatePythonEnvironment() {
    try {
      const result = await this.runCommand(`${this.pythonPath} --version`);
      this.logger.info('Python 버전:', result.trim());
      return true;
    } catch (error) {
      this.logger.error('Python 환경 검증 실패:', error);
      return false;
    }
  }

  getExecutionStats() {
    return {
      pythonPath: this.pythonPath,
      venvPath: this.venvPath,
      tempDir: this.tempDir,
      maxExecutionTime: this.maxExecutionTime
    };
  }
}
