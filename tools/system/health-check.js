import { Logger } from '../../utils/logger.js';
import os from 'os';
import { spawn } from 'child_process';

export class HealthCheck {
  constructor() {
    this.logger = new Logger();
  }

  async performHealthCheck() {
    try {
      const checks = await Promise.allSettled([
        this.checkSystemResources(),
        this.checkOllamaService(),
        this.checkPythonEnvironment(),
        this.checkFileSystem(),
        this.checkNetworkConnectivity()
      ]);

      const results = checks.map((check, index) => ({
        name: ['시스템 리소스', 'Ollama 서비스', 'Python 환경', '파일 시스템', '네트워크'][index],
        status: check.status === 'fulfilled' ? 'passed' : 'failed',
        details: check.status === 'fulfilled' ? check.value : check.reason?.message || 'Unknown error'
      }));

      const overallHealth = results.every(r => r.status === 'passed') ? 'healthy' : 'unhealthy';

      return {
        overallHealth,
        checks: results,
        timestamp: new Date().toISOString(),
        systemInfo: this.getSystemInfo()
      };
    } catch (error) {
      this.logger.error('헬스체크 실패:', error);
      throw error;
    }
  }

  async checkSystemResources() {
    const totalMemory = os.totalmem();
    const freeMemory = os.freemem();
    const usedMemory = totalMemory - freeMemory;
    const memoryUsage = (usedMemory / totalMemory) * 100;

    const cpuUsage = await this.getCPUUsage();
    const loadAverage = os.loadavg();

    if (memoryUsage > 90) {
      throw new Error(`메모리 사용률이 높습니다: ${memoryUsage.toFixed(1)}%`);
    }

    if (cpuUsage > 90) {
      throw new Error(`CPU 사용률이 높습니다: ${cpuUsage.toFixed(1)}%`);
    }

    return {
      memory: {
        total: Math.round(totalMemory / 1024 / 1024 / 1024),
        used: Math.round(usedMemory / 1024 / 1024 / 1024),
        usage: memoryUsage.toFixed(1)
      },
      cpu: {
        usage: cpuUsage.toFixed(1),
        loadAverage: loadAverage.map(load => load.toFixed(2))
      }
    };
  }

  async getCPUUsage() {
    return new Promise((resolve) => {
      const startUsage = process.cpuUsage();
      const startTime = process.hrtime();

      setTimeout(() => {
        const endUsage = process.cpuUsage(startUsage);
        const endTime = process.hrtime(startTime);
        
        const totalTime = endTime[0] * 1000000 + endTime[1] / 1000;
        const cpuTime = (endUsage.user + endUsage.system);
        const usage = (cpuTime / totalTime) * 100;
        
        resolve(Math.min(usage, 100));
      }, 1000);
    });
  }

  async checkOllamaService() {
    return new Promise((resolve, reject) => {
      const process = spawn('curl', ['-s', 'http://localhost:11434/api/version'], {
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let output = '';
      process.stdout.on('data', (data) => {
        output += data.toString();
      });

      process.on('close', (code) => {
        if (code === 0) {
          try {
            const version = JSON.parse(output);
            resolve({ status: 'running', version: version.version });
          } catch {
            resolve({ status: 'running', version: 'unknown' });
          }
        } else {
          reject(new Error('Ollama 서비스가 실행되지 않고 있습니다.'));
        }
      });

      process.on('error', () => {
        reject(new Error('Ollama 서비스 확인 실패'));
      });
    });
  }

  async checkPythonEnvironment() {
    return new Promise((resolve, reject) => {
      const process = spawn('python3', ['--version'], {
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let output = '';
      process.stdout.on('data', (data) => {
        output += data.toString();
      });

      process.stderr.on('data', (data) => {
        output += data.toString();
      });

      process.on('close', (code) => {
        if (code === 0) {
          resolve({ version: output.trim() });
        } else {
          reject(new Error('Python 3가 설치되지 않았거나 PATH에 없습니다.'));
        }
      });

      process.on('error', () => {
        reject(new Error('Python 환경 확인 실패'));
      });
    });
  }

  async checkFileSystem() {
    const fs = await import('fs/promises');
    
    try {
      const directories = ['./temp', './results', './data', './logs'];
      const checks = [];

      for (const dir of directories) {
        try {
          await fs.access(dir);
          checks.push({ directory: dir, status: 'accessible' });
        } catch {
          await fs.mkdir(dir, { recursive: true });
          checks.push({ directory: dir, status: 'created' });
        }
      }

      return { directories: checks };
    } catch (error) {
      throw new Error(`파일 시스템 접근 오류: ${error.message}`);
    }
  }

  async checkNetworkConnectivity() {
    // 기본적인 네트워크 연결성 체크 (localhost)
    return new Promise((resolve, reject) => {
      const process = spawn('ping', ['-c', '1', '127.0.0.1'], {
        stdio: ['pipe', 'pipe', 'pipe']
      });

      process.on('close', (code) => {
        if (code === 0) {
          resolve({ localhost: 'reachable' });
        } else {
          reject(new Error('로컬 네트워크 연결 실패'));
        }
      });

      process.on('error', () => {
        reject(new Error('네트워크 연결성 확인 실패'));
      });
    });
  }

  getSystemInfo() {
    return {
      platform: os.platform(),
      arch: os.arch(),
      release: os.release(),
      hostname: os.hostname(),
      uptime: os.uptime(),
      nodeVersion: process.version,
      pid: process.pid
    };
  }

  formatHealthCheckResult(healthCheck) {
    const { overallHealth, checks, systemInfo } = healthCheck;
    
    const healthEmoji = overallHealth === 'healthy' ? '✅' : '⚠️';
    let result = `${healthEmoji} **시스템 상태: ${overallHealth === 'healthy' ? '정상' : '주의 필요'}**\n\n`;
    
    result += '### 상세 검사 결과\n';
    checks.forEach(check => {
      const statusEmoji = check.status === 'passed' ? '✅' : '❌';
      result += `${statusEmoji} **${check.name}**: ${check.status === 'passed' ? '정상' : '오류'}\n`;
      if (typeof check.details === 'object') {
        result += `   상세: ${JSON.stringify(check.details, null, 2)}\n`;
      } else {
        result += `   상세: ${check.details}\n`;
      }
    });

    result += '\n### 시스템 정보\n';
    result += `- 플랫폼: ${systemInfo.platform} ${systemInfo.arch}\n`;
    result += `- 호스트명: ${systemInfo.hostname}\n`;
    result += `- 가동시간: ${Math.floor(systemInfo.uptime / 3600)}시간\n`;
    result += `- Node.js: ${systemInfo.nodeVersion}\n`;

    return {
      content: [{
        type: 'text',
        text: result
      }],
      metadata: {
        overallHealth,
        timestamp: healthCheck.timestamp
      }
    };
  }
}

