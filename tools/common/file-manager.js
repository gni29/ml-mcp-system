import { Logger } from '../../utils/logger.js';
import fs from 'fs/promises';
import path from 'path';
import { createReadStream, createWriteStream } from 'fs';
import { pipeline } from 'stream/promises';

export class FileManager {
  constructor() {
    this.logger = new Logger();
    this.allowedExtensions = ['.csv', '.xlsx', '.json', '.txt', '.parquet', '.h5'];
    this.maxFileSize = 100 * 1024 * 1024; // 100MB
    this.backupDir = './backups';
    this.tempDir = './temp';
  }

  async validateFile(filePath) {
    try {
      const stat = await fs.stat(filePath);
      const ext = path.extname(filePath).toLowerCase();
      
      if (!this.allowedExtensions.includes(ext)) {
        throw new Error(`지원하지 않는 파일 형식: ${ext}`);
      }
      
      if (stat.size > this.maxFileSize) {
        throw new Error(`파일이 너무 큽니다: ${stat.size} bytes`);
      }
      
      return {
        size: stat.size,
        extension: ext,
        isValid: true,
        lastModified: stat.mtime,
        isDirectory: stat.isDirectory(),
        permissions: {
          readable: true,
          writable: true
        }
      };
    } catch (error) {
      this.logger.error('파일 검증 실패:', error);
      throw error;
    }
  }

  async readFile(filePath, options = {}) {
    await this.validateFile(filePath);
    
    const { encoding = 'utf8', flag = 'r' } = options;
    
    try {
      const data = await fs.readFile(filePath, { encoding, flag });
      this.logger.info(`파일 읽기 완료: ${filePath}`);
      return data;
    } catch (error) {
      this.logger.error('파일 읽기 실패:', error);
      throw error;
    }
  }

  async writeFile(filePath, data, options = {}) {
    const { encoding = 'utf8', flag = 'w', backup = false } = options;
    
    try {
      const dir = path.dirname(filePath);
      await fs.mkdir(dir, { recursive: true });
      
      // 백업 생성 (기존 파일이 있고 backup 옵션이 true인 경우)
      if (backup && await this.exists(filePath)) {
        await this.createBackup(filePath);
      }
      
      await fs.writeFile(filePath, data, { encoding, flag });
      this.logger.info(`파일 쓰기 완료: ${filePath}`);
      return true;
    } catch (error) {
      this.logger.error('파일 쓰기 실패:', error);
      throw error;
    }
  }

  async listFiles(directory, filter = null) {
    try {
      const files = await fs.readdir(directory);
      
      let filteredFiles = files;
      if (filter) {
        filteredFiles = files.filter(file => {
          const ext = path.extname(file).toLowerCase();
          return this.allowedExtensions.includes(ext);
        });
      }
      
      const fileStats = await Promise.all(
        filteredFiles.map(async (file) => {
          const filePath = path.join(directory, file);
          try {
            const stat = await fs.stat(filePath);
            return {
              name: file,
              path: filePath,
              extension: path.extname(file),
              size: stat.size,
              lastModified: stat.mtime,
              isDirectory: stat.isDirectory(),
              isFile: stat.isFile()
            };
          } catch (error) {
            this.logger.warn(`파일 정보 조회 실패: ${filePath}`, error);
            return {
              name: file,
              path: filePath,
              extension: path.extname(file),
              size: null,
              lastModified: null,
              isDirectory: false,
              isFile: true,
              error: error.message
            };
          }
        })
      );
      
      return fileStats;
    } catch (error) {
      this.logger.error('디렉토리 읽기 실패:', error);
      return [];
    }
  }

  async deleteFile(filePath) {
    try {
      const stat = await fs.stat(filePath);
      
      if (stat.isDirectory()) {
        await fs.rmdir(filePath, { recursive: true });
        this.logger.info(`디렉토리 삭제 완료: ${filePath}`);
      } else {
        await fs.unlink(filePath);
        this.logger.info(`파일 삭제 완료: ${filePath}`);
      }
      
      return true;
    } catch (error) {
      this.logger.error('파일 삭제 실패:', error);
      throw error;
    }
  }

  async copyFile(sourcePath, targetPath, options = {}) {
    const { overwrite = false, preserveTimestamps = true } = options;
    
    try {
      // 소스 파일 검증
      await this.validateFile(sourcePath);
      
      // 타겟 디렉토리 생성
      const targetDir = path.dirname(targetPath);
      await fs.mkdir(targetDir, { recursive: true });
      
      // 타겟 파일 존재 확인
      if (!overwrite && await this.exists(targetPath)) {
        throw new Error(`대상 파일이 이미 존재합니다: ${targetPath}`);
      }
      
      // 스트림을 사용한 효율적인 파일 복사
      const sourceStream = createReadStream(sourcePath);
      const targetStream = createWriteStream(targetPath);
      
      await pipeline(sourceStream, targetStream);
      
      // 타임스탬프 보존
      if (preserveTimestamps) {
        const sourceStat = await fs.stat(sourcePath);
        await fs.utimes(targetPath, sourceStat.atime, sourceStat.mtime);
      }
      
      this.logger.info(`파일 복사 완료: ${sourcePath} -> ${targetPath}`);
      return true;
    } catch (error) {
      this.logger.error('파일 복사 실패:', error);
      throw error;
    }
  }

  async moveFile(sourcePath, targetPath, options = {}) {
    const { overwrite = false } = options;
    
    try {
      // 소스 파일 검증
      await this.validateFile(sourcePath);
      
      // 타겟 디렉토리 생성
      const targetDir = path.dirname(targetPath);
      await fs.mkdir(targetDir, { recursive: true });
      
      // 타겟 파일 존재 확인
      if (!overwrite && await this.exists(targetPath)) {
        throw new Error(`대상 파일이 이미 존재합니다: ${targetPath}`);
      }
      
      // 파일 이동
      await fs.rename(sourcePath, targetPath);
      
      this.logger.info(`파일 이동 완료: ${sourcePath} -> ${targetPath}`);
      return true;
    } catch (error) {
      this.logger.error('파일 이동 실패:', error);
      throw error;
    }
  }

  async exists(filePath) {
    try {
      await fs.access(filePath);
      return true;
    } catch (error) {
      return false;
    }
  }

  async createDirectory(dirPath, options = {}) {
    const { recursive = true } = options;
    
    try {
      await fs.mkdir(dirPath, { recursive });
      this.logger.info(`디렉토리 생성 완료: ${dirPath}`);
      return true;
    } catch (error) {
      this.logger.error('디렉토리 생성 실패:', error);
      throw error;
    }
  }

  async createBackup(filePath) {
    try {
      await fs.mkdir(this.backupDir, { recursive: true });
      
      const fileName = path.basename(filePath);
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const backupFileName = `${fileName}.${timestamp}.bak`;
      const backupPath = path.join(this.backupDir, backupFileName);
      
      await this.copyFile(filePath, backupPath, { overwrite: true });
      
      this.logger.info(`백업 생성 완료: ${backupPath}`);
      return backupPath;
    } catch (error) {
      this.logger.error('백업 생성 실패:', error);
      throw error;
    }
  }

  async restoreBackup(backupPath, targetPath) {
    try {
      if (!await this.exists(backupPath)) {
        throw new Error(`백업 파일이 존재하지 않습니다: ${backupPath}`);
      }
      
      await this.copyFile(backupPath, targetPath, { overwrite: true });
      
      this.logger.info(`백업 복원 완료: ${backupPath} -> ${targetPath}`);
      return true;
    } catch (error) {
      this.logger.error('백업 복원 실패:', error);
      throw error;
    }
  }

  async getFileInfo(filePath) {
    try {
      const stat = await fs.stat(filePath);
      const ext = path.extname(filePath).toLowerCase();
      
      return {
        name: path.basename(filePath),
        path: filePath,
        extension: ext,
        size: stat.size,
        sizeFormatted: this.formatFileSize(stat.size),
        lastModified: stat.mtime,
        lastAccessed: stat.atime,
        created: stat.birthtime,
        isDirectory: stat.isDirectory(),
        isFile: stat.isFile(),
        permissions: {
          readable: true,
          writable: true,
          executable: false
        },
        mimeType: this.getMimeType(ext)
      };
    } catch (error) {
      this.logger.error('파일 정보 조회 실패:', error);
      throw error;
    }
  }

  async cleanupTempFiles() {
    try {
      if (await this.exists(this.tempDir)) {
        const tempFiles = await this.listFiles(this.tempDir);
        
        for (const file of tempFiles) {
          if (file.isFile) {
            await this.deleteFile(file.path);
          }
        }
        
        this.logger.info('임시 파일 정리 완료');
      }
    } catch (error) {
      this.logger.error('임시 파일 정리 실패:', error);
    }
  }

  async watchFile(filePath, callback) {
    try {
      const watcher = fs.watch(filePath, { encoding: 'utf8' }, (eventType, filename) => {
        callback(eventType, filename, filePath);
      });
      
      this.logger.info(`파일 감시 시작: ${filePath}`);
      return watcher;
    } catch (error) {
      this.logger.error('파일 감시 실패:', error);
      throw error;
    }
  }

  async findFiles(directory, pattern) {
    try {
      const files = await this.listFiles(directory);
      const regex = new RegExp(pattern, 'i');
      
      return files.filter(file => regex.test(file.name));
    } catch (error) {
      this.logger.error('파일 검색 실패:', error);
      return [];
    }
  }

  async getDirectorySize(directory) {
    try {
      const files = await this.listFiles(directory);
      let totalSize = 0;
      
      for (const file of files) {
        if (file.isFile && file.size) {
          totalSize += file.size;
        } else if (file.isDirectory) {
          totalSize += await this.getDirectorySize(file.path);
        }
      }
      
      return totalSize;
    } catch (error) {
      this.logger.error('디렉토리 크기 계산 실패:', error);
      return 0;
    }
  }

  formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  getMimeType(extension) {
    const mimeTypes = {
      '.csv': 'text/csv',
      '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      '.xls': 'application/vnd.ms-excel',
      '.json': 'application/json',
      '.txt': 'text/plain',
      '.parquet': 'application/octet-stream',
      '.h5': 'application/octet-stream'
    };
    
    return mimeTypes[extension.toLowerCase()] || 'application/octet-stream';
  }

  async ensureDirectoryExists(dirPath) {
    try {
      if (!await this.exists(dirPath)) {
        await this.createDirectory(dirPath);
      }
      return true;
    } catch (error) {
      this.logger.error('디렉토리 확인/생성 실패:', error);
      throw error;
    }
  }

  async createTempFile(prefix = 'temp', extension = '.tmp') {
    try {
      await this.ensureDirectoryExists(this.tempDir);
      
      const timestamp = Date.now();
      const random = Math.random().toString(36).substring(2, 8);
      const tempFileName = `${prefix}_${timestamp}_${random}${extension}`;
      const tempFilePath = path.join(this.tempDir, tempFileName);
      
      // 빈 파일 생성
      await fs.writeFile(tempFilePath, '', 'utf8');
      
      this.logger.info(`임시 파일 생성: ${tempFilePath}`);
      return tempFilePath;
    } catch (error) {
      this.logger.error('임시 파일 생성 실패:', error);
      throw error;
    }
  }

  async isFileEmpty(filePath) {
    try {
      const stat = await fs.stat(filePath);
      return stat.size === 0;
    } catch (error) {
      this.logger.error('파일 비어있음 확인 실패:', error);
      return false;
    }
  }

  async getFileHash(filePath, algorithm = 'sha256') {
    try {
      const { createHash } = await import('crypto');
      const hash = createHash(algorithm);
      const data = await this.readFile(filePath);
      
      hash.update(data);
      return hash.digest('hex');
    } catch (error) {
      this.logger.error('파일 해시 계산 실패:', error);
      throw error;
    }
  }

  async compareFiles(filePath1, filePath2) {
    try {
      const hash1 = await this.getFileHash(filePath1);
      const hash2 = await this.getFileHash(filePath2);
      
      return hash1 === hash2;
    } catch (error) {
      this.logger.error('파일 비교 실패:', error);
      throw error;
    }
  }
}
