import { Logger } from '../../utils/logger.js';
import fs from 'fs/promises';
import path from 'path';

export class FileManager {
  constructor() {
    this.logger = new Logger();
    this.allowedExtensions = ['.csv', '.xlsx', '.json', '.txt', '.parquet', '.h5'];
    this.maxFileSize = 100 * 1024 * 1024; // 100MB
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
        isValid: true
      };
    } catch (error) {
      this.logger.error('파일 검증 실패:', error);
      throw error;
    }
  }

  async readFile(filePath, options = {}) {
    await this.validateFile(filePath);
    
    const { encoding = 'utf8' } = options;
    
    try {
      const data = await fs.readFile(filePath, encoding);
      this.logger.info(`파일 읽기 완료: ${filePath}`);
      return data;
    } catch (error) {
      this.logger.error('파일 읽기 실패:', error);
      throw error;
    }
  }

  async writeFile(filePath, data, options = {}) {
    const { encoding = 'utf8' } = options;
    
    try {
      const dir = path.dirname(filePath);
      await fs.mkdir(dir, { recursive: true });
      
      await fs.writeFile(filePath, data, encoding);
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
      
      return filteredFiles.map(file => ({
        name: file,
        path: path.join(directory, file),
        extension: path.extname(file)
      }));
    } catch (error) {
      this.logger.error('디렉토리 읽기 실패:', error);
      return [];
    }
  }

  async deleteFile(filePath) {
    try {
      await fs.unlink(filePath);
      this.logger.info(`파일 삭제 완료: ${filePath}`);
      return true;
    } catch (error) {
      this.logger.error('파일 삭제 실패:', error);
      throw error;
    }
  }

  async copyFile(sourcePath, targetPath) {
    try {
      await this.validateFile(sourcePath);
      
      const dir = path.dirname(targetPath);
      await fs.mkdir(dir, { recursive: true });
      
      await fs.copyFile(sourcePath, targetPath);
      this.logger.info(`파일 복사 완료: ${sourcePath} -> ${targetPath}`);
      return true;
    } catch (error) {
      this.logger.error('파일 복사 실패:', error);
      throw error;
    }
  }

  async moveFile(sourcePath, targetPath) {
    try {
      await this.copyFile(sourcePath, targetPath);
      await this.deleteFile(sourcePath);
      this.logger.info(`파일 이동 완료: ${sourcePath} -> ${targetPath}`);
      return true;
    } catch (error) {
      this.logger.error('파일 이동 실패:', error);
      throw error;
    }
  }

  getFileInfo(filePath) {
    return {
      name: path.basename(filePath),
      directory: path.dirname(filePath),
      extension: path.extname(filePath),
      nameWithoutExt: path.basename(filePath, path.extname(filePath))
    };
  }

  generateUniqueFileName(directory, baseName, extension) {
    let counter = 1;
    let fileName = `${baseName}${extension}`;
    let fullPath = path.join(directory, fileName);
    
    while (fs.access(fullPath).then(() => true).catch(() => false)) {
      fileName = `${baseName}_${counter}${extension}`;
      fullPath = path.join(directory, fileName);
      counter++;
    }
    
    return fileName;
  }
}

