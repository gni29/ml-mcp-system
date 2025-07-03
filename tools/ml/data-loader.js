import { Logger } from '../../utils/logger.js';
import { FileManager } from '../common/file-manager.js';
import fs from 'fs/promises';
import path from 'path';

export class DataLoader {
  constructor() {
    this.logger = new Logger();
    this.fileManager = new FileManager();
    this.supportedFormats = ['.csv', '.xlsx', '.json', '.parquet'];
  }

  async loadData(filePath, options = {}) {
    try {
      await this.fileManager.validateFile(filePath);
      
      const extension = path.extname(filePath).toLowerCase();
      
      switch (extension) {
        case '.csv':
          return await this.loadCSV(filePath, options);
        case '.xlsx':
          return await this.loadExcel(filePath, options);
        case '.json':
          return await this.loadJSON(filePath, options);
        default:
          throw new Error(`지원하지 않는 파일 형식: ${extension}`);
      }
    } catch (error) {
      this.logger.error('데이터 로드 실패:', error);
      throw error;
    }
  }

  async loadCSV(filePath, options = {}) {
    const { encoding = 'utf8', separator = ',' } = options;
    
    try {
      const data = await fs.readFile(filePath, encoding);
      const lines = data.split('\n').filter(line => line.trim());
      
      if (lines.length === 0) {
        throw new Error('빈 파일입니다.');
      }

      const headers = lines[0].split(separator).map(h => h.trim().replace(/"/g, ''));
      const rows = lines.slice(1).map(line => {
        const values = line.split(separator).map(v => v.trim().replace(/"/g, ''));
        const row = {};
        headers.forEach((header, index) => {
          row[header] = values[index] || null;
        });
        return row;
      });

      return {
        data: rows,
        headers: headers,
        rowCount: rows.length,
        columnCount: headers.length,
        filePath: filePath,
        fileType: 'csv'
      };
    } catch (error) {
      throw new Error(`CSV 파일 로드 실패: ${error.message}`);
    }
  }

  async loadJSON(filePath, options = {}) {
    const { encoding = 'utf8' } = options;
    
    try {
      const data = await fs.readFile(filePath, encoding);
      const jsonData = JSON.parse(data);
      
      let processedData;
      if (Array.isArray(jsonData)) {
        processedData = jsonData;
      } else if (typeof jsonData === 'object' && jsonData !== null) {
        processedData = [jsonData];
      } else {
        throw new Error('JSON 데이터가 객체 또는 배열이 아닙니다.');
      }

      const headers = Object.keys(processedData[0] || {});

      return {
        data: processedData,
        headers: headers,
        rowCount: processedData.length,
        columnCount: headers.length,
        filePath: filePath,
        fileType: 'json'
      };
    } catch (error) {
      throw new Error(`JSON 파일 로드 실패: ${error.message}`);
    }
  }

  async loadExcel(filePath, options = {}) {
    // Excel 파일은 Python을 통해 처리
    const pythonCode = `
import pandas as pd
import json

df = pd.read_excel('${filePath}')
result = {
    'data': df.to_dict('records'),
    'headers': df.columns.tolist(),
    'rowCount': len(df),
    'columnCount': len(df.columns),
    'filePath': '${filePath}',
    'fileType': 'excel'
}
print(json.dumps(result, default=str))
`;

    try {
      const pythonExecutor = new (await import('../common/python-executor.js')).PythonExecutor();
      await pythonExecutor.initialize();
      
      const result = await pythonExecutor.execute(pythonCode);
      if (result.success) {
        return JSON.parse(result.output);
      } else {
        throw new Error(result.error);
      }
    } catch (error) {
      throw new Error(`Excel 파일 로드 실패: ${error.message}`);
    }
  }

  async getDataInfo(filePath) {
    try {
      const data = await this.loadData(filePath);
      
      return {
        fileName: path.basename(filePath),
        fileType: data.fileType,
        rowCount: data.rowCount,
        columnCount: data.columnCount,
        headers: data.headers,
        sampleData: data.data.slice(0, 5), // 처음 5행 샘플
        fileSize: (await fs.stat(filePath)).size
      };
    } catch (error) {
      this.logger.error('데이터 정보 조회 실패:', error);
      throw error;
    }
  }

  async previewData(filePath, limit = 10) {
    try {
      const data = await this.loadData(filePath);
      
      return {
        headers: data.headers,
        preview: data.data.slice(0, limit),
        totalRows: data.rowCount,
        showingRows: Math.min(limit, data.rowCount)
      };
    } catch (error) {
      this.logger.error('데이터 미리보기 실패:', error);
      throw error;
    }
  }

  validateDataStructure(data) {
    const issues = [];
    
    if (!data.data || data.data.length === 0) {
      issues.push('데이터가 비어있습니다.');
    }
    
    if (!data.headers || data.headers.length === 0) {
      issues.push('헤더가 없습니다.');
    }
    
    // 데이터 타입 일관성 검사
    if (data.data.length > 0) {
      data.headers.forEach(header => {
        const values = data.data.map(row => row[header]).filter(v => v !== null && v !== '');
        if (values.length > 0) {
          const types = [...new Set(values.map(v => typeof v))];
          if (types.length > 1) {
            issues.push(`컬럼 '${header}'의 데이터 타입이 일관되지 않습니다.`);
          }
        }
      });
    }
    
    return {
      isValid: issues.length === 0,
      issues: issues
    };
  }
}
