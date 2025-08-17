// tools/visualization/plot-manager.js - 플롯 관리 및 최적화 도구
import { Logger } from '../../utils/logger.js';
import { ConfigLoader } from '../../utils/config-loader.js';
import { FileManager } from '../common/file-manager.js';
import { ChartGenerator } from './chart-generator.js';
import { PythonExecutor } from '../common/python-executor.js';
import fs from 'fs/promises';
import path from 'path';

export class PlotManager {
  constructor() {
    this.logger = new Logger();
    this.configLoader = new ConfigLoader();
    this.fileManager = new FileManager();
    this.chartGenerator = new ChartGenerator();
    this.pythonExecutor = new PythonExecutor();
    this.plotCache = new Map();
    this.plotMetadata = new Map();
    this.activeJobs = new Map();
    this.outputDirectory = './results/plots';
    this.cacheDirectory = './cache/plots';
    this.maxCacheSize = 500; // MB
    this.maxConcurrentJobs = 5;
    
    this.initializeManager();
  }

  async initializeManager() {
    try {
      await this.configLoader.initialize();
      await this.setupDirectories();
      await this.loadExistingPlots();
      this.logger.info('PlotManager 초기화 완료');
    } catch (error) {
      this.logger.error('PlotManager 초기화 실패:', error);
    }
  }

  async setupDirectories() {
    try {
      await this.fileManager.createDirectory(this.outputDirectory);
      await this.fileManager.createDirectory(this.cacheDirectory);
      await this.fileManager.createDirectory(path.join(this.outputDirectory, 'thumbnails'));
      this.logger.debug('디렉토리 설정 완료');
    } catch (error) {
      this.logger.error('디렉토리 설정 실패:', error);
      throw error;
    }
  }

  async loadExistingPlots() {
    try {
      const plotFiles = await this.scanPlotDirectory();
      let loadedCount = 0;

      for (const plotFile of plotFiles) {
        try {
          const metadata = await this.extractPlotMetadata(plotFile);
          if (metadata) {
            this.plotMetadata.set(plotFile, metadata);
            loadedCount++;
          }
        } catch (error) {
          this.logger.warn(`플롯 메타데이터 로드 실패: ${plotFile}`, error);
        }
      }

      this.logger.info(`기존 플롯 ${loadedCount}개 로드 완료`);
    } catch (error) {
      this.logger.error('기존 플롯 로드 실패:', error);
    }
  }

  async createPlot(plotConfig, options = {}) {
    const {
      cache_enabled = true,
      force_regenerate = false,
      priority = 'normal',
      callback = null
    } = options;

    try {
      const plotId = this.generatePlotId(plotConfig);
      this.logger.info(`플롯 생성 요청: ${plotId}`);

      // 캐시 확인
      if (cache_enabled && !force_regenerate) {
        const cachedPlot = await this.getCachedPlot(plotId);
        if (cachedPlot) {
          this.logger.info(`캐시된 플롯 반환: ${plotId}`);
          return cachedPlot;
        }
      }

      // 동시 실행 제한 확인
      if (this.activeJobs.size >= this.maxConcurrentJobs) {
        if (priority === 'high') {
          await this.waitForJobSlot();
        } else {
          throw new Error('동시 실행 가능한 작업 수 초과');
        }
      }

      // 작업 시작
      const jobPromise = this.executePlotGeneration(plotId, plotConfig, options);
      this.activeJobs.set(plotId, jobPromise);

      try {
        const result = await jobPromise;
        
        // 캐시에 저장
        if (cache_enabled) {
          await this.cachePlot(plotId, result);
        }

        // 메타데이터 저장
        await this.savePlotMetadata(plotId, result, plotConfig);

        // 콜백 실행
        if (callback) {
          try {
            await callback(result);
          } catch (callbackError) {
            this.logger.warn('플롯 생성 콜백 실행 실패:', callbackError);
          }
        }

        return result;

      } finally {
        this.activeJobs.delete(plotId);
      }

    } catch (error) {
      this.logger.error('플롯 생성 실패:', error);
      throw error;
    }
  }

  async executePlotGeneration(plotId, plotConfig, options) {
    const { chart_type, data, ...chartOptions } = plotConfig;

    switch (chart_type) {
      case 'scatter':
        return await this.chartGenerator.createScatterPlot(data, chartOptions);
      
      case 'line':
        return await this.chartGenerator.createLinePlot(data, chartOptions);
      
      case 'bar':
        return await this.chartGenerator.createBarChart(data, chartOptions);
      
      case 'histogram':
        return await this.chartGenerator.createHistogram(data, chartOptions);
      
      case 'boxplot':
        return await this.chartGenerator.createBoxPlot(data, chartOptions);
      
      case 'heatmap':
        return await this.chartGenerator.createHeatmap(data, chartOptions);
      
      case 'scatter_3d':
        return await this.chartGenerator.create3DScatterPlot(data, chartOptions);
      
      case 'interactive':
        return await this.chartGenerator.createInteractivePlot(data, chartOptions);
      
      case 'statistical':
        return await this.chartGenerator.createStatisticalPlots(data, chartOptions);
      
      case 'multiple':
        return await this.chartGenerator.createMultipleCharts(data, chartOptions.chart_configs, chartOptions);
      
      case 'auto':
        return await this.chartGenerator.autoGenerateCharts(data, chartOptions);
      
      default:
        throw new Error(`지원하지 않는 차트 타입: ${chart_type}`);
    }
  }

  async batchCreatePlots(plotConfigs, options = {}) {
    const {
      max_parallel = 3,
      fail_fast = false,
      progress_callback = null
    } = options;

    try {
      this.logger.info(`배치 플롯 생성 시작: ${plotConfigs.length}개 플롯`);

      const results = [];
      const semaphore = new Array(max_parallel).fill(null);
      let completedCount = 0;

      const createPlotWithSemaphore = async (plotConfig, index) => {
        try {
          const result = await this.createPlot(plotConfig);
          results[index] = { success: true, result };
        } catch (error) {
          results[index] = { success: false, error: error.message };
          if (fail_fast) {
            throw error;
          }
        } finally {
          completedCount++;
          if (progress_callback) {
            progress_callback(completedCount, plotConfigs.length);
          }
        }
      };

      // 병렬 실행
      const promises = plotConfigs.map((config, index) => 
        createPlotWithSemaphore(config, index)
      );

      await Promise.all(promises);

      const successCount = results.filter(r => r.success).length;
      this.logger.info(`배치 플롯 생성 완료: ${successCount}/${plotConfigs.length} 성공`);

      return {
        total: plotConfigs.length,
        successful: successCount,
        failed: plotConfigs.length - successCount,
        results
      };

    } catch (error) {
      this.logger.error('배치 플롯 생성 실패:', error);
      throw error;
    }
  }

  async optimizePlot(plotPath, options = {}) {
    const {
      quality = 85,
      max_size = { width: 1920, height: 1080 },
      format = 'png',
      compression = true
    } = options;

    try {
      this.logger.info(`플롯 최적화 시작: ${plotPath}`);

      const scriptPath = 'python/visualization/plot_optimizer.py';
      const params = {
        input_path: plotPath,
        quality,
        max_width: max_size.width,
        max_height: max_size.height,
        output_format: format,
        compression
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const optimizationResult = JSON.parse(result.output);
        this.logger.info('플롯 최적화 완료');
        return optimizationResult;
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('플롯 최적화 실패:', error);
      throw error;
    }
  }

  async createThumbnail(plotPath, options = {}) {
    const {
      size = { width: 200, height: 150 },
      quality = 80
    } = options;

    try {
      const thumbnailPath = path.join(
        this.outputDirectory, 
        'thumbnails', 
        `thumb_${path.basename(plotPath)}`
      );

      const scriptPath = 'python/visualization/thumbnail_generator.py';
      const params = {
        input_path: plotPath,
        output_path: thumbnailPath,
        width: size.width,
        height: size.height,
        quality
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        this.logger.debug(`썸네일 생성 완료: ${thumbnailPath}`);
        return thumbnailPath;
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.warn('썸네일 생성 실패:', error);
      return null;
    }
  }

  async generatePlotReport(plotPaths, options = {}) {
    const {
      include_thumbnails = true,
      include_metadata = true,
      output_format = 'html',
      template = 'default'
    } = options;

    try {
      this.logger.info(`플롯 리포트 생성 시작: ${plotPaths.length}개 플롯`);

      const reportData = {
        plots: [],
        generation_time: new Date().toISOString(),
        total_plots: plotPaths.length
      };

      for (const plotPath of plotPaths) {
        const plotInfo = {
          path: plotPath,
          filename: path.basename(plotPath)
        };

        if (include_metadata) {
          plotInfo.metadata = this.plotMetadata.get(plotPath) || {};
        }

        if (include_thumbnails) {
          plotInfo.thumbnail = await this.createThumbnail(plotPath);
        }

        reportData.plots.push(plotInfo);
      }

      const scriptPath = 'python/visualization/report_generator.py';
      const params = {
        report_data: JSON.stringify(reportData),
        output_format,
        template,
        output_dir: this.outputDirectory
      };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        const reportResult = JSON.parse(result.output);
        this.logger.info('플롯 리포트 생성 완료');
        return reportResult;
      } else {
        throw new Error(result.error);
      }

    } catch (error) {
      this.logger.error('플롯 리포트 생성 실패:', error);
      throw error;
    }
  }

  async cleanupCache() {
    try {
      this.logger.info('캐시 정리 시작');

      const cacheSize = await this.getCacheSize();
      
      if (cacheSize > this.maxCacheSize * 1024 * 1024) { // MB to bytes
        const cacheFiles = await this.getCacheFiles();
        
        // 접근 시간 기준으로 정렬 (오래된 것부터)
        cacheFiles.sort((a, b) => a.accessTime - b.accessTime);
        
        let deletedSize = 0;
        const targetSize = this.maxCacheSize * 0.7 * 1024 * 1024; // 70%까지 삭제
        
        for (const file of cacheFiles) {
          if (deletedSize >= (cacheSize - targetSize)) break;
          
          try {
            await fs.unlink(file.path);
            this.plotCache.delete(file.id);
            deletedSize += file.size;
          } catch (error) {
            this.logger.warn(`캐시 파일 삭제 실패: ${file.path}`, error);
          }
        }
        
        this.logger.info(`캐시 정리 완료: ${Math.round(deletedSize / 1024 / 1024)}MB 삭제`);
      }

    } catch (error) {
      this.logger.error('캐시 정리 실패:', error);
    }
  }

  generatePlotId(plotConfig) {
    const configString = JSON.stringify(plotConfig, Object.keys(plotConfig).sort());
    const crypto = require('crypto');
    return crypto.createHash('md5').update(configString).digest('hex');
  }

  async getCachedPlot(plotId) {
    if (this.plotCache.has(plotId)) {
      return this.plotCache.get(plotId);
    }

    const cachePath = path.join(this.cacheDirectory, `${plotId}.json`);
    
    try {
      const cacheData = await fs.readFile(cachePath, 'utf-8');
      const cachedPlot = JSON.parse(cacheData);
      this.plotCache.set(plotId, cachedPlot);
      return cachedPlot;
    } catch (error) {
      return null;
    }
  }

  async cachePlot(plotId, plotResult) {
    try {
      const cachePath = path.join(this.cacheDirectory, `${plotId}.json`);
      await fs.writeFile(cachePath, JSON.stringify(plotResult, null, 2));
      this.plotCache.set(plotId, plotResult);
    } catch (error) {
      this.logger.warn('플롯 캐시 저장 실패:', error);
    }
  }

  async savePlotMetadata(plotId, plotResult, plotConfig) {
    try {
      const metadata = {
        id: plotId,
        created_at: new Date().toISOString(),
        chart_type: plotConfig.chart_type,
        config: plotConfig,
        file_path: plotResult.results?.chart_path || null,
        file_size: null,
        dimensions: null,
        execution_time: plotResult.metadata?.execution_time || null
      };

      // 파일 정보 추가
      if (metadata.file_path && await this.fileManager.exists(metadata.file_path)) {
        try {
          const stats = await fs.stat(metadata.file_path);
          metadata.file_size = stats.size;
          
          // 이미지 차원 정보 (가능한 경우)
          const dimensions = await this.getImageDimensions(metadata.file_path);
          if (dimensions) {
            metadata.dimensions = dimensions;
          }
        } catch (error) {
          this.logger.warn('파일 정보 추출 실패:', error);
        }
      }

      this.plotMetadata.set(plotId, metadata);
      
      // 메타데이터 파일로 저장
      const metadataPath = path.join(this.outputDirectory, `${plotId}_metadata.json`);
      await fs.writeFile(metadataPath, JSON.stringify(metadata, null, 2));

    } catch (error) {
      this.logger.warn('플롯 메타데이터 저장 실패:', error);
    }
  }

  async getImageDimensions(imagePath) {
    try {
      const scriptPath = 'python/visualization/image_info.py';
      const params = { image_path: imagePath };

      const result = await this.pythonExecutor.executeScript(scriptPath, params);

      if (result.success) {
        return JSON.parse(result.output);
      }
    } catch (error) {
      this.logger.debug('이미지 차원 정보 추출 실패:', error);
    }
    return null;
  }

  async scanPlotDirectory() {
    try {
      const files = await fs.readdir(this.outputDirectory);
      return files
        .filter(file => /\.(png|jpg|jpeg|svg|pdf|html)$/i.test(file))
        .map(file => path.join(this.outputDirectory, file));
    } catch (error) {
      this.logger.warn('플롯 디렉토리 스캔 실패:', error);
      return [];
    }
  }

  async extractPlotMetadata(plotFile) {
    const basename = path.basename(plotFile, path.extname(plotFile));
    const metadataPath = path.join(this.outputDirectory, `${basename}_metadata.json`);
    
    try {
      const metadataContent = await fs.readFile(metadataPath, 'utf-8');
      return JSON.parse(metadataContent);
    } catch (error) {
      // 메타데이터 파일이 없으면 기본 정보만 생성
      try {
        const stats = await fs.stat(plotFile);
        return {
          file_path: plotFile,
          created_at: stats.birthtime.toISOString(),
          file_size: stats.size,
          chart_type: 'unknown'
        };
      } catch (statError) {
        return null;
      }
    }
  }

  async getCacheSize() {
    try {
      const files = await fs.readdir(this.cacheDirectory);
      let totalSize = 0;

      for (const file of files) {
        try {
          const filePath = path.join(this.cacheDirectory, file);
          const stats = await fs.stat(filePath);
          totalSize += stats.size;
        } catch (error) {
          // 파일 접근 실패 시 무시
        }
      }

      return totalSize;
    } catch (error) {
      this.logger.warn('캐시 크기 계산 실패:', error);
      return 0;
    }
  }

  async getCacheFiles() {
    try {
      const files = await fs.readdir(this.cacheDirectory);
      const cacheFiles = [];

      for (const file of files) {
        try {
          const filePath = path.join(this.cacheDirectory, file);
          const stats = await fs.stat(filePath);
          
          cacheFiles.push({
            id: path.basename(file, '.json'),
            path: filePath,
            size: stats.size,
            accessTime: stats.atime.getTime(),
            modifiedTime: stats.mtime.getTime()
          });
        } catch (error) {
          // 파일 접근 실패 시 무시
        }
      }

      return cacheFiles;
    } catch (error) {
      this.logger.warn('캐시 파일 목록 조회 실패:', error);
      return [];
    }
  }

  async waitForJobSlot() {
    return new Promise((resolve) => {
      const checkSlot = () => {
        if (this.activeJobs.size < this.maxConcurrentJobs) {
          resolve();
        } else {
          setTimeout(checkSlot, 100);
        }
      };
      checkSlot();
    });
  }

  // 플롯 관리 메서드들
  async listPlots(options = {}) {
    const {
      chart_type = null,
      created_after = null,
      created_before = null,
      sort_by = 'created_at',
      sort_order = 'desc',
      limit = 50
    } = options;

    try {
      let plots = Array.from(this.plotMetadata.values());

      // 필터링
      if (chart_type) {
        plots = plots.filter(plot => plot.chart_type === chart_type);
      }

      if (created_after) {
        const afterDate = new Date(created_after);
        plots = plots.filter(plot => new Date(plot.created_at) > afterDate);
      }

      if (created_before) {
        const beforeDate = new Date(created_before);
        plots = plots.filter(plot => new Date(plot.created_at) < beforeDate);
      }

      // 정렬
      plots.sort((a, b) => {
        let aValue = a[sort_by];
        let bValue = b[sort_by];

        if (sort_by === 'created_at') {
          aValue = new Date(aValue).getTime();
          bValue = new Date(bValue).getTime();
        }

        if (sort_order === 'desc') {
          return bValue - aValue;
        } else {
          return aValue - bValue;
        }
      });

      // 제한
      return plots.slice(0, limit);

    } catch (error) {
      this.logger.error('플롯 목록 조회 실패:', error);
      return [];
    }
  }

  async deletePlot(plotId) {
    try {
      this.logger.info(`플롯 삭제: ${plotId}`);

      const metadata = this.plotMetadata.get(plotId);
      if (!metadata) {
        throw new Error(`플롯을 찾을 수 없습니다: ${plotId}`);
      }

      // 실제 파일 삭제
      if (metadata.file_path && await this.fileManager.exists(metadata.file_path)) {
        await fs.unlink(metadata.file_path);
      }

      // 메타데이터 파일 삭제
      const metadataPath = path.join(this.outputDirectory, `${plotId}_metadata.json`);
      if (await this.fileManager.exists(metadataPath)) {
        await fs.unlink(metadataPath);
      }

      // 캐시 삭제
      const cachePath = path.join(this.cacheDirectory, `${plotId}.json`);
      if (await this.fileManager.exists(cachePath)) {
        await fs.unlink(cachePath);
      }

      // 메모리에서 제거
      this.plotMetadata.delete(plotId);
      this.plotCache.delete(plotId);

      this.logger.info('플롯 삭제 완료');
      return true;

    } catch (error) {
      this.logger.error('플롯 삭제 실패:', error);
      throw error;
    }
  }

  async getPlotInfo(plotId) {
    const metadata = this.plotMetadata.get(plotId);
    if (!metadata) {
      return null;
    }

    // 추가 정보 수집
    const info = { ...metadata };

    // 파일 존재 여부 확인
    if (info.file_path) {
      info.file_exists = await this.fileManager.exists(info.file_path);
    }

    // 캐시 상태 확인
    info.is_cached = this.plotCache.has(plotId);

    return info;
  }

  // 성능 모니터링 및 통계
  getManagerStatistics() {
    const totalPlots = this.plotMetadata.size;
    const cachedPlots = this.plotCache.size;
    const activeJobs = this.activeJobs.size;

    const chartTypes = {};
    for (const metadata of this.plotMetadata.values()) {
      chartTypes[metadata.chart_type] = (chartTypes[metadata.chart_type] || 0) + 1;
    }

    return {
      total_plots: totalPlots,
      cached_plots: cachedPlots,
      active_jobs: activeJobs,
      cache_hit_rate: totalPlots > 0 ? (cachedPlots / totalPlots) : 0,
      chart_type_distribution: chartTypes,
      output_directory: this.outputDirectory,
      cache_directory: this.cacheDirectory
    };
  }

  async getStorageUsage() {
    try {
      const outputSize = await this.getDirectorySize(this.outputDirectory);
      const cacheSize = await this.getCacheSize();

      return {
        output_directory_size_mb: Math.round(outputSize / 1024 / 1024 * 100) / 100,
        cache_directory_size_mb: Math.round(cacheSize / 1024 / 1024 * 100) / 100,
        total_size_mb: Math.round((outputSize + cacheSize) / 1024 / 1024 * 100) / 100,
        max_cache_size_mb: this.maxCacheSize
      };
    } catch (error) {
      this.logger.error('저장소 사용량 계산 실패:', error);
      return null;
    }
  }

  async getDirectorySize(dirPath) {
    try {
      const files = await fs.readdir(dirPath, { withFileTypes: true });
      let totalSize = 0;

      for (const file of files) {
        const filePath = path.join(dirPath, file.name);
        
        if (file.isDirectory()) {
          totalSize += await this.getDirectorySize(filePath);
        } else {
          const stats = await fs.stat(filePath);
          totalSize += stats.size;
        }
      }

      return totalSize;
    } catch (error) {
      return 0;
    }
  }

  // 설정 관리
  updateConfiguration(newConfig) {
    if (newConfig.maxCacheSize) {
      this.maxCacheSize = newConfig.maxCacheSize;
    }
    
    if (newConfig.maxConcurrentJobs) {
      this.maxConcurrentJobs = newConfig.maxConcurrentJobs;
    }

    if (newConfig.outputDirectory) {
      this.outputDirectory = newConfig.outputDirectory;
    }

    this.logger.info('PlotManager 설정 업데이트 완료');
  }

  getConfiguration() {
    return {
      maxCacheSize: this.maxCacheSize,
      maxConcurrentJobs: this.maxConcurrentJobs,
      outputDirectory: this.outputDirectory,
      cacheDirectory: this.cacheDirectory
    };
  }
}