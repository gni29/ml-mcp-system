// core/pipeline-manager.js
import { Logger } from '../utils/logger.js';
import { DataLoader } from '../tools/data/data-loader.js';
import { BasicAnalyzer } from '../tools/basic/basic-analyzer.js';
import { AdvancedAnalyzer } from '../tools/advanced/advanced-analyzer.js';
import { MLTrainer } from '../tools/ml/trainer.js';
import { Visualizer } from '../tools/visualization/visualizer.js';
import { ResultFormatter } from '../tools/common/result-formatter.js';
import fs from 'fs/promises';
import path from 'path';

export class PipelineManager {
  constructor() {
    this.logger = new Logger();
    this.isExecuting = false;
    this.currentSession = null;
    this.executionHistory = new Map();
    this.tools = {};
    this.resultFormatter = new ResultFormatter();
  }

  async initialize() {
    try {
      // 도구들 초기화
      this.tools = {
        data: new DataLoader(),
        basic: new BasicAnalyzer(),
        advanced: new AdvancedAnalyzer(),
        ml_traditional: new MLTrainer(),
        deep_learning: new MLTrainer(), // 딥러닝도 같은 트레이너 사용
        visualization: new Visualizer(),
        timeseries: new AdvancedAnalyzer(), // 시계열은 고급 분석기 사용
        preprocessing: new DataLoader(), // 전처리는 데이터 로더 사용
        postprocessing: new ResultFormatter()
      };

      // 각 도구 초기화
      for (const [name, tool] of Object.entries(this.tools)) {
        if (typeof tool.initialize === 'function') {
          await tool.initialize();
          this.logger.debug(`${name} 도구 초기화 완료`);
        }
      }

      this.logger.info('PipelineManager 초기화 완료');
    } catch (error) {
      this.logger.error('PipelineManager 초기화 실패:', error);
      throw error;
    }
  }

  async executeWorkflow(workflowData, sessionId, userQuery) {
    if (this.isExecuting) {
      throw new Error('다른 워크플로우가 실행 중입니다.');
    }

    this.isExecuting = true;
    this.currentSession = sessionId;

    try {
      this.logger.info('워크플로우 실행 시작', {
        sessionId,
        workflowName: workflowData.workflow.name
      });

      const startTime = Date.now();
      const results = {
        sessionId,
        workflowName: workflowData.workflow.name,
        userQuery,
        startTime: new Date().toISOString(),
        steps: [],
        intermediateResults: {},
        finalResult: null,
        status: 'running'
      };

      // 병렬 실행 그룹 처리
      const parallelGroups = workflowData.workflow.parallel_groups || [];
      let currentParallelGroup = -1;

      // 워크플로우 단계별 실행
      for (let i = 0; i < workflowData.workflow.steps.length; i++) {
        const step = workflowData.workflow.steps[i];
        
        // 병렬 그룹 확인
        const parallelGroup = parallelGroups.find(group => group.includes(i));
        
        if (parallelGroup && currentParallelGroup !== parallelGroups.indexOf(parallelGroup)) {
          // 새로운 병렬 그룹 시작
          currentParallelGroup = parallelGroups.indexOf(parallelGroup);
          const parallelSteps = parallelGroup.map(index => workflowData.workflow.steps[index]);
          const parallelResults = await this.executeParallelSteps(parallelSteps, results.intermediateResults);
          
          // 병렬 실행 결과 저장
          parallelResults.forEach((result, index) => {
            const stepIndex = parallelGroup[index];
            results.steps[stepIndex] = result;
            
            if (result.success) {
              results.intermediateResults[`step_${stepIndex + 1}`] = result.result;
              results.intermediateResults[`${parallelSteps[index].type}_${parallelSteps[index].method}`] = result.result;
            }
          });
          
          // 병렬 그룹의 모든 단계를 건너뛰기 위해 인덱스 조정
          i = Math.max(...parallelGroup);
        } else if (!parallelGroup) {
          // 순차 실행
          const stepResult = await this.executeStep(step, results.intermediateResults, i + 1);
          results.steps.push(stepResult);
          
          // 중간 결과 저장
          if (stepResult.success) {
            results.intermediateResults[`step_${i + 1}`] = stepResult.result;
            results.intermediateResults[`${step.type}_${step.method}`] = stepResult.result;
          }
        }
      }

      // 최종 결과 생성
      results.finalResult = await this.generateFinalResult(results);
      results.status = 'completed';
      results.endTime = new Date().toISOString();
      results.executionTime = Date.now() - startTime;

      // 결과 저장
      await this.saveWorkflowResults(results);

      this.logger.info('워크플로우 실행 완료', {
        sessionId,
        executionTime: results.executionTime
      });

      return results;

    } catch (error) {
      this.logger.error('워크플로우 실행 실패:', error);
      
      // 실패한 결과도 저장
      const failedResults = {
        sessionId,
        workflowName: workflowData.workflow?.name || 'Unknown',
        userQuery,
        startTime: new Date().toISOString(),
        status: 'failed',
        error: error.message,
        endTime: new Date().toISOString(),
        executionTime: Date.now() - (this.startTime || Date.now())
      };
      
      await this.saveWorkflowResults(failedResults);
      throw error;
    } finally {
      this.isExecuting = false;
      this.currentSession = null;
    }
  }

  async executeParallelSteps(steps, intermediateResults) {
    const promises = steps.map((step, index) => 
      this.executeStep(step, intermediateResults, index + 1)
    );

    try {
      const results = await Promise.allSettled(promises);
      return results.map((result, index) => {
        if (result.status === 'fulfilled') {
          return result.value;
        } else {
          this.logger.error(`병렬 단계 ${index + 1} 실행 실패:`, result.reason);
          return {
            success: false,
            error: result.reason.message,
            stepNumber: index + 1,
            type: steps[index].type,
            method: steps[index].method,
            executionTime: 0
          };
        }
      });
    } catch (error) {
      this.logger.error('병렬 단계 실행 실패:', error);
      throw error;
    }
  }

  async executeStep(step, intermediateResults, stepNumber) {
    const startTime = Date.now();
    
    try {
      this.logger.debug(`단계 ${stepNumber} 실행 시작: ${step.type}.${step.method}`);

      // 적절한 도구 선택
      const tool = this.selectTool(step.type);
      if (!tool) {
        throw new Error(`도구를 찾을 수 없습니다: ${step.type}`);
      }

      // 단계 실행
      let result;
      switch (step.type) {
        case 'data':
          result = await this.executeDataStep(tool, step, intermediateResults);
          break;
        case 'basic':
          result = await this.executeBasicStep(tool, step, intermediateResults);
          break;
        case 'advanced':
          result = await this.executeAdvancedStep(tool, step, intermediateResults);
          break;
        case 'ml_traditional':
        case 'deep_learning':
          result = await this.executeMLStep(tool, step, intermediateResults);
          break;
        case 'visualization':
          result = await this.executeVisualizationStep(tool, step, intermediateResults);
          break;
        case 'timeseries':
          result = await this.executeTimeSeriesStep(tool, step, intermediateResults);
          break;
        case 'preprocessing':
          result = await this.executePreprocessingStep(tool, step, intermediateResults);
          break;
        case 'postprocessing':
          result = await this.executePostprocessingStep(tool, step, intermediateResults);
          break;
        default:
          throw new Error(`지원하지 않는 단계 타입: ${step.type}`);
      }

      const executionTime = Date.now() - startTime;
      this.logger.debug(`단계 ${stepNumber} 실행 완료: ${executionTime}ms`);

      return {
        success: true,
        stepNumber,
        type: step.type,
        method: step.method,
        result,
        executionTime,
        timestamp: new Date().toISOString()
      };

    } catch (error) {
      const executionTime = Date.now() - startTime;
      this.logger.error(`단계 ${stepNumber} 실행 실패:`, error);

      return {
        success: false,
        stepNumber,
        type: step.type,
        method: step.method,
        error: error.message,
        executionTime,
        timestamp: new Date().toISOString()
      };
    }
  }

  selectTool(stepType) {
    return this.tools[stepType] || null;
  }

  async executeDataStep(tool, step, intermediateResults) {
    const { method, params } = step;
    
    switch (method) {
      case 'load':
        return await tool.loadData(params.file_path || params.file_type, params);
      case 'validate':
        const data = this.getDataFromResults(intermediateResults);
        return await tool.validateData(data, params);
      case 'transform':
        const dataToTransform = this.getDataFromResults(intermediateResults);
        return await tool.transformData(dataToTransform, params);
      default:
        throw new Error(`지원하지 않는 데이터 메서드: ${method}`);
    }
  }

  async executeBasicStep(tool, step, intermediateResults) {
    const { method, params } = step;
    const data = this.getDataFromResults(intermediateResults);
    
    switch (method) {
      case 'descriptive_stats':
        return await tool.calculateDescriptiveStats(data, params);
      case 'correlation':
        return await tool.calculateCorrelation(data, params);
      case 'distribution':
        return await tool.analyzeDistribution(data, params);
      default:
        throw new Error(`지원하지 않는 기본 분석 메서드: ${method}`);
    }
  }

  async executeAdvancedStep(tool, step, intermediateResults) {
    const { method, params } = step;
    const data = this.getDataFromResults(intermediateResults);
    
    switch (method) {
      case 'pca':
        return await tool.performPCA(data, params);
      case 'clustering':
        return await tool.performClustering(data, params);
      case 'outlier_detection':
        return await tool.detectOutliers(data, params);
      case 'feature_engineering':
        return await tool.performFeatureEngineering(data, params);
      default:
        throw new Error(`지원하지 않는 고급 분석 메서드: ${method}`);
    }
  }

  async executeMLStep(tool, step, intermediateResults) {
    const { method, params } = step;
    const data = this.getDataFromResults(intermediateResults);
    
    switch (method) {
      case 'classification':
        return await tool.trainClassificationModel(data, params);
      case 'regression':
        return await tool.trainRegressionModel(data, params);
      case 'ensemble':
        return await tool.trainEnsembleModel(data, params);
      case 'neural_network':
        return await tool.trainNeuralNetwork(data, params);
      case 'cnn':
        return await tool.trainCNN(data, params);
      case 'rnn':
        return await tool.trainRNN(data, params);
      default:
        throw new Error(`지원하지 않는 ML 메서드: ${method}`);
    }
  }

  async executeVisualizationStep(tool, step, intermediateResults) {
    const { method, params } = step;
    const data = this.getDataFromResults(intermediateResults);
    
    switch (method) {
      case 'scatter':
      case 'line':
      case 'bar':
      case 'histogram':
      case 'box':
        return await tool.createBasicChart(data, method, params);
      case 'heatmap':
        const correlationData = intermediateResults.correlation_matrix || 
                               this.extractCorrelationData(intermediateResults);
        return await tool.createHeatmap(correlationData, params);
      case '3d_scatter':
        return await tool.create3DScatter(data, params);
      case 'confusion_matrix':
        const mlResults = this.getMLResultsFromResults(intermediateResults);
        return await tool.createConfusionMatrix(mlResults, params);
      default:
        return await tool.createVisualization(data, method, params);
    }
  }

  async executeTimeSeriesStep(tool, step, intermediateResults) {
    const { method, params } = step;
    const data = this.getDataFromResults(intermediateResults);
    
    switch (method) {
      case 'trend_analysis':
        return await tool.analyzeTrend(data, params);
      case 'seasonality_analysis':
        return await tool.analyzeSeasonality(data, params);
      case 'forecasting':
        return await tool.performForecasting(data, params);
      case 'anomaly_detection':
        return await tool.detectAnomalies(data, params);
      default:
        throw new Error(`지원하지 않는 시계열 메서드: ${method}`);
    }
  }

  async executePreprocessingStep(tool, step, intermediateResults) {
    const { method, params } = step;
    const data = this.getDataFromResults(intermediateResults);
    
    switch (method) {
      case 'handle_missing_values':
        return await tool.handleMissingValues(data, params);
      case 'normalize':
        return await tool.normalizeData(data, params);
      case 'encode_categorical':
        return await tool.encodeCategorical(data, params);
      case 'feature_scaling':
        return await tool.scaleFeatures(data, params);
      case 'comprehensive':
        return await tool.comprehensivePreprocessing(data, params);
      default:
        throw new Error(`지원하지 않는 전처리 메서드: ${method}`);
    }
  }

  async executePostprocessingStep(tool, step, intermediateResults) {
    const { method, params } = step;
    
    switch (method) {
      case 'summarize_results':
        return await this.summarizeResults(intermediateResults, params);
      case 'generate_report':
        return await this.generateReport(intermediateResults, params);
      case 'export_results':
        return await this.exportResults(intermediateResults, params);
      default:
        throw new Error(`지원하지 않는 후처리 메서드: ${method}`);
    }
  }

  getDataFromResults(intermediateResults) {
    // 가장 최근의 데이터 로딩 결과 찾기
    for (const [key, result] of Object.entries(intermediateResults)) {
      if (key.includes('data_load') || key.includes('step_1')) {
        return result.data || result;
      }
    }
    
    // 데이터가 없으면 첫 번째 결과 반환
    const firstResult = Object.values(intermediateResults)[0];
    return firstResult?.data || firstResult || null;
  }

  extractCorrelationData(intermediateResults) {
    for (const [key, result] of Object.entries(intermediateResults)) {
      if (key.includes('correlation') && result.correlation_matrix) {
        return result.correlation_matrix;
      }
    }
    return null;
  }

  getMLResultsFromResults(intermediateResults) {
    for (const [key, result] of Object.entries(intermediateResults)) {
      if (key.includes('ml_') || key.includes('classification') || key.includes('regression')) {
        return result;
      }
    }
    return null;
  }
  async generateFinalResult(results) {
    try {
      const finalResult = {
        summary: this.generateSummary(results),
        outputs: this.collectOutputs(results),
        visualizations: this.collectVisualizations(results),
        statistics: this.collectStatistics(results),
        recommendations: this.generateRecommendations(results),
        artifacts: this.collectArtifacts(results),
        metadata: {
          workflowName: results.workflowName,
          sessionId: results.sessionId,
          userQuery: results.userQuery,
          executionTime: results.executionTime,
          timestamp: new Date().toISOString()
        }
      };

      return finalResult;
    } catch (error) {
      this.logger.error('최종 결과 생성 실패:', error);
      return {
        error: error.message,
        partialResults: results.intermediateResults,
        metadata: {
          sessionId: results.sessionId,
          error: true,
          timestamp: new Date().toISOString()
        }
      };
    }
  }

  generateSummary(results) {
    const { steps, executionTime, workflowName } = results;
    const successfulSteps = steps.filter(step => step.success).length;
    const failedSteps = steps.filter(step => !step.success).length;

    return {
      workflowName,
      totalSteps: steps.length,
      successfulSteps,
      failedSteps,
      executionTime: `${(executionTime / 1000).toFixed(2)}초`,
      successRate: `${((successfulSteps / steps.length) * 100).toFixed(1)}%`,
      status: failedSteps === 0 ? 'success' : successfulSteps > 0 ? 'partial' : 'failed'
    };
  }

  collectOutputs(results) {
    const outputs = {};
    
    results.steps.forEach((step, index) => {
      if (step.success && step.result) {
        outputs[`step_${index + 1}`] = {
          type: step.type,
          method: step.method,
          result: step.result,
          executionTime: step.executionTime,
          timestamp: step.timestamp
        };
      }
    });

    return outputs;
  }

  collectVisualizations(results) {
    const visualizations = [];
    
    results.steps.forEach((step, index) => {
      if (step.success && step.type === 'visualization' && step.result) {
        visualizations.push({
          stepNumber: index + 1,
          chartType: step.method,
          filePath: step.result.chart_path,
          description: step.result.description || `${step.method} 차트`,
          metadata: step.result.metadata || {}
        });
      }
    });

    return visualizations;
  }

  collectStatistics(results) {
    const statistics = {};
    
    results.steps.forEach((step, index) => {
      if (step.success &&
          (step.type === 'basic' || step.type === 'advanced') &&
          step.result &&
          step.result.statistics) {
        statistics[`${step.type}_${step.method}`] = step.result.statistics;
      }
    });

    return statistics;
  }

  generateRecommendations(results) {
    const recommendations = [];
    
    // 데이터 품질 기반 권장사항
    const dataQuality = this.assessDataQuality(results);
    if (dataQuality.missingValues > 0.1) {
      recommendations.push({
        type: 'data_quality',
        message: '데이터에 누락값이 많습니다. 데이터 전처리를 고려해보세요.',
        severity: 'warning',
        action: 'preprocessing'
      });
    }

    // 모델 성능 기반 권장사항
    const modelPerformance = this.assessModelPerformance(results);
    if (modelPerformance.accuracy && modelPerformance.accuracy < 0.8) {
      recommendations.push({
        type: 'model_performance',
        message: '모델 성능이 낮습니다. 하이퍼파라미터 튜닝을 고려해보세요.',
        severity: 'warning',
        action: 'hyperparameter_tuning',
        currentPerformance: modelPerformance.accuracy
      });
    }

    // 추가 분석 제안
    const analysisGaps = this.identifyAnalysisGaps(results);
    recommendations.push(...analysisGaps);

    // 시각화 제안
    const vizSuggestions = this.suggestAdditionalVisualizations(results);
    recommendations.push(...vizSuggestions);

    return recommendations;
  }

  identifyAnalysisGaps(results) {
    const suggestions = [];
    const performedAnalyses = new Set();
    
    // 수행된 분석 수집
    results.steps.forEach(step => {
      if (step.success) {
        performedAnalyses.add(`${step.type}_${step.method}`);
      }
    });

    // 기본 분석 확인
    if (!performedAnalyses.has('basic_correlation')) {
      suggestions.push({
        type: 'analysis_gap',
        message: '변수 간 상관관계 분석을 추가해보세요.',
        severity: 'info',
        action: 'add_correlation_analysis'
      });
    }

    if (!performedAnalyses.has('basic_distribution')) {
      suggestions.push({
        type: 'analysis_gap',
        message: '데이터 분포 분석을 추가해보세요.',
        severity: 'info',
        action: 'add_distribution_analysis'
      });
    }

    // 고급 분석 제안
    if (performedAnalyses.has('basic_descriptive_stats') && 
        !performedAnalyses.has('advanced_outlier_detection')) {
      suggestions.push({
        type: 'analysis_enhancement',
        message: '이상치 탐지 분석을 통해 데이터 품질을 향상시킬 수 있습니다.',
        severity: 'info',
        action: 'add_outlier_detection'
      });
    }

    return suggestions;
  }

  suggestAdditionalVisualizations(results) {
    const suggestions = [];
    const hasVisualizations = results.steps.some(step => 
      step.success && step.type === 'visualization'
    );

    if (!hasVisualizations) {
      suggestions.push({
        type: 'visualization_missing',
        message: '결과를 더 잘 이해하기 위해 시각화를 추가하는 것을 권장합니다.',
        severity: 'info',
        action: 'add_visualization'
      });
    }

    // 특정 분석에 대한 시각화 제안
    const hasCorrelation = results.steps.some(step => 
      step.success && step.type === 'basic' && step.method === 'correlation'
    );
    const hasCorrelationViz = results.steps.some(step => 
      step.success && step.type === 'visualization' && step.method === 'heatmap'
    );

    if (hasCorrelation && !hasCorrelationViz) {
      suggestions.push({
        type: 'visualization_suggestion',
        message: '상관관계 결과를 히트맵으로 시각화하면 패턴을 더 쉽게 파악할 수 있습니다.',
        severity: 'info',
        action: 'add_correlation_heatmap'
      });
    }

    return suggestions;
  }

  collectArtifacts(results) {
    const artifacts = [];
    
    results.steps.forEach((step, index) => {
      if (step.success && step.result) {
        // 생성된 파일들 수집
        if (step.result.chart_path) {
          artifacts.push({
            type: 'visualization',
            name: `${step.method}_chart`,
            path: step.result.chart_path,
            stepNumber: index + 1,
            description: `${step.type} ${step.method} 차트`,
            createdAt: step.timestamp
          });
        }
        
        if (step.result.model_path) {
          artifacts.push({
            type: 'model',
            name: `${step.method}_model`,
            path: step.result.model_path,
            stepNumber: index + 1,
            description: `${step.type} ${step.method} 모델`,
            createdAt: step.timestamp
          });
        }
        
        if (step.result.report_path) {
          artifacts.push({
            type: 'report',
            name: `${step.method}_report`,
            path: step.result.report_path,
            stepNumber: index + 1,
            description: `${step.type} ${step.method} 리포트`,
            createdAt: step.timestamp
          });
        }

        if (step.result.data_path) {
          artifacts.push({
            type: 'data',
            name: `${step.method}_data`,
            path: step.result.data_path,
            stepNumber: index + 1,
            description: `${step.type} ${step.method} 처리된 데이터`,
            createdAt: step.timestamp
          });
        }
      }
    });

    return artifacts;
  }

  assessDataQuality(results) {
    let missingValues = 0;
    let dataShape = null;
    let duplicateRows = 0;
    let qualityScore = 1.0;
    
    // 데이터 로딩 단계 결과 찾기
    const dataLoadStep = results.steps.find(step =>
      (step.type === 'data' && step.method === 'load' && step.success) ||
      (step.type === 'basic' && step.method === 'descriptive_stats' && step.success)
    );
    
    if (dataLoadStep && dataLoadStep.result) {
      const result = dataLoadStep.result;
      
      // 누락값 비율
      if (result.statistics && result.statistics.missing_percentage) {
        missingValues = result.statistics.missing_percentage / 100;
      } else if (result.data_info && result.data_info.null_counts) {
        const totalCells = result.data_info.shape[0] * result.data_info.shape[1];
        const totalNulls = Object.values(result.data_info.null_counts)
          .reduce((sum, count) => sum + count, 0);
        missingValues = totalNulls / totalCells;
      }

      // 데이터 형태
      if (result.data_info && result.data_info.shape) {
        dataShape = {
          rows: result.data_info.shape[0],
          columns: result.data_info.shape[1]
        };
      }

      // 중복 행 (있다면)
      if (result.statistics && result.statistics.duplicate_rows) {
        duplicateRows = result.statistics.duplicate_rows;
      }
    }
    
    // 품질 점수 계산
    if (missingValues > 0.3) qualityScore -= 0.4;
    else if (missingValues > 0.1) qualityScore -= 0.2;
    else if (missingValues > 0.05) qualityScore -= 0.1;

    if (duplicateRows > 0) {
      const duplicateRatio = dataShape ? duplicateRows / dataShape.rows : 0;
      if (duplicateRatio > 0.1) qualityScore -= 0.2;
      else if (duplicateRatio > 0.05) qualityScore -= 0.1;
    }
    
    return {
      missingValues,
      dataShape,
      duplicateRows,
      qualityScore: Math.max(qualityScore, 0),
      quality: qualityScore > 0.8 ? 'high' : qualityScore > 0.6 ? 'medium' : 'low'
    };
  }

  assessModelPerformance(results) {
    let accuracy = null;
    let metrics = {};
    let modelType = null;
    let performance = 'unknown';
    
    // ML 단계 결과 찾기
    const mlSteps = results.steps.filter(step =>
      (step.type === 'ml_traditional' || step.type === 'deep_learning') && step.success
    );
    
    if (mlSteps.length > 0) {
      const lastMlStep = mlSteps[mlSteps.length - 1];
      if (lastMlStep.result && lastMlStep.result.metrics) {
        metrics = lastMlStep.result.metrics;
        modelType = lastMlStep.method;
        
        // 분류 모델 성능 평가
        if (lastMlStep.method === 'classification') {
          accuracy = metrics.accuracy || metrics.f1_score || null;
          if (accuracy > 0.9) performance = 'excellent';
          else if (accuracy > 0.8) performance = 'good';
          else if (accuracy > 0.7) performance = 'fair';
          else performance = 'poor';
        }
        
        // 회귀 모델 성능 평가
        else if (lastMlStep.method === 'regression') {
          accuracy = metrics.r2_score || null;
          if (accuracy > 0.9) performance = 'excellent';
          else if (accuracy > 0.8) performance = 'good';
          else if (accuracy > 0.6) performance = 'fair';
          else performance = 'poor';
        }
      }
    }
    
    return {
      accuracy,
      metrics,
      modelType,
      performance,
      hasModel: mlSteps.length > 0
    };
  }

  async summarizeResults(intermediateResults, params) {
    const summary = {
      totalSteps: Object.keys(intermediateResults).length,
      analysisTypes: new Set(),
      keyFindings: [],
      dataOverview: null
    };

    // 수행된 분석 유형 수집
    for (const [key, result] of Object.entries(intermediateResults)) {
      if (key.includes('basic_')) summary.analysisTypes.add('기본 통계 분석');
      if (key.includes('advanced_')) summary.analysisTypes.add('고급 분석');
      if (key.includes('ml_')) summary.analysisTypes.add('머신러닝');
      if (key.includes('visualization')) summary.analysisTypes.add('시각화');
    }

    // 주요 발견사항 추출
    const correlationResult = intermediateResults.basic_correlation;
    if (correlationResult && correlationResult.strong_correlations) {
      summary.keyFindings.push({
        type: 'correlation',
        message: `${correlationResult.strong_correlations.length}개의 강한 상관관계 발견`,
        details: correlationResult.strong_correlations.slice(0, 3)
      });
    }

    const clusteringResult = Object.values(intermediateResults).find(r => 
      r.cluster_labels && r.cluster_centers
    );
    if (clusteringResult) {
      const uniqueClusters = new Set(clusteringResult.cluster_labels).size;
      summary.keyFindings.push({
        type: 'clustering',
        message: `${uniqueClusters}개의 클러스터로 데이터 그룹화`,
        details: { clusters: uniqueClusters, silhouette_score: clusteringResult.metrics?.silhouette_score }
      });
    }

    summary.analysisTypes = Array.from(summary.analysisTypes);

    return {
      content: [{
        type: 'text',
        text: this.formatSummaryText(summary)
      }],
      analysisType: 'summary',
      summary
    };
  }

  formatSummaryText(summary) {
    let text = '📊 **분석 결과 요약**\n\n';
    
    text += `### 수행된 분석\n`;
    text += `- 총 ${summary.totalSteps}개 단계 수행\n`;
    text += `- 분석 유형: ${summary.analysisTypes.join(', ')}\n\n`;
    
    if (summary.keyFindings.length > 0) {
      text += `### 주요 발견사항\n`;
      summary.keyFindings.forEach((finding, index) => {
        text += `${index + 1}. ${finding.message}\n`;
      });
    }
    
    return text;
  }

  async generateReport(intermediateResults, params) {
    const reportFormat = params.format || 'basic';
    const includeCharts = params.include_charts || false;
    
    const report = {
      title: '데이터 분석 리포트',
      createdAt: new Date().toISOString(),
      sections: []
    };

    // 데이터 개요 섹션
    const dataOverview = this.generateDataOverviewSection(intermediateResults);
    if (dataOverview) report.sections.push(dataOverview);

    // 분석 결과 섹션들
    const analysisSection = this.generateAnalysisSection(intermediateResults);
    if (analysisSection) report.sections.push(analysisSection);

    // 모델 성능 섹션 (ML이 수행된 경우)
    const modelSection = this.generateModelSection(intermediateResults);
    if (modelSection) report.sections.push(modelSection);

    // 시각화 섹션
    if (includeCharts) {
      const vizSection = this.generateVisualizationSection(intermediateResults);
      if (vizSection) report.sections.push(vizSection);
    }

    // 결론 및 권장사항
    const conclusionSection = this.generateConclusionSection(intermediateResults);
    report.sections.push(conclusionSection);

    // 리포트 파일 저장
    const reportPath = await this.saveReport(report, params);

    return {
      content: [{
        type: 'text',
        text: this.formatReportText(report)
      }],
      analysisType: 'report',
      report_path: reportPath,
      report
    };
  }

  generateDataOverviewSection(intermediateResults) {
    const dataResult = Object.values(intermediateResults).find(r => 
      r.data_info || r.statistics
    );
    
    if (!dataResult) return null;

    const section = {
      title: '데이터 개요',
      content: []
    };

    if (dataResult.data_info) {
      section.content.push({
        type: 'info',
        data: {
          shape: dataResult.data_info.shape,
          columns: dataResult.data_info.columns?.length || 'Unknown',
          nullCounts: dataResult.data_info.null_counts
        }
      });
    }

    if (dataResult.statistics) {
      section.content.push({
        type: 'statistics',
        data: dataResult.statistics
      });
    }

    return section;
  }

  generateAnalysisSection(intermediateResults) {
    const section = {
      title: '분석 결과',
      content: []
    };

    // 기본 통계 분석
    const basicStats = intermediateResults.basic_descriptive_stats;
    if (basicStats) {
      section.content.push({
        type: 'basic_stats',
        title: '기본 통계',
        data: basicStats
      });
    }

    // 상관관계 분석
    const correlation = intermediateResults.basic_correlation;
    if (correlation) {
      section.content.push({
        type: 'correlation',
        title: '상관관계 분석',
        data: correlation
      });
    }

    // 고급 분석 결과들
    Object.entries(intermediateResults).forEach(([key, result]) => {
      if (key.includes('advanced_')) {
        section.content.push({
          type: 'advanced',
          title: key.replace('advanced_', '').replace('_', ' '),
          data: result
        });
      }
    });

    return section.content.length > 0 ? section : null;
  }

  generateModelSection(intermediateResults) {
    const mlResults = Object.entries(intermediateResults).filter(([key, result]) => 
      key.includes('ml_') && result.metrics
    );

    if (mlResults.length === 0) return null;

    const section = {
      title: '모델 성능',
      content: []
    };

    mlResults.forEach(([key, result]) => {
      section.content.push({
        type: 'model_performance',
        title: key.replace('ml_', '').replace('_', ' '),
        metrics: result.metrics,
        model_info: result.model_info
      });
    });

    return section;
  }

  generateVisualizationSection(intermediateResults) {
    const vizResults = Object.entries(intermediateResults).filter(([key, result]) => 
      result.chart_path
    );

    if (vizResults.length === 0) return null;

    return {
      title: '시각화',
      content: vizResults.map(([key, result]) => ({
        type: 'visualization',
        title: result.chart_type || key,
        path: result.chart_path,
        description: result.description
      }))
    };
  }

  generateConclusionSection(intermediateResults) {
    const conclusions = [];
    const recommendations = [];

    // 데이터 품질 결론
    const dataQuality = this.assessDataQuality({ steps: [], intermediateResults });
    if (dataQuality.quality === 'high') {
      conclusions.push('데이터 품질이 우수하여 신뢰성 있는 분석이 가능합니다.');
    } else if (dataQuality.quality === 'medium') {
      conclusions.push('데이터 품질이 양호하나 일부 개선 여지가 있습니다.');
      recommendations.push('누락값 처리 및 이상치 제거를 고려해보세요.');
    } else {
      conclusions.push('데이터 품질에 문제가 있어 추가적인 전처리가 필요합니다.');
      recommendations.push('데이터 정제 및 품질 개선 작업을 우선적으로 수행하세요.');
    }

    // 모델 성능 결론
    const modelPerf = this.assessModelPerformance({ steps: [], intermediateResults });
    if (modelPerf.hasModel) {
      if (modelPerf.performance === 'excellent') {
        conclusions.push('모델 성능이 매우 우수합니다.');
      } else if (modelPerf.performance === 'good') {
        conclusions.push('모델 성능이 양호합니다.');
      } else {
        conclusions.push('모델 성능 개선이 필요합니다.');
        recommendations.push('하이퍼파라미터 튜닝이나 다른 알고리즘을 시도해보세요.');
      }
    }

    return {
      title: '결론 및 권장사항',
      content: [
        {
          type: 'conclusions',
          data: conclusions
        },
        {
          type: 'recommendations',
          data: recommendations
        }
      ]
    };
  }

  formatReportText(report) {
    let text = `# ${report.title}\n\n`;
    text += `생성일: ${new Date(report.createdAt).toLocaleString('ko-KR')}\n\n`;

    report.sections.forEach(section => {
      text += `## ${section.title}\n\n`;
      
      section.content.forEach(item => {
        switch (item.type) {
          case 'info':
            text += `- 데이터 크기: ${item.data.shape?.join(' × ') || 'Unknown'}\n`;
            text += `- 컬럼 수: ${item.data.columns}\n`;
            break;
          case 'conclusions':
            item.data.forEach(conclusion => {
              text += `- ${conclusion}\n`;
            });
            break;
          case 'recommendations':
            text += `### 권장사항\n`;
            item.data.forEach(rec => {
              text += `- ${rec}\n`;
            });
            break;
          default:
            if (item.title) {
              text += `### ${item.title}\n`;
            }
            break;
        }
      });
      
      text += '\n';
    });

    return text;
  }

  async saveReport(report, params) {
    try {
      const outputDir = params.output_dir || './results/reports';
      await fs.mkdir(outputDir, { recursive: true });
      
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const reportPath = path.join(outputDir, `report_${timestamp}.json`);
      
      await fs.writeFile(reportPath, JSON.stringify(report, null, 2), 'utf-8');
      
      return reportPath;
    } catch (error) {
      this.logger.error('리포트 저장 실패:', error);
      return null;
    }
  }

  async exportResults(intermediateResults, params) {
    const exportFormat = params.format || 'json';
    const outputDir = params.output_dir || './results/exports';
    
    try {
      await fs.mkdir(outputDir, { recursive: true });
      
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const exportPath = path.join(outputDir, `results_${timestamp}.${exportFormat}`);
      
      let exportData;
      switch (exportFormat) {
        case 'json':
          exportData = JSON.stringify(intermediateResults, null, 2);
          break;
        case 'csv':
          exportData = this.convertToCSV(intermediateResults);
          break;
        default:
          throw new Error(`지원하지 않는 내보내기 형식: ${exportFormat}`);
      }
      
      await fs.writeFile(exportPath, exportData, 'utf-8');
      
      return {
        content: [{
          type: 'text',
          text: `결과가 성공적으로 내보내졌습니다: ${exportPath}`
        }],
        analysisType: 'export',
        export_path: exportPath
      };
    } catch (error) {
      this.logger.error('결과 내보내기 실패:', error);
      throw error;
    }
  }

  convertToCSV(intermediateResults) {
    // 간단한 CSV 변환 (실제 구현에서는 더 정교한 로직 필요)
    const rows = [];
    rows.push(['Step', 'Type', 'Method', 'Status', 'Key_Results']);
    
    Object.entries(intermediateResults).forEach(([key, result]) => {
      const [stepInfo, type, method] = key.split('_');
      rows.push([
        stepInfo,
        type || 'unknown',
        method || 'unknown',
        result ? 'success' : 'failed',
        JSON.stringify(result).substring(0, 100) + '...'
      ]);
    });
    
    return rows.map(row => row.join(',')).join('\n');
  }

  async saveWorkflowResults(results) {
    try {
      const outputDir = './results/workflows';
      await fs.mkdir(outputDir, { recursive: true });
      
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const resultsPath = path.join(outputDir, `workflow_${results.sessionId}_${timestamp}.json`);
      
      await fs.writeFile(resultsPath, JSON.stringify(results, null, 2), 'utf-8');
      
      // 실행 히스토리에 추가
      this.executionHistory.set(results.sessionId, {
        ...results,
        resultsPath
      });
      
      this.logger.debug('워크플로우 결과 저장 완료:', resultsPath);
    } catch (error) {
      this.logger.error('워크플로우 결과 저장 실패:', error);
    }
  }

  // 워크플로우 상태 관리
  getExecutionStatus() {
    return {
      isExecuting: this.isExecuting,
      currentSession: this.currentSession,
      executionHistory: Array.from(this.executionHistory.keys()),
      lastExecution: this.executionHistory.size > 0 ? 
        Array.from(this.executionHistory.values()).pop().endTime : null
    };
  }

  async getExecutionResults(sessionId) {
    if (this.executionHistory.has(sessionId)) {
      return this.executionHistory.get(sessionId);
    }
    
    // 파일에서 로드 시도
    try {
      const outputDir = './results/workflows';
      const files = await fs.readdir(outputDir);
      const sessionFiles = files.filter(file => file.includes(sessionId));
      
      if (sessionFiles.length > 0) {
        const latestFile = sessionFiles.sort().pop();
        const filePath = path.join(outputDir, latestFile);
        const fileContent = await fs.readFile(filePath, 'utf-8');
        return JSON.parse(fileContent);
      }
    } catch (error) {
      this.logger.error('실행 결과 로드 실패:', error);
    }
    
    return null;
  }

  async cancelWorkflow() {
    if (!this.isExecuting) {
      throw new Error('실행 중인 워크플로우가 없습니다.');
    }

    try {
      // 현재 실행 중인 프로세스 중단 시도
      // 실제 구현에서는 실행 중인 Python 프로세스나 비동기 작업을 추적하고 중단해야 함
      
      this.isExecuting = false;
      this.currentSession = null;
      
      this.logger.info('워크플로우 실행이 취소되었습니다.');
      
      return {
        success: true,
        message: '워크플로우가 성공적으로 취소되었습니다.',
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      this.logger.error('워크플로우 취소 실패:', error);
      throw error;
    }
  }

  // 리소스 모니터링
  getResourceUsage() {
    // 실제 구현에서는 시스템 리소스 모니터링 로직 추가
    return {
      memory: {
        used: process.memoryUsage(),
        available: 'unknown'
      },
      cpu: {
        usage: 'unknown'
      },
      activeTools: Object.keys(this.tools),
      executionHistory: this.executionHistory.size
    };
  }

  // 정리 작업
  async cleanup() {
    try {
      // 도구들 정리
      for (const [name, tool] of Object.entries(this.tools)) {
        if (typeof tool.cleanup === 'function') {
          await tool.cleanup();
          this.logger.debug(`${name} 도구 정리 완료`);
        }
      }
      
      // 캐시 정리
      this.executionHistory.clear();
      
      this.logger.info('PipelineManager 정리 완료');
    } catch (error) {
      this.logger.error('PipelineManager 정리 실패:', error);
    }
  }

  // 디버깅 정보
  getDebugInfo() {
    return {
      isExecuting: this.isExecuting,
      currentSession: this.currentSession,
      toolsInitialized: Object.keys(this.tools),
      executionHistorySize: this.executionHistory.size,
      memoryUsage: process.memoryUsage()
    };
  }

  // 헬스 체크
  async healthCheck() {
    const health = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      checks: {}
    };

    try {
      // 도구들 상태 확인
      for (const [name, tool] of Object.entries(this.tools)) {
        if (typeof tool.healthCheck === 'function') {
          health.checks[name] = await tool.healthCheck();
        } else {
          health.checks[name] = { status: 'ok', message: 'No health check available' };
        }
      }

      // 전체 상태 결정
      const hasErrors = Object.values(health.checks).some(check => 
        check.status === 'error' || check.status === 'failed'
      );
      
      if (hasErrors) {
        health.status = 'degraded';
      }

    } catch (error) {
      health.status = 'error';
      health.error = error.message;
    }

    return health;
  }
}