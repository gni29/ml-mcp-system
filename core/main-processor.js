// core/main-processor.js
import { Logger } from '../utils/logger.js';
import axios from 'axios';

export class MainProcessor {
  constructor(modelManager) {
    this.modelManager = modelManager;
    this.logger = new Logger();
  }

  async handleComplexTask(args, tools) {
    try {
      this.logger.info('복잡한 작업 처리 시작', { args, tools });

      // 프로세서 모델 로딩 (필요시)
      await this.modelManager.loadProcessorModel();

      // 작업 유형에 따른 처리
      const taskType = this.determineTaskType(args, tools);
      
      switch (taskType) {
        case 'data_analysis':
          return await this.handleDataAnalysis(args);
        case 'model_training':
          return await this.handleModelTraining(args);
        case 'code_generation':
          return await this.handleCodeGeneration(args);
        default:
          return await this.handleGenericTask(args);
      }
    } catch (error) {
      this.logger.error('복잡한 작업 처리 실패:', error);
      return {
        content: [{
          type: 'text',
          text: `작업 처리 중 오류가 발생했습니다: ${error.message}`
        }],
        isError: true
      };
    }
  }

  determineTaskType(args, tools) {
    if (tools.includes('python_executor') || tools.includes('analyzer')) {
      return 'data_analysis';
    } else if (tools.includes('model_trainer')) {
      return 'model_training';
    } else if (args.code || args.programming) {
      return 'code_generation';
    }
    return 'generic';
  }

  async handleDataAnalysis(args) {
    const prompt = `데이터 분석 작업을 위한 Python 코드를 생성해주세요:
작업 요청: ${JSON.stringify(args)}

다음 형식으로 응답해주세요:
1. 필요한 라이브러리 import
2. 데이터 로딩 코드
3. 분석 수행 코드
4. 결과 출력 코드

코드는 실행 가능한 형태로 작성해주세요.`;

    const response = await this.modelManager.queryModel('processor', prompt, {
      temperature: 0.1,
      max_tokens: 1500
    });

    return {
      content: [{
        type: 'text',
        text: `데이터 분석 코드를 생성했습니다:\n\n${response}`
      }]
    };
  }

  async handleModelTraining(args) {
    const prompt = `머신러닝 모델 훈련을 위한 코드를 생성해주세요:
요청사항: ${JSON.stringify(args)}

포함할 내용:
1. 데이터 전처리
2. 모델 선택 및 설정
3. 훈련 과정
4. 성능 평가
5. 모델 저장`;

    const response = await this.modelManager.queryModel('processor', prompt, {
      temperature: 0.2,
      max_tokens: 2000
    });

    return {
      content: [{
        type: 'text',
        text: `모델 훈련 코드를 생성했습니다:\n\n${response}`
      }]
    };
  }

  async handleCodeGeneration(args) {
    const prompt = `다음 요청에 대한 코드를 생성해주세요:
${JSON.stringify(args)}

고품질의 실행 가능한 코드를 작성해주세요.`;

    const response = await this.modelManager.queryModel('processor', prompt, {
      temperature: 0.1,
      max_tokens: 1500
    });

    return {
      content: [{
        type: 'text',
        text: response
      }]
    };
  }

  async handleGenericTask(args) {
    const prompt = `다음 요청을 처리해주세요: ${JSON.stringify(args)}`;

    const response = await this.modelManager.queryModel('processor', prompt, {
      temperature: 0.3,
      max_tokens: 1000
    });

    return {
      content: [{
        type: 'text',
        text: response
      }]
    };
  }
}
