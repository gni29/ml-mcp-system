// core/main-processor.js
import { Logger } from '../utils/logger.js';
import axios from 'axios';
import { DynamicLoader } from '../tools/discovery/dynamic-loader.js';
export class MainProcessor {
  constructor(modelManager) {
    this.modelManager = modelManager;
    this.logger = new Logger();
    this.dynamicLoader = new DynamicLoader();
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
    async handleDynamicAnalysis(args) {
    const { query, data, options = {} } = args;

    try {
      this.logger.info(`동적 분석 요청: ${query}`);

      // 자동으로 모듈 찾기 및 실행
      const result = await this.dynamicLoader.findAndExecuteModule(query, data, options);

      return {
        content: [{
          type: 'text',
          text: `🎯 동적 분석 완료!\n\n**사용된 모듈:** ${result.module.displayName}\n**카테고리:** ${result.module.category}/${result.module.subcategory}\n\n**결과:**\n${JSON.stringify(result.result, null, 2)}`
        }]
      };

    } catch (error) {
      this.logger.error('동적 분석 실패:', error);
      
      // 대안 모듈 제안
      const suggestions = await this.dynamicLoader.suggestModules(query);
      
      let responseText = `❌ 분석 실행 실패: ${error.message}\n\n`;
      
      if (suggestions.length > 0) {
        responseText += `💡 다음 모듈들을 시도해보세요:\n`;
        suggestions.forEach((suggestion, index) => {
          responseText += `${index + 1}. **${suggestion.name}** (${suggestion.category})\n   ${suggestion.description}\n\n`;
        });
      }

      return {
        content: [{
          type: 'text', 
          text: responseText
        }]
      };
    }
  }

  async handleModuleSearch(args) {
    const { query, category = null, limit = 10 } = args;

    try {
      const modules = await this.dynamicLoader.getAvailableModules(category);
      const filteredModules = query ? 
        modules.filter(m => 
          m.name.toLowerCase().includes(query.toLowerCase()) ||
          m.description?.toLowerCase().includes(query.toLowerCase()) ||
          m.tags.some(tag => tag.includes(query.toLowerCase()))
        ) : modules;

      const limitedModules = filteredModules.slice(0, limit);

      let responseText = `🔍 **사용 가능한 분석 모듈** (${limitedModules.length}개)\n\n`;

      limitedModules.forEach((module, index) => {
        responseText += `**${index + 1}. ${module.name}**\n`;
        responseText += `   📂 카테고리: ${module.category}/${module.subcategory}\n`;
        responseText += `   📄 설명: ${module.description || '설명 없음'}\n`;
        responseText += `   🏷️ 태그: ${module.tags.join(', ')}\n`;
        responseText += `   ✅ 실행 가능: ${module.isExecutable ? 'Yes' : 'No'}\n\n`;
      });

      if (filteredModules.length === 0) {
        responseText += `❌ 검색 조건에 맞는 모듈을 찾을 수 없습니다.\n\n`;
        responseText += `💡 사용 가능한 카테고리:\n`;
        
        const stats = this.dynamicLoader.getModuleStats();
        Object.entries(stats.byCategory).forEach(([cat, count]) => {
          responseText += `   • ${cat}: ${count}개 모듈\n`;
        });
      }

      return {
        content: [{
          type: 'text',
          text: responseText
        }]
      };

    } catch (error) {
      this.logger.error('모듈 검색 실패:', error);
      return {
        content: [{
          type: 'text',
          text: `❌ 모듈 검색 중 오류 발생: ${error.message}`
        }]
      };
    }
  }

  async handleModuleRefresh(args) {
    try {
      this.logger.info('모듈 새로고침 시작');
      
      const scanResult = await this.dynamicLoader.refreshModules();
      
      return {
        content: [{
          type: 'text',
          text: `🔄 **모듈 새로고침 완료!**\n\n` +
               `📊 **스캔 결과:**\n` +
               `   • 총 모듈 수: ${scanResult.count}개\n` +
               `   • 스캔 시간: ${scanResult.scanTime}ms\n` +
               `   • 마지막 스캔: ${new Date(scanResult.lastScan).toLocaleString()}\n\n` +
               `📂 **카테고리별 분포:**\n` +
               Object.entries(this.dynamicLoader.getModuleStats().byCategory)
                 .map(([cat, count]) => `   • ${cat}: ${count}개`)
                 .join('\n')
        }]
      };

    } catch (error) {
      this.logger.error('모듈 새로고침 실패:', error);
      return {
        content: [{
          type: 'text',
          text: `❌ 모듈 새로고침 실패: ${error.message}`
        }]
      };
    }
  }

  async handleModuleStats(args) {
    try {
      const stats = this.dynamicLoader.getModuleStats();
      const history = this.dynamicLoader.getExecutionHistory(5);
      
      let responseText = `📊 **모듈 통계**\n\n`;
      
      responseText += `**📈 전체 현황:**\n`;
      responseText += `   • 총 모듈 수: ${stats.total}개\n`;
      responseText += `   • 실행 가능 모듈: ${stats.executable}개\n`;
      responseText += `   • 마지막 스캔: ${new Date(stats.lastScan).toLocaleString()}\n\n`;
      
      responseText += `**📂 카테고리별 분포:**\n`;
      Object.entries(stats.byCategory).forEach(([category, count]) => {
        responseText += `   • ${category}: ${count}개\n`;
      });
      
      responseText += `\n**🔧 세부 카테고리:**\n`;
      Object.entries(stats.bySubcategory).forEach(([subcategory, count]) => {
        responseText += `   • ${subcategory}: ${count}개\n`;
      });
      
      if (history.length > 0) {
        responseText += `\n**📋 최근 실행 기록:**\n`;
        history.forEach((record, index) => {
          const status = record.success ? '✅' : '❌';
          responseText += `   ${index + 1}. ${status} ${record.module.name} (${new Date(record.executedAt).toLocaleTimeString()})\n`;
        });
      }

      return {
        content: [{
          type: 'text',
          text: responseText
        }]
      };

    } catch (error) {
      this.logger.error('모듈 통계 조회 실패:', error);
      return {
        content: [{
          type: 'text',
          text: `❌ 모듈 통계 조회 실패: ${error.message}`
        }]
      };
    }
  }

  async handleModuleTest(args) {
    const { moduleId, testData = null } = args;

    try {
      this.logger.info(`모듈 테스트: ${moduleId}`);
      
      const testResult = await this.dynamicLoader.testModule(moduleId, testData);
      
      if (testResult.success) {
        return {
          content: [{
            type: 'text',
            text: `✅ **모듈 테스트 성공!**\n\n` +
                 `**모듈:** ${testResult.module}\n` +
                 `**결과:**\n${JSON.stringify(testResult.result, null, 2)}`
          }]
        };
      } else {
        return {
          content: [{
            type: 'text',
            text: `❌ **모듈 테스트 실패**\n\n` +
                 `**모듈:** ${testResult.module}\n` +
                 `**오류:** ${testResult.error}`
          }]
        };
      }

    } catch (error) {
      this.logger.error('모듈 테스트 실패:', error);
      return {
        content: [{
          type: 'text',
          text: `❌ 모듈 테스트 중 오류 발생: ${error.message}`
        }]
      };
    }
  }

  // 기존 handleGenericTask 메서드를 업데이트하여 동적 분석 지원
  async handleGenericTask(args) {
    const { query, data, options = {} } = args;

    // 먼저 동적 분석 시도
    try {
      return await this.handleDynamicAnalysis({ query, data, options });
    } catch (error) {
      this.logger.warn('동적 분석 실패, 기존 방식으로 처리:', error);
      
      // 기존 방식으로 폴백
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
}

