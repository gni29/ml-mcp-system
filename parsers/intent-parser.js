// parsers/intent-parser.js
import { Logger } from '../utils/logger.js';

export class IntentParser {
  constructor(modelManager) {
    this.modelManager = modelManager;
    this.logger = new Logger();
    this.intentCache = new Map();
    this.keywords = this.initializeKeywords();
  }

  initializeKeywords() {
    return {
      analysis: {
        basic: [
          '기본통계', '기술통계', '요약', '평균', '표준편차', '분위수',
          'descriptive', 'summary', 'stats', 'basic', 'mean', 'std'
        ],
        correlation: [
          '상관관계', '상관분석', '연관성', '관계', '피어슨', '스피어만',
          'correlation', 'relationship', 'association', 'pearson', 'spearman'
        ],
        distribution: [
          '분포', '분포분석', '정규성', '히스토그램', '정규분포',
          'distribution', 'histogram', 'normal', 'normality', 'skew'
        ],
        outlier: [
          '이상치', '아웃라이어', '특이값', '이상값', '극값',
          'outlier', 'anomaly', 'extreme', 'unusual', 'abnormal'
        ]
      },
      advanced: {
        pca: [
          'pca', '주성분분석', '차원축소', '주성분', '고유벡터',
          'principal', 'component', 'dimensionality', 'reduction', 'eigen'
        ],
        clustering: [
          '클러스터링', '클러스터', '군집', '군집분석', 'k-means', 'dbscan',
          'clustering', 'cluster', 'group', 'segmentation', 'kmeans'
        ],
        feature_engineering: [
          '피처엔지니어링', '변수생성', '파생변수', '특성공학', '변수변환',
          'feature', 'engineering', 'transformation', 'creation', 'derived'
        ]
      },
      ml: {
        regression: [
          '회귀', '회귀분석', '예측', '선형회귀', '다항회귀', '예측모델',
          'regression', 'predict', 'linear', 'polynomial', 'forecasting'
        ],
        classification: [
          '분류', '분류모델', '클래스', '카테고리', '로지스틱', '의사결정나무',
          'classification', 'classify', 'class', 'category', 'logistic', 'tree'
        ],
        ensemble: [
          '앙상블', '랜덤포레스트', '부스팅', '배깅', '보팅', '스태킹',
          'ensemble', 'random', 'forest', 'boosting', 'bagging', 'voting', 'stacking'
        ]
      },
      deep_learning: {
        general: [
          '딥러닝', '신경망', '뉴럴네트워크', '심층학습', 'deep', 'neural', 'network'
        ],
        cnn: [
          'cnn', '합성곱', '이미지', '컨볼루션', '이미지분류', '객체탐지',
          'convolutional', 'image', 'vision', 'detection', 'recognition'
        ],
        rnn: [
          'rnn', 'lstm', 'gru', '순환', '시계열', '자연어', '텍스트',
          'recurrent', 'sequence', 'time', 'series', 'text', 'nlp'
        ],
        transformer: [
          'transformer', 'bert', 'gpt', 'attention', '트랜스포머', '어텐션',
          'self-attention', 'encoder', 'decoder'
        ]
      },
      visualization: {
        '2d': [
          '산점도', '선그래프', '막대그래프', '히스토그램', '박스플롯',
          'scatter', 'line', 'bar', 'histogram', 'box', 'plot'
        ],
        '3d': [
          '3d', '3차원', '입체', '표면', '3d산점도', '3d그래프',
          'three', 'dimensional', 'surface', '3d'
        ],
        heatmap: [
          '히트맵', '열지도', '상관관계매트릭스', '매트릭스',
          'heatmap', 'heat', 'matrix', 'correlation'
        ]
      },
      actions: {
        analyze: [
          '분석', '분석해줘', '분석하자', '살펴보자', '확인해봐',
          'analyze', 'analysis', 'examine', 'investigate', 'study'
        ],
        visualize: [
          '시각화', '그래프', '차트', '그려줘', '플롯',
          'visualize', 'plot', 'chart', 'graph', 'draw', 'show'
        ],
        train: [
          '훈련', '학습', '모델생성', '만들어줘', '구축',
          'train', 'training', 'build', 'create', 'develop', 'fit'
        ],
        predict: [
          '예측', '추정', '예상', '포캐스트', '추론',
          'predict', 'forecast', 'estimate', 'inference', 'projection'
        ]
      },
      modifiers: {
        comparison: [
          '비교', '대비', '차이', '비교분석', '대조',
          'compare', 'comparison', 'versus', 'difference', 'contrast'
        ],
        combination: [
          '결합', '조합', '합쳐서', '같이', '함께',
          'combine', 'together', 'merge', 'joint', 'with'
        ],
        sequence: [
          '순서대로', '단계별', '차례로', '파이프라인',
          'step', 'sequence', 'pipeline', 'workflow', 'process'
        ]
      }
    };
  }

  async parseIntent(userQuery) {
    try {
      // 캐시 확인
      const cacheKey = this.generateCacheKey(userQuery);
      if (this.intentCache.has(cacheKey)) {
        return this.intentCache.get(cacheKey);
      }

      // 키워드 기반 초기 분석
      const keywordAnalysis = this.analyzeKeywords(userQuery);
      
      // AI 모델 기반 상세 분석
      const aiAnalysis = await this.analyzeWithAI(userQuery, keywordAnalysis);
      
      // 결과 통합
      const combinedIntent = this.combineAnalysis(keywordAnalysis, aiAnalysis);
      
      // 캐시 저장
      this.intentCache.set(cacheKey, combinedIntent);
      
      return combinedIntent;
    } catch (error) {
      this.logger.error('의도 분석 실패:', error);
      return this.getFallbackIntent(userQuery);
    }
  }

  analyzeKeywords(query) {
    const normalizedQuery = query.toLowerCase();
    const foundKeywords = {
      analysis: [],
      actions: [],
      modifiers: [],
      technical: []
    };

    // 각 카테고리별 키워드 검색
    for (const [category, subcategories] of Object.entries(this.keywords)) {
      for (const [subcategory, keywords] of Object.entries(subcategories)) {
        const matchedKeywords = keywords.filter(keyword =>
          normalizedQuery.includes(keyword.toLowerCase())
        );
        
        if (matchedKeywords.length > 0) {
          if (category === 'actions') {
            foundKeywords.actions.push({ type: subcategory, keywords: matchedKeywords });
          } else if (category === 'modifiers') {
            foundKeywords.modifiers.push({ type: subcategory, keywords: matchedKeywords });
          } else {
            foundKeywords.analysis.push({
              category: category,
              type: subcategory,
              keywords: matchedKeywords
            });
          }
        }
      }
    }

    // 기술적 키워드 추가 분석
    foundKeywords.technical = this.extractTechnicalKeywords(normalizedQuery);

    return {
      foundKeywords,
      confidence: this.calculateKeywordConfidence(foundKeywords),
      primaryAction: this.determinePrimaryAction(foundKeywords),
      primaryAnalysis: this.determinePrimaryAnalysis(foundKeywords)
    };
  }

  extractTechnicalKeywords(query) {
    const technicalPatterns = {
      numbers: /\d+/g,
      percentages: /\d+%/g,
      file_extensions: /\.(csv|xlsx|json|txt|png|jpg)/gi,
      model_names: /(xgboost|lightgbm|catboost|bert|gpt|resnet|efficientnet)/gi,
      parameters: /(epoch|batch|learning_rate|n_clusters|n_components)/gi
    };

    const technical = {};
    for (const [type, pattern] of Object.entries(technicalPatterns)) {
      const matches = query.match(pattern);
      if (matches) {
        technical[type] = matches;
      }
    }

    return technical;
  }

  calculateKeywordConfidence(foundKeywords) {
    const weights = {
      actions: 0.3,
      analysis: 0.4,
      modifiers: 0.2,
      technical: 0.1
    };

    let totalScore = 0;
    let maxScore = 0;

    for (const [category, weight] of Object.entries(weights)) {
      const categoryKeywords = foundKeywords[category];
      if (Array.isArray(categoryKeywords)) {
        const categoryScore = categoryKeywords.reduce((sum, item) => {
          return sum + (item.keywords ? item.keywords.length : 0);
        }, 0);
        totalScore += categoryScore * weight;
        maxScore += 10 * weight; // 최대 10개 키워드 가정
      }
    }

    return Math.min(totalScore / maxScore, 1.0);
  }

  determinePrimaryAction(foundKeywords) {
    const actions = foundKeywords.actions;
    if (actions.length === 0) return 'analyze';

    // 가장 많은 키워드를 가진 액션 선택
    return actions.reduce((prev, current) =>
      prev.keywords.length > current.keywords.length ? prev : current
    ).type;
  }

  determinePrimaryAnalysis(foundKeywords) {
    const analysis = foundKeywords.analysis;
    if (analysis.length === 0) return { category: 'basic', type: 'descriptive_stats' };

    // 가장 구체적인 분석 방법 선택
    const priorityOrder = ['deep_learning', 'ml', 'advanced', 'analysis'];
    
    for (const priority of priorityOrder) {
      const found = analysis.find(item => item.category === priority);
      if (found) {
        return { category: found.category, type: found.type };
      }
    }

    return analysis[0];
  }

  async analyzeWithAI(query, keywordAnalysis) {
    const prompt = `사용자 쿼리를 분석하여 의도를 파악해주세요:

쿼리: "${query}"

키워드 분석 결과:
- 주요 액션: ${keywordAnalysis.primaryAction}
- 주요 분석: ${keywordAnalysis.primaryAnalysis.category}.${keywordAnalysis.primaryAnalysis.type}
- 신뢰도: ${keywordAnalysis.confidence.toFixed(2)}

다음 JSON 형식으로 응답해주세요:
{
  "intent": "주요 의도",
  "confidence": 0.0-1.0,
  "complexity": 0.0-1.0,
  "requires_data": true/false,
  "requires_training": true/false,
  "requires_visualization": true/false,
  "workflow_type": "single|sequential|parallel",
  "estimated_steps": 1-10,
  "data_requirements": {
    "type": "tabular|image|text|time_series",
    "size": "small|medium|large",
    "preprocessing": "minimal|moderate|extensive"
  },
  "suggested_methods": ["method1", "method2"],
  "parameters": {
    "key": "value"
  }
}`;

    try {
      const response = await this.modelManager.queryModel('router', prompt, {
        temperature: 0.1,
        max_tokens: 800
      });

      return this.parseAIResponse(response);
    } catch (error) {
      this.logger.warn('AI 분석 실패:', error);
      return this.getDefaultAIAnalysis();
    }
  }

  parseAIResponse(response) {
    try {
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]);
      }
    } catch (error) {
      this.logger.warn('AI 응답 파싱 실패:', error);
    }

    return this.getDefaultAIAnalysis();
  }

  getDefaultAIAnalysis() {
    return {
      intent: 'analyze',
      confidence: 0.5,
      complexity: 0.3,
      requires_data: true,
      requires_training: false,
      requires_visualization: true,
      workflow_type: 'single',
      estimated_steps: 1,
      data_requirements: {
        type: 'tabular',
        size: 'medium',
        preprocessing: 'minimal'
      },
      suggested_methods: ['basic.descriptive_stats'],
      parameters: {}
    };
  }

  combineAnalysis(keywordAnalysis, aiAnalysis) {
    return {
      // 기본 정보
      intent: aiAnalysis.intent || keywordAnalysis.primaryAction,
      confidence: Math.max(keywordAnalysis.confidence, aiAnalysis.confidence || 0),
      complexity: aiAnalysis.complexity || this.estimateComplexity(keywordAnalysis),
      
      // 키워드 분석 결과
      keywords: keywordAnalysis.foundKeywords,
      primary_action: keywordAnalysis.primaryAction,
      primary_analysis: keywordAnalysis.primaryAnalysis,
      
      // AI 분석 결과
      ai_analysis: aiAnalysis,
      
      // 통합 분석
      workflow_type: this.determineWorkflowType(keywordAnalysis, aiAnalysis),
      estimated_steps: this.estimateSteps(keywordAnalysis, aiAnalysis),
      requires_pipeline: this.requiresPipeline(keywordAnalysis, aiAnalysis),
      
      // 제안사항
      suggested_methods: this.suggestMethods(keywordAnalysis, aiAnalysis),
      parameters: this.extractParameters(keywordAnalysis, aiAnalysis),
      
      // 메타데이터
      timestamp: new Date().toISOString(),
      analysis_version: '1.0'
    };
  }

  estimateComplexity(keywordAnalysis) {
    const complexityFactors = {
      deep_learning: 0.9,
      ml: 0.7,
      advanced: 0.5,
      analysis: 0.3
    };

    let maxComplexity = 0;
    for (const item of keywordAnalysis.foundKeywords.analysis) {
      const complexity = complexityFactors[item.category] || 0.2;
      maxComplexity = Math.max(maxComplexity, complexity);
    }

    // 키워드 수에 따른 복잡도 증가
    const keywordCount = keywordAnalysis.foundKeywords.analysis.length;
    if (keywordCount > 2) {
      maxComplexity = Math.min(maxComplexity + 0.2, 1.0);
    }

    return maxComplexity;
  }

  determineWorkflowType(keywordAnalysis, aiAnalysis) {
    const sequentialKeywords = ['다음', '그다음', '이후', '단계별', '순서대로'];
    const parallelKeywords = ['동시에', '함께', '같이', '한번에'];
    
    const query = keywordAnalysis.foundKeywords.technical.join(' ').toLowerCase();
    
    if (sequentialKeywords.some(keyword => query.includes(keyword))) {
      return 'sequential';
    }
    
    if (parallelKeywords.some(keyword => query.includes(keyword))) {
      return 'parallel';
    }
    
    // 분석 항목이 여러 개면 sequential
    if (keywordAnalysis.foundKeywords.analysis.length > 1) {
      return 'sequential';
    }
    
    return aiAnalysis.workflow_type || 'single';
  }

  estimateSteps(keywordAnalysis, aiAnalysis) {
    const baseSteps = keywordAnalysis.foundKeywords.analysis.length;
    const hasVisualization = keywordAnalysis.foundKeywords.actions.some(
      action => action.type === 'visualize'
    );
    
    let steps = Math.max(baseSteps, 1);
    if (hasVisualization) steps += 1;
    
    return Math.min(steps, aiAnalysis.estimated_steps || 10);
  }

  requiresPipeline(keywordAnalysis, aiAnalysis) {
    return keywordAnalysis.foundKeywords.analysis.length > 1 ||
           keywordAnalysis.foundKeywords.modifiers.some(mod => mod.type === 'sequence') ||
           (aiAnalysis.estimated_steps && aiAnalysis.estimated_steps > 1);
  }

  suggestMethods(keywordAnalysis, aiAnalysis) {
    const methods = [];
    
    // 키워드 기반 방법 추천
    for (const item of keywordAnalysis.foundKeywords.analysis) {
      methods.push(`${item.category}.${item.type}`);
    }
    
    // AI 추천 방법 추가
    if (aiAnalysis.suggested_methods) {
      methods.push(...aiAnalysis.suggested_methods);
    }
    
    // 중복 제거
    return [...new Set(methods)];
  }

  extractParameters(keywordAnalysis, aiAnalysis) {
    const parameters = {};
    
    // 기술적 키워드에서 파라미터 추출
    const technical = keywordAnalysis.foundKeywords.technical;
    
    if (technical.numbers) {
      // 숫자는 클러스터 수나 컴포넌트 수로 추정
      parameters.n_clusters = parseInt(technical.numbers[0]);
      parameters.n_components = parseInt(technical.numbers[0]);
    }
    
    if (technical.percentages) {
      parameters.test_size = parseFloat(technical.percentages[0]) / 100;
    }
    
    // AI 분석에서 파라미터 추가
    if (aiAnalysis.parameters) {
      Object.assign(parameters, aiAnalysis.parameters);
    }
    
    return parameters;
  }

  getFallbackIntent(query) {
    return {
      intent: 'analyze',
      confidence: 0.3,
      complexity: 0.3,
      keywords: { analysis: [], actions: [], modifiers: [], technical: [] },
      primary_action: 'analyze',
      primary_analysis: { category: 'basic', type: 'descriptive_stats' },
      workflow_type: 'single',
      estimated_steps: 1,
      requires_pipeline: false,
      suggested_methods: ['basic.descriptive_stats'],
      parameters: {},
      timestamp: new Date().toISOString(),
      analysis_version: '1.0'
    };
  }

  generateCacheKey(query) {
    return `intent_${query.toLowerCase().replace(/\s+/g, '_').substring(0, 50)}`;
  }

  // 캐시 정리
  clearCache() {
    this.intentCache.clear();
  }

  // 통계 정보
  getStatistics() {
    return {
      cache_size: this.intentCache.size,
      keyword_categories: Object.keys(this.keywords).length,
      total_keywords: Object.values(this.keywords).reduce((total, category) => {
        return total + Object.values(category).reduce((catTotal, keywords) => {
          return catTotal + keywords.length;
        }, 0);
      }, 0)
    };
  }
}
