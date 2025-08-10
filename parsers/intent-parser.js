// parsers/intent-parser.js
import { Logger } from '../utils/logger.js';

export class IntentParser {
  constructor(modelManager) {
    this.modelManager = modelManager;
    this.logger = new Logger();
    this.intentCache = new Map();
    this.keywords = this.initializeKeywords();
    this.patterns = this.initializePatterns();
    this.maxCacheSize = 100;
  }

  initializeKeywords() {
    return {
      analysis: {
        basic: [
          '기본통계', '기술통계', '요약', '평균', '표준편차', '분위수', '중앙값',
          'descriptive', 'summary', 'stats', 'basic', 'mean', 'std', 'median'
        ],
        correlation: [
          '상관관계', '상관분석', '연관성', '관계', '피어슨', '스피어만',
          'correlation', 'relationship', 'association', 'pearson', 'spearman'
        ],
        distribution: [
          '분포', '분포분석', '정규성', '히스토그램', '정규분포', '왜도', '첨도',
          'distribution', 'histogram', 'normal', 'normality', 'skewness', 'kurtosis'
        ],
        outlier: [
          '이상치', '아웃라이어', '특이값', '이상값', '극값', '이상치탐지',
          'outlier', 'anomaly', 'extreme', 'unusual', 'abnormal', 'detection'
        ]
      },
      
      advanced: {
        pca: [
          'pca', '주성분분석', '차원축소', '주성분', '고유벡터', '설명분산',
          'principal', 'component', 'dimensionality', 'reduction', 'eigen', 'variance'
        ],
        clustering: [
          '클러스터링', '클러스터', '군집', '군집분석', 'k-means', 'dbscan', '계층적',
          'clustering', 'cluster', 'group', 'segmentation', 'kmeans', 'hierarchical'
        ],
        feature_engineering: [
          '피처엔지니어링', '변수생성', '파생변수', '특성공학', '변수변환', '특성선택',
          'feature', 'engineering', 'transformation', 'creation', 'derived', 'selection'
        ]
      },
      
      ml: {
        regression: [
          '회귀', '회귀분석', '예측', '선형회귀', '다항회귀', '예측모델', '추정',
          'regression', 'linear', 'polynomial', 'prediction', 'forecast', 'estimate'
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
          '딥러닝', '신경망', '뉴럴네트워크', '심층학습', '인공신경망',
          'deep', 'neural', 'network', 'deeplearning', 'artificial'
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
          'self-attention', 'encoder', 'decoder', 'language'
        ]
      },
      
      visualization: {
        basic_charts: [
          '산점도', '선그래프', '막대그래프', '히스토그램', '박스플롯',
          'scatter', 'line', 'bar', 'histogram', 'box', 'plot'
        ],
        advanced_charts: [
          '3d', '3차원', '입체', '표면', '3d산점도', '3d그래프',
          'three', 'dimensional', 'surface', 'contour'
        ],
        heatmap: [
          '히트맵', '열지도', '상관관계매트릭스', '매트릭스',
          'heatmap', 'heat', 'matrix', 'correlation'
        ]
      },
      
      actions: {
        analyze: [
          '분석', '분석해줘', '분석하자', '살펴보자', '확인해봐', '조사',
          'analyze', 'analysis', 'examine', 'investigate', 'study', 'explore'
        ],
        visualize: [
          '시각화', '그래프', '차트', '그려줘', '플롯', '보여줘', '시각적',
          'visualize', 'plot', 'chart', 'graph', 'draw', 'show', 'display'
        ],
        train: [
          '훈련', '학습', '모델생성', '만들어줘', '구축', '개발', '모델링',
          'train', 'training', 'build', 'create', 'develop', 'fit', 'model'
        ],
        predict: [
          '예측', '추정', '예상', '포캐스트', '추론', '예측해줘',
          'predict', 'forecast', 'estimate', 'inference', 'projection'
        ]
      },
      
      modifiers: {
        comparison: [
          '비교', '대비', '차이', '비교분석', '대조', '대응',
          'compare', 'comparison', 'versus', 'difference', 'contrast', 'vs'
        ],
        combination: [
          '결합', '조합', '합쳐서', '같이', '함께', '통합',
          'combine', 'together', 'merge', 'joint', 'with', 'integrate'
        ],
        sequence: [
          '순서대로', '단계별', '차례로', '파이프라인', '워크플로우',
          'step', 'sequence', 'pipeline', 'workflow', 'process', 'sequential'
        ]
      },
      
      technical: {
        numbers: /\b(\d+)\b/g,
        percentages: /\b(\d+(?:\.\d+)?)\s*%/g,
        file_formats: /\.(csv|xlsx|json|txt|parquet|h5)/gi,
        models: ['svm', 'rf', 'xgboost', 'lightgbm', 'catboost', 'bert', 'gpt']
      }
    };
  }

  initializePatterns() {
    return {
      question_patterns: [
        /^(어떻게|how|what|why|when|where|which)/i,
        /\?$/,
        /(무엇|뭐|어디|언제|왜|어떤)/
      ],
      request_patterns: [
        /(해줘|해주세요|부탁|요청|원해)/,
        /(please|can you|could you|would you)/i
      ],
      comparison_patterns: [
        /(비교|compare|대비|vs|versus)/i,
        /(차이|difference|다른|different)/i
      ],
      urgency_patterns: [
        /(빨리|급해|urgent|asap|즉시)/i,
        /(지금|now|immediately)/i
      ]
    };
  }

  async parseIntent(userQuery, context = {}) {
    try {
      this.logger.debug(`의도 분석 시작: ${userQuery}`);
      
      // 캐시 확인
      const cacheKey = this.generateCacheKey(userQuery, context);
      if (this.intentCache.has(cacheKey)) {
        this.logger.debug('캐시에서 의도 결과 반환');
        return this.intentCache.get(cacheKey);
      }

      // 1. 키워드 기반 초기 분석
      const keywordAnalysis = this.analyzeKeywords(userQuery);
      
      // 2. 패턴 분석
      const patternAnalysis = this.analyzePatterns(userQuery);
      
      // 3. AI 모델 기반 상세 분석 (필요한 경우)
      let aiAnalysis = null;
      if (keywordAnalysis.confidence < 0.7) {
        aiAnalysis = await this.analyzeWithAI(userQuery, context);
      }
      
      // 4. 결과 통합
      const combinedIntent = this.combineAnalysis(
        keywordAnalysis, 
        patternAnalysis, 
        aiAnalysis, 
        context
      );
      
      // 5. 캐시 저장
      this.saveToCache(cacheKey, combinedIntent);
      
      this.logger.debug(`의도 분석 완료: ${combinedIntent.intent} (${combinedIntent.confidence})`);
      return combinedIntent;
      
    } catch (error) {
      this.logger.error('의도 분석 실패:', error);
      return this.getFallbackIntent(userQuery);
    }
  }

  analyzeKeywords(query) {
    const lowerQuery = query.toLowerCase();
    const foundKeywords = {
      analysis: [],
      actions: [],
      modifiers: [],
      technical: {
        numbers: [],
        percentages: [],
        file_formats: [],
        models: []
      }
    };

    // 분석 키워드 검색
    for (const [category, subcategories] of Object.entries(this.keywords.analysis)) {
      for (const [type, keywords] of Object.entries(subcategories)) {
        const matchCount = keywords.filter(keyword => 
          lowerQuery.includes(keyword.toLowerCase())
        ).length;
        
        if (matchCount > 0) {
          foundKeywords.analysis.push({
            category: 'analysis',
            type: type,
            subcategory: category,
            keywords: keywords.filter(k => lowerQuery.includes(k.toLowerCase())),
            score: matchCount
          });
        }
      }
    }

    // 고급 분석 키워드 검색
    for (const [category, subcategories] of Object.entries(this.keywords.advanced)) {
      for (const [type, keywords] of Object.entries(subcategories)) {
        const matchCount = keywords.filter(keyword => 
          lowerQuery.includes(keyword.toLowerCase())
        ).length;
        
        if (matchCount > 0) {
          foundKeywords.analysis.push({
            category: 'advanced',
            type: type,
            subcategory: category,
            keywords: keywords.filter(k => lowerQuery.includes(k.toLowerCase())),
            score: matchCount
          });
        }
      }
    }

    // ML 키워드 검색
    for (const [category, subcategories] of Object.entries(this.keywords.ml)) {
      for (const [type, keywords] of Object.entries(subcategories)) {
        const matchCount = keywords.filter(keyword => 
          lowerQuery.includes(keyword.toLowerCase())
        ).length;
        
        if (matchCount > 0) {
          foundKeywords.analysis.push({
            category: 'ml',
            type: type,
            subcategory: category,
            keywords: keywords.filter(k => lowerQuery.includes(k.toLowerCase())),
            score: matchCount
          });
        }
      }
    }

    // 딥러닝 키워드 검색
    for (const [category, subcategories] of Object.entries(this.keywords.deep_learning)) {
      for (const [type, keywords] of Object.entries(subcategories)) {
        const matchCount = keywords.filter(keyword => 
          lowerQuery.includes(keyword.toLowerCase())
        ).length;
        
        if (matchCount > 0) {
          foundKeywords.analysis.push({
            category: 'deep_learning',
            type: type,
            subcategory: category,
            keywords: keywords.filter(k => lowerQuery.includes(k.toLowerCase())),
            score: matchCount
          });
        }
      }
    }

    // 시각화 키워드 검색
    for (const [category, subcategories] of Object.entries(this.keywords.visualization)) {
      for (const [type, keywords] of Object.entries(subcategories)) {
        const matchCount = keywords.filter(keyword => 
          lowerQuery.includes(keyword.toLowerCase())
        ).length;
        
        if (matchCount > 0) {
          foundKeywords.analysis.push({
            category: 'visualization',
            type: type,
            subcategory: category,
            keywords: keywords.filter(k => lowerQuery.includes(k.toLowerCase())),
            score: matchCount
          });
        }
      }
    }

    // 액션 키워드 검색
    for (const [type, keywords] of Object.entries(this.keywords.actions)) {
      const matchCount = keywords.filter(keyword => 
        lowerQuery.includes(keyword.toLowerCase())
      ).length;
      
      if (matchCount > 0) {
        foundKeywords.actions.push({
          type: type,
          keywords: keywords.filter(k => lowerQuery.includes(k.toLowerCase())),
          score: matchCount
        });
      }
    }

    // 수정자 키워드 검색
    for (const [type, keywords] of Object.entries(this.keywords.modifiers)) {
      const matchCount = keywords.filter(keyword => 
        lowerQuery.includes(keyword.toLowerCase())
      ).length;
      
      if (matchCount > 0) {
        foundKeywords.modifiers.push({
          type: type,
          keywords: keywords.filter(k => lowerQuery.includes(k.toLowerCase())),
          score: matchCount
        });
      }
    }

    // 기술적 키워드 추출
    foundKeywords.technical = this.extractTechnicalKeywords(lowerQuery);

    return {
      foundKeywords,
      confidence: this.calculateKeywordConfidence(foundKeywords),
      primaryAction: this.determinePrimaryAction(foundKeywords),
      primaryAnalysis: this.determinePrimaryAnalysis(foundKeywords)
    };
  }

  extractTechnicalKeywords(query) {
    const technical = {
      numbers: [],
      percentages: [],
      file_formats: [],
      models: []
    };

    // 숫자 추출
    const numbers = query.match(this.keywords.technical.numbers);
    if (numbers) {
      technical.numbers = numbers.map(n => parseInt(n));
    }

    // 퍼센트 추출
    const percentages = query.match(this.keywords.technical.percentages);
    if (percentages) {
      technical.percentages = percentages.map(p => parseFloat(p.replace('%', '')));
    }

    // 파일 형식 추출
    const fileFormats = query.match(this.keywords.technical.file_formats);
    if (fileFormats) {
      technical.file_formats = fileFormats.map(f => f.toLowerCase());
    }

    // 모델명 추출
    const models = this.keywords.technical.models.filter(model => 
      query.includes(model.toLowerCase())
    );
    if (models.length > 0) {
      technical.models = models;
    }

    return technical;
  }

  analyzePatterns(query) {
    const patterns = {
      isQuestion: false,
      isRequest: false,
      hasComparison: false,
      hasUrgency: false,
      sentiment: 'neutral'
    };

    // 질문 패턴 검사
    patterns.isQuestion = this.patterns.question_patterns.some(pattern => 
      pattern.test(query)
    );

    // 요청 패턴 검사
    patterns.isRequest = this.patterns.request_patterns.some(pattern => 
      pattern.test(query)
    );

    // 비교 패턴 검사
    patterns.hasComparison = this.patterns.comparison_patterns.some(pattern => 
      pattern.test(query)
    );

    // 긴급성 패턴 검사
    patterns.hasUrgency = this.patterns.urgency_patterns.some(pattern => 
      pattern.test(query)
    );

    // 감정 분석 (간단한 버전)
    patterns.sentiment = this.analyzeSentiment(query);

    return patterns;
  }

  analyzeSentiment(query) {
    const positiveWords = ['좋은', '훌륭한', '멋진', '완벽한', '최고의', 'good', 'great', 'excellent', 'perfect', 'best'];
    const negativeWords = ['나쁜', '최악의', '문제', '에러', '실패', 'bad', 'worst', 'problem', 'error', 'fail'];

    const lowerQuery = query.toLowerCase();
    const positiveCount = positiveWords.filter(word => lowerQuery.includes(word)).length;
    const negativeCount = negativeWords.filter(word => lowerQuery.includes(word)).length;

    if (positiveCount > negativeCount) return 'positive';
    if (negativeCount > positiveCount) return 'negative';
    return 'neutral';
  }

  calculateKeywordConfidence(foundKeywords) {
    const weights = {
      analysis: 0.4,
      actions: 0.3,
      modifiers: 0.2,
      technical: 0.1
    };

    let totalScore = 0;
    let maxScore = 0;

    for (const [category, weight] of Object.entries(weights)) {
      if (foundKeywords[category]) {
        let categoryScore = 0;
        if (Array.isArray(foundKeywords[category])) {
          categoryScore = foundKeywords[category].reduce((sum, item) => {
            return sum + (item.keywords ? item.keywords.length : 0);
          }, 0);
        } else if (typeof foundKeywords[category] === 'object') {
          categoryScore = Object.values(foundKeywords[category]).reduce((sum, arr) => {
            return sum + (Array.isArray(arr) ? arr.length : 0);
          }, 0);
        }
        totalScore += categoryScore * weight;
        maxScore += 10 * weight; // 최대 10개 키워드 가정
      }
    }

    return Math.min(totalScore / maxScore, 1.0);
  }

  determinePrimaryAction(foundKeywords) {
    const actions = foundKeywords.actions;
    if (actions.length === 0) return 'analyze';

    // 가장 높은 점수를 가진 액션 선택
    return actions.reduce((prev, current) =>
      prev.score > current.score ? prev : current
    ).type;
  }

  determinePrimaryAnalysis(foundKeywords) {
    const analysis = foundKeywords.analysis;
    if (analysis.length === 0) return { category: 'analysis', type: 'basic' };

    // 우선순위: deep_learning > ml > advanced > analysis > visualization
    const priorityOrder = ['deep_learning', 'ml', 'advanced', 'analysis', 'visualization'];
    
    for (const priority of priorityOrder) {
      const found = analysis.find(item => item.category === priority);
      if (found) {
        return { category: found.category, type: found.type };
      }
    }

    // 가장 높은 점수를 가진 분석 방법 선택
    return analysis.reduce((prev, current) =>
      prev.score > current.score ? prev : current
    );
  }

  async analyzeWithAI(userQuery, context = {}) {
    const prompt = `사용자 쿼리를 분석하여 데이터 분석 의도를 파악해주세요:

쿼리: "${userQuery}"

컨텍스트: ${JSON.stringify(context, null, 2)}

다음 JSON 형식으로 응답해주세요:
{
  "intent": "주요 의도 (analyze/visualize/train/predict)",
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
  },
  "reasoning": "의도 분석 근거"
}`;

    try {
      const response = await this.modelManager.queryModel('router', prompt, {
        temperature: 0.1,
        max_tokens: 800,
        timeout: 10000
      });

      return this.parseAIResponse(response);
    } catch (error) {
      this.logger.warn('AI 분석 실패:', error);
      return this.getDefaultAIAnalysis();
    }
  }

  parseAIResponse(response) {
    try {
      // JSON 블록 찾기
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        return this.validateAIResponse(parsed);
      }
    } catch (error) {
      this.logger.warn('AI 응답 파싱 실패:', error);
    }

    return this.getDefaultAIAnalysis();
  }

  validateAIResponse(aiResponse) {
    // 필수 필드 검증
    const requiredFields = ['intent', 'confidence', 'complexity'];
    const validated = {};

    for (const field of requiredFields) {
      if (aiResponse[field] !== undefined) {
        validated[field] = aiResponse[field];
      }
    }

    // 기본값 설정
    validated.intent = validated.intent || 'analyze';
    validated.confidence = Math.max(0, Math.min(1, validated.confidence || 0.5));
    validated.complexity = Math.max(0, Math.min(1, validated.complexity || 0.3));
    validated.workflow_type = aiResponse.workflow_type || 'single';
    validated.estimated_steps = Math.max(1, Math.min(10, aiResponse.estimated_steps || 1));
    validated.suggested_methods = Array.isArray(aiResponse.suggested_methods) ? 
      aiResponse.suggested_methods : ['analysis.basic'];
    validated.parameters = aiResponse.parameters || {};
    validated.data_requirements = aiResponse.data_requirements || {
      type: 'tabular',
      size: 'medium',
      preprocessing: 'minimal'
    };

    return validated;
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
      suggested_methods: ['analysis.basic'],
      parameters: {},
      reasoning: '기본 분석 모드'
    };
  }

  combineAnalysis(keywordAnalysis, patternAnalysis, aiAnalysis, context = {}) {
    const combinedConfidence = Math.max(
      keywordAnalysis.confidence,
      aiAnalysis ? aiAnalysis.confidence : 0
    );

    return {
      // 기본 정보
      intent: aiAnalysis?.intent || keywordAnalysis.primaryAction,
      confidence: combinedConfidence,
      complexity: aiAnalysis?.complexity || this.estimateComplexity(keywordAnalysis),
      
      // 키워드 분석 결과
      keywords: keywordAnalysis.foundKeywords,
      primary_action: keywordAnalysis.primaryAction,
      primary_analysis: keywordAnalysis.primaryAnalysis,
      
      // 패턴 분석 결과
      patterns: patternAnalysis,
      
      // AI 분석 결과
      ai_analysis: aiAnalysis,
      
      // 통합 분석
      workflow_type: this.determineWorkflowType(keywordAnalysis, patternAnalysis, aiAnalysis),
      estimated_steps: this.estimateSteps(keywordAnalysis, aiAnalysis),
      requires_pipeline: this.requiresPipeline(keywordAnalysis, aiAnalysis),
      
      // 데이터 요구사항
      data_requirements: aiAnalysis?.data_requirements || this.inferDataRequirements(keywordAnalysis),
      
      // 제안사항
      suggested_methods: this.suggestMethods(keywordAnalysis, aiAnalysis),
      parameters: this.extractParameters(keywordAnalysis, aiAnalysis),
      
      // 메타데이터
      context: context,
      timestamp: new Date().toISOString(),
      analysis_version: '1.0'
    };
  }

  estimateComplexity(keywordAnalysis) {
    const complexityFactors = {
      deep_learning: 0.9,
      ml: 0.7,
      advanced: 0.5,
      analysis: 0.3,
      visualization: 0.2
    };

    let maxComplexity = 0.2; // 기본 복잡도

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

  determineWorkflowType(keywordAnalysis, patternAnalysis, aiAnalysis) {
    // 순차적 처리가 필요한 경우
    const hasSequenceModifier = keywordAnalysis.foundKeywords.modifiers.some(
      mod => mod.type === 'sequence'
    );
    
    // 비교 분석이 필요한 경우
    const hasComparison = patternAnalysis.hasComparison || 
      keywordAnalysis.foundKeywords.modifiers.some(mod => mod.type === 'comparison');
    
    // 다중 분석이 필요한 경우
    const multipleAnalysis = keywordAnalysis.foundKeywords.analysis.length > 1;
    
    if (hasSequenceModifier || multipleAnalysis) {
      return 'sequential';
    }
    
    if (hasComparison) {
      return 'parallel';
    }
    
    return aiAnalysis?.workflow_type || 'single';
  }

  estimateSteps(keywordAnalysis, aiAnalysis) {
    const baseSteps = Math.max(keywordAnalysis.foundKeywords.analysis.length, 1);
    const hasVisualization = keywordAnalysis.foundKeywords.actions.some(
      action => action.type === 'visualize'
    );
    const hasTraining = keywordAnalysis.foundKeywords.actions.some(
      action => action.type === 'train'
    );
    
    let steps = baseSteps;
    if (hasVisualization) steps += 1;
    if (hasTraining) steps += 1;
    
    return Math.min(steps, aiAnalysis?.estimated_steps || 10);
  }

  requiresPipeline(keywordAnalysis, aiAnalysis) {
    return keywordAnalysis.foundKeywords.analysis.length > 1 ||
           keywordAnalysis.foundKeywords.modifiers.some(mod => mod.type === 'sequence') ||
           (aiAnalysis?.estimated_steps && aiAnalysis.estimated_steps > 1);
  }

  inferDataRequirements(keywordAnalysis) {
    const analysisTypes = keywordAnalysis.foundKeywords.analysis.map(a => a.type);
    
    let dataType = 'tabular';
    let preprocessing = 'minimal';
    
    // 이미지 관련 키워드가 있으면
    if (analysisTypes.some(type => ['cnn', 'image'].includes(type))) {
      dataType = 'image';
      preprocessing = 'moderate';
    }
    
    // 텍스트 관련 키워드가 있으면
    if (analysisTypes.some(type => ['rnn', 'transformer', 'nlp'].includes(type))) {
      dataType = 'text';
      preprocessing = 'extensive';
    }
    
    // 시계열 관련 키워드가 있으면
    if (analysisTypes.some(type => ['time_series', 'forecast'].includes(type))) {
      dataType = 'time_series';
      preprocessing = 'moderate';
    }
    
    return {
      type: dataType,
      size: 'medium',
      preprocessing: preprocessing
    };
  }

  suggestMethods(keywordAnalysis, aiAnalysis) {
    const methods = [];
    
    // 키워드 기반 방법 추천
    for (const item of keywordAnalysis.foundKeywords.analysis) {
      methods.push(`${item.category}.${item.type}`);
    }
    
    // AI 추천 방법 추가
    if (aiAnalysis?.suggested_methods) {
      methods.push(...aiAnalysis.suggested_methods);
    }
    
    // 기본 방법이 없으면 추가
    if (methods.length === 0) {
      methods.push('analysis.basic');
    }
    
    // 중복 제거
    return [...new Set(methods)];
  }

  extractParameters(keywordAnalysis, aiAnalysis) {
    const parameters = {};
    
    // 기술적 키워드에서 파라미터 추출
    const technical = keywordAnalysis.foundKeywords.technical;
    
    if (technical.numbers && technical.numbers.length > 0) {
      const firstNumber = technical.numbers[0];
      // 숫자는 클러스터 수나 컴포넌트 수로 추정
      if (keywordAnalysis.foundKeywords.analysis.some(a => a.type === 'clustering')) {
        parameters.n_clusters = firstNumber;
      }
      if (keywordAnalysis.foundKeywords.analysis.some(a => a.type === 'pca')) {
        parameters.n_components = firstNumber;
      }
    }
    
    if (technical.percentages && technical.percentages.length > 0) {
      parameters.test_size = technical.percentages[0] / 100;
    }
    
    // 모델 특정 파라미터
    if (technical.models && technical.models.length > 0) {
      parameters.model_type = technical.models[0];
    }
    
    // AI 분석에서 파라미터 추가
    if (aiAnalysis?.parameters) {
      Object.assign(parameters, aiAnalysis.parameters);
    }
    
    return parameters;
  }

  generateCacheKey(userQuery, context = {}) {
    const queryKey = userQuery.toLowerCase().replace(/\s+/g, '_').substring(0, 50);
    const contextKey = Object.keys(context).length > 0 ? 
      JSON.stringify(context).substring(0, 20) : '';
    return `intent_${queryKey}_${contextKey}`;
  }

  saveToCache(cacheKey, intent) {
    // 캐시 크기 제한
    if (this.intentCache.size >= this.maxCacheSize) {
      const firstKey = this.intentCache.keys().next().value;
      this.intentCache.delete(firstKey);
    }
    
    this.intentCache.set(cacheKey, intent);
  }

  getFallbackIntent(query) {
    return {
      intent: 'analyze',
      confidence: 0.3,
      complexity: 0.3,
      keywords: { 
        analysis: [], 
        actions: [], 
        modifiers: [], 
        technical: { numbers: [], percentages: [], file_formats: [], models: [] }
      },
      primary_action: 'analyze',
      primary_analysis: { category: 'analysis', type: 'basic' },
      patterns: {
        isQuestion: false,
        isRequest: false,
        hasComparison: false,
        hasUrgency: false,
        sentiment: 'neutral'
      },
      workflow_type: 'single',
      estimated_steps: 1,
      requires_pipeline: false,
      data_requirements: {
        type: 'tabular',
        size: 'medium',
        preprocessing: 'minimal'
      },
      suggested_methods: ['analysis.basic'],
      parameters: {},
      context: {},
      timestamp: new Date().toISOString(),
      analysis_version: '1.0'
    };
  }

  // 캐시 관리
  clearCache() {
    this.intentCache.clear();
    this.logger.info('의도 분석 캐시가 정리되었습니다.');
  }

  getCacheSize() {
    return this.intentCache.size;
  }

  // 통계 정보
  getStatistics() {
    const totalKeywords = Object.values(this.keywords).reduce((total, category) => {
      if (typeof category === 'object' && !Array.isArray(category)) {
        return total + Object.values(category).reduce((catTotal, keywords) => {
          return catTotal + (Array.isArray(keywords) ? keywords.length : 0);
        }, 0);
      }
      return total;
    }, 0);

    return {
      cache_size: this.intentCache.size,
      max_cache_size: this.maxCacheSize,
      keyword_categories: Object.keys(this.keywords).length,
      total_keywords: totalKeywords,
      pattern_types: Object.keys(this.patterns).length
    };
  }

  // 키워드 추가/수정
  addKeywords(category, subcategory, keywords) {
    if (!this.keywords[category]) {
      this.keywords[category] = {};
    }
    if (!this.keywords[category][subcategory]) {
      this.keywords[category][subcategory] = [];
    }
    this.keywords[category][subcategory].push(...keywords);
    this.logger.info(`키워드 추가됨: ${category}.${subcategory} - ${keywords.length}개`);
  }

  // 디버깅 및 테스트용
  debugAnalysis(query) {
    const keywordAnalysis = this.analyzeKeywords(query);
    const patternAnalysis = this.analyzePatterns(query);
    
    return {
      query: query,
      keywords: keywordAnalysis,
      patterns: patternAnalysis,
      technical: keywordAnalysis.foundKeywords.technical
    };
  }
}