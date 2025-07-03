// workflows/nlp-workflows.js
import { Logger } from '../utils/logger.js';

export class NLPWorkflows {
  constructor() {
    this.logger = new Logger();
    this.nlpWorkflowTemplates = this.initializeNLPWorkflows();
  }

  initializeNLPWorkflows() {
    return {
      text_preprocessing: {
        name: 'text_preprocessing',
        description: '텍스트 전처리 워크플로우',
        category: 'nlp_preprocessing',
        steps: [
          {
            type: 'nlp_preprocessing',
            method: 'text_cleaning',
            params: {
              remove_html: true,
              remove_urls: true,
              remove_email: true,
              remove_phone: true,
              remove_special_chars: true,
              normalize_whitespace: true
            },
            outputs: ['cleaned_text']
          },
          {
            type: 'nlp_preprocessing',
            method: 'text_normalization',
            params: {
              lowercase: true,
              remove_accents: true,
              normalize_unicode: true
            },
            outputs: ['normalized_text']
          },
          {
            type: 'nlp_preprocessing',
            method: 'tokenization',
            params: {
              tokenizer: 'word_tokenize',
              language: 'korean'
            },
            outputs: ['tokens']
          },
          {
            type: 'nlp_preprocessing',
            method: 'stopword_removal',
            params: {
              language: 'korean',
              custom_stopwords: []
            },
            outputs: ['filtered_tokens']
          },
          {
            type: 'nlp_preprocessing',
            method: 'stemming_lemmatization',
            params: {
              method: 'lemmatization',
              language: 'korean'
            },
            outputs: ['processed_tokens']
          },
          {
            type: 'visualization',
            method: 'text_statistics',
            params: {
              show_word_count: true,
              show_char_count: true,
              show_sentence_count: true
            },
            outputs: ['text_stats_viz']
          }
        ],
        estimated_time: 120,
        resource_requirements: {
          memory_mb: 800,
          cpu_cores: 2,
          gpu_required: false
        }
      },

      sentiment_analysis: {
        name: 'sentiment_analysis',
        description: '감정 분석 워크플로우',
        category: 'nlp_analysis',
        steps: [
          {
            type: 'nlp_preprocessing',
            method: 'text_preprocessing',
            params: {
              basic_cleaning: true,
              tokenization: true,
              remove_stopwords: true
            },
            outputs: ['preprocessed_text']
          },
          {
            type: 'nlp_analysis',
            method: 'rule_based_sentiment',
            params: {
              lexicon: 'vader',
              language: 'korean'
            },
            outputs: ['rule_sentiment_scores']
          },
          {
            type: 'nlp_analysis',
            method: 'ml_sentiment_analysis',
            params: {
              model_type: 'bert',
              pretrained_model: 'bert-base-multilingual-cased',
              num_labels: 3,
              labels: ['negative', 'neutral', 'positive']
            },
            outputs: ['ml_sentiment_scores', 'confidence_scores']
          },
          {
            type: 'nlp_analysis',
            method: 'emotion_detection',
            params: {
              emotions: ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust'],
              model_type: 'transformer'
            },
            outputs: ['emotion_scores']
          },
          {
            type: 'visualization',
            method: 'sentiment_distribution',
            params: {
              plot_type: 'bar',
              show_percentages: true
            },
            outputs: ['sentiment_distribution_plot']
          },
          {
            type: 'visualization',
            method: 'emotion_radar_chart',
            params: {
              normalize_scores: true
            },
            outputs: ['emotion_radar_plot']
          }
        ],
        estimated_time: 180,
        resource_requirements: {
          memory_mb: 2000,
          cpu_cores: 2,
          gpu_required: true
        }
      },

      text_classification: {
        name: 'text_classification',
        description: '텍스트 분류 워크플로우',
        category: 'nlp_classification',
        steps: [
          {
            type: 'nlp_preprocessing',
            method: 'text_preprocessing',
            params: {
              full_preprocessing: true,
              max_length: 512
            },
            outputs: ['preprocessed_text']
          },
          {
            type: 'nlp_feature_extraction',
            method: 'tfidf_vectorization',
            params: {
              max_features: 10000,
              ngram_range: [1, 2],
              min_df: 2,
              max_df: 0.95
            },
            outputs: ['tfidf_features']
          },
          {
            type: 'nlp_feature_extraction',
            method: 'word_embeddings',
            params: {
              model_type: 'word2vec',
              vector_size: 300,
              window: 5,
              min_count: 1
            },
            outputs: ['word_embeddings']
          },
          {
            type: 'ml_traditional',
            method: 'supervised.classification.svm',
            params: {
              kernel: 'rbf',
              C: 1.0,
              random_state: 42
            },
            outputs: ['svm_model', 'svm_predictions']
          },
          {
            type: 'nlp_deep_learning',
            method: 'transformer_classification',
            params: {
              model_name: 'bert-base-multilingual-cased',
              num_labels: 'auto',
              epochs: 3,
              batch_size: 16,
              learning_rate: 2e-5
            },
            outputs: ['transformer_model', 'transformer_predictions']
          },
          {
            type: 'ml_evaluation',
            method: 'classification_metrics',
            params: {
              average: 'weighted'
            },
            outputs: ['classification_report']
          },
          {
            type: 'visualization',
            method: 'confusion_matrix',
            params: {
              normalize: true
            },
            outputs: ['confusion_matrix_plot']
          }
        ],
        estimated_time: 400,
        resource_requirements: {
          memory_mb: 4000,
          cpu_cores: 4,
          gpu_required: true
        }
      },

      named_entity_recognition: {
        name: 'named_entity_recognition',
        description: '개체명 인식 워크플로우',
        category: 'nlp_ner',
        steps: [
          {
            type: 'nlp_preprocessing',
            method: 'text_preprocessing',
            params: {
              basic_cleaning: true,
              keep_punctuation: true,
              sentence_tokenization: true
            },
            outputs: ['preprocessed_sentences']
          },
          {
            type: 'nlp_analysis',
            method: 'rule_based_ner',
            params: {
              entity_types: ['PERSON', 'ORG', 'LOC', 'MISC'],
              use_gazetteer: true,
              language: 'korean'
            },
            outputs: ['rule_entities']
          },
          {
            type: 'nlp_deep_learning',
            method: 'transformer_ner',
            params: {
              model_name: 'bert-base-multilingual-cased',
              entity_types: ['PERSON', 'ORG', 'LOC', 'MISC', 'DATE', 'TIME'],
              max_length: 128,
              batch_size: 16
            },
            outputs: ['transformer_entities']
          },
          {
            type: 'nlp_analysis',
            method: 'entity_linking',
            params: {
              knowledge_base: 'wikidata',
              confidence_threshold: 0.7
            },
            outputs: ['linked_entities']
          },
          {
            type: 'nlp_analysis',
            method: 'entity_resolution',
            params: {
              merge_similar: true,
              similarity_threshold: 0.8
            },
            outputs: ['resolved_entities']
          },
          {
            type: 'visualization',
            method: 'entity_distribution',
            params: {
              plot_type: 'bar',
              top_n: 20
            },
            outputs: ['entity_distribution_plot']
          },
          {
            type: 'visualization',
            method: 'entity_network',
            params: {
              show_relationships: true,
              layout: 'spring'
            },
            outputs: ['entity_network_plot']
          }
        ],
        estimated_time: 300,
        resource_requirements: {
          memory_mb: 3000,
          cpu_cores: 3,
          gpu_required: true
        }
      },

      topic_modeling: {
        name: 'topic_modeling',
        description: '토픽 모델링 워크플로우',
        category: 'nlp_topic_modeling',
        steps: [
          {
            type: 'nlp_preprocessing',
            method: 'text_preprocessing',
            params: {
              full_preprocessing: true,
              remove_short_documents: true,
              min_doc_length: 50
            },
            outputs: ['preprocessed_corpus']
          },
          {
            type: 'nlp_feature_extraction',
            method: 'document_term_matrix',
            params: {
              vectorizer: 'tfidf',
              max_features: 5000,
              min_df: 5,
              max_df: 0.8
            },
            outputs: ['doc_term_matrix']
          },
          {
            type: 'nlp_topic_modeling',
            method: 'lda_topic_modeling',
            params: {
              n_topics: 10,
              random_state: 42,
              max_iter: 1000,
              alpha: 'auto',
              beta: 'auto'
            },
            outputs: ['lda_model', 'document_topics', 'topic_terms']
          },
          {
            type: 'nlp_topic_modeling',
            method: 'bertopic_modeling',
            params: {
              embedding_model: 'sentence-transformers/all-MiniLM-L6-v2',
              min_topic_size: 10,
              nr_topics: 'auto'
            },
            outputs: ['bertopic_model', 'bertopic_topics']
          },
          {
            type: 'nlp_analysis',
            method: 'topic_coherence',
            params: {
              coherence_measures: ['c_v', 'c_npmi', 'c_uci'],
              texts: 'preprocessed_corpus'
            },
            outputs: ['coherence_scores']
          },
          {
            type: 'visualization',
            method: 'topic_word_cloud',
            params: {
              top_words: 20,
              colormap: 'viridis'
            },
            outputs: ['topic_wordclouds']
          },
          {
            type: 'visualization',
            method: 'topic_distribution',
            params: {
              plot_type: 'heatmap',
              show_document_topics: true
            },
            outputs: ['topic_distribution_plot']
          },
          {
            type: 'visualization',
            method: 'topic_evolution',
            params: {
              time_column: 'date',
              window_size: 30
            },
            outputs: ['topic_evolution_plot']
          }
        ],
        estimated_time: 360,
        resource_requirements: {
          memory_mb: 4000,
          cpu_cores: 4,
          gpu_required: true
        }
      },

      text_summarization: {
        name: 'text_summarization',
        description: '텍스트 요약 워크플로우',
        category: 'nlp_summarization',
        steps: [
          {
            type: 'nlp_preprocessing',
            method: 'text_preprocessing',
            params: {
              sentence_tokenization: true,
              preserve_structure: true
            },
            outputs: ['preprocessed_sentences']
          },
          {
            type: 'nlp_analysis',
            method: 'extractive_summarization',
            params: {
              algorithm: 'textrank',
              summary_ratio: 0.3,
              sentence_count: 5
            },
            outputs: ['extractive_summary']
          },
          {
            type: 'nlp_deep_learning',
            method: 'abstractive_summarization',
            params: {
              model_name: 'facebook/bart-large-cnn',
              max_length: 200,
              min_length: 50,
              beam_size: 4
            },
            outputs: ['abstractive_summary']
          },
          {
            type: 'nlp_analysis',
            method: 'keyword_extraction',
            params: {
              method: 'yake',
              num_keywords: 10,
              language: 'korean'
            },
            outputs: ['keywords']
          },
          {
            type: 'nlp_evaluation',
            method: 'summary_evaluation',
            params: {
              metrics: ['rouge', 'bleu', 'meteor'],
              reference_summary: 'ground_truth'
            },
            outputs: ['summary_scores']
          },
          {
            type: 'visualization',
            method: 'summary_comparison',
            params: {
              show_original: true,
              show_extractive: true,
              show_abstractive: true
            },
            outputs: ['summary_comparison_viz']
          }
        ],
        estimated_time: 250,
        resource_requirements: {
          memory_mb: 3000,
          cpu_cores: 3,
          gpu_required: true
        }
      },

      question_answering: {
        name: 'question_answering',
        description: '질의응답 워크플로우',
        category: 'nlp_qa',
        steps: [
          {
            type: 'nlp_preprocessing',
            method: 'context_preprocessing',
            params: {
              chunk_size: 512,
              overlap: 50,
              preserve_sentences: true
            },
            outputs: ['context_chunks']
          },
          {
            type: 'nlp_preprocessing',
            method: 'question_preprocessing',
            params: {
              normalize_question: true,
              question_type_detection: true
            },
            outputs: ['processed_questions']
          },
          {
            type: 'nlp_analysis',
            method: 'passage_retrieval',
            params: {
              retrieval_method: 'bm25',
              top_k: 5,
              similarity_threshold: 0.7
            },
            outputs: ['relevant_passages']
          },
          {
            type: 'nlp_deep_learning',
            method: 'reading_comprehension',
            params: {
              model_name: 'bert-large-uncased-whole-word-masking-finetuned-squad',
              max_seq_length: 384,
              doc_stride: 128,
              max_query_length: 64
            },
            outputs: ['answer_spans', 'confidence_scores']
          },
          {
            type: 'nlp_analysis',
            method: 'answer_validation',
            params: {
              min_confidence: 0.5,
              answer_type_matching: true
            },
            outputs: ['validated_answers']
          },
          {
            type: 'nlp_evaluation',
            method: 'qa_evaluation',
            params: {
              metrics: ['exact_match', 'f1_score', 'squad_score'],
              ground_truth_answers: 'reference_answers'
            },
            outputs: ['qa_metrics']
          },
          {
            type: 'visualization',
            method: 'qa_performance',
            params: {
              show_confidence_distribution: true,
              show_answer_lengths: true
            },
            outputs: ['qa_performance_viz']
          }
        ],
        estimated_time: 300,
        resource_requirements: {
          memory_mb: 4000,
          cpu_cores: 4,
          gpu_required: true
        }
      },

      language_detection: {
        name: 'language_detection',
        description: '언어 감지 워크플로우',
        category: 'nlp_language',
        steps: [
          {
            type: 'nlp_preprocessing',
            method: 'text_segmentation',
            params: {
              segment_by: 'sentence',
              min_length: 10
            },
            outputs: ['text_segments']
          },
          {
            type: 'nlp_analysis',
            method: 'language_detection',
            params: {
              detector: 'langdetect',
              confidence_threshold: 0.8,
              supported_languages: ['ko', 'en', 'ja', 'zh', 'es', 'fr', 'de', 'ru']
            },
            outputs: ['detected_languages', 'confidence_scores']
          },
          {
            type: 'nlp_analysis',
            method: 'script_detection',
            params: {
              detect_scripts: ['latin', 'hangul', 'hiragana', 'katakana', 'han']
            },
            outputs: ['detected_scripts']
          },
          {
            type: 'nlp_analysis',
            method: 'multilingual_analysis',
            params: {
              identify_code_switching: true,
              segment_by_language: true
            },
            outputs: ['multilingual_segments']
          },
          {
            type: 'visualization',
            method: 'language_distribution',
            params: {
              plot_type: 'pie',
              show_percentages: true
            },
            outputs: ['language_distribution_plot']
          }
        ],
        estimated_time: 90,
        resource_requirements: {
          memory_mb: 500,
          cpu_cores: 1,
          gpu_required: false
        }
      },

      text_similarity: {
        name: 'text_similarity',
        description: '텍스트 유사도 분석 워크플로우',
        category: 'nlp_similarity',
        steps: [
          {
            type: 'nlp_preprocessing',
            method: 'text_preprocessing',
            params: {
              full_preprocessing: true,
              preserve_order: true
            },
            outputs: ['preprocessed_texts']
          },
          {
            type: 'nlp_feature_extraction',
            method: 'sentence_embeddings',
            params: {
              model_name: 'sentence-transformers/all-MiniLM-L6-v2',
              normalize_embeddings: true
            },
            outputs: ['sentence_embeddings']
          },
          {
            type: 'nlp_analysis',
            method: 'cosine_similarity',
            params: {
              similarity_threshold: 0.7,
              return_matrix: true
            },
            outputs: ['cosine_similarity_matrix']
          },
          {
            type: 'nlp_analysis',
            method: 'semantic_similarity',
            params: {
              methods: ['jaccard', 'levenshtein', 'semantic'],
              aggregate_method: 'weighted_average'
            },
            outputs: ['semantic_similarity_scores']
          },
          {
            type: 'nlp_analysis',
            method: 'clustering_by_similarity',
            params: {
              clustering_method: 'hierarchical',
              similarity_threshold: 0.8
            },
            outputs: ['similarity_clusters']
          },
          {
            type: 'visualization',
            method: 'similarity_heatmap',
            params: {
              annotate: true,
              colormap: 'viridis'
            },
            outputs: ['similarity_heatmap']
          },
          {
            type: 'visualization',
            method: 'similarity_network',
            params: {
              edge_threshold: 0.7,
              layout: 'spring',
              node_size_by_degree: true
            },
            outputs: ['similarity_network_plot']
          }
        ],
        estimated_time: 200,
        resource_requirements: {
          memory_mb: 2000,
          cpu_cores: 2,
          gpu_required: true
        }
      },

      machine_translation: {
        name: 'machine_translation',
        description: '기계 번역 워크플로우',
        category: 'nlp_translation',
        steps: [
          {
            type: 'nlp_preprocessing',
            method: 'translation_preprocessing',
            params: {
              source_language: 'auto',
              target_language: 'en',
              preserve_formatting: true
            },
            outputs: ['preprocessed_source']
          },
          {
            type: 'nlp_analysis',
            method: 'language_detection',
            params: {
              confidence_threshold: 0.9
            },
            outputs: ['detected_source_language']
          },
          {
            type: 'nlp_deep_learning',
            method: 'neural_translation',
            params: {
              model_name: 'facebook/m2m100_418M',
              beam_size: 5,
              max_length: 512,
              early_stopping: true
            },
            outputs: ['neural_translation']
          },
          {
            type: 'nlp_analysis',
            method: 'rule_based_translation',
            params: {
              dictionary_augmentation: true,
              phrase_table: true
            },
            outputs: ['rule_based_translation']
          },
          {
            type: 'nlp_evaluation',
            method: 'translation_evaluation',
            params: {
              metrics: ['bleu', 'meteor', 'ter', 'bertscore'],
              reference_translation: 'ground_truth'
            },
            outputs: ['translation_scores']
          },
          {
            type: 'nlp_postprocessing',
            method: 'translation_postprocessing',
            params: {
              fix_casing: true,
              fix_punctuation: true,
              remove_extra_spaces: true
            },
            outputs: ['final_translation']
          },
          {
            type: 'visualization',
            method: 'translation_comparison',
            params: {
              show_source: true,
              show_multiple_translations: true,
              highlight_differences: true
            },
            outputs: ['translation_comparison_viz']
          }
        ],
        estimated_time: 180,
        resource_requirements: {
          memory_mb: 2500,
          cpu_cores: 3,
          gpu_required: true
        }
      }
    };
  }

  getWorkflow(workflowName) {
    return this.nlpWorkflowTemplates[workflowName] || null;
  }

  getAllWorkflows() {
    return this.nlpWorkflowTemplates;
  }

  getWorkflowsByCategory(category) {
    const workflows = {};
    for (const [name, workflow] of Object.entries(this.nlpWorkflowTemplates)) {
      if (workflow.category === category) {
        workflows[name] = workflow;
      }
    }
    return workflows;
  }

  getAvailableCategories() {
    const categories = new Set();
    for (const workflow of Object.values(this.nlpWorkflowTemplates)) {
      categories.add(workflow.category);
    }
    return Array.from(categories);
  }

  customizeNLPWorkflow(workflowName, customizations) {
    const baseWorkflow = this.getWorkflow(workflowName);
    if (!baseWorkflow) {
      throw new Error(`NLP 워크플로우 '${workflowName}'을 찾을 수 없습니다.`);
    }

    const customizedWorkflow = JSON.parse(JSON.stringify(baseWorkflow));

    // 언어 설정 커스터마이징
    if (customizations.language) {
      customizedWorkflow.steps.forEach(step => {
        if (step.params && step.params.language) {
          step.params.language = customizations.language;
        }
      });
    }

    // 모델 설정 커스터마이징
    if (customizations.models) {
      customizedWorkflow.steps.forEach(step => {
        if (step.type.includes('nlp_') && customizations.models[step.method]) {
          step.params.model_name = customizations.models[step.method];
        }
      });
    }

    // 배치 크기 및 성능 설정
    if (customizations.performance) {
      customizedWorkflow.steps.forEach(step => {
        if (step.type.includes('deep_learning')) {
          step.params.batch_size = customizations.performance.batch_size || step.params.batch_size;
          step.params.max_length = customizations.performance.max_length || step.params.max_length;
        }
      });
    }

    return customizedWorkflow;
  }

  generateNLPPipeline(taskType, textInfo, requirements = {}) {
    const generators = {
      'sentiment': this.generateSentimentPipeline,
      'classification': this.generateClassificationPipeline,
      'ner': this.generateNERPipeline,
      'topic_modeling': this.generateTopicModelingPipeline,
      'summarization': this.generateSummarizationPipeline,
      'qa': this.generateQAPipeline,
      'translation': this.generateTranslationPipeline,
      'similarity': this.generateSimilarityPipeline
    };

    const generator = generators[taskType];
    if (!generator) {
      throw new Error(`지원하지 않는 NLP 작업 유형: ${taskType}`);
    }

    return generator.call(this, textInfo, requirements);
  }

  generateSentimentPipeline(textInfo, requirements) {
    const pipeline = {
      name: 'custom_sentiment_pipeline',
      description: '커스텀 감정 분석 파이프라인',
      category: 'nlp_custom',
      steps: []
    };

    // 전처리
    pipeline.steps.push({
      type: 'nlp_preprocessing',
      method: 'text_preprocessing',
      params: {
        sentence_tokenization: true,
        keep_punctuation: true
      },
      outputs: ['preprocessed_text']
    });

    // 개체명 인식 방법
    if (requirements.use_rule_based) {
      pipeline.steps.push({
        type: 'nlp_analysis',
        method: 'rule_based_ner',
        params: {
          entity_types: requirements.entity_types || ['PERSON', 'ORG', 'LOC']
        },
        outputs: ['rule_entities']
      });
    }

    // 트랜스포머 기반 NER
    pipeline.steps.push({
      type: 'nlp_deep_learning',
      method: 'transformer_ner',
      params: {
        model_name: requirements.model_name || 'bert-base-multilingual-cased',
        entity_types: requirements.entity_types || ['PERSON', 'ORG', 'LOC', 'MISC']
      },
      outputs: ['transformer_entities']
    });

    // 시각화
    pipeline.steps.push({
      type: 'visualization',
      method: 'entity_distribution',
      params: {},
      outputs: ['entity_viz']
    });

    return pipeline;
  }

  generateTopicModelingPipeline(textInfo, requirements) {
    const pipeline = {
      name: 'custom_topic_modeling_pipeline',
      description: '커스텀 토픽 모델링 파이프라인',
      category: 'nlp_custom',
      steps: []
    };

    // 전처리
    pipeline.steps.push({
      type: 'nlp_preprocessing',
      method: 'text_preprocessing',
      params: {
        full_preprocessing: true,
        min_doc_length: requirements.min_doc_length || 50
      },
      outputs: ['preprocessed_corpus']
    });

    // 토픽 모델링 방법
    const methods = requirements.methods || ['lda'];
    methods.forEach(method => {
      if (method === 'lda') {
        pipeline.steps.push({
          type: 'nlp_topic_modeling',
          method: 'lda_topic_modeling',
          params: {
            n_topics: requirements.n_topics || 10
          },
          outputs: ['lda_topics']
        });
      } else if (method === 'bertopic') {
        pipeline.steps.push({
          type: 'nlp_topic_modeling',
          method: 'bertopic_modeling',
          params: {
            min_topic_size: requirements.min_topic_size || 10
          },
          outputs: ['bertopic_topics']
        });
      }
    });

    // 시각화
    pipeline.steps.push({
      type: 'visualization',
      method: 'topic_word_cloud',
      params: {},
      outputs: ['topic_viz']
    });

    return pipeline;
  }

  generateSummarizationPipeline(textInfo, requirements) {
    const pipeline = {
      name: 'custom_summarization_pipeline',
      description: '커스텀 텍스트 요약 파이프라인',
      category: 'nlp_custom',
      steps: []
    };

    // 전처리
    pipeline.steps.push({
      type: 'nlp_preprocessing',
      method: 'text_preprocessing',
      params: {
        sentence_tokenization: true,
        preserve_structure: true
      },
      outputs: ['preprocessed_text']
    });

    // 요약 방법
    if (requirements.method === 'extractive' || requirements.method === 'hybrid') {
      pipeline.steps.push({
        type: 'nlp_analysis',
        method: 'extractive_summarization',
        params: {
          algorithm: requirements.algorithm || 'textrank',
          summary_ratio: requirements.summary_ratio || 0.3
        },
        outputs: ['extractive_summary']
      });
    }

    if (requirements.method === 'abstractive' || requirements.method === 'hybrid') {
      pipeline.steps.push({
        type: 'nlp_deep_learning',
        method: 'abstractive_summarization',
        params: {
          model_name: requirements.model_name || 'facebook/bart-large-cnn',
          max_length: requirements.max_length || 200
        },
        outputs: ['abstractive_summary']
      });
    }

    return pipeline;
  }

  generateQAPipeline(textInfo, requirements) {
    const pipeline = {
      name: 'custom_qa_pipeline',
      description: '커스텀 질의응답 파이프라인',
      category: 'nlp_custom',
      steps: []
    };

    // 컨텍스트 전처리
    pipeline.steps.push({
      type: 'nlp_preprocessing',
      method: 'context_preprocessing',
      params: {
        chunk_size: requirements.chunk_size || 512,
        overlap: requirements.overlap || 50
      },
      outputs: ['context_chunks']
    });

    // 질문 전처리
    pipeline.steps.push({
      type: 'nlp_preprocessing',
      method: 'question_preprocessing',
      params: {
        normalize_question: true
      },
      outputs: ['processed_questions']
    });

    // 패세지 검색
    if (requirements.use_retrieval) {
      pipeline.steps.push({
        type: 'nlp_analysis',
        method: 'passage_retrieval',
        params: {
          retrieval_method: requirements.retrieval_method || 'bm25',
          top_k: requirements.top_k || 5
        },
        outputs: ['relevant_passages']
      });
    }

    // 질의응답 모델
    pipeline.steps.push({
      type: 'nlp_deep_learning',
      method: 'reading_comprehension',
      params: {
        model_name: requirements.model_name || 'bert-large-uncased-whole-word-masking-finetuned-squad'
      },
      outputs: ['answers']
    });

    return pipeline;
  }

  generateTranslationPipeline(textInfo, requirements) {
    const pipeline = {
      name: 'custom_translation_pipeline',
      description: '커스텀 기계번역 파이프라인',
      category: 'nlp_custom',
      steps: []
    };

    // 전처리
    pipeline.steps.push({
      type: 'nlp_preprocessing',
      method: 'translation_preprocessing',
      params: {
        source_language: requirements.source_language || 'auto',
        target_language: requirements.target_language || 'en'
      },
      outputs: ['preprocessed_source']
    });

    // 언어 감지
    if (requirements.source_language === 'auto') {
      pipeline.steps.push({
        type: 'nlp_analysis',
        method: 'language_detection',
        params: {},
        outputs: ['detected_language']
      });
    }

    // 번역 모델
    pipeline.steps.push({
      type: 'nlp_deep_learning',
      method: 'neural_translation',
      params: {
        model_name: requirements.model_name || 'facebook/m2m100_418M',
        beam_size: requirements.beam_size || 5
      },
      outputs: ['translation']
    });

    // 후처리
    pipeline.steps.push({
      type: 'nlp_postprocessing',
      method: 'translation_postprocessing',
      params: {
        fix_casing: true,
        fix_punctuation: true
      },
      outputs: ['final_translation']
    });

    return pipeline;
  }

  generateSimilarityPipeline(textInfo, requirements) {
    const pipeline = {
      name: 'custom_similarity_pipeline',
      description: '커스텀 텍스트 유사도 파이프라인',
      category: 'nlp_custom',
      steps: []
    };

    // 전처리
    pipeline.steps.push({
      type: 'nlp_preprocessing',
      method: 'text_preprocessing',
      params: {
        preserve_order: true
      },
      outputs: ['preprocessed_texts']
    });

    // 임베딩 생성
    pipeline.steps.push({
      type: 'nlp_feature_extraction',
      method: 'sentence_embeddings',
      params: {
        model_name: requirements.model_name || 'sentence-transformers/all-MiniLM-L6-v2'
      },
      outputs: ['embeddings']
    });

    // 유사도 계산
    pipeline.steps.push({
      type: 'nlp_analysis',
      method: 'cosine_similarity',
      params: {
        similarity_threshold: requirements.similarity_threshold || 0.7
      },
      outputs: ['similarity_matrix']
    });

    // 시각화
    pipeline.steps.push({
      type: 'visualization',
      method: 'similarity_heatmap',
      params: {},
      outputs: ['similarity_viz']
    });

    return pipeline;
  }

  validateNLPWorkflow(workflow) {
    const validationResult = {
      valid: true,
      errors: [],
      warnings: []
    };

    // NLP 워크플로우 특화 검증
    const nlpSteps = workflow.steps.filter(step => step.type.startsWith('nlp_'));
    if (nlpSteps.length === 0) {
      validationResult.errors.push('NLP 관련 단계가 없습니다.');
      validationResult.valid = false;
    }

    // 전처리 단계 확인
    const preprocessingSteps = workflow.steps.filter(step => step.type === 'nlp_preprocessing');
    if (preprocessingSteps.length === 0) {
      validationResult.warnings.push('텍스트 전처리 단계가 없습니다.');
    }

    // 언어 설정 확인
    workflow.steps.forEach((step, index) => {
      if (step.type.startsWith('nlp_') && step.params.language) {
        const supportedLanguages = ['korean', 'english', 'chinese', 'japanese', 'multilingual'];
        if (!supportedLanguages.includes(step.params.language)) {
          validationResult.warnings.push(`단계 ${index + 1}: 지원하지 않는 언어 설정입니다.`);
        }
      }
    });

    // GPU 요구사항 확인
    const deepLearningSteps = workflow.steps.filter(step => step.type === 'nlp_deep_learning');
    if (deepLearningSteps.length > 0) {
      if (!workflow.resource_requirements || !workflow.resource_requirements.gpu_required) {
        validationResult.warnings.push('딥러닝 모델 사용 시 GPU가 필요합니다.');
      }
    }

    // 메모리 사용량 검증
    const estimatedMemory = this.estimateNLPMemory(workflow);
    if (estimatedMemory > 8000) {
      validationResult.warnings.push('높은 메모리 사용량이 예상됩니다. 시스템 리소스를 확인하세요.');
    }

    return validationResult;
  }

  estimateNLPMemory(workflow) {
    const memoryEstimates = {
      'nlp_preprocessing': 200,
      'nlp_analysis': 300,
      'nlp_feature_extraction': 500,
      'nlp_deep_learning': 2000,
      'nlp_topic_modeling': 800,
      'nlp_evaluation': 100,
      'nlp_postprocessing': 100
    };

    let totalMemory = 0;
    workflow.steps.forEach(step => {
      const baseMemory = memoryEstimates[step.type] || 200;
      
      // 딥러닝 모델별 메모리 추가
      if (step.type === 'nlp_deep_learning') {
        if (step.params.model_name && step.params.model_name.includes('large')) {
          totalMemory += baseMemory * 2;
        } else {
          totalMemory += baseMemory;
        }
      } else {
        totalMemory += baseMemory;
      }
    });

    return totalMemory;
  }

  optimizeNLPWorkflow(workflow) {
    const optimized = JSON.parse(JSON.stringify(workflow));

    // 1. 전처리 단계 통합
    const preprocessingSteps = optimized.steps.filter(step => step.type === 'nlp_preprocessing');
    if (preprocessingSteps.length > 1) {
      // 중복 전처리 단계 제거 및 통합
      const combinedPreprocessing = {
        type: 'nlp_preprocessing',
        method: 'text_preprocessing',
        params: {},
        outputs: ['preprocessed_text']
      };

      // 모든 전처리 파라미터 통합
      preprocessingSteps.forEach(step => {
        Object.assign(combinedPreprocessing.params, step.params);
      });

      // 기존 전처리 단계 제거 후 통합 단계 추가
      optimized.steps = optimized.steps.filter(step => step.type !== 'nlp_preprocessing');
      optimized.steps.unshift(combinedPreprocessing);
    }

    // 2. 모델 크기 최적화
    optimized.steps.forEach(step => {
      if (step.type === 'nlp_deep_learning') {
        if (!step.params.model_name || step.params.model_name.includes('large')) {
          // 대용량 모델을 중간 크기로 변경
          const modelMap = {
            'bert-large-uncased': 'bert-base-uncased',
            'roberta-large': 'roberta-base',
            'facebook/bart-large-cnn': 'facebook/bart-base'
          };
          
          const currentModel = step.params.model_name;
          if (modelMap[currentModel]) {
            step.params.model_name = modelMap[currentModel];
          }
        }
      }
    });

    // 3. 배치 크기 최적화
    optimized.steps.forEach(step => {
      if (step.type === 'nlp_deep_learning' && step.params.batch_size > 16) {
        step.params.batch_size = Math.min(step.params.batch_size, 16);
      }
    });

    return optimized;
  }

  exportNLPWorkflow(workflowName, format = 'json') {
    const workflow = this.getWorkflow(workflowName);
    if (!workflow) {
      throw new Error(`NLP 워크플로우 '${workflowName}'을 찾을 수 없습니다.`);
    }

    switch (format) {
      case 'json':
        return JSON.stringify(workflow, null, 2);
      case 'python':
        return this.convertToNLPPythonScript(workflow);
      case 'huggingface':
        return this.convertToHuggingFaceScript(workflow);
      case 'spacy':
        return this.convertToSpacyScript(workflow);
      default:
        return workflow;
    }
  }

  convertToNLPPythonScript(workflow) {
    let script = `# ${workflow.name} - ${workflow.description}\n`;
    script += `import pandas as pd\nimport numpy as np\n`;
    script += `from transformers import pipeline, AutoTokenizer, AutoModel\n`;
    script += `import torch\nfrom sklearn.feature_extraction.text import TfidfVectorizer\n`;
    script += `import nltk\nfrom nltk.corpus import stopwords\n`;
    script += `from gensim import corpora, models\n\n`;

    script += `# 데이터 로드\n# texts = pd.read_csv('your_text_data.csv')['text'].tolist()\n\n`;

    workflow.steps.forEach((step, index) => {
      if (step.type.startsWith('nlp_')) {
        script += `# 단계 ${index + 1}: ${step.method}\n`;
        script += this.generateNLPPythonCode(step);
        script += `\n`;
      }
    });

    return script;
  }

  convertToHuggingFaceScript(workflow) {
    let script = `# ${workflow.name} - Hugging Face 전용\n`;
    script += `from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification\n`;
    script += `import torch\n\n`;

    workflow.steps.forEach((step, index) => {
      if (step.type === 'nlp_deep_learning') {
        script += `# 단계 ${index + 1}: ${step.method}\n`;
        script += this.generateHuggingFaceCode(step);
        script += `\n`;
      }
    });

    return script;
  }

  convertToSpacyScript(workflow) {
    let script = `# ${workflow.name} - spaCy 전용\n`;
    script += `import spacy\nfrom spacy import displacy\n`;
    script += `import pandas as pd\n\n`;
    script += `# 언어 모델 로드\nnlp = spacy.load('ko_core_news_sm')\n\n`;

    workflow.steps.forEach((step, index) => {
      if (step.type.startsWith('nlp_')) {
        script += `# 단계 ${index + 1}: ${step.method}\n`;
        script += this.generateSpacyCode(step);
        script += `\n`;
      }
    });

    return script;
  }

  generateNLPPythonCode(step) {
    const codeTemplates = {
      'text_preprocessing': `
# 텍스트 전처리
def preprocess_text(texts):
    processed = []
    for text in texts:
        # 기본 정제
        text = text.lower().strip()
        processed.append(text)
    return processed

preprocessed_texts = preprocess_text(texts)`,

      'sentiment_analysis': `
# 감정 분석
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
sentiment_results = sentiment_pipeline(preprocessed_texts)`,

      'text_classification': `
# 텍스트 분류
classifier = pipeline("text-classification", model="bert-base-multilingual-cased")
classification_results = classifier(preprocessed_texts)`,

      'named_entity_recognition': `
# 개체명 인식
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
ner_results = ner_pipeline(preprocessed_texts)`,

      'topic_modeling': `
# 토픽 모델링
from gensim import corpora, models
dictionary = corpora.Dictionary([text.split() for text in preprocessed_texts])
corpus = [dictionary.doc2bow(text.split()) for text in preprocessed_texts]
lda_model = models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=10)`,

      'text_summarization': `
# 텍스트 요약
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summaries = summarizer(preprocessed_texts, max_length=100, min_length=30, do_sample=False)`
    };

    return codeTemplates[step.method] || `# ${step.method} 코드를 여기에 추가하세요`;
  }

  generateHuggingFaceCode(step) {
    const codeTemplates = {
      'transformer_classification': `
tokenizer = AutoTokenizer.from_pretrained("${step.params.model_name}")
model = AutoModelForSequenceClassification.from_pretrained("${step.params.model_name}")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
results = classifier(texts)`,

      'transformer_ner': `
ner_pipeline = pipeline("ner", model="${step.params.model_name}", aggregation_strategy="simple")
ner_results = ner_pipeline(texts)`,

      'abstractive_summarization': `
summarizer = pipeline("summarization", model="${step.params.model_name}")
summaries = summarizer(texts, max_length=${step.params.max_length}, min_length=${step.params.min_length})`
    };

    return codeTemplates[step.method] || `# ${step.method} Hugging Face 코드를 여기에 추가하세요`;
  }

  generateSpacyCode(step) {
    const codeTemplates = {
      'text_preprocessing': `
# spaCy 전처리
docs = [nlp(text) for text in texts]
processed_texts = [' '.join([token.lemma_ for token in doc if not token.is_stop]) for doc in docs]`,

      'named_entity_recognition': `
# spaCy NER
docs = [nlp(text) for text in texts]
entities = []
for doc in docs:
    doc_entities = [(ent.text, ent.label_) for ent in doc.ents]
    entities.append(doc_entities)`,

      'pos_tagging': `
# 품사 태깅
docs = [nlp(text) for text in texts]
pos_tags = [[(token.text, token.pos_) for token in doc] for doc in docs]`
    };

    return codeTemplates[step.method] || `# ${step.method} spaCy 코드를 여기에 추가하세요`;
  }

  getNLPRecommendations(textInfo) {
    const recommendations = [];

    // 언어별 추천
    if (textInfo.language === 'korean') {
      recommendations.push({
        workflow: 'sentiment_analysis',
        reason: '한국어 감정 분석에 특화된 모델을 사용합니다.',
        priority: 'high'
      });
    }

    // 텍스트 길이별 추천
    if (textInfo.averageLength > 1000) {
      recommendations.push({
        workflow: 'text_summarization',
        reason: '긴 텍스트에 대한 요약이 필요합니다.',
        priority: 'high'
      });
    }

    // 문서 수량별 추천
    if (textInfo.documentCount > 100) {
      recommendations.push({
        workflow: 'topic_modeling',
        reason: '다수의 문서에서 토픽을 발견할 수 있습니다.',
        priority: 'medium'
      });
    }

    // 도메인별 추천
    if (textInfo.domain === 'news') {
      recommendations.push({
        workflow: 'named_entity_recognition',
        reason: '뉴스 텍스트에서 인명, 기관명 등을 추출할 수 있습니다.',
        priority: 'high'
      });
    }

    if (textInfo.domain === 'social_media') {
      recommendations.push({
        workflow: 'sentiment_analysis',
        reason: '소셜 미디어 텍스트의 감정 분석이 유용합니다.',
        priority: 'high'
      });
    }

    return recommendations.sort((a, b) => {
      const priorityOrder = { 'high': 3, 'medium': 2, 'low': 1 };
      return priorityOrder[b.priority] - priorityOrder[a.priority];
    });
  }

  createNLPSummary(workflowResults) {
    const summary = {
      totalNLPSteps: 0,
      nlpStepTypes: {},
      processedTexts: 0,
      extractedEntities: 0,
      detectedTopics: 0,
      errors: [],
      insights: []
    };

    workflowResults.steps.forEach(step => {
      if (step.type.startsWith('nlp_')) {
        summary.totalNLPSteps++;
        
        if (!summary.nlpStepTypes[step.type]) {
          summary.nlpStepTypes[step.type] = 0;
        }
        summary.nlpStepTypes[step.type]++;

        if (step.success && step.result) {
          // 처리된 텍스트 수 계산
          if (step.result.processed_count) {
            summary.processedTexts += step.result.processed_count;
          }

          // 추출된 엔티티 수 계산
          if (step.result.entities && step.result.entities.length) {
            summary.extractedEntities += step.result.entities.length;
          }

          // 발견된 토픽 수 계산
          if (step.result.topics && step.result.topics.length) {
            summary.detectedTopics += step.result.topics.length;
          }
        }

        if (!step.success) {
          summary.errors.push({
            step: step.method,
            error: step.error
          });
        }
      }
    });

    // 인사이트 생성
    if (summary.extractedEntities > 0) {
      summary.insights.push(`총 ${summary.extractedEntities}개의 개체명이 추출되었습니다.`);
    }

    if (summary.detectedTopics > 0) {
      summary.insights.push(`${summary.detectedTopics}개의 주요 토픽이 발견되었습니다.`);
    }

    if (summary.errors.length === 0) {
      summary.insights.push('모든 NLP 단계가 성공적으로 완료되었습니다.');
    }

    return summary;
  }
}
      type: 'nlp_preprocessing',
      method: 'text_preprocessing',
      params: {
        language: requirements.language || 'korean',
        remove_emoticons: requirements.remove_emoticons || false
      },
      outputs: ['preprocessed_text']
    });

    // 감정 분석 방법 선택
    if (requirements.method === 'rule_based' || requirements.method === 'hybrid') {
      pipeline.steps.push({
        type: 'nlp_analysis',
        method: 'rule_based_sentiment',
        params: {
          lexicon: requirements.lexicon || 'vader'
        },
        outputs: ['rule_sentiment']
      });
    }

    if (requirements.method === 'ml_based' || requirements.method === 'hybrid' || !requirements.method) {
      pipeline.steps.push({
        type: 'nlp_analysis',
        method: 'ml_sentiment_analysis',
        params: {
          model_type: requirements.model_type || 'bert',
          num_labels: requirements.num_labels || 3
        },
        outputs: ['ml_sentiment']
      });
    }

    // 시각화
    pipeline.steps.push({
      type: 'visualization',
      method: 'sentiment_distribution',
      params: {},
      outputs: ['sentiment_viz']
    });

    return pipeline;
  }

  generateClassificationPipeline(textInfo, requirements) {
    const pipeline = {
      name: 'custom_classification_pipeline',
      description: '커스텀 텍스트 분류 파이프라인',
      category: 'nlp_custom',
      steps: []
    };

    // 전처리
    pipeline.steps.push({
      type: 'nlp_preprocessing',
      method: 'text_preprocessing',
      params: {
        max_length: requirements.max_length || 512
      },
      outputs: ['preprocessed_text']
    });

    // 특징 추출
    if (requirements.use_tfidf) {
      pipeline.steps.push({
        type: 'nlp_feature_extraction',
        method: 'tfidf_vectorization',
        params: {
          max_features: requirements.max_features || 10000
        },
        outputs: ['tfidf_features']
      });
    }

    // 분류 모델
    const models = requirements.models || ['svm', 'transformer'];
    models.forEach(model => {
      if (model === 'svm') {
        pipeline.steps.push({
          type: 'ml_traditional',
          method: 'supervised.classification.svm',
          params: {},
          outputs: ['svm_predictions']
        });
      } else if (model === 'transformer') {
        pipeline.steps.push({
          type: 'nlp_deep_learning',
          method: 'transformer_classification',
          params: {
            model_name: requirements.model_name || 'bert-base-multilingual-cased'
          },
          outputs: ['transformer_predictions']
        });
      }
    });

    return pipeline;
  }

  generateNERPipeline(textInfo, requirements) {
    const pipeline = {
      name: 'custom_ner_pipeline',
      description: '커스텀 개체명 인식 파이프라인',
      category: 'nlp_custom',
      steps: []
    };

    // 전처리
    pipeline.steps.push({
