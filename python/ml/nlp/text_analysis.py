#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Natural Language Processing and Text Analysis
자연어 처리 및 텍스트 분석

이 모듈은 NLP 및 텍스트 분석 기능을 구현합니다.
주요 기능:
- 감정 분석 (Sentiment Analysis)
- 텍스트 분류 (Text Classification)
- 키워드 추출 및 TF-IDF 분석
- 언어 감지
- 텍스트 전처리 및 토큰화
- 한국어 텍스트 특화 처리
- 한국어 해석 및 인사이트
"""

import sys
import json
import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    import joblib
except ImportError as e:
    print(json.dumps({
        "success": False,
        "error": f"기본 라이브러리 누락: {str(e)}",
        "required_packages": ["scikit-learn", "joblib"]
    }, ensure_ascii=False))
    sys.exit(1)

# Optional dependencies with graceful fallback
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True

    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)

except ImportError:
    NLTK_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from langdetect import detect, LangDetectError
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

def clean_dict_for_json(obj):
    """JSON 직렬화를 위한 딕셔너리 정리"""
    if isinstance(obj, dict):
        return {str(k): clean_dict_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_dict_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return [clean_dict_for_json(item) for item in obj]
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

def preprocess_text(text: str, language: str = "english",
                   remove_stopwords: bool = True, stemming: bool = False,
                   lemmatization: bool = False) -> str:
    """
    텍스트 전처리

    Parameters:
    -----------
    text : str
        입력 텍스트
    language : str
        언어 설정
    remove_stopwords : bool
        불용어 제거 여부
    stemming : bool
        어간 추출 여부
    lemmatization : bool
        표제어 추출 여부

    Returns:
    --------
    str
        전처리된 텍스트
    """
    if not isinstance(text, str):
        return ""

    # 기본 전처리
    text = text.lower()
    text = re.sub(r'[^a-zA-Z가-힣\s]', '', text)  # 특수문자 제거 (한글 포함)
    text = re.sub(r'\s+', ' ', text).strip()  # 공백 정리

    if not NLTK_AVAILABLE:
        return text

    # 토큰화
    tokens = word_tokenize(text)

    # 불용어 제거
    if remove_stopwords:
        try:
            stop_words = set(stopwords.words(language))
            tokens = [token for token in tokens if token not in stop_words]
        except:
            pass  # 언어가 지원되지 않는 경우 건너뜀

    # 어간 추출
    if stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]

    # 표제어 추출
    if lemmatization:
        try:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
        except:
            pass

    return ' '.join(tokens)

def detect_language(text: str) -> Dict[str, Any]:
    """
    언어 감지

    Parameters:
    -----------
    text : str
        입력 텍스트

    Returns:
    --------
    Dict[str, Any]
        언어 감지 결과
    """
    if not LANGDETECT_AVAILABLE:
        return {"language": "unknown", "confidence": 0.0, "method": "unavailable"}

    try:
        detected_lang = detect(text)

        # 한국어 특별 처리
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.findall(r'[a-zA-Z가-힣]', text))

        if total_chars > 0 and korean_chars / total_chars > 0.3:
            detected_lang = "ko"
            confidence = korean_chars / total_chars
        else:
            confidence = 0.8  # langdetect 기본 신뢰도

        return {
            "language": detected_lang,
            "confidence": float(confidence),
            "korean_ratio": float(korean_chars / total_chars) if total_chars > 0 else 0.0,
            "method": "langdetect"
        }

    except LangDetectError:
        return {"language": "unknown", "confidence": 0.0, "method": "failed"}

def extract_keywords(texts: List[str], max_features: int = 20,
                    method: str = "tfidf", ngram_range: Tuple[int, int] = (1, 2)) -> Dict[str, Any]:
    """
    키워드 추출

    Parameters:
    -----------
    texts : List[str]
        텍스트 목록
    max_features : int
        최대 키워드 수
    method : str
        추출 방법 ('tfidf' 또는 'count')
    ngram_range : Tuple[int, int]
        n-gram 범위

    Returns:
    --------
    Dict[str, Any]
        키워드 추출 결과
    """
    try:
        # 전처리
        processed_texts = [preprocess_text(text) for text in texts]

        # 벡터라이저 선택
        if method == "tfidf":
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english'
            )
        else:
            vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english'
            )

        # 벡터화
        vectors = vectorizer.fit_transform(processed_texts)
        feature_names = vectorizer.get_feature_names_out()

        # 키워드 점수 계산
        if method == "tfidf":
            scores = vectors.mean(axis=0).A1
        else:
            scores = vectors.sum(axis=0).A1

        # 키워드 정렬
        keyword_scores = list(zip(feature_names, scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)

        return {
            "keywords": [{"word": word, "score": float(score)} for word, score in keyword_scores],
            "method": method,
            "total_documents": len(texts),
            "vocabulary_size": len(feature_names),
            "ngram_range": ngram_range
        }

    except Exception as e:
        return {"error": str(e), "method": method}

def analyze_sentiment(texts: List[str], method: str = "auto") -> Dict[str, Any]:
    """
    감정 분석

    Parameters:
    -----------
    texts : List[str]
        분석할 텍스트 목록
    method : str
        분석 방법 ('auto', 'vader', 'textblob')

    Returns:
    --------
    Dict[str, Any]
        감정 분석 결과
    """
    results = []
    sentiment_stats = {"positive": 0, "negative": 0, "neutral": 0}

    # 방법 자동 선택
    if method == "auto":
        if NLTK_AVAILABLE:
            method = "vader"
        elif TEXTBLOB_AVAILABLE:
            method = "textblob"
        else:
            return {"error": "감정 분석 라이브러리가 설치되지 않았습니다"}

    # VADER 감정 분석
    if method == "vader" and NLTK_AVAILABLE:
        try:
            analyzer = SentimentIntensityAnalyzer()

            for text in texts:
                scores = analyzer.polarity_scores(text)

                # 감정 레이블 결정
                if scores['compound'] >= 0.05:
                    sentiment = "positive"
                elif scores['compound'] <= -0.05:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"

                sentiment_stats[sentiment] += 1

                results.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "sentiment": sentiment,
                    "compound": float(scores['compound']),
                    "positive": float(scores['pos']),
                    "negative": float(scores['neg']),
                    "neutral": float(scores['neu'])
                })

        except Exception as e:
            return {"error": f"VADER 분석 실패: {str(e)}"}

    # TextBlob 감정 분석
    elif method == "textblob" and TEXTBLOB_AVAILABLE:
        try:
            for text in texts:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity

                # 감정 레이블 결정
                if polarity > 0.1:
                    sentiment = "positive"
                elif polarity < -0.1:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"

                sentiment_stats[sentiment] += 1

                results.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "sentiment": sentiment,
                    "polarity": float(polarity),
                    "subjectivity": float(subjectivity)
                })

        except Exception as e:
            return {"error": f"TextBlob 분석 실패: {str(e)}"}

    else:
        return {"error": f"방법 '{method}'를 사용할 수 없습니다"}

    # 통계 계산
    total_texts = len(texts)
    sentiment_distribution = {
        "positive": sentiment_stats["positive"] / total_texts,
        "negative": sentiment_stats["negative"] / total_texts,
        "neutral": sentiment_stats["neutral"] / total_texts
    }

    return {
        "method": method,
        "total_texts": total_texts,
        "results": results,
        "sentiment_counts": sentiment_stats,
        "sentiment_distribution": sentiment_distribution,
        "overall_sentiment": max(sentiment_stats, key=sentiment_stats.get)
    }

def train_text_classifier(df: pd.DataFrame, text_column: str, label_column: str,
                         algorithms: List[str] = ["logistic", "naive_bayes", "svm", "random_forest"],
                         test_size: float = 0.2, max_features: int = 5000,
                         ngram_range: Tuple[int, int] = (1, 2), cv_folds: int = 5,
                         model_save_path: str = "text_classifier.pkl") -> Dict[str, Any]:
    """
    텍스트 분류 모델 훈련

    Parameters:
    -----------
    df : pd.DataFrame
        입력 데이터프레임
    text_column : str
        텍스트 컬럼명
    label_column : str
        라벨 컬럼명
    algorithms : List[str]
        사용할 알고리즘 목록
    test_size : float
        테스트 데이터 비율
    max_features : int
        최대 특성 수
    ngram_range : Tuple[int, int]
        n-gram 범위
    cv_folds : int
        교차 검증 폴드 수
    model_save_path : str
        모델 저장 경로

    Returns:
    --------
    Dict[str, Any]
        분류 모델 훈련 결과
    """

    try:
        # 데이터 검증
        if text_column not in df.columns or label_column not in df.columns:
            return {
                "success": False,
                "error": f"필요한 컬럼이 없습니다: {text_column}, {label_column}",
                "available_columns": list(df.columns)
            }

        # 결측값 제거
        df = df.dropna(subset=[text_column, label_column])

        if df.empty:
            return {
                "success": False,
                "error": "유효한 데이터가 없습니다"
            }

        # 텍스트 전처리
        texts = df[text_column].astype(str).tolist()
        processed_texts = [preprocess_text(text) for text in texts]

        # 라벨 인코딩
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(df[label_column])

        # 텍스트 벡터화
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )

        X = vectorizer.fit_transform(processed_texts)
        y = labels

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # 모델 훈련 및 평가
        models = {}
        results = {}

        algorithm_map = {
            "logistic": LogisticRegression(random_state=42),
            "naive_bayes": MultinomialNB(),
            "svm": SVC(random_state=42, probability=True),
            "random_forest": RandomForestClassifier(random_state=42, n_estimators=100)
        }

        for algorithm in algorithms:
            if algorithm not in algorithm_map:
                continue

            try:
                model = algorithm_map[algorithm]

                # 모델 훈련
                model.fit(X_train, y_train)
                models[algorithm] = model

                # 예측
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

                # 성능 평가
                accuracy = accuracy_score(y_test, y_pred)

                # 교차 검증
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')

                # 분류 리포트
                class_report = classification_report(y_test, y_pred, output_dict=True)

                results[algorithm] = {
                    "accuracy": float(accuracy),
                    "cv_scores": cv_scores.tolist(),
                    "cv_mean": float(cv_scores.mean()),
                    "cv_std": float(cv_scores.std()),
                    "classification_report": class_report,
                    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
                }

                # 특성 중요도 (가능한 경우)
                if hasattr(model, "feature_importances_"):
                    feature_names = vectorizer.get_feature_names_out()
                    importances = model.feature_importances_
                    top_features = sorted(zip(feature_names, importances),
                                        key=lambda x: x[1], reverse=True)[:20]
                    results[algorithm]["top_features"] = [
                        {"feature": feat, "importance": float(imp)} for feat, imp in top_features
                    ]

                elif hasattr(model, "coef_"):
                    feature_names = vectorizer.get_feature_names_out()
                    coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                    top_features = sorted(zip(feature_names, np.abs(coef)),
                                        key=lambda x: x[1], reverse=True)[:20]
                    results[algorithm]["top_features"] = [
                        {"feature": feat, "coefficient": float(coef)} for feat, coef in top_features
                    ]

            except Exception as e:
                results[algorithm] = {"error": str(e)}

        # 최고 성능 모델 선택
        successful_models = {k: v for k, v in results.items() if "error" not in v}

        if not successful_models:
            return {
                "success": False,
                "error": "모든 모델 훈련이 실패했습니다",
                "results": results
            }

        best_algorithm = max(successful_models.keys(), key=lambda k: results[k]["accuracy"])
        best_model = models[best_algorithm]

        # 모델 저장
        model_data = {
            "model": best_model,
            "vectorizer": vectorizer,
            "label_encoder": label_encoder,
            "algorithm": best_algorithm,
            "classes": label_encoder.classes_.tolist()
        }
        joblib.dump(model_data, model_save_path)

        # 텍스트 분석
        text_analysis = analyze_text_dataset(texts, df[label_column].tolist())

        # 결과 정리
        result = {
            "success": True,
            "model_save_path": model_save_path,
            "best_algorithm": best_algorithm,
            "total_samples": len(df),
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "num_classes": len(label_encoder.classes_),
            "classes": label_encoder.classes_.tolist(),
            "vocabulary_size": X.shape[1],
            "results": clean_dict_for_json(results),
            "text_analysis": clean_dict_for_json(text_analysis),
            "insights": generate_text_classification_insights(results, text_analysis)
        }

        return clean_dict_for_json(result)

    except Exception as e:
        return {
            "success": False,
            "error": f"텍스트 분류 모델 훈련 실패: {str(e)}",
            "error_type": type(e).__name__
        }

def analyze_text_dataset(texts: List[str], labels: List[str]) -> Dict[str, Any]:
    """텍스트 데이터셋 분석"""

    analysis = {
        "text_statistics": {},
        "label_distribution": {},
        "language_analysis": {},
        "keywords_by_class": {}
    }

    # 텍스트 통계
    text_lengths = [len(text) for text in texts]
    word_counts = [len(text.split()) for text in texts]

    analysis["text_statistics"] = {
        "total_texts": len(texts),
        "avg_text_length": float(np.mean(text_lengths)),
        "avg_word_count": float(np.mean(word_counts)),
        "max_text_length": max(text_lengths),
        "min_text_length": min(text_lengths),
        "std_text_length": float(np.std(text_lengths))
    }

    # 라벨 분포
    label_counts = pd.Series(labels).value_counts()
    analysis["label_distribution"] = {
        "counts": label_counts.to_dict(),
        "percentages": (label_counts / len(labels) * 100).to_dict()
    }

    # 언어 분석 (샘플링)
    if LANGDETECT_AVAILABLE and len(texts) > 0:
        sample_size = min(100, len(texts))
        sample_texts = np.random.choice(texts, sample_size, replace=False)

        languages = []
        for text in sample_texts:
            lang_result = detect_language(text)
            languages.append(lang_result["language"])

        lang_counts = pd.Series(languages).value_counts()
        analysis["language_analysis"] = {
            "detected_languages": lang_counts.to_dict(),
            "primary_language": lang_counts.index[0] if len(lang_counts) > 0 else "unknown",
            "sample_size": sample_size
        }

    # 클래스별 키워드 추출
    unique_labels = list(set(labels))
    for label in unique_labels[:5]:  # 상위 5개 클래스만
        label_texts = [text for text, lbl in zip(texts, labels) if lbl == label]
        if len(label_texts) > 0:
            keywords_result = extract_keywords(label_texts, max_features=10)
            if "error" not in keywords_result:
                analysis["keywords_by_class"][str(label)] = keywords_result["keywords"][:10]

    return analysis

def generate_text_classification_insights(results: Dict[str, Any], text_analysis: Dict[str, Any]) -> List[str]:
    """텍스트 분류 인사이트 생성"""

    insights = []

    # 모델 성능 비교
    successful_models = {k: v for k, v in results.items() if "error" not in v}

    if len(successful_models) > 1:
        best_model = max(successful_models.keys(), key=lambda k: successful_models[k]["accuracy"])
        best_accuracy = successful_models[best_model]["accuracy"]

        worst_model = min(successful_models.keys(), key=lambda k: successful_models[k]["accuracy"])
        worst_accuracy = successful_models[worst_model]["accuracy"]

        if best_accuracy - worst_accuracy > 0.1:
            insights.append(f"'{best_model}' 알고리즘이 '{worst_model}'보다 {(best_accuracy - worst_accuracy)*100:.1f}% 더 좋은 성능을 보입니다")

    # 전체적인 성능 평가
    if successful_models:
        best_model = max(successful_models.keys(), key=lambda k: successful_models[k]["accuracy"])
        best_accuracy = successful_models[best_model]["accuracy"]

        if best_accuracy > 0.9:
            insights.append("매우 높은 분류 정확도를 달성했습니다")
        elif best_accuracy > 0.8:
            insights.append("좋은 분류 성능을 보입니다")
        elif best_accuracy < 0.7:
            insights.append("분류 성능이 낮습니다. 데이터 전처리나 특성 엔지니어링을 고려하세요")

    # 데이터 불균형 분석
    label_dist = text_analysis.get("label_distribution", {}).get("percentages", {})
    if label_dist:
        max_percentage = max(label_dist.values())
        min_percentage = min(label_dist.values())

        if max_percentage > 70:
            insights.append("클래스 불균형이 심합니다. 데이터 밸런싱을 고려하세요")
        elif max_percentage / min_percentage > 5:
            insights.append("클래스 간 데이터 분포에 차이가 있습니다")

    # 텍스트 특성 분석
    text_stats = text_analysis.get("text_statistics", {})
    if text_stats:
        avg_length = text_stats.get("avg_text_length", 0)
        if avg_length < 50:
            insights.append("텍스트가 매우 짧습니다. 더 많은 컨텍스트 정보가 도움될 수 있습니다")
        elif avg_length > 1000:
            insights.append("텍스트가 깁니다. 텍스트 요약이나 키워드 추출을 고려하세요")

    # 언어 분석
    lang_analysis = text_analysis.get("language_analysis", {})
    if lang_analysis:
        primary_lang = lang_analysis.get("primary_language", "unknown")
        if primary_lang == "ko":
            insights.append("한국어 텍스트가 감지되었습니다. 한국어 특화 전처리를 고려하세요")
        elif primary_lang not in ["en", "ko"]:
            insights.append(f"'{primary_lang}' 언어가 감지되었습니다")

    # 특성 중요도 분석
    for algorithm, result in successful_models.items():
        if "top_features" in result:
            top_features = result["top_features"][:5]
            feature_names = [f["feature"] for f in top_features]
            insights.append(f"{algorithm} 모델의 주요 특성: {', '.join(feature_names)}")
            break

    return insights

def main():
    """메인 실행 함수"""
    try:
        # stdin에서 JSON 데이터 읽기
        input_data = sys.stdin.read()
        params = json.loads(input_data)

        # 함수 실행
        if "function" in params:
            function_name = params.pop("function")

            if function_name == "analyze_sentiment":
                result = analyze_sentiment(**params)
            elif function_name == "train_text_classifier":
                result = train_text_classifier(**params)
            elif function_name == "extract_keywords":
                result = extract_keywords(**params)
            elif function_name == "detect_language":
                result = detect_language(**params)
            else:
                result = {
                    "success": False,
                    "error": f"알 수 없는 함수: {function_name}",
                    "available_functions": ["analyze_sentiment", "train_text_classifier", "extract_keywords", "detect_language"]
                }
        else:
            # 기본적으로 텍스트 분류 훈련
            result = train_text_classifier(**params)

        # JSON으로 결과 출력
        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
        print(json.dumps(error_result, ensure_ascii=False, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()