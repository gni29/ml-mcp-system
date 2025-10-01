#!/usr/bin/env python3
"""
Sentiment Analyzer for ML MCP System
Sentiment analysis using multiple methods (VADER, TextBlob, Transformers)
"""

import pandas as pd
import numpy as np
import json
import sys
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Check VADER availability
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# Check TextBlob availability
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

# Check Transformers availability
try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class SentimentAnalyzer:
    """Sentiment analysis with multiple methods"""

    def __init__(self, method: str = 'vader'):
        """
        Initialize sentiment analyzer

        Args:
            method: 'vader', 'textblob', or 'transformers'
        """
        self.method = method

        if method == 'vader':
            if not VADER_AVAILABLE:
                raise ImportError("VADER required. Install with: pip install vaderSentiment")
            self.analyzer = SentimentIntensityAnalyzer()

        elif method == 'textblob':
            if not TEXTBLOB_AVAILABLE:
                raise ImportError("TextBlob required. Install with: pip install textblob")
            self.analyzer = None  # TextBlob is called directly

        elif method == 'transformers':
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers required. Install with: pip install transformers torch")
            # Use sentiment analysis pipeline
            self.analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

        else:
            raise ValueError(f"Unknown method: {method}. Use 'vader', 'textblob', or 'transformers'")

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text

        Args:
            text: Input text

        Returns:
            Sentiment analysis results
        """
        if not isinstance(text, str) or not text.strip():
            return {
                'text': text,
                'sentiment': 'neutral',
                'score': 0.0,
                'method': self.method
            }

        if self.method == 'vader':
            return self._analyze_vader(text)

        elif self.method == 'textblob':
            return self._analyze_textblob(text)

        elif self.method == 'transformers':
            return self._analyze_transformers(text)

    def _analyze_vader(self, text: str) -> Dict[str, Any]:
        """Analyze using VADER"""
        scores = self.analyzer.polarity_scores(text)

        # Determine sentiment
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'method': 'vader',
            'sentiment': sentiment,
            'score': float(compound),
            'scores': {
                'positive': float(scores['pos']),
                'negative': float(scores['neg']),
                'neutral': float(scores['neu']),
                'compound': float(scores['compound'])
            }
        }

    def _analyze_textblob(self, text: str) -> Dict[str, Any]:
        """Analyze using TextBlob"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Determine sentiment
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'method': 'textblob',
            'sentiment': sentiment,
            'score': float(polarity),
            'polarity': float(polarity),
            'subjectivity': float(subjectivity)
        }

    def _analyze_transformers(self, text: str) -> Dict[str, Any]:
        """Analyze using Transformers"""
        # Truncate if too long
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]

        result = self.analyzer(text)[0]

        # Convert to standard format
        label = result['label'].lower()
        score = result['score']

        # Map to sentiment
        if label == 'positive':
            sentiment = 'positive'
            sentiment_score = score
        else:
            sentiment = 'negative'
            sentiment_score = -score

        return {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'method': 'transformers',
            'sentiment': sentiment,
            'score': float(sentiment_score),
            'confidence': float(score),
            'label': result['label']
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts

        Args:
            texts: List of texts

        Returns:
            List of sentiment results
        """
        results = []
        for text in texts:
            result = self.analyze(text)
            results.append(result)

        return results

    def get_sentiment_distribution(self, texts: List[str]) -> Dict[str, Any]:
        """
        Get sentiment distribution for multiple texts

        Args:
            texts: List of texts

        Returns:
            Sentiment distribution
        """
        results = self.analyze_batch(texts)

        # Count sentiments
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        scores = []

        for result in results:
            sentiment_counts[result['sentiment']] += 1
            scores.append(result['score'])

        total = len(texts)

        distribution = {
            'method': self.method,
            'total_texts': total,
            'sentiment_counts': sentiment_counts,
            'sentiment_percentages': {
                sentiment: (count / total * 100) if total > 0 else 0
                for sentiment, count in sentiment_counts.items()
            },
            'score_statistics': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores))
            },
            'overall_sentiment': max(sentiment_counts.items(), key=lambda x: x[1])[0]
        }

        return distribution

    def analyze_aspects(self, text: str, aspects: List[str]) -> Dict[str, Any]:
        """
        Aspect-based sentiment analysis

        Args:
            text: Input text
            aspects: List of aspects to analyze

        Returns:
            Aspect sentiments
        """
        text_lower = text.lower()
        aspect_sentiments = {}

        for aspect in aspects:
            # Find sentences containing the aspect
            sentences = text.split('.')
            relevant_sentences = [s for s in sentences if aspect.lower() in s.lower()]

            if relevant_sentences:
                # Analyze sentiment of relevant sentences
                sentiments = [self.analyze(s) for s in relevant_sentences]
                avg_score = np.mean([s['score'] for s in sentiments])

                if avg_score > 0.1:
                    sentiment = 'positive'
                elif avg_score < -0.1:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'

                aspect_sentiments[aspect] = {
                    'sentiment': sentiment,
                    'score': float(avg_score),
                    'mentions': len(relevant_sentences),
                    'example_sentence': relevant_sentences[0] if relevant_sentences else None
                }
            else:
                aspect_sentiments[aspect] = {
                    'sentiment': 'not_mentioned',
                    'score': 0.0,
                    'mentions': 0,
                    'example_sentence': None
                }

        return {
            'method': f'{self.method}_aspect_based',
            'text': text[:100] + '...' if len(text) > 100 else text,
            'aspects': aspect_sentiments,
            'summary': {
                'aspects_mentioned': sum(1 for a in aspect_sentiments.values() if a['mentions'] > 0),
                'positive_aspects': sum(1 for a in aspect_sentiments.values() if a['sentiment'] == 'positive'),
                'negative_aspects': sum(1 for a in aspect_sentiments.values() if a['sentiment'] == 'negative')
            }
        }

    def compare_methods(self, text: str) -> Dict[str, Any]:
        """
        Compare sentiment across different methods

        Args:
            text: Input text

        Returns:
            Comparison results
        """
        results = {}

        # Try each available method
        if VADER_AVAILABLE:
            analyzer_vader = SentimentAnalyzer('vader')
            results['vader'] = analyzer_vader.analyze(text)

        if TEXTBLOB_AVAILABLE:
            analyzer_textblob = SentimentAnalyzer('textblob')
            results['textblob'] = analyzer_textblob.analyze(text)

        if TRANSFORMERS_AVAILABLE:
            try:
                analyzer_transformers = SentimentAnalyzer('transformers')
                results['transformers'] = analyzer_transformers.analyze(text)
            except:
                pass  # Skip if model download fails

        # Calculate agreement
        sentiments = [r['sentiment'] for r in results.values()]
        agreement = len(set(sentiments)) == 1 if sentiments else False

        return {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'methods_compared': list(results.keys()),
            'results': results,
            'agreement': agreement,
            'consensus_sentiment': sentiments[0] if agreement else 'mixed'
        }


def main():
    """CLI interface"""
    if len(sys.argv) < 2:
        print("Usage: python sentiment_analyzer.py <action>")
        print("Actions: demo, check_availability")
        sys.exit(1)

    action = sys.argv[1]

    try:
        if action == 'check_availability':
            result = {
                'vader_available': VADER_AVAILABLE,
                'textblob_available': TEXTBLOB_AVAILABLE,
                'transformers_available': TRANSFORMERS_AVAILABLE,
                'install_commands': {
                    'vader': 'pip install vaderSentiment' if not VADER_AVAILABLE else None,
                    'textblob': 'pip install textblob' if not TEXTBLOB_AVAILABLE else None,
                    'transformers': 'pip install transformers torch' if not TRANSFORMERS_AVAILABLE else None
                }
            }

        elif action == 'demo':
            print("Sentiment Analyzer Demo")
            print("=" * 50)

            # Sample texts
            texts = [
                "This product is amazing! I absolutely love it!",
                "Terrible experience. Would not recommend.",
                "It's okay, nothing special.",
                "The quality is great but the price is too high.",
                "Best purchase ever! Highly recommend to everyone!"
            ]

            # Analyze with VADER (most commonly available)
            if VADER_AVAILABLE:
                analyzer = SentimentAnalyzer('vader')

                print("\nIndividual Sentiments:")
                for i, text in enumerate(texts, 1):
                    result_data = analyzer.analyze(text)
                    print(f"\n{i}. \"{text}\"")
                    print(f"   Sentiment: {result_data['sentiment']} (score: {result_data['score']:.3f})")

                # Distribution
                distribution = analyzer.get_sentiment_distribution(texts)
                print(f"\nOverall Distribution:")
                for sentiment, pct in distribution['sentiment_percentages'].items():
                    print(f"  {sentiment}: {pct:.1f}%")
                print(f"  Overall sentiment: {distribution['overall_sentiment']}")

                # Aspect-based analysis
                review_text = "The food quality is excellent but the service is poor. Prices are reasonable."
                aspects = ['food', 'service', 'price']
                aspect_result = analyzer.analyze_aspects(review_text, aspects)
                print(f"\nAspect-Based Analysis:")
                print(f"Text: \"{review_text}\"")
                for aspect, data in aspect_result['aspects'].items():
                    if data['sentiment'] != 'not_mentioned':
                        print(f"  {aspect}: {data['sentiment']} (score: {data['score']:.3f})")

                result = {
                    'success': True,
                    'method': 'vader',
                    'individual_results': [analyzer.analyze(t) for t in texts],
                    'distribution': distribution,
                    'aspect_analysis': aspect_result
                }
            else:
                result = {
                    'success': False,
                    'error': 'No sentiment analysis library available',
                    'install_command': 'pip install vaderSentiment'
                }

        else:
            result = {'error': f'Unknown action: {action}'}

        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
        print(json.dumps(error_result, ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()