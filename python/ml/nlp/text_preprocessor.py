#!/usr/bin/env python3
"""
Text Preprocessor for ML MCP System
Text cleaning, tokenization, normalization for NLP tasks
"""

import pandas as pd
import numpy as np
import json
import sys
import re
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Check NLTK availability
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_AVAILABLE = True

    # Try to load required data
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        NLTK_DATA_AVAILABLE = True
    except LookupError:
        NLTK_DATA_AVAILABLE = False
except ImportError:
    NLTK_AVAILABLE = False
    NLTK_DATA_AVAILABLE = False


class TextPreprocessor:
    """Text preprocessing for NLP tasks"""

    def __init__(self, language: str = 'english', lowercase: bool = True,
                 remove_punctuation: bool = True, remove_stopwords: bool = True,
                 remove_numbers: bool = False):
        """
        Initialize text preprocessor

        Args:
            language: Language for stopwords
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation
            remove_stopwords: Remove stopwords
            remove_numbers: Remove numbers
        """
        self.language = language
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.remove_numbers = remove_numbers

        if NLTK_AVAILABLE and NLTK_DATA_AVAILABLE:
            self.stopwords_set = set(stopwords.words(language))
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.stopwords_set = set()
            self.stemmer = None
            self.lemmatizer = None

    def clean(self, text: str) -> str:
        """
        Clean text

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove mentions and hashtags (for social media text)
        text = re.sub(r'@\w+|#\w+', '', text)

        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)

        # Remove punctuation
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize(self, text: str, method: str = 'word') -> List[str]:
        """
        Tokenize text

        Args:
            text: Input text
            method: 'word' or 'sentence'

        Returns:
            List of tokens
        """
        if not NLTK_AVAILABLE or not NLTK_DATA_AVAILABLE:
            # Fallback tokenization
            if method == 'word':
                return text.split()
            else:
                return [s.strip() for s in text.split('.') if s.strip()]

        if method == 'word':
            return word_tokenize(text)
        elif method == 'sentence':
            return sent_tokenize(text)
        else:
            raise ValueError(f"Unknown tokenization method: {method}")

    def remove_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list

        Args:
            tokens: List of tokens

        Returns:
            Filtered tokens
        """
        if not self.stopwords_set:
            return tokens

        return [token for token in tokens if token.lower() not in self.stopwords_set]

    def stem(self, tokens: List[str]) -> List[str]:
        """
        Apply stemming to tokens

        Args:
            tokens: List of tokens

        Returns:
            Stemmed tokens
        """
        if self.stemmer is None:
            return tokens

        return [self.stemmer.stem(token) for token in tokens]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Apply lemmatization to tokens

        Args:
            tokens: List of tokens

        Returns:
            Lemmatized tokens
        """
        if self.lemmatizer is None:
            return tokens

        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def get_ngrams(self, tokens: List[str], n: int = 2) -> List[tuple]:
        """
        Extract n-grams

        Args:
            tokens: List of tokens
            n: N-gram size

        Returns:
            List of n-grams
        """
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return ngrams

    def preprocess(self, text: str, steps: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline

        Args:
            text: Input text
            steps: List of preprocessing steps
                   ['clean', 'tokenize', 'remove_stopwords', 'stem', 'lemmatize']
                   If None, applies all steps

        Returns:
            Preprocessing results
        """
        if steps is None:
            steps = ['clean', 'tokenize', 'remove_stopwords', 'lemmatize']

        result = {
            'original_text': text,
            'original_length': len(text),
            'steps_applied': steps
        }

        current_text = text

        # Clean
        if 'clean' in steps:
            current_text = self.clean(current_text)
            result['cleaned_text'] = current_text
            result['cleaned_length'] = len(current_text)

        # Tokenize
        if 'tokenize' in steps:
            tokens = self.tokenize(current_text)
            result['tokens'] = tokens
            result['token_count'] = len(tokens)
        else:
            tokens = [current_text]

        # Remove stopwords
        if 'remove_stopwords' in steps and 'tokenize' in steps:
            tokens = self.remove_stopwords_from_tokens(tokens)
            result['tokens_no_stopwords'] = tokens
            result['stopwords_removed'] = result.get('token_count', 0) - len(tokens)

        # Stem
        if 'stem' in steps and 'tokenize' in steps:
            tokens = self.stem(tokens)
            result['stemmed_tokens'] = tokens

        # Lemmatize
        if 'lemmatize' in steps and 'tokenize' in steps:
            tokens = self.lemmatize(tokens)
            result['lemmatized_tokens'] = tokens

        # Final tokens
        if 'tokenize' in steps:
            result['final_tokens'] = tokens
            result['final_token_count'] = len(tokens)

        return result

    def batch_preprocess(self, texts: List[str],
                        steps: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Preprocess multiple texts

        Args:
            texts: List of texts
            steps: Preprocessing steps

        Returns:
            Batch preprocessing results
        """
        results = []
        total_original_length = 0
        total_final_tokens = 0

        for text in texts:
            result = self.preprocess(text, steps)
            results.append(result)
            total_original_length += result['original_length']
            total_final_tokens += result.get('final_token_count', 0)

        return {
            'method': 'batch_preprocess',
            'texts_processed': len(texts),
            'results': results,
            'summary': {
                'total_original_length': total_original_length,
                'total_final_tokens': total_final_tokens,
                'average_tokens_per_text': total_final_tokens / len(texts) if texts else 0
            }
        }

    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract text features

        Args:
            text: Input text

        Returns:
            Text features
        """
        tokens = self.tokenize(self.clean(text))

        features = {
            'character_count': len(text),
            'word_count': len(tokens),
            'sentence_count': len(self.tokenize(text, method='sentence')),
            'avg_word_length': np.mean([len(word) for word in tokens]) if tokens else 0,
            'unique_words': len(set(tokens)),
            'lexical_diversity': len(set(tokens)) / len(tokens) if tokens else 0,
            'uppercase_count': sum(1 for c in text if c.isupper()),
            'digit_count': sum(1 for c in text if c.isdigit()),
            'punctuation_count': sum(1 for c in text if c in '.,!?;:'),
            'whitespace_count': sum(1 for c in text if c.isspace())
        }

        # Most common words (excluding stopwords)
        tokens_no_stop = self.remove_stopwords_from_tokens(tokens)
        if tokens_no_stop:
            word_freq = pd.Series(tokens_no_stop).value_counts()
            features['most_common_words'] = word_freq.head(10).to_dict()

        return features


def main():
    """CLI interface"""
    if len(sys.argv) < 2:
        print("Usage: python text_preprocessor.py <action>")
        print("Actions: demo, check_availability")
        sys.exit(1)

    action = sys.argv[1]

    try:
        if action == 'check_availability':
            result = {
                'nltk_available': NLTK_AVAILABLE,
                'nltk_data_available': NLTK_DATA_AVAILABLE,
                'install_commands': {
                    'nltk': 'pip install nltk' if not NLTK_AVAILABLE else None,
                    'nltk_data': 'python -m nltk.downloader punkt stopwords wordnet' if NLTK_AVAILABLE and not NLTK_DATA_AVAILABLE else None
                }
            }

        elif action == 'demo':
            print("Text Preprocessor Demo")
            print("=" * 50)

            # Sample text
            sample_text = """
            This is an EXAMPLE text for NLP preprocessing!
            It contains URLs like https://example.com and emails like test@example.com.
            We'll clean, tokenize, and normalize this text. #NLP #MachineLearning
            """

            # Initialize preprocessor
            preprocessor = TextPreprocessor()

            # Preprocess
            result_data = preprocessor.preprocess(sample_text)

            print(f"\nOriginal text ({result_data['original_length']} chars):")
            print(f"  {sample_text[:100]}...")

            print(f"\nCleaned text ({result_data['cleaned_length']} chars):")
            print(f"  {result_data['cleaned_text'][:100]}...")

            print(f"\nTokens ({result_data['token_count']}):")
            print(f"  {result_data['tokens'][:10]}")

            if 'final_tokens' in result_data:
                print(f"\nFinal processed tokens ({result_data['final_token_count']}):")
                print(f"  {result_data['final_tokens'][:10]}")

            # Extract features
            features = preprocessor.extract_features(sample_text)
            print(f"\nText features:")
            print(f"  Word count: {features['word_count']}")
            print(f"  Unique words: {features['unique_words']}")
            print(f"  Lexical diversity: {features['lexical_diversity']:.2f}")

            result = {
                'success': True,
                'preprocessing_result': result_data,
                'features': features
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