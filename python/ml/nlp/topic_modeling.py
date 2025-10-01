"""
Topic Modeling for Text Analysis
LDA, NMF, and BERTopic implementations for discovering latent topics

Features:
- Multiple algorithms (LDA, NMF, BERTopic)
- Topic coherence scoring
- Topic visualization
- Document-topic distribution
- Top words per topic extraction
- Hyperparameter optimization

Usage:
    from topic_modeling import TopicModeler

    modeler = TopicModeler(n_topics=5, method='lda')
    topics = modeler.fit_transform(documents)
    modeler.print_topics()
    modeler.visualize_topics()

CLI:
    python -m python.ml.nlp.topic_modeling --input documents.csv --column text --n-topics 5 --method lda
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Required dependencies
try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Install with: pip install scikit-learn")

# Optional advanced dependencies
try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

try:
    import gensim
    from gensim.models import CoherenceModel
    from gensim.corpora import Dictionary
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import pyLDAvis
    import pyLDAvis.lda_model
    PYLDAVIS_AVAILABLE = True
except ImportError:
    PYLDAVIS_AVAILABLE = False


class TopicModeler:
    """
    Topic modeling with multiple algorithms

    Supports LDA, NMF, and optionally BERTopic for discovering topics in document collections
    """

    def __init__(
        self,
        n_topics: int = 10,
        method: str = 'lda',
        max_features: int = 5000,
        max_df: float = 0.95,
        min_df: int = 2,
        ngram_range: tuple = (1, 2),
        random_state: int = 42
    ):
        """
        Initialize topic modeler

        Args:
            n_topics: Number of topics to extract
            method: Topic modeling method ('lda', 'nmf', 'bertopic')
            max_features: Maximum number of features for vectorizer
            max_df: Maximum document frequency for terms
            min_df: Minimum document frequency for terms
            ngram_range: N-gram range for feature extraction
            random_state: Random seed for reproducibility
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

        self.n_topics = n_topics
        self.method = method.lower()
        self.max_features = max_features
        self.max_df = max_df
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.random_state = random_state

        # Models
        self.vectorizer = None
        self.model = None
        self.feature_names = None
        self.documents = None
        self.doc_topic_matrix = None

        # Results
        self.topics = {}
        self.coherence_score = None

    def _initialize_vectorizer(self):
        """Initialize text vectorizer based on method"""
        if self.method == 'lda':
            # LDA works best with count vectorizer
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                max_df=self.max_df,
                min_df=self.min_df,
                ngram_range=self.ngram_range,
                stop_words='english'
            )
        elif self.method == 'nmf':
            # NMF works best with TF-IDF
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                max_df=self.max_df,
                min_df=self.min_df,
                ngram_range=self.ngram_range,
                stop_words='english'
            )
        # BERTopic has its own vectorization

    def _initialize_model(self):
        """Initialize topic model based on method"""
        if self.method == 'lda':
            self.model = LatentDirichletAllocation(
                n_components=self.n_topics,
                max_iter=20,
                learning_method='online',
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.method == 'nmf':
            self.model = NMF(
                n_components=self.n_topics,
                init='nndsvda',
                max_iter=500,
                random_state=self.random_state
            )
        elif self.method == 'bertopic':
            if not BERTOPIC_AVAILABLE:
                raise ImportError("BERTopic not installed. Install with: pip install bertopic")
            self.model = BERTopic(
                nr_topics=self.n_topics,
                verbose=False,
                calculate_probabilities=True
            )
        else:
            raise ValueError(f"Unknown method: {self.method}. Choose from 'lda', 'nmf', 'bertopic'")

    def fit(self, documents: List[str]) -> 'TopicModeler':
        """
        Fit topic model to documents

        Args:
            documents: List of text documents

        Returns:
            Self for method chaining
        """
        self.documents = documents

        if self.method in ['lda', 'nmf']:
            # Traditional approach
            self._initialize_vectorizer()
            self._initialize_model()

            # Vectorize documents
            doc_term_matrix = self.vectorizer.fit_transform(documents)
            self.feature_names = self.vectorizer.get_feature_names_out()

            # Fit model
            self.doc_topic_matrix = self.model.fit_transform(doc_term_matrix)

            # Extract topics
            self._extract_topics()

        elif self.method == 'bertopic':
            # BERTopic approach
            self._initialize_model()
            topics, probs = self.model.fit_transform(documents)
            self.doc_topic_matrix = probs

            # Extract topics from BERTopic
            self._extract_topics_bertopic()

        # Calculate coherence if gensim available
        if GENSIM_AVAILABLE:
            self.coherence_score = self._calculate_coherence()

        return self

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """
        Fit model and return document-topic distribution

        Args:
            documents: List of text documents

        Returns:
            Document-topic matrix
        """
        self.fit(documents)
        return self.doc_topic_matrix

    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform new documents to topic distribution

        Args:
            documents: List of text documents

        Returns:
            Document-topic matrix
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.method in ['lda', 'nmf']:
            doc_term_matrix = self.vectorizer.transform(documents)
            return self.model.transform(doc_term_matrix)
        elif self.method == 'bertopic':
            topics, probs = self.model.transform(documents)
            return probs

    def _extract_topics(self, n_words: int = 10):
        """Extract top words for each topic (LDA/NMF)"""
        for topic_idx, topic in enumerate(self.model.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words = [self.feature_names[i] for i in top_indices]
            top_weights = [topic[i] for i in top_indices]

            self.topics[f'topic_{topic_idx}'] = {
                'id': topic_idx,
                'words': top_words,
                'weights': top_weights
            }

    def _extract_topics_bertopic(self, n_words: int = 10):
        """Extract topics from BERTopic model"""
        topic_info = self.model.get_topic_info()

        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id == -1:  # Skip outlier topic
                continue

            topic_words = self.model.get_topic(topic_id)
            if topic_words:
                words = [word for word, _ in topic_words[:n_words]]
                weights = [weight for _, weight in topic_words[:n_words]]

                self.topics[f'topic_{topic_id}'] = {
                    'id': topic_id,
                    'words': words,
                    'weights': weights,
                    'count': row['Count']
                }

    def _calculate_coherence(self) -> float:
        """
        Calculate topic coherence score using gensim

        Returns:
            Coherence score (higher is better)
        """
        if not GENSIM_AVAILABLE:
            return None

        try:
            # Tokenize documents
            tokenized_docs = [doc.lower().split() for doc in self.documents]

            # Create dictionary
            dictionary = Dictionary(tokenized_docs)

            # Get topics as word lists
            topics_words = [[word for word in topic['words']] for topic in self.topics.values()]

            # Calculate coherence
            coherence_model = CoherenceModel(
                topics=topics_words,
                texts=tokenized_docs,
                dictionary=dictionary,
                coherence='c_v'
            )

            return coherence_model.get_coherence()

        except Exception as e:
            print(f"Warning: Could not calculate coherence: {e}")
            return None

    def print_topics(self, n_words: int = 10):
        """
        Print topics with their top words

        Args:
            n_words: Number of top words to display per topic
        """
        print(f"\n{'='*80}")
        print(f"Topic Modeling Results ({self.method.upper()})")
        print(f"{'='*80}\n")

        for topic_name, topic_data in self.topics.items():
            print(f"Topic {topic_data['id']}:")
            words = topic_data['words'][:n_words]
            weights = topic_data['weights'][:n_words]

            for word, weight in zip(words, weights):
                print(f"  {word:20s} {weight:8.4f}")

            if 'count' in topic_data:
                print(f"  Documents: {topic_data['count']}")
            print()

        if self.coherence_score:
            print(f"Coherence Score: {self.coherence_score:.4f}\n")

    def get_document_topics(
        self,
        doc_idx: Optional[int] = None,
        threshold: float = 0.1
    ) -> Union[Dict[int, float], List[Dict[int, float]]]:
        """
        Get topic distribution for document(s)

        Args:
            doc_idx: Document index (None for all documents)
            threshold: Minimum probability threshold

        Returns:
            Topic distribution(s)
        """
        if self.doc_topic_matrix is None:
            raise ValueError("Model not fitted")

        if doc_idx is not None:
            # Single document
            doc_topics = self.doc_topic_matrix[doc_idx]
            return {
                topic_id: float(prob)
                for topic_id, prob in enumerate(doc_topics)
                if prob >= threshold
            }
        else:
            # All documents
            results = []
            for i in range(len(self.doc_topic_matrix)):
                doc_topics = self.doc_topic_matrix[i]
                results.append({
                    'document_id': i,
                    'topics': {
                        topic_id: float(prob)
                        for topic_id, prob in enumerate(doc_topics)
                        if prob >= threshold
                    },
                    'dominant_topic': int(np.argmax(doc_topics))
                })
            return results

    def visualize_topics(
        self,
        output_path: Optional[str] = None,
        figsize: tuple = (12, 8)
    ) -> Optional[str]:
        """
        Visualize topics

        Args:
            output_path: Path to save visualization
            figsize: Figure size

        Returns:
            Path to saved visualization or None
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available for visualization")
            return None

        try:
            fig, axes = plt.subplots(2, 1, figsize=figsize)

            # Topic word clouds (bar chart)
            topic_words = []
            topic_weights = []
            topic_labels = []

            for topic_data in list(self.topics.values())[:min(5, len(self.topics))]:
                topic_words.extend(topic_data['words'][:5])
                topic_weights.extend(topic_data['weights'][:5])
                topic_labels.extend([f"T{topic_data['id']}"] * 5)

            ax = axes[0]
            colors = plt.cm.viridis(np.linspace(0, 1, len(set(topic_labels))))
            color_map = {label: colors[i] for i, label in enumerate(set(topic_labels))}
            bar_colors = [color_map[label] for label in topic_labels]

            y_pos = np.arange(len(topic_words))
            ax.barh(y_pos, topic_weights, color=bar_colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"{label}: {word}" for label, word in zip(topic_labels, topic_words)])
            ax.set_xlabel('Weight')
            ax.set_title('Top Words per Topic')
            ax.invert_yaxis()

            # Document-topic distribution
            if self.doc_topic_matrix is not None and len(self.doc_topic_matrix) > 0:
                ax = axes[1]

                # Show topic distribution for first 20 documents
                n_docs = min(20, len(self.doc_topic_matrix))
                doc_topics = self.doc_topic_matrix[:n_docs]

                im = ax.imshow(doc_topics.T, aspect='auto', cmap='YlOrRd')
                ax.set_xlabel('Document')
                ax.set_ylabel('Topic')
                ax.set_title('Document-Topic Distribution')
                plt.colorbar(im, ax=ax, label='Probability')

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to {output_path}")
                plt.close()
                return output_path
            else:
                plt.show()
                return None

        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
            return None

    def visualize_interactive(self, output_path: str = 'topic_vis.html'):
        """
        Create interactive topic visualization with pyLDAvis

        Args:
            output_path: Path to save HTML visualization

        Returns:
            Path to saved visualization
        """
        if not PYLDAVIS_AVAILABLE:
            print("Warning: pyLDAvis not installed. Install with: pip install pyldavis")
            return None

        if self.method != 'lda':
            print("Warning: Interactive visualization only available for LDA")
            return None

        try:
            doc_term_matrix = self.vectorizer.transform(self.documents)

            vis = pyLDAvis.lda_model.prepare(
                self.model,
                doc_term_matrix,
                self.vectorizer,
                mds='tsne'
            )

            pyLDAvis.save_html(vis, output_path)
            print(f"Interactive visualization saved to {output_path}")
            return output_path

        except Exception as e:
            print(f"Warning: Interactive visualization failed: {e}")
            return None

    def get_results(self) -> Dict[str, Any]:
        """
        Get complete modeling results

        Returns:
            Dictionary with all results
        """
        return {
            'method': self.method,
            'n_topics': self.n_topics,
            'n_documents': len(self.documents) if self.documents else 0,
            'topics': self.topics,
            'coherence_score': self.coherence_score,
            'document_topics': self.get_document_topics() if self.doc_topic_matrix is not None else None
        }


# CLI interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Topic Modeling')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--column', type=str, required=True, help='Text column name')
    parser.add_argument('--n-topics', type=int, default=5, help='Number of topics')
    parser.add_argument('--method', type=str, default='lda',
                       choices=['lda', 'nmf', 'bertopic'], help='Topic modeling method')
    parser.add_argument('--max-features', type=int, default=5000, help='Max features')
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--visualize', type=str, help='Save visualization to file')
    parser.add_argument('--interactive', action='store_true', help='Create interactive visualization')

    args = parser.parse_args()

    try:
        # Load data
        print(f"Loading data from {args.input}...")
        df = pd.read_csv(args.input)

        if args.column not in df.columns:
            raise ValueError(f"Column '{args.column}' not found in CSV")

        documents = df[args.column].astype(str).tolist()
        print(f"Loaded {len(documents)} documents")

        # Initialize and fit model
        print(f"\nFitting {args.method.upper()} model with {args.n_topics} topics...")
        modeler = TopicModeler(
            n_topics=args.n_topics,
            method=args.method,
            max_features=args.max_features
        )

        modeler.fit(documents)

        # Print results
        modeler.print_topics()

        # Get results
        results = modeler.get_results()

        # Visualize
        if args.visualize:
            modeler.visualize_topics(output_path=args.visualize)

        if args.interactive and args.method == 'lda':
            modeler.visualize_interactive()

        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to {args.output}")
        else:
            print("\nResults:")
            print(json.dumps(results, indent=2, default=str))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
