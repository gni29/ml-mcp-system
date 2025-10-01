"""
Document Similarity and Semantic Search
Calculate similarity between documents using various methods

Features:
- TF-IDF cosine similarity (fast, traditional)
- BERT embeddings similarity (semantic, accurate)
- Semantic search and retrieval
- Duplicate detection
- Document clustering by similarity
- Batch similarity computation
- Top-K similar documents retrieval
- Similarity matrix generation

Usage:
    from document_similarity import DocumentSimilarity

    sim = DocumentSimilarity(method='tfidf')
    sim.fit(documents)
    similar_docs = sim.find_similar(query, top_k=5)
    similarity_matrix = sim.compute_similarity_matrix()

CLI:
    python -m python.ml.nlp.document_similarity --input docs.csv --column text --query "sample query" --method tfidf
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Required dependencies
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Install with: pip install scikit-learn")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not installed. Install with: pip install pandas")

# Optional BERT embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Optional visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class DocumentSimilarity:
    """
    Document similarity computation with multiple methods

    Supports TF-IDF and BERT-based semantic similarity
    """

    def __init__(
        self,
        method: str = 'tfidf',
        model_name: Optional[str] = None,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 0.95
    ):
        """
        Initialize document similarity calculator

        Args:
            method: Similarity method ('tfidf', 'bert')
            model_name: Model name for BERT (e.g., 'all-MiniLM-L6-v2')
            max_features: Maximum features for TF-IDF
            ngram_range: N-gram range for TF-IDF
            min_df: Minimum document frequency
            max_df: Maximum document frequency
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

        self.method = method.lower()
        self.model_name = model_name
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df

        # Models
        self.vectorizer = None
        self.bert_model = None
        self.document_vectors = None
        self.documents = None

        # Initialize
        self._initialize()

    def _initialize(self):
        """Initialize models based on method"""
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words='english',
                sublinear_tf=True
            )

        elif self.method == 'bert':
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )

            # Use default model if not specified
            if not self.model_name:
                self.model_name = 'all-MiniLM-L6-v2'

            print(f"Loading BERT model: {self.model_name}")
            self.bert_model = SentenceTransformer(self.model_name)
            print("Model loaded successfully")

        else:
            raise ValueError(f"Unknown method: {self.method}. Choose 'tfidf' or 'bert'")

    def fit(self, documents: List[str]) -> 'DocumentSimilarity':
        """
        Fit model on documents

        Args:
            documents: List of text documents

        Returns:
            Self for method chaining
        """
        self.documents = documents

        if self.method == 'tfidf':
            print(f"Vectorizing {len(documents)} documents with TF-IDF...")
            self.document_vectors = self.vectorizer.fit_transform(documents)

        elif self.method == 'bert':
            print(f"Encoding {len(documents)} documents with BERT...")
            self.document_vectors = self.bert_model.encode(
                documents,
                show_progress_bar=True,
                convert_to_numpy=True
            )

        print(f"Document vectors shape: {self.document_vectors.shape}")
        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform new documents to vectors

        Args:
            documents: List of text documents

        Returns:
            Document vectors
        """
        if self.method == 'tfidf':
            if self.vectorizer is None:
                raise ValueError("Model not fitted. Call fit() first.")
            return self.vectorizer.transform(documents)

        elif self.method == 'bert':
            if self.bert_model is None:
                raise ValueError("Model not fitted. Call fit() first.")
            return self.bert_model.encode(documents, convert_to_numpy=True)

    def compute_similarity(
        self,
        doc1: Union[str, int],
        doc2: Union[str, int]
    ) -> float:
        """
        Compute similarity between two documents

        Args:
            doc1: Document text or index
            doc2: Document text or index

        Returns:
            Similarity score (0-1)
        """
        # Get vectors
        if isinstance(doc1, str):
            vec1 = self.transform([doc1])
        else:
            vec1 = self.document_vectors[doc1:doc1+1]

        if isinstance(doc2, str):
            vec2 = self.transform([doc2])
        else:
            vec2 = self.document_vectors[doc2:doc2+1]

        # Compute cosine similarity
        return float(cosine_similarity(vec1, vec2)[0][0])

    def compute_similarity_matrix(self) -> np.ndarray:
        """
        Compute pairwise similarity matrix for all documents

        Returns:
            Similarity matrix (n_docs x n_docs)
        """
        if self.document_vectors is None:
            raise ValueError("Model not fitted. Call fit() first.")

        print("Computing similarity matrix...")
        return cosine_similarity(self.document_vectors)

    def find_similar(
        self,
        query: Union[str, int],
        top_k: int = 5,
        return_scores: bool = True
    ) -> Union[List[int], List[Tuple[int, float]]]:
        """
        Find most similar documents to query

        Args:
            query: Query text or document index
            top_k: Number of similar documents to return
            return_scores: Return similarity scores

        Returns:
            List of document indices or (index, score) tuples
        """
        if self.document_vectors is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get query vector
        if isinstance(query, str):
            query_vec = self.transform([query])
        else:
            query_vec = self.document_vectors[query:query+1]

        # Compute similarities
        similarities = cosine_similarity(query_vec, self.document_vectors)[0]

        # Get top-k indices (excluding query itself if it's an index)
        if isinstance(query, int):
            similarities[query] = -1  # Exclude self

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        if return_scores:
            return [(int(idx), float(similarities[idx])) for idx in top_indices]
        else:
            return top_indices.tolist()

    def find_duplicates(
        self,
        threshold: float = 0.95
    ) -> List[Tuple[int, int, float]]:
        """
        Find near-duplicate documents

        Args:
            threshold: Similarity threshold for duplicates

        Returns:
            List of (doc1_idx, doc2_idx, similarity) tuples
        """
        similarity_matrix = self.compute_similarity_matrix()

        duplicates = []
        n_docs = len(self.documents)

        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                sim = similarity_matrix[i, j]
                if sim >= threshold:
                    duplicates.append((i, j, float(sim)))

        # Sort by similarity
        duplicates.sort(key=lambda x: x[2], reverse=True)

        return duplicates

    def cluster_similar_documents(
        self,
        n_clusters: int = 5
    ) -> Dict[str, Any]:
        """
        Cluster documents by similarity

        Args:
            n_clusters: Number of clusters

        Returns:
            Clustering results
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError("scikit-learn required for clustering")

        if self.document_vectors is None:
            raise ValueError("Model not fitted. Call fit() first.")

        print(f"Clustering {len(self.documents)} documents into {n_clusters} clusters...")

        # Handle sparse matrix for TF-IDF
        if self.method == 'tfidf':
            X = self.document_vectors.toarray()
        else:
            X = self.document_vectors

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Organize results
        clusters = {}
        for i, label in enumerate(labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)

        return {
            'n_clusters': n_clusters,
            'labels': labels.tolist(),
            'clusters': clusters,
            'cluster_sizes': {k: len(v) for k, v in clusters.items()}
        }

    def semantic_search(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> List[List[Tuple[int, float]]]:
        """
        Perform semantic search for multiple queries

        Args:
            queries: List of query strings
            top_k: Number of results per query

        Returns:
            List of results for each query
        """
        results = []
        for query in queries:
            similar_docs = self.find_similar(query, top_k=top_k, return_scores=True)
            results.append(similar_docs)

        return results

    def get_document_summary(
        self,
        doc_idx: int,
        similar_count: int = 3
    ) -> Dict[str, Any]:
        """
        Get summary for a document including similar documents

        Args:
            doc_idx: Document index
            similar_count: Number of similar documents to include

        Returns:
            Document summary
        """
        if doc_idx >= len(self.documents):
            raise ValueError(f"Document index {doc_idx} out of range")

        similar_docs = self.find_similar(doc_idx, top_k=similar_count+1, return_scores=True)
        # Remove self
        similar_docs = [s for s in similar_docs if s[0] != doc_idx][:similar_count]

        return {
            'document_id': doc_idx,
            'text': self.documents[doc_idx][:200] + '...' if len(self.documents[doc_idx]) > 200 else self.documents[doc_idx],
            'length': len(self.documents[doc_idx]),
            'similar_documents': [
                {
                    'id': idx,
                    'similarity': score,
                    'text': self.documents[idx][:100] + '...' if len(self.documents[idx]) > 100 else self.documents[idx]
                }
                for idx, score in similar_docs
            ]
        }

    def visualize_similarity_matrix(
        self,
        output_path: Optional[str] = None,
        max_docs: int = 50,
        figsize: Tuple[int, int] = (12, 10)
    ) -> Optional[str]:
        """
        Visualize document similarity matrix as heatmap

        Args:
            output_path: Path to save visualization
            max_docs: Maximum number of documents to visualize
            figsize: Figure size

        Returns:
            Path to saved file or None
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available for visualization")
            return None

        similarity_matrix = self.compute_similarity_matrix()

        # Limit size for visualization
        n_docs = min(max_docs, len(self.documents))
        sim_subset = similarity_matrix[:n_docs, :n_docs]

        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(
            sim_subset,
            cmap='YlOrRd',
            xticklabels=False,
            yticklabels=False,
            cbar_kws={'label': 'Similarity'}
        )
        plt.title(f'Document Similarity Matrix ({self.method.upper()})\n({n_docs} documents)')
        plt.xlabel('Document Index')
        plt.ylabel('Document Index')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
            plt.close()
            return output_path
        else:
            plt.show()
            return None

    def export_results(
        self,
        output_path: str,
        include_similarity_matrix: bool = False
    ):
        """
        Export similarity results to JSON

        Args:
            output_path: Output file path
            include_similarity_matrix: Include full similarity matrix
        """
        results = {
            'method': self.method,
            'n_documents': len(self.documents),
            'model_name': self.model_name if self.method == 'bert' else None
        }

        if include_similarity_matrix:
            sim_matrix = self.compute_similarity_matrix()
            results['similarity_matrix'] = sim_matrix.tolist()

        # Add summary statistics
        if self.document_vectors is not None:
            sim_matrix = self.compute_similarity_matrix()
            # Get upper triangle (excluding diagonal)
            triu_indices = np.triu_indices_from(sim_matrix, k=1)
            similarities = sim_matrix[triu_indices]

            results['statistics'] = {
                'mean_similarity': float(np.mean(similarities)),
                'median_similarity': float(np.median(similarities)),
                'std_similarity': float(np.std(similarities)),
                'min_similarity': float(np.min(similarities)),
                'max_similarity': float(np.max(similarities))
            }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results exported to {output_path}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get similarity calculator statistics

        Returns:
            Statistics dictionary
        """
        stats = {
            'method': self.method,
            'n_documents': len(self.documents) if self.documents else 0,
        }

        if self.method == 'tfidf':
            if self.vectorizer and hasattr(self.vectorizer, 'vocabulary_'):
                stats['vocabulary_size'] = len(self.vectorizer.vocabulary_)
                stats['n_features'] = self.document_vectors.shape[1]

        elif self.method == 'bert':
            stats['model_name'] = self.model_name
            if self.document_vectors is not None:
                stats['embedding_dim'] = self.document_vectors.shape[1]

        return stats


# CLI interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Document Similarity')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--column', type=str, required=True, help='Text column name')
    parser.add_argument('--method', type=str, default='tfidf',
                       choices=['tfidf', 'bert'], help='Similarity method')
    parser.add_argument('--model', type=str, help='BERT model name (e.g., all-MiniLM-L6-v2)')
    parser.add_argument('--query', type=str, help='Query text for similarity search')
    parser.add_argument('--query-idx', type=int, help='Query document index')
    parser.add_argument('--top-k', type=int, default=5, help='Number of similar documents to return')
    parser.add_argument('--find-duplicates', action='store_true', help='Find duplicate documents')
    parser.add_argument('--duplicate-threshold', type=float, default=0.95,
                       help='Similarity threshold for duplicates')
    parser.add_argument('--cluster', type=int, help='Number of clusters')
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--visualize', type=str, help='Save similarity matrix visualization')

    args = parser.parse_args()

    try:
        # Load data
        print(f"Loading data from {args.input}...")
        df = pd.read_csv(args.input)

        if args.column not in df.columns:
            raise ValueError(f"Column '{args.column}' not found in CSV")

        documents = df[args.column].astype(str).tolist()
        print(f"Loaded {len(documents)} documents")

        # Initialize similarity calculator
        sim = DocumentSimilarity(
            method=args.method,
            model_name=args.model
        )

        # Fit on documents
        sim.fit(documents)

        results = {
            'method': args.method,
            'n_documents': len(documents),
            'stats': sim.get_stats()
        }

        # Query similarity
        if args.query or args.query_idx is not None:
            query = args.query if args.query else args.query_idx
            print(f"\nFinding {args.top_k} most similar documents...")

            similar_docs = sim.find_similar(query, top_k=args.top_k, return_scores=True)

            print(f"\n{'='*80}")
            print(f"Most Similar Documents")
            print(f"{'='*80}\n")

            for rank, (idx, score) in enumerate(similar_docs, 1):
                doc_text = documents[idx][:100] + '...' if len(documents[idx]) > 100 else documents[idx]
                print(f"{rank}. Document {idx} (similarity: {score:.4f})")
                print(f"   {doc_text}\n")

            results['query_results'] = [
                {'rank': i+1, 'document_id': idx, 'similarity': score, 'text': documents[idx][:200]}
                for i, (idx, score) in enumerate(similar_docs)
            ]

        # Find duplicates
        if args.find_duplicates:
            print(f"\nFinding duplicates (threshold: {args.duplicate_threshold})...")
            duplicates = sim.find_duplicates(threshold=args.duplicate_threshold)

            print(f"\n{'='*80}")
            print(f"Found {len(duplicates)} duplicate pairs")
            print(f"{'='*80}\n")

            for doc1, doc2, score in duplicates[:10]:
                print(f"Documents {doc1} and {doc2}: {score:.4f}")

            results['duplicates'] = [
                {'doc1': d1, 'doc2': d2, 'similarity': score}
                for d1, d2, score in duplicates
            ]

        # Clustering
        if args.cluster:
            cluster_results = sim.cluster_similar_documents(n_clusters=args.cluster)

            print(f"\n{'='*80}")
            print(f"Clustering Results")
            print(f"{'='*80}\n")

            for cluster_id, doc_indices in cluster_results['clusters'].items():
                print(f"Cluster {cluster_id}: {len(doc_indices)} documents")

            results['clustering'] = cluster_results

        # Visualize
        if args.visualize:
            sim.visualize_similarity_matrix(output_path=args.visualize)

        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
        else:
            print("\nResults:")
            print(json.dumps(results, indent=2, default=str))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
