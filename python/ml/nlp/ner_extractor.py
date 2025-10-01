"""
Named Entity Recognition (NER) Extractor
Extract entities like persons, organizations, locations, dates from text

Features:
- SpaCy-based NER (fast, pre-trained models)
- Transformer-based NER (higher accuracy, slower)
- Custom entity types support
- Entity linking and disambiguation
- Confidence scoring
- Batch processing
- Entity frequency analysis
- Export to various formats (JSON, CSV, HTML)

Usage:
    from ner_extractor import NERExtractor

    extractor = NERExtractor(model='en_core_web_sm')
    entities = extractor.extract(text)
    extractor.visualize_entities(text)

CLI:
    python -m python.ml.nlp.ner_extractor --input documents.csv --column text --model en_core_web_sm
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# SpaCy for NER
try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not installed. Install with: pip install spacy")
    print("Then download model: python -m spacy download en_core_web_sm")

# Transformers for advanced NER
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Data processing
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not installed. Install with: pip install pandas numpy")


class NERExtractor:
    """
    Named Entity Recognition with multiple backends

    Supports SpaCy and Transformer models for entity extraction
    """

    def __init__(
        self,
        model: str = 'en_core_web_sm',
        backend: str = 'spacy',
        custom_entities: Optional[Dict[str, List[str]]] = None,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize NER extractor

        Args:
            model: Model name (spacy: 'en_core_web_sm', transformers: 'dslim/bert-base-NER')
            backend: 'spacy' or 'transformers'
            custom_entities: Custom entity patterns {label: [patterns]}
            confidence_threshold: Minimum confidence for entity extraction
        """
        self.model_name = model
        self.backend = backend.lower()
        self.custom_entities = custom_entities or {}
        self.confidence_threshold = confidence_threshold

        self.nlp = None
        self.ner_pipeline = None

        # Load model
        self._load_model()

        # Statistics
        self.entity_stats = defaultdict(int)

    def _load_model(self):
        """Load NER model based on backend"""
        if self.backend == 'spacy':
            if not SPACY_AVAILABLE:
                raise ImportError("spaCy not installed. Install with: pip install spacy")

            try:
                self.nlp = spacy.load(self.model_name)
                print(f"Loaded spaCy model: {self.model_name}")
            except OSError:
                raise ValueError(
                    f"Model '{self.model_name}' not found. "
                    f"Download with: python -m spacy download {self.model_name}"
                )

            # Add custom entity patterns if provided
            if self.custom_entities:
                self._add_custom_patterns()

        elif self.backend == 'transformers':
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "transformers not installed. Install with: pip install transformers torch"
                )

            try:
                self.ner_pipeline = pipeline(
                    "ner",
                    model=self.model_name,
                    aggregation_strategy="simple"
                )
                print(f"Loaded transformer model: {self.model_name}")
            except Exception as e:
                raise ValueError(f"Failed to load model '{self.model_name}': {e}")

        else:
            raise ValueError(f"Unknown backend: {self.backend}. Choose 'spacy' or 'transformers'")

    def _add_custom_patterns(self):
        """Add custom entity patterns to spaCy pipeline"""
        if not self.nlp:
            return

        ruler = self.nlp.add_pipe("entity_ruler", before="ner")

        patterns = []
        for label, pattern_list in self.custom_entities.items():
            for pattern in pattern_list:
                patterns.append({"label": label, "pattern": pattern})

        ruler.add_patterns(patterns)
        print(f"Added {len(patterns)} custom entity patterns")

    def extract(
        self,
        text: str,
        return_confidence: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from text

        Args:
            text: Input text
            return_confidence: Include confidence scores

        Returns:
            List of entities with metadata
        """
        if self.backend == 'spacy':
            return self._extract_spacy(text, return_confidence)
        elif self.backend == 'transformers':
            return self._extract_transformers(text, return_confidence)

    def _extract_spacy(
        self,
        text: str,
        return_confidence: bool = True
    ) -> List[Dict[str, Any]]:
        """Extract entities using spaCy"""
        doc = self.nlp(text)

        entities = []
        for ent in doc.ents:
            entity_data = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'start_token': ent.start,
                'end_token': ent.end
            }

            # Add confidence if available
            if return_confidence and hasattr(ent, 'confidence'):
                entity_data['confidence'] = float(ent.confidence)

            entities.append(entity_data)

            # Update stats
            self.entity_stats[ent.label_] += 1

        return entities

    def _extract_transformers(
        self,
        text: str,
        return_confidence: bool = True
    ) -> List[Dict[str, Any]]:
        """Extract entities using Transformers"""
        results = self.ner_pipeline(text)

        entities = []
        for result in results:
            if result['score'] >= self.confidence_threshold:
                entity_data = {
                    'text': result['word'],
                    'label': result['entity_group'],
                    'start': result['start'],
                    'end': result['end']
                }

                if return_confidence:
                    entity_data['confidence'] = float(result['score'])

                entities.append(entity_data)

                # Update stats
                self.entity_stats[result['entity_group']] += 1

        return entities

    def extract_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[List[Dict[str, Any]]]:
        """
        Extract entities from multiple texts

        Args:
            texts: List of texts
            show_progress: Show progress bar

        Returns:
            List of entity lists
        """
        results = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="Processing documents")
            except ImportError:
                iterator = texts
                print(f"Processing {len(texts)} documents...")
        else:
            iterator = texts

        for text in iterator:
            entities = self.extract(text)
            results.append(entities)

        return results

    def filter_entities(
        self,
        entities: List[Dict[str, Any]],
        entity_types: Optional[List[str]] = None,
        min_confidence: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter entities by type and confidence

        Args:
            entities: List of entities
            entity_types: Filter by entity types
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered entities
        """
        filtered = entities

        if entity_types:
            filtered = [e for e in filtered if e['label'] in entity_types]

        if min_confidence is not None:
            filtered = [
                e for e in filtered
                if e.get('confidence', 1.0) >= min_confidence
            ]

        return filtered

    def get_entity_frequencies(
        self,
        entities: List[Dict[str, Any]],
        by: str = 'label'
    ) -> Dict[str, int]:
        """
        Get entity frequency counts

        Args:
            entities: List of entities
            by: Count by 'label' or 'text'

        Returns:
            Frequency dictionary
        """
        if by == 'label':
            return dict(Counter(e['label'] for e in entities))
        elif by == 'text':
            return dict(Counter(e['text'] for e in entities))
        else:
            raise ValueError("'by' must be 'label' or 'text'")

    def get_entity_summary(
        self,
        entities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get summary statistics of extracted entities

        Args:
            entities: List of entities

        Returns:
            Summary statistics
        """
        summary = {
            'total_entities': len(entities),
            'entity_types': len(set(e['label'] for e in entities)),
            'unique_entities': len(set(e['text'].lower() for e in entities)),
            'by_label': self.get_entity_frequencies(entities, by='label'),
            'most_common': []
        }

        # Most common entities
        text_freq = Counter(e['text'] for e in entities)
        summary['most_common'] = [
            {'text': text, 'count': count, 'label': self._get_entity_label(text, entities)}
            for text, count in text_freq.most_common(10)
        ]

        # Average confidence if available
        confidences = [e['confidence'] for e in entities if 'confidence' in e]
        if confidences:
            summary['avg_confidence'] = float(np.mean(confidences))
            summary['min_confidence'] = float(np.min(confidences))
            summary['max_confidence'] = float(np.max(confidences))

        return summary

    def _get_entity_label(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Get label for entity text"""
        for e in entities:
            if e['text'] == text:
                return e['label']
        return 'UNKNOWN'

    def visualize_entities(
        self,
        text: str,
        output_path: Optional[str] = None,
        style: str = 'ent'
    ) -> Optional[str]:
        """
        Visualize entities in text

        Args:
            text: Input text
            output_path: Path to save HTML visualization
            style: 'ent' or 'dep' for dependency parsing

        Returns:
            HTML string or path to saved file
        """
        if self.backend != 'spacy':
            print("Warning: Visualization only available with spaCy backend")
            return None

        doc = self.nlp(text)

        if output_path:
            html = displacy.render(doc, style=style, page=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"Visualization saved to {output_path}")
            return output_path
        else:
            # Return HTML string for notebook display
            return displacy.render(doc, style=style, jupyter=False)

    def link_entities(
        self,
        entities: List[Dict[str, Any]],
        kb_type: str = 'wikipedia'
    ) -> List[Dict[str, Any]]:
        """
        Link entities to knowledge base (requires additional setup)

        Args:
            entities: List of entities
            kb_type: Knowledge base type ('wikipedia', 'wikidata')

        Returns:
            Entities with linked information
        """
        # Placeholder for entity linking
        # Full implementation would require entity linking models
        print(f"Warning: Entity linking to {kb_type} not fully implemented")

        linked_entities = []
        for entity in entities:
            entity_copy = entity.copy()
            entity_copy['kb_id'] = None
            entity_copy['kb_url'] = None
            linked_entities.append(entity_copy)

        return linked_entities

    def export_to_dataframe(
        self,
        entities: List[Dict[str, Any]]
    ) -> 'pd.DataFrame':
        """
        Export entities to pandas DataFrame

        Args:
            entities: List of entities

        Returns:
            DataFrame with entity data
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas not installed")

        return pd.DataFrame(entities)

    def export_to_csv(
        self,
        entities: List[Dict[str, Any]],
        output_path: str
    ):
        """
        Export entities to CSV

        Args:
            entities: List of entities
            output_path: Output CSV file path
        """
        df = self.export_to_dataframe(entities)
        df.to_csv(output_path, index=False)
        print(f"Entities exported to {output_path}")

    def export_to_json(
        self,
        entities: List[Dict[str, Any]],
        output_path: str
    ):
        """
        Export entities to JSON

        Args:
            entities: List of entities
            output_path: Output JSON file path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(entities, f, indent=2, ensure_ascii=False)
        print(f"Entities exported to {output_path}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get extractor statistics

        Returns:
            Statistics dictionary
        """
        return {
            'backend': self.backend,
            'model': self.model_name,
            'entity_types_seen': dict(self.entity_stats),
            'total_entities_extracted': sum(self.entity_stats.values())
        }


# CLI interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Named Entity Recognition')
    parser.add_argument('--input', type=str, help='Input text file or CSV')
    parser.add_argument('--column', type=str, help='Column name for CSV input')
    parser.add_argument('--text', type=str, help='Direct text input')
    parser.add_argument('--model', type=str, default='en_core_web_sm',
                       help='Model name (spacy: en_core_web_sm, transformers: dslim/bert-base-NER)')
    parser.add_argument('--backend', type=str, default='spacy',
                       choices=['spacy', 'transformers'], help='NER backend')
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--output-csv', type=str, help='Output CSV file')
    parser.add_argument('--visualize', type=str, help='Save HTML visualization')
    parser.add_argument('--entity-types', type=str, nargs='+',
                       help='Filter by entity types (e.g., PERSON ORG GPE)')
    parser.add_argument('--min-confidence', type=float, help='Minimum confidence threshold')

    args = parser.parse_args()

    try:
        # Initialize extractor
        extractor = NERExtractor(
            model=args.model,
            backend=args.backend,
            confidence_threshold=args.min_confidence or 0.5
        )

        # Get input text
        if args.text:
            text = args.text
            texts = [text]
        elif args.input:
            input_path = Path(args.input)
            if input_path.suffix == '.csv':
                if not args.column:
                    raise ValueError("--column required for CSV input")
                df = pd.read_csv(input_path)
                texts = df[args.column].astype(str).tolist()
                text = texts[0]  # For visualization
            else:
                with open(input_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                texts = [text]
        else:
            raise ValueError("Provide --text or --input")

        print(f"Processing {len(texts)} document(s)...\n")

        # Extract entities
        if len(texts) == 1:
            entities = extractor.extract(texts[0])
        else:
            all_entities = extractor.extract_batch(texts)
            # Flatten for analysis
            entities = [e for doc_entities in all_entities for e in doc_entities]

        # Filter entities
        if args.entity_types or args.min_confidence:
            entities = extractor.filter_entities(
                entities,
                entity_types=args.entity_types,
                min_confidence=args.min_confidence
            )

        # Print summary
        summary = extractor.get_entity_summary(entities)
        print(f"{'='*80}")
        print(f"Entity Extraction Results")
        print(f"{'='*80}")
        print(f"Total entities: {summary['total_entities']}")
        print(f"Entity types: {summary['entity_types']}")
        print(f"Unique entities: {summary['unique_entities']}")

        if 'avg_confidence' in summary:
            print(f"Average confidence: {summary['avg_confidence']:.3f}")

        print(f"\nBy entity type:")
        for label, count in sorted(summary['by_label'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {label:15s}: {count:5d}")

        print(f"\nMost common entities:")
        for item in summary['most_common'][:10]:
            print(f"  {item['text']:30s} ({item['label']:10s}): {item['count']:3d}")

        # Visualize
        if args.visualize and len(texts) == 1:
            extractor.visualize_entities(text, output_path=args.visualize)

        # Export
        if args.output:
            result = {
                'summary': summary,
                'entities': entities,
                'stats': extractor.get_stats()
            }
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to {args.output}")

        if args.output_csv:
            extractor.export_to_csv(entities, args.output_csv)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
