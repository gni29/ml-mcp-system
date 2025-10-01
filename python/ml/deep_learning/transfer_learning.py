#!/usr/bin/env python3
"""
Transfer Learning Module for ML MCP System
Use pre-trained models for computer vision and NLP tasks
"""

import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import (
        VGG16, VGG19, ResNet50, ResNet101,
        MobileNetV2, InceptionV3, EfficientNetB0
    )
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # Define dummy types
    keras = None
    VGG16 = VGG19 = ResNet50 = ResNet101 = None
    MobileNetV2 = InceptionV3 = EfficientNetB0 = None


class TransferLearningModel:
    """Transfer learning for image classification"""

    AVAILABLE_MODELS = {
        'vgg16': (VGG16, (224, 224)),
        'vgg19': (VGG19, (224, 224)),
        'resnet50': (ResNet50, (224, 224)),
        'resnet101': (ResNet101, (224, 224)),
        'mobilenetv2': (MobileNetV2, (224, 224)),
        'inceptionv3': (InceptionV3, (299, 299)),
        'efficientnetb0': (EfficientNetB0, (224, 224))
    }

    def __init__(self, base_model_name: str = 'resnet50', num_classes: int = 10):
        """
        Initialize transfer learning model

        Args:
            base_model_name: Name of pre-trained model
            num_classes: Number of output classes
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required")

        if base_model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model {base_model_name} not available. Choose from: {list(self.AVAILABLE_MODELS.keys())}")

        self.base_model_name = base_model_name
        self.num_classes = num_classes
        self.model = None
        self.history = None

    def build_model(self, trainable_layers: int = 0,
                   dense_units: List[int] = [256],
                   dropout_rate: float = 0.5):
        """
        Build transfer learning model

        Args:
            trainable_layers: Number of top layers to fine-tune (0 = freeze all)
            dense_units: Units in dense layers before output
            dropout_rate: Dropout rate

        Returns:
            Compiled model
        """
        base_model_class, input_shape = self.AVAILABLE_MODELS[self.base_model_name]

        # Load pre-trained model
        base_model = base_model_class(
            include_top=False,
            weights='imagenet',
            input_shape=(*input_shape, 3),
            pooling='avg'
        )

        # Freeze base model layers
        base_model.trainable = False

        if trainable_layers > 0:
            # Fine-tune top layers
            for layer in base_model.layers[-trainable_layers:]:
                layer.trainable = True

        # Build model
        model = models.Sequential([
            base_model
        ])

        # Add custom layers
        for units in dense_units:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))

        # Output layer
        if self.num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(self.num_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss=loss,
            metrics=['accuracy']
        )

        self.model = model
        return model

    def get_model_summary(self) -> Dict[str, Any]:
        """Get model architecture summary"""
        if self.model is None:
            raise ValueError("Model not built yet")

        trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        non_trainable_params = sum([tf.size(w).numpy() for w in self.model.non_trainable_weights])

        return {
            'base_model': self.base_model_name,
            'num_classes': self.num_classes,
            'total_layers': len(self.model.layers),
            'trainable_parameters': int(trainable_params),
            'non_trainable_parameters': int(non_trainable_params),
            'total_parameters': int(trainable_params + non_trainable_params)
        }


class FeatureExtractor:
    """Extract features using pre-trained models"""

    def __init__(self, model_name: str = 'resnet50'):
        """
        Initialize feature extractor

        Args:
            model_name: Pre-trained model to use
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required")

        if model_name not in TransferLearningModel.AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} not available")

        self.model_name = model_name
        self.model = self._load_model()

    def _load_model(self):
        """Load pre-trained model for feature extraction"""
        model_class, input_shape = TransferLearningModel.AVAILABLE_MODELS[self.model_name]

        model = model_class(
            include_top=False,
            weights='imagenet',
            input_shape=(*input_shape, 3),
            pooling='avg'
        )

        return model

    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract features from images

        Args:
            images: Array of images (N, H, W, C)

        Returns:
            Feature vectors (N, feature_dim)
        """
        # Preprocess images
        if self.model_name.startswith('vgg'):
            from tensorflow.keras.applications.vgg16 import preprocess_input
        elif self.model_name.startswith('resnet'):
            from tensorflow.keras.applications.resnet import preprocess_input
        elif self.model_name == 'mobilenetv2':
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        elif self.model_name == 'inceptionv3':
            from tensorflow.keras.applications.inception_v3 import preprocess_input
        elif self.model_name.startswith('efficientnet'):
            from tensorflow.keras.applications.efficientnet import preprocess_input
        else:
            preprocess_input = lambda x: x

        images_preprocessed = preprocess_input(images)

        # Extract features
        features = self.model.predict(images_preprocessed, verbose=0)

        return features


class EmbeddingExtractor:
    """Extract embeddings for text or sequences"""

    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128,
                 max_length: int = 100):
        """
        Initialize embedding extractor

        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            max_length: Maximum sequence length
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.model = None

    def build_embedding_model(self, use_pretrained: bool = False) -> keras.Model:
        """
        Build embedding model

        Args:
            use_pretrained: Whether to use pre-trained embeddings

        Returns:
            Embedding model
        """
        model = models.Sequential([
            layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_length,
                mask_zero=True
            ),
            layers.GlobalAveragePooling1D()
        ])

        self.model = model
        return model

    def extract_embeddings(self, sequences: np.ndarray) -> np.ndarray:
        """
        Extract embeddings from sequences

        Args:
            sequences: Padded sequences (N, max_length)

        Returns:
            Embeddings (N, embedding_dim)
        """
        if self.model is None:
            self.build_embedding_model()

        embeddings = self.model.predict(sequences, verbose=0)
        return embeddings


def get_available_models() -> Dict[str, Any]:
    """Get list of available pre-trained models"""
    models_info = {}

    for name, (model_class, input_shape) in TransferLearningModel.AVAILABLE_MODELS.items():
        models_info[name] = {
            'input_shape': input_shape,
            'description': f"{model_class.__name__} pre-trained on ImageNet"
        }

    return models_info


def main():
    """CLI interface"""
    if len(sys.argv) < 2:
        print("Usage: python transfer_learning.py <action>")
        print("Actions:")
        print("  list_models - List available pre-trained models")
        print("  info <model_name> - Get model information")
        sys.exit(1)

    action = sys.argv[1]

    try:
        if action == 'list_models':
            models = get_available_models()
            result = {
                'available_models': list(models.keys()),
                'models_info': models
            }

        elif action == 'info' and len(sys.argv) >= 3:
            model_name = sys.argv[2]
            transfer_model = TransferLearningModel(model_name, num_classes=10)
            transfer_model.build_model()
            result = transfer_model.get_model_summary()

        else:
            result = {'error': 'Invalid action or missing arguments'}

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