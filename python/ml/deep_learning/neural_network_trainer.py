#!/usr/bin/env python3
"""
Neural Network Trainer Module for ML MCP System
Train deep learning models using TensorFlow/Keras
"""

import pandas as pd
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
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # Define dummy types for when TensorFlow is not available
    keras = None

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class NeuralNetworkTrainer:
    """Train neural networks for various tasks"""

    def __init__(self, random_state: int = 42):
        """
        Initialize neural network trainer

        Args:
            random_state: Random seed for reproducibility
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

        self.random_state = random_state
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.history = None

    def build_classifier(self, input_shape: int, num_classes: int,
                        hidden_layers: List[int] = [128, 64, 32],
                        dropout_rate: float = 0.3,
                        activation: str = 'relu'):
        """
        Build a neural network classifier

        Args:
            input_shape: Number of input features
            num_classes: Number of output classes
            hidden_layers: List of neurons per hidden layer
            dropout_rate: Dropout rate for regularization
            activation: Activation function

        Returns:
            Compiled Keras model
        """
        model = models.Sequential()

        # Input layer
        model.add(layers.Input(shape=(input_shape,)))

        # Hidden layers
        for i, units in enumerate(hidden_layers):
            model.add(layers.Dense(units, activation=activation, name=f'hidden_{i+1}'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))

        # Output layer
        if num_classes == 2:
            # Binary classification
            model.add(layers.Dense(1, activation='sigmoid', name='output'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
        else:
            # Multi-class classification
            model.add(layers.Dense(num_classes, activation='softmax', name='output'))
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )

        return model

    def build_regressor(self, input_shape: int,
                       hidden_layers: List[int] = [128, 64, 32],
                       dropout_rate: float = 0.2,
                       activation: str = 'relu'):
        """
        Build a neural network regressor

        Args:
            input_shape: Number of input features
            hidden_layers: List of neurons per hidden layer
            dropout_rate: Dropout rate
            activation: Activation function

        Returns:
            Compiled Keras model
        """
        model = models.Sequential()

        # Input layer
        model.add(layers.Input(shape=(input_shape,)))

        # Hidden layers
        for i, units in enumerate(hidden_layers):
            model.add(layers.Dense(units, activation=activation, name=f'hidden_{i+1}'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))

        # Output layer (single continuous value)
        model.add(layers.Dense(1, name='output'))

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )

        return model

    def train_classifier(self, X: pd.DataFrame, y: pd.Series,
                        test_size: float = 0.2,
                        epochs: int = 100,
                        batch_size: int = 32,
                        validation_split: float = 0.2,
                        early_stopping_patience: int = 10,
                        **kwargs) -> Dict[str, Any]:
        """
        Train a neural network classifier

        Args:
            X: Feature DataFrame
            y: Target series
            test_size: Test set size
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            early_stopping_patience: Early stopping patience
            **kwargs: Additional model parameters

        Returns:
            Training results and metrics
        """
        print("Training neural network classifier...", file=sys.stderr)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        num_classes = len(np.unique(y_encoded))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=self.random_state, stratify=y_encoded
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Build model
        hidden_layers = kwargs.get('hidden_layers', [128, 64, 32])
        dropout_rate = kwargs.get('dropout_rate', 0.3)

        self.model = self.build_classifier(
            input_shape=X_train_scaled.shape[1],
            num_classes=num_classes,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate
        )

        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        # Train model
        self.history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callback_list,
            verbose=1
        )

        # Evaluate on test set
        test_results = self.model.evaluate(X_test_scaled, y_test, verbose=0)

        # Make predictions
        if num_classes == 2:
            y_pred_proba = self.model.predict(X_test_scaled, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred_proba = self.model.predict(X_test_scaled, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)

        return {
            'model_type': 'neural_network_classifier',
            'num_classes': int(num_classes),
            'classes': self.label_encoder.classes_.tolist(),
            'input_features': X.columns.tolist(),
            'architecture': {
                'input_shape': X_train_scaled.shape[1],
                'hidden_layers': hidden_layers,
                'dropout_rate': dropout_rate,
                'total_parameters': self.model.count_params()
            },
            'training': {
                'epochs_trained': len(self.history.history['loss']),
                'final_train_loss': float(self.history.history['loss'][-1]),
                'final_val_loss': float(self.history.history['val_loss'][-1]),
                'best_epoch': int(np.argmin(self.history.history['val_loss'])) + 1
            },
            'test_metrics': {
                'accuracy': float(accuracy),
                'loss': float(test_results[0])
            },
            'training_history': {
                'loss': [float(x) for x in self.history.history['loss']],
                'val_loss': [float(x) for x in self.history.history['val_loss']],
                'accuracy': [float(x) for x in self.history.history['accuracy']],
                'val_accuracy': [float(x) for x in self.history.history['val_accuracy']]
            }
        }

    def train_regressor(self, X: pd.DataFrame, y: pd.Series,
                       test_size: float = 0.2,
                       epochs: int = 100,
                       batch_size: int = 32,
                       validation_split: float = 0.2,
                       early_stopping_patience: int = 10,
                       **kwargs) -> Dict[str, Any]:
        """
        Train a neural network regressor

        Args:
            X: Feature DataFrame
            y: Target series
            test_size: Test set size
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            early_stopping_patience: Early stopping patience
            **kwargs: Additional model parameters

        Returns:
            Training results and metrics
        """
        print("Training neural network regressor...", file=sys.stderr)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Build model
        hidden_layers = kwargs.get('hidden_layers', [128, 64, 32])
        dropout_rate = kwargs.get('dropout_rate', 0.2)

        self.model = self.build_regressor(
            input_shape=X_train_scaled.shape[1],
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate
        )

        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        # Train model
        self.history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callback_list,
            verbose=1
        )

        # Evaluate on test set
        test_results = self.model.evaluate(X_test_scaled, y_test, verbose=0)

        # Make predictions
        y_pred = self.model.predict(X_test_scaled, verbose=0).flatten()

        # Calculate R²
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)

        return {
            'model_type': 'neural_network_regressor',
            'input_features': X.columns.tolist(),
            'architecture': {
                'input_shape': X_train_scaled.shape[1],
                'hidden_layers': hidden_layers,
                'dropout_rate': dropout_rate,
                'total_parameters': self.model.count_params()
            },
            'training': {
                'epochs_trained': len(self.history.history['loss']),
                'final_train_loss': float(self.history.history['loss'][-1]),
                'final_val_loss': float(self.history.history['val_loss'][-1]),
                'best_epoch': int(np.argmin(self.history.history['val_loss'])) + 1
            },
            'test_metrics': {
                'r2_score': float(r2_score),
                'mae': float(test_results[1]),
                'mse': float(test_results[2]),
                'rmse': float(np.sqrt(test_results[2]))
            },
            'training_history': {
                'loss': [float(x) for x in self.history.history['loss']],
                'val_loss': [float(x) for x in self.history.history['val_loss']],
                'mae': [float(x) for x in self.history.history['mae']],
                'val_mae': [float(x) for x in self.history.history['val_mae']]
            }
        }

    def save_model(self, model_path: str):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model trained yet")

        self.model.save(model_path)
        print(f"Model saved to {model_path}", file=sys.stderr)

    def load_model(self, model_path: str):
        """Load trained model"""
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}", file=sys.stderr)


def main():
    """CLI interface"""
    if len(sys.argv) < 4:
        print("Usage: python neural_network_trainer.py <data_file> <target_column> <task_type>")
        print("task_type: classification or regression")
        sys.exit(1)

    data_file = sys.argv[1]
    target_column = sys.argv[2]
    task_type = sys.argv[3]

    try:
        # Load data
        df = pd.read_csv(data_file)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Select only numeric columns
        X = X.select_dtypes(include=[np.number])

        # Train model
        trainer = NeuralNetworkTrainer()

        if task_type == 'classification':
            results = trainer.train_classifier(X, y)
        elif task_type == 'regression':
            results = trainer.train_regressor(X, y)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        # Output results
        print(json.dumps(results, ensure_ascii=False, indent=2, default=str))

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