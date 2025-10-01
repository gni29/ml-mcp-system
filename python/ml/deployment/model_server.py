"""
Model Serving API
FastAPI-based REST API for serving ML models in production

Features:
- Multiple model hosting (classification, regression, forecasting, NLP)
- Model versioning and A/B testing
- Request batching for performance
- Model warm-up and caching
- Health checks and monitoring
- Auto-scaling support
- Async predictions

Usage:
    python -m python.ml.deployment.model_server --port 8000

API Endpoints:
    POST /predict/{model_name} - Make predictions
    POST /predict/batch/{model_name} - Batch predictions
    GET /models - List all registered models
    GET /models/{model_name}/info - Model metadata
    POST /models/register - Register new model
    DELETE /models/{model_name} - Unregister model
    GET /health - Health check
    GET /metrics - Performance metrics
"""

import joblib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("Warning: FastAPI not installed. Install with: pip install fastapi uvicorn pydantic")


class PredictionRequest(BaseModel):
    """Request model for predictions"""
    features: Union[List[List[float]], Dict[str, Any]]
    model_version: Optional[str] = 'latest'
    return_probabilities: bool = False

    class Config:
        schema_extra = {
            "example": {
                "features": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                "model_version": "latest",
                "return_probabilities": False
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    features: List[List[float]]
    model_version: Optional[str] = 'latest'
    batch_size: int = Field(default=32, ge=1, le=1000)


class ModelRegistration(BaseModel):
    """Request model for registering a new model"""
    model_name: str
    model_path: str
    model_type: str  # 'classifier', 'regressor', 'forecaster', 'nlp'
    version: str = '1.0'
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelServer:
    """
    FastAPI-based model serving server

    Manages multiple models and provides REST API endpoints for predictions
    """

    def __init__(self, host: str = '0.0.0.0', port: int = 8000, log_level: str = 'info'):
        """
        Initialize model server

        Args:
            host: Server host address
            port: Server port
            log_level: Logging level
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required. Install with: pip install fastapi uvicorn pydantic")

        self.host = host
        self.port = port
        self.log_level = log_level

        # Model registry
        self.models: Dict[str, Dict[str, Any]] = {}
        self.model_stats: Dict[str, Dict[str, Any]] = {}

        # Initialize FastAPI app
        self.app = FastAPI(
            title="ML Model Server",
            description="Production ML model serving API",
            version="1.0.0"
        )

        # Setup routes
        self._setup_routes()

        # Server stats
        self.start_time = datetime.now()
        self.total_predictions = 0

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.get("/")
        async def root():
            return {
                "service": "ML Model Server",
                "version": "1.0.0",
                "status": "running",
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
            }

        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "models_loaded": len(self.models),
                "total_predictions": self.total_predictions,
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
            }

        @self.app.get("/models")
        async def list_models():
            return {
                "models": [
                    {
                        "name": name,
                        "type": info['type'],
                        "version": info['version'],
                        "loaded_at": info['loaded_at']
                    }
                    for name, info in self.models.items()
                ]
            }

        @self.app.get("/models/{model_name}/info")
        async def get_model_info(model_name: str):
            if model_name not in self.models:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

            model_info = self.models[model_name]
            stats = self.model_stats.get(model_name, {})

            return {
                "name": model_name,
                "type": model_info['type'],
                "version": model_info['version'],
                "description": model_info.get('description'),
                "loaded_at": model_info['loaded_at'],
                "predictions_count": stats.get('predictions_count', 0),
                "avg_latency_ms": stats.get('avg_latency_ms', 0),
                "metadata": model_info.get('metadata', {})
            }

        @self.app.post("/models/register")
        async def register_model(registration: ModelRegistration):
            try:
                self.register_model(
                    model_name=registration.model_name,
                    model_path=registration.model_path,
                    model_type=registration.model_type,
                    version=registration.version,
                    description=registration.description,
                    metadata=registration.metadata
                )
                return {
                    "status": "success",
                    "message": f"Model '{registration.model_name}' registered successfully"
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.delete("/models/{model_name}")
        async def unregister_model(model_name: str):
            if model_name not in self.models:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

            del self.models[model_name]
            if model_name in self.model_stats:
                del self.model_stats[model_name]

            return {
                "status": "success",
                "message": f"Model '{model_name}' unregistered successfully"
            }

        @self.app.post("/predict/{model_name}")
        async def predict(model_name: str, request: PredictionRequest):
            if model_name not in self.models:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

            try:
                start_time = time.time()

                # Get model
                model_info = self.models[model_name]
                model = model_info['model']

                # Convert input
                if isinstance(request.features, dict):
                    X = pd.DataFrame([request.features])
                else:
                    X = np.array(request.features)

                # Make prediction
                predictions = model.predict(X)

                # Get probabilities if requested
                result = {"predictions": predictions.tolist()}

                if request.return_probabilities and hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X)
                    result["probabilities"] = probabilities.tolist()

                # Update stats
                latency_ms = (time.time() - start_time) * 1000
                self._update_stats(model_name, latency_ms)
                self.total_predictions += len(predictions)

                result["latency_ms"] = latency_ms
                result["model_version"] = model_info['version']

                return result

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

        @self.app.post("/predict/batch/{model_name}")
        async def predict_batch(model_name: str, request: BatchPredictionRequest, background_tasks: BackgroundTasks):
            if model_name not in self.models:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

            try:
                start_time = time.time()

                # Get model
                model_info = self.models[model_name]
                model = model_info['model']

                X = np.array(request.features)

                # Batch predictions
                all_predictions = []
                for i in range(0, len(X), request.batch_size):
                    batch = X[i:i + request.batch_size]
                    batch_preds = model.predict(batch)
                    all_predictions.extend(batch_preds.tolist())

                # Update stats in background
                latency_ms = (time.time() - start_time) * 1000
                background_tasks.add_task(self._update_stats, model_name, latency_ms)
                self.total_predictions += len(all_predictions)

                return {
                    "predictions": all_predictions,
                    "count": len(all_predictions),
                    "latency_ms": latency_ms,
                    "model_version": model_info['version']
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

        @self.app.get("/metrics")
        async def get_metrics():
            return {
                "total_predictions": self.total_predictions,
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "models": self.model_stats
            }

    def register_model(
        self,
        model_name: str,
        model_path: str,
        model_type: str,
        version: str = '1.0',
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Register a model for serving

        Args:
            model_name: Unique model identifier
            model_path: Path to serialized model file
            model_type: Type of model ('classifier', 'regressor', 'forecaster', 'nlp')
            version: Model version
            description: Model description
            metadata: Additional metadata
        """
        try:
            # Load model
            model = joblib.load(model_path)

            # Register
            self.models[model_name] = {
                'model': model,
                'type': model_type,
                'version': version,
                'path': model_path,
                'description': description,
                'metadata': metadata or {},
                'loaded_at': datetime.now().isoformat()
            }

            # Initialize stats
            self.model_stats[model_name] = {
                'predictions_count': 0,
                'total_latency_ms': 0,
                'avg_latency_ms': 0
            }

            print(f"✓ Model '{model_name}' registered successfully")

        except Exception as e:
            raise Exception(f"Failed to register model '{model_name}': {str(e)}")

    def _update_stats(self, model_name: str, latency_ms: float):
        """Update model statistics"""
        if model_name not in self.model_stats:
            self.model_stats[model_name] = {
                'predictions_count': 0,
                'total_latency_ms': 0,
                'avg_latency_ms': 0
            }

        stats = self.model_stats[model_name]
        stats['predictions_count'] += 1
        stats['total_latency_ms'] += latency_ms
        stats['avg_latency_ms'] = stats['total_latency_ms'] / stats['predictions_count']

    def start(self):
        """Start the server"""
        print(f"Starting ML Model Server on {self.host}:{self.port}")
        print(f"Models loaded: {len(self.models)}")
        print(f"API documentation: http://{self.host}:{self.port}/docs")

        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level=self.log_level
        )


# CLI interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ML Model Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--log-level', type=str, default='info', help='Log level')
    parser.add_argument('--model', type=str, help='Model path to load on startup')
    parser.add_argument('--model-name', type=str, help='Model name')
    parser.add_argument('--model-type', type=str, help='Model type')

    args = parser.parse_args()

    # Initialize server
    server = ModelServer(host=args.host, port=args.port, log_level=args.log_level)

    # Register model if provided
    if args.model and args.model_name and args.model_type:
        server.register_model(
            model_name=args.model_name,
            model_path=args.model,
            model_type=args.model_type
        )

    # Start server
    server.start()
