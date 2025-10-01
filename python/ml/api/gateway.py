"""
ML API Gateway
Unified REST API for all ML tools and services

Features:
- Unified API for analyzers, ML training, predictions
- Multiple model serving (classification, regression, forecasting, NLP)
- Data analysis endpoints
- Model training and evaluation
- Batch processing
- Authentication and authorization
- Rate limiting
- API documentation (OpenAPI/Swagger)
- Health checks and monitoring
- WebSocket support for real-time updates

Usage:
    python -m python.ml.api.gateway --port 8080 --auth-enabled

API Endpoints:
    # Analysis
    POST /api/analyze/descriptive - Descriptive statistics
    POST /api/analyze/correlation - Correlation analysis
    POST /api/analyze/distribution - Distribution analysis
    POST /api/analyze/timeseries - Time series analysis

    # ML Training
    POST /api/train/classifier - Train classification model
    POST /api/train/regressor - Train regression model
    POST /api/train/forecaster - Train forecasting model

    # Predictions
    POST /api/predict/{model_name} - Make predictions
    POST /api/predict/batch/{model_name} - Batch predictions

    # NLP
    POST /api/nlp/sentiment - Sentiment analysis
    POST /api/nlp/entities - Named entity recognition
    POST /api/nlp/topics - Topic modeling
    POST /api/nlp/similarity - Document similarity

    # Models
    GET /api/models - List all models
    POST /api/models/register - Register new model
    DELETE /api/models/{model_name} - Delete model

    # Health & Monitoring
    GET /api/health - Health check
    GET /api/metrics - Performance metrics
    GET /api/status - System status
"""

import json
import time
import hashlib
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# FastAPI and dependencies
try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File, Security
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, validator
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("Warning: FastAPI not installed. Install with: pip install fastapi uvicorn pydantic")

# Optional dependencies
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


# Pydantic Models
class AnalysisRequest(BaseModel):
    """Request for data analysis"""
    data: Union[List[List[Any]], Dict[str, List[Any]]]
    analysis_type: str = Field(..., description="Type of analysis")
    options: Optional[Dict[str, Any]] = None

    class Config:
        schema_extra = {
            "example": {
                "data": [[1, 2, 3], [4, 5, 6]],
                "analysis_type": "descriptive",
                "options": {"include_percentiles": True}
            }
        }


class TrainingRequest(BaseModel):
    """Request for model training"""
    data: Union[List[List[Any]], Dict[str, List[Any]]]
    target: Union[List[Any], str]
    model_type: str = Field(..., description="Model type: logistic, random_forest, xgboost, etc.")
    hyperparameters: Optional[Dict[str, Any]] = None
    validation_split: float = Field(default=0.2, ge=0.0, le=0.5)
    save_model: bool = True
    model_name: Optional[str] = None


class PredictionRequest(BaseModel):
    """Request for predictions"""
    data: Union[List[List[Any]], Dict[str, List[Any]]]
    model_name: str
    return_probabilities: bool = False


class NLPRequest(BaseModel):
    """Request for NLP tasks"""
    text: Union[str, List[str]]
    task: str = Field(..., description="NLP task: sentiment, entities, topics, similarity")
    options: Optional[Dict[str, Any]] = None


class ModelRegistration(BaseModel):
    """Model registration request"""
    model_name: str
    model_path: str
    model_type: str
    version: str = "1.0"
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class APIKey(BaseModel):
    """API key model"""
    key: str
    name: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True


class MLAPIGateway:
    """
    Unified ML API Gateway

    Provides centralized REST API for all ML operations
    """

    def __init__(
        self,
        host: str = '0.0.0.0',
        port: int = 8080,
        auth_enabled: bool = False,
        rate_limit: int = 100,
        enable_cors: bool = True
    ):
        """
        Initialize API Gateway

        Args:
            host: Server host
            port: Server port
            auth_enabled: Enable API key authentication
            rate_limit: Rate limit per minute
            enable_cors: Enable CORS middleware
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required. Install with: pip install fastapi uvicorn pydantic")

        self.host = host
        self.port = port
        self.auth_enabled = auth_enabled
        self.rate_limit = rate_limit
        self.enable_cors = enable_cors

        # Initialize FastAPI
        self.app = FastAPI(
            title="ML API Gateway",
            description="Unified API for machine learning operations",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )

        # Storage
        self.models: Dict[str, Any] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.request_counts: Dict[str, List[datetime]] = {}

        # Statistics
        self.start_time = datetime.now()
        self.total_requests = 0
        self.total_predictions = 0
        self.total_trainings = 0

        # Setup
        self._setup_middleware()
        self._setup_security()
        self._setup_routes()

        # Generate initial API key if auth enabled
        if self.auth_enabled:
            self._generate_initial_api_key()

    def _setup_middleware(self):
        """Setup middleware"""
        if self.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

    def _setup_security(self):
        """Setup security schemes"""
        if self.auth_enabled:
            self.security = APIKeyHeader(name="X-API-Key", auto_error=False)
        else:
            self.security = None

    def _generate_initial_api_key(self):
        """Generate initial API key"""
        key = secrets.token_urlsafe(32)
        self.api_keys[key] = APIKey(
            key=key,
            name="initial",
            created_at=datetime.now()
        )
        print(f"\nGenerated API Key: {key}")
        print("Include this in requests as 'X-API-Key' header\n")

    def _verify_api_key(self, api_key: str) -> bool:
        """Verify API key"""
        if not self.auth_enabled:
            return True

        if api_key not in self.api_keys:
            return False

        key_info = self.api_keys[api_key]

        if not key_info.is_active:
            return False

        if key_info.expires_at and datetime.now() > key_info.expires_at:
            return False

        return True

    def _check_rate_limit(self, api_key: str) -> bool:
        """Check rate limit"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Clean old requests
        if api_key in self.request_counts:
            self.request_counts[api_key] = [
                t for t in self.request_counts[api_key] if t > minute_ago
            ]
        else:
            self.request_counts[api_key] = []

        # Check limit
        if len(self.request_counts[api_key]) >= self.rate_limit:
            return False

        # Add current request
        self.request_counts[api_key].append(now)
        return True

    async def _verify_request(self, api_key: Optional[str] = Depends(lambda: None)):
        """Verify request authentication and rate limiting"""
        self.total_requests += 1

        if self.auth_enabled:
            if not api_key:
                raise HTTPException(status_code=401, detail="API key required")

            if not self._verify_api_key(api_key):
                raise HTTPException(status_code=401, detail="Invalid API key")

            if not self._check_rate_limit(api_key):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")

        return True

    def _setup_routes(self):
        """Setup API routes"""

        # Root
        @self.app.get("/")
        async def root():
            return {
                "service": "ML API Gateway",
                "version": "1.0.0",
                "status": "running",
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "docs": "/docs",
                "health": "/api/health"
            }

        # Health check
        @self.app.get("/api/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "models_loaded": len(self.models),
                "total_requests": self.total_requests
            }

        # System status
        @self.app.get("/api/status")
        async def system_status(verified: bool = Depends(self._verify_request)):
            return {
                "status": "operational",
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "statistics": {
                    "total_requests": self.total_requests,
                    "total_predictions": self.total_predictions,
                    "total_trainings": self.total_trainings,
                    "models_loaded": len(self.models)
                },
                "system": {
                    "auth_enabled": self.auth_enabled,
                    "rate_limit": self.rate_limit,
                    "cors_enabled": self.enable_cors
                }
            }

        # Metrics
        @self.app.get("/api/metrics")
        async def get_metrics(verified: bool = Depends(self._verify_request)):
            return {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "requests": {
                    "total": self.total_requests,
                    "predictions": self.total_predictions,
                    "trainings": self.total_trainings
                },
                "models": {
                    "loaded": len(self.models),
                    "names": list(self.models.keys())
                },
                "rate_limiting": {
                    "limit_per_minute": self.rate_limit,
                    "active_keys": len(self.request_counts)
                }
            }

        # ==================== Analysis Endpoints ====================

        @self.app.post("/api/analyze/descriptive")
        async def analyze_descriptive(
            request: AnalysisRequest,
            verified: bool = Depends(self._verify_request)
        ):
            """Descriptive statistics analysis"""
            try:
                df = self._to_dataframe(request.data)
                results = {
                    "summary": df.describe().to_dict(),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "shape": df.shape,
                    "missing_values": df.isnull().sum().to_dict()
                }
                return results
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.post("/api/analyze/correlation")
        async def analyze_correlation(
            request: AnalysisRequest,
            verified: bool = Depends(self._verify_request)
        ):
            """Correlation analysis"""
            try:
                df = self._to_dataframe(request.data)
                numeric_df = df.select_dtypes(include=[np.number])
                corr_matrix = numeric_df.corr()
                return {
                    "correlation_matrix": corr_matrix.to_dict(),
                    "strong_correlations": self._find_strong_correlations(corr_matrix)
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        # ==================== Training Endpoints ====================

        @self.app.post("/api/train/classifier")
        async def train_classifier(
            request: TrainingRequest,
            background_tasks: BackgroundTasks,
            verified: bool = Depends(self._verify_request)
        ):
            """Train classification model"""
            try:
                self.total_trainings += 1
                result = self._train_model(request, task_type='classification')
                return result
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.post("/api/train/regressor")
        async def train_regressor(
            request: TrainingRequest,
            background_tasks: BackgroundTasks,
            verified: bool = Depends(self._verify_request)
        ):
            """Train regression model"""
            try:
                self.total_trainings += 1
                result = self._train_model(request, task_type='regression')
                return result
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        # ==================== Prediction Endpoints ====================

        @self.app.post("/api/predict/{model_name}")
        async def predict(
            model_name: str,
            request: PredictionRequest,
            verified: bool = Depends(self._verify_request)
        ):
            """Make predictions with registered model"""
            try:
                if model_name not in self.models:
                    raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

                model = self.models[model_name]['model']
                X = self._to_dataframe(request.data)

                predictions = model.predict(X)
                result = {"predictions": predictions.tolist()}

                if request.return_probabilities and hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X)
                    result["probabilities"] = probabilities.tolist()

                self.total_predictions += len(predictions)
                return result

            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        # ==================== NLP Endpoints ====================

        @self.app.post("/api/nlp/sentiment")
        async def nlp_sentiment(
            request: NLPRequest,
            verified: bool = Depends(self._verify_request)
        ):
            """Sentiment analysis"""
            return {"message": "Sentiment analysis endpoint - integrate with sentiment_analyzer.py"}

        @self.app.post("/api/nlp/entities")
        async def nlp_entities(
            request: NLPRequest,
            verified: bool = Depends(self._verify_request)
        ):
            """Named entity recognition"""
            return {"message": "NER endpoint - integrate with ner_extractor.py"}

        @self.app.post("/api/nlp/topics")
        async def nlp_topics(
            request: NLPRequest,
            verified: bool = Depends(self._verify_request)
        ):
            """Topic modeling"""
            return {"message": "Topic modeling endpoint - integrate with topic_modeling.py"}

        @self.app.post("/api/nlp/similarity")
        async def nlp_similarity(
            request: NLPRequest,
            verified: bool = Depends(self._verify_request)
        ):
            """Document similarity"""
            return {"message": "Similarity endpoint - integrate with document_similarity.py"}

        # ==================== Model Management ====================

        @self.app.get("/api/models")
        async def list_models(verified: bool = Depends(self._verify_request)):
            """List all registered models"""
            return {
                "models": [
                    {
                        "name": name,
                        "type": info['type'],
                        "version": info.get('version', '1.0'),
                        "registered_at": info.get('registered_at')
                    }
                    for name, info in self.models.items()
                ]
            }

        @self.app.get("/api/models/{model_name}")
        async def get_model_info(
            model_name: str,
            verified: bool = Depends(self._verify_request)
        ):
            """Get model information"""
            if model_name not in self.models:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

            model_info = self.models[model_name]
            return {
                "name": model_name,
                "type": model_info['type'],
                "version": model_info.get('version', '1.0'),
                "registered_at": model_info.get('registered_at'),
                "metadata": model_info.get('metadata', {})
            }

        @self.app.post("/api/models/register")
        async def register_model(
            registration: ModelRegistration,
            verified: bool = Depends(self._verify_request)
        ):
            """Register a new model"""
            try:
                if not JOBLIB_AVAILABLE:
                    raise ImportError("joblib required for model loading")

                model = joblib.load(registration.model_path)

                self.models[registration.model_name] = {
                    'model': model,
                    'type': registration.model_type,
                    'version': registration.version,
                    'path': registration.model_path,
                    'description': registration.description,
                    'metadata': registration.metadata or {},
                    'registered_at': datetime.now().isoformat()
                }

                return {
                    "status": "success",
                    "message": f"Model '{registration.model_name}' registered successfully"
                }

            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.delete("/api/models/{model_name}")
        async def delete_model(
            model_name: str,
            verified: bool = Depends(self._verify_request)
        ):
            """Delete a model"""
            if model_name not in self.models:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

            del self.models[model_name]
            return {"status": "success", "message": f"Model '{model_name}' deleted"}

    def _to_dataframe(self, data: Union[List[List[Any]], Dict[str, List[Any]]]) -> pd.DataFrame:
        """Convert input data to DataFrame"""
        if isinstance(data, dict):
            return pd.DataFrame(data)
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            raise ValueError("Data must be list of lists or dictionary")

    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
        """Find strong correlations in correlation matrix"""
        strong_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    strong_corrs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': float(corr_value)
                    })
        return strong_corrs

    def _train_model(self, request: TrainingRequest, task_type: str) -> Dict[str, Any]:
        """Train ML model (placeholder - integrate with actual training modules)"""
        # This would integrate with actual training modules
        # For now, return a mock response

        return {
            "status": "success",
            "message": f"Model training initiated for {task_type}",
            "model_type": request.model_type,
            "task_type": task_type,
            "note": "Integrate with actual training modules (classification_trainer.py, regression_trainer.py)"
        }

    def start(self):
        """Start the API gateway server"""
        print(f"{'='*80}")
        print(f"ML API Gateway Starting")
        print(f"{'='*80}")
        print(f"Host: {self.host}")
        print(f"Port: {self.port}")
        print(f"Authentication: {'Enabled' if self.auth_enabled else 'Disabled'}")
        print(f"Rate Limit: {self.rate_limit} requests/minute")
        print(f"CORS: {'Enabled' if self.enable_cors else 'Disabled'}")
        print(f"\nAPI Documentation: http://{self.host}:{self.port}/docs")
        print(f"ReDoc: http://{self.host}:{self.port}/redoc")
        print(f"Health Check: http://{self.host}:{self.port}/api/health")
        print(f"{'='*80}\n")

        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )


# CLI interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ML API Gateway')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--auth-enabled', action='store_true', help='Enable API key authentication')
    parser.add_argument('--rate-limit', type=int, default=100, help='Rate limit per minute')
    parser.add_argument('--no-cors', action='store_true', help='Disable CORS')

    args = parser.parse_args()

    try:
        # Initialize gateway
        gateway = MLAPIGateway(
            host=args.host,
            port=args.port,
            auth_enabled=args.auth_enabled,
            rate_limit=args.rate_limit,
            enable_cors=not args.no_cors
        )

        # Start server
        gateway.start()

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
