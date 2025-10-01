# Phase 10: MLOps & Advanced NLP - Usage Guide

**Quick Start Guide for Phase 10 Features**

---

## 🚀 Quick Start

### Installation

```bash
# Install Phase 10 dependencies
pip install -r python/requirements.txt

# Optional: Install specific feature sets
pip install fastapi uvicorn mlflow  # MLOps
pip install gensim spacy sentence-transformers  # NLP
```

---

## 📊 1. MLOps: Experiment Tracking (MLflow)

### Track ML Experiments

```python
from python.ml.mlops.mlflow_tracker import MLflowTracker

# Initialize tracker
tracker = MLflowTracker(experiment_name='customer_churn')

# Start a run
with tracker.start_run(run_name='random_forest_v1'):
    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Log parameters
    tracker.log_params({
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5
    })

    # Log metrics
    tracker.log_metrics({
        'accuracy': 0.92,
        'f1_score': 0.89,
        'precision': 0.91,
        'recall': 0.87
    })

    # Log model
    tracker.log_model(model, 'random_forest_model')

    # Log artifacts
    tracker.log_artifact('feature_importance.png')

# Register model for production
tracker.register_model(
    'runs:/<run_id>/random_forest_model',
    'ChurnPredictor',
    stage='Production',
    description='Production churn prediction model'
)

# Load production model
production_model = tracker.load_model('models:/ChurnPredictor/Production')
```

### MCP Tool Usage

```javascript
// Use through MCP
await mcp.call('mlops_experiment_track', {
  action: 'start_run',
  experiment_name: 'customer_churn',
  run_name: 'random_forest_v1'
});

await mcp.call('mlops_experiment_track', {
  action: 'log_params',
  experiment_name: 'customer_churn',
  params: { n_estimators: 100, max_depth: 10 }
});

await mcp.call('mlops_experiment_track', {
  action: 'get_best_run',
  experiment_name: 'customer_churn',
  metric: 'accuracy'
});
```

### CLI Usage

```bash
# Start MLflow UI
python -m python.ml.mlops.mlflow_tracker ui --port 5000

# List experiments
python -m python.ml.mlops.mlflow_tracker list-experiments

# Compare runs
python -m python.ml.mlops.mlflow_tracker compare --experiment customer_churn --run-ids run1 run2
```

---

## 🔧 2. MLOps: Model Serving

### Serve Models via REST API

```python
from python.ml.deployment.model_server import ModelServer

# Initialize server
server = ModelServer(host='0.0.0.0', port=8000)

# Register models
server.register_model(
    model_name='fraud_detector',
    model_path='models/fraud_detector.pkl',
    model_type='classifier',
    version='1.0',
    description='Fraud detection model'
)

server.register_model(
    model_name='sales_forecast',
    model_path='models/sales_prophet.pkl',
    model_type='forecaster'
)

# Start serving
server.start()
```

### Make Predictions

```python
import requests

# Single prediction
response = requests.post(
    'http://localhost:8000/predict/fraud_detector',
    json={
        'features': [[1.5, 2.3, 0.8, 4.2]],
        'return_probabilities': True
    }
)

result = response.json()
print(f"Prediction: {result['predictions']}")
print(f"Probability: {result['probabilities']}")
print(f"Latency: {result['latency_ms']}ms")

# Batch prediction
response = requests.post(
    'http://localhost:8000/predict/batch/fraud_detector',
    json={
        'features': [[1.5, 2.3, 0.8, 4.2], [2.1, 3.4, 1.2, 5.1]],
        'batch_size': 32
    }
)

# List models
response = requests.get('http://localhost:8000/models')
models = response.json()

# Health check
response = requests.get('http://localhost:8000/health')
status = response.json()
```

### MCP Tool Usage

```javascript
// Register model
await mcp.call('mlops_model_serve', {
  action: 'register',
  model_name: 'fraud_detector',
  model_path: 'models/fraud_detector.pkl',
  model_type: 'classifier'
});

// Make prediction
await mcp.call('mlops_model_serve', {
  action: 'predict',
  model_name: 'fraud_detector',
  features: [[1.5, 2.3, 0.8, 4.2]]
});

// List models
await mcp.call('mlops_model_serve', {
  action: 'list_models'
});
```

---

## 📈 3. MLOps: Model Monitoring

### Monitor Production Models

```python
from python.ml.mlops.model_monitor import ModelMonitor
import pandas as pd

# Initialize monitor
monitor = ModelMonitor(
    model_name='fraud_detector',
    monitoring_window=1000,
    drift_threshold=0.1
)

# Set reference data (training data)
monitor.set_reference_data(X_train)

# Log predictions in production
for X_batch in production_stream:
    predictions = model.predict(X_batch)
    latency = measure_latency()

    monitor.log_prediction(
        input_features=X_batch,
        prediction=predictions,
        latency_ms=latency
    )

# Check for drift periodically
if monitor.total_predictions % 1000 == 0:
    drift_report = monitor.check_drift(current_data=X_recent)

    if drift_report['drift_detected']:
        print(f"⚠️ Drift detected!")
        print(f"Drifted features: {drift_report['drifted_features']}")

        # Trigger retraining
        retraining = monitor.trigger_retraining()
        print(retraining['message'])

# Get monitoring metrics
metrics_24h = monitor.get_metrics(period='24h')
print(f"Latency (p95): {metrics_24h['latency']['p95_ms']}ms")
print(f"Throughput: {metrics_24h['throughput']['predictions_per_hour']}")

# Generate report
report = monitor.generate_report(output_path='monitoring_report.json')
```

### MCP Tool Usage

```javascript
// Check for drift
await mcp.call('mlops_model_monitor', {
  action: 'check_drift',
  model_name: 'fraud_detector',
  reference_data_path: 'data/train.csv',
  current_data_path: 'data/production_recent.csv'
});

// Get metrics
await mcp.call('mlops_model_monitor', {
  action: 'get_metrics',
  model_name: 'fraud_detector',
  period: '24h'
});

// Generate report
await mcp.call('mlops_model_monitor', {
  action: 'generate_report',
  model_name: 'fraud_detector',
  output_path: 'monitoring_report.json'
});
```

---

## 📝 4. Advanced NLP: Topic Modeling

### Discover Topics in Documents

```python
from python.ml.nlp.topic_modeling import TopicModeler
import pandas as pd

# Load documents
df = pd.read_csv('customer_reviews.csv')
documents = df['review_text'].tolist()

# LDA Topic Modeling
modeler = TopicModeler(method='lda', n_topics=10)
result = modeler.fit(documents)

# Display topics
modeler.print_topics(top_n_words=10)

# Output:
# Topic 0: quality, product, excellent, great, amazing, perfect, love, best, happy, recommend
# Topic 1: service, customer, support, help, response, issue, problem, slow, disappointed, refund

# Get document topics
doc_topics = modeler.get_document_topics(documents)

# Visualize topics
modeler.visualize_topics(output_path='topics.png')
modeler.visualize_interactive(output_path='topics.html')

# NMF (faster, sparse)
nmf_modeler = TopicModeler(method='nmf', n_topics=10)
nmf_result = nmf_modeler.fit(documents)

# BERTopic (best accuracy, needs GPU)
bert_modeler = TopicModeler(method='bertopic', n_topics=10)
bert_result = bert_modeler.fit(documents)
```

### MCP Tool Usage

```javascript
await mcp.call('nlp_topic_modeling', {
  data_path: 'customer_reviews.csv',
  text_column: 'review_text',
  method: 'lda',
  n_topics: 10,
  visualize: true,
  output_dir: 'results/topics'
});
```

### CLI Usage

```bash
python -m python.ml.nlp.topic_modeling \
  --input customer_reviews.csv \
  --column review_text \
  --method lda \
  --n-topics 10 \
  --visualize \
  --output results/topics
```

---

## 🏷️ 5. Advanced NLP: Named Entity Recognition

### Extract Entities from Text

```python
from python.ml.nlp.ner_extractor import NERExtractor

# Initialize extractor (SpaCy)
extractor = NERExtractor(model='en_core_web_lg', backend='spacy')

# Extract entities from single text
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976."
entities = extractor.extract(text)

for entity in entities['entities']:
    print(f"{entity['text']}: {entity['label']} (confidence: {entity['score']:.2f})")

# Output:
# Apple Inc.: ORG (confidence: 0.95)
# Steve Jobs: PERSON (confidence: 0.98)
# Cupertino: GPE (confidence: 0.92)
# California: GPE (confidence: 0.94)
# 1976: DATE (confidence: 0.89)

# Batch processing
documents = df['article_text'].tolist()
results = extractor.extract_batch(documents, batch_size=32)

# Entity frequencies
frequencies = extractor.get_entity_frequencies(documents)
print("\nTop 5 organizations:")
for entity, count in frequencies['entity_freq']['ORG'][:5]:
    print(f"  {entity}: {count}")

# Visualize entities
extractor.visualize_entities(text, output_path='entities.html')

# Transformer-based NER (higher accuracy)
transformer_extractor = NERExtractor(
    model='dslim/bert-base-NER',
    backend='transformers'
)
entities_transformer = transformer_extractor.extract(text)
```

### MCP Tool Usage

```javascript
await mcp.call('nlp_entity_extraction', {
  data_path: 'articles.csv',
  text_column: 'article_text',
  model: 'en_core_web_lg',
  backend: 'spacy',
  entity_types: ['PERSON', 'ORG', 'GPE'],
  visualize: true,
  output_dir: 'results/entities'
});
```

---

## 🔍 6. Advanced NLP: Document Similarity

### Find Similar Documents

```python
from python.ml.nlp.document_similarity import DocumentSimilarity

# Load documents
df = pd.read_csv('documents.csv')
documents = df['text'].tolist()

# TF-IDF similarity (fast)
similarity = DocumentSimilarity(method='tfidf')
similarity.fit(documents)

# Find similar documents
query_doc = "Machine learning and artificial intelligence"
similar = similarity.find_similar(query_doc, top_k=5)

for doc in similar['similar_documents']:
    print(f"Doc {doc['index']}: similarity={doc['similarity']:.3f}")
    print(f"  {doc['text'][:100]}...")

# Semantic search
results = similarity.semantic_search(
    queries=['machine learning', 'data science', 'neural networks'],
    top_k=10
)

# Find duplicates
duplicates = similarity.find_duplicates(threshold=0.85)
print(f"Found {duplicates['total_duplicates']} duplicate pairs")

# Cluster documents
clusters = similarity.cluster_similar_documents(n_clusters=10)

# Visualize similarity matrix
similarity.visualize_similarity_matrix(
    sample_size=100,
    output_path='similarity_heatmap.png'
)

# BERT embeddings (semantic similarity)
bert_similarity = DocumentSimilarity(method='bert')
bert_similarity.fit(documents)
bert_similar = bert_similarity.find_similar(query_doc, top_k=5)
```

### MCP Tool Usage

```javascript
// Find similar documents
await mcp.call('nlp_document_similarity', {
  action: 'find_similar',
  data_path: 'documents.csv',
  text_column: 'text',
  method: 'tfidf',
  query: 'machine learning applications',
  top_k: 5
});

// Find duplicates
await mcp.call('nlp_document_similarity', {
  action: 'find_duplicates',
  data_path: 'documents.csv',
  text_column: 'text',
  threshold: 0.85
});

// Semantic search
await mcp.call('nlp_document_similarity', {
  action: 'semantic_search',
  data_path: 'documents.csv',
  text_column: 'text',
  queries: ['AI', 'machine learning', 'deep learning'],
  top_k: 10
});

// Cluster documents
await mcp.call('nlp_document_similarity', {
  action: 'cluster',
  data_path: 'documents.csv',
  text_column: 'text',
  n_clusters: 10,
  visualize: true
});
```

---

## 🌐 7. API Gateway

### Unified ML API

```python
from python.ml.api.gateway import APIGateway

# Initialize gateway
gateway = APIGateway(
    host='0.0.0.0',
    port=8080,
    enable_auth=True,
    api_key='your-secret-key',
    rate_limit_per_minute=100
)

# Register models (optional, can also use endpoints)
gateway.register_model('fraud_detector', 'models/fraud.pkl', 'classifier')

# Start server
gateway.start()
```

### API Endpoints

```bash
# API Documentation
http://localhost:8080/docs  # Swagger UI
http://localhost:8080/redoc # ReDoc

# Health & Status
GET http://localhost:8080/api/health
GET http://localhost:8080/api/status
GET http://localhost:8080/api/metrics

# Training
POST http://localhost:8080/api/train/classifier
POST http://localhost:8080/api/train/regressor

# Prediction
POST http://localhost:8080/api/predict/{model_name}
POST http://localhost:8080/api/predict/batch/{model_name}

# NLP
POST http://localhost:8080/api/nlp/sentiment
POST http://localhost:8080/api/nlp/entities
POST http://localhost:8080/api/nlp/topics
POST http://localhost:8080/api/nlp/similarity

# Model Management
GET  http://localhost:8080/api/models
GET  http://localhost:8080/api/models/{model_name}
POST http://localhost:8080/api/models/register
DEL  http://localhost:8080/api/models/{model_name}
```

### Client Usage

```python
import requests

headers = {'X-API-Key': 'your-secret-key'}

# Train a classifier
response = requests.post(
    'http://localhost:8080/api/train/classifier',
    json={
        'data': {
            'features': X_train.tolist(),
            'labels': y_train.tolist()
        },
        'model_type': 'random_forest',
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 10
        }
    },
    headers=headers
)

model_id = response.json()['model_id']

# Make predictions
response = requests.post(
    f'http://localhost:8080/api/predict/{model_id}',
    json={'features': [[1, 2, 3, 4]]},
    headers=headers
)

# NLP: Sentiment analysis
response = requests.post(
    'http://localhost:8080/api/nlp/sentiment',
    json={'texts': ['This is great!', 'This is terrible.']},
    headers=headers
)

# NLP: Topic modeling
response = requests.post(
    'http://localhost:8080/api/nlp/topics',
    json={
        'documents': documents,
        'n_topics': 10,
        'method': 'lda'
    },
    headers=headers
)
```

---

## 🎯 Complete MLOps Pipeline Example

```python
from python.ml.mlops.mlflow_tracker import MLflowTracker
from python.ml.deployment.model_server import ModelServer
from python.ml.mlops.model_monitor import ModelMonitor
from sklearn.ensemble import RandomForestClassifier

# 1. Train and track with MLflow
tracker = MLflowTracker(experiment_name='fraud_detection')

with tracker.start_run(run_name='rf_v1'):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    tracker.log_params({'n_estimators': 100})
    tracker.log_metrics({'accuracy': 0.95})
    tracker.log_model(model, 'random_forest')

    tracker.register_model(
        'runs:/<run_id>/random_forest',
        'FraudDetector',
        stage='Production'
    )

# 2. Serve model via API
server = ModelServer(port=8000)
server.register_model(
    'fraud_detector',
    'models/fraud_detector.pkl',
    'classifier'
)
# server.start()  # Run in separate process

# 3. Monitor in production
monitor = ModelMonitor('fraud_detector')
monitor.set_reference_data(X_train)

for X_batch in production_stream:
    predictions = model.predict(X_batch)
    monitor.log_prediction(X_batch, predictions, latency_ms=45)

    if monitor.total_predictions % 1000 == 0:
        drift = monitor.check_drift(current_data=X_recent)
        if drift['drift_detected']:
            monitor.trigger_retraining()
```

---

## 📚 Additional Resources

- **API Documentation**: http://localhost:8080/docs (when API Gateway is running)
- **MLflow UI**: http://localhost:5000 (start with `python -m python.ml.mlops.mlflow_tracker ui`)
- **Phase 10 Plan**: `docs/PHASE_10_PLAN.md`
- **Phase 10 Summary**: `progress/phase_10_completion_summary.md`
- **Requirements**: `python/requirements.txt`

---

## 🔧 Troubleshooting

### MLflow Issues
```bash
# Reset MLflow tracking
rm -rf mlruns/
mlflow experiments delete --experiment-name <name>
```

### Model Serving Issues
```bash
# Check if port is available
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Linux/Mac
```

### NLP Dependencies
```bash
# Install SpaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg

# Install NLTK data
python -m nltk.downloader punkt stopwords wordnet
```

---

*Created: October 1, 2025*
*Version: 1.0.0*
*Phase: 10 Complete*
