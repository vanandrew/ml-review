import { Topic } from '../../../types';

export const modelDeployment: Topic = {
  id: 'model-deployment',
  title: 'Model Deployment',
  category: 'ml-systems',
  description: 'Strategies and best practices for deploying ML models to production',
  content: `
    <h2>Model Deployment: From Notebook to Production</h2>
    
    <p>You've trained a model achieving 95% accuracy on your test set. Congratulations! Now what? The model sitting in your Jupyter notebook is worthless until it's deployed—integrated into a production system where it can actually make predictions for real users, generate business value, and justify the development investment. Model deployment bridges the gap between machine learning experiments and production systems, transforming research artifacts into reliable, scalable services.</p>

    <p>Deployment is where many ML projects fail. Models that work perfectly offline crash on edge cases, inference latency makes applications unusable, version mismatches cause silent failures, and technical debt accumulates as ad-hoc solutions proliferate. Successful deployment requires careful consideration of latency requirements, scalability needs, monitoring strategies, and failure modes. It's as much engineering as it is data science.</p>

    <h3>Deployment Patterns: Choosing Your Architecture</h3>

    <h4>Batch Prediction: Offline Intelligence</h4>

    <p><strong>Batch prediction</strong> processes large volumes of data offline, storing predictions for later retrieval. Every night at midnight, score all users for churn risk. Pre-compute recommendations for millions of customers. Generate next-day demand forecasts for inventory management. Predictions are made in bulk, stored in a database or cache, and served when needed.</p>

    <p><strong>When to use batch:</strong> Predictions don't need to be real-time (daily product recommendations), input data arrives in scheduled batches (nightly transaction logs), computationally expensive models that would be too slow for real-time (complex ensembles, deep learning models), or when simplicity outweighs immediacy.</p>

    <p><strong>Advantages</strong> are significant: you can use arbitrarily complex models since inference time doesn't directly impact user experience. Implementation is simpler—a cron job running inference on a database. Resource utilization is better—batch processing at off-peak hours. Debugging is easier—rerun failed batches, inspect intermediate outputs, iterate without user impact.</p>

    <p><strong>Examples:</strong> Netflix pre-computes recommendations overnight. Email providers batch-score emails for spam. Retail inventory systems generate overnight demand forecasts.</p>

    <h4>Online (Real-Time) Prediction: Interactive Intelligence</h4>

    <p><strong>Real-time prediction</strong> generates predictions on-demand in response to requests, typically with sub-100ms latency requirements. User submits credit card transaction—is it fraud? (must decide instantly to approve/decline). User searches for products—which to show? (search results must load immediately). User types a message—what's the sentiment? (feedback happens now).</p>

    <p><strong>When to use real-time:</strong> Immediate predictions needed (<100ms typically), user-facing applications where latency affects experience, dynamic inputs that can't be pre-computed (personalization based on current session), critical decisions requiring instant response (fraud detection, content moderation).</p>

    <p><strong>Considerations:</strong> Latency constraints drive model choice—complex ensembles may be too slow, requiring simpler models or optimizations. Scalability matters—need to handle traffic spikes (Black Friday, viral events). High availability is critical—downtime means user impact and lost revenue. Infrastructure complexity increases—load balancers, auto-scaling, monitoring, caching layers.</p>

    <p><strong>Examples:</strong> Credit card fraud detection (real-time transaction approval), ad serving (select ads in milliseconds), chatbots (instant response generation), ride-sharing pricing (dynamic surge pricing).</p>

    <h4>Edge Deployment: Bringing Intelligence to Devices</h4>

    <p><strong>Edge deployment</strong> places models directly on user devices (smartphones, IoT sensors, embedded systems) without requiring server communication. Your phone's face recognition works offline. Smart home devices process voice commands locally. Autonomous vehicles make split-second decisions without cloud latency.</p>

    <p><strong>Benefits:</strong> Zero network latency—predictions are instant. Works offline—no internet required. Better privacy—sensitive data never leaves the device (face images, voice recordings, location data). Lower operational costs—no servers to maintain, no data transfer charges. Reduced load on backend infrastructure.</p>

    <p><strong>Challenges:</strong> Limited computational resources—mobile devices have constrained CPU/GPU. Model size constraints—apps have size limits, models must be compressed (quantization, pruning, distillation). Model updating is complex—app store approval cycles, user update adoption. Battery consumption matters—inefficient models drain batteries. Hardware diversity—must work across device types (iPhone, Android, various chipsets).</p>

    <p><strong>Solutions:</strong> Model compression (quantization to int8, pruning redundant weights), knowledge distillation (train small model to mimic large one), framework support (TensorFlow Lite, Core ML, ONNX for mobile), on-device learning (federated learning, personalization without central server).</p>

    <h3>Deployment Technologies: Building the Infrastructure</h3>

    <h4>REST APIs: The Universal Interface</h4>

    <p>Most models are exposed through HTTP REST APIs using frameworks like FastAPI, Flask, or Django. Client sends HTTP POST request with input features; server returns prediction. Simple, language-agnostic, widely supported.</p>

    <p><strong>Best practices:</strong> Version endpoints (/v1/predict, /v2/predict) to allow backward compatibility. Implement rigorous input validation—check types, ranges, required fields, prevent injection attacks. Add authentication/authorization (API keys, OAuth, JWT tokens) to control access. Return structured error messages (status codes, detailed errors in JSON) for debugging. Include health check endpoints (/health, /ready) for load balancer integration.</p>

    <h4>Containerization: Reproducible Environments</h4>

    <p><strong>Docker containers</strong> package model, dependencies, preprocessing code, and serving infrastructure into a single deployable unit. The container that works on your laptop works identically in production—no more "works on my machine" failures.</p>

    <p><strong>Advantages:</strong> Reproducible environments eliminate dependency conflicts. Isolation from host system prevents interference. Easy scaling with Kubernetes orchestration—deploy hundreds of containers automatically. Version control for the entire stack—model v1.2 runs in container image v1.2, ensuring consistency.</p>

    <h4>Model Serving Frameworks: Production-Grade Infrastructure</h4>

    <p><strong>TensorFlow Serving</strong> provides high-performance serving specifically for TensorFlow models. Built-in model versioning allows multiple models served simultaneously. A/B testing support routes traffic to different model versions. Both gRPC (low latency, binary protocol) and REST APIs.</p>

    <p><strong>TorchServe</strong> is PyTorch's production serving framework. Multi-model serving on single endpoint. Automatic metrics collection and logging. Model management API for deployment/updates.</p>

    <p><strong>ONNX Runtime</strong> is framework-agnostic—convert models from any framework (PyTorch, TensorFlow, scikit-learn) to ONNX format for unified serving. Optimized inference with hardware acceleration (CPUs, GPUs, specialized accelerators). Cross-platform support (Linux, Windows, mobile, edge).</p>

    <h3>The Deployment Pipeline: From Development to Production</h3>

    <h4>1. Model Packaging: Bundling for Deployment</h4>

    <p>Serialize the trained model (pickle/joblib for sklearn, SavedModel for TensorFlow, TorchScript for PyTorch). Include the complete preprocessing pipeline—scalers, encoders, feature transformers—ensuring training and serving use identical transformations. Document input/output schemas rigorously (feature names, types, value ranges, nullable fields). Save metadata: model version, training date, performance metrics, hyperparameters, training data provenance.</p>

    <h4>2. Testing: Validating Before Launch</h4>

    <p><strong>Unit tests</strong> verify individual components—preprocessing functions return expected outputs, postprocessing logic handles edge cases. <strong>Integration tests</strong> validate the full prediction pipeline end-to-end—send sample inputs, verify outputs match expected predictions. <strong>Load tests</strong> ensure performance under expected traffic—use tools like JMeter or Locust to simulate thousands of concurrent requests, measure latency percentiles (p50, p95, p99), identify bottlenecks. <strong>Shadow mode</strong> runs the new model alongside the existing one without affecting users—compare predictions, identify discrepancies, build confidence before switching traffic.</p>

    <h4>3. Deployment Strategies: Minimizing Risk</h4>

    <p><strong>Blue-Green Deployment</strong> maintains two identical environments: blue (current production) and green (new version). Deploy new model to green environment, run smoke tests, then switch all traffic from blue to green instantly via load balancer. If issues arise, switch back to blue immediately—instant rollback with zero downtime.</p>

    <p><strong>Canary Deployment</strong> gradually routes traffic to the new model: start with 5% of users, monitor metrics (latency, error rate, prediction quality), if healthy increase to 25%, then 50%, then 100%. Each stage validates the model with a subset of users before full rollout. If problems detected at any stage, rollback affects only that traffic percentage.</p>

    <h4>4. Monitoring: Knowing What's Happening</h4>

    <p>Monitor at multiple levels: <strong>Performance metrics</strong>—latency (p50, p95, p99 percentiles), throughput (requests per second), error rates (4xx client errors, 5xx server errors). <strong>Model metrics</strong>—prediction distribution (are predictions reasonable?), confidence scores (is the model certain?), feature distributions (are inputs changing?). <strong>Business metrics</strong>—conversion rates, user satisfaction, revenue impact. <strong>Infrastructure</strong>—CPU, memory, GPU utilization, disk I/O, network bandwidth.</p>

    <h3>Production Challenges: What Can Go Wrong</h3>

    <h4>Model Versioning: Managing Multiple Models</h4>

    <p>In production, you often run multiple model versions simultaneously—old model serves most traffic, new model in canary. Versions must coexist without conflicts. <strong>Solutions:</strong> Use semantic versioning (v1.2.3 where major.minor.patch indicates compatibility). Store models in artifact repositories (MLflow, DVC, S3) with complete lineage (training data, code commit, hyperparameters). Implement graceful model switching—load new model, warm up, switch traffic, keep old model ready for rollback.</p>

    <h4>Training-Serving Skew: The Consistency Problem</h4>

    <p>The deadliest bug: features computed differently in training versus production. Training calculates user's average purchase price from SQL query. Production code computes it differently, introducing subtle bugs. Predictions in production deviate from offline expectations despite identical model.</p>

    <p><strong>Feature stores</strong> solve this by centralizing feature computation, acting as the single source of truth for features across training and serving. A feature store provides: (1) <strong>Feature definitions</strong>—centralized code defining how features are computed, ensuring training and serving use identical logic. (2) <strong>Dual serving modes</strong>—offline serving for training (batch, high throughput) and online serving for inference (real-time, low latency). (3) <strong>Feature versioning</strong>—track feature definitions over time, reproduce historical training data. (4) <strong>Feature discovery</strong>—catalog of available features, metadata, and statistics helps teams reuse features across projects. (5) <strong>Point-in-time correctness</strong>—retrieve feature values as they existed at specific timestamps, preventing label leakage in time series data.</p>

    <p><strong>Popular feature store platforms:</strong> <em>Feast (open-source)</em>—lightweight, Kubernetes-native, supports Redis/DynamoDB for online serving. Good for getting started. <em>Tecton (enterprise)</em>—fully managed, real-time feature computation, sophisticated monitoring. Production-grade for large organizations. <em>AWS Feature Store</em>—integrated with SageMaker, automatic feature group creation, built-in monitoring. Best for AWS-heavy stacks. <em>Google Vertex AI Feature Store</em>—managed service, integrated with BigQuery, automatic online serving. Ideal for GCP users. <em>Hopsworks</em>—open-source option with enterprise features, Python-centric API.</p>

    <p><strong>When to adopt a feature store:</strong> Multiple models sharing features (recommendation and ranking models both use user embeddings). Training-serving skew causing production issues. Team size growing and feature reuse becoming important. Real-time features needed (streaming aggregations, last-hour activity). The overhead of setting up a feature store pays off when you have >3-5 models in production or >5 data scientists.</p>

    <h4>Dependency Hell: Version Conflicts</h4>

    <p>Model trained with sklearn 1.0 serialized, production server has sklearn 1.2, model loads but predictions differ silently. NumPy version differences cause numerical precision changes. Library updates break backward compatibility.</p>

    <p><strong>Solutions:</strong> Pin exact dependency versions in requirements.txt (not >=, exactly ==). Use the same containerized environment for training and serving. Test model serialization/deserialization rigorously across versions. Version the entire stack together—model v1.2 requires container v1.2 with exact dependencies.</p>

    <h3>Security: Protecting Models and Data</h3>

    <p><strong>Input validation</strong> is critical—sanitize all inputs, validate types and ranges, prevent injection attacks (SQL injection in feature lookups, code injection in eval statements). <strong>Rate limiting</strong> prevents abuse—limit requests per user/IP to stop DDoS attacks and model extraction attempts. <strong>Authentication/authorization</strong> controls access—API keys, OAuth flows, JWT tokens, role-based permissions. <strong>Model extraction attacks</strong> query models repeatedly to reconstruct training data or steal the model—limit query rates, add noise to predictions, monitor for suspicious patterns. <strong>Data privacy</strong> requires encryption in transit (TLS/HTTPS) and at rest, PII handling compliant with GDPR/CCPA, anonymization of logs.</p>

    <h3>Pre-Deployment Checklist: Ensure Production Readiness</h3>

    <p><strong>Before deploying any ML model to production, verify:</strong></p>

    <p><strong>Model Quality ✓</strong></p>
    <ul>
      <li>Model meets offline performance requirements on holdout test set</li>
      <li>Model tested on edge cases and adversarial inputs</li>
      <li>Model fairness evaluated across demographic groups</li>
      <li>Performance validated on recent data (not stale test set)</li>
    </ul>

    <p><strong>Infrastructure ✓</strong></p>
    <ul>
      <li>Model packaged with all dependencies (Docker container recommended)</li>
      <li>Preprocessing pipeline identical to training (no training-serving skew)</li>
      <li>Inference latency meets SLA requirements (P95, P99 measured)</li>
      <li>Load testing completed at expected peak traffic (+ 50% buffer)</li>
      <li>Auto-scaling configured and tested</li>
    </ul>

    <p><strong>Monitoring ✓</strong></p>
    <ul>
      <li>Prediction logging enabled (with appropriate PII handling)</li>
      <li>Performance metrics dashboards created (latency, throughput, errors)</li>
      <li>Model quality metrics tracked (accuracy, precision, recall)</li>
      <li>Alerts configured for degradation thresholds</li>
      <li>On-call rotation established for production incidents</li>
    </ul>

    <p><strong>Safety ✓</strong></p>
    <ul>
      <li>Rollback procedure documented and tested</li>
      <li>Canary/blue-green deployment strategy implemented</li>
      <li>Circuit breakers configured for cascading failures</li>
      <li>Rate limiting and authentication enabled</li>
      <li>Shadow mode testing completed (if possible)</li>
    </ul>

    <p><strong>Documentation ✓</strong></p>
    <ul>
      <li>Model card created (architecture, training data, performance, limitations)</li>
      <li>API documentation published (input/output schemas, examples)</li>
      <li>Runbook created (common issues, debugging steps, escalation)</li>
      <li>Retraining procedures documented</li>
    </ul>

    <p><strong>Compliance & Ethics ✓</strong></p>
    <ul>
      <li>Privacy review completed (GDPR, CCPA compliance)</li>
      <li>Bias and fairness analysis documented</li>
      <li>Model interpretability/explainability available if required</li>
      <li>Legal and compliance teams approved (if regulated industry)</li>
    </ul>

    <p>This checklist prevents the most common deployment failures. Skip items at your peril—production issues are expensive in lost revenue, user trust, and team morale. Better to delay deployment by days than to cause outages that take weeks to resolve.</p>

    <p>Deployment transforms ML from research to reality. Choose deployment patterns based on latency and scale requirements. Build robust pipelines with packaging, testing, and gradual rollout strategies. Monitor continuously at all levels. Address versioning, consistency, and security proactively. Master these, and your models will thrive in production, delivering value reliably and at scale.</p>
  `,
  codeExamples: [
    {
      language: 'Python',
      code: `from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import joblib
import numpy as np
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and preprocessing pipeline
model = joblib.load('model.joblib')
preprocessor = joblib.load('preprocessor.joblib')

# Define API
app = FastAPI(title="ML Model API", version="1.0.0")

# Input/output schemas
class PredictionInput(BaseModel):
  features: List[float]

  @validator('features')
  def validate_features(cls, v):
      if len(v) != 10:  # Expected number of features
          raise ValueError('Expected 10 features')
      if any(np.isnan(v) or np.isinf(v)):
          raise ValueError('Features contain NaN or Inf')
      return v

class PredictionOutput(BaseModel):
  prediction: float
  confidence: float
  model_version: str

# Health check endpoint
@app.get("/health")
async def health_check():
  return {"status": "healthy", "model_loaded": model is not None}

# Prediction endpoint
@app.post("/v1/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
  try:
      # Preprocess input
      features = np.array(input_data.features).reshape(1, -1)
      features_processed = preprocessor.transform(features)

      # Make prediction
      prediction = model.predict(features_processed)[0]

      # Get confidence (for probabilistic models)
      if hasattr(model, 'predict_proba'):
          probabilities = model.predict_proba(features_processed)[0]
          confidence = float(max(probabilities))
      else:
          confidence = None

      # Log request (for monitoring)
      logger.info(f"Prediction made: {prediction}, confidence: {confidence}")

      return PredictionOutput(
          prediction=float(prediction),
          confidence=confidence,
          model_version="1.0.0"
      )

  except Exception as e:
      logger.error(f"Prediction error: {str(e)}")
      raise HTTPException(status_code=500, detail=str(e))

# Batch prediction endpoint
@app.post("/v1/predict/batch")
async def predict_batch(inputs: List[PredictionInput]):
  try:
      features = np.array([inp.features for inp in inputs])
      features_processed = preprocessor.transform(features)
      predictions = model.predict(features_processed)

      return {
          "predictions": predictions.tolist(),
          "count": len(predictions)
      }
  except Exception as e:
      logger.error(f"Batch prediction error: {str(e)}")
      raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000`,
      explanation: 'Complete FastAPI deployment with input validation, error handling, health checks, and both single and batch prediction endpoints. Includes proper logging for monitoring and structured error responses.'
    },
    {
      language: 'Python',
      code: `# Dockerfile for containerized deployment
"""
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts and code
COPY model.joblib preprocessor.joblib ./
COPY app.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
"""

# Docker Compose for multi-container setup with monitoring
"""
version: '3.8'

services:
  ml-api:
  build: .
  ports:
    - "8000:8000"
  environment:
    - MODEL_VERSION=1.0.0
    - LOG_LEVEL=INFO
  volumes:
    - ./logs:/app/logs
  deploy:
    replicas: 3
    resources:
      limits:
        cpus: '2'
        memory: 2G
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
    interval: 30s
    timeout: 10s
    retries: 3

  redis:
  image: redis:alpine
  ports:
    - "6379:6379"

  prometheus:
  image: prom/prometheus
  ports:
    - "9090:9090"
  volumes:
    - ./prometheus.yml:/etc/prometheus/prometheus.yml
"""

# Model deployment script with versioning
import joblib
import mlflow
from datetime import datetime
import os

class ModelDeployer:
  def __init__(self, registry_path='./model_registry'):
      self.registry_path = registry_path
      os.makedirs(registry_path, exist_ok=True)

  def package_model(self, model, preprocessor, metadata):
      """Package model with all necessary components."""
      version = metadata['version']
      timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
      model_dir = f"{self.registry_path}/v{version}_{timestamp}"
      os.makedirs(model_dir, exist_ok=True)

      # Save model and preprocessor
      joblib.dump(model, f"{model_dir}/model.joblib")
      joblib.dump(preprocessor, f"{model_dir}/preprocessor.joblib")

      # Save metadata
      import json
      with open(f"{model_dir}/metadata.json", 'w') as f:
          json.dump(metadata, f, indent=2)

      print(f"Model packaged: {model_dir}")
      return model_dir

  def deploy_canary(self, model_path, traffic_percent=10):
      """Deploy model with canary strategy."""
      print(f"Deploying canary with {traffic_percent}% traffic...")

      # In production, this would update load balancer rules
      # For example with Kubernetes:
      # kubectl apply -f canary-deployment.yaml
      # kubectl patch service ml-service -p '{"spec":{"selector":{"version":"canary"}}}'

      # Monitor metrics for specified duration
      import time
      monitor_duration = 300  # 5 minutes
      print(f"Monitoring canary for {monitor_duration}s...")
      time.sleep(monitor_duration)

      # Check metrics (simplified)
      metrics = self.check_canary_metrics()
      if metrics['error_rate'] < 0.01 and metrics['latency_p95'] < 100:
          print("Canary healthy, promoting to full deployment")
          return True
      else:
          print("Canary unhealthy, rolling back")
          return False

  def check_canary_metrics(self):
      """Check canary deployment metrics."""
      # In production, fetch from monitoring system (Prometheus, Datadog)
      return {
          'error_rate': 0.005,
          'latency_p95': 85,
          'throughput': 1200
      }

  def rollback(self, previous_version):
      """Rollback to previous model version."""
      print(f"Rolling back to version {previous_version}")
      # Update serving config to point to previous version
      # kubectl rollout undo deployment/ml-api

# Usage
metadata = {
  'version': '2.0.0',
  'training_date': '2024-01-15',
  'metrics': {'accuracy': 0.94, 'f1': 0.92},
  'features': ['feature1', 'feature2', 'feature3']
}

deployer = ModelDeployer()
model_dir = deployer.package_model(model, preprocessor, metadata)

# Deploy with canary strategy
success = deployer.deploy_canary(model_dir, traffic_percent=10)
if not success:
  deployer.rollback('1.0.0')`,
      explanation: 'Production deployment setup including Dockerfile for containerization, Docker Compose for multi-service orchestration, and a model deployer class with canary deployment strategy and rollback capability. Shows version management and health monitoring patterns.'
    }
  ],
  interviewQuestions: [
    {
      question: 'What are the key differences between batch and online inference, and when would you choose each?',
      answer: `Batch inference processes large volumes of data at scheduled intervals (e.g., daily recommendations), offering higher throughput and computational efficiency. Online inference serves real-time requests with low latency requirements (e.g., fraud detection). Choose batch for: non-urgent predictions, large datasets, cost optimization. Choose online for: real-time decisions, user-facing applications, time-sensitive predictions. Hybrid approaches can combine both based on use case requirements.`
    },
    {
      question: 'How do you handle model versioning in production? What happens when you need to roll back?',
      answer: `Model versioning involves tracking model artifacts, metadata, and dependencies with unique identifiers. Use MLOps tools (MLflow, Kubeflow) for version control. Implement blue-green deployments or canary releases for safe updates. For rollbacks: maintain previous model versions, automate rollback triggers based on performance metrics, ensure data compatibility, and have documented rollback procedures. Include model registry for centralized version management.`
    },
    {
      question: 'Explain training-serving skew. What causes it and how can you prevent it?',
      answer: `Training-serving skew occurs when data distributions differ between training and serving environments. Causes include: different preprocessing pipelines, data collection methods, temporal shifts, or feature computation differences. Prevention strategies: use identical preprocessing code, implement feature stores, validate data schemas, monitor input distributions, use consistent data sources, and implement integration tests comparing training and serving pipelines.`
    },
    {
      question: 'What strategies would you use to deploy a new model with minimal risk to production?',
      answer: `Risk mitigation strategies include: (1) Canary deployments - gradually increase traffic to new model, (2) A/B testing - compare new vs old model performance, (3) Shadow mode - run new model alongside old without affecting users, (4) Feature flags - quick enable/disable capabilities, (5) Extensive testing - unit, integration, load tests, (6) Monitoring - real-time metrics and alerts, (7) Rollback plans - automated reversion procedures.`
    },
    {
      question: 'How do you ensure your deployed model can handle the expected traffic load?',
      answer: `Load testing strategies: (1) Benchmark inference latency under various loads, (2) Use load testing tools (JMeter, Locust) to simulate traffic patterns, (3) Implement horizontal scaling with load balancers, (4) Set up auto-scaling based on metrics (CPU, memory, request count), (5) Monitor resource utilization, (6) Implement caching for frequent requests, (7) Use performance profiling to identify bottlenecks, (8) Plan capacity based on peak traffic projections.`
    },
    {
      question: 'What security considerations are important when deploying ML models as APIs?',
      answer: `Key security considerations: (1) Authentication/authorization - API keys, OAuth, role-based access, (2) Input validation - prevent injection attacks, validate data types/ranges, (3) Rate limiting - prevent abuse and DDoS, (4) Model protection - prevent model extraction/inversion attacks, (5) Data privacy - encryption in transit/rest, PII handling, (6) Logging/monitoring - audit trails, anomaly detection, (7) Network security - VPCs, firewalls, secure protocols.`
    }
  ],
  quizQuestions: [
    {
      id: 'deploy1',
      question: 'When is batch prediction preferred over real-time?',
      options: ['Never, real-time is always better', 'When predictions don\'t need to be immediate', 'Only for simple models', 'When you have unlimited compute'],
      correctAnswer: 1,
      explanation: 'Batch prediction is preferred when immediate predictions aren\'t required (e.g., daily recommendations), computationally expensive models, or when input data arrives in batches. It allows better resource utilization and simpler infrastructure.'
    },
    {
      id: 'deploy2',
      question: 'What is a canary deployment?',
      options: ['Deploying only on weekends', 'Gradually routing traffic to new model', 'Testing on birds', 'A/B testing with 50/50 split'],
      correctAnswer: 1,
      explanation: 'Canary deployment gradually routes a small percentage of traffic (e.g., 5%) to the new model while monitoring metrics. If successful, traffic is incrementally increased; if issues arise, it\'s easy to rollback.'
    },
    {
      id: 'deploy3',
      question: 'What causes training-serving skew?',
      options: ['Different hardware', 'Features computed differently in training vs production', 'Model overfitting', 'Poor data quality'],
      correctAnswer: 1,
      explanation: 'Training-serving skew occurs when features are computed differently during training (offline, batch) versus serving (online, real-time), leading to inconsistent predictions. Feature stores help solve this by centralizing feature computation.'
    }
  ]
};
