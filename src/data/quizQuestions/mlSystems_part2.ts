import { QuizQuestion } from '../../types';

// Model Deployment - 20 questions
export const modelDeploymentQuestions: QuizQuestion[] = [
  {
    id: 'md1',
    question: 'What is model deployment?',
    options: ['Training', 'Making trained model available for production use', 'Data collection', 'Feature engineering'],
    correctAnswer: 1,
    explanation: 'Deployment puts model into production environment to serve predictions to end users/systems.'
  },
  {
    id: 'md2',
    question: 'What are common deployment patterns?',
    options: ['Only one pattern', 'Batch inference, real-time API, edge deployment, streaming', 'No patterns', 'Training only'],
    correctAnswer: 1,
    explanation: 'Deployment patterns: batch (periodic predictions), online (low-latency API), edge (on-device), stream processing.'
  },
  {
    id: 'md3',
    question: 'What is batch inference?',
    options: ['Real-time', 'Processing large batches of data periodically (hourly, daily)', 'Single prediction', 'Streaming'],
    correctAnswer: 1,
    explanation: 'Batch inference: run predictions on accumulated data at scheduled intervals, not real-time.'
  },
  {
    id: 'md4',
    question: 'What is real-time inference?',
    options: ['Batch processing', 'Low-latency predictions on-demand (< 100ms typically)', 'Offline', 'Scheduled'],
    correctAnswer: 1,
    explanation: 'Real-time serving: instant predictions via API for live user interactions (recommendations, fraud detection).'
  },
  {
    id: 'md5',
    question: 'What is a REST API for ML?',
    options: ['Training interface', 'HTTP endpoint accepting inputs, returning predictions', 'No interface', 'Batch only'],
    correctAnswer: 1,
    explanation: 'REST API exposes model via HTTP: send features via POST, receive prediction response.'
  },
  {
    id: 'md6',
    question: 'What is model serialization?',
    options: ['Training', 'Saving trained model to disk for later loading', 'No saving', 'Deployment'],
    correctAnswer: 1,
    explanation: 'Serialization saves model state (pickle, joblib, ONNX, SavedModel) for deployment without retraining.'
  },
  {
    id: 'md7',
    question: 'What is containerization?',
    options: ['No packaging', 'Packaging model with dependencies in Docker container', 'Virtual machine', 'No isolation'],
    correctAnswer: 1,
    explanation: 'Containers (Docker) bundle model, code, libraries, ensuring consistent environment across deployments.'
  },
  {
    id: 'md8',
    question: 'What is model serving framework?',
    options: ['Training framework', 'Infrastructure for deploying and serving models: TensorFlow Serving, TorchServe, Seldon', 'No framework', 'Development only'],
    correctAnswer: 1,
    explanation: 'Serving frameworks handle API, batching, versioning: TF Serving, TorchServe, Triton, Seldon Core.'
  },
  {
    id: 'md9',
    question: 'What is A/B testing in deployment?',
    options: ['Single model', 'Comparing two model versions by routing traffic split', 'No testing', 'Training test'],
    correctAnswer: 1,
    explanation: 'A/B testing: serve model A to 50% users, model B to 50%, compare performance metrics.'
  },
  {
    id: 'md10',
    question: 'What is canary deployment?',
    options: ['All-at-once', 'Gradually rolling out new model to increasing % of traffic', 'No rollout', 'Instant switch'],
    correctAnswer: 1,
    explanation: 'Canary: deploy to 5% traffic, monitor, gradually increase to 100% if stable.'
  },
  {
    id: 'md11',
    question: 'What is blue-green deployment?',
    options: ['Single environment', 'Two identical environments: switch traffic from old (blue) to new (green)', 'No redundancy', 'One version'],
    correctAnswer: 1,
    explanation: 'Blue-green: maintain two environments, instant switch between versions, easy rollback.'
  },
  {
    id: 'md12',
    question: 'What is model versioning?',
    options: ['No versions', 'Tracking different model versions, enabling rollback and comparison', 'Single version', 'No tracking'],
    correctAnswer: 1,
    explanation: 'Versioning tracks models over time, enables serving multiple versions, rollback to previous.'
  },
  {
    id: 'md13',
    question: 'What is latency in serving?',
    options: ['Accuracy', 'Time from request to prediction response', 'Memory', 'No metric'],
    correctAnswer: 1,
    explanation: 'Latency measures response time; critical for real-time applications (target: <100ms).'
  },
  {
    id: 'md14',
    question: 'What is throughput?',
    options: ['Single request', 'Number of predictions per second system can handle', 'Latency', 'No metric'],
    correctAnswer: 1,
    explanation: 'Throughput measures capacity: predictions/second; optimize via batching, parallelism.'
  },
  {
    id: 'md15',
    question: 'What is batch prediction optimization?',
    options: ['One at a time', 'Processing multiple inputs together for efficiency', 'No batching', 'Single inference'],
    correctAnswer: 1,
    explanation: 'Batching amortizes overhead, improves GPU utilization, increases throughput significantly.'
  },
  {
    id: 'md16',
    question: 'What is edge deployment?',
    options: ['Cloud only', 'Running models on edge devices: phones, IoT, drones', 'Server only', 'No local'],
    correctAnswer: 1,
    explanation: 'Edge deployment: inference on device for low latency, offline capability, privacy.'
  },
  {
    id: 'md17',
    question: 'What challenges does edge deployment face?',
    options: ['No challenges', 'Limited compute, memory, power; requires model compression', 'Unlimited resources', 'Same as cloud'],
    correctAnswer: 1,
    explanation: 'Edge constraints: small models needed (quantization, pruning), battery-efficient, limited RAM.'
  },
  {
    id: 'md18',
    question: 'What is model monitoring in production?',
    options: ['No monitoring', 'Tracking predictions, performance, errors, drift in production', 'Training only', 'One-time check'],
    correctAnswer: 1,
    explanation: 'Production monitoring: log predictions, track metrics, detect anomalies, alert on degradation.'
  },
  {
    id: 'md19',
    question: 'What tools support deployment?',
    options: ['No tools', 'Kubernetes, Docker, AWS SageMaker, Azure ML, GCP Vertex AI, MLflow', 'Manual only', 'One tool'],
    correctAnswer: 1,
    explanation: 'Deployment tools: container orchestration (K8s), cloud platforms (SageMaker, Vertex), MLOps (MLflow, Kubeflow).'
  },
  {
    id: 'md20',
    question: 'What is shadow mode?',
    options: ['Direct deployment', 'Running new model in production but not using predictions (logging only)', 'No testing', 'Immediate use'],
    correctAnswer: 1,
    explanation: 'Shadow mode: new model makes predictions silently, compare with current model, no user impact.'
  }
];

// A/B Testing & Experimentation - 20 questions
export const abTestingQuestions: QuizQuestion[] = [
  {
    id: 'ab1',
    question: 'What is A/B testing?',
    options: ['Single version', 'Randomized experiment comparing two variants to measure impact', 'No testing', 'Sequential test'],
    correctAnswer: 1,
    explanation: 'A/B testing splits traffic randomly between control (A) and treatment (B), compares outcomes.'
  },
  {
    id: 'ab2',
    question: 'What is the control group?',
    options: ['New version', 'Baseline group (A) receiving existing/default experience', 'Treatment', 'No group'],
    correctAnswer: 1,
    explanation: 'Control (A) represents current state; treatment (B) is the variation being tested.'
  },
  {
    id: 'ab3',
    question: 'What is statistical significance?',
    options: ['Business impact', 'Probability that observed difference is not due to chance', 'Effect size', 'No meaning'],
    correctAnswer: 1,
    explanation: 'Statistical significance (p-value < 0.05) indicates results unlikely due to random variation.'
  },
  {
    id: 'ab4',
    question: 'What is a p-value?',
    options: ['Effect size', 'Probability of observing results if null hypothesis (no difference) is true', 'Mean difference', 'No metric'],
    correctAnswer: 1,
    explanation: 'P-value: probability of seeing results (or more extreme) under null hypothesis; p<0.05 typical threshold.'
  },
  {
    id: 'ab5',
    question: 'What is statistical power?',
    options: ['Significance', 'Probability of detecting true effect if it exists (1 - Type II error)', 'Sample size', 'No concept'],
    correctAnswer: 1,
    explanation: 'Power: ability to detect real effects; typically aim for 80%+ power, achieved via sufficient sample size.'
  },
  {
    id: 'ab6',
    question: 'What is minimum detectable effect (MDE)?',
    options: ['Any change', 'Smallest effect size experiment can reliably detect', 'Large change', 'No minimum'],
    correctAnswer: 1,
    explanation: 'MDE defines sensitivity: "can detect 2% conversion lift"; smaller MDE requires more samples.'
  },
  {
    id: 'ab7',
    question: 'How to determine sample size?',
    options: ['Random guess', 'Power analysis based on MDE, significance level, baseline rate, power', 'Trial and error', 'No calculation'],
    correctAnswer: 1,
    explanation: 'Sample size calculation balances: desired power (80%), significance (0.05), MDE, baseline metric.'
  },
  {
    id: 'ab8',
    question: 'What is Type I error?',
    options: ['Missing effect', 'False positive: concluding difference exists when it doesn\'t', 'Correct decision', 'True positive'],
    correctAnswer: 1,
    explanation: 'Type I error (α): falsely rejecting null hypothesis (saying B is better when it\'s not); controlled by p-value threshold.'
  },
  {
    id: 'ab9',
    question: 'What is Type II error?',
    options: ['False positive', 'False negative: failing to detect real effect', 'Correct decision', 'True negative'],
    correctAnswer: 1,
    explanation: 'Type II error (β): failing to detect true effect; power = 1 - β.'
  },
  {
    id: 'ab10',
    question: 'What is randomization importance?',
    options: ['Not important', 'Ensures unbiased group assignment, eliminating confounds', 'Sequential better', 'No need'],
    correctAnswer: 1,
    explanation: 'Randomization balances known and unknown confounders across groups, enabling causal inference.'
  },
  {
    id: 'ab11',
    question: 'What is the problem with peeking?',
    options: ['No problem', 'Checking results early inflates Type I error (false positives)', 'Recommended', 'Speeds up testing'],
    correctAnswer: 1,
    explanation: 'Peeking at interim results increases false positive rate; wait until planned sample size reached.'
  },
  {
    id: 'ab12',
    question: 'What is multiple testing problem?',
    options: ['No issue', 'Running many tests increases false positive rate; need correction', 'Better approach', 'No correction'],
    correctAnswer: 1,
    explanation: 'Testing multiple metrics/variants inflates Type I error; use Bonferroni correction or control FDR.'
  },
  {
    id: 'ab13',
    question: 'What is Bonferroni correction?',
    options: ['No correction', 'Adjusting significance threshold: α_new = α / n_tests', 'Increasing α', 'No change'],
    correctAnswer: 1,
    explanation: 'Bonferroni: divide α by number of tests (0.05/10 = 0.005); conservative but controls family-wise error.'
  },
  {
    id: 'ab14',
    question: 'What is a guardrail metric?',
    options: ['Primary metric', 'Secondary metric ensuring treatment doesn\'t harm other dimensions', 'No protection', 'Ignored metric'],
    correctAnswer: 1,
    explanation: 'Guardrails protect key metrics: can\'t ship if improves conversion but tanks revenue/latency.'
  },
  {
    id: 'ab15',
    question: 'What is novelty effect?',
    options: ['Permanent change', 'Temporary behavior change due to newness wearing off', 'Stable effect', 'No effect'],
    correctAnswer: 1,
    explanation: 'Novelty effect: users react to change initially; true long-term effect may differ. Run longer tests.'
  },
  {
    id: 'ab16',
    question: 'What is selection bias in experiments?',
    options: ['No bias', 'Non-random differences between groups due to poor randomization', 'Perfect randomization', 'Intended'],
    correctAnswer: 1,
    explanation: 'Selection bias: groups differ systematically (e.g., giving treatment to all weekday users).'
  },
  {
    id: 'ab17',
    question: 'What is network effects problem?',
    options: ['No problem', 'User interactions violate independence assumption (SUTVA)', 'Independence holds', 'No interactions'],
    correctAnswer: 1,
    explanation: 'Network effects: one user\'s treatment affects others (social networks); breaks standard A/B testing assumptions.'
  },
  {
    id: 'ab18',
    question: 'What is multivariate testing?',
    options: ['Two variants', 'Testing combinations of multiple changes simultaneously', 'A/B only', 'Single change'],
    correctAnswer: 1,
    explanation: 'Multivariate: test multiple factors together (e.g., headline × image × button color); finds interactions.'
  },
  {
    id: 'ab19',
    question: 'What is bandit algorithm alternative?',
    options: ['Fixed allocation', 'Adaptive allocation: shift traffic to better-performing variant during test', 'No adaptation', 'Random'],
    correctAnswer: 1,
    explanation: 'Multi-armed bandits balance exploration-exploitation, dynamically favoring better variants (less regret than A/B).'
  },
  {
    id: 'ab20',
    question: 'What tools support A/B testing?',
    options: ['Manual only', 'Optimizely, Google Optimize, Statsig, LaunchDarkly, custom platforms', 'No tools', 'One tool'],
    correctAnswer: 1,
    explanation: 'A/B testing platforms: Optimizely, VWO, Statsig (feature flags + experiments), in-house systems.'
  }
];

// Model Monitoring & Drift - 20 questions
export const monitoringDriftQuestions: QuizQuestion[] = [
  {
    id: 'mon1',
    question: 'Why monitor ML models in production?',
    options: ['Not needed', 'Detect degradation, drift, errors, ensure continued performance', 'One-time check', 'Training sufficient'],
    correctAnswer: 1,
    explanation: 'Production monitoring catches performance decay, data distribution shifts, system failures.'
  },
  {
    id: 'mon2',
    question: 'What is model drift?',
    options: ['No change', 'Performance degradation over time due to changing data/environment', 'Improvement', 'Static performance'],
    correctAnswer: 1,
    explanation: 'Drift: model accuracy decreases as real-world data diverges from training distribution.'
  },
  {
    id: 'mon3',
    question: 'What is data drift?',
    options: ['No change', 'Change in input feature distributions (P(X) changes)', 'Label change', 'No distribution shift'],
    correctAnswer: 1,
    explanation: 'Data drift (covariate shift): feature distributions change but X→Y relationship stable.'
  },
  {
    id: 'mon4',
    question: 'What is concept drift?',
    options: ['Data drift only', 'Change in relationship between features and target (P(Y|X) changes)', 'Feature change', 'No drift'],
    correctAnswer: 1,
    explanation: 'Concept drift: underlying patterns change (e.g., fraud tactics evolve, customer preferences shift).'
  },
  {
    id: 'mon5',
    question: 'What causes drift?',
    options: ['No causes', 'Seasonality, market changes, user behavior evolution, external events', 'Static world', 'No changes'],
    correctAnswer: 1,
    explanation: 'Drift sources: trends, seasonality, competition, regulations, black swan events (COVID).'
  },
  {
    id: 'mon6',
    question: 'How to detect data drift?',
    options: ['No detection', 'Statistical tests: KS test, Chi-square, KL divergence, population stability index', 'Visual only', 'No methods'],
    correctAnswer: 1,
    explanation: 'Drift detection: compare production vs training distributions using statistical distance measures.'
  },
  {
    id: 'mon7',
    question: 'What is Kolmogorov-Smirnov (KS) test?',
    options: ['Mean test', 'Tests if two continuous distributions differ significantly', 'Categorical', 'No test'],
    correctAnswer: 1,
    explanation: 'KS test compares cumulative distributions; detects if feature distribution shifted.'
  },
  {
    id: 'mon8',
    question: 'What is Population Stability Index (PSI)?',
    options: ['Accuracy metric', 'Measures distribution change: PSI = Σ (actual% - expected%) · ln(actual%/expected%)', 'Loss function', 'No metric'],
    correctAnswer: 1,
    explanation: 'PSI quantifies drift: PSI < 0.1 stable, 0.1-0.25 moderate, > 0.25 significant drift.'
  },
  {
    id: 'mon9',
    question: 'What is prediction drift?',
    options: ['Input drift', 'Change in model output distribution over time', 'Feature drift', 'No drift'],
    correctAnswer: 1,
    explanation: 'Prediction drift: output distribution shifts (e.g., fraud rate changes from 1% to 5%).'
  },
  {
    id: 'mon10',
    question: 'What metrics to monitor?',
    options: ['Accuracy only', 'Accuracy, precision, recall, latency, error rate, feature distributions', 'No metrics', 'One metric'],
    correctAnswer: 1,
    explanation: 'Monitor: model performance (accuracy, F1), system health (latency, errors), data quality (drift, nulls).'
  },
  {
    id: 'mon11',
    question: 'What is ground truth lag?',
    options: ['Instant labels', 'Delay between prediction and obtaining true label', 'No delay', 'Immediate feedback'],
    correctAnswer: 1,
    explanation: 'Ground truth lag: true outcomes arrive later (loan default: months, disease: years), delaying performance measurement.'
  },
  {
    id: 'mon12',
    question: 'How to handle ground truth lag?',
    options: ['Wait only', 'Use proxy metrics, monitor drift/outliers, sample for labeling', 'No solution', 'Ignore it'],
    correctAnswer: 1,
    explanation: 'Without immediate feedback: track leading indicators, data drift, sample predictions for manual review.'
  },
  {
    id: 'mon13',
    question: 'What is model retraining strategy?',
    options: ['Never retrain', 'Periodic (scheduled) or triggered (when drift detected)', 'Once only', 'Random timing'],
    correctAnswer: 1,
    explanation: 'Retraining approaches: fixed schedule (monthly), performance-based triggers, continuous learning.'
  },
  {
    id: 'mon14',
    question: 'What is online learning?',
    options: ['Batch only', 'Continuously updating model as new data arrives', 'Offline only', 'No updates'],
    correctAnswer: 1,
    explanation: 'Online learning incrementally updates model with streaming data, adapting to drift in real-time.'
  },
  {
    id: 'mon15',
    question: 'What are alerting thresholds?',
    options: ['No alerts', 'Setting limits on metrics to trigger notifications when exceeded', 'Always alert', 'No thresholds'],
    correctAnswer: 1,
    explanation: 'Alerts fire when: accuracy drops X%, latency exceeds Y ms, drift PSI > 0.25, error rate spikes.'
  },
  {
    id: 'mon16',
    question: 'What is shadow scoring?',
    options: ['Production scoring', 'Running new model alongside current, comparing without switching', 'No comparison', 'Direct replacement'],
    correctAnswer: 1,
    explanation: 'Shadow mode validates new model in production environment before directing traffic to it.'
  },
  {
    id: 'mon17',
    question: 'What tools help with monitoring?',
    options: ['No tools', 'Prometheus, Grafana, Evidently AI, Fiddler, Arize, WhyLabs', 'Manual only', 'One tool'],
    correctAnswer: 1,
    explanation: 'ML monitoring: Evidently (drift), Fiddler/Arize (ML observability), Prometheus+Grafana (infrastructure), WhyLabs (data quality).'
  },
  {
    id: 'mon18',
    question: 'What is model explainability monitoring?',
    options: ['Not monitored', 'Tracking feature importances, SHAP values over time for consistency', 'No tracking', 'One-time check'],
    correctAnswer: 1,
    explanation: 'Monitor explanations: if feature importances shift dramatically, may indicate drift or model issues.'
  },
  {
    id: 'mon19',
    question: 'What is feedback loop problem?',
    options: ['No problem', 'Model predictions influence future data, creating bias', 'Independent data', 'No effect'],
    correctAnswer: 1,
    explanation: 'Feedback loops: model shows certain recommendations, users click them, model learns to show more (self-fulfilling).'
  },
  {
    id: 'mon20',
    question: 'What is champion-challenger framework?',
    options: ['Single model', 'Current model (champion) continuously compared against new candidates (challengers)', 'No comparison', 'Replace immediately'],
    correctAnswer: 1,
    explanation: 'Champion-challenger: production model faces periodic challenges from retrained/new models; switch if challenger wins.'
  }
];

// Scaling ML Systems - 20 questions
export const scalingMLQuestions: QuizQuestion[] = [
  {
    id: 'sml1',
    question: 'What is distributed training?',
    options: ['Single GPU', 'Training models across multiple GPUs/machines', 'CPU only', 'No distribution'],
    correctAnswer: 1,
    explanation: 'Distributed training parallelizes computation across hardware to handle larger models/datasets faster.'
  },
  {
    id: 'sml2',
    question: 'What is data parallelism?',
    options: ['Model split', 'Splitting data across devices, each with full model copy', 'No parallelism', 'Parameter split'],
    correctAnswer: 1,
    explanation: 'Data parallelism: replicate model on each GPU, process different batches, synchronize gradients.'
  },
  {
    id: 'sml3',
    question: 'What is model parallelism?',
    options: ['Data split', 'Splitting model across devices when too large for single GPU', 'No parallelism', 'Batch split'],
    correctAnswer: 1,
    explanation: 'Model parallelism: partition layers/weights across GPUs when model exceeds single GPU memory.'
  },
  {
    id: 'sml4',
    question: 'What is gradient accumulation?',
    options: ['Direct update', 'Accumulating gradients over mini-batches before updating', 'Single batch', 'No accumulation'],
    correctAnswer: 1,
    explanation: 'Gradient accumulation simulates larger batch sizes by accumulating gradients from multiple small batches.'
  },
  {
    id: 'sml5',
    question: 'What is mixed precision training?',
    options: ['FP32 only', 'Using FP16 for speed, FP32 for stability', 'Single precision', 'No mixing'],
    correctAnswer: 1,
    explanation: 'Mixed precision (FP16 + FP32) speeds training on modern GPUs while maintaining numerical stability.'
  },
  {
    id: 'sml6',
    question: 'What is batch vs online learning?',
    options: ['Same thing', 'Batch: train on full dataset periodically; Online: update incrementally with each sample', 'No difference', 'Batch only'],
    correctAnswer: 1,
    explanation: 'Batch learning: periodic full retraining. Online learning: continuous updates with streaming data.'
  },
  {
    id: 'sml7',
    question: 'What is mini-batch training?',
    options: ['Full dataset', 'Processing small subsets (32-256 samples) per gradient update', 'Single sample', 'No batching'],
    correctAnswer: 1,
    explanation: 'Mini-batch balances: efficiency (vectorization), memory (fits in GPU), convergence (less noisy than SGD).'
  },
  {
    id: 'sml8',
    question: 'What is feature store?',
    options: ['Raw data store', 'Centralized repository for engineered features with versioning', 'Model store', 'No storage'],
    correctAnswer: 1,
    explanation: 'Feature stores (Feast, Tecton) provide consistent, reusable features across training and serving.'
  },
  {
    id: 'sml9',
    question: 'What is the training-serving skew?',
    options: ['No skew', 'Differences between training and production feature computation causing errors', 'Perfect match', 'No difference'],
    correctAnswer: 1,
    explanation: 'Skew: features computed differently in training vs production (e.g., different libraries, timing) degrades performance.'
  },
  {
    id: 'sml10',
    question: 'How to prevent training-serving skew?',
    options: ['Different code', 'Share feature code, use feature store, test feature parity', 'No prevention', 'Ignore it'],
    correctAnswer: 1,
    explanation: 'Prevent skew: identical feature logic for training/serving, feature stores, automated testing.'
  },
  {
    id: 'sml11',
    question: 'What is MLOps?',
    options: ['DevOps only', 'Practices for operationalizing ML: CI/CD, monitoring, automation', 'No operations', 'Manual only'],
    correctAnswer: 1,
    explanation: 'MLOps extends DevOps to ML: automated pipelines, versioning, monitoring, reproducibility.'
  },
  {
    id: 'sml12',
    question: 'What is CI/CD for ML?',
    options: ['No automation', 'Continuous Integration/Deployment: automated testing, training, deployment pipelines', 'Manual process', 'No testing'],
    correctAnswer: 1,
    explanation: 'ML CI/CD: automated data validation, model training, testing, deployment when code/data changes.'
  },
  {
    id: 'sml13',
    question: 'What is model registry?',
    options: ['No tracking', 'Centralized repository for model versions, metadata, lineage', 'Local storage', 'No versioning'],
    correctAnswer: 1,
    explanation: 'Model registry (MLflow, Weights & Biases) tracks models: versions, metrics, parameters, stage (staging/production).'
  },
  {
    id: 'sml14',
    question: 'What is experiment tracking?',
    options: ['No tracking', 'Logging hyperparameters, metrics, artifacts for each training run', 'Manual notes', 'No records'],
    correctAnswer: 1,
    explanation: 'Experiment tracking (MLflow, W&B, Neptune) records runs for reproducibility and comparison.'
  },
  {
    id: 'sml15',
    question: 'What is data versioning?',
    options: ['No versioning', 'Tracking dataset versions like code with DVC, Pachyderm', 'Manual backup', 'No tracking'],
    correctAnswer: 1,
    explanation: 'Data versioning enables reproducibility: tie model version to exact dataset version used for training.'
  },
  {
    id: 'sml16',
    question: 'What is horizontal vs vertical scaling?',
    options: ['Same thing', 'Horizontal: add more machines; Vertical: bigger machine', 'No scaling', 'One type only'],
    correctAnswer: 1,
    explanation: 'Horizontal (scale out): add nodes, better availability. Vertical (scale up): bigger GPU/RAM, simpler but limited.'
  },
  {
    id: 'sml17',
    question: 'What is caching for ML?',
    options: ['No caching', 'Storing computed features/predictions to avoid recomputation', 'Always compute', 'No storage'],
    correctAnswer: 1,
    explanation: 'Cache: feature vectors, embeddings, frequent predictions to reduce latency and compute costs.'
  },
  {
    id: 'sml18',
    question: 'What is approximate nearest neighbor (ANN)?',
    options: ['Exact search', 'Fast approximate similarity search for embeddings (FAISS, Annoy)', 'Slow search', 'No approximation'],
    correctAnswer: 1,
    explanation: 'ANN (FAISS, HNSW, Annoy) enables sub-linear embedding search, critical for recommendation/retrieval at scale.'
  },
  {
    id: 'sml19',
    question: 'What infrastructure do large ML systems use?',
    options: ['Single machine', 'Kubernetes, GPUs/TPUs, distributed storage (S3), orchestration (Airflow, Kubeflow)', 'No infrastructure', 'Desktop only'],
    correctAnswer: 1,
    explanation: 'Production ML: container orchestration (K8s), accelerators (GPUs/TPUs), workflow orchestration, cloud storage.'
  },
  {
    id: 'sml20',
    question: 'What is the data flywheel?',
    options: ['Static data', 'Virtuous cycle: more users → more data → better models → more users', 'No cycle', 'One-time'],
    correctAnswer: 1,
    explanation: 'Data flywheel: production usage generates data for retraining, improving model, attracting more users/data.'
  }
];
