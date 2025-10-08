import { Topic } from '../../../types';

export const scalingOptimization: Topic = {
  id: 'scaling-optimization',
  title: 'Scaling & Optimization',
  category: 'ml-systems',
  description: 'Strategies for scaling ML systems and optimizing inference performance',
  content: `
    <h2>Scaling & Optimization: Making ML Systems Fast, Scalable, and Cost-Effective</h2>
    
    <p>Your model works beautifully in the lab. Then you deploy it, and reality strikes. Ten users become a thousand. A thousand become a million. Response times that were 10ms balloon to 500ms. Your AWS bill explodes. Users complain. Stakeholders question whether ML was worth the investment. This is the moment where theoretical machine learning meets the harsh realities of production systems.</p>

    <p>Scaling and optimization transform research prototypes into production systems that serve millions of users reliably, quickly, and economically. This requires understanding both infrastructure scaling (adding capacity to handle load) and model optimization (making individual predictions faster and cheaper). The goal isn't just to make things work—it's to make them work at scale, under real-world constraints of latency, cost, and reliability.</p>

    <h3>Scaling Strategies: Adding Capacity to Handle Load</h3>

    <h4>Vertical Scaling: Bigger, Faster, Stronger Machines</h4>

    <p><strong>Vertical scaling</strong> (scaling up) means upgrading individual machines with more powerful hardware—more CPU cores, more RAM, faster GPUs. It's the simplest scaling approach: your code doesn't change, your architecture doesn't change, you just throw better hardware at the problem.</p>

    <p><strong>Advantages:</strong> Simplicity is the killer feature. No distributed system complexity. No load balancing. No network overhead. Single-machine architecture means simpler debugging, simpler deployment, simpler everything. For GPU-bound deep learning inference where you're maxing out a single GPU, upgrading to a more powerful GPU (V100 → A100 → H100) can double or triple throughput without code changes.</p>

    <p><strong>Disadvantages:</strong> Physics imposes limits. You can't buy infinite CPU or RAM. High-end hardware gets exponentially expensive—the jump from 32 to 64 CPU cores costs more than twice as much. Single point of failure: if that one powerful machine goes down, your entire service is offline. No redundancy, no fault tolerance.</p>

    <p><strong>When to use vertical scaling:</strong> Initial deployments where traffic is moderate and you're prototyping. GPU-bound inference where a single powerful GPU handles your load (deep learning models benefit enormously from better GPUs). Stateful systems where distributing state is complex. Low-to-medium traffic scenarios where the cost premium of high-end hardware is justified by simplicity.</p>

    <h4>Horizontal Scaling: An Army of Smaller Machines</h4>

    <p><strong>Horizontal scaling</strong> (scaling out) means adding more machines rather than upgrading existing ones. Instead of one powerful server, you have ten, fifty, a hundred commodity servers working together. Load balancers distribute incoming requests across this fleet, and each machine handles a fraction of total traffic.</p>

    <p><strong>Advantages:</strong> Nearly unlimited scalability—need more capacity? Add more machines. Fault tolerance: if one machine fails, the others continue serving traffic. Users don't notice. Cost-effectiveness at scale: commodity hardware is cheaper per unit of compute than high-end machines, and you can add capacity incrementally rather than big jumps.</p>

    <p><strong>Disadvantages:</strong> Complexity multiplies. You need load balancers to distribute traffic. Network latency between services becomes significant. State management across machines is hard—where do you store sessions, caches, models? Debugging distributed systems is notoriously difficult. More moving parts mean more potential failure modes.</p>

    <p><strong>When to use horizontal scaling:</strong> High traffic volume where single machines can't keep up. CPU-bound inference where adding more CPUs linearly increases capacity (classical ML models, smaller neural networks). High availability requirements where fault tolerance is critical. Applications with stateless serving (each request is independent, making distribution straightforward).</p>

    <h4>Auto-Scaling: Let the System Adapt Itself</h4>

    <p>Traffic doesn't arrive uniformly. You get spikes during business hours, lulls at night. Seasonal patterns, viral events, marketing campaigns—all create unpredictable load patterns. Manually managing capacity is reactive, expensive, and stressful. <strong>Auto-scaling</strong> automatically adjusts resources based on real-time demand, adding capacity when load increases, removing it when load drops.</p>

    <p><strong>Metrics for scaling decisions:</strong> <em>CPU utilization</em>—scale up when average CPU exceeds 70%. Simple, broadly applicable, but can be misleading (IO-bound systems might have low CPU despite being overloaded). <em>Request queue depth</em>—scale when backlog grows beyond threshold (e.g., >100 queued requests). Directly measures capacity strain. <em>Response time</em>—scale when latency degrades (P95 latency exceeds SLA). This measures user experience directly. <em>Time-based patterns</em>—pre-scale for known traffic patterns (scale up before daily 9am spike, scale down after 6pm). Proactive rather than reactive.</p>

    <p><strong>Best practices:</strong> Scale up aggressively, scale down conservatively. When load increases, add capacity quickly to prevent SLA violations. When load decreases, scale down slowly to avoid thrashing (rapid scale-up-then-down-then-up cycles waste time and money). Set minimum instance counts for baseline capacity—always have enough machines to handle unexpected traffic. Set maximum instance counts for budget protection—don't let runaway scaling bankrupt you. Use warmup periods: new instances need time to load models, fill caches, stabilize before receiving full traffic. Monitor scaling events: if auto-scaling triggers constantly, you have underlying capacity or efficiency problems.</p>

    <h3>Model Optimization: Making Individual Predictions Faster and Cheaper</h3>

    <p>Infrastructure scaling addresses <em>how many requests</em> you can handle. Model optimization addresses <em>how fast each prediction is</em>. A 2x speedup per prediction effectively doubles your capacity without adding hardware. The most powerful optimizations come from model compression—making models smaller and faster with minimal accuracy loss.</p>

    <h4>Quantization: Trading Precision for Speed</h4>

    <p><strong>Quantization</strong> reduces numerical precision from 32-bit floating point (FP32) to 8-bit integers (INT8). Models are typically trained in FP32 for numerical stability, but inference rarely needs that precision. Most of those 32 bits are wasted—quantization keeps what matters, discards what doesn't.</p>

    <p><strong>Benefits are dramatic:</strong> 4x smaller model size (32 bits → 8 bits). 2-4x faster inference because integer operations are faster than floating-point, and reduced memory bandwidth becomes the bottleneck in large models. Lower memory requirements mean you can batch more requests or serve larger models on the same hardware.</p>

    <p><strong>Two quantization approaches:</strong> <em>Post-training quantization</em> converts already-trained FP32 models to INT8. Simple—just run a conversion script. Slight accuracy loss (typically <1% for well-behaved models). Works out-of-the-box for most models. <em>Quantization-aware training</em> simulates quantization during training, allowing the model to adapt. Results in better accuracy—model learns to be robust to reduced precision. More complex: requires retraining from scratch or fine-tuning.</p>

    <p><strong>Trade-offs:</strong> Not all operations support INT8—some layers stay FP32, reducing benefits. Some models sensitive to precision (e.g., very small models, models with extreme values) lose more accuracy. Calibration dataset needed for post-training quantization to determine optimal quantization parameters. But for most production deep learning, quantization is free lunch—massive speedup with negligible accuracy cost.</p>

    <h4>Pruning: Cutting Away the Fat</h4>

    <p>Neural networks are over-parameterized. Research models are trained large to explore capacity, but inference doesn't need all those weights. <strong>Pruning</strong> identifies and removes unimportant weights, creating sparse networks that are smaller and faster.</p>

    <p><strong>Unstructured pruning</strong> removes individual weights based on magnitude (small weights contribute little, can be zeroed). Creates irregular sparsity—50-90% of weights can be removed with minimal accuracy loss. But irregular patterns don't map well to hardware—GPUs and CPUs aren't optimized for sparse matrix operations. Need specialized libraries or custom kernels to realize speedups.</p>

    <p><strong>Structured pruning</strong> removes entire structures: channels, filters, layers. Creates regular sparsity that standard hardware handles efficiently. Easier to deploy—pruned model is just smaller dense model. Less compression than unstructured pruning but guaranteed speedups without special hardware.</p>

    <p><strong>Pruning process:</strong> Train full model to convergence. Identify least important weights (by magnitude, gradient information, or more sophisticated metrics). Remove those weights. Fine-tune the pruned model to recover performance—remaining weights adjust to compensate for removed ones. Iterate if needed: prune more, fine-tune more.</p>

    <p><strong>Real-world impact:</strong> Pruned models can be 5-10x smaller with <2% accuracy loss. But realize speedups require hardware support or custom inference engines. Most effective for models where sparsity aligns with hardware capabilities or when model size (not compute) is bottleneck.</p>

    <h4>Knowledge Distillation: Learning from the Master</h4>

    <p><strong>Knowledge distillation</strong> trains a small, fast "student" model to mimic a large, accurate "teacher" model. The teacher has seen all the data, learned all the patterns, captured all the nuance. The student learns a compressed version of that knowledge.</p>

    <p><strong>How it works:</strong> Train large teacher model to high accuracy using standard methods. Use teacher's <em>soft predictions</em> (full probability distribution, not just top class) as training targets for student. Soft predictions contain more information than hard labels—they encode relationships between classes, uncertainty, similar categories. Student learns these relationships, not just memorizing labels. Student can be 10-100x smaller yet outperform student trained directly on hard labels.</p>

    <p><strong>Why it works:</strong> Teacher's predictions are smoother, more informative than one-hot labels. A teacher might say "90% cat, 8% dog, 2% fox" rather than just "cat". That tells student: cats and dogs are related, foxes somewhat similar. This generalization knowledge is what makes distillation powerful. Student learns not just what to predict, but <em>how the teacher thinks</em>.</p>

    <p><strong>Flexibility advantage:</strong> Student architecture can be completely different from teacher. Teacher might be a huge ensemble; student a single small network. This lets you target specific deployment constraints (mobile device, edge hardware) while maintaining teacher's knowledge.</p>

    <p><strong>Trade-offs:</strong> Requires training data—you need representative data to distill on. Requires training time—distillation is full training process. Teacher accuracy limits student—student can't surpass teacher. But when you need maximum compression with minimal accuracy loss, distillation is unmatched.</p>

    <h3>Inference Optimization: Squeezing Out Every Millisecond</h3>

    <h4>Batching: The Power of Parallelism</h4>

    <p>GPUs excel at parallel computation. A single prediction underutilizes that parallelism—most GPU cores sit idle. <strong>Batching</strong> processes multiple requests simultaneously, filling those idle cores and dramatically increasing throughput.</p>

    <p><strong>Benefits:</strong> Better hardware utilization—GPU processes 32 predictions almost as fast as one. Higher throughput—serve 5-10x more requests per second. Lower cost per prediction—amortize fixed overhead across batch.</p>

    <p><strong>Latency trade-off:</strong> Individual requests wait for batch to fill before processing begins. This increases per-request latency. If requests arrive one at a time, you're waiting for batch timeout before processing anything. The solution is <strong>dynamic batching</strong>: configure maximum batch size (e.g., 32) and maximum wait time (e.g., 10ms). Process batch when either threshold is reached. This balances throughput (larger batches) with latency (don't wait forever).</p>

    <p><strong>Configuration:</strong> Maximum batch size limited by GPU memory—larger batches need more memory. Maximum wait time should meet latency SLA—if P95 latency must be <100ms and inference takes 50ms, can't wait more than 50ms for batching. Optimal configuration depends on traffic patterns: high traffic naturally fills batches quickly; low traffic needs aggressive timeouts.</p>

    <h4>Specialized Model Serving Frameworks</h4>

    <p>Rolling your own serving infrastructure is tempting but rarely wise. Specialized frameworks provide battle-tested implementations of batching, multi-model serving, GPU optimization, and monitoring.</p>

    <p><strong>Framework Comparison Table:</strong></p>

    <table>
      <tr>
        <th>Framework</th>
        <th>Best For</th>
        <th>Supported Formats</th>
        <th>Key Strengths</th>
        <th>Limitations</th>
      </tr>
      <tr>
        <td><strong>TensorFlow Serving</strong></td>
        <td>TensorFlow models in production</td>
        <td>TensorFlow SavedModel</td>
        <td>Mature ecosystem, excellent docs, built-in versioning, gRPC + REST</td>
        <td>TensorFlow-only, less flexible for custom logic</td>
      </tr>
      <tr>
        <td><strong>TorchServe</strong></td>
        <td>PyTorch models, custom preprocessing</td>
        <td>PyTorch (.pt, .pth), TorchScript</td>
        <td>Custom handlers, easy extensibility, multi-model serving</td>
        <td>Younger than TF Serving, smaller community</td>
      </tr>
      <tr>
        <td><strong>NVIDIA Triton</strong></td>
        <td>Multi-framework, GPU-heavy workloads</td>
        <td>TensorFlow, PyTorch, ONNX, TensorRT, Python</td>
        <td>Framework-agnostic, best GPU optimization, model ensembles</td>
        <td>Complex setup, steeper learning curve</td>
      </tr>
      <tr>
        <td><strong>ONNX Runtime</strong></td>
        <td>Cross-platform, heterogeneous hardware</td>
        <td>ONNX (converts from any framework)</td>
        <td>Hardware-agnostic optimization, mobile/edge support</td>
        <td>Conversion overhead, not all ops supported</td>
      </tr>
      <tr>
        <td><strong>BentoML</strong></td>
        <td>Rapid prototyping, Python-first</td>
        <td>sklearn, XGBoost, PyTorch, TensorFlow, etc.</td>
        <td>Easy to use, Python-native, quick deployment</td>
        <td>Less optimized than specialized frameworks</td>
      </tr>
    </table>

    <p><strong>Selection guide:</strong> Use <strong>TensorFlow Serving</strong> for pure TensorFlow deployments where stability and maturity matter. Use <strong>TorchServe</strong> for PyTorch models needing custom preprocessing logic. Use <strong>NVIDIA Triton</strong> when serving multiple frameworks, GPU optimization is critical, or you need model ensembles. Use <strong>ONNX Runtime</strong> for cross-platform deployments (cloud + edge) or when targeting specialized hardware. Use <strong>BentoML</strong> for rapid prototyping and smaller-scale deployments where ease-of-use trumps performance.</p>

    <p><strong>Performance tiers:</strong> For maximum throughput on GPUs, choose Triton (with TensorRT backend). For balanced CPU performance, ONNX Runtime excels. For simplicity with good performance, framework-native serving (TF Serving, TorchServe) is optimal. Benchmark on your specific models and hardware—theoretical advantages don't always translate to your use case.</p>

    <h4>Hardware Acceleration: Choosing the Right Tool</h4>

    <p><strong>GPUs</strong> dominate deep learning inference. Massively parallel architecture perfect for tensor operations. Modern GPUs (A100, V100) provide 100-1000x speedup over CPUs for large neural networks. But expensive—both upfront cost and power consumption. Overkill for small models or low traffic.</p>

    <p><strong>CPUs</strong> remain relevant for classical ML (random forests, gradient boosting, linear models) and small neural networks. Low latency—no CPU-to-GPU data transfer. Cost-effective for models that don't benefit from GPU parallelism. Sufficient for many production use cases.</p>

    <p><strong>Custom accelerators</strong> target specific workloads: Google TPUs optimized for TensorFlow, particularly matrix multiplications. AWS Inferentia custom chips for cost-effective deep learning inference. Edge TPUs for on-device inference on mobile/IoT devices. These trade generality for efficiency in specific domains.</p>

    <h3>Caching: Avoiding Work is Faster Than Doing Work Faster</h3>

    <p>The fastest computation is the one you don't do. <strong>Caching</strong> stores results of expensive operations, reusing them when possible. For ML systems, caching applies at multiple levels.</p>

    <p><strong>Prediction caching:</strong> Store complete predictions keyed by input feature hash. Effective when same inputs appear repeatedly—product recommendations, fraud detection on similar transactions, content moderation on duplicate content. Requires careful cache invalidation: how long are predictions valid? For recommendations, maybe hours; for fraud detection, maybe seconds. Implementation with Redis or Memcached provides microsecond lookups. Can achieve 10-100x speedup for cache hits.</p>

    <p><strong>Feature caching:</strong> Pre-compute expensive features offline. User embeddings for recommendation systems—compute once daily, cache for 24 hours. Aggregated statistics (30-day purchase history)—compute in batch jobs, serve from cache. Entity features (product metadata)—rarely change, cache indefinitely. This separates online serving (fast, cached) from offline computation (slow, but not in critical path).</p>

    <p><strong>Model caching:</strong> Load models into memory on service startup, keep them resident. Avoid loading overhead on every request. For multi-model serving, use LRU eviction—keep recently-used models in memory, evict least-recently-used when memory fills. Warm models with dummy inputs on load to trigger JIT compilation and cache population.</p>

    <h3>Architecture Patterns for Efficient Serving</h3>

    <p><strong>Model cascade</strong> leverages cost-accuracy tradeoffs. Stage 1: lightweight, fast model (logistic regression, small tree ensemble) filters obvious negatives. Filters 90-95% of inputs in <1ms each. Stage 2: heavy, accurate model (large neural network) processes remaining inputs. This reduces average latency dramatically—most requests get fast path, few requests get slow path. Example: fraud detection might use rules and logistic regression to filter 95% of legitimate transactions in <1ms, then apply deep learning to suspicious 5%. Average latency drops from 50ms to 5ms.</p>

    <p><strong>Model ensemble</strong> combines multiple models for better accuracy. Parallel ensembles run all models simultaneously, aggregate predictions (voting, averaging). Sequential ensembles pass outputs through pipeline, each stage refining previous. Trade-off: better accuracy vs. higher latency and cost. Useful when accuracy is paramount (medical diagnosis, financial decisions) and latency budget allows.</p>

    <p><strong>Microservices architecture</strong> separates model serving from application logic. ML service is independent, scaled independently, updated independently. Application calls ML service via API. Benefits: technology stack flexibility (Python for ML, Java for business logic), independent scaling (scale ML service more aggressively than app), easier updates (deploy new model without touching application code). Additional network hop adds latency but gains operational flexibility.</p>

    <h3>Latency Optimization: Meeting SLA Requirements</h3>

    <p><strong>Profile to identify bottlenecks.</strong> Don't optimize blindly. Measure where time goes: preprocessing (feature extraction, data transformation), model inference (forward pass), postprocessing (decoding outputs, ranking). Network latency (data transfer to/from service). Queue time (waiting for GPU/CPU availability). Optimize the largest bottleneck first—optimizing a 1ms step when you have a 100ms bottleneck wastes effort.</p>

    <p><strong>Optimization techniques span multiple levels:</strong> Reduce model size through quantization, pruning, distillation. Optimize operations with fused kernels (combine multiple operations into single GPU kernel, reducing memory traffic). Use batch processing for throughput, but balance against latency. Employ asynchronous processing for I/O-heavy preprocessing—don't block on network calls or disk reads. Pre-compute static features offline, serve from cache. Move expensive computation out of critical path.</p>

    <p><strong>Define and monitor latency SLA.</strong> Set target: "P95 latency < 100ms" means 95% of requests complete within 100ms. Monitor continuously, alert on violations. When SLA is at risk, be prepared to trade accuracy for speed—switch to faster model, reduce ensemble size, skip expensive features. Production systems require operational discipline: latency budgets, monitoring, clear escalation when SLA is violated.</p>

    <h3>Cost Optimization: Making ML Economically Sustainable</h3>

    <p><strong>Compute costs</strong> dominate ML budgets. Right-size instances—don't over-provision. Profile real utilization, scale down oversized machines. Use spot or preemptible instances for fault-tolerant batch workloads—70% cost savings at the expense of potential interruption. Schedule batch workloads for off-peak hours when compute is cheaper. Model compression reduces compute needs—smaller models mean cheaper hardware.</p>

    <p><strong>Storage costs</strong> accumulate from feature stores, model artifacts, logs. Deduplicate and compress features in feature store. Archive old model versions to cold storage (S3 Glacier, Azure Archive)—keep recent versions hot, archive historical versions. Sample or aggregate logs before storage—do you need every request logged, or can you sample 10%?</p>

    <p><strong>Data transfer costs</strong> are hidden killers in cloud environments. Colocation: keep model and data in same region to avoid inter-region transfer fees. Compress payloads—request and response compression can reduce transfer by 10x. Edge caching: serve static predictions from CDN, reducing origin server load and data transfer.</p>

    <p>Scaling and optimization transform ML from expensive experiments into efficient production systems. Infrastructure scaling (vertical, horizontal, auto-scaling) handles increasing load. Model optimization (quantization, pruning, distillation) makes individual predictions faster and cheaper. Inference optimization (batching, specialized frameworks, caching) squeezes out every millisecond. Thoughtful architecture patterns and cost optimization make systems economically sustainable. Master these techniques, and your ML systems will scale from prototype to millions of users while meeting latency SLAs and staying within budget. This is the engineering that makes ML valuable in the real world.</p>
  `,
  codeExamples: [
    {
      language: 'Python',
      code: `import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, quantize_fx
import time
import numpy as np

# Define a simple neural network
class SimpleModel(nn.Module):
  def __init__(self):
      super().__init__()
      self.fc1 = nn.Linear(100, 500)
      self.fc2 = nn.Linear(500, 500)
      self.fc3 = nn.Linear(500, 10)
      self.relu = nn.ReLU()

  def forward(self, x):
      x = self.relu(self.fc1(x))
      x = self.relu(self.fc2(x))
      x = self.fc3(x)
      return x

# Original model
model_fp32 = SimpleModel()
model_fp32.eval()

# Benchmark function
def benchmark_model(model, input_tensor, num_runs=1000):
  """Measure inference time and model size."""
  # Warmup
  for _ in range(10):
      _ = model(input_tensor)

  # Benchmark
  start = time.time()
  for _ in range(num_runs):
      with torch.no_grad():
          _ = model(input_tensor)
  end = time.time()

  avg_time = (end - start) / num_runs * 1000  # ms

  # Model size
  torch.save(model.state_dict(), 'temp_model.pt')
  import os
  size_mb = os.path.getsize('temp_model.pt') / (1024 * 1024)
  os.remove('temp_model.pt')

  return avg_time, size_mb

# Test input
input_tensor = torch.randn(1, 100)

print("=== Model Optimization Comparison ===\\n")

# 1. Original FP32 model
print("1. Original FP32 Model:")
time_fp32, size_fp32 = benchmark_model(model_fp32, input_tensor)
print(f"   Inference time: {time_fp32:.3f} ms")
print(f"   Model size: {size_fp32:.2f} MB\\n")

# 2. Dynamic Quantization (FP32 → INT8)
print("2. Dynamic Quantization (INT8):")
model_int8 = quantize_dynamic(
  model_fp32,
  {nn.Linear},  # Quantize Linear layers
  dtype=torch.qint8
)
time_int8, size_int8 = benchmark_model(model_int8, input_tensor)
print(f"   Inference time: {time_int8:.3f} ms ({time_fp32/time_int8:.1f}x speedup)")
print(f"   Model size: {size_int8:.2f} MB ({size_fp32/size_int8:.1f}x smaller)\\n")

# 3. Pruning (simplified example)
print("3. Pruning:")
import torch.nn.utils.prune as prune

model_pruned = SimpleModel()
model_pruned.load_state_dict(model_fp32.state_dict())

# Prune 50% of weights in each Linear layer
for name, module in model_pruned.named_modules():
  if isinstance(module, nn.Linear):
      prune.l1_unstructured(module, name='weight', amount=0.5)
      prune.remove(module, 'weight')  # Make pruning permanent

time_pruned, size_pruned = benchmark_model(model_pruned, input_tensor)
print(f"   Inference time: {time_pruned:.3f} ms")
print(f"   Model size: {size_pruned:.2f} MB")
print(f"   Note: Sparse models need specialized hardware for speedup\\n")

# 4. Knowledge Distillation (concept)
print("4. Knowledge Distillation (Student Model):")

class StudentModel(nn.Module):
  """Smaller student model."""
  def __init__(self):
      super().__init__()
      self.fc1 = nn.Linear(100, 100)  # Much smaller
      self.fc2 = nn.Linear(100, 10)
      self.relu = nn.ReLU()

  def forward(self, x):
      x = self.relu(self.fc1(x))
      x = self.fc2(x)
      return x

student_model = StudentModel()
student_model.eval()

time_student, size_student = benchmark_model(student_model, input_tensor)
print(f"   Inference time: {time_student:.3f} ms ({time_fp32/time_student:.1f}x speedup)")
print(f"   Model size: {size_student:.2f} MB ({size_fp32/size_student:.1f}x smaller)")
print(f"   Note: Requires training with teacher model\\n")

# 5. Batching comparison
print("5. Batching (Throughput Optimization):")

def benchmark_batched(model, batch_sizes=[1, 8, 32]):
  """Compare throughput with different batch sizes."""
  results = {}
  for batch_size in batch_sizes:
      input_batch = torch.randn(batch_size, 100)
      num_runs = 100

      # Warmup
      for _ in range(10):
          _ = model(input_batch)

      # Benchmark
      start = time.time()
      for _ in range(num_runs):
          with torch.no_grad():
              _ = model(input_batch)
      end = time.time()

      total_time = end - start
      throughput = (num_runs * batch_size) / total_time  # predictions/sec
      latency = (total_time / num_runs) * 1000  # ms per batch

      results[batch_size] = {
          'throughput': throughput,
          'latency': latency
      }

  return results

batching_results = benchmark_batched(model_fp32)
for bs, metrics in batching_results.items():
  print(f"   Batch size {bs}:")
  print(f"     Throughput: {metrics['throughput']:.0f} predictions/sec")
  print(f"     Latency: {metrics['latency']:.2f} ms/batch")

print("\\n=== Summary ===")
print("Quantization: Best for CPU inference, 4x smaller, 2-4x faster")
print("Pruning: Requires sparse hardware support for speedup")
print("Distillation: Best compression, requires retraining")
print("Batching: Increases throughput but may increase latency")`,
      explanation: 'Comprehensive comparison of model optimization techniques including dynamic quantization (FP32→INT8), pruning, knowledge distillation, and batching. Shows real performance measurements for inference time and model size reduction, demonstrating the trade-offs between different optimization strategies.'
    },
    {
      language: 'Python',
      code: `import time
import numpy as np
from functools import lru_cache
from collections import defaultdict
from typing import Dict, List, Optional
import hashlib
import pickle

class OptimizedMLService:
  """Production ML service with caching, batching, and monitoring."""

  def __init__(self, model, max_batch_size=32, max_wait_ms=10):
      """
      Args:
          model: The ML model
          max_batch_size: Maximum batch size for inference
          max_wait_ms: Maximum time to wait for batch to fill
      """
      self.model = model
      self.max_batch_size = max_batch_size
      self.max_wait_ms = max_wait_ms / 1000  # Convert to seconds

      # Caching
      self.prediction_cache = {}
      self.cache_hits = 0
      self.cache_misses = 0

      # Batching queue
      self.batch_queue = []
      self.last_batch_time = time.time()

      # Monitoring
      self.latency_samples = []
      self.request_count = 0

      # Feature cache (pre-computed features)
      self.feature_cache = {}

  def _hash_input(self, features: Dict) -> str:
      """Create deterministic hash of input features for caching."""
      # Sort keys for consistency
      feature_str = str(sorted(features.items()))
      return hashlib.md5(feature_str.encode()).hexdigest()

  @lru_cache(maxsize=10000)
  def get_user_embedding(self, user_id: str) -> np.ndarray:
      """
      Cached user embeddings (expensive to compute).
      Using @lru_cache for automatic cache management.
      """
      # In production: fetch from feature store
      # Simulate expensive computation
      time.sleep(0.01)  # 10ms
      return np.random.randn(128)

  def preprocess_features(self, features: Dict) -> np.ndarray:
      """Preprocess features, using cache when possible."""
      # Check if we have cached preprocessed features
      cache_key = self._hash_input(features)

      if cache_key in self.feature_cache:
          return self.feature_cache[cache_key]

      # Expensive preprocessing
      processed = np.array([
          features.get('age', 0) / 100,
          features.get('income', 0) / 100000,
          # ... more feature engineering
      ])

      # Cache for future requests
      self.feature_cache[cache_key] = processed

      return processed

  def predict_with_cache(self, features: Dict) -> float:
      """Single prediction with caching."""
      # Check prediction cache
      cache_key = self._hash_input(features)

      if cache_key in self.prediction_cache:
          self.cache_hits += 1
          return self.prediction_cache[cache_key]

      self.cache_misses += 1

      # Preprocess
      processed_features = self.preprocess_features(features)

      # Make prediction
      prediction = self.model.predict([processed_features])[0]

      # Cache result (with TTL in production)
      self.prediction_cache[cache_key] = prediction

      return prediction

  def predict_batch(self, features_list: List[Dict]) -> List[float]:
      """Batched prediction for efficiency."""
      if len(features_list) == 0:
          return []

      start_time = time.time()

      # Preprocess all features
      processed_features = np.array([
          self.preprocess_features(f) for f in features_list
      ])

      # Batch prediction
      predictions = self.model.predict(processed_features)

      # Record latency
      latency = (time.time() - start_time) * 1000  # ms
      self.latency_samples.append(latency)

      return predictions.tolist()

  def predict_with_dynamic_batching(self, features: Dict,
                                    callback=None) -> None:
      """
      Async prediction with dynamic batching.
      Groups requests to optimize throughput.

      Args:
          features: Input features
          callback: Function to call with prediction result
      """
      # Add to batch queue
      self.batch_queue.append({
          'features': features,
          'callback': callback,
          'timestamp': time.time()
      })

      # Process batch if full or timeout
      should_process = (
          len(self.batch_queue) >= self.max_batch_size or
          (time.time() - self.last_batch_time) >= self.max_wait_ms
      )

      if should_process:
          self._process_batch()

  def _process_batch(self):
      """Process queued batch of predictions."""
      if len(self.batch_queue) == 0:
          return

      # Extract features and callbacks
      batch_items = self.batch_queue[:self.max_batch_size]
      self.batch_queue = self.batch_queue[self.max_batch_size:]

      features_list = [item['features'] for item in batch_items]
      callbacks = [item['callback'] for item in batch_items]

      # Batch predict
      predictions = self.predict_batch(features_list)

      # Invoke callbacks
      for pred, callback in zip(predictions, callbacks):
          if callback:
              callback(pred)

      self.last_batch_time = time.time()

  def get_metrics(self) -> Dict:
      """Get service performance metrics."""
      total_requests = self.cache_hits + self.cache_misses

      metrics = {
          'total_requests': total_requests,
          'cache_hit_rate': self.cache_hits / total_requests if total_requests > 0 else 0,
          'cache_size': len(self.prediction_cache),
      }

      if self.latency_samples:
          metrics['latency_p50'] = np.percentile(self.latency_samples, 50)
          metrics['latency_p95'] = np.percentile(self.latency_samples, 95)
          metrics['latency_p99'] = np.percentile(self.latency_samples, 99)

      return metrics

  def optimize_cache(self, max_size: int = 10000):
      """Evict oldest cache entries to manage memory."""
      if len(self.prediction_cache) > max_size:
          # Simple eviction: remove random entries
          # In production: use LRU or TTL
          keys_to_remove = list(self.prediction_cache.keys())[:len(self.prediction_cache) - max_size]
          for key in keys_to_remove:
              del self.prediction_cache[key]

# Simulate a simple model
class DummyModel:
  def predict(self, X):
      time.sleep(0.005)  # 5ms inference
      return np.random.randn(len(X))

# Usage example
print("=== Optimized ML Service Demo ===\\n")

model = DummyModel()
service = OptimizedMLService(model, max_batch_size=8, max_wait_ms=10)

# Test caching
print("1. Testing prediction caching:")
features1 = {'age': 30, 'income': 50000}

# First request (cache miss)
start = time.time()
pred1 = service.predict_with_cache(features1)
time1 = (time.time() - start) * 1000

# Second request (cache hit)
start = time.time()
pred2 = service.predict_with_cache(features1)
time2 = (time.time() - start) * 1000

print(f"   First request: {time1:.2f} ms (cache miss)")
print(f"   Second request: {time2:.2f} ms (cache hit)")
print(f"   Speedup: {time1/time2:.1f}x\\n")

# Test batching
print("2. Testing batched prediction:")

# Individual predictions
features_list = [{'age': i, 'income': 40000 + i*1000} for i in range(20, 40)]

start = time.time()
individual_preds = [service.predict_with_cache(f) for f in features_list]
individual_time = (time.time() - start) * 1000

# Clear cache for fair comparison
service.prediction_cache.clear()

# Batched prediction
start = time.time()
batch_preds = service.predict_batch(features_list)
batch_time = (time.time() - start) * 1000

print(f"   Individual predictions: {individual_time:.2f} ms")
print(f"   Batched prediction: {batch_time:.2f} ms")
print(f"   Speedup: {individual_time/batch_time:.1f}x\\n")

# Metrics
print("3. Service Metrics:")
metrics = service.get_metrics()
for key, value in metrics.items():
  if isinstance(value, float):
      print(f"   {key}: {value:.3f}")
  else:
      print(f"   {key}: {value}")

print("\\n=== Optimization Summary ===")
print("✓ Caching: 10-100x speedup for repeated queries")
print("✓ Batching: 2-5x throughput improvement")
print("✓ Feature caching: Reduces preprocessing overhead")
print("✓ Monitoring: Track performance and optimize bottlenecks")`,
      explanation: 'Production-ready ML service demonstrating key optimization patterns: prediction caching with hashing, feature caching with LRU, dynamic batching to improve throughput, and comprehensive metrics tracking. Shows real performance improvements from caching (10-100x for cache hits) and batching (2-5x throughput).'
    }
  ],
  interviewQuestions: [
    {
      question: 'What is the difference between vertical and horizontal scaling? When would you use each?',
      answer: `Vertical scaling increases resources (CPU, memory) on single machines; horizontal scaling adds more machines. Vertical scaling: easier implementation, no distributed computing complexity, limited by hardware constraints. Horizontal scaling: unlimited scaling potential, fault tolerance, complexity in coordination. Use vertical for: stateful applications, small-medium loads, quick solutions. Use horizontal for: large-scale systems, fault tolerance requirements, cost efficiency at scale.`
    },
    {
      question: 'Explain the trade-offs between quantization, pruning, and knowledge distillation for model compression.',
      answer: `Quantization reduces precision (FP32 to INT8) - fast implementation, significant speedup, minimal accuracy loss. Pruning removes less important weights/neurons - better compression ratios, requires retraining, may need specialized hardware. Knowledge distillation trains smaller student model from teacher - flexible architecture changes, maintains performance, requires training data. Choose based on deployment constraints, accuracy requirements, and available computational resources.`
    },
    {
      question: 'How does dynamic batching improve throughput, and what is the latency trade-off?',
      answer: `Dynamic batching groups multiple requests to leverage parallel processing, significantly improving GPU utilization and throughput. Benefits: better hardware efficiency, reduced per-request costs. Latency trade-off: individual requests wait for batch formation, increasing tail latencies. Mitigation strategies: adaptive batch sizes, timeout mechanisms, prioritization queues. Configure based on traffic patterns and SLA requirements - higher throughput vs. lower latency.`
    },
    {
      question: 'What caching strategies would you use for a recommendation system with millions of users?',
      answer: `Multi-tier caching: (1) Application cache - Redis/Memcached for user preferences, (2) Model cache - pre-computed recommendations for active users, (3) Feature cache - user/item embeddings, (4) CDN cache - static recommendations. Strategies: LRU eviction, cache warming for popular items, personalized cache based on user activity patterns, cache invalidation for real-time updates. Balance memory costs with response time improvements.`
    },
    {
      question: 'How would you optimize a model to meet a P95 latency SLA of 50ms?',
      answer: `Optimization approach: (1) Profile to identify bottlenecks - model inference, preprocessing, I/O, (2) Model optimization - quantization, pruning, TensorRT/ONNX, (3) Infrastructure - GPU acceleration, optimized serving frameworks (TensorFlow Serving, Triton), (4) Caching frequent requests, (5) Asynchronous processing where possible, (6) Load balancing, (7) Monitor tail latencies continuously. Iteratively optimize highest impact components first.`
    },
    {
      question: 'Explain how to use auto-scaling effectively for an ML serving system.',
      answer: `Effective auto-scaling requires: (1) Proper metrics - not just CPU/memory but request queue length, model latency, (2) Predictive scaling based on traffic patterns, (3) Warm-up time consideration for ML models, (4) Scale-up aggressiveness vs. scale-down conservatism, (5) Circuit breakers for cascading failures, (6) Cost optimization with spot instances, (7) Multi-region scaling for global traffic. Monitor business metrics alongside technical metrics for effectiveness.`
    }
  ],
  quizQuestions: [
    {
      id: 'scale1',
      question: 'What does INT8 quantization do?',
      options: ['Removes weights', 'Reduces precision from FP32 to 8-bit integers', 'Trains smaller model', 'Compresses images'],
      correctAnswer: 1,
      explanation: 'Quantization reduces numerical precision from floating point (FP32, 32 bits) to integers (INT8, 8 bits), resulting in 4x smaller models and 2-4x faster inference with minimal accuracy loss (typically < 1%).'
    },
    {
      id: 'scale2',
      question: 'What is the main benefit of dynamic batching?',
      options: ['Lower latency per request', 'Higher throughput', 'Smaller model size', 'Better accuracy'],
      correctAnswer: 1,
      explanation: 'Dynamic batching improves throughput by processing multiple requests together, which better utilizes GPU/CPU parallelism. However, it may increase per-request latency since requests wait for the batch to fill.'
    },
    {
      id: 'scale3',
      question: 'When is vertical scaling preferred over horizontal?',
      options: ['Always', 'For GPU-bound inference with moderate traffic', 'For high traffic', 'Never'],
      correctAnswer: 1,
      explanation: 'Vertical scaling (adding more resources to one machine) is preferred for GPU-bound deep learning inference with low-moderate traffic, as it avoids distributed system complexity. Horizontal scaling is better for high traffic and CPU-bound workloads.'
    }
  ]
};
