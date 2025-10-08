import { Topic } from '../../../types';

export const modelCompression: Topic = {
  id: 'model-compression',
  title: 'Model Compression',
  category: 'advanced',
  description: 'Techniques to reduce model size and inference cost',
  content: `
    <h2>Model Compression: Efficient Deep Learning</h2>
    <p>As deep learning models grow larger—GPT-3 with 175 billion parameters, modern vision models exceeding 10GB—deploying them becomes increasingly challenging. Model compression addresses this challenge through techniques that reduce model size, memory footprint, and computational requirements while preserving accuracy. These methods enable deployment on resource-constrained devices (smartphones, IoT, edge hardware), reduce inference latency for real-time applications, lower cloud serving costs, and decrease energy consumption. From quantization that reduces numerical precision, to pruning that removes redundant parameters, to knowledge distillation that transfers knowledge from large to small models, compression has become essential for practical deep learning deployment. The field balances competing objectives: maximum compression, minimal accuracy loss, and hardware compatibility.</p>

    <h3>The Deployment Challenge</h3>

    <h4>Why Compress Models?</h4>

    <h5>Resource Constraints</h5>
    <ul>
      <li><strong>Memory:</strong> Mobile devices typically 2-6GB RAM, large models exceed this</li>
      <li><strong>Storage:</strong> App size limits (iOS 200MB over-the-air), model must fit</li>
      <li><strong>Bandwidth:</strong> Slow networks, expensive data—smaller models faster to download/update</li>
      <li><strong>Energy:</strong> Battery-powered devices require energy-efficient inference</li>
    </ul>

    <h5>Performance Requirements</h5>
    <ul>
      <li><strong>Latency:</strong> Real-time applications need <100ms response (AR, speech, robotics)</li>
      <li><strong>Throughput:</strong> Cloud services must handle many requests per second</li>
      <li><strong>Cost:</strong> Cloud inference expensive at scale—smaller models reduce compute costs</li>
    </ul>

    <h5>The Scaling Problem</h5>
    <ul>
      <li><strong>Overparameterization:</strong> Modern models often have 10x more parameters than needed for task</li>
      <li><strong>Redundancy:</strong> Many weights contribute minimally to final predictions</li>
      <li><strong>Opportunity:</strong> Significant compression possible with minimal accuracy loss</li>
    </ul>

    <h3>Quantization: Reducing Numerical Precision</h3>

    <h4>The Core Idea</h4>
    <p><strong>Principle:</strong> Represent weights and activations with fewer bits—32-bit floating point (FP32) → 8-bit integers (INT8) or lower.</p>

    <h5>Precision Levels</h5>
    <ul>
      <li><strong>FP32 (32-bit float):</strong> Standard training precision, ~4 bytes per parameter</li>
      <li><strong>FP16 (16-bit float):</strong> Half precision, 2x smaller, 2x faster on compatible hardware</li>
      <li><strong>INT8 (8-bit integer):</strong> 4x smaller, 4x faster, minimal accuracy loss for most tasks</li>
      <li><strong>INT4/INT2:</strong> Extreme quantization, significant accuracy trade-offs</li>
    </ul>

    <h4>Post-Training Quantization (PTQ)</h4>

    <h5>Workflow</h5>
    <ol>
      <li><strong>Train model in FP32:</strong> Standard training procedure</li>
      <li><strong>Calibrate:</strong> Run representative data through model, collect activation statistics</li>
      <li><strong>Compute scale factors:</strong> Determine mapping from FP32 range to INT8 range</li>
      <li><strong>Quantize weights and activations:</strong> Convert to INT8</li>
      <li><strong>Deploy:</strong> Use quantized model for inference</li>
    </ol>

    <h5>Affine Quantization</h5>
    <p><strong>Map floating point to integers:</strong></p>
    <p style="text-align: center;">
      $x_{\\text{quant}} = \\text{round}(x / \\text{scale}) + \\text{zero\\_point}$
    </p>
    <ul>
      <li><strong>scale:</strong> Step size between quantized levels (range / 255 for INT8)</li>
      <li><strong>zero_point:</strong> Offset to handle asymmetric ranges</li>
      <li><strong>Dequantization:</strong> $x_{\\text{float}} = (x_{\\text{quant}} - \\text{zero\\_point}) \\times \\text{scale}$</li>
    </ul>

    <h5>Advantages and Limitations</h5>
    <ul>
      <li><strong>✓ No retraining:</strong> Fast, simple, apply to any model</li>
      <li><strong>✓ 4x compression (FP32→INT8):</strong> Significant size reduction</li>
      <li><strong>✓ Hardware acceleration:</strong> Most modern chips have INT8 optimized ops</li>
      <li><strong>✗ Accuracy loss:</strong> Typically 1-2% on standard tasks, more on sensitive tasks</li>
      <li><strong>✗ Calibration data needed:</strong> Representative dataset required</li>
    </ul>

    <h4>Quantization-Aware Training (QAT)</h4>

    <h5>Motivation</h5>
    <p>Train model to be robust to quantization by simulating low precision during training.</p>

    <h5>Fake Quantization</h5>
    <ul>
      <li><strong>Forward pass:</strong> Quantize to INT8, immediately dequantize to FP32</li>
      <li><strong>Backward pass:</strong> Gradients flow through as if no quantization (straight-through estimator)</li>
      <li><strong>Effect:</strong> Model learns weights that perform well when quantized</li>
    </ul>

    <h5>Process</h5>
    <ol>
      <li><strong>Insert fake quantization nodes:</strong> After each operation</li>
      <li><strong>Train with quantization simulation:</strong> Model adapts to discretization</li>
      <li><strong>Fine-tune:</strong> Few epochs usually sufficient</li>
      <li><strong>Deploy:</strong> Convert to actual quantized model</li>
    </ol>

    <h5>Benefits</h5>
    <ul>
      <li><strong>Better accuracy:</strong> Often <1% degradation, sometimes no loss</li>
      <li><strong>Aggressive quantization:</strong> Enables INT4, binary networks</li>
      <li><strong>Trade-off:</strong> Requires retraining (more expensive than PTQ)</li>
    </ul>

    <h4>Mixed Precision Quantization</h4>
    <ul>
      <li><strong>Heterogeneous precision:</strong> Different layers at different precisions</li>
      <li><strong>Sensitive layers FP16:</strong> First/last layers, attention layers</li>
      <li><strong>Robust layers INT8:</strong> Middle layers, simple convolutions</li>
      <li><strong>Automatic search:</strong> NAS-based methods find optimal bit-width per layer</li>
    </ul>

    <h3>Pruning: Removing Redundant Parameters</h3>

    <h4>The Redundancy Hypothesis</h4>
    <p>Deep networks are overparameterized—many weights contribute negligibly to predictions. Pruning removes these redundant parameters.</p>

    <h4>Unstructured (Magnitude) Pruning</h4>

    <h5>Method</h5>
    <ol>
      <li><strong>Train dense network:</strong> Full model to convergence</li>
      <li><strong>Identify unimportant weights:</strong> Typically by magnitude $|w_i|$</li>
      <li><strong>Set to zero:</strong> Prune bottom p% of weights (e.g., 50%, 90%)</li>
      <li><strong>Fine-tune:</strong> Retrain remaining weights to recover accuracy</li>
      <li><strong>Iterate:</strong> Optionally repeat pruning and fine-tuning</li>
    </ol>

    <h5>Sparse Matrices</h5>
    <ul>
      <li><strong>Storage:</strong> Store only non-zero weights + indices (CSR, COO formats)</li>
      <li><strong>Compression:</strong> 90% sparsity → 10x smaller (with overhead)</li>
      <li><strong>Hardware challenge:</strong> Irregular sparsity requires special hardware (Nvidia Ampere, TPUs)</li>
      <li><strong>Without hardware support:</strong> No speedup, only storage savings</li>
    </ul>

    <h5>Pruning Criteria</h5>
    <ul>
      <li><strong>Magnitude:</strong> $|w_i| <$ threshold (simple, effective)</li>
      <li><strong>Gradient-based:</strong> Prune weights with small $\\frac{\\partial L}{\\partial w}$ (negligible impact on loss)</li>
      <li><strong>Hessian-based:</strong> Optimal Brain Damage—second-order information</li>
      <li><strong>Movement pruning:</strong> Prune weights moving toward zero during training</li>
    </ul>

    <h4>Structured Pruning</h4>

    <h5>Motivation</h5>
    <p>Remove entire structures (channels, filters, layers) for actual speedups on standard hardware.</p>

    <h5>Granularities</h5>
    <ul>
      <li><strong>Filter pruning:</strong> Remove entire convolutional filters (e.g., 32 filters → 16 filters)</li>
      <li><strong>Channel pruning:</strong> Remove input/output channels</li>
      <li><strong>Layer pruning:</strong> Remove entire layers (e.g., skip connections in ResNet)</li>
      <li><strong>Block pruning:</strong> Remove structured blocks (e.g., attention heads in Transformers)</li>
    </ul>

    <h5>Selection Methods</h5>
    <ul>
      <li><strong>L1/L2 norm:</strong> Prune filters/channels with smallest norm</li>
      <li><strong>Activation-based:</strong> Prune channels with lowest average activation</li>
      <li><strong>Gradient-based:</strong> Importance measured by gradient magnitude</li>
      <li><strong>Taylor expansion:</strong> Approximate change in loss if filter removed</li>
    </ul>

    <h5>Advantages</h5>
    <ul>
      <li><strong>✓ Actual speedups:</strong> Reduced FLOPs, memory, latency on any hardware</li>
      <li><strong>✓ Simpler deployment:</strong> No sparse matrix support needed</li>
      <li><strong>✗ Lower compression ratios:</strong> Typically 2-5x vs 10-50x for unstructured</li>
    </ul>

    <h4>Iterative Magnitude Pruning (IMP)</h4>

    <h5>Algorithm</h5>
    <pre>
1. Train dense network to convergence
2. Prune p% of weights (e.g., 20%)
3. Fine-tune for k epochs
4. Repeat steps 2-3 until target sparsity
5. Final fine-tuning
    </pre>

    <h5>Lottery Ticket Hypothesis</h5>
    <ul>
      <li><strong>Discovery:</strong> Dense networks contain sparse subnetworks that train to comparable accuracy</li>
      <li><strong>Winning ticket:</strong> Sparse network at initialization that trains successfully</li>
      <li><strong>Implication:</strong> Good pruning recovers these subnetworks</li>
      <li><strong>Rewinding:</strong> Reset unpruned weights to early training checkpoint (not random init)</li>
    </ul>

    <h3>Knowledge Distillation: Teacher-Student Learning</h3>

    <h4>Core Concept</h4>
    <p><strong>Transfer knowledge from large teacher model to small student model through imitation learning.</strong></p>

    <h5>Why It Works</h5>
    <ul>
      <li><strong>Soft targets:</strong> Teacher's probability distribution contains more information than hard labels</li>
      <li><strong>Dark knowledge:</strong> Relative probabilities between classes (e.g., "cat" is more similar to "dog" than "car")</li>
      <li><strong>Regularization:</strong> Soft targets smooth decision boundaries</li>
    </ul>

    <h4>Distillation Process</h4>

    <h5>Standard Distillation</h5>
    <ol>
      <li><strong>Train large teacher model T:</strong> Achieve high accuracy</li>
      <li><strong>Define small student model S:</strong> Fewer layers, fewer parameters</li>
      <li><strong>Generate soft targets:</strong> Run data through teacher, collect probability distributions</li>
      <li><strong>Train student:</strong> Match teacher's outputs AND true labels</li>
    </ol>

    <h5>Distillation Loss</h5>
    <p style="text-align: center; font-size: 1.1em;">
      $L = \\alpha \\times L_{\\text{hard}}(y_{\\text{true}}, \\hat{y}_{\\text{student}}) + (1-\\alpha) \\times L_{\\text{soft}}(\\hat{y}_{\\text{teacher}}, \\hat{y}_{\\text{student}})$
    </p>
    <ul>
      <li><strong>$L_{\\text{hard}}$:</strong> Cross-entropy with true labels (standard supervision)</li>
      <li><strong>$L_{\\text{soft}}$:</strong> KL divergence between teacher and student outputs</li>
      <li><strong>$\\alpha$:</strong> Balance factor (typically 0.5-0.9)</li>
    </ul>

    <h5>Temperature Scaling</h5>
    <p><strong>Soften probability distributions for better knowledge transfer:</strong></p>
    <p style="text-align: center;">
      $p_i = \\frac{\\exp(z_i / T)}{\\sum_j \\exp(z_j / T)}$
    </p>
    <ul>
      <li><strong>$T=1$:</strong> Standard softmax</li>
      <li><strong>$T>1$:</strong> Softer distribution, reveals relative magnitudes</li>
      <li><strong>Typical T:</strong> 3-20 during distillation</li>
      <li><strong>Inference:</strong> Use $T=1$ (standard predictions)</li>
    </ul>

    <h4>Variants and Extensions</h4>

    <h5>Self-Distillation</h5>
    <ul>
      <li><strong>Same architecture:</strong> Student and teacher have same size</li>
      <li><strong>Regularization effect:</strong> Improves generalization</li>
      <li><strong>Born-again networks:</strong> Iteratively distill into same architecture</li>
    </ul>

    <h5>Multi-Teacher Distillation</h5>
    <ul>
      <li><strong>Ensemble knowledge:</strong> Distill from multiple teachers</li>
      <li><strong>Average predictions:</strong> Or learn weighted combination</li>
      <li><strong>Better performance:</strong> Than single teacher</li>
    </ul>

    <h5>Feature-Based Distillation</h5>
    <ul>
      <li><strong>Intermediate layers:</strong> Match hidden representations, not just outputs</li>
      <li><strong>Attention transfer:</strong> Transfer attention maps</li>
      <li><strong>Richer signal:</strong> More supervision from teacher</li>
    </ul>

    <h5>Online Distillation</h5>
    <ul>
      <li><strong>Collaborative learning:</strong> Student and teacher train together</li>
      <li><strong>No pre-trained teacher:</strong> Beneficial when teacher unavailable</li>
      <li><strong>Peer learning:</strong> Multiple students learn from each other</li>
    </ul>

    <h3>Low-Rank Factorization</h3>

    <h4>Matrix Decomposition</h4>
    <p><strong>Decompose large weight matrix into product of smaller matrices.</strong></p>

    <h5>For Fully Connected Layers</h5>
    <p><strong>Original:</strong> $W \\in \\mathbb{R}^{m \\times n}$ with $mn$ parameters</p>
    <p><strong>Factorized:</strong> $W = U \\times V$ where $U \\in \\mathbb{R}^{m \\times k}$, $V \\in \\mathbb{R}^{k \\times n}$</p>
    <p><strong>Parameters:</strong> $k(m+n)$ where $k \\ll \\min(m,n)$</p>
    <p><strong>Compression ratio:</strong> $\\frac{mn}{k(m+n)}$</p>

    <h5>Singular Value Decomposition (SVD)</h5>
    <ul>
      <li><strong>Decomposition:</strong> $W = U\\Sigma V^T$</li>
      <li><strong>Low-rank approximation:</strong> Keep top $k$ singular values</li>
      <li><strong>Optimal:</strong> Minimizes reconstruction error in Frobenius norm</li>
      <li><strong>Apply:</strong> After training, replace layer with factorized version</li>
    </ul>

    <h5>For Convolutional Layers</h5>
    <ul>
      <li><strong>Tucker decomposition:</strong> Factorize 4D tensor (kernels, channels, height, width)</li>
      <li><strong>Depthwise separable convolutions:</strong> Spatial convolution + pointwise convolution</li>
      <li><strong>Parameter reduction:</strong> $k^2 \\times C_{\\text{in}} \\times C_{\\text{out}} \\to k^2 \\times C_{\\text{in}} + C_{\\text{in}} \\times C_{\\text{out}}$</li>
    </ul>

    <h3>Efficient Architecture Design</h3>

    <h4>Mobile-Optimized Architectures</h4>

    <h5>MobileNet</h5>
    <ul>
      <li><strong>Depthwise separable convolutions:</strong> Dramatically reduce parameters and FLOPs</li>
      <li><strong>Width multiplier α:</strong> Scale number of channels (0.25, 0.5, 0.75, 1.0)</li>
      <li><strong>Resolution multiplier ρ:</strong> Scale input resolution</li>
      <li><strong>Trade-off curve:</strong> Accuracy vs latency, pick operating point</li>
    </ul>

    <h5>EfficientNet</h5>
    <ul>
      <li><strong>Compound scaling:</strong> Jointly scale depth, width, resolution</li>
      <li><strong>Neural Architecture Search:</strong> Automated design of efficient blocks</li>
      <li><strong>State-of-the-art:</strong> Best accuracy-efficiency trade-off</li>
      <li><strong>Scaling coefficients:</strong> Principled way to scale models (EfficientNet-B0 to B7)</li>
    </ul>

    <h5>SqueezeNet</h5>
    <ul>
      <li><strong>Fire modules:</strong> Squeeze layer (1×1 convolutions) + expand layer (1×1 and 3×3)</li>
      <li><strong>50x smaller:</strong> Than AlexNet with similar accuracy</li>
    </ul>

    <h3>Combining Compression Techniques</h3>

    <h4>Visual: Compression Pipeline</h4>
    <pre class="code-block">
Stage-by-Stage Compression (Example: ResNet-50)

┌────────────────────────────────────────────────────────┐
│ Original Model: ResNet-50                              │
│ Size: 97.8 MB  |  Params: 25.6M  |  Acc: 76.1%         │
└────────────────────────────────────────────────────────┘
       │
       │ Step 1: Structured Pruning (50% channels)
       ▼
┌────────────────────────────────────────────────────────┐
│ After Pruning                                          │
│ Size: 97.8 MB  |  Params: 12.8M  |  Acc: 75.3%         │
│ Compression: 2.0x params (same storage - FP32)         │
└────────────────────────────────────────────────────────┘
       │
       │ Step 2: Fine-tune pruned model
       ▼
┌────────────────────────────────────────────────────────┐
│ After Fine-tuning                                      │
│ Size: 97.8 MB  |  Params: 12.8M  |  Acc: 75.8%         │
│ Accuracy recovered!                                    │
└────────────────────────────────────────────────────────┘
       │
       │ Step 3: Quantization (FP32 → INT8)
       ▼
┌────────────────────────────────────────────────────────┐
│ After Quantization                                     │
│ Size: 12.2 MB  |  Params: 12.8M  |  Acc: 75.2%         │
│ Compression: 4x storage (INT8 vs FP32)                 │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ FINAL COMPRESSED MODEL                                 │
│ Total Compression: 8.0x (97.8 MB → 12.2 MB)            │
│ Accuracy Loss: 0.9% (76.1% → 75.2%)                    │
│ Speedup: ~4x faster inference on mobile devices        │
└────────────────────────────────────────────────────────┘
    </pre>

    <h4>Compression Pipeline</h4>
    <ol>
      <li><strong>Architecture design:</strong> Start with efficient architecture (MobileNet, EfficientNet)</li>
      <li><strong>Pruning:</strong> Remove 50-80% of weights (structured or unstructured)</li>
      <li><strong>Fine-tune:</strong> Recover accuracy</li>
      <li><strong>Quantization:</strong> FP32 → INT8</li>
      <li><strong>Knowledge distillation (optional):</strong> If smaller architecture needed</li>
    </ol>

    <h4>Real-World Compression Examples</h4>
    
    <h5>BERT-Base Compression</h5>
    <ul>
      <li><strong>Original:</strong> 440 MB, 110M parameters, 100% accuracy baseline</li>
      <li><strong>After distillation (DistilBERT):</strong> 220 MB, 66M params, 97% accuracy (2x compression)</li>
      <li><strong>After quantization:</strong> 55 MB, 66M params, 96.5% accuracy (8x total)</li>
      <li><strong>After pruning + quantization:</strong> 28 MB, 33M params, 95% accuracy (16x total)</li>
    </ul>

    <h5>MobileNetV3 vs ResNet-50</h5>
    <ul>
      <li><strong>ResNet-50:</strong> 98 MB, 25.6M params, 76.1% ImageNet accuracy</li>
      <li><strong>MobileNetV3-Large:</strong> 21 MB, 5.4M params, 75.2% accuracy (4.7x smaller, 5% fewer params)</li>
      <li><strong>MobileNetV3-Large INT8:</strong> 5.3 MB, 5.4M params, 74.8% accuracy (18.5x total compression)</li>
    </ul>

    <h4>Typical Compression Rates</h4>
    <ul>
      <li><strong>Quantization alone:</strong> 4x (FP32→INT8)</li>
      <li><strong>Pruning alone:</strong> 5-10x (50-90% sparsity)</li>
      <li><strong>Distillation alone:</strong> 2-5x (smaller architecture)</li>
      <li><strong>Combined:</strong> 20-50x compression with <1% accuracy loss possible</li>
    </ul>

    <h3>Evaluation Metrics</h3>

    <h4>Model Metrics</h4>
    <ul>
      <li><strong>Model size:</strong> Storage in MB (disk, memory)</li>
      <li><strong>Parameters:</strong> Total number of weights</li>
      <li><strong>FLOPs:</strong> Floating-point operations per inference</li>
      <li><strong>MACs:</strong> Multiply-accumulate operations</li>
    </ul>

    <h4>Runtime Metrics</h4>
    <ul>
      <li><strong>Latency:</strong> Time per inference (milliseconds)</li>
      <li><strong>Throughput:</strong> Inferences per second</li>
      <li><strong>Memory usage:</strong> Peak RAM during inference</li>
      <li><strong>Energy consumption:</strong> Joules per inference (for battery devices)</li>
    </ul>

    <h4>Quality Metrics</h4>
    <ul>
      <li><strong>Accuracy:</strong> Task performance (classification accuracy, mAP, etc.)</li>
      <li><strong>Perplexity:</strong> For language models</li>
      <li><strong>Compression ratio:</strong> Original size / compressed size</li>
      <li><strong>Efficiency:</strong> Accuracy per FLOP, per MB, per ms</li>
    </ul>

    <h3>Practical Deployment</h3>

    <h4>Frameworks and Tools</h4>
    <ul>
      <li><strong>TensorFlow Lite:</strong> Mobile and edge deployment</li>
      <li><strong>PyTorch Mobile:</strong> iOS and Android deployment</li>
      <li><strong>ONNX Runtime:</strong> Cross-platform optimized inference</li>
      <li><strong>TensorRT:</strong> Nvidia GPU optimization</li>
      <li><strong>OpenVINO:</strong> Intel CPU/GPU optimization</li>
      <li><strong>Core ML:</strong> Apple device optimization</li>
    </ul>

    <h4>Hardware Considerations</h4>
    <ul>
      <li><strong>INT8 acceleration:</strong> Most modern hardware (CPUs, GPUs, NPUs)</li>
      <li><strong>Sparse operations:</strong> Nvidia Ampere, specialized accelerators</li>
      <li><strong>Mixed precision:</strong> Tensor Cores on Nvidia GPUs</li>
      <li><strong>Profile on target:</strong> Different devices have different bottlenecks</li>
    </ul>

    <h3>Applications and Impact</h3>

    <h4>Mobile AI</h4>
    <ul>
      <li><strong>Real-time vision:</strong> Object detection, face recognition on smartphones</li>
      <li><strong>On-device NLP:</strong> Keyboard prediction, voice assistants</li>
      <li><strong>Privacy:</strong> Data stays on device</li>
    </ul>

    <h4>Edge Computing</h4>
    <ul>
      <li><strong>IoT devices:</strong> Smart cameras, sensors</li>
      <li><strong>Autonomous vehicles:</strong> Real-time perception and control</li>
      <li><strong>Robotics:</strong> On-robot inference for navigation and manipulation</li>
    </ul>

    <h4>Cloud Optimization</h4>
    <ul>
      <li><strong>Cost reduction:</strong> Serve more requests with same hardware</li>
      <li><strong>Latency:</strong> Faster inference for better user experience</li>
      <li><strong>Energy efficiency:</strong> Reduce data center power consumption</li>
    </ul>

    <h3>The Future of Model Compression</h3>
    <p>As models continue to grow (GPT-4, Gemini, LLaMA-3), compression becomes ever more critical. Emerging directions include: extreme quantization (INT4, ternary networks), neural architecture search for hardware-specific optimization, lottery ticket-inspired training-from-scratch approaches, and compression-aware pre-training. The ultimate goal: democratize AI by making state-of-the-art models accessible on any device, enabling privacy-preserving on-device intelligence, and reducing the environmental impact of large-scale inference. Model compression transforms cutting-edge research into practical deployed systems.</p>
  `,
  codeExamples: [
    {
      language: 'Python',
      code: `import torch
import torch.nn as nn
import torch.quantization as quantization

# === 1. QUANTIZATION ===

# Post-Training Static Quantization
class SimpleNet(nn.Module):
  def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(3, 32, 3)
      self.relu = nn.ReLU()
      self.fc = nn.Linear(32 * 30 * 30, 10)
      self.quant = quantization.QuantStub()
      self.dequant = quantization.DeQuantStub()

  def forward(self, x):
      x = self.quant(x)
      x = self.relu(self.conv1(x))
      x = x.view(x.size(0), -1)
      x = self.fc(x)
      x = self.dequant(x)
      return x

# Prepare model for quantization
model = SimpleNet()
model.eval()

# Specify quantization config
model.qconfig = quantization.get_default_qconfig('fbgemm')
quantization.prepare(model, inplace=True)

# Calibrate with representative data
calibration_data = torch.randn(100, 3, 32, 32)
with torch.no_grad():
  for i in range(10):
      model(calibration_data[i*10:(i+1)*10])

# Convert to quantized model
quantized_model = quantization.convert(model, inplace=False)

# Compare sizes
def print_size_of_model(model, label=""):
  torch.save(model.state_dict(), "temp.p")
  size = os.path.getsize("temp.p") / 1e6
  print(f"{label}: {size:.2f} MB")
  os.remove('temp.p')

print_size_of_model(model, "FP32 Model")
print_size_of_model(quantized_model, "INT8 Quantized Model")

# === 2. PRUNING ===
import torch.nn.utils.prune as prune

model = SimpleNet()

# Unstructured pruning: prune 30% of weights
prune.l1_unstructured(model.conv1, name='weight', amount=0.3)

# Check sparsity
print(f"Sparsity in conv1: {100. * float(torch.sum(model.conv1.weight == 0)) / float(model.conv1.weight.nelement()):.2f}%")

# Make pruning permanent
prune.remove(model.conv1, 'weight')

# Structured pruning: remove entire channels
prune.ln_structured(
  model.conv1,
  name='weight',
  amount=0.5,  # Remove 50% of channels
  n=2,  # L2 norm
  dim=0  # Prune along output channels
)

print(f"Conv1 output channels after pruning: {model.conv1.out_channels}")

# Global pruning across multiple layers
parameters_to_prune = (
  (model.conv1, 'weight'),
  (model.fc, 'weight'),
)

prune.global_unstructured(
  parameters_to_prune,
  pruning_method=prune.L1Unstructured,
  amount=0.5,  # 50% sparsity globally
)`,
      explanation: 'Quantization and pruning implementations showing post-training quantization and structured/unstructured pruning.'
    },
    {
      language: 'Python',
      code: `import torch
import torch.nn as nn
import torch.nn.functional as F

# === 3. KNOWLEDGE DISTILLATION ===

class TeacherModel(nn.Module):
  """Large, accurate model"""
  def __init__(self):
      super().__init__()
      self.layers = nn.Sequential(
          nn.Linear(784, 1200),
          nn.ReLU(),
          nn.Linear(1200, 1200),
          nn.ReLU(),
          nn.Linear(1200, 10)
      )

  def forward(self, x):
      return self.layers(x)

class StudentModel(nn.Module):
  """Small, efficient model"""
  def __init__(self):
      super().__init__()
      self.layers = nn.Sequential(
          nn.Linear(784, 400),
          nn.ReLU(),
          nn.Linear(400, 10)
      )

  def forward(self, x):
      return self.layers(x)

def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.5):
  """
  Combined loss for knowledge distillation

  Args:
      student_logits: Student model outputs
      teacher_logits: Teacher model outputs
      labels: True labels
      temperature: Softmax temperature for soft targets
      alpha: Weight for hard loss (1-alpha for soft loss)
  """
  # Hard loss: cross-entropy with true labels
  hard_loss = F.cross_entropy(student_logits, labels)

  # Soft loss: KL divergence with teacher
  soft_student = F.log_softmax(student_logits / temperature, dim=1)
  soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
  soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)

  # Combined loss
  return alpha * hard_loss + (1 - alpha) * soft_loss

# Training
teacher = TeacherModel()
student = StudentModel()

# Assume teacher is pre-trained
teacher.eval()

optimizer = torch.optim.Adam(student.parameters(), lr=0.001)

# Training loop
x = torch.randn(32, 784)
labels = torch.randint(0, 10, (32,))

with torch.no_grad():
  teacher_logits = teacher(x)

student_logits = student(x)
loss = distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.3)

loss.backward()
optimizer.step()

print(f"Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")
print(f"Student params: {sum(p.numel() for p in student.parameters()):,}")
print(f"Compression ratio: {sum(p.numel() for p in teacher.parameters()) / sum(p.numel() for p in student.parameters()):.2f}x")

# === 4. LOW-RANK FACTORIZATION ===

def low_rank_decomposition(layer, rank_ratio=0.5):
  """Decompose a linear layer using SVD"""
  W = layer.weight.data
  U, S, V = torch.svd(W)

  # Keep only top-k singular values
  k = int(rank_ratio * min(W.shape))
  U_k = U[:, :k]
  S_k = S[:k]
  V_k = V[:, :k]

  # Create two smaller layers
  layer1 = nn.Linear(layer.in_features, k, bias=False)
  layer2 = nn.Linear(k, layer.out_features, bias=True)

  layer1.weight.data = (V_k * S_k).t()
  layer2.weight.data = U_k
  if layer.bias is not None:
      layer2.bias.data = layer.bias.data

  return nn.Sequential(layer1, layer2)

# Original layer
original_layer = nn.Linear(1000, 1000)
print(f"Original params: {sum(p.numel() for p in original_layer.parameters()):,}")

# Compressed layer (rank=500)
compressed_layer = low_rank_decomposition(original_layer, rank_ratio=0.5)
print(f"Compressed params: {sum(p.numel() for p in compressed_layer.parameters()):,}")

# Test equivalence
x = torch.randn(1, 1000)
print(f"Output difference: {(original_layer(x) - compressed_layer(x)).abs().mean():.6f}")`,
      explanation: 'Knowledge distillation for training small models from large teachers, and low-rank matrix factorization for compression.'
    }
  ],
  interviewQuestions: [
    {
      question: 'What is the difference between quantization and pruning?',
      answer: `Quantization reduces numerical precision (FP32 → INT8) to decrease memory and computation while maintaining model structure. Pruning removes less important weights/neurons, creating sparse networks. Quantization: uniform compression, simpler implementation, compatible with standard hardware. Pruning: variable compression ratios, requires sparsity-aware software/hardware for full benefits, can dramatically reduce model size. Both can be combined for maximum compression. Choose quantization for deployment simplicity, pruning for aggressive size reduction.`
    },
    {
      question: 'Explain knowledge distillation and why soft targets help.',
      answer: `Knowledge distillation trains a smaller student model to mimic a larger teacher model by learning from both ground truth labels and teacher predictions. Soft targets (teacher's probability distribution) provide richer information than hard labels: (1) Reveal similarities between classes, (2) Encode uncertainty and confidence, (3) Transfer learned representations more effectively. Temperature scaling in softmax amplifies these differences. Student learns not just correct answers but teacher's reasoning process, often achieving better performance than training directly on hard labels.`
    },
    {
      question: 'What is the lottery ticket hypothesis?',
      answer: `The lottery ticket hypothesis states that dense neural networks contain sparse subnetworks ("winning tickets") that can achieve comparable accuracy to the full network when trained in isolation. Key findings: (1) Random pruning destroys performance, but structured pruning can preserve it, (2) Winning tickets require specific initialization values, (3) Early-bird tickets can be found early in training. Implications: suggests networks are over-parameterized and efficient architectures exist within them. Challenges traditional views on network size requirements.`
    },
    {
      question: 'Compare structured vs unstructured pruning.',
      answer: `Unstructured pruning removes individual weights based on magnitude, creating sparse but irregular patterns. Benefits: fine-grained control, higher compression ratios. Drawbacks: requires specialized sparse computation libraries for speedup. Structured pruning removes entire channels, filters, or layers, maintaining regular computation patterns. Benefits: immediate speedup on standard hardware, simpler implementation. Drawbacks: coarser granularity, potentially lower compression ratios. Choose structured for deployment simplicity, unstructured for maximum compression.`
    },
    {
      question: 'How does quantization-aware training differ from post-training quantization?',
      answer: `Post-training quantization applies after training by calibrating on small dataset - fast but may lose accuracy. Quantization-aware training (QAT) simulates quantization during training with fake quantization operators, allowing model to adapt to reduced precision. QAT benefits: (1) Better accuracy preservation, (2) Learns quantization-friendly representations, (3) Can optimize for specific hardware. Trade-offs: requires retraining, longer development time. Use post-training for quick deployment, QAT for accuracy-critical applications.`
    },
    {
      question: 'What are the trade-offs between different compression techniques?',
      answer: `Compression vs. accuracy: More aggressive compression typically reduces accuracy. Speed vs. size: Some techniques optimize inference speed (quantization), others model size (pruning). Hardware compatibility: Quantization works on most hardware; pruning needs sparse computation support. Development effort: Post-training methods are easier; training-aware methods require more work. Flexibility: Knowledge distillation allows architectural changes; other methods preserve structure. Choose combination based on deployment constraints, accuracy requirements, and development resources.`
    }
  ],
  quizQuestions: [
    {
      id: 'compress1',
      question: 'What does quantization do to a neural network?',
      options: ['Removes weights', 'Reduces numerical precision', 'Changes architecture', 'Adds regularization'],
      correctAnswer: 1,
      explanation: 'Quantization reduces the numerical precision of weights and activations (e.g., FP32 to INT8), significantly reducing model size and speeding up inference.'
    },
    {
      id: 'compress2',
      question: 'In knowledge distillation, what is the "teacher"?',
      options: ['Training algorithm', 'Large pre-trained model', 'Loss function', 'Dataset'],
      correctAnswer: 1,
      explanation: 'The teacher is a large, accurate model whose knowledge is transferred to a smaller student model through soft targets (probability distributions).'
    },
    {
      id: 'compress3',
      question: 'What is an advantage of structured pruning over unstructured?',
      options: ['Higher sparsity', 'Works on standard hardware', 'Better accuracy', 'Easier to implement'],
      correctAnswer: 1,
      explanation: 'Structured pruning removes entire units (channels, filters) producing dense smaller matrices that work efficiently on standard hardware, unlike unstructured pruning which creates sparse matrices requiring special support.'
    }
  ]
};
