import { Topic } from '../../../types';

export const gradientDescent: Topic = {
  id: 'gradient-descent',
  title: 'Gradient Descent & Optimizers',
  category: 'neural-networks',
  description: 'Optimization algorithms that update weights to minimize loss',
  content: `
    <h2>Gradient Descent & Optimizers: The Engines of Learning</h2>
    <p>Gradient descent is the fundamental optimization algorithm powering neural network training. It's an iterative first-order optimization method that adjusts model parameters in the direction that most rapidly decreases the loss function. Understanding gradient descent and its modern variants (optimizers) is essential because they directly control how—and whether—your network learns. The choice of optimizer and its hyperparameters can mean the difference between a model that converges quickly to excellent performance and one that trains slowly, gets stuck, or fails entirely.</p>

    <p>At its core, gradient descent leverages a simple insight from calculus: the gradient $\\nabla L(\\theta)$ points in the direction of steepest increase of the loss function L at parameters $\\theta$. Moving in the opposite direction ($-\\nabla L(\\theta)$) decreases the loss most rapidly. By repeatedly taking small steps downhill, gradient descent navigates the loss landscape toward minima. While conceptually simple, the practical implementation involves numerous subtleties: batch sizes, learning rates, momentum, adaptive learning rates, and learning rate schedules all dramatically impact training success.</p>

    <h3>The Basic Algorithm: Vanilla Gradient Descent</h3>
    <p>The fundamental gradient descent update rule is elegantly simple:</p>

    <p><strong>$\\theta \\leftarrow \\theta - \\eta \\nabla L(\\theta)$</strong></p>

    <p>Where:</p>
    <ul>
      <li><strong>$\\theta$:</strong> Model parameters (all weights and biases in the network)</li>
      <li><strong>$\\eta$:</strong> Learning rate (step size), typically 0.001-0.1</li>
      <li><strong>$\\nabla L(\\theta)$:</strong> Gradient of the loss function with respect to parameters (computed via backpropagation)</li>
      <li><strong>Minus sign:</strong> Move opposite to the gradient (downhill)</li>
    </ul>

    <p>The algorithm iterates: compute gradient → update parameters → repeat until convergence (when gradients become very small) or a maximum number of iterations is reached. The learning rate $\\eta$ controls how large each step is: too large and you overshoot the minimum; too small and convergence is painfully slow.</p>

    <h3>Three Variants: Batch, Stochastic, and Mini-Batch</h3>

    <h4>Batch Gradient Descent (BGD)</h4>
    <p>Compute the gradient using the <strong>entire training dataset</strong> before making a single parameter update:</p>

    <p><strong>$\\nabla L(\\theta) = \\frac{1}{N} \\sum_{i=1}^{N} \\nabla L(\\theta; x_i, y_i)$</strong></p>

    <p>Where N is the total number of training examples. This averages gradients over all examples, providing the most accurate estimate of the true gradient.</p>

    <p><strong>Advantages:</strong></p>
    <ul>
      <li><strong>Stable convergence:</strong> Smooth loss curves, deterministic trajectory toward minimum</li>
      <li><strong>Guaranteed progress:</strong> Each update definitively decreases loss (for convex functions)</li>
      <li><strong>Efficient for small datasets:</strong> Can compute gradient in one pass</li>
    </ul>

    <p><strong>Disadvantages:</strong></p>
    <ul>
      <li><strong>Extremely slow:</strong> For datasets with millions of examples, one update requires processing all data—impractical</li>
      <li><strong>Memory intensive:</strong> Must store gradients for entire dataset simultaneously</li>
      <li><strong>Gets stuck:</strong> No noise to escape poor local minima or saddle points</li>
      <li><strong>No online learning:</strong> Can't incorporate new data without recomputing everything</li>
    </ul>

    <p><strong>Use case:</strong> Only practical for very small datasets (< 10,000 examples) where memory is sufficient and dataset fits easily in RAM.</p>

    <h4>Stochastic Gradient Descent (SGD)</h4>
    <p>Compute gradient using just a <strong>single randomly selected training example</strong> and immediately update:</p>

    <p><strong>$\\nabla L(\\theta) = \\nabla L(\\theta; x_i, y_i)$</strong> for randomly sampled i</p>

    <p>This provides a noisy estimate of the true gradient but allows extremely frequent updates.</p>

    <p><strong>Advantages:</strong></p>
    <ul>
      <li><strong>Very fast updates:</strong> One gradient computation per update means rapid iteration</li>
      <li><strong>Escapes local minima:</strong> Noise helps the optimizer jump out of poor solutions</li>
      <li><strong>Online learning:</strong> Can process streaming data and adapt in real-time</li>
      <li><strong>Memory efficient:</strong> Only processes one example at a time</li>
    </ul>

    <p><strong>Disadvantages:</strong></p>
    <ul>
      <li><strong>Very noisy gradients:</strong> Erratic, zigzag training curves that never fully converge</li>
      <li><strong>Unstable:</strong> May oscillate wildly around minimum without settling</li>
      <li><strong>Slow wall-clock time:</strong> Despite many updates, each is so noisy that actual convergence is slow</li>
      <li><strong>No GPU parallelization:</strong> Processing one example at a time wastes GPU capabilities</li>
    </ul>

    <p><strong>Use case:</strong> Online learning scenarios (streaming data) or when memory is extremely limited. Rarely used in modern deep learning.</p>

    <h4>Mini-Batch Gradient Descent (The Standard Choice)</h4>
    <p>Compute gradient using a <strong>small random subset (batch) of training examples</strong>, typically 32-512 examples:</p>

    <p><strong>$\\nabla L(\\theta) = \\frac{1}{m} \\sum_{i=1}^{m} \\nabla L(\\theta; x_i, y_i)$</strong></p>

    <p>Where m is the mini-batch size. This strikes a balance between accurate gradient estimates and computational efficiency.</p>

    <p><strong>Advantages:</strong></p>
    <ul>
      <li><strong>Efficient GPU utilization:</strong> Batches enable parallel matrix operations, fully leveraging GPU hardware</li>
      <li><strong>Reduced gradient noise:</strong> Averaging over mini-batch smooths estimates compared to single-example SGD</li>
      <li><strong>Reasonable update frequency:</strong> More updates per epoch than batch GD, faster convergence than pure SGD</li>
      <li><strong>Generalization benefits:</strong> Some noise helps avoid overfitting and find flatter minima</li>
    </ul>

    <p><strong>Batch size considerations:</strong></p>
    <ul>
      <li><strong>Small batches (8-32):</strong> More updates per epoch, more noise (regularization), better generalization, but less GPU efficient</li>
      <li><strong>Medium batches (64-128):</strong> Sweet spot for many problems—good balance of speed and stability</li>
      <li><strong>Large batches (256-512+):</strong> Faster training (wall-clock time), more stable, better GPU utilization, but may generalize worse and require learning rate tuning</li>
    </ul>

    <p><strong>Universal choice:</strong> Mini-batch gradient descent is the standard in modern deep learning. When people say "SGD," they almost always mean mini-batch SGD.</p>

    <h3>Advanced Optimizers: Beyond Vanilla Gradient Descent</h3>

    <h4>Momentum: Accelerating Convergence</h4>
    <p><strong>Update rules:</strong></p>
    <ul>
      <li><strong>$v_t = \\beta v_{t-1} + \\nabla L(\\theta)$</strong> (accumulate velocity)</li>
      <li><strong>$\\theta = \\theta - \\eta v_t$</strong> (update using velocity)</li>
    </ul>

    <p>Where $\\beta$ (typically 0.9) is the momentum coefficient, and v is the velocity vector (moving average of gradients). Think of a ball rolling down a hill: momentum accumulates speed in consistent directions while damping oscillations.</p>

    <p><strong>Why it helps:</strong></p>
    <ul>
      <li><strong>Accelerates in consistent directions:</strong> If gradients point the same way across steps, velocity builds up, enabling faster progress</li>
      <li><strong>Dampens oscillations:</strong> In directions where gradients alternate (oscillations), velocity cancels out, stabilizing the trajectory</li>
      <li><strong>Escapes plateaus:</strong> Built-up momentum can carry the optimizer through flat regions</li>
      <li><strong>Better conditioning:</strong> Especially helps for ill-conditioned loss surfaces (elongated valleys)</li>
    </ul>

    <p><strong>Nesterov Momentum (NAG):</strong> A clever variant that "looks ahead" before computing gradients:</p>
    <ul>
      <li><strong>$v_t = \\beta v_{t-1} + \\nabla L(\\theta - \\eta \\beta v_{t-1})$</strong></li>
      <li><strong>$\\theta = \\theta - \\eta v_t$</strong></li>
    </ul>

    <p>By evaluating the gradient at the predicted future position ($\\theta - \\eta \\beta v_{t-1}$) rather than current position, NAG often converges faster and more accurately. Widely used in practice.</p>

    <h4>RMSprop: Adaptive Per-Parameter Learning Rates</h4>
    <p><strong>Update rules:</strong></p>
    <ul>
      <li><strong>$s_t = \\beta s_{t-1} + (1-\\beta)(\\nabla L(\\theta))^2$</strong> (moving average of squared gradients)</li>
      <li><strong>$\\theta = \\theta - \\frac{\\eta \\nabla L(\\theta)}{\\sqrt{s_t} + \\varepsilon}$</strong></li>
    </ul>

    <p>Where $\\beta \\approx 0.9$ and $\\varepsilon \\approx 10^{-8}$ prevents division by zero. RMSprop divides learning rate by the root of the moving average of squared gradients, adapting the learning rate per parameter.</p>

    <p><strong>Key insight:</strong> Parameters with consistently large gradients get smaller effective learning rates (divided by large $\\sqrt{s_t}$), while parameters with small gradients get larger effective learning rates (divided by small $\\sqrt{s_t}$). This automatic per-parameter adaptation helps optimization, especially when parameters have very different scales or update frequencies.</p>

    <p><strong>Advantages:</strong> Works well for non-stationary problems (RNNs) and handles sparse gradients better than plain SGD. Often enables higher base learning rates.</p>

    <h4>Adam: The Modern Default</h4>
    <p><strong>Adam (Adaptive Moment Estimation)</strong> combines momentum and RMSprop, maintaining running averages of both gradients (first moment) and squared gradients (second moment):</p>

    <p><strong>Update rules:</strong></p>
    <ul>
      <li><strong>$m_t = \\beta_1 m_{t-1} + (1-\\beta_1)\\nabla L(\\theta)$</strong> (first moment: momentum)</li>
      <li><strong>$v_t = \\beta_2 v_{t-1} + (1-\\beta_2)(\\nabla L(\\theta))^2$</strong> (second moment: adaptive LR)</li>
      <li><strong>Bias correction:</strong> $\\hat{m}_t = \\frac{m_t}{1-\\beta_1^t}$, $\\hat{v}_t = \\frac{v_t}{1-\\beta_2^t}$</li>
      <li><strong>$\\theta = \\theta - \\eta \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\varepsilon}$</strong></li>
    </ul>

    <p>Default hyperparameters (work well across many tasks): $\\beta_1=0.9$, $\\beta_2=0.999$, $\\eta=0.001$, $\\varepsilon=10^{-8}$</p>

    <p><strong>Why Adam is popular:</strong></p>
    <ul>
      <li><strong>Robust to hyperparameters:</strong> Default settings work well for most problems, minimal tuning required</li>
      <li><strong>Combines best of both worlds:</strong> Momentum for acceleration + adaptive learning rates for per-parameter tuning</li>
      <li><strong>Handles sparse gradients:</strong> Adaptive learning rates help with sparse features (NLP, recommender systems)</li>
      <li><strong>Fast convergence:</strong> Often reaches good solutions faster than SGD+momentum</li>
      <li><strong>Bias correction:</strong> Ensures proper behavior from first iteration despite zero initialization</li>
    </ul>

    <p><strong>Bias correction explained:</strong> $m_t$ and $v_t$ are initialized to zero, biasing them toward zero early in training. Without correction, Adam would take huge initial steps. Corrections $\\hat{m}_t$ and $\\hat{v}_t$ account for this, with the correction effect diminishing as t increases.</p>

    <h4>AdamW: Fixing Weight Decay</h4>
    <p>Standard Adam incorporates weight decay (L2 regularization) by adding it to gradients. However, this interacts poorly with adaptive learning rates, causing inconsistent regularization across parameters. <strong>AdamW</strong> decouples weight decay from gradient updates:</p>

    <p><strong>$\\theta = \\theta - \\eta \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\varepsilon} - \\lambda\\theta$</strong></p>

    <p>Where $\\lambda$ is weight decay coefficient (typically 0.01-0.1). The weight decay term is applied directly to parameters, independent of gradient-based updates. This ensures uniform regularization strength across all parameters.</p>

    <p><strong>Why decoupling matters - concrete example:</strong></p>
    <p>Consider two parameters: parameter A with large historical gradients ($\\hat{v} = 100$) and parameter B with small historical gradients ($\\hat{v} = 1$). With learning rate $\\eta = 0.001$ and weight decay $\\lambda = 0.01$:</p>

    <p><strong>Standard Adam with L2 in gradient:</strong></p>
    <ul>
      <li>Effective learning rate for A: $\\eta/\\sqrt{\\hat{v}_A} = 0.001/\\sqrt{100} = 0.0001$</li>
      <li>Effective learning rate for B: $\\eta/\\sqrt{\\hat{v}_B} = 0.001/\\sqrt{1} = 0.001$</li>
      <li>Weight decay for A: $0.0001 \\times \\lambda \\times \\theta_A$ (weak regularization)</li>
      <li>Weight decay for B: $0.001 \\times \\lambda \\times \\theta_B$ (10× stronger regularization)</li>
      <li><strong>Problem:</strong> Parameters with different gradient histories get inconsistent regularization!</li>
    </ul>

    <p><strong>AdamW with decoupled weight decay:</strong></p>
    <ul>
      <li>Gradient update for A: $\\eta \\times \\hat{m}_A/\\sqrt{\\hat{v}_A}$ (adaptive)</li>
      <li>Gradient update for B: $\\eta \\times \\hat{m}_B/\\sqrt{\\hat{v}_B}$ (adaptive)</li>
      <li>Weight decay for A: $\\lambda \\times \\theta_A$ (consistent)</li>
      <li>Weight decay for B: $\\lambda \\times \\theta_B$ (consistent)</li>
      <li><strong>Solution:</strong> All parameters get uniform regularization regardless of gradient history!</li>
    </ul>

    <p><strong>Advantages over Adam:</strong></p>
    <ul>
      <li><strong>Better generalization:</strong> Proper weight decay improves test performance</li>
      <li><strong>Easier tuning:</strong> Learning rate and weight decay can be optimized independently</li>
      <li><strong>Consistent regularization:</strong> All parameters penalized equally for large magnitudes</li>
      <li><strong>Standard for Transformers:</strong> Used in BERT, GPT, and most modern large language models</li>
    </ul>

    <p><strong>Recommendation:</strong> Use AdamW as default choice for most applications, especially large models where regularization matters.</p>

    <h3>Learning Rate: The Most Important Hyperparameter</h3>
    <p>The learning rate controls step size and is often the single most important hyperparameter for neural network training. Get it right and training converges quickly to good solutions; get it wrong and training fails completely.</p>

    <p><strong>Too high:</strong></p>
    <ul>
      <li>Overshooting: optimizer bounces around minimum without converging</li>
      <li>Divergence: loss increases, weights explode, NaN values appear</li>
      <li>Instability: training loss oscillates wildly</li>
      <li>Symptoms: loss curve shows large spikes, training crashes, gradient norms explode</li>
    </ul>

    <p><strong>Too low:</strong></p>
    <ul>
      <li>Slow convergence: takes forever to reach good solutions</li>
      <li>Getting stuck: insufficient energy to escape poor local minima or saddle points</li>
      <li>Wasted computation: spending hours on training that could finish in minutes with proper learning rate</li>
      <li>Symptoms: loss decreases very slowly, training plateaus early, gradient norms remain small</li>
    </ul>

    <p><strong>Finding a good learning rate:</strong></p>
    <ul>
      <li><strong>Learning rate range test:</strong> Train briefly with exponentially increasing learning rates (e.g., 10⁻⁶ to 10⁻¹), plot loss vs. LR, choose LR from steepest part of curve before divergence</li>
      <li><strong>Grid search:</strong> Try 0.1, 0.01, 0.001, 0.0001 and compare validation performance</li>
      <li><strong>Adaptive optimizers:</strong> Adam/AdamW reduce sensitivity to learning rate choice (but still need reasonable initial value)</li>
      <li><strong>Typical ranges:</strong> 0.001-0.01 for Adam, 0.01-0.1 for SGD with momentum</li>
    </ul>

    <h3>Learning Rate Schedules: Adapting Over Time</h3>
    <p>Fixed learning rates are suboptimal: large rates needed early for fast progress become too large later, preventing fine-tuning. Learning rate schedules adjust LR during training for better convergence.</p>

    <h4>Step Decay</h4>
    <p><strong>$\\eta_t = \\eta_0 \\times \\gamma^{\\lfloor t/k \\rfloor}$</strong></p>

    <p>Reduce LR by factor $\\gamma$ (e.g., 0.1, 0.5) every k epochs. Simple and effective. Example: start at 0.01, multiply by 0.1 every 30 epochs → 0.01, 0.001, 0.0001, ...</p>

    <h4>Exponential Decay</h4>
    <p><strong>$\\eta_t = \\eta_0 \\times e^{-kt}$</strong></p>

    <p>Smooth continuous decay. Less common than step decay but provides gradual reduction without abrupt changes.</p>

    <h4>Cosine Annealing</h4>
    <p><strong>$\\eta_t = \\eta_{\\text{min}} + 0.5(\\eta_{\\text{max}} - \\eta_{\\text{min}})(1 + \\cos(\\pi t/T))$</strong></p>

    <p>Follows cosine curve from $\\eta_{\\text{max}}$ to $\\eta_{\\text{min}}$ over T epochs. Smooth, gradual decay. Popular for training from scratch (ResNet, Transformers). Provides gentle, continuous reduction that often improves final performance.</p>

    <h4>ReduceLROnPlateau</h4>
    <p>Monitor validation metric; when it stops improving for N epochs (patience), reduce LR by factor (e.g., 0.5). Adaptive to training dynamics—automatically adjusts when progress stalls. No need to manually choose decay schedule.</p>

    <h4>Warmup</h4>
    <p>Linearly increase LR from small value to target LR over first few epochs/steps. Essential for Transformer training (BERT, GPT) where random initialization can cause large early gradients. Prevents early instability and improves final convergence.</p>

    <p><strong>Example warmup + cosine schedule:</strong> Linear increase for 5000 steps (warmup), then cosine decay for remaining training. Standard in modern language model training.</p>

    <h3>Practical Considerations and Best Practices</h3>

    <p><strong>Optimizer selection guide:</strong></p>
    <ul>
      <li><strong>Default choice: Adam or AdamW</strong> - Robust, requires minimal tuning, good for most tasks</li>
      <li><strong>Better generalization: SGD + Momentum</strong> - Often achieves slightly better test accuracy than Adam with careful tuning (lower LR, longer training)</li>
      <li><strong>Large-scale training: AdamW</strong> - Standard for Transformers, large language models, proven at scale</li>
      <li><strong>Computer vision: Either</strong> - ResNets trained with SGD+momentum, but Adam works well too</li>
      <li><strong>RNNs/LSTMs: Adam/RMSprop</strong> - Adaptive learning rates handle non-stationarity better</li>
    </ul>

    <p><strong>Batch size guidelines:</strong></p>
    <ul>
      <li><strong>32-64:</strong> Safe default for most problems, good balance of speed and generalization</li>
      <li><strong>128-256:</strong> Better GPU utilization, faster wall-clock training, may need LR tuning</li>
      <li><strong>512+:</strong> Large-scale training (ImageNet, BERT), requires careful learning rate scaling and warmup</li>
      <li><strong>Scaling rule:</strong> When doubling batch size, consider doubling learning rate (with warmup) or training longer</li>
    </ul>

    <p><strong>Convergence diagnostics:</strong></p>
    <ul>
      <li><strong>Monitor gradient norms:</strong> Extremely small (< 10⁻⁶) suggests vanishing gradients or convergence; extremely large (> 100) suggests exploding gradients</li>
      <li><strong>Learning rate sensitivity:</strong> If tiny LR changes cause training to fail, optimization landscape is difficult—consider better architecture, batch norm, or different optimizer</li>
      <li><strong>Validation vs training loss:</strong> If validation loss stops improving while training loss decreases, you're overfitting—use regularization, not optimization changes</li>
    </ul>

    <h3>Common Optimization Challenges</h3>

    <p><strong>Local Minima vs. Saddle Points:</strong></p>
    <p>Early neural network theory worried about local minima. Modern understanding: in high dimensions, local minima are rare; saddle points (points where gradient is zero but not a minimum) are the real problem. Fortunately, momentum-based optimizers naturally escape saddle points by accumulating velocity that carries them through flat regions.</p>

    <p><strong>Plateaus and Ravines:</strong></p>
    <p>Flat regions (plateaus) where gradients are tiny slow training dramatically. Adaptive learning rates (Adam, RMSprop) help by increasing effective step size when gradients are small. Ravines (narrow valleys with steep sides and gentle floor) cause oscillation; momentum helps by accumulating velocity along the valley while damping perpendicular oscillations.</p>

    <p><strong>Non-Convex Optimization:</strong></p>
    <p>Neural network loss surfaces are highly non-convex (multiple minima, saddle points, plateaus). Unlike convex optimization where gradient descent guarantees global minimum, neural networks only guarantee finding some local minimum. Surprisingly, this is often fine: many local minima achieve similar performance, and the optimization landscape is surprisingly well-behaved for overparameterized networks.</p>

    <h3>Modern Developments and Research Directions</h3>

    <p><strong>Second-order methods:</strong> Use curvature information (second derivatives, Hessian matrix) for better updates. Examples: Newton's method, L-BFGS. Theoretically superior but computationally prohibitive for large networks. Research on approximations (K-FAC, Shampoo) shows promise.</p>

    <p><strong>Layer-wise adaptive learning rates:</strong> Different layers might benefit from different learning rates (early layers learn slower). Research on layer-wise LR adaptation (LARS, LAMB) enables larger batch training.</p>

    <p><strong>Gradient noise:</strong> Adding noise to gradients can improve generalization. Related to implicit regularization of SGD's inherent noise.</p>

    <p><strong>Meta-learning optimizers:</strong> Using neural networks to learn optimization algorithms. Research area with interesting results but not yet practical for large-scale deployment.</p>
  `,
  codeExamples: [
    {
      language: 'Python',
      code: `import torch
import torch.nn as nn
import torch.optim as optim

# Create sample model
model = nn.Sequential(
  nn.Linear(10, 64),
  nn.ReLU(),
  nn.Linear(64, 32),
  nn.ReLU(),
  nn.Linear(32, 1)
)

# Compare different optimizers
optimizers = {
  'SGD': optim.SGD(model.parameters(), lr=0.01),
  'SGD+Momentum': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
  'RMSprop': optim.RMSprop(model.parameters(), lr=0.001),
  'Adam': optim.Adam(model.parameters(), lr=0.001),
  'AdamW': optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
}

# Sample data
X = torch.randn(100, 10)
y = torch.randn(100, 1)

# Train with each optimizer
for name, optimizer in optimizers.items():
  # Reset model
  for param in model.parameters():
      nn.init.xavier_uniform_(param) if param.dim() > 1 else nn.init.zeros_(param)

  criterion = nn.MSELoss()
  losses = []

  for epoch in range(100):
      optimizer.zero_grad()
      output = model(X)
      loss = criterion(output, y)
      loss.backward()
      optimizer.step()

      if epoch % 20 == 0:
          losses.append(loss.item())

  print(f"{name}: Final loss = {losses[-1]:.4f}")`,
      explanation: 'Compares different optimizers (SGD, SGD+Momentum, RMSprop, Adam, AdamW) on the same task. Adam typically converges faster and more reliably than plain SGD.'
    },
    {
      language: 'Python',
      code: `import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))

# Learning rate scheduling examples
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 1. Step Decay: reduce LR by 0.5 every 30 epochs
scheduler_step = StepLR(optimizer, step_size=30, gamma=0.5)

# 2. Cosine Annealing: smooth decay following cosine curve
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# 3. ReduceLROnPlateau: reduce when validation loss plateaus
scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# Training loop with scheduler
X_train, y_train = torch.randn(100, 10), torch.randn(100, 1)
X_val, y_val = torch.randn(20, 10), torch.randn(20, 1)
criterion = nn.MSELoss()

print("Learning Rate Schedule:")
for epoch in range(100):
  # Training
  optimizer.zero_grad()
  loss = criterion(model(X_train), y_train)
  loss.backward()
  optimizer.step()

  # Validation
  with torch.no_grad():
      val_loss = criterion(model(X_val), y_val)

  # Update learning rate
  scheduler_cosine.step()  # For step/cosine, call after optimizer.step()
  # scheduler_plateau.step(val_loss)  # For plateau, pass validation metric

  if epoch % 20 == 0:
      current_lr = optimizer.param_groups[0]['lr']
      print(f"Epoch {epoch}: LR = {current_lr:.6f}, Loss = {loss.item():.4f}")`,
      explanation: 'Demonstrates learning rate scheduling strategies. Reducing learning rate during training helps fine-tune weights and achieve better convergence. Cosine annealing is popular for training from scratch.'
    }
  ],
  interviewQuestions: [
    {
      question: 'Explain the difference between batch, stochastic, and mini-batch gradient descent.',
      answer: '**Batch Gradient Descent** computes gradients using the entire training dataset before making a single weight update. This approach provides the most accurate gradient estimate at each step, leading to stable convergence and guaranteed progress toward the minimum (for convex functions). However, it can be extremely slow for large datasets since each update requires processing all training examples, and it may require too much memory to store gradients for the entire dataset.\n\n**Stochastic Gradient Descent (SGD)** updates weights after computing gradients from just a single training example. This makes each update very fast and allows training to begin immediately without waiting to process the entire dataset. SGD also provides a form of regularization through its inherent noise, which can help escape local minima. However, the gradient estimates are very noisy, leading to zigzag convergence patterns and potential instability near the minimum.\n\n**Mini-batch Gradient Descent** strikes a balance by computing gradients using small subsets (typically 32-512 examples) of the training data. This approach combines the benefits of both methods: more stable gradients than SGD but faster updates than batch gradient descent. Mini-batches also enable efficient use of parallel hardware (GPUs) since matrix operations on mini-batches can be vectorized effectively.\n\nPractically, **mini-batch gradient descent** is the standard choice for deep learning because it provides good gradient estimates while being computationally efficient. The batch size becomes a hyperparameter that affects both training dynamics and computational efficiency. Smaller batches provide more frequent updates and regularization but noisier gradients, while larger batches provide more accurate gradients but fewer updates per epoch. Modern optimizers like Adam work particularly well with mini-batch gradients, and techniques like gradient accumulation allow simulating larger batch sizes when memory is limited.'
    },
    {
      question: 'What is momentum and why does it help optimization?',
      answer: '**Momentum** is a technique that accelerates gradient descent by accumulating a moving average of past gradients, helping the optimizer build velocity in consistent directions while dampening oscillations. The momentum update rule is: **v_t = βv_{t-1} + ∇θJ(θ)** and **θ = θ - αv_t**, where **v_t** is the velocity vector, **β** (typically 0.9) is the momentum coefficient, **α** is the learning rate, and **∇θJ(θ)** is the current gradient. This creates a "ball rolling down a hill" effect where the optimizer gains speed in directions of consistent gradients.\n\nMomentum helps overcome several optimization challenges. In **ill-conditioned** optimization landscapes (where the loss surface has very different curvatures in different directions), standard gradient descent oscillates slowly across narrow valleys. Momentum accumulates velocity along the consistent direction (down the valley) while canceling out oscillations in perpendicular directions, leading to faster convergence. This is particularly valuable in neural networks where loss surfaces often have this elongated valley structure.\n\nThe technique also helps **escape saddle points** and small local minima. When gradient descent gets stuck at points where gradients are small, accumulated momentum can carry the optimizer through these regions. Additionally, momentum provides some **noise averaging** effect, smoothing out noisy gradient estimates that are common in mini-batch training. This leads to more stable training and often better final performance.\n\n**Nesterov momentum** is an improved variant that "looks ahead" before computing gradients: **v_t = βv_{t-1} + ∇θJ(θ - αβv_{t-1})** and **θ = θ - αv_t**. By evaluating gradients at the anticipated future position rather than the current position, Nesterov momentum often converges faster and more accurately. Modern optimizers like Adam incorporate momentum-like mechanisms, and momentum remains a fundamental technique for accelerating neural network training, particularly important for training deep networks where optimization landscapes can be very challenging.'
    },
    {
      question: 'How does Adam optimizer work and why is it popular?',
      answer: '**Adam (Adaptive Moment Estimation)** combines the benefits of momentum and adaptive learning rates by maintaining separate running averages of both gradients (first moment) and squared gradients (second moment). The algorithm computes: **m_t = β₁m_{t-1} + (1-β₁)∇θJ(θ)** (momentum), **v_t = β₂v_{t-1} + (1-β₂)(∇θJ(θ))²** (adaptive learning rate), and then updates parameters using **θ = θ - α(m̂_t / (√v̂_t + ε))**, where **m̂_t** and **v̂_t** are bias-corrected estimates and **ε** prevents division by zero.\n\nThe **first moment estimate** **m_t** provides momentum-like acceleration by accumulating gradients with exponential decay (typically **β₁ = 0.9**). The **second moment estimate** **v_t** tracks the magnitude of recent gradients with **β₂ = 0.999**, allowing the optimizer to adapt learning rates per parameter. Parameters with consistently large gradients get smaller effective learning rates, while parameters with small gradients get larger effective learning rates. This adaptive behavior helps balance learning across different parameters and dimensions.\n\n**Bias correction** is crucial because **m_t** and **v_t** are initialized to zero, causing them to be biased toward zero early in training. The corrections **m̂_t = m_t/(1-β₁ᵗ)** and **v̂_t = v_t/(1-β₂ᵗ)** account for this initialization bias, ensuring proper behavior from the first iteration. Without bias correction, Adam would take very large steps initially, potentially destabilizing training.\n\nAdam\'s popularity stems from its **robustness and ease of use**. It typically works well with default hyperparameters (**α = 0.001, β₁ = 0.9, β₂ = 0.999, ε = 1e-8**) across a wide range of problems, requiring minimal tuning. It handles sparse gradients well, adapts to different parameter scales automatically, and provides both momentum and adaptive learning rate benefits. While newer optimizers like AdamW address some of Adam\'s limitations (like weight decay issues), Adam remains the default choice for many practitioners due to its reliability and broad applicability across different neural network architectures and tasks.'
    },
    {
      question: 'What happens if the learning rate is too high or too low?',
      answer: 'When the **learning rate is too high**, the optimizer takes steps that are too large, causing several problems. The most obvious issue is **overshooting** the minimum: instead of converging to the optimal point, the algorithm bounces around or even diverges away from it. This manifests as training loss that oscillates wildly, increases over time, or jumps to very large values (NaN). In extreme cases, the gradients themselves can explode, leading to numerical instability where weights become infinite or undefined.\n\nHigh learning rates also cause **poor convergence near the minimum**. Even if the algorithm approaches the optimal region, large steps prevent it from settling into the minimum precisely. Instead, it perpetually overshoots and oscillates around the target, never achieving the best possible loss value. This results in suboptimal final performance and unstable training curves that don\'t smooth out over time.\n\nWhen the **learning rate is too low**, the primary problem is **extremely slow convergence**. The optimizer takes tiny steps toward the minimum, requiring many more iterations to reach acceptable performance levels. Training that could complete in hours might take days or weeks. In practical scenarios with limited time and computational resources, this effectively prevents the model from learning adequately.\n\nLow learning rates also make the optimizer susceptible to getting **stuck in local minima** or **saddle points**. Without sufficient step size to escape these suboptimal regions, the algorithm may prematurely stop improving even though better solutions exist. This is particularly problematic in neural networks where loss landscapes contain many such traps. Additionally, very slow learning makes it difficult to distinguish between a model that\'s still improving versus one that\'s genuinely stuck.\n\n**Finding the optimal learning rate** is crucial and often requires experimentation. Techniques like **learning rate schedules** (starting high and decreasing over time), **learning rate range tests** (systematically trying different rates), and **adaptive optimizers** (like Adam) help address these challenges. The goal is finding the largest learning rate that maintains stable training while enabling fast convergence to good solutions.'
    },
    {
      question: 'Why do we use learning rate scheduling?',
      answer: '**Learning rate scheduling** adjusts the learning rate during training to optimize the balance between convergence speed and final performance. Early in training, when weights are far from optimal, larger learning rates enable fast progress toward better regions of the loss landscape. However, as training progresses and the model approaches good solutions, smaller learning rates become necessary for fine-tuning and achieving the best possible performance without overshooting.\n\nThe most common approach is **step decay** or **exponential decay**, where the learning rate decreases by a fixed factor (e.g., 0.1) at predetermined epochs or when validation performance plateaus. This allows rapid initial learning followed by careful refinement. **Cosine annealing** gradually reduces the learning rate following a cosine curve, providing smooth transitions and often better final performance than abrupt step changes.\n\n**Warmup** scheduling addresses initialization issues in deep networks. Starting with a very small learning rate and gradually increasing it over the first few epochs helps stabilize training when weights are randomly initialized and gradients might be unreliable. This is particularly important for large models or when using techniques like batch normalization that can create unstable training dynamics initially.\n\n**Adaptive scheduling** responds to training dynamics in real-time. **ReduceLROnPlateau** monitors validation metrics and decreases the learning rate when improvement stagnates, allowing automatic adjustment without manual tuning. **Cyclical learning rates** alternate between low and high values, helping escape local minima and often finding better solutions than monotonic schedules.\n\nLearning rate scheduling is essential because **fixed learning rates** are suboptimal: rates that work well initially become too large later, while rates that work well for fine-tuning are too small for initial learning. Modern training often combines scheduling with adaptive optimizers like Adam, where the base learning rate is scheduled while the optimizer handles parameter-specific adaptations. Proper scheduling can significantly improve final model performance and training stability, making it a crucial component of deep learning pipelines.'
    },
    {
      question: 'What is the difference between Adam and AdamW?',
      answer: 'The key difference between **Adam** and **AdamW** lies in how they handle **weight decay** (L2 regularization). Standard Adam incorporates weight decay by adding the regularization term directly to the gradients: **gradient = gradient + λ * weights**, where **λ** is the weight decay coefficient. However, this approach interacts poorly with Adam\'s adaptive learning rate mechanism, leading to inconsistent regularization strength across parameters with different gradient magnitudes.\n\nIn standard Adam with weight decay, parameters with small adaptive learning rates (due to large historical gradients) experience weaker regularization, while parameters with large adaptive learning rates experience stronger regularization. This coupling means that weight decay doesn\'t uniformly encourage smaller weights across all parameters, reducing its effectiveness and making it difficult to tune properly.\n\n**AdamW (Adam with decoupled Weight decay)** solves this by separating weight decay from gradient-based updates. Instead of modifying gradients, AdamW applies weight decay directly to the parameters: **θ = θ - α(m̂_t / (√v̂_t + ε)) - α_wd * θ**, where **α_wd** is the weight decay rate applied independently of the adaptive gradient update. This ensures uniform regularization strength across all parameters regardless of their gradient histories.\n\nThis decoupling provides several practical benefits: **better generalization** through more consistent regularization, **easier hyperparameter tuning** since weight decay and learning rate can be optimized independently, and **improved performance** particularly on tasks where regularization is important. AdamW often achieves better results than standard Adam, especially for transformer models and other large architectures where proper regularization is crucial.\n\n**Usage recommendations**: Use AdamW as the default choice for most applications, especially when training large models or when regularization is important for generalization. The hyperparameters remain similar to Adam (learning rate, β₁, β₂), with the addition of the weight decay coefficient (typically 0.01-0.1). AdamW has become the standard optimizer for training large language models and many state-of-the-art computer vision models due to its superior regularization properties.'
    }
  ],
  quizQuestions: [
    {
      id: 'gd-q1',
      question: 'Your model trains very slowly, taking many epochs to converge. The learning curve is smooth but progress is minimal. What is the most likely issue?',
      options: [
        'Learning rate is too high',
        'Learning rate is too low',
        'Batch size is too large',
        'Model is too complex'
      ],
      correctAnswer: 1,
      explanation: 'Slow, smooth convergence with minimal progress indicates learning rate is too low. Weights update in tiny steps. Solution: increase learning rate (try 10x larger) or use adaptive optimizer like Adam.'
    },
    {
      id: 'gd-q2',
      question: 'Why is Adam optimizer more popular than plain SGD for deep learning?',
      options: [
        'Adam is always faster to compute',
        'Adam adapts learning rate per parameter and includes momentum, requiring less tuning',
        'Adam guarantees finding global minimum',
        'Adam uses less memory'
      ],
      correctAnswer: 1,
      explanation: 'Adam combines momentum (first moment) and adaptive learning rates (second moment), making it robust across tasks with minimal hyperparameter tuning. Works well out-of-the-box, though SGD+momentum can generalize better with careful tuning.'
    },
    {
      id: 'gd-q3',
      question: 'What is the purpose of learning rate warmup in transformer training?',
      options: [
        'To save computation time',
        'To gradually increase learning rate at start, preventing early instability from large gradients',
        'To reduce memory usage',
        'To improve final accuracy'
      ],
      correctAnswer: 1,
      explanation: 'Warmup linearly increases learning rate for the first few epochs/steps. With random initialization, early gradients can be very large. Starting with small LR and warming up prevents exploding gradients and instability. Standard practice for transformers (BERT, GPT).'
    }
  ]
};
