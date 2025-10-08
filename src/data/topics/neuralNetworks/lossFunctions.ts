import { Topic } from '../../../types';

export const lossFunctions: Topic = {
  id: 'loss-functions',
  title: 'Loss Functions',
  category: 'neural-networks',
  description: 'Objective functions that quantify prediction error and guide learning',
  content: `
    <h2>Loss Functions: The Objectives That Drive Learning</h2>
    <p>Loss functions (also called objective functions, cost functions, or error functions) are the mathematical foundations that guide neural network learning. They quantify the difference between a model's predictions and the true target values, providing a scalar measure of "wrongness" that gradient descent seeks to minimize. The choice of loss function is fundamental—it directly determines what the network optimizes for, how gradients flow during backpropagation, and ultimately what the model learns. Using the wrong loss function for your task can make training fail entirely or produce a model that optimizes for the wrong objective.</p>

    <p>Loss functions must be <strong>differentiable</strong> (at least almost everywhere) to enable gradient-based optimization. They should be <strong>aligned with the evaluation metric</strong> you actually care about, though perfect alignment isn't always possible. They must provide <strong>useful gradient signals</strong>—gradients that guide the model toward better solutions without vanishing or exploding. Understanding the mathematical properties, use cases, and pitfalls of different loss functions is essential for successfully training neural networks.</p>

    <h3>Regression Loss Functions: Continuous Value Prediction</h3>

    <h4>Mean Squared Error (MSE) / L2 Loss</h4>
    <p><strong>$$L_{\\text{MSE}} = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2$$</strong></p>

    <p>MSE is the most common regression loss, computing the average squared difference between predictions ŷ and targets y. The squaring operation makes MSE highly sensitive to large errors—an error of 10 contributes 100 to the loss, while ten errors of 1 each contribute only 10 total. This quadratic penalty strongly encourages the model to avoid large mistakes.</p>

    <p><strong>Mathematical properties:</strong></p>
    <ul>
      <li><strong>Gradient:</strong> $\\frac{\\partial L}{\\partial \\hat{y}_i} = \\frac{2(\\hat{y}_i - y_i)}{n}$, proportional to error magnitude—larger errors get stronger correction signals</li>
      <li><strong>Smooth everywhere:</strong> No discontinuities, making optimization straightforward</li>
      <li><strong>Convex for linear models:</strong> Single global minimum, guaranteed convergence with gradient descent</li>
      <li><strong>Corresponds to Gaussian likelihood:</strong> Minimizing MSE is equivalent to maximum likelihood estimation assuming Gaussian errors</li>
    </ul>

    <p><strong>Strengths:</strong></p>
    <ul>
      <li><strong>Fast convergence:</strong> Large errors produce large gradients, enabling quick correction</li>
      <li><strong>Penalizes outliers heavily:</strong> Appropriate when large errors are catastrophic</li>
      <li><strong>Standard choice:</strong> Works well for most regression problems</li>
      <li><strong>Stable gradients:</strong> Smooth, well-behaved optimization</li>
    </ul>

    <p><strong>Weaknesses:</strong></p>
    <ul>
      <li><strong>Very sensitive to outliers:</strong> A few outliers can dominate the loss, distorting the model</li>
      <li><strong>Assumes Gaussian errors:</strong> Not ideal when error distribution is heavy-tailed</li>
      <li><strong>Units matter:</strong> Loss value depends on target scale (error of 1000 in prices vs. error of 1 in normalized values)</li>
    </ul>

    <p><strong>Use when:</strong> Standard regression tasks, outliers are genuine errors (not valid data), you want to heavily penalize large mistakes, Gaussian error assumptions are reasonable.</p>

    <h4>Mean Absolute Error (MAE) / L1 Loss</h4>
    <p><strong>$$L_{\\text{MAE}} = \\frac{1}{n} \\sum_{i=1}^{n} |y_i - \\hat{y}_i|$$</strong></p>

    <p>MAE computes the average absolute difference between predictions and targets. Unlike MSE, it treats all errors linearly—an error of 10 contributes exactly 10× as much as an error of 1. This makes MAE more robust to outliers: extreme values don't dominate the loss as they do with MSE.</p>

    <p><strong>Mathematical properties:</strong></p>
    <ul>
      <li><strong>Gradient:</strong> $\\frac{\\partial L}{\\partial \\hat{y}_i} = \\frac{\\text{sign}(\\hat{y}_i - y_i)}{n}$, constant magnitude regardless of error size</li>
      <li><strong>Discontinuous gradient at zero:</strong> The derivative doesn't exist at $\\hat{y}_i = y_i$, can cause optimization issues</li>
      <li><strong>Corresponds to Laplace likelihood:</strong> Minimizing MAE assumes Laplace (double exponential) error distribution</li>
      <li><strong>Median predictor:</strong> MAE encourages predicting the conditional median, not mean</li>
    </ul>

    <p><strong>Strengths:</strong></p>
    <ul>
      <li><strong>Robust to outliers:</strong> Outliers contribute linearly, not quadratically</li>
      <li><strong>Treats all errors equally:</strong> Appropriate when all mistakes matter the same</li>
      <li><strong>More interpretable:</strong> Loss value in same units as targets</li>
    </ul>

    <p><strong>Weaknesses:</strong></p>
    <ul>
      <li><strong>Slower convergence:</strong> Constant gradients mean less urgency to fix large errors</li>
      <li><strong>Gradient discontinuity:</strong> Optimization can be unstable near optimum</li>
      <li><strong>Less common:</strong> Libraries may have worse support/optimization than MSE</li>
    </ul>

    <p><strong>Use when:</strong> Data contains outliers that are valid (not errors), you want robust regression, all errors should be weighted equally, predicting the median is appropriate.</p>

    <h4>Huber Loss / Smooth L1 Loss</h4>
    <p><strong>$$L_{\\text{Huber}}(y, \\hat{y}) = \\begin{cases} \\frac{1}{2}(y - \\hat{y})^2 & \\text{if } |y - \\hat{y}| \\leq \\delta \\\\ \\delta(|y - \\hat{y}| - \\frac{1}{2}\\delta) & \\text{otherwise} \\end{cases}$$</strong></p>

    <p>Huber loss combines the best of MSE and MAE: quadratic for small errors (smooth, fast convergence) and linear for large errors (robust to outliers). The threshold δ determines the transition point. This gives smooth gradients everywhere (unlike MAE) while limiting outlier impact (unlike MSE).</p>

    <p><strong>Mathematical properties:</strong></p>
    <ul>
      <li><strong>Gradient:</strong> Proportional to error for small errors, constant for large errors</li>
      <li><strong>Smooth everywhere:</strong> Continuously differentiable (unlike MAE)</li>
      <li><strong>δ parameter:</strong> Controls robustness vs. convergence speed trade-off</li>
    </ul>

    <p><strong>Tuning δ:</strong> Smaller δ makes Huber more like MAE (more robust, slower convergence); larger δ makes it more like MSE (less robust, faster convergence). Common heuristic: set δ to the 90th percentile of absolute errors from an initial MSE model.</p>

    <p><strong>Use when:</strong> Data has outliers but you still want fast convergence, you want robustness without MAE's gradient discontinuity, object detection (Faster R-CNN uses Smooth L1 for bounding box regression).</p>

    <h3>Classification Loss Functions: Discrete Label Prediction</h3>

    <h4>Binary Cross-Entropy (BCE) / Log Loss</h4>
    <p><strong>$$L_{\\text{BCE}} = -\\frac{1}{n} \\sum_{i=1}^{n} [y_i \\log(p_i) + (1-y_i) \\log(1-p_i)]$$</strong></p>

    <p>Where $y_i \\in \\{0, 1\\}$ is the true binary label and $p_i \\in (0, 1)$ is the predicted probability (from sigmoid). BCE measures the divergence between the true distribution (all probability mass on the correct class) and the predicted distribution. It heavily penalizes confident wrong predictions while being gentle on uncertain predictions.</p>

    <p><strong>Why this form?</strong> BCE derives from maximum likelihood estimation for Bernoulli-distributed data. Minimizing BCE is equivalent to maximizing the likelihood of observing the true labels given the model's predicted probabilities. This provides a principled statistical foundation.</p>

    <p><strong>Key insight:</strong> When $y_i = 1$, the loss is $-\\log(p_i)$; when $y_i = 0$, the loss is $-\\log(1-p_i)$. As $p_i \\to 0$ (confident wrong prediction for positive class), $-\\log(p_i) \\to \\infty$—the loss explodes, strongly penalizing the error. As $p_i \\to 1$ (confident correct prediction), $-\\log(p_i) \\to 0$—minimal loss. This asymmetry ensures the model learns to produce calibrated probabilities.</p>

    <p><strong>Gradient with sigmoid:</strong> When paired with sigmoid activation, the gradient simplifies beautifully: $\\frac{\\partial L}{\\partial z} = p - y$ (where $z$ is pre-activation). This clean gradient is why sigmoid+BCE is the standard pairing for binary classification.</p>

    <p><strong>Use when:</strong> Binary classification (spam detection, medical diagnosis, sentiment analysis), you need probability outputs, evaluation metrics are based on probabilities or decisions.</p>

    <h4>Categorical Cross-Entropy</h4>
    <p><strong>$$L_{\\text{CE}} = -\\frac{1}{n} \\sum_{i=1}^{n} \\sum_{j=1}^{c} y_{ij} \\log(p_{ij})$$</strong></p>

    <p>Where $y_{ij}$ is the one-hot encoded true label ($y_{ij} = 1$ for correct class $j$, 0 otherwise) and $p_{ij}$ is the predicted probability for class $j$ (from softmax). For the true class $c$, this simplifies to $-\\log(p_{ic})$—only the predicted probability for the true class matters.</p>

    <p><strong>Softmax + Cross-Entropy:</strong> This pairing is mathematically optimal for multi-class classification. Softmax ensures outputs form a valid probability distribution (sum to 1, all positive), and cross-entropy measures the divergence from the true distribution. The combined gradient is simply p - y (predicted probabilities minus true one-hot).</p>

    <p><strong>Numerical stability:</strong> Computing softmax then log(softmax) separately can cause numerical issues (overflow in exp, undefined log(0)). Modern frameworks combine these operations using the log-sum-exp trick for stability. Always use built-in implementations (nn.CrossEntropyLoss in PyTorch) that handle this.</p>

    <p><strong>Use when:</strong> Multi-class classification (ImageNet, text classification), mutually exclusive classes, you need class probabilities, standard classification evaluation metrics.</p>

    <h4>Sparse Categorical Cross-Entropy</h4>
    <p>Mathematically identical to categorical cross-entropy but accepts integer class labels instead of one-hot encoding. For a true class $c$, computes $-\\log(p_c)$ directly. This is more memory-efficient when you have many classes—storing integers (4 bytes each) vs. one-hot vectors (4 bytes × num_classes).</p>

    <p><strong>Use when:</strong> Multi-class classification with many classes (ImageNet's 1000 classes, NLP with 50K+ word vocabularies), memory is constrained, you want cleaner code (no one-hot encoding needed).</p>

    <h4>Focal Loss: Tackling Class Imbalance</h4>
    <p><strong>$$L_{\\text{FL}} = -\\alpha_t(1-p_t)^\\gamma \\log(p_t)$$</strong></p>

    <p>Where $p_t$ is the predicted probability for the true class, $\\alpha$ is a weighting factor, and $\\gamma$ (gamma, typically 2) is the focusing parameter. The key innovation is the modulating factor $(1-p_t)^\\gamma$ that down-weights easy examples.</p>

    <p><strong>How it addresses imbalance:</strong> In severely imbalanced datasets (e.g., 99% background, 1% objects in detection), the abundant easy examples (background patches correctly classified with high confidence) dominate training, overshadowing the rare hard examples (actual objects, ambiguous cases). Focal loss reduces the loss contribution from easy examples while maintaining full loss for hard examples.</p>

    <p><strong>Focusing mechanism:</strong></p>
    <ul>
      <li>Easy example ($p_t = 0.9$): $(1-0.9)^2 = 0.01$, loss reduced by 99%</li>
      <li>Hard example ($p_t = 0.5$): $(1-0.5)^2 = 0.25$, loss reduced by 75%</li>
      <li>Very hard example ($p_t = 0.1$): $(1-0.1)^2 = 0.81$, loss reduced by 19%</li>
    </ul>

    <p>This automatic reweighting focuses training on examples the model struggles with.</p>

    <p><strong>γ parameter:</strong> Controls focusing strength. $\\gamma=0$ gives standard cross-entropy; $\\gamma=2$ is typical; higher $\\gamma$ focuses more aggressively on hard examples but can destabilize training.</p>

    <p><strong>Use when:</strong> Severe class imbalance (object detection, medical diagnosis of rare diseases), you want to focus on hard examples, standard weighted loss isn't sufficient.</p>

    <h3>Embedding and Metric Learning Losses</h3>

    <h4>Contrastive Loss</h4>
    <p><strong>$$L = (1-y) \\times \\frac{1}{2}D^2 + y \\times \\frac{1}{2}\\max(\\text{margin} - D, 0)^2$$</strong></p>

    <p>Where $D$ is the Euclidean distance between embeddings, $y \\in \\{0, 1\\}$ indicates whether the pair is similar ($y=1$) or dissimilar ($y=0$), and margin is a hyperparameter. For similar pairs, loss increases with distance (pull together). For dissimilar pairs, loss only applies if distance < margin (push apart until margin is reached, then stop caring).</p>

    <p><strong>Use when:</strong> Learning embeddings where similar items should be close, dissimilar items should be far apart. Face verification (same person vs. different people), signature verification, Siamese networks.</p>

    <h4>Triplet Loss</h4>
    <p><strong>L = max(D(a,p) - D(a,n) + margin, 0)</strong></p>

    <p>Where a is an anchor embedding, p is a positive example (same class), n is a negative example (different class), and D is distance. The loss ensures anchors are closer to positives than to negatives by at least margin. Unlike contrastive loss, triplet loss considers relative distances (anchor-to-positive vs. anchor-to-negative) rather than absolute distances.</p>

    <p><strong>Triplet mining:</strong> Selecting good triplets is crucial. Random triplets are often too easy (many satisfy the constraint, providing no learning signal). <strong>Hard negative mining</strong> (selecting negatives close to the anchor) and <strong>semi-hard mining</strong> (negatives farther than positive but within margin) provide better training signal.</p>

    <p><strong>Use when:</strong> Face recognition (FaceNet), person re-identification, learning similarity metrics, you have natural groupings (classes, IDs) for forming triplets.</p>

    <h3>Specialized Losses for Specific Domains</h3>

    <h4>Dice Loss / F1 Loss</h4>
    <p><strong>Dice = 2|X ∩ Y| / (|X| + |Y|)</strong>, <strong>L_Dice = 1 - Dice</strong></p>

    <p>Where X is predicted segmentation, Y is ground truth. Dice coefficient measures overlap between prediction and target. Dice loss works directly with the evaluation metric (Dice score) used in segmentation, making it well-aligned with the actual objective. It handles class imbalance naturally—focusing on overlap rather than pixel-wise accuracy.</p>

    <p><strong>Use when:</strong> Semantic segmentation, medical image segmentation (tumor detection), instance segmentation. Often combined with BCE: L_total = L_BCE + L_Dice.</p>

    <h4>IoU Loss (Intersection over Union)</h4>
    <p><strong>IoU = Area(box₁ ∩ box₂) / Area(box₁ ∪ box₂)</strong>, <strong>L_IoU = 1 - IoU</strong></p>

    <p>For bounding box regression in object detection. Directly optimizes the evaluation metric (IoU), ensuring the loss aligns with what's measured. Variants include <strong>GIoU</strong> (Generalized IoU), <strong>DIoU</strong> (Distance IoU), and <strong>CIoU</strong> (Complete IoU) that address limitations of basic IoU loss.</p>

    <p><strong>Use when:</strong> Object detection (YOLO, Faster R-CNN), instance segmentation, any task involving bounding boxes where IoU is the evaluation metric.</p>

    <h3>Practical Loss Function Selection Guide</h3>

    <p><strong>For Regression:</strong></p>
    <ul>
      <li><strong>Standard case → MSE:</strong> Default choice, works for most problems</li>
      <li><strong>Outliers present → MAE or Huber:</strong> Robust to extreme values</li>
      <li><strong>Financial/cost-sensitive → Custom weighted loss:</strong> Weight errors by business impact</li>
      <li><strong>Quantile prediction → Quantile loss:</strong> Predict specific percentiles (e.g., 90th)</li>
    </ul>

    <p><strong>For Classification:</strong></p>
    <ul>
      <li><strong>Binary classification → BCE (with sigmoid):</strong> Standard, produces probabilities</li>
      <li><strong>Multi-class (mutually exclusive) → Categorical CE (with softmax):</strong> Standard choice</li>
      <li><strong>Multi-label (non-exclusive) → Multiple BCE:</strong> Independent binary predictions per label</li>
      <li><strong>Imbalanced data → Weighted CE or Focal Loss:</strong> Handle class frequency imbalance</li>
      <li><strong>Many classes (>1000) → Sparse CE:</strong> Memory efficiency</li>
    </ul>

    <p><strong>For Specialized Tasks:</strong></p>
    <ul>
      <li><strong>Segmentation → Dice + BCE:</strong> Combines pixel-wise and overlap objectives</li>
      <li><strong>Object detection → Classification CE + Localization (IoU/Smooth L1):</strong> Multi-objective</li>
      <li><strong>Face recognition → Triplet Loss or ArcFace:</strong> Metric learning</li>
      <li><strong>Generative models → Custom (GAN: adversarial, VAE: reconstruction+KL):</strong> Domain-specific</li>
    </ul>

    <h3>Critical Activation-Loss Pairings</h3>
    <p><strong>Always pair these correctly:</strong></p>
    <ul>
      <li><strong>Sigmoid → Binary Cross-Entropy:</strong> Binary classification</li>
      <li><strong>Softmax → Categorical Cross-Entropy:</strong> Multi-class classification</li>
      <li><strong>Linear (no activation) → MSE/MAE:</strong> Regression</li>
      <li><strong>Tanh → MSE (if output range is [-1,1]):</strong> Regression with bounded output</li>
    </ul>

    <p><strong>Common mistakes to avoid:</strong></p>
    <ul>
      <li>❌ Using MSE for classification (treats labels as regression targets)</li>
      <li>❌ Applying softmax before nn.CrossEntropyLoss (it includes softmax internally)</li>
      <li>❌ Using BCE without sigmoid (need probabilities, not logits)</li>
      <li>❌ Using softmax for multi-label (classes aren't mutually exclusive)</li>
    </ul>

    <h3>Advanced Considerations</h3>

    <p><strong>Class Weighting:</strong> For imbalanced data, weight loss by inverse class frequency: w_c = N / (K × N_c), where N is total samples, K is number of classes, N_c is samples in class c. Apply as L_weighted = Σ w_c × L_c.</p>

    <p><strong>Label Smoothing:</strong> Instead of hard one-hot targets (0 or 1), use soft targets (ε or 1-ε, typically ε=0.1). This prevents overconfidence and can improve generalization. Commonly used in image classification (Inception, ResNet training).</p>

    <p><strong>Multi-task Learning:</strong> When training one model for multiple objectives, combine losses: L_total = λ₁L₁ + λ₂L₂ + .... The weights λᵢ balance different objectives and require careful tuning. Techniques like uncertainty weighting can automate this.</p>

    <p><strong>Curriculum Learning:</strong> Change loss function during training. Start with easier objective (e.g., MSE) then switch to harder one (e.g., perceptual loss). This can stabilize training for difficult objectives.</p>

    <h3>Debugging Loss Issues</h3>
    <ul>
      <li><strong>Loss is NaN:</strong> Numerical instability (log(0), exp overflow). Use combined softmax+CE, clip extreme values, reduce learning rate</li>
      <li><strong>Loss not decreasing:</strong> Wrong loss-activation pair, learning rate too low, dead neurons, vanishing gradients</li>
      <li><strong>Loss decreasing but evaluation metric not improving:</strong> Loss not aligned with metric, overfitting, need different objective</li>
      <li><strong>Training loss << validation loss:</strong> Overfitting, need regularization (not loss problem)</li>
      <li><strong>Both losses high:</strong> Underfitting, model capacity too small, need better architecture (not loss problem)</li>
    </ul>

    <h3>Common Pitfalls and Debugging</h3>
    <ul>
      <li><strong>Using MSE for classification:</strong> MSE treats discrete classes as continuous values. Always use cross-entropy (BCE for binary, categorical CE for multi-class).</li>
      <li><strong>Softmax before CrossEntropyLoss:</strong> PyTorch's nn.CrossEntropyLoss includes softmax. Applying softmax first gives wrong gradients. Pass raw logits!</li>
      <li><strong>Wrong activation-loss pairing:</strong> Sigmoid without BCE, or softmax without CE causes problems. Follow standard pairings: sigmoid→BCE, softmax→CE, linear→MSE.</li>
      <li><strong>Loss is NaN:</strong> Caused by log(0) or exp(large_number). Solutions: Use combined softmax+CE operations, clip probabilities away from 0/1, reduce learning rate, check for inf/nan in inputs.</li>
      <li><strong>Not weighting classes in imbalanced data:</strong> With 99:1 imbalance, model learns "always predict majority." Use class weights or Focal Loss to balance.</li>
      <li><strong>Loss decreasing but accuracy not improving:</strong> Loss and evaluation metric aren't aligned. Consider: different loss (e.g., Focal Loss), checking for bugs, or the model is learning something but not what you want.</li>
      <li><strong>Using sparse labels with wrong loss:</strong> Sparse labels are integers (0, 1, 2), dense labels are one-hot vectors. Use Sparse CE for integers, regular CE for one-hot.</li>
      <li><strong>Forgetting to average loss over batch:</strong> In custom loss implementations, forgetting to divide by batch size inflates gradients. Use reduction='mean' or manually average.</li>
      <li><strong>Multi-task loss weights not tuned:</strong> L_total = λ₁L₁ + λ₂L₂ requires careful tuning of λᵢ. Start with λᵢ=1, then adjust based on which loss dominates.</li>
    </ul>

    <h3>Historical Context and Modern Trends</h3>
    <p>Early neural networks used MSE for everything, including classification, leading to poor results. The adoption of cross-entropy loss in the 1990s-2000s dramatically improved classification performance. The 2010s saw specialized losses emerge: Focal Loss (2017) for detection, Triplet Loss for face recognition, perceptual losses for style transfer, adversarial losses for GANs. Modern research focuses on learning loss functions (meta-learning), combining multiple objectives efficiently, and designing losses that better align with evaluation metrics.</p>

    <p>Understanding loss functions deeply—their mathematical properties, gradient behavior, appropriate use cases, and common pitfalls—is fundamental to successful neural network training. The loss function is your primary tool for communicating to the network what you want it to learn. Choose wisely.</p>
  `,
  codeExamples: [
    {
      language: 'Python',
      code: `import torch
import torch.nn as nn
import torch.nn.functional as F

# Regression Losses
y_true = torch.tensor([3.0, -0.5, 2.0, 7.0])
y_pred = torch.tensor([2.5, 0.0, 2.0, 8.0])

# MSE Loss
mse = nn.MSELoss()
mse_loss = mse(y_pred, y_true)
print(f"MSE Loss: {mse_loss.item():.4f}")

# MAE Loss
mae = nn.L1Loss()
mae_loss = mae(y_pred, y_true)
print(f"MAE Loss: {mae_loss.item():.4f}")

# Huber Loss
huber = nn.SmoothL1Loss()
huber_loss = huber(y_pred, y_true)
print(f"Huber Loss: {huber_loss.item():.4f}")

# Classification Losses
# Binary Cross-Entropy
y_true_binary = torch.tensor([1.0, 0.0, 1.0, 0.0])
y_pred_binary = torch.tensor([0.9, 0.1, 0.8, 0.3])  # Sigmoid outputs
bce = nn.BCELoss()
bce_loss = bce(y_pred_binary, y_true_binary)
print(f"\\nBinary Cross-Entropy: {bce_loss.item():.4f}")

# Categorical Cross-Entropy
logits = torch.randn(4, 3)  # 4 samples, 3 classes
targets = torch.tensor([0, 1, 2, 1])  # Class indices
ce = nn.CrossEntropyLoss()
ce_loss = ce(logits, targets)
print(f"Categorical Cross-Entropy: {ce_loss.item():.4f}")

# Show effect of confidence on BCE
print("\\nEffect of prediction confidence on BCE loss:")
confidences = [0.6, 0.7, 0.8, 0.9, 0.99]
for conf in confidences:
  pred = torch.tensor([conf])
  target = torch.tensor([1.0])
  loss = F.binary_cross_entropy(pred, target)
  print(f"Prediction {conf:.2f} (true=1.0): Loss = {loss.item():.4f}")`,
      explanation: 'Demonstrates common loss functions for regression and classification. Shows how cross-entropy heavily penalizes confident wrong predictions, encouraging well-calibrated probabilities.'
    },
    {
      language: 'Python',
      code: `import torch
import torch.nn as nn

# Handling class imbalance with weighted loss
# Dataset: 90% class 0, 10% class 1
targets = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 9:1 imbalance
logits = torch.randn(10, 2)

# Standard loss (biased toward majority class)
ce_standard = nn.CrossEntropyLoss()
loss_standard = ce_standard(logits, targets)
print(f"Standard CE Loss: {loss_standard.item():.4f}")

# Weighted loss (give more weight to minority class)
# Weight inversely proportional to class frequency
class_weights = torch.tensor([1.0, 9.0])  # Class 1 has 9x weight
ce_weighted = nn.CrossEntropyLoss(weight=class_weights)
loss_weighted = ce_weighted(logits, targets)
print(f"Weighted CE Loss: {loss_weighted.item():.4f}")

# Custom Focal Loss implementation
class FocalLoss(nn.Module):
  def __init__(self, alpha=1.0, gamma=2.0):
      super().__init__()
      self.alpha = alpha
      self.gamma = gamma

  def forward(self, inputs, targets):
      ce_loss = F.cross_entropy(inputs, targets, reduction='none')
      pt = torch.exp(-ce_loss)  # Probability of true class
      focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
      return focal_loss.mean()

focal = FocalLoss(alpha=1.0, gamma=2.0)
loss_focal = focal(logits, targets)
print(f"Focal Loss: {loss_focal.item():.4f}")

print("\\nFocal Loss focuses on hard examples by down-weighting easy ones")`,
      explanation: 'Shows how to handle class imbalance using weighted loss and focal loss. Weighted loss gives more importance to minority class. Focal loss automatically focuses on hard-to-classify examples.'
    }
  ],
  interviewQuestions: [
    {
      question: 'What is the difference between MSE and MAE loss?',
      answer: '**Mean Squared Error (MSE)** computes the average of squared differences between predicted and actual values: **MSE = (1/n)Σ(y_i - ŷ_i)²**. **Mean Absolute Error (MAE)** computes the average of absolute differences: **MAE = (1/n)Σ|y_i - ŷ_i|**. The key difference lies in how they penalize errors: MSE gives **quadratic penalty** to errors (large errors are penalized disproportionately more), while MAE gives **linear penalty** (all errors are penalized proportionally to their magnitude).\n\nThe **sensitivity to outliers** differs dramatically between these loss functions. MSE is highly sensitive to outliers because squaring the error amplifies large deviations exponentially. A single data point with error 10 contributes 100 to MSE, while ten points with error 1 each contribute only 10 total. MAE treats all errors linearly, so the same outlier contributes only 10, making it more **robust to outliers** and providing more balanced learning when the dataset contains anomalous values.\n\n**Gradient properties** also differ significantly. MSE has smooth, continuous gradients that are proportional to the error magnitude: **∂MSE/∂ŷ = 2(ŷ - y)**. This means larger errors produce larger gradients, leading to faster correction of big mistakes. MAE has constant gradient magnitude: **∂MAE/∂ŷ = sign(ŷ - y)**, which provides consistent but potentially slower learning. However, MAE gradients are discontinuous at zero, which can cause optimization challenges near the optimum.\n\nPractically, use **MSE** when you want to heavily penalize large errors and when your data has few outliers. MSE is preferred for problems where large errors are much worse than small ones (e.g., financial forecasting where big mistakes are costly). Use **MAE** when your data contains outliers that shouldn\'t dominate the learning process, or when all errors should be treated equally important. MAE often produces more robust models for real-world data with noise and anomalies, but may converge more slowly due to its constant gradients.'
    },
    {
      question: 'Why do we use Cross-Entropy loss for classification instead of MSE?',
      answer: '**Cross-Entropy loss** is designed specifically for classification problems because it naturally handles **probability distributions** and provides appropriate gradient signals for learning class boundaries. For binary classification, cross-entropy loss is: **L = -[y log(p) + (1-y) log(1-p)]**, where **y** is the true label (0 or 1) and **p** is the predicted probability. This formulation heavily penalizes confident wrong predictions while providing gentle penalties for uncertain predictions near 0.5.\n\n**MSE treats classification as regression**, which is fundamentally problematic. When using MSE for classification, the model tries to output exact target values (0 or 1) rather than learning probability distributions. This can lead to **saturated gradients**: once a sigmoid output gets close to 0 or 1, its gradient becomes very small, causing the neuron to stop learning even if the prediction is wrong. Cross-entropy maintains reasonable gradients throughout the probability range, ensuring continued learning.\n\nThe **gradient behavior** reveals why cross-entropy is superior. For sigmoid activation with cross-entropy, the gradient simplifies to **∂L/∂z = p - y** (where **z** is the pre-activation), providing clean, proportional error signals. With MSE and sigmoid, the gradient includes the sigmoid derivative term **σ\'(z) = σ(z)(1-σ(z))**, which becomes very small for extreme values, leading to vanishing gradients and slow learning.\n\n**Probabilistic interpretation** is another crucial advantage. Cross-entropy loss directly corresponds to **maximum likelihood estimation** for the Bernoulli distribution (binary classification) or categorical distribution (multi-class). This means minimizing cross-entropy is equivalent to finding the most likely parameters given the data, providing a principled statistical foundation. MSE lacks this probabilistic interpretation for classification tasks.\n\nAdditionally, cross-entropy naturally handles **multi-class classification** through its extension to categorical cross-entropy: **L = -Σy_i log(p_i)**, where **y_i** is the one-hot encoded true label and **p_i** is the predicted probability for class **i**. This pairs perfectly with softmax activation to ensure valid probability distributions. Using MSE for multi-class problems would ignore the constraint that probabilities should sum to 1 and could produce nonsensical outputs.'
    },
    {
      question: 'What activation function should be paired with Binary Cross-Entropy?',
      answer: '**Sigmoid activation** should be paired with Binary Cross-Entropy (BCE) loss for binary classification. The sigmoid function **σ(z) = 1/(1 + e^(-z))** outputs values between 0 and 1, which can be naturally interpreted as probabilities. This pairing is mathematically optimal because it creates a clean gradient flow and corresponds to maximum likelihood estimation for Bernoulli-distributed data.\n\nThe mathematical elegance of this combination becomes clear when computing gradients. For sigmoid activation with BCE loss, the gradient with respect to the pre-activation **z** simplifies beautifully: **∂L/∂z = p - y**, where **p** is the predicted probability and **y** is the true label. This means the gradient is simply the prediction error, providing intuitive and proportional learning signals. No complex chain rule calculations involving activation function derivatives are needed.\n\n**Other activations are problematic** with BCE. Using **tanh** (outputs -1 to 1) would require shifting and scaling to get probabilities, making the loss function more complex. **ReLU** is inappropriate because it outputs unbounded positive values and zero for negative inputs, which cannot be interpreted as probabilities. **Linear activation** could output any real number, requiring additional constraints to ensure valid probabilities.\n\n**Softmax could theoretically work** for binary classification by using two outputs, but this is unnecessarily complex. Softmax with categorical cross-entropy for two classes is mathematically equivalent to sigmoid with binary cross-entropy, but sigmoid is more efficient and interpretable for binary problems. Use softmax with categorical cross-entropy only for multi-class classification (3+ classes).\n\nThe **probabilistic interpretation** is crucial: sigmoid with BCE allows you to set decision thresholds based on probability cutoffs (e.g., classify as positive if p > 0.7), enabling **uncertainty quantification** and **cost-sensitive decision making**. The output can be used directly for ranking, probability estimation, or feeding into downstream decision systems. This pairing has become the standard for binary classification because it provides both computational efficiency and meaningful probabilistic outputs that are essential for most real-world applications.'
    },
    {
      question: 'How does Focal Loss help with class imbalance?',
      answer: '**Focal Loss** addresses class imbalance by **down-weighting easy examples** and focusing training on hard-to-classify examples, which are often from the minority class. The focal loss formula is: **FL(p_t) = -α_t(1-p_t)^γ log(p_t)**, where **p_t** is the predicted probability for the true class, **α_t** is a class-specific weight, and **γ** (gamma) is the focusing parameter. The key innovation is the **(1-p_t)^γ** term that reduces the loss contribution from well-classified examples.\n\nIn standard cross-entropy loss, **easy examples** (those with high predicted probability for the correct class) still contribute significantly to the total loss and gradients. With severe class imbalance, the abundant easy examples from the majority class can overwhelm the learning signal from the minority class. Focal loss solves this by applying a **modulating factor**: when **p_t** is high (easy example), **(1-p_t)** is small, so the loss is down-weighted. When **p_t** is low (hard example), the full loss is retained.\n\nThe **focusing parameter γ** controls the strength of down-weighting. When **γ = 0**, focal loss reduces to standard cross-entropy. As **γ** increases, more emphasis is placed on hard examples. For example, with **γ = 2**, an example with **p_t = 0.9** (easy) has its loss reduced by a factor of **(1-0.9)² = 0.01**, while an example with **p_t = 0.5** (hard) has its loss reduced by only **(1-0.5)² = 0.25**.\n\nThe **α_t** parameter provides additional class-specific weighting, similar to class weights in standard approaches. For binary classification, **α** can be set to address class frequency imbalance: higher **α** for the minority class to increase its contribution to the loss. This combines with the focusing mechanism to provide both frequency-based and difficulty-based rebalancing.\n\n**Practical benefits** include: better performance on imbalanced datasets without requiring resampling techniques, automatic focus on challenging examples that need more attention, and particularly strong results in object detection where background vs. object imbalance is severe. Focal loss has become standard in applications like RetinaNet for object detection, where it significantly outperforms cross-entropy loss on highly imbalanced datasets with thousands of background examples per object.'
    },
    {
      question: 'Why is it important to combine softmax and cross-entropy in a single operation?',
      answer: 'Combining **softmax** and **cross-entropy** in a single operation (often called "softmax_cross_entropy_with_logits") provides crucial **numerical stability** and **computational efficiency** benefits. When implemented separately, the softmax function can produce extremely small probabilities (near machine epsilon) or overflow to infinity for large logits, leading to numerical issues when computing the logarithm in cross-entropy loss.\n\nThe **numerical stability** issue arises from the exponential function in softmax: **p_i = e^(z_i) / Σe^(z_j)**. For large positive logits, **e^(z_i)** can overflow to infinity; for large negative logits relative to the maximum, **e^(z_i)** can underflow to zero. When these extreme probabilities are passed to cross-entropy **L = -Σy_i log(p_i)**, the logarithm can produce undefined results (log(0) = -∞) or lose precision.\n\nThe **combined implementation** works with logits directly and applies mathematical simplifications. For the true class **c** with logit **z_c**, the combined loss becomes: **L = -z_c + log(Σe^(z_j))**, avoiding the intermediate probability computation entirely. This formulation is numerically stable because it can apply the **log-sum-exp trick**: **log(Σe^(z_j)) = z_max + log(Σe^(z_j - z_max))**, which prevents overflow by subtracting the maximum logit.\n\n**Gradient computation** also benefits from the combined approach. The gradient of the combined loss with respect to logits is simply: **∂L/∂z_i = p_i - y_i** (where **p_i** is computed stably), providing clean error signals. Computing this gradient through separate softmax and cross-entropy operations involves more complex chain rule calculations and potential numerical instabilities.\n\n**Computational efficiency** improves because the combined operation avoids computing and storing the full probability vector when only the loss value is needed. This saves memory and computation, especially important for models with large vocabulary sizes (like language models with 50K+ word vocabularies). Modern deep learning frameworks implement this optimization automatically, making it a best practice to use the combined operation whenever possible rather than implementing softmax and cross-entropy separately.'
    },
    {
      question: 'When would you use Huber loss instead of MSE?',
      answer: `**Huber loss** combines the best properties of MSE and MAE by being **quadratic for small errors** and **linear for large errors**. It is defined as a piecewise function where small errors are penalized quadratically like MSE, while large errors are penalized linearly like MAE, with **delta** as the threshold parameter. This makes Huber loss **less sensitive to outliers** than MSE while maintaining **smooth gradients** unlike MAE.

The primary use case is **regression with outliers**. When your dataset contains outliers that shouldn't dominate the learning process, MSE can be problematic because it gives quadratic penalty to large errors, causing the model to focus excessively on fitting outliers at the expense of the general pattern. Huber loss caps the penalty for large errors at a linear rate, making the model more robust to outliers while still providing strong gradients for small errors.

**Gradient properties** make Huber loss particularly attractive. Unlike MAE which has discontinuous gradients at zero (causing optimization difficulties), Huber loss has **continuous, smooth gradients** everywhere. For small errors, the gradient is proportional to the error like MSE, providing strong learning signals. For large errors, the gradient has constant magnitude like MAE, preventing outliers from dominating gradient updates.

The **delta parameter** requires tuning based on your problem. Smaller delta makes the loss more like MAE (more robust to outliers but potentially slower convergence), while larger delta makes it more like MSE (faster convergence but less robust). A common heuristic is to set delta to the 90th percentile of absolute errors from an initial MSE model, but cross-validation is often needed for optimal performance.

**Practical applications** include: **financial modeling** where extreme market events shouldn't dominate predictions, **sensor data processing** where occasional faulty readings occur, **computer vision** where lighting or occlusion can create outlier pixel values, and **time series forecasting** where occasional anomalous events shouldn't distort the overall trend. Huber loss is particularly valuable in **robust statistics** and **production environments** where data quality can't be guaranteed and model reliability is more important than fitting every data point perfectly.`
    }
  ],
  quizQuestions: [
    {
      id: 'loss-q1',
      question: 'You use MSE loss for binary classification. The model outputs probabilities via sigmoid. What problem will occur?',
      options: [
        'Model trains perfectly fine',
        'Gradients are not optimized for probability outputs - use BCE instead',
        'Training will be faster',
        'Model cannot learn at all'
      ],
      correctAnswer: 1,
      explanation: 'MSE treats classification as regression, giving poor gradients. For probabilities in [0,1], MSE gradients are weak near correct predictions. BCE (cross-entropy) provides stronger, more appropriate gradients for probability outputs. Always pair sigmoid→BCE or softmax→CrossEntropy.'
    },
    {
      id: 'loss-q2',
      question: 'Your dataset has 99% negative samples, 1% positive. You use standard Binary Cross-Entropy and the model predicts everything as negative (99% accuracy). What should you do?',
      options: [
        'The model is perfect',
        'Use weighted BCE with higher weight for positive class, or Focal Loss',
        'Collect more negative samples',
        'Use MSE instead'
      ],
      correctAnswer: 1,
      explanation: 'Model learned trivial solution (always predict majority). Use weighted loss (higher weight for minority class) or Focal Loss (focuses on hard examples) to balance learning. Can also try under-sampling majority or over-sampling minority class.'
    },
    {
      id: 'loss-q3',
      question: 'Why should softmax and cross-entropy be combined in a single operation (like nn.CrossEntropyLoss)?',
      options: [
        'Faster computation',
        'Numerical stability - avoids log(0) and overflow in exp()',
        'Uses less memory',
        'Required by PyTorch'
      ],
      correctAnswer: 1,
      explanation: 'Computing softmax then log(softmax) separately can cause numerical issues: exp(large_number) overflows, log(0) is undefined. Combined implementation uses log-sum-exp trick for numerical stability. This is why nn.CrossEntropyLoss takes logits, not probabilities.'
    }
  ]
};
