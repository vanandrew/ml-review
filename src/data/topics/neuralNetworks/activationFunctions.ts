import { Topic } from '../../../types';

export const activationFunctions: Topic = {
  id: 'activation-functions',
  title: 'Activation Functions',
  category: 'neural-networks',
  description: 'Non-linear functions that enable neural networks to learn complex patterns',
  content: `
    <h2>Activation Functions: The Source of Neural Network Power</h2>
    <p>Activation functions are the mathematical operations that introduce non-linearity into neural networks, transforming them from simple linear models into powerful universal function approximators. Without activation functions, even the deepest neural network would be mathematically equivalent to a single-layer linear model. Understanding activation functions is essential because they fundamentally determine what patterns a network can learn, how quickly it trains, and whether training succeeds at all.</p>

    <p>Each neuron in a neural network computes a weighted sum of its inputs plus a bias: <strong>z = w·x + b</strong>. This operation is purely linear. The activation function <strong>f</strong> is then applied to produce the neuron's output: <strong>a = f(z)</strong>. This seemingly simple additional step is what enables neural networks to model arbitrarily complex, non-linear relationships between inputs and outputs. The choice of activation function impacts training speed, convergence behavior, gradient flow, and ultimately model performance.</p>

    <h3>Why Non-Linearity is Absolutely Essential</h3>
    <p>Consider what happens when you stack multiple linear layers without non-linear activations:</p>

    <p><strong>Mathematical Proof of Linear Collapse:</strong></p>
    <ul>
      <li>Layer 1: <strong>$h_1 = W_1 x + b_1$</strong></li>
      <li>Layer 2: <strong>$h_2 = W_2 h_1 + b_2 = W_2(W_1 x + b_1) + b_2 = W_2 W_1 x + W_2 b_1 + b_2$</strong></li>
      <li>Layer 3: <strong>$h_3 = W_3 h_2 + b_3 = W_3(W_2 W_1 x + W_2 b_1 + b_2) + b_3 = W_3 W_2 W_1 x + \\text{(terms with b)}$</strong></li>
    </ul>

    <p>We can define <strong>$\\tilde{W} = W_3 W_2 W_1$</strong> (a single matrix) and <strong>$\\tilde{b}$</strong> as the combined bias terms. The entire deep network simplifies to: <strong>$h_3 = \\tilde{W}x + \\tilde{b}$</strong>—just a single linear transformation! No matter how many layers you add, the composition of linear functions is still linear. This network can only learn linear decision boundaries, meaning it would fail on even simple problems like XOR, and certainly couldn't learn the complex patterns in images, text, or speech.</p>

    <p><strong>What non-linearity enables:</strong></p>
    <ul>
      <li><strong>Complex decision boundaries:</strong> Instead of straight lines or flat hyperplanes, networks can learn curved, intricate boundaries that wrap around data in high-dimensional space</li>
      <li><strong>Hierarchical feature learning:</strong> Early layers learn simple features (edges, textures), deeper layers compose these into complex abstractions (objects, concepts)</li>
      <li><strong>Universal approximation:</strong> The universal approximation theorem only holds with non-linear activations—they're the mathematical requirement for approximating arbitrary functions</li>
      <li><strong>Representational power:</strong> With appropriate non-linearities, networks can represent vastly more functions with the same number of parameters compared to linear models</li>
    </ul>

    <h3>The ReLU Family: Modern Workhorses</h3>

    <h4>ReLU (Rectified Linear Unit)</h4>
    <p><strong>$f(x) = \\max(0, x)$</strong></p>
    <p><strong>Derivative: $f'(x) = \\begin{cases} 1 & \\text{if } x > 0 \\\\ 0 & \\text{else} \\end{cases}$</strong></p>
    
    <p>ReLU, introduced in 2010 and popularized by AlexNet (2012), revolutionized deep learning by addressing the vanishing gradient problem that plagued sigmoid and tanh networks. Its elegantly simple definition—just outputting the input if positive, zero otherwise—makes it extremely fast to compute and differentiate. The derivative being 1 for positive inputs means gradients flow backward without diminishing, enabling much deeper networks to train successfully.</p>

    <p><strong>Key advantages:</strong></p>
    <ul>
      <li><strong>Computational efficiency:</strong> Just a simple comparison and max operation, much faster than exponentials in sigmoid/tanh</li>
      <li><strong>Gradient flow:</strong> Derivative of 1 means no vanishing gradient for positive inputs; gradients propagate unchanged through active neurons</li>
      <li><strong>Sparse activation:</strong> About 50% of neurons output zero for random inputs, creating sparse representations that are computationally efficient and may aid generalization</li>
      <li><strong>Biological plausibility:</strong> Matches aspects of real neuron behavior better than sigmoid (neurons either fire or don't)</li>
      <li><strong>Scale invariant:</strong> ReLU(cx) = c·ReLU(x) for c > 0, which can help with optimization</li>
    </ul>

    <p><strong>The dying ReLU problem:</strong> If a neuron's weighted input becomes negative for all training examples, ReLU outputs zero, the gradient is zero, and the neuron stops learning permanently—it "dies." This can happen due to poor initialization, high learning rates causing large weight updates, or unfortunate data distribution. Once dead, the neuron contributes nothing to the network's output and never recovers. In severe cases, large portions of a network can die, dramatically reducing effective capacity. Solutions include careful learning rate selection, proper weight initialization (He initialization), and using ReLU variants like Leaky ReLU.</p>

    <h4>Leaky ReLU and Parametric ReLU (PReLU)</h4>
    <p><strong>Leaky ReLU: $f(x) = \\begin{cases} x & \\text{if } x > 0 \\\\ \\alpha x & \\text{else} \\end{cases}$</strong> (typically $\\alpha = 0.01$)</p>
    <p><strong>PReLU: $f(x) = \\begin{cases} x & \\text{if } x > 0 \\\\ \\alpha x & \\text{else} \\end{cases}$</strong> ($\\alpha$ is learned during training)</p>
    <p><strong>Derivative: $f'(x) = \\begin{cases} 1 & \\text{if } x > 0 \\\\ \\alpha & \\text{else} \\end{cases}$</strong></p>

    <p>Leaky ReLU addresses the dying ReLU problem by allowing a small, non-zero gradient (typically 0.01) when the input is negative. Instead of completely killing the gradient, negative inputs receive a small "leaky" gradient that allows neurons to potentially recover from negative activations. This simple modification prevents neurons from dying while maintaining most of ReLU's benefits. PReLU takes this further by making α a learnable parameter, allowing the network to decide the optimal negative slope for each neuron during training. In practice, Leaky ReLU with α=0.01 works well and is preferred over standard ReLU when dying neurons are a concern.</p>

    <h4>ELU (Exponential Linear Unit)</h4>
    <p><strong>$f(x) = \\begin{cases} x & \\text{if } x > 0 \\\\ \\alpha(e^x - 1) & \\text{else} \\end{cases}$</strong> (typically $\\alpha = 1.0$)</p>
    <p><strong>Derivative: $f'(x) = \\begin{cases} 1 & \\text{if } x > 0 \\\\ f(x) + \\alpha & \\text{else} \\end{cases}$</strong></p>

    <p>ELU uses a smooth exponential curve for negative values instead of a linear slope. This has several advantages: (1) the smooth transition can lead to faster learning; (2) ELU can produce negative outputs, pushing mean activation closer to zero, which helps reduce bias shift and can speed up learning; (3) the saturation for large negative values can provide robustness to noise. However, ELU's exponential computation is slower than ReLU's simple comparison. ELU often outperforms ReLU on smaller datasets or when each training epoch is important, but ReLU remains more common due to its simplicity and speed.</p>

    <h3>Classical Activation Functions: Historical But Still Relevant</h3>

    <h4>Sigmoid (Logistic Function)</h4>
    <p><strong>$f(x) = \\frac{1}{1 + e^{-x}}$</strong></p>
    <p><strong>Output range: $(0, 1)$</strong></p>
    <p><strong>Derivative: $f'(x) = f(x)(1 - f(x))$</strong></p>

    <p>The sigmoid function squashes any real-valued input into the range (0, 1), producing an S-shaped curve. It was once the default activation function, inspired by biological neurons having a maximum firing rate. Its output can be interpreted as a probability, making it perfect for binary classification outputs. However, sigmoid has severe problems for hidden layers in deep networks.</p>

    <p><strong>The vanishing gradient catastrophe:</strong> The sigmoid derivative peaks at 0.25 (when $x=0$) and rapidly approaches zero for large positive or negative inputs. During backpropagation, gradients are multiplied by these derivatives layer by layer. If you have 10 layers, gradients might be multiplied by 0.25 ten times: $(0.25)^{10} \\approx 0.0000001$—effectively zero! This means earlier layers receive almost no learning signal, training becomes glacially slow or stops entirely, and the network never learns the fundamental features in early layers that deeper layers depend on.</p>

    <p><strong>Additional problems:</strong></p>
    <ul>
      <li><strong>Not zero-centered:</strong> Outputs are always positive, causing gradients to all be positive or all negative, leading to zig-zagging during gradient descent and slower convergence</li>
      <li><strong>Computational cost:</strong> The exponential function is significantly slower than ReLU's simple comparison</li>
      <li><strong>Saturation regions:</strong> For |x| > 5, the function barely changes, making learning extremely slow for saturated neurons</li>
    </ul>

    <p><strong>Modern uses:</strong> Despite these problems, sigmoid remains essential for binary classification output layers (interpreting output as P(y=1|x)) and for gates in LSTM and GRU recurrent networks where the (0,1) range is needed to control information flow. Just avoid it for hidden layers in deep networks!</p>

    <h4>Tanh (Hyperbolic Tangent)</h4>
    <p><strong>$f(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\\sigma(2x) - 1$</strong></p>
    <p><strong>Output range: $(-1, 1)$</strong></p>
    <p><strong>Derivative: $f'(x) = 1 - f(x)^2$</strong></p>

    <p>Tanh is essentially a scaled and shifted sigmoid, offering one crucial improvement: zero-centered outputs. The range (-1, 1) means the mean activation is closer to zero, which generally makes learning easier. The derivative peaks at 1.0 (vs sigmoid's 0.25), giving stronger gradients. For these reasons, tanh was historically preferred over sigmoid for hidden layers. However, tanh still suffers from vanishing gradients in deep networks—the derivative still approaches zero for large |x|, just not quite as badly as sigmoid.</p>

    <p><strong>Modern relevance:</strong> Tanh is still used in recurrent neural networks (RNNs) for hidden state updates, where the (-1, 1) range provides a natural bounded representation. It's also occasionally used in shallow networks or specific architectural components. However, for general feed-forward hidden layers, ReLU and its variants have largely replaced tanh.</p>

    <h3>Modern Advanced Activations</h3>

    <h4>Softmax: The Multi-Class Specialist</h4>
    <p><strong>$f(x)_i = \\frac{e^{x_i}}{\\sum_j e^{x_j}}$</strong></p>

    <p>Softmax is unique among activations—it's not applied element-wise but rather transforms a vector of logits (raw scores) into a probability distribution. Each output is a positive value between 0 and 1, and all outputs sum to exactly 1, allowing interpretation as class probabilities. Softmax "soft-maximizes" the input: the largest input gets the highest probability, but unlike hard max (which outputs 1 for the largest and 0 for others), softmax gives non-zero probabilities to all classes, with the degree of differentiation controlled by the input magnitudes.</p>

    <p><strong>Mathematical properties:</strong></p>
    <ul>
      <li><strong>Temperature scaling:</strong> Softmax(x/T) with temperature T > 1 smooths the distribution (more uncertain), while T < 1 sharpens it (more confident)</li>
      <li><strong>Numerical stability:</strong> Implement as softmax(x - max(x)) to prevent overflow in exponentials</li>
      <li><strong>Gradient:</strong> Has convenient derivative properties when paired with cross-entropy loss, simplifying to just (predicted - actual)</li>
    </ul>

    <p><strong>Critical usage note:</strong> Always use softmax only in the output layer for multi-class classification (mutually exclusive classes). Pair it with categorical cross-entropy loss. For multi-label classification (non-exclusive classes), use independent sigmoid outputs instead. Never use softmax in hidden layers—it destroys information by normalizing activations.</p>

    <h4>Swish / SiLU (Sigmoid Linear Unit)</h4>
    <p><strong>$f(x) = x \\cdot \\sigma(x) = \\frac{x}{1 + e^{-x}}$</strong></p>
    <p><strong>Derivative: $f'(x) = f(x) + \\sigma(x)(1 - f(x))$</strong></p>

    <p>Discovered through extensive neural architecture search by Google researchers, Swish is a smooth, non-monotonic activation that often outperforms ReLU in deep networks. It's "self-gated"—the output is the input modulated by its own sigmoid, allowing the function to decide how much of the input to pass through. For large positive x, Swish ≈ x (like ReLU); for large negative x, Swish ≈ 0 (like ReLU); but the smooth transition and non-monotonicity seem to help optimization and generalization.</p>

    <p>Swish has been adopted in state-of-the-art architectures like EfficientNet and some Transformer variants. The main drawback is computational cost—computing both x and σ(x) is slower than ReLU's simple comparison. Use Swish when model quality is paramount and you can afford the extra computation.</p>

    <h4>GELU (Gaussian Error Linear Unit)</h4>
    <p><strong>Exact: $f(x) = x \\cdot \\Phi(x)$</strong> where $\\Phi$ is the Gaussian CDF</p>
    <p><strong>Approximation: $f(x) \\approx 0.5x\\left(1 + \\tanh\\left(\\sqrt{\\frac{2}{\\pi}}(x + 0.044715x^3)\\right)\\right)$</strong></p>

    <p>GELU provides a smooth approximation to ReLU with a probabilistic interpretation: it weights inputs by their probability under a standard normal distribution. Inputs significantly above zero are passed through, inputs significantly below are zeroed, and intermediate values are probabilistically gated. The smooth curve (no kink at zero like ReLU) may help optimization, and empirical results show GELU often outperforms ReLU in natural language processing.</p>

    <p>GELU has become the standard activation in Transformer models—it's used in BERT, GPT-2, GPT-3, and most modern language models. The smooth nature and probabilistic interpretation seem particularly beneficial for the attention mechanisms and massive scale typical of Transformers. If you're building a Transformer, use GELU.</p>

    <h3>Choosing the Right Activation Function: A Decision Guide</h3>

    <p><strong>For Hidden Layers:</strong></p>
    <ul>
      <li><strong>Default choice: ReLU</strong> - Fast, effective, well-understood. Start here unless you have specific reasons to choose otherwise.</li>
      <li><strong>If experiencing dying ReLU: Leaky ReLU</strong> - Prevents dead neurons with minimal computational overhead.</li>
      <li><strong>For smaller datasets or shallow networks: ELU</strong> - Smooth learning, negative outputs help normalize activations.</li>
      <li><strong>For Transformers and NLP: GELU</strong> - Empirically superior for large language models.</li>
      <li><strong>For highest accuracy (and willing to pay computation cost): Swish</strong> - Often achieves best performance in vision tasks.</li>
      <li><strong>Avoid: Sigmoid and Tanh</strong> - Vanishing gradient makes them poor choices for deep networks.</li>
    </ul>

    <p><strong>For Output Layers (task-dependent):</strong></p>
    <ul>
      <li><strong>Binary classification: Sigmoid</strong> - Outputs interpretable probability in (0,1).</li>
      <li><strong>Multi-class classification: Softmax</strong> - Outputs probability distribution over classes.</li>
      <li><strong>Multi-label classification: Multiple Sigmoids</strong> - Independent probabilities for each label.</li>
      <li><strong>Regression: Linear (no activation)</strong> - Allows unbounded output for continuous values.</li>
      <li><strong>Bounded regression: Sigmoid or Tanh</strong> - When output must be in specific range.</li>
    </ul>

    <p><strong>For Recurrent Networks:</strong></p>
    <ul>
      <li><strong>Hidden state: Tanh</strong> - Bounded (-1,1) range prevents hidden states from exploding.</li>
      <li><strong>Gates (LSTM/GRU): Sigmoid</strong> - (0,1) range perfect for gate values controlling information flow.</li>
    </ul>

    <h3>Common Problems and Solutions</h3>

    <h4>Vanishing Gradients: The Deep Network Killer</h4>
    <p>In deep networks, gradients must flow through many layers during backpropagation. Each layer multiplies the gradient by its local derivative. If these derivatives are consistently less than 1 (as with sigmoid/tanh), the product becomes exponentially smaller: $0.25^{20} \\approx 10^{-13}$. Early layers receive essentially zero gradient, stopping learning entirely.</p>

    <p><strong>Symptoms:</strong> Early layers don't improve during training, validation loss plateaus early, network performs poorly despite deep architecture, gradient norms decrease exponentially with depth.</p>

    <p><strong>Solutions:</strong></p>
    <ul>
      <li><strong>Use ReLU family:</strong> Derivative of 1 for positive inputs prevents gradient diminishing</li>
      <li><strong>Batch normalization:</strong> Normalizes inputs to each layer, keeping activations in healthy ranges</li>
      <li><strong>Residual connections:</strong> Skip connections allow gradients to bypass layers via direct paths</li>
      <li><strong>Proper initialization:</strong> He initialization for ReLU, Xavier for tanh, ensures initial gradients are well-scaled</li>
      <li><strong>Gradient clipping:</strong> Doesn't directly solve vanishing but prevents the opposite problem</li>
    </ul>

    <h4>Exploding Gradients: Numerical Instability</h4>
    <p>Less common than vanishing but equally problematic: if derivatives are consistently greater than 1 or weights are large, gradients grow exponentially, causing NaN values and training failure.</p>

    <p><strong>Symptoms:</strong> Loss becomes NaN, weights become extremely large, training loss oscillates wildly, model parameters explode to infinity.</p>

    <p><strong>Solutions:</strong></p>
    <ul>
      <li><strong>Gradient clipping:</strong> Cap gradient norms at a maximum value (e.g., clip to [-5, 5])</li>
      <li><strong>Proper weight initialization:</strong> Small initial weights prevent early explosion</li>
      <li><strong>Batch normalization:</strong> Keeps activations and gradients in reasonable ranges</li>
      <li><strong>Lower learning rate:</strong> Reduces step sizes, preventing dramatic weight updates</li>
      <li><strong>L2 regularization:</strong> Penalizes large weights, keeping them bounded</li>
    </ul>

    <h4>The Dying ReLU Problem: Permanent Neuron Death</h4>
    <p>A ReLU neuron that outputs zero for all training examples has zero gradient, receives no weight updates, and never recovers. This can cascade: if many neurons die, the network loses effective capacity.</p>

    <p><strong>Causes:</strong> High learning rates causing large negative weight updates, poor initialization placing most activations in negative region, imbalanced data causing persistent negative inputs.</p>

    <p><strong>Symptoms:</strong> Large percentage of neurons always output zero, effective network capacity much smaller than architecture suggests, training and validation performance both poor.</p>

    <p><strong>Solutions:</strong></p>
    <ul>
      <li><strong>Leaky ReLU:</strong> Small negative slope (0.01) prevents zero gradient</li>
      <li><strong>Lower learning rate:</strong> Prevents dramatic weight shifts into dead regions</li>
      <li><strong>He initialization:</strong> Properly scaled initial weights reduce chance of initial death</li>
      <li><strong>Batch normalization:</strong> Keeps pre-activation values centered, reducing likelihood of all-negative inputs</li>
      <li><strong>Increase learning rate gradually:</strong> Start with small learning rate, increase once stable</li>
    </ul>

    <h3>Common Pitfalls and Debugging</h3>
    <ul>
      <li><strong>Using sigmoid/tanh in deep networks:</strong> These cause vanishing gradients in networks >5 layers. Stick to ReLU family for hidden layers in deep networks.</li>
      <li><strong>Using ReLU in output layer:</strong> ReLU outputs unbounded positive values, inappropriate for classification probabilities or bounded regression. Use sigmoid (binary), softmax (multi-class), or linear (regression).</li>
      <li><strong>Not addressing dying ReLU:</strong> If >30% of neurons output zero consistently, switch to Leaky ReLU or reduce learning rate. Monitor activation statistics during training.</li>
      <li><strong>Softmax for multi-label classification:</strong> Softmax outputs sum to 1, forcing exclusivity. For non-exclusive labels (e.g., image tags: "cat", "outdoor", "sunny"), use independent sigmoid outputs.</li>
      <li><strong>Applying softmax before CrossEntropyLoss:</strong> PyTorch's CrossEntropyLoss includes softmax internally. Applying softmax first causes wrong gradients. Pass raw logits to the loss function.</li>
      <li><strong>Wrong activation scale for problem:</strong> If predicting values in [0, 100], using tanh (outputs [-1, 1]) requires rescaling. Match activation output range to target value range.</li>
      <li><strong>Inconsistent activations across layers:</strong> Mixing sigmoid and ReLU haphazardly can cause issues. Be consistent: ReLU for all hidden layers, task-specific for output.</li>
    </ul>

    <h3>Historical Evolution and Future Directions</h3>
    <p>The history of activation functions reflects deep learning's evolution. In the 1980s-90s, sigmoid was standard but limited networks to a few layers. The introduction of ReLU (2010) enabled the deep learning revolution—AlexNet (2012), the first CNN to dominate ImageNet, relied heavily on ReLU. Subsequent refinements (Leaky ReLU, ELU, PReLU) addressed specific ReLU limitations. Neural architecture search discovered Swish (2017), showing that automated methods could find better activations than human intuition. GELU emerged from Transformers research, proving different domains benefit from different non-linearities.</p>

    <p>Future research explores: learnable activations where the functional form adapts during training, adaptive activations that change based on input characteristics, activation ensembles combining multiple functions, and task-specific activations optimized for particular problem domains. The search for the "perfect" activation continues, though ReLU's simplicity and effectiveness ensure it remains the default workhorse for most applications.</p>
  `,
  codeExamples: [
    {
      language: 'Python',
      code: `import numpy as np
import matplotlib.pyplot as plt

# Implement common activation functions
def relu(x):
  return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
  return np.where(x > 0, x, alpha * x)

def sigmoid(x):
  return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # clip for numerical stability

def tanh(x):
  return np.tanh(x)

def softmax(x):
  exp_x = np.exp(x - np.max(x))  # subtract max for numerical stability
  return exp_x / exp_x.sum(axis=0)

def swish(x):
  return x * sigmoid(x)

# Test activations
x = np.linspace(-5, 5, 100)

print("Activation function outputs at x=2.0:")
print(f"ReLU: {relu(2.0):.4f}")
print(f"Leaky ReLU: {leaky_relu(2.0):.4f}")
print(f"Sigmoid: {sigmoid(2.0):.4f}")
print(f"Tanh: {tanh(2.0):.4f}")
print(f"Swish: {swish(2.0):.4f}")

print(f"\\nActivation outputs at x=-2.0:")
print(f"ReLU: {relu(-2.0):.4f}")
print(f"Leaky ReLU: {leaky_relu(-2.0):.4f}")
print(f"Sigmoid: {sigmoid(-2.0):.4f}")
print(f"Tanh: {tanh(-2.0):.4f}")
print(f"Swish: {swish(-2.0):.4f}")

# Softmax example for multi-class
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"\\nSoftmax({logits}) = {probs}")
print(f"Sum of probabilities: {probs.sum():.4f}")`,
      explanation: 'Implements common activation functions and demonstrates their behavior. Shows how different activations handle positive and negative inputs, and how softmax converts logits to probabilities.'
    },
    {
      language: 'Python',
      code: `import torch
import torch.nn as nn
import torch.nn.functional as F

# Compare activation functions in PyTorch
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

print("Comparing PyTorch Activations:")
print(f"Input: {x}")
print(f"ReLU: {F.relu(x)}")
print(f"Leaky ReLU: {F.leaky_relu(x, negative_slope=0.01)}")
print(f"ELU: {F.elu(x)}")
print(f"Sigmoid: {torch.sigmoid(x)}")
print(f"Tanh: {torch.tanh(x)}")
print(f"GELU: {F.gelu(x)}")

# Demonstrate dying ReLU problem
print("\\n--- Dying ReLU Demonstration ---")
# Simulating a neuron that receives large negative weighted sum
large_negative = torch.tensor([-10.0, -20.0, -30.0])
relu_out = F.relu(large_negative)
leaky_relu_out = F.leaky_relu(large_negative, negative_slope=0.01)

print(f"Large negative inputs: {large_negative}")
print(f"ReLU output: {relu_out} (all zeros - dying ReLU!)")
print(f"Leaky ReLU output: {leaky_relu_out} (small negative values preserved)")

# Neural network with different activations
class NetworkWithActivation(nn.Module):
  def __init__(self, activation='relu'):
      super().__init__()
      self.fc1 = nn.Linear(10, 64)
      self.fc2 = nn.Linear(64, 32)
      self.fc3 = nn.Linear(32, 1)
      self.activation_name = activation

  def forward(self, x):
      if self.activation_name == 'relu':
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
      elif self.activation_name == 'leaky_relu':
          x = F.leaky_relu(self.fc1(x))
          x = F.leaky_relu(self.fc2(x))
      elif self.activation_name == 'tanh':
          x = torch.tanh(self.fc1(x))
          x = torch.tanh(self.fc2(x))
      return self.fc3(x)

# Create models with different activations
model_relu = NetworkWithActivation('relu')
model_leaky = NetworkWithActivation('leaky_relu')
model_tanh = NetworkWithActivation('tanh')

print(f"\\nModels created with different activations")
print(f"ReLU model parameters: {sum(p.numel() for p in model_relu.parameters())}")`,
      explanation: 'Demonstrates PyTorch activation functions and the dying ReLU problem. Shows how Leaky ReLU preserves small gradients for negative inputs, preventing neurons from dying.'
    }
  ],
  interviewQuestions: [
    {
      question: 'Why do we need activation functions in neural networks?',
      answer: 'Activation functions are essential because they introduce **non-linearity** into neural networks, enabling them to learn and represent complex patterns. Without activation functions, a neural network would be nothing more than a series of linear transformations (matrix multiplications and additions). Since the composition of linear functions is itself linear, even a deep network with many layers would be equivalent to a single linear model, severely limiting its representational power to only learn linearly separable patterns.\n\nMathematically, if we have layers computing **z₁ = W₁x + b₁**, **z₂ = W₂z₁ + b₂**, etc., without activation functions, the entire network reduces to **z_final = W_combined × x + b_combined** for some combined weight matrix and bias vector. This means the network could only learn linear relationships like **y = mx + b**, making it impossible to solve problems like XOR, learn polynomial relationships, or recognize complex patterns in images, text, or other high-dimensional data.\n\nActivation functions transform the linear outputs **z = Wx + b** into non-linear outputs **a = f(z)**, where **f** is the activation function. This non-linear transformation allows each layer to learn different types of feature combinations and enables the network to approximate any continuous function (universal approximation theorem). Different activation functions serve different purposes: **ReLU** introduces non-linearity while being computationally efficient, **sigmoid** outputs probabilities for binary classification, **tanh** provides zero-centered outputs, and **softmax** creates probability distributions for multi-class classification.\n\nWithout activation functions, deep learning as we know it wouldn\'t exist. The ability to stack many non-linear transformations is what enables neural networks to learn hierarchical representations—early layers detecting simple patterns like edges, middle layers combining these into more complex shapes, and later layers recognizing complete objects or concepts. This hierarchical feature learning is fundamental to the success of deep learning in computer vision, natural language processing, and many other domains.'
    },
    {
      question: 'What is the vanishing gradient problem and how does ReLU help?',
      answer: 'The **vanishing gradient problem** occurs when gradients become exponentially smaller as they propagate backward through deep networks during training. This happens because backpropagation multiplies gradients from each layer using the chain rule, and when these individual gradients are consistently less than 1, their cumulative product approaches zero. Traditional activation functions like **sigmoid** and **tanh** are particularly problematic because their derivatives have maximum values of 0.25 and 1.0 respectively, and typically much smaller values across most of their range.\n\nWhen gradients vanish, early layers in the network receive extremely small weight updates and essentially stop learning. This creates a situation where later layers might train reasonably well while earlier layers remain stuck with poor feature representations. Since early layers typically learn fundamental features that later layers build upon, this severely limits the network\'s ability to learn complex patterns and often results in poor overall performance.\n\n**ReLU (Rectified Linear Unit)** addresses this problem through its simple definition: **f(z) = max(0, z)**. For positive inputs, ReLU has a derivative of exactly 1, meaning gradients pass through unchanged during backpropagation. This eliminates the multiplicative shrinking effect that plagues sigmoid and tanh networks. For negative inputs, ReLU outputs zero with a gradient of zero, which creates sparsity but doesn\'t contribute to the vanishing gradient problem.\n\nReLU\'s benefits extend beyond solving vanishing gradients: it\'s **computationally efficient** (just a max operation), **promotes sparsity** (many neurons output zero), and **doesn\'t saturate** for positive values (unlike sigmoid/tanh which flatten out for large inputs). However, ReLU isn\'t perfect—it suffers from the "dying ReLU" problem where neurons can get stuck outputting zero. This led to variations like **Leaky ReLU**, **ELU**, and **Swish** that maintain ReLU\'s gradient-friendly properties while addressing its limitations. The introduction of ReLU was crucial for enabling training of very deep networks and was instrumental in the deep learning revolution.'
    },
    {
      question: 'Explain the dying ReLU problem and how to address it.',
      answer: 'The **dying ReLU problem** occurs when ReLU neurons become permanently inactive, always outputting zero regardless of the input. This happens when a neuron\'s weighted sum **z = w·x + b** becomes consistently negative across all training examples. Since ReLU outputs zero for negative inputs and has a gradient of zero in this region, these neurons receive no gradient signal during backpropagation and their weights never update. Once "dead," these neurons contribute nothing to the network\'s learning or predictions.\n\nThis problem typically arises from **poor weight initialization** (weights initialized too negative), **learning rates that are too high** (causing large weight updates that push neurons into the negative region), or **unfortunate weight updates** during training that cause a neuron to output negative values for all training examples. When a significant portion of neurons die, the network\'s effective capacity is reduced, leading to underfitting and poor performance. In extreme cases, entire layers can become inactive.\n\nSeveral **solutions** address the dying ReLU problem: (1) **Leaky ReLU** uses **f(z) = max(αz, z)** where **α** (typically 0.01) is a small positive slope for negative inputs, allowing small gradients to flow through and potentially "revive" dead neurons, (2) **Parametric ReLU (PReLU)** makes the negative slope **α** a learnable parameter, (3) **Exponential Linear Unit (ELU)** uses exponential decay for negative values: **f(z) = z if z > 0, else α(e^z - 1)**, providing smooth gradients and zero-centered outputs.\n\n**Prevention strategies** include: **proper weight initialization** using methods like He initialization which accounts for ReLU\'s properties, **appropriate learning rates** to avoid large weight swings, **batch normalization** to keep activations in reasonable ranges, and **monitoring neuron activation statistics** during training to detect dead neurons early. Modern architectures often use **Swish** (**f(z) = z·sigmoid(z)**) or **GELU** which are smooth, non-monotonic functions that largely avoid the dying neuron problem while maintaining many of ReLU\'s benefits. The choice of activation function and proper initialization has become crucial for training robust deep networks.'
    },
    {
      question: 'When would you use sigmoid vs tanh vs ReLU?',
      answer: '**Sigmoid** (**f(z) = 1/(1 + e^(-z))**) is primarily used in **output layers for binary classification** because it outputs values between 0 and 1 that can be interpreted as probabilities. Its smooth, S-shaped curve makes it useful when you need probabilistic outputs, but it should generally be avoided in hidden layers due to vanishing gradient problems. Sigmoid saturates for large positive or negative inputs (outputs approach 1 or 0 with gradients near zero), making it problematic for deep networks. Use sigmoid when you need probability outputs for binary decisions or when interpretability of neuron outputs as probabilities is important.\n\n**Tanh** (**f(z) = (e^z - e^(-z))/(e^z + e^(-z))**) outputs values between -1 and 1, making it **zero-centered** unlike sigmoid. This property can help with training dynamics since the outputs can be both positive and negative, leading to more balanced weight updates. Tanh was commonly used in hidden layers before ReLU became popular, and it\'s still useful in **RNNs and LSTMs** where the zero-centered property helps with learning long-term dependencies. However, like sigmoid, tanh suffers from vanishing gradients and is less commonly used in deep feedforward networks.\n\n**ReLU** (**f(z) = max(0, z)**) has become the **default choice for hidden layers** in deep networks due to its computational efficiency, ability to mitigate vanishing gradients, and promotion of sparse representations. ReLU is ideal for **convolutional neural networks**, **deep feedforward networks**, and most modern architectures. It\'s particularly effective when you have sufficient data and proper initialization. However, avoid ReLU in the output layer for regression (use linear activation) or multi-class classification (use softmax).\n\n**Selection guidelines**: Use **sigmoid** for binary classification outputs and when you need probability interpretation; use **tanh** for RNN hidden states, when you need zero-centered outputs, or in shallow networks; use **ReLU** for hidden layers in deep networks, CNNs, and when computational efficiency matters. For modern deep learning, the typical pattern is **ReLU variants** (Leaky ReLU, ELU, Swish) in hidden layers and **task-specific activations** (sigmoid, softmax, linear) in output layers. Consider the network depth, task requirements, and training dynamics when making your choice.'
    },
    {
      question: 'Why is softmax used in the output layer for multi-class classification?',
      answer: '**Softmax** is the standard activation function for multi-class classification output layers because it converts a vector of real numbers (logits) into a **probability distribution** where all values are positive and sum to 1. Given input vector **z = [z₁, z₂, ..., zₖ]**, softmax computes **softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)**. This ensures each output represents the predicted probability of that class, making the results interpretable and suitable for decision-making under uncertainty.\n\nThe **exponential function** in softmax serves several important purposes: it ensures all outputs are positive (since e^x > 0 for any real x), it amplifies differences between logits (larger logits get disproportionately larger probabilities), and it provides smooth gradients for training. The **normalization by the sum** ensures the outputs form a valid probability distribution. This amplification property means that even small differences in logits translate to clear probability differences, helping the model make confident predictions.\n\nSoftmax naturally pairs with **cross-entropy loss** for multi-class classification. Cross-entropy loss measures the difference between predicted probabilities and true labels (one-hot encoded), and its gradient with respect to logits has a particularly clean form when combined with softmax: **gradient = predicted_probability - true_probability**. This mathematical elegance leads to stable training dynamics and clear error signals for learning.\n\nPractical advantages include: **interpretable outputs** (probabilities can guide decision-making), **temperature scaling** capability (dividing logits by temperature before softmax controls prediction confidence), **compatibility** with techniques like label smoothing and knowledge distillation, and **multinomial sampling** for generating diverse outputs. Softmax also handles **variable numbers of classes** well and naturally extends to hierarchical classification tasks. While alternatives like label smoothing or focal loss modify the training objective, softmax remains the standard output activation because it provides the probabilistic interpretation that most multi-class applications require for uncertainty quantification and decision-making.'
    },
    {
      question: 'What is the difference between ReLU and Leaky ReLU?',
      answer: '**ReLU** (**f(z) = max(0, z)**) and **Leaky ReLU** (**f(z) = max(αz, z)** where **α** is typically 0.01) differ primarily in how they handle negative inputs. ReLU completely zeros out negative values, while Leaky ReLU allows a small, non-zero output for negative inputs through the leak coefficient **α**. This seemingly small change has significant implications for training dynamics and network behavior.\n\nThe key difference lies in **gradient flow**: ReLU has a gradient of 0 for negative inputs, meaning neurons that consistently receive negative inputs stop learning entirely (the dying ReLU problem). Leaky ReLU maintains a small gradient **α** for negative inputs, allowing these neurons to potentially recover and contribute to learning. This makes Leaky ReLU more robust to poor initialization and aggressive learning rates that might push neurons into the "dead" state.\n\n**Computational complexity** differs slightly: ReLU requires only a comparison and max operation, while Leaky ReLU requires a comparison and a multiplication. However, this difference is negligible in practice. **Memory usage** is identical since both functions are applied element-wise without storing intermediate values. The mathematical properties also differ: ReLU creates **exact sparsity** (many neurons output exactly zero), while Leaky ReLU creates **approximate sparsity** (small but non-zero values for negative inputs).\n\n**Performance trade-offs** depend on the specific problem and architecture. ReLU often works well when properly initialized and can lead to more interpretable sparse representations. Leaky ReLU provides more robustness during training and can achieve better performance when the dataset is small or when neurons are prone to dying. However, the performance difference is often minimal in well-designed networks with proper initialization and batch normalization.\n\n**When to choose**: Use **ReLU** as the default for most applications, especially with proper initialization (He initialization) and batch normalization. Consider **Leaky ReLU** when experiencing dying neuron problems, working with smaller datasets, or when training is unstable. Modern variants like **ELU**, **Swish**, or **GELU** often outperform both but at increased computational cost. The choice between ReLU and Leaky ReLU is often less important than proper network architecture, initialization, and regularization techniques.'
    }
  ],
  quizQuestions: [
    {
      id: 'activ-q1',
      question: 'What would happen if you removed all activation functions from a deep neural network?',
      options: [
        'The network would train faster',
        'The network would collapse to a single linear transformation',
        'The network would become more accurate',
        'No change in behavior'
      ],
      correctAnswer: 1,
      explanation: 'Without activation functions, stacking multiple linear layers (y = Wx + b) just creates another linear transformation. The entire network becomes equivalent to a single linear layer, unable to learn non-linear patterns.'
    },
    {
      id: 'activ-q2',
      question: 'Your deep network with sigmoid activations learns quickly for the first few epochs but then stops improving. What is the most likely cause?',
      options: [
        'Learning rate is too high',
        'Vanishing gradient problem - gradients become too small in early layers',
        'The model has already converged',
        'Need more training data'
      ],
      correctAnswer: 1,
      explanation: 'Sigmoid activation has derivatives ≤ 0.25. In deep networks, gradients are multiplied across layers (chain rule), causing them to vanish exponentially. Early layers stop learning. Solution: use ReLU or add batch normalization.'
    },
    {
      id: 'activ-q3',
      question: 'You are building a binary classifier. Which activation should you use in the output layer?',
      options: [
        'ReLU',
        'Softmax',
        'Sigmoid',
        'Tanh'
      ],
      correctAnswer: 2,
      explanation: 'Sigmoid outputs values between 0 and 1, perfect for binary classification probabilities. Softmax is for multi-class (>2 classes), ReLU is for hidden layers, and tanh outputs [-1, 1] which is not suitable for probabilities.'
    }
  ]
};
