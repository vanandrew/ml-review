import { Topic } from '../../../types';

export const fewShotLearning: Topic = {
  id: 'few-shot-learning',
  title: 'Few-Shot Learning',
  category: 'advanced',
  description: 'Learning from very limited labeled examples',
  content: `
    <h2>Few-Shot Learning: Learning from Limited Examples</h2>
    
    <p>Traditional deep learning thrives on massive labeled datasets—ImageNet's 14 million images, BERT's billions of words. But what happens when we can only afford to label 5 examples per class? Few-Shot Learning (FSL) addresses this fundamental challenge: teaching machines to generalize from minimal supervision, mimicking human-like learning capabilities. When a child sees a few zebras, they can recognize new zebras—can we build AI systems with similar sample efficiency?</p>

    <p>Few-shot learning is critical in domains where labeled data is expensive (medical imaging requires expert radiologists), scarce (rare disease diagnosis with few patients), dangerous to collect (nuclear reactor anomaly detection), or rapidly changing (new product categories in e-commerce). Rather than treating each new task as starting from scratch requiring thousands of examples, few-shot learning systems learn how to learn efficiently from prior experience.</p>

    <h3>The Few-Shot Learning Problem</h3>

    <h4>Problem Formulation: N-Way K-Shot Classification</h4>
    
    <p>Few-shot learning problems are typically formulated as <strong>N-way K-shot</strong> classification tasks. This means we must distinguish between N classes, having only K labeled examples per class. The learning happens in two stages:</p>

    <h5>Visual Example: 5-Way 1-Shot Task</h5>
    <pre class="code-block">
┌────────────────────────────────────────────────────────────────┐
│           SUPPORT SET (Training - K=1 example per class)       │
├────────────────────────────────────────────────────────────────┤
│  Class 1: [🐈]     Class 2: [🐶]     Class 3: [🐦]              │
│     Cat            Dog            Bird                         │
│                                                                │
│  Class 4: [🐎]     Class 5: [🐰]                                │
│    Horse          Rabbit                                       │
└────────────────────────────────────────────────────────────────┘
                            │
                            │ Learn from these 5 examples only!
                            ▼
┌────────────────────────────────────────────────────────────────┐
│              QUERY SET (Testing - Classify these!)             │
├────────────────────────────────────────────────────────────────┤
│  [🐈?]  Which class?  →  Predict: Cat (Class 1)                │
│  [🐦?]  Which class?  →  Predict: Bird (Class 3)               │
│  [🐶?]  Which class?  →  Predict: Dog (Class 2)                │
└────────────────────────────────────────────────────────────────┘

Key Insight: Must generalize from just 5 total examples!
    </pre>

    <p><strong>Support Set (S):</strong> The small training set containing K examples for each of N classes. This is all the supervision available—the model must learn from just these K×N examples. For a 5-way 1-shot task, we have only 5 total training examples, one per class.</p>

    <p><strong>Query Set (Q):</strong> The test examples we want to classify using only the knowledge from the support set. The model cannot update its parameters using query examples—it must generalize immediately from the support set alone.</p>

    <h4>Problem Variants by Shot Number</h4>

    <p><strong>Zero-shot learning (K=0):</strong> No labeled examples at all! Instead, we have class descriptions, attributes, or semantic embeddings. For example, classifying "zebra" images without seeing any zebras, using only the description "horse-like animal with black and white stripes." This requires connecting visual features to semantic information.</p>

    <p><strong>One-shot learning (K=1):</strong> A single example per class—the hardest supervised scenario. With one cat image, can you recognize all cats? One-shot learning is the gold standard for testing sample efficiency, forcing models to extract maximum information from minimal data.</p>

    <p><strong>Few-shot learning (K=2-10):</strong> A handful of examples per class, more than one-shot but far less than traditional learning. Even 10 examples per class (50 total for 5-way classification) is orders of magnitude less than conventional datasets.</p>

    <h3>Core Approaches to Few-Shot Learning</h3>

    <h4>1. Metric Learning: Learning Similarity</h4>

    <p>The metric learning paradigm approaches few-shot learning by learning an embedding space where semantically similar examples are close together and dissimilar ones are far apart. Classification becomes a simple distance computation in this learned space—assign queries to the nearest support examples.</p>

    <h5>Siamese Networks: Pairwise Comparison</h5>

    <p>Siamese networks use twin neural networks with shared weights to process pairs of examples. During training, we feed pairs of images: some from the same class (positive pairs) and some from different classes (negative pairs). The network learns embeddings where same-class pairs have small distances and different-class pairs have large distances.</p>

    <p><strong>Contrastive loss</strong> drives this learning: $L = y \\times d^2 + (1-y) \\times \\max(0, \\text{margin}-d)^2$, where $y=1$ for same-class pairs and $y=0$ for different-class pairs, $d$ is the distance between embeddings, and margin defines how far apart different-class pairs should be. At test time, we compare query embeddings to support embeddings and classify based on smallest distance.</p>

    <h5>Prototypical Networks: Class Representatives</h5>

    <p>Prototypical Networks simplify metric learning by computing a single <strong>prototype</strong> (representative embedding) per class as the mean of all support examples for that class. Given support set embeddings, we compute:</p>

    <p style="text-align: center;">$c_k = \\frac{1}{K} \\sum f_\\theta(x_i)$ for all $x_i$ in class $k$</p>

    <p>where $f_\\theta$ is the embedding network and $c_k$ is the prototype for class $k$. To classify a query example $x$, we embed it as $f_\\theta(x)$ and find the nearest prototype using Euclidean distance:</p>

    <p style="text-align: center;">$\\hat{y} = \\arg\\min_k d(f_\\theta(x), c_k)$</p>

    <h5>Visual: Embedding Space with Prototypes</h5>
    <pre class="code-block">
Learned Embedding Space (2D projection for visualization):

      │
      │     🐶     🐶              Class prototypes:
      │       ╲   ╱                ● = Class center
  Cat │    🐶  ●  🐶  Dog           🐶 = Dog samples
  ●   │       ╱   ╲                🐈 = Cat samples
  🐈    │    🐶     🐶                🐦 = Bird samples
   ╲    │                            ? = Query
 🐈  ╲  │
────────┼────────────────────────────────
   ╱    │                   🐦
 🐈     │                  ╱   ╲
      │     ?          🐦  ●  🐦  Bird
      │      ╲             ╲   ╱
      │       ╲          🐦     🐦
      │        ╲
      │      Classify query:
      │      1. Embed query → ?
      │      2. Find nearest prototype
      │      3. Distance to Bird ● is smallest
      │      4. Predict: Bird!

Key: Same-class examples cluster together,
   different classes are separated
    </pre>

    <p>This approach is elegant and effective—prototypes act as cluster centers in the embedding space. Training uses episodic learning where the model repeatedly solves simulated few-shot tasks, learning an embedding space where prototypical classification works well.</p>

    <h5>Matching Networks: Attention Over Support Set</h5>

    <p>Matching Networks extend prototypical ideas by using attention mechanisms to weight different support examples when classifying queries. Rather than equal weighting (as in prototypes), they learn which support examples are most relevant for each query through an attention mechanism, effectively implementing a differentiable k-nearest neighbors classifier.</p>

    <h5>Relation Networks: Learned Similarity Metrics</h5>

    <p>Instead of using hand-crafted distance metrics (Euclidean, cosine), Relation Networks learn the similarity function itself using a neural network. Given embeddings of a query and a support example, a relation module (small neural network) outputs a similarity score. This provides more flexibility—the network learns task-specific notions of similarity rather than assuming Euclidean distance is appropriate.</p>

    <h6>Architecture Details</h6>
    <ul>
      <li><strong>Embedding module $f_\\phi$:</strong> CNN or other encoder producing feature maps</li>
      <li><strong>Relation module $g_\\phi$:</strong> Small neural network (2-3 layers) taking concatenated features</li>
      <li><strong>Process:</strong>
        <ol>
          <li>Embed support examples: $f_\\phi(x_{\\text{support}})$</li>
          <li>Embed query: $f_\\phi(x_{\\text{query}})$</li>
          <li>Concatenate feature pairs: $[f_\\phi(x_{\\text{query}}), f_\\phi(x_{\\text{support}})]$</li>
          <li>Compute relation score: $r = g_\\phi([f_\\phi(x_{\\text{query}}), f_\\phi(x_{\\text{support}})])$</li>
          <li>Classify as class with highest relation score</li>
        </ol>
      </li>
    </ul>

    <h6>Advantages over Fixed Metrics</h6>
    <ul>
      <li><strong>Task-adaptive:</strong> Learns what "similar" means for specific problem</li>
      <li><strong>Non-linear relations:</strong> Can capture complex similarity patterns</li>
      <li><strong>Better performance:</strong> Often outperforms fixed distance metrics</li>
      <li><strong>Interpretable:</strong> Can visualize learned relation patterns</li>
    </ul>

    <h4>2. Meta-Learning: Learning to Learn</h4>

    <p>Meta-learning takes a different philosophical approach: instead of learning a good embedding space, learn a learning algorithm itself. The key insight is that some model initializations are better starting points for rapid adaptation than others. Meta-learning finds these optimal initializations.</p>

    <h5>MAML: Model-Agnostic Meta-Learning</h5>

    <p>MAML (Model-Agnostic Meta-Learning) is the most influential meta-learning algorithm. It learns initial parameters $\\theta$ such that after a few gradient steps on a new task with minimal data, the model achieves good performance. This involves two nested optimization loops:</p>

    <h6>Visual: MAML Two-Level Optimization</h6>
    <pre class="code-block">
MAML: Learning to Learn Fast

┌────────────────────────────────────────────────────────────────┐
│                    META-LEARNING (Outer Loop)                  │
│                                                                │
│  Initial parameters θ  (The goal: find best starting point)    │
│         │                                                      │
│         ├──────── Task 1 ──────────┐                           │
│         │                          │                           │
│         │  ┌────────────────────┐  │                           │
│         │  │  INNER LOOP (Adapt)│  │                           │
│         │  │  θ → θ'_1          │  │                           │
│         │  └────────────────────┘  │                           │
│         │          │               │                           │
│         │          ▼               │                           │
│         │  Evaluate on query set   │                           │
│         │                          │                           │
│         ├──────── Task 2 ──────────┘                           │
│         │                          │                           │
│         │  ┌────────────────────┐  │                           │
│         │  │  INNER LOOP (Adapt)│  │                           │
│         │  │  θ → θ'_2          │  │                           │
│         │  └────────────────────┘  │                           │
│         │          │               │                           │
│         │          ▼               │                           │
│         │  Evaluate on query set   │                           │
│         │                          │                           │
│         └──────── Task 3 ──────────┘                           │
│                    │                                           │
│            [Same inner loop]                                   │
│                    │                                           │
│                    ▼                                           │
│  Update θ to minimize average query loss across all tasks      │
│  (Optimize for good post-adaptation performance)               │
└────────────────────────────────────────────────────────────────┘

Key Insight: Gradient descent THROUGH gradient descent!
           (Second-order optimization)
    </pre>

    <p><strong>Inner loop (task-specific adaptation):</strong> Given a new task with support set $D_{\\text{train}}$, perform a few gradient descent steps:</p>

    <p style="text-align: center;">$\\theta'_i = \\theta - \\alpha \\nabla_\\theta L_{D_{\\text{train}}}(\\theta)$</p>

    <p>where $\\alpha$ is the inner learning rate and $\\theta'_i$ are the task-adapted parameters after one or more gradient steps.</p>

    <p><strong>Outer loop (meta-optimization):</strong> Optimize the initialization $\\theta$ to minimize loss on query sets after inner-loop adaptation:</p>

    <p style="text-align: center;">$\\theta \\leftarrow \\theta - \\beta \\nabla_\\theta \\sum_{\\text{tasks}} L_{D_{\\text{test}}}(\\theta'_i)$</p>

    <p>where $\\beta$ is the meta-learning rate. This is gradient descent through gradient descent—we backpropagate through the inner loop optimization to find initializations that lead to good post-adaptation performance.</p>

    <p>The beauty of MAML is its model-agnosticity: it works with any model trainable by gradient descent. The learned initialization encodes prior knowledge about the task distribution, enabling rapid adaptation to new tasks from the same distribution.</p>

    <h4>3. Data Augmentation Approaches</h4>

    <p>When you have few examples, generate more! Data augmentation techniques create synthetic training data to alleviate data scarcity. <strong>Hallucination</strong> methods use generative models to synthesize new examples. <strong>Mixup</strong> creates virtual examples by linearly interpolating between pairs of examples and their labels. Transfer learning from large auxiliary datasets (ImageNet pre-training) provides strong feature representations that generalize with few examples.</p>

    <h4>4. Transductive and Semi-Supervised Methods</h4>

    <p>Graph-based methods leverage the structure of the query set itself (transductive inference). By constructing a graph where nodes are support and query examples and edges represent similarity, we can propagate labels from the few labeled support examples to unlabeled queries through the graph. This exploits the manifold structure of the data—nearby points in the embedding space should have similar labels.</p>

    <h3>Training Strategy: Episodic Learning</h3>

    <p>Few-shot learning requires a training methodology that matches the test-time scenario. <strong>Episodic training</strong> (also called meta-training) simulates few-shot episodes during training. Each training iteration:</p>

    <ol>
      <li><strong>Sample an episode:</strong> Randomly select N classes from training set</li>
      <li><strong>Create support set:</strong> Sample $K$ examples per class ($N \\times K$ total)</li>
      <li><strong>Create query set:</strong> Sample additional examples from same N classes</li>
      <li><strong>Train the model:</strong> Use only support set to classify query set</li>
      <li><strong>Update parameters:</strong> Based on query set performance</li>
    </ol>

    <p>By repeatedly solving different few-shot tasks, the model learns representations and strategies that generalize to new tasks. Critically, meta-training uses different classes than meta-testing—we evaluate on completely novel classes to test true few-shot generalization, not memorization.</p>

    <h3>Challenges and Considerations</h3>

    <p><strong>Overfitting risk:</strong> With 5 examples, models can easily memorize rather than generalize. Regularization, careful architecture design, and episodic training help mitigate this.</p>

    <p><strong>Domain shift:</strong> Meta-training classes and meta-testing classes come from the same distribution (e.g., all animals), but if meta-test introduces very different classes (e.g., vehicles when trained on animals), performance degrades. Few-shot learning assumes related task distributions.</p>

    <p><strong>Intra-class variation:</strong> A single example of "dog" might show a golden retriever—will the model recognize a chihuahua? Few examples struggle to capture full class diversity.</p>

    <p><strong>Computational cost:</strong> Meta-learning algorithms like MAML require second-order gradients and many episodes, increasing training time significantly.</p>

    <h3>Few-Shot Learning vs Transfer Learning</h3>

    <p>Both address learning with limited data, but differ in approach. <strong>Transfer learning</strong> pre-trains on a large source dataset (ImageNet), then fine-tunes on the target task. It assumes the target task has enough data for fine-tuning (typically hundreds of examples). <strong>Few-shot learning</strong> explicitly handles extreme data scarcity (1-10 examples), often using meta-learning to learn how to adapt rather than just learning good features. Transfer learning provides the features; few-shot learning provides the adaptation strategy.</p>

    <p>In practice, they're complementary: state-of-the-art few-shot systems often use pre-trained features (transfer learning) combined with meta-learning or metric learning for few-shot adaptation.</p>

    <h3>Real-World Applications</h3>

    <p><strong>Drug discovery:</strong> Predict whether a new molecule will bind to a protein target using only a handful of experimental measurements, where each experiment costs thousands of dollars.</p>

    <p><strong>Rare disease diagnosis:</strong> Detect rare diseases with only dozens of patient cases worldwide. Few-shot learning enables diagnostic models without massive patient datasets.</p>

    <p><strong>Personalization:</strong> Quickly adapt recommendation systems to new users with minimal interaction history, or customize voice assistants to individual speech patterns from brief recordings.</p>

    <p><strong>Robotics:</strong> Enable robots to learn new manipulation tasks from a few human demonstrations rather than thousands of trial-and-error attempts.</p>

    <p><strong>Low-resource NLP:</strong> Build language models for languages with limited digital text, using few-shot learning to transfer knowledge from high-resource languages.</p>

    <p><strong>Visual recognition:</strong> Recognize new product categories in e-commerce, new species in wildlife monitoring, or new defect types in manufacturing with minimal labeled examples.</p>

    <h3>Evaluation and Benchmarks</h3>

    <p>Few-shot learning is evaluated on standardized benchmarks designed to test generalization to novel classes. <strong>Omniglot</strong> (handwritten characters from 50 alphabets) tests recognition of completely new character sets. <strong>miniImageNet</strong> (100 classes, 600 images each) and <strong>tieredImageNet</strong> (more classes, hierarchical structure) test object recognition. <strong>Meta-Dataset</strong> includes multiple domains to test cross-domain generalization.</p>

    <p>Standard evaluation reports N-way K-shot accuracy across multiple randomly sampled episodes, with mean accuracy and 95% confidence intervals. Common settings include 5-way 1-shot and 5-way 5-shot classification.</p>

    <h3>Approach Comparison: Metric Learning vs Meta-Learning</h3>

    <table >
      <tr>
        <th>Aspect</th>
        <th>Metric Learning (e.g., Prototypical)</th>
        <th>Meta-Learning (e.g., MAML)</th>
      </tr>
      <tr>
        <td>Core Idea</td>
        <td>Learn embedding space where similar = close</td>
        <td>Learn initialization for fast adaptation</td>
      </tr>
      <tr>
        <td>Adaptation</td>
        <td>No adaptation needed (just compute distance)</td>
        <td>Few gradient steps on support set</td>
      </tr>
      <tr>
        <td>Inference Speed</td>
        <td>Fast (single forward pass + distance)</td>
        <td>Slower (requires gradient steps)</td>
      </tr>
      <tr>
        <td>Training Complexity</td>
        <td>Simpler, first-order optimization</td>
        <td>Complex, second-order optimization</td>
      </tr>
      <tr>
        <td>Interpretability</td>
        <td>High (can visualize embedding space)</td>
        <td>Lower (adaptation process less transparent)</td>
      </tr>
      <tr>
        <td>Flexibility</td>
        <td>Limited to distance-based classification</td>
        <td>Can adapt entire model behavior</td>
      </tr>
      <tr>
        <td>Performance</td>
        <td>Good, especially when similarity is well-defined</td>
        <td>Often better, but task-dependent</td>
      </tr>
      <tr>
        <td>Memory</td>
        <td>Low (just store embeddings)</td>
        <td>Higher (store gradients during adaptation)</td>
      </tr>
      <tr>
        <td>Best For</td>
        <td>Simple similarity-based tasks, fast inference</td>
        <td>Complex tasks requiring model adaptation</td>
      </tr>
    </table>

    <p>Few-shot learning represents a paradigm shift from data-hungry deep learning toward more sample-efficient AI. By learning how to learn from prior experience, few-shot systems achieve human-like rapid adaptation—a crucial step toward more general artificial intelligence.</p>
  `,
  codeExamples: [
    {
      language: 'Python',
      code: `import torch
import torch.nn as nn
import torch.nn.functional as F

# Prototypical Networks

class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
      super().__init__()
      self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
      self.bn = nn.BatchNorm2d(out_channels)
      self.relu = nn.ReLU()
      self.pool = nn.MaxPool2d(2)

  def forward(self, x):
      x = self.conv(x)
      x = self.bn(x)
      x = self.relu(x)
      x = self.pool(x)
      return x

class PrototypicalNetwork(nn.Module):
  def __init__(self, embedding_dim=64):
      super().__init__()
      # Feature extractor
      self.encoder = nn.Sequential(
          ConvBlock(3, 64),
          ConvBlock(64, 64),
          ConvBlock(64, 64),
          ConvBlock(64, embedding_dim)
      )

  def forward(self, x):
      return self.encoder(x).view(x.size(0), -1)

def compute_prototypes(support_embeddings, support_labels, n_classes):
  """
  Compute class prototypes (mean of support embeddings per class)

  Args:
      support_embeddings: [n_support, embedding_dim]
      support_labels: [n_support]
      n_classes: Number of classes

  Returns:
      prototypes: [n_classes, embedding_dim]
  """
  prototypes = []
  for c in range(n_classes):
      mask = (support_labels == c)
      class_embeddings = support_embeddings[mask]
      prototype = class_embeddings.mean(dim=0)
      prototypes.append(prototype)

  return torch.stack(prototypes)

def prototypical_loss(query_embeddings, prototypes, query_labels):
  """
  Compute prototypical loss (negative log probability)

  Args:
      query_embeddings: [n_query, embedding_dim]
      prototypes: [n_classes, embedding_dim]
      query_labels: [n_query]

  Returns:
      loss: scalar
  """
  # Compute distances to all prototypes
  dists = torch.cdist(query_embeddings, prototypes)  # [n_query, n_classes]

  # Convert to log probabilities (softmax over negative distances)
  log_p_y = F.log_softmax(-dists, dim=1)

  # Negative log likelihood loss
  loss = F.nll_loss(log_p_y, query_labels)

  return loss

# Example: 5-way 5-shot task
n_way = 5
k_shot = 5
n_query = 15

model = PrototypicalNetwork(embedding_dim=64)

# Support set: 5 classes × 5 examples = 25 images
support_images = torch.randn(n_way * k_shot, 3, 28, 28)
support_labels = torch.arange(n_way).repeat_interleave(k_shot)

# Query set: 15 images to classify
query_images = torch.randn(n_query, 3, 28, 28)
query_labels = torch.randint(0, n_way, (n_query,))

# Get embeddings
support_embeddings = model(support_images)
query_embeddings = model(query_images)

# Compute prototypes
prototypes = compute_prototypes(support_embeddings, support_labels, n_way)

# Compute loss
loss = prototypical_loss(query_embeddings, prototypes, query_labels)

# Predictions
dists = torch.cdist(query_embeddings, prototypes)
predictions = dists.argmin(dim=1)
accuracy = (predictions == query_labels).float().mean()

print(f"Loss: {loss.item():.4f}")
print(f"Accuracy: {accuracy.item():.4f}")`,
      explanation: 'Prototypical Networks implementation for few-shot learning using class prototypes.'
    },
    {
      language: 'Python',
      code: `import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

# MAML (Model-Agnostic Meta-Learning)

class MAMLModel(nn.Module):
  def __init__(self, input_dim=784, hidden_dim=256, output_dim=5):
      super().__init__()
      self.model = nn.Sequential(
          nn.Linear(input_dim, hidden_dim),
          nn.ReLU(),
          nn.Linear(hidden_dim, hidden_dim),
          nn.ReLU(),
          nn.Linear(hidden_dim, output_dim)
      )

  def forward(self, x, params=None):
      if params is None:
          return self.model(x)
      else:
          # Forward pass with custom parameters
          x = F.linear(x, params['model.0.weight'], params['model.0.bias'])
          x = F.relu(x)
          x = F.linear(x, params['model.2.weight'], params['model.2.bias'])
          x = F.relu(x)
          x = F.linear(x, params['model.4.weight'], params['model.4.bias'])
          return x

class MAML:
  def __init__(self, model, inner_lr=0.01, meta_lr=0.001, inner_steps=5):
      self.model = model
      self.inner_lr = inner_lr
      self.meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)
      self.inner_steps = inner_steps

  def inner_loop(self, support_x, support_y):
      """
      Adapt model to a specific task (inner loop)

      Args:
          support_x: Support set inputs
          support_y: Support set labels

      Returns:
          adapted_params: Task-specific parameters
      """
      # Clone current parameters
      params = OrderedDict(self.model.named_parameters())
      adapted_params = OrderedDict()
      for name, param in params.items():
          adapted_params[name] = param.clone()

      # Inner loop: gradient descent on support set
      for step in range(self.inner_steps):
          # Forward pass with current params
          logits = self.model(support_x)
          loss = F.cross_entropy(logits, support_y)

          # Compute gradients
          grads = torch.autograd.grad(
              loss,
              adapted_params.values(),
              create_graph=True  # For second-order gradients
          )

          # Update parameters
          adapted_params = OrderedDict()
          for (name, param), grad in zip(params.items(), grads):
              adapted_params[name] = param - self.inner_lr * grad

      return adapted_params

  def meta_train_step(self, tasks):
      """
      Meta-training step (outer loop)

      Args:
          tasks: List of (support_x, support_y, query_x, query_y) tuples

      Returns:
          meta_loss: Meta-training loss
      """
      self.meta_optimizer.zero_grad()
      meta_loss = 0

      for support_x, support_y, query_x, query_y in tasks:
          # Inner loop: adapt to task
          adapted_params = self.inner_loop(support_x, support_y)

          # Outer loop: evaluate on query set with adapted params
          query_logits = self.model(query_x, adapted_params)
          task_loss = F.cross_entropy(query_logits, query_y)

          meta_loss += task_loss

      # Meta-update
      meta_loss = meta_loss / len(tasks)
      meta_loss.backward()
      self.meta_optimizer.step()

      return meta_loss.item()

# Example usage
input_dim = 784
n_way = 5
k_shot = 5
n_query = 15

model = MAMLModel(input_dim=input_dim, output_dim=n_way)
maml = MAML(model, inner_lr=0.01, meta_lr=0.001, inner_steps=5)

# Simulate batch of tasks
batch_size = 4
tasks = []
for _ in range(batch_size):
  # Support set: 5-way 5-shot
  support_x = torch.randn(n_way * k_shot, input_dim)
  support_y = torch.arange(n_way).repeat_interleave(k_shot)

  # Query set
  query_x = torch.randn(n_query, input_dim)
  query_y = torch.randint(0, n_way, (n_query,))

  tasks.append((support_x, support_y, query_x, query_y))

# Meta-training step
meta_loss = maml.meta_train_step(tasks)
print(f"Meta-training loss: {meta_loss:.4f}")

# At test time: adapt to new task
test_support_x = torch.randn(n_way * k_shot, input_dim)
test_support_y = torch.arange(n_way).repeat_interleave(k_shot)

adapted_params = maml.inner_loop(test_support_x, test_support_y)

# Test on query set
test_query_x = torch.randn(n_query, input_dim)
test_query_y = torch.randint(0, n_way, (n_query,))

with torch.no_grad():
  logits = maml.model(test_query_x, adapted_params)
  predictions = logits.argmax(dim=1)
  accuracy = (predictions == test_query_y).float().mean()

print(f"Test accuracy: {accuracy.item():.4f}")`,
      explanation: 'MAML implementation showing meta-learning with inner loop adaptation and outer loop meta-optimization.'
    }
  ],
  interviewQuestions: [
    {
      question: 'What is the difference between few-shot learning and transfer learning?',
      answer: `Transfer learning pre-trains on large datasets then fine-tunes on target task with sufficient labeled data. Few-shot learning specifically handles scenarios with very few examples (1-10) per class. Key differences: (1) Data requirements - transfer learning needs moderate target data, few-shot needs minimal, (2) Adaptation strategy - transfer learning fine-tunes, few-shot often meta-learns, (3) Objective - transfer learning leverages related knowledge, few-shot learns to learn quickly from limited examples.`
    },
    {
      question: 'Explain how Prototypical Networks work.',
      answer: `Prototypical Networks create prototype representations for each class by averaging support set embeddings. Process: (1) Embed support and query examples using shared network, (2) Compute class prototypes as mean of support embeddings per class, (3) Classify queries based on distance to nearest prototype (typically Euclidean). Benefits: simple, interpretable, works well empirically. The embedding space is learned to make same-class examples cluster and different-class examples separate, enabling effective nearest-prototype classification.`
    },
    {
      question: 'What is the key idea behind MAML?',
      answer: `MAML (Model-Agnostic Meta-Learning) learns initialization parameters that enable fast adaptation to new tasks with few gradient steps. Two-level optimization: (1) Inner loop - adapt to specific task using few examples, (2) Outer loop - optimize initialization for good post-adaptation performance across tasks. Key insight: some initializations are better starting points for learning new tasks. MAML is model-agnostic (works with any gradient-based model) and learns representations that are easy to fine-tune rather than task-specific.`
    },
    {
      question: 'What is episodic training in few-shot learning?',
      answer: `Episodic training simulates few-shot scenarios during training by creating artificial episodes that mimic test conditions. Each episode: (1) Sample N classes from training set, (2) Sample K examples per class (support set), (3) Sample query examples to classify, (4) Train to classify queries using only support examples. This forces the model to learn from limited examples repeatedly, developing meta-learning capabilities. Ensures training and testing conditions match, leading to better few-shot performance.`
    },
    {
      question: 'How does metric learning help in few-shot scenarios?',
      answer: `Metric learning learns embedding spaces where similar examples are close and dissimilar ones are far apart. In few-shot learning: (1) Embeds support and query examples into learned space, (2) Uses distance-based classification (nearest neighbor, prototype), (3) Works well with limited data as it leverages similarity rather than complex decision boundaries. Benefits: interpretable, requires fewer parameters to learn, naturally handles new classes. Common losses: triplet loss, contrastive loss, prototypical loss.`
    },
    {
      question: 'What are the trade-offs between metric learning and meta-learning approaches?',
      answer: `Metric learning: Simpler to implement, more interpretable, faster inference, but may be limited by embedding quality and distance metric choice. Works well when good similarity measures exist. Meta-learning: More flexible, can adapt entire model behavior, potentially better performance, but more complex training, slower adaptation, risk of overfitting to meta-training distribution. Choose metric learning for simplicity and interpretability, meta-learning for maximum flexibility and performance when sufficient meta-training data available.`
    }
  ],
  quizQuestions: [
    {
      id: 'fsl1',
      question: 'What does "5-way 1-shot" mean in few-shot learning?',
      options: ['5 examples per class', '5 classes with 1 example each', '1 class with 5 examples', '5 training epochs'],
      correctAnswer: 1,
      explanation: 'N-way K-shot refers to a classification task with N classes and K examples per class. So 5-way 1-shot means 5 classes with 1 labeled example each.'
    },
    {
      id: 'fsl2',
      question: 'What is the main idea of Prototypical Networks?',
      options: ['Train on prototypes', 'Classify by distance to class prototypes', 'Generate prototypes', 'Prune prototypes'],
      correctAnswer: 1,
      explanation: 'Prototypical Networks compute a prototype (mean embedding) for each class from the support set, then classify queries by assigning them to the nearest prototype.'
    },
    {
      id: 'fsl3',
      question: 'What does MAML learn?',
      options: ['Final model weights', 'Good initialization for fast adaptation', 'Distance metric', 'Data augmentation'],
      correctAnswer: 1,
      explanation: 'MAML (Model-Agnostic Meta-Learning) learns an initialization that allows rapid adaptation to new tasks with just a few gradient steps.'
    }
  ]
};
