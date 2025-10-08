import { Topic } from '../../../types';

export const positionalEncoding: Topic = {
  id: 'positional-encoding',
  title: 'Positional Encoding',
  category: 'transformers',
  description: 'Methods to inject position information into Transformer models',
  content: `
    <h2>Positional Encoding: Injecting Sequential Information</h2>
    <p>A fundamental challenge in Transformer architecture is that self-attention is inherently permutation-invariant—it treats the input as an unordered set rather than a sequence. The model has no way to distinguish the sentence "The cat ate the fish" from "The fish ate the cat" based on attention alone. Positional encoding solves this by injecting information about token positions into the model, enabling it to leverage sequential structure while maintaining the parallelization benefits of attention.</p>

    <h3>The Problem: Attention Is Permutation-Invariant</h3>
    <p>Self-attention computes relationships between all pairs of positions based solely on content similarity. If we permute the input sequence, the attention outputs permute identically—there's no sensitivity to order:</p>

    <h4>Why Order Matters</h4>
    <ul>
      <li><strong>Language syntax:</strong> Word order determines meaning ("dog bites man" ≠ "man bites dog")</li>
      <li><strong>Temporal dependencies:</strong> Events have causal ordering ("I ate, then slept" ≠ "I slept, then ate")</li>
      <li><strong>Compositional structure:</strong> Phrases depend on adjacency ("New York" vs "York New")</li>
      <li><strong>Reference resolution:</strong> Pronouns refer to specific earlier mentions, not arbitrary tokens</li>
      <li><strong>Permutation invariance:</strong> Self-attention is permutation-equivariant</li>
      <li><strong>No sequential processing:</strong> Unlike RNNs, no inherent order</li>
      <li><strong>Need explicit encoding:</strong> Must add position information</li>
    </ul>

    <p><strong>Mathematical perspective:</strong> For any permutation $\\pi$, $\\text{Attention}(Q_\\pi, K_\\pi, V_\\pi) = (\\text{Attention}(Q, K, V))_\\pi$. The operation preserves permutation structure, making position information invisible to the model.</p>

    <h3>Absolute Positional Encoding: The Sinusoidal Approach</h3>
    <p>The original Transformer paper introduced elegant sinusoidal positional encodings that inject absolute position information while possessing useful mathematical properties.</p>

    <h4>The Sinusoidal Formula</h4>
    <p>For position $\\text{pos}$ and dimension $i$:</p>
    <ul>
      <li>$$\\text{PE}(\\text{pos}, 2i) = \\sin\\left(\\frac{\\text{pos}}{10000^{2i/d_{\\text{model}}}}\\right)$$</li>
      <li>$$\\text{PE}(\\text{pos}, 2i+1) = \\cos\\left(\\frac{\\text{pos}}{10000^{2i/d_{\\text{model}}}}\\right)$$</li>
    </ul>

    <p>Even dimensions use sine, odd dimensions use cosine. Each dimension has a different frequency, from $\\sin(\\text{pos})$ at dimension 0 to $\\sin(\\text{pos}/10000)$ at the highest dimension.</p>

    <h4>Intuition Behind Sinusoidal Encodings</h4>
    <ul>
      <li><strong>Unique representation:</strong> Each position gets a unique d_model-dimensional vector</li>
      <li><strong>Wavelength spectrum:</strong> Different dimensions oscillate at different frequencies, creating a unique "fingerprint" for each position</li>
      <li><strong>Low dimensions (high frequency):</strong> Change rapidly with position, capturing fine-grained local structure</li>
      <li><strong>High dimensions (low frequency):</strong> Change slowly, capturing coarse-grained global structure</li>
      <li><strong>Analogy to binary encoding:</strong> Like bits in binary numbers, each dimension contributes positional information at a different scale</li>
    </ul>

    <h4>Mathematical Properties</h4>
    <ul>
      <li><strong>Fixed:</strong> Not learned, deterministic function</li>
      <li><strong>Unique:</strong> Each position gets unique encoding</li>
      <li><strong>Smooth:</strong> Similar positions have similar encodings</li>
      <li><strong>Relative positions:</strong> PE(pos+k) is linear function of PE(pos)</li>
      <li><strong>Bounded values:</strong> All values in [-1, 1], preventing explosion and matching typical embedding scales</li>
      <li><strong>Smooth changes:</strong> Adjacent positions have similar encodings, reflecting proximity</li>
      <li><strong>No learned parameters:</strong> Fixed function, doesn't require training data, generalizes to unseen sequence lengths</li>
      <li><strong>Extrapolation:</strong> Can handle longer sequences than training</li>
    </ul>

    <h4>Implementation Details</h4>
    <ul>
      <li><strong>Addition to embeddings:</strong> PE added directly to word embeddings: $x = \\text{WordEmbed}(\\text{token}) + \\text{PE}(\\text{pos})$</li>
      <li><strong>Same dimension:</strong> PE has same $d_{\\text{model}}$ dimensionality as embeddings for direct addition</li>
      <li><strong>Precomputed:</strong> Can precompute PE matrix for maximum expected sequence length</li>
      <li><strong>No trainable parameters:</strong> Fixed sinusoidal patterns, though some implementations do train them</li>
    </ul>

    <h3>Learned Positional Embeddings: Data-Driven Approach</h3>
    <p>Instead of fixed sinusoidal functions, simply learn a positional embedding table through gradient descent.</p>

    <h4>Approach</h4>
    <ul>
      <li><strong>Trainable:</strong> Optimized during training</li>
      <li><strong>Embedding table:</strong> Create learnable parameter matrix of shape (max_seq_len, d_model)</li>
      <li><strong>Lookup:</strong> For position pos, retrieve PE_learned[pos] and add to word embedding</li>
      <li><strong>Training:</strong> Positional embeddings updated through backpropagation like word embeddings</li>
      <li><strong>Task-specific:</strong> Adapts to task requirements</li>
      <li><strong>Fixed length:</strong> Limited to max training sequence length</li>
      <li><strong>Used in BERT:</strong> Learned absolute position embeddings</li>
    </ul>

    <h4>Advantages of Learned Embeddings</h4>
    <ul>
      <li><strong>Task-specific adaptation:</strong> Learn position patterns specific to the data distribution</li>
      <li><strong>Flexibility:</strong> No constraints on representation, model discovers optimal encodings</li>
      <li><strong>Empirical performance:</strong> Often slightly outperforms sinusoidal on fixed-length tasks</li>
      <li><strong>Used in BERT, GPT:</strong> Most modern models use learned positional embeddings</li>
    </ul>

    <h4>Limitations of Learned Embeddings</h4>
    <ul>
      <li><strong>Fixed maximum length:</strong> Cannot generalize beyond max_seq_len seen during training</li>
      <li><strong>Data dependency:</strong> Requires sufficient training data to learn good representations</li>
      <li><strong>Parameters:</strong> Adds $\\text{max\\_seq\\_len} \\times d_{\\text{model}}$ parameters (e.g., $512 \\times 768 = 393K$ for BERT)</li>
    </ul>

    <h3>Relative Positional Encoding: Focus on Distances</h3>
    <p>Rather than encoding absolute positions, relative positional encoding captures the distance between positions i and j when computing attention.</p>

    <h4>Motivation</h4>
    <ul>
      <li><strong>Translation invariant:</strong> Same pattern at any position. Relationship between "the cat" should be same whether at start or middle of sentence</li>
      <li><strong>Better generalization:</strong> To different sequence lengths. Absolute position 50 may not appear in training if max length is 30, but relative distances up to 30 still meaningful</li>
      <li><strong>Length generalization:</strong> Absolute position 50 may not appear in training if max length is 30, but relative distances up to 30 still meaningful</li>
      <li><strong>Inductive bias:</strong> Many linguistic phenomena depend on relative distance (e.g., dependency length), not absolute position</li>
      <li><strong>Used in:</strong> Transformer-XL, T5, DeBERTa</li>
      <li><strong>Implementation:</strong> Add to attention scores or keys/values</li>
    </ul>

    <h4>T5 Relative Position Biases</h4>
    <p>T5 model introduced simple yet effective relative positional encoding:</p>
    <ul>
      <li><strong>Attention modification:</strong> $\\text{Attention}(Q, K, V) = \\text{softmax}((QK^T + R) / \\sqrt{d_k})V$</li>
      <li><strong>Relative bias R:</strong> Learnable scalar bias for each relative position distance</li>
      <li><strong>Bucketing:</strong> Discretize distances into buckets (e.g., 0, 1, 2-3, 4-7, 8-15, ...) to limit parameters</li>
      <li><strong>Per-head biases:</strong> Each attention head learns its own position biases</li>
      <li><strong>Simplicity:</strong> Only scalar biases, not full embeddings, reducing parameters significantly</li>
    </ul>

    <h3>Modern Approaches: RoPE and ALiBi</h3>

    <h4>Rotary Position Embeddings (RoPE)</h4>
    <p>Modern approach used in models like GPT-NeoX, PaLM, LLaMA:</p>
    <ul>
      <li><strong>Rotation matrices:</strong> Rotate query and key vectors</li>
      <li><strong>Core idea:</strong> Apply rotation matrix to Q and K based on position before computing attention</li>
      <li><strong>Mathematics:</strong> Rotation by angle $\\theta = \\text{pos} \\cdot \\omega$ where $\\omega$ depends on dimension, creating position-dependent rotations</li>
      <li><strong>Relative encoding:</strong> Dot product captures relative positions. Dot product $Q_i \\cdot K_j$ naturally captures relative position $i-j$ through rotation geometry</li>
      <li><strong>Better extrapolation:</strong> Works well beyond training length. No addition to embeddings, works seamlessly with any model depth</li>
      <li><strong>Efficiency:</strong> No additional parameters</li>
      <li><strong>Practical success:</strong> Enables models like LLaMA to handle sequences much longer than training length</li>
    </ul>

    <h4>Attention with Linear Biases (ALiBi)</h4>
    <p>Simpler alternative that adds static, non-learned biases to attention scores:</p>
    <ul>
      <li><strong>Formula:</strong> Attention scores = $QK^T / \\sqrt{d_k} - m \\cdot \\text{distance}$ where $m$ is head-specific slope</li>
      <li><strong>Effect:</strong> Penalizes attention to distant tokens linearly based on distance</li>
      <li><strong>No parameters:</strong> Only hyperparameter is set of slopes (e.g., geometric sequence $2^{-8/h}$ for head $h$)</li>
      <li><strong>Training efficiency:</strong> Faster than learned positional embeddings</li>
      <li><strong>Extrapolation:</strong> Strong length generalization, models handle sequences $2{-}5\\times$ training length</li>
    </ul>

    <h3>Comparison and Trade-offs</h3>

    <h4>Design Considerations</h4>
    <ul>
      <li><strong>Absolute vs relative:</strong> Trade-offs in expressiveness and generalization</li>
      <li><strong>Learned vs fixed:</strong> Flexibility vs parameter efficiency</li>
      <li><strong>Extrapolation:</strong> Handling sequences longer than training</li>
      <li><strong>Efficiency:</strong> Computational and memory costs</li>
    </ul>

    <h4>Sinusoidal vs Learned</h4>
    <table border="1" cellpadding="8" style="border-collapse: collapse; width: 100%;">
      <tr>
        <th>Aspect</th>
        <th>Sinusoidal</th>
        <th>Learned</th>
      </tr>
      <tr>
        <td>Parameters</td>
        <td>0 (fixed function)</td>
        <td>$\\text{max\\_len} \\times d_{\\text{model}}$</td>
      </tr>
      <tr>
        <td>Generalization</td>
        <td>Excellent (unlimited length)</td>
        <td>Limited to training length</td>
      </tr>
      <tr>
        <td>Performance (fixed length)</td>
        <td>Good</td>
        <td>Slightly better (task-specific)</td>
      </tr>
      <tr>
        <td>Interpretability</td>
        <td>Clear (frequency spectrum)</td>
        <td>Opaque (learned representations)</td>
      </tr>
    </table>

    <h4>Absolute vs Relative</h4>
    <table border="1" cellpadding="8" style="border-collapse: collapse; width: 100%;">
      <tr>
        <th>Aspect</th>
        <th>Absolute</th>
        <th>Relative</th>
      </tr>
      <tr>
        <td>Inductive bias</td>
        <td>Position-specific patterns</td>
        <td>Distance-dependent patterns</td>
      </tr>
      <tr>
        <td>Translation invariance</td>
        <td>No</td>
        <td>Yes</td>
      </tr>
      <tr>
        <td>Length generalization</td>
        <td>Moderate</td>
        <td>Better (especially RoPE, ALiBi)</td>
      </tr>
      <tr>
        <td>Implementation complexity</td>
        <td>Simple (add to embeddings)</td>
        <td>Moderate (modify attention)</td>
      </tr>
    </table>

    <h3>Practical Considerations</h3>

    <h4>Choosing a Positional Encoding: Decision Tree</h4>
    <pre>
START: What's your use case?
│
├─ Training from scratch with small dataset?
│  └─→ Use: Sinusoidal (no parameters to overfit)
│
├─ Fine-tuning pre-trained model?
│  └─→ Keep: Whatever the base model uses (maintain compatibility)
│
├─ Need to handle sequences LONGER than training length?
│  │
│  ├─ By a lot ($2{-}5\\times$ longer)?
│  │  └─→ Use: ALiBi or RoPE (best extrapolation)
│  │
│  └─ By a little ($1.5{-}2\\times$ longer)?
│     └─→ Use: RoPE or interpolated learned embeddings
│
├─ Building modern LLM (100B+ params)?
│  └─→ Use: RoPE (current best practice, see LLaMA, PaLM)
│
├─ Memory-constrained?
│  └─→ Use: ALiBi (no embedding parameters, just slopes)
│
├─ Need best possible performance on fixed-length tasks?
│  └─→ Use: Learned absolute positions (task-specific optimization)
│
└─ Translation/seq2seq tasks?
   └─→ Use: Relative position biases (T5-style) or RoPE

Default recommendation: RoPE (best all-around choice for 2024+)
</pre>

    <h4>Choosing a Positional Encoding</h4>
    <ul>
      <li><strong>Fixed-length tasks (classification):</strong> Learned positional embeddings work well</li>
      <li><strong>Variable-length generation:</strong> RoPE or ALiBi for better extrapolation</li>
      <li><strong>Very long sequences (16K+ tokens):</strong> ALiBi or relative encodings to manage attention patterns</li>
      <li><strong>Limited data:</strong> Sinusoidal or RoPE avoid overfitting on positional patterns</li>
      <li><strong>Modern LLMs:</strong> Trend toward RoPE (LLaMA, PaLM) or ALiBi for flexibility</li>
    </ul>

    <h4>Empirical Comparison: Length Generalization</h4>
    <table border="1" cellpadding="8" style="border-collapse: collapse; width: 100%;">
      <tr>
        <th>Encoding Type</th>
        <th>Trained on 512</th>
        <th>Tested on 1024</th>
        <th>Tested on 2048</th>
        <th>Notes</th>
      </tr>
      <tr>
        <td>Learned Absolute</td>
        <td>100%</td>
        <td>60-70%</td>
        <td>Fails</td>
        <td>Cannot extrapolate beyond training</td>
      </tr>
      <tr>
        <td>Sinusoidal</td>
        <td>100%</td>
        <td>85-90%</td>
        <td>70-80%</td>
        <td>Graceful degradation</td>
      </tr>
      <tr>
        <td>T5 Relative</td>
        <td>100%</td>
        <td>90-95%</td>
        <td>80-85%</td>
        <td>Better than sinusoidal</td>
      </tr>
      <tr>
        <td>RoPE</td>
        <td>100%</td>
        <td>95-98%</td>
        <td>85-92%</td>
        <td>Excellent extrapolation</td>
      </tr>
      <tr>
        <td>ALiBi</td>
        <td>100%</td>
        <td>95-99%</td>
        <td>90-95%</td>
        <td>Best extrapolation, no params</td>
      </tr>
    </table>
    <p><em>Note: Percentages are approximate relative performance on language modeling perplexity. Actual numbers vary by task and model.</em></p>

    <h4>Implementation Tips</h4>
    <ul>
      <li><strong>Scaling:</strong> Ensure positional encoding magnitude matches embedding magnitude (typically achieved through layer normalization after addition)</li>
      <li><strong>Dropout:</strong> Apply dropout to positional encodings during training for regularization</li>
      <li><strong>Inspection:</strong> Visualize learned positional embeddings to verify they capture positional structure</li>
      <li><strong>Ablation:</strong> Test model without positional encoding to quantify impact on your specific task</li>
    </ul>

    <h3>The Essential Role of Position Information</h3>
    <p>Positional encoding is not a minor detail—it's fundamental to Transformer's success. By injecting positional information while preserving parallelization and attention's flexibility, positional encoding enables Transformers to handle sequential data effectively. The evolution from sinusoidal to learned to relative to RoPE/ALiBi reflects ongoing refinement of this critical component, with modern approaches enabling models to handle increasingly long sequences with better generalization. Every major Transformer model uses some form of positional encoding, making it an essential element of the architecture.</p>
  `,
  codeExamples: [
    {
      language: 'Python',
      code: `import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=5000):
      super().__init__()

      # Create positional encoding matrix
      pe = torch.zeros(max_len, d_model)
      position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

      # Compute div_term: 10000^(2i/d_model)
      div_term = torch.exp(
          torch.arange(0, d_model, 2).float() *
          (-math.log(10000.0) / d_model)
      )

      # Apply sin to even indices
      pe[:, 0::2] = torch.sin(position * div_term)
      # Apply cos to odd indices
      pe[:, 1::2] = torch.cos(position * div_term)

      # Add batch dimension
      pe = pe.unsqueeze(0)

      # Register as buffer (not a parameter)
      self.register_buffer('pe', pe)

  def forward(self, x):
      # x: [batch, seq_len, d_model]
      seq_len = x.size(1)
      # Add positional encoding
      x = x + self.pe[:, :seq_len, :]
      return x

# Usage
d_model = 512
pos_encoder = SinusoidalPositionalEncoding(d_model)
x = torch.randn(32, 100, d_model)
x_with_pos = pos_encoder(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {x_with_pos.shape}")

# Visualize positional encoding
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.imshow(pos_encoder.pe[0, :50, :].T, aspect='auto', cmap='RdBu')
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.colorbar()
plt.title('Sinusoidal Positional Encoding')`,
      explanation: 'Sinusoidal positional encoding using sin/cos functions at different frequencies for each dimension.'
    },
    {
      language: 'Python',
      code: `import torch
import torch.nn as nn

class LearnedPositionalEmbedding(nn.Module):
  def __init__(self, d_model, max_len=512):
      super().__init__()
      # Learned position embeddings
      self.position_embeddings = nn.Embedding(max_len, d_model)
      self.max_len = max_len

  def forward(self, x):
      # x: [batch, seq_len, d_model]
      batch_size, seq_len, d_model = x.size()

      # Create position indices
      positions = torch.arange(
          seq_len,
          dtype=torch.long,
          device=x.device
      ).unsqueeze(0).expand(batch_size, -1)

      # Get position embeddings and add to input
      pos_emb = self.position_embeddings(positions)
      return x + pos_emb

class RelativePositionalEncoding(nn.Module):
  def __init__(self, d_model, max_relative_position=128):
      super().__init__()
      self.max_relative_position = max_relative_position
      # Embeddings for relative positions
      vocab_size = 2 * max_relative_position + 1
      self.relative_positions_embeddings = nn.Embedding(
          vocab_size, d_model
      )

  def forward(self, length):
      # Generate relative position matrix
      range_vec = torch.arange(length)
      range_mat = range_vec.unsqueeze(0).expand(length, -1)
      distance_mat = range_mat - range_mat.transpose(0, 1)

      # Clip to maximum relative position
      distance_mat_clipped = torch.clamp(
          distance_mat,
          -self.max_relative_position,
          self.max_relative_position
      )

      # Shift to be non-negative
      final_mat = distance_mat_clipped + self.max_relative_position
      embeddings = self.relative_positions_embeddings(final_mat)
      return embeddings

# Usage comparison
d_model = 512
seq_len = 100

# Learned absolute positions
learned_pos = LearnedPositionalEmbedding(d_model, max_len=512)
x = torch.randn(32, seq_len, d_model)
output1 = learned_pos(x)
print(f"Learned positional encoding: {output1.shape}")

# Relative positions
relative_pos = RelativePositionalEncoding(d_model, max_relative_position=128)
rel_embeddings = relative_pos(seq_len)
print(f"Relative position embeddings: {rel_embeddings.shape}")`,
      explanation: 'Learned and relative positional encoding implementations showing different approaches to position information.'
    }
  ],
  interviewQuestions: [
    {
      question: 'Why do Transformers need positional encodings?',
      answer: `Transformers need positional encodings because the self-attention mechanism is inherently permutation-invariant, meaning it treats input sequences as unordered sets rather than ordered sequences. Without explicit position information, a Transformer would process "The cat sat on the mat" identically to "Mat the on sat cat the," losing crucial sequential structure that is fundamental to language understanding. Positional encodings solve this by adding position-specific patterns to input embeddings, enabling the model to distinguish between tokens based on both content and location in the sequence.`
    },
    {
      question: 'What are the advantages of sinusoidal positional encodings over learned ones?',
      answer: `Sinusoidal positional encodings offer several advantages: (1) No additional parameters to learn, reducing model size and training complexity, (2) Deterministic and mathematically elegant patterns that can extrapolate to sequence lengths longer than those seen during training, (3) Built-in relative position relationships through trigonometric properties, and (4) Theoretical foundation that enables analysis and understanding. While learned positional embeddings can sometimes perform slightly better by adapting to task-specific patterns, sinusoidal encodings provide robust performance across diverse tasks without requiring task-specific tuning.`
    },
    {
      question: 'Explain the difference between absolute and relative positional encodings.',
      answer: `Absolute positional encodings assign unique representations to each position in a sequence (position 1, position 2, etc.), while relative positional encodings capture relationships between positions (distance of 2, distance of 5, etc.). Absolute encodings are simpler to implement and understand but may struggle with position-sensitive tasks and length generalization. Relative encodings better capture the intuition that nearby words are more related than distant ones and can potentially generalize better to unseen sequence lengths, leading to improved performance on many NLP tasks that depend on word order relationships.`
    },
    {
      question: 'How do sinusoidal encodings allow the model to extrapolate to longer sequences?',
      answer: `Sinusoidal encodings use mathematical functions (sine and cosine) that are defined for all real numbers, not just the positions seen during training. The encoding PE(pos) = [sin(pos/10000^(2i/d)), cos(pos/10000^(2i/d))] creates smooth, continuous patterns that naturally extend beyond training sequence lengths. Each dimension oscillates at different frequencies, providing unique but continuous representations for any position. This mathematical foundation enables the model to generate meaningful position representations for sequences longer than those encountered during training, though performance may still degrade due to other factors.`
    },
    {
      question: 'What is the main advantage of Rotary Position Embeddings (RoPE)?',
      answer: `RoPE's main advantage is encoding position information directly into the attention computation through rotation matrices, rather than adding position information to input embeddings. This approach naturally incorporates relative position relationships into the attention mechanism itself, leading to better length extrapolation, improved performance on position-sensitive tasks, and more intuitive geometric interpretation of position relationships. RoPE enables models to handle much longer sequences than their training length while maintaining strong performance, making it particularly valuable for applications requiring long-range understanding.`
    }
  ],
  quizQuestions: [
    {
      id: 'pos1',
      question: 'Why do Transformers require positional encodings?',
      options: ['To reduce parameters', 'Self-attention has no inherent position information', 'To prevent overfitting', 'To speed up training'],
      correctAnswer: 1,
      explanation: 'Self-attention is permutation-equivariant and processes all positions in parallel, so it has no inherent notion of position or order. Positional encodings explicitly inject this information.'
    },
    {
      id: 'pos2',
      question: 'What is an advantage of sinusoidal positional encodings?',
      options: ['Uses fewer parameters', 'Can extrapolate to longer sequences', 'Faster to compute', 'More accurate'],
      correctAnswer: 1,
      explanation: 'Sinusoidal encodings are deterministic functions that can generate encodings for any position, allowing the model to handle sequences longer than those seen during training.'
    },
    {
      id: 'pos3',
      question: 'What is the key property of relative positional encodings?',
      options: ['Encode absolute positions', 'Encode distances between positions', 'Reduce memory', 'Increase speed'],
      correctAnswer: 1,
      explanation: 'Relative positional encodings capture the distance between positions rather than absolute positions, making them translation-invariant and often better at generalizing to different sequence lengths.'
    }
  ]
};
