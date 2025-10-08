import { Topic } from '../../../types';

export const transformerArchitecture: Topic = {
  id: 'transformer-architecture',
  title: 'Transformer Architecture',
  category: 'transformers',
  description: 'Revolutionary architecture based purely on attention mechanisms',
  content: `
    <h2>Transformer Architecture: Attention Is All You Need</h2>
    <p>The Transformer, introduced in the landmark 2017 paper "Attention Is All You Need" by Vaswani et al., represents one of the most significant breakthroughs in deep learning history. By eliminating recurrence entirely and relying solely on attention mechanisms, Transformers solved fundamental limitations of RNN-based models while enabling unprecedented parallelization and scaling. This architecture didn't just improve upon previous approaches—it redefined the landscape of natural language processing and sparked a revolution that extended far beyond text to vision, speech, biology, and countless other domains.</p>

    <h3>The Motivation: Beyond Sequential Processing</h3>
    <p>Despite the success of LSTM and GRU architectures, RNN-based models faced insurmountable constraints:</p>

    <ul>
      <li><strong>Sequential bottleneck:</strong> Each time step depends on the previous one, preventing parallel computation and limiting training speed</li>
      <li><strong>Limited context:</strong> Even with attention, RNNs struggle with very long sequences due to the sequential information flow</li>
      <li><strong>Path length:</strong> Information between distant positions must traverse many steps, with path length $O(n)$ making gradient flow difficult</li>
      <li><strong>Memory constraints:</strong> Maintaining hidden states for long sequences consumes significant memory</li>
      <li><strong>Hardware underutilization:</strong> Sequential computation cannot fully leverage modern GPU parallelism</li>
    </ul>

    <p><strong>The transformative question:</strong> What if we eliminated recurrence entirely and relied purely on attention to model dependencies? The Transformer was the elegant answer that unlocked a new era of AI.</p>

    <h3>Core Principles and Innovations</h3>

    <h4>1. No Recurrence: Full Parallelization</h4>
    <p>Transformers process all positions simultaneously rather than sequentially. Every token in a sequence is encoded or decoded in parallel during training, dramatically reducing wall-clock time. This enables processing of entire sequences in $O(1)$ sequential steps rather than $O(n)$.</p>

    <h4>2. Self-Attention: Direct Connections</h4>
    <p>Every position directly attends to every other position through self-attention, creating $O(1)$ path length between any two tokens regardless of distance. This enables modeling of arbitrary long-range dependencies without the vanishing gradient problems that plague RNNs.</p>

    <h4>3. Positional Encoding: Injecting Sequence Order</h4>
    <p>Since attention is inherently permutation-invariant, Transformers explicitly add positional information through encoding functions. This provides sequence order awareness without requiring sequential processing.</p>

    <h4>4. Multi-Head Attention: Parallel Perspectives</h4>
    <p>Computing attention multiple times in parallel with different learned projections allows the model to jointly attend to information from different representation subspaces, capturing diverse relationships simultaneously.</p>

    <h3>The Encoder Stack: Building Understanding</h3>
    <p>The encoder consists of N identical layers (typically N=6 in the original paper, but modern models use up to 24+ layers). Each encoder layer contains two primary sub-layers with carefully designed connections.</p>

    <h4>Multi-Head Self-Attention Sub-layer</h4>
    <p>Computes attention among all input positions, allowing each token to gather information from the entire sequence:</p>
    <ul>
      <li><strong>Input:</strong> Sequence of embeddings $X \\in \\mathbb{R}^{n \\times d}$</li>
      <li><strong>Operation:</strong> MultiHead(X, X, X) where the sequence attends to itself</li>
      <li><strong>Output:</strong> Contextualized representations incorporating information from all positions</li>
      <li><strong>Bidirectional:</strong> Each position can attend to all positions (past and future)</li>
    </ul>

    <h4>Position-wise Feedforward Network</h4>
    <p>Applies the same feedforward network independently to each position:</p>
    <ul>
      <li><strong>Architecture:</strong> Two linear transformations with ReLU activation: $\\text{FFN}(x) = \\max(0, xW_1 + b_1)W_2 + b_2$</li>
      <li><strong>Dimensions:</strong> Typically $d_{\\text{model}}=512$ expands to $d_{\\text{ff}}=2048$, then back to $512$</li>
      <li><strong>Purpose:</strong> Adds non-linear transformations and allows positions to process information independently after gathering context</li>
      <li><strong>Parameters:</strong> Not shared across layers but shared across positions within a layer</li>
    </ul>

    <h4>Residual Connections and Layer Normalization</h4>
    <p>Each sub-layer is wrapped with residual connections and layer normalization:</p>
    <ul>
      <li><strong>Pattern:</strong> $\\text{LayerNorm}(x + \\text{Sublayer}(x))$</li>
      <li><strong>Residual benefits:</strong> Enables gradient flow through deep networks, provides identity mapping path, allows learning of incremental refinements</li>
      <li><strong>Layer normalization:</strong> Normalizes activations across features (not batch), stabilizes training, accelerates convergence</li>
    </ul>

    <h3>The Decoder Stack: Generating Output</h3>
    <p>The decoder also consists of N identical layers, but with an additional sub-layer and modified attention mechanism for autoregressive generation.</p>

    <h4>Masked Multi-Head Self-Attention</h4>
    <p>Attends to all previous positions in the output sequence but prevents attending to future positions:</p>
    <ul>
      <li><strong>Masking mechanism:</strong> Set attention scores for future positions to $-\\infty$ before softmax</li>
      <li><strong>Causal property:</strong> Ensures position i can only depend on positions 1 to i-1</li>
      <li><strong>Training-inference consistency:</strong> Same information constraints during both training and generation</li>
      <li><strong>Autoregressive generation:</strong> Enables left-to-right sequence generation</li>
    </ul>

    <h4>Cross-Attention (Encoder-Decoder Attention)</h4>
    <p>Allows decoder positions to attend to all encoder positions:</p>
    <ul>
      <li><strong>Queries:</strong> From decoder (what we're generating)</li>
      <li><strong>Keys and Values:</strong> From encoder output (what we're conditioning on)</li>
      <li><strong>Purpose:</strong> Connects output generation to input understanding</li>
      <li><strong>Learned alignment:</strong> Automatically learns which input positions are relevant for each output position</li>
    </ul>

    <h4>Position-wise Feedforward Network</h4>
    <p>Identical architecture to encoder feedforward layer, processes each position independently after gathering both self and cross-attention context.</p>

    <h3>Complete Architecture: Information Flow</h3>

    <h4>Visual Data Flow Overview</h4>
    <p><strong>Encoding Phase:</strong></p>
    <pre>
Input Tokens: ["The", "cat", "sat"]
  ↓
Token Embeddings: [768-dim vectors] 
  +
Positional Encodings: [768-dim vectors]
  ↓
Encoder Layer 1: Self-Attention $\\to$ Add & Norm $\\to$ FFN $\\to$ Add & Norm
  ↓
Encoder Layer 2-N: (same structure, different weights)
  ↓
Final Encoder Output: Contextualized representations [seq_len $\\times$ 768]
</pre>

    <p><strong>Decoding Phase (for generation):</strong></p>
    <pre>
Output so far: ["Le", "chat"]
  ↓
Output Embeddings + Positional Encodings
  ↓
Decoder Layer 1: 
  - Masked Self-Attention (can't see future)
  - Cross-Attention (attend to encoder output)
  - FFN
  ↓
Decoder Layer 2-N: (same structure)
  ↓
Linear + Softmax $\\to$ Probabilities over vocabulary
  ↓
Sample next token: "s'est"
  ↓
Repeat autoregressively until [EOS]
</pre>

    <h4>Encoding Phase</h4>
    <ul>
      <li><strong>Step 1:</strong> Input embeddings + positional encoding $\\to$ Initial representations</li>
      <li><strong>Step 2:</strong> Pass through N encoder layers, each refining representations through self-attention and feedforward</li>
      <li><strong>Step 3:</strong> Final encoder output = rich bidirectional contextualized representations</li>
      <li><strong>Parallel processing:</strong> All positions processed simultaneously</li>
    </ul>

    <h4>Decoding Phase</h4>
    <ul>
      <li><strong>Step 1:</strong> Output embeddings (shifted right) + positional encoding $\\to$ Initial decoder representations</li>
      <li><strong>Step 2:</strong> Pass through N decoder layers: masked self-attention $\\to$ cross-attention $\\to$ feedforward</li>
      <li><strong>Step 3:</strong> Final linear + softmax $\\to$ probability distribution over vocabulary</li>
      <li><strong>Autoregressive generation:</strong> Generate one token at a time, feeding previous outputs as inputs</li>
    </ul>

    <h3>Common Misconceptions</h3>
    <ul>
      <li><strong>"More attention layers = always better":</strong> Not necessarily. Very deep models face training instability and diminishing returns. Optimal depth depends on data size and task complexity. Most successful models use 6-24 encoder layers.</li>
      <li><strong>"Transformers replace all other architectures":</strong> While powerful, CNNs remain superior for some vision tasks (especially with limited data), and RNNs can be more efficient for very long sequences with strong temporal dependencies.</li>
      <li><strong>"Attention learns everything from scratch":</strong> Positional encoding provides crucial inductive bias. Pure attention without positional info would be permutation-invariant and struggle with sequential tasks.</li>
      <li><strong>"Bigger context window = always better":</strong> Quadratic memory means very long contexts (>4K tokens) become impractical without sparse attention. Quality of context matters more than quantity.</li>
      <li><strong>"Transformers are always faster than RNNs":</strong> True for training (parallelization), but inference speed depends on sequence length and hardware. For very short sequences on CPUs, RNNs can be faster.</li>
    </ul>

    <h3>Key Advantages Over RNN Architectures</h3>

    <h4>Parallelization and Speed</h4>
    <ul>
      <li><strong>Training:</strong> $10\\text{-}100\\times$ faster than RNNs on modern GPUs due to full parallelization</li>
      <li><strong>Utilization:</strong> Better hardware utilization through matrix operations</li>
      <li><strong>Scalability:</strong> Enables training on much larger datasets and model sizes</li>
    </ul>

    <h4>Long-Range Dependencies</h4>
    <ul>
      <li><strong>Path length:</strong> $O(1)$ between any positions vs $O(n)$ in RNNs</li>
      <li><strong>No vanishing gradients:</strong> Direct gradient paths between all positions</li>
      <li><strong>Effective context:</strong> Can model dependencies across entire sequence (hundreds or thousands of tokens)</li>
    </ul>

    <h4>Representational Power</h4>
    <ul>
      <li><strong>Flexible attention:</strong> Can learn any dependency pattern through attention weights</li>
      <li><strong>Multi-head diversity:</strong> Different heads capture different relationship types</li>
      <li><strong>Depth benefits:</strong> Stacking many layers builds increasingly abstract representations</li>
    </ul>

    <h4>Interpretability</h4>
    <ul>
      <li><strong>Attention visualization:</strong> Can inspect which positions attend to which</li>
      <li><strong>Layer analysis:</strong> Lower layers capture syntax, higher layers capture semantics</li>
      <li><strong>Head specialization:</strong> Different attention heads learn different linguistic phenomena</li>
    </ul>

    <h3>Computational Complexity Analysis</h3>

    <h4>Self-Attention: $O(n^2 \\cdot d)$</h4>
    <ul>
      <li><strong>Quadratic in sequence length:</strong> All pairs of positions interact</li>
      <li><strong>Linear in model dimension:</strong> Scales with embedding size</li>
      <li><strong>Bottleneck:</strong> Long sequences (n > 1000) become expensive</li>
    </ul>

    <h4>RNN: $O(n \\cdot d^2)$</h4>
    <ul>
      <li><strong>Linear in sequence length:</strong> Sequential processing</li>
      <li><strong>Quadratic in model dimension:</strong> Matrix multiplications at each step</li>
      <li><strong>Sequential bottleneck:</strong> Cannot parallelize across time</li>
    </ul>

    <h4>Trade-offs</h4>
    <p>For typical settings ($n \\approx 100$, $d \\approx 512$), Transformers are much faster despite quadratic complexity because parallelization outweighs the $O(n^2)$ factor. For very long sequences ($n > 2000$), specialized variants like sparse attention or memory-efficient attention become necessary.</p>

    <h3>Training Techniques and Optimizations</h3>

    <ul>
      <li><strong>Warmup learning rate:</strong> Linear warmup followed by inverse square root decay, crucial for stable training</li>
      <li><strong>Label smoothing:</strong> Softens target distributions (0.1 typical), improves generalization</li>
      <li><strong>Dropout:</strong> Applied to attention weights and after each sub-layer (0.1 typical)</li>
      <li><strong>Weight initialization:</strong> Xavier/Glorot initialization scaled appropriately for layer depth</li>
      <li><strong>Gradient clipping:</strong> Clip gradients to prevent instability in early training</li>
      <li><strong>Mixed precision training:</strong> Use FP16 for speed, FP32 for stability</li>
    </ul>

    <h3>Variants and Improvements</h3>

    <ul>
      <li><strong>Transformer-XL:</strong> Segment-level recurrence for longer context</li>
      <li><strong>Sparse Transformers:</strong> Reduced complexity through sparse attention patterns</li>
      <li><strong>Linformer:</strong> Linear complexity approximation of attention</li>
      <li><strong>Reformer:</strong> Locality-sensitive hashing for efficient attention</li>
      <li><strong>Performer:</strong> Kernel-based approximation achieving linear complexity</li>
    </ul>

    <h3>Applications Beyond Translation</h3>

    <ul>
      <li><strong>Language modeling:</strong> GPT series demonstrates power of decoder-only Transformers</li>
      <li><strong>Understanding:</strong> BERT uses encoder for bidirectional representations</li>
      <li><strong>Generation:</strong> T5, BART use full encoder-decoder for text-to-text tasks</li>
      <li><strong>Vision:</strong> Vision Transformers (ViT) treat images as sequences of patches</li>
      <li><strong>Speech:</strong> Whisper, Wav2Vec2 apply Transformers to audio</li>
      <li><strong>Multimodal:</strong> CLIP, Flamingo combine text and vision</li>
      <li><strong>Biology:</strong> AlphaFold uses Transformers for protein structure prediction</li>
      <li><strong>Code:</strong> Codex, AlphaCode generate and understand programming languages</li>
    </ul>

    <h3>The Foundation of Modern AI</h3>
    <p>The Transformer architecture didn't just solve machine translation—it provided a universal framework for sequence modeling that scales magnificently. From GPT-3's 175 billion parameters to modern models exceeding a trillion parameters, Transformers enabled the large language model revolution. Their combination of parallel processing, flexible attention, and scalable architecture made them the foundation for the current era of AI, demonstrating that attention truly is all you need.</p>
  `,
  codeExamples: [
    {
      language: 'Python',
      code: `import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
  def __init__(self, d_model=512, nhead=8, dim_ff=2048, dropout=0.1):
      super().__init__()
      self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
      self.norm1 = nn.LayerNorm(d_model)
      self.norm2 = nn.LayerNorm(d_model)
      self.ff = nn.Sequential(
          nn.Linear(d_model, dim_ff),
          nn.ReLU(),
          nn.Dropout(dropout),
          nn.Linear(dim_ff, d_model),
          nn.Dropout(dropout)
      )

  def forward(self, x, mask=None):
      # Self-attention with residual
      attn_out, _ = self.attention(x, x, x, attn_mask=mask)
      x = self.norm1(x + attn_out)

      # Feedforward with residual
      ff_out = self.ff(x)
      x = self.norm2(x + ff_out)
      return x

# Usage
model = TransformerBlock()
x = torch.randn(32, 10, 512)  # [batch, seq_len, d_model]
output = model(x)
print(f"Output shape: {output.shape}")`,
      explanation: 'Basic Transformer encoder block with self-attention and feedforward layers.'
    }
  ],
  interviewQuestions: [
    {
      question: 'What are the main advantages of Transformers over RNNs?',
      answer: `Transformers offer several revolutionary advantages over RNNs that have fundamentally changed how we approach sequence modeling, making them the dominant architecture for most NLP tasks and enabling the development of large-scale language models.

Parallelization is perhaps the most significant advantage. RNNs must process sequences sequentially, where each hidden state depends on the previous one, preventing parallel computation during training. Transformers eliminate this constraint by using self-attention to directly connect all positions, allowing all tokens to be processed simultaneously. This parallel processing capability dramatically reduces training time and enables efficient utilization of modern GPU architectures.

Long-range dependency modeling is substantially improved in Transformers. RNNs suffer from vanishing gradients that make it difficult to learn dependencies spanning many time steps, typically limiting effective context to 5-10 positions. Transformers provide direct connections between any two positions through self-attention, enabling modeling of dependencies across entire sequences with path lengths of $O(1)$ rather than $O(n)$.

Computational efficiency manifests in multiple ways: (1) Parallel training reduces wall-clock time significantly, (2) More efficient use of hardware accelerators through matrix operations, (3) Better scaling properties as sequence length increases, and (4) Reduced memory requirements for very long sequences due to elimination of sequential hidden state chains.

Information flow improvements include: (1) No information bottleneck - all positions have access to all other positions, (2) No forgetting of early sequence information through iterative state updates, (3) Direct gradient paths between any positions improving training efficiency, and (4) Explicit modeling of position relationships rather than implicit temporal encoding.

Interpretability benefits come from attention weights that provide direct visualization of which positions influence each output. Unlike RNN hidden states that encode complex sequential information implicitly, attention patterns can be analyzed to understand model behavior and decision-making processes.

However, Transformers also have trade-offs: (1) Quadratic complexity in sequence length due to all-pairs attention, (2) Need for explicit positional encoding since attention is permutation-invariant, (3) Higher memory requirements for attention matrices, and (4) Potential for less inductive bias about sequential structure. Despite these limitations, the advantages have proven so significant that Transformers have largely replaced RNNs in most applications.`
    },
    {
      question: 'Explain the role of positional encoding in Transformers.',
      answer: `Positional encoding is a crucial component of Transformer architectures that solves the fundamental problem of providing sequence order information to an inherently permutation-invariant attention mechanism. Without positional encoding, Transformers would treat input sequences as unordered sets, losing essential structural information.

The core issue arises because self-attention computes relationships between tokens based purely on content similarity, with no inherent understanding of token positions. A sentence like "The cat sat on the mat" would be processed identically to "Mat the on sat cat the" without positional information, clearly problematic for language understanding where word order carries crucial meaning.

Transformers address this by adding positional encodings directly to input embeddings before the first attention layer. The original Transformer paper used sinusoidal positional encoding with sine and cosine functions of different frequencies: $\\text{PE}(\\text{pos}, 2i) = \\sin(\\text{pos}/10000^{2i/d_{\\text{model}}})$ and $\\text{PE}(\\text{pos}, 2i+1) = \\cos(\\text{pos}/10000^{2i/d_{\\text{model}}})$, where $\\text{pos}$ is the position, $i$ is the dimension index, and $d_{\\text{model}}$ is the model dimension.

This sinusoidal scheme has several elegant properties: (1) Each position receives a unique encoding pattern, (2) The model can potentially learn to attend to relative positions through trigonometric identities, (3) The encoding can extrapolate to sequence lengths longer than those seen during training, (4) Different dimensions capture position information at different scales, and (5) The encoding is deterministic and doesn't require learning additional parameters.

Alternative approaches include learned positional embeddings where position encodings are treated as trainable parameters, similar to word embeddings. While this requires learning and cannot extrapolate beyond training lengths, it often performs slightly better in practice by adapting to task-specific position patterns.

Recent advances have introduced more sophisticated schemes: (1) Relative positional encoding that explicitly models position differences rather than absolute positions, (2) Rotary Position Embedding (RoPE) that encodes position through rotation matrices applied to attention computations, (3) ALiBi (Attention with Linear Biases) that adds position-dependent biases to attention scores, and (4) T5's relative position biases learned during training.

The integration of positional encoding with attention enables the model to understand both content-based relationships (through attention weights) and position-based relationships (through positional encoding), creating a rich representation that captures both semantic and structural aspects of sequences. This combination has proven essential for the success of Transformer architectures across diverse sequence modeling tasks.`
    },
    {
      question: 'What is the difference between encoder and decoder in Transformers?',
      answer: `The encoder and decoder in Transformer architectures serve distinct but complementary roles, designed for different aspects of sequence processing and generation tasks. Understanding their differences is crucial for applying Transformers effectively to various NLP problems.

The encoder stack consists of multiple identical layers that process the input sequence to create rich contextual representations. Each encoder layer contains: (1) Multi-head self-attention that allows each position to attend to all positions in the input sequence bidirectionally, (2) A position-wise feedforward network that applies non-linear transformations, (3) Residual connections around each sub-layer, and (4) Layer normalization for training stability.

The decoder stack processes the target sequence autoregressively during training and generation. Each decoder layer contains: (1) Masked multi-head self-attention that prevents positions from attending to future positions, maintaining the causal property necessary for generation, (2) Multi-head cross-attention that allows decoder positions to attend to all encoder positions, (3) A position-wise feedforward network identical to the encoder, and (4) Residual connections and layer normalization following the same pattern.

Key architectural differences include attention mechanisms: Encoder self-attention is bidirectional, allowing each position to see the entire input context, while decoder self-attention uses causal masking to maintain the autoregressive property. The decoder additionally includes cross-attention layers that connect to encoder representations, enabling the decoder to access input information when generating output.

Processing patterns differ significantly: Encoders process complete input sequences in parallel during both training and inference, building comprehensive bidirectional representations. Decoders process target sequences autoregressively - during training, they see the entire target sequence but with causal masking, while during inference, they generate tokens one at a time based on previously generated tokens.

Information flow varies between components: Encoders focus on understanding and encoding input sequences, creating representations that capture relationships and dependencies within the input. Decoders focus on generation, using both their own context (through masked self-attention) and input context (through cross-attention) to produce appropriate output sequences.

Task suitability reflects these design differences: Encoder-only models (like BERT) excel at understanding tasks such as classification, named entity recognition, and question answering where the goal is to analyze and understand input text. Decoder-only models (like GPT) excel at generation tasks and can handle both understanding and generation through careful prompting. Full encoder-decoder models (like T5) are optimal for sequence-to-sequence tasks like translation and summarization where input and output are distinct sequences.

The combination enables powerful sequence-to-sequence modeling where the encoder builds comprehensive input understanding while the decoder generates contextually appropriate output, leveraging both input information and generation constraints to produce high-quality results.`
    },
    {
      question: 'Why is masked self-attention needed in the decoder?',
      answer: `Masked self-attention in the decoder is essential for maintaining the autoregressive property that enables proper sequence generation, preventing the model from "cheating" by accessing future tokens during training and ensuring consistency between training and inference procedures.

The fundamental requirement for autoregressive generation is that when predicting token at position t, the model should only have access to tokens at positions 1 through t-1, never to future tokens at positions t+1 and beyond. This constraint reflects the reality of generation: when producing text, we don't know what we'll write next, only what we've written so far.

Without masking, standard self-attention would allow each position to attend to all positions in the sequence, including future ones. During training, this would mean the model could "see ahead" to the correct answer when making predictions, leading to a form of data leakage where the model learns to rely on information that won't be available during actual generation.

The masking mechanism implements this constraint by modifying the attention computation to set attention scores to negative infinity (or a very large negative number) for positions that should not be accessible. When softmax is applied, these masked positions receive attention weights of effectively zero, preventing information flow from future positions.

Mathematically, the mask is applied before the softmax operation: $\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T + M}{\\sqrt{d_k}}\\right)V$, where $M$ is the mask matrix with 0s for allowed connections and $-\\infty$ for forbidden connections. This ensures that position $i$ can only attend to positions $j$ where $j \\leq i$.

Training-inference consistency is crucial for model performance. During training with teacher forcing, the model has access to the entire target sequence but must learn to make predictions as if generating autoregressively. Masked self-attention ensures the model experiences the same information constraints during training as it will during inference, preventing a distribution mismatch.

The causal structure enables several important properties: (1) Left-to-right generation order that matches natural language production, (2) Ability to generate sequences of arbitrary length by continuing the autoregressive process, (3) Consistent probability distributions over next tokens that don't depend on future context, and (4) Proper modeling of sequential dependencies where each token depends on its prefix.

Alternative approaches exist but have limitations: Bidirectional attention in the decoder would break the autoregressive property, making generation impossible. Non-autoregressive generation can work for some tasks but typically requires additional constraints or iterative refinement to achieve comparable quality.

Masked self-attention thus serves as the crucial mechanism that enables Transformer decoders to maintain the essential properties of autoregressive generation while benefiting from the parallel processing and representational power of the Transformer architecture.`
    }
  ],
  quizQuestions: [
    {
      id: 'trans1',
      question: 'What is the main innovation of the Transformer architecture?',
      options: ['Uses CNNs', 'Replaces recurrence with attention', 'Larger models', 'Better initialization'],
      correctAnswer: 1,
      explanation: 'Transformers replace sequential recurrence with self-attention mechanisms, enabling parallel processing and better long-range dependency modeling.'
    }
  ]
};
