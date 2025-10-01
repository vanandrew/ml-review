import { Topic } from '../../types';

export const transformersTopics: Record<string, Topic> = {
  'transformer-architecture': {
    id: 'transformer-architecture',
    title: 'Transformer Architecture',
    category: 'transformers',
    description: 'Revolutionary architecture based purely on attention mechanisms',
    content: `
      <h2>Transformer Architecture</h2>
      <p>The Transformer, introduced in "Attention is All You Need" (2017), revolutionized NLP by replacing recurrent layers with pure attention mechanisms. It enables parallel processing and captures long-range dependencies more effectively than RNNs.</p>

      <h3>Key Innovation</h3>
      <p>Unlike RNNs that process sequences sequentially, Transformers:</p>
      <ul>
        <li><strong>No recurrence:</strong> All positions processed in parallel</li>
        <li><strong>Self-attention:</strong> Each position attends to all positions</li>
        <li><strong>Positional encoding:</strong> Inject position information explicitly</li>
        <li><strong>Scalability:</strong> Can leverage GPU parallelism fully</li>
      </ul>

      <h3>Architecture Components</h3>

      <h4>Encoder Stack</h4>
      <p>N identical layers (typically N=6), each with:</p>
      <ul>
        <li><strong>Multi-Head Self-Attention:</strong> Attend to all input positions</li>
        <li><strong>Feedforward Network:</strong> Two linear layers with ReLU</li>
        <li><strong>Layer Normalization:</strong> After each sub-layer</li>
        <li><strong>Residual Connections:</strong> Around each sub-layer</li>
      </ul>

      <h4>Decoder Stack</h4>
      <p>N identical layers, each with:</p>
      <ul>
        <li><strong>Masked Self-Attention:</strong> Prevents attending to future positions</li>
        <li><strong>Cross-Attention:</strong> Attends to encoder output</li>
        <li><strong>Feedforward Network:</strong> Same as encoder</li>
        <li><strong>Layer Normalization + Residuals:</strong> Same pattern</li>
      </ul>

      <h3>Advantages Over RNNs</h3>
      <ul>
        <li><strong>Parallelization:</strong> All positions processed simultaneously</li>
        <li><strong>Long-range dependencies:</strong> Direct connections between any positions</li>
        <li><strong>Training speed:</strong> Much faster than sequential RNNs</li>
        <li><strong>Path length:</strong> Constant O(1) vs O(n) in RNNs</li>
        <li><strong>Interpretability:</strong> Attention weights show relationships</li>
      </ul>

      <h3>Applications</h3>
      <ul>
        <li><strong>Translation:</strong> Original application, state-of-the-art results</li>
        <li><strong>Language models:</strong> BERT, GPT, T5</li>
        <li><strong>Vision:</strong> Vision Transformer (ViT) for images</li>
        <li><strong>Speech:</strong> Wav2Vec, Whisper</li>
        <li><strong>Multimodal:</strong> CLIP, Flamingo</li>
      </ul>
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

Long-range dependency modeling is substantially improved in Transformers. RNNs suffer from vanishing gradients that make it difficult to learn dependencies spanning many time steps, typically limiting effective context to 5-10 positions. Transformers provide direct connections between any two positions through self-attention, enabling modeling of dependencies across entire sequences with path lengths of O(1) rather than O(n).

Computational efficiency manifests in multiple ways: (1) Parallel training reduces wall-clock time significantly, (2) More efficient use of hardware accelerators through matrix operations, (3) Better scaling properties as sequence length increases, and (4) Reduced memory requirements for very long sequences due to elimination of sequential hidden state chains.

Information flow improvements include: (1) No information bottleneck - all positions have access to all other positions, (2) No forgetting of early sequence information through iterative state updates, (3) Direct gradient paths between any positions improving training efficiency, and (4) Explicit modeling of position relationships rather than implicit temporal encoding.

Interpretability benefits come from attention weights that provide direct visualization of which positions influence each output. Unlike RNN hidden states that encode complex sequential information implicitly, attention patterns can be analyzed to understand model behavior and decision-making processes.

However, Transformers also have trade-offs: (1) Quadratic complexity in sequence length due to all-pairs attention, (2) Need for explicit positional encoding since attention is permutation-invariant, (3) Higher memory requirements for attention matrices, and (4) Potential for less inductive bias about sequential structure. Despite these limitations, the advantages have proven so significant that Transformers have largely replaced RNNs in most applications.`
      },
      {
        question: 'Explain the role of positional encoding in Transformers.',
        answer: `Positional encoding is a crucial component of Transformer architectures that solves the fundamental problem of providing sequence order information to an inherently permutation-invariant attention mechanism. Without positional encoding, Transformers would treat input sequences as unordered sets, losing essential structural information.

The core issue arises because self-attention computes relationships between tokens based purely on content similarity, with no inherent understanding of token positions. A sentence like "The cat sat on the mat" would be processed identically to "Mat the on sat cat the" without positional information, clearly problematic for language understanding where word order carries crucial meaning.

Transformers address this by adding positional encodings directly to input embeddings before the first attention layer. The original Transformer paper used sinusoidal positional encoding with sine and cosine functions of different frequencies: PE(pos, 2i) = sin(pos/10000^(2i/d_model)) and PE(pos, 2i+1) = cos(pos/10000^(2i/d_model)), where pos is the position, i is the dimension index, and d_model is the model dimension.

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

Mathematically, the mask is applied before the softmax operation: Attention(Q, K, V) = softmax((QK^T + M) / √d_k)V, where M is the mask matrix with 0s for allowed connections and -∞ for forbidden connections. This ensures that position i can only attend to positions j where j ≤ i.

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
  },

  'self-attention-multi-head': {
    id: 'self-attention-multi-head',
    title: 'Self-Attention and Multi-Head Attention',
    category: 'transformers',
    description: 'Core mechanism allowing models to weigh the importance of different positions',
    content: `
      <h2>Self-Attention and Multi-Head Attention</h2>
      <p>Self-attention is the core mechanism in Transformers that allows each position to attend to all positions in the input sequence. Multi-head attention extends this by running multiple attention operations in parallel, capturing different aspects of relationships.</p>

      <h3>Self-Attention Mechanism</h3>
      <p>Self-attention computes three vectors for each input:</p>
      <ul>
        <li><strong>Query (Q):</strong> What information is this position looking for?</li>
        <li><strong>Key (K):</strong> What information does this position offer?</li>
        <li><strong>Value (V):</strong> The actual information at this position</li>
      </ul>

      <h4>Attention Formula</h4>
      <p>Attention(Q, K, V) = softmax(QK^T / √d_k)V</p>
      <ul>
        <li><strong>QK^T:</strong> Compute similarity between queries and keys</li>
        <li><strong>/ √d_k:</strong> Scale by sqrt of key dimension (prevents large dot products)</li>
        <li><strong>softmax:</strong> Normalize to get attention weights (sum to 1)</li>
        <li><strong>× V:</strong> Weighted sum of values</li>
      </ul>

      <h3>Multi-Head Attention</h3>
      <p>Instead of one attention operation, use multiple "heads" in parallel:</p>
      <ul>
        <li><strong>Multiple perspectives:</strong> Each head learns different relationships</li>
        <li><strong>Parallel computation:</strong> All heads computed simultaneously</li>
        <li><strong>Concatenation:</strong> Head outputs concatenated and projected</li>
        <li><strong>Richer representations:</strong> Captures diverse semantic relationships</li>
      </ul>

      <h4>Multi-Head Formula</h4>
      <p>MultiHead(Q, K, V) = Concat(head₁, ..., head_h)W^O</p>
      <p>where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)</p>

      <h3>Why Multiple Heads?</h3>
      <ul>
        <li><strong>Different subspaces:</strong> Each head can focus on different semantic aspects</li>
        <li><strong>Syntactic vs semantic:</strong> Some heads capture syntax, others semantics</li>
        <li><strong>Local vs global:</strong> Some attend to nearby tokens, others to distant</li>
        <li><strong>Redundancy:</strong> Improves robustness through ensemble effect</li>
      </ul>

      <h3>Key Properties</h3>
      <ul>
        <li><strong>Permutation equivariant:</strong> No inherent position information</li>
        <li><strong>O(n²) complexity:</strong> Every position attends to every other</li>
        <li><strong>Parallelizable:</strong> All positions computed simultaneously</li>
        <li><strong>Interpretable:</strong> Attention weights show relationships</li>
      </ul>

      <h3>Attention Variants</h3>
      <ul>
        <li><strong>Masked attention:</strong> Prevent attending to future positions (decoder)</li>
        <li><strong>Cross-attention:</strong> Query from one sequence, keys/values from another</li>
        <li><strong>Local attention:</strong> Restrict attention to nearby positions</li>
        <li><strong>Sparse attention:</strong> Only attend to subset of positions</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: [batch, seq_len, d_model]
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_model)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

# Usage
model = SelfAttention(d_model=512)
x = torch.randn(32, 10, 512)  # [batch, seq_len, d_model]
output, weights = model(x)
print(f"Output: {output.shape}, Weights: {weights.shape}")`,
        explanation: 'Self-attention implementation showing query, key, value projections and scaled dot-product attention.'
      },
      {
        language: 'Python',
        code: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Single matrices for all heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x, batch_size):
        # Reshape to [batch, num_heads, seq_len, d_k]
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections and split into heads
        Q = self.split_heads(self.W_q(query), batch_size)
        K = self.split_heads(self.W_k(key), batch_size)
        V = self.split_heads(self.W_v(value), batch_size)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.W_o(context)
        return output, attn_weights

# Usage
model = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(32, 10, 512)
output, weights = model(x, x, x)
print(f"Output: {output.shape}, Attention: {weights.shape}")`,
        explanation: 'Multi-head attention with parallel attention heads, each capturing different relationships.'
      }
    ],
    interviewQuestions: [
      {
        question: 'Explain the purpose of Query, Key, and Value vectors in self-attention.',
        answer: `Query, Key, and Value vectors are the fundamental components of the attention mechanism that enable Transformers to compute relationships between different positions in a sequence. This decomposition provides a flexible and powerful framework for modeling dependencies and similarities between tokens.

The Query vector represents "what information am I looking for?" - it encodes the representation of the current position that will be used to compute attention weights with other positions. When processing a particular token, its query vector captures what type of information would be most relevant for understanding or generating that token in context.

The Key vector represents "what information do I contain?" - it encodes the representation of each position that will be compared against queries to determine relevance. Key vectors act as indexed content that can be searched against, enabling the attention mechanism to find relevant information across the sequence.

The Value vector represents "what information do I provide?" - it contains the actual content that will be aggregated based on attention weights. While queries and keys determine which positions are important, values contain the information that will actually be passed forward in the network.

The attention computation follows this framework: (1) Compute similarity scores between queries and keys to determine relevance, (2) Apply softmax to create attention weights that sum to 1, (3) Use these weights to compute a weighted average of value vectors. This separation allows the model to decouple the "searching" mechanism (query-key similarity) from the "content" mechanism (value aggregation).

This design provides several advantages: (1) Flexibility - the same input can play different roles as query, key, or value through different learned projections, (2) Expressiveness - separate transformations allow learning specialized representations for each role, (3) Interpretability - attention weights show which positions influence each output, and (4) Efficiency - matrix operations enable parallel computation across all positions.

In self-attention, all three vectors are derived from the same input sequence through learned linear projections: Q = XW_Q, K = XW_K, V = XW_V, where X is the input and W matrices are learned parameters. This enables the model to learn task-specific transformations for each role while maintaining the flexible query-key-value framework.`
      },
      {
        question: 'Why do we scale the dot product by √d_k in attention?',
        answer: `Scaling the dot product by √d_k in attention is crucial for maintaining stable training dynamics and preventing the softmax function from producing overly peaked probability distributions that would harm gradient flow and model performance.

The fundamental issue arises from the variance of dot products increasing with dimensionality. When computing attention scores as q·k where both q and k are d_k-dimensional vectors with elements drawn from distributions with variance σ², the variance of their dot product grows proportionally to d_k·σ². As d_k increases (typical values are 64-128 per attention head), dot products can become very large in magnitude.

Large dot product values create problems in the softmax computation. When input values to softmax are large, the function produces extremely peaked distributions where one element approaches 1 and others approach 0. This "sharpening" effect reduces the effective gradient signal during backpropagation because the gradients of softmax become very small when the distribution is highly concentrated.

Mathematically, if attention scores before softmax are [10, 9, 8], the softmax outputs approximately [0.67, 0.24, 0.09]. But if scores are [100, 90, 80], softmax produces approximately [1.0, 0.0, 0.0], eliminating the attention to other positions entirely and reducing gradient flow.

The scaling factor √d_k is chosen to normalize the variance of dot products back to approximately 1, regardless of the key dimension. If q and k have unit variance, then q·k/√d_k also has approximately unit variance, keeping attention scores in a reasonable range that doesn't saturate the softmax function.

This scaling provides several benefits: (1) Training stability - prevents extreme attention distributions that can destabilize learning, (2) Better gradient flow - maintains meaningful gradients through the attention mechanism, (3) Dimension independence - allows using the same learning rates and optimization strategies across different model sizes, and (4) Attention diversity - enables the model to attend to multiple positions rather than focusing too sharply on single positions.

Empirical evidence supports this choice: experiments show that removing the scaling factor leads to slower convergence and worse final performance, particularly for larger models with higher dimensional keys. The √d_k scaling has become a standard component of attention mechanisms across all major Transformer implementations.`
      },
      {
        question: 'What are the advantages of using multiple attention heads?',
        answer: `Multi-head attention is a crucial architectural innovation that enables Transformers to learn multiple types of relationships simultaneously by running several attention functions in parallel, each focusing on different aspects of the input relationships and operating in different representation subspaces.

The core advantage is representation diversity - different attention heads can specialize in different types of linguistic or semantic relationships. For example, one head might focus on syntactic dependencies like subject-verb relationships, another on semantic similarities, and another on positional patterns. This specialization allows the model to capture the rich, multifaceted nature of language more effectively than a single attention function.

Subspace learning is another key benefit. Each attention head operates on different learned projections of the input, effectively working in different representational subspaces. If the model dimension is d_model and we use h heads, each head works with dimension d_k = d_model/h. This partitioning allows heads to learn complementary representations that, when concatenated, provide a richer overall representation than a single large attention head.

Capacity and expressiveness increase significantly with multiple heads. Instead of learning one set of Query, Key, and Value projections, the model learns h different sets, multiplying the number of learnable parameters in the attention mechanism. This increased capacity enables more sophisticated pattern recognition and relationship modeling.

Attention pattern diversity is observable in practice - different heads learn to attend to different position patterns. Some heads focus on local dependencies (adjacent words), others on long-range relationships (sentence-level structure), and still others on semantic similarity regardless of position. This diversity enables comprehensive sequence understanding.

Robustness improves through redundancy - if one attention head learns suboptimal patterns or gets stuck in poor local minima during training, other heads can compensate. This redundancy makes the overall attention mechanism more reliable and less sensitive to initialization or training dynamics.

Computational efficiency is maintained despite increased parameters because heads can be computed in parallel. The total computational cost scales linearly with the number of heads, not quadratically, making multi-head attention practical even for large numbers of heads.

Empirical benefits are well-documented: ablation studies consistently show that multi-head attention outperforms single-head attention across diverse tasks. The optimal number of heads varies by task and model size, but 8-16 heads are common choices that balance expressiveness with computational efficiency. The multi-head design has become fundamental to Transformer success across domains.`
      },
      {
        question: 'What is the computational complexity of self-attention and why?',
        answer: `Self-attention has O(n²d) computational complexity where n is the sequence length and d is the model dimension, arising from the need to compute pairwise relationships between all positions in the sequence. Understanding this complexity is crucial for designing efficient Transformer architectures and understanding their scalability limitations.

The quadratic term O(n²) comes from computing attention scores between every pair of positions. For a sequence of length n, there are n² possible pairs, and each pair requires computing the similarity between their query and key vectors. This all-pairs computation is fundamental to self-attention's ability to model arbitrary long-range dependencies but creates a scalability bottleneck for long sequences.

The linear term O(d) reflects the cost of computing each pairwise similarity. Query and key vectors are d-dimensional, so computing their dot product requires d operations. Additionally, the subsequent aggregation of value vectors (also d-dimensional) based on attention weights contributes to the linear complexity in the model dimension.

Breaking down the computation steps: (1) Computing Q, K, V projections: O(n·d²) for matrix multiplications, (2) Computing attention scores QK^T: O(n²·d) for all pairwise dot products, (3) Applying softmax: O(n²) for normalizing each row, (4) Computing weighted values: O(n²·d) for aggregating value vectors. The dominant terms are the O(n²·d) operations.

Memory complexity also scales as O(n²) for storing the attention matrix, which becomes prohibitive for very long sequences. For sequences of 10,000 tokens, the attention matrix alone requires storing 100 million values, before considering gradients needed for backpropagation.

This complexity creates practical limitations: most Transformer models are limited to sequences of 512-4096 tokens due to memory and computational constraints. Processing longer sequences requires specialized techniques or architectural modifications.

Comparison with RNNs reveals interesting trade-offs: RNNs have O(n·d²) complexity (linear in sequence length but quadratic in hidden dimension), while Transformers have O(n²·d) complexity (quadratic in sequence length but linear in model dimension). For typical scenarios where d >> n, RNNs can be more efficient, but Transformers' parallelizability often compensates in practice.

Numerous solutions address this complexity: sparse attention patterns reduce the effective number of pairs computed, linear attention approximations achieve O(n) complexity, hierarchical attention applies attention at multiple granularities, and sliding window attention limits attention to local neighborhoods. These innovations aim to preserve self-attention's modeling benefits while improving scalability.`
      },
      {
        question: 'How does masked attention differ from regular attention?',
        answer: `Masked attention is a variant of self-attention that prevents certain positions from attending to others by setting their attention scores to negative infinity before applying softmax, effectively making their attention weights zero. This mechanism is essential for maintaining causal constraints in autoregressive generation and implementing various attention patterns.

The key difference lies in the attention score computation. Regular self-attention computes scores between all position pairs: scores = QK^T/√d_k, then applies softmax directly. Masked attention modifies this by adding a mask matrix: scores = (QK^T + M)/√d_k, where M contains 0 for allowed connections and -∞ (or a very large negative number) for forbidden connections.

Causal masking is the most common application, used in decoder self-attention to prevent positions from attending to future positions. The mask matrix M is upper triangular with -∞ above the diagonal, ensuring position i can only attend to positions j where j ≤ i. This maintains the autoregressive property essential for text generation.

The masking mechanism preserves the mathematical properties of attention while enforcing structural constraints. When -∞ values are passed through softmax, they become 0, effectively removing those connections from the weighted sum. This allows flexible control over attention patterns without changing the fundamental attention computation.

Different mask types serve various purposes: (1) Causal masks for autoregressive generation, (2) Padding masks to ignore padded tokens in variable-length sequences, (3) Attention masks to prevent attending to specific tokens (like special tokens), and (4) Structural masks to enforce linguistic or syntactic constraints.

Computational implications include both costs and benefits. Adding masks requires additional memory to store mask matrices and computational overhead to apply them. However, masks can also enable computational optimizations - causal masks allow triangular matrix operations, and sparse masks can reduce the number of computations needed.

Training dynamics are affected by masking patterns. Causal masking ensures consistency between training (with teacher forcing) and inference (autoregressive generation) by preventing the model from learning to rely on future information. This consistency is crucial for effective sequence generation.

Implementation considerations include efficient mask computation, proper handling of different mask types, and ensuring numerical stability when working with very large negative values. Modern frameworks provide optimized implementations that handle these details transparently while maintaining the flexibility to specify custom masking patterns.

Masked attention has enabled various architectural innovations including bidirectional encoders with causal decoders, sparse attention patterns for long sequences, and structured attention that incorporates linguistic knowledge into attention patterns.`
      },
      {
        question: 'What types of relationships can different attention heads learn?',
        answer: `Different attention heads in multi-head attention demonstrate remarkable ability to specialize in learning distinct types of linguistic, semantic, and structural relationships, making Transformers capable of modeling the rich complexity of natural language through distributed representation learning.

Syntactic relationships are commonly learned by specific attention heads. Research has identified heads that specialize in detecting subject-verb relationships, noun-adjective dependencies, prepositional phrase attachments, and other grammatical structures. These heads often show clear patterns where words attend to their syntactic governors or dependents, effectively learning implicit parse tree structures.

Semantic relationships emerge in other heads that focus on meaning-based connections rather than structural ones. These heads might connect semantically similar words (synonyms, antonyms), thematically related concepts, or words that participate in common semantic frames. For example, heads might learn to connect "doctor" with "hospital" or "teach" with "student."

Positional patterns represent another category where heads specialize in location-based relationships. Some heads focus on local dependencies (adjacent words), others on specific distance relationships (words separated by fixed distances), and still others on sentence-boundary or paragraph-level structures. These positional specialists help the model understand structural organization.

Coreferent relationships are learned by heads that track entity references across long spans. These heads excel at connecting pronouns to their antecedents, linking repeated mentions of entities, and maintaining coherent entity representations throughout documents. This capability is crucial for discourse understanding and coherent generation.

Functional word relationships involve heads that specialize in connecting content words with function words like articles, prepositions, and auxiliary verbs. These heads help the model understand how grammatical particles modify or relate to content-bearing elements, essential for proper syntactic interpretation.

Multi-word expression detection emerges in heads that learn to group words forming idiomatic expressions, compound nouns, or other multi-token units. These heads recognize that certain word combinations should be treated as coherent units rather than independent elements.

Hierarchical structure learning appears in heads that capture different levels of linguistic organization simultaneously. Some heads might focus on phrase-level groupings while others capture sentence-level or document-level relationships, creating a hierarchical understanding of text structure.

Task-specific specialization occurs when models are fine-tuned for particular applications. Heads might specialize in question-answer alignment for QA tasks, sentiment-bearing word detection for sentiment analysis, or entity-relationship patterns for information extraction.

Dynamic adaptation shows that attention heads can learn different patterns for different types of input or context. The same head might exhibit different specialization patterns when processing formal versus informal text, or technical versus conversational language, demonstrating remarkable flexibility.

This specialization emerges naturally through training rather than being explicitly programmed, highlighting the power of multi-head attention to discover and leverage diverse relationship types that contribute to effective language understanding and generation.`
      }
    ],
    quizQuestions: [
      {
        id: 'attn1',
        question: 'What is the computational complexity of self-attention with sequence length n?',
        options: ['O(n)', 'O(n log n)', 'O(n²)', 'O(n³)'],
        correctAnswer: 2,
        explanation: 'Self-attention has O(n²) complexity because each position must attend to every other position, resulting in n² attention scores.'
      },
      {
        id: 'attn2',
        question: 'Why do we use multiple heads in multi-head attention?',
        options: ['Reduce computation', 'Capture different aspects of relationships', 'Prevent overfitting', 'Speed up training'],
        correctAnswer: 1,
        explanation: 'Multiple heads allow the model to attend to different representation subspaces, capturing diverse semantic and syntactic relationships.'
      }
    ]
  },

  'positional-encoding': {
    id: 'positional-encoding',
    title: 'Positional Encoding',
    category: 'transformers',
    description: 'Methods to inject position information into Transformer models',
    content: `
      <h2>Positional Encoding</h2>
      <p>Since Transformers process all positions in parallel without recurrence or convolution, they have no inherent notion of position or order. Positional encodings inject information about token positions into the model.</p>

      <h3>The Problem</h3>
      <ul>
        <li><strong>Permutation invariance:</strong> Self-attention is permutation-equivariant</li>
        <li><strong>No sequential processing:</strong> Unlike RNNs, no inherent order</li>
        <li><strong>Position matters:</strong> "dog bites man" ≠ "man bites dog"</li>
        <li><strong>Need explicit encoding:</strong> Must add position information</li>
      </ul>

      <h3>Sinusoidal Positional Encoding</h3>
      <p>Original Transformer uses fixed sinusoidal functions:</p>
      <p>PE(pos, 2i) = sin(pos / 10000^(2i/d_model))</p>
      <p>PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))</p>

      <h4>Properties</h4>
      <ul>
        <li><strong>Fixed:</strong> Not learned, deterministic function</li>
        <li><strong>Unique:</strong> Each position gets unique encoding</li>
        <li><strong>Smooth:</strong> Similar positions have similar encodings</li>
        <li><strong>Relative positions:</strong> PE(pos+k) is linear function of PE(pos)</li>
        <li><strong>Extrapolation:</strong> Can handle longer sequences than training</li>
      </ul>

      <h3>Learned Positional Embeddings</h3>
      <p>Alternative approach: learn position embeddings like word embeddings</p>
      <ul>
        <li><strong>Trainable:</strong> Optimized during training</li>
        <li><strong>Task-specific:</strong> Adapts to task requirements</li>
        <li><strong>Fixed length:</strong> Limited to max training sequence length</li>
        <li><strong>Used in BERT:</strong> Learned absolute position embeddings</li>
      </ul>

      <h3>Relative Positional Encodings</h3>
      <p>Encode relative distances between positions instead of absolute positions:</p>
      <ul>
        <li><strong>Translation invariant:</strong> Same pattern at any position</li>
        <li><strong>Better generalization:</strong> To different sequence lengths</li>
        <li><strong>Used in:</strong> Transformer-XL, T5, DeBERTa</li>
        <li><strong>Implementation:</strong> Add to attention scores or keys/values</li>
      </ul>

      <h3>Rotary Position Embeddings (RoPE)</h3>
      <p>Modern approach used in models like GPT-NeoX, PaLM:</p>
      <ul>
        <li><strong>Rotation matrices:</strong> Rotate query and key vectors</li>
        <li><strong>Relative encoding:</strong> Dot product captures relative positions</li>
        <li><strong>Better extrapolation:</strong> Works well beyond training length</li>
        <li><strong>Efficiency:</strong> No additional parameters</li>
      </ul>

      <h3>Design Considerations</h3>
      <ul>
        <li><strong>Absolute vs relative:</strong> Trade-offs in expressiveness and generalization</li>
        <li><strong>Learned vs fixed:</strong> Flexibility vs parameter efficiency</li>
        <li><strong>Extrapolation:</strong> Handling sequences longer than training</li>
        <li><strong>Efficiency:</strong> Computational and memory costs</li>
      </ul>
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
  },

  'bert': {
    id: 'bert',
    title: 'BERT (Bidirectional Encoder Representations from Transformers)',
    category: 'transformers',
    description: 'Bidirectional pre-training for language understanding tasks',
    content: `
      <h2>BERT</h2>
      <p>BERT (2018) revolutionized NLP by introducing bidirectional pre-training. Unlike previous models that read text left-to-right or combined left-to-right and right-to-left training, BERT reads text in both directions simultaneously.</p>

      <h3>Key Innovation</h3>
      <ul>
        <li><strong>Bidirectional context:</strong> Attends to both left and right context</li>
        <li><strong>Pre-training + fine-tuning:</strong> Learn general representations, then adapt</li>
        <li><strong>Encoder-only:</strong> Uses only Transformer encoder stack</li>
        <li><strong>State-of-the-art:</strong> Dominated 11 NLP tasks when released</li>
      </ul>

      <h3>Architecture</h3>
      <ul>
        <li><strong>BERT-Base:</strong> 12 layers, 768 hidden, 12 heads, 110M parameters</li>
        <li><strong>BERT-Large:</strong> 24 layers, 1024 hidden, 16 heads, 340M parameters</li>
        <li><strong>Input:</strong> Token embeddings + segment embeddings + position embeddings</li>
        <li><strong>Special tokens:</strong> [CLS] for classification, [SEP] for separation</li>
      </ul>

      <h3>Pre-training Tasks</h3>

      <h4>1. Masked Language Modeling (MLM)</h4>
      <ul>
        <li><strong>Randomly mask 15% of tokens</strong></li>
        <li><strong>80%:</strong> Replace with [MASK]</li>
        <li><strong>10%:</strong> Replace with random token</li>
        <li><strong>10%:</strong> Keep original token</li>
        <li><strong>Objective:</strong> Predict masked tokens using bidirectional context</li>
      </ul>

      <h4>2. Next Sentence Prediction (NSP)</h4>
      <ul>
        <li><strong>Input:</strong> Two sentences A and B</li>
        <li><strong>50% of time:</strong> B follows A (IsNext)</li>
        <li><strong>50% of time:</strong> B is random sentence (NotNext)</li>
        <li><strong>Objective:</strong> Predict if B follows A</li>
        <li><strong>Purpose:</strong> Learn sentence-level relationships</li>
      </ul>

      <h3>Fine-tuning</h3>
      <p>BERT can be fine-tuned for various tasks:</p>
      <ul>
        <li><strong>Classification:</strong> Use [CLS] token representation</li>
        <li><strong>Named Entity Recognition:</strong> Token-level classification</li>
        <li><strong>Question Answering:</strong> Predict start/end positions</li>
        <li><strong>Sentence pairs:</strong> Entailment, similarity, paraphrase</li>
      </ul>

      <h3>Advantages</h3>
      <ul>
        <li><strong>Bidirectional:</strong> Full context from both directions</li>
        <li><strong>Transfer learning:</strong> Pre-train once, fine-tune for many tasks</li>
        <li><strong>Performance:</strong> Huge improvements on NLP benchmarks</li>
        <li><strong>Interpretability:</strong> Attention weights show linguistic patterns</li>
      </ul>

      <h3>Limitations</h3>
      <ul>
        <li><strong>Slow:</strong> Large model, computationally expensive</li>
        <li><strong>Memory intensive:</strong> Requires significant GPU memory</li>
        <li><strong>Not generative:</strong> Encoder-only, can't generate text</li>
        <li><strong>Fixed length:</strong> Maximum 512 tokens</li>
      </ul>

      <h3>Variants and Successors</h3>
      <ul>
        <li><strong>RoBERTa:</strong> Removes NSP, trains longer, dynamic masking</li>
        <li><strong>ALBERT:</strong> Parameter sharing, factorized embeddings</li>
        <li><strong>DistilBERT:</strong> 40% smaller, 60% faster, retains 97% performance</li>
        <li><strong>DeBERTa:</strong> Disentangled attention, enhanced masking</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch
import torch.nn as nn

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2  # Binary classification
)

# Example: Sentiment analysis
text = "This movie was absolutely fantastic!"
inputs = tokenizer(
    text,
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length=512
)

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.softmax(logits, dim=-1)

print(f"Positive: {predictions[0][1]:.3f}")
print(f"Negative: {predictions[0][0]:.3f}")

# Fine-tuning example
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Training loop
for epoch in range(3):
    # Your training data
    batch_texts = ["I love this!", "This is terrible"]
    batch_labels = torch.tensor([1, 0])  # 1: positive, 0: negative

    # Tokenize
    inputs = tokenizer(
        batch_texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )

    # Forward pass
    outputs = model(**inputs, labels=batch_labels)
    loss = outputs.loss

    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")`,
        explanation: 'Using pre-trained BERT for classification and fine-tuning on custom data.'
      },
      {
        language: 'Python',
        code: `from transformers import BertTokenizer, BertModel
import torch

# Load BERT for feature extraction
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# Example text
text = "BERT learns contextual representations."

# Tokenize with special tokens
inputs = tokenizer(
    text,
    return_tensors='pt',
    add_special_tokens=True  # Adds [CLS] and [SEP]
)

# Get BERT outputs
with torch.no_grad():
    outputs = model(**inputs)

    # Last hidden states: [batch, seq_len, hidden_size]
    last_hidden_states = outputs.last_hidden_state

    # Pooled output: [CLS] token representation
    pooled_output = outputs.pooler_output

print(f"Sequence length: {last_hidden_states.shape[1]}")
print(f"Hidden size: {last_hidden_states.shape[2]}")
print(f"[CLS] representation: {pooled_output.shape}")

# Token-level representations
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
for i, (token, hidden_state) in enumerate(zip(tokens, last_hidden_states[0])):
    print(f"{i}: {token:15} -> {hidden_state[:5]}")  # First 5 dims

# Attention visualization
from transformers import BertModel
model_with_attn = BertModel.from_pretrained(
    'bert-base-uncased',
    output_attentions=True
)

with torch.no_grad():
    outputs = model_with_attn(**inputs)
    attentions = outputs.attentions  # Tuple of attention weights per layer

# Attentions[0]: [batch, num_heads, seq_len, seq_len]
layer_0_attn = attentions[0]
print(f"Layer 0 attention shape: {layer_0_attn.shape}")
print(f"Number of layers with attention: {len(attentions)}")`,
        explanation: 'Extracting BERT representations and attention weights for analysis and visualization.'
      }
    ],
    interviewQuestions: [
      {
        question: 'What is the main difference between BERT and GPT?',
        answer: `BERT is an encoder-only model using bidirectional attention for understanding tasks, while GPT is a decoder-only model using causal attention for generation tasks. BERT uses masked language modeling (predicting masked tokens) for pre-training and excels at classification, QA, and understanding tasks. GPT uses next-token prediction for pre-training and excels at text generation and completion. BERT sees full context bidirectionally but cannot generate text, while GPT generates text autoregressively but only sees previous context.`
      },
      {
        question: 'Explain the Masked Language Modeling (MLM) pre-training objective.',
        answer: `MLM randomly masks 15% of input tokens and trains the model to predict the original tokens using bidirectional context. The masking strategy uses 80% [MASK] tokens, 10% random tokens, and 10% unchanged tokens to prevent over-reliance on [MASK] during fine-tuning. This objective enables learning rich bidirectional representations by requiring the model to understand context from both directions, making BERT effective for understanding tasks but unable to generate text sequentially.`
      },
      {
        question: 'Why does BERT use [CLS] and [SEP] tokens?',
        answer: `[CLS] (classification) provides a special position for aggregate sequence representation used in classification tasks - its final hidden state represents the entire input for downstream tasks. [SEP] (separator) marks boundaries between different segments in tasks requiring multiple inputs (like question-answering or sentence pair classification), enabling BERT to distinguish between different parts of the input while processing them jointly through attention mechanisms.`
      },
      {
        question: 'How would you fine-tune BERT for a named entity recognition task?',
        answer: `Add a token classification head on top of BERT's hidden states, where each token's representation is passed through a linear layer with softmax to predict entity labels (B-PER, I-ORG, O, etc.). Use sequence labeling loss (cross-entropy) over all tokens. Fine-tune the entire model or use techniques like gradual unfreezing. Handle subword tokenization by aligning BERT's wordpiece tokens with word-level entity labels, typically using the first subword's prediction or averaging subword predictions for each word.`
      },
      {
        question: 'What are the limitations of BERT compared to GPT models?',
        answer: `BERT cannot generate text due to bidirectional attention that breaks causality. It requires task-specific fine-tuning for most applications, unlike GPT's prompt-based versatility. BERT's [MASK] tokens create train-test mismatch since they don't appear during fine-tuning. GPT models are more suitable for few-shot learning and can handle diverse tasks through prompting, while BERT excels primarily at understanding tasks but requires supervised fine-tuning for each new application.`
      },
      {
        question: 'Why does MLM mask 15% of tokens with different strategies (80/10/10)?',
        answer: `The 80/10/10 strategy (80% [MASK], 10% random, 10% unchanged) prevents over-reliance on [MASK] tokens during fine-tuning when they're absent. Random token replacement forces the model to use context rather than token identity. Keeping 10% unchanged adds noise that improves robustness. This strategy ensures the model learns robust representations that work even when the exact pre-training condition ([MASK] tokens) isn't present during downstream task application.`
      }
    ],
    quizQuestions: [
      {
        id: 'bert1',
        question: 'What makes BERT bidirectional?',
        options: ['Uses two encoders', 'Reads text twice', 'Masked Language Modeling allows seeing full context', 'Processes text backwards'],
        correctAnswer: 2,
        explanation: 'BERT uses Masked Language Modeling (MLM) where masked tokens are predicted using context from both directions simultaneously, enabling truly bidirectional representations.'
      },
      {
        id: 'bert2',
        question: 'Which architecture component does BERT use?',
        options: ['Encoder only', 'Decoder only', 'Both encoder and decoder', 'Neither'],
        correctAnswer: 0,
        explanation: 'BERT uses only the Transformer encoder stack, making it suitable for understanding tasks but not for text generation.'
      },
      {
        id: 'bert3',
        question: 'What is the purpose of the [CLS] token in BERT?',
        options: ['Marks end of sequence', 'Separates sentences', 'Classification representation', 'Masking token'],
        correctAnswer: 2,
        explanation: 'The [CLS] token is added at the beginning of every input, and its final hidden state is used as the aggregate sequence representation for classification tasks.'
      }
    ]
  },

  'gpt': {
    id: 'gpt',
    title: 'GPT (Generative Pre-trained Transformer)',
    category: 'transformers',
    description: 'Autoregressive language models for text generation',
    content: `
      <h2>GPT (Generative Pre-trained Transformer)</h2>
      <p>The GPT series (GPT, GPT-2, GPT-3, GPT-4) represents decoder-only Transformer models trained for autoregressive text generation. Unlike BERT's bidirectional approach, GPT predicts the next token given all previous tokens.</p>

      <h3>Key Characteristics</h3>
      <ul>
        <li><strong>Decoder-only:</strong> Uses only Transformer decoder with causal masking</li>
        <li><strong>Autoregressive:</strong> Predicts next token given previous tokens</li>
        <li><strong>Left-to-right:</strong> Can only attend to previous positions</li>
        <li><strong>Pre-training:</strong> Language modeling on massive text corpora</li>
        <li><strong>Few-shot learning:</strong> Task performance from examples in prompt</li>
      </ul>

      <h3>Architecture</h3>
      <ul>
        <li><strong>Causal self-attention:</strong> Masked to prevent attending to future tokens</li>
        <li><strong>Layer normalization:</strong> Pre-norm (before attention/FFN)</li>
        <li><strong>Position embeddings:</strong> Learned absolute positions</li>
        <li><strong>Token prediction:</strong> Softmax over vocabulary at each position</li>
      </ul>

      <h4>Model Sizes</h4>
      <ul>
        <li><strong>GPT-1:</strong> 117M parameters, 12 layers</li>
        <li><strong>GPT-2:</strong> Up to 1.5B parameters, 48 layers</li>
        <li><strong>GPT-3:</strong> Up to 175B parameters, 96 layers</li>
        <li><strong>GPT-4:</strong> Size undisclosed, mixture of experts architecture</li>
      </ul>

      <h3>Pre-training Objective</h3>
      <p>Language modeling: maximize P(x_t | x_1, ..., x_{t-1})</p>
      <ul>
        <li><strong>Simple:</strong> Just predict next token</li>
        <li><strong>Self-supervised:</strong> No labels needed, learn from raw text</li>
        <li><strong>Scalable:</strong> Can train on internet-scale data</li>
        <li><strong>Universal:</strong> One objective, many downstream tasks</li>
      </ul>

      <h3>Inference and Generation</h3>

      <h4>Sampling Strategies</h4>
      <ul>
        <li><strong>Greedy:</strong> Always pick highest probability token</li>
        <li><strong>Beam search:</strong> Keep top-k sequences</li>
        <li><strong>Temperature sampling:</strong> Scale logits to control randomness</li>
        <li><strong>Top-k sampling:</strong> Sample from k most likely tokens</li>
        <li><strong>Top-p (nucleus):</strong> Sample from smallest set with cumulative probability p</li>
      </ul>

      <h3>Few-Shot Learning</h3>
      <p>GPT-3 demonstrated strong few-shot learning without fine-tuning:</p>
      <ul>
        <li><strong>Zero-shot:</strong> Task description only</li>
        <li><strong>One-shot:</strong> One example in prompt</li>
        <li><strong>Few-shot:</strong> Multiple examples in prompt</li>
        <li><strong>In-context learning:</strong> Learn from prompt without weight updates</li>
      </ul>

      <h3>GPT vs BERT</h3>
      <ul>
        <li><strong>Direction:</strong> GPT unidirectional (left-to-right), BERT bidirectional</li>
        <li><strong>Architecture:</strong> GPT decoder-only, BERT encoder-only</li>
        <li><strong>Task:</strong> GPT generation, BERT understanding</li>
        <li><strong>Objective:</strong> GPT language modeling, BERT masked LM + NSP</li>
        <li><strong>Use case:</strong> GPT for text generation, BERT for classification/NER</li>
      </ul>

      <h3>Applications</h3>
      <ul>
        <li><strong>Text generation:</strong> Stories, articles, code</li>
        <li><strong>Completion:</strong> Sentence/paragraph completion</li>
        <li><strong>Translation:</strong> Machine translation via prompting</li>
        <li><strong>Summarization:</strong> Condense long documents</li>
        <li><strong>Question answering:</strong> Answer questions in prompt</li>
        <li><strong>Code generation:</strong> Codex, GitHub Copilot</li>
        <li><strong>Chatbots:</strong> ChatGPT, conversational AI</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

# Text generation
prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors='pt')

# Generate with different strategies
print("=== Greedy Decoding ===")
outputs = model.generate(
    inputs['input_ids'],
    max_length=50,
    do_sample=False  # Greedy
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("\\n=== Temperature Sampling ===")
outputs = model.generate(
    inputs['input_ids'],
    max_length=50,
    do_sample=True,
    temperature=0.7  # Lower = more conservative
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("\\n=== Top-k Sampling ===")
outputs = model.generate(
    inputs['input_ids'],
    max_length=50,
    do_sample=True,
    top_k=50
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("\\n=== Top-p (Nucleus) Sampling ===")
outputs = model.generate(
    inputs['input_ids'],
    max_length=50,
    do_sample=True,
    top_p=0.9,
    temperature=0.8
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))`,
        explanation: 'GPT-2 text generation with different sampling strategies: greedy, temperature, top-k, and top-p.'
      },
      {
        language: 'Python',
        code: `import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=1024, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Causal mask: upper triangular matrix of -inf
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Linear projections and reshape for multi-head
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # Apply causal mask
        scores = scores.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))

        # Attention weights and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)

        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)

        return output

class GPTBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        # Pre-norm architecture
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pre-norm: normalize before attention
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# Usage
d_model = 768
num_heads = 12
block = GPTBlock(d_model, num_heads)

x = torch.randn(2, 10, d_model)  # [batch, seq_len, d_model]
output = block(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")`,
        explanation: 'GPT decoder block with causal self-attention and pre-norm architecture.'
      }
    ],
    interviewQuestions: [
      {
        question: 'What is the key difference between GPT and BERT architectures?',
        answer: `GPT uses decoder-only architecture with causal self-attention for autoregressive text generation, while BERT uses encoder-only architecture with bidirectional self-attention for understanding tasks. GPT predicts next tokens sequentially, making it ideal for generation, while BERT predicts masked tokens using full context, making it ideal for classification and understanding.`
      },
      {
        question: 'Explain why GPT uses causal (masked) self-attention.',
        answer: `Causal attention ensures that when predicting token t, the model only sees tokens 1 through t-1, never future tokens. This maintains the autoregressive property essential for text generation and ensures training-inference consistency. Without causal masking, the model would learn to "cheat" by looking ahead during training, creating a mismatch with generation where future tokens are unknown.`
      },
      {
        question: 'What is in-context learning and how does GPT-3 achieve it?',
        answer: `In-context learning is the ability to perform tasks by providing examples in the prompt without updating model parameters. GPT-3 achieves this through its large scale (175B parameters) and diverse training data, enabling it to recognize patterns in few-shot examples and apply them to new instances. The model learns to interpret prompts as task specifications and adapt its behavior accordingly.`
      },
      {
        question: 'Compare greedy decoding, beam search, and nucleus sampling for text generation.',
        answer: `Greedy decoding always picks the highest probability token (fast but can be repetitive). Beam search maintains multiple hypotheses (better quality but computationally expensive). Nucleus sampling selects from the top-p probability mass (balanced creativity and coherence). Each offers different trade-offs between speed, quality, and diversity for generation tasks.`
      },
      {
        question: 'Why is GPT better suited for text generation than BERT?',
        answer: `GPT's causal attention naturally supports autoregressive generation by predicting next tokens sequentially. BERT's bidirectional attention breaks causality needed for generation. GPT is trained on next-token prediction which directly matches the generation task, while BERT's masked language modeling doesn't train sequential generation capabilities.`
      },
      {
        question: 'How does temperature affect the quality and diversity of generated text?',
        answer: `Temperature controls randomness in sampling: low temperature (0.1-0.7) makes text more deterministic and coherent but potentially repetitive, while high temperature (0.8-1.5) increases diversity but may reduce coherence. Temperature = 0 is deterministic (argmax), while higher values flatten the probability distribution, increasing randomness in token selection.`
      }
    ],
    quizQuestions: [
      {
        id: 'gpt1',
        question: 'What architectural component does GPT use?',
        options: ['Encoder only', 'Decoder only', 'Encoder-decoder', 'CNN-based'],
        correctAnswer: 1,
        explanation: 'GPT uses only the Transformer decoder with causal self-attention, making it suitable for autoregressive text generation.'
      },
      {
        id: 'gpt2',
        question: 'What is the pre-training objective for GPT?',
        options: ['Masked Language Modeling', 'Next Sentence Prediction', 'Language Modeling (predict next token)', 'Translation'],
        correctAnswer: 2,
        explanation: 'GPT is trained on language modeling: predicting the next token given all previous tokens in an autoregressive manner.'
      },
      {
        id: 'gpt3',
        question: 'What does "causal" mean in causal self-attention?',
        options: ['Uses causality analysis', 'Prevents attending to future tokens', 'Finds causal relationships', 'Faster computation'],
        correctAnswer: 1,
        explanation: 'Causal self-attention masks future positions so each token can only attend to itself and previous tokens, maintaining the autoregressive property needed for generation.'
      }
    ]
  },

  't5-bart': {
    id: 't5-bart',
    title: 'T5 and BART',
    category: 'transformers',
    description: 'Encoder-decoder models for sequence-to-sequence tasks',
    content: `
      <h2>T5 and BART</h2>
      <p>T5 (Text-to-Text Transfer Transformer) and BART (Bidirectional and AutoRegressive Transformers) are encoder-decoder Transformer models that excel at sequence-to-sequence tasks like translation, summarization, and question answering.</p>

      <h3>T5: Text-to-Text Transfer Transformer</h3>

      <h4>Core Philosophy</h4>
      <p>Treat every NLP task as text-to-text:</p>
      <ul>
        <li><strong>Translation:</strong> "translate English to German: [text]" → German text</li>
        <li><strong>Classification:</strong> "sentiment: [review]" → "positive" or "negative"</li>
        <li><strong>Summarization:</strong> "summarize: [article]" → summary</li>
        <li><strong>Question answering:</strong> "question: [q] context: [c]" → answer</li>
      </ul>

      <h4>Architecture</h4>
      <ul>
        <li><strong>Encoder-decoder:</strong> Full Transformer architecture</li>
        <li><strong>Encoder:</strong> Bidirectional self-attention</li>
        <li><strong>Decoder:</strong> Causal self-attention + cross-attention to encoder</li>
        <li><strong>Relative position embeddings:</strong> Better length generalization</li>
      </ul>

      <h4>Pre-training</h4>
      <p>Span corruption objective (similar to masked LM but with spans):</p>
      <ul>
        <li><strong>Mask continuous spans:</strong> Replace spans with sentinel tokens</li>
        <li><strong>Input:</strong> "Thank you [X] me to [Y] party"</li>
        <li><strong>Target:</strong> "[X] for inviting [Y] your [Z]"</li>
        <li><strong>Corruption rate:</strong> 15% of tokens</li>
        <li><strong>Mean span length:</strong> 3 tokens</li>
      </ul>

      <h4>T5 Sizes</h4>
      <ul>
        <li><strong>T5-Small:</strong> 60M parameters</li>
        <li><strong>T5-Base:</strong> 220M parameters</li>
        <li><strong>T5-Large:</strong> 770M parameters</li>
        <li><strong>T5-3B:</strong> 3B parameters</li>
        <li><strong>T5-11B:</strong> 11B parameters</li>
      </ul>

      <h3>BART</h3>

      <h4>Architecture</h4>
      <ul>
        <li><strong>Encoder-decoder:</strong> Like T5, full Transformer</li>
        <li><strong>Encoder:</strong> Bidirectional (like BERT)</li>
        <li><strong>Decoder:</strong> Autoregressive (like GPT)</li>
        <li><strong>Position embeddings:</strong> Learned absolute positions</li>
      </ul>

      <h4>Pre-training Objectives</h4>
      <p>BART uses multiple noising strategies:</p>
      <ul>
        <li><strong>Token masking:</strong> Random tokens replaced with [MASK]</li>
        <li><strong>Token deletion:</strong> Random tokens deleted</li>
        <li><strong>Text infilling:</strong> Spans replaced with single [MASK]</li>
        <li><strong>Sentence permutation:</strong> Sentences shuffled</li>
        <li><strong>Document rotation:</strong> Document rotated to start at random token</li>
      </ul>

      <h4>Best Configuration</h4>
      <p>Text infilling + sentence permutation works best for most tasks</p>

      <h3>T5 vs BART</h3>
      <ul>
        <li><strong>Pre-training:</strong> T5 span corruption vs BART multiple noising</li>
        <li><strong>Task format:</strong> T5 text-to-text vs BART standard seq2seq</li>
        <li><strong>Position embeddings:</strong> T5 relative vs BART absolute</li>
        <li><strong>Training data:</strong> T5 on C4 (Colossal Clean Crawled Corpus)</li>
        <li><strong>Performance:</strong> Both strong, task-dependent which is better</li>
      </ul>

      <h3>Advantages</h3>
      <ul>
        <li><strong>Versatile:</strong> Handle many task types</li>
        <li><strong>Bidirectional encoder:</strong> Full context understanding</li>
        <li><strong>Generative decoder:</strong> Can produce variable-length outputs</li>
        <li><strong>Strong performance:</strong> SOTA on many benchmarks</li>
        <li><strong>Transfer learning:</strong> Pre-train once, fine-tune for tasks</li>
      </ul>

      <h3>Applications</h3>
      <ul>
        <li><strong>Summarization:</strong> Abstractive summarization of documents</li>
        <li><strong>Translation:</strong> Machine translation</li>
        <li><strong>Question answering:</strong> Generative QA</li>
        <li><strong>Dialogue:</strong> Conversational systems</li>
        <li><strong>Data-to-text:</strong> Generate text from structured data</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load T5 model
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')
model.eval()

# Summarization
text = """
The Transformer architecture has revolutionized natural language processing.
It uses self-attention mechanisms to process sequences in parallel, unlike
recurrent neural networks. This enables better performance and faster training
on modern hardware.
"""

# T5 requires task prefix
input_text = "summarize: " + text
inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)

with torch.no_grad():
    outputs = model.generate(
        inputs['input_ids'],
        max_length=50,
        num_beams=4,
        early_stopping=True
    )

summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Summary: {summary}")

# Translation
input_text = "translate English to German: The house is wonderful."
inputs = tokenizer(input_text, return_tensors='pt')

with torch.no_grad():
    outputs = model.generate(inputs['input_ids'], max_length=40)

translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Translation: {translation}")

# Question Answering
input_text = "question: What is the capital of France? context: Paris is the capital and largest city of France."
inputs = tokenizer(input_text, return_tensors='pt')

with torch.no_grad():
    outputs = model.generate(inputs['input_ids'], max_length=20)

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Answer: {answer}")`,
        explanation: 'T5 usage for multiple tasks using text-to-text format with task prefixes.'
      },
      {
        language: 'Python',
        code: `from transformers import BartForConditionalGeneration, BartTokenizer
import torch

# Load BART model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
model.eval()

# Summarization example
article = """
Machine learning has transformed many industries in recent years. From healthcare
to finance, ML algorithms are being deployed to make predictions, automate tasks,
and extract insights from data. Deep learning, a subset of machine learning, has
been particularly successful in domains like computer vision and natural language
processing. Neural networks with many layers can learn complex patterns from
large datasets, achieving human-level performance on many tasks.
"""

inputs = tokenizer(article, return_tensors='pt', max_length=1024, truncation=True)

# Generate summary
with torch.no_grad():
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=60,
        min_length=20,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Original length:", len(article.split()))
print("Summary length:", len(summary.split()))
print(f"\\nSummary: {summary}")

# Fine-tuning BART example
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

# Training data
source_text = "The quick brown fox jumps over the lazy dog."
target_text = "A fox jumps over a dog."

# Tokenize
source_ids = tokenizer(
    source_text,
    return_tensors='pt',
    padding='max_length',
    max_length=128,
    truncation=True
)['input_ids']

target_ids = tokenizer(
    target_text,
    return_tensors='pt',
    padding='max_length',
    max_length=128,
    truncation=True
)['input_ids']

# Forward pass
outputs = model(input_ids=source_ids, labels=target_ids)
loss = outputs.loss

# Backward pass
loss.backward()
optimizer.step()
optimizer.zero_grad()

print(f"\\nTraining loss: {loss.item():.4f}")`,
        explanation: 'BART for summarization with beam search and fine-tuning example.'
      }
    ],
    interviewQuestions: [
      {
        question: 'What is the key innovation of T5\'s text-to-text framework?',
        answer: `T5 treats every NLP task as text-to-text generation, using the same input-output format for all tasks. Whether classification, QA, or translation, inputs are converted to text prompts and outputs are generated as text. This unified approach enables training one model for diverse tasks and leverages the full power of sequence-to-sequence learning across all NLP applications.`
      },
      {
        question: 'Explain the difference between T5 span corruption and BERT masked LM.',
        answer: `T5 span corruption masks and removes contiguous spans of tokens (rather than individual tokens), replacing them with special tokens, then trains the model to generate the missing spans. BERT MLM masks individual tokens and predicts them using bidirectional context. T5's approach better matches downstream tasks requiring text generation and enables learning longer-range dependencies.`
      },
      {
        question: 'What are the advantages of encoder-decoder models over encoder-only or decoder-only?',
        answer: `Encoder-decoder models excel at tasks requiring clear input-output transformations (translation, summarization) by processing input bidirectionally in the encoder while maintaining autoregressive generation in the decoder. They provide better separation of understanding and generation phases compared to decoder-only models, and can perform generation tasks unlike encoder-only models.`
      },
      {
        question: 'How does BART combine ideas from BERT and GPT?',
        answer: `BART uses a denoising autoencoder approach: corrupt input text with various noise functions (token masking, deletion, shuffling), then train an encoder-decoder to reconstruct the original text. This combines BERT's bidirectional encoding for understanding with GPT's autoregressive decoding for generation, making it effective for both understanding and generation tasks.`
      },
      {
        question: 'What is text infilling and why is it effective for BART pre-training?',
        answer: `Text infilling involves removing spans of text and training the model to generate the missing content. It's effective because it requires both understanding the context (like BERT) and generating coherent text (like GPT). This pre-training task closely matches many downstream applications like summarization and translation where content must be generated based on input context.`
      },
      {
        question: 'When would you choose T5/BART over GPT or BERT?',
        answer: `Choose T5/BART for sequence-to-sequence tasks like translation, summarization, or any task requiring structured input-output transformation. They excel when you need both bidirectional understanding of input and controlled generation of output. Use GPT for open-ended generation and BERT for pure understanding tasks like classification.`
      }
    ],
    quizQuestions: [
      {
        id: 't5-1',
        question: 'What makes T5 unique among Transformer models?',
        options: ['Largest model', 'Treats all tasks as text-to-text', 'Fastest inference', 'Uses CNNs'],
        correctAnswer: 1,
        explanation: 'T5 reformulates all NLP tasks as text-to-text problems, using task prefixes to specify the operation and generating text as output for all tasks including classification.'
      },
      {
        id: 'bart1',
        question: 'What architecture does BART use?',
        options: ['Encoder only', 'Decoder only', 'Encoder-decoder', 'Recurrent'],
        correctAnswer: 2,
        explanation: 'BART uses a full encoder-decoder Transformer architecture, combining bidirectional encoding (like BERT) with autoregressive decoding (like GPT).'
      },
      {
        id: 't5-2',
        question: 'What is T5\'s span corruption objective?',
        options: ['Mask individual tokens', 'Predict next sentence', 'Mask continuous spans', 'Translate sentences'],
        correctAnswer: 2,
        explanation: 'T5 masks continuous spans of tokens and replaces them with sentinel tokens. The model must predict the original tokens for each masked span.'
      }
    ]
  },

  'fine-tuning-vs-prompt-engineering': {
    id: 'fine-tuning-vs-prompt-engineering',
    title: 'Fine-tuning vs Prompt Engineering',
    category: 'transformers',
    description: 'Different approaches to adapting pre-trained models for specific tasks',
    content: `
      <h2>Fine-tuning vs Prompt Engineering</h2>
      <p>Two main paradigms for adapting large pre-trained language models: fine-tuning (updating model weights) and prompt engineering (carefully crafting inputs). Each has distinct advantages, trade-offs, and use cases.</p>

      <h3>Fine-tuning</h3>

      <h4>Definition</h4>
      <p>Continue training a pre-trained model on task-specific data, updating the model's weights.</p>

      <h4>Approaches</h4>
      <ul>
        <li><strong>Full fine-tuning:</strong> Update all model parameters</li>
        <li><strong>Partial fine-tuning:</strong> Freeze some layers, update others</li>
        <li><strong>Adapter layers:</strong> Add small trainable modules, freeze base model</li>
        <li><strong>LoRA:</strong> Low-rank adaptation, add trainable low-rank matrices</li>
        <li><strong>Prefix tuning:</strong> Only tune continuous prefix vectors</li>
      </ul>

      <h4>Advantages</h4>
      <ul>
        <li><strong>Performance:</strong> Usually achieves best task-specific performance</li>
        <li><strong>Data efficiency:</strong> Works well with moderate amounts of labeled data</li>
        <li><strong>Consistency:</strong> More reliable, less sensitive to prompt wording</li>
        <li><strong>Specialization:</strong> Can learn task-specific patterns deeply</li>
      </ul>

      <h4>Disadvantages</h4>
      <ul>
        <li><strong>Computational cost:</strong> Requires training compute and time</li>
        <li><strong>Storage:</strong> Need to store model weights for each task</li>
        <li><strong>Data requirements:</strong> Needs labeled training data</li>
        <li><strong>Deployment:</strong> More complex to deploy multiple models</li>
        <li><strong>Catastrophic forgetting:</strong> May lose general capabilities</li>
      </ul>

      <h3>Prompt Engineering</h3>

      <h4>Definition</h4>
      <p>Carefully design input prompts to elicit desired behavior from pre-trained models without weight updates.</p>

      <h4>Techniques</h4>
      <ul>
        <li><strong>Zero-shot:</strong> Task description only, no examples</li>
        <li><strong>Few-shot:</strong> Include example input-output pairs</li>
        <li><strong>Chain-of-thought:</strong> Prompt for step-by-step reasoning</li>
        <li><strong>Instruction following:</strong> Clear task instructions</li>
        <li><strong>Role prompting:</strong> "You are an expert in..."</li>
        <li><strong>Template-based:</strong> Structured prompt templates</li>
      </ul>

      <h4>Advantages</h4>
      <ul>
        <li><strong>No training:</strong> Immediate deployment, no compute cost</li>
        <li><strong>Flexibility:</strong> Easy to iterate and modify</li>
        <li><strong>Single model:</strong> One model for many tasks</li>
        <li><strong>No labeled data:</strong> Can work with just examples or descriptions</li>
        <li><strong>Rapid prototyping:</strong> Test ideas quickly</li>
      </ul>

      <h4>Disadvantages</h4>
      <ul>
        <li><strong>Sensitivity:</strong> Performance varies with prompt wording</li>
        <li><strong>Requires large models:</strong> Only works well with very large LMs</li>
        <li><strong>Context limits:</strong> Limited by model's context window</li>
        <li><strong>Less specialized:</strong> May not match fine-tuned performance</li>
        <li><strong>Inconsistent:</strong> Can be unpredictable across inputs</li>
      </ul>

      <h3>When to Use Each</h3>

      <h4>Use Fine-tuning When:</h4>
      <ul>
        <li>You have sufficient labeled training data (100s to 1000s of examples)</li>
        <li>Need maximum performance on specific task</li>
        <li>Task requires specialized knowledge or behavior</li>
        <li>Consistency and reliability are critical</li>
        <li>Deployment complexity is acceptable</li>
      </ul>

      <h4>Use Prompt Engineering When:</h4>
      <ul>
        <li>Limited or no labeled training data</li>
        <li>Need to support many diverse tasks</li>
        <li>Rapid iteration and experimentation needed</li>
        <li>Have access to very large models (GPT-3, GPT-4)</li>
        <li>Training compute is unavailable or expensive</li>
      </ul>

      <h3>Hybrid Approaches</h3>
      <ul>
        <li><strong>Prompt tuning:</strong> Learn soft prompts, freeze model</li>
        <li><strong>Instruction tuning:</strong> Fine-tune on diverse instruction-following tasks</li>
        <li><strong>Few-shot then fine-tune:</strong> Use prompting for prototyping, fine-tune for production</li>
        <li><strong>Fine-tune with augmentation:</strong> Use prompting to generate training data</li>
      </ul>

      <h3>Parameter-Efficient Fine-tuning (PEFT)</h3>
      <p>Modern techniques balance both approaches:</p>
      <ul>
        <li><strong>LoRA:</strong> Add low-rank matrices, train only 0.1% of parameters</li>
        <li><strong>Adapters:</strong> Small bottleneck layers between Transformer layers</li>
        <li><strong>Prefix/Prompt tuning:</strong> Learn continuous prompt embeddings</li>
        <li><strong>Benefits:</strong> Fine-tuning performance with prompting efficiency</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# === PROMPT ENGINEERING ===
print("=== Prompt Engineering Examples ===\\n")

# Zero-shot
prompt_zero = "Classify the sentiment of this review as positive or negative:\\nReview: This movie was terrible.\\nSentiment:"
inputs = tokenizer(prompt_zero, return_tensors='pt')
outputs = model.generate(**inputs, max_length=inputs['input_ids'].shape[1] + 10)
print("Zero-shot:", tokenizer.decode(outputs[0], skip_special_tokens=True))

# Few-shot
prompt_few = """Classify sentiment as positive or negative:

Review: I loved this movie!
Sentiment: positive

Review: Waste of time and money.
Sentiment: negative

Review: Absolutely fantastic experience.
Sentiment: positive

Review: This product is awful.
Sentiment:"""

inputs = tokenizer(prompt_few, return_tensors='pt')
outputs = model.generate(**inputs, max_length=inputs['input_ids'].shape[1] + 5)
print("\\nFew-shot:", tokenizer.decode(outputs[0], skip_special_tokens=True)[-10:])

# Chain-of-thought
prompt_cot = """Question: If I have 3 apples and buy 2 more, then give 1 away, how many do I have?
Let's think step by step:
1. Start with 3 apples
2. Buy 2 more: 3 + 2 = 5 apples
3. Give 1 away: 5 - 1 = 4 apples
Answer: 4

Question: If I have 5 books and buy 3 more, then give 2 away, how many do I have?
Let's think step by step:"""

inputs = tokenizer(prompt_cot, return_tensors='pt')
outputs = model.generate(**inputs, max_length=inputs['input_ids'].shape[1] + 50)
print("\\nChain-of-thought:", tokenizer.decode(outputs[0], skip_special_tokens=True)[-100:])`,
        explanation: 'Prompt engineering examples: zero-shot, few-shot, and chain-of-thought prompting.'
      },
      {
        language: 'Python',
        code: `from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# === FINE-TUNING ===
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Custom dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.encodings['input_ids'][idx]
        return item

# Training data
texts = [
    "This movie is great! Positive:",
    "I hated this product. Negative:",
    "Absolutely loved it! Positive:",
    "Terrible experience. Negative:"
]
labels = [1, 0, 1, 0]

dataset = SentimentDataset(texts, labels, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=100,
    save_total_limit=2,
    learning_rate=5e-5,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Fine-tune
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete!")

# === LoRA (Parameter-Efficient Fine-tuning) ===
from peft import get_peft_model, LoraConfig, TaskType

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # Low-rank dimension
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn"]  # Which layers to apply LoRA
)

# Wrap model with LoRA
model = GPT2LMHeadModel.from_pretrained('gpt2')
lora_model = get_peft_model(model, lora_config)

# Check trainable parameters
trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in lora_model.parameters())
print(f"\\nLoRA trainable parameters: {trainable_params:,} / {total_params:,}")
print(f"Percentage: {100 * trainable_params / total_params:.2f}%")`,
        explanation: 'Fine-tuning examples: full fine-tuning with Trainer and parameter-efficient LoRA fine-tuning.'
      }
    ],
    interviewQuestions: [
      {
        question: 'What are the main trade-offs between fine-tuning and prompt engineering?',
        answer: `Fine-tuning adapts model parameters for specific tasks, providing higher performance but requiring labeled data, computational resources, and separate model storage. Prompt engineering guides model behavior through input design without parameter updates, enabling quick adaptation but with potentially lower performance. Fine-tuning offers customization while prompting offers flexibility and rapid deployment.`
      },
      {
        question: 'When would you choose prompt engineering over fine-tuning?',
        answer: `Choose prompt engineering when: (1) Limited labeled data or computational resources, (2) Need rapid prototyping or deployment, (3) Want to leverage existing model capabilities without modification, (4) Handling multiple tasks with one model, (5) Regulatory constraints prevent model modification, or (6) Working with very large models where fine-tuning is impractical.`
      },
      {
        question: 'Explain parameter-efficient fine-tuning methods like LoRA.',
        answer: `LoRA (Low-Rank Adaptation) freezes pretrained weights and adds trainable low-rank matrices to attention layers, dramatically reducing trainable parameters (often 1000x fewer) while maintaining performance. Other methods include adapters (small feedforward networks), prefix tuning (learnable prefixes), and P-tuning (optimized prompts). These methods enable task adaptation with minimal storage and computational overhead.`
      },
      {
        question: 'What is chain-of-thought prompting and why is it effective?',
        answer: `Chain-of-thought prompting includes step-by-step reasoning examples in prompts, encouraging models to generate intermediate reasoning steps before final answers. It's effective because it activates the model's reasoning capabilities learned during training, improves performance on complex tasks requiring multi-step logic, and provides interpretable reasoning paths that can be verified.`
      },
      {
        question: 'How does instruction tuning differ from traditional fine-tuning?',
        answer: `Instruction tuning trains models on diverse tasks formatted as natural language instructions, teaching models to follow instructions generally rather than solving specific tasks. Traditional fine-tuning adapts models for single tasks with task-specific formats. Instruction tuning creates more versatile models that can handle new tasks through instructions without additional training.`
      },
      {
        question: 'What are the storage and deployment implications of fine-tuning vs prompting?',
        answer: `Fine-tuning requires storing separate model copies for each task (GBs per model), complex deployment pipelines, and version management. Prompting uses one model for multiple tasks, requires only prompt storage (KBs), enables simpler deployment, and easier A/B testing. However, prompting may have higher inference costs due to longer inputs and potentially more API calls.`
      }
    ],
    quizQuestions: [
      {
        id: 'ft-pe-1',
        question: 'What is a key advantage of prompt engineering over fine-tuning?',
        options: ['Better performance', 'No training compute required', 'More consistent', 'Works with small models'],
        correctAnswer: 1,
        explanation: 'Prompt engineering requires no training or weight updates, making it immediately deployable without computational cost, though it typically requires very large pre-trained models.'
      },
      {
        id: 'ft-pe-2',
        question: 'What is LoRA (Low-Rank Adaptation)?',
        options: ['A prompting technique', 'Parameter-efficient fine-tuning method', 'A new architecture', 'Data augmentation'],
        correctAnswer: 1,
        explanation: 'LoRA is a parameter-efficient fine-tuning method that adds trainable low-rank matrices to model layers, allowing fine-tuning with only ~0.1% of parameters trainable.'
      },
      {
        id: 'ft-pe-3',
        question: 'When is fine-tuning preferred over prompt engineering?',
        options: ['No training data', 'Need maximum task performance', 'Supporting many tasks', 'Rapid prototyping'],
        correctAnswer: 1,
        explanation: 'Fine-tuning is preferred when you need maximum performance on a specific task and have sufficient labeled training data, as it allows the model to deeply specialize.'
      }
    ]
  },

  'large-language-models': {
    id: 'large-language-models',
    title: 'Large Language Models (LLMs)',
    category: 'transformers',
    description: 'Modern foundation models and their capabilities',
    content: `
      <h2>Large Language Models (LLMs)</h2>
      <p>Large Language Models are Transformer-based models trained on massive text corpora, exhibiting emergent capabilities like few-shot learning, reasoning, and instruction following. They serve as foundation models for diverse NLP applications.</p>

      <h3>Defining Characteristics</h3>
      <ul>
        <li><strong>Scale:</strong> Billions to trillions of parameters</li>
        <li><strong>Training data:</strong> Trained on internet-scale text (hundreds of billions to trillions of tokens)</li>
        <li><strong>Emergent abilities:</strong> Capabilities that appear only at large scale</li>
        <li><strong>General purpose:</strong> Can handle diverse tasks without task-specific training</li>
        <li><strong>Few-shot learning:</strong> Learn from examples in context</li>
      </ul>

      <h3>Notable LLMs</h3>

      <h4>GPT Family (OpenAI)</h4>
      <ul>
        <li><strong>GPT-3:</strong> 175B parameters, 2020</li>
        <li><strong>ChatGPT:</strong> GPT-3.5 with RLHF, 2022</li>
        <li><strong>GPT-4:</strong> Multimodal, undisclosed size, 2023</li>
      </ul>

      <h4>PaLM (Google)</h4>
      <ul>
        <li><strong>PaLM:</strong> 540B parameters, Pathways architecture</li>
        <li><strong>PaLM 2:</strong> More efficient, multilingual</li>
      </ul>

      <h4>LLaMA (Meta)</h4>
      <ul>
        <li><strong>Open source:</strong> 7B to 70B parameters</li>
        <li><strong>Efficient:</strong> Strong performance at smaller sizes</li>
        <li><strong>LLaMA 2:</strong> Commercially licensed, chat-optimized</li>
      </ul>

      <h4>Claude (Anthropic)</h4>
      <ul>
        <li><strong>Constitutional AI:</strong> Values-aligned training</li>
        <li><strong>Long context:</strong> 100K+ token context window</li>
        <li><strong>Helpful, harmless, honest:</strong> Focus on safety</li>
      </ul>

      <h3>Emergent Abilities</h3>
      <p>Capabilities that emerge at scale but not in smaller models:</p>
      <ul>
        <li><strong>In-context learning:</strong> Learn new tasks from prompts</li>
        <li><strong>Chain-of-thought reasoning:</strong> Multi-step logical reasoning</li>
        <li><strong>Instruction following:</strong> Understand and execute complex instructions</li>
        <li><strong>Task composition:</strong> Combine multiple skills</li>
        <li><strong>World knowledge:</strong> Broad factual knowledge</li>
      </ul>

      <h3>Training Pipeline</h3>

      <h4>1. Pre-training</h4>
      <ul>
        <li><strong>Objective:</strong> Next token prediction</li>
        <li><strong>Data:</strong> Massive web crawls, books, code</li>
        <li><strong>Compute:</strong> Thousands of GPUs/TPUs, weeks to months</li>
        <li><strong>Cost:</strong> Millions to tens of millions of dollars</li>
      </ul>

      <h4>2. Instruction Tuning</h4>
      <ul>
        <li><strong>Data:</strong> Instruction-response pairs</li>
        <li><strong>Goal:</strong> Improve instruction following</li>
        <li><strong>Examples:</strong> Flan, InstructGPT</li>
      </ul>

      <h4>3. RLHF (Reinforcement Learning from Human Feedback)</h4>
      <ul>
        <li><strong>Reward model:</strong> Train on human preferences</li>
        <li><strong>RL optimization:</strong> PPO to maximize reward</li>
        <li><strong>Goal:</strong> Align with human values and preferences</li>
        <li><strong>Used in:</strong> ChatGPT, Claude</li>
      </ul>

      <h3>Technical Challenges</h3>
      <ul>
        <li><strong>Computational cost:</strong> Training and inference are expensive</li>
        <li><strong>Memory requirements:</strong> Models too large for single GPU</li>
        <li><strong>Inference latency:</strong> Autoregressive generation is slow</li>
        <li><strong>Context length:</strong> Limited by quadratic attention complexity</li>
        <li><strong>Hallucinations:</strong> Generate plausible but false information</li>
      </ul>

      <h3>Optimization Techniques</h3>
      <ul>
        <li><strong>Model parallelism:</strong> Split model across devices</li>
        <li><strong>Quantization:</strong> Reduce precision (FP16, INT8, INT4)</li>
        <li><strong>Flash Attention:</strong> Memory-efficient attention</li>
        <li><strong>KV caching:</strong> Cache key-value pairs during generation</li>
        <li><strong>Sparse attention:</strong> Reduce O(n²) complexity</li>
        <li><strong>Mixture of Experts:</strong> Activate subset of parameters</li>
      </ul>

      <h3>Applications</h3>
      <ul>
        <li><strong>Chatbots:</strong> Conversational AI assistants</li>
        <li><strong>Content generation:</strong> Articles, stories, marketing copy</li>
        <li><strong>Code generation:</strong> GitHub Copilot, code completion</li>
        <li><strong>Question answering:</strong> Information retrieval and synthesis</li>
        <li><strong>Translation:</strong> Multilingual translation</li>
        <li><strong>Summarization:</strong> Document and article summarization</li>
        <li><strong>Data analysis:</strong> Extract insights from text</li>
      </ul>

      <h3>Safety and Alignment</h3>
      <ul>
        <li><strong>Bias:</strong> Models reflect biases in training data</li>
        <li><strong>Toxicity:</strong> Can generate harmful content</li>
        <li><strong>Misinformation:</strong> Hallucinations and false facts</li>
        <li><strong>Alignment:</strong> Ensuring models follow intended goals</li>
        <li><strong>Red teaming:</strong> Adversarial testing for failures</li>
        <li><strong>Constitutional AI:</strong> Values-based training</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load LLaMA 2 (open source LLM)
model_name = "meta-llama/Llama-2-7b-chat-hf"  # 7B parameter model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use FP16 for efficiency
    device_map="auto"  # Automatically distribute across GPUs
)

# Chat template for LLaMA 2
def create_prompt(system_message, user_message):
    return f"""<s>[INST] <<SYS>>
{system_message}
<</SYS>>

{user_message} [/INST]"""

# Example conversation
system = "You are a helpful AI assistant."
user_msg = "Explain what a large language model is in simple terms."

prompt = create_prompt(system, user_msg)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

# === Streaming generation (token by token) ===
from transformers import TextIteratorStreamer
from threading import Thread

streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

generation_kwargs = dict(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    streamer=streamer
)

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

print("\\n=== Streaming Response ===")
for new_text in streamer:
    print(new_text, end="", flush=True)

thread.join()`,
        explanation: 'Loading and using an open-source LLM (LLaMA 2) with efficient inference and streaming generation.'
      },
      {
        language: 'Python',
        code: `import openai
import os

# Using OpenAI API for GPT-4 (closed-source LLM)
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Basic completion ===
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    temperature=0.7,
    max_tokens=200
)
print("GPT-4 Response:")
print(response.choices[0].message.content)

# === Few-shot learning ===
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a sentiment classifier."},
        {"role": "user", "content": "Review: I loved this movie!"},
        {"role": "assistant", "content": "Sentiment: Positive"},
        {"role": "user", "content": "Review: Terrible waste of time."},
        {"role": "assistant", "content": "Sentiment: Negative"},
        {"role": "user", "content": "Review: This product exceeded my expectations."}
    ],
    temperature=0.3,
    max_tokens=10
)
print("\\nFew-shot classification:", response.choices[0].message.content)

# === Function calling (tool use) ===
functions = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
]

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "What's the weather in Paris?"}
    ],
    functions=functions,
    function_call="auto"
)

message = response.choices[0].message
if message.get("function_call"):
    print("\\nFunction call:", message.function_call)

# === Streaming response ===
print("\\n=== Streaming Response ===")
stream = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Write a haiku about AI."}
    ],
    stream=True,
    temperature=0.8
)

for chunk in stream:
    if chunk.choices[0].delta.get("content"):
        print(chunk.choices[0].delta.content, end="", flush=True)`,
        explanation: 'Using OpenAI GPT-4 API for various LLM capabilities: completion, few-shot learning, function calling, and streaming.'
      }
    ],
    interviewQuestions: [
      {
        question: 'What makes a language model "large" and why does scale matter?',
        answer: `LLMs are defined by scale: billions/trillions of parameters, massive training datasets, and significant computational requirements. Scale matters due to emergent abilities - capabilities that appear suddenly at certain model sizes, improved few-shot learning, better reasoning abilities, and more robust performance across diverse tasks. Scaling laws show consistent improvements with size, though with diminishing returns.`
      },
      {
        question: 'Explain RLHF and why it is used to train models like ChatGPT.',
        answer: `RLHF (Reinforcement Learning from Human Feedback) aligns LLM outputs with human preferences through three stages: supervised fine-tuning on demonstrations, training a reward model from human preferences, and using RL to optimize the language model against the reward model. This addresses the misalignment between maximizing likelihood and generating helpful, harmless, honest responses.`
      },
      {
        question: 'What are emergent abilities in LLMs and at what scale do they appear?',
        answer: `Emergent abilities are capabilities that appear suddenly at certain model scales rather than gradually improving. Examples include in-context learning, chain-of-thought reasoning, and complex instruction following. These typically emerge around 10-100B parameters, though the exact thresholds vary by task. The phenomenon suggests qualitative changes in model capabilities with scale.`
      },
      {
        question: 'How do instruction-tuned models differ from base language models?',
        answer: `Base LLMs are trained only on next-token prediction and may not follow instructions well. Instruction-tuned models undergo additional training on instruction-following datasets, learning to understand and execute diverse tasks based on natural language instructions. This makes them more useful as AI assistants while potentially reducing some generative capabilities.`
      },
      {
        question: 'What optimization techniques make LLM inference practical?',
        answer: `Key techniques include: model quantization (reducing precision from FP32 to INT8/4), KV-cache optimization for autoregressive generation, attention pattern optimization (sparse/local attention), model parallelism across multiple GPUs, speculative decoding, batching strategies, and specialized hardware (TPUs, inference-optimized chips). These collectively reduce memory, computation, and latency.`
      },
      {
        question: 'Explain the trade-offs between open-source (LLaMA) and closed-source (GPT-4) LLMs.',
        answer: `Open-source models offer customization, transparency, data privacy, and cost control but may have lower performance and require technical expertise. Closed-source models provide higher performance, easier integration, and professional support but limit customization, raise privacy concerns, and create vendor dependence. Choice depends on specific requirements and constraints.`
      },
      {
        question: 'What is the hallucination problem in LLMs and how can it be mitigated?',
        answer: `Hallucination refers to LLMs generating plausible-sounding but factually incorrect information. Mitigation strategies include: retrieval-augmented generation (grounding in external knowledge), improved training data quality, RLHF for truthfulness, confidence estimation, fact-checking systems, and prompt engineering techniques that encourage accuracy and source citation.`
      }
    ],
    quizQuestions: [
      {
        id: 'llm1',
        question: 'What is RLHF (Reinforcement Learning from Human Feedback)?',
        options: ['A pre-training method', 'Fine-tuning using human preferences', 'Data augmentation', 'Model compression'],
        correctAnswer: 1,
        explanation: 'RLHF trains a reward model on human preferences, then uses reinforcement learning to optimize the LLM to generate outputs that maximize the reward, aligning it with human values.'
      },
      {
        id: 'llm2',
        question: 'What is an "emergent ability" in LLMs?',
        options: ['Any learned capability', 'Abilities that appear only at large scale', 'Pre-trained skills', 'Fast inference'],
        correctAnswer: 1,
        explanation: 'Emergent abilities are capabilities like chain-of-thought reasoning and few-shot learning that appear suddenly at a certain scale but are not present in smaller models.'
      },
      {
        id: 'llm3',
        question: 'What is the primary objective during LLM pre-training?',
        options: ['Classification', 'Next token prediction', 'Translation', 'Summarization'],
        correctAnswer: 1,
        explanation: 'LLMs are pre-trained using language modeling: predicting the next token given previous tokens. This simple objective, when applied at massive scale, leads to broad language understanding.'
      }
    ]
  }
};