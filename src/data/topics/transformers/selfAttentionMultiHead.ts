import { Topic } from '../../../types';

export const selfAttentionMultiHead: Topic = {
  id: 'self-attention-multi-head',
  title: 'Self-Attention and Multi-Head Attention',
  category: 'transformers',
  description: 'Core mechanism allowing models to weigh the importance of different positions',
  content: `
    <h2>Self-Attention and Multi-Head Attention: The Heart of Transformers</h2>
    <p>Self-attention represents the fundamental innovation that powers Transformer architectures, enabling models to weigh the importance of different positions when processing sequences. By allowing every position to directly attend to every other position, self-attention provides flexible, content-based routing of information that captures arbitrary dependencies. Multi-head attention extends this mechanism by computing attention multiple times in parallel with different learned projections, enabling the model to simultaneously focus on different types of relationships and representation subspaces.</p>

    <h3>The Self-Attention Mechanism: Query, Key, Value</h3>
    <p>Self-attention is elegantly simple yet remarkably powerful. For each position in the input sequence, we compute three vectors through learned linear projections:</p>

    <h4>The Three Fundamental Vectors</h4>
    <ul>
      <li><strong>Query (Q):</strong> Represents "what information is this position looking for?" The query vector encodes what kind of information the current position needs from other positions in the sequence.</li>
      <li><strong>Key (K):</strong> Represents "what information does this position offer?" The key vector encodes what information this position can provide to queries from other positions.</li>
      <li><strong>Value (V):</strong> Represents "the actual content at this position." The value vector contains the information that will be propagated through the attention mechanism.</li>
    </ul>

    <p><strong>Intuition:</strong> Think of self-attention as a database lookup. The query is your search request, keys are indices that help match relevant content, and values are the actual data you retrieve. Each position generates all three roles simultaneously.</p>

    <h4>The Attention Computation: Scaled Dot-Product</h4>
    <p><strong>Formula:</strong> $$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$</p>

    <p><strong>Step-by-step breakdown:</strong></p>
    <ol>
      <li><strong>$QK^T$:</strong> Compute dot products between all queries and all keys, creating an $n \\times n$ similarity matrix. High dot product means query $i$ and key $j$ are well-aligned.</li>
      <li><strong>$/ \\sqrt{d_k}$:</strong> Scale by square root of key dimension. Without scaling, dot products grow large as dimensionality increases, pushing softmax into regions with extremely small gradients. This normalization keeps values in a reasonable range.</li>
      <li><strong>softmax:</strong> Convert similarity scores into probability distribution over positions. Each row sums to 1, representing how much attention position $i$ pays to all positions $j$.</li>
      <li><strong>$\\times V$:</strong> Weighted sum of value vectors. Each position's output is a combination of all value vectors, weighted by attention scores.</li>
    </ol>

    <h4>Concrete Numerical Example</h4>
    <p>Let's walk through attention with 3 tokens and 4 dimensions:</p>
    <pre>
Tokens: ["cat", "sat", "mat"]
d_k = 4 (dimension)

Step 1: Query, Key, Value matrices (simplified, actual values would be from learned projections)
Q = [[1, 0, 1, 0],   # "cat" query
   [0, 1, 0, 1],   # "sat" query  
   [1, 1, 0, 0]]   # "mat" query

K = [[1, 0, 1, 0],   # "cat" key
   [0, 1, 1, 0],   # "sat" key
   [1, 1, 0, 1]]   # "mat" key

V = [[1, 0, 0, 0],   # "cat" value
   [0, 1, 0, 0],   # "sat" value
   [0, 0, 1, 1]]   # "mat" value

Step 2: Compute QK^T (dot products)
QK^T = [[2, 1, 2],    # "cat" attends to each token
      [1, 2, 1],    # "sat" attends to each token
      [1, 1, 0]]    # "mat" attends to each token

Step 3: Scale by $\\sqrt{d_k} = \\sqrt{4} = 2$
Scaled = [[1.0, 0.5, 1.0],
        [0.5, 1.0, 0.5],
        [0.5, 0.5, 0.0]]

Step 4: Softmax (per row)
Attention Weights = [[0.38, 0.24, 0.38],  # "cat" attention distribution
                   [0.24, 0.51, 0.24],  # "sat" attention distribution  
                   [0.38, 0.38, 0.24]]  # "mat" attention distribution

Interpretation: "sat" attends most to itself (0.51), while "cat" splits
attention between itself and "mat" (0.38 each)

Step 5: Multiply by V (weighted sum)
Output for "cat" = 0.38*[1,0,0,0] + 0.24*[0,1,0,0] + 0.38*[0,0,1,1]
               = [0.38, 0.24, 0.38, 0.38]

Similarly for "sat" and "mat". Each output is a contextualized representation
combining information from all tokens based on attention weights.
</pre>

    <p><strong>Why scaling matters:</strong> For dimension $d_k=64$, random dot products have variance $d_k$. Without scaling, as $d_k$ increases, softmax becomes peaked around maximum values with tiny gradients elsewhere. Scaling by $\\sqrt{d_k}$ normalizes variance to 1, maintaining gradient flow.</p>

    <h4>Mathematical Properties</h4>
    <ul>
      <li><strong>Permutation equivariance:</strong> Attention output for position i doesn't depend on the order of positions, only on their content and distances in embedding space. This is why positional encoding is necessary.</li>
      <li><strong>Complexity:</strong> $O(n^2d)$ where $n$ is sequence length, $d$ is model dimension. Quadratic in sequence length but linear in dimension. Matrix multiplication $QK^T$ is $O(n^2d)$.</li>
      <li><strong>Parallelizability:</strong> All positions computed simultaneously through matrix operations, fully utilizing GPU parallelism.</li>
      <li><strong>Differentiability:</strong> Entire operation is differentiable, enabling end-to-end training with backpropagation.</li>
    </ul>

    <h3>Multi-Head Attention: Parallel Perspectives</h3>
    <p>Single attention mechanisms might miss important patterns by focusing on one type of relationship. Multi-head attention addresses this by running multiple attention operations in parallel, each with its own learned parameters.</p>

    <h4>The Multi-Head Mechanism</h4>
    <p><strong>Formula:</strong> $$\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, \\text{head}_2, \\ldots, \\text{head}_h)W^O$$</p>
    <p>where $\\text{head}_i = \\text{Attention}(QW^Q_i, KW^K_i, VW^V_i)$</p>

    <p><strong>Architecture details:</strong></p>
    <ul>
      <li><strong>Projection matrices:</strong> Each head has independent projection matrices $W^Q_i$, $W^K_i$, $W^V_i$ that project the input into different subspaces</li>
      <li><strong>Reduced dimension:</strong> If model dimension is $d_{\\text{model}}=512$ and we use $h=8$ heads, each head operates in $d_k = d_{\\text{model}}/h = 64$ dimensions</li>
      <li><strong>Parallel computation:</strong> All heads computed simultaneously, concatenated, then projected through $W^O$</li>
      <li><strong>Total parameters:</strong> Same as single full-dimensional attention: $h$ heads $\\times (3 \\times d_k \\times d_{\\text{model}}) + d_{\\text{model}}^2 \\approx 4d_{\\text{model}}^2$</li>
    </ul>

    <h4>Why Multiple Heads? Representation Diversity</h4>
    <p>Different heads learn to capture fundamentally different types of relationships:</p>

    <ul>
      <li><strong>Syntactic vs semantic:</strong> Some heads capture grammatical dependencies (subject-verb agreement, modifier relationships), while others capture semantic similarities and meaning</li>
      <li><strong>Local vs global:</strong> Some heads attend primarily to nearby tokens (capturing local context), while others attend to distant tokens (capturing long-range dependencies)</li>
      <li><strong>Position vs content:</strong> Some heads may focus on positional relationships (sequential patterns), while others focus on content similarity</li>
      <li><strong>Task-specific patterns:</strong> For translation, some heads align source-target words; for sentiment, some heads identify sentiment-bearing words</li>
    </ul>

    <p><strong>Empirical observations:</strong> Studies of BERT and GPT reveal that different heads specialize: some track coreference ("he" $\\to$ "John"), some track syntax trees, some focus on next-word prediction patterns. This specialization emerges through training without explicit supervision.</p>

    <h4>Benefits Over Single-Head</h4>
    <ul>
      <li><strong>Representational capacity:</strong> Multiple subspaces enable richer, more nuanced representations than single attention</li>
      <li><strong>Robustness:</strong> Ensemble effect—if one head fails on a pattern, others may capture it</li>
      <li><strong>Reduced overfitting:</strong> Splitting parameters across heads provides regularization</li>
      <li><strong>Interpretability:</strong> Different heads can be analyzed separately, revealing what patterns the model learned</li>
    </ul>

    <h3>Attention Variants: Adapting for Different Tasks</h3>

    <h4>Masked (Causal) Self-Attention</h4>
    <p>Prevents positions from attending to subsequent positions, maintaining autoregressive property:</p>
    <ul>
      <li><strong>Implementation:</strong> Set attention scores to $-\\infty$ for future positions before softmax</li>
      <li><strong>Use case:</strong> Language modeling and decoder stacks where causality must be preserved</li>
      <li><strong>Effect:</strong> Position i can only attend to positions 1...i, not i+1...n</li>
    </ul>

    <h4>Cross-Attention (Encoder-Decoder Attention)</h4>
    <p>Queries come from one sequence, keys and values from another:</p>
    <ul>
      <li><strong>Formula:</strong> $\\text{CrossAttention}(Q_{\\text{dec}}, K_{\\text{enc}}, V_{\\text{enc}})$</li>
      <li><strong>Use case:</strong> Translation, summarization—decoder attends to encoder representations</li>
      <li><strong>Information flow:</strong> Enables decoder to access full input information dynamically</li>
    </ul>

    <h4>Local (Windowed) Attention</h4>
    <p>Restricts attention to nearby positions within a fixed window:</p>
    <ul>
      <li><strong>Complexity reduction:</strong> $O(n \\cdot w \\cdot d)$ where $w$ is window size, vs $O(n^2d)$ for full attention</li>
      <li><strong>Use case:</strong> Very long sequences where quadratic complexity is prohibitive</li>
      <li><strong>Trade-off:</strong> Faster but may miss long-range dependencies</li>
    </ul>

    <h4>Sparse Attention</h4>
    <p>Attends to strategic subset of positions (e.g., strided, fixed patterns):</p>
    <ul>
      <li><strong>Patterns:</strong> Strided (every k-th position), fixed (predefined connections), learned (data-driven sparsity)</li>
      <li><strong>Use case:</strong> Scaling to very long sequences (4K+ tokens)</li>
      <li><strong>Examples:</strong> Sparse Transformers, Longformer, BigBird</li>
    </ul>

    <h3>Computational Considerations</h3>

    <h4>Memory Requirements</h4>
    <ul>
      <li><strong>Attention matrix:</strong> $O(n^2)$ per head, or $O(h \\cdot n^2)$ total for all heads</li>
      <li><strong>Bottleneck for long sequences:</strong> For $n=2048$, $h=16$: $16 \\times 2048^2 \\approx 67M$ values per layer</li>
      <li><strong>Memory-efficient implementations:</strong> Recompute attention during backward pass instead of storing</li>
    </ul>

    <h4>Computational Complexity</h4>
    <ul>
      <li><strong>Self-attention:</strong> $O(n^2d)$ dominated by $QK^T$ matrix multiplication</li>
      <li><strong>Feedforward:</strong> $O(nd^2)$ but with $d_{\\text{ff}}$ typically $4d$, so $O(4nd^2)$</li>
      <li><strong>Break-even point:</strong> When $n < d/4$, attention is cheaper; when $n > d/4$, feedforward dominates</li>
      <li><strong>Typical scenarios:</strong> For sentences ($n\\approx50$, $d\\approx512$), attention is manageable. For documents ($n\\approx2000$), attention becomes expensive</li>
    </ul>

    <h3>Implementation Insights</h3>

    <h4>Efficient Implementations</h4>
    <ul>
      <li><strong>Batched matrix multiplication:</strong> Compute all heads simultaneously through efficient batched operations</li>
      <li><strong>Fused kernels:</strong> Combine softmax with scaling and masking for speed</li>
      <li><strong>Flash Attention:</strong> Recent optimization that reduces memory from $O(n^2)$ to $O(n)$ for attention computation</li>
    </ul>

    <h4>Training Tips</h4>
    <ul>
      <li><strong>Dropout on attention weights:</strong> Typically 0.1, applied after softmax but before value multiplication</li>
      <li><strong>Attention weight visualization:</strong> Monitor to ensure heads learn diverse patterns, not redundant ones</li>
      <li><strong>Warmup schedule:</strong> Critical for training stability, especially with multiple heads</li>
    </ul>

    <h3>The Foundation of Modern Transformers</h3>
    <p>Self-attention and multi-head attention are not just components of Transformers—they define the architecture's fundamental nature. By enabling flexible, content-based information routing with direct connections between all positions, attention mechanisms solved the sequential bottleneck of RNNs while providing representational power that scaled to unprecedented model sizes. The success of BERT, GPT, and modern LLMs directly traces back to the elegant simplicity and remarkable effectiveness of this core mechanism.</p>
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
      question: 'Why do we scale the dot product by $\\sqrt{d_k}$ in attention?',
      answer: `Scaling the dot product by $\\sqrt{d_k}$ in attention is crucial for maintaining stable training dynamics and preventing the softmax function from producing overly peaked probability distributions that would harm gradient flow and model performance.

The fundamental issue arises from the variance of dot products increasing with dimensionality. When computing attention scores as $q \\cdot k$ where both $q$ and $k$ are $d_k$-dimensional vectors with elements drawn from distributions with variance $\\sigma^2$, the variance of their dot product grows proportionally to $d_k \\cdot \\sigma^2$. As $d_k$ increases (typical values are 64-128 per attention head), dot products can become very large in magnitude.

Large dot product values create problems in the softmax computation. When input values to softmax are large, the function produces extremely peaked distributions where one element approaches 1 and others approach 0. This "sharpening" effect reduces the effective gradient signal during backpropagation because the gradients of softmax become very small when the distribution is highly concentrated.

Mathematically, if attention scores before softmax are [10, 9, 8], the softmax outputs approximately [0.67, 0.24, 0.09]. But if scores are [100, 90, 80], softmax produces approximately [1.0, 0.0, 0.0], eliminating the attention to other positions entirely and reducing gradient flow.

The scaling factor $\\sqrt{d_k}$ is chosen to normalize the variance of dot products back to approximately 1, regardless of the key dimension. If $q$ and $k$ have unit variance, then $q \\cdot k / \\sqrt{d_k}$ also has approximately unit variance, keeping attention scores in a reasonable range that doesn't saturate the softmax function.

This scaling provides several benefits: (1) Training stability - prevents extreme attention distributions that can destabilize learning, (2) Better gradient flow - maintains meaningful gradients through the attention mechanism, (3) Dimension independence - allows using the same learning rates and optimization strategies across different model sizes, and (4) Attention diversity - enables the model to attend to multiple positions rather than focusing too sharply on single positions.

Empirical evidence supports this choice: experiments show that removing the scaling factor leads to slower convergence and worse final performance, particularly for larger models with higher dimensional keys. The $\\sqrt{d_k}$ scaling has become a standard component of attention mechanisms across all major Transformer implementations.`
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
      answer: `Self-attention has $O(n^2d)$ computational complexity where $n$ is the sequence length and $d$ is the model dimension, arising from the need to compute pairwise relationships between all positions in the sequence. Understanding this complexity is crucial for designing efficient Transformer architectures and understanding their scalability limitations.

The quadratic term $O(n^2)$ comes from computing attention scores between every pair of positions. For a sequence of length $n$, there are $n^2$ possible pairs, and each pair requires computing the similarity between their query and key vectors. This all-pairs computation is fundamental to self-attention's ability to model arbitrary long-range dependencies but creates a scalability bottleneck for long sequences.

The linear term $O(d)$ reflects the cost of computing each pairwise similarity. Query and key vectors are $d$-dimensional, so computing their dot product requires $d$ operations. Additionally, the subsequent aggregation of value vectors (also $d$-dimensional) based on attention weights contributes to the linear complexity in the model dimension.

Breaking down the computation steps: (1) Computing Q, K, V projections: $O(n \\cdot d^2)$ for matrix multiplications, (2) Computing attention scores $QK^T$: $O(n^2 \\cdot d)$ for all pairwise dot products, (3) Applying softmax: $O(n^2)$ for normalizing each row, (4) Computing weighted values: $O(n^2 \\cdot d)$ for aggregating value vectors. The dominant terms are the $O(n^2 \\cdot d)$ operations.

Memory complexity also scales as $O(n^2)$ for storing the attention matrix, which becomes prohibitive for very long sequences. For sequences of 10,000 tokens, the attention matrix alone requires storing 100 million values, before considering gradients needed for backpropagation.

This complexity creates practical limitations: most Transformer models are limited to sequences of 512-4096 tokens due to memory and computational constraints. Processing longer sequences requires specialized techniques or architectural modifications.

Comparison with RNNs reveals interesting trade-offs: RNNs have $O(n \\cdot d^2)$ complexity (linear in sequence length but quadratic in hidden dimension), while Transformers have $O(n^2 \\cdot d)$ complexity (quadratic in sequence length but linear in model dimension). For typical scenarios where $d >> n$, RNNs can be more efficient, but Transformers' parallelizability often compensates in practice.

Numerous solutions address this complexity: sparse attention patterns reduce the effective number of pairs computed, linear attention approximations achieve $O(n)$ complexity, hierarchical attention applies attention at multiple granularities, and sliding window attention limits attention to local neighborhoods. These innovations aim to preserve self-attention's modeling benefits while improving scalability.`
    },
    {
      question: 'How does masked attention differ from regular attention?',
      answer: `Masked attention is a variant of self-attention that prevents certain positions from attending to others by setting their attention scores to negative infinity before applying softmax, effectively making their attention weights zero. This mechanism is essential for maintaining causal constraints in autoregressive generation and implementing various attention patterns.

The key difference lies in the attention score computation. Regular self-attention computes scores between all position pairs: scores = $QK^T/\\sqrt{d_k}$, then applies softmax directly. Masked attention modifies this by adding a mask matrix: scores = $(QK^T + M)/\\sqrt{d_k}$, where $M$ contains 0 for allowed connections and $-\\infty$ (or a very large negative number) for forbidden connections.

Causal masking is the most common application, used in decoder self-attention to prevent positions from attending to future positions. The mask matrix $M$ is upper triangular with $-\\infty$ above the diagonal, ensuring position $i$ can only attend to positions $j$ where $j \\leq i$. This maintains the autoregressive property essential for text generation.

The masking mechanism preserves the mathematical properties of attention while enforcing structural constraints. When $-\\infty$ values are passed through softmax, they become 0, effectively removing those connections from the weighted sum. This allows flexible control over attention patterns without changing the fundamental attention computation.

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
      options: ['$O(n)$', '$O(n \\log n)$', '$O(n^2)$', '$O(n^3)$'],
      correctAnswer: 2,
      explanation: 'Self-attention has $O(n^2)$ complexity because each position must attend to every other position, resulting in $n^2$ attention scores.'
    },
    {
      id: 'attn2',
      question: 'Why do we use multiple heads in multi-head attention?',
      options: ['Reduce computation', 'Capture different aspects of relationships', 'Prevent overfitting', 'Speed up training'],
      correctAnswer: 1,
      explanation: 'Multiple heads allow the model to attend to different representation subspaces, capturing diverse semantic and syntactic relationships.'
    }
  ]
};
