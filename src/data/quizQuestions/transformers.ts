import { QuizQuestion } from '../../types';

// Transformer Architecture - 25 questions
export const transformerArchitectureQuestions: QuizQuestion[] = [
  {
    id: 'tran1',
    question: 'What paper introduced the Transformer?',
    options: ['ImageNet paper', '"Attention is All You Need" (2017)', 'AlexNet paper', 'BERT paper'],
    correctAnswer: 1,
    explanation: 'The 2017 paper "Attention is All You Need" by Vaswani et al. introduced the Transformer architecture.'
  },
  {
    id: 'tran2',
    question: 'What is the key innovation of Transformers?',
    options: ['Using CNNs', 'Relying entirely on self-attention, no recurrence or convolution', 'Using RNNs', 'Using fully connected layers'],
    correctAnswer: 1,
    explanation: 'Transformers replace recurrent layers with self-attention mechanisms, enabling parallelization and better long-range dependencies.'
  },
  {
    id: 'tran3',
    question: 'What are the two main parts of the original Transformer?',
    options: ['CNN and RNN', 'Encoder and Decoder stacks', 'Input and Output', 'Attention and Pooling'],
    correctAnswer: 1,
    explanation: 'The original Transformer has an encoder stack (processes input) and decoder stack (generates output).'
  },
  {
    id: 'tran4',
    question: 'How many layers does the original Transformer have?',
    options: ['3', '6 encoder layers and 6 decoder layers', '12', '24'],
    correctAnswer: 1,
    explanation: 'The base Transformer model has 6 identical encoder layers and 6 identical decoder layers.'
  },
  {
    id: 'tran5',
    question: 'What are the sublayers in a Transformer encoder layer?',
    options: ['One sublayer', 'Multi-head self-attention and feedforward network', 'Only attention', 'Only feedforward'],
    correctAnswer: 1,
    explanation: 'Each encoder layer has: (1) multi-head self-attention, (2) position-wise feedforward network, both with residual connections and layer norm.'
  },
  {
    id: 'tran6',
    question: 'What are the sublayers in a Transformer decoder layer?',
    options: ['Two sublayers', 'Masked self-attention, cross-attention to encoder, feedforward', 'Only attention', 'Only feedforward'],
    correctAnswer: 1,
    explanation: 'Decoder layers have: (1) masked self-attention, (2) cross-attention to encoder outputs, (3) feedforward, all with residuals and norms.'
  },
  {
    id: 'tran7',
    question: 'Why is self-attention "masked" in the decoder?',
    options: ['Random masking', 'Prevents attending to future positions (maintains autoregressive property)', 'Removes attention', 'Adds noise'],
    correctAnswer: 1,
    explanation: 'Masking prevents the decoder from "cheating" by looking at future tokens during training, preserving the generation order.'
  },
  {
    id: 'tran8',
    question: 'What is the feedforward network in Transformers?',
    options: ['Single linear layer', 'Two linear layers with ReLU: FFN(x) = max(0, xW₁+b₁)W₂+b₂', 'No feedforward', 'Attention layer'],
    correctAnswer: 1,
    explanation: 'The position-wise feedforward network applies two linear transformations with ReLU, independently to each position.'
  },
  {
    id: 'tran9',
    question: 'What is the hidden dimension of the feedforward network in the base Transformer?',
    options: ['512', '2048 (4× the model dimension)', '128', '4096'],
    correctAnswer: 1,
    explanation: 'The feedforward network expands from model dimension 512 to 2048, then projects back to 512.'
  },
  {
    id: 'tran10',
    question: 'What are residual connections in Transformers?',
    options: ['No connections', 'Skip connections adding input to sublayer output: x + Sublayer(x)', 'Only in RNNs', 'Random connections'],
    correctAnswer: 1,
    explanation: 'Residual connections (x + Sublayer(x)) help gradient flow and enable training of deep networks.'
  },
  {
    id: 'tran11',
    question: 'What normalization is used in Transformers?',
    options: ['Batch normalization', 'Layer normalization', 'Instance normalization', 'No normalization'],
    correctAnswer: 1,
    explanation: 'Layer normalization is applied after each sublayer (post-norm) or before (pre-norm) for stable training.'
  },
  {
    id: 'tran12',
    question: 'What is the model dimension (d_model) in the base Transformer?',
    options: ['256', '512', '1024', '2048'],
    correctAnswer: 1,
    explanation: 'The base Transformer uses d_model = 512 for all layer inputs and outputs.'
  },
  {
    id: 'tran13',
    question: 'How many attention heads in the base Transformer?',
    options: ['4', '8', '16', '32'],
    correctAnswer: 1,
    explanation: 'The base Transformer uses 8 attention heads, each with dimension d_k = d_v = 64.'
  },
  {
    id: 'tran14',
    question: 'What is the advantage of Transformers over RNNs?',
    options: ['Fewer parameters', 'Parallelizable computation, better long-range dependencies', 'Simpler', 'Sequential only'],
    correctAnswer: 1,
    explanation: 'Transformers process all positions in parallel and use direct attention paths, unlike sequential RNNs.'
  },
  {
    id: 'tran15',
    question: 'What is the computational complexity of self-attention?',
    options: ['O(n)', 'O(n²d) for sequence length n and dimension d', 'O(n³)', 'O(1)'],
    correctAnswer: 1,
    explanation: 'Self-attention computes attention between all pairs (n²) and projects to dimension d, giving O(n²d) complexity.'
  },
  {
    id: 'tran16',
    question: 'What is a limitation of standard Transformers for long sequences?',
    options: ['No limitation', 'Quadratic memory and compute with sequence length', 'Too fast', 'Too accurate'],
    correctAnswer: 1,
    explanation: 'O(n²) complexity makes standard Transformers expensive for long sequences, motivating efficient variants.'
  },
  {
    id: 'tran17',
    question: 'What is the typical activation function in Transformer feedforward layers?',
    options: ['Sigmoid', 'ReLU or GELU', 'Tanh', 'Linear'],
    correctAnswer: 1,
    explanation: 'Original used ReLU; modern Transformers often use GELU (Gaussian Error Linear Unit) for smoother gradients.'
  },
  {
    id: 'tran18',
    question: 'How does the Transformer handle variable-length sequences?',
    options: ['Cannot handle', 'Processes sequences of any length, uses masking for padding', 'Fixed length only', 'Truncates all'],
    correctAnswer: 1,
    explanation: 'Transformers handle variable lengths naturally; padding tokens are masked so they don\'t affect attention.'
  },
  {
    id: 'tran19',
    question: 'What is cross-attention in the Transformer decoder?',
    options: ['Self-attention', 'Attention where Q comes from decoder, K and V from encoder', 'No attention', 'Random attention'],
    correctAnswer: 1,
    explanation: 'Cross-attention allows decoder to attend to encoder outputs, enabling the decoder to use input information.'
  },
  {
    id: 'tran20',
    question: 'Why was the Transformer a breakthrough?',
    options: ['Smallest model', 'Enabled large-scale pre-training and scaling (BERT, GPT lineage)', 'Slowest model', 'No impact'],
    correctAnswer: 1,
    explanation: 'Transformers\' parallelization enabled training on massive datasets and scaling to billions of parameters, revolutionizing NLP.'
  },
  {
    id: 'tran21',
    question: 'What is the Transformer-XL?',
    options: ['Smaller Transformer', 'Extends context with segment-level recurrence and relative positional encoding', 'Same as Transformer', 'Image model'],
    correctAnswer: 1,
    explanation: 'Transformer-XL introduces recurrence across segments and relative positions to handle longer contexts efficiently.'
  },
  {
    id: 'tran22',
    question: 'What are efficient Transformers?',
    options: ['Standard Transformers', 'Variants reducing O(n²) complexity (Linformer, Performer, Longformer)', 'Slower Transformers', 'No efficiency'],
    correctAnswer: 1,
    explanation: 'Efficient Transformers use techniques like sparse attention, low-rank approximations to reduce quadratic complexity.'
  },
  {
    id: 'tran23',
    question: 'What is the Vision Transformer (ViT)?',
    options: ['Text model', 'Applies Transformer to images by treating patches as tokens', 'RNN for images', 'Standard CNN'],
    correctAnswer: 1,
    explanation: 'ViT splits images into patches, linearly embeds them, and processes with standard Transformer encoder.'
  },
  {
    id: 'tran24',
    question: 'Can Transformers be used beyond NLP?',
    options: ['No, only NLP', 'Yes: vision (ViT), speech, protein folding, multi-modal tasks', 'Only text', 'Only translation'],
    correctAnswer: 1,
    explanation: 'Transformers are universal: used in computer vision, speech recognition, AlphaFold, DALL-E, and more.'
  },
  {
    id: 'tran25',
    question: 'What is the typical output of a Transformer encoder?',
    options: ['Single vector', 'Sequence of contextualized representations for each input position', 'Class label', 'Image'],
    correctAnswer: 1,
    explanation: 'Encoder outputs a sequence of vectors, one per input token, each containing rich contextual information from the entire sequence.'
  }
];

// Self-Attention - 25 questions
export const selfAttentionQuestions: QuizQuestion[] = [
  {
    id: 'sa1',
    question: 'What is self-attention?',
    options: ['Attention between sequences', 'Attention mechanism within a single sequence', 'No attention', 'Cross-attention'],
    correctAnswer: 1,
    explanation: 'Self-attention allows each position in a sequence to attend to all positions in the same sequence.'
  },
  {
    id: 'sa2',
    question: 'What are Q, K, V in self-attention?',
    options: ['Random vectors', 'Query, Key, Value - all derived from the same input sequence', 'Different sequences', 'Fixed vectors'],
    correctAnswer: 1,
    explanation: 'In self-attention, Q, K, and V are all linear projections of the same input: Q=XW_Q, K=XW_K, V=XW_V.'
  },
  {
    id: 'sa3',
    question: 'What is the self-attention formula?',
    options: ['QK', 'Attention(Q,K,V) = softmax(QK^T/√d_k)V', 'QV', 'KV'],
    correctAnswer: 1,
    explanation: 'Scaled dot-product attention: compute similarity scores QK^T, scale by √d_k, apply softmax, weight the values V.'
  },
  {
    id: 'sa4',
    question: 'Why compute QK^T in self-attention?',
    options: ['Random operation', 'Measures similarity/relevance between all pairs of positions', 'Reduces dimension', 'No reason'],
    correctAnswer: 1,
    explanation: 'QK^T produces an n×n matrix of attention scores, where each element measures how much position i should attend to position j.'
  },
  {
    id: 'sa5',
    question: 'What does softmax do in self-attention?',
    options: ['Adds values', 'Normalizes attention scores into probability distribution', 'Removes values', 'Multiplies values'],
    correctAnswer: 1,
    explanation: 'Softmax converts attention scores to probabilities that sum to 1, determining how much to weight each value.'
  },
  {
    id: 'sa6',
    question: 'What is the output of self-attention?',
    options: ['Single vector', 'Weighted sum of values for each position based on attention', 'Attention scores only', 'Random output'],
    correctAnswer: 1,
    explanation: 'For each position, self-attention outputs a weighted combination of all value vectors, where weights come from attention scores.'
  },
  {
    id: 'sa7',
    question: 'Why scale by √d_k?',
    options: ['Random choice', 'Prevents large dot products that push softmax into saturated regions', 'Speeds computation', 'No reason'],
    correctAnswer: 1,
    explanation: 'Without scaling, dot products can be large (variance scales with d_k), causing tiny softmax gradients and slow learning.'
  },
  {
    id: 'sa8',
    question: 'What is multi-head self-attention?',
    options: ['Single attention', 'Multiple self-attention operations in parallel with different learned projections', 'Sequential attention', 'No attention'],
    correctAnswer: 1,
    explanation: 'Multi-head attention runs h parallel attention heads, each learning different aspects, then concatenates and projects the results.'
  },
  {
    id: 'sa9',
    question: 'Why use multiple attention heads?',
    options: ['Slower computation', 'Allows attending to different representation subspaces simultaneously', 'No benefit', 'Reduces parameters'],
    correctAnswer: 1,
    explanation: 'Different heads can focus on different types of relationships (syntax, semantics, long-range, short-range).'
  },
  {
    id: 'sa10',
    question: 'How are dimensions split in multi-head attention?',
    options: ['Not split', 'Model dimension d_model split across h heads: each head has dimension d_k = d_model/h', 'All heads same size', 'Random split'],
    correctAnswer: 1,
    explanation: 'With h=8 and d_model=512, each head operates on d_k=64 dimensions, keeping total parameters similar to single-head.'
  },
  {
    id: 'sa11',
    question: 'What is the computational complexity of self-attention?',
    options: ['O(n)', 'O(n²d) for sequence length n and dimension d', 'O(nd)', 'O(1)'],
    correctAnswer: 1,
    explanation: 'Computing attention for all n² pairs with dimension d gives O(n²d) time and memory complexity.'
  },
  {
    id: 'sa12',
    question: 'Can self-attention process sequences in parallel?',
    options: ['No, sequential only', 'Yes, all positions computed simultaneously', 'Only small sequences', 'Only with RNN'],
    correctAnswer: 1,
    explanation: 'Unlike RNNs, self-attention computes all positions at once via matrix operations, enabling massive parallelization.'
  },
  {
    id: 'sa13',
    question: 'What is masked self-attention?',
    options: ['No masking', 'Prevents attending to future positions by masking them in softmax', 'Random masking', 'Removes all attention'],
    correctAnswer: 1,
    explanation: 'Masked attention sets future positions to -∞ before softmax, ensuring autoregressive generation (used in decoders).'
  },
  {
    id: 'sa14',
    question: 'Why mask future positions in decoder self-attention?',
    options: ['Random choice', 'Maintains causality: each position can only attend to previous positions', 'Speeds up', 'No reason'],
    correctAnswer: 1,
    explanation: 'Masking ensures the model can\'t "cheat" by looking ahead during training, matching inference where future tokens are unknown.'
  },
  {
    id: 'sa15',
    question: 'What information does each head capture?',
    options: ['Same information', 'Different heads learn different patterns (syntax, semantics, dependencies)', 'Random patterns', 'No patterns'],
    correctAnswer: 1,
    explanation: 'Analysis shows different heads specialize: some focus on syntax, others on semantic relationships or positional patterns.'
  },
  {
    id: 'sa16',
    question: 'How does self-attention handle long-range dependencies?',
    options: ['Cannot handle', 'Direct connections between all positions enable efficient modeling', 'Only short-range', 'Requires RNN'],
    correctAnswer: 1,
    explanation: 'Every position can directly attend to any other position, unlike RNNs where information must propagate sequentially.'
  },
  {
    id: 'sa17',
    question: 'What is the role of the final linear projection in multi-head attention?',
    options: ['No role', 'Combines information from all heads into single representation', 'Splits heads', 'Removes attention'],
    correctAnswer: 1,
    explanation: 'After concatenating head outputs, a final linear layer W_O projects back to model dimension, integrating all heads.'
  },
  {
    id: 'sa18',
    question: 'Can self-attention be visualized?',
    options: ['No', 'Yes, attention matrices show which tokens attend to which', 'Only for small models', 'Never useful'],
    correctAnswer: 1,
    explanation: 'Attention weight visualizations reveal learned patterns like syntactic dependencies, coreferences, and semantic relationships.'
  },
  {
    id: 'sa19',
    question: 'What is the attention matrix size?',
    options: ['n×d', 'n×n where n is sequence length', 'd×d', '1×n'],
    correctAnswer: 1,
    explanation: 'The attention matrix is n×n, showing attention weights from each position to every other position.'
  },
  {
    id: 'sa20',
    question: 'Does self-attention have position information?',
    options: ['Yes, inherently', 'No, requires positional encoding to be added', 'Only for short sequences', 'Always positional'],
    correctAnswer: 1,
    explanation: 'Self-attention is position-invariant (permutation equivariant), requiring explicit positional encodings to know token order.'
  },
  {
    id: 'sa21',
    question: 'What are the learnable parameters in self-attention?',
    options: ['None', 'W_Q, W_K, W_V projection matrices (and W_O for multi-head)', 'Only attention weights', 'Fixed parameters'],
    correctAnswer: 1,
    explanation: 'Self-attention learns projection matrices W_Q, W_K, W_V (and W_O in multi-head) but attention weights are computed, not learned.'
  },
  {
    id: 'sa22',
    question: 'How does self-attention compare to convolution?',
    options: ['Same operation', 'Self-attention has global receptive field; convolution is local', 'Convolution is better', 'No comparison'],
    correctAnswer: 1,
    explanation: 'Self-attention sees entire sequence; convolution sees local window. Both can be useful in different contexts.'
  },
  {
    id: 'sa23',
    question: 'What is the memory requirement of self-attention?',
    options: ['O(n)', 'O(n²) for storing attention matrix', 'O(1)', 'O(nd)'],
    correctAnswer: 1,
    explanation: 'Storing the n×n attention matrix requires O(n²) memory, problematic for very long sequences.'
  },
  {
    id: 'sa24',
    question: 'What modifications reduce self-attention complexity?',
    options: ['No modifications exist', 'Sparse attention, linear attention, local windows', 'Only increase complexity', 'Remove attention'],
    correctAnswer: 1,
    explanation: 'Techniques like sparse patterns (Longformer), low-rank (Linformer), or local windows reduce O(n²) to O(n log n) or O(n).'
  },
  {
    id: 'sa25',
    question: 'Is self-attention differentiable?',
    options: ['No', 'Yes, fully differentiable, enabling end-to-end gradient-based training', 'Only partially', 'No gradients'],
    correctAnswer: 1,
    explanation: 'All operations (matrix multiply, softmax) are differentiable, allowing backpropagation through self-attention layers.'
  }
];
