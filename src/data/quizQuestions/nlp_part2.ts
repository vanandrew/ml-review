import { QuizQuestion } from '../../types';

// Seq2Seq - 20 questions
export const seq2seqQuestions: QuizQuestion[] = [
  {
    id: 's2s1',
    question: 'What is Seq2Seq (Sequence-to-Sequence)?',
    options: ['Single RNN', 'Encoder-decoder architecture that transforms one sequence to another', 'Feedforward network', 'CNN variant'],
    correctAnswer: 1,
    explanation: 'Seq2Seq uses an encoder to compress input sequence into context and a decoder to generate output sequence.'
  },
  {
    id: 's2s2',
    question: 'What are the two main components of Seq2Seq?',
    options: ['Input-output', 'Encoder (processes input) and Decoder (generates output)', 'Forward-backward', 'Conv-pool'],
    correctAnswer: 1,
    explanation: 'Encoder reads and encodes input sequence into fixed context vector; decoder generates output from this context.'
  },
  {
    id: 's2s3',
    question: 'What is the encoder in Seq2Seq?',
    options: ['Decoder', 'RNN/LSTM that processes input sequence and produces context vector', 'Fully connected layer', 'Attention layer'],
    correctAnswer: 1,
    explanation: 'The encoder (typically LSTM/GRU) reads the input sequence and compresses it into a fixed-size context/thought vector.'
  },
  {
    id: 's2s4',
    question: 'What is the context vector?',
    options: ['Input sequence', 'Fixed-size representation of input sequence produced by encoder', 'Random vector', 'Output sequence'],
    correctAnswer: 1,
    explanation: 'The context vector (often the final hidden state) encapsulates the entire input sequence information for the decoder.'
  },
  {
    id: 's2s5',
    question: 'What is the decoder in Seq2Seq?',
    options: ['Encoder', 'RNN/LSTM that generates output sequence from context vector', 'Input layer', 'Loss function'],
    correctAnswer: 1,
    explanation: 'The decoder takes the context vector and generates the target sequence one token at a time.'
  },
  {
    id: 's2s6',
    question: 'What is a limitation of basic Seq2Seq?',
    options: ['Too simple', 'Information bottleneck: entire input compressed into fixed-size vector', 'Too many parameters', 'Too fast'],
    correctAnswer: 1,
    explanation: 'Compressing long sequences into a single fixed vector loses information and makes it hard to capture long-range dependencies.'
  },
  {
    id: 's2s7',
    question: 'What are typical applications of Seq2Seq?',
    options: ['Image classification', 'Machine translation, text summarization, chatbots, question answering', 'Object detection', 'Clustering'],
    correctAnswer: 1,
    explanation: 'Seq2Seq excels at tasks transforming sequences: translation (English→French), summarization (long→short), dialogue (question→answer).'
  },
  {
    id: 's2s8',
    question: 'How does the decoder generate output during training?',
    options: ['Random generation', 'Using teacher forcing: ground truth as input at each step', 'No generation', 'Copies input'],
    correctAnswer: 1,
    explanation: 'During training, teacher forcing feeds the correct previous token as input, accelerating and stabilizing learning.'
  },
  {
    id: 's2s9',
    question: 'How does the decoder generate output during inference?',
    options: ['Teacher forcing', 'Uses its own previous predictions as input at each step', 'Random tokens', 'Copies encoder'],
    correctAnswer: 1,
    explanation: 'At inference, no ground truth is available, so the decoder feeds its own predictions back as input (autoregressive generation).'
  },
  {
    id: 's2s10',
    question: 'What is the <EOS> token?',
    options: ['Start token', 'End-of-sequence token signaling when to stop generation', 'Random token', 'Padding'],
    correctAnswer: 1,
    explanation: '<EOS> (End of Sequence) tells the model when the sequence is complete, crucial for variable-length output.'
  },
  {
    id: 's2s11',
    question: 'What is the <SOS> token?',
    options: ['End token', 'Start-of-sequence token to initialize decoder generation', 'Random token', 'Padding'],
    correctAnswer: 1,
    explanation: '<SOS> (Start of Sequence) is fed as the first input to the decoder to begin generation.'
  },
  {
    id: 's2s12',
    question: 'Can encoder and decoder have different architectures?',
    options: ['No, must match', 'Yes, can use different RNN types or even different architectures', 'Only same type', 'Only same size'],
    correctAnswer: 1,
    explanation: 'Encoder and decoder can differ: e.g., encoder could be bidirectional LSTM, decoder unidirectional GRU.'
  },
  {
    id: 's2s13',
    question: 'What is beam search in Seq2Seq decoding?',
    options: ['Random search', 'Keeps top-k sequences at each step instead of just the best', 'Greedy search', 'No search'],
    correctAnswer: 1,
    explanation: 'Beam search maintains multiple hypothesis sequences (beam width=k), exploring more possibilities than greedy decoding.'
  },
  {
    id: 's2s14',
    question: 'What is greedy decoding?',
    options: ['Best search', 'Always selecting the most probable token at each step', 'Beam search', 'Random selection'],
    correctAnswer: 1,
    explanation: 'Greedy decoding picks the single most likely token at each step. Fast but may miss globally optimal sequences.'
  },
  {
    id: 's2s15',
    question: 'Why is beam search better than greedy decoding?',
    options: ['Faster', 'Explores multiple paths, finding better overall sequences', 'Simpler', 'Uses less memory'],
    correctAnswer: 1,
    explanation: 'Beam search avoids local optima by keeping multiple candidates, often producing higher-quality outputs than greedy selection.'
  },
  {
    id: 's2s16',
    question: 'What problem does attention mechanism solve in Seq2Seq?',
    options: ['Speed', 'Allows decoder to focus on relevant parts of input, avoiding bottleneck', 'Reduces parameters', 'Simplifies architecture'],
    correctAnswer: 1,
    explanation: 'Attention lets the decoder access all encoder states, not just the final context vector, eliminating the information bottleneck.'
  },
  {
    id: 's2s17',
    question: 'Can Seq2Seq handle variable-length inputs and outputs?',
    options: ['No', 'Yes, naturally handles variable lengths', 'Only fixed length', 'Only same length'],
    correctAnswer: 1,
    explanation: 'Seq2Seq is designed for variable lengths: input and output can have different, dynamic lengths.'
  },
  {
    id: 's2s18',
    question: 'What is the typical loss function for Seq2Seq?',
    options: ['MSE', 'Cross-entropy loss over token predictions at each time step', 'Hinge loss', 'Dice loss'],
    correctAnswer: 1,
    explanation: 'Cross-entropy loss is computed for each generated token, comparing predicted distribution with the ground truth token.'
  },
  {
    id: 's2s19',
    question: 'What is exposure bias in Seq2Seq?',
    options: ['No bias', 'Mismatch between training (teacher forcing) and inference (own predictions)', 'Data bias', 'Model bias'],
    correctAnswer: 1,
    explanation: 'Model trained on ground truth but tests on its own outputs, leading to error accumulation at inference.'
  },
  {
    id: 's2s20',
    question: 'Have Transformers replaced Seq2Seq?',
    options: ['No change', 'Transformers are now the dominant Seq2Seq architecture, replacing RNN-based models', 'RNNs still dominate', 'No replacement'],
    correctAnswer: 1,
    explanation: 'Transformer-based Seq2Seq (encoder-decoder) models dominate due to parallelization and better long-range dependencies.'
  }
];

// Attention Mechanism - 25 questions
export const attentionQuestions: QuizQuestion[] = [
  {
    id: 'attn1',
    question: 'What is the attention mechanism?',
    options: ['Optimization method', 'Allows model to focus on relevant parts of input when generating output', 'Loss function', 'Activation function'],
    correctAnswer: 1,
    explanation: 'Attention dynamically weighs different parts of the input, allowing the model to "attend" to relevant information.'
  },
  {
    id: 'attn2',
    question: 'What problem does attention solve in Seq2Seq?',
    options: ['Speed', 'Information bottleneck of fixed-size context vector', 'Too many parameters', 'Overfitting'],
    correctAnswer: 1,
    explanation: 'Attention allows decoder to access all encoder states, not just a single context vector, eliminating the bottleneck.'
  },
  {
    id: 'attn3',
    question: 'What are the three components of attention?',
    options: ['Input-hidden-output', 'Query, Key, Value', 'Encoder-context-decoder', 'Conv-pool-fc'],
    correctAnswer: 1,
    explanation: 'Attention uses Query (what we\'re looking for), Key (what to match against), and Value (what to retrieve).'
  },
  {
    id: 'attn4',
    question: 'What is the Query in attention?',
    options: ['Encoder output', 'Current decoder state asking "what should I attend to?"', 'Input sequence', 'Loss value'],
    correctAnswer: 1,
    explanation: 'The query (typically decoder hidden state) represents what information the decoder is seeking at each step.'
  },
  {
    id: 'attn5',
    question: 'What are Keys in attention?',
    options: ['Decoder states', 'Encoder hidden states used to match against the query', 'Input tokens', 'Weights'],
    correctAnswer: 1,
    explanation: 'Keys (encoder hidden states) are compared with the query to determine which inputs are relevant.'
  },
  {
    id: 'attn6',
    question: 'What are Values in attention?',
    options: ['Queries', 'Encoder hidden states that are weighted and summed based on attention', 'Keys', 'Random vectors'],
    correctAnswer: 1,
    explanation: 'Values (often same as keys) are the actual information retrieved and combined based on attention weights.'
  },
  {
    id: 'attn7',
    question: 'How are attention weights calculated?',
    options: ['Random', 'Score function between query and keys, normalized with softmax', 'Fixed', 'Learned only'],
    correctAnswer: 1,
    explanation: 'Attention scores = similarity(Query, Keys), then softmax to get weights that sum to 1.'
  },
  {
    id: 'attn8',
    question: 'What is the attention score?',
    options: ['Final output', 'Similarity measure between query and each key', 'Loss value', 'Gradient'],
    correctAnswer: 1,
    explanation: 'Attention scores quantify how relevant each input position is to the current decoding step.'
  },
  {
    id: 'attn9',
    question: 'What is the context vector in attention?',
    options: ['Fixed encoder output', 'Weighted sum of values based on attention weights, dynamically computed', 'Decoder state', 'Input'],
    correctAnswer: 1,
    explanation: 'Unlike fixed context in basic Seq2Seq, attention computes a different context vector at each decoding step.'
  },
  {
    id: 'attn10',
    question: 'What are common attention scoring functions?',
    options: ['Only dot product', 'Dot product, additive (Bahdanau), multiplicative (Luong)', 'Random', 'Fixed'],
    correctAnswer: 1,
    explanation: 'Popular scores: dot product (Q·K), additive/concat (learned weights), scaled dot-product (Q·K/√d).'
  },
  {
    id: 'attn11',
    question: 'What is additive attention (Bahdanau)?',
    options: ['Simple addition', 'Uses feedforward network to compute alignment scores', 'Multiplication only', 'No computation'],
    correctAnswer: 1,
    explanation: 'Bahdanau attention: score = v^T tanh(W_q Q + W_k K), using learned weights and non-linearity.'
  },
  {
    id: 'attn12',
    question: 'What is multiplicative attention (Luong)?',
    options: ['Addition', 'Uses dot product (optionally with learned weights) for scores', 'Division', 'Random'],
    correctAnswer: 1,
    explanation: 'Luong attention: score = Q^T W K (general) or Q^T K (dot), simpler and faster than additive.'
  },
  {
    id: 'attn13',
    question: 'What is scaled dot-product attention?',
    options: ['Unscaled dot product', 'Dot product divided by √d_k to prevent large values', 'Random scaling', 'No scaling'],
    correctAnswer: 1,
    explanation: 'Scaled attention: (Q·K^T)/√d_k prevents dot products from growing large, keeping softmax gradients healthy.'
  },
  {
    id: 'attn14',
    question: 'Why scale by √d_k in scaled dot-product attention?',
    options: ['Random choice', 'Prevents dot products from growing large with dimension, keeping softmax well-behaved', 'Faster computation', 'No reason'],
    correctAnswer: 1,
    explanation: 'Without scaling, large dot products push softmax into saturated regions with tiny gradients, hurting training.'
  },
  {
    id: 'attn15',
    question: 'What is self-attention?',
    options: ['Attention between encoder-decoder', 'Attention within same sequence (Q, K, V from same source)', 'No attention', 'External attention'],
    correctAnswer: 1,
    explanation: 'Self-attention attends to different positions within the same sequence, allowing each position to interact with others.'
  },
  {
    id: 'attn16',
    question: 'What is cross-attention?',
    options: ['Self-attention', 'Attention between two different sequences (e.g., decoder attends to encoder)', 'No attention', 'Internal attention'],
    correctAnswer: 1,
    explanation: 'Cross-attention has Q from one sequence (decoder) and K, V from another (encoder), enabling interaction between sequences.'
  },
  {
    id: 'attn17',
    question: 'Can attention be visualized?',
    options: ['No', 'Yes, attention weights show what the model focuses on', 'Only for small models', 'Never useful'],
    correctAnswer: 1,
    explanation: 'Attention weight matrices can be visualized as heatmaps, showing which input tokens the model attends to for each output.'
  },
  {
    id: 'attn18',
    question: 'What is soft attention?',
    options: ['Hard selection', 'Weighted average over all positions (differentiable)', 'Binary selection', 'No attention'],
    correctAnswer: 1,
    explanation: 'Soft attention computes weighted average using softmax, allowing gradient flow (used in most attention mechanisms).'
  },
  {
    id: 'attn19',
    question: 'What is hard attention?',
    options: ['Soft attention', 'Selects single position (non-differentiable, needs reinforcement learning)', 'Weighted average', 'No selection'],
    correctAnswer: 1,
    explanation: 'Hard attention samples a single location, making it non-differentiable and requiring techniques like REINFORCE to train.'
  },
  {
    id: 'attn20',
    question: 'What is local attention?',
    options: ['Global attention', 'Attends to small window around a position, not entire sequence', 'No attention', 'Random attention'],
    correctAnswer: 1,
    explanation: 'Local attention restricts attention to a window, reducing computation while still capturing relevant context.'
  },
  {
    id: 'attn21',
    question: 'What is global attention?',
    options: ['Local attention', 'Attends to all positions in the source sequence', 'Window-based', 'No attention'],
    correctAnswer: 1,
    explanation: 'Global attention (standard approach) considers all encoder positions when computing context for each decoder step.'
  },
  {
    id: 'attn22',
    question: 'What is the computational complexity of attention?',
    options: ['O(n)', 'O(n²) for sequence length n', 'O(1)', 'O(log n)'],
    correctAnswer: 1,
    explanation: 'Computing attention between all pairs of positions requires O(n²) operations and memory for length-n sequences.'
  },
  {
    id: 'attn23',
    question: 'Is attention parallelizable?',
    options: ['No', 'Yes, all attention computations can be done in parallel', 'Only partially', 'Only for small sequences'],
    correctAnswer: 1,
    explanation: 'Unlike RNNs, attention operations can be fully parallelized across positions, enabling efficient GPU computation.'
  },
  {
    id: 'attn24',
    question: 'What is multi-head attention?',
    options: ['Single attention', 'Multiple attention mechanisms in parallel with different learned projections', 'Sequential attention', 'No attention'],
    correctAnswer: 1,
    explanation: 'Multi-head attention runs multiple attention functions in parallel, allowing the model to attend to different aspects simultaneously.'
  },
  {
    id: 'attn25',
    question: 'What was the impact of attention on NLP?',
    options: ['Minor improvement', 'Revolutionary: enabled Transformers and modern LLMs', 'No impact', 'Slowed progress'],
    correctAnswer: 1,
    explanation: 'Attention, especially self-attention, enabled the Transformer architecture, revolutionizing NLP and leading to BERT, GPT, and modern LLMs.'
  }
];

// Encoder-Decoder Architecture - 20 questions
export const encoderDecoderQuestions: QuizQuestion[] = [
  {
    id: 'ed1',
    question: 'What is the encoder-decoder architecture?',
    options: ['Single network', 'Two-part architecture: encoder processes input, decoder generates output', 'Feedforward only', 'CNN variant'],
    correctAnswer: 1,
    explanation: 'Encoder-decoder separates input processing (encoder) from output generation (decoder), common in Seq2Seq tasks.'
  },
  {
    id: 'ed2',
    question: 'What does the encoder do?',
    options: ['Generates output', 'Processes and encodes input into a representation', 'Computes loss', 'Optimizes weights'],
    correctAnswer: 1,
    explanation: 'The encoder reads the input sequence and transforms it into a rich representation (context) for the decoder.'
  },
  {
    id: 'ed3',
    question: 'What does the decoder do?',
    options: ['Processes input', 'Generates output sequence from the encoded representation', 'Encodes input', 'No function'],
    correctAnswer: 1,
    explanation: 'The decoder takes the encoder\'s representation and produces the target output sequence step by step.'
  },
  {
    id: 'ed4',
    question: 'Can encoder and decoder be different architectures?',
    options: ['No, must match', 'Yes, can be RNN, CNN, Transformer, or mixed', 'Only same type', 'Only RNN'],
    correctAnswer: 1,
    explanation: 'Encoders and decoders can mix architectures: e.g., CNN encoder + RNN decoder, or both Transformers.'
  },
  {
    id: 'ed5',
    question: 'What tasks use encoder-decoder architecture?',
    options: ['Classification only', 'Machine translation, summarization, image captioning, speech recognition', 'Regression only', 'Clustering'],
    correctAnswer: 1,
    explanation: 'Encoder-decoder is ideal for tasks transforming one representation to another: text→text, image→text, speech→text.'
  },
  {
    id: 'ed6',
    question: 'What is the bottleneck in basic encoder-decoder?',
    options: ['Decoder', 'Fixed-size vector between encoder and decoder', 'Encoder', 'No bottleneck'],
    correctAnswer: 1,
    explanation: 'Compressing entire input into a fixed-size vector loses information, especially for long sequences (solved by attention).'
  },
  {
    id: 'ed7',
    question: 'How does attention improve encoder-decoder?',
    options: ['No improvement', 'Allows decoder to access all encoder states, not just fixed vector', 'Makes it slower', 'Adds complexity only'],
    correctAnswer: 1,
    explanation: 'Attention lets decoder dynamically attend to relevant encoder states at each step, eliminating the bottleneck.'
  },
  {
    id: 'ed8',
    question: 'Can encoder-decoder be used for classification?',
    options: ['No', 'Yes, encoder processes input, decoder/classifier produces label', 'Only for generation', 'Never'],
    correctAnswer: 1,
    explanation: 'While designed for generation, encoder-only (like BERT) or encoder with classifier head works well for classification.'
  },
  {
    id: 'ed9',
    question: 'What is the Transformer encoder-decoder?',
    options: ['RNN-based', 'Both encoder and decoder use self-attention layers instead of RNNs', 'CNN-based', 'No attention'],
    correctAnswer: 1,
    explanation: 'Transformer architecture uses self-attention in both encoder and decoder, enabling parallelization and better long-range dependencies.'
  },
  {
    id: 'ed10',
    question: 'What is teacher forcing in encoder-decoder training?',
    options: ['Testing method', 'Feeding ground truth tokens to decoder during training', 'Encoder technique', 'Loss function'],
    correctAnswer: 1,
    explanation: 'Teacher forcing uses correct previous tokens as decoder input during training, accelerating convergence.'
  },
  {
    id: 'ed11',
    question: 'What is autoregressive decoding?',
    options: ['Parallel decoding', 'Generating one token at a time, using previous outputs as input', 'Batch generation', 'Random generation'],
    correctAnswer: 1,
    explanation: 'Autoregressive: decoder generates sequentially, feeding its own predictions back as input for the next token.'
  },
  {
    id: 'ed12',
    question: 'Can encoder-decoder handle variable-length sequences?',
    options: ['No', 'Yes, both input and output can have variable lengths', 'Only fixed', 'Only encoder'],
    correctAnswer: 1,
    explanation: 'Encoder-decoder naturally handles variable-length inputs and outputs, processing until <EOS> token.'
  },
  {
    id: 'ed13',
    question: 'What is BART?',
    options: ['Encoder-only', 'Transformer encoder-decoder trained with denoising objectives', 'Decoder-only', 'CNN'],
    correctAnswer: 1,
    explanation: 'BART is a Transformer encoder-decoder pre-trained on text denoising, effective for generation and comprehension tasks.'
  },
  {
    id: 'ed14',
    question: 'What is T5?',
    options: ['Encoder-only', 'Text-to-Text Transfer Transformer: encoder-decoder treating all tasks as text generation', 'Decoder-only', 'RNN'],
    correctAnswer: 1,
    explanation: 'T5 frames all NLP tasks (classification, translation, QA) as text-to-text, using unified encoder-decoder architecture.'
  },
  {
    id: 'ed15',
    question: 'What is the advantage of encoder-only models (like BERT)?',
    options: ['Better for generation', 'Bidirectional context, great for understanding tasks', 'Faster generation', 'Smaller'],
    correctAnswer: 1,
    explanation: 'Encoder-only models see full bidirectional context, excelling at classification, NER, QA but can\'t generate text.'
  },
  {
    id: 'ed16',
    question: 'What is the advantage of decoder-only models (like GPT)?',
    options: ['Better understanding', 'Simpler architecture, powerful for text generation', 'Bidirectional', 'Smaller'],
    correctAnswer: 1,
    explanation: 'Decoder-only models are simpler and excel at generation, trained autoregressively to predict next tokens.'
  },
  {
    id: 'ed17',
    question: 'When to use encoder-decoder vs decoder-only?',
    options: ['Always encoder-decoder', 'Encoder-decoder for transformation tasks; decoder-only for open-ended generation', 'Always decoder-only', 'No difference'],
    correctAnswer: 1,
    explanation: 'Encoder-decoder suits translation/summarization; decoder-only works for both but is simpler and scales well for generation.'
  },
  {
    id: 'ed18',
    question: 'What is the typical pre-training objective for encoder-decoder?',
    options: ['Classification', 'Denoising autoencoding or span corruption', 'Next token prediction', 'Contrastive learning'],
    correctAnswer: 1,
    explanation: 'Models like BART use denoising (reconstruct corrupted text); T5 uses span corruption (fill in masked spans).'
  },
  {
    id: 'ed19',
    question: 'Can encoder-decoder be fine-tuned for specific tasks?',
    options: ['No', 'Yes, commonly fine-tuned on task-specific data', 'Only pre-training', 'Never beneficial'],
    correctAnswer: 1,
    explanation: 'Pre-trained encoder-decoders (T5, BART) are fine-tuned on translation, summarization, QA, etc., for best performance.'
  },
  {
    id: 'ed20',
    question: 'What is the computational cost of encoder-decoder vs decoder-only?',
    options: ['Same', 'Encoder-decoder is more expensive (runs both encoder and decoder)', 'Decoder-only is slower', 'Encoder-decoder is faster'],
    correctAnswer: 1,
    explanation: 'Encoder-decoder processes input twice (encoder + decoder attention), while decoder-only has a single pass.'
  }
];
