import { QuizQuestion } from '../../types';

// Positional Encoding - 20 questions
export const positionalEncodingQuestions: QuizQuestion[] = [
  {
    id: 'pe1',
    question: 'Why do Transformers need positional encoding?',
    options: ['They don\'t', 'Self-attention is position-invariant; PE provides order information', 'For speed', 'For regularization'],
    correctAnswer: 1,
    explanation: 'Self-attention treats input as an unordered set. Positional encoding injects information about token positions.'
  },
  {
    id: 'pe2',
    question: 'When is positional encoding added?',
    options: ['At output', 'Added to input embeddings before the first layer', 'Between layers', 'Never added'],
    correctAnswer: 1,
    explanation: 'Positional encodings are added to input embeddings at the very beginning, then processed through all layers.'
  },
  {
    id: 'pe3',
    question: 'What type of positional encoding did the original Transformer use?',
    options: ['Learned', 'Sinusoidal (sine and cosine functions)', 'Random', 'One-hot'],
    correctAnswer: 1,
    explanation: 'The original Transformer used fixed sinusoidal positional encodings based on sine and cosine with different frequencies.'
  },
  {
    id: 'pe4',
    question: 'What is the sinusoidal positional encoding formula?',
    options: ['PE = pos', 'PE(pos,2i) = sin(pos/10000^(2i/d)), PE(pos,2i+1) = cos(pos/10000^(2i/d))', 'PE = i', 'PE = random'],
    correctAnswer: 1,
    explanation: 'Each dimension uses sine (even) or cosine (odd) with wavelengths forming geometric progression from 2π to 10000·2π.'
  },
  {
    id: 'pe5',
    question: 'Why use sinusoidal functions for positional encoding?',
    options: ['Random choice', 'Allows model to attend to relative positions; may generalize to longer sequences', 'Faster computation', 'Simpler'],
    correctAnswer: 1,
    explanation: 'Sinusoidal functions create fixed relationships between positions, potentially helping with relative position and extrapolation.'
  },
  {
    id: 'pe6',
    question: 'What are learned positional embeddings?',
    options: ['Fixed encodings', 'Positional embeddings learned during training like word embeddings', 'Sinusoidal only', 'No embeddings'],
    correctAnswer: 1,
    explanation: 'Learned embeddings: each position has its own learned vector, updated during training (used in BERT, GPT).'
  },
  {
    id: 'pe7',
    question: 'What is an advantage of learned positional embeddings?',
    options: ['No advantage', 'Can adapt to task-specific positional patterns', 'Always better', 'Faster'],
    correctAnswer: 1,
    explanation: 'Learned embeddings can capture task-specific position importance, often performing as well or better than sinusoidal.'
  },
  {
    id: 'pe8',
    question: 'What is a limitation of learned positional embeddings?',
    options: ['No limitation', 'Cannot handle sequences longer than training maximum', 'Too slow', 'Too complex'],
    correctAnswer: 1,
    explanation: 'Learned embeddings only exist for positions seen during training; longer sequences require interpolation or extrapolation.'
  },
  {
    id: 'pe9',
    question: 'What is relative positional encoding?',
    options: ['Absolute positions', 'Encoding relative distances between positions rather than absolute positions', 'No encoding', 'Random encoding'],
    correctAnswer: 1,
    explanation: 'Relative encoding models distance between positions (e.g., "3 positions apart") rather than absolute indices.'
  },
  {
    id: 'pe10',
    question: 'What models use relative positional encoding?',
    options: ['None', 'Transformer-XL, T5, DeBERTa', 'Only BERT', 'Only GPT'],
    correctAnswer: 1,
    explanation: 'Transformer-XL introduced relative positional encoding; T5 uses relative position biases; DeBERTa uses disentangled attention with relative positions.'
  },
  {
    id: 'pe11',
    question: 'How does T5 handle positional information?',
    options: ['Sinusoidal', 'Relative position bias added to attention scores', 'Learned absolute', 'No positions'],
    correctAnswer: 1,
    explanation: 'T5 uses simplified relative position biases: shared bias values based on bucket distance between positions.'
  },
  {
    id: 'pe12',
    question: 'What is ALiBi (Attention with Linear Biases)?',
    options: ['Learned embeddings', 'Adds linear bias proportional to distance directly to attention scores', 'Sinusoidal encoding', 'No bias'],
    correctAnswer: 1,
    explanation: 'ALiBi penalizes attention to distant tokens linearly, eliminating explicit positional embeddings while enabling length extrapolation.'
  },
  {
    id: 'pe13',
    question: 'Can positional encoding be extended to 2D (images)?',
    options: ['No, 1D only', 'Yes, using 2D positional encodings for height and width', 'Never needed', 'Only for text'],
    correctAnswer: 1,
    explanation: 'Vision Transformers use 2D positional encodings, either learned or based on patch positions in the image grid.'
  },
  {
    id: 'pe14',
    question: 'Do all Transformer variants use positional encoding?',
    options: ['Yes, always required', 'Most do, but some like ALiBi use alternative approaches', 'None use it', 'Only old models'],
    correctAnswer: 1,
    explanation: 'While most add positional embeddings, alternatives like ALiBi, RoPE modify attention mechanisms directly for position info.'
  },
  {
    id: 'pe15',
    question: 'What is RoPE (Rotary Position Embedding)?',
    options: ['Standard embedding', 'Rotates Q and K by angle proportional to position', 'No rotation', 'Random rotation'],
    correctAnswer: 1,
    explanation: 'RoPE applies rotation matrices to Q and K based on position, naturally encoding relative positions in dot products (used in GPT-Neo, PaLM).'
  },
  {
    id: 'pe16',
    question: 'What is the maximum sequence length issue?',
    options: ['No issue', 'Models trained with fixed max length may not generalize to longer sequences', 'Always generalizes', 'No maximum'],
    correctAnswer: 1,
    explanation: 'Absolute learned embeddings and some encoding schemes don\'t extrapolate well beyond training sequence lengths.'
  },
  {
    id: 'pe17',
    question: 'How do you handle longer sequences than training maximum?',
    options: ['Cannot handle', 'Use relative positions, interpolation, or position-independent methods', 'Truncate always', 'Retrain'],
    correctAnswer: 1,
    explanation: 'Relative encodings, interpolation (BERT), ALiBi, or re-training with longer sequences enable handling increased lengths.'
  },
  {
    id: 'pe18',
    question: 'Are positional encodings updated during training?',
    options: ['Sinusoidal are fixed', 'Sinusoidal are fixed; learned embeddings are updated', 'All are fixed', 'All are updated'],
    correctAnswer: 1,
    explanation: 'Sinusoidal encodings are fixed (not parameters); learned embeddings have gradients and update during training.'
  },
  {
    id: 'pe19',
    question: 'What dimension do positional encodings have?',
    options: ['Smaller than embeddings', 'Same as model dimension (d_model), added element-wise to embeddings', 'Larger', 'Variable'],
    correctAnswer: 1,
    explanation: 'Positional encodings match embedding dimension (e.g., 512, 768) to enable element-wise addition.'
  },
  {
    id: 'pe20',
    question: 'Can you visualize positional encodings?',
    options: ['No', 'Yes, as heatmaps showing patterns across positions and dimensions', 'Only sinusoidal', 'Never useful'],
    correctAnswer: 1,
    explanation: 'Visualizations reveal structure: sinusoidal show wave patterns; learned embeddings show position-specific patterns.'
  }
];

// BERT - 25 questions
export const bertQuestions: QuizQuestion[] = [
  {
    id: 'bert1',
    question: 'What does BERT stand for?',
    options: ['Binary Encoder', 'Bidirectional Encoder Representations from Transformers', 'Basic Embedding', 'Bayesian Ensemble'],
    correctAnswer: 1,
    explanation: 'BERT (2018) by Google uses bidirectional Transformers to pre-train deep contextualized representations.'
  },
  {
    id: 'bert2',
    question: 'What is the key innovation of BERT?',
    options: ['Unidirectional', 'Bidirectional pre-training: sees both left and right context simultaneously', 'No context', 'Only left context'],
    correctAnswer: 1,
    explanation: 'Unlike GPT (left-to-right), BERT uses bidirectional attention to condition on both past and future tokens.'
  },
  {
    id: 'bert3',
    question: 'What architecture does BERT use?',
    options: ['Decoder only', 'Transformer encoder only', 'RNN', 'CNN'],
    correctAnswer: 1,
    explanation: 'BERT uses only the Transformer encoder stack (no decoder), making it great for understanding but not generation.'
  },
  {
    id: 'bert4',
    question: 'What are the two pre-training objectives of BERT?',
    options: ['Classification only', 'Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)', 'Translation', 'Summarization'],
    correctAnswer: 1,
    explanation: 'BERT is pre-trained on MLM (predict masked tokens) and NSP (predict if sentences are consecutive).'
  },
  {
    id: 'bert5',
    question: 'What is Masked Language Modeling (MLM)?',
    options: ['Predict next word', 'Randomly mask tokens and predict them using bidirectional context', 'Remove all masks', 'No masking'],
    correctAnswer: 1,
    explanation: 'MLM masks 15% of tokens and trains BERT to predict them using both left and right context.'
  },
  {
    id: 'bert6',
    question: 'What percentage of tokens are masked in MLM?',
    options: ['5%', '15%', '50%', '100%'],
    correctAnswer: 1,
    explanation: 'BERT masks 15% of tokens: 80% replaced with [MASK], 10% random token, 10% unchanged.'
  },
  {
    id: 'bert7',
    question: 'Why not always replace masked tokens with [MASK]?',
    options: ['Random choice', 'To reduce train-test mismatch ([MASK] only appears in training)', 'Faster training', 'No reason'],
    correctAnswer: 1,
    explanation: 'Since [MASK] doesn\'t appear during fine-tuning, sometimes using random/unchanged tokens reduces the mismatch.'
  },
  {
    id: 'bert8',
    question: 'What is Next Sentence Prediction (NSP)?',
    options: ['Predict last word', 'Binary classification: are two sentences consecutive in corpus?', 'Generate sentences', 'No prediction'],
    correctAnswer: 1,
    explanation: 'NSP trains BERT to understand sentence relationships by predicting if sentence B follows sentence A.'
  },
  {
    id: 'bert9',
    question: 'How many parameters does BERT-base have?',
    options: ['12M', '110M', '340M', '1B'],
    correctAnswer: 1,
    explanation: 'BERT-base has 110M parameters (12 layers, 768 hidden, 12 heads); BERT-large has 340M (24 layers, 1024 hidden, 16 heads).'
  },
  {
    id: 'bert10',
    question: 'What are the special tokens in BERT?',
    options: ['None', '[CLS] (classification), [SEP] (separator), [MASK]', 'Only [MASK]', '<s>, </s>'],
    correctAnswer: 1,
    explanation: '[CLS] token representation used for classification; [SEP] separates sentences; [MASK] for MLM.'
  },
  {
    id: 'bert11',
    question: 'What is the [CLS] token?',
    options: ['Mask token', 'Special token at start; its representation used for sequence-level tasks', 'Separator', 'End token'],
    correctAnswer: 1,
    explanation: '[CLS] (classification) token is prepended; its final hidden state represents the entire sequence for classification.'
  },
  {
    id: 'bert12',
    question: 'How does BERT handle multiple sentences?',
    options: ['Separately', 'Concatenates with [SEP] token and uses segment embeddings', 'Cannot handle', 'Single sentence only'],
    correctAnswer: 1,
    explanation: 'BERT concatenates sentences with [SEP], adding segment embeddings (A/B) to distinguish them.'
  },
  {
    id: 'bert13',
    question: 'What are BERT\'s three embeddings added together?',
    options: ['Only word embeddings', 'Token embeddings + position embeddings + segment embeddings', 'Only positions', 'Only segments'],
    correctAnswer: 1,
    explanation: 'BERT input: token embeddings (what) + positional (where) + segment (which sentence), all summed element-wise.'
  },
  {
    id: 'bert14',
    question: 'What is BERT fine-tuning?',
    options: ['Pre-training only', 'Adding task-specific layer and training on downstream task', 'No training', 'Feature extraction only'],
    correctAnswer: 1,
    explanation: 'Fine-tuning adapts pre-trained BERT to specific tasks by adding output layer and training end-to-end with small learning rate.'
  },
  {
    id: 'bert15',
    question: 'What tasks can BERT be fine-tuned for?',
    options: ['Only classification', 'Classification, NER, QA, NLI, many NLP tasks', 'Only generation', 'Only translation'],
    correctAnswer: 1,
    explanation: 'BERT excels at understanding tasks: sentiment analysis, named entity recognition, question answering, natural language inference.'
  },
  {
    id: 'bert16',
    question: 'Can BERT generate text?',
    options: ['Yes, primarily', 'Not designed for generation (encoder-only, bidirectional)', 'Best for generation', 'Only generation'],
    correctAnswer: 1,
    explanation: 'BERT\'s bidirectional nature makes it unsuitable for autoregressive generation; use GPT-style decoders for that.'
  },
  {
    id: 'bert17',
    question: 'What datasets was BERT pre-trained on?',
    options: ['Single book', 'BooksCorpus (800M words) and English Wikipedia (2.5B words)', 'Only Wikipedia', 'Small dataset'],
    correctAnswer: 1,
    explanation: 'BERT used large-scale unsupervised text from BooksCorpus and Wikipedia for pre-training.'
  },
  {
    id: 'bert18',
    question: 'What is the maximum sequence length of BERT?',
    options: ['128', '512 tokens', '1024', 'Unlimited'],
    correctAnswer: 1,
    explanation: 'BERT was trained with maximum 512 tokens due to quadratic self-attention complexity.'
  },
  {
    id: 'bert19',
    question: 'What is WordPiece tokenization in BERT?',
    options: ['Character level', 'Subword tokenization splitting words into frequent subunits', 'Word level only', 'No tokenization'],
    correctAnswer: 1,
    explanation: 'WordPiece breaks words into subword units (e.g., "playing" → "play" + "##ing"), handling rare words better.'
  },
  {
    id: 'bert20',
    question: 'What is RoBERTa?',
    options: ['Smaller BERT', 'Robustly Optimized BERT: improved training (no NSP, more data, longer)', 'Older than BERT', 'Different architecture'],
    correctAnswer: 1,
    explanation: 'RoBERTa removes NSP, trains longer with more data and larger batches, achieving better performance than BERT.'
  },
  {
    id: 'bert21',
    question: 'What is ALBERT?',
    options: ['Larger BERT', 'A Lite BERT: parameter sharing and factorization for efficiency', 'Same as BERT', 'No parameters'],
    correctAnswer: 1,
    explanation: 'ALBERT shares parameters across layers and factors embeddings, achieving BERT performance with fewer parameters.'
  },
  {
    id: 'bert22',
    question: 'What is DistilBERT?',
    options: ['Larger BERT', 'Distilled version: smaller, faster, retaining 97% performance', 'Same size', 'Slower BERT'],
    correctAnswer: 1,
    explanation: 'DistilBERT uses knowledge distillation to create a 6-layer student model (40% smaller, 60% faster) from BERT teacher.'
  },
  {
    id: 'bert23',
    question: 'What is the impact of BERT?',
    options: ['Minor impact', 'Revolutionized NLP: set new SOTA on many tasks, popularized pre-training + fine-tuning', 'No impact', 'Slowed progress'],
    correctAnswer: 1,
    explanation: 'BERT achieved state-of-the-art on 11 NLP tasks, demonstrating the power of bidirectional pre-training and transforming NLP.'
  },
  {
    id: 'bert24',
    question: 'Can BERT embeddings be used as features?',
    options: ['No', 'Yes, extract frozen embeddings for downstream models', 'Only fine-tuning', 'Never beneficial'],
    correctAnswer: 1,
    explanation: 'BERT embeddings can be extracted and used as features for other models, though fine-tuning usually performs better.'
  },
  {
    id: 'bert25',
    question: 'What is multilingual BERT (mBERT)?',
    options: ['English only', 'BERT trained on 104 languages, enabling cross-lingual transfer', 'Two languages', 'No multilingual'],
    correctAnswer: 1,
    explanation: 'mBERT was trained on Wikipedia in 104 languages, learning shared representations that transfer across languages.'
  }
];

// GPT & Language Models - 25 questions
export const gptQuestions: QuizQuestion[] = [
  {
    id: 'gpt1',
    question: 'What does GPT stand for?',
    options: ['General Purpose Transformer', 'Generative Pre-trained Transformer', 'Gradient Pre-Training', 'Global Position Transformer'],
    correctAnswer: 1,
    explanation: 'GPT uses autoregressive pre-training on language modeling, then fine-tunes for downstream tasks.'
  },
  {
    id: 'gpt2',
    question: 'What architecture does GPT use?',
    options: ['Encoder only', 'Transformer decoder only (unidirectional/causal attention)', 'Encoder-decoder', 'RNN'],
    correctAnswer: 1,
    explanation: 'GPT uses only the Transformer decoder with masked/causal self-attention for autoregressive generation.'
  },
  {
    id: 'gpt3',
    question: 'What is the training objective of GPT?',
    options: ['MLM', 'Next token prediction (autoregressive language modeling)', 'NSP', 'Translation'],
    correctAnswer: 1,
    explanation: 'GPT predicts the next token given all previous tokens, trained to maximize likelihood of text sequences.'
  },
  {
    id: 'gpt4',
    question: 'How does GPT differ from BERT in attention?',
    options: ['No difference', 'GPT uses causal (unidirectional) attention; BERT uses bidirectional', 'BERT is causal', 'Both same'],
    correctAnswer: 1,
    explanation: 'GPT masks future tokens (left-to-right), while BERT sees full context (bidirectional).'
  },
  {
    id: 'gpt5',
    question: 'What tasks is GPT designed for?',
    options: ['Only classification', 'Text generation and language understanding', 'Only understanding', 'Image tasks'],
    correctAnswer: 1,
    explanation: 'GPT excels at generation (completion, dialogue, stories) and can handle understanding tasks with appropriate prompting.'
  },
  {
    id: 'gpt6',
    question: 'How many parameters did GPT-1 have?',
    options: ['12M', '117M', '1.5B', '175B'],
    correctAnswer: 1,
    explanation: 'GPT-1 (2018) had 117M parameters with 12 layers. GPT-2: 1.5B, GPT-3: 175B, GPT-4: undisclosed but much larger.'
  },
  {
    id: 'gpt7',
    question: 'What was GPT-2 known for?',
    options: ['Small size', 'Larger scale (1.5B), controversial withheld release, impressive generation', 'Slow speed', 'Only classification'],
    correctAnswer: 1,
    explanation: 'GPT-2 showed that scale enables better generation, initially withheld due to concerns about misuse.'
  },
  {
    id: 'gpt8',
    question: 'How many parameters does GPT-3 have?',
    options: ['1.5B', '175B', '12B', '340M'],
    correctAnswer: 1,
    explanation: 'GPT-3 (2020) scaled to 175 billion parameters, demonstrating few-shot learning and in-context learning abilities.'
  },
  {
    id: 'gpt9',
    question: 'What is few-shot learning in GPT-3?',
    options: ['Fine-tuning', 'Learning from few examples provided in the prompt without parameter updates', 'Pre-training', 'No learning'],
    correctAnswer: 1,
    explanation: 'GPT-3 can perform tasks given just a few examples in the prompt, without fine-tuning or gradient updates.'
  },
  {
    id: 'gpt10',
    question: 'What is in-context learning?',
    options: ['Fine-tuning', 'Model adapts to task from examples/instructions in the prompt', 'Pre-training', 'No learning'],
    correctAnswer: 1,
    explanation: 'In-context learning means the model learns the task at inference time from the prompt context alone.'
  },
  {
    id: 'gpt11',
    question: 'What is zero-shot learning in GPT?',
    options: ['Requires examples', 'Performing task from instruction alone, without examples', 'Fine-tuning', 'Pre-training'],
    correctAnswer: 1,
    explanation: 'Zero-shot: GPT performs task based only on natural language instruction, no examples needed.'
  },
  {
    id: 'gpt12',
    question: 'What is prompt engineering?',
    options: ['Model architecture', 'Crafting effective prompts to elicit desired model behavior', 'Data preprocessing', 'Fine-tuning'],
    correctAnswer: 1,
    explanation: 'Prompt engineering designs input prompts to guide model outputs, crucial for getting good results from LLMs.'
  },
  {
    id: 'gpt13',
    question: 'What tokenization does GPT use?',
    options: ['Character level', 'Byte-Pair Encoding (BPE)', 'Word level', 'No tokenization'],
    correctAnswer: 1,
    explanation: 'GPT models use BPE, which merges frequent byte/character pairs into tokens, efficiently handling any text.'
  },
  {
    id: 'gpt14',
    question: 'What is the context window of GPT-3?',
    options: ['512', '2048 tokens', '4096', 'Unlimited'],
    correctAnswer: 1,
    explanation: 'GPT-3 has a 2048 token context window; newer models like GPT-4 have larger contexts (8K, 32K, 128K).'
  },
  {
    id: 'gpt15',
    question: 'What is temperature in GPT sampling?',
    options: ['Model size', 'Controls randomness: low=deterministic, high=creative', 'Learning rate', 'Batch size'],
    correctAnswer: 1,
    explanation: 'Temperature scales logits before softmax: T→0 becomes greedy, T>1 increases diversity/randomness.'
  },
  {
    id: 'gpt16',
    question: 'What is top-k sampling?',
    options: ['Greedy', 'Samples from k most likely tokens', 'Random sampling', 'No sampling'],
    correctAnswer: 1,
    explanation: 'Top-k restricts sampling to k most probable tokens, balancing quality and diversity (e.g., k=40).'
  },
  {
    id: 'gpt17',
    question: 'What is nucleus (top-p) sampling?',
    options: ['Top-k', 'Samples from smallest set of tokens with cumulative probability ≥ p', 'Greedy', 'Random'],
    correctAnswer: 1,
    explanation: 'Top-p (nucleus) sampling dynamically adjusts vocabulary size based on cumulative probability (e.g., p=0.9).'
  },
  {
    id: 'gpt18',
    question: 'What is ChatGPT?',
    options: ['Original GPT', 'GPT fine-tuned for dialogue using RLHF (Reinforcement Learning from Human Feedback)', 'BERT variant', 'Small model'],
    correctAnswer: 1,
    explanation: 'ChatGPT applies RLHF to GPT-3.5/4, training it to be helpful, harmless, and honest through human feedback.'
  },
  {
    id: 'gpt19',
    question: 'What is RLHF?',
    options: ['Pre-training method', 'Reinforcement Learning from Human Feedback: aligns model with human preferences', 'Supervised learning', 'No learning'],
    correctAnswer: 1,
    explanation: 'RLHF trains a reward model from human preferences, then uses RL (PPO) to optimize the language model.'
  },
  {
    id: 'gpt20',
    question: 'What is InstructGPT?',
    options: ['Original GPT', 'GPT fine-tuned to follow instructions using RLHF', 'BERT', 'No instructions'],
    correctAnswer: 1,
    explanation: 'InstructGPT uses RLHF to make GPT-3 better at following user instructions, predecessor to ChatGPT.'
  },
  {
    id: 'gpt21',
    question: 'Can GPT be fine-tuned?',
    options: ['No', 'Yes, can be fine-tuned on specific tasks or domains', 'Only pre-trained', 'Never beneficial'],
    correctAnswer: 1,
    explanation: 'GPT can be fine-tuned on custom data for specific applications, though large models often perform well with prompting alone.'
  },
  {
    id: 'gpt22',
    question: 'What is the scaling law observation?',
    options: ['Size doesn\'t matter', 'Performance improves predictably with model size, data, and compute', 'Random relationship', 'Performance decreases'],
    correctAnswer: 1,
    explanation: 'Scaling laws show that loss decreases as a power law with scale, motivating ever-larger models.'
  },
  {
    id: 'gpt23',
    question: 'What are emergent abilities in large language models?',
    options: ['Present in small models', 'Abilities that appear only above certain scale thresholds', 'No abilities', 'Decrease with scale'],
    correctAnswer: 1,
    explanation: 'Emergent abilities like few-shot learning, chain-of-thought reasoning appear suddenly in sufficiently large models.'
  },
  {
    id: 'gpt24',
    question: 'What is chain-of-thought prompting?',
    options: ['Simple prompt', 'Encouraging model to show reasoning steps before answering', 'No reasoning', 'Direct answer only'],
    correctAnswer: 1,
    explanation: 'Chain-of-thought prompting adds "let\'s think step by step," dramatically improving reasoning and math performance.'
  },
  {
    id: 'gpt25',
    question: 'What distinguishes GPT-4 from GPT-3?',
    options: ['Smaller', 'Multimodal (vision+text), more capable, safer, longer context', 'Only text', 'No improvements'],
    correctAnswer: 1,
    explanation: 'GPT-4 handles images, has better reasoning, fewer hallucinations, and supports up to 128K token context.'
  }
];
