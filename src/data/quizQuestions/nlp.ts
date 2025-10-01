import { QuizQuestion } from '../../types';

// Word Embeddings - 20 questions
export const wordEmbeddingsQuestions: QuizQuestion[] = [
  {
    id: 'we1',
    question: 'What are word embeddings?',
    options: ['One-hot vectors', 'Dense vector representations of words that capture semantic meaning', 'Character sequences', 'Word counts'],
    correctAnswer: 1,
    explanation: 'Word embeddings are dense, low-dimensional vectors that represent words in continuous space, capturing semantic relationships.'
  },
  {
    id: 'we2',
    question: 'Why are embeddings better than one-hot encoding?',
    options: ['Simpler', 'Capture semantic similarity and are much lower dimensional', 'Faster to compute', 'More interpretable'],
    correctAnswer: 1,
    explanation: 'Embeddings are dense (e.g., 300D vs 50,000D), capture meaning, and similar words have similar vectors, unlike sparse one-hot.'
  },
  {
    id: 'we3',
    question: 'What is Word2Vec?',
    options: ['Transformer model', 'Neural method to learn word embeddings from context', 'Tokenizer', 'Loss function'],
    correctAnswer: 1,
    explanation: 'Word2Vec (2013) learns word embeddings by predicting context words (CBOW) or target words (Skip-gram) from large corpora.'
  },
  {
    id: 'we4',
    question: 'What are the two Word2Vec architectures?',
    options: ['Encoder-decoder', 'CBOW (predict word from context) and Skip-gram (predict context from word)', 'Attention-based', 'Recurrent'],
    correctAnswer: 1,
    explanation: 'CBOW predicts center word from surrounding words; Skip-gram predicts surrounding words from center word.'
  },
  {
    id: 'we5',
    question: 'Which Word2Vec method works better for rare words?',
    options: ['CBOW', 'Skip-gram', 'Both equal', 'Neither'],
    correctAnswer: 1,
    explanation: 'Skip-gram treats each context word as a separate observation, providing more training examples for rare words.'
  },
  {
    id: 'we6',
    question: 'What is GloVe?',
    options: ['RNN model', 'Global Vectors: learns embeddings from word co-occurrence statistics', 'Transformer', 'Tokenizer'],
    correctAnswer: 1,
    explanation: 'GloVe (Global Vectors for Word Representation) factorizes a word co-occurrence matrix to learn embeddings.'
  },
  {
    id: 'we7',
    question: 'What is the main difference between Word2Vec and GloVe?',
    options: ['No difference', 'Word2Vec uses local context windows; GloVe uses global co-occurrence statistics', 'GloVe is slower', 'Word2Vec is newer'],
    correctAnswer: 1,
    explanation: 'Word2Vec is a predictive model using local context; GloVe is a count-based model leveraging global corpus statistics.'
  },
  {
    id: 'we8',
    question: 'What does "king - man + woman = ?" famously demonstrate?',
    options: ['Nothing', 'Word embeddings capture semantic relationships (answer: queen)', 'Randomness', 'Word frequency'],
    correctAnswer: 1,
    explanation: 'This example shows embeddings capture analogies: king-man+woman ≈ queen, demonstrating linear relationships in embedding space.'
  },
  {
    id: 'we9',
    question: 'What is cosine similarity used for in embeddings?',
    options: ['Training', 'Measuring semantic similarity between word vectors', 'Loss function', 'Optimization'],
    correctAnswer: 1,
    explanation: 'Cosine similarity measures the angle between vectors, with values close to 1 indicating similar meanings.'
  },
  {
    id: 'we10',
    question: 'What is a limitation of Word2Vec/GloVe embeddings?',
    options: ['Too large', 'Fixed vectors: same word has same embedding regardless of context (polysemy)', 'Too slow', 'Too complex'],
    correctAnswer: 1,
    explanation: 'Static embeddings can\'t handle multiple meanings of words (e.g., "bank" as financial institution vs river bank).'
  },
  {
    id: 'we11',
    question: 'What is FastText?',
    options: ['Faster Word2Vec', 'Extends Word2Vec by representing words as bags of character n-grams', 'Transformer', 'Tokenizer'],
    correctAnswer: 1,
    explanation: 'FastText learns embeddings using subword information (character n-grams), handling rare words and morphology better.'
  },
  {
    id: 'we12',
    question: 'What advantage does FastText have over Word2Vec?',
    options: ['Faster only', 'Can generate embeddings for unseen words using subword information', 'Larger embeddings', 'Older method'],
    correctAnswer: 1,
    explanation: 'FastText can create embeddings for out-of-vocabulary words by combining character n-gram vectors.'
  },
  {
    id: 'we13',
    question: 'What is negative sampling in Word2Vec?',
    options: ['Removing bad data', 'Training technique that samples negative examples to make training efficient', 'Data augmentation', 'Regularization'],
    correctAnswer: 1,
    explanation: 'Negative sampling randomly selects non-context words as negative examples, avoiding expensive softmax over entire vocabulary.'
  },
  {
    id: 'we14',
    question: 'How are word embeddings typically initialized for downstream tasks?',
    options: ['Random', 'Pre-trained on large corpus, then fine-tuned or frozen', 'Zeros', 'Ones'],
    correctAnswer: 1,
    explanation: 'Embeddings are usually pre-trained on large corpora (Wikipedia, Common Crawl) and then used or fine-tuned for specific tasks.'
  },
  {
    id: 'we15',
    question: 'What is the typical dimensionality of word embeddings?',
    options: ['1-10', '50-300', '10,000+', '1,000,000'],
    correctAnswer: 1,
    explanation: 'Common dimensions are 50, 100, 200, or 300, balancing expressiveness and computational efficiency.'
  },
  {
    id: 'we16',
    question: 'Can word embeddings capture syntactic relationships?',
    options: ['No, only semantic', 'Yes, they capture both semantic and syntactic patterns', 'Only syntax', 'Neither'],
    correctAnswer: 1,
    explanation: 'Embeddings capture both meaning (semantic) and grammar (syntactic) patterns, e.g., verb tenses, pluralization.'
  },
  {
    id: 'we17',
    question: 'What is contextualized word embedding?',
    options: ['Static embedding', 'Embedding that changes based on context (e.g., from BERT, ELMo)', 'One-hot encoding', 'Character-level'],
    correctAnswer: 1,
    explanation: 'Contextualized embeddings (from models like BERT, ELMo) generate different vectors for the same word in different contexts.'
  },
  {
    id: 'we18',
    question: 'What is ELMo?',
    options: ['Static embeddings', 'Embeddings from Language Models: contextualized embeddings from bi-directional LSTM', 'Transformer', 'Word2Vec variant'],
    correctAnswer: 1,
    explanation: 'ELMo (2018) generates contextualized embeddings using a deep bidirectional LSTM language model.'
  },
  {
    id: 'we19',
    question: 'What problem does ELMo solve compared to Word2Vec?',
    options: ['Speed', 'Handles polysemy by providing context-dependent embeddings', 'Size reduction', 'Simplicity'],
    correctAnswer: 1,
    explanation: 'ELMo gives different embeddings for "bank" in "river bank" vs "bank account" based on context.'
  },
  {
    id: 'we20',
    question: 'How do you evaluate word embeddings?',
    options: ['No evaluation', 'Intrinsic (analogy tasks, similarity) and extrinsic (downstream task performance)', 'Only visualization', 'Random testing'],
    correctAnswer: 1,
    explanation: 'Intrinsic: word similarity, analogies. Extrinsic: performance on tasks like classification, NER, or QA when used as features.'
  }
];

// RNNs - 20 questions
export const rnnQuestions: QuizQuestion[] = [
  {
    id: 'rnn1',
    question: 'What is an RNN (Recurrent Neural Network)?',
    options: ['Feedforward network', 'Network with loops that process sequential data by maintaining hidden state', 'CNN variant', 'Decision tree'],
    correctAnswer: 1,
    explanation: 'RNNs process sequences by maintaining a hidden state that gets updated at each time step, creating a form of memory.'
  },
  {
    id: 'rnn2',
    question: 'Why use RNNs for sequential data?',
    options: ['Faster than CNNs', 'They maintain temporal dependencies and have memory of previous inputs', 'Simpler architecture', 'Less parameters'],
    correctAnswer: 1,
    explanation: 'RNNs are designed for sequences, sharing weights across time steps and maintaining context through hidden states.'
  },
  {
    id: 'rnn3',
    question: 'What is the hidden state in an RNN?',
    options: ['Input', 'Vector that stores information from previous time steps', 'Output', 'Weights'],
    correctAnswer: 1,
    explanation: 'The hidden state acts as memory, updated at each step: h_t = f(h_{t-1}, x_t), carrying information through the sequence.'
  },
  {
    id: 'rnn4',
    question: 'What is the main problem with vanilla RNNs?',
    options: ['Too fast', 'Vanishing/exploding gradients, difficulty learning long-term dependencies', 'Too many parameters', 'Too accurate'],
    correctAnswer: 1,
    explanation: 'Gradients can vanish (become tiny) or explode (become huge) when backpropagated through many time steps.'
  },
  {
    id: 'rnn5',
    question: 'Why do RNNs suffer from vanishing gradients?',
    options: ['Bad initialization', 'Repeated multiplication of gradients through time causes exponential decay', 'Too deep', 'Wrong optimizer'],
    correctAnswer: 1,
    explanation: 'Backpropagation through time involves many matrix multiplications. If weights are < 1, gradients vanish exponentially.'
  },
  {
    id: 'rnn6',
    question: 'What is BPTT (Backpropagation Through Time)?',
    options: ['Standard backprop', 'Unrolling RNN through time and applying backpropagation across time steps', 'Forward pass only', 'No gradients'],
    correctAnswer: 1,
    explanation: 'BPTT unrolls the RNN into a feedforward network across time steps, then computes gradients backward through time.'
  },
  {
    id: 'rnn7',
    question: 'What is truncated BPTT?',
    options: ['Full BPTT', 'Limiting backpropagation to fixed number of time steps for efficiency', 'No truncation', 'Forward only'],
    correctAnswer: 1,
    explanation: 'Truncated BPTT only backpropagates through k steps instead of entire sequence, reducing memory and computation.'
  },
  {
    id: 'rnn8',
    question: 'What types of sequence tasks can RNNs handle?',
    options: ['Only one-to-one', 'One-to-many, many-to-one, many-to-many', 'Only classification', 'No sequences'],
    correctAnswer: 1,
    explanation: 'RNNs are flexible: one-to-many (image captioning), many-to-one (sentiment analysis), many-to-many (translation, NER).'
  },
  {
    id: 'rnn9',
    question: 'What is a bidirectional RNN?',
    options: ['Single direction', 'Processes sequence in both forward and backward directions', 'No direction', 'Random direction'],
    correctAnswer: 1,
    explanation: 'Bidirectional RNNs have two hidden states: one processes left-to-right, other right-to-left, capturing full context.'
  },
  {
    id: 'rnn10',
    question: 'When are bidirectional RNNs useful?',
    options: ['Real-time prediction', 'When entire sequence is available and context from both directions helps', 'Never', 'Only for images'],
    correctAnswer: 1,
    explanation: 'Bidirectional RNNs are great for tasks like NER, POS tagging where you can see the entire sentence, but not for real-time generation.'
  },
  {
    id: 'rnn11',
    question: 'Can RNNs handle variable-length sequences?',
    options: ['No, fixed only', 'Yes, naturally process variable-length inputs', 'Only with padding', 'Only short sequences'],
    correctAnswer: 1,
    explanation: 'RNNs naturally handle sequences of any length since they process one element at a time.'
  },
  {
    id: 'rnn12',
    question: 'What is teacher forcing?',
    options: ['Testing method', 'Training technique where true output is fed as input at next time step', 'Regularization', 'Optimizer'],
    correctAnswer: 1,
    explanation: 'Teacher forcing uses ground truth as input at each time step during training, rather than the model\'s own predictions.'
  },
  {
    id: 'rnn13',
    question: 'What is a disadvantage of teacher forcing?',
    options: ['Slower training', 'Exposure bias: model not trained on its own predictions, causing train-test mismatch', 'Too complex', 'No disadvantage'],
    correctAnswer: 1,
    explanation: 'At inference, the model uses its own predictions which may differ from training where it saw ground truth.'
  },
  {
    id: 'rnn14',
    question: 'How do RNNs share parameters?',
    options: ['Different weights per step', 'Same weights applied at every time step', 'No sharing', 'Random sharing'],
    correctAnswer: 1,
    explanation: 'RNNs use the same weight matrices at every time step, enabling generalization across sequence positions.'
  },
  {
    id: 'rnn15',
    question: 'What is a many-to-one RNN used for?',
    options: ['Translation', 'Sentiment analysis, sequence classification', 'Image generation', 'Segmentation'],
    correctAnswer: 1,
    explanation: 'Many-to-one: processes entire sequence and outputs single value, e.g., classifying sentiment of a sentence.'
  },
  {
    id: 'rnn16',
    question: 'What is a one-to-many RNN used for?',
    options: ['Classification', 'Image captioning, music generation', 'Sentiment analysis', 'Object detection'],
    correctAnswer: 1,
    explanation: 'One-to-many: takes single input and generates sequence, e.g., image → caption or seed → music sequence.'
  },
  {
    id: 'rnn17',
    question: 'What is a many-to-many RNN used for?',
    options: ['Classification only', 'Machine translation, video captioning, named entity recognition', 'Single predictions', 'Images only'],
    correctAnswer: 1,
    explanation: 'Many-to-many: sequence input → sequence output, e.g., translation, video captioning, POS tagging.'
  },
  {
    id: 'rnn18',
    question: 'Can RNNs be stacked in layers?',
    options: ['No, single layer only', 'Yes, creating deep RNNs by stacking multiple layers', 'Only two layers', 'Never beneficial'],
    correctAnswer: 1,
    explanation: 'Deep RNNs stack multiple layers, with each layer\'s hidden states feeding the next layer, learning hierarchical representations.'
  },
  {
    id: 'rnn19',
    question: 'What is the computational complexity of RNNs?',
    options: ['Parallel', 'Sequential (cannot parallelize across time), O(T) for T time steps', 'Constant time', 'Exponential'],
    correctAnswer: 1,
    explanation: 'RNNs process sequentially since each step depends on previous hidden state, making them slow for long sequences.'
  },
  {
    id: 'rnn20',
    question: 'Why have transformers largely replaced RNNs?',
    options: ['RNNs are better', 'Transformers parallelize better and capture long-range dependencies more effectively', 'RNNs are newer', 'No replacement'],
    correctAnswer: 1,
    explanation: 'Transformers process entire sequence in parallel and use attention to capture dependencies, overcoming RNN limitations.'
  }
];

// LSTM & GRU - 25 questions
export const lstmGruQuestions: QuizQuestion[] = [
  {
    id: 'lstm1',
    question: 'What does LSTM stand for?',
    options: ['Linear Short-Term Memory', 'Long Short-Term Memory', 'Large Scale Training Model', 'Layered Sequential Trained Model'],
    correctAnswer: 1,
    explanation: 'LSTM (Long Short-Term Memory) is an RNN variant designed to remember information for long periods.'
  },
  {
    id: 'lstm2',
    question: 'What problem do LSTMs solve?',
    options: ['Too fast training', 'Vanishing gradient problem in vanilla RNNs, enabling long-term dependencies', 'Too few parameters', 'Classification only'],
    correctAnswer: 1,
    explanation: 'LSTMs use gating mechanisms to maintain gradients, allowing them to learn dependencies across long sequences.'
  },
  {
    id: 'lstm3',
    question: 'What is the cell state in LSTM?',
    options: ['Hidden state', 'Separate memory that runs through the entire chain with minimal modifications', 'Input', 'Output'],
    correctAnswer: 1,
    explanation: 'The cell state acts as a "conveyor belt" carrying information across time steps with minimal linear interactions.'
  },
  {
    id: 'lstm4',
    question: 'How many gates does a standard LSTM have?',
    options: ['One', 'Three: forget, input, output', 'Five', 'None'],
    correctAnswer: 1,
    explanation: 'LSTM has three gates: forget gate (what to remove), input gate (what to add), output gate (what to output).'
  },
  {
    id: 'lstm5',
    question: 'What does the forget gate do?',
    options: ['Adds information', 'Decides what information to discard from cell state', 'Outputs results', 'Initializes weights'],
    correctAnswer: 1,
    explanation: 'Forget gate uses sigmoid to output 0-1 for each cell state element, determining what to keep (1) or forget (0).'
  },
  {
    id: 'lstm6',
    question: 'What does the input gate do?',
    options: ['Forgets information', 'Decides what new information to add to cell state', 'Outputs results', 'Nothing'],
    correctAnswer: 1,
    explanation: 'Input gate determines which values to update and creates candidate values to add to the cell state.'
  },
  {
    id: 'lstm7',
    question: 'What does the output gate do?',
    options: ['Updates cell state', 'Decides what to output based on cell state', 'Adds information', 'Removes information'],
    correctAnswer: 1,
    explanation: 'Output gate filters the cell state to produce the hidden state output for the current time step.'
  },
  {
    id: 'lstm8',
    question: 'Why do LSTM gates use sigmoid activation?',
    options: ['Random choice', 'Sigmoid outputs 0-1, acting as a gating mechanism (0=block, 1=allow)', 'Fastest activation', 'No reason'],
    correctAnswer: 1,
    explanation: 'Sigmoid outputs values between 0 and 1, making it perfect for controlling information flow (0=completely block, 1=completely allow).'
  },
  {
    id: 'lstm9',
    question: 'What activation is typically used for candidate values in LSTM?',
    options: ['Sigmoid', 'Tanh', 'ReLU', 'Linear'],
    correctAnswer: 1,
    explanation: 'Tanh is used to create candidate values in [-1, 1] range, suitable for adding to the cell state.'
  },
  {
    id: 'lstm10',
    question: 'Are LSTMs more parameter-heavy than vanilla RNNs?',
    options: ['No, fewer parameters', 'Yes, ~4x more parameters due to gates', 'Same parameters', 'Only 2x more'],
    correctAnswer: 1,
    explanation: 'LSTMs have forget, input, output gates, and candidate generation, requiring roughly 4× the parameters of vanilla RNN.'
  },
  {
    id: 'lstm11',
    question: 'What is GRU?',
    options: ['Same as LSTM', 'Gated Recurrent Unit: simpler variant of LSTM with fewer gates', 'Better than LSTM always', 'Older than LSTM'],
    correctAnswer: 1,
    explanation: 'GRU (Gated Recurrent Unit) is a simplified LSTM variant with only 2 gates (reset and update) instead of 3.'
  },
  {
    id: 'lstm12',
    question: 'How many gates does GRU have?',
    options: ['Three', 'Two: reset and update', 'One', 'Four'],
    correctAnswer: 1,
    explanation: 'GRU has reset gate (decides what to forget) and update gate (decides what to update), merging forget and input.'
  },
  {
    id: 'lstm13',
    question: 'What does the update gate in GRU do?',
    options: ['Only forgets', 'Controls how much past information to keep and new information to add', 'Only adds', 'No function'],
    correctAnswer: 1,
    explanation: 'Update gate combines LSTM\'s forget and input gates, deciding the balance between past and new information.'
  },
  {
    id: 'lstm14',
    question: 'What does the reset gate in GRU do?',
    options: ['Outputs results', 'Decides how much past information to forget when computing new candidate', 'Adds information', 'Nothing'],
    correctAnswer: 1,
    explanation: 'Reset gate determines how much of the previous hidden state to use when creating the candidate activation.'
  },
  {
    id: 'lstm15',
    question: 'Which is simpler: LSTM or GRU?',
    options: ['LSTM', 'GRU (fewer parameters and computations)', 'Equally complex', 'Neither is simple'],
    correctAnswer: 1,
    explanation: 'GRU has fewer gates and no separate cell state, making it simpler, faster, and using fewer parameters than LSTM.'
  },
  {
    id: 'lstm16',
    question: 'Which performs better: LSTM or GRU?',
    options: ['LSTM always', 'Task-dependent; GRU often comparable with faster training', 'GRU always', 'Neither works'],
    correctAnswer: 1,
    explanation: 'Performance is task-dependent. GRU is often as good as LSTM while being faster to train due to fewer parameters.'
  },
  {
    id: 'lstm17',
    question: 'Can LSTMs/GRUs be bidirectional?',
    options: ['No', 'Yes, processing sequence forward and backward simultaneously', 'Only unidirectional', 'Only for images'],
    correctAnswer: 1,
    explanation: 'Bidirectional LSTMs/GRUs use two layers (forward and backward), concatenating outputs for full context awareness.'
  },
  {
    id: 'lstm18',
    question: 'What is peephole connection in LSTM?',
    options: ['Standard connection', 'Variant where gates can look at cell state', 'Input connection', 'No such thing'],
    correctAnswer: 1,
    explanation: 'Peephole LSTMs allow gate layers to see the cell state, potentially improving performance on certain tasks.'
  },
  {
    id: 'lstm19',
    question: 'Are LSTMs still widely used today?',
    options: ['Completely obsolete', 'Still used but often replaced by Transformers for many NLP tasks', 'Most popular', 'Never were used'],
    correctAnswer: 1,
    explanation: 'LSTMs are still used for time series and some sequential tasks, but Transformers dominate NLP due to better parallelization.'
  },
  {
    id: 'lstm20',
    question: 'What is the typical initialization for LSTM forget gate bias?',
    options: ['Zero', 'Positive value (e.g., 1) to remember by default', 'Negative', 'Random'],
    correctAnswer: 1,
    explanation: 'Initializing forget gate bias to 1 makes the gate open by default, helping with gradient flow during early training.'
  },
  {
    id: 'lstm21',
    question: 'Can LSTMs handle variable-length sequences?',
    options: ['No', 'Yes, process sequences of any length', 'Only fixed length', 'Only short sequences'],
    correctAnswer: 1,
    explanation: 'Like all RNNs, LSTMs naturally handle variable-length sequences by processing one time step at a time.'
  },
  {
    id: 'lstm22',
    question: 'What is the gradient flow advantage in LSTM?',
    options: ['No advantage', 'Cell state provides path for gradients with minimal obstruction', 'Worse than RNN', 'Only forward flow'],
    correctAnswer: 1,
    explanation: 'The cell state allows gradients to flow through many time steps with only linear operations, avoiding vanishing gradients.'
  },
  {
    id: 'lstm23',
    question: 'Are LSTMs good for time series prediction?',
    options: ['No', 'Yes, commonly used for time series forecasting and anomaly detection', 'Only for text', 'Only for images'],
    correctAnswer: 1,
    explanation: 'LSTMs excel at time series tasks (stock prediction, weather forecasting, sensor data) due to their temporal memory.'
  },
  {
    id: 'lstm24',
    question: 'What is the computational bottleneck of LSTMs?',
    options: ['Memory only', 'Sequential processing prevents parallelization across time', 'Too few parameters', 'No bottleneck'],
    correctAnswer: 1,
    explanation: 'LSTMs must process sequentially (each step depends on previous), unlike Transformers which parallelize across sequence.'
  },
  {
    id: 'lstm25',
    question: 'When would you choose LSTM over Transformer?',
    options: ['Never', 'For streaming/online learning, time series, or when sequential processing is natural', 'Always', 'Only for images'],
    correctAnswer: 1,
    explanation: 'LSTMs are good for streaming data, time series, or scenarios where you process one element at a time without full sequence.'
  }
];
