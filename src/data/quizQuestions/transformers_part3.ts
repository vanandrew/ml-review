import { QuizQuestion } from '../../types';

// T5 & BART - 20 questions
export const t5BartQuestions: QuizQuestion[] = [
  {
    id: 't5b1',
    question: 'What does T5 stand for?',
    options: ['Transformer Type 5', 'Text-to-Text Transfer Transformer', 'Training Task 5', 'Token Transform 5'],
    correctAnswer: 1,
    explanation: 'T5 treats every NLP task as text-to-text: input text, output text, using a unified encoder-decoder framework.'
  },
  {
    id: 't5b2',
    question: 'What is the T5 approach to NLP tasks?',
    options: ['Different models per task', 'Unified text-to-text framework for all tasks', 'Only classification', 'Only generation'],
    correctAnswer: 1,
    explanation: 'T5 frames classification, translation, QA, summarization all as: input text → output text.'
  },
  {
    id: 't5b3',
    question: 'How does T5 handle classification?',
    options: ['Special head', 'Generates class label as text (e.g., "positive" or "negative")', 'Binary output', 'Numerical output'],
    correctAnswer: 1,
    explanation: 'T5 generates class names as text tokens rather than using classification heads.'
  },
  {
    id: 't5b4',
    question: 'What pre-training objective does T5 use?',
    options: ['MLM', 'Span corruption: mask and predict contiguous spans', 'Next token', 'NSP'],
    correctAnswer: 1,
    explanation: 'T5 masks contiguous spans (not random tokens) and trains to reconstruct them, similar to denoising.'
  },
  {
    id: 't5b5',
    question: 'What architecture does T5 use?',
    options: ['Encoder only', 'Transformer encoder-decoder', 'Decoder only', 'RNN'],
    correctAnswer: 1,
    explanation: 'T5 uses standard Transformer encoder-decoder architecture like the original Transformer.'
  },
  {
    id: 't5b6',
    question: 'How does T5 add task information?',
    options: ['Special tokens only', 'Task prefix in input text (e.g., "translate English to German: ...")', 'Separate model', 'No task info'],
    correctAnswer: 1,
    explanation: 'T5 prepends task description to input text, allowing one model to handle multiple tasks.'
  },
  {
    id: 't5b7',
    question: 'What is the C4 dataset?',
    options: ['Small dataset', 'Colossal Clean Crawled Corpus: 750GB cleaned web text for T5 pre-training', 'Wikipedia only', '1GB dataset'],
    correctAnswer: 1,
    explanation: 'C4 is a massive cleaned Common Crawl dataset created for T5 pre-training.'
  },
  {
    id: 't5b8',
    question: 'How does T5 handle position information?',
    options: ['Absolute learned', 'Relative position biases added to attention scores', 'Sinusoidal', 'No positions'],
    correctAnswer: 1,
    explanation: 'T5 uses simplified relative position biases based on bucketed distances between tokens.'
  },
  {
    id: 't5b9',
    question: 'What sizes does T5 come in?',
    options: ['Only one size', 'Small, Base, Large, XL, XXL (60M to 11B parameters)', 'Two sizes', 'No variants'],
    correctAnswer: 1,
    explanation: 'T5 ranges from T5-Small (60M) to T5-XXL (11B), allowing size-quality-compute tradeoffs.'
  },
  {
    id: 't5b10',
    question: 'What does BART stand for?',
    options: ['Basic Transformer', 'Bidirectional and Auto-Regressive Transformers', 'Binary Auto-Regressive', 'Batch Regression'],
    correctAnswer: 1,
    explanation: 'BART combines bidirectional encoder (like BERT) with autoregressive decoder (like GPT).'
  },
  {
    id: 't5b11',
    question: 'What architecture does BART use?',
    options: ['Encoder only', 'Standard Transformer encoder-decoder', 'Decoder only', 'CNN'],
    correctAnswer: 1,
    explanation: 'BART is a standard sequence-to-sequence Transformer with encoder and decoder.'
  },
  {
    id: 't5b12',
    question: 'What pre-training objective does BART use?',
    options: ['Only MLM', 'Denoising autoencoding: corrupt text and reconstruct it', 'Next token only', 'No objective'],
    correctAnswer: 1,
    explanation: 'BART is trained to reconstruct original text from corrupted versions using various noise functions.'
  },
  {
    id: 't5b13',
    question: 'What corruption methods does BART use?',
    options: ['Only masking', 'Token masking, deletion, shuffling, rotation, text infilling', 'No corruption', 'Only deletion'],
    correctAnswer: 1,
    explanation: 'BART uses diverse noise: masking tokens, deleting them, permuting sentences, and document rotation.'
  },
  {
    id: 't5b14',
    question: 'What tasks is BART good at?',
    options: ['Only classification', 'Generation tasks: summarization, translation, dialogue', 'Only understanding', 'No tasks'],
    correctAnswer: 1,
    explanation: 'BART excels at generation tasks, especially abstractive summarization, due to its encoder-decoder architecture.'
  },
  {
    id: 't5b15',
    question: 'How do T5 and BART differ?',
    options: ['No difference', 'T5 uses span corruption, BART uses diverse noise; T5 is text-to-text framed', 'Same objectives', 'Same everything'],
    correctAnswer: 1,
    explanation: 'T5 focuses on unified text-to-text with span masking; BART uses richer corruption schemes.'
  },
  {
    id: 't5b16',
    question: 'Can BART and T5 do zero-shot tasks?',
    options: ['No', 'Limited; better with fine-tuning, unlike massive models like GPT-3', 'Better than GPT-3', 'Always'],
    correctAnswer: 1,
    explanation: 'BART and T5 perform well when fine-tuned but have limited zero-shot abilities compared to very large LLMs.'
  },
  {
    id: 't5b17',
    question: 'What is mBART?',
    options: ['Mini BART', 'Multilingual BART trained on 25 languages', 'English only', 'No multilingual'],
    correctAnswer: 1,
    explanation: 'mBART pre-trains on 25 languages, enabling cross-lingual transfer and multilingual generation.'
  },
  {
    id: 't5b18',
    question: 'What is mT5?',
    options: ['Mini T5', 'Multilingual T5 trained on 101 languages', 'English only', 'No multilingual'],
    correctAnswer: 1,
    explanation: 'mT5 extends T5 to 101 languages using mC4 (multilingual C4) corpus.'
  },
  {
    id: 't5b19',
    question: 'When should you use T5/BART over GPT?',
    options: ['Never', 'When you need encoder-decoder for transformation tasks with explicit inputs', 'Always', 'No difference'],
    correctAnswer: 1,
    explanation: 'T5/BART suit tasks with clear input-output pairs (translation, summarization); GPT better for open-ended generation.'
  },
  {
    id: 't5b20',
    question: 'Are T5 and BART still relevant?',
    options: ['Obsolete', 'Yes, still widely used for fine-tuning on generation tasks', 'No uses', 'Only research'],
    correctAnswer: 1,
    explanation: 'T5 and BART remain popular for fine-tuning on summarization, translation, and other generation tasks.'
  }
];

// Fine-tuning vs Prompting - 20 questions
export const fineTuningPromptingQuestions: QuizQuestion[] = [
  {
    id: 'ftp1',
    question: 'What is fine-tuning?',
    options: ['Using pre-trained model as-is', 'Continuing training on task-specific data with small learning rate', 'No training', 'Pre-training'],
    correctAnswer: 1,
    explanation: 'Fine-tuning adapts pre-trained weights to specific tasks by training on labeled task data.'
  },
  {
    id: 'ftp2',
    question: 'What is prompting?',
    options: ['Training model', 'Providing instructions/examples in input to guide model behavior without updating weights', 'Fine-tuning', 'Pre-training'],
    correctAnswer: 1,
    explanation: 'Prompting uses carefully designed input text to elicit desired behavior from frozen pre-trained models.'
  },
  {
    id: 'ftp3',
    question: 'What are advantages of fine-tuning?',
    options: ['No advantages', 'Best task performance, adapts weights to task-specific patterns', 'Faster than prompting', 'No data needed'],
    correctAnswer: 1,
    explanation: 'Fine-tuning typically achieves highest accuracy by specializing model weights for the target task.'
  },
  {
    id: 'ftp4',
    question: 'What are disadvantages of fine-tuning?',
    options: ['No disadvantages', 'Requires labeled data, compute for training, separate model per task', 'Too fast', 'Too simple'],
    correctAnswer: 1,
    explanation: 'Fine-tuning needs training data, GPU time, and creates task-specific model copies.'
  },
  {
    id: 'ftp5',
    question: 'What are advantages of prompting?',
    options: ['Best accuracy', 'No training needed, flexible, one model for many tasks', 'Requires data', 'Needs GPUs'],
    correctAnswer: 1,
    explanation: 'Prompting needs no training, works with small or no examples, and uses one shared model.'
  },
  {
    id: 'ftp6',
    question: 'What are disadvantages of prompting?',
    options: ['No disadvantages', 'Lower accuracy than fine-tuning, sensitive to prompt wording, requires large models', 'Too accurate', 'Too stable'],
    correctAnswer: 1,
    explanation: 'Prompting performance varies with phrasing, may underperform fine-tuning, and needs very large models.'
  },
  {
    id: 'ftp7',
    question: 'What is zero-shot prompting?',
    options: ['With examples', 'Task instruction only, no examples', 'Fine-tuning', 'Many examples'],
    correctAnswer: 1,
    explanation: 'Zero-shot: describe task in natural language without providing any examples.'
  },
  {
    id: 'ftp8',
    question: 'What is few-shot prompting?',
    options: ['No examples', 'Providing few examples (typically 1-10) in the prompt', 'Fine-tuning', 'Many examples'],
    correctAnswer: 1,
    explanation: 'Few-shot: include several input-output examples in prompt to demonstrate the task.'
  },
  {
    id: 'ftp9',
    question: 'What is in-context learning?',
    options: ['Fine-tuning', 'Model learns task from prompt context without weight updates', 'Pre-training', 'No learning'],
    correctAnswer: 1,
    explanation: 'In-context learning means the model adapts behavior based on prompt examples at inference time.'
  },
  {
    id: 'ftp10',
    question: 'What is prompt engineering?',
    options: ['Random prompts', 'Systematic design and optimization of prompts for best performance', 'Fine-tuning', 'No engineering'],
    correctAnswer: 1,
    explanation: 'Prompt engineering involves crafting, testing, and refining prompts to maximize model performance.'
  },
  {
    id: 'ftp11',
    question: 'What is instruction tuning?',
    options: ['Prompting only', 'Fine-tuning on diverse tasks formatted as instructions', 'No tuning', 'Pre-training'],
    correctAnswer: 1,
    explanation: 'Instruction tuning fine-tunes on many tasks phrased as instructions, improving zero-shot instruction following.'
  },
  {
    id: 'ftp12',
    question: 'What is FLAN?',
    options: ['Prompting method', 'Fine-tuned LAnguage Net: instruction tuning across 60+ tasks', 'Model architecture', 'Dataset'],
    correctAnswer: 1,
    explanation: 'FLAN applies instruction tuning to LLMs, dramatically improving zero-shot performance on new instructions.'
  },
  {
    id: 'ftp13',
    question: 'What is parameter-efficient fine-tuning (PEFT)?',
    options: ['Full fine-tuning', 'Updating only small subset of parameters or adding small trainable modules', 'No training', 'Pre-training'],
    correctAnswer: 1,
    explanation: 'PEFT methods (LoRA, adapters, prompt tuning) update fewer parameters, reducing compute and memory.'
  },
  {
    id: 'ftp14',
    question: 'What is LoRA?',
    options: ['Full fine-tuning', 'Low-Rank Adaptation: adds trainable low-rank matrices to frozen weights', 'Prompting', 'Architecture'],
    correctAnswer: 1,
    explanation: 'LoRA freezes pre-trained weights and trains low-rank decomposition matrices, reducing trainable parameters dramatically.'
  },
  {
    id: 'ftp15',
    question: 'What are adapter layers?',
    options: ['Full layers', 'Small trainable modules inserted between frozen Transformer layers', 'No layers', 'Entire model'],
    correctAnswer: 1,
    explanation: 'Adapters add small bottleneck layers to frozen models, allowing task adaptation with few parameters.'
  },
  {
    id: 'ftp16',
    question: 'What is prompt tuning?',
    options: ['Manual prompting', 'Learning continuous prompt embeddings while keeping model frozen', 'Fine-tuning all', 'No tuning'],
    correctAnswer: 1,
    explanation: 'Prompt tuning learns task-specific "soft prompts" (continuous vectors) prepended to inputs, freezing the LLM.'
  },
  {
    id: 'ftp17',
    question: 'When should you fine-tune vs prompt?',
    options: ['Always fine-tune', 'Fine-tune for max accuracy with data; prompt for flexibility without training', 'Always prompt', 'No difference'],
    correctAnswer: 1,
    explanation: 'Fine-tune when you have data and need best accuracy; prompt for quick iteration or limited data.'
  },
  {
    id: 'ftp18',
    question: 'Can you combine fine-tuning and prompting?',
    options: ['No', 'Yes, instruction tuning then prompting, or fine-tune then use prompts', 'Mutually exclusive', 'Never beneficial'],
    correctAnswer: 1,
    explanation: 'Models can be instruction-tuned for better prompt following, then used with task-specific prompts.'
  },
  {
    id: 'ftp19',
    question: 'What is catastrophic forgetting in fine-tuning?',
    options: ['No issue', 'Model forgets pre-trained knowledge when fine-tuning aggressively', 'Better memory', 'No forgetting'],
    correctAnswer: 1,
    explanation: 'Excessive fine-tuning can degrade general capabilities; use small learning rates and regularization to mitigate.'
  },
  {
    id: 'ftp20',
    question: 'What is the trend in modern LLMs?',
    options: ['More fine-tuning', 'Shift toward larger models with better prompting, less fine-tuning', 'No prompting', 'No change'],
    correctAnswer: 1,
    explanation: 'Modern very large LLMs (GPT-4, Claude) emphasize strong in-context learning, reducing need for task-specific fine-tuning.'
  }
];

// Large Language Models (LLMs) - 25 questions
export const llmQuestions: QuizQuestion[] = [
  {
    id: 'llm1',
    question: 'What defines a Large Language Model?',
    options: ['Small models', 'Transformer-based models with billions of parameters trained on massive text', 'Rule-based systems', '< 100M parameters'],
    correctAnswer: 1,
    explanation: 'LLMs are typically Transformer models with billions to trillions of parameters, pre-trained on vast text corpora.'
  },
  {
    id: 'llm2',
    question: 'What is an emergent ability of LLMs?',
    options: ['Present in small models', 'Capability appearing only above certain scale threshold', 'Always present', 'Decreases with scale'],
    correctAnswer: 1,
    explanation: 'Emergent abilities like chain-of-thought reasoning emerge suddenly in sufficiently large models.'
  },
  {
    id: 'llm3',
    question: 'What are examples of LLMs?',
    options: ['Word2Vec', 'GPT-3/4, PaLM, LLaMA, Claude, Gemini', 'BERT-base', 'Small RNNs'],
    correctAnswer: 1,
    explanation: 'Modern LLMs include GPT series, Google\'s PaLM/Gemini, Meta\'s LLaMA, Anthropic\'s Claude, etc.'
  },
  {
    id: 'llm4',
    question: 'How many parameters does GPT-3 have?',
    options: ['1.5B', '175B', '340M', '1T'],
    correctAnswer: 1,
    explanation: 'GPT-3 has 175 billion parameters; GPT-4 is larger but size undisclosed.'
  },
  {
    id: 'llm5',
    question: 'What is the typical training data size for LLMs?',
    options: ['MB', 'Hundreds of GB to TB (billions to trillions of tokens)', 'KB', '1 GB'],
    correctAnswer: 1,
    explanation: 'LLMs train on massive datasets: GPT-3 used ~500B tokens, modern models use trillions.'
  },
  {
    id: 'llm6',
    question: 'What is instruction following in LLMs?',
    options: ['Ignoring instructions', 'Ability to understand and execute natural language instructions', 'Only code', 'No understanding'],
    correctAnswer: 1,
    explanation: 'Instruction-tuned LLMs can follow diverse natural language commands without task-specific training.'
  },
  {
    id: 'llm7',
    question: 'What is chain-of-thought (CoT) reasoning?',
    options: ['Direct answer', 'Step-by-step reasoning process shown before final answer', 'No reasoning', 'Random steps'],
    correctAnswer: 1,
    explanation: 'CoT prompting ("let\'s think step by step") improves reasoning by having model show intermediate steps.'
  },
  {
    id: 'llm8',
    question: 'What is hallucination in LLMs?',
    options: ['Perfect accuracy', 'Generating plausible-sounding but incorrect or fabricated information', 'No errors', 'Only truth'],
    correctAnswer: 1,
    explanation: 'LLMs sometimes confidently generate false facts, a major challenge requiring verification and mitigation.'
  },
  {
    id: 'llm9',
    question: 'What causes hallucinations?',
    options: ['Perfect training', 'Training data gaps, lack of grounding, optimization for fluency over factuality', 'No causes', 'Small models'],
    correctAnswer: 1,
    explanation: 'Hallucinations arise from incomplete knowledge, lack of external grounding, and language modeling objective.'
  },
  {
    id: 'llm10',
    question: 'What is Retrieval-Augmented Generation (RAG)?',
    options: ['LLM alone', 'Combining LLM with external knowledge retrieval', 'No retrieval', 'Fine-tuning only'],
    correctAnswer: 1,
    explanation: 'RAG retrieves relevant documents for queries, then uses LLM to generate answers grounded in retrieved text.'
  },
  {
    id: 'llm11',
    question: 'What is the context window?',
    options: ['Training data', 'Maximum number of tokens model can process at once', 'Parameter count', 'Layer depth'],
    correctAnswer: 1,
    explanation: 'Context window is the maximum sequence length (e.g., 4K, 32K, 128K tokens) model can attend to.'
  },
  {
    id: 'llm12',
    question: 'What are longer context windows useful for?',
    options: ['Nothing', 'Processing long documents, maintaining conversation history, few-shot examples', 'Only short text', 'Slows model'],
    correctAnswer: 1,
    explanation: 'Longer contexts enable reasoning over full documents, richer conversations, and more in-context examples.'
  },
  {
    id: 'llm13',
    question: 'What is model alignment?',
    options: ['Physical alignment', 'Making model behavior match human values and intentions', 'No alignment', 'Data preprocessing'],
    correctAnswer: 1,
    explanation: 'Alignment uses techniques like RLHF to make models helpful, honest, and harmless.'
  },
  {
    id: 'llm14',
    question: 'What is RLHF in LLMs?',
    options: ['Pre-training', 'Reinforcement Learning from Human Feedback: trains reward model, then optimizes LLM', 'No feedback', 'Supervised only'],
    correctAnswer: 1,
    explanation: 'RLHF collects human preferences, trains reward model, then uses PPO to fine-tune LLM to maximize reward.'
  },
  {
    id: 'llm15',
    question: 'What are constitutional AI methods?',
    options: ['Random rules', 'Using principles and self-critique to align AI without extensive human feedback', 'No principles', 'RLHF only'],
    correctAnswer: 1,
    explanation: 'Constitutional AI (Anthropic) uses written principles and AI self-evaluation to improve alignment.'
  },
  {
    id: 'llm16',
    question: 'What is model compression for LLMs?',
    options: ['Making models larger', 'Techniques to reduce model size: quantization, pruning, distillation', 'No compression', 'Only expansion'],
    correctAnswer: 1,
    explanation: 'Compression reduces LLM size for deployment: quantization (int8, int4), pruning (remove weights), distillation (train smaller model).'
  },
  {
    id: 'llm17',
    question: 'What is quantization?',
    options: ['Increasing precision', 'Reducing parameter precision (e.g., float32 → int8) to save memory', 'No change', 'Training method'],
    correctAnswer: 1,
    explanation: 'Quantization converts weights to lower precision (8-bit, 4-bit), reducing memory 4-8× with minimal accuracy loss.'
  },
  {
    id: 'llm18',
    question: 'What are open-source LLMs?',
    options: ['Proprietary only', 'Publicly available models: LLaMA, Falcon, MPT, Mistral', 'No open models', 'All closed'],
    correctAnswer: 1,
    explanation: 'Open LLMs like Meta\'s LLaMA, Mistral, Falcon enable research and deployment without API costs.'
  },
  {
    id: 'llm19',
    question: 'What is the difference between GPT-3.5 and GPT-4?',
    options: ['No difference', 'GPT-4 is larger, multimodal, more accurate, better reasoning', 'Same size', 'GPT-3.5 is better'],
    correctAnswer: 1,
    explanation: 'GPT-4 has better reasoning, fewer errors, multimodal vision, and longer context than GPT-3.5.'
  },
  {
    id: 'llm20',
    question: 'What are mixture-of-experts (MoE) models?',
    options: ['Single expert', 'Model with multiple "expert" networks, routing tokens to relevant experts', 'No experts', 'All experts active'],
    correctAnswer: 1,
    explanation: 'MoE activates only subset of parameters per token, enabling larger total capacity with similar compute.'
  },
  {
    id: 'llm21',
    question: 'What is the cost of training LLMs?',
    options: ['Free', 'Millions to tens of millions of dollars in compute', 'A few dollars', '$100'],
    correctAnswer: 1,
    explanation: 'Training GPT-3 scale models costs $1M-10M+; GPT-4 likely cost tens of millions in compute.'
  },
  {
    id: 'llm22',
    question: 'What is inference cost for LLMs?',
    options: ['Free', 'Significant: powerful GPUs needed, costs scale with length and usage', 'Negligible', 'One-time'],
    correctAnswer: 1,
    explanation: 'Running LLM inference requires expensive GPUs, with costs per API call adding up at scale.'
  },
  {
    id: 'llm23',
    question: 'What is multi-modal LLM?',
    options: ['Text only', 'LLM processing multiple modalities: text, images, audio, video', 'Single mode', 'No media'],
    correctAnswer: 1,
    explanation: 'Multi-modal LLMs like GPT-4, Gemini can understand and generate across text, images, and more.'
  },
  {
    id: 'llm24',
    question: 'What are LLM safety concerns?',
    options: ['No concerns', 'Misuse, bias, misinformation, privacy, alignment challenges', 'Perfectly safe', 'Only technical'],
    correctAnswer: 1,
    explanation: 'Safety issues include harmful content generation, bias amplification, privacy risks, and potential misuse.'
  },
  {
    id: 'llm25',
    question: 'What is the future trend for LLMs?',
    options: ['Stagnation', 'Continued scaling, better alignment, efficiency, multi-modality, reasoning', 'Decline', 'No change'],
    correctAnswer: 1,
    explanation: 'LLMs evolving toward: larger scale, multi-modal, more efficient, better aligned, enhanced reasoning capabilities.'
  }
];
