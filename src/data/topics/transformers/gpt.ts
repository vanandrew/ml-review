import { Topic } from '../../../types';

export const gpt: Topic = {
  id: 'gpt',
  title: 'GPT (Generative Pre-trained Transformer)',
  category: 'transformers',
  description: 'Autoregressive language models for text generation',
  content: `
    <h2>GPT: Generative Pre-trained Transformer</h2>
    <p>The GPT series (GPT-1 2018, GPT-2 2019, GPT-3 2020, GPT-4 2023) represents a fundamentally different approach to Transformers compared to BERT. While BERT uses bidirectional encoding for understanding tasks, GPT uses unidirectional decoding for generation. GPT models are autoregressive language models that predict the next token given all previous tokens, enabling natural text generation while also achieving strong performance on downstream tasks through few-shot learning. The evolution from GPT-1's 117M parameters to GPT-3's 175B parameters revealed emergent capabilities that transformed AI applications.</p>

    <h3>Autoregressive Language Modeling: The Core Paradigm</h3>
    
    <h4>The Language Modeling Objective</h4>
    <p><strong>Goal:</strong> Maximize the probability of the next token given all previous tokens:</p>
    <p>$$P(x_t | x_1, x_2, \\ldots, x_{t-1})$$</p>
    <p>For a sequence of length $n$, maximize joint probability: $$P(x_1, \\ldots, x_n) = \\prod_{t=1}^{n} P(x_t | x_1, \\ldots, x_{t-1})$$</p>

    <h4>Why Autoregressive Generation?</h4>
    <ul>
      <li><strong>Natural text generation:</strong> Humans write/speak sequentially, one word at a time</li>
      <li><strong>Self-supervised learning:</strong> Every position provides training signal—no labels needed</li>
      <li><strong>Versatility:</strong> Generation subsumes many tasks (completion, translation, summarization, Q&A via prompting)</li>
      <li><strong>Scalability:</strong> Can train on entire internet, unlimited raw text data</li>
      <li><strong>Coherent generation:</strong> Maintains causal consistency—generated text follows from context</li>
    </ul>

    <h4>Causal Masking: Preventing Information Leakage</h4>
    <ul>
      <li><strong>Requirement:</strong> Position t can only attend to positions 1...t, never t+1...n</li>
      <li><strong>Implementation:</strong> Mask attention scores for future positions with -∞ before softmax</li>
      <li><strong>Training-inference consistency:</strong> Same constraints during training and generation</li>
      <li><strong>Contrast with BERT:</strong> BERT sees full context (bidirectional), GPT sees only past (unidirectional)</li>
    </ul>

    <h3>Decoder-Only Architecture</h3>

    <h4>Why Decoder-Only, Not Encoder-Decoder?</h4>
    <ul>
      <li><strong>Simplicity:</strong> Single stack instead of separate encoder and decoder</li>
      <li><strong>Unified processing:</strong> Input and output in same representation space</li>
      <li><strong>Scalability:</strong> Easier to scale to massive sizes without cross-attention complexity</li>
      <li><strong>Flexibility:</strong> Can handle variable-length inputs and outputs naturally</li>
      <li><strong>Historical note:</strong> Original Transformer used encoder-decoder for translation; GPT showed decoder-only suffices for many tasks</li>
    </ul>

    <h4>Architectural Components</h4>
    <ul>
      <li><strong>Causal self-attention:</strong> Multi-head attention with future masking</li>
      <li><strong>Layer normalization:</strong> Pre-norm (before attention/FFN) for training stability</li>
      <li><strong>Position embeddings:</strong> Learned absolute positions (GPT-1/2) or ALiBi/RoPE (modern variants)</li>
      <li><strong>Feedforward networks:</strong> Position-wise FFN with GELU activation (vs ReLU)</li>
      <li><strong>Output layer:</strong> Linear projection + softmax over vocabulary for token prediction</li>
    </ul>

    <h3>GPT Evolution: From 117M to 175B+ Parameters</h3>

    <h4>GPT-1 (2018): Proof of Concept</h4>
    <ul>
      <li><strong>Size:</strong> 117M parameters, 12 layers, 768 hidden dimensions, 12 attention heads</li>
      <li><strong>Training data:</strong> BooksCorpus (7,000 books, ~1GB text)</li>
      <li><strong>Key insight:</strong> Pre-training on language modeling + task-specific fine-tuning beats training from scratch</li>
      <li><strong>Performance:</strong> SOTA on 9 of 12 tasks with minimal fine-tuning</li>
      <li><strong>Paradigm:</strong> Pre-train on unlabeled text, fine-tune on labeled task data</li>
    </ul>

    <h4>GPT-2 (2019): Scaling and Zero-Shot</h4>
    <ul>
      <li><strong>Sizes:</strong> 117M (small), 345M (medium), 762M (large), 1.5B (XL)</li>
      <li><strong>Training data:</strong> WebText (8M web pages, 40GB text) - higher quality than random scraping</li>
      <li><strong>Key innovation:</strong> Strong zero-shot performance without fine-tuning by framing tasks as text generation</li>
      <li><strong>Controversy:</strong> Initially not released due to misuse concerns (fake news generation)</li>
      <li><strong>Findings:</strong> Larger models continue improving; no clear saturation</li>
    </ul>

    <h4>GPT-3 (2020): Emergent Few-Shot Learning</h4>
    <ul>
      <li><strong>Sizes:</strong> 125M, 350M, 760M, 1.3B, 2.7B, 6.7B, 13B, 175B parameters</li>
      <li><strong>Largest variant:</strong> 175B parameters, 96 layers, 12,288 hidden dimensions, 96 attention heads</li>
      <li><strong>Training data:</strong> 300B tokens from Common Crawl, WebText2, Books, Wikipedia (570GB text)</li>
      <li><strong>Training cost:</strong> ~$4.6M in compute, months on thousands of GPUs</li>
      <li><strong>Breakthrough:</strong> Strong few-shot learning—provide 3-10 examples in prompt, model performs task without weight updates</li>
      <li><strong>Emergent abilities:</strong> Arithmetic, word unscrambling, novel word usage not explicitly in training data</li>
      <li><strong>Limitations:</strong> Still makes factual errors, lacks reasoning, prone to biases</li>
    </ul>

    <h4>GPT-4 (2023): Multimodal and Enhanced</h4>
    <ul>
      <li><strong>Size:</strong> Undisclosed (estimated 1T+ parameters, mixture of experts architecture)</li>
      <li><strong>Multimodal:</strong> Accepts both text and images as input</li>
      <li><strong>Improvements:</strong> Better reasoning, fewer hallucinations, improved alignment</li>
      <li><strong>Context length:</strong> 8K tokens (standard) to 32K tokens (extended)</li>
      <li><strong>Safety:</strong> RLHF for alignment, adversarial testing, refusal training</li>
      <li><strong>Performance:</strong> Passes bar exam (90th percentile), SAT (1410/1600), solves competition-level problems</li>
    </ul>

    <h3>Scaling Laws: Bigger is Better (Predictably)</h3>
    
    <h4>Empirical Observations</h4>
    <ul>
      <li><strong>Power law relationship:</strong> Loss scales as $L \\propto N^{-\\alpha}$ where $N$ is parameters, $\\alpha \\approx 0.076$</li>
      <li><strong>Compute-optimal scaling:</strong> For compute budget $C$, optimal allocation: $N \\propto C^{0.73}$ and $D \\propto C^{0.27}$ (tokens)</li>
      <li><strong>No saturation:</strong> Performance continues improving with scale, no plateau observed yet</li>
      <li><strong>Sample efficiency:</strong> Larger models need fewer examples for same performance</li>
    </ul>

    <h4>Emergent Capabilities</h4>
    <p>Abilities that appear suddenly at certain scales, not present in smaller models:</p>
    <ul>
      <li><strong>Arithmetic:</strong> 3-digit addition emerges around 13B parameters</li>
      <li><strong>Multi-step reasoning:</strong> Chain-of-thought reasoning emerges with scale</li>
      <li><strong>Instruction following:</strong> Understanding complex multi-part instructions</li>
      <li><strong>Few-shot learning:</strong> Learning new tasks from examples without training</li>
      <li><strong>Analogy and abstraction:</strong> Recognizing patterns and applying to new domains</li>
    </ul>

    <h3>Inference and Generation Strategies</h3>

    <h4>Greedy Decoding</h4>
    <ul>
      <li><strong>Method:</strong> Always select highest probability token: x_t = argmax P(x_t | x_{<t})</li>
      <li><strong>Pros:</strong> Deterministic, fast, reproducible</li>
      <li><strong>Cons:</strong> Repetitive, lacks diversity, can get stuck in loops</li>
      <li><strong>Use case:</strong> When deterministic output desired (code generation, factual answers)</li>
    </ul>

    <h4>Beam Search</h4>
    <ul>
      <li><strong>Method:</strong> Maintain k highest-probability sequences at each step</li>
      <li><strong>Pros:</strong> Higher-quality than greedy, explores multiple paths</li>
      <li><strong>Cons:</strong> Computationally expensive ($k\\times$ cost), still can be repetitive</li>
      <li><strong>Use case:</strong> Machine translation, summarization where quality matters</li>
    </ul>

    <h4>Temperature Sampling</h4>
    <ul>
      <li><strong>Method:</strong> Scale logits by temperature $T$ before softmax: $P(x_t) \\propto \\exp(\\text{logit}_t / T)$</li>
      <li><strong>$T \\to 0$:</strong> Approaches greedy (deterministic)</li>
      <li><strong>$T = 1$:</strong> Sample from original distribution</li>
      <li><strong>$T > 1$:</strong> More uniform, more random</li>
      <li><strong>Use case:</strong> Creative writing (T=0.7-0.9), diverse responses</li>
    </ul>

    <h4>Top-k Sampling</h4>
    <ul>
      <li><strong>Method:</strong> Sample from k most likely tokens, redistribute probability mass</li>
      <li><strong>Benefit:</strong> Prevents sampling very unlikely tokens (errors, nonsense)</li>
      <li><strong>Limitation:</strong> Fixed k doesn't adapt to distribution shape</li>
      <li><strong>Typical k:</strong> 40-50 for balanced creativity and coherence</li>
    </ul>

    <h4>Top-p (Nucleus) Sampling</h4>
    <ul>
      <li><strong>Method:</strong> Sample from smallest set of tokens with cumulative probability $\\geq p$</li>
      <li><strong>Adaptive:</strong> Varies number of tokens based on distribution confidence</li>
      <li><strong>Benefit:</strong> More tokens when uncertain, fewer when confident</li>
      <li><strong>Typical p:</strong> 0.9-0.95 for good balance</li>
      <li><strong>Best practice:</strong> Combine with temperature (p=0.9, T=0.8)</li>
    </ul>

    <h3>Few-Shot Learning and In-Context Learning</h3>

    <h4>The Paradigm Shift</h4>
    <p>Instead of fine-tuning for each task, provide examples in the prompt:</p>

    <h4>Zero-Shot</h4>
    <p><strong>Format:</strong> Task description only</p>
    <p><strong>Example:</strong> "Translate to French: Hello → Bonjour. Translate to French: Goodbye →"</p>
    <p><strong>Performance:</strong> Varies by task; strong for common tasks, weak for niche domains</p>

    <h4>One-Shot and Few-Shot</h4>
    <p><strong>Format:</strong> Task description + k examples</p>
    <p><strong>Example:</strong> "Translate: Hello → Bonjour. Translate: Thank you → Merci. Translate: Goodbye →"</p>
    <p><strong>Performance:</strong> Often matches or exceeds fine-tuned models for GPT-3 175B</p>

    <h4>Why Does In-Context Learning Work?</h4>
    <ul>
      <li><strong>Hypothesis 1:</strong> Model learns tasks during pre-training by seeing examples in natural text</li>
      <li><strong>Hypothesis 2:</strong> Massive model capacity allows meta-learning across diverse tasks</li>
      <li><strong>Hypothesis 3:</strong> Attention mechanism implements implicit gradient descent on examples</li>
      <li><strong>Evidence:</strong> Larger models show stronger in-context learning, suggesting capacity matters</li>
    </ul>

    <h3>GPT vs BERT: Complementary Approaches</h3>

    <table border="1" cellpadding="8" style="border-collapse: collapse; width: 100%;">
      <tr>
        <th>Aspect</th>
        <th>GPT (Decoder)</th>
        <th>BERT (Encoder)</th>
      </tr>
      <tr>
        <td>Attention</td>
        <td>Unidirectional (causal)</td>
        <td>Bidirectional</td>
      </tr>
      <tr>
        <td>Pre-training</td>
        <td>Next-token prediction</td>
        <td>Masked LM + NSP</td>
      </tr>
      <tr>
        <td>Primary use</td>
        <td>Generation, completion</td>
        <td>Understanding, classification</td>
      </tr>
      <tr>
        <td>Task adaptation</td>
        <td>Prompting, few-shot</td>
        <td>Fine-tuning required</td>
      </tr>
      <tr>
        <td>Context</td>
        <td>Past only (x_1...x_t)</td>
        <td>Full sequence (x_1...x_n)</td>
      </tr>
      <tr>
        <td>Generation</td>
        <td>Natural (autoregressive)</td>
        <td>Cannot generate</td>
      </tr>
    </table>

    <h3>Applications and Impact</h3>

    <h4>Text Generation and Completion</h4>
    <ul>
      <li><strong>Creative writing:</strong> Stories, poetry, screenplays</li>
      <li><strong>Content creation:</strong> Blog posts, marketing copy, product descriptions</li>
      <li><strong>Autocomplete:</strong> Email completion, writing assistants</li>
    </ul>

    <h4>Code Generation (Codex, GitHub Copilot)</h4>
    <ul>
      <li><strong>Function completion:</strong> Generate code from docstrings or comments</li>
      <li><strong>Bug fixing:</strong> Suggest fixes for errors</li>
      <li><strong>Code explanation:</strong> Generate comments explaining code</li>
      <li><strong>Multi-language:</strong> Python, JavaScript, Go, etc.</li>
    </ul>

    <h4>Conversational AI (ChatGPT)</h4>
    <ul>
      <li><strong>Customer support:</strong> Answer questions, troubleshoot issues</li>
      <li><strong>Tutoring:</strong> Explain concepts, provide examples</li>
      <li><strong>Brainstorming:</strong> Generate ideas, outline projects</li>
      <li><strong>Role-playing:</strong> Simulate interviews, practice conversations</li>
    </ul>

    <h4>Task Versatility via Prompting</h4>
    <ul>
      <li><strong>Translation:</strong> "Translate to Spanish: ..."</li>
      <li><strong>Summarization:</strong> "Summarize this article: ..."</li>
      <li><strong>Question answering:</strong> "Q: ... A:"</li>
      <li><strong>Classification:</strong> "Classify sentiment: ..."</li>
    </ul>

    <h3>Limitations and Challenges</h3>
    <ul>
      <li><strong>Hallucinations:</strong> Generates plausible but false information confidently</li>
      <li><strong>No grounding:</strong> Cannot verify facts or access external knowledge beyond training</li>
      <li><strong>Reasoning limitations:</strong> Struggles with complex logical reasoning, math</li>
      <li><strong>Context window:</strong> Limited to 2K-32K tokens, cannot process very long documents</li>
      <li><strong>Computational cost:</strong> Inference expensive for 175B+ models</li>
      <li><strong>Biases:</strong> Inherits biases from training data (gender, race, political)</li>
      <li><strong>Safety concerns:</strong> Can generate harmful, offensive, or misleading content</li>
      <li><strong>Training data cutoff:</strong> Knowledge frozen at training time, no awareness of recent events</li>
    </ul>

    <h3>The GPT Revolution</h3>
    <p>GPT models, particularly GPT-3 and GPT-4, fundamentally changed how we think about AI. They demonstrated that scaling language models leads to emergent capabilities and that prompting can replace fine-tuning for many tasks. The shift from task-specific models to general-purpose models that adapt via prompting has transformed AI from a research tool to a widely accessible technology. ChatGPT's release in November 2022 brought AI to mainstream awareness, spawning countless applications and raising important questions about AI safety, alignment, and societal impact. GPT's legacy is the paradigm shift toward large, general-purpose models that can be steered through natural language instructions.</p>
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
};
