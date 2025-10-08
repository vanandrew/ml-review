import { Topic } from '../../../types';

export const bert: Topic = {
  id: 'bert',
  title: 'BERT (Bidirectional Encoder Representations from Transformers)',
  category: 'transformers',
  description: 'Bidirectional pre-training for language understanding tasks',
  content: `
    <h2>BERT: Bidirectional Encoder Representations from Transformers</h2>
    <p>BERT (2018) revolutionized NLP by introducing truly bidirectional pre-training through Masked Language Modeling. Unlike previous models (GPT reads left-to-right, ELMo concatenates separate left-to-right and right-to-left models), BERT reads text in both directions simultaneously through a single model. This breakthrough enabled unprecedented performance gains across diverse NLP tasks and established the pre-training + fine-tuning paradigm that dominates modern NLP.</p>

    <h3>The Bidirectional Revolution</h3>
    
    <h4>Why Bidirectionality Matters</h4>
    <ul>
      <li><strong>Full context understanding:</strong> To understand "bank" in "I deposited money at the bank," we need both left context ("deposited money") and right context (financial institution, not river bank)</li>
      <li><strong>Cloze task analogy:</strong> Humans naturally use surrounding context from both directions to understand ambiguous words</li>
      <li><strong>Richer representations:</strong> Each token representation incorporates information from entire sequence, not just preceding tokens</li>
      <li><strong>Better for understanding:</strong> Classification, QA, NER benefit more from bidirectional context than generation tasks</li>
    </ul>

    <h4>The Challenge: Traditional Pre-training is Unidirectional</h4>
    <ul>
      <li><strong>Language modeling limitation:</strong> Standard LM predicts next word, requiring causal (left-to-right) attention to prevent "cheating"</li>
      <li><strong>GPT approach:</strong> Decoder-only with causal masking—only sees previous tokens</li>
      <li><strong>ELMo approach:</strong> Train separate left-to-right and right-to-left LSTMs, concatenate—not truly joint</li>
      <li><strong>BERT's insight:</strong> Use Masked Language Modeling to enable bidirectional pre-training</li>
    </ul>

    <h3>Masked Language Modeling: Enabling Bidirectional Pre-training</h3>
    <p>MLM is the key innovation that allows BERT to be bidirectional. Instead of predicting the next word, MLM randomly masks some tokens and trains the model to predict the masked tokens using full bidirectional context.</p>

    <h4>The Masking Strategy</h4>
    <p><strong>Process:</strong> Randomly select 15% of tokens for masking, then:</p>
    <ul>
      <li><strong>80% of the time:</strong> Replace with [MASK] token. Example: "The cat sat on the [MASK]" $\\to$ predict "mat"</li>
      <li><strong>10% of the time:</strong> Replace with random token. Example: "The cat sat on the apple" $\\to$ predict "mat"</li>
      <li><strong>10% of the time:</strong> Keep original token unchanged. Example: "The cat sat on the mat" $\\to$ predict "mat"</li>
    </ul>

    <h4>Why This Complex Strategy?</h4>
    <ul>
      <li><strong>Train-test mismatch problem:</strong> [MASK] token appears during pre-training but never during fine-tuning. If model only sees [MASK], it might learn patterns specific to [MASK] that don't transfer</li>
      <li><strong>Random token replacement (10%):</strong> Forces model to use context rather than memorize token identity. Can't rely on current token being correct</li>
      <li><strong>Unchanged tokens (10%):</strong> Model must maintain representations for non-masked tokens, bridging pre-training and fine-tuning</li>
      <li><strong>80% [MASK] majority:</strong> Still provides strong training signal while mitigating train-test mismatch</li>
    </ul>

    <h4>MLM Training Objective</h4>
    <p>For masked token at position i, predict original token using cross-entropy loss:</p>
    <p><strong>Loss = -log P(token_i | context)</strong></p>
    <p>Where context includes all other tokens (both left and right), and P is computed via softmax over vocabulary.</p>

    <h4>Why MLM Enables Bidirectionality</h4>
    <ul>
      <li><strong>No causality constraint:</strong> Masked tokens are known to be masked, so no "cheating" by seeing future tokens</li>
      <li><strong>Full attention:</strong> Each position can attend to all positions without restrictions</li>
      <li><strong>Cloze task parallel:</strong> Similar to how humans use full context to fill in blanks</li>
      <li><strong>Rich signal:</strong> Every masked position provides training signal from full sequence</li>
    </ul>

    <h3>Next Sentence Prediction (NSP): Learning Discourse Relationships</h3>
    <p>MLM captures token-level understanding, but many NLP tasks (QA, NLI) require understanding relationships between sentences. NSP addresses this.</p>

    <h4>NSP Task Design</h4>
    <ul>
      <li><strong>Input format:</strong> [CLS] Sentence A [SEP] Sentence B [SEP]</li>
      <li><strong>Positive examples (50%):</strong> B actually follows A in original text (IsNext label)</li>
      <li><strong>Negative examples (50%):</strong> B is random sentence from corpus (NotNext label)</li>
      <li><strong>Objective:</strong> Binary classification using [CLS] representation</li>
    </ul>

    <h4>Purpose and Effectiveness</h4>
    <ul>
      <li><strong>Discourse coherence:</strong> Learn relationships between sentences, not just within sentences</li>
      <li><strong>Sentence pair tasks:</strong> Directly applicable to NLI, QA, paraphrase detection</li>
      <li><strong>Later findings:</strong> RoBERTa showed NSP may not be necessary—document-level MLM may suffice</li>
      <li><strong>Trade-off:</strong> Adds complexity but marginal benefit; most successors omit NSP</li>
    </ul>

    <h3>BERT Architecture: Encoder-Only Transformer</h3>

    <h4>Model Configurations</h4>
    <table border="1" cellpadding="8" style="border-collapse: collapse; width: 100%;">
      <tr>
        <th>Configuration</th>
        <th>Layers (L)</th>
        <th>Hidden Size (H)</th>
        <th>Attention Heads (A)</th>
        <th>Parameters</th>
      </tr>
      <tr>
        <td>BERT-Base</td>
        <td>12</td>
        <td>768</td>
        <td>12</td>
        <td>110M</td>
      </tr>
      <tr>
        <td>BERT-Large</td>
        <td>24</td>
        <td>1024</td>
        <td>16</td>
        <td>340M</td>
      </tr>
    </table>

    <h4>Input Representation</h4>
    <p>BERT's input is sum of three embeddings:</p>
    <ul>
      <li><strong>Token embeddings:</strong> WordPiece vocabulary (30K tokens), handles out-of-vocabulary through subwords</li>
      <li><strong>Segment embeddings:</strong> Distinguish sentence A from sentence B (learned embeddings for segment 0 vs 1)</li>
      <li><strong>Position embeddings:</strong> Learned absolute positional embeddings (0-511)</li>
    </ul>
    <p><strong>Final input = Token_Emb + Segment_Emb + Position_Emb</strong></p>

    <h4>Special Tokens</h4>
    <ul>
      <li><strong>[CLS]:</strong> Classification token at start. Final hidden state used for sequence-level classification tasks</li>
      <li><strong>[SEP]:</strong> Separator token between sentences and at end</li>
      <li><strong>[MASK]:</strong> Mask token used during MLM pre-training</li>
      <li><strong>[PAD]:</strong> Padding token for variable-length sequences in batches</li>
    </ul>

    <h4>Why Encoder-Only?</h4>
    <ul>
      <li><strong>Bidirectional attention:</strong> Encoder allows attending to full context, decoder requires causal masking</li>
      <li><strong>Understanding focus:</strong> BERT designed for tasks requiring understanding, not generation</li>
      <li><strong>Efficiency:</strong> Simpler architecture without cross-attention and decoder complexity</li>
      <li><strong>Task alignment:</strong> Classification, QA, NER don't need autoregressive generation</li>
    </ul>

    <h3>Pre-training Details</h3>

    <h4>Training Data</h4>
    <ul>
      <li><strong>BooksCorpus:</strong> 800M words from unpublished books (diverse, long-form text)</li>
      <li><strong>English Wikipedia:</strong> 2,500M words (factual, diverse topics)</li>
      <li><strong>Total:</strong> ~3.3B words, providing rich linguistic patterns</li>
      <li><strong>Preprocessing:</strong> Sentence segmentation, WordPiece tokenization</li>
    </ul>

    <h4>Training Configuration</h4>
    <ul>
      <li><strong>Batch size:</strong> 256 sequences (BERT-Base), 16 TPUs for 4 days</li>
      <li><strong>Sequence length:</strong> Maximum 512 tokens (90% of examples use 128 for efficiency)</li>
      <li><strong>Optimizer:</strong> Adam with learning rate warmup and linear decay</li>
      <li><strong>Learning rate:</strong> 1e-4 with 10K warmup steps</li>
      <li><strong>Dropout:</strong> 0.1 on all layers and attention</li>
    </ul>

    <h3>Fine-tuning BERT for Downstream Tasks</h3>

    <h4>Classification Tasks</h4>
    <ul>
      <li><strong>Approach:</strong> Add linear layer on top of [CLS] representation: $\\text{output} = \\text{softmax}(W \\cdot \\text{[CLS]} + b)$</li>
      <li><strong>Examples:</strong> Sentiment analysis, topic classification, spam detection</li>
      <li><strong>Fine-tuning:</strong> Update all BERT parameters + classification head</li>
      <li><strong>Typical setup:</strong> 2-4 epochs, learning rate 2e-5 to 5e-5, small batch size (16-32)</li>
    </ul>

    <h4>Named Entity Recognition (NER)</h4>
    <ul>
      <li><strong>Approach:</strong> Token-level classification. Add linear layer on each token representation</li>
      <li><strong>Output:</strong> Predict entity label (B-PER, I-ORG, O, etc.) for each token</li>
      <li><strong>Handling WordPiece:</strong> Use first subword representation, or average subword representations</li>
      <li><strong>Loss:</strong> Cross-entropy over all token positions</li>
    </ul>

    <h4>Question Answering (SQuAD)</h4>
    <ul>
      <li><strong>Input:</strong> [CLS] Question [SEP] Context [SEP]</li>
      <li><strong>Approach:</strong> Predict start and end positions of answer span in context</li>
      <li><strong>Output layers:</strong> Two linear layers (start and end) applied to each token representation</li>
      <li><strong>Training:</strong> Maximize log-likelihood of correct start and end positions</li>
    </ul>

    <h4>Sentence Pair Tasks (NLI, Paraphrase, Similarity)</h4>
    <ul>
      <li><strong>Input:</strong> [CLS] Sentence A [SEP] Sentence B [SEP]</li>
      <li><strong>Approach:</strong> Classification on [CLS] representation</li>
      <li><strong>Examples:</strong> Entailment (3 classes), paraphrase detection (binary), semantic similarity (regression)</li>
    </ul>

    <h3>BERT's Impact and Performance</h3>

    <h4>Benchmark Results (at release, 2018)</h4>
    <ul>
      <li><strong>GLUE benchmark:</strong> 80.5% average (7.7% improvement over previous SOTA)</li>
      <li><strong>SQuAD v1.1:</strong> F1 93.2% (1.5% improvement, matching human performance)</li>
      <li><strong>SQuAD v2.0:</strong> F1 83.1% (5.1% improvement)</li>
      <li><strong>SWAG:</strong> 86.3% (27.1% improvement)</li>
      <li><strong>Dominated 11 NLP tasks:</strong> Established new state-of-the-art across diverse benchmarks</li>
    </ul>

    <h4>Why Such Strong Performance?</h4>
    <ul>
      <li><strong>Bidirectional context:</strong> Richer representations than unidirectional models</li>
      <li><strong>Large-scale pre-training:</strong> Learned robust linguistic patterns from 3.3B words</li>
      <li><strong>Transfer learning:</strong> Pre-trained representations adapt efficiently to downstream tasks</li>
      <li><strong>Architecture scaling:</strong> Deep Transformers (12-24 layers) capture complex patterns</li>
    </ul>

    <h3>BERT Variants and Successors</h3>

    <h4>RoBERTa (Robustly Optimized BERT)</h4>
    <ul>
      <li><strong>Key changes:</strong> Remove NSP, train longer (500K steps), larger batches (8K), more data (160GB text)</li>
      <li><strong>Dynamic masking:</strong> Change masking pattern every epoch instead of static masks</li>
      <li><strong>Byte-level BPE:</strong> Better handling of rare words</li>
      <li><strong>Result:</strong> Outperforms BERT on most benchmarks, showing pre-training recipe matters</li>
    </ul>

    <h4>ALBERT (A Lite BERT)</h4>
    <ul>
      <li><strong>Factorized embeddings:</strong> Decompose large vocabulary embedding into two smaller matrices ($V \\times H \\to V \\times E + E \\times H$ where $E \\ll H$)</li>
      <li><strong>Cross-layer parameter sharing:</strong> Share parameters across all layers (especially feedforward and attention)</li>
      <li><strong>SOP instead of NSP:</strong> Sentence Order Prediction (predict if sentences are swapped) instead of NSP</li>
      <li><strong>Result:</strong> $18\\times$ fewer parameters than BERT-Large with comparable performance</li>
    </ul>

    <h4>DistilBERT</h4>
    <ul>
      <li><strong>Approach:</strong> Knowledge distillation from BERT-Base teacher to smaller student model</li>
      <li><strong>Architecture:</strong> 6 layers (vs 12), 66M parameters (vs 110M)</li>
      <li><strong>Performance:</strong> Retains 97% of BERT's performance while being 40% smaller and 60% faster</li>
      <li><strong>Use case:</strong> Production deployment where inference speed and memory matter</li>
    </ul>

    <h4>DeBERTa (Decoding-enhanced BERT)</h4>
    <ul>
      <li><strong>Disentangled attention:</strong> Separate content and position attention mechanisms</li>
      <li><strong>Enhanced mask decoder:</strong> Incorporate absolute positions when predicting masked tokens</li>
      <li><strong>Virtual adversarial training:</strong> Improve model robustness</li>
      <li><strong>Result:</strong> State-of-the-art on SuperGLUE and other benchmarks</li>
    </ul>

    <h3>Advantages of BERT</h3>
    <ul>
      <li><strong>True bidirectionality:</strong> Full context from both directions in single model</li>
      <li><strong>Transfer learning paradigm:</strong> Pre-train once on large corpus, fine-tune for many tasks with minimal data</li>
      <li><strong>Performance gains:</strong> Massive improvements (5-30%) on diverse NLP benchmarks</li>
      <li><strong>Interpretability:</strong> Attention weights reveal syntactic and semantic patterns</li>
      <li><strong>Versatility:</strong> Single architecture adaptable to classification, QA, NER, sentence pairs</li>
      <li><strong>Open source:</strong> Pre-trained models publicly available, democratizing NLP</li>
    </ul>

    <h3>Limitations and Challenges</h3>
    <ul>
      <li><strong>Computational cost:</strong> Large model (110M-340M parameters), expensive pre-training and inference</li>
      <li><strong>Memory intensive:</strong> Requires significant GPU memory (16GB+ for BERT-Large)</li>
      <li><strong>Not generative:</strong> Encoder-only architecture cannot generate text autoregressively</li>
      <li><strong>Fixed sequence length:</strong> Maximum 512 tokens limits long document processing</li>
      <li><strong>WordPiece artifacts:</strong> Subword tokenization can split meaningful units awkwardly</li>
      <li><strong>Training data bias:</strong> Inherits biases from pre-training corpora</li>
      <li><strong>Fine-tuning required:</strong> Cannot do zero-shot or few-shot learning like GPT-3</li>
    </ul>

    <h3>BERT's Legacy</h3>
    <p>BERT revolutionized NLP by demonstrating that bidirectional pre-training with Transformers could achieve unprecedented performance across diverse tasks. It established the pre-training + fine-tuning paradigm that became the standard approach in NLP. While GPT models later showed the power of scaling and few-shot learning, BERT's insights about bidirectionality and masked language modeling remain fundamental. Modern models like RoBERTa, DeBERTa, and encoder components of T5 and BART build directly on BERT's innovations, cementing its place as a pivotal breakthrough in NLP history.</p>
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
};
