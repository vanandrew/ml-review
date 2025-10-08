import { Topic } from '../../../types';

export const t5Bart: Topic = {
  id: 't5-bart',
  title: 'T5 and BART',
  category: 'transformers',
  description: 'Encoder-decoder models for sequence-to-sequence tasks',
  content: `
    <h2>T5 and BART: Encoder-Decoder Transformers for Sequence-to-Sequence Tasks</h2>
    <p>T5 (Text-to-Text Transfer Transformer) and BART (Bidirectional and AutoRegressive Transformers) represent the encoder-decoder branch of Transformer evolution, sitting between BERT (encoder-only) and GPT (decoder-only). These models excel at tasks requiring transformation of input sequences to output sequences: translation, summarization, question answering, and more. By combining bidirectional encoding with autoregressive decoding, they leverage the strengths of both paradigms while introducing innovative pre-training objectives that better align with downstream seq2seq tasks.</p>

    <h3>T5: Text-to-Text Transfer Transformer</h3>

    <h4>The Revolutionary Text-to-Text Framework</h4>
    <p>T5's core innovation isn't architectural—it's conceptual. Google researchers asked: "What if we treated EVERY NLP task as text generation?" This unified framework eliminates task-specific architectures and heads, using the same model structure for all tasks with only the prompt format changing.</p>

    <h4>Core Philosophy</h4>
    <p>Treat every NLP task as text-to-text transformation:</p>
    <ul>
      <li><strong>Translation:</strong> "translate English to German: The house is wonderful." → "Das Haus ist wunderbar."</li>
      <li><strong>Classification:</strong> "sentiment: This movie was terrible!" → "negative"</li>
      <li><strong>Summarization:</strong> "summarize: [long article]" → "[concise summary]"</li>
      <li><strong>Question answering:</strong> "question: What is the capital? context: Paris is France's capital." → "Paris"</li>
      <li><strong>NER:</strong> "ner: John works at Google" → "John: PERSON, Google: ORG"</li>
      <li><strong>Coreference:</strong> "coref: Mary said she likes cats" → "Mary said Mary likes cats"</li>
    </ul>

    <h4>Why Text-to-Text Works</h4>
    <ul>
      <li><strong>Unified training:</strong> Single model learns diverse tasks simultaneously, enabling transfer between tasks</li>
      <li><strong>No architectural changes:</strong> Same encoder-decoder for all tasks, only prompt format differs</li>
      <li><strong>Natural evaluation:</strong> Output is text, can use string matching, BLEU, ROUGE without task-specific metrics</li>
      <li><strong>Easy extensibility:</strong> Add new tasks by designing prompts, no code changes</li>
      <li><strong>Multi-task learning:</strong> Training on diverse tasks improves generalization</li>
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

    <h3>Detailed Comparison: T5 vs BART vs GPT vs BERT</h3>

    <table border="1" cellpadding="8" style="border-collapse: collapse; width: 100%;">
      <tr>
        <th>Aspect</th>
        <th>T5</th>
        <th>BART</th>
        <th>GPT</th>
        <th>BERT</th>
      </tr>
      <tr>
        <td>Architecture</td>
        <td>Encoder-Decoder</td>
        <td>Encoder-Decoder</td>
        <td>Decoder-only</td>
        <td>Encoder-only</td>
      </tr>
      <tr>
        <td>Pre-training</td>
        <td>Span corruption</td>
        <td>Multiple noising strategies</td>
        <td>Next token prediction</td>
        <td>Masked LM + NSP</td>
      </tr>
      <tr>
        <td>Best for</td>
        <td>Seq2seq, multi-task</td>
        <td>Summarization, generation</td>
        <td>Open-ended generation</td>
        <td>Classification, NER</td>
      </tr>
      <tr>
        <td>Can generate?</td>
        <td>Yes</td>
        <td>Yes</td>
        <td>Yes</td>
        <td>No</td>
      </tr>
      <tr>
        <td>Bidirectional?</td>
        <td>Yes (encoder)</td>
        <td>Yes (encoder)</td>
        <td>No</td>
        <td>Yes</td>
      </tr>
      <tr>
        <td>Typical size</td>
        <td>60M-11B</td>
        <td>140M-400M</td>
        <td>117M-175B+</td>
        <td>110M-340M</td>
      </tr>
    </table>

    <h3>When to Choose Each Model</h3>

    <h4>Choose T5 when:</h4>
    <ul>
      <li><strong>Multi-task deployment:</strong> Single model for many tasks through prompts</li>
      <li><strong>Custom task formats:</strong> Text-to-text framework is very flexible</li>
      <li><strong>Need instruction following:</strong> Flan-T5 variants excellent</li>
      <li><strong>Longer sequences:</strong> Relative positions handle length better</li>
    </ul>

    <h4>Choose BART when:</h4>
    <ul>
      <li><strong>Summarization focus:</strong> Pre-training well-suited for this</li>
      <li><strong>Standard seq2seq:</strong> Don't need text-to-text abstraction</li>
      <li><strong>Smaller scale:</strong> BART models typically smaller, faster</li>
    </ul>

    <h4>Choose GPT when:</h4>
    <ul>
      <li><strong>Open-ended generation:</strong> Creative writing, dialogue</li>
      <li><strong>Few-shot learning:</strong> Large GPT models excel here</li>
      <li><strong>Don't need bidirectional encoding</strong></li>
    </ul>

    <h4>Choose BERT when:</h4>
    <ul>
      <li><strong>Classification only:</strong> Don't need generation</li>
      <li><strong>Speed critical:</strong> Encoder-only is fastest</li>
      <li><strong>Smaller models</strong></li>
    </ul>

    <h3>Advantages of Encoder-Decoder Architecture</h3>
    <ul>
      <li><strong>Versatile:</strong> Handle many task types</li>
      <li><strong>Bidirectional encoder:</strong> Full context understanding</li>
      <li><strong>Generative decoder:</strong> Can produce variable-length outputs</li>
      <li><strong>Strong performance:</strong> SOTA on many benchmarks</li>
      <li><strong>Transfer learning:</strong> Pre-train once, fine-tune for tasks</li>
      <li><strong>Separation of concerns:</strong> Understanding vs generation</li>
    </ul>

    <h3>Applications</h3>
    <ul>
      <li><strong>Summarization:</strong> Abstractive summarization of documents</li>
      <li><strong>Translation:</strong> Machine translation</li>
      <li><strong>Question answering:</strong> Generative QA</li>
      <li><strong>Dialogue:</strong> Conversational systems</li>
      <li><strong>Data-to-text:</strong> Generate text from structured data</li>
      <li><strong>Paraphrasing:</strong> Rephrase while maintaining meaning</li>
      <li><strong>Text simplification:</strong> Convert complex to simple</li>
      <li><strong>Grammar correction:</strong> Fix grammatical errors</li>
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
};
