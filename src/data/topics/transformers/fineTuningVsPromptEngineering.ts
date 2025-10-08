import { Topic } from '../../../types';

export const fineTuningVsPromptEngineering: Topic = {
  id: 'fine-tuning-vs-prompt-engineering',
  title: 'Fine-tuning vs Prompt Engineering',
  category: 'transformers',
  description: 'Different approaches to adapting pre-trained models for specific tasks',
  content: `
    <h2>Fine-tuning vs Prompt Engineering: Adapting Language Models</h2>
    <p>The emergence of large pre-trained language models introduced two fundamentally different paradigms for task adaptation: fine-tuning (updating model weights through gradient descent on task data) and prompt engineering (crafting inputs to elicit desired behavior without weight updates). The choice between these approaches—or hybrid combinations—has profound implications for development cost, performance, flexibility, and deployment architecture. Understanding when and how to use each approach is essential for effectively leveraging modern language models.</p>

    <h3>Fine-tuning: Supervised Adaptation Through Weight Updates</h3>

    <h4>The Fine-tuning Process</h4>
    <p>Fine-tuning continues training a pre-trained model on task-specific labeled data, updating weights through backpropagation:</p>
    <ul>
      <li><strong>Start with pre-trained model:</strong> BERT, GPT, T5, etc. with weights learned from large corpus</li>
      <li><strong>Add task-specific head:</strong> Linear layer for classification, span prediction layers for QA, etc.</li>
      <li><strong>Train on labeled data:</strong> Update weights using task loss (cross-entropy, MSE, etc.)</li>
      <li><strong>Hyperparameters:</strong> Lower learning rate (1e-5 to 5e-5), few epochs (2-4), small batches</li>
      <li><strong>Result:</strong> Model specialized for specific task, weights diverge from pre-trained initialization</li>
    </ul>

    <h4>Fine-tuning Approaches</h4>

    <h5>Full Fine-tuning</h5>
    <ul>
      <li><strong>Method:</strong> Update all model parameters + task head</li>
      <li><strong>Typical scenario:</strong> BERT-Base (110M params) fine-tuned for sentiment classification</li>
      <li><strong>Pros:</strong> Maximum flexibility, best performance</li>
      <li><strong>Cons:</strong> Expensive (GPU memory, compute), separate model per task</li>
      <li><strong>Storage:</strong> Must save full model copy for each task (100M-10B+ parameters)</li>
    </ul>

    <h5>Partial Fine-tuning (Layer Freezing)</h5>
    <ul>
      <li><strong>Method:</strong> Freeze early layers, update later layers + task head</li>
      <li><strong>Rationale:</strong> Early layers capture general features, later layers task-specific</li>
      <li><strong>Typical setup:</strong> Freeze bottom 6 layers of 12-layer BERT, train top 6 + head</li>
      <li><strong>Pros:</strong> Faster training, less overfitting risk, reduced compute</li>
      <li><strong>Cons:</strong> Slightly lower performance than full fine-tuning</li>
    </ul>

    <h5>Adapter Layers</h5>
    <ul>
      <li><strong>Method:</strong> Insert small trainable modules (adapters) between Transformer layers, freeze base model</li>
      <li><strong>Architecture:</strong> Bottleneck: $d_{\\text{model}} \\to d_{\\text{adapter}}$ (e.g., $768 \\to 64$) $\\to d_{\\text{model}}$</li>
      <li><strong>Parameters:</strong> Only ~1-5% of original model (e.g., 1M vs 110M for BERT)</li>
      <li><strong>Pros:</strong> Tiny storage per task, fast training, nearly full fine-tuning performance</li>
      <li><strong>Cons:</strong> Additional inference cost per layer, architectural modification required</li>
    </ul>

    <h5>LoRA (Low-Rank Adaptation)</h5>
    <ul>
      <li><strong>Insight:</strong> Weight updates during fine-tuning have low intrinsic dimensionality</li>
      <li><strong>Method:</strong> Represent weight updates as low-rank decomposition: $\\Delta W = BA$ where $B$ is $d\\times r$, $A$ is $r\\times k$, $r \\ll \\min(d,k)$</li>
      <li><strong>Application:</strong> Add LoRA matrices to attention query/key/value projections</li>
      <li><strong>Parameters:</strong> Typically 0.1-1% of original (e.g., 300K vs 110M for BERT)</li>
      <li><strong>Rank:</strong> r=8 or r=16 often sufficient, balancing expressiveness and efficiency</li>
      <li><strong>Pros:</strong> Minimal storage, no inference overhead (can merge LoRA into weights), excellent performance</li>
      <li><strong>Cons:</strong> Requires implementation support, rank selection hyperparameter</li>
    </ul>

    <h5>Prefix/Prompt Tuning</h5>
    <ul>
      <li><strong>Method:</strong> Prepend learnable continuous vectors (virtual tokens) to input, freeze model</li>
      <li><strong>Parameters:</strong> Only prefix embeddings (e.g., $20$ tokens $\\times$ $768$ dims $= 15K$ parameters)</li>
      <li><strong>Training:</strong> Optimize prefix embeddings through backpropagation, model weights fixed</li>
      <li><strong>Pros:</strong> Extremely parameter-efficient, single model serves all tasks</li>
      <li><strong>Cons:</strong> Requires longer sequences (prefix reduces available context), performance gap vs fine-tuning</li>
    </ul>

    <h4>Advantages of Fine-tuning</h4>
    <ul>
      <li><strong>Performance ceiling:</strong> Typically achieves best task-specific performance, especially for specialized domains</li>
      <li><strong>Data efficiency:</strong> Works well with 100s-1000s labeled examples, less than prompt engineering with weaker models</li>
      <li><strong>Consistency:</strong> Deterministic, less sensitive to input variations or prompt wording</li>
      <li><strong>Specialization depth:</strong> Can learn complex task-specific patterns, subtle domain knowledge</li>
      <li><strong>Proven approach:</strong> Well-understood, extensive literature, established best practices</li>
    </ul>

    <h4>Disadvantages of Fine-tuning</h4>
    <ul>
      <li><strong>Computational cost:</strong> Requires GPU training (hours to days), ongoing experiment iterations expensive</li>
      <li><strong>Storage overhead:</strong> Separate model per task (100MB-10GB+ each), multiplied by task count</li>
      <li><strong>Data requirements:</strong> Needs labeled training data (annotation cost, privacy concerns)</li>
      <li><strong>Deployment complexity:</strong> Manage multiple models, routing, version control</li>
      <li><strong>Catastrophic forgetting:</strong> Fine-tuned model may lose general capabilities from pre-training</li>
      <li><strong>Slow iteration:</strong> Each change requires retraining (hours), slows experimentation</li>
    </ul>

    <h4>Rough Cost Estimates (2024-2025)</h4>
    <table border="1" cellpadding="8" style="border-collapse: collapse; width: 100%;">
      <tr>
        <th>Approach</th>
        <th>Setup Cost</th>
        <th>Per-Task Cost</th>
        <th>Inference Cost</th>
        <th>Break-even Volume</th>
      </tr>
      <tr>
        <td>Full Fine-tuning (BERT-Base)</td>
        <td>$0</td>
        <td>$5-20 per run</td>
        <td>$0.0001-0.001 per request</td>
        <td>< 100K requests/month</td>
      </tr>
      <tr>
        <td>LoRA Fine-tuning</td>
        <td>$0</td>
        <td>$2-10 per run</td>
        <td>$0.0001-0.001 per request</td>
        <td>Low-medium volume</td>
      </tr>
      <tr>
        <td>GPT-4 API</td>
        <td>$0</td>
        <td>$0</td>
        <td>$0.03-0.06 per 1K tokens</td>
        <td>< 10K requests/month</td>
      </tr>
      <tr>
        <td>GPT-3.5 API</td>
        <td>$0</td>
        <td>$0</td>
        <td>$0.0015-0.002 per 1K tokens</td>
        <td>10-100K requests/month</td>
      </tr>
      <tr>
        <td>Open LLM (LLaMA 2)</td>
        <td>$0</td>
        <td>$0</td>
        <td>$0.001-0.01 per request</td>
        <td>> 50K requests/month</td>
      </tr>
    </table>

    <h4>Cost Analysis Example</h4>
    <p><strong>Scenario: Sentiment classification with 100K requests/month</strong></p>
    <pre>
Fine-tuned BERT-Base (self-hosted):
  Training: $20 one-time
  GPU server: $200/month
  Per request: $0.0002
  Monthly: $240 total

GPT-3.5 API:
  Training: $0
  Per request: $0.002 (500 tokens avg)
  Monthly: $100K \\times \\$0.002 = \\$200$

Break-even: ~100K requests/month
Below: Use API
Above: Use fine-tuned model
</pre>

    <h3>Prompt Engineering: Steering Models Through Input Design</h3>

    <h4>The Prompting Paradigm</h4>
    <p>Instead of updating weights, carefully craft input text to guide model behavior:</p>
    <ul>
      <li><strong>Core idea:</strong> Pre-trained LLM already contains knowledge; right prompt unlocks it</li>
      <li><strong>No training:</strong> Use model as-is, only modify input format</li>
      <li><strong>Natural language programming:</strong> Instructions in English, not code</li>
      <li><strong>Requires scale:</strong> Effective primarily with very large models (10B+ parameters)</li>
    </ul>

    <h4>Prompting Techniques</h4>

    <h5>Zero-Shot Prompting</h5>
    <p><strong>Format:</strong> Task description + input, no examples</p>
    <p><strong>Example:</strong> "Classify the sentiment as positive or negative. Review: The movie was fantastic! Sentiment:"</p>
    <ul>
      <li><strong>When it works:</strong> Common tasks model saw during pre-training (sentiment, translation)</li>
      <li><strong>Performance:</strong> Varies widely; strong for familiar tasks, weak for novel ones</li>
      <li><strong>Advantage:</strong> No examples needed, fastest to deploy</li>
    </ul>

    <h5>Few-Shot Prompting</h5>
    <p><strong>Format:</strong> Task description + k examples + query</p>
    <p><strong>Example:</strong></p>
    <pre>Classify sentiment:
Review: I loved it! → Positive
Review: Terrible experience. → Negative  
Review: Best purchase ever! → Positive
Review: Would not recommend. → [Model generates]</pre>
    <ul>
      <li><strong>Typical k:</strong> 3-10 examples (limited by context window)</li>
      <li><strong>Performance:</strong> Often approaches or matches fine-tuned models for large LLMs (GPT-3 175B)</li>
      <li><strong>Example selection matters:</strong> Diverse, representative examples improve performance</li>
    </ul>

    <h5>Chain-of-Thought (CoT) Prompting</h5>
    <p><strong>Innovation:</strong> Prompt model to generate intermediate reasoning steps before final answer</p>
    <p><strong>Example:</strong></p>
    <pre>Q: Roger has 5 balls. He buys 2 cans of 3 balls each. How many balls does he have?
A: Roger started with 5 balls. 2 cans of 3 balls is $2 \\times 3 = 6$ balls. $5 + 6 = 11$. Answer: 11 balls.</pre>
    <ul>
      <li><strong>Dramatic improvements:</strong> 10-30% accuracy gains on reasoning tasks</li>
      <li><strong>Emergent with scale:</strong> Only effective with models >60B parameters</li>
      <li><strong>Applications:</strong> Math word problems, logical reasoning, multi-step inference</li>
    </ul>

    <h5>Instruction Following</h5>
    <p><strong>Format:</strong> Clear, explicit task instructions</p>
    <p><strong>Example:</strong> "Summarize the following article in 2-3 sentences, focusing on key findings: [article text]"</p>
    <ul>
      <li><strong>Works best with:</strong> Instruction-tuned models (InstructGPT, GPT-3.5/4, Flan-T5)</li>
      <li><strong>Benefit:</strong> More predictable, aligned with user intent</li>
    </ul>

    <h5>Role Prompting</h5>
    <p><strong>Method:</strong> Assign model a role/persona</p>
    <p><strong>Example:</strong> "You are an expert cardiologist. Explain the risks of high cholesterol..."</p>
    <ul>
      <li><strong>Effect:</strong> Encourages domain-appropriate language and knowledge</li>
      <li><strong>Limitation:</strong> Model doesn't truly have expertise, may hallucinate confidently</li>
    </ul>

    <h4>Advantages of Prompt Engineering</h4>
    <ul>
      <li><strong>Zero training cost:</strong> No GPU compute, immediate deployment</li>
      <li><strong>Rapid iteration:</strong> Test new prompts in seconds, A/B test easily</li>
      <li><strong>Single model for many tasks:</strong> One API endpoint serves all use cases</li>
      <li><strong>No labeled data needed:</strong> Can work with just task description or few examples</li>
      <li><strong>Flexibility:</strong> Easy to modify behavior, adjust to new requirements</li>
      <li><strong>Lower deployment complexity:</strong> Single model to maintain, no multi-model routing</li>
    </ul>

    <h4>Disadvantages of Prompt Engineering</h4>
    <ul>
      <li><strong>Prompt sensitivity:</strong> Minor wording changes cause large performance swings</li>
      <li><strong>Requires massive models:</strong> Only GPT-3 scale (175B+) shows strong few-shot learning</li>
      <li><strong>Context window limits:</strong> Few-shot examples consume limited context (e.g., 4K tokens)</li>
      <li><strong>Lower ceiling:</strong> May not match specialized fine-tuned models on niche tasks</li>
      <li><strong>Inconsistency:</strong> Same prompt can yield different outputs (sampling), hard to debug</li>
      <li><strong>Inference cost:</strong> Large model inference expensive, especially for high-volume applications</li>
    </ul>

    <h3>When to Choose Each Approach</h3>

    <table border="1" cellpadding="8" style="border-collapse: collapse; width: 100%;">
      <tr>
        <th>Scenario</th>
        <th>Recommended Approach</th>
        <th>Rationale</th>
      </tr>
      <tr>
        <td>1000+ labeled examples</td>
        <td>Fine-tuning</td>
        <td>Data available, can achieve best performance</td>
      </tr>
      <tr>
        <td>Few/no labeled examples</td>
        <td>Prompt Engineering</td>
        <td>Annotation expensive, prompting leverages pre-trained knowledge</td>
      </tr>
      <tr>
        <td>Specialized domain (medical, legal)</td>
        <td>Fine-tuning</td>
        <td>Domain-specific patterns require weight adaptation</td>
      </tr>
      <tr>
        <td>Many diverse tasks (50+)</td>
        <td>Prompt Engineering</td>
        <td>Managing 50 fine-tuned models impractical</td>
      </tr>
      <tr>
        <td>Rapid prototyping phase</td>
        <td>Prompt Engineering</td>
        <td>Iterate quickly, validate idea before investing in fine-tuning</td>
      </tr>
      <tr>
        <td>Production deployment, consistency critical</td>
        <td>Fine-tuning (or PEFT)</td>
        <td>More reliable, deterministic behavior</td>
      </tr>
      <tr>
        <td>Need model to adapt daily</td>
        <td>Prompt Engineering</td>
        <td>Can't retrain daily; prompts update instantly</td>
      </tr>
      <tr>
        <td>Limited compute budget</td>
        <td>Prompt Engineering (if have LLM access) OR PEFT</td>
        <td>No training compute needed, or train tiny fraction of params</td>
      </tr>
    </table>

    <h3>Hybrid and Modern Approaches</h3>

    <h4>Instruction Tuning: Best of Both Worlds</h4>
    <ul>
      <li><strong>Method:</strong> Fine-tune LLM on diverse instruction-following tasks</li>
      <li><strong>Examples:</strong> InstructGPT (GPT-3 + RLHF), Flan-T5, Alpaca</li>
      <li><strong>Result:</strong> Model that follows instructions well via prompting while maintaining general capabilities</li>
      <li><strong>One-time cost:</strong> Expensive instruction tuning once, then pure prompting for all tasks</li>
    </ul>

    <h4>Parameter-Efficient Fine-Tuning (PEFT): Combining Benefits</h4>
    <ul>
      <li><strong>LoRA in production:</strong> Train tiny task-specific modules (0.1% of params), deploy as plugins</li>
      <li><strong>Workflow:</strong> One base model + swappable LoRA modules per task</li>
      <li><strong>Benefits:</strong> Fine-tuning performance, prompting-like efficiency</li>
      <li><strong>Real-world example:</strong> Serve 100 tasks with 1 base model + 100 small LoRA modules (few MB each)</li>
    </ul>

    <h4>Prompt-Based Data Augmentation</h4>
    <ul>
      <li><strong>Use prompting to generate training data:</strong> Ask GPT-4 to create labeled examples</li>
      <li><strong>Then fine-tune smaller model:</strong> Distill knowledge into task-specific model</li>
      <li><strong>Benefit:</strong> Cheaper inference (small model) with large model's knowledge</li>
    </ul>

    <h4>Iterative Refinement</h4>
    <ul>
      <li><strong>Phase 1:</strong> Prompt engineering for prototyping, gather user feedback</li>
      <li><strong>Phase 2:</strong> Collect interaction data, use as training set</li>
      <li><strong>Phase 3:</strong> Fine-tune (or PEFT) for production deployment</li>
      <li><strong>Ongoing:</strong> Continue prompt engineering for edge cases</li>
    </ul>

    <h3>The Future: Converging Paradigms</h3>
    <p>The distinction between fine-tuning and prompting is blurring:</p>
    <ul>
      <li><strong>Soft prompting:</strong> Learn continuous prompts through gradient descent (fine-tuning prompts, not weights)</li>
      <li><strong>Mixture of experts:</strong> Route inputs to specialized sub-models based on prompt</li>
      <li><strong>Retrieval-augmented generation:</strong> Dynamically fetch relevant examples as "prompts"</li>
      <li><strong>Meta-learning:</strong> Models that learn how to learn from prompts</li>
    </ul>

    <h3>Practical Recommendations</h3>
    <ul>
      <li><strong>Start with prompting:</strong> Validate concept with GPT-4/Claude, iterate on prompts</li>
      <li><strong>Measure prompt sensitivity:</strong> Test variations, ensure robustness</li>
      <li><strong>Consider PEFT for production:</strong> If need better performance, try LoRA before full fine-tuning</li>
      <li><strong>Hybrid approach:</strong> Prompt engineering for most tasks, fine-tuning for critical high-volume ones</li>
      <li><strong>Monitor costs:</strong> Large model prompting can exceed fine-tuned model cost at high volume</li>
      <li><strong>Version control prompts:</strong> Treat prompts like code, track changes, A/B test</li>
    </ul>

    <h3>Conclusion</h3>
    <p>Fine-tuning and prompt engineering are not mutually exclusive but complementary tools. Fine-tuning offers maximum performance and consistency when data and compute are available. Prompt engineering provides flexibility and rapid iteration when working with large models. Modern techniques like LoRA and instruction tuning blur the boundary, combining the best of both approaches. The optimal strategy depends on data availability, performance requirements, deployment constraints, and development velocity. As models continue growing and PEFT methods mature, we're moving toward a future where adaptation is lightweight, efficient, and accessible.</p>
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
};
