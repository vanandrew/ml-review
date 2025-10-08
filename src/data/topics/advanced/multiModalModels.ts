import { Topic } from '../../../types';

export const multiModalModels: Topic = {
  id: 'multi-modal-models',
  title: 'Multi-Modal Models',
  category: 'advanced',
  description: 'Models that process and integrate multiple types of data',
  content: `
    <h2>Multi-Modal Models: Integrating Vision, Language, and Beyond</h2>
    
    <p>Human perception is inherently multi-modal. We don't just see objects—we see them while hearing sounds, reading text, feeling textures, and integrating all these signals into coherent understanding. A video of a dog barking combines visual motion, audio, and potentially captions. Medical diagnosis integrates X-ray images with patient history and lab results. Traditional machine learning models operate on single modalities: image classifiers see, language models read, speech recognizers hear. Multi-modal models bridge these artificial boundaries, processing and integrating information across different types of data to enable richer, more human-like AI systems.</p>

    <p>The explosion of multi-modal AI, exemplified by models like CLIP, DALL-E, and GPT-4 Vision, represents a paradigm shift. These systems understand images through language descriptions, generate images from text prompts, and answer complex visual questions—tasks requiring deep cross-modal understanding impossible for single-modality models.</p>

    <h3>The Multi-Modal Landscape</h3>

    <h4>Common Modalities and Their Characteristics</h4>

    <p><strong>Vision (images and video):</strong> High-dimensional spatial data, typically processed by CNNs or Vision Transformers. Visual information is rich but ambiguous—an image can be described many ways. Video adds temporal dynamics, requiring spatiotemporal reasoning. Visual data is dense: a $224 \\times 224$ RGB image has $150{,}528$ dimensions.</p>

    <p><strong>Language (text):</strong> Sequential symbolic data with discrete tokens from a finite vocabulary. Language is precise and compositional—words combine to form complex meanings. Processed by Transformers, language models capture syntax, semantics, and world knowledge. Unlike images, text is inherently hierarchical (characters → words → sentences → documents).</p>

    <p><strong>Audio (speech, music, sounds):</strong> Temporal waveforms capturing acoustic information. Speech combines linguistic content with prosody, emotion, and speaker identity. Audio is continuous and high-frequency ($16$ kHz$+$ sample rates), often processed as spectrograms. Environmental sounds carry semantic information (dog barking, car honking).</p>

    <p><strong>Sensor data:</strong> LiDAR for 3D geometry, depth cameras, thermal imaging, radar. Critical for robotics and autonomous vehicles where RGB vision alone is insufficient. Different sensors capture complementary information—cameras provide texture, LiDAR provides precise distance.</p>

    <p><strong>Structured data:</strong> Tables, knowledge graphs, time series. Highly informative but require different architectures than unstructured data. Combining structured medical records with imaging enables better diagnosis.</p>

    <h3>Fundamental Challenges in Multi-Modal Learning</h3>

    <h4>1. The Representation Problem</h4>

    <p>Different modalities have fundamentally different structures. Images are spatial grids of pixels, text is sequential tokens, audio is temporal waveforms. How do we create a common representation space? This requires modality-specific encoders that project diverse inputs into a shared semantic space where "dog" (word), dog images, and dog barking sounds have similar representations.</p>

    <h4>2. The Alignment Challenge</h4>

    <p>Corresponding elements across modalities must be aligned. In video, audio and visual frames must be temporally synchronized. In image captioning, visual regions must correspond to words. Alignment can be explicit (paired data like image-caption) or learned implicitly through weak supervision. Misalignment causes models to associate wrong concepts.</p>

    <h4>3. The Fusion Question</h4>

    <p>When and how should information from different modalities be combined? Early fusion (combining raw inputs) allows maximum interaction but is computationally expensive and inflexible. Late fusion (combining final predictions) is modular but misses cross-modal interactions during processing. The optimal fusion strategy depends on the task and modality relationships.</p>

    <h4>4. Missing Modalities at Inference</h4>

    <p>Real-world systems must handle incomplete inputs gracefully. A medical diagnosis system trained on images + clinical notes shouldn't fail when notes are unavailable. Models need to learn robust representations that degrade gracefully rather than catastrophically when modalities are missing.</p>

    <h4>5. Heterogeneity: Different Scales and Distributions</h4>

    <p>Modalities have different learning dynamics—visual features may converge faster than language features. They have different scales (image pixels $0$-$255$, text token IDs $0$-$50{,}000$). Effective multi-modal learning requires balancing these heterogeneities through careful normalization and loss weighting.</p>

    <h3>Fusion Strategies: When and How to Combine</h3>

    <h4>Visual Comparison of Fusion Strategies</h4>
    <pre class="code-block">
┌─────────────────────────────────────────────────────────────────────┐
│                       EARLY FUSION                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Image → ┌─────────────────────────────────┐                        │
│          │  Concatenate at Input           │ ─────────────► Output  │
│  Text  → │  → Single Joint Model           │                        │
│          └─────────────────────────────────┘                        │
│  + Maximum interaction, - High dimensionality                       │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                        LATE FUSION                                           │
├──────────────────────────────────────────────────────────────────────────────┤
│  Image → ┌────────────────┐ ────► Pred 1                                     │
│          │ Image Encoder  │         │                                        │
│          └────────────────┘         ▼                                        │
│                                             ┌─────────────────┐              │
│  Text  → ┌────────────────┐ ────► Pred 2 ──►│ Combine         │────► Output  │
│          │ Text Encoder   │                 │ (Average/Concat)│              │
│          └────────────────┘                 └─────────────────┘              │
│                                                                              │
│  + Modular & interpretable, - Limited cross-modal interaction                │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       HYBRID FUSION (Modern)                        │
├─────────────────────────────────────────────────────────────────────┤
│  Image →  ┌────────────────┐           ┌──────────────┐             │
│           │ Image Encoder  │──────────►│ Cross-       │             │
│           └────────────────┘    ⇅      │ Attention    │             │
│                                 ⇅      │ Layers       │───► Output  │
│  Text  →  ┌────────────────┐    ⇅      │ (Interact    │             │
│           │ Text Encoder   │──────────►│ at multiple  │             │
│           │                │           │  levels)     │             │
│           └────────────────┘           └──────────────┘             │
│                                                                     │
│  + Best of both worlds: modality-specific + cross-modal             │
│  + Most effective for complex tasks (VQA, captioning)               │
└─────────────────────────────────────────────────────────────────────┘
    </pre>

    <h4>Early Fusion: Combine at Input</h4>

    <p>Early fusion concatenates features from different modalities at the input or early layers, then processes them jointly with a single model. For example, concatenating image patches with word embeddings and feeding them to a unified Transformer. This allows maximum cross-modal interaction from the start—the model can learn complex joint representations.</p>

    <p><strong>Advantages:</strong> Maximum expressiveness, captures fine-grained interactions, simple architecture (one model for all modalities).</p>

    <p><strong>Disadvantages:</strong> High dimensionality (image + text is huge), difficult to handle missing modalities, requires careful feature scaling, computationally expensive.</p>

    <h4>Late Fusion: Combine at Output</h4>

    <p>Late fusion processes each modality independently with specialized encoders, then combines their outputs (predictions or final representations) for the final decision. For example, training separate image and text classifiers, then averaging their logits.</p>

    <p><strong>Advantages:</strong> Modular (can improve/replace individual encoders), handles missing modalities naturally (just drop that modality's contribution), easier to train (can pre-train components separately), interpretable (see each modality's contribution).</p>

    <p><strong>Disadvantages:</strong> Limited cross-modal interaction (modalities don't inform each other during encoding), may miss complementary information, suboptimal for tasks requiring tight integration.</p>

    <h4>Hybrid Fusion: Best of Both Worlds</h4>

    <p>Modern multi-modal Transformers use hybrid fusion: modality-specific encoders (like late fusion) with cross-modal attention layers (like early fusion) enabling information exchange at multiple levels. This is the dominant paradigm—think of it as "mid-level fusion."</p>

    <p>For example, encode images with Vision Transformer and text with Language Transformer separately, then add cross-attention layers where image tokens attend to text tokens and vice versa. This allows both modality-specific processing and rich cross-modal interaction.</p>

    <h5>Visual: Cross-Attention Mechanism</h5>
    <pre class="code-block">
Cross-Modal Attention: How Image and Text Interact

Image Tokens: [I₁, I₂, I₃, ..., Iₙ]  (n=196 for 14×14 patches)
Text Tokens:  [T₁, T₂, T₃, ..., Tₘ]  (m=sequence length)

┌─────────────────────────────────────────────────────────────────────┐
│            Text-to-Image Cross-Attention                            │
├─────────────────────────────────────────────────────────────────────┤
│  Query:  Text tokens [T₁, T₂, ..., Tₘ]                              │
│  Keys:   Image tokens [I₁, I₂, ..., Iₙ]                             │
│  Values: Image tokens [I₁, I₂, ..., Iₙ]                             │
│                                                                     │
│  Each text token attends to all image regions:                      │
│                                                                     │
│    "dog"  ─── high attention ───► [🐶 region]                       │
│      │                                                              │
│      └─── low attention  ───► [background]                          │
│                                                                     │
│  Output: Text tokens enriched with visual information               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│            Image-to-Text Cross-Attention                            │
├─────────────────────────────────────────────────────────────────────┤
│  Query:  Image tokens [I₁, I₂, ..., Iₙ]                             │
│  Keys:   Text tokens [T₁, T₂, ..., Tₘ]                              │
│  Values: Text tokens [T₁, T₂, ..., Tₘ]                              │
│                                                                     │
│  Each image region attends to relevant words:                       │
│                                                                     │
│    [face region] ── high attention ──► "smiling"                    │
│         │                                                           │
│         └──── low attention ───► "building"                         │
│                                                                     │
│  Output: Image tokens enriched with semantic information            │
└─────────────────────────────────────────────────────────────────────┘

Bidirectional Information Flow:
Image ⇄ Text through multiple cross-attention layers
→ Tight integration of visual and linguistic understanding
    </pre>

    <h3>Landmark Multi-Modal Models</h3>

    <h4>CLIP: Vision-Language Alignment via Contrastive Learning</h4>

    <p>CLIP (Contrastive Language-Image Pre-training) from OpenAI revolutionized multi-modal learning by demonstrating that language can be a supervision signal for vision at web-scale. Rather than manually labeling images with predefined categories, CLIP learns from 400 million image-text pairs collected from the internet.</p>

    <p><strong>Architecture:</strong> Two separate encoders—an image encoder (ResNet or Vision Transformer) and a text encoder (Transformer). No fusion layers! Instead, the magic happens in the shared embedding space.</p>

    <p><strong>Training objective:</strong> Contrastive learning. Given a batch of $N$ image-text pairs, compute similarity scores between all $N^2$ possible combinations. The correct pairs (diagonal elements) should have high similarity, incorrect pairs (off-diagonal) should have low similarity. Formally, maximize:</p>

    <p style="text-align: center;">$\\text{Similarity}(\\text{Image}_i, \\text{Text}_i) - \\log \\sum_j \\exp(\\text{Similarity}(\\text{Image}_i, \\text{Text}_j))$</p>

    <p><strong>Zero-shot transfer:</strong> The killer application. To classify an image into categories {cat, dog, bird}, create text prompts "a photo of a cat," "a photo of a dog," "a photo of a bird," embed them with the text encoder, compare with the image embedding, and classify as the highest-similarity text. No training on the specific task!</p>

    <p>CLIP learns rich visual-semantic representations enabling zero-shot classification, image-text retrieval, and guidance for generative models. Its success sparked the multi-modal revolution.</p>

    <h4>DALL-E and Stable Diffusion: Text-to-Image Generation</h4>

    <p>Can we generate images from text descriptions? "A corgi wearing a crown, oil painting style"—no such image exists, but DALL-E can create it. This requires understanding language, visual composition, artistic styles, and how to synthesize coherent images.</p>

    <p><strong>DALL-E (2021):</strong> Uses a discrete VAE to tokenize images (compress $256 \\times 256$ image to grid of discrete codes), then trains an autoregressive Transformer to generate image tokens conditioned on text. Generate token by token, like language generation but for images.</p>

    <p><strong>Stable Diffusion (2022):</strong> Uses latent diffusion—operates in the latent space of a VAE rather than pixel space. Text encoder (often CLIP's text encoder) conditions the diffusion process. Iteratively denoises random latent vectors guided by text embeddings. More efficient and controllable than DALL-E.</p>

    <p>Both models demonstrate deep cross-modal understanding: they compose concepts (corgi + crown), understand artistic styles (oil painting), handle spatial relationships (person riding horse), and generate novel combinations never seen during training.</p>

    <h4>Flamingo: Few-Shot Multi-Modal Reasoning</h4>

    <p>Flamingo from DeepMind is a visual language model that can process interleaved sequences of images and text. You can show it a few examples of a task (few-shot prompting), then ask it to perform the task on new images—like showing it 2 examples of "describe this image poetically" then asking it to describe a new image poetically.</p>

    <p><strong>Architecture:</strong> Builds on a frozen pre-trained language model (Chinchilla), adding cross-attention layers that allow text tokens to attend to visual features extracted by a vision encoder. A Perceiver Resampler compresses variable numbers of images into fixed-size representations.</p>

    <p>Flamingo excels at Visual Question Answering (VQA), captioning, and visual reasoning, adapting to new tasks through in-context learning—a multi-modal version of GPT's few-shot prompting.</p>

    <h4>Whisper: Robust Speech Recognition</h4>

    <p>Whisper from OpenAI tackles speech-to-text across $99$ languages. It's multi-modal (audio → text) and multi-task (transcription, translation, language identification, timestamp detection). Trained on $680{,}000$ hours of web-collected audio-text pairs using weak supervision.</p>

    <p><strong>Architecture:</strong> Standard Transformer encoder-decoder. Audio converted to log-mel spectrogram features, encoded, then decoded as text tokens. Special tokens indicate task type ([TRANSCRIBE], [TRANSLATE]).</p>

    <p>Whisper's robustness comes from training diversity—different accents, background noise, speaking styles—demonstrating that scale and diversity in multi-modal data drive generalization.</p>

    <h4>GPT-4 Vision (GPT-4V): Multi-Modal Reasoning</h4>

    <p>GPT-4 Vision extends GPT-4's language capabilities to images, handling complex visual reasoning. It can analyze charts, read text in images (OCR), solve visual puzzles, describe scenes in detail, and even generate code from UI mockups. The architecture details are proprietary, but it likely uses a vision encoder with cross-attention to GPT-4's language model, enabling the model to "see" and reason about visual content alongside text.</p>

    <h3>Training Techniques for Multi-Modal Models</h3>

    <h4>Contrastive Learning: Aligning Modality Spaces</h4>

    <p>Contrastive learning is the dominant approach for learning aligned multi-modal representations. The core idea: pull together representations of matched cross-modal pairs (an image and its caption) while pushing apart unmatched pairs (an image and an irrelevant caption).</p>

    <p><strong>InfoNCE loss</strong> (used by CLIP): Given $N$ image-text pairs in a batch, treat the correct pair as positive and the $N-1$ incorrect pairings as negatives:</p>

    <p style="text-align: center;">$L = -\\log\\left( \\frac{\\exp(\\text{sim}(\\text{img}, \\text{text}_{\\text{match}})/\\tau)}{\\sum_j \\exp(\\text{sim}(\\text{img}, \\text{text}_j)/\\tau)} \\right)$</p>

    <p>where $\\text{sim}$ is cosine similarity and $\\tau$ is temperature. Lower temperature makes the model more discriminative. This is essentially cross-entropy loss over similarity scores.</p>

    <h4>Masked Modeling: Self-Supervised Cross-Modal Prediction</h4>

    <p>Masked language modeling (BERT-style) extends to multi-modal settings: mask some image regions and predict them from surrounding context and text, or mask words and predict them from images. This forces the model to learn cross-modal dependencies—you can't predict a masked word from an image unless you understand what the image depicts.</p>

    <h4>Alignment Objectives: Explicit Correspondence</h4>

    <p>Tasks like image-text matching (binary classification: does this text describe this image?), image-text retrieval (find the text that matches this image from a large database), and image captioning (generate text describing the image) provide explicit supervision for learning alignments.</p>

    <h3>Applications Transforming Industries</h3>

    <h4>Vision + Language Applications</h4>

    <p><strong>Visual Question Answering (VQA):</strong> "What color is the umbrella?" → "Red." Requires localizing objects (umbrella), recognizing attributes (color), and generating language. Used in accessibility tools for the blind.</p>

    <p><strong>Image captioning:</strong> Generate natural language descriptions of images. Assists content creation, image indexing, and accessibility.</p>

    <p><strong>Text-to-image generation:</strong> Creative tools for artists, designers, and content creators. DALL-E, Midjourney, and Stable Diffusion enable anyone to create visual content from text.</p>

    <p><strong>Visual reasoning:</strong> Complex tasks like solving geometry problems from diagrams, analyzing graphs, understanding memes (which require cultural and visual knowledge).</p>

    <h4>Audio + Language</h4>

    <p><strong>Speech recognition:</strong> Transcribe spoken language to text. Enables voice assistants, transcription services, accessibility tools.</p>

    <p><strong>Audio captioning:</strong> Describe sound events ("dog barking," "rain falling"). Useful for video indexing and hearing-impaired assistance.</p>

    <p><strong>Music generation:</strong> Generate music from text descriptions ("upbeat jazz piano"). Companies like Riffusion explore this space.</p>

    <h4>Real-World Multi-Modal Fusion</h4>

    <p><strong>Autonomous driving:</strong> Fuse camera (texture), LiDAR (depth), radar (velocity), GPS (location), and maps. No single sensor is sufficient—cameras fail in darkness, LiDAR struggles with rain, radar lacks resolution. Multi-modal fusion provides robust perception.</p>

    <p><strong>Healthcare:</strong> Combine medical images (X-rays, MRIs) with electronic health records (demographics, history, labs) for improved diagnosis. Text provides context that images alone lack.</p>

    <p><strong>Robotics:</strong> Integrate vision (what's in front?), language (human instructions), proprioception (body position), and force sensors (touch) for manipulation and navigation.</p>

    <h3>Evaluation: Measuring Multi-Modal Success</h3>

    <p><strong>Retrieval tasks:</strong> Image-text retrieval measured by Recall@$K$ (is correct match in top-$K$ results?), mean rank of correct match.</p>

    <p><strong>Generation tasks:</strong> Text-to-image evaluated with FID (Fréchet Inception Distance) for image quality, CLIP score for text-image alignment, and human evaluation for subjective quality.</p>

    <p><strong>Question answering:</strong> Accuracy on VQA benchmarks like VQAv2, GQA (compositional reasoning), Visual7W.</p>

    <p><strong>Zero-shot transfer:</strong> Performance on unseen tasks/datasets without fine-tuning, measuring generalization.</p>

    <h3>The Future of Multi-Modal AI</h3>

    <p>Multi-modal models are evolving from specialized systems (image captioning, speech recognition) toward general-purpose models that seamlessly integrate any modality. Future directions include: more modalities (touch, smell, taste for embodied AI), longer contexts (entire movies, not just clips), better reasoning (solving complex multi-step problems), and true multi-modal generation (creating videos with synchronized audio and text). The ultimate goal: AI systems that perceive and interact with the world as richly as humans do, understanding the full spectrum of sensory information and context.</p>
  `,
  codeExamples: [
    {
      language: 'Python',
      code: `import torch
import torch.nn as nn
import torch.nn.functional as F

# CLIP-style contrastive learning

class ImageEncoder(nn.Module):
  def __init__(self, embed_dim=512):
      super().__init__()
      # Simplified vision encoder (in practice, use ResNet or ViT)
      self.conv = nn.Sequential(
          nn.Conv2d(3, 64, 3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2),
          nn.Conv2d(64, 128, 3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2),
          nn.AdaptiveAvgPool2d((1, 1))
      )
      self.projection = nn.Linear(128, embed_dim)

  def forward(self, x):
      x = self.conv(x)
      x = x.view(x.size(0), -1)
      x = self.projection(x)
      # L2 normalize
      x = F.normalize(x, dim=-1)
      return x

class TextEncoder(nn.Module):
  def __init__(self, vocab_size=10000, embed_dim=512, hidden_dim=256):
      super().__init__()
      # Simplified text encoder (in practice, use Transformer)
      self.embedding = nn.Embedding(vocab_size, hidden_dim)
      self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
      self.projection = nn.Linear(hidden_dim, embed_dim)

  def forward(self, x):
      # x: [batch, seq_len]
      x = self.embedding(x)
      _, (h, _) = self.lstm(x)
      x = h[-1]  # Last hidden state
      x = self.projection(x)
      # L2 normalize
      x = F.normalize(x, dim=-1)
      return x

class CLIPModel(nn.Module):
  def __init__(self, embed_dim=512):
      super().__init__()
      self.image_encoder = ImageEncoder(embed_dim)
      self.text_encoder = TextEncoder(embed_dim=embed_dim)
      # Learnable temperature parameter
      self.temperature = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

  def forward(self, images, texts):
      # Encode images and texts
      image_features = self.image_encoder(images)  # [batch, embed_dim]
      text_features = self.text_encoder(texts)      # [batch, embed_dim]

      return image_features, text_features

def contrastive_loss(image_features, text_features, temperature):
  """
  Compute symmetric contrastive loss (InfoNCE)

  Args:
      image_features: [batch, embed_dim]
      text_features: [batch, embed_dim]
      temperature: scalar

  Returns:
      loss: scalar
  """
  # Compute similarity matrix
  logits = (image_features @ text_features.T) / temperature

  # Labels: diagonal elements are positive pairs
  batch_size = logits.shape[0]
  labels = torch.arange(batch_size, device=logits.device)

  # Symmetric loss
  loss_i2t = F.cross_entropy(logits, labels)  # Image → Text
  loss_t2i = F.cross_entropy(logits.T, labels)  # Text → Image

  loss = (loss_i2t + loss_t2i) / 2

  return loss

# Training example
model = CLIPModel(embed_dim=512)

# Batch of images and texts
batch_size = 32
images = torch.randn(batch_size, 3, 224, 224)
texts = torch.randint(0, 10000, (batch_size, 50))  # Token IDs

# Forward pass
image_features, text_features = model(images, texts)

# Compute loss
loss = contrastive_loss(image_features, text_features, model.temperature.exp())

print(f"Contrastive loss: {loss.item():.4f}")

# Zero-shot classification at inference
with torch.no_grad():
  # Encode query image
  query_image = torch.randn(1, 3, 224, 224)
  image_emb = model.image_encoder(query_image)

  # Encode class descriptions
  class_texts = [
      "a photo of a cat",
      "a photo of a dog",
      "a photo of a bird"
  ]
  # Simplified: use random token IDs
  text_tokens = torch.randint(0, 10000, (3, 50))
  text_embs = model.text_encoder(text_tokens)

  # Compute similarities
  similarities = (image_emb @ text_embs.T) / model.temperature.exp()
  probs = F.softmax(similarities, dim=-1)

  print("Class probabilities:", probs)
  predicted_class = probs.argmax(dim=-1)
  print(f"Predicted class: {class_texts[predicted_class]}")`,
      explanation: 'CLIP-style contrastive learning for image-text alignment with zero-shot classification.'
    },
    {
      language: 'Python',
      code: `import torch
import torch.nn as nn

# Multi-modal Transformer with cross-attention

class CrossModalAttention(nn.Module):
  def __init__(self, d_model=512, num_heads=8):
      super().__init__()
      self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
      self.norm = nn.LayerNorm(d_model)

  def forward(self, query, key_value):
      # query: [batch, seq_len_q, d_model]
      # key_value: [batch, seq_len_kv, d_model]

      # Cross-attention: query from one modality, keys/values from another
      attn_output, attn_weights = self.multihead_attn(
          query, key_value, key_value
      )

      # Residual + norm
      output = self.norm(query + attn_output)

      return output, attn_weights

class MultiModalTransformer(nn.Module):
  def __init__(self, d_model=512, num_heads=8, num_layers=6):
      super().__init__()

      # Modality-specific encoders
      self.vision_encoder = nn.Sequential(
          nn.Linear(2048, d_model),  # From pre-extracted visual features
          nn.LayerNorm(d_model)
      )

      self.text_encoder = nn.Embedding(10000, d_model)

      # Cross-modal layers
      self.cross_modal_layers = nn.ModuleList([
          nn.ModuleDict({
              'vision_to_text': CrossModalAttention(d_model, num_heads),
              'text_to_vision': CrossModalAttention(d_model, num_heads),
              'ffn_vision': nn.Sequential(
                  nn.Linear(d_model, d_model * 4),
                  nn.GELU(),
                  nn.Linear(d_model * 4, d_model)
              ),
              'ffn_text': nn.Sequential(
                  nn.Linear(d_model, d_model * 4),
                  nn.GELU(),
                  nn.Linear(d_model * 4, d_model)
              )
          })
          for _ in range(num_layers)
      ])

      # Task head (e.g., VQA)
      self.output_head = nn.Linear(d_model, 3129)  # VQA answer vocabulary

  def forward(self, visual_features, text_tokens):
      # visual_features: [batch, num_regions, 2048]
      # text_tokens: [batch, seq_len]

      # Encode modalities
      vision_emb = self.vision_encoder(visual_features)
      text_emb = self.text_encoder(text_tokens)

      # Cross-modal interaction
      for layer in self.cross_modal_layers:
          # Vision attends to text
          vision_emb_new, _ = layer['vision_to_text'](vision_emb, text_emb)

          # Text attends to vision
          text_emb_new, _ = layer['text_to_vision'](text_emb, vision_emb)

          # FFN
          vision_emb = vision_emb_new + layer['ffn_vision'](vision_emb_new)
          text_emb = text_emb_new + layer['ffn_text'](text_emb_new)

      # Aggregate and predict
      # Use [CLS] token or mean pooling
      fused = torch.cat([
          vision_emb.mean(dim=1),
          text_emb.mean(dim=1)
      ], dim=-1)

      # For VQA: predict answer
      logits = self.output_head(fused[:, :512])  # Use text representation

      return logits

# Usage example
model = MultiModalTransformer(d_model=512, num_heads=8, num_layers=6)

# Visual Question Answering example
batch_size = 16
num_regions = 36  # Number of detected regions in image
seq_len = 20      # Question length

visual_features = torch.randn(batch_size, num_regions, 2048)
text_tokens = torch.randint(0, 10000, (batch_size, seq_len))

# Forward pass
logits = model(visual_features, text_tokens)
print(f"Output logits shape: {logits.shape}")  # [batch_size, num_answers]

# Predict answer
predicted_answers = logits.argmax(dim=-1)
print(f"Predicted answers: {predicted_answers}")

# Training: use cross-entropy with answer labels
answer_labels = torch.randint(0, 3129, (batch_size,))
loss = nn.CrossEntropyLoss()(logits, answer_labels)
print(f"VQA loss: {loss.item():.4f}")`,
      explanation: 'Multi-modal Transformer with cross-attention for Visual Question Answering, showing bidirectional cross-modal fusion.'
    }
  ],
  interviewQuestions: [
    {
      question: 'What are the main challenges in multi-modal learning?',
      answer: `Key challenges include: (1) Modality gap - different data types (image, text, audio) have different statistical properties and representations, (2) Alignment - corresponding elements across modalities may not be perfectly synchronized, (3) Missing modalities - handling scenarios where some modalities are unavailable, (4) Fusion strategy - determining how to combine information from different modalities, (5) Scale differences - modalities may have different learning rates and convergence properties, (6) Representation learning - finding shared or complementary representations.`
    },
    {
      question: 'Explain the difference between early, late, and hybrid fusion.',
      answer: `Early fusion combines raw features at input level - simple but may lose modality-specific patterns. Late fusion processes each modality separately then combines final outputs - preserves modality-specific processing but limited cross-modal interaction. Hybrid fusion combines features at multiple levels - more flexible, enables both modality-specific and cross-modal learning. Trade-offs: early fusion may lose information, late fusion may miss interactions, hybrid fusion is more complex but often most effective for capturing complementary information.`
    },
    {
      question: 'How does CLIP enable zero-shot image classification?',
      answer: `CLIP (Contrastive Language-Image Pre-training) learns joint embeddings of images and text by training on image-caption pairs using contrastive learning. For zero-shot classification: (1) Convert class names to text prompts ("a photo of a [class]"), (2) Embed both image and text prompts, (3) Calculate similarity scores between image and all text embeddings, (4) Classify as highest-scoring class. Works because CLIP learns semantic relationships between visual and textual concepts, enabling classification of unseen classes through natural language descriptions.`
    },
    {
      question: 'What is contrastive learning and why is it useful for multi-modal models?',
      answer: `Contrastive learning trains models to make similar examples close and dissimilar examples far apart in embedding space. For multi-modal models: learns alignments between modalities by pulling together corresponding pairs (image-caption) and pushing apart non-corresponding pairs. Benefits: (1) No need for explicit labels, (2) Learns semantic relationships, (3) Enables zero-shot transfer, (4) Robust to noise in correspondences. Key insight: multi-modal correspondences provide natural positive/negative pairs for contrastive training, enabling self-supervised learning of cross-modal representations.`
    },
    {
      question: 'How would you handle missing modalities at inference time?',
      answer: `Strategies include: (1) Modality dropout during training - randomly mask modalities to learn robust representations, (2) Imputation - predict missing modality from available ones using learned mappings, (3) Graceful degradation - design architectures that work with subsets of modalities, (4) Attention mechanisms - automatically weight available modalities, (5) Ensemble methods - combine predictions from different modality subsets, (6) Default representations - use learned averages for missing modalities. Key is training the model to expect and handle missing modalities.`
    },
    {
      question: 'Compare cross-attention vs concatenation for multi-modal fusion.',
      answer: `Concatenation: Simple feature combination at specific layer, preserves all information but may not learn interactions effectively. Limited modeling of cross-modal dependencies. Cross-attention: Enables each modality to attend to relevant parts of other modalities, learns complex interactions, more parameter efficient as it doesn't require fixed-size concatenation. Benefits: dynamic interaction, selective attention, better handling of variable-length sequences. Trade-off: concatenation is simpler and computationally cheaper, cross-attention is more expressive but requires more computation and parameters.`
    }
  ],
  quizQuestions: [
    {
      id: 'mm1',
      question: 'What is the main advantage of CLIP?',
      options: ['Faster training', 'Zero-shot transfer to new tasks', 'Smaller models', 'Better optimization'],
      correctAnswer: 1,
      explanation: 'CLIP learns aligned image-text representations through contrastive learning, enabling zero-shot transfer to new tasks by using text prompts as classifiers without additional training.'
    },
    {
      id: 'mm2',
      question: 'What is early fusion in multi-modal learning?',
      options: ['Combine modalities at input level', 'Combine at output level', 'Train modalities separately', 'Use only one modality'],
      correctAnswer: 0,
      explanation: 'Early fusion combines different modalities at the input level before processing, allowing maximum interaction but at the cost of flexibility and increased dimensionality.'
    },
    {
      id: 'mm3',
      question: 'What does contrastive learning optimize in multi-modal models?',
      options: ['Reconstruction error', 'Classification accuracy', 'Similarity of matched pairs', 'Generation quality'],
      correctAnswer: 2,
      explanation: 'Contrastive learning maximizes the similarity of matched cross-modal pairs (e.g., image-caption) while minimizing similarity of unmatched pairs, creating aligned representations.'
    }
  ]
};
