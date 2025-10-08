import { Topic } from '../../../types';

export const visionTransformers: Topic = {
  id: 'vision-transformers',
  title: 'Vision Transformers (ViT)',
  category: 'transformers',
  description: 'Applying Transformers to computer vision tasks',
  content: `
    <h2>Vision Transformers: Transformers Beyond Language</h2>
    <p>Vision Transformers (ViT), introduced by Google Research in 2020, challenged the dominance of Convolutional Neural Networks in computer vision by directly applying the Transformer architecture to images. By treating an image as a sequence of patches and processing them with standard Transformer encoders, ViT demonstrated that attention mechanisms could match or exceed CNN performance on image classification without any image-specific inductive biases. This breakthrough opened the door to unified architectures across vision and language, multimodal models, and sparked a revolution in computer vision research.</p>

    <h3>The Core Challenge: Images Are Not Sequences</h3>

    <h4>Why CNNs Dominated Vision</h4>
    <ul>
      <li><strong>Spatial inductive biases:</strong> Convolutions naturally capture local spatial relationships, translation invariance, hierarchical feature extraction</li>
      <li><strong>Parameter efficiency:</strong> Weight sharing across spatial locations, far fewer parameters than fully connected layers</li>
      <li><strong>Proven success:</strong> AlexNet (2012) $\\to$ ResNet (2015) $\\to$ EfficientNet (2019) progressively improved ImageNet performance</li>
      <li><strong>Image-specific design:</strong> Architectures explicitly designed for 2D spatial data</li>
    </ul>

    <h4>The Transformer Advantage</h4>
    <ul>
      <li><strong>Global context:</strong> Self-attention captures long-range dependencies from layer 1, CNNs need deep stacks for large receptive fields</li>
      <li><strong>Flexibility:</strong> Same architecture for images, text, audio, video—no task-specific design</li>
      <li><strong>Scalability:</strong> Transformers scale better with data and compute than CNNs</li>
      <li><strong>Interpretability:</strong> Attention maps show which image regions the model focuses on</li>
    </ul>

    <h4>The Key Question</h4>
    <p><strong>"Can a pure Transformer, without convolutional inductive biases, compete with state-of-the-art CNNs on image classification?"</strong></p>
    <p>Vision Transformer's answer: Yes—with sufficient data and scale.</p>

    <h3>Vision Transformer (ViT) Architecture</h3>

    <h4>Step 1: Image to Sequence (Patch Embedding)</h4>
    <p>Transform 2D image into 1D sequence of patch embeddings:</p>

    <h5>Patch Extraction</h5>
    <ul>
      <li><strong>Input image:</strong> $H \\times W \\times C$ (e.g., $224 \\times 224 \\times 3$ RGB image)</li>
      <li><strong>Patch size:</strong> $P \\times P$ (typically $16 \\times 16$ or $32 \\times 32$)</li>
      <li><strong>Number of patches:</strong> $N = (H \\times W) / (P \\times P) = (224/16)^2 = 196$ patches for $224\\times224$ image with $16\\times16$ patches</li>
      <li><strong>Flatten each patch:</strong> $P \\times P \\times C \\to D$-dimensional vector via learned linear projection</li>
      <li><strong>Result:</strong> Sequence of $N$ patch embeddings, each of dimension $D$</li>
    </ul>

    <h5>Concrete Calculation Example</h5>
    <pre>
Given: $224\\times224$ RGB image, patch size $16\\times16$, embed_dim 768

Step 1: Calculate number of patches per dimension
  - Patches per row: $224 / 16 = 14$
  - Patches per column: $224 / 16 = 14$
  - Total patches: $14 \\times 14 = 196$

Step 2: Flatten each patch
  - Each patch: $16\\times16\\times3 = 768$ values
  - Linear projection: $768$ input dims $\\to$ $768$ embed dims
  - Parameters in projection: $768 \\times 768 = 589{,}824$

Step 3: Final sequence
  - Patch embeddings: $[196, 768]$
  - Add [CLS] token: $[1, 768]$
  - Add positional embeddings: $[197, 768]$
  - Result: $[197, 768]$ sequence fed to Transformer

For comparison with $8\\times8$ patches:
  - Patches: $(224/8)^2 = 28^2 = 784$ patches
  - Each patch: $8\\times8\\times3 = 192$ values
  - Sequence length: $785$ ($4\\times$ longer, $4\\times$ more compute)
</pre>

    <h4>When NOT to Use ViT</h4>
    <ul>
      <li><strong>Small datasets ($<10K$ images):</strong> ViT severely underperforms CNNs without pre-training. Use ResNet or EfficientNet instead.</li>
      <li><strong>Limited compute budget:</strong> Training ViT from scratch is expensive. Consider pre-trained CNNs or smaller hybrid models.</li>
      <li><strong>Real-time mobile inference:</strong> ViT's quadratic attention is slower than CNN convolutions on edge devices. Use MobileNet or EfficientNet.</li>
      <li><strong>Very high resolution images ($>1024\\times1024$):</strong> Quadratic complexity in number of patches becomes prohibitive. Use hierarchical approaches like Swin Transformer.</li>
      <li><strong>Strong locality requirements:</strong> Some tasks benefit from CNN's inductive bias (e.g., texture classification). Hybrid CNN-ViT can be better.</li>
      <li><strong>Production with strict latency SLAs:</strong> CNNs have more predictable inference times. ViT attention patterns can vary significantly.</li>
    </ul>

    <h5>Mathematical Formulation</h5>
    <p><strong>Patch embedding:</strong> For image $x \\in \\mathbb{R}^{H\\times W\\times C}$, split into patches $x_p \\in \\mathbb{R}^{N\\times(P^2\\cdot C)}$</p>
    <p><strong>Linear projection:</strong> $z_0 = [x_{\\text{class}}; x_p^1E; x_p^2E; \\ldots; x_p^NE] + E_{\\text{pos}}$</p>
    <ul>
      <li><strong>$E \\in \\mathbb{R}^{(P^2\\cdot C)\\times D}$:</strong> Learned patch embedding matrix</li>
      <li><strong>$x_{\\text{class}}$:</strong> Learnable [CLS] token prepended to sequence (for classification)</li>
      <li><strong>$E_{\\text{pos}} \\in \\mathbb{R}^{(N+1)\\times D}$:</strong> Positional embeddings (learned or sinusoidal)</li>
    </ul>

    <h4>Step 2: Standard Transformer Encoder</h4>
    <p>Apply L layers of standard Transformer encoder blocks (identical to BERT/original Transformer encoder):</p>

    <h5>Encoder Block (repeated L times)</h5>
    <pre>
z'_l = MSA(LN(z_(l-1))) + z_(l-1)        # Multi-head self-attention + residual
z_l = MLP(LN(z'_l)) + z'_l               # Feedforward + residual
    </pre>
    <ul>
      <li><strong>LN:</strong> Layer Normalization (pre-norm configuration)</li>
      <li><strong>MSA:</strong> Multi-head Self-Attention with h heads</li>
      <li><strong>MLP:</strong> Two-layer feedforward network with GELU activation: $\\text{MLP}(x) = \\text{GELU}(xW_1 + b_1)W_2 + b_2$</li>
      <li><strong>Hidden dimension:</strong> Typically $D_{\\text{ff}} = 4D$ (e.g., $768 \\to 3072 \\to 768$)</li>
    </ul>

    <h5>No Modifications for Vision</h5>
    <p>Critically, ViT uses the <strong>exact same</strong> Transformer encoder as BERT/GPT—no convolutional layers, no image-specific components. The only vision-specific part is patch embedding.</p>

    <h4>Step 3: Classification Head</h4>
    <ul>
      <li><strong>Extract [CLS] token:</strong> Use representation of first token $z_L^0$ (analogous to BERT's [CLS])</li>
      <li><strong>Classification:</strong> $y = \\text{softmax}(z_L^0 W_{\\text{head}} + b_{\\text{head}})$ where $W_{\\text{head}} \\in \\mathbb{R}^{D\\times K}$, $K$ = number of classes</li>
      <li><strong>Alternative:</strong> Some implementations use global average pooling over all patch tokens instead of [CLS]</li>
    </ul>

    <h3>ViT Model Configurations</h3>

    <table border="1" cellpadding="8" style="border-collapse: collapse; width: 100%;">
      <tr>
        <th>Model</th>
        <th>Layers (L)</th>
        <th>Hidden Size (D)</th>
        <th>MLP Size</th>
        <th>Heads (h)</th>
        <th>Params</th>
      </tr>
      <tr>
        <td>ViT-Base/16</td>
        <td>12</td>
        <td>768</td>
        <td>3072</td>
        <td>12</td>
        <td>86M</td>
      </tr>
      <tr>
        <td>ViT-Large/16</td>
        <td>24</td>
        <td>1024</td>
        <td>4096</td>
        <td>16</td>
        <td>307M</td>
      </tr>
      <tr>
        <td>ViT-Huge/14</td>
        <td>32</td>
        <td>1280</td>
        <td>5120</td>
        <td>16</td>
        <td>632M</td>
      </tr>
    </table>

    <p><strong>Naming convention:</strong> ViT-{Size}/{Patch size}. E.g., ViT-Base/16 uses Base configuration with $16\\times16$ patches.</p>

    <h3>Key Insights and Design Decisions</h3>

    <h4>Patch Size Trade-offs</h4>
    <ul>
      <li><strong>Smaller patches (P=14 or 16):</strong> More patches $\\to$ longer sequence $\\to$ more computation, but finer-grained spatial information</li>
      <li><strong>Larger patches (P=32):</strong> Fewer patches $\\to$ faster, but coarser spatial resolution</li>
      <li><strong>Typical choice:</strong> P=16 balances efficiency and performance</li>
      <li><strong>Sequence length:</strong> $224\\times224$ image: $196$ patches (P=16), $784$ patches (P=8), $49$ patches (P=32)</li>
    </ul>

    <h4>Positional Encoding for 2D</h4>
    <ul>
      <li><strong>1D learned embeddings:</strong> ViT uses standard 1D positional embeddings (same as BERT), treats patches as 1D sequence</li>
      <li><strong>Ignores 2D structure:</strong> No explicit 2D spatial encoding (row/column positions)</li>
      <li><strong>Learned through attention:</strong> Model learns spatial relationships through self-attention</li>
      <li><strong>Alternative:</strong> Some variants use 2D sinusoidal or learned 2D positional encodings</li>
      <li><strong>Finding:</strong> 1D embeddings work well; attention learns spatial structure implicitly</li>
    </ul>

    <h4>The [CLS] Token</h4>
    <ul>
      <li><strong>Borrowed from BERT:</strong> Special token prepended to sequence for classification</li>
      <li><strong>Aggregation mechanism:</strong> [CLS] token's final representation aggregates information from all patches via attention</li>
      <li><strong>Why it works:</strong> Self-attention allows [CLS] to attend to all image patches, creating global image representation</li>
      <li><strong>Alternative:</strong> Global average pooling (GAP) over all patch tokens performs similarly</li>
    </ul>

    <h3>Training Vision Transformers</h3>

    <h4>The Data Requirement: Scale Matters</h4>
    <p><strong>Critical finding:</strong> ViT requires more data than CNNs to achieve competitive performance.</p>

    <h5>Performance vs Dataset Size</h5>
    <ul>
      <li><strong>Small datasets (ImageNet-1K, 1.3M images):</strong> ViT underperforms ResNets. CNNs' inductive biases help with limited data</li>
      <li><strong>Medium datasets (ImageNet-21K, 14M images):</strong> ViT matches ResNet performance</li>
      <li><strong>Large datasets (JFT-300M, 300M images):</strong> ViT surpasses ResNets, benefits more from scale</li>
    </ul>

    <h5>Why More Data Needed?</h5>
    <ul>
      <li><strong>No built-in inductive biases:</strong> ViT doesn't assume locality, translation invariance—must learn from data</li>
      <li><strong>More flexible but needs more examples:</strong> Greater model capacity requires more data to constrain</li>
      <li><strong>Scaling advantage:</strong> Once data is sufficient, ViT scales better than CNNs</li>
    </ul>

    <h4>Pre-training and Transfer Learning</h4>

    <h5>Standard Workflow</h5>
    <ol>
      <li><strong>Pre-train on large dataset:</strong> ImageNet-21K or JFT-300M with image classification objective</li>
      <li><strong>Fine-tune on target task:</strong> Transfer to ImageNet-1K, CIFAR, or domain-specific datasets</li>
      <li><strong>Benefits:</strong> Pre-trained ViT transfers exceptionally well, often better than CNNs</li>
    </ol>

    <h5>Transfer Learning Details</h5>
    <ul>
      <li><strong>Resolution adaptation:</strong> Pre-train at $224\\times224$, fine-tune at higher resolution ($384\\times384$) for better performance</li>
      <li><strong>Position embedding interpolation:</strong> When resolution changes, interpolate positional embeddings to match new patch count</li>
      <li><strong>Fine-tuning speed:</strong> Much faster than pre-training (hours vs days)</li>
    </ul>

    <h4>Training Configuration</h4>
    <ul>
      <li><strong>Optimizer:</strong> Adam/AdamW with weight decay</li>
      <li><strong>Learning rate:</strong> Warmup for first few epochs, then cosine decay</li>
      <li><strong>Augmentation:</strong> RandAugment, Mixup, Cutmix—same as modern CNN training</li>
      <li><strong>Regularization:</strong> Dropout, stochastic depth (drop entire layers randomly)</li>
      <li><strong>Pre-training time:</strong> ViT-Huge on TPUv3: ~1 month on JFT-300M</li>
    </ul>

    <h3>Performance and Comparison</h3>

    <h4>ImageNet Results (Original ViT Paper, 2020)</h4>
    <ul>
      <li><strong>ViT-Huge/14 pre-trained on JFT-300M:</strong> 88.55% top-1 accuracy on ImageNet (SOTA at time)</li>
      <li><strong>ViT-Large/16 pre-trained on JFT-300M:</strong> 87.76% accuracy</li>
      <li><strong>BiT-ResNet152x4 (CNN baseline):</strong> 87.54% accuracy</li>
      <li><strong>Result:</strong> ViT surpasses best CNNs when pre-trained on sufficient data</li>
    </ul>

    <h4>Efficiency Comparison</h4>
    <ul>
      <li><strong>Training FLOPs:</strong> ViT requires fewer FLOPs to reach same accuracy as ResNet during pre-training</li>
      <li><strong>Inference speed:</strong> Similar to ResNets of comparable accuracy, depends on patch size and model size</li>
      <li><strong>Parameter efficiency:</strong> ViT-Base (86M params) competitive with ResNet-152 (60M params)</li>
    </ul>

    <h3>Variants and Improvements</h3>

    <h4>DeiT (Data-efficient Image Transformers)</h4>
    <ul>
      <li><strong>Goal:</strong> Train ViT on ImageNet-1K without massive pre-training dataset</li>
      <li><strong>Distillation token:</strong> Add special token that learns from CNN teacher model</li>
      <li><strong>Strong augmentation:</strong> Extensive data augmentation compensates for smaller dataset</li>
      <li><strong>Result:</strong> Competitive performance with ViT using only ImageNet-1K</li>
    </ul>

    <h4>Swin Transformer</h4>
    <ul>
      <li><strong>Hierarchical architecture:</strong> Multi-scale feature maps like CNNs (not flat like ViT)</li>
      <li><strong>Shifted windows:</strong> Attention within local windows, shifted across layers for cross-window connections</li>
      <li><strong>Efficiency:</strong> Linear complexity in image size (vs quadratic for ViT)</li>
      <li><strong>Versatility:</strong> Backbone for detection, segmentation, not just classification</li>
      <li><strong>Performance:</strong> SOTA on many vision benchmarks</li>
    </ul>

    <h4>Hybrid Models (ViT + CNN)</h4>
    <ul>
      <li><strong>Approach:</strong> Use CNN to extract initial features, then Transformer for global reasoning</li>
      <li><strong>Example:</strong> Replace patch embedding with ResNet stem (early conv layers)</li>
      <li><strong>Benefits:</strong> Combines CNN's local inductive biases with Transformer's global modeling</li>
      <li><strong>Finding:</strong> Hybrids perform well but pure ViT scales better with data</li>
    </ul>

    <h4>BEiT (BERT Pre-training for Images)</h4>
    <ul>
      <li><strong>Inspiration:</strong> Apply BERT's masked language modeling to images</li>
      <li><strong>Method:</strong> Mask image patches, predict visual tokens from discrete VAE codebook</li>
      <li><strong>Self-supervised:</strong> Pre-train without labels, learn visual representations</li>
      <li><strong>Performance:</strong> Competitive with supervised pre-training</li>
    </ul>

    <h4>MAE (Masked Autoencoders)</h4>
    <ul>
      <li><strong>Method:</strong> Mask large portion of image (75%), reconstruct pixel values</li>
      <li><strong>Asymmetric encoder-decoder:</strong> Large encoder for visible patches, small decoder for reconstruction</li>
      <li><strong>Efficiency:</strong> Only encode visible patches, very fast pre-training</li>
      <li><strong>Result:</strong> Simple, effective self-supervised learning for ViT</li>
    </ul>

    <h3>Beyond Image Classification</h3>

    <h4>Object Detection (DETR)</h4>
    <ul>
      <li><strong>Detection Transformer (DETR):</strong> End-to-end object detection with Transformers</li>
      <li><strong>Set prediction:</strong> Predict set of bounding boxes and classes directly, no anchor boxes or NMS</li>
      <li><strong>Architecture:</strong> CNN backbone $\\to$ Transformer encoder-decoder $\\to$ detection heads</li>
      <li><strong>Impact:</strong> Simplified detection pipeline, removed hand-crafted components</li>
    </ul>

    <h4>Segmentation</h4>
    <ul>
      <li><strong>SegFormer, Segmenter:</strong> ViT-based segmentation models</li>
      <li><strong>Approach:</strong> Encoder-decoder with Transformer encoder, produce pixel-wise predictions</li>
      <li><strong>Performance:</strong> Competitive with CNN-based segmentation models</li>
    </ul>

    <h4>Video Understanding</h4>
    <ul>
      <li><strong>TimeSformer, ViViT:</strong> Extend ViT to video by treating video as space-time patches</li>
      <li><strong>Factorized attention:</strong> Separate spatial and temporal attention for efficiency</li>
      <li><strong>Applications:</strong> Action recognition, video classification</li>
    </ul>

    <h3>Attention Pattern Analysis</h3>

    <h4>What Do Vision Transformers Learn?</h4>

    <h5>Early Layers</h5>
    <ul>
      <li><strong>Local patterns:</strong> Early attention heads focus on nearby patches, similar to CNN receptive fields</li>
      <li><strong>Emergent convolution:</strong> Some heads learn to attend to spatially adjacent patches</li>
    </ul>

    <h5>Middle Layers</h5>
    <ul>
      <li><strong>Semantic grouping:</strong> Attention clusters semantically related regions (e.g., all pixels of an object)</li>
      <li><strong>Long-range dependencies:</strong> Heads start connecting distant but related patches</li>
    </ul>

    <h5>Late Layers</h5>
    <ul>
      <li><strong>Global context:</strong> Attention widely distributed across entire image</li>
      <li><strong>Object-level reasoning:</strong> [CLS] token attends to discriminative object regions</li>
    </ul>

    <h4>Comparison to CNNs</h4>
    <ul>
      <li><strong>More flexible attention:</strong> ViT can attend to distant regions from layer 1; CNNs need deep stacks</li>
      <li><strong>Less texture bias:</strong> ViT less biased toward texture than CNNs, more shape-focused</li>
      <li><strong>Better at long-range relationships:</strong> Natural for modeling global structure</li>
    </ul>

    <h3>Advantages of Vision Transformers</h3>
    <ul>
      <li><strong>Global receptive field from layer 1:</strong> Every patch can attend to every other patch immediately, no need for deep stacks</li>
      <li><strong>Unified architecture:</strong> Same model for vision and language enables multimodal learning (CLIP, DALL-E)</li>
      <li><strong>Scalability:</strong> Performance improves more with data/compute than CNNs, better scaling laws</li>
      <li><strong>Transfer learning:</strong> Pre-trained ViT transfers exceptionally well to diverse tasks</li>
      <li><strong>Interpretability:</strong> Attention maps visualize what model focuses on, easier to interpret than CNN activations</li>
      <li><strong>Flexibility:</strong> Easy to adapt to different input sizes, modalities (add text/audio patches)</li>
    </ul>

    <h3>Limitations and Challenges</h3>
    <ul>
      <li><strong>Data hungry:</strong> Requires large pre-training datasets, underperforms CNNs on small datasets</li>
      <li><strong>Computational cost:</strong> Quadratic complexity in number of patches $O(N^2)$, expensive for high-resolution images</li>
      <li><strong>Memory intensive:</strong> Attention matrix storage scales quadratically</li>
      <li><strong>Less efficient for small tasks:</strong> Overkill for simple classification with limited data</li>
      <li><strong>Pre-training cost:</strong> Training on JFT-300M extremely expensive (months, thousands of TPUs)</li>
      <li><strong>Patch boundary artifacts:</strong> Hard splits at patch boundaries, no smooth spatial continuity</li>
    </ul>

    <h3>The Vision Transformer Revolution</h3>
    <p>Vision Transformers fundamentally challenged the assumption that convolutional inductive biases are necessary for computer vision. By demonstrating that pure attention mechanisms can match or exceed CNN performance given sufficient scale, ViT opened new paradigms: unified architectures across modalities (CLIP, Flamingo), self-supervised visual learning (MAE, DINO), and efficient adaptation (prompt tuning for vision). While CNNs remain relevant for resource-constrained scenarios and small datasets, Transformers have become the architecture of choice for large-scale vision systems. The success of ViT exemplifies a broader trend in AI: with enough data and compute, flexible general-purpose architectures can outperform hand-crafted domain-specific designs. Vision Transformers are not just an alternative to CNNs—they represent the convergence of vision and language AI toward unified foundation models.</p>
  `,
  codeExamples: [
    {
      language: 'Python',
      code: `import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
  """Split image into patches and embed them"""
  def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
      super().__init__()
      self.img_size = img_size
      self.patch_size = patch_size
      self.num_patches = (img_size // patch_size) ** 2
      
      # Patch embedding using convolution
      # This is equivalent to splitting into patches and linear projection
      self.projection = nn.Sequential(
          nn.Conv2d(in_channels, embed_dim, 
                   kernel_size=patch_size, stride=patch_size),
          Rearrange('b c h w -> b (h w) c')  # Flatten spatial dimensions
      )
      
  def forward(self, x):
      # x: [batch, channels, height, width]
      # output: [batch, num_patches, embed_dim]
      return self.projection(x)

class TransformerEncoder(nn.Module):
  """Standard Transformer encoder block"""
  def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
      super().__init__()
      self.norm1 = nn.LayerNorm(embed_dim)
      self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
      self.norm2 = nn.LayerNorm(embed_dim)
      
      mlp_hidden_dim = int(embed_dim * mlp_ratio)
      self.mlp = nn.Sequential(
          nn.Linear(embed_dim, mlp_hidden_dim),
          nn.GELU(),
          nn.Dropout(dropout),
          nn.Linear(mlp_hidden_dim, embed_dim),
          nn.Dropout(dropout)
      )
      
  def forward(self, x):
      # Multi-head self-attention with residual
      x_norm = self.norm1(x)
      attn_out, _ = self.attn(x_norm, x_norm, x_norm)
      x = x + attn_out
      
      # MLP with residual
      x = x + self.mlp(self.norm2(x))
      return x

class VisionTransformer(nn.Module):
  """Vision Transformer (ViT) implementation"""
  def __init__(self, img_size=224, patch_size=16, in_channels=3, 
               num_classes=1000, embed_dim=768, depth=12, num_heads=12,
               mlp_ratio=4.0, dropout=0.1):
      super().__init__()
      
      # Patch embedding
      self.patch_embed = PatchEmbedding(img_size, patch_size, 
                                       in_channels, embed_dim)
      num_patches = self.patch_embed.num_patches
      
      # Class token (learnable)
      self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
      
      # Positional embeddings (learnable)
      self.pos_embed = nn.Parameter(
          torch.randn(1, num_patches + 1, embed_dim)
      )
      
      self.dropout = nn.Dropout(dropout)
      
      # Transformer encoder blocks
      self.blocks = nn.ModuleList([
          TransformerEncoder(embed_dim, num_heads, mlp_ratio, dropout)
          for _ in range(depth)
      ])
      
      self.norm = nn.LayerNorm(embed_dim)
      
      # Classification head
      self.head = nn.Linear(embed_dim, num_classes)
      
  def forward(self, x):
      B = x.shape[0]
      
      # Patch embedding: [B, C, H, W] -> [B, num_patches, embed_dim]
      x = self.patch_embed(x)
      
      # Prepend class token: [B, num_patches, D] -> [B, num_patches+1, D]
      cls_tokens = self.cls_token.expand(B, -1, -1)
      x = torch.cat([cls_tokens, x], dim=1)
      
      # Add positional embeddings
      x = x + self.pos_embed
      x = self.dropout(x)
      
      # Apply Transformer blocks
      for block in self.blocks:
          x = block(x)
          
      x = self.norm(x)
      
      # Extract class token and classify
      cls_token_final = x[:, 0]
      logits = self.head(cls_token_final)
      
      return logits

# Create ViT-Base/16
model = VisionTransformer(
  img_size=224,
  patch_size=16,
  num_classes=1000,
  embed_dim=768,
  depth=12,
  num_heads=12
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# Forward pass
x = torch.randn(2, 3, 224, 224)  # Batch of 2 images
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")  # [2, 1000] class logits`,
      explanation: 'Complete Vision Transformer implementation showing patch embedding, transformer encoder blocks, and classification head.'
    },
    {
      language: 'Python',
      code: `from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# Load pre-trained ViT model
model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)
model.eval()

# Load and preprocess image
image = Image.open("cat.jpg")
inputs = processor(images=image, return_tensors="pt")

# Forward pass
with torch.no_grad():
  outputs = model(**inputs)
  logits = outputs.logits
  predicted_class = logits.argmax(-1).item()

print(f"Predicted class: {model.config.id2label[predicted_class]}")
print(f"Logits shape: {logits.shape}")  # [1, 1000]

# === Visualize attention maps ===
from transformers import ViTModel
import matplotlib.pyplot as plt

# Load model with attention outputs
vit_model = ViTModel.from_pretrained(model_name, output_attentions=True)
vit_model.eval()

with torch.no_grad():
  outputs = vit_model(**inputs)
  attentions = outputs.attentions  # Tuple of attention weights per layer

# Attention shape: [batch, num_heads, seq_len, seq_len]
# seq_len = num_patches + 1 (for class token)

# Visualize attention from class token in last layer
last_layer_attn = attentions[-1]  # Last layer
cls_attn = last_layer_attn[0, :, 0, 1:]  # [num_heads, num_patches]

# Average over heads
cls_attn_avg = cls_attn.mean(0)  # [num_patches]

# Reshape to 2D (14x14 for 224x224 image with 16x16 patches)
num_patches_per_dim = 14
attn_map = cls_attn_avg.reshape(num_patches_per_dim, num_patches_per_dim)

# Plot
plt.figure(figsize=(8, 8))
plt.imshow(attn_map.cpu(), cmap='viridis')
plt.title('Attention from [CLS] token (Last Layer)')
plt.colorbar()
plt.savefig('vit_attention.png')

print(f"Number of layers: {len(attentions)}")
print(f"Attention shape per layer: {attentions[0].shape}")`,
      explanation: 'Using pre-trained Vision Transformer from Hugging Face for image classification and visualizing attention patterns.'
    },
    {
      language: 'Python',
      code: `import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data augmentation for ViT training
train_transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.RandomHorizontalFlip(),
  transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
  transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                      std=[0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, 
                               download=True, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Create ViT model
from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained(
  'google/vit-base-patch16-224',
  num_labels=10,  # CIFAR-10 has 10 classes
  ignore_mismatched_sizes=True  # Adjust classification head
)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(10):
  total_loss = 0
  correct = 0
  total = 0
  
  for batch_idx, (images, labels) in enumerate(train_loader):
      images, labels = images.to(device), labels.to(device)
      
      # Forward pass
      outputs = model(images).logits
      loss = criterion(outputs, labels)
      
      # Backward pass
      optimizer.zero_grad()
      loss.backward()
      
      # Gradient clipping
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      
      optimizer.step()
      
      # Track metrics
      total_loss += loss.item()
      _, predicted = outputs.max(1)
      total += labels.size(0)
      correct += predicted.eq(labels).sum().item()
      
      if batch_idx % 100 == 0:
          print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%")
  
  scheduler.step()
  
  print(f"\\nEpoch {epoch}: Loss: {total_loss/len(train_loader):.4f} "
        f"Acc: {100.*correct/total:.2f}%\\n")

# Save fine-tuned model
model.save_pretrained('./vit_cifar10_finetuned')`,
      explanation: 'Fine-tuning a pre-trained Vision Transformer on CIFAR-10 with proper data augmentation and training techniques.'
    }
  ],
  interviewQuestions: [
    {
      question: 'How does Vision Transformer handle 2D images when Transformers are designed for 1D sequences?',
      answer: `ViT splits the image into fixed-size patches (e.g., $16\\times16$ pixels), flattens each patch into a vector, and linearly projects them into embeddings. This converts a 2D image $(H\\times W\\times C)$ into a 1D sequence of $N$ patch embeddings, where $N = (H\\times W)/(P^2)$. A learnable [CLS] token is prepended for classification. Position embeddings (typically 1D learned embeddings) are added to preserve spatial information. The resulting sequence is then processed by standard Transformer encoder blocks, identical to those used in NLP.`
    },
    {
      question: 'Why does ViT require more training data than CNNs to achieve comparable performance?',
      answer: `ViT lacks the inductive biases built into CNNs—locality (pixels near each other are related), translation invariance (features should be detected anywhere), and hierarchical structure (low-level to high-level features). CNNs encode these biases through convolution operations, helping them learn efficiently from smaller datasets. ViT must learn these spatial relationships from data through attention, requiring more examples. However, this flexibility becomes an advantage with sufficient data—ViT scales better than CNNs on large datasets (100M+ images), ultimately achieving superior performance.`
    },
    {
      question: 'Explain the computational complexity of ViT compared to CNNs.',
      answer: `ViT's self-attention has $O(N^2\\cdot D)$ complexity where $N$ is the number of patches and $D$ is embedding dimension. For a $224\\times224$ image with $16\\times16$ patches, $N=196$, making attention $O(38K\\cdot D)$. CNNs have $O(K^2\\cdot C\\cdot H\\cdot W)$ per layer where $K$ is kernel size. The key difference: attention is quadratic in sequence length (number of patches), so high-resolution images become expensive. CNNs scale linearly with spatial resolution but need many layers for global receptive fields. Variants like Swin Transformer use windowed attention to achieve $O(N\\cdot D)$ complexity, making them more practical for dense prediction tasks.`
    },
    {
      question: 'What role does the [CLS] token play in Vision Transformer?',
      answer: `The [CLS] token, borrowed from BERT, is a learnable embedding prepended to the patch sequence. Through self-attention across all layers, it can aggregate information from all image patches. The final layer's [CLS] token representation is used for classification by passing it through a linear layer with softmax. This provides a single vector summarizing the entire image. Alternatively, some implementations use global average pooling over all patch tokens, which performs similarly. The [CLS] token approach allows the model to learn what information to aggregate for classification through attention weights.`
    },
    {
      question: 'How does ViT handle images of different resolutions at inference time?',
      answer: `ViT can handle different resolutions through positional embedding interpolation. If trained at $224\\times224$ (196 patches with P=16) but testing at $384\\times384$ (576 patches), we interpolate the learned positional embeddings from 196 to 576 dimensions using 2D interpolation. This works because patch positions are spatially meaningful. The model can then process longer sequences. Fine-tuning at higher resolution after pre-training at lower resolution is a common strategy—the model benefits from higher resolution details while leveraging pre-trained weights. This flexibility is an advantage over CNNs with fixed-size pooling layers.`
    },
    {
      question: 'What are the key differences between ViT and Swin Transformer?',
      answer: `ViT uses global self-attention where every patch attends to every other patch (flat architecture), while Swin uses hierarchical windowed attention. Swin computes attention within local windows (e.g., $7\\times7$ patches), then shifts windows between layers to enable cross-window connections. This reduces complexity from $O(N^2)$ to $O(N)$ in sequence length. Swin also creates hierarchical feature maps by merging patches across stages (like CNN feature pyramids), making it suitable for dense prediction tasks (detection, segmentation). ViT is simpler and more similar to NLP Transformers, but Swin is more efficient and versatile for diverse vision tasks.`
    }
  ],
  quizQuestions: [
    {
      id: 'vit1',
      question: 'How does Vision Transformer convert an image into a sequence?',
      options: ['Pixel by pixel', 'Row by row', 'Split into patches and embed', 'Use CNN features'],
      correctAnswer: 2,
      explanation: 'ViT splits the image into fixed-size patches (e.g., $16\\times16$), flattens each patch, and projects them through a learned linear transformation into embeddings, creating a sequence of patch embeddings.'
    },
    {
      id: 'vit2',
      question: 'What is the primary reason ViT needs more training data than CNNs?',
      options: ['Larger model size', 'Slower training', 'Lacks convolutional inductive biases', 'More parameters'],
      correctAnswer: 2,
      explanation: 'ViT lacks the inductive biases built into CNNs (locality, translation invariance, hierarchy), so it must learn spatial relationships from data. With sufficient data, this flexibility becomes an advantage.'
    },
    {
      id: 'vit3',
      question: 'What is the computational complexity of self-attention in ViT?',
      options: ['$O(N)$', '$O(N \\log N)$', '$O(N^2)$', '$O(N^3)$'],
      correctAnswer: 2,
      explanation: 'Self-attention in ViT has $O(N^2)$ complexity where $N$ is the number of patches, because each patch must attend to every other patch, creating an $N\\times N$ attention matrix.'
    },
    {
      id: 'vit4',
      question: 'What is the purpose of the [CLS] token in Vision Transformer?',
      options: ['Mark patch boundaries', 'Aggregate global image information for classification', 'Store positional encodings', 'Enable convolution'],
      correctAnswer: 1,
      explanation: 'The [CLS] token is prepended to the patch sequence and aggregates information from all patches through self-attention. Its final representation is used for image classification.'
    },
    {
      id: 'vit5',
      question: 'What patch size is commonly used in the original Vision Transformer?',
      options: ['$8\\times8$', '$16\\times16$', '$32\\times32$', '$64\\times64$'],
      correctAnswer: 1,
      explanation: 'The original ViT paper commonly uses $16\\times16$ pixel patches, which provides a good balance between sequence length and patch detail.'
    },
    {
      id: 'vit6',
      question: 'How are positional embeddings typically handled in ViT?',
      options: ['Not needed', 'Sinusoidal encodings', '1D learnable embeddings added to patch embeddings', '2D convolutions'],
      correctAnswer: 2,
      explanation: 'ViT uses 1D learnable positional embeddings that are added to patch embeddings. These are learned during training to encode spatial relationships.'
    },
    {
      id: 'vit7',
      question: 'Can Vision Transformer handle different image resolutions at test time?',
      options: ['No, fixed resolution only', 'Yes, through positional embedding interpolation', 'Only smaller resolutions', 'Requires retraining'],
      correctAnswer: 1,
      explanation: 'ViT can handle different resolutions by interpolating the learned positional embeddings. For example, trained on $224\\times224$ but tested on $384\\times384$ using 2D interpolation.'
    },
    {
      id: 'vit8',
      question: 'What is the main difference between ViT and Swin Transformer?',
      options: ['Number of layers', 'ViT uses global attention, Swin uses local windowed attention', 'Training data size', 'Activation functions'],
      correctAnswer: 1,
      explanation: 'ViT applies global self-attention across all patches, while Swin uses hierarchical local windowed attention for better efficiency and creates multi-scale feature maps.'
    },
    {
      id: 'vit9',
      question: 'What dataset did the original ViT paper use for pre-training to achieve best results?',
      options: ['ImageNet-1K', 'CIFAR-100', 'JFT-300M (300 million images)', 'COCO'],
      correctAnswer: 2,
      explanation: 'ViT achieved its best performance when pre-trained on JFT-300M, a Google internal dataset with 300 million images. This demonstrated ViT\'s need for large-scale data.'
    },
    {
      id: 'vit10',
      question: 'How many patches does a $224\\times224$ image produce with $16\\times16$ patches?',
      options: ['49', '196', '256', '784'],
      correctAnswer: 1,
      explanation: 'A $224\\times224$ image divided into $16\\times16$ patches produces $(224/16)^2 = 14^2 = 196$ patches.'
    },
    {
      id: 'vit11',
      question: 'What inductive bias from CNNs does ViT NOT have?',
      options: ['Depth', 'Translation invariance and locality', 'Activation functions', 'Normalization'],
      correctAnswer: 1,
      explanation: 'ViT lacks CNN inductive biases like translation invariance (features detected anywhere) and locality (nearby pixels are related). It must learn these from data.'
    },
    {
      id: 'vit12',
      question: 'What architecture component does ViT use from standard Transformers?',
      options: ['Modified attention', 'Standard Transformer encoder blocks with multi-head self-attention', 'LSTM layers', 'Convolutional layers'],
      correctAnswer: 1,
      explanation: 'ViT uses standard Transformer encoder blocks identical to NLP models (multi-head attention + MLP + layer norm + residuals), without vision-specific modifications.'
    },
    {
      id: 'vit13',
      question: 'What happens to patch embeddings before entering the Transformer encoder?',
      options: ['Nothing', 'Linear projection + positional embeddings + [CLS] token prepended', 'Convolution', 'Pooling'],
      correctAnswer: 1,
      explanation: 'Each flattened patch is linearly projected to the model dimension, positional embeddings are added, and the [CLS] token is prepended to form the input sequence.'
    },
    {
      id: 'vit14',
      question: 'How does ViT perform on small datasets compared to CNNs?',
      options: ['Always better', 'Worse than CNNs due to lack of inductive biases', 'Same performance', 'Faster training'],
      correctAnswer: 1,
      explanation: 'On small datasets, ViT underperforms CNNs because it lacks built-in inductive biases. ViT needs large-scale pre-training to compensate for this.'
    },
    {
      id: 'vit15',
      question: 'What is a common alternative to using the [CLS] token for classification?',
      options: ['Use first patch', 'No alternatives', 'Global average pooling over all patch tokens', 'Max pooling'],
      correctAnswer: 2,
      explanation: 'Instead of the [CLS] token, some implementations use global average pooling over all patch representations, which performs similarly.'
    },
    {
      id: 'vit16',
      question: 'What is DeiT (Data-efficient image Transformers)?',
      options: ['Larger ViT', 'ViT variant using distillation for efficient training on ImageNet', 'CNN model', 'Slower ViT'],
      correctAnswer: 1,
      explanation: 'DeiT uses knowledge distillation from a CNN teacher and improved data augmentation to train ViT efficiently on ImageNet-1K without massive pre-training datasets.'
    },
    {
      id: 'vit17',
      question: 'What attention pattern do ViT models learn in early layers?',
      options: ['Random patterns', 'Local patterns similar to CNN receptive fields', 'Only global', 'No patterns'],
      correctAnswer: 1,
      explanation: 'Analysis shows ViT early layers learn to attend to local neighborhoods similar to CNN receptive fields, even without being explicitly designed to do so.'
    },
    {
      id: 'vit18',
      question: 'Can Vision Transformers be used for tasks beyond classification?',
      options: ['No, classification only', 'Yes, for detection, segmentation, and other vision tasks', 'Only with CNNs', 'Never effective'],
      correctAnswer: 1,
      explanation: 'ViT and variants (DETR, Mask2Former, Segmenter) are used successfully for object detection, semantic segmentation, instance segmentation, and more.'
    },
    {
      id: 'vit19',
      question: 'What is the role of layer normalization in ViT?',
      options: ['No role', 'Applied before attention and MLP (pre-norm), stabilizes training', 'Only after attention', 'Replaces attention'],
      correctAnswer: 1,
      explanation: 'ViT uses pre-normalization (layer norm before each sub-layer) which stabilizes training of deep Transformers and enables better gradient flow.'
    },
    {
      id: 'vit20',
      question: 'What is the typical embedding dimension used in ViT-Base?',
      options: ['256', '512', '768', '1024'],
      correctAnswer: 2,
      explanation: 'ViT-Base uses 768-dimensional embeddings (same as BERT-Base), with 12 layers and 12 attention heads. ViT-Large uses 1024 dimensions.'
    }
  ]
};
