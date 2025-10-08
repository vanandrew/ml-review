import { Topic } from '../../../types';

export const generativeAdversarialNetworks: Topic = {
  id: 'generative-adversarial-networks',
  title: 'Generative Adversarial Networks (GANs)',
  category: 'advanced',
  description: 'Two neural networks competing to generate realistic data',
  content: `
    <h2>Generative Adversarial Networks: Adversarial Learning</h2>
    <p>Generative Adversarial Networks (GANs), introduced by Ian Goodfellow and colleagues in 2014, revolutionized generative modeling by framing it as a two-player game. Instead of explicitly modeling the data distribution, GANs pit two neural networks against each other: a generator that creates fake data, and a discriminator that distinguishes real from fake. This adversarial competition drives both networks to improve, ultimately producing remarkably realistic synthetic data. GANs have achieved breakthrough results in image generation, producing photorealistic faces, artwork, and enabling applications from data augmentation to creative AI. However, training GANs is notoriously challenging, requiring careful balancing of the adversarial dynamics and sophisticated techniques to ensure stability and diversity.</p>

    <h3>The Core Concept: Adversarial Training</h3>

    <h4>Generative Modeling Challenge</h4>
    <p><strong>Goal:</strong> Learn to generate new samples from a data distribution p_data(x) given only training examples.</p>

    <h5>Traditional Approaches</h5>
    <ul>
      <li><strong>Maximum likelihood:</strong> Explicitly model p(x), maximize likelihood on training data (VAEs, autoregressive models)</li>
      <li><strong>Challenge:</strong> Difficult to specify tractable likelihood for complex distributions (high-dimensional images)</li>
      <li><strong>Results:</strong> Often produces blurry samples due to averaging over multiple modes</li>
    </ul>

    <h5>GAN Innovation</h5>
    <ul>
      <li><strong>Implicit density:</strong> Never explicitly model p(x), only learn to sample from it</li>
      <li><strong>Adversarial signal:</strong> Use discriminator as learned loss function that adapts during training</li>
      <li><strong>Sharp outputs:</strong> Adversarial training produces crisp, realistic samples</li>
      <li><strong>Game theory framework:</strong> Training as Nash equilibrium of two-player game</li>
    </ul>

    <h4>The Two-Player Game</h4>
    <p>GAN training is a minimax game between two networks with opposing objectives:</p>

    <h5>Visual Overview</h5>
    <pre class="code-block">
┌─────────────────────────────────────────────────────────┐
│                    GAN Training Loop                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Noise z ~ N(0,I)                                       │
│       │                                                 │
│       ▼                                                 │
│  ┌──────────┐         Fake Image                        │
│  │Generator │────────────────────┐                      │
│  │    G     │                    │                      │
│  └──────────┘                    ▼                      │
│                            ┌─────────────┐              │
│  Real Image                │             │  Real/Fake   │
│  from Dataset ────────────►│Discriminator│─────────►    │
│                            │      D      │  Probability │
│                            └─────────────┘              │
│                                   │                     │
│                                   │                     │
│         ┌─────────────────────────┴──────────┐          │
│         ▼                                    ▼          │
│   Update D to                          Update G to      │
│   maximize detection                   fool D           │
└─────────────────────────────────────────────────────────┘
    </pre>

    <h5>Player 1: Generator G</h5>
    <ul>
      <li><strong>Input:</strong> Random noise vector z ∼ p_z(z) from latent space (typically Gaussian or uniform)</li>
      <li><strong>Output:</strong> Fake sample G(z) attempting to mimic real data</li>
      <li><strong>Objective:</strong> Fool the discriminator—make D(G(z)) close to 1 (discriminator thinks it's real)</li>
      <li><strong>Architecture:</strong> Decoder/upsampling network (e.g., transposed convolutions for images)</li>
      <li><strong>Interpretation:</strong> Artist creating forgeries</li>
    </ul>

    <h5>Player 2: Discriminator D</h5>
    <ul>
      <li><strong>Input:</strong> Either real sample x ∼ p_data or fake sample G(z)</li>
      <li><strong>Output:</strong> Probability $D(x) \\in [0,1]$ that input is real</li>
      <li><strong>Objective:</strong> Correctly classify real vs fake—$D(x) \\approx 1$ for real, $D(G(z)) \\approx 0$ for fake</li>
      <li><strong>Architecture:</strong> Encoder/classifier network (e.g., convolutional layers for images)</li>
      <li><strong>Interpretation:</strong> Art detective identifying forgeries</li>
    </ul>

    <h4>The Minimax Objective</h4>
    <p><strong>Value function V(G, D):</strong></p>
    <p style="text-align: center; font-size: 1.1em;">
      $\\min_G \\max_D V(D, G) = \\mathbb{E}_{x\\sim p_{\\text{data}}}[\\log D(x)] + \\mathbb{E}_{z\\sim p_z}[\\log(1 - D(G(z)))]$
    </p>

    <h5>Breaking Down the Objective</h5>
    <ul>
      <li><strong>First term 𝔼[log D(x)]:</strong> Discriminator's ability to identify real samples (wants to maximize)</li>
      <li><strong>Second term 𝔼[log(1 - D(G(z)))]:</strong> Discriminator's ability to identify fakes (wants to maximize)</li>
      <li><strong>Discriminator maximizes V:</strong> Improve classification accuracy</li>
      <li><strong>Generator minimizes V:</strong> Reduce discriminator's confidence on fakes</li>
    </ul>

    <h5>Game-Theoretic Interpretation</h5>
    <ul>
      <li><strong>Nash equilibrium:</strong> Neither player can improve by changing strategy unilaterally</li>
      <li><strong>Optimal solution:</strong> p_g = p_data (generator distribution matches data distribution)</li>
      <li><strong>At equilibrium:</strong> D(x) = 1/2 everywhere (discriminator cannot distinguish real from fake)</li>
    </ul>

    <h3>GAN Architecture Components</h3>

    <h4>Generator Network Design</h4>

    <h5>Latent Space</h5>
    <ul>
      <li><strong>Dimension:</strong> Typically 100-512 dimensional vector</li>
      <li><strong>Distribution:</strong> Uniform [-1,1] or Gaussian N(0, I)</li>
      <li><strong>Role:</strong> Encodes variation in generated samples (different z → different outputs)</li>
      <li><strong>Continuity:</strong> Smooth latent space enables interpolation between samples</li>
    </ul>

    <h5>Architecture for Images (DCGAN Guidelines)</h5>
    <ul>
      <li><strong>Transposed convolutions:</strong> Upsample from low to high resolution</li>
      <li><strong>Batch normalization:</strong> After each layer (stabilizes training, prevents mode collapse)</li>
      <li><strong>ReLU activations:</strong> In all layers except output</li>
      <li><strong>Tanh output:</strong> Produces values in [-1, 1] matching normalized images</li>
      <li><strong>No fully connected layers:</strong> All convolutional for spatial structure</li>
      <li><strong>Progressive upsampling:</strong> 4×4 → 8×8 → 16×16 → 32×32 → 64×64 (example for 64×64 output)</li>
    </ul>

    <h5>Example Generator Flow (64×64 Images)</h5>
    <pre>
Latent z (100-dim vector) 
  → Dense(512×4×4) + Reshape to (512, 4, 4)           [512 channels, 4×4]
  → TransposeConv(512→256, k=4, s=2, p=1) + BN + ReLU  [256 channels, 8×8]
  → TransposeConv(256→128, k=4, s=2, p=1) + BN + ReLU  [128 channels, 16×16]
  → TransposeConv(128→64,  k=4, s=2, p=1) + BN + ReLU  [64 channels, 32×32]
  → TransposeConv(64→3,    k=4, s=2, p=1) + Tanh       [3 channels, 64×64]
  → RGB Image (3, 64, 64)

<strong>Parameters:</strong> k=kernel size, s=stride, p=padding
<strong>Total parameters:</strong> ~3.5M for this architecture
    </pre>

    <h4>Discriminator Network Design</h4>

    <h5>Architecture (DCGAN Guidelines)</h5>
    <ul>
      <li><strong>Convolutional layers:</strong> Downsample from high to low resolution</li>
      <li><strong>Strided convolutions:</strong> Replace pooling for downsampling</li>
      <li><strong>LeakyReLU activations:</strong> $\\alpha=0.2$, prevents dead neurons</li>
      <li><strong>Batch normalization:</strong> After each layer except first</li>
      <li><strong>No fully connected layers:</strong> Until final classification layer</li>
      <li><strong>Sigmoid output:</strong> Single probability score $D(x) \\in [0,1]$</li>
    </ul>

    <h5>Example Discriminator Flow</h5>
    <pre>
RGB Image (3, 64, 64)
  → Conv 3→64, stride 2 → (64, 32, 32) + LeakyReLU (no BN on first layer)
  → Conv 64→128, stride 2 → (128, 16, 16) + BN + LeakyReLU
  → Conv 128→256, stride 2 → (256, 8, 8) + BN + LeakyReLU
  → Conv 256→512, stride 2 → (512, 4, 4) + BN + LeakyReLU
  → FC 512×4×4 → 1 + Sigmoid
  → Probability [0, 1]
    </pre>

    <h3>Training Procedure</h3>

    <h4>Alternating Optimization</h4>
    <p>Train discriminator and generator in alternating steps:</p>

    <h5>Algorithm: GAN Training Loop</h5>
    <pre>
for epoch in range(num_epochs):
  for batch in dataloader:
      # ===== Train Discriminator (k steps) =====
      for _ in range(k):  # k typically 1-5
          # Sample real batch
          x_real ~ p_data
          
          # Sample noise and generate fakes
          z ~ p_z(z)
          x_fake = G(z)
          
          # Discriminator forward pass
          D_real = D(x_real)
          D_fake = D(x_fake.detach())  # Detach to avoid G gradient
          
          # Discriminator loss
          L_D = -[log(D_real) + log(1 - D_fake)]
          
          # Update discriminator
          L_D.backward()
          optimizer_D.step()
      
      # ===== Train Generator (1 step) =====
      # Generate fakes
      z ~ p_z(z)
      x_fake = G(z)
      
      # Discriminator evaluation
      D_fake = D(x_fake)
      
      # Generator loss (non-saturating variant)
      L_G = -log(D_fake)
      
      # Update generator
      L_G.backward()
      optimizer_G.step()
    </pre>

    <h4>Loss Functions</h4>

    <h5>Discriminator Loss</h5>
    <p><strong>Binary cross-entropy for real vs fake classification:</strong></p>
    <p>$L_D = -\\mathbb{E}_{x\\sim p_{\\text{data}}}[\\log D(x)] - \\mathbb{E}_{z\\sim p_z}[\\log(1 - D(G(z)))]$</p>
    <ul>
      <li><strong>Maximize D(x) for real samples:</strong> Penalize when D(x) < 1</li>
      <li><strong>Minimize D(G(z)) for fakes:</strong> Penalize when D(G(z)) > 0</li>
      <li><strong>Interpretation:</strong> Standard supervised learning for binary classification</li>
    </ul>

    <h5>Generator Loss: Two Formulations</h5>

    <h6>1. Original Minimax (Saturating)</h6>
    <p>$L_G = \\mathbb{E}_{z\\sim p_z}[\\log(1 - D(G(z)))]$</p>
    <ul>
      <li><strong>Problem:</strong> When D is confident fake ($D(G(z)) \\approx 0$), gradient $\\approx 0$ (vanishing gradient)</li>
      <li><strong>Early training:</strong> Generator produces obviously fake samples, discriminator easily rejects them</li>
      <li><strong>Result:</strong> Generator receives little learning signal</li>
    </ul>

    <h6>2. Non-Saturating (Standard Practice)</h6>
    <p>$L_G = -\\mathbb{E}_{z\\sim p_z}[\\log D(G(z))]$</p>
    <ul>
      <li><strong>Maximize log D(G(z)):</strong> Stronger gradient when D is confident fake</li>
      <li><strong>Same optimal point:</strong> Still minimizes JS divergence at equilibrium</li>
      <li><strong>Better gradients:</strong> Generator learns faster early in training</li>
      <li><strong>Standard practice:</strong> Used in most implementations</li>
    </ul>

    <h4>Training Ratio: k Discriminator Steps per Generator Step</h4>
    <ul>
      <li><strong>k=1 (most common):</strong> Balanced training, one D update per G update</li>
      <li><strong>k>1 (e.g., 5):</strong> Train D more to provide better gradient signal for G</li>
      <li><strong>k<1 (rare):</strong> Train G more if D becomes too strong</li>
      <li><strong>Rule of thumb:</strong> Keep D slightly ahead but not too strong (prevents vanishing gradients)</li>
    </ul>

    <h3>Training Challenges and Solutions</h3>

    <h4>1. Mode Collapse</h4>

    <h5>Problem</h5>
    <ul>
      <li><strong>Symptom:</strong> Generator produces limited variety, ignores large portions of data distribution</li>
      <li><strong>Cause:</strong> Generator finds a few samples that fool D, exploits them rather than exploring diversity</li>
      <li><strong>Example:</strong> MNIST GAN producing only 2-3 digits repeatedly</li>
      <li><strong>Theoretical issue:</strong> G optimizes independently each step, doesn't account for D's adaptation</li>
    </ul>

    <h5>Solutions</h5>
    <ul>
      <li><strong>Minibatch discrimination:</strong> Discriminator compares samples within batch, penalizes lack of diversity</li>
      <li><strong>Feature matching:</strong> Match statistics of intermediate layers rather than fool D directly</li>
      <li><strong>Unrolled GANs:</strong> Optimize G considering k future D updates (computationally expensive)</li>
      <li><strong>Multiple discriminators:</strong> Use ensemble to prevent exploiting single D weakness</li>
      <li><strong>Experience replay:</strong> D sees past generated samples, prevents G from "forgetting" modes</li>
    </ul>

    <h4>2. Vanishing Gradients</h4>

    <h5>Problem</h5>
    <ul>
      <li><strong>Cause:</strong> When D becomes too strong, $D(G(z)) \\approx 0$, gradient for G vanishes</li>
      <li><strong>Effect:</strong> Generator stops learning, training stalls</li>
      <li><strong>Occurs:</strong> Early training when G produces obviously fake samples</li>
    </ul>

    <h5>Solutions</h5>
    <ul>
      <li><strong>Non-saturating loss:</strong> Use -log D(G(z)) instead of log(1 - D(G(z)))</li>
      <li><strong>Balance D/G training:</strong> Don't let D get too far ahead</li>
      <li><strong>Label smoothing:</strong> Use soft labels (0.9 instead of 1.0) to prevent D overconfidence</li>
      <li><strong>Noise injection:</strong> Add noise to D inputs to prevent overfitting</li>
    </ul>

    <h4>3. Training Instability</h4>

    <h5>Problem</h5>
    <ul>
      <li><strong>Oscillations:</strong> Loss oscillates without converging</li>
      <li><strong>Collapse:</strong> Training suddenly fails after initial progress</li>
      <li><strong>Sensitivity:</strong> Small hyperparameter changes cause failure</li>
    </ul>

    <h5>Solutions</h5>
    <ul>
      <li><strong>Architecture best practices:</strong> Follow DCGAN guidelines (BatchNorm, LeakyReLU, etc.)</li>
      <li><strong>Spectral normalization:</strong> Constrain discriminator Lipschitz constant</li>
      <li><strong>Two-timescale update rule (TTUR):</strong> Different learning rates for G and D</li>
      <li><strong>Gradient penalty:</strong> Regularize discriminator gradient norm (WGAN-GP)</li>
      <li><strong>Self-attention:</strong> Improve long-range dependencies (SAGAN)</li>
      <li><strong>Progressive growing:</strong> Gradually increase resolution (ProGAN)</li>
    </ul>

    <h3>Major GAN Variants</h3>

    <h4>DCGAN (Deep Convolutional GAN, 2015)</h4>
    <ul>
      <li><strong>Contribution:</strong> First stable architecture for training GANs on images</li>
      <li><strong>Guidelines:</strong> Use strided/transposed convolutions instead of pooling, BatchNorm, ReLU/LeakyReLU</li>
      <li><strong>Impact:</strong> Established best practices, enabled high-quality image generation</li>
      <li><strong>Results:</strong> Generated 64×64 bedroom images with realistic details</li>
    </ul>

    <h4>WGAN (Wasserstein GAN, 2017)</h4>

    <h5>Core Innovation</h5>
    <ul>
      <li><strong>Wasserstein distance:</strong> Replace JS divergence with Earth Mover's distance</li>
      <li><strong>Benefits:</strong> Meaningful loss correlating with sample quality, improved training stability</li>
      <li><strong>Lipschitz constraint:</strong> Discriminator must be 1-Lipschitz continuous</li>
    </ul>

    <h5>Implementation</h5>
    <ul>
      <li><strong>Weight clipping (WGAN):</strong> Clip discriminator weights to [-c, c] after each update</li>
      <li><strong>Gradient penalty (WGAN-GP):</strong> Soft constraint via penalty on gradient norm (better than clipping)</li>
      <li><strong>Remove sigmoid:</strong> Discriminator outputs unbounded score (critic), not probability</li>
      <li><strong>Loss:</strong> $L_D = \\mathbb{E}[D(x)] - \\mathbb{E}[D(G(z))] + \\lambda \\cdot GP$, $L_G = -\\mathbb{E}[D(G(z))]$</li>
    </ul>

    <h5>Advantages</h5>
    <ul>
      <li><strong>Correlation with quality:</strong> Loss decreases as samples improve</li>
      <li><strong>No mode collapse:</strong> Wasserstein distance covers all modes</li>
      <li><strong>Stable training:</strong> Works with variety of architectures</li>
      <li><strong>Useful for monitoring:</strong> Loss indicates training progress</li>
    </ul>

    <h4>Conditional GAN (cGAN, 2014)</h4>

    <h5>Motivation</h5>
    <ul>
      <li><strong>Problem:</strong> Standard GAN has no control over generated samples</li>
      <li><strong>Solution:</strong> Condition generation on auxiliary information (class labels, text, images)</li>
    </ul>

    <h5>Architecture</h5>
    <ul>
      <li><strong>Generator:</strong> G(z, y) where y is conditioning information (e.g., class label)</li>
      <li><strong>Discriminator:</strong> D(x, y) evaluates if x is real given condition y</li>
      <li><strong>Implementation:</strong> Concatenate y as additional input (embedding for labels, feature map for images)</li>
    </ul>

    <h5>Applications</h5>
    <ul>
      <li><strong>Class-conditional generation:</strong> Generate specific MNIST digit, ImageNet class</li>
      <li><strong>Image-to-image translation:</strong> Pix2Pix (edges→photos, day→night)</li>
      <li><strong>Text-to-image:</strong> Generate images from text descriptions</li>
      <li><strong>Super-resolution:</strong> Low-res image → high-res image</li>
    </ul>

    <h4>StyleGAN (2018) and StyleGAN2 (2019)</h4>

    <h5>Architecture Innovation</h5>
    <ul>
      <li><strong>Style-based generator:</strong> Latent code controls "style" at different scales via AdaIN</li>
      <li><strong>Progressive synthesis:</strong> Generate from coarse to fine (4×4 → 8×8 → ... → 1024×1024)</li>
      <li><strong>Mapping network:</strong> Map latent z to intermediate latent w (more disentangled)</li>
      <li><strong>Style mixing:</strong> Apply different latent codes at different layers for fine-grained control</li>
    </ul>

    <h5>Results</h5>
    <ul>
      <li><strong>Quality:</strong> Photorealistic faces at 1024×1024 resolution</li>
      <li><strong>Controllability:</strong> Manipulate specific attributes (age, gender, expression) via latent directions</li>
      <li><strong>Applications:</strong> "This Person Does Not Exist", deepfakes, art generation</li>
      <li><strong>Impact:</strong> State-of-the-art generative model for faces</li>
    </ul>

    <h4>CycleGAN (2017)</h4>

    <h5>Problem: Unpaired Image-to-Image Translation</h5>
    <ul>
      <li><strong>Goal:</strong> Translate images between domains (horses ↔ zebras) without paired examples</li>
      <li><strong>Challenge:</strong> No supervision signal for translation</li>
    </ul>

    <h5>Solution: Cycle-Consistency</h5>
    <ul>
      <li><strong>Two generators:</strong> G: X→Y and F: Y→X</li>
      <li><strong>Two discriminators:</strong> D_X and D_Y for each domain</li>
      <li><strong>Cycle-consistency loss:</strong> $F(G(x)) \\approx x$ and $G(F(y)) \\approx y$ (reconstruction after round-trip)</li>
      <li><strong>Intuition:</strong> Translation must be reversible, preserving content</li>
    </ul>

    <h5>Applications</h5>
    <ul>
      <li><strong>Artistic style transfer:</strong> Photos → paintings (Monet, Van Gogh)</li>
      <li><strong>Object transfiguration:</strong> Horses → zebras, apples → oranges</li>
      <li><strong>Season transfer:</strong> Summer → winter scenes</li>
      <li><strong>Domain adaptation:</strong> Synthetic → real for training data augmentation</li>
    </ul>

    <h4>BigGAN (2018)</h4>
    <ul>
      <li><strong>Scale:</strong> Large batch sizes (2048), large models (160M params)</li>
      <li><strong>Techniques:</strong> Class-conditional BatchNorm, truncation trick, self-attention</li>
      <li><strong>Results:</strong> State-of-the-art on ImageNet 128×128 and 512×512</li>
      <li><strong>FID score:</strong> 6.95 on ImageNet 128×128 (lower is better)</li>
    </ul>

    <h3>Evaluation Metrics</h3>

    <h4>The Evaluation Challenge</h4>
    <p>No ground truth distribution, samples can look good but miss modes or have artifacts.</p>

    <h4>Inception Score (IS)</h4>
    <ul>
      <li><strong>Idea:</strong> Good samples should be confident (low entropy per sample) and diverse (high entropy overall)</li>
      <li><strong>Computation:</strong> Use pre-trained Inception classifier, IS = $\\exp(\\mathbb{E}_x[\\text{KL}(p(y|x) || p(y))])$</li>
      <li><strong>Interpretation:</strong> Higher is better, typical range 2-12 for ImageNet</li>
      <li><strong>Limitations:</strong> Biased toward Inception training data, ignores spatial statistics, can be gamed</li>
    </ul>

    <h4>Fréchet Inception Distance (FID)</h4>
    <ul>
      <li><strong>Idea:</strong> Compare statistics of generated vs real samples in Inception feature space</li>
      <li><strong>Computation:</strong> Fit Gaussian to features, compute Fréchet distance: $\\text{FID} = ||\\mu_r - \\mu_g||^2 + \\text{Tr}(\\Sigma_r + \\Sigma_g - 2\\sqrt{\\Sigma_r \\Sigma_g})$</li>
      <li><strong>Interpretation:</strong> Lower is better, 0 = perfect match, typical <50 for good models</li>
      <li><strong>Advantages:</strong> More robust than IS, correlates better with human judgment, detects mode collapse</li>
      <li><strong>Standard metric:</strong> Most widely used for GAN evaluation</li>
    </ul>

    <h4>Precision and Recall</h4>
    <ul>
      <li><strong>Precision:</strong> Fraction of generated samples that are realistic (quality)</li>
      <li><strong>Recall:</strong> Fraction of real distribution covered by generator (diversity)</li>
      <li><strong>Trade-off:</strong> High precision but low recall = mode collapse, both high = ideal</li>
      <li><strong>Computation:</strong> Define manifolds in feature space, measure overlap</li>
    </ul>

    <h4>Human Evaluation</h4>
    <ul>
      <li><strong>Gold standard:</strong> Subjective quality assessment by humans</li>
      <li><strong>Methods:</strong> Pairwise comparisons, rating scales, discrimination tasks</li>
      <li><strong>Expensive but accurate:</strong> Ultimate measure of realism</li>
    </ul>

    <h3>Applications of GANs</h3>

    <h4>Image Generation and Synthesis</h4>
    <ul>
      <li><strong>Faces:</strong> StyleGAN generates photorealistic faces, "This Person Does Not Exist"</li>
      <li><strong>Objects:</strong> Generate furniture, animals, vehicles for design and simulation</li>
      <li><strong>Scenes:</strong> Indoor/outdoor scene generation for VR, gaming</li>
      <li><strong>Art:</strong> Generate paintings, artistic styles, creative content</li>
    </ul>

    <h4>Image-to-Image Translation</h4>
    <ul>
      <li><strong>Pix2Pix:</strong> Paired translation (edges→photos, labels→scenes)</li>
      <li><strong>CycleGAN:</strong> Unpaired translation (photos→paintings, domain adaptation)</li>
      <li><strong>Super-resolution:</strong> SRGAN enhances low-res images to high-res</li>
      <li><strong>Colorization:</strong> Black & white → color images</li>
      <li><strong>Inpainting:</strong> Fill missing image regions</li>
    </ul>

    <h4>Data Augmentation</h4>
    <ul>
      <li><strong>Generate training data:</strong> Augment small datasets with synthetic samples</li>
      <li><strong>Rare cases:</strong> Generate underrepresented classes to balance datasets</li>
      <li><strong>Simulation:</strong> Create realistic synthetic data for training (e.g., driving scenarios)</li>
    </ul>

    <h4>Text-to-Image Synthesis</h4>
    <ul>
      <li><strong>StackGAN, AttnGAN:</strong> Generate images from text descriptions</li>
      <li><strong>DALL-E:</strong> Text-to-image with Transformer-based architectures</li>
      <li><strong>Applications:</strong> Creative tools, content generation, accessibility</li>
    </ul>

    <h4>Medical Imaging</h4>
    <ul>
      <li><strong>Data augmentation:</strong> Generate synthetic medical images to augment scarce data</li>
      <li><strong>Modality translation:</strong> MRI ↔ CT, generate missing modalities</li>
      <li><strong>Anomaly detection:</strong> Train on normal images, detect abnormalities</li>
    </ul>

    <h4>Video Generation</h4>
    <ul>
      <li><strong>VideoGAN:</strong> Generate short video clips</li>
      <li><strong>Prediction:</strong> Future frame prediction from past frames</li>
      <li><strong>Deepfakes:</strong> Face swapping in videos (ethical concerns)</li>
    </ul>

    <h3>Theoretical Insights</h3>

    <h4>Optimal Discriminator</h4>
    <p>For fixed G, optimal discriminator is:</p>
    <p style="text-align: center;">$D^*(x) = \\frac{p_{\\text{data}}(x)}{p_{\\text{data}}(x) + p_g(x)}$</p>
    <p>At equilibrium where $p_g = p_{\\text{data}}$: $D^*(x) = 1/2$ everywhere.</p>

    <h4>Jensen-Shannon Divergence</h4>
    <p>At optimal D, training G minimizes JS divergence between p_data and p_g:</p>
    <p style="text-align: center;">JSD(p_data || p_g) = (1/2)[KL(p_data || m) + KL(p_g || m)]</p>
    <p>where m = (p_data + p_g)/2. Minimum JSD = 0 when p_data = p_g.</p>

    <h4>Nash Equilibrium</h4>
    <ul>
      <li><strong>Equilibrium:</strong> p_g = p_data and D(x) = 1/2 everywhere</li>
      <li><strong>Challenge:</strong> Alternating gradient descent doesn't guarantee convergence to Nash equilibrium</li>
      <li><strong>Non-convex game:</strong> Multiple local equilibria, no convergence guarantees in general</li>
    </ul>

    <h3>GANs vs Other Generative Models</h3>

    <table >
      <tr>
        <th>Aspect</th>
        <th>GANs</th>
        <th>VAEs</th>
        <th>Autoregressive</th>
        <th>Diffusion</th>
      </tr>
      <tr>
        <td>Sample Quality</td>
        <td>Sharp, realistic</td>
        <td>Slightly blurry</td>
        <td>Sharp</td>
        <td>Sharp, high-quality</td>
      </tr>
      <tr>
        <td>Training Stability</td>
        <td>Challenging</td>
        <td>Stable</td>
        <td>Stable</td>
        <td>Stable</td>
      </tr>
      <tr>
        <td>Diversity</td>
        <td>Mode collapse risk</td>
        <td>Good</td>
        <td>Excellent</td>
        <td>Excellent</td>
      </tr>
      <tr>
        <td>Speed (Sampling)</td>
        <td>Fast (single pass)</td>
        <td>Fast (single pass)</td>
        <td>Slow (sequential)</td>
        <td>Slow (many steps)</td>
      </tr>
      <tr>
        <td>Likelihood</td>
        <td>No explicit</td>
        <td>Approximate (ELBO)</td>
        <td>Exact</td>
        <td>Exact (up to approx)</td>
      </tr>
      <tr>
        <td>Control</td>
        <td>Moderate (cGAN)</td>
        <td>Good (latent space)</td>
        <td>Excellent</td>
        <td>Good</td>
      </tr>
    </table>

    <h3>The Evolution Beyond Classic GANs</h3>
    <p>While GANs revolutionized generative modeling and remain influential, recent years have seen diffusion models (DALL-E 2, Stable Diffusion, Midjourney) achieve superior results for text-to-image synthesis. These models combine GAN-like sample quality with VAE-like training stability. However, GANs remain important for specific applications requiring fast inference (single forward pass), image-to-image translation, and scenarios where adversarial training's implicit density modeling is advantageous. The core ideas of GANs—adversarial training, minimax games, discriminator as learned loss—continue to influence modern architectures, and GANs remain a fundamental technique in every machine learning researcher's toolkit.</p>

    <h3>Recent Developments (2023-2025)</h3>
    <ul>
      <li><strong>GigaGAN:</strong> Adobe's large-scale GAN achieving 4× faster generation than diffusion models while maintaining quality</li>
      <li><strong>StyleGAN3:</strong> Alias-free architecture producing smoother animations and better video generation</li>
      <li><strong>Projected GANs:</strong> Using frozen pre-trained features (CLIP, DINO) as discriminator improves training stability</li>
      <li><strong>GAN-based super-resolution:</strong> Real-ESRGAN and similar models dominate image upscaling tasks</li>
      <li><strong>Hybrid models:</strong> LDM-GAN combines latent diffusion with adversarial training for best of both worlds</li>
      <li><strong>3D GANs:</strong> EG3D and GET3D generate 3D-consistent objects from 2D images</li>
    </ul>

    <h3>Practical Tips for Training GANs</h3>
    <h4>Common Pitfalls and Solutions</h4>
    <ul>
      <li><strong>Pitfall:</strong> Generator collapse (outputs same image repeatedly)
        <br><strong>Solution:</strong> Use minibatch discrimination, increase diversity penalty, check discriminator isn't too strong</li>
      <li><strong>Pitfall:</strong> Training instability (loss oscillating wildly)
        <br><strong>Solution:</strong> Reduce learning rates, use gradient penalty (WGAN-GP), add spectral normalization</li>
      <li><strong>Pitfall:</strong> Discriminator overpowers generator early
        <br><strong>Solution:</strong> Use non-saturating loss, train D for fewer steps (k=1), add noise to D inputs</li>
      <li><strong>Pitfall:</strong> Poor sample quality
        <br><strong>Solution:</strong> Follow DCGAN guidelines strictly, increase model capacity, ensure data normalization is correct</li>
    </ul>

    <h4>Hyperparameter Tuning Guidelines</h4>
    <ul>
      <li><strong>Learning rate:</strong> Start with lr=0.0002 for both G and D, use Adam with $\\beta_1=0.5$, $\\beta_2=0.999$</li>
      <li><strong>Batch size:</strong> Larger is better (32-128), helps stabilize training</li>
      <li><strong>Architecture:</strong> Start with DCGAN, gradually add complexity (self-attention, progressive growing)</li>
      <li><strong>Training ratio:</strong> k=1 (one D update per G update) usually works best</li>
      <li><strong>Regularization:</strong> Spectral norm for D, gradient penalty for stability</li>
      <li><strong>Monitoring:</strong> Track both losses AND generated samples quality—don't rely on loss alone</li>
    </ul>
  `,
  codeExamples: [
    {
      language: 'Python',
      code: `import torch
import torch.nn as nn
import torch.optim as optim

# Generator network
class Generator(nn.Module):
  def __init__(self, latent_dim=100, img_channels=1, img_size=28):
      super().__init__()
      self.img_size = img_size

      # Input: latent vector [batch, latent_dim]
      # Output: image [batch, channels, height, width]
      self.model = nn.Sequential(
          # Fully connected to start
          nn.Linear(latent_dim, 256 * 7 * 7),
          nn.BatchNorm1d(256 * 7 * 7),
          nn.ReLU(inplace=True),
          nn.Unflatten(1, (256, 7, 7)),

          # Upsample to 14x14
          nn.ConvTranspose2d(256, 128, 4, 2, 1),
          nn.BatchNorm2d(128),
          nn.ReLU(inplace=True),

          # Upsample to 28x28
          nn.ConvTranspose2d(128, img_channels, 4, 2, 1),
          nn.Tanh()  # Output in [-1, 1]
      )

  def forward(self, z):
      return self.model(z)

# Discriminator network
class Discriminator(nn.Module):
  def __init__(self, img_channels=1, img_size=28):
      super().__init__()

      # Input: image [batch, channels, height, width]
      # Output: probability [batch, 1]
      self.model = nn.Sequential(
          # 28x28 -> 14x14
          nn.Conv2d(img_channels, 64, 4, 2, 1),
          nn.LeakyReLU(0.2, inplace=True),

          # 14x14 -> 7x7
          nn.Conv2d(64, 128, 4, 2, 1),
          nn.BatchNorm2d(128),
          nn.LeakyReLU(0.2, inplace=True),

          # Flatten and classify
          nn.Flatten(),
          nn.Linear(128 * 7 * 7, 1),
          nn.Sigmoid()
      )

  def forward(self, x):
      return self.model(x)

# Initialize networks
latent_dim = 100
generator = Generator(latent_dim)
discriminator = Discriminator()

# Optimizers
lr = 0.0002
betas = (0.5, 0.999)
opt_g = optim.Adam(generator.parameters(), lr=lr, betas=betas)
opt_d = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# Loss function
criterion = nn.BCELoss()

# Training step
def train_step(real_images, batch_size):
  # Labels
  real_labels = torch.ones(batch_size, 1)
  fake_labels = torch.zeros(batch_size, 1)

  # ===== Train Discriminator =====
  opt_d.zero_grad()

  # Real images
  real_output = discriminator(real_images)
  d_loss_real = criterion(real_output, real_labels)

  # Fake images
  z = torch.randn(batch_size, latent_dim)
  fake_images = generator(z)
  fake_output = discriminator(fake_images.detach())
  d_loss_fake = criterion(fake_output, fake_labels)

  # Total discriminator loss
  d_loss = d_loss_real + d_loss_fake
  d_loss.backward()
  opt_d.step()

  # ===== Train Generator =====
  opt_g.zero_grad()

  # Generate fake images and try to fool discriminator
  z = torch.randn(batch_size, latent_dim)
  fake_images = generator(z)
  fake_output = discriminator(fake_images)
  g_loss = criterion(fake_output, real_labels)  # Want D to output 1

  g_loss.backward()
  opt_g.step()

  return d_loss.item(), g_loss.item()

print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")`,
      explanation: 'Basic GAN implementation with generator and discriminator, showing the adversarial training loop.'
    },
    {
      language: 'Python',
      code: `import torch
import torch.nn as nn

# Conditional GAN - both networks receive class labels
class ConditionalGenerator(nn.Module):
  def __init__(self, latent_dim=100, num_classes=10, img_channels=1):
      super().__init__()

      # Embedding for class labels
      self.label_embedding = nn.Embedding(num_classes, num_classes)

      # Combine latent vector and label embedding
      input_dim = latent_dim + num_classes

      self.model = nn.Sequential(
          nn.Linear(input_dim, 256 * 7 * 7),
          nn.BatchNorm1d(256 * 7 * 7),
          nn.ReLU(inplace=True),
          nn.Unflatten(1, (256, 7, 7)),

          nn.ConvTranspose2d(256, 128, 4, 2, 1),
          nn.BatchNorm2d(128),
          nn.ReLU(inplace=True),

          nn.ConvTranspose2d(128, img_channels, 4, 2, 1),
          nn.Tanh()
      )

  def forward(self, z, labels):
      # Concatenate latent vector with label embedding
      label_emb = self.label_embedding(labels)
      gen_input = torch.cat([z, label_emb], dim=1)
      return self.model(gen_input)

class ConditionalDiscriminator(nn.Module):
  def __init__(self, num_classes=10, img_channels=1):
      super().__init__()

      # Embedding for class labels
      self.label_embedding = nn.Embedding(num_classes, 28 * 28)

      # Image channels + label channel
      self.conv = nn.Sequential(
          nn.Conv2d(img_channels + 1, 64, 4, 2, 1),
          nn.LeakyReLU(0.2, inplace=True),

          nn.Conv2d(64, 128, 4, 2, 1),
          nn.BatchNorm2d(128),
          nn.LeakyReLU(0.2, inplace=True),

          nn.Flatten(),
          nn.Linear(128 * 7 * 7, 1),
          nn.Sigmoid()
      )

  def forward(self, x, labels):
      # Create label map
      label_emb = self.label_embedding(labels)
      label_map = label_emb.view(-1, 1, 28, 28)

      # Concatenate image with label map
      d_input = torch.cat([x, label_map], dim=1)
      return self.conv(d_input)

# Usage example
latent_dim = 100
num_classes = 10

cond_gen = ConditionalGenerator(latent_dim, num_classes)
cond_disc = ConditionalDiscriminator(num_classes)

# Generate specific class (e.g., digit 7)
batch_size = 32
z = torch.randn(batch_size, latent_dim)
target_class = torch.full((batch_size,), 7, dtype=torch.long)

fake_images = cond_gen(z, target_class)
print(f"Generated images for class 7: {fake_images.shape}")

# Discriminator with conditioning
output = cond_disc(fake_images, target_class)
print(f"Discriminator output: {output.shape}")

# Generate different classes
for digit in range(10):
  labels = torch.full((4,), digit, dtype=torch.long)
  z = torch.randn(4, latent_dim)
  samples = cond_gen(z, labels)
  print(f"Digit {digit}: {samples.shape}")`,
      explanation: 'Conditional GAN implementation allowing controlled generation based on class labels.'
    }
  ],
  interviewQuestions: [
    {
      question: 'Explain the minimax game between generator and discriminator.',
      answer: `The minimax game is the core training paradigm where generator and discriminator have opposing objectives. The discriminator maximizes its ability to distinguish real from fake data, while the generator minimizes the discriminator's ability to detect fakes. Mathematically: min_G max_D V(D,G) = E[log D(x)] + E[log(1-D(G(z)))]. This adversarial process drives both networks to improve, ideally reaching Nash equilibrium where the generator produces perfect fakes and the discriminator cannot distinguish them.`
    },
    {
      question: 'What is mode collapse and how can it be addressed?',
      answer: `Mode collapse occurs when the generator produces limited diversity, mapping different noise inputs to similar outputs, failing to capture all modes of the data distribution. Solutions include: (1) Unrolled GANs - considering discriminator's future updates, (2) Minibatch discrimination - comparing samples within batches, (3) Feature matching - matching intermediate discriminator features rather than final output, (4) WGAN with weight clipping for stable training, (5) Spectral normalization for Lipschitz constraints, (6) Progressive growing for gradual complexity increase.`
    },
    {
      question: 'Why is GAN training unstable and what techniques improve stability?',
      answer: `GAN instability stems from: (1) Non-convex optimization landscape, (2) Moving targets during alternating training, (3) Vanishing gradients when discriminator becomes too strong, (4) Oscillating behavior rather than convergence. Stability improvements: (1) Balanced training ratios, (2) Learning rate scheduling, (3) Batch normalization, (4) Label smoothing, (5) Experience replay, (6) Gradient penalty (WGAN-GP), (7) Self-attention mechanisms (SAGAN), (8) Progressive growing, (9) Spectral normalization.`
    },
    {
      question: 'How does WGAN improve upon the original GAN?',
      answer: `WGAN uses Wasserstein distance instead of Jensen-Shannon divergence, providing: (1) Meaningful loss correlating with sample quality, (2) Improved training stability through Lipschitz constraint, (3) No mode collapse issues, (4) Continuous optimization landscape. Implementation: weight clipping to enforce Lipschitz constraint (WGAN) or gradient penalty (WGAN-GP). Benefits: stable training regardless of architecture, meaningful loss monitoring, reduced hyperparameter sensitivity, and theoretical guarantees for convergence.`
    },
    {
      question: 'What is the difference between GAN and VAE for generation?',
      answer: `GANs: Adversarial training produces sharp, realistic images but training can be unstable with mode collapse. No explicit likelihood model, making evaluation difficult. Excellent for high-quality synthesis but limited controllability. VAEs: Probabilistic approach with encoder-decoder architecture, stable training with clear objective (ELBO). Produces slightly blurry images due to reconstruction loss but offers better controllability and interpretable latent space. Easier to evaluate through likelihood estimation. Choose GANs for quality, VAEs for stability and control.`
    },
    {
      question: 'Explain how conditional GANs enable controlled generation.',
      answer: `Conditional GANs (cGANs) extend standard GANs by conditioning both generator and discriminator on additional information (labels, text, images). Generator: G(z, y) where y is conditioning information. Discriminator: D(x, y) evaluates real/fake given condition. This enables controlled generation: class-conditional images, text-to-image synthesis, style transfer. Training requires paired examples or semi-supervised techniques. Applications include Pix2Pix, CycleGAN, and text-to-image models like DALL-E.`
    }
  ],
  quizQuestions: [
    {
      id: 'gan1',
      question: 'What is the role of the generator in a GAN?',
      options: ['Classify real vs fake', 'Generate fake data from noise', 'Extract features', 'Compress data'],
      correctAnswer: 1,
      explanation: 'The generator takes random noise as input and generates fake data, learning to produce increasingly realistic samples to fool the discriminator.'
    },
    {
      id: 'gan2',
      question: 'What is mode collapse in GANs?',
      options: ['Training fails completely', 'Generator produces limited variety', 'Discriminator always wins', 'Model overfits'],
      correctAnswer: 1,
      explanation: 'Mode collapse occurs when the generator learns to produce only a limited variety of samples, failing to capture the full diversity of the data distribution.'
    },
    {
      id: 'gan3',
      question: 'How are generator and discriminator typically trained?',
      options: ['Simultaneously', 'Generator first', 'Discriminator first', 'Alternating updates'],
      correctAnswer: 3,
      explanation: 'GANs use alternating training: update discriminator to better distinguish real/fake, then update generator to better fool the discriminator, repeating until equilibrium.'
    }
  ]
};
