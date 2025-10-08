import { Topic } from '../../../types';

export const variationalAutoencoders: Topic = {
  id: 'variational-autoencoders',
  title: 'Variational Autoencoders (VAEs)',
  category: 'advanced',
  description: 'Probabilistic generative models with latent variable representations',
  content: `
    <h2>Variational Autoencoders: Probabilistic Generative Modeling</h2>
    <p>Variational Autoencoders (VAEs), introduced by Kingma and Welling in 2013, combine deep learning with variational Bayesian methods to create powerful generative models. Unlike standard autoencoders that learn deterministic compressions, VAEs learn probability distributions over latent representations, enabling principled generation of new samples. By framing generation as probabilistic inference and optimizing a tractable lower bound on the data likelihood, VAEs provide a stable training paradigm with theoretical guarantees. While VAE-generated images tend to be slightly blurrier than GAN outputs, VAEs offer superior training stability, interpretable latent spaces, and the ability to compute likelihoods—making them invaluable for applications requiring controllable generation, anomaly detection, and representation learning.</p>

    <h3>The Generative Modeling Framework</h3>

    <h4>Latent Variable Models</h4>
    <p><strong>Goal:</strong> Model complex data distribution p(x) through simpler latent variables z.</p>

    <h5>Generative Process</h5>
    <ol>
      <li><strong>Sample latent code:</strong> z ~ p(z) from prior distribution (typically Gaussian)</li>
      <li><strong>Generate observation:</strong> x ~ p(x|z) from conditional distribution given z</li>
      <li><strong>Marginal distribution:</strong> p(x) = ∫ p(x|z)p(z) dz (intractable integral)</li>
    </ol>

    <h5>The Inference Challenge</h5>
    <ul>
      <li><strong>Posterior p(z|x):</strong> Given observation x, what latent z generated it?</li>
      <li><strong>Bayes' rule:</strong> p(z|x) = p(x|z)p(z) / p(x)</li>
      <li><strong>Problem:</strong> Computing p(x) = ∫ p(x|z)p(z) dz is intractable for complex models</li>
      <li><strong>VAE solution:</strong> Approximate posterior with learned recognition network q_φ(z|x)</li>
    </ul>

    <h3>VAE Architecture Components</h3>

    <h4>Visual Architecture Overview</h4>
    <pre class="code-block">
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│                        Variational Autoencoder                         │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Input x (e.g., 28×28 image)                                           │
│         │                                                              │
│         ▼                                                              │
│  ┌──────────────────┐                                                  │
│  │   ENCODER qφ(z|x) │                                                 │
│  │  (Neural Network) │                                                 │
│  └──────────────────┘                                                  │
│         │                                                              │
│         ├──────────────────────────────────────┐                       │
│         │                                      │                       │
│         ▼                                      ▼                       │
│      μ(x)                                   log σ²(x)                  │
│         │                                      │                       │
│         └───────────┐            ┌─────────────┘                       │
│                     │            │                                     │
│                     ▼            ▼                                     │
│              z = μ + σ ⊙ ε   (ε ~ N(0,I))                              │
│                Reparameterization Trick                                │
│                         │                                              │
│                         ▼                                              │
│              Latent Code z (e.g., 20-dim)                              │
│                         │                                              │
│                         ▼                                              │
│                  ┌──────────────────┐                                  │
│                  │  DECODER pθ(x|z) │                                  │
│                  │ (Neural Network) │                                  │
│                  └──────────────────┘                                  │
│                         │                                              │
│                         ▼                                              │
│                Reconstruction x̂ (28×28)                                │
│                                                                        │
│  Loss = Reconstruction Loss + KL Divergence                            │
│       = -E[log p(x|z)]      + KL(q(z|x) || p(z))                       │
└────────────────────────────────────────────────────────────────────────┘
    </pre>

    <h4>Encoder: q_φ(z|x) - Recognition Network</h4>

    <h5>Role and Design</h5>
    <ul>
      <li><strong>Input:</strong> Data point x (e.g., 28×28 image)</li>
      <li><strong>Output:</strong> Parameters of approximate posterior distribution over z</li>
      <li><strong>Typical assumption:</strong> $q_\\phi(z|x) = N(z; \\mu_\\phi(x), \\sigma^2_\\phi(x))$ (diagonal Gaussian)</li>
      <li><strong>Neural network:</strong> Maps $x \\to (\\mu, \\log \\sigma^2)$ where $\\mu$, $\\sigma^2$ are latent_dim-dimensional vectors</li>
      <li><strong>Variational approximation:</strong> $q_\\phi(z|x) \\approx$ true posterior $p(z|x)$</li>
    </ul>

    <h5>Architecture Example (Images)</h5>
    <pre>
Input image x (28×28)
  → Flatten or Conv layers
  → Hidden layers with ReLU
  → Split into two heads:
    - μ branch: FC → latent_dim (unbounded)
    - log σ² branch: FC → latent_dim (unbounded)
  → Output: (μ, log σ²) defining q(z|x) = N(μ, σ²)
    </pre>

    <h4>Latent Space: z ~ q_φ(z|x)</h4>

    <h5>Sampling with Reparameterization Trick</h5>
    <ul>
      <li><strong>Challenge:</strong> Sampling $z \\sim N(\\mu, \\sigma^2)$ is non-differentiable—can't backpropagate through random operation</li>
      <li><strong>Solution:</strong> Reparameterize: $z = \\mu + \\sigma \\odot \\varepsilon$ where $\\varepsilon \\sim N(0, I)$</li>
      <li><strong>Key insight:</strong> Move randomness to external noise $\\varepsilon$ independent of parameters $\\phi$</li>
      <li><strong>Gradients:</strong> $\\frac{\\partial z}{\\partial \\mu} = 1$, $\\frac{\\partial z}{\\partial \\sigma} = \\varepsilon$ (well-defined)</li>
      <li><strong>Enables training:</strong> Backpropagation through stochastic layer</li>
    </ul>

    <h5>Latent Space Properties</h5>
    <ul>
      <li><strong>Dimensionality:</strong> Typically 10-512 dimensions (much smaller than data)</li>
      <li><strong>Prior p(z):</strong> Standard Gaussian N(0, I) (simple, spherical)</li>
      <li><strong>Continuity:</strong> Similar z values decode to similar x (smooth interpolation)</li>
      <li><strong>Disentanglement:</strong> Ideally, each dimension captures independent factor of variation</li>
    </ul>

    <h4>Decoder: $p_\\theta(x|z)$ - Generative Network</h4>

    <h5>Role and Design</h5>
    <ul>
      <li><strong>Input:</strong> Latent code z (e.g., 20-dimensional vector)</li>
      <li><strong>Output:</strong> Reconstruction parameters for data distribution p_θ(x|z)</li>
      <li><strong>For images:</strong> Often outputs mean of Gaussian or Bernoulli probabilities</li>
      <li><strong>Architecture:</strong> Mirror of encoder (upsampling, transposed convolutions)</li>
      <li><strong>Stochastic or deterministic:</strong> Can output distribution parameters or direct reconstruction</li>
    </ul>

    <h5>Architecture Example (Images)</h5>
    <pre>
Latent z (20-dim)
  → FC → Hidden units
  → Hidden layers with ReLU
  → Transposed Conv or Upsampling (for images)
  → Output layer: Sigmoid (Bernoulli) or Identity (Gaussian)
  → Reconstruction x̂ (28×28)
    </pre>

    <h3>Training Objective: The ELBO</h3>

    <h4>Evidence Lower Bound (ELBO)</h4>

    <h5>Derivation from Marginal Likelihood</h5>
    <p><strong>Goal:</strong> Maximize log p(x) = log ∫ p(x|z)p(z) dz (intractable)</p>

    <h6>Variational Lower Bound</h6>
    <p>For any distribution q(z|x):</p>
    <p style="text-align: center;">
      $\\log p(x) = \\text{ELBO} + \\text{KL}(q(z|x) || p(z|x))$
    </p>
    <p style="text-align: center;">
      $\\log p(x) \\geq \\text{ELBO}$ &nbsp;&nbsp;(since $\\text{KL} \\geq 0$)
    </p>
    <p style="text-align: center;">
      $\\text{ELBO} = \\mathbb{E}_{q(z|x)}[\\log p(x|z)] - \\text{KL}(q(z|x) || p(z))$
    </p>

    <h6>Interpretation</h6>
    <ul>
      <li><strong>First term:</strong> Reconstruction likelihood—how well decoder reconstructs from z sampled from encoder</li>
      <li><strong>Second term:</strong> KL regularization—how close approximate posterior is to prior</li>
      <li><strong>Maximizing ELBO:</strong> Tightens lower bound on log p(x), improving generative model</li>
      <li><strong>Perfect approximation:</strong> When q(z|x) = p(z|x), ELBO = log p(x)</li>
    </ul>

    <h4>Loss Function Components</h4>

    <h5>1. Reconstruction Loss</h5>
    <p><strong>Formula:</strong> $-\\mathbb{E}_{q(z|x)}[\\log p_\\theta(x|z)]$</p>

    <h6>For Different Data Types</h6>
    <ul>
      <li><strong>Binary images:</strong> Binary cross-entropy (BCE) per pixel</li>
      <li><strong>Continuous images:</strong> Mean squared error (MSE) assuming Gaussian p(x|z)</li>
      <li><strong>Monte Carlo estimate:</strong> Sample z ~ q(z|x), compute -log p(x|z)</li>
      <li><strong>Practical:</strong> Often use single sample for efficiency</li>
    </ul>

    <h5>2. KL Divergence Regularization</h5>
    <p><strong>Formula:</strong> KL(q_φ(z|x) || p(z))</p>

    <h6>For Gaussian Distributions</h6>
    <p>When $q(z|x) = N(\\mu, \\sigma^2 I)$ and $p(z) = N(0, I)$:</p>
    <p style="text-align: center;">
      $\\text{KL} = \\frac{1}{2} \\sum_{j=1}^J [\\mu_j^2 + \\sigma_j^2 - \\log \\sigma_j^2 - 1]$
    </p>
    <ul>
      <li><strong>Closed form:</strong> No sampling needed, exact computation</li>
      <li><strong>Interpretation:</strong> Penalizes deviation from prior N(0, I)</li>
      <li><strong>Effect:</strong> Prevents encoder from ignoring latent code (posterior collapse)</li>
    </ul>

    <h4>Complete VAE Objective</h4>
    <p><strong>Minimize:</strong></p>
    <p style="text-align: center; font-size: 1.1em;">
      L_VAE = -ELBO = Reconstruction Loss + KL Divergence
    </p>
    <p style="text-align: center;">
      $L_{\\text{VAE}} = -\\mathbb{E}_{q(z|x)}[\\log p(x|z)] + \\text{KL}(q(z|x) || p(z))$
    </p>

    <h3>The Reparameterization Trick: Making Sampling Differentiable</h3>

    <h4>The Problem</h4>
    <ul>
      <li><strong>Need to backpropagate through z ~ q(z|x):</strong> Gradient $\\nabla_\\phi \\mathbb{E}_{z\\sim q_\\phi(z|x)}[f(z)]$</li>
      <li><strong>Direct sampling non-differentiable:</strong> Can't compute $\\frac{\\partial z}{\\partial \\phi}$ when z is stochastic</li>
      <li><strong>Naive approach fails:</strong> Taking expectation outside gradient gives high-variance estimates</li>
    </ul>

    <h4>The Solution: Reparameterization</h4>

    <h5>Transform Sampling</h5>
    <p><strong>Instead of:</strong> $z \\sim N(\\mu_\\phi(x), \\sigma^2_\\phi(x))$</p>
    <p><strong>Write as:</strong> $z = \\mu_\\phi(x) + \\sigma_\\phi(x) \\odot \\varepsilon$, where $\\varepsilon \\sim N(0, I)$</p>

    <h5>Benefits</h5>
    <ul>
      <li><strong>Deterministic function:</strong> z is deterministic given x and $\\varepsilon$</li>
      <li><strong>External randomness:</strong> $\\varepsilon$ is independent of parameters $\\phi$, $\\theta$</li>
      <li><strong>Differentiable:</strong> Clear gradients: $\\frac{\\partial z}{\\partial \\mu} = 1$, $\\frac{\\partial z}{\\partial \\sigma} = \\varepsilon$</li>
      <li><strong>Low variance:</strong> Gradient estimator has much lower variance than alternatives</li>
    </ul>

    <h5>Gradient Flow</h5>
    <p>
      $\\nabla_\\phi \\mathbb{E}_{\\varepsilon\\sim N(0,I)}[f(\\mu_\\phi(x) + \\sigma_\\phi(x)\\odot\\varepsilon)]$
    </p>
    <p>
      $= \\mathbb{E}_{\\varepsilon\\sim N(0,I)}[\\nabla_\\phi f(\\mu_\\phi(x) + \\sigma_\\phi(x)\\odot\\varepsilon)]$ &nbsp;&nbsp;(exchange gradient and expectation)
    </p>
    <p>
      $\\approx \\nabla_\\phi f(\\mu_\\phi(x) + \\sigma_\\phi(x)\\odot\\varepsilon)$ &nbsp;&nbsp;(single sample Monte Carlo)
    </p>

    <h3>Training Procedure</h3>

    <h4>VAE Training Algorithm</h4>
    <pre>
for epoch in range(num_epochs):
  for batch x in dataloader:
      # ===== Forward Pass =====
      # Encode
      μ, log σ² = Encoder(x)
      
      # Reparameterize
      ε ~ N(0, I)
      z = μ + exp(0.5 × log σ²) ⊙ ε
      
      # Decode
      x̂ = Decoder(z)
      
      # ===== Compute Loss =====
      # Reconstruction loss (e.g., BCE for binary data)
      recon_loss = BCE(x̂, x)
      
      # KL divergence (closed form for Gaussian)
      kl_loss = -0.5 × sum(1 + log σ² - μ² - σ²)
      
      # Total loss
      loss = recon_loss + kl_loss
      
      # ===== Backward Pass =====
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
    </pre>

    <h4>Balancing Reconstruction and KL</h4>
    <ul>
      <li><strong>Trade-off:</strong> Reconstruction wants to use latent code fully; KL wants to match prior</li>
      <li><strong>β-VAE weighting:</strong> loss = recon_loss + $\\beta \\times$ kl_loss ($\\beta > 1$ for disentanglement)</li>
      <li><strong>KL annealing:</strong> Gradually increase KL weight from 0 to 1 during training</li>
      <li><strong>Free bits:</strong> Don't penalize KL below threshold per dimension</li>
    </ul>

    <h3>VAE vs Standard Autoencoder</h3>

    <table >
      <tr>
        <th>Aspect</th>
        <th>Standard Autoencoder</th>
        <th>Variational Autoencoder</th>
      </tr>
      <tr>
        <td>Objective</td>
        <td>Minimize reconstruction error</td>
        <td>Maximize ELBO (recon + KL)</td>
      </tr>
      <tr>
        <td>Latent Space</td>
        <td>Deterministic point</td>
        <td>Probability distribution</td>
      </tr>
      <tr>
        <td>Encoder Output</td>
        <td>Latent code z</td>
        <td>Distribution parameters (μ, σ²)</td>
      </tr>
      <tr>
        <td>Regularization</td>
        <td>None (or explicit penalty)</td>
        <td>KL divergence to prior</td>
      </tr>
      <tr>
        <td>Generation</td>
        <td>Cannot generate (latent space unstructured)</td>
        <td>Sample from prior p(z)</td>
      </tr>
      <tr>
        <td>Interpolation</td>
        <td>Gaps/"holes" in latent space</td>
        <td>Smooth, meaningful interpolation</td>
      </tr>
      <tr>
        <td>Purpose</td>
        <td>Compression, denoising, feature learning</td>
        <td>Generative modeling, sampling</td>
      </tr>
    </table>

    <h3>Applications of VAEs</h3>

    <h4>1. Data Generation</h4>
    <ul>
      <li><strong>Image synthesis:</strong> Sample z ~ N(0,I), decode to generate new images</li>
      <li><strong>Conditional generation:</strong> Condition on labels for class-specific generation</li>
      <li><strong>Face generation:</strong> Generate diverse faces from latent space</li>
      <li><strong>Molecular design:</strong> Generate novel drug molecules (encode SMILES strings)</li>
    </ul>

    <h4>2. Latent Space Interpolation</h4>
    <ul>
      <li><strong>Smooth transitions:</strong> Interpolate z between two encoded images, decode intermediate points</li>
      <li><strong>Morphing:</strong> Face A → Face B by traversing latent space</li>
      <li><strong>Disentangled factors:</strong> Modify specific dimensions to change attributes (smile, age, pose)</li>
    </ul>

    <h4>3. Anomaly Detection</h4>
    <ul>
      <li><strong>Reconstruction error:</strong> Normal data reconstructs well; anomalies have high error</li>
      <li><strong>Likelihood:</strong> Compute p(x) via ELBO—low likelihood indicates anomaly</li>
      <li><strong>Applications:</strong> Fraud detection, medical imaging (detect tumors), manufacturing defects</li>
    </ul>

    <h4>4. Representation Learning</h4>
    <ul>
      <li><strong>Latent features:</strong> Use z as features for downstream tasks (classification, clustering)</li>
      <li><strong>Semi-supervised learning:</strong> Train VAE on unlabeled data, fine-tune encoder for classification</li>
      <li><strong>Transfer learning:</strong> Pre-trained VAE encoder as feature extractor</li>
    </ul>

    <h4>5. Data Imputation and Denoising</h4>
    <ul>
      <li><strong>Missing data:</strong> Encode partial observation, decode to impute missing values</li>
      <li><strong>Denoising:</strong> Encode noisy data, decode to clean reconstruction</li>
      <li><strong>Inpainting:</strong> Fill missing image regions</li>
    </ul>

    <h3>VAE Variants and Extensions</h3>

    <h4>β-VAE: Disentangled Representations</h4>
    <ul>
      <li><strong>Objective:</strong> $L = \\text{recon\\_loss} + \\beta \\times \\text{KL\\_loss}$ where $\\beta > 1$</li>
      <li><strong>Effect:</strong> Higher β encourages independence between latent dimensions</li>
      <li><strong>Disentanglement:</strong> Each z_j captures separate factor (color, shape, position)</li>
      <li><strong>Trade-off:</strong> Improved disentanglement but reduced reconstruction quality</li>
      <li><strong>Applications:</strong> Interpretable generation, controllable synthesis</li>
    </ul>

    <h4>Conditional VAE (CVAE)</h4>
    <ul>
      <li><strong>Architecture:</strong> Encoder q(z|x,y), Decoder p(x|z,y) conditioned on label y</li>
      <li><strong>Generation:</strong> Specify desired class/attribute, sample z ~ N(0,I), decode with condition</li>
      <li><strong>Applications:</strong> Class-conditional image generation, controlled synthesis</li>
    </ul>

    <h4>VQ-VAE (Vector Quantized VAE)</h4>
    <ul>
      <li><strong>Discrete latent space:</strong> Replace continuous z with discrete codebook vectors</li>
      <li><strong>Encoder output:</strong> Index into learned codebook instead of μ, σ²</li>
      <li><strong>Benefits:</strong> No posterior collapse, better for autoregressive models</li>
      <li><strong>Applications:</strong> High-quality image generation, audio synthesis</li>
    </ul>

    <h4>Hierarchical VAE</h4>
    <ul>
      <li><strong>Multiple latent layers:</strong> z1, z2, ..., zL at different abstraction levels</li>
      <li><strong>Top-down generation:</strong> Sample from coarse to fine-grained details</li>
      <li><strong>Better modeling:</strong> Captures hierarchical structure of data</li>
      <li><strong>Applications:</strong> Complex images, hierarchical concepts</li>
    </ul>

    <h4>Importance Weighted Autoencoders (IWAE)</h4>
    <ul>
      <li><strong>Tighter bound:</strong> Use multiple samples to estimate ELBO more accurately</li>
      <li><strong>Improved likelihood:</strong> Better approximation of log p(x)</li>
      <li><strong>Cost:</strong> Increased computation (k samples instead of 1)</li>
    </ul>

    <h3>VAE vs GAN: Complementary Approaches</h3>

    <table >
      <tr>
        <th>Aspect</th>
        <th>VAE</th>
        <th>GAN</th>
      </tr>
      <tr>
        <td>Training</td>
        <td>Stable (single objective)</td>
        <td>Unstable (adversarial game)</td>
      </tr>
      <tr>
        <td>Sample Quality</td>
        <td>Slightly blurry</td>
        <td>Sharp, realistic</td>
      </tr>
      <tr>
        <td>Diversity</td>
        <td>Good coverage</td>
        <td>Mode collapse risk</td>
      </tr>
      <tr>
        <td>Likelihood</td>
        <td>Can estimate via ELBO</td>
        <td>No explicit likelihood</td>
      </tr>
      <tr>
        <td>Latent Space</td>
        <td>Structured, interpretable</td>
        <td>Less structured</td>
      </tr>
      <tr>
        <td>Control</td>
        <td>Excellent (encoder maps data to latent)</td>
        <td>Limited (no encoder)</td>
      </tr>
      <tr>
        <td>Evaluation</td>
        <td>ELBO, likelihood</td>
        <td>FID, IS, human judgment</td>
      </tr>
    </table>

    <h3>Why VAEs Produce Blurry Images</h3>
    
    <h4>Intuitive Explanation</h4>
    <p><strong>The averaging problem:</strong> Imagine trying to draw the "average" face from 100 different people. You'd blend all their features together, resulting in a blurry, generic face with no sharp details. VAEs face the same challenge—when the model is uncertain about which specific details to generate, it averages across possibilities, producing blur.</p>
    
    <p><strong>Example:</strong> If a VAE sees both dogs with pointy ears and dogs with floppy ears during training, when generating a new dog, it might produce ears that are somewhat in-between, creating a blurry compromise rather than committing to one crisp option.</p>

    <h4>Technical Explanation</h4>
    <ul>
      <li><strong>Reconstruction loss:</strong> MSE or BCE penalizes pixel-wise errors, encourages averaging</li>
      <li><strong>Gaussian assumption:</strong> $p(x|z) = N(\\text{decoder}(z), \\sigma^2 I)$ cannot capture sharp edges (multimodal pixel distributions)</li>
      <li><strong>KL regularization:</strong> Forces latent space to match simple prior, reducing capacity</li>
      <li><strong>Solution:</strong> More powerful decoders, perceptual losses, or hybrid VAE-GAN models</li>
    </ul>

    <h3>Posterior Collapse Problem</h3>

    <h4>The Issue</h4>
    <ul>
      <li><strong>Symptom:</strong> Encoder outputs z ≈ prior N(0,I) regardless of input x</li>
      <li><strong>Effect:</strong> Decoder ignores latent code, generates average samples</li>
      <li><strong>KL → 0:</strong> Posterior matches prior, but latent code carries no information</li>
    </ul>

    <h4>Causes and Solutions</h4>
    <ul>
      <li><strong>Powerful decoder:</strong> If decoder can model p(x) without z, encoder becomes redundant. Solution: Limit decoder capacity</li>
      <li><strong>KL annealing:</strong> Start with β=0, gradually increase to 1 to let encoder learn before regularization</li>
      <li><strong>Free bits:</strong> Only penalize KL above threshold: max(KL - λ, 0)</li>
      <li><strong>Skip connections:</strong> Allow decoder to see original x (forces encoder to provide useful z)</li>
    </ul>

    <h3>The Role of VAEs in Modern AI</h3>
    <p>While diffusion models have recently achieved superior image generation quality, VAEs remain fundamental for their stability, interpretability, and theoretical grounding. VAEs excel in applications requiring explicit density estimation, controllable generation via latent space manipulation, and representation learning. The VAE framework underpins modern architectures like VQ-VAE-2 (high-quality image generation), ProteinVAE (protein design), and hierarchical VAEs for video modeling. As hybrid models combining VAE stability with GAN-like sharpness emerge, VAEs continue to influence the evolution of generative AI.</p>

    <h3>Practical Tips for Training VAEs</h3>

    <h4>Common Issues and Solutions</h4>
    <ul>
      <li><strong>Issue:</strong> Posterior collapse (KL → 0, model ignores latent code)
        <br><strong>Debug:</strong> Monitor KL divergence per dimension, check if decoder is too powerful
        <br><strong>Solution:</strong> KL annealing (start β=0, increase gradually), free bits (min KL threshold), cyclical annealing</li>
      
      <li><strong>Issue:</strong> Poor reconstruction quality
        <br><strong>Debug:</strong> Visualize reconstructions, check if KL dominates loss
        <br><strong>Solution:</strong> Increase model capacity, reduce β weight on KL, use perceptual loss</li>
      
      <li><strong>Issue:</strong> Latent space not smooth (interpolations look bad)
        <br><strong>Debug:</strong> Interpolate between pairs, check KL per dimension
        <br><strong>Solution:</strong> Increase KL weight slightly, ensure proper normalization, use β-VAE</li>
      
      <li><strong>Issue:</strong> Training instability
        <br><strong>Debug:</strong> Plot both losses over time
        <br><strong>Solution:</strong> Reduce learning rate, use gradient clipping, warmup KL weight</li>
    </ul>

    <h4>Hyperparameter Guidelines</h4>
    <ul>
      <li><strong>Latent dimensions:</strong> Start with 64-128 for images, 8-32 for simple data. Too large risks posterior collapse.</li>
      <li><strong>β weight:</strong> Start with 1.0 (standard VAE), increase to 2-10 for disentanglement (β-VAE), decrease to 0.1-0.5 for better reconstruction</li>
      <li><strong>Learning rate:</strong> 1e-3 to 1e-4 with Adam optimizer works well</li>
      <li><strong>Batch size:</strong> 64-256, larger helps stabilize gradients</li>
      <li><strong>Architecture:</strong> Mirror encoder/decoder architectures, use BatchNorm or LayerNorm</li>
    </ul>

    <h4>Monitoring Training</h4>
    <ul>
      <li><strong>Track separately:</strong> Reconstruction loss, KL divergence (total and per dimension)</li>
      <li><strong>Visualize:</strong> Reconstructions every few epochs, random samples from prior</li>
      <li><strong>Latent space:</strong> Plot 2D projections (PCA/t-SNE), check for clustering</li>
      <li><strong>Red flags:</strong> KL → 0 (posterior collapse), reconstruction loss plateau while KL increases (overfitting latent space)</li>
    </ul>
  `,
  codeExamples: [
    {
      language: 'Python',
      code: `import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
  def __init__(self, input_dim=784, latent_dim=20):
      super().__init__()

      # Encoder
      self.encoder = nn.Sequential(
          nn.Linear(input_dim, 512),
          nn.ReLU(),
          nn.Linear(512, 256),
          nn.ReLU()
      )

      # Latent space parameters
      self.fc_mu = nn.Linear(256, latent_dim)
      self.fc_logvar = nn.Linear(256, latent_dim)

      # Decoder
      self.decoder = nn.Sequential(
          nn.Linear(latent_dim, 256),
          nn.ReLU(),
          nn.Linear(256, 512),
          nn.ReLU(),
          nn.Linear(512, input_dim),
          nn.Sigmoid()  # For [0, 1] output
      )

  def encode(self, x):
      h = self.encoder(x)
      mu = self.fc_mu(h)
      logvar = self.fc_logvar(h)
      return mu, logvar

  def reparameterize(self, mu, logvar):
      # z = μ + σ ⊙ ε
      std = torch.exp(0.5 * logvar)
      eps = torch.randn_like(std)
      return mu + eps * std

  def decode(self, z):
      return self.decoder(z)

  def forward(self, x):
      mu, logvar = self.encode(x)
      z = self.reparameterize(mu, logvar)
      recon_x = self.decode(z)
      return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
  # Reconstruction loss (binary cross-entropy)
  BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

  # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

  return BCE + KLD, BCE, KLD

# Usage
vae = VAE(input_dim=784, latent_dim=20)

# Training step
x = torch.randn(128, 784)  # Batch of flattened images
recon_x, mu, logvar = vae(x)

loss, bce, kld = vae_loss(recon_x, x, mu, logvar)
print(f"Total Loss: {loss.item():.2f}")
print(f"Reconstruction: {bce.item():.2f}")
print(f"KL Divergence: {kld.item():.2f}")

# Generate new samples
with torch.no_grad():
  z = torch.randn(10, 20)  # Sample from prior N(0, I)
  samples = vae.decode(z)
  print(f"Generated samples: {samples.shape}")

# Interpolate between two images
with torch.no_grad():
  x1, x2 = torch.randn(1, 784), torch.randn(1, 784)
  mu1, _ = vae.encode(x1)
  mu2, _ = vae.encode(x2)

  # Linear interpolation in latent space
  alphas = torch.linspace(0, 1, 10)
  for alpha in alphas:
      z_interp = alpha * mu1 + (1 - alpha) * mu2
      x_interp = vae.decode(z_interp)
      print(f"α={alpha:.1f}: {x_interp.shape}")`,
      explanation: 'Complete VAE implementation with encoder, decoder, reparameterization trick, and loss function.'
    },
    {
      language: 'Python',
      code: `import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolutional VAE for images
class ConvVAE(nn.Module):
  def __init__(self, latent_dim=128, img_channels=1):
      super().__init__()

      # Encoder
      self.encoder = nn.Sequential(
          nn.Conv2d(img_channels, 32, 4, 2, 1),  # 28x28 -> 14x14
          nn.ReLU(),
          nn.Conv2d(32, 64, 4, 2, 1),  # 14x14 -> 7x7
          nn.ReLU(),
          nn.Conv2d(64, 128, 3, 2, 1),  # 7x7 -> 4x4
          nn.ReLU(),
          nn.Flatten()
      )

      # Latent space
      self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
      self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

      # Decoder
      self.decoder_input = nn.Linear(latent_dim, 128 * 4 * 4)

      self.decoder = nn.Sequential(
          nn.Unflatten(1, (128, 4, 4)),
          nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=0),  # 4x4 -> 7x7
          nn.ReLU(),
          nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 7x7 -> 14x14
          nn.ReLU(),
          nn.ConvTranspose2d(32, img_channels, 4, 2, 1),  # 14x14 -> 28x28
          nn.Sigmoid()
      )

  def encode(self, x):
      h = self.encoder(x)
      mu = self.fc_mu(h)
      logvar = self.fc_logvar(h)
      return mu, logvar

  def reparameterize(self, mu, logvar):
      std = torch.exp(0.5 * logvar)
      eps = torch.randn_like(std)
      return mu + eps * std

  def decode(self, z):
      h = self.decoder_input(z)
      return self.decoder(h)

  def forward(self, x):
      mu, logvar = self.encode(x)
      z = self.reparameterize(mu, logvar)
      recon_x = self.decode(z)
      return recon_x, mu, logvar

# Beta-VAE: weighted KL divergence for disentanglement
class BetaVAE(ConvVAE):
  def __init__(self, latent_dim=128, img_channels=1, beta=4.0):
      super().__init__(latent_dim, img_channels)
      self.beta = beta

def beta_vae_loss(recon_x, x, mu, logvar, beta=4.0):
  # Reconstruction loss
  BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

  # KL divergence with beta weighting
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

  return BCE + beta * KLD, BCE, KLD

# Usage
conv_vae = ConvVAE(latent_dim=128, img_channels=1)
beta_vae = BetaVAE(latent_dim=128, img_channels=1, beta=4.0)

# Training
x = torch.randn(32, 1, 28, 28)  # Batch of images

# Standard VAE
recon_x, mu, logvar = conv_vae(x)
loss, bce, kld = vae_loss(recon_x, x, mu, logvar)
print(f"VAE - Total: {loss.item():.2f}, BCE: {bce.item():.2f}, KLD: {kld.item():.2f}")

# Beta-VAE (encourages disentanglement)
recon_x, mu, logvar = beta_vae(x)
loss, bce, kld = beta_vae_loss(recon_x, x, mu, logvar, beta=4.0)
print(f"β-VAE - Total: {loss.item():.2f}, BCE: {bce.item():.2f}, KLD: {kld.item():.2f}")

# Latent space traversal (for interpretability)
with torch.no_grad():
  z = torch.zeros(1, 128)
  for dim in range(10):  # Traverse first 10 dimensions
      for val in torch.linspace(-3, 3, 7):
          z_modified = z.clone()
          z_modified[0, dim] = val
          sample = beta_vae.decode(z_modified)
          print(f"Dim {dim}, val {val:.1f}: {sample.shape}")`,
      explanation: 'Convolutional VAE and β-VAE variant for images, with latent space traversal for disentanglement.'
    }
  ],
  interviewQuestions: [
    {
      question: 'Explain the reparameterization trick and why it\'s necessary.',
      answer: `The reparameterization trick enables backpropagation through stochastic nodes by expressing random variables as deterministic functions of noise. Instead of sampling $z \\sim N(\\mu, \\sigma^2)$, we compute $z = \\mu + \\sigma \\odot \\varepsilon$ where $\\varepsilon \\sim N(0,I)$. This transforms stochastic operation into deterministic computation with external randomness, allowing gradients to flow through $\\mu$ and $\\sigma$ parameters. Essential for training VAEs because it makes the latent variable sampling differentiable while maintaining the desired probability distribution.`
    },
    {
      question: 'What is the role of KL divergence in the VAE loss?',
      answer: `KL divergence KL(qφ(z|x)||p(z)) regularizes the encoder by penalizing deviation from the prior p(z), typically N(0,I). This prevents overfitting and ensures: (1) Latent space structure suitable for generation, (2) Smooth interpolation between points, (3) Preventing "holes" in latent space, (4) Enabling sampling from prior for generation. Without KL regularization, encoder could map data to arbitrary latent representations that decoder couldn't handle during generation.`
    },
    {
      question: 'How do VAEs differ from standard autoencoders?',
      answer: `Standard autoencoders learn deterministic mappings for dimensionality reduction or denoising, with no probabilistic interpretation. VAEs are generative models learning probability distributions: encoder outputs parameters of posterior distribution q(z|x), decoder models conditional p(x|z). VAEs enable: (1) Generation by sampling from prior, (2) Uncertainty quantification, (3) Interpolation in latent space, (4) Principled training objective (ELBO). Standard autoencoders cannot generate new samples as latent space lacks proper structure.`
    },
    {
      question: 'What are the trade-offs between VAEs and GANs?',
      answer: `VAEs: Probabilistic approach with stable training, clear objective (ELBO), interpretable latent space, but produces slightly blurry images due to reconstruction loss. Better for controllability and likelihood estimation. GANs: Adversarial training produces sharp, realistic images but training can be unstable with mode collapse. No explicit likelihood model. Excellent for high-quality synthesis but limited controllability. Choose VAEs for stability and control, GANs for image quality and realism.`
    },
    {
      question: 'What is β-VAE and how does it encourage disentanglement?',
      answer: `β-VAEs modify standard VAE objective by weighting KL term: ELBO = $\\mathbb{E}[\\log p(x|z)] - \\beta \\times \\text{KL}(q(z|x)||p(z))$. Higher $\\beta$ values encourage stronger independence between latent dimensions, promoting disentanglement where each dimension captures distinct factors of variation. Trade-off: increased $\\beta$ improves disentanglement but may reduce reconstruction quality. Disentangled representations enable interpretable generation and manipulation by modifying individual latent dimensions corresponding to specific semantic factors.`
    },
    {
      question: 'Why do VAE-generated images tend to be blurrier than GAN images?',
      answer: `Blurriness results from: (1) Pixel-wise reconstruction loss (MSE) encouraging averaging, (2) Gaussian output assumptions, (3) Mode averaging in decoder. The reconstruction loss optimizes for pixel-wise accuracy rather than perceptual quality, leading to averaging effects that produce blurry outputs. Solutions include: perceptual losses using pre-trained networks, adversarial training (VAE-GAN hybrids), different output distributions, and VQ-VAE for discrete latent representations.`
    }
  ],
  quizQuestions: [
    {
      id: 'vae1',
      question: 'What does the reparameterization trick enable?',
      options: ['Faster training', 'Backpropagation through sampling', 'Better image quality', 'Smaller models'],
      correctAnswer: 1,
      explanation: 'The reparameterization trick ($z = \\mu + \\sigma\\odot\\varepsilon$) moves randomness to an independent $\\varepsilon$, making the sampling operation differentiable so gradients can flow through it.'
    },
    {
      id: 'vae2',
      question: 'What does the KL divergence term in VAE loss encourage?',
      options: ['Better reconstruction', 'Structured latent space', 'Faster convergence', 'Larger capacity'],
      correctAnswer: 1,
      explanation: 'The KL divergence regularizes the latent space to match a prior (usually standard Gaussian), keeping it well-structured and enabling smooth interpolation and sampling.'
    },
    {
      id: 'vae3',
      question: 'What is the main advantage of VAEs over GANs?',
      options: ['Sharper images', 'Stable training', 'Faster inference', 'Larger models'],
      correctAnswer: 1,
      explanation: 'VAEs have more stable training than GANs because they optimize a well-defined loss function (ELBO), unlike GANs which involve a minimax game that can be unstable.'
    }
  ]
};
