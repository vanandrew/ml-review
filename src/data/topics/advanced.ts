import { Topic } from '../../types';

export const advancedTopics: Record<string, Topic> = {
  'generative-adversarial-networks': {
    id: 'generative-adversarial-networks',
    title: 'Generative Adversarial Networks (GANs)',
    category: 'advanced',
    description: 'Two neural networks competing to generate realistic data',
    content: `
      <h2>Generative Adversarial Networks (GANs)</h2>
      <p>GANs, introduced by Ian Goodfellow in 2014, consist of two neural networks—a generator and discriminator—competing in a minimax game. This adversarial training produces highly realistic synthetic data.</p>

      <h3>Architecture</h3>

      <h4>Generator (G)</h4>
      <ul>
        <li><strong>Input:</strong> Random noise vector z from latent space (e.g., Gaussian)</li>
        <li><strong>Goal:</strong> Generate fake data G(z) that looks real</li>
        <li><strong>Architecture:</strong> Upsampling layers (transpose convolutions)</li>
        <li><strong>Training:</strong> Maximize probability of fooling discriminator</li>
      </ul>

      <h4>Discriminator (D)</h4>
      <ul>
        <li><strong>Input:</strong> Real data or fake data from generator</li>
        <li><strong>Goal:</strong> Classify if input is real or fake</li>
        <li><strong>Architecture:</strong> Convolutional layers (like classifier)</li>
        <li><strong>Training:</strong> Maximize classification accuracy</li>
      </ul>

      <h3>Training Process</h3>
      <p>Minimax game with value function V(G, D):</p>
      <p>min_G max_D V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]</p>

      <h4>Alternating Training</h4>
      <ul>
        <li><strong>Step 1:</strong> Train discriminator to distinguish real vs fake</li>
        <li><strong>Step 2:</strong> Train generator to fool discriminator</li>
        <li><strong>Repeat:</strong> Continue until equilibrium (Nash equilibrium)</li>
      </ul>

      <h3>Loss Functions</h3>

      <h4>Discriminator Loss</h4>
      <p>L_D = -E[log D(x)] - E[log(1 - D(G(z)))]</p>
      <p>Minimize negative log-likelihood of correct classification</p>

      <h4>Generator Loss</h4>
      <p>L_G = -E[log D(G(z))]</p>
      <p>Maximize discriminator's probability of calling fakes "real"</p>

      <h3>Training Challenges</h3>
      <ul>
        <li><strong>Mode collapse:</strong> Generator produces limited variety</li>
        <li><strong>Vanishing gradients:</strong> Generator stops learning when D is too good</li>
        <li><strong>Training instability:</strong> Oscillations, failure to converge</li>
        <li><strong>Hyperparameter sensitivity:</strong> Requires careful tuning</li>
        <li><strong>Evaluation difficulty:</strong> No clear metric for quality</li>
      </ul>

      <h3>Popular GAN Variants</h3>

      <h4>DCGAN (Deep Convolutional GAN)</h4>
      <ul>
        <li><strong>Architecture guidelines:</strong> All-convolutional networks</li>
        <li><strong>Batch normalization:</strong> In both G and D</li>
        <li><strong>No pooling:</strong> Use strided convolutions</li>
        <li><strong>ReLU in G, LeakyReLU in D</strong></li>
      </ul>

      <h4>WGAN (Wasserstein GAN)</h4>
      <ul>
        <li><strong>Wasserstein distance:</strong> Better loss function</li>
        <li><strong>Weight clipping:</strong> Enforce Lipschitz constraint</li>
        <li><strong>More stable training:</strong> Reduced mode collapse</li>
      </ul>

      <h4>StyleGAN</h4>
      <ul>
        <li><strong>Style-based generator:</strong> Control at different scales</li>
        <li><strong>High-quality images:</strong> Photorealistic faces</li>
        <li><strong>Latent space manipulation:</strong> Fine-grained control</li>
      </ul>

      <h4>Conditional GAN (cGAN)</h4>
      <ul>
        <li><strong>Class conditioning:</strong> Generate specific classes</li>
        <li><strong>Both networks see labels:</strong> G(z, y) and D(x, y)</li>
        <li><strong>Controlled generation:</strong> Specify desired output</li>
      </ul>

      <h3>Applications</h3>
      <ul>
        <li><strong>Image generation:</strong> Realistic faces, objects, scenes</li>
        <li><strong>Image-to-image translation:</strong> Pix2Pix, CycleGAN</li>
        <li><strong>Super-resolution:</strong> Enhance image quality</li>
        <li><strong>Data augmentation:</strong> Generate training data</li>
        <li><strong>Art and creativity:</strong> Style transfer, artistic generation</li>
        <li><strong>Text-to-image:</strong> Generate images from descriptions</li>
      </ul>

      <h3>Evaluation Metrics</h3>
      <ul>
        <li><strong>Inception Score (IS):</strong> Measures quality and diversity</li>
        <li><strong>Fréchet Inception Distance (FID):</strong> Distance between real and fake distributions</li>
        <li><strong>Precision & Recall:</strong> Quality vs diversity trade-off</li>
        <li><strong>Human evaluation:</strong> Subjective quality assessment</li>
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
  },

  'variational-autoencoders': {
    id: 'variational-autoencoders',
    title: 'Variational Autoencoders (VAEs)',
    category: 'advanced',
    description: 'Probabilistic generative models with latent variable representations',
    content: `
      <h2>Variational Autoencoders (VAEs)</h2>
      <p>VAEs are generative models that learn a probabilistic mapping between data and a latent space. Unlike standard autoencoders, VAEs learn a distribution over latent codes, enabling sampling of new data points.</p>

      <h3>Architecture</h3>

      <h4>Encoder (Recognition Network)</h4>
      <ul>
        <li><strong>Input:</strong> Data point x</li>
        <li><strong>Output:</strong> Parameters of latent distribution q(z|x)</li>
        <li><strong>Typically:</strong> Mean μ and log-variance log σ² of Gaussian</li>
        <li><strong>Goal:</strong> Approximate posterior p(z|x)</li>
      </ul>

      <h4>Latent Space</h4>
      <ul>
        <li><strong>Representation:</strong> Low-dimensional code z</li>
        <li><strong>Distribution:</strong> Usually Gaussian N(μ, σ²)</li>
        <li><strong>Sampling:</strong> z = μ + σ ⊙ ε, where ε ~ N(0, I)</li>
        <li><strong>Reparameterization trick:</strong> Makes sampling differentiable</li>
      </ul>

      <h4>Decoder (Generative Network)</h4>
      <ul>
        <li><strong>Input:</strong> Latent code z</li>
        <li><strong>Output:</strong> Reconstruction x̂</li>
        <li><strong>Goal:</strong> Model p(x|z)</li>
        <li><strong>Architecture:</strong> Mirror of encoder (upsampling)</li>
      </ul>

      <h3>Loss Function</h3>
      <p>VAE loss = Reconstruction Loss + KL Divergence</p>

      <h4>Reconstruction Loss</h4>
      <ul>
        <li><strong>Binary data:</strong> Binary cross-entropy</li>
        <li><strong>Continuous data:</strong> Mean squared error</li>
        <li><strong>Goal:</strong> Decoded output matches input</li>
        <li><strong>Formula:</strong> E[log p(x|z)]</li>
      </ul>

      <h4>KL Divergence</h4>
      <ul>
        <li><strong>Formula:</strong> KL(q(z|x) || p(z))</li>
        <li><strong>Measures:</strong> Distance between learned and prior distribution</li>
        <li><strong>Prior:</strong> Usually standard Gaussian N(0, I)</li>
        <li><strong>Closed form:</strong> For Gaussian distributions</li>
        <li><strong>Regularization:</strong> Keeps latent space well-structured</li>
      </ul>

      <h4>Evidence Lower Bound (ELBO)</h4>
      <p>Maximize: ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))</p>
      <p>Equivalent to minimizing: Reconstruction Loss + KL Divergence</p>

      <h3>Reparameterization Trick</h3>
      <p>Problem: Can't backpropagate through random sampling</p>
      <p>Solution: z = μ(x) + σ(x) ⊙ ε, where ε ~ N(0, I)</p>
      <ul>
        <li><strong>Randomness:</strong> Moved to ε (independent of parameters)</li>
        <li><strong>Differentiable:</strong> Can compute gradients w.r.t. μ and σ</li>
        <li><strong>Enables training:</strong> Backpropagation through stochastic node</li>
      </ul>

      <h3>VAE vs Standard Autoencoder</h3>
      <ul>
        <li><strong>Latent space:</strong> VAE probabilistic, AE deterministic</li>
        <li><strong>Regularization:</strong> VAE has KL term, AE doesn't</li>
        <li><strong>Generation:</strong> VAE can generate new samples, AE can't reliably</li>
        <li><strong>Interpolation:</strong> VAE smoother, more meaningful</li>
        <li><strong>Training:</strong> VAE more complex, requires balancing losses</li>
      </ul>

      <h3>VAE vs GAN</h3>
      <ul>
        <li><strong>Training:</strong> VAE stable, GAN unstable</li>
        <li><strong>Quality:</strong> GAN sharper images, VAE blurrier</li>
        <li><strong>Diversity:</strong> VAE better coverage, GAN mode collapse risk</li>
        <li><strong>Evaluation:</strong> VAE has likelihood, GAN doesn't</li>
        <li><strong>Latent space:</strong> VAE more structured, GAN less interpretable</li>
      </ul>

      <h3>Applications</h3>
      <ul>
        <li><strong>Image generation:</strong> Sample from latent space</li>
        <li><strong>Interpolation:</strong> Smooth transitions between samples</li>
        <li><strong>Anomaly detection:</strong> High reconstruction error</li>
        <li><strong>Data compression:</strong> Efficient latent representations</li>
        <li><strong>Semi-supervised learning:</strong> Learn from labeled and unlabeled</li>
        <li><strong>Drug discovery:</strong> Generate molecular structures</li>
      </ul>

      <h3>Variants</h3>
      <ul>
        <li><strong>β-VAE:</strong> Weighted KL term for disentanglement</li>
        <li><strong>Conditional VAE:</strong> Generate specific classes</li>
        <li><strong>VQ-VAE:</strong> Discrete latent representations</li>
        <li><strong>Hierarchical VAE:</strong> Multiple latent layers</li>
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
        answer: `The reparameterization trick enables backpropagation through stochastic nodes by expressing random variables as deterministic functions of noise. Instead of sampling z ~ N(μ, σ²), we compute z = μ + σ ⊙ ε where ε ~ N(0,I). This transforms stochastic operation into deterministic computation with external randomness, allowing gradients to flow through μ and σ parameters. Essential for training VAEs because it makes the latent variable sampling differentiable while maintaining the desired probability distribution.`
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
        answer: `β-VAEs modify standard VAE objective by weighting KL term: ELBO = E[log p(x|z)] - β × KL(q(z|x)||p(z)). Higher β values encourage stronger independence between latent dimensions, promoting disentanglement where each dimension captures distinct factors of variation. Trade-off: increased β improves disentanglement but may reduce reconstruction quality. Disentangled representations enable interpretable generation and manipulation by modifying individual latent dimensions corresponding to specific semantic factors.`
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
        explanation: 'The reparameterization trick (z = μ + σ⊙ε) moves randomness to an independent ε, making the sampling operation differentiable so gradients can flow through it.'
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
  },

  'reinforcement-learning-basics': {
    id: 'reinforcement-learning-basics',
    title: 'Reinforcement Learning Basics',
    category: 'advanced',
    description: 'Learning through interaction with an environment to maximize rewards',
    content: `
      <h2>Reinforcement Learning Basics</h2>
      <p>Reinforcement Learning (RL) is a paradigm where an agent learns to make decisions by interacting with an environment, receiving rewards or penalties, and learning to maximize cumulative reward over time.</p>

      <h3>Key Components</h3>

      <h4>Agent</h4>
      <ul>
        <li><strong>Role:</strong> Makes decisions and takes actions</li>
        <li><strong>Goal:</strong> Learn policy to maximize expected return</li>
        <li><strong>Components:</strong> Policy, value function, sometimes model</li>
      </ul>

      <h4>Environment</h4>
      <ul>
        <li><strong>Role:</strong> World agent interacts with</li>
        <li><strong>Provides:</strong> States and rewards in response to actions</li>
        <li><strong>Examples:</strong> Game, robot simulator, real world</li>
      </ul>

      <h4>State (s)</h4>
      <ul>
        <li><strong>Current situation:</strong> Observable information</li>
        <li><strong>Markov property:</strong> Contains all relevant information</li>
        <li><strong>State space:</strong> Set of all possible states</li>
      </ul>

      <h4>Action (a)</h4>
      <ul>
        <li><strong>Decision:</strong> What agent can do</li>
        <li><strong>Action space:</strong> Discrete or continuous</li>
        <li><strong>Examples:</strong> Move up/down, motor torques</li>
      </ul>

      <h4>Reward (r)</h4>
      <ul>
        <li><strong>Feedback signal:</strong> Scalar value from environment</li>
        <li><strong>Immediate:</strong> Received after each action</li>
        <li><strong>Goal:</strong> Maximize cumulative reward</li>
      </ul>

      <h4>Policy (π)</h4>
      <ul>
        <li><strong>Strategy:</strong> Mapping from states to actions</li>
        <li><strong>Deterministic:</strong> a = π(s)</li>
        <li><strong>Stochastic:</strong> π(a|s) = probability of action a in state s</li>
      </ul>

      <h3>Key Concepts</h3>

      <h4>Return (G)</h4>
      <p>Total cumulative reward from time t:</p>
      <p>G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...</p>
      <ul>
        <li><strong>γ (gamma):</strong> Discount factor (0 ≤ γ ≤ 1)</li>
        <li><strong>γ = 0:</strong> Only immediate reward matters</li>
        <li><strong>γ = 1:</strong> All future rewards equally important</li>
        <li><strong>γ < 1:</strong> Prefer immediate rewards, ensures convergence</li>
      </ul>

      <h4>Value Functions</h4>

      <h5>State-Value Function V(s)</h5>
      <p>Expected return starting from state s:</p>
      <p>V^π(s) = E[G_t | S_t = s]</p>

      <h5>Action-Value Function Q(s, a)</h5>
      <p>Expected return from state s, taking action a:</p>
      <p>Q^π(s, a) = E[G_t | S_t = s, A_t = a]</p>

      <h4>Bellman Equation</h4>
      <p>Recursive relationship for value functions:</p>
      <p>V(s) = E[R + γV(s')]</p>
      <p>Q(s, a) = E[R + γ max_a' Q(s', a')]</p>

      <h3>Exploration vs Exploitation</h3>
      <ul>
        <li><strong>Exploitation:</strong> Choose best known action</li>
        <li><strong>Exploration:</strong> Try new actions to learn more</li>
        <li><strong>Trade-off:</strong> Balance needed for optimal learning</li>
        <li><strong>ε-greedy:</strong> Explore with probability ε, exploit with 1-ε</li>
        <li><strong>Softmax:</strong> Probabilistic action selection</li>
      </ul>

      <h3>Main RL Algorithms</h3>

      <h4>Q-Learning (Off-Policy)</h4>
      <ul>
        <li><strong>Model-free:</strong> Learn Q(s, a) directly</li>
        <li><strong>Off-policy:</strong> Learn optimal policy while following different policy</li>
        <li><strong>Update rule:</strong> Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]</li>
        <li><strong>Converges to optimal Q*</strong></li>
      </ul>

      <h4>SARSA (On-Policy)</h4>
      <ul>
        <li><strong>State-Action-Reward-State-Action</strong></li>
        <li><strong>On-policy:</strong> Learn policy being followed</li>
        <li><strong>Update rule:</strong> Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]</li>
        <li><strong>More conservative than Q-learning</strong></li>
      </ul>

      <h4>Deep Q-Networks (DQN)</h4>
      <ul>
        <li><strong>Neural network:</strong> Approximate Q-function</li>
        <li><strong>Experience replay:</strong> Break correlation in data</li>
        <li><strong>Target network:</strong> Stabilize training</li>
        <li><strong>Breakthrough:</strong> Play Atari games from pixels</li>
      </ul>

      <h4>Policy Gradient Methods</h4>
      <ul>
        <li><strong>Direct optimization:</strong> Learn policy directly</li>
        <li><strong>REINFORCE:</strong> Monte Carlo policy gradient</li>
        <li><strong>Actor-Critic:</strong> Combine policy and value function</li>
        <li><strong>Continuous actions:</strong> Natural for continuous control</li>
      </ul>

      <h3>Applications</h3>
      <ul>
        <li><strong>Game playing:</strong> AlphaGo, Dota 2, StarCraft</li>
        <li><strong>Robotics:</strong> Locomotion, manipulation</li>
        <li><strong>Autonomous vehicles:</strong> Driving decisions</li>
        <li><strong>Resource management:</strong> Data center cooling, traffic lights</li>
        <li><strong>Finance:</strong> Trading strategies</li>
        <li><strong>Recommendation systems:</strong> Content suggestions</li>
      </ul>

      <h3>Challenges</h3>
      <ul>
        <li><strong>Sample efficiency:</strong> Requires many interactions</li>
        <li><strong>Credit assignment:</strong> Which actions led to reward?</li>
        <li><strong>Sparse rewards:</strong> Long time between feedback</li>
        <li><strong>Exploration:</strong> Finding good strategies</li>
        <li><strong>Stability:</strong> Training can be unstable</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `import numpy as np
import gym

# Q-Learning implementation
class QLearning:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def choose_action(self, state):
        # ε-greedy policy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)  # Explore
        else:
            return np.argmax(self.Q[state])  # Exploit

    def update(self, state, action, reward, next_state):
        # Q-learning update rule
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state, best_next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.lr * td_error

# Train on simple environment
env = gym.make('FrozenLake-v1')
n_states = env.observation_space.n
n_actions = env.action_space.n

agent = QLearning(n_states, n_actions, lr=0.1, gamma=0.99, epsilon=0.1)

# Training loop
n_episodes = 10000
rewards = []

for episode in range(n_episodes):
    state = env.reset()[0]
    total_reward = 0
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.update(state, action, reward, next_state)

        state = next_state
        total_reward += reward

    rewards.append(total_reward)

    # Decay epsilon
    agent.epsilon = max(0.01, agent.epsilon * 0.995)

    if episode % 1000 == 0:
        avg_reward = np.mean(rewards[-100:])
        print(f"Episode {episode}, Avg Reward: {avg_reward:.3f}, ε: {agent.epsilon:.3f}")

# Test learned policy
state = env.reset()[0]
done = False
total_reward = 0

agent.epsilon = 0  # Greedy policy
while not done:
    action = agent.choose_action(state)
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward

print(f"\\nTest episode reward: {total_reward}")`,
        explanation: 'Q-Learning implementation with ε-greedy exploration on a simple grid world environment.'
      },
      {
        language: 'Python',
        code: `import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Deep Q-Network (DQN)
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-network and target network
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state)
            return q_values.argmax().item()

    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Current Q-values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Loss and optimization
        loss = nn.MSELoss()(q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Usage
state_dim = 4
action_dim = 2
agent = DQNAgent(state_dim, action_dim)

# Training loop example
for episode in range(1000):
    state = np.random.randn(state_dim)  # Replace with env.reset()
    done = False

    while not done:
        action = agent.choose_action(state)
        # next_state, reward, done = env.step(action)
        next_state = np.random.randn(state_dim)
        reward = np.random.rand()
        done = np.random.rand() > 0.95

        agent.replay_buffer.push(state, action, reward, next_state, done)
        loss = agent.train()

        state = next_state

    # Update target network periodically
    if episode % 10 == 0:
        agent.update_target_network()

    # Decay epsilon
    agent.epsilon = max(0.01, agent.epsilon * 0.995)

    if episode % 100 == 0:
        print(f"Episode {episode}, ε: {agent.epsilon:.3f}")`,
        explanation: 'Deep Q-Network (DQN) implementation with experience replay and target network for stable training.'
      }
    ],
    interviewQuestions: [
      {
        question: 'Explain the difference between on-policy and off-policy learning.',
        answer: `On-policy methods learn about the policy they follow during exploration (e.g., SARSA learns Q-values for the ε-greedy policy it uses). Off-policy methods can learn about a target policy while following a different behavior policy (e.g., Q-learning learns optimal policy while following ε-greedy). Off-policy enables: (1) Learning from historical data, (2) Sample efficiency through experience replay, (3) Exploratory behavior separate from target policy. On-policy methods are often more stable but less sample efficient.`
      },
      {
        question: 'What is the exploration-exploitation trade-off?',
        answer: `The exploration-exploitation trade-off balances between taking known good actions (exploitation) and trying new actions to discover potentially better ones (exploration). Too much exploitation leads to suboptimal policies; too much exploration wastes time on poor actions. Strategies include: (1) ε-greedy - random exploration with probability ε, (2) UCB - upper confidence bound selection, (3) Thompson sampling - probabilistic exploration, (4) Optimistic initialization, (5) Decay schedules reducing exploration over time.`
      },
      {
        question: 'How does Q-learning differ from SARSA?',
        answer: `Q-learning (off-policy): Updates Q(s,a) using maximum next action: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]. SARSA (on-policy): Updates using actual next action taken: Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]. Q-learning learns optimal policy regardless of exploration strategy; SARSA learns policy for the actual behavior being followed. SARSA is often safer in practice as it accounts for exploration during learning.`
      },
      {
        question: 'What is the role of the discount factor γ?',
        answer: `Discount factor γ ∈ [0,1] determines how much future rewards are valued relative to immediate rewards. γ = 0: only immediate rewards matter (myopic). γ = 1: all future rewards equally weighted (infinite horizon). Practical effects: (1) Controls learning horizon, (2) Ensures convergence in infinite scenarios, (3) Represents uncertainty about future, (4) Balances short vs long-term planning. Typical values: 0.9-0.99. Lower γ for shorter episodes, higher for long-term planning tasks.`
      },
      {
        question: 'Explain experience replay and why it\'s important for DQN.',
        answer: `Experience replay stores transitions (s, a, r, s') in a buffer and randomly samples batches for training. Benefits: (1) Breaks temporal correlations in sequential data, (2) Enables multiple learning updates per environment step, (3) Stabilizes training by reducing variance, (4) Allows off-policy learning from historical experiences. Critical for DQN because neural networks require i.i.d. data for stable learning, but RL naturally produces correlated sequential experiences that can cause catastrophic forgetting.`
      },
      {
        question: 'What is the credit assignment problem in RL?',
        answer: `Credit assignment determines which actions were responsible for observed rewards, especially challenging when rewards are delayed or sparse. Problems: (1) Temporal credit assignment - which past actions led to current reward, (2) Structural credit assignment - which components of complex actions were important. Solutions: (1) Eligibility traces for temporal credit, (2) Advantage estimation, (3) Return decomposition, (4) Hierarchical RL, (5) Attention mechanisms in neural policies, (6) Causal inference methods.`
      }
    ],
    quizQuestions: [
      {
        id: 'rl1',
        question: 'What is the goal of reinforcement learning?',
        options: ['Classify data', 'Maximize cumulative reward', 'Minimize loss', 'Cluster samples'],
        correctAnswer: 1,
        explanation: 'The goal of RL is for an agent to learn a policy that maximizes the expected cumulative reward over time through interaction with an environment.'
      },
      {
        id: 'rl2',
        question: 'What does the discount factor γ control?',
        options: ['Learning rate', 'Importance of future rewards', 'Exploration rate', 'Batch size'],
        correctAnswer: 1,
        explanation: 'The discount factor γ (0 ≤ γ ≤ 1) controls how much the agent values future rewards. γ near 0 makes the agent myopic (only immediate rewards), while γ near 1 makes future rewards important.'
      },
      {
        id: 'rl3',
        question: 'What problem does experience replay solve in DQN?',
        options: ['Slow training', 'Correlated samples', 'Large memory', 'Overfitting'],
        correctAnswer: 1,
        explanation: 'Experience replay stores transitions in a buffer and samples randomly for training, breaking the correlation between consecutive samples and stabilizing learning.'
      }
    ]
  },

  'model-compression': {
    id: 'model-compression',
    title: 'Model Compression',
    category: 'advanced',
    description: 'Techniques to reduce model size and inference cost',
    content: `
      <h2>Model Compression</h2>
      <p>Model compression techniques reduce the size, memory footprint, and computational requirements of neural networks while maintaining performance. Essential for deploying models on resource-constrained devices.</p>

      <h3>Why Model Compression?</h3>
      <ul>
        <li><strong>Memory constraints:</strong> Mobile devices have limited RAM</li>
        <li><strong>Latency requirements:</strong> Real-time applications need fast inference</li>
        <li><strong>Energy efficiency:</strong> Battery-powered devices</li>
        <li><strong>Bandwidth:</strong> Smaller models faster to download/update</li>
        <li><strong>Cost:</strong> Reduce cloud inference costs</li>
      </ul>

      <h3>Main Techniques</h3>

      <h4>1. Quantization</h4>
      <p>Reduce numerical precision of weights and activations</p>

      <h5>Post-Training Quantization</h5>
      <ul>
        <li><strong>FP32 → FP16:</strong> Half precision, ~2x speedup</li>
        <li><strong>FP32 → INT8:</strong> 8-bit integers, ~4x speedup</li>
        <li><strong>INT4/INT2:</strong> Extreme quantization</li>
        <li><strong>No retraining:</strong> Apply after training</li>
        <li><strong>Some accuracy loss:</strong> Usually 1-2%</li>
      </ul>

      <h5>Quantization-Aware Training (QAT)</h5>
      <ul>
        <li><strong>Simulate quantization:</strong> During training</li>
        <li><strong>Learn to be robust:</strong> To low precision</li>
        <li><strong>Better accuracy:</strong> Than post-training quantization</li>
        <li><strong>Fake quantization:</strong> Forward low-precision, backward high-precision</li>
      </ul>

      <h4>2. Pruning</h4>
      <p>Remove unnecessary weights or neurons</p>

      <h5>Unstructured Pruning</h5>
      <ul>
        <li><strong>Individual weights:</strong> Set small weights to zero</li>
        <li><strong>Sparse matrices:</strong> Requires special hardware support</li>
        <li><strong>High compression:</strong> 90%+ sparsity possible</li>
        <li><strong>Magnitude-based:</strong> Prune weights with smallest absolute value</li>
      </ul>

      <h5>Structured Pruning</h5>
      <ul>
        <li><strong>Entire units:</strong> Remove channels, filters, or layers</li>
        <li><strong>Hardware friendly:</strong> Works on standard hardware</li>
        <li><strong>Lower sparsity:</strong> But actual speedups</li>
        <li><strong>Examples:</strong> Channel pruning, filter pruning</li>
      </ul>

      <h5>Iterative Pruning</h5>
      <ul>
        <li><strong>Train → Prune → Fine-tune → Repeat</strong></li>
        <li><strong>Gradual:</strong> Removes weights slowly</li>
        <li><strong>Better accuracy:</strong> Than one-shot pruning</li>
        <li><strong>Lottery Ticket Hypothesis:</strong> Sparse sub-networks exist from initialization</li>
      </ul>

      <h4>3. Knowledge Distillation</h4>
      <p>Transfer knowledge from large teacher to small student</p>
      <ul>
        <li><strong>Teacher model:</strong> Large, accurate model</li>
        <li><strong>Student model:</strong> Small, efficient model</li>
        <li><strong>Soft targets:</strong> Use teacher's probability distributions</li>
        <li><strong>Temperature:</strong> Soften probabilities for better transfer</li>
        <li><strong>Loss:</strong> Combination of hard labels and soft targets</li>
      </ul>

      <h5>Distillation Loss</h5>
      <p>L = αL_hard + (1-α)L_soft</p>
      <ul>
        <li><strong>L_hard:</strong> Cross-entropy with true labels</li>
        <li><strong>L_soft:</strong> KL divergence with teacher outputs</li>
        <li><strong>α:</strong> Balance between losses</li>
        <li><strong>Temperature T:</strong> Softmax(logits/T)</li>
      </ul>

      <h4>4. Low-Rank Factorization</h4>
      <p>Decompose weight matrices into smaller matrices</p>
      <ul>
        <li><strong>SVD:</strong> Singular Value Decomposition</li>
        <li><strong>Tucker decomposition:</strong> For convolutional layers</li>
        <li><strong>Example:</strong> W (m×n) → U (m×k) × V (k×n) where k < min(m,n)</li>
        <li><strong>Parameter reduction:</strong> From mn to k(m+n)</li>
      </ul>

      <h4>5. Neural Architecture Search (NAS)</h4>
      <p>Automatically design efficient architectures</p>
      <ul>
        <li><strong>Search space:</strong> Possible architectures</li>
        <li><strong>Objective:</strong> Accuracy + efficiency</li>
        <li><strong>Examples:</strong> MobileNet, EfficientNet</li>
        <li><strong>Depthwise separable convolutions:</strong> Reduce parameters</li>
      </ul>

      <h3>Compression Metrics</h3>
      <ul>
        <li><strong>Model size:</strong> Storage in MB</li>
        <li><strong>Inference time:</strong> Latency per sample</li>
        <li><strong>FLOPs:</strong> Floating point operations</li>
        <li><strong>Memory usage:</strong> Peak RAM during inference</li>
        <li><strong>Accuracy:</strong> Task performance</li>
        <li><strong>Energy consumption:</strong> Battery usage</li>
      </ul>

      <h3>Practical Considerations</h3>
      <ul>
        <li><strong>Hardware support:</strong> Not all optimizations work on all devices</li>
        <li><strong>Framework support:</strong> TensorFlow Lite, ONNX, TorchScript</li>
        <li><strong>Accuracy vs size:</strong> Trade-off to balance</li>
        <li><strong>Combined techniques:</strong> Quantization + pruning often best</li>
        <li><strong>Task-dependent:</strong> Different tasks tolerate compression differently</li>
      </ul>

      <h3>Applications</h3>
      <ul>
        <li><strong>Mobile apps:</strong> On-device inference</li>
        <li><strong>Edge devices:</strong> IoT, embedded systems</li>
        <li><strong>Autonomous vehicles:</strong> Real-time processing</li>
        <li><strong>Cloud serving:</strong> Reduce infrastructure costs</li>
        <li><strong>Federated learning:</strong> Smaller models for clients</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `import torch
import torch.nn as nn
import torch.quantization as quantization

# === 1. QUANTIZATION ===

# Post-Training Static Quantization
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * 30 * 30, 10)
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

# Prepare model for quantization
model = SimpleNet()
model.eval()

# Specify quantization config
model.qconfig = quantization.get_default_qconfig('fbgemm')
quantization.prepare(model, inplace=True)

# Calibrate with representative data
calibration_data = torch.randn(100, 3, 32, 32)
with torch.no_grad():
    for i in range(10):
        model(calibration_data[i*10:(i+1)*10])

# Convert to quantized model
quantized_model = quantization.convert(model, inplace=False)

# Compare sizes
def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    print(f"{label}: {size:.2f} MB")
    os.remove('temp.p')

print_size_of_model(model, "FP32 Model")
print_size_of_model(quantized_model, "INT8 Quantized Model")

# === 2. PRUNING ===
import torch.nn.utils.prune as prune

model = SimpleNet()

# Unstructured pruning: prune 30% of weights
prune.l1_unstructured(model.conv1, name='weight', amount=0.3)

# Check sparsity
print(f"Sparsity in conv1: {100. * float(torch.sum(model.conv1.weight == 0)) / float(model.conv1.weight.nelement()):.2f}%")

# Make pruning permanent
prune.remove(model.conv1, 'weight')

# Structured pruning: remove entire channels
prune.ln_structured(
    model.conv1,
    name='weight',
    amount=0.5,  # Remove 50% of channels
    n=2,  # L2 norm
    dim=0  # Prune along output channels
)

print(f"Conv1 output channels after pruning: {model.conv1.out_channels}")

# Global pruning across multiple layers
parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.fc, 'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.5,  # 50% sparsity globally
)`,
        explanation: 'Quantization and pruning implementations showing post-training quantization and structured/unstructured pruning.'
      },
      {
        language: 'Python',
        code: `import torch
import torch.nn as nn
import torch.nn.functional as F

# === 3. KNOWLEDGE DISTILLATION ===

class TeacherModel(nn.Module):
    """Large, accurate model"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 1200),
            nn.ReLU(),
            nn.Linear(1200, 1200),
            nn.ReLU(),
            nn.Linear(1200, 10)
        )

    def forward(self, x):
        return self.layers(x)

class StudentModel(nn.Module):
    """Small, efficient model"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 10)
        )

    def forward(self, x):
        return self.layers(x)

def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.5):
    """
    Combined loss for knowledge distillation

    Args:
        student_logits: Student model outputs
        teacher_logits: Teacher model outputs
        labels: True labels
        temperature: Softmax temperature for soft targets
        alpha: Weight for hard loss (1-alpha for soft loss)
    """
    # Hard loss: cross-entropy with true labels
    hard_loss = F.cross_entropy(student_logits, labels)

    # Soft loss: KL divergence with teacher
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)

    # Combined loss
    return alpha * hard_loss + (1 - alpha) * soft_loss

# Training
teacher = TeacherModel()
student = StudentModel()

# Assume teacher is pre-trained
teacher.eval()

optimizer = torch.optim.Adam(student.parameters(), lr=0.001)

# Training loop
x = torch.randn(32, 784)
labels = torch.randint(0, 10, (32,))

with torch.no_grad():
    teacher_logits = teacher(x)

student_logits = student(x)
loss = distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.3)

loss.backward()
optimizer.step()

print(f"Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")
print(f"Student params: {sum(p.numel() for p in student.parameters()):,}")
print(f"Compression ratio: {sum(p.numel() for p in teacher.parameters()) / sum(p.numel() for p in student.parameters()):.2f}x")

# === 4. LOW-RANK FACTORIZATION ===

def low_rank_decomposition(layer, rank_ratio=0.5):
    """Decompose a linear layer using SVD"""
    W = layer.weight.data
    U, S, V = torch.svd(W)

    # Keep only top-k singular values
    k = int(rank_ratio * min(W.shape))
    U_k = U[:, :k]
    S_k = S[:k]
    V_k = V[:, :k]

    # Create two smaller layers
    layer1 = nn.Linear(layer.in_features, k, bias=False)
    layer2 = nn.Linear(k, layer.out_features, bias=True)

    layer1.weight.data = (V_k * S_k).t()
    layer2.weight.data = U_k
    if layer.bias is not None:
        layer2.bias.data = layer.bias.data

    return nn.Sequential(layer1, layer2)

# Original layer
original_layer = nn.Linear(1000, 1000)
print(f"Original params: {sum(p.numel() for p in original_layer.parameters()):,}")

# Compressed layer (rank=500)
compressed_layer = low_rank_decomposition(original_layer, rank_ratio=0.5)
print(f"Compressed params: {sum(p.numel() for p in compressed_layer.parameters()):,}")

# Test equivalence
x = torch.randn(1, 1000)
print(f"Output difference: {(original_layer(x) - compressed_layer(x)).abs().mean():.6f}")`,
        explanation: 'Knowledge distillation for training small models from large teachers, and low-rank matrix factorization for compression.'
      }
    ],
    interviewQuestions: [
      {
        question: 'What is the difference between quantization and pruning?',
        answer: `Quantization reduces numerical precision (FP32 → INT8) to decrease memory and computation while maintaining model structure. Pruning removes less important weights/neurons, creating sparse networks. Quantization: uniform compression, simpler implementation, compatible with standard hardware. Pruning: variable compression ratios, requires sparsity-aware software/hardware for full benefits, can dramatically reduce model size. Both can be combined for maximum compression. Choose quantization for deployment simplicity, pruning for aggressive size reduction.`
      },
      {
        question: 'Explain knowledge distillation and why soft targets help.',
        answer: `Knowledge distillation trains a smaller student model to mimic a larger teacher model by learning from both ground truth labels and teacher predictions. Soft targets (teacher's probability distribution) provide richer information than hard labels: (1) Reveal similarities between classes, (2) Encode uncertainty and confidence, (3) Transfer learned representations more effectively. Temperature scaling in softmax amplifies these differences. Student learns not just correct answers but teacher's reasoning process, often achieving better performance than training directly on hard labels.`
      },
      {
        question: 'What is the lottery ticket hypothesis?',
        answer: `The lottery ticket hypothesis states that dense neural networks contain sparse subnetworks ("winning tickets") that can achieve comparable accuracy to the full network when trained in isolation. Key findings: (1) Random pruning destroys performance, but structured pruning can preserve it, (2) Winning tickets require specific initialization values, (3) Early-bird tickets can be found early in training. Implications: suggests networks are over-parameterized and efficient architectures exist within them. Challenges traditional views on network size requirements.`
      },
      {
        question: 'Compare structured vs unstructured pruning.',
        answer: `Unstructured pruning removes individual weights based on magnitude, creating sparse but irregular patterns. Benefits: fine-grained control, higher compression ratios. Drawbacks: requires specialized sparse computation libraries for speedup. Structured pruning removes entire channels, filters, or layers, maintaining regular computation patterns. Benefits: immediate speedup on standard hardware, simpler implementation. Drawbacks: coarser granularity, potentially lower compression ratios. Choose structured for deployment simplicity, unstructured for maximum compression.`
      },
      {
        question: 'How does quantization-aware training differ from post-training quantization?',
        answer: `Post-training quantization applies after training by calibrating on small dataset - fast but may lose accuracy. Quantization-aware training (QAT) simulates quantization during training with fake quantization operators, allowing model to adapt to reduced precision. QAT benefits: (1) Better accuracy preservation, (2) Learns quantization-friendly representations, (3) Can optimize for specific hardware. Trade-offs: requires retraining, longer development time. Use post-training for quick deployment, QAT for accuracy-critical applications.`
      },
      {
        question: 'What are the trade-offs between different compression techniques?',
        answer: `Compression vs. accuracy: More aggressive compression typically reduces accuracy. Speed vs. size: Some techniques optimize inference speed (quantization), others model size (pruning). Hardware compatibility: Quantization works on most hardware; pruning needs sparse computation support. Development effort: Post-training methods are easier; training-aware methods require more work. Flexibility: Knowledge distillation allows architectural changes; other methods preserve structure. Choose combination based on deployment constraints, accuracy requirements, and development resources.`
      }
    ],
    quizQuestions: [
      {
        id: 'compress1',
        question: 'What does quantization do to a neural network?',
        options: ['Removes weights', 'Reduces numerical precision', 'Changes architecture', 'Adds regularization'],
        correctAnswer: 1,
        explanation: 'Quantization reduces the numerical precision of weights and activations (e.g., FP32 to INT8), significantly reducing model size and speeding up inference.'
      },
      {
        id: 'compress2',
        question: 'In knowledge distillation, what is the "teacher"?',
        options: ['Training algorithm', 'Large pre-trained model', 'Loss function', 'Dataset'],
        correctAnswer: 1,
        explanation: 'The teacher is a large, accurate model whose knowledge is transferred to a smaller student model through soft targets (probability distributions).'
      },
      {
        id: 'compress3',
        question: 'What is an advantage of structured pruning over unstructured?',
        options: ['Higher sparsity', 'Works on standard hardware', 'Better accuracy', 'Easier to implement'],
        correctAnswer: 1,
        explanation: 'Structured pruning removes entire units (channels, filters) producing dense smaller matrices that work efficiently on standard hardware, unlike unstructured pruning which creates sparse matrices requiring special support.'
      }
    ]
  },

  'federated-learning': {
    id: 'federated-learning',
    title: 'Federated Learning',
    category: 'advanced',
    description: 'Collaborative training across decentralized devices without sharing data',
    content: `
      <h2>Federated Learning</h2>
      <p>Federated Learning (FL) enables training machine learning models across multiple decentralized devices or servers holding local data samples, without exchanging the data itself. This preserves privacy while enabling collaborative learning.</p>

      <h3>Key Principles</h3>
      <ul>
        <li><strong>Data stays local:</strong> Never leaves user devices</li>
        <li><strong>Privacy-preserving:</strong> No raw data shared</li>
        <li><strong>Collaborative:</strong> Multiple parties contribute</li>
        <li><strong>Decentralized:</strong> No central data repository</li>
        <li><strong>Model aggregation:</strong> Combine local updates</li>
      </ul>

      <h3>Federated Averaging (FedAvg)</h3>
      <p>Most common FL algorithm:</p>

      <h4>Algorithm Steps</h4>
      <ul>
        <li><strong>1. Server initialization:</strong> Initialize global model w</li>
        <li><strong>2. Client selection:</strong> Select subset of clients (e.g., 10%)</li>
        <li><strong>3. Download:</strong> Clients download current global model</li>
        <li><strong>4. Local training:</strong> Each client trains on local data</li>
        <li><strong>5. Upload:</strong> Clients send model updates to server</li>
        <li><strong>6. Aggregation:</strong> Server averages updates: w ← Σ(n_k/n)w_k</li>
        <li><strong>7. Repeat:</strong> Until convergence</li>
      </ul>

      <h4>Weighted Averaging</h4>
      <p>w_global = Σ(n_k / n_total) × w_k</p>
      <ul>
        <li><strong>n_k:</strong> Number of samples on client k</li>
        <li><strong>n_total:</strong> Total samples across all clients</li>
        <li><strong>w_k:</strong> Model weights from client k</li>
      </ul>

      <h3>Challenges</h3>

      <h4>1. Non-IID Data</h4>
      <ul>
        <li><strong>Problem:</strong> Data distribution varies across clients</li>
        <li><strong>Example:</strong> User's keyboard learns their vocabulary</li>
        <li><strong>Impact:</strong> Slow convergence, lower accuracy</li>
        <li><strong>Solutions:</strong> Data augmentation, personalization</li>
      </ul>

      <h4>2. Communication Efficiency</h4>
      <ul>
        <li><strong>Problem:</strong> Limited bandwidth, costly communication</li>
        <li><strong>Bottleneck:</strong> Uploading/downloading model weights</li>
        <li><strong>Solutions:</strong> Gradient compression, fewer rounds, federated dropout</li>
      </ul>

      <h4>3. Systems Heterogeneity</h4>
      <ul>
        <li><strong>Compute power:</strong> Different device capabilities</li>
        <li><strong>Network speed:</strong> Variable connection quality</li>
        <li><strong>Availability:</strong> Devices drop in/out</li>
        <li><strong>Solutions:</strong> Asynchronous updates, client sampling</li>
      </ul>

      <h4>4. Privacy and Security</h4>
      <ul>
        <li><strong>Model inversion:</strong> Reconstruct training data from updates</li>
        <li><strong>Membership inference:</strong> Determine if data was in training set</li>
        <li><strong>Poisoning attacks:</strong> Malicious clients corrupt model</li>
        <li><strong>Solutions:</strong> Differential privacy, secure aggregation</li>
      </ul>

      <h3>Privacy-Enhancing Technologies</h3>

      <h4>Differential Privacy (DP)</h4>
      <ul>
        <li><strong>Add noise:</strong> To gradients before sharing</li>
        <li><strong>Privacy budget ε:</strong> Controls privacy-utility trade-off</li>
        <li><strong>Clip gradients:</strong> Bound sensitivity</li>
        <li><strong>Guarantees:</strong> Mathematical privacy bounds</li>
      </ul>

      <h4>Secure Aggregation</h4>
      <ul>
        <li><strong>Cryptographic protocol:</strong> Server learns only aggregate</li>
        <li><strong>No individual updates:</strong> Revealed to server</li>
        <li><strong>Homomorphic encryption:</strong> Compute on encrypted data</li>
        <li><strong>Multi-party computation:</strong> Collaborative without revealing</li>
      </ul>

      <h3>Variants</h3>

      <h4>Cross-Device FL</h4>
      <ul>
        <li><strong>Many clients:</strong> Millions of mobile devices</li>
        <li><strong>Unreliable:</strong> Devices frequently offline</li>
        <li><strong>Small data:</strong> Per device</li>
        <li><strong>Example:</strong> Gboard keyboard predictions</li>
      </ul>

      <h4>Cross-Silo FL</h4>
      <ul>
        <li><strong>Few clients:</strong> Organizations, hospitals</li>
        <li><strong>Reliable:</strong> Stable servers</li>
        <li><strong>Large data:</strong> Per organization</li>
        <li><strong>Example:</strong> Healthcare collaborations</li>
      </ul>

      <h4>Personalized FL</h4>
      <ul>
        <li><strong>Local adaptation:</strong> Personalize global model</li>
        <li><strong>Meta-learning:</strong> Learn good initialization</li>
        <li><strong>Fine-tuning:</strong> Last layers on local data</li>
        <li><strong>Multi-task:</strong> Shared and personal layers</li>
      </ul>

      <h3>Applications</h3>
      <ul>
        <li><strong>Mobile keyboards:</strong> Next-word prediction (Gboard)</li>
        <li><strong>Healthcare:</strong> Multi-hospital collaborations</li>
        <li><strong>Finance:</strong> Fraud detection across banks</li>
        <li><strong>IoT:</strong> Smart home devices</li>
        <li><strong>Autonomous vehicles:</strong> Collaborative perception</li>
        <li><strong>Recommendations:</strong> Privacy-preserving personalization</li>
      </ul>

      <h3>Frameworks</h3>
      <ul>
        <li><strong>TensorFlow Federated (TFF):</strong> Google's FL framework</li>
        <li><strong>PySyft:</strong> Privacy-preserving ML</li>
        <li><strong>FATE:</strong> Federated AI Technology Enabler</li>
        <li><strong>Flower:</strong> Flexible federated learning framework</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `import torch
import torch.nn as nn
import torch.optim as optim
import copy
from typing import List

# Simple Federated Learning implementation

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class FederatedServer:
    def __init__(self, model):
        self.global_model = model

    def aggregate(self, client_models, client_weights):
        """
        Federated Averaging: weighted average of client models

        Args:
            client_models: List of client model state dicts
            client_weights: List of weights (e.g., data size per client)
        """
        # Normalize weights
        total_weight = sum(client_weights)
        weights = [w / total_weight for w in client_weights]

        # Initialize aggregated state dict
        global_dict = self.global_model.state_dict()

        # Average each parameter
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
            for client_dict, weight in zip(client_models, weights):
                global_dict[key] += weight * client_dict[key].float()

        self.global_model.load_state_dict(global_dict)

    def get_global_model(self):
        return copy.deepcopy(self.global_model)

class FederatedClient:
    def __init__(self, client_id, train_data, train_labels):
        self.client_id = client_id
        self.train_data = train_data
        self.train_labels = train_labels
        self.model = None

    def download_model(self, global_model):
        """Download global model from server"""
        self.model = copy.deepcopy(global_model)

    def train(self, epochs=1, lr=0.01):
        """Train on local data"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(self.train_data)
            loss = criterion(outputs, self.train_labels)
            loss.backward()
            optimizer.step()

        return loss.item()

    def upload_model(self):
        """Upload model to server"""
        return self.model.state_dict()

    def get_data_size(self):
        return len(self.train_data)

# Federated Learning simulation
def federated_learning(server, clients, num_rounds=10, clients_per_round=5, local_epochs=1):
    """
    Simulate federated learning

    Args:
        server: FederatedServer instance
        clients: List of FederatedClient instances
        num_rounds: Number of communication rounds
        clients_per_round: Number of clients selected per round
        local_epochs: Local training epochs per client
    """
    for round_num in range(num_rounds):
        print(f"\\n=== Round {round_num + 1}/{num_rounds} ===")

        # Select random subset of clients
        import random
        selected_clients = random.sample(clients, min(clients_per_round, len(clients)))

        client_models = []
        client_weights = []

        # Each selected client trains locally
        for client in selected_clients:
            # Download global model
            client.download_model(server.get_global_model())

            # Train locally
            loss = client.train(epochs=local_epochs)
            print(f"Client {client.client_id}: loss = {loss:.4f}")

            # Upload model and data size
            client_models.append(client.upload_model())
            client_weights.append(client.get_data_size())

        # Server aggregates updates
        server.aggregate(client_models, client_weights)

# Example usage
global_model = SimpleModel()
server = FederatedServer(global_model)

# Create clients with different data distributions (simulated)
clients = []
for i in range(10):
    # Each client has different amount of data (non-IID simulation)
    n_samples = torch.randint(50, 200, (1,)).item()
    data = torch.randn(n_samples, 784)
    labels = torch.randint(0, 10, (n_samples,))
    clients.append(FederatedClient(i, data, labels))

# Run federated learning
federated_learning(server, clients, num_rounds=5, clients_per_round=3, local_epochs=2)`,
        explanation: 'Basic federated learning implementation with FedAvg algorithm, showing client-server architecture.'
      },
      {
        language: 'Python',
        code: `import torch
import torch.nn as nn

# Differential Privacy in Federated Learning

class DPFederatedClient:
    def __init__(self, client_id, train_data, train_labels, epsilon=1.0, delta=1e-5):
        self.client_id = client_id
        self.train_data = train_data
        self.train_labels = train_labels
        self.model = None
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta
        self.clip_norm = 1.0  # Gradient clipping threshold

    def download_model(self, global_model):
        self.model = copy.deepcopy(global_model)

    def clip_gradients(self):
        """Clip gradients to bound sensitivity"""
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.clip_norm
        )
        return total_norm

    def add_noise_to_gradients(self, noise_scale):
        """Add Gaussian noise to gradients for differential privacy"""
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * noise_scale
                    param.grad += noise

    def train_with_dp(self, epochs=1, lr=0.01):
        """Train with differential privacy"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        # Compute noise scale based on privacy budget
        sensitivity = self.clip_norm
        noise_scale = (sensitivity / self.epsilon) * torch.sqrt(
            torch.tensor(2.0 * torch.log(torch.tensor(1.25 / self.delta)))
        )

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(self.train_data)
            loss = criterion(outputs, self.train_labels)
            loss.backward()

            # Clip gradients
            self.clip_gradients()

            # Add noise
            self.add_noise_to_gradients(noise_scale)

            optimizer.step()

        return loss.item()

# Secure Aggregation (simplified simulation)
class SecureAggregationServer:
    def __init__(self, model):
        self.global_model = model

    def secure_aggregate(self, encrypted_models, client_weights):
        """
        Simplified secure aggregation
        In practice, uses cryptographic protocols like homomorphic encryption
        """
        # Here we just aggregate, but imagine each model is encrypted
        # and we can only see the aggregate

        total_weight = sum(client_weights)
        weights = [w / total_weight for w in client_weights]

        global_dict = self.global_model.state_dict()

        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
            for encrypted_dict, weight in zip(encrypted_models, weights):
                # In real secure aggregation, this operation happens
                # on encrypted data using homomorphic encryption
                global_dict[key] += weight * encrypted_dict[key].float()

        self.global_model.load_state_dict(global_dict)
        return self.global_model

# Personalized Federated Learning
class PersonalizedFLClient:
    def __init__(self, client_id, train_data, train_labels):
        self.client_id = client_id
        self.train_data = train_data
        self.train_labels = train_labels
        self.global_model = None
        self.personal_model = None

    def download_and_personalize(self, global_model, personal_epochs=5):
        """
        Download global model and personalize on local data
        """
        # Start with global model
        self.personal_model = copy.deepcopy(global_model)

        # Fine-tune on local data (only last layer)
        criterion = nn.CrossEntropyLoss()

        # Freeze all layers except last
        for param in self.personal_model.parameters():
            param.requires_grad = False
        for param in self.personal_model.fc2.parameters():
            param.requires_grad = True

        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.personal_model.parameters()),
            lr=0.01
        )

        self.personal_model.train()
        for epoch in range(personal_epochs):
            optimizer.zero_grad()
            outputs = self.personal_model(self.train_data)
            loss = criterion(outputs, self.train_labels)
            loss.backward()
            optimizer.step()

        print(f"Client {self.client_id} personalized model, loss: {loss.item():.4f}")

# Example: Differential Privacy FL
global_model = SimpleModel()
dp_clients = [
    DPFederatedClient(i, torch.randn(100, 784), torch.randint(0, 10, (100,)), epsilon=1.0)
    for i in range(5)
]

for client in dp_clients:
    client.download_model(global_model)
    loss = client.train_with_dp(epochs=1)
    print(f"DP Client {client.client_id} loss: {loss:.4f}")`,
        explanation: 'Differential privacy in federated learning and personalized FL, showing privacy-preserving techniques.'
      }
    ],
    interviewQuestions: [
      {
        question: 'What problem does federated learning solve?',
        answer: `Federated learning enables training ML models across decentralized data without centralizing raw data. Solves: (1) Privacy concerns - data remains on client devices, (2) Regulatory compliance - GDPR, healthcare regulations, (3) Communication costs - sending models vs. data, (4) Data sovereignty - organizations keep control over their data. Applications include mobile keyboard prediction, healthcare analytics, financial fraud detection where data sharing is prohibited or impractical.`
      },
      {
        question: 'Explain the FedAvg algorithm.',
        answer: `FedAvg (Federated Averaging) is the foundational federated learning algorithm: (1) Server broadcasts global model to clients, (2) Each client trains locally for E epochs on local data, (3) Clients send model updates (parameters) to server, (4) Server aggregates updates using weighted averaging based on data size, (5) Process repeats. Benefits: communication efficient, simple implementation. Challenges: non-IID data across clients, varying computational capabilities, client dropouts during training.`
      },
      {
        question: 'What challenges arise from non-IID data in federated learning?',
        answer: `Non-IID (independent and identically distributed) data occurs when clients have different data distributions. Challenges: (1) Model divergence - local updates may conflict, (2) Slower convergence, (3) Reduced final accuracy, (4) Client drift - models become specialized to local data. Solutions: (1) Data sharing (privacy-preserving), (2) Personalization techniques, (3) FedProx algorithm with proximal term, (4) Scaffold algorithm correcting for client drift, (5) Clustered federated learning grouping similar clients.`
      },
      {
        question: 'How does differential privacy work in federated learning?',
        answer: `Differential privacy adds calibrated noise to protect individual privacy while preserving aggregate patterns. In federated learning: (1) Local differential privacy - clients add noise before sending updates, (2) Central differential privacy - server adds noise during aggregation. Privacy budget (ε, δ) parameters control privacy-utility trade-off. Techniques include Gaussian noise addition, gradient clipping, and privacy accounting across multiple rounds. Stronger privacy (lower ε) reduces model accuracy but provides formal privacy guarantees.`
      },
      {
        question: 'What is secure aggregation and why is it important?',
        answer: `Secure aggregation allows the server to compute aggregate statistics (sum/average) of client updates without seeing individual contributions. Uses cryptographic techniques like homomorphic encryption or secure multi-party computation. Benefits: (1) Protects against honest-but-curious servers, (2) Prevents inference attacks on individual updates, (3) Enables stronger privacy guarantees. Trade-offs: increased computational and communication overhead. Essential for sensitive applications where even encrypted model updates could leak information.`
      },
      {
        question: 'Compare cross-device vs cross-silo federated learning.',
        answer: `Cross-device: Millions of clients (phones, IoT devices) with limited data each. Characteristics: high client churn, limited communication windows, heterogeneous hardware. Use cases: mobile keyboards, recommendation systems. Cross-silo: Fewer participants (organizations, hospitals) with substantial datasets. Characteristics: stable participation, better connectivity, similar hardware. Use cases: healthcare collaborations, financial consortiums. Different algorithms and communication strategies needed for each setting.`
      }
    ],
    quizQuestions: [
      {
        id: 'fl1',
        question: 'What is the main advantage of federated learning?',
        options: ['Faster training', 'Data privacy - data stays local', 'Better accuracy', 'Smaller models'],
        correctAnswer: 1,
        explanation: 'Federated learning enables collaborative training without sharing raw data. Data remains on local devices, preserving privacy while still benefiting from collective learning.'
      },
      {
        id: 'fl2',
        question: 'In FedAvg, how are client models combined?',
        options: ['Take best model', 'Weighted average', 'Concatenate', 'Use last client'],
        correctAnswer: 1,
        explanation: 'FedAvg (Federated Averaging) combines client models using a weighted average, where weights are typically proportional to the amount of data each client has.'
      },
      {
        id: 'fl3',
        question: 'What is a challenge unique to federated learning?',
        options: ['Overfitting', 'Non-IID data distribution', 'Vanishing gradients', 'Mode collapse'],
        correctAnswer: 1,
        explanation: 'Non-IID (non-independent and identically distributed) data is a key challenge in FL. Different clients have different data distributions, making convergence slower and more difficult.'
      }
    ]
  },

  'few-shot-learning': {
    id: 'few-shot-learning',
    title: 'Few-Shot Learning',
    category: 'advanced',
    description: 'Learning from very limited labeled examples',
    content: `
      <h2>Few-Shot Learning</h2>
      <p>Few-Shot Learning (FSL) enables models to generalize from a small number of labeled examples. This is crucial when labeled data is expensive, scarce, or impractical to collect at scale.</p>

      <h3>Problem Setting</h3>
      <ul>
        <li><strong>N-way K-shot:</strong> N classes with K examples each</li>
        <li><strong>Support set:</strong> Few labeled examples for learning</li>
        <li><strong>Query set:</strong> Test examples to classify</li>
        <li><strong>Zero-shot:</strong> K=0, learn from descriptions only</li>
        <li><strong>One-shot:</strong> K=1, one example per class</li>
        <li><strong>Few-shot:</strong> K=2-10, few examples per class</li>
      </ul>

      <h3>Main Approaches</h3>

      <h4>1. Metric Learning</h4>
      <p>Learn an embedding space where similar examples are close</p>

      <h5>Siamese Networks</h5>
      <ul>
        <li><strong>Twin networks:</strong> Shared weights, process pairs</li>
        <li><strong>Contrastive loss:</strong> Pull similar pairs together, push different apart</li>
        <li><strong>At test time:</strong> Compare query to support examples</li>
        <li><strong>Distance metric:</strong> Euclidean, cosine similarity</li>
      </ul>

      <h5>Prototypical Networks</h5>
      <ul>
        <li><strong>Class prototypes:</strong> Mean of support embeddings per class</li>
        <li><strong>Classify:</strong> Assign to nearest prototype</li>
        <li><strong>Simple and effective:</strong> Strong baseline</li>
        <li><strong>Episodic training:</strong> Sample N-way K-shot tasks</li>
      </ul>

      <h5>Matching Networks</h5>
      <ul>
        <li><strong>Attention mechanism:</strong> Weight support examples</li>
        <li><strong>Full context embeddings:</strong> Use all support set</li>
        <li><strong>Differentiable k-NN:</strong> Soft nearest neighbor</li>
      </ul>

      <h5>Relation Networks</h5>
      <ul>
        <li><strong>Learn similarity:</strong> Instead of using fixed metric</li>
        <li><strong>Relation module:</strong> Neural network for comparison</li>
        <li><strong>More flexible:</strong> Than hand-crafted metrics</li>
      </ul>

      <h4>2. Meta-Learning (Learning to Learn)</h4>
      <p>Learn how to quickly adapt to new tasks</p>

      <h5>MAML (Model-Agnostic Meta-Learning)</h5>
      <ul>
        <li><strong>Learn initialization:</strong> Good starting point for fine-tuning</li>
        <li><strong>Inner loop:</strong> Task-specific adaptation</li>
        <li><strong>Outer loop:</strong> Meta-optimization across tasks</li>
        <li><strong>Few gradient steps:</strong> Quickly adapt</li>
        <li><strong>Second-order:</strong> Gradient through gradient</li>
      </ul>

      <h4>3. Data Augmentation</h4>
      <ul>
        <li><strong>Hallucination:</strong> Generate synthetic examples</li>
        <li><strong>Mixup:</strong> Interpolate between examples</li>
        <li><strong>Transfer:</strong> Pre-train on related large dataset</li>
      </ul>

      <h4>4. Embedding Propagation</h4>
      <ul>
        <li><strong>Graph neural networks:</strong> Propagate labels through graph</li>
        <li><strong>Semi-supervised:</strong> Leverage unlabeled data</li>
        <li><strong>Transductive:</strong> Use query set structure</li>
      </ul>

      <h3>Training Strategy: Episodic Learning</h3>
      <p>Simulate few-shot scenarios during training:</p>
      <ul>
        <li><strong>Episode:</strong> Sample N classes, K examples each</li>
        <li><strong>Support set:</strong> K examples per class for "training"</li>
        <li><strong>Query set:</strong> Separate examples for evaluation</li>
        <li><strong>Meta-train:</strong> Many episodes from training classes</li>
        <li><strong>Meta-test:</strong> Episodes from held-out test classes</li>
      </ul>

      <h3>Key Challenges</h3>
      <ul>
        <li><strong>Overfitting:</strong> Easy with very few examples</li>
        <li><strong>Domain shift:</strong> Training vs test classes may differ</li>
        <li><strong>Intra-class variation:</strong> Hard to capture from few samples</li>
        <li><strong>Computational cost:</strong> Meta-learning can be expensive</li>
      </ul>

      <h3>Relation to Transfer Learning</h3>
      <ul>
        <li><strong>Pre-training:</strong> Learn good features on large dataset</li>
        <li><strong>Fine-tuning:</strong> Adapt to target task with few examples</li>
        <li><strong>Frozen features:</strong> Use pre-trained as feature extractor</li>
        <li><strong>Linear probing:</strong> Train only final layer</li>
      </ul>

      <h3>Applications</h3>
      <ul>
        <li><strong>Drug discovery:</strong> Predict properties with few molecules</li>
        <li><strong>Rare disease diagnosis:</strong> Limited patient data</li>
        <li><strong>Personalization:</strong> Adapt to individual users</li>
        <li><strong>Robotics:</strong> Learn new tasks quickly</li>
        <li><strong>Object recognition:</strong> Recognize new objects from few examples</li>
        <li><strong>Low-resource languages:</strong> NLP with limited data</li>
      </ul>

      <h3>Evaluation</h3>
      <ul>
        <li><strong>Benchmark:</strong> miniImageNet, tieredImageNet, Omniglot</li>
        <li><strong>N-way K-shot accuracy:</strong> Standard metric</li>
        <li><strong>Multiple episodes:</strong> Report mean and confidence intervals</li>
        <li><strong>Different K:</strong> Test with varying support sizes</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `import torch
import torch.nn as nn
import torch.nn.functional as F

# Prototypical Networks

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class PrototypicalNetwork(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        # Feature extractor
        self.encoder = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 64),
            ConvBlock(64, embedding_dim)
        )

    def forward(self, x):
        return self.encoder(x).view(x.size(0), -1)

def compute_prototypes(support_embeddings, support_labels, n_classes):
    """
    Compute class prototypes (mean of support embeddings per class)

    Args:
        support_embeddings: [n_support, embedding_dim]
        support_labels: [n_support]
        n_classes: Number of classes

    Returns:
        prototypes: [n_classes, embedding_dim]
    """
    prototypes = []
    for c in range(n_classes):
        mask = (support_labels == c)
        class_embeddings = support_embeddings[mask]
        prototype = class_embeddings.mean(dim=0)
        prototypes.append(prototype)

    return torch.stack(prototypes)

def prototypical_loss(query_embeddings, prototypes, query_labels):
    """
    Compute prototypical loss (negative log probability)

    Args:
        query_embeddings: [n_query, embedding_dim]
        prototypes: [n_classes, embedding_dim]
        query_labels: [n_query]

    Returns:
        loss: scalar
    """
    # Compute distances to all prototypes
    dists = torch.cdist(query_embeddings, prototypes)  # [n_query, n_classes]

    # Convert to log probabilities (softmax over negative distances)
    log_p_y = F.log_softmax(-dists, dim=1)

    # Negative log likelihood loss
    loss = F.nll_loss(log_p_y, query_labels)

    return loss

# Example: 5-way 5-shot task
n_way = 5
k_shot = 5
n_query = 15

model = PrototypicalNetwork(embedding_dim=64)

# Support set: 5 classes × 5 examples = 25 images
support_images = torch.randn(n_way * k_shot, 3, 28, 28)
support_labels = torch.arange(n_way).repeat_interleave(k_shot)

# Query set: 15 images to classify
query_images = torch.randn(n_query, 3, 28, 28)
query_labels = torch.randint(0, n_way, (n_query,))

# Get embeddings
support_embeddings = model(support_images)
query_embeddings = model(query_images)

# Compute prototypes
prototypes = compute_prototypes(support_embeddings, support_labels, n_way)

# Compute loss
loss = prototypical_loss(query_embeddings, prototypes, query_labels)

# Predictions
dists = torch.cdist(query_embeddings, prototypes)
predictions = dists.argmin(dim=1)
accuracy = (predictions == query_labels).float().mean()

print(f"Loss: {loss.item():.4f}")
print(f"Accuracy: {accuracy.item():.4f}")`,
        explanation: 'Prototypical Networks implementation for few-shot learning using class prototypes.'
      },
      {
        language: 'Python',
        code: `import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

# MAML (Model-Agnostic Meta-Learning)

class MAMLModel(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, params=None):
        if params is None:
            return self.model(x)
        else:
            # Forward pass with custom parameters
            x = F.linear(x, params['model.0.weight'], params['model.0.bias'])
            x = F.relu(x)
            x = F.linear(x, params['model.2.weight'], params['model.2.bias'])
            x = F.relu(x)
            x = F.linear(x, params['model.4.weight'], params['model.4.bias'])
            return x

class MAML:
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001, inner_steps=5):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)
        self.inner_steps = inner_steps

    def inner_loop(self, support_x, support_y):
        """
        Adapt model to a specific task (inner loop)

        Args:
            support_x: Support set inputs
            support_y: Support set labels

        Returns:
            adapted_params: Task-specific parameters
        """
        # Clone current parameters
        params = OrderedDict(self.model.named_parameters())
        adapted_params = OrderedDict()
        for name, param in params.items():
            adapted_params[name] = param.clone()

        # Inner loop: gradient descent on support set
        for step in range(self.inner_steps):
            # Forward pass with current params
            logits = self.model(support_x)
            loss = F.cross_entropy(logits, support_y)

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=True  # For second-order gradients
            )

            # Update parameters
            adapted_params = OrderedDict()
            for (name, param), grad in zip(params.items(), grads):
                adapted_params[name] = param - self.inner_lr * grad

        return adapted_params

    def meta_train_step(self, tasks):
        """
        Meta-training step (outer loop)

        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples

        Returns:
            meta_loss: Meta-training loss
        """
        self.meta_optimizer.zero_grad()
        meta_loss = 0

        for support_x, support_y, query_x, query_y in tasks:
            # Inner loop: adapt to task
            adapted_params = self.inner_loop(support_x, support_y)

            # Outer loop: evaluate on query set with adapted params
            query_logits = self.model(query_x, adapted_params)
            task_loss = F.cross_entropy(query_logits, query_y)

            meta_loss += task_loss

        # Meta-update
        meta_loss = meta_loss / len(tasks)
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

# Example usage
input_dim = 784
n_way = 5
k_shot = 5
n_query = 15

model = MAMLModel(input_dim=input_dim, output_dim=n_way)
maml = MAML(model, inner_lr=0.01, meta_lr=0.001, inner_steps=5)

# Simulate batch of tasks
batch_size = 4
tasks = []
for _ in range(batch_size):
    # Support set: 5-way 5-shot
    support_x = torch.randn(n_way * k_shot, input_dim)
    support_y = torch.arange(n_way).repeat_interleave(k_shot)

    # Query set
    query_x = torch.randn(n_query, input_dim)
    query_y = torch.randint(0, n_way, (n_query,))

    tasks.append((support_x, support_y, query_x, query_y))

# Meta-training step
meta_loss = maml.meta_train_step(tasks)
print(f"Meta-training loss: {meta_loss:.4f}")

# At test time: adapt to new task
test_support_x = torch.randn(n_way * k_shot, input_dim)
test_support_y = torch.arange(n_way).repeat_interleave(k_shot)

adapted_params = maml.inner_loop(test_support_x, test_support_y)

# Test on query set
test_query_x = torch.randn(n_query, input_dim)
test_query_y = torch.randint(0, n_way, (n_query,))

with torch.no_grad():
    logits = maml.model(test_query_x, adapted_params)
    predictions = logits.argmax(dim=1)
    accuracy = (predictions == test_query_y).float().mean()

print(f"Test accuracy: {accuracy.item():.4f}")`,
        explanation: 'MAML implementation showing meta-learning with inner loop adaptation and outer loop meta-optimization.'
      }
    ],
    interviewQuestions: [
      {
        question: 'What is the difference between few-shot learning and transfer learning?',
        answer: `Transfer learning pre-trains on large datasets then fine-tunes on target task with sufficient labeled data. Few-shot learning specifically handles scenarios with very few examples (1-10) per class. Key differences: (1) Data requirements - transfer learning needs moderate target data, few-shot needs minimal, (2) Adaptation strategy - transfer learning fine-tunes, few-shot often meta-learns, (3) Objective - transfer learning leverages related knowledge, few-shot learns to learn quickly from limited examples.`
      },
      {
        question: 'Explain how Prototypical Networks work.',
        answer: `Prototypical Networks create prototype representations for each class by averaging support set embeddings. Process: (1) Embed support and query examples using shared network, (2) Compute class prototypes as mean of support embeddings per class, (3) Classify queries based on distance to nearest prototype (typically Euclidean). Benefits: simple, interpretable, works well empirically. The embedding space is learned to make same-class examples cluster and different-class examples separate, enabling effective nearest-prototype classification.`
      },
      {
        question: 'What is the key idea behind MAML?',
        answer: `MAML (Model-Agnostic Meta-Learning) learns initialization parameters that enable fast adaptation to new tasks with few gradient steps. Two-level optimization: (1) Inner loop - adapt to specific task using few examples, (2) Outer loop - optimize initialization for good post-adaptation performance across tasks. Key insight: some initializations are better starting points for learning new tasks. MAML is model-agnostic (works with any gradient-based model) and learns representations that are easy to fine-tune rather than task-specific.`
      },
      {
        question: 'What is episodic training in few-shot learning?',
        answer: `Episodic training simulates few-shot scenarios during training by creating artificial episodes that mimic test conditions. Each episode: (1) Sample N classes from training set, (2) Sample K examples per class (support set), (3) Sample query examples to classify, (4) Train to classify queries using only support examples. This forces the model to learn from limited examples repeatedly, developing meta-learning capabilities. Ensures training and testing conditions match, leading to better few-shot performance.`
      },
      {
        question: 'How does metric learning help in few-shot scenarios?',
        answer: `Metric learning learns embedding spaces where similar examples are close and dissimilar ones are far apart. In few-shot learning: (1) Embeds support and query examples into learned space, (2) Uses distance-based classification (nearest neighbor, prototype), (3) Works well with limited data as it leverages similarity rather than complex decision boundaries. Benefits: interpretable, requires fewer parameters to learn, naturally handles new classes. Common losses: triplet loss, contrastive loss, prototypical loss.`
      },
      {
        question: 'What are the trade-offs between metric learning and meta-learning approaches?',
        answer: `Metric learning: Simpler to implement, more interpretable, faster inference, but may be limited by embedding quality and distance metric choice. Works well when good similarity measures exist. Meta-learning: More flexible, can adapt entire model behavior, potentially better performance, but more complex training, slower adaptation, risk of overfitting to meta-training distribution. Choose metric learning for simplicity and interpretability, meta-learning for maximum flexibility and performance when sufficient meta-training data available.`
      }
    ],
    quizQuestions: [
      {
        id: 'fsl1',
        question: 'What does "5-way 1-shot" mean in few-shot learning?',
        options: ['5 examples per class', '5 classes with 1 example each', '1 class with 5 examples', '5 training epochs'],
        correctAnswer: 1,
        explanation: 'N-way K-shot refers to a classification task with N classes and K examples per class. So 5-way 1-shot means 5 classes with 1 labeled example each.'
      },
      {
        id: 'fsl2',
        question: 'What is the main idea of Prototypical Networks?',
        options: ['Train on prototypes', 'Classify by distance to class prototypes', 'Generate prototypes', 'Prune prototypes'],
        correctAnswer: 1,
        explanation: 'Prototypical Networks compute a prototype (mean embedding) for each class from the support set, then classify queries by assigning them to the nearest prototype.'
      },
      {
        id: 'fsl3',
        question: 'What does MAML learn?',
        options: ['Final model weights', 'Good initialization for fast adaptation', 'Distance metric', 'Data augmentation'],
        correctAnswer: 1,
        explanation: 'MAML (Model-Agnostic Meta-Learning) learns an initialization that allows rapid adaptation to new tasks with just a few gradient steps.'
      }
    ]
  },

  'multi-modal-models': {
    id: 'multi-modal-models',
    title: 'Multi-Modal Models',
    category: 'advanced',
    description: 'Models that process and integrate multiple types of data',
    content: `
      <h2>Multi-Modal Models</h2>
      <p>Multi-modal models process and integrate information from multiple modalities (vision, language, audio, etc.) to perform tasks that require understanding across different types of data. They enable richer representations and more versatile AI systems.</p>

      <h3>Common Modalities</h3>
      <ul>
        <li><strong>Vision:</strong> Images, videos</li>
        <li><strong>Language:</strong> Text, speech transcriptions</li>
        <li><strong>Audio:</strong> Speech, sounds, music</li>
        <li><strong>Sensor data:</strong> LiDAR, depth, thermal</li>
        <li><strong>Structured data:</strong> Tables, graphs, time series</li>
      </ul>

      <h3>Key Challenges</h3>
      <ul>
        <li><strong>Representation:</strong> Different modalities have different structures</li>
        <li><strong>Alignment:</strong> Correspond elements across modalities</li>
        <li><strong>Fusion:</strong> Combine information effectively</li>
        <li><strong>Missing modalities:</strong> Handle incomplete inputs</li>
        <li><strong>Heterogeneity:</strong> Different scales, distributions, semantics</li>
      </ul>

      <h3>Fusion Strategies</h3>

      <h4>Early Fusion</h4>
      <ul>
        <li><strong>Concatenate raw:</strong> Combine modalities at input</li>
        <li><strong>Joint processing:</strong> Single model processes all</li>
        <li><strong>Advantages:</strong> Maximum interaction between modalities</li>
        <li><strong>Disadvantages:</strong> High dimensionality, less flexible</li>
      </ul>

      <h4>Late Fusion</h4>
      <ul>
        <li><strong>Separate processing:</strong> Each modality processed independently</li>
        <li><strong>Combine decisions:</strong> Merge at prediction level</li>
        <li><strong>Advantages:</strong> Modular, easier to train</li>
        <li><strong>Disadvantages:</strong> Limited cross-modal interaction</li>
      </ul>

      <h4>Hybrid Fusion</h4>
      <ul>
        <li><strong>Multi-stage:</strong> Fusion at multiple levels</li>
        <li><strong>Cross-attention:</strong> Attend across modalities</li>
        <li><strong>Advantages:</strong> Balance between early and late</li>
        <li><strong>Examples:</strong> Most modern multi-modal Transformers</li>
      </ul>

      <h3>Notable Multi-Modal Models</h3>

      <h4>CLIP (Contrastive Language-Image Pre-training)</h4>
      <ul>
        <li><strong>Approach:</strong> Contrastive learning on image-text pairs</li>
        <li><strong>Architecture:</strong> Separate image and text encoders</li>
        <li><strong>Training:</strong> Maximize similarity of matched pairs</li>
        <li><strong>Zero-shot:</strong> Classify images with text prompts</li>
        <li><strong>Applications:</strong> Image search, classification, generation guidance</li>
      </ul>

      <h4>DALL-E / Stable Diffusion</h4>
      <ul>
        <li><strong>Task:</strong> Text-to-image generation</li>
        <li><strong>Architecture:</strong> Text encoder + image decoder</li>
        <li><strong>DALL-E:</strong> Transformer-based, discrete VAE</li>
        <li><strong>Stable Diffusion:</strong> Latent diffusion model</li>
        <li><strong>Applications:</strong> Art generation, design, content creation</li>
      </ul>

      <h4>Flamingo</h4>
      <ul>
        <li><strong>Visual language model:</strong> Images + text → text</li>
        <li><strong>Few-shot:</strong> Learn from few image-text examples</li>
        <li><strong>Perceiver Resampler:</strong> Process variable images</li>
        <li><strong>Interleaved input:</strong> Mix images and text</li>
      </ul>

      <h4>Whisper</h4>
      <ul>
        <li><strong>Speech-to-text:</strong> Audio → transcription</li>
        <li><strong>Multi-lingual:</strong> 99 languages</li>
        <li><strong>Multi-task:</strong> Transcription, translation, language ID</li>
        <li><strong>Encoder-decoder:</strong> Transformer architecture</li>
      </ul>

      <h4>GPT-4 Vision / GPT-4V</h4>
      <ul>
        <li><strong>Vision-language:</strong> Images + text → text</li>
        <li><strong>Multi-modal reasoning:</strong> Complex visual understanding</li>
        <li><strong>Various tasks:</strong> VQA, OCR, visual reasoning</li>
      </ul>

      <h3>Training Techniques</h3>

      <h4>Contrastive Learning</h4>
      <ul>
        <li><strong>Positive pairs:</strong> Matched cross-modal samples</li>
        <li><strong>Negative pairs:</strong> Unmatched samples</li>
        <li><strong>Pull together, push apart:</strong> In embedding space</li>
        <li><strong>InfoNCE loss:</strong> Common objective</li>
      </ul>

      <h4>Masked Modeling</h4>
      <ul>
        <li><strong>Mask tokens:</strong> In one or both modalities</li>
        <li><strong>Predict masked:</strong> Using other modality</li>
        <li><strong>Examples:</strong> MAE for vision, masked language modeling</li>
      </ul>

      <h4>Alignment Objectives</h4>
      <ul>
        <li><strong>Image-text matching:</strong> Binary classification</li>
        <li><strong>Image-text retrieval:</strong> Find corresponding pairs</li>
        <li><strong>Captioning:</strong> Generate text from images</li>
      </ul>

      <h3>Applications</h3>

      <h4>Vision + Language</h4>
      <ul>
        <li><strong>Visual Question Answering:</strong> Answer questions about images</li>
        <li><strong>Image captioning:</strong> Generate descriptions</li>
        <li><strong>Visual reasoning:</strong> Multi-step inference</li>
        <li><strong>Text-to-image:</strong> Generate images from descriptions</li>
        <li><strong>Image-text retrieval:</strong> Search images with text</li>
      </ul>

      <h4>Audio + Language</h4>
      <ul>
        <li><strong>Speech recognition:</strong> Audio → text</li>
        <li><strong>Text-to-speech:</strong> Text → audio</li>
        <li><strong>Audio captioning:</strong> Describe sounds</li>
        <li><strong>Music generation:</strong> From text descriptions</li>
      </ul>

      <h4>Multi-Modal Fusion</h4>
      <ul>
        <li><strong>Autonomous driving:</strong> Camera, LiDAR, radar</li>
        <li><strong>Robotics:</strong> Vision, proprioception, language</li>
        <li><strong>Healthcare:</strong> Medical images + clinical notes</li>
        <li><strong>Video understanding:</strong> Visual + audio + subtitles</li>
      </ul>

      <h3>Evaluation</h3>
      <ul>
        <li><strong>Retrieval:</strong> Recall@K, mean rank</li>
        <li><strong>Generation:</strong> FID, CLIP score, human evaluation</li>
        <li><strong>VQA:</strong> Accuracy on question answering</li>
        <li><strong>Zero-shot transfer:</strong> Performance on unseen tasks</li>
      </ul>
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
  }
};