import { QuizQuestion } from '../../types';

// GANs - 20 questions
export const ganQuestions: QuizQuestion[] = [
  {
    id: 'gan1',
    question: 'What does GAN stand for?',
    options: ['General Adversarial Network', 'Generative Adversarial Network', 'Gradient Analysis Network', 'Generic AI Network'],
    correctAnswer: 1,
    explanation: 'GAN = Generative Adversarial Network, where two networks compete: generator creates, discriminator judges.'
  },
  {
    id: 'gan2',
    question: 'What are the two components of a GAN?',
    options: ['Encoder-decoder', 'Generator (creates fake data) and Discriminator (judges real vs fake)', 'Two encoders', 'Two decoders'],
    correctAnswer: 1,
    explanation: 'Generator learns to create realistic samples; discriminator learns to distinguish real from generated.'
  },
  {
    id: 'gan3',
    question: 'What is the training objective?',
    options: ['Cooperation', 'Adversarial: generator tries to fool discriminator, discriminator tries to detect fakes', 'Supervised', 'Unsupervised clustering'],
    correctAnswer: 1,
    explanation: 'Min-max game: generator minimizes discriminator\'s ability to detect fakes; discriminator maximizes detection accuracy.'
  },
  {
    id: 'gan4',
    question: 'What is the generator input?',
    options: ['Real images', 'Random noise (latent vector z)', 'Labels', 'Nothing'],
    correctAnswer: 1,
    explanation: 'Generator takes random noise (latent code) and transforms it into realistic samples.'
  },
  {
    id: 'gan5',
    question: 'What is the discriminator output?',
    options: ['Image', 'Probability that input is real (0-1)', 'Class label', 'Latent code'],
    correctAnswer: 1,
    explanation: 'Discriminator outputs probability that input came from real data distribution vs generator.'
  },
  {
    id: 'gan6',
    question: 'What is mode collapse?',
    options: ['Good behavior', 'Generator producing limited variety, mapping many z to same output', 'Perfect diversity', 'Discriminator failure'],
    correctAnswer: 1,
    explanation: 'Mode collapse: generator finds one/few modes that fool discriminator, losing diversity.'
  },
  {
    id: 'gan7',
    question: 'What causes training instability in GANs?',
    options: ['Stability is easy', 'Adversarial dynamics, gradient imbalance, sensitive hyperparameters', 'Perfect stability', 'No issues'],
    correctAnswer: 1,
    explanation: 'GANs notoriously unstable: oscillation, vanishing gradients, mode collapse require careful tuning.'
  },
  {
    id: 'gan8',
    question: 'What is the Nash equilibrium in GAN training?',
    options: ['Generator wins', 'Ideal state where generator creates perfect fakes, discriminator outputs 0.5', 'Discriminator wins', 'No equilibrium'],
    correctAnswer: 1,
    explanation: 'Nash equilibrium: generator distribution matches real data, discriminator can\'t distinguish (50% accuracy).'
  },
  {
    id: 'gan9',
    question: 'What is conditional GAN (cGAN)?',
    options: ['Unconditional only', 'GAN conditioned on additional information (labels, text)', 'No conditioning', 'Same as GAN'],
    correctAnswer: 1,
    explanation: 'cGAN provides labels/conditions to both generator and discriminator, enabling controlled generation.'
  },
  {
    id: 'gan10',
    question: 'What is DCGAN?',
    options: ['Fully connected GAN', 'Deep Convolutional GAN: uses CNNs with specific architecture guidelines', 'RNN GAN', 'No convolutions'],
    correctAnswer: 1,
    explanation: 'DCGAN introduced stable architecture: strided convolutions, BatchNorm, LeakyReLU, no FC layers.'
  },
  {
    id: 'gan11',
    question: 'What is StyleGAN?',
    options: ['Basic GAN', 'GAN with style-based architecture for high-quality controllable image generation', 'Text GAN', 'No style'],
    correctAnswer: 1,
    explanation: 'StyleGAN (NVIDIA) uses adaptive instance normalization for unprecedented control over generated image styles.'
  },
  {
    id: 'gan12',
    question: 'What is CycleGAN?',
    options: ['Paired data', 'Unpaired image-to-image translation using cycle consistency', 'Supervised', 'Classification'],
    correctAnswer: 1,
    explanation: 'CycleGAN learns translations (e.g., horses↔zebras) without paired examples using cycle consistency loss.'
  },
  {
    id: 'gan13',
    question: 'What is Pix2Pix?',
    options: ['Unpaired', 'Paired image-to-image translation with conditional GAN', 'No pairs', 'Classification'],
    correctAnswer: 1,
    explanation: 'Pix2Pix uses paired training data (e.g., sketch-photo pairs) with cGAN and L1 loss.'
  },
  {
    id: 'gan14',
    question: 'What is Wasserstein GAN (WGAN)?',
    options: ['Same loss', 'Uses Wasserstein distance for more stable training', 'Less stable', 'No change'],
    correctAnswer: 1,
    explanation: 'WGAN uses Earth Mover\'s Distance, providing meaningful loss and improved training stability.'
  },
  {
    id: 'gan15',
    question: 'What is Progressive GAN?',
    options: ['Single resolution', 'Trains generator/discriminator progressively from low to high resolution', 'High res only', 'No progression'],
    correctAnswer: 1,
    explanation: 'Progressive GAN starts at 4×4, gradually adding layers to reach 1024×1024, improving stability and quality.'
  },
  {
    id: 'gan16',
    question: 'What applications do GANs have?',
    options: ['None', 'Image generation, super-resolution, style transfer, data augmentation, deepfakes', 'Only classification', 'No applications'],
    correctAnswer: 1,
    explanation: 'GANs used for: photorealistic synthesis, image editing, domain adaptation, creating training data.'
  },
  {
    id: 'gan17',
    question: 'What is GAN evaluation challenge?',
    options: ['Easy to evaluate', 'No single metric; use Inception Score, FID, human evaluation', 'Perfect metrics', 'Accuracy only'],
    correctAnswer: 1,
    explanation: 'GAN quality hard to measure; Inception Score (IS) and Fréchet Inception Distance (FID) commonly used.'
  },
  {
    id: 'gan18',
    question: 'What is FID?',
    options: ['Accuracy', 'Fréchet Inception Distance: measures distance between real and generated feature distributions', 'Loss function', 'No metric'],
    correctAnswer: 1,
    explanation: 'FID computes distance between Inception features of real and fake images; lower is better.'
  },
  {
    id: 'gan19',
    question: 'How do GANs compare to diffusion models?',
    options: ['GANs always better', 'Diffusion models now achieve better quality but slower; GANs faster inference', 'Same quality', 'No difference'],
    correctAnswer: 1,
    explanation: 'Diffusion models (DALL-E 2, Stable Diffusion) surpassed GAN quality but GANs remain faster at inference.'
  },
  {
    id: 'gan20',
    question: 'What ethical concerns do GANs raise?',
    options: ['No concerns', 'Deepfakes, misinformation, copyright, consent for training data', 'Perfectly safe', 'Only technical'],
    correctAnswer: 1,
    explanation: 'GANs enable deepfakes, raise questions about consent, copyright, and malicious use.'
  }
];

// VAEs - 20 questions
export const vaeQuestions: QuizQuestion[] = [
  {
    id: 'vae1',
    question: 'What does VAE stand for?',
    options: ['Variable Autoencoder', 'Variational Autoencoder', 'Vector Analysis Engine', 'Visual AI Encoder'],
    correctAnswer: 1,
    explanation: 'VAE = Variational Autoencoder, combining autoencoders with probabilistic latent variables.'
  },
  {
    id: 'vae2',
    question: 'What are the components of a VAE?',
    options: ['Generator-discriminator', 'Encoder (inference network) and Decoder (generative network)', 'Two discriminators', 'One network'],
    correctAnswer: 1,
    explanation: 'Encoder maps input to latent distribution; decoder reconstructs from latent samples.'
  },
  {
    id: 'vae3',
    question: 'What does the VAE encoder output?',
    options: ['Single vector', 'Parameters of latent distribution (mean μ and variance σ²)', 'Image', 'Class'],
    correctAnswer: 1,
    explanation: 'Encoder outputs μ and σ² defining Gaussian distribution in latent space, not deterministic code.'
  },
  {
    id: 'vae4',
    question: 'What is the reparameterization trick?',
    options: ['No trick needed', 'Sampling z = μ + σ ⊙ ε where ε ~ N(0,1) to allow backprop through sampling', 'Direct sampling', 'No sampling'],
    correctAnswer: 1,
    explanation: 'Reparameterization makes stochastic sampling differentiable by moving randomness to external ε.'
  },
  {
    id: 'vae5',
    question: 'What are the two loss terms in VAE?',
    options: ['Only reconstruction', 'Reconstruction loss + KL divergence regularization', 'Only KL', 'Cross-entropy only'],
    correctAnswer: 1,
    explanation: 'VAE loss = reconstruction (data likelihood) + KL divergence (regularization to prior).'
  },
  {
    id: 'vae6',
    question: 'What is the reconstruction loss?',
    options: ['KL divergence', 'Measures how well decoder reconstructs input (e.g., MSE or BCE)', 'Entropy', 'No loss'],
    correctAnswer: 1,
    explanation: 'Reconstruction loss ensures decoder generates outputs similar to inputs; typically MSE or binary cross-entropy.'
  },
  {
    id: 'vae7',
    question: 'What is the KL divergence term?',
    options: ['Reconstruction', 'Measures divergence between encoded distribution and prior N(0,I)', 'Entropy', 'No regularization'],
    correctAnswer: 1,
    explanation: 'KL term regularizes latent space to match standard normal prior, enabling sampling.'
  },
  {
    id: 'vae8',
    question: 'Why is KL regularization important?',
    options: ['Not important', 'Prevents overfitting, ensures smooth latent space, enables generation from prior', 'Only for loss', 'Slows training'],
    correctAnswer: 1,
    explanation: 'KL regularization ensures latent space is well-structured, allowing generation by sampling from N(0,I).'
  },
  {
    id: 'vae9',
    question: 'How do you generate new samples with VAE?',
    options: ['Use encoder', 'Sample z ~ N(0,I) and pass through decoder', 'Random input', 'Use discriminator'],
    correctAnswer: 1,
    explanation: 'Generation: sample random latent vector from standard normal, decode to output.'
  },
  {
    id: 'vae10',
    question: 'What is latent space interpolation?',
    options: ['Discrete jumps', 'Smooth transitions between points in latent space create meaningful variations', 'No interpolation', 'Random walk'],
    correctAnswer: 1,
    explanation: 'Interpolating between latent codes produces smooth semantic transitions (e.g., smile → no smile).'
  },
  {
    id: 'vae11',
    question: 'How do VAEs compare to GANs?',
    options: ['Same quality', 'VAEs easier to train, more stable, but often blurrier outputs than GANs', 'VAEs always better', 'No difference'],
    correctAnswer: 1,
    explanation: 'VAEs more stable training but produce blurrier images; GANs sharper but harder to train.'
  },
  {
    id: 'vae12',
    question: 'What causes blurriness in VAE outputs?',
    options: ['Perfect sharpness', 'Reconstruction loss (MSE) penalizes variance, averaging over modes', 'Too sharp', 'No blurriness'],
    correctAnswer: 1,
    explanation: 'MSE loss encourages averaging, producing blurry reconstructions rather than sharp samples.'
  },
  {
    id: 'vae13',
    question: 'What is β-VAE?',
    options: ['Standard VAE', 'VAE with weighted KL term (β·KL) for disentangled representations', 'No β', 'Same loss'],
    correctAnswer: 1,
    explanation: 'β-VAE increases weight of KL term, encouraging disentangled latent factors (e.g., separate pose/color).'
  },
  {
    id: 'vae14',
    question: 'What is a conditional VAE (CVAE)?',
    options: ['Unconditional', 'VAE conditioned on labels/attributes for controlled generation', 'No conditioning', 'Same as VAE'],
    correctAnswer: 1,
    explanation: 'CVAE conditions encoder and decoder on additional information, enabling attribute-specific generation.'
  },
  {
    id: 'vae15',
    question: 'What are VAE applications?',
    options: ['Only generation', 'Image generation, compression, anomaly detection, representation learning', 'Classification only', 'No applications'],
    correctAnswer: 1,
    explanation: 'VAEs used for: generative modeling, dimensionality reduction, outlier detection, learning representations.'
  },
  {
    id: 'vae16',
    question: 'Can VAEs do semi-supervised learning?',
    options: ['No', 'Yes, by modeling both labels and data jointly', 'Supervised only', 'Unsupervised only'],
    correctAnswer: 1,
    explanation: 'VAEs can incorporate labeled and unlabeled data, learning from both for better representations.'
  },
  {
    id: 'vae17',
    question: 'What is the ELBO?',
    options: ['Loss function', 'Evidence Lower Bound: variational lower bound on log-likelihood maximized in VAE', 'Upper bound', 'No bound'],
    correctAnswer: 1,
    explanation: 'ELBO is the objective VAEs maximize, consisting of reconstruction term minus KL divergence.'
  },
  {
    id: 'vae18',
    question: 'What is posterior collapse?',
    options: ['No issue', 'Decoder ignores latent code z, KL → 0, losing latent information', 'Perfect behavior', 'Encoder problem'],
    correctAnswer: 1,
    explanation: 'Posterior collapse: decoder becomes too powerful, ignoring z; KL approaches zero, defeating VAE purpose.'
  },
  {
    id: 'vae19',
    question: 'How to mitigate posterior collapse?',
    options: ['Ignore it', 'KL annealing, free bits, architectural changes, weakening decoder', 'No solutions', 'Increase KL'],
    correctAnswer: 1,
    explanation: 'Solutions: gradually increase KL weight, enforce minimum KL, limit decoder capacity.'
  },
  {
    id: 'vae20',
    question: 'What is the relationship between VAEs and probabilistic modeling?',
    options: ['No relationship', 'VAEs are deep latent variable models using variational inference', 'Deterministic', 'No probability'],
    correctAnswer: 1,
    explanation: 'VAEs implement variational Bayesian inference for latent variable models, approximating intractable posteriors.'
  }
];

// Reinforcement Learning Basics - 20 questions
export const rlBasicsQuestions: QuizQuestion[] = [
  {
    id: 'rl1',
    question: 'What is reinforcement learning?',
    options: ['Supervised learning', 'Learning by trial and error through rewards and penalties', 'Unsupervised clustering', 'Classification'],
    correctAnswer: 1,
    explanation: 'RL agents learn optimal behavior by interacting with environment and receiving reward signals.'
  },
  {
    id: 'rl2',
    question: 'What are the key components of RL?',
    options: ['Data and labels', 'Agent, Environment, State, Action, Reward', 'Input-output', 'Features only'],
    correctAnswer: 1,
    explanation: 'RL formalized as MDP: agent takes actions in states, receives rewards, transitions to new states.'
  },
  {
    id: 'rl3',
    question: 'What is a policy?',
    options: ['Reward function', 'Mapping from states to actions (π: S → A)', 'Environment', 'Value function'],
    correctAnswer: 1,
    explanation: 'Policy π defines agent behavior: which action to take in each state.'
  },
  {
    id: 'rl4',
    question: 'What is the goal in RL?',
    options: ['Minimize loss', 'Maximize cumulative reward (return)', 'Classify correctly', 'Minimize actions'],
    correctAnswer: 1,
    explanation: 'RL seeks policy that maximizes expected sum of rewards over time.'
  },
  {
    id: 'rl5',
    question: 'What is the reward signal?',
    options: ['Loss', 'Scalar feedback indicating desirability of action', 'Gradient', 'Class label'],
    correctAnswer: 1,
    explanation: 'Reward r(s,a) provides immediate feedback; agent learns to maximize cumulative rewards.'
  },
  {
    id: 'rl6',
    question: 'What is the discount factor γ?',
    options: ['Learning rate', 'Weighs future rewards (0 ≤ γ ≤ 1)', 'Momentum', 'Regularization'],
    correctAnswer: 1,
    explanation: 'γ controls preference for immediate vs future rewards; γ=0 myopic, γ→1 far-sighted.'
  },
  {
    id: 'rl7',
    question: 'What is the return (cumulative reward)?',
    options: ['Single reward', 'Sum of discounted future rewards: G_t = Σ γ^k r_{t+k}', 'Immediate reward', 'Average reward'],
    correctAnswer: 1,
    explanation: 'Return is discounted sum of all future rewards from current timestep.'
  },
  {
    id: 'rl8',
    question: 'What is a value function V(s)?',
    options: ['Immediate reward', 'Expected return starting from state s under policy π', 'Action value', 'Reward only'],
    correctAnswer: 1,
    explanation: 'V^π(s) = E[G_t | s_t=s, π] estimates long-term value of being in state s.'
  },
  {
    id: 'rl9',
    question: 'What is Q-function Q(s,a)?',
    options: ['State value', 'Expected return from taking action a in state s', 'Immediate reward', 'Policy'],
    correctAnswer: 1,
    explanation: 'Q^π(s,a) = E[G_t | s_t=s, a_t=a, π] values state-action pairs.'
  },
  {
    id: 'rl10',
    question: 'What is the Bellman equation?',
    options: ['Loss function', 'Recursive relationship: V(s) = E[r + γV(s\')], Q(s,a) = E[r + γV(s\')]', 'Gradient', 'No recursion'],
    correctAnswer: 1,
    explanation: 'Bellman equations express value as immediate reward plus discounted future value.'
  },
  {
    id: 'rl11',
    question: 'What is exploration vs exploitation?',
    options: ['Same thing', 'Tradeoff: try new actions (explore) vs use known good actions (exploit)', 'Only explore', 'Only exploit'],
    correctAnswer: 1,
    explanation: 'RL agents must balance gathering information (explore) with maximizing rewards (exploit).'
  },
  {
    id: 'rl12',
    question: 'What is ε-greedy policy?',
    options: ['Always greedy', 'With probability ε explore randomly, else exploit best action', 'Always random', 'No exploration'],
    correctAnswer: 1,
    explanation: 'ε-greedy balances exploration (random) and exploitation (greedy) with tunable ε parameter.'
  },
  {
    id: 'rl13',
    question: 'What is model-based RL?',
    options: ['No model', 'Agent learns/uses model of environment dynamics to plan', 'Model-free', 'No planning'],
    correctAnswer: 1,
    explanation: 'Model-based RL learns transition P(s\'|s,a) and reward R(s,a) models for planning.'
  },
  {
    id: 'rl14',
    question: 'What is model-free RL?',
    options: ['Uses model', 'Learns policy/value directly from experience without explicit environment model', 'Model-based', 'No learning'],
    correctAnswer: 1,
    explanation: 'Model-free RL (Q-learning, policy gradients) learns from interaction without modeling dynamics.'
  },
  {
    id: 'rl15',
    question: 'What is Q-learning?',
    options: ['Policy gradient', 'Model-free off-policy algorithm learning optimal Q-function', 'Model-based', 'Supervised'],
    correctAnswer: 1,
    explanation: 'Q-learning iteratively updates Q(s,a) ← Q(s,a) + α[r + γ max Q(s\',a\') - Q(s,a)].'
  },
  {
    id: 'rl16',
    question: 'What is Deep Q-Networks (DQN)?',
    options: ['Linear Q-learning', 'Uses neural network to approximate Q-function', 'No network', 'Policy method'],
    correctAnswer: 1,
    explanation: 'DQN (DeepMind) uses CNN to learn Q-function, enabling RL on high-dimensional inputs like images.'
  },
  {
    id: 'rl17',
    question: 'What innovations did DQN introduce?',
    options: ['Nothing new', 'Experience replay and target network for stable training', 'Only replay', 'No innovations'],
    correctAnswer: 1,
    explanation: 'DQN uses experience replay (break correlation) and separate target network (stability).'
  },
  {
    id: 'rl18',
    question: 'What are policy gradient methods?',
    options: ['Value-based', 'Directly optimize policy parameters to maximize expected return', 'Q-learning', 'Model-based'],
    correctAnswer: 1,
    explanation: 'Policy gradient methods (REINFORCE, PPO) optimize policy π_θ directly via gradient ascent on J(θ).'
  },
  {
    id: 'rl19',
    question: 'What is actor-critic?',
    options: ['Only actor', 'Combines policy (actor) and value function (critic)', 'Only critic', 'No combination'],
    correctAnswer: 1,
    explanation: 'Actor-critic: actor (policy) takes actions, critic (value function) evaluates them for lower variance.'
  },
  {
    id: 'rl20',
    question: 'What RL applications exist?',
    options: ['None', 'Game playing (AlphaGo), robotics, resource management, recommendation', 'Only games', 'No applications'],
    correctAnswer: 1,
    explanation: 'RL powers: game AI, robot control, traffic optimization, personalized recommendations, drug discovery.'
  }
];
