import { QuizQuestion } from '../../types';

// Backpropagation - 20 questions
export const backpropagationQuestions: QuizQuestion[] = [
  {
    id: 'bp1',
    question: 'What is backpropagation?',
    options: ['Forward pass', 'Algorithm for computing gradients of loss with respect to weights', 'Weight initialization', 'Activation function'],
    correctAnswer: 1,
    explanation: 'Backpropagation is the algorithm that computes gradients by propagating errors backward through the network using the chain rule.'
  },
  {
    id: 'bp2',
    question: 'What mathematical principle does backpropagation use?',
    options: ['Bayes theorem', 'Chain rule of calculus', 'Fourier transform', 'Linear algebra only'],
    correctAnswer: 1,
    explanation: 'Backpropagation applies the chain rule to efficiently compute gradients of the loss with respect to each weight.'
  },
  {
    id: 'bp3',
    question: 'What are the two main phases in training a neural network?',
    options: ['Training and testing', 'Forward pass (compute output) and backward pass (compute gradients)', 'Initialize and optimize', 'Load and save'],
    correctAnswer: 1,
    explanation: 'Forward pass computes predictions, backward pass (backpropagation) computes gradients for weight updates.'
  },
  {
    id: 'bp4',
    question: 'In which direction does backpropagation compute gradients?',
    options: ['Forward from input to output', 'Backward from output to input', 'Random order', 'Simultaneously all layers'],
    correctAnswer: 1,
    explanation: 'Backpropagation starts from the output layer and propagates gradients backward to earlier layers.'
  },
  {
    id: 'bp5',
    question: 'Why is backpropagation efficient?',
    options: ['Uses random sampling', 'Reuses computed gradients through chain rule, avoiding redundant calculations', 'Simplifies the network', 'Reduces layers'],
    correctAnswer: 1,
    explanation: 'Backpropagation efficiently computes all gradients in one backward pass by reusing intermediate gradients via the chain rule.'
  },
  {
    id: 'bp6',
    question: 'What would be the alternative to backpropagation for gradient computation?',
    options: ['No alternative', 'Numerical differentiation (finite differences), which is much slower', 'Random search', 'Genetic algorithms'],
    correctAnswer: 1,
    explanation: 'Numerical differentiation requires evaluating the loss function multiple times per weight, making it impractical for large networks.'
  },
  {
    id: 'bp7',
    question: 'What is the gradient of a layer?',
    options: ['The layer output', 'The derivative of loss with respect to that layer\'s weights', 'The activation function', 'The input'],
    correctAnswer: 1,
    explanation: 'The gradient indicates how much the loss changes with respect to each weight, guiding the weight update direction.'
  },
  {
    id: 'bp8',
    question: 'What happens after backpropagation computes gradients?',
    options: ['Training ends', 'Weights are updated using an optimizer (e.g., gradient descent)', 'Network resets', 'Data is shuffled'],
    correctAnswer: 1,
    explanation: 'After computing gradients, an optimization algorithm like gradient descent updates weights to minimize the loss.'
  },
  {
    id: 'bp9',
    question: 'Does backpropagation update weights?',
    options: ['Yes, directly', 'No, it only computes gradients; optimizers update weights', 'Only in output layer', 'Only with SGD'],
    correctAnswer: 1,
    explanation: 'Backpropagation computes gradients. The optimizer (SGD, Adam, etc.) uses these gradients to update weights.'
  },
  {
    id: 'bp10',
    question: 'What is the vanishing gradient problem in backpropagation?',
    options: ['Gradients become too large', 'Gradients become very small in early layers, slowing learning', 'No gradients exist', 'Gradients are random'],
    correctAnswer: 1,
    explanation: 'In deep networks with certain activations (sigmoid/tanh), gradients can become exponentially small when propagated backward.'
  },
  {
    id: 'bp11',
    question: 'What causes vanishing gradients?',
    options: ['Too much data', 'Repeated multiplication of small derivatives (< 1) through many layers', 'Bad initialization', 'High learning rate'],
    correctAnswer: 1,
    explanation: 'Sigmoid/tanh derivatives are < 1. Multiplying many such values (chain rule) makes gradients vanish in deep networks.'
  },
  {
    id: 'bp12',
    question: 'What is the exploding gradient problem?',
    options: ['Gradients vanish', 'Gradients become very large, causing unstable training', 'Network stops learning', 'Loss increases'],
    correctAnswer: 1,
    explanation: 'When gradients grow exponentially during backpropagation, weights can update drastically, causing instability and divergence.'
  },
  {
    id: 'bp13',
    question: 'How can you address exploding gradients?',
    options: ['Use more layers', 'Gradient clipping: cap gradients at a threshold', 'Remove regularization', 'Increase learning rate'],
    correctAnswer: 1,
    explanation: 'Gradient clipping limits the magnitude of gradients, preventing extreme weight updates and stabilizing training.'
  },
  {
    id: 'bp14',
    question: 'How does ReLU activation help with vanishing gradients?',
    options: ['It doesn\'t', 'ReLU has gradient 1 for positive values, avoiding the < 1 multiplication issue', 'Increases gradients', 'Removes backprop'],
    correctAnswer: 1,
    explanation: 'ReLU\'s gradient is 1 for positive inputs, so repeated multiplication doesn\'t shrink gradients as sigmoid/tanh do.'
  },
  {
    id: 'bp15',
    question: 'What is automatic differentiation?',
    options: ['Manual gradient calculation', 'Computational technique to automatically compute derivatives of functions', 'Numerical differentiation', 'Random gradients'],
    correctAnswer: 1,
    explanation: 'Automatic differentiation (autodiff) automatically computes exact derivatives. Modern frameworks (PyTorch, TensorFlow) use it for backprop.'
  },
  {
    id: 'bp16',
    question: 'How do modern deep learning frameworks handle backpropagation?',
    options: ['You must code it manually', 'They automatically compute gradients (autograd)', 'Use numerical methods', 'Approximate gradients'],
    correctAnswer: 1,
    explanation: 'Frameworks like PyTorch and TensorFlow provide automatic differentiation, computing gradients automatically during backward pass.'
  },
  {
    id: 'bp17',
    question: 'What is the computational graph in backpropagation?',
    options: ['Network architecture', 'Graph representing operations; used to compute gradients via chain rule', 'Loss plot', 'Data flow diagram'],
    correctAnswer: 1,
    explanation: 'The computational graph tracks operations in the forward pass, enabling automatic gradient computation in the backward pass.'
  },
  {
    id: 'bp18',
    question: 'Is backpropagation specific to neural networks?',
    options: ['Yes, only neural networks', 'No, it\'s a general method for computing gradients in any differentiable function', 'Only for CNNs', 'Only for deep networks'],
    correctAnswer: 1,
    explanation: 'Backpropagation is a general algorithm for computing gradients in any composite differentiable function, not just neural networks.'
  },
  {
    id: 'bp19',
    question: 'What is the derivative of the ReLU function?',
    options: ['Always 1', '1 if x > 0, else 0', '0 always', 'x'],
    correctAnswer: 1,
    explanation: 'ReLU\'s derivative: 1 for positive inputs (where ReLU(x) = x), 0 for negative inputs (where ReLU(x) = 0).'
  },
  {
    id: 'bp20',
    question: 'Can you train a neural network without backpropagation?',
    options: ['No, impossible', 'Yes, but inefficiently (e.g., evolutionary algorithms, numerical methods)', 'Only shallow networks', 'Only with SGD'],
    correctAnswer: 1,
    explanation: 'Alternative methods exist (genetic algorithms, numerical differentiation) but are far less efficient than backpropagation.'
  }
];

// Gradient Descent & Optimizers - 25 questions
export const gradientDescentQuestions: QuizQuestion[] = [
  {
    id: 'gd1',
    question: 'What is gradient descent?',
    options: ['Forward propagation', 'Optimization algorithm that minimizes loss by moving in direction of steepest descent', 'Activation function', 'Loss function'],
    correctAnswer: 1,
    explanation: 'Gradient descent iteratively updates parameters in the direction opposite to the gradient to minimize the loss function.'
  },
  {
    id: 'gd2',
    question: 'What does the learning rate control?',
    options: ['Number of epochs', 'Step size of parameter updates', 'Batch size', 'Number of layers'],
    correctAnswer: 1,
    explanation: 'Learning rate determines how much to adjust parameters in each update: w = w - learning_rate × gradient.'
  },
  {
    id: 'gd3',
    question: 'What happens if the learning rate is too high?',
    options: ['Slow convergence', 'Overshooting the minimum, unstable training, or divergence', 'Perfect convergence', 'No change'],
    correctAnswer: 1,
    explanation: 'A high learning rate can cause large updates that overshoot minima, leading to oscillation or divergence.'
  },
  {
    id: 'gd4',
    question: 'What happens if the learning rate is too small?',
    options: ['Fast convergence', 'Very slow convergence, may get stuck in local minima', 'Divergence', 'Better accuracy'],
    correctAnswer: 1,
    explanation: 'A low learning rate makes tiny steps, resulting in very slow training and potential trapping in local minima.'
  },
  {
    id: 'gd5',
    question: 'What is Batch Gradient Descent?',
    options: ['Uses one sample', 'Uses entire dataset to compute gradient each iteration', 'Uses mini-batches', 'Random sampling'],
    correctAnswer: 1,
    explanation: 'Batch GD computes gradient over the full dataset per update. Accurate but slow and memory-intensive for large datasets.'
  },
  {
    id: 'gd6',
    question: 'What is Stochastic Gradient Descent (SGD)?',
    options: ['Uses entire dataset', 'Uses one sample at a time to compute gradient', 'Uses mini-batches', 'No randomness'],
    correctAnswer: 1,
    explanation: 'SGD updates weights after each single training example, making it faster but noisier than batch gradient descent.'
  },
  {
    id: 'gd7',
    question: 'What is Mini-Batch Gradient Descent?',
    options: ['Uses one sample', 'Uses small batches of samples to compute gradient', 'Uses entire dataset', 'Random updates'],
    correctAnswer: 1,
    explanation: 'Mini-batch GD uses small batches (e.g., 32-256 samples), balancing efficiency and stability between batch and stochastic GD.'
  },
  {
    id: 'gd8',
    question: 'What is an advantage of SGD over Batch GD?',
    options: ['More accurate gradients', 'Faster updates, can escape local minima due to noise', 'Uses more memory', 'Always converges'],
    correctAnswer: 1,
    explanation: 'SGD updates frequently with noisy gradients, enabling faster iterations and potential escape from shallow local minima.'
  },
  {
    id: 'gd9',
    question: 'What is momentum in optimization?',
    options: ['Learning rate decay', 'Accumulates velocity vector to accelerate in consistent directions', 'Batch size', 'Regularization'],
    correctAnswer: 1,
    explanation: 'Momentum adds a fraction of the previous update to the current one, smoothing updates and accelerating convergence.'
  },
  {
    id: 'gd10',
    question: 'How does momentum help optimization?',
    options: ['Increases learning rate', 'Reduces oscillations and speeds up convergence', 'Adds layers', 'Removes gradients'],
    correctAnswer: 1,
    explanation: 'Momentum dampens oscillations in narrow valleys and accelerates movement in consistent gradient directions.'
  },
  {
    id: 'gd11',
    question: 'What is the momentum coefficient (β)?',
    options: ['Learning rate', 'Controls how much of previous velocity to retain (typically 0.9)', 'Batch size', 'Regularization parameter'],
    correctAnswer: 1,
    explanation: 'β (typically 0.9) determines how much of the previous velocity is carried forward: v = βv + gradient.'
  },
  {
    id: 'gd12',
    question: 'What is AdaGrad optimizer?',
    options: ['Fixed learning rate', 'Adapts learning rate per parameter based on historical gradients', 'Momentum variant', 'No learning rate'],
    correctAnswer: 1,
    explanation: 'AdaGrad scales learning rate inversely to square root of sum of squared gradients, giving smaller updates to frequent features.'
  },
  {
    id: 'gd13',
    question: 'What is a problem with AdaGrad?',
    options: ['Too fast', 'Learning rate decays too aggressively, can stop learning', 'No adaptation', 'Too complex'],
    correctAnswer: 1,
    explanation: 'AdaGrad\'s accumulation of squared gradients causes the learning rate to become vanishingly small over time.'
  },
  {
    id: 'gd14',
    question: 'What is RMSprop?',
    options: ['Variant of SGD', 'Uses exponentially weighted moving average of squared gradients to adapt learning rate', 'Momentum only', 'Batch normalization'],
    correctAnswer: 1,
    explanation: 'RMSprop addresses AdaGrad\'s aggressive decay by using exponential moving average, keeping recent gradient history.'
  },
  {
    id: 'gd15',
    question: 'What is Adam optimizer?',
    options: ['Simple SGD', 'Combines momentum and RMSprop: adaptive learning rates with momentum', 'Only momentum', 'Only learning rate adaptation'],
    correctAnswer: 1,
    explanation: 'Adam (Adaptive Moment Estimation) combines momentum (first moment) and RMSprop (second moment) for efficient optimization.'
  },
  {
    id: 'gd16',
    question: 'Why is Adam popular?',
    options: ['Simplest optimizer', 'Works well out-of-the-box with default hyperparameters for many problems', 'Always best', 'Fastest'],
    correctAnswer: 1,
    explanation: 'Adam is robust and requires little tuning, making it a popular default choice, though not always optimal.'
  },
  {
    id: 'gd17',
    question: 'What are Adam\'s hyperparameters?',
    options: ['Only learning rate', 'Learning rate (α), momentum decay (β₁), RMSprop decay (β₂), epsilon', 'Only batch size', 'No hyperparameters'],
    correctAnswer: 1,
    explanation: 'Adam has learning rate α, first moment decay β₁ (usually 0.9), second moment decay β₂ (usually 0.999), and ε for stability.'
  },
  {
    id: 'gd18',
    question: 'What is learning rate scheduling?',
    options: ['Fixed learning rate', 'Adjusting learning rate during training (e.g., decay, warm-up)', 'Batch size changes', 'Architecture changes'],
    correctAnswer: 1,
    explanation: 'Learning rate scheduling changes the learning rate over time, such as decaying it or warming it up, to improve convergence.'
  },
  {
    id: 'gd19',
    question: 'What is learning rate decay?',
    options: ['Increasing learning rate', 'Gradually reducing learning rate over training', 'Fixed learning rate', 'Random learning rate'],
    correctAnswer: 1,
    explanation: 'Decay reduces learning rate over time (e.g., exponential, step, or cosine decay) to refine convergence near minima.'
  },
  {
    id: 'gd20',
    question: 'What is learning rate warm-up?',
    options: ['Starting with low learning rate, gradually increasing', 'Starting with high learning rate, gradually increasing', 'No change', 'Random start'],
    correctAnswer: 0,
    explanation: 'Warm-up starts with a small learning rate and gradually increases it, helping stabilize early training.'
  },
  {
    id: 'gd21',
    question: 'What is the AdamW optimizer?',
    options: ['Adam without momentum', 'Adam with decoupled weight decay regularization', 'Faster Adam', 'Simplified Adam'],
    correctAnswer: 1,
    explanation: 'AdamW correctly implements weight decay by decoupling it from gradient updates, often improving generalization.'
  },
  {
    id: 'gd22',
    question: 'What is Nesterov Accelerated Gradient (NAG)?',
    options: ['Standard momentum', 'Looks ahead to where momentum will take you, then computes gradient', 'No momentum', 'Random updates'],
    correctAnswer: 1,
    explanation: 'NAG computes gradient at the anticipated future position, providing better updates than standard momentum.'
  },
  {
    id: 'gd23',
    question: 'When would you use SGD over Adam?',
    options: ['Never', 'When you want better generalization with careful tuning', 'Always', 'For small datasets only'],
    correctAnswer: 1,
    explanation: 'SGD with momentum can generalize better than Adam in some cases but requires more careful learning rate tuning.'
  },
  {
    id: 'gd24',
    question: 'What is the role of the optimizer in training?',
    options: ['Defines architecture', 'Updates weights using computed gradients to minimize loss', 'Computes loss', 'Loads data'],
    correctAnswer: 1,
    explanation: 'The optimizer takes gradients from backpropagation and updates model weights to reduce the loss function.'
  },
  {
    id: 'gd25',
    question: 'Can different layers use different learning rates?',
    options: ['No, must be uniform', 'Yes, through parameter groups (useful for transfer learning)', 'Only in CNNs', 'Only in RNNs'],
    correctAnswer: 1,
    explanation: 'You can set different learning rates for different layers, often used in transfer learning to fine-tune pre-trained layers slowly.'
  }
];

// Batch Normalization - 20 questions
export const batchNormQuestions: QuizQuestion[] = [
  {
    id: 'bn1',
    question: 'What is Batch Normalization?',
    options: ['Data preprocessing', 'Technique that normalizes layer inputs within each mini-batch during training', 'Weight initialization', 'Loss function'],
    correctAnswer: 1,
    explanation: 'Batch Normalization normalizes inputs to each layer for each mini-batch, stabilizing and accelerating training.'
  },
  {
    id: 'bn2',
    question: 'What problem does Batch Normalization address?',
    options: ['Overfitting', 'Internal covariate shift: changing distributions of layer inputs during training', 'Vanishing weights', 'Data imbalance'],
    correctAnswer: 1,
    explanation: 'BN addresses internal covariate shift, where layer input distributions change as previous layers update, slowing training.'
  },
  {
    id: 'bn3',
    question: 'Where is Batch Normalization typically applied?',
    options: ['Input layer only', 'Between layers, usually after linear transformation and before activation', 'Output layer only', 'Only in CNNs'],
    correctAnswer: 1,
    explanation: 'BN is typically applied after linear/conv layers and before activation functions, though placement can vary.'
  },
  {
    id: 'bn4',
    question: 'What does Batch Normalization do mathematically?',
    options: ['Scales weights', 'Normalizes to zero mean and unit variance, then scales and shifts with learned parameters', 'Adds noise', 'Drops neurons'],
    correctAnswer: 1,
    explanation: 'BN normalizes to mean 0 and variance 1, then applies learned scale (γ) and shift (β): y = γ × x_norm + β.'
  },
  {
    id: 'bn5',
    question: 'Why does Batch Normalization learn scale (γ) and shift (β) parameters?',
    options: ['For fun', 'To allow network to undo normalization if beneficial', 'To add complexity', 'To reduce parameters'],
    correctAnswer: 1,
    explanation: 'γ and β are learned parameters allowing the network to recover the original distribution if normalization hurts performance.'
  },
  {
    id: 'bn6',
    question: 'What are benefits of Batch Normalization?',
    options: ['Reduces parameters', 'Faster training, allows higher learning rates, provides regularization', 'Simplifies architecture', 'Reduces data needs'],
    correctAnswer: 1,
    explanation: 'BN stabilizes training (allowing higher learning rates), accelerates convergence, and acts as regularizer, reducing overfitting.'
  },
  {
    id: 'bn7',
    question: 'How does Batch Normalization provide regularization?',
    options: ['Adds L2 penalty', 'Adds noise through batch statistics, similar to dropout', 'Reduces weights', 'Increases dropout'],
    correctAnswer: 1,
    explanation: 'Using mini-batch statistics introduces noise, providing a regularizing effect similar to dropout.'
  },
  {
    id: 'bn8',
    question: 'What is the difference between training and inference for Batch Normalization?',
    options: ['No difference', 'Training uses batch statistics; inference uses running averages from training', 'Inference is faster', 'Training is simpler'],
    correctAnswer: 1,
    explanation: 'During training, BN uses mini-batch mean/variance. At inference, it uses running averages accumulated during training for stability.'
  },
  {
    id: 'bn9',
    question: 'Why not use batch statistics during inference?',
    options: ['Too slow', 'Batch sizes might be 1, or statistics would vary between batches', 'Not needed', 'Always use batch stats'],
    correctAnswer: 1,
    explanation: 'Inference might have single samples or small batches with unreliable statistics. Running averages provide consistent normalization.'
  },
  {
    id: 'bn10',
    question: 'What is a limitation of Batch Normalization?',
    options: ['Too simple', 'Performance depends on batch size; small batches give poor estimates', 'Too slow', 'Too much memory'],
    correctAnswer: 1,
    explanation: 'BN estimates mean/variance from the batch. Small batches provide noisy estimates, degrading performance.'
  },
  {
    id: 'bn11',
    question: 'What is Layer Normalization?',
    options: ['Same as BN', 'Normalizes across features for each sample, not across batch', 'Batch-independent normalization', 'Weight normalization'],
    correctAnswer: 1,
    explanation: 'Layer Norm normalizes across all features for each sample independently, unaffected by batch size.'
  },
  {
    id: 'bn12',
    question: 'When is Layer Normalization preferred over Batch Normalization?',
    options: ['Always', 'For RNNs, transformers, or small batch sizes', 'Never', 'Only for images'],
    correctAnswer: 1,
    explanation: 'Layer Norm works well with variable-length sequences (RNNs) and is batch-size independent, making it ideal for transformers.'
  },
  {
    id: 'bn13',
    question: 'What is Group Normalization?',
    options: ['BN variant', 'Normalizes within groups of channels, between BN and Layer Norm', 'Instance normalization', 'No normalization'],
    correctAnswer: 1,
    explanation: 'Group Norm divides channels into groups and normalizes within each group, working well for small batches (e.g., in computer vision).'
  },
  {
    id: 'bn14',
    question: 'What is Instance Normalization?',
    options: ['Batch normalization', 'Normalizes each sample and channel independently', 'Layer normalization', 'Weight normalization'],
    correctAnswer: 1,
    explanation: 'Instance Norm normalizes each sample and each channel independently, used in style transfer and GANs.'
  },
  {
    id: 'bn15',
    question: 'Can Batch Normalization help with vanishing gradients?',
    options: ['No', 'Yes, by keeping activations in a reasonable range', 'Only with ReLU', 'Only in shallow networks'],
    correctAnswer: 1,
    explanation: 'BN prevents activations from becoming too small or large, helping maintain healthy gradient flow through deep networks.'
  },
  {
    id: 'bn16',
    question: 'Does Batch Normalization replace dropout?',
    options: ['Yes, always', 'Often reduces need for dropout, but they can be complementary', 'No, always use both', 'BN is worse than dropout'],
    correctAnswer: 1,
    explanation: 'BN provides regularization, often reducing or eliminating the need for dropout, though both can be used together.'
  },
  {
    id: 'bn17',
    question: 'What is the typical value of the epsilon parameter in BN?',
    options: ['1.0', 'Small value like 1e-5 for numerical stability', '0', '0.5'],
    correctAnswer: 1,
    explanation: 'Epsilon (e.g., 1e-5) is added to variance to prevent division by zero: x_norm = (x - μ) / sqrt(σ² + ε).'
  },
  {
    id: 'bn18',
    question: 'Can Batch Normalization be used in any type of network?',
    options: ['Only MLPs', 'Yes, MLPs, CNNs, RNNs, though effectiveness varies', 'Only CNNs', 'Only RNNs'],
    correctAnswer: 1,
    explanation: 'BN can be applied to various architectures, though alternatives like Layer Norm are often better for RNNs/transformers.'
  },
  {
    id: 'bn19',
    question: 'What was the original motivation for Batch Normalization?',
    options: ['Reduce overfitting', 'Allow higher learning rates and faster training', 'Reduce parameters', 'Improve accuracy only'],
    correctAnswer: 1,
    explanation: 'BN was introduced to enable higher learning rates and faster convergence by reducing internal covariate shift.'
  },
  {
    id: 'bn20',
    question: 'How does Batch Normalization affect the loss landscape?',
    options: ['Makes it rougher', 'Smooths the loss landscape, making optimization easier', 'No effect', 'Makes it more complex'],
    correctAnswer: 1,
    explanation: 'BN smooths the optimization landscape, making gradients more predictable and allowing for more aggressive learning rates.'
  }
];

// Loss Functions - 20 questions
export const lossFunctionsQuestions: QuizQuestion[] = [
  {
    id: 'loss1',
    question: 'What is a loss function?',
    options: ['Activation function', 'Measures how wrong predictions are; guides optimization', 'Weight initialization', 'Data augmentation'],
    correctAnswer: 1,
    explanation: 'The loss (cost/objective) function quantifies the difference between predictions and true labels, guiding training.'
  },
  {
    id: 'loss2',
    question: 'What is Mean Squared Error (MSE)?',
    options: ['Classification loss', 'Average of squared differences: (1/n) × Σ(y_pred - y_true)²', 'Log loss', 'Hinge loss'],
    correctAnswer: 1,
    explanation: 'MSE is a common regression loss that penalizes large errors quadratically: MSE = (1/n) Σ(y_pred - y_true)².'
  },
  {
    id: 'loss3',
    question: 'When is MSE used?',
    options: ['Classification', 'Regression tasks', 'Clustering', 'Ranking'],
    correctAnswer: 1,
    explanation: 'MSE is primarily used for regression where you predict continuous values.'
  },
  {
    id: 'loss4',
    question: 'What is a disadvantage of MSE?',
    options: ['Too simple', 'Sensitive to outliers due to squaring', 'Not differentiable', 'Only for binary'],
    correctAnswer: 1,
    explanation: 'Squaring errors makes MSE sensitive to outliers, which can dominate the loss and distort training.'
  },
  {
    id: 'loss5',
    question: 'What is Mean Absolute Error (MAE)?',
    options: ['Squared error', 'Average of absolute differences: (1/n) × Σ|y_pred - y_true|', 'Log loss', 'Cross-entropy'],
    correctAnswer: 1,
    explanation: 'MAE measures average absolute error: MAE = (1/n) Σ|y_pred - y_true|, less sensitive to outliers than MSE.'
  },
  {
    id: 'loss6',
    question: 'What is Binary Cross-Entropy (BCE)?',
    options: ['Regression loss', 'Loss for binary classification: -[y×log(p) + (1-y)×log(1-p)]', 'Multi-class loss', 'Regression metric'],
    correctAnswer: 1,
    explanation: 'BCE measures dissimilarity between true labels (0/1) and predicted probabilities for binary classification.'
  },
  {
    id: 'loss7',
    question: 'What output activation is used with Binary Cross-Entropy?',
    options: ['ReLU', 'Sigmoid', 'Softmax', 'Linear'],
    correctAnswer: 1,
    explanation: 'Sigmoid activation outputs probabilities [0,1] for binary classification, paired with BCE loss.'
  },
  {
    id: 'loss8',
    question: 'What is Categorical Cross-Entropy?',
    options: ['Binary loss', 'Loss for multi-class classification: -Σ(y_true × log(y_pred))', 'Regression loss', 'Clustering loss'],
    correctAnswer: 1,
    explanation: 'Categorical cross-entropy measures error for multi-class problems with one-hot encoded labels.'
  },
  {
    id: 'loss9',
    question: 'What output activation is used with Categorical Cross-Entropy?',
    options: ['Sigmoid', 'Softmax', 'ReLU', 'Tanh'],
    correctAnswer: 1,
    explanation: 'Softmax outputs a probability distribution over classes, paired with categorical cross-entropy for multi-class classification.'
  },
  {
    id: 'loss10',
    question: 'What is Sparse Categorical Cross-Entropy?',
    options: ['Different algorithm', 'Same as categorical cross-entropy but accepts integer labels instead of one-hot', 'Binary loss', 'Regression loss'],
    correctAnswer: 1,
    explanation: 'Sparse categorical cross-entropy is computationally equivalent but accepts integer class labels instead of one-hot vectors.'
  },
  {
    id: 'loss11',
    question: 'What is Hinge Loss?',
    options: ['Regression loss', 'Loss for SVMs and margin-based classification: max(0, 1 - y×f(x))', 'Cross-entropy', 'MSE variant'],
    correctAnswer: 1,
    explanation: 'Hinge loss is used in SVMs, penalizing predictions that are on the wrong side or too close to the decision boundary.'
  },
  {
    id: 'loss12',
    question: 'What is Huber Loss?',
    options: ['MSE', 'Combines MSE and MAE: quadratic for small errors, linear for large', 'Cross-entropy', 'Hinge loss'],
    correctAnswer: 1,
    explanation: 'Huber loss is less sensitive to outliers than MSE (using MAE for large errors) while remaining smooth.'
  },
  {
    id: 'loss13',
    question: 'When would you use Huber Loss?',
    options: ['Classification', 'Regression with outliers', 'Clustering', 'Always'],
    correctAnswer: 1,
    explanation: 'Huber loss is robust to outliers in regression tasks, providing a balance between MSE and MAE.'
  },
  {
    id: 'loss14',
    question: 'What is KL Divergence?',
    options: ['Classification loss', 'Measures how one probability distribution differs from another', 'Regression loss', 'Activation function'],
    correctAnswer: 1,
    explanation: 'Kullback-Leibler divergence quantifies the difference between two probability distributions, used in VAEs and some losses.'
  },
  {
    id: 'loss15',
    question: 'What is Focal Loss?',
    options: ['MSE variant', 'Modified cross-entropy that focuses on hard examples by down-weighting easy ones', 'Regression loss', 'Clustering loss'],
    correctAnswer: 1,
    explanation: 'Focal loss adds a modulating factor to cross-entropy, addressing class imbalance by focusing on hard-to-classify examples.'
  },
  {
    id: 'loss16',
    question: 'When is Focal Loss useful?',
    options: ['Balanced datasets', 'Highly imbalanced datasets (e.g., object detection)', 'Regression', 'Clustering'],
    correctAnswer: 1,
    explanation: 'Focal loss was designed for object detection with extreme class imbalance, reducing the weight of easy negatives.'
  },
  {
    id: 'loss17',
    question: 'What is Triplet Loss?',
    options: ['Three-class classification', 'Used in metric learning: encourages similar examples to be close, dissimilar to be far', 'Regression loss', 'Binary loss'],
    correctAnswer: 1,
    explanation: 'Triplet loss trains embeddings by minimizing distance between anchor and positive, maximizing distance to negative.'
  },
  {
    id: 'loss18',
    question: 'Where is Triplet Loss commonly used?',
    options: ['Regression', 'Face recognition and similarity learning', 'Classification', 'Clustering'],
    correctAnswer: 1,
    explanation: 'Triplet loss is used in face verification, person re-identification, and other tasks requiring learning similarity metrics.'
  },
  {
    id: 'loss19',
    question: 'Can you use multiple loss functions together?',
    options: ['No, only one', 'Yes, as weighted sum of multiple losses', 'Only for multi-task', 'Never beneficial'],
    correctAnswer: 1,
    explanation: 'Multiple losses can be combined (e.g., classification + regularization + auxiliary losses) for multi-objective optimization.'
  },
  {
    id: 'loss20',
    question: 'What should you consider when choosing a loss function?',
    options: ['Random choice', 'Task type (classification/regression), data characteristics, and evaluation metric', 'Always use MSE', 'Always use cross-entropy'],
    correctAnswer: 1,
    explanation: 'Loss function should match the task (regression vs classification), handle data properties (imbalance, outliers), and align with metrics.'
  }
];
