import { QuizQuestion } from '../../types';

// Perceptron - 20 questions
export const perceptronQuestions: QuizQuestion[] = [
  {
    id: 'perc1',
    question: 'What is a perceptron?',
    options: ['A clustering algorithm', 'The simplest neural network unit with linear decision boundary', 'A decision tree', 'A regression model'],
    correctAnswer: 1,
    explanation: 'A perceptron is the simplest artificial neuron that computes a weighted sum of inputs and applies a step function.'
  },
  {
    id: 'perc2',
    question: 'What is the output function of a perceptron?',
    options: ['Sigmoid', 'Step function (threshold)', 'ReLU', 'Softmax'],
    correctAnswer: 1,
    explanation: 'The classic perceptron uses a step (threshold) function: output 1 if weighted sum ≥ threshold, else 0.'
  },
  {
    id: 'perc3',
    question: 'What types of problems can a single perceptron solve?',
    options: ['Any problem', 'Only linearly separable problems', 'Non-linear problems', 'Multi-class problems'],
    correctAnswer: 1,
    explanation: 'A single perceptron can only solve linearly separable problems as it creates a linear decision boundary.'
  },
  {
    id: 'perc4',
    question: 'Can a single perceptron solve the XOR problem?',
    options: ['Yes, easily', 'No, XOR is not linearly separable', 'Only with ReLU', 'Only with many epochs'],
    correctAnswer: 1,
    explanation: 'The XOR problem is not linearly separable, so a single perceptron cannot solve it. You need a multi-layer network.'
  },
  {
    id: 'perc5',
    question: 'What is the perceptron learning rule?',
    options: ['Gradient descent', 'Update weights based on classification errors', 'Random updates', 'Backpropagation'],
    correctAnswer: 1,
    explanation: 'The perceptron learning rule updates weights when misclassification occurs: w = w + learning_rate × (target - predicted) × input.'
  },
  {
    id: 'perc6',
    question: 'Does the perceptron learning algorithm guarantee convergence?',
    options: ['Never converges', 'Yes, if data is linearly separable', 'Always converges', 'Only for binary classification'],
    correctAnswer: 1,
    explanation: 'The perceptron convergence theorem states it will converge in finite steps if the data is linearly separable.'
  },
  {
    id: 'perc7',
    question: 'What is the bias term in a perceptron?',
    options: ['Error term', 'Allows decision boundary to not pass through origin', 'Learning rate', 'Activation function'],
    correctAnswer: 1,
    explanation: 'The bias term shifts the decision boundary away from the origin, adding flexibility to the model.'
  },
  {
    id: 'perc8',
    question: 'How many layers does a perceptron have?',
    options: ['One - input layer only', 'Two - input and output', 'Three', 'Many hidden layers'],
    correctAnswer: 1,
    explanation: 'A basic perceptron has an input layer and a single output unit, making it a single-layer network.'
  },
  {
    id: 'perc9',
    question: 'What happens when you stack multiple perceptrons in layers?',
    options: ['Still linear', 'Creates a multi-layer perceptron (MLP) that can learn non-linear functions', 'Decreases performance', 'No change'],
    correctAnswer: 1,
    explanation: 'Stacking perceptrons with non-linear activation functions creates an MLP capable of learning non-linear decision boundaries.'
  },
  {
    id: 'perc10',
    question: 'Why can\'t we use perceptron learning rule for multi-layer networks?',
    options: ['Too slow', 'It only works for single layer; need backpropagation for multiple layers', 'Not accurate', 'Too complex'],
    correctAnswer: 1,
    explanation: 'The simple perceptron rule doesn\'t work for hidden layers. Backpropagation is needed to compute gradients through multiple layers.'
  },
  {
    id: 'perc11',
    question: 'What is the decision boundary of a perceptron?',
    options: ['Curve', 'Hyperplane', 'Circle', 'Point'],
    correctAnswer: 1,
    explanation: 'A perceptron creates a hyperplane decision boundary (a line in 2D, plane in 3D, etc.).'
  },
  {
    id: 'perc12',
    question: 'Is the perceptron a probabilistic model?',
    options: ['Yes, outputs probabilities', 'No, outputs hard classifications', 'Only with sigmoid', 'Only for regression'],
    correctAnswer: 1,
    explanation: 'The classic perceptron outputs binary decisions (0 or 1), not probabilities.'
  },
  {
    id: 'perc13',
    question: 'What is the key limitation of the perceptron?',
    options: ['Too slow', 'Can only learn linear decision boundaries', 'Requires too much data', 'Cannot handle binary classification'],
    correctAnswer: 1,
    explanation: 'The fundamental limitation is that perceptrons can only model linear relationships and linearly separable data.'
  },
  {
    id: 'perc14',
    question: 'How does a perceptron make a prediction?',
    options: ['Random guess', 'Compute weighted sum of inputs + bias, apply activation function', 'Average inputs', 'Nearest neighbor'],
    correctAnswer: 1,
    explanation: 'Prediction: output = activation(w₁x₁ + w₂x₂ + ... + wₙxₙ + b) where w are weights and b is bias.'
  },
  {
    id: 'perc15',
    question: 'Can perceptrons handle multi-class classification?',
    options: ['No', 'Yes, using multiple perceptrons (one per class)', 'Only binary', 'Only with hidden layers'],
    correctAnswer: 1,
    explanation: 'Multi-class classification can be done using multiple perceptrons in the output layer (one-vs-all approach).'
  },
  {
    id: 'perc16',
    question: 'What inspired the perceptron model?',
    options: ['Decision trees', 'Biological neurons', 'Physical laws', 'Statistics'],
    correctAnswer: 1,
    explanation: 'The perceptron was inspired by biological neurons: inputs (dendrites), weighted sum (cell body), output (axon).'
  },
  {
    id: 'perc17',
    question: 'When was the perceptron invented?',
    options: ['2010s', 'Late 1950s by Frank Rosenblatt', '1990s', '2000s'],
    correctAnswer: 1,
    explanation: 'Frank Rosenblatt invented the perceptron in 1957-1958, one of the earliest artificial neural network models.'
  },
  {
    id: 'perc18',
    question: 'What caused the "AI winter" related to perceptrons?',
    options: ['Too successful', 'Minsky and Papert\'s book showed limitations (XOR problem)', 'Too complex', 'No interest'],
    correctAnswer: 1,
    explanation: 'In 1969, Minsky and Papert\'s book highlighted that perceptrons couldn\'t solve XOR, dampening enthusiasm for neural networks.'
  },
  {
    id: 'perc19',
    question: 'What is the relationship between perceptron and logistic regression?',
    options: ['Completely different', 'Similar, but logistic regression uses sigmoid for probabilities', 'Identical', 'No relationship'],
    correctAnswer: 1,
    explanation: 'Both are linear models, but logistic regression uses sigmoid activation to output probabilities instead of hard decisions.'
  },
  {
    id: 'perc20',
    question: 'Is feature scaling important for perceptrons?',
    options: ['No', 'Yes, features with larger scales will dominate the weighted sum', 'Only for deep networks', 'Only for images'],
    correctAnswer: 1,
    explanation: 'Feature scaling ensures all features contribute proportionally to the weighted sum and helps with faster, stable convergence.'
  }
];

// Multi-Layer Perceptron (MLP) - 20 questions
export const mlpQuestions: QuizQuestion[] = [
  {
    id: 'mlp1',
    question: 'What is a Multi-Layer Perceptron (MLP)?',
    options: ['Single layer network', 'Feed-forward neural network with one or more hidden layers', 'Convolutional network', 'Recurrent network'],
    correctAnswer: 1,
    explanation: 'An MLP is a feed-forward artificial neural network with at least one hidden layer between input and output layers.'
  },
  {
    id: 'mlp2',
    question: 'Why do MLPs need non-linear activation functions?',
    options: ['For speed', 'Without them, multiple layers collapse to a single linear transformation', 'For accuracy', 'For visualization'],
    correctAnswer: 1,
    explanation: 'Stacking linear layers without non-linear activations is equivalent to a single linear layer; non-linearity enables learning complex patterns.'
  },
  {
    id: 'mlp3',
    question: 'What types of problems can MLPs solve that perceptrons cannot?',
    options: ['Only linear problems', 'Non-linearly separable problems like XOR', 'Simple problems only', 'Clustering only'],
    correctAnswer: 1,
    explanation: 'MLPs with hidden layers and non-linear activations can learn non-linear decision boundaries, solving problems like XOR.'
  },
  {
    id: 'mlp4',
    question: 'What is a fully-connected layer in an MLP?',
    options: ['Sparse connections', 'Every neuron connects to all neurons in the previous layer', 'Random connections', 'Convolutional layer'],
    correctAnswer: 1,
    explanation: 'In a fully-connected (dense) layer, each neuron receives input from every neuron in the previous layer.'
  },
  {
    id: 'mlp5',
    question: 'How are MLPs trained?',
    options: ['Random search', 'Backpropagation algorithm with gradient descent', 'Perceptron rule', 'Genetic algorithms'],
    correctAnswer: 1,
    explanation: 'MLPs are trained using backpropagation to compute gradients and gradient descent to update weights.'
  },
  {
    id: 'mlp6',
    question: 'What is forward propagation in an MLP?',
    options: ['Weight update', 'Computing outputs layer by layer from input to output', 'Error calculation', 'Gradient computation'],
    correctAnswer: 1,
    explanation: 'Forward propagation passes inputs through the network layer by layer, applying weights and activations to produce output.'
  },
  {
    id: 'mlp7',
    question: 'What does the Universal Approximation Theorem state?',
    options: ['All models are equal', 'An MLP with one hidden layer can approximate any continuous function', 'Deep is always better', 'Simple models best'],
    correctAnswer: 1,
    explanation: 'The theorem states that a feed-forward network with a single hidden layer can approximate any continuous function, given enough neurons.'
  },
  {
    id: 'mlp8',
    question: 'If one hidden layer can approximate anything, why use deep networks?',
    options: ['No reason', 'Deep networks can be more efficient, requiring fewer total parameters', 'Just fashion', 'Only for images'],
    correctAnswer: 1,
    explanation: 'While one layer suffices theoretically, deep networks can learn hierarchical features more efficiently with fewer parameters.'
  },
  {
    id: 'mlp9',
    question: 'How do you choose the number of hidden layers?',
    options: ['Always use 1', 'Through experimentation and validation performance', 'Random choice', 'Always use 100'],
    correctAnswer: 1,
    explanation: 'The number of hidden layers is a hyperparameter typically chosen through cross-validation and experimentation.'
  },
  {
    id: 'mlp10',
    question: 'How do you choose the number of neurons per hidden layer?',
    options: ['Always 10', 'Depends on problem complexity; found through experimentation', 'Always equal to input size', 'Random'],
    correctAnswer: 1,
    explanation: 'The number of neurons affects capacity and is tuned as a hyperparameter, balancing complexity and generalization.'
  },
  {
    id: 'mlp11',
    question: 'What is the role of the output layer in an MLP?',
    options: ['Feature extraction', 'Produces final predictions (classification or regression)', 'Data preprocessing', 'Weight initialization'],
    correctAnswer: 1,
    explanation: 'The output layer produces the final prediction: one neuron for binary classification/regression, multiple for multi-class.'
  },
  {
    id: 'mlp12',
    question: 'What activation is typically used in the output layer for binary classification?',
    options: ['ReLU', 'Sigmoid', 'Tanh', 'Linear'],
    correctAnswer: 1,
    explanation: 'Sigmoid activation outputs a probability between 0 and 1, suitable for binary classification.'
  },
  {
    id: 'mlp13',
    question: 'What activation is typically used in the output layer for multi-class classification?',
    options: ['ReLU', 'Softmax', 'Sigmoid', 'Tanh'],
    correctAnswer: 1,
    explanation: 'Softmax converts output logits into a probability distribution over classes, summing to 1.'
  },
  {
    id: 'mlp14',
    question: 'What activation is typically used in the output layer for regression?',
    options: ['Sigmoid', 'Linear (no activation)', 'ReLU', 'Softmax'],
    correctAnswer: 1,
    explanation: 'For regression, a linear activation (or no activation) allows the output to take any real value.'
  },
  {
    id: 'mlp15',
    question: 'What is the vanishing gradient problem in deep MLPs?',
    options: ['Too many gradients', 'Gradients become very small in early layers, slowing learning', 'Exploding weights', 'No convergence'],
    correctAnswer: 1,
    explanation: 'With sigmoid/tanh activations, gradients can become very small when backpropagated through many layers, hampering learning.'
  },
  {
    id: 'mlp16',
    question: 'How does ReLU activation help with vanishing gradients?',
    options: ['It doesn\'t', 'ReLU has constant gradient of 1 for positive values', 'Makes gradients larger', 'Removes backpropagation'],
    correctAnswer: 1,
    explanation: 'ReLU has gradient 1 for positive inputs (0 for negative), avoiding the vanishing gradient problem of sigmoid/tanh.'
  },
  {
    id: 'mlp17',
    question: 'Can MLPs handle sequential/temporal data naturally?',
    options: ['Yes, perfectly', 'No, MLPs don\'t have memory; use RNNs for sequences', 'Only with many layers', 'Only for short sequences'],
    correctAnswer: 1,
    explanation: 'MLPs process inputs independently without memory of previous inputs. RNNs/LSTMs are designed for sequential data.'
  },
  {
    id: 'mlp18',
    question: 'Can MLPs handle spatial structure in images?',
    options: ['Yes, optimally', 'Not efficiently; CNNs are better for images', 'Only small images', 'Only grayscale'],
    correctAnswer: 1,
    explanation: 'MLPs treat pixels as independent features, ignoring spatial structure. CNNs use convolutional layers to capture spatial patterns.'
  },
  {
    id: 'mlp19',
    question: 'What is dropout in MLPs?',
    options: ['Removing layers', 'Randomly dropping neurons during training to prevent overfitting', 'Early stopping', 'Data augmentation'],
    correctAnswer: 1,
    explanation: 'Dropout randomly sets a fraction of neuron outputs to zero during training, acting as regularization to reduce overfitting.'
  },
  {
    id: 'mlp20',
    question: 'Are MLPs suitable for tabular data?',
    options: ['No', 'Yes, MLPs work well for structured/tabular data', 'Only for images', 'Only for text'],
    correctAnswer: 1,
    explanation: 'MLPs are commonly used for tabular data and can learn complex non-linear relationships between features.'
  }
];

// Activation Functions - 25 questions
export const activationFunctionsQuestions: QuizQuestion[] = [
  {
    id: 'act1',
    question: 'What is the purpose of activation functions?',
    options: ['Speed up training', 'Introduce non-linearity to learn complex patterns', 'Initialize weights', 'Reduce overfitting'],
    correctAnswer: 1,
    explanation: 'Activation functions introduce non-linearity, enabling neural networks to learn complex, non-linear relationships.'
  },
  {
    id: 'act2',
    question: 'What is the Sigmoid activation function?',
    options: ['f(x) = x', 'f(x) = 1/(1+e^(-x)), outputs between 0 and 1', 'f(x) = max(0,x)', 'f(x) = x²'],
    correctAnswer: 1,
    explanation: 'Sigmoid function: σ(x) = 1/(1+e^(-x)) squashes inputs to (0,1), useful for probabilities.'
  },
  {
    id: 'act3',
    question: 'What is a problem with Sigmoid activation?',
    options: ['Too simple', 'Vanishing gradients for extreme values', 'Too fast', 'No non-linearity'],
    correctAnswer: 1,
    explanation: 'Sigmoid gradients approach zero for large positive/negative inputs, causing vanishing gradient problem in deep networks.'
  },
  {
    id: 'act4',
    question: 'What is the Tanh activation function?',
    options: ['f(x) = 1/(1+e^(-x))', 'f(x) = (e^x - e^(-x))/(e^x + e^(-x)), outputs between -1 and 1', 'f(x) = max(0,x)', 'f(x) = x'],
    correctAnswer: 1,
    explanation: 'Tanh: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x)) squashes inputs to (-1,1), zero-centered unlike sigmoid.'
  },
  {
    id: 'act5',
    question: 'What advantage does Tanh have over Sigmoid?',
    options: ['Faster computation', 'Zero-centered output', 'No vanishing gradient', 'Unbounded output'],
    correctAnswer: 1,
    explanation: 'Tanh is zero-centered (outputs from -1 to 1), which can make optimization easier compared to sigmoid (0 to 1).'
  },
  {
    id: 'act6',
    question: 'What is the ReLU activation function?',
    options: ['f(x) = 1/(1+e^(-x))', 'f(x) = max(0, x)', 'f(x) = x²', 'f(x) = |x|'],
    correctAnswer: 1,
    explanation: 'ReLU (Rectified Linear Unit): f(x) = max(0, x). It outputs x if positive, else 0.'
  },
  {
    id: 'act7',
    question: 'What are advantages of ReLU?',
    options: ['Bounded output', 'Computationally efficient and helps avoid vanishing gradients', 'Always smooth', 'Outputs probabilities'],
    correctAnswer: 1,
    explanation: 'ReLU is simple to compute and has gradient 1 for positive inputs, helping with vanishing gradient problem.'
  },
  {
    id: 'act8',
    question: 'What is the "dying ReLU" problem?',
    options: ['ReLU is too slow', 'Neurons output 0 and stop learning (gradient is 0 for negative inputs)', 'ReLU is too complex', 'No problem exists'],
    correctAnswer: 1,
    explanation: 'If a neuron outputs negative values, ReLU gives 0 with gradient 0, causing the neuron to stop learning permanently.'
  },
  {
    id: 'act9',
    question: 'What is Leaky ReLU?',
    options: ['ReLU with randomness', 'f(x) = x if x>0, else αx (small slope for negative)', 'Slower ReLU', 'ReLU with dropout'],
    correctAnswer: 1,
    explanation: 'Leaky ReLU: f(x) = max(αx, x) where α is small (e.g., 0.01), allowing small gradient for negative inputs.'
  },
  {
    id: 'act10',
    question: 'How does Leaky ReLU address dying ReLU?',
    options: ['It doesn\'t', 'Allows small gradient for negative inputs so neurons can recover', 'Makes neurons larger', 'Increases learning rate'],
    correctAnswer: 1,
    explanation: 'The small slope for negative values means gradients don\'t completely die, neurons can still learn.'
  },
  {
    id: 'act11',
    question: 'What is PReLU (Parametric ReLU)?',
    options: ['Fixed ReLU', 'Leaky ReLU where α is learned during training', 'Sigmoid variant', 'Random activation'],
    correctAnswer: 1,
    explanation: 'PReLU learns the slope α for negative inputs as a parameter during training, rather than fixing it.'
  },
  {
    id: 'act12',
    question: 'What is ELU (Exponential Linear Unit)?',
    options: ['Linear function', 'f(x) = x if x>0, else α(e^x - 1) for smooth negative values', 'Discrete function', 'Step function'],
    correctAnswer: 1,
    explanation: 'ELU: smooth curve for negative inputs using α(e^x - 1), which can lead to faster convergence and robustness.'
  },
  {
    id: 'act13',
    question: 'What is Swish activation?',
    options: ['f(x) = max(0,x)', 'f(x) = x * sigmoid(x), smooth and non-monotonic', 'f(x) = x²', 'f(x) = |x|'],
    correctAnswer: 1,
    explanation: 'Swish: f(x) = x × σ(x). Developed by Google, it\'s smooth and can outperform ReLU in some deep networks.'
  },
  {
    id: 'act14',
    question: 'What is the Softmax activation function?',
    options: ['For hidden layers', 'Converts logits to probability distribution (multi-class output)', 'For binary classification', 'For regression'],
    correctAnswer: 1,
    explanation: 'Softmax: exponentiates and normalizes logits so outputs sum to 1, creating probability distribution over classes.'
  },
  {
    id: 'act15',
    question: 'Where is Softmax typically used?',
    options: ['Hidden layers', 'Output layer for multi-class classification', 'Input layer', 'Everywhere in network'],
    correctAnswer: 1,
    explanation: 'Softmax is used in the output layer for multi-class classification to produce class probabilities.'
  },
  {
    id: 'act16',
    question: 'What is the Linear activation function?',
    options: ['f(x) = max(0,x)', 'f(x) = x (identity function)', 'f(x) = 1/(1+e^(-x))', 'f(x) = 0'],
    correctAnswer: 1,
    explanation: 'Linear activation: f(x) = x, simply passes the input through. Used for regression output layers.'
  },
  {
    id: 'act17',
    question: 'When would you use linear activation?',
    options: ['Always', 'Output layer for regression tasks', 'Hidden layers', 'Never'],
    correctAnswer: 1,
    explanation: 'Linear activation is used in output layers for regression when you want to predict continuous values without bounds.'
  },
  {
    id: 'act18',
    question: 'Can you stack multiple layers with only linear activations?',
    options: ['Yes, makes network deeper', 'No, it collapses to single layer equivalent', 'Only with regularization', 'Only for small networks'],
    correctAnswer: 1,
    explanation: 'Composing linear functions yields another linear function, so multiple linear layers = single linear layer.'
  },
  {
    id: 'act19',
    question: 'What is the gradient of ReLU for positive inputs?',
    options: ['0', '1', 'x', 'e^x'],
    correctAnswer: 1,
    explanation: 'For x > 0, ReLU(x) = x, so gradient is 1. For x ≤ 0, ReLU(x) = 0, so gradient is 0.'
  },
  {
    id: 'act20',
    question: 'Why is ReLU more popular than Sigmoid/Tanh today?',
    options: ['More complex', 'Faster computation and better gradient flow in deep networks', 'Outputs probabilities', 'Older and proven'],
    correctAnswer: 1,
    explanation: 'ReLU is computationally efficient and avoids vanishing gradients, making it the default for modern deep networks.'
  },
  {
    id: 'act21',
    question: 'What is GELU (Gaussian Error Linear Unit)?',
    options: ['ReLU variant', 'Smooth approximation: x * Φ(x) where Φ is Gaussian CDF', 'Sigmoid variant', 'Linear function'],
    correctAnswer: 1,
    explanation: 'GELU: x × Φ(x) where Φ is standard normal CDF. Used in transformers (BERT, GPT), smoother than ReLU.'
  },
  {
    id: 'act22',
    question: 'Can activation functions be learned?',
    options: ['No, always fixed', 'Yes, some like PReLU learn parameters, or use adaptive activations', 'Only in CNNs', 'Only in RNNs'],
    correctAnswer: 1,
    explanation: 'PReLU learns α, and research explores fully learned activation functions, though most use fixed functions.'
  },
  {
    id: 'act23',
    question: 'What happens if you use no activation function?',
    options: ['Faster training', 'Network becomes purely linear, loses expressive power', 'Better accuracy', 'No change'],
    correctAnswer: 1,
    explanation: 'Without activation functions, the network can only learn linear transformations, unable to model complex patterns.'
  },
  {
    id: 'act24',
    question: 'Should you use the same activation for all hidden layers?',
    options: ['No, always vary', 'Commonly yes (e.g., all ReLU), but can experiment', 'Must be different', 'Only for small networks'],
    correctAnswer: 1,
    explanation: 'Typically the same activation (like ReLU) is used for hidden layers for simplicity, but mixing is possible.'
  },
  {
    id: 'act25',
    question: 'What is the main criterion for choosing an activation function?',
    options: ['Random choice', 'Task requirements, gradient behavior, and empirical performance', 'Always use ReLU', 'Alphabetical order'],
    correctAnswer: 1,
    explanation: 'Choice depends on the problem (classification/regression), network depth, and empirical testing on validation data.'
  }
];

// Scenario-based questions for deeper understanding
export const neuralNetworksScenarioQuestions: QuizQuestion[] = [
  {
    id: 'nn-scenario-1',
    question: 'Your deep network (10 layers, sigmoid activations) trains for 100 epochs. Training loss drops from 2.5 to 2.3, then plateaus. Validation loss barely moves. The first 3 layers\' weights barely change. What are the top 3 things to try?',
    options: [
      'Get more data, use a bigger model, train longer',
      'Switch to ReLU activations, add batch normalization, use residual connections (vanishing gradient problem)',
      'Lower learning rate, use SGD instead of Adam, reduce batch size',
      'Add more layers, use dropout everywhere, use L2 regularization'
    ],
    correctAnswer: 1,
    explanation: 'Classic vanishing gradient: sigmoid derivatives shrink exponentially through layers. Early layers can\'t learn. Solutions: ReLU (derivative=1), BatchNorm (stabilizes gradients), residual connections (gradient highways). This is why modern deep networks use ReLU, not sigmoid.'
  },
  {
    id: 'nn-scenario-2',
    question: 'You train an image classifier with BatchNorm. Training accuracy: 95%. When you deploy and test on single images, accuracy drops to 62%. What\'s the most likely issue?',
    options: [
      'Model is overfitting to training data',
      'Forgot to call model.eval() - BatchNorm using wrong statistics',
      'Test images are too different from training',
      'Need data augmentation'
    ],
    correctAnswer: 1,
    explanation: 'BatchNorm behavior differs in train vs eval mode. Training uses batch statistics (unreliable for batch=1). Eval mode uses running averages from training. Forgetting model.eval() is the #1 BatchNorm mistake, causing dramatic performance drop on single examples.'
  },
  {
    id: 'nn-scenario-3',
    question: 'Your binary classifier outputs: [0.51, 0.49, 0.50, 0.52, 0.48] for all samples (basically 0.5 for everything). Training with BCE loss and sigmoid output. Loss decreases slowly but predictions stay near 0.5. What\'s wrong?',
    options: [
      'Learning rate too low - increase by 10x',
      'Model is learning but stuck at trivial solution. Check: class balance, learning rate, model capacity, feature quality',
      'Need more epochs - train longer',
      'BatchNorm is causing the issue'
    ],
    correctAnswer: 1,
    explanation: 'Outputting 0.5 is a trivial "hedge" solution. Possible causes: (1) Extreme class imbalance - model learned "always predict majority", (2) Learning rate too low - no meaningful updates, (3) Insufficient capacity - model too simple, (4) Features don\'t contain signal. Check class distribution first, then try higher LR, bigger model, or feature engineering.'
  },
  {
    id: 'nn-scenario-4',
    question: 'Training loss becomes NaN after 50 iterations. Before NaN, gradients for later layers were ~100, early layers ~0.01. What happened and how to fix?',
    options: [
      'Underfitting - need bigger model',
      'Exploding gradients in later layers. Solutions: lower learning rate (10x), gradient clipping, better initialization',
      'Vanishing gradients - switch to ReLU',
      'Bad data - clean dataset'
    ],
    correctAnswer: 1,
    explanation: 'Gradient ~100 indicates exploding gradients. Eventually causes NaN when weights/gradients become infinite. Fixes: (1) Lower learning rate (0.001 → 0.0001), (2) Gradient clipping (torch.nn.utils.clip_grad_norm_), (3) He initialization, (4) BatchNorm. The vanishing gradients in early layers is a separate issue also needing attention.'
  },
  {
    id: 'nn-scenario-5',
    question: 'Your model with BatchNorm works great with batch size 32 but training becomes unstable with batch size 4 (loss oscillates wildly). Why and what to do?',
    options: [
      'Batch size 4 is too small - just use 32',
      'BatchNorm relies on batch statistics - with batch=4, statistics are too noisy. Use GroupNorm or LayerNorm instead',
      'Lower learning rate for batch size 4',
      'Add more dropout'
    ],
    correctAnswer: 1,
    explanation: 'BatchNorm quality degrades with small batches. Batch=4 gives noisy mean/variance estimates, destabilizing normalization. Solutions: (1) Use GroupNorm (normalizes within channel groups, batch-independent), (2) Use LayerNorm (normalizes across features per sample), (3) Increase batch size if memory allows. GroupNorm is specifically designed for small-batch training.'
  },
  {
    id: 'nn-scenario-6',
    question: 'You use MSE loss for a 10-class classification problem (classes 0-9). The model outputs probabilities via softmax. Training works but accuracy is only 35%. What\'s wrong?',
    options: [
      'Need more data',
      'Using wrong loss function - use CrossEntropyLoss for classification, not MSE',
      'Model architecture is bad',
      'Learning rate is wrong'
    ],
    correctAnswer: 1,
    explanation: 'MSE is for regression, not classification. It treats classes as numbers (class 8 is "closer" to class 9 than class 1), which is nonsensical. MSE also gives poor gradients for softmax outputs. Use CrossEntropyLoss (includes softmax) or NLLLoss (with log_softmax). This single fix often improves accuracy by 30-50%.'
  },
  {
    id: 'nn-scenario-7',
    question: 'Your network has 40% of ReLU neurons permanently outputting 0 for all inputs (dead neurons). Training/validation accuracy is poor (both ~60%). What caused this and how to fix?',
    options: [
      'This is normal for ReLU - no fix needed',
      'Dying ReLU problem from high learning rate or bad initialization. Fix: lower LR, use Leaky ReLU, use He initialization',
      'Need more layers',
      'Batch size is too small'
    ],
    correctAnswer: 1,
    explanation: 'Dying ReLU: neurons stuck outputting 0, gradients permanently 0, can\'t recover. Causes: (1) Learning rate too high - large updates push weights negative, (2) Bad initialization - starts in dead region. Fixes: (1) Lower learning rate (0.01 → 0.001), (2) Leaky ReLU (small negative slope allows recovery), (3) He initialization, (4) BatchNorm. 40% dead neurons severely limits model capacity.'
  },
  {
    id: 'nn-scenario-8',
    question: 'Training an image segmentation model. Training loss: 0.05, validation loss: 0.08. Dice score (evaluation metric): training 0.92, validation 0.65. What\'s the issue?',
    options: [
      'Validation set is harder',
      'Severe overfitting - model memorizes training data. Add regularization: dropout, weight decay, data augmentation, or get more data',
      'Loss function is wrong',
      'Learning rate too high'
    ],
    correctAnswer: 1,
    explanation: 'Large train-validation gap (0.92 vs 0.65 Dice) indicates overfitting. Model learns training-specific patterns, doesn\'t generalize. Solutions: (1) Dropout (0.3-0.5), (2) L2 regularization (weight_decay=0.01), (3) Data augmentation (flips, rotations, color jitter), (4) More training data, (5) Reduce model size, (6) Early stopping. Try multiple techniques together.'
  },
  {
    id: 'nn-scenario-9',
    question: 'You apply softmax before passing outputs to nn.CrossEntropyLoss in PyTorch. Model trains but converges very slowly and final accuracy is 15% below baseline. Why?',
    options: [
      'Need to train longer',
      'CrossEntropyLoss includes softmax internally - applying it twice gives wrong gradients. Pass raw logits to the loss',
      'Learning rate is too low',
      'Model architecture needs improvement'
    ],
    correctAnswer: 1,
    explanation: 'Critical PyTorch gotcha: nn.CrossEntropyLoss = softmax + log + negative log likelihood, all in one numerically stable operation. Applying softmax first: (1) Double-applies softmax (softmax of softmax), (2) Breaks numerical stability, (3) Wrong gradients - prevents proper learning. Always pass raw logits (pre-softmax) to CrossEntropyLoss.'
  },
  {
    id: 'nn-scenario-10',
    question: 'Training a multi-label classifier (images can have multiple labels: "cat", "outdoor", "sunny"). You use softmax + CrossEntropyLoss. Model performs poorly. What\'s wrong?',
    options: [
      'Need more data',
      'Softmax forces mutual exclusivity (probabilities sum to 1). Use independent sigmoid outputs + BCELoss for multi-label',
      'Learning rate too low',
      'Need deeper network'
    ],
    correctAnswer: 1,
    explanation: 'Softmax outputs sum to 1, implying classes are mutually exclusive. For multi-label (non-exclusive classes), this is wrong. Solution: Use K independent sigmoid outputs (one per class) with BCELoss. Each sigmoid outputs independent probability for its class. Image can then have multiple labels with high probability simultaneously.'
  }
];
