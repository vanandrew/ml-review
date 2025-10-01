import { QuizQuestion } from '../../types';

export const regularizationQuestions: QuizQuestion[] = [
  {
    id: 'reg1',
    question: 'What is the main purpose of regularization?',
    options: ['Speed up training', 'Prevent overfitting', 'Increase accuracy', 'Reduce memory usage'],
    correctAnswer: 1,
    explanation: 'Regularization prevents overfitting by adding penalties that discourage model complexity.'
  },
  {
    id: 'reg2',
    question: 'Which regularization adds absolute values of weights to the loss?',
    options: ['L1 (Lasso)', 'L2 (Ridge)', 'Elastic Net', 'Dropout'],
    correctAnswer: 0,
    explanation: 'L1 regularization (Lasso) adds the sum of absolute values of weights as a penalty term.'
  },
  {
    id: 'reg3',
    question: 'What unique property does L1 regularization have?',
    options: ['Faster training', 'Drives weights to exactly zero', 'Better accuracy', 'Lower memory'],
    correctAnswer: 1,
    explanation: 'L1 regularization can drive some weights to exactly zero, performing automatic feature selection.'
  },
  {
    id: 'reg4',
    question: 'L2 regularization penalizes:',
    options: ['Absolute weights', 'Squared weights', 'Number of features', 'Training time'],
    correctAnswer: 1,
    explanation: 'L2 regularization (Ridge) adds the sum of squared weights as a penalty term.'
  },
  {
    id: 'reg5',
    question: 'What does the regularization parameter λ (lambda) control?',
    options: ['Learning rate', 'Strength of penalty', 'Number of iterations', 'Batch size'],
    correctAnswer: 1,
    explanation: 'Lambda controls the strength of regularization - higher values mean stronger penalty on complexity.'
  },
  {
    id: 'reg6',
    question: 'If λ = 0, what happens to regularization?',
    options: ['Maximum regularization', 'No regularization', 'Invalid configuration', 'Training fails'],
    correctAnswer: 1,
    explanation: 'When λ = 0, there is no penalty term, so regularization is effectively disabled.'
  },
  {
    id: 'reg7',
    question: 'If λ is very large, what happens to the model?',
    options: ['Overfitting', 'Underfitting', 'Optimal performance', 'No change'],
    correctAnswer: 1,
    explanation: 'Very large λ heavily penalizes complexity, potentially making the model too simple (underfitting).'
  },
  {
    id: 'reg8',
    question: 'Which technique combines L1 and L2 regularization?',
    options: ['Dropout', 'Elastic Net', 'Batch Normalization', 'Early Stopping'],
    correctAnswer: 1,
    explanation: 'Elastic Net regularization combines both L1 and L2 penalties for their complementary benefits.'
  },
  {
    id: 'reg9',
    question: 'Dropout is a regularization technique primarily used in:',
    options: ['Linear regression', 'Decision trees', 'Neural networks', 'K-means clustering'],
    correctAnswer: 2,
    explanation: 'Dropout is a regularization technique specifically designed for neural networks.'
  },
  {
    id: 'reg10',
    question: 'How does dropout work?',
    options: ['Removes features', 'Randomly deactivates neurons during training', 'Reduces learning rate', 'Stops training early'],
    correctAnswer: 1,
    explanation: 'Dropout randomly sets a fraction of neurons to zero during each training iteration, preventing co-adaptation.'
  },
  {
    id: 'reg11',
    question: 'What is typically NOT done during testing/inference with dropout?',
    options: ['Use all neurons', 'Scale activations', 'Apply dropout', 'Make predictions'],
    correctAnswer: 2,
    explanation: 'During testing, dropout is turned off - all neurons are active (with scaled activations).'
  },
  {
    id: 'reg12',
    question: 'Batch normalization can act as:',
    options: ['Only an optimizer', 'Only a regularizer', 'Both optimizer and regularizer', 'Neither'],
    correctAnswer: 2,
    explanation: 'Batch normalization stabilizes training (optimization) and adds slight noise that provides regularization effects.'
  },
  {
    id: 'reg13',
    question: 'Data augmentation is a form of:',
    options: ['Explicit regularization', 'Implicit regularization', 'Not regularization', 'Feature engineering only'],
    correctAnswer: 1,
    explanation: 'Data augmentation implicitly regularizes by increasing data diversity, making the model more robust.'
  },
  {
    id: 'reg14',
    question: 'What is weight decay equivalent to?',
    options: ['L1 regularization', 'L2 regularization', 'Dropout', 'Early stopping'],
    correctAnswer: 1,
    explanation: 'Weight decay is mathematically equivalent to L2 regularization in the context of gradient descent.'
  },
  {
    id: 'reg15',
    question: 'Which regularization is better for feature selection?',
    options: ['L1 (Lasso)', 'L2 (Ridge)', 'Dropout', 'Early stopping'],
    correctAnswer: 0,
    explanation: 'L1 regularization drives weights to exactly zero, effectively selecting features by removing unimportant ones.'
  },
  {
    id: 'reg16',
    question: 'When should you use regularization?',
    options: ['Always', 'When overfitting', 'When underfitting', 'Never'],
    correctAnswer: 1,
    explanation: 'Regularization is used when the model shows signs of overfitting (high variance) to improve generalization.'
  },
  {
    id: 'reg17',
    question: 'L2 regularization encourages weights to be:',
    options: ['Exactly zero', 'Small but non-zero', 'Large', 'Binary'],
    correctAnswer: 1,
    explanation: 'L2 regularization penalizes large weights, encouraging them to be small but not exactly zero.'
  },
  {
    id: 'reg18',
    question: 'What happens to the loss function with regularization?',
    options: ['Loss = Error', 'Loss = Error + Penalty', 'Loss = Error - Penalty', 'Loss = Error × Penalty'],
    correctAnswer: 1,
    explanation: 'Regularized loss adds a penalty term: Loss = Prediction Error + λ × Complexity Penalty.'
  },
  {
    id: 'reg19',
    question: 'Max-norm regularization constrains:',
    options: ['Number of layers', 'Norm of weight vectors', 'Learning rate', 'Batch size'],
    correctAnswer: 1,
    explanation: 'Max-norm regularization constrains the L2 norm of weight vectors to not exceed a maximum value.'
  },
  {
    id: 'reg20',
    question: 'How do you choose the regularization parameter λ?',
    options: ['Always use 0.01', 'Random guess', 'Validation set performance', 'Training set performance'],
    correctAnswer: 2,
    explanation: 'The optimal λ is chosen by evaluating different values on a validation set and selecting the best performing one.'
  }
];
