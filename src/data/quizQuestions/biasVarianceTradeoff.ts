import { QuizQuestion } from '../../types';

export const biasVarianceTradeoffQuestions: QuizQuestion[] = [
  {
    id: 'bv1',
    question: 'What does high bias typically lead to?',
    options: ['Overfitting', 'Underfitting', 'Perfect fit', 'High variance'],
    correctAnswer: 1,
    explanation: 'High bias means the model is too simple to capture the underlying patterns, leading to underfitting.'
  },
  {
    id: 'bv2',
    question: 'A model that performs well on training data but poorly on test data likely has:',
    options: ['High bias', 'High variance', 'Low bias and low variance', 'Irreducible error'],
    correctAnswer: 1,
    explanation: 'High variance models are sensitive to the training data and overfit, performing well on training but poorly on new data.'
  },
  {
    id: 'bv3',
    question: 'Which technique helps reduce variance?',
    options: ['Adding more features', 'Increasing model complexity', 'Regularization', 'Removing training data'],
    correctAnswer: 2,
    explanation: 'Regularization penalizes model complexity and helps reduce variance by preventing overfitting.'
  },
  {
    id: 'bv4',
    question: 'What happens to bias as model complexity increases?',
    options: ['Bias increases', 'Bias decreases', 'Bias stays the same', 'Bias becomes undefined'],
    correctAnswer: 1,
    explanation: 'As model complexity increases, the model can fit more complex patterns, reducing bias.'
  },
  {
    id: 'bv5',
    question: 'What is irreducible error?',
    options: ['Error from high bias', 'Error from high variance', 'Error inherent in the data/noise', 'Error from poor optimization'],
    correctAnswer: 2,
    explanation: 'Irreducible error is the noise inherent in the data that no model can eliminate, regardless of its quality.'
  },
  {
    id: 'bv6',
    question: 'Which of these helps reduce bias?',
    options: ['L2 regularization', 'Early stopping', 'Adding more features', 'Using less training data'],
    correctAnswer: 2,
    explanation: 'Adding more features gives the model more information to capture complex patterns, reducing bias.'
  },
  {
    id: 'bv7',
    question: 'A linear model used on highly non-linear data will likely have:',
    options: ['Low bias, low variance', 'High bias, low variance', 'Low bias, high variance', 'High bias, high variance'],
    correctAnswer: 1,
    explanation: 'A linear model lacks the capacity to fit non-linear patterns (high bias) but is stable across datasets (low variance).'
  },
  {
    id: 'bv8',
    question: 'What is the goal in bias-variance tradeoff?',
    options: ['Minimize bias only', 'Minimize variance only', 'Minimize total error', 'Maximize model complexity'],
    correctAnswer: 2,
    explanation: 'The goal is to minimize total error, which is the sum of bias, variance, and irreducible error.'
  },
  {
    id: 'bv9',
    question: 'Cross-validation helps with bias-variance tradeoff by:',
    options: ['Eliminating bias', 'Eliminating variance', 'Providing better performance estimates', 'Increasing training data'],
    correctAnswer: 2,
    explanation: 'Cross-validation provides more reliable estimates of model performance, helping choose the right complexity level.'
  },
  {
    id: 'bv10',
    question: 'Which ensemble method primarily reduces variance?',
    options: ['Boosting', 'Bagging', 'Stacking', 'Cascading'],
    correctAnswer: 1,
    explanation: 'Bagging (Bootstrap Aggregating) reduces variance by averaging predictions from multiple models trained on different data subsets.'
  },
  {
    id: 'bv11',
    question: 'What effect does increasing training data typically have?',
    options: ['Increases bias', 'Reduces variance', 'Increases variance', 'No effect on either'],
    correctAnswer: 1,
    explanation: 'More training data reduces variance by providing a more representative sample of the underlying distribution.'
  },
  {
    id: 'bv12',
    question: 'A very deep decision tree without pruning is likely to have:',
    options: ['High bias, low variance', 'Low bias, high variance', 'Low bias, low variance', 'High bias, high variance'],
    correctAnswer: 1,
    explanation: 'Deep unpruned trees can fit training data very closely (low bias) but are sensitive to small changes (high variance).'
  },
  {
    id: 'bv13',
    question: 'Which learning curve pattern indicates high bias?',
    options: ['Large gap between training and validation error', 'Both errors are high and converged', 'Training error is zero', 'Validation error decreases rapidly'],
    correctAnswer: 1,
    explanation: 'High bias is indicated when both training and validation errors are high and close together, showing the model cannot fit even the training data well.'
  },
  {
    id: 'bv14',
    question: 'Dropout in neural networks primarily helps reduce:',
    options: ['Bias', 'Variance', 'Irreducible error', 'Learning rate'],
    correctAnswer: 1,
    explanation: 'Dropout is a regularization technique that reduces variance by preventing co-adaptation of neurons.'
  },
  {
    id: 'bv15',
    question: 'What is the "sweet spot" in bias-variance tradeoff?',
    options: ['Zero bias', 'Zero variance', 'Minimum total error', 'Maximum complexity'],
    correctAnswer: 2,
    explanation: 'The sweet spot balances bias and variance to achieve minimum total error, which is the test/generalization error.'
  },
  {
    id: 'bv16',
    question: 'Feature engineering that adds polynomial features will:',
    options: ['Increase bias, decrease variance', 'Decrease bias, increase variance', 'Decrease both', 'Increase both'],
    correctAnswer: 1,
    explanation: 'Polynomial features increase model capacity, reducing bias but potentially increasing variance if not regularized.'
  },
  {
    id: 'bv17',
    question: 'Which statement about the bias-variance decomposition is TRUE?',
    options: ['Total error = bias + variance', 'Total error = bias × variance', 'Total error = bias² + variance + irreducible error', 'Total error = bias - variance'],
    correctAnswer: 2,
    explanation: 'The mathematical decomposition shows: Expected test error = bias² + variance + irreducible error.'
  },
  {
    id: 'bv18',
    question: 'Early stopping in neural network training primarily prevents:',
    options: ['Underfitting', 'Overfitting', 'Convergence', 'Gradient descent'],
    correctAnswer: 1,
    explanation: 'Early stopping halts training when validation performance stops improving, preventing overfitting (high variance).'
  },
  {
    id: 'bv19',
    question: 'L1 regularization (Lasso) can reduce variance by:',
    options: ['Adding more features', 'Performing feature selection', 'Increasing model complexity', 'Eliminating bias'],
    correctAnswer: 1,
    explanation: 'L1 regularization drives some weights to zero, effectively performing feature selection and reducing model complexity/variance.'
  },
  {
    id: 'bv20',
    question: 'A model with perfect training accuracy but poor test accuracy suffers from:',
    options: ['High bias', 'High variance', 'Irreducible error', 'Insufficient training'],
    correctAnswer: 1,
    explanation: 'This is a classic sign of overfitting due to high variance - the model memorizes training data but fails to generalize.'
  },
  {
    id: 'bv21',
    question: 'Which ensemble technique primarily reduces bias?',
    options: ['Bagging', 'Boosting', 'Stacking with same model type', 'Random dropout'],
    correctAnswer: 1,
    explanation: 'Boosting sequentially trains weak learners that focus on mistakes, primarily reducing bias by creating a strong learner.'
  },
  {
    id: 'bv22',
    question: 'In the context of neural networks, which increases model capacity and reduces bias?',
    options: ['Adding dropout layers', 'Increasing layer width or depth', 'Increasing batch size', 'Decreasing learning rate'],
    correctAnswer: 1,
    explanation: 'Increasing network width (more neurons) or depth (more layers) increases capacity, allowing the model to fit more complex patterns and reduce bias.'
  },
  {
    id: 'bv23',
    question: 'What is the typical behavior of bias and variance as you increase model training time?',
    options: ['Both increase', 'Bias increases, variance decreases', 'Bias decreases, variance increases', 'Both decrease initially'],
    correctAnswer: 2,
    explanation: 'With more training, the model fits the training data better (reducing bias) but may start overfitting (increasing variance).'
  },
  {
    id: 'bv24',
    question: 'In a bias-variance tradeoff plot, where is the optimal model complexity?',
    options: ['At minimum bias', 'At minimum variance', 'At minimum total error', 'At maximum complexity'],
    correctAnswer: 2,
    explanation: 'The optimal complexity is where total error (bias² + variance + irreducible error) is minimized, balancing both components.'
  },
  {
    id: 'bv25',
    question: 'Which statement about ensemble methods is TRUE?',
    options: ['Ensembles always increase both bias and variance', 'Ensembles can reduce variance without increasing bias', 'Ensembles only work with linear models', 'Ensembles increase irreducible error'],
    correctAnswer: 1,
    explanation: 'Ensemble methods like bagging can significantly reduce variance by averaging multiple models without substantially increasing bias.'
  }
];
