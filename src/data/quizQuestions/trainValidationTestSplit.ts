import { QuizQuestion } from '../../types';

export const trainValidationTestSplitQuestions: QuizQuestion[] = [
  {
    id: 'split1',
    question: 'What is the primary purpose of the test set?',
    options: ['Training the model', 'Tuning hyperparameters', 'Final unbiased evaluation', 'Feature selection'],
    correctAnswer: 2,
    explanation: 'The test set provides an unbiased estimate of model performance on unseen data and should only be used once at the end.'
  },
  {
    id: 'split2',
    question: 'Which split is used for hyperparameter tuning?',
    options: ['Training set', 'Validation set', 'Test set', 'All of them'],
    correctAnswer: 1,
    explanation: 'The validation set is specifically used for hyperparameter tuning and model selection during development.'
  },
  {
    id: 'split3',
    question: 'What is a common ratio for train-validation-test split?',
    options: ['50-25-25', '70-15-15', '90-5-5', '33-33-33'],
    correctAnswer: 1,
    explanation: '70-15-15 is a common balanced approach for medium-sized datasets, though ratios vary based on data availability.'
  },
  {
    id: 'split4',
    question: 'Why should you NOT use the test set multiple times?',
    options: ['It\'s too slow', 'It causes data leakage', 'It inflates performance estimates', 'It requires too much memory'],
    correctAnswer: 2,
    explanation: 'Using the test set multiple times leads to indirect overfitting - you optimize for the test set, inflating performance estimates.'
  },
  {
    id: 'split5',
    question: 'What happens if you train on the validation set?',
    options: ['Better generalization', 'Data leakage occurs', 'Faster training', 'Lower variance'],
    correctAnswer: 1,
    explanation: 'Training on the validation set causes data leakage, making performance estimates unreliable since the model has seen the data.'
  },
  {
    id: 'split6',
    question: 'For very large datasets (millions of examples), which split ratio is often used?',
    options: ['60-20-20', '80-10-10', '98-1-1', '70-15-15'],
    correctAnswer: 2,
    explanation: 'With millions of examples, even 1% provides sufficient validation/test samples, so more data can be used for training.'
  },
  {
    id: 'split7',
    question: 'What is stratified sampling in data splitting?',
    options: ['Random selection', 'Preserving class proportions', 'Sequential selection', 'Clustering-based selection'],
    correctAnswer: 1,
    explanation: 'Stratified sampling maintains the same class distribution across train, validation, and test sets, crucial for imbalanced datasets.'
  },
  {
    id: 'split8',
    question: 'When should you create your train-validation-test splits?',
    options: ['After feature engineering', 'After seeing all data', 'Before any data exploration', 'During model training'],
    correctAnswer: 2,
    explanation: 'Splits should be created at the very beginning to prevent any information leakage from validation/test sets into the training process.'
  },
  {
    id: 'split9',
    question: 'What problem occurs if temporal data is split randomly?',
    options: ['Class imbalance', 'Data leakage from future', 'Reduced training size', 'Increased variance'],
    correctAnswer: 1,
    explanation: 'Random splitting of temporal data causes data leakage - the model learns from future examples to predict the past.'
  },
  {
    id: 'split10',
    question: 'For time-series data, how should you split the data?',
    options: ['Randomly', 'Stratified', 'Chronologically', 'By clusters'],
    correctAnswer: 2,
    explanation: 'Time-series data must be split chronologically - train on past, validate on recent past, test on most recent to avoid data leakage.'
  },
  {
    id: 'split11',
    question: 'What is the purpose of having both validation and test sets?',
    options: ['Doubling the data', 'Validation for tuning, test for final unbiased estimate', 'Parallel processing', 'Redundancy'],
    correctAnswer: 1,
    explanation: 'Validation is used iteratively for hyperparameter tuning (becomes biased), while test remains untouched for unbiased final evaluation.'
  },
  {
    id: 'split12',
    question: 'If you only have 100 samples, what approach is better than a fixed split?',
    options: ['Use all for training', 'K-fold cross-validation', 'Use 90-10 split', 'Use bootstrap sampling'],
    correctAnswer: 1,
    explanation: 'With limited data, k-fold cross-validation provides more reliable performance estimates than a single fixed split.'
  },
  {
    id: 'split13',
    question: 'What is holdout validation?',
    options: ['Using all data', 'Single train-test split', 'K-fold cross-validation', 'Leave-one-out'],
    correctAnswer: 1,
    explanation: 'Holdout validation is a single fixed split of data into training and test (or validation) sets.'
  },
  {
    id: 'split14',
    question: 'When training neural networks, which set determines when to stop training (early stopping)?',
    options: ['Training set', 'Validation set', 'Test set', 'All sets'],
    correctAnswer: 1,
    explanation: 'Early stopping monitors validation set performance to halt training when validation loss stops improving, preventing overfitting.'
  },
  {
    id: 'split15',
    question: 'What is data leakage in the context of splitting?',
    options: ['Losing data during processing', 'Information from test set influencing training', 'Memory overflow', 'Network data transfer'],
    correctAnswer: 1,
    explanation: 'Data leakage occurs when information from validation/test sets (directly or indirectly) influences the training process.'
  },
  {
    id: 'split16',
    question: 'For grouped data (e.g., multiple images per patient), how should you split?',
    options: ['Random split', 'By groups (patients)', 'Stratified split', 'Sequential split'],
    correctAnswer: 1,
    explanation: 'Split by groups to avoid the same patient appearing in both training and test sets, which would cause data leakage.'
  },
  {
    id: 'split17',
    question: 'What does "touching the test set" mean?',
    options: ['Data corruption', 'Using it for anything before final evaluation', 'Physical access', 'Data augmentation'],
    correctAnswer: 1,
    explanation: '"Touching the test set" means using it for any purpose (viewing, tuning, analyzing) before final evaluation, which biases results.'
  },
  {
    id: 'split18',
    question: 'If validation performance is much better than test performance, what might be wrong?',
    options: ['Nothing, this is normal', 'Possible data leakage or overfitting to validation', 'Test set is too hard', 'Training was insufficient'],
    correctAnswer: 1,
    explanation: 'Large gap between validation and test performance suggests overfitting to the validation set or data leakage issues.'
  },
  {
    id: 'split19',
    question: 'What is the training set used for?',
    options: ['Model evaluation', 'Hyperparameter tuning', 'Learning model parameters', 'Final testing'],
    correctAnswer: 2,
    explanation: 'The training set is used to learn model parameters (weights) through optimization algorithms like gradient descent.'
  },
  {
    id: 'split20',
    question: 'In production ML, what additional split is sometimes maintained?',
    options: ['Development set', 'Shadow set', 'Holdout set for monitoring production', 'Cache set'],
    correctAnswer: 2,
    explanation: 'A production holdout set is maintained to continuously monitor model performance in production and detect degradation.'
  }
];
