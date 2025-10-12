import { QuizQuestion } from '../../types';

export const crossValidationQuestions: QuizQuestion[] = [
  {
    id: 'cv-q1',
    question: 'What is the main purpose of cross-validation?',
    options: ['Speed up training', 'Get reliable performance estimates', 'Increase accuracy', 'Reduce overfitting'],
    correctAnswer: 1,
    explanation: 'Cross-validation provides more reliable performance estimates by using multiple train-test splits.'
  },
  {
    id: 'cv-q2',
    question: 'In k-fold cross-validation, how many times is the model trained?',
    options: ['Once', 'k times', 'k-1 times', 'Depends on data size'],
    correctAnswer: 1,
    explanation: 'The model is trained k times, once for each fold serving as the validation set.'
  },
  {
    id: 'cv-q3',
    question: 'What is the typical value of k in k-fold cross-validation?',
    options: ['2', '5 or 10', '100', '1000'],
    correctAnswer: 1,
    explanation: '5-fold or 10-fold cross-validation are most commonly used, balancing computation and reliability.'
  },
  {
    id: 'cv4',
    question: 'In 5-fold cross-validation, what percentage of data is used for validation in each fold?',
    options: ['10%', '20%', '50%', '80%'],
    correctAnswer: 1,
    explanation: 'In 5-fold CV, each fold uses 1/5 = 20% of the data for validation and 80% for training.'
  },
  {
    id: 'cv5',
    question: 'What is Leave-One-Out Cross-Validation (LOOCV)?',
    options: ['k=2', 'k=n (number of samples)', 'k=10', 'No validation'],
    correctAnswer: 1,
    explanation: 'LOOCV is k-fold CV where k equals the number of samples - each sample is held out once as validation.'
  },
  {
    id: 'cv6',
    question: 'What is a disadvantage of LOOCV?',
    options: ['Poor estimates', 'Very computationally expensive', 'Too simple', 'Not applicable to all models'],
    correctAnswer: 1,
    explanation: 'LOOCV requires training the model n times (once per sample), which is computationally very expensive.'
  },
  {
    id: 'cv7',
    question: 'Stratified k-fold cross-validation is important for:',
    options: ['Large datasets', 'Time-series data', 'Imbalanced classification', 'Regression problems'],
    correctAnswer: 2,
    explanation: 'Stratified k-fold maintains class proportions in each fold, crucial for imbalanced classification problems.'
  },
  {
    id: 'cv8',
    question: 'What does cross-validation help prevent?',
    options: ['Underfitting', 'Overfitting to a single validation split', 'Slow training', 'Memory issues'],
    correctAnswer: 1,
    explanation: 'Cross-validation prevents overfitting to a single validation split by averaging performance across multiple splits.'
  },
  {
    id: 'cv9',
    question: 'For time-series data, which CV method is appropriate?',
    options: ['Standard k-fold', 'Stratified k-fold', 'Time-series split (forward chaining)', 'LOOCV'],
    correctAnswer: 2,
    explanation: 'Time-series split maintains temporal order, training on past data and validating on future data.'
  },
  {
    id: 'cv10',
    question: 'What is the final CV score typically computed as?',
    options: ['Best fold score', 'Worst fold score', 'Average across all folds', 'Median of fold scores'],
    correctAnswer: 2,
    explanation: 'The CV score is typically the average (mean) performance across all k folds.'
  },
  {
    id: 'cv11',
    question: 'When should you use cross-validation instead of a simple train-test split?',
    options: ['Always', 'With limited data', 'With massive datasets', 'Never'],
    correctAnswer: 1,
    explanation: 'Cross-validation is especially valuable with limited data, providing more reliable estimates than a single split.'
  },
  {
    id: 'cv12',
    question: 'Can cross-validation be used for hyperparameter tuning?',
    options: ['No, only for testing', 'Yes, it provides reliable validation', 'Only with LOOCV', 'Only for neural networks'],
    correctAnswer: 1,
    explanation: 'Cross-validation is commonly used for hyperparameter tuning, providing robust performance estimates for each configuration.'
  },
  {
    id: 'cv13',
    question: 'What is nested cross-validation?',
    options: ['Running CV twice', 'CV within CV for unbiased hyperparameter tuning', 'Using multiple models', 'Parallel CV'],
    correctAnswer: 1,
    explanation: 'Nested CV uses an outer CV loop for evaluation and inner CV loops for hyperparameter tuning to get unbiased estimates.'
  },
  {
    id: 'cv14',
    question: 'What is the main disadvantage of cross-validation?',
    options: ['Poor accuracy', 'Computational cost', 'Cannot use all data', 'Only works for classification'],
    correctAnswer: 1,
    explanation: 'The main disadvantage is computational cost - training k models instead of one.'
  },
  {
    id: 'cv15',
    question: 'After cross-validation, what model do you deploy?',
    options: ['Best fold model', 'Average of all models', 'Model trained on all data', 'Random fold model'],
    correctAnswer: 2,
    explanation: 'After CV evaluation, you typically train a final model on all available data for deployment.'
  },
  {
    id: 'cv16',
    question: 'What is group k-fold cross-validation?',
    options: ['Random grouping', 'Ensures samples from same group stay together', 'Groups by class', 'Groups by size'],
    correctAnswer: 1,
    explanation: 'Group k-fold ensures samples belonging to the same group don\'t appear in both train and validation sets.'
  },
  {
    id: 'cv17',
    question: 'What does a high variance across CV folds indicate?',
    options: ['Good performance', 'Model is unstable', 'Perfect fit', 'Need more folds'],
    correctAnswer: 1,
    explanation: 'High variance across folds suggests the model performance is unstable and sensitive to the training data.'
  },
  {
    id: 'cv18',
    question: 'Bootstrap aggregating (bagging) is related to which CV concept?',
    options: ['k-fold', 'LOOCV', 'Repeated sampling with replacement', 'Stratification'],
    correctAnswer: 2,
    explanation: 'Bootstrap methods sample with replacement, creating multiple training sets similar to CV but with overlap.'
  },
  {
    id: 'cv19',
    question: 'What is repeated k-fold cross-validation?',
    options: ['Training k times', 'Running k-fold CV multiple times with different splits', 'Using k models', 'Training on k datasets'],
    correctAnswer: 1,
    explanation: 'Repeated k-fold runs the entire k-fold CV process multiple times with different random splits for more robust estimates.'
  },
  {
    id: 'cv20',
    question: 'For image classification with multiple images per patient, what CV method prevents data leakage?',
    options: ['Standard k-fold', 'Stratified k-fold', 'Group k-fold by patient', 'LOOCV'],
    correctAnswer: 2,
    explanation: 'Group k-fold by patient ensures all images from one patient stay in the same fold, preventing data leakage.'
  },
  {
    id: 'cv21',
    question: 'What is the bias-variance tradeoff consideration in choosing k for k-fold CV?',
    options: ['Higher k reduces bias but increases variance', 'Higher k increases both bias and variance', 'Lower k reduces both bias and variance', 'k doesn\'t affect bias or variance'],
    correctAnswer: 0,
    explanation: 'Higher k (more folds) uses more training data per fold, reducing bias in estimates but increasing variance due to higher correlation between folds.'
  },
  {
    id: 'cv22',
    question: 'Why might LOOCV have high variance in performance estimates?',
    options: ['Too much data', 'Training sets are highly correlated', 'Not enough folds', 'Random initialization'],
    correctAnswer: 1,
    explanation: 'In LOOCV, training sets overlap by n-2 samples, making them highly similar and correlated, leading to high variance in estimates.'
  },
  {
    id: 'cv23',
    question: 'When is cross-validation particularly important?',
    options: ['With millions of training samples', 'With limited data and many hyperparameters to tune', 'For simple linear models', 'When you have unlimited compute'],
    correctAnswer: 1,
    explanation: 'CV is crucial when data is limited and you need reliable estimates for comparing many hyperparameter configurations.'
  },
  {
    id: 'cv24',
    question: 'What is Monte Carlo cross-validation?',
    options: ['Same as k-fold', 'Repeated random train-test splits', 'A specific k value', 'Cross-validation for games'],
    correctAnswer: 1,
    explanation: 'Monte Carlo CV repeatedly creates random train-test splits (not necessarily exhaustive like k-fold) and averages results.'
  },
  {
    id: 'cv25',
    question: 'In nested cross-validation, what is the purpose of the outer loop?',
    options: ['Hyperparameter tuning', 'Unbiased performance estimation', 'Feature selection', 'Data augmentation'],
    correctAnswer: 1,
    explanation: 'The outer CV loop provides an unbiased estimate of model performance, while inner loops handle hyperparameter tuning.'
  }
];
