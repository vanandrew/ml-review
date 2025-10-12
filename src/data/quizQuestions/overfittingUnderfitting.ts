import { QuizQuestion } from '../../types';

export const overfittingUnderfittingQuestions: QuizQuestion[] = [
  {
    id: 'ou1',
    question: 'What is overfitting?',
    options: ['Model too simple', 'Model memorizes training data', 'Model has high bias', 'Model trains too slowly'],
    correctAnswer: 1,
    explanation: 'Overfitting occurs when a model learns the training data too well, including noise, failing to generalize to new data.'
  },
  {
    id: 'ou2',
    question: 'What is a key sign of underfitting?',
    options: ['High training accuracy, low test accuracy', 'Low training accuracy, low test accuracy', 'Perfect accuracy', 'High variance'],
    correctAnswer: 1,
    explanation: 'Underfitting shows poor performance on both training and test sets, indicating the model is too simple.'
  },
  {
    id: 'ou3',
    question: 'Which technique helps prevent overfitting?',
    options: ['Adding more parameters', 'Regularization', 'Training longer', 'Using all features'],
    correctAnswer: 1,
    explanation: 'Regularization adds penalties to model complexity, preventing overfitting by constraining parameter values.'
  },
  {
    id: 'ou4',
    question: 'A model with 100% training accuracy but 60% test accuracy is likely:',
    options: ['Underfitting', 'Overfitting', 'Well-fitted', 'Untrainable'],
    correctAnswer: 1,
    explanation: 'Perfect training accuracy with poor test accuracy is a classic sign of overfitting - the model memorized training data.'
  },
  {
    id: 'ou5',
    question: 'How can you fix underfitting?',
    options: ['Add regularization', 'Reduce features', 'Increase model complexity', 'Use less data'],
    correctAnswer: 2,
    explanation: 'Underfitting is fixed by increasing model capacity - more features, more parameters, or more complex architectures.'
  },
  {
    id: 'ou6',
    question: 'What happens when you train a linear model on highly non-linear data?',
    options: ['Overfitting', 'Underfitting', 'Perfect fit', 'Model crashes'],
    correctAnswer: 1,
    explanation: 'A linear model lacks capacity to capture non-linear patterns, resulting in underfitting (high bias).'
  },
  {
    id: 'ou7',
    question: 'Which is NOT a sign of overfitting?',
    options: ['Large gap between train and test error', 'Model performs well on test data', 'High variance', 'Memorizing training examples'],
    correctAnswer: 1,
    explanation: 'Good test performance indicates proper generalization, not overfitting. Overfitting shows poor test performance.'
  },
  {
    id: 'ou8',
    question: 'Adding more training data typically helps with:',
    options: ['Underfitting', 'Overfitting', 'Both equally', 'Neither'],
    correctAnswer: 1,
    explanation: 'More training data helps reduce overfitting by providing more examples of the underlying distribution.'
  },
  {
    id: 'ou9',
    question: 'A very deep decision tree without pruning is prone to:',
    options: ['Underfitting', 'Overfitting', 'Neither', 'Both'],
    correctAnswer: 1,
    explanation: 'Deep unpruned decision trees can create highly specific rules that memorize training data, leading to overfitting.'
  },
  {
    id: 'ou10',
    question: 'What is the relationship between model capacity and overfitting?',
    options: ['Inversely proportional', 'No relationship', 'Higher capacity increases overfitting risk', 'Lower capacity increases overfitting risk'],
    correctAnswer: 2,
    explanation: 'Higher model capacity (more parameters) increases the risk of overfitting if not properly regularized.'
  },
  {
    id: 'ou11',
    question: 'Feature selection can help prevent:',
    options: ['Underfitting only', 'Overfitting only', 'Both overfitting and underfitting', 'Neither'],
    correctAnswer: 1,
    explanation: 'Feature selection reduces model complexity by removing irrelevant features, helping prevent overfitting.'
  },
  {
    id: 'ou12',
    question: 'Which learning curve pattern indicates overfitting?',
    options: ['Both curves high and close', 'Both curves low and close', 'Training error low, validation error high with large gap', 'Both curves decreasing together'],
    correctAnswer: 2,
    explanation: 'A large gap between low training error and high validation error indicates the model is overfitting to training data.'
  },
  {
    id: 'ou13',
    question: 'Dropout in neural networks helps with:',
    options: ['Underfitting', 'Overfitting', 'Faster training', 'Memory reduction'],
    correctAnswer: 1,
    explanation: 'Dropout is a regularization technique that prevents overfitting by randomly deactivating neurons during training.'
  },
  {
    id: 'ou14',
    question: 'What is the "sweet spot" between overfitting and underfitting?',
    options: ['Maximum complexity', 'Minimum complexity', 'Optimal complexity that generalizes well', 'No such spot exists'],
    correctAnswer: 2,
    explanation: 'The sweet spot is the optimal model complexity that balances bias and variance for best generalization.'
  },
  {
    id: 'ou15',
    question: 'If both training and test error are high, you should:',
    options: ['Add regularization', 'Increase model complexity', 'Remove features', 'Stop training'],
    correctAnswer: 1,
    explanation: 'High error on both sets indicates underfitting - the model needs more capacity to capture the patterns.'
  },
  {
    id: 'ou16',
    question: 'Which scenario suggests your model is well-fitted?',
    options: ['Train: 99%, Test: 60%', 'Train: 60%, Test: 58%', 'Train: 85%, Test: 83%', 'Train: 100%, Test: 50%'],
    correctAnswer: 2,
    explanation: 'Train: 85%, Test: 83% shows good performance with a small gap, indicating good generalization without overfitting.'
  },
  {
    id: 'ou17',
    question: 'Ensemble methods like Random Forest help reduce:',
    options: ['Underfitting only', 'Overfitting only', 'Both', 'Neither'],
    correctAnswer: 1,
    explanation: 'Ensemble methods like Random Forest average multiple models to reduce variance and prevent overfitting of individual trees.'
  },
  {
    id: 'ou18',
    question: 'What does "noise fitting" refer to?',
    options: ['Underfitting', 'Overfitting', 'Data preprocessing', 'Audio processing'],
    correctAnswer: 1,
    explanation: 'Noise fitting is another term for overfitting - the model learns random noise in training data instead of true patterns.'
  },
  {
    id: 'ou19',
    question: 'Early stopping prevents overfitting by:',
    options: ['Adding regularization', 'Stopping training when validation performance degrades', 'Removing features', 'Using less data'],
    correctAnswer: 1,
    explanation: 'Early stopping monitors validation performance and halts training before the model overfits to training data.'
  },
  {
    id: 'ou20',
    question: 'A model that makes the same prediction for all inputs is:',
    options: ['Overfitting', 'Underfitting', 'Well-fitted', 'Impossible'],
    correctAnswer: 1,
    explanation: 'Making the same prediction regardless of input indicates severe underfitting - the model hasn\'t learned any patterns.'
  },
  {
    id: 'ou21',
    question: 'What is double descent in deep learning?',
    options: ['Training loss increases twice', 'Test error can decrease again after initial increase with model size', 'Using two learning rates', 'Training for two epochs'],
    correctAnswer: 1,
    explanation: 'Double descent is a phenomenon where test error first decreases, then increases (classical overfitting), then decreases again as model capacity continues to grow.'
  },
  {
    id: 'ou22',
    question: 'Which visualization best helps identify overfitting?',
    options: ['Confusion matrix', 'Learning curves (train vs validation loss)', 'ROC curve', 'Feature importance plot'],
    correctAnswer: 1,
    explanation: 'Learning curves showing training vs validation loss over time clearly reveal overfitting when training loss decreases but validation loss increases.'
  },
  {
    id: 'ou23',
    question: 'Data augmentation helps prevent overfitting by:',
    options: ['Reducing model size', 'Artificially increasing training data diversity', 'Removing outliers', 'Simplifying the model'],
    correctAnswer: 1,
    explanation: 'Data augmentation creates variations of training samples, effectively increasing data diversity and making the model more robust to variations.'
  },
  {
    id: 'ou24',
    question: 'What is "benign overfitting"?',
    options: ['Overfitting that doesn\'t hurt performance', 'Overfitting in interpolating models that still generalize well', 'Overfitting on training data only', 'A type of regularization'],
    correctAnswer: 1,
    explanation: 'Benign overfitting occurs in overparameterized models (like large neural networks) that fit training data perfectly yet still generalize well.'
  },
  {
    id: 'ou25',
    question: 'If your model has high training error and the error doesn\'t decrease with more training, you likely have:',
    options: ['Overfitting', 'Underfitting', 'Vanishing gradients or underfitting', 'Perfect model'],
    correctAnswer: 2,
    explanation: 'Persistently high training error suggests the model cannot learn from the data, indicating underfitting or optimization issues like vanishing gradients.'
  }
];
