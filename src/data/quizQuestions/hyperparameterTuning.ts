import { QuizQuestion } from '../../types';

export const hyperparameterTuningQuestions: QuizQuestion[] = [
  {
    id: 'hpt1',
    question: 'What is the difference between parameters and hyperparameters?',
    options: ['No difference', 'Parameters are learned from data; hyperparameters are set before training', 'Hyperparameters are learned; parameters are fixed', 'They are the same thing'],
    correctAnswer: 1,
    explanation: 'Parameters (like neural network weights) are learned during training from data. Hyperparameters (like learning rate, number of layers) are configuration choices set before training begins.'
  },
  {
    id: 'hpt2',
    question: 'What is grid search?',
    options: ['Random sampling', 'Exhaustively testing all combinations of specified hyperparameter values', 'Gradient-based optimization', 'Manual selection'],
    correctAnswer: 1,
    explanation: 'Grid search creates a grid of hyperparameter values and tests every combination. For example, with 3 learning rates and 4 regularization values, it tests all 12 combinations.'
  },
  {
    id: 'hpt3',
    question: 'What is a key advantage of random search over grid search?',
    options: ['Always finds optimal solution', 'Better explores hyperparameter space, especially when some hyperparameters are more important', 'Faster in all cases', 'No advantage'],
    correctAnswer: 1,
    explanation: 'Random search samples different values each iteration, giving better coverage of important hyperparameters. It\'s particularly effective when only a few hyperparameters significantly impact performance.'
  },
  {
    id: 'hpt4',
    question: 'Which dataset should you use for hyperparameter tuning?',
    options: ['Test set', 'Validation set (or cross-validation on training set)', 'Training set', 'All data together'],
    correctAnswer: 1,
    explanation: 'Always tune on validation data or use cross-validation on training data. Never tune on the test set—it must remain completely unseen until final evaluation.'
  },
  {
    id: 'hpt5',
    question: 'What is Bayesian optimization for hyperparameter tuning?',
    options: ['Random sampling', 'Uses probabilistic model to predict promising hyperparameter regions', 'Grid search variant', 'Manual tuning'],
    correctAnswer: 1,
    explanation: 'Bayesian optimization builds a surrogate model (often Gaussian Process) to predict performance, then samples hyperparameters where the model suggests improvement is likely.'
  },
  {
    id: 'hpt6',
    question: 'Why use logarithmic scale for learning rate search?',
    options: ['No reason', 'Learning rate impact is roughly logarithmic; equal spacing on log scale tests diverse magnitudes', 'Faster computation', 'Required by algorithms'],
    correctAnswer: 1,
    explanation: 'Learning rates like 0.001, 0.01, 0.1 (log scale) explore different behavior regimes better than 0.001, 0.002, 0.003 (linear scale). Performance often changes more between orders of magnitude.'
  },
  {
    id: 'hpt7',
    question: 'What is overfitting to the validation set?',
    options: ['Model memorizes validation data', 'Excessive hyperparameter tuning makes choices specific to validation set quirks', 'Validation accuracy is low', 'No such thing'],
    correctAnswer: 1,
    explanation: 'If you test hundreds of hyperparameter configurations, you may find combinations that work well on validation set by chance, but don\'t generalize. This is why you need a separate test set.'
  },
  {
    id: 'hpt8',
    question: 'What is early stopping in the context of hyperparameter tuning?',
    options: ['Stop training early', 'Stop evaluating poor hyperparameter configurations early to save compute', 'Stop when validation accuracy is high', 'No early stopping'],
    correctAnswer: 1,
    explanation: 'Early stopping during tuning means terminating poorly-performing configurations before full training completes, allocating more budget to promising configurations.'
  },
  {
    id: 'hpt9',
    question: 'Which hyperparameter is typically MOST important for neural networks?',
    options: ['Batch size', 'Learning rate', 'Number of epochs', 'Activation function'],
    correctAnswer: 1,
    explanation: 'Learning rate is usually the single most critical hyperparameter for neural networks. Too high causes instability; too low causes slow or stuck learning.'
  },
  {
    id: 'hpt10',
    question: 'What is nested cross-validation?',
    options: ['Single CV', 'Outer loop for performance estimation, inner loop for hyperparameter tuning', 'No CV', 'Two models'],
    correctAnswer: 1,
    explanation: 'Nested CV uses an outer loop to evaluate model performance and an inner loop to tune hyperparameters, giving unbiased performance estimates.'
  },
  {
    id: 'hpt11',
    question: 'How many hyperparameter combinations does grid search test with 4 hyperparameters, each with 5 values?',
    options: ['20', '625 (5^4)', '25', '100'],
    correctAnswer: 1,
    explanation: 'Grid search tests all combinations: 5 × 5 × 5 × 5 = 625. This exponential growth is the curse of dimensionality for grid search.'
  },
  {
    id: 'hpt12',
    question: 'What is the purpose of cross-validation during hyperparameter tuning?',
    options: ['Speed up training', 'Get more reliable performance estimates, reducing noise', 'Not useful', 'Only for test evaluation'],
    correctAnswer: 1,
    explanation: 'Cross-validation averages performance across multiple train/validation splits, giving more stable estimates and reducing the risk of tuning to a particular validation split\'s noise.'
  },
  {
    id: 'hpt13',
    question: 'What does scikit-learn\'s GridSearchCV do?',
    options: ['Only grid search', 'Performs grid search with cross-validation automatically', 'Manual search', 'Bayesian optimization'],
    correctAnswer: 1,
    explanation: 'GridSearchCV combines grid search with k-fold cross-validation, automatically training and evaluating models for all hyperparameter combinations across all folds.'
  },
  {
    id: 'hpt14',
    question: 'Which hyperparameters should you prioritize tuning for random forests?',
    options: ['Only one parameter', 'n_estimators (number of trees), max_depth, min_samples_split', 'Batch size', 'Learning rate'],
    correctAnswer: 1,
    explanation: 'For random forests: n_estimators (more is usually better), max_depth (controls overfitting), and min_samples_split (also affects overfitting) are most important.'
  },
  {
    id: 'hpt15',
    question: 'What is the interaction effect between hyperparameters?',
    options: ['No interaction', 'Optimal value of one hyperparameter depends on values of others', 'Independent effects', 'Always the same'],
    correctAnswer: 1,
    explanation: 'Hyperparameters often interact: optimal learning rate depends on batch size, optimal dropout depends on model size. Joint tuning captures these interactions.'
  },
  {
    id: 'hpt16',
    question: 'What is the "coarse to fine" strategy?',
    options: ['Only fine search', 'First search wide ranges, then narrow down to promising regions', 'Only coarse search', 'Random strategy'],
    correctAnswer: 1,
    explanation: 'Start with coarse search over wide ranges (e.g., learning rate 0.0001 to 1), identify promising region (e.g., around 0.01), then search finely in that region (e.g., 0.005 to 0.05).'
  },
  {
    id: 'hpt17',
    question: 'What is Hyperband?',
    options: ['Grid search variant', 'Adaptive resource allocation: allocates more budget to promising configurations', 'Random search', 'Manual tuning'],
    correctAnswer: 1,
    explanation: 'Hyperband allocates small budget to many configurations initially, then progressively allocates more resources to the most promising ones, efficiently exploring the space.'
  },
  {
    id: 'hpt18',
    question: 'Why not tune hyperparameters on the test set?',
    options: ['Too slow', 'Test set must remain unseen to give unbiased final performance estimate', 'No reason', 'It\'s recommended'],
    correctAnswer: 1,
    explanation: 'If you tune on test set, you\'re effectively training on it, leading to overly optimistic performance estimates that won\'t generalize to new data.'
  },
  {
    id: 'hpt19',
    question: 'What is the advantage of Optuna over grid search?',
    options: ['No advantage', 'Sample-efficient Bayesian optimization finds good hyperparameters with fewer trials', 'Simpler', 'Always faster'],
    correctAnswer: 1,
    explanation: 'Optuna uses Bayesian optimization to intelligently explore hyperparameter space, typically finding good configurations in 10-50 trials vs hundreds for grid search.'
  },
  {
    id: 'hpt20',
    question: 'How should you handle the exploration-exploitation tradeoff in hyperparameter tuning?',
    options: ['Only exploit', 'Balance trying new regions (exploration) with refining known good regions (exploitation)', 'Only explore', 'No tradeoff'],
    correctAnswer: 1,
    explanation: 'Good tuning strategies balance exploration (trying diverse configurations to find promising regions) with exploitation (refining configurations in known good regions).'
  },
  {
    id: 'hpt21',
    question: 'What is warm starting in hyperparameter tuning?',
    options: ['Random initialization', 'Using knowledge from previous tuning runs or similar tasks', 'Cold start only', 'Not possible'],
    correctAnswer: 1,
    explanation: 'Warm starting initializes search with hyperparameters that worked well on similar tasks, often finding good configurations faster than starting from scratch.'
  },
  {
    id: 'hpt22',
    question: 'Which tool is specifically designed for neural network hyperparameter tuning?',
    options: ['GridSearchCV', 'Keras Tuner', 'Pandas', 'NumPy'],
    correctAnswer: 1,
    explanation: 'Keras Tuner is built for tuning neural network architectures and hyperparameters, supporting various search strategies (random, Bayesian, Hyperband).'
  },
  {
    id: 'hpt23',
    question: 'What is the curse of dimensionality in hyperparameter tuning?',
    options: ['No curse', 'Search space grows exponentially with number of hyperparameters', 'Linear growth', 'Helps tuning'],
    correctAnswer: 1,
    explanation: 'With more hyperparameters, the space explodes exponentially. Grid search becomes impractical; random search or Bayesian methods are needed.'
  },
  {
    id: 'hpt24',
    question: 'When is it acceptable to use test set information during tuning?',
    options: ['Always acceptable', 'Never—test set must be completely held out until final evaluation', 'Sometimes', 'Recommended'],
    correctAnswer: 1,
    explanation: 'Test set is your final, unbiased estimate of real-world performance. Any use during development (even looking at it) introduces bias. Use it once at the very end.'
  },
  {
    id: 'hpt25',
    question: 'What is Ray Tune?',
    options: ['Audio tool', 'Scalable distributed hyperparameter tuning framework', 'Grid search only', 'No such tool'],
    correctAnswer: 1,
    explanation: 'Ray Tune is a scalable hyperparameter tuning library supporting distributed search across multiple machines, with various search algorithms and early stopping strategies.'
  },
  {
    id: 'hpt26',
    question: 'What is the main benefit of successive halving in hyperparameter optimization?',
    options: ['Tests all configurations equally', 'Quickly eliminates poor configurations, allocating more resources to promising ones', 'Only works with neural networks', 'Requires manual intervention'],
    correctAnswer: 1,
    explanation: 'Successive halving trains many configurations with small budgets initially, then iteratively eliminates the worst half and doubles the budget for survivors.'
  },
  {
    id: 'hpt27',
    question: 'When tuning batch size for neural networks, what tradeoff should you consider?',
    options: ['No tradeoff exists', 'Larger batch size: faster training but may reduce generalization', 'Smaller is always better', 'Size doesn\'t matter'],
    correctAnswer: 1,
    explanation: 'Larger batches are computationally efficient but may lead to sharp minima (poor generalization). Smaller batches add noise that can help find flatter, more generalizable minima.'
  },
  {
    id: 'hpt28',
    question: 'What is population-based training (PBT)?',
    options: ['Training one model', 'Evolves population of models, periodically replacing poor performers with mutations of good ones', 'Standard grid search', 'Random search variant'],
    correctAnswer: 1,
    explanation: 'PBT trains a population in parallel, periodically copying weights from top performers to poor performers while mutating their hyperparameters, jointly optimizing hyperparameters and weights.'
  },
  {
    id: 'hpt29',
    question: 'Why should you consider the total computational budget when choosing a tuning strategy?',
    options: ['Budget doesn\'t matter', 'Limited budget favors efficient methods like Bayesian optimization over exhaustive grid search', 'Only matters for neural networks', 'Budget only affects final model'],
    correctAnswer: 1,
    explanation: 'With limited compute, sample-efficient methods (Bayesian optimization, Hyperband) find good configurations faster than grid search, which wastes budget on clearly poor regions.'
  },
  {
    id: 'hpt30',
    question: 'What is transfer learning in the context of hyperparameter tuning?',
    options: ['Not applicable', 'Using optimal hyperparameters from similar tasks as starting point', 'Only for image tasks', 'Random initialization'],
    correctAnswer: 1,
    explanation: 'Hyperparameters that work well on similar tasks (e.g., other image classification problems) often work well on your task, providing a warm start for optimization.'
  }
];
