import { QuizQuestion } from '../../types';

// Decision Trees - 20 questions
export const decisionTreesQuestions: QuizQuestion[] = [
  {
    id: 'dt1',
    question: 'What is the main goal of a decision tree algorithm?',
    options: ['Minimize variance', 'Recursively split data to create pure subsets', 'Find linear boundaries', 'Cluster similar data'],
    correctAnswer: 1,
    explanation: 'Decision trees recursively partition the data into subsets that are increasingly homogeneous (pure) with respect to the target variable.'
  },
  {
    id: 'dt2',
    question: 'What is entropy in the context of decision trees?',
    options: ['Tree depth', 'Measure of impurity/disorder', 'Number of nodes', 'Prediction accuracy'],
    correctAnswer: 1,
    explanation: 'Entropy measures the impurity or disorder in a dataset. Lower entropy means more homogeneous data.'
  },
  {
    id: 'dt3',
    question: 'What is information gain?',
    options: ['Number of features', 'Reduction in entropy after a split', 'Tree accuracy', 'Node count'],
    correctAnswer: 1,
    explanation: 'Information gain measures the reduction in entropy achieved by splitting on a particular attribute.'
  },
  {
    id: 'dt4',
    question: 'What is the Gini impurity?',
    options: ['Tree depth metric', 'Measure of how often a randomly chosen element would be incorrectly labeled', 'Accuracy score', 'Pruning threshold'],
    correctAnswer: 1,
    explanation: 'Gini impurity measures the probability of incorrectly classifying a randomly chosen element if it was randomly labeled according to the class distribution.'
  },
  {
    id: 'dt5',
    question: 'What is the difference between ID3 and CART algorithms?',
    options: ['No difference', 'ID3 uses information gain, CART uses Gini or variance', 'ID3 is for regression only', 'CART is only for classification'],
    correctAnswer: 1,
    explanation: 'ID3 uses information gain (entropy-based) for splitting, while CART can use Gini impurity for classification or variance reduction for regression.'
  },
  {
    id: 'dt6',
    question: 'What is pruning in decision trees?',
    options: ['Adding more nodes', 'Removing nodes to reduce overfitting', 'Increasing tree depth', 'Feature selection'],
    correctAnswer: 1,
    explanation: 'Pruning removes branches from the tree to reduce complexity and prevent overfitting, improving generalization.'
  },
  {
    id: 'dt7',
    question: 'What is pre-pruning?',
    options: ['Pruning after tree is built', 'Stopping tree growth early based on criteria', 'Removing all leaves', 'Feature preprocessing'],
    correctAnswer: 1,
    explanation: 'Pre-pruning (early stopping) halts tree construction based on criteria like maximum depth or minimum samples per node.'
  },
  {
    id: 'dt8',
    question: 'What is post-pruning?',
    options: ['Pruning during growth', 'Growing full tree then removing branches', 'Never pruning', 'Pre-processing data'],
    correctAnswer: 1,
    explanation: 'Post-pruning grows a complete tree then removes branches that provide little improvement, often using cross-validation.'
  },
  {
    id: 'dt9',
    question: 'What is a major disadvantage of decision trees?',
    options: ['Too slow', 'Prone to overfitting', 'Cannot handle categorical data', 'Requires normalization'],
    correctAnswer: 1,
    explanation: 'Decision trees are prone to overfitting, especially when grown deep without pruning, as they can memorize training data.'
  },
  {
    id: 'dt10',
    question: 'How do decision trees handle categorical features?',
    options: ['Cannot handle them', 'Naturally, by testing category membership', 'Must convert to continuous first', 'Only with one-hot encoding'],
    correctAnswer: 1,
    explanation: 'Decision trees can naturally handle categorical features by creating splits based on category membership.'
  },
  {
    id: 'dt11',
    question: 'What is the splitting criterion for regression trees?',
    options: ['Information gain', 'Gini impurity', 'Variance reduction (MSE)', 'Entropy'],
    correctAnswer: 2,
    explanation: 'Regression trees use variance reduction (or MSE reduction) to determine the best splits that minimize prediction error.'
  },
  {
    id: 'dt12',
    question: 'What does maximum depth control in a decision tree?',
    options: ['Number of features', 'How deep the tree can grow', 'Number of predictions', 'Training speed'],
    correctAnswer: 1,
    explanation: 'Maximum depth limits how many levels of splits the tree can have, controlling model complexity and preventing overfitting.'
  },
  {
    id: 'dt13',
    question: 'What is minimum samples split parameter?',
    options: ['Minimum samples in dataset', 'Minimum samples required to split an internal node', 'Minimum features', 'Minimum accuracy'],
    correctAnswer: 1,
    explanation: 'Minimum samples split specifies the minimum number of samples required at a node to attempt splitting it further.'
  },
  {
    id: 'dt14',
    question: 'Are decision trees sensitive to feature scaling?',
    options: ['Yes, very sensitive', 'No, they are scale-invariant', 'Only for continuous features', 'Only for categorical features'],
    correctAnswer: 1,
    explanation: 'Decision trees are scale-invariant because they only compare values within the same feature, not across features.'
  },
  {
    id: 'dt15',
    question: 'What makes decision trees interpretable?',
    options: ['Mathematical complexity', 'Visual tree structure mirrors human decision-making', 'High accuracy', 'Fast training'],
    correctAnswer: 1,
    explanation: 'Decision trees are highly interpretable because their tree structure can be visualized and follows logical if-then rules.'
  },
  {
    id: 'dt16',
    question: 'What is a decision tree\'s prediction for a regression problem?',
    options: ['Majority class', 'Mean of target values in leaf node', 'Weighted sum', 'Median of all values'],
    correctAnswer: 1,
    explanation: 'For regression, a decision tree predicts the mean (or sometimes median) of the target values in the leaf node.'
  },
  {
    id: 'dt17',
    question: 'What problem does a single decision tree have with stability?',
    options: ['Too stable', 'High variance - small changes in data can lead to very different trees', 'Too slow', 'Cannot fit data'],
    correctAnswer: 1,
    explanation: 'Decision trees have high variance, meaning small changes in training data can result in completely different tree structures.'
  },
  {
    id: 'dt18',
    question: 'How does a decision tree handle missing values?',
    options: ['Automatically fails', 'Can use surrogate splits or send to majority branch', 'Must be imputed first', 'Ignores them'],
    correctAnswer: 1,
    explanation: 'Some implementations use surrogate splits (alternative features) or send missing values to the majority child node.'
  },
  {
    id: 'dt19',
    question: 'What is the computational complexity of making a prediction with a decision tree?',
    options: ['O(n)', 'O(log n) where n is training samples', 'O(n²)', 'O(1)'],
    correctAnswer: 1,
    explanation: 'Prediction time is O(log n) as you traverse down the tree depth, which is logarithmic in the number of training samples for balanced trees.'
  },
  {
    id: 'dt20',
    question: 'Can decision trees capture feature interactions?',
    options: ['No, linear models only', 'Yes, naturally through hierarchical splits', 'Only with preprocessing', 'Only for categorical features'],
    correctAnswer: 1,
    explanation: 'Decision trees naturally capture feature interactions because subsequent splits depend on previous splits in the tree path.'
  }
];

// Random Forests - 20 questions
export const randomForestsQuestions: QuizQuestion[] = [
  {
    id: 'rf1',
    question: 'What is a Random Forest?',
    options: ['A single decision tree', 'An ensemble of decision trees', 'A clustering algorithm', 'A neural network'],
    correctAnswer: 1,
    explanation: 'Random Forest is an ensemble learning method that builds multiple decision trees and combines their predictions.'
  },
  {
    id: 'rf2',
    question: 'How does Random Forest reduce overfitting compared to a single decision tree?',
    options: ['By using less data', 'By averaging predictions from multiple trees', 'By using fewer features', 'By training slower'],
    correctAnswer: 1,
    explanation: 'Random Forest reduces overfitting through ensemble averaging - errors from individual trees tend to cancel out.'
  },
  {
    id: 'rf3',
    question: 'What is bagging in Random Forest?',
    options: ['Feature selection', 'Bootstrap Aggregating - sampling with replacement', 'Pruning technique', 'Loss function'],
    correctAnswer: 1,
    explanation: 'Bagging (Bootstrap Aggregating) creates multiple training subsets by sampling with replacement, introducing diversity among trees.'
  },
  {
    id: 'rf4',
    question: 'What is the purpose of random feature selection in Random Forest?',
    options: ['Speed up training', 'Decorrelate trees by considering only subset of features at each split', 'Reduce memory', 'Improve accuracy always'],
    correctAnswer: 1,
    explanation: 'Random feature selection at each split decorrelates the trees, preventing them from all making the same mistakes.'
  },
  {
    id: 'rf5',
    question: 'How does Random Forest make predictions for classification?',
    options: ['Uses first tree only', 'Majority voting across all trees', 'Averages probabilities', 'Uses deepest tree'],
    correctAnswer: 1,
    explanation: 'For classification, Random Forest uses majority voting - the class predicted by most trees wins.'
  },
  {
    id: 'rf6',
    question: 'How does Random Forest make predictions for regression?',
    options: ['Median of predictions', 'Average (mean) of all tree predictions', 'Maximum prediction', 'First tree\'s prediction'],
    correctAnswer: 1,
    explanation: 'For regression, Random Forest averages the predictions from all individual trees.'
  },
  {
    id: 'rf7',
    question: 'What is Out-of-Bag (OOB) error?',
    options: ['Training error', 'Validation error using samples not in bootstrap', 'Test error', 'Cross-validation error'],
    correctAnswer: 1,
    explanation: 'OOB error is calculated using samples that were not selected in the bootstrap sample for each tree, providing an unbiased estimate.'
  },
  {
    id: 'rf8',
    question: 'What is a key advantage of OOB error estimation?',
    options: ['More accurate than CV', 'No need for separate validation set', 'Faster training', 'Better predictions'],
    correctAnswer: 1,
    explanation: 'OOB error provides a validation estimate without needing a separate validation set, since ~37% of data is left out of each bootstrap.'
  },
  {
    id: 'rf9',
    question: 'How does Random Forest calculate feature importance?',
    options: ['By coefficient magnitude', 'By decrease in impurity when splitting on that feature', 'By correlation', 'Randomly'],
    correctAnswer: 1,
    explanation: 'Feature importance is calculated by measuring the average decrease in impurity (Gini or entropy) when splitting on each feature across all trees.'
  },
  {
    id: 'rf10',
    question: 'What is the typical number of features considered at each split (max_features)?',
    options: ['All features', 'sqrt(n_features) for classification', '1 feature always', 'log(n_features)'],
    correctAnswer: 1,
    explanation: 'Typically sqrt(n_features) for classification and n_features/3 for regression, though this can be tuned.'
  },
  {
    id: 'rf11',
    question: 'Are individual trees in Random Forest typically pruned?',
    options: ['Yes, heavily pruned', 'No, usually grown to maximum depth', 'Only some trees', 'Depends on data size'],
    correctAnswer: 1,
    explanation: 'Individual trees are usually grown deep without pruning, and the ensemble averaging reduces overfitting.'
  },
  {
    id: 'rf12',
    question: 'What is the main tradeoff when increasing the number of trees?',
    options: ['Accuracy vs memory', 'Training time vs marginal improvement', 'Bias vs variance', 'Precision vs recall'],
    correctAnswer: 1,
    explanation: 'More trees increase training and prediction time with diminishing returns in performance improvement after a certain point.'
  },
  {
    id: 'rf13',
    question: 'Can Random Forest be parallelized?',
    options: ['No, must be sequential', 'Yes, trees can be trained independently in parallel', 'Only for small datasets', 'Only for classification'],
    correctAnswer: 1,
    explanation: 'Random Forest is easily parallelizable since each tree can be trained independently on different CPU cores.'
  },
  {
    id: 'rf14',
    question: 'What is a disadvantage of Random Forest compared to single decision trees?',
    options: ['Lower accuracy', 'Less interpretable', 'Cannot handle categorical variables', 'Requires feature scaling'],
    correctAnswer: 1,
    explanation: 'Random Forests are less interpretable than single decision trees because you can\'t easily visualize or understand the ensemble.'
  },
  {
    id: 'rf15',
    question: 'Does Random Forest require feature scaling?',
    options: ['Yes, always', 'No, it is scale-invariant like decision trees', 'Only for continuous features', 'Only for categorical features'],
    correctAnswer: 1,
    explanation: 'Random Forest, like decision trees, is scale-invariant and doesn\'t require feature normalization or standardization.'
  },
  {
    id: 'rf16',
    question: 'How does Random Forest handle imbalanced datasets?',
    options: ['Automatically perfectly', 'Can use class weights or balanced sampling', 'Cannot handle imbalance', 'Only with preprocessing'],
    correctAnswer: 1,
    explanation: 'Random Forest can handle imbalance through class weights, balanced random sampling, or adjusting prediction thresholds.'
  },
  {
    id: 'rf17',
    question: 'What happens to bias and variance in Random Forest compared to single decision tree?',
    options: ['Both increase', 'Variance decreases, bias similar or slightly higher', 'Bias decreases, variance increases', 'Both decrease'],
    correctAnswer: 1,
    explanation: 'Random Forest reduces variance through averaging, while bias remains similar to or slightly higher than a single deep tree.'
  },
  {
    id: 'rf18',
    question: 'What is extremely randomized trees (Extra Trees)?',
    options: ['Random Forest with bugs', 'Random Forest with random thresholds for splits', 'Very small trees', 'Very large trees'],
    correctAnswer: 1,
    explanation: 'Extra Trees use random thresholds for splits instead of optimal ones, introducing more randomness and potentially reducing variance further.'
  },
  {
    id: 'rf19',
    question: 'Can Random Forest output probability estimates?',
    options: ['No, only class labels', 'Yes, by averaging predicted probabilities across trees', 'Only in scikit-learn', 'Only for binary classification'],
    correctAnswer: 1,
    explanation: 'Random Forest can output probability estimates by averaging the predicted probabilities from all trees.'
  },
  {
    id: 'rf20',
    question: 'What is the bootstrap sample size for each tree?',
    options: ['10% of data', 'Same as original dataset (with replacement)', '50% of data', '2× dataset size'],
    correctAnswer: 1,
    explanation: 'Each tree is trained on a bootstrap sample of the same size as the original dataset, sampled with replacement.'
  }
];

// Gradient Boosting - 20 questions  
export const gradientBoostingQuestions: QuizQuestion[] = [
  {
    id: 'gb1',
    question: 'What is the main idea behind gradient boosting?',
    options: ['Train trees in parallel', 'Sequentially build trees to correct previous trees\' errors', 'Random feature selection', 'Deep trees only'],
    correctAnswer: 1,
    explanation: 'Gradient boosting builds trees sequentially, where each new tree tries to correct the errors (residuals) of the previous ensemble.'
  },
  {
    id: 'gb2',
    question: 'How does gradient boosting differ from Random Forest?',
    options: ['No difference', 'Boosting is sequential, RF is parallel; boosting reduces bias, RF reduces variance', 'Boosting is faster', 'RF is more accurate'],
    correctAnswer: 1,
    explanation: 'Gradient boosting trains trees sequentially to reduce bias, while Random Forest trains trees in parallel to reduce variance.'
  },
  {
    id: 'gb3',
    question: 'What does each new tree in gradient boosting predict?',
    options: ['The target directly', 'The residuals (errors) of previous predictions', 'Random values', 'Feature importance'],
    correctAnswer: 1,
    explanation: 'Each new tree fits the residuals (errors) from the current ensemble, gradually improving predictions.'
  },
  {
    id: 'gb4',
    question: 'What is the learning rate (shrinkage) in gradient boosting?',
    options: ['Optimization speed', 'Factor that scales each tree\'s contribution', 'Number of iterations', 'Tree depth'],
    correctAnswer: 1,
    explanation: 'Learning rate scales the contribution of each tree, with lower values requiring more trees but often generalizing better.'
  },
  {
    id: 'gb5',
    question: 'What is a typical tree depth in gradient boosting?',
    options: ['Very deep (unlimited)', 'Shallow (3-8 levels)', 'Single level', 'Medium depth (10-20)'],
    correctAnswer: 1,
    explanation: 'Gradient boosting typically uses shallow trees (weak learners) as each tree only needs to capture residual patterns.'
  },
  {
    id: 'gb6',
    question: 'What happens if the learning rate is too high?',
    options: ['Too slow', 'Risk of overfitting and missing optimal solution', 'Perfect performance', 'Cannot train'],
    correctAnswer: 1,
    explanation: 'High learning rate can cause overfitting as each tree contributes too much, potentially overshooting the optimal solution.'
  },
  {
    id: 'gb7',
    question: 'What happens if the learning rate is too low?',
    options: ['Overfitting', 'Slow convergence requiring many trees', 'Better accuracy always', 'Training fails'],
    correctAnswer: 1,
    explanation: 'Very low learning rate requires many more trees to achieve good performance, increasing training time.'
  },
  {
    id: 'gb8',
    question: 'What is early stopping in gradient boosting?',
    options: ['Stop after one tree', 'Stop when validation performance stops improving', 'Stop randomly', 'Never stop'],
    correctAnswer: 1,
    explanation: 'Early stopping halts training when validation performance plateaus or degrades, preventing overfitting.'
  },
  {
    id: 'gb9',
    question: 'What is XGBoost?',
    options: ['A neural network', 'An optimized gradient boosting library with regularization', 'A random forest variant', 'A clustering algorithm'],
    correctAnswer: 1,
    explanation: 'XGBoost is a highly optimized gradient boosting implementation with additional regularization and speed improvements.'
  },
  {
    id: 'gb10',
    question: 'What is subsample in gradient boosting?',
    options: ['Feature sampling', 'Fraction of samples used to train each tree', 'Tree depth', 'Number of trees'],
    correctAnswer: 1,
    explanation: 'Subsample is the fraction of training samples randomly selected for training each tree, adding randomness and reducing overfitting.'
  },
  {
    id: 'gb11',
    question: 'How does gradient boosting handle missing values in XGBoost?',
    options: ['Drops rows', 'Learns optimal direction for missing values', 'Imputes with mean', 'Cannot handle them'],
    correctAnswer: 1,
    explanation: 'XGBoost learns the optimal direction (left or right) to send missing values at each split based on the data.'
  },
  {
    id: 'gb12',
    question: 'What is the difference between GBM and AdaBoost?',
    options: ['No difference', 'GBM optimizes arbitrary loss functions, AdaBoost focuses on exponential loss', 'GBM is slower', 'AdaBoost is more accurate'],
    correctAnswer: 1,
    explanation: 'Gradient boosting can optimize any differentiable loss function, while AdaBoost specifically uses exponential loss.'
  },
  {
    id: 'gb13',
    question: 'What is LightGBM?',
    options: ['A lightweight model', 'A fast gradient boosting framework using leaf-wise growth', 'A linear model', 'A deep learning library'],
    correctAnswer: 1,
    explanation: 'LightGBM uses leaf-wise tree growth (vs level-wise) and histogram-based learning for faster training.'
  },
  {
    id: 'gb14',
    question: 'What is CatBoost designed for?',
    options: ['Image classification', 'Handling categorical features efficiently', 'Text processing', 'Time series only'],
    correctAnswer: 1,
    explanation: 'CatBoost is optimized for handling categorical features natively without extensive preprocessing.'
  },
  {
    id: 'gb15',
    question: 'Can gradient boosting be parallelized like Random Forest?',
    options: ['Yes, equally parallelizable', 'Partially - tree construction can be parallelized but trees are sequential', 'No, completely sequential', 'Only for prediction'],
    correctAnswer: 1,
    explanation: 'While individual tree construction can be parallelized, the trees themselves must be built sequentially, limiting parallelization.'
  },
  {
    id: 'gb16',
    question: 'What is the typical relationship between number of trees and learning rate?',
    options: ['Independent', 'Lower learning rate requires more trees', 'Higher learning rate requires more trees', 'No relationship'],
    correctAnswer: 1,
    explanation: 'Lower learning rates need more trees to reach optimal performance, while higher rates need fewer trees.'
  },
  {
    id: 'gb17',
    question: 'What is regularization in gradient boosting?',
    options: ['Number of trees', 'Penalties on tree complexity (depth, leaves, weights)', 'Learning rate', 'Sample size'],
    correctAnswer: 1,
    explanation: 'Regularization adds penalties for tree complexity, such as limiting depth, number of leaves, or L1/L2 weight penalties.'
  },
  {
    id: 'gb18',
    question: 'Is gradient boosting more prone to overfitting than Random Forest?',
    options: ['No, less prone', 'Yes, more prone without proper regularization', 'Same', 'Neither overfits'],
    correctAnswer: 1,
    explanation: 'Gradient boosting can more easily overfit without proper learning rate, regularization, and early stopping.'
  },
  {
    id: 'gb19',
    question: 'What is histogram-based gradient boosting?',
    options: ['Plotting results', 'Bucketing continuous features for faster training', 'Image processing', 'Data visualization'],
    correctAnswer: 1,
    explanation: 'Histogram-based methods bucket continuous values into discrete bins, significantly speeding up training while maintaining accuracy.'
  },
  {
    id: 'gb20',
    question: 'Why is gradient boosting popular in machine learning competitions?',
    options: ['Fastest training', 'Often achieves state-of-the-art results on structured data', 'Easiest to use', 'Requires no tuning'],
    correctAnswer: 1,
    explanation: 'Gradient boosting often achieves the best results on structured/tabular data and is a go-to choice in competitions like Kaggle.'
  }
];
