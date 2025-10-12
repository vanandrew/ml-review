import { QuizQuestion } from '../../types';

// PCA - 20 questions
export const pcaQuestions: QuizQuestion[] = [
  {
    id: 'pca1',
    question: 'What is PCA primarily used for?',
    options: ['Classification', 'Dimensionality reduction', 'Clustering', 'Regression'],
    correctAnswer: 1,
    explanation: 'PCA (Principal Component Analysis) is primarily used for dimensionality reduction by projecting data onto principal components.'
  },
  {
    id: 'pca2',
    question: 'What are principal components?',
    options: ['Original features', 'New orthogonal axes that capture maximum variance', 'Cluster centers', 'Class labels'],
    correctAnswer: 1,
    explanation: 'Principal components are new orthogonal (uncorrelated) directions that capture the maximum variance in the data.'
  },
  {
    id: 'pca3',
    question: 'How does PCA order principal components?',
    options: ['Randomly', 'By decreasing variance explained', 'Alphabetically', 'By correlation'],
    correctAnswer: 1,
    explanation: 'Principal components are ordered by the amount of variance they explain, with PC1 capturing the most variance.'
  },
  {
    id: 'pca4',
    question: 'Is PCA supervised or unsupervised?',
    options: ['Supervised', 'Unsupervised', 'Semi-supervised', 'Reinforcement'],
    correctAnswer: 1,
    explanation: 'PCA is unsupervised as it doesn\'t use target labels, only finding directions of maximum variance in features.'
  },
  {
    id: 'pca5',
    question: 'Why is centering data important for PCA?',
    options: ['Not important', 'PCA assumes data is centered at origin', 'Only for visualization', 'Only for speed'],
    correctAnswer: 1,
    explanation: 'PCA finds directions from the origin, so data should be centered (mean subtracted) for correct principal components.'
  },
  {
    id: 'pca6',
    question: 'Should you scale features before PCA?',
    options: ['Never', 'Yes, if features have different scales', 'Only for small datasets', 'Only for high dimensions'],
    correctAnswer: 1,
    explanation: 'Features with larger scales will dominate variance. Standardization is recommended unless features are already comparable.'
  },
  {
    id: 'pca7',
    question: 'What does explained variance ratio tell you?',
    options: ['Number of components', 'Proportion of total variance captured by each component', 'Error rate', 'Sample size'],
    correctAnswer: 1,
    explanation: 'Explained variance ratio shows what percentage of total variance each principal component captures.'
  },
  {
    id: 'pca8',
    question: 'How do you choose the number of components to keep?',
    options: ['Always keep all', 'Based on cumulative explained variance threshold (e.g., 95%)', 'Random selection', 'Always keep 2'],
    correctAnswer: 1,
    explanation: 'Typically keep enough components to retain a desired percentage of variance (e.g., 95%) or use scree plot analysis.'
  },
  {
    id: 'pca9',
    question: 'What is the scree plot?',
    options: ['3D visualization', 'Plot of explained variance vs component number', 'Correlation matrix', 'Feature importance'],
    correctAnswer: 1,
    explanation: 'A scree plot shows explained variance for each component, helping identify the "elbow" where additional components add little value.'
  },
  {
    id: 'pca10',
    question: 'Are principal components interpretable?',
    options: ['Yes, always', 'No, they are linear combinations of original features', 'Only the first one', 'Only for standardized data'],
    correctAnswer: 1,
    explanation: 'Principal components are mathematical constructs (weighted sums of original features) that often lack clear interpretability.'
  },
  {
    id: 'pca11',
    question: 'What mathematical technique does PCA use?',
    options: ['K-means', 'Eigenvalue decomposition of covariance matrix or SVD', 'Gradient descent', 'Decision trees'],
    correctAnswer: 1,
    explanation: 'PCA uses eigenvalue decomposition of the covariance matrix (or Singular Value Decomposition) to find principal components.'
  },
  {
    id: 'pca12',
    question: 'What is the relationship between PCA components?',
    options: ['Correlated', 'Orthogonal (uncorrelated)', 'Parallel', 'Random'],
    correctAnswer: 1,
    explanation: 'Principal components are orthogonal to each other, meaning they are uncorrelated and capture different aspects of variance.'
  },
  {
    id: 'pca13',
    question: 'Can PCA be used for feature engineering?',
    options: ['No, only for visualization', 'Yes, as a preprocessing step before modeling', 'Only for regression', 'Only for classification'],
    correctAnswer: 1,
    explanation: 'PCA is commonly used as preprocessing to reduce dimensionality, remove multicollinearity, and speed up training.'
  },
  {
    id: 'pca14',
    question: 'What is a limitation of PCA?',
    options: ['Too slow', 'Assumes linear relationships and may not capture non-linear patterns', 'Cannot handle missing values', 'Only works in 2D'],
    correctAnswer: 1,
    explanation: 'PCA is a linear method and may not capture non-linear relationships. For non-linear dimensionality reduction, use methods like t-SNE or UMAP.'
  },
  {
    id: 'pca15',
    question: 'Is PCA sensitive to outliers?',
    options: ['No, robust', 'Yes, outliers can distort principal components', 'Only for small datasets', 'Only for many features'],
    correctAnswer: 1,
    explanation: 'PCA uses variance which is sensitive to outliers. Robust PCA variants exist to handle this.'
  },
  {
    id: 'pca16',
    question: 'Can you reconstruct original data from PCA components?',
    options: ['No, information is lost', 'Approximately, with some information loss if components are dropped', 'Perfectly always', 'Only for 2 components'],
    correctAnswer: 1,
    explanation: 'You can reconstruct data by inverse transform, but if you dropped components, some information (variance) is lost.'
  },
  {
    id: 'pca17',
    question: 'What is the curse of dimensionality that PCA helps with?',
    options: ['Too many samples', 'Data becomes sparse in high dimensions', 'Too few features', 'Slow training'],
    correctAnswer: 1,
    explanation: 'In high dimensions, data becomes sparse and distances less meaningful. PCA reduces dimensions while retaining important information.'
  },
  {
    id: 'pca18',
    question: 'What is Kernel PCA?',
    options: ['Standard PCA', 'PCA extended to capture non-linear relationships using kernel trick', 'PCA for small data', 'PCA for images only'],
    correctAnswer: 1,
    explanation: 'Kernel PCA applies the kernel trick to perform PCA in a higher-dimensional feature space, capturing non-linear patterns.'
  },
  {
    id: 'pca19',
    question: 'Can PCA improve model performance?',
    options: ['Always improves', 'Sometimes - reduces overfitting and speeds up training, but may lose information', 'Never improves', 'Only for neural networks'],
    correctAnswer: 1,
    explanation: 'PCA can help by reducing overfitting and training time, but may hurt if important information is in dropped components.'
  },
  {
    id: 'pca20',
    question: 'What is the difference between PCA and LDA?',
    options: ['No difference', 'PCA is unsupervised (maximizes variance), LDA is supervised (maximizes class separation)', 'PCA is supervised', 'LDA is unsupervised'],
    correctAnswer: 1,
    explanation: 'PCA finds directions of maximum variance ignoring labels, while LDA finds directions that maximize class separation using labels.'
  },
  {
    id: 'pca21',
    question: 'What is incremental PCA?',
    options: ['Faster PCA', 'PCA that processes data in mini-batches for large datasets', 'PCA with more components', 'PCA for streaming data'],
    correctAnswer: 1,
    explanation: 'Incremental PCA processes data in mini-batches, allowing PCA on datasets too large to fit in memory.'
  },
  {
    id: 'pca22',
    question: 'What is sparse PCA?',
    options: ['PCA for sparse data', 'PCA with L1 regularization to get sparse loadings', 'PCA with missing values', 'Fast PCA'],
    correctAnswer: 1,
    explanation: 'Sparse PCA adds L1 penalty to get principal components with many zero loadings, improving interpretability.'
  },
  {
    id: 'pca23',
    question: 'How does PCA relate to SVD (Singular Value Decomposition)?',
    options: ['Unrelated', 'PCA eigenvectors are SVD right singular vectors', 'PCA is simpler', 'SVD is for matrices only'],
    correctAnswer: 1,
    explanation: 'PCA can be computed via SVD: principal components are the right singular vectors, and eigenvalues relate to singular values squared.'
  },
  {
    id: 'pca24',
    question: 'What is randomized PCA?',
    options: ['Random initialization', 'Approximation algorithm using random projections for faster computation', 'PCA with noise', 'Unreliable PCA'],
    correctAnswer: 1,
    explanation: 'Randomized PCA uses randomized algorithms to approximate principal components much faster, especially useful for high-dimensional data.'
  },
  {
    id: 'pca25',
    question: 'Can PCA be used for data compression?',
    options: ['No', 'Yes, by keeping top k components and reconstructing approximate data', 'Only for images', 'Only lossless'],
    correctAnswer: 1,
    explanation: 'PCA provides lossy compression by projecting to k dimensions and reconstructing, minimizing reconstruction error for given dimensionality.'
  }
];

// Naive Bayes - 20 questions
export const naiveBayesQuestions: QuizQuestion[] = [
  {
    id: 'nb1',
    question: 'What theorem is Naive Bayes based on?',
    options: ['Fermat\'s theorem', 'Bayes\' theorem', 'Central limit theorem', 'Pythagorean theorem'],
    correctAnswer: 1,
    explanation: 'Naive Bayes is based on Bayes\' theorem which describes the probability of an event based on prior knowledge.'
  },
  {
    id: 'nb2',
    question: 'What is the "naive" assumption in Naive Bayes?',
    options: ['Simple model', 'Features are independent given the class', 'Small dataset', 'Binary classification only'],
    correctAnswer: 1,
    explanation: 'The "naive" assumption is that all features are conditionally independent given the class label, which simplifies calculations.'
  },
  {
    id: 'nb3',
    question: 'What does Bayes\' theorem calculate?',
    options: ['P(Class)', 'P(Class|Features) using P(Features|Class) and P(Class)', 'P(Features)', 'Accuracy'],
    correctAnswer: 1,
    explanation: 'Bayes\' theorem calculates posterior probability P(Class|Features) from the likelihood P(Features|Class) and prior P(Class).'
  },
  {
    id: 'nb4',
    question: 'What is Gaussian Naive Bayes used for?',
    options: ['Categorical features', 'Continuous features assuming normal distribution', 'Text classification only', 'Images only'],
    correctAnswer: 1,
    explanation: 'Gaussian Naive Bayes assumes continuous features follow a Gaussian (normal) distribution within each class.'
  },
  {
    id: 'nb5',
    question: 'What is Multinomial Naive Bayes used for?',
    options: ['Continuous features', 'Discrete count data like word frequencies', 'Images', 'Time series'],
    correctAnswer: 1,
    explanation: 'Multinomial Naive Bayes works with discrete counts and is commonly used for text classification with word counts/frequencies.'
  },
  {
    id: 'nb6',
    question: 'What is Bernoulli Naive Bayes used for?',
    options: ['Continuous features', 'Binary/boolean features', 'Multi-class only', 'Regression'],
    correctAnswer: 1,
    explanation: 'Bernoulli Naive Bayes works with binary features (present/absent) and is used for text with binary word occurrence.'
  },
  {
    id: 'nb7',
    question: 'What is a major advantage of Naive Bayes?',
    options: ['Highest accuracy', 'Fast training and prediction, works well with high dimensions', 'Captures feature dependencies', 'No assumptions'],
    correctAnswer: 1,
    explanation: 'Naive Bayes is very fast to train and predict, and performs surprisingly well even in high-dimensional spaces like text.'
  },
  {
    id: 'nb8',
    question: 'What is the zero-frequency problem in Naive Bayes?',
    options: ['No data', 'When a feature value never appears with a class in training', 'Division by zero', 'Missing values'],
    correctAnswer: 1,
    explanation: 'If a feature value never occurs with a class in training, the probability is zero, making the entire prediction zero.'
  },
  {
    id: 'nb9',
    question: 'What is Laplace smoothing?',
    options: ['Data preprocessing', 'Adding small constant to counts to avoid zero probabilities', 'Feature scaling', 'Optimization method'],
    correctAnswer: 1,
    explanation: 'Laplace (additive) smoothing adds a small value (usually 1) to all counts to ensure no probability is exactly zero.'
  },
  {
    id: 'nb10',
    question: 'Does Naive Bayes require feature scaling?',
    options: ['Yes, always', 'No, it uses probabilities not distances', 'Only for continuous features', 'Only for text'],
    correctAnswer: 1,
    explanation: 'Naive Bayes works with probabilities, not distances, so feature scaling is not necessary.'
  },
  {
    id: 'nb11',
    question: 'Can Naive Bayes handle missing values?',
    options: ['No, must impute', 'Yes, can ignore missing values in probability calculations', 'Only for categorical features', 'Only with preprocessing'],
    correctAnswer: 1,
    explanation: 'Naive Bayes can handle missing values by simply not including them in the probability calculation for that instance.'
  },
  {
    id: 'nb12',
    question: 'Is Naive Bayes a discriminative or generative model?',
    options: ['Discriminative', 'Generative', 'Neither', 'Both'],
    correctAnswer: 1,
    explanation: 'Naive Bayes is a generative model as it models the joint probability P(X, Y) and can generate new samples.'
  },
  {
    id: 'nb13',
    question: 'Why does Naive Bayes work well despite the independence assumption being violated?',
    options: ['The assumption is always true', 'For classification, only the ranking of probabilities matters', 'It doesn\'t work well', 'Random luck'],
    correctAnswer: 1,
    explanation: 'Even when independence is violated, Naive Bayes often produces correct classifications because relative probability rankings matter more than exact values.'
  },
  {
    id: 'nb14',
    question: 'What is a limitation of Naive Bayes?',
    options: ['Too slow', 'Cannot capture feature interactions/dependencies', 'Requires too much data', 'Cannot handle multi-class'],
    correctAnswer: 1,
    explanation: 'The independence assumption means Naive Bayes cannot capture relationships or interactions between features.'
  },
  {
    id: 'nb15',
    question: 'Is Naive Bayes suitable for spam detection?',
    options: ['No, too simple', 'Yes, it\'s fast and works well with word frequency features', 'Only for small emails', 'Only for images'],
    correctAnswer: 1,
    explanation: 'Naive Bayes is a classic choice for spam detection due to its speed and effectiveness with text features (word frequencies).'
  },
  {
    id: 'nb16',
    question: 'Can Naive Bayes be used for regression?',
    options: ['Yes, commonly', 'Primarily for classification, not typically for regression', 'Only for continuous targets', 'Only for binary targets'],
    correctAnswer: 1,
    explanation: 'Naive Bayes is designed for classification. For regression, other methods like linear regression are more appropriate.'
  },
  {
    id: 'nb17',
    question: 'What is the computational complexity of Naive Bayes training?',
    options: ['O(n²)', 'O(n × d) - linear in samples and features', 'O(n³)', 'Exponential'],
    correctAnswer: 1,
    explanation: 'Naive Bayes training is very fast with linear complexity in the number of samples and features.'
  },
  {
    id: 'nb18',
    question: 'Does Naive Bayes provide well-calibrated probabilities?',
    options: ['Yes, always perfect', 'No, probabilities may need calibration for decision-making', 'Only for binary classification', 'Only with smoothing'],
    correctAnswer: 1,
    explanation: 'Naive Bayes probabilities are often poorly calibrated (too extreme) due to the independence assumption, though rankings are usually correct.'
  },
  {
    id: 'nb19',
    question: 'How does Naive Bayes handle multi-class classification?',
    options: ['Cannot handle it', 'Naturally handles it by computing posterior for each class', 'Needs one-vs-rest', 'Only binary'],
    correctAnswer: 1,
    explanation: 'Naive Bayes naturally extends to multi-class by computing posterior probability for each class and predicting the maximum.'
  },
  {
    id: 'nb20',
    question: 'When is Naive Bayes most effective?',
    options: ['Always', 'High-dimensional data (like text) where features are relatively independent', 'Low-dimensional data', 'When features are highly correlated'],
    correctAnswer: 1,
    explanation: 'Naive Bayes excels in high-dimensional spaces (e.g., text classification) where despite some dependence, the independence assumption is reasonable enough.'
  },
  {
    id: 'nb21',
    question: 'What is complement Naive Bayes?',
    options: ['Standard NB', 'Variant using complement of each class for imbalanced data', 'NB with extra features', 'NB for regression'],
    correctAnswer: 1,
    explanation: 'Complement Naive Bayes estimates parameters from the complement (all other classes) and is particularly suited for imbalanced datasets.'
  },
  {
    id: 'nb22',
    question: 'How does Naive Bayes handle continuous features?',
    options: ['Cannot handle', 'Assumes distribution (usually Gaussian) for each class', 'Discretizes all features', 'Uses only mean'],
    correctAnswer: 1,
    explanation: 'Gaussian Naive Bayes assumes continuous features follow a Gaussian distribution within each class, estimating mean and variance.'
  },
  {
    id: 'nb23',
    question: 'What is the prior probability in Naive Bayes?',
    options: ['Feature probability', 'P(Class) before seeing data', 'Likelihood', 'Posterior'],
    correctAnswer: 1,
    explanation: 'Prior probability P(C) represents the baseline probability of each class before observing any features, typically estimated from training data frequencies.'
  },
  {
    id: 'nb24',
    question: 'What makes Naive Bayes particularly fast?',
    options: ['GPU acceleration', 'Linear time training and prediction due to independence assumption', 'Small model size', 'No training needed'],
    correctAnswer: 1,
    explanation: 'The independence assumption allows computing probabilities as simple products, making both training and prediction very fast and scalable.'
  },
  {
    id: 'nb25',
    question: 'Can Naive Bayes be used for semi-supervised learning?',
    options: ['No', 'Yes, can use EM algorithm to iteratively label unlabeled data', 'Only supervised', 'Only unsupervised'],
    correctAnswer: 1,
    explanation: 'Naive Bayes can be extended to semi-supervised learning using Expectation-Maximization to iteratively improve with unlabeled data.'
  }
];
