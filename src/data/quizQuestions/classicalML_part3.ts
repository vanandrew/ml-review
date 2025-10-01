import { QuizQuestion } from '../../types';

// Support Vector Machines - 20 questions
export const svmQuestions: QuizQuestion[] = [
  {
    id: 'svm1',
    question: 'What is the main objective of SVM?',
    options: ['Minimize error', 'Find the hyperplane with maximum margin', 'Cluster data', 'Reduce dimensions'],
    correctAnswer: 1,
    explanation: 'SVM aims to find the hyperplane that maximizes the margin between different classes, providing better generalization.'
  },
  {
    id: 'svm2',
    question: 'What are support vectors?',
    options: ['All training points', 'Data points closest to the decision boundary', 'Outliers', 'Test points'],
    correctAnswer: 1,
    explanation: 'Support vectors are the training points closest to the decision boundary that determine the position and orientation of the hyperplane.'
  },
  {
    id: 'svm3',
    question: 'What is the margin in SVM?',
    options: ['Error rate', 'Distance between hyperplane and nearest points from each class', 'Number of support vectors', 'Accuracy'],
    correctAnswer: 1,
    explanation: 'The margin is the distance from the decision boundary to the nearest training points (support vectors) from each class.'
  },
  {
    id: 'svm4',
    question: 'What is a hard margin SVM?',
    options: ['Very accurate SVM', 'SVM that requires perfect linear separation', 'SVM with many support vectors', 'SVM for regression'],
    correctAnswer: 1,
    explanation: 'Hard margin SVM requires that all points be correctly classified with no points within the margin, only works for linearly separable data.'
  },
  {
    id: 'svm5',
    question: 'What is a soft margin SVM?',
    options: ['Weak classifier', 'Allows some misclassifications using slack variables', 'SVM without kernel', 'Linear SVM only'],
    correctAnswer: 1,
    explanation: 'Soft margin SVM allows some misclassifications through slack variables, making it more flexible for non-linearly separable data.'
  },
  {
    id: 'svm6',
    question: 'What does the C parameter control in SVM?',
    options: ['Number of support vectors', 'Tradeoff between margin size and training error', 'Kernel type', 'Learning rate'],
    correctAnswer: 1,
    explanation: 'C controls the penalty for misclassification: high C = small margin with fewer errors, low C = larger margin with more errors allowed.'
  },
  {
    id: 'svm7',
    question: 'What is the kernel trick?',
    options: ['Optimization technique', 'Implicitly mapping data to higher dimensions', 'Pruning method', 'Sampling technique'],
    correctAnswer: 1,
    explanation: 'The kernel trick allows SVM to operate in high-dimensional space without explicitly computing the transformation, using kernel functions.'
  },
  {
    id: 'svm8',
    question: 'What does the RBF (Gaussian) kernel do?',
    options: ['Linear separation', 'Maps data to infinite dimensions for non-linear separation', 'Reduces dimensions', 'Clusters data'],
    correctAnswer: 1,
    explanation: 'The RBF kernel maps data to an infinite-dimensional space, enabling separation of complex non-linear patterns.'
  },
  {
    id: 'svm9',
    question: 'What is the polynomial kernel?',
    options: ['Linear kernel', 'Kernel that computes polynomial combinations of features', 'Exponential kernel', 'Sigmoid kernel'],
    correctAnswer: 1,
    explanation: 'Polynomial kernel computes dot products of polynomial feature combinations, controlled by the degree parameter.'
  },
  {
    id: 'svm10',
    question: 'What does the gamma parameter control in RBF kernel?',
    options: ['Margin size', 'Influence range of a single training example', 'Number of features', 'Training speed'],
    correctAnswer: 1,
    explanation: 'Gamma controls how far the influence of a single training example reaches: high gamma = close influence (risk of overfitting).'
  },
  {
    id: 'svm11',
    question: 'Why is feature scaling important for SVM?',
    options: ['Not important', 'SVM uses distances, so features should be on similar scales', 'Only for speed', 'Only for accuracy'],
    correctAnswer: 1,
    explanation: 'SVM is sensitive to feature scales because it uses distances in the feature space, so standardization is crucial.'
  },
  {
    id: 'svm12',
    question: 'What is the computational complexity of training SVM?',
    options: ['O(n)', 'O(n²) to O(n³)', 'O(log n)', 'O(n log n)'],
    correctAnswer: 1,
    explanation: 'SVM training complexity is typically O(n²) to O(n³) where n is the number of samples, making it slow for large datasets.'
  },
  {
    id: 'svm13',
    question: 'Can SVM handle multi-class classification?',
    options: ['No, binary only', 'Yes, using one-vs-one or one-vs-rest', 'Only with special kernels', 'Only with preprocessing'],
    correctAnswer: 1,
    explanation: 'SVM extends to multi-class using strategies like one-vs-one (all pairs) or one-vs-rest (each class vs all others).'
  },
  {
    id: 'svm14',
    question: 'What is SVR (Support Vector Regression)?',
    options: ['SVM for feature selection', 'SVM adapted for regression problems', 'SVM variant for clustering', 'SVM preprocessing'],
    correctAnswer: 1,
    explanation: 'SVR adapts SVM for regression by finding a function with maximum epsilon (ε) deviation from actual targets.'
  },
  {
    id: 'svm15',
    question: 'What happens if C is very large?',
    options: ['Underfitting', 'Small margin, risk of overfitting', 'Large margin', 'Training fails'],
    correctAnswer: 1,
    explanation: 'Large C heavily penalizes misclassifications, leading to smaller margins and potential overfitting.'
  },
  {
    id: 'svm16',
    question: 'What happens if C is very small?',
    options: ['Overfitting', 'Large margin, risk of underfitting', 'Small margin', 'Perfect fit'],
    correctAnswer: 1,
    explanation: 'Small C allows more misclassifications for a larger margin, which may underfit if too permissive.'
  },
  {
    id: 'svm17',
    question: 'What is a disadvantage of SVM?',
    options: ['Too simple', 'Slow training on large datasets and difficult to interpret', 'Cannot handle non-linear data', 'Requires categorical features'],
    correctAnswer: 1,
    explanation: 'SVM is slow for large datasets and difficult to interpret compared to decision trees or linear models.'
  },
  {
    id: 'svm18',
    question: 'Does SVM provide probability estimates?',
    options: ['Yes, naturally', 'Can estimate probabilities using calibration methods', 'No, never', 'Only with linear kernel'],
    correctAnswer: 1,
    explanation: 'SVM doesn\'t naturally provide probabilities, but methods like Platt scaling can calibrate outputs to probabilities.'
  },
  {
    id: 'svm19',
    question: 'What is the decision function in SVM?',
    options: ['Probability', 'Signed distance from hyperplane', 'Accuracy score', 'Class label'],
    correctAnswer: 1,
    explanation: 'The decision function returns the signed distance from the hyperplane, with sign indicating class and magnitude indicating confidence.'
  },
  {
    id: 'svm20',
    question: 'When is linear SVM preferred over kernel SVM?',
    options: ['Never', 'When data is linearly separable and speed is important', 'For small datasets only', 'For multi-class only'],
    correctAnswer: 1,
    explanation: 'Linear SVM is much faster and preferred when data is linearly separable or has very high dimensions.'
  }
];

// K-Nearest Neighbors - 20 questions
export const knnQuestions: QuizQuestion[] = [
  {
    id: 'knn1',
    question: 'What is the basic principle of K-NN?',
    options: ['Find clusters', 'Classify based on majority vote of K nearest neighbors', 'Find linear boundaries', 'Reduce dimensions'],
    correctAnswer: 1,
    explanation: 'K-NN classifies a point based on the majority class among its K nearest neighbors in the feature space.'
  },
  {
    id: 'knn2',
    question: 'What does K represent in K-NN?',
    options: ['Number of classes', 'Number of neighbors to consider', 'Number of features', 'Number of iterations'],
    correctAnswer: 1,
    explanation: 'K is the number of nearest neighbors used to make a prediction for a new data point.'
  },
  {
    id: 'knn3',
    question: 'Is K-NN a parametric or non-parametric algorithm?',
    options: ['Parametric', 'Non-parametric', 'Both', 'Neither'],
    correctAnswer: 1,
    explanation: 'K-NN is non-parametric as it makes no assumptions about the underlying data distribution and stores all training data.'
  },
  {
    id: 'knn4',
    question: 'What is lazy learning in context of K-NN?',
    options: ['Slow training', 'No explicit training phase - computation happens at prediction time', 'Inefficient algorithm', 'Random predictions'],
    correctAnswer: 1,
    explanation: 'K-NN is a lazy learner because it doesn\'t build a model during training; all computation happens when making predictions.'
  },
  {
    id: 'knn5',
    question: 'What distance metric is most commonly used in K-NN?',
    options: ['Manhattan distance', 'Euclidean distance', 'Cosine similarity', 'Hamming distance'],
    correctAnswer: 1,
    explanation: 'Euclidean distance (straight-line distance) is the most commonly used metric in K-NN.'
  },
  {
    id: 'knn6',
    question: 'Why is feature scaling critical for K-NN?',
    options: ['Not important', 'K-NN uses distances, features with larger scales dominate', 'Only for speed', 'Only for accuracy'],
    correctAnswer: 1,
    explanation: 'Features with larger scales will dominate the distance calculation, so normalization/standardization is essential.'
  },
  {
    id: 'knn7',
    question: 'What happens if K=1?',
    options: ['No predictions', 'Nearest neighbor only - risk of overfitting', 'Best accuracy always', 'Error'],
    correctAnswer: 1,
    explanation: 'K=1 uses only the single nearest neighbor, making the model highly sensitive to noise and prone to overfitting.'
  },
  {
    id: 'knn8',
    question: 'What happens if K is too large?',
    options: ['Overfitting', 'Underfitting - boundaries become smoother', 'Better accuracy', 'Faster prediction'],
    correctAnswer: 1,
    explanation: 'Very large K smooths decision boundaries too much, potentially causing underfitting and misclassifying minority classes.'
  },
  {
    id: 'knn9',
    question: 'How do you choose the optimal K?',
    options: ['Always use 5', 'Use cross-validation to test different values', 'K = number of classes', 'Random selection'],
    correctAnswer: 1,
    explanation: 'The optimal K is typically found through cross-validation, testing different values and selecting the best performing one.'
  },
  {
    id: 'knn10',
    question: 'Should K be odd or even for binary classification?',
    options: ['Must be even', 'Odd is preferred to avoid ties', 'No difference', 'Must be even'],
    correctAnswer: 1,
    explanation: 'Odd K is preferred for binary classification to avoid ties in voting.'
  },
  {
    id: 'knn11',
    question: 'What is the computational complexity of K-NN prediction?',
    options: ['O(1)', 'O(n) where n is training size', 'O(log n)', 'O(n²)'],
    correctAnswer: 1,
    explanation: 'Making a prediction requires computing distances to all n training points, giving O(n) complexity.'
  },
  {
    id: 'knn12',
    question: 'How can K-NN prediction be made faster?',
    options: ['Use more K', 'Use space-partitioning structures like KD-trees or Ball trees', 'Reduce K', 'Use fewer features'],
    correctAnswer: 1,
    explanation: 'Data structures like KD-trees or Ball trees can reduce search time from O(n) to O(log n) for finding nearest neighbors.'
  },
  {
    id: 'knn13',
    question: 'Can K-NN be used for regression?',
    options: ['No, classification only', 'Yes, by averaging the target values of K nearest neighbors', 'Only with special preprocessing', 'Only for binary targets'],
    correctAnswer: 1,
    explanation: 'K-NN regression predicts the average (or weighted average) of the target values of the K nearest neighbors.'
  },
  {
    id: 'knn14',
    question: 'What is weighted K-NN?',
    options: ['Using weighted features', 'Giving closer neighbors more influence', 'Using class weights', 'Using sample weights'],
    correctAnswer: 1,
    explanation: 'Weighted K-NN gives closer neighbors higher weights (e.g., inverse distance), reducing influence of farther neighbors.'
  },
  {
    id: 'knn15',
    question: 'What is a major disadvantage of K-NN?',
    options: ['Cannot fit data', 'Slow prediction and high memory usage', 'Cannot handle multiple features', 'Too complex'],
    correctAnswer: 1,
    explanation: 'K-NN stores all training data and computes distances at prediction time, making it slow and memory-intensive for large datasets.'
  },
  {
    id: 'knn16',
    question: 'How does K-NN handle imbalanced datasets?',
    options: ['Automatically handles it', 'May be biased toward majority class; can use distance weighting', 'Cannot handle imbalance', 'Always perfect'],
    correctAnswer: 1,
    explanation: 'K-NN can be biased toward majority classes. Solutions include distance weighting or adjusting K based on local density.'
  },
  {
    id: 'knn17',
    question: 'Is K-NN sensitive to irrelevant features?',
    options: ['No, robust to all features', 'Yes, very sensitive - irrelevant features add noise to distance', 'Only for large K', 'Only for small K'],
    correctAnswer: 1,
    explanation: 'Irrelevant features contribute to distance calculations and add noise, degrading performance. Feature selection is important.'
  },
  {
    id: 'knn18',
    question: 'Can K-NN handle missing values?',
    options: ['Yes, automatically', 'No, missing values must be imputed first', 'Only for some features', 'Only for small datasets'],
    correctAnswer: 1,
    explanation: 'K-NN cannot compute distances with missing values, so imputation or special distance metrics are needed.'
  },
  {
    id: 'knn19',
    question: 'What is the curse of dimensionality for K-NN?',
    options: ['Too many dimensions slow training', 'High dimensions make distances less meaningful', 'Cannot work in high dimensions', 'Only affects small datasets'],
    correctAnswer: 1,
    explanation: 'In high dimensions, all points become roughly equidistant, making nearest neighbors less meaningful and K-NN less effective.'
  },
  {
    id: 'knn20',
    question: 'When is K-NN most appropriate?',
    options: ['Always', 'Small to medium datasets with low dimensionality and non-linear patterns', 'Large datasets', 'High dimensional data'],
    correctAnswer: 1,
    explanation: 'K-NN works best on smaller datasets with low to medium dimensionality where non-linear decision boundaries are needed.'
  }
];

// K-Means Clustering - 20 questions
export const kMeansQuestions: QuizQuestion[] = [
  {
    id: 'km1',
    question: 'What type of learning is K-Means?',
    options: ['Supervised', 'Unsupervised clustering', 'Reinforcement', 'Semi-supervised'],
    correctAnswer: 1,
    explanation: 'K-Means is an unsupervised learning algorithm that groups similar data points into clusters without using labels.'
  },
  {
    id: 'km2',
    question: 'What does K represent in K-Means?',
    options: ['Number of features', 'Number of clusters', 'Number of iterations', 'Number of samples'],
    correctAnswer: 1,
    explanation: 'K is the number of clusters you want to partition the data into, which must be specified beforehand.'
  },
  {
    id: 'km3',
    question: 'What is the objective of K-Means?',
    options: ['Maximize inter-cluster distance', 'Minimize within-cluster sum of squared distances (inertia)', 'Maximize accuracy', 'Minimize features'],
    correctAnswer: 1,
    explanation: 'K-Means minimizes the within-cluster variance (WCSS), making points within each cluster as similar as possible.'
  },
  {
    id: 'km4',
    question: 'How does the K-Means algorithm work?',
    options: ['Single step', 'Iteratively: assign points to nearest centroid, update centroids', 'Random assignment', 'Hierarchical splitting'],
    correctAnswer: 1,
    explanation: 'K-Means alternates between assigning points to nearest centroids and updating centroids as the mean of assigned points.'
  },
  {
    id: 'km5',
    question: 'What is a centroid in K-Means?',
    options: ['Random point', 'Mean/center point of a cluster', 'Outlier', 'Boundary point'],
    correctAnswer: 1,
    explanation: 'A centroid is the center of a cluster, calculated as the mean position of all points assigned to that cluster.'
  },
  {
    id: 'km6',
    question: 'How are initial centroids typically chosen?',
    options: ['Deterministically', 'Random selection or K-Means++ algorithm', 'Always the same', 'By the user'],
    correctAnswer: 1,
    explanation: 'Initial centroids are randomly selected or chosen using K-Means++ for better initialization.'
  },
  {
    id: 'km7',
    question: 'What is K-Means++?',
    options: ['Faster K-Means', 'Smart initialization method that spreads initial centroids', 'K-Means with more clusters', 'Parallel K-Means'],
    correctAnswer: 1,
    explanation: 'K-Means++ chooses initial centroids probabilistically based on distance, leading to better convergence and results.'
  },
  {
    id: 'km8',
    question: 'When does K-Means converge?',
    options: ['After one iteration', 'When centroids no longer change significantly', 'Never', 'After K iterations'],
    correctAnswer: 1,
    explanation: 'K-Means converges when centroid positions stabilize and point assignments no longer change (or change minimally).'
  },
  {
    id: 'km9',
    question: 'What is inertia (WCSS) in K-Means?',
    options: ['Number of iterations', 'Sum of squared distances from points to their nearest centroid', 'Number of clusters', 'Accuracy'],
    correctAnswer: 1,
    explanation: 'Inertia (Within-Cluster Sum of Squares) measures how tight the clusters are - lower is better.'
  },
  {
    id: 'km10',
    question: 'What is the elbow method?',
    options: ['Optimization technique', 'Plotting inertia vs K to find optimal K at the "elbow"', 'Initialization method', 'Distance metric'],
    correctAnswer: 1,
    explanation: 'The elbow method plots inertia against different K values; the "elbow" point suggests a good K where additional clusters give diminishing returns.'
  },
  {
    id: 'km11',
    question: 'What shape clusters does K-Means work best with?',
    options: ['Any shape', 'Spherical/globular clusters', 'Elongated clusters', 'Irregular shapes'],
    correctAnswer: 1,
    explanation: 'K-Means assumes spherical clusters of similar size due to its use of Euclidean distance to centroids.'
  },
  {
    id: 'km12',
    question: 'Is K-Means sensitive to initialization?',
    options: ['No, always same result', 'Yes, different initializations can lead to different results', 'Only for small K', 'Only for large datasets'],
    correctAnswer: 1,
    explanation: 'K-Means can converge to local optima depending on initialization, which is why multiple runs with different seeds are recommended.'
  },
  {
    id: 'km13',
    question: 'Is K-Means sensitive to outliers?',
    options: ['No, robust to outliers', 'Yes, outliers can significantly affect centroid positions', 'Only for small datasets', 'Only for large K'],
    correctAnswer: 1,
    explanation: 'Outliers can skew centroid positions since K-Means uses mean, potentially creating poor clusters.'
  },
  {
    id: 'km14',
    question: 'Why is feature scaling important for K-Means?',
    options: ['Not important', 'K-Means uses distances, so features should be on similar scales', 'Only for speed', 'Only for visualization'],
    correctAnswer: 1,
    explanation: 'Features with larger scales will dominate distance calculations, so standardization is crucial for K-Means.'
  },
  {
    id: 'km15',
    question: 'What is the computational complexity of K-Means?',
    options: ['O(n)', 'O(n × K × i × d) where n=samples, K=clusters, i=iterations, d=dimensions', 'O(n²)', 'O(log n)'],
    correctAnswer: 1,
    explanation: 'K-Means complexity depends on number of samples (n), clusters (K), iterations (i), and dimensions (d).'
  },
  {
    id: 'km16',
    question: 'Can K-Means handle categorical features?',
    options: ['Yes, directly', 'No, requires encoding to numerical first (or use K-Modes)', 'Only binary categories', 'Only with scaling'],
    correctAnswer: 1,
    explanation: 'K-Means requires numerical features for distance calculation. Categorical data needs encoding or use K-Modes variant.'
  },
  {
    id: 'km17',
    question: 'What is the Silhouette Score?',
    options: ['Distance metric', 'Measure of how similar a point is to its cluster vs other clusters', 'Number of clusters', 'Iteration count'],
    correctAnswer: 1,
    explanation: 'Silhouette Score measures cluster quality, ranging from -1 to 1, with higher values indicating better-defined clusters.'
  },
  {
    id: 'km18',
    question: 'What happens if you choose K that\'s too small?',
    options: ['Overfitting', 'Underfitting - distinct groups forced into same cluster', 'Perfect clustering', 'Algorithm fails'],
    correctAnswer: 1,
    explanation: 'Too few clusters result in underfitting where distinct groups are merged, losing important structure.'
  },
  {
    id: 'km19',
    question: 'What happens if you choose K that\'s too large?',
    options: ['Better results', 'Overfitting - natural groups split unnecessarily', 'Faster convergence', 'Lower inertia always'],
    correctAnswer: 1,
    explanation: 'Too many clusters cause overfitting, splitting natural groups and finding patterns in noise.'
  },
  {
    id: 'km20',
    question: 'What is Mini-Batch K-Means?',
    options: ['K-Means for small data', 'K-Means using random subsets for faster training', 'K-Means with fewer clusters', 'K-Means++ variant'],
    correctAnswer: 1,
    explanation: 'Mini-Batch K-Means uses random subsets of data in each iteration, trading slight accuracy for much faster computation on large datasets.'
  }
];
