import { Topic } from '../../types';

export const foundationsTopics: Record<string, Topic> = {
  'supervised-vs-unsupervised-vs-reinforcement': {
    id: 'supervised-vs-unsupervised-vs-reinforcement',
    title: 'Supervised vs Unsupervised vs Reinforcement Learning',
    category: 'foundations',
    description: 'Understanding the three main paradigms of machine learning and their applications.',
    content: `
      <h2>Overview</h2>
      <p>Machine learning can be broadly categorized into three main paradigms based on the type of learning signal or feedback available to the learning system.</p>

      <h3>Supervised Learning</h3>
      <p>In supervised learning, algorithms learn from labeled training data to make predictions or decisions. The algorithm learns a mapping function from input variables (X) to output variables (Y).</p>

      <p><strong>Key characteristics:</strong></p>
      <ul>
        <li>Training data includes both input features and correct answers (labels)</li>
        <li>Goal is to predict outcomes for new, unseen data</li>
        <li>Performance can be measured against known correct answers</li>
      </ul>

      <p><strong>Common algorithms:</strong> Linear Regression, Logistic Regression, Decision Trees, Random Forest, SVM, Neural Networks</p>

      <h3>Unsupervised Learning</h3>
      <p>Unsupervised learning works with data that has no labels or target variables. The algorithm must find hidden patterns, structures, or relationships in the data.</p>

      <p><strong>Key characteristics:</strong></p>
      <ul>
        <li>No labeled examples or target variables</li>
        <li>Goal is to discover hidden patterns or structures</li>
        <li>More exploratory in nature</li>
      </ul>

      <p><strong>Common algorithms:</strong> K-Means Clustering, Hierarchical Clustering, PCA, Autoencoders</p>

      <h3>Reinforcement Learning</h3>
      <p>Reinforcement learning involves an agent learning to make decisions by taking actions in an environment to maximize cumulative reward.</p>

      <p><strong>Key characteristics:</strong></p>
      <ul>
        <li>Agent learns through trial and error</li>
        <li>Feedback comes in the form of rewards or penalties</li>
        <li>Goal is to learn an optimal policy for decision making</li>
      </ul>

      <p><strong>Common algorithms:</strong> Q-Learning, Policy Gradient, Actor-Critic methods</p>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `# Supervised Learning Example: Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Generate sample data
X = np.random.randn(100, 1)
y = 2 * X.flatten() + 1 + np.random.randn(100) * 0.1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)`,
        explanation: 'This example shows supervised learning where we have both input features (X) and target values (y) to train a linear regression model.'
      },
      {
        language: 'Python',
        code: `# Unsupervised Learning Example: K-Means Clustering
from sklearn.cluster import KMeans
import numpy as np

# Generate sample data (no labels)
X = np.random.randn(100, 2)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Get cluster centers
centers = kmeans.cluster_centers_`,
        explanation: 'This example shows unsupervised learning where we only have input data (X) and try to discover hidden structures (clusters) without any labels.'
      }
    ],
    interviewQuestions: [
      {
        question: 'What is the main difference between supervised and unsupervised learning?',
        answer: 'The fundamental difference lies in the availability and use of labeled data. In supervised learning, each training example comes with a label or target value that the model tries to predict. The algorithm learns a mapping function from inputs to outputs by minimizing the difference between its predictions and the true labels. For example, in image classification, each training image has a label indicating its class (cat, dog, etc.).\n\nUnsupervised learning, on the other hand, works with unlabeled data where no target values are provided. The algorithm must discover inherent structures, patterns, or relationships in the data without explicit guidance. Common tasks include clustering (grouping similar data points), dimensionality reduction (finding compact representations), and anomaly detection (identifying unusual patterns).\n\nThis distinction has profound implications for when each approach is applicable. Supervised learning requires labeled data which can be expensive and time-consuming to obtain, but provides clear optimization objectives and performance metrics. Unsupervised learning can work with abundant unlabeled data but has more subjective evaluation criteria since there\'s no ground truth to compare against.'
      },
      {
        question: 'Can you give examples of when you would use each type of learning?',
        answer: 'Supervised learning is ideal when you have labeled data and a clear prediction task. Common applications include spam email detection (labeled as spam/not spam), medical diagnosis (labeled patient outcomes), credit risk assessment (historical loan default data), and recommendation systems with explicit ratings. It\'s particularly valuable in production systems where you can collect labeled data from user feedback or expert annotations.\n\nUnsupervised learning excels when labels are unavailable, expensive to obtain, or when you want to discover hidden structures. Use cases include customer segmentation for marketing (grouping customers by behavior patterns), anomaly detection in network security (identifying unusual traffic patterns without labeled attacks), topic modeling in text analysis (discovering themes in document collections), and data preprocessing through dimensionality reduction before applying supervised methods.\n\nReinforcement learning is appropriate for sequential decision-making problems where an agent interacts with an environment. Classic examples include game playing (Chess, Go, Atari games), robotics (learning locomotion or manipulation), autonomous driving (navigating traffic), and resource allocation (managing server loads, trading algorithms). It\'s particularly powerful when the optimal strategy isn\'t obvious and must be learned through trial and error.'
      },
      {
        question: 'What are some challenges specific to unsupervised learning?',
        answer: 'The most significant challenge in unsupervised learning is the lack of objective evaluation metrics. Without ground truth labels, it\'s difficult to definitively assess whether the discovered patterns are meaningful or simply artifacts of the algorithm. Different clustering algorithms may produce vastly different results on the same data, and determining which is "correct" often requires domain expertise and subjective judgment.\n\nAnother major challenge is determining the right number of patterns or clusters. In k-means clustering, for example, you must specify k beforehand, but the optimal value is often unknown. While techniques like the elbow method or silhouette analysis can help, they provide guidance rather than definitive answers. This hyperparameter selection problem extends to other unsupervised methods like dimensionality reduction, where choosing the number of components involves balancing information preservation with compression.\n\nInterpretability and actionability of results can also be problematic. A clustering algorithm might group customers into distinct segments, but understanding why these groups formed and how to leverage them for business decisions requires additional analysis. The patterns discovered might be statistically valid but practically meaningless, or they might capture spurious correlations in the data rather than meaningful relationships.'
      },
      {
        question: 'How does reinforcement learning differ from supervised learning?',
        answer: 'The key difference is in the nature of feedback. Supervised learning receives immediate, explicit feedback for each prediction through labeled examples—if the model predicts "cat" for a dog image, it immediately knows it\'s wrong and by how much. The learning signal is direct and unambiguous. Reinforcement learning, however, receives delayed, sparse, and often ambiguous feedback through rewards. An action taken now might only show its consequences many steps later (credit assignment problem), and the reward signal doesn\'t explicitly tell the agent what it should have done differently.\n\nThe temporal and sequential nature of reinforcement learning creates additional complexity. In supervised learning, training examples are typically independent and identically distributed (i.i.d.), and you can shuffle and batch them freely. In RL, the agent\'s actions affect which states it visits next, creating dependencies between consecutive experiences. The agent must balance exploration (trying new actions to discover their effects) with exploitation (using known good actions), whereas supervised learning doesn\'t face this dilemma.\n\nReinforcement learning must also handle partial observability and learn from its own experience. The agent generates its own training data through interaction with the environment, and the distribution of this data depends on its current policy. This creates a moving target problem—as the agent improves, it visits different states, generating different training data. Additionally, RL typically optimizes long-term cumulative reward rather than minimizing error on individual predictions, requiring reasoning about trade-offs between immediate and future rewards.'
      },
      {
        question: 'What is the role of rewards in reinforcement learning?',
        answer: 'Rewards serve as the fundamental learning signal that guides the agent toward desirable behavior. They define the objective the agent is trying to optimize—maximizing cumulative expected reward over time. Unlike supervised learning where every action has explicit feedback, rewards in RL can be sparse (only received at episode end) or dense (received after every action), and this reward structure profoundly affects learning difficulty and speed.\n\nThe reward function effectively encodes what you want the agent to accomplish, making reward design critical. A poorly designed reward can lead to unintended behavior—for example, a robot rewarded for "moving forward" might learn to somersault endlessly rather than walk properly. This is called reward hacking or reward gaming. In practice, reward shaping (adding intermediate rewards to guide learning) can help, but must be done carefully to avoid introducing shortcuts that prevent learning the true objective.\n\nRewards also create the credit assignment problem—determining which past actions were responsible for current rewards. When an action\'s consequences only manifest many steps later (like in chess, where a move might enable a winning position much later), the agent must learn to assign credit appropriately. Techniques like temporal difference learning and eligibility traces help solve this by propagating reward information backward through the sequence of actions, allowing the agent to learn which early actions contributed to later success.'
      },
      {
        question: 'Can you think of a real-world example where reinforcement learning would be appropriate?',
        answer: 'Autonomous driving is an excellent example where reinforcement learning\'s strengths shine. The driving task inherently involves sequential decision-making in a dynamic environment with delayed consequences. An action like changing lanes doesn\'t immediately result in success or failure—its outcome depends on subsequent decisions and the behavior of other drivers. The agent must learn a policy that handles diverse scenarios (highway driving, city traffic, parking) while optimizing for multiple objectives: safety, passenger comfort, traffic rules compliance, and efficiency.\n\nThe environment provides natural reward signals: negative rewards for collisions, violations, or jerky movements; positive rewards for smooth, efficient navigation to the destination. The sparse reward structure (major rewards only at destination arrival or accidents) combined with dense intermediate rewards (for smooth driving, maintaining lanes) creates a complex learning problem. The agent must also handle partial observability (can\'t see around corners), uncertainty (unpredictable other drivers), and continuous state/action spaces.\n\nRL is particularly well-suited here because the optimal driving policy can\'t easily be manually specified—it emerges from experience across millions of diverse scenarios. Simulation environments allow safe exploration before real-world deployment. Transfer learning enables policies learned in simulation to adapt to reality. The approach also naturally handles the multi-agent aspect (other drivers) and can continuously improve through fleet learning, where experiences from all vehicles contribute to improving the shared policy.'
      }
    ],
    quizQuestions: [
      {
        id: 'q1',
        question: 'Which type of learning uses labeled training data?',
        options: ['Supervised Learning', 'Unsupervised Learning', 'Reinforcement Learning', 'Semi-supervised Learning'],
        correctAnswer: 0,
        explanation: 'Supervised learning uses labeled training data where both input features and correct output labels are provided.'
      },
      {
        id: 'q2',
        question: 'What is the main goal of unsupervised learning?',
        options: ['Predict future values', 'Discover hidden patterns', 'Maximize rewards', 'Classify data points'],
        correctAnswer: 1,
        explanation: 'Unsupervised learning aims to discover hidden patterns or structures in data without using labeled examples.'
      },
      {
        id: 'q3',
        question: 'In reinforcement learning, what guides the learning process?',
        options: ['Labeled examples', 'Hidden patterns', 'Rewards and penalties', 'Feature correlations'],
        correctAnswer: 2,
        explanation: 'Reinforcement learning uses rewards and penalties as feedback to guide the agent\'s learning process.'
      }
    ]
  },

  'bias-variance-tradeoff': {
    id: 'bias-variance-tradeoff',
    title: 'Bias-Variance Tradeoff',
    category: 'foundations',
    description: 'Understanding the fundamental tradeoff between bias and variance in machine learning models.',
    hasInteractiveDemo: true,
    content: `
      <h2>Understanding Bias-Variance Tradeoff</h2>
      <p>The bias-variance tradeoff is a fundamental concept in machine learning that describes the relationship between model complexity and generalization performance.</p>

      <h3>Bias</h3>
      <p>Bias refers to the error introduced by approximating a real-world problem with a simplified model. High bias can cause underfitting.</p>

      <p><strong>Characteristics of high bias:</strong></p>
      <ul>
        <li>Model is too simple</li>
        <li>Unable to capture underlying patterns</li>
        <li>Poor performance on both training and test data</li>
        <li>Underfitting occurs</li>
      </ul>

      <h3>Variance</h3>
      <p>Variance refers to the model's sensitivity to small fluctuations in the training data. High variance can cause overfitting.</p>

      <p><strong>Characteristics of high variance:</strong></p>
      <ul>
        <li>Model is too complex</li>
        <li>Sensitive to noise in training data</li>
        <li>Good training performance, poor test performance</li>
        <li>Overfitting occurs</li>
      </ul>

      <h3>The Tradeoff</h3>
      <p>The total error can be decomposed into three components:</p>
      <p><strong>Total Error = Bias² + Variance + Irreducible Error</strong></p>

      <p>Finding the optimal model complexity involves balancing bias and variance to minimize total error.</p>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 1, 100).reshape(-1, 1)
y = 1.5 * X.flatten() + 0.5 * np.sin(2 * np.pi * X.flatten()) + np.random.normal(0, 0.1, 100)

# Test different polynomial degrees
degrees = range(1, 16)
train_scores = []
val_scores = []

for degree in degrees:
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    # Train model
    model = LinearRegression()
    model.fit(X_poly, y)

    # Calculate scores
    train_score = model.score(X_poly, y)
    val_score = np.mean(cross_val_score(model, X_poly, y, cv=5))

    train_scores.append(train_score)
    val_scores.append(val_score)`,
        explanation: 'This code demonstrates how model complexity (polynomial degree) affects bias and variance by plotting training vs validation performance.'
      }
    ],
    interviewQuestions: [
      {
        question: 'Explain the bias-variance tradeoff in your own words.',
        answer: 'The bias-variance tradeoff is the fundamental tension between a model\'s ability to fit the training data well (low bias) and its ability to generalize to new data (low variance). It describes how increasing model complexity affects these two types of errors in opposite ways. Bias represents systematic errors from incorrect assumptions in the model—a high-bias model underfits, failing to capture the true relationship between features and targets. Variance represents sensitivity to fluctuations in the training data—a high-variance model overfits, learning noise and random patterns that don\'t generalize.\n\nMathematically, the expected prediction error can be decomposed into three components: bias squared, variance, and irreducible error. As you increase model complexity (adding polynomial terms, deepening neural networks, growing decision trees), bias tends to decrease because the model can capture more intricate patterns. However, variance increases because the model has more freedom to fit noise in the specific training sample. The irreducible error comes from inherent noise in the data and cannot be reduced by any model.\n\nThe optimal model lies at the sweet spot where the sum of bias and variance is minimized. Too simple, and high bias dominates (underfitting). Too complex, and high variance dominates (overfitting). In practice, this tradeoff guides model selection—you want the most complex model that doesn\'t overfit your validation data, balancing capacity to learn patterns with stability across different training samples. Techniques like regularization, cross-validation, and ensemble methods help manage this tradeoff.'
      },
      {
        question: 'What happens when a model has high bias? High variance?',
        answer: 'A high-bias model is too simple to capture the underlying patterns in your data, resulting in underfitting. Practically, this means poor performance on both training and test sets—the model can\'t even fit the training data well. For example, using linear regression to model a clearly non-linear relationship will yield high bias. The model makes strong assumptions that don\'t match reality, systematically missing important patterns. Signs include low training accuracy, similar (low) validation accuracy, and the model\'s predictions consistently deviating from actual values in predictable ways.\n\nA high-variance model is too complex and overfits the training data, capturing noise and random fluctuations rather than just the signal. This manifests as excellent training performance but poor test performance—a large gap between training and validation accuracy. The model essentially memorizes the training data rather than learning generalizable patterns. For instance, a very deep decision tree might perfectly classify all training examples by creating hyper-specific rules, but these rules won\'t transfer to new data. Small changes in the training set would produce wildly different models.\n\nThe practical implications differ significantly. High bias is often easier to diagnose (obviously poor performance) and fix (add complexity, more features, less regularization). High variance is trickier—the model appears to work during training, but fails silently on new data. Detection requires careful validation, and solutions involve reducing complexity (pruning, dropout, regularization), getting more training data, or using ensemble methods that average out the variance across multiple models.'
      },
      {
        question: 'How can you detect if your model is suffering from high bias or high variance?',
        answer: 'The most reliable diagnostic is comparing training and validation performance. Plot learning curves that show both training and validation error as functions of training set size. A high-bias model shows high error on both curves that converge to a similar value—adding more data doesn\'t help because the model is fundamentally too simple. The gap between training and validation error is small. If you see this pattern, your model is underfitting and needs more capacity: add features, use a more complex model family, reduce regularization, or train longer.\n\nA high-variance model shows a large gap between training and validation error. Training error is low (the model fits the training data well), but validation error is much higher and may even increase with more complex models. Learning curves for high variance show training error continuing to decrease while validation error plateaus or increases. This gap indicates overfitting. Solutions include regularization (L1/L2 penalties, dropout), reducing model complexity (fewer features, shallower networks, tree pruning), getting more training data, or using techniques like early stopping.\n\nCross-validation provides additional insight. High variance manifests as high variability in performance across different validation folds—the model is unstable and sensitive to which specific samples were included in training. High bias shows consistent (but poor) performance across folds. You can also examine predictions directly: high bias models make systematic errors (consistently over or under predicting in certain regions), while high variance models make erratic errors that seem random and depend heavily on training data specifics. Residual plots and prediction intervals can help visualize these patterns.'
      },
      {
        question: 'What techniques can you use to reduce bias? To reduce variance?',
        answer: 'To reduce bias (address underfitting), you need to increase model capacity and flexibility. Add more features through feature engineering or polynomial features to give the model more information. Use a more complex model class—switch from linear to polynomial regression, from shallow to deeper neural networks, or from simple models to ensemble methods. Reduce regularization strength (lower lambda in L1/L2 penalties, reduce dropout rate). Train longer to ensure the model has fully learned the available patterns. Remove or weaken constraints that may be preventing the model from capturing important relationships.\n\nTo reduce variance (address overfitting), apply regularization techniques that penalize complexity. L1 regularization (Lasso) encourages sparsity and feature selection. L2 regularization (Ridge) penalizes large weights, keeping them small and stable. Dropout randomly deactivates neurons during training, preventing co-adaptation. Early stopping halts training when validation performance stops improving. Reduce model complexity directly: use fewer features through feature selection, shallower networks, pruned trees, or simpler model classes. Most importantly, gather more training data if possible—more data generally reduces variance significantly.\n\nEnsemble methods offer a sophisticated approach to reducing variance without increasing bias. Bagging (Bootstrap Aggregating) trains multiple models on different data subsets and averages predictions, reducing variance through averaging. Random forests extend this for decision trees. Boosting sequentially builds models that correct previous mistakes, reducing both bias and variance. Cross-validation helps navigate the tradeoff by providing unbiased performance estimates. The key is diagnosing which problem you have first (via learning curves), then applying the appropriate solution—don\'t add regularization if you have high bias, and don\'t increase complexity if you have high variance.'
      },
      {
        question: 'How does model complexity relate to the bias-variance tradeoff?',
        answer: 'Model complexity sits at the heart of the bias-variance tradeoff, controlling the balance between these two error sources. As complexity increases—more parameters, deeper architectures, higher-degree polynomials—bias systematically decreases because the model can represent more intricate functions and capture subtle patterns. Simultaneously, variance increases because the model has more degrees of freedom to fit noise and peculiarities of the specific training sample. The relationship is often visualized as a U-shaped curve for total error: initially, increasing complexity reduces bias faster than it increases variance (total error decreases), but eventually variance growth dominates (total error increases).\n\nDifferent model classes have different inherent complexity levels. Linear models have low complexity: a line (in 2D) or hyperplane (in higher dimensions) has limited capacity regardless of dataset size, leading to high bias in non-linear problems. Polynomial regression complexity depends on degree—quadratic adds curvature, cubic adds inflection points, and very high degrees can wiggle through every training point (high variance). Neural networks\' complexity scales with depth and width: more layers and neurons enable learning hierarchical abstractions but risk overfitting without proper regularization. Decision trees grow more complex with depth: deep trees partition the space finely (can overfit), shallow trees use crude partitions (can underfit).\n\nThe optimal complexity depends on the problem, data quantity, and noise level. With abundant clean data, you can afford higher complexity because variance is kept in check by the large sample. With limited or noisy data, simpler models often generalize better. This is why no single model dominates—the "No Free Lunch" theorem essentially states that averaged over all possible problems, all models perform equally. In practice, you navigate complexity through cross-validation: try multiple complexity levels, measure generalization via validation, and select the complexity that minimizes validation error. Regularization offers fine-grained control, letting you use high-capacity models while penalizing complexity, effectively tuning the complexity-to-data ratio.'
      }
    ],
    quizQuestions: [
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
      }
    ]
  },

  'train-validation-test-split': {
    id: 'train-validation-test-split',
    title: 'Train-Validation-Test Split',
    category: 'foundations',
    description: 'Understanding data splitting strategies for model development and evaluation',
    content: `
      <h2>Train-Validation-Test Split</h2>
      <p>Splitting data into separate sets is a fundamental practice in machine learning to ensure models generalize well to unseen data and to prevent overfitting.</p>

      <h3>Purpose of Each Split</h3>
      <ul>
        <li><strong>Training Set (60-80%):</strong> Used to train the model by adjusting weights and parameters. The model learns patterns from this data.</li>
        <li><strong>Validation Set (10-20%):</strong> Used for hyperparameter tuning and model selection. Helps in evaluating different model architectures without touching the test set.</li>
        <li><strong>Test Set (10-20%):</strong> Used only once at the end to evaluate the final model's performance. Provides an unbiased estimate of model generalization.</li>
      </ul>

      <h3>Common Split Ratios</h3>
      <p>The most common splitting strategies include:</p>
      <ul>
        <li><strong>70-15-15 split:</strong> Balanced approach for medium-sized datasets</li>
        <li><strong>80-10-10 split:</strong> When you have sufficient data and want more training examples</li>
        <li><strong>60-20-20 split:</strong> When you need robust validation and testing</li>
      </ul>

      <h3>Best Practices</h3>
      <ul>
        <li><strong>Stratification:</strong> For classification problems, maintain class distribution across all splits</li>
        <li><strong>Shuffling:</strong> Randomly shuffle data before splitting to avoid bias from data ordering</li>
        <li><strong>Time-Based Splits:</strong> For time-series data, use chronological splits instead of random sampling</li>
        <li><strong>Data Leakage Prevention:</strong> Ensure no information from validation/test sets influences training</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `from sklearn.model_selection import train_test_split
import numpy as np

# Sample dataset
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)

# First split: separate test set (80-20 split)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Second split: separate validation from training
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")`,
        explanation: 'This example demonstrates the standard approach for creating train-validation-test splits with stratification to maintain class balance.'
      },
      {
        language: 'Python',
        code: `import pandas as pd

# Time-series dataset example
dates = pd.date_range('2020-01-01', periods=1000, freq='D')
df = pd.DataFrame({
    'date': dates,
    'value': np.random.randn(1000)
})

# Chronological split (NO shuffling for time series)
train_size = int(0.6 * len(df))
val_size = int(0.2 * len(df))

train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:train_size + val_size]
test_df = df.iloc[train_size + val_size:]

print(f"Train: {train_df['date'].min()} to {train_df['date'].max()}")
print(f"Val: {val_df['date'].min()} to {val_df['date'].max()}")
print(f"Test: {test_df['date'].min()} to {test_df['date'].max()}")`,
        explanation: 'For time-series data, we must preserve chronological order and never shuffle. The test set contains the most recent data.'
      }
    ],
    interviewQuestions: [
      {
        question: 'Why do we need three separate datasets instead of just training and testing?',
        answer: 'The three-way split—training, validation, and test—serves distinct purposes in the machine learning pipeline. The training set is used to fit model parameters (weights, coefficients). The validation set is used for hyperparameter tuning and model selection without touching the test set. The test set provides an unbiased estimate of final model performance on truly unseen data. Without this separation, you risk overfitting your model selection process to the test set.\n\nWith just training and testing, you face a dilemma during model development. If you tune hyperparameters (learning rate, regularization strength, tree depth) based on test performance, you\'re indirectly fitting the test set—not through direct training, but through the iterative model selection process. After dozens of experiments choosing the model with best test performance, that test performance becomes an overoptimistic estimate. The test set has been "used up" through repeated evaluation. The validation set solves this by providing a separate dataset for these model selection decisions, preserving the test set for a final, unbiased evaluation.\n\nIn practice, the workflow is: train multiple models on training data, evaluate them on validation data to choose the best architecture/hyperparameters, then report final performance on the untouched test set. This discipline ensures honest performance reporting. Some practitioners use k-fold cross-validation for model selection (the validation phase), which uses the training data more efficiently. The key principle remains: the test set must only be used once, at the very end, after all model decisions are finalized. This prevents "validation set overfitting" and maintains statistical validity of your performance claims.'
      },
      {
        question: 'What is data leakage and how can improper splitting cause it?',
        answer: 'Data leakage occurs when information from outside the training data influences the model in ways that won\'t be available at prediction time, leading to overly optimistic performance estimates and poor real-world results. Improper splitting is a common source of leakage. The most basic form is test set leakage: accidentally including test samples in training, or applying transformations (normalization, feature engineering) on the combined dataset before splitting. This gives the model information about the test distribution during training, inflating performance metrics.\n\nTemporal leakage is particularly insidious with time-series data. If you shuffle before splitting, future information leaks into the training set—the model learns from tomorrow to predict yesterday, which is impossible in deployment. For example, in stock price prediction, shuffling mixes future prices into training, yielding unrealistically good results. The correct approach is chronological splitting: train on oldest data, validate on middle data, test on most recent. Similarly, with patient medical records, training on later visits while testing on earlier ones leaks information about disease progression.\n\nFeature engineering leakage is subtle but critical. If you compute statistics (mean, standard deviation, min/max for normalization) using all data before splitting, your training set knows about test set statistics. The solution is to compute these statistics only on training data, then apply the same transformation to validation and test sets. Other leakage sources include duplicate samples across sets (common with oversampling), target variable information in features (e.g., a "was_converted" feature in a conversion prediction task), or using forward-looking information (features that wouldn\'t be available at prediction time, like "total_purchases_this_year" in a model predicting January purchases).'
      },
      {
        question: 'How would you split a highly imbalanced dataset?',
        answer: 'For imbalanced datasets, stratified splitting is essential to maintain class distribution across train, validation, and test sets. Without stratification, random splitting might put most or all minority class samples in one set, making it impossible to learn or evaluate that class properly. Stratified sampling ensures each split contains approximately the same percentage of each class as the full dataset. For example, with 95% negative and 5% positive samples, stratification ensures training, validation, and test sets each have roughly this 95:5 ratio.\n\nThe implementation is straightforward in sklearn: use stratify parameter in train_test_split, passing the target labels. For multi-way splits, apply stratification twice: first split off the test set with stratification, then split the remainder into train/validation, again with stratification. This preserves class distribution at each step. For extreme imbalance (99:1 or worse), consider absolute counts—ensure the minority class has enough samples in each set for meaningful learning and evaluation, even if it means adjusting split ratios.\n\nBeyond stratification, consider your evaluation strategy. With severe imbalance, accuracy is meaningless (predicting all majority class gives high accuracy), so use appropriate metrics: precision, recall, F1-score, AUC-ROC, or AUC-PR. Your validation set must be large enough to reliably estimate these metrics for the minority class. Sometimes stratified k-fold cross-validation is better than a single train/val/test split, as it provides more robust estimates and uses data more efficiently. If the imbalance is so extreme that even stratified splitting leaves too few minority samples per fold, consider stratified sampling with replacement or advanced techniques like stratified group k-fold for grouped data.'
      },
      {
        question: 'Why should you never shuffle time-series data before splitting?',
        answer: 'Shuffling time-series data before splitting creates severe temporal leakage, fundamentally breaking the prediction task. In time-series problems, you\'re predicting the future based on the past—the temporal ordering is intrinsic to the problem. Shuffling mixes future observations into the training set, allowing the model to learn from future data when predicting the past. This produces artificially inflated performance that completely fails in production, where future data isn\'t available.\n\nThe consequences extend beyond just leakage. Many time-series have autocorrelation (correlation between observations at different time lags) and trends. Shuffling destroys these temporal dependencies that the model needs to learn. For example, in stock prices, consecutive days are correlated—today\'s price informs tomorrow\'s. Shuffling breaks these correlations, creating a jumbled dataset that doesn\'t reflect the sequential nature of the real problem. Your model might learn spurious patterns from the shuffled data that don\'t exist in actual time sequences.\n\nThe correct approach is chronological splitting: use oldest data for training, recent data for validation, and most recent for testing. This mimics deployment conditions where you train on historical data and predict future values. For cross-validation with time-series, use specialized techniques like TimeSeriesSplit which respects temporal order, creating multiple train/test splits where each test set is later than its corresponding training set. Walk-forward validation is another approach, where you repeatedly train on historical windows and test on the immediate next period, rolling forward through time. These methods maintain temporal integrity while still providing robust performance estimates.'
      },
      {
        question: 'If you have only 500 samples, what splitting strategy would you recommend?',
        answer: 'With limited data (500 samples), every sample is precious, and traditional splits (60/20/20 or 70/15/15) leave validation and test sets too small for reliable performance estimates. K-fold cross-validation is typically the best approach here—it uses data more efficiently by ensuring every sample serves in both training and validation across different folds. For 500 samples, 5 or 10-fold cross-validation works well: each fold uses 80-90% of data for training and 10-20% for validation, providing more robust performance estimates through averaging.\n\nThe workflow changes slightly: instead of a single validation set, you train k models (one per fold) and report average validation performance plus standard deviation across folds. This gives both a performance estimate and uncertainty quantification. For hyperparameter tuning, use nested cross-validation: an outer loop for performance estimation and an inner loop for hyperparameter selection within each outer fold. This prevents overfitting the validation process while maximizing data usage. The computational cost increases linearly with k, but with only 500 samples, this is usually manageable.\n\nIf you need a held-out test set for final evaluation (recommended for production models), consider a modified approach: set aside a stratified 15-20% test set (75-100 samples), then use cross-validation on the remaining 80-85% for model development. This balances efficient data usage during development with an unbiased final test. Alternatively, use repeated k-fold cross-validation (running k-fold multiple times with different random seeds) or leave-one-out cross-validation (LOOCV, where k equals sample size) for very small datasets, though LOOCV has high variance and computational cost. The key is avoiding waste through excessive splitting while maintaining reliable performance estimates through resampling techniques.'
      },
      {
        question: 'What is stratified splitting and when should you use it?',
        answer: 'Stratified splitting ensures that each split (train, validation, test) maintains the same class distribution as the original dataset. Instead of random sampling, stratified sampling samples separately from each class proportionally. If your dataset has 70% class A and 30% class B, stratified splitting ensures each set has approximately the same 70:30 ratio. This is implemented by sampling 70% of class A samples and 70% of class B samples for training, leaving 30% of each for validation/testing, then further splitting that 30% into validation and test sets.\n\nYou should use stratified splitting for any classification task with imbalanced classes. Even moderate imbalance (60:40) can benefit, as it reduces variance in performance estimates and ensures all classes are represented in each set. For severe imbalance (95:5 or worse), stratification is critical—random splitting might accidentally place most minority class samples in one set, making it impossible to train or evaluate properly. Stratification also matters for small datasets where random fluctuations could create misleading splits. For example, with 100 samples and 20% minority class, random splitting might give training sets with 15-25% minority samples just by chance, whereas stratification ensures consistent 20%.\n\nBeyond binary classification, use stratified splitting for multi-class problems to maintain representation of all classes, especially if some classes are rare. For continuous regression targets, you can create stratified splits by binning the target into quantiles and stratifying on these bins—this ensures each set spans the full range of target values rather than accidentally concentrating high values in training and low values in testing. Don\'t use stratified splitting for time-series (violates temporal ordering) or when class distribution is expected to shift between training and deployment (though this indicates a more fundamental problem with your modeling approach).'
      }
    ],
    quizQuestions: [
      {
        id: 'split1',
        question: 'What is the primary purpose of the validation set?',
        options: [
          'To train the model',
          'To tune hyperparameters and select models',
          'To provide final performance evaluation',
          'To augment training data'
        ],
        correctAnswer: 1,
        explanation: 'The validation set is used for hyperparameter tuning and model selection during development, without touching the test set.'
      },
      {
        id: 'split2',
        question: 'For time-series prediction, how should you split your data?',
        options: [
          'Randomly shuffle and split',
          'Use stratified sampling',
          'Split chronologically with earlier data for training',
          'Use k-fold cross-validation with random folds'
        ],
        correctAnswer: 2,
        explanation: 'Time-series data must be split chronologically to simulate real prediction scenarios where you predict future from past.'
      }
    ]
  },

  'overfitting-underfitting': {
    id: 'overfitting-underfitting',
    title: 'Overfitting and Underfitting',
    category: 'foundations',
    description: 'Understanding model complexity and the bias-variance tradeoff in practice',
    content: `
      <h2>Overfitting and Underfitting</h2>
      <p>Overfitting and underfitting represent the two extremes of model complexity, both resulting in poor generalization to new data.</p>

      <h3>Underfitting (High Bias)</h3>
      <p>Occurs when a model is too simple to capture underlying patterns:</p>
      <ul>
        <li><strong>Symptoms:</strong> Poor performance on both training and test sets</li>
        <li><strong>Causes:</strong> Model too simple, insufficient features, excessive regularization</li>
        <li><strong>Solutions:</strong> Increase model complexity, add more features, reduce regularization</li>
      </ul>

      <h3>Overfitting (High Variance)</h3>
      <p>Occurs when a model learns training data too well, including noise:</p>
      <ul>
        <li><strong>Symptoms:</strong> Excellent training performance but poor test performance</li>
        <li><strong>Causes:</strong> Model too complex, too many features, insufficient training data</li>
        <li><strong>Solutions:</strong> Add more data, use regularization, reduce complexity, early stopping</li>
      </ul>

      <h3>The Bias-Variance Tradeoff</h3>
      <p><strong>Total Error = Bias² + Variance + Irreducible Error</strong></p>
      <ul>
        <li><strong>Bias:</strong> Error from incorrect assumptions. High bias leads to underfitting.</li>
        <li><strong>Variance:</strong> Sensitivity to training data fluctuations. High variance leads to overfitting.</li>
        <li><strong>Irreducible Error:</strong> Noise in data that cannot be reduced.</li>
      </ul>

      <h3>Detection Methods</h3>
      <ul>
        <li><strong>Learning Curves:</strong> Plot training vs validation error</li>
        <li><strong>Underfitting Pattern:</strong> Both curves plateau at high error</li>
        <li><strong>Overfitting Pattern:</strong> Large gap between training (low) and validation (high) error</li>
        <li><strong>Good Fit:</strong> Both curves converge at low error with small gap</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(42)
X = np.sort(np.random.rand(100, 1) * 10, axis=0)
y = np.sin(X).ravel() + np.random.randn(100) * 0.3

X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# Test different polynomial degrees
degrees = [1, 4, 15]  # Underfitting, Good fit, Overfitting

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    train_mse = mean_squared_error(y_train, model.predict(X_train_poly))
    test_mse = mean_squared_error(y_test, model.predict(X_test_poly))

    print(f"Degree {degree}: Train MSE={train_mse:.4f}, Test MSE={test_mse:.4f}, Gap={abs(test_mse-train_mse):.4f}")`,
        explanation: 'This demonstrates underfitting (degree 1), good fit (degree 4), and overfitting (degree 15) using polynomial regression. Notice the gap between train and test error.'
      },
      {
        language: 'Python',
        code: `from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    if val_mean[-1] < 0.7 and train_mean[-1] < 0.75:
        print("UNDERFITTING - both scores low")
    elif train_mean[-1] - val_mean[-1] > 0.1:
        print("OVERFITTING - large gap")
    else:
        print("GOOD FIT - small gap, good performance")

# Test different model complexities
simple_model = RandomForestClassifier(max_depth=2)
complex_model = RandomForestClassifier(max_depth=None)`,
        explanation: 'Learning curves help diagnose overfitting/underfitting by showing how performance changes with training set size.'
      }
    ],
    interviewQuestions: [
      {
        question: 'Explain the bias-variance tradeoff.',
        answer: 'The bias-variance tradeoff is the fundamental tension in machine learning between a model\'s ability to capture complex patterns (low bias) and its sensitivity to noise in the training data (low variance). Bias refers to errors from overly simplistic assumptions—a high-bias model underfits, unable to capture the true relationship between features and target. Variance refers to errors from excessive sensitivity to training data fluctuations—a high-variance model overfits, learning noise as if it were signal.\n\nMathematically, the expected prediction error decomposes into three components: bias squared (systematic error from wrong assumptions), variance (error from sensitivity to training sample), and irreducible error (inherent noise). As model complexity increases—adding parameters, deepening networks, growing trees—bias decreases because the model can represent more complex functions, but variance increases because the model has more freedom to fit noise. The total error typically forms a U-shape: initially decreasing as bias reduction outweighs variance increase, then increasing as variance dominates.\n\nThe practical implication is that there\'s no universally "best" model complexity—it depends on your data quantity, noise level, and true underlying pattern. With abundant clean data, you can afford complex models because large samples stabilize variance. With limited noisy data, simpler models often generalize better. The goal is finding the sweet spot that minimizes total error, which is what techniques like cross-validation help achieve. Regularization offers a nuanced approach, using complex models but penalizing certain types of complexity to manage the tradeoff.'
      },
      {
        question: 'How do you detect if your model is overfitting or underfitting?',
        answer: 'The primary diagnostic tool is comparing training and validation performance. Underfitting (high bias) manifests as poor performance on both training and validation sets—the model can\'t even fit the training data well. Training and validation errors are high and similar, with a small gap between them. On learning curves (error vs. training size), both curves plateau at high error values and converge. If you see this, your model is too simple: try adding features, increasing model complexity (deeper networks, higher polynomial degree), reducing regularization, or training longer.\n\nOverfitting (high variance) shows excellent training performance but poor validation performance—a large gap between training and validation error. The model memorizes training data rather than learning generalizable patterns. On learning curves, training error is low and continues decreasing, while validation error is much higher and may even increase with more complex models. The curves don\'t converge even with more data. Solutions include regularization (L1/L2, dropout), reducing complexity (feature selection, shallower models), early stopping, or gathering more training data.\n\nAdditional indicators include cross-validation variance: high variance models show high performance variability across folds (unstable, dependent on which samples were in the training set), while high bias models show consistent but poor performance. Examining predictions directly also helps—overfitting models make erratic errors that seem random, while underfitting models make systematic errors (consistently off in certain regions). Regularization path plots (performance vs. regularization strength) help identify the optimal point: decreasing regularization from high values first improves performance (reducing bias), then harms it (increasing variance).'
      },
      {
        question: 'Your model has 99% training accuracy but 65% test accuracy. What would you do?',
        answer: 'This is classic overfitting—a 34 percentage point gap between training (99%) and test (65%) accuracy indicates the model is memorizing training data rather than learning generalizable patterns. My first step would be to analyze whether 65% is actually problematic for the task—if random guessing gives 50% for binary classification, 65% might be reasonable given data quality. But assuming we need better generalization, I\'d proceed systematically through several interventions.\n\nFirst, apply regularization to penalize model complexity. For linear models, add L1 or L2 penalties. For neural networks, implement dropout (randomly deactivating neurons during training) and/or L2 weight decay. For decision trees, limit depth, require minimum samples per leaf, or prune after training. Start with moderate regularization and tune via validation set. Second, reduce model complexity directly: use feature selection to remove irrelevant features, decrease network depth/width, or use a simpler model class altogether. Third, implement early stopping: monitor validation performance during training and stop when it stops improving, even if training accuracy could go higher.\n\nIf these don\'t sufficiently close the gap, gather more training data if possible—more data is often the most effective overfitting cure, as it reduces variance. Use data augmentation if applicable (image rotations/crops, text paraphrasing). Employ ensemble methods like bagging or random forests that average multiple models to reduce variance. Cross-validation during model selection ensures you\'re not accidentally selecting hyperparameters that overfit. Finally, verify there\'s no data leakage (test samples in training, feature engineering using test set statistics) and that train/test distributions are similar (if test comes from different distribution, the gap might reflect distribution shift rather than overfitting).'
      },
      {
        question: 'Why does adding more training data help with overfitting but not underfitting?',
        answer: 'Overfitting fundamentally stems from the model having too much capacity relative to available data, allowing it to fit noise and random fluctuations in the finite training sample. With limited data, a complex model can find spurious patterns that look predictive in the training set but don\'t generalize. Adding more training data helps because it reduces the variance component of error—with more samples, random noise averages out, and the model must find patterns that hold across a larger, more representative sample. The model\'s capacity remains constant, but the effective data-to-parameter ratio increases, reducing the model\'s ability to memorize noise.\n\nMathematically, variance decreases roughly as 1/n where n is training size. As you add data, the model\'s predictions become more stable—less dependent on which specific samples happened to be in the training set. Eventually, with enough data, even complex models stop overfitting because there\'s insufficient freedom to fit noise while achieving low training error on the large sample. This is why deep learning works: given millions or billions of training examples, massive neural networks (billions of parameters) can generalize well despite their huge capacity.\n\nUnderfitting, however, arises from insufficient model capacity to capture the true underlying pattern, regardless of data quantity. If you\'re using linear regression for a clearly non-linear relationship, adding more data just gives you more evidence of the same systematic error—the model still can\'t capture the non-linearity. Learning curves for underfitting show both training and validation error high and plateaued; more data doesn\'t help because the problem is the model\'s representational capacity, not estimation variance. The solution is increasing model capacity (more features, higher polynomial degree, deeper networks), not more data. Of course, after increasing capacity, you might then need more data to avoid overfitting with your now-complex model, illustrating the interconnection between model complexity, data size, and the bias-variance tradeoff.'
      },
      {
        question: 'What is the difference between high bias and high variance?',
        answer: 'High bias and high variance represent opposite failure modes in machine learning, corresponding to underfitting and overfitting respectively. High bias occurs when the model is too simple to capture the underlying data pattern, making strong, incorrect assumptions about the relationship between features and target. It results in systematic errors—the model consistently misses important patterns. For example, using linear regression for a clearly quadratic relationship yields high bias: the straight line can\'t capture the curvature regardless of how you optimize it. Symptoms include poor training accuracy, similar (poor) validation accuracy, and small gap between them.\n\nHigh variance occurs when the model is too complex and overfits training data, capturing noise and random fluctuations as if they were meaningful patterns. The model is excessively sensitive to the specific training sample—small changes in training data produce wildly different models. For instance, a very deep decision tree might perfectly classify all training examples by creating hyper-specific rules that don\'t generalize. Symptoms include excellent training accuracy, much worse validation accuracy, and large gap between them. The model performs differently on different validation folds (unstable), and predictions seem erratic rather than systematic.\n\nThe bias-variance tradeoff creates tension between these errors. Addressing high bias (underfitting) requires increasing model complexity: add features, use more complex model families, reduce regularization, train longer. Addressing high variance (overfitting) requires the opposite: regularization, reduced complexity, more training data, early stopping, or ensemble methods. Crucially, techniques that fix one often exacerbate the other. Adding polynomial features reduces bias (can now capture non-linearity) but increases variance (more parameters to fit noise). Adding L2 regularization reduces variance (keeps weights small and stable) but increases bias (constrains the function space). The art of machine learning is diagnosing which problem you have (via learning curves and validation metrics) and applying appropriate interventions to find the optimal complexity level for your specific dataset.'
      }
    ],
    quizQuestions: [
      {
        id: 'ou1',
        question: 'A model achieves 95% training accuracy and 60% test accuracy. What is the problem?',
        options: ['Underfitting', 'Overfitting', 'High bias', 'Perfect fit'],
        correctAnswer: 1,
        explanation: 'The large gap between training (95%) and test (60%) accuracy indicates overfitting - the model memorized the training data.'
      },
      {
        id: 'ou2',
        question: 'Which scenario indicates underfitting?',
        options: ['Train: 5%, Test: 25%', 'Train: 35%, Test: 40%', 'Train: 2%, Test: 3%', 'Train: 10%, Test: 35%'],
        correctAnswer: 1,
        explanation: 'Underfitting shows high error on both training (35%) and test (40%) sets with a small gap, indicating the model is too simple.'
      }
    ]
  },

  'regularization': {
    id: 'regularization',
    title: 'Regularization (L1, L2, Dropout)',
    category: 'foundations',
    description: 'Techniques to prevent overfitting and improve model generalization',
    content: `
      <h2>Regularization</h2>
      <p>Regularization techniques add constraints to prevent overfitting by encouraging simpler models that generalize better.</p>

      <h3>L2 Regularization (Ridge)</h3>
      <p><strong>Formula:</strong> Loss = Original Loss + λ × Σ(w²)</p>
      <ul>
        <li>Shrinks weights toward zero but rarely makes them exactly zero</li>
        <li>Prevents weights from becoming too large</li>
        <li>Also called weight decay</li>
        <li>Use when you want to keep all features but reduce their impact</li>
      </ul>

      <h3>L1 Regularization (Lasso)</h3>
      <p><strong>Formula:</strong> Loss = Original Loss + λ × Σ|w|</p>
      <ul>
        <li>Can drive weights to exactly zero (feature selection)</li>
        <li>Creates sparse models</li>
        <li>Good for interpretability and identifying important features</li>
        <li>Can be unstable with correlated features</li>
      </ul>

      <h3>Dropout (for Neural Networks)</h3>
      <ul>
        <li>Randomly drops neurons during training (typically 20-50%)</li>
        <li>Prevents co-adaptation of neurons</li>
        <li>Only applied during training, not inference</li>
        <li>Acts like training an ensemble of networks</li>
      </ul>

      <h3>Other Regularization Techniques</h3>
      <ul>
        <li><strong>Early Stopping:</strong> Stop training when validation error increases</li>
        <li><strong>Data Augmentation:</strong> Artificially increase training data</li>
        <li><strong>Batch Normalization:</strong> Has regularization side effect</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `from sklearn.linear_model import Ridge, Lasso, ElasticNet
import numpy as np

X = np.random.randn(200, 20)
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(200) * 0.5

# L2 Regularization (Ridge)
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
print(f"Ridge - Non-zero coefficients: {np.sum(np.abs(ridge.coef_) > 0.01)}/20")

# L1 Regularization (Lasso)
lasso = Lasso(alpha=1.0)
lasso.fit(X, y)
print(f"Lasso - Non-zero coefficients: {np.sum(np.abs(lasso.coef_) > 0.01)}/20")
print(f"Lasso - Exactly zero: {np.sum(lasso.coef_ == 0)}")

# Elastic Net (combines L1 and L2)
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic.fit(X, y)`,
        explanation: 'L2 keeps all features but shrinks coefficients. L1 performs feature selection by setting some coefficients to zero. Elastic Net combines both.'
      },
      {
        language: 'Python',
        code: `import tensorflow as tf
from tensorflow.keras import layers, models

# Model with L2 regularization and Dropout
model = models.Sequential([
    layers.Dense(128, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dropout(0.3),  # 30% dropout
    layers.Dense(64, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')`,
        explanation: 'Neural networks typically use both L2 regularization (weight decay) and Dropout for effective regularization.'
      }
    ],
    interviewQuestions: [
      {
        question: 'What is the difference between L1 and L2 regularization?',
        answer: 'L1 (Lasso) and L2 (Ridge) regularization both penalize large weights but in fundamentally different ways with distinct consequences. L2 regularization adds a penalty term proportional to the sum of squared weights (λ∑w²) to the loss function. This encourages weights to be small but doesn\'t force them exactly to zero—weights shrink proportionally toward zero but rarely reach it exactly. The penalty is differentiable everywhere, making optimization straightforward with gradient descent. L2 tends to spread weights across all features, giving many features small non-zero weights.\n\nL1 regularization adds a penalty proportional to the sum of absolute values of weights (λ∑|w|). The key difference is that L1 actively drives some weights exactly to zero, performing automatic feature selection. The absolute value creates a non-differentiable point at zero, which geometrically favors sparse solutions—many weights become exactly zero while others remain relatively large. This makes L1 useful when you suspect many features are irrelevant or want an interpretable model with fewer active features. L1 can be more computationally expensive to optimize due to the non-smooth penalty.\n\nThe geometric intuition helps: visualize the loss surface and the constraint region (where the penalty equals a constant). For L2, this region is a circle/sphere (smooth), so the optimal point tends to have non-zero values in all dimensions. For L1, the region is a diamond/polytope with sharp corners along axes—solutions often land on these corners where some coordinates are exactly zero. Practically, L2 is the default choice for general regularization (stable, easy to optimize, good generalization), while L1 is chosen when you want sparsity/feature selection or suspect the true model uses only a subset of available features. Elastic Net combines both, getting benefits of each: L1\'s sparsity and L2\'s grouping of correlated features.'
      },
      {
        question: 'How does dropout work and why does it prevent overfitting?',
        answer: 'Dropout is a regularization technique for neural networks where, during each training step, we randomly "drop" (set to zero) a fraction of neurons (typically 20-50%) along with their connections. For each training batch, a different random subset of neurons is dropped, meaning each forward/backward pass uses a different sub-network. This randomness prevents neurons from co-adapting—they can\'t rely on the presence of other specific neurons, forcing each to learn more robust features independently useful for making predictions.\n\nDropout prevents overfitting through multiple mechanisms. First, it acts like training an ensemble of exponentially many different sub-networks (2^n possible networks for n neurons), then averaging their predictions. Ensembles reduce variance by averaging out individual model errors, similar to how random forests average many decision trees. Second, it prevents complex co-adaptations where specific combinations of neurons fire together to memorize training data. Without dropout, a neuron might learn to correct another neuron\'s mistakes on training data, creating brittle dependencies that don\'t generalize. Dropout breaks these dependencies, forcing more distributed representations.\n\nDuring training, dropped neurons don\'t participate in forward propagation or backpropagation for that iteration. The remaining neurons must compensate, learning to make good predictions even when their partners are absent. At inference time, dropout is turned off (all neurons active), but their outputs are scaled by the dropout probability to account for more neurons being active than during training. This ensures expected output magnitude matches training conditions. Modern implementations often use "inverted dropout" which scales up during training instead, avoiding extra computation at inference. The dropout rate is a hyperparameter: higher rates provide stronger regularization but can lead to underfitting; typical values are 0.2-0.5 for hidden layers, 0.5 for fully-connected layers, and lower (0.1-0.2) or zero for convolutional layers which are less prone to overfitting.'
      },
      {
        question: 'When would you use L1 over L2 regularization?',
        answer: 'Choose L1 regularization when you want automatic feature selection and suspect many features are irrelevant. L1 drives weights exactly to zero, effectively removing features from the model, producing sparse solutions where only important features have non-zero weights. This is valuable when interpretability matters—a model using 10 out of 1000 features is much easier to understand and deploy than one using all features with small weights. In domains like genomics or text analysis where you have thousands or millions of features but believe only a few drive the outcome, L1\'s sparsity is crucial.\n\nL1 is also preferable when features are highly correlated. L2 tends to give correlated features similar weights (spreading penalty across both), while L1 typically picks one and zeros out the others. This arbitrary selection among correlated features isn\'t ideal for inference but can improve computational efficiency (fewer active features) and prevent multicollinearity issues. For high-dimensional datasets where p > n (more features than samples), L1 can identify a small subset of predictive features, making the problem tractable.\n\nUse L2 in most other scenarios: when you want to use all features but prevent overfitting, when features aren\'t clearly categorizable as relevant/irrelevant, when you need stable gradient-based optimization, or when computationally cheaper solutions matter (L2 has closed-form solutions for some models like linear/ridge regression). L2 tends to give slightly better predictive performance when most features are at least weakly relevant. Elastic Net combines both penalties (αL1 + (1-α)L2), letting you tune between sparsity and stable shrinkage, often outperforming either alone. In neural networks, L2 (weight decay) is more common than L1 because the network\'s architecture already provides feature learning, but L1 can be used for structured sparsity (e.g., pruning entire channels). The choice ultimately depends on your goals: predictive performance only → L2 or Elastic Net; interpretability and feature selection → L1 or Elastic Net with high α.'
      },
      {
        question: 'What happens to dropout during inference?',
        answer: 'During inference (making predictions on new data), dropout is turned off entirely—all neurons are active and contribute to the prediction. However, to maintain consistent output magnitudes, the neuron outputs must be scaled appropriately. During training with dropout probability p, each neuron\'s output is randomly set to zero with probability p, so the expected value of its output is (1-p) times its actual computed value. To match this expected behavior at inference where all neurons are active, we need to scale outputs.\n\nThere are two equivalent approaches. Standard dropout scales neuron outputs at inference by multiplying them by (1-p). If you trained with p=0.5 dropout, at inference you multiply each neuron\'s output by 0.5, ensuring the magnitude matches training expectations. The alternative, inverted dropout (more common in modern implementations), does the scaling during training instead: when a neuron isn\'t dropped during training, its output is divided by (1-p), scaling it up to compensate for other neurons being dropped. At inference with inverted dropout, you simply use all neurons without any scaling—cleaner and computationally cheaper since inference happens more frequently than training.\n\nThe mathematical justification is maintaining E[output] consistent between training and inference. During training, each neuron has probability (1-p) of being active with scaled output, and probability p of being inactive (zero output). The expected output is (1-p) × (scaled value). At inference, all neurons are always active, so without adjustment, the expected output would be higher, creating a train-test mismatch. The scaling correction ensures the network sees similar activation magnitudes whether in training or inference mode, preventing unexpected behavior when deploying the model. Frameworks like TensorFlow and PyTorch handle this automatically—you set model.train() for training mode (dropout active) or model.eval() for evaluation mode (dropout off, appropriate scaling applied), and the framework manages the details.'
      },
      {
        question: 'If your model is underfitting, should you increase or decrease regularization?',
        answer: 'If your model is underfitting (high bias), you should decrease regularization or remove it entirely. Regularization penalizes model complexity, intentionally constraining the model to prevent overfitting. When underfitting, the problem is the opposite—your model is too simple and can\'t capture the underlying patterns in the data. Adding more constraints through regularization makes this worse, further limiting the model\'s capacity to fit the training data. Reducing regularization allows the model more freedom to learn complex patterns and fit the training data better.\n\nConcretely, if using L1 or L2 regularization, reduce the regularization parameter λ (sometimes called alpha). Smaller λ means less penalty on large weights, allowing the model to use its full capacity. If using dropout in neural networks, reduce the dropout rate or remove dropout from some layers. If applying early stopping, train for more epochs to let the model fully learn available patterns. The extreme case is λ=0 or dropout rate=0, meaning no regularization at all, which is appropriate when underfitting is severe.\n\nThe diagnostic pattern is: if you see poor performance on both training and validation sets with a small gap between them, you have high bias (underfitting). The solution is to increase model capacity, which includes reducing regularization but also adding features, using more complex model architectures (deeper networks, higher polynomial degrees, more trees in ensemble), or training longer. After reducing regularization and increasing capacity, you might then see overfitting (large train-test gap), at which point you\'d reintroduce regularization at a moderate level. The goal is finding the sweet spot: enough model capacity to capture patterns (low bias) with sufficient regularization to prevent fitting noise (low variance). This is typically found through cross-validation across different regularization strengths, choosing the value that minimizes validation error.'
      }
    ],
    quizQuestions: [
      {
        id: 'reg1',
        question: 'What is the main advantage of L1 over L2 regularization?',
        options: ['Faster computation', 'Automatic feature selection', 'Always better accuracy', 'Works better with neural networks'],
        correctAnswer: 1,
        explanation: 'L1 regularization performs automatic feature selection by driving some coefficients to exactly zero, creating sparse models.'
      },
      {
        id: 'reg2',
        question: 'During inference in a neural network with dropout, what happens?',
        options: ['30% of neurons are dropped', 'All neurons are active', 'Random neurons are dropped', 'Only training neurons are used'],
        correctAnswer: 1,
        explanation: 'During inference, all neurons are active and dropout is turned off. Weights or outputs are scaled to account for this.'
      }
    ]
  },

  'cross-validation': {
    id: 'cross-validation',
    title: 'Cross-Validation',
    category: 'foundations',
    description: 'Robust techniques for evaluating model performance and preventing overfitting',
    content: `
      <h2>Cross-Validation</h2>
      <p>Cross-validation is a resampling technique used to evaluate machine learning models on limited data samples. It provides a more robust estimate of model performance than a single train-test split.</p>

      <h3>Why Cross-Validation?</h3>
      <ul>
        <li><strong>Reduces variance:</strong> Multiple train-test splits provide more reliable performance estimates</li>
        <li><strong>Better data utilization:</strong> All data is used for both training and validation</li>
        <li><strong>Detects overfitting:</strong> Identifies models that memorize training data</li>
        <li><strong>Model selection:</strong> Compare different algorithms and hyperparameters objectively</li>
      </ul>

      <h3>K-Fold Cross-Validation</h3>
      <p>The most common approach:</p>
      <ul>
        <li>Split data into k equal-sized folds (typically k=5 or k=10)</li>
        <li>Train on k-1 folds, validate on the remaining fold</li>
        <li>Repeat k times, each time with a different validation fold</li>
        <li>Average the k performance scores for final estimate</li>
      </ul>

      <h3>Stratified K-Fold</h3>
      <p>Ensures each fold maintains the same class distribution as the original dataset. Essential for:</p>
      <ul>
        <li>Imbalanced classification problems</li>
        <li>Small datasets where class distribution matters</li>
        <li>Ensuring representative samples in each fold</li>
      </ul>

      <h3>Time Series Cross-Validation</h3>
      <p>For temporal data where future cannot leak into past:</p>
      <ul>
        <li><strong>Forward chaining:</strong> Train on past, validate on future</li>
        <li>Each fold uses progressively more training data</li>
        <li>Never shuffle time-ordered data</li>
        <li>Respects temporal dependencies</li>
      </ul>

      <h3>Leave-One-Out Cross-Validation (LOOCV)</h3>
      <p>Special case where k = n (number of samples):</p>
      <ul>
        <li>Maximum data utilization for training</li>
        <li>Computationally expensive for large datasets</li>
        <li>High variance in performance estimates</li>
        <li>Useful for very small datasets</li>
      </ul>

      <h3>Nested Cross-Validation</h3>
      <p>Two-level CV for unbiased hyperparameter tuning:</p>
      <ul>
        <li><strong>Outer loop:</strong> Estimates true model performance</li>
        <li><strong>Inner loop:</strong> Tunes hyperparameters</li>
        <li>Prevents information leakage from validation to test</li>
        <li>Gold standard for model evaluation</li>
      </ul>

      <h3>Best Practices</h3>
      <ul>
        <li>Use stratified k-fold for classification tasks</li>
        <li>Choose k=5 or k=10 for good bias-variance tradeoff</li>
        <li>Use time series CV for temporal data</li>
        <li>Report both mean and standard deviation of CV scores</li>
        <li>Set random seed for reproducibility</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np

# Create sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2,
                           weights=[0.7, 0.3], random_state=42)

# Initialize model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Standard k-fold cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"5-Fold CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
print(f"Individual fold scores: {scores}")

# Stratified k-fold (maintains class distribution)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratified_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
print(f"\\nStratified 5-Fold CV: {stratified_scores.mean():.3f} (+/- {stratified_scores.std():.3f})")

# Multiple scoring metrics
from sklearn.model_selection import cross_validate
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
results = cross_validate(model, X, y, cv=5, scoring=scoring, return_train_score=True)

print("\\nMultiple Metrics:")
for metric in scoring:
    test_score = results[f'test_{metric}'].mean()
    train_score = results[f'train_{metric}'].mean()
    print(f"{metric}: Train={train_score:.3f}, Test={test_score:.3f}")`,
        explanation: 'Demonstrates standard k-fold, stratified k-fold, and multi-metric cross-validation for classification. Stratified CV is crucial for imbalanced datasets.'
      },
      {
        language: 'Python',
        code: `from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Simulate time series data
n_samples = 1000
X = np.random.randn(n_samples, 10)
y = np.cumsum(np.random.randn(n_samples))  # Time-dependent target

# Time series cross-validation (respects temporal order)
tscv = TimeSeriesSplit(n_splits=5)

model = RandomForestRegressor(random_state=42)
scores = []

print("Time Series Cross-Validation:")
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    scores.append(score)

    print(f"Fold {fold+1}: Train size={len(train_idx)}, Val size={len(val_idx)}, Score={score:.3f}")

print(f"\\nMean R² Score: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")

# Nested cross-validation for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None]
}

# Inner CV: hyperparameter tuning
inner_cv = TimeSeriesSplit(n_splits=3)
clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv, scoring='r2')

# Outer CV: performance estimation
outer_cv = TimeSeriesSplit(n_splits=5)
nested_scores = cross_val_score(clf, X, y, cv=outer_cv, scoring='r2')

print(f"\\nNested CV R² Score: {nested_scores.mean():.3f} (+/- {nested_scores.std():.3f})")`,
        explanation: 'Shows time series cross-validation that respects temporal order, and nested CV for unbiased hyperparameter tuning. Essential for financial, weather, or any time-dependent data.'
      }
    ],
    interviewQuestions: [
      {
        question: 'Why is cross-validation better than a single train-test split?',
        answer: 'Cross-validation provides more reliable and robust performance estimates than a single train-test split by using your data more efficiently and reducing variance in the evaluation. With a single split, your performance estimate depends heavily on which specific samples happened to land in the test set—you might get lucky (easy test samples) or unlucky (hard test samples), leading to misleading conclusions. Cross-validation averages performance across multiple different train-test splits, giving both an expected performance (the mean across folds) and uncertainty quantification (standard deviation across folds).\n\nThe data efficiency argument is compelling, especially for small datasets. In a single 80-20 split, you train on 80% of data and evaluate on 20%. In 5-fold cross-validation, each fold trains on 80% and evaluates on the remaining 20%, but across the 5 folds, every data point serves in testing exactly once and in training four times. You get five performance estimates instead of one, each on a different 20% of the data, providing much more thorough evaluation coverage. For small datasets where every sample is precious, this efficiency is crucial.\n\nCross-validation also helps detect overfitting to the test set through model selection. If you try many model variations and select the one with best test performance on a single held-out test set, that test performance becomes overly optimistic—you\'ve indirectly fitted the test set through the selection process. Cross-validation mitigates this because the same test samples aren\'t used repeatedly; each fold sees different test data. The main downside is computational cost (k-fold requires training k models instead of one), but for most applications this is manageable and worthwhile for the improved reliability. For production systems, it\'s common to use cross-validation during development for robust model selection, then train a final model on all available data once the architecture and hyperparameters are chosen.'
      },
      {
        question: 'When should you use stratified k-fold cross-validation?',
        answer: 'Use stratified k-fold cross-validation whenever you have imbalanced class distributions in classification tasks. Stratified sampling ensures that each fold maintains approximately the same class distribution as the overall dataset. Without stratification, random folding might create folds with very different class distributions—one fold might have 10% positive class while another has 25%—making performance estimates unreliable and training unstable. For example, with 95% negative and 5% positive classes, a random fold might accidentally have zero positive samples, making it impossible to compute recall or F1-score for that fold.\n\nStratification is critical for severe imbalance (99:1 or worse) but beneficial even for moderate imbalance (70:30). It reduces variance in performance estimates across folds and ensures all folds are representative. This means you get more consistent fold-to-fold results, making the average performance a better estimate of true generalization. Stratification also ensures that all classes appear in both training and validation sets for each fold, which is essential for the model to learn all classes and for metrics to be computable.\n\nFor multi-class problems, stratified cross-validation maintains the proportion of all classes across folds, which is especially important if some classes are rare. For regression tasks, you can create stratified folds by binning the continuous target into quantiles and stratifying on these bins, ensuring each fold spans the full range of target values rather than concentrating high values in some folds and low values in others. The only situations where you shouldn\'t use stratification are: time-series data (where temporal order must be preserved), grouped data (where samples within groups must stay together), or when class distribution is expected to differ between training and deployment (though this indicates a more fundamental problem). In sklearn, it\'s as simple as using StratifiedKFold instead of KFold, with no computational downside.'
      },
      {
        question: 'What is the main limitation of leave-one-out cross-validation (LOOCV)?',
        answer: 'The primary limitation of LOOCV is computational cost—it requires training n models where n is the number of samples, which becomes prohibitive for large datasets or computationally expensive models (deep neural networks, gradient boosting with many trees). Training thousands or millions of models is simply infeasible in most practical scenarios. Unlike k-fold CV where you can choose k=5 or 10 to balance reliability and computation, LOOCV has no such flexibility; the number of folds equals your sample size.\n\nA more subtle but equally important limitation is high variance in the performance estimate. LOOCV has low bias (each training set contains n-1 samples, nearly all the data), but high variance because the n training sets are highly correlated—they overlap in n-2 samples and differ by only one sample. Changes in that single different sample can\'t create much variation in the resulting models, so the n performance estimates are not independent. Averaging non-independent estimates doesn\'t reduce variance as effectively as averaging independent estimates would. Empirically, 5 or 10-fold CV often gives performance estimates with lower variance than LOOCV, despite using less data per fold.\n\nLOOCV can also be misleading for model selection with certain algorithms. Since each model is trained on nearly all data (n-1 samples), the performance estimates are optimistic compared to training on your actual training set size (which would be smaller if you held out a proper test set). For unstable algorithms (high-variance models like decision trees or k-NN with small k), LOOCV can produce highly variable predictions across folds, making the average performance less meaningful. LOOCV is primarily useful for small datasets (n<100) where you can\'t afford to lose 20% of data to a validation fold, and for algorithms where training is cheap (linear models, k-NN). For most modern applications with moderate-to-large datasets and complex models, 5 or 10-fold cross-validation is preferred as a better balance of statistical properties, computational cost, and practical utility.'
      },
      {
        question: 'How does time series cross-validation differ from standard k-fold CV?',
        answer: 'Time series cross-validation must respect temporal ordering, whereas standard k-fold CV randomly shuffles data before splitting. The fundamental principle is that you can only train on past data and validate on future data, never the reverse. Shuffling destroys this temporal structure, creating leakage where future information influences training. Standard k-fold would train on a random 80% (including future observations) and test on a random 20% (including past observations), which is nonsensical for time series—you can\'t predict yesterday using tomorrow\'s data.\n\nTimeSeriesSplit in sklearn implements the correct approach using expanding or rolling windows. In expanding window mode, each successive fold includes all previous data: fold 1 trains on samples 1-100 and tests on 101-150; fold 2 trains on 1-150 and tests on 151-200; fold 3 trains on 1-200 and tests on 201-250, etc. This mimics realistic deployment where you continuously retrain on all historical data. Rolling window mode maintains fixed training size: fold 1 uses 1-100 for training; fold 2 uses 51-150; fold 3 uses 101-200, etc. Rolling windows are useful when recent data is more relevant (concept drift) or when computational constraints limit training on all historical data.\n\nA crucial difference is that test sets must always come after training sets chronologically. This creates fewer, sequential folds rather than random permutations. You also can\'t use all data equally—early data appears in training more often than late data, and the final data only appears in test sets. This is intentional and necessary to prevent leakage. When evaluating performance, be aware that each fold tests on different time periods which might have different characteristics (seasonality, trends, regime changes). Report performance on each fold separately in addition to the average, as this reveals whether your model performance is stable over time or degrades for certain periods. Never use standard k-fold, stratified k-fold, or LOOCV for time series—they all violate temporal causality and will produce misleadingly optimistic results that fail catastrophically in production.'
      },
      {
        question: 'What is nested cross-validation and when is it necessary?',
        answer: 'Nested cross-validation is a two-level cross-validation procedure that separates model selection (choosing hyperparameters) from performance estimation. The outer loop provides an unbiased estimate of the final model\'s generalization performance, while the inner loop performs hyperparameter tuning without contaminating the outer performance estimate. This is necessary whenever you need both reliable hyperparameter optimization and honest performance reporting, particularly for research or production systems where accurate performance guarantees matter.\n\nThe structure involves an outer k-fold split (typically 5-fold) for performance estimation, and for each outer fold, an inner k-fold split (typically 3 or 5-fold) for hyperparameter tuning. For each outer fold: take the outer training data, run inner cross-validation across different hyperparameter values, select the best hyperparameters based on inner validation performance, train a model with those hyperparameters on the full outer training data, and evaluate on the outer test fold. Repeat for all outer folds, then average the outer test performances. This gives an unbiased performance estimate because the outer test folds were never used for hyperparameter selection.\n\nWithout nested CV, if you use the same cross-validation splits for both hyperparameter tuning and performance estimation, you get overly optimistic estimates. After trying many hyperparameter combinations and selecting the best based on CV performance, that CV performance is biased upward—you\'ve indirectly fitted the validation data through the selection process. Nested CV solves this by keeping outer test data completely isolated from the model selection process. The computational cost is significant (k_outer × k_inner × n_hyperparameter_combinations models), but necessary for honest reporting. In practice, use nested CV when publishing research (to report unbiased performance), deploying high-stakes models (medical, financial), or when you need confidence intervals on performance. For informal model comparison or when computational budget is tight, standard CV for hyperparameter tuning followed by a separate held-out test set is a reasonable compromise.'
      },
      {
        question: 'Why might you get overly optimistic performance estimates if you tune hyperparameters using the same CV splits?',
        answer: 'This creates a subtle form of overfitting where you indirectly fit the validation data through the hyperparameter selection process, even though you never directly trained on validation samples. When you try many hyperparameter combinations (50 learning rates, 10 regularization strengths, 5 architectures = 2500 combinations) and select the one with best cross-validation performance, you\'re essentially running 2500 experiments and choosing the luckiest result. Some combinations will perform well by chance—random fluctuations in the specific validation samples favor certain hyperparameters. Reporting the best CV score as your expected performance is overly optimistic.\n\nThe validation data has been "used up" through repeated evaluation. Each time you evaluate a new hyperparameter configuration on the validation folds, you gain information about those specific samples and adjust your choices accordingly. After extensive hyperparameter search, the selected configuration is optimized for the peculiarities of those validation folds, not just for the underlying data distribution. This is particularly severe with automated hyperparameter optimization (grid search, random search, Bayesian optimization) that might evaluate hundreds or thousands of configurations. The more configurations you try, the more likely you are to find one that excels on your validation set by chance.\n\nThe solution is nested cross-validation or a three-way split. Nested CV uses inner folds for hyperparameter selection and outer folds for unbiased performance estimation. The three-way approach uses training data for model fitting, validation data for hyperparameter selection, and a completely held-out test set for final performance reporting. The test set must only be evaluated once after all model decisions are finalized. The magnitude of optimism depends on search intensity: trying 5 hyperparameter values introduces modest bias, while trying 1000 introduces substantial bias. This is why kaggle competitions often have public and private leaderboards—the public leaderboard (validation set) is visible during the competition for model development, but final ranking uses the hidden private leaderboard (test set) to prevent overfitting to the public scores through repeated submissions.'
      }
    ],
    quizQuestions: [
      {
        id: 'cv-q1',
        question: 'You are building a fraud detection model where only 2% of transactions are fraudulent. Which cross-validation strategy is most appropriate?',
        options: [
          'Standard k-fold cross-validation',
          'Stratified k-fold cross-validation',
          'Leave-one-out cross-validation (LOOCV)',
          'Simple train-test split'
        ],
        correctAnswer: 1,
        explanation: 'Stratified k-fold ensures each fold maintains the 2% fraud rate. Standard k-fold might create folds with 0% or highly variable fraud rates, leading to unreliable performance estimates.'
      },
      {
        id: 'cv-q2',
        question: 'You are predicting stock prices using historical data. Your model performs well in cross-validation (R²=0.85) but poorly in production (R²=0.30). What is the most likely cause?',
        options: [
          'The model is underfitting',
          'You used standard k-fold CV instead of time series CV, causing data leakage',
          'The test set is too small',
          'The model needs more regularization'
        ],
        correctAnswer: 1,
        explanation: 'Standard k-fold randomly shuffles data, allowing the model to "peek" at future information during training. Time series CV respects temporal order, training only on past data to predict future values.'
      },
      {
        id: 'cv-q3',
        question: 'When performing hyperparameter tuning with GridSearchCV, why should you use nested cross-validation for final model evaluation?',
        options: [
          'It trains faster than single-level CV',
          'It prevents data leakage between hyperparameter tuning and performance estimation',
          'It requires less data than standard CV',
          'It always produces higher accuracy scores'
        ],
        correctAnswer: 1,
        explanation: 'Using the same CV folds for both hyperparameter tuning and performance estimation leaks information, giving overly optimistic results. Nested CV uses separate outer folds for unbiased performance estimation.'
      }
    ]
  },

  'evaluation-metrics': {
    id: 'evaluation-metrics',
    title: 'Evaluation Metrics',
    category: 'foundations',
    description: 'Understanding and selecting appropriate metrics for different ML tasks',
    content: `
      <h2>Evaluation Metrics</h2>
      <p>Choosing the right evaluation metric is crucial for assessing model performance and aligning with business objectives. Different metrics capture different aspects of model behavior.</p>

      <h3>Classification Metrics</h3>

      <h4>Confusion Matrix</h4>
      <p>Foundation for understanding classification performance:</p>
      <ul>
        <li><strong>True Positive (TP):</strong> Correctly predicted positive class</li>
        <li><strong>True Negative (TN):</strong> Correctly predicted negative class</li>
        <li><strong>False Positive (FP):</strong> Incorrectly predicted positive (Type I error)</li>
        <li><strong>False Negative (FN):</strong> Incorrectly predicted negative (Type II error)</li>
      </ul>

      <h4>Accuracy</h4>
      <p><strong>Accuracy = (TP + TN) / (TP + TN + FP + FN)</strong></p>
      <ul>
        <li>Simple and intuitive</li>
        <li><strong>Problem:</strong> Misleading for imbalanced datasets</li>
        <li>Example: 99% accuracy on 99% negative class just predicts everything as negative</li>
      </ul>

      <h4>Precision</h4>
      <p><strong>Precision = TP / (TP + FP)</strong></p>
      <ul>
        <li>Of all positive predictions, how many were correct?</li>
        <li>Minimizes false alarms</li>
        <li>Use when FP cost is high (spam detection, medical diagnosis)</li>
      </ul>

      <h4>Recall (Sensitivity)</h4>
      <p><strong>Recall = TP / (TP + FN)</strong></p>
      <ul>
        <li>Of all actual positives, how many did we catch?</li>
        <li>Minimizes missed cases</li>
        <li>Use when FN cost is high (cancer detection, fraud detection)</li>
      </ul>

      <h4>F1 Score</h4>
      <p><strong>F1 = 2 × (Precision × Recall) / (Precision + Recall)</strong></p>
      <ul>
        <li>Harmonic mean of precision and recall</li>
        <li>Balances both metrics</li>
        <li>Good default for imbalanced datasets</li>
        <li>Use when you need balance between precision and recall</li>
      </ul>

      <h4>ROC-AUC</h4>
      <p>Receiver Operating Characteristic - Area Under Curve:</p>
      <ul>
        <li>Plots TPR (recall) vs FPR at various thresholds</li>
        <li>AUC = 0.5: random classifier, AUC = 1.0: perfect classifier</li>
        <li>Threshold-independent metric</li>
        <li>Useful for comparing models across different thresholds</li>
        <li><strong>Problem:</strong> Optimistic for highly imbalanced datasets</li>
      </ul>

      <h4>Precision-Recall AUC</h4>
      <ul>
        <li>Better than ROC-AUC for imbalanced datasets</li>
        <li>Focuses on positive class performance</li>
        <li>More informative when minority class matters most</li>
      </ul>

      <h3>Regression Metrics</h3>

      <h4>Mean Squared Error (MSE)</h4>
      <p><strong>MSE = (1/n) Σ(yᵢ - ŷᵢ)²</strong></p>
      <ul>
        <li>Penalizes large errors heavily (squared term)</li>
        <li>Same units as target variable squared</li>
        <li>Sensitive to outliers</li>
      </ul>

      <h4>Root Mean Squared Error (RMSE)</h4>
      <p><strong>RMSE = √MSE</strong></p>
      <ul>
        <li>Same units as target variable</li>
        <li>More interpretable than MSE</li>
        <li>Standard choice for regression</li>
      </ul>

      <h4>Mean Absolute Error (MAE)</h4>
      <p><strong>MAE = (1/n) Σ|yᵢ - ŷᵢ|</strong></p>
      <ul>
        <li>More robust to outliers than MSE/RMSE</li>
        <li>Linear penalty for errors</li>
        <li>Use when outliers should not dominate</li>
      </ul>

      <h4>R² Score (Coefficient of Determination)</h4>
      <p><strong>R² = 1 - (SS_res / SS_tot)</strong></p>
      <ul>
        <li>Measures proportion of variance explained</li>
        <li>Range: (-∞, 1], where 1 = perfect fit</li>
        <li>0 = model performs as well as mean baseline</li>
        <li>Negative = model worse than predicting mean</li>
        <li>Not comparable across different datasets</li>
      </ul>

      <h3>Metric Selection Guide</h3>
      <ul>
        <li><strong>Balanced classification:</strong> Accuracy, F1</li>
        <li><strong>Imbalanced classification:</strong> F1, PR-AUC, class-weighted metrics</li>
        <li><strong>High FP cost:</strong> Precision</li>
        <li><strong>High FN cost:</strong> Recall</li>
        <li><strong>Ranking/probability:</strong> ROC-AUC, Log Loss</li>
        <li><strong>Regression (general):</strong> RMSE, R²</li>
        <li><strong>Regression (with outliers):</strong> MAE, Huber loss</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
import numpy as np

# Simulate predictions for imbalanced dataset (5% positive class)
np.random.seed(42)
y_true = np.random.choice([0, 1], size=1000, p=[0.95, 0.05])
y_pred = np.random.choice([0, 1], size=1000, p=[0.90, 0.10])
y_proba = np.random.rand(1000)  # Predicted probabilities

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
print(f"TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}\\n")

# Basic metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}\\n")

# Threshold-independent metrics
roc_auc = roc_auc_score(y_true, y_proba)
pr_auc = average_precision_score(y_true, y_proba)

print(f"ROC-AUC: {roc_auc:.3f}")
print(f"PR-AUC:  {pr_auc:.3f}\\n")

# Comprehensive classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

# Imbalanced dataset example
print("\\n--- Imbalanced Dataset Analysis ---")
# Model 1: Always predicts negative
y_pred_baseline = np.zeros(1000)
print(f"Baseline (all negative) - Accuracy: {accuracy_score(y_true, y_pred_baseline):.3f}")
print(f"Baseline F1: {f1_score(y_true, y_pred_baseline, zero_division=0):.3f}")`,
        explanation: 'Comprehensive classification metrics evaluation showing how accuracy can be misleading on imbalanced datasets. F1 score and PR-AUC provide better insight into model performance on minority class.'
      },
      {
        language: 'Python',
        code: `from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Generate sample regression data with outliers
np.random.seed(42)
y_true = np.random.randn(100) * 10 + 50
y_pred = y_true + np.random.randn(100) * 5

# Add some outliers
y_true[95:] = [100, 105, 110, 95, 102]
y_pred[95:] = [60, 65, 58, 62, 63]  # Model fails on outliers

# Calculate regression metrics
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("Regression Metrics:")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")
print(f"R²:   {r2:.3f}\\n")

# Compare metrics with and without outliers
y_true_clean = y_true[:95]
y_pred_clean = y_pred[:95]

rmse_clean = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
mae_clean = mean_absolute_error(y_true_clean, y_pred_clean)
r2_clean = r2_score(y_true_clean, y_pred_clean)

print("Without Outliers:")
print(f"RMSE: {rmse_clean:.2f} (vs {rmse:.2f} with outliers)")
print(f"MAE:  {mae_clean:.2f} (vs {mae:.2f} with outliers)")
print(f"R²:   {r2_clean:.3f} (vs {r2:.3f} with outliers)\\n")

# RMSE vs MAE sensitivity
print("Impact of outliers:")
print(f"RMSE increased by: {((rmse - rmse_clean) / rmse_clean * 100):.1f}%")
print(f"MAE increased by:  {((mae - mae_clean) / mae_clean * 100):.1f}%")
print("\\nRMSE is more sensitive to outliers due to squaring errors!")`,
        explanation: 'Compares regression metrics (MSE, RMSE, MAE, R²) and demonstrates how RMSE is more sensitive to outliers than MAE. Essential for choosing appropriate metrics based on data characteristics.'
      }
    ],
    interviewQuestions: [
      {
        question: 'Why is accuracy a poor metric for imbalanced datasets? What metrics should you use instead?',
        answer: 'Accuracy is misleading for imbalanced datasets because a naive model that always predicts the majority class can achieve high accuracy without learning anything useful. For example, in fraud detection where 99% of transactions are legitimate, a model that classifies everything as "not fraud" achieves 99% accuracy while being completely useless—it catches zero fraud cases. Accuracy treats all errors equally, but in imbalanced scenarios, errors on the minority class are typically much more costly and important than errors on the majority class.\n\nFor imbalanced classification, use metrics that account for both classes properly. Precision measures what fraction of positive predictions are correct (TP / (TP + FP)), crucial when false positives are costly. Recall (also called sensitivity or TPR) measures what fraction of actual positives you catch (TP / (TP + FN)), critical when missing positive cases is dangerous. F1-score is the harmonic mean of precision and recall, providing a single metric that balances both. For severe imbalance, precision-recall curves and PR-AUC (area under precision-recall curve) are more informative than ROC-AUC because they focus on performance on the minority class.\n\nAlternatively, use metrics designed for imbalance. Balanced accuracy averages recall across classes, preventing majority class dominance. Cohen\'s Kappa measures agreement above chance, accounting for class imbalance. Matthews Correlation Coefficient (MCC) is a balanced metric that works well for imbalanced datasets, returning a value between -1 and 1. For multi-class imbalance, use macro-averaged metrics (compute metric per class, then average) rather than micro-averaged (aggregate all classes\' true positives, false positives, etc. first). The key is choosing metrics aligned with your business objective: if catching all fraud is critical, optimize recall; if false alarms are expensive, optimize precision; for balance, use F1 or F2 (weighs recall higher) scores.'
      },
      {
        question: 'When would you optimize for precision vs recall? Provide real-world examples.',
        answer: 'Optimize for precision when false positives are expensive or harmful, and you can tolerate missing some true positives. Email spam filtering is a classic example: marking a legitimate email as spam (false positive) is very bad—users might miss important messages from clients, jobs, or family. Missing some spam (false negative) is annoying but acceptable. You want high precision so that emails marked as spam are almost certainly spam. Similarly, in content moderation for removing illegal content, you want high precision to avoid accidentally censoring legitimate speech, even if it means some bad content slips through initially.\n\nOther precision-focused scenarios include: medical treatment recommendations (giving wrong treatment is worse than suggesting conservative monitoring), legal document review (marking wrong documents as relevant wastes expensive lawyer time), product recommendations (suggesting irrelevant products annoys users and reduces trust), and fraud alerts sent to customers (too many false alarms train customers to ignore real fraud warnings). In these cases, you\'re optimizing to be "right when you speak up," accepting that you might miss some cases.\n\nOptimize for recall when false negatives are very costly and false positives are manageable. Medical screening tests are the paradigm example: missing a cancer diagnosis (false negative) could be fatal, while a false positive just means an unnecessary follow-up test. You want high recall to catch all potential cases, accepting some false alarms that get filtered out by confirmatory testing. Airport security screening similarly prioritizes recall—better to flag innocent passengers for additional screening than to miss a threat. Other recall-focused applications include: fraud detection (missing fraud causes direct financial loss), safety monitoring systems (missing a critical equipment failure could cause accidents), missing children alerts (false alarms are acceptable when a child\'s safety is at risk), and initial resume screening (false positives get filtered in interviews, but missing a great candidate is permanent loss). The general principle: optimize for recall when the cost of missing a positive case is much higher than the cost of false alarms.'
      },
      {
        question: 'What is the difference between ROC-AUC and PR-AUC? When is each more appropriate?',
        answer: 'ROC (Receiver Operating Characteristic) curve plots True Positive Rate (TPR = recall) vs False Positive Rate (FPR) at various classification thresholds. AUC-ROC is the area under this curve, representing the probability that the model ranks a random positive sample higher than a random negative sample. PR (Precision-Recall) curve plots precision vs recall at various thresholds. AUC-PR is the area under this curve. Both measure classifier quality across all possible thresholds, but they emphasize different aspects of performance.\n\nThe key difference emerges with class imbalance. ROC-AUC can be misleadingly optimistic for imbalanced datasets because FPR uses true negatives in the denominator, and with many negatives, FPR stays low even with substantial false positives. For example, with 99% negative class, 100 false positives and 9900 true negatives gives FPR = 100/10000 = 1%, appearing excellent. PR-AUC is more sensitive to imbalance because precision has false positives in the denominator without the buffering effect of many true negatives. The same 100 false positives with 50 true positives gives precision = 50/150 = 33%, clearly showing the problem.\n\nUse ROC-AUC when classes are roughly balanced and you care about both positive and negative classes equally. It\'s also standard in domains like medical diagnostics where you need to balance sensitivity (catching disease) and specificity (not alarming healthy patients). Use PR-AUC for imbalanced datasets (especially minority class <10%) or when you primarily care about performance on the positive class. Fraud detection, rare disease screening, information retrieval, and anomaly detection should use PR-AUC. A perfect classifier has AUC-ROC = 1.0 and AUC-PR = 1.0, but random guessing gives AUC-ROC = 0.5 regardless of imbalance while AUC-PR equals the positive class frequency (e.g., 0.01 for 1% positive class), making PR-AUC a higher bar. In practice, report both when possible to give a complete picture of performance.'
      },
      {
        question: 'How do you interpret an R² score of -0.5 in regression?',
        answer: 'An R² of -0.5 means your model performs worse than simply predicting the mean of the target variable for every sample—it\'s making predictions that are systematically worse than the simplest baseline. R² is defined as 1 - (SS_res / SS_tot) where SS_res is the sum of squared residuals (prediction errors) and SS_tot is total sum of squares (variance around the mean). When SS_res > SS_tot, R² becomes negative. With R² = -0.5, your residual error is 1.5× larger than the variance around the mean, indicating severe model failure.\n\nThis typically indicates fundamental problems. The model might be completely mis-specified—for example, fitting a linear model to exponential growth, or using features totally unrelated to the target. It could result from severe overfitting on training data that doesn\'t generalize at all to test data, though overfitting usually shows as low R² rather than negative. Negative R² can also occur from data leakage in reverse: testing on a different distribution than training, where the training distribution\'s mean is actually a worse predictor than the model would be on its own distribution. Preprocessing errors like scaling the test set incorrectly or features missing in test data can also cause this.\n\nPractically, negative R² demands immediate investigation. First, check for data issues: ensure train and test come from the same distribution, verify no data leakage or preprocessing errors, confirm target variable is measured consistently. Second, examine model assumptions: plot predictions vs actuals to see if there\'s any relationship, check residual plots for patterns indicating model mis-specification. Third, try the simplest possible baseline (mean prediction) and verify it actually outperforms your model. If baseline is indeed better, you likely need a completely different modeling approach, more relevant features, or to reconsider whether the problem is predictable with available data. A negative R² is a strong signal that something is seriously wrong—don\'t try to tweak hyperparameters, rebuild from scratch.'
      },
      {
        question: 'Why is RMSE more sensitive to outliers than MAE?',
        answer: 'RMSE (Root Mean Squared Error) is more sensitive to outliers than MAE (Mean Absolute Error) because it squares the errors before averaging, which disproportionately penalizes large errors. Consider two predictions with errors [1, 1, 10]: MAE = (1+1+10)/3 = 4.0, while RMSE = √[(1²+1²+10²)/3] = √(102/3) = 5.83. The single large error (10) has modest impact on MAE but substantially inflates RMSE. With errors [1, 1, 1], MAE = 1.0 and RMSE = 1.0, but with [0, 0, 3] (same total error), MAE = 1.0 while RMSE = 1.73, showing how RMSE penalizes concentrated errors.\n\nMathematically, squaring errors means a prediction that\'s off by 10 contributes 100 to the squared error sum, while ten predictions each off by 1 only contribute 10 total. The ratio scales quadratically: doubling the error quadruples its contribution to RMSE but only doubles its contribution to MAE. This makes RMSE more sensitive to the worst predictions—a single very bad prediction can dominate RMSE while having limited impact on MAE. Taking the square root at the end brings the units back to match the target variable, but doesn\'t undo the disproportionate weighting of large errors.\n\nChoose RMSE when large errors are particularly undesirable and you want to heavily penalize them. For example, in real estate price prediction, being off by $100k on a luxury home is much worse than being off by $10k on ten houses, and RMSE captures this. However, if outliers in your data are due to measurement errors or rare anomalies that you don\'t want to dominate your metric, MAE is better as it treats all errors linearly. MAE is also more robust and interpretable—it directly represents average absolute error in the target\'s units. In domains with naturally occurring outliers you must handle (extreme weather, epidemic forecasting), RMSE\'s outlier sensitivity might lead to models overfitting to rare extreme cases at the expense of typical performance. The choice depends on your loss function\'s true shape: quadratic losses naturally correspond to RMSE, linear losses to MAE.'
      },
      {
        question: 'You are building a cancer detection model. Which metric(s) would you prioritize and why?',
        answer: 'For cancer detection, prioritize recall (sensitivity) as the primary metric, while monitoring precision to avoid excessive false alarms. Missing a cancer case (false negative) has catastrophic consequences—delayed treatment can mean the difference between curable and terminal disease, or even life and death. A false positive (flagging cancer when there is none) is much less costly—it leads to additional testing (biopsies, imaging) which causes stress and expense but no permanent harm. The cost asymmetry is extreme: false negatives are potentially fatal, false positives are inconvenient and expensive but manageable.\n\nAim for very high recall (>95%, ideally >99%) to catch nearly all cancer cases, accepting moderate precision (perhaps 20-50% depending on cancer type and screening context). This means your model acts as a sensitive screening tool: it flags many patients for follow-up, knowing that confirmatory tests will filter out most false positives. For example, if 1% of screened patients have cancer, a model with 99% recall and 20% precision would correctly identify 99 of 100 cancer patients while also flagging 396 false positives (495 total flagged patients). Those 495 people get diagnostic workup, catch 99 real cancers, and clear 396 healthy people—acceptable trade-off.\n\nSecondary metrics matter too. Use F2-score (weights recall 2× higher than precision) for a single balanced metric, or F0.5-score if false positives are moderately costly. Monitor specificity (true negative rate) to ensure you\'re not flagging everyone—a model that flags 100% of patients has perfect recall but is useless. Track precision at your operating recall level to understand false alarm burden on the healthcare system. For different cancer types, adjust thresholds: aggressive cancers demand higher recall, slow-growing cancers might accept slightly lower recall with higher precision. Finally, consider calibration—if the model outputs cancer probability, ensure probabilities are reliable so doctors can make informed decisions about aggressive vs conservative follow-up based on risk level. The overarching principle: optimize to catch cancers even at the cost of false alarms, because the downside of missing cancer far outweighs the downside of unnecessary testing.'
      }
    ],
    quizQuestions: [
      {
        id: 'metrics-q1',
        question: 'You are building a spam email classifier. Your model achieves 99% accuracy, but users complain that spam emails still reach their inbox. What is the most likely issue?',
        options: [
          'The model has high precision but low recall for spam',
          'The model has high recall but low precision for spam',
          'The accuracy metric is appropriate for this task',
          'The model needs more training data'
        ],
        correctAnswer: 0,
        explanation: 'High accuracy with user complaints suggests the model rarely labels emails as spam (high precision = few false positives) but misses many spam emails (low recall = many false negatives). The dataset is likely imbalanced toward non-spam, making accuracy misleading.'
      },
      {
        id: 'metrics-q2',
        question: 'You are predicting house prices. Your model achieves RMSE=50,000 and MAE=20,000. What does this tell you?',
        options: [
          'The model is biased and consistently overestimates prices',
          'There are likely outliers or large errors in predictions',
          'The model is perfect with no errors',
          'MAE should always be larger than RMSE'
        ],
        correctAnswer: 1,
        explanation: 'RMSE (50k) being much larger than MAE (20k) indicates some predictions have large errors. RMSE amplifies large errors due to squaring, while MAE treats all errors equally. This suggests outliers or occasional large mispredictions.'
      },
      {
        id: 'metrics-q3',
        question: 'For a fraud detection system where fraudulent transactions are 0.1% of all transactions, which metric is MOST appropriate?',
        options: [
          'Accuracy',
          'ROC-AUC',
          'Precision-Recall AUC',
          'Mean Squared Error'
        ],
        correctAnswer: 2,
        explanation: 'PR-AUC is best for highly imbalanced datasets. Accuracy would be 99.9% by predicting everything as non-fraud. ROC-AUC can be overly optimistic due to the large number of true negatives. PR-AUC focuses on positive class performance.'
      }
    ]
  },
  'hyperparameter-tuning': {
    id: 'hyperparameter-tuning',
    title: 'Hyperparameter Tuning',
    category: 'foundations',
    description: 'Techniques and strategies for optimizing model hyperparameters to improve performance.',
    content: `
      <h2>Overview</h2>
      <p>Hyperparameter tuning is the process of finding the optimal configuration of hyperparameters—settings that control the learning process but are not learned from data—to maximize model performance.</p>

      <h3>Hyperparameters vs Parameters</h3>
      <p><strong>Parameters</strong> are learned from training data (e.g., weights in neural networks, coefficients in linear regression).</p>
      <p><strong>Hyperparameters</strong> are set before training and control the learning process (e.g., learning rate, number of trees, regularization strength).</p>

      <h3>Common Hyperparameters</h3>
      <ul>
        <li><strong>Learning rate:</strong> Step size for gradient descent</li>
        <li><strong>Batch size:</strong> Number of samples per gradient update</li>
        <li><strong>Number of epochs:</strong> How many times to iterate through training data</li>
        <li><strong>Regularization strength (α, λ):</strong> Penalty for model complexity</li>
        <li><strong>Network architecture:</strong> Number of layers, neurons per layer</li>
        <li><strong>Tree depth:</strong> Maximum depth for decision trees</li>
        <li><strong>Number of estimators:</strong> Number of trees in ensemble methods</li>
      </ul>

      <h3>Tuning Strategies</h3>

      <h4>1. Manual Search</h4>
      <p>Trying different values based on intuition and domain knowledge. Simple but inefficient and requires expertise.</p>

      <h4>2. Grid Search</h4>
      <p>Exhaustively searches through a manually specified subset of hyperparameter space. Tests all combinations of specified values.</p>
      <ul>
        <li><strong>Pros:</strong> Comprehensive, reproducible, parallelizable</li>
        <li><strong>Cons:</strong> Computationally expensive, curse of dimensionality, may miss optimal values between grid points</li>
      </ul>

      <h4>3. Random Search</h4>
      <p>Samples random combinations of hyperparameters from specified distributions. Often more efficient than grid search.</p>
      <ul>
        <li><strong>Pros:</strong> Better coverage of hyperparameter space, more efficient for high-dimensional spaces</li>
        <li><strong>Cons:</strong> No guarantee of finding optimal values, may need many iterations</li>
      </ul>

      <h4>4. Bayesian Optimization</h4>
      <p>Uses probabilistic model to predict promising hyperparameter regions, focusing search on areas likely to improve performance.</p>
      <ul>
        <li><strong>Pros:</strong> Sample efficient, can find good solutions with fewer evaluations</li>
        <li><strong>Cons:</strong> More complex to implement, overhead for building surrogate model</li>
      </ul>

      <h4>5. Automated Methods</h4>
      <p>Advanced techniques like Hyperband, BOHB (Bayesian Optimization and HyperBand), and population-based training that combine multiple strategies.</p>

      <h3>Best Practices</h3>
      <ul>
        <li><strong>Use validation set:</strong> Tune on validation data, never on test set</li>
        <li><strong>Use cross-validation:</strong> Get more reliable estimates of performance</li>
        <li><strong>Start coarse, then refine:</strong> Begin with wide ranges, narrow down to promising regions</li>
        <li><strong>Log scale for learning rates:</strong> Try 0.001, 0.01, 0.1 rather than 0.01, 0.02, 0.03</li>
        <li><strong>Prioritize important hyperparameters:</strong> Focus on those with largest impact (learning rate, regularization)</li>
        <li><strong>Use early stopping:</strong> Save computation by stopping poor configurations early</li>
        <li><strong>Track experiments:</strong> Record all configurations and results for analysis</li>
      </ul>

      <h3>Common Pitfalls</h3>
      <ul>
        <li><strong>Overfitting to validation set:</strong> Too much tuning can overfit; use separate test set for final evaluation</li>
        <li><strong>Ignoring computational cost:</strong> Balance performance gains against training time</li>
        <li><strong>Not considering interaction effects:</strong> Hyperparameters often interact; tune related ones together</li>
        <li><strong>Using test set for tuning:</strong> This leaks information and inflates performance estimates</li>
      </ul>

      <h3>Tools and Libraries</h3>
      <ul>
        <li><strong>Scikit-learn:</strong> GridSearchCV, RandomizedSearchCV</li>
        <li><strong>Optuna:</strong> Bayesian optimization framework</li>
        <li><strong>Ray Tune:</strong> Scalable hyperparameter tuning library</li>
        <li><strong>Keras Tuner:</strong> Hyperparameter tuning for neural networks</li>
        <li><strong>Hyperopt:</strong> Distributed hyperparameter optimization</li>
        <li><strong>Weights & Biases Sweeps:</strong> Experiment tracking with hyperparameter optimization</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'python',
        explanation: 'Grid Search with Cross-Validation',
        code: `from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize model and grid search
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,  # Use all CPU cores
    verbose=2
)

# Perform grid search
grid_search.fit(X_train, y_train)

# Best hyperparameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
print(f"Test score: {grid_search.score(X_test, y_test):.3f}")

# All results
import pandas as pd
results = pd.DataFrame(grid_search.cv_results_)
print(results[['params', 'mean_test_score', 'std_test_score']].sort_values('mean_test_score', ascending=False).head())`
      },
      {
        language: 'python',
        explanation: 'Random Search with Cross-Validation',
        code: `from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import numpy as np

# Define hyperparameter distributions
param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)  # Fraction of features
}

# Initialize random search
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,  # Number of random combinations to try
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=2
)

# Perform random search
random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.3f}")
print(f"Test score: {random_search.score(X_test, y_test):.3f}")`
      },
      {
        language: 'python',
        explanation: 'Bayesian Optimization with Optuna',
        code: `import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0)
    }
    
    # Create model and evaluate
    model = RandomForestClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    
    return score

# Create study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

# Best results
print(f"Best parameters: {study.best_params}")
print(f"Best score: {study.best_value:.3f}")

# Visualize optimization history
import matplotlib.pyplot as plt
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.show()

# Feature importances
optuna.visualization.matplotlib.plot_param_importances(study)
plt.show()`
      },
      {
        language: 'python',
        explanation: 'Neural Network Hyperparameter Tuning',
        code: `import tensorflow as tf
from tensorflow import keras
from keras_tuner import RandomSearch

def build_model(hp):
    model = keras.Sequential()
    
    # Tune number of layers and units
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(keras.layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
            activation='relu'
        ))
        
        # Tune dropout
        if hp.Boolean('dropout'):
            model.add(keras.layers.Dropout(rate=hp.Float('dropout_rate', 0.1, 0.5, step=0.1)))
    
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    # Tune learning rate
    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=50,
    executions_per_trial=2,
    directory='tuning_results',
    project_name='nn_tuning'
)

# Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255
x_test = x_test.reshape(-1, 784).astype('float32') / 255

# Search for best hyperparameters
tuner.search(
    x_train, y_train,
    epochs=10,
    validation_split=0.2,
    callbacks=[keras.callbacks.EarlyStopping(patience=3)]
)

# Get best model
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Best hyperparameters: {best_hyperparameters.values}")
test_loss, test_accuracy = best_model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.3f}")`
      }
    ],
    interviewQuestions: [
      {
        question: 'Why might random search outperform grid search, even with fewer iterations?',
        answer: 'Random search often outperforms grid search because it explores the hyperparameter space more effectively, particularly when some hyperparameters are more important than others. Consider tuning two hyperparameters: one critical (learning rate) and one less important (batch size). Grid search with 9 values per parameter tests 81 combinations but only 9 distinct values for each hyperparameter. Random search with 81 trials samples different values each time, effectively exploring more diverse values for the important hyperparameter.\n\nMathematically, if one hyperparameter has much larger impact on performance, random search is more likely to find good values for it. Grid search might waste computation testing poor values of the important hyperparameter paired with different values of the less important one. Random search also doesn\'t suffer from the curse of dimensionality as severely—with 5 hyperparameters and 5 values each, grid search requires 3,125 evaluations, while random search can sample any number of points, focusing budget efficiently.\n\nPractically, random search provides better coverage when you\'re uncertain about hyperparameter ranges. If optimal learning rate is 0.007 but your grid tests [0.001, 0.01, 0.1], you\'ll miss it. Random search sampling from log-uniform[0.0001, 1] is more likely to try values near 0.007. Additionally, random search is embarrassingly parallel and can be stopped anytime, while grid search requires completing all combinations to avoid bias. Research by Bergstra & Bengio (2012) showed random search can find comparable or better solutions than grid search with 2-3× fewer evaluations in practice.'
      },
      {
        question: 'How would you avoid overfitting to the validation set during hyperparameter tuning?',
        answer: 'Overfitting to the validation set occurs when you tune hyperparameters extensively, essentially using validation performance to "train" your hyperparameter choices. The solution is a three-way split: training set for learning parameters, validation set for tuning hyperparameters, and a held-out test set for final evaluation that\'s never used during development.\n\nBest practices: Use cross-validation during hyperparameter search to get more robust estimates—5-fold or 10-fold CV on your training data gives better signal than a single validation split, reducing the risk of tuning to noise. Limit the number of hyperparameter configurations you try relative to validation set size. With 100 validation samples, testing 1000 configurations is likely to overfit; with 10,000 samples, testing 1000 is reasonable. Keep the test set completely separate until the very end—one evaluation only, after all development decisions are final.\n\nFor nested cross-validation, the outer loop evaluates model performance while the inner loop tunes hyperparameters. This gives unbiased performance estimates but is computationally expensive: 5x5 nested CV means 25 model trainings per hyperparameter configuration. Use early stopping during tuning—if 50 configurations haven\'t improved over the best in 10 trials, stop searching. This prevents endless tuning that fits validation noise.\n\nMonitor the gap between validation and test performance. If validation accuracy is 95% but test is 85%, you\'ve overfit to validation. In this case, use simpler models, reduce hyperparameter search space, or get more validation data. For competitions or critical applications, use time-based splits if data has temporal structure, ensuring validation and test come from later time periods than training. This prevents leakage and tests generalization to future data, which is ultimately what matters in production.'
      },
      {
        question: 'You have limited compute budget. How would you prioritize which hyperparameters to tune?',
        answer: 'With limited budget, focus on hyperparameters with the largest impact on performance, typically learning rate and regularization strength. Start with a coarse random search over these critical hyperparameters using wide ranges on log scales (e.g., learning rate from 1e-5 to 1, L2 penalty from 1e-5 to 10). These often account for 80% of the performance variance.\n\nFor tree-based models, prioritize: (1) number of trees/estimators—more is usually better until diminishing returns, (2) max depth—controls overfitting, (3) learning rate for boosting—critical for gradient boosting. For neural networks: (1) learning rate—single most important, (2) network architecture (depth and width), (3) regularization (dropout, weight decay), (4) batch size and optimizer type. For SVMs: (1) regularization parameter C, (2) kernel type, (3) kernel-specific parameters like gamma for RBF.\n\nUse a sequential strategy: first tune the most important hyperparameters with other values at reasonable defaults. Once you find good values, fix those and tune the next tier. For example, find optimal learning rate and regularization, then tune batch size and momentum with the optimal learning rate fixed. This multi-stage approach is more efficient than joint optimization when budget is tight.\n\nApply early stopping aggressively—allocate initial budget to quick evaluations (fewer epochs, smaller data samples) to eliminate poor configurations, then allocate remaining budget to train promising configurations fully. Use learning curves: if a configuration performs poorly after 10% of training, it\'s unlikely to become best by the end. Modern methods like Hyperband and BOHB implement this principle systematically, achieving good results with 10-100× less compute than exhaustive search. Finally, leverage transfer learning—if tuning similar models, start with hyperparameters that worked well on related tasks rather than searching from scratch.'
      }
    ]
  }
};
