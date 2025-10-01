import { QuizQuestion } from '../../types';

// Linear Regression - 20 questions
export const linearRegressionQuestions: QuizQuestion[] = [
  {
    id: 'lr1',
    question: 'What is the goal of linear regression?',
    options: ['Classify data points', 'Find the best-fitting line', 'Cluster similar data', 'Reduce dimensionality'],
    correctAnswer: 1,
    explanation: 'Linear regression aims to find the best-fitting line (or hyperplane) that minimizes the error between predicted and actual values.'
  },
  {
    id: 'lr2',
    question: 'What cost function does linear regression typically minimize?',
    options: ['Cross-entropy', 'Mean Squared Error (MSE)', 'Hinge loss', 'KL divergence'],
    correctAnswer: 1,
    explanation: 'Linear regression typically uses Mean Squared Error (MSE) or its variant, which measures the average squared difference between predictions and actual values.'
  },
  {
    id: 'lr3',
    question: 'What does the slope (β₁) represent in simple linear regression?',
    options: ['The y-intercept', 'Rate of change of y with respect to x', 'The error term', 'The correlation coefficient'],
    correctAnswer: 1,
    explanation: 'The slope β₁ represents how much y changes for each unit change in x.'
  },
  {
    id: 'lr4',
    question: 'What assumption does linear regression make about the relationship between variables?',
    options: ['Exponential', 'Linear', 'Logarithmic', 'Polynomial'],
    correctAnswer: 1,
    explanation: 'Linear regression assumes a linear relationship between the independent and dependent variables.'
  },
  {
    id: 'lr5',
    question: 'What is multicollinearity in linear regression?',
    options: ['Multiple target variables', 'High correlation between predictors', 'Non-linear relationships', 'Missing data'],
    correctAnswer: 1,
    explanation: 'Multicollinearity occurs when independent variables are highly correlated with each other, making it difficult to determine individual effects.'
  },
  {
    id: 'lr6',
    question: 'Which method is commonly used to find optimal parameters in linear regression?',
    options: ['K-means clustering', 'Ordinary Least Squares (OLS)', 'Decision trees', 'Principal Component Analysis'],
    correctAnswer: 1,
    explanation: 'Ordinary Least Squares (OLS) is the standard method for finding parameters that minimize the sum of squared residuals.'
  },
  {
    id: 'lr7',
    question: 'What does R² (R-squared) measure in linear regression?',
    options: ['The slope', 'Proportion of variance explained', 'Mean error', 'Number of parameters'],
    correctAnswer: 1,
    explanation: 'R² measures the proportion of variance in the dependent variable that is explained by the independent variables.'
  },
  {
    id: 'lr8',
    question: 'What is a residual in linear regression?',
    options: ['The predicted value', 'The difference between actual and predicted values', 'The slope', 'The intercept'],
    correctAnswer: 1,
    explanation: 'A residual is the difference between an observed value and the value predicted by the model.'
  },
  {
    id: 'lr9',
    question: 'Which assumption requires residuals to have constant variance?',
    options: ['Linearity', 'Independence', 'Homoscedasticity', 'Normality'],
    correctAnswer: 2,
    explanation: 'Homoscedasticity means residuals have constant variance across all levels of the independent variables.'
  },
  {
    id: 'lr10',
    question: 'What happens if linear regression assumptions are violated?',
    options: ['Model becomes more accurate', 'Predictions and inference may be unreliable', 'Nothing changes', 'Model trains faster'],
    correctAnswer: 1,
    explanation: 'Violating assumptions can lead to biased estimates, incorrect inference, and unreliable predictions.'
  },
  {
    id: 'lr11',
    question: 'What is polynomial regression?',
    options: ['Multiple linear regression', 'Extension of linear regression with polynomial terms', 'Logistic regression', 'Ridge regression'],
    correctAnswer: 1,
    explanation: 'Polynomial regression extends linear regression by including polynomial terms (x², x³, etc.) to model non-linear relationships.'
  },
  {
    id: 'lr12',
    question: 'When should you use linear regression instead of other models?',
    options: ['For classification problems', 'When relationship is linear and interpretability is important', 'For clustering', 'Only for time series'],
    correctAnswer: 1,
    explanation: 'Linear regression is ideal when the relationship is approximately linear and you need an interpretable model.'
  },
  {
    id: 'lr13',
    question: 'What is the "normal equation" in linear regression?',
    options: ['Iterative optimization', 'Closed-form solution: β = (XᵀX)⁻¹Xᵀy', 'Gradient descent', 'Random initialization'],
    correctAnswer: 1,
    explanation: 'The normal equation provides a direct analytical solution for the optimal parameters without iteration.'
  },
  {
    id: 'lr14',
    question: 'Why might gradient descent be preferred over the normal equation?',
    options: ['More accurate', 'Better for large datasets (avoids matrix inversion)', 'Simpler to implement', 'Always gives exact solution'],
    correctAnswer: 1,
    explanation: 'For large datasets, computing (XᵀX)⁻¹ becomes computationally expensive, making gradient descent more efficient.'
  },
  {
    id: 'lr15',
    question: 'What does standardization of features help with in linear regression?',
    options: ['Increases accuracy', 'Makes coefficients comparable and helps optimization', 'Reduces overfitting', 'Creates new features'],
    correctAnswer: 1,
    explanation: 'Standardization puts features on the same scale, making coefficients comparable and helping gradient descent converge faster.'
  },
  {
    id: 'lr16',
    question: 'What is heteroscedasticity?',
    options: ['Constant variance of residuals', 'Non-constant variance of residuals', 'Linear relationship', 'Normal distribution'],
    correctAnswer: 1,
    explanation: 'Heteroscedasticity occurs when the variance of residuals changes across levels of independent variables, violating linear regression assumptions.'
  },
  {
    id: 'lr17',
    question: 'Can linear regression handle categorical variables?',
    options: ['No, never', 'Yes, with dummy/one-hot encoding', 'Only binary categories', 'Only with scaling'],
    correctAnswer: 1,
    explanation: 'Categorical variables can be included using dummy/one-hot encoding to convert them into numerical format.'
  },
  {
    id: 'lr18',
    question: 'What is the difference between simple and multiple linear regression?',
    options: ['Simple uses one predictor, multiple uses several', 'Simple is for classification', 'Multiple is non-linear', 'No difference'],
    correctAnswer: 0,
    explanation: 'Simple linear regression has one independent variable, while multiple linear regression has two or more.'
  },
  {
    id: 'lr19',
    question: 'What does a p-value tell you about a coefficient in linear regression?',
    options: ['The coefficient value', 'Statistical significance of the predictor', 'R² value', 'Prediction accuracy'],
    correctAnswer: 1,
    explanation: 'The p-value indicates whether a predictor has a statistically significant relationship with the outcome.'
  },
  {
    id: 'lr20',
    question: 'What is an outlier\'s effect on linear regression?',
    options: ['No effect', 'Can significantly influence the fitted line', 'Improves accuracy', 'Only affects intercept'],
    correctAnswer: 1,
    explanation: 'Outliers can have a large impact on the regression line since linear regression minimizes squared errors, giving more weight to large errors.'
  }
];

// Logistic Regression - 20 questions
export const logisticRegressionQuestions: QuizQuestion[] = [
  {
    id: 'log1',
    question: 'What type of problem is logistic regression used for?',
    options: ['Regression', 'Classification', 'Clustering', 'Dimensionality reduction'],
    correctAnswer: 1,
    explanation: 'Despite its name, logistic regression is used for binary and multinomial classification problems.'
  },
  {
    id: 'log2',
    question: 'What function does logistic regression use to produce probabilities?',
    options: ['ReLU', 'Sigmoid (logistic function)', 'Softmax', 'Tanh'],
    correctAnswer: 1,
    explanation: 'Logistic regression uses the sigmoid function to map predictions to probabilities between 0 and 1.'
  },
  {
    id: 'log3',
    question: 'What is the range of the sigmoid function?',
    options: ['[-1, 1]', '[0, 1]', '[-∞, ∞]', '[0, ∞]'],
    correctAnswer: 1,
    explanation: 'The sigmoid function outputs values between 0 and 1, making it suitable for probability estimation.'
  },
  {
    id: 'log4',
    question: 'What loss function does logistic regression typically use?',
    options: ['Mean Squared Error', 'Binary Cross-Entropy (Log Loss)', 'Hinge Loss', 'Huber Loss'],
    correctAnswer: 1,
    explanation: 'Logistic regression uses binary cross-entropy (log loss) to measure the difference between predicted probabilities and true labels.'
  },
  {
    id: 'log5',
    question: 'What is the decision boundary in logistic regression?',
    options: ['A curve', 'The threshold (typically 0.5) for classification', 'The sigmoid function', 'The loss function'],
    correctAnswer: 1,
    explanation: 'The decision boundary is the threshold probability (commonly 0.5) above which we predict one class and below which we predict the other.'
  },
  {
    id: 'log6',
    question: 'How do you extend logistic regression to multi-class problems?',
    options: ['Use multiple binary classifiers', 'Use softmax regression (multinomial logistic regression)', 'Cannot be extended', 'Use clustering first'],
    correctAnswer: 1,
    explanation: 'Softmax regression (multinomial logistic regression) extends logistic regression to handle multiple classes.'
  },
  {
    id: 'log7',
    question: 'What does regularization do in logistic regression?',
    options: ['Increases model complexity', 'Prevents overfitting by penalizing large weights', 'Speeds up training', 'Increases accuracy always'],
    correctAnswer: 1,
    explanation: 'Regularization (L1 or L2) adds penalties to the loss function to prevent overfitting by constraining weight magnitudes.'
  },
  {
    id: 'log8',
    question: 'What is the "odds ratio" in logistic regression?',
    options: ['Probability of success', 'P(success) / P(failure)', 'Number of successes', 'The intercept'],
    correctAnswer: 1,
    explanation: 'The odds ratio is the ratio of the probability of success to the probability of failure: P(Y=1) / P(Y=0).'
  },
  {
    id: 'log9',
    question: 'Why can\'t we use MSE as the loss function for logistic regression?',
    options: ['Too slow to compute', 'Creates non-convex optimization problem', 'Only works for regression', 'Gives wrong answers'],
    correctAnswer: 1,
    explanation: 'MSE with sigmoid creates a non-convex optimization landscape with local minima, making optimization difficult.'
  },
  {
    id: 'log10',
    question: 'What does the logit (log-odds) represent?',
    options: ['The probability', 'The natural log of the odds ratio', 'The loss', 'The accuracy'],
    correctAnswer: 1,
    explanation: 'The logit is the natural logarithm of the odds ratio: log(P(Y=1) / P(Y=0)) = β₀ + β₁x.'
  },
  {
    id: 'log11',
    question: 'How do you interpret a coefficient in logistic regression?',
    options: ['Direct change in probability', 'Change in log-odds for unit change in predictor', 'Change in accuracy', 'Change in loss'],
    correctAnswer: 1,
    explanation: 'A coefficient represents the change in log-odds of the outcome for a one-unit increase in the predictor.'
  },
  {
    id: 'log12',
    question: 'What is one-vs-rest (OvR) in multi-class logistic regression?',
    options: ['Using one feature', 'Training one classifier per class vs. all others', 'Using one data point', 'One optimization step'],
    correctAnswer: 1,
    explanation: 'One-vs-Rest trains K binary classifiers, each distinguishing one class from all other classes.'
  },
  {
    id: 'log13',
    question: 'What is the maximum likelihood estimation in logistic regression?',
    options: ['Finding weights that minimize loss', 'Finding weights that maximize probability of observed data', 'Random initialization', 'Cross-validation'],
    correctAnswer: 1,
    explanation: 'Maximum likelihood estimation finds parameters that maximize the probability of observing the training data.'
  },
  {
    id: 'log14',
    question: 'When is logistic regression preferred over other classifiers?',
    options: ['Always', 'When you need probability estimates and interpretability', 'Only for small datasets', 'Never'],
    correctAnswer: 1,
    explanation: 'Logistic regression is preferred when you need probability estimates and an interpretable, fast model.'
  },
  {
    id: 'log15',
    question: 'What is a limitation of logistic regression?',
    options: ['Too complex', 'Assumes linear decision boundary', 'Cannot handle categorical variables', 'Too slow'],
    correctAnswer: 1,
    explanation: 'Logistic regression assumes a linear relationship between features and log-odds, limiting its ability to capture non-linear patterns.'
  },
  {
    id: 'log16',
    question: 'How does logistic regression handle imbalanced datasets?',
    options: ['Automatically', 'May need class weights or resampling', 'Cannot handle imbalance', 'Only with preprocessing'],
    correctAnswer: 1,
    explanation: 'Imbalanced datasets may require class weights, resampling, or adjusting the decision threshold for better performance.'
  },
  {
    id: 'log17',
    question: 'What does it mean if a logistic regression coefficient is 0?',
    options: ['Feature is important', 'Feature has no effect on log-odds', 'Model is broken', 'Feature is categorical'],
    correctAnswer: 1,
    explanation: 'A coefficient of 0 means the feature has no linear effect on the log-odds of the outcome.'
  },
  {
    id: 'log18',
    question: 'Can logistic regression be used with non-linear features?',
    options: ['No, never', 'Yes, by adding polynomial or interaction terms', 'Only with neural networks', 'Only with PCA'],
    correctAnswer: 1,
    explanation: 'Logistic regression can model non-linear relationships by including polynomial features, interactions, or transformations.'
  },
  {
    id: 'log19',
    question: 'What is the gradient descent update rule for logistic regression?',
    options: ['Add random noise', 'θ := θ - α × gradient of loss', 'θ := θ + learning rate', 'θ := 0'],
    correctAnswer: 1,
    explanation: 'Parameters are updated by moving in the direction opposite to the gradient, scaled by the learning rate.'
  },
  {
    id: 'log20',
    question: 'What evaluation metric is most appropriate for imbalanced logistic regression?',
    options: ['Accuracy', 'AUC-ROC or F1 score', 'MSE', 'R²'],
    correctAnswer: 1,
    explanation: 'AUC-ROC and F1 score are better metrics for imbalanced datasets as they account for both precision and recall.'
  }
];
