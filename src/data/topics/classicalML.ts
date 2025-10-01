import { Topic } from '../../types';

export const classicalMLTopics: Record<string, Topic> = {
  'linear-regression': {
    id: 'linear-regression',
    title: 'Linear Regression',
    category: 'classical-ml',
    description: 'Understanding linear regression, the foundation of many machine learning algorithms.',
    content: `
      <h2>Linear Regression</h2>
      <p>Linear regression is one of the simplest and most widely used machine learning algorithms. It models the relationship between a dependent variable and independent variables using a linear equation.</p>

      <h3>Mathematical Foundation</h3>
      <p>For simple linear regression with one feature:</p>
      <p><strong>y = β₀ + β₁x + ε</strong></p>
      <p>Where:</p>
      <ul>
        <li>y is the dependent variable (target)</li>
        <li>x is the independent variable (feature)</li>
        <li>β₀ is the y-intercept</li>
        <li>β₁ is the slope</li>
        <li>ε is the error term</li>
      </ul>

      <h3>Multiple Linear Regression</h3>
      <p>For multiple features:</p>
      <p><strong>y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε</strong></p>

      <h3>Cost Function</h3>
      <p>Linear regression uses Mean Squared Error (MSE) as the cost function:</p>
      <p><strong>MSE = (1/n) Σ(yᵢ - ŷᵢ)²</strong></p>

      <h3>Assumptions</h3>
      <ul>
        <li>Linear relationship between features and target</li>
        <li>Independence of residuals</li>
        <li>Homoscedasticity (constant variance of residuals)</li>
        <li>Normal distribution of residuals</li>
      </ul>

      <h3>Advantages</h3>
      <ul>
        <li>Simple and interpretable</li>
        <li>Fast training and prediction</li>
        <li>No hyperparameters to tune</li>
        <li>Good baseline model</li>
      </ul>

      <h3>Disadvantages</h3>
      <ul>
        <li>Assumes linear relationships</li>
        <li>Sensitive to outliers</li>
        <li>Can suffer from multicollinearity</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 1)
y = 2 * X.flatten() + 1 + np.random.randn(100) * 0.1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")`,
        explanation: 'This example demonstrates how to implement linear regression using scikit-learn, including model training, prediction, and evaluation.'
      },
      {
        language: 'Python',
        code: `# Manual implementation of linear regression
import numpy as np

class LinearRegressionManual:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Add bias term to X
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])

        # Normal equation: θ = (X^T X)^(-1) X^T y
        theta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y

        self.bias = theta[0]
        self.weights = theta[1:]

    def predict(self, X):
        return X @ self.weights + self.bias

# Example usage
model = LinearRegressionManual()
model.fit(X_train, y_train)
predictions = model.predict(X_test)`,
        explanation: 'This shows a manual implementation of linear regression using the normal equation method.'
      }
    ],
    interviewQuestions: [
      {
        question: 'How does linear regression work?',
        answer: 'Linear regression models the relationship between input features and a continuous target variable by fitting a linear equation. For simple linear regression with one feature, it finds the best-fit line y = mx + b that minimizes prediction errors. For multiple linear regression with many features, it finds the hyperplane y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ that best predicts the target. The "best" fit is typically defined as minimizing the sum of squared residuals (differences between predicted and actual values), known as Ordinary Least Squares (OLS).\n\nThe training process finds optimal coefficients (weights) that minimize the loss function, usually Mean Squared Error (MSE). This can be done analytically using the normal equation for small datasets, or iteratively using gradient descent for large datasets. During training, the algorithm adjusts weights to reduce the average squared error across all training samples. The learned coefficients represent the importance and direction of each feature\'s relationship with the target—positive coefficients indicate the target increases as the feature increases, negative coefficients indicate inverse relationships.\n\nOnce trained, making predictions is straightforward: multiply each input feature by its corresponding coefficient, sum them up, and add the bias term. The model outputs a continuous value, making it suitable for regression tasks like predicting house prices, temperatures, or sales figures. Linear regression assumes a linear relationship between features and target, which is a strong assumption but makes the model highly interpretable—you can directly see how each feature contributes to predictions through its coefficient.'
      },
      {
        question: 'What are the assumptions of linear regression?',
        answer: 'Linear regression makes several key assumptions that, when violated, can lead to unreliable predictions and invalid statistical inference. First, **linearity**: the relationship between features and target must be linear. If the true relationship is quadratic, exponential, or otherwise non-linear, linear regression will systematically underfit. You can check this by plotting residuals vs predicted values—random scatter indicates linearity, while patterns suggest non-linearity. Solutions include adding polynomial features, transforming variables (log, sqrt), or using non-linear models.\n\nSecond, **independence**: observations must be independent of each other. Violations occur with time-series data (today\'s value depends on yesterday\'s), clustered data (patients from the same hospital), or spatial data (nearby locations are correlated). This affects standard error estimates and confidence intervals. Solutions include using time-series models (ARIMA), mixed-effects models for clustered data, or spatial regression techniques. Third, **homoscedasticity**: residuals should have constant variance across all prediction levels. Heteroscedasticity (non-constant variance) means the model is more certain for some predictions than others, violating statistical inference assumptions. Check with residual plots; fix with weighted least squares, robust standard errors, or transforming the target variable.\n\nFourth, **normality of residuals**: for valid statistical inference (confidence intervals, hypothesis tests), residuals should be approximately normally distributed. This matters less for large samples due to the Central Limit Theorem, but is important for small datasets. Check with Q-Q plots or Shapiro-Wilk tests. Fifth, **no multicollinearity**: features should not be highly correlated with each other, as this makes coefficient estimates unstable and hard to interpret. Check with VIF (Variance Inflation Factor); values >10 indicate problems. Solutions include removing redundant features, using PCA, or applying regularization (Ridge regression). While these assumptions are ideal, linear regression can still perform well with minor violations, especially for prediction (as opposed to inference) tasks.'
      },
      {
        question: 'What is the difference between simple and multiple linear regression?',
        answer: 'Simple linear regression uses a single feature to predict the target (y = β₀ + β₁x₁), producing a straight line in 2D space. It models how one independent variable affects the dependent variable. For example, predicting house price based solely on square footage. The model has two parameters: slope (β₁, how much y changes per unit change in x) and intercept (β₀, y-value when x=0). Visualization is straightforward—you can plot the data points and the fitted line on a 2D graph.\n\nMultiple linear regression uses multiple features (y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ), fitting a hyperplane in n-dimensional space. For example, predicting house price using square footage, number of bedrooms, age, and location. This is more realistic since most real-world outcomes depend on multiple factors. The model has n+1 parameters: one coefficient per feature plus an intercept. Each coefficient represents the partial effect of that feature while holding all other features constant—a crucial distinction from simple regression where the single feature captures all variation.\n\nThe key advantage of multiple regression is controlling for confounding variables. In simple regression predicting salary from years of experience, the coefficient might be inflated because it captures both experience and education level (correlated with experience). Multiple regression with both experience and education separates these effects, giving more accurate and interpretable coefficients. However, multiple regression is more complex: harder to visualize (>3D), more prone to multicollinearity issues, requires more data to avoid overfitting (rule of thumb: at least 10-20 samples per feature), and coefficients become harder to interpret when features are correlated. The choice depends on your problem: use simple regression for exploratory analysis or when you truly have one key predictor; use multiple regression for realistic modeling of complex phenomena with multiple influences.'
      },
      {
        question: 'How do you evaluate the performance of a linear regression model?',
        answer: 'For regression models, the primary evaluation metrics quantify prediction error. **Mean Squared Error (MSE)** averages the squared differences between predictions and actuals, penalizing large errors heavily. **Root Mean Squared Error (RMSE)** is the square root of MSE, returning error in the target\'s original units, making it more interpretable (e.g., "on average we\'re off by $15,000 in house price predictions"). **Mean Absolute Error (MAE)** averages absolute errors, treating all errors linearly and being more robust to outliers than RMSE. Choose RMSE when large errors are particularly bad; choose MAE when you want robustness to outliers.\n\n**R² (coefficient of determination)** measures the proportion of variance in the target explained by the model, ranging from negative infinity to 1. R²=1 means perfect predictions, R²=0 means the model performs no better than predicting the mean, and negative R² means worse than the mean baseline. R² is intuitive and widely used, but has limitations: it always increases when adding features (even random ones), doesn\'t indicate whether the model assumptions are met, and can be misleading with non-linear relationships. **Adjusted R²** penalizes model complexity, only increasing when new features genuinely improve fit, making it better for comparing models with different numbers of features.\n\nBeyond metrics, perform **residual analysis** to check assumptions. Plot residuals vs predicted values (should show random scatter with no patterns), Q-Q plots (should be roughly linear if residuals are normal), and residuals vs each feature (checking for non-linear relationships). Look for outliers with high residuals and high leverage points that disproportionately influence the model. Use **cross-validation** to assess generalization: k-fold CV gives robust performance estimates less dependent on a single train-test split. For statistical inference, examine coefficient p-values (are features significantly different from zero?), confidence intervals (range of plausible coefficient values), and F-statistic (is the overall model better than the null model?). The appropriate evaluation depends on your goal: prediction accuracy → focus on RMSE/MAE on held-out data; inference about relationships → focus on coefficient significance and assumption checking; model comparison → use adjusted R² or cross-validated metrics.'
      },
      {
        question: 'What is multicollinearity and how does it affect linear regression?',
        answer: 'Multicollinearity occurs when two or more features are highly correlated, meaning they contain redundant information. For example, in predicting house prices, "square footage" and "number of rooms" are likely correlated—bigger houses tend to have more rooms. Perfect multicollinearity (one feature is an exact linear combination of others, like having both "price in dollars" and "price in cents") makes the regression mathematically unsolvable, as there are infinitely many coefficient combinations that fit the data equally well. High multicollinearity (strong but imperfect correlation) causes instability in coefficient estimates.\n\nThe effects are problematic for interpretation but less so for prediction. Coefficient estimates become highly sensitive to small data changes—adding or removing a few samples can drastically change coefficients, even flipping their signs. Standard errors of coefficients inflate, making it harder to determine statistical significance. Coefficients become unreliable indicators of feature importance and can contradict domain knowledge (e.g., negative coefficient for bedrooms in house price prediction when bedrooms should logically increase price, because bedrooms correlate with square footage which captures the effect). However, prediction accuracy often remains good because what matters is the combined effect of correlated features, not individual coefficients.\n\nDetect multicollinearity using **Variance Inflation Factor (VIF)**: calculate how much each feature\'s variance is inflated due to correlation with other features. VIF=1 means no correlation; VIF>5 suggests problematic correlation; VIF>10 indicates severe multicollinearity requiring action. Alternatively, examine the correlation matrix for feature pairs with |r|>0.8-0.9. Solutions include: removing one of each correlated pair (keep the more interpretable or easier to collect), combining correlated features (e.g., create "total living space" from combining square footage features), using dimensionality reduction (PCA transforms to uncorrelated components), or applying Ridge regression which handles multicollinearity by penalizing large coefficients. Choose the solution based on whether you need interpretable coefficients (remove features) or just good predictions (use regularization).'
      },
      {
        question: 'When would you choose linear regression over other algorithms?',
        answer: 'Choose linear regression when interpretability is crucial and you need to explain how features affect predictions to stakeholders, regulators, or for scientific understanding. The linear model provides direct, quantifiable relationships: "each additional bedroom increases house price by $15,000, holding other factors constant." This transparency is invaluable in healthcare (understanding risk factors), economics (policy analysis), and legal contexts (fair lending decisions). More complex models like neural networks or gradient boosting may predict better but offer little insight into the underlying mechanisms.\n\nLinear regression excels when the true relationship is approximately linear or when you have limited data. With few samples (say, <1000), complex models easily overfit while linear regression\'s simplicity provides stable estimates. It also works well in high-dimensional settings (many features, few samples) when combined with regularization—Lasso and Ridge regression handle p >> n scenarios where traditional methods fail. The computational efficiency of linear regression matters for real-time prediction systems or when training millions of models (per-user personalization), as training and prediction are extremely fast compared to ensemble methods or deep learning.\n\nHowever, avoid linear regression when relationships are clearly non-linear (exponential growth, multiplicative interactions, threshold effects), when you need to capture complex interactions between many features, or when maximizing pure predictive accuracy is paramount regardless of interpretability. In these cases, consider: polynomial regression or GAMs for smooth non-linearities while maintaining some interpretability; tree-based methods (Random Forests, Gradient Boosting) for automatic interaction detection and non-linear patterns; neural networks for highly complex, unstructured data. Often, start with linear regression as a strong baseline—it establishes a minimum performance threshold and provides interpretable insights, then try more complex models if the accuracy gain justifies the loss of interpretability. Use regularization (Ridge, Lasso, Elastic Net) to modernize linear regression for contemporary high-dimensional data while retaining its fundamental advantages.'
      },
      {
        question: 'How do you handle outliers in linear regression?',
        answer: 'Start by identifying outliers using multiple approaches: examine residual plots for points with large errors, leverage plots for influential points (high influence on the fitted line), and Cook\'s distance combining both. Not all outliers are equal—some are data errors (typos, measurement errors) that should be corrected or removed, while others are legitimate rare events containing important information. Before removing anything, investigate: is it a data collection error, a rare but valid case, or a sign that your model is mis-specified?\n\nIf outliers are legitimate but skewing the model, consider several approaches. **Robust regression** methods like RANSAC or Huber regression give less weight to outliers compared to standard OLS, which squares residuals and thus heavily penalizes outliers. **Transformation** of the target variable can help: log-transform for right-skewed data with high-value outliers (common in house prices, income), square root for count data. These transformations compress the range of extreme values, reducing their influence. **Winsorization** caps extreme values at a percentile (e.g., 95th) rather than removing them, preserving sample size while limiting outlier impact.\n\nFor feature outliers (extreme predictor values), consider: feature clipping at reasonable thresholds based on domain knowledge, creating binary indicators for extreme values while capping the continuous variable (e.g., "income" clipped at $500k plus "high_earner" flag), or using tree-based models instead which are naturally robust to outliers. If you decide to remove outliers, document the decision and its impact: report model performance with and without outliers, ensure removing them doesn\'t disproportionately affect certain subgroups (creating bias), and be cautious with automatic removal rules (e.g., "remove all points >3 standard deviations") as they can discard valuable information. Remember: an outlier in your training data might become common in production, so ensure your handling strategy generalizes. Sometimes, the outliers indicate you need a different model class entirely—if many points don\'t fit the linear pattern, consider non-linear models rather than forcing linearity by removing inconvenient data.'
      },
      {
        question: 'What is the normal equation and when would you use it vs gradient descent?',
        answer: 'The normal equation is a closed-form analytical solution to find optimal coefficients in linear regression: β = (XᵀX)⁻¹Xᵀy, where X is the feature matrix and y is the target vector. This directly computes the coefficients that minimize mean squared error without iteration—you calculate the matrix inverse and matrix multiplications, and you\'re done. It\'s exact (not approximate), requires no hyperparameter tuning (no learning rate), and always converges to the global optimum for linear regression (the loss surface is convex with a single minimum).\n\nUse the normal equation for small to medium datasets (roughly <10,000 samples and <1,000 features) where computation time isn\'t prohibitive. The bottleneck is computing (XᵀX)⁻¹, which has O(n³) complexity where n is the number of features. For 100 features, this is manageable; for 10,000 features, it becomes impractical. The normal equation also requires XᵀX to be invertible (non-singular). If you have perfect multicollinearity or more features than samples (p > n), the matrix is singular and can\'t be inverted. In practice, libraries use pseudo-inverse (Moore-Penrose inverse) which handles these cases, but the solution may be numerically unstable.\n\nGradient descent is an iterative optimization algorithm that updates coefficients step by step: β := β - α∇L(β), moving in the direction of steepest descent. Use gradient descent when: dataset is large (>10,000 samples), as each iteration processes all samples but is still faster than the normal equation; you have many features (>1,000), avoiding expensive matrix inversion; you\'re using variations like stochastic gradient descent (SGD) or mini-batch GD for even better scaling; you need online learning (updating the model as new data arrives); or you\'re using regularization like Ridge regression (has closed form) or Lasso (no closed form, requires iterative methods). Gradient descent requires tuning the learning rate and monitoring convergence, but scales much better: O(knd) where k is iterations, n is samples, d is features, typically much faster than O(d³) for large d. Modern practice: use the normal equation for quick prototyping with small data, gradient descent (especially with libraries\' optimized implementations) for production systems and large-scale problems.'
      }
    ],
    quizQuestions: [
      {
        id: 'lr1',
        question: 'What is the cost function typically used in linear regression?',
        options: ['Mean Absolute Error', 'Mean Squared Error', 'Cross-entropy', 'Hinge Loss'],
        correctAnswer: 1,
        explanation: 'Linear regression typically uses Mean Squared Error (MSE) as its cost function because it\'s differentiable and penalizes larger errors more heavily.'
      },
      {
        id: 'lr2',
        question: 'Which assumption is NOT required for linear regression?',
        options: ['Linear relationship', 'Normal distribution of features', 'Independence of residuals', 'Homoscedasticity'],
        correctAnswer: 1,
        explanation: 'Linear regression requires normal distribution of residuals, not features. The features themselves don\'t need to be normally distributed.'
      }
    ]
  },

  'logistic-regression': {
    id: 'logistic-regression',
    title: 'Logistic Regression',
    category: 'classical-ml',
    description: 'Learn about logistic regression for binary and multiclass classification problems.',
    content: `
      <h2>Logistic Regression</h2>
      <p>Logistic regression is a statistical method used for binary classification problems. Despite its name, it's a classification algorithm, not a regression algorithm.</p>

      <h3>Sigmoid Function</h3>
      <p>Logistic regression uses the sigmoid (logistic) function to map any real number to a value between 0 and 1:</p>
      <p><strong>σ(z) = 1 / (1 + e^(-z))</strong></p>
      <p>Where z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ</p>

      <h3>Decision Boundary</h3>
      <p>The decision boundary is where p(y=1|x) = 0.5, which occurs when z = 0.</p>

      <h3>Cost Function</h3>
      <p>Logistic regression uses the logistic loss (cross-entropy loss):</p>
      <p><strong>J(θ) = -(1/m) Σ[y log(h(x)) + (1-y) log(1-h(x))]</strong></p>

      <h3>Multiclass Extension</h3>
      <p>For multiclass problems, logistic regression can be extended using:</p>
      <ul>
        <li><strong>One-vs-Rest (OvR):</strong> Train one binary classifier per class</li>
        <li><strong>Multinomial Logistic Regression:</strong> Use softmax function</li>
      </ul>

      <h3>Advantages</h3>
      <ul>
        <li>Provides probability estimates</li>
        <li>No tuning of hyperparameters required</li>
        <li>Less prone to overfitting</li>
        <li>Fast and efficient</li>
        <li>Interpretable coefficients</li>
      </ul>

      <h3>Disadvantages</h3>
      <ul>
        <li>Assumes linear relationship between features and log-odds</li>
        <li>Sensitive to outliers</li>
        <li>Requires large sample sizes for stable results</li>
        <li>Can struggle with complex relationships</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                          n_informative=2, n_clusters_per_class=1, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"Classification Report:\\n{classification_report(y_test, y_pred)}")

# Model coefficients
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")`,
        explanation: 'This example shows how to implement logistic regression for binary classification using scikit-learn.'
      }
    ],
    interviewQuestions: [
      {
        question: 'What is the difference between linear and logistic regression?',
        answer: 'The fundamental difference is in what they predict and their output ranges. Linear regression predicts continuous values (any real number) and outputs y = β₀ + β₁x₁ + β₂x₂ + ... directly. Logistic regression predicts probabilities (bounded between 0 and 1) for classification tasks and outputs P(y=1|X) = σ(β₀ + β₁x₁ + β₂x₂ + ...), where σ is the sigmoid function that squashes any real number into [0,1]. Linear regression is for regression problems (predicting house prices, temperature), while logistic regression is for binary classification (spam/not spam, disease/healthy).\n\nThe loss functions differ significantly. Linear regression uses Mean Squared Error (MSE), which is appropriate for continuous targets and has nice mathematical properties (convex, differentiable everywhere). Logistic regression uses log loss (binary cross-entropy): -[y log(p) + (1-y) log(1-p)], which heavily penalizes confident wrong predictions. This asymmetric penalty is crucial for classification—being 99% confident and wrong is much worse than being 60% confident and wrong. MSE would be a poor choice for logistic regression as it can lead to non-convex optimization surfaces and doesn\'t properly capture the classification objective.\n\nInterpretation of coefficients also differs. In linear regression, a coefficient tells you how much the target changes per unit change in the feature. In logistic regression, coefficients represent log-odds ratios: a one-unit increase in feature x₁ multiplies the odds of the positive class by e^β₁. For example, if β₁ = 0.5, increasing x₁ by 1 unit multiplies odds by e^0.5 ≈ 1.65 (65% increase in odds). This non-linear relationship through the sigmoid means feature effects on probability are not constant—they depend on the baseline probability. Despite the name similarity, these are fundamentally different algorithms serving different purposes, though both are linear models (linear in their parameters, before any transformation).'
      },
      {
        question: 'Why do we use the sigmoid function in logistic regression?',
        answer: 'The sigmoid function σ(z) = 1/(1 + e^(-z)) transforms any real-valued number into a probability between 0 and 1, which is essential for binary classification. Without it, the linear combination β₀ + β₁x₁ + β₂x₂ + ... could output any value (-∞ to +∞), which can\'t be interpreted as a probability. The sigmoid provides this crucial mapping: as z approaches +∞, σ(z) approaches 1; as z approaches -∞, σ(z) approaches 0; at z=0, σ(z)=0.5. This creates a smooth S-shaped curve that transitions from "definitely class 0" to "definitely class 1" with a decision boundary at the midpoint.\n\nThe sigmoid has desirable mathematical properties that make optimization tractable. It\'s differentiable everywhere with a particularly nice derivative: σ\'(z) = σ(z)(1 - σ(z)), which appears naturally when computing gradients for backpropagation. This derivative is always positive (between 0 and 0.25), ensuring smooth gradient descent. The sigmoid also has a probabilistic interpretation via the logistic distribution and connects to log-odds: if p = σ(z), then z = log(p/(1-p)), the log-odds or logit. This means logistic regression is modeling log-odds as a linear function of features, which is why it\'s called logistic regression.\n\nAlternatives exist but have drawbacks. The step function (0 if z<0, 1 if z≥0) gives binary outputs but isn\'t differentiable at 0, making gradient-based optimization impossible. Tanh is similar to sigmoid (also S-shaped) but outputs [-1,1] rather than [0,1], requiring rescaling for probabilities. The probit function (cumulative Gaussian) is sometimes used but is computationally more expensive and lacks the nice derivative property. The sigmoid\'s combination of probabilistic interpretation, smooth differentiability, and computational efficiency makes it the standard choice for binary classification. For multiclass problems, we generalize to softmax (vector version of sigmoid), but the same principles apply—transform unbounded scores into valid probabilities.'
      },
      {
        question: 'What is the cost function for logistic regression and why?',
        answer: 'Logistic regression uses **log loss** (also called binary cross-entropy or logistic loss): L = -[y log(p) + (1-y) log(1-p)], where y is the true label (0 or 1) and p is the predicted probability. For the positive class (y=1), this reduces to -log(p), which approaches 0 when p→1 (correct, confident prediction) and approaches ∞ when p→0 (incorrect, confident prediction). For the negative class (y=0), it becomes -log(1-p), which similarly penalizes confident wrong predictions. This asymmetric penalty is crucial: being 99% confident and wrong incurs massive loss, while being 51% confident and correct incurs small loss.\n\nWhy not use Mean Squared Error (MSE) as in linear regression? MSE with sigmoid creates a non-convex loss surface with many local minima, making optimization difficult and unreliable—gradient descent might get stuck. Log loss with sigmoid produces a convex loss surface with a single global minimum, guaranteeing gradient descent will converge to the optimal solution. Mathematically, log loss is the negative log-likelihood of the Bernoulli distribution, meaning maximizing likelihood (statistics) equals minimizing log loss (machine learning). This connects logistic regression to maximum likelihood estimation (MLE), providing a principled probabilistic foundation.\n\nThe gradient of log loss with respect to predictions has a particularly elegant form: ∂L/∂z = p - y (where z is the linear combination before sigmoid). This simplicity—the gradient is just the error—makes backpropagation straightforward and computationally efficient. The loss also automatically calibrates predictions: with log loss, predicted probabilities tend to match true frequencies (e.g., among samples predicted at 70%, about 70% are actually positive). This calibration is valuable for decision-making under uncertainty, unlike hinge loss (used in SVMs) which only cares about correct classification, not probability accuracy. Log loss generalizes naturally to multiclass via categorical cross-entropy, making it the standard classification loss for both shallow models (logistic regression) and deep learning (neural networks with softmax output).'
      },
      {
        question: 'How do you interpret the coefficients in logistic regression?',
        answer: 'Logistic regression coefficients represent **log-odds ratios**, which requires careful interpretation. If β₁ = 0.5 for feature x₁, a one-unit increase in x₁ increases the log-odds by 0.5. Equivalently, it multiplies the odds by e^0.5 ≈ 1.65. Odds are defined as P(y=1)/P(y=0), so if initial odds are 1:1 (50% probability), they become 1.65:1 (62% probability) after increasing x₁ by one unit. Positive coefficients increase the probability of the positive class; negative coefficients decrease it. The magnitude indicates strength of effect: larger |β| means stronger influence.\n\nThe non-linearity of the sigmoid creates complexity: the same coefficient change has different effects on probability depending on the baseline. Near probabilities of 0.5, small coefficient changes significantly affect probability. Near 0 or 1, even large coefficient changes barely move the probability (sigmoid is nearly flat at the extremes). For example, with β₁=1, increasing x₁ from 0 to 1 when all other features are 0 (and intercept=0) moves probability from 0.5 to 0.73. But if starting probability is 0.01, the same change only moves it to 0.027. This means you can\'t say "x₁ increases probability by X%" universally—it depends on context.\n\nFor practical interpretation, consider several approaches. Calculate **marginal effects**: the change in probability from a one-unit increase in the feature, evaluated at meaningful points (mean values of other features, or specific scenarios of interest). Exponentiate coefficients to get **odds ratios**: e^β is easier to communicate than log-odds. For categorical variables, the odds ratio directly compares categories (e.g., "smokers have 2.5× the odds of disease compared to non-smokers"). Visualize the relationship: plot predicted probability vs the feature of interest while holding others constant, showing the S-curve. For standardized coefficients (features scaled to mean 0, std 1), magnitudes are comparable, indicating relative importance. Remember: logistic regression is linear in log-odds space, not probability space, which is why interpretation requires these transformations.'
      },
      {
        question: 'How do you extend logistic regression to multiclass problems?',
        answer: 'The standard extension is **multinomial logistic regression** (also called softmax regression), which generalizes binary logistic regression to K classes. Instead of one sigmoid output, you have K outputs (one per class) using the softmax function: P(y=k|X) = e^(z_k) / Σ(e^(z_j)) for j=1 to K, where z_k = β₀ᵏ + β₁ᵏx₁ + β₂ᵏx₂ + .... Each class gets its own set of weights. Softmax ensures outputs are valid probabilities: all between 0 and 1, summing to 1. For prediction, choose the class with highest probability: argmax_k P(y=k|X).\n\nThe loss function extends to **categorical cross-entropy**: L = -Σ y_k log(p_k), where y is a one-hot encoded vector (1 for true class, 0 elsewhere) and p is the predicted probability vector. For a sample with true class 2, this reduces to -log(p₂), heavily penalizing low probability for the correct class. Training uses gradient descent just like binary logistic regression, but now you\'re optimizing K sets of weights simultaneously. The model learns how each feature affects the probability of each class relative to a reference class (typically the last class, whose weights can be set to zero for identifiability).\n\nAn alternative approach is **one-vs-rest** (OvR): train K binary classifiers, each distinguishing one class from all others. For a 3-class problem (A, B, C), train three binary classifiers: A vs (B,C), B vs (A,C), C vs (A,B). At prediction time, run all K classifiers and choose the class with highest probability. OvR is simpler to implement and can use any binary classifier, but has issues: predicted probabilities may not sum to 1 (each classifier is independent), and doesn\'t directly model class correlations. Multinomial logistic regression is generally preferred as it\'s theoretically more principled and often more accurate.\n\nFor ordinal multiclass problems (ordered classes like low/medium/high severity), use **ordinal logistic regression** which respects the ordering. This uses cumulative probabilities and ensures predictions follow the natural order. For many-class problems (100s of classes), computational cost grows linearly with K for both approaches. Hierarchical softmax or other specialized techniques can improve efficiency. The choice depends on your problem structure: unordered classes → multinomial logistic regression; need quick implementation → one-vs-rest; ordinal classes → ordinal logistic regression.'
      },
      {
        question: 'What are the assumptions of logistic regression?',
        answer: 'Logistic regression makes several key assumptions, though they\'re generally less restrictive than linear regression. First, **linearity of log-odds**: the model assumes the log-odds (logit) is a linear function of features: log(p/(1-p)) = β₀ + β₁x₁ + β₂x₂ + .... This means each feature has a constant effect on log-odds regardless of feature values. If the true relationship is non-linear (e.g., quadratic), the model will underfit. Check this by plotting log-odds vs each feature or using Box-Tidwell test. Solutions include adding polynomial terms, interaction terms, or using non-linear models (decision trees, neural networks).\n\nSecond, **independence of observations**: samples must be independent. Violations occur with repeated measures (same patient over time), clustered data (students within schools), or matched pairs. This affects standard errors and p-values, making significance tests unreliable. Solutions include using mixed-effects logistic regression for clustered data, conditional logistic regression for matched pairs, or GEE (Generalized Estimating Equations) for correlated data. Third, **no perfect multicollinearity**: features shouldn\'t be perfectly correlated (one is a linear combination of others). This makes coefficients unidentifiable—infinite solutions exist. High (but not perfect) multicollinearity inflates standard errors and makes coefficients unstable. Check with VIF; address by removing redundant features or using regularization (L1/L2).\n\nFourth, **large sample size**: logistic regression is asymptotic, meaning nice statistical properties (coefficient estimates, standard errors, confidence intervals) hold in large samples. Rule of thumb: at least 10-15 events per predictor variable for the minority class. With small samples (<100) or rare events (<10% of samples in minority class), consider exact logistic regression, Firth\'s penalized likelihood, or other specialized methods. Unlike linear regression, logistic regression does NOT assume normality of features or residuals, homoscedasticity of residuals, or continuous features (categorical features are fine). It also handles non-constant variance naturally through the binomial variance structure. These relaxed assumptions make logistic regression quite flexible and applicable to many real-world classification problems, though checking the linearity of log-odds assumption is crucial for reliable predictions.'
      }
    ],
    quizQuestions: [
      {
        id: 'log1',
        question: 'What function does logistic regression use to map inputs to probabilities?',
        options: ['Linear function', 'Sigmoid function', 'ReLU function', 'Softmax function'],
        correctAnswer: 1,
        explanation: 'Logistic regression uses the sigmoid (logistic) function to map any real number to a probability between 0 and 1.'
      }
    ]
  },

  'decision-trees': {
    id: 'decision-trees',
    title: 'Decision Trees',
    category: 'classical-ml',
    description: 'Understanding decision trees for both classification and regression tasks.',
    content: `
      <h2>Decision Trees</h2>
      <p>Decision trees are versatile machine learning algorithms that can perform both classification and regression tasks. They model decisions through a series of questions, creating a tree-like structure.</p>

      <h3>How Decision Trees Work</h3>
      <p>A decision tree splits the data based on feature values, creating internal nodes (decisions) and leaf nodes (predictions). Each internal node represents a test on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label or regression value.</p>

      <h3>Splitting Criteria</h3>
      <p>For <strong>Classification:</strong></p>
      <ul>
        <li><strong>Gini Impurity:</strong> Measures how often a randomly chosen element would be incorrectly labeled</li>
        <li><strong>Entropy:</strong> Measures the amount of information or uncertainty</li>
        <li><strong>Information Gain:</strong> Reduction in entropy after splitting</li>
      </ul>

      <p>For <strong>Regression:</strong></p>
      <ul>
        <li><strong>Mean Squared Error (MSE):</strong> Average of squared differences</li>
        <li><strong>Mean Absolute Error (MAE):</strong> Average of absolute differences</li>
      </ul>

      <h3>Tree Building Process</h3>
      <ol>
        <li>Select the best feature to split on</li>
        <li>Create child nodes for each possible value of that feature</li>
        <li>Recursively repeat for each child node</li>
        <li>Stop when stopping criteria are met</li>
      </ol>

      <h3>Stopping Criteria</h3>
      <ul>
        <li>Maximum depth reached</li>
        <li>Minimum samples per leaf</li>
        <li>Minimum samples to split</li>
        <li>No improvement in purity</li>
      </ul>

      <h3>Advantages</h3>
      <ul>
        <li>Easy to understand and interpret</li>
        <li>Requires little data preparation</li>
        <li>Handles both numerical and categorical data</li>
        <li>Can capture non-linear patterns</li>
        <li>Handles missing values naturally</li>
      </ul>

      <h3>Disadvantages</h3>
      <ul>
        <li>Prone to overfitting</li>
        <li>Can create overly complex trees</li>
        <li>Unstable (small data changes can result in different trees)</li>
        <li>Biased toward features with more levels</li>
        <li>Difficulty capturing linear relationships</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn import tree
import matplotlib.pyplot as plt

# Classification example
X_clf, y_clf = make_classification(n_samples=1000, n_features=4, n_informative=3,
                                  n_redundant=1, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42)

# Train classification tree
clf_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
clf_tree.fit(X_train_clf, y_train_clf)

# Predictions and evaluation
y_pred_clf = clf_tree.predict(X_test_clf)
clf_accuracy = accuracy_score(y_test_clf, y_pred_clf)
print(f"Classification Accuracy: {clf_accuracy:.4f}")

# Feature importance
print("Feature Importances:")
for i, importance in enumerate(clf_tree.feature_importances_):
    print(f"Feature {i}: {importance:.4f}")`,
        explanation: 'This example demonstrates how to train a decision tree classifier and examine feature importances.'
      },
      {
        language: 'Python',
        code: `# Regression example
X_reg, y_reg = make_regression(n_samples=1000, n_features=4, noise=0.1, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

# Train regression tree
reg_tree = DecisionTreeRegressor(max_depth=5, min_samples_split=20, random_state=42)
reg_tree.fit(X_train_reg, y_train_reg)

# Predictions and evaluation
y_pred_reg = reg_tree.predict(X_test_reg)
reg_mse = mean_squared_error(y_test_reg, y_pred_reg)
print(f"Regression MSE: {reg_mse:.4f}")

# Visualize tree structure (first few levels)
# tree.plot_tree(clf_tree, max_depth=3, feature_names=[f'Feature_{i}' for i in range(4)])
# plt.show()`,
        explanation: 'This shows how to use decision trees for regression tasks and evaluate performance.'
      }
    ],
    interviewQuestions: [
      {
        question: 'How do decision trees work?',
        answer: 'Decision trees learn by recursively partitioning the feature space into regions and making predictions based on the training examples within each region. The algorithm starts with all data at the root node and iteratively asks binary questions about feature values to split the data into increasingly pure subsets. Each internal node represents a decision based on a feature threshold (e.g., "is age > 30?"), each branch represents the outcome (yes/no), and each leaf node contains the final prediction—either a class label (classification) or a value (regression).\n\nThe splitting process follows a greedy, top-down approach. At each node, the algorithm evaluates all possible splits (every feature and every threshold value) and selects the one that best separates the data according to a splitting criterion like Gini impurity or entropy. For example, if predicting loan default, it might first split on "income > $50k", then split the high-income group on "credit score > 700", and the low-income group on "employment length > 2 years". This creates a hierarchical decision structure that mimics human decision-making: a series of simple questions leading to a conclusion.\n\nThe tree continues growing until a stopping criterion is met: reaching maximum depth, having too few samples to split, or achieving perfect purity. For prediction, you traverse the tree from root to leaf following the path determined by the input features, then output the leaf\'s prediction. The key insight is that decision trees partition the feature space into rectangular regions (each defined by a sequence of splits) and make constant predictions within each region. This allows them to capture non-linear patterns and interactions between features without requiring explicit feature engineering, though it can lead to overfitting if the tree grows too deep.'
      },
      {
        question: 'What are the different splitting criteria used in decision trees?',
        answer: 'For classification tasks, the two primary criteria are **Gini impurity** and **entropy** (information gain). Gini impurity measures how often a randomly chosen element would be incorrectly labeled if randomly labeled according to the class distribution in the node: Gini = 1 - Σ(p_i)², where p_i is the fraction of samples belonging to class i. It ranges from 0 (perfect purity, all samples same class) to 0.5 for binary classification (maximum impurity, 50-50 split). Information gain uses entropy: Entropy = -Σ(p_i log₂(p_i)), ranging from 0 (pure) to log₂(num_classes) (uniform distribution). The algorithm selects splits that maximize the reduction in impurity/entropy.\n\nThese criteria differ slightly in behavior. Gini impurity is computationally faster (no logarithm) and tends to favor creating pure nodes, isolating the most frequent class. Entropy is more sensitive to changes in class probabilities and may create more balanced trees. In practice, they usually yield similar results, and scikit-learn defaults to Gini for speed. A third criterion, **classification error** (1 - max(p_i)), is simpler but less sensitive to changes in class probabilities, making it inferior for growing trees (though sometimes used for pruning decisions).\n\nFor regression tasks, the primary criteria are **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)**. MSE minimizes variance: it computes the average squared difference between samples in a node and the node\'s mean prediction. Splits are chosen to minimize the weighted sum of MSE in child nodes. MAE uses absolute differences instead of squared ones, making it more robust to outliers—extreme values don\'t disproportionately influence splits. MSE is more common due to its connection to variance reduction, but MAE is preferable when your data contains outliers or when you care equally about all errors. Some implementations also support Friedman MSE, a variant that helps with splits near node boundaries. The choice of criterion depends on your problem: for classification, Gini vs entropy rarely matters; for regression, choose MSE for typical cases and MAE when outliers are problematic.'
      },
      {
        question: 'What is the difference between Gini impurity and entropy?',
        answer: 'Gini impurity and entropy are both measures of node impurity for classification decision trees, but they have different mathematical formulations and subtle behavioral differences. **Gini impurity** is calculated as Gini = 1 - Σ(p_i)², representing the probability of misclassifying a randomly chosen element if labeled according to the class distribution. For binary classification, it\'s Gini = 1 - (p² + (1-p)²), reaching maximum (0.5) at p=0.5 (50-50 split) and minimum (0) when p=0 or p=1 (pure node). **Entropy** measures information or uncertainty: Entropy = -Σ(p_i log₂(p_i)), where log₂ reflects "bits of information." For binary classification, max entropy is 1 bit at p=0.5; min is 0 for pure nodes.\n\nComputationally, Gini is faster to calculate since it avoids logarithms, which matters when evaluating thousands of potential splits. Entropy requires log operations that are more expensive. Both are concave functions that peak at uniform distributions, but their shapes differ slightly. Gini is more like an inverted parabola, while entropy is more gradually curved. This means Gini tends to be slightly more biased toward isolating the most frequent class into pure nodes, while entropy is more sensitive to probability changes and may create more balanced splits. However, these differences are usually minor—empirical studies show they produce similar trees in most cases.\n\nThe choice between them often comes down to convention and performance. Scikit-learn defaults to Gini for computational efficiency. Information gain (entropy reduction) has stronger theoretical foundations in information theory and connects to concepts like KL divergence and mutual information. Some argue it\'s more "principled" for this reason. In practice, choose Gini for faster training and entropy if you want more balanced trees or are interested in information-theoretic interpretations. Cross-validation will usually reveal minimal performance differences. A more important decision is choosing appropriate stopping criteria (max_depth, min_samples_split) and handling class imbalance, which affect model quality far more than the Gini vs entropy choice.'
      },
      {
        question: 'How do you prevent overfitting in decision trees?',
        answer: 'Decision trees are highly prone to overfitting because they can grow arbitrarily deep, creating complex decision boundaries that memorize training data noise. The primary prevention strategies involve **pre-pruning** (stopping early) and **post-pruning** (growing full then cutting back). Pre-pruning uses stopping criteria: **max_depth** limits tree depth (typical values 3-10), **min_samples_split** requires minimum samples to split a node (typical 20-50), **min_samples_leaf** requires minimum samples per leaf (typical 10-20), and **max_leaf_nodes** caps total leaves. These directly constrain model complexity. For example, max_depth=3 limits the tree to 3 levels of questions, preventing overly specific rules like "if age=37.2 and income=52,341 then...".\n\n**Post-pruning** grows a full tree then removes branches that don\'t improve performance on validation data. Cost-complexity pruning (minimal cost-complexity pruning) is most common: it penalizes tree complexity by adding α × (number of leaves) to the error function, then finds the α that minimizes cross-validated error. This is more sophisticated than pre-pruning as it makes data-driven decisions about which branches to remove, rather than applying uniform constraints. Scikit-learn\'s DecisionTreeClassifier supports this via ccp_alpha parameter. Post-pruning often produces better results but is computationally more expensive since you must grow the full tree first.\n\n**Ensemble methods** provide the most effective overfitting prevention. Random Forests train many trees on bootstrap samples with random feature subsets, averaging predictions to reduce variance. Gradient Boosting builds shallow trees sequentially, with each correcting previous errors. Even a single decision tree with max_depth=1 (decision stump) can be powerful in an ensemble. Other techniques include **feature sampling** (consider only a random subset of features for each split, even in single trees), **minimum impurity decrease** (only split if impurity reduction exceeds a threshold), and using **cross-validation** to select hyperparameters. In practice, start with moderate depth (5-7) and minimum samples (20-50), then tune via cross-validation. Or simply use Random Forest instead of a single tree—it\'s more robust with less hyperparameter sensitivity.'
      },
      {
        question: 'What are the advantages and disadvantages of decision trees?',
        answer: '**Advantages**: Decision trees excel in interpretability—you can visualize the entire decision process and explain predictions to non-technical stakeholders ("you were rejected because income < $50k AND credit score < 600"). This transparency is valuable in regulated industries (healthcare, finance) where model decisions must be justified. Trees require minimal data preprocessing: no need for feature scaling, normalization, or one-hot encoding of categoricals (they handle them natively). They automatically capture non-linear relationships and feature interactions without manual engineering. Trees also handle missing values naturally through surrogate splits and are non-parametric (make no assumptions about data distributions).\n\nTrees are fast to train and predict, making them suitable for real-time applications. Feature importance is automatically calculated, aiding in understanding which variables matter most. They work for both classification and regression, handle mixed data types (numeric and categorical), and scale reasonably well with parallelization. For small to medium datasets, a well-tuned tree can be highly competitive with more complex models while remaining far more interpretable.\n\n**Disadvantages**: The biggest issue is instability—small changes in data can produce completely different trees, making them unreliable for inference (coefficients/splits vary wildly across bootstrap samples). Trees are prone to overfitting, creating overly complex boundaries that don\'t generalize, especially with noisy data or many features. They struggle with extrapolation (can\'t predict outside the range of training data) and have difficulty modeling linear relationships (require many splits to approximate a simple line). Trees are biased toward features with more levels/values since they provide more splitting opportunities. They often underperform compared to linear models on problems with strong linear relationships or compared to ensembles on complex problems. The greedy splitting algorithm can miss better splits that would emerge from looking ahead multiple levels. In practice, Random Forests and Gradient Boosting address many of these weaknesses (especially instability and overfitting) while sacrificing interpretability, making single decision trees less common in production except for their simplicity and explainability advantages.'
      },
      {
        question: 'How do decision trees handle missing values?',
        answer: 'Decision trees can handle missing values in several ways, with the approach varying by implementation. The most sophisticated method is **surrogate splits**, used by CART (Classification and Regression Trees) and implemented in R\'s rpart. When a split uses a feature that has missing values, the algorithm finds surrogate splits—alternative features that produce similar partitions. During training, it identifies backup splits that correlate with the primary split. At prediction time, if the primary feature is missing, the algorithm uses the best available surrogate. This approach leverages the data\'s structure to impute the missing value\'s likely direction implicitly.\n\n**Scikit-learn\'s approach** differs: it treats missing values as informative and learns the best direction to send them. When evaluating a split, it tries sending all missing values left, then right, and chooses the direction that maximizes impurity reduction. This is simpler than surrogates but still data-driven—if missingness correlates with the target (e.g., income often missing for low earners), the tree learns this pattern. However, scikit-learn requires preprocessing missing values (won\'t accept NaN), so in practice, you must explicitly mark them (e.g., with -999 or a separate binary feature) before training.\n\n**XGBoost and LightGBM** have native missing value support and automatically learn the optimal direction. During training, they evaluate splits by trying missing values in both directions and choosing the one that improves the objective most. This is efficient and effective, automatically discovering whether missing values should be grouped with high or low values of the feature. The advantage is zero preprocessing required—just pass data with NaN and the algorithm handles it. An alternative approach used by some implementations is to simply remove samples with missing values for that split (send them down both branches with fractional weights), though this is less efficient. In practice, tree-based models\' native missing value handling is a significant advantage over linear models, which always require imputation or deletion, though explicit imputation (mean, median, or sophisticated methods like iterative imputation) often still improves performance.'
      },
      {
        question: 'What is pruning and why is it important?',
        answer: '**Pruning** is the process of reducing decision tree size by removing branches that provide little predictive power, typically to prevent overfitting. A fully grown tree often memorizes training data, creating leaves with very few samples and complex decision boundaries based on noise. Pruning simplifies the tree by eliminating these overspecialized branches, improving generalization to unseen data. The core idea is the bias-variance tradeoff: a full tree has low bias but high variance (overfits), while a pruned tree increases bias slightly but decreases variance substantially, often improving overall test performance.\n\n**Pre-pruning** (early stopping) prevents the tree from growing in the first place by applying stopping criteria during training: max_depth, min_samples_split, min_samples_leaf, etc. It\'s computationally efficient since you never build the complex structure. However, it uses fixed thresholds that may be too aggressive (stopping before finding good splits deeper in the tree, "horizon effect") or too lenient (still overfitting). Pre-pruning can\'t see into the future—it doesn\'t know if a seemingly poor split might enable excellent splits in child nodes.\n\n**Post-pruning** grows the full tree then removes branches retroactively, making data-driven decisions about which branches to eliminate. The most common technique is **cost-complexity pruning** (minimal cost-complexity pruning or weakest link pruning). It defines a cost function: Total Cost = Error + α × (number of leaves), where α controls the complexity penalty. For each value of α, it finds the smallest tree that minimizes total cost, creating a sequence of nested trees from full (α=0) to just the root (α=∞). Then, it uses cross-validation to select the α with the best validation performance. This approach is principled and data-driven but computationally expensive since you must build the full tree first.\n\nPruning is crucial because unpruned trees almost always overfit on real-world data, especially with limited samples or noisy features. It improves test accuracy, reduces model complexity (fewer nodes = faster predictions, less memory), and enhances interpretability (simpler trees are easier to understand and visualize). In scikit-learn, use the ccp_alpha parameter to control post-pruning strength. In practice, post-pruning generally outperforms pre-pruning when computational resources allow, but ensembles like Random Forest (which use pre-pruned diverse trees) often work better than carefully pruned single trees.'
      },
      {
        question: 'How do you interpret feature importance in decision trees?',
        answer: 'Decision trees calculate feature importance based on how much each feature reduces impurity (weighted by the number of samples it affects). Specifically, for each feature, the algorithm sums the impurity decrease across all nodes where that feature was used for splitting, weighted by the proportion of samples reaching that node. Features used higher in the tree (near the root) and for splits that significantly improve purity receive higher importance scores. The scores are then normalized to sum to 1, giving each feature a proportion of total importance. A score of 0.3 means that feature contributed 30% of the model\'s total impurity reduction.\n\nThis importance measure captures which features are most useful for making predictions in this particular dataset and tree. Features that never appear in the tree have zero importance. Features that appear multiple times or in high-impact splits get high scores. However, this measure has important limitations. It\'s **biased toward high-cardinality features**—features with many unique values (like continuous variables or IDs) have more potential split points, giving them more opportunities to appear important even if they\'re not truly predictive. This is especially problematic if you include spurious high-cardinality features. The importance is also **unstable**: training on a different sample can produce very different importance rankings due to the tree\'s instability.\n\n**Correlated features** create another interpretation challenge. If two features are highly correlated (e.g., height in inches and height in cm), the tree will pick one arbitrarily for early splits, giving it high importance and the other low importance, even though they\'re equally informative. This makes importance scores unreliable for causal inference or understanding "true" feature effects. To get more reliable importance estimates, use **ensemble methods**: Random Forest and Gradient Boosting average importance across many trees, smoothing out the instability. Random Forest also provides **permutation importance**: shuffle each feature and measure performance drop, which is more robust to correlation and cardinality biases. For critical decisions, combine multiple importance measures (tree-based, permutation, SHAP values) and be cautious about interpreting them causally—they show predictive utility in this model, not necessarily causal relationships in the real world.'
      }
    ],
    quizQuestions: [
      {
        id: 'dt1',
        question: 'Which metric is commonly used for splitting in classification decision trees?',
        options: ['Mean Squared Error', 'Gini Impurity', 'Mean Absolute Error', 'R-squared'],
        correctAnswer: 1,
        explanation: 'Gini Impurity is commonly used for splitting in classification decision trees as it measures the probability of misclassifying a randomly chosen element.'
      },
      {
        id: 'dt2',
        question: 'What is a major disadvantage of decision trees?',
        options: ['Cannot handle categorical data', 'Difficult to interpret', 'Prone to overfitting', 'Cannot handle missing values'],
        correctAnswer: 2,
        explanation: 'Decision trees are prone to overfitting, especially when they grow deep and complex, capturing noise in the training data.'
      }
    ]
  },

  'random-forests': {
    id: 'random-forests',
    title: 'Random Forests',
    category: 'classical-ml',
    description: 'Ensemble learning method that combines multiple decision trees for robust predictions',
    content: `
      <h2>Random Forests</h2>
      <p>Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode (classification) or mean (regression) of individual trees. It addresses the overfitting problem of single decision trees.</p>

      <h3>Key Concepts</h3>

      <h4>Bootstrap Aggregating (Bagging)</h4>
      <ul>
        <li>Creates multiple subsets of training data through random sampling with replacement</li>
        <li>Each tree is trained on a different bootstrap sample</li>
        <li>Reduces variance by averaging predictions from multiple trees</li>
        <li>Each sample typically contains ~63% of unique training instances</li>
      </ul>

      <h4>Feature Randomness</h4>
      <ul>
        <li>At each split, only a random subset of features is considered</li>
        <li>Default: √n features for classification, n/3 for regression (where n = total features)</li>
        <li>Decorrelates trees, reducing correlation between predictions</li>
        <li>Makes the ensemble more robust and diverse</li>
      </ul>

      <h3>How Random Forests Work</h3>
      <ol>
        <li><strong>Bootstrap Sampling:</strong> Create B bootstrap samples from training data</li>
        <li><strong>Train Trees:</strong> For each sample, grow a decision tree:
          <ul>
            <li>At each node, randomly select m features</li>
            <li>Choose best split from these m features</li>
            <li>Grow tree to maximum depth (no pruning)</li>
          </ul>
        </li>
        <li><strong>Aggregate Predictions:</strong>
          <ul>
            <li>Classification: majority vote from all trees</li>
            <li>Regression: average of all tree predictions</li>
          </ul>
        </li>
      </ol>

      <h3>Out-of-Bag (OOB) Error</h3>
      <ul>
        <li>Each tree is trained on ~63% of data; remaining ~37% is "out-of-bag"</li>
        <li>OOB samples serve as validation set for each tree</li>
        <li>Provides unbiased error estimate without separate validation set</li>
        <li>Useful for model selection and hyperparameter tuning</li>
      </ul>

      <h3>Feature Importance</h3>
      <p>Random Forests can measure feature importance by:</p>
      <ul>
        <li><strong>Mean Decrease in Impurity (MDI):</strong> Average reduction in split criterion (Gini/entropy) across all trees</li>
        <li><strong>Mean Decrease in Accuracy (MDA):</strong> Drop in OOB accuracy when feature is permuted</li>
        <li>Helps identify most predictive features for the task</li>
      </ul>

      <h3>Hyperparameters</h3>
      <ul>
        <li><strong>n_estimators:</strong> Number of trees (more is better, but diminishing returns)</li>
        <li><strong>max_features:</strong> Number of features to consider at each split</li>
        <li><strong>max_depth:</strong> Maximum depth of each tree</li>
        <li><strong>min_samples_split:</strong> Minimum samples required to split a node</li>
        <li><strong>min_samples_leaf:</strong> Minimum samples required at leaf node</li>
      </ul>

      <h3>Advantages</h3>
      <ul>
        <li>Reduces overfitting compared to single decision trees</li>
        <li>Handles high-dimensional data well</li>
        <li>Provides feature importance rankings</li>
        <li>Works well with default hyperparameters</li>
        <li>Handles missing values and maintains accuracy</li>
        <li>Not sensitive to outliers</li>
        <li>Can handle both classification and regression</li>
      </ul>

      <h3>Disadvantages</h3>
      <ul>
        <li>Less interpretable than single decision trees</li>
        <li>Slower prediction time (must query all trees)</li>
        <li>Larger model size (stores multiple trees)</li>
        <li>Can overfit noisy datasets with too many trees</li>
        <li>Biased toward features with more categories</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum depth of each tree
    max_features='sqrt',   # Number of features to consider at each split
    min_samples_split=10,  # Minimum samples to split a node
    min_samples_leaf=4,    # Minimum samples at leaf node
    oob_score=True,        # Enable out-of-bag score
    random_state=42
)

rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"OOB Score: {rf.oob_score_:.4f}")  # Out-of-bag error estimate

# Feature importance
feature_importance = rf.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]

print("\\nTop 5 Most Important Features:")
for i in range(5):
    idx = sorted_idx[i]
    print(f"Feature {idx}: {feature_importance[idx]:.4f}")`,
        explanation: 'Demonstrates training a Random Forest classifier with key hyperparameters and extracting feature importances. OOB score provides validation performance without a separate test set.'
      },
      {
        language: 'Python',
        code: `from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Generate regression data
X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'max_features': ['sqrt', 'log2', None]
}

rf_reg = RandomForestRegressor(random_state=42, oob_score=True)
grid_search = GridSearchCV(rf_reg, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Evaluate
y_pred = best_rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\\nTest RMSE: {rmse:.2f}")
print(f"Test R²: {r2:.4f}")
print(f"OOB R²: {best_rf.oob_score_:.4f}")

# Compare individual tree vs forest
from sklearn.tree import DecisionTreeRegressor
single_tree = DecisionTreeRegressor(max_depth=10, random_state=42)
single_tree.fit(X_train, y_train)
y_pred_tree = single_tree.predict(X_test)

print(f"\\nSingle Tree R²: {r2_score(y_test, y_pred_tree):.4f}")
print(f"Random Forest R²: {r2:.4f}")
print("Random Forest significantly reduces overfitting!")`,
        explanation: 'Shows Random Forest for regression with hyperparameter tuning via GridSearchCV. Compares performance against a single decision tree, demonstrating variance reduction.'
      }
    ],
    interviewQuestions: [
      {
        question: 'How does Random Forest reduce overfitting compared to a single decision tree?',
        answer: 'Random Forest reduces overfitting through the principle of **ensemble averaging**, which decreases variance without substantially increasing bias. A single decision tree, when grown deep, perfectly memorizes training data (high variance, low bias). It creates overly complex decision boundaries based on noise, leading to poor generalization. Random Forest trains many such trees (typically 100-500) on different bootstrap samples of the data, then averages their predictions. Since each tree overfits differently (they see different data subsets), their errors are partially uncorrelated, and averaging cancels out much of the noise while retaining the signal.\n\nThe key insight comes from the bias-variance decomposition: Error = Bias² + Variance + Irreducible Error. Individual deep trees have low bias (can model complex patterns) but high variance (predictions change dramatically with small data changes). Averaging many diverse models reduces variance without increasing bias, assuming the models are reasonably decorrelated. If you average N independent models each with variance σ², the ensemble has variance σ²/N. In practice, trees aren\'t fully independent (correlation ≈ 0.3-0.7), but you still get substantial variance reduction. This is why Random Forest can use fully grown trees (no pruning) without overfitting—the ensemble mechanism provides regularization.\n\n**Bootstrap sampling** (bagging) and **feature randomness** ensure tree diversity. Each tree sees only ~63% of unique training samples (bootstrap with replacement), and at each split, only considers a random subset of features (√n for classification, n/3 for regression). This prevents all trees from making similar mistakes based on the same strong features. Trees are forced to discover alternative splits and patterns, increasing their decorrelation. Even if one tree makes a bizarre split due to noise, the other 99+ trees are unlikely to make the same mistake, so the ensemble prediction remains sensible. The result is a model that maintains the flexibility of deep trees (low bias) while achieving much better generalization than any single tree (lower variance).'
      },
      {
        question: 'What is the difference between bagging and Random Forest?',
        answer: 'Both bagging and Random Forest are ensemble methods that train multiple models on bootstrap samples and average predictions, but Random Forest adds an additional layer of randomization. **Bagging** (Bootstrap Aggregating) trains each model on a random sample (with replacement) of the training data—typically sampling N examples from N available, which includes ~63% unique samples and ~37% duplicates. Each model (often decision trees) has access to all features when making split decisions. The final prediction aggregates across models: voting for classification, averaging for regression. Bagging can be applied to any base model (trees, neural networks, etc.).\n\n**Random Forest** is bagging specifically for decision trees with added **feature randomness**. At each split point in each tree, the algorithm randomly selects a subset of features to consider (typically √n for classification, n/3 for regression, where n is total features). The best split is chosen only from this random subset, not all available features. This additional randomness significantly decorrelates trees. In bagging, all trees might use the same strong feature for their top splits, creating correlation between trees and limiting ensemble benefits. Random Forest forces trees to explore different features, discovering alternative but still informative splits, which increases diversity.\n\nThe impact is substantial: Random Forest typically achieves better performance than plain bagging on the same base learner (decision trees), especially when there are strong dominant features. With feature randomness, if one or two features are very predictive, they won\'t dominate every tree—other features get opportunities to contribute, revealing interactions and patterns that would be masked by always splitting on the strongest feature first. The trade-off is that individual Random Forest trees are typically weaker than bagged trees (higher bias) because they\'re constrained to suboptimal splits when the best feature isn\'t in the random subset. However, the increased diversity more than compensates, resulting in a stronger ensemble overall. In practice, when people say "bagging" they often implicitly mean bagging decision trees without feature randomness, while "Random Forest" always includes both bootstrap sampling and feature randomness.'
      },
      {
        question: 'Explain the concept of Out-of-Bag (OOB) error and its usefulness.',
        answer: 'Out-of-Bag (OOB) error is a built-in validation technique for Random Forest that provides an unbiased estimate of test performance without requiring a separate validation set. When training each tree on a bootstrap sample (sampling N examples with replacement from N available), approximately 37% of training examples are left out of that tree\'s sample (the "out-of-bag" samples). These OOB samples weren\'t used to train that particular tree, so they serve as a held-out test set for that tree. For each training example, you can identify which trees didn\'t see it during training (typically ~37% of all trees, or ~37 trees if n_estimators=100), use those trees to predict it, and compare predictions to true labels.\n\nThe OOB error is computed by aggregating these predictions: for each sample, average predictions from trees that didn\'t train on it, then calculate error across all samples. This is essentially performing cross-validation automatically during training without any extra computational cost beyond tracking which samples were OOB for which trees. The OOB error closely approximates test set error and is often nearly identical to k-fold cross-validation results, providing a reliable performance estimate "for free." In scikit-learn, set oob_score=True and access via model.oob_score_ after training.\n\nThe usefulness is significant in several scenarios. First, **when data is limited**, you don\'t need to sacrifice 20-30% for a validation set—you can use all data for training and still get reliable validation error via OOB. Second, for **hyperparameter tuning**, OOB error can guide decisions without separate validation: try different n_estimators, max_depth, or min_samples_split values and compare OOB scores. Third, for **monitoring convergence**, you can track OOB error as trees are added to see when performance plateaus. Fourth, for **feature selection**, compute OOB error, remove a feature, retrain, and compare OOB errors to assess feature importance. The main limitation is that OOB error is specific to Random Forest and bagging methods—it doesn\'t apply to gradient boosting (which doesn\'t use bootstrap sampling) or other model types. But for Random Forest, it\'s an elegant solution that effectively gives you cross-validation without the computational cost.'
      },
      {
        question: 'How does Random Forest calculate feature importance?',
        answer: 'Random Forest calculates feature importance by aggregating importance scores across all trees in the ensemble. For each tree, importance is measured by how much each feature reduces impurity (Gini or entropy) across all nodes where that feature is used for splitting, weighted by the number of samples affected. These per-tree importance scores are then averaged across all trees in the forest. Features that consistently enable good splits across many trees receive high importance; features that are rarely used or don\'t improve splits receive low importance. Scores are normalized to sum to 1, so each feature gets a proportion of total importance.\n\nThis approach improves upon single-tree importance in several ways. First, it\'s much more **stable**—while importance in a single tree can change drastically with small data perturbations, averaging across 100+ trees trained on different bootstrap samples smooths out the variance. A feature that\'s spuriously important in one tree due to noise will likely have low importance in most other trees. Second, Random Forest importance captures features that are important in **different contexts**—one tree might split on feature A in certain regions of the feature space, while another tree uses feature B in similar regions, revealing that both are informative. The ensemble perspective gives a more complete picture of feature utility.\n\nHowever, limitations remain. The **bias toward high-cardinality features** persists: features with many unique values (continuous variables, IDs) still get inflated importance because they offer more split possibilities. With correlated features, one will often be chosen more frequently than the other, receiving higher importance even though they\'re equally informative—though Random Forest\'s feature randomness mitigates this somewhat compared to single trees. For more robust importance estimates, scikit-learn offers **permutation importance**: shuffle each feature\'s values and measure the decrease in accuracy on OOB samples. This approach is slower but less biased by cardinality and correlation. Modern alternatives include SHAP (SHapley Additive exPlanations) values, which provide theoretically grounded importance measures based on game theory. In practice, use Random Forest\'s built-in importance for quick insights and computational efficiency, but verify with permutation importance or SHAP for critical decisions or when dealing with correlated features.'
      },
      {
        question: 'Why do we use a random subset of features at each split?',
        answer: 'Using a random subset of features at each split (feature bagging or feature randomness) is the key innovation that distinguishes Random Forest from standard bagging and dramatically improves performance. The primary purpose is **decorrelating trees**. In ordinary bagging, if one or two features are very strong predictors, all trees will likely use them for top splits, creating highly correlated trees that make similar predictions and errors. When trees are correlated, averaging provides less variance reduction: if all trees overfit in similar ways, their average still overfits. The variance reduction formula is σ²_ensemble = ρσ² + (1-ρ)σ²/N, where ρ is correlation between trees. High correlation limits variance reduction even with many trees.\n\nBy randomly restricting feature availability at each split, you **force tree diversity**. Sometimes the strongest predictor won\'t be in the random subset, so the tree must use the second-best or third-best feature. This creates different tree structures—one tree might split early on feature A, another on feature B. These alternative splits still capture predictive patterns (that\'s why they\'re second-best), but lead to different decision paths and different overfitting patterns. Since the errors are more independent, averaging is more effective. It\'s somewhat counterintuitive: making each tree worse individually (by restricting their choices) makes the ensemble better collectively.\n\nFeature randomness also provides implicit **feature selection** and handles **correlated features**. If you have many redundant features (e.g., temperature in Celsius, Fahrenheit, and Kelvin), they won\'t all dominate every tree—different trees use different versions, and the ensemble learns they\'re all useful. This reveals which features genuinely contribute predictive power beyond the most obvious ones. The typical values (√n for classification, n/3 for regression) balance diversity vs. individual tree quality. Too few features (e.g., 1-2 out of 100) makes trees too weak and random; too many (e.g., 90 out of 100) reduces diversity benefit. The √n heuristic works well empirically, though it\'s worth tuning max_features as a hyperparameter. Feature randomness is why Random Forest often outperforms gradient boosting when you have many correlated features—boosting can get stuck repeatedly using the same strong feature, while Random Forest explores alternatives.'
      },
      {
        question: 'What are the key hyperparameters to tune in Random Forests?',
        answer: 'The most important hyperparameter is **n_estimators** (number of trees). More trees always improve or maintain performance (never hurt by averaging more models), reduce variance, and make predictions more stable. However, returns diminish after a point—going from 10 to 100 trees helps a lot, but 500 to 1000 provides marginal gains. The trade-off is training and prediction time, which scale linearly with number of trees. Start with 100 trees (scikit-learn default), check learning curves, and increase to 200-500 if computational resources allow. Unlike gradient boosting, you can\'t overfit by adding more trees in Random Forest, so "more is better" up to your time/memory budget.\n\n**max_features** controls how many features to consider at each split (the feature randomness). For classification, √n is default; for regression, n/3. Smaller values increase tree diversity (decorrelation) but may make individual trees too weak. Larger values improve individual tree quality but reduce ensemble diversity. Tune this especially if your features are highly correlated or if there are a few dominant predictors—reducing max_features (e.g., from √n to log₂(n)) can help by forcing greater diversity. Conversely, if your features are all weakly predictive, increasing max_features may help.\n\n**Tree-specific parameters** control individual tree complexity. **max_depth** limits how deep trees grow; None (unlimited, default) works well for Random Forest since ensemble averaging prevents overfitting, but limiting depth (10-30) speeds up training and prediction. **min_samples_split** (default 2) and **min_samples_leaf** (default 1) control when to stop splitting; increasing these (e.g., 10-20 for split, 5-10 for leaf) creates simpler trees, which may help with very noisy data. **max_leaf_nodes** directly caps tree size, providing another way to control complexity. Generally, Random Forest is relatively insensitive to these parameters (can use fully grown trees without problems), unlike single decision trees.\n\nSecondary parameters include **max_samples** (bootstrap sample size, defaults to n), **min_impurity_decrease** (minimum improvement required to split), and **class_weight** (for imbalanced classification). For parallel training, set **n_jobs=-1** to use all CPU cores, providing significant speedups. **oob_score=True** computes out-of-bag error for validation without a separate test set. In practice, prioritize tuning n_estimators and max_features via grid search or random search, using OOB score or cross-validation for evaluation. Random Forest is generally robust to hyperparameters—default settings often work well, making it a low-maintenance model compared to gradient boosting or neural networks.'
      }
    ],
    quizQuestions: [
      {
        id: 'rf-q1',
        question: 'What is the main difference between bagging and Random Forest?',
        options: [
          'Random Forest uses boosting while bagging does not',
          'Random Forest considers only a random subset of features at each split',
          'Random Forest can only be used for classification',
          'Random Forest trains trees sequentially while bagging trains in parallel'
        ],
        correctAnswer: 1,
        explanation: 'Random Forest extends bagging by adding feature randomness: at each split, only a random subset of features is considered. This decorrelates the trees, making the ensemble more robust.'
      },
      {
        id: 'rf-q2',
        question: 'You train a Random Forest with 100 trees and n_samples=1000. Approximately how many unique training samples does each tree see?',
        options: [
          '1000 samples (all of them)',
          '~630 unique samples (~63%)',
          '500 samples (half)',
          '100 samples (1/10th)'
        ],
        correctAnswer: 1,
        explanation: 'Bootstrap sampling with replacement means each tree sees ~63% unique samples. The remaining ~37% are out-of-bag samples used for validation.'
      },
      {
        id: 'rf-q3',
        question: 'Your Random Forest achieves 98% training accuracy but only 75% test accuracy. What is the most likely issue?',
        options: [
          'The number of trees is too low',
          'The trees are too deep and overfitting',
          'Feature randomness is too high',
          'The model is underfitting'
        ],
        correctAnswer: 1,
        explanation: 'Large gap between training and test accuracy indicates overfitting. Trees that are too deep (or no max_depth limit) can memorize training data. Solutions: reduce max_depth, increase min_samples_split/leaf.'
      }
    ]
  },

  'gradient-boosting': {
    id: 'gradient-boosting',
    title: 'Gradient Boosting (XGBoost, LightGBM)',
    category: 'classical-ml',
    description: 'Sequential ensemble method that builds trees to correct previous errors',
    content: `
      <h2>Gradient Boosting</h2>
      <p>Gradient Boosting is a powerful ensemble technique that builds models sequentially, where each new model corrects errors made by previous models. Unlike Random Forest (parallel ensemble), boosting creates a strong learner from weak learners iteratively.</p>

      <h3>Core Concept</h3>
      <p>Key idea: Train models sequentially, each focusing on the mistakes of previous models.</p>
      <ul>
        <li><strong>Weak learners:</strong> Typically shallow decision trees (depth 3-6)</li>
        <li><strong>Residual learning:</strong> Each tree predicts the residuals (errors) of previous predictions</li>
        <li><strong>Additive model:</strong> Final prediction = sum of all tree predictions × learning rate</li>
      </ul>

      <h3>Algorithm</h3>
      <ol>
        <li><strong>Initialize:</strong> Start with a simple prediction (e.g., mean for regression)</li>
        <li><strong>For each iteration m = 1 to M:</strong>
          <ul>
            <li>Compute residuals: r = y - ŷ (current predictions)</li>
            <li>Train tree h_m on residuals r</li>
            <li>Update predictions: ŷ = ŷ + η × h_m(x)  (η = learning rate)</li>
          </ul>
        </li>
        <li><strong>Final model:</strong> ŷ = f₀(x) + η × Σ h_m(x)</li>
      </ol>

      <h3>Popular Implementations</h3>

      <h4>XGBoost (Extreme Gradient Boosting)</h4>
      <ul>
        <li>Highly optimized implementation with parallel tree construction</li>
        <li>Built-in regularization (L1/L2) to prevent overfitting</li>
        <li>Handles missing values automatically</li>
        <li>Tree pruning using max_depth (backward pruning)</li>
        <li>Supports custom objective functions and evaluation metrics</li>
        <li>Hardware optimization (CPU cache awareness, parallel processing)</li>
      </ul>

      <h4>LightGBM (Light Gradient Boosting Machine)</h4>
      <ul>
        <li>Faster training on large datasets</li>
        <li>Leaf-wise tree growth (vs level-wise in XGBoost) for deeper, more accurate trees</li>
        <li>Gradient-based One-Side Sampling (GOSS) for efficient sampling</li>
        <li>Exclusive Feature Bundling (EFB) to reduce feature dimensionality</li>
        <li>Better for high-dimensional sparse data</li>
        <li>Lower memory usage</li>
      </ul>

      <h4>CatBoost</h4>
      <ul>
        <li>Excellent handling of categorical features (no manual encoding needed)</li>
        <li>Ordered boosting to prevent target leakage</li>
        <li>Robust to hyperparameter changes</li>
        <li>Good out-of-the-box performance</li>
      </ul>

      <h3>Key Hyperparameters</h3>
      <ul>
        <li><strong>n_estimators:</strong> Number of boosting rounds/trees (50-1000+)</li>
        <li><strong>learning_rate:</strong> Step size shrinkage (0.01-0.3) - lower = slower but better</li>
        <li><strong>max_depth:</strong> Maximum tree depth (3-10) - shallow trees prevent overfitting</li>
        <li><strong>subsample:</strong> Fraction of samples for each tree (0.5-1.0)</li>
        <li><strong>colsample_bytree:</strong> Fraction of features per tree (0.5-1.0)</li>
        <li><strong>min_child_weight:</strong> Minimum sum of instance weight in a leaf</li>
        <li><strong>reg_lambda (L2):</strong> L2 regularization term on weights</li>
        <li><strong>reg_alpha (L1):</strong> L1 regularization term on weights</li>
      </ul>

      <h3>Advantages</h3>
      <ul>
        <li>Excellent predictive performance (often wins Kaggle competitions)</li>
        <li>Handles mixed data types (numerical + categorical)</li>
        <li>Built-in handling of missing values</li>
        <li>Feature importance rankings</li>
        <li>Robust to outliers</li>
        <li>Less feature engineering required</li>
      </ul>

      <h3>Disadvantages</h3>
      <ul>
        <li>Prone to overfitting if not tuned properly</li>
        <li>Sensitive to hyperparameters (requires careful tuning)</li>
        <li>Sequential training (slower than Random Forest)</li>
        <li>Less interpretable than single trees</li>
        <li>Can overfit noisy data</li>
        <li>Not ideal for high-cardinality categorical features (except CatBoost)</li>
      </ul>

      <h3>Boosting vs Bagging</h3>
      <table>
        <tr><th>Aspect</th><th>Boosting (GBM)</th><th>Bagging (Random Forest)</th></tr>
        <tr><td>Training</td><td>Sequential</td><td>Parallel</td></tr>
        <tr><td>Focus</td><td>Corrects previous errors</td><td>Reduces variance</td></tr>
        <tr><td>Tree depth</td><td>Shallow (3-6)</td><td>Deep/unpruned</td></tr>
        <tr><td>Overfitting risk</td><td>Higher</td><td>Lower</td></tr>
        <tr><td>Performance</td><td>Often better</td><td>Good baseline</td></tr>
        <tr><td>Training speed</td><td>Slower</td><td>Faster</td></tr>
      </table>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

# Generate imbalanced dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15,
                           weights=[0.9, 0.1], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                      stratify=y, random_state=42)

# Train XGBoost classifier
xgb = XGBClassifier(
    n_estimators=100,           # Number of boosting rounds
    learning_rate=0.1,          # Step size shrinkage
    max_depth=5,                # Maximum tree depth
    subsample=0.8,              # Subsample ratio of training data
    colsample_bytree=0.8,       # Subsample ratio of features
    reg_lambda=1.0,             # L2 regularization
    reg_alpha=0.1,              # L1 regularization
    scale_pos_weight=9,         # Balance class weights (ratio of negative to positive)
    eval_metric='auc',
    random_state=42
)

# Fit with early stopping
xgb.fit(X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
        verbose=False)

# Predictions
y_pred = xgb.predict(X_test)
y_proba = xgb.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Feature importance
feature_importance = xgb.feature_importances_
top_features = np.argsort(feature_importance)[::-1][:5]
print("\\nTop 5 Features:")
for idx in top_features:
    print(f"Feature {idx}: {feature_importance[idx]:.4f}")`,
        explanation: 'XGBoost classification with key hyperparameters for imbalanced data. Uses scale_pos_weight to handle class imbalance and early stopping to prevent overfitting.'
      },
      {
        language: 'Python',
        code: `from lightgbm import LGBMRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Generate regression data
X, y = make_regression(n_samples=5000, n_features=50, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# LightGBM regressor
lgbm = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,              # LightGBM uses leaf-wise growth
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.5,
    random_state=42,
    verbose=-1
)

# Train with early stopping
lgbm.fit(X_train, y_train,
         eval_set=[(X_val, y_val)],
         eval_metric='rmse',
         callbacks=[early_stopping(stopping_rounds=20)])

# Predictions
y_pred = lgbm.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.2f}")
print(f"Test R²: {r2:.4f}")
print(f"Best iteration: {lgbm.best_iteration_}")

# Cross-validation
cv_scores = cross_val_score(lgbm, X_train, y_train, cv=5,
                             scoring='r2', n_jobs=-1)
print(f"\\nCV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")`,
        explanation: 'LightGBM regression demonstrating leaf-wise tree growth and early stopping. Shows how to use validation set for monitoring and preventing overfitting during training.'
      }
    ],
    interviewQuestions: [
      {
        question: 'How does gradient boosting differ from bagging (Random Forest)?',
        answer: 'The fundamental difference is **sequential vs. parallel** ensemble building. Random Forest (bagging) trains all trees independently and in parallel on bootstrap samples, then averages their predictions. Each tree is trained on random data and makes predictions independently without knowledge of other trees. Gradient boosting trains trees sequentially, where each new tree explicitly tries to correct the errors of the ensemble built so far. Tree t+1 is trained to predict the residuals (errors) of trees 1 through t, gradually improving the ensemble by focusing on examples where previous trees performed poorly.\n\nThis creates different optimization approaches and bias-variance trade-offs. Random Forest reduces variance through averaging many high-variance, low-bias models (deep trees). Each tree overfits differently, and averaging cancels noise. Gradient boosting reduces bias through additive learning—it starts with a simple model (high bias, low variance) and incrementally adds complexity by fitting residuals. Each shallow tree (weak learner) contributes a small improvement, and the sum of many small improvements yields a strong learner. Random Forest\'s final prediction is an average: ŷ = (1/N)Σf_i(x). Gradient boosting is a sum: ŷ = f_0(x) + η·f_1(x) + η·f_2(x) + ... + η·f_M(x), where η is the learning rate.\n\nThese differences have practical implications. Random Forest is more robust to hyperparameters (hard to overfit with more trees), parallelizes perfectly (each tree trains independently), and is less sensitive to outliers. Gradient boosting typically achieves better accuracy with careful tuning but is easier to overfit (more trees can hurt if not using early stopping), trains sequentially (slower), and is more sensitive to noisy data and outliers (early trees\' errors propagate). Random Forest works well out-of-the-box with minimal tuning. Gradient boosting requires careful hyperparameter tuning (learning rate, max depth, number of estimators) but can squeeze out higher performance. In Kaggle competitions, gradient boosting (XGBoost, LightGBM, CatBoost) dominates on structured/tabular data due to its superior accuracy when properly tuned, while Random Forest provides a strong, reliable baseline with less effort.'
      },
      {
        question: 'What is the role of the learning rate in gradient boosting?',
        answer: 'The learning rate (also called shrinkage, typically denoted η or lr) controls how much each tree contributes to the final ensemble. After training a tree that predicts residuals, we don\'t add its full predictions—instead, we add η times its predictions, where η is typically between 0.01 and 0.3. The formula is: ŷ = f_0 + η·f_1 + η·f_2 + ... + η·f_M. With η = 0.1, each tree contributes only 10% of what it could, making incremental, conservative updates. This is the "learning rate" because it controls how fast the ensemble learns (adapts to training data).\n\nLower learning rates (0.01-0.1) generally produce better models because they require more trees to achieve the same training fit, and adding more small contributions is a form of regularization. Instead of one tree making a large correction that might overfit, many trees make small corrections that average out noise. Think of it like gradient descent with a small step size—you\'re less likely to overshoot the optimum. The trade-off is computational cost: η = 0.01 might require 5000 trees to reach the same training error as η = 0.1 with 500 trees. Training time scales with the number of trees, so smaller learning rates mean longer training.\n\nThe learning rate interacts strongly with n_estimators (number of trees). A common strategy is: use a small learning rate (0.01-0.05) with many trees (1000-5000) and early stopping to automatically determine when to stop. This gives the best performance. For faster experimentation, use a larger learning rate (0.1-0.3) with fewer trees (100-500). The learning rate also provides regularization—smaller values prevent overfitting even with many trees, while larger values can overfit if n_estimators is too high. In practice, tune learning rate and n_estimators together: start with lr=0.1 and n_estimators=100, check performance, then decrease learning rate and increase trees, using cross-validation or a holdout set to monitor validation error. Modern implementations (XGBoost, LightGBM) support early stopping, which automatically finds the optimal number of trees for a given learning rate, making the process more automated. Learning rate is one of the most important hyperparameters in gradient boosting, often having the largest impact on final performance.'
      },
      {
        question: 'Why are shallow trees (low max_depth) preferred in gradient boosting?',
        answer: 'Gradient boosting works best with shallow trees (typically max_depth = 3-6, called "weak learners") because the sequential additive nature of boosting provides complexity through the ensemble, not through individual trees. Each shallow tree captures a simple pattern or interaction (e.g., "if income > $50k and age < 30, residual is +$5k"), and hundreds or thousands of such trees combine to model complex relationships. Deep individual trees would capture too much complexity at once, leading to overfitting—early trees would fit the data (including noise) too well, and subsequent trees would fit residuals of an already overfit model, amplifying noise rather than correcting signal.\n\nShallow trees also directly address the bias-variance tradeoff in the context of boosting. Boosting reduces bias by sequentially adding models that correct errors, but it can increase variance if base learners are too complex. A deep tree (max_depth = 20) has low bias but high variance—it overfits its training data. In bagging (Random Forest), averaging many high-variance trees cancels out the variance. In boosting, you\'re sequentially fitting to errors, which amplifies variance if trees are too complex. Shallow trees have higher bias (under fit individually) but lower variance, making them ideal for boosting to iteratively reduce bias while keeping variance manageable.\n\nInteraction depth is another key consideration. A tree with max_depth = d can capture up to d-way feature interactions. max_depth = 3 captures up to 3-way interactions (e.g., "effect of feature A depends on features B and C"), which is often sufficient for most problems. Going deeper (max_depth = 10+) captures higher-order interactions that are rarely present in real data and often represent noise. Friedman recommended max_depth = 3-6 as a sweet spot that captures meaningful interactions without overfitting. In practice, typical values are: max_depth = 3 for small datasets or simple problems, 5-6 for medium complexity, up to 8-10 for large datasets with many features. Contrast this with Random Forest, which uses deep trees (often unlimited depth) because averaging decorrelates the trees and prevents overfitting. In boosting, trees are highly correlated by design (each fits the previous ensemble\'s errors), so depth must be limited. Empirically, gradient boosting with shallow trees consistently outperforms boosting with deep trees, making it a fundamental principle of the algorithm.'
      },
      {
        question: 'Explain how XGBoost handles missing values.',
        answer: 'XGBoost has native missing value support that automatically learns the optimal direction to send missing values at each split, treating missingness as informative. During training, when evaluating a split on a feature, XGBoost tries three options: (1) send all non-missing values to the optimal child as usual, with missing values going left; (2) same but missing values going right; (3) in some implementations, also consider missing values as a separate category. It calculates the gain for each option and chooses the direction that maximizes the split objective, then stores this "default direction" with the split node.\n\nThis approach has several advantages over traditional imputation. First, it\'s **data-driven**—the algorithm learns where missing values should go based on what improves predictions, rather than using a fixed rule like "impute with mean." If missingness correlates with the target (e.g., income is often missing for low earners), XGBoost discovers this pattern and groups missing values accordingly. Second, it\'s **efficient**—no preprocessing required, and evaluation is fast since missing values are handled during tree construction. Third, it **captures missingness as signal**—if missing values tend to be similar to low values of a feature, they\'ll be sent to the left child; if similar to high values, they\'ll go right.\n\nThe technical implementation uses **sparsity-aware split finding**. XGBoost stores data in a sparse format and iterates only over non-missing values when computing split statistics (histograms of gradients). For each candidate split, it computes gain as if all missing values went left, then as if they went right, choosing the better option. This is much faster than explicitly imputing values, especially for high-dimensional sparse data common in applications like recommendation systems or NLP. At prediction time, when traversing a tree, if a feature value is missing, the sample follows the default direction learned during training. This native handling is a major reason XGBoost works so well in practice—real-world data often has missing values, and XGBoost handles them seamlessly without manual intervention. However, for features where missingness is truly random and uninformative, explicit imputation (mean, median, forward fill) might still help by reducing data sparsity and allowing the tree to make splits based on imputed values rather than just routing missing values. In practice, XGBoost\'s native handling works well most of the time, but it\'s worth experimenting with imputation as an alternative preprocessing step for features with high missingness rates.'
      },
      {
        question: 'What is the difference between XGBoost and LightGBM?',
        answer: 'Both are highly optimized gradient boosting implementations, but they differ in tree-building algorithms, speed, and memory efficiency. **XGBoost** uses a **level-wise (depth-wise) tree growth** strategy: it grows all nodes at the same level before moving to the next level. For max_depth = 3, it first splits the root, then all 2 nodes at level 1, then all 4 nodes at level 2. This creates balanced trees but can be inefficient—it might split nodes with small gain just to maintain level-wise structure. **LightGBM** uses a **leaf-wise (best-first) growth** strategy: it always splits the leaf with maximum gain, regardless of level. This creates unbalanced trees but typically achieves better accuracy with fewer leaves.\n\nLightGBM is generally **faster and more memory-efficient**, especially on large datasets (>10k samples, >100 features). It uses histogram-based algorithms: instead of evaluating every possible split point (XGBoost\'s exact method), it bins continuous features into discrete bins (typically 255) and computes histograms of gradients. This is much faster—O(#bins × #features) instead of O(#data × #features)—and uses less memory. LightGBM also supports **categorical features natively** without one-hot encoding, using optimal split-finding algorithms for categories, which can significantly improve both speed and accuracy. XGBoost requires one-hot or label encoding for categoricals.\n\nXGBoost has better support for **regularization**—its loss function explicitly includes L1 (alpha) and L2 (lambda) penalties on leaf weights, which helps prevent overfitting. It also has more mature ecosystem, better documentation, and built-in support for various objective functions and metrics. LightGBM counters overfitting through its min_data_in_leaf parameter (typically set higher than XGBoost\'s min_child_weight) and max_depth, but has fewer explicit regularization levers. In terms of accuracy, LightGBM often performs slightly better on large datasets due to its leaf-wise growth, while XGBoost may be preferable on smaller datasets where overfitting is a greater concern.\n\n**When to choose which**: Use LightGBM for large datasets (>10k rows), when training speed matters, when you have categorical features, or when memory is limited. Use XGBoost for smaller datasets, when you want more conservative (less prone to overfitting) default behavior, or when you need specific features like monotonic constraints or interaction constraints. In practice, try both—they have similar APIs and often produce comparable results, but one may work better for your specific problem. CatBoost is a third option that excels with categorical features and small datasets, using ordered boosting to reduce overfitting. Modern practitioners often try all three implementations (XGBoost, LightGBM, CatBoost) via hyperparameter tuning and select the best performer.'
      },
      {
        question: 'How does early stopping work in gradient boosting, and why is it important?',
        answer: 'Early stopping monitors validation error during training and stops adding trees when performance stops improving, preventing overfitting and saving computation. The process: split data into train and validation sets, train trees sequentially on the train set, evaluate ensemble performance on validation set after adding each tree, and stop if validation error hasn\'t improved for N consecutive trees (N is the "patience" or "early_stopping_rounds" parameter, typically 50-100). The final model uses only the trees up to the best validation score, discarding later trees that didn\'t help.\n\nThis is crucial because gradient boosting can overfit with too many trees. Unlike Random Forest (where more trees always help or don\'t hurt), boosting sequentially fits residuals, and after a certain point, new trees start fitting noise in the training data rather than signal. Training error keeps decreasing (trees keep reducing residuals), but validation error increases (overfitting). Early stopping automatically finds the optimal number of trees without manual tuning, balancing underfitting (too few trees) and overfitting (too many trees). It\'s essentially cross-validation built into the training process.\n\nThe typical workflow: set n_estimators to a large value (e.g., 5000), use early_stopping_rounds=50, train on training set, evaluate on validation set, and stop when validation metric hasn\'t improved for 50 trees. XGBoost and LightGBM both support this via an eval_set parameter and early_stopping_rounds. For example: `model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)`. The model automatically tracks the best iteration and can revert to it. This is much faster than grid searching over n_estimators—instead of training separate models with 100, 200, 300, ... trees, you train once with early stopping.\n\nEarly stopping interacts with learning rate: smaller learning rates (0.01-0.05) require more trees to converge, so use larger n_estimators (5000+) and rely on early stopping. Larger learning rates (0.1-0.3) converge faster but may overfit sooner, needing fewer trees. Using early stopping with a validation set is best practice for gradient boosting—it prevents overfitting, reduces training time, and eliminates the need to manually tune n_estimators. The main caution: ensure your validation set is representative and not too small (<1000 samples can give noisy validation metrics), and use stratified splits for classification to maintain class balance. Early stopping is one reason gradient boosting works so well in practice—it provides automatic regularization that\'s hard to achieve manually through other hyperparameters alone.'
      }
    ],
    quizQuestions: [
      {
        id: 'gb-q1',
        question: 'In gradient boosting, what does each subsequent tree learn?',
        options: [
          'The same patterns as the first tree',
          'Random patterns from the data',
          'The residual errors of previous trees',
          'A completely independent representation of the data'
        ],
        correctAnswer: 2,
        explanation: 'Each tree in gradient boosting is trained to predict the residuals (errors) of the cumulative predictions from all previous trees, progressively reducing the overall error.'
      },
      {
        id: 'gb-q2',
        question: 'You observe that your XGBoost model achieves 95% training accuracy but only 70% test accuracy. Which hyperparameter change is MOST likely to help?',
        options: [
          'Increase n_estimators',
          'Increase learning_rate',
          'Decrease max_depth or increase reg_lambda',
          'Decrease subsample ratio'
        ],
        correctAnswer: 2,
        explanation: 'The large gap indicates overfitting. Reducing max_depth makes trees less complex, and increasing reg_lambda adds L2 regularization. Both help prevent overfitting by limiting model complexity.'
      },
      {
        id: 'gb-q3',
        question: 'What is the main advantage of LightGBM over XGBoost?',
        options: [
          'Better handling of categorical features',
          'Faster training on large datasets with lower memory usage',
          'More accurate on small datasets',
          'Simpler hyperparameter tuning'
        ],
        correctAnswer: 1,
        explanation: 'LightGBM uses leaf-wise tree growth, GOSS, and EFB techniques that make it faster and more memory-efficient on large datasets compared to XGBoost\'s level-wise approach.'
      }
    ]
  },

  'support-vector-machines': {
    id: 'support-vector-machines',
    title: 'Support Vector Machines (SVM)',
    category: 'classical-ml',
    description: 'Powerful classification algorithm that finds optimal decision boundaries',
    content: `
      <h2>Support Vector Machines (SVM)</h2>
      <p>Support Vector Machines are supervised learning models that find the optimal hyperplane to separate data into classes. SVMs are particularly effective in high-dimensional spaces and for both linear and non-linear classification.</p>

      <h3>Core Concept</h3>
      <p><strong>Goal:</strong> Find the hyperplane that maximizes the margin between classes.</p>
      <ul>
        <li><strong>Hyperplane:</strong> Decision boundary that separates classes (line in 2D, plane in 3D, hyperplane in n-D)</li>
        <li><strong>Support Vectors:</strong> Data points closest to the hyperplane that determine its position</li>
        <li><strong>Margin:</strong> Distance between hyperplane and nearest support vectors from each class</li>
        <li><strong>Maximum Margin:</strong> SVM finds the hyperplane with the largest possible margin</li>
      </ul>

      <h3>Linear SVM</h3>
      <p>For linearly separable data:</p>
      <ul>
        <li>Decision function: f(x) = w·x + b</li>
        <li>Classification: sign(f(x)) gives class label (+1 or -1)</li>
        <li>Margin = 2/||w||, so we minimize ||w|| to maximize margin</li>
        <li>Only support vectors affect the hyperplane position</li>
      </ul>

      <h3>Soft Margin SVM</h3>
      <p>For non-perfectly separable data:</p>
      <ul>
        <li>Introduces slack variables ξᵢ to allow some misclassifications</li>
        <li>C parameter controls trade-off between margin size and violations</li>
        <li><strong>High C:</strong> Smaller margin, fewer violations (risk overfitting)</li>
        <li><strong>Low C:</strong> Larger margin, more violations (more regularization)</li>
        <li>Objective: minimize ||w|| + C·Σξᵢ</li>
      </ul>

      <h3>Kernel Trick</h3>
      <p>For non-linear decision boundaries, map data to higher-dimensional space:</p>

      <h4>Common Kernels</h4>
      <ul>
        <li><strong>Linear:</strong> K(x, x') = x·x' (no transformation)</li>
        <li><strong>Polynomial:</strong> K(x, x') = (γx·x' + r)^d (degree d polynomial)</li>
        <li><strong>RBF (Radial Basis Function/Gaussian):</strong> K(x, x') = exp(-γ||x - x'||²)
          <ul>
            <li>Most popular kernel</li>
            <li>γ controls influence of single training example (low γ = far reach, high γ = close reach)</li>
            <li>Can model complex non-linear boundaries</li>
          </ul>
        </li>
        <li><strong>Sigmoid:</strong> K(x, x') = tanh(γx·x' + r)</li>
      </ul>

      <h3>Hyperparameters</h3>
      <ul>
        <li><strong>C (regularization):</strong> Trade-off between margin and violations (0.1-100)</li>
        <li><strong>kernel:</strong> Type of kernel function (linear, poly, rbf, sigmoid)</li>
        <li><strong>gamma (for RBF/poly):</strong> Kernel coefficient (0.001-10)</li>
        <li><strong>degree (for poly):</strong> Polynomial degree (2-5)</li>
      </ul>

      <h3>Advantages</h3>
      <ul>
        <li>Effective in high-dimensional spaces</li>
        <li>Memory efficient (only stores support vectors)</li>
        <li>Versatile (different kernels for different problems)</li>
        <li>Works well with clear margin of separation</li>
        <li>Less prone to overfitting (especially with high C)</li>
        <li>Robust to outliers (only support vectors matter)</li>
      </ul>

      <h3>Disadvantages</h3>
      <ul>
        <li>Not suitable for large datasets (O(n²) to O(n³) training time)</li>
        <li>Doesn't provide probability estimates directly</li>
        <li>Sensitive to feature scaling</li>
        <li>Difficult to interpret (especially with kernels)</li>
        <li>Requires careful hyperparameter tuning (C, gamma)</li>
        <li>Doesn't perform well with noisy data or overlapping classes</li>
      </ul>

      <h3>SVM for Regression (SVR)</h3>
      <ul>
        <li>Fits as many instances as possible within margin (epsilon tube)</li>
        <li>Only points outside epsilon tube contribute to loss</li>
        <li>Useful for regression tasks with outliers</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `from sklearn import svm
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Linear SVM for linearly separable data
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling is CRUCIAL for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear SVM
linear_svm = svm.SVC(kernel='linear', C=1.0)
linear_svm.fit(X_train_scaled, y_train)

y_pred = linear_svm.predict(X_test_scaled)
print(f"Linear SVM Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Number of support vectors: {len(linear_svm.support_vectors_)}")
print(f"Support vector indices: {linear_svm.support_[:10]}...")  # First 10

# Effect of C parameter
print("\\nEffect of C parameter:")
for C in [0.1, 1.0, 10.0]:
    svm_model = svm.SVC(kernel='linear', C=C)
    svm_model.fit(X_train_scaled, y_train)
    train_acc = svm_model.score(X_train_scaled, y_train)
    test_acc = svm_model.score(X_test_scaled, y_test)
    print(f"C={C}: Train={train_acc:.3f}, Test={test_acc:.3f}, Support vectors={len(svm_model.support_vectors_)}")`,
        explanation: 'Demonstrates linear SVM with feature scaling and the effect of C parameter. Lower C allows more margin violations (more support vectors), higher C enforces stricter classification.'
      },
      {
        language: 'Python',
        code: `# Non-linear SVM with RBF kernel for non-linearly separable data
X_circle, y_circle = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_circle, y_circle, test_size=0.2, random_state=42)

# Scale features
scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

# RBF kernel SVM
rbf_svm = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
rbf_svm.fit(X_train_c_scaled, y_train_c)

y_pred_rbf = rbf_svm.predict(X_test_c_scaled)
print(f"\\nRBF SVM Accuracy: {accuracy_score(y_test_c, y_pred_rbf):.4f}")

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(svm.SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_c_scaled, y_train_c)

print(f"\\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

best_svm = grid_search.best_estimator_
test_score = best_svm.score(X_test_c_scaled, y_test_c)
print(f"Test accuracy with best params: {test_score:.4f}")`,
        explanation: 'Shows RBF kernel SVM for non-linear classification with circular decision boundary. Demonstrates hyperparameter tuning for C and gamma using GridSearchCV.'
      }
    ],
    interviewQuestions: [
      {
        question: 'What is the intuition behind Support Vector Machines?',
        answer: 'Support Vector Machines find the optimal decision boundary (hyperplane) that maximally separates different classes in the feature space. The key intuition is **maximum margin**: among all possible hyperplanes that separate classes, SVM chooses the one with the largest distance (margin) to the nearest data points of any class. This margin represents the model\'s confidence—a wide margin means the decision boundary is far from any training examples, suggesting it will generalize better to new data. The decision boundary is positioned such that it\'s equidistant from the closest points of each class.\n\nFor linearly separable data in 2D, imagine drawing a line between two groups of points. You could draw infinitely many lines that separate them, but intuitively, a line that passes very close to some points seems risky—a slight perturbation might misclassify them. SVM finds the line with maximum clearance on both sides. Mathematically, if the decision boundary is defined by weights w and bias b (w·x + b = 0), the margin is 2/||w||, so maximizing margin is equivalent to minimizing ||w||. The optimization problem becomes: minimize ||w||² subject to all points being correctly classified (y_i(w·x_i + b) ≥ 1 for all i).\n\nThis maximum margin principle provides good generalization through structural risk minimization—by maximizing the margin, SVM minimizes a bound on the generalization error, not just the training error. The model is also **sparse**: only the points closest to the boundary (support vectors) matter; removing far-away points doesn\'t change the solution. This makes SVM elegant and efficient. The approach extends to non-linear boundaries through the kernel trick (implicitly mapping to higher dimensions) and to non-separable data through soft margins (allowing some misclassifications). The core idea remains: find the boundary with maximum separation, which tends to generalize well by not committing too strongly to any particular training point.'
      },
      {
        question: 'Explain the role of support vectors in SVM.',
        answer: 'Support vectors are the training data points that lie closest to the decision boundary—specifically, those points that lie exactly on the margin boundaries (the two parallel hyperplanes on either side of the decision boundary, at distance margin from it). These are the critical data points that define the decision boundary. In the dual formulation of SVM, the decision function is f(x) = Σ(α_i y_i K(x_i, x)) + b, where α_i are the learned weights (Lagrange multipliers). Most α_i are zero; only the support vectors have α_i > 0, meaning only these points contribute to the decision function.\n\nThe term "support vectors" captures their role: they "support" or define the decision boundary. If you removed non-support vectors from the training set and retrained, you\'d get exactly the same decision boundary—they\'re redundant. But if you removed or moved a support vector, the decision boundary would change. This makes SVM a **sparse model**: predictions depend only on a subset of training data (typically 10-50% become support vectors, depending on data complexity and C parameter). This is computationally advantageous for prediction and memory storage—you only need to keep support vectors.\n\nThe number and identity of support vectors provide insights into the problem difficulty and model behavior. A large number of support vectors suggests complex decision boundaries or overlapping classes (difficult problem). Very few support vectors suggests well-separated classes or potential underfitting. Points with α_i = C (at the upper bound) are support vectors that lie within the margin or are misclassified—these are the problematic points that violate the ideal separation. Points with 0 < α_i < C lie exactly on the margin boundaries and are correctly classified. The support vectors thus identify the "difficult" or "boundary" examples that the model must carefully balance, while ignoring easy examples far from the decision boundary. This focus on boundary examples is why SVM works well for problems where most data is easy to classify and only a minority lie in ambiguous regions.'
      },
      {
        question: 'What is the kernel trick and why is it useful?',
        answer: 'The kernel trick allows SVM to learn non-linear decision boundaries efficiently by implicitly mapping data to a higher-dimensional space where it becomes linearly separable, without ever explicitly computing that high-dimensional representation. The trick relies on the fact that SVM\'s dual formulation only requires computing dot products between data points: f(x) = Σ(α_i y_i x_i·x) + b. If we map inputs to a higher dimension via φ(x), we\'d need to compute φ(x_i)·φ(x_j). A kernel function K(x_i, x_j) computes this dot product directly in the higher-dimensional space without explicitly computing φ(x_i) and φ(x_j).\n\nFor example, the polynomial kernel K(x, x\') = (x·x\' + c)^d corresponds to mapping to a space of all polynomial combinations of features up to degree d. For 2D input [x₁, x₂] with degree 2, this implicitly creates features [x₁², x₂², √2x₁x₂, ...] in the transformed space. Computing this mapping explicitly would require creating all these new features (expensive in high dimensions), then computing dot products. The kernel computes the same result by simply evaluating (x·x\' + c)². The RBF (Gaussian) kernel K(x, x\') = exp(-γ||x - x\'||²) corresponds to mapping to an infinite-dimensional space, which would be impossible to compute explicitly.\n\nThe kernel trick is useful because it enables SVM to capture complex non-linear patterns while maintaining computational efficiency. Training remains manageable—you compute O(n²) kernel evaluations for n training points, which is feasible for thousands of points. The kernel matrix (Gram matrix) stores all pairwise kernel computations. This approach is far more efficient than explicitly creating high-dimensional or infinite-dimensional feature spaces. It also provides flexibility: you can swap kernel functions to match your prior knowledge about the problem (polynomial for polynomial boundaries, RBF for smooth curved boundaries, string kernels for text, graph kernels for structured data). The mathematical elegance is that the entire optimization and prediction depends only on dot products in the original space (via the kernel), never requiring explicit feature representation. This insight applies beyond SVM to other algorithms (kernel ridge regression, kernel PCA, kernel k-means), making it a fundamental technique in machine learning.'
      },
      {
        question: 'How does the C parameter affect SVM behavior?',
        answer: 'The C parameter (regularization parameter) controls the trade-off between maximizing the margin and minimizing training errors (misclassifications or margin violations). It appears in the soft-margin SVM formulation: minimize (1/2)||w||² + C·Σξ_i, where ξ_i are slack variables representing margin violations. **Large C** (e.g., 100, 1000) heavily penalizes violations, forcing the model to classify training points correctly even if it means a smaller margin. This leads to a complex decision boundary that closely fits training data (low bias, high variance, prone to overfitting). With very large C, the model approaches hard-margin SVM behavior, insisting on perfect separation if possible.\n\n**Small C** (e.g., 0.01, 0.1) gives low penalty to violations, allowing the model to tolerate more misclassifications in favor of a wider margin. This results in a simpler, smoother decision boundary (high bias, low variance, more regularization). The model generalizes better by not trying too hard to fit every training point perfectly. In the extreme, C→0 would ignore training errors entirely, caring only about maximizing margin. The optimal C depends on data: for separable, clean data, large C works well; for noisy, overlapping classes, small C prevents overfitting.\n\nC interacts with the kernel and its parameters. With RBF kernel, you typically tune both C and gamma together. Large C with large gamma creates very complex boundaries (overfitting risk), while small C with small gamma creates very simple boundaries (underfitting risk). The relationship to regularization in other models: C is inversely related to λ in Ridge regression (C = 1/(2λ)). **Small C = strong regularization, large C = weak regularization**. In practice, tune C via cross-validation, searching log-scale values like [0.01, 0.1, 1, 10, 100]. Signs of poor C: training accuracy >> test accuracy suggests C too large (overfitting); both training and test accuracy low suggests C too small (underfitting). For imbalanced datasets, class_weight parameter adjusts C per class, and you may need different effective C values for minority vs majority classes. SVM\'s performance is quite sensitive to C, making it one of the most important hyperparameters to tune.'
      },
      {
        question: 'What is the difference between hard-margin and soft-margin SVM?',
        answer: '**Hard-margin SVM** assumes data is linearly separable and finds the hyperplane that perfectly separates all training points with maximum margin. The optimization requires all points satisfy y_i(w·x_i + b) ≥ 1 (correctly classified beyond the margin). This is a strict constraint—no violations allowed. Hard-margin SVM has no regularization parameter; it simply maximizes margin subject to perfect separation. It works beautifully on toy datasets where classes are cleanly separated, but fails catastrophically in practice for two reasons: (1) most real-world data isn\'t linearly separable due to class overlap or outliers, making the optimization infeasible; (2) even if separable, a single outlier can drastically reduce the margin, harming generalization.\n\n**Soft-margin SVM** allows violations through slack variables ξ_i, relaxing the constraint to y_i(w·x_i + b) ≥ 1 - ξ_i. Points can be: (a) correctly classified outside the margin (ξ_i = 0, ideal); (b) correctly classified inside the margin (0 < ξ_i < 1, violation but still right side of boundary); (c) misclassified (ξ_i ≥ 1, wrong side of boundary). The objective becomes: minimize (1/2)||w||² + C·Σξ_i, balancing margin maximization with violation minimization. The C parameter controls this trade-off: large C severely penalizes violations (approaches hard-margin), small C tolerates violations for a wider margin (more regularization).\n\nPractical differences: hard-margin SVM is non-robust—a single outlier can force a tiny margin or make the problem infeasible. Soft-margin SVM is robust, treating outliers as acceptable violations rather than letting them dictate the boundary. Hard-margin has no hyperparameters to tune (besides kernel choice); soft-margin requires tuning C. Computationally, hard-margin is a quadratic programming (QP) problem with linear constraints; soft-margin adds slack variables and box constraints (0 ≤ α_i ≤ C in the dual), slightly more complex but still efficiently solvable. In practice, you always use soft-margin SVM—even if data appears separable, using soft-margin with reasonably large C provides robustness to outliers and noise. Hard-margin is primarily of theoretical interest, illustrating the core SVM concept before relaxing assumptions for real-world applicability. The introduction of soft margins (by Cortes and Vapnik, 1995) was crucial for SVM\'s practical success.'
      },
      {
        question: 'When would you use RBF kernel vs linear kernel?',
        answer: 'Use **linear kernel** when you expect a linear relationship between features and target, when interpretability matters, or when you have high-dimensional sparse data (text, genomics). Linear SVM finds the hyperplane w·x + b = 0, where coefficients w are interpretable as feature importances. It\'s computationally efficient, training and predicting faster than non-linear kernels, and scales well to millions of samples (using libraries like LIBLINEAR). For high-dimensional data (n_features >> n_samples), relationships are often approximately linear in the original space, making complex kernels unnecessary and prone to overfitting. Text classification with TF-IDF features (10,000+ dimensions) typically works best with linear SVM.\n\n**RBF kernel** (Gaussian kernel) works when relationships are non-linear, when decision boundaries are complex and curved, or when feature interactions are important. RBF measures similarity via exp(-γ||x - x\'||²), effectively computing an infinite-dimensional feature space where almost any decision boundary is possible. It\'s a universal kernel—with appropriate hyperparameters, it can approximate any continuous function. Use it for low-to-medium dimensional data (< 1000 features) where you suspect non-linear patterns: image features, sensor data, engineered features. RBF requires tuning two hyperparameters (C and gamma), adding complexity but providing flexibility.\n\nPractical guidelines: **Try linear first**, especially for text, sparse data, or when n_features > n_samples. If performance is unsatisfactory, try RBF. Linear SVM\'s simplicity and speed make it an excellent baseline. If linear performs well, stick with it for interpretability and efficiency. Only use RBF if you need the expressiveness and can afford hyperparameter tuning. Check learning curves: if training accuracy is low, the model is underfitting—try RBF. If training accuracy is high but test accuracy is low, overfitting—try linear or increase regularization.\n\n**Feature engineering matters**: sometimes linear SVM on well-engineered features (polynomial features, domain-specific transforms) outperforms RBF on raw features. RBF with poor gamma can either underfit (gamma too small → decision boundary too smooth) or overfit (gamma too large → memorizes training data). For mixed problems, you can also try **polynomial kernel** (captures polynomial interactions explicitly) or **combining kernels** (weighted sum of linear and RBF). In modern practice, tree-based methods (Random Forest, XGBoost) often outperform RBF SVM for non-linear problems on tabular data, relegating SVM to cases where maximum-margin properties are specifically beneficial or where kernel methods provide domain-specific advantages (string kernels, graph kernels).'
      }
    ],
    quizQuestions: [
      {
        id: 'svm-q1',
        question: 'What happens when you increase the C parameter in SVM?',
        options: [
          'Larger margin, more misclassifications allowed',
          'Smaller margin, fewer misclassifications allowed',
          'No effect on the model',
          'Kernel changes automatically'
        ],
        correctAnswer: 1,
        explanation: 'Higher C increases the penalty for misclassifications, leading to a smaller margin but stricter classification. This can lead to overfitting. Lower C allows more margin violations for a larger, more generalized margin.'
      },
      {
        id: 'svm-q2',
        question: 'You have a dataset with 100,000 samples and need real-time predictions. Which algorithm is likely better than SVM?',
        options: [
          'Use SVM, it always performs best',
          'Logistic Regression or Random Forest (faster training and prediction)',
          'Increase C parameter in SVM',
          'Use polynomial kernel in SVM'
        ],
        correctAnswer: 1,
        explanation: 'SVM has O(n²) to O(n³) training complexity and prediction requires computing kernel with all support vectors. For large datasets requiring real-time prediction, Logistic Regression or tree-based methods are typically better choices.'
      },
      {
        id: 'svm-q3',
        question: 'Your linear SVM achieves 95% training accuracy but only 60% test accuracy. What should you do?',
        options: [
          'Increase C to fit training data better',
          'Decrease C or use regularization to reduce overfitting',
          'Switch to polynomial kernel with high degree',
          'Remove feature scaling'
        ],
        correctAnswer: 1,
        explanation: 'Large gap between training and test accuracy indicates overfitting. Decreasing C allows more margin violations, creating a simpler model with better generalization.'
      }
    ]
  },

  'k-nearest-neighbors': {
    id: 'k-nearest-neighbors',
    title: 'K-Nearest Neighbors (KNN)',
    category: 'classical-ml',
    description: 'Instance-based learning algorithm for classification and regression',
    content: `
      <h2>K-Nearest Neighbors (KNN)</h2>
      <p>KNN is a simple, instance-based learning algorithm that makes predictions based on the k most similar training examples. It's non-parametric (makes no assumptions about data distribution) and lazy (doesn't learn during training).</p>

      <h3>How KNN Works</h3>
      <ol>
        <li><strong>Choose k:</strong> Number of nearest neighbors to consider</li>
        <li><strong>Calculate distance:</strong> Compute distance between query point and all training points</li>
        <li><strong>Find k nearest:</strong> Select k training points with smallest distances</li>
        <li><strong>Make prediction:</strong>
          <ul>
            <li><strong>Classification:</strong> Majority vote from k neighbors</li>
            <li><strong>Regression:</strong> Average (or weighted average) of k neighbor values</li>
          </ul>
        </li>
      </ol>

      <h3>Distance Metrics</h3>
      <ul>
        <li><strong>Euclidean (L2):</strong> √Σ(xᵢ - yᵢ)² (most common)</li>
        <li><strong>Manhattan (L1):</strong> Σ|xᵢ - yᵢ| (better for high dimensions)</li>
        <li><strong>Minkowski:</strong> (Σ|xᵢ - yᵢ|^p)^(1/p) (generalization, p=2 is Euclidean)</li>
        <li><strong>Cosine:</strong> 1 - (x·y)/(||x||·||y||) (good for text/sparse data)</li>
        <li><strong>Hamming:</strong> Number of differing attributes (for categorical data)</li>
      </ul>

      <h3>Choosing K</h3>
      <ul>
        <li><strong>Small k (k=1, k=3):</strong>
          <ul>
            <li>More sensitive to noise</li>
            <li>Complex decision boundaries</li>
            <li>Prone to overfitting</li>
          </ul>
        </li>
        <li><strong>Large k:</strong>
          <ul>
            <li>Smoother decision boundaries</li>
            <li>More resistant to noise</li>
            <li>Risk of underfitting</li>
            <li>May ignore local patterns</li>
          </ul>
        </li>
        <li><strong>Rule of thumb:</strong> k = √n (where n = number of training samples)</li>
        <li><strong>Use odd k</strong> for binary classification to avoid ties</li>
        <li><strong>Cross-validation</strong> is best for choosing optimal k</li>
      </ul>

      <h3>Weighted KNN</h3>
      <p>Give more weight to closer neighbors:</p>
      <ul>
        <li><strong>Uniform weights:</strong> All neighbors vote equally</li>
        <li><strong>Distance weights:</strong> Closer neighbors have more influence (weight = 1/distance)</li>
        <li>Helps when k is large or neighbors are not equally distant</li>
      </ul>

      <h3>Advantages</h3>
      <ul>
        <li>Simple and intuitive</li>
        <li>No training phase (lazy learning)</li>
        <li>No assumptions about data distribution</li>
        <li>Naturally handles multi-class problems</li>
        <li>Can capture complex decision boundaries</li>
        <li>Easy to update with new data (just add to training set)</li>
      </ul>

      <h3>Disadvantages</h3>
      <ul>
        <li>Slow prediction (O(n·d) for n samples, d features)</li>
        <li>Memory intensive (stores all training data)</li>
        <li>Curse of dimensionality (performance degrades in high dimensions)</li>
        <li>Sensitive to irrelevant features</li>
        <li>Requires feature scaling</li>
        <li>Sensitive to imbalanced data</li>
        <li>Doesn't work well with categorical features</li>
      </ul>

      <h3>Curse of Dimensionality</h3>
      <p>In high-dimensional spaces:</p>
      <ul>
        <li>All points become approximately equidistant</li>
        <li>Concept of "nearest" becomes meaningless</li>
        <li>Requires exponentially more data to maintain density</li>
        <li><strong>Solution:</strong> Dimensionality reduction (PCA, feature selection)</li>
      </ul>

      <h3>Optimization Techniques</h3>
      <ul>
        <li><strong>KD-Trees:</strong> Space-partitioning data structure (O(log n) search, works well in low dimensions)</li>
        <li><strong>Ball Trees:</strong> Better for high dimensions (d > 20)</li>
        <li><strong>Approximate methods:</strong> Locality-Sensitive Hashing (LSH) for very large datasets</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=8,
                           n_redundant=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling is CRITICAL for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
print(f"KNN (k=5) Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Find optimal k using cross-validation
k_values = range(1, 31)
cv_scores = []

for k in k_values:
    knn_cv = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_cv, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_k = k_values[np.argmax(cv_scores)]
print(f"\\nOptimal k: {optimal_k}")
print(f"Best CV accuracy: {max(cv_scores):.4f}")

# Train with optimal k
best_knn = KNeighborsClassifier(n_neighbors=optimal_k)
best_knn.fit(X_train_scaled, y_train)
test_acc = best_knn.score(X_test_scaled, y_test)
print(f"Test accuracy with k={optimal_k}: {test_acc:.4f}")`,
        explanation: 'Demonstrates KNN classification with feature scaling and finding optimal k through cross-validation. Shows how different k values affect performance.'
      },
      {
        language: 'Python',
        code: `from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Generate regression data
X, y = make_regression(n_samples=500, n_features=5, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare uniform vs distance weights
print("KNN Regression Comparison:")
for weights in ['uniform', 'distance']:
    knn_reg = KNeighborsRegressor(n_neighbors=5, weights=weights)
    knn_reg.fit(X_train_scaled, y_train)

    y_pred = knn_reg.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\\nWeights={weights}:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.4f}")

# Distance metrics comparison
print("\\n\\nDistance Metric Comparison:")
for metric in ['euclidean', 'manhattan', 'minkowski']:
    knn_metric = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn_metric.fit(X_train_scaled, y_train)
    acc = knn_metric.score(X_test_scaled, y_test)
    print(f"{metric}: {acc:.4f}")`,
        explanation: 'Shows KNN for regression with comparison of uniform vs distance-weighted predictions, and different distance metrics. Distance weighting often improves performance by giving more influence to closer neighbors.'
      }
    ],
    interviewQuestions: [
      {
        question: 'How does KNN make predictions?',
        answer: 'K-Nearest Neighbors makes predictions by finding the k closest training examples to the query point and aggregating their labels. The algorithm: (1) compute the distance from the query point to all training points using a distance metric (typically Euclidean distance: √Σ(x_i - y_i)²); (2) identify the k training points with smallest distances; (3) for classification, predict the majority class among these k neighbors (majority voting); for regression, predict the average (or weighted average) of their target values. For example, with k=5 and a binary classification, if 3 neighbors are class A and 2 are class B, predict class A.\n\nThe algorithm is **instance-based**: it memorizes the training data and makes predictions by direct comparison to stored examples, rather than learning an explicit model (like coefficients in linear regression). When a new query arrives, it performs a similarity search over all training points. With n training samples and d features, prediction requires O(n × d) distance computations per query—expensive for large datasets. This is why efficient implementations use data structures like KD-trees or Ball trees to accelerate neighbor search, reducing complexity to O(log n × d) in favorable conditions (low dimensions).\n\nKNN is **non-parametric**: it makes no assumptions about the underlying data distribution, allowing it to capture arbitrarily complex decision boundaries. The decision boundary emerges implicitly from the training data density—regions with many class A examples will be classified as A. This flexibility is powerful but comes at a cost: KNN needs sufficient data density in all regions of the feature space to make good predictions, and it suffers from the curse of dimensionality (as dimensions increase, all points become equidistant). The choice of k controls the bias-variance tradeoff: small k (like 1) gives low bias but high variance (sensitive to noise); large k gives high bias but low variance (smoother decision boundaries). KNN\'s simplicity and flexibility make it a useful baseline, though it\'s rarely optimal for high-dimensional or large-scale problems.'
      },
      {
        question: 'Why is feature scaling critical for KNN?',
        answer: 'Feature scaling is essential for KNN because the algorithm uses distance metrics to find nearest neighbors, and distances are affected by feature magnitudes. Without scaling, features with larger ranges dominate the distance calculation. Consider predicting house prices using [income in dollars, age in years]: income ranges from $20,000 to $200,000 while age ranges from 20 to 80. Computing Euclidean distance √((income₁-income₂)² + (age₁-age₂)²), the income difference (potentially tens of thousands) dwarfs the age difference (at most 60), making age essentially irrelevant to the distance measure. KNN will make predictions based almost entirely on income similarity, ignoring age.\n\n**Standardization** (z-score normalization: (x - μ)/σ) scales each feature to mean 0 and standard deviation 1, making features comparable regardless of original units. After standardization, income and age contribute equally to distances. **Min-max scaling** (x\' = (x - min)/(max - min)) scales features to a fixed range like [0, 1], also ensuring equal contribution. Standardization is generally preferred for KNN as it\'s less sensitive to outliers (which affect min and max), and it works better when feature distributions are approximately Gaussian.\n\nThe impact on model performance can be dramatic. On datasets with mixed-scale features (like UCI Adult dataset with income, age, hours-per-week), KNN without scaling may achieve 60-70% accuracy while the same model with scaling achieves 80-85%. Features with large scales become de facto feature selection: only those features matter. This is problematic because scale is often arbitrary (meters vs centimeters) and shouldn\'t determine feature importance. **When to scale**: Always scale for KNN (and other distance-based methods like K-Means, SVM with RBF kernel). Fit the scaler on training data, then transform both training and test data with those parameters to avoid data leakage. Tree-based methods (Random Forest, Gradient Boosting) don\'t require scaling since they use split thresholds that are invariant to monotonic transformations. Linear models benefit from scaling for optimization (faster convergence) but predictions remain unchanged (coefficients adjust inversely to scaling). For KNN specifically, feature scaling is not optional—it\'s a prerequisite for sensible predictions.'
      },
      {
        question: 'How do you choose the optimal value of k?',
        answer: 'Choosing k involves balancing the bias-variance tradeoff and is typically done via cross-validation. **Small k** (like k=1) has low bias (flexible, can fit complex patterns) but high variance (sensitive to noise—if a single noisy point is closest, prediction will be wrong). The decision boundary is highly irregular, with islands and tendrils around individual points. k=1 achieves 100% training accuracy (each point\'s nearest neighbor is itself) but often poor test performance. **Large k** (like k=n/2) has high bias (assumes local homogeneity) but low variance (smooth predictions averaging over many points). The decision boundary is very smooth, potentially underfitting. In the extreme k=n, every prediction is the global mode/mean.\n\nThe standard approach: try multiple k values (e.g., 1, 3, 5, 7, 9, 15, 21, 31, 51, 101), use k-fold cross-validation to estimate test performance for each, and select the k with best cross-validated accuracy (classification) or lowest RMSE (regression). Typical optimal k is often in the range 3-20 for small to medium datasets (1000-10000 samples). For large datasets, larger k values become computationally feasible and often beneficial. Some guidelines: start with k = √n as a rule of thumb; prefer odd k for binary classification to avoid ties; consider the class distribution—with imbalanced classes, larger k may help (but also consider class weighting).\n\n**Data characteristics matter**. For noisy data, use larger k to average out noise. For clean data with clear boundaries, smaller k works well. For small datasets (<100 samples), use smaller k (3-7) since large k would average too broadly. For large datasets (>10000 samples), larger k (50-100) may provide better generalization. Check learning curves: plot training and validation accuracy vs k. If both are low, all k values underfit (KNN may not be appropriate); if training is high but validation is low across k, there may be fundamental issues (insufficient data density, too many dimensions). The **elbow method** can help: plot validation error vs k and look for the "elbow" where error stops decreasing significantly. Also inspect decision boundaries visually (in 2D or 3D) to ensure they make sense—overly complex boundaries suggest k too small, overly simple suggest k too large. In practice, KNN\'s performance is quite sensitive to k, so thorough tuning is important.'
      },
      {
        question: 'What is the curse of dimensionality and how does it affect KNN?',
        answer: 'The curse of dimensionality refers to phenomena where high-dimensional spaces behave counterintuitively, causing algorithms like KNN to fail. In high dimensions, data becomes sparse: the volume of a unit hypercube grows exponentially with dimensions (volume = side_length^d), so fixing the number of data points means density decreases exponentially. With 100 samples uniformly distributed in 1D ([0,1]), average spacing is 0.01; in 10D, you\'d need 10^10 samples for the same density. Practically, we never have enough data to densely populate high-dimensional spaces, leaving KNN\'s neighborhoods empty or containing unrepresentative points.\n\nA more subtle issue: in high dimensions, distances become less informative. The ratio of the farthest to nearest neighbor approaches 1 as dimensions increase: max_dist/min_dist → 1. If all training points are approximately equidistant from the query point, the notion of "nearest" neighbors becomes meaningless—why should we trust predictions from the "closest" points when they\'re barely closer than "distant" points? This is because Euclidean distance in high dimensions is dominated by the cumulative effect of small differences across many dimensions, losing ability to discriminate. The phenomenon is measurable: in 1D, 10% of points lie within 10% of the range; in 10D, virtually all points lie far from any given point.\n\n**Practical impacts on KNN**: (1) predictions become unreliable as "nearest" neighbors aren\'t truly similar; (2) k needs to be very large to include enough meaningful neighbors, but this makes predictions overly smooth and averaged; (3) computation slows dramatically since distance calculations involve more features; (4) irrelevant features (noise dimensions) corrupt distance metrics—if only 3 of 100 dimensions are relevant, those 3 get drowned out by 97 dimensions of noise. **Mitigation strategies**: Use dimensionality reduction (PCA, t-SNE, UMAP) to project to lower dimensions preserving relevant structure; perform feature selection to remove irrelevant features; use distance metrics less sensitive to dimensionality (Manhattan distance sometimes better than Euclidean, or learned distance metrics); collect more data (though exponentially more is needed); or switch to algorithms less affected by high dimensions (tree-based methods, linear models, neural networks with appropriate regularization). As a rule, KNN becomes unreliable beyond ~20-30 dimensions without careful feature engineering or dimensionality reduction. This is why KNN works well for image recognition with engineered features (10-100 dimensions) but fails on raw pixel data (10,000+ dimensions) without reduction.'
      },
      {
        question: 'What is the difference between uniform and distance-weighted KNN?',
        answer: '**Uniform weighting** (standard KNN) gives equal weight to all k nearest neighbors when making predictions. For classification, it counts votes: if 3 of 5 neighbors are class A and 2 are class B, predict A with no consideration for how close each neighbor is. For regression, it averages values equally: ŷ = (1/k)Σy_i. This treats the 1st nearest neighbor (very close) and the kth nearest neighbor (farther away) identically. It\'s simple and works well when all k neighbors are similar distances away, but can be suboptimal when distances vary significantly within the k-neighborhood.\n\n**Distance weighting** gives more influence to closer neighbors and less to farther ones. The most common scheme is inverse distance weighting: weight_i = 1/distance_i (or 1/distance_i² for stronger emphasis on close points). For classification, compute weighted votes: Σ(weight_i × indicator(class_i = c)) for each class c, predict the class with maximum weighted vote. For regression: ŷ = Σ(weight_i × y_i) / Σ(weight_i), a weighted average. Neighbors very close to the query point dominate the prediction, while distant neighbors contribute minimally. This is intuitively appealing: why should a point at distance 10 influence the prediction as much as a point at distance 1?\n\nDistance weighting provides several advantages: **better with varying neighbor distances**—if k=10 but only 3 neighbors are very close, those 3 dominate (appropriate); **smoother predictions**—transitions between regions are more gradual; **less sensitive to k**—since distant neighbors contribute little, using k=20 vs k=10 matters less; **no tie-breaking issues**—even with even k in binary classification, weighted votes rarely tie exactly. The downsides: **computational cost**—must compute and apply weights; **sensitivity to the weighting function**—should you use 1/d, 1/d², exp(-d), or something else?; **problems when distances are zero**—if k=1 and distance is exactly 0 (duplicate points), weight becomes infinite (handle by setting weight to very large finite value or excluding duplicate).\n\n**When to use which**: Use distance weighting when your data has non-uniform density—regions where nearest neighbors are clustered close vs spread out. Use uniform weighting for simplicity when computational efficiency matters and distances within k-neighborhoods are similar. Scikit-learn\'s KNeighborsClassifier supports weights=\'uniform\' (default) or weights=\'distance\'. In practice, distance weighting often improves performance slightly (2-5% accuracy gain), especially with larger k. Try both via cross-validation to see which works better for your specific dataset. Distance weighting also helps when using large k to smooth predictions while still allowing nearby points to dominate.'
      },
      {
        question: 'Why is KNN called a "lazy learner"?',
        answer: 'KNN is called a "lazy learner" or "instance-based learner" because it performs no training phase—it doesn\'t learn a model, extract patterns, or build any data structure during training. The "training" consists entirely of storing the raw training data in memory: X_train and y_train. All computation is deferred to prediction time, when the algorithm compares the query point to all stored training examples. This contrasts with "eager learners" like linear regression, decision trees, or neural networks, which invest significant upfront computation to build a model (learn coefficients, tree structure, weights) but make fast predictions using that model.\n\nThe implications are significant. **Training time**: O(1)—just store data, making KNN trivially fast to "train." This is appealing for scenarios where you need to quickly add new training data (just append to storage). **Prediction time**: O(n × d) where n is training set size and d is dimensions—must compute distance to every training point for each query. This is expensive for large datasets or real-time applications. A trained neural network might require milliseconds for prediction; KNN with 1M training samples could take seconds per query. **Memory**: O(n × d)—must store entire training set. For 1M samples with 100 features (float32), that\'s ~400MB, which seems reasonable but pales compared to tree-based ensembles that only store tree structures (much smaller).\n\nThe lazy approach has trade-offs. **Advantages**: simple to implement and understand; no assumptions about data distribution; trivial to add new training data (online learning); naturally handles multi-class problems; decision boundary adapts instantly to new data. **Disadvantages**: slow predictions (problematic for production systems needing sub-millisecond latency); high memory usage (storing millions of samples); sensitive to irrelevant features and curse of dimensionality; no dimensionality reduction or feature learning; requires full preprocessing (scaling) on entire training set before any queries.\n\n**Eager learning** alternatives (SVM, Random Forest, Neural Networks) do the opposite: expensive training (building a compressed representation of patterns) but cheap prediction (evaluating the model on new inputs). For production systems with many predictions and infrequent retraining, eager learners are usually preferred. KNN shines in scenarios where training data changes frequently, prediction volume is low, or interpretability via similar examples is valuable (recommendation systems: "users who liked X also liked Y" essentially uses KNN logic). Modern variants like **approximate nearest neighbors** (ANN) algorithms (Annoy, FAISS, HNSW) mitigate the prediction speed issue by building indexes during "training" (making KNN slightly less lazy) and performing approximate searches in O(log n) time, making KNN competitive for large-scale applications like image retrieval.'
      }
    ],
    quizQuestions: [
      {
        id: 'knn-q1',
        question: 'What is the main disadvantage of KNN for large datasets?',
        options: [
          'Cannot handle multi-class problems',
          'Slow prediction time (must compute distance to all training points)',
          'Cannot capture non-linear patterns',
          'Requires extensive training time'
        ],
        correctAnswer: 1,
        explanation: 'KNN has no training phase but requires computing distances to all training points for each prediction (O(n) complexity). This makes prediction slow for large datasets, unlike algorithms that learn a model during training.'
      },
      {
        id: 'knn-q2',
        question: 'You apply KNN without scaling features. Feature A ranges from 0-1 while Feature B ranges from 0-1000. What happens?',
        options: [
          'Both features contribute equally',
          'Feature B dominates distance calculations, Feature A is essentially ignored',
          'KNN automatically scales features internally',
          'The model will fail to train'
        ],
        correctAnswer: 1,
        explanation: 'Without scaling, Feature B with larger range will dominate Euclidean distance calculations. A difference of 1 in Feature B outweighs the entire range of Feature A. Always scale features for KNN!'
      },
      {
        id: 'knn-q3',
        question: 'Your KNN model with k=1 achieves 100% training accuracy but 65% test accuracy. What is the best solution?',
        options: [
          'Decrease k to k=0',
          'Increase k to reduce overfitting',
          'Remove feature scaling',
          'Switch to different distance metric'
        ],
        correctAnswer: 1,
        explanation: 'k=1 memorizes training data perfectly (overfitting). Each training point predicts its own label correctly. Increasing k smooths the decision boundary by considering more neighbors, reducing overfitting.'
      }
    ]
  },

  'k-means-clustering': {
    id: 'k-means-clustering',
    title: 'K-Means Clustering',
    category: 'classical-ml',
    description: 'Unsupervised learning algorithm that partitions data into K clusters',
    content: `
      <h2>K-Means Clustering</h2>
      <p>K-Means is an unsupervised learning algorithm that partitions n observations into k clusters, where each observation belongs to the cluster with the nearest mean (centroid). It's one of the simplest and most popular clustering algorithms.</p>

      <h3>Algorithm Steps</h3>
      <ol>
        <li><strong>Initialize:</strong> Randomly select k data points as initial centroids</li>
        <li><strong>Assignment:</strong> Assign each data point to nearest centroid (using distance metric, typically Euclidean)</li>
        <li><strong>Update:</strong> Recalculate centroids as the mean of all points assigned to each cluster</li>
        <li><strong>Repeat:</strong> Steps 2-3 until convergence (centroids don't change or max iterations reached)</li>
      </ol>

      <h3>Objective Function</h3>
      <p>K-Means minimizes the within-cluster sum of squares (WCSS/inertia):</p>
      <ul>
        <li><strong>J = Σᵏᵢ₌₁ Σₓ∈Cᵢ ||x - μᵢ||²</strong></li>
        <li>Where μᵢ is the centroid of cluster Cᵢ</li>
        <li>Goal: minimize distance between points and their cluster centroids</li>
      </ul>

      <h3>Choosing K (Number of Clusters)</h3>

      <h4>Elbow Method</h4>
      <ul>
        <li>Plot WCSS vs number of clusters k</li>
        <li>Look for "elbow" where WCSS decrease slows dramatically</li>
        <li>WCSS always decreases with more clusters (k=n gives WCSS=0)</li>
        <li>Elbow indicates optimal balance between clusters and complexity</li>
      </ul>

      <h4>Silhouette Score</h4>
      <ul>
        <li>Measures how similar a point is to its cluster vs other clusters</li>
        <li>Range: [-1, 1], higher is better</li>
        <li>s = (b - a) / max(a, b) where:
          <ul>
            <li>a = average distance to points in same cluster</li>
            <li>b = average distance to points in nearest other cluster</li>
          </ul>
        </li>
        <li>Choose k that maximizes average silhouette score</li>
      </ul>

      <h4>Domain Knowledge</h4>
      <ul>
        <li>Business requirements may dictate k</li>
        <li>Natural groupings in data</li>
        <li>Practical constraints (e.g., number of customer segments)</li>
      </ul>

      <h3>Initialization Methods</h3>

      <h4>Random Initialization</h4>
      <ul>
        <li>Simple but can lead to poor local optima</li>
        <li>Run multiple times with different random seeds</li>
        <li>Choose best result (lowest WCSS)</li>
      </ul>

      <h4>K-Means++ (Recommended)</h4>
      <ul>
        <li>Smart initialization that spreads initial centroids</li>
        <li>First centroid: random point</li>
        <li>Subsequent centroids: choose points far from existing centroids (probability proportional to distance²)</li>
        <li>Leads to faster convergence and better results</li>
        <li>Default in scikit-learn</li>
      </ul>

      <h3>Advantages</h3>
      <ul>
        <li>Simple and easy to implement</li>
        <li>Scales well to large datasets (O(nkt) complexity)</li>
        <li>Guaranteed to converge (though possibly to local optimum)</li>
        <li>Works well with spherical clusters of similar size</li>
        <li>Fast and efficient</li>
      </ul>

      <h3>Disadvantages</h3>
      <ul>
        <li>Must specify k in advance</li>
        <li>Sensitive to initial centroid placement</li>
        <li>Assumes spherical clusters of similar size/density</li>
        <li>Sensitive to outliers (outliers can skew centroids)</li>
        <li>Struggles with non-convex shapes</li>
        <li>Only works with numerical data</li>
        <li>Requires feature scaling</li>
        <li>Can converge to local optima</li>
      </ul>

      <h3>Variants and Alternatives</h3>
      <ul>
        <li><strong>K-Medoids (PAM):</strong> Uses actual data points as centers (more robust to outliers)</li>
        <li><strong>Mini-Batch K-Means:</strong> Uses mini-batches for faster training on large datasets</li>
        <li><strong>DBSCAN:</strong> Density-based, doesn't require k, handles arbitrary shapes</li>
        <li><strong>Hierarchical Clustering:</strong> Creates tree of clusters, no need to specify k upfront</li>
        <li><strong>GMM (Gaussian Mixture Models):</strong> Probabilistic approach, soft clustering</li>
      </ul>

      <h3>Applications</h3>
      <ul>
        <li>Customer segmentation</li>
        <li>Image compression (color quantization)</li>
        <li>Document classification</li>
        <li>Anomaly detection</li>
        <li>Feature engineering (cluster-based features)</li>
        <li>Data preprocessing</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
X, y_true = make_blobs(n_samples=500, n_features=2, centers=4,
                       cluster_std=1.0, random_state=42)

# Feature scaling is important for K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit K-Means with k=4
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
y_pred = kmeans.fit_predict(X_scaled)

print(f"Inertia (WCSS): {kmeans.inertia_:.2f}")
print(f"Number of iterations: {kmeans.n_iter_}")
print(f"Silhouette score: {silhouette_score(X_scaled, y_pred):.3f}")

# Cluster centers
print(f"\\nCluster centers:\\n{kmeans.cluster_centers_}")

# Elbow method to find optimal k
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    y_temp = kmeans_temp.fit_predict(X_scaled)
    inertias.append(kmeans_temp.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, y_temp))

print(f"\\nElbow Method Results:")
for k, inertia, sil_score in zip(K_range, inertias, silhouette_scores):
    print(f"k={k}: Inertia={inertia:.2f}, Silhouette={sil_score:.3f}")

# Optimal k is typically where silhouette score is highest
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\\nOptimal k (by silhouette score): {optimal_k}")`,
        explanation: 'Demonstrates K-Means clustering with feature scaling, evaluation metrics (inertia and silhouette score), and the elbow method for finding optimal k. K-means++ initialization ensures better starting points.'
      },
      {
        language: 'Python',
        code: `from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
import time
import numpy as np

# Generate large dataset
X_large, _ = make_blobs(n_samples=100000, n_features=10, centers=5, random_state=42)

# Scale features
scaler = StandardScaler()
X_large_scaled = scaler.fit_transform(X_large)

# Standard K-Means
print("Standard K-Means:")
start = time.time()
kmeans_standard = KMeans(n_clusters=5, random_state=42)
kmeans_standard.fit(X_large_scaled)
standard_time = time.time() - start
print(f"Time: {standard_time:.2f}s")
print(f"Inertia: {kmeans_standard.inertia_:.2f}")

# Mini-Batch K-Means (faster for large datasets)
print("\\nMini-Batch K-Means:")
start = time.time()
kmeans_minibatch = MiniBatchKMeans(n_clusters=5, batch_size=1000, random_state=42)
kmeans_minibatch.fit(X_large_scaled)
minibatch_time = time.time() - start
print(f"Time: {minibatch_time:.2f}s")
print(f"Inertia: {kmeans_minibatch.inertia_:.2f}")
print(f"Speedup: {standard_time/minibatch_time:.1f}x")

# Predict on new data
new_point = np.random.randn(1, 10)
new_point_scaled = scaler.transform(new_point)
cluster_assignment = kmeans_standard.predict(new_point_scaled)
distance_to_centroid = kmeans_standard.transform(new_point_scaled).min()

print(f"\\nNew point assigned to cluster: {cluster_assignment[0]}")
print(f"Distance to nearest centroid: {distance_to_centroid:.2f}")`,
        explanation: 'Compares standard K-Means with Mini-Batch K-Means for large datasets. Mini-Batch is significantly faster with minimal loss in quality. Shows how to predict cluster assignments for new data points.'
      }
    ],
    interviewQuestions: [
      {
        question: 'How does the K-Means algorithm work?',
        answer: 'K-Means is an iterative clustering algorithm that partitions n data points into k clusters by minimizing within-cluster variance. The algorithm alternates between two steps: (1) **Assignment step**: assign each point to the nearest centroid (cluster center) based on Euclidean distance; (2) **Update step**: recalculate each centroid as the mean of all points assigned to that cluster. This process repeats until convergence (centroids no longer change significantly) or a maximum number of iterations is reached. The algorithm is guaranteed to converge, though not necessarily to the global optimum.\n\nThe algorithm begins with initialization: randomly select k data points as initial centroids (or use a smarter initialization like K-Means++). Then iterate: compute distance from each point to each centroid, assign each point to the closest centroid (creating k clusters), compute the new centroid of each cluster as the mean position of its points, repeat until centroids stabilize. For example, with 2D data and k=3, you might start with centroids at random positions, assign each point to the nearest centroid (creating three clusters), compute the center of mass of each cluster, update centroids to those positions, reassign points based on new centroids, and continue until assignments no longer change.\n\nK-Means is simple, fast, and scalable—it runs in O(n × k × i × d) where n is points, k is clusters, i is iterations (typically <100), and d is dimensions. However, it has limitations: requires specifying k beforehand, assumes clusters are spherical and similar size, sensitive to initialization (can converge to local minima), affected by outliers (since means are not robust), and struggles with non-convex cluster shapes. Despite these limitations, K-Means is widely used for its efficiency and simplicity, serving as a go-to baseline for clustering tasks. Variants address some limitations: K-Medoids uses median instead of mean (more robust to outliers), K-Means++ improves initialization, Mini-batch K-Means scales to massive datasets through sampling.'
      },
      {
        question: 'What is the objective function that K-Means minimizes?',
        answer: 'K-Means minimizes the **within-cluster sum of squares (WCSS)**, also called inertia or distortion: J = ΣΣ ||x - μ_c||², where the outer sum is over all k clusters c, the inner sum is over all points x in cluster c, and μ_c is the centroid of cluster c. In words: for each cluster, compute the squared Euclidean distance from each point to its centroid, sum those distances within the cluster, then sum across all clusters. This measures how compact the clusters are—smaller WCSS means points are closer to their centroids, indicating tighter clusters.\n\nThe algorithm minimizes this objective through **coordinate descent**: the assignment step optimizes cluster assignments with centroids fixed, and the update step optimizes centroids with assignments fixed. Each step is guaranteed to decrease (or keep constant) the objective, ensuring convergence. In the assignment step, assigning each point to its nearest centroid minimizes the total distance (proof: any other assignment would increase the sum of squared distances). In the update step, setting the centroid to the mean of cluster points minimizes the sum of squared distances within that cluster (proof: the mean is the point that minimizes sum of squared distances to a set of points).\n\nMinimizing WCSS has an intuitive interpretation: we want clusters where points are similar (close together) and dissimilar to other clusters (far from other centroids). However, WCSS always decreases as k increases—with k=n (each point its own cluster), WCSS=0. So you can\'t just pick k that minimizes WCSS; you need methods like the elbow method (plot WCSS vs k, look for the "elbow" where the decrease slows) or silhouette analysis (measures how well points fit their clusters vs other clusters). The objective is also sensitive to scale: features with larger ranges dominate the distance calculation, so feature scaling (standardization) is important before applying K-Means. The squared Euclidean distance makes K-Means sensitive to outliers (they contribute disproportionately to the objective), which is why alternative objectives like K-Medoids (uses L1 distance, more robust) or DBSCAN (density-based, no explicit objective) may be preferable for noisy data.'
      },
      {
        question: 'How do you choose the optimal number of clusters k?',
        answer: 'Choosing k is challenging because clustering is unsupervised—there\'s no ground truth to validate against. Several methods exist, each with trade-offs. The **elbow method** plots within-cluster sum of squares (WCSS) against k. WCSS always decreases with k, but the rate of decrease slows. The "elbow" is the point where adding more clusters yields diminishing returns. For example, if WCSS drops from 1000 to 400 (k=1 to k=2), to 200 (k=3), to 150 (k=4), to 140 (k=5), the elbow is around k=4. The method is intuitive but subjective—the elbow isn\'t always clear.\n\n**Silhouette analysis** computes the silhouette coefficient for each point: s = (b - a) / max(a, b), where a is the mean distance to other points in the same cluster, and b is the mean distance to points in the nearest other cluster. s ranges from -1 (poor clustering, point closer to other cluster) to +1 (good clustering, point far from other clusters). Average silhouette across all points gives a quality score for k. Higher average silhouette indicates better-defined clusters. Plot average silhouette vs k and choose k with the highest score. Silhouette is more rigorous than the elbow method but computationally expensive for large datasets.\n\n**Domain knowledge** often provides the best guidance. If clustering customers, business needs might dictate 3-5 segments for marketing campaigns. If compressing images, k is determined by the desired compression ratio. **Gap statistic** compares WCSS to that expected under a null reference distribution (uniform random data), choosing k where the gap is largest. **Dendrogram** (hierarchical clustering) visualizes cluster merging at different levels, helping identify natural cluster counts. In practice, try multiple methods and validate results: do the clusters make sense? Are they actionable? For exploratory analysis, try a range of k values (e.g., 2-10) and examine cluster characteristics (size, mean values) to see which k tells the most interesting or useful story. Remember: the "optimal" k depends on your goal—data may have natural structure at multiple scales (3 high-level groups, 10 fine-grained segments), and the best k depends on your use case.'
      },
      {
        question: 'What is the difference between K-Means and K-Means++?',
        answer: 'K-Means++ is an improved initialization method for K-Means that addresses the algorithm\'s sensitivity to initial centroid placement. Standard K-Means randomly selects k data points as initial centroids, which can lead to poor results: if initial centroids are clustered together, the algorithm may converge to a local minimum with uneven cluster sizes or high WCSS. K-Means must be run multiple times (typically 10-50) with different random initializations, selecting the run with lowest WCSS. This is computationally expensive and still may miss good solutions.\n\n**K-Means++** initializes centroids to be far apart, increasing the chance of starting near a good solution. The algorithm: (1) choose the first centroid uniformly at random from the data points; (2) for each remaining point, compute distance D(x) to the nearest already-chosen centroid; (3) choose the next centroid from remaining points with probability proportional to D(x)²—points farther from existing centroids are more likely to be selected; (4) repeat until k centroids are chosen. This ensures initial centroids are spread out across the data, capturing different regions.\n\nThe probabilistic selection (proportional to D²) is key: it balances exploration (selecting distant points) with avoiding outliers (which would be selected deterministically if we just picked the farthest point each time). K-Means++ provides two major benefits: **better final clustering**—empirically produces lower WCSS and more balanced clusters; **faster convergence**—requires fewer iterations since initialization is closer to the optimal solution, and often needs fewer random restarts (3-5 vs 10-50). Scikit-learn\'s KMeans defaults to init=\'k-means++\', and it\'s generally recommended over random initialization.\n\nThe computational cost of K-Means++ initialization is O(n × k) (evaluating distances for each point to choose each centroid), which is negligible compared to the O(n × k × i × d) cost of the main algorithm. The original 2007 paper by Arthur and Vassilvitskii proved that K-Means++ is O(log k)-competitive with the optimal clustering in expectation, providing theoretical guarantees beyond empirical performance. In practice, always use K-Means++—it\'s strictly better than random initialization with minimal additional cost. The main time you might skip it is for massive datasets (>10M points) where even the initialization becomes expensive, in which case Mini-batch K-Means with its own fast initialization might be preferable.'
      },
      {
        question: 'What are the limitations of K-Means clustering?',
        answer: 'K-Means has several significant limitations. **Requires specifying k beforehand**: You must decide how many clusters exist before seeing the data, which is often unclear. Methods like the elbow method or silhouette analysis help but don\'t fully solve this chicken-and-egg problem. **Assumes spherical clusters of similar size**: K-Means uses Euclidean distance from points to centroids, implicitly assuming clusters are spherical (circular in 2D, spherical in 3D, hyperspherical in higher dimensions) and roughly equal in variance. It fails on elongated, irregular, or nested clusters. For example, concentric circles or crescent-shaped clusters will be incorrectly split.\n\n**Sensitive to initialization**: Can converge to local minima depending on initial centroids. K-Means++ helps but doesn\'t eliminate this issue. **Not robust to outliers**: Uses mean for centroids, which is heavily influenced by extreme values. A single outlier can pull a centroid away from the cluster center, distorting assignments. K-Medoids (using median) is more robust but computationally expensive. **Scale-dependent**: Features with larger ranges dominate distance calculations. Always standardize features before clustering. **Assumes Euclidean distance is meaningful**: For categorical data, text, or complex objects, Euclidean distance may not capture similarity well.\n\n**Hard assignments**: Each point belongs to exactly one cluster with no uncertainty. In reality, some points may be ambiguous (between clusters) or outliers (belong to no cluster). Alternatives: Gaussian Mixture Models provide soft assignments (probabilities); DBSCAN can mark outliers. **Struggles with varying densities**: If clusters have very different densities (one dense, one sparse), K-Means may split the dense cluster. **Curse of dimensionality**: Like KNN, K-Means degrades in high dimensions where distances become less meaningful. Dimensionality reduction (PCA) before clustering can help.\n\n**When K-Means fails, consider alternatives**: Hierarchical clustering (no need to specify k upfront, handles non-spherical clusters better), DBSCAN (density-based, finds arbitrary shapes, identifies outliers), Gaussian Mixture Models (soft assignments, handles elliptical clusters), Spectral clustering (uses graph structure, handles complex shapes). Despite limitations, K-Means remains popular for its simplicity, speed, and scalability—it works well enough for many practical clustering tasks, especially with proper preprocessing (scaling, outlier removal) and when clusters are roughly spherical and well-separated.'
      },
      {
        question: 'How does K-Means handle outliers, and what can you do about it?',
        answer: 'K-Means handles outliers poorly because it uses the **mean** to compute centroids, and means are highly sensitive to extreme values. A single outlier far from a cluster can pull the centroid toward it, distorting the cluster boundary and causing nearby points to be misclassified. In the worst case, an outlier might be assigned its own cluster (if initialized near it) or pull a centroid so far that the cluster splits unnaturally. Since the algorithm minimizes squared distances, outliers (with large distances) contribute disproportionately to the objective, forcing the algorithm to "pay attention" to them.\n\nFor example, imagine a spherical cluster of 100 points with one outlier 10× farther away. The centroid will shift toward the outlier, and the cluster boundary will extend to include the outlier, potentially pulling in points from other clusters. This is exacerbated when k is incorrectly specified—if k is too large, outliers may form singleton clusters; if k is too small, outliers distort legitimate clusters. Outliers also affect initialization: if an outlier is selected as an initial centroid (even with K-Means++), it may persist as a singleton cluster or distort nearby clusters.\n\n**Solutions include**: **Preprocessing**: Detect and remove outliers before clustering using statistical methods (Z-score > 3, IQR method) or domain knowledge. This is the simplest approach if outliers are genuinely errors or irrelevant. **Robust clustering algorithms**: Use **K-Medoids** (PAM algorithm), which uses the median instead of mean—medoids are actual data points and more robust to outliers. The trade-off is computational cost: O(k(n-k)²) per iteration vs O(nk) for K-Means. **DBSCAN** (Density-Based Spatial Clustering) explicitly identifies outliers as points in low-density regions, leaving them unassigned to any cluster.\n\n**Soft assignments**: Use **Gaussian Mixture Models (GMM)** with outlier detection—fit the model, then flag points with very low probability under any component as outliers. **Trimmed K-Means**: A variant that ignores a fixed percentage (e.g., 5%) of points farthest from centroids in each iteration, effectively removing outliers dynamically. **HDBSCAN** (Hierarchical DBSCAN) is even more robust, finding clusters of varying density and marking outliers. **Weighted K-Means**: Assign lower weights to suspected outliers (though this requires identifying them first).\n\nIn practice, the best approach depends on your data and goals. If outliers are errors, remove them. If they\'re rare but legitimate points (e.g., high-value customers), use K-Medoids or DBSCAN to prevent them from distorting clusters. If you have many outliers or noisy data, DBSCAN or HDBSCAN may be more appropriate than K-Means entirely. Always visualize your clusters (at least in 2D/3D via PCA) to spot outliers pulling centroids and validate that cluster assignments make sense.'
      }
    ],
    quizQuestions: [
      {
        id: 'kmeans-q1',
        question: 'Why does K-Means require feature scaling?',
        options: [
          'It improves algorithm speed',
          'Features with larger scales dominate distance calculations',
          'It is not required for K-Means',
          'It helps visualize clusters better'
        ],
        correctAnswer: 1,
        explanation: 'K-Means uses Euclidean distance to assign points to clusters. Without scaling, features with larger ranges (e.g., income: 0-100k vs age: 0-100) will dominate the distance calculation, leading to poor clusters.'
      },
      {
        id: 'kmeans-q2',
        question: 'You run K-Means with k=5 and get widely different results each time. What is the most likely cause?',
        options: [
          'The dataset is too large',
          'Poor random initialization of centroids',
          'Features are not scaled',
          'K-Means cannot handle 5 clusters'
        ],
        correctAnswer: 1,
        explanation: 'K-Means is sensitive to initial centroid placement and can converge to different local optima. Use K-Means++ initialization or run multiple times (n_init parameter) and select the best result.'
      },
      {
        id: 'kmeans-q3',
        question: 'Using the elbow method, you plot inertia vs k and see continuous smooth decrease. What should you do?',
        options: [
          'Choose k=1 (lowest k)',
          'Choose k=n (highest k)',
          'Use silhouette score or domain knowledge to select k',
          'K-Means is not suitable for this data'
        ],
        correctAnswer: 2,
        explanation: 'When there\'s no clear elbow, inertia alone is insufficient. Use silhouette score to measure cluster quality, or leverage domain knowledge about natural groupings in the data. Consider alternative clustering methods like DBSCAN.'
      }
    ]
  },

  'principal-component-analysis': {
    id: 'principal-component-analysis',
    title: 'Principal Component Analysis (PCA)',
    category: 'classical-ml',
    description: 'Dimensionality reduction technique that transforms data to uncorrelated principal components',
    content: `
      <h2>Principal Component Analysis (PCA)</h2>
      <p>PCA is an unsupervised dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving as much variance as possible. It identifies orthogonal directions (principal components) that capture maximum variance in the data.</p>

      <h3>Core Concept</h3>
      <p>PCA finds a new coordinate system where:</p>
      <ul>
        <li><strong>First principal component (PC1):</strong> Direction of maximum variance</li>
        <li><strong>Second principal component (PC2):</strong> Direction of maximum remaining variance, orthogonal to PC1</li>
        <li><strong>Subsequent PCs:</strong> Each orthogonal to all previous, capturing remaining variance</li>
        <li>Components are ordered by variance explained</li>
        <li>Transform data by projecting onto selected components</li>
      </ul>

      <h3>Mathematical Foundation</h3>
      <ol>
        <li><strong>Standardize data:</strong> Center by subtracting mean (and optionally scale)</li>
        <li><strong>Compute covariance matrix:</strong> C = (1/n)X^T X</li>
        <li><strong>Eigendecomposition:</strong> Find eigenvectors and eigenvalues of C</li>
        <li><strong>Sort by eigenvalues:</strong> Larger eigenvalues = more variance</li>
        <li><strong>Select top k eigenvectors:</strong> These are the principal components</li>
        <li><strong>Transform data:</strong> X_new = X · W (where W = selected eigenvectors)</li>
      </ol>

      <h3>Variance Explained</h3>
      <ul>
        <li>Each eigenvalue represents variance captured by its principal component</li>
        <li><strong>Explained variance ratio:</strong> eigenvalue / sum(all eigenvalues)</li>
        <li><strong>Cumulative explained variance:</strong> Sum of variance ratios up to component k</li>
        <li>Typically retain components capturing 95-99% cumulative variance</li>
      </ul>

      <h3>Choosing Number of Components</h3>

      <h4>Explained Variance Threshold</h4>
      <ul>
        <li>Keep components until cumulative variance ≥ threshold (e.g., 0.95)</li>
        <li>Balance between dimensionality reduction and information retention</li>
      </ul>

      <h4>Scree Plot</h4>
      <ul>
        <li>Plot eigenvalues vs component number</li>
        <li>Look for "elbow" where variance drops sharply</li>
        <li>Keep components before the elbow</li>
      </ul>

      <h4>Cross-Validation</h4>
      <ul>
        <li>Use PCA as preprocessing for supervised learning</li>
        <li>Choose k that optimizes downstream model performance</li>
      </ul>

      <h3>Advantages</h3>
      <ul>
        <li>Reduces dimensionality while preserving variance</li>
        <li>Removes multicollinearity (components are orthogonal)</li>
        <li>Speeds up training for downstream models</li>
        <li>Helps visualize high-dimensional data (2D/3D projection)</li>
        <li>Can denoise data (remove low-variance components)</li>
        <li>Fast and deterministic (no hyperparameters to tune)</li>
      </ul>

      <h3>Disadvantages</h3>
      <ul>
        <li>Components are linear combinations (can't capture non-linear relationships)</li>
        <li>Loss of interpretability (components are mixtures of original features)</li>
        <li>Sensitive to feature scaling (must standardize first)</li>
        <li>Assumes variance = importance (not always true)</li>
        <li>Outliers can distort principal components</li>
        <li>Computational cost for very high dimensions (O(n²d + d³))</li>
      </ul>

      <h3>Use Cases</h3>
      <ul>
        <li><strong>Dimensionality reduction:</strong> Reduce features before modeling</li>
        <li><strong>Visualization:</strong> Project to 2D/3D for plotting</li>
        <li><strong>Noise reduction:</strong> Keep top components, discard noisy low-variance ones</li>
        <li><strong>Feature engineering:</strong> Create uncorrelated features</li>
        <li><strong>Data compression:</strong> Store data with fewer dimensions</li>
        <li><strong>Multicollinearity removal:</strong> For linear regression</li>
      </ul>

      <h3>Variants</h3>
      <ul>
        <li><strong>Kernel PCA:</strong> Non-linear dimensionality reduction using kernel trick</li>
        <li><strong>Incremental PCA:</strong> For datasets too large to fit in memory</li>
        <li><strong>Sparse PCA:</strong> Components with many zero weights (more interpretable)</li>
        <li><strong>Probabilistic PCA:</strong> Adds noise model for missing data</li>
      </ul>

      <h3>PCA vs Other Methods</h3>
      <ul>
        <li><strong>t-SNE/UMAP:</strong> Better for visualization, captures non-linear structure (but not for modeling)</li>
        <li><strong>Autoencoders:</strong> Non-linear, can learn complex representations</li>
        <li><strong>Feature selection:</strong> Keeps original features (maintains interpretability)</li>
        <li><strong>LDA:</strong> Supervised, maximizes class separability (not just variance)</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# IMPORTANT: Standardize features before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("Explained variance by component:")
for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
    print(f"PC{i+1}: {var:.3f} (cumulative: {cum_var:.3f})")

# How many components for 95% variance?
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\\nComponents needed for 95% variance: {n_components_95}")

# Reduce to 2 components for visualization
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

print(f"\\nOriginal shape: {X.shape}")
print(f"Reduced shape: {X_2d.shape}")
print(f"Variance retained: {pca_2d.explained_variance_ratio_.sum():.3f}")

# Component loadings (contribution of each feature to PC)
loadings = pca_2d.components_.T * np.sqrt(pca_2d.explained_variance_)
print(f"\\nFeature loadings on PC1:\\n{loadings[:, 0]}")`,
        explanation: 'Demonstrates PCA with proper feature scaling, variance analysis, and dimensionality reduction. Shows how to determine the number of components needed and interpret feature contributions to principal components.'
      },
      {
        language: 'Python',
        code: `from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load high-dimensional dataset (64 features)
digits = load_digits()
X, y = digits.data, digits.target

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Original features: {X.shape[1]}")

# Compare model performance with different numbers of PCA components
results = []
n_components_list = [5, 10, 20, 30, 40, 50, 64]

for n_comp in n_components_list:
    if n_comp < X.shape[1]:
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X_scaled)
        var_explained = pca.explained_variance_ratio_.sum()
    else:
        X_pca = X_scaled
        var_explained = 1.0

    # Evaluate with Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(rf, X_pca, y, cv=5)

    results.append({
        'n_components': n_comp,
        'variance': var_explained,
        'accuracy': scores.mean(),
        'std': scores.std()
    })

    print(f"n_components={n_comp}: Variance={var_explained:.3f}, "
          f"Accuracy={scores.mean():.3f} (+/- {scores.std():.3f})")

# Find optimal number of components
best = max(results, key=lambda x: x['accuracy'])
print(f"\\nBest: {best['n_components']} components with {best['accuracy']:.3f} accuracy")`,
        explanation: 'Shows how to use PCA as preprocessing for machine learning. Compares model performance with different numbers of components using cross-validation to find the optimal dimensionality reduction.'
      }
    ],
    interviewQuestions: [
      {
        question: 'What is PCA and what problem does it solve?',
        answer: 'Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional representation while preserving as much variance (information) as possible. It solves the problem of the **curse of dimensionality**—high-dimensional data is sparse, hard to visualize, computationally expensive, and prone to overfitting. PCA identifies the directions (principal components) along which data varies most, then projects data onto these directions, discarding dimensions with low variance that contribute little information.\n\nMathematically, PCA finds an orthogonal (perpendicular) set of axes that maximize variance. The first principal component (PC1) is the direction of maximum variance in the data. The second PC (PC2) is orthogonal to PC1 and captures the maximum remaining variance. This continues for all dimensions. For data with d features, PCA produces d principal components, but typically only the first k components (k << d) are kept, reducing dimensionality from d to k. For example, 100-dimensional data might be reduced to 10 dimensions, retaining 95% of variance.\n\nPCA is useful for: **visualization** (project to 2D or 3D for plotting), **noise reduction** (low-variance dimensions often contain noise), **feature extraction** (create new features that are linear combinations of originals), **speeding up algorithms** (fewer dimensions = faster training), **addressing multicollinearity** (PCs are uncorrelated by construction), and **data compression** (store data more efficiently). The trade-off is interpretability—principal components are linear combinations of original features and don\'t have inherent meaning. PCA is unsupervised (doesn\'t use labels) and assumes linear relationships. For non-linear dimensionality reduction, alternatives like t-SNE, UMAP, or kernel PCA are more appropriate.'
      },
      {
        question: 'How do you determine the number of principal components to retain?',
        answer: 'Choosing the number of components k involves balancing dimensionality reduction benefits against information loss. The most common method is the **explained variance ratio**: each principal component explains a fraction of total variance, and these fractions sum to 1. Plot cumulative explained variance vs number of components and choose k where the curve plateaus (elbow method) or where cumulative variance reaches a threshold like 90-95%. For example, if PC1-PC3 explain [40%, 30%, 15%] of variance respectively, three components retain 85%, which may be sufficient.\n\nThe **scree plot** visualizes individual explained variance per component. It typically shows exponential decay—early components explain a lot, later components explain very little. Look for the "elbow" where the curve flattens, suggesting additional components add minimal value. This is subjective but provides intuition about dimensionality. Some implementations also provide **Kaiser criterion**: retain components with eigenvalues >1 (in standardized data, each original feature has variance 1, so eigenvalue >1 means the PC captures more variance than a single original feature). This is a rough heuristic and can be too conservative or aggressive depending on data.\n\n**Cross-validation** provides a more rigorous approach: try different k values (e.g., 5, 10, 20, 50), train your downstream model on k-dimensional data, evaluate performance via cross-validation, and choose k that optimizes the accuracy-simplicity trade-off. This directly optimizes for your task rather than an arbitrary variance threshold. For example, maybe 10 components give 90% accuracy while 50 components give 92%—you might choose 10 for simplicity.\n\n**Domain considerations** matter: for visualization, k=2 or k=3 (human perception limit). For compression, k depends on storage vs quality trade-off. For preprocessing before classification, try multiple k values and validate. For exploratory analysis, examine how much variance is explained by top components—if PC1-PC2 explain 80%, your data is essentially 2D; if you need 50 components for 80%, it\'s truly high-dimensional. In practice, starting with k that retains 90-95% of variance is a safe default, then refining based on downstream task performance. Always check that the reduced representation actually helps—sometimes all d dimensions are necessary, and PCA provides no benefit.'
      },
      {
        question: 'Why is feature scaling important for PCA?',
        answer: 'Feature scaling is critical for PCA because the algorithm identifies directions of maximum variance, and variance is scale-dependent. Features with larger scales (magnitude) will dominate the principal components, even if they\'re less informative. Consider data with [income in dollars, age in years]: income ranges from 20,000 to 200,000 (variance ~10^9), while age ranges from 20 to 80 (variance ~400). PC1 will almost entirely align with the income dimension because it has far greater variance, and age will be ignored even if it\'s equally important. This defeats PCA\'s purpose of finding meaningful structure.\n\n**Standardization** (z-score normalization: subtract mean, divide by standard deviation) scales each feature to mean 0 and variance 1, making them comparable. After standardization, each feature contributes proportionally to its "relative variance" (how spread out it is compared to its own scale) rather than its absolute magnitude. This ensures PCA discovers structure based on data patterns, not arbitrary units. For example, measuring height in millimeters vs meters would give vastly different PCA results without scaling, which is clearly wrong—the underlying structure shouldn\'t change with unit choice.\n\nWithout scaling, PCA essentially performs feature selection by variance magnitude: high-variance features are kept, low-variance features are discarded. Sometimes this is desired—if you have sensor data where variance genuinely indicates information content, you might skip scaling. But usually, this is problematic. **When to scale**: Always standardize for PCA unless you have a specific reason not to (e.g., features are already on the same scale, like pixel intensities 0-255). Scikit-learn\'s PCA doesn\'t auto-scale, so you must apply StandardScaler first: scaler.fit(X_train), X_train_scaled = scaler.transform(X_train), pca.fit(X_train_scaled).\n\n**Min-max scaling** (to [0,1] range) is an alternative but less common for PCA—it preserves relative variances better than leaving data unscaled but doesn\'t account for different spreads around the mean. Standardization is generally preferred. The impact of not scaling can be dramatic: on mixed-scale data, PCA without scaling may retain the wrong features entirely, while PCA with scaling discovers meaningful patterns. This is one of the most common PCA mistakes—always remember to standardize first.'
      },
      {
        question: 'What is the difference between PCA and feature selection?',
        answer: '**PCA (feature extraction)** creates new features as linear combinations of original features. The principal components are derived features: PC1 = w₁₁·x₁ + w₁₂·x₂ + ... + w₁d·xd, where w values are the loadings (weights). You transform your original d-dimensional data into k-dimensional data where k < d, and the new dimensions don\'t correspond to any single original feature—they\'re synthetic. For example, in a dataset with height and weight, PC1 might be "size" (a mix of both). You lose the original features; you can\'t directly interpret results as "feature 5 is important."\n\n**Feature selection** chooses a subset of the original features to keep, discarding the rest. Methods include: filter methods (rank features by correlation with target, keep top k); wrapper methods (search feature subsets via cross-validation, e.g., recursive feature elimination); embedded methods (use regularization like Lasso which sets coefficients to zero). If you start with 100 features and select 10, you have exactly those 10 original features—nothing new is created. You can still interpret results in terms of the original variables: "income and education are the most important features."\n\n**Trade-offs**: PCA can combine correlated features effectively (if x₁ and x₂ are highly correlated, PC1 captures their shared information), potentially using information from all features. Feature selection discards features entirely, potentially losing information, but maintains interpretability. PCA is unsupervised (doesn\'t consider the target variable), so it might retain variance that\'s irrelevant for prediction. Feature selection can be supervised (directly optimizes for prediction), focusing on features that matter for your specific task.\n\n**When to use PCA**: High multicollinearity (many correlated features), need dimensionality reduction for speed/memory, visualization, or when curse of dimensionality is an issue. **When to use feature selection**: Need interpretability, want to understand which original features matter, have domain knowledge suggesting some features are noise, or have genuinely independent features where combinations don\'t make sense. In practice, you can combine both: use feature selection to remove obviously irrelevant features, then use PCA on the remaining features. Or compare both via cross-validation to see which works better for your problem. PCA is a transformation; feature selection is a subset choice—fundamentally different approaches to dimensionality reduction.'
      },
      {
        question: 'Can PCA capture non-linear relationships in data?',
        answer: 'No, standard PCA is a **linear** method—it finds linear combinations of features and projects data along linear axes. It can only capture linear structure: if data lies on or near a linear subspace (e.g., points clustered along a line, plane, or hyperplane), PCA will represent it efficiently. For non-linear structure—data lying on a curve, spiral, Swiss roll, or nonlinear manifold—PCA will fail to find a compact representation. It will see the bounding box of the structure and allocate components to span that box, requiring many components to approximate something that\'s fundamentally low-dimensional but non-linear.\n\nFor example, consider data on a circle in 2D. The data is intrinsically 1D (parameterized by angle θ), but PCA needs 2 components to represent it because it uses linear projections. Projecting a circle onto any 1D line loses information. Similarly, the Swiss roll (a 2D manifold embedded in 3D) requires all 3 PCA components, though it\'s intrinsically 2D. PCA can\'t "unroll" the structure because that requires non-linear transformations.\n\n**Non-linear alternatives**: **Kernel PCA** extends PCA by implicitly mapping data to a higher-dimensional space via a kernel function (like the kernel trick in SVM), then applying PCA in that space. This can capture polynomial or RBF-kernel-defined non-linear relationships. For example, polynomial kernel PCA can separate data lying on concentric circles. **t-SNE** (t-distributed Stochastic Neighbor Embedding) and **UMAP** (Uniform Manifold Approximation and Projection) are modern non-linear techniques specifically designed for visualization—they preserve local structure and can unroll manifolds beautifully. They\'re better for visualization but less interpretable and not invertible (can\'t map back to original space reliably). **Autoencoders** (neural networks) learn non-linear encodings for dimensionality reduction and are highly flexible but require more data and computation.\n\n**When linear PCA is sufficient**: Many real-world datasets have approximately linear structure, at least locally, making PCA effective despite its linearity. High-dimensional data often lies near a lower-dimensional linear subspace due to correlations and redundancy. For preprocessing before classification/regression, linear PCA often suffices. **When you need non-linear**: Complex manifolds (images, high-dimensional sensor data), visualization where preserving local neighborhood structure is critical, or when PCA explains little variance (suggesting non-linear structure). Try PCA first for its simplicity and speed; if it fails (poor variance capture, poor downstream performance), explore non-linear alternatives.'
      },
      {
        question: 'How do you interpret principal components?',
        answer: 'Interpreting principal components involves understanding the **loadings** (weights) that define how each PC is constructed from original features. Each PC is a linear combination: PCᵢ = wᵢ₁·x₁ + wᵢ₂·x₂ + ... + wᵢd·xd. The loadings wᵢⱼ indicate how much feature xⱼ contributes to PCᵢ. Large positive/negative loadings mean that feature strongly influences the component. Examine the loading matrix (components × features) to understand what each PC represents.\n\nFor example, in a dataset with [height, weight, age, income], if PC1 has high positive loadings on height and weight but near-zero on age and income, PC1 represents "physical size." If PC2 has high positive loading on income and moderate negative loading on age, it might represent "career stage" (high income, lower age = early career high earner; low income, higher age = late career low earner). This requires domain knowledge—loadings are mathematical but interpretation is semantic.\n\n**Challenges**: Components are combinations of multiple features, making them less interpretable than original features. Loadings can be mixed (many features contribute), making it hard to assign meaning. Signs (positive/negative) are arbitrary—flipping all signs of a component doesn\'t change anything mathematically, so "high PC1" vs "low PC1" interpretation requires care. **Visualizations help**: plot loadings as bar charts or heatmaps to see which features dominate each PC. For 2D/3D projections, plot data colored by an attribute (e.g., class labels) and see how classes separate—this shows what structure the PCs capture.\n\n**Biplot** simultaneously shows data points projected onto PC1-PC2 and the loading vectors for original features, revealing how features contribute to the projection. **Cumulative explained variance** tells you how much information each PC captures but not what that information means. In practice, interpret the first few PCs (which explain most variance) by examining their top-loading features and visualizing data in PC space. Later PCs often represent noise or subtle patterns and may not need interpretation.\n\n**When interpretation matters**: Exploratory data analysis (understand structure), communicating results to non-technical stakeholders, validating that PCA makes sense (sanity check). **When it doesn\'t**: If you\'re just using PCA as preprocessing for a black-box model, interpretability is less important—focus on whether it improves downstream performance. Accept that PCA trades interpretability for dimensionality reduction, and if you need interpretability, consider feature selection instead.'
      }
    ],
    quizQuestions: [
      {
        id: 'pca-q1',
        question: 'You have a dataset with 100 features. You apply PCA and the first principal component explains 95% of variance. What should you do?',
        options: [
          'Use only PC1 and discard all other components',
          'Investigate further - one component capturing 95% variance may indicate issues (redundant features, scaling problems)',
          'Always use all 100 components',
          'PCA failed, use original features'
        ],
        correctAnswer: 1,
        explanation: 'While it seems efficient, one component capturing 95% variance is unusual and may indicate: (1) features are highly correlated/redundant, (2) one feature dominates due to scaling issues, or (3) data lies on a low-dimensional manifold. Investigate before proceeding.'
      },
      {
        id: 'pca-q2',
        question: 'You apply PCA without standardizing features. Feature A has range [0, 1] and Feature B has range [0, 1000]. What happens?',
        options: [
          'Both features contribute equally to principal components',
          'Feature B dominates PCA because it has higher variance',
          'PCA automatically standardizes features internally',
          'PCA will fail to converge'
        ],
        correctAnswer: 1,
        explanation: 'PCA finds directions of maximum variance. Without standardization, Feature B with range [0, 1000] has much higher variance than Feature A and will dominate the principal components. Always standardize before PCA!'
      },
      {
        id: 'pca-q3',
        question: 'Your model performs worse after applying PCA. What is a likely reason?',
        options: [
          'PCA always improves performance',
          'Important information for classification was in low-variance directions that PCA discarded',
          'Too many components were retained',
          'PCA introduces randomness'
        ],
        correctAnswer: 1,
        explanation: 'PCA maximizes variance, not classification accuracy. Features with low variance can still be important for discrimination. Consider supervised methods like LDA, or use cross-validation to select the number of components.'
      }
    ]
  },

  'naive-bayes': {
    id: 'naive-bayes',
    title: 'Naive Bayes',
    category: 'classical-ml',
    description: 'Probabilistic classifier based on Bayes theorem with strong independence assumptions',
    content: `
      <h2>Naive Bayes</h2>
      <p>Naive Bayes is a probabilistic classification algorithm based on Bayes' Theorem with the "naive" assumption that features are conditionally independent given the class label. Despite this strong assumption, it performs surprisingly well in many real-world applications.</p>

      <h3>Bayes' Theorem</h3>
      <p><strong>P(C|X) = [P(X|C) × P(C)] / P(X)</strong></p>
      <ul>
        <li><strong>P(C|X):</strong> Posterior probability (probability of class C given features X)</li>
        <li><strong>P(X|C):</strong> Likelihood (probability of features X given class C)</li>
        <li><strong>P(C):</strong> Prior probability (probability of class C)</li>
        <li><strong>P(X):</strong> Evidence (probability of features X, acts as normalizing constant)</li>
      </ul>

      <h3>Naive Independence Assumption</h3>
      <p>Assumes features are conditionally independent given the class:</p>
      <ul>
        <li><strong>P(X|C) = P(x₁|C) × P(x₂|C) × ... × P(xₙ|C)</strong></li>
        <li>Simplifies computation significantly</li>
        <li>"Naive" because features are usually dependent in practice</li>
        <li>Works well despite violated assumption</li>
      </ul>

      <h3>Classification</h3>
      <p>Predict the class with highest posterior probability:</p>
      <ul>
        <li><strong>ŷ = argmax_c P(C=c|X)</strong></li>
        <li>Since P(X) is constant, we maximize: P(X|C) × P(C)</li>
        <li>Taking log for numerical stability: log P(C) + Σ log P(xᵢ|C)</li>
      </ul>

      <h3>Types of Naive Bayes</h3>

      <h4>Gaussian Naive Bayes</h4>
      <ul>
        <li>For continuous features</li>
        <li>Assumes features follow Gaussian (normal) distribution</li>
        <li>P(xᵢ|C) = (1/√(2πσ²)) × exp(-(xᵢ-μ)²/(2σ²))</li>
        <li>Learn mean μ and variance σ² for each feature per class</li>
        <li>Best for: Continuous numerical features</li>
      </ul>

      <h4>Multinomial Naive Bayes</h4>
      <ul>
        <li>For discrete count features</li>
        <li>Originally for document classification (word counts)</li>
        <li>P(xᵢ|C) = (count of feature i in class C) / (total count in class C)</li>
        <li>Laplace smoothing prevents zero probabilities</li>
        <li>Best for: Text classification, count data</li>
      </ul>

      <h4>Bernoulli Naive Bayes</h4>
      <ul>
        <li>For binary/boolean features</li>
        <li>Models presence/absence of features</li>
        <li>Explicitly models non-occurrence of features</li>
        <li>Best for: Binary document classification (word present/absent)</li>
      </ul>

      <h3>Laplace Smoothing</h3>
      <p>Prevents zero probabilities for unseen feature values:</p>
      <ul>
        <li><strong>P(xᵢ|C) = (count + α) / (total_count + α × n_features)</strong></li>
        <li>α is smoothing parameter (typically α=1, called "add-one smoothing")</li>
        <li>Ensures no probability is exactly zero</li>
        <li>Critical for avoiding P(X|C) = 0 which eliminates that class</li>
      </ul>

      <h3>Advantages</h3>
      <ul>
        <li>Fast training and prediction (O(nd) complexity)</li>
        <li>Works well with small training sets</li>
        <li>Naturally handles multi-class problems</li>
        <li>Provides probability estimates</li>
        <li>Handles high-dimensional data well (curse of dimensionality less severe)</li>
        <li>Simple to implement and interpret</li>
        <li>Requires minimal hyperparameter tuning</li>
        <li>Excellent for text classification</li>
      </ul>

      <h3>Disadvantages</h3>
      <ul>
        <li>Strong independence assumption (rarely true in practice)</li>
        <li>Poor probability estimates (though classifications can still be good)</li>
        <li>Zero-frequency problem (mitigated by smoothing)</li>
        <li>Correlated features reduce performance</li>
        <li>Sensitive to irrelevant features</li>
        <li>Cannot learn feature interactions</li>
      </ul>

      <h3>Applications</h3>
      <ul>
        <li><strong>Spam filtering:</strong> Classic use case (spam vs ham)</li>
        <li><strong>Text classification:</strong> Sentiment analysis, topic categorization</li>
        <li><strong>Real-time prediction:</strong> Fast prediction makes it suitable for real-time systems</li>
        <li><strong>Recommendation systems:</strong> As baseline or feature</li>
        <li><strong>Medical diagnosis:</strong> Disease prediction from symptoms</li>
      </ul>

      <h3>Best Practices</h3>
      <ul>
        <li>Use appropriate variant for your data type</li>
        <li>Apply Laplace smoothing to avoid zero probabilities</li>
        <li>Remove highly correlated features</li>
        <li>Feature selection improves performance</li>
        <li>Consider as baseline before complex models</li>
        <li>Works better for balanced datasets</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np

# Gaussian Naive Bayes for continuous features
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gaussian NB (for continuous features)
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
y_proba = gnb.predict_proba(X_test)

print("Gaussian Naive Bayes:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Class priors: {gnb.class_prior_}")
print(f"\\nProbability estimates (first 3 samples):")
print(y_proba[:3])

# Cross-validation
cv_scores = cross_val_score(gnb, X, y, cv=5, scoring='accuracy')
print(f"\\nCV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Compare with and without feature scaling
# Note: Gaussian NB doesn't require scaling, but let's see the effect
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

gnb_scaled = GaussianNB()
gnb_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = gnb_scaled.predict(X_test_scaled)

print(f"\\nWith scaling: {accuracy_score(y_test, y_pred_scaled):.4f}")
print(f"Without scaling: {accuracy_score(y_test, y_pred):.4f}")`,
        explanation: 'Demonstrates Gaussian Naive Bayes for continuous features. Shows probability estimates, class priors, and cross-validation. Unlike distance-based methods, NB doesn\'t strictly require feature scaling.'
      },
      {
        language: 'Python',
        code: `from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Text classification example
docs_train = [
    'python is great for data science',
    'machine learning with python',
    'deep learning and neural networks',
    'this movie was terrible',
    'worst film ever made',
    'great acting and cinematography'
]
labels_train = [1, 1, 1, 0, 0, 1]  # 1=positive, 0=negative

docs_test = [
    'python for machine learning',
    'terrible acting in this movie'
]
labels_test = [1, 0]

# Convert text to word count vectors
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(docs_train)
X_test_counts = vectorizer.transform(docs_test)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Feature names: {vectorizer.get_feature_names_out()[:10]}...")

# Multinomial Naive Bayes with Laplace smoothing
mnb = MultinomialNB(alpha=1.0)  # alpha=1 is Laplace smoothing
mnb.fit(X_train_counts, labels_train)

y_pred = mnb.predict(X_test_counts)
y_proba = mnb.predict_proba(X_test_counts)

print(f"\\nPredictions: {y_pred}")
print(f"True labels: {labels_test}")
print(f"\\nProbabilities:")
for i, (doc, proba) in enumerate(zip(docs_test, y_proba)):
    print(f"'{doc}'")
    print(f"  Negative: {proba[0]:.3f}, Positive: {proba[1]:.3f}")

# Feature log probabilities (most important words per class)
feature_names = vectorizer.get_feature_names_out()
log_probs = mnb.feature_log_prob_

print(f"\\nTop 5 words for each class:")
for class_idx in [0, 1]:
    top_features = np.argsort(log_probs[class_idx])[-5:]
    print(f"Class {class_idx}: {[feature_names[i] for i in top_features]}")`,
        explanation: 'Text classification with Multinomial Naive Bayes. Shows how to convert text to count vectors, apply Laplace smoothing, and interpret feature probabilities. Common for spam detection and sentiment analysis.'
      }
    ],
    interviewQuestions: [
      {
        question: 'What is the "naive" assumption in Naive Bayes?',
        answer: 'The "naive" assumption is that all features are **conditionally independent** given the class label. In other words, knowing the value of one feature provides no information about the value of another feature, once you know the class. Mathematically: P(x₁, x₂, ..., xₙ | y) = P(x₁|y) × P(x₂|y) × ... × P(xₙ|y). This simplifies computation dramatically—instead of estimating the joint distribution P(x₁, ..., xₙ | y) which requires exponentially many parameters, you estimate n separate conditional distributions P(xᵢ|y), which is linear in the number of features.\n\nFor example, in spam classification with features [contains "free", contains "winner", length > 100 words], Naive Bayes assumes that whether an email contains "free" is independent of whether it contains "winner," given that we know it\'s spam. In reality, this is often false—spam emails that contain "free" are more likely to also contain "winner" because they come from the same template or scam strategy. The features are correlated. Yet Naive Bayes ignores these correlations and treats each feature independently when computing probabilities.\n\nDespite this strong (and often violated) assumption, Naive Bayes works surprisingly well in practice. The reason is subtle: while the estimated probabilities P(y|x) are often inaccurate (poorly calibrated), the class rankings tend to be correct. For classification, you only need to know which class has the highest probability, not the exact probability values. Even if Naive Bayes incorrectly estimates P(spam|email) = 0.9 when the true value is 0.7, as long as P(spam|email) > P(ham|email), the classification is correct. The independence assumption creates bias in probability estimates but doesn\'t necessarily hurt classification accuracy. This makes Naive Bayes a practical and efficient classifier despite its "naive" simplification.'
      },
      {
        question: 'Explain Bayes\' Theorem and how it\'s used in Naive Bayes classification.',
        answer: '**Bayes\' Theorem** relates conditional probabilities: P(y|x) = [P(x|y) × P(y)] / P(x), where y is the class label and x is the feature vector. In words: the probability of class y given features x (posterior) equals the probability of features x given class y (likelihood) times the prior probability of class y, divided by the probability of features x (evidence). For classification, we want to find the class with maximum posterior probability: argmax_y P(y|x) = argmax_y [P(x|y) × P(y)], dropping P(x) since it\'s constant across classes.\n\nNaive Bayes applies Bayes\' theorem with the independence assumption: P(x|y) = P(x₁, x₂, ..., xₙ|y) = ∏P(xᵢ|y). This transforms the problem into estimating simpler probabilities from training data: **P(y)** (prior) is the fraction of training examples with class y; **P(xᵢ|y)** (likelihood) is estimated differently depending on feature type (Gaussian for continuous, multinomial for counts, Bernoulli for binary). For a test example, compute P(y) × ∏P(xᵢ|y) for each class and predict the class with the highest value.\n\nFor example, classifying an email as spam/ham with features [word counts of "free", "winner", "meeting"]. First, estimate priors: P(spam) = 0.3 (30% of training emails are spam), P(ham) = 0.7. Then estimate likelihoods: P("free"|spam) = 0.8 (word appears in 80% of spam), P("free"|ham) = 0.05. Do this for all words and classes. For a test email with specific word counts, compute: P(spam) × P(words|spam) and P(ham) × P(words|ham). If the first is larger, predict spam.\n\nThe beauty of Bayes\' theorem is it inverts the problem: rather than directly modeling P(y|x) (discriminative), which is hard, we model P(x|y) (generative), which is easier because we can process one feature at a time under the independence assumption. This is why Naive Bayes is a **generative classifier**—it models how data is generated (features given class) and uses Bayes\' theorem to infer classification probabilities. The approach is principled, probabilistically interpretable, and computationally efficient.'
      },
      {
        question: 'What is Laplace smoothing and why is it necessary?',
        answer: 'Laplace smoothing (add-one smoothing) addresses the **zero-probability problem** in Naive Bayes. When estimating P(xᵢ|y) from training data, if a particular feature value never appears with class y in the training set, the estimated probability is 0. This causes problems: when classifying a test example, if any P(xᵢ|y) = 0, the entire product ∏P(xᵢ|y) becomes 0, making P(y|x) = 0, regardless of other features. A single unseen feature-class combination zeroes out the entire prediction, which is overly harsh and leads to poor generalization.\n\nFor example, in spam classification, suppose the word "blockchain" never appeared in any training spam emails. The estimated P("blockchain"|spam) = 0. Now a test spam email about cryptocurrency (containing "blockchain") will be incorrectly classified as ham because P(spam|email) = P(spam) × 0 × ... = 0, even if all other words strongly suggest spam. The model is too confident that spam can\'t contain "blockchain" based on limited training data.\n\n**Laplace smoothing** adds a small count (typically 1) to all feature-class combinations: P(xᵢ|y) = (count(xᵢ, y) + α) / (count(y) + α × k), where α is the smoothing parameter (usually 1), and k is the number of possible values for feature xᵢ. This ensures no probability is exactly 0—even unseen combinations get a small non-zero probability. With α = 1 (add-one smoothing), if "blockchain" never appeared in spam training data (count = 0 out of 1000 spam emails, vocabulary size 10,000), we get: P("blockchain"|spam) = (0 + 1) / (1000 + 1×10000) ≈ 0.0001, a small but non-zero value.\n\nThe amount of smoothing (α) is a hyperparameter: **α = 0** (no smoothing) risks zero probabilities; **α = 1** (Laplace) is standard and works well; **α > 1** (more aggressive smoothing) for very small datasets or high sparsity; **α < 1** (lighter smoothing) when you have ample data. Smoothing is especially critical for text classification where vocabulary is large (10,000+ words) and training data is sparse—many word-class combinations are unseen. Without smoothing, Naive Bayes fails catastrophically on test data containing any new feature values. With smoothing, it gracefully handles unseen data by assigning plausible low probabilities rather than impossible zeros. Other smoothing variants include **Lidstone smoothing** (generalized Laplace with tunable α) and **Good-Turing smoothing** (more sophisticated, adjusts based on frequency-of-frequency statistics), but Laplace is simplest and most commonly used.'
      },
      {
        question: 'When would you use Gaussian vs Multinomial vs Bernoulli Naive Bayes?',
        answer: 'The choice depends on your feature types and data distribution. **Gaussian Naive Bayes** assumes features are continuous and follow a Gaussian (normal) distribution for each class. It estimates P(xᵢ|y) as a Gaussian with class-specific mean μᵢ,y and variance σ²ᵢ,y. Use it for continuous features like height, weight, sensor readings, or measurements. For example, classifying iris flowers based on petal length/width, predicting disease based on lab test values, or anomaly detection with sensor data. It works well when features are roughly normally distributed, but can still perform reasonably even when they\'re not, due to the robustness of Naive Bayes to assumption violations.\n\n**Multinomial Naive Bayes** is designed for discrete count data, typically word counts or term frequencies in text. It models P(xᵢ|y) as a multinomial distribution: features represent counts (how many times each word appears). The model estimates the probability that word i appears in class y. Use it for text classification (spam detection, sentiment analysis, topic classification) with bag-of-words or TF-IDF features, document categorization, or any task with count-based features. For example, an email with word counts [3 "free", 0 "meeting", 1 "winner"] is treated as drawing words from the spam/ham multinomial distributions.\n\n**Bernoulli Naive Bayes** is for binary (presence/absence) features. Each feature xᵢ is 0 or 1, indicating whether a word appears (not how many times). It models P(xᵢ=1|y) and P(xᵢ=0|y), explicitly accounting for absent features. Use it for text classification with binary features (word presence), document filtering where you only care if a term appears, or any binary feature domain (yes/no questions, has_symptom features). Bernoulli is particularly good when documents are short and word frequency is less informative than mere presence.\n\n**Comparison for text**: Multinomial uses counts ("free" appears 3 times matters), Bernoulli uses presence ("free" appears, regardless of count). For long documents, multinomial is typically better (frequency information helps). For short documents (tweets, SMS), Bernoulli may work better since counts are less reliable. In practice, try both on your data via cross-validation. **Gaussian for non-text**: Use Gaussian for numerical features, never for text (word counts aren\'t Gaussian). You can mix: use Gaussian Naive Bayes for numerical features and Multinomial for text features in different classifiers, though you\'d need to combine them carefully (or just apply appropriate preprocessing). Scikit-learn provides GaussianNB, MultinomialNB, and BernoulliNB—experiment to find the best fit for your data.'
      },
      {
        question: 'Why does Naive Bayes work well despite the independence assumption being violated?',
        answer: 'Naive Bayes often performs well in practice even though features are usually correlated, violating the independence assumption. The key insight is that **classification depends on ranking classes, not on accurate probability estimates**. Naive Bayes predicts argmax_y P(y|x), so you only need the relative ordering of P(y|x) across classes to be correct, not the absolute values. The independence assumption creates biased probability estimates (usually over-confident: predicted probabilities too close to 0 or 1), but the ranking of classes often remains correct because the bias affects all classes similarly.\n\nFormally, if features are correlated, the true posterior is P(y|x) ∝ P(x|y) × P(y), while Naive Bayes computes P_NB(y|x) ∝ [∏P(xᵢ|y)] × P(y). These aren\'t equal, but they may be **monotonically related**: if P(y₁|x) > P(y₂|x), then P_NB(y₁|x) > P_NB(y₂|x). When this holds, classifications are identical even though probabilities differ. This is more likely when features are **redundant** (many features provide overlapping information) rather than **complementary**—redundancy makes correlations less impactful because each feature independently points toward the correct class.\n\n**Empirical reasons for success**: (1) **Simplicity helps generalization**: Naive Bayes has few parameters (linear in features, not exponential), reducing overfitting risk. With limited data, a simple biased model often outperforms a complex unbiased model. (2) **Robustness to noise**: Correlations between features might be noisy or inconsistent across train/test, so ignoring them can actually help. (3) **High dimensionality**: In high-dimensional spaces (text with 10,000+ features), the effective amount of correlation is diluted—many features are only weakly correlated with each other. (4) **Class separation**: If classes are well-separated in feature space, even a crude approximation to the decision boundary (via independent features) suffices.\n\n**When Naive Bayes fails**: Strongly dependent features where the dependency is critical for classification (e.g., medical diagnosis where symptom combinations matter more than individual symptoms). If feature A only matters when feature B is present, Naive Bayes misses this interaction. In such cases, use discriminative models (logistic regression captures feature interactions via coefficients; decision trees explicitly model interactions via sequential splits) or relax the independence assumption (Tree-Augmented Naive Bayes, Bayesian Networks). Despite its "naive" assumption, Naive Bayes remains a competitive baseline, especially for high-dimensional sparse data like text, where its simplicity and speed make it highly practical.'
      },
      {
        question: 'What are the advantages of Naive Bayes for text classification?',
        answer: '**Speed and efficiency**: Naive Bayes is one of the fastest machine learning algorithms. Training computes simple frequency counts (O(n×d), linear in samples and features) with no optimization required. Prediction multiplies probabilities, which is also O(d), extremely fast. For large text corpora (millions of documents, 100,000+ vocabulary), Naive Bayes trains and predicts orders of magnitude faster than SVM, neural networks, or ensemble methods. This makes it ideal for real-time systems, prototyping, or resource-constrained environments.\n\n**Handles high dimensionality well**: Text data is inherently high-dimensional (vocabulary size = features), often 10,000-100,000 dimensions. Many algorithms struggle with high dimensions (overfitting, computational cost), but Naive Bayes thrives because: (1) it makes the independence assumption, reducing parameters to O(d) instead of O(d²) or worse; (2) sparsity is natural (most words don\'t appear in most documents), and Naive Bayes handles sparse data efficiently; (3) high dimensions often mean features are less correlated (many weak signals instead of few strong correlated ones), making the naive assumption more reasonable.\n\n**Works with limited training data**: Naive Bayes is a low-variance, high-bias estimator. It makes strong assumptions (independence) and has few parameters, so it doesn\'t require massive training data to generalize well. With just hundreds or thousands of labeled examples, Naive Bayes can achieve decent performance, while deep learning might need millions. This is crucial for domains where labeling is expensive (medical, legal text classification). It also provides a strong baseline: always try Naive Bayes first to establish minimum acceptable performance before trying more complex models.\n\n**Naturally handles multi-class problems**: Extends trivially to many classes (not just binary). Compute P(y|x) for each class y and predict the max, regardless of how many classes exist. Many other algorithms require one-vs-rest or pairwise strategies for multi-class, adding complexity. **Interpretability**: Probabilities have clear meanings; you can inspect P(word|spam) to see which words are indicative of spam. Feature importance is transparent: high P(xᵢ|y) means feature xᵢ strongly indicates class y. This helps debugging and understanding model decisions.\n\n**Robust to irrelevant features**: If many features are noise (common in text with large vocabulary), Naive Bayes is relatively unaffected. Irrelevant words have similar probabilities across classes, contributing little to the classification decision. Other models might overfit to these features. **Online learning**: Easy to update with new data incrementally—just update counts without retraining from scratch. Important for evolving text streams (news, social media). The combination of speed, efficiency with high-dimensional sparse data, and minimal tuning requirements makes Naive Bayes a go-to baseline for text classification tasks like spam filtering, sentiment analysis, and topic categorization.'
      }
    ],
    quizQuestions: [
      {
        id: 'nb-q1',
        question: 'You are building a spam filter and encounter a word in the test email that never appeared in training data. Without Laplace smoothing, what happens?',
        options: [
          'The word is ignored',
          'P(word|spam) = 0, making P(spam|email) = 0, incorrectly ruling out spam',
          'Naive Bayes automatically handles this',
          'The model predicts randomly'
        ],
        correctAnswer: 1,
        explanation: 'Zero probability for any feature makes the entire product P(X|C) = 0, eliminating that class from consideration regardless of other evidence. Laplace smoothing (alpha > 0) prevents this by adding small pseudo-counts.'
      },
      {
        id: 'nb-q2',
        question: 'You have a dataset with highly correlated features. How will this affect Naive Bayes?',
        options: [
          'No effect - Naive Bayes handles correlation well',
          'Performance degrades because independence assumption is violated more severely',
          'Naive Bayes will fail to train',
          'Training time increases significantly'
        ],
        correctAnswer: 1,
        explanation: 'Naive Bayes assumes features are independent. With highly correlated features, the assumption is violated more severely, and the model may over-weight correlated evidence. Consider removing redundant features or using a different algorithm.'
      },
      {
        id: 'nb-q3',
        question: 'Which scenario is BEST suited for Naive Bayes?',
        options: [
          'Small dataset with complex feature interactions',
          'Large text dataset for spam classification with real-time prediction requirements',
          'Image classification with pixel correlations',
          'Time series forecasting'
        ],
        correctAnswer: 1,
        explanation: 'Naive Bayes excels at text classification: (1) handles high-dimensional sparse data well, (2) works with small training sets, (3) very fast prediction, (4) text features are somewhat independent. Poor for images (pixel correlations) or time series (temporal dependencies).'
      }
    ]
  }
};
