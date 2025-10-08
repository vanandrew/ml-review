import { Topic } from '../../../types';

export const linearRegression: Topic = {
  id: 'linear-regression',
  title: 'Linear Regression',
  category: 'classical-ml',
  description: 'Understanding linear regression, the foundation of many machine learning algorithms.',
  content: `
    <h2>Linear Regression: The Foundation of Predictive Modeling</h2>
    <p>Linear regression is the cornerstone of statistical learning and machine learning, modeling the relationship between a continuous target variable and one or more predictor variables using a linear equation. Despite its simplicity, linear regression remains one of the most widely used algorithms in practice due to its interpretability, computational efficiency, and effectiveness when relationships are approximately linear. It serves as both a powerful tool in its own right and a conceptual foundation for understanding more complex models.</p>

    <h3>Mathematical Foundation</h3>
    
    <p><strong>Simple Linear Regression</strong> (one feature):</p>
    <p>$y = \\beta_0 + \\beta_1 x + \\varepsilon$</p>
    <ul>
      <li><strong>y:</strong> Dependent variable (target, response) — what we're predicting</li>
      <li><strong>x:</strong> Independent variable (feature, predictor) — what we use to predict</li>
      <li><strong>β₀:</strong> Intercept (bias) — predicted value when x = 0</li>
      <li><strong>β₁:</strong> Slope (weight, coefficient) — rate of change of y with respect to x; how much y changes for each unit increase in x</li>
      <li><strong>ε:</strong> Error term (residual) — captures noise and unexplained variation</li>
    </ul>
    
    <p>The goal is to find the line that "best fits" the data by choosing optimal β₀ and β₁ values. Geometrically, this is finding the straight line through a 2D scatter plot that comes closest to all data points.</p>

    <p><strong>Multiple Linear Regression</strong> (many features):</p>
    <p>$y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + ... + \\beta_n x_n + \\varepsilon$</p>
    <p>Or in matrix form: <strong>$y = X\\beta + \\varepsilon$</strong></p>
    
    <p>With multiple features, we're fitting a hyperplane in n-dimensional space. Each coefficient β᷈ represents the partial effect of feature xᵢ on the target while holding all other features constant. This is crucial: β₁ tells us how y changes with x₁ <em>after accounting for</em> x₂, x₃, etc.</p>

    <h3>The Cost Function: Mean Squared Error</h3>
    
    <p>To find optimal coefficients, linear regression minimizes the <strong>Mean Squared Error (MSE)</strong>, which measures average squared difference between predictions and actual values:</p>
    
    <p><strong>$\\text{MSE} = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2 = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - (\\beta_0 + \\beta_1 x_{1i} + ... + \\beta_n x_{ni}))^2$</strong></p>
    
    <p>Why squared error? Squaring makes errors positive (so positive and negative errors don't cancel), penalizes large errors more heavily than small ones (quadratic penalty), and makes the math tractable (differentiable everywhere, convex optimization landscape). The method of minimizing sum of squared residuals is called <strong>Ordinary Least Squares (OLS)</strong>.</p>
    
    <p><strong>Residuals</strong> are the differences between observed and predicted values: $r_i = y_i - \\hat{y}_i$. OLS finds coefficients where $\\sum r_i^2$ is minimized. Each data point "pulls" the line toward itself with force proportional to its squared distance from the line.</p>

    <h3>Finding Optimal Coefficients: Two Approaches</h3>
    
    <p><strong>1. Normal Equation (Closed-Form Solution):</strong></p>
    <p><strong>$\\beta = (X^T X)^{-1} X^T y$</strong></p>

    <p>This analytical formula directly computes optimal coefficients without iteration. It's exact, always finds the global optimum (MSE is convex), and requires no hyperparameter tuning. However, computing $(X^T X)^{-1}$ has $O(n^3)$ complexity in the number of features, making it slow for high-dimensional data (>1000 features). Also requires $X^T X$ to be invertible; if features are perfectly collinear or you have more features than samples (p > n), the matrix is singular. Modern libraries use pseudo-inverse to handle this.</p>
    
    <p><strong>2. Gradient Descent (Iterative Optimization):</strong></p>
    <p>Iteratively update coefficients in the direction that decreases MSE:</p>
    <ul>
      <li>Initialize β randomly</li>
      <li>Compute gradient $\\nabla \\text{MSE} = -\\frac{2}{n} X^T(y - X\\beta)$</li>
      <li>Update $\\beta := \\beta - \\alpha \\nabla \\text{MSE}$ ($\\alpha$ is learning rate)</li>
      <li>Repeat until convergence</li>
    </ul>
    
    <p>Gradient descent scales much better for large datasets: O(knd) where k is iterations, n is samples, d is features. Each iteration is fast, and mini-batch or stochastic variants process subsets of data, enabling online learning. Requires tuning learning rate, but modern optimizers (Adam, RMSprop) handle this automatically.</p>
    
    <p><strong>When to use which:</strong> Normal equation for small/medium datasets (<10k samples, <1k features) where you want exact solution. Gradient descent for large-scale problems, online learning, or when using regularization variants.</p>

    <h3>Key Assumptions of Linear Regression</h3>
    
    <p>Linear regression makes several assumptions that, when violated, can compromise model reliability:</p>
    
    <p><strong>1. Linearity:</strong> The relationship between predictors and target is linear. Non-linear relationships lead to systematic errors. <em>Diagnostic:</em> Plot residuals vs predicted values; patterns indicate non-linearity. <em>Solution:</em> Add polynomial features (x²), transform variables (log, sqrt), or use non-linear models.</p>
    
    <p><strong>2. Independence:</strong> Observations are independent of each other. Violations occur in time-series (autocorrelation), clustered data (patients from same hospital), or spatial data. <em>Diagnostic:</em> Durbin-Watson test for autocorrelation. <em>Solution:</em> Use time-series models, mixed-effects models, or account for clustering structure.</p>
    
    <p><strong>3. Homoscedasticity:</strong> Residuals have constant variance across all predictor levels (uniform spread). <em>Heteroscedasticity</em> (non-constant variance) means the model is more uncertain for some predictions than others, violating standard error calculations. <em>Diagnostic:</em> Plot residuals vs fitted values; funnel shape indicates heteroscedasticity. <em>Solution:</em> Transform target (log), use weighted least squares, or robust standard errors.</p>
    
    <p><strong>4. Normality of Residuals:</strong> For valid statistical inference (p-values, confidence intervals), residuals should be approximately normally distributed. Less critical for prediction or large samples (Central Limit Theorem helps). <em>Diagnostic:</em> Q-Q plot, Shapiro-Wilk test. <em>Solution:</em> Transform target variable or use non-parametric methods.</p>
    
    <p><strong>5. No Multicollinearity:</strong> Predictors should not be highly correlated with each other. Multicollinearity makes coefficient estimates unstable, inflates standard errors, and complicates interpretation. <em>Diagnostic:</em> Variance Inflation Factor (VIF); VIF > 10 is problematic. <em>Solution:</em> Remove redundant features, use PCA, or apply Ridge regularization.</p>

    <h3>Multicollinearity: A Critical Issue</h3>
    
    <p><strong>What it is:</strong> High correlation between predictor variables. Example: predicting house price using both "square feet" and "number of rooms" (correlated — bigger houses have more rooms).</p>
    
    <p><strong>Why it's problematic:</strong> When features are correlated, the model can't distinguish their individual effects. There are infinitely many coefficient combinations that fit the data similarly well. Small data changes cause large coefficient swings, even sign flips. Standard errors inflate, making it hard to determine significance. Coefficients become unreliable for interpretation.</p>
    
    <p><strong>Detection:</strong> Calculate VIF for each feature: $\\text{VIF} = \\frac{1}{1 - R_i^2}$, where $R_i^2$ is R² from regressing feature i on all other features. VIF = 1 means no correlation, VIF > 5-10 indicates problematic multicollinearity. Also check correlation matrix for $|r| > 0.8$-$0.9$.</p>
    
    <p><strong>Solutions:</strong> Remove one of each correlated pair, combine correlated features (e.g., "total living space"), use PCA to create uncorrelated components, or apply Ridge regularization (L2 penalty handles multicollinearity by shrinking coefficients).</p>

    <h3>Polynomial Regression: Extending Linearity</h3>
    
    <p>Polynomial regression extends linear regression to model non-linear relationships by adding polynomial features:</p>
    <p>$y = \\beta_0 + \\beta_1 x + \\beta_2 x^2 + \\beta_3 x^3 + ... + \\beta_d x^d$</p>
    
    <p>Despite modeling non-linear relationships, this is still "linear" in the parameters β, so we can use OLS. The model is linear in the coefficients but non-linear in the features. Create new features (x², x³, etc.) and apply standard linear regression.</p>
    
    <p><strong>Caution:</strong> Higher-degree polynomials (d > 3-4) easily overfit, especially at data boundaries. Use cross-validation to select polynomial degree. Regularization (Ridge, Lasso) is essential for high-degree polynomials.</p>

    <h3>Feature Engineering for Linear Regression</h3>
    
    <p><strong>Categorical Variables:</strong> Convert categories to numerical format using one-hot encoding (dummy variables). For a categorical feature with k categories, create k-1 binary features (drop one to avoid multicollinearity — the "dummy variable trap"). Example: Color {Red, Blue, Green} → is_Blue, is_Green (Red is reference category when both are 0).</p>
    
    <p><strong>Feature Scaling/Standardization:</strong> Linear regression coefficients depend on feature scales. Standardizing (mean=0, std=1) makes coefficients comparable and helps gradient descent converge faster. Not required for prediction accuracy with normal equation, but highly recommended for gradient descent and regularization methods.</p>
    
    <p><strong>Interaction Terms:</strong> Capture combined effects of features: $y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + \\beta_3(x_1 \\times x_2)$. The interaction term $\\beta_3(x_1 \\times x_2)$ models how the effect of $x_1$ depends on the value of $x_2$. Example: advertising spend and product quality might have synergistic effects on sales.</p>

    <h3>Evaluation Metrics</h3>
    
    <p><strong>R² (Coefficient of Determination):</strong> Proportion of variance explained by the model. R² = 1 - (SSres / SStot), ranges from -∞ to 1. R²=1 is perfect, R²=0 means no better than predicting the mean, negative R² means worse than the baseline. Limitation: always increases with more features, even if they're random.</p>
    
    <p><strong>Adjusted R²:</strong> Penalizes model complexity, only increases if new features genuinely improve fit: $\\text{Adjusted } R^2 = 1 - \\frac{(1-R^2)(n-1)}{n-p-1}$. Use this for comparing models with different numbers of features.</p>
    
    <p><strong>RMSE, MAE:</strong> Directly measure prediction error in target units. RMSE penalizes large errors more (squared), MAE treats all errors equally. Use MAE for robustness to outliers, RMSE when large errors are particularly costly.</p>

    <h3>Advantages of Linear Regression</h3>
    <ul>
      <li><strong>Interpretability:</strong> Coefficients directly show feature effects: "each additional bedroom increases price by $15k." Invaluable for explanation to stakeholders, scientific understanding, and regulatory compliance.</li>
      <li><strong>Computational Efficiency:</strong> Training and prediction are extremely fast, enabling real-time systems and large-scale applications.</li>
      <li><strong>Statistical Properties:</strong> Well-understood inference tools (confidence intervals, hypothesis tests, p-values) with solid theoretical foundations.</li>
      <li><strong>No Hyperparameters:</strong> Basic linear regression has no hyperparameters to tune (though regularized variants do).</li>
      <li><strong>Works Well with Limited Data:</strong> Simple models are less prone to overfitting when data is scarce.</li>
      <li><strong>Established Diagnostic Tools:</strong> Decades of research provide comprehensive methods for assumption checking and model diagnosis.</li>
    </ul>

    <h3>Limitations and When to Avoid</h3>
    <ul>
      <li><strong>Assumes Linearity:</strong> Performs poorly when true relationships are non-linear (exponential growth, threshold effects, interactions).</li>
      <li><strong>Sensitive to Outliers:</strong> Squared error heavily weights outliers, which can skew the fitted line. Robust regression methods (RANSAC, Huber) can help.</li>
      <li><strong>Requires Careful Feature Engineering:</strong> Must manually encode categorical variables, create interactions, add polynomial terms. Tree-based models handle these automatically.</li>
      <li><strong>Multicollinearity Issues:</strong> Correlated features cause instability; regularization helps but adds complexity.</li>
      <li><strong>Limited Capacity:</strong> Cannot capture complex non-linear patterns without extensive feature engineering.</li>
    </ul>

    <h3>Variants and Extensions</h3>
    <ul>
      <li><strong>Ridge Regression:</strong> Adds L2 penalty (Σβ²) to handle multicollinearity and prevent overfitting. Shrinks coefficients toward zero.</li>
      <li><strong>Lasso Regression:</strong> Adds L1 penalty (Σ|β|) for automatic feature selection. Drives some coefficients to exactly zero.</li>
      <li><strong>Elastic Net:</strong> Combines L1 and L2 penalties, balancing Ridge's stability with Lasso's sparsity.</li>
      <li><strong>Robust Regression:</strong> Uses loss functions less sensitive to outliers (Huber loss, quantile regression).</li>
      <li><strong>Weighted Least Squares:</strong> Gives different weights to observations, handling heteroscedasticity or emphasizing certain data points.</li>
    </ul>

    <h3>Practical Recommendations</h3>
    <ul>
      <li><strong>Start Simple:</strong> Use linear regression as a baseline before trying complex models. It often performs surprisingly well and provides interpretable insights.</li>
      <li><strong>Check Assumptions:</strong> Always perform residual analysis to validate assumptions. Violations guide model improvements.</li>
      <li><strong>Use Regularization:</strong> For high-dimensional data, always use Ridge, Lasso, or Elastic Net to prevent overfitting.</li>
      <li><strong>Scale Features:</strong> Standardize features for gradient descent and when using regularization.</li>
      <li><strong>Cross-Validate:</strong> Use k-fold CV for reliable performance estimates, especially when tuning regularization strength.</li>
      <li><strong>Beware P-Values:</strong> With large datasets, everything becomes "statistically significant." Focus on effect sizes and practical significance.</li>
      <li><strong>Handle Outliers Carefully:</strong> Investigate before removing. They might be legitimate rare events or indicate model misspecification.</li>
    </ul>

    <h3>Visual Understanding</h3>
    <p>Imagine a scatter plot with data points scattered around. Linear regression finds the "best-fit" line through these points—the line that minimizes the vertical distances (residuals) from points to the line. In 2D, you see a straight line through the cloud of points. In 3D, it's a plane cutting through the data. For higher dimensions, it's a hyperplane that you can't visualize but follows the same principle.</p>
    
    <p><strong>Key visualizations to understand:</strong></p>
    <ul>
      <li><strong>Fitted line plot:</strong> Original data points (scatter) with the regression line overlaid. Points above the line have positive residuals, below have negative residuals.</li>
      <li><strong>Residual plot:</strong> Residuals (y - ŷ) on y-axis vs predicted values (ŷ) on x-axis. Should show random scatter with no pattern. Patterns indicate assumption violations (non-linearity, heteroscedasticity).</li>
      <li><strong>Q-Q plot:</strong> Quantiles of residuals vs quantiles of normal distribution. Points should fall on diagonal line if residuals are normally distributed.</li>
      <li><strong>Scale-location plot:</strong> Square root of standardized residuals vs fitted values. Check for constant variance (horizontal line with equal spread).</li>
    </ul>

    <h3>Worked Example: Simple Linear Regression by Hand</h3>
    <p>Let's fit a line to predict house price (y) from size in 1000 sq ft (x) using 4 data points:</p>
    <table>
      <tr><th>Size (x)</th><th>Price (y)</th></tr>
      <tr><td>1.0</td><td>100k</td></tr>
      <tr><td>2.0</td><td>200k</td></tr>
      <tr><td>3.0</td><td>250k</td></tr>
      <tr><td>4.0</td><td>300k</td></tr>
    </table>
    
    <p><strong>Step 1: Calculate means</strong></p>
    <ul>
      <li>x̄ = (1 + 2 + 3 + 4)/4 = 2.5</li>
      <li>ȳ = (100 + 200 + 250 + 300)/4 = 212.5</li>
    </ul>
    
    <p><strong>Step 2: Calculate slope β₁</strong></p>
    <p>$\\beta_1 = \\frac{\\sum(x_i - \\bar{x})(y_i - \\bar{y})}{\\sum(x_i - \\bar{x})^2}$</p>
    <ul>
      <li>Numerator: (1-2.5)(100-212.5) + (2-2.5)(200-212.5) + (3-2.5)(250-212.5) + (4-2.5)(300-212.5)</li>
      <li>= (-1.5)(-112.5) + (-0.5)(-12.5) + (0.5)(37.5) + (1.5)(87.5)</li>
      <li>= 168.75 + 6.25 + 18.75 + 131.25 = 325</li>
      <li>Denominator: (1-2.5)² + (2-2.5)² + (3-2.5)² + (4-2.5)²</li>
      <li>= 2.25 + 0.25 + 0.25 + 2.25 = 5</li>
      <li><strong>β₁ = 325/5 = 65</strong></li>
    </ul>
    
    <p><strong>Step 3: Calculate intercept β₀</strong></p>
    <ul>
      <li>β₀ = ȳ - β₁x̄ = 212.5 - 65(2.5) = 212.5 - 162.5 = 50</li>
    </ul>
    
    <p><strong>Final model: ŷ = 50 + 65x</strong></p>
    <p>Interpretation: Base price is $50k (intercept), and each 1000 sq ft adds $65k (slope).</p>
    
    <p><strong>Step 4: Make predictions</strong></p>
    <ul>
      <li>For x = 2.5 (2500 sq ft): ŷ = 50 + 65(2.5) = 212.5k ✓ (matches mean, as expected)</li>
      <li>For x = 5.0 (5000 sq ft): ŷ = 50 + 65(5) = 375k (extrapolation)</li>
    </ul>
    
    <p><strong>Step 5: Calculate R²</strong></p>
    <p>Predictions: [115, 180, 245, 310]. Residuals: [-15, +20, +5, -10]</p>
    <ul>
      <li>SSres = 15² + 20² + 5² + 10² = 225 + 400 + 25 + 100 = 750</li>
      <li>SStot = (100-212.5)² + (200-212.5)² + (250-212.5)² + (300-212.5)² = 12656.25 + 156.25 + 1406.25 + 7656.25 = 21875</li>
      <li><strong>R² = 1 - (750/21875) = 1 - 0.0343 = 0.9657 ≈ 96.6%</strong></li>
    </ul>
    <p>The model explains 96.6% of variance in price—excellent fit!</p>

    <h3>Common Mistakes to Avoid</h3>
    <ul>
      <li><strong>❌ Forgetting to check assumptions:</strong> Always create residual plots. Many use linear regression blindly without verifying linearity, independence, homoscedasticity, or normality assumptions. Violations lead to unreliable predictions and invalid p-values.</li>
      <li><strong>❌ Ignoring multicollinearity:</strong> When features are highly correlated (VIF > 10), coefficients become unstable and uninterpretable. Check VIF and remove or combine correlated features, or use Ridge regression.</li>
      <li><strong>❌ Over-interpreting R²:</strong> High R² doesn't mean good model—could still violate assumptions or overfit. R² always increases with more features, even random ones. Use adjusted R² for model comparison.</li>
      <li><strong>❌ Extrapolating beyond data range:</strong> Linear regression is unreliable outside the range of training data. Predicting house price for 10,000 sq ft when max training example is 4,000 sq ft is dangerous.</li>
      <li><strong>❌ Confusing correlation with causation:</strong> Regression coefficients show association, not causation. Controlling for confounders requires causal inference techniques.</li>
      <li><strong>❌ Using with categorical targets:</strong> Linear regression is for continuous targets. For binary outcomes, use logistic regression; for counts, use Poisson regression.</li>
      <li><strong>❌ Keeping outliers without investigation:</strong> Outliers heavily influence the fitted line. Investigate before removing—they might be data errors or legitimate rare events revealing model misspecification.</li>
      <li><strong>❌ Not standardizing features for regularization:</strong> When using Ridge/Lasso, features must be standardized. Otherwise, regularization penalizes large-scale features more than small-scale ones.</li>
    </ul>

    <h3>Summary</h3>
    <p>Linear regression is the workhorse of predictive modeling, offering an optimal blend of simplicity, interpretability, and effectiveness for linear relationships. While limited to modeling linear patterns without extensive feature engineering, its transparency and solid statistical foundation make it indispensable for both explanation and prediction. Master linear regression deeply—understanding its assumptions, diagnostics, and limitations—as it forms the conceptual basis for understanding more advanced models. In practice, start with linear regression to establish a performance baseline and gain insights, then consider more complex models only if the accuracy gains justify the loss of interpretability.</p>
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
};
