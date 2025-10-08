import { Topic } from '../../../types';

export const logisticRegression: Topic = {
  id: 'logistic-regression',
  title: 'Logistic Regression',
  category: 'classical-ml',
  description: 'Learn about logistic regression for binary and multiclass classification problems.',
  content: `
    <h2>Logistic Regression: Classification Through Probability</h2>
    <p>Despite its name, logistic regression is a classification algorithm, not a regression algorithm. It predicts the probability that an instance belongs to a particular class, making it one of the most widely used methods for binary classification. Logistic regression extends the linear model framework to classification by applying a non-linear transformation (the sigmoid function) that converts continuous scores into probabilities, enabling principled probabilistic predictions with solid statistical foundations.</p>

    <h3>From Linear to Logistic: The Sigmoid Function</h3>
    
    <p>Linear regression outputs unbounded continuous values: $y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + ...$ This is problematic for classification where we need probabilities (bounded between 0 and 1). Logistic regression solves this by applying the <strong>sigmoid (logistic) function</strong> to the linear combination:</p>

    <p><strong>$\\sigma(z) = \\frac{1}{1 + e^{-z}}$</strong></p>
    <p>Where $z = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + ... + \\beta_n x_n$</p>
    
    <p>The sigmoid creates a smooth S-shaped curve that:</p>
    <ul>
      <li>Maps any real number (-∞ to +∞) to (0, 1)</li>
      <li>Outputs σ(0) = 0.5 at z = 0 (the decision boundary)</li>
      <li>Approaches 1 as z → +∞ (strong positive class signal)</li>
      <li>Approaches 0 as z → -∞ (strong negative class signal)</li>
      <li>Is symmetric around 0.5: σ(-z) = 1 - σ(z)</li>
      <li>Has a nice derivative: σ'(z) = σ(z)(1 - σ(z)), simplifying optimization</li>
    </ul>
    
    <p>The output P(y=1|X) = σ(z) is interpreted as the probability of the positive class given features X. For binary classification, P(y=0|X) = 1 - P(y=1|X).</p>

    <h3>The Logit: Log-Odds Interpretation</h3>
    
    <p>Logistic regression is called "logistic" because it models the <strong>logit</strong> (log-odds) as a linear function of features. The <strong>odds</strong> of an event are the ratio of probability of success to probability of failure:</p>
    
    <p><strong>$\\text{Odds} = \\frac{P(y=1)}{P(y=0)} = \\frac{P(y=1)}{1 - P(y=1)}$</strong></p>

    <p>The <strong>log-odds</strong> (logit) is the natural logarithm of the odds:</p>
    <p><strong>$\\text{logit}(p) = \\log\\left(\\frac{p}{1-p}\\right) = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + ... + \\beta_n x_n$</strong></p>

    <p>This transformation converts probabilities (bounded, non-linear) into log-odds (unbounded, linear), allowing us to use linear modeling techniques. The relationship is invertible: given log-odds z, we recover probability via the sigmoid $p = \\sigma(z) = \\frac{1}{1 + e^{-z}}$.</p>
    
    <p><strong>Interpreting Coefficients:</strong> A coefficient $\\beta_j$ represents the change in log-odds for a one-unit increase in $x_j$, holding other features constant. Equivalently, $e^{\\beta_j}$ is the odds ratio: how much the odds multiply when $x_j$ increases by 1. For example, $\\beta_1 = 0.5$ means a one-unit increase in $x_1$ multiplies the odds by $e^{0.5} \\approx 1.65$ (65% increase in odds).</p>

    <h3>Cost Function: Binary Cross-Entropy (Log Loss)</h3>
    
    <p>Logistic regression minimizes <strong>log loss</strong> (binary cross-entropy):</p>
    <p><strong>$L = -\\frac{1}{n} \\sum_i [y_i \\log(p_i) + (1-y_i) \\log(1-p_i)]$</strong></p>

    <p>Where $y_i \\in \\{0,1\\}$ is the true label and $p_i$ is the predicted probability for sample i. For a single sample:</p>
    <ul>
      <li>If $y=1$ (positive class): $L = -\\log(p)$. This is 0 when $p=1$ (correct, confident), $\\infty$ when $p=0$ (wrong, confident)</li>
      <li>If $y=0$ (negative class): $L = -\\log(1-p)$. This is 0 when $p=0$ (correct, confident), $\\infty$ when $p=1$ (wrong, confident)</li>
    </ul>
    
    <p>This asymmetric penalty heavily punishes confident incorrect predictions. Being 99% confident and wrong incurs much more loss than being 55% confident and wrong, encouraging calibrated probability estimates.</p>
    
    <p><strong>Why not MSE?</strong> Mean Squared Error with sigmoid activation creates a <em>non-convex</em> loss surface with many local minima, making optimization unreliable. Log loss with sigmoid produces a <em>convex</em> loss surface, guaranteeing gradient descent converges to the global optimum. Log loss also naturally arises from maximum likelihood estimation (MLE) for Bernoulli distributions, providing solid statistical foundations.</p>

    <h3>Training: Gradient Descent</h3>
    
    <p>Unlike linear regression which has a closed-form solution, logistic regression requires iterative optimization. The gradient of log loss with respect to the linear combination z is remarkably simple:</p>
    
    <p><strong>$\\frac{\\partial L}{\\partial \\beta_j} = \\frac{1}{n} \\sum_i (p_i - y_i)x_{ij}$</strong></p>

    <p>This is the same form as linear regression! The gradient is the average of errors $(p_i - y_i)$ weighted by features. We update weights using gradient descent:</p>
    <ul>
      <li>Initialize $\\beta$ randomly or to zeros</li>
      <li>Compute predictions: $p = \\sigma(X\\beta)$</li>
      <li>Compute gradient: $\\nabla L = \\frac{1}{n}X^T(p - y)$</li>
      <li>Update: $\\beta := \\beta - \\alpha \\nabla L$ ($\\alpha$ is learning rate)</li>
      <li>Repeat until convergence</li>
    </ul>
    
    <p>Convergence is typically fast due to the convex loss surface. Variants like stochastic gradient descent (SGD) or mini-batch SGD scale to large datasets.</p>

    <h3>Decision Boundary</h3>
    
    <p>The <strong>decision boundary</strong> is where $P(y=1|X) = 0.5$, which occurs when $z = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + ... = 0$. This defines a hyperplane in feature space:</p>
    <ul>
      <li>For 2D (two features): $z = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 = 0$ is a line</li>
      <li>For 3D: $z = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + \\beta_3 x_3 = 0$ is a plane</li>
      <li>For n-D: a hyperplane dividing the space</li>
    </ul>
    
    <p>Points on one side have P(y=1|X) > 0.5 (predicted as class 1), points on the other side have P(y=1|X) < 0.5 (predicted as class 0). The boundary is <em>linear</em> in feature space, which is why logistic regression is a linear classifier despite the non-linear sigmoid transformation.</p>
    
    <p>To create non-linear decision boundaries, add polynomial features (x², x₁x₂, etc.) or use kernel methods, just as with linear regression. The linearity in log-odds space becomes non-linearity in probability space.</p>

    <h3>Multiclass Extension: Softmax Regression</h3>
    
    <p><strong>Multinomial Logistic Regression</strong> (softmax regression) extends binary logistic regression to K > 2 classes. Instead of one sigmoid output, we have K outputs using the <strong>softmax function</strong>:</p>
    
    <p><strong>$P(y=k|X) = \\frac{e^{z_k}}{\\sum_j e^{z_j}}$</strong> for j = 1 to K</p>

    <p>Where $z_k = \\beta_0^{(k)} + \\beta_1^{(k)}x_1 + \\beta_2^{(k)}x_2 + ...$ Each class has its own weight vector $\\beta^{(k)}$. Softmax ensures:</p>
    <ul>
      <li>All probabilities are between 0 and 1</li>
      <li>Probabilities sum to 1 across all classes</li>
      <li>Larger z_k leads to higher P(y=k|X)</li>
      <li>Reduces to sigmoid for K=2 (binary case)</li>
    </ul>
    
    <p>The loss function becomes <strong>categorical cross-entropy</strong>:</p>
    <p><strong>$L = -\\frac{1}{n} \\sum_i \\sum_k y_{ik} \\log(p_{ik})$</strong></p>

    <p>Where $y_{ik}$ is 1 if sample i belongs to class k, 0 otherwise (one-hot encoding). For a sample with true class k, this reduces to $-\\log(p_{ik})$, heavily penalizing low probability for the correct class.</p>
    
    <p><strong>Alternative: One-vs-Rest (OvR):</strong> Train K binary classifiers, each distinguishing one class from all others. At prediction, run all classifiers and choose the class with highest probability. Simpler to implement but less principled than softmax (probabilities may not sum to 1).</p>

    <h3>Regularization: L1 and L2</h3>
    
    <p>Like linear regression, logistic regression benefits from regularization to prevent overfitting, especially with many features or limited data:</p>
    
    <p><strong>L2 (Ridge):</strong> $L = \\text{Log Loss} + \\lambda \\sum \\beta_j^2$</p>
    <ul>
      <li>Shrinks all coefficients toward zero</li>
      <li>Handles multicollinearity</li>
      <li>No feature selection (all features retained)</li>
      <li>Standard choice for most applications</li>
    </ul>

    <p><strong>L1 (Lasso):</strong> $L = \\text{Log Loss} + \\lambda \\sum |\\beta_j|$</p>
    <ul>
      <li>Drives some coefficients to exactly zero</li>
      <li>Performs automatic feature selection</li>
      <li>Creates sparse models (fewer features)</li>
      <li>Useful for high-dimensional data with many irrelevant features</li>
    </ul>

    <p><strong>Elastic Net:</strong> $L = \\text{Log Loss} + \\lambda_1 \\sum |\\beta_j| + \\lambda_2 \\sum \\beta_j^2$</p>
    <ul>
      <li>Combines L1 and L2 benefits</li>
      <li>More stable than pure L1 with correlated features</li>
    </ul>
    
    <p>The regularization parameter λ (or C = 1/λ in sklearn) controls the strength: larger λ means more regularization (simpler model), smaller λ means less regularization (more complex model). Use cross-validation to select optimal λ.</p>

    <h3>Key Assumptions</h3>
    
    <p><strong>1. Linearity of Log-Odds:</strong> The log-odds must be a linear function of features. Non-linear relationships require feature engineering (polynomials, interactions) or non-linear models.</p>
    
    <p><strong>2. Independence of Observations:</strong> Samples must be independent. Violations occur with repeated measures, clustered data, or time series. Use mixed-effects models or GEE for dependent data.</p>
    
    <p><strong>3. No Perfect Multicollinearity:</strong> Features shouldn't be perfect linear combinations of each other. High (but not perfect) multicollinearity inflates standard errors. Use regularization or remove redundant features.</p>
    
    <p><strong>4. Large Sample Size:</strong> Need sufficient data for reliable estimates, especially for the minority class. Rule of thumb: at least 10-15 events per predictor variable for the less frequent class. With small samples or rare events, use penalized likelihood methods.</p>
    
    <p><strong>Unlike linear regression:</strong> Logistic regression does NOT assume normality of features or residuals, homoscedasticity, or continuous features. Categorical features are fine. These relaxed assumptions make logistic regression broadly applicable.</p>

    <h3>Probability Calibration</h3>
    
    <p>Logistic regression trained with log loss produces <strong>well-calibrated probabilities</strong>: if the model predicts 70% probability for many samples, about 70% of them should actually be positive. This is valuable for decision-making under uncertainty (e.g., medical diagnosis where probability matters, not just classification).</p>
    
    <p>To assess calibration, create a <strong>calibration plot</strong>: bin predictions by probability (0-0.1, 0.1-0.2, ..., 0.9-1.0) and plot predicted probability vs actual frequency. Perfectly calibrated models lie on the diagonal. Other classifiers (like SVMs or some tree models) may achieve high accuracy but poor calibration—their probability estimates are unreliable.</p>

    <h3>Advantages</h3>
    <ul>
      <li><strong>Probabilistic Output:</strong> Provides probability estimates, enabling risk-based decision making and calibrated uncertainty quantification</li>
      <li><strong>Interpretability:</strong> Coefficients show log-odds effects; odds ratios (e^β) are intuitive for stakeholders</li>
      <li><strong>Computational Efficiency:</strong> Fast training (especially with optimized solvers) and prediction</li>
      <li><strong>No Hyperparameters:</strong> Unregularized version has no hyperparameters; regularized version has only λ</li>
      <li><strong>Solid Statistical Foundation:</strong> Maximum likelihood estimation, confidence intervals, hypothesis tests</li>
      <li><strong>Works with Limited Data:</strong> Simpler than complex models, less prone to overfitting with small datasets</li>
      <li><strong>Well-Calibrated:</strong> Predicted probabilities match true frequencies (with proper training)</li>
      <li><strong>Extends Naturally:</strong> Multiclass extension (softmax) is straightforward and principled</li>
    </ul>

    <h3>Disadvantages</h3>
    <ul>
      <li><strong>Assumes Linearity:</strong> Decision boundary is linear in feature space; struggles with complex non-linear patterns without feature engineering</li>
      <li><strong>Feature Engineering Required:</strong> Must manually create polynomial terms, interactions, or transformations for non-linear relationships</li>
      <li><strong>Sensitive to Outliers:</strong> Extreme feature values can disproportionately influence coefficients</li>
      <li><strong>Requires Large Samples:</strong> Needs sufficient data per feature, especially for minority class in imbalanced problems</li>
      <li><strong>Can Underfit:</strong> Limited capacity to capture complex patterns compared to ensemble methods or neural networks</li>
      <li><strong>Multicollinearity Issues:</strong> Correlated features cause unstable coefficients (though regularization helps)</li>
    </ul>

    <h3>Practical Recommendations</h3>
    <ul>
      <li><strong>Always Use Regularization:</strong> L2 by default; L1 if you need feature selection. Tune λ via cross-validation</li>
      <li><strong>Standardize Features:</strong> Put features on the same scale (mean=0, std=1) for comparable coefficients and faster convergence</li>
      <li><strong>Check Assumptions:</strong> Verify log-odds linearity, inspect for multicollinearity (VIF), ensure sufficient sample size</li>
      <li><strong>Handle Class Imbalance:</strong> Use class weights, oversampling (SMOTE), undersampling, or adjust decision threshold</li>
      <li><strong>Interpret via Odds Ratios:</strong> Exponentiate coefficients (e^β) for easier communication to non-technical audiences</li>
      <li><strong>Validate Calibration:</strong> Check calibration plots; recalibrate if needed (Platt scaling, isotonic regression)</li>
      <li><strong>Start Simple:</strong> Use logistic regression as a baseline before trying complex models—it often performs surprisingly well</li>
      <li><strong>Feature Engineering:</strong> Create polynomial terms, interactions, or use domain knowledge to capture non-linearities</li>
    </ul>

    <h3>Visual Understanding</h3>
    <p>Picture a 2D scatter plot with two classes (red and blue points). Logistic regression draws an S-shaped curve (sigmoid) that smoothly transitions from 0 to 1 as you move across the feature space. The decision boundary (where probability = 0.5) is a straight line (in 2D) or hyperplane (higher dimensions) that separates the classes. Unlike linear regression's fitted line, logistic regression's curve asymptotes at 0 and 1, never going below/above these probability bounds.</p>
    
    <p><strong>Key visualizations to understand:</strong></p>
    <ul>
      <li><strong>Decision boundary plot:</strong> 2D feature space with colored regions (blue = predict class 0, red = predict class 1). The boundary line shows where P(y=1) = 0.5. Points closer to the boundary have uncertain predictions (P ≈ 0.5), while points far from it have confident predictions (P near 0 or 1).</li>
      <li><strong>Sigmoid curve:</strong> S-shaped curve showing how probability changes with the linear combination z = β₀ + β₁x. At z=0, P=0.5; as z→∞, P→1; as z→-∞, P→0.</li>
      <li><strong>Calibration plot:</strong> Predicted probabilities (x-axis) vs actual frequency (y-axis) in bins. Perfectly calibrated models lie on the diagonal—if model predicts 70% for 100 samples, about 70 should be positive.</li>
      <li><strong>ROC curve:</strong> True Positive Rate vs False Positive Rate at different classification thresholds. Area under curve (AUC) measures overall classification ability.</li>
    </ul>

    <h3>Common Mistakes to Avoid</h3>
    <ul>
      <li><strong>❌ Using MSE as loss function:</strong> Mean Squared Error creates non-convex optimization with sigmoid, leading to local minima. Always use log loss (binary cross-entropy) for logistic regression.</li>
      <li><strong>❌ Forgetting to standardize features:</strong> While predictions remain valid without scaling, gradient descent converges much faster with standardized features. Regularization also requires scaling to penalize features equally.</li>
      <li><strong>❌ Misinterpreting coefficients as probabilities:</strong> Coefficients represent log-odds, not probabilities. A coefficient of 0.5 means odds multiply by e^0.5 ≈ 1.65, not that probability increases by 0.5.</li>
      <li><strong>❌ Ignoring class imbalance:</strong> With 95% class 0 and 5% class 1, the model might predict everything as class 0 (95% accuracy but useless). Use class_weight='balanced', adjust decision threshold, or use resampling techniques.</li>
      <li><strong>❌ Not checking for separation:</strong> If classes are perfectly separable, maximum likelihood diverges (coefficients → ±∞). Use regularization (L2 penalty) to prevent this.</li>
      <li><strong>❌ Expecting linear boundaries to fit non-linear data:</strong> Logistic regression has linear decision boundaries. For circles-within-circles or XOR patterns, add polynomial features or use non-linear models (kernels, trees, neural networks).</li>
      <li><strong>❌ Over-relying on default threshold (0.5):</strong> The 0.5 threshold is arbitrary. For imbalanced data or when false positives/negatives have different costs, tune the threshold based on business requirements.</li>
      <li><strong>❌ Treating probability outputs as calibrated by default:</strong> While logistic regression generally produces well-calibrated probabilities, always verify with calibration plots, especially after regularization or with small datasets.</li>
    </ul>

    <h3>Connection to Neural Networks</h3>
    <p><strong>Important insight:</strong> Logistic regression is a single-layer neural network with one neuron and sigmoid activation. The linear combination z = β₀ + β₁x₁ + β₂x₂ + ... is the neuron's pre-activation, and σ(z) is the activation function output. This makes logistic regression the perfect bridge to understanding deep learning—concepts like log loss, gradient descent, and weight updates carry directly to neural networks, just with more layers and neurons.</p>

    <h3>Summary</h3>
    <p>Logistic regression is the foundational algorithm for binary and multiclass classification, extending linear models to probabilistic prediction through the sigmoid transformation. Its combination of interpretability, computational efficiency, well-calibrated probabilities, and solid statistical foundations makes it indispensable for classification tasks where transparency matters. While limited to linear decision boundaries without extensive feature engineering, regularized logistic regression remains competitive with more complex methods in many real-world applications. Master logistic regression deeply—understanding coefficients as log-odds, the sigmoid transformation, and probability calibration—as these concepts extend to neural networks, where logistic regression is essentially a single-neuron network with sigmoid activation.</p>
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
};
