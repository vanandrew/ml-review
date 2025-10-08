import { Topic } from '../../../types';

export const overfittingUnderfitting: Topic = {
  id: 'overfitting-underfitting',
  title: 'Overfitting and Underfitting',
  category: 'foundations',
  description: 'Understanding model complexity and the bias-variance tradeoff in practice',
  content: `
    <h2>Overfitting and Underfitting: The Twin Perils of Machine Learning</h2>
    <p>Overfitting and underfitting represent the two fundamental failure modes in machine learning\u2014the practical manifestations of the bias-variance tradeoff. Understanding these concepts deeply and learning to diagnose and address them is essential for building models that generalize well to real-world data.</p>

    <h3>Understanding Underfitting (High Bias)</h3>
    <p>Underfitting occurs when your model is too simple to capture the underlying patterns in the data. The model makes overly strong assumptions about the data's structure, resulting in systematic errors that persist regardless of how much data you provide or how long you train.</p>
    
    <p><strong>What It Looks Like in Practice:</strong></p>
    <p>Imagine trying to fit a straight line to data that clearly follows a parabolic curve. No matter how you adjust that line's slope and intercept, it will always systematically miss the curvature. The model is fundamentally incapable of representing the true relationship because of its limited capacity.</p>
    
    <p><strong>Symptoms and Diagnostic Signs:</strong></p>
    <ul>
      <li><strong>Poor training accuracy:</strong> The model struggles to fit even the training data (e.g., 65% accuracy when 85% is achievable)</li>
      <li><strong>Similar validation accuracy:</strong> Training and validation errors are both high and close together (e.g., train: 65%, validation: 67%)</li>
      <li><strong>Small train-validation gap:</strong> The 2-5 percentage point difference indicates the model isn't overfitting\u2014it's just not learning well at all</li>
      <li><strong>Plateaued learning curves:</strong> Both training and validation error curves flatten early and remain high</li>
      <li><strong>Systematic errors:</strong> The model consistently underpredicts or overpredicts in certain regions</li>
      <li><strong>More data doesn't help:</strong> Adding training examples doesn't improve performance because the problem is model capacity, not sample size</li>
    </ul>
    
    <p><strong>Common Causes:</strong></p>
    <ul>
      <li><strong>Model too simple:</strong> Linear model for non-linear data, shallow network for complex patterns</li>
      <li><strong>Insufficient features:</strong> Missing important predictive information</li>
      <li><strong>Excessive regularization:</strong> λ too high, overly constraining the model</li>
      <li><strong>Inadequate training:</strong> Stopped too early before convergence</li>
      <li><strong>Poor feature representation:</strong> Features don't capture relevant aspects of the problem</li>
    </ul>
    
    <p><strong>Real-World Examples:</strong></p>
    <ul>
      <li>Using linear regression to predict house prices when the relationship with square footage is quadratic</li>
      <li>Training a 2-layer neural network on complex image classification (ImageNet) that requires deep representations</li>
      <li>Predicting customer churn with only demographic features, missing behavioral patterns</li>
      <li>Using a decision tree with max_depth=2 on a dataset with complex interactions</li>
    </ul>
    
    <p><strong>How to Fix Underfitting:</strong></p>
    <ul>
      <li><strong>Increase model complexity:</strong> Use polynomial features, deeper neural networks, more trees in ensemble, higher-degree polynomials</li>
      <li><strong>Add more features:</strong> Create interaction terms, polynomial features, domain-specific features</li>
      <li><strong>Reduce regularization:</strong> Lower λ in L1/L2 penalties, reduce dropout rate, allow deeper trees</li>
      <li><strong>Train longer:</strong> More epochs to reach convergence, especially for iterative algorithms</li>
      <li><strong>Switch model families:</strong> Move from linear to non-linear models, from simple to more expressive architectures</li>
      <li><strong>Feature engineering:</strong> Transform features to better capture relationships (log transforms, ratios, etc.)</li>
    </ul>

    <h3>Understanding Overfitting (High Variance)</h3>
    <p>Overfitting is the more insidious problem\u2014your model appears to work beautifully during training but fails on new data. The model has learned the training data too well, memorizing noise and idiosyncrasies rather than learning generalizable patterns.</p>
    
    <p><strong>What It Looks Like in Practice:</strong></p>
    <p>Imagine a decision tree that grows so deep it creates a unique leaf for nearly every training example, with hyper-specific rules like \"if age=32 AND income=$54,231 AND has_pet=True, then class=1.\" This rule might perfectly classify one training example but will never apply to new data with slightly different values.</p>
    
    <p><strong>Symptoms and Diagnostic Signs:</strong></p>
    <ul>
      <li><strong>Excellent training accuracy:</strong> Near-perfect or perfect fit to training data (e.g., 98-100%)</li>
      <li><strong>Poor validation accuracy:</strong> Much worse performance on validation set (e.g., 65-70%)</li>
      <li><strong>Large train-validation gap:</strong> Significant difference (e.g., 30+ percentage points) indicates memorization</li>
      <li><strong>Diverging learning curves:</strong> Training error continues decreasing while validation error plateaus or increases</li>
      <li><strong>Erratic predictions:</strong> Small changes in input cause large changes in output</li>
      <li><strong>High cross-validation variance:</strong> Performance varies dramatically across different folds</li>
      <li><strong>Model performs differently on similar inputs:</strong> Inconsistent predictions on examples that should be treated similarly</li>
    </ul>
    
    <p><strong>Common Causes:</strong></p>
    <ul>
      <li><strong>Model too complex for available data:</strong> More parameters than necessary given dataset size</li>
      <li><strong>Too many features:</strong> High-dimensional feature space with many irrelevant features</li>
      <li><strong>Insufficient training data:</strong> Not enough examples to constrain the model</li>
      <li><strong>Training too long:</strong> Model continues fitting training noise past the point of best generalization</li>
      <li><strong>No regularization:</strong> Nothing prevents the model from fitting every detail</li>
      <li><strong>Noisy training data:</strong> Errors or outliers in labels that model tries to fit</li>
    </ul>
    
    <p><strong>Real-World Examples:</strong></p>
    <ul>
      <li>A 15th-degree polynomial fitted to 20 data points\u2014wiggles wildly between points</li>
      <li>A decision tree with max_depth=50 on a dataset of 1,000 samples\u2014memorizes individual examples</li>
      <li>A neural network with 10 million parameters trained on 5,000 images without regularization</li>
      <li>K-nearest neighbors with K=1, making predictions based on single (possibly noisy) neighbors</li>
    </ul>
    
    <p><strong>How to Fix Overfitting:</strong></p>
    <ul>
      <li><strong>Get more training data:</strong> The single most effective solution\u2014dilutes the noise, provides more representative examples</li>
      <li><strong>Add regularization:</strong> L1/L2 penalties on weights, dropout in neural networks, pruning decision trees</li>
      <li><strong>Reduce model complexity:</strong> Fewer layers/neurons, lower polynomial degree, shallower trees, smaller ensemble</li>
      <li><strong>Feature selection:</strong> Remove irrelevant or redundant features that add noise</li>
      <li><strong>Early stopping:</strong> Halt training when validation performance stops improving</li>
      <li><strong>Data augmentation:</strong> Create synthetic training examples (image rotations, translations, noise injection)</li>
      <li><strong>Ensemble methods:</strong> Bagging/Random Forests average out variance across models</li>
      <li><strong>Cross-validation:</strong> Ensure model selection doesn't overfit to a single validation split</li>
      <li><strong>Simplify architecture:</strong> Use simpler model families or smaller architectures</li>
    </ul>

    <h3>The Relationship to Bias-Variance Tradeoff</h3>
    <p>Underfitting and overfitting are the practical manifestations of bias and variance:</p>
    
    <p><strong>Underfitting = High Bias, Low Variance:</strong></p>
    <ul>
      <li>Model makes consistent (but wrong) predictions across different training sets</li>
      <li>Error comes from systematic misrepresentation of the true function</li>
      <li>Predictions are stable but systematically incorrect</li>
      <li>The model is \"biased\" toward a particular (incorrect) solution form</li>
    </ul>
    
    <p><strong>Overfitting = Low Bias, High Variance:</strong></p>
    <ul>
      <li>Model makes wildly different predictions when trained on different samples</li>
      <li>Error comes from sensitivity to random fluctuations in training data</li>
      <li>Predictions vary dramatically with small changes to training set</li>
      <li>The model has high \"variance\" across different training samples</li>
    </ul>
    
    <p><strong>The Mathematical Connection:</strong></p>
    <p>$\\text{Expected Error} = \\text{Bias}^2 + \\text{Variance} + \\text{Irreducible Error}$</p>
    <ul>
      <li><strong>Underfitting:</strong> Bias² dominates the error term</li>
      <li><strong>Overfitting:</strong> Variance dominates the error term</li>
      <li><strong>Good fit:</strong> Both bias and variance are minimized (at the sweet spot)</li>
    </ul>

    <h3>Detecting and Diagnosing: Learning Curves</h3>
    <p>Learning curves\u2014plots of training and validation error as a function of training set size or training iterations\u2014are your most powerful diagnostic tool.</p>
    
    <p><strong>Underfitting Pattern (High Bias):</strong></p>
    <ul>
      <li><strong>Training error:</strong> High and relatively flat (e.g., 35-40%)</li>
      <li><strong>Validation error:</strong> High and similar to training error (e.g., 38-42%)</li>
      <li><strong>Gap:</strong> Small (2-5 percentage points)</li>
      <li><strong>Behavior with more data:</strong> Both curves plateau early\u2014more data doesn't help</li>
      <li><strong>Interpretation:</strong> Model can't even fit training data well; problem is capacity not data</li>
      <li><strong>Action:</strong> Increase model complexity or add features</li>
    </ul>
    
    <p><strong>Overfitting Pattern (High Variance):</strong></p>
    <ul>
      <li><strong>Training error:</strong> Very low (e.g., 2-5%)</li>
      <li><strong>Validation error:</strong> Much higher (e.g., 25-35%)</li>
      <li><strong>Gap:</strong> Large (20-30+ percentage points)</li>
      <li><strong>Behavior with more data:</strong> Gap persists or decreases slowly; validation error may improve slightly but gap remains</li>
      <li><strong>Interpretation:</strong> Model fits training data too well, capturing noise</li>
      <li><strong>Action:</strong> Add regularization, get more data, or reduce complexity</li>
    </ul>
    
    <p><strong>Good Fit Pattern (Sweet Spot):</strong></p>
    <ul>
      <li><strong>Training error:</strong> Acceptably low for the task (e.g., 10-15%)</li>
      <li><strong>Validation error:</strong> Similar to training error (e.g., 12-18%)</li>
      <li><strong>Gap:</strong> Small (2-5 percentage points)</li>
      <li><strong>Behavior with more data:</strong> Both converge to low, acceptable error</li>
      <li><strong>Interpretation:</strong> Model is well-calibrated for available data and problem complexity</li>
      <li><strong>Action:</strong> This is your target! Model is ready for testing</li>
    </ul>
    
    <p><strong>Special Case: Both High Bias and High Variance:</strong></p>
    <p>Rarely, models can exhibit both\u2014training error is moderate (not great fit), but validation error is much worse. This can occur with poorly designed models or wrong feature representations. Solution: fundamentally rethink your modeling approach.</p>

    <h3>Practical Scenarios and Solutions</h3>
    
    <p><strong>Scenario 1: Training 95%, Validation 60%</strong></p>
    <p><strong>Diagnosis:</strong> Classic overfitting\u201435% gap is enormous</p>
    <p><strong>Solution priority:</strong></p>
    <ol>
      <li>Add strong regularization (increase λ, add dropout)</li>
      <li>Reduce model complexity (fewer layers, shallower trees)</li>
      <li>Get more training data if possible</li>
      <li>Apply early stopping</li>
      <li>Remove features (try feature selection)</li>
    </ol>
    
    <p><strong>Scenario 2: Training 50%, Validation 52%</strong></p>
    <p><strong>Diagnosis:</strong> Clear underfitting\u2014both errors too high, tiny gap</p>
    <p><strong>Solution priority:</strong></p>
    <ol>
      <li>Increase model complexity (deeper network, higher polynomial degree)</li>
      <li>Add more features or create feature interactions</li>
      <li>Reduce regularization (lower λ, less dropout)</li>
      <li>Train longer to ensure convergence</li>
      <li>Try a more expressive model family</li>
    </ol>
    
    <p><strong>Scenario 3: Training 10%, Validation 12%</strong></p>
    <p><strong>Diagnosis:</strong> Good fit! Small gap, acceptable performance</p>
    <p><strong>Action:</strong> Evaluate on test set. If performance is consistent, model is production-ready. May try slight increases in complexity to see if you can improve further without overfitting.</p>
    
    <p><strong>Scenario 4: Training 15%, Validation 40%</strong></p>
    <p><strong>Diagnosis:</strong> Overfitting, but moderate training error suggests room for improvement</p>
    <p><strong>Solution:</strong> This is tricky\u2014you need more capacity (to reduce training error) but also regularization (to reduce gap). Try: increase complexity slightly but add regularization, or use ensemble methods that naturally balance bias and variance.</p>

    <h3>The Role of Data Quantity</h3>
    <p><strong>More data reduces variance but not bias:</strong></p>
    <ul>
      <li>With more samples, random noise averages out (variance reduction)</li>
      <li>But if your model is fundamentally too simple, more data won't help (bias persists)</li>
      <li>Learning curves reveal this: underfitting curves plateau early, overfitting curves continue improving with more data</li>
    </ul>
    
    <p><strong>How much data is enough?</strong></p>
    <ul>
      <li>Depends on problem complexity, model complexity, and noise level</li>
      <li>Rule of thumb: at least 10x more samples than parameters (very rough guideline)</li>
      <li>Simple linear model: hundreds to thousands of examples</li>
      <li>Complex neural network: thousands to millions of examples</li>
      <li>Look at learning curves: if validation error still decreasing as you add data, get more data</li>
    </ul>

    <h3>Model Complexity Spectrum</h3>
    <p>Different models and hyperparameters occupy different points on the complexity spectrum:</p>
    
    <p><strong>Increasing Complexity →</strong></p>
    <ul>
      <li><strong>Polynomial Regression:</strong> Degree 1 → Degree 2 → Degree 5 → Degree 15</li>
      <li><strong>Decision Trees:</strong> max_depth=2 → max_depth=5 → max_depth=10 → max_depth=None</li>
      <li><strong>Neural Networks:</strong> 1 layer, 10 neurons → 2 layers, 50 neurons → 5 layers, 200 neurons → 20 layers, 1000 neurons</li>
      <li><strong>KNN:</strong> K=50 → K=10 → K=5 → K=1</li>
      <li><strong>Ensemble Size:</strong> 10 trees → 100 trees → 1000 trees</li>
    </ul>
    
    <p><strong>Finding the sweet spot:</strong> Start simple, gradually increase complexity while monitoring validation performance. Stop when validation error stops decreasing or starts increasing.</p>

    <h3>Prevention and Best Practices</h3>
    <ul>
      <li><strong>Always use validation sets:</strong> Never evaluate only on training data</li>
      <li><strong>Plot learning curves:</strong> Makes diagnosis visual and obvious</li>
      <li><strong>Start simple:</strong> Begin with simple models, add complexity only when justified</li>
      <li><strong>Regular checkpoints:</strong> Save models at different training stages to revert if overfitting emerges</li>
      <li><strong>Cross-validation:</strong> For robust estimates, especially with limited data</li>
      <li><strong>Monitor multiple metrics:</strong> Accuracy, precision, recall\u2014overfitting may manifest differently across metrics</li>
      <li><strong>Use regularization by default:</strong> Easier to reduce it if underfitting than to add it after overfitting</li>
      <li><strong>Keep test set pristine:</strong> Don't touch it until final evaluation</li>
    </ul>

    <h3>Summary: The Complete Picture</h3>
    <p>Overfitting and underfitting are not binary states but opposite ends of a spectrum. Your goal is to find the optimal point where your model is complex enough to capture true patterns (avoiding underfitting) but not so complex that it captures noise (avoiding overfitting). This sweet spot depends on your data quantity, quality, noise level, and problem complexity. Diagnostic tools like learning curves, train-validation gaps, and cross-validation variance help you identify where you are on this spectrum and guide you toward the optimal model.</p>
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
};
