import { Topic } from '../../../types';

export const hyperparameterTuning: Topic = {
  id: 'hyperparameter-tuning',
  title: 'Hyperparameter Tuning',
  category: 'foundations',
  description: 'Techniques and strategies for optimizing model hyperparameters to improve performance.',
  content: `
    <h2>Hyperparameter Tuning: Optimizing Model Configuration</h2>
    <p>Hyperparameter tuning is the process of finding the optimal configuration of settings that control the learning process but aren't learned from data. While model parameters (like neural network weights or linear regression coefficients) are learned during training, hyperparameters must be specified beforehand and can dramatically affect performance. The difference between a mediocre model and a state-of-the-art one often lies not in the algorithm itself, but in how well its hyperparameters are tuned.</p>

    <p>Poor hyperparameter choices can lead to underfitting (model too simple, high bias), overfitting (model too complex, high variance), or slow convergence (inefficient training). Good hyperparameter tuning accelerates development, improves generalization, and can often deliver larger performance gains than algorithm selection or feature engineering. However, hyperparameter tuning is expensive—each configuration requires full model training—so efficient search strategies are essential.</p>

    <div class="info-box info-box-orange">
      <h4>🔍 Tuning Strategy Comparison</h4>
      <table>
        <tr>
          <th>Strategy</th>
          <th>Pros</th>
          <th>Cons</th>
          <th>When to Use</th>
        </tr>
        <tr>
          <td><strong>Manual</strong></td>
          <td>• Builds intuition<br/>• Flexible</td>
          <td>• Slow<br/>• Requires expertise</td>
          <td>Initial exploration, debugging</td>
        </tr>
        <tr>
          <td><strong>Grid Search</strong></td>
          <td>• Comprehensive<br/>• Simple<br/>• Reproducible</td>
          <td>• Exponential cost<br/>• Inefficient</td>
          <td>≤3 hyperparameters, coarse search</td>
        </tr>
        <tr>
          <td><strong>Random Search</strong></td>
          <td>• Efficient<br/>• Scales well<br/>• Anytime</td>
          <td>• No guarantees<br/>• Stochastic</td>
          <td><strong>Default choice</strong>, >3 hyperparameters</td>
        </tr>
        <tr>
          <td><strong>Bayesian</strong></td>
          <td>• Sample efficient<br/>• Smart search</td>
          <td>• Complex<br/>• Overhead</td>
          <td>Expensive evaluations, refinement</td>
        </tr>
        <tr>
          <td><strong>Hyperband/BOHB</strong></td>
          <td>• Very efficient<br/>• Early stopping</td>
          <td>• Most complex<br/>• Needs framework</td>
          <td>Large-scale, neural networks</td>
        </tr>
      </table>
      <p><strong>💡 Recommended Workflow:</strong> (1) Manual exploration → (2) Random search (50-100 trials) → (3) Bayesian optimization for refinement</p>
      <p><strong>⚠️ Priority:</strong> For neural nets: learning rate >> architecture >> batch size | For trees: n_estimators, max_depth >> other params</p>
    </div>

    <h3>Hyperparameters vs. Parameters: A Critical Distinction</h3>
    <p><strong>Parameters</strong> are the internal variables that a machine learning model learns from training data. In linear regression, parameters are the coefficients (weights) for each feature. In neural networks, parameters are the millions of weights connecting neurons. These are optimized automatically during training via algorithms like gradient descent, minimizing a loss function. You don't manually set parameters—the training process finds their optimal values.</p>

    <p><strong>Hyperparameters</strong> are configuration settings that control the learning process itself. They must be specified before training begins and remain fixed during training. Examples include: how fast to learn (learning rate), how complex the model should be (number of layers, regularization strength), how to sample data (batch size), and when to stop (number of epochs). Unlike parameters, hyperparameters can't be learned from data using standard optimization—they require a separate tuning process.</p>

    <p>The distinction matters because hyperparameters define the hypothesis space your model can explore and the optimization strategy it uses. Wrong hyperparameters can make even the best algorithm perform poorly. For example, a neural network with optimal weights for learning rate 0.01 will fail completely if you use learning rate 10.0 (diverging gradients) or 0.0001 (slow convergence). The same training data and architecture yield drastically different results depending on hyperparameter choices.</p>

    <h3>Common Hyperparameters Across Machine Learning</h3>
    <p>While specific hyperparameters vary by algorithm, common categories appear across methods:</p>

    <h4>Optimization Hyperparameters</h4>
    <ul>
      <li><strong>Learning rate (α, η):</strong> Step size for gradient-based optimization. Too high causes divergence, too low causes slow convergence. Typically 0.001-0.1 for neural networks. Often the single most important hyperparameter.</li>
      <li><strong>Batch size:</strong> Number of samples per gradient update. Affects training speed, memory usage, and generalization. Common values: 32, 64, 128, 256.</li>
      <li><strong>Number of epochs:</strong> How many times to iterate through the entire training dataset. Too few undertrains, too many overtrains. Use early stopping instead of fixing this.</li>
      <li><strong>Momentum/optimizer parameters:</strong> For Adam, SGD with momentum, RMSprop—control how past gradients influence current updates.</li>
    </ul>

    <h4>Regularization Hyperparameters</h4>
    <ul>
      <li><strong>Regularization strength (λ, α, C):</strong> Penalty for model complexity. L1/L2 regularization in linear models, dropout rate in neural networks, C parameter in SVM. Controls overfitting.</li>
      <li><strong>Dropout rate:</strong> Fraction of neurons to randomly deactivate during training (0.2-0.5 typical). Prevents overfitting in neural networks.</li>
      <li><strong>Weight decay:</strong> L2 penalty on weights, equivalent to regularization in many optimizers.</li>
    </ul>

    <h4>Model Architecture Hyperparameters</h4>
    <ul>
      <li><strong>Network depth and width:</strong> Number of layers and neurons per layer in neural networks. Deeper models can learn more complex functions but are harder to train.</li>
      <li><strong>Tree depth:</strong> Maximum depth in decision trees, max_depth in tree-based ensembles. Controls model complexity.</li>
      <li><strong>Number of estimators:</strong> Number of trees in Random Forests or Gradient Boosting. More trees generally improve performance but slow training/prediction.</li>
      <li><strong>Kernel type and parameters:</strong> For SVMs—RBF vs polynomial vs linear kernel, gamma for RBF, degree for polynomial.</li>
    </ul>

    <h4>Algorithm-Specific Hyperparameters</h4>
    <ul>
      <li><strong>K in KNN:</strong> Number of neighbors to consider.</li>
      <li><strong>min_samples_split, min_samples_leaf:</strong> Stopping criteria for tree-based models.</li>
      <li><strong>n_clusters:</strong> Number of clusters in K-Means.</li>
      <li><strong>n_components:</strong> Number of components in PCA or other dimensionality reduction.</li>
    </ul>

    <h3>Hyperparameter Tuning Strategies</h3>
    <p>The challenge is that hyperparameter space is vast—even with just 5 hyperparameters and 10 values each, there are 100,000 possible configurations. Trying all is infeasible. Different search strategies balance exploration (trying diverse configurations) against exploitation (refining promising regions).</p>

    <h4>1. Manual Search: Expert-Driven Tuning</h4>
    <p>Manually trying different hyperparameter values based on intuition, domain knowledge, and iterative experimentation. Look at training curves, validation performance, and error analysis to decide which hyperparameters to adjust and how.</p>

    <p><strong>Process:</strong> Start with reasonable defaults, train the model, examine results, adjust hyperparameters that seem problematic (e.g., if overfitting, increase regularization; if underfitting, add model capacity), repeat. Requires understanding of how each hyperparameter affects learning.</p>

    <p><strong>Advantages:</strong> Builds intuition about the model, can be efficient if you have experience, allows incorporating domain knowledge not captured by automated search, flexible and adaptive.</p>

    <p><strong>Disadvantages:</strong> Time-consuming, requires significant expertise, not reproducible, human bias may miss non-obvious configurations, doesn't scale to large hyperparameter spaces.</p>

    <p><strong>When to use:</strong> Initial exploration with new algorithms, debugging specific issues, when computational budget is extremely limited and you want to make every evaluation count, or when you're an expert with strong intuitions about the problem.</p>

    <h4>2. Grid Search: Exhaustive Exploration</h4>
    <p>Define a grid of hyperparameter values and exhaustively evaluate all combinations. For example, with learning_rate ∈ {0.001, 0.01, 0.1} and regularization ∈ {0.001, 0.01, 0.1, 1.0}, grid search tests all 3 × 4 = 12 combinations.</p>

    <p><strong>Process:</strong> Specify discrete values for each hyperparameter, compute the Cartesian product of all combinations, train and evaluate the model for each combination (typically with cross-validation), select the configuration with best validation performance.</p>

    <p><strong>Advantages:</strong> Comprehensive—guaranteed to find the best combination within the grid, reproducible (deterministic results), embarrassingly parallel (each configuration can be evaluated independently), simple to implement and understand.</p>

    <p><strong>Disadvantages:</strong> Exponential growth in combinations—2 hyperparameters with 10 values each = 100 evaluations, 5 hyperparameters = 100,000 evaluations (curse of dimensionality), wastes computation on unpromising regions, can miss optimal values between grid points (e.g., if optimal learning rate is 0.007 but you only test {0.001, 0.01, 0.1}), inefficient for continuous hyperparameters.</p>

    <p><strong>When to use:</strong> Small hyperparameter spaces (≤3 hyperparameters, ≤5 values each), when you need comprehensive exploration, when computational resources allow, as an initial coarse search before refining with other methods.</p>

    <h4>3. Random Search: Statistical Sampling</h4>
    <p>Sample random combinations of hyperparameters from specified distributions (e.g., learning rate from log-uniform[0.0001, 0.1], batch size from {32, 64, 128}). Evaluate a fixed number of random configurations.</p>

    <p><strong>Why it works better than grid search:</strong> Bergstra & Bengio (2012) showed that random search is more efficient than grid search, particularly when some hyperparameters are more important than others. Consider tuning learning rate (critical) and batch size (less critical). Grid search with 9 values per parameter tests 81 combinations but only explores 9 distinct values for learning rate. Random search with 81 trials samples 81 different learning rate values, providing better coverage of the important hyperparameter. For high-dimensional spaces, random search efficiently explores without exponential blowup.</p>

    <p><strong>Advantages:</strong> More efficient than grid search for high-dimensional spaces, better coverage of important hyperparameters, can specify continuous distributions (not just discrete values), easy to add more trials incrementally (anytime algorithm), parallelizable.</p>

    <p><strong>Disadvantages:</strong> No guarantee of finding optimal configuration (stochastic), may require many trials for good coverage, doesn't exploit information from previous trials to guide search.</p>

    <p><strong>When to use:</strong> As a default over grid search, medium to high-dimensional hyperparameter spaces (>3 hyperparameters), when you're uncertain about good ranges, as a first pass before Bayesian optimization. Empirically, random search with 50-100 trials often finds comparable or better solutions than grid search with similar budget.</p>

    <h4>4. Bayesian Optimization: Smart Exploration</h4>
    <p>Use a probabilistic model (often Gaussian Processes) to predict which hyperparameter regions are likely to yield improvements. The model builds a surrogate function approximating validation performance based on evaluated configurations, then uses an acquisition function to decide which configuration to try next, balancing exploration (trying uncertain regions) and exploitation (refining promising regions).</p>

    <p><strong>Process:</strong> Start with a few random evaluations, fit a probabilistic model (e.g., Gaussian Process) to predict performance as a function of hyperparameters, use an acquisition function (e.g., Expected Improvement, Upper Confidence Bound) to select the next configuration to evaluate by maximizing expected gain, update the model with new results, repeat until budget exhausted or convergence.</p>

    <p><strong>Advantages:</strong> Sample efficient—finds good configurations with fewer evaluations than random/grid search (often 10-50 trials vs 100+ for random search), intelligently focuses search on promising regions, handles expensive-to-evaluate functions well (perfect for machine learning where each evaluation is slow), can handle continuous and discrete hyperparameters, incorporates uncertainty to avoid premature convergence.</p>

    <p><strong>Disadvantages:</strong> More complex to implement, overhead of building and updating surrogate model (though this is negligible compared to model training time), can get stuck in local optima, requires careful tuning of acquisition function, not naturally parallel (though parallel variants exist like batch Bayesian optimization), can struggle with high-dimensional spaces (>20 hyperparameters).</p>

    <p><strong>When to use:</strong> When evaluations are expensive (each model training takes hours), limited computational budget (want best results with fewest trials), relatively low-dimensional hyperparameter spaces (<10 hyperparameters), when you've already done random search and want to refine. Libraries like Optuna, Hyperopt, and GPyOpt make this accessible.</p>

    <h4>5. Advanced Methods: Hyperband, BOHB, and Population-Based Training</h4>
    <p><strong>Hyperband:</strong> Uses successive halving—start many configurations with small budgets (few epochs), eliminate poor performers, double budget for remaining configurations, repeat. This efficiently allocates resources: bad configurations are killed early, good ones get more training. Particularly effective when many configurations are poor.</p>

    <p><strong>BOHB (Bayesian Optimization and Hyperband):</strong> Combines Bayesian optimization's smart sampling with Hyperband's efficient resource allocation. Uses Bayesian optimization to decide which configurations to evaluate, then Hyperband to allocate training budget. Often achieves state-of-the-art efficiency.</p>

    <p><strong>Population-Based Training (PBT):</strong> Maintains a population of models training in parallel. Periodically, poorly-performing models are killed and replaced with mutated copies of well-performing models (transfer learned weights, perturb hyperparameters). Effectively does online hyperparameter optimization while training. Particularly effective for neural networks with many hyperparameters.</p>

    <p><strong>When to use advanced methods:</strong> Large-scale experiments with significant compute budgets, neural networks with many hyperparameters, when you need to squeeze out the last few percentage points of performance. Tools like Ray Tune implement these methods.</p>

    <h3>Best Practices for Effective Hyperparameter Tuning</h3>
    <ul>
      <li><strong>Always use a separate validation set or cross-validation:</strong> Never tune on the test set—this leaks information and inflates performance estimates. Use k-fold cross-validation for more robust estimates, especially with limited data. The test set should be touched only once at the very end.</li>
      <li><strong>Start coarse, then refine:</strong> Begin with wide ranges to explore the space broadly (e.g., learning_rate in [1e-5, 1e-1]), identify promising regions, then narrow ranges for fine-tuning (e.g., [3e-4, 3e-3]). Two-stage tuning is more efficient than immediately searching narrow ranges.</li>
      <li><strong>Use appropriate scales:</strong> Learning rates, regularization parameters, and other hyperparameters often span many orders of magnitude. Sample them on log scale: log_uniform(1e-5, 1e-1) not uniform(0, 0.1). This ensures equal coverage of 0.001, 0.01, 0.1 rather than biasing toward larger values.</li>
      <li><strong>Prioritize important hyperparameters:</strong> For neural networks, learning rate >> architecture >> batch size. For tree ensembles, n_estimators and max_depth >> min_samples_split. Focus budget on high-impact hyperparameters. Use random search or Bayesian optimization's feature importance to identify which matter.</li>
      <li><strong>Tune related hyperparameters together:</strong> Learning rate and learning rate schedule, L1 and L2 regularization, network depth and width—these interact. Don't fix one while tuning the other; tune jointly or iteratively.</li>
      <li><strong>Use early stopping:</strong> For sequential algorithms (boosting, neural networks), use early stopping to halt training when validation performance plateaus. This prevents overfitting and speeds up tuning—you can try more configurations in the same time.</li>
      <li><strong>Track everything:</strong> Log all hyperparameter configurations, validation/test metrics, training curves, and random seeds. Tools like Weights & Biases, MLflow, or Neptune make this easy. You'll want to revisit configurations, analyze what worked, and ensure reproducibility.</li>
      <li><strong>Check for overfitting to validation set:</strong> With extensive tuning, validation performance becomes optimistic (you've effectively "trained" on it by selecting based on it). Monitor the gap between validation and test performance. If it's large, you may have overfit to validation—use more data, simpler models, or less tuning.</li>
      <li><strong>Balance performance vs computational cost:</strong> Don't chase 0.1% accuracy improvements if they require 10× more training time. Consider wall-clock time, memory usage, and inference latency alongside validation metrics. Sometimes a slightly worse but much faster model is preferable.</li>
    </ul>

    <h3>Common Pitfalls and How to Avoid Them</h3>
    <ul>
      <li><strong>Testing on the test set during development:</strong> This is the cardinal sin of machine learning. Every time you look at test performance and adjust anything (hyperparameters, features, algorithms), you leak information. The test set must be used exactly once at the very end. Use validation set or cross-validation for all development decisions.</li>
      <li><strong>Not using cross-validation:</strong> A single train-validation split can be misleading due to random chance. 5-fold or 10-fold CV provides more reliable estimates, especially for small datasets. The extra computation is usually worth it.</li>
      <li><strong>Ignoring computational constraints:</strong> Grid searching 10 hyperparameters with 5 values each requires 9.7 million evaluations. Be realistic about computational budget. Use random search or Bayesian optimization for large spaces.</li>
      <li><strong>Using the same random seed everywhere:</strong> Always use different random seeds for different CV folds and different hyperparameter trials. Otherwise, you're just measuring noise from one random split rather than true performance.</li>
      <li><strong>Not checking for interactions:</strong> Optimal learning rate often depends on batch size, optimal tree depth depends on number of trees. Tune interacting hyperparameters jointly. Grid search handles this naturally; for random/Bayesian search, ensure you're sampling configurations, not individual hyperparameters.</li>
      <li><strong>Assuming more complex is better:</strong> Hyperparameter tuning sometimes reveals that simpler models (shallower networks, fewer trees, less regularization) work best. Don't fix complex architectures then tune around them—include architectural simplicity in your search space.</li>
      <li><strong>Forgetting about overfitting to validation:</strong> If you tune for 1000 iterations, you're implicitly optimizing validation performance. This will overfit. Use nested cross-validation for unbiased estimates or strictly limit the number of configurations you try relative to validation set size.</li>
    </ul>

    <h3>Tools and Frameworks</h3>
    <p>Modern machine learning libraries provide extensive hyperparameter tuning support:</p>
    <ul>
      <li><strong>Scikit-learn:</strong> GridSearchCV and RandomizedSearchCV for grid and random search with built-in cross-validation. Simple, well-documented, great for traditional ML algorithms.</li>
      <li><strong>Optuna:</strong> State-of-the-art Bayesian optimization framework. Easy API, supports pruning (early stopping of poor trials), visualization tools, scales from laptops to clusters. Excellent for deep learning.</li>
      <li><strong>Ray Tune:</strong> Scalable hyperparameter tuning from single machines to large clusters. Supports all search algorithms (grid, random, Bayesian, Hyperband, PBT), integrates with major ML frameworks (PyTorch, TensorFlow, scikit-learn, XGBoost).</li>
      <li><strong>Keras Tuner:</strong> Hyperparameter tuning specifically for Keras/TensorFlow models. Supports random, Hyperband, Bayesian optimization, easy integration with existing Keras code.</li>
      <li><strong>Hyperopt:</strong> One of the earliest Bayesian optimization libraries for Python. Supports Tree-structured Parzen Estimators (TPE), parallelization via MongoDB.</li>
      <li><strong>Weights & Biases Sweeps:</strong> Combines hyperparameter tuning with experiment tracking. Bayesian optimization, grid, and random search with beautiful visualizations and team collaboration.</li>
      <li><strong>AutoML tools:</strong> Auto-sklearn, AutoGluon, H2O AutoML—fully automated pipelines that tune hyperparameters as part of end-to-end model selection. Great when you want hands-off optimization.</li>
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
};
