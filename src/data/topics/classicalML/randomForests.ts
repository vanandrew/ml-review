import { Topic } from '../../../types';

export const randomForests: Topic = {
  id: 'random-forests',
  title: 'Random Forests',
  category: 'classical-ml',
  description: 'Ensemble learning method that combines multiple decision trees for robust predictions',
  content: `
    <h2>Random Forests</h2>
    <p>Random Forest is an ensemble learning method that combines multiple decision trees through bootstrap aggregating (bagging) and feature randomness to create a more robust and accurate model. Introduced by Leo Breiman in 2001, it addresses the key weakness of decision trees—high variance and overfitting—by training many trees on different subsets of data and features, then averaging their predictions. Random Forest is one of the most popular and effective machine learning algorithms, offering excellent performance with minimal tuning across diverse domains including classification, regression, feature selection, and outlier detection.</p>

    <p>The fundamental insight is that while individual decision trees are high-variance models (small changes in training data lead to very different trees), averaging many diverse trees dramatically reduces variance without substantially increasing bias. If trees are perfectly independent with variance $\\sigma^2$, averaging N trees gives ensemble variance $\\sigma^2/N$. In practice, trees aren't fully independent (correlation $\\rho \\approx 0.3\\text{-}0.7$), but variance reduction is still substantial following $\\sigma^2_{\\text{ensemble}} = \\rho\\sigma^2 + (1-\\rho)\\sigma^2/N$. The challenge is ensuring tree diversity—if all trees are similar, averaging provides little benefit. Random Forest achieves diversity through two mechanisms: <strong>bootstrap sampling</strong> (each tree trains on different data) and <strong>feature randomness</strong> (each split considers different features).</p>

    <h3>Bootstrap Aggregating (Bagging)</h3>
    <p><strong>Bootstrap sampling</strong> is the foundation of bagging. For a dataset with N training examples, each tree is trained on a bootstrap sample—N examples sampled with replacement from the original data. Since sampling is with replacement, some examples appear multiple times in a sample while others don't appear at all. Through probability theory, we can show that each bootstrap sample contains approximately 63.2% unique examples:</p>
    
    <p>The probability an example is <em>not</em> selected in one draw is $\\frac{N-1}{N}$. Over N draws with replacement, the probability it's never selected is $\\left(\\frac{N-1}{N}\\right)^N$. As $N \\to \\infty$, this converges to $\\frac{1}{e} \\approx 0.368$. Therefore, ~63.2% of examples are selected at least once, and ~36.8% are left out.</p>

    <p>This creates natural diversity: each tree sees a different random subset of training data, learning slightly different patterns. Trees trained on different samples make different errors—one tree might overfit noise in certain regions, but other trees, trained on different data, won't make the same mistake in those regions. When predictions are averaged, these independent errors cancel out while the systematic patterns (signal) are reinforced. The 36.8% of examples not used to train a particular tree become that tree's <strong>out-of-bag (OOB) samples</strong>, providing a built-in validation set without sacrificing training data.</p>

    <h3>Feature Randomness: The Key Innovation</h3>
    <p>While bagging reduces variance, Random Forest adds a second layer of randomization that proves crucial for performance. At each split point in each tree, instead of considering all n features to find the best split, the algorithm randomly selects a subset of <strong>m features</strong> and chooses the best split from only this subset. The best feature overall might not be in the random subset, forcing the tree to use the second-best or third-best feature—creating a suboptimal split for that tree but increasing diversity across the ensemble.</p>

    <p>The standard choices for m are:</p>
    <ul>
      <li><strong>Classification:</strong> $m = \\sqrt{n}$ (square root of total features)</li>
      <li><strong>Regression:</strong> $m = n/3$ (one-third of total features)</li>
    </ul>

    <p>These heuristics balance individual tree quality against ensemble diversity. Too few features ($m = 1$ or $2$) makes trees overly weak and random, while too many features ($m$ close to $n$) reduces diversity benefits. The $\\sqrt{n}$ rule works remarkably well empirically, though it's worth tuning $m$ as a hyperparameter for specific problems.</p>

    <p><strong>Why feature randomness matters:</strong> In many datasets, one or two features are much stronger predictors than others. Without feature randomness, all trees in a bagged ensemble would likely use the same strong feature for their root split, then the same second-strongest feature for subsequent splits, creating highly correlated trees. When trees are correlated, averaging provides limited variance reduction—if all trees overfit in similar ways, their average still overfits. By restricting feature availability, different trees must use different features, discovering alternative but still informative split patterns. This decorrelation is why Random Forest typically outperforms plain bagging on decision trees.</p>

    <p>Feature randomness also provides implicit regularization and handles correlated features elegantly. If you have redundant features (e.g., temperature in Celsius, Fahrenheit, and Kelvin), they won't all dominate every tree—different trees use different versions, and the ensemble learns they're all useful. This reveals which features genuinely contribute predictive power beyond the most obvious ones.</p>

    <h3>Random Forest Algorithm</h3>
    <p>The complete Random Forest training procedure:</p>
    <ol>
      <li><strong>Specify parameters:</strong> Number of trees B (typically 100-500), number of features per split m (√n or n/3), and tree hyperparameters (max_depth, min_samples_split, min_samples_leaf).</li>
      <li><strong>For b = 1 to B (each tree):</strong>
        <ul>
          <li><strong>Bootstrap sampling:</strong> Draw N samples with replacement from training data D to create bootstrap sample D_b. Approximately 63% of unique samples from D will be in D_b; the remaining 37% are out-of-bag samples OOB_b.</li>
          <li><strong>Train decision tree T_b on D_b:</strong>
            <ul>
              <li>At each node (starting from root):
                <ul>
                  <li>If stopping criteria met (max_depth reached, min_samples_split not satisfied, node is pure), create leaf node with majority class (classification) or mean value (regression).</li>
                  <li>Otherwise, randomly select m features from the n available features.</li>
                  <li>For each of the m features, evaluate all possible splits and calculate impurity reduction (Gini, entropy for classification; MSE for regression).</li>
                  <li>Choose the split that maximizes impurity reduction among the m features (not among all n features).</li>
                  <li>Create left and right child nodes and recurse.</li>
                </ul>
              </li>
            </ul>
          </li>
        </ul>
      </li>
      <li><strong>Prediction (ensemble aggregation):</strong>
        <ul>
          <li><strong>Classification:</strong> For a new example $x$, pass it through all B trees to get predictions $T_1(x), T_2(x), \\ldots, T_B(x)$. Return the majority vote: $\\hat{y} = \\text{mode}(T_1(x), \\ldots, T_B(x))$. Alternatively, aggregate class probabilities: $\\hat{y}_{\\text{prob}}(c) = \\frac{1}{B} \\sum_{b=1}^{B} P_b(y=c|x)$, then predict $\\arg\\max_c \\hat{y}_{\\text{prob}}(c)$.</li>
          <li><strong>Regression:</strong> Return the average prediction: $\\hat{y} = \\frac{1}{B} \\sum_{b=1}^{B} T_b(x)$.</li>
        </ul>
      </li>
    </ol>

    <p>Note that Random Forest trees are typically grown to maximum depth without pruning—the ensemble averaging provides regularization, so individual trees can be as complex as possible to capture all patterns in their bootstrap samples. This contrasts with single decision trees, which require pruning or early stopping to avoid overfitting.</p>

    <h3>Out-of-Bag (OOB) Error Estimation</h3>
    <p>Random Forest includes an elegant built-in validation mechanism that provides an unbiased estimate of test error without requiring a separate validation set. Recall that each tree is trained on a bootstrap sample containing ~63% of training examples; the remaining ~37% are out-of-bag (OOB) for that tree. For any training example $x_i$, we can identify which trees didn't see $x_i$ during training (typically ~37% of all trees, or ~37 trees if B=100), use only those trees to predict $x_i$, and compare to the true label $y_i$.</p>

    <p>The <strong>OOB error</strong> is computed by aggregating these OOB predictions across all training examples:</p>
    <ol>
      <li>For each training example $(x_i, y_i)$:
        <ul>
          <li>Identify trees where $x_i$ was OOB: $S_i = \\{b : x_i \\notin D_b\\}$.</li>
          <li>Predict using only these trees: $\\hat{y}_i^{\\text{OOB}} = $ majority vote (classification) or average (regression) of $\\{T_b(x_i) : b \\in S_i\\}$.</li>
        </ul>
      </li>
      <li>Calculate OOB error: $\\frac{1}{N} \\sum_{i=1}^{N} L(y_i, \\hat{y}_i^{\\text{OOB}})$, where $L$ is loss function (0-1 loss for classification, MSE for regression).</li>
    </ol>

    <p>OOB error closely approximates test set error and is often nearly identical to k-fold cross-validation results, yet requires no extra computation beyond tracking which samples were OOB for which trees. This provides several benefits:</p>
    <ul>
      <li><strong>No validation set needed:</strong> When data is limited, you can use all data for training and still get reliable validation error via OOB—no need to sacrifice 20-30% for validation.</li>
      <li><strong>Hyperparameter tuning:</strong> OOB error can guide hyperparameter selection without separate validation—try different n_estimators, max_depth, or min_samples_split values and compare OOB scores.</li>
      <li><strong>Monitoring convergence:</strong> Track OOB error as trees are added to see when performance plateaus, helping choose n_estimators.</li>
      <li><strong>Feature selection:</strong> Compute OOB error, remove a feature, retrain, and compare OOB errors to assess feature importance.</li>
    </ul>

    <p>In scikit-learn, enable OOB scoring with <code>oob_score=True</code> and access results via <code>model.oob_score_</code> (accuracy for classification, R² for regression) after training. Note that OOB error is specific to bagging methods and doesn't apply to gradient boosting (which doesn't use bootstrap sampling).</p>

    <h3>Feature Importance in Random Forests</h3>
    <p>Random Forest provides feature importance scores by aggregating importance across all trees. For each tree, importance is measured by how much each feature reduces impurity (Gini, entropy, or MSE) across all nodes where that feature is used for splitting, weighted by the number of samples affected. These per-tree scores are averaged across all trees and normalized to sum to 1, giving each feature a proportion of total importance.</p>

    <p>Formally, for tree $T_b$, the importance of feature $X_j$ is:</p>
    <p style="margin-left: 20px;">$I_b(X_j) = \\sum_{\\text{nodes using } X_j} \\frac{n_{\\text{samples\\_node}}}{N} \\times \\left(\\text{impurity}_{\\text{before}} - \\frac{n_{\\text{left}}}{n_{\\text{node}}}\\times\\text{impurity}_{\\text{left}} - \\frac{n_{\\text{right}}}{n_{\\text{node}}}\\times\\text{impurity}_{\\text{right}}\\right)$</p>

    <p>The Random Forest importance is the average across trees: $I(X_j) = \\frac{1}{B} \\sum_{b=1}^{B} I_b(X_j)$.</p>

    <p><strong>Advantages over single-tree importance:</strong></p>
    <ul>
      <li><strong>Stability:</strong> While importance in a single tree can change drastically with small data perturbations, averaging across 100+ trees trained on different bootstrap samples smooths out variance. A feature spuriously important in one tree due to noise will likely have low importance in most others.</li>
      <li><strong>Context diversity:</strong> Different trees split on different features in different contexts, revealing which features are important in various regions of feature space—providing a more complete picture of feature utility.</li>
    </ul>

    <p><strong>Limitations and alternatives:</strong></p>
    <ul>
      <li><strong>Bias toward high-cardinality features:</strong> Features with many unique values (continuous variables, IDs) get inflated importance because they offer more split possibilities. This bias persists from single trees, though feature randomness mitigates it somewhat.</li>
      <li><strong>Correlated features:</strong> With correlated features, one will often be chosen more frequently than the other, receiving higher importance even though they're equally informative. Feature randomness helps but doesn't eliminate this issue.</li>
      <li><strong>Permutation importance:</strong> A more robust alternative is to shuffle each feature's values and measure the decrease in OOB accuracy. This approach is slower but less biased by cardinality and correlation. In scikit-learn: <code>from sklearn.inspection import permutation_importance</code>.</li>
      <li><strong>SHAP values:</strong> SHapley Additive exPlanations provide theoretically grounded importance measures based on cooperative game theory, offering more reliable feature importance for critical decisions.</li>
    </ul>

    <p>In practice, use Random Forest's built-in importance (Mean Decrease in Impurity) for quick insights and computational efficiency, but verify with permutation importance or SHAP for critical decisions or when dealing with correlated features.</p>

    <h3>Hyperparameters and Tuning</h3>
    <p>Random Forest has several hyperparameters controlling ensemble and individual tree behavior:</p>

    <h4>Ensemble-Level Parameters</h4>
    <ul>
      <li><strong>n_estimators</strong> (number of trees, default 100): More trees always improve or maintain performance—adding trees never hurts because averaging more models only reduces variance. However, returns diminish after a point (10→100 trees helps a lot; 500→1000 provides marginal gains). The trade-off is training and prediction time, which scale linearly with n_estimators. Start with 100, check learning curves, and increase to 200-500 if computational resources allow. Unlike gradient boosting, you can't overfit by adding more trees.</li>
      <li><strong>max_features</strong> (features per split, default $\\sqrt{n}$ for classification, $n/3$ for regression): Controls feature randomness and tree decorrelation. Smaller values increase diversity but may make individual trees too weak. Larger values improve individual tree quality but reduce ensemble diversity. Tune this especially if features are highly correlated or a few dominant predictors exist—reducing max_features (e.g., from $\\sqrt{n}$ to $\\log_2(n)$) can help. Conversely, if all features are weakly predictive, increasing max_features may help.</li>
      <li><strong>max_samples</strong> (bootstrap sample size, default N): Can reduce to <1.0 (fraction) or <N (integer) for faster training with slight performance trade-off.</li>
      <li><strong>oob_score</strong> (default False): Enable OOB error computation for validation without separate test set.</li>
      <li><strong>n_jobs</strong> (default 1): Set to -1 to use all CPU cores for parallel training, providing significant speedups.</li>
    </ul>

    <h4>Tree-Level Parameters</h4>
    <ul>
      <li><strong>max_depth</strong> (default None = unlimited): Controls tree depth. None works well for Random Forest since ensemble averaging prevents overfitting, but limiting depth (10-30) speeds up training and prediction. Rarely needs tuning.</li>
      <li><strong>min_samples_split</strong> (default 2): Minimum samples required to split a node. Increasing (e.g., 10-20) creates simpler trees, which may help with very noisy data.</li>
      <li><strong>min_samples_leaf</strong> (default 1): Minimum samples required at leaf nodes. Increasing (e.g., 5-10) creates simpler trees and smoother decision boundaries.</li>
      <li><strong>max_leaf_nodes</strong> (default None): Directly caps tree size, providing another way to control complexity.</li>
      <li><strong>min_impurity_decrease</strong> (default 0.0): Minimum improvement required to split—higher values create simpler trees.</li>
    </ul>

    <h4>Other Parameters</h4>
    <ul>
      <li><strong>class_weight</strong> (classification only): 'balanced' automatically adjusts weights inversely proportional to class frequencies for imbalanced datasets.</li>
      <li><strong>criterion</strong> (default 'gini' for classification, 'squared_error' for regression): Splitting criterion—'entropy' and 'log_loss' are alternatives for classification.</li>
    </ul>

    <p><strong>Tuning strategy:</strong> Random Forest is generally robust to hyperparameters—default settings often work well, making it a low-maintenance model. Prioritize tuning n_estimators (more is better up to time budget) and max_features (tune for feature correlation patterns) via grid search or random search, using OOB score or cross-validation for evaluation. Tree-specific parameters (max_depth, min_samples_split, min_samples_leaf) rarely need adjustment—the ensemble provides sufficient regularization even with fully grown trees.</p>

    <h3>Advantages</h3>
    <ul>
      <li><strong>Reduces overfitting:</strong> Ensemble averaging dramatically reduces variance compared to single decision trees without substantially increasing bias.</li>
      <li><strong>Excellent out-of-the-box performance:</strong> Works well with default hyperparameters across diverse problems, requiring minimal tuning.</li>
      <li><strong>Handles high-dimensional data:</strong> Effective even when number of features approaches or exceeds number of samples, thanks to feature randomness.</li>
      <li><strong>Robust to outliers and noise:</strong> No single outlier can dominate all trees; its impact is averaged out across the ensemble.</li>
      <li><strong>No feature scaling required:</strong> Tree-based splits are scale-invariant, so no need for standardization or normalization.</li>
      <li><strong>Handles missing values:</strong> Can use surrogate splits (sklearn) or built-in missing value handling (implementations like XGBoost).</li>
      <li><strong>Handles mixed data types:</strong> Works with both numerical and categorical features (though categorical encoding may still be needed depending on implementation).</li>
      <li><strong>Feature importance:</strong> Provides interpretable feature rankings for understanding model behavior.</li>
      <li><strong>Non-linear relationships:</strong> Captures complex non-linear interactions between features without manual feature engineering.</li>
      <li><strong>Parallel training:</strong> Trees can be trained independently in parallel, speeding up training on multi-core systems.</li>
      <li><strong>Built-in validation:</strong> OOB error provides reliable performance estimates without separate validation set.</li>
    </ul>

    <h3>Disadvantages</h3>
    <ul>
      <li><strong>Less interpretable:</strong> While individual trees are interpretable, an ensemble of 100+ trees is a black box—difficult to explain specific predictions.</li>
      <li><strong>Larger model size:</strong> Storing multiple full trees requires significant memory (tens to hundreds of MB for large datasets), making deployment challenging for resource-constrained environments.</li>
      <li><strong>Slower prediction:</strong> Must query all trees at inference time, making predictions slower than single trees or linear models (though still reasonably fast for most applications).</li>
      <li><strong>Biased toward majority class:</strong> In imbalanced classification, trees can be biased toward majority class—use class_weight='balanced' or other imbalance techniques.</li>
      <li><strong>Poor extrapolation:</strong> Like all tree-based models, Random Forest cannot extrapolate beyond the range of training data—predictions for inputs outside training range default to the closest leaf values.</li>
      <li><strong>Difficulty with linear relationships:</strong> Captures linear relationships less efficiently than linear models, requiring more trees and deeper trees to approximate linear functions.</li>
      <li><strong>Bias in feature importance:</strong> Built-in importance (MDI) is biased toward high-cardinality and continuous features—use permutation importance for more reliable estimates.</li>
    </ul>

    <h3>Random Forest vs. Single Decision Tree</h3>
    <table>
      <thead>
        <tr><th>Aspect</th><th>Single Decision Tree</th><th>Random Forest</th></tr>
      </thead>
      <tbody>
        <tr><td>Variance</td><td>Very high—small data changes cause completely different trees</td><td>Much lower—ensemble averaging smooths out variance</td></tr>
        <tr><td>Bias</td><td>Low (with sufficient depth)</td><td>Slightly higher (due to feature randomness constraints)</td></tr>
        <tr><td>Overfitting</td><td>Severe without pruning</td><td>Minimal due to ensemble regularization</td></tr>
        <tr><td>Accuracy</td><td>Lower and unstable</td><td>Higher and stable</td></tr>
        <tr><td>Interpretability</td><td>Excellent—visualize entire tree</td><td>Poor—black box ensemble</td></tr>
        <tr><td>Training time</td><td>Fast (single tree)</td><td>Slower (B trees), but parallelizable</td></tr>
        <tr><td>Prediction time</td><td>Very fast (one tree traversal)</td><td>Slower (B tree traversals)</td></tr>
        <tr><td>Feature importance</td><td>Unstable, varies with data</td><td>Stable, averaged across trees</td></tr>
        <tr><td>Hyperparameter sensitivity</td><td>Very sensitive—requires careful tuning</td><td>Robust—good default performance</td></tr>
      </tbody>
    </table>

    <h3>Visual Understanding</h3>
    <p>Imagine training 100 separate decision trees, each seeing a different random subset of data and a random subset of features at each split. Each tree makes its own prediction—some might be wildly wrong, others accurate. Random Forest aggregates these predictions by voting (classification) or averaging (regression). This "wisdom of crowds" effect cancels out individual tree errors, as long as trees make independent mistakes.</p>
    
    <p><strong>Key visualizations to understand:</strong></p>
    <ul>
      <li><strong>Ensemble prediction diagram:</strong> Show 5-10 individual tree predictions as bars (e.g., Tree 1: Class A 80%, Class B 20%) and the final ensemble average highlighting convergence to the correct answer. Illustrates how individual tree errors (some predict wrong class) average out.</li>
      <li><strong>Bootstrap sampling visualization:</strong> Display original dataset as colored dots, then show 3-4 bootstrap samples as subsets with some dots repeated (appearing multiple times) and others missing (~37% not selected). Shows how each tree sees different data.</li>
      <li><strong>Feature randomness at splits:</strong> For a single node split, show all 20 features grayed out with 4-5 highlighted as the randomly selected subset considered for that split. Different splits highlight different random subsets, forcing tree diversity.</li>
      <li><strong>Forest decision boundary:</strong> For 2D data, overlay decision boundaries from 3-5 individual trees (thin colored lines, showing jagged boundaries) and the ensemble boundary (thick line, showing smoothed result). Demonstrates how averaging creates smoother, more robust boundaries.</li>
      <li><strong>Out-of-bag validation:</strong> Diagram showing that for each training point, ~37% of trees didn't see it in their bootstrap sample. These "out-of-bag" trees can validate that point without overfitting, providing free cross-validation.</li>
    </ul>

    <h3>Common Mistakes to Avoid</h3>
    <ul>
      <li><strong>❌ Not tuning n_estimators (number of trees):</strong> Default (often 100) may be too few. More trees almost always improve performance (with diminishing returns). Try 200-500 trees if computationally feasible. Unlike other hyperparameters, more trees rarely hurt (though training slows).</li>
      <li><strong>❌ Over-tuning tree depth on small datasets:</strong> Deep trees (max_depth>20) on small data (<1000 samples) lead to overfitting despite ensemble averaging. Start with max_depth=10-20, then tune via cross-validation.</li>
      <li><strong>❌ Ignoring class imbalance:</strong> Random Forest can be biased toward majority class. Use class_weight='balanced', or apply SMOTE/undersampling before training.</li>
      <li><strong>❌ Using Random Forest for extrapolation:</strong> Tree-based models cannot predict outside the range of training data. For time-series forecasting or regression where test values exceed training range, use linear models or neural networks instead.</li>
      <li><strong>❌ Not leveraging parallelization:</strong> Training is embarrassingly parallel (trees are independent). Always set n_jobs=-1 to use all CPU cores, drastically speeding up training.</li>
      <li><strong>❌ Relying solely on built-in feature importance:</strong> Gini-based importance is biased toward high-cardinality features. Use permutation importance (more robust) for critical feature selection decisions.</li>
      <li><strong>❌ Expecting high interpretability:</strong> While individual trees are interpretable, a forest of 100+ trees is a black box. If interpretability is critical, use a single decision tree (less accurate) or SHAP values (more complex).</li>
      <li><strong>❌ Forgetting to set random_state:</strong> Without fixing the seed, results vary across runs, making debugging and comparison difficult. Always set random_state for reproducibility.</li>
    </ul>

    <h3>Practical Recommendations</h3>
    <ul>
      <li><strong>Start with defaults:</strong> Random Forest often works well out-of-the-box—try default hyperparameters before tuning.</li>
      <li><strong>Increase n_estimators:</strong> If you have computational budget, increase to 200-500 trees for better performance.</li>
      <li><strong>Use OOB score:</strong> Enable oob_score=True for validation without separate test set, especially with limited data.</li>
      <li><strong>Tune max_features:</strong> If features are highly correlated, reduce max_features to increase diversity.</li>
      <li><strong>Enable parallelization:</strong> Set n_jobs=-1 to use all CPU cores for faster training.</li>
      <li><strong>Handle imbalance:</strong> For imbalanced classification, use class_weight='balanced' or other resampling techniques.</li>
      <li><strong>Check feature importance:</strong> Use built-in importance for quick insights, but verify with permutation importance for critical decisions.</li>
      <li><strong>Compare to gradient boosting:</strong> If Random Forest performance is insufficient, try gradient boosting (XGBoost, LightGBM) for potentially better accuracy at the cost of more tuning and overfitting risk.</li>
      <li><strong>Consider computational constraints:</strong> For deployment on resource-limited devices, reduce n_estimators or max_depth, or consider simpler models like logistic regression or single decision trees.</li>
      <li><strong>Avoid overfitting with noise:</strong> If training data is very noisy, limit max_depth or increase min_samples_split/leaf to prevent memorization.</li>
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
};
