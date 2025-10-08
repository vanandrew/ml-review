import { Topic } from '../../../types';

export const decisionTrees: Topic = {
  id: 'decision-trees',
  title: 'Decision Trees',
  category: 'classical-ml',
  description: 'Understanding decision trees for both classification and regression tasks.',
  content: `
    <h2>Decision Trees: Intuitive Hierarchical Decision Making</h2>
    <p>Decision trees are one of the most intuitive and interpretable machine learning algorithms, modeling decisions through a tree-like structure of sequential questions about feature values. Each path from root to leaf represents a decision rule: a series of if-then statements that leads to a prediction. Despite their simplicity, decision trees can capture complex non-linear relationships and feature interactions without requiring feature engineering. While single trees are prone to overfitting and instability, they form the foundation for powerful ensemble methods like Random Forests and Gradient Boosting.</p>

    <h3>How Decision Trees Work</h3>
    
    <p>A decision tree recursively partitions the feature space into rectangular regions and makes predictions based on the training examples within each region. The tree structure consists of:</p>
    <ul>
      <li><strong>Root Node:</strong> Top of the tree, contains all training data</li>
      <li><strong>Internal Nodes:</strong> Represent decisions based on feature thresholds (e.g., "Is age > 30?")</li>
      <li><strong>Branches:</strong> Outcomes of decisions (yes/no for binary splits)</li>
      <li><strong>Leaf Nodes:</strong> Terminal nodes containing predictions (class label for classification, value for regression)</li>
    </ul>
    
    <p><strong>Building Process (Greedy Recursive Splitting):</strong></p>
    <ol>
      <li>Start with all data at the root node</li>
      <li>For each feature, evaluate all possible split thresholds</li>
      <li>Select the feature and threshold that best separates the data (maximizes impurity reduction)</li>
      <li>Create child nodes by partitioning data according to the split</li>
      <li>Recursively repeat steps 2-4 for each child node</li>
      <li>Stop when stopping criteria are met (max depth, min samples, pure node)</li>
    </ol>
    
    <p>The algorithm is <em>greedy</em>—it makes locally optimal decisions at each step without looking ahead. This is computationally efficient but can miss globally better splits. It's also <em>top-down</em>—once a split is made, it's never reconsidered.</p>

    <h3>Splitting Criteria: Measuring Impurity</h3>
    
    <p>The quality of a split is measured by how much it reduces <strong>impurity</strong>—the degree of "mixing" of classes or values in a node.</p>
    
    <p><strong>Classification Criteria:</strong></p>
    
    <p><strong>1. Gini Impurity</strong> (default in scikit-learn):</p>
    <p>$\\text{Gini} = 1 - \\sum_i p_i^2$</p>
    <ul>
      <li>Probability of misclassifying a randomly chosen element</li>
      <li>Range: 0 (pure node, all samples same class) to 0.5 (binary, 50-50 split)</li>
      <li>For binary classification: $\\text{Gini} = 1 - (p^2 + (1-p)^2) = 2p(1-p)$</li>
      <li>Fast to compute (no logarithms)</li>
      <li>Tends to isolate the most frequent class into pure nodes</li>
    </ul>
    
    <p><strong>2. Entropy (Information Gain)</strong>:</p>
    <p>$\\text{Entropy} = -\\sum_i p_i \\log_2(p_i)$</p>
    <ul>
      <li>Measures information or uncertainty in bits</li>
      <li>Range: 0 (pure) to $\\log_2(K)$ for K classes (binary: 0 to 1 bit)</li>
      <li>Information Gain = $\\text{Entropy}(\\text{parent}) - \\text{Weighted Average Entropy}(\\text{children})$</li>
      <li>More computationally expensive than Gini</li>
      <li>More sensitive to changes in probabilities</li>
      <li>Theoretical foundation in information theory</li>
    </ul>
    
    <p>In practice, Gini and Entropy produce similar trees. Gini is preferred for speed; Entropy for information-theoretic interpretability.</p>
    
    <p><strong>Regression Criteria:</strong></p>
    
    <p><strong>1. Mean Squared Error (MSE)</strong>:</p>
    <p>$\\text{MSE} = \\frac{1}{n} \\sum (y_i - \\bar{y})^2$</p>
    <ul>
      <li>Measures variance within a node</li>
      <li>Splits minimize weighted sum of child MSEs</li>
      <li>Equivalent to maximizing variance reduction</li>
      <li>Sensitive to outliers (squared term)</li>
      <li>Standard choice for regression trees</li>
    </ul>
    
    <p><strong>2. Mean Absolute Error (MAE)</strong>:</p>
    <p>$\\text{MAE} = \\frac{1}{n} \\sum |y_i - \\text{median}(y)|$</p>
    <ul>
      <li>More robust to outliers than MSE</li>
      <li>Uses median instead of mean for predictions</li>
      <li>Linear penalty for errors</li>
    </ul>

    <h3>Stopping Criteria: Controlling Tree Growth</h3>
    
    <p>Trees continue growing until stopping criteria are met. These hyperparameters control model complexity:</p>
    
    <ul>
      <li><strong>max_depth:</strong> Maximum depth of the tree (typical: 3-10). Limits how many questions can be asked in sequence. Deeper trees capture more complex patterns but risk overfitting.</li>
      <li><strong>min_samples_split:</strong> Minimum samples required to split a node (typical: 20-50). Prevents splits on very small groups where patterns might be noise.</li>
      <li><strong>min_samples_leaf:</strong> Minimum samples required in a leaf node (typical: 10-20). Ensures predictions are based on sufficient data.</li>
      <li><strong>max_leaf_nodes:</strong> Maximum number of leaf nodes. Alternative to max_depth for limiting complexity.</li>
      <li><strong>min_impurity_decrease:</strong> Minimum impurity reduction required to split. Only splits that improve purity sufficiently are made.</li>
    </ul>
    
    <p>Smaller values (deeper trees, fewer samples) → more complex model → higher risk of overfitting. Larger values → simpler model → may underfit.</p>

    <h3>Making Predictions</h3>
    
    <p><strong>Classification:</strong> Traverse the tree from root to leaf following the path determined by feature values. The leaf node contains class probabilities based on training samples that reached it. Predict the majority class or output probabilities.</p>
    
    <p><strong>Regression:</strong> Same traversal, but the leaf outputs the mean (or median with MAE) of training targets that reached it.</p>
    
    <p>Prediction is fast: O(log n) for balanced trees, O(depth) in general. The tree creates a piecewise-constant approximation of the target function—constant predictions within each rectangular region of feature space.</p>

    <h3>Feature Importance</h3>
    
    <p>Decision trees automatically calculate feature importance based on impurity reduction:</p>
    
    <p><strong>$\\text{Importance(feature)} = \\sum \\text{(weighted impurity decrease for all splits using that feature)}$</strong></p>
    
    <ul>
      <li>Features used higher in the tree (near root) typically have higher importance</li>
      <li>Features used in many splits accumulate importance</li>
      <li>Features never used have zero importance</li>
      <li>Scores normalized to sum to 1</li>
    </ul>
    
    <p><strong>Cautions:</strong></p>
    <ul>
      <li><strong>Bias toward high-cardinality features:</strong> Features with more unique values have more split opportunities</li>
      <li><strong>Instability:</strong> Small data changes can drastically alter importance rankings</li>
      <li><strong>Correlation effects:</strong> Correlated features compete; one may be selected arbitrarily, masking the other's importance</li>
      <li>Use ensemble methods (Random Forest) for more stable importance estimates</li>
    </ul>

    <h3>Handling Categorical Variables</h3>
    
    <p>Decision trees can handle categorical features natively (in some implementations like R's rpart), treating each category as a potential split point. For binary splits with K categories, this requires evaluating 2^(K-1) - 1 possible partitions, which is expensive for high-cardinality features.</p>
    
    <p>Scikit-learn requires preprocessing: one-hot encode categorical variables before training. Each category becomes a binary feature (0/1), and the tree learns splits like "is_category_A = 1". This is less efficient (creates many features) but ensures the tree can leverage categorical information.</p>

    <h3>Handling Missing Values</h3>
    
    <p>Decision trees handle missing values better than most algorithms:</p>
    <ul>
      <li><strong>Surrogate splits:</strong> Find backup features that produce similar partitions. If primary feature is missing, use the best surrogate (CART algorithm).</li>
      <li><strong>Learn missing direction:</strong> Try sending missing values left vs right, choose the direction that maximizes impurity reduction (XGBoost, LightGBM).</li>
      <li><strong>Treat as separate category:</strong> Missing becomes its own branch (for categorical features).</li>
    </ul>
    
    <p>This native handling is a major advantage over linear models, which require explicit imputation.</p>

    <h3>Pruning: Preventing Overfitting</h3>
    
    <p><strong>Pre-Pruning (Early Stopping):</strong> Stop growing the tree early using stopping criteria (max_depth, min_samples_split, etc.). Fast and simple but may suffer from "horizon effect"—stopping before discovering good splits deeper in the tree.</p>
    
    <p><strong>Post-Pruning (Cost-Complexity Pruning):</strong> Grow a full tree, then remove branches that don't improve validation performance. Define cost: Total Cost = Error + α × (number of leaves). Find the subtree that minimizes this cost for various α values, then select α via cross-validation. More principled than pre-pruning but computationally expensive.</p>
    
    <p>In scikit-learn, use <code>ccp_alpha</code> parameter for post-pruning. In practice, pre-pruning with moderate constraints (max_depth=5-10, min_samples_split=20-50) often works well and is much faster.</p>

    <h3>Advantages of Decision Trees</h3>
    <ul>
      <li><strong>Highly Interpretable:</strong> Visual tree structure shows the entire decision process. Easy to explain predictions to non-technical stakeholders.</li>
      <li><strong>Minimal Preprocessing:</strong> No need for feature scaling, normalization, or handling of categorical variables (in some implementations). Works with mixed data types.</li>
      <li><strong>Captures Non-Linearity:</strong> Automatically models complex non-linear relationships and interactions without manual feature engineering.</li>
      <li><strong>Handles Missing Values:</strong> Native support through surrogates or learned directions.</li>
      <li><strong>Feature Selection:</strong> Automatically ignores irrelevant features by not using them for splits.</li>
      <li><strong>Non-Parametric:</strong> Makes no assumptions about data distributions.</li>
      <li><strong>Fast Training and Prediction:</strong> O(n log n) training, O(log n) prediction for balanced trees.</li>
      <li><strong>Works for Classification and Regression:</strong> Unified framework for both tasks.</li>
    </ul>

    <h3>Disadvantages of Decision Trees</h3>
    <ul>
      <li><strong>Prone to Overfitting:</strong> Unpruned trees grow complex, memorizing training noise. Requires careful tuning of stopping criteria.</li>
      <li><strong>High Variance (Instability):</strong> Small changes in data can produce completely different trees. Makes them unreliable for inference about feature effects.</li>
      <li><strong>Greedy Algorithm:</strong> Makes locally optimal splits without considering future splits. Can miss globally better solutions.</li>
      <li><strong>Poor Extrapolation:</strong> Cannot predict outside the range of training data. Predictions are constant beyond training bounds.</li>
      <li><strong>Difficulty with Linear Relationships:</strong> Requires many splits to approximate a simple linear relationship (inefficient representation).</li>
      <li><strong>Bias Toward High-Cardinality Features:</strong> Features with many unique values get more split opportunities, appearing more important than they are.</li>
      <li><strong>Class Imbalance Issues:</strong> Tends to favor majority class without proper handling (class weights).</li>
      <li><strong>Axis-Aligned Splits:</strong> Can only split parallel to axes (feature boundaries), making diagonal decision boundaries inefficient.</li>
    </ul>

    <h3>Practical Recommendations</h3>
    <ul>
      <li><strong>Start with Moderate Constraints:</strong> max_depth=5-10, min_samples_split=20-50, min_samples_leaf=10-20. Tune via cross-validation.</li>
      <li><strong>Use Cross-Validation:</strong> Trees are sensitive to data; CV gives reliable performance estimates.</li>
      <li><strong>Visualize the Tree:</strong> Use <code>plot_tree</code> to understand what the model learned. Helps diagnose overfitting.</li>
      <li><strong>Check Feature Importance:</strong> Identify which features drive predictions. Useful for feature selection and understanding.</li>
      <li><strong>Consider Ensembles:</strong> Single trees are unstable and overfit. Random Forests and Gradient Boosting address these issues while sacrificing interpretability.</li>
      <li><strong>Handle Class Imbalance:</strong> Use <code>class_weight='balanced'</code> or adjust thresholds to prevent majority class dominance.</li>
      <li><strong>Use for Exploration:</strong> Decision trees are excellent for initial data exploration, revealing important features and interactions before trying complex models.</li>
      <li><strong>Beware Production Use:</strong> Single trees are rarely deployed in production due to instability. Use ensembles for better robustness.</li>
    </ul>

    <h3>When to Use Decision Trees</h3>
    <ul>
      <li><strong>Interpretability is Critical:</strong> When you must explain predictions (medical diagnosis, loan approval, legal contexts)</li>
      <li><strong>Exploratory Analysis:</strong> Quick baseline model to understand feature importance and interactions</li>
      <li><strong>Mixed Data Types:</strong> Data contains both numerical and categorical features</li>
      <li><strong>Non-Linear Relationships:</strong> Underlying patterns are non-linear or involve feature interactions</li>
      <li><strong>Missing Data:</strong> Dataset has missing values and you want native handling</li>
      <li><strong>As Base Learners:</strong> Building blocks for Random Forests, Gradient Boosting, and other ensemble methods</li>
    </ul>

    <h3>Visual Understanding</h3>
    <p>Picture an upside-down tree structure starting from a single box (root node) at the top. Each internal box asks a yes/no question about a feature ("Is age > 30?"). Two branches emerge: one for "yes", one for "no", leading to more boxes with more questions. This continues until you reach leaf boxes at the bottom containing final predictions. Following any path from root to leaf is like playing "20 questions"—a series of if-then rules.</p>
    
    <p><strong>Key visualizations to understand:</strong></p>
    <ul>
      <li><strong>Tree diagram:</strong> Nodes show split conditions ("age ≤ 30"), branches show paths, leaves show predictions. Node color intensity often indicates class probability or average value. Deeper trees have more levels and more complex decision paths.</li>
      <li><strong>Decision boundary plot (2D):</strong> Feature space divided into rectangular regions (axis-aligned boxes). Each region corresponds to a leaf node. Boundaries are always parallel to feature axes—trees can't create diagonal boundaries directly.</li>
      <li><strong>Feature importance bar chart:</strong> Bars showing each feature's contribution to impurity reduction across all splits. Longer bars = more important features.</li>
      <li><strong>Learning curves:</strong> Training vs validation accuracy as tree depth increases. Training accuracy rises to 100% (overfitting), while validation accuracy peaks then declines, showing the optimal depth.</li>
    </ul>

    <h3>Worked Example: Building a Simple Tree by Hand</h3>
    <p>Let's build a decision tree to predict "Play Tennis" (Yes/No) based on weather conditions:</p>
    
    <table>
      <tr><th>Outlook</th><th>Temp (°F)</th><th>Humidity (%)</th><th>Wind</th><th>Play?</th></tr>
      <tr><td>Sunny</td><td>85</td><td>85</td><td>Weak</td><td>No</td></tr>
      <tr><td>Sunny</td><td>80</td><td>90</td><td>Strong</td><td>No</td></tr>
      <tr><td>Overcast</td><td>83</td><td>78</td><td>Weak</td><td>Yes</td></tr>
      <tr><td>Rain</td><td>70</td><td>96</td><td>Weak</td><td>Yes</td></tr>
      <tr><td>Rain</td><td>68</td><td>80</td><td>Weak</td><td>Yes</td></tr>
      <tr><td>Rain</td><td>65</td><td>70</td><td>Strong</td><td>No</td></tr>
      <tr><td>Overcast</td><td>64</td><td>65</td><td>Strong</td><td>Yes</td></tr>
      <tr><td>Sunny</td><td>72</td><td>95</td><td>Weak</td><td>No</td></tr>
    </table>
    
    <p><strong>Step 1: Calculate root impurity (Gini)</strong></p>
    <ul>
      <li>Total: 8 samples (5 Yes, 3 No)</li>
      <li>Gini = 1 - (P(Yes)² + P(No)²) = 1 - ((5/8)² + (3/8)²) = 1 - (0.391 + 0.141) = 0.468</li>
    </ul>
    
    <p><strong>Step 2: Evaluate split on "Outlook"</strong></p>
    <ul>
      <li><strong>Sunny (3 samples):</strong> 0 Yes, 3 No → Gini = 1 - (0² + 1²) = 0 (pure!)</li>
      <li><strong>Overcast (2 samples):</strong> 2 Yes, 0 No → Gini = 1 - (1² + 0²) = 0 (pure!)</li>
      <li><strong>Rain (3 samples):</strong> 2 Yes, 1 No → Gini = 1 - ((2/3)² + (1/3)²) = 1 - (0.444 + 0.111) = 0.445</li>
      <li><strong>Weighted average:</strong> (3/8)×0 + (2/8)×0 + (3/8)×0.445 = 0.167</li>
      <li><strong>Information gain:</strong> 0.468 - 0.167 = 0.301 ✓ (Good split!)</li>
    </ul>
    
    <p><strong>Step 3: Compare with split on "Humidity > 80"</strong></p>
    <ul>
      <li><strong>High humidity (4 samples):</strong> 1 Yes, 3 No → Gini = 1 - ((1/4)² + (3/4)²) = 1 - (0.063 + 0.563) = 0.375</li>
      <li><strong>Normal humidity (4 samples):</strong> 4 Yes, 0 No → Gini = 0</li>
      <li><strong>Weighted average:</strong> (4/8)×0.375 + (4/8)×0 = 0.188</li>
      <li><strong>Information gain:</strong> 0.468 - 0.188 = 0.280 (Good, but less than Outlook)</li>
    </ul>
    
    <p><strong>Decision: Split on "Outlook" (higher gain)</strong></p>
    
    <p><strong>Resulting tree:</strong></p>
    <pre>
    Root: [5 Yes, 3 No]
        |
        ├─ Outlook = Sunny → Predict: No (pure)
        |
        ├─ Outlook = Overcast → Predict: Yes (pure)
        |
        └─ Outlook = Rain → [2 Yes, 1 No]
              └─ Further split on Wind or Humidity...
    </pre>
    
    <p>For the Rain branch (still impure), we'd continue splitting on the next best feature until reaching stopping criteria (max_depth, min_samples, or pure nodes).</p>

    <h3>Common Mistakes to Avoid</h3>
    <ul>
      <li><strong>❌ Using unpruned trees on noisy data:</strong> Without depth limits or pruning, trees memorize training noise. Always set max_depth (5-10) or use post-pruning, especially for small datasets.</li>
      <li><strong>❌ Ignoring feature importance bias:</strong> Trees favor high-cardinality features (many unique values) because they offer more split opportunities. Don't blindly trust importance rankings without considering this bias.</li>
      <li><strong>❌ Forgetting that trees can't extrapolate:</strong> Decision trees predict constant values (leaf node predictions). Outside training range, they return the nearest leaf value. For time series or trend prediction, trees are poor choices.</li>
      <li><strong>❌ Applying feature scaling:</strong> Unlike distance-based methods, trees don't need feature scaling—splits are based on thresholds invariant to scale. Scaling wastes computation and doesn't help.</li>
      <li><strong>❌ Using deep trees for production:</strong> Single deep trees are unstable and overfit. Use ensembles (Random Forest, Gradient Boosting) for production systems—they're more robust and accurate.</li>
      <li><strong>❌ Not handling class imbalance:</strong> Trees can be biased toward majority class. Use class_weight='balanced' or stratified sampling to ensure minority class representation.</li>
      <li><strong>❌ Treating greedy algorithm as optimal:</strong> Trees make locally optimal splits without lookahead. A seemingly poor split might enable excellent child splits, but the algorithm won't discover this. Accept that trees are approximate, not optimal.</li>
      <li><strong>❌ Over-interpreting single tree structure:</strong> Small data changes completely alter tree structure (high variance). Don't make strong conclusions about feature relationships from one tree. Use ensembles for stable interpretations.</li>
    </ul>

    <h3>Summary</h3>
    <p>Decision trees provide an intuitive, interpretable framework for both classification and regression through hierarchical decision making. Their ability to capture non-linear patterns, handle mixed data types, and require minimal preprocessing makes them versatile and easy to use. However, their tendency to overfit, high variance, and greedy splitting algorithm limit their standalone performance. In practice, decision trees shine as exploratory tools for understanding data and as building blocks for ensemble methods (Random Forests, Gradient Boosting) that aggregate many trees to achieve state-of-the-art predictive performance while retaining some interpretability through feature importance measures. Master decision trees deeply—understanding splitting criteria, pruning strategies, and their biases—as they form the foundation for some of the most powerful machine learning algorithms in production today.</p>
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
};
