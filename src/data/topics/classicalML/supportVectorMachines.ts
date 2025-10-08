import { Topic } from '../../../types';

export const supportVectorMachines: Topic = {
  id: 'support-vector-machines',
  title: 'Support Vector Machines (SVM)',
  category: 'classical-ml',
  description: 'Powerful classification algorithm that finds optimal decision boundaries',
  content: `
    <h2>Support Vector Machines: Maximum Margin Classification</h2>
    <p>Support Vector Machines represent one of the most elegant and theoretically grounded approaches to machine learning, combining geometric intuition with rigorous mathematical optimization. SVMs find the decision boundary (hyperplane) that maximally separates different classes—not just any boundary that works, but the one with maximum confidence (margin). This maximum margin principle, grounded in statistical learning theory, provides strong generalization guarantees: by maximizing the distance to the nearest training examples, SVMs minimize a bound on generalization error, not just training error.</p>

    <p>The "support vectors" are the critical training examples that lie on the margin boundaries—the points closest to the decision boundary. These examples alone define the classifier; all other points could be removed without changing the solution. This sparsity makes SVMs both elegant (most training data is redundant) and efficient (prediction depends only on support vectors). The algorithm's extension to non-linear boundaries through the kernel trick—implicitly mapping data to high-dimensional spaces without ever computing that mapping explicitly—is a triumph of mathematical insight that enables SVMs to handle complex decision boundaries while maintaining computational tractability.</p>

    <h3>The Core Intuition: Maximum Margin Classification</h3>
    <p>Imagine drawing a line to separate two clusters of points. You could draw infinitely many lines that separate them, but intuitively, a line that passes very close to some points seems risky—any noise or slight perturbation might cause misclassification. SVM finds the line (in 2D) or hyperplane (in higher dimensions) with maximum clearance on both sides, creating the widest possible "street" between classes. This street width is called the margin, and maximizing it is equivalent to maximizing the model's confidence.</p>

    <p>The decision boundary is the center line of this street, defined by weights $\\mathbf{w}$ and bias $b$: $\\mathbf{w} \\cdot \\mathbf{x} + b = 0$. Points on one side ($\\mathbf{w} \\cdot \\mathbf{x} + b > 0$) belong to class +1, points on the other side ($\\mathbf{w} \\cdot \\mathbf{x} + b < 0$) belong to class -1. The margin boundaries are parallel to the decision boundary at distance ±1 from it: $\\mathbf{w} \\cdot \\mathbf{x} + b = +1$ (upper margin) and $\\mathbf{w} \\cdot \\mathbf{x} + b = -1$ (lower margin). Support vectors lie exactly on these boundaries—they're the points closest to the decision boundary from each class.</p>

    <p>The margin width is $\\frac{2}{||\\mathbf{w}||}$ (where $||\\mathbf{w}||$ is the Euclidean norm of $\\mathbf{w}$), so maximizing margin is equivalent to minimizing $||\\mathbf{w}||^2$. This transforms the problem into a convex quadratic optimization: minimize $\\frac{1}{2}||\\mathbf{w}||^2$ subject to $y_i(\\mathbf{w} \\cdot \\mathbf{x}_i + b) \\geq 1$ for all training points $i$. This constraint ensures all points are correctly classified and outside the margin. The solution is guaranteed to be unique (convexity) and found efficiently via quadratic programming.</p>

    <h3>Hard-Margin SVM: Perfect Separation</h3>
    <p>Hard-margin SVM assumes data is linearly separable—there exists a hyperplane that perfectly separates all training examples. The optimization is: $$\\text{minimize } \\frac{1}{2}||\\mathbf{w}||^2 \\text{ subject to } y_i(\\mathbf{w} \\cdot \\mathbf{x}_i + b) \\geq 1 \\text{ for all } i$$. This is an elegant formulation: the objective (minimizing $||\\mathbf{w}||$) maximizes margin, and the constraints ensure correct classification with margin at least 1.</p>

    <p>Hard-margin SVM works beautifully on toy datasets but fails in practice for two reasons: (1) Real-world data is rarely linearly separable due to noise, measurement error, or inherent class overlap. If no separating hyperplane exists, the optimization is infeasible. (2) Even if data is separable, a single outlier can drastically reduce the margin, yielding a classifier that generalizes poorly. The hard-margin formulation is too rigid, treating all training examples as equally important and demanding perfect classification.</p>

    <h3>Soft-Margin SVM: Tolerating Violations</h3>
    <p>Soft-margin SVM relaxes the hard constraints by introducing slack variables $\\xi_i$ for each training point, allowing controlled violations of the margin. The formulation becomes: $$\\text{minimize } \\frac{1}{2}||\\mathbf{w}||^2 + C \\cdot \\sum \\xi_i \\text{ subject to } y_i(\\mathbf{w} \\cdot \\mathbf{x}_i + b) \\geq 1 - \\xi_i \\text{ and } \\xi_i \\geq 0$$. The slack variable $\\xi_i$ measures the violation for point $i$: if $\\xi_i = 0$, the point is correctly classified outside the margin (ideal); if $0 < \\xi_i < 1$, the point is correctly classified but inside the margin (margin violation); if $\\xi_i \\geq 1$, the point is misclassified (wrong side of the decision boundary).</p>

    <p>The hyperparameter $C$ controls the trade-off between margin size and violations. <strong>Large $C$</strong> (e.g., 100, 1000) heavily penalizes violations: the model tries hard to classify all training points correctly, even at the cost of a smaller margin. This leads to a complex decision boundary that closely fits training data (low bias, high variance, prone to overfitting). With very large $C$, soft-margin SVM approaches hard-margin behavior. <strong>Small $C$</strong> (e.g., 0.01, 0.1) gives low penalty to violations, allowing many points to be misclassified or inside the margin in favor of a wider margin. This produces a simpler, smoother decision boundary (high bias, low variance, strong regularization) that generalizes better by not trying to fit every training point perfectly.</p>

    <p>Intuitively, $C$ balances two competing objectives: fitting training data ($\\sum \\xi_i$ should be small) and maximizing margin ($||\\mathbf{w}||^2$ should be small). Small $C$ emphasizes margin, large $C$ emphasizes fitting. The optimal $C$ depends on data characteristics: for clean, separable data, large $C$ works well; for noisy, overlapping classes, small $C$ prevents overfitting. Tune $C$ via cross-validation, searching log-scale values like [0.01, 0.1, 1, 10, 100]. The relationship to regularization in other models: $C$ is inversely proportional to $\\lambda$ in Ridge regression ($C = \\frac{1}{2\\lambda}$), so <strong>small $C$ = strong regularization</strong>.</p>

    <h3>Support Vectors: The Critical Points</h3>
    <p>Support vectors are training points that lie exactly on the margin boundaries or violate the margin. In the dual formulation of SVM (derived via Lagrange multipliers), the decision function is: $$f(\\mathbf{x}) = \\sum (\\alpha_i y_i K(\\mathbf{x}_i, \\mathbf{x})) + b$$, where $\\alpha_i$ are learned weights (Lagrange multipliers) and $K$ is the kernel function. Crucially, most $\\alpha_i$ are zero; only support vectors have $\\alpha_i > 0$. This means non-support vectors contribute nothing to the decision function—they could be removed from the training set without changing the model.</p>

    <p>This sparsity has profound implications: <strong>Memory efficiency</strong>—only support vectors need to be stored (typically 10-50% of training data, depending on problem difficulty and $C$). For a dataset with 10,000 training points, you might only store 2,000 support vectors. <strong>Prediction efficiency</strong>—computing $f(\\mathbf{x})$ requires evaluating the kernel only between the test point and support vectors, not all training points. <strong>Interpretability</strong>—support vectors are the "difficult" examples that define the decision boundary. Points with $\\alpha_i = C$ (at the upper bound) are problematic: they lie inside the margin or are misclassified. Points with $0 < \\alpha_i < C$ lie exactly on the margin boundaries.</p>

    <p>The number of support vectors provides insight into problem difficulty. Very few support vectors (< 10% of data) suggest well-separated classes or potential underfitting. Many support vectors (> 50% of data) suggest overlapping classes, noisy data, or overfitting (too large $C$). The support vectors identify the boundary region where the model is uncertain—far from the boundary, classification is confident and doesn't depend on these specific examples.</p>

    <h3>The Kernel Trick: Non-Linear Classification</h3>
    <p>Linear SVMs find linear decision boundaries: $\\mathbf{w} \\cdot \\mathbf{x} + b = 0$, a straight line (2D), plane (3D), or hyperplane (higher dimensions). Real-world data often has non-linear decision boundaries: concentric circles, XOR patterns, curved separations. A naive approach would explicitly transform features into a higher-dimensional space where linear separation is possible, then apply linear SVM. For example, transforming 2D data $[x_1, x_2]$ into 5D via $\\phi(\\mathbf{x}) = [x_1, x_2, x_1^2, x_2^2, x_1x_2]$, then finding a hyperplane in 5D. But this is computationally expensive: high-dimensional transformations require computing and storing many features.</p>

    <p>The <strong>kernel trick</strong> avoids explicit transformation by observing that the SVM dual formulation only requires dot products: $f(\\mathbf{x}) = \\sum (\\alpha_i y_i \\phi(\\mathbf{x}_i) \\cdot \\phi(\\mathbf{x})) + b$. If we define a kernel function $K(\\mathbf{x}_i, \\mathbf{x}_j) = \\phi(\\mathbf{x}_i) \\cdot \\phi(\\mathbf{x}_j)$ that computes the dot product in the transformed space directly, we never need to compute $\\phi(\\mathbf{x})$ explicitly. The kernel computes the similarity between two points in the high-dimensional space using only the original features.</p>

    <h4>Common Kernels and Their Intuitions</h4>
    <ul>
      <li><strong>Linear Kernel: $K(\\mathbf{x}, \\mathbf{x}') = \\mathbf{x} \\cdot \\mathbf{x}'$</strong>
        <p>No transformation, standard dot product. Use when data is linearly separable or you want interpretability (coefficients $\\mathbf{w}$ are meaningful). Fastest to compute and train.</p>
      </li>
      <li><strong>Polynomial Kernel: $K(\\mathbf{x}, \\mathbf{x}') = (\\gamma \\mathbf{x} \\cdot \\mathbf{x}' + r)^d$</strong>
        <p>Corresponds to mapping into a space of all polynomial combinations up to degree $d$. For $d=2$, transforms $[x_1, x_2]$ into $[x_1^2, x_2^2, \\sqrt{2}x_1x_2, \\sqrt{2}x_1, \\sqrt{2}x_2, 1]$. Captures polynomial decision boundaries (parabolas, ellipses). Parameter $d$ (degree, typically 2-5) controls complexity; $\\gamma$ scales the dot product; $r$ is a constant. Higher $d$ = more complex boundaries but risk of overfitting.</p>
      </li>
      <li><strong>RBF (Radial Basis Function / Gaussian Kernel): $K(\\mathbf{x}, \\mathbf{x}') = \\exp(-\\gamma||\\mathbf{x} - \\mathbf{x}'||^2)$</strong>
        <p>The most popular kernel. Measures similarity based on Euclidean distance: $K \\approx 1$ when $\\mathbf{x}$ and $\\mathbf{x}'$ are close (similar), $K \\approx 0$ when far apart (dissimilar). Corresponds to mapping into an infinite-dimensional space, making it a universal kernel—with appropriate $\\gamma$, it can approximate any continuous function. Parameter $\\gamma$ controls the "reach" of each training example: <strong>low $\\gamma$</strong> (e.g., 0.001) = each example influences a large region, creating smooth decision boundaries; <strong>high $\\gamma$</strong> (e.g., 10) = each example influences only nearby points, creating complex, wiggly boundaries (risk of overfitting). Tune $C$ and $\\gamma$ together via grid search.</p>
      </li>
      <li><strong>Sigmoid Kernel: $K(\\mathbf{x}, \\mathbf{x}') = \\tanh(\\gamma \\mathbf{x} \\cdot \\mathbf{x}' + r)$</strong>
        <p>Behaves like a neural network with one hidden layer. Less commonly used in practice; can be unstable for some parameter values.</p>
      </li>
    </ul>

    <p>The kernel trick's elegance: for RBF kernel with infinite-dimensional mapping, we compute $K(\\mathbf{x}, \\mathbf{x}') = \\exp(-\\gamma||\\mathbf{x} - \\mathbf{x}'||^2)$ in $O(d)$ time ($d$ = original dimensions) instead of computing an infinite-dimensional dot product (impossible!). This enables powerful non-linear classification while maintaining computational efficiency. The Gram matrix ($K_{ij} = K(\\mathbf{x}_i, \\mathbf{x}_j)$) of kernel values for all training pairs is the only additional structure needed, an $n \\times n$ matrix where $n$ is the number of training points.</p>

    <h3>Hyperparameter Tuning: C and Gamma</h3>
    <p>SVM performance is highly sensitive to hyperparameters. For linear SVM, tune $C$ only. For RBF (most common), tune both $C$ and $\\gamma$. These parameters interact: different $(C, \\gamma)$ combinations can produce similar accuracy but with different complexity and generalization. <strong>Grid search</strong> is standard: define ranges $C \\in [0.01, 0.1, 1, 10, 100, 1000]$ and $\\gamma \\in [0.001, 0.01, 0.1, 1, 10]$, evaluate all $6 \\times 5=30$ combinations via cross-validation, select the best. Use log-scale spacing since parameters span orders of magnitude.</p>

    <p><strong>Typical patterns:</strong> If training accuracy $\\approx$ test accuracy and both are low, underfitting—increase $C$ or $\\gamma$ (more complex model). If training accuracy $\\gg$ test accuracy, overfitting—decrease $C$ or $\\gamma$ (simpler model). For RBF, $(C=1, \\gamma=\\frac{1}{\\text{n\\_features}})$ is a reasonable default starting point. Check learning curves: plot training/validation accuracy vs $C$ (holding $\\gamma$ fixed) and vs $\\gamma$ (holding $C$ fixed) to understand their effects. Modern libraries (scikit-learn) provide GridSearchCV and RandomizedSearchCV to automate this process with parallel cross-validation.</p>

    <h3>Advantages of SVM</h3>
    <ul>
      <li><strong>Effective in high dimensions:</strong> Works well when number of features exceeds number of samples (common in text, genomics). The margin-maximization principle provides good generalization even in high-dimensional spaces.</li>
      <li><strong>Memory efficient:</strong> Stores only support vectors (subset of training data), not the entire dataset. Crucial for large datasets where you can discard non-support vectors after training.</li>
      <li><strong>Versatile:</strong> Different kernels for different data structures (linear for text, RBF for complex patterns, string kernels for sequences, graph kernels for networks). Custom kernels can be designed for domain-specific similarity measures.</li>
      <li><strong>Strong theoretical guarantees:</strong> Maximum margin principle minimizes an upper bound on generalization error (VC dimension theory). Well-grounded in statistical learning theory.</li>
      <li><strong>Robust to overfitting in high dimensions:</strong> Margin maximization and regularization (via C) prevent overfitting better than naive methods. Works well even when features >> samples.</li>
      <li><strong>Global optimum:</strong> Convex optimization guarantees finding the global optimum (no local minima like neural networks). Reproducible results (deterministic, unlike stochastic methods with random initialization).</li>
    </ul>

    <h3>Disadvantages and Limitations</h3>
    <ul>
      <li><strong>Poor scalability:</strong> Training complexity is O(n²) to O(n³) where n is the number of training samples (due to computing Gram matrix and quadratic programming). Becomes prohibitively slow for n > 100,000. Prediction also requires computing kernel with all support vectors, though this is faster (O(n_sv × d) where n_sv is usually < n).</li>
      <li><strong>No native probability estimates:</strong> SVM outputs decision function values (distance to hyperplane), not probabilities. While probabilities can be estimated via Platt scaling or cross-validation, they're less reliable than methods that output probabilities natively (logistic regression, naive Bayes, neural networks).</li>
      <li><strong>Sensitive to feature scaling:</strong> Since SVM uses distance/dot product computations, features with larger scales dominate. Always standardize features before applying SVM. This is critical—forgetting to scale often leads to poor performance.</li>
      <li><strong>Black box with kernels:</strong> Non-linear kernels create complex decision boundaries that are hard to interpret. You know the model classifies well, but understanding why is difficult. Linear SVM provides interpretable weights w, but RBF SVM does not.</li>
      <li><strong>Hyperparameter sensitivity:</strong> Performance varies significantly with C and γ (for RBF). Requires extensive grid search or Bayesian optimization. Choosing wrong parameters can degrade performance drastically.</li>
      <li><strong>Struggles with noise and overlap:</strong> If classes heavily overlap or labels are noisy, SVM may not find a satisfactory solution. Decision boundary tries to separate everything, leading to unstable results. Methods that explicitly model uncertainty (Gaussian Processes, probabilistic classifiers) may be better.</li>
      <li><strong>Not ideal for very large or very small datasets:</strong> Large datasets (>100k samples): too slow, use linear SVM with stochastic optimization (e.g., LinearSVC with SGD), Logistic Regression, or tree-based methods. Very small datasets (<100 samples): SVM may overfit; try simpler models or regularization.</li>
    </ul>

    <h3>Linear vs RBF Kernel: When to Use Which</h3>
    <p><strong>Use Linear Kernel when:</strong> Features > Samples (high-dimensional, e.g., text with 10,000+ words, genomics with thousands of genes). In high dimensions, data is often approximately linearly separable, and complex kernels risk overfitting. Linear SVM is also much faster (O(n) vs O(n²)), scales to millions of samples with LinearSVC, provides interpretability (feature weights), and works well for sparse data (text, one-hot encoded features). Text classification with TF-IDF features almost always uses linear SVM.</p>

    <p><strong>Use RBF Kernel when:</strong> Features < Samples (low/medium dimensions, e.g., tabular data with 10-100 features), complex non-linear decision boundaries (image features, sensor data, engineered features), you suspect feature interactions are important, and you can afford the computational cost (n < 10,000 samples). RBF is a universal approximator and can fit almost any continuous function with proper hyperparameters, making it a powerful default for non-linear problems.</p>

    <p><strong>Practical workflow:</strong> Always try linear SVM first (fast, interpretable, strong baseline, especially for high-dimensional data). If performance is unsatisfactory, try RBF with grid search over C and γ. Check learning curves: if linear SVM has high training error, the model is underfitting—RBF might help. If linear SVM has low training error but high test error, overfitting—increase regularization (reduce C) or simplify data (feature selection). For very large datasets, use LinearSVC with SGD (stochastic gradient descent), which scales to millions of samples. For complex tasks where SVM is too slow, consider tree-based methods (Random Forest, XGBoost) which often outperform SVM on tabular data and scale better.</p>

    <h3>SVM vs Other Classifiers</h3>
    <p><strong>SVM vs Logistic Regression:</strong> Both are linear classifiers (in the original space), but LR minimizes log loss (probabilistic) while SVM maximizes margin (geometric). LR provides calibrated probabilities; SVM provides better separation. For high-dimensional data, both perform similarly. LR is faster and easier to tune; SVM with RBF kernel is more flexible but slower.</p>

    <p><strong>SVM vs Neural Networks:</strong> Neural networks can learn arbitrary non-linear mappings through depth, are highly flexible, and scale to massive datasets with stochastic gradient descent. SVMs are simpler, have fewer hyperparameters (C, γ vs architecture, learning rate, regularization, initialization), and work well with small-to-medium data (100-10,000 samples). For images/text/audio, neural networks dominate; for tabular data with < 10,000 samples, SVM is competitive.</p>

    <p><strong>SVM vs Tree-Based Methods (Random Forest, XGBoost):</strong> Tree-based methods handle mixed data types (categorical + numerical) naturally, don't require feature scaling, provide feature importance, and scale well. SVM requires careful preprocessing, is sensitive to scaling, and doesn't handle categorical features directly. For tabular data in practice, gradient boosting (XGBoost, LightGBM) often outperforms SVM and is faster. SVM shines when maximum-margin properties are beneficial or for specific kernel tricks (string kernels for text, graph kernels for networks).</p>

    <h3>Visual Understanding</h3>
    <p>Picture two clusters of colored points (red and blue) on a 2D plane, separated by various possible lines. SVM finds the line that maximizes the "buffer zone" (margin) between the clusters. This line sits exactly in the middle of the widest corridor you can draw without touching any points. The points closest to the line (touching the margin boundaries) are support vectors—they determine the line's position. All other points could be removed without changing the decision boundary.</p>
    
    <p><strong>Key visualizations to understand:</strong></p>
    <ul>
      <li><strong>Linear SVM decision boundary:</strong> 2D scatter plot with red/blue points, a decision line (hyperplane), and two parallel dashed lines (margin boundaries). Support vectors are circled. The margin (width between dashed lines) is maximized. Points outside margins don't affect the boundary—only support vectors matter.</li>
      <li><strong>Soft margin with slack:</strong> Similar to above, but some points violate the margin or even cross to the wrong side (misclassified). These have slack variables ξ_i > 0, shown as short line segments from the point to where it "should" be. Parameter C controls tolerance: high C = few violations (narrow margin), low C = many violations (wide margin, more robust).</li>
      <li><strong>RBF kernel transformation:</strong> 2D data that's not linearly separable (e.g., red points inside, blue points outside a circle). In original space, no line separates them. RBF kernel implicitly maps to infinite dimensions where a hyperplane does separate them. Show before (non-separable circles) and after (conceptually, a 3D plot where classes lift to different heights, now linearly separable).</li>
      <li><strong>Effect of C parameter:</strong> Side-by-side plots for C=0.1 (wide margin, many misclassifications, smooth boundary), C=1 (moderate), C=100 (narrow margin, few violations, jagged boundary that overfits training noise). Demonstrates regularization tradeoff.</li>
      <li><strong>Effect of γ parameter (RBF):</strong> Low γ=0.01 (each point's influence extends far, smooth boundary, underfits), medium γ=1 (balanced), high γ=10 (each point's influence is local, decision boundary wraps tightly around individual points, overfits). Shows complexity control.</li>
    </ul>

    <h3>Common Mistakes to Avoid</h3>
    <ul>
      <li><strong>❌ Forgetting feature scaling:</strong> SVM is extremely sensitive to scale. Features with large ranges dominate the margin calculation. ALWAYS use StandardScaler before SVM. This is the #1 reason SVM performs poorly for beginners.</li>
      <li><strong>❌ Using RBF kernel with high-dimensional data:</strong> When features >> samples (e.g., text with 10,000 words, only 1,000 documents), data is often linearly separable in the original space. RBF adds complexity unnecessarily and slows training. Use linear SVM.</li>
      <li><strong>❌ Not tuning C and γ:</strong> Default parameters are arbitrary. Always grid search: C ∈ {0.1, 1, 10, 100}, γ ∈ {0.001, 0.01, 0.1, 1}. Performance can improve 10-20% with proper tuning.</li>
      <li><strong>❌ Applying SVM to large datasets without LinearSVC:</strong> Standard SVC is O(n²), too slow for n > 100k. Use LinearSVC (linear kernel only) which scales linearly via SGD, or switch to logistic regression / tree-based methods.</li>
      <li><strong>❌ Using SVM when you need probabilities:</strong> SVM outputs decision function values (distance to hyperplane), not probabilities. While SVC has probability=True, it fits a separate model (Platt scaling) on top, which is slower and less reliable than natively probabilistic models (LR, Naive Bayes, neural nets). If you need calibrated probabilities, use those models instead.</li>
      <li><strong>❌ Expecting SVM to handle categorical features:</strong> SVM requires numerical input. You must one-hot encode categoricals, which can explode dimensionality. Tree-based methods (Random Forest, XGBoost) handle categoricals natively—consider them for mixed-type data.</li>
      <li><strong>❌ Ignoring class imbalance:</strong> If 90% of samples are class A, SVM may predict everything as A for high accuracy. Use class_weight='balanced' to penalize errors on minority class more heavily.</li>
      <li><strong>❌ Not using cross-validation:</strong> Single train-test split can be misleading. Use 5-fold CV to get reliable performance estimates, especially when tuning hyperparameters.</li>
    </ul>

    <h3>Practical Recommendations</h3>
    <ul>
      <li><strong>Always standardize features:</strong> Use StandardScaler to transform features to mean=0, std=1 before training. This is non-negotiable for SVM.</li>
      <li><strong>Start with linear SVM:</strong> Fast, interpretable, works well for high-dimensional data. Use sklearn.svm.LinearSVC for large datasets (>10k samples) as it uses a faster optimization algorithm.</li>
      <li><strong>Try RBF if linear underperforms:</strong> Tune both C and γ via GridSearchCV with cross-validation. Standard ranges: C=[0.1, 1, 10, 100], γ=[0.001, 0.01, 0.1, 1].</li>
      <li><strong>Use cross-validation:</strong> 5-fold CV provides robust estimates. Don't rely on a single train-test split.</li>
      <li><strong>For large datasets (>100k samples):</strong> SVM is too slow. Use LinearSVC with SGD, Logistic Regression, or tree-based methods (Random Forest, XGBoost).</li>
      <li><strong>For imbalanced classes:</strong> Use class_weight='balanced' to automatically adjust weights inversely proportional to class frequencies, or set custom weights via class_weight parameter.</li>
      <li><strong>Check learning curves:</strong> Plot training and validation accuracy vs training set size to diagnose underfitting (both low) or overfitting (training high, validation low).</li>
      <li><strong>Consider alternatives for production:</strong> If prediction latency is critical and you have many support vectors, consider ensemble methods or neural networks that may be faster at inference time.</li>
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
};
