import { Topic } from '../../../types';

export const principalComponentAnalysis: Topic = {
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
      <li><strong>First principal component (PC1):</strong> Direction of maximum variance in the data. Imagine finding the axis along which data is most spread out.</li>
      <li><strong>Second principal component (PC2):</strong> Direction of maximum remaining variance, orthogonal (perpendicular) to PC1. The second-most spread axis, independent of the first.</li>
      <li><strong>Subsequent PCs:</strong> Each orthogonal to all previous, capturing remaining variance in decreasing order</li>
      <li>Components are ordered by variance explained—PC1 > PC2 > PC3 > ...</li>
      <li>Transform data by projecting onto selected components (matrix multiplication)</li>
    </ul>
    
    <p><strong>Intuitive Analogy:</strong> Imagine photographing a pencil. If you take the photo from the side (along its length), you see maximum variation (length dimension). This is like PC1. Rotate 90° and photograph from the end; you see the pencil's cross-section (width dimension)—less variation. This is like PC2. PCA automatically finds these informative viewing angles for your data.</p>

    <h3>Mathematical Foundation</h3>
    <ol>
      <li><strong>Standardize data:</strong> Center by subtracting mean (X_centered = X - mean(X)) and scale by dividing by standard deviation (X_scaled = X_centered / std(X)). This makes features comparable.</li>
      <li><strong>Compute covariance matrix:</strong> C = (1/n)X^T X. This d×d matrix captures pairwise correlations between all features. Diagonal elements are variances; off-diagonal are covariances.</li>
      <li><strong>Eigendecomposition:</strong> Solve Cv = λv to find eigenvectors v (directions) and eigenvalues λ (variance along those directions). Each eigenvector is a principal component.</li>
      <li><strong>Sort by eigenvalues:</strong> Larger eigenvalues = more variance explained by that PC. Order: λ₁ ≥ λ₂ ≥ ... ≥ λ_d.</li>
      <li><strong>Select top k eigenvectors:</strong> Choose first k eigenvectors corresponding to k largest eigenvalues. These k vectors form the transformation matrix W (d × k).</li>
      <li><strong>Transform data:</strong> Project original data onto principal components: X_new = X · W. Result is n × k matrix (reduced from n × d).</li>
    </ol>
    
    <p><strong>Why eigendecomposition?</strong> Eigenvectors of the covariance matrix are the directions of maximum variance. Eigenvalues tell us how much variance. This is a deep result from linear algebra: the best k-dimensional linear subspace for representing data (minimizing reconstruction error) is spanned by the top k eigenvectors.</p>

    <h3>Variance Explained</h3>
    <ul>
      <li>Each eigenvalue represents variance captured by its principal component</li>
      <li><strong>Explained variance ratio:</strong> eigenvalue / sum(all eigenvalues)</li>
      <li><strong>Cumulative explained variance:</strong> Sum of variance ratios up to component k</li>
      <li>Typically retain components capturing 95-99% cumulative variance</li>
    </ul>

    <h3>Concrete Example: PCA on 3D Data</h3>
    <p>Consider a dataset with 3 features measuring student performance: [test_score, study_hours, assignments_completed]. After standardization, we compute the covariance matrix and find:</p>
    
    <ul>
      <li><strong>PC1:</strong> Explains 65% variance, loadings: [0.60, 0.55, 0.58]
        <ul>
          <li><strong>Interpretation:</strong> "Overall Academic Effort" — all three features contribute positively and similarly. Students with high PC1 scores high on tests, study long hours, and complete assignments.</li>
        </ul>
      </li>
      <li><strong>PC2:</strong> Explains 25% variance, loadings: [0.70, -0.50, -0.50]
        <ul>
          <li><strong>Interpretation:</strong> "Efficiency" — high test scores despite lower study hours and fewer assignments. Positive: test scores; Negative: study hours and assignments. High PC2 = high test efficiency.</li>
        </ul>
      </li>
      <li><strong>PC3:</strong> Explains 10% variance, loadings: [0.10, 0.65, -0.75]
        <ul>
          <li><strong>Interpretation:</strong> Contrast between study hours and assignments (with little test score contribution). Might represent "study strategy preference" but explains little variance—likely noise.</li>
        </ul>
      </li>
    </ul>
    
    <p>With k=2, we retain 90% variance and reduce from 3D to 2D. PC1 and PC2 provide interpretable axes: effort level and efficiency, capturing most information.</p>

    <h3>Interpreting Principal Components: A Detailed Guide</h3>
    <p>Principal components are linear combinations of original features. Understanding what each PC represents requires examining the <strong>loadings</strong> (weights).</p>
    
    <p><strong>Loading Analysis:</strong></p>
    <ul>
      <li><strong>Magnitude:</strong> Larger absolute values |wᵢⱼ| mean feature j contributes more to PCᵢ</li>
      <li><strong>Sign:</strong> Positive loadings increase PC value when feature increases; negative loadings decrease PC value</li>
      <li><strong>Pattern:</strong> Look for groups of features with similar loadings—they move together</li>
    </ul>
    
    <p><strong>Interpretation Workflow:</strong></p>
    <ol>
      <li>Examine the first few PCs (typically 1-3) that explain most variance</li>
      <li>Identify features with highest absolute loadings (|w| > 0.3 is a rough threshold)</li>
      <li>Group features by sign: which features increase together, which oppose?</li>
      <li>Assign semantic meaning based on domain knowledge</li>
      <li>Validate interpretation by plotting data in PC space colored by known attributes</li>
    </ol>
    
    <p><strong>Visualization Techniques:</strong></p>
    <ul>
      <li><strong>Loading plot:</strong> Bar chart showing feature contributions to PC1, PC2, etc.</li>
      <li><strong>Biplot:</strong> Scatter plot of data in PC1-PC2 space with arrows showing feature directions</li>
      <li><strong>Heatmap:</strong> Loadings matrix as heatmap (rows=PCs, cols=features) reveals patterns</li>
      <li><strong>Scatter with color:</strong> Plot PC1 vs PC2, color points by class/attribute to see what PCs capture</li>
    </ul>
    
    <p><strong>Common Interpretation Patterns:</strong></p>
    <ul>
      <li><strong>PC1 often represents "size" or "scale":</strong> All features have same sign → PC1 measures overall magnitude</li>
      <li><strong>PC2 often represents "contrast":</strong> Features split into positive/negative groups → PC2 measures difference between groups</li>
      <li><strong>Later PCs represent noise:</strong> No clear pattern, low variance → often discarded</li>
    </ul>
    
    <p><strong>Caveats:</strong></p>
    <ul>
      <li><strong>Sign ambiguity:</strong> Flipping all signs of a PC doesn't change anything mathematically. "High PC1" vs "low PC1" interpretation requires context.</li>
      <li><strong>No unique interpretation:</strong> Multiple semantic labels might fit the same PC. Domain expertise is crucial.</li>
      <li><strong>Complex loadings:</strong> When many features contribute moderately, interpretation becomes difficult or impossible.</li>
    </ul>

    <h3>Choosing Number of Components</h3>

    <h4>Explained Variance Threshold</h4>
    <ul>
      <li>Keep components until cumulative variance ≥ threshold (e.g., 0.95)</li>
      <li>Balance between dimensionality reduction and information retention</li>
      <li><strong>Conservative:</strong> 95-99% for critical applications (preserve almost all information)</li>
      <li><strong>Moderate:</strong> 80-90% for most applications (good compression while retaining structure)</li>
      <li><strong>Aggressive:</strong> 50-70% for visualization or when noise is high</li>
    </ul>

    <h4>Scree Plot</h4>
    <ul>
      <li>Plot eigenvalues (or explained variance) vs component number</li>
      <li>Look for "elbow" where curve flattens—indicates diminishing returns</li>
      <li>Keep components before the elbow (steep part of curve)</li>
      <li><strong>Example:</strong> If variance is [40%, 25%, 15%, 8%, 5%, 3%, 2%, 2%, ...], elbow is around PC3-PC4</li>
    </ul>

    <h4>Kaiser Criterion</h4>
    <ul>
      <li>Keep components with eigenvalue > 1 (for standardized data)</li>
      <li>Rationale: Each original feature has variance 1, so PC with eigenvalue >1 captures more info than a single feature</li>
      <li>Often too conservative (keeps too many components) or too aggressive (discards useful components)</li>
      <li>Use as rough heuristic, not definitive rule</li>
    </ul>

    <h4>Cross-Validation</h4>
    <ul>
      <li>Use PCA as preprocessing for supervised learning</li>
      <li>Try k ∈ {5, 10, 20, 50, 100}, train model on k-dimensional data, evaluate via CV</li>
      <li>Choose k that optimizes downstream model performance (accuracy, RMSE, etc.)</li>
      <li>Most rigorous approach when PCA is used for prediction tasks</li>
    </ul>
    
    <h4>Domain-Specific Rules</h4>
    <ul>
      <li><strong>Visualization:</strong> k=2 or k=3 (human perception limit)</li>
      <li><strong>Compression:</strong> k depends on acceptable quality loss (image compression: k for 90-95% variance)</li>
      <li><strong>Noise reduction:</strong> Keep components explaining >1-2% variance, discard the rest as noise</li>
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

    <h3>Disadvantages and Limitations</h3>
    <ul>
      <li><strong>Linear method only:</strong> Cannot capture non-linear relationships. Data on a curved manifold (e.g., Swiss roll, circle) requires many components even though it's low-dimensional.</li>
      <li><strong>Loss of interpretability:</strong> Components are linear combinations of features. "PC1" doesn't have inherent meaning like "age" does. Difficult to explain to non-technical stakeholders.</li>
      <li><strong>Sensitive to feature scaling:</strong> Must standardize first. Without scaling, high-variance features dominate, making PCA essentially perform feature selection by magnitude.</li>
      <li><strong>Assumes variance = importance:</strong> PCA maximizes variance, not predictive power. Low-variance features can still be crucial for classification (rare but discriminative events).</li>
      <li><strong>Outliers distort components:</strong> PCA uses covariance matrix, which is sensitive to outliers. Single extreme points can skew principal directions.</li>
      <li><strong>Computational cost:</strong> O(min(n²d, nd²)) for covariance computation, O(d³) for eigendecomposition. Prohibitive for very high dimensions (d > 10,000) without sparse/randomized methods.</li>
      <li><strong>Information loss:</strong> Discarding components always loses information. May discard dimensions crucial for specific tasks.</li>
    </ul>

    <h3>When PCA Fails and What to Do</h3>
    <table>
      <thead>
        <tr><th>Failure Mode</th><th>Symptom</th><th>Solution</th></tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Non-linear structure</strong></td>
          <td>Many components needed, poor variance capture</td>
          <td>Kernel PCA, t-SNE, UMAP, Autoencoders</td>
        </tr>
        <tr>
          <td><strong>Low variance ≠ low importance</strong></td>
          <td>PCA removes features critical for classification</td>
          <td>Use supervised methods like LDA, or validate via cross-validation</td>
        </tr>
        <tr>
          <td><strong>Unscaled features</strong></td>
          <td>One feature dominates all PCs</td>
          <td>Apply StandardScaler before PCA (always!)</td>
        </tr>
        <tr>
          <td><strong>Outliers present</strong></td>
          <td>PC directions skewed toward outliers</td>
          <td>Remove outliers first, or use Robust PCA variants</td>
        </tr>
        <tr>
          <td><strong>Sparse data</strong></td>
          <td>Many zero values, dense PCs lose sparsity</td>
          <td>Sparse PCA (maintains sparsity in loadings)</td>
        </tr>
        <tr>
          <td><strong>Need interpretability</strong></td>
          <td>Can't explain transformed features</td>
          <td>Use feature selection instead, or Sparse PCA for interpretable loadings</td>
        </tr>
        <tr>
          <td><strong>Very high dimensions</strong></td>
          <td>Computational cost too high</td>
          <td>Incremental PCA (batches), Randomized PCA (approximation)</td>
        </tr>
      </tbody>
    </table>

    <h3>Common Pitfalls</h3>
    <ul>
      <li><strong>Forgetting to scale:</strong> Most common mistake. Always use StandardScaler before PCA.</li>
      <li><strong>Fitting on test data:</strong> Fit PCA on training set only, then transform both train and test with those components (data leakage otherwise).</li>
      <li><strong>Choosing k arbitrarily:</strong> Don't just use k=10 because it's round. Use variance threshold or cross-validation.</li>
      <li><strong>Over-interpreting components:</strong> PCs are mathematical constructs, not always meaningful. Don't force interpretations.</li>
      <li><strong>Using PCA when features are already uncorrelated:</strong> PCA won't help if features are independent—it's designed for correlated data.</li>
      <li><strong>Expecting PCA to improve all models:</strong> Tree-based models don't benefit from PCA (they handle correlated features well). Linear models and distance-based methods benefit most.</li>
      <li><strong>Ignoring computational cost:</strong> For very large datasets, use Incremental PCA or Mini-Batch PCA to avoid memory issues.</li>
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

    <h3>Variants and Extensions</h3>
    <ul>
      <li><strong>Kernel PCA:</strong> Non-linear dimensionality reduction using kernel trick (RBF, polynomial). Maps data to high-dimensional space via kernel, applies PCA there. Use when data lies on non-linear manifolds. Example: separating concentric circles.</li>
      <li><strong>Incremental PCA:</strong> Processes data in mini-batches, suitable for datasets too large to fit in memory (>GB scale). Slight approximation but enables PCA on massive datasets.</li>
      <li><strong>Sparse PCA:</strong> Adds L1 penalty to loadings, forcing many weights to zero. Produces interpretable components (only few features contribute). Trade-off: less variance explained but more interpretable.</li>
      <li><strong>Probabilistic PCA:</strong> Adds Gaussian noise model, enabling likelihood-based model selection and handling missing data naturally. Basis for more complex models like Factor Analysis.</li>
      <li><strong>Robust PCA:</strong> Decomposes data into low-rank + sparse components. Robust to outliers and corruption. Use when data has outliers or missing entries.</li>
      <li><strong>Randomized PCA:</strong> Uses random projections for fast approximation. O(ndk) instead of O(nd²), making it feasible for very high dimensions. Slight loss of accuracy for major speed gain.</li>
    </ul>

    <h3>PCA vs Other Dimensionality Reduction Methods</h3>
    <table>
      <thead>
        <tr><th>Method</th><th>Type</th><th>Best For</th><th>Limitations</th></tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>PCA</strong></td>
          <td>Linear, unsupervised</td>
          <td>Correlated features, preprocessing, speed</td>
          <td>Only linear, ignores labels</td>
        </tr>
        <tr>
          <td><strong>LDA</strong></td>
          <td>Linear, supervised</td>
          <td>Classification preprocessing, maximizing class separation</td>
          <td>Requires labels, max k-1 components for k classes</td>
        </tr>
        <tr>
          <td><strong>t-SNE</strong></td>
          <td>Non-linear, unsupervised</td>
          <td>Visualization (2D/3D), preserving local structure</td>
          <td>Not for modeling (non-deterministic), slow, no inverse transform</td>
        </tr>
        <tr>
          <td><strong>UMAP</strong></td>
          <td>Non-linear, unsupervised</td>
          <td>Visualization, faster than t-SNE, preserves global+local structure</td>
          <td>Not for modeling, sensitive to hyperparameters</td>
        </tr>
        <tr>
          <td><strong>Autoencoders</strong></td>
          <td>Non-linear, unsupervised (neural)</td>
          <td>Complex non-linear patterns, images, large data</td>
          <td>Requires training, black box, needs lots of data</td>
        </tr>
        <tr>
          <td><strong>Feature Selection</strong></td>
          <td>Discrete, supervised/unsupervised</td>
          <td>Interpretability, removing noise, keeping original features</td>
          <td>Discards potentially useful information, doesn't combine features</td>
        </tr>
        <tr>
          <td><strong>Kernel PCA</strong></td>
          <td>Non-linear, unsupervised</td>
          <td>Non-linear manifolds, moderate dimensions</td>
          <td>Expensive (O(n³)), hard to choose kernel, less interpretable</td>
        </tr>
        <tr>
          <td><strong>ICA</strong></td>
          <td>Linear, unsupervised</td>
          <td>Signal separation (cocktail party problem), non-Gaussian sources</td>
          <td>Assumes independence (stronger than PCA), sensitive to initialization</td>
        </tr>
      </tbody>
    </table>
    
    <p><strong>Decision Guide:</strong></p>
    <ul>
      <li><strong>Need interpretability:</strong> Feature selection > Sparse PCA > standard PCA</li>
      <li><strong>Preprocessing for classification:</strong> LDA (supervised) > PCA (unsupervised)</li>
      <li><strong>Visualization only:</strong> t-SNE or UMAP (non-linear, beautiful plots)</li>
      <li><strong>Non-linear relationships:</strong> Kernel PCA or Autoencoders</li>
      <li><strong>Speed matters:</strong> PCA (fastest) > Randomized PCA > others</li>
      <li><strong>Large data:</strong> Incremental PCA or Randomized PCA</li>
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
};
