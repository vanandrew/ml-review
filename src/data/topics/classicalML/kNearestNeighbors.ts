import { Topic } from '../../../types';

export const kNearestNeighbors: Topic = {
  id: 'k-nearest-neighbors',
  title: 'K-Nearest Neighbors (KNN)',
  category: 'classical-ml',
  description: 'Instance-based learning algorithm for classification and regression',
  content: `
    <h2>K-Nearest Neighbors: Memory-Based Learning</h2>
    <p>K-Nearest Neighbors represents a fundamentally different approach to machine learning: instead of abstracting training data into a model (like coefficients or tree structures), KNN memorizes the entire training set and makes predictions by direct comparison with stored examples. This instance-based or memory-based learning is conceptually simple—"you are the average of your k nearest neighbors"—yet remarkably effective for many problems. KNN embodies the principle that similar inputs should produce similar outputs, using proximity in feature space as a proxy for similarity.</p>

    <p>The algorithm is "lazy" or "non-parametric": there's no training phase (just store the data), no learned parameters, and no assumptions about data distribution. All computation is deferred to prediction time, when the algorithm identifies the k most similar training examples and aggregates their labels. This makes KNN trivial to update with new data (just add to the storage) but expensive for predictions (must compare to all training examples). Despite its simplicity, KNN serves as a powerful baseline, excels in domains where the similarity-based reasoning is natural (recommendation systems, image recognition with engineered features), and provides a interpretable form of prediction through example-based reasoning.</p>

    <h3>How KNN Works: The Algorithm</h3>
    <p>For a query point x (new data to classify or predict), KNN follows four steps:</p>
    <ol>
      <li><strong>Choose k:</strong> Select the number of neighbors to consider (hyperparameter, typically 3-20). Smaller k = more flexible, larger k = more robust but smoother.</li>
      <li><strong>Compute distances:</strong> Calculate the distance from x to every training point using a distance metric (Euclidean, Manhattan, etc.). This requires O(n×d) operations where n is training samples and d is features.</li>
      <li><strong>Find k-nearest:</strong> Identify the k training points with smallest distances to x. This can be done via sorting (O(n log n)) or partial sorting (O(n log k)).</li>
      <li><strong>Aggregate predictions:</strong>
        <ul>
          <li><strong>Classification:</strong> Majority vote—predict the class that appears most frequently among the k neighbors. For example, if k=5 and neighbors have labels [A, A, B, A, C], predict A (appears 3 times).</li>
          <li><strong>Regression:</strong> Average—predict the mean (or weighted mean) of neighbor values. For k=5 neighbors with values [10, 12, 11, 15, 13], predict (10+12+11+15+13)/5 = 12.2.</li>
        </ul>
      </li>
    </ol>

    <p>The entire prediction depends on local structure around the query point. If the k-nearest neighbors are mostly class A, the prediction is A; if they're evenly split, the prediction is uncertain (ties are broken arbitrarily or via distance weighting). This locality is both a strength (captures local patterns, handles complex boundaries) and a weakness (sensitive to local noise, requires dense data everywhere).</p>

    <h3>Distance Metrics: Measuring Similarity</h3>
    <p>The choice of distance metric profoundly affects KNN's behavior, defining what "near" means:</p>

    <ul>
      <li><strong>Euclidean Distance (L2 norm): d(x, y) = √(Σ(x_i - y_i)²)</strong>
        <p>The most common metric. Measures straight-line distance in feature space. Geometrically intuitive (shortest path between points) and works well for continuous numerical features where Euclidean geometry applies. Sensitive to scale—features with larger ranges dominate. **Always standardize features before using Euclidean distance.**</p>
      </li>
      <li><strong>Manhattan Distance (L1 norm, City Block): d(x, y) = Σ|x_i - y_i|</strong>
        <p>Sum of absolute differences along each dimension. Useful when movement is restricted to axes (like navigating city blocks). More robust to outliers than Euclidean (no squaring amplifies extremes). Can work better in high dimensions where Euclidean distances become less discriminative. Preferred for discrete or grid-like data.</p>
      </li>
      <li><strong>Minkowski Distance: d(x, y) = (Σ|x_i - y_i|^p)^(1/p)</strong>
        <p>Generalization of both Euclidean (p=2) and Manhattan (p=1). Parameter p controls sensitivity to large differences. p→∞ gives Chebyshev distance (max difference along any dimension). Rarely used in practice except as a way to interpolate between L1 and L2.</p>
      </li>
      <li><strong>Cosine Distance: d(x, y) = 1 - (x·y)/(||x||·||y||)</strong>
        <p>Measures angle between vectors, not magnitude. Two vectors pointing in the same direction have distance 0, regardless of length. Ideal for text data (TF-IDF vectors), where document length doesn't indicate similarity—"AI is great" and "AI is great great great" should be similar. Also used for high-dimensional sparse data (recommendation systems) where magnitude is less meaningful than direction.</p>
      </li>
      <li><strong>Hamming Distance: d(x, y) = number of differing positions</strong>
        <p>For categorical or binary features. Counts how many features differ between two points. For binary strings [1,0,1,1] and [1,1,1,0], Hamming distance = 2 (positions 2 and 4 differ). Used for DNA sequences, error-correcting codes, or purely categorical data.</p>
      </li>
    </ul>

    <p><strong>Choosing the right metric:</strong> Use Euclidean for continuous numerical features (most common), Manhattan for high-dimensional or when robustness to outliers matters, Cosine for text/sparse data where direction matters more than magnitude, and Hamming for categorical data. Scikit-learn's KNN supports many metrics via the metric parameter. Experiment via cross-validation if unsure.</p>

    <h3>Worked Example: Classifying a House with KNN</h3>
    <p><strong>Problem:</strong> Predict whether a house will sell above market price (class = "High") or not (class = "Low") based on two features: Size (square feet) and Distance to City Center (miles).</p>

    <p><strong>Training data (5 houses):</strong></p>
    <ul>
      <li>House A: Size=1500 sqft, Distance=2 mi → Low</li>
      <li>House B: Size=2500 sqft, Distance=1 mi → High</li>
      <li>House C: Size=1800 sqft, Distance=5 mi → Low</li>
      <li>House D: Size=3000 sqft, Distance=3 mi → High</li>
      <li>House E: Size=2200 sqft, Distance=2.5 mi → High</li>
    </ul>

    <p><strong>Query point:</strong> Size=2000 sqft, Distance=2 mi. Predict class.</p>

    <p><strong>Step 1: Feature scaling.</strong> Standardize both features to mean=0, std=1.</p>
    <ul>
      <li>Size: μ=2200, σ≈589. Standardized values: A'=-1.19, B'=0.51, C'=-0.68, D'=1.36, E'=0.00, Query'=-0.34</li>
      <li>Distance: μ=2.7, σ≈1.36. Standardized values: A'=-0.51, B'=-1.25, C'=1.69, D'=0.22, E'=-0.15, Query'=-0.51</li>
    </ul>

    <p><strong>Step 2: Compute Euclidean distances</strong> from query (standardized) to each training point:</p>
    <ul>
      <li>d(Query, A) = √[(−0.34−(−1.19))² + (−0.51−(−0.51))²] = √[0.85² + 0²] = 0.85</li>
      <li>d(Query, B) = √[(−0.34−0.51)² + (−0.51−(−1.25))²] = √[0.85² + 0.74²] = 1.13</li>
      <li>d(Query, C) = √[(−0.34−(−0.68))² + (−0.51−1.69)²] = √[0.34² + 2.20²] = 2.23</li>
      <li>d(Query, D) = √[(−0.34−1.36)² + (−0.51−0.22)²] = √[1.70² + 0.73²] = 1.85</li>
      <li>d(Query, E) = √[(−0.34−0.00)² + (−0.51−(−0.15))²] = √[0.34² + 0.36²] = 0.50</li>
    </ul>

    <p><strong>Step 3: Find k=3 nearest neighbors.</strong> Sorting distances: E (0.50), A (0.85), B (1.13), D (1.85), C (2.23). The 3 nearest are E, A, B.</p>

    <p><strong>Step 4: Majority vote.</strong> Labels of 3 nearest: E→High, A→Low, B→High. Votes: High=2, Low=1. <strong>Prediction: High</strong> (house will sell above market price).</p>

    <p><strong>Interpretation:</strong> The query house (2000 sqft, 2 mi) is most similar to House E (2200 sqft, 2.5 mi, High) and House B (2500 sqft, 1 mi, High), both of which sold above market. Though House A (1500 sqft, 2 mi, Low) is also nearby, the majority vote favors High. If we used k=1 (only nearest neighbor E), prediction would be High. If k=5 (all points), votes are High=3, Low=2, still High—but the margin would narrow.</p>

    <p><strong>Effect of distance weighting:</strong> If we weight by inverse distance (weight = 1/distance), we get: High votes = 1/0.50 + 1/1.13 = 2.00 + 0.88 = 2.88, Low votes = 1/0.85 = 1.18. Weighted prediction: High (with stronger confidence since E is much closer). This shows how distance weighting amplifies the influence of very close neighbors.</p>

    <h3>Choosing k: The Bias-Variance Tradeoff</h3>
    <p>The number of neighbors k is KNN's primary hyperparameter, controlling model complexity:</p>

    <p><strong>Small k (k=1, k=3, k=5):</strong> Low bias, high variance. The model is very flexible—decision boundaries can be arbitrarily complex, wrapping around individual points. k=1 (nearest neighbor) achieves 100% training accuracy (each training point predicts itself) but is maximally sensitive to noise: a single mislabeled or outlier point creates an island of incorrect predictions. Small k captures fine-grained local structure but overfits to noise and outliers. Decision boundaries are jagged, with many small regions.</p>

    <p><strong>Large k (k=50, k=100):</strong> High bias, low variance. The model is smooth—predictions average over many points, creating broad decision boundaries. Large k is robust to noise (individual noisy points are outvoted) but risks underfitting: it may ignore legitimate local patterns and treat everything as the global majority class. In the extreme k=n (all training points), every prediction is the mode (classification) or mean (regression) of the entire training set, ignoring the query point entirely.</p>

    <p><strong>Selecting optimal k:</strong> Use cross-validation—try k ∈ {1, 3, 5, 7, 9, 15, 21, 31, 51, 101}, evaluate performance via k-fold CV, plot validation accuracy vs k (learning curve), and choose k with best validation performance. Look for the "sweet spot" where validation accuracy peaks. Typical optimal k: 3-20 for small/medium datasets (100-10,000 samples), larger for big datasets (10,000+ samples). **Practical tips:** (1) Use odd k for binary classification to avoid ties in voting. (2) Start with k = √n as a rule of thumb. (3) For imbalanced data, larger k may help (more points to vote) but can drown out minority class—consider distance weighting.</p>

    <h3>Weighted KNN: Giving Closer Neighbors More Say</h3>
    <p>Standard KNN treats all k neighbors equally—each gets one vote (classification) or equal contribution (regression). But intuitively, a neighbor at distance 0.1 should influence the prediction more than one at distance 5.0. <strong>Distance-weighted KNN</strong> addresses this by weighting neighbors inversely by distance:</p>

    <p><strong>Uniform weighting (standard):</strong> weight_i = 1 for all k neighbors. Prediction = majority vote (classification) or mean (regression). Simple but ignores distance information within the k-neighborhood.</p>

    <p><strong>Distance weighting:</strong> weight_i = 1/distance_i (or 1/distance_i² for stronger emphasis on close points). For classification, compute weighted votes: score(class_c) = Σ{i: y_i = c} weight_i, predict argmax_c score(c). For regression: ŷ = Σ(weight_i × y_i) / Σweight_i. Neighbors very close to the query dominate, while distant neighbors contribute minimally.</p>

    <p><strong>Advantages of weighting:</strong> (1) Better handles varying neighbor distances—if k=10 but 3 neighbors are very close and 7 are far, the close ones dominate (appropriate). (2) Smoother predictions—gradual transitions between regions. (3) Less sensitive to k—using k=20 vs k=10 matters less because distant neighbors have little weight. (4) Avoids ties—even with even k in binary classification, weighted votes rarely tie exactly.</p>

    <p><strong>When to use which:</strong> Use distance weighting when data has non-uniform density (clusters with varying tightness), when you want smoother predictions, or when using larger k to be safe but still want nearby points to dominate. Use uniform weighting for simplicity, when computational efficiency matters (slightly faster—no weight computation), or when distances within the k-neighborhood are similar anyway. Scikit-learn's KNeighborsClassifier supports weights='uniform' (default) or weights='distance'. Empirically, distance weighting often improves performance 2-5%.</p>

    <h3>The Curse of Dimensionality: KNN's Achilles Heel</h3>
    <p>KNN's performance degrades catastrophically in high-dimensional spaces due to the curse of dimensionality, a fundamental property of high-dimensional geometry:</p>

    <p><strong>Sparsity:</strong> The volume of a unit hypercube grows exponentially with dimensions: V = side_length^d. Fixing the number of data points n, as d increases, density = n/V decreases exponentially. With 1000 points uniformly distributed in [0,1]^2, average neighbor distance ≈ 0.03; in [0,1]^10, it's ≈ 0.45; in [0,1]^100, nearly 1.0 (edge of the space). You'd need 10^100 points to maintain 2D density in 100D, which is impossibly large. Practically, we never have enough data to densely populate high-dimensional spaces, leaving KNN's neighborhoods empty or unrepresentative.</p>

    <p><strong>Distance concentration:</strong> In high dimensions, distances between all pairs of points become approximately equal. The ratio of farthest to nearest neighbor approaches 1 as d→∞: max_dist/min_dist → 1. If all training points are roughly equidistant from the query, the notion of "nearest" neighbors is meaningless—why trust the "closest" k points when they're barely closer than distant points? Euclidean distance loses its discriminative power because the cumulative effect of small differences across many dimensions dominates, making all points far apart.</p>

    <p><strong>Practical impacts:</strong> (1) Predictions become unreliable—"nearest" neighbors aren't truly similar. (2) k must be very large to include meaningful neighbors, but this over-smooths predictions. (3) Computation slows (more features to compute distances). (4) Irrelevant features corrupt distances: if 3 of 100 features are relevant, the 97 noise dimensions drown out the 3 signal dimensions, making distances uninformative.</p>

    <p><strong>Mitigation strategies:</strong> (1) **Dimensionality reduction:** Apply PCA, t-SNE, UMAP, or autoencoders to project data to lower dimensions (5-50D) preserving structure. (2) **Feature selection:** Remove irrelevant features via univariate tests, recursive elimination, or L1 regularization. (3) **Distance metric learning:** Learn a Mahalanobis distance or neural embedding that emphasizes discriminative dimensions. (4) **Collect more data:** Exponentially more samples are needed (infeasible for very high d). (5) **Switch algorithms:** Tree-based methods (Random Forest, XGBoost), linear models, or neural networks are less affected by dimensionality.</p>

    <p>As a rule of thumb, KNN becomes unreliable beyond ~20-30 dimensions without careful feature engineering or dimensionality reduction. This is why KNN works well for image recognition with engineered features (10-100 dimensions, e.g., SIFT, HOG) but fails on raw pixel data (10,000+ dimensions) without reduction.</p>

    <h3>Computational Considerations: Speed and Scalability</h3>
    <p><strong>Training:</strong> O(1)—just store the data. Trivially fast, making KNN excellent for online learning (add new data instantly) or scenarios with frequently changing training sets.</p>

    <p><strong>Prediction:</strong> O(n×d) per query for naive implementation—compute distance to all n training points (d operations each), then find k smallest (O(n log k)). For 100,000 training points with 100 features, this is 10 million operations per query. For real-time systems needing sub-millisecond latency, this is prohibitively slow. By contrast, a trained neural network or decision tree requires only O(depth) or O(layers) operations, often orders of magnitude faster.</p>

    <p><strong>Optimizations:</strong> Specialized data structures accelerate neighbor search at the cost of preprocessing:</p>
    <ul>
      <li><strong>KD-Trees:</strong> Space-partitioning tree that recursively splits data along alternating dimensions. Reduces search to O(log n) in low dimensions (d ≤ 10). Builds in O(n log n), stores in O(n). Degrades to O(n) in high dimensions due to curse of dimensionality—splits become ineffective when all points are equidistant. Scikit-learn uses KD-Tree by default for d ≤ 10.</li>
      <li><strong>Ball Trees:</strong> Tree structure using hyperspheres instead of axis-aligned splits. More robust to high dimensions (d ≤ 30) than KD-Trees. Builds in O(n log n), queries in O(log n) to O(n) depending on d. Used by scikit-learn for 10 < d ≤ 30.</li>
      <li><strong>Locality-Sensitive Hashing (LSH):</strong> Probabilistic method that hashes similar points to the same buckets. Approximate k-NN (may miss true neighbors but fast). O(1) average query time with appropriate hash functions. Scales to millions of points and high dimensions (100+). Used in production for large-scale similarity search (recommendation systems, image retrieval).</li>
      <li><strong>Approximate Nearest Neighbors (ANN) libraries:</strong> FAISS (Facebook), Annoy (Spotify), HNSW (Hierarchical Navigable Small World graphs)—all provide fast approximate k-NN with tunable accuracy-speed tradeoffs. Essential for large-scale applications (>1M points).</li>
    </ul>

    <p><strong>When KNN is too slow:</strong> For large datasets (>100k samples) or real-time requirements (<10ms latency), consider: (1) Use ANN libraries for approximate but fast search. (2) Switch to eager learners (train once, predict fast): Logistic Regression, Random Forest, Neural Networks. (3) Use KNN for initial prototyping or as a baseline, then migrate to faster models for production.</p>

    <h3>Feature Scaling: Absolutely Critical for KNN</h3>
    <p>Feature scaling is non-negotiable for KNN because the algorithm uses distance metrics, and distances are scale-dependent. Without scaling, features with larger ranges dominate distance calculations, effectively ignoring smaller-scale features.</p>

    <p><strong>Example:</strong> Predicting house prices using [square feet, number of bedrooms]. Square feet ranges from 500 to 5000 (range = 4500), bedrooms range from 1 to 5 (range = 4). Computing Euclidean distance: d = √((sqft₁ - sqft₂)² + (beds₁ - beds₂)²). A difference of 1000 sqft contributes 1,000,000 to the squared distance, while a difference of 4 bedrooms contributes only 16. Square feet dominates overwhelmingly—bedrooms are essentially ignored, even if they're equally important for predicting price.</p>

    <p><strong>Standardization (z-score normalization):</strong> Transform each feature to mean=0, std=1 via x' = (x - μ)/σ. After standardization, both features contribute proportionally to their "relative variance" (spread relative to their own scale). This is the standard preprocessing for KNN. Use sklearn.preprocessing.StandardScaler: fit on training data, transform both training and test data.</p>

    <p><strong>Min-max scaling:</strong> Transform to a fixed range [0, 1] via x' = (x - min)/(max - min). Also effective but more sensitive to outliers (which affect min and max). Less common for KNN than standardization.</p>

    <p><strong>Impact:</strong> Without scaling, KNN may achieve 60-70% accuracy on mixed-scale data; with scaling, 80-85% on the same data. The difference can be dramatic. Feature scaling is also critical for SVM, K-Means, PCA—any algorithm using distances or dot products. Not needed for tree-based methods (Random Forest, XGBoost), which split on thresholds invariant to scale.</p>

    <p><strong>Always remember:** Fit scaler on training data only, then transform both training and test with those parameters. Never fit on test data (data leakage). For KNN, standardization should be the first step in your pipeline, always.</p>

    <h3>Advantages of KNN</h3>
    <ul>
      <li><strong>Simplicity:</strong> Conceptually straightforward, easy to implement and explain. No complex math or optimization.</li>
      <li><strong>No training phase:</strong> Instant "training" (just store data), making it ideal for online learning or frequently updated datasets.</li>
      <li><strong>Non-parametric:</strong> Makes no assumptions about data distribution (Gaussian, linear, etc.), allowing it to model any distribution or relationship.</li>
      <li><strong>Naturally multi-class:</strong> Handles any number of classes without modification (no one-vs-rest schemes needed).</li>
      <li><strong>Flexible decision boundaries:</strong> Can capture arbitrarily complex, non-linear boundaries (with appropriate k).</li>
      <li><strong>Interpretable predictions:</strong> Can explain predictions by showing the k nearest neighbors—example-based reasoning that non-technical users understand.</li>
      <li><strong>Effective for small-to-medium datasets:</strong> With 100-10,000 samples and low-to-medium dimensions (≤30 features), KNN is competitive.</li>
    </ul>

    <h3>Disadvantages and Limitations</h3>
    <ul>
      <li><strong>Slow prediction:</strong> O(n×d) makes it impractical for large datasets or real-time applications without ANN libraries.</li>
      <li><strong>Memory intensive:</strong> Stores entire training dataset. For 1M samples with 100 features (float32), that's ~400MB. Compared to a neural network storing just weights (often <10MB), this is substantial.</li>
      <li><strong>Curse of dimensionality:</strong> Fails in high dimensions (>30) where distances become uninformative. Requires dimensionality reduction or feature selection.</li>
      <li><strong>Sensitive to irrelevant features:</strong> Noise dimensions corrupt distance calculations. Requires careful feature engineering.</li>
      <li><strong>Requires feature scaling:</strong> Essential preprocessing step, often forgotten by beginners.</li>
      <li><strong>Sensitive to imbalanced data:</strong> Majority class dominates voting. Use stratified sampling, class weighting, or SMOTE for imbalance.</li>
      <li><strong>Doesn't learn anything:</strong> No model to interpret, no coefficients showing feature importance, no compression of patterns. Just stores raw data.</li>
      <li><strong>Categorical features problematic:</strong> Distance metrics for categorical data (Hamming) are less effective than for continuous features. One-hot encoding inflates dimensionality.</li>
    </ul>

    <h3>When to Use KNN vs Alternatives</h3>
    <p><strong>Use KNN when:</strong> Small-to-medium datasets (100-10,000 samples), low-to-medium dimensions (≤30 features after reduction), irregular decision boundaries, need for interpretable example-based predictions, online learning (frequent data updates), as a baseline to establish minimum performance before trying complex models.</p>

    <p><strong>Prefer alternatives when:</strong> Large datasets (>100k samples)—use Logistic Regression, Random Forest, XGBoost, or Neural Networks (faster training and prediction). High dimensions (>30 features)—use Linear models, tree-based methods, or reduce dimensionality first. Real-time predictions needed (<10ms latency)—use eager learners (trained models predict quickly). Categorical features—use tree-based methods (handle categoricals natively). Need feature importance or model interpretation—use Linear models (coefficients), tree-based methods (feature importance, SHAP values).</p>

    <h3>Visual Understanding</h3>
    <p>Imagine a 2D scatter plot with labeled points (different colors for different classes). When a new unlabeled point appears, draw circles expanding from it until you capture k nearest points. These k neighbors "vote" on the new point's label. The visualization shows clustering patterns—regions where one class dominates will vote for that class. The decision boundary is where neighborhoods split evenly between classes.</p>
    
    <p><strong>Key visualizations to understand:</strong></p>
    <ul>
      <li><strong>KNN scatter plot:</strong> Training points as colored dots, query point as a star or larger marker. Draw lines from query to its k nearest neighbors, highlighting those k points. The majority color among them determines the prediction.</li>
      <li><strong>Decision boundary (Voronoi diagram):</strong> For k=1, the space is divided into regions where each training point is closest—creating polygonal cells around each point. The color of each cell shows the prediction for any query landing there. For k>1, boundaries become smoother.</li>
      <li><strong>Distance circles:</strong> Concentric circles around the query point at increasing radii, showing how neighbors are selected. The k-th circle's radius is the distance to the k-th nearest neighbor.</li>
      <li><strong>Effect of k visualization:</strong> Side-by-side plots showing decision boundaries for k=1 (jagged, complex), k=5 (smoother), k=50 (very smooth, possibly underfit). Demonstrates bias-variance tradeoff visually.</li>
    </ul>

    <h3>Common Mistakes to Avoid</h3>
    <ul>
      <li><strong>❌ Forgetting feature scaling:</strong> The #1 mistake with KNN. Features with large ranges dominate distance calculations. ALWAYS standardize features before KNN. This is not optional.</li>
      <li><strong>❌ Using KNN in high dimensions (>30):</strong> Curse of dimensionality makes all points equidistant. Use dimensionality reduction (PCA, feature selection) or switch algorithms.</li>
      <li><strong>❌ Not tuning k:</strong> Default k=5 may be terrible for your data. Always tune k via cross-validation—try k ∈ {1, 3, 5, 7, 9, 15, 21, 31}.</li>
      <li><strong>❌ Using even k for binary classification:</strong> Leads to ties in voting. Use odd k (3, 5, 7) or implement tie-breaking rules.</li>
      <li><strong>❌ Expecting fast predictions:</strong> KNN is slow for large datasets (must compare to all training points). For real-time systems with >10k training samples, use approximate nearest neighbors (Annoy, FAISS) or different algorithms.</li>
      <li><strong>❌ Including irrelevant features:</strong> Noise dimensions corrupt distance measurements. Perform feature selection to remove low-information features.</li>
      <li><strong>❌ Not handling imbalanced data:</strong> Majority class dominates voting. Use distance weighting, stratified sampling, or adjust k to help minority class representation.</li>
      <li><strong>❌ Using default Euclidean for all data types:</strong> For text, use cosine distance. For binary features, use Hamming distance. Match the distance metric to your data type.</li>
    </ul>

    <h3>Practical Recommendations</h3>
    <ul>
      <li><strong>Always standardize features:</strong> Use StandardScaler before KNN. This is critical.</li>
      <li><strong>Tune k via cross-validation:</strong> Try k ∈ {1, 3, 5, 7, 9, 15, 21, 31}, evaluate with 5-fold CV, plot validation accuracy vs k, select best k.</li>
      <li><strong>Use distance weighting:</strong> Set weights='distance' in scikit-learn for better performance with minimal cost.</li>
      <li><strong>Handle high dimensions:</strong> Apply PCA, feature selection, or domain-specific dimensionality reduction before KNN if d > 30.</li>
      <li><strong>For large datasets:</strong> Use ANN libraries (Annoy, FAISS) for approximate but fast k-NN, or switch to faster algorithms.</li>
      <li><strong>Check for imbalanced classes:</strong> Use stratified cross-validation, distance weighting, or class-balanced sampling.</li>
      <li><strong>Visualize decision boundaries:</strong> For 2D/3D data, plot decision regions to ensure they make sense and aren't overfitting.</li>
      <li><strong>Compare distance metrics:</strong> If Euclidean underperforms, try Manhattan, Cosine, or learned metrics.</li>
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
};
