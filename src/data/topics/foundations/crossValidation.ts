import { Topic } from '../../../types';

export const crossValidation: Topic = {
  id: 'cross-validation',
  title: 'Cross-Validation',
  category: 'foundations',
  description: 'Robust techniques for evaluating model performance and preventing overfitting',
  content: `
    <h2>Cross-Validation: Robust Model Evaluation</h2>
    <p>Cross-validation is a statistical resampling technique that provides more reliable estimates of model performance than a single train-test split. By systematically using different portions of data for training and validation across multiple iterations, cross-validation reduces the variance in performance estimates and makes more efficient use of limited data. It's an essential tool for model selection, hyperparameter tuning, and honest performance reporting.</p>

    <div class="info-box info-box-purple">
      <h4>🎯 Which CV Method Should I Use?</h4>
      <table>
        <tr>
          <th>Your Situation</th>
          <th>Recommended Method</th>
          <th>Why</th>
        </tr>
        <tr>
          <td>Balanced classification</td>
          <td><strong>Standard k-fold</strong> (k=5 or 10)</td>
          <td>Simple, efficient, standard choice</td>
        </tr>
        <tr>
          <td>Imbalanced classes</td>
          <td><strong>Stratified k-fold</strong></td>
          <td>Maintains class distribution</td>
        </tr>
        <tr>
          <td>Time series data</td>
          <td><strong>TimeSeriesSplit</strong></td>
          <td>Respects temporal order</td>
        </tr>
        <tr>
          <td>Very small dataset (<100)</td>
          <td><strong>LOOCV</strong> or k=10</td>
          <td>Maximizes training data</td>
        </tr>
        <tr>
          <td>Large dataset (>100K)</td>
          <td><strong>k-fold</strong> (k=3 or 5)</td>
          <td>Faster, diminishing returns</td>
        </tr>
        <tr>
          <td>Grouped/hierarchical data</td>
          <td><strong>GroupKFold</strong></td>
          <td>Keeps groups together</td>
        </tr>
        <tr>
          <td>Hyperparameter tuning</td>
          <td><strong>Nested CV</strong></td>
          <td>Unbiased performance estimate</td>
        </tr>
      </table>
      <p><strong>⚠️ Never:</strong> Use standard k-fold for time series | Forget to stratify for imbalanced data | Tune on test set</p>
    </div>

    <h3>The Problem with Single Train-Test Splits</h3>
    <p>When you split your data once into training and test sets, your performance estimate depends heavily on which specific samples happened to land in each set. You might get "lucky" with an easy test set that inflates your performance, or "unlucky" with a hard test set that underestimates your model. This variance makes it difficult to distinguish genuine model improvements from random luck. Additionally, in small datasets, setting aside 20-30% for testing wastes precious training examples.</p>
    
    <p>Cross-validation solves both problems by evaluating the model multiple times on different data splits, providing both an average performance estimate (more reliable) and a measure of uncertainty (standard deviation across folds). Every data point contributes to both training and testing, maximizing data utilization.</p>

    <h3>K-Fold Cross-Validation: The Standard Approach</h3>
    
    <p><strong>How It Works:</strong></p>
    <ol>
      <li><strong>Split:</strong> Divide your dataset into k equal-sized "folds" (typically k=5 or k=10)</li>
      <li><strong>Iterate:</strong> For each of the k folds:
        <ul>
          <li>Use that fold as the validation set</li>
          <li>Use the remaining k-1 folds as the training set</li>
          <li>Train the model on the training set</li>
          <li>Evaluate on the validation fold and record the performance</li>
        </ul>
      </li>
      <li><strong>Aggregate:</strong> Average the k performance scores for the final estimate</li>
      <li><strong>Report:</strong> Report both mean and standard deviation of scores</li>
    </ol>
    
    <p><strong>Example with 5-Fold CV:</strong></p>
    <p>With 1000 samples and k=5, each fold contains 200 samples:</p>
    <ul>
      <li><strong>Fold 1:</strong> Train on samples 201-1000 (800 samples), validate on samples 1-200</li>
      <li><strong>Fold 2:</strong> Train on samples 1-200 + 401-1000 (800 samples), validate on samples 201-400</li>
      <li><strong>Fold 3:</strong> Train on samples 1-400 + 601-1000 (800 samples), validate on samples 401-600</li>
      <li><strong>Fold 4:</strong> Train on samples 1-600 + 801-1000 (800 samples), validate on samples 601-800</li>
      <li><strong>Fold 5:</strong> Train on samples 1-800 (800 samples), validate on samples 801-1000</li>
    </ul>
    
    <p>Each sample appears in exactly one validation set and four training sets. You get 5 performance estimates, then report: Mean = 0.85 ± 0.03 (std dev), indicating both expected performance and stability.</p>
    
    <p><strong>Choosing K:</strong></p>
    <ul>
      <li><strong>k=5:</strong> Good balance of computational cost and reliability; standard choice</li>
      <li><strong>k=10:</strong> More reliable estimates, slightly more computation; common in research</li>
      <li><strong>Larger k (>10):</strong> Diminishing returns; much higher computational cost</li>
      <li><strong>Smaller k (2-3):</strong> Less reliable, faster; use when computation is prohibitive</li>
    </ul>
    
    <p>The choice involves a bias-variance tradeoff: larger k means each training set is closer in size to the full dataset (lower bias) but the k training sets overlap more (higher variance in estimates). k=5 or k=10 typically provides the best balance.</p>

    <h3>Stratified K-Fold: Essential for Classification</h3>
    
    <p>Standard k-fold randomly assigns samples to folds, which can create problems for classification with imbalanced classes. With 95% negative and 5% positive classes, random folding might create folds with 2% positive in one fold and 8% in another, or even zero positive samples in some folds.</p>
    
    <p><strong>Stratified Sampling Solution:</strong></p>
    <p>Stratified k-fold ensures each fold maintains approximately the same class distribution as the overall dataset. It splits each class separately, then combines them:</p>
    <ol>
      <li>Separate samples by class</li>
      <li>Divide each class into k folds</li>
      <li>Combine corresponding folds from each class</li>
      <li>Result: each fold has ~same proportion of each class as the full dataset</li>
    </ol>
    
    <p><strong>When Stratification is Critical:</strong></p>
    <ul>
      <li><strong>Imbalanced datasets:</strong> Essential when minority class is <10%, very helpful even at 20-80%</li>
      <li><strong>Small datasets:</strong> Random variation can significantly skew class distributions</li>
      <li><strong>Multi-class problems:</strong> Ensures all classes appear in each fold, especially rare classes</li>
      <li><strong>Metric computation:</strong> Prevents folds with zero samples of a class (which breaks recall, precision)</li>
    </ul>
    
    <p><strong>Example:</strong> With 1000 samples (900 class A, 100 class B) and k=5:</p>
    <ul>
      <li><strong>Unstratified:</strong> Folds might have 50-150 class B samples (5-15%) by chance</li>
      <li><strong>Stratified:</strong> Each fold has exactly 180 class A and 20 class B samples (10%)</li>
    </ul>
    
    <p>In sklearn, simply use <code>StratifiedKFold</code> instead of <code>KFold</code> for classification tasks.</p>

    <h3>Time Series Cross-Validation: Respecting Temporal Order</h3>
    
    <p>Time series data has inherent temporal dependencies—today depends on yesterday, this month on last month. Standard k-fold CV randomly shuffles data, destroying temporal structure and creating leakage where future information influences training. This produces misleadingly optimistic results that collapse in production.</p>
    
    <p><strong>Forward Chaining (Expanding Window):</strong></p>
    <p>Time series CV always trains on past data and validates on future data, never the reverse:</p>
    <ul>
      <li><strong>Fold 1:</strong> Train on months 1-6, validate on month 7</li>
      <li><strong>Fold 2:</strong> Train on months 1-7, validate on month 8</li>
      <li><strong>Fold 3:</strong> Train on months 1-8, validate on month 9</li>
      <li><strong>Fold 4:</strong> Train on months 1-9, validate on month 10</li>
    </ul>
    
    <p>Each fold uses an expanding training window (progressively more historical data) and validates on the immediate next time period. This mimics production deployment where you continuously retrain on all available history.</p>
    
    <p><strong>Rolling Window:</strong></p>
    <p>Alternative approach using a fixed-size training window:</p>
    <ul>
      <li><strong>Fold 1:</strong> Train on months 1-6, validate on month 7</li>
      <li><strong>Fold 2:</strong> Train on months 2-7, validate on month 8</li>
      <li><strong>Fold 3:</strong> Train on months 3-8, validate on month 9</li>
    </ul>
    
    <p>Rolling windows are useful when older data becomes less relevant (concept drift) or when training on all history is computationally prohibitive.</p>
    
    <p><strong>Critical Rules:</strong></p>
    <ul>
      <li><strong>Never shuffle:</strong> Maintain chronological order strictly</li>
      <li><strong>Train on past, validate on future:</strong> Simulates real prediction scenario</li>
      <li><strong>Gap period:</strong> Sometimes include a gap between training and validation (e.g., if you need 1 day to deploy, validate on day t+2 after training on through day t)</li>
      <li><strong>Report per-fold:</strong> Performance on different time periods reveals temporal stability</li>
    </ul>
    
    <p><strong>When to Use:</strong></p>
    <ul>
      <li>Financial time series (stock prices, trading)</li>
      <li>Weather forecasting</li>
      <li>Sales/demand forecasting</li>
      <li>Any sequential data where temporal causality matters</li>
    </ul>
    
    <p>In sklearn, use <code>TimeSeriesSplit</code> which implements expanding window by default.</p>

    <h3>Leave-One-Out Cross-Validation (LOOCV)</h3>
    
    <p>LOOCV is k-fold CV where k equals the number of samples (n). Each fold holds out exactly one sample for validation while training on all n-1 remaining samples. For n=100 samples, you train 100 models.</p>
    
    <p><strong>Advantages:</strong></p>
    <ul>
      <li><strong>Maximum training data:</strong> Each model trains on n-1 samples, nearly the full dataset</li>
      <li><strong>Deterministic:</strong> No randomness in fold assignment</li>
      <li><strong>Useful for tiny datasets:</strong> When you literally can't afford to hold out 20%</li>
    </ul>
    
    <p><strong>Disadvantages:</strong></p>
    <ul>
      <li><strong>Computationally prohibitive:</strong> Training n models is infeasible for large datasets or slow algorithms</li>
      <li><strong>High variance estimates:</strong> The n training sets are highly correlated (differ by only one sample), so averaging them doesn't reduce variance as much as averaging more independent estimates</li>
      <li><strong>Unstable for high-variance models:</strong> Models like decision trees or k-NN can vary wildly based on single sample changes</li>
    </ul>
    
    <p><strong>When to Use LOOCV:</strong></p>
    <ul>
      <li>Very small datasets (n < 100) where every sample is precious</li>
      <li>Fast training algorithms (linear models, k-NN)</li>
      <li>Stable low-variance models</li>
    </ul>
    
    <p><strong>When to Avoid:</strong></p>
    <ul>
      <li>Large datasets (n > 1000): use 5 or 10-fold instead</li>
      <li>Expensive models (deep neural networks): computationally infeasible</li>
      <li>High-variance algorithms: LOOCV estimates will be noisy</li>
    </ul>
    
    <p>For most modern applications, 5 or 10-fold CV provides better practical tradeoffs than LOOCV.</p>

    <h3>Nested Cross-Validation: Unbiased Hyperparameter Tuning</h3>
    
    <p>When you use cross-validation for both hyperparameter tuning and performance estimation on the same folds, you get biased (overly optimistic) performance estimates. Nested CV solves this with two levels of cross-validation: an outer loop for unbiased performance estimation and an inner loop for hyperparameter selection.</p>
    
    <p><strong>Structure:</strong></p>
    <ol>
      <li><strong>Outer loop (k_outer folds, typically 5):</strong>
        <ul>
          <li>For each outer fold i:</li>
          <li>Set aside outer fold i as final test set</li>
          <li>Use remaining outer folds as development data</li>
        </ul>
      </li>
      <li><strong>Inner loop (k_inner folds, typically 3-5):</strong>
        <ul>
          <li>Run k_inner-fold CV on the development data</li>
          <li>Try different hyperparameters</li>
          <li>Select hyperparameters with best inner CV performance</li>
        </ul>
      </li>
      <li><strong>Final evaluation:</strong>
        <ul>
          <li>Train model with selected hyperparameters on all development data</li>
          <li>Evaluate on outer test fold i</li>
          <li>Record performance</li>
        </ul>
      </li>
      <li><strong>Aggregate:</strong> Average performance across k_outer folds</li>
    </ol>
    
    <p><strong>Why Nested CV is Necessary:</strong></p>
    <p>If you try 100 hyperparameter combinations using 5-fold CV and select the best one, that best CV score is optimistically biased—you've searched over the validation folds to find what works best for them specifically. Using the same CV score for performance reporting is like peeking at the test set. Nested CV keeps outer test folds completely separate from hyperparameter selection, providing unbiased performance estimates.</p>
    
    <p><strong>Computational Cost:</strong></p>
    <p>Nested CV trains k_outer × k_inner × n_hyperparameters models. With 5 outer folds, 3 inner folds, and 20 hyperparameter combinations: 5 × 3 × 20 = 300 models. This is expensive but necessary for honest reporting.</p>
    
    <p><strong>When to Use:</strong></p>
    <ul>
      <li><strong>Research/publication:</strong> Standard for reporting unbiased performance</li>
      <li><strong>High-stakes applications:</strong> Medical, financial, safety-critical systems</li>
      <li><strong>Model comparison:</strong> Fair comparison between fundamentally different approaches</li>
      <li><strong>Confidence intervals:</strong> When you need reliable uncertainty estimates</li>
    </ul>
    
    <p><strong>Practical Alternative:</strong></p>
    <p>For development, use standard CV for hyperparameter tuning, then evaluate on a separate held-out test set that was never used during development. This is faster than nested CV and provides a reasonable compromise.</p>

    <h3>Cross-Validation Best Practices</h3>
    
    <p><strong>Choosing the Right CV Strategy:</strong></p>
    <ul>
      <li><strong>Classification (balanced):</strong> Standard k-fold (k=5 or 10)</li>
      <li><strong>Classification (imbalanced):</strong> Stratified k-fold (k=5 or 10)</li>
      <li><strong>Regression:</strong> Standard k-fold (k=5 or 10)</li>
      <li><strong>Time series:</strong> TimeSeriesSplit (forward chaining)</li>
      <li><strong>Small data (n<100):</strong> LOOCV or 10-fold</li>
      <li><strong>Large data (n>10,000):</strong> 5-fold or even 3-fold (diminishing returns from more folds)</li>
      <li><strong>Grouped data:</strong> GroupKFold (samples from same group stay together)</li>
    </ul>
    
    <p><strong>Reporting Results:</strong></p>
    <ul>
      <li>Report mean performance across folds</li>
      <li>Report standard deviation (indicates stability/variance)</li>
      <li>Report performance on each fold individually for analysis</li>
      <li>For research: report confidence intervals</li>
      <li>Example: "Accuracy: 0.85 ± 0.03 (mean ± std across 5 folds)"</li>
    </ul>
    
    <p><strong>Common Pitfalls:</strong></p>
    <ul>
      <li><strong>Data leakage:</strong> Preprocessing (scaling, feature selection) must happen inside CV loop, not before</li>
      <li><strong>Using test set multiple times:</strong> Defeats the purpose of CV; test set should be used once at the end</li>
      <li><strong>Shuffling time series:</strong> Always use TimeSeriesSplit for temporal data</li>
      <li><strong>Not stratifying:</strong> Use stratified CV for classification, especially with imbalance</li>
      <li><strong>Forgetting to set random seed:</strong> Makes results non-reproducible</li>
      <li><strong>Optimistic reporting:</strong> Don't report the best fold's performance; report the average</li>
    </ul>
    
    <p><strong>Practical Workflow:</strong></p>
    <ol>
      <li>Split off final test set (20%), never touch it</li>
      <li>Use CV on remaining 80% for model development:
        <ul>
          <li>Model selection (which algorithm?)</li>
          <li>Feature selection (which features?)</li>
          <li>Hyperparameter tuning (which settings?)</li>
        </ul>
      </li>
      <li>After all decisions finalized, train on full 80%</li>
      <li>Evaluate once on held-out 20% test set</li>
      <li>Report test performance as final unbiased estimate</li>
    </ol>
    
    <p><strong>When NOT to Use Cross-Validation:</strong></p>
    <ul>
      <li>When you have abundant data and computational resources for large held-out validation sets</li>
      <li>During initial rapid prototyping (use simple train-val split for speed)</li>
      <li>When data has strict temporal or privacy constraints preventing random sampling</li>
      <li>For final production model training (use all available data after validation)</li>
    </ul>

    <h3>Cross-Validation vs Holdout Validation</h3>
    
    <p><strong>Holdout Validation (Single Split):</strong></p>
    <ul>
      <li><strong>Pros:</strong> Fast, simple, works well with large datasets</li>
      <li><strong>Cons:</strong> High variance in estimates, wastes data, single evaluation</li>
    </ul>
    
    <p><strong>Cross-Validation:</strong></p>
    <ul>
      <li><strong>Pros:</strong> Reliable estimates, uses all data, quantifies uncertainty, detects instability</li>
      <li><strong>Cons:</strong> Computationally expensive (k× cost), more complex implementation</li>
    </ul>
    
    <p><strong>Decision Guide:</strong></p>
    <ul>
      <li><strong>Use holdout when:</strong> Data is abundant (>100k samples), fast iteration is critical, computational budget is limited</li>
      <li><strong>Use CV when:</strong> Data is limited, need reliable estimates, model selection/tuning, research/publication, high-stakes applications</li>
    </ul>

    <h3>Summary</h3>
    <p>Cross-validation is a cornerstone technique in machine learning, providing reliable performance estimates through systematic resampling. Standard k-fold CV (k=5 or 10) works for most problems. Use stratified k-fold for classification to maintain class balance. Use TimeSeriesSplit for temporal data to respect causality. LOOCV maximizes training data for tiny datasets but is computationally expensive. Nested CV separates hyperparameter tuning from performance estimation for unbiased reporting. The key is choosing the right CV strategy for your data structure and reporting both mean and standard deviation to quantify performance and uncertainty. Cross-validation is more than just an evaluation technique—it's a discipline that ensures your model selection process is rigorous and your performance claims are honest.</p>
  `,
  codeExamples: [
    {
      language: 'Python',
      code: `from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np

# Create sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2,
                         weights=[0.7, 0.3], random_state=42)

# Initialize model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Standard k-fold cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"5-Fold CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
print(f"Individual fold scores: {scores}")

# Stratified k-fold (maintains class distribution)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratified_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
print(f"\\nStratified 5-Fold CV: {stratified_scores.mean():.3f} (+/- {stratified_scores.std():.3f})")

# Multiple scoring metrics
from sklearn.model_selection import cross_validate
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
results = cross_validate(model, X, y, cv=5, scoring=scoring, return_train_score=True)

print("\\nMultiple Metrics:")
for metric in scoring:
  test_score = results[f'test_{metric}'].mean()
  train_score = results[f'train_{metric}'].mean()
  print(f"{metric}: Train={train_score:.3f}, Test={test_score:.3f}")`,
      explanation: 'Demonstrates standard k-fold, stratified k-fold, and multi-metric cross-validation for classification. Stratified CV is crucial for imbalanced datasets.'
    },
    {
      language: 'Python',
      code: `from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Simulate time series data
n_samples = 1000
X = np.random.randn(n_samples, 10)
y = np.cumsum(np.random.randn(n_samples))  # Time-dependent target

# Time series cross-validation (respects temporal order)
tscv = TimeSeriesSplit(n_splits=5)

model = RandomForestRegressor(random_state=42)
scores = []

print("Time Series Cross-Validation:")
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
  X_train, X_val = X[train_idx], X[val_idx]
  y_train, y_val = y[train_idx], y[val_idx]

  model.fit(X_train, y_train)
  score = model.score(X_val, y_val)
  scores.append(score)

  print(f"Fold {fold+1}: Train size={len(train_idx)}, Val size={len(val_idx)}, Score={score:.3f}")

print(f"\\nMean R² Score: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")

# Nested cross-validation for hyperparameter tuning
param_grid = {
  'n_estimators': [50, 100, 200],
  'max_depth': [5, 10, None]
}

# Inner CV: hyperparameter tuning
inner_cv = TimeSeriesSplit(n_splits=3)
clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv, scoring='r2')

# Outer CV: performance estimation
outer_cv = TimeSeriesSplit(n_splits=5)
nested_scores = cross_val_score(clf, X, y, cv=outer_cv, scoring='r2')

print(f"\\nNested CV R² Score: {nested_scores.mean():.3f} (+/- {nested_scores.std():.3f})")`,
      explanation: 'Shows time series cross-validation that respects temporal order, and nested CV for unbiased hyperparameter tuning. Essential for financial, weather, or any time-dependent data.'
    }
  ],
  interviewQuestions: [
    {
      question: 'Why is cross-validation better than a single train-test split?',
      answer: 'Cross-validation provides more reliable and robust performance estimates than a single train-test split by using your data more efficiently and reducing variance in the evaluation. With a single split, your performance estimate depends heavily on which specific samples happened to land in the test set—you might get lucky (easy test samples) or unlucky (hard test samples), leading to misleading conclusions. Cross-validation averages performance across multiple different train-test splits, giving both an expected performance (the mean across folds) and uncertainty quantification (standard deviation across folds).\n\nThe data efficiency argument is compelling, especially for small datasets. In a single 80-20 split, you train on 80% of data and evaluate on 20%. In 5-fold cross-validation, each fold trains on 80% and evaluates on the remaining 20%, but across the 5 folds, every data point serves in testing exactly once and in training four times. You get five performance estimates instead of one, each on a different 20% of the data, providing much more thorough evaluation coverage. For small datasets where every sample is precious, this efficiency is crucial.\n\nCross-validation also helps detect overfitting to the test set through model selection. If you try many model variations and select the one with best test performance on a single held-out test set, that test performance becomes overly optimistic—you\'ve indirectly fitted the test set through the selection process. Cross-validation mitigates this because the same test samples aren\'t used repeatedly; each fold sees different test data. The main downside is computational cost (k-fold requires training k models instead of one), but for most applications this is manageable and worthwhile for the improved reliability. For production systems, it\'s common to use cross-validation during development for robust model selection, then train a final model on all available data once the architecture and hyperparameters are chosen.'
    },
    {
      question: 'When should you use stratified k-fold cross-validation?',
      answer: 'Use stratified k-fold cross-validation whenever you have imbalanced class distributions in classification tasks. Stratified sampling ensures that each fold maintains approximately the same class distribution as the overall dataset. Without stratification, random folding might create folds with very different class distributions—one fold might have 10% positive class while another has 25%—making performance estimates unreliable and training unstable. For example, with 95% negative and 5% positive classes, a random fold might accidentally have zero positive samples, making it impossible to compute recall or F1-score for that fold.\n\nStratification is critical for severe imbalance (99:1 or worse) but beneficial even for moderate imbalance (70:30). It reduces variance in performance estimates across folds and ensures all folds are representative. This means you get more consistent fold-to-fold results, making the average performance a better estimate of true generalization. Stratification also ensures that all classes appear in both training and validation sets for each fold, which is essential for the model to learn all classes and for metrics to be computable.\n\nFor multi-class problems, stratified cross-validation maintains the proportion of all classes across folds, which is especially important if some classes are rare. For regression tasks, you can create stratified folds by binning the continuous target into quantiles and stratifying on these bins, ensuring each fold spans the full range of target values rather than concentrating high values in some folds and low values in others. The only situations where you shouldn\'t use stratification are: time-series data (where temporal order must be preserved), grouped data (where samples within groups must stay together), or when class distribution is expected to differ between training and deployment (though this indicates a more fundamental problem). In sklearn, it\'s as simple as using StratifiedKFold instead of KFold, with no computational downside.'
    },
    {
      question: 'What is the main limitation of leave-one-out cross-validation (LOOCV)?',
      answer: 'The primary limitation of LOOCV is computational cost—it requires training n models where n is the number of samples, which becomes prohibitive for large datasets or computationally expensive models (deep neural networks, gradient boosting with many trees). Training thousands or millions of models is simply infeasible in most practical scenarios. Unlike k-fold CV where you can choose k=5 or 10 to balance reliability and computation, LOOCV has no such flexibility; the number of folds equals your sample size.\n\nA more subtle but equally important limitation is high variance in the performance estimate. LOOCV has low bias (each training set contains n-1 samples, nearly all the data), but high variance because the n training sets are highly correlated—they overlap in n-2 samples and differ by only one sample. Changes in that single different sample can\'t create much variation in the resulting models, so the n performance estimates are not independent. Averaging non-independent estimates doesn\'t reduce variance as effectively as averaging independent estimates would. Empirically, 5 or 10-fold CV often gives performance estimates with lower variance than LOOCV, despite using less data per fold.\n\nLOOCV can also be misleading for model selection with certain algorithms. Since each model is trained on nearly all data (n-1 samples), the performance estimates are optimistic compared to training on your actual training set size (which would be smaller if you held out a proper test set). For unstable algorithms (high-variance models like decision trees or k-NN with small k), LOOCV can produce highly variable predictions across folds, making the average performance less meaningful. LOOCV is primarily useful for small datasets (n<100) where you can\'t afford to lose 20% of data to a validation fold, and for algorithms where training is cheap (linear models, k-NN). For most modern applications with moderate-to-large datasets and complex models, 5 or 10-fold cross-validation is preferred as a better balance of statistical properties, computational cost, and practical utility.'
    },
    {
      question: 'How does time series cross-validation differ from standard k-fold CV?',
      answer: 'Time series cross-validation must respect temporal ordering, whereas standard k-fold CV randomly shuffles data before splitting. The fundamental principle is that you can only train on past data and validate on future data, never the reverse. Shuffling destroys this temporal structure, creating leakage where future information influences training. Standard k-fold would train on a random 80% (including future observations) and test on a random 20% (including past observations), which is nonsensical for time series—you can\'t predict yesterday using tomorrow\'s data.\n\nTimeSeriesSplit in sklearn implements the correct approach using expanding or rolling windows. In expanding window mode, each successive fold includes all previous data: fold 1 trains on samples 1-100 and tests on 101-150; fold 2 trains on 1-150 and tests on 151-200; fold 3 trains on 1-200 and tests on 201-250, etc. This mimics realistic deployment where you continuously retrain on all historical data. Rolling window mode maintains fixed training size: fold 1 uses 1-100 for training; fold 2 uses 51-150; fold 3 uses 101-200, etc. Rolling windows are useful when recent data is more relevant (concept drift) or when computational constraints limit training on all historical data.\n\nA crucial difference is that test sets must always come after training sets chronologically. This creates fewer, sequential folds rather than random permutations. You also can\'t use all data equally—early data appears in training more often than late data, and the final data only appears in test sets. This is intentional and necessary to prevent leakage. When evaluating performance, be aware that each fold tests on different time periods which might have different characteristics (seasonality, trends, regime changes). Report performance on each fold separately in addition to the average, as this reveals whether your model performance is stable over time or degrades for certain periods. Never use standard k-fold, stratified k-fold, or LOOCV for time series—they all violate temporal causality and will produce misleadingly optimistic results that fail catastrophically in production.'
    },
    {
      question: 'What is nested cross-validation and when is it necessary?',
      answer: 'Nested cross-validation is a two-level cross-validation procedure that separates model selection (choosing hyperparameters) from performance estimation. The outer loop provides an unbiased estimate of the final model\'s generalization performance, while the inner loop performs hyperparameter tuning without contaminating the outer performance estimate. This is necessary whenever you need both reliable hyperparameter optimization and honest performance reporting, particularly for research or production systems where accurate performance guarantees matter.\n\nThe structure involves an outer k-fold split (typically 5-fold) for performance estimation, and for each outer fold, an inner k-fold split (typically 3 or 5-fold) for hyperparameter tuning. For each outer fold: take the outer training data, run inner cross-validation across different hyperparameter values, select the best hyperparameters based on inner validation performance, train a model with those hyperparameters on the full outer training data, and evaluate on the outer test fold. Repeat for all outer folds, then average the outer test performances. This gives an unbiased performance estimate because the outer test folds were never used for hyperparameter selection.\n\nWithout nested CV, if you use the same cross-validation splits for both hyperparameter tuning and performance estimation, you get overly optimistic estimates. After trying many hyperparameter combinations and selecting the best based on CV performance, that CV performance is biased upward—you\'ve indirectly fitted the validation data through the selection process. Nested CV solves this by keeping outer test data completely isolated from the model selection process. The computational cost is significant (k_outer × k_inner × n_hyperparameter_combinations models), but necessary for honest reporting. In practice, use nested CV when publishing research (to report unbiased performance), deploying high-stakes models (medical, financial), or when you need confidence intervals on performance. For informal model comparison or when computational budget is tight, standard CV for hyperparameter tuning followed by a separate held-out test set is a reasonable compromise.'
    },
    {
      question: 'Why might you get overly optimistic performance estimates if you tune hyperparameters using the same CV splits?',
      answer: 'This creates a subtle form of overfitting where you indirectly fit the validation data through the hyperparameter selection process, even though you never directly trained on validation samples. When you try many hyperparameter combinations (50 learning rates, 10 regularization strengths, 5 architectures = 2500 combinations) and select the one with best cross-validation performance, you\'re essentially running 2500 experiments and choosing the luckiest result. Some combinations will perform well by chance—random fluctuations in the specific validation samples favor certain hyperparameters. Reporting the best CV score as your expected performance is overly optimistic.\n\nThe validation data has been "used up" through repeated evaluation. Each time you evaluate a new hyperparameter configuration on the validation folds, you gain information about those specific samples and adjust your choices accordingly. After extensive hyperparameter search, the selected configuration is optimized for the peculiarities of those validation folds, not just for the underlying data distribution. This is particularly severe with automated hyperparameter optimization (grid search, random search, Bayesian optimization) that might evaluate hundreds or thousands of configurations. The more configurations you try, the more likely you are to find one that excels on your validation set by chance.\n\nThe solution is nested cross-validation or a three-way split. Nested CV uses inner folds for hyperparameter selection and outer folds for unbiased performance estimation. The three-way approach uses training data for model fitting, validation data for hyperparameter selection, and a completely held-out test set for final performance reporting. The test set must only be evaluated once after all model decisions are finalized. The magnitude of optimism depends on search intensity: trying 5 hyperparameter values introduces modest bias, while trying 1000 introduces substantial bias. This is why kaggle competitions often have public and private leaderboards—the public leaderboard (validation set) is visible during the competition for model development, but final ranking uses the hidden private leaderboard (test set) to prevent overfitting to the public scores through repeated submissions.'
    }
  ],
  quizQuestions: [
    {
      id: 'cv-q1',
      question: 'You are building a fraud detection model where only 2% of transactions are fraudulent. Which cross-validation strategy is most appropriate?',
      options: [
        'Standard k-fold cross-validation',
        'Stratified k-fold cross-validation',
        'Leave-one-out cross-validation (LOOCV)',
        'Simple train-test split'
      ],
      correctAnswer: 1,
      explanation: 'Stratified k-fold ensures each fold maintains the 2% fraud rate. Standard k-fold might create folds with 0% or highly variable fraud rates, leading to unreliable performance estimates.'
    },
    {
      id: 'cv-q2',
      question: 'You are predicting stock prices using historical data. Your model performs well in cross-validation (R²=0.85) but poorly in production (R²=0.30). What is the most likely cause?',
      options: [
        'The model is underfitting',
        'You used standard k-fold CV instead of time series CV, causing data leakage',
        'The test set is too small',
        'The model needs more regularization'
      ],
      correctAnswer: 1,
      explanation: 'Standard k-fold randomly shuffles data, allowing the model to "peek" at future information during training. Time series CV respects temporal order, training only on past data to predict future values.'
    },
    {
      id: 'cv-q3',
      question: 'When performing hyperparameter tuning with GridSearchCV, why should you use nested cross-validation for final model evaluation?',
      options: [
        'It trains faster than single-level CV',
        'It prevents data leakage between hyperparameter tuning and performance estimation',
        'It requires less data than standard CV',
        'It always produces higher accuracy scores'
      ],
      correctAnswer: 1,
      explanation: 'Using the same CV folds for both hyperparameter tuning and performance estimation leaks information, giving overly optimistic results. Nested CV uses separate outer folds for unbiased performance estimation.'
    }
  ]
};
