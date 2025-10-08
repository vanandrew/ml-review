import { Topic } from '../../../types';

export const trainValidationTestSplit: Topic = {
  id: 'train-validation-test-split',
  title: 'Train-Validation-Test Split',
  category: 'foundations',
  description: 'Understanding data splitting strategies for model development and evaluation',
  content: `
    <h2>The Fundamental Practice of Data Splitting</h2>
    <p>Data splitting is one of the most critical practices in machine learning, yet it's often misunderstood or improperly executed. The way you divide your data fundamentally affects your ability to train effective models and honestly assess their performance. Poor splitting strategies can lead to overly optimistic performance estimates that collapse in production, wasted time tuning models on contaminated validation sets, or models that fail to generalize because they've seen test data during development.</p>

    <div class="info-box info-box-green">
      <h4>📋 Quick Reference: Recommended Split Ratios</h4>
      <table>
        <tr>
          <th>Dataset Size</th>
          <th>Recommended Split</th>
          <th>Notes</th>
        </tr>
        <tr>
          <td>Very Large (>1M)</td>
          <td>98-1-1</td>
          <td>1% is plenty for validation/test</td>
        </tr>
        <tr>
          <td>Large (100K-1M)</td>
          <td>80-10-10</td>
          <td>Standard for deep learning</td>
        </tr>
        <tr>
          <td>Medium (10K-100K)</td>
          <td>70-15-15</td>
          <td>Balanced approach</td>
        </tr>
        <tr>
          <td>Small (<10K)</td>
          <td>60-20-20 + CV</td>
          <td>Use cross-validation</td>
        </tr>
        <tr>
          <td>Very Small (<1K)</td>
          <td>80-20 (CV only)</td>
          <td>k-fold CV, small test set</td>
        </tr>
      </table>
      <p><strong>Special Cases:</strong> Time series → chronological splits | Imbalanced → stratified | Grouped data → split by groups</p>
    </div>

    <h3>Why We Split Data: The Core Problem</h3>
    <p>The fundamental challenge in machine learning is <strong>generalization</strong>\u2014building models that perform well on new, unseen data, not just the data they were trained on. Without proper data splitting, you have no reliable way to estimate how your model will perform in the real world. If you train and test on the same data, perfect performance tells you nothing\u2014the model may have simply memorized the data.</p>
    
    <p>Data splitting simulates the real-world scenario where your model will encounter new examples. By holding out portions of your data and never using them during training, you create a realistic test of the model's ability to generalize. This separation is crucial for honest performance assessment and guides practically every decision in model development.</p>

    <h3>The Three Essential Splits: Purpose and Roles</h3>
    
    <p><strong>1. Training Set (Typical size: 60-80% of data)</strong></p>
    <p>The training set is the data your model directly learns from. During training, the model sees the input features and their corresponding labels (in supervised learning), and adjusts its internal parameters to minimize prediction error on these examples. This is where the actual learning happens\u2014weights are updated, decision boundaries are formed, patterns are recognized.</p>
    
    <p><strong>What it's used for:</strong></p>
    <ul>
      <li>Fitting model parameters (weights, coefficients, tree structures)</li>
      <li>Learning the mapping from features to targets</li>
      <li>Gradient descent optimization</li>
      <li>Pattern recognition and representation learning</li>
    </ul>
    
    <p><strong>Key principle:</strong> The training set should be large enough to learn meaningful patterns but small enough to leave sufficient data for validation and testing. Too small, and your model won't learn well. Too large (using validation/test data for training), and you lose the ability to assess generalization.</p>
    
    <p><strong>2. Validation Set (Typical size: 10-20% of data)</strong></p>
    <p>The validation set (also called development set or dev set) serves as a proxy for unseen data during model development. It's used iteratively throughout the modeling process to make decisions about model architecture, hyperparameters, and features. Critically, the model never trains on this data\u2014it only uses it for evaluation to guide development choices.</p>
    
    <p><strong>What it's used for:</strong></p>
    <ul>
      <li><strong>Hyperparameter tuning:</strong> Choosing learning rate, regularization strength, tree depth, number of layers, etc.</li>
      <li><strong>Model selection:</strong> Comparing different algorithms or architectures</li>
      <li><strong>Early stopping:</strong> Deciding when to halt training (when validation performance stops improving)</li>
      <li><strong>Feature selection:</strong> Determining which features improve generalization</li>
      <li><strong>Architecture search:</strong> Finding optimal neural network structures</li>
      <li><strong>Debugging:</strong> Understanding when and how your model fails</li>
    </ul>
    
    <p><strong>Why it becomes "biased":</strong> Through repeated evaluation and model selection, you indirectly optimize for the validation set. After trying 100 different hyperparameter configurations and choosing the one with best validation performance, that validation score is optimistically biased\u2014you've effectively searched over the validation set to find what works best for it specifically.</p>
    
    <p><strong>3. Test Set (Typical size: 10-20% of data)</strong></p>
    <p>The test set is your final, unbiased assessment of model performance. It should be touched exactly once\u2014after all modeling decisions are completely finalized. This set answers the question: "How well will this model perform in production on truly new data?" Because it's never used during development, it provides an honest estimate of generalization.</p>
    
    <p><strong>What it's used for:</strong></p>
    <ul>
      <li><strong>Final performance evaluation:</strong> Reporting honest metrics to stakeholders</li>
      <li><strong>Model comparison:</strong> Fair comparison between different complete modeling pipelines</li>
      <li><strong>Production readiness:</strong> Determining if the model meets requirements</li>
      <li><strong>Research reporting:</strong> Publishing unbiased results</li>
    </ul>
    
    <p><strong>Critical rule:</strong> Use the test set exactly once. If you use it multiple times to make decisions, it becomes another validation set and loses its unbiased property. If you need to iterate further after seeing test results, you should ideally collect new test data.</p>

    <h3>Common Split Ratios and When to Use Them</h3>
    
    <p><strong>70-15-15 Split (Standard Approach):</strong></p>
    <ul>
      <li>Balanced approach for medium-sized datasets (10,000-100,000 samples)</li>
      <li>Provides sufficient data for all three purposes</li>
      <li>15% validation set allows reliable hyperparameter tuning</li>
      <li>15% test set gives stable performance estimates</li>
    </ul>
    
    <p><strong>80-10-10 Split (For Larger Datasets):</strong></p>
    <ul>
      <li>Use when you have ample data (100,000+ samples)</li>
      <li>Maximizes training data while maintaining adequate validation/test sets</li>
      <li>10% of 100,000 is 10,000 samples\u2014plenty for validation and testing</li>
      <li>Preferred when model is complex and needs more training examples</li>
    </ul>
    
    <p><strong>60-20-20 Split (For Smaller Datasets or Complex Models):</strong></p>
    <ul>
      <li>Use when you need robust validation and testing despite limited data</li>
      <li>Larger validation set supports more extensive hyperparameter search</li>
      <li>Larger test set provides more stable performance estimates</li>
      <li>Trade-off: less training data may limit model performance</li>
    </ul>
    
    <p><strong>98-1-1 Split (For Very Large Datasets):</strong></p>
    <ul>
      <li>Appropriate when you have millions of examples</li>
      <li>1% of 10 million is 100,000 samples\u2014more than sufficient for validation/testing</li>
      <li>Common in deep learning with massive datasets</li>
      <li>Maximizes training data for hungry models</li>
    </ul>
    
    <p><strong>No Fixed Split (Cross-Validation for Small Datasets):</strong></p>
    <ul>
      <li>When you have very limited data (hundreds to few thousand samples)</li>
      <li>Use k-fold cross-validation instead of fixed splits</li>
      <li>Still hold out a final test set if possible (e.g., 80% for CV, 20% for testing)</li>
      <li>Provides more reliable estimates with limited data</li>
    </ul>

    <h3>Critical Best Practices</h3>
    
    <p><strong>1. Create Splits BEFORE Any Analysis</strong></p>
    <p>This is perhaps the most important rule: split your data <em>before</em> looking at it, before exploratory data analysis, before feature engineering. Any insights gained from examining the full dataset can unconsciously bias your modeling decisions toward the test set. The test and validation sets should represent truly unseen data.</p>
    
    <p><strong>2. Stratified Sampling for Classification</strong></p>
    <p>Stratified sampling maintains the same class distribution across all splits as in the original dataset. This is essential for imbalanced datasets where random sampling might accidentally place most minority class samples in one set.</p>
    
    <p><strong>Example:</strong> If your dataset has 90% class A and 10% class B, stratified splitting ensures training, validation, and test sets all have approximately 90-10 distribution. Without stratification, you might end up with 95-5 in training and 80-20 in test, making them non-representative.</p>
    
    <p><strong>When to use:</strong></p>
    <ul>
      <li>Any classification task with imbalanced classes</li>
      <li>Even moderately imbalanced data (60-40) benefits from stratification</li>
      <li>Multi-class problems to ensure all classes are represented in each split</li>
      <li>Small datasets where random variation could cause significant skew</li>
    </ul>
    
    <p><strong>3. Chronological Splits for Time-Series Data</strong></p>
    <p>Time-series data has temporal dependencies and ordering that must be respected. Shuffling before splitting creates <strong>temporal leakage</strong>\u2014the model learns from the future to predict the past, which is impossible in deployment.</p>
    
    <p><strong>Correct approach for time-series:</strong></p>
    <ul>
      <li><strong>Training set:</strong> Oldest data (e.g., January-August)</li>
      <li><strong>Validation set:</strong> Middle period (e.g., September-October)</li>
      <li><strong>Test set:</strong> Most recent data (e.g., November-December)</li>
      <li><strong>Never shuffle:</strong> Maintain chronological order</li>
      <li><strong>Forward validation:</strong> Always train on past, predict on future</li>
    </ul>
    
    <p><strong>Why this matters:</strong> In production, your model will predict the future based on historical data. Your evaluation should simulate this. If you shuffle, excellent test performance might disappear in deployment because the model was trained on future information it won't have access to.</p>
    
    <p><strong>4. Preventing Data Leakage</strong></p>
    <p>Data leakage is when information from validation or test sets influences training, either directly or indirectly. This is insidious because it inflates performance metrics while model development but leads to poor real-world performance.</p>
    
    <p><strong>Common leakage sources:</strong></p>
    <ul>
      <li><strong>Feature scaling on combined data:</strong> Computing mean/std on all data before splitting leaks test statistics to training. Compute on training set only, then apply to validation/test.</li>
      <li><strong>Feature engineering with global statistics:</strong> Creating features using all data (e.g., user's average behavior) leaks information. Use only training data for statistics.</li>
      <li><strong>Imputation:</strong> Filling missing values using all data. Should use training set statistics only.</li>
      <li><strong>Feature selection:</strong> Selecting features based on correlation with target using all data. Should use training set only.</li>
      <li><strong>Duplicate examples:</strong> Same example appearing in training and test (common after oversampling). Remove duplicates across splits.</li>
      <li><strong>Temporal leakage:</strong> Using future information to predict the past in time-series.</li>
      <li><strong>Group leakage:</strong> Same patient/user/entity appearing in both training and test with correlated examples.</li>
    </ul>
    
    <p><strong>The golden rule:</strong> Any transformation, statistic, or decision should be based solely on the training set, then applied to validation and test sets using the training set's parameters.</p>
    
    <p><strong>5. Random Shuffling (For Non-Temporal Data)</strong></p>
    <p>Before splitting non-temporal data, shuffle it randomly. This prevents bias from any ordering in your dataset (e.g., if all positive examples come first, unshuffled splitting might put them all in training).</p>
    
    <p><strong>6. Set Random Seeds for Reproducibility</strong></p>
    <p>Always set random seeds when splitting so you and others can reproduce your exact splits. This is crucial for debugging, collaboration, and scientific reproducibility.</p>
    
    <p><strong>7. Holdout Validation vs. Cross-Validation</strong></p>
    <p><strong>Holdout validation</strong> is a single fixed split (the approach described above). It's simple and fast but can have high variance\u2014your performance estimate depends on which specific samples ended up in validation.</p>
    
    <p><strong>Cross-validation</strong> (typically k-fold) uses multiple train-validation splits, training k models and averaging their performance. This provides more robust estimates and uses data more efficiently but is k times more computationally expensive.</p>
    
    <p><strong>When to use each:</strong></p>
    <ul>
      <li><strong>Holdout:</strong> Large datasets, expensive models, quick iteration needed</li>
      <li><strong>Cross-validation:</strong> Small/medium datasets, model selection, research reporting, when you need reliable estimates</li>
      <li><strong>Hybrid:</strong> Use cross-validation on training data for model selection, maintain a held-out test set for final evaluation</li>
    </ul>

    <h3>Special Considerations</h3>
    
    <p><strong>Very Large Datasets (Millions of Examples):</strong></p>
    <p>With abundant data, you can use smaller percentages for validation and test while maintaining absolute size. Even 0.5% of 10 million examples gives 50,000 samples\u2014plenty for reliable validation. Prioritize giving as much data as possible to training.</p>
    
    <p><strong>Very Small Datasets (Hundreds of Examples):</strong></p>
    <p>Fixed splits waste precious data and give unreliable estimates. Use k-fold cross-validation (k=5 or 10) for the train-validation phase. If possible, still hold out a small test set (10-20%) for final evaluation, but use cross-validation for all development.</p>
    
    <p><strong>Imbalanced Data:</strong></p>
    <p>Always use stratified splitting. For severe imbalance (99:1), consider stratified k-fold cross-validation even with moderate dataset sizes. Ensure minority class has enough examples in each split for meaningful learning and evaluation (absolute counts matter, not just percentages).</p>
    
    <p><strong>Grouped Data (Multiple Samples Per Entity):</strong></p>
    <p>If you have multiple examples per patient, user, or device, split by entity, not by example. Having the same patient in both training and test creates leakage\u2014the model learns that patient's patterns in training and exploits them in testing, overestimating generalization to new patients. Use GroupKFold or GroupShuffleSplit from scikit-learn.</p>
    
    <p><strong>Early Stopping in Neural Networks:</strong></p>
    <p>The validation set plays a crucial role in training neural networks. Monitor validation loss during training and stop when it stops improving (early stopping). This prevents overfitting by halting training at the point of best generalization, even if training loss could continue decreasing.</p>

    <h3>Common Mistakes and How to Avoid Them</h3>
    
    <p><strong>Mistake 1: Using test set multiple times</strong></p>
    <p><em>Solution:</em> Treat test set as sacred. Use only once at the very end. For iterative development, rely on validation set or cross-validation.</p>
    
    <p><strong>Mistake 2: Preprocessing before splitting</strong></p>
    <p><em>Solution:</em> Split first, then preprocess. Fit transformers (scalers, encoders) on training data only, then transform all sets.</p>
    
    <p><strong>Mistake 3: Shuffling time-series data</strong></p>
    <p><em>Solution:</em> Use chronological splits. Training on past, validation on recent past, test on most recent.</p>
    
    <p><strong>Mistake 4: Not stratifying imbalanced data</strong></p>
    <p><em>Solution:</em> Always use stratified splitting for classification, especially with imbalance.</p>
    
    <p><strong>Mistake 5: Ignoring grouped structure</strong></p>
    <p><em>Solution:</em> Split by groups (patients, users) not individual examples when data has hierarchical structure.</p>
    
    <p><strong>Mistake 6: Wrong split ratios for dataset size</strong></p>
    <p><em>Solution:</em> With millions of examples, use 98-1-1. With hundreds, use cross-validation. Scale ratios to absolute sample counts.</p>

    <h3>Verification: Are Your Splits Valid?</h3>
    <p>After splitting, always verify:</p>
    <ul>
      <li><strong>No overlap:</strong> No examples appear in multiple splits</li>
      <li><strong>Class distribution:</strong> Similar class proportions across splits (for classification)</li>
      <li><strong>Statistical properties:</strong> Similar feature distributions across splits</li>
      <li><strong>Temporal order:</strong> Test set is chronologically after training for time-series</li>
      <li><strong>Group separation:</strong> No group appears in multiple splits</li>
      <li><strong>Size expectations:</strong> Each split has expected number of samples</li>
    </ul>
  `,
  codeExamples: [
    {
      language: 'Python',
      code: `from sklearn.model_selection import train_test_split
import numpy as np

# Sample dataset
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)

# First split: separate test set (80-20 split)
X_temp, X_test, y_temp, y_test = train_test_split(
  X, y,
  test_size=0.2,
  random_state=42,
  stratify=y
)

# Second split: separate validation from training
X_train, X_val, y_train, y_val = train_test_split(
  X_temp, y_temp,
  test_size=0.25,
  random_state=42,
  stratify=y_temp
)

print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")`,
      explanation: 'This example demonstrates the standard approach for creating train-validation-test splits with stratification to maintain class balance.'
    },
    {
      language: 'Python',
      code: `import pandas as pd

# Time-series dataset example
dates = pd.date_range('2020-01-01', periods=1000, freq='D')
df = pd.DataFrame({
  'date': dates,
  'value': np.random.randn(1000)
})

# Chronological split (NO shuffling for time series)
train_size = int(0.6 * len(df))
val_size = int(0.2 * len(df))

train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:train_size + val_size]
test_df = df.iloc[train_size + val_size:]

print(f"Train: {train_df['date'].min()} to {train_df['date'].max()}")
print(f"Val: {val_df['date'].min()} to {val_df['date'].max()}")
print(f"Test: {test_df['date'].min()} to {test_df['date'].max()}")`,
      explanation: 'For time-series data, we must preserve chronological order and never shuffle. The test set contains the most recent data.'
    }
  ],
  interviewQuestions: [
    {
      question: 'Why do we need three separate datasets instead of just training and testing?',
      answer: 'The three-way split—training, validation, and test—serves distinct purposes in the machine learning pipeline. The training set is used to fit model parameters (weights, coefficients). The validation set is used for hyperparameter tuning and model selection without touching the test set. The test set provides an unbiased estimate of final model performance on truly unseen data. Without this separation, you risk overfitting your model selection process to the test set.\n\nWith just training and testing, you face a dilemma during model development. If you tune hyperparameters (learning rate, regularization strength, tree depth) based on test performance, you\'re indirectly fitting the test set—not through direct training, but through the iterative model selection process. After dozens of experiments choosing the model with best test performance, that test performance becomes an overoptimistic estimate. The test set has been "used up" through repeated evaluation. The validation set solves this by providing a separate dataset for these model selection decisions, preserving the test set for a final, unbiased evaluation.\n\nIn practice, the workflow is: train multiple models on training data, evaluate them on validation data to choose the best architecture/hyperparameters, then report final performance on the untouched test set. This discipline ensures honest performance reporting. Some practitioners use k-fold cross-validation for model selection (the validation phase), which uses the training data more efficiently. The key principle remains: the test set must only be used once, at the very end, after all model decisions are finalized. This prevents "validation set overfitting" and maintains statistical validity of your performance claims.'
    },
    {
      question: 'What is data leakage and how can improper splitting cause it?',
      answer: 'Data leakage occurs when information from outside the training data influences the model in ways that won\'t be available at prediction time, leading to overly optimistic performance estimates and poor real-world results. Improper splitting is a common source of leakage. The most basic form is test set leakage: accidentally including test samples in training, or applying transformations (normalization, feature engineering) on the combined dataset before splitting. This gives the model information about the test distribution during training, inflating performance metrics.\n\nTemporal leakage is particularly insidious with time-series data. If you shuffle before splitting, future information leaks into the training set—the model learns from tomorrow to predict yesterday, which is impossible in deployment. For example, in stock price prediction, shuffling mixes future prices into training, yielding unrealistically good results. The correct approach is chronological splitting: train on oldest data, validate on middle data, test on most recent. Similarly, with patient medical records, training on later visits while testing on earlier ones leaks information about disease progression.\n\nFeature engineering leakage is subtle but critical. If you compute statistics (mean, standard deviation, min/max for normalization) using all data before splitting, your training set knows about test set statistics. The solution is to compute these statistics only on training data, then apply the same transformation to validation and test sets. Other leakage sources include duplicate samples across sets (common with oversampling), target variable information in features (e.g., a "was_converted" feature in a conversion prediction task), or using forward-looking information (features that wouldn\'t be available at prediction time, like "total_purchases_this_year" in a model predicting January purchases).'
    },
    {
      question: 'How would you split a highly imbalanced dataset?',
      answer: 'For imbalanced datasets, stratified splitting is essential to maintain class distribution across train, validation, and test sets. Without stratification, random splitting might put most or all minority class samples in one set, making it impossible to learn or evaluate that class properly. Stratified sampling ensures each split contains approximately the same percentage of each class as the full dataset. For example, with 95% negative and 5% positive samples, stratification ensures training, validation, and test sets each have roughly this 95:5 ratio.\n\nThe implementation is straightforward in sklearn: use stratify parameter in train_test_split, passing the target labels. For multi-way splits, apply stratification twice: first split off the test set with stratification, then split the remainder into train/validation, again with stratification. This preserves class distribution at each step. For extreme imbalance (99:1 or worse), consider absolute counts—ensure the minority class has enough samples in each set for meaningful learning and evaluation, even if it means adjusting split ratios.\n\nBeyond stratification, consider your evaluation strategy. With severe imbalance, accuracy is meaningless (predicting all majority class gives high accuracy), so use appropriate metrics: precision, recall, F1-score, AUC-ROC, or AUC-PR. Your validation set must be large enough to reliably estimate these metrics for the minority class. Sometimes stratified k-fold cross-validation is better than a single train/val/test split, as it provides more robust estimates and uses data more efficiently. If the imbalance is so extreme that even stratified splitting leaves too few minority samples per fold, consider stratified sampling with replacement or advanced techniques like stratified group k-fold for grouped data.'
    },
    {
      question: 'Why should you never shuffle time-series data before splitting?',
      answer: 'Shuffling time-series data before splitting creates severe temporal leakage, fundamentally breaking the prediction task. In time-series problems, you\'re predicting the future based on the past—the temporal ordering is intrinsic to the problem. Shuffling mixes future observations into the training set, allowing the model to learn from future data when predicting the past. This produces artificially inflated performance that completely fails in production, where future data isn\'t available.\n\nThe consequences extend beyond just leakage. Many time-series have autocorrelation (correlation between observations at different time lags) and trends. Shuffling destroys these temporal dependencies that the model needs to learn. For example, in stock prices, consecutive days are correlated—today\'s price informs tomorrow\'s. Shuffling breaks these correlations, creating a jumbled dataset that doesn\'t reflect the sequential nature of the real problem. Your model might learn spurious patterns from the shuffled data that don\'t exist in actual time sequences.\n\nThe correct approach is chronological splitting: use oldest data for training, recent data for validation, and most recent for testing. This mimics deployment conditions where you train on historical data and predict future values. For cross-validation with time-series, use specialized techniques like TimeSeriesSplit which respects temporal order, creating multiple train/test splits where each test set is later than its corresponding training set. Walk-forward validation is another approach, where you repeatedly train on historical windows and test on the immediate next period, rolling forward through time. These methods maintain temporal integrity while still providing robust performance estimates.'
    },
    {
      question: 'If you have only 500 samples, what splitting strategy would you recommend?',
      answer: 'With limited data (500 samples), every sample is precious, and traditional splits (60/20/20 or 70/15/15) leave validation and test sets too small for reliable performance estimates. K-fold cross-validation is typically the best approach here—it uses data more efficiently by ensuring every sample serves in both training and validation across different folds. For 500 samples, 5 or 10-fold cross-validation works well: each fold uses 80-90% of data for training and 10-20% for validation, providing more robust performance estimates through averaging.\n\nThe workflow changes slightly: instead of a single validation set, you train k models (one per fold) and report average validation performance plus standard deviation across folds. This gives both a performance estimate and uncertainty quantification. For hyperparameter tuning, use nested cross-validation: an outer loop for performance estimation and an inner loop for hyperparameter selection within each outer fold. This prevents overfitting the validation process while maximizing data usage. The computational cost increases linearly with k, but with only 500 samples, this is usually manageable.\n\nIf you need a held-out test set for final evaluation (recommended for production models), consider a modified approach: set aside a stratified 15-20% test set (75-100 samples), then use cross-validation on the remaining 80-85% for model development. This balances efficient data usage during development with an unbiased final test. Alternatively, use repeated k-fold cross-validation (running k-fold multiple times with different random seeds) or leave-one-out cross-validation (LOOCV, where k equals sample size) for very small datasets, though LOOCV has high variance and computational cost. The key is avoiding waste through excessive splitting while maintaining reliable performance estimates through resampling techniques.'
    },
    {
      question: 'What is stratified splitting and when should you use it?',
      answer: 'Stratified splitting ensures that each split (train, validation, test) maintains the same class distribution as the original dataset. Instead of random sampling, stratified sampling samples separately from each class proportionally. If your dataset has 70% class A and 30% class B, stratified splitting ensures each set has approximately the same 70:30 ratio. This is implemented by sampling 70% of class A samples and 70% of class B samples for training, leaving 30% of each for validation/testing, then further splitting that 30% into validation and test sets.\n\nYou should use stratified splitting for any classification task with imbalanced classes. Even moderate imbalance (60:40) can benefit, as it reduces variance in performance estimates and ensures all classes are represented in each set. For severe imbalance (95:5 or worse), stratification is critical—random splitting might accidentally place most minority class samples in one set, making it impossible to train or evaluate properly. Stratification also matters for small datasets where random fluctuations could create misleading splits. For example, with 100 samples and 20% minority class, random splitting might give training sets with 15-25% minority samples just by chance, whereas stratification ensures consistent 20%.\n\nBeyond binary classification, use stratified splitting for multi-class problems to maintain representation of all classes, especially if some classes are rare. For continuous regression targets, you can create stratified splits by binning the target into quantiles and stratifying on these bins—this ensures each set spans the full range of target values rather than accidentally concentrating high values in training and low values in testing. Don\'t use stratified splitting for time-series (violates temporal ordering) or when class distribution is expected to shift between training and deployment (though this indicates a more fundamental problem with your modeling approach).'
    }
  ],
  quizQuestions: [
    {
      id: 'split1',
      question: 'What is the primary purpose of the validation set?',
      options: [
        'To train the model',
        'To tune hyperparameters and select models',
        'To provide final performance evaluation',
        'To augment training data'
      ],
      correctAnswer: 1,
      explanation: 'The validation set is used for hyperparameter tuning and model selection during development, without touching the test set.'
    },
    {
      id: 'split2',
      question: 'For time-series prediction, how should you split your data?',
      options: [
        'Randomly shuffle and split',
        'Use stratified sampling',
        'Split chronologically with earlier data for training',
        'Use k-fold cross-validation with random folds'
      ],
      correctAnswer: 2,
      explanation: 'Time-series data must be split chronologically to simulate real prediction scenarios where you predict future from past.'
    }
  ]
};
