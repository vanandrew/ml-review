import { Topic } from '../../../types';

export const handlingImbalancedData: Topic = {
  id: 'handling-imbalanced-data',
  title: 'Handling Imbalanced Data',
  category: 'ml-systems',
  description: 'Techniques for dealing with class imbalance in classification',
  content: `
    <h2>Handling Imbalanced Data: When Minority Matters Most</h2>
    
    <p>Train a classifier on a dataset of 10,000 transactions—9,950 legitimate, 50 fraudulent. Your model predicts "legitimate" for every single transaction and achieves 99.5% accuracy. Congratulations? Not quite. You've missed every fraudulent transaction—the only ones that actually matter. This is the insidious challenge of imbalanced data, common in fraud detection (99.9% normal transactions), medical diagnosis (rare diseases), anomaly detection (1% outliers), and spam filtering (mostly legitimate emails).</p>

    <p>Class imbalance creates multiple problems. Models optimize overall accuracy, so predicting the majority class minimizes loss. Gradient descent amplifies this—the majority class dominates batch gradients, pushing the model toward always predicting "normal." The minority class, often the target of interest, gets ignored. Standard training produces models with excellent overall accuracy but terrible recall on the class that matters.</p>

    <h3>The Accuracy Trap: Why Traditional Metrics Fail</h3>

    <p>Accuracy—(TP+TN)/(TP+TN+FP+FN)—is  the wrong metric for imbalanced data. With 99:1 imbalance, predicting always majority gives 99% accuracy while providing zero value. You need metrics focused on minority class performance.</p>

    <p><strong>Precision</strong> measures "of predicted positives, how many are actually positive" (TP/(TP+FP)). High precision means few false alarms—when the model predicts fraud, it's usually right. <strong>Recall</strong> (sensitivity) measures "of actual positives, how many did we find" (TP/(TP+FN)). High recall means catching most fraud cases, even if it triggers some false alarms.</p>

    <p>These metrics trade off. Predicting positive more liberally increases recall but decreases precision (more false positives). Predicting conservatively increases precision but decreases recall (miss true positives). <strong>F1-score</strong>—the harmonic mean 2×(precision×recall)/(precision+recall)—balances both. For applications where recall matters more (cancer screening, missing cases is worse), use <strong>F-beta score</strong> with β>1 to weight recall higher.</p>

    <p><strong>AUC-ROC</strong> (Area Under the Receiver Operating Characteristic curve) plots true positive rate vs false positive rate at various thresholds—measures overall discriminative ability. However, for severe imbalance, <strong>AUC-PR</strong> (Precision-Recall curve) is more informative since it focuses on minority class performance. The confusion matrix shows all error types—use it to understand where the model fails.</p>

    <h3>Resampling: Rebalancing the Training Distribution</h3>

    <h4>Oversampling: Amplifying Minority Voices</h4>

    <p><strong>Random oversampling</strong> simply duplicates minority samples until classes balance. Simple but risky—exact copies lead to overfitting. The model memorizes specific instances rather than learning patterns.</p>

    <p><strong>SMOTE (Synthetic Minority Over-sampling Technique)</strong> solves this by creating synthetic samples. For each minority sample, find its k-nearest minority neighbors. Generate new samples along the line segments connecting neighbors: new_sample = x + λ×(neighbor - x) where λ∈[0,1]. This creates diverse synthetic examples that interpolate between real samples, expanding the minority class region without exact duplication. SMOTE reduces overfitting while providing balanced training data.</p>

    <p>Variants improve SMOTE: <strong>Borderline-SMOTE</strong> focuses on minority samples near the decision boundary (where class overlap occurs), creating synthetic examples where they matter most. <strong>ADASYN (Adaptive Synthetic Sampling)</strong> adaptively generates more synthetic samples for minority instances that are harder to learn (surrounded by majority samples), focusing effort on difficult regions.</p>

    <h4>Undersampling: Removing Redundancy</h4>

    <p><strong>Random undersampling</strong> removes majority samples until balance. Fast and works with large datasets where losing data isn't critical. But discarding information risks losing important patterns.</p>

    <p>Informed undersampling is smarter. <strong>Tomek links</strong> identify pairs of opposite-class nearest neighbors (ambiguous boundary points) and remove the majority sample, cleaning the boundary. <strong>Edited Nearest Neighbors (ENN)</strong> removes samples misclassified by k-NN, eliminating noise and overlapping samples. <strong>NearMiss</strong> selects majority samples closest to minority samples, retaining boundary information while reducing majority class.</p>

    <h4>Combined Strategies: Best of Both</h4>

    <p><strong>SMOTE + Tomek</strong> first oversamples minority (SMOTE), then removes noisy boundary points (Tomek), creating balanced, clean decision boundaries. <strong>SMOTE + ENN</strong> similarly oversamples then removes misclassified samples. Combined methods leverage oversam pling's diversity and undersampling's noise reduction.</p>

    <h3>Algorithm-Level Solutions: Making Models Imbalance-Aware</h3>

    <h4>Class Weights: Penalizing Minority Errors</h4>

    <p>Instead of resampling data, penalize minority class errors more heavily in the loss function. With 99:1 imbalance, weight minority errors 99× higher. Sklearn's class_weight='balanced' automatically computes weights as n_samples/(n_classes×n_samples_class), inversely proportional to frequency. Now a single minority misclassification costs as much as 99 majority errors—forcing the model to pay attention.</p>

    <p>Class weighting is simpler than resampling (no data modification), works with all algorithms supporting sample weights, and avoids creating synthetic data. But finding optimal weights may require tuning.</p>

    <h4>Threshold Tuning: Shifting the Decision Boundary</h4>

    <p>Most classifiers default to 0.5 probability threshold: predict positive if P(positive)>0.5. For imbalanced data, this is suboptimal. By lowering the threshold (e.g., 0.2), you predict positive more liberally—increasing recall at the cost of precision. Plot the precision-recall curve and choose the threshold optimizing your target metric (F1, F-beta, or business-specific objective).</p>

    <h4>Cost-Sensitive Learning: Business-Aligned Objectives</h4>

    <p>Different errors have different costs. In fraud detection, missing fraud (false negative) might cost $1000 while a false alarm (false positive) costs $10 in investigation. Incorporate these into the loss function: loss = C_FN×(false negatives) + C_FP×(false positives). The model learns to minimize business cost, not just classification errors.</p>

    <h3>Ensemble Methods: Power in Numbers</h3>

    <p><strong>Easy Ensemble</strong> creates multiple balanced subsets by repeatedly undersampling majority class, trains a classifier on each balanced subset, then aggregates predictions (voting/averaging). This retains majority class diversity (different samples in each subset) while ensuring balanced training.</p>

    <p><strong>Balanced Random Forest</strong> builds each tree on a balanced bootstrap sample (automatically undersampling majority for each tree). Since different trees see different majority samples, the forest sees the full majority distribution while each tree trains on balanced data. Sklearn's BalancedRandomForestClassifier implements this seamlessly.</p>

    <p><strong>Balanced Bagging</strong> combines bagging with resampling—each bootstrap sample is balanced via SMOTE or undersampling before training a weak learner, then predictions aggregate.</p>

    <h3>Anomaly Detection: When Imbalance is Extreme</h3>

    <p>With severe imbalance (<1% minority), standard classification struggles. Reframe as anomaly detection: learn what "normal" (majority) looks like, flag deviations as anomalies (minority). <strong>One-class SVM</strong> learns the boundary enclosing majority samples; points outside are anomalies. <strong>Isolation Forest</strong> isolates anomalies using random partitions—anomalies require fewer partitions. <strong>Autoencoders</strong> learn to reconstruct normal samples; high reconstruction error indicates anomalies. These methods work with tiny minority classes where standard classification fails.</p>

    <h3>Strategic Guidelines</h3>

    <p><strong>Always use stratified splits</strong>—train_test_split(stratify=y) maintains class ratios, ensuring test set represents real distribution. Use stratified k-fold cross-validation for the same reason.</p>

    <p><strong>Start simple:</strong> try class weights or threshold tuning before resampling. If insufficient, move to SMOTE or combined methods. For severe imbalance, consider anomaly detection or ensemble methods.</p>

    <p><strong>Imbalance severity guide:</strong> Moderate (1:10) → class weights, threshold tuning; High (1:100) → SMOTE, ensemble methods; Severe (1:1000+) → anomaly detection.</p>

    <p><strong>Data size matters:</strong> Large datasets can tolerate undersampling. Small datasets benefit from oversampling (preserves all information).</p>

    <p><strong>Critical pitfall—resample AFTER splitting!</strong> Resampling before train-test split causes data leakage: synthetic samples in test set are derived from training samples, overestimating performance. Always split first, resample training set only, test on original distribution.</p>

    <p><strong>Validate on original distribution:</strong> if training on synthetic data (SMOTE), ensure test set contains real samples only—evaluate how the model performs on actual data.</p>

    <p><strong>Domain knowledge trumps algorithms:</strong> understand the cost of different errors. In medical diagnosis, false negatives (missing disease) may be catastrophic while false positives (extra tests) are acceptable. Choose metrics and thresholds accordingly.</p>

    <p>Imbalanced data requires thoughtful handling—don't trust accuracy, choose appropriate metrics, resample intelligently, and always remember: in imbalanced data, the minority often holds the insight you seek.</p>
  `,
  codeExamples: [
    {
      language: 'Python',
      code: `import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
  accuracy_score, precision_score, recall_score,
  f1_score, roc_auc_score, confusion_matrix,
  classification_report
)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek
from collections import Counter

# Create imbalanced dataset
X, y = make_classification(
  n_samples=10000,
  n_features=20,
  n_informative=15,
  n_redundant=5,
  n_classes=2,
  weights=[0.95, 0.05],  # 95% class 0, 5% class 1
  random_state=42
)

print(f"Original class distribution: {Counter(y)}")
print(f"Imbalance ratio: {Counter(y)[0] / Counter(y)[1]:.1f}:1")

# Split data (stratified to maintain ratio)
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42, stratify=y
)

# === BASELINE: No handling ===

clf_baseline = LogisticRegression(max_iter=1000, random_state=42)
clf_baseline.fit(X_train, y_train)
y_pred_baseline = clf_baseline.predict(X_test)

print(f"\\n=== Baseline (No Imbalance Handling) ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_baseline):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_baseline):.3f}")
print(f"Recall: {recall_score(y_test, y_pred_baseline):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred_baseline):.3f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_baseline):.3f}")
print(f"Confusion Matrix:\\n{confusion_matrix(y_test, y_pred_baseline)}")

# === METHOD 1: Class Weights ===

clf_weighted = LogisticRegression(
  max_iter=1000,
  class_weight='balanced',  # Automatically compute weights
  random_state=42
)
clf_weighted.fit(X_train, y_train)
y_pred_weighted = clf_weighted.predict(X_test)

print(f"\\n=== Class Weights ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_weighted):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_weighted):.3f}")
print(f"Recall: {recall_score(y_test, y_pred_weighted):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred_weighted):.3f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_weighted):.3f}")

# === METHOD 2: SMOTE (Oversampling) ===

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\\n=== SMOTE ===")
print(f"After SMOTE: {Counter(y_train_smote)}")

clf_smote = LogisticRegression(max_iter=1000, random_state=42)
clf_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = clf_smote.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred_smote):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_smote):.3f}")
print(f"Recall: {recall_score(y_test, y_pred_smote):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred_smote):.3f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_smote):.3f}")

# === METHOD 3: Random Undersampling ===

rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

print(f"\\n=== Random Undersampling ===")
print(f"After undersampling: {Counter(y_train_rus)}")

clf_rus = LogisticRegression(max_iter=1000, random_state=42)
clf_rus.fit(X_train_rus, y_train_rus)
y_pred_rus = clf_rus.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred_rus):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_rus):.3f}")
print(f"Recall: {recall_score(y_test, y_pred_rus):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred_rus):.3f}")

# === METHOD 4: SMOTE + Tomek Links (Combined) ===

smt = SMOTETomek(random_state=42)
X_train_smt, y_train_smt = smt.fit_resample(X_train, y_train)

print(f"\\n=== SMOTE + Tomek Links ===")
print(f"After SMOTE+Tomek: {Counter(y_train_smt)}")

clf_smt = LogisticRegression(max_iter=1000, random_state=42)
clf_smt.fit(X_train_smt, y_train_smt)
y_pred_smt = clf_smt.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred_smt):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_smt):.3f}")
print(f"Recall: {recall_score(y_test, y_pred_smt):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred_smt):.3f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_smt):.3f}")

# === METHOD 5: Threshold Tuning ===

from sklearn.metrics import precision_recall_curve

# Get probability predictions
y_prob_baseline = clf_baseline.predict_proba(X_test)[:, 1]

# Find optimal threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_prob_baseline)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

y_pred_tuned = (y_prob_baseline >= optimal_threshold).astype(int)

print(f"\\n=== Threshold Tuning ===")
print(f"Optimal threshold: {optimal_threshold:.3f} (default is 0.5)")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tuned):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_tuned):.3f}")
print(f"Recall: {recall_score(y_test, y_pred_tuned):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred_tuned):.3f}")`,
      explanation: 'Comprehensive comparison of imbalanced data techniques: class weights, SMOTE, undersampling, combined methods, and threshold tuning.'
    },
    {
      language: 'Python',
      code: `from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create severely imbalanced dataset
X, y = make_classification(
  n_samples=10000,
  n_features=20,
  n_informative=15,
  n_classes=2,
  weights=[0.99, 0.01],  # 99% class 0, 1% class 1
  random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Severe imbalance - Class distribution: {np.bincount(y_train)}")
print(f"Ratio: {np.bincount(y_train)[0] / np.bincount(y_train)[1]:.0f}:1\\n")

# === Standard Random Forest ===

rf_standard = RandomForestClassifier(n_estimators=100, random_state=42)
rf_standard.fit(X_train, y_train)
y_pred_standard = rf_standard.predict(X_test)

print("=== Standard Random Forest ===")
print(classification_report(y_test, y_pred_standard, target_names=['Majority', 'Minority']))

# === Balanced Random Forest ===

brf = BalancedRandomForestClassifier(
  n_estimators=100,
  sampling_strategy='all',  # Undersample all classes
  replacement=True,
  random_state=42
)
brf.fit(X_train, y_train)
y_pred_brf = brf.predict(X_test)

print("\\n=== Balanced Random Forest ===")
print(classification_report(y_test, y_pred_brf, target_names=['Majority', 'Minority']))

# === Easy Ensemble ===

eec = EasyEnsembleClassifier(
  n_estimators=10,
  random_state=42
)
eec.fit(X_train, y_train)
y_pred_eec = eec.predict(X_test)

print("\\n=== Easy Ensemble ===")
print(classification_report(y_test, y_pred_eec, target_names=['Majority', 'Minority']))

# === Custom Cost-Sensitive Approach ===

# Define custom sample weights (penalize minority class errors more)
sample_weights = np.ones(len(y_train))
sample_weights[y_train == 1] = 50  # 50x weight for minority class

rf_weighted = RandomForestClassifier(n_estimators=100, random_state=42)
rf_weighted.fit(X_train, y_train, sample_weight=sample_weights)
y_pred_weighted = rf_weighted.predict(X_test)

print("\\n=== Random Forest with Sample Weights ===")
print(classification_report(y_test, y_pred_weighted, target_names=['Majority', 'Minority']))

# === Anomaly Detection Approach ===

from sklearn.ensemble import IsolationForest

# Train only on majority class
X_train_majority = X_train[y_train == 0]

iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(X_train_majority)

# Predict: -1 for anomaly (minority), 1 for normal (majority)
y_pred_iso = iso_forest.predict(X_test)
y_pred_iso_binary = (y_pred_iso == -1).astype(int)

print("\\n=== Isolation Forest (Anomaly Detection) ===")
print(classification_report(y_test, y_pred_iso_binary, target_names=['Majority', 'Minority']))

# === Compare All Methods ===

from sklearn.metrics import roc_auc_score, f1_score

methods = {
  'Standard RF': y_pred_standard,
  'Balanced RF': y_pred_brf,
  'Easy Ensemble': y_pred_eec,
  'Weighted RF': y_pred_weighted,
  'Isolation Forest': y_pred_iso_binary
}

print("\\n=== Summary Comparison ===")
print(f"{'Method':<20} {'F1-Score':<12} {'AUC-ROC':<12} {'Minority Recall'}")
print("-" * 60)

for name, y_pred in methods.items():
  f1 = f1_score(y_test, y_pred)
  auc = roc_auc_score(y_test, y_pred)
  minority_recall = recall_score(y_test, y_pred)
  print(f"{name:<20} {f1:<12.3f} {auc:<12.3f} {minority_recall:.3f}")`,
      explanation: 'Ensemble methods and anomaly detection for severe class imbalance, comparing multiple advanced techniques.'
    }
  ],
  interviewQuestions: [
    {
      question: 'Why is accuracy a poor metric for imbalanced data?',
      answer: `Accuracy can be misleadingly high on imbalanced data by simply predicting the majority class. For 99% majority class, a model predicting always majority achieves 99% accuracy but provides no value. Better metrics: precision, recall, F1-score, AUC-ROC, AUC-PR focus on minority class performance and provide more meaningful evaluation of model effectiveness.`
    },
    {
      question: 'Explain the difference between SMOTE and random oversampling.',
      answer: `Random oversampling duplicates existing minority class samples, potentially causing overfitting to specific instances. SMOTE (Synthetic Minority Oversampling Technique) creates synthetic samples by interpolating between existing minority samples using k-nearest neighbors, generating more diverse examples and reducing overfitting risk while maintaining class distribution patterns.`
    },
    {
      question: 'What is the trade-off between precision and recall?',
      answer: `Precision measures "of predicted positives, how many are actually positive" (reduces false positives). Recall measures "of actual positives, how many were found" (reduces false negatives). Higher precision threshold reduces false alarms but may miss true positives. Higher recall catches more positives but increases false alarms. F1-score balances both; choose based on business cost of false positives vs false negatives.`
    },
    {
      question: 'When would you use undersampling vs oversampling?',
      answer: `Use oversampling when: (1) Limited data overall, (2) Computational resources allow, (3) Risk of losing important information. Use undersampling when: (1) Very large datasets, (2) Computational constraints, (3) Majority class has redundant samples. Consider combined approaches (SMOTEENN) or ensemble methods. Always evaluate impact on validation set performance.`
    },
    {
      question: 'How does class weighting work in loss functions?',
      answer: `Class weighting assigns higher weights to minority class samples in loss functions, making misclassification of minority samples more costly. Implementation: multiply loss by class weights (inverse frequency, balanced, or custom). Effect: model pays more attention to minority class during training. Simpler than resampling but may require hyperparameter tuning to find optimal weights.`
    },
    {
      question: 'What is threshold tuning and when should you use it?',
      answer: `Threshold tuning adjusts the decision boundary (default 0.5) to optimize for specific metrics like F1-score or to balance precision/recall based on business needs. Use when: (1) Default threshold doesn't align with business objectives, (2) Imbalanced data, (3) Different costs for false positives vs false negatives. Find optimal threshold using validation data and target metric.`
    }
  ],
  quizQuestions: [
    {
      id: 'imb1',
      question: 'Why is accuracy misleading for imbalanced data?',
      options: ['It\'s not misleading', 'High accuracy by always predicting majority class', 'Accuracy is always wrong', 'It\'s too complex'],
      correctAnswer: 1,
      explanation: 'With 99:1 imbalance, a model predicting always majority achieves 99% accuracy but completely fails on the minority class. Better metrics: precision, recall, F1, AUC.'
    },
    {
      id: 'imb2',
      question: 'What does SMOTE do?',
      options: ['Remove majority samples', 'Create synthetic minority samples', 'Adjust class weights', 'Change threshold'],
      correctAnswer: 1,
      explanation: 'SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic minority class samples by interpolating between existing minority samples, rather than just duplicating them.'
    },
    {
      id: 'imb3',
      question: 'When should you apply resampling?',
      options: ['Before train-test split', 'After train-test split, only on train', 'On both train and test', 'Doesn\'t matter'],
      correctAnswer: 1,
      explanation: 'Resample only the training set after splitting to avoid data leakage. Test set should remain unchanged to evaluate on the true distribution.'
    }
  ]
};
