import { Topic } from '../../../types';

export const evaluationMetrics: Topic = {
  id: 'evaluation-metrics',
  title: 'Evaluation Metrics',
  category: 'foundations',
  description: 'Understanding and selecting appropriate metrics for different ML tasks',
  content: `
    <h2>Evaluation Metrics for Machine Learning</h2>
    <p>Choosing the right evaluation metric is one of the most critical decisions in machine learning, as it defines what "success" means for your model and guides the entire development process. A model optimized for the wrong metric can achieve impressive numbers while failing to solve the actual business problem. Evaluation metrics must align with real-world objectives, account for data characteristics like class imbalance, and reflect the relative costs of different types of errors.</p>

    <p>Different problem types (classification vs. regression), different data distributions (balanced vs. imbalanced), and different business contexts (medical diagnosis vs. movie recommendations) demand different metrics. Understanding the nuances of each metric—what it measures, what it ignores, when it's appropriate, and when it misleads—is essential for building models that deliver real value.</p>

    <div class="info-box info-box-green">
      <h4>📊 Metric Selection Cheat Sheet</h4>
      <p><strong>Classification Tasks:</strong></p>
      <table>
        <tr>
          <th>Scenario</th>
          <th>Primary Metric</th>
          <th>Secondary Metrics</th>
        </tr>
        <tr>
          <td>Balanced classes</td>
          <td><strong>Accuracy</strong>, F1</td>
          <td>Precision, Recall, ROC-AUC</td>
        </tr>
        <tr>
          <td>Imbalanced (e.g., fraud, rare disease)</td>
          <td><strong>PR-AUC</strong>, F1</td>
          <td>Precision, Recall separately</td>
        </tr>
        <tr>
          <td>False positives very costly</td>
          <td><strong>Precision</strong></td>
          <td>F1, Specificity</td>
        </tr>
        <tr>
          <td>False negatives very costly</td>
          <td><strong>Recall</strong></td>
          <td>F2, Sensitivity</td>
        </tr>
        <tr>
          <td>Need probability estimates</td>
          <td><strong>Log Loss</strong></td>
          <td>Brier Score, Calibration</td>
        </tr>
      </table>
      <p><strong>Regression Tasks:</strong></p>
      <table>
        <tr>
          <th>Scenario</th>
          <th>Primary Metric</th>
          <th>Why</th>
        </tr>
        <tr>
          <td>General regression</td>
          <td><strong>RMSE</strong> + R²</td>
          <td>Standard, interpretable</td>
        </tr>
        <tr>
          <td>Data with outliers</td>
          <td><strong>MAE</strong></td>
          <td>Robust to outliers</td>
        </tr>
        <tr>
          <td>Large errors very bad</td>
          <td><strong>RMSE</strong></td>
          <td>Penalizes large errors heavily</td>
        </tr>
        <tr>
          <td>Relative performance</td>
          <td><strong>R²</strong></td>
          <td>Variance explained (unitless)</td>
        </tr>
      </table>
      <p><strong>⚠️ Warning:</strong> Never use accuracy alone for imbalanced data! | Always track multiple metrics | Align metrics with business objectives</p>
    </div>

    <h3>Classification Metrics: Measuring Categorical Predictions</h3>
    <p>Classification tasks predict discrete categories or classes. Evaluation metrics for classification derive from the <strong>confusion matrix</strong>, which tabulates predictions against ground truth.</p>

    <h4>The Confusion Matrix: Foundation of Classification Metrics</h4>
    <p>For binary classification (positive and negative classes), the confusion matrix has four cells:</p>
    <ul>
      <li><strong>True Positives (TP):</strong> Correctly predicted positive examples (model says positive, reality is positive)</li>
      <li><strong>True Negatives (TN):</strong> Correctly predicted negative examples (model says negative, reality is negative)</li>
      <li><strong>False Positives (FP):</strong> Incorrectly predicted positive (model says positive, reality is negative)—also called Type I error or "false alarm"</li>
      <li><strong>False Negatives (FN):</strong> Incorrectly predicted negative (model says negative, reality is positive)—also called Type II error or "missed detection"</li>
    </ul>

    <p>From these four values, all classification metrics are derived. The key insight is that different metrics emphasize different cells of the confusion matrix, reflecting different priorities about which errors matter most.</p>

    <h4>Accuracy: The Simplest But Often Misleading Metric</h4>
    <p><strong>Accuracy = (TP + TN) / (TP + TN + FP + FN)</strong></p>
    <p>Accuracy measures the fraction of predictions that are correct. It's intuitive, easy to explain to non-technical stakeholders, and works well when classes are balanced and errors have equal cost. However, accuracy is notoriously misleading for imbalanced datasets.</p>

    <p><strong>The imbalance problem:</strong> Suppose you're detecting fraud in credit card transactions, where 99.9% of transactions are legitimate. A naive model that classifies every transaction as "not fraud" achieves 99.9% accuracy while being completely useless—it catches zero fraud cases. Accuracy hides this failure because the denominator is dominated by the abundant negative class. In imbalanced settings, a model can have high accuracy by simply predicting the majority class for everything.</p>

    <p><strong>When to use accuracy:</strong> Balanced datasets where all classes are equally important and errors have roughly equal cost. Examples: classifying balanced datasets of cat/dog images, predicting coin flips, or multi-class problems with equal representation. Avoid accuracy for imbalanced data, rare event detection, or when different error types have different costs.</p>

    <h4>Precision: Minimizing False Alarms</h4>
    <p><strong>Precision = TP / (TP + FP)</strong></p>
    <p>Precision answers the question: "Of all the examples my model labeled as positive, what fraction actually were positive?" It measures how "precise" or "pure" your positive predictions are. High precision means few false alarms—when your model says positive, it's usually right.</p>

    <p><strong>When to optimize for precision:</strong> Situations where false positives are expensive or harmful. Email spam filtering is the classic example: marking a legitimate email as spam (false positive) is very bad—users might miss important messages from clients, jobs, or family. Missing some spam (false negative) is annoying but acceptable. Similarly, in content moderation, false positives (censoring legitimate speech) may have legal or ethical consequences. Other precision-critical domains include medical treatment recommendations (giving wrong treatment is worse than conservative monitoring), legal document review (flagging wrong documents wastes expensive lawyer time), and fraud alerts sent to customers (too many false alarms train customers to ignore real alerts).</p>

    <p><strong>The trade-off:</strong> You can achieve perfect precision = 1.0 by being extremely conservative—only predicting positive when you're absolutely certain. But this will miss many true positives, giving low recall. Precision alone doesn't tell you whether you're catching most positive cases.</p>

    <h4>Recall (Sensitivity, True Positive Rate): Minimizing Missed Cases</h4>
    <p><strong>Recall = TP / (TP + FN)</strong></p>
    <p>Recall answers: "Of all the actual positive examples, what fraction did my model correctly identify?" It measures how "complete" your positive predictions are. High recall means you're catching most of the positive cases, with few slipping through.</p>

    <p><strong>When to optimize for recall:</strong> Situations where false negatives are catastrophic. Medical screening tests are the paradigm: missing a cancer diagnosis (false negative) could be fatal, while a false positive just means an unnecessary follow-up test. You want high recall to catch all potential cases, accepting some false alarms that get filtered by confirmatory testing. Airport security screening similarly prioritizes recall—better to flag innocent passengers for additional screening than miss a threat. Other recall-critical applications include fraud detection (missing fraud causes direct financial loss), safety monitoring (missing equipment failures causes accidents), and missing children alerts (false alarms are acceptable when safety is at risk).</p>

    <p><strong>The trade-off:</strong> You can achieve perfect recall = 1.0 by predicting everything as positive—you'll catch all true positives but also flag all negatives as false positives. Recall alone doesn't tell you how many false alarms you're generating.</p>

    <h4>F1 Score: Balancing Precision and Recall</h4>
    <p><strong>F1 = 2 × (Precision × Recall) / (Precision + Recall)</strong></p>
    <p>The F1 score is the harmonic mean of precision and recall. The harmonic mean (unlike arithmetic mean) heavily penalizes low values—if either precision or recall is very low, F1 will be low. This makes F1 a balanced metric that requires both precision and recall to be reasonably high.</p>

    <p><strong>Why harmonic mean?</strong> Suppose precision = 1.0 (perfect) and recall = 0.01 (terrible). The arithmetic mean would be 0.505 (appearing decent), but the harmonic mean (F1) is 0.0198 (correctly reflecting the terrible recall). The harmonic mean is more conservative and appropriate when you need both metrics to be good.</p>

    <p><strong>Generalizations:</strong> The F1 score is a special case of the F$\\beta$ score: $F_\\beta = (1 + \\beta^2) \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\beta^2 \\times \\text{Precision} + \\text{Recall}}$. With $\\beta = 1$, you get F1 (equal weight). $\\beta = 2$ (F2 score) weighs recall twice as much as precision—useful when recall is more important. $\\beta = 0.5$ weighs precision higher. In practice, F1 is the most common choice for imbalanced classification.</p>

    <p><strong>Limitations:</strong> F1 requires choosing a classification threshold. It also doesn't account for true negatives at all—it focuses purely on positive class performance. For severely imbalanced data, this is actually a feature, not a bug.</p>

    <h4>ROC Curve and AUC: Threshold-Independent Evaluation</h4>
    <p>The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate (TPR = Recall = TP/(TP+FN)) against the False Positive Rate (FPR = FP/(FP+TN)) at all possible classification thresholds. Most classifiers output probabilities or scores; by varying the threshold from 0 to 1, you get different TPR/FPR trade-offs.</p>

    <p>The Area Under the ROC Curve (AUC-ROC or simply AUC) summarizes this curve into a single number between 0 and 1. AUC = 0.5 means random guessing (the ROC curve is the diagonal line). AUC = 1.0 means perfect separation (there exists a threshold that achieves 100% TPR and 0% FPR). AUC can also be interpreted as the probability that the model ranks a randomly chosen positive example higher than a randomly chosen negative example.</p>

    <p><strong>Advantages:</strong> AUC is threshold-independent—you don't need to pick a classification threshold. It measures the model's ability to discriminate between classes across all operating points. It's useful for comparing models when the optimal threshold isn't known or may change depending on deployment context.</p>

    <p><strong>The imbalance problem:</strong> ROC-AUC can be misleadingly optimistic for highly imbalanced datasets. FPR uses true negatives in the denominator (FPR = FP/(FP+TN)), and with many negatives, FPR stays low even with substantial false positives. For example, with 99% negative class, 100 false positives and 9,900 true negatives gives FPR = 100/10,000 = 1%, appearing excellent on the ROC curve. But if there are only 50 true positives, precision = 50/(50+100) = 33%, revealing poor performance.</p>

    <p><strong>When to use ROC-AUC:</strong> Balanced datasets where you care about both classes equally, model comparison when the operating threshold is flexible, or domains like medical diagnostics where you need to balance sensitivity (TPR) and specificity (1 - FPR). Avoid for highly imbalanced data (use PR-AUC instead).</p>

    <h4>Precision-Recall Curve and PR-AUC: Better for Imbalanced Data</h4>
    <p>The Precision-Recall (PR) curve plots precision against recall at all classification thresholds. PR-AUC is the area under this curve. Unlike ROC curves, PR curves focus entirely on positive class performance and don't include true negatives in their calculation, making them more informative for imbalanced datasets.</p>

    <p><strong>Why it's better for imbalance:</strong> With 99% negative class and 1% positive class, a random classifier achieves AUC-ROC = 0.5 but PR-AUC ≈ 0.01 (the positive class frequency). PR-AUC more accurately reflects that random guessing is terrible on imbalanced data. Precision has false positives in the denominator without the buffering effect of many true negatives, so it's more sensitive to classification quality on the minority class.</p>

    <p><strong>When to use PR-AUC:</strong> Imbalanced datasets (especially minority class <10%), rare event detection (fraud, disease, equipment failure), information retrieval and recommendation systems, or any scenario where you primarily care about positive class performance. Fraud detection, medical screening for rare diseases, and document retrieval should always use PR-AUC over ROC-AUC.</p>

    <h3>Regression Metrics: Measuring Continuous Predictions</h3>
    <p>Regression tasks predict continuous numerical values. Metrics measure the difference (error or residual) between predicted and actual values.</p>

    <h4>Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)</h4>
    <p><strong>$\\text{MSE} = \\frac{1}{n} \\sum (y_i - \\hat{y}_i)^2$</strong></p>
    <p><strong>$\\text{RMSE} = \\sqrt{\\text{MSE}}$</strong></p>

    <p>MSE is the average of squared errors. Squaring errors ensures they're positive and gives quadratic penalty—an error of 10 contributes 100 to the sum while ten errors of 1 contribute only 10 total. This makes MSE very sensitive to large errors and outliers. RMSE takes the square root to return to the original units of the target variable, making it more interpretable.</p>

    <p><strong>Why squared errors?</strong> MSE corresponds to Gaussian likelihood under certain assumptions and has nice mathematical properties (differentiable, convex for linear models). It's the loss function optimized by ordinary least squares regression. Squaring heavily penalizes outliers, which can be desirable (large errors are worse than proportionally worse) or problematic (outliers dominate the metric).</p>

    <p><strong>When to use RMSE:</strong> Standard choice for regression, especially when large errors are particularly bad. Predicting house prices, stock values, or engineering quantities where being off by $100k is much worse than being off by $10k. RMSE is interpretable ("on average, predictions are off by $X") and widely used, making it easy to communicate and compare to baselines. Avoid when data has outliers that you don't want to dominate the metric.</p>

    <h4>Mean Absolute Error (MAE): Robust to Outliers</h4>
    <p><strong>$\\text{MAE} = \\frac{1}{n} \\sum |y_i - \\hat{y}_i|$</strong></p>
    <p>MAE is the average of absolute errors. Unlike MSE, it treats all errors linearly—an error of 10 contributes 10 to the sum, same as ten errors of 1. This makes MAE much more robust to outliers and easier to interpret: "on average, predictions are off by X units."</p>

    <p><strong>RMSE vs MAE:</strong> RMSE will always be ≥ MAE, with equality only when all errors are identical. A large gap between RMSE and MAE indicates some predictions have very large errors (outliers or occasional large mistakes). For example, RMSE = $50k and MAE = $20k suggests most predictions are off by ~$20k but a few are off by much more, pulling RMSE up. If RMSE ≈ MAE, errors are relatively uniform.</p>

    <p><strong>When to use MAE:</strong> When outliers in your data are due to measurement errors or rare anomalies that shouldn't dominate your metric. Predicting delivery times (occasional delays shouldn't dominate), demand forecasting with occasional spikes, or any domain where you want to measure typical error rather than worst-case error. MAE is also preferred when your loss function is truly linear (economic cost proportional to error magnitude, not squared).</p>

    <h4>R² Score (Coefficient of Determination): Variance Explained</h4>
    <p><strong>$R^2 = 1 - \\frac{SS_{res}}{SS_{tot}}$</strong></p>
    <p>Where $SS_{res} = \\sum (y_i - \\hat{y}_i)^2$ (residual sum of squares) and $SS_{tot} = \\sum (y_i - \\bar{y})^2$ (total sum of squares, variance around the mean).</p>

    <p>R² measures the proportion of variance in the target variable explained by the model. R² = 1 means perfect predictions (SS_res = 0). R² = 0 means your model performs no better than simply predicting the mean for every sample. R² < 0 means your model performs worse than the mean baseline—it's making predictions that systematically increase error.</p>

    <p><strong>Interpretation:</strong> R² = 0.85 means your model explains 85% of the variance in the target variable; the remaining 15% is unexplained (noise, missing features, or irreducible error). Unlike RMSE/MAE, R² is unitless and ranges from -∞ to 1, making it comparable across problems (though not directly—R² on easy vs. hard problems aren't comparable).</p>

    <p><strong>When R² can be negative:</strong> If your model is very poor (severe overfitting to training data that doesn't generalize, completely wrong model specification, or testing on a different distribution), SS_res can exceed SS_tot, yielding negative R². This indicates fundamental model failure—the simplest baseline (predicting the mean) is better than your complex model.</p>

    <p><strong>Limitations:</strong> R² can be artificially inflated by adding more features (even irrelevant ones), leading to adjusted R² which penalizes model complexity. R² also doesn't indicate whether predictions are biased or whether the model satisfies assumptions. High R² doesn't guarantee the model is useful—you might have excellent R² on training data but terrible generalization.</p>

    <p><strong>When to use R²:</strong> Explaining model performance to non-technical audiences ("the model explains 80% of price variation"), comparing models on the same dataset, or understanding how much variance your features capture. Use alongside RMSE/MAE to get a complete picture—R² tells you relative performance vs. baseline, RMSE/MAE tells you absolute error in meaningful units.</p>

    <h3>Metric Selection Guidelines</h3>
    <p>Choosing the right metric depends on your problem type, data characteristics, and business context:</p>

    <ul>
      <li><strong>Balanced binary classification:</strong> Accuracy, F1 score, or AUC-ROC. These work well when both classes are roughly equal in size and importance.</li>
      <li><strong>Imbalanced classification:</strong> F1 score, PR-AUC, or class-weighted metrics. Focus on positive class performance and avoid accuracy.</li>
      <li><strong>High false positive cost:</strong> Precision (spam filtering, content moderation, medical treatment decisions).</li>
      <li><strong>High false negative cost:</strong> Recall (cancer detection, fraud detection, safety monitoring).</li>
      <li><strong>Need balance with imbalance:</strong> F1 score or F$\\beta$ score with appropriate $\\beta$.</li>
      <li><strong>Ranking or probability quality:</strong> ROC-AUC (if balanced), log loss/cross-entropy for well-calibrated probabilities.</li>
      <li><strong>Multi-class classification:</strong> Macro-averaged F1 (average F1 per class) if classes are important equally, weighted F1 if class sizes vary.</li>
      <li><strong>Regression (general):</strong> RMSE and R² together. RMSE for absolute error in target units, R² for relative performance.</li>
      <li><strong>Regression with outliers:</strong> MAE or Huber loss (robust to outliers).</li>
      <li><strong>Regression where large errors are catastrophic:</strong> RMSE or custom metrics with even higher penalties for large errors.</li>
    </ul>

    <h3>Advanced Considerations</h3>
    <p><strong>Business alignment:</strong> The best metric aligns with business objectives. If false alarms cost $100 each and missed detections cost $10,000 each, your metric should reflect this asymmetry (perhaps use weighted precision/recall or a custom cost-sensitive metric).</p>

    <p><strong>Multiple metrics:</strong> Don't rely on a single metric. Use primary metrics for optimization and secondary metrics for monitoring. For example, optimize for F1 but monitor precision and recall separately to understand the trade-off. Track training metrics to detect overfitting.</p>

    <p><strong>Threshold selection:</strong> For binary classification, the default 0.5 threshold is arbitrary. Use precision-recall or ROC curves to find the optimal threshold for your cost structure. In production, you might use different thresholds for different users or contexts.</p>

    <p><strong>Stratified evaluation:</strong> Don't just report overall metrics—break down performance by subgroups (demographics, difficulty level, time period) to find where your model fails and ensure fairness.</p>

    <p><strong>Calibration:</strong> For probability-outputting models, check calibration (are predicted probabilities accurate?). A model might have good discrimination (high AUC) but poor calibration (predicted 90% confidence doesn't mean 90% accuracy). Use calibration plots and Brier score to assess this.</p>
  `,
  codeExamples: [
    {
      language: 'Python',
      code: `from sklearn.metrics import (
  accuracy_score, precision_score, recall_score, f1_score,
  roc_auc_score, average_precision_score, confusion_matrix,
  classification_report
)
import numpy as np

# Simulate predictions for imbalanced dataset (5% positive class)
np.random.seed(42)
y_true = np.random.choice([0, 1], size=1000, p=[0.95, 0.05])
y_pred = np.random.choice([0, 1], size=1000, p=[0.90, 0.10])
y_proba = np.random.rand(1000)  # Predicted probabilities

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
print(f"TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}\\n")

# Basic metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}\\n")

# Threshold-independent metrics
roc_auc = roc_auc_score(y_true, y_proba)
pr_auc = average_precision_score(y_true, y_proba)

print(f"ROC-AUC: {roc_auc:.3f}")
print(f"PR-AUC:  {pr_auc:.3f}\\n")

# Comprehensive classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

# Imbalanced dataset example
print("\\n--- Imbalanced Dataset Analysis ---")
# Model 1: Always predicts negative
y_pred_baseline = np.zeros(1000)
print(f"Baseline (all negative) - Accuracy: {accuracy_score(y_true, y_pred_baseline):.3f}")
print(f"Baseline F1: {f1_score(y_true, y_pred_baseline, zero_division=0):.3f}")`,
      explanation: 'Comprehensive classification metrics evaluation showing how accuracy can be misleading on imbalanced datasets. F1 score and PR-AUC provide better insight into model performance on minority class.'
    },
    {
      language: 'Python',
      code: `from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Generate sample regression data with outliers
np.random.seed(42)
y_true = np.random.randn(100) * 10 + 50
y_pred = y_true + np.random.randn(100) * 5

# Add some outliers
y_true[95:] = [100, 105, 110, 95, 102]
y_pred[95:] = [60, 65, 58, 62, 63]  # Model fails on outliers

# Calculate regression metrics
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("Regression Metrics:")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")
print(f"R²:   {r2:.3f}\\n")

# Compare metrics with and without outliers
y_true_clean = y_true[:95]
y_pred_clean = y_pred[:95]

rmse_clean = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
mae_clean = mean_absolute_error(y_true_clean, y_pred_clean)
r2_clean = r2_score(y_true_clean, y_pred_clean)

print("Without Outliers:")
print(f"RMSE: {rmse_clean:.2f} (vs {rmse:.2f} with outliers)")
print(f"MAE:  {mae_clean:.2f} (vs {mae:.2f} with outliers)")
print(f"R²:   {r2_clean:.3f} (vs {r2:.3f} with outliers)\\n")

# RMSE vs MAE sensitivity
print("Impact of outliers:")
print(f"RMSE increased by: {((rmse - rmse_clean) / rmse_clean * 100):.1f}%")
print(f"MAE increased by:  {((mae - mae_clean) / mae_clean * 100):.1f}%")
print("\\nRMSE is more sensitive to outliers due to squaring errors!")`,
      explanation: 'Compares regression metrics (MSE, RMSE, MAE, R²) and demonstrates how RMSE is more sensitive to outliers than MAE. Essential for choosing appropriate metrics based on data characteristics.'
    }
  ],
  interviewQuestions: [
    {
      question: 'Why is accuracy a poor metric for imbalanced datasets? What metrics should you use instead?',
      answer: 'Accuracy is misleading for imbalanced datasets because a naive model that always predicts the majority class can achieve high accuracy without learning anything useful. For example, in fraud detection where 99% of transactions are legitimate, a model that classifies everything as "not fraud" achieves 99% accuracy while being completely useless—it catches zero fraud cases. Accuracy treats all errors equally, but in imbalanced scenarios, errors on the minority class are typically much more costly and important than errors on the majority class.\n\nFor imbalanced classification, use metrics that account for both classes properly. Precision measures what fraction of positive predictions are correct (TP / (TP + FP)), crucial when false positives are costly. Recall (also called sensitivity or TPR) measures what fraction of actual positives you catch (TP / (TP + FN)), critical when missing positive cases is dangerous. F1-score is the harmonic mean of precision and recall, providing a single metric that balances both. For severe imbalance, precision-recall curves and PR-AUC (area under precision-recall curve) are more informative than ROC-AUC because they focus on performance on the minority class.\n\nAlternatively, use metrics designed for imbalance. Balanced accuracy averages recall across classes, preventing majority class dominance. Cohen\'s Kappa measures agreement above chance, accounting for class imbalance. Matthews Correlation Coefficient (MCC) is a balanced metric that works well for imbalanced datasets, returning a value between -1 and 1. For multi-class imbalance, use macro-averaged metrics (compute metric per class, then average) rather than micro-averaged (aggregate all classes\' true positives, false positives, etc. first). The key is choosing metrics aligned with your business objective: if catching all fraud is critical, optimize recall; if false alarms are expensive, optimize precision; for balance, use F1 or F2 (weighs recall higher) scores.'
    },
    {
      question: 'When would you optimize for precision vs recall? Provide real-world examples.',
      answer: 'Optimize for precision when false positives are expensive or harmful, and you can tolerate missing some true positives. Email spam filtering is a classic example: marking a legitimate email as spam (false positive) is very bad—users might miss important messages from clients, jobs, or family. Missing some spam (false negative) is annoying but acceptable. You want high precision so that emails marked as spam are almost certainly spam. Similarly, in content moderation for removing illegal content, you want high precision to avoid accidentally censoring legitimate speech, even if it means some bad content slips through initially.\n\nOther precision-focused scenarios include: medical treatment recommendations (giving wrong treatment is worse than suggesting conservative monitoring), legal document review (marking wrong documents as relevant wastes expensive lawyer time), product recommendations (suggesting irrelevant products annoys users and reduces trust), and fraud alerts sent to customers (too many false alarms train customers to ignore real fraud warnings). In these cases, you\'re optimizing to be "right when you speak up," accepting that you might miss some cases.\n\nOptimize for recall when false negatives are very costly and false positives are manageable. Medical screening tests are the paradigm example: missing a cancer diagnosis (false negative) could be fatal, while a false positive just means an unnecessary follow-up test. You want high recall to catch all potential cases, accepting some false alarms that get filtered out by confirmatory testing. Airport security screening similarly prioritizes recall—better to flag innocent passengers for additional screening than to miss a threat. Other recall-focused applications include: fraud detection (missing fraud causes direct financial loss), safety monitoring systems (missing a critical equipment failure could cause accidents), missing children alerts (false alarms are acceptable when a child\'s safety is at risk), and initial resume screening (false positives get filtered in interviews, but missing a great candidate is permanent loss). The general principle: optimize for recall when the cost of missing a positive case is much higher than the cost of false alarms.'
    },
    {
      question: 'What is the difference between ROC-AUC and PR-AUC? When is each more appropriate?',
      answer: 'ROC (Receiver Operating Characteristic) curve plots True Positive Rate (TPR = recall) vs False Positive Rate (FPR) at various classification thresholds. AUC-ROC is the area under this curve, representing the probability that the model ranks a random positive sample higher than a random negative sample. PR (Precision-Recall) curve plots precision vs recall at various thresholds. AUC-PR is the area under this curve. Both measure classifier quality across all possible thresholds, but they emphasize different aspects of performance.\n\nThe key difference emerges with class imbalance. ROC-AUC can be misleadingly optimistic for imbalanced datasets because FPR uses true negatives in the denominator, and with many negatives, FPR stays low even with substantial false positives. For example, with 99% negative class, 100 false positives and 9900 true negatives gives FPR = 100/10000 = 1%, appearing excellent. PR-AUC is more sensitive to imbalance because precision has false positives in the denominator without the buffering effect of many true negatives. The same 100 false positives with 50 true positives gives precision = 50/150 = 33%, clearly showing the problem.\n\nUse ROC-AUC when classes are roughly balanced and you care about both positive and negative classes equally. It\'s also standard in domains like medical diagnostics where you need to balance sensitivity (catching disease) and specificity (not alarming healthy patients). Use PR-AUC for imbalanced datasets (especially minority class <10%) or when you primarily care about performance on the positive class. Fraud detection, rare disease screening, information retrieval, and anomaly detection should use PR-AUC. A perfect classifier has AUC-ROC = 1.0 and AUC-PR = 1.0, but random guessing gives AUC-ROC = 0.5 regardless of imbalance while AUC-PR equals the positive class frequency (e.g., 0.01 for 1% positive class), making PR-AUC a higher bar. In practice, report both when possible to give a complete picture of performance.'
    },
    {
      question: 'How do you interpret an R² score of -0.5 in regression?',
      answer: 'An R² of -0.5 means your model performs worse than simply predicting the mean of the target variable for every sample—it\'s making predictions that are systematically worse than the simplest baseline. R² is defined as 1 - (SS_res / SS_tot) where SS_res is the sum of squared residuals (prediction errors) and SS_tot is total sum of squares (variance around the mean). When SS_res > SS_tot, R² becomes negative. With R² = -0.5, your residual error is 1.5× larger than the variance around the mean, indicating severe model failure.\n\nThis typically indicates fundamental problems. The model might be completely mis-specified—for example, fitting a linear model to exponential growth, or using features totally unrelated to the target. It could result from severe overfitting on training data that doesn\'t generalize at all to test data, though overfitting usually shows as low R² rather than negative. Negative R² can also occur from data leakage in reverse: testing on a different distribution than training, where the training distribution\'s mean is actually a worse predictor than the model would be on its own distribution. Preprocessing errors like scaling the test set incorrectly or features missing in test data can also cause this.\n\nPractically, negative R² demands immediate investigation. First, check for data issues: ensure train and test come from the same distribution, verify no data leakage or preprocessing errors, confirm target variable is measured consistently. Second, examine model assumptions: plot predictions vs actuals to see if there\'s any relationship, check residual plots for patterns indicating model mis-specification. Third, try the simplest possible baseline (mean prediction) and verify it actually outperforms your model. If baseline is indeed better, you likely need a completely different modeling approach, more relevant features, or to reconsider whether the problem is predictable with available data. A negative R² is a strong signal that something is seriously wrong—don\'t try to tweak hyperparameters, rebuild from scratch.'
    },
    {
      question: 'Why is RMSE more sensitive to outliers than MAE?',
      answer: 'RMSE (Root Mean Squared Error) is more sensitive to outliers than MAE (Mean Absolute Error) because it squares the errors before averaging, which disproportionately penalizes large errors. Consider two predictions with errors [1, 1, 10]: MAE = (1+1+10)/3 = 4.0, while RMSE = √[(1²+1²+10²)/3] = √(102/3) = 5.83. The single large error (10) has modest impact on MAE but substantially inflates RMSE. With errors [1, 1, 1], MAE = 1.0 and RMSE = 1.0, but with [0, 0, 3] (same total error), MAE = 1.0 while RMSE = 1.73, showing how RMSE penalizes concentrated errors.\n\nMathematically, squaring errors means a prediction that\'s off by 10 contributes 100 to the squared error sum, while ten predictions each off by 1 only contribute 10 total. The ratio scales quadratically: doubling the error quadruples its contribution to RMSE but only doubles its contribution to MAE. This makes RMSE more sensitive to the worst predictions—a single very bad prediction can dominate RMSE while having limited impact on MAE. Taking the square root at the end brings the units back to match the target variable, but doesn\'t undo the disproportionate weighting of large errors.\n\nChoose RMSE when large errors are particularly undesirable and you want to heavily penalize them. For example, in real estate price prediction, being off by $100k on a luxury home is much worse than being off by $10k on ten houses, and RMSE captures this. However, if outliers in your data are due to measurement errors or rare anomalies that you don\'t want to dominate your metric, MAE is better as it treats all errors linearly. MAE is also more robust and interpretable—it directly represents average absolute error in the target\'s units. In domains with naturally occurring outliers you must handle (extreme weather, epidemic forecasting), RMSE\'s outlier sensitivity might lead to models overfitting to rare extreme cases at the expense of typical performance. The choice depends on your loss function\'s true shape: quadratic losses naturally correspond to RMSE, linear losses to MAE.'
    },
    {
      question: 'You are building a cancer detection model. Which metric(s) would you prioritize and why?',
      answer: 'For cancer detection, prioritize recall (sensitivity) as the primary metric, while monitoring precision to avoid excessive false alarms. Missing a cancer case (false negative) has catastrophic consequences—delayed treatment can mean the difference between curable and terminal disease, or even life and death. A false positive (flagging cancer when there is none) is much less costly—it leads to additional testing (biopsies, imaging) which causes stress and expense but no permanent harm. The cost asymmetry is extreme: false negatives are potentially fatal, false positives are inconvenient and expensive but manageable.\n\nAim for very high recall (>95%, ideally >99%) to catch nearly all cancer cases, accepting moderate precision (perhaps 20-50% depending on cancer type and screening context). This means your model acts as a sensitive screening tool: it flags many patients for follow-up, knowing that confirmatory tests will filter out most false positives. For example, if 1% of screened patients have cancer, a model with 99% recall and 20% precision would correctly identify 99 of 100 cancer patients while also flagging 396 false positives (495 total flagged patients). Those 495 people get diagnostic workup, catch 99 real cancers, and clear 396 healthy people—acceptable trade-off.\n\nSecondary metrics matter too. Use F2-score (weights recall 2× higher than precision) for a single balanced metric, or F0.5-score if false positives are moderately costly. Monitor specificity (true negative rate) to ensure you\'re not flagging everyone—a model that flags 100% of patients has perfect recall but is useless. Track precision at your operating recall level to understand false alarm burden on the healthcare system. For different cancer types, adjust thresholds: aggressive cancers demand higher recall, slow-growing cancers might accept slightly lower recall with higher precision. Finally, consider calibration—if the model outputs cancer probability, ensure probabilities are reliable so doctors can make informed decisions about aggressive vs conservative follow-up based on risk level. The overarching principle: optimize to catch cancers even at the cost of false alarms, because the downside of missing cancer far outweighs the downside of unnecessary testing.'
    }
  ],
  quizQuestions: [
    {
      id: 'metrics-q1',
      question: 'You are building a spam email classifier. Your model achieves 99% accuracy, but users complain that spam emails still reach their inbox. What is the most likely issue?',
      options: [
        'The model has high precision but low recall for spam',
        'The model has high recall but low precision for spam',
        'The accuracy metric is appropriate for this task',
        'The model needs more training data'
      ],
      correctAnswer: 0,
      explanation: 'High accuracy with user complaints suggests the model rarely labels emails as spam (high precision = few false positives) but misses many spam emails (low recall = many false negatives). The dataset is likely imbalanced toward non-spam, making accuracy misleading.'
    },
    {
      id: 'metrics-q2',
      question: 'You are predicting house prices. Your model achieves RMSE=50,000 and MAE=20,000. What does this tell you?',
      options: [
        'The model is biased and consistently overestimates prices',
        'There are likely outliers or large errors in predictions',
        'The model is perfect with no errors',
        'MAE should always be larger than RMSE'
      ],
      correctAnswer: 1,
      explanation: 'RMSE (50k) being much larger than MAE (20k) indicates some predictions have large errors. RMSE amplifies large errors due to squaring, while MAE treats all errors equally. This suggests outliers or occasional large mispredictions.'
    },
    {
      id: 'metrics-q3',
      question: 'For a fraud detection system where fraudulent transactions are 0.1% of all transactions, which metric is MOST appropriate?',
      options: [
        'Accuracy',
        'ROC-AUC',
        'Precision-Recall AUC',
        'Mean Squared Error'
      ],
      correctAnswer: 2,
      explanation: 'PR-AUC is best for highly imbalanced datasets. Accuracy would be 99.9% by predicting everything as non-fraud. ROC-AUC can be overly optimistic due to the large number of true negatives. PR-AUC focuses on positive class performance.'
    }
  ]
};
