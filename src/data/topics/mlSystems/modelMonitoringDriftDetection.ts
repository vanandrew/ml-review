import { Topic } from '../../../types';

export const modelMonitoringDriftDetection: Topic = {
  id: 'model-monitoring-drift-detection',
  title: 'Model Monitoring & Drift Detection',
  category: 'ml-systems',
  description: 'Monitoring ML models in production and detecting performance degradation',
  content: `
    <h2>Model Monitoring & Drift Detection: Keeping Models Healthy in Production</h2>
    
    <p>Your model achieves 95% accuracy in production on launch day. Six months later, it's at 78%. What happened? The world changed. User behavior evolved. Competitors altered the landscape. Fraudsters adapted their tactics. Your model, frozen in time with patterns learned from old data, struggles with this new reality. This is drift—the silent killer of production ML systems.</p>

    <p>Unlike traditional software where bugs are deterministic and obvious, ML model degradation is gradual and subtle. The code runs fine. No errors appear in logs. But predictions slowly become less accurate, less relevant, less valuable. Without comprehensive monitoring and drift detection, you won't notice until business metrics tank and users complain. By then, damage is done. Proactive monitoring catches drift early, triggering retraining before users notice anything wrong.</p>

    <h3>Types of Drift: Understanding How Models Fail</h3>

    <h4>Data Drift (Covariate Shift): The Input Changes</h4>

    <p><strong>Data drift</strong> occurs when input feature distributions change while the underlying relationship between features and target remains constant: P<sub>train</sub>(X) ≠ P<sub>prod</sub>(X), but P(Y|X) stays stable. Your features look different, but the patterns relating them to outcomes haven't changed.</p>

    <p><strong>Examples:</strong> User demographics shift (younger users join, older users leave). Seasonal patterns emerge in e-commerce (holiday shopping differs from summer). Economic conditions change financial models (recession vs. boom). New product categories appear in recommendation systems.</p>

    <p><strong>Impact:</strong> Predictions become less reliable for the new distribution. Model might still be accurate where it has data but is now extrapolating to unfamiliar territory. Often detected before significant performance degradation—feature distributions shift before accuracy drops, giving you early warning.</p>

    <h4>Concept Drift: The Relationship Changes</h4>

    <p><strong>Concept drift</strong> is more insidious: the relationship between features and target changes. P(Y|X) evolves over time, even if P(X) stays constant. The same features now mean something different. A fraud detection model finds that patterns indicating fraud last year now indicate legitimate behavior because fraudsters evolved.</p>

    <p><strong>Examples:</strong> Fraud patterns evolve as criminals adapt to detection. User preferences change (fashion trends, music tastes). Market conditions shift in trading models (bull to bear market). Disease symptoms change for medical diagnosis (virus mutations).</p>

    <p><strong>Types of concept drift:</strong> <em>Sudden (abrupt) drift</em>—rapid change requiring immediate response (new regulation changes all fraud patterns overnight). <em>Gradual drift</em>—slow, continuous change where scheduled retraining suffices (aging population, shifting preferences). <em>Incremental drift</em>—step-wise changes over time, monitor and retrain at inflection points. <em>Recurring concepts</em>—patterns repeat periodically (seasonal demand), consider ensembles of season-specific models.</p>

    <h4>Label Drift (Prior Probability Shift): The Outcome Distribution Changes</h4>

    <p><strong>Label drift</strong> occurs when target label distribution changes: P<sub>train</sub>(Y) ≠ P<sub>prod</sub>(Y). The relative frequency of classes shifts. Your fraud model trained on 1% fraud rate now faces 3% fraud. Class imbalance changes, affecting precision/recall tradeoffs and optimal decision thresholds.</p>

    <p><strong>Examples:</strong> Fraud increases during economic downturns. Product popularity shifts (some products dominate sales). Disease prevalence changes (pandemic vs. endemic phases).</p>

    <h4>Prediction Drift: The Symptom of Upstream Problems</h4>

    <p><strong>Prediction drift</strong> is when the model's prediction distribution changes—not a root cause but a symptom indicating upstream issues. Sudden spike in positive predictions might mean data pipeline changed, features are computed differently, or real drift occurred. Consistently low confidence scores suggest model uncertainty about inputs it's seeing. Prediction distribution becoming uniform indicates the model can't discriminate anymore.</p>

    <h3>Monitoring Metrics: What to Track</h3>

    <h4>Model Performance Metrics: The Ground Truth</h4>

    <p>Track the same metrics used during training: <strong>Classification</strong>—accuracy, precision, recall, F1, AUC-ROC. <strong>Regression</strong>—MAE, RMSE, R². <strong>Ranking</strong>—NDCG, MAP, MRR. These directly measure whether the model still works.</p>

    <p><strong>Challenge: The label lag problem.</strong> Ground truth labels are often delayed or unavailable in production. Fraud investigations take weeks. Medical outcomes emerge months later. Purchase decisions happen days after recommendations. You can't wait for labels to detect problems—you need faster signals.</p>

    <h4>Proxy Metrics: Leading Indicators</h4>

    <p>Business metrics that correlate with model performance provide earlier signals: Click-through rate (CTR), conversion rate, user engagement time, revenue per user, customer satisfaction scores. If recommendation model degrades, engagement drops before you get ground truth on whether recommendations were good. These proxy metrics alert you faster.</p>

    <h4>System Performance Metrics: The Infrastructure View</h4>

    <p>Track operational health: <strong>Latency</strong>—P50, P95, P99 prediction times. <strong>Throughput</strong>—requests per second. <strong>Error rate</strong>—failed predictions, timeouts, exceptions. <strong>Resource usage</strong>—CPU, memory, GPU utilization. Performance degradation might indicate model complexity increased, batch sizes changed, or infrastructure issues. These metrics don't tell you if predictions are good, but they tell you if the system is healthy.</p>

    <h3>Drift Detection Methods: Catching Problems Early</h3>

    <h4>Choosing the Right Drift Detection Method</h4>

    <p><strong>Decision guide for selecting drift detection tests:</strong></p>

    <p><strong>For continuous numerical features:</strong> Use <strong>Kolmogorov-Smirnov (K-S) test</strong> as your default—non-parametric, no distribution assumptions, widely applicable. Use <strong>Population Stability Index (PSI)</strong> when you need interpretable magnitude of drift ("PSI = 0.3 means significant drift"). Use <strong>Kullback-Leibler divergence</strong> when comparing probability distributions directly, especially for binned or discretized features.</p>

    <p><strong>For categorical features:</strong> Use <strong>Chi-square test</strong> for low-to-moderate cardinality (<50 categories)—tests if category frequencies changed significantly. Use <strong>PSI adapted for categories</strong> for interpretable drift magnitude. For high-cardinality categories (>100), aggregate rare categories or use embedding-based distance metrics.</p>

    <p><strong>For multivariate drift (all features together):</strong> Use <strong>discriminator/adversarial approach</strong>—train classifier to distinguish training from production data. If AUC > 0.7, significant drift exists. Feature importance reveals which features drifted most. This is powerful when you care about "overall drift" rather than feature-by-feature analysis.</p>

    <p><strong>For model performance drift:</strong> Track <strong>prediction distribution</strong> using K-S test on prediction values—are predictions shifting? Monitor <strong>prediction confidence/uncertainty</strong>—increasing uncertainty suggests model encountering unfamiliar data. Use <strong>reconstruction error</strong> (autoencoder approach) for deep learning—high reconstruction error flags out-of-distribution inputs.</p>

    <p><strong>Practical deployment strategy:</strong> Start with PSI for all numerical features (easy to interpret) and Chi-square for categoricals. Add discriminator approach monthly for comprehensive multivariate check. Track prediction distribution daily. Alert when multiple signals fire simultaneously—single feature drift might be noise, but PSI + discriminator + prediction shift together is actionable.</p>

    <h4>Statistical Tests for Data Drift</h4>

    <p><strong>Kolmogorov-Smirnov (K-S) test</strong> compares two samples to test if they come from the same distribution. Works for continuous features. Non-parametric—no distribution assumptions. Null hypothesis: same distribution. If p-value < 0.05, reject null—distributions differ significantly. Run K-S test on each feature comparing training vs. recent production data.</p>

    <p><strong>Chi-square test</strong> for categorical features. Tests if observed frequencies match expected frequencies. Requires sufficient sample size in each category. If significant, categorical distribution shifted.</p>

    <p><strong>Population Stability Index (PSI)</strong> quantifies distribution shift magnitude. Bin features, compare actual vs. expected percentages per bin: PSI = Σ (actual% - expected%) × ln(actual% / expected%). Interpretation: PSI < 0.1 (no significant change), 0.1 < PSI < 0.25 (moderate change, investigate), PSI > 0.25 (significant change, retrain likely needed). PSI provides intuitive magnitude—not just "different" but "how much different."</p>

    <p><strong>Kullback-Leibler (KL) divergence</strong> measures how one distribution differs from reference. Not symmetric—D<sub>KL</sub>(P||Q) ≠ D<sub>KL</sub>(Q||P). Use Jensen-Shannon divergence for symmetric version. Higher KL divergence indicates greater distribution shift.</p>

    <h4>Model-Based Drift Detection</h4>

    <p><strong>Discriminator approach:</strong> Train binary classifier to distinguish training data (label 0) from production data (label 1). If classifier achieves high accuracy (>70%), significant drift exists—distributions are distinguishable. Feature importances reveal which features drifted most, guiding investigation. Clever technique: if model can tell which era data comes from, data must be different.</p>

    <p><strong>Uncertainty monitoring:</strong> Track prediction confidence/uncertainty. Bayesian models and ensembles provide uncertainty estimates. Increasing uncertainty suggests model encountering unfamiliar data—possible drift indicator. Particularly useful for deep learning with dropout-based uncertainty or ensemble disagreement.</p>

    <p><strong>Reconstruction error:</strong> Train autoencoder on training data to learn normal data manifold. Apply to production data—high reconstruction error indicates out-of-distribution samples. Autoencoder can't reconstruct what it hasn't seen. Rising reconstruction error means production data deviating from training distribution.</p>

    <h3>Monitoring System Architecture: Building the Infrastructure</h3>

    <p><strong>Data collection:</strong> Log all predictions with timestamps. Store input features (respecting privacy—anonymize PII, aggregate if necessary). Collect ground truth labels when available (asynchronously—fraud investigations complete, purchases finalized). Record metadata: model version, user segment, A/B test variant, geographic region. This enables detailed post-hoc analysis.</p>

    <p><strong>Metrics computation:</strong> Batch computation on regular intervals (hourly for critical systems, daily for stable ones). Compute windowed statistics: 7-day, 30-day moving averages smooth out noise while capturing trends. Compare against baseline—training distribution or recent stable period. Detect significant deviations.</p>

    <p><strong>Alerting:</strong> <em>Threshold-based</em>—trigger alerts when metrics exceed predefined thresholds (accuracy drops below 90%, PSI > 0.25). <em>Anomaly detection</em>—use statistical methods to detect outliers (Z-score, IQR, Isolation Forest). <em>Trend-based</em>—alert on sustained degradation, not single anomalies (3 consecutive days of declining accuracy). <em>Severity levels</em>—warning (investigate) vs. critical (immediate action, page on-call). Avoid alert fatigue—calibrate thresholds to minimize false positives.</p>

    <p><strong>Visualization dashboard:</strong> Time series plots of key metrics showing trends. Feature distribution comparisons (training vs. production histograms side-by-side). Prediction distribution over time (are predictions changing?). Error analysis breakdowns by segment, time, feature values. Dashboards enable quick diagnosis—operators spot patterns humans see better than automated rules.</p>

    <h3>Response Strategies: What to Do When Drift Strikes</h3>

    <h4>Investigate Root Cause</h4>

    <p>Not all drift requires retraining. First, investigate: <strong>Data quality issues</strong>—missing values increased, feature computation bug, upstream pipeline change. Fix the pipeline, not the model. <strong>Upstream system changes</strong>—feature service updated, data source changed format. Restore compatibility. <strong>True distribution shift</strong>—real world changed, model needs updating. <strong>Seasonality or expected variation</strong>—holiday shopping, weekend patterns. Don't retrain on noise.</p>

    <h4>Model Retraining Strategies</h4>

    <p><strong>Scheduled retraining</strong> on regular cadence (weekly, monthly) regardless of drift. Simple, predictable, prevents drift before it manifests. Risk: unnecessary retraining wastes resources; delayed retraining if drift occurs between schedules.</p>

    <p><strong>Triggered retraining</strong> automatically when drift detected. Efficient—only retrain when needed. Responsive—catch problems fast. Requires robust drift detection to avoid false triggers.</p>

    <p><strong>Incremental learning (online learning)</strong> updates model with new data continuously without full retraining. Adapts quickly, lower computational cost. But risk of catastrophic forgetting (model forgets old patterns) and drift amplification (errors compound). Use carefully with safeguards.</p>

    <p><strong>Feature engineering improvements:</strong> When drift occurs, analyze which features drifted most. Add new features capturing emergent patterns. Remove features that became irrelevant or noisy. Create time-aware features if temporal patterns emerged.</p>

    <p><strong>Model architecture changes:</strong> If concept drift is fundamental, feature engineering alone won't suffice. Consider more complex models capturing new patterns. Ensemble methods combining multiple models for robustness. Domain adaptation techniques explicitly modeling distribution shift.</p>

    <h3>Best Practices: Building Robust Monitoring</h3>

    <p><strong>Start simple:</strong> Monitor basic metrics first (accuracy, latency, throughput). Add complexity as you understand system behavior. Don't build elaborate monitoring before understanding what matters.</p>

    <p><strong>Establish baselines:</strong> During shadow deployment, collect baseline metrics representing healthy model behavior. All future monitoring compares against this baseline. Without baselines, you can't tell normal from abnormal.</p>

    <p><strong>Segment analysis:</strong> Don't just monitor overall metrics. Break down by user segment (new vs. returning), device type (mobile vs. desktop), geographic region. Model might degrade for specific segments while overall metrics look fine.</p>

    <p><strong>Calibrate alert thresholds:</strong> Tune to balance sensitivity (catch real issues) vs. specificity (avoid false alarms). Alert fatigue is real—too many false alerts and people ignore them. Start conservative, tighten as confidence grows.</p>

    <p><strong>Adaptive retraining cadence:</strong> High drift domains (fraud detection, trending content) need frequent retraining (daily/weekly). Moderate drift (recommendations, search) benefit from monthly retraining. Low drift domains (credit scoring, medical diagnosis on stable populations) can retrain quarterly. Adjust frequency based on observed drift patterns.</p>

    <p><strong>Data retention strategy:</strong> Store recent production data for retraining (last 3-6 months at full resolution). Sample historical data for long-term trend analysis (keep 10% of older data). Balance storage costs with model quality needs. Legal/compliance requirements may mandate retention periods.</p>

    <p>Production ML systems are living organisms requiring continuous care. Monitor comprehensively at multiple levels—features, predictions, performance, business metrics. Detect drift early with statistical tests and model-based methods. Respond appropriately: investigate root causes, retrain when needed, validate before deployment. Build robust monitoring infrastructure with automated alerts and intuitive dashboards. The best model is the one that stays good over time, adapting as the world evolves. Master monitoring and drift detection, and your models will remain healthy, accurate, and valuable long after deployment.</p>
  `,
  codeExamples: [
    {
      language: 'Python',
      code: `import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from typing import Dict, Tuple

class DriftDetector:
  """Detect data drift using statistical tests."""

  def __init__(self, reference_data: pd.DataFrame):
      """
      Args:
          reference_data: Training/baseline data to compare against
      """
      self.reference_data = reference_data

  def kolmogorov_smirnov_test(self, feature: str, current_data: pd.DataFrame,
                                alpha: float = 0.05) -> Dict:
      """
      Perform K-S test to detect drift in continuous features.

      Returns:
          Dict with test statistic, p-value, and drift detected flag
      """
      ref_values = self.reference_data[feature].dropna()
      cur_values = current_data[feature].dropna()

      # Two-sample K-S test
      statistic, p_value = stats.ks_2samp(ref_values, cur_values)

      return {
          'feature': feature,
          'ks_statistic': statistic,
          'p_value': p_value,
          'drift_detected': p_value < alpha,
          'alpha': alpha
      }

  def chi_square_test(self, feature: str, current_data: pd.DataFrame,
                      alpha: float = 0.05) -> Dict:
      """
      Perform chi-square test for categorical features.
      """
      # Get value counts
      ref_counts = self.reference_data[feature].value_counts()
      cur_counts = current_data[feature].value_counts()

      # Align categories
      all_categories = set(ref_counts.index) | set(cur_counts.index)
      ref_freq = np.array([ref_counts.get(cat, 0) for cat in all_categories])
      cur_freq = np.array([cur_counts.get(cat, 0) for cat in all_categories])

      # Normalize to proportions
      ref_prop = ref_freq / ref_freq.sum()
      cur_prop = cur_freq / cur_freq.sum()

      # Chi-square test
      chi2_stat = np.sum((cur_freq - ref_prop * cur_freq.sum())**2 /
                         (ref_prop * cur_freq.sum() + 1e-10))
      p_value = 1 - stats.chi2.cdf(chi2_stat, df=len(all_categories) - 1)

      return {
          'feature': feature,
          'chi2_statistic': chi2_stat,
          'p_value': p_value,
          'drift_detected': p_value < alpha,
          'alpha': alpha
      }

  def population_stability_index(self, feature: str, current_data: pd.DataFrame,
                                  n_bins: int = 10) -> Dict:
      """
      Calculate PSI for continuous features.

      PSI < 0.1: No significant change
      0.1 < PSI < 0.25: Moderate change
      PSI > 0.25: Significant change
      """
      # Create bins based on reference data
      _, bin_edges = np.histogram(self.reference_data[feature].dropna(),
                                    bins=n_bins)

      # Calculate proportions in each bin
      ref_hist, _ = np.histogram(self.reference_data[feature].dropna(),
                                   bins=bin_edges)
      cur_hist, _ = np.histogram(current_data[feature].dropna(),
                                   bins=bin_edges)

      # Convert to proportions
      ref_prop = (ref_hist + 1) / (ref_hist.sum() + n_bins)  # Add smoothing
      cur_prop = (cur_hist + 1) / (cur_hist.sum() + n_bins)

      # Calculate PSI
      psi = np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop))

      # Interpret PSI
      if psi < 0.1:
          interpretation = "No significant change"
      elif psi < 0.25:
          interpretation = "Moderate change"
      else:
          interpretation = "Significant change"

      return {
          'feature': feature,
          'psi': psi,
          'interpretation': interpretation,
          'drift_detected': psi > 0.1
      }

  def discriminator_test(self, current_data: pd.DataFrame,
                        features: list, alpha: float = 0.05) -> Dict:
      """
      Train classifier to distinguish training vs production data.
      High AUC indicates significant drift.
      """
      from sklearn.ensemble import RandomForestClassifier
      from sklearn.model_selection import cross_val_score

      # Label data: 0 = reference, 1 = current
      ref_labeled = self.reference_data[features].copy()
      ref_labeled['is_current'] = 0
      cur_labeled = current_data[features].copy()
      cur_labeled['is_current'] = 1

      # Combine and shuffle
      combined = pd.concat([ref_labeled, cur_labeled], ignore_index=True)
      combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

      X = combined[features]
      y = combined['is_current']

      # Train discriminator
      clf = RandomForestClassifier(n_estimators=100, random_state=42,
                                    max_depth=5)
      cv_scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
      mean_auc = cv_scores.mean()

      # If AUC significantly > 0.5, drift detected
      # AUC = 0.5 means can't distinguish → no drift
      # AUC close to 1.0 means strong drift
      drift_detected = mean_auc > 0.55  # Threshold

      # Fit full model to get feature importances
      clf.fit(X, y)
      feature_importance = pd.DataFrame({
          'feature': features,
          'importance': clf.feature_importances_
      }).sort_values('importance', ascending=False)

      return {
          'discriminator_auc': mean_auc,
          'drift_detected': drift_detected,
          'feature_importances': feature_importance.to_dict('records'),
          'interpretation': f"AUC of {mean_auc:.3f} ({'strong' if mean_auc > 0.7 else 'moderate' if mean_auc > 0.6 else 'weak'} drift)"
      }

# Usage example
# Generate synthetic data
np.random.seed(42)
n_samples = 10000

# Reference data (training distribution)
reference_data = pd.DataFrame({
  'age': np.random.normal(40, 10, n_samples),
  'income': np.random.normal(50000, 15000, n_samples),
  'category': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2])
})

# Current data (with drift)
current_data = pd.DataFrame({
  'age': np.random.normal(35, 12, n_samples),  # Shifted mean and variance
  'income': np.random.normal(52000, 18000, n_samples),  # Shifted
  'category': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.3, 0.4, 0.3])  # Different distribution
})

# Detect drift
detector = DriftDetector(reference_data)

print("=== Data Drift Detection Results ===\\n")

# K-S test for continuous features
print("1. Kolmogorov-Smirnov Test (age):")
ks_result = detector.kolmogorov_smirnov_test('age', current_data)
print(f"   Statistic: {ks_result['ks_statistic']:.4f}")
print(f"   P-value: {ks_result['p_value']:.4f}")
print(f"   Drift detected: {ks_result['drift_detected']}\\n")

# PSI for income
print("2. Population Stability Index (income):")
psi_result = detector.population_stability_index('income', current_data)
print(f"   PSI: {psi_result['psi']:.4f}")
print(f"   Interpretation: {psi_result['interpretation']}\\n")

# Chi-square for categorical
print("3. Chi-Square Test (category):")
chi_result = detector.chi_square_test('category', current_data)
print(f"   Chi2: {chi_result['chi2_statistic']:.4f}")
print(f"   P-value: {chi_result['p_value']:.4f}")
print(f"   Drift detected: {chi_result['drift_detected']}\\n")

# Discriminator approach
print("4. Discriminator Test (all features):")
disc_result = detector.discriminator_test(current_data, ['age', 'income'])
print(f"   AUC: {disc_result['discriminator_auc']:.4f}")
print(f"   {disc_result['interpretation']}")
print(f"   Most important features:")
for feat in disc_result['feature_importances'][:2]:
  print(f"   - {feat['feature']}: {feat['importance']:.3f}")`,
      explanation: 'Comprehensive drift detection toolkit implementing K-S test for continuous features, chi-square test for categorical features, Population Stability Index (PSI), and discriminator-based drift detection. Shows how to quantify and interpret different types of data drift.'
    },
    {
      language: 'Python',
      code: `import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List
import warnings

class ModelMonitor:
  """Production model monitoring system."""

  def __init__(self, model_name: str, window_size: int = 1000):
      """
      Args:
          model_name: Name of the model being monitored
          window_size: Number of recent predictions to keep in memory
      """
      self.model_name = model_name
      self.window_size = window_size

      # Circular buffers for recent data
      self.predictions = deque(maxlen=window_size)
      self.actuals = deque(maxlen=window_size)
      self.features = deque(maxlen=window_size)
      self.timestamps = deque(maxlen=window_size)

      # Reference statistics (from training)
      self.reference_stats = {}

      # Alert thresholds
      self.thresholds = {
          'accuracy_drop': 0.05,  # Alert if accuracy drops by 5%
          'latency_p95': 100,     # Alert if P95 latency > 100ms
          'error_rate': 0.01      # Alert if error rate > 1%
      }

  def set_reference_stats(self, training_data: pd.DataFrame,
                         feature_cols: List[str]):
      """Store reference statistics from training data."""
      self.reference_stats = {
          'feature_means': training_data[feature_cols].mean().to_dict(),
          'feature_stds': training_data[feature_cols].std().to_dict(),
          'feature_mins': training_data[feature_cols].min().to_dict(),
          'feature_maxs': training_data[feature_cols].max().to_dict()
      }

  def log_prediction(self, features: Dict, prediction: float,
                    actual: float = None, latency_ms: float = None):
      """Log a single prediction for monitoring."""
      self.predictions.append(prediction)
      self.features.append(features)
      self.timestamps.append(datetime.now())

      if actual is not None:
          self.actuals.append(actual)

      if latency_ms is not None:
          # Monitor latency
          if latency_ms > self.thresholds['latency_p95']:
              self._trigger_alert('latency', f"High latency: {latency_ms:.1f}ms")

  def calculate_metrics(self) -> Dict:
      """Calculate monitoring metrics from recent predictions."""
      if len(self.predictions) == 0:
          return {}

      metrics = {}

      # Prediction distribution
      predictions_arr = np.array(list(self.predictions))
      metrics['prediction_mean'] = float(np.mean(predictions_arr))
      metrics['prediction_std'] = float(np.std(predictions_arr))
      metrics['prediction_min'] = float(np.min(predictions_arr))
      metrics['prediction_max'] = float(np.max(predictions_arr))

      # If we have ground truth labels
      if len(self.actuals) > 0:
          actuals_arr = np.array(list(self.actuals))
          preds_for_eval = predictions_arr[-len(actuals_arr):]

          # Classification metrics (assuming binary with threshold 0.5)
          pred_labels = (preds_for_eval > 0.5).astype(int)
          actual_labels = actuals_arr.astype(int)

          accuracy = (pred_labels == actual_labels).mean()
          metrics['accuracy'] = float(accuracy)

          # Regression metrics
          mae = np.mean(np.abs(preds_for_eval - actuals_arr))
          rmse = np.sqrt(np.mean((preds_for_eval - actuals_arr) ** 2))
          metrics['mae'] = float(mae)
          metrics['rmse'] = float(rmse)

      # Feature drift detection (simplified)
      if self.reference_stats and len(self.features) > 0:
          recent_features = pd.DataFrame(list(self.features)[-100:])  # Last 100
          drift_scores = {}

          for col in recent_features.columns:
              if col in self.reference_stats['feature_means']:
                  ref_mean = self.reference_stats['feature_means'][col]
                  ref_std = self.reference_stats['feature_stds'][col]

                  current_mean = recent_features[col].mean()

                  # Z-score: how many std devs away from reference
                  if ref_std > 0:
                      z_score = abs(current_mean - ref_mean) / ref_std
                      drift_scores[col] = float(z_score)

          metrics['feature_drift_scores'] = drift_scores

          # Alert on significant drift
          max_drift = max(drift_scores.values()) if drift_scores else 0
          if max_drift > 3:  # 3 standard deviations
              drifted_features = [f for f, s in drift_scores.items() if s > 3]
              self._trigger_alert('drift',
                                 f"Significant drift detected in: {drifted_features}")

      return metrics

  def detect_prediction_drift(self, historical_window: int = 7) -> Dict:
      """
      Detect if prediction distribution has changed over time.
      Compare recent predictions to historical baseline.
      """
      if len(self.predictions) < historical_window * 2:
          return {'status': 'insufficient_data'}

      # Split into historical and recent
      all_preds = list(self.predictions)
      split_point = len(all_preds) // 2

      historical = np.array(all_preds[:split_point])
      recent = np.array(all_preds[split_point:])

      # K-S test
      from scipy.stats import ks_2samp
      statistic, p_value = ks_2samp(historical, recent)

      drift_detected = p_value < 0.05

      return {
          'ks_statistic': float(statistic),
          'p_value': float(p_value),
          'drift_detected': drift_detected,
          'historical_mean': float(historical.mean()),
          'recent_mean': float(recent.mean())
      }

  def check_data_quality(self, features: Dict) -> Dict:
      """Check for data quality issues in incoming features."""
      issues = []

      # Check for missing values
      missing = [k for k, v in features.items() if v is None or
                 (isinstance(v, float) and np.isnan(v))]
      if missing:
          issues.append(f"Missing values: {missing}")

      # Check for out-of-range values
      if self.reference_stats:
          for feat, value in features.items():
              if feat in self.reference_stats['feature_mins']:
                  ref_min = self.reference_stats['feature_mins'][feat]
                  ref_max = self.reference_stats['feature_maxs'][feat]

                  # Allow some margin (3x the training range)
                  margin = (ref_max - ref_min) * 2
                  if value < ref_min - margin or value > ref_max + margin:
                      issues.append(f"{feat} out of range: {value:.2f} (expected {ref_min:.2f} to {ref_max:.2f})")

      return {
          'is_valid': len(issues) == 0,
          'issues': issues
      }

  def _trigger_alert(self, alert_type: str, message: str):
      """Trigger monitoring alert."""
      alert = {
          'timestamp': datetime.now().isoformat(),
          'model': self.model_name,
          'type': alert_type,
          'message': message
      }
      print(f"🚨 ALERT: {alert}")
      # In production: send to alerting system (PagerDuty, Slack, etc.)

  def generate_report(self) -> Dict:
      """Generate comprehensive monitoring report."""
      metrics = self.calculate_metrics()
      drift = self.detect_prediction_drift()

      report = {
          'model_name': self.model_name,
          'timestamp': datetime.now().isoformat(),
          'window_size': len(self.predictions),
          'metrics': metrics,
          'prediction_drift': drift,
          'health_status': 'healthy' if not drift.get('drift_detected', False) else 'degraded'
      }

      return report

# Usage example
monitor = ModelMonitor(model_name='fraud_detection_v1', window_size=1000)

# Set reference stats from training
training_data = pd.DataFrame({
  'transaction_amount': np.random.exponential(100, 10000),
  'account_age_days': np.random.uniform(0, 1000, 10000)
})
monitor.set_reference_stats(training_data, ['transaction_amount', 'account_age_days'])

# Simulate production predictions
print("Simulating production monitoring...\\n")

for i in range(100):
  # Normal predictions
  features = {
      'transaction_amount': float(np.random.exponential(100)),
      'account_age_days': float(np.random.uniform(0, 1000))
  }

  # Check data quality
  quality_check = monitor.check_data_quality(features)
  if not quality_check['is_valid']:
      print(f"Data quality issue: {quality_check['issues']}")
      continue

  prediction = np.random.random()  # Model prediction
  actual = np.random.choice([0, 1], p=[0.95, 0.05])  # 5% fraud rate
  latency = np.random.gamma(2, 10)  # Latency in ms

  monitor.log_prediction(features, prediction, actual, latency)

# Simulate drift scenario
print("\\nSimulating data drift...\\n")
for i in range(100):
  # Drifted distribution
  features = {
      'transaction_amount': float(np.random.exponential(200)),  # Mean doubled!
      'account_age_days': float(np.random.uniform(0, 1000))
  }

  prediction = np.random.random()
  monitor.log_prediction(features, prediction)

# Generate monitoring report
report = monitor.generate_report()
print("\\n=== Monitoring Report ===")
print(f"Model: {report['model_name']}")
print(f"Status: {report['health_status']}")
print(f"\\nPrediction Statistics:")
print(f"  Mean: {report['metrics']['prediction_mean']:.3f}")
print(f"  Std: {report['metrics']['prediction_std']:.3f}")
print(f"\\nPrediction Drift:")
print(f"  Detected: {report['prediction_drift'].get('drift_detected', 'N/A')}")
if 'p_value' in report['prediction_drift']:
  print(f"  P-value: {report['prediction_drift']['p_value']:.4f}")`,
      explanation: 'Production monitoring system that tracks predictions, calculates performance metrics, detects data quality issues, monitors feature drift, and triggers alerts. Implements sliding window analysis, prediction drift detection using K-S test, and comprehensive reporting for model health tracking.'
    }
  ],
  interviewQuestions: [
    {
      question: 'What is the difference between data drift and concept drift? Give examples of each.',
      answer: `Data drift occurs when input feature distributions change over time while relationships remain constant (e.g., customer age distribution shifting). Concept drift occurs when the relationship between inputs and outputs changes (e.g., economic conditions changing fraud patterns). Data drift affects feature statistics; concept drift affects model accuracy. Both require different detection and mitigation strategies - statistical tests for data drift, performance monitoring for concept drift.`
    },
    {
      question: 'How would you detect drift when ground truth labels are delayed or unavailable?',
      answer: `Use proxy metrics and indirect detection methods: (1) Monitor prediction distributions and confidence scores, (2) Track feature distributions using statistical tests (KS test, Chi-square), (3) Use auxiliary models trained on immediate feedback, (4) Monitor business metrics that correlate with model performance, (5) Implement anomaly detection on input features, (6) Use unsupervised drift detection methods like adversarial validation or domain adaptation techniques.`
    },
    {
      question: 'What is the Population Stability Index (PSI) and how do you interpret its values?',
      answer: `PSI measures distribution stability between training and production data by comparing probability distributions across bins. Formula: PSI = Σ[(Actual% - Expected%) × ln(Actual%/Expected%)]. Interpretation: PSI < 0.1 (no significant change), 0.1-0.2 (moderate change, investigate), PSI > 0.2 (significant shift, likely retrain needed). PSI helps identify which features are driving distribution changes and when model retraining is necessary.`
    },
    {
      question: 'How would you design a monitoring system for a real-time fraud detection model?',
      answer: `Comprehensive monitoring includes: (1) Real-time performance metrics - latency, throughput, error rates, (2) Model performance - precision, recall, false positive rates by time windows, (3) Feature monitoring - distribution shifts, missing values, outliers, (4) Business metrics - revenue impact, customer experience, (5) Alert system - automated notifications for threshold breaches, (6) Dashboard - real-time visualization, (7) Feedback loop - incorporate investigator feedback for continuous improvement.`
    },
    {
      question: 'What metrics would you track to monitor a recommendation system in production?',
      answer: `Multi-layered metrics: (1) Technical - response time, cache hit rates, system uptime, (2) Model performance - click-through rate, conversion rate, recommendation diversity, coverage, (3) Business metrics - revenue, user engagement, session length, (4) User experience - recommendation freshness, novelty, serendipity, (5) Fairness metrics - demographic parity, equal opportunity across user groups, (6) A/B testing results for model updates, (7) Cold start performance for new users/items.`
    },
    {
      question: 'When would you choose to retrain a model vs. adjusting feature engineering?',
      answer: `Retrain when: (1) Concept drift detected - fundamental relationships changed, (2) Significant performance degradation, (3) New data available with different patterns, (4) Major external changes (regulations, market shifts). Adjust features when: (1) Data drift without concept drift, (2) Feature engineering issues identified, (3) New relevant data sources available, (4) Feature quality improvements possible. Consider computational costs, urgency, and root cause analysis results.`
    }
  ],
  quizQuestions: [
    {
      id: 'drift1',
      question: 'What does data drift (covariate shift) mean?',
      options: ['Target variable distribution changes', 'Input feature distribution changes', 'Model predictions change', 'Code changes'],
      correctAnswer: 1,
      explanation: 'Data drift (covariate shift) occurs when the input feature distribution P(X) changes, while the relationship P(Y|X) stays the same. For example, user demographics shifting over time.'
    },
    {
      id: 'drift2',
      question: 'What PSI value indicates significant drift?',
      options: ['PSI < 0.1', '0.1 < PSI < 0.25', 'PSI > 0.25', 'PSI = 1.0'],
      correctAnswer: 2,
      explanation: 'PSI (Population Stability Index) > 0.25 indicates significant drift that requires investigation. PSI < 0.1 is no significant change, 0.1-0.25 is moderate change.'
    },
    {
      id: 'drift3',
      question: 'What is the "label lag problem" in production monitoring?',
      options: ['Models predict slowly', 'Ground truth labels arrive late or not at all', 'Feature computation is slow', 'Predictions are cached'],
      correctAnswer: 1,
      explanation: 'Label lag problem refers to ground truth labels being delayed or unavailable in production (e.g., knowing if a loan defaults takes months). This makes it hard to calculate performance metrics, requiring proxy metrics instead.'
    }
  ]
};
