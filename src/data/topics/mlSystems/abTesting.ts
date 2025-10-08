import { Topic } from '../../../types';

export const abTesting: Topic = {
  id: 'ab-testing',
  title: 'A/B Testing for ML Models',
  category: 'ml-systems',
  description: 'Statistical methods for comparing model performance in production',
  content: `
    <h2>A/B Testing: The Scientific Method for ML in Production</h2>
    
    <p>You've trained two models. Model A achieves 92% accuracy offline. Model B achieves 91.5%. Deploy Model A, right? Not so fast. Offline metrics don't always correlate with real-world business value. Model B might be faster, leading to better user experience. It might perform better on the long-tail cases you actually care about. The only way to know which model truly performs better in production is through rigorous experimentation—A/B testing.</p>

    <p>A/B testing (split testing) is the gold standard for comparing variants—different models, features, or configurations—by measuring their impact on real users and business metrics. You split traffic: 50% of users see variant A (control), 50% see variant B (treatment). After collecting sufficient data, statistical tests reveal whether observed differences are real or just random noise. This rigorous methodology prevents costly mistakes and ensures data-driven decisions rather than gut feelings.</p>

    <h3>The Hypothesis Testing Framework: Statistical Rigor</h3>

    <p>Every A/B test is a statistical hypothesis test. The <strong>null hypothesis (H₀)</strong> states there's no difference between variants—any observed difference is due to random chance. The <strong>alternative hypothesis (H₁)</strong> states variant B performs differently (better or worse) than variant A. Our goal: collect enough evidence to reject H₀ with confidence.</p>

    <p>The <strong>p-value</strong> quantifies evidence against H₀: the probability of observing results as extreme as actual results if H₀ were true. If variant B shows 12% conversion and variant A shows 10%, the p-value answers: "What's the probability of seeing this difference (or larger) purely by chance if there's actually no real difference?" A small p-value (< 0.05 typically) suggests the difference is real, not random—we reject H₀.</p>

    <p>The <strong>significance level (α)</strong>, typically 0.05, is our threshold for rejecting H₀. If p < α, results are "statistically significant"—we're confident (95% confident for α=0.05) the difference is real. But significance isn't the whole story.</p>

    <p><strong>Statistical power (1-β)</strong> is the probability of detecting a true effect when it actually exists—avoiding false negatives. Typically we aim for 80% power. Higher power requires larger sample sizes but ensures we don't miss real improvements. Low power means even if variant B is truly better, we might fail to detect it, concluding "no significant difference" when a difference exists.</p>

    <h3>Choosing the Right Metrics: What Really Matters</h3>

    <p>Metrics drive decisions, so choose carefully. <strong>Primary metrics</strong> are the main business objectives you're optimizing: click-through rate (CTR), conversion rate, revenue per user, user engagement time. Pick one primary metric before the test—this is your decision criterion. Testing multiple primary metrics inflates false positive rates (more on this pitfall later).</p>

    <p><strong>Guardrail metrics</strong> protect against unintended consequences. Maybe variant B increases conversions but also increases page load time, degrading user experience. Guardrails like page load time, error rates, user retention, and overall satisfaction ensure improvements on the primary metric don't come at unacceptable costs elsewhere.</p>

    <p><strong>Model-specific metrics</strong> provide diagnostic insight: prediction accuracy, inference latency, prediction diversity, calibration error. These help you understand *why* business metrics changed and guide future iterations.</p>

    <h3>Experimental Design: The Foundation of Valid Tests</h3>

    <h4>Sample Size: How Much Data Do You Need?</h4>

    <p>Underpowered tests waste time and resources, concluding "no significant difference" when you simply didn't collect enough data. Overpowered tests waste resources and delay launches. Calculate required sample size upfront using power analysis.</p>

    <p>Key inputs: <strong>Minimum detectable effect (MDE)</strong>—the smallest improvement worth detecting (e.g., 2% CTR increase). Smaller MDEs require exponentially more data. <strong>Baseline rate</strong>—current metric value (e.g., 10% conversion). <strong>Significance level α</strong> (typically 0.05). <strong>Power (1-β)</strong> (typically 0.80).</p>

    <p>For proportion tests: n = 2 × [(Z<sub>α/2</sub> + Z<sub>β</sub>)²] × [p(1-p)] / (MDE)². Want to detect a 2% absolute increase in 10% baseline CTR with 80% power and α=0.05? You need ~3,800 users per variant—7,600 total.</p>

    <p><strong>Practical walkthrough:</strong> You're testing a new recommendation model. Current CTR is 8%. You want to detect if the new model achieves at least 9% CTR (1% absolute improvement = 12.5% relative lift). You want 80% power and α=0.05 significance.</p>

    <p><strong>Step 1—Set parameters:</strong> Baseline rate p = 0.08, MDE = 0.01 (detect 1% improvement), α = 0.05 (5% false positive rate), Power = 0.80 (80% chance to detect real effect). From Z-tables: Z<sub>α/2</sub> = 1.96 (for 95% confidence), Z<sub>β</sub> = 0.84 (for 80% power).</p>

    <p><strong>Step 2—Calculate sample size:</strong> n = 2 × [(1.96 + 0.84)²] × [0.08 × 0.92] / (0.01)² = 2 × [7.84] × [0.0736] / 0.0001 = 11,542 users per variant, so 23,084 total users needed.</p>

    <p><strong>Step 3—Estimate duration:</strong> If you get 1,000 users per day, the test needs 23,084 / 1,000 = 24 days minimum. Add time for weekly patterns: run for at least 4 full weeks = 28 days to capture weekday/weekend variation.</p>

    <p><strong>Step 4—Consider practical constraints:</strong> 28 days feels long? Options: (1) Accept larger MDE—detecting 1.5% improvement instead of 1% reduces sample size to ~5,200 per variant (11 days). (2) Reduce power to 70%—drops sample size to ~9,400 per variant (19 days). (3) Use unequal split (90/10)—limits risk but requires longer duration for same power. Always document your choices and their implications.</p>

    <h4>Randomization: Eliminating Bias</h4>

    <p><strong>User-level randomization</strong> (most common) assigns each user to a variant consistently throughout the experiment. User 12345 always sees variant A. This prevents within-user inconsistency (confusing users by switching variants mid-session) and enables measuring long-term effects. Implementation: hash user_id to deterministically assign variants—same user always gets same variant, even across sessions.</p>

    <p><strong>Request-level randomization</strong> assigns each request independently. Higher variance (same user sees both variants), requiring more data, but useful when user behavior is irrelevant (e.g., testing ranking algorithms where requests are independent).</p>

    <h4>Traffic Allocation: Balancing Risk and Power</h4>

    <p><strong>Equal split (50/50)</strong> maximizes statistical power—you get maximum data for both variants, detecting differences fastest. Use this when both variants are safe and you're confident neither will harm users.</p>

    <p><strong>Unequal split (e.g., 90/10)</strong> limits exposure to a potentially worse variant. New, risky changes start at 10% while 90% of users stay on the safe control. Trade-off: less statistical power, requiring larger overall sample size and longer test duration. Good for cautious rollouts.</p>

    <h4>Duration: Capturing True Effects</h4>

    <p>Run tests long enough to capture weekly patterns—weekends differ from weekdays. Minimum 1-2 weeks, ideally full weeks. Account for the <strong>novelty effect</strong>: users react differently to new experiences initially. That CTR spike in day 1 might fade by day 7. The <strong>primacy effect</strong>: long-time users may resist change, temporarily depressing metrics before adapting. Let these effects settle before concluding.</p>

    <h3>Statistical Tests: Choosing the Right Analysis</h3>

    <p>For <strong>binary outcomes</strong> (click/no-click, convert/don't-convert), use the <strong>two-proportion z-test</strong> to compare conversion rates between A and B. Tests whether observed rate difference is statistically significant.</p>

    <p>For <strong>continuous outcomes</strong> (revenue, time spent, latency), use the <strong>two-sample t-test</strong> to compare means. If data isn't normally distributed, use the <strong>Mann-Whitney U test</strong>—a non-parametric alternative making no distribution assumptions.</p>

    <p>For <strong>multiple variants</strong> (A/B/C/D...), first use <strong>ANOVA (Analysis of Variance)</strong> to test if any variant differs from others. If ANOVA is significant, perform pairwise comparisons with <strong>multiple comparison corrections</strong>: Bonferroni (divide α by number of comparisons—conservative) or FDR control (less conservative, controls false discovery rate).</p>

    <h3>Common Pitfalls: Where A/B Tests Go Wrong</h3>

    <h4>1. The Peeking Problem: Don't Check Early and Stop</h4>

    <p>Checking results repeatedly and stopping when significance appears inflates false positive rates dramatically. Each peek is another chance to find spurious significance. Solution: pre-commit to sample size and duration. Use sequential testing methods (like sequential probability ratio tests) if you must monitor continuously.</p>

    <h4>2. Multiple Testing: Testing Everything Finds Nothing</h4>

    <p>Testing 20 metrics with α=0.05? You'll get one false positive on average even if nothing changed. Designate one primary metric before the test. Use Bonferroni or FDR correction for secondary metrics, or treat them as exploratory.</p>

    <h4>3. Simpson's Paradox: Aggregate Lies, Segments Tell Truth</h4>

    <p>Variant B appears better overall but is worse for every subgroup. How? Variant B happens to get more high-converting users. Always analyze key segments (new vs. returning users, mobile vs. desktop, regions) to catch this. Stratified randomization prevents it.</p>

    <h4>4. Network Effects: Users Aren't Independent</h4>

    <p>In social networks or marketplaces, user behavior affects others. Control users see fewer viral posts because their friends are in treatment. Solution: cluster randomization—assign groups of connected users together, or use time-based switching (entire platform switches between variants periodically).</p>

    <h4>5. Sample Ratio Mismatch (SRM): Your Randomization is Broken</h4>

    <p>Expected 50/50 split but observed 52/48? SRM indicates bugs in randomization, bot traffic filtering differences, or data pipeline issues. Chi-square test detects SRM. Never analyze results with SRM—fix the underlying issue first. SRM invalidates test results completely.</p>

    <h3>ML-Specific Considerations: Models Aren't Web Buttons</h3>

    <p>When A/B testing models, measure both business metrics (CTR, revenue) and model metrics (accuracy, latency). Offline accuracy improvements might not translate to online gains if latency degrades user experience. Test the full pipeline—model, preprocessing, serving infrastructure.</p>

    <p>A/B test features by comparing models with/without them. Is that expensive feature worth the 0.5% accuracy gain? Does it justify the data collection and compute costs? Real-world testing answers questions offline evaluation can't.</p>

    <p>Hyperparameter tuning in production reveals discrepancies between offline and online performance. That aggressive regularization hurts offline metrics but improves generalization to real user behavior. Offline doesn't capture the full picture.</p>

    <h3>Advanced Techniques: Beyond Basic A/B</h3>

    <p><strong>Multi-armed bandits</strong> dynamically allocate more traffic to better-performing variants during the experiment, minimizing regret (cost of showing inferior variants). Faster convergence than fixed allocation, but less statistical rigor and harder retrospective analysis. Good for continuous optimization, not one-time decisions.</p>

    <p><strong>Stratified sampling</strong> ensures equal representation of important segments in both variants, reducing variance and allowing smaller sample sizes. Particularly effective with heterogeneous populations.</p>

    <p><strong>CUPED (Controlled-experiment Using Pre-Experiment Data)</strong> leverages pre-experiment user behavior to reduce variance. If you know high-spending users will spend more regardless of variant, control for this. Can reduce required sample size by 30-50%, particularly effective for high-variance metrics like revenue.</p>

    <p>A/B testing transforms guesses into knowledge, ensuring only genuinely better models reach users. Design tests rigorously: calculate sample sizes, randomize properly, choose metrics carefully, avoid pitfalls. Statistical significance isn't magic—it's the result of disciplined experimental methodology. Master this, and your models will improve continuously, backed by evidence rather than hope.</p>
  `,
  codeExamples: [
    {
      language: 'Python',
      code: `import numpy as np
from scipy import stats
from statsmodels.stats.power import zt_ind_solve_power
from statsmodels.stats.proportion import proportions_ztest
import pandas as pd

# Sample size calculation
def calculate_sample_size(baseline_rate, mde, alpha=0.05, power=0.80):
  """
  Calculate required sample size per variant for A/B test.

  Args:
      baseline_rate: Current conversion rate (e.g., 0.10 for 10%)
      mde: Minimum detectable effect (e.g., 0.02 for 2% absolute increase)
      alpha: Significance level (Type I error rate)
      power: Statistical power (1 - Type II error rate)

  Returns:
      Required sample size per variant
  """
  # Effect size (Cohen's h for proportions)
  p1 = baseline_rate
  p2 = baseline_rate + mde
  effect_size = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))

  # Calculate sample size using power analysis
  sample_size = zt_ind_solve_power(
      effect_size=effect_size,
      alpha=alpha,
      power=power,
      ratio=1.0,  # Equal sample sizes
      alternative='two-sided'
  )

  return int(np.ceil(sample_size))

# Example: How many users needed to detect 2% CTR increase?
baseline_ctr = 0.10  # 10% baseline click-through rate
mde = 0.02          # Want to detect 2% absolute increase (10% → 12%)

sample_size_per_variant = calculate_sample_size(baseline_ctr, mde)
print(f"Required sample size per variant: {sample_size_per_variant}")
print(f"Total required sample size: {sample_size_per_variant * 2}")

# Two-Proportion Z-Test
def ab_test_proportions(conversions_a, total_a, conversions_b, total_b):
  """
  Perform two-proportion z-test for A/B test.

  Returns:
      Dictionary with test results
  """
  # Conversion rates
  rate_a = conversions_a / total_a
  rate_b = conversions_b / total_b

  # Statistical test
  count = np.array([conversions_a, conversions_b])
  nobs = np.array([total_a, total_b])
  z_stat, p_value = proportions_ztest(count, nobs)

  # Confidence interval for difference
  pooled_p = (conversions_a + conversions_b) / (total_a + total_b)
  se = np.sqrt(pooled_p * (1 - pooled_p) * (1/total_a + 1/total_b))
  ci_low = (rate_b - rate_a) - 1.96 * se
  ci_high = (rate_b - rate_a) + 1.96 * se

  # Relative lift
  lift = (rate_b - rate_a) / rate_a * 100

  return {
      'control_rate': rate_a,
      'variant_rate': rate_b,
      'absolute_difference': rate_b - rate_a,
      'relative_lift_pct': lift,
      'z_statistic': z_stat,
      'p_value': p_value,
      'significant': p_value < 0.05,
      'ci_95': (ci_low, ci_high)
  }

# Example A/B test results
results = ab_test_proportions(
  conversions_a=1000,  # Control: 1000 conversions
  total_a=10000,       # out of 10,000 users
  conversions_b=1150,  # Variant: 1150 conversions
  total_b=10000        # out of 10,000 users
)

print("\\nA/B Test Results:")
print(f"Control conversion rate: {results['control_rate']:.3%}")
print(f"Variant conversion rate: {results['variant_rate']:.3%}")
print(f"Absolute difference: {results['absolute_difference']:.3%}")
print(f"Relative lift: {results['relative_lift_pct']:.1f}%")
print(f"P-value: {results['p_value']:.4f}")
print(f"Statistically significant: {results['significant']}")
print(f"95% CI: [{results['ci_95'][0]:.3%}, {results['ci_95'][1]:.3%}]")

# Sample Ratio Mismatch (SRM) check
def check_sample_ratio_mismatch(count_a, count_b, expected_ratio=0.5):
  """
  Check if observed traffic split matches expected ratio.
  Uses chi-square goodness-of-fit test.
  """
  total = count_a + count_b
  expected_a = total * expected_ratio
  expected_b = total * (1 - expected_ratio)

  # Chi-square test
  chi2_stat = ((count_a - expected_a)**2 / expected_a +
               (count_b - expected_b)**2 / expected_b)
  p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)

  observed_ratio_a = count_a / total

  return {
      'expected_ratio_a': expected_ratio,
      'observed_ratio_a': observed_ratio_a,
      'chi2_statistic': chi2_stat,
      'p_value': p_value,
      'srm_detected': p_value < 0.01  # Stricter threshold for SRM
  }

# Check for SRM
srm_check = check_sample_ratio_mismatch(10000, 10100, expected_ratio=0.5)
print("\\nSample Ratio Mismatch Check:")
print(f"Expected split: {srm_check['expected_ratio_a']:.1%} / {1-srm_check['expected_ratio_a']:.1%}")
print(f"Observed split: {srm_check['observed_ratio_a']:.1%} / {1-srm_check['observed_ratio_a']:.1%}")
print(f"SRM detected: {srm_check['srm_detected']} (p={srm_check['p_value']:.4f})")`,
      explanation: 'Statistical framework for A/B testing including sample size calculation based on desired power and minimum detectable effect, two-proportion z-test for comparing conversion rates, and sample ratio mismatch detection to ensure valid randomization.'
    },
    {
      language: 'Python',
      code: `import hashlib
from typing import Dict, List
import pandas as pd
from datetime import datetime

class ABTestFramework:
  """Production-ready A/B testing framework for ML models."""

  def __init__(self, experiments_config: Dict):
      """
      Args:
          experiments_config: Dict mapping experiment_id to config
              {
                  'model_comparison_v1': {
                      'variants': ['control', 'model_v2', 'model_v3'],
                      'traffic_allocation': [0.5, 0.25, 0.25],
                      'start_date': '2024-01-01',
                      'end_date': '2024-01-14'
                  }
              }
      """
      self.experiments = experiments_config

  def assign_variant(self, user_id: str, experiment_id: str) -> str:
      """
      Deterministically assign user to experiment variant.
      Uses hash function for consistent assignment.
      """
      if experiment_id not in self.experiments:
          return 'control'

      config = self.experiments[experiment_id]

      # Check if experiment is active
      now = datetime.now().date()
      start = datetime.strptime(config['start_date'], '%Y-%m-%d').date()
      end = datetime.strptime(config['end_date'], '%Y-%m-%d').date()

      if not (start <= now <= end):
          return 'control'

      # Hash user_id + experiment_id for deterministic assignment
      hash_input = f"{user_id}_{experiment_id}".encode('utf-8')
      hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)

      # Map hash to [0, 1) range
      random_value = (hash_value % 10000) / 10000

      # Assign variant based on traffic allocation
      variants = config['variants']
      allocation = config['traffic_allocation']

      cumulative = 0
      for variant, prob in zip(variants, allocation):
          cumulative += prob
          if random_value < cumulative:
              return variant

      return variants[-1]  # Fallback

  def log_exposure(self, user_id: str, experiment_id: str, variant: str):
      """Log user exposure to variant (for analysis)."""
      exposure_data = {
          'timestamp': datetime.now().isoformat(),
          'user_id': user_id,
          'experiment_id': experiment_id,
          'variant': variant
      }
      # In production: write to data warehouse/logging system
      print(f"EXPOSURE: {exposure_data}")

  def log_metric(self, user_id: str, experiment_id: str,
                 metric_name: str, value: float):
      """Log metric for A/B test analysis."""
      metric_data = {
          'timestamp': datetime.now().isoformat(),
          'user_id': user_id,
          'experiment_id': experiment_id,
          'metric_name': metric_name,
          'value': value
      }
      # In production: write to metrics database
      print(f"METRIC: {metric_data}")

# Production usage example
experiments_config = {
  'ranking_model_v2': {
      'variants': ['baseline_model', 'new_model'],
      'traffic_allocation': [0.5, 0.5],
      'start_date': '2024-01-01',
      'end_date': '2024-01-14'
  }
}

ab_framework = ABTestFramework(experiments_config)

# In prediction endpoint
def predict_with_ab_test(user_id: str, features: List[float]) -> float:
  """Make prediction with A/B testing."""
  experiment_id = 'ranking_model_v2'

  # Assign variant
  variant = ab_framework.assign_variant(user_id, experiment_id)
  ab_framework.log_exposure(user_id, experiment_id, variant)

  # Select model based on variant
  if variant == 'baseline_model':
      prediction = baseline_model.predict([features])[0]
  else:  # new_model
      prediction = new_model.predict([features])[0]

  return prediction

# In business logic (e.g., after user clicks)
def track_conversion(user_id: str, converted: bool, revenue: float):
  """Track conversion metrics for A/B test."""
  ab_framework.log_metric(user_id, 'ranking_model_v2', 'conversion', 1.0 if converted else 0.0)
  ab_framework.log_metric(user_id, 'ranking_model_v2', 'revenue', revenue)

# Analysis: Calculate metrics per variant
def analyze_experiment(experiment_id: str, metrics_df: pd.DataFrame):
  """
  Analyze A/B test results.

  Args:
      metrics_df: DataFrame with columns [user_id, variant, conversion, revenue]
  """
  results = metrics_df.groupby('variant').agg({
      'user_id': 'count',
      'conversion': ['sum', 'mean', 'std'],
      'revenue': ['sum', 'mean', 'std']
  }).round(4)

  print(f"\\nExperiment: {experiment_id}")
  print(results)

  # Statistical test for conversion rate
  variants = metrics_df['variant'].unique()
  if len(variants) == 2:
      control = metrics_df[metrics_df['variant'] == variants[0]]
      treatment = metrics_df[metrics_df['variant'] == variants[1]]

      from scipy.stats import ttest_ind

      # T-test for revenue
      t_stat, p_value = ttest_ind(control['revenue'], treatment['revenue'])
      print(f"\\nRevenue t-test: t={t_stat:.3f}, p={p_value:.4f}")

      # Z-test for conversion
      conversions = [control['conversion'].sum(), treatment['conversion'].sum()]
      totals = [len(control), len(treatment)]
      from statsmodels.stats.proportion import proportions_ztest
      z_stat, p_value = proportions_ztest(conversions, totals)
      print(f"Conversion z-test: z={z_stat:.3f}, p={p_value:.4f}")

# Simulate experiment data
np.random.seed(42)
n_users = 10000

simulation_data = {
  'user_id': [f'user_{i}' for i in range(n_users)],
  'variant': np.random.choice(['baseline_model', 'new_model'], n_users),
  'conversion': np.random.binomial(1, 0.1, n_users),  # 10% baseline conversion
  'revenue': np.random.exponential(50, n_users) * np.random.binomial(1, 0.1, n_users)
}

# Simulate improvement in new model
simulation_data['conversion'] = [
  1 if np.random.random() < (0.12 if v == 'new_model' else 0.10) else 0
  for v in simulation_data['variant']
]

metrics_df = pd.DataFrame(simulation_data)
analyze_experiment('ranking_model_v2', metrics_df)`,
      explanation: 'Production A/B testing framework with deterministic user assignment using hashing, experiment configuration management, exposure logging, metric tracking, and statistical analysis. Shows how to integrate A/B testing into ML prediction endpoints and analyze results.'
    }
  ],
  interviewQuestions: [
    {
      question: 'How do you calculate the required sample size for an A/B test? What factors affect it?',
      answer: `Sample size calculation requires: (1) Significance level (α, typically 0.05), (2) Power (1-β, typically 0.8), (3) Effect size (minimum detectable difference), (4) Baseline conversion rate. Use power analysis formulas or tools. Larger samples needed for: smaller effect sizes, lower baseline rates, higher confidence requirements. Consider practical constraints: traffic volume, test duration, and business impact of longer testing periods.`
    },
    {
      question: 'Explain the "peeking problem" in A/B testing and how to avoid it.',
      answer: `Peeking problem occurs when repeatedly checking test results and stopping early upon seeing significance, inflating Type I error rates. This happens because multiple comparisons increase false positive probability. Avoid by: (1) Pre-defining sample size and test duration, (2) Using sequential testing methods with adjusted significance levels, (3) Implementing proper stopping rules, (4) Using Bayesian methods, (5) Setting up automated analysis at predetermined intervals.`
    },
    {
      question: 'What is Sample Ratio Mismatch (SRM) and why is it important to check for it?',
      answer: `SRM occurs when observed traffic split differs significantly from expected allocation (e.g., expecting 50/50 but getting 48/52). Indicates potential issues: biased randomization, implementation bugs, or data collection problems. Important because it can invalidate test results and indicate systematic biases. Check using chi-square tests and investigate causes: user eligibility criteria, filtering logic, or technical implementation issues.`
    },
    {
      question: 'How would you handle A/B testing when there are network effects between users?',
      answer: `Network effects violate the stable unit treatment value assumption (SUTVA). Solutions: (1) Cluster randomization - randomize by groups (cities, schools) rather than individuals, (2) Ego-cluster randomization - randomize user's network neighborhood, (3) Time-based switching - switch entire platform periodically, (4) Use synthetic control methods, (5) Model network structure explicitly. Choose method based on network density and business context.`
    },
    {
      question: 'When would you choose a multi-armed bandit approach over traditional A/B testing?',
      answer: `Choose bandits when: (1) Exploration cost is high (losing conversions to inferior variants), (2) Need dynamic optimization during test, (3) Many variants to test simultaneously, (4) Prior beliefs about variant performance exist, (5) Traffic allocation can be adjusted based on performance. Traditional A/B testing better for: definitive statistical conclusions, regulatory requirements, or when equal sample sizes are needed for power calculations.`
    },
    {
      question: 'How do you ensure consistent variant assignment for the same user across multiple sessions?',
      answer: `Use deterministic hashing based on stable user identifiers (user ID, device ID). Implementation: hash(user_id + experiment_name) % 100 to assign to buckets. Considerations: (1) Handle logged-out users with device IDs or cookies, (2) Account for user ID changes (account merging), (3) Use salt values to prevent correlation between experiments, (4) Implement consistent assignment across platforms/services, (5) Handle edge cases like new user registration during test.`
    }
  ],
  quizQuestions: [
    {
      id: 'ab1',
      question: 'What does a p-value of 0.03 mean in an A/B test?',
      options: ['Variant B is 3% better', '3% chance of error', '3% chance of observing results if no true difference exists', 'Need 3% more data'],
      correctAnswer: 2,
      explanation: 'A p-value of 0.03 means there\'s a 3% probability of observing results as extreme as yours (or more extreme) if the null hypothesis (no difference) were true. Since 0.03 < 0.05, we reject the null hypothesis.'
    },
    {
      id: 'ab2',
      question: 'Why use user-level rather than request-level randomization?',
      options: ['It\'s faster', 'Prevents inconsistent user experience', 'Requires less data', 'It\'s more random'],
      correctAnswer: 1,
      explanation: 'User-level randomization ensures the same user always sees the same variant, providing a consistent experience. Request-level randomization could show different variants on different requests, confusing users and increasing variance.'
    },
    {
      id: 'ab3',
      question: 'What is statistical power in A/B testing?',
      options: ['Sample size', 'Probability of detecting a true effect', 'Significance level', 'Effect size'],
      correctAnswer: 1,
      explanation: 'Statistical power (1-β) is the probability of detecting a true effect when it exists (avoiding Type II error). Higher power requires larger sample sizes. Typically aim for 80% power.'
    }
  ]
};
