import { Topic } from '../../../types';

export const featureEngineering: Topic = {
  id: 'feature-engineering',
  title: 'Feature Engineering',
  category: 'ml-systems',
  description: 'Transforming raw data into useful features for ML models',
  content: `
    <h2>Feature Engineering: The Art of Representation</h2>
    
    <p>"Coming up with features is difficult, time-consuming, requires expert knowledge. 'Applied machine learning' is basically feature engineering." — Andrew Ng's observation captures a fundamental truth: the quality of features often matters more than the choice of algorithm. Feature engineering is the process of transforming raw data into representations that expose the underlying patterns to machine learning algorithms. It's where domain expertise meets data science, where creativity meets systematic analysis.</p>

    <p>Consider predicting house prices: raw data includes address, transaction date, square footage. But powerful features emerge through engineering: price per square foot (ratio), years since construction (transformation), neighborhood median income (aggregation), proximity to schools and transit (geospatial). These engineered features capture relationships that help models learn more effectively than raw inputs alone.</p>

    <h3>Why Feature Engineering Matters</h3>

    <p><strong>Performance multiplier:</strong> Well-engineered features can improve model accuracy by 10-50% or more, often exceeding gains from algorithm choice or hyperparameter tuning. A simple linear model with great features frequently outperforms a complex neural network with poor features. Features determine what the model can learn—no algorithm can discover patterns not represented in the input.</p>

    <p><strong>Model efficiency:</strong> Good features reduce the need for model complexity. When features clearly separate classes or capture relationships, simpler models suffice, leading to faster training, easier deployment, and better interpretability. This principle drove much of classical machine learning's success before deep learning.</p>

    <p><strong>Generalization:</strong> Features that capture true underlying relationships rather than spurious correlations improve generalization to new data. Domain-informed features encode prior knowledge, reducing the model's dependence on learning everything from data alone.</p>

    <p><strong>Interpretability:</strong> Meaningful features make models interpretable. When features correspond to domain concepts (customer lifetime value, seasonal demand, risk score), stakeholders understand model behavior and trust predictions.</p>

    <h3>The Feature Landscape: Types and Structures</h3>

    <h4>Numerical Features: Continuous and Discrete</h4>

    <p>Numerical features are the workhorses of machine learning—continuous values like age, price, temperature; discrete counts like number of purchases, page views, errors. These require careful consideration of scale, distribution, and relationships. A 30-year-old and a 60-year-old differ by 30 years, but is someone earning $30K and someone earning $300K truly ten times different? The answer depends on context—sometimes linear scale works, often logarithmic or other transformations reveal better patterns.</p>

    <p><strong>Ratios and rates</strong> create powerful derived features: price per square foot normalizes for size, click-through rate normalizes for impressions, velocity captures distance over time. These ratios often capture efficiency, density, or relative relationships more informative than raw values.</p>

    <h4>Categorical Features: Nominal and Ordinal</h4>

    <p>Categorical features represent discrete groups or labels. <strong>Nominal</strong> categories have no inherent order—colors (red, blue, green), countries, product categories. <strong>Ordinal</strong> categories have meaningful ordering—education levels (high school < bachelor < master < PhD), satisfaction ratings (poor < fair < good < excellent). The distinction matters for encoding: ordinal features can use ordered integers, nominal features need encoding that doesn't impose false order.</p>

    <p><strong>High cardinality</strong> challenges arise with features like ZIP codes (40,000+ values), product IDs (millions), user IDs. One-hot encoding explodes dimensionality. Solutions include target encoding (replace category with target mean), embedding learning, frequency encoding, or hierarchical grouping.</p>

    <h4>Datetime Features: Unlocking Temporal Patterns</h4>

    <p>Timestamps are deceptively simple—they contain rich structure. A single datetime can yield dozens of features: year, month, day, hour, minute, day of week, quarter, week of year. Binary indicators capture special conditions: is_weekend, is_holiday, is_business_hours. Time-based calculations reveal patterns: days_since_account_creation, time_until_deadline, recency_of_last_purchase.</p>

    <p><strong>Cyclic encoding</strong> handles circularity: December (12) and January (1) are adjacent despite distant numeric values. Transform monthly cycles with sin(2π×month/12) and cos(2π×month/12), creating continuous representations that respect periodicity. Similarly for days of week, hours of day.</p>

    <h4>Text Features: From Words to Vectors</h4>

    <p>Text requires extraction into numerical form. Start simple: length (character count, word count), readability scores, presence of keywords. <strong>Bag-of-words</strong> represents documents as word frequency vectors—losing order but capturing content. <strong>TF-IDF</strong> (Term Frequency-Inverse Document Frequency) weights words by informativeness, downweighting common terms. <strong>N-grams</strong> (bigrams, trigrams) capture local context.</p>

    <p>Modern approaches use <strong>embeddings</strong>: Word2Vec, GloVe, or contextual embeddings from BERT that encode semantic meaning in dense vectors. Pre-trained language models provide ready-made features capturing years of training on billions of words.</p>

    <h3>Feature Engineering Techniques</h3>

    <h4>Feature Transformation: Revealing Hidden Structure</h4>

    <p><strong>Scaling</strong> addresses different feature magnitudes. Min-Max scaling maps values to [0,1] range, preserving zero and relationships but sensitive to outliers. Standardization (z-score) transforms to mean=0, std=1, making features comparable and helping gradient-based algorithms converge faster. Robust scaling uses median and interquartile range, resisting outlier influence.</p>

    <p><strong>Mathematical transformations</strong> reshape distributions. Log transformation, log(x+1), compresses right-skewed distributions (income, sales, population) into more symmetric forms. Square root similarly reduces skew. Box-Cox transformation automatically finds optimal power transform. These help models that assume normality (linear regression) and improve interpretability.</p>

    <p><strong>Polynomial features</strong> capture non-linear relationships. From features x₁ and x₂, create x₁², x₂², x₁×x₂. A linear model with polynomial features can learn curves and interactions, becoming a polynomial regression. But dimensionality explodes quickly—two features to degree 3 yields 9 features; ten features yield 285.</p>

    <h4>Feature Interactions: Capturing Synergies</h4>

    <p>Real-world relationships aren't always additive. The effect of advertising spend depends on product quality; the impact of education level varies by job market. <strong>Interaction features</strong> capture these synergies by combining features: multiply, divide, or apply domain-specific formulas.</p>

    <p>Domain knowledge guides interactions: BMI = weight/height², total_cost = price×quantity, profit_margin = (revenue-cost)/revenue. These formulas encode relationships that models would struggle to discover from raw features alone. Systematically generating all pairs (x_i × x_j) can uncover unexpected interactions but requires feature selection to manage dimensionality.</p>

    <h4>Aggregation Features: Learning from Groups</h4>

    <p><strong>Group statistics</strong> extract patterns from subsets. For each customer, compute avg_purchase_by_city (average purchases for their city), max_price_in_category (maximum price in product category). These features provide context—how does this instance compare to its group?</p>

    <p><strong>Rolling window features</strong> for time series capture recent trends: 7-day moving average of sales, 30-day rolling standard deviation of stock prices, 90-day cumulative sum. <strong>Lag features</strong> use past values: previous day's temperature, last week's traffic, last month's revenue. These expose temporal patterns and autocorrelation.</p>

    <h4>Binning and Discretization: Simplifying Complexity</h4>

    <p>Converting continuous features to categorical bins can help. Age → age groups (0-18, 18-35, 36-60, 60+), income → income brackets, scores → letter grades. Benefits: captures non-linear patterns (risk changes at thresholds), reduces overfitting to outliers, creates interpretable groups. Methods include equal-width bins (same range), equal-frequency bins (same count), or custom boundaries based on domain knowledge.</p>

    <p>But binning loses information—people aged 28 and 32 treated identically if in same bin. Use cautiously, typically when domain suggests natural boundaries or for interpretability.</p>

    <h3>Feature Selection: Finding the Signal</h3>

    <p>More features aren't always better. Irrelevant features add noise, increase computational cost, cause overfitting, and reduce interpretability. Feature selection identifies the most informative subset.</p>

    <h4>Filter Methods: Fast Statistical Tests</h4>

    <p>Filter methods evaluate features independently using statistical measures, before modeling. <strong>Variance threshold</strong> removes features with near-zero variance—if a feature is almost constant, it provides little information. <strong>Correlation analysis</strong> identifies highly correlated features; remove one from each redundant pair. <strong>Chi-square test</strong> (categorical features) and <strong>mutual information</strong> (any features) measure dependence between features and target.</p>

    <p>Filters are fast and model-agnostic but ignore feature interactions and don't account for specific model behavior.</p>

    <h4>Wrapper Methods: Optimizing Model Performance</h4>

    <p>Wrapper methods use model performance to evaluate feature subsets. <strong>Forward selection</strong> starts with no features, iteratively adding the one that most improves performance. <strong>Backward elimination</strong> starts with all features, iteratively removing the least useful. <strong>Recursive Feature Elimination (RFE)</strong> trains a model, ranks features by importance, removes the weakest, and repeats.</p>

    <p>Wrappers find better feature sets for specific models but are computationally expensive—each iteration requires training and evaluating a model.</p>

    <h4>Embedded Methods: Selection During Training</h4>

    <p><strong>L1 regularization (Lasso)</strong> penalizes the absolute value of coefficients, shrinking some to exactly zero, performing automatic feature selection. Features with non-zero coefficients are selected. <strong>Tree-based feature importance</strong> ranks features by how much they reduce impurity (Gini, entropy) when used for splits. Random forests provide robust importance scores by averaging across trees.</p>

    <p><strong>SHAP values</strong> (SHapley Additive exPlanations) quantify each feature's contribution to predictions using game theory, providing consistent and interpretable importance scores.</p>

    <h4>Choosing the Right Feature Selection Method</h4>

    <p><strong>Decision guide for feature selection:</strong></p>

    <p><strong>Use Filter Methods when:</strong> You have very high-dimensional data (thousands of features) and need fast initial screening. You want model-agnostic selection before trying different algorithms. Computational resources are limited. Example: genomics data with 20,000 features—use correlation or mutual information to reduce to manageable size.</p>

    <p><strong>Use Wrapper Methods when:</strong> You have moderate dimensionality (<100 features) and want optimal feature subset for a specific model. Model performance is the primary concern and computational cost is acceptable. You need to account for feature interactions. Example: selecting best 20 features from 50 candidates for a specific production model.</p>

    <p><strong>Use Embedded Methods when:</strong> You want feature selection integrated with model training for efficiency. You're using tree-based models (natural feature importance) or linear models with regularization. You need to balance selection quality with computational efficiency. Example: using Random Forest feature importance during model development.</p>

    <p><strong>Common Mistakes to Avoid</h3>

    <p><strong>Creating features from test data:</strong> The deadliest mistake. Never compute statistics (mean, max, standard deviation) using test data—this leaks information from your test set into training. Always compute feature transformations on training data only, then apply those same transformations to test data.</p>

    <p><strong>Target leakage through aggregations:</strong> Be careful with group statistics. If predicting user churn, "average_churn_rate_by_city" computed on all data includes the target variable. Compute such features excluding the current row or use only historical data (users who churned before this timestamp).</p>

    <p><strong>Look-ahead bias in time series:</strong> Using future information to predict the past. When creating lag features, ensure you only use data from before the prediction timestamp. "Next_day_sales" can't be a feature to predict today's inventory needs.</p>

    <p><strong>Ignoring missing value patterns:</strong> Treating all missing values the same way is naive. Sometimes missingness is informative (high earners refuse income questions). Create "is_missing" indicator features before imputing—preserve the signal in the missingness pattern itself.</p>

    <p><strong>Over-engineering features:</strong> Creating hundreds of complex interaction features without validation often leads to overfitting. The model memorizes noise rather than learning patterns. Use cross-validation to verify each feature actually helps generalization.</p>

    <p><strong>Feature selection on entire dataset:</strong> Running feature selection on combined train+test data leaks information. Split data first, then perform feature selection only on training set. Selected features are then used on test set.</p>

    <h3>Quick reference:</strong> Start with filter methods for initial reduction (thousands → hundreds). Apply wrapper/embedded methods for final selection (hundreds → tens). Use domain knowledge throughout—no statistical method beats expert insight about which features actually matter.</p>

    <h3>Common Mistakes to Avoid</h3>

    <p><strong>Start simple:</strong> Begin with basic transformations and fundamental features before elaborate engineering. Simple baselines establish performance floor and reveal what additional engineering is needed.</p>

    <p><strong>Leverage domain expertise:</strong> Subject matter experts know which relationships matter. A financial analyst understands debt-to-equity ratios; a doctor recognizes vital sign thresholds. Collaborate to create meaningful features.</p>

    <p><strong>Avoid data leakage:</strong> The cardinal sin—using information that wouldn't be available at prediction time. Target leakage occurs when features contain the target or future information. Example: using total_annual_spend to predict monthly purchases when annual spend isn't known monthly. Use only data from before the prediction timestamp.</p>

    <p><strong>Validate carefully:</strong> Engineer features on training data only, then apply the same transformations to validation/test data. Cross-validation ensures features generalize. Time-based splits prevent leakage in temporal data.</p>

    <p><strong>Document everything:</strong> Maintain clear definitions for engineered features. As pipelines grow complex, documentation prevents errors and enables collaboration.</p>

    <h3>Automated Feature Engineering</h3>

    <p>Tools automate tedious manual work. <strong>Featuretools</strong> performs deep feature synthesis, automatically creating aggregation and transformation features from relational data. <strong>tsfresh</strong> extracts hundreds of time series features. <strong>AutoML</strong> systems include automated feature engineering.</p>

    <p><strong>Deep learning</strong> learns features automatically through neural network layers, reducing manual engineering needs for images, audio, and text. But domain-informed features still help—even deep learning benefits from good input representations.</p>

    <p>Feature engineering remains part art, part science. It requires creativity, domain knowledge, and systematic experimentation. Master it, and you master a superpower of applied machine learning.</p>
  `,
  codeExamples: [
    {
      language: 'Python',
      code: `import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Sample dataset
df = pd.DataFrame({
  'age': [25, 35, 45, 55, 65],
  'income': [30000, 50000, 70000, 90000, 110000],
  'education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor'],
  'city': ['NYC', 'LA', 'NYC', 'SF', 'LA'],
  'purchase_date': pd.date_range('2024-01-01', periods=5, freq='M'),
  'num_purchases': [3, 7, 5, 10, 8]
})

# === 1. NUMERICAL FEATURES ===

# Log transformation for skewed data
df['log_income'] = np.log1p(df['income'])

# Polynomial features
df['age_squared'] = df['age'] ** 2

# Binning
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Senior'])

# Interaction features
df['income_per_purchase'] = df['income'] / (df['num_purchases'] + 1)  # Add 1 to avoid division by zero

# === 2. DATETIME FEATURES ===

df['purchase_year'] = df['purchase_date'].dt.year
df['purchase_month'] = df['purchase_date'].dt.month
df['purchase_quarter'] = df['purchase_date'].dt.quarter
df['purchase_day_of_week'] = df['purchase_date'].dt.dayofweek
df['is_weekend'] = df['purchase_day_of_week'].isin([5, 6]).astype(int)

# Days since first purchase
df['days_since_first_purchase'] = (df['purchase_date'] - df['purchase_date'].min()).dt.days

# Cyclic encoding for month (to capture circular nature)
df['month_sin'] = np.sin(2 * np.pi * df['purchase_month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['purchase_month'] / 12)

# === 3. CATEGORICAL ENCODING ===

# One-hot encoding for nominal categories
ohe = OneHotEncoder(sparse_output=False, drop='first')
city_encoded = ohe.fit_transform(df[['city']])
city_df = pd.DataFrame(city_encoded, columns=['city_LA', 'city_SF'])
df = pd.concat([df, city_df], axis=1)

# Label encoding for ordinal categories
education_order = ['High School', 'Bachelor', 'Master', 'PhD']
df['education_encoded'] = df['education'].map({edu: i for i, edu in enumerate(education_order)})

# Frequency encoding
city_freq = df['city'].value_counts(normalize=True).to_dict()
df['city_frequency'] = df['city'].map(city_freq)

# === 4. AGGREGATION FEATURES ===

# Group statistics
df['avg_purchases_by_city'] = df.groupby('city')['num_purchases'].transform('mean')
df['max_income_by_city'] = df.groupby('city')['income'].transform('max')

print("\\n=== Engineered Features ===")
print(df.head())
print(f"\\nOriginal features: {['age', 'income', 'education', 'city', 'purchase_date', 'num_purchases']}")
print(f"Total features after engineering: {df.shape[1]}")`,
      explanation: 'Comprehensive feature engineering: transformations, datetime features, categorical encoding, and aggregations.'
    },
    {
      language: 'Python',
      code: `import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Sample dataset
np.random.seed(42)
n_samples = 1000
X = pd.DataFrame({
  'feature_1': np.random.randn(n_samples),
  'feature_2': np.random.randn(n_samples),
  'feature_3': np.random.randn(n_samples),
  'feature_4': np.random.randn(n_samples) * 0.01,  # Low variance
  'feature_5': np.random.randn(n_samples),
})
X['feature_6'] = X['feature_1'] + np.random.randn(n_samples) * 0.1  # Highly correlated with feature_1

y = (X['feature_1'] + X['feature_2'] > 0).astype(int)

# === 1. VARIANCE THRESHOLD ===
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
X_high_var = selector.fit_transform(X)
print(f"Features after variance threshold: {X_high_var.shape[1]} (removed low variance)")

# === 2. CORRELATION-BASED SELECTION ===

# Find highly correlated features
corr_matrix = X.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
print(f"\\nHighly correlated features to remove: {high_corr_features}")

X_uncorr = X.drop(columns=high_corr_features)

# === 3. UNIVARIATE FEATURE SELECTION ===

# Chi-square test (for non-negative features)
X_positive = X - X.min()
selector = SelectKBest(score_func=chi2, k=3)
X_chi2 = selector.fit_transform(X_positive, y)
selected_features_chi2 = X.columns[selector.get_support()]
print(f"\\nTop 3 features by chi-square: {list(selected_features_chi2)}")

# Mutual information
selector = SelectKBest(score_func=mutual_info_classif, k=3)
X_mi = selector.fit_transform(X, y)
selected_features_mi = X.columns[selector.get_support()]
print(f"Top 3 features by mutual information: {list(selected_features_mi)}")

# === 4. RECURSIVE FEATURE ELIMINATION (RFE) ===

estimator = LogisticRegression(max_iter=1000)
selector = RFE(estimator, n_features_to_select=3, step=1)
selector.fit(X, y)
selected_features_rfe = X.columns[selector.support_]
print(f"\\nTop 3 features by RFE: {list(selected_features_rfe)}")

# === 5. TREE-BASED FEATURE IMPORTANCE ===

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

feature_importance = pd.DataFrame({
  'feature': X.columns,
  'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\\nFeature importance from Random Forest:")
print(feature_importance)

# Select top K features
k = 3
top_features = feature_importance.head(k)['feature'].values
print(f"\\nTop {k} features: {list(top_features)}")

# === 6. L1 REGULARIZATION (LASSO) ===

from sklearn.linear_model import LassoCV

# Standardize first
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_scaled, y)

# Features with non-zero coefficients
lasso_features = X.columns[lasso.coef_ != 0]
print(f"\\nFeatures selected by Lasso: {list(lasso_features)}")
print(f"Lasso coefficients: {dict(zip(X.columns, lasso.coef_))}")`,
      explanation: 'Feature selection methods: variance threshold, correlation, univariate selection, RFE, tree importance, and L1 regularization.'
    }
  ],
  interviewQuestions: [
    {
      question: 'What is feature engineering and why is it important?',
      answer: `Feature engineering transforms raw data into meaningful representations that help machine learning models perform better. It's crucial because: (1) Models learn from features, not raw data - good features dramatically improve performance, (2) Domain knowledge can be incorporated through engineered features, (3) Well-designed features improve model interpretability and generalization, (4) Proper feature engineering can reduce the need for complex models, and (5) It often has more impact on model performance than algorithm choice.`
    },
    {
      question: 'Explain the difference between one-hot encoding and label encoding.',
      answer: `Label encoding assigns unique integers to categorical values (Red=0, Blue=1, Green=2), creating ordinal relationships where none exist. One-hot encoding creates binary columns for each category ([1,0,0] for Red), avoiding false ordinal relationships. Use label encoding for ordinal data with natural ordering; use one-hot encoding for nominal data. One-hot can cause dimensionality issues with high cardinality features.`
    },
    {
      question: 'How would you handle datetime features?',
      answer: `Extract meaningful components: year, month, day, hour, day_of_week, quarter, is_weekend, is_holiday. Create time-based features: days_since_event, time_to_next_event. Use cyclic encoding (sin/cos) for circular time features to capture periodicity. Consider time zones for global data. Extract domain-specific patterns like business hours, seasons, or fiscal periods relevant to your problem.`
    },
    {
      question: 'What is target leakage and how do you avoid it?',
      answer: `Target leakage occurs when features contain information from the future or directly about the target variable that wouldn't be available at prediction time. Avoid it by: (1) Using only data available before the prediction timestamp, (2) Checking feature creation logic for future information, (3) Validating with proper time-based splits, (4) Being careful with aggregated features that might include target information, and (5) Conducting temporal data analysis to ensure causality.`
    },
    {
      question: 'Compare filter, wrapper, and embedded feature selection methods.',
      answer: `Filter methods evaluate features independently of models using statistical measures (correlation, mutual information) - fast but ignore feature interactions. Wrapper methods evaluate feature subsets using model performance (RFE, forward/backward selection) - more accurate but computationally expensive. Embedded methods perform selection during model training (L1 regularization, tree feature importance) - balanced approach combining efficiency with model-aware selection.`
    },
    {
      question: 'How would you create features for a time series problem?',
      answer: `Create lag features (values from previous time steps), rolling statistics (moving averages, rolling std), seasonal features (yearly/monthly patterns), trend features (linear/polynomial trends), time-based features (day of week, hour), change features (differences, percentage changes), and domain-specific indicators. Consider window sizes carefully and ensure no data leakage when creating rolling features.`
    }
  ],
  quizQuestions: [
    {
      id: 'fe1',
      question: 'What is the purpose of feature engineering?',
      options: ['Increase dataset size', 'Transform data into better representations', 'Remove outliers', 'Balance classes'],
      correctAnswer: 1,
      explanation: 'Feature engineering transforms raw data into features that better represent the underlying problem, improving model performance and interpretability.'
    },
    {
      id: 'fe2',
      question: 'When should you use one-hot encoding vs label encoding?',
      options: ['Always use one-hot', 'One-hot for nominal, label for ordinal', 'Label for nominal, one-hot for ordinal', 'No difference'],
      correctAnswer: 1,
      explanation: 'One-hot encoding is for nominal categories (no order) to avoid imposing false ordering. Label encoding is for ordinal categories (with natural order) where the numeric order is meaningful.'
    },
    {
      id: 'fe3',
      question: 'What is target leakage?',
      options: ['Missing target values', 'Using future information to predict', 'Imbalanced targets', 'Wrong target encoding'],
      correctAnswer: 1,
      explanation: 'Target leakage occurs when features contain information about the target that would not be available at prediction time, leading to overly optimistic results that don\'t generalize.'
    }
  ]
};
