import { Topic } from '../../types';

export const mlSystemsTopics: Record<string, Topic> = {
  'feature-engineering': {
    id: 'feature-engineering',
    title: 'Feature Engineering',
    category: 'ml-systems',
    description: 'Transforming raw data into useful features for ML models',
    content: `
      <h2>Feature Engineering</h2>
      <p>Feature engineering is the process of transforming raw data into features that better represent the underlying problem to improve model performance. It's often cited as the most important factor in determining ML model success.</p>

      <h3>Why Feature Engineering Matters</h3>
      <ul>
        <li><strong>Model performance:</strong> Good features improve accuracy dramatically</li>
        <li><strong>Interpretability:</strong> Meaningful features aid understanding</li>
        <li><strong>Generalization:</strong> Well-engineered features transfer better</li>
        <li><strong>Efficiency:</strong> Reduces need for complex models</li>
        <li><strong>Domain knowledge:</strong> Incorporate expert insights</li>
      </ul>

      <h3>Types of Features</h3>

      <h4>Numerical Features</h4>
      <ul>
        <li><strong>Continuous:</strong> Age, price, temperature</li>
        <li><strong>Discrete:</strong> Count of events, number of items</li>
        <li><strong>Ratios:</strong> Price per square foot, clicks per impression</li>
      </ul>

      <h4>Categorical Features</h4>
      <ul>
        <li><strong>Nominal:</strong> Color, category (no order)</li>
        <li><strong>Ordinal:</strong> Low/Medium/High (ordered)</li>
        <li><strong>Binary:</strong> True/False, Yes/No</li>
      </ul>

      <h4>Datetime Features</h4>
      <ul>
        <li><strong>Year, month, day, hour</strong></li>
        <li><strong>Day of week, quarter</strong></li>
        <li><strong>Is weekend, is holiday</strong></li>
        <li><strong>Time since event</strong></li>
        <li><strong>Cyclic encoding:</strong> Sin/cos for circular time</li>
      </ul>

      <h4>Text Features</h4>
      <ul>
        <li><strong>Length:</strong> Character count, word count</li>
        <li><strong>Bag of words, TF-IDF</strong></li>
        <li><strong>N-grams:</strong> Bi-grams, tri-grams</li>
        <li><strong>Embeddings:</strong> Word2Vec, BERT embeddings</li>
      </ul>

      <h3>Feature Engineering Techniques</h3>

      <h4>1. Feature Transformation</h4>

      <h5>Scaling</h5>
      <ul>
        <li><strong>Min-Max:</strong> Scale to [0, 1]</li>
        <li><strong>Standardization:</strong> Mean=0, Std=1</li>
        <li><strong>Robust scaling:</strong> Use median and IQR</li>
      </ul>

      <h5>Encoding</h5>
      <ul>
        <li><strong>One-hot encoding:</strong> Binary columns for categories</li>
        <li><strong>Label encoding:</strong> Map categories to integers</li>
        <li><strong>Target encoding:</strong> Replace with target mean</li>
        <li><strong>Frequency encoding:</strong> Replace with frequency</li>
      </ul>

      <h5>Mathematical Transforms</h5>
      <ul>
        <li><strong>Log transform:</strong> log(x) for skewed distributions</li>
        <li><strong>Square root:</strong> Reduce right skew</li>
        <li><strong>Box-Cox:</strong> Find optimal power transformation</li>
        <li><strong>Polynomial features:</strong> x², x³, etc.</li>
      </ul>

      <h4>2. Feature Interactions</h4>
      <ul>
        <li><strong>Multiplication:</strong> price × quantity</li>
        <li><strong>Division:</strong> ratio of two features</li>
        <li><strong>Polynomial interactions:</strong> x₁ × x₂, x₁²</li>
        <li><strong>Domain-specific:</strong> BMI = weight / height²</li>
      </ul>

      <h4>3. Aggregation Features</h4>
      <ul>
        <li><strong>Groupby statistics:</strong> Mean, median, std per group</li>
        <li><strong>Rolling windows:</strong> Moving average, moving sum</li>
        <li><strong>Lag features:</strong> Previous values (time series)</li>
        <li><strong>Expanding windows:</strong> Cumulative statistics</li>
      </ul>

      <h4>4. Binning/Discretization</h4>
      <ul>
        <li><strong>Equal width:</strong> Same size bins</li>
        <li><strong>Equal frequency:</strong> Same count in each bin</li>
        <li><strong>Custom bins:</strong> Domain knowledge boundaries</li>
        <li><strong>Example:</strong> Age → Age groups (0-18, 18-35, etc.)</li>
      </ul>

      <h3>Feature Selection</h3>

      <h4>Filter Methods</h4>
      <ul>
        <li><strong>Correlation:</strong> Remove highly correlated features</li>
        <li><strong>Variance:</strong> Remove low-variance features</li>
        <li><strong>Chi-square test:</strong> For categorical features</li>
        <li><strong>Mutual information:</strong> Measure dependence</li>
      </ul>

      <h4>Wrapper Methods</h4>
      <ul>
        <li><strong>Forward selection:</strong> Add features iteratively</li>
        <li><strong>Backward elimination:</strong> Remove features iteratively</li>
        <li><strong>Recursive feature elimination (RFE)</strong></li>
      </ul>

      <h4>Embedded Methods</h4>
      <ul>
        <li><strong>L1 regularization (Lasso):</strong> Sparse feature selection</li>
        <li><strong>Tree-based importance:</strong> Feature importance from trees</li>
        <li><strong>SHAP values:</strong> Feature contribution analysis</li>
      </ul>

      <h3>Best Practices</h3>
      <ul>
        <li><strong>Start simple:</strong> Begin with basic features</li>
        <li><strong>Domain knowledge:</strong> Leverage expert insights</li>
        <li><strong>Iterative:</strong> Feature engineering is iterative</li>
        <li><strong>Cross-validation:</strong> Validate on holdout data</li>
        <li><strong>Avoid leakage:</strong> Don't use future information</li>
        <li><strong>Document:</strong> Track feature definitions</li>
      </ul>

      <h3>Automated Feature Engineering</h3>
      <ul>
        <li><strong>Featuretools:</strong> Automated deep feature synthesis</li>
        <li><strong>tsfresh:</strong> Time series features</li>
        <li><strong>AutoFeat:</strong> Linear prediction models with automatic features</li>
        <li><strong>Deep learning:</strong> Learn features automatically</li>
      </ul>
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
  },

  'data-preprocessing-normalization': {
    id: 'data-preprocessing-normalization',
    title: 'Data Preprocessing and Normalization',
    category: 'ml-systems',
    description: 'Preparing and scaling data for machine learning',
    content: `
      <h2>Data Preprocessing and Normalization</h2>
      <p>Data preprocessing and normalization are crucial steps that prepare raw data for machine learning models, ensuring features are on comparable scales and handling data quality issues.</p>

      <h3>Why Preprocessing Matters</h3>
      <ul>
        <li><strong>Model convergence:</strong> Scaled features help gradient descent converge faster</li>
        <li><strong>Numerical stability:</strong> Prevent overflow/underflow in calculations</li>
        <li><strong>Fair feature weighting:</strong> Features on different scales can dominate</li>
        <li><strong>Algorithm requirements:</strong> Some algorithms (SVM, k-NN) sensitive to scale</li>
        <li><strong>Regularization:</strong> L1/L2 penalties need comparable scales</li>
      </ul>

      <h3>Data Cleaning</h3>

      <h4>Missing Values</h4>
      <ul>
        <li><strong>Deletion:</strong> Remove rows/columns with missing data</li>
        <li><strong>Mean/median imputation:</strong> Fill with central tendency</li>
        <li><strong>Mode imputation:</strong> For categorical features</li>
        <li><strong>Forward/backward fill:</strong> For time series</li>
        <li><strong>Model-based:</strong> Predict missing values (KNN, regression)</li>
        <li><strong>Indicator variable:</strong> Add "is_missing" flag</li>
      </ul>

      <h4>Outliers</h4>
      <ul>
        <li><strong>Detection:</strong> Z-score, IQR, isolation forest</li>
        <li><strong>Removal:</strong> Drop extreme values (use carefully)</li>
        <li><strong>Capping:</strong> Clip to percentile thresholds</li>
        <li><strong>Transformation:</strong> Log transform to reduce impact</li>
        <li><strong>Separate model:</strong> Build model specifically for outliers</li>
      </ul>

      <h4>Duplicates</h4>
      <ul>
        <li><strong>Exact duplicates:</strong> Remove identical rows</li>
        <li><strong>Near duplicates:</strong> Use similarity measures</li>
        <li><strong>Time-based:</strong> Keep most recent in time series</li>
      </ul>

      <h3>Scaling Techniques</h3>

      <h4>1. Min-Max Scaling (Normalization)</h4>
      <p>x_scaled = (x - x_min) / (x_max - x_min)</p>
      <ul>
        <li><strong>Range:</strong> [0, 1] or custom [a, b]</li>
        <li><strong>Pros:</strong> Bounded, preserves zero values</li>
        <li><strong>Cons:</strong> Sensitive to outliers</li>
        <li><strong>Use case:</strong> Neural networks, bounded algorithms</li>
      </ul>

      <h4>2. Standardization (Z-score Normalization)</h4>
      <p>x_scaled = (x - μ) / σ</p>
      <ul>
        <li><strong>Mean:</strong> 0, Std: 1</li>
        <li><strong>Pros:</strong> Less sensitive to outliers than min-max</li>
        <li><strong>Cons:</strong> Unbounded</li>
        <li><strong>Use case:</strong> Linear models, SVM, PCA</li>
      </ul>

      <h4>3. Robust Scaling</h4>
      <p>x_scaled = (x - median) / IQR</p>
      <ul>
        <li><strong>Uses:</strong> Median and interquartile range</li>
        <li><strong>Pros:</strong> Robust to outliers</li>
        <li><strong>Cons:</strong> Less common, not for all algorithms</li>
        <li><strong>Use case:</strong> Data with many outliers</li>
      </ul>

      <h4>4. Max Abs Scaling</h4>
      <p>x_scaled = x / |x_max|</p>
      <ul>
        <li><strong>Range:</strong> [-1, 1]</li>
        <li><strong>Pros:</strong> Preserves sparsity, doesn't shift center</li>
        <li><strong>Use case:</strong> Sparse data</li>
      </ul>

      <h4>5. Unit Vector Scaling (Normalization)</h4>
      <p>x_scaled = x / ||x||</p>
      <ul>
        <li><strong>Makes:</strong> L2 norm = 1</li>
        <li><strong>Use case:</strong> Text data (TF-IDF), cosine similarity</li>
      </ul>

      <h3>Distribution Transforms</h3>

      <h4>Log Transform</h4>
      <ul>
        <li><strong>Formula:</strong> log(x + 1) or log(x)</li>
        <li><strong>Purpose:</strong> Reduce right skew</li>
        <li><strong>Use case:</strong> Income, prices, counts</li>
      </ul>

      <h4>Square Root / Cube Root</h4>
      <ul>
        <li><strong>Purpose:</strong> Moderate skew reduction</li>
        <li><strong>Benefit:</strong> Works with zeros (unlike log)</li>
      </ul>

      <h4>Box-Cox Transform</h4>
      <ul>
        <li><strong>Finds:</strong> Optimal power transformation</li>
        <li><strong>Formula:</strong> (x^λ - 1) / λ when λ ≠ 0</li>
        <li><strong>Constraint:</strong> Requires positive values</li>
      </ul>

      <h4>Yeo-Johnson Transform</h4>
      <ul>
        <li><strong>Like Box-Cox:</strong> But works with negative values</li>
        <li><strong>More flexible:</strong> Handles zeros and negatives</li>
      </ul>

      <h3>Best Practices</h3>

      <h4>Train-Test Split Order</h4>
      <ul>
        <li><strong>1. Split data first</strong></li>
        <li><strong>2. Fit scaler on training set only</strong></li>
        <li><strong>3. Transform both train and test with same scaler</strong></li>
        <li><strong>Why:</strong> Prevent data leakage from test set</li>
      </ul>

      <h4>When to Apply</h4>
      <ul>
        <li><strong>Before distance-based:</strong> k-NN, k-means, SVM</li>
        <li><strong>Before gradient descent:</strong> Neural networks, linear regression</li>
        <li><strong>Before regularization:</strong> Lasso, Ridge</li>
        <li><strong>Not needed:</strong> Tree-based models (scale-invariant)</li>
      </ul>

      <h4>Pipeline Integration</h4>
      <ul>
        <li><strong>Use sklearn Pipeline:</strong> Ensures correct order</li>
        <li><strong>ColumnTransformer:</strong> Different transforms per column</li>
        <li><strong>Save preprocessor:</strong> For production deployment</li>
      </ul>

      <h3>Special Cases</h3>

      <h4>Time Series</h4>
      <ul>
        <li><strong>Rolling statistics:</strong> Use expanding window for scaling</li>
        <li><strong>Avoid future leakage:</strong> Only use past data</li>
        <li><strong>Differencing:</strong> Make stationary</li>
      </ul>

      <h4>Sparse Data</h4>
      <ul>
        <li><strong>Preserve sparsity:</strong> Use MaxAbsScaler</li>
        <li><strong>Avoid:</strong> StandardScaler (densifies data)</li>
      </ul>

      <h4>Categorical Features</h4>
      <ul>
        <li><strong>Don't scale:</strong> One-hot encoded features (already 0/1)</li>
        <li><strong>Ordinal:</strong> Can scale if appropriate</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, PowerTransformer
)
from sklearn.model_selection import train_test_split

# Sample data with different scales
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.randint(18, 80, 100),
    'income': np.random.exponential(50000, 100),  # Skewed distribution
    'credit_score': np.random.normal(700, 50, 100),
    'num_purchases': np.random.poisson(5, 100)
})

# Add some missing values
data.loc[5:8, 'income'] = np.nan
data.loc[15:17, 'age'] = np.nan

print("Original data:")
print(data.head())
print(f"\\nMissing values:\\n{data.isnull().sum()}")
print(f"\\nData statistics:\\n{data.describe()}")

# === MISSING VALUE IMPUTATION ===

# Mean imputation for numerical features
data['income_imputed'] = data['income'].fillna(data['income'].mean())

# Median imputation (more robust to outliers)
data['age_imputed'] = data['age'].fillna(data['age'].median())

# Forward fill (for time series)
data['income_ffill'] = data['income'].fillna(method='ffill')

print(f"\\nAfter imputation:\\n{data[['income', 'income_imputed', 'age', 'age_imputed']].head(10)}")

# === OUTLIER DETECTION AND HANDLING ===

# Z-score method
z_scores = np.abs((data['income_imputed'] - data['income_imputed'].mean()) / data['income_imputed'].std())
outliers_z = z_scores > 3
print(f"\\nOutliers detected (z-score > 3): {outliers_z.sum()}")

# IQR method
Q1 = data['income_imputed'].quantile(0.25)
Q3 = data['income_imputed'].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = (data['income_imputed'] < (Q1 - 1.5 * IQR)) | (data['income_imputed'] > (Q3 + 1.5 * IQR))
print(f"Outliers detected (IQR method): {outliers_iqr.sum()}")

# Capping outliers at percentiles
data['income_capped'] = data['income_imputed'].clip(
    lower=data['income_imputed'].quantile(0.01),
    upper=data['income_imputed'].quantile(0.99)
)

# === SCALING TECHNIQUES ===

# Use only complete cases for scaling examples
clean_data = data[['age_imputed', 'income_capped', 'credit_score', 'num_purchases']].copy()

# Split data first (important!)
X_train, X_test = train_test_split(clean_data, test_size=0.2, random_state=42)

# 1. StandardScaler (Z-score normalization)
scaler_standard = StandardScaler()
X_train_standard = scaler_standard.fit_transform(X_train)
X_test_standard = scaler_standard.transform(X_test)  # Use fitted scaler

print(f"\\n=== StandardScaler ===")
print(f"Mean after scaling: {X_train_standard.mean(axis=0)}")
print(f"Std after scaling: {X_train_standard.std(axis=0)}")

# 2. MinMaxScaler
scaler_minmax = MinMaxScaler(feature_range=(0, 1))
X_train_minmax = scaler_minmax.fit_transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)

print(f"\\n=== MinMaxScaler ===")
print(f"Min after scaling: {X_train_minmax.min(axis=0)}")
print(f"Max after scaling: {X_train_minmax.max(axis=0)}")

# 3. RobustScaler (robust to outliers)
scaler_robust = RobustScaler()
X_train_robust = scaler_robust.fit_transform(X_train)
X_test_robust = scaler_robust.transform(X_test)

print(f"\\n=== RobustScaler ===")
print(f"Median after scaling: {np.median(X_train_robust, axis=0)}")

# 4. MaxAbsScaler (for sparse data)
scaler_maxabs = MaxAbsScaler()
X_train_maxabs = scaler_maxabs.fit_transform(X_train)

print(f"\\n=== MaxAbsScaler ===")
print(f"Range: [{X_train_maxabs.min():.2f}, {X_train_maxabs.max():.2f}]")

# === DISTRIBUTION TRANSFORMS ===

# Log transform for skewed income
income_log = np.log1p(data['income_capped'])

# Power transform (Box-Cox / Yeo-Johnson)
pt = PowerTransformer(method='yeo-johnson', standardize=True)
income_transformed = pt.fit_transform(data[['income_capped']])

print(f"\\n=== Distribution Transforms ===")
print(f"Original income skewness: {data['income_capped'].skew():.2f}")
print(f"Log-transformed skewness: {pd.Series(income_log).skew():.2f}")
print(f"Power-transformed skewness: {pd.Series(income_transformed.flatten()).skew():.2f}")`,
        explanation: 'Comprehensive preprocessing: missing value imputation, outlier handling, and various scaling techniques.'
      },
      {
        language: 'Python',
        code: `from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

# Sample dataset with mixed types
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.randint(18, 80, 1000),
    'income': np.random.exponential(50000, 1000),
    'education': np.random.choice(['HS', 'BS', 'MS', 'PhD'], 1000),
    'city': np.random.choice(['NYC', 'LA', 'SF'], 1000),
    'target': np.random.randint(0, 2, 1000)
})

# Add missing values
data.loc[np.random.choice(1000, 50), 'age'] = np.nan
data.loc[np.random.choice(1000, 50), 'income'] = np.nan

# Separate features and target
X = data.drop('target', axis=1)
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === BUILD PREPROCESSING PIPELINE ===

# Define numerical and categorical features
numerical_features = ['age', 'income']
categorical_features = ['education', 'city']

# Numerical pipeline: impute → scale
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: impute → encode
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine pipelines with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)

# Full pipeline: preprocessing → model
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# === TRAIN MODEL ===

print("Training model with preprocessing pipeline...")
full_pipeline.fit(X_train, y_train)

# Evaluate
train_score = full_pipeline.score(X_train, y_train)
test_score = full_pipeline.score(X_test, y_test)

print(f"Training accuracy: {train_score:.3f}")
print(f"Test accuracy: {test_score:.3f}")

# === INSPECT PREPROCESSING ===

# Transform data to see preprocessing results
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"\\nOriginal features: {X_train.shape[1]}")
print(f"After preprocessing: {X_train_processed.shape[1]}")
print(f"(One-hot encoding expanded categorical features)")

# Get feature names after preprocessing
num_feature_names = numerical_features
cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
all_feature_names = list(num_feature_names) + list(cat_feature_names)

print(f"\\nFeature names after preprocessing:")
for i, name in enumerate(all_feature_names[:10]):  # Show first 10
    print(f"  {i}: {name}")

# === SAVE PIPELINE FOR PRODUCTION ===

import joblib

# Save the entire pipeline
joblib.dump(full_pipeline, 'model_pipeline.pkl')
print("\\nPipeline saved to 'model_pipeline.pkl'")

# Load and use
loaded_pipeline = joblib.load('model_pipeline.pkl')

# Predict on new data (with same preprocessing)
new_data = pd.DataFrame({
    'age': [25, 45],
    'income': [45000, 85000],
    'education': ['BS', 'MS'],
    'city': ['NYC', 'SF']
})

predictions = loaded_pipeline.predict(new_data)
probabilities = loaded_pipeline.predict_proba(new_data)

print(f"\\nPredictions on new data: {predictions}")
print(f"Probabilities:\\n{probabilities}")`,
        explanation: 'Production-ready preprocessing pipeline with ColumnTransformer, handling mixed data types and integration with model training.'
      }
    ],
    interviewQuestions: [
      {
        question: 'What is the difference between normalization and standardization?',
        answer: `Normalization (min-max scaling) scales features to [0,1] range using (x-min)/(max-min), preserving original distribution shape but sensitive to outliers. Standardization (z-score) transforms to mean=0, std=1 using (x-μ)/σ, making features comparable across different scales and more robust to outliers. Use normalization when you need bounded values; use standardization when features have different units/scales.`
      },
      {
        question: 'Why is it important to fit the scaler only on training data?',
        answer: `Fitting scalers on entire dataset causes data leakage - test data statistics influence training, leading to overly optimistic performance estimates. Proper approach: fit scaler on training data, transform training data, then transform validation/test data using training statistics. This simulates real-world deployment where future data statistics are unknown. Violating this principle can significantly overestimate model performance.`
      },
      {
        question: 'How would you handle missing values in a dataset?',
        answer: `Strategies include: (1) Simple imputation - mean/median/mode for numerical/categorical features, (2) Forward/backward fill for time series, (3) Advanced imputation - KNN, iterative (MICE), or model-based imputation, (4) Domain-specific approaches using business logic, (5) Creating missing indicators as features, (6) Dropping rows/columns if missingness is minimal. Choice depends on missingness pattern, data size, and domain constraints.`
      },
      {
        question: 'When should you use RobustScaler vs StandardScaler?',
        answer: `Use RobustScaler when data contains significant outliers - it uses median and IQR instead of mean and standard deviation, making it less sensitive to extreme values. Use StandardScaler for normally distributed data without major outliers. RobustScaler preserves outlier relationships while reducing their scaling impact; StandardScaler assumes Gaussian distribution and can be skewed by outliers.`
      },
      {
        question: 'Why don\'t tree-based models require feature scaling?',
        answer: `Tree-based models make split decisions based on relative feature value comparisons, not absolute magnitudes. They ask "is feature X > threshold?" regardless of scale. Decision boundaries are axis-parallel and invariant to monotonic transformations. However, feature scaling can still help with: (1) feature importance interpretation, (2) ensemble methods that combine trees with other algorithms, and (3) regularization techniques.`
      },
      {
        question: 'How would you detect and handle outliers?',
        answer: `Detection methods: (1) Statistical - IQR, z-score, modified z-score, (2) Visualization - box plots, scatter plots, histograms, (3) Model-based - isolation forest, one-class SVM, (4) Domain knowledge. Handling strategies: (1) Remove if measurement errors, (2) Cap/winsorize to percentiles, (3) Transform using log/sqrt, (4) Use robust models, (5) Treat as separate class if meaningful, (6) Impute using robust statistics.`
      }
    ],
    quizQuestions: [
      {
        id: 'prep1',
        question: 'What is the correct order for train-test split and scaling?',
        options: ['Scale then split', 'Split then fit scaler on train only', 'Fit scaler on all data', 'Doesn\'t matter'],
        correctAnswer: 1,
        explanation: 'You must split first, then fit the scaler only on training data to prevent data leakage. Test data is transformed using the scaler fitted on training data.'
      },
      {
        id: 'prep2',
        question: 'Which scaler is most robust to outliers?',
        options: ['MinMaxScaler', 'StandardScaler', 'RobustScaler', 'MaxAbsScaler'],
        correctAnswer: 2,
        explanation: 'RobustScaler uses median and IQR instead of mean and standard deviation, making it robust to outliers that would otherwise heavily influence the scaling.'
      },
      {
        id: 'prep3',
        question: 'Which models require feature scaling?',
        options: ['Decision trees', 'Random forests', 'Neural networks and SVM', 'All models equally'],
        correctAnswer: 2,
        explanation: 'Distance-based algorithms (k-NN, SVM) and gradient descent-based algorithms (neural networks, linear regression) benefit from scaling. Tree-based models are scale-invariant.'
      }
    ]
  },

  'handling-imbalanced-data': {
    id: 'handling-imbalanced-data',
    title: 'Handling Imbalanced Data',
    category: 'ml-systems',
    description: 'Techniques for dealing with class imbalance in classification',
    content: `
      <h2>Handling Imbalanced Data</h2>
      <p>Imbalanced data occurs when classes are not represented equally in a dataset. This is common in fraud detection, medical diagnosis, and anomaly detection, where the minority class is often the most important.</p>

      <h3>The Problem</h3>
      <ul>
        <li><strong>Bias toward majority:</strong> Models predict majority class too often</li>
        <li><strong>Poor minority recall:</strong> Miss important minority cases</li>
        <li><strong>Misleading accuracy:</strong> 99% accuracy when 99% is one class</li>
        <li><strong>Gradient dominance:</strong> Majority class dominates loss</li>
      </ul>

      <h3>Evaluation Metrics for Imbalanced Data</h3>

      <h4>Don't Use Accuracy!</h4>
      <p>Accuracy is misleading for imbalanced data</p>

      <h4>Better Metrics</h4>
      <ul>
        <li><strong>Precision:</strong> TP / (TP + FP) - Of predicted positives, how many are correct?</li>
        <li><strong>Recall (Sensitivity):</strong> TP / (TP + FN) - Of actual positives, how many found?</li>
        <li><strong>F1-Score:</strong> Harmonic mean of precision and recall</li>
        <li><strong>F-beta Score:</strong> Weighted F-score (F2 emphasizes recall)</li>
        <li><strong>AUC-ROC:</strong> Area under ROC curve</li>
        <li><strong>AUC-PR:</strong> Area under Precision-Recall curve (better for severe imbalance)</li>
        <li><strong>Confusion Matrix:</strong> See all error types</li>
      </ul>

      <h3>Approaches to Handle Imbalance</h3>

      <h4>1. Data-Level Methods (Resampling)</h4>

      <h5>Oversampling (Increase Minority)</h5>
      <ul>
        <li><strong>Random oversampling:</strong> Duplicate minority samples</li>
        <li><strong>SMOTE:</strong> Synthetic Minority Over-sampling Technique</li>
        <li><strong>ADASYN:</strong> Adaptive Synthetic Sampling</li>
        <li><strong>Borderline-SMOTE:</strong> Focus on decision boundary</li>
      </ul>

      <h5>Undersampling (Reduce Majority)</h5>
      <ul>
        <li><strong>Random undersampling:</strong> Remove majority samples</li>
        <li><strong>Tomek links:</strong> Remove noisy majority samples</li>
        <li><strong>Edited Nearest Neighbors:</strong> Remove misclassified samples</li>
        <li><strong>NearMiss:</strong> Select majority samples near minority</li>
      </ul>

      <h5>Combined Methods</h5>
      <ul>
        <li><strong>SMOTE + Tomek:</strong> Oversample then clean</li>
        <li><strong>SMOTE + ENN:</strong> Oversample then remove noise</li>
      </ul>

      <h4>2. Algorithm-Level Methods</h4>

      <h5>Class Weights</h5>
      <ul>
        <li><strong>Weighted loss:</strong> Higher penalty for minority errors</li>
        <li><strong>sklearn:</strong> class_weight='balanced'</li>
        <li><strong>Custom weights:</strong> Inversely proportional to frequency</li>
      </ul>

      <h5>Threshold Moving</h5>
      <ul>
        <li><strong>Default:</strong> 0.5 threshold may not be optimal</li>
        <li><strong>Adjust:</strong> Find threshold that balances precision/recall</li>
        <li><strong>ROC curve:</strong> Choose point based on requirements</li>
      </ul>

      <h5>Cost-Sensitive Learning</h5>
      <ul>
        <li><strong>Misclassification costs:</strong> Assign different costs to errors</li>
        <li><strong>Example:</strong> False negative in fraud detection more costly</li>
      </ul>

      <h4>3. Ensemble Methods</h4>

      <h5>Easy Ensemble</h5>
      <ul>
        <li><strong>Multiple balanced subsets:</strong> Undersample majority multiple times</li>
        <li><strong>Train multiple models:</strong> One per subset</li>
        <li><strong>Aggregate:</strong> Combine predictions</li>
      </ul>

      <h5>Balanced Random Forest</h5>
      <ul>
        <li><strong>Each tree:</strong> Trained on balanced bootstrap sample</li>
        <li><strong>Built-in:</strong> imblearn.ensemble.BalancedRandomForestClassifier</li>
      </ul>

      <h5>Balanced Bagging</h5>
      <ul>
        <li><strong>Bootstrap samples:</strong> Balanced via resampling</li>
        <li><strong>Combines:</strong> Bagging with resampling</li>
      </ul>

      <h4>4. Anomaly Detection Approach</h4>
      <ul>
        <li><strong>When:</strong> Extreme imbalance (< 1% minority)</li>
        <li><strong>One-class SVM:</strong> Learn boundary of majority class</li>
        <li><strong>Isolation Forest:</strong> Detect anomalies</li>
        <li><strong>Autoencoders:</strong> Reconstruction error for anomalies</li>
      </ul>

      <h3>Best Practices</h3>
      <ul>
        <li><strong>Stratified splits:</strong> Maintain class ratio in train/test</li>
        <li><strong>Cross-validation:</strong> Use stratified k-fold</li>
        <li><strong>Try multiple approaches:</strong> No one-size-fits-all solution</li>
        <li><strong>Domain knowledge:</strong> Understand cost of errors</li>
        <li><strong>Start simple:</strong> Class weights before resampling</li>
        <li><strong>Validate on original:</strong> If using synthetic samples</li>
      </ul>

      <h3>When to Use What</h3>
      <ul>
        <li><strong>Moderate imbalance (1:10):</strong> Class weights, threshold tuning</li>
        <li><strong>High imbalance (1:100):</strong> SMOTE, ensemble methods</li>
        <li><strong>Severe imbalance (1:1000+):</strong> Anomaly detection</li>
        <li><strong>Large dataset:</strong> Undersampling okay</li>
        <li><strong>Small dataset:</strong> Oversampling preferred (don't lose data)</li>
      </ul>

      <h3>Pitfalls to Avoid</h3>
      <ul>
        <li><strong>Resampling before split:</strong> Causes data leakage</li>
        <li><strong>Over-reliance on accuracy:</strong> Use appropriate metrics</li>
        <li><strong>Ignoring domain:</strong> All errors may not be equal</li>
        <li><strong>Only synthetic data:</strong> May not reflect real distribution</li>
      </ul>
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
  },

  'model-deployment': {
    id: 'model-deployment',
    title: 'Model Deployment',
    category: 'ml-systems',
    description: 'Strategies and best practices for deploying ML models to production',
    content: `
      <h2>Model Deployment</h2>
      <p>Model deployment is the process of integrating a trained machine learning model into a production environment where it can accept input data and return predictions to end users or other systems.</p>

      <h3>Deployment Patterns</h3>

      <h4>1. Batch Prediction</h4>
      <p>Process large volumes of data offline and store predictions for later retrieval.</p>
      <h5>When to Use:</h5>
      <ul>
        <li>Predictions don't need to be real-time</li>
        <li>Input data arrives in batches (e.g., daily)</li>
        <li>Computationally expensive models</li>
        <li>Recommendation systems, email filtering</li>
      </ul>
      <h5>Advantages:</h5>
      <ul>
        <li>Can use more complex models</li>
        <li>Easier to implement and debug</li>
        <li>Better resource utilization</li>
        <li>Simpler infrastructure</li>
      </ul>

      <h4>2. Online (Real-time) Prediction</h4>
      <p>Generate predictions on-demand in response to requests with low latency requirements.</p>
      <h5>When to Use:</h5>
      <ul>
        <li>Immediate predictions needed (< 100ms typically)</li>
        <li>User-facing applications</li>
        <li>Dynamic inputs that can't be pre-computed</li>
        <li>Fraud detection, ad serving, chatbots</li>
      </ul>
      <h5>Considerations:</h5>
      <ul>
        <li>Latency constraints (model size, inference time)</li>
        <li>Scalability and load balancing</li>
        <li>High availability requirements</li>
        <li>More complex infrastructure</li>
      </ul>

      <h4>3. Edge Deployment</h4>
      <p>Deploy models directly on edge devices (mobile, IoT) without server communication.</p>
      <h5>Benefits:</h5>
      <ul>
        <li>No network latency</li>
        <li>Works offline</li>
        <li>Better privacy (data stays on device)</li>
        <li>Lower operational costs</li>
      </ul>
      <h5>Challenges:</h5>
      <ul>
        <li>Limited computational resources</li>
        <li>Model size constraints</li>
        <li>Model updating complexity</li>
        <li>Battery consumption</li>
      </ul>

      <h3>Deployment Technologies</h3>

      <h4>REST APIs</h4>
      <p>Expose models through HTTP endpoints using frameworks like Flask, FastAPI, or Django.</p>
      <h5>Best Practices:</h5>
      <ul>
        <li>Version your API endpoints (/v1/predict)</li>
        <li>Implement input validation</li>
        <li>Add authentication/authorization</li>
        <li>Return structured error messages</li>
        <li>Include health check endpoints</li>
      </ul>

      <h4>Containerization (Docker)</h4>
      <p>Package model, dependencies, and serving code in containers for consistency across environments.</p>
      <h5>Advantages:</h5>
      <ul>
        <li>Reproducible environments</li>
        <li>Isolation from host system</li>
        <li>Easy scaling and orchestration (Kubernetes)</li>
        <li>Version control for entire stack</li>
      </ul>

      <h4>Model Serving Frameworks</h4>
      <h5>TensorFlow Serving</h5>
      <ul>
        <li>High-performance serving for TensorFlow models</li>
        <li>Built-in versioning and A/B testing</li>
        <li>gRPC and REST APIs</li>
      </ul>
      <h5>TorchServe</h5>
      <ul>
        <li>Production serving for PyTorch models</li>
        <li>Multi-model serving</li>
        <li>Metrics and logging</li>
      </ul>
      <h5>ONNX Runtime</h5>
      <ul>
        <li>Framework-agnostic (convert models to ONNX format)</li>
        <li>Optimized inference across platforms</li>
        <li>Hardware acceleration support</li>
      </ul>

      <h3>Deployment Pipeline</h3>

      <h4>1. Model Packaging</h4>
      <ul>
        <li>Serialize trained model (pickle, joblib, SavedModel)</li>
        <li>Include preprocessing pipeline</li>
        <li>Document input/output schemas</li>
        <li>Save metadata (version, training date, metrics)</li>
      </ul>

      <h4>2. Testing</h4>
      <ul>
        <li><strong>Unit tests:</strong> Test preprocessing, postprocessing functions</li>
        <li><strong>Integration tests:</strong> Test full prediction pipeline</li>
        <li><strong>Load tests:</strong> Verify performance under expected traffic</li>
        <li><strong>Shadow mode:</strong> Run new model alongside existing one without affecting users</li>
      </ul>

      <h4>3. Deployment Strategy</h4>
      <h5>Blue-Green Deployment</h5>
      <ul>
        <li>Run two identical environments (blue=current, green=new)</li>
        <li>Switch traffic to green when ready</li>
        <li>Easy rollback by switching back to blue</li>
      </ul>
      <h5>Canary Deployment</h5>
      <ul>
        <li>Gradually route traffic to new model (5% → 25% → 50% → 100%)</li>
        <li>Monitor metrics at each stage</li>
        <li>Rollback if issues detected</li>
      </ul>

      <h4>4. Monitoring</h4>
      <ul>
        <li><strong>Performance metrics:</strong> Latency, throughput, error rates</li>
        <li><strong>Model metrics:</strong> Prediction distribution, confidence scores</li>
        <li><strong>Business metrics:</strong> Conversion rates, user satisfaction</li>
        <li><strong>Infrastructure:</strong> CPU, memory, GPU utilization</li>
      </ul>

      <h3>Challenges and Solutions</h3>

      <h4>Model Versioning</h4>
      <p><strong>Problem:</strong> Managing multiple model versions in production.</p>
      <p><strong>Solutions:</strong></p>
      <ul>
        <li>Use semantic versioning (v1.2.3)</li>
        <li>Store models in artifact repositories (MLflow, DVC)</li>
        <li>Track model lineage (data, code, hyperparameters)</li>
        <li>Implement graceful model switching</li>
      </ul>

      <h4>Feature Store</h4>
      <p><strong>Problem:</strong> Training-serving skew (features computed differently in training vs production).</p>
      <p><strong>Solutions:</strong></p>
      <ul>
        <li>Centralized feature computation and storage</li>
        <li>Serve pre-computed features for low latency</li>
        <li>Ensure consistency between offline and online features</li>
        <li>Tools: Feast, Tecton, AWS Feature Store</li>
      </ul>

      <h4>Dependency Management</h4>
      <p><strong>Problem:</strong> Library version mismatches between training and serving.</p>
      <p><strong>Solutions:</strong></p>
      <ul>
        <li>Pin exact dependency versions (requirements.txt, poetry.lock)</li>
        <li>Use same environment for training and serving (containers)</li>
        <li>Test model serialization/deserialization</li>
      </ul>

      <h3>Security Considerations</h3>
      <ul>
        <li><strong>Input validation:</strong> Sanitize and validate all inputs to prevent injection attacks</li>
        <li><strong>Rate limiting:</strong> Prevent abuse and DDoS attacks</li>
        <li><strong>Authentication:</strong> API keys, OAuth, JWT tokens</li>
        <li><strong>Model extraction attacks:</strong> Limit query rates, add noise to predictions</li>
        <li><strong>Data privacy:</strong> Encrypt data in transit and at rest, comply with regulations (GDPR, CCPA)</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import joblib
import numpy as np
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and preprocessing pipeline
model = joblib.load('model.joblib')
preprocessor = joblib.load('preprocessor.joblib')

# Define API
app = FastAPI(title="ML Model API", version="1.0.0")

# Input/output schemas
class PredictionInput(BaseModel):
    features: List[float]

    @validator('features')
    def validate_features(cls, v):
        if len(v) != 10:  # Expected number of features
            raise ValueError('Expected 10 features')
        if any(np.isnan(v) or np.isinf(v)):
            raise ValueError('Features contain NaN or Inf')
        return v

class PredictionOutput(BaseModel):
    prediction: float
    confidence: float
    model_version: str

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# Prediction endpoint
@app.post("/v1/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        # Preprocess input
        features = np.array(input_data.features).reshape(1, -1)
        features_processed = preprocessor.transform(features)

        # Make prediction
        prediction = model.predict(features_processed)[0]

        # Get confidence (for probabilistic models)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_processed)[0]
            confidence = float(max(probabilities))
        else:
            confidence = None

        # Log request (for monitoring)
        logger.info(f"Prediction made: {prediction}, confidence: {confidence}")

        return PredictionOutput(
            prediction=float(prediction),
            confidence=confidence,
            model_version="1.0.0"
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch prediction endpoint
@app.post("/v1/predict/batch")
async def predict_batch(inputs: List[PredictionInput]):
    try:
        features = np.array([inp.features for inp in inputs])
        features_processed = preprocessor.transform(features)
        predictions = model.predict(features_processed)

        return {
            "predictions": predictions.tolist(),
            "count": len(predictions)
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000`,
        explanation: 'Complete FastAPI deployment with input validation, error handling, health checks, and both single and batch prediction endpoints. Includes proper logging for monitoring and structured error responses.'
      },
      {
        language: 'Python',
        code: `# Dockerfile for containerized deployment
"""
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts and code
COPY model.joblib preprocessor.joblib ./
COPY app.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
"""

# Docker Compose for multi-container setup with monitoring
"""
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_VERSION=1.0.0
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
"""

# Model deployment script with versioning
import joblib
import mlflow
from datetime import datetime
import os

class ModelDeployer:
    def __init__(self, registry_path='./model_registry'):
        self.registry_path = registry_path
        os.makedirs(registry_path, exist_ok=True)

    def package_model(self, model, preprocessor, metadata):
        """Package model with all necessary components."""
        version = metadata['version']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = f"{self.registry_path}/v{version}_{timestamp}"
        os.makedirs(model_dir, exist_ok=True)

        # Save model and preprocessor
        joblib.dump(model, f"{model_dir}/model.joblib")
        joblib.dump(preprocessor, f"{model_dir}/preprocessor.joblib")

        # Save metadata
        import json
        with open(f"{model_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model packaged: {model_dir}")
        return model_dir

    def deploy_canary(self, model_path, traffic_percent=10):
        """Deploy model with canary strategy."""
        print(f"Deploying canary with {traffic_percent}% traffic...")

        # In production, this would update load balancer rules
        # For example with Kubernetes:
        # kubectl apply -f canary-deployment.yaml
        # kubectl patch service ml-service -p '{"spec":{"selector":{"version":"canary"}}}'

        # Monitor metrics for specified duration
        import time
        monitor_duration = 300  # 5 minutes
        print(f"Monitoring canary for {monitor_duration}s...")
        time.sleep(monitor_duration)

        # Check metrics (simplified)
        metrics = self.check_canary_metrics()
        if metrics['error_rate'] < 0.01 and metrics['latency_p95'] < 100:
            print("Canary healthy, promoting to full deployment")
            return True
        else:
            print("Canary unhealthy, rolling back")
            return False

    def check_canary_metrics(self):
        """Check canary deployment metrics."""
        # In production, fetch from monitoring system (Prometheus, Datadog)
        return {
            'error_rate': 0.005,
            'latency_p95': 85,
            'throughput': 1200
        }

    def rollback(self, previous_version):
        """Rollback to previous model version."""
        print(f"Rolling back to version {previous_version}")
        # Update serving config to point to previous version
        # kubectl rollout undo deployment/ml-api

# Usage
metadata = {
    'version': '2.0.0',
    'training_date': '2024-01-15',
    'metrics': {'accuracy': 0.94, 'f1': 0.92},
    'features': ['feature1', 'feature2', 'feature3']
}

deployer = ModelDeployer()
model_dir = deployer.package_model(model, preprocessor, metadata)

# Deploy with canary strategy
success = deployer.deploy_canary(model_dir, traffic_percent=10)
if not success:
    deployer.rollback('1.0.0')`,
        explanation: 'Production deployment setup including Dockerfile for containerization, Docker Compose for multi-service orchestration, and a model deployer class with canary deployment strategy and rollback capability. Shows version management and health monitoring patterns.'
      }
    ],
    interviewQuestions: [
      {
        question: 'What are the key differences between batch and online inference, and when would you choose each?',
        answer: `Batch inference processes large volumes of data at scheduled intervals (e.g., daily recommendations), offering higher throughput and computational efficiency. Online inference serves real-time requests with low latency requirements (e.g., fraud detection). Choose batch for: non-urgent predictions, large datasets, cost optimization. Choose online for: real-time decisions, user-facing applications, time-sensitive predictions. Hybrid approaches can combine both based on use case requirements.`
      },
      {
        question: 'How do you handle model versioning in production? What happens when you need to roll back?',
        answer: `Model versioning involves tracking model artifacts, metadata, and dependencies with unique identifiers. Use MLOps tools (MLflow, Kubeflow) for version control. Implement blue-green deployments or canary releases for safe updates. For rollbacks: maintain previous model versions, automate rollback triggers based on performance metrics, ensure data compatibility, and have documented rollback procedures. Include model registry for centralized version management.`
      },
      {
        question: 'Explain training-serving skew. What causes it and how can you prevent it?',
        answer: `Training-serving skew occurs when data distributions differ between training and serving environments. Causes include: different preprocessing pipelines, data collection methods, temporal shifts, or feature computation differences. Prevention strategies: use identical preprocessing code, implement feature stores, validate data schemas, monitor input distributions, use consistent data sources, and implement integration tests comparing training and serving pipelines.`
      },
      {
        question: 'What strategies would you use to deploy a new model with minimal risk to production?',
        answer: `Risk mitigation strategies include: (1) Canary deployments - gradually increase traffic to new model, (2) A/B testing - compare new vs old model performance, (3) Shadow mode - run new model alongside old without affecting users, (4) Feature flags - quick enable/disable capabilities, (5) Extensive testing - unit, integration, load tests, (6) Monitoring - real-time metrics and alerts, (7) Rollback plans - automated reversion procedures.`
      },
      {
        question: 'How do you ensure your deployed model can handle the expected traffic load?',
        answer: `Load testing strategies: (1) Benchmark inference latency under various loads, (2) Use load testing tools (JMeter, Locust) to simulate traffic patterns, (3) Implement horizontal scaling with load balancers, (4) Set up auto-scaling based on metrics (CPU, memory, request count), (5) Monitor resource utilization, (6) Implement caching for frequent requests, (7) Use performance profiling to identify bottlenecks, (8) Plan capacity based on peak traffic projections.`
      },
      {
        question: 'What security considerations are important when deploying ML models as APIs?',
        answer: `Key security considerations: (1) Authentication/authorization - API keys, OAuth, role-based access, (2) Input validation - prevent injection attacks, validate data types/ranges, (3) Rate limiting - prevent abuse and DDoS, (4) Model protection - prevent model extraction/inversion attacks, (5) Data privacy - encryption in transit/rest, PII handling, (6) Logging/monitoring - audit trails, anomaly detection, (7) Network security - VPCs, firewalls, secure protocols.`
      }
    ],
    quizQuestions: [
      {
        id: 'deploy1',
        question: 'When is batch prediction preferred over real-time?',
        options: ['Never, real-time is always better', 'When predictions don\'t need to be immediate', 'Only for simple models', 'When you have unlimited compute'],
        correctAnswer: 1,
        explanation: 'Batch prediction is preferred when immediate predictions aren\'t required (e.g., daily recommendations), computationally expensive models, or when input data arrives in batches. It allows better resource utilization and simpler infrastructure.'
      },
      {
        id: 'deploy2',
        question: 'What is a canary deployment?',
        options: ['Deploying only on weekends', 'Gradually routing traffic to new model', 'Testing on birds', 'A/B testing with 50/50 split'],
        correctAnswer: 1,
        explanation: 'Canary deployment gradually routes a small percentage of traffic (e.g., 5%) to the new model while monitoring metrics. If successful, traffic is incrementally increased; if issues arise, it\'s easy to rollback.'
      },
      {
        id: 'deploy3',
        question: 'What causes training-serving skew?',
        options: ['Different hardware', 'Features computed differently in training vs production', 'Model overfitting', 'Poor data quality'],
        correctAnswer: 1,
        explanation: 'Training-serving skew occurs when features are computed differently during training (offline, batch) versus serving (online, real-time), leading to inconsistent predictions. Feature stores help solve this by centralizing feature computation.'
      }
    ]
  },

  'ab-testing': {
    id: 'ab-testing',
    title: 'A/B Testing for ML Models',
    category: 'ml-systems',
    description: 'Statistical methods for comparing model performance in production',
    content: `
      <h2>A/B Testing for ML Models</h2>
      <p>A/B testing (split testing) is a statistical methodology for comparing two or more variants (e.g., different ML models) to determine which performs better on a specific business metric.</p>

      <h3>Core Concepts</h3>

      <h4>Hypothesis Testing Framework</h4>
      <p><strong>Null Hypothesis (H₀):</strong> There is no difference between variants A and B.</p>
      <p><strong>Alternative Hypothesis (H₁):</strong> Variant B performs differently (better or worse) than variant A.</p>

      <h5>Statistical Significance</h5>
      <ul>
        <li><strong>p-value:</strong> Probability of observing results as extreme as actual results, assuming H₀ is true</li>
        <li><strong>Significance level (α):</strong> Threshold for rejecting H₀, typically 0.05 (5%)</li>
        <li><strong>If p < α:</strong> Reject H₀, results are statistically significant</li>
      </ul>

      <h5>Statistical Power</h5>
      <ul>
        <li><strong>Power (1-β):</strong> Probability of detecting a true effect when it exists</li>
        <li><strong>Typically aim for 80% power</strong></li>
        <li>Higher power requires larger sample size</li>
      </ul>

      <h4>Key Metrics</h4>

      <h5>Primary Metrics</h5>
      <p>Main business metric you're trying to improve:</p>
      <ul>
        <li>Click-through rate (CTR)</li>
        <li>Conversion rate</li>
        <li>Revenue per user</li>
        <li>User engagement time</li>
      </ul>

      <h5>Guardrail Metrics</h5>
      <p>Metrics that should not degrade:</p>
      <ul>
        <li>Page load time</li>
        <li>Error rate</li>
        <li>User retention</li>
        <li>Overall user satisfaction</li>
      </ul>

      <h5>Model-Specific Metrics</h5>
      <ul>
        <li>Prediction accuracy</li>
        <li>Inference latency</li>
        <li>Prediction diversity</li>
        <li>Calibration error</li>
      </ul>

      <h3>Experimental Design</h3>

      <h4>1. Sample Size Calculation</h4>
      <p>Determine required users/observations for statistical power:</p>
      <h5>Factors affecting sample size:</h5>
      <ul>
        <li><strong>Minimum detectable effect (MDE):</strong> Smallest improvement worth detecting (e.g., 2% CTR increase)</li>
        <li><strong>Baseline conversion rate:</strong> Current metric value</li>
        <li><strong>Significance level (α):</strong> Usually 0.05</li>
        <li><strong>Power (1-β):</strong> Usually 0.80</li>
      </ul>
      <p><strong>Formula for proportion tests:</strong></p>
      <p>n = 2 × [(Z<sub>α/2</sub> + Z<sub>β</sub>)<sup>2</sup>] × [p(1-p)] / (MDE)<sup>2</sup></p>

      <h4>2. Randomization</h4>
      <h5>User-level randomization (most common):</h5>
      <ul>
        <li>Assign each user to a variant consistently</li>
        <li>Prevents within-user inconsistency</li>
        <li>Use user_id hash for deterministic assignment</li>
      </ul>
      <h5>Request-level randomization:</h5>
      <ul>
        <li>Each request assigned independently</li>
        <li>Higher variance, need more data</li>
        <li>Useful when user behavior irrelevant</li>
      </ul>

      <h4>3. Traffic Allocation</h4>
      <h5>Equal split (50/50):</h5>
      <ul>
        <li>Most statistical power</li>
        <li>Suitable when both variants are safe</li>
      </ul>
      <h5>Unequal split (e.g., 90/10):</h5>
      <ul>
        <li>Limit exposure to potentially worse variant</li>
        <li>Useful for risky changes</li>
        <li>Requires larger overall sample size</li>
      </ul>

      <h4>4. Duration</h4>
      <ul>
        <li><strong>Run long enough</strong> to capture weekly patterns (at least 1-2 weeks)</li>
        <li><strong>Account for novelty effect:</strong> Users may react differently to new experience initially</li>
        <li><strong>Primacy effect:</strong> Long-time users may resist change</li>
      </ul>

      <h3>Statistical Tests</h3>

      <h4>For Binary Outcomes (e.g., click/no-click)</h4>
      <h5>Two-Proportion Z-Test</h5>
      <p>Test if conversion rates differ significantly between A and B.</p>

      <h4>For Continuous Outcomes (e.g., revenue, time)</h4>
      <h5>Two-Sample t-Test</h5>
      <p>Test if means differ significantly between A and B.</p>
      <h5>Mann-Whitney U Test</h5>
      <p>Non-parametric alternative when data is not normally distributed.</p>

      <h4>For Multiple Variants (A/B/C/D...)</h4>
      <h5>ANOVA (Analysis of Variance)</h5>
      <p>Test if any variant differs from others.</p>
      <h5>Multiple Comparison Correction</h5>
      <ul>
        <li><strong>Bonferroni correction:</strong> Divide α by number of comparisons</li>
        <li><strong>FDR control:</strong> Less conservative, controls false discovery rate</li>
      </ul>

      <h3>Common Pitfalls</h3>

      <h4>1. Peeking Problem</h4>
      <p><strong>Issue:</strong> Checking results multiple times and stopping when significant.</p>
      <p><strong>Solution:</strong> Use sequential testing methods or pre-commit to test duration.</p>

      <h4>2. Multiple Testing</h4>
      <p><strong>Issue:</strong> Testing many metrics increases false positive rate.</p>
      <p><strong>Solution:</strong> Designate one primary metric, use correction for others.</p>

      <h4>3. Simpson's Paradox</h4>
      <p><strong>Issue:</strong> Aggregate result differs from subgroup results.</p>
      <p><strong>Example:</strong> Model B better overall but worse for each segment.</p>
      <p><strong>Solution:</strong> Analyze important subgroups separately.</p>

      <h4>4. Network Effects</h4>
      <p><strong>Issue:</strong> User behavior affects others (social media, marketplaces).</p>
      <p><strong>Solution:</strong> Cluster randomization (assign groups of related users).</p>

      <h4>5. Sample Ratio Mismatch (SRM)</h4>
      <p><strong>Issue:</strong> Observed traffic split differs from expected (50/50 → 52/48).</p>
      <p><strong>Indicates:</strong> Bug in randomization, bot traffic, or other data quality issues.</p>
      <p><strong>Solution:</strong> Investigate and fix before analyzing results.</p>

      <h3>ML-Specific Considerations</h3>

      <h4>Model vs. Model</h4>
      <ul>
        <li>Compare new model against baseline model</li>
        <li>Measure both business metrics and model metrics</li>
        <li>Consider inference latency impact on user experience</li>
      </ul>

      <h4>Feature Importance</h4>
      <ul>
        <li>A/B test adding/removing features</li>
        <li>Measure cost (data collection, compute) vs. benefit</li>
      </ul>

      <h4>Hyperparameter Tuning</h4>
      <ul>
        <li>Test different model configurations in production</li>
        <li>Offline metrics don't always correlate with online performance</li>
      </ul>

      <h3>Advanced Techniques</h3>

      <h4>Multi-Armed Bandits</h4>
      <p>Dynamically allocate more traffic to better-performing variants during the experiment.</p>
      <h5>Advantages:</h5>
      <ul>
        <li>Minimize regret (cost of not choosing best variant)</li>
        <li>Faster convergence</li>
      </ul>
      <h5>Disadvantages:</h5>
      <ul>
        <li>Less statistical rigor</li>
        <li>Harder to analyze retrospectively</li>
      </ul>

      <h4>Stratified Sampling</h4>
      <p>Ensure equal representation of important segments in both variants.</p>
      <ul>
        <li>Reduces variance</li>
        <li>Allows smaller sample sizes</li>
        <li>Useful when population is heterogeneous</li>
      </ul>

      <h4>CUPED (Controlled-experiment Using Pre-Experiment Data)</h4>
      <p>Use pre-experiment user behavior to reduce variance.</p>
      <ul>
        <li>Can reduce required sample size by 30-50%</li>
        <li>Particularly effective for high-variance metrics</li>
      </ul>
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
  },

  'model-monitoring-drift-detection': {
    id: 'model-monitoring-drift-detection',
    title: 'Model Monitoring & Drift Detection',
    category: 'ml-systems',
    description: 'Monitoring ML models in production and detecting performance degradation',
    content: `
      <h2>Model Monitoring & Drift Detection</h2>
      <p>Model monitoring is the practice of tracking ML model performance in production to detect issues, ensure reliability, and maintain prediction quality over time. Drift detection identifies when the relationship between inputs and outputs changes, requiring model retraining.</p>

      <h3>Types of Drift</h3>

      <h4>1. Data Drift (Covariate Shift)</h4>
      <p>Input feature distribution changes while the relationship between features and target remains the same.</p>
      <h5>Formula:</h5>
      <p>P<sub>train</sub>(X) ≠ P<sub>prod</sub>(X), but P(Y|X) stays constant</p>
      <h5>Examples:</h5>
      <ul>
        <li>User demographics change (age, location)</li>
        <li>Seasonal patterns in e-commerce</li>
        <li>Economic conditions affecting financial models</li>
        <li>New product categories in recommendation systems</li>
      </ul>
      <h5>Impact:</h5>
      <ul>
        <li>Model predictions less reliable for new distribution</li>
        <li>May still be accurate but less calibrated</li>
        <li>Often detected before performance degradation</li>
      </ul>

      <h4>2. Concept Drift</h4>
      <p>The relationship between features and target changes, meaning the underlying concept being learned has changed.</p>
      <h5>Formula:</h5>
      <p>P(Y|X) changes over time, even if P(X) stays constant</p>
      <h5>Examples:</h5>
      <ul>
        <li>Fraud patterns evolve as fraudsters adapt</li>
        <li>User preferences change (fashion, music)</li>
        <li>Market conditions shift in trading models</li>
        <li>Disease symptoms change for medical diagnosis</li>
      </ul>
      <h5>Types of Concept Drift:</h5>
      <h6>Sudden (Abrupt) Drift:</h6>
      <ul>
        <li>Rapid change in concept (e.g., new regulations)</li>
        <li>Requires immediate model update</li>
      </ul>
      <h6>Gradual Drift:</h6>
      <ul>
        <li>Slow, continuous change (e.g., aging population)</li>
        <li>Scheduled retraining often sufficient</li>
      </ul>
      <h6>Incremental Drift:</h6>
      <ul>
        <li>Step-wise changes over time</li>
        <li>Monitor and retrain at inflection points</li>
      </ul>
      <h6>Recurring Concepts:</h6>
      <ul>
        <li>Patterns repeat periodically (e.g., seasonal)</li>
        <li>Consider ensemble of season-specific models</li>
      </ul>

      <h4>3. Label Drift (Prior Probability Shift)</h4>
      <p>The distribution of target labels changes.</p>
      <h5>Formula:</h5>
      <p>P<sub>train</sub>(Y) ≠ P<sub>prod</sub>(Y)</p>
      <h5>Examples:</h5>
      <ul>
        <li>Class imbalance shifts (more fraud cases)</li>
        <li>Popularity of products changes</li>
        <li>Disease prevalence changes</li>
      </ul>

      <h4>4. Prediction Drift</h4>
      <p>Model's prediction distribution changes, which may indicate upstream issues.</p>
      <h5>Examples:</h5>
      <ul>
        <li>Sudden spike in positive predictions</li>
        <li>Confidence scores consistently low</li>
        <li>Prediction distribution becomes more uniform</li>
      </ul>

      <h3>Monitoring Metrics</h3>

      <h4>Model Performance Metrics</h4>
      <p>Track the same metrics used during training, but in production:</p>
      <ul>
        <li><strong>Classification:</strong> Accuracy, precision, recall, F1, AUC-ROC</li>
        <li><strong>Regression:</strong> MAE, RMSE, R²</li>
        <li><strong>Ranking:</strong> NDCG, MAP, MRR</li>
      </ul>
      <h5>Challenge:</h5>
      <p>Ground truth labels often delayed or unavailable in production (label lag problem).</p>

      <h4>Proxy Metrics</h4>
      <p>Business metrics that correlate with model performance:</p>
      <ul>
        <li>Click-through rate (CTR)</li>
        <li>Conversion rate</li>
        <li>User engagement time</li>
        <li>Revenue per user</li>
        <li>Customer satisfaction scores</li>
      </ul>

      <h4>System Performance Metrics</h4>
      <ul>
        <li><strong>Latency:</strong> P50, P95, P99 prediction time</li>
        <li><strong>Throughput:</strong> Requests per second</li>
        <li><strong>Error rate:</strong> Failed predictions, timeouts</li>
        <li><strong>Resource usage:</strong> CPU, memory, GPU utilization</li>
      </ul>

      <h3>Drift Detection Methods</h3>

      <h4>Statistical Tests for Data Drift</h4>

      <h5>1. Kolmogorov-Smirnov (K-S) Test</h5>
      <ul>
        <li>Tests if two samples come from same distribution</li>
        <li>Works for continuous features</li>
        <li>Non-parametric (no distribution assumptions)</li>
        <li>Null hypothesis: Same distribution</li>
      </ul>

      <h5>2. Chi-Square Test</h5>
      <ul>
        <li>Tests independence for categorical features</li>
        <li>Compares observed vs. expected frequencies</li>
        <li>Requires sufficient sample size</li>
      </ul>

      <h5>3. Population Stability Index (PSI)</h5>
      <ul>
        <li>Measures distribution shift magnitude</li>
        <li>PSI < 0.1: No significant change</li>
        <li>0.1 < PSI < 0.25: Moderate change</li>
        <li>PSI > 0.25: Significant change, investigate</li>
      </ul>
      <p><strong>Formula:</strong> PSI = Σ (actual% - expected%) × ln(actual% / expected%)</p>

      <h5>4. Kullback-Leibler (KL) Divergence</h5>
      <ul>
        <li>Measures how one distribution differs from another</li>
        <li>Not symmetric (D<sub>KL</sub>(P||Q) ≠ D<sub>KL</sub>(Q||P))</li>
        <li>Use Jensen-Shannon divergence for symmetric version</li>
      </ul>

      <h4>Model-Based Drift Detection</h4>

      <h5>1. Discriminator Approach</h5>
      <ul>
        <li>Train binary classifier to distinguish training vs. production data</li>
        <li>If classifier achieves high accuracy → significant drift</li>
        <li>Feature importances show which features drifted most</li>
      </ul>

      <h5>2. Uncertainty Monitoring</h5>
      <ul>
        <li>Track prediction confidence/uncertainty</li>
        <li>Increasing uncertainty may indicate drift</li>
        <li>Particularly useful for deep learning models</li>
      </ul>

      <h5>3. Reconstruction Error</h5>
      <ul>
        <li>Train autoencoder on training data</li>
        <li>High reconstruction error on production data indicates drift</li>
        <li>Identifies out-of-distribution samples</li>
      </ul>

      <h3>Monitoring System Architecture</h3>

      <h4>Components</h4>

      <h5>1. Data Collection</h5>
      <ul>
        <li>Log all predictions with timestamps</li>
        <li>Store input features (consider privacy)</li>
        <li>Collect ground truth labels when available</li>
        <li>Record metadata (model version, user segments)</li>
      </ul>

      <h5>2. Metrics Computation</h5>
      <ul>
        <li>Batch computation (hourly/daily)</li>
        <li>Windowed statistics (7-day, 30-day moving averages)</li>
        <li>Compare against baseline (training distribution)</li>
      </ul>

      <h5>3. Alerting</h5>
      <ul>
        <li><strong>Threshold-based:</strong> Alert if metric exceeds threshold</li>
        <li><strong>Anomaly detection:</strong> Use statistical methods to detect outliers</li>
        <li><strong>Trend-based:</strong> Alert on sustained degradation</li>
        <li><strong>Severity levels:</strong> Warning vs. critical alerts</li>
      </ul>

      <h5>4. Visualization Dashboard</h5>
      <ul>
        <li>Time series plots of key metrics</li>
        <li>Feature distribution comparisons</li>
        <li>Prediction distribution over time</li>
        <li>Error analysis breakdowns</li>
      </ul>

      <h3>Response Strategies</h3>

      <h4>When Drift Detected</h4>

      <h5>1. Investigate Root Cause</h5>
      <ul>
        <li>Data quality issues (missing values, errors)</li>
        <li>Upstream system changes</li>
        <li>True distribution shift</li>
        <li>Seasonality or expected variation</li>
      </ul>

      <h5>2. Model Retraining</h5>
      <ul>
        <li><strong>Scheduled retraining:</strong> Regular cadence (weekly, monthly)</li>
        <li><strong>Triggered retraining:</strong> Automatic when drift detected</li>
        <li><strong>Incremental learning:</strong> Update model with new data (online learning)</li>
      </ul>

      <h5>3. Feature Engineering</h5>
      <ul>
        <li>Add features to capture new patterns</li>
        <li>Remove features that became irrelevant</li>
        <li>Create time-based features</li>
      </ul>

      <h5>4. Model Architecture Changes</h5>
      <ul>
        <li>More complex model if patterns changed</li>
        <li>Ensemble methods for robustness</li>
        <li>Domain adaptation techniques</li>
      </ul>

      <h3>Best Practices</h3>

      <h4>Monitoring Strategy</h4>
      <ul>
        <li><strong>Start simple:</strong> Monitor basic metrics first, add complexity as needed</li>
        <li><strong>Baselines:</strong> Establish baseline metrics during shadow deployment</li>
        <li><strong>Segmentation:</strong> Monitor metrics by user segment, device type, region</li>
        <li><strong>Alert fatigue:</strong> Tune thresholds to minimize false alarms</li>
      </ul>

      <h4>Retraining Cadence</h4>
      <ul>
        <li><strong>High drift rate:</strong> Daily/weekly retraining (fraud detection)</li>
        <li><strong>Moderate drift:</strong> Monthly retraining (recommendation systems)</li>
        <li><strong>Low drift:</strong> Quarterly retraining (credit scoring)</li>
        <li><strong>Adaptive:</strong> Adjust frequency based on drift detection</li>
      </ul>

      <h4>Data Retention</h4>
      <ul>
        <li>Store recent production data for retraining</li>
        <li>Sample historical data for long-term trends</li>
        <li>Balance storage costs with model quality</li>
      </ul>
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
  },

  'scaling-optimization': {
    id: 'scaling-optimization',
    title: 'Scaling & Optimization',
    category: 'ml-systems',
    description: 'Strategies for scaling ML systems and optimizing inference performance',
    content: `
      <h2>Scaling & Optimization</h2>
      <p>Scaling and optimization ensure ML systems can handle increasing traffic, maintain low latency, and operate cost-effectively in production. This involves both horizontal/vertical scaling and model-level optimizations.</p>

      <h3>Scaling Strategies</h3>

      <h4>Vertical Scaling (Scale Up)</h4>
      <p>Add more resources to a single machine.</p>
      <h5>Advantages:</h5>
      <ul>
        <li>Simpler to implement</li>
        <li>No distributed system complexity</li>
        <li>Better for single-machine bottlenecks (GPU inference)</li>
      </ul>
      <h5>Disadvantages:</h5>
      <ul>
        <li>Hardware limits (max CPU, RAM, GPU)</li>
        <li>Expensive at high end</li>
        <li>Single point of failure</li>
      </ul>
      <h5>When to use:</h5>
      <ul>
        <li>GPU-bound inference (deep learning models)</li>
        <li>Low to moderate traffic</li>
        <li>Initial deployment</li>
      </ul>

      <h4>Horizontal Scaling (Scale Out)</h4>
      <p>Add more machines to distribute load.</p>
      <h5>Advantages:</h5>
      <ul>
        <li>Nearly unlimited scaling</li>
        <li>Fault tolerance (if one fails, others continue)</li>
        <li>Cost-effective at scale</li>
      </ul>
      <h5>Disadvantages:</h5>
      <ul>
        <li>Requires load balancing</li>
        <li>More complex infrastructure</li>
        <li>Network latency between services</li>
      </ul>
      <h5>When to use:</h5>
      <ul>
        <li>High traffic volume</li>
        <li>CPU-bound inference</li>
        <li>Need for high availability</li>
      </ul>

      <h4>Auto-scaling</h4>
      <p>Automatically adjust resources based on demand.</p>
      <h5>Metrics for scaling:</h5>
      <ul>
        <li><strong>CPU utilization:</strong> Scale when > 70%</li>
        <li><strong>Request queue depth:</strong> Scale when backlog grows</li>
        <li><strong>Response time:</strong> Scale when latency degrades</li>
        <li><strong>Time-based:</strong> Pre-scale for known traffic patterns</li>
      </ul>
      <h5>Best practices:</h5>
      <ul>
        <li>Scale up faster than down (avoid thrashing)</li>
        <li>Set min/max instance counts</li>
        <li>Use warmup period for new instances</li>
        <li>Monitor scaling events and adjust thresholds</li>
      </ul>

      <h3>Model Optimization</h3>

      <h4>1. Model Compression</h4>

      <h5>Quantization</h5>
      <p>Reduce numerical precision (FP32 → INT8).</p>
      <h6>Benefits:</h6>
      <ul>
        <li>4x smaller model size (FP32 → INT8)</li>
        <li>Faster inference (2-4x speedup)</li>
        <li>Lower memory bandwidth</li>
      </ul>
      <h6>Types:</h6>
      <ul>
        <li><strong>Post-training quantization:</strong> Convert trained model (easy, slight accuracy loss)</li>
        <li><strong>Quantization-aware training:</strong> Train with quantization in mind (better accuracy)</li>
      </ul>
      <h6>Trade-offs:</h6>
      <ul>
        <li>Minimal accuracy loss (typically < 1%)</li>
        <li>Not all operations supported in lower precision</li>
      </ul>

      <h5>Pruning</h5>
      <p>Remove unnecessary weights from neural networks.</p>
      <h6>Types:</h6>
      <ul>
        <li><strong>Unstructured pruning:</strong> Remove individual weights (irregular sparsity)</li>
        <li><strong>Structured pruning:</strong> Remove entire channels/layers (regular sparsity, easier hardware support)</li>
      </ul>
      <h6>Benefits:</h6>
      <ul>
        <li>Can remove 50-90% of weights with minimal accuracy loss</li>
        <li>Faster inference and smaller models</li>
      </ul>
      <h6>Process:</h6>
      <ul>
        <li>Train full model</li>
        <li>Identify least important weights (by magnitude)</li>
        <li>Remove and fine-tune</li>
      </ul>

      <h5>Knowledge Distillation</h5>
      <p>Train small "student" model to mimic large "teacher" model.</p>
      <h6>Process:</h6>
      <ul>
        <li>Train large, accurate teacher model</li>
        <li>Use teacher's soft predictions as targets for student</li>
        <li>Student learns compressed representation</li>
      </ul>
      <h6>Benefits:</h6>
      <ul>
        <li>Student can be 10-100x smaller</li>
        <li>Often better than training student from scratch</li>
        <li>Preserves more knowledge than just using hard labels</li>
      </ul>

      <h4>2. Inference Optimization</h4>

      <h5>Batching</h5>
      <p>Process multiple requests together for efficiency.</p>
      <h6>Benefits:</h6>
      <ul>
        <li>Better GPU utilization (GPUs designed for parallel work)</li>
        <li>Higher throughput</li>
      </ul>
      <h6>Trade-offs:</h6>
      <ul>
        <li>Increased latency (wait for batch to fill)</li>
        <li>Dynamic batching: Balance batch size and wait time</li>
      </ul>
      <h6>Best practices:</h6>
      <ul>
        <li>Max batch size: Fit GPU memory</li>
        <li>Max wait time: Meet latency SLA (e.g., 10ms)</li>
      </ul>

      <h5>Model Serving Frameworks</h5>
      <h6>TensorFlow Serving</h6>
      <ul>
        <li>Built for TensorFlow models</li>
        <li>Dynamic batching, model versioning</li>
        <li>gRPC and REST APIs</li>
      </ul>
      <h6>TorchServe</h6>
      <ul>
        <li>Official PyTorch serving</li>
        <li>Multi-model serving</li>
        <li>Custom handlers for preprocessing</li>
      </ul>
      <h6>NVIDIA Triton</h6>
      <ul>
        <li>Framework-agnostic (TF, PyTorch, ONNX, etc.)</li>
        <li>GPU optimization, dynamic batching</li>
        <li>Model ensemble support</li>
      </ul>
      <h6>ONNX Runtime</h6>
      <ul>
        <li>Cross-platform inference</li>
        <li>Graph optimizations</li>
        <li>Hardware acceleration (GPU, NPU)</li>
      </ul>

      <h5>Hardware Acceleration</h5>
      <ul>
        <li><strong>GPUs:</strong> Best for deep learning (tensor operations)</li>
        <li><strong>CPUs:</strong> Good for classical ML, low-latency small models</li>
        <li><strong>TPUs:</strong> Google's custom accelerators for TensorFlow</li>
        <li><strong>AWS Inferentia:</strong> Custom chips for inference</li>
        <li><strong>Edge TPUs:</strong> For on-device inference</li>
      </ul>

      <h4>3. Caching Strategies</h4>

      <h5>Prediction Caching</h5>
      <p>Cache predictions for frequently requested inputs.</p>
      <h6>When effective:</h6>
      <ul>
        <li>Same inputs requested repeatedly (product recommendations)</li>
        <li>Predictions don't change frequently</li>
        <li>High compute cost per prediction</li>
      </ul>
      <h6>Implementation:</h6>
      <ul>
        <li>Use Redis/Memcached</li>
        <li>Cache key: Hash of input features</li>
        <li>Set TTL based on staleness tolerance</li>
      </ul>

      <h5>Feature Caching</h5>
      <p>Pre-compute and cache expensive features.</p>
      <h6>Examples:</h6>
      <ul>
        <li>User embeddings for recommendation</li>
        <li>Aggregated statistics (30-day average)</li>
        <li>Entity features (product metadata)</li>
      </ul>

      <h5>Model Caching</h5>
      <p>Keep models in memory to avoid loading overhead.</p>
      <ul>
        <li>Load model on service startup</li>
        <li>Use model warmup with dummy inputs</li>
        <li>For multiple models, use LRU eviction</li>
      </ul>

      <h3>Architecture Patterns</h3>

      <h4>1. Microservices</h4>
      <p>Separate model serving from business logic.</p>
      <h5>Benefits:</h5>
      <ul>
        <li>Independent scaling of ML service</li>
        <li>Easier to update model without affecting application</li>
        <li>Technology stack flexibility</li>
      </ul>

      <h4>2. Model Cascade</h4>
      <p>Use fast, simple model first; complex model only when needed.</p>
      <h5>Example:</h5>
      <ul>
        <li>Stage 1: Lightweight model filters 95% of negatives</li>
        <li>Stage 2: Heavy model processes remaining 5%</li>
      </ul>
      <h5>Benefits:</h5>
      <ul>
        <li>Reduces average latency</li>
        <li>Lower compute cost</li>
      </ul>

      <h4>3. Model Ensemble</h4>
      <p>Combine predictions from multiple models.</p>
      <h5>Strategies:</h5>
      <ul>
        <li><strong>Parallel:</strong> All models run, combine results</li>
        <li><strong>Sequential:</strong> Second model refines first's output</li>
      </ul>
      <h5>Trade-offs:</h5>
      <ul>
        <li>Better accuracy</li>
        <li>Higher latency and cost</li>
      </ul>

      <h3>Latency Optimization</h3>

      <h4>Identify Bottlenecks</h4>
      <ul>
        <li><strong>Profile inference:</strong> Measure time for preprocessing, model forward pass, postprocessing</li>
        <li><strong>Network latency:</strong> Time to send/receive data</li>
        <li><strong>Queue time:</strong> Waiting for GPU/CPU availability</li>
      </ul>

      <h4>Optimization Techniques</h4>
      <ul>
        <li><strong>Reduce model size:</strong> Quantization, pruning, distillation</li>
        <li><strong>Optimize operations:</strong> Use fused ops, avoid unnecessary computation</li>
        <li><strong>Batch processing:</strong> Increase throughput (may increase latency)</li>
        <li><strong>Asynchronous processing:</strong> Non-blocking I/O for preprocessing</li>
        <li><strong>Pre-computation:</strong> Move static computation to offline</li>
      </ul>

      <h4>Latency SLA</h4>
      <ul>
        <li>Define target latency (e.g., P95 < 100ms)</li>
        <li>Monitor and alert on violations</li>
        <li>Trade off latency vs. accuracy when needed</li>
      </ul>

      <h3>Cost Optimization</h3>

      <h4>Compute Costs</h4>
      <ul>
        <li><strong>Right-size instances:</strong> Don't over-provision</li>
        <li><strong>Use spot/preemptible instances:</strong> For fault-tolerant workloads</li>
        <li><strong>Batch workloads:</strong> Process overnight when compute is cheaper</li>
        <li><strong>Model compression:</strong> Smaller models use less compute</li>
      </ul>

      <h4>Storage Costs</h4>
      <ul>
        <li><strong>Feature store:</strong> Dedup and compress features</li>
        <li><strong>Model artifacts:</strong> Archive old versions to cold storage</li>
        <li><strong>Logs:</strong> Sample or aggregate before storing</li>
      </ul>

      <h4>Data Transfer</h4>
      <ul>
        <li><strong>Colocation:</strong> Keep model and data in same region</li>
        <li><strong>Compression:</strong> Compress payloads</li>
        <li><strong>Edge caching:</strong> Serve from CDN when possible</li>
      </ul>
    `,
    codeExamples: [
      {
        language: 'Python',
        code: `import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, quantize_fx
import time
import numpy as np

# Define a simple neural network
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Original model
model_fp32 = SimpleModel()
model_fp32.eval()

# Benchmark function
def benchmark_model(model, input_tensor, num_runs=1000):
    """Measure inference time and model size."""
    # Warmup
    for _ in range(10):
        _ = model(input_tensor)

    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(input_tensor)
    end = time.time()

    avg_time = (end - start) / num_runs * 1000  # ms

    # Model size
    torch.save(model.state_dict(), 'temp_model.pt')
    import os
    size_mb = os.path.getsize('temp_model.pt') / (1024 * 1024)
    os.remove('temp_model.pt')

    return avg_time, size_mb

# Test input
input_tensor = torch.randn(1, 100)

print("=== Model Optimization Comparison ===\\n")

# 1. Original FP32 model
print("1. Original FP32 Model:")
time_fp32, size_fp32 = benchmark_model(model_fp32, input_tensor)
print(f"   Inference time: {time_fp32:.3f} ms")
print(f"   Model size: {size_fp32:.2f} MB\\n")

# 2. Dynamic Quantization (FP32 → INT8)
print("2. Dynamic Quantization (INT8):")
model_int8 = quantize_dynamic(
    model_fp32,
    {nn.Linear},  # Quantize Linear layers
    dtype=torch.qint8
)
time_int8, size_int8 = benchmark_model(model_int8, input_tensor)
print(f"   Inference time: {time_int8:.3f} ms ({time_fp32/time_int8:.1f}x speedup)")
print(f"   Model size: {size_int8:.2f} MB ({size_fp32/size_int8:.1f}x smaller)\\n")

# 3. Pruning (simplified example)
print("3. Pruning:")
import torch.nn.utils.prune as prune

model_pruned = SimpleModel()
model_pruned.load_state_dict(model_fp32.state_dict())

# Prune 50% of weights in each Linear layer
for name, module in model_pruned.named_modules():
    if isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.5)
        prune.remove(module, 'weight')  # Make pruning permanent

time_pruned, size_pruned = benchmark_model(model_pruned, input_tensor)
print(f"   Inference time: {time_pruned:.3f} ms")
print(f"   Model size: {size_pruned:.2f} MB")
print(f"   Note: Sparse models need specialized hardware for speedup\\n")

# 4. Knowledge Distillation (concept)
print("4. Knowledge Distillation (Student Model):")

class StudentModel(nn.Module):
    """Smaller student model."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 100)  # Much smaller
        self.fc2 = nn.Linear(100, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

student_model = StudentModel()
student_model.eval()

time_student, size_student = benchmark_model(student_model, input_tensor)
print(f"   Inference time: {time_student:.3f} ms ({time_fp32/time_student:.1f}x speedup)")
print(f"   Model size: {size_student:.2f} MB ({size_fp32/size_student:.1f}x smaller)")
print(f"   Note: Requires training with teacher model\\n")

# 5. Batching comparison
print("5. Batching (Throughput Optimization):")

def benchmark_batched(model, batch_sizes=[1, 8, 32]):
    """Compare throughput with different batch sizes."""
    results = {}
    for batch_size in batch_sizes:
        input_batch = torch.randn(batch_size, 100)
        num_runs = 100

        # Warmup
        for _ in range(10):
            _ = model(input_batch)

        # Benchmark
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(input_batch)
        end = time.time()

        total_time = end - start
        throughput = (num_runs * batch_size) / total_time  # predictions/sec
        latency = (total_time / num_runs) * 1000  # ms per batch

        results[batch_size] = {
            'throughput': throughput,
            'latency': latency
        }

    return results

batching_results = benchmark_batched(model_fp32)
for bs, metrics in batching_results.items():
    print(f"   Batch size {bs}:")
    print(f"     Throughput: {metrics['throughput']:.0f} predictions/sec")
    print(f"     Latency: {metrics['latency']:.2f} ms/batch")

print("\\n=== Summary ===")
print("Quantization: Best for CPU inference, 4x smaller, 2-4x faster")
print("Pruning: Requires sparse hardware support for speedup")
print("Distillation: Best compression, requires retraining")
print("Batching: Increases throughput but may increase latency")`,
        explanation: 'Comprehensive comparison of model optimization techniques including dynamic quantization (FP32→INT8), pruning, knowledge distillation, and batching. Shows real performance measurements for inference time and model size reduction, demonstrating the trade-offs between different optimization strategies.'
      },
      {
        language: 'Python',
        code: `import time
import numpy as np
from functools import lru_cache
from collections import defaultdict
from typing import Dict, List, Optional
import hashlib
import pickle

class OptimizedMLService:
    """Production ML service with caching, batching, and monitoring."""

    def __init__(self, model, max_batch_size=32, max_wait_ms=10):
        """
        Args:
            model: The ML model
            max_batch_size: Maximum batch size for inference
            max_wait_ms: Maximum time to wait for batch to fill
        """
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000  # Convert to seconds

        # Caching
        self.prediction_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Batching queue
        self.batch_queue = []
        self.last_batch_time = time.time()

        # Monitoring
        self.latency_samples = []
        self.request_count = 0

        # Feature cache (pre-computed features)
        self.feature_cache = {}

    def _hash_input(self, features: Dict) -> str:
        """Create deterministic hash of input features for caching."""
        # Sort keys for consistency
        feature_str = str(sorted(features.items()))
        return hashlib.md5(feature_str.encode()).hexdigest()

    @lru_cache(maxsize=10000)
    def get_user_embedding(self, user_id: str) -> np.ndarray:
        """
        Cached user embeddings (expensive to compute).
        Using @lru_cache for automatic cache management.
        """
        # In production: fetch from feature store
        # Simulate expensive computation
        time.sleep(0.01)  # 10ms
        return np.random.randn(128)

    def preprocess_features(self, features: Dict) -> np.ndarray:
        """Preprocess features, using cache when possible."""
        # Check if we have cached preprocessed features
        cache_key = self._hash_input(features)

        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        # Expensive preprocessing
        processed = np.array([
            features.get('age', 0) / 100,
            features.get('income', 0) / 100000,
            # ... more feature engineering
        ])

        # Cache for future requests
        self.feature_cache[cache_key] = processed

        return processed

    def predict_with_cache(self, features: Dict) -> float:
        """Single prediction with caching."""
        # Check prediction cache
        cache_key = self._hash_input(features)

        if cache_key in self.prediction_cache:
            self.cache_hits += 1
            return self.prediction_cache[cache_key]

        self.cache_misses += 1

        # Preprocess
        processed_features = self.preprocess_features(features)

        # Make prediction
        prediction = self.model.predict([processed_features])[0]

        # Cache result (with TTL in production)
        self.prediction_cache[cache_key] = prediction

        return prediction

    def predict_batch(self, features_list: List[Dict]) -> List[float]:
        """Batched prediction for efficiency."""
        if len(features_list) == 0:
            return []

        start_time = time.time()

        # Preprocess all features
        processed_features = np.array([
            self.preprocess_features(f) for f in features_list
        ])

        # Batch prediction
        predictions = self.model.predict(processed_features)

        # Record latency
        latency = (time.time() - start_time) * 1000  # ms
        self.latency_samples.append(latency)

        return predictions.tolist()

    def predict_with_dynamic_batching(self, features: Dict,
                                      callback=None) -> None:
        """
        Async prediction with dynamic batching.
        Groups requests to optimize throughput.

        Args:
            features: Input features
            callback: Function to call with prediction result
        """
        # Add to batch queue
        self.batch_queue.append({
            'features': features,
            'callback': callback,
            'timestamp': time.time()
        })

        # Process batch if full or timeout
        should_process = (
            len(self.batch_queue) >= self.max_batch_size or
            (time.time() - self.last_batch_time) >= self.max_wait_ms
        )

        if should_process:
            self._process_batch()

    def _process_batch(self):
        """Process queued batch of predictions."""
        if len(self.batch_queue) == 0:
            return

        # Extract features and callbacks
        batch_items = self.batch_queue[:self.max_batch_size]
        self.batch_queue = self.batch_queue[self.max_batch_size:]

        features_list = [item['features'] for item in batch_items]
        callbacks = [item['callback'] for item in batch_items]

        # Batch predict
        predictions = self.predict_batch(features_list)

        # Invoke callbacks
        for pred, callback in zip(predictions, callbacks):
            if callback:
                callback(pred)

        self.last_batch_time = time.time()

    def get_metrics(self) -> Dict:
        """Get service performance metrics."""
        total_requests = self.cache_hits + self.cache_misses

        metrics = {
            'total_requests': total_requests,
            'cache_hit_rate': self.cache_hits / total_requests if total_requests > 0 else 0,
            'cache_size': len(self.prediction_cache),
        }

        if self.latency_samples:
            metrics['latency_p50'] = np.percentile(self.latency_samples, 50)
            metrics['latency_p95'] = np.percentile(self.latency_samples, 95)
            metrics['latency_p99'] = np.percentile(self.latency_samples, 99)

        return metrics

    def optimize_cache(self, max_size: int = 10000):
        """Evict oldest cache entries to manage memory."""
        if len(self.prediction_cache) > max_size:
            # Simple eviction: remove random entries
            # In production: use LRU or TTL
            keys_to_remove = list(self.prediction_cache.keys())[:len(self.prediction_cache) - max_size]
            for key in keys_to_remove:
                del self.prediction_cache[key]

# Simulate a simple model
class DummyModel:
    def predict(self, X):
        time.sleep(0.005)  # 5ms inference
        return np.random.randn(len(X))

# Usage example
print("=== Optimized ML Service Demo ===\\n")

model = DummyModel()
service = OptimizedMLService(model, max_batch_size=8, max_wait_ms=10)

# Test caching
print("1. Testing prediction caching:")
features1 = {'age': 30, 'income': 50000}

# First request (cache miss)
start = time.time()
pred1 = service.predict_with_cache(features1)
time1 = (time.time() - start) * 1000

# Second request (cache hit)
start = time.time()
pred2 = service.predict_with_cache(features1)
time2 = (time.time() - start) * 1000

print(f"   First request: {time1:.2f} ms (cache miss)")
print(f"   Second request: {time2:.2f} ms (cache hit)")
print(f"   Speedup: {time1/time2:.1f}x\\n")

# Test batching
print("2. Testing batched prediction:")

# Individual predictions
features_list = [{'age': i, 'income': 40000 + i*1000} for i in range(20, 40)]

start = time.time()
individual_preds = [service.predict_with_cache(f) for f in features_list]
individual_time = (time.time() - start) * 1000

# Clear cache for fair comparison
service.prediction_cache.clear()

# Batched prediction
start = time.time()
batch_preds = service.predict_batch(features_list)
batch_time = (time.time() - start) * 1000

print(f"   Individual predictions: {individual_time:.2f} ms")
print(f"   Batched prediction: {batch_time:.2f} ms")
print(f"   Speedup: {individual_time/batch_time:.1f}x\\n")

# Metrics
print("3. Service Metrics:")
metrics = service.get_metrics()
for key, value in metrics.items():
    if isinstance(value, float):
        print(f"   {key}: {value:.3f}")
    else:
        print(f"   {key}: {value}")

print("\\n=== Optimization Summary ===")
print("✓ Caching: 10-100x speedup for repeated queries")
print("✓ Batching: 2-5x throughput improvement")
print("✓ Feature caching: Reduces preprocessing overhead")
print("✓ Monitoring: Track performance and optimize bottlenecks")`,
        explanation: 'Production-ready ML service demonstrating key optimization patterns: prediction caching with hashing, feature caching with LRU, dynamic batching to improve throughput, and comprehensive metrics tracking. Shows real performance improvements from caching (10-100x for cache hits) and batching (2-5x throughput).'
      }
    ],
    interviewQuestions: [
      {
        question: 'What is the difference between vertical and horizontal scaling? When would you use each?',
        answer: `Vertical scaling increases resources (CPU, memory) on single machines; horizontal scaling adds more machines. Vertical scaling: easier implementation, no distributed computing complexity, limited by hardware constraints. Horizontal scaling: unlimited scaling potential, fault tolerance, complexity in coordination. Use vertical for: stateful applications, small-medium loads, quick solutions. Use horizontal for: large-scale systems, fault tolerance requirements, cost efficiency at scale.`
      },
      {
        question: 'Explain the trade-offs between quantization, pruning, and knowledge distillation for model compression.',
        answer: `Quantization reduces precision (FP32 to INT8) - fast implementation, significant speedup, minimal accuracy loss. Pruning removes less important weights/neurons - better compression ratios, requires retraining, may need specialized hardware. Knowledge distillation trains smaller student model from teacher - flexible architecture changes, maintains performance, requires training data. Choose based on deployment constraints, accuracy requirements, and available computational resources.`
      },
      {
        question: 'How does dynamic batching improve throughput, and what is the latency trade-off?',
        answer: `Dynamic batching groups multiple requests to leverage parallel processing, significantly improving GPU utilization and throughput. Benefits: better hardware efficiency, reduced per-request costs. Latency trade-off: individual requests wait for batch formation, increasing tail latencies. Mitigation strategies: adaptive batch sizes, timeout mechanisms, prioritization queues. Configure based on traffic patterns and SLA requirements - higher throughput vs. lower latency.`
      },
      {
        question: 'What caching strategies would you use for a recommendation system with millions of users?',
        answer: `Multi-tier caching: (1) Application cache - Redis/Memcached for user preferences, (2) Model cache - pre-computed recommendations for active users, (3) Feature cache - user/item embeddings, (4) CDN cache - static recommendations. Strategies: LRU eviction, cache warming for popular items, personalized cache based on user activity patterns, cache invalidation for real-time updates. Balance memory costs with response time improvements.`
      },
      {
        question: 'How would you optimize a model to meet a P95 latency SLA of 50ms?',
        answer: `Optimization approach: (1) Profile to identify bottlenecks - model inference, preprocessing, I/O, (2) Model optimization - quantization, pruning, TensorRT/ONNX, (3) Infrastructure - GPU acceleration, optimized serving frameworks (TensorFlow Serving, Triton), (4) Caching frequent requests, (5) Asynchronous processing where possible, (6) Load balancing, (7) Monitor tail latencies continuously. Iteratively optimize highest impact components first.`
      },
      {
        question: 'Explain how to use auto-scaling effectively for an ML serving system.',
        answer: `Effective auto-scaling requires: (1) Proper metrics - not just CPU/memory but request queue length, model latency, (2) Predictive scaling based on traffic patterns, (3) Warm-up time consideration for ML models, (4) Scale-up aggressiveness vs. scale-down conservatism, (5) Circuit breakers for cascading failures, (6) Cost optimization with spot instances, (7) Multi-region scaling for global traffic. Monitor business metrics alongside technical metrics for effectiveness.`
      }
    ],
    quizQuestions: [
      {
        id: 'scale1',
        question: 'What does INT8 quantization do?',
        options: ['Removes weights', 'Reduces precision from FP32 to 8-bit integers', 'Trains smaller model', 'Compresses images'],
        correctAnswer: 1,
        explanation: 'Quantization reduces numerical precision from floating point (FP32, 32 bits) to integers (INT8, 8 bits), resulting in 4x smaller models and 2-4x faster inference with minimal accuracy loss (typically < 1%).'
      },
      {
        id: 'scale2',
        question: 'What is the main benefit of dynamic batching?',
        options: ['Lower latency per request', 'Higher throughput', 'Smaller model size', 'Better accuracy'],
        correctAnswer: 1,
        explanation: 'Dynamic batching improves throughput by processing multiple requests together, which better utilizes GPU/CPU parallelism. However, it may increase per-request latency since requests wait for the batch to fill.'
      },
      {
        id: 'scale3',
        question: 'When is vertical scaling preferred over horizontal?',
        options: ['Always', 'For GPU-bound inference with moderate traffic', 'For high traffic', 'Never'],
        correctAnswer: 1,
        explanation: 'Vertical scaling (adding more resources to one machine) is preferred for GPU-bound deep learning inference with low-moderate traffic, as it avoids distributed system complexity. Horizontal scaling is better for high traffic and CPU-bound workloads.'
      }
    ]
  }
};