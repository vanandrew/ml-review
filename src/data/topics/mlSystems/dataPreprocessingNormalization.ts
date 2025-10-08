import { Topic } from '../../../types';

export const dataPreprocessingNormalization: Topic = {
  id: 'data-preprocessing-normalization',
  title: 'Data Preprocessing & Normalization',
  category: 'ml-systems',
  description: 'Cleaning and transforming data for machine learning',
  content: `
    <h2>Data Preprocessing & Normalization: From Raw Data to ML-Ready</h2>
    
    <p>Raw data is rarely ready for machine learning. It arrives messy, incomplete, inconsistent—sensor readings missing values, user inputs with typos, measurements at wildly different scales. Data preprocessing transforms this chaos into clean, consistent, properly scaled features that algorithms can learn from effectively. It's been said that data scientists spend 80% of their time on data preparation—not because it's inefficient, but because it's critical. Poor preprocessing guarantees poor models, regardless of algorithm sophistication.</p>

    <p>Consider a model predicting customer churn using age (18-80) and income ($20K-$200K). Without scaling, income's large magnitude dominates distance calculations in k-NN, making age nearly irrelevant. Worse, if 20% of income values are missing and you simply delete those rows, you've lost valuable information and potentially introduced bias. Preprocessing addresses these challenges systematically.</p>

    <h3>Data Cleaning: Handling Imperfections</h3>

    <h4>Missing Values: The Universal Challenge</h4>

    <p>Missing data appears everywhere—sensor failures, user skip questions, data collection errors, privacy-preserving deletions. The pattern of missingness matters: <strong>Missing Completely At Random (MCAR)</strong> occurs when missingness is independent of observed and unobserved data (sensor randomly fails). <strong>Missing At Random (MAR)</strong> occurs when missingness depends on observed data but not the missing value itself (young people skip income questions). <strong>Missing Not At Random (MNAR)</strong> occurs when missingness depends on the unobserved value (high earners refuse to disclose income).</p>

    <p><strong>Simple imputation</strong> fills gaps with statistics. Mean imputation for numerical features is fast but ignores distribution and relationships. Median imputation is robust to outliers—better for skewed data. Mode imputation for categorical features. Constant fill (e.g., "unknown") explicitly marks missingness. For time series, forward fill propagates last known value, backward fill propagates next value.</p>

    <p><strong>Advanced imputation</strong> leverages relationships. KNN imputation uses similar instances' values—if customers with similar characteristics have income around $50K, impute $50K for the missing customer. Iterative imputation (MICE - Multiple Imputation by Chained Equations) treats each feature with missing values as a target, predicting it from other features iteratively until convergence. Model-based imputation trains a regression model to predict missing values.</p>

    <p><strong>Indicator variables</strong> preserve information about missingness itself. Add binary feature "income_was_missing"—maybe the fact that income is missing is predictive (high earners refusing to disclose). This transforms missingness from a problem into a feature.</p>

    <h4>Outliers: Noise or Signal?</h4>

    <p>Outliers are extreme values far from the rest. They might be measurement errors (temperature reading of 500°C), data entry mistakes (age 999), or legitimate rare events (million-dollar transaction). The distinction determines handling strategy.</p>

    <p><strong>Detection methods:</strong> Z-score identifies points >3 standard deviations from mean—assumes normal distribution. IQR (Interquartile Range) method flags points below Q1-1.5×IQR or above Q3+1.5×IQR—robust to distribution. Isolation Forest uses ensemble of trees to isolate anomalies. Visual inspection through box plots, scatter plots reveals patterns.</p>

    <p><strong>Handling strategies:</strong> Removal deletes outliers—use only when confident they're errors. Capping/winsorizing clips values to percentiles (e.g., 1st and 99th), preserving ordinal relationships while reducing impact. Transformation (log, sqrt) compresses scales, reducing outlier influence naturally. Robust models (tree-based, robust scalers) handle outliers gracefully without removal.</p>

    <h4>Duplicates: Redundancy and Errors</h4>

    <p>Exact duplicates—identical rows—arise from data collection errors or pipeline bugs. Remove them straightforwardly. Near duplicates—rows almost identical—require similarity measures and domain judgment. In time series, keep the most recent duplicate; in cross-sectional data, aggregate or keep representative samples.</p>

    <h3>Scaling and Normalization: Leveling the Playing Field</h3>

    <p>Features often live in different universes. Age ranges 0-100, income $0-$1M, click-through-rate 0-1%. When algorithms compute distances, large-scale features dominate; when gradient descent optimizes, poorly scaled features cause slow, unstable convergence. Scaling harmonizes these ranges.</p>

    <h4>Min-Max Scaling (Normalization): Bounded Transformation</h4>

    <p>Min-Max scales to [0,1] using (x - x_min)/(x_max - x_min), or custom ranges [a,b]. Every value maps proportionally into the bounded range. Benefits: preserves zero values and relationships, bounded output useful for algorithms requiring specific ranges (neural network activations). Drawback: sensitive to outliers—a single extreme value squashes all others into tiny range. Use for: bounded algorithms, when outliers are handled, neural networks needing specific input ranges.</p>

    <h4>Standardization (Z-score Normalization): Mean-Centered Scaling</h4>

    <p>Standardization transforms to mean=0, std=1 using (x - μ)/σ. Features become comparable regardless of original scale. Result is normally distributed if input was normal. Benefits: less outlier-sensitive than Min-Max, works well with algorithms assuming normal distributions, makes gradients well-behaved. Drawback: unbounded—values can be any real number. Use for: linear models, SVM, PCA, neural networks, when features have Gaussian-like distributions.</p>

    <h4>Robust Scaling: Outlier-Resistant Transformation</h4>

    <p>Robust Scaler uses median and IQR: (x - median)/IQR. Since median and IQR resist outlier influence, extreme values don't distort scaling. Use when: data contains many outliers that shouldn't be removed, robust statistics preferred, outliers are meaningful but shouldn't dominate scaling.</p>

    <h4>MaxAbsScaling: Preserving Sparsity</h4>

    <p>MaxAbsScaler divides by maximum absolute value: x/|x_max|, mapping to [-1,1]. Crucially, it doesn't shift/center data, preserving sparsity—zero stays zero. For sparse matrices (text TF-IDF, one-hot encoded features), this avoids densification. Use for: sparse data where preserving zeros is critical.</p>

    <h4>Unit Vector Scaling (L2 Normalization)</h4>

    <p>Normalizes samples (rows) to unit norm: x/||x||, making vector length=1. Common in text (TF-IDF vectors) and when cosine similarity is used. Focuses on direction rather than magnitude.</p>

    <h3>Distribution Transforms: Reshaping for Normality</h3>

    <p>Many algorithms assume or benefit from normal distributions. Real data is often skewed—income, prices, event counts follow power laws. Transformations reshape distributions toward normality.</p>

    <p><strong>Log transformation</strong> log(x+1) compresses right-skewed distributions common in real-world data. Multiplicative relationships become additive. Use log1p (log(1+x)) to handle zeros.</p>

    <p><strong>Square root/cube root</strong> moderately reduce skew, work with zeros unlike log.</p>

    <p><strong>Box-Cox transform</strong> automatically finds optimal power transformation (x^λ - 1)/λ to maximize normality. Requires strictly positive values.</p>

    <p><strong>Yeo-Johnson transform</strong> extends Box-Cox to handle negatives and zeros, making it more flexible.</p>

    <h3>Critical Best Practices</h3>

    <h4>The Sacred Rule: Split First, Fit on Training Only</h4>

    <p>The most common data leakage error: fitting scalers/imputers on entire dataset. Correct workflow: (1) Split into train/test, (2) Fit preprocessing on training data only, (3) Transform both train and test using training statistics. This simulates production where future data statistics are unknown. Violating this overestimates performance—you've used test data information during training.</p>

    <h4>When Scaling Matters (and When It Doesn't)</h4>

    <p><strong>Require scaling:</strong> Distance-based algorithms (k-NN, k-means, SVM), gradient descent algorithms (neural networks, linear/logistic regression), regularized models (Lasso, Ridge), PCA and dimensionality reduction.</p>

    <p><strong>Don't need scaling:</strong> Tree-based models (decision trees, random forests, XGBoost) which make split decisions based on relative comparisons, invariant to monotonic transformations.</p>

    <h4>Pipeline Integration: Ensuring Consistency</h4>

    <p>Sklearn's Pipeline ensures preprocessing steps apply in correct order, prevents leakage, and enables easy production deployment. ColumnTransformer applies different transformations to different feature types—numerical features get imputed and scaled, categorical features get encoded. Save entire pipeline for production to guarantee training and serving use identical preprocessing.</p>

    <h3>Special Considerations</h3>

    <p><strong>Time series:</strong> Use expanding windows for scaling (only past data), avoid future leakage, consider differencing for stationarity.</p>

    <p><strong>Sparse data:</strong> Use MaxAbsScaler to preserve sparsity; StandardScaler densifies sparse matrices.</p>

    <p><strong>Categorical features:</strong> Don't scale one-hot encoded features (already 0/1); ordinal features can be scaled if numeric interpretation makes sense.</p>

    <p>Data preprocessing transforms raw chaos into machine learning gold. Master it, and your models will thank you with better performance, faster training, and reliable predictions.</p>
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
};
