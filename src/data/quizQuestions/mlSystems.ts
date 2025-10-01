import { QuizQuestion } from '../../types';

// Feature Engineering - 20 questions
export const featureEngineeringQuestions: QuizQuestion[] = [
  {
    id: 'fe1',
    question: 'What is feature engineering?',
    options: ['Model training', 'Creating, transforming, and selecting features from raw data', 'Deployment', 'No processing'],
    correctAnswer: 1,
    explanation: 'Feature engineering designs input features to improve model performance and interpretability.'
  },
  {
    id: 'fe2',
    question: 'Why is feature engineering important?',
    options: ['Not important', 'Good features can dramatically improve performance, even with simple models', 'Only deep learning', 'No impact'],
    correctAnswer: 1,
    explanation: 'Quality features often matter more than model complexity; domain knowledge crucial.'
  },
  {
    id: 'fe3',
    question: 'What is feature extraction?',
    options: ['Creating features', 'Deriving features from raw data (e.g., text → TF-IDF, images → edges)', 'Selection', 'Scaling'],
    correctAnswer: 1,
    explanation: 'Feature extraction transforms raw inputs into meaningful representations for models.'
  },
  {
    id: 'fe4',
    question: 'What is feature selection?',
    options: ['Creating all features', 'Choosing subset of relevant features, removing irrelevant/redundant ones', 'No selection', 'Use everything'],
    correctAnswer: 1,
    explanation: 'Feature selection reduces dimensionality, improves generalization, decreases compute.'
  },
  {
    id: 'fe5',
    question: 'What are filter methods?',
    options: ['Model-based', 'Selecting features based on statistical tests (correlation, chi-square)', 'No statistics', 'Wrapper methods'],
    correctAnswer: 1,
    explanation: 'Filter methods use statistical measures independent of model: variance, correlation, mutual information.'
  },
  {
    id: 'fe6',
    question: 'What are wrapper methods?',
    options: ['Statistical only', 'Using model performance to select features (forward/backward selection)', 'No model', 'Filter methods'],
    correctAnswer: 1,
    explanation: 'Wrapper methods evaluate feature subsets by training models, computationally expensive but accurate.'
  },
  {
    id: 'fe7',
    question: 'What are embedded methods?',
    options: ['Separate selection', 'Feature selection during model training (LASSO, tree importances)', 'Post-training', 'No integration'],
    correctAnswer: 1,
    explanation: 'Embedded methods select features as part of model training: L1 regularization, decision tree splits.'
  },
  {
    id: 'fe8',
    question: 'What is one-hot encoding?',
    options: ['Numerical encoding', 'Converting categorical variable to binary vectors', 'Label encoding', 'No encoding'],
    correctAnswer: 1,
    explanation: 'One-hot encoding creates binary column per category: "red" → [1,0,0], "blue" → [0,1,0].'
  },
  {
    id: 'fe9',
    question: 'What is label encoding?',
    options: ['One-hot', 'Mapping categories to integers (red=0, blue=1, green=2)', 'Binary', 'No encoding'],
    correctAnswer: 1,
    explanation: 'Label encoding assigns integer to each category; implies ordering, suitable for ordinal data.'
  },
  {
    id: 'fe10',
    question: 'What is target encoding?',
    options: ['One-hot', 'Replacing category with mean target value for that category', 'Label encoding', 'No encoding'],
    correctAnswer: 1,
    explanation: 'Target encoding uses target statistics (mean, smoothed); powerful but risk of leakage.'
  },
  {
    id: 'fe11',
    question: 'What is binning/discretization?',
    options: ['Continuous only', 'Converting continuous features to discrete bins', 'No transformation', 'One-hot'],
    correctAnswer: 1,
    explanation: 'Binning groups continuous values: age → [child, adult, senior]; can capture non-linearity.'
  },
  {
    id: 'fe12',
    question: 'What is polynomial features?',
    options: ['Linear only', 'Creating interaction and higher-order terms (x₁·x₂, x₁²)', 'No interactions', 'First order'],
    correctAnswer: 1,
    explanation: 'Polynomial features enable linear models to capture non-linear relationships: x, x², x·y.'
  },
  {
    id: 'fe13',
    question: 'What is domain knowledge in feature engineering?',
    options: ['Not needed', 'Expert understanding guides meaningful feature creation', 'Automated only', 'No expertise'],
    correctAnswer: 1,
    explanation: 'Domain knowledge crucial: knowing disease risk factors, financial indicators, seasonal patterns.'
  },
  {
    id: 'fe14',
    question: 'What are temporal features?',
    options: ['No time', 'Features from timestamps: hour, day of week, seasonality, time since event', 'Static only', 'No temporal'],
    correctAnswer: 1,
    explanation: 'Temporal features extract patterns from time: weekday vs weekend, holidays, trends.'
  },
  {
    id: 'fe15',
    question: 'What are text features?',
    options: ['Images only', 'TF-IDF, n-grams, word embeddings, sentiment, length', 'No text', 'Raw text'],
    correctAnswer: 1,
    explanation: 'Text features: bag-of-words, TF-IDF, word2vec embeddings, character n-grams, sentiment scores.'
  },
  {
    id: 'fe16',
    question: 'What is feature scaling importance?',
    options: ['Not needed', 'Many algorithms sensitive to feature magnitude; scaling improves convergence', 'No impact', 'Harmful'],
    correctAnswer: 1,
    explanation: 'Scaling critical for distance-based models (KNN, SVM), gradient descent convergence.'
  },
  {
    id: 'fe17',
    question: 'What is the curse of dimensionality?',
    options: ['More features better', 'Too many features cause sparsity, overfitting, computational issues', 'No curse', 'Always helpful'],
    correctAnswer: 1,
    explanation: 'High dimensionality: data becomes sparse, distances lose meaning, requires exponentially more data.'
  },
  {
    id: 'fe18',
    question: 'What is automated feature engineering?',
    options: ['Manual only', 'Tools automatically generating features (Featuretools, AutoFeat)', 'No automation', 'Impossible'],
    correctAnswer: 1,
    explanation: 'AutoML tools generate features via deep feature synthesis, transformations, aggregations.'
  },
  {
    id: 'fe19',
    question: 'When is feature engineering less critical?',
    options: ['Always critical', 'Deep learning can learn features automatically from raw data', 'Never', 'All models'],
    correctAnswer: 1,
    explanation: 'Deep learning reduces manual feature engineering need; learns hierarchical features from raw inputs.'
  },
  {
    id: 'fe20',
    question: 'What is feature importance?',
    options: ['All equal', 'Measuring contribution of each feature to model predictions', 'No measurement', 'Random'],
    correctAnswer: 1,
    explanation: 'Feature importance quantifies relevance: tree-based importances, permutation importance, SHAP values.'
  }
];

// Data Preprocessing - 20 questions
export const dataPreprocessingQuestions: QuizQuestion[] = [
  {
    id: 'dp1',
    question: 'What is data preprocessing?',
    options: ['Model training', 'Cleaning and transforming raw data before modeling', 'Deployment', 'No processing'],
    correctAnswer: 1,
    explanation: 'Preprocessing prepares data: handling missing values, outliers, scaling, encoding for ML.'
  },
  {
    id: 'dp2',
    question: 'Why is preprocessing important?',
    options: ['Not important', 'Real data is messy; cleaning improves model quality and prevents errors', 'Raw data perfect', 'No impact'],
    correctAnswer: 1,
    explanation: 'Preprocessing fixes issues: missing values, inconsistent formats, outliers, scale differences.'
  },
  {
    id: 'dp3',
    question: 'What is data cleaning?',
    options: ['No cleaning', 'Fixing errors, inconsistencies, duplicates, formatting issues', 'Only scaling', 'Model training'],
    correctAnswer: 1,
    explanation: 'Data cleaning corrects typos, removes duplicates, standardizes formats, validates values.'
  },
  {
    id: 'dp4',
    question: 'How to handle missing values?',
    options: ['Ignore them', 'Deletion, imputation (mean/median/mode), predictive models, indicator variable', 'Crash', 'No handling'],
    correctAnswer: 1,
    explanation: 'Missing value strategies: delete rows/columns, fill with statistics, predict, flag as missing.'
  },
  {
    id: 'dp5',
    question: 'What is mean imputation?',
    options: ['Delete rows', 'Replacing missing values with feature mean', 'Median', 'Mode'],
    correctAnswer: 1,
    explanation: 'Mean imputation fills missing values with average; simple but ignores relationships.'
  },
  {
    id: 'dp6',
    question: 'What is the problem with mean imputation?',
    options: ['Perfect method', 'Reduces variance, distorts distributions, ignores correlations', 'No issues', 'Too complex'],
    correctAnswer: 1,
    explanation: 'Mean imputation artificially reduces spread, can distort statistical properties.'
  },
  {
    id: 'dp7',
    question: 'What is normalization?',
    options: ['No scaling', 'Scaling features to range [0,1]: (x - min)/(max - min)', 'Standardization', 'No change'],
    correctAnswer: 1,
    explanation: 'Min-max normalization scales to fixed range, sensitive to outliers.'
  },
  {
    id: 'dp8',
    question: 'What is standardization?',
    options: ['Range [0,1]', 'Scaling to zero mean, unit variance: (x - μ)/σ', 'No scaling', 'Normalization'],
    correctAnswer: 1,
    explanation: 'Standardization (z-score) centers at 0 with std=1; less sensitive to outliers than normalization.'
  },
  {
    id: 'dp9',
    question: 'When to use normalization vs standardization?',
    options: ['Same thing', 'Normalization for bounded range; Standardization for unbounded, Gaussian data', 'Random choice', 'No difference'],
    correctAnswer: 1,
    explanation: 'Normalization: neural networks, image pixels. Standardization: algorithms assuming Gaussian (SVM, logistic).'
  },
  {
    id: 'dp10',
    question: 'What is outlier detection?',
    options: ['No outliers', 'Identifying extreme values far from typical data', 'Mean calculation', 'No detection'],
    correctAnswer: 1,
    explanation: 'Outlier detection finds anomalous values: statistical methods (IQR, z-score), isolation forest.'
  },
  {
    id: 'dp11',
    question: 'How to handle outliers?',
    options: ['Always keep', 'Remove, cap (winsorization), transform (log), robust methods, or keep if meaningful', 'Always remove', 'Ignore'],
    correctAnswer: 1,
    explanation: 'Outlier handling depends on context: errors (remove), rare events (keep), skewed data (transform).'
  },
  {
    id: 'dp12',
    question: 'What is the IQR method?',
    options: ['Mean-based', 'Outliers outside [Q1 - 1.5·IQR, Q3 + 1.5·IQR]', 'Standard deviation', 'No method'],
    correctAnswer: 1,
    explanation: 'IQR (Interquartile Range) method: values beyond 1.5×IQR from quartiles considered outliers.'
  },
  {
    id: 'dp13',
    question: 'What is data transformation?',
    options: ['No change', 'Applying functions to features: log, sqrt, Box-Cox', 'Scaling only', 'Selection'],
    correctAnswer: 1,
    explanation: 'Transformations modify distributions: log for skewed data, polynomial for non-linearity.'
  },
  {
    id: 'dp14',
    question: 'What is log transformation useful for?',
    options: ['Normal data', 'Right-skewed data, reducing impact of outliers, multiplicative relationships', 'Left-skewed', 'No use'],
    correctAnswer: 1,
    explanation: 'Log transform compresses large values, makes skewed distributions more symmetric.'
  },
  {
    id: 'dp15',
    question: 'What is data validation?',
    options: ['No checking', 'Verifying data meets constraints: types, ranges, formats, consistency', 'Model testing', 'No validation'],
    correctAnswer: 1,
    explanation: 'Validation checks: correct dtypes, values in expected ranges, no invalid entries.'
  },
  {
    id: 'dp16',
    question: 'What is data leakage?',
    options: ['No problem', 'Using information from test set during training, causing overly optimistic performance', 'Good practice', 'No leakage'],
    correctAnswer: 1,
    explanation: 'Leakage: including future information, target encoding on full data, fitting scaler on test set.'
  },
  {
    id: 'dp17',
    question: 'What is train-test contamination?',
    options: ['No issue', 'Applying transformations fit on entire dataset before splitting', 'Good practice', 'No contamination'],
    correctAnswer: 1,
    explanation: 'Contamination: must fit scaler/imputer on training data only, then transform train and test separately.'
  },
  {
    id: 'dp18',
    question: 'What is the correct preprocessing pipeline?',
    options: ['Fit on all data', 'Split data → Fit preprocessing on train → Transform train and test', 'Transform then split', 'Random order'],
    correctAnswer: 1,
    explanation: 'Proper order: split first, fit preprocessing on training set, apply to both train and test.'
  },
  {
    id: 'dp19',
    question: 'What tools help with preprocessing?',
    options: ['Manual only', 'scikit-learn Pipeline, pandas, numpy, feature-engine', 'No tools', 'One tool'],
    correctAnswer: 1,
    explanation: 'Libraries: scikit-learn (Pipeline, preprocessing), pandas (data manipulation), feature-engine (specialized).'
  },
  {
    id: 'dp20',
    question: 'What is the 80-20 rule in data science?',
    options: ['80% modeling', '80% of time spent on data collection and preprocessing', '20% preprocessing', 'Equal time'],
    correctAnswer: 1,
    explanation: 'Data scientists spend ~80% of time on data wrangling, cleaning, preprocessing; only 20% on modeling.'
  }
];

// Imbalanced Data - 20 questions
export const imbalancedDataQuestions: QuizQuestion[] = [
  {
    id: 'id1',
    question: 'What is class imbalance?',
    options: ['Equal classes', 'Unequal distribution of classes: one class much more frequent', 'Balanced data', 'No imbalance'],
    correctAnswer: 1,
    explanation: 'Class imbalance: minority class has far fewer examples than majority (e.g., fraud: 0.1%, normal: 99.9%).'
  },
  {
    id: 'id2',
    question: 'Why is imbalance a problem?',
    options: ['Not a problem', 'Models biased toward majority class, poor minority class performance', 'Good for models', 'No impact'],
    correctAnswer: 1,
    explanation: 'Imbalance causes models to ignore minority class, achieving high accuracy but missing rare important cases.'
  },
  {
    id: 'id3',
    question: 'What is the accuracy paradox?',
    options: ['Accuracy perfect metric', 'High accuracy can be misleading with imbalanced data', 'Low accuracy good', 'No paradox'],
    correctAnswer: 1,
    explanation: 'Accuracy paradox: predicting always majority achieves high accuracy but is useless (99% accuracy predicting "no fraud").'
  },
  {
    id: 'id4',
    question: 'What metrics are better for imbalanced data?',
    options: ['Only accuracy', 'Precision, recall, F1-score, ROC-AUC, PR-AUC', 'Accuracy only', 'No metrics'],
    correctAnswer: 1,
    explanation: 'Imbalanced metrics: precision (correct positives), recall (found positives), F1 (harmonic mean), PR curve.'
  },
  {
    id: 'id5',
    question: 'What is precision?',
    options: ['TP / (TP + FN)', 'TP / (TP + FP): proportion of predicted positives that are correct', 'TN / (TN + FP)', 'No metric'],
    correctAnswer: 1,
    explanation: 'Precision: of predicted positives, how many are actually positive. Important when false positives costly.'
  },
  {
    id: 'id6',
    question: 'What is recall (sensitivity)?',
    options: ['TP / (TP + FP)', 'TP / (TP + FN): proportion of actual positives correctly identified', 'TN / (TN + FN)', 'No metric'],
    correctAnswer: 1,
    explanation: 'Recall: of actual positives, how many were found. Important when false negatives costly (disease detection).'
  },
  {
    id: 'id7',
    question: 'What is F1-score?',
    options: ['Average of P and R', 'Harmonic mean of precision and recall: 2PR/(P+R)', 'Geometric mean', 'No combination'],
    correctAnswer: 1,
    explanation: 'F1-score balances precision and recall; useful single metric for imbalanced classification.'
  },
  {
    id: 'id8',
    question: 'What is random oversampling?',
    options: ['Remove samples', 'Duplicating minority class samples randomly', 'Undersample', 'No sampling'],
    correctAnswer: 1,
    explanation: 'Random oversampling replicates minority examples until balanced; simple but risks overfitting.'
  },
  {
    id: 'id9',
    question: 'What is random undersampling?',
    options: ['Add samples', 'Removing majority class samples randomly', 'Oversample', 'No sampling'],
    correctAnswer: 1,
    explanation: 'Random undersampling discards majority samples to balance; fast but loses information.'
  },
  {
    id: 'id10',
    question: 'What is SMOTE?',
    options: ['Random duplication', 'Synthetic Minority Over-sampling: creates synthetic examples along minority sample lines', 'Undersampling', 'No synthesis'],
    correctAnswer: 1,
    explanation: 'SMOTE generates synthetic minority samples by interpolating between existing minority neighbors.'
  },
  {
    id: 'id11',
    question: 'How does SMOTE work?',
    options: ['Random noise', 'Select minority sample, find k neighbors, create synthetic point along line to neighbor', 'Duplication', 'No process'],
    correctAnswer: 1,
    explanation: 'SMOTE: for each minority sample, pick random k-neighbor, create new sample: x_new = x + λ(x_neighbor - x).'
  },
  {
    id: 'id12',
    question: 'What is class weighting?',
    options: ['Equal weights', 'Assigning higher loss weight to minority class errors', 'No weighting', 'Weight majority'],
    correctAnswer: 1,
    explanation: 'Class weights penalize minority errors more: loss = Σ w_i · loss_i, where w_minority > w_majority.'
  },
  {
    id: 'id13',
    question: 'How to set class weights?',
    options: ['All equal to 1', 'Inversely proportional to class frequency: w = n_samples / (n_classes · n_class)', 'Random', 'No formula'],
    correctAnswer: 1,
    explanation: 'Balanced class weights: assign weight inversely proportional to class frequency in training data.'
  },
  {
    id: 'id14',
    question: 'What is the precision-recall tradeoff?',
    options: ['No tradeoff', 'Increasing recall often decreases precision and vice versa', 'Both increase', 'Independent'],
    correctAnswer: 1,
    explanation: 'Lowering threshold increases recall (more positives) but decreases precision (more false positives).'
  },
  {
    id: 'id15',
    question: 'What is PR-AUC?',
    options: ['ROC curve', 'Area under Precision-Recall curve: better for imbalanced data than ROC-AUC', 'Accuracy', 'No metric'],
    correctAnswer: 1,
    explanation: 'PR-AUC more informative than ROC-AUC for imbalanced data, focusing on minority class performance.'
  },
  {
    id: 'id16',
    question: 'What is cost-sensitive learning?',
    options: ['Equal cost', 'Assigning different misclassification costs to different errors', 'No cost', 'Same cost all'],
    correctAnswer: 1,
    explanation: 'Cost-sensitive learning assigns higher cost to false negatives (miss fraud) than false positives.'
  },
  {
    id: 'id17',
    question: 'What is ensemble method for imbalance?',
    options: ['Single model', 'BalancedBagging, EasyEnsemble: multiple models on balanced subsets', 'No ensemble', 'Standard ensemble'],
    correctAnswer: 1,
    explanation: 'Ensemble approaches: train multiple models on balanced subsamples, combine predictions.'
  },
  {
    id: 'id18',
    question: 'What is anomaly detection approach?',
    options: ['Classification', 'Treating minority class as anomalies, using outlier detection', 'Supervised only', 'No detection'],
    correctAnswer: 1,
    explanation: 'Extreme imbalance: frame as anomaly detection using one-class SVM, isolation forest, autoencoders.'
  },
  {
    id: 'id19',
    question: 'What is threshold tuning?',
    options: ['Always 0.5', 'Adjusting decision threshold to optimize for specific metric', 'Fixed threshold', 'No tuning'],
    correctAnswer: 1,
    explanation: 'Threshold tuning: lower threshold to increase recall, raise to increase precision based on business needs.'
  },
  {
    id: 'id20',
    question: 'What are real-world imbalanced examples?',
    options: ['All balanced', 'Fraud detection, disease diagnosis, spam filtering, defect detection', 'No examples', 'Rare'],
    correctAnswer: 1,
    explanation: 'Common imbalanced problems: fraud (0.1%), rare diseases (1%), spam (10%), manufacturing defects (0.01%).'
  }
];
