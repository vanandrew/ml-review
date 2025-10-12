import { QuizQuestion } from '../../types';

export const evaluationMetricsQuestions: QuizQuestion[] = [
  {
    id: 'em1',
    question: 'What is accuracy in classification?',
    options: ['True Positives / Total', 'Correct Predictions / Total Predictions', 'Precision × Recall', 'TP / (TP + FP)'],
    correctAnswer: 1,
    explanation: 'Accuracy is the ratio of correct predictions to total predictions: (TP + TN) / (TP + TN + FP + FN).'
  },
  {
    id: 'em2',
    question: 'Why is accuracy misleading for imbalanced datasets?',
    options: ['It\'s too complex', 'High accuracy possible by predicting majority class', 'It requires normalization', 'It only works for binary classification'],
    correctAnswer: 1,
    explanation: 'With imbalanced data (e.g., 95% class A), a model predicting always class A gets 95% accuracy but is useless.'
  },
  {
    id: 'em3',
    question: 'What does precision measure?',
    options: ['TP / (TP + FN)', 'TP / (TP + FP)', 'TN / (TN + FP)', '(TP + TN) / Total'],
    correctAnswer: 1,
    explanation: 'Precision = TP / (TP + FP) measures the accuracy of positive predictions - of those predicted positive, how many were correct?'
  },
  {
    id: 'em4',
    question: 'What does recall measure?',
    options: ['TP / (TP + FN)', 'TP / (TP + FP)', 'TN / (TN + FN)', 'FP / (FP + TN)'],
    correctAnswer: 0,
    explanation: 'Recall = TP / (TP + FN) measures coverage of actual positives - of all actual positives, how many were found?'
  },
  {
    id: 'em5',
    question: 'What is another name for recall?',
    options: ['Specificity', 'Sensitivity', 'Precision', 'Accuracy'],
    correctAnswer: 1,
    explanation: 'Recall is also called sensitivity or true positive rate (TPR).'
  },
  {
    id: 'em6',
    question: 'What is the F1 score?',
    options: ['Precision + Recall', 'Precision × Recall', '2 × (Precision × Recall) / (Precision + Recall)', 'Precision / Recall'],
    correctAnswer: 2,
    explanation: 'F1 score is the harmonic mean of precision and recall: 2PR / (P + R).'
  },
  {
    id: 'em7',
    question: 'When should you optimize for high precision?',
    options: ['Spam detection', 'Cancer screening', 'Disease detection', 'Fraud detection with costly investigations'],
    correctAnswer: 3,
    explanation: 'High precision is important when false positives are costly - you want to be sure when you flag something as positive.'
  },
  {
    id: 'em8',
    question: 'When should you optimize for high recall?',
    options: ['Spam filtering', 'Cancer/disease screening', 'Recommendation systems', 'Ad placement'],
    correctAnswer: 1,
    explanation: 'High recall is critical when missing positives is dangerous (e.g., cancer) - you want to catch all true cases.'
  },
  {
    id: 'em9',
    question: 'What does ROC stand for?',
    options: ['Rate of Classification', 'Receiver Operating Characteristic', 'Relative Output Curve', 'Regression Optimization Curve'],
    correctAnswer: 1,
    explanation: 'ROC stands for Receiver Operating Characteristic, originally from signal detection theory.'
  },
  {
    id: 'em10',
    question: 'What does the ROC curve plot?',
    options: ['Precision vs Recall', 'TPR vs FPR', 'Accuracy vs Threshold', 'Error vs Complexity'],
    correctAnswer: 1,
    explanation: 'ROC curves plot True Positive Rate (recall) against False Positive Rate at various classification thresholds.'
  },
  {
    id: 'em11',
    question: 'What does AUC-ROC measure?',
    options: ['Model speed', 'Area Under ROC Curve', 'Average Utility Cost', 'Accuracy Under Constraint'],
    correctAnswer: 1,
    explanation: 'AUC is Area Under the ROC Curve, measuring overall model discrimination ability across all thresholds.'
  },
  {
    id: 'em12',
    question: 'What AUC value indicates a perfect classifier?',
    options: ['0', '0.5', '1.0', '100'],
    correctAnswer: 2,
    explanation: 'AUC = 1.0 indicates perfect classification (all positives ranked higher than all negatives).'
  },
  {
    id: 'em13',
    question: 'What does AUC = 0.5 indicate?',
    options: ['Perfect model', 'Random guessing', 'Worst possible', 'Invalid model'],
    correctAnswer: 1,
    explanation: 'AUC = 0.5 means the model performs no better than random guessing.'
  },
  {
    id: 'em14',
    question: 'What is specificity?',
    options: ['TP / (TP + FN)', 'TN / (TN + FP)', 'TP / (TP + FP)', 'FN / (FN + TN)'],
    correctAnswer: 1,
    explanation: 'Specificity = TN / (TN + FP) measures the true negative rate - of actual negatives, how many were correctly identified?'
  },
  {
    id: 'em15',
    question: 'For regression problems, what is MSE?',
    options: ['Mean Squared Error', 'Maximum Standard Error', 'Model Selection Error', 'Median Sampling Error'],
    correctAnswer: 0,
    explanation: 'MSE is Mean Squared Error, the average of squared differences between predictions and actual values.'
  },
  {
    id: 'em16',
    question: 'What is the relationship between MSE and RMSE?',
    options: ['RMSE = MSE²', 'RMSE = √MSE', 'RMSE = 1/MSE', 'RMSE = log(MSE)'],
    correctAnswer: 1,
    explanation: 'RMSE (Root Mean Squared Error) is the square root of MSE, bringing error back to the original scale.'
  },
  {
    id: 'em17',
    question: 'What does MAE stand for in regression?',
    options: ['Maximum Absolute Error', 'Mean Absolute Error', 'Model Average Error', 'Median Approximation Error'],
    correctAnswer: 1,
    explanation: 'MAE is Mean Absolute Error, the average of absolute differences between predictions and actual values.'
  },
  {
    id: 'em18',
    question: 'Which error metric is more robust to outliers?',
    options: ['MSE', 'RMSE', 'MAE', 'They are equally robust'],
    correctAnswer: 2,
    explanation: 'MAE is more robust to outliers than MSE/RMSE because it doesn\'t square the errors.'
  },
  {
    id: 'em19',
    question: 'What does R² (R-squared) measure?',
    options: ['Total error', 'Proportion of variance explained', 'Correlation', 'Mean error'],
    correctAnswer: 1,
    explanation: 'R² measures the proportion of variance in the target variable explained by the model, ranging from 0 to 1.'
  },
  {
    id: 'em20',
    question: 'What is a confusion matrix?',
    options: ['Model architecture', 'Table of TP, TN, FP, FN', 'Loss function', 'Optimization algorithm'],
    correctAnswer: 1,
    explanation: 'A confusion matrix is a table showing true positives, true negatives, false positives, and false negatives.'
  },
  {
    id: 'em21',
    question: 'What is log loss (cross-entropy loss)?',
    options: ['Classification metric measuring probability estimates', 'Regression metric', 'Clustering metric', 'Optimization algorithm'],
    correctAnswer: 0,
    explanation: 'Log loss measures the quality of probability predictions in classification, penalizing confident wrong predictions heavily.'
  },
  {
    id: 'em22',
    question: 'For multi-class classification, what is macro-average?',
    options: ['Weighted by class frequency', 'Simple average across classes', 'Best class score', 'Worst class score'],
    correctAnswer: 1,
    explanation: 'Macro-average computes the metric for each class separately, then takes the unweighted mean across all classes.'
  },
  {
    id: 'em23',
    question: 'What is micro-average in multi-class classification?',
    options: ['Average per class', 'Aggregate TP/FP/FN globally then compute metric', 'Minimum score', 'Maximum score'],
    correctAnswer: 1,
    explanation: 'Micro-average aggregates all true positives, false positives, and false negatives across classes before computing the metric.'
  },
  {
    id: 'em24',
    question: 'What is Cohen\'s Kappa?',
    options: ['Regression metric', 'Agreement metric accounting for chance', 'Neural network layer', 'Optimization rate'],
    correctAnswer: 1,
    explanation: 'Cohen\'s Kappa measures inter-rater agreement (or model performance) while accounting for chance agreement.'
  },
  {
    id: 'em25',
    question: 'In a binary classifier, what is a true positive?',
    options: ['Predicted negative, actually negative', 'Predicted positive, actually positive', 'Predicted positive, actually negative', 'Predicted negative, actually positive'],
    correctAnswer: 1,
    explanation: 'A true positive (TP) occurs when the model correctly predicts the positive class.'
  },
  {
    id: 'em26',
    question: 'What is the Precision-Recall curve useful for?',
    options: ['Only for regression', 'Evaluating classifiers on imbalanced datasets', 'Clustering evaluation', 'Feature selection'],
    correctAnswer: 1,
    explanation: 'PR curves are more informative than ROC curves for imbalanced datasets, showing the tradeoff between precision and recall at different thresholds.'
  },
  {
    id: 'em27',
    question: 'What is the Matthews Correlation Coefficient (MCC)?',
    options: ['Regression metric only', 'Balanced measure for binary classification, even with imbalanced classes', 'Same as accuracy', 'Only for multi-class'],
    correctAnswer: 1,
    explanation: 'MCC ranges from -1 to +1 and considers all confusion matrix elements, providing a balanced measure even for imbalanced datasets.'
  },
  {
    id: 'em28',
    question: 'For ranking problems (e.g., search engines), what metric is commonly used?',
    options: ['Accuracy', 'Mean Average Precision (MAP) or NDCG', 'MSE', 'R²'],
    correctAnswer: 1,
    explanation: 'MAP and NDCG (Normalized Discounted Cumulative Gain) evaluate ranking quality, considering both relevance and position in ranked results.'
  },
  {
    id: 'em29',
    question: 'What does the Brier score measure?',
    options: ['Ranking quality', 'Accuracy of probabilistic predictions', 'Clustering quality', 'Feature importance'],
    correctAnswer: 1,
    explanation: 'Brier score measures the mean squared difference between predicted probabilities and actual outcomes, assessing calibration of probability estimates.'
  },
  {
    id: 'em30',
    question: 'In time series forecasting, what is MAPE?',
    options: ['Mean Absolute Percentage Error', 'Maximum Prediction Error', 'Model Average Performance', 'Median Approximation Error'],
    correctAnswer: 0,
    explanation: 'MAPE (Mean Absolute Percentage Error) expresses forecast error as a percentage of actual values, useful for comparing across different scales.'
  }
];
