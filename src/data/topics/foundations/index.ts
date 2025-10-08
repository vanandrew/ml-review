import { Topic } from '../../../types';

// Import all topics
import { supervisedVsUnsupervisedVsReinforcement } from './supervisedVsUnsupervisedVsReinforcement';
import { biasVarianceTradeoff } from './biasVarianceTradeoff';
import { trainValidationTestSplit } from './trainValidationTestSplit';
import { overfittingUnderfitting } from './overfittingUnderfitting';
import { regularization } from './regularization';
import { crossValidation } from './crossValidation';
import { evaluationMetrics } from './evaluationMetrics';
import { hyperparameterTuning } from './hyperparameterTuning';

export const foundationsTopics: Record<string, Topic> = {
  'supervised-vs-unsupervised-vs-reinforcement': supervisedVsUnsupervisedVsReinforcement,
  'bias-variance-tradeoff': biasVarianceTradeoff,
  'train-validation-test-split': trainValidationTestSplit,
  'overfitting-underfitting': overfittingUnderfitting,
  'regularization': regularization,
  'cross-validation': crossValidation,
  'evaluation-metrics': evaluationMetrics,
  'hyperparameter-tuning': hyperparameterTuning,
};
