import { Topic } from '../../../types';

// Import all topics
import { featureEngineering } from './featureEngineering';
import { dataPreprocessingNormalization } from './dataPreprocessingNormalization';
import { handlingImbalancedData } from './handlingImbalancedData';
import { modelDeployment } from './modelDeployment';
import { abTesting } from './abTesting';
import { modelMonitoringDriftDetection } from './modelMonitoringDriftDetection';
import { scalingOptimization } from './scalingOptimization';

export const mlSystemsTopics: Record<string, Topic> = {
  'feature-engineering': featureEngineering,
  'data-preprocessing-normalization': dataPreprocessingNormalization,
  'handling-imbalanced-data': handlingImbalancedData,
  'model-deployment': modelDeployment,
  'ab-testing': abTesting,
  'model-monitoring-drift-detection': modelMonitoringDriftDetection,
  'scaling-optimization': scalingOptimization,
};
