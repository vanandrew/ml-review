import { Topic } from '../../../types';

// Import all topics
import { perceptron } from './perceptron';
import { multiLayerPerceptron } from './multiLayerPerceptron';
import { activationFunctions } from './activationFunctions';
import { backpropagation } from './backpropagation';
import { gradientDescent } from './gradientDescent';
import { batchNormalization } from './batchNormalization';
import { lossFunctions } from './lossFunctions';

export const neuralNetworksTopics: Record<string, Topic> = {
  'perceptron': perceptron,
  'multi-layer-perceptron': multiLayerPerceptron,
  'activation-functions': activationFunctions,
  'backpropagation': backpropagation,
  'gradient-descent': gradientDescent,
  'batch-normalization': batchNormalization,
  'loss-functions': lossFunctions,
};
