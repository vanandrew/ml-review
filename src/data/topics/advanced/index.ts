import { Topic } from '../../../types';

// Import all topics
import { generativeAdversarialNetworks } from './generativeAdversarialNetworks';
import { variationalAutoencoders } from './variationalAutoencoders';
import { reinforcementLearningBasics } from './reinforcementLearningBasics';
import { modelCompression } from './modelCompression';
import { federatedLearning } from './federatedLearning';
import { fewShotLearning } from './fewShotLearning';
import { multiModalModels } from './multiModalModels';

export const advancedTopics: Record<string, Topic> = {
  'generative-adversarial-networks': generativeAdversarialNetworks,
  'variational-autoencoders': variationalAutoencoders,
  'reinforcement-learning-basics': reinforcementLearningBasics,
  'model-compression': modelCompression,
  'federated-learning': federatedLearning,
  'few-shot-learning': fewShotLearning,
  'multi-modal-models': multiModalModels,
};
