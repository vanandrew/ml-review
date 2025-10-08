import { Topic } from '../types';

// Import topics by category (now from subdirectories)
import { foundationsTopics } from './topics/foundations/index';
import { classicalMLTopics } from './topics/classicalML/index';
import { neuralNetworksTopics } from './topics/neuralNetworks/index';
import { computerVisionTopics } from './topics/computerVision/index';
import { nlpTopics } from './topics/nlp/index';
import { transformersTopics } from './topics/transformers/index';
import { advancedTopics } from './topics/advanced/index';
import { mlSystemsTopics } from './topics/mlSystems/index';

// Combine all topics
export const topics: Record<string, Topic> = {
  ...foundationsTopics,
  ...classicalMLTopics,
  ...neuralNetworksTopics,
  ...computerVisionTopics,
  ...nlpTopics,
  ...transformersTopics,
  ...advancedTopics,
  ...mlSystemsTopics
};

export const getTopicById = (id: string): Topic | undefined => {
  return topics[id];
};