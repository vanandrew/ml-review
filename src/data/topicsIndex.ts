import { Topic } from '../types';

// Import topics by category
import { foundationsTopics } from './topics/foundations';
import { classicalMLTopics } from './topics/classicalML';
import { neuralNetworksTopics } from './topics/neuralNetworks';
import { computerVisionTopics } from './topics/computerVision';
import { nlpTopics } from './topics/nlp';
import { transformersTopics } from './topics/transformers';
import { advancedTopics } from './topics/advanced';
import { mlSystemsTopics } from './topics/mlSystems';

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