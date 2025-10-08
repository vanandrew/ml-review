import { Topic } from '../../../types';

// Import all topics
import { linearRegression } from './linearRegression';
import { logisticRegression } from './logisticRegression';
import { decisionTrees } from './decisionTrees';
import { randomForests } from './randomForests';
import { gradientBoosting } from './gradientBoosting';
import { supportVectorMachines } from './supportVectorMachines';
import { kNearestNeighbors } from './kNearestNeighbors';
import { kMeansClustering } from './kMeansClustering';
import { principalComponentAnalysis } from './principalComponentAnalysis';
import { naiveBayes } from './naiveBayes';

export const classicalMLTopics: Record<string, Topic> = {
  'linear-regression': linearRegression,
  'logistic-regression': logisticRegression,
  'decision-trees': decisionTrees,
  'random-forests': randomForests,
  'gradient-boosting': gradientBoosting,
  'support-vector-machines': supportVectorMachines,
  'k-nearest-neighbors': kNearestNeighbors,
  'k-means-clustering': kMeansClustering,
  'principal-component-analysis': principalComponentAnalysis,
  'naive-bayes': naiveBayes,
};
