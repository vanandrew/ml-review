import { Topic } from '../../../types';

// Import all topics
import { convolutionalNeuralNetworks } from './convolutionalNeuralNetworks';
import { poolingLayers } from './poolingLayers';
import { classicArchitectures } from './classicArchitectures';
import { transferLearning } from './transferLearning';
import { objectDetection } from './objectDetection';
import { imageSegmentation } from './imageSegmentation';

export const computerVisionTopics: Record<string, Topic> = {
  'convolutional-neural-networks': convolutionalNeuralNetworks,
  'pooling-layers': poolingLayers,
  'classic-architectures': classicArchitectures,
  'transfer-learning': transferLearning,
  'object-detection': objectDetection,
  'image-segmentation': imageSegmentation,
};
