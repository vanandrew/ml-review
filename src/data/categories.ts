import { Category } from '../types';

export const categories: Category[] = [
  {
    id: 'foundations',
    title: 'Foundations',
    color: 'blue',
    topics: [
      'supervised-vs-unsupervised-vs-reinforcement',
      'train-validation-test-split',
      'bias-variance-tradeoff',
      'overfitting-underfitting',
      'regularization',
      'cross-validation',
      'evaluation-metrics'
    ]
  },
  {
    id: 'classical-ml',
    title: 'Classical ML',
    color: 'green',
    topics: [
      'linear-regression',
      'logistic-regression',
      'decision-trees',
      'random-forests',
      'gradient-boosting',
      'support-vector-machines',
      'k-nearest-neighbors',
      'k-means-clustering',
      'principal-component-analysis',
      'naive-bayes'
    ]
  },
  {
    id: 'neural-networks',
    title: 'Neural Networks',
    color: 'purple',
    topics: [
      'perceptron',
      'multi-layer-perceptron',
      'activation-functions',
      'backpropagation',
      'gradient-descent',
      'batch-normalization',
      'loss-functions'
    ]
  },
  {
    id: 'computer-vision',
    title: 'Computer Vision',
    color: 'orange',
    topics: [
      'convolutional-neural-networks',
      'pooling-layers',
      'classic-architectures',
      'transfer-learning',
      'object-detection',
      'image-segmentation'
    ]
  },
  {
    id: 'nlp',
    title: 'NLP',
    color: 'pink',
    topics: [
      'word-embeddings',
      'recurrent-neural-networks',
      'lstm-gru',
      'seq2seq-models',
      'attention-mechanism',
      'encoder-decoder-architecture'
    ]
  },
  {
    id: 'transformers',
    title: 'Transformers & Modern NLP',
    color: 'indigo',
    topics: [
      'transformer-architecture',
      'self-attention-multi-head',
      'positional-encoding',
      'bert',
      'gpt',
      't5-bart',
      'fine-tuning-vs-prompt-engineering',
      'large-language-models'
    ]
  },
  {
    id: 'advanced',
    title: 'Advanced Topics',
    color: 'red',
    topics: [
      'generative-adversarial-networks',
      'variational-autoencoders',
      'reinforcement-learning-basics',
      'model-compression',
      'federated-learning',
      'few-shot-learning',
      'multi-modal-models'
    ]
  },
  {
    id: 'ml-systems',
    title: 'ML Systems & Production',
    color: 'yellow',
    topics: [
      'feature-engineering',
      'data-preprocessing-normalization',
      'handling-imbalanced-data',
      'model-deployment',
      'ab-testing',
      'model-monitoring-drift-detection',
      'scaling-optimization'
    ]
  }
];