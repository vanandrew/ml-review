import { QuizQuestion } from '../../types';

export const supervisedVsUnsupervisedVsReinforcementQuestions: QuizQuestion[] = [
  {
    id: 'suvsr1',
    question: 'Which type of learning uses labeled training data?',
    options: ['Supervised Learning', 'Unsupervised Learning', 'Reinforcement Learning', 'Semi-supervised Learning'],
    correctAnswer: 0,
    explanation: 'Supervised learning uses labeled training data where both input features and correct output labels are provided.'
  },
  {
    id: 'suvsr2',
    question: 'What is the main goal of unsupervised learning?',
    options: ['Predict future values', 'Discover hidden patterns', 'Maximize rewards', 'Classify data points'],
    correctAnswer: 1,
    explanation: 'Unsupervised learning aims to discover hidden patterns or structures in data without using labeled examples.'
  },
  {
    id: 'suvsr3',
    question: 'In reinforcement learning, what guides the learning process?',
    options: ['Labeled examples', 'Hidden patterns', 'Rewards and penalties', 'Feature correlations'],
    correctAnswer: 2,
    explanation: 'Reinforcement learning uses rewards and penalties as feedback to guide the agent\'s learning process.'
  },
  {
    id: 'suvsr4',
    question: 'Which learning paradigm is best suited for email spam detection?',
    options: ['Unsupervised Learning', 'Supervised Learning', 'Reinforcement Learning', 'Transfer Learning'],
    correctAnswer: 1,
    explanation: 'Email spam detection requires labeled examples (spam/not spam), making it a supervised learning problem.'
  },
  {
    id: 'suvsr5',
    question: 'K-means clustering is an example of which type of learning?',
    options: ['Supervised Learning', 'Reinforcement Learning', 'Unsupervised Learning', 'Semi-supervised Learning'],
    correctAnswer: 2,
    explanation: 'K-means is an unsupervised learning algorithm that finds clusters in data without using labels.'
  },
  {
    id: 'suvsr6',
    question: 'Which scenario is most appropriate for reinforcement learning?',
    options: ['Predicting house prices', 'Segmenting customers', 'Training a robot to walk', 'Classifying images'],
    correctAnswer: 2,
    explanation: 'Robot locomotion involves sequential decision-making and learning from interaction with the environment, making it ideal for reinforcement learning.'
  },
  {
    id: 'suvsr7',
    question: 'What distinguishes semi-supervised learning from other paradigms?',
    options: ['Uses only labeled data', 'Uses only unlabeled data', 'Uses both labeled and unlabeled data', 'Uses reward signals'],
    correctAnswer: 2,
    explanation: 'Semi-supervised learning combines a small amount of labeled data with a large amount of unlabeled data during training.'
  },
  {
    id: 'suvsr8',
    question: 'In supervised learning, what is the target variable also called?',
    options: ['Feature', 'Label', 'Cluster', 'Policy'],
    correctAnswer: 1,
    explanation: 'The target variable in supervised learning is commonly referred to as the label or ground truth.'
  },
  {
    id: 'suvsr9',
    question: 'Which algorithm is used in reinforcement learning to learn optimal policies?',
    options: ['Linear Regression', 'K-means', 'Q-Learning', 'PCA'],
    correctAnswer: 2,
    explanation: 'Q-Learning is a classic reinforcement learning algorithm that learns optimal action-selection policies.'
  },
  {
    id: 'suvsr10',
    question: 'What is the main challenge in unsupervised learning?',
    options: ['Lack of computational resources', 'No ground truth to evaluate against', 'Too much labeled data', 'Insufficient reward signals'],
    correctAnswer: 1,
    explanation: 'Unsupervised learning lacks ground truth labels, making it difficult to objectively evaluate the quality of discovered patterns.'
  },
  {
    id: 'suvsr11',
    question: 'Which type of learning is most commonly used in game playing AI (like AlphaGo)?',
    options: ['Supervised Learning only', 'Unsupervised Learning only', 'Reinforcement Learning', 'Transfer Learning only'],
    correctAnswer: 2,
    explanation: 'Game playing AI primarily uses reinforcement learning where agents learn strategies through self-play and reward maximization.'
  },
  {
    id: 'suvsr12',
    question: 'Dimensionality reduction using PCA is an example of:',
    options: ['Supervised Learning', 'Reinforcement Learning', 'Unsupervised Learning', 'Active Learning'],
    correctAnswer: 2,
    explanation: 'PCA is an unsupervised technique that finds principal components without using any labels.'
  },
  {
    id: 'suvsr13',
    question: 'What type of feedback does a supervised learning model receive?',
    options: ['Delayed rewards', 'Cluster assignments', 'Correct outputs for inputs', 'No feedback'],
    correctAnswer: 2,
    explanation: 'Supervised learning receives direct feedback in the form of correct outputs (labels) for each training input.'
  },
  {
    id: 'suvsr14',
    question: 'Which application is best suited for unsupervised learning?',
    options: ['Credit card fraud detection with labeled examples', 'Customer segmentation without predefined groups', 'Predicting stock prices', 'Medical diagnosis with patient records'],
    correctAnswer: 1,
    explanation: 'Customer segmentation without predefined groups is ideal for unsupervised learning, which discovers natural groupings in data.'
  },
  {
    id: 'suvsr15',
    question: 'In reinforcement learning, what is an "episode"?',
    options: ['A single training example', 'A complete sequence from start to terminal state', 'One iteration of gradient descent', 'A cluster of similar states'],
    correctAnswer: 1,
    explanation: 'An episode in RL is a complete sequence of states, actions, and rewards from an initial state to a terminal state.'
  },
  {
    id: 'suvsr16',
    question: 'What is "exploration vs exploitation" a dilemma in?',
    options: ['Supervised Learning', 'Unsupervised Learning', 'Reinforcement Learning', 'Transfer Learning'],
    correctAnswer: 2,
    explanation: 'The exploration-exploitation tradeoff is a fundamental challenge in reinforcement learning, balancing trying new actions vs. using known good actions.'
  },
  {
    id: 'suvsr17',
    question: 'Which statement about supervised learning is FALSE?',
    options: ['Requires labeled training data', 'Can be used for classification', 'Discovers hidden patterns without labels', 'Can be used for regression'],
    correctAnswer: 2,
    explanation: 'Discovering hidden patterns without labels is the goal of unsupervised learning, not supervised learning.'
  },
  {
    id: 'suvsr18',
    question: 'Anomaly detection in network traffic (without prior labels) uses:',
    options: ['Supervised Learning', 'Unsupervised Learning', 'Reinforcement Learning', 'None of the above'],
    correctAnswer: 1,
    explanation: 'Without labeled examples of anomalies, unsupervised learning techniques are used to identify unusual patterns.'
  },
  {
    id: 'suvsr19',
    question: 'What does a reinforcement learning agent aim to maximize?',
    options: ['Prediction accuracy', 'Cluster cohesion', 'Cumulative reward', 'Classification error'],
    correctAnswer: 2,
    explanation: 'RL agents aim to maximize the cumulative (or expected) reward over time through their action choices.'
  },
  {
    id: 'suvsr20',
    question: 'Which scenario requires the MOST human labeling effort?',
    options: ['Pure supervised learning', 'Pure unsupervised learning', 'Reinforcement learning with simulated environments', 'Self-supervised learning'],
    correctAnswer: 0,
    explanation: 'Pure supervised learning requires human-labeled examples for all training data, which is the most labor-intensive.'
  }
];
