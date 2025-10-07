import { QuizQuestion } from '../../types';
// Foundations
import { supervisedVsUnsupervisedVsReinforcementQuestions } from './supervisedVsUnsupervisedVsReinforcement';
import { biasVarianceTradeoffQuestions } from './biasVarianceTradeoff';
import { trainValidationTestSplitQuestions } from './trainValidationTestSplit';
import { overfittingUnderfittingQuestions } from './overfittingUnderfitting';
import { regularizationQuestions } from './regularization';
import { crossValidationQuestions } from './crossValidation';
import { hyperparameterTuningQuestions } from './hyperparameterTuning';
import { evaluationMetricsQuestions } from './evaluationMetrics';
// Classical ML
import { linearRegressionQuestions, logisticRegressionQuestions } from './classicalML';
import { decisionTreesQuestions, randomForestsQuestions, gradientBoostingQuestions } from './classicalML_part2';
import { svmQuestions, knnQuestions, kMeansQuestions } from './classicalML_part3';
import { pcaQuestions, naiveBayesQuestions } from './classicalML_part4';
// Neural Networks
import { perceptronQuestions, mlpQuestions, activationFunctionsQuestions, neuralNetworksScenarioQuestions } from './neuralNetworks';
import { backpropagationQuestions, gradientDescentQuestions, batchNormQuestions, lossFunctionsQuestions } from './neuralNetworks_part2';
// Computer Vision
import { cnnQuestions, poolingQuestions, classicArchitecturesQuestions } from './computerVision';
import { transferLearningQuestions, objectDetectionQuestions, imageSegmentationQuestions } from './computerVision_part2';
// NLP
import { wordEmbeddingsQuestions, rnnQuestions, lstmGruQuestions } from './nlp';
import { seq2seqQuestions, attentionQuestions, encoderDecoderQuestions } from './nlp_part2';
// Transformers
import { transformerArchitectureQuestions, selfAttentionQuestions } from './transformers';
import { positionalEncodingQuestions, bertQuestions, gptQuestions } from './transformers_part2';
import { t5BartQuestions, fineTuningPromptingQuestions, llmQuestions } from './transformers_part3';
// Advanced Topics
import { ganQuestions, vaeQuestions, rlBasicsQuestions } from './advancedTopics';
import { modelCompressionQuestions, federatedLearningQuestions, fewShotLearningQuestions, multiModalQuestions } from './advancedTopics_part2';
// ML Systems
import { featureEngineeringQuestions, dataPreprocessingQuestions, imbalancedDataQuestions } from './mlSystems';
import { modelDeploymentQuestions, abTestingQuestions, monitoringDriftQuestions, scalingMLQuestions } from './mlSystems_part2';

// Map of topic IDs to their quiz question pools
export const quizQuestionPools: Record<string, QuizQuestion[]> = {
  // Foundations
  'supervised-vs-unsupervised-vs-reinforcement': supervisedVsUnsupervisedVsReinforcementQuestions,
  'bias-variance-tradeoff': biasVarianceTradeoffQuestions,
  'train-validation-test-split': trainValidationTestSplitQuestions,
  'overfitting-underfitting': overfittingUnderfittingQuestions,
  'regularization': regularizationQuestions,
  'cross-validation': crossValidationQuestions,
  'hyperparameter-tuning': hyperparameterTuningQuestions,
  'evaluation-metrics': evaluationMetricsQuestions,
  // Classical ML
  'linear-regression': linearRegressionQuestions,
  'logistic-regression': logisticRegressionQuestions,
  'decision-trees': decisionTreesQuestions,
  'random-forests': randomForestsQuestions,
  'gradient-boosting': gradientBoostingQuestions,
  'svm': svmQuestions,
  'knn': knnQuestions,
  'k-means': kMeansQuestions,
  'pca': pcaQuestions,
  'naive-bayes': naiveBayesQuestions,
  // Neural Networks
  'perceptron': perceptronQuestions,
  'mlp': [...mlpQuestions, ...neuralNetworksScenarioQuestions.slice(0, 3)],
  'activation-functions': [...activationFunctionsQuestions, ...neuralNetworksScenarioQuestions.slice(3, 5)],
  'backpropagation': [...backpropagationQuestions, ...neuralNetworksScenarioQuestions.slice(5, 7)],
  'gradient-descent': [...gradientDescentQuestions, ...neuralNetworksScenarioQuestions.slice(7, 8)],
  'batch-normalization': [...batchNormQuestions, ...neuralNetworksScenarioQuestions.slice(1, 2).concat(neuralNetworksScenarioQuestions.slice(4, 5))],
  'loss-functions': [...lossFunctionsQuestions, ...neuralNetworksScenarioQuestions.slice(8, 10)],
  // Computer Vision
  'cnns': cnnQuestions,
  'pooling-layers': poolingQuestions,
  'classic-architectures': classicArchitecturesQuestions,
  'transfer-learning': transferLearningQuestions,
  'object-detection': objectDetectionQuestions,
  'image-segmentation': imageSegmentationQuestions,
  // NLP
  'word-embeddings': wordEmbeddingsQuestions,
  'rnns': rnnQuestions,
  'lstm-gru': lstmGruQuestions,
  'seq2seq': seq2seqQuestions,
  'attention': attentionQuestions,
  'encoder-decoder': encoderDecoderQuestions,
  // Transformers
  'transformer-architecture': transformerArchitectureQuestions,
  'self-attention': selfAttentionQuestions,
  'positional-encoding': positionalEncodingQuestions,
  'bert': bertQuestions,
  'gpt': gptQuestions,
  't5-bart': t5BartQuestions,
  'fine-tuning-prompting': fineTuningPromptingQuestions,
  'llms': llmQuestions,
  // Advanced Topics
  'gans': ganQuestions,
  'vaes': vaeQuestions,
  'rl-basics': rlBasicsQuestions,
  'model-compression': modelCompressionQuestions,
  'federated-learning': federatedLearningQuestions,
  'few-shot-learning': fewShotLearningQuestions,
  'multi-modal': multiModalQuestions,
  // ML Systems
  'feature-engineering': featureEngineeringQuestions,
  'data-preprocessing': dataPreprocessingQuestions,
  'imbalanced-data': imbalancedDataQuestions,
  'model-deployment': modelDeploymentQuestions,
  'ab-testing': abTestingQuestions,
  'model-monitoring': monitoringDriftQuestions,
  'scaling-ml': scalingMLQuestions,
};

// Helper function to get questions for a topic
export function getQuizQuestionsForTopic(topicId: string): QuizQuestion[] {
  return quizQuestionPools[topicId] || [];
}
