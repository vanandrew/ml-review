import {
  AIQuestionPrompt,
  AIQuizQuestion,
  MultipleChoiceQuestion,
  FreeFormQuestion,
  AIEvaluationResult,
} from '../types/ai';
import { AIService } from './aiService';

/**
 * Helper function to simulate async delay
 */
const sleep = (ms: number): Promise<void> => {
  return new Promise((resolve) => setTimeout(resolve, ms));
};

/**
 * Generates a random ID for questions
 */
const generateId = (): string => {
  return `mock-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

/**
 * Mock AI Service for development and testing
 * Simulates AI API calls without actual API costs
 */
export class MockAIService implements AIService {
  private mockQuestions: Record<string, MultipleChoiceQuestion[]> = {
    'linear-regression': [
      {
        id: generateId(),
        type: 'multiple-choice',
        question: 'What is the primary goal of linear regression?',
        options: [
          'Minimize the sum of squared residuals',
          'Maximize the variance in predictions',
          'Create a random fit through data points',
          'Eliminate all prediction errors',
        ],
        correctAnswer: 0,
        explanation: 'Linear regression aims to find the best-fit line by minimizing the sum of squared residuals (differences between predicted and actual values).',
        source: 'ai-generated',
        generatedAt: new Date(),
      },
      {
        id: generateId(),
        type: 'multiple-choice',
        question: 'Which assumption is NOT required for ordinary least squares regression?',
        options: [
          'Linear relationship between variables',
          'Homoscedasticity of residuals',
          'Perfect correlation between features',
          'Independence of observations',
        ],
        correctAnswer: 2,
        explanation: 'Perfect correlation between features (multicollinearity) is actually a problem to avoid, not a required assumption.',
        source: 'ai-generated',
        generatedAt: new Date(),
      },
    ],
    'neural-networks': [
      {
        id: generateId(),
        type: 'multiple-choice',
        question: 'What is the purpose of an activation function in a neural network?',
        options: [
          'To introduce non-linearity into the model',
          'To reduce the learning rate',
          'To increase training time',
          'To eliminate gradients',
        ],
        correctAnswer: 0,
        explanation: 'Activation functions introduce non-linearity, allowing neural networks to learn complex patterns beyond linear relationships.',
        source: 'ai-generated',
        generatedAt: new Date(),
      },
    ],
    'gradient-descent': [
      {
        id: generateId(),
        type: 'multiple-choice',
        question: 'What does the learning rate control in gradient descent?',
        options: [
          'The size of steps taken toward the minimum',
          'The number of iterations required',
          'The initial parameter values',
          'The complexity of the model',
        ],
        correctAnswer: 0,
        explanation: 'The learning rate determines how large of a step we take in the direction of steepest descent.',
        source: 'ai-generated',
        generatedAt: new Date(),
      },
    ],
  };

  private mockFreeFormQuestions: Record<string, FreeFormQuestion[]> = {
    'linear-regression': [
      {
        id: generateId(),
        type: 'free-form',
        question: 'Explain the concept of overfitting in the context of linear regression and how you would detect it.',
        sampleAnswer: 'Overfitting occurs when a model learns the training data too well, including noise and outliers, resulting in poor generalization to new data. In linear regression with polynomial features, this manifests as a model with very high training accuracy but poor test accuracy. You can detect overfitting by comparing training and validation errors - a large gap indicates overfitting. Regularization techniques like Ridge or Lasso can help prevent it.',
        evaluationCriteria: [
          'Definition of overfitting',
          'Impact on generalization',
          'Detection methods (train/test gap)',
          'Prevention strategies (regularization)',
        ],
        source: 'ai-generated',
        generatedAt: new Date(),
      },
    ],
  };

  /**
   * Generate a mock quiz question
   */
  async generateQuestion(
    prompt: AIQuestionPrompt,
    _apiKey: string
  ): Promise<AIQuizQuestion> {
    // Simulate API delay (2-4 seconds)
    const delay = 2000 + Math.random() * 2000;
    await sleep(delay);

    // Get questions for this topic
    const questions = prompt.questionType === 'free-form'
      ? this.mockFreeFormQuestions[prompt.topicId] || []
      : this.mockQuestions[prompt.topicId] || [];

    // Return a random question from the pool
    if (questions.length > 0) {
      const randomIndex = Math.floor(Math.random() * questions.length);
      return questions[randomIndex];
    }

    // Generate a generic fallback question
    if (prompt.questionType === 'free-form') {
      return {
        id: generateId(),
        type: 'free-form',
        question: `Explain the key concepts and applications of ${prompt.topicTitle}.`,
        sampleAnswer: `${prompt.topicTitle} is an important concept in machine learning with various practical applications.`,
        evaluationCriteria: [
          'Understanding of core concepts',
          'Real-world applications',
          'Clarity of explanation',
        ],
        source: 'ai-generated',
        generatedAt: new Date(),
      };
    }

    return {
      id: generateId(),
      type: 'multiple-choice',
      question: `Which of the following best describes ${prompt.topicTitle}?`,
      options: [
        `A core concept in ${prompt.difficulty} machine learning`,
        'An outdated technique rarely used today',
        'A simple algorithm with no practical applications',
        'A purely theoretical construct',
      ],
      correctAnswer: 0,
      explanation: `This is a mock question about ${prompt.topicTitle} at ${prompt.difficulty} level.`,
      source: 'ai-generated',
      generatedAt: new Date(),
    };
  }

  /**
   * Evaluate a free-form answer using simple keyword matching
   */
  async evaluateAnswer(
    question: FreeFormQuestion,
    userAnswer: string,
    _apiKey: string
  ): Promise<AIEvaluationResult> {
    // Simulate evaluation delay (3-5 seconds)
    const delay = 3000 + Math.random() * 2000;
    await sleep(delay);

    // Simple keyword matching for mock evaluation
    const keywords = question.sampleAnswer?.toLowerCase().split(/\s+/) || [];
    const userWords = userAnswer.toLowerCase().split(/\s+/);

    const matchCount = keywords.filter(keyword =>
      userWords.some(word => word.includes(keyword) || keyword.includes(word))
    ).length;

    const matchRatio = keywords.length > 0 ? matchCount / keywords.length : 0;
    const score = Math.min(100, Math.round(matchRatio * 100));

    // Generate feedback based on score
    let feedback: string;
    const strengths: string[] = [];
    const improvements: string[] = [];

    if (score >= 80) {
      feedback = 'Excellent answer! You demonstrated a strong understanding of the concept.';
      strengths.push('Comprehensive coverage of key points');
      strengths.push('Clear and well-structured explanation');
      improvements.push('Consider adding more specific examples');
    } else if (score >= 60) {
      feedback = 'Good answer with room for improvement. You covered the main ideas but could expand on some details.';
      strengths.push('Addressed the core concepts');
      improvements.push('Include more technical details');
      improvements.push('Expand on practical applications');
    } else if (score >= 40) {
      feedback = 'Your answer touches on some relevant points but needs more depth and accuracy.';
      strengths.push('Attempted to address the question');
      improvements.push('Review the core concepts more thoroughly');
      improvements.push('Provide more specific examples');
      improvements.push('Ensure technical accuracy');
    } else {
      feedback = 'Your answer needs significant improvement. Please review the topic material.';
      improvements.push('Study the fundamental concepts');
      improvements.push('Use more precise terminology');
      improvements.push('Provide concrete examples');
    }

    return {
      score,
      feedback,
      strengths,
      improvements,
      evaluatedAt: new Date(),
    };
  }

  /**
   * Generate multiple questions in batch
   */
  async generateQuestionBatch(
    prompt: AIQuestionPrompt,
    count: number,
    apiKey: string
  ): Promise<AIQuizQuestion[]> {
    const questions: AIQuizQuestion[] = [];

    for (let i = 0; i < count; i++) {
      // Stagger the generation to simulate realistic timing
      if (i > 0) {
        await sleep(500);
      }
      const question = await this.generateQuestion(prompt, apiKey);
      questions.push(question);
    }

    return questions;
  }

  /**
   * Add custom mock questions for testing
   */
  addMockQuestion(topicId: string, question: MultipleChoiceQuestion): void {
    if (!this.mockQuestions[topicId]) {
      this.mockQuestions[topicId] = [];
    }
    this.mockQuestions[topicId].push(question);
  }

  /**
   * Add custom mock free-form questions for testing
   */
  addMockFreeFormQuestion(topicId: string, question: FreeFormQuestion): void {
    if (!this.mockFreeFormQuestions[topicId]) {
      this.mockFreeFormQuestions[topicId] = [];
    }
    this.mockFreeFormQuestions[topicId].push(question);
  }

  /**
   * Clear all mock questions
   */
  clearMockQuestions(): void {
    this.mockQuestions = {};
    this.mockFreeFormQuestions = {};
  }
}
