import {
  AIQuestionPrompt,
  AIQuizQuestion,
  FreeFormQuestion,
  AIEvaluationResult,
} from '../types/ai';
import { MockAIService } from './mockAIService';
import { ClaudeAIService } from './claudeAIService';

/**
 * Interface for AI question generation and evaluation services
 */
export interface AIService {
  /**
   * Generate a new quiz question based on the prompt
   */
  generateQuestion(
    prompt: AIQuestionPrompt,
    apiKey: string
  ): Promise<AIQuizQuestion>;

  /**
   * Evaluate a free-form answer
   */
  evaluateAnswer(
    question: FreeFormQuestion,
    userAnswer: string,
    apiKey: string
  ): Promise<AIEvaluationResult>;

  /**
   * Generate multiple questions in batch
   */
  generateQuestionBatch(
    prompt: AIQuestionPrompt,
    count: number,
    apiKey: string
  ): Promise<AIQuizQuestion[]>;
}

/**
 * Creates an AI service instance based on environment configuration
 */
export const createAIService = (useMock: boolean = false): AIService => {
  // Force mock mode via parameter
  if (useMock) {
    console.log('[AI Service] Using Mock AI Service (forced)');
    return new MockAIService();
  }

  // Check for explicit mock mode via environment variable
  if (import.meta.env.VITE_USE_MOCK_AI === 'true') {
    console.log('[AI Service] Using Mock AI Service (forced via VITE_USE_MOCK_AI)');
    return new MockAIService();
  }

  // Always use Claude AI service (dev and production)
  console.log('[AI Service] Using Claude AI Service');
  return new ClaudeAIService();
};

/**
 * Singleton instance of the AI service
 */
let aiServiceInstance: AIService | null = null;

/**
 * Gets the singleton AI service instance
 */
export const getAIService = (): AIService => {
  if (!aiServiceInstance) {
    aiServiceInstance = createAIService();
  }
  return aiServiceInstance;
};

/**
 * Resets the AI service instance (useful for testing)
 */
export const resetAIService = (): void => {
  aiServiceInstance = null;
};
