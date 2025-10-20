import {
  AIQuestionPrompt,
  AIQuizQuestion,
  FreeFormQuestion,
  AIEvaluationResult,
  MultipleChoiceQuestion,
} from '../types/ai';
import { AIService } from './aiService';
import { handleAIError, logAIError } from '../utils/aiErrorHandler';

/**
 * Claude API Service implementation using Firebase Cloud Functions
 * This calls a Cloud Function which then calls the Claude API server-side
 */
export class ClaudeAIService implements AIService {

  /**
   * Generate a single question using Cloud Function
   */
  async generateQuestion(
    prompt: AIQuestionPrompt,
    apiKey: string
  ): Promise<AIQuizQuestion> {
    try {
      const functionUrl = 'https://us-central1-ml-review.cloudfunctions.net/generateQuestions';

      const requestData = {
        topicId: prompt.topicId,
        topicTitle: prompt.topicTitle,
        topicContent: prompt.topicContent,
        difficulty: prompt.difficulty,
        count: 1,
        apiKey,
      };

      const response = await fetch(functionUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      const data = result.data;

      if (!data.questions || data.questions.length === 0) {
        throw new Error('No questions returned from Cloud Function');
      }

      return data.questions[0];
    } catch (error: any) {
      console.error('[Claude AI Service] Error generating question:', error);

      // Handle specific error messages
      if (error.message.includes('Invalid API key')) {
        throw new Error('Invalid API key');
      }

      if (error.message.includes('Rate limit')) {
        throw new Error('Rate limit exceeded. Please try again later.');
      }

      const aiError = handleAIError(error);
      logAIError(aiError);
      throw aiError;
    }
  }

  /**
   * Generate multiple questions in batch using Cloud Function
   */
  async generateQuestionBatch(
    prompt: AIQuestionPrompt,
    count: number,
    apiKey: string
  ): Promise<AIQuizQuestion[]> {
    try {
      console.log(`[Claude AI Service] Generating ${count} questions via Cloud Function...`);
      console.log('[Claude AI Service] API key length:', apiKey?.length);
      console.log('[Claude AI Service] API key starts with:', apiKey?.substring(0, 10));

      const functionUrl = 'https://us-central1-ml-review.cloudfunctions.net/generateQuestions';

      const requestData = {
        topicId: prompt.topicId,
        topicTitle: prompt.topicTitle,
        topicContent: prompt.topicContent,
        difficulty: prompt.difficulty,
        count,
        apiKey,
      };

      console.log('[Claude AI Service] Calling function with data:', {
        topicId: prompt.topicId,
        topicTitle: prompt.topicTitle,
        difficulty: prompt.difficulty,
        count,
        contentLength: prompt.topicContent?.length
      });

      const response = await fetch(functionUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      const data = result.data;

      console.log('[Claude AI Service] Function call completed successfully');
      console.log('[Claude AI Service] Result:', result);

      if (!data.questions || data.questions.length === 0) {
        throw new Error('No questions returned from Cloud Function');
      }

      // Log detailed cost information
      const metadata = data.metadata;
      if (metadata.inputTokens && metadata.outputTokens) {
        console.log(
          `[Claude AI Service] Successfully generated ${data.questions.length} questions. ` +
          `Cost: $${metadata.actualCost.toFixed(4)} ` +
          `(${metadata.inputTokens} input + ${metadata.outputTokens} output = ${metadata.totalTokens} tokens)`
        );
      } else {
        console.log(
          `[Claude AI Service] Successfully generated ${data.questions.length} questions. ` +
          `Estimated cost: $${metadata.estimatedCost.toFixed(3)}`
        );
      }

      return data.questions;
    } catch (error: any) {
      console.error('[Claude AI Service] Error generating questions:', error);

      // Handle specific error messages
      if (error.message.includes('Invalid API key')) {
        throw new Error('Invalid API key');
      }

      if (error.message.includes('Rate limit')) {
        throw new Error('Rate limit exceeded. Please try again later.');
      }

      const aiError = handleAIError(error);
      logAIError(aiError);
      throw aiError;
    }
  }

  /**
   * Evaluate a free-form answer (not yet implemented for Cloud Functions)
   */
  async evaluateAnswer(
    question: FreeFormQuestion,
    userAnswer: string,
    apiKey: string
  ): Promise<AIEvaluationResult> {
    throw new Error('Free-form question evaluation not yet implemented');
  }
}
