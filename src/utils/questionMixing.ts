import { QuizQuestion, Topic } from '../types';
import { AIQuizQuestion, AIQuestionPrompt, CachedAIQuestion } from '../types/ai';
import { AISettings } from '../types';
import { getAIService } from '../services/aiService';
import { loadAPIKey, loadCachedQuestions, saveCachedQuestion } from './aiFirestore';
import { CostTracker } from './aiCostTracking';
import { getCondensedTopicContent } from './topicSummarization';

/**
 * Converts an AI quiz question to the standard QuizQuestion format
 */
function convertAIQuestionToQuizQuestion(aiQuestion: AIQuizQuestion): QuizQuestion {
  if (aiQuestion.type === 'multiple-choice') {
    return {
      id: aiQuestion.id,
      question: aiQuestion.question,
      options: aiQuestion.options,
      correctAnswer: aiQuestion.correctAnswer,
      explanation: aiQuestion.explanation,
    };
  }

  // Free-form questions not yet supported in Phase 1
  throw new Error('Free-form questions not yet supported');
}

/**
 * Result of question generation with metadata
 */
export interface QuestionMixResult {
  questions: QuizQuestion[];
  metadata: {
    staticCount: number;
    aiCount: number;
    newlyGenerated: number;
    costEstimate: number;
    error?: string;
  };
}

/**
 * Generate AI quiz questions for a topic
 *
 * @param staticQuestions - Unused, kept for backward compatibility (pass empty array)
 * @param topicId - ID of the topic (for tracking)
 * @param topicTitle - Title of the topic (for AI context)
 * @param topicContent - HTML content of the topic (for AI context)
 * @param totalCount - Total number of questions to generate
 * @param userId - User ID (for loading API key)
 * @param aiSettings - User's AI settings (difficulty, provider)
 * @param aiCostTracking - Current cost tracking data
 * @returns Promise<QuestionMixResult> - AI-generated questions with metadata
 */
export async function mixQuestionsWithAI(
  staticQuestions: QuizQuestion[], // Unused, kept for backward compatibility
  topicId: string,
  topicTitle: string,
  topicContent: string,
  totalCount: number,
  userId: string,
  aiSettings: AISettings | undefined,
  aiCostTracking: any
): Promise<QuestionMixResult> {
  console.log('[Question Mixing] Starting mixQuestionsWithAI');
  console.log('[Question Mixing] Topic:', topicTitle);
  console.log('[Question Mixing] Total count requested:', totalCount);
  console.log('[Question Mixing] User ID:', userId);
  console.log('[Question Mixing] AI Settings:', aiSettings);

  // Default error result if AI is not configured
  const errorResult: QuestionMixResult = {
    questions: [],
    metadata: {
      staticCount: 0,
      aiCount: 0,
      newlyGenerated: 0,
      costEstimate: 0,
      error: 'AI question generation is not configured. Please add an API key in Settings.',
    },
  };

  // Return error if AI provider is not configured
  if (!aiSettings || !aiSettings.provider) {
    console.log('[Question Mixing] AI provider not configured, cannot generate questions');
    return errorResult;
  }

  console.log('[Question Mixing] AI is enabled, proceeding with generation');

  try {
    // Load API key
    console.log('[Question Mixing] Loading API key for user:', userId);
    const { apiKey } = await loadAPIKey(userId);
    console.log('[Question Mixing] API key loaded:', apiKey ? `Yes (length: ${apiKey.length})` : 'No');

    if (!apiKey) {
      console.log('[Question Mixing] No API key found');
      return {
        questions: [],
        metadata: {
          staticCount: 0,
          aiCount: 0,
          newlyGenerated: 0,
          costEstimate: 0,
          error: 'API key not found. Please configure AI settings.',
        },
      };
    }

    // When AI is enabled, use 100% AI questions
    const aiQuestionCount = totalCount;
    console.log('[Question Mixing] Generating', aiQuestionCount, 'AI questions');

    // Initialize cost tracker (no limits, just tracking)
    const costTracker = new CostTracker(aiSettings.provider, aiCostTracking);

    // Always generate fresh questions (no caching)
    const aiQuestions: QuizQuestion[] = [];

    // Summarize topic content
    const mockTopic: Topic = {
      id: topicId,
      title: topicTitle,
      description: '',
      content: topicContent,
      category: '',
    };
    const topicSummary = getCondensedTopicContent(mockTopic);
    console.log('[Question Mixing] Topic summary length:', topicSummary.length);

    // Create AI service
    const aiService = getAIService();
    console.log('[Question Mixing] AI service created:', aiService.constructor.name);

    // Generate questions
    const prompt: AIQuestionPrompt = {
      topicId: topicId,
      topicTitle: topicTitle,
      topicContent: topicSummary,
      difficulty: aiSettings.preferences.questionDifficulty,
      questionType: 'multiple-choice',
    };

    console.log('[Question Mixing] Calling generateQuestionBatch...');
    try {
      const generatedQuestions = await aiService.generateQuestionBatch(
        prompt,
        aiQuestionCount,
        apiKey
      );

      console.log('[Question Mixing] Successfully generated', generatedQuestions.length, 'questions');

      // Convert questions
      for (const aiQ of generatedQuestions) {
        const quizQuestion = convertAIQuestionToQuizQuestion(aiQ);
        aiQuestions.push(quizQuestion);
      }

      // Record cost
      costTracker.recordQuestionGeneration(generatedQuestions.length);
    } catch (error) {
      console.error('[Question Mixing] Error generating AI questions:', error);
      // On error, return empty with error message
      return {
        questions: [],
        metadata: {
          staticCount: 0,
          aiCount: 0,
          newlyGenerated: 0,
          costEstimate: 0,
          error: error instanceof Error ? error.message : 'Failed to generate AI questions',
        },
      };
    }

    // Shuffle AI questions for variety
    for (let i = aiQuestions.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [aiQuestions[i], aiQuestions[j]] = [aiQuestions[j], aiQuestions[i]];
    }

    return {
      questions: aiQuestions,
      metadata: {
        staticCount: 0,
        aiCount: aiQuestions.length,
        newlyGenerated: aiQuestions.length,
        costEstimate: aiQuestions.length * costTracker.tracking.estimatedCostPerQuestion,
      },
    };
  } catch (error) {
    console.error('[Question Mixing] Unexpected error:', error);
    return {
      questions: [],
      metadata: {
        staticCount: 0,
        aiCount: 0,
        newlyGenerated: 0,
        costEstimate: 0,
        error: error instanceof Error ? error.message : 'Unexpected error during question generation',
      },
    };
  }
}

/**
 * Shuffle an array in place (Fisher-Yates algorithm)
 */
function shuffleArray<T>(array: T[]): T[] {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

/**
 * Get a message describing the question generation result
 */
export function getQuestionMixMessage(metadata: QuestionMixResult['metadata']): string {
  if (metadata.error) {
    return `⚠️ ${metadata.error}`;
  }

  if (metadata.aiCount === 0) {
    return `Using ${metadata.staticCount} static questions`;
  }

  let message = `${metadata.aiCount} fresh AI-generated questions`;

  if (metadata.costEstimate > 0) {
    message += ` · Cost: $${metadata.costEstimate.toFixed(3)}`;
  }

  return message;
}
