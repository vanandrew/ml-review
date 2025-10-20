// AI Question Generation Types

// Base question interface
export interface BaseQuestion {
  id: string;
  question: string;
  source: 'static' | 'ai-generated';
  generatedAt?: Date;
  topicId?: string;
}

// Multiple choice question (existing + extended)
export interface MultipleChoiceQuestion extends BaseQuestion {
  type: 'multiple-choice';
  options: string[];
  correctAnswer: number;
  explanation: string;
}

// Free-form question (new)
export interface FreeFormQuestion extends BaseQuestion {
  type: 'free-form';
  sampleAnswer?: string;
  evaluationCriteria?: string[];
  rubric?: {
    excellent: string;
    good: string;
    needs_improvement: string;
  };
}

// Union type for all questions
export type AIQuizQuestion = MultipleChoiceQuestion | FreeFormQuestion;

// AI Evaluation Result
export interface AIEvaluationResult {
  score: number; // 0-100 score
  feedback: string;
  strengths: string[];
  improvements: string[];
  evaluatedAt: Date;
}

// Free-form quiz score with evaluation
export interface FreeFormQuizScore {
  userAnswer: string;
  evaluation: AIEvaluationResult;
}

// AI Settings
export interface AISettings {
  provider: 'claude' | 'openai' | 'gemini' | null;
  apiKey: string | null;
  enabled: boolean;
  preferences: {
    questionDifficulty: 'beginner' | 'intermediate' | 'advanced';
    questionTypes: ('multiple-choice' | 'free-form')[];
    questionsPerQuiz: number;
  };
}

// AI Question Prompt
export interface AIQuestionPrompt {
  topicId: string;
  topicTitle: string;
  topicContent: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  questionType: 'multiple-choice' | 'free-form';
  existingQuestions?: string[];
}

// Cached AI Question
export interface CachedAIQuestion {
  question: AIQuizQuestion;
  topicId: string;
  usageCount: number;
  createdAt: Date;
  lastUsedAt: Date;

  // Cache metadata
  cachedAt: Date;
  expiresAt: Date;

  // Invalidation triggers
  topicContentHash: string;
  modelVersion: string;
  providerVersion: string;
  promptVersion: string;

  // Quality tracking
  averageScore: number;
  reportCount: number;
  reportReasons: string[];
  validationScore: number; // 0-100, community votes
  validationVotes: number;

  // Status
  status: 'active' | 'flagged' | 'hidden' | 'validated';
  flaggedAt?: Date;
  validatedAt?: Date;
}

// Question Report
export interface QuestionReport {
  questionId: string;
  userId: string;
  reason: 'incorrect-answer' | 'ambiguous' | 'poor-quality' | 'off-topic' | 'other';
  description: string;
  timestamp: Date;
}

// Cost Tracking
export interface CostTracking {
  dailySpend: number;
  monthlySpend: number;
  dailyLimit: number;
  monthlyLimit: number;
  questionsGeneratedToday: number;
  evaluationsToday: number;
  lastResetDate: string;
  estimatedCostPerQuestion: number;
}

// Rate Limit Config
export interface RateLimitConfig {
  provider: 'claude' | 'openai' | 'gemini';
  requestsPerMinute: number;
  requestsPerHour: number;
  requestsPerDay: number;
}

// AI Service Error Types
export type AIErrorCode =
  | 'INVALID_API_KEY'
  | 'RATE_LIMIT'
  | 'TOKEN_LIMIT'
  | 'INVALID_RESPONSE'
  | 'TIMEOUT'
  | 'SERVER_ERROR'
  | 'CONTENT_POLICY'
  | 'NETWORK_ERROR'
  | 'UNKNOWN_ERROR';

// AI Service Error
export class AIServiceError extends Error {
  constructor(
    message: string,
    public code: AIErrorCode,
    public recoverable: boolean,
    public retryAfter?: number
  ) {
    super(message);
    this.name = 'AIServiceError';
    Object.setPrototypeOf(this, AIServiceError.prototype);
  }
}

// Answer Validation
export interface AnswerValidation {
  valid: boolean;
  error?: string;
  warnings?: string[];
}

// Evaluation Config
export interface EvaluationConfig {
  temperature: number;
  cacheEnabled: boolean;
  cacheDuration: number;
  multipleEvaluations: boolean;
  evaluationCount: number;
}

// Topic Context (condensed for AI)
export interface TopicContext {
  title: string;
  keyPoints: string[];
  definitions: string[];
  formulas: string[];
}

// Pre-generation Queue
export interface PreGenerationQueue {
  topicId: string;
  targetCount: number;
  currentCount: number;
  lastGenerated: Date;
}

// AI Question Metadata
export interface AIQuestionMetadata {
  id: string;
  question: AIQuizQuestion;
  source: 'ai-generated';
  provider: string;
  model: string;
  generatedAt: Date;

  // Quality tracking
  timesUsed: number;
  reportCount: number;
  reportReasons: string[];
  validationScore: number;
  validationVotes: number;

  // Status
  status: 'active' | 'flagged' | 'hidden' | 'validated';
  flaggedAt?: Date;
  validatedAt?: Date;
}
