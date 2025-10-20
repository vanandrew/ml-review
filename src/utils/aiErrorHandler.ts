import { AIServiceError, AIErrorCode } from '../types/ai';

/**
 * Handles errors from AI API calls and converts them to user-friendly AIServiceErrors
 */
export const handleAIError = (error: any): AIServiceError => {
  // 1. Invalid API key
  if (error.status === 401 || error.status === 403) {
    return new AIServiceError(
      'Invalid API key. Please check your settings.',
      'INVALID_API_KEY',
      false
    );
  }

  // 2. Rate limit
  if (error.status === 429) {
    const retryAfter = parseInt(error.headers?.['retry-after']) || 60;
    return new AIServiceError(
      `Rate limit reached. Try again in ${retryAfter} seconds.`,
      'RATE_LIMIT',
      true,
      retryAfter
    );
  }

  // 3. Token limit exceeded
  if (error.message?.includes('token limit') || error.status === 413) {
    return new AIServiceError(
      'Content too large. Try a shorter topic or answer.',
      'TOKEN_LIMIT',
      false
    );
  }

  // 4. Malformed response
  if (error instanceof SyntaxError || error.message?.includes('JSON')) {
    return new AIServiceError(
      'AI returned invalid response. Please try again.',
      'INVALID_RESPONSE',
      true
    );
  }

  // 5. Network timeout
  if (error.name === 'AbortError' || error.message?.includes('timeout')) {
    return new AIServiceError(
      'Request timed out. Check your connection.',
      'TIMEOUT',
      true
    );
  }

  // 6. Server error
  if (error.status >= 500) {
    return new AIServiceError(
      'AI service is temporarily unavailable.',
      'SERVER_ERROR',
      true,
      30
    );
  }

  // 7. Content policy violation
  if (error.message?.includes('content policy') || error.status === 451) {
    return new AIServiceError(
      'Content violates AI provider policy.',
      'CONTENT_POLICY',
      false
    );
  }

  // 8. Network error
  if (!navigator.onLine || error.message?.includes('network')) {
    return new AIServiceError(
      'No internet connection.',
      'NETWORK_ERROR',
      true
    );
  }

  // 9. Unknown error
  return new AIServiceError(
    'An unexpected error occurred.',
    'UNKNOWN_ERROR',
    true
  );
};

/**
 * Logs AI errors for analytics
 */
export const logAIError = (error: AIServiceError): void => {
  console.error('[AI Error]', {
    code: error.code,
    message: error.message,
    recoverable: error.recoverable,
    retryAfter: error.retryAfter,
    timestamp: new Date().toISOString(),
  });

  // TODO: Send to analytics service when implemented
};

/**
 * Determines if an error should trigger a fallback to static questions
 */
export const shouldFallbackToStatic = (error: AIServiceError): boolean => {
  const fallbackCodes: AIErrorCode[] = [
    'TIMEOUT',
    'SERVER_ERROR',
    'NETWORK_ERROR',
    'RATE_LIMIT',
    'UNKNOWN_ERROR',
  ];

  return fallbackCodes.includes(error.code);
};

/**
 * Gets a user-friendly error message with action steps
 */
export const getErrorActionMessage = (error: AIServiceError): string => {
  switch (error.code) {
    case 'INVALID_API_KEY':
      return 'Please check your API key in Settings.';
    case 'RATE_LIMIT':
      return `Try again in ${error.retryAfter || 60} seconds, or use static questions.`;
    case 'TOKEN_LIMIT':
      return 'The content is too large. Please try a different topic.';
    case 'TIMEOUT':
      return 'The request took too long. Check your internet connection.';
    case 'SERVER_ERROR':
      return 'The AI service is temporarily down. Using static questions.';
    case 'NETWORK_ERROR':
      return 'Check your internet connection and try again.';
    case 'CONTENT_POLICY':
      return 'The content violates the AI provider\'s policy.';
    default:
      return 'Please try again or use static questions.';
  }
};
