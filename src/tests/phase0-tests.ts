/**
 * Phase 0 Component Tests
 * Tests all foundational AI components
 */

import { AIServiceError } from '../types/ai';
import { handleAIError, shouldFallbackToStatic, getErrorActionMessage } from '../utils/aiErrorHandler';
import { MockAIService } from '../services/mockAIService';
import { getAIService } from '../services/aiService';
import { CostTracker, RateLimiter } from '../utils/aiCostTracking';
import {
  getCondensedTopicContent,
  calculateTokenSavings,
  hashContent,
  estimateTokenCount,
} from '../utils/topicSummarization';
import { Topic } from '../types';

// Test utilities
const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// Test results tracking
interface TestResult {
  name: string;
  passed: boolean;
  error?: string;
  duration?: number;
}

const results: TestResult[] = [];

function test(name: string, fn: () => void | Promise<void>): void {
  const start = Date.now();

  Promise.resolve(fn())
    .then(() => {
      results.push({
        name,
        passed: true,
        duration: Date.now() - start,
      });
      console.log(`✓ ${name}`);
    })
    .catch((error) => {
      results.push({
        name,
        passed: false,
        error: error.message,
        duration: Date.now() - start,
      });
      console.error(`✗ ${name}: ${error.message}`);
    });
}

function assert(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(message);
  }
}

// ============================================================================
// 1. ERROR HANDLING TESTS
// ============================================================================

console.log('\n=== Testing Error Handling ===\n');

test('AIServiceError creation', () => {
  const error = new AIServiceError('Test error', 'INVALID_API_KEY', false);
  assert(error.message === 'Test error', 'Error message should match');
  assert(error.code === 'INVALID_API_KEY', 'Error code should match');
  assert(error.recoverable === false, 'Recoverable flag should match');
  assert(error.name === 'AIServiceError', 'Error name should be AIServiceError');
});

test('handleAIError - 401 Unauthorized', () => {
  const mockError = { status: 401 };
  const aiError = handleAIError(mockError);

  assert(aiError.code === 'INVALID_API_KEY', 'Should detect invalid API key');
  assert(aiError.recoverable === false, 'Invalid API key is not recoverable');
  assert(aiError.message.includes('Invalid API key'), 'Should have descriptive message');
});

test('handleAIError - 429 Rate Limit', () => {
  const mockError = { status: 429, headers: { 'retry-after': '30' } };
  const aiError = handleAIError(mockError);

  assert(aiError.code === 'RATE_LIMIT', 'Should detect rate limit');
  assert(aiError.recoverable === true, 'Rate limit is recoverable');
  assert(aiError.retryAfter === 30, 'Should extract retry-after header');
});

test('handleAIError - Network Error', () => {
  const mockError = { message: 'network error' };
  const aiError = handleAIError(mockError);

  assert(aiError.code === 'NETWORK_ERROR', 'Should detect network error');
  assert(aiError.recoverable === true, 'Network errors are recoverable');
});

test('shouldFallbackToStatic - Timeout', () => {
  const error = new AIServiceError('Timeout', 'TIMEOUT', true);
  assert(shouldFallbackToStatic(error) === true, 'Should fallback on timeout');
});

test('shouldFallbackToStatic - Invalid API Key', () => {
  const error = new AIServiceError('Invalid', 'INVALID_API_KEY', false);
  assert(shouldFallbackToStatic(error) === false, 'Should not fallback on invalid key');
});

test('getErrorActionMessage - All codes', () => {
  const codes: Array<any> = [
    'INVALID_API_KEY',
    'RATE_LIMIT',
    'TOKEN_LIMIT',
    'TIMEOUT',
    'SERVER_ERROR',
    'NETWORK_ERROR',
    'CONTENT_POLICY',
    'UNKNOWN_ERROR',
  ];

  codes.forEach(code => {
    const error = new AIServiceError('Test', code, true);
    const message = getErrorActionMessage(error);
    assert(message.length > 0, `Should have action message for ${code}`);
  });
});

// ============================================================================
// 2. MOCK AI SERVICE TESTS
// ============================================================================

console.log('\n=== Testing Mock AI Service ===\n');

test('MockAIService - Generate multiple choice question', async () => {
  const service = new MockAIService();

  const question = await service.generateQuestion({
    topicId: 'linear-regression',
    topicTitle: 'Linear Regression',
    topicContent: 'Test content',
    difficulty: 'intermediate',
    questionType: 'multiple-choice',
  }, 'test-key');

  assert(question.type === 'multiple-choice', 'Should be multiple choice');
  assert(question.question.length > 0, 'Should have question text');
  assert(question.options.length === 4, 'Should have 4 options');
  assert(question.correctAnswer >= 0 && question.correctAnswer <= 3, 'Should have valid answer');
  assert(question.source === 'ai-generated', 'Should be marked as AI generated');
});

test('MockAIService - Generate free-form question', async () => {
  const service = new MockAIService();

  const question = await service.generateQuestion({
    topicId: 'linear-regression',
    topicTitle: 'Linear Regression',
    topicContent: 'Test content',
    difficulty: 'intermediate',
    questionType: 'free-form',
  }, 'test-key');

  assert(question.type === 'free-form', 'Should be free-form');
  assert(question.question.length > 0, 'Should have question text');
  assert('sampleAnswer' in question, 'Should have sample answer');
});

test('MockAIService - Evaluate answer (good)', async () => {
  const service = new MockAIService();

  const freeFormQuestion = await service.generateQuestion({
    topicId: 'linear-regression',
    topicTitle: 'Linear Regression',
    topicContent: 'Test content',
    difficulty: 'intermediate',
    questionType: 'free-form',
  }, 'test-key');

  if (freeFormQuestion.type !== 'free-form') {
    throw new Error('Expected free-form question');
  }

  // Use keywords from sample answer for high score
  const userAnswer = freeFormQuestion.sampleAnswer || 'overfitting regression training validation';

  const evaluation = await service.evaluateAnswer(
    freeFormQuestion,
    userAnswer,
    'test-key'
  );

  assert(evaluation.score >= 0 && evaluation.score <= 100, 'Score should be 0-100');
  assert(evaluation.feedback.length > 0, 'Should have feedback');
  assert(Array.isArray(evaluation.strengths), 'Should have strengths array');
  assert(Array.isArray(evaluation.improvements), 'Should have improvements array');
  assert(evaluation.evaluatedAt instanceof Date, 'Should have evaluation date');
});

test('MockAIService - Evaluate answer (poor)', async () => {
  const service = new MockAIService();

  const freeFormQuestion = await service.generateQuestion({
    topicId: 'linear-regression',
    topicTitle: 'Linear Regression',
    topicContent: 'Test content',
    difficulty: 'intermediate',
    questionType: 'free-form',
  }, 'test-key');

  if (freeFormQuestion.type !== 'free-form') {
    throw new Error('Expected free-form question');
  }

  const evaluation = await service.evaluateAnswer(
    freeFormQuestion,
    'I dont know',
    'test-key'
  );

  assert(evaluation.score < 50, 'Poor answer should score low');
  assert(evaluation.improvements.length > 0, 'Should have improvement suggestions');
});

test('MockAIService - Batch generation', async () => {
  const service = new MockAIService();

  const questions = await service.generateQuestionBatch({
    topicId: 'linear-regression',
    topicTitle: 'Linear Regression',
    topicContent: 'Test content',
    difficulty: 'intermediate',
    questionType: 'multiple-choice',
  }, 3, 'test-key');

  assert(questions.length === 3, 'Should generate requested number');
  assert(questions.every(q => q.type === 'multiple-choice'), 'All should be multiple choice');
});

test('MockAIService - Fallback for unknown topic', async () => {
  const service = new MockAIService();

  const question = await service.generateQuestion({
    topicId: 'unknown-topic-xyz',
    topicTitle: 'Unknown Topic',
    topicContent: 'Test content',
    difficulty: 'intermediate',
    questionType: 'multiple-choice',
  }, 'test-key');

  assert(question.type === 'multiple-choice', 'Should generate fallback question');
  assert(question.question.includes('Unknown Topic'), 'Should reference topic title');
});

test('MockAIService - Add custom question', () => {
  const service = new MockAIService();

  service.addMockQuestion('custom-topic', {
    id: 'custom-1',
    type: 'multiple-choice',
    question: 'Custom question?',
    options: ['A', 'B', 'C', 'D'],
    correctAnswer: 0,
    explanation: 'Custom explanation',
    source: 'ai-generated',
    generatedAt: new Date(),
  });

  // No error means success
  assert(true, 'Should add custom question without error');
});

test('getAIService - Returns singleton', () => {
  const service1 = getAIService();
  const service2 = getAIService();

  assert(service1 === service2, 'Should return same instance');
});

// ============================================================================
// 3. COST TRACKING TESTS
// ============================================================================

console.log('\n=== Testing Cost Tracking ===\n');

test('CostTracker - Initialization', () => {
  const tracker = new CostTracker('claude');
  const tracking = tracker.getTracking();

  assert(tracking.dailyLimit === 1.0, 'Default daily limit should be $1');
  assert(tracking.monthlyLimit === 10.0, 'Default monthly limit should be $10');
  assert(tracking.dailySpend === 0, 'Initial spend should be 0');
  assert(tracking.estimatedCostPerQuestion === 0.003, 'Claude cost should be $0.003');
});

test('CostTracker - Record question generation', () => {
  const tracker = new CostTracker('claude');

  tracker.recordQuestionGeneration(10);
  const tracking = tracker.getTracking();

  assert(tracking.questionsGeneratedToday === 10, 'Should track question count');
  assert(tracking.dailySpend === 0.03, 'Should calculate cost (10 * $0.003)');
  assert(tracking.monthlySpend === 0.03, 'Should update monthly spend');
});

test('CostTracker - Can make request check', () => {
  const tracker = new CostTracker('claude');

  assert(tracker.canMakeRequest(10) === true, 'Should allow request under limit');

  tracker.recordQuestionGeneration(300); // $0.90
  assert(tracker.canMakeRequest(10) === true, 'Should allow request approaching limit');

  tracker.recordQuestionGeneration(40); // Total $1.02 > $1.00 limit
  assert(tracker.canMakeRequest(1) === false, 'Should block request over limit');
});

test('CostTracker - Estimate cost', () => {
  const tracker = new CostTracker('claude');

  const cost = tracker.estimateCost(10);
  assert(cost === 0.03, 'Should estimate $0.03 for 10 questions');
});

test('CostTracker - Remaining budget', () => {
  const tracker = new CostTracker('claude');

  tracker.recordQuestionGeneration(100); // $0.30
  const remaining = tracker.getRemainingDailyBudget();

  assert(remaining === 0.70, 'Should calculate remaining budget');
});

test('CostTracker - Usage percentage', () => {
  const tracker = new CostTracker('claude');

  tracker.recordQuestionGeneration(100); // $0.30 = 30%
  const percentage = tracker.getDailyUsagePercentage();

  assert(percentage === 30, 'Should calculate usage percentage');
});

test('CostTracker - Approaching limit detection', () => {
  const tracker = new CostTracker('claude');

  assert(tracker.isApproachingDailyLimit() === false, 'Should not be approaching initially');

  tracker.recordQuestionGeneration(270); // $0.81 = 81%
  assert(tracker.isApproachingDailyLimit() === true, 'Should detect approaching limit');
});

test('CostTracker - Remaining questions', () => {
  const tracker = new CostTracker('claude');

  const remaining = tracker.getRemainingQuestionsToday();
  assert(remaining === 333, 'Should calculate remaining questions ($1 / $0.003)');

  tracker.recordQuestionGeneration(100);
  const remainingAfter = tracker.getRemainingQuestionsToday();
  assert(remainingAfter === 233, 'Should update remaining questions');
});

test('CostTracker - Set limits', () => {
  const tracker = new CostTracker('claude');

  tracker.setDailyLimit(2.0);
  tracker.setMonthlyLimit(20.0);

  const tracking = tracker.getTracking();
  assert(tracking.dailyLimit === 2.0, 'Should update daily limit');
  assert(tracking.monthlyLimit === 20.0, 'Should update monthly limit');
});

test('CostTracker - Record evaluation', () => {
  const tracker = new CostTracker('claude');

  tracker.recordEvaluation(5);
  const tracking = tracker.getTracking();

  assert(tracking.evaluationsToday === 5, 'Should track evaluation count');
  // Evaluations cost 1.5x question cost
  assert(tracking.dailySpend === 0.0225, 'Should calculate evaluation cost');
});

test('CostTracker - Summary', () => {
  const tracker = new CostTracker('claude');

  tracker.recordQuestionGeneration(10);
  const summary = tracker.getSummary();

  assert(summary.includes('$0.03'), 'Summary should include spend');
  assert(summary.includes('10'), 'Summary should include question count');
});

// ============================================================================
// 4. RATE LIMITING TESTS
// ============================================================================

console.log('\n=== Testing Rate Limiting ===\n');

test('RateLimiter - Initialization', () => {
  const limiter = new RateLimiter('claude');

  assert(limiter.canMakeRequest() === true, 'Should allow initial request');
});

test('RateLimiter - Record request', () => {
  const limiter = new RateLimiter('claude');

  limiter.recordRequest();
  assert(limiter.canMakeRequest() === true, 'Should still allow request after 1');
});

test('RateLimiter - Requests remaining', () => {
  const limiter = new RateLimiter('claude');

  limiter.recordRequest();
  limiter.recordRequest();

  const remaining = limiter.getRequestsRemainingThisMinute();
  assert(remaining === 48, 'Should have 48 remaining (50 - 2)');
});

test('RateLimiter - Approaching limit detection', () => {
  const limiter = new RateLimiter('claude');

  // Record 41 requests (82% of 50)
  for (let i = 0; i < 41; i++) {
    limiter.recordRequest();
  }

  assert(limiter.isApproachingRateLimit() === true, 'Should detect approaching limit');
});

test('RateLimiter - Different providers', () => {
  const claudeLimiter = new RateLimiter('claude');
  const openaiFreeLimiter = new RateLimiter('openai', 'free');

  assert(claudeLimiter.getRequestsRemainingThisMinute() === 50, 'Claude should have 50/min');
  assert(openaiFreeLimiter.getRequestsRemainingThisMinute() === 3, 'OpenAI free should have 3/min');
});

// ============================================================================
// 5. TOPIC SUMMARIZATION TESTS
// ============================================================================

console.log('\n=== Testing Topic Summarization ===\n');

test('estimateTokenCount', () => {
  const text = 'This is a test string';
  const tokens = estimateTokenCount(text);

  assert(tokens > 0, 'Should estimate tokens');
  assert(tokens === Math.ceil(text.length / 4), 'Should use 4 chars per token');
});

test('hashContent - Consistent', () => {
  const content = 'Test content for hashing';
  const hash1 = hashContent(content);
  const hash2 = hashContent(content);

  assert(hash1 === hash2, 'Same content should produce same hash');
});

test('hashContent - Different', () => {
  const hash1 = hashContent('Content A');
  const hash2 = hashContent('Content B');

  assert(hash1 !== hash2, 'Different content should produce different hash');
});

test('getCondensedTopicContent - Basic', () => {
  const topic: Topic = {
    id: 'test-topic',
    title: 'Test Topic',
    category: 'test',
    description: 'A test topic',
    content: `
      <h2>Introduction</h2>
      <p>This is a test topic with <strong>key concepts</strong>.</p>
      <ul>
        <li>Bullet point 1</li>
        <li>Bullet point 2</li>
      </ul>
      <p><strong>Definition:</strong> A term defined here.</p>
      <p>Formula: $y = mx + b$</p>
    `,
  };

  const condensed = getCondensedTopicContent(topic);

  assert(condensed.includes('Test Topic'), 'Should include topic title');
  assert(condensed.includes('Key Concepts'), 'Should have sections');
  assert(condensed.length < topic.content.length, 'Should be shorter than original');
});

test('calculateTokenSavings', () => {
  const original = 'A'.repeat(2000); // ~500 tokens
  const condensed = 'A'.repeat(400); // ~100 tokens

  const savings = calculateTokenSavings(original, condensed);

  assert(savings.original === 500, 'Should calculate original tokens');
  assert(savings.condensed === 100, 'Should calculate condensed tokens');
  assert(savings.savings === 400, 'Should calculate savings');
  assert(savings.savingsPercent === 80, 'Should calculate percentage');
});

test('getCondensedTopicContent - Token reduction', () => {
  const topic: Topic = {
    id: 'large-topic',
    title: 'Large Topic',
    category: 'test',
    description: 'A large topic',
    content: '<p>' + 'Lorem ipsum dolor sit amet. '.repeat(500) + '</p>', // Large content
  };

  const originalTokens = estimateTokenCount(topic.content);
  const condensed = getCondensedTopicContent(topic);
  const condensedTokens = estimateTokenCount(condensed);

  assert(condensedTokens < originalTokens, 'Condensed should have fewer tokens');
  assert(condensedTokens < 1000, 'Condensed should be under 1000 tokens');
});

// ============================================================================
// SUMMARY
// ============================================================================

export function runTests(): void {
  console.log('\n=== Test Summary ===\n');

  // Wait for all tests to complete
  setTimeout(() => {
    const passed = results.filter(r => r.passed).length;
    const failed = results.filter(r => !r.passed).length;
    const total = results.length;

    console.log(`Total: ${total}`);
    console.log(`Passed: ${passed} ✓`);
    console.log(`Failed: ${failed} ✗`);
    console.log(`Success Rate: ${((passed / total) * 100).toFixed(1)}%`);

    if (failed > 0) {
      console.log('\n=== Failed Tests ===\n');
      results.filter(r => !r.passed).forEach(r => {
        console.log(`✗ ${r.name}`);
        console.log(`  Error: ${r.error}`);
      });
    }

    const totalDuration = results.reduce((sum, r) => sum + (r.duration || 0), 0);
    console.log(`\nTotal Duration: ${totalDuration}ms`);

    console.log('\n=== Phase 0 Component Tests Complete ===\n');
  }, 15000); // Wait 15 seconds for async tests
}
