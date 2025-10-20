import { CostTracking, RateLimitConfig } from '../types/ai';

/**
 * Rate limit configurations for different AI providers
 */
export const RATE_LIMITS: Record<string, RateLimitConfig> = {
  'claude': {
    provider: 'claude',
    requestsPerMinute: 50,
    requestsPerHour: 1000,
    requestsPerDay: 10000,
  },
  'openai-free': {
    provider: 'openai',
    requestsPerMinute: 3,
    requestsPerHour: 200,
    requestsPerDay: 500,
  },
  'openai-paid': {
    provider: 'openai',
    requestsPerMinute: 60,
    requestsPerHour: 3500,
    requestsPerDay: 10000,
  },
  'gemini-free': {
    provider: 'gemini',
    requestsPerMinute: 15,
    requestsPerHour: 1500,
    requestsPerDay: 1500,
  },
};

/**
 * Estimated costs per question for different providers
 */
export const COST_PER_QUESTION: Record<string, number> = {
  'claude': 0.003,
  'openai': 0.01,
  'gemini': 0.002,
};

/**
 * Default cost tracking data
 */
export const getDefaultCostTracking = (): CostTracking => {
  return {
    dailySpend: 0,
    monthlySpend: 0,
    dailyLimit: 1.0, // $1.00 default
    monthlyLimit: 10.0, // $10.00 default
    questionsGeneratedToday: 0,
    evaluationsToday: 0,
    lastResetDate: new Date().toISOString().split('T')[0],
    estimatedCostPerQuestion: 0.003, // Default to Claude pricing
  };
};

/**
 * Cost Tracking Manager
 */
export class CostTracker {
  private tracking: CostTracking;
  private provider: string;

  constructor(provider: string, tracking?: CostTracking) {
    this.provider = provider;
    this.tracking = tracking || getDefaultCostTracking();
    this.tracking.estimatedCostPerQuestion = COST_PER_QUESTION[provider] || 0.003;
    this.checkAndResetIfNeeded();
  }

  /**
   * Check if daily reset is needed
   */
  private checkAndResetIfNeeded(): void {
    const today = new Date().toISOString().split('T')[0];
    if (this.tracking.lastResetDate !== today) {
      this.resetDaily();
    }

    // Check if monthly reset is needed
    const lastResetDate = new Date(this.tracking.lastResetDate);
    const now = new Date();
    if (lastResetDate.getMonth() !== now.getMonth() ||
        lastResetDate.getFullYear() !== now.getFullYear()) {
      this.resetMonthly();
    }
  }

  /**
   * Reset daily counters
   */
  private resetDaily(): void {
    this.tracking.dailySpend = 0;
    this.tracking.questionsGeneratedToday = 0;
    this.tracking.evaluationsToday = 0;
    this.tracking.lastResetDate = new Date().toISOString().split('T')[0];
  }

  /**
   * Reset monthly counters
   */
  private resetMonthly(): void {
    this.tracking.monthlySpend = 0;
  }

  /**
   * Check if user can make a request based on daily limit
   */
  canMakeRequest(count: number = 1): boolean {
    this.checkAndResetIfNeeded();

    const estimatedCost = count * this.tracking.estimatedCostPerQuestion;
    const newDailyTotal = this.tracking.dailySpend + estimatedCost;
    const newMonthlyTotal = this.tracking.monthlySpend + estimatedCost;

    return newDailyTotal <= this.tracking.dailyLimit &&
           newMonthlyTotal <= this.tracking.monthlyLimit;
  }

  /**
   * Record a question generation
   */
  recordQuestionGeneration(count: number = 1): void {
    this.checkAndResetIfNeeded();

    const cost = count * this.tracking.estimatedCostPerQuestion;
    this.tracking.dailySpend += cost;
    this.tracking.monthlySpend += cost;
    this.tracking.questionsGeneratedToday += count;
  }

  /**
   * Record an answer evaluation
   */
  recordEvaluation(count: number = 1): void {
    this.checkAndResetIfNeeded();

    // Evaluations typically cost more than generation
    const evaluationCost = this.tracking.estimatedCostPerQuestion * 1.5;
    const cost = count * evaluationCost;
    this.tracking.dailySpend += cost;
    this.tracking.monthlySpend += cost;
    this.tracking.evaluationsToday += count;
  }

  /**
   * Get estimated cost for N questions
   */
  estimateCost(count: number): number {
    return count * this.tracking.estimatedCostPerQuestion;
  }

  /**
   * Get remaining budget for today
   */
  getRemainingDailyBudget(): number {
    this.checkAndResetIfNeeded();
    return Math.max(0, this.tracking.dailyLimit - this.tracking.dailySpend);
  }

  /**
   * Get remaining budget for this month
   */
  getRemainingMonthlyBudget(): number {
    this.checkAndResetIfNeeded();
    return Math.max(0, this.tracking.monthlyLimit - this.tracking.monthlySpend);
  }

  /**
   * Get percentage of daily limit used
   */
  getDailyUsagePercentage(): number {
    this.checkAndResetIfNeeded();
    return (this.tracking.dailySpend / this.tracking.dailyLimit) * 100;
  }

  /**
   * Get percentage of monthly limit used
   */
  getMonthlyUsagePercentage(): number {
    this.checkAndResetIfNeeded();
    return (this.tracking.monthlySpend / this.tracking.monthlyLimit) * 100;
  }

  /**
   * Check if approaching daily limit (>80%)
   */
  isApproachingDailyLimit(): boolean {
    return this.getDailyUsagePercentage() >= 80;
  }

  /**
   * Check if approaching monthly limit (>80%)
   */
  isApproachingMonthlyLimit(): boolean {
    return this.getMonthlyUsagePercentage() >= 80;
  }

  /**
   * Get remaining questions within daily limit
   */
  getRemainingQuestionsToday(): number {
    const remainingBudget = this.getRemainingDailyBudget();
    return Math.floor(remainingBudget / this.tracking.estimatedCostPerQuestion);
  }

  /**
   * Update daily limit
   */
  setDailyLimit(limit: number): void {
    this.tracking.dailyLimit = limit;
  }

  /**
   * Update monthly limit
   */
  setMonthlyLimit(limit: number): void {
    this.tracking.monthlyLimit = limit;
  }

  /**
   * Get current tracking data
   */
  getTracking(): CostTracking {
    this.checkAndResetIfNeeded();
    return { ...this.tracking };
  }

  /**
   * Get formatted cost summary
   */
  getSummary(): string {
    this.checkAndResetIfNeeded();

    return `
Today: $${this.tracking.dailySpend.toFixed(2)} / $${this.tracking.dailyLimit.toFixed(2)} (${this.getDailyUsagePercentage().toFixed(0)}%)
This Month: $${this.tracking.monthlySpend.toFixed(2)} / $${this.tracking.monthlyLimit.toFixed(2)} (${this.getMonthlyUsagePercentage().toFixed(0)}%)
Questions Generated Today: ${this.tracking.questionsGeneratedToday}
Evaluations Today: ${this.tracking.evaluationsToday}
Remaining Questions: ~${this.getRemainingQuestionsToday()}
    `.trim();
  }
}

/**
 * Rate Limiter for API requests
 */
export class RateLimiter {
  private requests: Date[] = [];
  private config: RateLimitConfig;

  constructor(provider: string, tier: 'free' | 'paid' = 'paid') {
    const configKey = provider === 'openai' ? `${provider}-${tier}` : provider;
    this.config = RATE_LIMITS[configKey] || RATE_LIMITS['claude'];
  }

  /**
   * Check if a request can be made
   */
  canMakeRequest(): boolean {
    const now = new Date();
    const oneMinuteAgo = new Date(now.getTime() - 60000);
    const oneHourAgo = new Date(now.getTime() - 3600000);

    // Clean old requests
    this.requests = this.requests.filter(r => r > oneHourAgo);

    // Check limits
    const lastMinute = this.requests.filter(r => r > oneMinuteAgo).length;
    const lastHour = this.requests.length;

    return lastMinute < this.config.requestsPerMinute &&
           lastHour < this.config.requestsPerHour;
  }

  /**
   * Record a request
   */
  recordRequest(): void {
    this.requests.push(new Date());
  }

  /**
   * Get time until next request is available (in milliseconds)
   */
  getTimeUntilAvailable(): number {
    if (this.canMakeRequest()) return 0;

    const now = new Date();
    const oneMinuteAgo = new Date(now.getTime() - 60000);
    const recentRequests = this.requests.filter(r => r > oneMinuteAgo);

    if (recentRequests.length >= this.config.requestsPerMinute) {
      const oldestRecent = recentRequests[0];
      return 60000 - (now.getTime() - oldestRecent.getTime());
    }

    return 0;
  }

  /**
   * Get requests remaining this minute
   */
  getRequestsRemainingThisMinute(): number {
    const now = new Date();
    const oneMinuteAgo = new Date(now.getTime() - 60000);
    const recentRequests = this.requests.filter(r => r > oneMinuteAgo).length;
    return Math.max(0, this.config.requestsPerMinute - recentRequests);
  }

  /**
   * Check if approaching rate limit (>80% of per-minute limit)
   */
  isApproachingRateLimit(): boolean {
    const remaining = this.getRequestsRemainingThisMinute();
    const used = this.config.requestsPerMinute - remaining;
    return (used / this.config.requestsPerMinute) >= 0.8;
  }
}
