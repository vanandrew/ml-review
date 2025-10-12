import { UserProgress, TopicProgress, QuizScore } from '../types';

export type ReviewStatus = 'new' | 'learning' | 'mastered' | 'needs-review';

export interface ReviewItem {
  topicId: string;
  categoryId: string;
  status: ReviewStatus;
  nextReviewDate: Date;
  lastScore: number | null;
  reviewCount: number;
  priority: number; // 0-10, higher = more urgent
}

/**
 * Spaced repetition intervals in days based on review count and performance
 */
const REVIEW_INTERVALS = {
  new: 0,        // Review immediately
  learning: 1,   // Review after 1 day
  mastered: {
    low: 3,      // Low score (60-79%)
    medium: 7,   // Medium score (80-89%)
    high: 14,    // High score (90-99%)
    perfect: 30, // Perfect score (100%)
  }
};

/**
 * Decay configuration for mastered topics
 * Topics decay faster if they have lower mastery strength
 */
const DECAY_CONFIG = {
  BASE_DECAY_DAYS: 30,          // Base days before decay starts
  MIN_DECAY_DAYS: 7,            // Minimum days before decay (weak mastery)
  MAX_DECAY_DAYS: 90,           // Maximum days before decay (strong mastery)
  MASTERY_STRENGTH_THRESHOLD: 50, // Below this, decay is faster
};

/**
 * Calculate mastery strength based on quiz performance history
 * Returns a value from 0-100 indicating how strong the mastery is
 * Higher values = more consistent high performance = slower decay
 */
export function calculateMasteryStrength(quizScores: QuizScore[]): number {
  if (quizScores.length === 0) return 0;
  
  // Factors that contribute to mastery strength:
  // 1. Number of quizzes taken (more = stronger)
  // 2. Consistency of high scores (90%+)
  // 3. Recency of perfect scores
  // 4. Overall average performance
  
  const recentScores = quizScores.slice(-10); // Last 10 quizzes
  const totalQuizzes = quizScores.length;
  
  // Calculate average score percentage
  const avgPercent = recentScores.reduce((sum, score) => {
    return sum + (score.score / score.totalQuestions) * 100;
  }, 0) / recentScores.length;
  
  // Count high scores (90%+) in recent quizzes
  const highScoreCount = recentScores.filter(score => 
    (score.score / score.totalQuestions) * 100 >= 90
  ).length;
  
  // Calculate streak of consecutive high scores (90%+)
  let highScoreStreak = 0;
  for (let i = recentScores.length - 1; i >= 0; i--) {
    const percent = (recentScores[i].score / recentScores[i].totalQuestions) * 100;
    if (percent >= 90) {
      highScoreStreak++;
    } else {
      break;
    }
  }
  
  // Calculate components (each out of 25 points, total = 100)
  const avgComponent = Math.min((avgPercent / 100) * 25, 25);
  const consistencyComponent = (highScoreCount / recentScores.length) * 25;
  const streakComponent = Math.min((highScoreStreak / 5) * 25, 25); // 5+ streak = max points
  const volumeComponent = Math.min((totalQuizzes / 20) * 25, 25); // 20+ quizzes = max points
  
  return Math.round(avgComponent + consistencyComponent + streakComponent + volumeComponent);
}

/**
 * Calculate days until decay based on mastery strength
 * Stronger mastery = longer time before needing review
 */
export function calculateDecayDays(masteryStrength: number, lastScore: number | null): number {
  // Base decay days modified by mastery strength
  const strengthMultiplier = masteryStrength / 100;
  
  let decayDays = DECAY_CONFIG.BASE_DECAY_DAYS + 
    (DECAY_CONFIG.MAX_DECAY_DAYS - DECAY_CONFIG.BASE_DECAY_DAYS) * strengthMultiplier;
  
  // Further adjust based on last score
  if (lastScore !== null) {
    if (lastScore === 100) {
      decayDays *= 1.2; // 20% longer for perfect scores
    } else if (lastScore >= 90) {
      decayDays *= 1.1; // 10% longer for high scores
    } else if (lastScore < 80) {
      decayDays *= 0.7; // 30% shorter for lower scores
    }
  }
  
  // Ensure within bounds
  return Math.max(
    DECAY_CONFIG.MIN_DECAY_DAYS,
    Math.min(Math.round(decayDays), DECAY_CONFIG.MAX_DECAY_DAYS)
  );
}

/**
 * Check if a mastered topic has decayed and needs review
 */
export function hasDecayed(progress: TopicProgress): boolean {
  if (progress.status !== 'mastered') return false;
  if (!progress.lastMasteredDate) return false;
  
  const masteryStrength = progress.masteryStrength || 0;
  const lastScore = progress.quizScores?.[progress.quizScores.length - 1];
  const lastScorePercent = lastScore ? (lastScore.score / lastScore.totalQuestions) * 100 : null;
  
  const decayDays = calculateDecayDays(masteryStrength, lastScorePercent);
  const lastMasteredDate = progress.lastMasteredDate instanceof Date 
    ? progress.lastMasteredDate 
    : new Date(progress.lastMasteredDate);
  const daysSinceMastery = Math.floor(
    (Date.now() - lastMasteredDate.getTime()) / (1000 * 60 * 60 * 24)
  );
  
  return daysSinceMastery >= decayDays;
}

/**
 * Calculate review status based on topic progress
 */
export function calculateReviewStatus(progress: TopicProgress | undefined): ReviewStatus {
  if (!progress) return 'new';
  
  if (progress.status === 'mastered') {
    // Check if topic has decayed due to time
    if (hasDecayed(progress)) {
      return 'needs-review';
    }
    
    // Check if review is due based on spaced repetition
    if (progress.lastAccessed) {
      const lastAccessed = progress.lastAccessed instanceof Date 
        ? progress.lastAccessed 
        : new Date(progress.lastAccessed);
      const daysSinceReview = Math.floor(
        (Date.now() - lastAccessed.getTime()) / (1000 * 60 * 60 * 24)
      );
      
      // Get the last quiz score to determine interval
      const lastScore = progress.quizScores?.[progress.quizScores.length - 1];
      const scorePercent = lastScore ? (lastScore.score / lastScore.totalQuestions) * 100 : 0;
      
      let interval = REVIEW_INTERVALS.mastered.medium;
      if (scorePercent === 100) interval = REVIEW_INTERVALS.mastered.perfect;
      else if (scorePercent >= 90) interval = REVIEW_INTERVALS.mastered.high;
      else if (scorePercent >= 80) interval = REVIEW_INTERVALS.mastered.medium;
      else interval = REVIEW_INTERVALS.mastered.low;
      
      if (daysSinceReview >= interval) {
        return 'needs-review';
      }
    }
    return 'mastered';
  }
  
  if (progress.status === 'reviewing') {
    return 'learning';
  }
  
  return 'new';
}

/**
 * Calculate next review date based on performance
 */
export function calculateNextReviewDate(
  lastReviewDate: Date,
  scorePercent: number,
  reviewCount: number
): Date {
  let daysUntilReview: number;
  
  if (reviewCount === 0) {
    daysUntilReview = REVIEW_INTERVALS.learning;
  } else if (scorePercent === 100) {
    daysUntilReview = REVIEW_INTERVALS.mastered.perfect;
  } else if (scorePercent >= 90) {
    daysUntilReview = REVIEW_INTERVALS.mastered.high;
  } else if (scorePercent >= 80) {
    daysUntilReview = REVIEW_INTERVALS.mastered.medium;
  } else {
    daysUntilReview = REVIEW_INTERVALS.mastered.low;
  }
  
  const nextDate = new Date(lastReviewDate);
  nextDate.setDate(nextDate.getDate() + daysUntilReview);
  return nextDate;
}

/**
 * Calculate priority for review (0-10, higher = more urgent)
 */
export function calculateReviewPriority(
  status: ReviewStatus,
  nextReviewDate: Date,
  lastScore: number | null
): number {
  const now = new Date();
  const daysOverdue = Math.floor((now.getTime() - nextReviewDate.getTime()) / (1000 * 60 * 60 * 24));
  
  let priority = 5; // Base priority
  
  // Increase priority based on how overdue the review is
  if (status === 'needs-review') {
    priority += Math.min(daysOverdue, 5); // +1 per day overdue, max +5
  }
  
  // Increase priority for lower scores
  if (lastScore !== null && lastScore < 80) {
    priority += 2;
  }
  
  // New topics have medium priority
  if (status === 'new') {
    priority = 3;
  }
  
  // Learning topics have slightly higher priority
  if (status === 'learning') {
    priority = 6;
  }
  
  return Math.min(Math.max(priority, 0), 10); // Clamp to 0-10
}

/**
 * Get all topics that need review
 */
export function getReviewQueue(userProgress: UserProgress, allTopicIds: string[]): ReviewItem[] {
  const reviewItems: ReviewItem[] = [];
  
  allTopicIds.forEach((topicId) => {
    const progress = userProgress[topicId];
    const status = calculateReviewStatus(progress);
    
    // Calculate next review date
    let nextReviewDate = new Date();
    if (progress?.lastAccessed) {
      const lastScore = progress.quizScores?.[progress.quizScores.length - 1];
      const scorePercent = lastScore ? (lastScore.score / lastScore.totalQuestions) * 100 : 0;
      const reviewCount = progress.quizScores?.length || 0;
      nextReviewDate = calculateNextReviewDate(progress.lastAccessed, scorePercent, reviewCount);
    }
    
    // Get last score
    const lastScore = progress?.quizScores?.[progress.quizScores.length - 1];
    const scorePercent = lastScore ? (lastScore.score / lastScore.totalQuestions) * 100 : null;
    
    // Calculate priority
    const priority = calculateReviewPriority(status, nextReviewDate, scorePercent);
    
    reviewItems.push({
      topicId,
      categoryId: '', // Will be filled by caller
      status,
      nextReviewDate,
      lastScore: scorePercent,
      reviewCount: progress?.quizScores?.length || 0,
      priority,
    });
  });
  
  // Sort by priority (highest first), then by next review date
  return reviewItems.sort((a, b) => {
    if (b.priority !== a.priority) {
      return b.priority - a.priority;
    }
    return a.nextReviewDate.getTime() - b.nextReviewDate.getTime();
  });
}

/**
 * Get topics that are due for review today
 */
export function getDueReviews(reviewQueue: ReviewItem[]): ReviewItem[] {
  const now = new Date();
  now.setHours(0, 0, 0, 0); // Start of today
  
  return reviewQueue.filter(item => {
    return item.status === 'needs-review' && item.nextReviewDate <= now;
  });
}

/**
 * Get weak topics (low scores that need practice)
 */
export function getWeakTopics(reviewQueue: ReviewItem[]): ReviewItem[] {
  return reviewQueue
    .filter(item => item.lastScore !== null && item.lastScore < 80)
    .sort((a, b) => (a.lastScore || 100) - (b.lastScore || 100))
    .slice(0, 5); // Top 5 weakest
}
