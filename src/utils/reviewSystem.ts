import { UserProgress, TopicProgress } from '../types';

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
 * Calculate review status based on topic progress
 */
export function calculateReviewStatus(progress: TopicProgress | undefined): ReviewStatus {
  if (!progress) return 'new';
  
  if (progress.status === 'mastered') {
    // Check if review is due
    if (progress.lastAccessed) {
      const daysSinceReview = Math.floor(
        (Date.now() - progress.lastAccessed.getTime()) / (1000 * 60 * 60 * 24)
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
  
  allTopicIds.forEach((topicId, index) => {
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
