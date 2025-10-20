/**
 * Decay System for Mastered Topics
 * 
 * This module handles the automatic downgrade of mastered topics back to reviewing
 * based on time elapsed and mastery strength. Topics with higher mastery strength
 * (from consistent high performance) decay more slowly.
 */

import { UserProgress, TopicProgress } from '../types';
import { hasDecayed } from './reviewSystem';

export interface DecayCheckResult {
  topicId: string;
  wasDecayed: boolean;
  daysUntilDecay: number | null;
  masteryStrength: number;
}

/**
 * Check all mastered topics for decay and return which ones need review
 */
export function checkAllTopicsForDecay(userProgress: UserProgress): DecayCheckResult[] {
  const results: DecayCheckResult[] = [];
  
  Object.keys(userProgress).forEach(topicId => {
    const progress = userProgress[topicId];
    
    // Only check mastered topics
    if (progress.status === 'mastered') {
      const decayed = hasDecayed(progress);
      const daysUntilDecay = calculateDaysUntilDecay(progress);
      
      results.push({
        topicId,
        wasDecayed: decayed,
        daysUntilDecay,
        masteryStrength: progress.masteryStrength || 0,
      });
    }
  });
  
  return results;
}

/**
 * Calculate days until a mastered topic will decay
 * Returns null if not mastered or missing data
 */
export function calculateDaysUntilDecay(progress: TopicProgress): number | null {
  if (progress.status !== 'mastered' || !progress.lastMasteredDate) {
    return null;
  }
  
  const masteryStrength = progress.masteryStrength || 0;
  const lastScore = progress.quizScores?.[progress.quizScores.length - 1];
  const lastScorePercent = lastScore ? (lastScore.score / lastScore.totalQuestions) * 100 : null;
  
  // Calculate decay days based on mastery strength
  const BASE_DECAY_DAYS = 7;
  const MIN_DECAY_DAYS = 2;
  const MAX_DECAY_DAYS = 30;
  
  const strengthMultiplier = masteryStrength / 100;
  let decayDays = BASE_DECAY_DAYS + (MAX_DECAY_DAYS - BASE_DECAY_DAYS) * strengthMultiplier;
  
  // Adjust based on last score
  if (lastScorePercent !== null) {
    if (lastScorePercent === 100) {
      decayDays *= 1.2;
    } else if (lastScorePercent >= 90) {
      decayDays *= 1.1;
    } else if (lastScorePercent < 80) {
      decayDays *= 0.7;
    }
  }
  
  decayDays = Math.max(MIN_DECAY_DAYS, Math.min(Math.round(decayDays), MAX_DECAY_DAYS));
  
  const lastMasteredDate = progress.lastMasteredDate instanceof Date 
    ? progress.lastMasteredDate 
    : new Date(progress.lastMasteredDate);
  const daysSinceMastery = Math.floor(
    (Date.now() - lastMasteredDate.getTime()) / (1000 * 60 * 60 * 24)
  );
  
  return Math.max(0, decayDays - daysSinceMastery);
}

/**
 * Apply decay to all eligible topics in user progress
 * Returns updated progress with decayed topics marked as 'reviewing'
 */
export function applyDecayToProgress(userProgress: UserProgress): {
  updatedProgress: UserProgress;
  decayedTopicIds: string[];
} {
  const updatedProgress = { ...userProgress };
  const decayedTopicIds: string[] = [];
  
  Object.keys(updatedProgress).forEach(topicId => {
    const progress = updatedProgress[topicId];
    
    if (progress.status === 'mastered' && hasDecayed(progress)) {
      updatedProgress[topicId] = {
        ...progress,
        status: 'reviewing',
        lastAccessed: new Date(), // Update access time to trigger review
      };
      decayedTopicIds.push(topicId);
    }
  });
  
  return { updatedProgress, decayedTopicIds };
}

/**
 * Get statistics about decay across all topics
 */
export function getDecayStatistics(userProgress: UserProgress): {
  totalMastered: number;
  decayingSoon: number; // Within 7 days
  needsReview: number; // Already decayed
  averageMasteryStrength: number;
} {
  const masteredTopics = Object.values(userProgress).filter(p => p.status === 'mastered');
  
  let decayingSoon = 0;
  let needsReview = 0;
  let totalStrength = 0;
  
  masteredTopics.forEach(progress => {
    if (hasDecayed(progress)) {
      needsReview++;
    } else {
      const daysUntil = calculateDaysUntilDecay(progress);
      if (daysUntil !== null && daysUntil <= 7) {
        decayingSoon++;
      }
    }
    totalStrength += progress.masteryStrength || 0;
  });
  
  return {
    totalMastered: masteredTopics.length,
    decayingSoon,
    needsReview,
    averageMasteryStrength: masteredTopics.length > 0 
      ? Math.round(totalStrength / masteredTopics.length) 
      : 0,
  };
}
