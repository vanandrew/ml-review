import { QuizScore } from '../types';

/**
 * Configuration constants for status calculation
 */
export const STATUS_CONFIG = {
  // Mastery thresholds
  MIN_QUIZZES_FOR_MASTERY: 2,
  MASTERY_SCORE_THRESHOLD: 80, // percent
  RECENT_QUIZ_WINDOW: 3, // last N quizzes to consider
  PERFECT_STREAK_FOR_MASTERY: 2, // consecutive 100% scores
  
  // Degradation thresholds
  DEGRADATION_SCORE_THRESHOLD: 70, // single quiz below this
  DEGRADATION_AVERAGE_THRESHOLD: 75, // average below this
};

/**
 * Calculate percentage score from a quiz score
 */
function calculateScorePercent(score: QuizScore): number {
  return (score.score / score.totalQuestions) * 100;
}

/**
 * Get the most recent N quiz scores
 */
function getRecentScores(quizScores: QuizScore[], count: number): QuizScore[] {
  return quizScores.slice(-count);
}

/**
 * Calculate average score percentage from quiz scores
 */
function calculateAveragePercent(quizScores: QuizScore[]): number {
  if (quizScores.length === 0) return 0;
  
  const totalPercent = quizScores.reduce((sum, score) => {
    return sum + calculateScorePercent(score);
  }, 0);
  
  return totalPercent / quizScores.length;
}

/**
 * Check if user has achieved a perfect streak
 */
function hasPerfectStreak(quizScores: QuizScore[], streakLength: number): boolean {
  if (quizScores.length < streakLength) return false;
  
  const recentScores = getRecentScores(quizScores, streakLength);
  return recentScores.every(score => 
    calculateScorePercent(score) === 100
  );
}

/**
 * Check if topic meets mastery criteria
 * 
 * Mastery is achieved when:
 * - User has taken at least MIN_QUIZZES_FOR_MASTERY quizzes, AND
 * - Recent performance average >= MASTERY_SCORE_THRESHOLD, OR
 * - User has achieved PERFECT_STREAK_FOR_MASTERY consecutive perfect scores
 */
export function meetsMasteryCriteria(quizScores: QuizScore[]): boolean {
  if (quizScores.length < STATUS_CONFIG.MIN_QUIZZES_FOR_MASTERY) {
    return false;
  }
  
  // Check for perfect streak (fast track to mastery)
  if (hasPerfectStreak(quizScores, STATUS_CONFIG.PERFECT_STREAK_FOR_MASTERY)) {
    return true;
  }
  
  // Check recent performance average
  const recentScores = getRecentScores(quizScores, STATUS_CONFIG.RECENT_QUIZ_WINDOW);
  const averagePercent = calculateAveragePercent(recentScores);
  
  return averagePercent >= STATUS_CONFIG.MASTERY_SCORE_THRESHOLD;
}

/**
 * Check if mastered topic has degraded
 * 
 * Degradation occurs when:
 * - Latest quiz score < DEGRADATION_SCORE_THRESHOLD, OR
 * - Average of recent quizzes < DEGRADATION_AVERAGE_THRESHOLD
 */
export function hasDegraded(quizScores: QuizScore[]): boolean {
  if (quizScores.length === 0) return false;
  
  // Check latest score
  const latestScore = quizScores[quizScores.length - 1];
  const latestPercent = calculateScorePercent(latestScore);
  
  if (latestPercent < STATUS_CONFIG.DEGRADATION_SCORE_THRESHOLD) {
    return true;
  }
  
  // Check recent average (if we have enough scores)
  if (quizScores.length >= STATUS_CONFIG.RECENT_QUIZ_WINDOW) {
    const recentScores = getRecentScores(quizScores, STATUS_CONFIG.RECENT_QUIZ_WINDOW);
    const averagePercent = calculateAveragePercent(recentScores);
    
    if (averagePercent < STATUS_CONFIG.DEGRADATION_AVERAGE_THRESHOLD) {
      return true;
    }
  }
  
  return false;
}

/**
 * Automatically calculate topic status based on quiz performance
 * 
 * Status transitions:
 * - not_started → reviewing: When user takes first quiz
 * - reviewing → mastered: When user meets mastery criteria
 * - mastered → reviewing: When performance degrades
 */
export function calculateTopicStatus(
  currentStatus: 'not_started' | 'reviewing' | 'mastered',
  quizScores: QuizScore[]
): 'not_started' | 'reviewing' | 'mastered' {
  // If no quizzes taken, status remains not_started
  if (quizScores.length === 0) {
    return 'not_started';
  }
  
  // If at least one quiz taken, minimum status is reviewing
  if (currentStatus === 'not_started') {
    // Check if already meets mastery on first attempt
    if (meetsMasteryCriteria(quizScores)) {
      return 'mastered';
    }
    return 'reviewing';
  }
  
  // Check if reviewing → mastered
  if (currentStatus === 'reviewing') {
    if (meetsMasteryCriteria(quizScores)) {
      return 'mastered';
    }
    return 'reviewing';
  }
  
  // Check if mastered → reviewing (degradation)
  if (currentStatus === 'mastered') {
    if (hasDegraded(quizScores)) {
      return 'reviewing';
    }
    return 'mastered';
  }
  
  return currentStatus;
}

/**
 * Get status change information for notifications
 * 
 * Returns information about what changed and why
 */
export function getStatusChangeInfo(
  oldStatus: 'not_started' | 'reviewing' | 'mastered',
  newStatus: 'not_started' | 'reviewing' | 'mastered',
  quizScores: QuizScore[]
): {
  changed: boolean;
  message: string;
  isUpgrade: boolean;
} {
  if (oldStatus === newStatus) {
    return {
      changed: false,
      message: '',
      isUpgrade: false,
    };
  }
  
  // not_started → reviewing
  if (oldStatus === 'not_started' && newStatus === 'reviewing') {
    return {
      changed: true,
      message: 'Great start! Keep practicing to achieve mastery.',
      isUpgrade: true,
    };
  }
  
  // not_started → mastered (exceptional performance)
  if (oldStatus === 'not_started' && newStatus === 'mastered') {
    return {
      changed: true,
      message: '🎓 Amazing! You\'ve mastered this topic on your first try!',
      isUpgrade: true,
    };
  }
  
  // reviewing → mastered
  if (oldStatus === 'reviewing' && newStatus === 'mastered') {
    const recentScores = getRecentScores(quizScores, STATUS_CONFIG.RECENT_QUIZ_WINDOW);
    const avgPercent = Math.round(calculateAveragePercent(recentScores));
    
    return {
      changed: true,
      message: `🎓 Congratulations! You've mastered this topic with an average score of ${avgPercent}%!`,
      isUpgrade: true,
    };
  }
  
  // mastered → reviewing (degradation)
  if (oldStatus === 'mastered' && newStatus === 'reviewing') {
    const latestScore = quizScores[quizScores.length - 1];
    const latestPercent = Math.round(calculateScorePercent(latestScore));
    
    return {
      changed: true,
      message: `Keep practicing! Your recent score of ${latestPercent}% shows you need more review to maintain mastery.`,
      isUpgrade: false,
    };
  }
  
  return {
    changed: true,
    message: 'Status updated based on your quiz performance.',
    isUpgrade: false,
  };
}
