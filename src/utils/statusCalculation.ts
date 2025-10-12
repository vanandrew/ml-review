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
 * Calculate high score streak (consecutive quizzes with 90%+ scores)
 */
export function calculateHighScoreStreak(quizScores: QuizScore[]): number {
  let streak = 0;
  for (let i = quizScores.length - 1; i >= 0; i--) {
    const percent = calculateScorePercent(quizScores[i]);
    if (percent >= 90) {
      streak++;
    } else {
      break;
    }
  }
  return streak;
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
 * 
 * Returns updated status and metadata for mastery tracking
 */
export function calculateTopicStatus(
  currentStatus: 'not_started' | 'reviewing' | 'mastered',
  quizScores: QuizScore[],
  _currentProgress?: { masteryStrength?: number; lastMasteredDate?: Date; highScoreStreak?: number }
): {
  status: 'not_started' | 'reviewing' | 'mastered';
  masteryStrength: number;
  highScoreStreak: number;
  shouldUpdateMasteredDate: boolean;
} {
  // Calculate current metrics
  const masteryStrength = calculateMasteryStrength(quizScores);
  const highScoreStreak = calculateHighScoreStreak(quizScores);
  
  // If no quizzes taken, status remains not_started
  if (quizScores.length === 0) {
    return {
      status: 'not_started',
      masteryStrength: 0,
      highScoreStreak: 0,
      shouldUpdateMasteredDate: false,
    };
  }
  
  // If at least one quiz taken, minimum status is reviewing
  if (currentStatus === 'not_started') {
    // Check if already meets mastery on first attempt
    if (meetsMasteryCriteria(quizScores)) {
      return {
        status: 'mastered',
        masteryStrength,
        highScoreStreak,
        shouldUpdateMasteredDate: true,
      };
    }
    return {
      status: 'reviewing',
      masteryStrength,
      highScoreStreak,
      shouldUpdateMasteredDate: false,
    };
  }
  
  // Check if reviewing → mastered
  if (currentStatus === 'reviewing') {
    if (meetsMasteryCriteria(quizScores)) {
      return {
        status: 'mastered',
        masteryStrength,
        highScoreStreak,
        shouldUpdateMasteredDate: true,
      };
    }
    return {
      status: 'reviewing',
      masteryStrength,
      highScoreStreak,
      shouldUpdateMasteredDate: false,
    };
  }
  
  // Check if mastered → reviewing (degradation)
  if (currentStatus === 'mastered') {
    if (hasDegraded(quizScores)) {
      return {
        status: 'reviewing',
        masteryStrength,
        highScoreStreak,
        shouldUpdateMasteredDate: false,
      };
    }
    return {
      status: 'mastered',
      masteryStrength,
      highScoreStreak,
      shouldUpdateMasteredDate: false,
    };
  }
  
  return {
    status: currentStatus,
    masteryStrength,
    highScoreStreak,
    shouldUpdateMasteredDate: false,
  };
}

/**
 * Calculate mastery strength based on quiz performance history
 * Returns a value from 0-100 indicating how strong the mastery is
 */
function calculateMasteryStrength(quizScores: QuizScore[]): number {
  if (quizScores.length === 0) return 0;
  
  const recentScores = quizScores.slice(-10); // Last 10 quizzes
  const totalQuizzes = quizScores.length;
  
  // Calculate average score percentage
  const avgPercent = recentScores.reduce((sum, score) => {
    return sum + calculateScorePercent(score);
  }, 0) / recentScores.length;
  
  // Count high scores (90%+) in recent quizzes
  const highScoreCount = recentScores.filter(score => 
    calculateScorePercent(score) >= 90
  ).length;
  
  // Calculate components
  const avgComponent = Math.min((avgPercent / 100) * 40, 40);
  const consistencyComponent = (highScoreCount / recentScores.length) * 30;
  const volumeComponent = Math.min((totalQuizzes / 20) * 30, 30);
  
  return Math.round(avgComponent + consistencyComponent + volumeComponent);
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
