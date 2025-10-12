import { UserProgress, GamificationData } from '../types';

// ============================================
// LESSON RANKING SYSTEM
// ============================================
// Reflects learning journey progression based on topics mastered,
// quiz performance, and consistency

export interface LessonRank {
  id: string;
  title: string;
  minPoints: number;
  icon: string;
  color: string;
  description: string;
}

export const LESSON_RANKS: LessonRank[] = [
  {
    id: 'novice',
    title: 'Novice Learner',
    minPoints: 0,
    icon: '🌱',
    color: 'text-gray-600 dark:text-gray-400',
    description: 'Just starting your ML journey',
  },
  {
    id: 'apprentice',
    title: 'Apprentice',
    minPoints: 100,
    icon: '📚',
    color: 'text-blue-600 dark:text-blue-400',
    description: 'Building foundational knowledge',
  },
  {
    id: 'student',
    title: 'Dedicated Student',
    minPoints: 300,
    icon: '🎓',
    color: 'text-green-600 dark:text-green-400',
    description: 'Consistently learning and growing',
  },
  {
    id: 'practitioner',
    title: 'ML Practitioner',
    minPoints: 600,
    icon: '⚙️',
    color: 'text-purple-600 dark:text-purple-400',
    description: 'Applying ML concepts effectively',
  },
  {
    id: 'specialist',
    title: 'ML Specialist',
    minPoints: 1000,
    icon: '💎',
    color: 'text-cyan-600 dark:text-cyan-400',
    description: 'Deep knowledge across multiple areas',
  },
  {
    id: 'expert',
    title: 'ML Expert',
    minPoints: 1500,
    icon: '🔬',
    color: 'text-indigo-600 dark:text-indigo-400',
    description: 'Mastering advanced concepts',
  },
  {
    id: 'master',
    title: 'ML Master',
    minPoints: 2200,
    icon: '🏆',
    color: 'text-yellow-600 dark:text-yellow-400',
    description: 'Elite understanding of ML',
  },
  {
    id: 'grandmaster',
    title: 'Grand Master',
    minPoints: 3000,
    icon: '👑',
    color: 'text-orange-600 dark:text-orange-400',
    description: 'Legendary ML knowledge',
  },
  {
    id: 'sage',
    title: 'ML Sage',
    minPoints: 4000,
    icon: '✨',
    color: 'text-pink-600 dark:text-pink-400',
    description: 'Transcendent mastery',
  },
  {
    id: 'enlightened',
    title: 'Enlightened One',
    minPoints: 5500,
    icon: '🌟',
    color: 'text-amber-600 dark:text-amber-400',
    description: 'Ultimate ML enlightenment',
  },
];

/**
 * Calculate lesson ranking points based on learning journey
 */
export function calculateLessonRankPoints(
  userProgress: UserProgress,
  gamificationData: GamificationData
): number {
  let points = 0;

  // Topics mastered (50 points each)
  const masteredCount = Object.values(userProgress).filter(
    p => p.status === 'mastered'
  ).length;
  points += masteredCount * 50;

  // Topics in review (20 points each)
  const reviewingCount = Object.values(userProgress).filter(
    p => p.status === 'reviewing'
  ).length;
  points += reviewingCount * 20;

  // Quiz performance bonus (up to 300 points)
  const totalQuizzes = gamificationData.totalQuizzes;
  const perfectQuizzes = gamificationData.perfectQuizzes;
  if (totalQuizzes > 0) {
    const perfectRate = perfectQuizzes / totalQuizzes;
    const quizBonus = Math.min(300, Math.floor(totalQuizzes * perfectRate * 5));
    points += quizBonus;
  }

  // Consistency bonus - Streak (2 points per day, up to 200)
  points += Math.min(200, gamificationData.currentStreak * 2);

  // Achievement bonus (10 points each)
  points += gamificationData.achievements.length * 10;

  // Diversity bonus - topics across different categories (up to 150 points)
  const categoriesWithProgress = new Set(
    Object.entries(userProgress)
      .filter(([_, p]) => p.status !== 'not_started')
      .map(([topicId, _]) => {
        // Extract category from topic ID (e.g., "foundations-basics" -> "foundations")
        return topicId.split('-')[0];
      })
  );
  points += Math.min(150, categoriesWithProgress.size * 20);

  // High mastery strength bonus (up to 100 points)
  const highStrengthTopics = Object.values(userProgress).filter(
    p => (p.masteryStrength || 0) >= 80
  ).length;
  points += Math.min(100, highStrengthTopics * 10);

  return points;
}

/**
 * Get current lesson rank
 */
export function getLessonRank(points: number): LessonRank {
  // Find the highest rank that user qualifies for
  const sortedRanks = [...LESSON_RANKS].sort((a, b) => b.minPoints - a.minPoints);
  return sortedRanks.find(rank => points >= rank.minPoints) || LESSON_RANKS[0];
}

/**
 * Get next lesson rank and progress toward it
 */
export function getNextLessonRank(points: number): {
  nextRank: LessonRank | null;
  currentRank: LessonRank;
  progress: number;
  pointsNeeded: number;
} {
  const currentRank = getLessonRank(points);
  const currentIndex = LESSON_RANKS.findIndex(r => r.id === currentRank.id);
  const nextRank = currentIndex < LESSON_RANKS.length - 1 ? LESSON_RANKS[currentIndex + 1] : null;

  if (!nextRank) {
    return { nextRank: null, currentRank, progress: 100, pointsNeeded: 0 };
  }

  const pointsNeeded = nextRank.minPoints - points;
  const pointsIntoRank = points - currentRank.minPoints;
  const pointsForNextRank = nextRank.minPoints - currentRank.minPoints;
  const progress = Math.min(100, Math.round((pointsIntoRank / pointsForNextRank) * 100));

  return { nextRank, currentRank, progress, pointsNeeded };
}

// ============================================
// CHALLENGE MODE RANKING SYSTEM
// ============================================
// Reflects ML expertise based on challenge performance,
// correct answers, and difficulty

export interface ChallengeRank {
  id: string;
  title: string;
  minRating: number;
  icon: string;
  color: string;
  badge: string;
  description: string;
}

export const CHALLENGE_RANKS: ChallengeRank[] = [
  {
    id: 'bronze',
    title: 'Bronze',
    minRating: 0,
    icon: '🥉',
    badge: 'Bronze',
    color: 'text-amber-700 dark:text-amber-600',
    description: 'Beginning to understand ML concepts',
  },
  {
    id: 'silver',
    title: 'Silver',
    minRating: 800,
    icon: '🥈',
    badge: 'Silver',
    color: 'text-gray-500 dark:text-gray-400',
    description: 'Solid foundational knowledge',
  },
  {
    id: 'gold',
    title: 'Gold',
    minRating: 1200,
    icon: '🥇',
    badge: 'Gold',
    color: 'text-yellow-500 dark:text-yellow-400',
    description: 'Strong grasp of ML principles',
  },
  {
    id: 'platinum',
    title: 'Platinum',
    minRating: 1600,
    icon: '💠',
    badge: 'Platinum',
    color: 'text-cyan-500 dark:text-cyan-400',
    description: 'Advanced ML understanding',
  },
  {
    id: 'diamond',
    title: 'Diamond',
    minRating: 2000,
    icon: '💎',
    badge: 'Diamond',
    color: 'text-blue-500 dark:text-blue-400',
    description: 'Expert-level ML knowledge',
  },
  {
    id: 'master',
    title: 'Master',
    minRating: 2400,
    icon: '🔷',
    badge: 'Master',
    color: 'text-purple-500 dark:text-purple-400',
    description: 'Mastery of complex ML topics',
  },
  {
    id: 'grandmaster',
    title: 'Grandmaster',
    minRating: 2800,
    icon: '🔶',
    badge: 'Grandmaster',
    color: 'text-orange-500 dark:text-orange-400',
    description: 'Elite ML expertise',
  },
  {
    id: 'legend',
    title: 'Legend',
    minRating: 3200,
    icon: '⭐',
    badge: 'Legend',
    color: 'text-red-500 dark:text-red-400',
    description: 'Legendary ML mastery',
  },
];

/**
 * Calculate challenge rating using ELO-like system
 * Based on questions answered correctly in challenge modes
 */
export function calculateChallengeRating(
  gamificationData: GamificationData,
  userProgress: UserProgress
): number {
  // Start at base rating (0 for new users)
  let rating = 0;

  // Factor 1: Challenge mode performance (up to 1500 points)
  const challengeStats = gamificationData.challengeModeStats;
  
  // Normal mode completions (5 points each)
  rating += challengeStats.normalCompleted * 5;
  
  // Timed completions (8 points each - harder)
  rating += challengeStats.timedCompleted * 8;
  
  // Lightning completions (12 points each - much harder)
  rating += challengeStats.lightningCompleted * 12;
  
  // Random mix completions (10 points each)
  rating += challengeStats.randomMixCompleted * 10;
  
  // Hard mode completions (15 points each - hardest)
  rating += challengeStats.hardModeCompleted * 15;

  // Factor 2: Perfect run streak bonus (up to 300 points)
  rating += Math.min(300, challengeStats.perfectRunStreak * 20);

  // Factor 3: Best perfect run (up to 500 points)
  // Longer perfect runs = higher skill
  rating += Math.min(500, challengeStats.bestPerfectRun * 10);

  // Factor 4: High score in endless challenge mode (up to 400 points)
  rating += Math.min(400, gamificationData.challengeModeHighScore * 2);

  // Factor 5: Quiz accuracy across all topics (up to 300 points)
  const allQuizScores = Object.values(userProgress).flatMap(p => p.quizScores || []);
  if (allQuizScores.length > 0) {
    const totalCorrect = allQuizScores.reduce((sum, s) => sum + s.score, 0);
    const totalQuestions = allQuizScores.reduce((sum, s) => sum + s.totalQuestions, 0);
    if (totalQuestions > 0) {
      const accuracy = totalCorrect / totalQuestions;
      rating += Math.floor(accuracy * 300);
    }
  }

  // Factor 6: Consecutive perfect quizzes (up to 200 points)
  rating += Math.min(200, gamificationData.consecutivePerfectQuizzes * 15);

  // Factor 7: Mastered topics bonus (indicates broad expertise)
  const masteredCount = Object.values(userProgress).filter(
    p => p.status === 'mastered'
  ).length;
  rating += masteredCount * 8;

  return Math.floor(rating);
}

/**
 * Get current challenge rank
 */
export function getChallengeRank(rating: number): ChallengeRank {
  const sortedRanks = [...CHALLENGE_RANKS].sort((a, b) => b.minRating - a.minRating);
  return sortedRanks.find(rank => rating >= rank.minRating) || CHALLENGE_RANKS[0];
}

/**
 * Get next challenge rank and progress toward it
 */
export function getNextChallengeRank(rating: number): {
  nextRank: ChallengeRank | null;
  currentRank: ChallengeRank;
  progress: number;
  ratingNeeded: number;
} {
  const currentRank = getChallengeRank(rating);
  const currentIndex = CHALLENGE_RANKS.findIndex(r => r.id === currentRank.id);
  const nextRank = currentIndex < CHALLENGE_RANKS.length - 1 ? CHALLENGE_RANKS[currentIndex + 1] : null;

  if (!nextRank) {
    return { nextRank: null, currentRank, progress: 100, ratingNeeded: 0 };
  }

  const ratingNeeded = nextRank.minRating - rating;
  const ratingIntoRank = rating - currentRank.minRating;
  const ratingForNextRank = nextRank.minRating - currentRank.minRating;
  const progress = Math.min(100, Math.round((ratingIntoRank / ratingForNextRank) * 100));

  return { nextRank, currentRank, progress, ratingNeeded };
}

/**
 * Get ranking percentile (0-100)
 * Estimates where user stands compared to theoretical player base
 */
export function getRankingPercentile(rating: number): number {
  // Using a normal distribution-like curve
  // Bronze: 0-30th percentile
  // Silver: 30-50th
  // Gold: 50-70th
  // Platinum: 70-85th
  // Diamond: 85-93rd
  // Master: 93-97th
  // Grandmaster: 97-99th
  // Legend: 99-100th

  if (rating < 800) return Math.min(30, (rating / 800) * 30);
  if (rating < 1200) return 30 + ((rating - 800) / 400) * 20;
  if (rating < 1600) return 50 + ((rating - 1200) / 400) * 20;
  if (rating < 2000) return 70 + ((rating - 1600) / 400) * 15;
  if (rating < 2400) return 85 + ((rating - 2000) / 400) * 8;
  if (rating < 2800) return 93 + ((rating - 2400) / 400) * 4;
  if (rating < 3200) return 97 + ((rating - 2800) / 400) * 2;
  return Math.min(100, 99 + ((rating - 3200) / 800) * 1);
}

/**
 * Get tips for improving ranking
 */
export function getRankingTips(
  type: 'lesson' | 'challenge',
  gamificationData: GamificationData,
  userProgress: UserProgress
): string[] {
  const tips: string[] = [];

  if (type === 'lesson') {
    const masteredCount = Object.values(userProgress).filter(p => p.status === 'mastered').length;
    const reviewingCount = Object.values(userProgress).filter(p => p.status === 'reviewing').length;

    if (masteredCount < 5) {
      tips.push('🎯 Master more topics to gain 50 points each');
    }
    if (reviewingCount < 3) {
      tips.push('📖 Start reviewing topics to earn 20 points each');
    }
    if (gamificationData.currentStreak < 7) {
      tips.push('🔥 Build your streak! Each day adds 2 points (up to 200)');
    }
    if (gamificationData.achievements.length < 5) {
      tips.push('🏆 Unlock achievements for 10 points each');
    }
    const perfectRate = gamificationData.totalQuizzes > 0 
      ? gamificationData.perfectQuizzes / gamificationData.totalQuizzes 
      : 0;
    if (perfectRate < 0.3) {
      tips.push('⭐ Improve quiz performance for up to 300 bonus points');
    }
  } else {
    if (gamificationData.challengeModeStats.lightningCompleted < 5) {
      tips.push('⚡ Complete Lightning Rounds for 12 points each');
    }
    if (gamificationData.challengeModeStats.hardModeCompleted < 5) {
      tips.push('💪 Try Hard Mode challenges for 15 points each');
    }
    if (gamificationData.challengeModeStats.perfectRunStreak < 5) {
      tips.push('🎯 Build your perfect run streak for major bonuses');
    }
    if (gamificationData.challengeModeHighScore < 50) {
      tips.push('🚀 Push your endless challenge high score higher');
    }
    if (gamificationData.consecutivePerfectQuizzes < 5) {
      tips.push('💯 Maintain consecutive perfect quizzes for up to 200 points');
    }
  }

  return tips.slice(0, 3); // Return top 3 tips
}
