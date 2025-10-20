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


/**
 * Get tips for improving ranking
 */
export function getRankingTips(
  gamificationData: GamificationData,
  userProgress: UserProgress
): string[] {
  const tips: string[] = [];

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

  return tips.slice(0, 3); // Return top 3 tips
}
