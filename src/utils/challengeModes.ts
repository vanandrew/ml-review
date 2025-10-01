export type QuizMode = 'normal' | 'timed' | 'lightning' | 'random-mix' | 'hard-mode';

export interface QuizModeConfig {
  id: QuizMode;
  name: string;
  description: string;
  icon: string;
  xpMultiplier: number;
  timeLimit?: number; // seconds per question
  totalQuestions?: number;
  requirements?: string;
  color: string;
  bgColor: string;
}

export const QUIZ_MODES: QuizModeConfig[] = [
  {
    id: 'normal',
    name: 'Normal Mode',
    description: 'Standard quiz with no time pressure',
    icon: '📝',
    xpMultiplier: 1.0,
    color: 'text-blue-600 dark:text-blue-400',
    bgColor: 'bg-blue-50 dark:bg-blue-900/20',
  },
  {
    id: 'timed',
    name: 'Timed Challenge',
    description: 'Complete within time limit for bonus XP',
    icon: '⚡',
    xpMultiplier: 1.5,
    timeLimit: 30, // 30 seconds per question
    color: 'text-yellow-600 dark:text-yellow-400',
    bgColor: 'bg-yellow-50 dark:bg-yellow-900/20',
  },
  {
    id: 'lightning',
    name: 'Lightning Round',
    description: '20 questions, 10 seconds each, double XP!',
    icon: '🌩️',
    xpMultiplier: 2.0,
    timeLimit: 10,
    totalQuestions: 20,
    color: 'text-purple-600 dark:text-purple-400',
    bgColor: 'bg-purple-50 dark:bg-purple-900/20',
  },
  {
    id: 'random-mix',
    name: 'Random Mix',
    description: 'Questions from all your mastered topics',
    icon: '🎲',
    xpMultiplier: 1.3,
    requirements: 'Requires at least 3 mastered topics',
    color: 'text-green-600 dark:text-green-400',
    bgColor: 'bg-green-50 dark:bg-green-900/20',
  },
  {
    id: 'hard-mode',
    name: 'Hard Mode',
    description: 'Advanced topics only, higher rewards',
    icon: '💪',
    xpMultiplier: 1.5,
    requirements: 'Requires level 10+',
    color: 'text-red-600 dark:text-red-400',
    bgColor: 'bg-red-50 dark:bg-red-900/20',
  },
];

/**
 * Get quiz mode config by ID
 */
export function getQuizMode(modeId: QuizMode): QuizModeConfig {
  return QUIZ_MODES.find(m => m.id === modeId) || QUIZ_MODES[0];
}

/**
 * Check if user meets requirements for a quiz mode
 */
export function canAccessQuizMode(
  mode: QuizMode,
  userLevel: number,
  masteredTopicsCount: number
): boolean {
  switch (mode) {
    case 'normal':
    case 'timed':
      return true;
    case 'lightning':
      return userLevel >= 5;
    case 'random-mix':
      return masteredTopicsCount >= 3;
    case 'hard-mode':
      return userLevel >= 10;
    default:
      return true;
  }
}

/**
 * Calculate XP with mode multiplier
 */
export function calculateModeXP(baseXP: number, mode: QuizMode): number {
  const config = getQuizMode(mode);
  return Math.floor(baseXP * config.xpMultiplier);
}

/**
 * Track challenge mode statistics
 */
export interface ChallengeModeStats {
  normalCompleted: number;
  timedCompleted: number;
  lightningCompleted: number;
  randomMixCompleted: number;
  hardModeCompleted: number;
  perfectRunStreak: number; // Current streak of perfect scores
  bestPerfectRun: number; // Best streak of perfect scores
}

export function initializeChallengeModeStats(): ChallengeModeStats {
  return {
    normalCompleted: 0,
    timedCompleted: 0,
    lightningCompleted: 0,
    randomMixCompleted: 0,
    hardModeCompleted: 0,
    perfectRunStreak: 0,
    bestPerfectRun: 0,
  };
}

export function updateChallengeModeStats(
  stats: ChallengeModeStats,
  mode: QuizMode,
  isPerfect: boolean
): ChallengeModeStats {
  const updatedStats = { ...stats };
  
  // Update completion counts
  switch (mode) {
    case 'normal':
      updatedStats.normalCompleted++;
      break;
    case 'timed':
      updatedStats.timedCompleted++;
      break;
    case 'lightning':
      updatedStats.lightningCompleted++;
      break;
    case 'random-mix':
      updatedStats.randomMixCompleted++;
      break;
    case 'hard-mode':
      updatedStats.hardModeCompleted++;
      break;
  }
  
  // Update perfect run streak
  if (isPerfect) {
    updatedStats.perfectRunStreak++;
    if (updatedStats.perfectRunStreak > updatedStats.bestPerfectRun) {
      updatedStats.bestPerfectRun = updatedStats.perfectRunStreak;
    }
  } else {
    updatedStats.perfectRunStreak = 0;
  }
  
  return updatedStats;
}
