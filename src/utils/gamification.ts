import { GamificationData } from '../types';

// XP Calculation Constants
export const XP_PER_CORRECT_ANSWER = 10;
export const XP_PERFECT_BONUS = 20;
export const XP_DAILY_STREAK_BONUS = 5;
export const XP_FIRST_TIME_COMPLETION = 50;
export const XP_REVIEW_MASTERED = 15;

// Level System
export function calculateLevel(xp: number): number {
  const levels = [
    0, 100, 250, 450, 700, 1000, 1350, 1750, 2200, 2700
  ];
  
  let level = 1;
  for (let i = 0; i < levels.length; i++) {
    if (xp >= levels[i]) {
      level = i + 1;
    } else {
      break;
    }
  }
  
  if (xp >= levels[levels.length - 1]) {
    const baseXP = levels[levels.length - 1];
    const baseLevel = levels.length;
    const increment = 550;
    const additionalLevels = Math.floor((xp - baseXP) / increment);
    level = baseLevel + additionalLevels;
  }
  
  return level;
}

export function getXPForNextLevel(currentXP: number): number {
  const currentLevel = calculateLevel(currentXP);
  const nextLevel = currentLevel + 1;
  const levels = [0, 100, 250, 450, 700, 1000, 1350, 1750, 2200, 2700];
  
  if (nextLevel - 1 < levels.length) {
    return levels[nextLevel - 1];
  }
  
  const baseXP = levels[levels.length - 1];
  const baseLevel = levels.length;
  const increment = 550;
  return baseXP + (nextLevel - baseLevel) * increment;
}

export function getXPProgress(currentXP: number): { current: number; required: number; percentage: number } {
  const currentLevel = calculateLevel(currentXP);
  const nextLevelXP = getXPForNextLevel(currentXP);
  const levels = [0, 100, 250, 450, 700, 1000, 1350, 1750, 2200, 2700];
  let currentLevelStartXP = 0;
  
  if (currentLevel - 1 < levels.length) {
    currentLevelStartXP = levels[currentLevel - 1];
  } else {
    const baseXP = levels[levels.length - 1];
    const baseLevel = levels.length;
    const increment = 550;
    currentLevelStartXP = baseXP + (currentLevel - baseLevel) * increment;
  }
  
  const current = currentXP - currentLevelStartXP;
  const required = nextLevelXP - currentLevelStartXP;
  const percentage = Math.min(100, Math.round((current / required) * 100));
  
  return { current, required, percentage };
}

export function initializeGamificationData(): GamificationData {
  return {
    totalXP: 0,
    currentStreak: 0,
    longestStreak: 0,
    lastActivityDate: null,
    streakFreezeAvailable: true,
    lastStreakFreezeDate: null,
    achievements: [],
    completedTopics: [],
    perfectQuizzes: 0,
    totalQuizzes: 0,
    consecutivePerfectQuizzes: 0,
    quizzesByTimeOfDay: {
      morning: 0,
      night: 0,
    },
    themeChanges: 0,
    dailyXP: 0,
    dailyGoal: 20,
    lastDailyReset: null,
    weeklyChallenge: null,
    activityHistory: [],
    // Phase 4: Gems and Challenge Modes
    gems: 0,
    lastDailyLoginGems: null,
    gemTransactions: [],
    purchasedItems: [],
    challengeModeStats: {
      normalCompleted: 0,
      timedCompleted: 0,
      lightningCompleted: 0,
      randomMixCompleted: 0,
      hardModeCompleted: 0,
      perfectRunStreak: 0,
      bestPerfectRun: 0,
    },
    selectedTheme: 'default',
    selectedBadge: '⭐',
    // Endless Challenge Mode
    challengeModeHighScore: 0,
  };
}

// Daily Goal Management
export function checkAndResetDailyProgress(
  lastDailyReset: string | null,
  currentDailyXP: number
): { shouldReset: boolean; xpEarned: number } {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  
  if (!lastDailyReset) {
    return { shouldReset: true, xpEarned: currentDailyXP };
  }
  
  const lastReset = new Date(lastDailyReset);
  lastReset.setHours(0, 0, 0, 0);
  
  const daysDiff = Math.floor((today.getTime() - lastReset.getTime()) / (1000 * 60 * 60 * 24));
  
  if (daysDiff >= 1) {
    return { shouldReset: true, xpEarned: currentDailyXP };
  }
  
  return { shouldReset: false, xpEarned: 0 };
}
