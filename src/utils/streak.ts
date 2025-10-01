import { XP_DAILY_STREAK_BONUS } from './gamification';

export function updateStreak(
  lastActivityDate: string | null,
  currentStreak: number,
  streakFreezeAvailable: boolean
): { newStreak: number; streakFreezeUsed: boolean; bonusXP: number } {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  
  if (!lastActivityDate) {
    return { newStreak: 1, streakFreezeUsed: false, bonusXP: XP_DAILY_STREAK_BONUS };
  }
  
  const lastDate = new Date(lastActivityDate);
  lastDate.setHours(0, 0, 0, 0);
  
  const daysDiff = Math.floor((today.getTime() - lastDate.getTime()) / (1000 * 60 * 60 * 24));
  
  if (daysDiff === 0) {
    return { newStreak: currentStreak, streakFreezeUsed: false, bonusXP: 0 };
  } else if (daysDiff === 1) {
    const newStreak = currentStreak + 1;
    return { newStreak, streakFreezeUsed: false, bonusXP: newStreak * XP_DAILY_STREAK_BONUS };
  } else if (daysDiff === 2 && streakFreezeAvailable) {
    const newStreak = currentStreak + 1;
    return { newStreak, streakFreezeUsed: true, bonusXP: newStreak * XP_DAILY_STREAK_BONUS };
  } else {
    return { newStreak: 1, streakFreezeUsed: false, bonusXP: XP_DAILY_STREAK_BONUS };
  }
}

export function canEarnStreakFreeze(lastStreakFreezeDate: string | null): boolean {
  if (!lastStreakFreezeDate) return true;
  
  const lastDate = new Date(lastStreakFreezeDate);
  const today = new Date();
  const daysSinceLastFreeze = Math.floor(
    (today.getTime() - lastDate.getTime()) / (1000 * 60 * 60 * 24)
  );
  
  return daysSinceLastFreeze >= 7;
}
