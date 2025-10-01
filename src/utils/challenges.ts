import { WeeklyChallenge } from '../types';

// Weekly Challenge Definitions
const WEEKLY_CHALLENGES = [
  {
    id: 'master-topics',
    title: 'Master 3 Topics',
    description: 'Complete and master 3 topics this week',
    target: 3,
    reward: 100,
    type: 'master_topics' as const,
  },
  {
    id: 'complete-quizzes',
    title: 'Quiz Marathon',
    description: 'Complete 10 quizzes this week',
    target: 10,
    reward: 150,
    type: 'complete_quizzes' as const,
  },
  {
    id: 'review-topics',
    title: 'Review Master',
    description: 'Review 5 mastered topics this week',
    target: 5,
    reward: 75,
    type: 'review_topics' as const,
  },
];

export function generateWeeklyChallenge(): WeeklyChallenge {
  const today = new Date();
  const startOfWeek = new Date(today);
  startOfWeek.setHours(0, 0, 0, 0);
  startOfWeek.setDate(today.getDate() - today.getDay()); // Start on Sunday
  
  const endOfWeek = new Date(startOfWeek);
  endOfWeek.setDate(startOfWeek.getDate() + 7);
  
  // Rotate challenges based on week number
  const weekNumber = Math.floor(today.getTime() / (7 * 24 * 60 * 60 * 1000));
  const challengeTemplate = WEEKLY_CHALLENGES[weekNumber % WEEKLY_CHALLENGES.length];
  
  return {
    ...challengeTemplate,
    progress: 0,
    startDate: startOfWeek.toISOString(),
    endDate: endOfWeek.toISOString(),
  };
}

export function shouldGenerateNewChallenge(
  currentChallenge: WeeklyChallenge | null
): boolean {
  if (!currentChallenge) return true;
  
  const now = new Date();
  const endDate = new Date(currentChallenge.endDate);
  
  return now >= endDate;
}

export function updateChallengeProgress(
  challenge: WeeklyChallenge,
  activityType: 'master_topics' | 'complete_quizzes' | 'review_topics',
  increment: number = 1
): WeeklyChallenge {
  if (challenge.type !== activityType) {
    return challenge;
  }
  
  return {
    ...challenge,
    progress: Math.min(challenge.progress + increment, challenge.target),
  };
}

export function isChallengeComplete(challenge: WeeklyChallenge): boolean {
  return challenge.progress >= challenge.target;
}
