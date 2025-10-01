import { GamificationData, UserProgress, AchievementDefinition } from '../types';

export const ACHIEVEMENTS: AchievementDefinition[] = [
  {
    id: 'first-steps',
    title: 'First Steps',
    description: 'Complete your first quiz',
    icon: '🎯',
    category: 'learning',
    check: (data) => data.totalQuizzes >= 1,
  },
  {
    id: 'foundation-master',
    title: 'Foundation Master',
    description: 'Master all foundation topics',
    icon: '🏗️',
    category: 'learning',
    check: (_data, progress) => {
      const foundationTopics = [
        'supervised-vs-unsupervised-vs-reinforcement',
        'bias-variance-tradeoff',
        'overfitting-underfitting',
        'regularization',
        'cross-validation',
        'train-validation-test-split',
        'evaluation-metrics',
        'hyperparameter-tuning',
      ];
      return foundationTopics.every(topic => 
        progress[topic]?.status === 'mastered'
      );
    },
  },
  {
    id: 'quiz-champion',
    title: 'Quiz Champion',
    description: 'Score 100% on 10 quizzes',
    icon: '🧠',
    category: 'learning',
    check: (data) => data.perfectQuizzes >= 10,
  },
  {
    id: 'knowledge-seeker',
    title: 'Knowledge Seeker',
    description: 'Complete 50 quizzes total',
    icon: '📚',
    category: 'learning',
    check: (data) => data.totalQuizzes >= 50,
  },
  {
    id: 'speed-learner',
    title: 'Speed Learner',
    description: 'Complete 5 topics in one day',
    icon: '⚡',
    category: 'learning',
    check: (_data, progress) => {
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      
      let topicsCompletedToday = 0;
      Object.values(progress).forEach(topicProgress => {
        if (topicProgress.firstCompletion) {
          const completionDate = new Date(topicProgress.firstCompletion);
          completionDate.setHours(0, 0, 0, 0);
          if (completionDate.getTime() === today.getTime()) {
            topicsCompletedToday++;
          }
        }
      });
      
      return topicsCompletedToday >= 5;
    },
  },
  {
    id: 'week-warrior',
    title: 'Week Warrior',
    description: 'Maintain a 7-day streak',
    icon: '🔥',
    category: 'streak',
    check: (data) => data.currentStreak >= 7,
  },
  {
    id: 'month-master',
    title: 'Month Master',
    description: 'Maintain a 30-day streak',
    icon: '🌟',
    category: 'streak',
    check: (data) => data.currentStreak >= 30,
  },
  {
    id: 'century-club',
    title: 'Century Club',
    description: 'Maintain a 100-day streak',
    icon: '💎',
    category: 'streak',
    check: (data) => data.currentStreak >= 100,
  },
  {
    id: 'perfect-performer',
    title: 'Perfect Performer',
    description: 'Score 100% on your first perfect quiz',
    icon: '✨',
    category: 'perfection',
    check: (data) => data.perfectQuizzes >= 1,
  },
  {
    id: 'consistency-king',
    title: 'Consistency King',
    description: 'Score 100% on 5 quizzes in a row',
    icon: '🎪',
    category: 'perfection',
    check: (data) => data.consecutivePerfectQuizzes >= 5,
  },
  {
    id: 'flawless',
    title: 'Flawless',
    description: 'Score 100% on 25 quizzes total',
    icon: '🏅',
    category: 'perfection',
    check: (data) => data.perfectQuizzes >= 25,
  },
  {
    id: 'night-owl',
    title: 'Night Owl',
    description: 'Complete 10 quizzes after 10 PM',
    icon: '🌙',
    category: 'special',
    check: (data) => data.quizzesByTimeOfDay.night >= 10,
  },
  {
    id: 'early-bird',
    title: 'Early Bird',
    description: 'Complete 10 quizzes before 8 AM',
    icon: '🌅',
    category: 'special',
    check: (data) => data.quizzesByTimeOfDay.morning >= 10,
  },
  {
    id: 'theme-explorer',
    title: 'Theme Explorer',
    description: 'Switch themes 10 times',
    icon: '🎨',
    category: 'special',
    check: (data) => data.themeChanges >= 10,
  },
];

export function checkForNewAchievements(
  data: GamificationData,
  progress: UserProgress
): AchievementDefinition[] {
  const unlockedIds = new Set(data.achievements.map(a => a.id));
  const newAchievements: AchievementDefinition[] = [];
  
  ACHIEVEMENTS.forEach(achievement => {
    if (!unlockedIds.has(achievement.id) && achievement.check(data, progress)) {
      newAchievements.push(achievement);
    }
  });
  
  return newAchievements;
}
