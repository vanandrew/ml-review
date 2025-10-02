export interface Topic {
  id: string;
  title: string;
  category: string;
  description: string;
  content: string;
  codeExamples?: CodeExample[];
  interviewQuestions?: InterviewQuestion[];
  quizQuestions?: QuizQuestion[];
  hasInteractiveDemo?: boolean;
}

export interface InterviewQuestion {
  question: string;
  answer: string;
}

export interface CodeExample {
  language: string;
  code: string;
  explanation: string;
}

export interface QuizQuestion {
  id: string;
  question: string;
  options: string[];
  correctAnswer: number;
  explanation: string;
}

export interface UserProgress {
  [topicId: string]: TopicProgress;
}

export interface TopicProgress {
  status: 'not_started' | 'reviewing' | 'mastered';
  lastAccessed: Date;
  quizScores: QuizScore[];
  firstCompletion?: Date;
}

export interface QuizScore {
  score: number;
  totalQuestions: number;
  date: Date;
  xpEarned?: number;
}

export interface Category {
  id: string;
  title: string;
  topics: string[];
  color: string;
}

export type ThemeMode = 'light' | 'dark';

// Gamification Types

export interface GamificationData {
  totalXP: number;
  currentStreak: number;
  longestStreak: number;
  lastActivityDate: string | null;
  streakFreezeAvailable: boolean;
  lastStreakFreezeDate: string | null;
  achievements: Achievement[];
  completedTopics: string[];
  perfectQuizzes: number;
  totalQuizzes: number;
  consecutivePerfectQuizzes: number;
  quizzesByTimeOfDay: {
    morning: number; // before 8 AM
    night: number; // after 10 PM
  };
  themeChanges: number;
  dailyXP: number;
  dailyGoal: number;
  lastDailyReset: string | null;
  weeklyChallenge: WeeklyChallenge | null;
  activityHistory: ActivityRecord[];
  // Phase 4: Gems and Challenge Modes
  gems: number;
  lastDailyLoginGems: string | null;
  gemTransactions: GemTransaction[];
  purchasedItems: string[];
  challengeModeStats: ChallengeModeStats;
  selectedTheme: string;
  selectedBadge: string;
  // Endless Challenge Mode
  challengeModeHighScore: number;
}

export interface GemTransaction {
  id: string;
  amount: number;
  type: 'earn' | 'spend';
  reason: string;
  timestamp: Date;
}

export interface ChallengeModeStats {
  normalCompleted: number;
  timedCompleted: number;
  lightningCompleted: number;
  randomMixCompleted: number;
  hardModeCompleted: number;
  perfectRunStreak: number;
  bestPerfectRun: number;
}

export interface WeeklyChallenge {
  id: string;
  title: string;
  description: string;
  target: number;
  progress: number;
  startDate: string;
  endDate: string;
  reward: number; // XP reward
  type: 'master_topics' | 'complete_quizzes' | 'review_topics';
}

export interface ActivityRecord {
  date: string; // ISO date string
  xpEarned: number;
  quizzesTaken: number;
  topicsCompleted: number;
}

export interface Achievement {
  id: string;
  unlockedDate: Date;
}

export interface AchievementDefinition {
  id: string;
  title: string;
  description: string;
  icon: string;
  category: 'learning' | 'streak' | 'perfection' | 'special';
  check: (data: GamificationData, progress: UserProgress) => boolean;
}

export interface XPReward {
  amount: number;
  reason: string;
}