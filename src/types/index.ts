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
  masteryStrength?: number; // 0-100, based on consistency of high scores
  lastMasteredDate?: Date; // When topic was last marked as mastered
  highScoreStreak?: number; // Consecutive quizzes with 90%+ scores
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
  // Phase 5: Consumables & Power-ups
  consumableInventory: ConsumableInventory;
  activePowerUps: ActivePowerUp[];
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

export interface ConsumableInventory {
  hints: number;
  streakFreezes: number;
  xpBoosts: number; // Number of quizzes with 2x XP remaining
  knowledgePotions: number; // Number of questions with elimination remaining
  timeExtensions: number;
  secondChances: number;
  extraLives: number;
  multiplierBoosts: number;
}

export interface ActivePowerUp {
  id: string;
  type: 'double-gems' | 'premium-week' | 'scholars-blessing' | 'xp-boost';
  activatedAt: Date;
  expiresAt: Date;
  remaining?: number; // For counted items like xp-boost (remaining quizzes)
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

// Authentication Types

export interface User {
  uid: string;
  email: string | null;
  displayName: string | null;
  photoURL: string | null;
  emailVerified: boolean;
}

export interface AuthState {
  user: User | null;
  loading: boolean;
  error: string | null;
}

export interface FirestoreUserData {
  uid: string;
  email: string | null;
  displayName: string | null;
  photoURL: string | null;
  createdAt: Date;
  lastLoginAt: Date;
  progress: UserProgress;
  gamification: GamificationData;
}

export interface SyncStatus {
  isSyncing: boolean;
  lastSyncTime: Date | null;
  syncError: string | null;
}
