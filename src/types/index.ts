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
}

export interface QuizScore {
  score: number;
  totalQuestions: number;
  date: Date;
}

export interface Category {
  id: string;
  title: string;
  topics: string[];
  color: string;
}

export type ThemeMode = 'light' | 'dark';