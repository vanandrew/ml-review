export interface MotivationalQuote {
  text: string;
  author?: string;
  category: 'learning' | 'ml' | 'perseverance' | 'success';
}

export const MOTIVATIONAL_QUOTES: MotivationalQuote[] = [
  // Learning quotes
  {
    text: "The capacity to learn is a gift; the ability to learn is a skill; the willingness to learn is a choice.",
    author: "Brian Herbert",
    category: "learning"
  },
  {
    text: "Live as if you were to die tomorrow. Learn as if you were to live forever.",
    author: "Mahatma Gandhi",
    category: "learning"
  },
  {
    text: "The beautiful thing about learning is that no one can take it away from you.",
    author: "B.B. King",
    category: "learning"
  },
  {
    text: "Education is not the filling of a pail, but the lighting of a fire.",
    author: "William Butler Yeats",
    category: "learning"
  },
  
  // ML/AI quotes
  {
    text: "Artificial Intelligence is the new electricity.",
    author: "Andrew Ng",
    category: "ml"
  },
  {
    text: "Machine learning is the last invention that humanity will ever need to make.",
    author: "Nick Bostrom",
    category: "ml"
  },
  {
    text: "The question of whether a computer can think is no more interesting than the question of whether a submarine can swim.",
    author: "Edsger W. Dijkstra",
    category: "ml"
  },
  {
    text: "AI is not going to replace humans, but humans with AI are going to replace humans without AI.",
    author: "Karim Lakhani",
    category: "ml"
  },
  {
    text: "Data is the new oil.",
    author: "Clive Humby",
    category: "ml"
  },
  
  // Perseverance quotes
  {
    text: "Success is not final, failure is not fatal: it is the courage to continue that counts.",
    author: "Winston Churchill",
    category: "perseverance"
  },
  {
    text: "The expert in anything was once a beginner.",
    author: "Helen Hayes",
    category: "perseverance"
  },
  {
    text: "Don't watch the clock; do what it does. Keep going.",
    author: "Sam Levenson",
    category: "perseverance"
  },
  {
    text: "It always seems impossible until it's done.",
    author: "Nelson Mandela",
    category: "perseverance"
  },
  {
    text: "The only way to do great work is to love what you do.",
    author: "Steve Jobs",
    category: "perseverance"
  },
  
  // Success quotes
  {
    text: "Success is the sum of small efforts repeated day in and day out.",
    author: "Robert Collier",
    category: "success"
  },
  {
    text: "Your time is limited, don't waste it living someone else's life.",
    author: "Steve Jobs",
    category: "success"
  },
  {
    text: "The way to get started is to quit talking and begin doing.",
    author: "Walt Disney",
    category: "success"
  },
  {
    text: "Believe you can and you're halfway there.",
    author: "Theodore Roosevelt",
    category: "success"
  },
];

/**
 * Get a random motivational quote
 */
export function getRandomQuote(): MotivationalQuote {
  return MOTIVATIONAL_QUOTES[Math.floor(Math.random() * MOTIVATIONAL_QUOTES.length)];
}

/**
 * Get a quote by category
 */
export function getQuoteByCategory(category: MotivationalQuote['category']): MotivationalQuote {
  const categoryQuotes = MOTIVATIONAL_QUOTES.filter(q => q.category === category);
  return categoryQuotes[Math.floor(Math.random() * categoryQuotes.length)];
}

/**
 * Get a daily quote (consistent for the same day)
 */
export function getDailyQuote(): MotivationalQuote {
  const today = new Date();
  const dayOfYear = Math.floor((today.getTime() - new Date(today.getFullYear(), 0, 0).getTime()) / (1000 * 60 * 60 * 24));
  const index = dayOfYear % MOTIVATIONAL_QUOTES.length;
  return MOTIVATIONAL_QUOTES[index];
}

/**
 * Get encouraging messages based on progress
 */
export function getEncouragingMessage(context: {
  currentStreak: number;
  totalXP: number;
  topicsMastered: number;
  quizzesCompleted: number;
  perfectScores: number;
}): string {
  const messages: string[] = [];
  
  // Streak messages
  if (context.currentStreak === 0) {
    messages.push("Start your learning streak today! 🔥");
  } else if (context.currentStreak === 1) {
    messages.push("Great start! Come back tomorrow to build your streak! 💪");
  } else if (context.currentStreak >= 7) {
    messages.push(`Amazing ${context.currentStreak}-day streak! You're on fire! 🔥`);
  } else if (context.currentStreak >= 30) {
    messages.push(`Incredible ${context.currentStreak}-day streak! You're a learning machine! 🚀`);
  } else {
    messages.push(`${context.currentStreak}-day streak! Keep it going! ⭐`);
  }
  
  // XP messages
  if (context.totalXP >= 1000) {
    messages.push(`You've earned ${context.totalXP.toLocaleString()} XP! Impressive! 🎯`);
  }
  
  // Topics mastered
  if (context.topicsMastered === 0) {
    messages.push("Master your first topic by scoring 100% on its quiz! 🎓");
  } else if (context.topicsMastered === 1) {
    messages.push("You've mastered your first topic! Many more to go! 📚");
  } else if (context.topicsMastered >= 10) {
    messages.push(`${context.topicsMastered} topics mastered! You're becoming an ML expert! 🧠`);
  }
  
  // Perfect scores
  if (context.perfectScores >= 5) {
    messages.push(`${context.perfectScores} perfect scores! You're a quiz champion! 🏆`);
  }
  
  // Quizzes completed
  if (context.quizzesCompleted >= 50) {
    messages.push(`${context.quizzesCompleted} quizzes completed! Your dedication is inspiring! ✨`);
  }
  
  // Return a random encouraging message
  return messages[Math.floor(Math.random() * messages.length)];
}

/**
 * Get streak reminder message
 */
export function getStreakReminder(hourOfDay: number, hasActivityToday: boolean): string | null {
  // Only show reminders after 6 PM if no activity today
  if (hourOfDay < 18 || hasActivityToday) {
    return null;
  }
  
  if (hourOfDay >= 22) {
    return "⏰ Don't break your streak! Complete a quick quiz before midnight! 🔥";
  } else if (hourOfDay >= 20) {
    return "🌙 Evening reminder: Keep your streak alive with a quick review! 💪";
  } else {
    return "✨ You haven't practiced today. A few minutes now will keep your streak going! 🎯";
  }
}
