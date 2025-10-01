import { QuizQuestion } from '../types';

/**
 * Randomly selects a specified number of questions from a question pool
 * @param questions - The full pool of quiz questions
 * @param count - Number of questions to select (default: 10)
 * @returns Array of randomly selected questions
 */
export function selectRandomQuestions(
  questions: QuizQuestion[],
  count: number = 10
): QuizQuestion[] {
  if (questions.length <= count) {
    // If we have fewer questions than requested, return all in random order
    return shuffleArray([...questions]);
  }

  // Fisher-Yates shuffle algorithm
  const shuffled = shuffleArray([...questions]);
  return shuffled.slice(0, count);
}

/**
 * Shuffles an array using Fisher-Yates algorithm
 * @param array - Array to shuffle
 * @returns Shuffled array
 */
function shuffleArray<T>(array: T[]): T[] {
  const result = [...array];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}
