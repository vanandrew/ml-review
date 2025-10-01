import { useState } from 'react';
import { QuizQuestion, QuizScore } from '../types';
import { CheckCircle, XCircle, RotateCcw, Trophy, Clock } from 'lucide-react';
import { XP_PER_CORRECT_ANSWER, XP_PERFECT_BONUS } from '../utils/gamification';

interface QuizProps {
  questions: QuizQuestion[];
  onComplete: (score: QuizScore) => void;
  onClose: () => void;
}

interface QuizState {
  currentQuestion: number;
  selectedAnswers: Record<number, number>;
  showResults: boolean;
  startTime: Date;
  endTime: Date | null;
}

export default function Quiz({ questions, onComplete, onClose }: QuizProps) {
  const [quizState, setQuizState] = useState<QuizState>({
    currentQuestion: 0,
    selectedAnswers: {},
    showResults: false,
    startTime: new Date(),
    endTime: null
  });

  const [showExplanation, setShowExplanation] = useState(false);

  const currentQ = questions[quizState.currentQuestion];
  const isLastQuestion = quizState.currentQuestion === questions.length - 1;
  const hasAnswered = quizState.selectedAnswers[quizState.currentQuestion] !== undefined;

  const handleAnswerSelect = (answerIndex: number) => {
    setQuizState(prev => ({
      ...prev,
      selectedAnswers: {
        ...prev.selectedAnswers,
        [prev.currentQuestion]: answerIndex
      }
    }));
    setShowExplanation(false);
  };

  const handleNext = () => {
    if (isLastQuestion) {
      // Complete quiz
      const endTime = new Date();
      const score = calculateScore();
      
      // Calculate XP earned
      const xpEarned = (score * XP_PER_CORRECT_ANSWER) + (score === questions.length ? XP_PERFECT_BONUS : 0);
      
      const quizScore: QuizScore = {
        score,
        totalQuestions: questions.length,
        date: endTime,
        xpEarned
      };

      setQuizState(prev => ({
        ...prev,
        showResults: true,
        endTime
      }));

      onComplete(quizScore);
    } else {
      // Next question
      setQuizState(prev => ({
        ...prev,
        currentQuestion: prev.currentQuestion + 1
      }));
      setShowExplanation(false);
    }
  };

  const handlePrevious = () => {
    if (quizState.currentQuestion > 0) {
      setQuizState(prev => ({
        ...prev,
        currentQuestion: prev.currentQuestion - 1
      }));
      setShowExplanation(false);
    }
  };

  const calculateScore = () => {
    let correct = 0;
    questions.forEach((question, index) => {
      if (quizState.selectedAnswers[index] === question.correctAnswer) {
        correct++;
      }
    });
    return correct;
  };

  const getScoreColor = (percentage: number) => {
    if (percentage >= 80) return 'text-green-600 dark:text-green-400';
    if (percentage >= 60) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const restartQuiz = () => {
    setQuizState({
      currentQuestion: 0,
      selectedAnswers: {},
      showResults: false,
      startTime: new Date(),
      endTime: null
    });
    setShowExplanation(false);
  };

  if (quizState.showResults) {
    const score = calculateScore();
    const percentage = Math.round((score / questions.length) * 100);
    const duration = quizState.endTime
      ? Math.round((quizState.endTime.getTime() - quizState.startTime.getTime()) / 1000)
      : 0;

    return (
      <div className="max-w-2xl mx-auto">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 p-8">
          <div className="text-center">
            <div className="w-16 h-16 mx-auto mb-4 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center">
              <Trophy className="w-8 h-8 text-blue-600 dark:text-blue-400" />
            </div>

            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
              Quiz Complete!
            </h2>

            <div className="mb-6">
              <div className={`text-4xl font-bold mb-2 ${getScoreColor(percentage)}`}>
                {percentage}%
              </div>
              <p className="text-gray-600 dark:text-gray-400">
                You scored {score} out of {questions.length} questions correct
              </p>
              <div className="flex items-center justify-center mt-2 text-sm text-gray-500 dark:text-gray-400">
                <Clock className="w-4 h-4 mr-1" />
                <span>Completed in {duration} seconds</span>
              </div>
            </div>

            <div className="space-y-4 mb-8">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Review Your Answers
              </h3>

              <div className="space-y-3">
                {questions.map((question, index) => {
                  const userAnswer = quizState.selectedAnswers[index];
                  const isCorrect = userAnswer === question.correctAnswer;

                  return (
                    <div key={index} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                      <div className="flex items-center space-x-3">
                        {isCorrect ? (
                          <CheckCircle className="w-5 h-5 text-green-500" />
                        ) : (
                          <XCircle className="w-5 h-5 text-red-500" />
                        )}
                        <span className="text-sm text-gray-700 dark:text-gray-300">
                          Question {index + 1}
                        </span>
                      </div>
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        {isCorrect ? 'Correct' : `Your answer: ${question.options[userAnswer]} | Correct: ${question.options[question.correctAnswer]}`}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="flex space-x-3">
              <button
                onClick={restartQuiz}
                className="flex-1 px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors flex items-center justify-center space-x-2"
              >
                <RotateCcw className="w-4 h-4" />
                <span>Retake Quiz</span>
              </button>
              <button
                onClick={onClose}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Back to Topic
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
        {/* Quiz Header */}
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                Practice Quiz
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Question {quizState.currentQuestion + 1} of {questions.length}
              </p>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              ✕
            </button>
          </div>

          {/* Progress Bar */}
          <div className="mt-4">
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{
                  width: `${((quizState.currentQuestion + 1) / questions.length) * 100}%`
                }}
              />
            </div>
          </div>
        </div>

        {/* Question Content */}
        <div className="p-6">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-6">
            {currentQ.question}
          </h3>

          <div className="space-y-3">
            {currentQ.options.map((option, index) => {
              const isSelected = quizState.selectedAnswers[quizState.currentQuestion] === index;
              const isCorrect = index === currentQ.correctAnswer;
              const userAnswer = quizState.selectedAnswers[quizState.currentQuestion];
              const showCorrectness = showExplanation && hasAnswered;

              let buttonClass = 'w-full p-4 text-left border rounded-lg transition-colors ';

              if (showCorrectness) {
                if (isCorrect) {
                  buttonClass += 'border-green-500 bg-green-50 dark:bg-green-900/20 text-green-900 dark:text-green-100';
                } else if (isSelected && !isCorrect) {
                  buttonClass += 'border-red-500 bg-red-50 dark:bg-red-900/20 text-red-900 dark:text-red-100';
                } else {
                  buttonClass += 'border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300';
                }
              } else if (isSelected) {
                buttonClass += 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 text-blue-900 dark:text-blue-100';
              } else {
                buttonClass += 'border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:border-blue-300 dark:hover:border-blue-500';
              }

              return (
                <button
                  key={index}
                  onClick={() => handleAnswerSelect(index)}
                  disabled={showCorrectness}
                  className={buttonClass}
                >
                  <div className="flex items-center space-x-3">
                    <span className="flex-shrink-0 w-6 h-6 rounded-full border border-current flex items-center justify-center text-sm font-medium">
                      {String.fromCharCode(65 + index)}
                    </span>
                    <span>{option}</span>
                    {showCorrectness && isCorrect && (
                      <CheckCircle className="ml-auto w-5 h-5 text-green-500" />
                    )}
                    {showCorrectness && isSelected && !isCorrect && (
                      <XCircle className="ml-auto w-5 h-5 text-red-500" />
                    )}
                  </div>
                </button>
              );
            })}
          </div>

          {/* Explanation */}
          {showExplanation && hasAnswered && (
            <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-2">
                Explanation
              </h4>
              <p className="text-blue-800 dark:text-blue-200 text-sm">
                {currentQ.explanation}
              </p>
            </div>
          )}
        </div>

        {/* Quiz Footer */}
        <div className="p-6 border-t border-gray-200 dark:border-gray-700">
          <div className="flex justify-between">
            <button
              onClick={handlePrevious}
              disabled={quizState.currentQuestion === 0}
              className="px-4 py-2 text-gray-600 dark:text-gray-400 disabled:opacity-50 disabled:cursor-not-allowed hover:text-gray-800 dark:hover:text-gray-200"
            >
              Previous
            </button>

            <div className="flex space-x-3">
              {hasAnswered && !showExplanation && (
                <button
                  onClick={() => setShowExplanation(true)}
                  className="px-4 py-2 text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-200"
                >
                  Show Explanation
                </button>
              )}

              <button
                onClick={handleNext}
                disabled={!hasAnswered}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isLastQuestion ? 'Finish Quiz' : 'Next'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}