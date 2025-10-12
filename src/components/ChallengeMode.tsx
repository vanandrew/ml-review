import { useState, useEffect } from 'react';
import { QuizQuestion, ConsumableInventory } from '../types';
import { Trophy, Zap, X, Check, Award } from 'lucide-react';

interface ChallengeModeProps {
  allQuestions: QuizQuestion[];
  highScore: number;
  onComplete: (score: number) => void;
  onExit: () => void;
  consumableInventory?: ConsumableInventory;
  onUseConsumable?: (itemType: keyof ConsumableInventory) => void;
}

export default function ChallengeMode({ allQuestions, highScore, onComplete, onExit, consumableInventory, onUseConsumable }: ChallengeModeProps) {
  const [remainingQuestions, setRemainingQuestions] = useState<QuizQuestion[]>([]);
  const [currentQuestion, setCurrentQuestion] = useState<QuizQuestion | null>(null);
  const [currentScore, setCurrentScore] = useState(0);
  // TODO: Implement extra life feature
  // const [showExtraLifePrompt, setShowExtraLifePrompt] = useState(false);
  const [showExtraLifePrompt, setShowExtraLifePrompt] = useState(false);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null);
  const [gameOver, setGameOver] = useState(false);
  const [showResult, setShowResult] = useState(false);

  // Initialize and shuffle questions
  useEffect(() => {
    const shuffled = [...allQuestions].sort(() => Math.random() - 0.5);
    setRemainingQuestions(shuffled);
    setCurrentQuestion(shuffled[0] || null);
  }, [allQuestions]);

  const handleAnswerSelect = (answerIndex: number) => {
    if (selectedAnswer !== null || gameOver) return;

    setSelectedAnswer(answerIndex);
    const correct = answerIndex === currentQuestion?.correctAnswer;
    setIsCorrect(correct);
    setShowResult(true);

    if (correct) {
      // Correct answer - continue
      setTimeout(() => {
        const newScore = currentScore + 1;
        setCurrentScore(newScore);
        
        // Remove current question from pool
        const newRemaining = remainingQuestions.slice(1);
        
        if (newRemaining.length === 0) {
          // All questions answered correctly!
          setGameOver(true);
          onComplete(newScore);
        } else {
          // Move to next question
          setCurrentQuestion(newRemaining[0]);
          setRemainingQuestions(newRemaining);
          setSelectedAnswer(null);
          setIsCorrect(null);
          setShowResult(false);
        }
      }, 1500);
    } else {
      // Wrong answer - check for extra life
      if (consumableInventory && consumableInventory.extraLives > 0 && !showExtraLifePrompt) {
        // Show extra life prompt
        setShowExtraLifePrompt(true);
      } else {
        // Game over
        setTimeout(() => {
          setGameOver(true);
          onComplete(currentScore);
        }, 1500);
      }
    }
  };

  const handleExit = () => {
    if (gameOver || window.confirm('Are you sure you want to exit? Your progress will be lost.')) {
      onExit();
    }
  };

  const handleUseExtraLife = () => {
    if (onUseConsumable) {
      onUseConsumable('extraLives');
      setShowExtraLifePrompt(false);
      // Reset for next question
      const newRemaining = remainingQuestions.slice(1);
      if (newRemaining.length > 0) {
        setCurrentQuestion(newRemaining[0]);
        setRemainingQuestions(newRemaining);
      } else {
        setGameOver(true);
        onComplete(currentScore);
      }
      setSelectedAnswer(null);
      setIsCorrect(null);
      setShowResult(false);
    }
  };

  const handleDeclineExtraLife = () => {
    setShowExtraLifePrompt(false);
    setTimeout(() => {
      setGameOver(true);
      onComplete(currentScore);
    }, 500);
  };

  if (!currentQuestion) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-8 text-center">
        <Zap className="w-16 h-16 text-yellow-500 mx-auto mb-4" />
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
          Challenge Mode
        </h2>
        <p className="text-gray-600 dark:text-gray-400">
          No questions available. Master some topics first!
        </p>
        <button
          onClick={onExit}
          className="mt-6 px-6 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
        >
          Go Back
        </button>
      </div>
    );
  }

  if (gameOver) {
    const isNewHighScore = currentScore > highScore;
    const isPerfect = currentScore === allQuestions.length;

    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-8">
        <div className="text-center">
          {isPerfect ? (
            <>
              <Trophy className="w-20 h-20 text-yellow-500 mx-auto mb-4 animate-bounce" />
              <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                🎉 Perfect Run! 🎉
              </h2>
              <p className="text-xl text-gray-600 dark:text-gray-400 mb-6">
                You answered all {currentScore} questions correctly!
              </p>
            </>
          ) : (
            <>
              <Award className="w-20 h-20 text-blue-500 mx-auto mb-4" />
              <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                Challenge Complete!
              </h2>
              <p className="text-xl text-gray-600 dark:text-gray-400 mb-6">
                You answered {currentScore} question{currentScore !== 1 ? 's' : ''} correctly
              </p>
            </>
          )}

          {isNewHighScore && (
            <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-lg p-4 mb-6 border-2 border-yellow-400 dark:border-yellow-600">
              <p className="text-lg font-bold text-yellow-900 dark:text-yellow-300">
                🏆 New High Score!
              </p>
              <p className="text-yellow-800 dark:text-yellow-400 text-sm">
                Previous best: {highScore}
              </p>
            </div>
          )}

          <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-6 mb-6">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Your Score</p>
                <p className="text-3xl font-bold text-blue-600 dark:text-blue-400">
                  {currentScore}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">High Score</p>
                <p className="text-3xl font-bold text-purple-600 dark:text-purple-400">
                  {Math.max(currentScore, highScore)}
                </p>
              </div>
            </div>
          </div>

          <div className="flex gap-3 justify-center">
            <button
              onClick={() => window.location.reload()}
              className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-medium"
            >
              Try Again
            </button>
            <button
              onClick={onExit}
              className="px-6 py-3 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors font-medium"
            >
              Exit
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <>
      {/* Extra Life Prompt Modal */}
      {showExtraLifePrompt && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl border-2 border-red-500 p-8 max-w-md w-full">
            <div className="text-center">
              <div className="w-20 h-20 bg-red-100 dark:bg-red-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
                <X className="w-12 h-12 text-red-500" />
              </div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                Wrong Answer!
              </h2>
              <p className="text-gray-600 dark:text-gray-400 mb-6">
                You have {consumableInventory?.extraLives} extra {consumableInventory?.extraLives === 1 ? 'life' : 'lives'} available.
                Use one to continue?
              </p>
              
              <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-300 dark:border-yellow-700 rounded-lg p-4 mb-6">
                <p className="text-sm text-yellow-800 dark:text-yellow-300">
                  ❤️ Using an extra life will skip this question and let you continue your run!
                </p>
              </div>

              <div className="flex gap-3">
                <button
                  onClick={handleUseExtraLife}
                  className="flex-1 px-6 py-3 bg-gradient-to-r from-red-500 to-pink-500 text-white rounded-lg hover:from-red-600 hover:to-pink-600 transition-all font-bold flex items-center justify-center gap-2"
                >
                  ❤️ Use Extra Life
                </button>
                <button
                  onClick={handleDeclineExtraLife}
                  className="flex-1 px-6 py-3 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors font-medium"
                >
                  End Run
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
      {/* Header */}
      <div className="border-b border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-r from-yellow-400 to-orange-500 rounded-lg p-2">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                Challenge Mode
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                One wrong answer ends the run!
              </p>
            </div>
          </div>
          <button
            onClick={handleExit}
            className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-3 text-center border border-green-200 dark:border-green-800">
            <p className="text-sm text-green-700 dark:text-green-400 mb-1">Current Streak</p>
            <p className="text-2xl font-bold text-green-600 dark:text-green-300">
              {currentScore}
            </p>
          </div>
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3 text-center border border-purple-200 dark:border-purple-800">
            <p className="text-sm text-purple-700 dark:text-purple-400 mb-1">High Score</p>
            <p className="text-2xl font-bold text-purple-600 dark:text-purple-300">
              {highScore}
            </p>
          </div>
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3 text-center border border-blue-200 dark:border-blue-800">
            <p className="text-sm text-blue-700 dark:text-blue-400 mb-1">Remaining</p>
            <p className="text-2xl font-bold text-blue-600 dark:text-blue-300">
              {remainingQuestions.length}
            </p>
          </div>
        </div>
      </div>

      {/* Question */}
      <div className="p-6">
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            {currentQuestion.question}
          </h3>

          <div className="space-y-3">
            {currentQuestion.options.map((option, index) => {
              const isSelected = selectedAnswer === index;
              const isCorrectAnswer = index === currentQuestion.correctAnswer;
              const showCorrect = showResult && isCorrectAnswer;
              const showWrong = showResult && isSelected && !isCorrect;

              return (
                <button
                  key={index}
                  onClick={() => handleAnswerSelect(index)}
                  disabled={selectedAnswer !== null}
                  className={`
                    w-full text-left p-4 rounded-lg border-2 transition-all
                    ${showCorrect
                      ? 'bg-green-50 dark:bg-green-900/20 border-green-500 dark:border-green-600'
                      : showWrong
                      ? 'bg-red-50 dark:bg-red-900/20 border-red-500 dark:border-red-600'
                      : isSelected
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
                    }
                    ${selectedAnswer !== null ? 'cursor-not-allowed' : 'cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700/50'}
                  `}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-gray-900 dark:text-white">{option}</span>
                    {showCorrect && (
                      <Check className="w-5 h-5 text-green-600 dark:text-green-400" />
                    )}
                    {showWrong && (
                      <X className="w-5 h-5 text-red-600 dark:text-red-400" />
                    )}
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {showResult && (
          <div className={`p-4 rounded-lg border-2 ${
            isCorrect
              ? 'bg-green-50 dark:bg-green-900/20 border-green-500 dark:border-green-600 text-green-900 dark:text-green-300'
              : 'bg-red-50 dark:bg-red-900/20 border-red-500 dark:border-red-600 text-red-900 dark:text-red-300'
          }`}>
            <p className="font-medium">
              {isCorrect ? '✓ Correct!' : '✗ Wrong! Game Over'}
            </p>
            {currentQuestion.explanation && (
              <p className="text-sm mt-2 opacity-90">
                {currentQuestion.explanation}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
    </>
  );
}
