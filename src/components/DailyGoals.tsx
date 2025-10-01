import { Target, CheckCircle } from 'lucide-react';
import { GamificationData } from '../types';

interface DailyGoalsProps {
  gamificationData: GamificationData;
  onSetGoal: (goal: number) => void;
}

export default function DailyGoals({ gamificationData, onSetGoal }: DailyGoalsProps) {
  const { dailyXP, dailyGoal } = gamificationData;
  const progress = Math.min(100, Math.round((dailyXP / dailyGoal) * 100));
  const isComplete = dailyXP >= dailyGoal;

  const goalOptions = [10, 20, 50, 100];

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <Target className={`w-5 h-5 ${isComplete ? 'text-green-500' : 'text-blue-500'}`} />
          <h3 className="font-semibold text-gray-900 dark:text-white">Daily Goal</h3>
        </div>
        {isComplete && (
          <div className="flex items-center space-x-1 text-green-500">
            <CheckCircle className="w-5 h-5" />
            <span className="text-sm font-medium">Complete!</span>
          </div>
        )}
      </div>

      {/* Progress Circle */}
      <div className="flex items-center justify-center mb-4">
        <div className="relative w-32 h-32">
          <svg className="w-32 h-32 transform -rotate-90">
            <circle
              cx="64"
              cy="64"
              r="56"
              stroke="currentColor"
              strokeWidth="8"
              fill="none"
              className="text-gray-200 dark:text-gray-700"
            />
            <circle
              cx="64"
              cy="64"
              r="56"
              stroke="currentColor"
              strokeWidth="8"
              fill="none"
              strokeDasharray={`${2 * Math.PI * 56}`}
              strokeDashoffset={`${2 * Math.PI * 56 * (1 - progress / 100)}`}
              className={isComplete ? 'text-green-500' : 'text-blue-500'}
              strokeLinecap="round"
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {dailyXP}
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400">
              / {dailyGoal} XP
            </div>
          </div>
        </div>
      </div>

      {/* Goal Options */}
      <div className="space-y-2">
        <p className="text-xs text-gray-600 dark:text-gray-400 text-center mb-2">
          Set your daily XP goal:
        </p>
        <div className="grid grid-cols-4 gap-2">
          {goalOptions.map((goal) => (
            <button
              key={goal}
              onClick={() => onSetGoal(goal)}
              className={`py-2 px-3 rounded-md text-sm font-medium transition-colors ${
                dailyGoal === goal
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              {goal}
            </button>
          ))}
        </div>
      </div>

      {/* Progress Text */}
      <div className="mt-3 text-center">
        <p className="text-sm text-gray-600 dark:text-gray-400">
          {isComplete ? (
            <span className="text-green-600 dark:text-green-400 font-medium">
              🎉 Goal achieved! Keep going!
            </span>
          ) : (
            <span>
              {dailyGoal - dailyXP} XP remaining today
            </span>
          )}
        </p>
      </div>
    </div>
  );
}
