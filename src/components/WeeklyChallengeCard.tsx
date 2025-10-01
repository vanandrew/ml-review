import { Trophy, Calendar } from 'lucide-react';
import { WeeklyChallenge } from '../types';

interface WeeklyChallengeProps {
  challenge: WeeklyChallenge | null;
}

export default function WeeklyChallengeCard({ challenge }: WeeklyChallengeProps) {
  if (!challenge) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
        <p className="text-sm text-gray-500 dark:text-gray-400 text-center">
          No active challenge
        </p>
      </div>
    );
  }

  const progress = Math.min(100, Math.round((challenge.progress / challenge.target) * 100));
  const isComplete = challenge.progress >= challenge.target;
  const daysRemaining = Math.ceil(
    (new Date(challenge.endDate).getTime() - new Date().getTime()) / (1000 * 60 * 60 * 24)
  );

  return (
    <div className="bg-gradient-to-br from-orange-500 to-pink-500 rounded-lg shadow-lg p-4 text-white">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <Trophy className="w-5 h-5" />
          <h3 className="font-semibold">Weekly Challenge</h3>
        </div>
        <div className="flex items-center space-x-1 text-sm opacity-90">
          <Calendar className="w-4 h-4" />
          <span>{daysRemaining}d left</span>
        </div>
      </div>

      <h4 className="text-lg font-bold mb-1">{challenge.title}</h4>
      <p className="text-sm opacity-90 mb-4">{challenge.description}</p>

      {/* Progress Bar */}
      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span>{challenge.progress} / {challenge.target}</span>
          <span>{progress}%</span>
        </div>
        <div className="w-full bg-white bg-opacity-30 rounded-full h-2">
          <div
            className="bg-white rounded-full h-2 transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Reward */}
      <div className="mt-3 flex items-center justify-between text-sm">
        <span className="opacity-90">Reward:</span>
        <span className="font-bold">+{challenge.reward} XP</span>
      </div>

      {isComplete && (
        <div className="mt-3 bg-white bg-opacity-20 rounded-md p-2 text-center">
          <p className="text-sm font-medium">✨ Challenge Complete! ✨</p>
        </div>
      )}
    </div>
  );
}
