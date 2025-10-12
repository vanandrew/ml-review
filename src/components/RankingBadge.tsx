import { Trophy } from 'lucide-react';
import { GamificationData, UserProgress } from '../types';
import {
  calculateLessonRankPoints,
  getLessonRank,
  calculateChallengeRating,
  getChallengeRank,
} from '../utils/ranking';

interface RankingBadgeProps {
  gamificationData: GamificationData;
  userProgress: UserProgress;
  compact?: boolean;
}

export default function RankingBadge({ gamificationData, userProgress, compact = false }: RankingBadgeProps) {
  const lessonPoints = calculateLessonRankPoints(userProgress, gamificationData);
  const lessonRank = getLessonRank(lessonPoints);
  
  const challengeRating = calculateChallengeRating(gamificationData, userProgress);
  const challengeRank = getChallengeRank(challengeRating);

  if (compact) {
    return (
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1 bg-blue-100 dark:bg-blue-900/30 px-2 py-1 rounded-md border border-blue-300 dark:border-blue-700">
          <span className="text-sm">{lessonRank.icon}</span>
          <span className="text-xs font-semibold text-blue-700 dark:text-blue-300">
            {lessonPoints}
          </span>
        </div>
        <div className="flex items-center gap-1 bg-orange-100 dark:bg-orange-900/30 px-2 py-1 rounded-md border border-orange-300 dark:border-orange-700">
          <span className="text-sm">{challengeRank.icon}</span>
          <span className="text-xs font-semibold text-orange-700 dark:text-orange-300">
            {challengeRating}
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
      <div className="flex items-center gap-2 mb-3">
        <Trophy className="w-5 h-5 text-purple-600 dark:text-purple-400" />
        <h3 className="font-bold text-gray-900 dark:text-white">Your Ranks</h3>
      </div>

      <div className="space-y-3">
        {/* Lesson Rank */}
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-3 border border-blue-200 dark:border-blue-800">
          <div className="flex items-center gap-3">
            <span className="text-3xl">{lessonRank.icon}</span>
            <div className="flex-1 min-w-0">
              <div className="text-xs text-gray-600 dark:text-gray-400">Learning Journey</div>
              <div className={`text-sm font-bold ${lessonRank.color} truncate`}>
                {lessonRank.title}
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-400">
                {lessonPoints.toLocaleString()} pts
              </div>
            </div>
          </div>
        </div>

        {/* Challenge Rank */}
        <div className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-lg p-3 border border-orange-200 dark:border-orange-800">
          <div className="flex items-center gap-3">
            <span className="text-3xl">{challengeRank.icon}</span>
            <div className="flex-1 min-w-0">
              <div className="text-xs text-gray-600 dark:text-gray-400">Challenge Mastery</div>
              <div className={`text-sm font-bold ${challengeRank.color} truncate`}>
                {challengeRank.title}
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-400">
                {challengeRating.toLocaleString()} rating
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
