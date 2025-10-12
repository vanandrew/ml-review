import { Trophy, TrendingUp, Target, Zap, BookOpen } from 'lucide-react';
import { GamificationData, UserProgress } from '../types';
import {
  calculateLessonRankPoints,
  getNextLessonRank,
  calculateChallengeRating,
  getNextChallengeRank,
  getRankingPercentile,
  getRankingTips,
  LESSON_RANKS,
  CHALLENGE_RANKS,
} from '../utils/ranking';

interface RankingDisplayProps {
  gamificationData: GamificationData;
  userProgress: UserProgress;
}

export default function RankingDisplay({ gamificationData, userProgress }: RankingDisplayProps) {
  // Calculate lesson ranking
  const lessonPoints = calculateLessonRankPoints(userProgress, gamificationData);
  const lessonRankInfo = getNextLessonRank(lessonPoints);
  const lessonTips = getRankingTips('lesson', gamificationData, userProgress);

  // Calculate challenge ranking
  const challengeRating = calculateChallengeRating(gamificationData, userProgress);
  const challengeRankInfo = getNextChallengeRank(challengeRating);
  const percentile = getRankingPercentile(challengeRating);
  const challengeTips = getRankingTips('challenge', gamificationData, userProgress);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-pink-600 dark:from-purple-700 dark:to-pink-700 rounded-lg p-6 text-white shadow-lg">
        <div className="flex items-center gap-3 mb-2">
          <Trophy className="w-8 h-8" />
          <h2 className="text-2xl font-bold">Your Rankings</h2>
        </div>
        <p className="text-purple-100">Track your progress across learning and challenges</p>
      </div>

      {/* Lesson Ranking */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <BookOpen className="w-6 h-6 text-blue-500" />
            <h3 className="text-xl font-bold text-gray-900 dark:text-white">Learning Journey</h3>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-600 dark:text-gray-400">Points</div>
            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
              {lessonPoints.toLocaleString()}
            </div>
          </div>
        </div>

        {/* Current Rank */}
        <div className="mb-6">
          <div className="flex items-center gap-4 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
            <span className="text-5xl">{lessonRankInfo.currentRank.icon}</span>
            <div className="flex-1">
              <div className={`text-2xl font-bold ${lessonRankInfo.currentRank.color}`}>
                {lessonRankInfo.currentRank.title}
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                {lessonRankInfo.currentRank.description}
              </p>
            </div>
          </div>
        </div>

        {/* Progress to Next Rank */}
        {lessonRankInfo.nextRank && (
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Progress to {lessonRankInfo.nextRank.title}
              </span>
              <span className="text-sm font-bold text-blue-600 dark:text-blue-400">
                {lessonRankInfo.pointsNeeded} points needed
              </span>
            </div>
            <div className="relative h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div
                className="absolute inset-y-0 left-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full transition-all duration-500"
                style={{ width: `${lessonRankInfo.progress}%` }}
              />
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
              {lessonRankInfo.progress}% complete
            </div>
          </div>
        )}

        {/* Tips */}
        {lessonTips.length > 0 && (
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
            <div className="flex items-center gap-2 mb-2">
              <Target className="w-4 h-4 text-blue-600 dark:text-blue-400" />
              <h4 className="font-semibold text-blue-900 dark:text-blue-300">
                How to Rank Up
              </h4>
            </div>
            <ul className="space-y-1">
              {lessonTips.map((tip, idx) => (
                <li key={idx} className="text-sm text-blue-800 dark:text-blue-400">
                  {tip}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Challenge Ranking */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <Zap className="w-6 h-6 text-orange-500" />
            <h3 className="text-xl font-bold text-gray-900 dark:text-white">Challenge Mastery</h3>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-600 dark:text-gray-400">Rating</div>
            <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
              {challengeRating.toLocaleString()}
            </div>
          </div>
        </div>

        {/* Current Rank */}
        <div className="mb-6">
          <div className="flex items-center gap-4 bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-lg p-4 border border-orange-200 dark:border-orange-800">
            <span className="text-5xl">{challengeRankInfo.currentRank.icon}</span>
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <div className={`text-2xl font-bold ${challengeRankInfo.currentRank.color}`}>
                  {challengeRankInfo.currentRank.title}
                </div>
                <div className="px-3 py-1 bg-white dark:bg-gray-700 rounded-full text-sm font-bold border border-gray-300 dark:border-gray-600">
                  {challengeRankInfo.currentRank.badge}
                </div>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                {challengeRankInfo.currentRank.description}
              </p>
              <div className="flex items-center gap-2 mt-2">
                <TrendingUp className="w-4 h-4 text-green-600" />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Top {(100 - percentile).toFixed(1)}% of players
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Progress to Next Rank */}
        {challengeRankInfo.nextRank && (
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Progress to {challengeRankInfo.nextRank.title}
              </span>
              <span className="text-sm font-bold text-orange-600 dark:text-orange-400">
                {challengeRankInfo.ratingNeeded} rating needed
              </span>
            </div>
            <div className="relative h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div
                className="absolute inset-y-0 left-0 bg-gradient-to-r from-orange-500 to-red-500 rounded-full transition-all duration-500"
                style={{ width: `${challengeRankInfo.progress}%` }}
              />
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
              {challengeRankInfo.progress}% complete
            </div>
          </div>
        )}

        {/* Stats Breakdown */}
        <div className="grid grid-cols-2 gap-3 mb-4">
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
            <div className="text-xs text-gray-600 dark:text-gray-400">Challenge Wins</div>
            <div className="text-lg font-bold text-gray-900 dark:text-white">
              {Object.values(gamificationData.challengeModeStats).reduce((a, b) => 
                typeof b === 'number' ? a + b : a, 0
              ) - gamificationData.challengeModeStats.perfectRunStreak - gamificationData.challengeModeStats.bestPerfectRun}
            </div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
            <div className="text-xs text-gray-600 dark:text-gray-400">Perfect Streak</div>
            <div className="text-lg font-bold text-gray-900 dark:text-white">
              {gamificationData.challengeModeStats.perfectRunStreak}
            </div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
            <div className="text-xs text-gray-600 dark:text-gray-400">High Score</div>
            <div className="text-lg font-bold text-gray-900 dark:text-white">
              {gamificationData.challengeModeHighScore}
            </div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
            <div className="text-xs text-gray-600 dark:text-gray-400">Accuracy</div>
            <div className="text-lg font-bold text-gray-900 dark:text-white">
              {gamificationData.totalQuizzes > 0
                ? Math.round((Object.values(userProgress).flatMap(p => p.quizScores || [])
                    .reduce((sum, s) => sum + s.score, 0) /
                    Object.values(userProgress).flatMap(p => p.quizScores || [])
                    .reduce((sum, s) => sum + s.totalQuestions, 0)) * 100)
                : 0}%
            </div>
          </div>
        </div>

        {/* Tips */}
        {challengeTips.length > 0 && (
          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4 border border-orange-200 dark:border-orange-800">
            <div className="flex items-center gap-2 mb-2">
              <Target className="w-4 h-4 text-orange-600 dark:text-orange-400" />
              <h4 className="font-semibold text-orange-900 dark:text-orange-300">
                How to Rank Up
              </h4>
            </div>
            <ul className="space-y-1">
              {challengeTips.map((tip, idx) => (
                <li key={idx} className="text-sm text-orange-800 dark:text-orange-400">
                  {tip}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* All Ranks Reference */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
          🏆 All Ranks
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Lesson Ranks */}
          <div>
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3 flex items-center gap-2">
              <BookOpen className="w-4 h-4" />
              Learning Journey
            </h4>
            <div className="space-y-2">
              {[...LESSON_RANKS].map((rank) => {
                const isCurrentOrPast = lessonPoints >= rank.minPoints;
                return (
                  <div
                    key={rank.id}
                    className={`flex items-center gap-3 p-2 rounded-lg transition-all ${
                      lessonRankInfo.currentRank.id === rank.id
                        ? 'bg-blue-100 dark:bg-blue-900/30 border-2 border-blue-500'
                        : isCurrentOrPast
                        ? 'bg-gray-50 dark:bg-gray-700/50'
                        : 'opacity-50'
                    }`}
                  >
                    <span className="text-2xl">{rank.icon}</span>
                    <div className="flex-1 min-w-0">
                      <div className={`text-sm font-semibold ${rank.color}`}>
                        {rank.title}
                      </div>
                      <div className="text-xs text-gray-600 dark:text-gray-400">
                        {rank.minPoints.toLocaleString()} pts
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Challenge Ranks */}
          <div>
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3 flex items-center gap-2">
              <Zap className="w-4 h-4" />
              Challenge Mastery
            </h4>
            <div className="space-y-2">
              {[...CHALLENGE_RANKS].map((rank) => {
                const isCurrentOrPast = challengeRating >= rank.minRating;
                return (
                  <div
                    key={rank.id}
                    className={`flex items-center gap-3 p-2 rounded-lg transition-all ${
                      challengeRankInfo.currentRank.id === rank.id
                        ? 'bg-orange-100 dark:bg-orange-900/30 border-2 border-orange-500'
                        : isCurrentOrPast
                        ? 'bg-gray-50 dark:bg-gray-700/50'
                        : 'opacity-50'
                    }`}
                  >
                    <span className="text-2xl">{rank.icon}</span>
                    <div className="flex-1 min-w-0">
                      <div className={`text-sm font-semibold ${rank.color}`}>
                        {rank.title}
                      </div>
                      <div className="text-xs text-gray-600 dark:text-gray-400">
                        {rank.minRating.toLocaleString()} rating
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
