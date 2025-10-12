import { UserProgress } from '../types';
import { getDecayStatistics, calculateDaysUntilDecay } from '../utils/decaySystem';
import { Clock, TrendingDown, AlertTriangle, Award } from 'lucide-react';

interface DecayWarningProps {
  userProgress: UserProgress;
  onSelectTopic?: (topicId: string) => void;
  getTopicTitle?: (topicId: string) => string;
}

export default function DecayWarning({ userProgress, onSelectTopic, getTopicTitle }: DecayWarningProps) {
  const stats = getDecayStatistics(userProgress);
  
  // Get topics that are decaying soon
  const decayingSoonTopics = Object.entries(userProgress)
    .filter(([_, progress]) => {
      if (progress.status !== 'mastered') return false;
      const daysUntil = calculateDaysUntilDecay(progress);
      return daysUntil !== null && daysUntil > 0 && daysUntil <= 7;
    })
    .map(([topicId, progress]) => ({
      topicId,
      daysUntil: calculateDaysUntilDecay(progress) || 0,
      masteryStrength: progress.masteryStrength || 0,
    }))
    .sort((a, b) => a.daysUntil - b.daysUntil)
    .slice(0, 5);

  // Don't show if no mastered topics
  if (stats.totalMastered === 0) return null;

  const hasWarnings = stats.needsReview > 0 || stats.decayingSoon > 0;

  return (
    <div className={`rounded-lg shadow-sm border p-6 ${
      hasWarnings 
        ? 'bg-orange-50 dark:bg-orange-900/10 border-orange-200 dark:border-orange-800'
        : 'bg-blue-50 dark:bg-blue-900/10 border-blue-200 dark:border-blue-800'
    }`}>
      <div className="flex items-start gap-4">
        <div className={`flex-shrink-0 p-3 rounded-full ${
          hasWarnings
            ? 'bg-orange-100 dark:bg-orange-900/30'
            : 'bg-blue-100 dark:bg-blue-900/30'
        }`}>
          {hasWarnings ? (
            <AlertTriangle className="w-6 h-6 text-orange-600 dark:text-orange-400" />
          ) : (
            <Award className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          )}
        </div>

        <div className="flex-1">
          <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">
            {hasWarnings ? '⚠️ Mastery Maintenance' : '✨ Mastery Status'}
          </h3>

          <div className="space-y-3">
            {/* Statistics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-1">
                  <Award className="w-4 h-4 text-green-600 dark:text-green-400" />
                  <span className="text-xs text-gray-600 dark:text-gray-400">Mastered</span>
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {stats.totalMastered}
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-1">
                  <Clock className="w-4 h-4 text-yellow-600 dark:text-yellow-400" />
                  <span className="text-xs text-gray-600 dark:text-gray-400">Decaying Soon</span>
                </div>
                <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">
                  {stats.decayingSoon}
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-1">
                  <TrendingDown className="w-4 h-4 text-red-600 dark:text-red-400" />
                  <span className="text-xs text-gray-600 dark:text-gray-400">Needs Review</span>
                </div>
                <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                  {stats.needsReview}
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-1">
                  <Award className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                  <span className="text-xs text-gray-600 dark:text-gray-400">Avg. Strength</span>
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {stats.averageMasteryStrength}
                </div>
              </div>
            </div>

            {/* Warning Message */}
            {hasWarnings && (
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  {stats.needsReview > 0 && (
                    <span className="block mb-1">
                      <strong className="text-red-600 dark:text-red-400">{stats.needsReview}</strong> mastered {stats.needsReview === 1 ? 'topic has' : 'topics have'} decayed and need review.
                    </span>
                  )}
                  {stats.decayingSoon > 0 && (
                    <span className="block">
                      <strong className="text-yellow-600 dark:text-yellow-400">{stats.decayingSoon}</strong> {stats.decayingSoon === 1 ? 'topic is' : 'topics are'} decaying soon (within 7 days).
                    </span>
                  )}
                </p>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                  💡 Keep practicing to maintain your mastery! Topics with higher mastery strength decay more slowly.
                </p>
              </div>
            )}

            {/* Decaying Soon Topics */}
            {decayingSoonTopics.length > 0 && onSelectTopic && getTopicTitle && (
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">
                  Topics Decaying Soon
                </h4>
                <div className="space-y-2">
                  {decayingSoonTopics.map(({ topicId, daysUntil, masteryStrength }) => (
                    <button
                      key={topicId}
                      onClick={() => onSelectTopic(topicId)}
                      className="w-full flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
                    >
                      <div className="flex-1 text-left">
                        <div className="font-medium text-gray-900 dark:text-white text-sm">
                          {getTopicTitle(topicId)}
                        </div>
                        <div className="text-xs text-gray-600 dark:text-gray-400 mt-1 flex items-center gap-2">
                          <span>Strength: {masteryStrength}/100</span>
                        </div>
                      </div>
                      <div className="flex-shrink-0 text-right">
                        <div className={`text-sm font-semibold ${
                          daysUntil <= 3 
                            ? 'text-red-600 dark:text-red-400'
                            : 'text-yellow-600 dark:text-yellow-400'
                        }`}>
                          {daysUntil} {daysUntil === 1 ? 'day' : 'days'}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          until decay
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {!hasWarnings && (
              <p className="text-sm text-gray-600 dark:text-gray-400">
                🎉 Great job! All your mastered topics are in good shape. Keep up the excellent work!
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
