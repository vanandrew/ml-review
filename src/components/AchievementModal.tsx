import { AchievementDefinition } from '../types';

interface AchievementModalProps {
  achievement: AchievementDefinition;
  onClose: () => void;
}

export default function AchievementModal({ achievement, onClose }: AchievementModalProps) {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-2xl max-w-md w-full p-8">
        <div className="text-center">
          <div className="text-6xl mb-4">{achievement.icon}</div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
            Achievement Unlocked!
          </h2>
          <h3 className="text-xl font-semibold text-purple-600 dark:text-purple-400 mb-3">
            {achievement.title}
          </h3>
          <p className="text-gray-600 dark:text-gray-300 mb-6">
            {achievement.description}
          </p>
          <button
            onClick={onClose}
            className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
          >
            Awesome!
          </button>
        </div>
      </div>
    </div>
  );
}
