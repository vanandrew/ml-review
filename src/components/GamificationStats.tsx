import { Trophy, Flame, Star } from 'lucide-react';
import { calculateLevel, getXPProgress } from '../utils/gamification';
import { GamificationData } from '../types';

interface GamificationStatsProps {
  gamificationData: GamificationData;
}

const THEME_GRADIENTS: Record<string, string> = {
  'default': 'from-purple-500 to-pink-500',
  'theme-ocean': 'from-blue-400 to-cyan-500',
  'theme-forest': 'from-green-400 to-emerald-600',
  'theme-sunset': 'from-orange-400 to-pink-500',
};

export default function GamificationStats({ gamificationData }: GamificationStatsProps) {
  const level = calculateLevel(gamificationData.totalXP);
  const progress = getXPProgress(gamificationData.totalXP);
  const gradientColors = THEME_GRADIENTS[gamificationData.selectedTheme] || THEME_GRADIENTS['default'];

  return (
    <div className={`p-4 bg-gradient-to-r ${gradientColors} text-white rounded-lg shadow-lg`}>
      {/* Level Display */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          {gamificationData.selectedBadge && (
            <span className="text-xl">{gamificationData.selectedBadge}</span>
          )}
          <Trophy className="w-5 h-5" />
          <span className="font-semibold">Level {level}</span>
        </div>
        <div className="text-sm opacity-90">
          {gamificationData.totalXP} XP
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mb-3">
        <div className="w-full bg-white bg-opacity-30 rounded-full h-2">
          <div
            className="bg-white rounded-full h-2 transition-all duration-300"
            style={{ width: `${progress.percentage}%` }}
          />
        </div>
        <div className="text-xs mt-1 text-center opacity-90">
          {progress.current} / {progress.required} XP to Level {level + 1}
        </div>
      </div>

      {/* Streak Display */}
      <div className="flex items-center justify-between pt-3 border-t border-white border-opacity-30">
        <div className="flex items-center space-x-2">
          <Flame className="w-5 h-5 text-orange-300" />
          <span className="font-semibold">{gamificationData.currentStreak} Day Streak</span>
        </div>
        <div className="flex items-center space-x-1">
          <Star className="w-4 h-4 text-yellow-300" />
          <span className="text-sm">{gamificationData.achievements.length}</span>
        </div>
      </div>
    </div>
  );
}
