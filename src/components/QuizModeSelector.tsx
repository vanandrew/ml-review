import { QUIZ_MODES, QuizMode, canAccessQuizMode } from '../utils/challengeModes';
import { calculateLevel } from '../utils/gamification';
import { Lock } from 'lucide-react';

interface QuizModeSelectorProps {
  selectedMode: QuizMode;
  onSelectMode: (mode: QuizMode) => void;
  totalXP: number;
  masteredTopicsCount: number;
}

export default function QuizModeSelector({ selectedMode, onSelectMode, totalXP, masteredTopicsCount }: QuizModeSelectorProps) {
  const userLevel = calculateLevel(totalXP);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 mb-6">
      <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
        🎮 Choose Quiz Mode
      </h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {QUIZ_MODES.map(mode => {
          const canAccess = canAccessQuizMode(mode.id, userLevel, masteredTopicsCount);
          const isSelected = selectedMode === mode.id;
          
          return (
            <button
              key={mode.id}
              onClick={() => canAccess && onSelectMode(mode.id)}
              disabled={!canAccess}
              className={`
                relative p-4 rounded-lg border-2 text-left transition-all
                ${isSelected 
                  ? `${mode.bgColor} border-current ${mode.color} shadow-md` 
                  : 'bg-white dark:bg-gray-700 border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500'
                }
                ${!canAccess ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer hover:scale-[1.02]'}
              `}
            >
              {!canAccess && (
                <div className="absolute top-2 right-2">
                  <Lock className="w-4 h-4 text-gray-400" />
                </div>
              )}
              
              <div className="flex items-start gap-3 mb-2">
                <span className="text-2xl">{mode.icon}</span>
                <div className="flex-1">
                  <h4 className={`font-semibold mb-1 ${isSelected ? mode.color : 'text-gray-900 dark:text-white'}`}>
                    {mode.name}
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {mode.description}
                  </p>
                </div>
              </div>
              
              <div className="flex items-center justify-between mt-3 pt-3 border-t border-gray-200 dark:border-gray-600">
                <span className={`text-sm font-semibold ${isSelected ? mode.color : 'text-gray-700 dark:text-gray-300'}`}>
                  {mode.xpMultiplier}x XP
                </span>
                
                {mode.timeLimit && (
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    ⏱️ {mode.timeLimit}s/question
                  </span>
                )}
              </div>
              
              {mode.requirements && !canAccess && (
                <div className="mt-2 text-xs text-red-600 dark:text-red-400">
                  🔒 {mode.requirements}
                </div>
              )}
            </button>
          );
        })}
      </div>
      
      {selectedMode !== 'normal' && (
        <div className={`mt-4 p-3 rounded-lg ${QUIZ_MODES.find(m => m.id === selectedMode)?.bgColor} border ${QUIZ_MODES.find(m => m.id === selectedMode)?.color}`}>
          <p className="text-sm font-medium">
            💎 Bonus XP active! You'll earn {QUIZ_MODES.find(m => m.id === selectedMode)?.xpMultiplier}x XP for this quiz.
          </p>
        </div>
      )}
    </div>
  );
}
