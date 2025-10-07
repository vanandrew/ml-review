import { useEffect, useState } from 'react';
import { Sparkles, Trophy } from 'lucide-react';

interface LevelUpModalProps {
  level: number;
  show: boolean;
  onClose: () => void;
  selectedTheme?: string;
}

const THEME_GRADIENTS: Record<string, string> = {
  'default': 'from-purple-500 to-pink-500',
  'theme-ocean': 'from-blue-400 to-cyan-500',
  'theme-forest': 'from-green-400 to-emerald-600',
  'theme-sunset': 'from-orange-400 to-pink-500',
};

export default function LevelUpModal({ level, show, onClose, selectedTheme = 'default' }: LevelUpModalProps) {
  const [isVisible, setIsVisible] = useState(false);
  const gradientColors = THEME_GRADIENTS[selectedTheme] || THEME_GRADIENTS['default'];

  useEffect(() => {
    if (show) {
      setIsVisible(true);
      // Auto-close after 4 seconds
      const timer = setTimeout(() => {
        handleClose();
      }, 4000);
      
      return () => clearTimeout(timer);
    }
  }, [show]);

  const handleClose = () => {
    setIsVisible(false);
    setTimeout(() => {
      onClose();
    }, 300);
  };

  if (!show) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className={`fixed inset-0 bg-black/50 z-50 transition-opacity duration-300 ${
          isVisible ? 'opacity-100' : 'opacity-0'
        }`}
        onClick={handleClose}
      />

      {/* Modal */}
      <div
        className={`fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-50 transition-all duration-300 ${
          isVisible ? 'scale-100 opacity-100' : 'scale-75 opacity-0'
        }`}
      >
        <div className={`bg-gradient-to-br ${gradientColors} rounded-2xl p-8 shadow-2xl text-white text-center min-w-[320px] relative overflow-hidden`}>
          {/* Decorative sparkles */}
          <div className="absolute top-2 left-2 animate-pulse">
            <Sparkles className="w-6 h-6 text-yellow-300" />
          </div>
          <div className="absolute top-4 right-4 animate-pulse animation-delay-200">
            <Sparkles className="w-5 h-5 text-yellow-200" />
          </div>
          <div className="absolute bottom-4 left-6 animate-pulse animation-delay-400">
            <Sparkles className="w-4 h-4 text-yellow-300" />
          </div>

          {/* Trophy icon */}
          <div className="flex justify-center mb-4">
            <div className="bg-white/20 rounded-full p-4">
              <Trophy className="w-16 h-16" />
            </div>
          </div>

          {/* Title */}
          <h2 className="text-3xl font-bold mb-2">
            Level Up! 🎉
          </h2>

          {/* Level display */}
          <div className="bg-white/20 rounded-lg p-4 mb-4 backdrop-blur-sm">
            <div className="text-5xl font-bold mb-2">{level}</div>
            <div className="text-sm uppercase tracking-wider opacity-90">
              You've reached Level {level}!
            </div>
          </div>

          {/* Motivational message */}
          <p className="text-lg opacity-90 mb-4">
            Keep up the amazing work! 🚀
          </p>

          {/* Close button */}
          <button
            onClick={handleClose}
            className="bg-white text-purple-600 font-semibold px-6 py-2 rounded-lg hover:bg-purple-50 transition-colors"
          >
            Continue Learning
          </button>
        </div>
      </div>

      <style>{`
        .animation-delay-200 {
          animation-delay: 0.2s;
        }
        .animation-delay-400 {
          animation-delay: 0.4s;
        }
      `}</style>
    </>
  );
}
