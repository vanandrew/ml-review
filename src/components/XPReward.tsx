import { useEffect, useState } from 'react';
import { Sparkles } from 'lucide-react';

interface XPRewardProps {
  amount: number;
  reason: string;
  onComplete?: () => void;
  selectedTheme?: string;
}

const THEME_GRADIENTS: Record<string, string> = {
  'default': 'from-purple-500 to-pink-500',
  'theme-ocean': 'from-blue-400 to-cyan-500',
  'theme-forest': 'from-green-400 to-emerald-600',
  'theme-sunset': 'from-orange-400 to-pink-500',
};

export default function XPReward({ amount, reason, onComplete, selectedTheme = 'default' }: XPRewardProps) {
  const [isVisible, setIsVisible] = useState(true);
  const gradientColors = THEME_GRADIENTS[selectedTheme] || THEME_GRADIENTS['default'];

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(false);
      if (onComplete) onComplete();
    }, 3000);

    return () => clearTimeout(timer);
  }, [onComplete]);

  if (!isVisible) return null;

  return (
    <div className="fixed top-20 right-4 z-50 animate-bounce">
      <div className={`bg-gradient-to-r ${gradientColors} text-white px-6 py-3 rounded-lg shadow-lg flex items-center space-x-2`}>
        <Sparkles className="w-5 h-5" />
        <div>
          <div className="font-bold text-lg">+{amount} XP</div>
          <div className="text-xs opacity-90">{reason}</div>
        </div>
      </div>
    </div>
  );
}
