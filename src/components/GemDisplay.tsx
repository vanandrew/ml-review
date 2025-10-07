import { Gem } from 'lucide-react';

interface GemDisplayProps {
  gems: number;
  onClick?: () => void;
  selectedTheme?: string;
}

const THEME_GRADIENTS: Record<string, string> = {
  'default': 'from-purple-500 to-pink-500',
  'theme-ocean': 'from-blue-400 to-cyan-500',
  'theme-forest': 'from-green-400 to-emerald-600',
  'theme-sunset': 'from-orange-400 to-pink-500',
};

export default function GemDisplay({ gems, onClick, selectedTheme = 'default' }: GemDisplayProps) {
  const gradientColors = THEME_GRADIENTS[selectedTheme] || THEME_GRADIENTS['default'];
  const isDisabled = !onClick;
  
  return (
    <button
      onClick={onClick}
      disabled={isDisabled}
      className={`flex items-center gap-2 bg-gradient-to-r ${gradientColors} text-white px-4 py-2 rounded-lg font-bold shadow-md transition-all w-full justify-center ${
        isDisabled 
          ? 'opacity-50 cursor-not-allowed' 
          : 'hover:shadow-lg hover:scale-105'
      }`}
    >
      <Gem className="w-5 h-5" />
      <span>{gems} Gems</span>
    </button>
  );
}
