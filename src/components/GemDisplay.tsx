import { Gem } from 'lucide-react';

interface GemDisplayProps {
  gems: number;
  onClick?: () => void;
}

export default function GemDisplay({ gems, onClick }: GemDisplayProps) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-2 bg-gradient-to-r from-blue-500 to-purple-500 text-white px-4 py-2 rounded-lg font-bold shadow-md hover:shadow-lg hover:scale-105 transition-all w-full justify-center"
    >
      <Gem className="w-5 h-5" />
      <span>{gems} Gems</span>
    </button>
  );
}
