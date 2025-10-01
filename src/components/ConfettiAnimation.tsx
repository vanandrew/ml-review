import { useEffect, useState } from 'react';

interface ConfettiPiece {
  id: number;
  left: number;
  animationDelay: number;
  backgroundColor: string;
}

interface ConfettiAnimationProps {
  trigger: boolean;
  onComplete?: () => void;
}

const COLORS = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff', '#ff8800', '#8800ff'];

export default function ConfettiAnimation({ trigger, onComplete }: ConfettiAnimationProps) {
  const [pieces, setPieces] = useState<ConfettiPiece[]>([]);
  const [isActive, setIsActive] = useState(false);

  useEffect(() => {
    if (trigger && !isActive) {
      setIsActive(true);
      
      // Generate confetti pieces
      const newPieces = Array.from({ length: 50 }, (_, i) => ({
        id: i,
        left: Math.random() * 100,
        animationDelay: Math.random() * 0.3,
        backgroundColor: COLORS[Math.floor(Math.random() * COLORS.length)],
      }));
      
      setPieces(newPieces);
      
      // Clear after animation completes
      const timer = setTimeout(() => {
        setIsActive(false);
        setPieces([]);
        if (onComplete) onComplete();
      }, 3000);
      
      return () => clearTimeout(timer);
    }
  }, [trigger, isActive, onComplete]);

  if (!isActive) return null;

  return (
    <div className="fixed inset-0 pointer-events-none z-50 overflow-hidden">
      {pieces.map((piece) => (
        <div
          key={piece.id}
          className="absolute w-2 h-2 animate-confetti"
          style={{
            left: `${piece.left}%`,
            top: '-10px',
            backgroundColor: piece.backgroundColor,
            animationDelay: `${piece.animationDelay}s`,
          }}
        />
      ))}
      
      <style>{`
        @keyframes confetti {
          0% {
            transform: translateY(0) rotate(0deg);
            opacity: 1;
          }
          100% {
            transform: translateY(100vh) rotate(720deg);
            opacity: 0;
          }
        }
        
        .animate-confetti {
          animation: confetti 3s ease-in forwards;
        }
      `}</style>
    </div>
  );
}
