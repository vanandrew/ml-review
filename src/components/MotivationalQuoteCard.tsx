import { useEffect, useState } from 'react';
import { Quote } from 'lucide-react';
import { getDailyQuote, MotivationalQuote } from '../utils/motivationalMessages';

export default function MotivationalQuoteCard() {
  const [quote, setQuote] = useState<MotivationalQuote | null>(null);

  useEffect(() => {
    setQuote(getDailyQuote());
  }, []);

  if (!quote) return null;

  return (
    <div className="bg-gradient-to-br from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 rounded-lg p-6 border border-purple-200 dark:border-purple-800">
      <div className="flex items-start gap-4">
        <div className="flex-shrink-0">
          <div className="bg-purple-500 rounded-full p-3">
            <Quote className="w-6 h-6 text-white" />
          </div>
        </div>
        
        <div className="flex-1">
          <h3 className="text-sm font-semibold text-purple-900 dark:text-purple-300 uppercase tracking-wide mb-2">
            Quote of the Day
          </h3>
          
          <blockquote className="text-lg text-gray-900 dark:text-white font-medium italic mb-3">
            "{quote.text}"
          </blockquote>
          
          {quote.author && (
            <p className="text-sm text-gray-600 dark:text-gray-400">
              — {quote.author}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
