import { ReviewItem, ReviewStatus } from '../utils/reviewSystem';
import { Clock, CheckCircle, AlertCircle, BookOpen } from 'lucide-react';

interface ReviewQueueProps {
  reviewItems: ReviewItem[];
  onSelectTopic: (topicId: string, categoryId: string) => void;
  getTopicTitle: (topicId: string) => string;
}

const STATUS_CONFIG: Record<ReviewStatus, { icon: any; label: string; color: string; bgColor: string }> = {
  'new': {
    icon: BookOpen,
    label: 'New',
    color: 'text-blue-600 dark:text-blue-400',
    bgColor: 'bg-blue-50 dark:bg-blue-900/20',
  },
  'learning': {
    icon: Clock,
    label: 'Learning',
    color: 'text-yellow-600 dark:text-yellow-400',
    bgColor: 'bg-yellow-50 dark:bg-yellow-900/20',
  },
  'mastered': {
    icon: CheckCircle,
    label: 'Mastered',
    color: 'text-green-600 dark:text-green-400',
    bgColor: 'bg-green-50 dark:bg-green-900/20',
  },
  'needs-review': {
    icon: AlertCircle,
    label: 'Needs Review',
    color: 'text-red-600 dark:text-red-400',
    bgColor: 'bg-red-50 dark:bg-red-900/20',
  },
};

export default function ReviewQueue({ reviewItems, onSelectTopic, getTopicTitle }: ReviewQueueProps) {
  const dueReviews = reviewItems.filter(item => item.status === 'needs-review');
  const weakTopics = reviewItems.filter(item => item.lastScore !== null && item.lastScore < 80);

  const formatDate = (date: Date) => {
    const now = new Date();
    const diffDays = Math.floor((date.getTime() - now.getTime()) / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Tomorrow';
    if (diffDays === -1) return 'Yesterday';
    if (diffDays < -1) return `${Math.abs(diffDays)} days overdue`;
    if (diffDays > 1) return `In ${diffDays} days`;
    return date.toLocaleDateString();
  };

  const renderReviewItem = (item: ReviewItem) => {
    const config = STATUS_CONFIG[item.status];
    const Icon = config.icon;
    
    return (
      <button
        key={item.topicId}
        onClick={() => onSelectTopic(item.topicId, item.categoryId)}
        className={`w-full p-4 rounded-lg border-2 transition-all hover:scale-[1.02] hover:shadow-md
          ${config.bgColor} border-transparent hover:border-current ${config.color}
        `}
      >
        <div className="flex items-start gap-3">
          <div className="flex-shrink-0 mt-0.5">
            <Icon className="w-5 h-5" />
          </div>
          
          <div className="flex-1 text-left">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-1">
              {getTopicTitle(item.topicId)}
            </h4>
            
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <span className={`px-2 py-0.5 rounded-full ${config.bgColor} ${config.color} font-medium`}>
                {config.label}
              </span>
              
              {item.lastScore !== null && (
                <span className="text-gray-600 dark:text-gray-400">
                  Last: {Math.round(item.lastScore)}%
                </span>
              )}
              
              <span className="text-gray-500 dark:text-gray-500 text-xs">
                {formatDate(item.nextReviewDate)}
              </span>
            </div>
            
            {item.priority >= 7 && (
              <div className="mt-2 text-xs font-medium text-orange-600 dark:text-orange-400">
                ⚠️ High Priority
              </div>
            )}
          </div>
        </div>
      </button>
    );
  };

  return (
    <div className="space-y-6">
      {/* Due Reviews Section */}
      {dueReviews.length > 0 && (
        <div>
          <div className="flex items-center gap-2 mb-4">
            <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400" />
            <h3 className="text-lg font-bold text-gray-900 dark:text-white">
              Due for Review ({dueReviews.length})
            </h3>
          </div>
          
          <div className="space-y-3">
            {dueReviews.slice(0, 5).map(renderReviewItem)}
          </div>
        </div>
      )}

      {/* Weak Topics Section */}
      {weakTopics.length > 0 && (
        <div>
          <div className="flex items-center gap-2 mb-4">
            <BookOpen className="w-5 h-5 text-yellow-600 dark:text-yellow-400" />
            <h3 className="text-lg font-bold text-gray-900 dark:text-white">
              Practice Weak Areas ({weakTopics.length})
            </h3>
          </div>
          
          <div className="space-y-3">
            {weakTopics.slice(0, 5).map(renderReviewItem)}
          </div>
        </div>
      )}

      {/* All Topics Section */}
      <div>
        <div className="flex items-center gap-2 mb-4">
          <Clock className="w-5 h-5 text-gray-600 dark:text-gray-400" />
          <h3 className="text-lg font-bold text-gray-900 dark:text-white">
            All Topics ({reviewItems.length})
          </h3>
        </div>
        
        <div className="space-y-3 max-h-[400px] overflow-y-auto pr-2">
          {reviewItems.map(renderReviewItem)}
        </div>
      </div>

      {reviewItems.length === 0 && (
        <div className="text-center py-12 text-gray-500 dark:text-gray-400">
          <BookOpen className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <p className="text-lg">No topics to review yet</p>
          <p className="text-sm mt-2">Start learning some topics to build your review queue!</p>
        </div>
      )}
    </div>
  );
}
