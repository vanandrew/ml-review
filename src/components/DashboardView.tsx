import { GamificationData, UserProgress } from '../types';
import StatsDashboard from './StatsDashboard';
import DailyGoals from './DailyGoals';
import WeeklyChallengeCard from './WeeklyChallengeCard';
import MotivationalQuoteCard from './MotivationalQuoteCard';
import ReviewQueue from './ReviewQueue';
import DecayWarning from './DecayWarning';
import RankingBadge from './RankingBadge';
import { getReviewQueue } from '../utils/reviewSystem';
import { getTopicById } from '../data/topicsIndex';

interface DashboardViewProps {
  gamificationData: GamificationData;
  userProgress: UserProgress;
  onSetDailyGoal: (goal: number) => void;
  onSelectTopic: (topicId: string, categoryId: string) => void;
}

export default function DashboardView({ gamificationData, userProgress, onSetDailyGoal, onSelectTopic }: DashboardViewProps) {
  // Get all topic IDs for review queue
  const allTopicIds = Object.keys(userProgress);
  const reviewQueue = getReviewQueue(userProgress, allTopicIds);
  
  // Helper function to get topic title
  const getTopicTitle = (topicId: string): string => {
    const topic = getTopicById(topicId);
    return topic?.title || topicId.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Your Dashboard
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Track your learning progress and achievements
        </p>
      </div>

      {/* Motivational Quote */}
      <MotivationalQuoteCard />

      {/* Rankings Display */}
      <RankingBadge 
        gamificationData={gamificationData}
        userProgress={userProgress}
      />

      {/* Decay Warning - Shows mastery maintenance status */}
      <DecayWarning 
        userProgress={userProgress}
        onSelectTopic={(topicId) => {
          const topic = getTopicById(topicId);
          if (topic) {
            onSelectTopic(topicId, topic.category);
          }
        }}
        getTopicTitle={getTopicTitle}
      />

      {/* Top Row: Daily Goals and Weekly Challenge */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <DailyGoals 
          gamificationData={gamificationData}
          onSetGoal={onSetDailyGoal}
        />
        <WeeklyChallengeCard challenge={gamificationData.weeklyChallenge} />
      </div>

      {/* Review Queue Section */}
      {reviewQueue.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            📚 Review Queue
          </h2>
          <ReviewQueue 
            reviewItems={reviewQueue}
            onSelectTopic={onSelectTopic}
            getTopicTitle={getTopicTitle}
          />
        </div>
      )}

      {/* Statistics Dashboard */}
      <StatsDashboard 
        gamificationData={gamificationData}
        userProgress={userProgress}
      />
    </div>
  );
}
