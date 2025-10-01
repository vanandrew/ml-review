import { BarChart3, TrendingUp, Award, Calendar, Clock, Target } from 'lucide-react';
import { GamificationData, UserProgress } from '../types';
import { calculateLevel } from '../utils/gamification';
import { categories } from '../data/categories';

interface StatsDashboardProps {
  gamificationData: GamificationData;
  userProgress: UserProgress;
}

export default function StatsDashboard({ gamificationData, userProgress }: StatsDashboardProps) {
  const level = calculateLevel(gamificationData.totalXP);
  
  // Calculate statistics
  const topicsStarted = Object.values(userProgress).filter(
    p => p.status === 'reviewing' || p.status === 'mastered'
  ).length;
  
  const topicsMastered = Object.values(userProgress).filter(
    p => p.status === 'mastered'
  ).length;
  
  const totalQuizzesTaken = gamificationData.totalQuizzes;
  const averageScore = totalQuizzesTaken > 0
    ? Math.round(
        Object.values(userProgress)
          .flatMap(p => p.quizScores)
          .reduce((sum, score) => sum + (score.score / score.totalQuestions) * 100, 0) /
        totalQuizzesTaken
      )
    : 0;
  
  // Category breakdown
  const categoryStats = categories.map(category => {
    const categoryTopics = category.topics;
    const masteredCount = categoryTopics.filter(
      topicId => userProgress[topicId]?.status === 'mastered'
    ).length;
    const percentage = Math.round((masteredCount / categoryTopics.length) * 100);
    
    return {
      name: category.title,
      mastered: masteredCount,
      total: categoryTopics.length,
      percentage,
      color: category.color,
    };
  });
  
  // Activity heatmap data (last 30 days)
  const activityMap = new Map<string, number>();
  gamificationData.activityHistory.forEach(record => {
    const date = new Date(record.date).toISOString().split('T')[0];
    activityMap.set(date, record.xpEarned);
  });
  
  const last30Days = Array.from({ length: 30 }, (_, i) => {
    const date = new Date();
    date.setDate(date.getDate() - (29 - i));
    const dateStr = date.toISOString().split('T')[0];
    return {
      date: dateStr,
      xp: activityMap.get(dateStr) || 0,
    };
  });

  return (
    <div className="space-y-6">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={<TrendingUp className="w-6 h-6" />}
          title="Level"
          value={level.toString()}
          subtitle={`${gamificationData.totalXP} Total XP`}
          color="blue"
        />
        <StatCard
          icon={<Award className="w-6 h-6" />}
          title="Achievements"
          value={gamificationData.achievements.length.toString()}
          subtitle={`${gamificationData.currentStreak} day streak`}
          color="purple"
        />
        <StatCard
          icon={<BarChart3 className="w-6 h-6" />}
          title="Topics Mastered"
          value={topicsMastered.toString()}
          subtitle={`${topicsStarted} total started`}
          color="green"
        />
        <StatCard
          icon={<Target className="w-6 h-6" />}
          title="Average Score"
          value={`${averageScore}%`}
          subtitle={`${totalQuizzesTaken} quizzes taken`}
          color="orange"
        />
      </div>

      {/* Quiz Statistics */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center space-x-2">
          <BarChart3 className="w-5 h-5" />
          <span>Quiz Performance</span>
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatItem label="Total Quizzes" value={totalQuizzesTaken} />
          <StatItem label="Perfect Scores" value={gamificationData.perfectQuizzes} />
          <StatItem label="Best Streak" value={gamificationData.consecutivePerfectQuizzes} />
          <StatItem label="Avg. Score" value={`${averageScore}%`} />
        </div>
      </div>

      {/* Category Breakdown */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Category Progress
        </h3>
        <div className="space-y-4">
          {categoryStats.map(stat => (
            <div key={stat.name}>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-700 dark:text-gray-300">{stat.name}</span>
                <span className="text-gray-600 dark:text-gray-400">
                  {stat.mastered}/{stat.total} ({stat.percentage}%)
                </span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all duration-300 bg-${stat.color}-500`}
                  style={{ width: `${stat.percentage}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Activity Heatmap */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center space-x-2">
          <Calendar className="w-5 h-5" />
          <span>Activity (Last 30 Days)</span>
        </h3>
        <div className="grid grid-cols-10 gap-1">
          {last30Days.map((day, index) => {
            const intensity = day.xp > 0 ? Math.min(4, Math.ceil(day.xp / 25)) : 0;
            return (
              <div
                key={index}
                className={`aspect-square rounded-sm ${
                  intensity === 0
                    ? 'bg-gray-200 dark:bg-gray-700'
                    : intensity === 1
                    ? 'bg-green-200 dark:bg-green-900'
                    : intensity === 2
                    ? 'bg-green-400 dark:bg-green-700'
                    : intensity === 3
                    ? 'bg-green-500 dark:bg-green-600'
                    : 'bg-green-600 dark:bg-green-500'
                }`}
                title={`${day.date}: ${day.xp} XP`}
              />
            );
          })}
        </div>
        <div className="flex items-center justify-end space-x-2 mt-3 text-xs text-gray-500 dark:text-gray-400">
          <span>Less</span>
          <div className="flex space-x-1">
            {[0, 1, 2, 3, 4].map(i => (
              <div
                key={i}
                className={`w-3 h-3 rounded-sm ${
                  i === 0
                    ? 'bg-gray-200 dark:bg-gray-700'
                    : i === 1
                    ? 'bg-green-200 dark:bg-green-900'
                    : i === 2
                    ? 'bg-green-400 dark:bg-green-700'
                    : i === 3
                    ? 'bg-green-500 dark:bg-green-600'
                    : 'bg-green-600 dark:bg-green-500'
                }`}
              />
            ))}
          </div>
          <span>More</span>
        </div>
      </div>

      {/* Time of Day Stats */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center space-x-2">
          <Clock className="w-5 h-5" />
          <span>Study Habits</span>
        </h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
            <div className="text-3xl mb-2">🌅</div>
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {gamificationData.quizzesByTimeOfDay.morning}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Morning Quizzes
            </div>
          </div>
          <div className="text-center p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg">
            <div className="text-3xl mb-2">🌙</div>
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {gamificationData.quizzesByTimeOfDay.night}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Night Quizzes
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Helper Components
interface StatCardProps {
  icon: React.ReactNode;
  title: string;
  value: string;
  subtitle: string;
  color: string;
}

function StatCard({ icon, title, value, subtitle, color }: StatCardProps) {
  const colorMap: Record<string, string> = {
    blue: 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400',
    purple: 'bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400',
    green: 'bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-400',
    orange: 'bg-orange-50 dark:bg-orange-900/20 text-orange-600 dark:text-orange-400',
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <div className={`w-12 h-12 rounded-lg flex items-center justify-center mb-3 ${colorMap[color]}`}>
        {icon}
      </div>
      <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">{title}</div>
      <div className="text-2xl font-bold text-gray-900 dark:text-white mb-1">{value}</div>
      <div className="text-xs text-gray-500 dark:text-gray-400">{subtitle}</div>
    </div>
  );
}

interface StatItemProps {
  label: string;
  value: string | number;
}

function StatItem({ label, value }: StatItemProps) {
  return (
    <div className="text-center">
      <div className="text-2xl font-bold text-gray-900 dark:text-white">{value}</div>
      <div className="text-sm text-gray-600 dark:text-gray-400">{label}</div>
    </div>
  );
}
