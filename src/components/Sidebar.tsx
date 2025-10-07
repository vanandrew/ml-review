
import { ChevronRight, BookOpen, CheckCircle, Circle, LayoutDashboard, ShoppingCart, Settings, Zap } from 'lucide-react';
import { categories } from '../data/categories';
import { UserProgress, GamificationData } from '../types';
import GamificationStats from './GamificationStats';
import GemDisplay from './GemDisplay';

interface SidebarProps {
  selectedTopic: string | null;
  selectedCategory: string | null;
  userProgress: UserProgress;
  gamificationData: GamificationData;
  onTopicSelect: (topicId: string, categoryId: string) => void;
  onCategorySelect: (categoryId: string) => void;
  onDashboardSelect: () => void;
  showingDashboard: boolean;
  onShopSelect?: () => void;
  onSettingsSelect?: () => void;
  onChallengeSelect?: () => void;
  showingShop?: boolean;
  showingSettings?: boolean;
  showingChallengeMode?: boolean;
}

export default function Sidebar({
  selectedTopic,
  selectedCategory,
  userProgress,
  gamificationData,
  onTopicSelect,
  onCategorySelect,
  onDashboardSelect,
  showingDashboard,
  onShopSelect,
  onSettingsSelect,
  onChallengeSelect,
  showingShop,
  showingSettings,
  showingChallengeMode,
}: SidebarProps) {
  const getProgressIcon = (topicId: string) => {
    const progress = userProgress[topicId];
    if (!progress || progress.status === 'not_started') {
      return <Circle className="w-4 h-4 text-gray-400" />;
    }
    if (progress.status === 'reviewing') {
      return <Circle className="w-4 h-4 text-yellow-500 fill-yellow-200" />;
    }
    return <CheckCircle className="w-4 h-4 text-green-500" />;
  };

  const getColorClasses = (color: string) => {
    const colorMap: Record<string, string> = {
      blue: 'text-blue-600 bg-blue-50 border-blue-200 dark:text-blue-400 dark:bg-blue-900/20 dark:border-blue-800',
      green: 'text-green-600 bg-green-50 border-green-200 dark:text-green-400 dark:bg-green-900/20 dark:border-green-800',
      purple: 'text-purple-600 bg-purple-50 border-purple-200 dark:text-purple-400 dark:bg-purple-900/20 dark:border-purple-800',
      orange: 'text-orange-600 bg-orange-50 border-orange-200 dark:text-orange-400 dark:bg-orange-900/20 dark:border-orange-800',
      pink: 'text-pink-600 bg-pink-50 border-pink-200 dark:text-pink-400 dark:bg-pink-900/20 dark:border-pink-800',
      indigo: 'text-indigo-600 bg-indigo-50 border-indigo-200 dark:text-indigo-400 dark:bg-indigo-900/20 dark:border-indigo-800',
      red: 'text-red-600 bg-red-50 border-red-200 dark:text-red-400 dark:bg-red-900/20 dark:border-red-800',
      yellow: 'text-yellow-600 bg-yellow-50 border-yellow-200 dark:text-yellow-400 dark:bg-yellow-900/20 dark:border-yellow-800',
    };
    return colorMap[color] || colorMap.blue;
  };

  const formatTopicTitle = (topicId: string) => {
    return topicId
      .split('-')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  return (
    <aside className="w-64 min-h-screen bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col">
      <div className="p-4">
        <h1 className="text-xl font-bold text-gray-900 dark:text-white">
          ML Interview Prep
        </h1>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
          Master ML concepts for interviews
        </p>
      </div>

      {/* Gamification Stats */}
      <div className="px-4 mb-4">
        <GamificationStats gamificationData={gamificationData} />
      </div>

      {/* Gem Display */}
      <div className="px-4 mb-4">
        <GemDisplay 
          gems={gamificationData.gems} 
          onClick={showingChallengeMode ? undefined : onShopSelect} 
          selectedTheme={gamificationData.selectedTheme} 
        />
      </div>

      <nav className="flex-1 scrollbar-thin overflow-y-auto">
        {/* Dashboard Button */}
        <div className="px-2 mb-2">
          <button
            onClick={() => !showingChallengeMode && onDashboardSelect()}
            disabled={showingChallengeMode}
            className={`w-full flex items-center space-x-2 px-3 py-2 text-sm rounded-lg transition-colors ${
              showingDashboard
                ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 font-medium'
                : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
            } ${showingChallengeMode ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            <LayoutDashboard className="w-4 h-4" />
            <span>Dashboard</span>
          </button>
        </div>

        {/* Challenge Mode Button */}
        <div className="px-2 mb-2">
          <button
            onClick={onChallengeSelect}
            className={`w-full flex items-center space-x-2 px-3 py-2 text-sm rounded-lg transition-colors ${
              showingChallengeMode
                ? 'bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 text-yellow-700 dark:text-yellow-400 font-medium border border-yellow-300 dark:border-yellow-700'
                : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
            }`}
          >
            <Zap className="w-4 h-4" />
            <span>Challenge Mode</span>
          </button>
        </div>

        {/* Shop Button */}
        <div className="px-2 mb-2">
          <button
            onClick={() => !showingChallengeMode && onShopSelect && onShopSelect()}
            disabled={showingChallengeMode}
            className={`w-full flex items-center space-x-2 px-3 py-2 text-sm rounded-lg transition-colors ${
              showingShop
                ? 'bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400 font-medium'
                : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
            } ${showingChallengeMode ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            <ShoppingCart className="w-4 h-4" />
            <span>Gem Shop</span>
          </button>
        </div>

        {/* Settings Button */}
        <div className="px-2 mb-2">
          <button
            onClick={() => !showingChallengeMode && onSettingsSelect && onSettingsSelect()}
            disabled={showingChallengeMode}
            className={`w-full flex items-center space-x-2 px-3 py-2 text-sm rounded-lg transition-colors ${
              showingSettings
                ? 'bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-white font-medium'
                : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
            } ${showingChallengeMode ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            <Settings className="w-4 h-4" />
            <span>Settings</span>
          </button>
        </div>

        <div className="px-4 py-2">
          <h2 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">
            Categories
          </h2>
        </div>

        <div className="space-y-1 px-2">
          {categories.map((category) => {
            const isExpanded = selectedCategory === category.id;
            const completedTopics = category.topics.filter(
              topicId => userProgress[topicId]?.status === 'mastered'
            ).length;

            return (
              <div key={category.id}>
                <button
                  onClick={() => !showingChallengeMode && onCategorySelect(category.id)}
                  disabled={showingChallengeMode}
                  className={`w-full flex items-center justify-between px-3 py-2 text-sm rounded-lg transition-colors ${
                    isExpanded
                      ? getColorClasses(category.color)
                      : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                  } ${showingChallengeMode ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <div className="flex items-center space-x-2">
                    <BookOpen className="w-4 h-4" />
                    <span className="font-medium">{category.title}</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {completedTopics}/{category.topics.length}
                    </span>
                    <ChevronRight
                      className={`w-4 h-4 transition-transform ${
                        isExpanded ? 'rotate-90' : ''
                      }`}
                    />
                  </div>
                </button>

                {isExpanded && (
                  <div className="ml-4 mt-1 space-y-1">
                    {category.topics.map((topicId) => (
                      <button
                        key={topicId}
                        onClick={() => !showingChallengeMode && onTopicSelect(topicId, category.id)}
                        disabled={showingChallengeMode}
                        className={`w-full flex items-center space-x-2 px-3 py-1.5 text-sm rounded-md transition-colors ${
                          selectedTopic === topicId
                            ? 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-white'
                            : 'text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700/50'
                        } ${showingChallengeMode ? 'opacity-50 cursor-not-allowed' : ''}`}
                      >
                        {getProgressIcon(topicId)}
                        <span className="truncate">
                          {formatTopicTitle(topicId)}
                        </span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </nav>
    </aside>
  );
}