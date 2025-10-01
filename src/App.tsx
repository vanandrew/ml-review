import React, { useState, useEffect } from 'react';
import { ThemeMode, UserProgress } from './types';
import Sidebar from './components/Sidebar';
import TopicView from './components/TopicView';
import { getTopicById } from './data/topicsIndex';

function App() {
  const [theme, setTheme] = useState<ThemeMode>('light');
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string | null>('foundations');
  const [userProgress, setUserProgress] = useState<UserProgress>({});

  useEffect(() => {
    // Load theme from localStorage
    const savedTheme = localStorage.getItem('ml-review-theme') as ThemeMode;
    if (savedTheme) {
      setTheme(savedTheme);
    }

    // Load user progress from localStorage
    const savedProgress = localStorage.getItem('ml-review-progress');
    if (savedProgress) {
      try {
        const progress = JSON.parse(savedProgress);
        // Convert date strings back to Date objects
        Object.keys(progress).forEach(topicId => {
          if (progress[topicId].lastAccessed) {
            progress[topicId].lastAccessed = new Date(progress[topicId].lastAccessed);
          }
          if (progress[topicId].quizScores) {
            progress[topicId].quizScores = progress[topicId].quizScores.map((score: any) => ({
              ...score,
              date: new Date(score.date)
            }));
          }
        });
        setUserProgress(progress);
      } catch (error) {
        console.error('Failed to load progress:', error);
      }
    }
  }, []);

  useEffect(() => {
    // Save theme to localStorage and apply to document
    localStorage.setItem('ml-review-theme', theme);
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  useEffect(() => {
    // Save user progress to localStorage
    localStorage.setItem('ml-review-progress', JSON.stringify(userProgress));
  }, [userProgress]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  const handleTopicSelect = (topicId: string, categoryId: string) => {
    setSelectedTopic(topicId);
    setSelectedCategory(categoryId);

    // Update last accessed time
    setUserProgress(prev => ({
      ...prev,
      [topicId]: {
        ...prev[topicId],
        status: prev[topicId]?.status || 'reviewing',
        lastAccessed: new Date(),
        quizScores: prev[topicId]?.quizScores || []
      }
    }));
  };

  const handleCategorySelect = (categoryId: string) => {
    setSelectedCategory(prev => prev === categoryId ? null : categoryId);
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors">
      <div className="flex">
        <Sidebar
          selectedTopic={selectedTopic}
          selectedCategory={selectedCategory}
          userProgress={userProgress}
          onTopicSelect={handleTopicSelect}
          onCategorySelect={handleCategorySelect}
        />

        {/* Main Content */}
        <main className="flex-1">
          <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                {selectedTopic
                  ? selectedTopic.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')
                  : 'Getting Started'
                }
              </h2>
              <button
                onClick={toggleTheme}
                className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              >
                {theme === 'light' ? '🌙' : '☀️'}
              </button>
            </div>
          </header>

          <div className="p-6">
            <div className="max-w-4xl">
              {selectedTopic ? (
                (() => {
                  const topic = getTopicById(selectedTopic);
                  if (!topic) {
                    return (
                      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                        <p className="text-gray-600 dark:text-gray-300">Topic not found.</p>
                      </div>
                    );
                  }
                  return (
                    <TopicView
                      topic={topic}
                      userProgress={userProgress[selectedTopic]}
                      onProgressUpdate={(progress) => {
                        setUserProgress(prev => ({
                          ...prev,
                          [selectedTopic]: progress
                        }));
                      }}
                    />
                  );
                })()
              ) : (
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
                    Welcome to ML Interview Prep
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300 mb-4">
                    This comprehensive platform will help you master machine learning concepts for technical interviews.
                    Navigate through different categories using the sidebar to start learning.
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                      <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-2">
                        📚 Comprehensive Content
                      </h4>
                      <p className="text-sm text-blue-700 dark:text-blue-300">
                        Theory, code examples, and interview questions for each topic
                      </p>
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                      <h4 className="font-medium text-green-900 dark:text-green-100 mb-2">
                        🧪 Practice Quizzes
                      </h4>
                      <p className="text-sm text-green-700 dark:text-green-300">
                        Multiple choice questions with score tracking
                      </p>
                    </div>
                    <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                      <h4 className="font-medium text-purple-900 dark:text-purple-100 mb-2">
                        📈 Progress Tracking
                      </h4>
                      <p className="text-sm text-purple-700 dark:text-purple-300">
                        Mark topics as reviewing or mastered
                      </p>
                    </div>
                    <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                      <h4 className="font-medium text-orange-900 dark:text-orange-100 mb-2">
                        🎮 Interactive Demos
                      </h4>
                      <p className="text-sm text-orange-700 dark:text-orange-300">
                        Visualizations to understand key concepts
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;