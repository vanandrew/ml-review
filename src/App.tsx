import { useState, useEffect } from 'react';
import { ThemeMode, UserProgress, GamificationData, AchievementDefinition, QuizScore } from './types';
import Sidebar from './components/Sidebar';
import TopicView from './components/TopicView';
import DashboardView from './components/DashboardView';
import GemShop from './components/GemShop';
import ProfileSettings from './components/ProfileSettings';
import ChallengeMode from './components/ChallengeMode';
import XPReward from './components/XPReward';
import AchievementModal from './components/AchievementModal';
import ConfettiAnimation from './components/ConfettiAnimation';
import LevelUpModal from './components/LevelUpModal';
import { getTopicById } from './data/topicsIndex';
import { quizQuestionPools } from './data/quizQuestions';
import { initializeGamificationData, calculateLevel, XP_PER_CORRECT_ANSWER, XP_PERFECT_BONUS, XP_FIRST_TIME_COMPLETION, checkAndResetDailyProgress } from './utils/gamification';
import { updateStreak } from './utils/streak';
import { checkForNewAchievements } from './utils/achievements';
import { generateWeeklyChallenge, shouldGenerateNewChallenge, updateChallengeProgress, isChallengeComplete } from './utils/challenges';
import { awardGems as awardGemsUtil, spendGems, checkDailyLoginGems, calculateGemsEarned, GEM_SHOP_ITEMS } from './utils/gems';

function App() {
  // Initialize theme from localStorage synchronously to avoid flash
  const [theme, setTheme] = useState<ThemeMode>(() => {
    const savedTheme = localStorage.getItem('ml-review-theme') as ThemeMode;
    return savedTheme || 'light';
  });
  
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string | null>('foundations');
  const [showingDashboard, setShowingDashboard] = useState(false);
  const [showingShop, setShowingShop] = useState(false);
  const [showingSettings, setShowingSettings] = useState(false);
  const [showingChallengeMode, setShowingChallengeMode] = useState(false);
  const [userProgress, setUserProgress] = useState<UserProgress>({});
  const [gamificationData, setGamificationData] = useState<GamificationData>({
    ...initializeGamificationData(),
    challengeModeHighScore: 0,
  });
  const [xpReward, setXpReward] = useState<{ amount: number; reason: string } | null>(null);
  const [newAchievement, setNewAchievement] = useState<AchievementDefinition | null>(null);
  const [showConfetti, setShowConfetti] = useState(false);
  const [levelUpData, setLevelUpData] = useState<{ show: boolean; level: number }>({ show: false, level: 0 });

  useEffect(() => {
    // Apply initial theme class on mount
    document.documentElement.classList.toggle('dark', theme === 'dark');

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
          if (progress[topicId].firstCompletion) {
            progress[topicId].firstCompletion = new Date(progress[topicId].firstCompletion);
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

    // Load gamification data from localStorage
    const savedGamification = localStorage.getItem('ml-review-gamification');
    if (savedGamification) {
      try {
        const data = JSON.parse(savedGamification);
        // Convert date strings back to Date objects
        if (data.achievements) {
          data.achievements = data.achievements.map((a: any) => ({
            ...a,
            unlockedDate: new Date(a.unlockedDate)
          }));
        }
        // Ensure challengeModeHighScore exists (for backwards compatibility)
        if (data.challengeModeHighScore === undefined) {
          data.challengeModeHighScore = 0;
        }
        setGamificationData(data);
      } catch (error) {
        console.error('Failed to load gamification data:', error);
      }
    }
  }, []);

  // Update streak on app load
  useEffect(() => {
    if (gamificationData.lastActivityDate) {
      const streakUpdate = updateStreak(
        gamificationData.lastActivityDate,
        gamificationData.currentStreak,
        gamificationData.streakFreezeAvailable
      );

      if (streakUpdate.newStreak !== gamificationData.currentStreak) {
        setGamificationData(prev => ({
          ...prev,
          currentStreak: streakUpdate.newStreak,
          longestStreak: Math.max(prev.longestStreak, streakUpdate.newStreak),
          streakFreezeAvailable: streakUpdate.streakFreezeUsed ? false : prev.streakFreezeAvailable,
          lastStreakFreezeDate: streakUpdate.streakFreezeUsed ? new Date().toISOString() : prev.lastStreakFreezeDate,
        }));
      }
    }

    // Check and reset daily progress
    const dailyReset = checkAndResetDailyProgress(
      gamificationData.lastDailyReset,
      gamificationData.dailyXP
    );
    if (dailyReset.shouldReset) {
      setGamificationData(prev => ({
        ...prev,
        dailyXP: 0,
        lastDailyReset: new Date().toISOString(),
      }));
    }

    // Check daily login gems
    if (checkDailyLoginGems(gamificationData.lastDailyLoginGems)) {
      const today = new Date().toISOString().split('T')[0];
      const gemsEarned = calculateGemsEarned('daily-login');
      setGamificationData(prev => {
        const result = awardGemsUtil(prev.gems, gemsEarned, 'Daily login', prev.gemTransactions);
        return {
          ...prev,
          gems: result.gems,
          gemTransactions: result.transactions,
          lastDailyLoginGems: today,
        };
      });
    }

    // Check and generate weekly challenge (initialize if null or regenerate if expired)
    if (!gamificationData.weeklyChallenge || shouldGenerateNewChallenge(gamificationData.weeklyChallenge)) {
      const newChallenge = generateWeeklyChallenge();
      setGamificationData(prev => ({
        ...prev,
        weeklyChallenge: newChallenge,
      }));
    }
  }, [gamificationData.weeklyChallenge]); // Re-run when challenge changes

  useEffect(() => {
    // Save theme to localStorage and apply to document
    localStorage.setItem('ml-review-theme', theme);
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  useEffect(() => {
    // Save user progress to localStorage
    localStorage.setItem('ml-review-progress', JSON.stringify(userProgress));
  }, [userProgress]);

  useEffect(() => {
    // Save gamification data to localStorage
    localStorage.setItem('ml-review-gamification', JSON.stringify(gamificationData));
  }, [gamificationData]);

  const awardXP = (amount: number, reason: string) => {
    setXpReward({ amount, reason });
    
    const today = new Date().toISOString().split('T')[0];
    
    setGamificationData(prev => {
      const oldLevel = calculateLevel(prev.totalXP);
      const newTotalXP = prev.totalXP + amount;
      const newLevel = calculateLevel(newTotalXP);
      
      const newData = {
        ...prev,
        totalXP: newTotalXP,
        dailyXP: prev.dailyXP + amount,
        lastActivityDate: new Date().toISOString(),
      };

      // Check for level up and award gems
      if (newLevel > oldLevel) {
        setTimeout(() => {
          setLevelUpData({ show: true, level: newLevel });
          // Award level up gems
          const levelUpGems = calculateGemsEarned('level-up');
          setGamificationData(prevData => {
            const gemResult = awardGemsUtil(prevData.gems, levelUpGems, `Level ${newLevel} reached!`, prevData.gemTransactions);
            return {
              ...prevData,
              gems: gemResult.gems,
              gemTransactions: gemResult.transactions,
            };
          });
        }, 1000);
      }

      // Update activity history
      const existingRecord = prev.activityHistory.find(r => r.date === today);
      if (existingRecord) {
        newData.activityHistory = prev.activityHistory.map(r =>
          r.date === today
            ? { ...r, xpEarned: r.xpEarned + amount }
            : r
        );
      } else {
        newData.activityHistory = [
          ...prev.activityHistory,
          { date: today, xpEarned: amount, quizzesTaken: 0, topicsCompleted: 0 },
        ];
      }
      
      // Check for new achievements
      const newAchievements = checkForNewAchievements(newData, userProgress);
      if (newAchievements.length > 0) {
        // Show first new achievement
        setNewAchievement(newAchievements[0]);
        // Add all new achievements and award gems
        const achievementGems = newAchievements.length * calculateGemsEarned('achievement');
        const gemResult = awardGemsUtil(prev.gems, achievementGems, `${newAchievements.length} achievement${newAchievements.length > 1 ? 's' : ''} unlocked!`, prev.gemTransactions);
        
        newData.achievements = [
          ...prev.achievements,
          ...newAchievements.map(a => ({
            id: a.id,
            unlockedDate: new Date(),
          })),
        ];
        newData.gems = gemResult.gems;
        newData.gemTransactions = gemResult.transactions;
      }
      
      return newData;
    });
  };

  const handleQuizComplete = (score: QuizScore, topicId: string) => {
    // Calculate XP earned
    const isPerfect = score.score === score.totalQuestions;
    const baseXP = score.score * XP_PER_CORRECT_ANSWER;
    const bonusXP = isPerfect ? XP_PERFECT_BONUS : 0;
    const totalXP = baseXP + bonusXP;

    // Trigger confetti for perfect scores
    if (isPerfect) {
      setShowConfetti(true);
      // Award gems for perfect quiz
      setTimeout(() => {
        const perfectGems = calculateGemsEarned('perfect-quiz');
        setGamificationData(prev => {
          const gemResult = awardGemsUtil(prev.gems, perfectGems, 'Perfect quiz!', prev.gemTransactions);
          return {
            ...prev,
            gems: gemResult.gems,
            gemTransactions: gemResult.transactions,
          };
        });
      }, 500);
    }

    // Award XP
    awardXP(totalXP, isPerfect ? `Perfect quiz! ${score.score}/${score.totalQuestions}` : `Quiz completed: ${score.score}/${score.totalQuestions}`);

    // Track time of day for special achievements
    const hour = new Date().getHours();
    const isEarlyBird = hour < 8;
    const isNightOwl = hour >= 22;

    // Check if this is a review of an already mastered topic
    const wasAlreadyMastered = userProgress[topicId]?.status === 'mastered';

    // Update gamification stats and check weekly challenge
    setGamificationData(prev => {
      const updatedData = {
        ...prev,
        totalQuizzes: prev.totalQuizzes + 1,
        perfectQuizzes: isPerfect ? prev.perfectQuizzes + 1 : prev.perfectQuizzes,
        consecutivePerfectQuizzes: isPerfect ? prev.consecutivePerfectQuizzes + 1 : 0,
        quizzesByTimeOfDay: {
          morning: isEarlyBird ? prev.quizzesByTimeOfDay.morning + 1 : prev.quizzesByTimeOfDay.morning,
          night: isNightOwl ? prev.quizzesByTimeOfDay.night + 1 : prev.quizzesByTimeOfDay.night,
        },
      };

      // Update weekly challenge progress if exists
      let updatedChallenge = prev.weeklyChallenge;
      if (updatedChallenge) {
        // Track quiz completions
        updatedChallenge = updateChallengeProgress(updatedChallenge, 'complete_quizzes', 1);
        
        // Track reviews of mastered topics
        if (wasAlreadyMastered) {
          updatedChallenge = updateChallengeProgress(updatedChallenge, 'review_topics', 1);
        }
        
        // Track newly mastered topics
        if (isPerfect && !wasAlreadyMastered) {
          updatedChallenge = updateChallengeProgress(updatedChallenge, 'master_topics', 1);
        }
        
        // Check if challenge just completed and award bonus XP and gems
        const wasIncomplete = prev.weeklyChallenge && !isChallengeComplete(prev.weeklyChallenge);
        if (wasIncomplete && isChallengeComplete(updatedChallenge)) {
          setTimeout(() => {
            awardXP(updatedChallenge!.reward, `🎉 Weekly Challenge Complete! ${updatedChallenge!.description}`);
            // Award weekly challenge gems
            const challengeGems = calculateGemsEarned('weekly-challenge');
            setGamificationData(prevData => {
              const gemResult = awardGemsUtil(prevData.gems, challengeGems, 'Weekly challenge completed!', prevData.gemTransactions);
              return {
                ...prevData,
                gems: gemResult.gems,
                gemTransactions: gemResult.transactions,
              };
            });
          }, 500);
        }
      }

      return {
        ...updatedData,
        weeklyChallenge: updatedChallenge,
      };
    });

    // Check if this is first completion of this topic
    const isFirstCompletion = !userProgress[topicId]?.firstCompletion;
    if (isFirstCompletion && isPerfect) {
      awardXP(XP_FIRST_TIME_COMPLETION, 'First time completing this topic!');
    }
  };

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
    setGamificationData(prev => ({
      ...prev,
      themeChanges: prev.themeChanges + 1,
    }));
  };

  const handleTopicSelect = (topicId: string, categoryId: string) => {
    setSelectedTopic(topicId);
    setSelectedCategory(categoryId);
    setShowingDashboard(false);
    setShowingShop(false);
    setShowingSettings(false);

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

  const handleDashboardSelect = () => {
    setShowingDashboard(true);
    setShowingShop(false);
    setShowingSettings(false);
    setSelectedTopic(null);
  };

  const handleShopSelect = () => {
    setShowingShop(true);
    setShowingDashboard(false);
    setShowingSettings(false);
    setSelectedTopic(null);
  };

  const handleSettingsSelect = () => {
    setShowingSettings(true);
    setShowingShop(false);
    setShowingDashboard(false);
    setShowingChallengeMode(false);
    setSelectedTopic(null);
  };

  const handleChallengeSelect = () => {
    setShowingChallengeMode(true);
    setShowingDashboard(false);
    setShowingShop(false);
    setShowingSettings(false);
    setSelectedTopic(null);
  };

  const handleChallengeComplete = (score: number) => {
    if (score > gamificationData.challengeModeHighScore) {
      setGamificationData(prev => ({
        ...prev,
        challengeModeHighScore: score,
      }));
    }
  };

  const handleChallengeExit = () => {
    setShowingChallengeMode(false);
  };

  const handlePurchaseItem = (itemId: string) => {
    const item = GEM_SHOP_ITEMS.find(i => i.id === itemId);
    if (!item) return;

    setGamificationData(prev => {
      const result = spendGems(prev.gems, item.cost, `Purchased ${item.name}`, prev.gemTransactions);
      if (!result.success) {
        alert('Not enough gems!');
        return prev;
      }
      
      return {
        ...prev,
        gems: result.gems,
        gemTransactions: result.transactions,
        purchasedItems: [...prev.purchasedItems, itemId],
      };
    });
  };

  const handleThemeChange = (themeId: string) => {
    setGamificationData(prev => ({
      ...prev,
      selectedTheme: themeId,
    }));
  };

  const handleBadgeChange = (badge: string) => {
    setGamificationData(prev => ({
      ...prev,
      selectedBadge: badge,
    }));
  };

  const handleSetDailyGoal = (goal: number) => {
    setGamificationData(prev => ({
      ...prev,
      dailyGoal: goal,
    }));
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors">
      {/* XP Reward Notification */}
      {xpReward && (
        <XPReward
          amount={xpReward.amount}
          reason={xpReward.reason}
          onComplete={() => setXpReward(null)}
          selectedTheme={gamificationData.selectedTheme}
        />
      )}

      {/* Achievement Modal */}
      {newAchievement && (
        <AchievementModal
          achievement={newAchievement}
          onClose={() => setNewAchievement(null)}
        />
      )}

      {/* Confetti Animation */}
      <ConfettiAnimation 
        trigger={showConfetti}
        onComplete={() => setShowConfetti(false)}
      />

      {/* Level Up Modal */}
      <LevelUpModal
        level={levelUpData.level}
        show={levelUpData.show}
        onClose={() => setLevelUpData({ show: false, level: 0 })}
        selectedTheme={gamificationData.selectedTheme}
      />

      <div className="flex">
        <Sidebar
          selectedTopic={selectedTopic}
          selectedCategory={selectedCategory}
          userProgress={userProgress}
          gamificationData={gamificationData}
          onTopicSelect={handleTopicSelect}
          onCategorySelect={handleCategorySelect}
          onDashboardSelect={handleDashboardSelect}
          showingDashboard={showingDashboard}
          onShopSelect={handleShopSelect}
          onSettingsSelect={handleSettingsSelect}
          onChallengeSelect={handleChallengeSelect}
          showingShop={showingShop}
          showingSettings={showingSettings}
          showingChallengeMode={showingChallengeMode}
        />

        {/* Main Content */}
        <main className="flex-1">
          <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                {showingDashboard
                  ? 'Dashboard'
                  : showingShop
                  ? 'Gem Shop'
                  : showingSettings
                  ? 'Settings'
                  : showingChallengeMode
                  ? 'Challenge Mode'
                  : selectedTopic
                  ? selectedTopic.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')
                  : 'Getting Started'
                }
              </h2>
              <button
                onClick={toggleTheme}
                className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              >
                {theme === 'light' ? '☀️' : '🌙'}
              </button>
            </div>
          </header>

          <div className="p-6">
            <div className="max-w-4xl">
              {showingChallengeMode ? (
                (() => {
                  // Get all questions from mastered topics
                  const masteredTopicIds = Object.entries(userProgress)
                    .filter(([_, progress]) => progress.status === 'mastered')
                    .map(([topicId]) => topicId);

                  const allMasteredQuestions = masteredTopicIds.flatMap(topicId => 
                    quizQuestionPools[topicId] || []
                  );

                  return (
                    <ChallengeMode
                      allQuestions={allMasteredQuestions}
                      highScore={gamificationData.challengeModeHighScore}
                      onComplete={handleChallengeComplete}
                      onExit={handleChallengeExit}
                    />
                  );
                })()
              ) : showingShop ? (
                <GemShop
                  currentGems={gamificationData.gems}
                  purchasedItems={gamificationData.purchasedItems}
                  onPurchase={handlePurchaseItem}
                  selectedTheme={gamificationData.selectedTheme}
                />
              ) : showingSettings ? (
                <ProfileSettings
                  selectedTheme={gamificationData.selectedTheme}
                  selectedBadge={gamificationData.selectedBadge}
                  dailyGoal={gamificationData.dailyGoal}
                  onThemeChange={handleThemeChange}
                  onBadgeChange={handleBadgeChange}
                  onDailyGoalChange={handleSetDailyGoal}
                  purchasedItems={gamificationData.purchasedItems}
                />
              ) : showingDashboard ? (
                <DashboardView
                  gamificationData={gamificationData}
                  userProgress={userProgress}
                  onSetDailyGoal={handleSetDailyGoal}
                  onSelectTopic={handleTopicSelect}
                />
              ) : selectedTopic ? (
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
                      gamificationData={gamificationData}
                      onProgressUpdate={(progress) => {
                        setUserProgress(prev => ({
                          ...prev,
                          [selectedTopic]: progress
                        }));
                      }}
                      onAwardXP={awardXP}
                      onQuizComplete={(score) => handleQuizComplete(score, selectedTopic)}
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