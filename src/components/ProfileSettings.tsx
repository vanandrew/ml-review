import { useState } from 'react';
import { User, Palette, Bell, Settings as SettingsIcon, Mail, Lock, UserCircle as UserCircleIcon, Save, Sparkles } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { AISettings as AISettingsType, AICostTracking } from '../types';
import AISettingsComponent from './AISettings';

interface ProfileSettingsProps {
  selectedTheme: string;
  selectedBadge: string;
  dailyGoal: number;
  onThemeChange: (theme: string) => void;
  onBadgeChange: (badge: string) => void;
  onDailyGoalChange: (goal: number) => void;
  purchasedItems: string[];
  onResetAllProgress: () => void;
  aiSettings?: AISettingsType;
  aiCostTracking?: AICostTracking;
  onAISettingsUpdate?: (settings: AISettingsType) => void;
  onAICostTrackingUpdate?: (tracking: AICostTracking) => void;
}

const AVAILABLE_BADGES = ['⭐', '🚀', '🧠', '🎯', '💎', '🏆', '🔥', '⚡'];
const AVAILABLE_THEMES = [
  { id: 'default', name: 'Default', colors: 'from-blue-500 to-purple-500', free: true },
  { id: 'theme-ocean', name: 'Ocean', colors: 'from-blue-400 to-cyan-500', free: false },
  { id: 'theme-forest', name: 'Forest', colors: 'from-green-400 to-emerald-600', free: false },
  { id: 'theme-sunset', name: 'Sunset', colors: 'from-orange-400 to-pink-500', free: false },
];

export default function ProfileSettings({
  selectedTheme,
  selectedBadge,
  dailyGoal,
  onThemeChange,
  onBadgeChange,
  onDailyGoalChange,
  purchasedItems,
  onResetAllProgress,
  aiSettings,
  aiCostTracking,
  onAISettingsUpdate,
  onAICostTrackingUpdate,
}: ProfileSettingsProps) {
  const { user, updateUserProfile } = useAuth();
  const [editingProfile, setEditingProfile] = useState(false);
  const [displayName, setDisplayName] = useState(user?.displayName || '');
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const handleSaveProfile = async () => {
    if (!displayName.trim()) {
      setSaveMessage({ type: 'error', text: 'Display name cannot be empty' });
      return;
    }

    try {
      setSaving(true);
      setSaveMessage(null);
      await updateUserProfile(displayName.trim());
      setSaveMessage({ type: 'success', text: 'Profile updated successfully!' });
      setEditingProfile(false);
      setTimeout(() => setSaveMessage(null), 3000);
    } catch (error: any) {
      setSaveMessage({ type: 'error', text: error.message || 'Failed to update profile' });
    } finally {
      setSaving(false);
    }
  };
  const canUseTheme = (themeId: string) => {
    const theme = AVAILABLE_THEMES.find(t => t.id === themeId);
    return theme?.free || purchasedItems.includes(themeId);
  };

  const canUseBadge = (badge: string) => {
    const badgeMap: Record<string, string> = {
      '⭐': 'free',
      '🚀': 'badge-rocket',
      '🧠': 'badge-brain',
      '🎯': 'free',
      '💎': 'free',
      '🏆': 'free',
      '🔥': 'free',
      '⚡': 'free',
    };
    const itemId = badgeMap[badge];
    return itemId === 'free' || purchasedItems.includes(itemId);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2 flex items-center gap-2">
          <SettingsIcon className="w-8 h-8" />
          Profile Settings
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Customize your learning experience
        </p>
      </div>

      {/* Account Information */}
      {user && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <UserCircleIcon className="w-6 h-6" />
            Account Information
          </h2>
          
          <div className="space-y-4">
            <div className="flex items-center space-x-4">
              {user.photoURL ? (
                <img 
                  src={user.photoURL} 
                  alt={user.displayName || 'User'} 
                  className="w-16 h-16 rounded-full"
                />
              ) : (
                <div className="w-16 h-16 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center">
                  <UserCircleIcon className="w-10 h-10 text-blue-600 dark:text-blue-400" />
                </div>
              )}
              <div className="flex-1">
                {editingProfile ? (
                  <div className="space-y-2">
                    <input
                      type="text"
                      value={displayName}
                      onChange={(e) => setDisplayName(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg dark:bg-gray-700 dark:text-white"
                      placeholder="Display name"
                    />
                    <div className="flex gap-2">
                      <button
                        onClick={handleSaveProfile}
                        disabled={saving}
                        className="flex items-center gap-1 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg disabled:opacity-50"
                      >
                        <Save className="w-4 h-4" />
                        {saving ? 'Saving...' : 'Save'}
                      </button>
                      <button
                        onClick={() => {
                          setEditingProfile(false);
                          setDisplayName(user.displayName || '');
                          setSaveMessage(null);
                        }}
                        disabled={saving}
                        className="px-3 py-1.5 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 text-sm rounded-lg disabled:opacity-50"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <div>
                    <p className="text-lg font-semibold text-gray-900 dark:text-white">
                      {user.displayName || 'No name set'}
                    </p>
                    <button
                      onClick={() => setEditingProfile(true)}
                      className="text-sm text-blue-600 dark:text-blue-400 hover:underline"
                    >
                      Edit profile
                    </button>
                  </div>
                )}
              </div>
            </div>

            {saveMessage && (
              <div className={`p-3 rounded-lg text-sm ${
                saveMessage.type === 'success' 
                  ? 'bg-green-50 dark:bg-green-900/20 text-green-800 dark:text-green-200'
                  : 'bg-red-50 dark:bg-red-900/20 text-red-800 dark:text-red-200'
              }`}>
                {saveMessage.text}
              </div>
            )}

            <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400 mb-2">
                <Mail className="w-4 h-4" />
                <span className="text-sm">Email</span>
              </div>
              <p className="text-gray-900 dark:text-white font-medium">
                {user.email}
              </p>
            </div>

            <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400 mb-2">
                <Lock className="w-4 h-4" />
                <span className="text-sm">Account Status</span>
              </div>
              <div className="flex items-center gap-2">
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  user.emailVerified
                    ? 'bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-200'
                    : 'bg-yellow-100 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-200'
                }`}>
                  {user.emailVerified ? '✓ Verified' : '⚠ Not Verified'}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Profile Badge */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <User className="w-6 h-6" />
          Profile Badge
        </h2>
        
        <div className="grid grid-cols-4 md:grid-cols-8 gap-3">
          {AVAILABLE_BADGES.map(badge => {
            const canUse = canUseBadge(badge);
            const isSelected = selectedBadge === badge;
            
            return (
              <button
                key={badge}
                onClick={() => canUse && onBadgeChange(badge)}
                disabled={!canUse}
                className={`
                  text-4xl p-4 rounded-lg border-2 transition-all
                  ${isSelected 
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 scale-110' 
                    : canUse
                    ? 'border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500 hover:scale-105'
                    : 'border-gray-200 dark:border-gray-700 opacity-30 cursor-not-allowed'
                  }
                `}
              >
                {badge}
                {!canUse && <div className="text-xs mt-1">🔒</div>}
              </button>
            );
          })}
        </div>
      </div>

      {/* Theme Selection */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <Palette className="w-6 h-6" />
          Accent Theme
        </h2>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {AVAILABLE_THEMES.map(theme => {
            const canUse = canUseTheme(theme.id);
            const isSelected = selectedTheme === theme.id;
            
            return (
              <button
                key={theme.id}
                onClick={() => canUse && onThemeChange(theme.id)}
                disabled={!canUse}
                className={`
                  relative p-4 rounded-lg border-2 transition-all
                  ${isSelected 
                    ? 'border-gray-900 dark:border-white scale-105' 
                    : canUse
                    ? 'border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500'
                    : 'border-gray-200 dark:border-gray-700 opacity-50 cursor-not-allowed'
                  }
                `}
              >
                <div className={`w-full h-16 rounded-lg bg-gradient-to-r ${theme.colors} mb-2`} />
                <p className="font-medium text-gray-900 dark:text-white">{theme.name}</p>
                {!canUse && (
                  <div className="absolute top-2 right-2 text-xl">🔒</div>
                )}
              </button>
            );
          })}
        </div>
        
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-4">
          💎 Purchase additional themes in the Gem Shop
        </p>
      </div>

      {/* Daily Goal */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <Bell className="w-6 h-6" />
          Daily XP Goal
        </h2>
        
        <div className="flex gap-3">
          {[50, 100, 200, 300].map(goal => (
            <button
              key={goal}
              onClick={() => onDailyGoalChange(goal)}
              className={`
                flex-1 py-3 px-4 rounded-lg font-bold transition-all
                ${dailyGoal === goal
                  ? 'bg-blue-500 text-white scale-105 shadow-lg'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                }
              `}
            >
              {goal} XP
            </button>
          ))}
        </div>
      </div>

      {/* Notification Preferences */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
          Preferences
        </h2>
        
        <div className="space-y-3">
          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              defaultChecked={true}
              className="w-5 h-5 rounded border-gray-300 text-blue-500 focus:ring-blue-500"
            />
            <span className="text-gray-700 dark:text-gray-300">
              Show XP animations
            </span>
          </label>
          
          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              defaultChecked={true}
              className="w-5 h-5 rounded border-gray-300 text-blue-500 focus:ring-blue-500"
            />
            <span className="text-gray-700 dark:text-gray-300">
              Show confetti on perfect scores
            </span>
          </label>
          
          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              defaultChecked={true}
              className="w-5 h-5 rounded border-gray-300 text-blue-500 focus:ring-blue-500"
            />
            <span className="text-gray-700 dark:text-gray-300">
              Show daily motivational quote
            </span>
          </label>
        </div>
      </div>

      {/* AI Question Generation Settings */}
      {user && onAISettingsUpdate && onAICostTrackingUpdate && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <Sparkles className="w-6 h-6" />
            AI Question Generation
          </h2>
          <AISettingsComponent
            aiSettings={aiSettings}
            aiCostTracking={aiCostTracking}
            onUpdate={onAISettingsUpdate}
            onCostUpdate={onAICostTrackingUpdate}
          />
        </div>
      )}

      {/* Danger Zone - Reset All Progress */}
      <div className="bg-red-50 dark:bg-red-900/20 rounded-lg shadow-sm border-2 border-red-200 dark:border-red-800 p-6">
        <h2 className="text-xl font-bold text-red-900 dark:text-red-300 mb-2 flex items-center gap-2">
          <SettingsIcon className="w-5 h-5" />
          Danger Zone
        </h2>
        <p className="text-sm text-red-700 dark:text-red-400 mb-4">
          This action cannot be undone. All your progress, XP, gems, achievements, and purchased items will be permanently deleted.
        </p>
        <button
          onClick={() => {
            const confirmed = window.confirm(
              'Are you sure you want to reset ALL progress?\n\n' +
              'This will delete:\n' +
              '• All topic progress and mastery\n' +
              '• XP, level, and ranking\n' +
              '• Gems and purchased items\n' +
              '• Achievements and streaks\n' +
              '• Consumable inventory\n\n' +
              'This action CANNOT be undone!'
            );
            if (confirmed) {
              const doubleConfirm = window.confirm(
                '⚠️ FINAL WARNING ⚠️\n\n' +
                'This will permanently delete all your progress!\n\n' +
                'Are you absolutely sure?'
              );
              if (doubleConfirm) {
                onResetAllProgress();
              }
            }
          }}
          className="bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-lg font-semibold transition-colors flex items-center gap-2 shadow-md hover:shadow-lg"
        >
          <SettingsIcon className="w-5 h-5" />
          Reset All Progress
        </button>
      </div>
    </div>
  );
}
