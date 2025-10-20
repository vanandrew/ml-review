import React, { useState, useEffect } from 'react';
import { AISettings as AISettingsType, AICostTracking } from '../types';
import { useAuth } from '../contexts/AuthContext';
import { saveAPIKey, loadAPIKey, deleteAPIKey } from '../utils/aiFirestore';
import { CostTracker } from '../utils/aiCostTracking';

interface AISettingsProps {
  aiSettings: AISettingsType | undefined;
  aiCostTracking: AICostTracking | undefined;
  onUpdate: (settings: AISettingsType) => void;
  onCostUpdate: (tracking: AICostTracking) => void;
}

export const AISettingsComponent: React.FC<AISettingsProps> = ({
  aiSettings,
  aiCostTracking,
  onUpdate,
  onCostUpdate,
}) => {
  const { user } = useAuth();
  const [apiKey, setApiKey] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [hasStoredKey, setHasStoredKey] = useState(false);

  const settings: AISettingsType = aiSettings || {
    provider: null,
    preferences: {
      questionDifficulty: 'intermediate',
    },
  };

  const tracking: AICostTracking = aiCostTracking || {
    dailySpend: 0,
    monthlySpend: 0,
    questionsGeneratedToday: 0,
    lastResetDate: new Date().toISOString().split('T')[0],
  };

  // Load API key status on mount
  useEffect(() => {
    const checkStoredKey = async () => {
      if (user) {
        const { apiKey: storedKey } = await loadAPIKey(user.uid);
        setHasStoredKey(!!storedKey);
      }
    };
    checkStoredKey();
  }, [user]);

  const handleSaveAPIKey = async () => {
    if (!user) {
      setMessage({ type: 'error', text: 'Please sign in to save API key' });
      return;
    }

    if (!apiKey.trim()) {
      setMessage({ type: 'error', text: 'Please enter an API key' });
      return;
    }

    setIsLoading(true);
    setMessage(null);

    try {
      await saveAPIKey(user.uid, 'claude', apiKey);

      // Update settings (provider set automatically enables AI questions)
      onUpdate({
        ...settings,
        provider: 'claude',
      });

      setMessage({ type: 'success', text: 'API key saved successfully!' });
      setHasStoredKey(true);
      setApiKey(''); // Clear input after saving
      setShowApiKey(false);
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to save API key. Please try again.' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteAPIKey = async () => {
    if (!user) return;

    if (!confirm('Are you sure you want to delete your API key? This will disable AI question generation.')) {
      return;
    }

    setIsLoading(true);
    setMessage(null);

    try {
      await deleteAPIKey(user.uid);

      // Update settings (removing provider disables AI questions)
      onUpdate({
        ...settings,
        provider: null,
      });

      setMessage({ type: 'success', text: 'API key deleted successfully' });
      setHasStoredKey(false);
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to delete API key' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDifficultyChange = (difficulty: 'beginner' | 'intermediate' | 'advanced') => {
    onUpdate({
      ...settings,
      preferences: {
        ...settings.preferences,
        questionDifficulty: difficulty,
      },
    });
  };

  // Cost tracking (no limits, just display)
  const costTracker = new CostTracker('claude', tracking);

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
          AI Question Generation
        </h3>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          Enable AI-powered question generation using your own API key. Fresh questions generated every time.
        </p>
      </div>

      {/* API Key Management */}
      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="font-medium text-gray-900 dark:text-white">API Key Status</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {hasStoredKey ? '✓ API key saved (encrypted)' : '○ No API key saved'}
            </p>
          </div>
          {hasStoredKey && (
            <button
              onClick={handleDeleteAPIKey}
              disabled={isLoading}
              className="px-3 py-1 text-sm bg-red-500 text-white rounded hover:bg-red-600 disabled:opacity-50"
            >
              Delete Key
            </button>
          )}
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Claude API Key
          </label>
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <input
                type={showApiKey ? 'text' : 'password'}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="sk-ant-..."
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
              />
              <button
                type="button"
                onClick={() => setShowApiKey(!showApiKey)}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
              >
                {showApiKey ? '👁️' : '👁️‍🗨️'}
              </button>
            </div>
            <button
              onClick={handleSaveAPIKey}
              disabled={isLoading || !apiKey.trim()}
              className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? 'Saving...' : 'Save'}
            </button>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Get your API key from{' '}
            <a
              href="https://console.anthropic.com/settings/keys"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-500 hover:underline"
            >
              console.anthropic.com
            </a>
          </p>
        </div>
      </div>

      {/* Preferences */}
      {hasStoredKey && (
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Question Difficulty
            </label>
            <div className="flex gap-2">
              {(['beginner', 'intermediate', 'advanced'] as const).map((diff) => (
                <button
                  key={diff}
                  onClick={() => handleDifficultyChange(diff)}
                  className={`px-4 py-2 rounded-lg capitalize ${
                    settings.preferences.questionDifficulty === diff
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                  }`}
                >
                  {diff}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Cost Tracking */}
      {hasStoredKey && (
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 space-y-2">
          <h4 className="font-medium text-gray-900 dark:text-white">Usage Tracking</h4>
          <div className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
            <div className="flex justify-between">
              <span>Today's spend:</span>
              <span className="font-medium">${tracking.dailySpend.toFixed(3)}</span>
            </div>
            <div className="flex justify-between">
              <span>This month:</span>
              <span className="font-medium">${tracking.monthlySpend.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span>Questions generated today:</span>
              <span className="font-medium">{tracking.questionsGeneratedToday}</span>
            </div>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
            Estimated cost: ~$0.003 per question
          </p>
        </div>
      )}

      {/* Messages */}
      {message && (
        <div
          className={`p-3 rounded-lg ${
            message.type === 'success'
              ? 'bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-300'
              : 'bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-300'
          }`}
        >
          {message.text}
        </div>
      )}

      {/* Info Box */}
      <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 text-sm text-gray-600 dark:text-gray-400">
        <h5 className="font-medium text-gray-900 dark:text-white mb-2">How it works:</h5>
        <ul className="space-y-1 list-disc list-inside">
          <li>Your API key is stored encrypted in Firestore</li>
          <li>Fresh questions generated every quiz attempt</li>
          <li>All questions are AI-generated for maximum variety</li>
          <li>Estimated cost: ~$0.003 per question</li>
        </ul>
      </div>
    </div>
  );
};

export default AISettingsComponent;
