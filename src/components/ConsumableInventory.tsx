import { Package, Zap, Clock } from 'lucide-react';
import { ConsumableInventory as Inventory, ActivePowerUp } from '../types';
import { cleanExpiredPowerUps } from '../utils/gems';

interface ConsumableInventoryProps {
  inventory: Inventory;
  activePowerUps: ActivePowerUp[];
}

export default function ConsumableInventory({ inventory, activePowerUps }: ConsumableInventoryProps) {
  const activePowerUpsClean = cleanExpiredPowerUps(activePowerUps);
  
  const inventoryItems = [
    { key: 'hints', label: 'Hints', icon: '💡', count: inventory.hints },
    { key: 'streakFreezes', label: 'Streak Freezes', icon: '🧊', count: inventory.streakFreezes },
    { key: 'xpBoosts', label: 'XP Boosts', icon: '⚡', count: Math.floor(inventory.xpBoosts / 3) },
    { key: 'knowledgePotions', label: 'Knowledge Potions', icon: '🧪', count: Math.floor(inventory.knowledgePotions / 5) },
    { key: 'timeExtensions', label: 'Time Extensions', icon: '⏱️', count: inventory.timeExtensions },
    { key: 'secondChances', label: 'Second Chances', icon: '🔄', count: inventory.secondChances },
    { key: 'extraLives', label: 'Extra Lives', icon: '❤️', count: inventory.extraLives },
    { key: 'multiplierBoosts', label: 'Multiplier Boosts', icon: '🚀', count: inventory.multiplierBoosts },
  ].filter(item => item.count > 0);

  if (inventoryItems.length === 0 && activePowerUpsClean.length === 0) {
    return null;
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4 mb-6">
      <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
        <Package className="w-5 h-5" />
        Your Inventory
      </h3>

      {/* Active Power-Ups */}
      {activePowerUpsClean.length > 0 && (
        <div className="mb-4">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2 flex items-center gap-1">
            <Zap className="w-4 h-4 text-yellow-500" />
            Active Power-Ups
          </h4>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {activePowerUpsClean.map(powerUp => {
              const timeRemaining = new Date(powerUp.expiresAt).getTime() - new Date().getTime();
              const hoursRemaining = Math.ceil(timeRemaining / (1000 * 60 * 60));
              const daysRemaining = Math.floor(hoursRemaining / 24);
              
              let displayName = '';
              let displayIcon = '⚡';
              
              switch (powerUp.type) {
                case 'double-gems':
                  displayName = 'Double Gems';
                  displayIcon = '💎';
                  break;
                case 'premium-week':
                  displayName = 'Premium Week';
                  displayIcon = '🎫';
                  break;
                case 'scholars-blessing':
                  displayName = "Scholar's Blessing";
                  displayIcon = '📚';
                  break;
                case 'xp-boost':
                  displayName = 'XP Boost';
                  displayIcon = '⚡';
                  break;
              }

              return (
                <div
                  key={powerUp.id}
                  className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 border border-yellow-300 dark:border-yellow-700 rounded-lg p-3"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-2xl">{displayIcon}</span>
                      <div>
                        <div className="font-bold text-sm text-gray-900 dark:text-white">
                          {displayName}
                        </div>
                        {powerUp.remaining !== undefined ? (
                          <div className="text-xs text-gray-600 dark:text-gray-400">
                            {powerUp.remaining} uses left
                          </div>
                        ) : (
                          <div className="text-xs text-gray-600 dark:text-gray-400 flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            {daysRemaining > 0 ? `${daysRemaining}d ${hoursRemaining % 24}h` : `${hoursRemaining}h`}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Consumable Items */}
      {inventoryItems.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2 flex items-center gap-1">
            <Package className="w-4 h-4 text-blue-500" />
            Consumables
          </h4>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {inventoryItems.map(item => (
              <div
                key={item.key}
                className="bg-gray-50 dark:bg-gray-700 rounded-lg p-2 text-center border border-gray-200 dark:border-gray-600"
              >
                <div className="text-2xl mb-1">{item.icon}</div>
                <div className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                  {item.label}
                </div>
                <div className="text-lg font-bold text-blue-600 dark:text-blue-400">
                  {item.count}×
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
