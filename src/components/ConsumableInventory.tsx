import { Package } from 'lucide-react';
import { ConsumableInventory as Inventory } from '../types';

interface ConsumableInventoryProps {
  inventory: Inventory;
}

export default function ConsumableInventory({ inventory }: ConsumableInventoryProps) {
  const inventoryItems = [
    { key: 'hints', label: 'Hints', icon: '💡', count: inventory.hints },
    { key: 'streakFreezes', label: 'Streak Freezes', icon: '🧊', count: inventory.streakFreezes },
  ].filter(item => item.count > 0);

  if (inventoryItems.length === 0) {
    return null;
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4 mb-6">
      <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
        <Package className="w-5 h-5" />
        Your Inventory
      </h3>

      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        {inventoryItems.map(item => (
          <div
            key={item.key}
            className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 text-center border border-gray-200 dark:border-gray-600"
          >
            <div className="text-3xl mb-2">{item.icon}</div>
            <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              {item.label}
            </div>
            <div className="text-xl font-bold text-blue-600 dark:text-blue-400">
              {item.count}×
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
