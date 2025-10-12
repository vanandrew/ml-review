import { useState } from 'react';
import { GEM_SHOP_ITEMS, canAfford } from '../utils/gems';
import { Gem, ShoppingCart, Check, Zap, Package, Unlock, Sparkles, Crown, Users } from 'lucide-react';
import { ConsumableInventory as Inventory, ActivePowerUp } from '../types';
import ConsumableInventory from './ConsumableInventory';

interface GemShopProps {
  currentGems: number;
  purchasedItems: string[];
  consumableInventory: Inventory;
  activePowerUps: ActivePowerUp[];
  onPurchase: (itemId: string) => void;
  selectedTheme?: string;
}

const THEME_GRADIENTS: Record<string, string> = {
  'default': 'from-purple-500 to-pink-500',
  'theme-ocean': 'from-blue-400 to-cyan-500',
  'theme-forest': 'from-green-400 to-emerald-600',
  'theme-sunset': 'from-orange-400 to-pink-500',
  'theme-midnight': 'from-indigo-600 to-purple-700',
  'theme-cherry': 'from-pink-400 to-rose-500',
};

const CATEGORY_INFO = {
  all: { 
    icon: ShoppingCart, 
    label: 'All Items',
    activeClass: 'bg-blue-500 text-white shadow-md',
    inactiveClass: 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
  },
  consumable: { 
    icon: Package, 
    label: 'Consumables',
    activeClass: 'bg-green-500 text-white shadow-md',
    inactiveClass: 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
  },
  'power-up': { 
    icon: Zap, 
    label: 'Power-Ups',
    activeClass: 'bg-yellow-500 text-white shadow-md',
    inactiveClass: 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
  },
  unlock: { 
    icon: Unlock, 
    label: 'Unlocks',
    activeClass: 'bg-purple-500 text-white shadow-md',
    inactiveClass: 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
  },
  cosmetic: { 
    icon: Sparkles, 
    label: 'Cosmetics',
    activeClass: 'bg-pink-500 text-white shadow-md',
    inactiveClass: 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
  },
  premium: { 
    icon: Crown, 
    label: 'Premium',
    activeClass: 'bg-amber-500 text-white shadow-md',
    inactiveClass: 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
  },
  social: { 
    icon: Users, 
    label: 'Social',
    activeClass: 'bg-cyan-500 text-white shadow-md',
    inactiveClass: 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
  },
};

export default function GemShop({ currentGems, purchasedItems, consumableInventory, activePowerUps, onPurchase, selectedTheme = 'default' }: GemShopProps) {
  const [selectedCategory, setSelectedCategory] = useState<'all' | 'consumable' | 'power-up' | 'unlock' | 'cosmetic' | 'premium' | 'social'>('all');
  const gradientColors = THEME_GRADIENTS[selectedTheme] || THEME_GRADIENTS['default'];

  // Provide default values for consumableInventory if it's undefined (for legacy users)
  const inventory = consumableInventory || {
    hints: 0,
    streakFreezes: 0,
    xpBoosts: 0,
    knowledgePotions: 0,
    timeExtensions: 0,
    secondChances: 0,
    extraLives: 0,
    multiplierBoosts: 0,
  };

  const filteredItems = GEM_SHOP_ITEMS.filter(item => 
    selectedCategory === 'all' || item.category === selectedCategory
  );

  const handlePurchase = (itemId: string, cost: number) => {
    if (canAfford(currentGems, cost)) {
      onPurchase(itemId);
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
          <ShoppingCart className="w-7 h-7" />
          Gem Shop
        </h2>
        
        <div className={`flex items-center gap-2 bg-gradient-to-r ${gradientColors} text-white px-4 py-2 rounded-lg font-bold shadow-md`}>
          <Gem className="w-5 h-5" />
          {currentGems} Gems
        </div>
      </div>

      {/* Inventory Display */}
      <ConsumableInventory 
        inventory={inventory}
        activePowerUps={activePowerUps}
      />

      {/* Category Filter */}
      <div className="flex flex-wrap gap-2 mb-6">
        {Object.entries(CATEGORY_INFO).map(([key, info]) => {
          const Icon = info.icon;
          const isSelected = selectedCategory === key;
          return (
            <button
              key={key}
              onClick={() => setSelectedCategory(key as any)}
              className={`px-4 py-2 rounded-lg font-medium transition-all flex items-center gap-2 ${
                isSelected
                  ? info.activeClass
                  : info.inactiveClass
              }`}
            >
              <Icon className="w-4 h-4" />
              {info.label}
            </button>
          );
        })}
      </div>

      {/* Shop Items Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredItems.map(item => {
          const isPurchased = !item.consumable && purchasedItems.includes(item.id);
          const canBuy = canAfford(currentGems, item.cost);
          const isConsumable = item.consumable;
          
          // Get inventory count for consumables
          let inventoryCount = 0;
          if (isConsumable) {
            if (item.id === 'hint-pack') inventoryCount = inventory.hints;
            else if (item.id === 'streak-freeze') inventoryCount = inventory.streakFreezes;
            else if (item.id === 'xp-boost') inventoryCount = Math.floor(inventory.xpBoosts / 3);
            else if (item.id === 'knowledge-potion') inventoryCount = Math.floor(inventory.knowledgePotions / 5);
            else if (item.id === 'time-extension') inventoryCount = inventory.timeExtensions;
            else if (item.id === 'second-chance') inventoryCount = inventory.secondChances;
            else if (item.id === 'extra-life') inventoryCount = inventory.extraLives;
            else if (item.id === 'multiplier-boost') inventoryCount = inventory.multiplierBoosts;
          }
          
          return (
            <div
              key={item.id}
              className={`bg-gradient-to-br rounded-lg p-4 border-2 transition-all hover:scale-105 ${
                item.category === 'premium'
                  ? 'from-amber-50 to-yellow-100 dark:from-amber-900/30 dark:to-yellow-900/30 border-amber-300 dark:border-amber-700'
                  : item.category === 'power-up'
                  ? 'from-yellow-50 to-orange-100 dark:from-yellow-900/30 dark:to-orange-900/30 border-yellow-300 dark:border-yellow-700'
                  : item.category === 'unlock'
                  ? 'from-purple-50 to-pink-100 dark:from-purple-900/30 dark:to-pink-900/30 border-purple-300 dark:border-purple-700'
                  : 'from-gray-50 to-gray-100 dark:from-gray-700 dark:to-gray-800 border-gray-200 dark:border-gray-600'
              }`}
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2">
                  <span className="text-4xl">{item.icon}</span>
                  {isConsumable && inventoryCount > 0 && (
                    <div className="bg-blue-500 text-white rounded-full px-2 py-1 text-xs font-bold">
                      {inventoryCount}×
                    </div>
                  )}
                </div>
                {isPurchased && (
                  <div className="bg-green-500 text-white rounded-full p-1">
                    <Check className="w-4 h-4" />
                  </div>
                )}
              </div>
              
              <h3 className="font-bold text-lg text-gray-900 dark:text-white mb-1">
                {item.name}
              </h3>
              
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                {item.description}
              </p>
              
              {item.duration && (
                <p className="text-xs text-blue-600 dark:text-blue-400 mb-2">
                  ⏱️ Duration: {item.duration >= 24 ? `${item.duration / 24} days` : `${item.duration} hours`}
                </p>
              )}
              
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1 font-bold text-purple-600 dark:text-purple-400">
                  <Gem className="w-4 h-4" />
                  {item.cost}
                </span>
                
                <button
                  onClick={() => handlePurchase(item.id, item.cost)}
                  disabled={(!isConsumable && isPurchased) || !canBuy}
                  className={`px-4 py-2 rounded-lg font-medium transition-all ${
                    !isConsumable && isPurchased
                      ? 'bg-green-500 text-white cursor-default'
                      : canBuy
                      ? 'bg-blue-500 text-white hover:bg-blue-600 hover:scale-105'
                      : 'bg-gray-300 dark:bg-gray-600 text-gray-500 dark:text-gray-400 cursor-not-allowed'
                  }`}
                >
                  {!isConsumable && isPurchased ? 'Owned' : canBuy ? 'Buy' : 'Not enough'}
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {filteredItems.length === 0 && (
        <div className="text-center py-12 text-gray-500 dark:text-gray-400">
          <p>No items in this category</p>
        </div>
      )}

      {/* How to Earn Gems Info */}
      <div className="mt-8 bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
        <h3 className="font-bold text-blue-900 dark:text-blue-300 mb-2 flex items-center gap-2">
          <Gem className="w-5 h-5" />
          How to Earn Gems
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-1">
          <ul className="text-sm text-blue-800 dark:text-blue-400 space-y-1">
            <li>• Daily login: <strong>2 gems</strong></li>
            <li>• Perfect quiz: <strong>5 gems</strong></li>
            <li>• Quiz completion: <strong>1 gem</strong></li>
            <li>• First time topic: <strong>3 gems</strong></li>
          </ul>
          <ul className="text-sm text-blue-800 dark:text-blue-400 space-y-1">
            <li>• Unlock achievement: <strong>10 gems</strong></li>
            <li>• Complete weekly challenge: <strong>25 gems</strong></li>
            <li>• Level up: <strong>5 gems</strong></li>
            <li>• Mastery milestones: <strong>15 gems</strong></li>
          </ul>
        </div>
      </div>
    </div>
  );
}
