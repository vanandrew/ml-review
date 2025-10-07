import { useState } from 'react';
import { GEM_SHOP_ITEMS, canAfford } from '../utils/gems';
import { Gem, ShoppingCart, Check } from 'lucide-react';

interface GemShopProps {
  currentGems: number;
  purchasedItems: string[];
  onPurchase: (itemId: string) => void;
  selectedTheme?: string;
}

const THEME_GRADIENTS: Record<string, string> = {
  'default': 'from-purple-500 to-pink-500',
  'theme-ocean': 'from-blue-400 to-cyan-500',
  'theme-forest': 'from-green-400 to-emerald-600',
  'theme-sunset': 'from-orange-400 to-pink-500',
};

export default function GemShop({ currentGems, purchasedItems, onPurchase, selectedTheme = 'default' }: GemShopProps) {
  const [selectedCategory, setSelectedCategory] = useState<'all' | 'utility' | 'cosmetic'>('all');
  const gradientColors = THEME_GRADIENTS[selectedTheme] || THEME_GRADIENTS['default'];

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

      {/* Category Filter */}
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setSelectedCategory('all')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            selectedCategory === 'all'
              ? 'bg-blue-500 text-white'
              : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
          }`}
        >
          All Items
        </button>
        <button
          onClick={() => setSelectedCategory('utility')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            selectedCategory === 'utility'
              ? 'bg-green-500 text-white'
              : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
          }`}
        >
          🛠️ Utility
        </button>
        <button
          onClick={() => setSelectedCategory('cosmetic')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            selectedCategory === 'cosmetic'
              ? 'bg-purple-500 text-white'
              : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
          }`}
        >
          ✨ Cosmetic
        </button>
      </div>

      {/* Shop Items Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredItems.map(item => {
          const isPurchased = purchasedItems.includes(item.id);
          const canBuy = canAfford(currentGems, item.cost);
          
          return (
            <div
              key={item.id}
              className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-700 dark:to-gray-800 rounded-lg p-4 border-2 border-gray-200 dark:border-gray-600"
            >
              <div className="flex items-start justify-between mb-3">
                <span className="text-4xl">{item.icon}</span>
                {isPurchased && (
                  <div className="bg-green-500 text-white rounded-full p-1">
                    <Check className="w-4 h-4" />
                  </div>
                )}
              </div>
              
              <h3 className="font-bold text-lg text-gray-900 dark:text-white mb-1">
                {item.name}
              </h3>
              
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                {item.description}
              </p>
              
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1 font-bold text-purple-600 dark:text-purple-400">
                  <Gem className="w-4 h-4" />
                  {item.cost}
                </span>
                
                <button
                  onClick={() => handlePurchase(item.id, item.cost)}
                  disabled={isPurchased || !canBuy}
                  className={`px-4 py-2 rounded-lg font-medium transition-all ${
                    isPurchased
                      ? 'bg-green-500 text-white cursor-default'
                      : canBuy
                      ? 'bg-blue-500 text-white hover:bg-blue-600 hover:scale-105'
                      : 'bg-gray-300 dark:bg-gray-600 text-gray-500 dark:text-gray-400 cursor-not-allowed'
                  }`}
                >
                  {isPurchased ? 'Owned' : canBuy ? 'Buy' : 'Not enough gems'}
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
        <h3 className="font-bold text-blue-900 dark:text-blue-300 mb-2">
          💎 How to Earn Gems
        </h3>
        <ul className="text-sm text-blue-800 dark:text-blue-400 space-y-1">
          <li>• Daily login: 1 gem</li>
          <li>• Perfect quiz: 2 gems</li>
          <li>• Unlock achievement: 5 gems</li>
          <li>• Complete weekly challenge: 10 gems</li>
          <li>• Level up: 3 gems</li>
        </ul>
      </div>
    </div>
  );
}
