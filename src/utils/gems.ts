export interface GemTransaction {
  id: string;
  amount: number;
  type: 'earn' | 'spend';
  reason: string;
  timestamp: Date;
}

export interface GemShopItem {
  id: string;
  name: string;
  description: string;
  cost: number;
  icon: string;
  category: 'utility' | 'cosmetic';
  available: boolean;
}

export const GEM_SHOP_ITEMS: GemShopItem[] = [
  {
    id: 'streak-freeze',
    name: 'Streak Freeze',
    description: 'Protect your streak for one day',
    cost: 10,
    icon: '🧊',
    category: 'utility',
    available: true,
  },
  {
    id: 'hint',
    name: 'Quiz Hint',
    description: 'Reveal a hint for a quiz question',
    cost: 5,
    icon: '💡',
    category: 'utility',
    available: true,
  },
  {
    id: 'theme-ocean',
    name: 'Ocean Theme',
    description: 'Blue ocean-inspired color scheme',
    cost: 15,
    icon: '🌊',
    category: 'cosmetic',
    available: true,
  },
  {
    id: 'theme-forest',
    name: 'Forest Theme',
    description: 'Green forest-inspired color scheme',
    cost: 15,
    icon: '🌲',
    category: 'cosmetic',
    available: true,
  },
  {
    id: 'theme-sunset',
    name: 'Sunset Theme',
    description: 'Warm sunset-inspired color scheme',
    cost: 15,
    icon: '🌅',
    category: 'cosmetic',
    available: true,
  },
  {
    id: 'badge-star',
    name: 'Star Badge',
    description: 'Custom star profile badge',
    cost: 20,
    icon: '⭐',
    category: 'cosmetic',
    available: true,
  },
  {
    id: 'badge-rocket',
    name: 'Rocket Badge',
    description: 'Custom rocket profile badge',
    cost: 20,
    icon: '🚀',
    category: 'cosmetic',
    available: true,
  },
  {
    id: 'badge-brain',
    name: 'Brain Badge',
    description: 'Custom brain profile badge',
    cost: 20,
    icon: '🧠',
    category: 'cosmetic',
    available: true,
  },
];

/**
 * Calculate gems earned for an action
 */
export function calculateGemsEarned(action: 'daily-login' | 'perfect-quiz' | 'achievement' | 'weekly-challenge' | 'level-up'): number {
  switch (action) {
    case 'daily-login':
      return 1;
    case 'perfect-quiz':
      return 2;
    case 'achievement':
      return 5;
    case 'weekly-challenge':
      return 10;
    case 'level-up':
      return 3;
    default:
      return 0;
  }
}

/**
 * Award gems to user
 */
export function awardGems(
  currentGems: number,
  amount: number,
  reason: string,
  transactions: GemTransaction[]
): { gems: number; transactions: GemTransaction[] } {
  const transaction: GemTransaction = {
    id: `gem-${Date.now()}-${Math.random()}`,
    amount,
    type: 'earn',
    reason,
    timestamp: new Date(),
  };
  
  return {
    gems: currentGems + amount,
    transactions: [...transactions, transaction],
  };
}

/**
 * Spend gems from user
 */
export function spendGems(
  currentGems: number,
  amount: number,
  reason: string,
  transactions: GemTransaction[]
): { success: boolean; gems: number; transactions: GemTransaction[] } {
  if (currentGems < amount) {
    return { success: false, gems: currentGems, transactions };
  }
  
  const transaction: GemTransaction = {
    id: `gem-${Date.now()}-${Math.random()}`,
    amount,
    type: 'spend',
    reason,
    timestamp: new Date(),
  };
  
  return {
    success: true,
    gems: currentGems - amount,
    transactions: [...transactions, transaction],
  };
}

/**
 * Check if user can afford an item
 */
export function canAfford(currentGems: number, cost: number): boolean {
  return currentGems >= cost;
}

/**
 * Get shop item by ID
 */
export function getShopItem(itemId: string): GemShopItem | undefined {
  return GEM_SHOP_ITEMS.find(item => item.id === itemId);
}

/**
 * Initialize daily login gems (once per day)
 */
export function checkDailyLoginGems(lastLoginDate: string | null): boolean {
  if (!lastLoginDate) return true;
  
  const today = new Date().toISOString().split('T')[0];
  return lastLoginDate !== today;
}
