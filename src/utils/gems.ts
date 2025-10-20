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
  category: 'consumable' | 'cosmetic';
  available: boolean;
  consumable?: boolean; // Can be purchased multiple times
  quantity?: number; // For stackable items
}

export const GEM_SHOP_ITEMS: GemShopItem[] = [
  // === CONSUMABLES === (Essential gameplay items)
  {
    id: 'hint-pack',
    name: 'Hint Pack',
    description: 'Get 5 hints to use on any quiz questions',
    cost: 15,
    icon: '💡',
    category: 'consumable',
    available: true,
    consumable: true,
    quantity: 5,
  },
  {
    id: 'streak-freeze',
    name: 'Streak Freeze',
    description: 'Protect your streak for one missed day',
    cost: 10,
    icon: '🧊',
    category: 'consumable',
    available: true,
    consumable: true,
    quantity: 1,
  },

  // === COSMETICS === (Themes & Badges)
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
    id: 'theme-midnight',
    name: 'Midnight Theme',
    description: 'Deep purple midnight color scheme',
    cost: 15,
    icon: '🌙',
    category: 'cosmetic',
    available: true,
  },
  {
    id: 'theme-cherry',
    name: 'Cherry Blossom Theme',
    description: 'Soft pink cherry blossom colors',
    cost: 15,
    icon: '🌸',
    category: 'cosmetic',
    available: true,
  },
  {
    id: 'badge-star',
    name: 'Star Badge',
    description: 'Classic star profile badge',
    cost: 20,
    icon: '⭐',
    category: 'cosmetic',
    available: true,
  },
  {
    id: 'badge-rocket',
    name: 'Rocket Badge',
    description: 'Rocket ship profile badge',
    cost: 20,
    icon: '🚀',
    category: 'cosmetic',
    available: true,
  },
  {
    id: 'badge-brain',
    name: 'Brain Badge',
    description: 'Intelligence brain profile badge',
    cost: 20,
    icon: '🧠',
    category: 'cosmetic',
    available: true,
  },
  {
    id: 'badge-trophy',
    name: 'Trophy Badge',
    description: 'Championship trophy badge',
    cost: 20,
    icon: '🏆',
    category: 'cosmetic',
    available: true,
  },
  {
    id: 'badge-fire',
    name: 'Fire Badge',
    description: 'On fire streak badge',
    cost: 20,
    icon: '🔥',
    category: 'cosmetic',
    available: true,
  },
  {
    id: 'badge-crown',
    name: 'Crown Badge',
    description: 'Royal crown badge',
    cost: 25,
    icon: '👑',
    category: 'cosmetic',
    available: true,
  },
];

/**
 * Calculate gems earned for an action
 */
export function calculateGemsEarned(action: 'daily-login' | 'perfect-quiz' | 'achievement' | 'weekly-challenge' | 'level-up' | 'quiz-completion' | 'first-time-topic' | 'mastery-milestone'): number {
  switch (action) {
    case 'daily-login':
      return 2;
    case 'perfect-quiz':
      return 5;
    case 'quiz-completion':
      return 1;
    case 'achievement':
      return 10;
    case 'weekly-challenge':
      return 25;
    case 'level-up':
      return 5;
    case 'first-time-topic':
      return 3;
    case 'mastery-milestone':
      return 15;
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

// ============================================
// CONSUMABLES MANAGEMENT
// ============================================

import { ConsumableInventory } from '../types';

/**
 * Add consumable to inventory
 */
export function addConsumable(
  inventory: ConsumableInventory,
  itemId: string,
  quantity: number = 1
): ConsumableInventory {
  const updated = { ...inventory };

  switch (itemId) {
    case 'hint-pack':
      updated.hints += quantity * 5; // 5 hints per pack
      break;
    case 'streak-freeze':
      updated.streakFreezes += quantity;
      break;
  }

  return updated;
}

/**
 * Use consumable from inventory
 */
export function useConsumable(
  inventory: ConsumableInventory,
  itemType: keyof ConsumableInventory
): { success: boolean; inventory: ConsumableInventory } {
  if (inventory[itemType] <= 0) {
    return { success: false, inventory };
  }

  const updated = { ...inventory };
  updated[itemType]--;

  return { success: true, inventory: updated };
}
