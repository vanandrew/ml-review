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
  category: 'consumable' | 'power-up' | 'unlock' | 'cosmetic' | 'premium' | 'social';
  available: boolean;
  consumable?: boolean; // Can be purchased multiple times
  duration?: number; // For time-based items (in hours)
  quantity?: number; // For stackable items
}

export const GEM_SHOP_ITEMS: GemShopItem[] = [
  // === CONSUMABLE POWER-UPS === (Repeatable purchases)
  {
    id: 'xp-boost',
    name: 'XP Boost',
    description: '2x XP for your next 3 quizzes',
    cost: 30,
    icon: '⚡',
    category: 'power-up',
    available: true,
    consumable: true,
    quantity: 3,
  },
  {
    id: 'double-gems',
    name: 'Double Gems',
    description: '2x gems earned for 24 hours',
    cost: 25,
    icon: '💎',
    category: 'power-up',
    available: true,
    consumable: true,
    duration: 24,
  },
  {
    id: 'time-extension',
    name: 'Time Extension',
    description: '+10 seconds per question in timed challenges',
    cost: 15,
    icon: '⏱️',
    category: 'power-up',
    available: true,
    consumable: true,
    quantity: 1,
  },
  {
    id: 'second-chance',
    name: 'Second Chance',
    description: 'Retry a quiz immediately with new questions',
    cost: 20,
    icon: '🔄',
    category: 'power-up',
    available: true,
    consumable: true,
    quantity: 1,
  },
  {
    id: 'knowledge-potion',
    name: 'Knowledge Potion',
    description: 'Eliminate 2 wrong answers in your next 5 questions',
    cost: 40,
    icon: '🧪',
    category: 'power-up',
    available: true,
    consumable: true,
    quantity: 5,
  },
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
  {
    id: 'extra-life',
    name: 'Extra Life',
    description: 'Continue from where you failed in challenge mode',
    cost: 10,
    icon: '❤️',
    category: 'consumable',
    available: true,
    consumable: true,
    quantity: 1,
  },
  {
    id: 'multiplier-boost',
    name: 'Multiplier Boost',
    description: '+0.5x XP multiplier for one challenge attempt',
    cost: 35,
    icon: '🚀',
    category: 'power-up',
    available: true,
    consumable: true,
    quantity: 1,
  },

  // === PROGRESSION UNLOCKS === (One-time functional purchases)
  {
    id: 'custom-quiz-maker',
    name: 'Custom Quiz Maker',
    description: 'Create custom quizzes mixing any topics',
    cost: 100,
    icon: '🎯',
    category: 'unlock',
    available: true,
  },
  {
    id: 'advanced-analytics',
    name: 'Advanced Analytics',
    description: 'Unlock detailed performance charts & insights',
    cost: 75,
    icon: '📊',
    category: 'unlock',
    available: true,
  },
  {
    id: 'review-optimizer',
    name: 'Review Optimizer',
    description: 'AI-powered review schedule recommendations',
    cost: 60,
    icon: '🤖',
    category: 'unlock',
    available: true,
  },
  {
    id: 'flashcard-generator',
    name: 'Flashcard Generator',
    description: 'Generate printable flashcards for any topic',
    cost: 50,
    icon: '�',
    category: 'unlock',
    available: true,
  },
  {
    id: 'topic-summary',
    name: 'Topic Summary Pro',
    description: 'Get AI-generated summaries for any topic',
    cost: 45,
    icon: '📝',
    category: 'unlock',
    available: true,
  },

  // === BASIC COSMETICS === (Themes & Badges)
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
    icon: '�',
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

  // === PREMIUM COSMETICS === (Higher tier)
  {
    id: 'theme-animated-galaxy',
    name: 'Animated Galaxy',
    description: 'Stunning animated galaxy theme with particles',
    cost: 50,
    icon: '🌌',
    category: 'premium',
    available: true,
  },
  {
    id: 'theme-animated-aurora',
    name: 'Animated Aurora',
    description: 'Beautiful northern lights animation',
    cost: 50,
    icon: '🌈',
    category: 'premium',
    available: true,
  },
  {
    id: 'theme-animated-matrix',
    name: 'Matrix Theme',
    description: 'Animated matrix code rain effect',
    cost: 50,
    icon: '🖥️',
    category: 'premium',
    available: true,
  },
  {
    id: 'victory-fireworks',
    name: 'Fireworks Victory',
    description: 'Epic fireworks celebration for perfect scores',
    cost: 30,
    icon: '🎆',
    category: 'premium',
    available: true,
  },
  {
    id: 'victory-sparkles',
    name: 'Sparkle Victory',
    description: 'Magical sparkles for perfect scores',
    cost: 30,
    icon: '✨',
    category: 'premium',
    available: true,
  },
  {
    id: 'profile-banner-neural',
    name: 'Neural Network Banner',
    description: 'Animated neural network background',
    cost: 45,
    icon: '🧬',
    category: 'premium',
    available: true,
  },
  {
    id: 'profile-banner-circuit',
    name: 'Circuit Board Banner',
    description: 'Tech-inspired circuit board background',
    cost: 45,
    icon: '⚡',
    category: 'premium',
    available: true,
  },
  {
    id: 'sound-pack-retro',
    name: 'Retro Sound Pack',
    description: '8-bit style sounds for quiz interactions',
    cost: 35,
    icon: '🎮',
    category: 'premium',
    available: true,
  },
  {
    id: 'sound-pack-zen',
    name: 'Zen Sound Pack',
    description: 'Calming zen sounds for focus',
    cost: 35,
    icon: '🎵',
    category: 'premium',
    available: true,
  },

  // === SUBSCRIPTION-STYLE PERKS ===
  {
    id: 'premium-week',
    name: 'Premium Week Pass',
    description: 'All power-ups + 10% XP boost for 7 days',
    cost: 150,
    icon: '🎫',
    category: 'premium',
    available: true,
    consumable: true,
    duration: 168, // 7 days in hours
  },
  {
    id: 'scholars-blessing',
    name: "Scholar's Blessing",
    description: '+10% XP permanently + 5 bonus gems daily for 30 days',
    cost: 200,
    icon: '📚',
    category: 'premium',
    available: true,
    consumable: true,
    duration: 720, // 30 days
  },
  {
    id: 'vip-badge',
    name: 'VIP Status',
    description: 'Exclusive animated VIP badge + all cosmetics unlocked',
    cost: 300,
    icon: '💫',
    category: 'premium',
    available: true,
  },
];

/**
 * Calculate gems earned for an action
 */
export function calculateGemsEarned(action: 'daily-login' | 'perfect-quiz' | 'achievement' | 'weekly-challenge' | 'level-up' | 'quiz-completion' | 'first-time-topic' | 'mastery-milestone'): number {
  switch (action) {
    case 'daily-login':
      return 2; // Increased from 1
    case 'perfect-quiz':
      return 5; // Increased from 2
    case 'quiz-completion':
      return 1; // New: even non-perfect quizzes earn gems
    case 'achievement':
      return 10; // Increased from 5
    case 'weekly-challenge':
      return 25; // Increased from 10
    case 'level-up':
      return 5; // Increased from 3
    case 'first-time-topic':
      return 3; // New: bonus for trying new topics
    case 'mastery-milestone':
      return 15; // New: bonus for mastering 5/10/20 topics
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
// CONSUMABLES & POWER-UPS MANAGEMENT
// ============================================

import { ConsumableInventory, ActivePowerUp } from '../types';

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
      updated.hints += quantity;
      break;
    case 'streak-freeze':
      updated.streakFreezes += quantity;
      break;
    case 'xp-boost':
      updated.xpBoosts += quantity * 3; // 3 quizzes per purchase
      break;
    case 'knowledge-potion':
      updated.knowledgePotions += quantity * 5; // 5 questions per purchase
      break;
    case 'time-extension':
      updated.timeExtensions += quantity;
      break;
    case 'second-chance':
      updated.secondChances += quantity;
      break;
    case 'extra-life':
      updated.extraLives += quantity;
      break;
    case 'multiplier-boost':
      updated.multiplierBoosts += quantity;
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

/**
 * Activate time-based power-up
 */
export function activatePowerUp(
  activePowerUps: ActivePowerUp[],
  itemId: string,
  durationHours: number,
  remaining?: number
): ActivePowerUp[] {
  const now = new Date();
  const expiresAt = new Date(now.getTime() + durationHours * 60 * 60 * 1000);
  
  let type: ActivePowerUp['type'];
  switch (itemId) {
    case 'double-gems':
      type = 'double-gems';
      break;
    case 'premium-week':
      type = 'premium-week';
      break;
    case 'scholars-blessing':
      type = 'scholars-blessing';
      break;
    case 'xp-boost':
      type = 'xp-boost';
      break;
    default:
      return activePowerUps;
  }
  
  const newPowerUp: ActivePowerUp = {
    id: `${itemId}-${Date.now()}`,
    type,
    activatedAt: now,
    expiresAt,
    remaining,
  };
  
  return [...activePowerUps, newPowerUp];
}

/**
 * Clean up expired power-ups
 */
export function cleanExpiredPowerUps(activePowerUps: ActivePowerUp[]): ActivePowerUp[] {
  // Handle undefined or null activePowerUps (for legacy users)
  if (!activePowerUps || !Array.isArray(activePowerUps)) {
    return [];
  }
  
  const now = new Date();
  return activePowerUps.filter(powerUp => {
    // Remove if expired
    if (new Date(powerUp.expiresAt) <= now) {
      return false;
    }
    // Remove count-based power-ups that are depleted
    if (powerUp.remaining !== undefined && powerUp.remaining <= 0) {
      return false;
    }
    return true;
  });
}

/**
 * Check if specific power-up is active
 */
export function hasPowerUp(
  activePowerUps: ActivePowerUp[],
  type: ActivePowerUp['type']
): boolean {
  const cleaned = cleanExpiredPowerUps(activePowerUps);
  return cleaned.some(powerUp => powerUp.type === type);
}

/**
 * Use one charge of a count-based power-up (like xp-boost)
 */
export function usePowerUpCharge(
  activePowerUps: ActivePowerUp[],
  type: ActivePowerUp['type']
): ActivePowerUp[] {
  return activePowerUps.map(powerUp => {
    if (powerUp.type === type && powerUp.remaining !== undefined && powerUp.remaining > 0) {
      return { ...powerUp, remaining: powerUp.remaining - 1 };
    }
    return powerUp;
  });
}

/**
 * Calculate XP multiplier from active power-ups
 */
export function calculateXPMultiplier(activePowerUps: ActivePowerUp[]): number {
  const cleaned = cleanExpiredPowerUps(activePowerUps);
  let multiplier = 1.0;
  
  // XP Boost gives 2x
  if (cleaned.some(p => p.type === 'xp-boost' && (p.remaining === undefined || p.remaining > 0))) {
    multiplier *= 2.0;
  }
  
  // Premium Week gives 1.1x
  if (cleaned.some(p => p.type === 'premium-week')) {
    multiplier *= 1.1;
  }
  
  // Scholar's Blessing gives 1.1x
  if (cleaned.some(p => p.type === 'scholars-blessing')) {
    multiplier *= 1.1;
  }
  
  return multiplier;
}

/**
 * Calculate gem multiplier from active power-ups
 */
export function calculateGemMultiplier(activePowerUps: ActivePowerUp[]): number {
  const cleaned = cleanExpiredPowerUps(activePowerUps);
  let multiplier = 1.0;
  
  // Double Gems gives 2x
  if (cleaned.some(p => p.type === 'double-gems')) {
    multiplier *= 2.0;
  }
  
  return multiplier;
}

/**
 * Check if user has premium perks active
 */
export function hasPremiumPerks(activePowerUps: ActivePowerUp[]): boolean {
  const cleaned = cleanExpiredPowerUps(activePowerUps);
  return cleaned.some(p => p.type === 'premium-week' || p.type === 'scholars-blessing');
}

/**
 * Get daily bonus gems from Scholar's Blessing
 */
export function getDailyBonusGems(activePowerUps: ActivePowerUp[]): number {
  const cleaned = cleanExpiredPowerUps(activePowerUps);
  if (cleaned.some(p => p.type === 'scholars-blessing')) {
    return 5;
  }
  return 0;
}
