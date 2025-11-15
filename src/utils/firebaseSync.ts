import { doc, getDoc, setDoc, updateDoc, serverTimestamp } from 'firebase/firestore';
import { db } from '../config/firebase';
import { UserProgress, GamificationData, User } from '../types';

/**
 * Migrate local storage data to Firestore on first login
 */
export const migrateLocalDataToFirestore = async (userId: string): Promise<void> => {
  try {
    // Check if user already has data in Firestore
    const userDocRef = doc(db, 'users', userId);
    const userDoc = await getDoc(userDocRef);

    // If user data already exists in Firestore, skip migration
    if (userDoc.exists()) {
      console.log('User data already exists in Firestore, skipping migration');
      return;
    }

    // Get local storage data
    const localProgress = localStorage.getItem('ml-review-progress');
    const localGamification = localStorage.getItem('ml-review-gamification');

    let progress: UserProgress = {};
    let gamification: GamificationData | null = null;

    if (localProgress) {
      try {
        progress = JSON.parse(localProgress);
        // Convert date strings back to Date objects
        Object.keys(progress).forEach(topicId => {
          if (progress[topicId].lastAccessed) {
            progress[topicId].lastAccessed = new Date(progress[topicId].lastAccessed);
          }
          if (progress[topicId].firstCompletion) {
            progress[topicId].firstCompletion = new Date(progress[topicId].firstCompletion);
          }
          if (progress[topicId].lastMasteredDate) {
            const convertedDate = new Date(progress[topicId].lastMasteredDate);
            // Only set if it's a valid date
            if (!isNaN(convertedDate.getTime())) {
              progress[topicId].lastMasteredDate = convertedDate;
            } else {
              // Remove invalid date
              delete progress[topicId].lastMasteredDate;
            }
          }
          if (progress[topicId].quizScores) {
            progress[topicId].quizScores = progress[topicId].quizScores.map((score: any) => ({
              ...score,
              date: new Date(score.date)
            }));
          }
        });
      } catch (error) {
        console.error('Failed to parse local progress:', error);
      }
    }

    if (localGamification) {
      try {
        gamification = JSON.parse(localGamification);
        if (gamification && gamification.achievements) {
          gamification.achievements = gamification.achievements.map((a: any) => ({
            ...a,
            unlockedDate: new Date(a.unlockedDate)
          }));
        }
        if (gamification && gamification.gemTransactions) {
          gamification.gemTransactions = gamification.gemTransactions.map((t: any) => ({
            ...t,
            timestamp: new Date(t.timestamp)
          }));
        }
      } catch (error) {
        console.error('Failed to parse local gamification:', error);
      }
    }

    // Only migrate if there's actual data
    if (Object.keys(progress).length > 0 || gamification) {
      console.log('Migrating local data to Firestore...');
      await saveUserDataToFirestore(userId, progress, gamification || undefined);
      console.log('Migration complete!');
    }
  } catch (error) {
    console.error('Error migrating data:', error);
    throw error;
  }
};

/**
 * Remove undefined values from an object recursively
 * Firestore doesn't accept undefined values
 */
const removeUndefinedFields = (obj: any): any => {
  if (obj === null || obj === undefined) {
    return null;
  }

  if (Array.isArray(obj)) {
    return obj.map(item => removeUndefinedFields(item));
  }

  if (obj instanceof Date) {
    return obj;
  }

  if (typeof obj === 'object') {
    const cleaned: any = {};
    for (const key in obj) {
      if (obj[key] !== undefined) {
        cleaned[key] = removeUndefinedFields(obj[key]);
      }
    }
    return cleaned;
  }

  return obj;
};

/**
 * Save user progress and gamification data to Firestore
 */
export const saveUserDataToFirestore = async (
  userId: string,
  progress: UserProgress,
  gamification?: GamificationData
): Promise<void> => {
  try {
    console.log('[FirebaseSync] Saving data for user:', userId, {
      topics: Object.keys(progress).length,
      totalQuizScores: Object.values(progress).reduce((sum, p) => sum + (p.quizScores?.length || 0), 0)
    });

    const userDocRef = doc(db, 'users', userId);
    const userDoc = await getDoc(userDocRef);

    const updateData: any = {
      lastSyncAt: serverTimestamp(),
    };

    if (progress) {
      // Clean undefined values from progress
      updateData.progress = removeUndefinedFields(progress);
    }

    if (gamification) {
      // Clean undefined values from gamification
      updateData.gamification = removeUndefinedFields(gamification);
    }

    if (userDoc.exists()) {
      console.log('[FirebaseSync] Updating existing document');
      await updateDoc(userDocRef, updateData);
    } else {
      console.log('[FirebaseSync] Creating new document');
      // Create new user document
      await setDoc(userDocRef, {
        ...updateData,
        createdAt: serverTimestamp(),
      });
    }
    console.log('[FirebaseSync] ✅ Save successful');
  } catch (error) {
    console.error('[FirebaseSync] ❌ Error saving to Firestore:', error);
    throw error;
  }
};

/**
 * Load user data from Firestore
 */
export const loadUserDataFromFirestore = async (
  userId: string
): Promise<{ progress: UserProgress; gamification: GamificationData | null } | null> => {
  try {
    const userDocRef = doc(db, 'users', userId);
    const userDoc = await getDoc(userDocRef);

    if (!userDoc.exists()) {
      return null;
    }

    const data = userDoc.data();
    
    // Convert Firestore timestamps back to Date objects
    let progress: UserProgress = data.progress || {};
    Object.keys(progress).forEach(topicId => {
      if (progress[topicId].lastAccessed) {
        const lastAccessed = progress[topicId].lastAccessed as any;
        progress[topicId].lastAccessed = lastAccessed.toDate ? 
          lastAccessed.toDate() : 
          new Date(lastAccessed);
      }
      if (progress[topicId].firstCompletion) {
        const firstCompletion = progress[topicId].firstCompletion as any;
        progress[topicId].firstCompletion = firstCompletion.toDate ?
          firstCompletion.toDate() :
          new Date(firstCompletion);
      }
      if (progress[topicId].lastMasteredDate) {
        const lastMasteredDate = progress[topicId].lastMasteredDate as any;
        const convertedDate = lastMasteredDate.toDate ?
          lastMasteredDate.toDate() :
          new Date(lastMasteredDate);
        // Only set if it's a valid date
        if (!isNaN(convertedDate.getTime())) {
          progress[topicId].lastMasteredDate = convertedDate;
        } else {
          // Remove invalid date
          delete progress[topicId].lastMasteredDate;
        }
      }
      if (progress[topicId].quizScores) {
        progress[topicId].quizScores = progress[topicId].quizScores.map((score: any) => ({
          ...score,
          date: score.date.toDate ? score.date.toDate() : new Date(score.date)
        }));
      }
    });

    let gamification: GamificationData | null = data.gamification || null;
    if (gamification) {
      if (gamification.achievements) {
        gamification.achievements = gamification.achievements.map((a: any) => ({
          ...a,
          unlockedDate: a.unlockedDate.toDate ? a.unlockedDate.toDate() : new Date(a.unlockedDate)
        }));
      }
      if (gamification.gemTransactions) {
        gamification.gemTransactions = gamification.gemTransactions.map((t: any) => ({
          ...t,
          timestamp: t.timestamp.toDate ? t.timestamp.toDate() : new Date(t.timestamp)
        }));
      }
    }

    return { progress, gamification };
  } catch (error) {
    console.error('Error loading from Firestore:', error);
    throw error;
  }
};

/**
 * Sync local storage with Firestore
 * This will merge local changes with cloud data
 */
export const syncWithFirestore = async (
  userId: string,
  localProgress: UserProgress,
  localGamification: GamificationData
): Promise<{ progress: UserProgress; gamification: GamificationData }> => {
  try {
    // Load cloud data
    const cloudData = await loadUserDataFromFirestore(userId);

    if (!cloudData) {
      // No cloud data, upload local data
      await saveUserDataToFirestore(userId, localProgress, localGamification);
      return { progress: localProgress, gamification: localGamification };
    }

    // Merge strategy: Use cloud data if it exists, otherwise use local
    // For more complex merging (e.g., based on timestamps), implement conflict resolution here
    const mergedProgress = { ...localProgress, ...cloudData.progress };
    const mergedGamification = cloudData.gamification || localGamification;

    // Save merged data back to Firestore
    await saveUserDataToFirestore(userId, mergedProgress, mergedGamification);

    return { progress: mergedProgress, gamification: mergedGamification };
  } catch (error) {
    console.error('Error syncing with Firestore:', error);
    throw error;
  }
};

/**
 * Create or update user profile in Firestore
 */
export const updateUserProfile = async (user: User): Promise<void> => {
  try {
    const userDocRef = doc(db, 'users', user.uid);
    const userDoc = await getDoc(userDocRef);

    const profileData = {
      email: user.email,
      displayName: user.displayName,
      photoURL: user.photoURL,
      lastLoginAt: serverTimestamp(),
    };

    if (userDoc.exists()) {
      await updateDoc(userDocRef, profileData);
    } else {
      await setDoc(userDocRef, {
        ...profileData,
        createdAt: serverTimestamp(),
        progress: {},
        gamification: null,
      });
    }
  } catch (error) {
    console.error('Error updating user profile:', error);
    throw error;
  }
};
