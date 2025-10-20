import { doc, setDoc, getDoc, collection, addDoc, query, where, getDocs, deleteDoc } from 'firebase/firestore';
import { db } from '../config/firebase';
import { CachedAIQuestion } from '../types/ai';

/**
 * Save API key to Firestore (user-specific, private subcollection)
 */
export const saveAPIKey = async (
  userId: string,
  provider: 'claude',
  apiKey: string
): Promise<void> => {
  try {
    const userPrivateRef = doc(db, 'users', userId, 'private', 'aiSettings');
    await setDoc(userPrivateRef, {
      provider,
      apiKey,
      updatedAt: new Date().toISOString(),
    });
    console.log('[AI Firestore] API key saved successfully');
  } catch (error) {
    console.error('[AI Firestore] Error saving API key:', error);
    throw new Error('Failed to save API key');
  }
};

/**
 * Load API key from Firestore
 */
export const loadAPIKey = async (userId: string): Promise<{ provider: 'claude' | null; apiKey: string | null }> => {
  try {
    const userPrivateRef = doc(db, 'users', userId, 'private', 'aiSettings');
    const docSnap = await getDoc(userPrivateRef);

    if (docSnap.exists()) {
      const data = docSnap.data();
      return {
        provider: data.provider || null,
        apiKey: data.apiKey || null,
      };
    }

    return { provider: null, apiKey: null };
  } catch (error) {
    console.error('[AI Firestore] Error loading API key:', error);
    return { provider: null, apiKey: null };
  }
};

/**
 * Delete API key from Firestore
 */
export const deleteAPIKey = async (userId: string): Promise<void> => {
  try {
    const userPrivateRef = doc(db, 'users', userId, 'private', 'aiSettings');
    await deleteDoc(userPrivateRef);
    console.log('[AI Firestore] API key deleted successfully');
  } catch (error) {
    console.error('[AI Firestore] Error deleting API key:', error);
    throw new Error('Failed to delete API key');
  }
};

/**
 * Save cached AI question to Firestore
 */
export const saveCachedQuestion = async (
  userId: string,
  question: CachedAIQuestion
): Promise<string> => {
  try {
    const cacheRef = collection(db, 'users', userId, 'aiQuestionCache');
    const docRef = await addDoc(cacheRef, {
      ...question,
      question: JSON.stringify(question.question), // Serialize question object
      createdAt: question.createdAt.toISOString(),
      lastUsedAt: question.lastUsedAt.toISOString(),
      cachedAt: question.cachedAt.toISOString(),
      expiresAt: question.expiresAt.toISOString(),
      flaggedAt: question.flaggedAt?.toISOString() || null,
      validatedAt: question.validatedAt?.toISOString() || null,
    });

    console.log('[AI Firestore] Cached question saved:', docRef.id);
    return docRef.id;
  } catch (error) {
    console.error('[AI Firestore] Error saving cached question:', error);
    throw new Error('Failed to save cached question');
  }
};

/**
 * Load cached questions for a topic
 */
export const loadCachedQuestions = async (
  userId: string,
  topicId: string,
  limit: number = 20
): Promise<CachedAIQuestion[]> => {
  try {
    const cacheRef = collection(db, 'users', userId, 'aiQuestionCache');
    const q = query(
      cacheRef,
      where('topicId', '==', topicId),
      where('status', '==', 'active')
    );

    const querySnapshot = await getDocs(q);
    const questions: CachedAIQuestion[] = [];

    querySnapshot.forEach((doc) => {
      const data = doc.data();

      // Parse dates and question object
      const question: CachedAIQuestion = {
        ...data,
        question: JSON.parse(data.question),
        createdAt: new Date(data.createdAt),
        lastUsedAt: new Date(data.lastUsedAt),
        cachedAt: new Date(data.cachedAt),
        expiresAt: new Date(data.expiresAt),
        flaggedAt: data.flaggedAt ? new Date(data.flaggedAt) : undefined,
        validatedAt: data.validatedAt ? new Date(data.validatedAt) : undefined,
      } as CachedAIQuestion;

      // Check if expired
      if (new Date() < question.expiresAt) {
        questions.push(question);
      }
    });

    console.log(`[AI Firestore] Loaded ${questions.length} cached questions for topic ${topicId}`);
    return questions.slice(0, limit);
  } catch (error) {
    console.error('[AI Firestore] Error loading cached questions:', error);
    return [];
  }
};

/**
 * Update cached question usage
 */
export const updateQuestionUsage = async (
  userId: string,
  questionId: string
): Promise<void> => {
  try {
    const questionRef = doc(db, 'users', userId, 'aiQuestionCache', questionId);
    const docSnap = await getDoc(questionRef);

    if (docSnap.exists()) {
      const data = docSnap.data();
      await setDoc(questionRef, {
        ...data,
        usageCount: (data.usageCount || 0) + 1,
        lastUsedAt: new Date().toISOString(),
      });
    }
  } catch (error) {
    console.error('[AI Firestore] Error updating question usage:', error);
  }
};

/**
 * Report a question
 */
export const reportQuestion = async (
  userId: string,
  questionId: string,
  reason: string
): Promise<void> => {
  try {
    const questionRef = doc(db, 'users', userId, 'aiQuestionCache', questionId);
    const docSnap = await getDoc(questionRef);

    if (docSnap.exists()) {
      const data = docSnap.data();
      const reportCount = (data.reportCount || 0) + 1;
      const reportReasons = [...(data.reportReasons || []), reason];

      // Flag if 3+ reports
      const shouldFlag = reportCount >= 3;

      await setDoc(questionRef, {
        ...data,
        reportCount,
        reportReasons,
        status: shouldFlag ? 'flagged' : data.status,
        flaggedAt: shouldFlag ? new Date().toISOString() : data.flaggedAt,
      });

      console.log(`[AI Firestore] Question reported (count: ${reportCount})`);
    }
  } catch (error) {
    console.error('[AI Firestore] Error reporting question:', error);
    throw new Error('Failed to report question');
  }
};

/**
 * Clean up expired cache entries
 */
export const cleanExpiredCache = async (userId: string): Promise<number> => {
  try {
    const cacheRef = collection(db, 'users', userId, 'aiQuestionCache');
    const querySnapshot = await getDocs(cacheRef);

    let deletedCount = 0;
    const now = new Date();

    const deletePromises: Promise<void>[] = [];

    querySnapshot.forEach((doc) => {
      const data = doc.data();
      const expiresAt = new Date(data.expiresAt);

      if (now > expiresAt) {
        deletePromises.push(deleteDoc(doc.ref));
        deletedCount++;
      }
    });

    await Promise.all(deletePromises);

    console.log(`[AI Firestore] Cleaned ${deletedCount} expired cache entries`);
    return deletedCount;
  } catch (error) {
    console.error('[AI Firestore] Error cleaning expired cache:', error);
    return 0;
  }
};
