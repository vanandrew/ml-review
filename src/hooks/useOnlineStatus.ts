import { useState, useEffect } from 'react';

/**
 * Custom hook to detect online/offline status
 * Returns true if online, false if offline
 */
export const useOnlineStatus = (): boolean => {
  const [isOnline, setIsOnline] = useState<boolean>(navigator.onLine);

  useEffect(() => {
    const handleOnline = () => {
      console.log('[Online Status] Connection restored');
      setIsOnline(true);
    };

    const handleOffline = () => {
      console.log('[Online Status] Connection lost');
      setIsOnline(false);
    };

    // Add event listeners
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // Cleanup
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return isOnline;
};

/**
 * Enhanced online status hook with additional features
 */
export interface OnlineStatusInfo {
  isOnline: boolean;
  wasOffline: boolean; // Track if user was recently offline
  offlineDuration: number; // How long offline in milliseconds
}

export const useEnhancedOnlineStatus = (): OnlineStatusInfo => {
  const [isOnline, setIsOnline] = useState<boolean>(navigator.onLine);
  const [wasOffline, setWasOffline] = useState<boolean>(false);
  const [offlineStartTime, setOfflineStartTime] = useState<number | null>(null);
  const [offlineDuration, setOfflineDuration] = useState<number>(0);

  useEffect(() => {
    const handleOnline = () => {
      console.log('[Online Status] Connection restored');
      setIsOnline(true);
      setWasOffline(true);

      // Calculate offline duration
      if (offlineStartTime) {
        const duration = Date.now() - offlineStartTime;
        setOfflineDuration(duration);
        setOfflineStartTime(null);

        // Reset wasOffline flag after 5 seconds
        setTimeout(() => setWasOffline(false), 5000);
      }
    };

    const handleOffline = () => {
      console.log('[Online Status] Connection lost');
      setIsOnline(false);
      setOfflineStartTime(Date.now());
    };

    // Add event listeners
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // Cleanup
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [offlineStartTime]);

  return {
    isOnline,
    wasOffline,
    offlineDuration,
  };
};
