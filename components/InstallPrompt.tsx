import React, { useEffect, useState } from 'react';
import { soundManager } from '../services/audio';

export const InstallPrompt: React.FC = () => {
  const [deferredPrompt, setDeferredPrompt] = useState<any>(null);
  const [showButton, setShowButton] = useState(false);

  useEffect(() => {
    const handler = (e: any) => {
      e.preventDefault();
      setDeferredPrompt(e);
      setShowButton(true);
    };

    window.addEventListener('beforeinstallprompt', handler);

    return () => {
      window.removeEventListener('beforeinstallprompt', handler);
    };
  }, []);

  const handleInstallClick = async () => {
    soundManager.playClick();
    if (!deferredPrompt) return;
    
    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    
    if (outcome === 'accepted') {
      setShowButton(false);
    }
    setDeferredPrompt(null);
  };

  if (!showButton) return null;

  return (
    <div className="fixed bottom-4 left-4 right-4 z-50 animate-bounce">
      <button
        onClick={handleInstallClick}
        className="w-full bg-blue-600 text-white font-bold py-3 px-4 rounded-xl shadow-lg border-2 border-blue-400 flex items-center justify-center gap-2 hover:bg-blue-700 transition-colors"
      >
        <i className="fas fa-download"></i>
        Install App
      </button>
    </div>
  );
};
