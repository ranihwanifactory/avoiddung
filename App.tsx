import React, { useState, useEffect } from 'react';
import { User, onAuthStateChanged } from 'firebase/auth';
import { auth, saveScore, logout } from './services/firebase';
import { GameState } from './types';
import { soundManager } from './services/audio';
import { Auth } from './components/Auth';
import { GameCanvas } from './components/GameCanvas';
import { Leaderboard } from './components/Leaderboard';
import { InstallPrompt } from './components/InstallPrompt';

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [gameState, setGameState] = useState<GameState>(GameState.MENU); // Start at MENU directly
  const [lastScore, setLastScore] = useState(0);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (u) => {
      setUser(u);
      // If user logs in while on AUTH screen, go back to MENU
      if (u && gameState === GameState.AUTH) {
          setGameState(GameState.MENU);
      }
    });
    return () => unsubscribe();
  }, [gameState]);

  const handleStartGame = () => {
    soundManager.playClick();
    setGameState(GameState.PLAYING);
  };

  const handleGameOver = async (score: number) => {
    setLastScore(score);
    if (user) {
      try {
        await saveScore(user, score);
      } catch (err) {
        console.error("Failed to save score", err);
      }
    }
    setGameState(GameState.GAME_OVER);
  };

  const handleQuitGame = () => {
      setGameState(GameState.MENU);
  };

  const handleShare = async () => {
    soundManager.playClick();
    if (navigator.share) {
      try {
        await navigator.share({
          title: '똥 피하기!',
          text: `똥 피하기 게임에서 ${lastScore}점을 기록했습니다! 저를 이겨보세요! 💩🏃`,
          url: window.location.href,
        });
      } catch (err) {
        console.log('Share canceled');
      }
    } else {
      // Fallback: Copy to clipboard
      navigator.clipboard.writeText(`똥 피하기 게임에서 ${lastScore}점을 기록했습니다! 게임하러 가기: ${window.location.href}`);
      alert('클립보드에 복사되었습니다!');
    }
  };

  return (
    <div className="h-screen w-screen overflow-hidden bg-amber-50 flex flex-col font-sans">
      
      {/* Header / Nav (Hidden during gameplay) */}
      {gameState !== GameState.PLAYING && (
        <header className="p-4 flex justify-between items-center bg-white shadow-sm z-50">
           <h1 className="text-xl font-black text-amber-600 tracking-tighter flex items-center gap-2">
             <span className="text-2xl">💩</span> 똥 피하기
           </h1>
           {user ? (
             <button 
                onClick={() => { soundManager.playClick(); logout(); }}
                className="text-sm font-semibold text-gray-500 hover:text-red-500"
               >
                 로그아웃
               </button>
           ) : (
             <button 
                onClick={() => { soundManager.playClick(); setGameState(GameState.AUTH); }}
                className="text-sm font-semibold text-amber-600 border border-amber-600 px-3 py-1 rounded-lg hover:bg-amber-50"
             >
                로그인
             </button>
           )}
        </header>
      )}

      {/* Main Content Area */}
      <main className="flex-1 relative">
        
        {/* Auth Screen */}
        {gameState === GameState.AUTH && (
           <div className="h-full flex items-center justify-center bg-sky-100 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')]">
             <Auth onCancel={() => setGameState(GameState.MENU)} />
           </div>
        )}

        {/* Main Menu */}
        {gameState === GameState.MENU && (
          <div className="h-full flex flex-col items-center justify-center p-6 space-y-6 bg-sky-100">
             <div className="text-center animate-bounce">
                <span className="text-8xl block mb-2">💩</span>
             </div>
             <h2 className="text-4xl font-black text-center text-gray-800 drop-shadow-md">
               피할 준비 되셨나요?
             </h2>
             <p className="text-gray-600 text-center max-w-xs">
               <span className="font-bold">화살표 키</span>나 <span className="font-bold">버튼</span>을 사용하여 이동하세요. 똥을 맞으면 안돼요!
             </p>

             <button 
               onClick={handleStartGame}
               className="w-full max-w-xs bg-amber-500 hover:bg-amber-600 text-white text-2xl font-black py-4 rounded-2xl shadow-[0_6px_0_rgb(180,83,9)] active:shadow-none active:translate-y-1.5 transition-all"
             >
               게임 시작
             </button>

             <button 
               onClick={() => { soundManager.playClick(); setGameState(GameState.LEADERBOARD); }}
               className="w-full max-w-xs bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 rounded-xl shadow-[0_4px_0_rgb(29,78,216)] active:shadow-none active:translate-y-1 transition-all flex items-center justify-center gap-2"
             >
               <i className="fas fa-trophy"></i> 순위표
             </button>
          </div>
        )}

        {/* Gameplay */}
        {gameState === GameState.PLAYING && (
          <GameCanvas onGameOver={handleGameOver} onQuit={handleQuitGame} />
        )}

        {/* Game Over Screen */}
        {gameState === GameState.GAME_OVER && (
          <div className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-black/80 backdrop-blur-sm p-6 text-white animate-fadeIn">
             <h2 className="text-6xl font-black text-red-500 mb-2 drop-shadow-[0_2px_0_#fff]">으악!</h2>
             <p className="text-xl mb-8 font-bold">똥을 밟았어요.</p>
             
             <div className="bg-white/10 p-6 rounded-2xl border-2 border-white/20 text-center mb-8 w-full max-w-xs">
               <p className="text-sm uppercase tracking-widest text-gray-300 mb-2">최종 점수</p>
               <p className="text-6xl font-mono font-black text-amber-400">{lastScore}</p>
             </div>

             {!user && (
                 <div className="mb-6 w-full max-w-xs">
                     <button
                        onClick={() => {
                             soundManager.playClick();
                             setGameState(GameState.AUTH);
                        }}
                        className="w-full bg-amber-100 text-amber-800 font-bold py-2 rounded-lg border-2 border-amber-300 hover:bg-amber-200"
                     >
                         로그인하고 점수 저장하기
                     </button>
                 </div>
             )}

             <div className="grid grid-cols-2 gap-4 w-full max-w-xs">
                <button 
                  onClick={handleStartGame}
                  className="col-span-2 bg-green-500 hover:bg-green-600 text-white py-4 rounded-xl font-black shadow-[0_4px_0_rgb(21,128,61)] active:translate-y-1 active:shadow-none transition-all"
                >
                  다시 하기
                </button>
                <button 
                   onClick={handleShare}
                   className="bg-blue-500 hover:bg-blue-600 text-white py-3 rounded-xl font-bold shadow-[0_4px_0_rgb(29,78,216)] active:translate-y-1 active:shadow-none transition-all"
                >
                  <i className="fas fa-share-alt mr-2"></i> 공유
                </button>
                <button 
                   onClick={() => { soundManager.playClick(); setGameState(GameState.MENU); }}
                   className="bg-gray-600 hover:bg-gray-700 text-white py-3 rounded-xl font-bold shadow-[0_4px_0_rgb(55,65,81)] active:translate-y-1 active:shadow-none transition-all"
                >
                  메뉴
                </button>
             </div>
          </div>
        )}

        {/* Leaderboard Overlay */}
        {gameState === GameState.LEADERBOARD && (
          <div className="absolute inset-0 z-40 bg-sky-100 p-4 pt-8">
            <Leaderboard 
              onBack={() => { setGameState(GameState.MENU); }} 
              currentUserScore={(!user && lastScore > 0) ? undefined : (lastScore > 0 ? lastScore : undefined)}
            />
          </div>
        )}

        <InstallPrompt />
      </main>
    </div>
  );
}