import React, { useEffect, useState } from 'react';
import { subscribeToLeaderboard } from '../services/firebase';
import { ScoreEntry } from '../types';
import { soundManager } from '../services/audio';

interface Props {
  onBack: () => void;
  currentUserScore?: number;
}

export const Leaderboard: React.FC<Props> = ({ onBack, currentUserScore }) => {
  const [scores, setScores] = useState<ScoreEntry[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const unsubscribe = subscribeToLeaderboard((data) => {
      setScores(data);
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  return (
    <div className="flex flex-col h-full max-w-md mx-auto p-4 bg-white/90 rounded-xl shadow-2xl backdrop-blur-sm">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-3xl font-black text-amber-600 drop-shadow-sm">
          <i className="fas fa-trophy text-yellow-500 mr-2"></i>
          Hall of Fame
        </h2>
        <button 
          onClick={() => { soundManager.playClick(); onBack(); }}
          className="bg-gray-200 hover:bg-gray-300 text-gray-700 p-2 rounded-full w-10 h-10 flex items-center justify-center transition-colors"
        >
          <i className="fas fa-times"></i>
        </button>
      </div>

      {currentUserScore !== undefined && currentUserScore > 0 && (
        <div className="bg-amber-100 p-4 rounded-lg mb-4 border-2 border-amber-300 text-center animate-pulse">
          <p className="text-sm text-amber-800 font-bold uppercase">Your Last Run</p>
          <p className="text-3xl font-black text-amber-600">{currentUserScore}</p>
        </div>
      )}

      <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
        {loading ? (
          <div className="flex justify-center items-center h-40">
            <i className="fas fa-spinner fa-spin text-4xl text-amber-500"></i>
          </div>
        ) : (
          <table className="w-full">
            <thead>
              <tr className="text-left text-gray-500 text-sm border-b-2 border-gray-100">
                <th className="pb-2 pl-2">Rank</th>
                <th className="pb-2">Player</th>
                <th className="pb-2 text-right pr-2">Score</th>
              </tr>
            </thead>
            <tbody>
              {scores.map((entry, index) => {
                let rankIcon;
                if (index === 0) rankIcon = '🥇';
                else if (index === 1) rankIcon = '🥈';
                else if (index === 2) rankIcon = '🥉';
                else rankIcon = `#${index + 1}`;

                return (
                  <tr key={entry.id || index} className={`border-b border-gray-50 hover:bg-amber-50 transition-colors ${index < 3 ? 'font-bold' : ''}`}>
                    <td className="py-3 pl-2 text-lg">{rankIcon}</td>
                    <td className="py-3 truncate max-w-[150px]">
                      {entry.displayName}
                    </td>
                    <td className="py-3 text-right pr-2 font-mono text-lg text-amber-600">
                      {entry.score}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};
