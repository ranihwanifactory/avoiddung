import React, { useRef, useEffect, useState, useCallback } from 'react';
import { soundManager } from '../services/audio';

interface Props {
  onGameOver: (score: number) => void;
}

interface GameObject {
  x: number;
  y: number;
  width: number;
  height: number;
  speed?: number;
}

export const GameCanvas: React.FC<Props> = ({ onGameOver }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number>(0);
  const scoreRef = useRef<number>(0);
  const [currentScore, setCurrentScore] = useState(0);
  const [isPaused, setIsPaused] = useState(false);

  // Game State Refs (avoid stale closures)
  const playerRef = useRef<GameObject>({ x: 0, y: 0, width: 40, height: 40 });
  const enemiesRef = useRef<GameObject[]>([]);
  const lastSpawnTime = useRef<number>(0);
  const gameSpeedRef = useRef<number>(1);
  const canvasSizeRef = useRef({ width: 0, height: 0 });
  
  // Touch handling
  const touchXRef = useRef<number | null>(null);

  const initGame = () => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    
    // Set initial size
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    canvasSizeRef.current = { width: canvas.width, height: canvas.height };

    // Player start position (bottom center)
    playerRef.current = {
      x: canvas.width / 2 - 20,
      y: canvas.height - 100, // Slightly above bottom
      width: 40,
      height: 40
    };

    enemiesRef.current = [];
    scoreRef.current = 0;
    gameSpeedRef.current = 3; // Initial fall speed
    setCurrentScore(0);
  };

  const spawnEnemy = (timestamp: number) => {
    // Spawn rate increases as score increases
    const spawnRate = Math.max(500, 2000 - (scoreRef.current * 10)); 
    
    if (timestamp - lastSpawnTime.current > spawnRate) {
      const size = 35 + Math.random() * 20; // Random size
      const x = Math.random() * (canvasSizeRef.current.width - size);
      
      enemiesRef.current.push({
        x,
        y: -50,
        width: size,
        height: size,
        speed: gameSpeedRef.current + Math.random() * 2 // Some are faster
      });
      lastSpawnTime.current = timestamp;
    }
  };

  const checkCollision = (rect1: GameObject, rect2: GameObject) => {
    return (
      rect1.x < rect2.x + rect2.width &&
      rect1.x + rect1.width > rect2.x &&
      rect1.y < rect2.y + rect2.height &&
      rect1.y + rect1.height > rect2.y
    );
  };

  const gameLoop = useCallback((timestamp: number) => {
    if (isPaused) {
       requestRef.current = requestAnimationFrame(gameLoop);
       return;
    }

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Update Logic
    spawnEnemy(timestamp);

    // Increase difficulty
    gameSpeedRef.current = 3 + (scoreRef.current * 0.01);

    // Update Enemies
    enemiesRef.current.forEach((enemy, index) => {
        enemy.y += enemy.speed || 3;

        // Collision Check
        // Hitbox reduction (make it slightly forgiving)
        const hitboxReduction = 10;
        const playerHitbox = {
            x: playerRef.current.x + hitboxReduction,
            y: playerRef.current.y + hitboxReduction,
            width: playerRef.current.width - (hitboxReduction*2),
            height: playerRef.current.height - (hitboxReduction*2)
        };
        const enemyHitbox = {
             x: enemy.x + 5,
             y: enemy.y + 5,
             width: enemy.width - 10,
             height: enemy.height - 10
        }

        if (checkCollision(playerHitbox, enemyHitbox)) {
            soundManager.playCrash();
            onGameOver(scoreRef.current);
            cancelAnimationFrame(requestRef.current);
            return;
        }

        // Remove if off screen and add score
        if (enemy.y > canvas.height) {
            enemiesRef.current.splice(index, 1);
            scoreRef.current += 10;
            setCurrentScore(scoreRef.current);
            if (scoreRef.current % 100 === 0) {
                soundManager.playScore();
            }
        }
    });

    // Draw Player (Using emoji for simplicity & aesthetics)
    ctx.font = `${playerRef.current.width}px Arial`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.shadowColor = 'rgba(0,0,0,0.2)';
    ctx.shadowBlur = 10;
    // Determine player direction or state if needed, just use a running person
    ctx.fillText('🏃', playerRef.current.x + playerRef.current.width/2, playerRef.current.y + playerRef.current.height/2);

    // Draw Enemies (Poop emoji)
    enemiesRef.current.forEach(enemy => {
        ctx.font = `${enemy.width}px Arial`;
        ctx.fillText('💩', enemy.x + enemy.width/2, enemy.y + enemy.height/2);
    });

    // Draw Ground
    ctx.fillStyle = '#8B4513';
    ctx.fillRect(0, canvas.height - 20, canvas.width, 20);

    requestRef.current = requestAnimationFrame(gameLoop);
  }, [isPaused, onGameOver]);

  // Input Handling
  useEffect(() => {
    initGame();
    requestRef.current = requestAnimationFrame(gameLoop);

    const handleResize = () => {
        if (canvasRef.current) {
            canvasRef.current.width = window.innerWidth;
            canvasRef.current.height = window.innerHeight;
            canvasSizeRef.current = { width: window.innerWidth, height: window.innerHeight };
            // Reset player Y to keep them on ground
            playerRef.current.y = window.innerHeight - 100;
        }
    };

    const handleTouchMove = (e: TouchEvent) => {
        e.preventDefault(); // Prevent scrolling
        const touch = e.touches[0];
        playerRef.current.x = touch.clientX - playerRef.current.width / 2;
        // Clamp to screen
        if (playerRef.current.x < 0) playerRef.current.x = 0;
        if (playerRef.current.x > canvasSizeRef.current.width - playerRef.current.width) {
            playerRef.current.x = canvasSizeRef.current.width - playerRef.current.width;
        }
    };

    // Mouse support for desktop testing
    const handleMouseMove = (e: MouseEvent) => {
        playerRef.current.x = e.clientX - playerRef.current.width / 2;
        // Clamp
        if (playerRef.current.x < 0) playerRef.current.x = 0;
        if (playerRef.current.x > canvasSizeRef.current.width - playerRef.current.width) {
            playerRef.current.x = canvasSizeRef.current.width - playerRef.current.width;
        }
    }

    window.addEventListener('resize', handleResize);
    window.addEventListener('touchmove', handleTouchMove, { passive: false });
    window.addEventListener('mousemove', handleMouseMove);

    return () => {
      cancelAnimationFrame(requestRef.current);
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, [gameLoop]);

  return (
    <div className="relative w-full h-full bg-sky-100 overflow-hidden">
        {/* Score Display HUD */}
        <div className="absolute top-4 right-4 z-10 pointer-events-none">
            <div className="bg-white/80 backdrop-blur rounded-full px-6 py-2 shadow-lg border-2 border-amber-400">
                <span className="text-2xl font-black text-amber-600 font-mono">
                    {currentScore}
                </span>
            </div>
        </div>

        {/* Pause Button */}
        <button 
            className="absolute top-4 left-4 z-10 bg-white/80 p-3 rounded-full shadow-lg border-2 border-gray-300 hover:bg-white active:scale-95 transition-all"
            onClick={() => {
                soundManager.playClick();
                setIsPaused(!isPaused);
            }}
        >
            <i className={`fas ${isPaused ? 'fa-play' : 'fa-pause'} text-gray-700 w-4 h-4 flex items-center justify-center`}></i>
        </button>

        {isPaused && (
            <div className="absolute inset-0 z-20 bg-black/40 backdrop-blur-sm flex items-center justify-center">
                <div className="text-white text-4xl font-black drop-shadow-lg tracking-wider">PAUSED</div>
            </div>
        )}

        <canvas ref={canvasRef} className="block touch-none" />
    </div>
  );
};
