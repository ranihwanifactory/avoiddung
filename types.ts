export interface UserProfile {
  uid: string;
  email: string | null;
  displayName: string | null;
  photoURL: string | null;
}

export interface ScoreEntry {
  id?: string;
  uid: string;
  displayName: string;
  score: number;
  timestamp: number;
}

export enum GameState {
  MENU = 'MENU',
  PLAYING = 'PLAYING',
  GAME_OVER = 'GAME_OVER',
  LEADERBOARD = 'LEADERBOARD',
  AUTH = 'AUTH'
}

export interface SoundManager {
  playJump: () => void;
  playCrash: () => void;
  playScore: () => void;
  playClick: () => void;
}
