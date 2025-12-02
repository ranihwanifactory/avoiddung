import { initializeApp } from "firebase/app";
import { 
  getAuth, 
  GoogleAuthProvider, 
  signInWithPopup, 
  signOut, 
  User,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword
} from "firebase/auth";
import { 
  getDatabase, 
  ref, 
  onValue, 
  query, 
  orderByChild, 
  limitToLast,
  runTransaction
} from "firebase/database";
import { ScoreEntry } from "../types";

const firebaseConfig = {
  apiKey: "AIzaSyDJoI2d4yhRHl-jOsZMp57V41Skn8HYFa8",
  authDomain: "touchgame-bf7e2.firebaseapp.com",
  databaseURL: "https://touchgame-bf7e2-default-rtdb.firebaseio.com",
  projectId: "touchgame-bf7e2",
  storageBucket: "touchgame-bf7e2.firebasestorage.app",
  messagingSenderId: "289443560144",
  appId: "1:289443560144:web:6ef844f5e4a022fca13cd5"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getDatabase(app);

// Auth Helpers
export const signInWithGoogle = async () => {
  const provider = new GoogleAuthProvider();
  return signInWithPopup(auth, provider);
};

export const logout = () => signOut(auth);

// DB Helpers
export const saveScore = async (user: User, score: number) => {
  // Use a user-specific path to ensure one entry per user
  // Using a new table 'dodge_poop_best_scores'
  const userScoreRef = ref(db, `dodge_poop_best_scores/${user.uid}`);
  
  // Use transaction to check if new score is higher than existing score
  return runTransaction(userScoreRef, (currentData) => {
    if (currentData === null) {
      // No existing score, create new
      return {
        uid: user.uid,
        displayName: user.displayName || user.email?.split('@')[0] || '익명',
        score: score,
        timestamp: Date.now()
      };
    } else {
      // Check if new score is higher
      if (score > currentData.score) {
        return {
          ...currentData, // Keep existing data (like name) if needed, or overwrite
          score: score,
          timestamp: Date.now(),
          // Ensure display name is up to date
          displayName: user.displayName || user.email?.split('@')[0] || currentData.displayName
        };
      }
      // If score is not higher, abort the transaction (return undefined) to do nothing
      return; 
    }
  });
};

export const subscribeToLeaderboard = (callback: (scores: ScoreEntry[]) => void) => {
  const scoresRef = ref(db, 'dodge_poop_best_scores');
  // Get top 20 scores
  const topScoresQuery = query(scoresRef, orderByChild('score'), limitToLast(20));

  return onValue(topScoresQuery, (snapshot) => {
    const data = snapshot.val();
    if (data) {
      const scoreList: ScoreEntry[] = Object.keys(data).map(key => ({
        id: key,
        ...data[key]
      }));
      // Sort descending
      scoreList.sort((a, b) => b.score - a.score);
      callback(scoreList);
    } else {
      callback([]);
    }
  });
};