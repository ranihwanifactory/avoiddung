import { initializeApp } from "firebase/app";
import { 
  getAuth, 
  GoogleAuthProvider, 
  signInWithPopup, 
  signOut, 
  onAuthStateChanged,
  User,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword
} from "firebase/auth";
import { 
  getDatabase, 
  ref, 
  push, 
  onValue, 
  query, 
  orderByChild, 
  limitToLast,
  DatabaseReference
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
  const scoresRef = ref(db, 'scores');
  const newScore: ScoreEntry = {
    uid: user.uid,
    displayName: user.displayName || user.email?.split('@')[0] || 'Anonymous',
    score: score,
    timestamp: Date.now()
  };
  return push(scoresRef, newScore);
};

export const subscribeToLeaderboard = (callback: (scores: ScoreEntry[]) => void) => {
  const scoresRef = ref(db, 'scores');
  // Get top 20 scores
  const topScoresQuery = query(scoresRef, orderByChild('score'), limitToLast(20));

  return onValue(topScoresQuery, (snapshot) => {
    const data = snapshot.val();
    if (data) {
      // Firebase returns object with keys, convert to array
      const scoreList: ScoreEntry[] = Object.keys(data).map(key => ({
        id: key,
        ...data[key]
      }));
      // Sort descending (Firebase returns ascending for limitToLast)
      scoreList.sort((a, b) => b.score - a.score);
      callback(scoreList);
    } else {
      callback([]);
    }
  });
};
