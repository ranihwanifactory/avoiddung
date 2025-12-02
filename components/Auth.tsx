import React, { useState } from 'react';
import { signInWithGoogle, auth } from '../services/firebase';
import { createUserWithEmailAndPassword, signInWithEmailAndPassword } from 'firebase/auth';
import { soundManager } from '../services/audio';

interface Props {
  onPlayAsGuest: () => void;
}

export const Auth: React.FC<Props> = ({ onPlayAsGuest }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleGoogle = async () => {
    soundManager.playClick();
    try {
      await signInWithGoogle();
    } catch (e: any) {
      setError(e.message);
    }
  };

  const handleEmailAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    soundManager.playClick();
    setError('');
    
    try {
      if (isLogin) {
        await signInWithEmailAndPassword(auth, email, password);
      } else {
        await createUserWithEmailAndPassword(auth, email, password);
      }
    } catch (e: any) {
      setError(e.message.replace('Firebase: ', ''));
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-[50vh] p-4 w-full max-w-sm mx-auto">
      <div className="bg-white p-8 rounded-2xl shadow-xl w-full border-b-8 border-amber-200">
        <h2 className="text-2xl font-bold text-center mb-6 text-gray-800">
            {isLogin ? 'Welcome Back!' : 'Join the Fun!'}
        </h2>
        
        {error && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-2 rounded mb-4 text-sm">
                {error}
            </div>
        )}

        <button 
          onClick={handleGoogle}
          className="w-full bg-white border-2 border-gray-200 hover:bg-gray-50 text-gray-700 font-bold py-3 px-4 rounded-xl flex items-center justify-center gap-3 transition-colors mb-4"
        >
          <img src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg" className="w-6 h-6" alt="Google" />
          Sign in with Google
        </button>

        <div className="relative my-6">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-gray-200"></div>
          </div>
          <div className="relative flex justify-center text-sm">
            <span className="px-2 bg-white text-gray-500">Or continue with email</span>
          </div>
        </div>

        <form onSubmit={handleEmailAuth} className="space-y-4 mb-6">
          <div>
            <input 
              type="email" 
              required
              placeholder="Email address"
              className="w-full px-4 py-3 rounded-lg border-2 border-gray-200 focus:border-amber-400 focus:ring-0 outline-none transition-colors"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>
          <div>
            <input 
              type="password" 
              required
              placeholder="Password"
              className="w-full px-4 py-3 rounded-lg border-2 border-gray-200 focus:border-amber-400 focus:ring-0 outline-none transition-colors"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
          
          <button 
            type="submit"
            className="w-full bg-amber-500 hover:bg-amber-600 text-white font-bold py-3 px-4 rounded-xl shadow-md active:translate-y-0.5 active:shadow-sm transition-all"
          >
            {isLogin ? 'Log In' : 'Sign Up'}
          </button>
        </form>

        <button 
            onClick={() => {
                soundManager.playClick();
                onPlayAsGuest();
            }}
            className="w-full bg-gray-100 hover:bg-gray-200 text-gray-600 font-bold py-3 px-4 rounded-xl transition-all mb-4"
        >
            Play as Guest
        </button>

        <div className="mt-2 text-center text-sm text-gray-600">
          {isLogin ? "Don't have an account? " : "Already have an account? "}
          <button 
            onClick={() => {
                soundManager.playClick();
                setIsLogin(!isLogin);
                setError('');
            }}
            className="text-amber-600 font-bold hover:underline"
          >
            {isLogin ? 'Sign up' : 'Log in'}
          </button>
        </div>
      </div>
    </div>
  );
};