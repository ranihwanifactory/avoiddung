import React, { useState } from 'react';
import { signInWithGoogle, auth } from '../services/firebase';
import { createUserWithEmailAndPassword, signInWithEmailAndPassword } from 'firebase/auth';
import { soundManager } from '../services/audio';

interface Props {
  onCancel: () => void;
}

export const Auth: React.FC<Props> = ({ onCancel }) => {
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
      if (e.message.includes('auth/email-already-in-use')) {
        setError('이미 사용 중인 이메일입니다.');
      } else if (e.message.includes('auth/weak-password')) {
        setError('비밀번호가 너무 약합니다.');
      } else if (e.message.includes('auth/invalid-email')) {
        setError('유효하지 않은 이메일 주소입니다.');
      } else {
        setError('로그인/회원가입 실패: 정보를 확인해주세요.');
      }
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-[50vh] p-4 w-full max-w-sm mx-auto">
      <div className="bg-white p-8 rounded-2xl shadow-xl w-full border-b-8 border-amber-200 relative">
        <button 
            onClick={onCancel}
            className="absolute top-4 right-4 text-gray-400 hover:text-gray-600"
        >
            <i className="fas fa-times text-xl"></i>
        </button>

        <h2 className="text-2xl font-bold text-center mb-6 text-gray-800">
            {isLogin ? '로그인' : '회원가입'}
        </h2>
        
        {error && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-2 rounded mb-4 text-sm break-words">
                {error}
            </div>
        )}

        <button 
          onClick={handleGoogle}
          className="w-full bg-white border-2 border-gray-200 hover:bg-gray-50 text-gray-700 font-bold py-3 px-4 rounded-xl flex items-center justify-center gap-3 transition-colors mb-4"
        >
          <img src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg" className="w-6 h-6" alt="Google" />
          구글 계정으로 로그인
        </button>

        <div className="relative my-6">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-gray-200"></div>
          </div>
          <div className="relative flex justify-center text-sm">
            <span className="px-2 bg-white text-gray-500">또는 이메일로 계속하기</span>
          </div>
        </div>

        <form onSubmit={handleEmailAuth} className="space-y-4 mb-6">
          <div>
            <input 
              type="email" 
              required
              placeholder="이메일 주소"
              className="w-full px-4 py-3 rounded-lg border-2 border-gray-200 focus:border-amber-400 focus:ring-0 outline-none transition-colors"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>
          <div>
            <input 
              type="password" 
              required
              placeholder="비밀번호"
              className="w-full px-4 py-3 rounded-lg border-2 border-gray-200 focus:border-amber-400 focus:ring-0 outline-none transition-colors"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
          
          <button 
            type="submit"
            className="w-full bg-amber-500 hover:bg-amber-600 text-white font-bold py-3 px-4 rounded-xl shadow-md active:translate-y-0.5 active:shadow-sm transition-all"
          >
            {isLogin ? '로그인' : '회원가입'}
          </button>
        </form>

        <div className="mt-2 text-center text-sm text-gray-600">
          {isLogin ? "계정이 없으신가요? " : "이미 계정이 있으신가요? "}
          <button 
            onClick={() => {
                soundManager.playClick();
                setIsLogin(!isLogin);
                setError('');
            }}
            className="text-amber-600 font-bold hover:underline"
          >
            {isLogin ? '회원가입' : '로그인'}
          </button>
        </div>
      </div>
    </div>
  );
};