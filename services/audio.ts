import { SoundManager } from '../types';

class WebAudioSoundManager implements SoundManager {
  private context: AudioContext | null = null;
  private masterGain: GainNode | null = null;

  constructor() {
    try {
      this.context = new (window.AudioContext || (window as any).webkitAudioContext)();
      this.masterGain = this.context.createGain();
      this.masterGain.gain.value = 0.3; // Global volume
      this.masterGain.connect(this.context.destination);
    } catch (e) {
      console.warn('Web Audio API not supported', e);
    }
  }

  private ensureContext() {
    if (this.context?.state === 'suspended') {
      this.context.resume();
    }
  }

  private playTone(freq: number, type: OscillatorType, duration: number, startTime: number = 0) {
    if (!this.context || !this.masterGain) return;
    this.ensureContext();

    const osc = this.context.createOscillator();
    const gain = this.context.createGain();

    osc.type = type;
    osc.frequency.setValueAtTime(freq, this.context.currentTime + startTime);
    
    gain.gain.setValueAtTime(0.5, this.context.currentTime + startTime);
    gain.gain.exponentialRampToValueAtTime(0.01, this.context.currentTime + startTime + duration);

    osc.connect(gain);
    gain.connect(this.masterGain);

    osc.start(this.context.currentTime + startTime);
    osc.stop(this.context.currentTime + startTime + duration);
  }

  playJump() {
    // Swoosh sound
    this.playTone(400, 'sine', 0.1);
    if(this.context && this.masterGain) {
        const osc = this.context.createOscillator();
        const gain = this.context.createGain();
        osc.frequency.setValueAtTime(200, this.context.currentTime);
        osc.frequency.linearRampToValueAtTime(600, this.context.currentTime + 0.1);
        gain.gain.setValueAtTime(0.3, this.context.currentTime);
        gain.gain.linearRampToValueAtTime(0, this.context.currentTime + 0.1);
        osc.connect(gain);
        gain.connect(this.masterGain);
        osc.start();
        osc.stop(this.context.currentTime + 0.1);
    }
  }

  playCrash() {
    // Noise/Low thud
    this.playTone(100, 'sawtooth', 0.3);
    this.playTone(80, 'square', 0.3, 0.05);
  }

  playScore() {
    // Ding!
    this.playTone(800, 'sine', 0.1);
    this.playTone(1200, 'sine', 0.1, 0.05);
  }

  playClick() {
    this.playTone(600, 'triangle', 0.05);
  }
}

export const soundManager = new WebAudioSoundManager();
