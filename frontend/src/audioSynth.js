// High-Tech Web Audio API Synthesizer
let audioCtx = null;
let isAudioEnabled = true;

export function toggleAudioState() {
  isAudioEnabled = !isAudioEnabled;
  if (isAudioEnabled) playSciFiChime();
  return isAudioEnabled;
}

export function playBeep(freq = 880, duration = 0.08, type = 'sine') {
  if (!isAudioEnabled) return;
  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
    gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + duration);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + duration);
  } catch (e) {}
}

export function playSciFiChime() {
  playBeep(523.25, 0.06);
  setTimeout(() => playBeep(659.25, 0.06), 60);
  setTimeout(() => playBeep(1046.50, 0.12), 120);
}

