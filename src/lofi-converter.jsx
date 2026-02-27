import { useState, useRef, useCallback, useEffect } from "react";

const THEMES = {
  light: {
    bg: "#f5f0eb",
    bgCard: "#ffffff",
    bgInput: "#ede7e0",
    border: "#d4cac0",
    borderHover: "#b8a99a",
    accent: "#c4845a",
    accentDim: "#a66b3f",
    accentGlow: "rgba(196, 132, 90, 0.2)",
    text: "#2d2420",
    textDim: "#6b5f54",
    textMuted: "#9a8d7f",
    vinylBlack: "#2a2420",
    vinylGroove: "#3d352e",
  },
  dark: {
    bg: "#1a1714",
    bgCard: "#231f1b",
    bgInput: "#2d2722",
    border: "#3d352e",
    borderHover: "#5a4d42",
    accent: "#e8a87c",
    accentDim: "#c4845a",
    accentGlow: "rgba(232, 168, 124, 0.15)",
    text: "#e8ddd0",
    textDim: "#9a8d7f",
    textMuted: "#6b5f54",
    vinylBlack: "#0d0b09",
    vinylGroove: "#2a2420",
  },
};

// Saturation curve for warm soft-clipping
function createSaturationCurve(amount) {
  const nSamples = 65536;
  const curve = new Float32Array(nSamples);
  for (let i = 0; i < nSamples; i++) {
    const x = (i * 2) / nSamples - 1;
    curve[i] = Math.tanh(x * amount);
  }
  return curve;
}

// Vocal removal via mid-side: cancels center-panned vocals
function removeVocals(audioBuffer, ctx) {
  if (audioBuffer.numberOfChannels < 2) return audioBuffer;
  const length = audioBuffer.length;
  const sampleRate = audioBuffer.sampleRate;
  const output = ctx.createBuffer(2, length, sampleRate);
  const left = audioBuffer.getChannelData(0);
  const right = audioBuffer.getChannelData(1);
  const outL = output.getChannelData(0);
  const outR = output.getChannelData(1);
  for (let i = 0; i < length; i++) {
    const side = (left[i] - right[i]) / 2;
    outL[i] = side;
    outR[i] = side;
  }
  return output;
}

// Generate vinyl crackle noise buffer
function createCrackleBuffer(audioCtx, duration = 4) {
  const sampleRate = audioCtx.sampleRate;
  const length = sampleRate * duration;
  const buffer = audioCtx.createBuffer(1, length, sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < length; i++) {
    // Sparse crackle pops
    if (Math.random() < 0.002) {
      data[i] = (Math.random() - 0.5) * 0.15;
    } else if (Math.random() < 0.01) {
      data[i] = (Math.random() - 0.5) * 0.03;
    } else {
      data[i] = (Math.random() - 0.5) * 0.003;
    }
  }
  return buffer;
}

// Bitcrusher via WaveShaperNode approximation
function createBitcrushCurve(bits) {
  const steps = Math.pow(2, bits);
  const nSamples = 65536;
  const curve = new Float32Array(nSamples);
  for (let i = 0; i < nSamples; i++) {
    const x = (i * 2) / nSamples - 1;
    curve[i] = Math.round(x * steps) / steps;
  }
  return curve;
}

const defaultParams = {
  lowpass: 2500,
  highpass: 120,
  crackleVol: 0.35,
  bitcrush: 12,
  saturation: 3,
  reverbMix: 0.4,
  speed: 0.90,
  wobbleDepth: 3,
  wobbleRate: 0.3,
  vocalRemoval: false,
};

function Slider({ label, value, min, max, step, onChange, unit = "", colors }) {
  const pct = ((value - min) / (max - min)) * 100;
  const display = typeof value === "number" ? (Number.isInteger(value) ? value : value.toFixed(2)) : value;

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
      <span style={{
        fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: colors.textDim,
        textTransform: "uppercase", letterSpacing: "0.1em", width: 72, flexShrink: 0,
      }}>
        {label}
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="lofi-slider"
        style={{
          flex: 1,
          "--pct": `${pct}%`,
          "--accent": colors.accent,
          "--accentDim": colors.accentDim,
          "--track": colors.bgInput,
          "--glow": colors.accentGlow,
        }}
      />
      <span style={{
        fontSize: 12, fontFamily: "'JetBrains Mono', monospace", color: colors.accent,
        width: 64, textAlign: "right", flexShrink: 0,
      }}>
        {display}{unit}
      </span>
    </div>
  );
}

function VinylSpinner({ isPlaying, colors }) {
  return (
    <div style={{
      width: 120, height: 120, borderRadius: "50%",
      background: `repeating-radial-gradient(circle, ${colors.vinylBlack} 0px, ${colors.vinylBlack} 3px, ${colors.vinylGroove} 3px, ${colors.vinylGroove} 4px)`,
      display: "flex", alignItems: "center", justifyContent: "center",
      animation: isPlaying ? "spin 3s linear infinite" : "none",
      boxShadow: `0 0 30px rgba(0,0,0,0.5), inset 0 0 20px rgba(0,0,0,0.3)`,
      border: `1px solid ${colors.border}`,
    }}>
      <div style={{
        width: 30, height: 30, borderRadius: "50%",
        background: `radial-gradient(circle, ${colors.accent} 0%, ${colors.accentDim} 40%, ${colors.bgCard} 100%)`,
        border: `2px solid ${colors.border}`,
      }} />
    </div>
  );
}

function WaveformDisplay({ analyserRef, isPlaying, colors }) {
  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const colorsRef = useRef(colors);
  colorsRef.current = colors;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    const draw = () => {
      const c = colorsRef.current;
      ctx.fillStyle = c.vinylBlack;
      ctx.fillRect(0, 0, W, H);

      if (analyserRef.current && isPlaying) {
        const analyser = analyserRef.current;
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyser.getByteTimeDomainData(dataArray);

        ctx.lineWidth = 2;
        ctx.strokeStyle = c.accent;
        ctx.shadowColor = c.accent;
        ctx.shadowBlur = 6;
        ctx.beginPath();
        const sliceWidth = W / bufferLength;
        let x = 0;
        for (let i = 0; i < bufferLength; i++) {
          const v = dataArray[i] / 128.0;
          const y = (v * H) / 2;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
          x += sliceWidth;
        }
        ctx.lineTo(W, H / 2);
        ctx.stroke();
        ctx.shadowBlur = 0;
      } else {
        ctx.strokeStyle = c.border;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, H / 2);
        ctx.lineTo(W, H / 2);
        ctx.stroke();
      }

      animRef.current = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(animRef.current);
  }, [isPlaying, analyserRef]);

  return (
    <canvas
      ref={canvasRef}
      width={500}
      height={80}
      style={{ width: "100%", height: 80, borderRadius: 8, border: `1px solid ${colors.border}` }}
    />
  );
}

export default function LofiConverter() {
  const [theme, setTheme] = useState("light");
  const [inputMode, setInputMode] = useState("file"); // "file" or "youtube"
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [fileName, setFileName] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [params, setParams] = useState(defaultParams);
  const [audioLoaded, setAudioLoaded] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");
  const [isExporting, setIsExporting] = useState(false);
  const [ytHint, setYtHint] = useState(null);

  const C = THEMES[theme];

  const audioCtxRef = useRef(null);
  const sourceRef = useRef(null);
  const audioBufferRef = useRef(null);
  const originalBufferRef = useRef(null);
  const nodesRef = useRef({});
  const analyserRef = useRef(null);
  const startTimeRef = useRef(0);
  const offsetRef = useRef(0);
  const fileInputRef = useRef(null);

  const initAudioContext = () => {
    if (!audioCtxRef.current) {
      audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtxRef.current.state === "suspended") {
      audioCtxRef.current.resume();
    }
    return audioCtxRef.current;
  };

  const buildGraph = useCallback(
    (ctx, source) => {
      // High-pass filter
      const hpf = ctx.createBiquadFilter();
      hpf.type = "highpass";
      hpf.frequency.value = params.highpass;
      hpf.Q.value = 0.7;

      // Low-pass filter
      const lpf = ctx.createBiquadFilter();
      lpf.type = "lowpass";
      lpf.frequency.value = params.lowpass;
      lpf.Q.value = 0.7;

      // Saturation
      const saturation = ctx.createWaveShaper();
      saturation.curve = createSaturationCurve(params.saturation);
      saturation.oversample = "4x";

      // Bitcrusher
      const bitcrush = ctx.createWaveShaper();
      bitcrush.curve = createBitcrushCurve(params.bitcrush);
      bitcrush.oversample = "none";

      // Reverb via convolver
      const convolver = ctx.createConvolver();
      const reverbLen = ctx.sampleRate * 2.5;
      const reverbBuf = ctx.createBuffer(2, reverbLen, ctx.sampleRate);
      for (let ch = 0; ch < 2; ch++) {
        const data = reverbBuf.getChannelData(ch);
        for (let i = 0; i < reverbLen; i++) {
          data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / reverbLen, 2.5);
        }
      }
      convolver.buffer = reverbBuf;

      const dryGain = ctx.createGain();
      dryGain.gain.value = 1 - params.reverbMix;
      const wetGain = ctx.createGain();
      wetGain.gain.value = params.reverbMix;
      const reverbMerge = ctx.createGain();

      // Crackle
      const crackleSource = ctx.createBufferSource();
      crackleSource.buffer = createCrackleBuffer(ctx);
      crackleSource.loop = true;
      const crackleGain = ctx.createGain();
      crackleGain.gain.value = params.crackleVol;

      // Tape wobble LFO
      const wobbleLfo = ctx.createOscillator();
      wobbleLfo.type = "sine";
      wobbleLfo.frequency.value = params.wobbleRate;
      const wobbleGain = ctx.createGain();
      wobbleGain.gain.value = params.wobbleDepth;
      wobbleLfo.connect(wobbleGain);
      wobbleGain.connect(source.detune);
      wobbleLfo.start();

      // Analyser
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      analyserRef.current = analyser;

      // Master
      const master = ctx.createGain();
      master.gain.value = 0.85;

      // Chain: source -> hpf -> lpf -> saturation -> bitcrush -> dry/wet reverb -> merge -> master -> analyser -> dest
      source.connect(hpf);
      hpf.connect(lpf);
      lpf.connect(saturation);
      saturation.connect(bitcrush);

      bitcrush.connect(dryGain);
      bitcrush.connect(convolver);
      convolver.connect(wetGain);
      dryGain.connect(reverbMerge);
      wetGain.connect(reverbMerge);

      reverbMerge.connect(master);
      crackleSource.connect(crackleGain);
      crackleGain.connect(master);
      master.connect(analyser);
      analyser.connect(ctx.destination);

      crackleSource.start();

      nodesRef.current = { hpf, lpf, saturation, bitcrush, convolver, dryGain, wetGain, crackleGain, crackleSource, wobbleLfo, wobbleGain, master, analyser };
      return nodesRef.current;
    },
    [params]
  );

  const handleFileSelect = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setStatusMsg("Decoding audio...");
    setIsProcessing(true);
    setFileName(file.name);

    try {
      const ctx = initAudioContext();
      const arrayBuf = await file.arrayBuffer();
      const audioBuf = await ctx.decodeAudioData(arrayBuf);
      originalBufferRef.current = audioBuf;
      audioBufferRef.current = params.vocalRemoval ? removeVocals(audioBuf, ctx) : audioBuf;
      setAudioLoaded(true);
      setStatusMsg(`Loaded: ${file.name} (${(audioBuf.duration / 60).toFixed(1)}min)`);
    } catch (err) {
      setStatusMsg("Error decoding audio file. Try a different format.");
      console.error(err);
    }
    setIsProcessing(false);
  };

  const handleYoutubeSubmit = async () => {
    if (!youtubeUrl.trim()) {
      setYtHint({ error: true, text: "Paste a YouTube URL first." });
      return;
    }
    setIsProcessing(true);
    setYtHint({ error: false, text: "Fetching audio from YouTube..." });

    try {
      const res = await fetch("http://localhost:5001/fetch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: youtubeUrl.trim() }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `Server error (${res.status})`);
      }

      const blob = await res.blob();
      const ctx = initAudioContext();
      const arrayBuf = await blob.arrayBuffer();
      const audioBuf = await ctx.decodeAudioData(arrayBuf);
      originalBufferRef.current = audioBuf;
      audioBufferRef.current = params.vocalRemoval ? removeVocals(audioBuf, ctx) : audioBuf;
      setAudioLoaded(true);
      setFileName("youtube-audio.mp3");
      setYtHint({ error: false, text: `Loaded (${(audioBuf.duration / 60).toFixed(1)}min)` });
    } catch (err) {
      if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
        setYtHint({ error: true, text: "Cannot reach server. Is python server.py running?" });
      } else {
        setYtHint({ error: true, text: err.message });
      }
      console.error(err);
    }
    setIsProcessing(false);
  };

  const stopPlayback = () => {
    if (sourceRef.current) {
      try { sourceRef.current.stop(); } catch (e) {}
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }
    if (nodesRef.current.crackleSource) {
      try { nodesRef.current.crackleSource.stop(); } catch (e) {}
    }
    if (nodesRef.current.wobbleLfo) {
      try { nodesRef.current.wobbleLfo.stop(); } catch (e) {}
    }
    setIsPlaying(false);
  };

  const handlePlay = () => {
    if (isPlaying) {
      const ctx = audioCtxRef.current;
      if (ctx) offsetRef.current += ctx.currentTime - startTimeRef.current;
      stopPlayback();
      return;
    }

    if (!audioBufferRef.current) return;
    const ctx = initAudioContext();
    const source = ctx.createBufferSource();
    source.buffer = audioBufferRef.current;
    source.onended = () => {
      setIsPlaying(false);
      offsetRef.current = 0;
    };
    source.playbackRate.value = params.speed;
    sourceRef.current = source;
    buildGraph(ctx, source);
    source.start(0, offsetRef.current);
    startTimeRef.current = ctx.currentTime;
    setIsPlaying(true);
  };

  const handleStop = () => {
    stopPlayback();
    offsetRef.current = 0;
  };

  const handleExport = async () => {
    if (!audioBufferRef.current) return;
    setIsExporting(true);
    setStatusMsg("Rendering lofi version...");

    try {
      const originalBuf = audioBufferRef.current;
      const outputLength = Math.ceil(originalBuf.length / params.speed);
      const offlineCtx = new OfflineAudioContext(
        originalBuf.numberOfChannels,
        outputLength,
        originalBuf.sampleRate
      );

      const source = offlineCtx.createBufferSource();
      source.buffer = originalBuf;
      source.playbackRate.value = params.speed;

      // High-pass filter
      const hpf = offlineCtx.createBiquadFilter();
      hpf.type = "highpass";
      hpf.frequency.value = params.highpass;
      hpf.Q.value = 0.7;

      // Low-pass filter
      const lpf = offlineCtx.createBiquadFilter();
      lpf.type = "lowpass";
      lpf.frequency.value = params.lowpass;
      lpf.Q.value = 0.7;

      // Saturation
      const saturation = offlineCtx.createWaveShaper();
      saturation.curve = createSaturationCurve(params.saturation);
      saturation.oversample = "4x";

      const bitcrush = offlineCtx.createWaveShaper();
      bitcrush.curve = createBitcrushCurve(params.bitcrush);

      const convolver = offlineCtx.createConvolver();
      const reverbLen = offlineCtx.sampleRate * 2.5;
      const reverbBuf = offlineCtx.createBuffer(2, reverbLen, offlineCtx.sampleRate);
      for (let ch = 0; ch < 2; ch++) {
        const data = reverbBuf.getChannelData(ch);
        for (let i = 0; i < reverbLen; i++) {
          data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / reverbLen, 2.5);
        }
      }
      convolver.buffer = reverbBuf;

      const dryGain = offlineCtx.createGain();
      dryGain.gain.value = 1 - params.reverbMix;
      const wetGain = offlineCtx.createGain();
      wetGain.gain.value = params.reverbMix;
      const merge = offlineCtx.createGain();
      const master = offlineCtx.createGain();
      master.gain.value = 0.85;

      const crackleSource = offlineCtx.createBufferSource();
      crackleSource.buffer = createCrackleBuffer(offlineCtx, originalBuf.duration + 1);
      const crackleGain = offlineCtx.createGain();
      crackleGain.gain.value = params.crackleVol;

      // Tape wobble LFO
      const wobbleLfo = offlineCtx.createOscillator();
      wobbleLfo.type = "sine";
      wobbleLfo.frequency.value = params.wobbleRate;
      const wobbleGain = offlineCtx.createGain();
      wobbleGain.gain.value = params.wobbleDepth;
      wobbleLfo.connect(wobbleGain);
      wobbleGain.connect(source.detune);

      source.connect(hpf);
      hpf.connect(lpf);
      lpf.connect(saturation);
      saturation.connect(bitcrush);
      bitcrush.connect(dryGain);
      bitcrush.connect(convolver);
      convolver.connect(wetGain);
      dryGain.connect(merge);
      wetGain.connect(merge);
      merge.connect(master);
      crackleSource.connect(crackleGain);
      crackleGain.connect(master);
      master.connect(offlineCtx.destination);

      source.start();
      crackleSource.start();
      wobbleLfo.start();

      const rendered = await offlineCtx.startRendering();

      // Encode as WAV
      const numCh = rendered.numberOfChannels;
      const length = rendered.length;
      const sr = rendered.sampleRate;
      const bitsPerSample = 16;
      const byteRate = sr * numCh * (bitsPerSample / 8);
      const blockAlign = numCh * (bitsPerSample / 8);
      const dataSize = length * blockAlign;
      const bufferSize = 44 + dataSize;
      const buffer = new ArrayBuffer(bufferSize);
      const view = new DataView(buffer);

      const writeString = (offset, str) => {
        for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
      };

      writeString(0, "RIFF");
      view.setUint32(4, bufferSize - 8, true);
      writeString(8, "WAVE");
      writeString(12, "fmt ");
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, numCh, true);
      view.setUint32(24, sr, true);
      view.setUint32(28, byteRate, true);
      view.setUint16(32, blockAlign, true);
      view.setUint16(34, bitsPerSample, true);
      writeString(36, "data");
      view.setUint32(40, dataSize, true);

      const channels = [];
      for (let ch = 0; ch < numCh; ch++) channels.push(rendered.getChannelData(ch));

      let offset = 44;
      for (let i = 0; i < length; i++) {
        for (let ch = 0; ch < numCh; ch++) {
          let sample = Math.max(-1, Math.min(1, channels[ch][i]));
          sample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
          view.setInt16(offset, sample, true);
          offset += 2;
        }
      }

      const blob = new Blob([buffer], { type: "audio/wav" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const baseName = fileName.replace(/\.[^.]+$/, "");
      a.download = `${baseName || "track"}_lofi.wav`;
      a.click();
      URL.revokeObjectURL(url);
      setStatusMsg("Export complete!");
    } catch (err) {
      setStatusMsg("Export failed: " + err.message);
      console.error(err);
    }
    setIsExporting(false);
  };

  // Live-update params on playing nodes
  useEffect(() => {
    const n = nodesRef.current;
    if (n.hpf) n.hpf.frequency.value = params.highpass;
    if (n.lpf) n.lpf.frequency.value = params.lowpass;
    if (n.saturation) n.saturation.curve = createSaturationCurve(params.saturation);
    if (n.bitcrush) n.bitcrush.curve = createBitcrushCurve(params.bitcrush);
    if (n.dryGain) n.dryGain.gain.value = 1 - params.reverbMix;
    if (n.wetGain) n.wetGain.gain.value = params.reverbMix;
    if (n.crackleGain) n.crackleGain.gain.value = params.crackleVol;
    if (n.wobbleGain) n.wobbleGain.gain.value = params.wobbleDepth;
    if (n.wobbleLfo) n.wobbleLfo.frequency.value = params.wobbleRate;
    if (sourceRef.current) sourceRef.current.playbackRate.value = params.speed;
  }, [params]);

  const updateParam = (key) => (val) => setParams((p) => ({ ...p, [key]: val }));

  const toggleVocalRemoval = () => {
    if (!originalBufferRef.current) return;
    const ctx = initAudioContext();
    const newVal = !params.vocalRemoval;
    if (newVal) {
      audioBufferRef.current = removeVocals(originalBufferRef.current, ctx);
    } else {
      audioBufferRef.current = originalBufferRef.current;
    }
    setParams((p) => ({ ...p, vocalRemoval: newVal }));
    if (isPlaying) {
      // Restart playback with new buffer
      stopPlayback();
      offsetRef.current = 0;
    }
  };

  return (
    <div style={{
      minHeight: "100vh", background: C.bg, color: C.text,
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      display: "flex", flexDirection: "column", alignItems: "center",
      padding: "40px 20px", transition: "background 0.3s, color 0.3s",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Playfair+Display:ital,wght@0,700;1,700&display=swap');
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes pulse { 0%, 100% { opacity: 0.6; } 50% { opacity: 1; } }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::selection { background: ${C.accent}; color: ${C.bg}; }
        .lofi-slider {
          -webkit-appearance: none;
          appearance: none;
          height: 6px;
          border-radius: 3px;
          background: linear-gradient(to right, var(--accent) 0%, var(--accentDim) var(--pct), var(--track) var(--pct));
          outline: none;
          cursor: pointer;
        }
        .lofi-slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 16px;
          height: 16px;
          border-radius: 50%;
          background: var(--accent);
          box-shadow: 0 0 8px var(--glow);
          border: 2px solid var(--accentDim);
          cursor: grab;
        }
        .lofi-slider::-moz-range-thumb {
          width: 16px;
          height: 16px;
          border-radius: 50%;
          background: var(--accent);
          box-shadow: 0 0 8px var(--glow);
          border: 2px solid var(--accentDim);
          cursor: grab;
        }
        .lofi-slider:active::-webkit-slider-thumb { cursor: grabbing; }
        .lofi-slider:active::-moz-range-thumb { cursor: grabbing; }
      `}</style>

      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 40, position: "relative", width: "100%", maxWidth: 800 }}>
        <button
          onClick={() => setTheme(t => t === "light" ? "dark" : "light")}
          style={{
            position: "absolute", right: 0, top: 0,
            padding: "6px 14px", borderRadius: 6,
            border: `1px solid ${C.border}`, background: C.bgCard,
            color: C.textDim, cursor: "pointer", fontFamily: "inherit",
            fontSize: 12, transition: "all 0.3s",
          }}
        >
          {theme === "light" ? "dark" : "light"}
        </button>
        <h1 style={{
          fontFamily: "'Playfair Display', serif", fontSize: 42, fontWeight: 700,
          fontStyle: "italic", color: C.accent, letterSpacing: "-0.02em",
          textShadow: `0 0 40px ${C.accentGlow}`,
        }}>
          lofi machine
        </h1>
      </div>

      <div style={{ width: "100%", maxWidth: 800, display: "flex", flexDirection: "column", gap: 24 }}>

        {/* Input Mode Toggle */}
        <div style={{
          display: "flex", gap: 0, borderRadius: 8, overflow: "hidden",
          border: `1px solid ${C.border}`,
        }}>
          {[["file", "File Upload"], ["youtube", "YouTube URL"]].map(([mode, label]) => (
            <button
              key={mode}
              onClick={() => setInputMode(mode)}
              style={{
                flex: 1, padding: "12px 16px", border: "none", cursor: "pointer",
                background: inputMode === mode ? C.accent : C.bgCard,
                color: inputMode === mode ? C.bg : C.textDim,
                fontFamily: "inherit", fontSize: 12, fontWeight: 500,
                letterSpacing: "0.08em", textTransform: "uppercase",
                transition: "all 0.2s",
              }}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Input Area */}
        <div style={{
          background: C.bgCard, borderRadius: 12, padding: 24,
          border: `1px solid ${C.border}`,
        }}>
          {inputMode === "file" ? (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={handleFileSelect}
                style={{ display: "none" }}
              />
              <div
                onClick={() => fileInputRef.current?.click()}
                style={{
                  width: "100%", padding: "32px 20px", borderRadius: 8,
                  border: `2px dashed ${C.border}`, cursor: "pointer",
                  textAlign: "center", transition: "border-color 0.2s",
                }}
                onMouseEnter={(e) => (e.currentTarget.style.borderColor = C.accent)}
                onMouseLeave={(e) => (e.currentTarget.style.borderColor = C.border)}
              >
                <div style={{ fontSize: 28, marginBottom: 8 }}>🎵</div>
                <div style={{ fontSize: 13, color: C.textDim }}>
                  {fileName || "Click to select an audio file"}
                </div>
                <div style={{ fontSize: 11, color: C.textMuted, marginTop: 4 }}>
                  MP3, WAV, FLAC, OGG, M4A
                </div>
              </div>
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <div style={{ display: "flex", gap: 8 }}>
                <input
                  type="text"
                  placeholder="https://youtube.com/watch?v=..."
                  value={youtubeUrl}
                  onChange={(e) => setYoutubeUrl(e.target.value)}
                  style={{
                    flex: 1, padding: "12px 16px", borderRadius: 8,
                    background: C.bgInput, border: `1px solid ${C.border}`,
                    color: C.text, fontFamily: "inherit", fontSize: 13,
                    outline: "none",
                  }}
                  onFocus={(e) => (e.target.style.borderColor = C.accent)}
                  onBlur={(e) => (e.target.style.borderColor = C.border)}
                />
                <button
                  onClick={handleYoutubeSubmit}
                  disabled={isProcessing}
                  style={{
                    padding: "12px 20px", borderRadius: 8, border: "none",
                    background: isProcessing ? C.bgInput : C.accent,
                    color: isProcessing ? C.textMuted : C.bg,
                    cursor: isProcessing ? "not-allowed" : "pointer",
                    fontFamily: "inherit", fontSize: 12, fontWeight: 600,
                  }}
                >
                  {isProcessing ? "Fetching..." : "Fetch"}
                </button>
              </div>
              <div style={{
                padding: 12, borderRadius: 8, background: C.bgInput,
                fontSize: 11, color: ytHint?.error ? "#d9534f" : C.textMuted, lineHeight: 1.6,
              }}>
                {ytHint ? ytHint.text : <>Requires <code style={{ color: C.accent }}>python server.py</code> running locally.</>}
              </div>
            </div>
          )}

          {/* Vocal Removal Toggle */}
          {audioLoaded && (
            <div style={{ marginTop: 16, display: "flex", justifyContent: "center" }}>
              <button
                onClick={toggleVocalRemoval}
                style={{
                  padding: "8px 20px", borderRadius: 6,
                  border: `1px solid ${params.vocalRemoval ? C.accent : C.border}`,
                  background: params.vocalRemoval ? C.accent : "transparent",
                  color: params.vocalRemoval ? C.bg : C.textDim,
                  cursor: "pointer", fontFamily: "inherit",
                  fontSize: 12, fontWeight: 500, letterSpacing: "0.05em",
                  transition: "all 0.2s",
                }}
              >
                Vocals: {params.vocalRemoval ? "OFF" : "ON"}
              </button>
            </div>
          )}
        </div>

        {/* Vinyl + Waveform */}
        <div style={{
          background: C.bgCard, borderRadius: 12, padding: 24,
          border: `1px solid ${C.border}`,
          display: "flex", flexDirection: "column", alignItems: "center", gap: 20,
        }}>
          <VinylSpinner isPlaying={isPlaying} colors={C} />
          <WaveformDisplay analyserRef={analyserRef} isPlaying={isPlaying} colors={C} />

          {/* Transport controls */}
          <div style={{ display: "flex", gap: 12 }}>
            <button
              onClick={handlePlay}
              disabled={!audioLoaded || isProcessing}
              style={{
                padding: "10px 32px", borderRadius: 8, border: "none",
                background: audioLoaded ? C.accent : C.bgInput,
                color: audioLoaded ? C.bg : C.textMuted,
                cursor: audioLoaded ? "pointer" : "not-allowed",
                fontFamily: "inherit", fontSize: 13, fontWeight: 600,
                letterSpacing: "0.05em", transition: "all 0.2s",
              }}
            >
              {isPlaying ? "⏸ Pause" : "▶ Play"}
            </button>
            <button
              onClick={handleStop}
              disabled={!audioLoaded}
              style={{
                padding: "10px 20px", borderRadius: 8,
                border: `1px solid ${C.border}`, background: "transparent",
                color: audioLoaded ? C.textDim : C.textMuted,
                cursor: audioLoaded ? "pointer" : "not-allowed",
                fontFamily: "inherit", fontSize: 13,
              }}
            >
              ⏹ Stop
            </button>
            <button
              onClick={handleExport}
              disabled={!audioLoaded || isExporting}
              style={{
                padding: "10px 20px", borderRadius: 8,
                border: `1px solid ${audioLoaded ? C.accent : C.border}`,
                background: "transparent",
                color: audioLoaded ? C.accent : C.textMuted,
                cursor: audioLoaded ? "pointer" : "not-allowed",
                fontFamily: "inherit", fontSize: 13, fontWeight: 500,
                animation: isExporting ? "pulse 1.5s ease-in-out infinite" : "none",
              }}
            >
              {isExporting ? "Rendering..." : "⬇ Export WAV"}
            </button>
          </div>
        </div>

        {/* Effect Knobs */}
        <div style={{
          background: C.bgCard, borderRadius: 12, padding: 28,
          border: `1px solid ${C.border}`,
        }}>
          <div style={{
            fontSize: 11, color: C.textMuted, textTransform: "uppercase",
            letterSpacing: "0.15em", marginBottom: 20, textAlign: "center",
          }}>
            Effect Parameters
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <Slider label="High-Pass" value={params.highpass} min={20} max={500} step={10} onChange={updateParam("highpass")} unit="Hz" colors={C} />
            <Slider label="Low-Pass" value={params.lowpass} min={500} max={10000} step={100} onChange={updateParam("lowpass")} unit="Hz" colors={C} />
            <Slider label="Saturate" value={params.saturation} min={1} max={10} step={0.5} onChange={updateParam("saturation")} colors={C} />
            <Slider label="Crackle" value={params.crackleVol} min={0} max={1} step={0.01} onChange={updateParam("crackleVol")} colors={C} />
            <Slider label="Bitcrush" value={params.bitcrush} min={3} max={16} step={1} onChange={updateParam("bitcrush")} unit="bit" colors={C} />
            <Slider label="Reverb" value={params.reverbMix} min={0} max={0.9} step={0.01} onChange={updateParam("reverbMix")} colors={C} />
            <Slider label="Wobble" value={params.wobbleDepth} min={0} max={8} step={0.5} onChange={updateParam("wobbleDepth")} unit="ct" colors={C} />
            <Slider label="Wob Rate" value={params.wobbleRate} min={0.1} max={2} step={0.1} onChange={updateParam("wobbleRate")} unit="Hz" colors={C} />
            <Slider label="Speed" value={params.speed} min={0.75} max={1.0} step={0.01} onChange={updateParam("speed")} unit="x" colors={C} />
          </div>
        </div>

        {/* Status */}
        {statusMsg && (
          <div style={{
            padding: "10px 16px", borderRadius: 8,
            background: C.bgCard, border: `1px solid ${C.border}`,
            fontSize: 12, color: C.textDim, textAlign: "center",
          }}>
            {statusMsg}
          </div>
        )}

        {/* Reset */}
        <button
          onClick={() => setParams(defaultParams)}
          style={{
            alignSelf: "center", padding: "8px 20px", borderRadius: 6,
            border: `1px solid ${C.border}`, background: "transparent",
            color: C.textMuted, cursor: "pointer", fontFamily: "inherit",
            fontSize: 11, letterSpacing: "0.1em", textTransform: "uppercase",
          }}
        >
          Reset to defaults
        </button>
      </div>
    </div>
  );
}
