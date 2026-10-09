/**
 * EMOPIA Studio — Client Application
 * Handles emotion selection, LSTM inference API calls, audio playback,
 * canvas visualizer, and dynamic metrics visualization.
 */

document.addEventListener("DOMContentLoaded", () => {
  // ── State ──────────────────────────────────────────────────────────────────
  const state = {
    selectedEmotion: 1, // 1=Happy, 2=Angry, 3=Sad, 4=Calm
    length: 250,
    temperature: 0.85,
    isPlaying: false,
    currentTrack: null,
    presets: [],
  };

  const EMOTION_THEMES = {
    1: { accent: "#ea580c", bg: "#fff7ed", border: "#fed7aa" },
    2: { accent: "#dc2626", bg: "#fef2f2", border: "#fecaca" },
    3: { accent: "#4f46e5", bg: "#eef2ff", border: "#c7d2fe" },
    4: { accent: "#059669", bg: "#ecfdf5", border: "#a7f3d0" },
  };

  // ── DOM References ─────────────────────────────────────────────────────────
  const statusLabel = document.getElementById("deviceStatusText");
  const quadrantSelector = document.getElementById("quadrantSelector");
  const lengthSlider = document.getElementById("lengthSlider");
  const lengthDisplay = document.getElementById("lengthDisplay");
  const tempSlider = document.getElementById("tempSlider");
  const tempDisplay = document.getElementById("tempDisplay");
  const generateBtn = document.getElementById("generateBtn");

  // Player Stage
  const stageEmotionPill = document.getElementById("stageEmotionPill");
  const stageTrackTitle = document.getElementById("stageTrackTitle");
  const stageTrackMeta = document.getElementById("stageTrackMeta");
  const downloadMp3Btn = document.getElementById("downloadMp3Btn");
  const downloadWavBtn = document.getElementById("downloadWavBtn");
  const downloadMidiBtn = document.getElementById("downloadMidiBtn");

  const audioPlayer = document.getElementById("audioPlayer");
  const playPauseBtn = document.getElementById("playPauseBtn");
  const playIcon = document.getElementById("playIcon");
  const pauseIcon = document.getElementById("pauseIcon");
  const playheadTrack = document.getElementById("playheadTrack");
  const playheadProgress = document.getElementById("playheadProgress");
  const currentTimeDisplay = document.getElementById("currentTimeDisplay");
  const totalTimeDisplay = document.getElementById("totalTimeDisplay");
  const volumeSlider = document.getElementById("volumeSlider");
  const visualizerCanvas = document.getElementById("visualizerCanvas");
  const vCtx = visualizerCanvas.getContext("2d");

  // Metrics
  const metricValenceArousal = document.getElementById("metricValenceArousal");
  const metricCircumplex = document.getElementById("metricCircumplex");
  const metricDuration = document.getElementById("metricDuration");
  const metricTempo = document.getElementById("metricTempo");
  const metricDensity = document.getElementById("metricDensity");
  const metricTotalNotes = document.getElementById("metricTotalNotes");
  const metricKey = document.getElementById("metricKey");
  const metricPitchRange = document.getElementById("metricPitchRange");
  const metricPitchSpan = document.getElementById("metricPitchSpan");
  const metricLatency = document.getElementById("metricLatency");
  const metricDevice = document.getElementById("metricDevice");

  const presetsList = document.getElementById("presetsList");

  // ── Visualizer Animation ───────────────────────────────────────────────────
  let animFrameId = null;
  const numBars = 64;
  const barHeights = new Array(numBars).fill(8);

  function drawVisualizer() {
    const width = visualizerCanvas.width;
    const height = visualizerCanvas.height;
    vCtx.clearRect(0, 0, width, height);

    const theme = EMOTION_THEMES[state.selectedEmotion] || EMOTION_THEMES[1];
    const barWidth = (width / numBars) - 2;

    for (let i = 0; i < numBars; i++) {
      if (state.isPlaying) {
        // Dynamic rhythmic wave pattern
        const t = performance.now() * 0.003;
        const target = Math.sin(t + i * 0.3) * 20 + Math.cos(t * 1.5 - i * 0.2) * 15 + 40;
        barHeights[i] += (target - barHeights[i]) * 0.25;
      } else {
        barHeights[i] += (8 - barHeights[i]) * 0.1;
      }

      const h = Math.max(4, Math.min(height - 10, barHeights[i]));
      const x = i * (barWidth + 2);
      const y = height - h - 14;

      // Subtle gradient
      const grad = vCtx.createLinearGradient(0, y, 0, height);
      grad.addColorStop(0, theme.accent);
      grad.addColorStop(1, "rgba(255, 255, 255, 0.1)");

      vCtx.fillStyle = grad;
      vCtx.beginPath();
      vCtx.roundRect(x, y, barWidth, h, [3, 3, 0, 0]);
      vCtx.fill();
    }

    animFrameId = requestAnimationFrame(drawVisualizer);
  }

  // ── Theme / Quadrant Switching ─────────────────────────────────────────────
  function setEmotion(emotionId) {
    state.selectedEmotion = emotionId;
    const theme = EMOTION_THEMES[emotionId];

    document.documentElement.style.setProperty("--theme-accent", theme.accent);
    document.documentElement.style.setProperty("--theme-bg", theme.bg);
    document.documentElement.style.setProperty("--theme-border", theme.border);

    // Update active card
    document.querySelectorAll(".emotion-card").forEach((card) => {
      card.classList.toggle("active", parseInt(card.dataset.emotion) === emotionId);
    });
  }

  quadrantSelector.addEventListener("click", (e) => {
    const card = e.target.closest(".emotion-card");
    if (card) {
      const eid = parseInt(card.dataset.emotion);
      setEmotion(eid);
    }
  });

  // ── Sliders Sync ───────────────────────────────────────────────────────────
  lengthSlider.addEventListener("input", (e) => {
    state.length = parseInt(e.target.value);
    lengthDisplay.textContent = `${state.length} notes`;
  });

  tempSlider.addEventListener("input", (e) => {
    state.temperature = parseFloat(e.target.value);
    tempDisplay.textContent = state.temperature.toFixed(2);
  });

  // ── Metrics Display ────────────────────────────────────────────────────────
  function updateMetrics(m) {
    if (!m) return;
    metricValenceArousal.textContent = `${m.valence} / ${m.arousal}`;
    if (m.valence_score !== undefined) {
      const vSign = m.valence_score > 0 ? "+" : "";
      const aSign = m.arousal_score > 0 ? "+" : "";
      metricCircumplex.textContent = `Coord: (${vSign}${m.valence_score}, ${aSign}${m.arousal_score})`;
    }
    metricDuration.textContent = `${m.duration_seconds}s`;
    metricTempo.textContent = `${m.tempo_bpm || 120} BPM • 4/4 Time`;
    metricDensity.textContent = `${m.note_density} notes/s`;
    metricTotalNotes.textContent = `${m.notes_count} total notes`;
    metricKey.textContent = m.dominant_key || "C Major";
    metricPitchRange.textContent = `${m.pitch_min_name || 'C3'} – ${m.pitch_max_name || 'G5'}`;
    metricPitchSpan.textContent = `${m.pitch_range_semitones || 24} semitones span`;
    if (m.inference_time_ms) {
      metricLatency.textContent = `${m.inference_time_ms} ms`;
    }
    if (m.device) {
      metricDevice.textContent = m.device;
    }
  }

  // ── Load Track Into Player ─────────────────────────────────────────────────
  function loadTrack(trackData, autoPlay = false) {
    state.currentTrack = trackData;
    const m = trackData.metrics || {};

    setEmotion(trackData.emotion?.id || trackData.id || 1);

    stageEmotionPill.textContent = `${m.emotion_quadrant || 'Q1'} • ${m.emotion_name || 'Happy / Excited'}`;
    stageTrackTitle.textContent = trackData.name || `Composition #${trackData.composition_id || '01'}`;
    stageTrackMeta.textContent = `LSTM Autoregressive • ${m.duration_seconds || 35}s • ${m.dominant_key || 'C Major'}`;

    // Download links
    downloadMp3Btn.href = trackData.mp3_url || "#";
    downloadWavBtn.href = trackData.wav_url || "#";
    downloadMidiBtn.href = trackData.midi_url || "#";

    // Set audio source (prefer MP3, fallback WAV)
    const audioSrc = trackData.mp3_url || trackData.wav_url;
    audioPlayer.src = audioSrc;
    audioPlayer.load();

    totalTimeDisplay.textContent = m.duration_formatted || "0:35";
    currentTimeDisplay.textContent = "0:00";
    playheadProgress.style.width = "0%";

    updateMetrics(m);

    if (autoPlay) {
      audioPlayer.play().then(() => {
        setPlaying(true);
      }).catch(() => {
        setPlaying(false);
      });
    } else {
      setPlaying(false);
    }
  }

  // ── Audio Transport ────────────────────────────────────────────────────────
  function setPlaying(playing) {
    state.isPlaying = playing;
    playIcon.style.display = playing ? "none" : "block";
    pauseIcon.style.display = playing ? "block" : "none";
  }

  playPauseBtn.addEventListener("click", () => {
    if (!audioPlayer.src) return;
    if (audioPlayer.paused) {
      audioPlayer.play();
      setPlaying(true);
    } else {
      audioPlayer.pause();
      setPlaying(false);
    }
  });

  audioPlayer.addEventListener("timeupdate", () => {
    if (!audioPlayer.duration) return;
    const cur = audioPlayer.currentTime;
    const dur = audioPlayer.duration;
    const pct = (cur / dur) * 100;
    playheadProgress.style.width = `${pct}%`;

    const cMin = Math.floor(cur / 60);
    const cSec = Math.floor(cur % 60);
    currentTimeDisplay.textContent = `${cMin}:${cSec < 10 ? '0' : ''}${cSec}`;
  });

  audioPlayer.addEventListener("loadedmetadata", () => {
    const dur = audioPlayer.duration;
    const dMin = Math.floor(dur / 60);
    const dSec = Math.floor(dur % 60);
    totalTimeDisplay.textContent = `${dMin}:${dSec < 10 ? '0' : ''}${dSec}`;
  });

  audioPlayer.addEventListener("ended", () => {
    setPlaying(false);
    playheadProgress.style.width = "0%";
    currentTimeDisplay.textContent = "0:00";
  });

  playheadTrack.addEventListener("click", (e) => {
    if (!audioPlayer.duration) return;
    const rect = playheadTrack.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, clickX / rect.width));
    audioPlayer.currentTime = ratio * audioPlayer.duration;
  });

  volumeSlider.addEventListener("input", (e) => {
    audioPlayer.volume = parseFloat(e.target.value);
  });

  // ── API: Generate Music ────────────────────────────────────────────────────
  generateBtn.addEventListener("click", async () => {
    generateBtn.classList.add("loading");
    generateBtn.querySelector(".btn-text").textContent = "Composing...";

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          emotion: state.selectedEmotion,
          length: state.length,
          temperature: state.temperature,
        }),
      });

      if (!response.ok) {
        throw new Error(`Generation failed: ${response.statusText}`);
      }

      const data = await response.json();
      if (data.success) {
        loadTrack({
          ...data,
          name: `${data.emotion.name} Piece`,
        }, true);
      }
    } catch (err) {
      console.error(err);
      alert("Composition failed. Please verify the Python backend is running.");
    } finally {
      generateBtn.classList.remove("loading");
      generateBtn.querySelector(".btn-text").textContent = "Generate Music";
    }
  });

  // ── Init: Status & Presets ─────────────────────────────────────────────────
  async function initApp() {
    drawVisualizer();

    try {
      const statusRes = await fetch("/api/status");
      const statusData = await statusRes.json();
      if (statusData.cuda_available) {
        statusLabel.textContent = `${statusData.device_name} • CUDA Ready`;
      } else {
        statusLabel.textContent = `CPU Inference • Ready`;
      }
    } catch (e) {
      statusLabel.textContent = "Connecting to Engine...";
    }

    try {
      const presRes = await fetch("/api/presets");
      const presData = await presRes.json();
      state.presets = presData.presets || [];

      // Render preset tiles
      presetsList.innerHTML = "";
      state.presets.forEach((p) => {
        const tile = document.createElement("div");
        tile.className = "preset-tile";
        tile.innerHTML = `
          <div>
            <div class="preset-tile-header">
              <span class="preset-badge" style="background: ${p.badge_bg}; color: ${p.color};">${p.quadrant}</span>
              <span class="preset-time">${p.metrics?.duration_seconds || 35}s</span>
            </div>
            <h4 class="preset-name">${p.name}</h4>
            <p class="preset-meta-info">${p.description}</p>
          </div>
          <div style="margin-top: 10px; font-size: 0.72rem; color: ${p.color}; font-weight: 600;">
            Click to Play & View Metrics &rarr;
          </div>
        `;
        tile.addEventListener("click", () => {
          loadTrack(p, true);
        });
        presetsList.appendChild(tile);
      });

      // Load first preset by default
      if (state.presets.length > 0) {
        loadTrack(state.presets[0], false);
      }
    } catch (e) {
      console.warn("Could not load presets:", e);
    }
  }

  initApp();
});
