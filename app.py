"""
FastAPI Backend for EMOPIA Emotion-Conditioned Music Generator.
Serves real-time neural music generation, piano audio synthesis, and music theory metrics.
"""

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import pretty_midi
import scipy.io.wavfile
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    import lameenc
    HAS_LAMEENC = True
except ImportError:
    HAS_LAMEENC = False

# ── Vocabulary & Tokenizer Setup ──────────────────────────────────────────────
DURATION_EDGES = np.array([0.0, 0.06, 0.12, 0.20, 0.30, 0.45, 0.65, 0.9, 1.3, 1.8, 2.5, np.inf])
NUM_DUR_BUCKETS = len(DURATION_EDGES) - 1  # 11
REST_TOKEN = 0
TOKEN_OFFSET = 1

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

EMOTION_META = {
    1: {
        "id": 1,
        "quadrant": "Q1",
        "name": "Happy / Excited",
        "short": "happy",
        "valence": "High",
        "arousal": "High",
        "valence_score": 0.85,
        "arousal_score": 0.85,
        "color": "#ea580c",
        "badge_bg": "#fff7ed",
        "badge_text": "#c2410c",
        "description": "Euphoric, energetic melodies with bright harmonies and rapid rhythmic cadence.",
    },
    2: {
        "id": 2,
        "quadrant": "Q2",
        "name": "Angry / Tense",
        "short": "angry",
        "valence": "Low",
        "arousal": "High",
        "valence_score": -0.80,
        "arousal_score": 0.90,
        "color": "#dc2626",
        "badge_bg": "#fef2f2",
        "badge_text": "#b91c1c",
        "description": "Dramatic, tense, and turbulent progressions with sharp dissonance and driving rhythm.",
    },
    3: {
        "id": 3,
        "quadrant": "Q3",
        "name": "Sad / Depressed",
        "short": "sad",
        "valence": "Low",
        "arousal": "Low",
        "valence_score": -0.85,
        "arousal_score": -0.75,
        "color": "#4f46e5",
        "badge_bg": "#eef2ff",
        "badge_text": "#4338ca",
        "description": "Melancholic, poignant minor phrases with slow, lingering notes and reflective spacing.",
    },
    4: {
        "id": 4,
        "quadrant": "Q4",
        "name": "Calm / Relaxed",
        "short": "calm",
        "valence": "High",
        "arousal": "Low",
        "valence_score": 0.80,
        "arousal_score": -0.70,
        "color": "#059669",
        "badge_bg": "#ecfdf5",
        "badge_text": "#047857",
        "description": "Peaceful, pastoral piano voicing with gentle transitions and warm resonant warmth.",
    },
}

# ── Model Architecture ────────────────────────────────────────────────────────
class EmotionMusicLSTM(nn.Module):
    def __init__(self, vocab_size, num_emotions=4,
                 embed_dim=128, emotion_embed_dim=32,
                 hidden1=512, hidden2=256, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emotion_embed = nn.Embedding(num_emotions, emotion_embed_dim)
        lstm_input_dim = embed_dim + emotion_embed_dim
        self.lstm1   = nn.LSTM(lstm_input_dim, hidden1, batch_first=True)
        self.drop1   = nn.Dropout(dropout)
        self.lstm2   = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.drop2   = nn.Dropout(dropout)
        self.fc_out  = nn.Linear(hidden2, vocab_size)

    def forward(self, token_seq, emotion_ids):
        B, T = token_seq.shape
        tok_emb = self.token_embed(token_seq)
        emo_emb = self.emotion_embed(emotion_ids)
        emo_emb = emo_emb.unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([tok_emb, emo_emb], dim=-1)
        x, _ = self.lstm1(x)
        x = self.drop1(x)
        x, _ = self.lstm2(x)
        x = self.drop2(x)
        last = x[:, -1, :]
        return self.fc_out(last)


# ── Global Model Loader ───────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
MODEL = None
VOCAB_SIZE = 1157
SEQ_LEN = 64
CKPT_PATH = Path("best_emotion_lstm.pt")
GENERATED_DIR = Path("static/generated")
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    global MODEL, VOCAB_SIZE, SEQ_LEN
    if not CKPT_PATH.exists():
        print(f"[!] Warning: Checkpoint {CKPT_PATH} not found.")
        return None

    print(f"[+] Loading model from {CKPT_PATH} onto {DEVICE} ({DEVICE_NAME})...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    VOCAB_SIZE = ckpt.get("vocab_size", 1157)
    SEQ_LEN = ckpt.get("seq_len", 64)

    m = EmotionMusicLSTM(vocab_size=VOCAB_SIZE).to(DEVICE)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    MODEL = m
    print(f"[OK] Model successfully loaded (Epoch {ckpt.get('epoch')}, val_loss: {ckpt.get('val_loss'):.4f})")
    return MODEL


# ── Musical Synthesis & Audio Helpers ─────────────────────────────────────────
def pitch_to_name(pitch: int) -> str:
    octave = (pitch // 12) - 1
    note = NOTE_NAMES[pitch % 12]
    return f"{note}{octave}"


def get_seed_for_emotion(emotion_id: int, seq_len: int = 64, midi_dir: str = "EMOPIA_1.0/midis") -> List[int]:
    """Retrieve seed notes from genuine EMOPIA clips."""
    q_str = f"Q{emotion_id + 1}_"
    mp = Path(midi_dir)
    if mp.exists():
        matches = list(mp.glob(f"*{q_str}*.mid"))
        if matches:
            pm = pretty_midi.PrettyMIDI(str(matches[0]))
            notes = []
            for inst in pm.instruments:
                if not inst.is_drum:
                    for n in inst.notes:
                        notes.append((n.start, n.pitch, n.velocity, n.end - n.start))
            notes.sort(key=lambda x: x[0])
            toks = []
            for i, (onset, pitch, vel, dur) in enumerate(notes):
                ioi = notes[i + 1][0] - onset if i < len(notes) - 1 else dur
                dur_bucket = min(int(np.searchsorted(DURATION_EDGES[1:], ioi, side="left")), NUM_DUR_BUCKETS - 1)
                toks.append(pitch * NUM_DUR_BUCKETS + dur_bucket + TOKEN_OFFSET)
            if len(toks) >= 16:
                return toks[:seq_len]

    # Harmonious fallback progressions
    fallback_chords = {
        0: [60, 64, 67, 72, 65, 69, 72, 76], # Major (Happy)
        1: [57, 60, 63, 66, 58, 62, 65, 68], # Diminished/Dissonant (Angry)
        2: [57, 60, 64, 69, 53, 57, 60, 65], # Minor (Sad)
        3: [60, 64, 67, 71, 62, 65, 69, 72], # Major 7th (Calm)
    }
    pitches = fallback_chords.get(emotion_id, [60, 64, 67, 72])
    return [p * NUM_DUR_BUCKETS + 4 + TOKEN_OFFSET for p in pitches]


def tokens_to_pm(tokens: List[int], bpm: int = 120, velocity: int = 85) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    piano = pretty_midi.Instrument(program=0, name="Acoustic Grand Piano")
    current_time = 0.0
    dur_centers = [0.03, 0.09, 0.16, 0.25, 0.375, 0.55, 0.775, 1.1, 1.55, 2.15, 3.0]

    for token in tokens:
        if token <= 0:
            continue
        token_val = token - TOKEN_OFFSET
        dur_bucket = token_val % NUM_DUR_BUCKETS
        pitch = token_val // NUM_DUR_BUCKETS
        if pitch < 0 or pitch > 127:
            continue

        ioi = dur_centers[dur_bucket] if dur_bucket < len(dur_centers) else 0.5
        note_dur = max(ioi * 0.9, 0.05)
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(pitch),
            start=current_time,
            end=current_time + note_dur,
        )
        piano.notes.append(note)
        current_time += ioi

    pm.instruments.append(piano)
    return pm


def synthesize_piano_audio(pm: pretty_midi.PrettyMIDI, fs: int = 44100) -> np.ndarray:
    """Rich piano physical modeling with harmonic overtones and percussive decay."""
    total_time = pm.get_end_time() + 1.0
    total_samples = int(np.ceil(total_time * fs))
    if total_samples <= 0:
        return np.zeros(fs, dtype=np.int16)

    audio = np.zeros(total_samples, dtype=np.float32)
    harmonics = [(1, 1.0), (2, 0.55), (3, 0.28), (4, 0.14), (5, 0.07), (6, 0.03)]

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            f0 = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
            start_idx = int(note.start * fs)
            dur = max(note.end - note.start, 0.05)
            dur_total = dur + min(0.35, dur * 0.5)
            n_samples = int(dur_total * fs)

            if start_idx >= total_samples or n_samples <= 0:
                continue

            t = np.arange(n_samples, dtype=np.float32) / fs
            attack_samples = int(0.004 * fs)
            decay_rate = 2.2 + 45.0 / (note.pitch + 1)
            env = np.exp(-t * decay_rate)
            if attack_samples > 0 and len(env) > attack_samples:
                env[:attack_samples] *= np.linspace(0.0, 1.0, attack_samples, dtype=np.float32)
            rel_len = int(0.04 * fs)
            if len(env) > rel_len:
                env[-rel_len:] *= np.linspace(1.0, 0.0, rel_len, dtype=np.float32)

            note_signal = np.zeros(n_samples, dtype=np.float32)
            for h_num, h_amp in harmonics:
                freq = f0 * h_num
                if freq < fs / 2:
                    note_signal += h_amp * np.sin(2 * np.pi * freq * t)

            vel_scale = (note.velocity / 127.0) ** 1.25
            note_signal *= env * vel_scale
            end_idx = min(start_idx + n_samples, total_samples)
            audio[start_idx:end_idx] += note_signal[:end_idx - start_idx]

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = (audio / peak) * 0.95
    return (audio * 32767).astype(np.int16)


def estimate_key(pitches: List[int]) -> str:
    """Krumhansl-Schmuckler-inspired pitch profile key estimation."""
    if not pitches:
        return "C Major"

    pitch_classes = [p % 12 for p in pitches]
    counts = np.bincount(pitch_classes, minlength=12)

    # Simplified profile matching: Major vs Minor
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    best_score = -1e9
    best_key = "C Major"

    for root in range(12):
        rolled = np.roll(counts, -root)
        maj_corr = np.corrcoef(rolled, major_profile)[0, 1]
        min_corr = np.corrcoef(rolled, minor_profile)[0, 1]

        if not np.isnan(maj_corr) and maj_corr > best_score:
            best_score = maj_corr
            best_key = f"{NOTE_NAMES[root]} Major"

        if not np.isnan(min_corr) and min_corr > best_score:
            best_score = min_corr
            best_key = f"{NOTE_NAMES[root]} Minor"

    return best_key


def calculate_metrics(pm: pretty_midi.PrettyMIDI, emotion_id: int, inference_time_ms: float = 0.0) -> Dict[str, Any]:
    """Extract deep musical, structural, and affective metrics from PrettyMIDI."""
    all_notes = []
    for inst in pm.instruments:
        if not inst.is_drum:
            all_notes.extend(inst.notes)

    duration = max(pm.get_end_time(), 0.1)
    notes_count = len(all_notes)

    if notes_count > 0:
        pitches = [n.pitch for n in all_notes]
        durations = [n.end - n.start for n in all_notes]
        pitch_min = min(pitches)
        pitch_max = max(pitches)
        pitch_mean = float(np.mean(pitches))
        pitch_range = pitch_max - pitch_min
        avg_dur = float(np.mean(durations))
        note_density = round(notes_count / duration, 2)
        dominant_key = estimate_key(pitches)
    else:
        pitch_min = 60
        pitch_max = 72
        pitch_mean = 66.0
        pitch_range = 12
        avg_dur = 0.5
        note_density = 0.0
        dominant_key = "C Major"

    meta = EMOTION_META[emotion_id + 1]

    return {
        "emotion_quadrant": meta["quadrant"],
        "emotion_name": meta["name"],
        "valence": meta["valence"],
        "arousal": meta["arousal"],
        "valence_score": meta["valence_score"],
        "arousal_score": meta["arousal_score"],
        "duration_seconds": round(duration, 1),
        "duration_formatted": f"{int(duration // 60)}:{int(duration % 60):02d}",
        "notes_count": notes_count,
        "note_density": note_density,  # notes / sec
        "tempo_bpm": 120,
        "pitch_min": pitch_min,
        "pitch_min_name": pitch_to_name(pitch_min),
        "pitch_max": pitch_max,
        "pitch_max_name": pitch_to_name(pitch_max),
        "pitch_mean": round(pitch_mean, 1),
        "pitch_range_semitones": pitch_range,
        "avg_note_duration_sec": round(avg_dur, 2),
        "dominant_key": dominant_key,
        "inference_time_ms": round(inference_time_ms, 1),
        "device": f"{DEVICE} ({DEVICE_NAME})",
    }


# ── FastAPI App Configuration ─────────────────────────────────────────────────
app = FastAPI(
    title="EMOPIA Neural Music Studio",
    description="Emotion-conditioned pop piano music generation powered by PyTorch LSTM and EMOPIA dataset.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    emotion: int = Field(1, ge=1, le=4, description="Emotion Quadrant: 1=Happy, 2=Angry, 3=Sad, 4=Calm")
    length: int = Field(250, ge=50, le=500, description="Number of notes to generate")
    temperature: float = Field(0.85, ge=0.5, le=1.3, description="Sampling creativity/randomness")


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/api/status")
def get_status():
    return {
        "status": "ready" if MODEL is not None else "loading",
        "device": str(DEVICE),
        "device_name": DEVICE_NAME,
        "cuda_available": torch.cuda.is_available(),
        "vocab_size": VOCAB_SIZE,
        "model_checkpoint": "best_emotion_lstm.pt",
        "has_lameenc_mp3": HAS_LAMEENC,
    }


@app.get("/api/presets")
def get_presets():
    """Return the 4 reference emotion pieces generated during training."""
    presets = []
    for q_id, q_data in EMOTION_META.items():
        tag = q_data["quadrant"]
        short = q_data["short"]
        base_name = f"generated_{tag}_{short}"
        mid_file = GENERATED_DIR / f"{base_name}.mid"

        if mid_file.exists():
            pm = pretty_midi.PrettyMIDI(str(mid_file))
            metrics = calculate_metrics(pm, emotion_id=q_id - 1)
        else:
            metrics = {
                "emotion_quadrant": tag,
                "emotion_name": q_data["name"],
                "valence": q_data["valence"],
                "arousal": q_data["arousal"],
                "duration_seconds": 35.0,
                "duration_formatted": "0:35",
                "notes_count": 314,
                "note_density": 8.9,
                "tempo_bpm": 120,
                "dominant_key": "C Major",
                "pitch_min_name": "C3",
                "pitch_max_name": "G5",
                "pitch_range_semitones": 31,
                "avg_note_duration_sec": 0.25,
            }

        presets.append({
            "id": q_id,
            "quadrant": tag,
            "name": q_data["name"],
            "short": short,
            "color": q_data["color"],
            "badge_bg": q_data["badge_bg"],
            "badge_text": q_data["badge_text"],
            "description": q_data["description"],
            "midi_url": f"/static/generated/{base_name}.mid",
            "wav_url": f"/static/generated/{base_name}.wav",
            "mp3_url": f"/static/generated/{base_name}.mp3",
            "metrics": metrics,
        })
    return {"presets": presets}


@app.post("/api/generate")
def generate_endpoint(req: GenerateRequest):
    if MODEL is None:
        load_model()
        if MODEL is None:
            raise HTTPException(status_code=500, detail="Model could not be loaded")

    t_start = time.time()
    emotion_idx = req.emotion - 1
    seed = get_seed_for_emotion(emotion_idx, seq_len=SEQ_LEN)

    # 1. Autoregressive inference
    generated = list(seed)
    with torch.no_grad():
        for _ in range(req.length):
            context = generated[-SEQ_LEN:]
            if len(context) < SEQ_LEN:
                context = [0] * (SEQ_LEN - len(context)) + context

            ctx_t = torch.tensor([context], dtype=torch.long, device=DEVICE)
            emo_t = torch.tensor([emotion_idx], dtype=torch.long, device=DEVICE)

            logits = MODEL(ctx_t, emo_t).squeeze(0) / max(req.temperature, 1e-4)
            probs = torch.softmax(logits, dim=-1)

            next_tok = torch.multinomial(probs, num_samples=1).item()
            if next_tok == 0:
                next_tok = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_tok)

    inference_ms = (time.time() - t_start) * 1000

    # 2. Convert to MIDI
    pm = tokens_to_pm(generated, bpm=120)

    # 3. Unique filename
    meta = EMOTION_META[req.emotion]
    uid = uuid.uuid4().hex[:8]
    base_name = f"composition_{meta['quadrant']}_{meta['short']}_{uid}"

    midi_path = GENERATED_DIR / f"{base_name}.mid"
    wav_path  = GENERATED_DIR / f"{base_name}.wav"
    mp3_path  = GENERATED_DIR / f"{base_name}.mp3"

    pm.write(str(midi_path))

    # 4. Audio Synthesis
    fs = 44100
    audio_int16 = synthesize_piano_audio(pm, fs=fs)
    scipy.io.wavfile.write(str(wav_path), fs, audio_int16)

    # 5. MP3 encode
    if HAS_LAMEENC:
        encoder = lameenc.Encoder()
        encoder.set_bit_rate(192)
        encoder.set_in_sample_rate(fs)
        encoder.set_channels(1)
        encoder.set_quality(2)
        mp3_bytes = encoder.encode(audio_int16.tobytes()) + encoder.flush()
        with open(mp3_path, "wb") as f:
            f.write(mp3_bytes)

    # 6. Metrics extraction
    metrics = calculate_metrics(pm, emotion_id=emotion_idx, inference_time_ms=inference_ms)

    return {
        "success": True,
        "composition_id": uid,
        "emotion": meta,
        "midi_url": f"/static/generated/{base_name}.mid",
        "wav_url": f"/static/generated/{base_name}.wav",
        "mp3_url": f"/static/generated/{base_name}.mp3",
        "metrics": metrics,
    }


# Mount Static Files (Frontend UI & Audio Assets)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/", StaticFiles(directory="static", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
