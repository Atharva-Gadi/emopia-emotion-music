# %% [markdown]
# # [NOTE] Emotion-Conditioned Music Generation with LSTM
# ## Trained on the EMOPIA Dataset
#
# **EMOPIA** (Emotion-labeled Pop Piano) contains ~1,087 MIDI clips from 387 songs,
# each labeled with one of four emotion quadrants based on Russell's Valence-Arousal model:
#
# | Quadrant | Valence | Arousal | Emotion |
# |----------|---------|---------|---------|
# | Q1 | High | High | Happy / Excited |
# | Q2 | Low | High | Angry / Tense |
# | Q3 | Low | Low | Sad / Depressed |
# | Q4 | High | Low | Calm / Relaxed |
#
# The model learns to predict the next musical token conditioned on an emotion label,
# so at generation time the user simply picks an emotion and the LSTM composes
# a piece that reflects that mood.
#
# **Citation:**
# ```
# Hung et al., "EMOPIA: A Multi-Modal Pop Piano Dataset For Emotion Recognition
# and Emotion-based Music Generation", ISMIR 2021.
# ```

# %% [markdown]
# ## 0. Install Dependencies
# Run this cell once to install all required packages.

# %%
import subprocess
import sys

packages = [
    "pretty_midi",
    "mido",
    "numpy",
    "scipy",
    "lameenc",
    "matplotlib",
    "tqdm",
    "requests",
]
for pkg in packages:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("[OK] All dependencies verified.")

# %% [markdown]
# ## 1. Download & Extract the EMOPIA Dataset

# %%
import os
import zipfile
import requests
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
DATASET_URL = "https://zenodo.org/records/5257995/files/EMOPIA_1.0.zip?download=1"
DATASET_ZIP = Path("EMOPIA_1.0.zip")
DATASET_DIR = Path("EMOPIA_1.0")
MIDI_DIR    = DATASET_DIR / "midis"

# ── Download ───────────────────────────────────────────────────────────────
if not DATASET_ZIP.exists():
    print(f"[DOWNLOAD]  Downloading EMOPIA dataset (~5.5 MB) …")
    resp = requests.get(DATASET_URL, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(DATASET_ZIP, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r   {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", end="")
    print("\n[OK] Download complete.")
else:
    print(f"[OK] {DATASET_ZIP} already exists, skipping download.")

# ── Extract ────────────────────────────────────────────────────────────────
if not MIDI_DIR.exists():
    print("[EXTRACT] Extracting …")
    with zipfile.ZipFile(DATASET_ZIP, "r") as zf:
        zf.extractall(".")
    print(f"[OK] Extracted to {DATASET_DIR}")
else:
    print(f"[OK] {MIDI_DIR} already exists, skipping extraction.")

midi_files = sorted(MIDI_DIR.glob("*.mid"))
print(f"\n[PIANO] Found {len(midi_files)} MIDI files in {MIDI_DIR}")

# %% [markdown]
# ## 2. Parse MIDI Files & Extract Emotion Labels
#
# Each filename encodes the emotion quadrant: `Q1_xxxxxxx_2.mid`
# We extract `(onset_time, pitch, velocity, duration)` per note using `pretty_midi`,
# then pair each clip with its emotion label.

# %%
import pretty_midi
import numpy as np
from tqdm import tqdm
from collections import defaultdict

EMOTION_MAP = {
    "Q1": 0,  # Happy / Excited    (high valence, high arousal)
    "Q2": 1,  # Angry / Tense      (low valence,  high arousal)
    "Q3": 2,  # Sad   / Depressed   (low valence,  low arousal)
    "Q4": 3,  # Calm  / Relaxed     (high valence, low arousal)
}
EMOTION_NAMES = {0: "Happy/Excited", 1: "Angry/Tense", 2: "Sad/Depressed", 3: "Calm/Relaxed"}
NUM_EMOTIONS = len(EMOTION_MAP)


def parse_midi(midi_path: Path):
    """Parse a MIDI file → sorted list of (onset_time, pitch, velocity, duration)."""
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        return None  # skip corrupt files

    notes = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            notes.append((note.start, note.pitch, note.velocity, note.end - note.start))
    if not notes:
        return None

    # Sort by onset time, then by pitch (for deterministic ordering of chords)
    notes.sort(key=lambda x: (x[0], x[1]))
    return notes


def get_emotion_label(filename: str) -> int:
    """Extract the emotion quadrant from the EMOPIA filename convention."""
    basename = Path(filename).stem  # e.g. "Q1_abc123_2"
    quadrant = basename.split("_")[0].upper()  # "Q1"
    return EMOTION_MAP.get(quadrant, -1)


# ── Parse all files ────────────────────────────────────────────────────────
all_clips = []   # list of (notes_list, emotion_label)
skipped = 0

for midi_path in tqdm(midi_files, desc="Parsing MIDI files"):
    emotion = get_emotion_label(midi_path.name)
    if emotion < 0:
        skipped += 1
        continue
    notes = parse_midi(midi_path)
    if notes is None or len(notes) < 10:
        skipped += 1
        continue
    all_clips.append((notes, emotion))

print(f"\n[OK] Parsed {len(all_clips)} clips  (skipped {skipped})")

# ── Distribution ───────────────────────────────────────────────────────────
counts = defaultdict(int)
for _, emo in all_clips:
    counts[emo] += 1
print("\nEmotion distribution:")
for emo_id in sorted(counts):
    print(f"  {EMOTION_NAMES[emo_id]:20s}  (Q{emo_id+1}):  {counts[emo_id]:4d} clips")

# %% [markdown]
# ## 3. Tokenization
#
# Each note → token = `(pitch_class, octave_bucket, duration_bucket)` mapped to an integer.
# We use inter-onset intervals for duration, binned into discrete buckets.
#
# This keeps the vocabulary manageable while preserving both pitch and rhythmic information.

# %%

# ── Duration bucket edges (seconds) ───────────────────────────────────────
DURATION_EDGES = np.array([0.0, 0.06, 0.12, 0.20, 0.30, 0.45, 0.65, 0.9, 1.3, 1.8, 2.5, np.inf])
NUM_DUR_BUCKETS = len(DURATION_EDGES) - 1  # 11

# Special tokens
REST_TOKEN = 0
TOKEN_OFFSET = 1  # real tokens start at 1


def notes_to_tokens(notes):
    """Convert a list of (onset, pitch, velocity, duration) → integer token sequence."""
    tokens = []
    for i, (onset, pitch, vel, dur) in enumerate(notes):
        # Inter-onset interval to the next note (for rhythm)
        if i < len(notes) - 1:
            ioi = notes[i + 1][0] - onset
        else:
            ioi = dur  # last note: use its own duration

        dur_bucket = int(np.searchsorted(DURATION_EDGES[1:], ioi, side="left"))
        dur_bucket = min(dur_bucket, NUM_DUR_BUCKETS - 1)

        # Token = pitch * NUM_DUR_BUCKETS + dur_bucket + TOKEN_OFFSET
        token = pitch * NUM_DUR_BUCKETS + dur_bucket + TOKEN_OFFSET
        tokens.append(token)

    return tokens


# ── Tokenize all clips ────────────────────────────────────────────────────
tokenized_clips = []  # list of (token_sequence, emotion_label)

for notes, emotion in tqdm(all_clips, desc="Tokenizing"):
    toks = notes_to_tokens(notes)
    tokenized_clips.append((toks, emotion))

# Compute vocabulary size
all_tokens_flat = [t for toks, _ in tokenized_clips for t in toks]
vocab_size = max(all_tokens_flat) + 1  # includes REST_TOKEN=0

total_notes = sum(len(toks) for toks, _ in tokenized_clips)
print(f"\n[OK] Tokenized {len(tokenized_clips)} clips")
print(f"   Total tokens:     {total_notes:,}")
print(f"   Vocabulary size:  {vocab_size}")
print(f"   Duration buckets: {NUM_DUR_BUCKETS}")

# %% [markdown]
# ## 4. Build Training Windows (Emotion-Conditioned)
#
# For each clip, we slide a window of length `SEQ_LEN` across the tokens.
# Each training sample is `(emotion_label, context_tokens[0:SEQ_LEN]) → next_token`.
# The emotion label is prepended as an additional feature to condition the LSTM.

# %%
import torch
from torch.utils.data import Dataset, DataLoader

SEQ_LEN = 64       # context window length
BATCH_SIZE = 128

class EmotionMusicDataset(Dataset):
    """
    Index-based sliding-window dataset (memory-efficient).
    Stores clips as numpy arrays and computes windows lazily in __getitem__.
    """

    def __init__(self, tokenized_clips, seq_len, vocab_size):
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Store clips as numpy arrays for fast slicing
        self.clip_arrays = []
        self.clip_emotions = []
        self.indices = []  # (clip_idx, start_pos)

        for clip_idx, (tokens, emotion) in enumerate(tokenized_clips):
            if len(tokens) <= seq_len:
                continue
            arr = np.array(tokens, dtype=np.int64)
            self.clip_arrays.append(arr)
            self.clip_emotions.append(emotion)
            num_windows = len(tokens) - seq_len
            real_clip_idx = len(self.clip_arrays) - 1
            for start in range(num_windows):
                self.indices.append((real_clip_idx, start))

        print(f"   Built {len(self.indices):,} training windows  (seq_len={seq_len})")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        clip_idx, start = self.indices[idx]
        arr = self.clip_arrays[clip_idx]
        context = arr[start : start + self.seq_len]
        target  = arr[start + self.seq_len]
        emotion = self.clip_emotions[clip_idx]
        return (
            torch.from_numpy(context.copy()),
            torch.tensor(emotion, dtype=torch.long),
            torch.tensor(target,  dtype=torch.long),
        )


# ── Train / Validation split (80/20, by clip, not by window) ──────────────
np.random.seed(42)
indices = np.random.permutation(len(tokenized_clips))
split = int(0.8 * len(indices))
train_clips = [tokenized_clips[i] for i in indices[:split]]
val_clips   = [tokenized_clips[i] for i in indices[split:]]

print(f"Train clips: {len(train_clips)},  Val clips: {len(val_clips)}")

train_dataset = EmotionMusicDataset(train_clips, SEQ_LEN, vocab_size)
val_dataset   = EmotionMusicDataset(val_clips,   SEQ_LEN, vocab_size)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

print(f"\nTrain batches: {len(train_loader)},  Val batches: {len(val_loader)}")

# %% [markdown]
# ## 5. Define the Emotion-Conditioned LSTM Model
#
# Architecture:
# ```
# Token Embedding  (vocab_size → embed_dim)
#        +
# Emotion Embedding (4 → emotion_embed_dim)  ──┐
#        │                                      │
#        └──────── Concatenated ────────────────┘
#                      │
#              LSTM Layer 1 (hidden=512)
#              Dropout (0.3)
#              LSTM Layer 2 (hidden=256)
#              Dropout (0.3)
#                      │
#              Fully Connected → vocab_size
#              (cross-entropy loss)
# ```

# %%
import torch.nn as nn

class EmotionMusicLSTM(nn.Module):
    """Two-layer LSTM conditioned on emotion via a learned emotion embedding."""

    def __init__(self, vocab_size, num_emotions=4,
                 embed_dim=128, emotion_embed_dim=32,
                 hidden1=512, hidden2=256, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size

        # Embeddings
        self.token_embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emotion_embed = nn.Embedding(num_emotions, emotion_embed_dim)

        lstm_input_dim = embed_dim + emotion_embed_dim

        # Stacked LSTM
        self.lstm1   = nn.LSTM(lstm_input_dim, hidden1, batch_first=True)
        self.drop1   = nn.Dropout(dropout)
        self.lstm2   = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.drop2   = nn.Dropout(dropout)

        # Classification head
        self.fc_out  = nn.Linear(hidden2, vocab_size)

    def forward(self, token_seq, emotion_ids):
        """
        token_seq:   (batch, seq_len)  — integer token indices
        emotion_ids: (batch,)          — emotion quadrant 0-3
        Returns:     (batch, vocab_size) logits for the next token
        """
        B, T = token_seq.shape

        # Embed tokens: (B, T, embed_dim)
        tok_emb = self.token_embed(token_seq)

        # Embed emotion and broadcast across time: (B, T, emotion_embed_dim)
        emo_emb = self.emotion_embed(emotion_ids)       # (B, emotion_embed_dim)
        emo_emb = emo_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, emotion_embed_dim)

        # Concatenate: (B, T, embed_dim + emotion_embed_dim)
        x = torch.cat([tok_emb, emo_emb], dim=-1)

        # LSTM layers
        x, _ = self.lstm1(x)
        x = self.drop1(x)
        x, _ = self.lstm2(x)
        x = self.drop2(x)

        # Take the last time-step output → next-token prediction
        last = x[:, -1, :]        # (B, hidden2)
        logits = self.fc_out(last) # (B, vocab_size)
        return logits


# ── Instantiate ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE]  Using device: {device}")

model = EmotionMusicLSTM(vocab_size, num_emotions=NUM_EMOTIONS).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"[MODEL] Model parameters: {num_params:,}")
print(model)

# %% [markdown]
# ## 6. Training Loop
#
# - Optimizer: AdamW with weight decay
# - Scheduler: Cosine annealing with warm restarts
# - Best model checkpoint saved by validation loss

# %%
import time
import matplotlib.pyplot as plt

# ── Hyperparameters ────────────────────────────────────────────────────────
NUM_EPOCHS      = 30
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

# ── History tracking ───────────────────────────────────────────────────────
history = {
    "train_loss": [], "val_loss": [],
    "train_acc": [],  "val_acc": [],
}

best_val_loss = float("inf")
best_model_path = "best_emotion_lstm.pt"

# ── Training ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Training for {NUM_EPOCHS} epochs  |  device={device}")
print(f"{'='*60}\n")

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()

    # ─── Train ─────────────────────────────────────────────────────────
    model.train()
    train_loss_sum, train_correct, train_total = 0.0, 0, 0

    for context, emotion, target in train_loader:
        context = context.to(device)
        emotion = emotion.to(device)
        target  = target.to(device)

        logits = model(context, emotion)
        loss = criterion(logits, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        train_loss_sum += loss.item() * context.size(0)
        preds = logits.argmax(dim=-1)
        train_correct += (preds == target).sum().item()
        train_total += context.size(0)

    scheduler.step()

    train_loss = train_loss_sum / train_total
    train_acc  = train_correct / train_total * 100

    # ─── Validate ──────────────────────────────────────────────────────
    model.eval()
    val_loss_sum, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for context, emotion, target in val_loader:
            context = context.to(device)
            emotion = emotion.to(device)
            target  = target.to(device)

            logits = model(context, emotion)
            loss = criterion(logits, target)

            val_loss_sum += loss.item() * context.size(0)
            preds = logits.argmax(dim=-1)
            val_correct += (preds == target).sum().item()
            val_total += context.size(0)

    val_loss = val_loss_sum / max(val_total, 1)
    val_acc  = val_correct / max(val_total, 1) * 100

    # ─── Checkpoint ────────────────────────────────────────────────────
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "seq_len": SEQ_LEN,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }, best_model_path)
        marker = "  *BEST* best"
    else:
        marker = ""

    elapsed = time.time() - t0
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  |  "
          f"train_loss={train_loss:.4f}  train_acc={train_acc:5.2f}%  |  "
          f"val_loss={val_loss:.4f}  val_acc={val_acc:5.2f}%  |  "
          f"{elapsed:.1f}s{marker}")

print(f"\n[OK] Training complete.  Best val loss: {best_val_loss:.4f}")
print(f"   Model saved to: {best_model_path}")

# %% [markdown]
# ## 7. Plot Training Curves

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs_range = range(1, NUM_EPOCHS + 1)

# Loss
ax1.plot(epochs_range, history["train_loss"], "o-", label="Train Loss", linewidth=2)
ax1.plot(epochs_range, history["val_loss"],   "s-", label="Val Loss",   linewidth=2)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.set_title("Loss Curves")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy
ax2.plot(epochs_range, history["train_acc"], "o-", label="Train Accuracy", linewidth=2)
ax2.plot(epochs_range, history["val_acc"],   "s-", label="Val Accuracy",   linewidth=2)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Accuracy Curves")
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.suptitle("Emotion-Conditioned LSTM — EMOPIA Training", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("training_curves_emopia.png", dpi=150, bbox_inches="tight")
plt.close()
print("[CHART] Saved training_curves_emopia.png")

# %% [markdown]
# ## 8. Music Generation — Choose Your Emotion! [EMOTION]
#
# The user picks an emotion, and the model generates a new piece conditioned on it.

# %%
def generate_music(model, emotion_id, seed_tokens, gen_length=200,
                   temperature=1.0, seq_len=SEQ_LEN, device=device):
    """
    Autoregressively generate a token sequence conditioned on `emotion_id`.

    Args:
        model:        trained EmotionMusicLSTM
        emotion_id:   int 0-3 (Q1-Q4)
        seed_tokens:  list of ints — initial context (at least a few tokens)
        gen_length:   number of new tokens to generate
        temperature:  sampling temperature (lower = more deterministic)
        seq_len:      context window size
        device:       torch device

    Returns:
        list[int] — the full generated token sequence
    """
    model.eval()
    generated = list(seed_tokens)

    with torch.no_grad():
        for _ in range(gen_length):
            # Take the last seq_len tokens as context
            context = generated[-seq_len:]
            # Pad if shorter
            if len(context) < seq_len:
                context = [0] * (seq_len - len(context)) + context

            ctx_tensor = torch.tensor([context], dtype=torch.long, device=device)
            emo_tensor = torch.tensor([emotion_id], dtype=torch.long, device=device)

            logits = model(ctx_tensor, emo_tensor)  # (1, vocab_size)
            logits = logits.squeeze(0) / temperature

            # Sample from the distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Avoid the REST/padding token
            if next_token == 0:
                next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token)

    return generated


import scipy.io.wavfile
try:
    import lameenc
    HAS_LAMEENC = True
except ImportError:
    HAS_LAMEENC = False


def synthesize_piano_audio(pm, fs=44100):
    """Synthesize MIDI into a natural piano sound with harmonic overtones and percussive decay."""
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


def tokens_to_midi_and_audio(tokens, base_name, bpm=120, velocity=85, fs=44100):
    """Convert tokens to MIDI (.mid), WAV (.wav), and MP3 (.mp3)."""
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

    midi_path = f"{base_name}.mid"
    wav_path  = f"{base_name}.wav"
    mp3_path  = f"{base_name}.mp3"

    pm.write(midi_path)

    # Synthesize audio
    audio_int16 = synthesize_piano_audio(pm, fs=fs)
    scipy.io.wavfile.write(wav_path, fs, audio_int16)

    # Encode MP3
    if HAS_LAMEENC:
        encoder = lameenc.Encoder()
        encoder.set_bit_rate(192)
        encoder.set_in_sample_rate(fs)
        encoder.set_channels(1)
        encoder.set_quality(2)
        mp3_bytes = encoder.encode(audio_int16.tobytes()) + encoder.flush()
        with open(mp3_path, "wb") as f:
            f.write(mp3_bytes)

    return current_time

# %% [markdown]
# ## 9. [EMOTION] Interactive Generation — Pick an Emotion!

# %%
# ── Load best model ────────────────────────────────────────────────────────
checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"[OK] Loaded best model (epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f})")

# ── Choose your emotion ───────────────────────────────────────────────────
print("\n" + "="*50)
print("  [EMOTION] EMOTION-CONDITIONED MUSIC GENERATOR")
print("="*50)
print()
print("  Choose an emotion to compose music:")
print("  1. [HAPPY] Happy / Excited   (Q1 — high valence, high arousal)")
print("  2. [ANGRY] Angry / Tense     (Q2 — low valence,  high arousal)")
print("  3. [SAD] Sad / Depressed   (Q3 — low valence,  low arousal)")
print("  4. [CALM] Calm / Relaxed    (Q4 — high valence, low arousal)")
print()

try:
    choice = int(input("Enter 1-4: "))
except (ValueError, EOFError, OSError):
    choice = 1
    print(f"(non-interactive mode: defaulting to {choice} = Happy/Excited)")

emotion_id = max(0, min(3, choice - 1))
print(f"\n[NOTE] Generating music with emotion: {EMOTION_NAMES[emotion_id]} …")

# ── Pick a seed from a clip with the same emotion ─────────────────────────
same_emotion_clips = [(t, e) for t, e in tokenized_clips if e == emotion_id]
if same_emotion_clips:
    seed_clip = same_emotion_clips[0][0]
else:
    seed_clip = tokenized_clips[0][0]

seed_tokens = seed_clip[:min(SEQ_LEN, len(seed_clip))]

# ── Generate ───────────────────────────────────────────────────────────────
generated_tokens = generate_music(
    model,
    emotion_id=emotion_id,
    seed_tokens=seed_tokens,
    gen_length=250,
    temperature=0.9,
)

# ── Export to MIDI, WAV, and MP3 ───────────────────────────────────────────
emotion_tag = f"Q{emotion_id + 1}"
base_output = f"generated_{emotion_tag}_{EMOTION_NAMES[emotion_id].split('/')[0].lower()}"

duration = tokens_to_midi_and_audio(generated_tokens, base_output, bpm=120)

print(f"\n[OK] Generated song saved to:")
print(f"   MIDI:     {base_output}.mid")
print(f"   Audio:    {base_output}.wav & {base_output}.mp3")
print(f"   Emotion:  {EMOTION_NAMES[emotion_id]}")
print(f"   Notes:    {len(generated_tokens)}")
print(f"   Duration: {duration:.1f} seconds ({duration/60:.1f} min)")
print(f"\n[LISTEN] Play the .mp3 or .wav directly in any audio player!")

# %% [markdown]
# ## 10. Generate All Four Emotions (Batch)
#
# Generate one piece for each emotion for easy comparison.

# %%
print("\n" + "="*60)
print("  Generating one piece per emotion for comparison (MIDI + Audio) …")
print("="*60 + "\n")

for emo_id in range(NUM_EMOTIONS):
    emo_name = EMOTION_NAMES[emo_id]
    tag = f"Q{emo_id + 1}"

    # Seed from a matching clip
    matching = [(t, e) for t, e in tokenized_clips if e == emo_id]
    if matching:
        seed = matching[0][0][:SEQ_LEN]
    else:
        seed = tokenized_clips[0][0][:SEQ_LEN]

    tokens = generate_music(model, emo_id, seed, gen_length=250, temperature=0.9)
    base_out = f"generated_{tag}_{emo_name.split('/')[0].lower().strip()}"
    dur = tokens_to_midi_and_audio(tokens, base_out, bpm=120)

    print(f"  {tag} {emo_name:20s} → {base_out}.[mid|wav|mp3]  ({dur:.1f}s, {len(tokens)} notes)")

print("\n[OK] All four emotion-conditioned pieces generated in MIDI, WAV, and MP3!")
print("   Compare them side-by-side to hear how emotion conditioning affects the output.")

# %% [markdown]
# ## 📋 Summary
#
# | Component | Detail |
# |-----------|--------|
# | **Dataset** | EMOPIA v1.0 — 1,078 MIDI clips, 4 emotion quadrants |
# | **Tokenization** | `(pitch × 11 duration_buckets) + offset` → integer token |
# | **Model** | 2-layer LSTM (512→256) with emotion embedding concatenation |
# | **Conditioning** | Emotion label embedded and broadcast across all time steps |
# | **Training** | AdamW + cosine annealing, best-validation-loss checkpoint |
# | **Generation** | Autoregressive sampling with temperature scaling |
# | **Output** | Standard MIDI files playable in any MIDI software |
#
# ### How emotion conditioning works
#
# The emotion is **not** a post-processing filter — it's baked into every LSTM step:
# 1. The emotion ID (0-3) is mapped through a learned `Embedding(4, 32)`
# 2. This 32-dim vector is concatenated with every token embedding at every timestep
# 3. The LSTM sees `[token_info | emotion_info]` at each step
# 4. Over training, different emotions learn different transition patterns
#
# This means the model genuinely learns different note/rhythm distributions
# per emotion, rather than generating generic music and relabeling it.
