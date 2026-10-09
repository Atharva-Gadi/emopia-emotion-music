"""
Generate emotion-conditioned music using the trained EMOPIA LSTM model.
Automatically exports to MIDI (.mid), Audio WAV (.wav), and MP3 (.mp3).
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import pretty_midi
import scipy.io.wavfile

try:
    import lameenc
    HAS_LAMEENC = True
except ImportError:
    HAS_LAMEENC = False

# ── Vocabulary & Tokenizer Parameters ─────────────────────────────────────────
DURATION_EDGES = np.array([0.0, 0.06, 0.12, 0.20, 0.30, 0.45, 0.65, 0.9, 1.3, 1.8, 2.5, np.inf])
NUM_DUR_BUCKETS = len(DURATION_EDGES) - 1  # 11
REST_TOKEN = 0
TOKEN_OFFSET = 1

EMOTION_NAMES = {
    0: "Happy/Excited (Q1)",
    1: "Angry/Tense (Q2)",
    2: "Sad/Depressed (Q3)",
    3: "Calm/Relaxed (Q4)",
}

EMOTION_SHORT = {
    0: "happy",
    1: "angry",
    2: "sad",
    3: "calm",
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


def parse_clip(midi_path):
    """Parse notes from a MIDI file."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    for inst in pm.instruments:
        if not inst.is_drum:
            for n in inst.notes:
                notes.append((n.start, n.pitch, n.velocity, n.end - n.start))
    notes.sort(key=lambda x: x[0])
    return notes


def notes_to_tokens(notes):
    """Convert notes to integer tokens."""
    tokens = []
    for i, (onset, pitch, vel, dur) in enumerate(notes):
        if i < len(notes) - 1:
            ioi = notes[i + 1][0] - onset
        else:
            ioi = dur
        dur_bucket = int(np.searchsorted(DURATION_EDGES[1:], ioi, side="left"))
        dur_bucket = min(dur_bucket, NUM_DUR_BUCKETS - 1)
        token = pitch * NUM_DUR_BUCKETS + dur_bucket + TOKEN_OFFSET
        tokens.append(token)
    return tokens


def get_seed_for_emotion(emotion_id, seq_len=64, midi_dir="EMOPIA_1.0/midis"):
    """Find a real clip with the requested emotion quadrant to use as a seed."""
    q_str = f"Q{emotion_id + 1}_"
    midi_path = Path(midi_dir)
    if midi_path.exists():
        matches = list(midi_path.glob(f"*{q_str}*.mid"))
        if matches:
            notes = parse_clip(matches[0])
            toks = notes_to_tokens(notes)
            if len(toks) >= 16:
                return toks[:seq_len]

    default_pitches = [60, 64, 67, 72, 60, 64, 67, 72]
    return [p * NUM_DUR_BUCKETS + 4 + TOKEN_OFFSET for p in default_pitches]


def generate_music(model, emotion_id, seed_tokens, gen_length=250,
                   temperature=0.85, seq_len=64, device="cpu"):
    """Autoregressively generate tokens conditioned on emotion."""
    model.eval()
    generated = list(seed_tokens)

    with torch.no_grad():
        for _ in range(gen_length):
            context = generated[-seq_len:]
            if len(context) < seq_len:
                context = [0] * (seq_len - len(context)) + context

            ctx_t = torch.tensor([context], dtype=torch.long, device=device)
            emo_t = torch.tensor([emotion_id], dtype=torch.long, device=device)

            logits = model(ctx_t, emo_t).squeeze(0) / max(temperature, 1e-4)
            probs = torch.softmax(logits, dim=-1)

            next_tok = torch.multinomial(probs, num_samples=1).item()
            if next_tok == 0:
                next_tok = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_tok)

    return generated


def tokens_to_pretty_midi(tokens, bpm=120, velocity=85):
    """Convert token sequence to a PrettyMIDI object."""
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


def synthesize_piano_audio(pm, fs=44100):
    """
    Synthesize MIDI notes into a realistic acoustic piano waveform
    using harmonic additive synthesis with percussive attack and natural decay.
    """
    total_time = pm.get_end_time() + 1.0
    total_samples = int(np.ceil(total_time * fs))
    if total_samples <= 0:
        return np.zeros(fs, dtype=np.int16)

    audio = np.zeros(total_samples, dtype=np.float32)

    # Piano overtone harmonics (f0, 2f0, 3f0, 4f0, 5f0, 6f0)
    harmonics = [(1, 1.0), (2, 0.55), (3, 0.28), (4, 0.14), (5, 0.07), (6, 0.03)]

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            f0 = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
            start_idx = int(note.start * fs)
            dur = max(note.end - note.start, 0.05)
            # Add sustain release tail
            dur_total = dur + min(0.35, dur * 0.5)
            n_samples = int(dur_total * fs)

            if start_idx >= total_samples or n_samples <= 0:
                continue

            t = np.arange(n_samples, dtype=np.float32) / fs

            # Percussive piano envelope: sharp attack (4ms) + pitch-dependent decay
            attack_samples = int(0.004 * fs)
            decay_rate = 2.2 + 45.0 / (note.pitch + 1)
            env = np.exp(-t * decay_rate)
            if attack_samples > 0 and len(env) > attack_samples:
                env[:attack_samples] *= np.linspace(0.0, 1.0, attack_samples, dtype=np.float32)

            # Smooth release fadeout
            rel_len = int(0.04 * fs)
            if len(env) > rel_len:
                env[-rel_len:] *= np.linspace(1.0, 0.0, rel_len, dtype=np.float32)

            note_signal = np.zeros(n_samples, dtype=np.float32)
            for h_num, h_amp in harmonics:
                freq = f0 * h_num
                if freq < fs / 2:
                    note_signal += h_amp * np.sin(2 * np.pi * freq * t)

            # Velocity dynamics
            vel_scale = (note.velocity / 127.0) ** 1.25
            note_signal *= env * vel_scale

            end_idx = min(start_idx + n_samples, total_samples)
            audio[start_idx:end_idx] += note_signal[:end_idx - start_idx]

    # Peak normalization with 0.5 dB headroom
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = (audio / peak) * 0.95

    return (audio * 32767).astype(np.int16)


def save_audio_files(audio_int16, base_path, fs=44100, save_wav=True, save_mp3=True):
    """Save synthesized PCM audio to WAV and MP3 files."""
    base = Path(base_path)
    wav_path = base.with_suffix(".wav")
    mp3_path = base.with_suffix(".mp3")
    saved = []

    if save_wav:
        scipy.io.wavfile.write(str(wav_path), fs, audio_int16)
        saved.append(str(wav_path))

    if save_mp3:
        if HAS_LAMEENC:
            encoder = lameenc.Encoder()
            encoder.set_bit_rate(192)
            encoder.set_in_sample_rate(fs)
            encoder.set_channels(1)
            encoder.set_quality(2)
            mp3_bytes = encoder.encode(audio_int16.tobytes()) + encoder.flush()
            with open(mp3_path, "wb") as f:
                f.write(mp3_bytes)
            saved.append(str(mp3_path))
        else:
            print("   [!] lameenc not installed; skipped MP3 export. Run: pip install lameenc")

    return saved


def convert_midi_file(midi_file_path, save_wav=True, save_mp3=True):
    """Convert an existing .mid file to .wav and .mp3."""
    midi_path = Path(midi_file_path)
    if not midi_path.exists():
        print(f"[ERROR] MIDI file not found: {midi_path}")
        return None

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    audio_int16 = synthesize_piano_audio(pm)
    saved = save_audio_files(audio_int16, midi_path, save_wav=save_wav, save_mp3=save_mp3)
    duration = pm.get_end_time()
    return duration, saved


def main():
    parser = argparse.ArgumentParser(description="Generate music with trained EMOPIA LSTM model and export to MIDI, WAV, and MP3")
    parser.add_argument("--checkpoint", default="best_emotion_lstm.pt", help="Path to checkpoint file")
    parser.add_argument("--emotion", type=int, choices=[1, 2, 3, 4], default=None,
                        help="Emotion quadrant: 1=Happy, 2=Angry, 3=Sad, 4=Calm")
    parser.add_argument("--all", action="store_true", help="Generate pieces for all 4 emotions")
    parser.add_argument("--length", type=int, default=250, help="Number of notes to generate")
    parser.add_argument("--temperature", type=float, default=0.85, help="Sampling temperature")
    parser.add_argument("--convert", type=str, default=None, help="Convert existing .mid file or directory to MP3/WAV")
    parser.add_argument("--no-mp3", action="store_true", help="Skip MP3 generation")
    parser.add_argument("--no-wav", action="store_true", help="Skip WAV generation")
    args = parser.parse_args()

    # Handle conversion of existing files
    if args.convert:
        target = Path(args.convert)
        if target.is_dir():
            midi_files = sorted(target.glob("*.mid"))
        elif target.exists():
            midi_files = [target]
        else:
            # Maybe glob pattern
            midi_files = sorted(Path(".").glob(args.convert))

        if not midi_files:
            print(f"[!] No MIDI files found matching: {args.convert}")
            sys.exit(1)

        print(f"Converting {len(midi_files)} MIDI files to WAV & MP3...\n")
        for mf in midi_files:
            res = convert_midi_file(mf, save_wav=not args.no_wav, save_mp3=not args.no_mp3)
            if res:
                dur, files = res
                print(f"  [+] {mf.name:30s} ({dur:.1f}s) -> {', '.join(Path(f).name for f in files)}")
        print("\n[DONE] Audio conversion complete.")
        return

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint {ckpt_path} on {device}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    vocab_size = ckpt.get("vocab_size", 1157)
    seq_len = ckpt.get("seq_len", 64)

    model = EmotionMusicLSTM(vocab_size=vocab_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"[OK] Model loaded: Trained for {ckpt.get('epoch')} epochs (val_loss: {ckpt.get('val_loss'):.4f}, val_acc: {ckpt.get('val_acc'):.2f}%)")

    emotions_to_generate = []
    if args.all or (args.emotion is None and len(sys.argv) == 1):
        emotions_to_generate = [0, 1, 2, 3]
    elif args.emotion is not None:
        emotions_to_generate = [args.emotion - 1]

    print(f"\nGenerating music (length={args.length} notes, temp={args.temperature})...\n")

    for emo_id in emotions_to_generate:
        emo_name = EMOTION_NAMES[emo_id]
        short_name = EMOTION_SHORT[emo_id]
        tag = f"Q{emo_id + 1}"
        base_name = f"generated_{tag}_{short_name}"
        midi_file = f"{base_name}.mid"

        seed = get_seed_for_emotion(emo_id, seq_len=seq_len)
        tokens = generate_music(
            model,
            emotion_id=emo_id,
            seed_tokens=seed,
            gen_length=args.length,
            temperature=args.temperature,
            seq_len=seq_len,
            device=device,
        )

        pm = tokens_to_pretty_midi(tokens, bpm=120)
        pm.write(midi_file)
        duration = pm.get_end_time()

        # Audio synthesis
        audio_int16 = synthesize_piano_audio(pm)
        audio_files = save_audio_files(audio_int16, base_name, save_wav=not args.no_wav, save_mp3=not args.no_mp3)

        file_list = [Path(midi_file).name] + [Path(f).name for f in audio_files]
        print(f"  [+] {tag} {emo_name:25s} -> {duration:.1f}s ({len(tokens)} notes)")
        print(f"      Saved: {', '.join(file_list)}")

    print("\n[DONE] Generation complete! Audio (MP3/WAV) and MIDI files saved in the current directory.")


if __name__ == "__main__":
    main()
