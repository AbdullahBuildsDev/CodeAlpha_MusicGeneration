"""
CodeAlpha - AI Music Generation (model + data + generation)
------------------------------------------------------------
A deep-learning music generator. An LSTM neural network (PyTorch) learns note
patterns and then generates brand-new melodies, which are saved as MIDI and
synthesized to audio.

Data: to stay free and copyright-safe, the training corpus is generated
programmatically as tonal melodies (scales, steps, and arpeggios in a key).
You can also drop your own `.mid` files into a `data/` folder and they will be
included in training.

Everything is local and free. No paid APIs, no external sound fonts
(audio is synthesized with pretty_midi's built-in sine synthesizer).

Author: Abdullah Awan  |  CodeAlpha AI Internship
"""

import os
import glob
import json
import random

import numpy as np
import torch
import torch.nn as nn

SEQ_LEN = 24                      # notes of context the model sees
# Project root (one level up from this src/ package) - keeps weights, data,
# and generated files at the top level, not inside the source package.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(HERE, "music_model.pt")
VOCAB_PATH = os.path.join(HERE, "vocab.json")

# Major / minor scales (semitone offsets from the root).
_SCALES = {
    "C major": (60, [0, 2, 4, 5, 7, 9, 11]),
    "A minor": (57, [0, 2, 3, 5, 7, 8, 10]),
    "G major": (67, [0, 2, 4, 5, 7, 9, 11]),
    "D minor": (62, [0, 2, 3, 5, 7, 8, 10]),
}


# --------------------------------------------------------------------- data
def _generate_corpus(n_phrases=600):
    """Create a corpus of pleasant tonal note sequences (as MIDI pitch lists)."""
    sequences = []
    for _ in range(n_phrases):
        root, scale = random.choice(list(_SCALES.values()))
        degrees = [root + off + 12 * o for o in (0, 1) for off in scale]
        degrees.sort()
        phrase, idx = [], random.randrange(len(degrees))
        motif = random.choice(["step", "arp", "mix"])
        for _ in range(random.randint(16, 32)):
            phrase.append(degrees[idx])
            if motif == "step":
                idx += random.choice([-1, 1, 1])
            elif motif == "arp":
                idx += random.choice([-2, 2, 2, -1])
            else:
                idx += random.choice([-2, -1, 1, 1, 2])
            idx = max(0, min(len(degrees) - 1, idx))
        sequences.append(phrase)
    return sequences


def _load_user_midis():
    """Load pitch sequences from any .mid files in a local data/ folder."""
    seqs = []
    try:
        import pretty_midi
        for path in glob.glob(os.path.join(HERE, "data", "*.mid")):
            pm = pretty_midi.PrettyMIDI(path)
            for inst in pm.instruments:
                if inst.is_drum:
                    continue
                notes = sorted(inst.notes, key=lambda n: n.start)
                pitches = [n.pitch for n in notes]
                if len(pitches) > SEQ_LEN:
                    seqs.append(pitches)
    except Exception:
        pass
    return seqs


# --------------------------------------------------------------------- model
class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, embed=64, hidden=256, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed)
        self.lstm = nn.LSTM(embed, hidden, layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        return self.fc(out), hidden


# --------------------------------------------------------------------- train
def train_model(epochs=12, progress=None):
    """Train the LSTM on the corpus and save the model + vocabulary."""
    sequences = _generate_corpus() + _load_user_midis()

    vocab = sorted({p for seq in sequences for p in seq})
    p2i = {p: i for i, p in enumerate(vocab)}

    # Build (context -> next note) training pairs.
    X, Y = [], []
    for seq in sequences:
        ids = [p2i[p] for p in seq]
        for i in range(len(ids) - SEQ_LEN):
            X.append(ids[i:i + SEQ_LEN])
            Y.append(ids[i + SEQ_LEN])
    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)

    device = "cpu"
    model = MusicLSTM(len(vocab)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()

    n, batch = len(X), 256
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for b in range(0, n, batch):
            idx = perm[b:b + batch]
            xb, yb = X[idx].to(device), Y[idx].to(device)
            opt.zero_grad()
            out, _ = model(xb)
            loss = loss_fn(out[:, -1, :], yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if progress:
            progress((ep + 1) / epochs, desc=f"Training epoch {ep + 1}/{epochs}")

    torch.save(model.state_dict(), MODEL_PATH)
    with open(VOCAB_PATH, "w") as f:
        json.dump(vocab, f)
    return model, vocab


def _load_or_train(progress=None):
    if os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH) as f:
            vocab = json.load(f)
        model = MusicLSTM(len(vocab))
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        return model, vocab
    return train_model(progress=progress)


# ------------------------------------------------------------------ generate
def generate(n_notes=160, temperature=1.0, key="C major", progress=None):
    """Generate a new melody and return (midi_path, (audio_sr, waveform))."""
    model, vocab = _load_or_train(progress)
    p2i = {p: i for i, p in enumerate(vocab)}
    model.eval()

    # Seed with a short run up the chosen key's scale.
    root, scale = _SCALES.get(key, _SCALES["C major"])
    seed = [root + off for off in scale if (root + off) in p2i] or [vocab[0]]
    seq = [p2i[p] for p in (seed * SEQ_LEN)[:SEQ_LEN]]

    generated = []
    hidden = None
    with torch.no_grad():
        for _ in range(n_notes):
            x = torch.tensor([seq[-SEQ_LEN:]], dtype=torch.long)
            logits, hidden = model(x)
            logits = logits[0, -1] / max(0.1, temperature)
            probs = torch.softmax(logits, dim=-1).numpy()
            nxt = int(np.random.choice(len(vocab), p=probs))
            generated.append(vocab[nxt])
            seq.append(nxt)

    midi_path, audio = _to_midi_and_audio(generated)
    return midi_path, audio


def _to_midi_and_audio(pitches):
    """Turn a pitch list into a MIDI file + a synthesized audio waveform."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=100)
    piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    # A light, musical rhythm pattern (in beats): mostly eighths with some quarters.
    pattern = [0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0]
    t, spb = 0.0, 60.0 / 100.0           # seconds per beat at 100 BPM
    for i, pitch in enumerate(pitches):
        dur = pattern[i % len(pattern)] * spb
        piano.notes.append(pretty_midi.Note(
            velocity=random.randint(72, 100), pitch=int(pitch),
            start=t, end=t + dur * 0.95))
        t += dur
    pm.instruments.append(piano)

    midi_path = os.path.join(HERE, "generated.mid")
    pm.write(midi_path)
    audio = pm.synthesize(fs=22050)      # built-in sine synth (no sound font needed)
    return midi_path, (22050, audio)


if __name__ == "__main__":
    print("Training (first run) / generating a melody...")
    path, (sr, wav) = generate(n_notes=120)
    print(f"Saved MIDI -> {path}  | audio {len(wav)/sr:.1f}s")
