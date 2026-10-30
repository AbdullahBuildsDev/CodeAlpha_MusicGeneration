"""
CodeAlpha - AI Music Generation (model + data + arrangement)
------------------------------------------------------------
A deep-learning music generator. An LSTM neural network (PyTorch) learns a
melodic "contour", which is then arranged into a full, pleasant piece:

  * MELODY  - the LSTM output, snapped to the chosen mood's musical scale so it
              always sounds in-key.
  * CHORDS  - diatonic triads from a mood-specific chord progression (harmony).
  * BASS    - the root of each chord (gives the music body).

Audio is rendered with a small built-in synthesizer that uses soft waveforms
and smooth attack/release envelopes, so it sounds warm instead of a harsh beep.
A standard MIDI file is also written (for download / opening in any DAW).

Everything is local and free - no paid APIs, no external sound fonts.

Author: Abdullah Awan  |  CodeAlpha AI Internship
"""

import os
import glob
import json
import random

import numpy as np
import torch
import torch.nn as nn

SEQ_LEN = 24
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
MODEL_PATH = os.path.join(HERE, "music_model.pt")
VOCAB_PATH = os.path.join(HERE, "vocab.json")

# Scale shapes (semitone offsets from the root).
SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "penta_major": [0, 2, 4, 7, 9],
    "penta_minor": [0, 3, 5, 7, 10],
}

# MOODS: each one sets a scale, a root note, a tempo, and a 4-chord progression
# (as scale-degree indices). This is what makes each mood feel different.
MOODS = {
    "😊 Happy":       {"scale": "major",       "root": 60, "tempo": 126, "prog": [0, 4, 5, 3]},
    "😢 Sad":         {"scale": "minor",       "root": 57, "tempo": 70,  "prog": [0, 5, 3, 4]},
    "😌 Calm":        {"scale": "major",       "root": 60, "tempo": 80,  "prog": [0, 3, 4, 0]},
    "⚡ Energetic":   {"scale": "major",       "root": 62, "tempo": 148, "prog": [0, 4, 5, 3]},
    "🌧️ Rainy / Lo-fi":{"scale": "dorian",    "root": 57, "tempo": 68,  "prog": [0, 3, 5, 4]},
    "💭 Dreamy":      {"scale": "penta_major", "root": 65, "tempo": 92,  "prog": [0, 5, 3, 4]},
    "🎬 Epic":        {"scale": "minor",       "root": 52, "tempo": 102, "prog": [0, 5, 3, 4]},
    "❤️ Romantic":    {"scale": "major",       "root": 55, "tempo": 86,  "prog": [0, 3, 4, 5]},
    "🌙 Mysterious":  {"scale": "penta_minor", "root": 59, "tempo": 88,  "prog": [0, 3, 4, 2]},
}


# --------------------------------------------------------------------- data
def _generate_corpus(n_phrases=700):
    """Create a corpus of smooth, tonal melodies (as MIDI pitch lists)."""
    sequences = []
    for _ in range(n_phrases):
        root = random.choice([57, 60, 62, 65, 67])
        scale = random.choice([SCALES["major"], SCALES["minor"]])
        degrees = sorted(root + off + 12 * o for o in (0, 1) for off in scale)
        phrase, idx = [], random.randrange(len(degrees))
        direction = random.choice([-1, 1])
        for step in range(random.randint(16, 32)):
            phrase.append(degrees[idx])
            # Mostly stepwise motion with occasional small leaps; drift back to
            # the centre so phrases don't run off the top or bottom.
            move = random.choices([-2, -1, 1, 2, 3], weights=[2, 4, 4, 2, 1])[0]
            if idx < 3:
                move = abs(move)
            elif idx > len(degrees) - 4:
                move = -abs(move)
            idx = max(0, min(len(degrees) - 1, idx + move))
        sequences.append(phrase)
    return sequences


def _load_user_midis():
    seqs = []
    try:
        import pretty_midi
        for path in glob.glob(os.path.join(HERE, "data", "*.mid")):
            pm = pretty_midi.PrettyMIDI(path)
            for inst in pm.instruments:
                if inst.is_drum:
                    continue
                pitches = [n.pitch for n in sorted(inst.notes, key=lambda n: n.start)]
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
def train_model(epochs=18, progress=None):
    sequences = _generate_corpus() + _load_user_midis()
    vocab = sorted({p for seq in sequences for p in seq})
    p2i = {p: i for i, p in enumerate(vocab)}

    X, Y = [], []
    for seq in sequences:
        ids = [p2i[p] for p in seq]
        for i in range(len(ids) - SEQ_LEN):
            X.append(ids[i:i + SEQ_LEN])
            Y.append(ids[i + SEQ_LEN])
    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)

    model = MusicLSTM(len(vocab))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()

    n, batch = len(X), 256
    for ep in range(epochs):
        perm = torch.randperm(n)
        for b in range(0, n, batch):
            idx = perm[b:b + batch]
            opt.zero_grad()
            out, _ = model(X[idx])
            loss = loss_fn(out[:, -1, :], Y[idx])
            loss.backward()
            opt.step()
        if progress:
            progress((ep + 1) / epochs, desc=f"Training epoch {ep + 1}/{epochs}")

    torch.save(model.state_dict(), MODEL_PATH)
    with open(VOCAB_PATH, "w") as f:
        json.dump(vocab, f)
    return model, vocab


def _load_or_train(progress=None):
    if os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH):
        try:
            with open(VOCAB_PATH) as f:
                vocab = json.load(f)
            model = MusicLSTM(len(vocab))
            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            model.eval()
            return model, vocab
        except Exception:
            pass  # vocab/model mismatch -> retrain
    return train_model(progress=progress)


# ------------------------------------------------------------- melody (LSTM)
def _generate_melody(n_notes, temperature, progress=None):
    model, vocab = _load_or_train(progress)
    model.eval()
    seq = [random.randrange(len(vocab)) for _ in range(SEQ_LEN)]
    out = []
    with torch.no_grad():
        for _ in range(n_notes):
            x = torch.tensor([seq[-SEQ_LEN:]], dtype=torch.long)
            logits, _ = model(x)
            logits = logits[0, -1] / max(0.1, temperature)
            probs = torch.softmax(logits, dim=-1).numpy()
            nxt = int(np.random.choice(len(vocab), p=probs))
            out.append(vocab[nxt])
            seq.append(nxt)
    return out


# ----------------------------------------------------------------- arrange
def _scale_notes(root, scale_name, octaves=3, low=48):
    """Full list of MIDI notes in a scale across several octaves."""
    offs = SCALES[scale_name]
    notes = []
    for o in range(octaves + 1):
        for off in offs:
            n = root + off + 12 * (o - 1)
            if low <= n <= low + 36:
                notes.append(n)
    return sorted(set(notes))


def _snap(pitch, scale):
    """Snap any pitch to the nearest note in the scale."""
    return min(scale, key=lambda s: abs(s - pitch))


def _triad(scale_ext, degree):
    """Diatonic triad (3 stacked thirds) starting at a scale degree."""
    return [scale_ext[degree], scale_ext[degree + 2], scale_ext[degree + 4]]


def generate(n_notes=160, temperature=1.0, mood="😊 Happy", progress=None):
    """Generate a full arranged piece. Returns (midi_path, (sr, audio))."""
    cfg = MOODS.get(mood, MOODS["😊 Happy"])
    raw = _generate_melody(n_notes, temperature, progress)

    # Snap the LSTM melody to this mood's scale so it's always in-key.
    mel_scale = _scale_notes(cfg["root"], cfg["scale"], octaves=2, low=cfg["root"])
    melody = [_snap(p, mel_scale) for p in raw]

    # A 2-octave scale for building chords by degree.
    chord_scale = [cfg["root"] + off + 12 * o
                   for o in range(3) for off in SCALES[cfg["scale"]]]
    chord_scale.sort()

    return _arrange(melody, cfg, chord_scale)


def _arrange(melody, cfg, chord_scale):
    import pretty_midi
    tempo = cfg["tempo"]
    spb = 60.0 / tempo               # seconds per beat
    bar = 4 * spb                    # 4 beats per bar
    prog = cfg["prog"]

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    lead = pretty_midi.Instrument(program=0)
    harm = pretty_midi.Instrument(program=0)
    bass = pretty_midi.Instrument(program=32)

    notes_per_bar = 8                # eighth notes
    n_bars = max(2, len(melody) // notes_per_bar)
    render = []  # (pitch, start, dur, gain)

    i = 0
    for b in range(n_bars):
        root_deg = prog[b % len(prog)]
        t0 = b * bar
        triad = _triad(chord_scale, root_deg)

        # Chord pad (soft, held most of the bar).
        for p in triad:
            pm_note = pretty_midi.Note(velocity=42, pitch=int(p), start=t0, end=t0 + bar * 0.98)
            harm.notes.append(pm_note)
            render.append((p, t0, bar * 0.98, 0.16))

        # Bass: root on beats 1 and 3.
        broot = triad[0] - 12
        for beat in (0, 2):
            bs = t0 + beat * spb
            bass.notes.append(pretty_midi.Note(velocity=72, pitch=int(broot), start=bs, end=bs + spb * 0.9))
            render.append((broot, bs, spb * 0.9, 0.30))

        # Melody: eighth notes across the bar, with a little rhythmic variety.
        tt = t0
        for k in range(notes_per_bar):
            if i >= len(melody):
                break
            dur = spb * (1.0 if k == notes_per_bar - 1 and random.random() < 0.4 else 0.5)
            p = melody[i]; i += 1
            lead.notes.append(pretty_midi.Note(velocity=96, pitch=int(p), start=tt, end=tt + dur * 0.92))
            render.append((p, tt, dur * 0.92, 0.42))
            tt += dur

    pm.instruments += [lead, harm, bass]
    midi_path = os.path.join(HERE, "generated.mid")
    pm.write(midi_path)

    audio = _render(render, sr=22050)
    return midi_path, (22050, audio)


# ------------------------------------------------------------- audio synth
def _env(n, sr, attack=0.012, release=0.06):
    """Smooth attack/release + gentle decay -> warm, click-free note."""
    env = np.ones(n)
    a, r = int(sr * attack), int(sr * release)
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    if r > 0:
        env[-r:] *= np.linspace(1, 0, r)
    env *= np.exp(-np.linspace(0, 1.8, n)) * 0.55 + 0.45
    return env


def _render(notes, sr=22050):
    """Mix all notes into one soft waveform using a small additive synth."""
    if not notes:
        return np.zeros(sr, dtype=np.float32)
    total = max(start + dur for _, start, dur, _ in notes) + 0.3
    buf = np.zeros(int(sr * total) + sr, dtype=np.float32)

    for pitch, start, dur, gain in notes:
        n = max(1, int(sr * dur))
        t = np.arange(n) / sr
        freq = 440.0 * 2 ** ((pitch - 69) / 12.0)
        # Soft tone: fundamental + a few quieter harmonics (mellow, not a pure beep).
        wave = (np.sin(2 * np.pi * freq * t)
                + 0.28 * np.sin(2 * np.pi * 2 * freq * t)
                + 0.12 * np.sin(2 * np.pi * 3 * freq * t))
        seg = (gain * wave * _env(n, sr)).astype(np.float32)
        s = int(sr * start)
        buf[s:s + n] += seg

    # Normalise and apply a soft limiter to avoid harsh clipping.
    peak = np.max(np.abs(buf)) or 1.0
    buf = np.tanh(buf / peak * 1.1) * 0.9
    return buf.astype(np.float32)


if __name__ == "__main__":
    print("Training / generating a short piece...")
    path, (sr, wav) = generate(n_notes=120, mood="😊 Happy")
    print(f"Saved MIDI -> {path} | audio {len(wav)/sr:.1f}s")
