# CodeAlpha — AI Music Generation (Task 3)

Generate **brand-new original melodies** with a deep-learning model. An **LSTM
neural network** (PyTorch) learns musical note patterns and composes fresh
music, which you can **listen to in the browser** and **download as MIDI** — all
running **locally and 100% free** (no paid APIs, no external sound fonts).

![status](https://img.shields.io/badge/CodeAlpha-AI%20Internship-7209b7)

## ✨ Features
- 🎹 Choose a **key / mood** (C major, A minor, G major, D minor)
- 🎶 Control **length** and **creativity** (sampling temperature)
- 🧠 Real **LSTM deep-learning model** trained from scratch
- 🔊 Instant **audio playback** (built-in synthesizer — no sound font needed)
- 💾 **Download** the result as a `.mid` file
- 🎨 Clean, modern, custom-themed UI

## 🛠️ Tech Stack
- **Python 3**
- **PyTorch** — LSTM neural network
- **pretty_midi** — MIDI creation + built-in audio synthesis
- **NumPy** — sampling
- **Gradio** — web UI

## 🚀 How to Run

```bash
# 1. (Recommended) virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
python app.py
```

> **First run** trains the model on a built-in tonal corpus (~1 minute on CPU)
> and caches the weights (`music_model.pt`). After that, generation is instant.

## 📖 How It Works
1. **Data** — a corpus of tonal note sequences (scales, steps, arpeggios in
   several keys) is generated programmatically, so no copyrighted data is
   needed. You can also add your own `.mid` files to a `data/` folder.
2. **Preprocess** — sequences are turned into `(context → next note)` training
   pairs over a fixed context window.
3. **Model** — an embedding + 2-layer **LSTM** + linear head predicts the next
   note; it's trained with cross-entropy loss.
4. **Generate** — starting from a seed in the chosen key, the model samples
   notes one at a time (with a temperature control for creativity).
5. **Render** — the notes become a MIDI file and are synthesized to audio.

## 📂 Project Structure
```
CodeAlpha_MusicGeneration/
├── app.py            # Gradio web app (entry point)
├── src/
│   └── musicgen.py   # Data + LSTM model + training + generation
├── data/             # (optional) drop your own .mid files here to train on them
├── requirements.txt  # Dependencies
├── LICENSE
└── README.md
```

## 🎛️ Tips
- **Lower creativity (0.4–0.8)** → safer, more repetitive, more tonal.
- **Higher creativity (1.1–1.5)** → more surprising, more adventurous.
- Add real `.mid` files to `data/` and delete `music_model.pt` to retrain on them.

---
**CodeAlpha AI Internship — Task 3** · Built by Abdullah Awan
