"""
CodeAlpha - AI Music Generation (web app)
-------------------------------------------------
A polished Gradio app that generates original melodies with an LSTM neural
network. Pick a key, length, and creativity, click Generate, and listen to a
brand-new piece. Download the MIDI too.

On the very first run the model trains itself on a programmatic tonal corpus
(takes ~1 minute on CPU) and caches the weights, so later generations are fast.

Run:  python app.py

Author: Abdullah Awan  |  CodeAlpha AI Internship
"""

import gradio as gr
from src import musicgen


def do_generate(n_notes, temperature, key, progress=gr.Progress()):
    midi_path, (sr, wav) = musicgen.generate(
        n_notes=int(n_notes), temperature=float(temperature),
        key=key, progress=progress)
    info = (f"🎼 Generated {int(n_notes)} notes in <b>{key}</b> · "
            f"{len(wav)/sr:.1f}s · creativity {temperature:.1f}")
    return (sr, wav), midi_path, f"<div class='info-card'>{info}</div>"


CSS = """
.gradio-container {max-width: 820px !important; margin:auto;}
#hero {text-align:center; padding:26px 18px; border-radius:18px; margin-bottom:14px;
  background:linear-gradient(120deg,#4361ee 0%,#7209b7 55%,#f72585 100%); color:#fff;
  box-shadow:0 10px 30px rgba(67,97,238,.35);}
#hero h1 {font-size:1.9rem; margin:0; font-weight:800;}
#hero p {opacity:.92; margin:6px 0 0;}
.info-card {padding:12px 16px; border-radius:12px; background:var(--background-fill-secondary);
  font-size:.95rem; text-align:center;}
footer {display:none !important;}
"""

theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="purple",
                       radius_size="lg", font=["Inter", "system-ui", "sans-serif"])

with gr.Blocks(title="CodeAlpha · AI Music") as demo:
    gr.HTML("""
      <div id='hero'>
        <h1>🎵 AI Music Generation</h1>
        <p>An LSTM neural network composes brand-new melodies · CodeAlpha AI Internship</p>
      </div>""")

    with gr.Row():
        with gr.Column(scale=2):
            key = gr.Dropdown(["C major", "A minor", "G major", "D minor"],
                              value="C major", label="🎹 Key / mood")
            n_notes = gr.Slider(40, 320, value=160, step=20, label="🎶 Length (notes)")
            temperature = gr.Slider(0.4, 1.5, value=1.0, step=0.1,
                                    label="✨ Creativity (higher = more adventurous)")
            btn = gr.Button("🎼 Generate Music", variant="primary", size="lg")
            gr.Markdown("<sub>First run trains the model (~1 min), then it's instant.</sub>")
        with gr.Column(scale=3):
            audio = gr.Audio(label="Your generated melody", type="numpy", autoplay=True)
            midi_file = gr.File(label="Download MIDI (.mid)")
            info = gr.HTML()

    btn.click(do_generate, [n_notes, temperature, key], [audio, midi_file, info])

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=theme, css=CSS)
