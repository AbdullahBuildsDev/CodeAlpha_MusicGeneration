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


def do_generate(n_notes, temperature, mood, progress=gr.Progress()):
    midi_path, (sr, wav) = musicgen.generate(
        n_notes=int(n_notes), temperature=float(temperature),
        mood=mood, progress=progress)
    info = (f"🎼 Generated a <b>{mood}</b> piece · melody + chords + bass · "
            f"{len(wav)/sr:.1f}s · creativity {temperature:.1f}")
    return (sr, wav), midi_path, f"<div class='info-card'>{info}</div>"


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

.gradio-container{
  --body-background-fill:#09090f; --background-fill-primary:#09090f;
  --background-fill-secondary:rgba(255,255,255,.03);
  --block-background-fill:rgba(255,255,255,.04);
  --block-border-color:rgba(255,255,255,.07); --border-color-primary:rgba(255,255,255,.07);
  --body-text-color:#ededf3; --body-text-color-subdued:#9a9aa8;
  --button-primary-background-fill:linear-gradient(92deg,#8b5cf6,#ec4899);
  --button-primary-background-fill-hover:linear-gradient(92deg,#a78bfa,#f472b6);
  --button-primary-text-color:#fff; --button-primary-border-color:transparent;
  --input-background-fill:rgba(255,255,255,.045); --block-radius:18px;
  font-family:'Inter',system-ui,sans-serif !important;
  max-width:880px !important; margin:auto !important;
}
body{background:
  radial-gradient(900px 500px at 82% -12%, rgba(236,72,153,.13), transparent 60%),
  radial-gradient(900px 520px at 8% -4%, rgba(139,92,246,.15), transparent 56%),
  #09090f !important;}
#hero{padding:34px 8px 12px; text-align:center;}
#hero h1{font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:2.4rem;
  margin:0; letter-spacing:-.5px;
  background:linear-gradient(90deg,#c4b5fd,#f9a8d4); -webkit-background-clip:text;
  -webkit-text-fill-color:transparent; background-clip:text;}
#hero p{color:#9a9aa8; margin:9px 0 0; font-size:.97rem; letter-spacing:.2px;}
.block,.form{backdrop-filter:blur(14px); box-shadow:0 10px 40px rgba(0,0,0,.32) !important;}
/* When a dropdown is open, let its block/form show the popup (no clipping) */
.block:has(.options), .form:has(.options){overflow:visible !important; backdrop-filter:none !important;}
button.primary,button[variant=primary]{font-weight:600 !important; letter-spacing:.3px;
  box-shadow:0 10px 26px rgba(236,72,153,.26) !important; transition:transform .15s ease;}
button.primary:hover{transform:translateY(-2px);}
.info-card{padding:14px 18px; border-radius:14px; background:rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.07); text-align:center; color:#d2d2e0; font-size:.95rem;}
label span{color:#b9b9c6 !important; font-weight:500 !important;}
/* Dropdown popup must be solid + visible (not glassy/transparent) */
ul.options, .options, div[role=listbox]{
  background:#15151e !important; border:1px solid rgba(255,255,255,.12) !important;
  border-radius:12px !important; box-shadow:0 14px 40px rgba(0,0,0,.55) !important;
  z-index:1000 !important; backdrop-filter:none !important; overflow:auto !important;}
ul.options li, .options .item, li[role=option], [role=option]{
  color:#ededf3 !important; background:transparent !important;}
ul.options li:hover, .options .item:hover, .options .item.selected,
li[role=option]:hover, [role=option][aria-selected=true]{
  background:rgba(139,92,246,.30) !important; color:#fff !important;}
footer{display:none !important;}
"""

theme = gr.themes.Base(primary_hue="violet", neutral_hue="slate")

# Force dark mode regardless of the visitor's system light/dark setting.
HEAD = """
<script>
(function(){
  var u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'dark') {
    u.searchParams.set('__theme','dark');
    window.location.replace(u.toString());
  }
})();
</script>
"""

with gr.Blocks(title="CodeAlpha · AI Music") as demo:
    gr.HTML("""
      <div id='hero'>
        <h1>AI Music Studio</h1>
        <p>Neural melodies, composed live — with chords &amp; bass in your chosen mood</p>
      </div>""")

    with gr.Row():
        with gr.Column(scale=2):
            mood = gr.Dropdown(list(musicgen.MOODS), value="😊 Happy",
                               label="🎭 Mood", info="Each mood has its own scale, tempo & chords")
            n_notes = gr.Slider(40, 320, value=160, step=20, label="🎶 Length (notes)")
            temperature = gr.Slider(0.4, 1.5, value=1.0, step=0.1,
                                    label="✨ Creativity (higher = more adventurous)")
            btn = gr.Button("🎼 Generate Music", variant="primary", size="lg")
            gr.Markdown("<sub>First run trains the model (~1 min), then it's instant.</sub>")
        with gr.Column(scale=3):
            audio = gr.Audio(label="Your generated melody", type="numpy", autoplay=True)
            midi_file = gr.File(label="Download MIDI (.mid)")
            info = gr.HTML()

    btn.click(do_generate, [n_notes, temperature, mood], [audio, midi_file, info])

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=theme, css=CSS, head=HEAD)
