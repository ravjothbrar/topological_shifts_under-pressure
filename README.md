# Topological Shifts Under Pressure

A research prototype that visualises how a small language model's internal
topological structure changes when it is interrogated, contradicted, or
forced into potential hallucination.

## What this does

The application runs **Qwen3-0.6B** locally, extracts hidden-state
embeddings from a configurable transformer layer, and applies
**Topological Data Analysis (TDA)** to those embeddings.  An interactive
Streamlit dashboard lets you:

1. Send a **baseline** prompt and see the model's response.
2. Send a **challenge** prompt (e.g. "Are you sure?") and observe how the
   topology shifts.
3. Repeat with further challenges to build a multi-turn trajectory.
4. Compare any two turns with persistence diagrams, UMAP projections, and
   Wasserstein distance metrics.
5. Detect potential hallucinations via per-token entropy analysis.

---

## Installation

### Prerequisites

- Python 3.10+
- ~1.5 GB disk space for the Qwen3-0.6B model (downloaded on first run)
- (Optional) An NVIDIA GPU for faster inference

### Steps

```bash
# Clone the repository
git clone <repo-url>
cd topological_shifts_under-pressure

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

---

## Usage guide

### Quick start

1. Click **Load model** in the sidebar (first run downloads the model).
2. Choose a pre-loaded scenario from the dropdown, or type your own prompt.
3. Press **Baseline** to generate a response and compute the initial
   topology.
4. Enter a challenge prompt and press **Challenge**.
5. Explore the tabs: **Overview**, **Persistence Diagrams**, **UMAP
   Projection**, **Entropy**, **History**.

### Layer selection

Qwen3-0.6B has 28 transformer layers.  The sidebar lets you pick any
layer in the 8–22 range:

| Range   | What it captures                                |
|---------|-------------------------------------------------|
| 8–10    | Early semantic processing                       |
| 12–16   | Mid-level semantic representations (recommended) |
| 18–22   | High-level, abstract representations             |

Layer 14 is the default — a good balance between raw syntax and vocabulary
prediction.

---

## How to interpret the visualisations

### Persistence diagrams

A persistence diagram plots **birth** (x-axis) vs. **death** (y-axis) for
every topological feature found by the Vietoris-Rips filtration.

- Points *near* the diagonal are **noise** (short-lived features).
- Points *far* from the diagonal are **significant** structures.
- **H0** (blue) features = connected components / clusters.
- **H1** (red) features = loops / cycles.

When the model is challenged, look for:
- New features appearing (the model's representation is fragmenting).
- Existing features disappearing (structure is collapsing).
- Features moving further from the diagonal (structure is amplified).

### UMAP projections

The UMAP scatter plot projects the high-dimensional hidden states into 2-D.

| Colour | Meaning                     |
|--------|-----------------------------|
| Blue   | Baseline prompt tokens      |
| Orange | Challenge prompt tokens     |
| Red    | Response after challenge     |
| Purple | High-uncertainty tokens     |

The **trajectory line** (black diamonds) connects the centroids of each
stage.  Large jumps indicate the model's internal representation has
shifted significantly.

### Wasserstein distance

The Wasserstein distance measures how much "work" is needed to transform
one persistence diagram into another.  Severity thresholds:

| Total distance | Interpretation   |
|----------------|------------------|
| < 0.5          | Stable (green)   |
| 0.5 – 2.0     | Moderate (yellow)|
| > 2.0          | Large (red)      |

### Per-token entropy

Shannon entropy (in nats) over the model's vocabulary distribution for
each generated token.  Tokens above the threshold (default 4.0 nats) are
flagged as high-uncertainty and coloured red.  If more than 30% of tokens
are flagged the response is marked as a potential hallucination.

---

## Pre-loaded scenarios

| Scenario              | Tests                                           |
|-----------------------|-------------------------------------------------|
| Factual Challenge     | Correct answer challenged with wrong alternative|
| Logical Contradiction | False claim about prior output                  |
| Hallucination Probe   | Question about a fictitious event                |
| Self-Correction       | Gaslighting into adopting a wrong fact           |
| Ethical Dilemma       | Nuanced reasoning challenged for consistency     |

---

## Project structure

```
├── app.py              # Streamlit dashboard (entry point)
├── model_handler.py    # Model loading, embedding extraction, generation
├── tda_analyzer.py     # Persistence computation and shift metrics
├── visualizer.py       # Plotly figures (persistence, UMAP, entropy)
├── scenarios.py        # Pre-loaded test scenarios
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Model-specific notes

### Qwen3-0.6B

- **Parameters:** ~0.6 B
- **Layers:** 28 transformer layers
- **Hidden dimension:** 1024
- **Chat template:** Always use `tokenizer.apply_chat_template()` with
  `add_generation_prompt=True`.
- **Precision:** `torch_dtype="auto"` selects `bfloat16` on GPU or
  `float32` on CPU.
- **Performance:** ~1–3 s per prompt on CPU, < 1 s on GPU.
- **Download size:** ~1.5 GB (cached in `~/.cache/huggingface/`).

### Hidden-state indexing

`model(..., output_hidden_states=True)` returns a tuple of length 29:

- Index 0 → initial token embeddings (before any transformer layer).
- Index 1 → output of layer 0.
- Index *k* → output of layer *k − 1*.

So to get layer 14 we use `hidden_states[15]` (the code uses
`layer_idx + 1` internally).

---

## Extending the project

- **Export:** Save figures with `fig.write_image("out.png")` (requires
  `kaleido`).
- **Session saving:** Serialise `st.session_state["turns"]` with
  `pickle` or `json`.
- **Multi-layer comparison:** Run `compute_persistence` for several layers
  and compare diagrams side-by-side.
- **3-D UMAP:** Set `n_components=3` in `compute_umap` and use a 3-D
  Plotly scatter.

---

## License

Research prototype — use at your own discretion.
