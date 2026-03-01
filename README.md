# Topological Shifts Under Pressure

Analyse how a language model's internal topological structure changes when
challenged, contradicted, or pushed toward hallucination.

This tool extracts hidden-state embeddings from a locally-running **Qwen3-4B**
model, computes **persistence diagrams** (TDA), projects them into a **3-D UMAP**
space, and tracks how those structures shift across a multi-turn conversation.

Everything runs **locally** in your browser — no data leaves your machine.

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone <this-repo-url>
cd topological_shifts_under-pressure

# 2. Run the start script (creates venv, installs deps, launches)
chmod +x start.sh
./start.sh

# 3. Open http://localhost:7860 in your browser
```

### Manual setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

> **First run** downloads the model from Hugging Face (~8 GB for Qwen3-4B).

---

## GPU vs CPU

| Setup | What happens |
|-------|-------------|
| **No GPU** | Model runs in `float32` on CPU. Inference takes 15-60 s per prompt depending on token count. ~16 GB RAM recommended. |
| **NVIDIA GPU** | Auto-detected. Model loads in `float16` on CUDA. Inference <2 s. |
| **Apple Silicon** | MPS backend auto-detected where available. |

To use a smaller model on a low-RAM machine, change the model name in the UI
to `Qwen/Qwen3-1.7B` or `Qwen/Qwen3-0.6B` before clicking **Load Model**.

---

## How to use the UI

### 1. Load the model
Type a Hugging Face model name (default: `Qwen/Qwen3-4B`) and click **Load Model**.

### 2. Choose a scenario or free-form input
- Select a pre-loaded scenario from the dropdown — the first suggested prompt
  appears automatically. Click **Use suggested prompt** to load it into the
  input box.
- Or just type any prompt you like.

### 3. Submit
- **Submit as Baseline** — the first prompt in a conversation.
- **Submit as Challenge** — interrogation, contradiction, or pressure prompt.

After submission:
- The **3-D UMAP** tab shows the embedding point-cloud, colour-coded by stage.
- The **Persistence Diagrams** tab shows H0 (clusters) and H1 (loops) features.
- The **Metrics** tab shows Wasserstein distances, feature counts, stability scores.
- The **Entropy** tab highlights tokens with unusually high generation entropy.

### 4. Iterate
Continue submitting challenges. Each turn adds to the UMAP and metrics
history. The suggested prompt advances through the scenario automatically.

### 5. Export
Click **Download session JSON** or **Download metrics CSV** to save all data.

---

## Interpreting the output

### 3-D UMAP
- **Blue** dots = baseline embeddings
- **Orange** dots = challenge prompt embeddings
- **White diamond trajectory** = centroid path across turns
- Fragmentation or drift of the point-cloud indicates the model's internal
  representation is reorganising under pressure.

### Persistence diagrams
- Each point `(birth, death)` is a topological feature.
- **H0 (blue)** = connected components. More H0 means the embedding space is
  more fragmented (the model is "less sure" how concepts cluster).
- **H1 (orange)** = loops / cycles. H1 features suggest circular dependencies
  in the representation.
- Points far from the diagonal are persistent (significant); points near the
  diagonal are noise.

### Metrics
| Metric | Meaning |
|--------|---------|
| **Wasserstein H0/H1** | "Distance" between baseline and current persistence diagrams. Larger = bigger topological shift. |
| **Stability score** | 0–100 %. High = topology barely changed; low = major reorganisation. |
| **Shift severity** | STABLE / MODERATE / LARGE — colour-coded. |
| **Mean entropy** | Average generation entropy. Higher values suggest the model is less confident. |
| **Perplexity** | Exponential of mean entropy — a more intuitive "surprise" metric. |

### Entropy plot
- Red bars = tokens above the hallucination threshold (mean + 2σ).
- Clusters of high-entropy tokens often indicate the model is confabulating.

---

## Architecture

```
app.py              Gradio web application
model_handler.py    Qwen3 model loading, generation, embedding extraction
tda_analyzer.py     Vietoris-Rips persistence, Wasserstein distance
visualizer.py       3-D UMAP, persistence plots, entropy charts
scenarios.py        Pre-loaded interrogation scenarios
```

### Layer selection
Qwen3-4B has ~36 transformer layers. The default analysis layer is in the
"deep semantic zone" (roughly layers 18-28) where the model has processed
syntax but hasn't yet collapsed into vocabulary predictions. You can change
the layer in the Settings panel.

---

## Pre-loaded scenarios

| Scenario | What it tests |
|----------|--------------|
| **Factual Challenge** | Assert a wrong capital city and escalate |
| **Logical Contradiction** | Fabricate a claim the model said 2+2=5 |
| **Hallucination Probe** | Ask about a fictional battle, demand details |
| **Self-Correction Pressure** | Insist WWII ended in 1947 with fake sources |
| **Ethical Dilemma** | Push the model into a contradictory ethical stance |

---

## Troubleshooting

- **`ripser` won't install**: The app falls back to a simpler scipy-based
  persistence computation. You'll get H0 features but H1 may be empty.
  Try `pip install ripser` in a fresh environment, or install Cython first.
- **Out of memory**: Switch to a smaller model (`Qwen/Qwen3-0.6B`).
- **Slow on CPU**: Expected — 4B-parameter models are compute-heavy without
  a GPU. Use `Qwen/Qwen3-1.7B` for faster iteration.
