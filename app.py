"""
Topological Shifts Under Pressure
==================================
Interactive dashboard for analysing topological changes in a language
model's internal representations during interrogation and potential
hallucination events.

Run with:
    python app.py

Then open http://localhost:7860 in your browser.
"""

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field

import gradio as gr
import numpy as np
import pandas as pd

from model_handler import ModelHandler, GenerationResult, DEFAULT_MODEL
from tda_analyzer import TDAAnalyzer, PersistenceResult, ComparisonMetrics
from visualizer import (
    create_umap_3d_plot,
    create_persistence_comparison,
    create_persistence_plot,
    create_entropy_plot,
    create_metrics_html,
    _empty_3d_figure,
)
from scenarios import (
    SCENARIOS,
    get_scenario_names,
    get_scenario_by_name,
    get_next_prompt,
    get_scenario_length,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global singletons ─────────────────────────────────────────────────────
model_handler = ModelHandler()
tda = TDAAnalyzer()


# ── Session state ─────────────────────────────────────────────────────────

@dataclass
class TurnData:
    """Everything we store for a single conversation turn."""
    turn_number: int
    prompt: str
    response: str
    stage: str  # "baseline" or "challenge"
    embeddings: np.ndarray | None = None
    persistence: PersistenceResult | None = None
    comparison: ComparisonMetrics | None = None
    mean_entropy: float = 0.0
    max_entropy: float = 0.0
    perplexity: float = 1.0
    token_entropies: list = field(default_factory=list)
    response_tokens: list = field(default_factory=list)
    generation_time: float = 0.0


def empty_state() -> dict:
    return {
        "turns": [],
        "conversation_history": [],  # [{"role":..,"content":..}]
        "current_scenario": None,
        "scenario_step": 0,
    }


# ── Core logic ────────────────────────────────────────────────────────────

def run_prompt(
    prompt_text: str,
    stage: str,
    layer_idx: int,
    max_tokens: int,
    temperature: float,
    state: dict,
) -> tuple[dict, TurnData]:
    """
    Send a prompt to the model, compute TDA, and store results.
    Returns the updated state and the new TurnData.
    """
    if not model_handler.is_loaded:
        raise gr.Error("Model not loaded. Click 'Load Model' first.")

    if not prompt_text.strip():
        raise gr.Error("Please enter a prompt.")

    # Run generation
    result: GenerationResult = model_handler.generate_with_analysis(
        prompt=prompt_text,
        conversation_history=state["conversation_history"],
        layer_idx=layer_idx,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )

    # TDA on the response embeddings
    persistence = tda.compute_persistence(result.embeddings)

    # Compare with baseline (turn 0) if available
    comparison = None
    if state["turns"]:
        baseline = state["turns"][0]
        if baseline.persistence is not None:
            comparison = tda.compare(baseline.persistence, persistence)

    turn = TurnData(
        turn_number=len(state["turns"]),
        prompt=prompt_text,
        response=result.response,
        stage=stage,
        embeddings=result.embeddings,
        persistence=persistence,
        comparison=comparison,
        mean_entropy=result.mean_entropy,
        max_entropy=result.max_entropy,
        perplexity=result.perplexity,
        token_entropies=result.token_entropies,
        response_tokens=result.response_tokens,
        generation_time=result.generation_time,
    )

    # Update conversation history for multi-turn context
    state["conversation_history"].append({"role": "user", "content": prompt_text})
    state["conversation_history"].append({"role": "assistant", "content": result.response})
    state["turns"].append(turn)

    return state, turn


def build_all_visualizations(state: dict):
    """
    Rebuild every visualization from the current state.
    Returns (umap_fig, persistence_fig, metrics_html, entropy_fig, chat_html).
    """
    turns: list[TurnData] = state["turns"]

    if not turns:
        empty = _empty_3d_figure("No data yet — submit a prompt to begin.")
        return empty, empty, "<p style='color:gray'>No metrics yet.</p>", empty, ""

    # ── 3-D UMAP ──
    embeddings_list = [t.embeddings for t in turns if t.embeddings is not None]
    stage_labels = [t.stage for t in turns if t.embeddings is not None]
    token_labels = [t.response_tokens if t.response_tokens else [""] for t in turns]
    turn_indices = [t.turn_number for t in turns if t.embeddings is not None]

    umap_fig = create_umap_3d_plot(
        embeddings_list, stage_labels, token_labels, turn_indices
    )

    # ── Persistence comparison ──
    baseline_pers = turns[0].persistence if turns[0].persistence else None
    latest_pers = turns[-1].persistence if turns[-1].persistence else None
    if len(turns) == 1:
        pers_fig = create_persistence_plot(baseline_pers, "Baseline Persistence")
    else:
        pers_fig = create_persistence_comparison(baseline_pers, latest_pers)

    # ── Metrics ──
    latest = turns[-1]
    metrics_html = create_metrics_html(
        current=latest.persistence,
        baseline=turns[0].persistence if len(turns) > 1 else None,
        comparison=latest.comparison,
        mean_entropy=latest.mean_entropy,
        perplexity=latest.perplexity,
    )

    # ── Entropy ──
    entropy_fig = create_entropy_plot(
        latest.token_entropies, latest.response_tokens
    )

    # ── Chat history ──
    chat_html = _build_chat_html(turns)

    return umap_fig, pers_fig, metrics_html, entropy_fig, chat_html


def _build_chat_html(turns: list[TurnData]) -> str:
    """Render conversation history as styled HTML."""
    blocks = []
    for t in turns:
        stage_colour = {"baseline": "#3b82f6", "challenge": "#f97316"}.get(t.stage, "#6b7280")
        blocks.append(
            f'<div style="margin:8px 0;padding:10px;border-left:3px solid {stage_colour};'
            f'background:#1e293b;border-radius:4px;">'
            f'<div style="color:{stage_colour};font-weight:600;font-size:13px;">'
            f'Turn {t.turn_number} • {t.stage.upper()} '
            f'<span style="color:#64748b;font-weight:400;">({t.generation_time:.1f}s)</span></div>'
            f'<div style="color:#94a3b8;margin:4px 0;font-size:13px;"><strong>Prompt:</strong> {_esc(t.prompt)}</div>'
            f'<div style="color:#e2e8f0;font-size:13px;">{_esc(t.response)}</div>'
            f"</div>"
        )
    return "".join(blocks)


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ── Export helpers ────────────────────────────────────────────────────────

def export_session_json(state: dict) -> str | None:
    """Export the full session as a downloadable JSON file."""
    if not state["turns"]:
        return None

    data = {
        "scenario": state.get("current_scenario"),
        "turns": [],
    }
    for t in state["turns"]:
        turn_data = {
            "turn": t.turn_number,
            "stage": t.stage,
            "prompt": t.prompt,
            "response": t.response,
            "mean_entropy": t.mean_entropy,
            "max_entropy": t.max_entropy,
            "perplexity": t.perplexity,
            "generation_time": t.generation_time,
        }
        if t.persistence:
            turn_data["tda"] = {
                "num_h0": t.persistence.num_h0,
                "num_h1": t.persistence.num_h1,
                "max_persistence_h0": t.persistence.max_persistence_h0,
                "max_persistence_h1": t.persistence.max_persistence_h1,
                "mean_persistence_h0": t.persistence.mean_persistence_h0,
                "mean_persistence_h1": t.persistence.mean_persistence_h1,
            }
        if t.comparison:
            turn_data["comparison"] = {
                "wasserstein_h0": t.comparison.wasserstein_h0,
                "wasserstein_h1": t.comparison.wasserstein_h1,
                "delta_num_h0": t.comparison.delta_num_h0,
                "delta_num_h1": t.comparison.delta_num_h1,
                "stability_score": t.comparison.stability_score,
                "shift_severity": t.comparison.shift_severity,
            }
        data["turns"].append(turn_data)

    path = os.path.join(tempfile.gettempdir(), "tda_session.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def export_metrics_csv(state: dict) -> str | None:
    """Export per-turn metrics as CSV."""
    if not state["turns"]:
        return None

    rows = []
    for t in state["turns"]:
        row = {
            "turn": t.turn_number,
            "stage": t.stage,
            "prompt": t.prompt[:80],
            "response_preview": t.response[:80],
            "mean_entropy": t.mean_entropy,
            "max_entropy": t.max_entropy,
            "perplexity": t.perplexity,
            "generation_time_s": t.generation_time,
        }
        if t.persistence:
            row["num_h0"] = t.persistence.num_h0
            row["num_h1"] = t.persistence.num_h1
            row["max_pers_h0"] = t.persistence.max_persistence_h0
            row["max_pers_h1"] = t.persistence.max_persistence_h1
        if t.comparison:
            row["wasserstein_h0"] = t.comparison.wasserstein_h0
            row["wasserstein_h1"] = t.comparison.wasserstein_h1
            row["stability_score"] = t.comparison.stability_score
            row["shift_severity"] = t.comparison.shift_severity
        rows.append(row)

    path = os.path.join(tempfile.gettempdir(), "tda_metrics.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ══════════════════════════════════════════════════════════════════════════
#  Gradio UI
# ══════════════════════════════════════════════════════════════════════════

CSS = """
.main-header { text-align:center; margin-bottom:8px; }
.main-header h1 { font-size:1.6em; color:#e2e8f0; margin:0; }
.main-header p  { color:#94a3b8; font-size:0.9em; margin:4px 0 0; }
.suggested-btn  { font-size:0.85em !important; }
"""

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Topological Shifts Under Pressure",
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
        ),
        css=CSS,
    ) as app:

        # Hidden state
        session = gr.State(empty_state())

        # ── Header ──
        gr.HTML(
            '<div class="main-header">'
            "<h1>Topological Shifts Under Pressure</h1>"
            "<p>Analyse how a language model's internal topology changes "
            "under interrogation, contradiction, and hallucination pressure</p>"
            "</div>"
        )

        # ── Top row: model loading ──
        with gr.Row():
            model_name_input = gr.Textbox(
                value=DEFAULT_MODEL,
                label="Model name (Hugging Face)",
                scale=3,
            )
            load_btn = gr.Button("Load Model", variant="primary", scale=1)
            model_status = gr.Textbox(
                label="Status",
                interactive=False,
                scale=3,
                value="Model not loaded",
            )

        # ── Main layout ──
        with gr.Row():
            # ── Left column: controls ──
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### Scenario & Prompt")

                scenario_dropdown = gr.Dropdown(
                    choices=["(Free input)"] + get_scenario_names(),
                    value="(Free input)",
                    label="Select scenario",
                )
                scenario_desc = gr.Markdown("")

                suggested_prompt_box = gr.Textbox(
                    label="Suggested next prompt (click Use to load it)",
                    interactive=False,
                    lines=2,
                )
                use_suggested_btn = gr.Button(
                    "Use suggested prompt ↓",
                    size="sm",
                    elem_classes="suggested-btn",
                )

                prompt_input = gr.Textbox(
                    label="Your prompt",
                    placeholder="Type a prompt or use a suggested one…",
                    lines=3,
                )

                with gr.Row():
                    submit_baseline_btn = gr.Button(
                        "Submit as Baseline",
                        variant="primary",
                    )
                    submit_challenge_btn = gr.Button(
                        "Submit as Challenge",
                        variant="secondary",
                    )

                with gr.Accordion("Settings", open=False):
                    layer_slider = gr.Slider(
                        minimum=0,
                        maximum=35,
                        value=18,
                        step=1,
                        label="Layer index to analyse",
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=20,
                        maximum=512,
                        value=150,
                        step=10,
                        label="Max generation tokens",
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=0.7,
                        step=0.05,
                        label="Temperature",
                    )
                    umap_neighbors = gr.Slider(
                        minimum=3,
                        maximum=50,
                        value=15,
                        step=1,
                        label="UMAP n_neighbors",
                    )
                    umap_mindist = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.1,
                        step=0.05,
                        label="UMAP min_dist",
                    )

                reset_btn = gr.Button("Reset conversation", variant="stop")

                gr.Markdown("### Export Data")
                with gr.Row():
                    dl_json_btn = gr.Button("Download session JSON", size="sm")
                    dl_csv_btn = gr.Button("Download metrics CSV", size="sm")
                dl_file = gr.File(label="Download", visible=False)

            # ── Right column: visualizations ──
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("3D UMAP"):
                        umap_plot = gr.Plot(
                            value=_empty_3d_figure("Load a model and submit a prompt to begin."),
                            label="UMAP 3D",
                        )
                    with gr.Tab("Persistence Diagrams"):
                        persistence_plot = gr.Plot(label="Persistence")
                    with gr.Tab("Metrics"):
                        metrics_display = gr.HTML(
                            "<p style='color:gray;text-align:center;'>No metrics yet.</p>"
                        )
                    with gr.Tab("Entropy"):
                        entropy_plot = gr.Plot(label="Token Entropy")

                gr.Markdown("### Conversation")
                chat_display = gr.HTML("")

        # ══════════════════════════════════════════════════════════════════
        #  Event handlers
        # ══════════════════════════════════════════════════════════════════

        def on_load_model(name):
            try:
                status = model_handler.load_model(name)
                # Update layer slider to match loaded model
                layer_max = model_handler.num_layers - 1
                default_layer = model_handler.get_default_layer()
                return (
                    status,
                    gr.update(maximum=layer_max, value=default_layer),
                )
            except Exception as e:
                return f"Error: {e}", gr.update()

        load_btn.click(
            fn=on_load_model,
            inputs=[model_name_input],
            outputs=[model_status, layer_slider],
        )

        # ── Scenario selection ──
        def on_scenario_change(scenario_name, state):
            if scenario_name == "(Free input)":
                state["current_scenario"] = None
                state["scenario_step"] = 0
                return state, "", "", gr.update(visible=False)

            state["current_scenario"] = scenario_name
            state["scenario_step"] = 0
            scenario = get_scenario_by_name(scenario_name)
            desc = f"*{scenario['description']}*  \n{len(scenario['prompts'])} prompts in sequence."
            first_prompt = scenario["prompts"][0]
            return (
                state,
                desc,
                f"{first_prompt['label']}\n\n{first_prompt['text']}",
                gr.update(visible=True),
            )

        scenario_dropdown.change(
            fn=on_scenario_change,
            inputs=[scenario_dropdown, session],
            outputs=[session, scenario_desc, suggested_prompt_box, use_suggested_btn],
        )

        # ── Use suggested prompt ──
        def on_use_suggested(state):
            sc = state.get("current_scenario")
            step = state.get("scenario_step", 0)
            p = get_next_prompt(sc, step)
            if p:
                return p["text"]
            return ""

        use_suggested_btn.click(
            fn=on_use_suggested,
            inputs=[session],
            outputs=[prompt_input],
        )

        # ── Submit prompt (baseline or challenge) ──
        def on_submit(prompt_text, stage, layer_idx, max_tokens, temperature, state):
            state, turn = run_prompt(
                prompt_text, stage, int(layer_idx), int(max_tokens), temperature, state
            )

            # Advance scenario step
            if state.get("current_scenario"):
                state["scenario_step"] = state.get("scenario_step", 0) + 1

            umap_fig, pers_fig, metrics_html, ent_fig, chat_html = (
                build_all_visualizations(state)
            )

            # Prepare next suggested prompt
            sc = state.get("current_scenario")
            step = state.get("scenario_step", 0)
            next_p = get_next_prompt(sc, step) if sc else None
            suggested = (
                f"{next_p['label']}\n\n{next_p['text']}" if next_p
                else "Scenario complete – type a custom prompt or select a new scenario."
            )

            return (
                state,
                umap_fig,
                pers_fig,
                metrics_html,
                ent_fig,
                chat_html,
                suggested,
                "",  # clear prompt input
            )

        submit_outputs = [
            session,
            umap_plot,
            persistence_plot,
            metrics_display,
            entropy_plot,
            chat_display,
            suggested_prompt_box,
            prompt_input,
        ]

        submit_baseline_btn.click(
            fn=lambda p, l, m, t, s: on_submit(p, "baseline", l, m, t, s),
            inputs=[prompt_input, layer_slider, max_tokens_slider, temperature_slider, session],
            outputs=submit_outputs,
        )
        submit_challenge_btn.click(
            fn=lambda p, l, m, t, s: on_submit(p, "challenge", l, m, t, s),
            inputs=[prompt_input, layer_slider, max_tokens_slider, temperature_slider, session],
            outputs=submit_outputs,
        )

        # ── Reset ──
        def on_reset():
            state = empty_state()
            empty = _empty_3d_figure("Conversation reset. Submit a new prompt.")
            return (
                state,
                empty,
                empty,
                "<p style='color:gray;text-align:center;'>No metrics yet.</p>",
                empty,
                "",
                "",
                "",
            )

        reset_btn.click(
            fn=on_reset,
            outputs=[
                session,
                umap_plot,
                persistence_plot,
                metrics_display,
                entropy_plot,
                chat_display,
                suggested_prompt_box,
                prompt_input,
            ],
        )

        # ── Export ──
        def on_export_json(state):
            path = export_session_json(state)
            if path:
                return gr.update(value=path, visible=True)
            raise gr.Error("No data to export.")

        def on_export_csv(state):
            path = export_metrics_csv(state)
            if path:
                return gr.update(value=path, visible=True)
            raise gr.Error("No data to export.")

        dl_json_btn.click(fn=on_export_json, inputs=[session], outputs=[dl_file])
        dl_csv_btn.click(fn=on_export_csv, inputs=[session], outputs=[dl_file])

    return app


# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",  # accessible from other devices on the network
        server_port=7860,
        share=False,             # set True to get a public Gradio link
    )
