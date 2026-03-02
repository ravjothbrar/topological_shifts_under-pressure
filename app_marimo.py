"""
Marimo application — Topological Shifts Under Pressure.

Run with:
    marimo run app_marimo.py          # production (browser UI, no edit)
    marimo edit app_marimo.py         # notebook editor

Architecture
------------
Reactive state is managed with ``mo.state()``.  Generation happens inside an
async cell that runs a manual KV-cache token loop and calls ``set_stream_state``
after every checkpoint.  Downstream visualisation cells depend on
``stream_state`` and therefore re-execute automatically as new checkpoints
arrive — giving true real-time topology tracking.

Layout (top to bottom):
  1. Title banner
  2. Model controls  (load model, layer, temperature …)
  3. Scenario picker
  4. Conversation input  (baseline + challenge text areas + run buttons)
  5. Live view  (PCA-3D trajectory + entropy — updates token by token)
  6. Analysis tabs  (persistence, UMAP, shift metrics — populated after each turn)
  7. History  (expandable turn-by-turn breakdown)
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full", title="Topological Shifts Under Pressure")


# ---------------------------------------------------------------------------
# Cell 1: Python stdlib / third-party imports
# ---------------------------------------------------------------------------

@app.cell
def __():
    import asyncio
    import numpy as np
    return asyncio, np


# ---------------------------------------------------------------------------
# Cell 2: marimo import (separate so `mo` is available to all other cells)
# ---------------------------------------------------------------------------

@app.cell
def __():
    import marimo as mo
    return (mo,)


# ---------------------------------------------------------------------------
# Cell 3: Project-module imports
# ---------------------------------------------------------------------------

@app.cell
def __():
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

    from model_handler import (
        ModelHandler,
        StreamCheckpoint,
        DEFAULT_LAYER_IDX,
        DEFAULT_MAX_NEW_TOKENS,
        LAYER_RANGE,
        detect_hallucination,
    )
    from tda_analyzer import (
        compute_persistence,
        compute_persistence_pca,
        compute_shift_metrics,
        PersistenceResult,
    )
    from visualizer import (
        plot_persistence_diagram,
        plot_persistence_comparison,
        compute_umap,
        plot_umap,
        plot_token_entropy,
        plot_pca_3d,
        plot_pca_3d_animated,
        save_pca_animation_mp4,
        interpret_shift_metrics,
    )
    from scenarios import SCENARIOS, get_scenario_names, get_scenario_by_name

    return (
        ModelHandler,
        StreamCheckpoint,
        DEFAULT_LAYER_IDX,
        DEFAULT_MAX_NEW_TOKENS,
        LAYER_RANGE,
        detect_hallucination,
        compute_persistence,
        compute_persistence_pca,
        compute_shift_metrics,
        PersistenceResult,
        plot_persistence_diagram,
        plot_persistence_comparison,
        compute_umap,
        plot_umap,
        plot_token_entropy,
        plot_pca_3d,
        plot_pca_3d_animated,
        save_pca_animation_mp4,
        interpret_shift_metrics,
        SCENARIOS,
        get_scenario_names,
        get_scenario_by_name,
    )


# ---------------------------------------------------------------------------
# Cell 4: Reactive state
# ---------------------------------------------------------------------------

@app.cell
def __(mo):
    # The ModelHandler instance lives inside this state object.
    # None → not loaded; ModelHandler → ready.
    handler_state, set_handler_state = mo.state(None)

    # Human-readable loading status string.
    model_status, set_model_status = mo.state("not_loaded")

    # Accumulated conversation turns.  Each entry is a dict with keys:
    #   role, prompt, messages, response_text,
    #   prompt_embeddings, response_embeddings, token_entropies,
    #   response_tokens, persistence
    turns, set_turns = mo.state([])

    # Streaming state: updated checkpoint-by-checkpoint during generation.
    stream_state, set_stream_state = mo.state({
        "checkpoints": [],
        "is_generating": False,
        "current_role": None,
    })

    # Pending generation request: set by button-handler cells, consumed by
    # the async generator cell.  None means idle.
    pending_gen, set_pending_gen = mo.state(None)

    return (
        handler_state, set_handler_state,
        model_status, set_model_status,
        turns, set_turns,
        stream_state, set_stream_state,
        pending_gen, set_pending_gen,
    )


# ---------------------------------------------------------------------------
# Cell 5: Model-control widgets
# ---------------------------------------------------------------------------

@app.cell
def __(mo, DEFAULT_LAYER_IDX, DEFAULT_MAX_NEW_TOKENS, LAYER_RANGE):
    model_name_input = mo.ui.text(
        value="Qwen/Qwen3.5-9B",
        label="Model (HF hub ID)",
        full_width=True,
    )
    load_btn = mo.ui.run_button(label="⚡ Load / reload model")

    layer_slider = mo.ui.slider(
        start=min(LAYER_RANGE),
        stop=max(LAYER_RANGE),
        value=DEFAULT_LAYER_IDX,
        label="Layer",
    )
    temperature_slider = mo.ui.slider(
        start=0.1, stop=2.0, step=0.1, value=0.7, label="Temperature",
    )
    max_tokens_slider = mo.ui.slider(
        start=20, stop=500, step=10, value=DEFAULT_MAX_NEW_TOKENS, label="Max new tokens",
    )
    checkpoint_slider = mo.ui.slider(
        start=1, stop=20, step=1, value=5, label="Checkpoint every N tokens",
    )
    noise_threshold_slider = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.0, label="TDA noise threshold",
    )
    entropy_threshold_slider = mo.ui.slider(
        start=1.0, stop=8.0, step=0.5, value=4.0, label="Entropy threshold (nats)",
    )

    return (
        model_name_input, load_btn,
        layer_slider, temperature_slider, max_tokens_slider,
        checkpoint_slider, noise_threshold_slider, entropy_threshold_slider,
    )


# ---------------------------------------------------------------------------
# Cell 6: Scenario picker
# ---------------------------------------------------------------------------

@app.cell
def __(mo, get_scenario_names):
    scenario_dropdown = mo.ui.dropdown(
        options=["(none)"] + get_scenario_names(),
        value="(none)",
        label="Scenario preset",
    )
    return (scenario_dropdown,)


# ---------------------------------------------------------------------------
# Cell 7: Async model loader
# ---------------------------------------------------------------------------

@app.cell
async def __(
    mo, asyncio,
    load_btn, model_name_input,
    ModelHandler,
    set_handler_state, set_model_status,
):
    mo.stop(not load_btn.value)
    set_model_status("loading")
    try:
        handler = ModelHandler(model_name=model_name_input.value)
        # Run the blocking load() in a thread so the event loop stays alive.
        await asyncio.to_thread(handler.load)
        set_handler_state(handler)
        set_model_status("loaded")
    except Exception as _exc:
        set_model_status(f"error: {_exc}")
    return ()


# ---------------------------------------------------------------------------
# Cell 8: Conversation input widgets
# ---------------------------------------------------------------------------

@app.cell
def __(mo, scenario_dropdown, get_scenario_by_name):
    _scenario = (
        get_scenario_by_name(scenario_dropdown.value)
        if scenario_dropdown.value != "(none)"
        else None
    )
    _default_base = _scenario.baseline_prompt if _scenario else ""
    _default_chal = (
        _scenario.challenge_prompts[0]
        if (_scenario and _scenario.challenge_prompts)
        else ""
    )

    baseline_input = mo.ui.text_area(
        value=_default_base,
        label="Baseline prompt",
        placeholder="Enter a question or statement…",
        rows=3,
        full_width=True,
    )
    challenge_input = mo.ui.text_area(
        value=_default_chal,
        label="Challenge prompt",
        placeholder="Enter a challenge or follow-up…",
        rows=3,
        full_width=True,
    )
    run_baseline_btn = mo.ui.run_button(label="▶ Run baseline")
    run_challenge_btn = mo.ui.run_button(label="▶ Run challenge")

    return baseline_input, challenge_input, run_baseline_btn, run_challenge_btn


# ---------------------------------------------------------------------------
# Cell 9: Baseline button handler
# Sets pending_gen so the generator cell fires.
# ---------------------------------------------------------------------------

@app.cell
def __(mo, run_baseline_btn, baseline_input, set_pending_gen):
    mo.stop(not run_baseline_btn.value)
    _prompt = baseline_input.value.strip()
    mo.stop(not _prompt)
    set_pending_gen({
        "role": "baseline",
        "prompt": _prompt,
        "messages": [{"role": "user", "content": _prompt}],
    })
    return ()


# ---------------------------------------------------------------------------
# Cell 10: Challenge button handler
# Builds multi-turn message history from the last turn before setting pending_gen.
# ---------------------------------------------------------------------------

@app.cell
def __(mo, run_challenge_btn, challenge_input, turns, set_pending_gen):
    mo.stop(not run_challenge_btn.value)
    _prompt = challenge_input.value.strip()
    mo.stop(not _prompt or not turns)

    # Reconstruct full multi-turn history: previous messages + assistant reply +
    # new user challenge.
    _prev = turns[-1]
    _history = list(_prev["messages"])
    _history.append({"role": "assistant", "content": _prev["response_text"]})
    _history.append({"role": "user", "content": _prompt})

    set_pending_gen({
        "role": "challenge",
        "prompt": _prompt,
        "messages": _history,
    })
    return ()


# ---------------------------------------------------------------------------
# Cell 11: Async streaming generator
# This cell re-executes when pending_gen changes (i.e. when a run button is
# clicked).  It streams token checkpoints into stream_state, then appends the
# completed turn to turns.
# ---------------------------------------------------------------------------

@app.cell
async def __(
    mo, np,
    pending_gen, handler_state,
    layer_slider, temperature_slider, max_tokens_slider, checkpoint_slider,
    noise_threshold_slider,
    set_stream_state, set_turns,
    compute_persistence,
):
    mo.stop(pending_gen is None)
    mo.stop(handler_state is None)

    _handler = handler_state
    _role = pending_gen["role"]
    _messages = pending_gen["messages"]
    _prompt = pending_gen["prompt"]

    # Reset stream state for this new turn.
    set_stream_state({
        "checkpoints": [],
        "is_generating": True,
        "current_role": _role,
    })

    _final_cp = None
    try:
        async for _cp in _handler.stream_tokens_async(
            messages=_messages,
            layer_idx=layer_slider.value,
            max_new_tokens=max_tokens_slider.value,
            temperature=temperature_slider.value,
            checkpoint_every=checkpoint_slider.value,
        ):
            # Capture _cp in the default-argument to avoid late-binding closure.
            set_stream_state(lambda s, c=_cp: {
                **s,
                "checkpoints": s["checkpoints"] + [c],
            })
            _final_cp = _cp
    except Exception as _exc:
        set_stream_state(lambda s: {**s, "is_generating": False, "error": str(_exc)})
        return ()

    # Compute final persistence on the complete embedding point cloud.
    if _final_cp is not None and _final_cp.response_embeddings.shape[0] > 0:
        _all_embs = np.vstack([
            _final_cp.prompt_embeddings,
            _final_cp.response_embeddings,
        ])
        _persistence = compute_persistence(
            _all_embs,
            noise_threshold=noise_threshold_slider.value,
        )

        _turn = {
            "role": _role,
            "prompt": _prompt,
            "messages": _messages,
            "response_text": _final_cp.response_text,
            "prompt_embeddings": _final_cp.prompt_embeddings,
            "response_embeddings": _final_cp.response_embeddings,
            "token_entropies": _final_cp.token_entropies,
            "response_tokens": _final_cp.tokens_so_far,
            "persistence": _persistence,
        }
        set_turns(lambda t: t + [_turn])

    set_stream_state(lambda s: {**s, "is_generating": False})
    return ()


# ---------------------------------------------------------------------------
# Cell 12: Model status banner
# ---------------------------------------------------------------------------

@app.cell
def __(mo, model_status, handler_state):
    _icons = {
        "not_loaded": "⚪",
        "loading": "🔄",
        "loaded": "🟢",
    }
    _icon = _icons.get(model_status.split(":")[0], "🔴")
    _device = (
        f" on **{handler_state.device.upper()}**"
        if handler_state is not None
        else ""
    )
    status_banner = mo.callout(
        mo.md(f"{_icon} Model: `{model_status}`{_device}"),
        kind="info" if model_status == "loaded" else "warn",
    )
    return (status_banner,)


# ---------------------------------------------------------------------------
# Cell 13: Live view (PCA-3D + entropy — reactive to stream_state)
# ---------------------------------------------------------------------------

@app.cell
def __(mo, stream_state, entropy_threshold_slider, plot_pca_3d, plot_token_entropy):
    _cps = stream_state["checkpoints"]
    _generating = stream_state["is_generating"]
    _role = stream_state.get("current_role")

    if not _cps:
        if _generating:
            _msg = "**Generating…** waiting for first checkpoint."
        else:
            _msg = (
                "*Run **baseline** first, then optionally **challenge**.*  \n"
                "The live PCA-3D trajectory and entropy chart will appear here "
                "token by token as the model generates."
            )
        live_view = mo.vstack([mo.md(_msg)])
    else:
        _cp = _cps[-1]
        _n_tokens = _cp.response_embeddings.shape[0]
        _status = (
            f"{'⏳ Generating…' if _generating else '✅ Done'} | "
            f"Role: **{_role}** | "
            f"Embedded tokens: **{_n_tokens}**"
        )

        # --- PCA-3D scatter --------------------------------------------------
        if _n_tokens >= 2:
            _pca_fig = plot_pca_3d(
                _cp.response_embeddings,
                tokens=_cp.tokens_so_far,
                entropy_values=_cp.token_entropies,
                entropy_threshold=entropy_threshold_slider.value,
                title=f"Token trajectory — layer {_cp.response_embeddings.shape} · {_n_tokens} tokens",
            )
            _pca_widget = mo.ui.plotly(_pca_fig)
        else:
            _pca_widget = mo.md("*Waiting for ≥ 2 tokens to draw trajectory…*")

        # --- Live entropy bar ------------------------------------------------
        if _cp.token_entropies:
            _ent_fig = plot_token_entropy(
                _cp.tokens_so_far[: len(_cp.token_entropies)],
                _cp.token_entropies,
                threshold=entropy_threshold_slider.value,
                title=f"Entropy · {len(_cp.token_entropies)} tokens",
            )
            _ent_widget = mo.ui.plotly(_ent_fig)
        else:
            _ent_widget = mo.md("")

        # --- Response text preview ------------------------------------------
        _resp_preview = mo.callout(
            mo.md(f"**Response so far:**  \n{_cp.response_text or '…'}"),
            kind="neutral",
        )

        live_view = mo.vstack([
            mo.md(_status),
            _resp_preview,
            _pca_widget,
            _ent_widget,
        ])

    return (live_view,)


# ---------------------------------------------------------------------------
# Cell 13b: Animated PCA-3D replay + MP4 export (after generation finishes)
# ---------------------------------------------------------------------------

@app.cell
def __(mo, stream_state, entropy_threshold_slider, plot_pca_3d_animated, save_pca_animation_mp4):
    _cps = stream_state["checkpoints"]
    _generating = stream_state["is_generating"]

    # Only show replay when generation is done and we have checkpoints.
    if not _cps or _generating:
        animation_view = mo.md("")
    else:
        # Collect per-checkpoint embeddings (response only, growing).
        _embs = [
            cp.response_embeddings
            for cp in _cps
            if cp.response_embeddings.shape[0] >= 2
        ]
        _toks = [
            cp.tokens_so_far
            for cp in _cps
            if cp.response_embeddings.shape[0] >= 2
        ]
        _ents = [
            cp.token_entropies
            for cp in _cps
            if cp.response_embeddings.shape[0] >= 2
        ]

        if len(_embs) < 2:
            animation_view = mo.md("*Need at least 2 checkpoints with ≥ 2 tokens for replay.*")
        else:
            _anim_fig = plot_pca_3d_animated(
                _embs,
                checkpoint_tokens=_toks,
                checkpoint_entropies=_ents,
                entropy_threshold=entropy_threshold_slider.value,
                title="PCA-3D Generation Replay",
            )

            # --- HTML download (interactive animated figure) ---
            import io, json as _json
            _html_bytes = _anim_fig.to_html(full_html=True, include_plotlyjs="cdn").encode()
            _html_dl = mo.download(
                data=_html_bytes,
                filename="pca3d_animation.html",
                mimetype="text/html",
                label="Download interactive animation (HTML)",
            )

            # --- MP4 download ---
            import tempfile, os as _os, pathlib as _pl
            _mp4_dl_widget = mo.md("")  # fallback if ffmpeg missing
            try:
                _tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                _tmp.close()
                save_pca_animation_mp4(
                    _embs, _tmp.name,
                    checkpoint_tokens=_toks,
                    fps=4,
                    title="Token Trajectory",
                )
                _mp4_bytes = _pl.Path(_tmp.name).read_bytes()
                _os.unlink(_tmp.name)
                _mp4_dl_widget = mo.download(
                    data=_mp4_bytes,
                    filename="pca3d_animation.mp4",
                    mimetype="video/mp4",
                    label="Download video (MP4, requires ffmpeg)",
                )
            except Exception as _e:
                _mp4_dl_widget = mo.callout(
                    mo.md(f"MP4 export unavailable: `{_e}`  \nInstall **ffmpeg** for video export."),
                    kind="warn",
                )

            animation_view = mo.vstack([
                mo.md("### 🎬 Generation Replay"),
                mo.md(
                    "The animated plot replays how the token trajectory evolved "
                    "during generation.  Use **▶ Play** or drag the slider. "
                    "Blue = early tokens, yellow/red = late tokens."
                ),
                mo.ui.plotly(_anim_fig),
                mo.hstack([_html_dl, _mp4_dl_widget], gap=1),
            ])

    return (animation_view,)


# ---------------------------------------------------------------------------
# Cell 14: Analysis view (persistence, UMAP, shift metrics — after turns)
# ---------------------------------------------------------------------------

@app.cell
def __(
    mo, np,
    turns,
    entropy_threshold_slider,
    plot_persistence_diagram, plot_persistence_comparison,
    compute_umap, plot_umap, plot_token_entropy,
    compute_shift_metrics,
    detect_hallucination,
    interpret_shift_metrics,
):
    if not turns:
        analysis_view = mo.md(
            "*Analysis tabs will appear here after the first generation.*"
        )
    else:
        _tab_contents = {}

        # --- Persistence diagrams + qualitative interpretation -------------
        _last = turns[-1]
        _pers_fig = plot_persistence_diagram(
            _last["persistence"],
            title=f"Persistence — {_last['role']} (latest turn)",
        )
        _pers_parts = [mo.ui.plotly(_pers_fig)]

        if len(turns) >= 2:
            _shift = compute_shift_metrics(
                turns[-2]["persistence"], turns[-1]["persistence"]
            )
            _cmp_fig = plot_persistence_comparison(
                turns[-2]["persistence"], turns[-1]["persistence"]
            )
            _pers_parts.append(mo.ui.plotly(_cmp_fig))
            _pers_parts.append(mo.md(
                f"**Wasserstein total:** {_shift.wasserstein_total:.3f}  \n"
                f"**H0:** {_shift.wasserstein_h0:.3f}  ·  "
                f"**H1:** {_shift.wasserstein_h1:.3f}  \n"
                f"**ΔH0:** {_shift.delta_num_h0:+d}  ·  "
                f"**ΔH1:** {_shift.delta_num_h1:+d}  \n"
                f"**Stability H0:** {_shift.stability_h0:.1%}  ·  "
                f"**Stability H1:** {_shift.stability_h1:.1%}  \n"
                f"**Severity:** `{_shift.shift_severity}`"
            ))
            # Qualitative interpretation
            _qual = interpret_shift_metrics(_shift)
            _pers_parts.append(mo.callout(mo.md(_qual), kind="info"))

        _tab_contents["Persistence"] = mo.vstack(_pers_parts)

        # --- UMAP projection -----------------------------------------------
        _emb_list, _labels = [], []
        for _t in turns:
            _emb_list.append(_t["prompt_embeddings"])
            _labels.append(f"{_t['role']}_prompt")
            _emb_list.append(_t["response_embeddings"])
            _labels.append(f"{_t['role']}_response")

        _valid = [(e, l) for e, l in zip(_emb_list, _labels) if e.shape[0] > 0]
        if _valid:
            _ve, _vl = zip(*_valid)
            _coords, _pt_labels = compute_umap(list(_ve), list(_vl))
            if _coords.shape[0] > 0:
                _umap_fig = plot_umap(_coords, _pt_labels, title="UMAP — all turns")
                _tab_contents["UMAP"] = mo.ui.plotly(_umap_fig)

        # --- Entropy + hallucination ---------------------------------------
        _ent_parts = []
        for _t in turns:
            if not _t["token_entropies"]:
                continue
            _is_hall, _conf, _flagged = detect_hallucination(
                type("_R", (), {
                    "token_entropies": _t["token_entropies"],
                    "response_tokens": _t["response_tokens"],
                })(),
                entropy_threshold=entropy_threshold_slider.value,
            )
            _hall_note = (
                f"🚨 Potential hallucination (confidence {_conf:.0%})"
                if _is_hall else "✅ No hallucination signal"
            )
            _ent_fig = plot_token_entropy(
                _t["response_tokens"][: len(_t["token_entropies"])],
                _t["token_entropies"],
                threshold=entropy_threshold_slider.value,
                title=f"Entropy — {_t['role']}",
            )
            _ent_parts.extend([mo.md(f"**{_t['role'].capitalize()}** — {_hall_note}"), mo.ui.plotly(_ent_fig)])

        if _ent_parts:
            _tab_contents["Entropy"] = mo.vstack(_ent_parts)

        # --- Shift timeline + qualitative summaries ------------------------
        if len(turns) >= 2:
            _all_shifts = []
            _metrics_rows = []
            for _i in range(1, len(turns)):
                _sm = compute_shift_metrics(
                    turns[_i - 1]["persistence"], turns[_i]["persistence"]
                )
                _all_shifts.append((_i, turns[_i]["role"], _sm))
                _metrics_rows.append(
                    f"| {_i} | {turns[_i]['role']} "
                    f"| {_sm.wasserstein_h0:.3f} "
                    f"| {_sm.wasserstein_h1:.3f} "
                    f"| {_sm.wasserstein_total:.3f} "
                    f"| {_sm.delta_num_h0:+d} "
                    f"| {_sm.delta_num_h1:+d} "
                    f"| {_sm.stability_h0:.1%} "
                    f"| {_sm.stability_h1:.1%} "
                    f"| `{_sm.shift_severity}` |"
                )
            _metrics_md = (
                "| Turn | Role | W-H0 | W-H1 | W-total | ΔH0 | ΔH1 | Stab-H0 | Stab-H1 | Severity |\n"
                "|------|------|------|------|---------|-----|-----|---------|---------|----------|\n"
                + "\n".join(_metrics_rows)
            )
            _qual_parts = []
            for _i, _role, _sm in _all_shifts:
                _qual_parts.append(mo.md(f"**Turn {_i} ({_role}):**  \n{interpret_shift_metrics(_sm)}"))
            _tab_contents["Shift timeline"] = mo.vstack([
                mo.md(_metrics_md),
                mo.md("---"),
                mo.md("### Qualitative interpretations"),
                *_qual_parts,
            ])

        # --- Metrics export (JSON download) --------------------------------
        import json as _json
        _export_data = []
        for _i2, _t in enumerate(turns):
            _t_entry: dict = {
                "turn": _i2,
                "role": _t["role"],
                "prompt": _t["prompt"],
                "response_text": _t["response_text"],
                "mean_entropy": float(sum(_t["token_entropies"]) / max(len(_t["token_entropies"]), 1)),
                "max_entropy": float(max(_t["token_entropies"], default=0.0)),
                "num_response_tokens": len(_t["response_tokens"]),
                "h0_features": int(_t["persistence"].num_h0),
                "h1_features": int(_t["persistence"].num_h1),
                "mean_persistence_h0": float(_t["persistence"].mean_persistence_h0),
                "mean_persistence_h1": float(_t["persistence"].mean_persistence_h1),
                "max_persistence_h0": float(_t["persistence"].max_persistence_h0),
                "max_persistence_h1": float(_t["persistence"].max_persistence_h1),
            }
            if _i2 >= 1:
                _sm_ex = compute_shift_metrics(
                    turns[_i2 - 1]["persistence"], _t["persistence"]
                )
                _t_entry.update({
                    "wasserstein_h0": float(_sm_ex.wasserstein_h0),
                    "wasserstein_h1": float(_sm_ex.wasserstein_h1),
                    "wasserstein_total": float(_sm_ex.wasserstein_total),
                    "delta_h0": int(_sm_ex.delta_num_h0),
                    "delta_h1": int(_sm_ex.delta_num_h1),
                    "stability_h0": float(_sm_ex.stability_h0),
                    "stability_h1": float(_sm_ex.stability_h1),
                    "shift_severity": _sm_ex.shift_severity,
                    "qualitative_interpretation": interpret_shift_metrics(_sm_ex),
                })
            _export_data.append(_t_entry)

        _json_bytes = _json.dumps(_export_data, indent=2).encode()
        _json_dl = mo.download(
            data=_json_bytes,
            filename="topological_shift_metrics.json",
            mimetype="application/json",
            label="Download all metrics (JSON)",
        )

        # CSV export
        import io as _io
        _csv_buf = _io.StringIO()
        _csv_fields = [
            "turn", "role", "mean_entropy", "max_entropy", "num_response_tokens",
            "h0_features", "h1_features", "mean_persistence_h0", "mean_persistence_h1",
            "wasserstein_h0", "wasserstein_h1", "wasserstein_total",
            "delta_h0", "delta_h1", "stability_h0", "stability_h1", "shift_severity",
        ]
        _csv_buf.write(",".join(_csv_fields) + "\n")
        for _row in _export_data:
            _csv_buf.write(",".join(str(_row.get(f, "")) for f in _csv_fields) + "\n")
        _csv_bytes = _csv_buf.getvalue().encode()
        _csv_dl = mo.download(
            data=_csv_bytes,
            filename="topological_shift_metrics.csv",
            mimetype="text/csv",
            label="Download all metrics (CSV)",
        )

        _tab_contents["Export metrics"] = mo.vstack([
            mo.md("Download the full quantitative metrics table for all turns."),
            mo.hstack([_json_dl, _csv_dl], gap=1),
            mo.md("---"),
            mo.md("**Preview (latest turn):**"),
            mo.md(
                "```json\n" +
                _json.dumps(_export_data[-1], indent=2) +
                "\n```"
            ),
        ])

        analysis_view = mo.tabs(_tab_contents)

    return (analysis_view,)


# ---------------------------------------------------------------------------
# Cell 15: History view
# ---------------------------------------------------------------------------

@app.cell
def __(mo, turns):
    if not turns:
        history_view = mo.md("*No turns yet.*")
    else:
        _accordions = {}
        for _i, _t in enumerate(turns):
            _key = f"Turn {_i + 1} — {_t['role'].capitalize()}: {_t['prompt'][:60]}…"
            _accordions[_key] = mo.vstack([
                mo.md(f"**Prompt:** {_t['prompt']}"),
                mo.md(f"**Response:** {_t['response_text']}"),
                mo.md(
                    f"Mean entropy: `{float(sum(_t['token_entropies']) / max(len(_t['token_entropies']), 1)):.3f}` nats  ·  "
                    f"Max: `{max(_t['token_entropies'], default=0.0):.3f}` nats  ·  "
                    f"H0 features: `{_t['persistence'].num_h0}`  ·  "
                    f"H1 features: `{_t['persistence'].num_h1}`"
                ),
            ])
        history_view = mo.accordion(_accordions)

    return (history_view,)


# ---------------------------------------------------------------------------
# Cell 16: Controls panel (assembled sidebar-like column)
# ---------------------------------------------------------------------------

@app.cell
def __(
    mo,
    model_name_input, load_btn,
    layer_slider, temperature_slider, max_tokens_slider,
    checkpoint_slider, noise_threshold_slider, entropy_threshold_slider,
    scenario_dropdown,
):
    controls_panel = mo.vstack([
        mo.md("## ⚙️ Model"),
        model_name_input,
        load_btn,
        mo.md("---"),
        mo.md("## 🎛 Generation"),
        layer_slider,
        temperature_slider,
        max_tokens_slider,
        checkpoint_slider,
        mo.md("---"),
        mo.md("## 🔬 Analysis"),
        noise_threshold_slider,
        entropy_threshold_slider,
        mo.md("---"),
        mo.md("## 📋 Scenarios"),
        scenario_dropdown,
    ])
    return (controls_panel,)


# ---------------------------------------------------------------------------
# Cell 17: Main layout
# ---------------------------------------------------------------------------

@app.cell
def __(
    mo,
    status_banner,
    controls_panel,
    baseline_input, challenge_input, run_baseline_btn, run_challenge_btn,
    live_view,
    animation_view,
    analysis_view,
    history_view,
    turns,
):
    _conversation_panel = mo.vstack([
        mo.md("### 💬 Conversation"),
        baseline_input,
        run_baseline_btn,
        mo.md("---"),
        challenge_input,
        mo.ui.text(
            value="" if not turns else f"Challenge will extend turn {len(turns)}",
            label="",
            disabled=True,
        ) if turns else mo.md("*Run baseline first to unlock challenge.*"),
        run_challenge_btn,
    ])

    _main_panel = mo.vstack([
        status_banner,
        _conversation_panel,
        mo.md("---"),
        mo.md("### 📡 Live view"),
        live_view,
        animation_view,
        mo.md("---"),
        mo.md("### 📊 Analysis"),
        analysis_view,
        mo.md("---"),
        mo.md("### 🗂 History"),
        history_view,
    ])

    mo.hstack(
        [controls_panel, _main_panel],
        widths=[1, 3],
        align="start",
        gap=2,
    )
