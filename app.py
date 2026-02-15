"""
Topological Shifts Under Pressure
==================================

Interactive Streamlit dashboard for analysing how a small language model's
internal topological structure changes during interrogation, contradiction,
and potential hallucination events.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
import numpy as np

from model_handler import (
    ModelHandler,
    GenerationResult,
    ConversationTurn,
    detect_hallucination,
    compute_embedding_consistency,
    LAYER_RANGE,
    DEFAULT_LAYER_IDX,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_NAME,
)
from tda_analyzer import (
    compute_persistence,
    compute_shift_metrics,
    PersistenceResult,
    TopologicalShiftMetrics,
)
from visualizer import (
    plot_persistence_diagram,
    plot_persistence_comparison,
    compute_umap,
    plot_umap,
    plot_token_entropy,
    STAGE_COLORS,
)
from scenarios import get_scenario_names, get_scenario_by_name

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Topological Shifts Under Pressure",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, object] = {
    "handler": None,
    "model_loaded": False,
    "turns": [],            # list[ConversationTurn]
    "persistence": [],      # list[PersistenceResult]  (parallel to turns)
    "shift_metrics": [],    # list[TopologicalShiftMetrics | None]
}

for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _handler() -> ModelHandler:
    if st.session_state["handler"] is None:
        st.session_state["handler"] = ModelHandler()
    return st.session_state["handler"]


# ---------------------------------------------------------------------------
# Sidebar — input controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Controls")

    # Model settings
    st.subheader("Model")
    model_name = st.text_input("Model name", value=DEFAULT_MODEL_NAME)
    layer_idx = st.selectbox(
        "Layer to analyse",
        options=LAYER_RANGE,
        index=LAYER_RANGE.index(DEFAULT_LAYER_IDX),
    )
    max_tokens = st.slider(
        "Max generation tokens",
        min_value=20,
        max_value=300,
        value=DEFAULT_MAX_NEW_TOKENS,
        step=10,
    )
    entropy_threshold = st.slider(
        "Entropy threshold (hallucination)",
        min_value=1.0,
        max_value=8.0,
        value=4.0,
        step=0.5,
    )

    # Load model button
    if st.button("Load model", type="primary"):
        handler = _handler()
        handler.model_name = model_name
        handler.layer_idx = layer_idx
        with st.spinner("Downloading / loading model …"):
            handler.load()
        st.session_state["model_loaded"] = True
        st.success("Model loaded.")

    st.divider()

    # Scenario picker
    st.subheader("Scenarios")
    scenario_name = st.selectbox("Pre-loaded scenario", ["(none)"] + get_scenario_names())
    if scenario_name != "(none)" and st.button("Load scenario"):
        sc = get_scenario_by_name(scenario_name)
        if sc is not None:
            st.session_state["baseline_text"] = sc.baseline_prompt
            st.session_state["challenge_text"] = sc.challenge_prompts[0] if sc.challenge_prompts else ""
            st.rerun()

    st.divider()

    # Prompt inputs
    st.subheader("Prompts")
    baseline_text = st.text_area(
        "Baseline prompt",
        value=st.session_state.get("baseline_text", ""),
        key="baseline_input",
    )
    challenge_text = st.text_area(
        "Challenge / interrogation prompt",
        value=st.session_state.get("challenge_text", ""),
        key="challenge_input",
    )

    col_b, col_c, col_r = st.columns(3)
    submit_baseline = col_b.button("Baseline")
    submit_challenge = col_c.button("Challenge")
    reset = col_r.button("Reset")

    if reset:
        for key in ("turns", "persistence", "shift_metrics"):
            st.session_state[key] = []
        st.rerun()

    st.divider()

    # Compare turns
    turns: list[ConversationTurn] = st.session_state["turns"]
    if len(turns) >= 2:
        st.subheader("Compare turns")
        options = [f"Turn {i}" for i in range(len(turns))]
        turn_a = st.selectbox("Turn A", options, index=0)
        turn_b = st.selectbox("Turn B", options, index=len(options) - 1)
        compare_btn = st.button("Compare")
    else:
        compare_btn = False


# ---------------------------------------------------------------------------
# Processing helpers
# ---------------------------------------------------------------------------

def _run_turn(prompt: str, role: str) -> None:
    """Generate a response, extract embeddings, compute persistence."""
    handler = _handler()
    if not handler.is_loaded:
        st.error("Load the model first.")
        return

    # Build message list including history
    messages: list[dict] = []
    for t in st.session_state["turns"]:
        messages.append({"role": "user", "content": t.prompt})
        if t.result is not None:
            messages.append({"role": "assistant", "content": t.result.response_text})
    messages.append({"role": "user", "content": prompt})

    with st.spinner("Generating response & extracting embeddings …"):
        result = handler.generate_with_embeddings(
            messages,
            layer_idx=layer_idx,
            max_new_tokens=max_tokens,
        )

    turn = ConversationTurn(role=role, prompt=prompt, result=result, messages=messages)
    st.session_state["turns"].append(turn)

    # Persistence on full embeddings
    with st.spinner("Computing persistence diagram …"):
        pr = compute_persistence(result.full_embeddings)
    st.session_state["persistence"].append(pr)

    # Shift metrics vs. previous turn (if any)
    persis = st.session_state["persistence"]
    if len(persis) >= 2:
        sm = compute_shift_metrics(persis[-2], persis[-1])
    else:
        sm = None
    st.session_state["shift_metrics"].append(sm)


# Handle button presses
if submit_baseline and baseline_text.strip():
    _run_turn(baseline_text.strip(), "baseline")
    st.rerun()

if submit_challenge and challenge_text.strip():
    _run_turn(challenge_text.strip(), "challenge")
    st.rerun()


# ---------------------------------------------------------------------------
# Main area layout
# ---------------------------------------------------------------------------

st.title("Topological Shifts Under Pressure")
st.caption(
    "Analyse how a language model's internal topology changes under "
    "interrogation, contradiction, and hallucination."
)

turns = st.session_state["turns"]
persis = st.session_state["persistence"]
metrics_list: list[TopologicalShiftMetrics | None] = st.session_state["shift_metrics"]

if not turns:
    st.info("Load the model, enter a baseline prompt, and press **Baseline** to start.")
    st.stop()

# ---- Tabs ----------------------------------------------------------------

tab_overview, tab_persistence, tab_umap, tab_entropy, tab_history = st.tabs(
    ["Overview", "Persistence Diagrams", "UMAP Projection", "Entropy", "History"],
)

# ---- Overview tab --------------------------------------------------------

with tab_overview:
    st.subheader("Latest turn")
    latest = turns[-1]
    assert latest.result is not None

    col_resp, col_met = st.columns([2, 1])

    with col_resp:
        st.markdown(f"**Prompt ({latest.role}):** {latest.prompt}")
        st.markdown(f"**Response:** {latest.result.response_text}")

        # Hallucination detection
        is_hall, hall_conf, flagged = detect_hallucination(
            latest.result, entropy_threshold=entropy_threshold,
        )
        if is_hall:
            st.warning(
                f"Potential hallucination detected (confidence: {hall_conf:.0%}). "
                f"{len(flagged)}/{len(latest.result.token_entropies)} tokens flagged."
            )

    with col_met:
        st.metric("Mean entropy", f"{latest.result.mean_entropy:.2f}")
        st.metric("Max entropy", f"{latest.result.max_entropy:.2f}")
        st.metric("Response tokens", len(latest.result.response_tokens))

        if len(turns) >= 2 and turns[-2].result is not None:
            consistency = compute_embedding_consistency(
                turns[-2].result.full_embeddings,
                latest.result.full_embeddings,
            )
            st.metric("Embedding consistency", f"{consistency:.3f}")

    # Metrics card
    sm = metrics_list[-1] if metrics_list else None
    if sm is not None:
        st.subheader("Topological shift metrics")
        c1, c2, c3, c4, c5 = st.columns(5)

        severity_color = {"stable": "🟢", "moderate": "🟡", "large": "🔴"}

        c1.metric("Wasserstein (H0)", f"{sm.wasserstein_h0:.3f}")
        c2.metric("Wasserstein (H1)", f"{sm.wasserstein_h1:.3f}")
        c3.metric("ΔH0 features", sm.delta_num_h0)
        c4.metric("ΔH1 features", sm.delta_num_h1)
        c5.metric(
            "Severity",
            f"{severity_color.get(sm.shift_severity, '')} {sm.shift_severity}",
        )

        c6, c7, c8 = st.columns(3)
        c6.metric("Wasserstein total", f"{sm.wasserstein_total:.3f}")
        c7.metric("Stability H0", f"{sm.stability_h0:.1%}")
        c8.metric("Stability H1", f"{sm.stability_h1:.1%}")


# ---- Persistence tab -----------------------------------------------------

with tab_persistence:
    if len(persis) == 1:
        st.plotly_chart(
            plot_persistence_diagram(persis[0], title="Baseline persistence"),
            use_container_width=True,
        )
    elif len(persis) >= 2:
        # Default: compare first and last
        idx_a = 0
        idx_b = len(persis) - 1

        # If compare mode active, override
        if compare_btn:
            idx_a = int(turn_a.split()[-1])  # type: ignore[possibly-undefined]
            idx_b = int(turn_b.split()[-1])  # type: ignore[possibly-undefined]

        st.plotly_chart(
            plot_persistence_comparison(persis[idx_a], persis[idx_b]),
            use_container_width=True,
        )

        sm_comp = compute_shift_metrics(persis[idx_a], persis[idx_b])
        st.caption(
            f"Wasserstein(H0)={sm_comp.wasserstein_h0:.3f}  |  "
            f"Wasserstein(H1)={sm_comp.wasserstein_h1:.3f}  |  "
            f"Severity: {sm_comp.shift_severity}"
        )

    # Individual diagrams
    with st.expander("All persistence diagrams"):
        for i, pr in enumerate(persis):
            role = turns[i].role if i < len(turns) else "?"
            st.plotly_chart(
                plot_persistence_diagram(pr, title=f"Turn {i} ({role})"),
                use_container_width=True,
            )
            st.caption(
                f"H0 features: {pr.num_h0}  |  H1 features: {pr.num_h1}  |  "
                f"Mean pers H0: {pr.mean_persistence_h0:.3f}  |  "
                f"Mean pers H1: {pr.mean_persistence_h1:.3f}"
            )


# ---- UMAP tab ------------------------------------------------------------

with tab_umap:
    # Collect embeddings + labels from all turns
    emb_list: list[np.ndarray] = []
    label_list: list[str] = []
    all_tokens_flat: list[str] = []

    for i, t in enumerate(turns):
        if t.result is None:
            continue
        r = t.result
        stage_prefix = t.role  # "baseline" or "challenge"

        # Mark high-uncertainty response tokens
        _, _, flagged_idx = detect_hallucination(r, entropy_threshold=entropy_threshold)
        flagged_set = set(flagged_idx)

        # Prompt embeddings
        emb_list.append(r.prompt_embeddings)
        label_list.append(f"{stage_prefix}_prompt")
        all_tokens_flat.extend(r.prompt_tokens)

        # Response embeddings – split normal vs. high-uncertainty
        resp_labels = []
        for j in range(r.response_embeddings.shape[0]):
            if j in flagged_set:
                resp_labels.append("high_uncertainty")
            else:
                resp_labels.append(f"{stage_prefix}_response")

        emb_list.append(r.response_embeddings)
        label_list.append("__mixed__")  # placeholder – overridden below
        all_tokens_flat.extend(r.response_tokens)

    # Build flat label array properly (we need per-point labels)
    flat_embs: list[np.ndarray] = []
    flat_labels: list[str] = []
    flat_tokens: list[str] = []
    token_cursor = 0

    for i, t in enumerate(turns):
        if t.result is None:
            continue
        r = t.result
        stage_prefix = t.role
        _, _, flagged_idx = detect_hallucination(r, entropy_threshold=entropy_threshold)
        flagged_set = set(flagged_idx)

        # Prompt
        flat_embs.append(r.prompt_embeddings)
        flat_labels.extend([f"{stage_prefix}_prompt"] * r.prompt_embeddings.shape[0])
        flat_tokens.extend(r.prompt_tokens)

        # Response
        flat_embs.append(r.response_embeddings)
        for j in range(r.response_embeddings.shape[0]):
            if j in flagged_set:
                flat_labels.append("high_uncertainty")
            else:
                flat_labels.append(f"{stage_prefix}_response")
        flat_tokens.extend(r.response_tokens)

    if flat_embs:
        combined = np.vstack(flat_embs)

        if combined.shape[0] >= 5:
            coords, _ = compute_umap(
                [combined], ["_"], n_neighbors=min(15, combined.shape[0] - 1),
            )
            fig = plot_umap(coords, flat_labels, tokens=flat_tokens)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 5 embedding points for UMAP projection.")
    else:
        st.info("No embeddings to project yet.")


# ---- Entropy tab ---------------------------------------------------------

with tab_entropy:
    for i, t in enumerate(turns):
        if t.result is None:
            continue
        r = t.result
        if not r.token_entropies:
            continue
        st.subheader(f"Turn {i} ({t.role})")
        fig = plot_token_entropy(
            r.response_tokens[: len(r.token_entropies)],
            r.token_entropies,
            threshold=entropy_threshold,
            title=f"Turn {i} – per-token entropy",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Highlight high-entropy tokens
        is_hall, conf, flagged = detect_hallucination(r, entropy_threshold)
        if flagged:
            flagged_tokens = [
                r.response_tokens[j]
                for j in flagged
                if j < len(r.response_tokens)
            ]
            st.caption(
                f"High-entropy tokens ({len(flagged)}): "
                + ", ".join(f"``{tok}``" for tok in flagged_tokens[:20])
            )


# ---- History tab ---------------------------------------------------------

with tab_history:
    for i, t in enumerate(turns):
        with st.expander(f"Turn {i} — {t.role}", expanded=(i == len(turns) - 1)):
            st.markdown(f"**Prompt:** {t.prompt}")
            if t.result:
                st.markdown(f"**Response:** {t.result.response_text}")
                st.caption(
                    f"Mean entropy: {t.result.mean_entropy:.2f} | "
                    f"Max entropy: {t.result.max_entropy:.2f} | "
                    f"Tokens: {len(t.result.response_tokens)}"
                )
            if i < len(persis):
                pr = persis[i]
                st.caption(
                    f"H0: {pr.num_h0} features | H1: {pr.num_h1} features | "
                    f"Max pers H0: {pr.max_persistence_h0:.3f}"
                )
            if i < len(metrics_list) and metrics_list[i] is not None:
                sm = metrics_list[i]
                st.caption(
                    f"Shift vs prev → Wasserstein total: {sm.wasserstein_total:.3f} "
                    f"| Severity: {sm.shift_severity}"
                )
