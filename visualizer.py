"""
Visualization module.

Produces interactive Plotly figures for:
- Persistence diagrams (H0 and H1 features)
- UMAP embedding projections (2-D and optional 3-D)
- Metric summary cards
- Per-token entropy bar charts
- PCA-3D live token trajectory (for real-time streaming view)

All public functions return ``plotly.graph_objects.Figure`` instances
suitable for ``mo.ui.plotly`` in marimo or ``st.plotly_chart`` in Streamlit.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from umap import UMAP

from tda_analyzer import PersistenceResult, TopologicalShiftMetrics, compute_persistence_pca


# ---------------------------------------------------------------------------
# Colour palette for conversation stages
# ---------------------------------------------------------------------------

STAGE_COLORS: dict[str, str] = {
    "baseline_prompt": "#1f77b4",       # blue
    "baseline_response": "#4a90d9",     # lighter blue
    "challenge_prompt": "#ff7f0e",      # orange
    "challenge_response": "#d62728",    # red
    "high_uncertainty": "#9467bd",      # purple
}

STAGE_LABELS: dict[str, str] = {
    "baseline_prompt": "Baseline prompt",
    "baseline_response": "Baseline response",
    "challenge_prompt": "Challenge prompt",
    "challenge_response": "Challenge response",
    "high_uncertainty": "High uncertainty",
}


# ---------------------------------------------------------------------------
# Persistence diagram
# ---------------------------------------------------------------------------

def plot_persistence_diagram(
    result: PersistenceResult,
    title: str = "Persistence Diagram",
) -> go.Figure:
    """Interactive scatter plot of a persistence diagram.

    Points above the diagonal have ``death > birth`` (i.e. nonzero
    persistence).  H0 features are coloured blue, H1 features red.
    """
    fig = go.Figure()

    # Reference diagonal
    all_vals: list[float] = []
    for d, dgm in enumerate(result.diagrams[:2]):
        finite = dgm[np.isfinite(dgm[:, 1])] if dgm.size else dgm
        if finite.size:
            all_vals.extend(finite.ravel().tolist())

    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        margin = (hi - lo) * 0.1 + 0.01
        lo -= margin
        hi += margin
    else:
        lo, hi = 0, 1

    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines",
        line=dict(color="grey", dash="dash"),
        showlegend=False,
        hoverinfo="skip",
    ))

    dim_colors = {0: "#1f77b4", 1: "#d62728"}
    dim_names = {0: "H0 (components)", 1: "H1 (loops)"}

    for d in range(min(2, len(result.diagrams))):
        dgm = result.diagrams[d]
        if dgm.size == 0:
            continue
        finite_mask = np.isfinite(dgm[:, 1])
        dgm_f = dgm[finite_mask]
        if dgm_f.size == 0:
            continue
        births = dgm_f[:, 0]
        deaths = dgm_f[:, 1]
        persistence = deaths - births

        fig.add_trace(go.Scatter(
            x=births,
            y=deaths,
            mode="markers",
            marker=dict(
                size=8,
                color=dim_colors.get(d, "#333"),
                opacity=0.7,
                line=dict(width=1, color="white"),
            ),
            name=dim_names.get(d, f"H{d}"),
            hovertemplate=(
                "Birth: %{x:.3f}<br>"
                "Death: %{y:.3f}<br>"
                "Persistence: %{customdata:.3f}"
                "<extra></extra>"
            ),
            customdata=persistence,
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Birth",
        yaxis_title="Death",
        width=450,
        height=400,
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
    )
    return fig


def plot_persistence_comparison(
    baseline: PersistenceResult,
    challenged: PersistenceResult,
) -> go.Figure:
    """Side-by-side persistence diagrams for baseline vs. challenged state."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Baseline", "Post-Challenge"),
        horizontal_spacing=0.12,
    )

    for col, result in enumerate([baseline, challenged], start=1):
        _add_persistence_traces(fig, result, row=1, col=col)

    fig.update_layout(
        height=420,
        template="plotly_white",
        showlegend=True,
        legend=dict(x=0.40, y=-0.15, orientation="h"),
    )
    return fig


# ---------------------------------------------------------------------------
# UMAP embedding space
# ---------------------------------------------------------------------------

def compute_umap(
    embeddings_list: list[np.ndarray],
    labels: list[str],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    random_state: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Fit UMAP on concatenated embeddings and return 2-D/3-D coordinates.

    Parameters
    ----------
    embeddings_list:
        One array per conversation stage, each ``(n_tokens, hidden_dim)``.
    labels:
        Stage label for each array (must be same length as
        *embeddings_list*).

    Returns
    -------
    (coords, per_point_labels)
        ``coords`` has shape ``(total_points, n_components)``.
        ``per_point_labels`` assigns a stage label to every point.
    """
    # Build combined matrix + per-point labels
    arrays = []
    per_point: list[str] = []
    for emb, lbl in zip(embeddings_list, labels):
        if emb.size == 0:
            continue
        arr = np.atleast_2d(emb)
        arrays.append(arr)
        per_point.extend([lbl] * arr.shape[0])

    if not arrays:
        return np.empty((0, n_components)), []

    combined = np.vstack(arrays)

    # UMAP needs at least n_neighbors points.
    effective_neighbors = min(n_neighbors, max(2, combined.shape[0] - 1))

    reducer = UMAP(
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
    )
    coords = reducer.fit_transform(combined)
    return coords, per_point


def plot_umap(
    coords: np.ndarray,
    labels: list[str],
    tokens: list[str] | None = None,
    title: str = "UMAP Embedding Space",
    show_trajectory: bool = True,
) -> go.Figure:
    """Interactive 2-D UMAP scatter plot with trajectory lines.

    Points are coloured by conversation stage.  If *tokens* is provided
    they appear in the hover tooltip.
    """
    fig = go.Figure()

    if coords.size == 0:
        fig.update_layout(title=title, template="plotly_white")
        return fig

    unique_labels = list(dict.fromkeys(labels))  # preserve order

    # Per-stage scatter
    start_idx = 0
    stage_centroids: list[tuple[float, float]] = []

    for stage in unique_labels:
        mask = np.array([l == stage for l in labels])
        pts = coords[mask]
        color = STAGE_COLORS.get(stage, "#999")
        display = STAGE_LABELS.get(stage, stage)

        hover: list[str] | None = None
        if tokens is not None:
            hover = [tokens[i] for i, m in enumerate(mask) if m]

        fig.add_trace(go.Scatter(
            x=pts[:, 0],
            y=pts[:, 1],
            mode="markers",
            marker=dict(size=6, color=color, opacity=0.75,
                        line=dict(width=0.5, color="white")),
            name=display,
            text=hover,
            hovertemplate=(
                "Token: %{text}<br>x: %{x:.2f}<br>y: %{y:.2f}"
                "<extra></extra>"
            ) if hover else None,
        ))

        if pts.shape[0] > 0:
            stage_centroids.append(
                (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
            )

    # Trajectory line through stage centroids
    if show_trajectory and len(stage_centroids) > 1:
        cx = [c[0] for c in stage_centroids]
        cy = [c[1] for c in stage_centroids]
        fig.add_trace(go.Scatter(
            x=cx, y=cy,
            mode="lines+markers",
            marker=dict(size=12, symbol="diamond", color="black"),
            line=dict(color="black", width=2, dash="dot"),
            name="Trajectory",
            hoverinfo="skip",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        template="plotly_white",
        height=500,
        legend=dict(x=0.01, y=0.99),
    )
    return fig


# ---------------------------------------------------------------------------
# Token entropy chart
# ---------------------------------------------------------------------------

def plot_token_entropy(
    tokens: list[str],
    entropies: list[float],
    threshold: float = 4.0,
    title: str = "Per-Token Entropy",
) -> go.Figure:
    """Bar chart of per-token entropy.  High-entropy tokens are highlighted."""
    colors = [
        "#d62728" if e > threshold else "#1f77b4"
        for e in entropies
    ]

    fig = go.Figure(go.Bar(
        x=list(range(len(tokens))),
        y=entropies,
        marker_color=colors,
        text=tokens,
        hovertemplate="Token: %{text}<br>Entropy: %{y:.3f}<extra></extra>",
    ))

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"threshold = {threshold:.1f}",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Token index",
        yaxis_title="Entropy (nats)",
        template="plotly_white",
        height=300,
    )
    return fig


# ---------------------------------------------------------------------------
# PCA-3D live token trajectory (streaming view)
# ---------------------------------------------------------------------------

def plot_pca_3d(
    embeddings: np.ndarray,
    tokens: list[str] | None = None,
    entropy_values: list[float] | None = None,
    entropy_threshold: float = 4.0,
    title: str = "Token Trajectory (PCA-3D)",
) -> go.Figure:
    """Interactive 3-D scatter of token embeddings projected via PCA.

    Points are coloured by token index (temporal ordering) so you can see
    the trajectory the model traces through its internal space while
    generating.  High-entropy tokens are outlined in red.

    PCA is performed internally so the function is self-contained and fast
    (< 5 ms for 1024-D → 3-D on 100 points on CPU).

    Parameters
    ----------
    embeddings:
        Array of shape ``(n_tokens, hidden_dim)``.
    tokens:
        Optional list of BPE token strings for hover text.
    entropy_values:
        Optional per-token Shannon entropy values.  Tokens above
        *entropy_threshold* receive a red outline.
    entropy_threshold:
        Entropy value above which a token is considered high-uncertainty.
    """
    fig = go.Figure()

    if embeddings.shape[0] < 2:
        fig.update_layout(title=title, template="plotly_white")
        return fig

    _, pca_coords = compute_persistence_pca(embeddings, n_pca_components=3)
    n = pca_coords.shape[0]

    # Temporal colour gradient: blue (early) → red (late)
    color_vals = list(range(n))

    marker_line_colors = ["rgba(0,0,0,0)"] * n
    marker_line_widths = [0] * n
    if entropy_values is not None:
        for i, e in enumerate(entropy_values[:n]):
            if e > entropy_threshold:
                marker_line_colors[i] = "red"
                marker_line_widths[i] = 2

    hover_text = [tokens[i] if tokens and i < len(tokens) else str(i) for i in range(n)]

    # 3-D when we have the z dimension, else fall back to 2-D.
    if pca_coords.shape[1] >= 3:
        fig.add_trace(go.Scatter3d(
            x=pca_coords[:, 0],
            y=pca_coords[:, 1],
            z=pca_coords[:, 2],
            mode="markers+lines",
            marker=dict(
                size=5,
                color=color_vals,
                colorscale="Plasma",
                colorbar=dict(title="Token idx", thickness=12),
                line=dict(color=marker_line_colors, width=marker_line_widths),
            ),
            line=dict(color="rgba(100,100,100,0.3)", width=1),
            text=hover_text,
            hovertemplate="Token: <b>%{text}</b><br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3",
            ),
        )
    else:
        fig.add_trace(go.Scatter(
            x=pca_coords[:, 0],
            y=pca_coords[:, 1] if pca_coords.shape[1] > 1 else np.zeros(n),
            mode="markers+lines",
            marker=dict(
                size=7,
                color=color_vals,
                colorscale="Plasma",
                colorbar=dict(title="Token idx", thickness=12),
            ),
            text=hover_text,
            hovertemplate="Token: <b>%{text}</b><br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>",
        ))
        fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=480,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_persistence_traces(
    fig: go.Figure,
    result: PersistenceResult,
    row: int,
    col: int,
) -> None:
    """Add H0/H1 scatter + diagonal to a subplot."""

    all_vals: list[float] = []
    for dgm in result.diagrams[:2]:
        finite = dgm[np.isfinite(dgm[:, 1])] if dgm.size else dgm
        if finite.size:
            all_vals.extend(finite.ravel().tolist())

    lo = min(all_vals) - 0.05 if all_vals else 0
    hi = max(all_vals) + 0.05 if all_vals else 1

    fig.add_trace(
        go.Scatter(
            x=[lo, hi], y=[lo, hi],
            mode="lines", line=dict(color="grey", dash="dash"),
            showlegend=False, hoverinfo="skip",
        ),
        row=row, col=col,
    )

    dim_colors = {0: "#1f77b4", 1: "#d62728"}
    dim_names = {0: "H0", 1: "H1"}

    for d in range(min(2, len(result.diagrams))):
        dgm = result.diagrams[d]
        if dgm.size == 0:
            continue
        mask = np.isfinite(dgm[:, 1])
        dgm_f = dgm[mask]
        if dgm_f.size == 0:
            continue

        fig.add_trace(
            go.Scatter(
                x=dgm_f[:, 0], y=dgm_f[:, 1],
                mode="markers",
                marker=dict(size=7, color=dim_colors.get(d, "#333"),
                            opacity=0.7),
                name=dim_names.get(d, f"H{d}"),
                legendgroup=dim_names.get(d),
                showlegend=(col == 1),
                hovertemplate=(
                    "Birth: %{x:.3f}<br>Death: %{y:.3f}<extra></extra>"
                ),
            ),
            row=row, col=col,
        )
