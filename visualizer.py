"""
Visualization module — 3D UMAP projections, persistence diagrams,
entropy plots, and metrics dashboards using Plotly.

All plots return Plotly Figure objects so they can be rendered
directly in Gradio with interactive pan/zoom/rotate.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from umap import UMAP

from tda_analyzer import PersistenceResult, ComparisonMetrics

# Colour palette for conversation stages
STAGE_COLOURS = {
    "baseline": "#3b82f6",       # blue
    "challenge": "#f97316",      # orange
    "response_after": "#ef4444", # red
    "hallucination": "#a855f7",  # purple
}
STAGE_LABELS = {
    "baseline": "Baseline",
    "challenge": "Challenge",
    "response_after": "Post-challenge",
    "hallucination": "High uncertainty",
}


# ── 3-D UMAP ──────────────────────────────────────────────────────────────


def compute_umap_3d(
    embeddings_list: list[np.ndarray],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """
    Project a list of embedding arrays into a shared 3-D UMAP space.

    Args:
        embeddings_list: One array per conversation turn, each (n_tokens, dim).
        n_neighbors: UMAP locality parameter.
        min_dist: UMAP minimum distance parameter.

    Returns:
        (total_points, 3) array of UMAP coordinates.
    """
    combined = np.vstack(embeddings_list)
    n_samples = combined.shape[0]

    # UMAP needs at least n_neighbors points
    effective_neighbors = min(n_neighbors, max(2, n_samples - 1))

    reducer = UMAP(
        n_components=3,
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        random_state=42,
        metric="euclidean",
    )
    coords = reducer.fit_transform(combined)
    return coords


def create_umap_3d_plot(
    embeddings_list: list[np.ndarray],
    stage_labels: list[str],
    token_labels: list[list[str]],
    turn_indices: list[int],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> go.Figure:
    """
    Build an interactive 3-D UMAP scatter plot.

    Points are colour-coded by conversation stage and connected
    with trajectory lines to show how the embedding space evolves.

    Args:
        embeddings_list: Embedding arrays, one per turn.
        stage_labels: Stage tag per turn ("baseline", "challenge", …).
        token_labels: Decoded tokens per turn (for hover text).
        turn_indices: Integer turn number per turn.
        n_neighbors: UMAP param.
        min_dist: UMAP param.
    """
    if not embeddings_list:
        return _empty_3d_figure("No data yet – submit a prompt to begin.")

    coords_3d = compute_umap_3d(embeddings_list, n_neighbors, min_dist)

    fig = go.Figure()

    offset = 0
    centroids = []
    for i, (emb, stage, tokens, turn) in enumerate(
        zip(embeddings_list, stage_labels, token_labels, turn_indices)
    ):
        n = emb.shape[0]
        pts = coords_3d[offset : offset + n]
        offset += n

        colour = STAGE_COLOURS.get(stage, "#6b7280")
        label = f"Turn {turn}: {STAGE_LABELS.get(stage, stage)}"

        # Truncate long tokens for hover
        hover = [t[:40] for t in tokens[: len(pts)]]

        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(size=3, color=colour, opacity=0.7),
                name=label,
                text=hover,
                hovertemplate="%{text}<extra>" + label + "</extra>",
            )
        )

        centroid = pts.mean(axis=0)
        centroids.append(centroid)

    # Trajectory line through centroids
    if len(centroids) > 1:
        c = np.array(centroids)
        fig.add_trace(
            go.Scatter3d(
                x=c[:, 0],
                y=c[:, 1],
                z=c[:, 2],
                mode="lines+markers",
                line=dict(color="#ffffff", width=4, dash="dot"),
                marker=dict(size=6, color="#ffffff", symbol="diamond"),
                name="Trajectory",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title="3-D UMAP Embedding Space",
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="UMAP-3",
            bgcolor="#1a1a2e",
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig


def _empty_3d_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16, color="gray"))
    fig.update_layout(
        template="plotly_dark",
        scene=dict(bgcolor="#1a1a2e"),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ── Persistence diagrams ──────────────────────────────────────────────────


def create_persistence_plot(
    result: PersistenceResult, title: str = "Persistence Diagram"
) -> go.Figure:
    """
    Draw a single persistence diagram with H0 (blue) and H1 (orange) features.
    The diagonal y=x is the "noise line" — points far from it are significant.
    """
    fig = go.Figure()

    # Diagonal reference
    all_pts = []
    for diag in [result.diagram_h0, result.diagram_h1]:
        if len(diag) > 0:
            finite = diag[np.isfinite(diag).all(axis=1)]
            if len(finite) > 0:
                all_pts.append(finite)
    if all_pts:
        combined = np.vstack(all_pts)
        lo = float(combined.min()) - 0.1
        hi = float(combined.max()) + 0.1
    else:
        lo, hi = 0, 1

    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line=dict(color="gray", dash="dash", width=1),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    def _add_dim(diag, dim_label, colour):
        if len(diag) == 0:
            return
        finite = diag[np.isfinite(diag).all(axis=1)]
        if len(finite) == 0:
            return
        persistence = finite[:, 1] - finite[:, 0]
        fig.add_trace(
            go.Scatter(
                x=finite[:, 0],
                y=finite[:, 1],
                mode="markers",
                marker=dict(size=7, color=colour, opacity=0.8),
                name=dim_label,
                text=[f"pers={p:.3f}" for p in persistence],
                hovertemplate="birth=%{x:.3f}<br>death=%{y:.3f}<br>%{text}<extra>"
                + dim_label
                + "</extra>",
            )
        )

    _add_dim(result.diagram_h0, "H0 (clusters)", "#3b82f6")
    _add_dim(result.diagram_h1, "H1 (loops)", "#f97316")

    fig.update_layout(
        title=title,
        xaxis_title="Birth",
        yaxis_title="Death",
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def create_persistence_comparison(
    baseline: PersistenceResult | None,
    challenged: PersistenceResult | None,
) -> go.Figure:
    """Side-by-side persistence diagrams for baseline vs post-challenge."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Baseline", "Post-Challenge"],
        horizontal_spacing=0.12,
    )

    def _add_to_subplot(result, col):
        if result is None:
            return
        for diag, dim_label, colour in [
            (result.diagram_h0, "H0", "#3b82f6"),
            (result.diagram_h1, "H1", "#f97316"),
        ]:
            if len(diag) == 0:
                continue
            finite = diag[np.isfinite(diag).all(axis=1)]
            if len(finite) == 0:
                continue
            fig.add_trace(
                go.Scatter(
                    x=finite[:, 0],
                    y=finite[:, 1],
                    mode="markers",
                    marker=dict(size=6, color=colour, opacity=0.8),
                    name=f"{dim_label} ({'base' if col == 1 else 'chal'})",
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )

        # diagonal
        all_finite = []
        for d in [result.diagram_h0, result.diagram_h1]:
            if len(d) > 0:
                f = d[np.isfinite(d).all(axis=1)]
                if len(f) > 0:
                    all_finite.append(f)
        if all_finite:
            c = np.vstack(all_finite)
            lo, hi = float(c.min()) - 0.1, float(c.max()) + 0.1
            fig.add_trace(
                go.Scatter(
                    x=[lo, hi], y=[lo, hi],
                    mode="lines",
                    line=dict(color="gray", dash="dash", width=1),
                    showlegend=False, hoverinfo="skip",
                ),
                row=1,
                col=col,
            )

    _add_to_subplot(baseline, 1)
    _add_to_subplot(challenged, 2)

    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(title_text="Birth", row=1, col=1)
    fig.update_xaxes(title_text="Birth", row=1, col=2)
    fig.update_yaxes(title_text="Death", row=1, col=1)
    fig.update_yaxes(title_text="Death", row=1, col=2)
    return fig


# ── Entropy plot ──────────────────────────────────────────────────────────


def create_entropy_plot(
    entropies: list[float],
    tokens: list[str],
    threshold: float | None = None,
) -> go.Figure:
    """
    Bar chart of per-token entropy during generation.
    Tokens above the threshold are highlighted in red (potential hallucination).
    """
    if not entropies:
        fig = go.Figure()
        fig.add_annotation(text="No entropy data", showarrow=False, font=dict(size=14, color="gray"))
        fig.update_layout(template="plotly_dark", height=250)
        return fig

    if threshold is None:
        threshold = float(np.mean(entropies) + 2 * np.std(entropies))

    colours = ["#ef4444" if e > threshold else "#3b82f6" for e in entropies]
    labels = [t[:20] for t in tokens[: len(entropies)]]

    fig = go.Figure(
        go.Bar(
            x=list(range(len(entropies))),
            y=entropies,
            marker_color=colours,
            text=labels,
            hovertemplate="Token: %{text}<br>Entropy: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=threshold,
        line_dash="dot",
        line_color="#f97316",
        annotation_text="Hallucination threshold",
    )
    fig.update_layout(
        title="Per-token Entropy",
        xaxis_title="Token position",
        yaxis_title="Entropy",
        template="plotly_dark",
        height=250,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


# ── Metrics dashboard (HTML) ─────────────────────────────────────────────


def create_metrics_html(
    current: PersistenceResult,
    baseline: PersistenceResult | None = None,
    comparison: ComparisonMetrics | None = None,
    mean_entropy: float = 0.0,
    perplexity: float = 1.0,
) -> str:
    """Return an HTML block summarising all key metrics with colour coding."""

    def _severity_colour(severity: str) -> str:
        return {"stable": "#22c55e", "moderate": "#eab308", "large": "#ef4444"}.get(
            severity, "#6b7280"
        )

    def _metric_row(label: str, value: str, colour: str = "#e2e8f0") -> str:
        return (
            f'<tr><td style="padding:6px 12px;color:#94a3b8;">{label}</td>'
            f'<td style="padding:6px 12px;color:{colour};font-weight:600;">{value}</td></tr>'
        )

    rows = [
        _metric_row("H0 features (clusters)", str(current.num_h0)),
        _metric_row("H1 features (loops)", str(current.num_h1)),
        _metric_row("Max persistence H0", f"{current.max_persistence_h0:.4f}"),
        _metric_row("Max persistence H1", f"{current.max_persistence_h1:.4f}"),
        _metric_row("Mean entropy", f"{mean_entropy:.3f}"),
        _metric_row("Perplexity", f"{perplexity:.2f}"),
    ]

    if comparison:
        sev_colour = _severity_colour(comparison.shift_severity)
        rows += [
            "<tr><td colspan='2' style='padding:8px 0;border-top:1px solid #334155;'>"
            "<strong style='color:#e2e8f0;'>Comparison with baseline</strong></td></tr>",
            _metric_row("Wasserstein H0", f"{comparison.wasserstein_h0:.4f}"),
            _metric_row("Wasserstein H1", f"{comparison.wasserstein_h1:.4f}"),
            _metric_row("ΔH0 features", f"{comparison.delta_num_h0:+d}"),
            _metric_row("ΔH1 features", f"{comparison.delta_num_h1:+d}"),
            _metric_row(
                "Stability score",
                f"{comparison.stability_score:.1%}",
                sev_colour,
            ),
            _metric_row(
                "Shift severity",
                comparison.shift_severity.upper(),
                sev_colour,
            ),
        ]

    table = (
        '<table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:14px;">'
        + "".join(rows)
        + "</table>"
    )
    return f'<div style="background:#0f172a;border-radius:8px;padding:12px;">{table}</div>'
