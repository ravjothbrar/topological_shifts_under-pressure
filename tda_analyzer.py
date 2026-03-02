"""
Topological Data Analysis (TDA) module.

This module computes persistence diagrams from point-cloud embeddings using
Vietoris-Rips filtration and provides metrics that quantify topological
shifts between two states of the model's internal representation.

Background
----------
*Persistent homology* tracks the birth and death of topological features
(connected components, loops, voids, …) as we grow a simplicial complex
around a point cloud.

- **H0 (dimension 0)** features correspond to *connected components* /
  clusters.  A long-lived H0 feature indicates a well-separated cluster.
- **H1 (dimension 1)** features correspond to *loops* / cycles in the data.
  Their presence indicates a ring-shaped or toroidal structure.

The *persistence* of a feature is ``death − birth``.  Features with high
persistence are topologically significant; low-persistence features are
typically noise.

The *Wasserstein distance* between two persistence diagrams quantifies how
different two topological signatures are.  A large distance after
interrogation suggests the model's internal geometry reorganised.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ripser import ripser
from persim import wasserstein, PersistenceImager
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PersistenceResult:
    """Holds the output of a persistence computation."""

    # Raw persistence diagrams as returned by ripser.
    # diagrams[d] is an (N, 2) array for dimension d.
    diagrams: list[np.ndarray] = field(default_factory=list)

    # Summary statistics
    num_h0: int = 0          # number of H0 features (excl. the essential one)
    num_h1: int = 0          # number of H1 features
    mean_persistence_h0: float = 0.0
    mean_persistence_h1: float = 0.0
    max_persistence_h0: float = 0.0
    max_persistence_h1: float = 0.0


@dataclass
class TopologicalShiftMetrics:
    """Metrics comparing two persistence diagrams."""

    wasserstein_h0: float = 0.0
    wasserstein_h1: float = 0.0
    wasserstein_total: float = 0.0

    delta_num_h0: int = 0      # change in number of H0 features
    delta_num_h1: int = 0      # change in number of H1 features

    stability_h0: float = 1.0  # fraction of features that persist
    stability_h1: float = 1.0

    shift_severity: str = "stable"  # "stable" | "moderate" | "large"


# ---------------------------------------------------------------------------
# Core TDA functions
# ---------------------------------------------------------------------------

def compute_persistence(
    embeddings: np.ndarray,
    max_dim: int = 1,
    max_edge_length: float | None = None,
    noise_threshold: float = 0.0,
    subsample: int | None = 300,
) -> PersistenceResult:
    """Compute Vietoris-Rips persistence on an embedding point cloud.

    Parameters
    ----------
    embeddings:
        Array of shape ``(n_points, n_features)``.
    max_dim:
        Maximum homology dimension to compute (0 and 1 by default).
    max_edge_length:
        Upper bound on edge lengths in the Rips complex.  ``None`` lets
        ripser choose automatically.
    noise_threshold:
        Minimum persistence to keep.  Features whose ``death − birth``
        is below this value are discarded as noise.
    subsample:
        If the point cloud has more than this many points, randomly
        subsample to keep computation tractable.  Set to ``None`` to
        disable.

    Returns
    -------
    PersistenceResult
    """
    pts = np.asarray(embeddings, dtype=np.float64)

    if pts.ndim == 1:
        pts = pts.reshape(1, -1)

    # Sub-sample large point clouds so Rips stays fast.
    if subsample is not None and pts.shape[0] > subsample:
        rng = np.random.RandomState(42)
        idx = rng.choice(pts.shape[0], size=subsample, replace=False)
        pts = pts[idx]

    # Need at least 2 points for Rips.
    if pts.shape[0] < 2:
        empty = np.empty((0, 2))
        return PersistenceResult(diagrams=[empty, empty])

    kwargs: dict = {"maxdim": max_dim, "do_cocycles": False}
    if max_edge_length is not None:
        kwargs["thresh"] = max_edge_length

    rips = ripser(pts, **kwargs)
    diagrams: list[np.ndarray] = rips["dgms"]

    # Apply noise filter.
    if noise_threshold > 0:
        diagrams = _filter_noise(diagrams, noise_threshold)

    # Ensure we always have at least 2 entries (H0 and H1).
    while len(diagrams) < max_dim + 1:
        diagrams.append(np.empty((0, 2)))

    return _summarize(diagrams)


def compute_shift_metrics(
    baseline: PersistenceResult,
    challenged: PersistenceResult,
) -> TopologicalShiftMetrics:
    """Compare two persistence results and return shift metrics.

    The Wasserstein distance is computed for H0 and H1 independently.
    The *stability score* estimates what fraction of features are
    "shared" between the two diagrams (a heuristic based on the
    Wasserstein cost relative to total persistence).
    """
    # Clean diagrams (remove infinities).
    b0 = _finite(baseline.diagrams[0])
    c0 = _finite(challenged.diagrams[0])
    b1 = _finite(baseline.diagrams[1]) if len(baseline.diagrams) > 1 else np.empty((0, 2))
    c1 = _finite(challenged.diagrams[1]) if len(challenged.diagrams) > 1 else np.empty((0, 2))

    w0 = _safe_wasserstein(b0, c0)
    w1 = _safe_wasserstein(b1, c1)

    delta_h0 = challenged.num_h0 - baseline.num_h0
    delta_h1 = challenged.num_h1 - baseline.num_h1

    stab_h0 = _stability(baseline.mean_persistence_h0, w0)
    stab_h1 = _stability(baseline.mean_persistence_h1, w1)

    total = w0 + w1
    if total < 0.5:
        severity = "stable"
    elif total < 2.0:
        severity = "moderate"
    else:
        severity = "large"

    return TopologicalShiftMetrics(
        wasserstein_h0=w0,
        wasserstein_h1=w1,
        wasserstein_total=total,
        delta_num_h0=delta_h0,
        delta_num_h1=delta_h1,
        stability_h0=stab_h0,
        stability_h1=stab_h1,
        shift_severity=severity,
    )


def compute_persistence_pca(
    embeddings: np.ndarray,
    n_pca_components: int = 3,
    noise_threshold: float = 0.0,
    subsample: int | None = 300,
) -> tuple[PersistenceResult, np.ndarray]:
    """Fast persistence computation via PCA pre-reduction.

    Reduces high-dimensional embeddings to *n_pca_components* dimensions with
    PCA (a linear projection that is instant even for 1024-D inputs), then
    runs Vietoris-Rips on the low-dimensional point cloud.  H0 and H1
    computation on ~100 points in 3-D takes < 50 ms on CPU, making this
    suitable for use inside a streaming generation loop.

    Parameters
    ----------
    embeddings:
        Array of shape ``(n_points, n_features)``.
    n_pca_components:
        Target dimensionality for PCA (default 3 for 3-D scatter visualisation).
    noise_threshold:
        Minimum persistence for a feature to be kept.
    subsample:
        Sub-sample if more than this many points (see :func:`compute_persistence`).

    Returns
    -------
    (PersistenceResult, pca_coords)
        ``pca_coords`` is ``(n_points, n_pca_components)`` and can be used
        directly for scatter-plot visualisation.
    """
    pts = np.asarray(embeddings, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)

    if pts.shape[0] < 2:
        empty = np.empty((0, 2))
        pca_coords = pts[:, :n_pca_components] if pts.shape[1] >= n_pca_components else pts
        return PersistenceResult(diagrams=[empty, empty]), pca_coords.astype(np.float32)

    n_components = min(n_pca_components, pts.shape[0], pts.shape[1])
    pca = PCA(n_components=n_components)
    pca_coords = pca.fit_transform(pts).astype(np.float32)

    result = compute_persistence(
        pca_coords,
        max_dim=1,
        noise_threshold=noise_threshold,
        subsample=subsample,
    )
    return result, pca_coords


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _filter_noise(
    diagrams: list[np.ndarray],
    threshold: float,
) -> list[np.ndarray]:
    """Remove features with persistence below *threshold*."""
    filtered = []
    for dgm in diagrams:
        if dgm.size == 0:
            filtered.append(dgm)
            continue
        persistence = dgm[:, 1] - dgm[:, 0]
        mask = persistence >= threshold
        filtered.append(dgm[mask])
    return filtered


def _finite(dgm: np.ndarray) -> np.ndarray:
    """Return only finite-lifetime features (remove the essential class)."""
    if dgm.size == 0:
        return dgm
    mask = np.isfinite(dgm[:, 1])
    return dgm[mask]


def _summarize(diagrams: list[np.ndarray]) -> PersistenceResult:
    """Compute summary statistics for a persistence result."""
    res = PersistenceResult(diagrams=diagrams)

    h0 = _finite(diagrams[0])
    if h0.size > 0:
        pers = h0[:, 1] - h0[:, 0]
        res.num_h0 = len(pers)
        res.mean_persistence_h0 = float(pers.mean())
        res.max_persistence_h0 = float(pers.max())

    if len(diagrams) > 1:
        h1 = _finite(diagrams[1])
        if h1.size > 0:
            pers = h1[:, 1] - h1[:, 0]
            res.num_h1 = len(pers)
            res.mean_persistence_h1 = float(pers.mean())
            res.max_persistence_h1 = float(pers.max())

    return res


def _safe_wasserstein(dgm_a: np.ndarray, dgm_b: np.ndarray) -> float:
    """Wasserstein distance that handles empty diagrams gracefully."""
    # persim expects (N, 2) arrays; handle edge cases.
    if dgm_a.size == 0 and dgm_b.size == 0:
        return 0.0
    if dgm_a.size == 0:
        dgm_a = np.empty((0, 2))
    if dgm_b.size == 0:
        dgm_b = np.empty((0, 2))
    try:
        return float(wasserstein(dgm_a, dgm_b))
    except Exception:
        return 0.0


def _stability(
    mean_persistence: float,
    wasserstein_dist: float,
) -> float:
    """Heuristic stability score in [0, 1].

    If the wasserstein distance is 0, stability is 1 (identical).
    Otherwise we scale relative to the baseline mean persistence so
    the metric is somewhat normalised.
    """
    if mean_persistence < 1e-12:
        return 1.0 if wasserstein_dist < 1e-12 else 0.0
    ratio = wasserstein_dist / mean_persistence
    return float(max(0.0, 1.0 - ratio))
