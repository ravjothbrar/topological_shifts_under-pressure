"""
Topological Data Analysis (TDA) module.

Computes persistence diagrams using Vietoris-Rips filtration and
derives metrics that quantify topological shifts between conversation states.

Key concepts:
  - **Persistence diagram**: A scatter plot where each point (birth, death)
    represents a topological feature. Features that persist over a wide
    range of filtration values (far from the diagonal) are "significant";
    those close to the diagonal are noise.
  - **H0 features (dim 0)**: Connected components / clusters. Many H0
    features mean the point cloud is fragmented.
  - **H1 features (dim 1)**: Loops / cycles. H1 features indicate circular
    or ring-like structures in the embedding space.
  - **Wasserstein distance**: A metric on the space of persistence diagrams.
    Larger distance = bigger topological shift between two states.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import ripser + persim; fall back to a simple scipy-based approach.
# ---------------------------------------------------------------------------
try:
    from ripser import ripser
    from persim import wasserstein as persim_wasserstein

    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    logger.warning(
        "ripser/persim not found – falling back to scipy-based TDA. "
        "Install with: pip install ripser persim"
    )


@dataclass
class PersistenceResult:
    """Holds the output of a single persistence computation."""

    diagram_h0: np.ndarray  # (n, 2) birth-death pairs for dim 0
    diagram_h1: np.ndarray  # (n, 2) birth-death pairs for dim 1
    num_h0: int
    num_h1: int
    max_persistence_h0: float
    max_persistence_h1: float
    mean_persistence_h0: float
    mean_persistence_h1: float


@dataclass
class ComparisonMetrics:
    """Metrics comparing two persistence states (baseline vs challenged)."""

    wasserstein_h0: float
    wasserstein_h1: float
    delta_num_h0: int  # change in number of H0 features
    delta_num_h1: int
    stability_score: float  # 0-1, higher = more stable
    shift_severity: str  # "stable", "moderate", "large"


# ── Fallback TDA using scipy ──────────────────────────────────────────────


def _simple_persistence(distance_matrix: np.ndarray, max_dim: int = 1):
    """
    Minimal persistence computation using distance matrix thresholding.
    This is a rough approximation; install ripser for proper results.
    """
    n = distance_matrix.shape[0]
    thresholds = np.sort(np.unique(distance_matrix.ravel()))
    if len(thresholds) > 200:
        thresholds = np.linspace(thresholds[0], thresholds[-1], 200)

    # ── H0: track connected components via union-find ──
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            return True
        return False

    births_h0 = [0.0] * n
    deaths_h0 = [float("inf")] * n
    alive = set(range(n))

    for t in thresholds:
        for i in range(n):
            for j in range(i + 1, n):
                if distance_matrix[i, j] <= t:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        # The younger component dies
                        younger = max(ri, rj)
                        union(ri, rj)
                        deaths_h0[younger] = t
                        if younger in alive:
                            alive.discard(younger)

    diagram_h0 = np.array(
        [[births_h0[i], deaths_h0[i]] for i in range(n) if deaths_h0[i] != float("inf")]
    )
    if len(diagram_h0) == 0:
        diagram_h0 = np.empty((0, 2))

    # For H1 we return an empty diagram in fallback mode
    diagram_h1 = np.empty((0, 2))
    return [diagram_h0, diagram_h1]


# ── Main analyser class ──────────────────────────────────────────────────


class TDAAnalyzer:
    """Compute and compare persistence diagrams for embedding point clouds."""

    def __init__(self, pca_dims: int = 50, max_points: int = 300):
        """
        Args:
            pca_dims: Reduce embeddings to this many dimensions before TDA.
                      High-dimensional distance matrices are noisy; PCA helps.
            max_points: Subsample to at most this many points to keep
                        Vietoris-Rips tractable (O(n^3) in the worst case).
        """
        self.pca_dims = pca_dims
        self.max_points = max_points

    def _preprocess(self, embeddings: np.ndarray) -> np.ndarray:
        """Subsample and reduce dimensionality before TDA."""
        n, d = embeddings.shape
        # Subsample if too many points
        if n > self.max_points:
            idx = np.random.default_rng(42).choice(n, self.max_points, replace=False)
            embeddings = embeddings[idx]
            n = self.max_points

        # PCA reduction
        target_dim = min(self.pca_dims, n - 1, d)
        if target_dim < d and target_dim > 0:
            pca = PCA(n_components=target_dim, random_state=42)
            embeddings = pca.fit_transform(embeddings)

        return embeddings

    def compute_persistence(self, embeddings: np.ndarray) -> PersistenceResult:
        """
        Compute the persistence diagram for a point cloud of embeddings.

        Uses Vietoris-Rips filtration up to dimension 1 (H0 and H1).
        """
        if embeddings.shape[0] < 3:
            return PersistenceResult(
                diagram_h0=np.empty((0, 2)),
                diagram_h1=np.empty((0, 2)),
                num_h0=0,
                num_h1=0,
                max_persistence_h0=0.0,
                max_persistence_h1=0.0,
                mean_persistence_h0=0.0,
                mean_persistence_h1=0.0,
            )

        processed = self._preprocess(embeddings)

        if HAS_RIPSER:
            result = ripser(processed, maxdim=1)
            diagrams = result["dgms"]
        else:
            dist_matrix = squareform(pdist(processed))
            diagrams = _simple_persistence(dist_matrix, max_dim=1)

        diag_h0 = diagrams[0] if len(diagrams) > 0 else np.empty((0, 2))
        diag_h1 = diagrams[1] if len(diagrams) > 1 else np.empty((0, 2))

        # Filter out infinite-death points for statistics
        finite_h0 = diag_h0[np.isfinite(diag_h0[:, 1])] if len(diag_h0) > 0 else np.empty((0, 2))
        finite_h1 = diag_h1[np.isfinite(diag_h1[:, 1])] if len(diag_h1) > 0 else np.empty((0, 2))

        def _persistence_vals(diag):
            if len(diag) == 0:
                return 0.0, 0.0
            lifetimes = diag[:, 1] - diag[:, 0]
            return float(np.max(lifetimes)), float(np.mean(lifetimes))

        max_h0, mean_h0 = _persistence_vals(finite_h0)
        max_h1, mean_h1 = _persistence_vals(finite_h1)

        return PersistenceResult(
            diagram_h0=diag_h0,
            diagram_h1=diag_h1,
            num_h0=len(finite_h0),
            num_h1=len(finite_h1),
            max_persistence_h0=max_h0,
            max_persistence_h1=max_h1,
            mean_persistence_h0=mean_h0,
            mean_persistence_h1=mean_h1,
        )

    def compute_wasserstein(
        self, diag_a: np.ndarray, diag_b: np.ndarray
    ) -> float:
        """
        Wasserstein distance between two persistence diagrams.

        This measures how much "work" is needed to transform one diagram
        into the other — larger values indicate a bigger topological shift.
        """
        # Clean: keep only finite points
        a = diag_a[np.isfinite(diag_a).all(axis=1)] if len(diag_a) > 0 else np.empty((0, 2))
        b = diag_b[np.isfinite(diag_b).all(axis=1)] if len(diag_b) > 0 else np.empty((0, 2))

        if len(a) == 0 and len(b) == 0:
            return 0.0

        if HAS_RIPSER:
            return float(persim_wasserstein(a, b))

        # Fallback: simple L2 distance between sorted persistence values
        lifetimes_a = np.sort(a[:, 1] - a[:, 0])[::-1] if len(a) > 0 else np.array([0.0])
        lifetimes_b = np.sort(b[:, 1] - b[:, 0])[::-1] if len(b) > 0 else np.array([0.0])
        max_len = max(len(lifetimes_a), len(lifetimes_b))
        la = np.pad(lifetimes_a, (0, max_len - len(lifetimes_a)))
        lb = np.pad(lifetimes_b, (0, max_len - len(lifetimes_b)))
        return float(np.sqrt(np.sum((la - lb) ** 2)))

    def compare(
        self, baseline: PersistenceResult, challenged: PersistenceResult
    ) -> ComparisonMetrics:
        """
        Compare two persistence states and produce summary metrics.

        The stability score is computed as:
          1.0 - normalised_wasserstein_distance
        clamped to [0, 1].  A score near 1 means the topology barely
        changed; near 0 means a massive shift.
        """
        w_h0 = self.compute_wasserstein(baseline.diagram_h0, challenged.diagram_h0)
        w_h1 = self.compute_wasserstein(baseline.diagram_h1, challenged.diagram_h1)

        total_w = w_h0 + w_h1
        # Normalise by the baseline scale so we get a 0-1 stability score
        baseline_scale = max(
            baseline.max_persistence_h0, baseline.max_persistence_h1, 1e-6
        )
        stability = max(0.0, 1.0 - total_w / (baseline_scale * 4))

        if stability > 0.7:
            severity = "stable"
        elif stability > 0.4:
            severity = "moderate"
        else:
            severity = "large"

        return ComparisonMetrics(
            wasserstein_h0=w_h0,
            wasserstein_h1=w_h1,
            delta_num_h0=challenged.num_h0 - baseline.num_h0,
            delta_num_h1=challenged.num_h1 - baseline.num_h1,
            stability_score=stability,
            shift_severity=severity,
        )
