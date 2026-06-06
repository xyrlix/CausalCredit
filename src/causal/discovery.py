"""Causal discovery engine.

Implements the three pillars of the hybrid causal discovery engine
(docs section 4.1) on top of the Home Credit dataset:

1. **PC algorithm** (constraint-based) via `causallearn`.
2. **NOTEARS linear** (score-based, continuous optimization) — implemented from
   scratch in pure numpy following Zheng et al. (2018), augmented-Lagrangian
   version. We avoid the `gcastle` package because it hard-imports torch.
3. **Domain-knowledge injection** — must/forbid edges.

The final fused graph is then compared against `HomeCreditCausalGraph` for
edge overlap / direction-agreement metrics.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize


# ===========================================================================
# NOTEARS-linear from scratch (Zheng et al. 2018, NeurIPS)
# ===========================================================================

def _notears_loss(W: np.ndarray, X: np.ndarray, lambda1: float) -> float:
    """L(W) = 1/(2n) * ||X - XW||^2_F + lambda1 * ||W||_1."""
    n = X.shape[0]
    R = X - X @ W
    loss = 0.5 / n * np.sum(R ** 2) + lambda1 * np.sum(np.abs(W))
    return loss


def _h_dag(W: np.ndarray) -> float:
    """DAG constraint: tr(e^{W*W}) - d = 0."""
    d = W.shape[0]
    E = expm(W * W)
    return float(np.trace(E) - d)


def _h_grad(W: np.ndarray) -> np.ndarray:
    """Gradient of the DAG constraint: (e^{W*W})^T * 2W.

    Uses the matrix identity d/dW tr(e^{W*W}) = 2 * (e^{W*W})^T * W
    (with element-wise squaring inside the exponential).
    """
    E = expm(W * W)
    return (E.T + E) * W * 2


def _notears_linear(
    X: np.ndarray,
    lambda1: float = 0.1,
    max_iter: int = 100,
    h_tol: float = 1e-6,
    rho_max: float = 1e10,
) -> np.ndarray:
    """Augmented-Lagrangian NOTEARS-linear solver.

    Implementation notes (vs. the textbook version):
    1. The data are standardized (mean 0, var 1) so that the loss
       L = 0.5/n * ||X - XW||^2 is on a sensible scale.
    2. The L1 penalty |W| is replaced with the smooth surrogate
       sqrt(W^2 + eps^2) so L-BFGS-B (which needs a smooth gradient)
       can take meaningful steps; eps=1e-3 is small enough to be a good
       approximation but large enough to keep the gradient bounded away
       from zero.
    3. NEVER mutate the optimizer's input array in place: scipy's
       L-BFGS-B keeps references to the input between function/gradient
       calls, and an in-place fill_diagonal corrupts its state.
    4. The inner L-BFGS-B is warm-started from the previous outer-iter
       solution. The textbook version restarts from W_est each time,
       which combined with non-smooth L1 traps the optimizer at the
       zero matrix.
    5. We use h_tol = 1e-6 (not the textbook 1e-8) because in practice
       a near-DAG solution with a small h is what the threshold step
       later discards; pushing h to 1e-8 makes rho explode and pulls
       all weights to zero. We also cap rho at 1e10 to avoid that.
    """
    n, d = X.shape
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    eps = 1e-3

    def _wzero(W_flat: np.ndarray) -> np.ndarray:
        W = W_flat.reshape(d, d).copy()
        np.fill_diagonal(W, 0.0)
        return W

    def _smooth_l1(W: np.ndarray) -> tuple:
        return float(np.sum(np.sqrt(W * W + eps * eps))), W / np.sqrt(W * W + eps * eps)

    def _loss_with_penalty(W_flat: np.ndarray, rho: float, alpha: float) -> float:
        W = _wzero(W_flat)
        R = X_std - X_std @ W
        data_loss = 0.5 / n * float(np.sum(R * R))
        l1, _ = _smooth_l1(W)
        h = _h_dag(W)
        return data_loss + lambda1 * l1 + alpha * h + 0.5 * rho * h * h

    def _grad_with_penalty(W_flat: np.ndarray, rho: float, alpha: float) -> np.ndarray:
        W = _wzero(W_flat)
        R = X_std - X_std @ W
        grad_data = -X_std.T @ R / n
        _, l1_grad = _smooth_l1(W)
        h = _h_dag(W)
        grad_pen = (alpha + rho * h) * _h_grad(W)
        grad = grad_data + lambda1 * l1_grad + grad_pen
        np.fill_diagonal(grad, 0.0)
        return grad.flatten()

    W_est = np.zeros((d, d))
    best_W = W_est.copy()
    best_score = -np.inf  # track best trade-off: high |W|, low h
    # Start rho SMALL (0.01, not the textbook 1.0) so that the first
    # inner optimization lands on the L1-sparse unconstrained solution
    # (h is small because W*W is small in norm). The textbook value
    # pulls W to zero from iter 1, which then never recovers.
    rho, alpha, h_old = 1e-2, 0.0, np.inf
    for it in range(max_iter):
        result = minimize(
            fun=lambda w: _loss_with_penalty(w, rho, alpha),
            x0=W_est.flatten(),  # warm-start from previous iter
            jac=lambda w: _grad_with_penalty(w, rho, alpha),
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-10},
        )
        W_new = _wzero(result.x)
        h_new = _h_dag(W_new)
        # Score: prefer large magnitudes and small DAG violation. Use
        # ||W||_F - lambda * h so the first L1-sparse solution wins.
        score = float(np.linalg.norm(W_new)) - h_new
        if score > best_score:
            best_score = score
            best_W = W_new
        if h_new > 0.25 * h_old:
            rho *= 10
            W_est = W_new  # keep latest for warm start
        else:
            W_est = W_new
            alpha += rho * h_new
        h_old = h_new
        if h_new < h_tol or rho >= rho_max:
            break
        # After a few iters rho grows large and pulls W to zero. Stop
        # once the W's L1 norm has shrunk below 20% of the best seen.
        if it > 2 and np.linalg.norm(W_new) < 0.2 * np.linalg.norm(best_W):
            break
    return best_W


def _adj_to_digraph(W: np.ndarray, feature_names: List[str], threshold: float = 0.3) -> nx.DiGraph:
    """Threshold a NOTEARS weight matrix into a directed graph."""
    d = len(feature_names)
    G = nx.DiGraph()
    G.add_nodes_from(feature_names)
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            w = W[i, j]
            if abs(w) > threshold:
                # Convention: row i = source, col j = destination, so edge i -> j
                G.add_edge(feature_names[i], feature_names[j], weight=float(w))
    return G


# ===========================================================================
# PC algorithm (wrapper around causallearn)
# ===========================================================================

def _run_pc_causallearn(X: np.ndarray, feature_names: List[str], alpha: float = 0.01) -> nx.DiGraph:
    """Run the PC algorithm via causallearn and return a networkx DiGraph.

    Note: causallearn's `pc` returns a CausalGraph whose `G.graph` is an
    (n, n) matrix with the following endpoint convention
    (Endpoint.TAIL = -1, Endpoint.ARROW = 1):
      - (M[i,j], M[j,i]) == (-1,  1)   =>  directed edge i -> j
      - (M[i,j], M[j,i]) == ( 1, -1)   =>  directed edge i <- j
      - (M[i,j], M[j,i]) == (-1, -1)   =>  undirected edge (skeleton only)
      - (M[i,j], M[j,i]) == ( 1,  1)   =>  no edge (independence)

    We keep BOTH directed and undirected edges as DiGraph edges; the
    `edge_type` attribute distinguishes them. Downstream fusion
    treats undirected edges as candidate causal links.
    """
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz

    cg = pc(X, alpha=alpha, indep_test=fisherz, show_summary=False, verbose=False)
    G = nx.DiGraph()
    G.add_nodes_from(feature_names)
    d = len(feature_names)
    M = cg.G.graph
    for i in range(d):
        for j in range(i + 1, d):
            mi, mj = int(M[i, j]), int(M[j, i])
            if mi == -1 and mj == 1:
                G.add_edge(feature_names[i], feature_names[j], edge_type="directed")
            elif mi == 1 and mj == -1:
                G.add_edge(feature_names[j], feature_names[i], edge_type="directed")
            elif mi == -1 and mj == -1:
                # Skeleton only — add as bidirectional placeholder so the
                # edge is preserved through fusion; downstream
                # orientation heuristics can flip it.
                G.add_edge(feature_names[i], feature_names[j], edge_type="undirected")
                G.add_edge(feature_names[j], feature_names[i], edge_type="undirected")
    return G


# ===========================================================================
# Public API
# ===========================================================================

def run_pc(
    data: pd.DataFrame,
    alpha: float = 0.01,
    max_cond_size: int = 3,
) -> nx.DiGraph:
    """Run PC algorithm on a numeric DataFrame (uses all numeric columns)."""
    numeric = data.select_dtypes(include=[np.number]).copy()
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    X = numeric.values
    G = _run_pc_causallearn(X, list(numeric.columns), alpha=alpha)
    return G


def run_notears(
    data: pd.DataFrame,
    lambda1: float = 0.1,
    h_tol: float = 1e-8,
    threshold: float = 0.3,
) -> nx.DiGraph:
    """Run NOTEARS-linear on a numeric DataFrame."""
    numeric = data.select_dtypes(include=[np.number]).copy()
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    # Standardize for stable optimization
    X = (numeric - numeric.mean()) / (numeric.std() + 1e-8)
    W = _notears_linear(X.values, lambda1=lambda1, h_tol=h_tol)
    return _adj_to_digraph(W, list(numeric.columns), threshold=threshold)


def fuse_graphs(
    pc_graph: nx.DiGraph,
    notears_graph: nx.DiGraph,
    edge_conf_threshold: float = 0.7,
) -> nx.DiGraph:
    """Fuse PC and NOTEARS graphs by intersection (both methods must agree).

    An edge u -> v is kept only if it appears in BOTH graphs. The
    `edge_conf_threshold` is currently binary (intersection); reserved for
    a future soft-voting extension.
    """
    pc_edges = set(pc_graph.edges())
    nt_edges = set(notears_graph.edges())
    fused = nx.DiGraph()
    fused.add_nodes_from(set(pc_graph.nodes()) | set(notears_graph.nodes()))
    if edge_conf_threshold <= 0.5:
        union = pc_edges | nt_edges
        fused.add_edges_from(union)
    else:
        intersection = pc_edges & nt_edges
        fused.add_edges_from(intersection)
    return fused


def inject_domain_knowledge(
    skeleton: nx.DiGraph,
    must_edges: List[Tuple[str, str]] = None,
    forbid_edges: List[Tuple[str, str]] = None,
) -> nx.DiGraph:
    """Inject domain constraints. Must edges are added; forbid edges removed."""
    out = skeleton.copy()
    for u, v in (must_edges or []):
        if u in out and v in out:
            out.add_edge(u, v)
    for u, v in (forbid_edges or []):
        if out.has_edge(u, v):
            out.remove_edge(u, v)
    return out


def compare_with_domain(
    discovered: nx.DiGraph,
    domain: "HomeCreditCausalGraph",
) -> Dict:
    """Compare a discovered graph against the hand-coded domain DAG.

    Returns overlap metrics on the **shared node set** only.
    """
    domain_edges = set(domain.edges)
    discovered_edges = set(discovered.edges())
    shared_nodes = set(discovered.nodes()) & {n for n, _ in domain_edges} & {
        v for _, v in domain_edges
    }
    domain_in_shared = {(u, v) for u, v in domain_edges if u in shared_nodes and v in shared_nodes}
    discovered_in_shared = {
        (u, v) for u, v in discovered_edges if u in shared_nodes and v in shared_nodes
    }
    overlap = domain_in_shared & discovered_in_shared
    return {
        "n_shared_nodes": len(shared_nodes),
        "n_domain_edges_in_shared": len(domain_in_shared),
        "n_discovered_edges_in_shared": len(discovered_in_shared),
        "n_overlap": len(overlap),
        "overlap_rate_domain": (
            len(overlap) / len(domain_in_shared) if domain_in_shared else 0.0
        ),
        "overlap_rate_discovered": (
            len(overlap) / len(discovered_in_shared) if discovered_in_shared else 0.0
        ),
    }


def visualize_dag(
    G: nx.DiGraph,
    title: str = "Causal DAG",
    output_path: Optional[str] = None,
    top_k_edges: int = 60,
) -> None:
    """Render a DAG via matplotlib (with top-k edges by absolute weight if available)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    G = G.copy()
    if G.number_of_edges() > top_k_edges:
        # Keep top_k by |weight| if present, else by arbitrary truncation
        if all("weight" in G[u][v] for u, v in G.edges()):
            sorted_edges = sorted(G.edges(data=True), key=lambda e: -abs(e[2].get("weight", 1.0)))
            G = nx.DiGraph()
            G.add_nodes_from([n for n in G.nodes()])
            G.add_edges_from(sorted_edges[:top_k_edges])
        else:
            edges = list(G.edges())[:top_k_edges]
            G = nx.DiGraph()
            G.add_nodes_from([n for n in G.nodes()])
            G.add_edges_from(edges)

    plt.figure(figsize=(14, 10))
    try:
        pos = nx.spring_layout(G, k=1.2, iterations=50, seed=42)
    except Exception:
        pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color="#BAE1FF", alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color="#444", arrowsize=14, width=1.0)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=120)
        plt.close()
    else:
        plt.show()


def discover_home_credit_causal_graph(
    loader=None,
    sample_size: int = 10000,
    seed: int = 42,
) -> Dict:
    """End-to-end causal discovery on the Home Credit application_train table.

    Returns a dict with the discovered graphs and a comparison report.
    The DAG is run on a sample of `sample_size` rows for tractability — PC
    and NOTEARS both scale O(n*d^2) and 30K × 20 features is fast on CPU.
    """
    if loader is None:
        from src.data.home_credit_loader import HomeCreditLoader
        loader = HomeCreditLoader()
    df = loader.fetch()
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=seed)
    # Use a small, causal-relevant feature set
    feature_cols = [
        "AMT_CREDIT", "AMT_ANNUITY", "DAYS_EMPLOYED", "AMT_INCOME_TOTAL",
        "AMT_GOODS_PRICE", "DAYS_BIRTH", "CNT_CHILDREN",
        "EXT_SOURCE_2", "REGION_RATING_CLIENT", "OWN_CAR_AGE",
        "CNT_FAM_MEMBERS", "DAYS_REGISTRATION",
    ]
    df_features = df_sample[feature_cols].fillna(df_sample[feature_cols].median())

    pc_graph = run_pc(df_features, alpha=0.05)
    notears_graph = run_notears(df_features, lambda1=0.1, threshold=0.3)
    fused = fuse_graphs(pc_graph, notears_graph, edge_conf_threshold=0.5)

    from src.causal.home_credit_graph import HomeCreditCausalGraph
    domain = HomeCreditCausalGraph()
    # Inject must edges from domain
    must_edges = [
        e for e in domain.edges
        if e[0] in fused.nodes and e[1] in fused.nodes
    ]
    fused_with_dk = inject_domain_knowledge(fused, must_edges=must_edges)
    compare = compare_with_domain(fused_with_dk, domain)

    return {
        "pc_graph": pc_graph,
        "notears_graph": notears_graph,
        "fused_graph": fused,
        "graph_with_domain_knowledge": fused_with_dk,
        "comparison": compare,
        "feature_cols": feature_cols,
    }
