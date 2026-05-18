"""
Interpretability tools: attention weight extraction and t-SNE visualization.

Implements:
  - extract_attention_weights():  register forward hooks on the last
    TransformerEncoderLayer to capture cross-variable attention weights
    (averaged over heads, as reported in Section 3.6 and Figure 5B).
  - extract_encoder_embeddings(): collect flattened encoder outputs for t-SNE.
  - tsne_visualization():        2-D t-SNE projection (Figure 5A).
  - plot_attention_matrix():     4x4 heatmap of mean attention weights (Figure 5B).
  - plot_tsne():                 scatter plot coloured by TIC/control label.

Notes
-----
PyTorch's nn.MultiheadAttention does NOT return attention weights by default
in nn.TransformerEncoderLayer. We enable them by monkey-patching the layer's
self_attn call inside a forward hook (need_weights=True, average_attn_weights=True).
This approach is compatible with PyTorch >= 2.0 and does not modify the model
weights or training behaviour.
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE


# ---------------------------------------------------------------------------
# Attention-weight extraction via forward hook
# ---------------------------------------------------------------------------

class _AttentionWeightCapture:
    """
    Context manager that installs a patched forward on every
    nn.MultiheadAttention module inside the TransformerEncoder and
    requests attention weights.

    After the forward pass, `self.weights` is a list of tensors of shape
    (batch, N, N) - one entry per encoder layer - where N = 4 (variables),
    already averaged over attention heads.
    """

    def __init__(self, encoder: nn.TransformerEncoder) -> None:
        self.encoder = encoder
        self.weights: List[torch.Tensor] = []
        self._hooks: list = []

    def __enter__(self) -> "_AttentionWeightCapture":
        self.weights.clear()

        def make_hook(layer: nn.TransformerEncoderLayer):
            original_forward = layer.self_attn.forward

            def patched_forward(query, key, value, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = True   # -> (batch, N, N)
                out, attn = original_forward(query, key, value, **kwargs)
                self.weights.append(attn.detach().cpu())
                return out, attn

            return patched_forward

        for layer in self.encoder.layers:
            if hasattr(layer, "self_attn"):
                orig = layer.self_attn.forward
                layer.self_attn.forward = make_hook(layer)
                self._hooks.append((layer.self_attn, orig))

        return self

    def __exit__(self, *args) -> None:
        for mha, orig in self._hooks:
            mha.forward = orig
        self._hooks.clear()


def extract_attention_weights(
    model: nn.Module,
    data_loader: "torch.utils.data.DataLoader",
    device: torch.device,
    layer_idx: int = -1,
    max_batches: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract cross-variable attention weights from Trauma-Former's encoder.

    Implements Figure 5B: mean attention weight matrix over TIC-positive windows.
    Paper Section 3.6 reports: mean HR-SBP weight 0.35, HR-DBP 0.32, all others ~0.16.

    Parameters
    ----------
    model       : TraumaFormer instance (must have .encoder attribute).
    data_loader : yields (x, mask, y, pid) batches.
    device      : torch device.
    layer_idx   : which encoder layer to read (-1 = last layer per Section 3.6).
    max_batches : cap to limit memory use.

    Returns
    -------
    attn_tic  : (K_tic, N, N)  attention matrices for TIC-positive windows.
    attn_ctrl : (K_ctrl, N, N) attention matrices for control windows.
    """
    if not hasattr(model, "encoder"):
        warnings.warn(
            "Model has no .encoder attribute; returning empty arrays. "
            "Ensure you are passing a TraumaFormer instance."
        )
        return np.zeros((0, 4, 4)), np.zeros((0, 4, 4))

    model.eval()
    all_weights: List[torch.Tensor] = []
    all_labels:  List[torch.Tensor] = []

    n_layers     = len(model.encoder.layers)
    target_layer = layer_idx % n_layers

    with _AttentionWeightCapture(model.encoder) as capture:
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= max_batches:
                    break
                x_b, mask_b, y_b, _ = batch
                x_b = x_b.to(device)
                capture.weights.clear()
                _ = model(x_b)

                if len(capture.weights) > target_layer:
                    w = capture.weights[target_layer]   # (batch, N, N)
                    all_weights.append(w)
                    all_labels.append(y_b)

    if not all_weights:
        warnings.warn("No attention weights captured.")
        return np.zeros((0, 4, 4)), np.zeros((0, 4, 4))

    W = torch.cat(all_weights, dim=0).numpy()   # (total_windows, N, N)
    Y = torch.cat(all_labels,  dim=0).numpy()   # (total_windows,)

    attn_tic  = W[Y == 1]
    attn_ctrl = W[Y == 0]
    return attn_tic, attn_ctrl


# ---------------------------------------------------------------------------
# Encoder embedding extraction (for t-SNE)
# ---------------------------------------------------------------------------

def extract_encoder_embeddings(
    model: nn.Module,
    data_loader: "torch.utils.data.DataLoader",
    device: torch.device,
    max_batches: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect flattened encoder outputs (before the classifier head) for t-SNE.

    Returns
    -------
    embeddings : (N_windows, N * d_model) float32 array.
    labels     : (N_windows,) int array.
    """
    model.eval()
    embeddings_list: List[torch.Tensor] = []
    labels_list:     List[torch.Tensor] = []

    captured: dict = {}

    def _hook(module, input, output):
        # output: (batch, N_vars, d_model) -> flatten
        captured["out"] = output.detach().cpu().flatten(start_dim=1)

    hook_handle = model.encoder.register_forward_hook(_hook)

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            x_b, _, y_b, _ = batch
            x_b = x_b.to(device)
            _ = model(x_b)
            if "out" in captured:
                embeddings_list.append(captured["out"])
                labels_list.append(y_b)

    hook_handle.remove()

    if not embeddings_list:
        return np.zeros((0, 1)), np.zeros((0,), dtype=int)

    embeddings = torch.cat(embeddings_list, dim=0).numpy()
    labels     = torch.cat(labels_list,     dim=0).numpy().astype(int)
    return embeddings, labels


# ---------------------------------------------------------------------------
# t-SNE visualization (Figure 5A)
# ---------------------------------------------------------------------------

def tsne_visualization(
    embeddings: np.ndarray,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """
    2-D t-SNE projection of high-dimensional encoder embeddings.

    Returns
    -------
    coords : (N, 2) float array of 2-D t-SNE coordinates.
    """
    tsne   = TSNE(n_components=2, perplexity=perplexity,
                  random_state=random_state, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(embeddings)
    return coords


def plot_tsne(
    coords: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Scatter plot of t-SNE coordinates coloured by TIC/control (Figure 5A)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed; skipping t-SNE plot.")
        return

    plt.figure(figsize=(8, 6))
    tic_mask  = labels == 1
    ctrl_mask = labels == 0
    plt.scatter(coords[ctrl_mask, 0], coords[ctrl_mask, 1],
                c="steelblue", alpha=0.5, s=8, label="Control")
    plt.scatter(coords[tic_mask,  0], coords[tic_mask,  1],
                c="tomato",    alpha=0.5, s=8, label="TIC-positive")
    plt.legend(markerscale=2)
    plt.title("t-SNE of Trauma-Former latent representations (Figure 5A)")
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Attention matrix heatmap (Figure 5B)
# ---------------------------------------------------------------------------

VITAL_NAMES = ["HR", "SBP", "DBP", "SpO2"]


def plot_attention_matrix(
    attn_tic: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
) -> np.ndarray:
    """
    Plot the mean cross-variable attention weight matrix for TIC-positive windows
    (Figure 5B). Highlights HR-SBP and HR-DBP cells which show highest weights
    (mean 0.35 and 0.32 respectively per Section 3.6).

    Parameters
    ----------
    attn_tic  : (K, N, N) attention weights for TIC-positive windows.
    save_path : optional file path for saving the figure.

    Returns
    -------
    mean_matrix : (N, N) mean attention weight matrix.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed; skipping attention heatmap.")
        return np.zeros((4, 4))

    if attn_tic.shape[0] == 0:
        warnings.warn("No TIC-positive attention weights provided.")
        return np.zeros((4, 4))

    mean_mat = attn_tic.mean(axis=0)   # (N, N)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mean_mat, cmap="YlOrRd", vmin=0, vmax=mean_mat.max())
    fig.colorbar(im, ax=ax, label="Mean attention weight")

    ax.set_xticks(range(4)); ax.set_xticklabels(VITAL_NAMES)
    ax.set_yticks(range(4)); ax.set_yticklabels(VITAL_NAMES)
    ax.set_xlabel("Key variable")
    ax.set_ylabel("Query variable")
    ax.set_title(
        "Cross-variable attention (TIC-positive, last encoder layer)\n"
        "Figure 5B: HR-SBP and HR-DBP interactions (gold border)"
    )

    # Annotate all cells
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{mean_mat[i, j]:.2f}",
                    ha="center", va="center",
                    color="black" if mean_mat[i, j] < 0.5 * mean_mat.max() else "white",
                    fontsize=9)

    # Gold border on HR-SBP (row 0, col 1) and HR-DBP (row 0, col 2)
    for col in [1, 2]:
        rect = plt.Rectangle((col - 0.5, -0.5), 1, 1,
                              linewidth=2, edgecolor="gold", facecolor="none")
        ax.add_patch(rect)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return mean_mat
