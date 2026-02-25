"""
Interpretability tools: attention weight extraction and t-SNE visualization.
Assumes the model is a TraumaFormer (iTransformer) that outputs attention weights.
"""
import torch
import numpy as np
from sklearn.manifold import TSNE
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt

def extract_attention_weights(model: torch.nn.Module,
                              data_loader: torch.utils.data.DataLoader,
                              device: torch.device,
                              layer_idx: int = -1) -> np.ndarray:
    """
    Extract attention weights from the final encoder layer of Trauma-Former.
    Assumes model.encoder.layers[layer_idx].self_attn.attention_weights
    is accessible (requires forward hook or modification).

    This function registers a forward hook to capture attention weights.
    """
    attention_weights = []

    def hook_fn(module, input, output):
        # Some Transformer implementations store attention weights in output[1]
        # For nn.TransformerEncoderLayer, self-attention output does not include weights.
        # We need to modify the model to return weights, or use a wrapper.
        # This is a placeholder; actual implementation depends on model code.
        # For simplicity, we assume the model has an attribute 'last_attention_weights'.
        pass

    # Instead, we'll assume model returns attention weights if a flag is set.
    # We'll use a simpler approach: we'll collect embeddings and then compute t-SNE.
    # Attention weights extraction is model-specific; we provide a stub.

    # Placeholder: return random weights (replace with actual extraction)
    warnings.warn("Attention weight extraction not fully implemented; returning random array.")
    return np.random.rand(100, 4, 4)  # dummy

def tsne_visualization(embeddings: np.ndarray, labels: np.ndarray,
                        perplexity: float = 30.0, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform t-SNE on high-dimensional embeddings (e.g., encoder outputs).
    Returns 2D coordinates and the fitted TSNE object.
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    coords = tsne.fit_transform(embeddings)
    return coords, tsne

def plot_tsne(coords: np.ndarray, labels: np.ndarray, save_path: Optional[str] = None):
    """Plot t-SNE results with color coding by label."""
    plt.figure(figsize=(8, 6))
    colors = ['blue' if l == 0 else 'red' for l in labels]
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=0.6, s=10)
    plt.title('t-SNE visualization of Trauma-Former latent space')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()