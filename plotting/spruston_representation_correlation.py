"""Near vs far trial representation correlation (Spruston-style) + training evolution grid."""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from checkpoint import list_available_checkpoints, load_checkpoint


def _pearson_r(u: np.ndarray, v: np.ndarray) -> float:
    """Pearson r with u_k, v_k as paired samples across channel index k."""
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if u.shape != v.shape:
        return float("nan")
    su = float(np.std(u))
    sv = float(np.std(v))
    if su < 1e-12 or sv < 1e-12:
        return 1.0 if np.allclose(u, v) else 0.0
    return float(np.corrcoef(u, v)[0, 1])


def _corr_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (Ta, d), b: (Tb, d) → (Ta, Tb) Pearson r between rows across dims."""
    ta, tb = a.shape[0], b.shape[0]
    out = np.zeros((ta, tb), dtype=np.float64)
    for i in range(ta):
        for j in range(tb):
            out[i, j] = _pearson_r(a[i], b[j])
    return out


def _attention_heads(model):
    if getattr(model, "blocks", None) is None:
        return model.sa_heads.heads
    return model.blocks[-1].sa_heads.heads


@torch.no_grad()
def _representation_dict(model: torch.nn.Module, idx: torch.Tensor) -> dict[str, np.ndarray]:
    """idx: (1, T). Stacks for Pearson r **across channels** (paired u_k, v_k).

    Token+position **sum** has only n_embd channels; Pearson needs ≥3 paired samples,
    so with n_embd=2 correlating only the sum collapses to r ∈ {−1, 1}. We therefore
    concatenate **separate** token and position embeddings, and prepend that to Q/K/V/pre.
    """
    device = idx.device
    B, T = idx.shape
    token_emb = model.token_embedding(idx)
    positions = torch.arange(T, device=device) % model.block_size
    pos_emb = model.position_embedding_table(positions)
    e = token_emb + pos_emb
    base = torch.cat([token_emb, pos_emb], dim=-1)

    heads = _attention_heads(model)
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        attn_input = e
    else:
        x = e
        for block in blocks[:-1]:
            x, _ = block(x)
        attn_input = blocks[-1].ln1(x)

    qs = torch.cat([h.query(attn_input) for h in heads], dim=-1)
    ks = torch.cat([h.key(attn_input) for h in heads], dim=-1)
    vs = torch.cat([h.value(attn_input) for h in heads], dim=-1)

    _, _, pre = model(idx, return_hidden=True)

    return {
        "emb": base[0].detach().cpu().numpy(),
        "q": torch.cat([base, qs], dim=-1)[0].detach().cpu().numpy(),
        "k": torch.cat([base, ks], dim=-1)[0].detach().cpu().numpy(),
        "v": torch.cat([base, vs], dim=-1)[0].detach().cpu().numpy(),
        "pre": torch.cat([base, pre], dim=-1)[0].detach().cpu().numpy(),
    }


def plot_spruston_near_far_evolution_grid(
    config_name_actual: str,
    itos: dict,
    seq_near_ids: list[int],
    seq_far_ids: list[int],
    save_path: str | None = None,
    n_times: int = 5,
):
    """
    5 rows × n_times columns: columns are checkpoints; rows are Embedding (tok ‖ pos),
    Q/K/V/pre (each = tok ‖ pos ‖ that representation).

    Each cell: Pearson r across channels between stacked vectors at near vs far positions
    (emb = tok‖pos; q/k/v/pre = tok‖pos‖· ; cf. Sun et al.-style population correlation).
    """
    steps_sorted = sorted(list_available_checkpoints(config_name_actual))

    def _even_indices(n: int, k: int) -> list[int]:
        """k indices from 0..n-1 including both endpoints (evenly spaced)."""
        if k <= 1:
            return [n - 1] if n else []
        return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]

    if len(steps_sorted) >= n_times:
        idxs = _even_indices(len(steps_sorted), n_times)
        steps_for_cols = [steps_sorted[i] for i in idxs]
    elif steps_sorted:
        steps_for_cols = list(steps_sorted)
        while len(steps_for_cols) < n_times:
            steps_for_cols.append(steps_sorted[-1])
        steps_for_cols = steps_for_cols[:n_times]
    else:
        steps_for_cols = [None] * n_times

    labels_near = [str(itos[int(seq_near_ids[t])]) for t in range(len(seq_near_ids))]
    labels_far = [str(itos[int(seq_far_ids[t])]) for t in range(len(seq_far_ids))]

    row_keys = ["emb", "q", "k", "v", "pre"]
    row_labels = ["Embedding\n(tok ‖ pos)", "Q", "K", "V", "Pre-logit"]

    fig, axes = plt.subplots(
        len(row_keys),
        n_times,
        figsize=(3.4 * n_times + 1.5, 2.9 * len(row_keys) + 1),
        squeeze=False,
    )
    axes = np.asarray(axes)

    last_im = None
    for col, step in enumerate(steps_for_cols):
        ckpt = load_checkpoint(config_name_actual, step=step)
        if ckpt is None:
            for row in range(len(row_keys)):
                axes[row, col].text(0.5, 0.5, "missing ckpt", ha="center", va="center")
                axes[row, col].axis("off")
            continue

        model = ckpt["model"]
        device = next(model.parameters()).device
        Xn = torch.tensor([seq_near_ids], dtype=torch.long, device=device)
        Xf = torch.tensor([seq_far_ids], dtype=torch.long, device=device)

        rn = _representation_dict(model, Xn)
        rf = _representation_dict(model, Xf)

        for row, key in enumerate(row_keys):
            ax = axes[row, col]
            M = _corr_matrix(rn[key], rf[key])
            im = ax.imshow(M, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
            last_im = im
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ttl = f"step {step}" if step is not None else "final"
                ax.set_title(ttl, fontsize=11)
            if col == 0:
                ax.text(
                    -0.28,
                    0.5,
                    row_labels[row],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10,
                )
            if row == len(row_keys) - 1:
                ax.set_xticks(range(len(labels_far)))
                ax.set_xticklabels(labels_far, rotation=55, ha="right", fontsize=7)
                ax.set_xlabel("Far trial position", fontsize=9)
            if col == 0:
                ax.set_yticks(range(len(labels_near)))
                ax.set_yticklabels(labels_near, fontsize=7)

    fig.suptitle(
        "Near vs far: Pearson r across channels over training\n"
        "(emb = tok ‖ pos; other rows prepend tok ‖ pos to Q/K/V/pre-logit)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout(rect=[0.06, 0.03, 0.92, 0.94])
    if last_im is not None:
        fig.colorbar(
            last_im,
            ax=axes.ravel().tolist(),
            shrink=0.72,
            pad=0.02,
            label="Pearson r",
        )

    if save_path:
        d = os.path.dirname(save_path)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)


def plot_spruston_near_far_representation_correlation(
    model,
    itos: dict,
    seq_near_ids: list[int],
    seq_far_ids: list[int],
    save_path: str | None = None,
    title: str | None = None,
):
    """Single-panel pre-logit correlation: tok ‖ pos ‖ pre-logit activations."""
    model.eval()
    device = next(model.parameters()).device
    Ln, Lf = len(seq_near_ids), len(seq_far_ids)

    Xn = torch.tensor([seq_near_ids], dtype=torch.long, device=device)
    Xf = torch.tensor([seq_far_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        rn = _representation_dict(model, Xn)
        rf = _representation_dict(model, Xf)
        mat = _corr_matrix(rn["pre"], rf["pre"])
    labels_near = [str(itos[int(seq_near_ids[t])]) for t in range(Ln)]
    labels_far = [str(itos[int(seq_far_ids[t])]) for t in range(Lf)]

    fig_w = max(8.0, Lf * 0.65 + 3)
    fig_h = max(6.0, Ln * 0.55 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_xticks(range(Lf))
    ax.set_xticklabels(labels_far, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(Ln))
    ax.set_yticklabels(labels_near, fontsize=10)
    ax.set_xlabel("Far trial — position / segment symbol", fontsize=11)
    ax.set_ylabel("Near trial — position / segment symbol", fontsize=11)
    ttl = title or (
        "Cross-track correlation (Pre-logit)\n"
        "(Pearson r across tok ‖ pos ‖ pre channels)"
    )
    ax.set_title(ttl, fontsize=12)
    fig.tight_layout()
    if save_path:
        d = os.path.dirname(save_path)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
