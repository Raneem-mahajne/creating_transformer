"""Shared utilities: journal mode, annotation helpers, data extraction."""
import random
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib.text as _mtext

from data import get_batch_from_sequences

# ---------------------------------------------------------------------------
# Journal mode: constrain figure sizes to A4 page dimensions and scale fonts
# ---------------------------------------------------------------------------
_JOURNAL_MODE = False
_JOURNAL_MAX_WIDTH = 7.0     # inches (A4 text width with typical margins)
_JOURNAL_MAX_HEIGHT = 9.5    # inches (A4 text height with typical margins)
_JOURNAL_DPI = 300           # print-quality DPI
_JOURNAL_BASE_FONT_SCALE = 0.7
_JOURNAL_FONT_SCALE = 0.7    # active scale — updated per figure
_orig_rcparams = {}           # saved rcParams to restore later

_orig_plt_figure = plt.figure
_orig_plt_subplots = plt.subplots
_orig_plt_savefig = plt.savefig
_orig_text_set_fontsize = _mtext.Text.set_fontsize


def _constrain_figsize(figsize):
    """Scale figsize to fit A4 page while keeping reasonable proportions.

    Width is capped at MAX_WIDTH.  Height is reduced by the *square root*
    of the width scale factor so wide figures get narrower without
    collapsing vertically.  Height is then capped at MAX_HEIGHT.
    """
    if figsize is None:
        return figsize
    w, h = figsize
    if w > _JOURNAL_MAX_WIDTH:
        width_scale = _JOURNAL_MAX_WIDTH / w
        w = _JOURNAL_MAX_WIDTH
        h = h * (width_scale ** 0.65)
    h = min(h, _JOURNAL_MAX_HEIGHT)
    return (w, h)


def _update_font_scale_for_figure(orig_w):
    """Set per-figure font scale based on how much width was reduced."""
    global _JOURNAL_FONT_SCALE
    if orig_w <= _JOURNAL_MAX_WIDTH:
        _JOURNAL_FONT_SCALE = _JOURNAL_BASE_FONT_SCALE
    else:
        width_ratio = _JOURNAL_MAX_WIDTH / orig_w
        _JOURNAL_FONT_SCALE = max(0.45, width_ratio ** 0.45)


def _patched_figure(*args, **kwargs):
    if _JOURNAL_MODE and 'figsize' in kwargs:
        orig_w = kwargs['figsize'][0]
        kwargs['figsize'] = _constrain_figsize(kwargs['figsize'])
        _update_font_scale_for_figure(orig_w)
    return _orig_plt_figure(*args, **kwargs)


def _patched_subplots(*args, **kwargs):
    if _JOURNAL_MODE and 'figsize' in kwargs:
        orig_w = kwargs['figsize'][0]
        kwargs['figsize'] = _constrain_figsize(kwargs['figsize'])
        _update_font_scale_for_figure(orig_w)
    return _orig_plt_subplots(*args, **kwargs)


def _patched_savefig(*args, **kwargs):
    if _JOURNAL_MODE:
        if 'dpi' not in kwargs:
            kwargs['dpi'] = _JOURNAL_DPI
        elif kwargs['dpi'] < _JOURNAL_DPI:
            kwargs['dpi'] = _JOURNAL_DPI
    return _orig_plt_savefig(*args, **kwargs)


def _patched_text_set_fontsize(self, fontsize):
    """Scale numeric font sizes by the current per-figure scale in journal mode."""
    if _JOURNAL_MODE and fontsize is not None:
        try:
            fontsize = float(fontsize) * _JOURNAL_FONT_SCALE
        except (TypeError, ValueError):
            pass  # string like 'large' — resolved via scaled rcParams['font.size']
    return _orig_text_set_fontsize(self, fontsize)


plt.figure = _patched_figure
plt.subplots = _patched_subplots
plt.savefig = _patched_savefig
_mtext.Text.set_fontsize = _patched_text_set_fontsize
_mtext.Text.set_size = _patched_text_set_fontsize  # alias used by some callers


_JOURNAL_RCPARAMS = {
    'axes.titlepad': 4,
    'axes.labelpad': 2,
    'xtick.major.pad': 2,
    'ytick.major.pad': 2,
}


def set_journal_mode(max_width=7.0, max_height=9.5, dpi=300, font_scale=0.7):
    """Activate journal mode: cap figure sizes to A4 page, use print-quality DPI, and scale fonts."""
    global _JOURNAL_MODE, _JOURNAL_MAX_WIDTH, _JOURNAL_MAX_HEIGHT, _JOURNAL_DPI
    global _JOURNAL_BASE_FONT_SCALE, _JOURNAL_FONT_SCALE, _orig_rcparams
    _JOURNAL_MODE = True
    _JOURNAL_MAX_WIDTH = max_width
    _JOURNAL_MAX_HEIGHT = max_height
    _JOURNAL_DPI = dpi
    _JOURNAL_BASE_FONT_SCALE = font_scale
    _JOURNAL_FONT_SCALE = font_scale
    _orig_rcparams = {}
    _orig_rcparams['font.size'] = plt.rcParams['font.size']
    for key, val in _JOURNAL_RCPARAMS.items():
        _orig_rcparams[key] = plt.rcParams.get(key)
        plt.rcParams[key] = val
    plt.rcParams['font.size'] = _orig_rcparams['font.size'] * font_scale


def clear_journal_mode():
    """Deactivate journal mode: restore original figure sizes, DPI, and font sizes."""
    global _JOURNAL_MODE
    _JOURNAL_MODE = False
    for key, val in _orig_rcparams.items():
        if val is not None:
            plt.rcParams[key] = val


def _label_panels(axes, fontsize=None, x=-0.02, y=1.06):
    """Add (a), (b), (c)... panel labels to a list of axes (journal mode only).

    For figures with a tall bottom panel (e.g., embeddings figure where panel (e)
    spans two columns), the last axis often needs a slightly lower label to
    avoid overlapping the title. We therefore nudge the last label down a bit.
    """
    if not _JOURNAL_MODE or len(axes) <= 1:
        return
    fs = fontsize or 11
    n = len(axes)
    for i, ax in enumerate(axes):
        label = chr(ord('a') + i)
        # Slightly lower label for the final, tall panel (e.g., panel (e)).
        y_pos = y - 0.06 if i == n - 1 else y
        ax.text(x, y, f'({label})', transform=ax.transAxes,
                fontsize=fs, fontweight='bold', va='bottom', ha='left')


def annotate_sequence(ax, chars, y=1.02, fontsize=10):
    s = "".join(chars)
    ax.text(
        0.5, y, s,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=fontsize,
        fontfamily="monospace",
        clip_on=False,
    )

def sparse_ticks(chars, every=2):
    """Return a list same length as chars, with only every Nth label shown."""
    out = []
    for i, ch in enumerate(chars):
        out.append(ch if i % every == 0 else "")
    return out

# Unicode subscript digits
_SUBSCRIPT_DIGITS = "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"

def _pos_subscript(n):
    """Format integer n as subscript (e.g. 3 -> subscript 3)."""
    return "".join(_SUBSCRIPT_DIGITS[int(d)] for d in str(n))

def _token_pos_label(token_str, pos_idx):
    """Label for token at position: token with position as subscript."""
    return f"{token_str}{_pos_subscript(pos_idx)}"

def _pos_only_label(pos_idx):
    """Label for position-only: p with subscript."""
    return f"p{_pos_subscript(pos_idx)}"


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_epoch_stats(model, train_sequences, block_size, batch_size):
    model.eval()

    # single representative batch
    X, _ = get_batch_from_sequences(train_sequences, block_size, batch_size)

    # embeddings
    token_emb = model.token_embedding(X)  # (B,T,C)
    pos_emb = model.position_embedding_table(
        torch.arange(token_emb.size(1), device=X.device)
    )
    x = token_emb + pos_emb

    # attention internals
    out, wei = model.sa_heads(x)

    # q,k,v (recompute explicitly)
    q = model.sa_heads.query(x)
    k = model.sa_heads.key(x)
    v = model.sa_heads.value(x)

    stats = {
        "emb_norm": token_emb.norm(dim=-1).mean().item(),
        "x_norm": x.norm(dim=-1).mean().item(),
        "q_norm": q.norm(dim=-1).mean().item(),
        "k_norm": k.norm(dim=-1).mean().item(),
        "v_norm": v.norm(dim=-1).mean().item(),
        "attn_entropy": (-wei * (wei + 1e-9).log()).sum(dim=-1).mean().item(),
        "out_norm": out.norm(dim=-1).mean().item(),
    }

    return stats

@torch.no_grad()
def get_attention_snapshot_from_X(model, X, itos):
    model.eval()
    B, T = X.shape
    chars = [itos[i.item()] for i in X[0]]

    token_emb = model.token_embedding(X)
    pos = torch.arange(T, device=X.device) % model.block_size
    pos_emb = model.position_embedding_table(pos)
    x = token_emb + pos_emb

    out, wei = model.sa_heads(x)  # wei will be a tuple/list of per-head weights

    snap = {
        "chars": chars,
        "token_emb": token_emb[0].cpu().numpy(),
        "x": x[0].cpu().numpy(),
        "wei": np.stack([w[0].cpu().numpy() for w in wei], axis=0),  # (num_heads, T, T)
        "out": out[0].cpu().numpy(),
    }
    model.train()
    return snap

@torch.no_grad()
def get_multihead_snapshot_from_X(model, X, itos):
    model.eval()
    B, T = X.shape
    chars = [itos[i.item()] for i in X[0]]

    token_emb = model.token_embedding(X)                          # (B,T,C)
    pos = torch.arange(T, device=X.device) % model.block_size
    pos_emb = model.position_embedding_table(pos)                 # (T,C)
    x = token_emb + pos_emb                                       # (B,T,C)

    q_all, k_all, v_all = [], [], []
    wei_all, out_all = [], []

    Wq_all, Wk_all, Wv_all = [], [], []

    for h in model.sa_heads.heads:
        q = h.query(x)
        k = h.key(x)
        v = h.value(x)
        out, wei = h(x)

        q_all.append(q[0].cpu().numpy())
        k_all.append(k[0].cpu().numpy())
        v_all.append(v[0].cpu().numpy())
        out_all.append(out[0].cpu().numpy())
        wei_all.append(wei[0].cpu().numpy())

        Wq_all.append(h.query.weight.cpu().numpy())
        Wk_all.append(h.key.weight.cpu().numpy())
        Wv_all.append(h.value.weight.cpu().numpy())

    snap = {
        "chars": chars,
        "token_emb": token_emb[0].cpu().numpy(),
        "pos_emb": pos_emb.cpu().numpy(),
        "x": x[0].cpu().numpy(),
        "q": np.stack(q_all, axis=0),
        "k": np.stack(k_all, axis=0),
        "v": np.stack(v_all, axis=0),
        "out": np.stack(out_all, axis=0),
        "wei": np.stack(wei_all, axis=0),
        "W_Q": np.stack(Wq_all, axis=0).transpose(0, 2, 1),
        "W_K": np.stack(Wk_all, axis=0).transpose(0, 2, 1),
        "W_V": np.stack(Wv_all, axis=0).transpose(0, 2, 1),
    }

    model.train()
    return snap


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def estimate_loss(model, train_sequences, val_sequences, block_size, batch_size, eval_iterations):
    """Average loss on 'train' and 'validation' splits."""
    out = {}
    model.eval()

    for split_name, sequences in [("train", train_sequences), ("validation", val_sequences)]:
        losses = torch.zeros(eval_iterations)
        for i in range(eval_iterations):
            X, Y = get_batch_from_sequences(sequences, block_size, batch_size)
            _, loss = model(X, Y)
            losses[i] = loss.item()
        out[split_name] = losses.mean()

    model.train()
    return out


def estimate_rule_error(model, generator, decode, block_size, num_samples=20, seq_length=30):
    """Generate sequences and check rule error.
    Returns the fraction of CONSTRAINED positions that violate the rule.
    """
    model.eval()

    total_constrained = 0
    incorrect_constrained = 0

    vocab_size = model.token_embedding.weight.shape[0]

    for _ in range(num_samples):
        start_token = random.randint(0, vocab_size - 1)
        start = torch.tensor([[start_token]], dtype=torch.long)
        sample = model.generate(start, max_new_tokens=seq_length - 1)[0].tolist()
        generated_integers = decode(sample)

        correctness, _ = generator.verify_sequence(generated_integers)
        valence = generator.valence_mask(generated_integers)
        for i, is_constrained in enumerate(valence):
            if i < len(correctness) and is_constrained:
                total_constrained += 1
                if correctness[i] == 0:
                    incorrect_constrained += 1

    model.train()
    return incorrect_constrained / total_constrained if total_constrained > 0 else 0.0
