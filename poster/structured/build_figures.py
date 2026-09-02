"""Regenerate poster figures with learning-curve title style into this folder.

Does not modify plotting/*.py, plus_last_even/plots/a4/, or poster/*.html.
Titles: fontsize 13, fontweight normal (same as the learning-curve figure).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
TMP_SUBFOLDER = "_poster_structured_tmp"

TITLE_FS = 13
TITLE_WEIGHT = "normal"

POSTER_MAP = {
    "01_architecture_overview.png": "fig01.png",
    "02_training_data.png": "fig02.png",
    "03_learning_curve.png": "fig03.png",
    "04_generated_sequences.png": "fig04.png",
    "05_token_embeddings.png": "fig05.png",
    "08_qkv_transforms.png": "fig06.png",
    "09_10_qk_space_combined.png": "fig07.png",
    "11_1_qk_full_heatmap_last_row.png": "fig08.png",
    "06_output_probs.png": "fig09.png",
    "07_output_landscape_summary.png": "fig10.png",
    "13_sequence_embeddings.png": "fig11.png",
    "14_qk_attention.png": "fig12.png",
    "15_q_dot_product_gradients.png": "fig13.png",
    "16_value_output.png": "fig14.png",
    "17_residuals.png": "fig15.png",
}


def _is_inset_title(label) -> bool:
    s = str(label)
    return s.startswith("$W_") or s.startswith(r"$W_")


def _style_title_kwargs(label, kwargs: dict) -> dict:
    kwargs = dict(kwargs)
    kwargs.pop("weight", None)
    kwargs["fontweight"] = TITLE_WEIGHT
    if not _is_inset_title(label):
        kwargs["fontsize"] = TITLE_FS
    return kwargs


def _style_fontdict(label, fontdict):
    if not fontdict:
        return fontdict
    fd = dict(fontdict)
    fd.pop("weight", None)
    fd["fontweight"] = TITLE_WEIGHT
    if not _is_inset_title(label):
        fd["fontsize"] = TITLE_FS
        fd["size"] = TITLE_FS
    return fd


def install_title_style():
    orig_axes_set_title = Axes.set_title
    orig_fig_suptitle = Figure.suptitle
    orig_plt_title = plt.title

    def set_title(self, label, fontdict=None, loc=None, pad=None, *, y=None, **kwargs):
        kwargs = _style_title_kwargs(label, kwargs)
        fontdict = _style_fontdict(label, fontdict)
        return orig_axes_set_title(self, label, fontdict=fontdict, loc=loc, pad=pad, y=y, **kwargs)

    def suptitle(self, t, **kwargs):
        kwargs = _style_title_kwargs(t, kwargs)
        if "fontdict" in kwargs:
            kwargs["fontdict"] = _style_fontdict(t, kwargs["fontdict"])
        return orig_fig_suptitle(self, t, **kwargs)

    def pyplot_title(*args, **kwargs):
        label = args[0] if args else kwargs.get("label", "")
        kwargs = _style_title_kwargs(label, kwargs)
        return orig_plt_title(*args, **kwargs)

    Axes.set_title = set_title
    Figure.suptitle = suptitle
    plt.title = pyplot_title

    import matplotlib.figure as mfig

    if hasattr(mfig, "FigureBase"):
        orig_text = mfig.FigureBase.text

        def fig_text(self, x, y, s, **kwargs):
            if kwargs.get("ha") == "center" and kwargs.get("fontweight") in ("bold", "heavy"):
                kwargs = _style_title_kwargs(s, kwargs)
            return orig_text(self, x, y, s, **kwargs)

        mfig.FigureBase.text = fig_text


def wrap_architecture_title():
    """Give the architecture diagram the same figure-title style as the learning curve."""
    import visualize as vis

    orig = vis.plot_architecture_diagram

    def titled(config, save_path=None, **kwargs):
        orig_save = plt.savefig

        def save_with_title(*a, **k):
            fig = plt.gcf()
            fig.suptitle(
                "Architecture of the Minimal Transformer",
                fontsize=TITLE_FS,
                fontweight=TITLE_WEIGHT,
                y=1.0,
            )
            return orig_save(*a, **k)

        plt.savefig = save_with_title
        try:
            return orig(config, save_path=save_path, **kwargs)
        finally:
            plt.savefig = orig_save

    vis.plot_architecture_diagram = titled


def crop_tight(src: Path, dst: Path, pad: int = 24, white_thr: int = 248):
    im = Image.open(src).convert("RGB")
    a = np.array(im)
    ink = a.min(axis=2) < white_thr
    ys, xs = np.where(ink)
    if len(xs) == 0:
        im.save(dst)
        return
    box = (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(im.size[0], int(xs.max()) + 1 + pad),
        min(im.size[1], int(ys.max()) + 1 + pad),
    )
    im.crop(box).save(dst, optimize=True)


def copy_assets():
    src_poster = ROOT / "poster"
    for name in ("logos", "vendor"):
        dst = OUT_DIR / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src_poster / name, dst)
    shutil.copy2(src_poster / "qr.png", OUT_DIR / "qr.png")

    def retarget(html_name: str):
        text = (src_poster / html_name).read_text(encoding="utf-8")
        for i in range(1, 16):
            text = text.replace(f'src="fig{i:02d}.png"', f'src="figures/fig{i:02d}.png"')
        text = text.replace('src="fig01-tight.png"', 'src="figures/fig01-tight.png"')
        (OUT_DIR / html_name).write_text(text, encoding="utf-8")

    retarget("no-sequence.html")
    retarget("index.html")


def main():
    install_title_style()

    from checkpoint import load_checkpoint, get_plots_dir
    from config_loader import load_config
    from plotting._utils import set_journal_mode, clear_journal_mode
    from visualize import visualize_from_checkpoint

    wrap_architecture_title()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    copy_assets()

    config = load_config("plus_last_even")
    checkpoint_data = load_checkpoint("plus_last_even")
    if checkpoint_data is None:
        raise SystemExit("No plus_last_even checkpoint found")

    tmp_dir = get_plots_dir("plus_last_even", subfolder=TMP_SUBFOLDER)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    set_journal_mode(max_width=7.0, max_height=9.5, dpi=300)
    try:
        visualize_from_checkpoint(
            "plus_last_even",
            checkpoint_data,
            config,
            step=None,
            plots_subfolder=TMP_SUBFOLDER,
            generate_journal=False,
            _is_journal_pass=True,
            only_figures=[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17],
        )
    finally:
        clear_journal_mode()

    missing = []
    for src_name, dst_name in POSTER_MAP.items():
        src = tmp_dir / src_name
        if not src.exists():
            missing.append(src_name)
            continue
        shutil.copy2(src, FIG_DIR / dst_name)
        print(f"copied {src_name} -> figures/{dst_name}")

    if missing:
        print("MISSING:", missing)

    fig01 = FIG_DIR / "fig01.png"
    if fig01.exists():
        crop_tight(fig01, FIG_DIR / "fig01-tight.png")
        print("wrote figures/fig01-tight.png")

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        print(f"removed temp plots dir {tmp_dir}")

    print("done:", FIG_DIR)


if __name__ == "__main__":
    main()
