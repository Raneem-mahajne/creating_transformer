"""New poster folder from poster/poster_transformer.pdf figures.

Does not modify poster/*.png, poster/no-sequence.html, or plotting/*.py.

Architecture (fig01-tight.png) is copied from the current poster and never
regenerated. Other figures are regenerated only so plot titles match Figure 3:
fontsize 13, fontweight normal.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


THIS_DIR = Path(__file__).resolve().parent
POSTER_DIR = THIS_DIR.parent
ROOT = POSTER_DIR.parent
FIG_DIR = THIS_DIR / "figures"
ORIG_DIR = THIS_DIR / "figures_original"
TMP_SUBFOLDER = "_structured_from_transformer_tmp"

TITLE_FS = 13
TITLE_WEIGHT = "normal"

# Exact files used by poster/poster_transformer.pdf (poster/no-sequence.html).
USED_POSTER_FIGURES = [
    "fig01-tight.png",
    "fig02.png",
    "fig03.png",
    "fig04.png",
    "fig05.png",
    "fig06.png",
    "fig07.png",
    "fig08.png",
    "fig09.png",
    "fig10.png",
]

GENERATED_TO_POSTER = {
    "02_training_data.png": "fig02.png",
    "03_learning_curve.png": "fig03.png",
    "04_generated_sequences.png": "fig04.png",
    "05_token_embeddings.png": "fig05.png",
    "08_qkv_transforms.png": "fig06.png",
    "09_10_qk_space_combined.png": "fig07.png",
    "11_1_qk_full_heatmap_last_row.png": "fig08.png",
    "06_output_probs.png": "fig09.png",
    "07_output_landscape_summary.png": "fig10.png",
}


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _is_inset_title(label) -> bool:
    s = str(label)
    return s.startswith("$W_") or s.startswith(r"$W_")


def restyle_titles_on_figure(fig: Figure) -> None:
    """Force every plot/subplot title to the learning-curve style."""
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_fontweight(TITLE_WEIGHT)
        fig._suptitle.set_fontsize(TITLE_FS)
    for ax in fig.axes:
        title = ax.get_title()
        if not title or _is_inset_title(title):
            continue
        ax.title.set_fontweight(TITLE_WEIGHT)
        ax.title.set_fontsize(TITLE_FS)


def copy_current_poster_assets() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ORIG_DIR.mkdir(parents=True, exist_ok=True)

    for name in USED_POSTER_FIGURES:
        shutil.copy2(POSTER_DIR / name, ORIG_DIR / name)
        shutil.copy2(POSTER_DIR / name, FIG_DIR / name)

    _copy_tree(POSTER_DIR / "logos", THIS_DIR / "logos")
    _copy_tree(POSTER_DIR / "vendor", THIS_DIR / "vendor")
    shutil.copy2(POSTER_DIR / "qr.png", THIS_DIR / "qr.png")

    html = (POSTER_DIR / "no-sequence.html").read_text(encoding="utf-8")
    for name in USED_POSTER_FIGURES:
        html = html.replace(f'src="{name}"', f'src="figures/{name}"')
    (THIS_DIR / "index.html").write_text(html, encoding="utf-8")


def install_learning_curve_title_style() -> None:
    orig_axes_set_title = Axes.set_title
    orig_fig_suptitle = Figure.suptitle
    orig_plt_title = plt.title
    orig_fig_savefig = Figure.savefig
    orig_plt_savefig = plt.savefig

    def style_kwargs(label, kwargs: dict) -> dict:
        styled = dict(kwargs)
        styled.pop("weight", None)
        styled["fontweight"] = TITLE_WEIGHT
        if not _is_inset_title(label):
            styled["fontsize"] = TITLE_FS
        return styled

    def style_fontdict(label, fontdict):
        if not fontdict:
            return fontdict
        fd = dict(fontdict)
        fd.pop("weight", None)
        fd["fontweight"] = TITLE_WEIGHT
        if not _is_inset_title(label):
            fd["fontsize"] = TITLE_FS
            fd["size"] = TITLE_FS
        return fd

    def set_title(self, label, fontdict=None, loc=None, pad=None, *, y=None, **kwargs):
        return orig_axes_set_title(
            self,
            label,
            fontdict=style_fontdict(label, fontdict),
            loc=loc,
            pad=pad,
            y=y,
            **style_kwargs(label, kwargs),
        )

    def suptitle(self, t, **kwargs):
        if "fontdict" in kwargs:
            kwargs["fontdict"] = style_fontdict(t, kwargs["fontdict"])
        return orig_fig_suptitle(self, t, **style_kwargs(t, kwargs))

    def pyplot_title(*args, **kwargs):
        label = args[0] if args else kwargs.get("label", "")
        return orig_plt_title(*args, **style_kwargs(label, kwargs))

    def fig_savefig(self, *args, **kwargs):
        restyle_titles_on_figure(self)
        return orig_fig_savefig(self, *args, **kwargs)

    def pyplot_savefig(*args, **kwargs):
        fig = plt.gcf()
        restyle_titles_on_figure(fig)
        return orig_plt_savefig(*args, **kwargs)

    Axes.set_title = set_title
    Figure.suptitle = suptitle
    plt.title = pyplot_title
    Figure.savefig = fig_savefig
    plt.savefig = pyplot_savefig


def regenerate_structured_figures() -> None:
    """Regenerate non-architecture figures only. Architecture stays the poster PNG."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from checkpoint import get_plots_dir, load_checkpoint
    from config_loader import load_config
    from plotting._utils import clear_journal_mode, set_journal_mode
    from visualize import visualize_from_checkpoint

    checkpoint_data = load_checkpoint("plus_last_even")
    if checkpoint_data is None:
        raise SystemExit("No plus_last_even checkpoint found")
    config = load_config("plus_last_even")

    tmp_dir = get_plots_dir("plus_last_even", subfolder=TMP_SUBFOLDER)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    set_journal_mode(max_width=7.0, max_height=9.5, dpi=300)
    install_learning_curve_title_style()
    try:
        visualize_from_checkpoint(
            "plus_last_even",
            checkpoint_data,
            config,
            step=None,
            plots_subfolder=TMP_SUBFOLDER,
            generate_journal=False,
            _is_journal_pass=True,
            only_figures=[2, 3, 4, 5, 6, 7, 8, 9, 11],
        )
    finally:
        clear_journal_mode()

    for generated_name, poster_name in GENERATED_TO_POSTER.items():
        src = tmp_dir / generated_name
        if not src.exists():
            raise FileNotFoundError(f"Expected generated figure missing: {src}")
        shutil.copy2(src, FIG_DIR / poster_name)
        print(f"structured {generated_name} -> figures/{poster_name}")

    # Restore architecture from the poster after everything else.
    shutil.copy2(ORIG_DIR / "fig01-tight.png", FIG_DIR / "fig01-tight.png")
    print("kept poster architecture: figures/fig01-tight.png")

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)


def main() -> None:
    copy_current_poster_assets()
    regenerate_structured_figures()
    print(f"Done. New poster folder: {THIS_DIR}")


if __name__ == "__main__":
    main()
