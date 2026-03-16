from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

def demo_grid_spec():
    fig = plt.figure(figsize=(10, 10))
    # Adjust the GridSpec to use height_ratios for rows
    # Example: Make first two rows shorter, bottom panels taller
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 2, 2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax_combined = fig.add_subplot(gs[2:4, 0:2])

    for ax in [ax1, ax2, ax3, ax4, ax_combined]:
        plt.scatter(np.random.randn(100), np.random.randn(100), alpha=0.5)

    plt.show()

demo_grid_spec()