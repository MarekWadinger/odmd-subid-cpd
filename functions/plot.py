import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .preprocessing import normalize as _normalize


RED_ALPHA05 = "#ea9293"
GRAY_ALPHA025 = "#dedede"

def plot_chd(
    datas: list[np.ndarray],
    y_true: list[float] | np.ndarray | None = None,
    labels: list[str] | None = None,
    idx_start: int | None = None,
    idx_end: int | None = None,
    idx_in_start: int | None = None,
    idx_in_end: int | None = None,
    grace_period: int | None = None,
    normalize: bool = False,
):
    """Plot hange-Point Detection Results.

    Args:
        datas: List of data to plot. Each data is plot on a separate subplot.
        y_true: True change-point locations. Plotted as vertical lines.
        labels: List of labels for each data.
        idx_start: Starting index to plot. Plot from the beginning if None.
        idx_end: Ending index to plot. Plot till the end if None.
        idx_in_start: Starting index for inlay plot. No inlay plot if None.
        idx_in_end: Ending index for inlay plot. No inlay plot if None.
        grace_period: Grace period for change-point. Plots grayed out region where peak of detection could be expected.
        normalize: Normalize data. If False, no normalization is done.

    """
    fig, axs = plt.subplots(len(datas), 1, sharex=True)

    idx_start = 0 if idx_start is None else idx_start
    idx_end = len(datas[0]) if idx_end is None else idx_end
    x = range(idx_start, idx_end)

    if idx_in_start and idx_in_end:
        x_in = range(idx_in_start, idx_in_end)
    else:
        x_in = None

    if labels is None:
        labels = [""] * len(datas)

    for ax, data, label in zip(axs, datas, labels):
        if isinstance(ax, Axes):
            if y_true is not None:
                for i in y_true:
                    ax.axvline(i, color=RED_ALPHA05)
                    if grace_period:
                        ax.axvline(i + grace_period, color=GRAY_ALPHA025)
                        # ax.add_patch(patches.Rectangle(
                        #         (i, 0),
                        #         grace_period,
                        #         data.max(),
                        #         linewidth=0,
                        #         facecolor=GRAY_ALPHA025,
                        #     ))
            ax.plot(x, data[idx_start:idx_end], label=label)
            if normalize:
                ax_norm = ax.twinx()
                ax_norm.plot(  # type: ignore
                    _normalize(data[idx_start:idx_end]),
                    label=label + " (norm)",
                )
            ax.legend([label])
            ax.grid(True, axis="y")

            if x_in and idx_in_start and idx_in_end:
                # Add inlay plot
                inlay_ax = ax.inset_axes(
                    (0.1, 0.4, 0.6, 0.6)
                )  # Adjust the position and size of the inlay plot
                if y_true is not None:
                    for i in y_true:
                        if idx_in_start < i < idx_in_end:
                            inlay_ax.axvline(i, color=RED_ALPHA05)
                            if grace_period:
                                inlay_ax.axvline(
                                    i + grace_period, color=GRAY_ALPHA025
                                )
                inlay_ax.plot(x_in, data[idx_in_start:idx_in_end], label=label)
                inlay_ax.grid(True, axis="y")
                inlay_ax.set_yticklabels([])
                inlay_ax.set_xticklabels([])
                ax.indicate_inset_zoom(inlay_ax, edgecolor="black")

    return fig, axs
