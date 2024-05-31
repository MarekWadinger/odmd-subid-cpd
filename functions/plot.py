from typing import Literal

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .preprocessing import normalize as _normalize

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Times New Roman",
        "font.serif": "Times New Roman",
        "axes.labelsize": 12,
        "axes.grid": True,
        "font.size": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.subplot.left": 0.1,
        "figure.subplot.bottom": 0.05,
        "figure.subplot.right": 0.95,
        "figure.subplot.top": 0.95,
        # "backend": "macOsX"
    }
)

locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(
    locator,
    formats=["%Y", "%d %b", "%d %b", "%H:%M", "%H:%M", "%S.%f"],
    offset_formats=["", "%Y", "", "", "", "%Y-%b-%d %H:%M"],
)


def set_size(
    width: float | int | Literal["article", "thesis", "beamer"] = 307.28987,
    fraction=1.0,
    subplots=(1, 1),
):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the height which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "article":
        width_pt = 390.0
    elif width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = (
        fig_width_in * golden_ratio * ((subplots[0] * fraction) / subplots[1])
    )

    return (fig_width_in * 1.2, fig_height_in * 1.2)


RED_ALPHA05 = "#ea9293"
GRAY_ALPHA025 = "#dedede"


def plot_chd(
    datas: dict[str, np.ndarray | None] | list[np.ndarray | None],
    y_true: list[float] | np.ndarray | None = None,
    labels: list[str] | None = None,
    idx_start: int | None = None,
    idx_end: int | None = None,
    ids_in_start: list[int] | None = None,
    ids_in_end: list[int] | None = None,
    grace_period: int | None = None,
    normalize: bool = False,
    axs: np.ndarray | None = None,
    **fig_kwargs: dict,
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
    fig_kwargs_ = {
        "width": "article",
        "subplots": (len(datas), 1),
        "fraction": 0.5,
    }
    fig_kwargs_.update(fig_kwargs)
    if axs is None:
        fig, axs_ = plt.subplots(
            len(datas),
            1,
            sharex="col",
            sharey="row",
            figsize=set_size(**fig_kwargs_),
        )
    else:
        axs_ = axs
        fig = axs[0].figure

    idx_start = 0 if idx_start is None else idx_start
    # idx_end = len(datas[0]) if idx_end is None else idx_end

    if labels is None:
        labels = [""] * len(datas)

    for ax, data, label in zip(axs_, datas, labels):
        if isinstance(data, str) and isinstance(datas, dict):
            name = data
            data = datas[data]
        else:
            name = ""
        if not isinstance(ax, Axes):
            return fig, axs_

        if y_true is not None:
            for i in y_true:
                ax.axvline(i, color=RED_ALPHA05)
                if grace_period:
                    ax.axvline(
                        i + grace_period, color=RED_ALPHA05, linestyle="--"
                    )
        if data is not None:
            ax.plot(data[idx_start:idx_end], label=label)
            if normalize:
                ax_norm = ax.twinx()
                ax_norm.plot(  # type: ignore
                    _normalize(data[idx_start:idx_end]),
                    label=label + " (norm)",
                )
            if name != "":
                ax.set_ylabel(name)
            if label != "":
                ax.legend()
            ax.grid(True, axis="y")

            if ids_in_start is not None and ids_in_end is not None:
                n_ins = len(ids_in_start)
                for i, (idx_in_start, idx_in_end) in enumerate(
                    zip(ids_in_start, ids_in_end)
                ):
                    x_in = range(idx_in_start, idx_in_end)
                    # Add inlay plot
                    inlay_ax = ax.inset_axes(
                        (
                            0.1 + (0.6 / n_ins * i),
                            0.4,
                            (0.6 / n_ins) - 0.05,
                            0.6,
                        )
                    )  # Adjust the position and size of the inlay plot
                    if y_true is not None:
                        for i in y_true:
                            if idx_in_start < i < idx_in_end:
                                inlay_ax.axvline(i, color=RED_ALPHA05)
                                if grace_period:
                                    inlay_ax.axvline(
                                        i + grace_period,
                                        color=RED_ALPHA05,
                                        linestyle="--",
                                    )
                    inlay_ax.plot(
                        x_in, data[idx_in_start:idx_in_end], label=label
                    )
                    inlay_ax.grid(True, axis="y")
                    inlay_ax.set_yticklabels([])
                    inlay_ax.set_xticklabels([])
                    inlay_ax.patch.set_alpha(
                        0.75
                    )  # Set the background transparency
                    ax.indicate_inset_zoom(inlay_ax, edgecolor="black")

    return fig, axs_
