import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .preprocessing import normalize as _normalize


def plot_chd(
    datas: list[np.ndarray],
    y_true: list[float] | np.ndarray | None = None,
    labels: list[str] | None = None,
    idx_start: int | None = None,
    idx_end: int | None = None,
    idx_in_start: int | None = None,
    idx_in_end: int | None = None,
    normalize: bool = False,
):
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
            ax.plot(x, data[idx_start:idx_end], label=label)
            if y_true is not None:
                for i in y_true:
                    ax.axvline(i, color="tab:red", linestyle="--")
            if normalize:
                ax_norm = ax.twinx()
                ax_norm.plot(  # type: ignore
                    _normalize(data[idx_start:idx_end]),
                    label=label + " (norm)",
                )
            ax.legend([label])
            ax.grid(True)

            if x_in:
                # Add inlay plot
                inlay_ax = ax.inset_axes(
                    (0.1, 0.4, 0.6, 0.6)
                )  # Adjust the position and size of the inlay plot
                inlay_ax.plot(x_in, data[idx_in_start:idx_in_end], label=label)
                inlay_ax.grid(True)
                inlay_ax.set_yticklabels([])
                inlay_ax.set_xticklabels([])
                ax.indicate_inset_zoom(inlay_ax, edgecolor="black")

    return fig, axs
