from typing import Literal

import numpy as np
import pandas as pd
from river.preprocessing import Hankelizer


class Hankelizer(Hankelizer):
    """Mini-batch Hankelizer that keeps track of the transformation.

    Similar to the original Hankelizer, it
    The _memory_usage differs from the original Hankelizer due to storing the
    transformation track which can be significant for large datasets.

    Examples:
    Using Mini-batch Hankelizer is equivalent to
    >>> import numpy as np
    >>> import pandas as pd
    >>> from river.preprocessing import Hankelizer as H
    >>> X_train = pd.DataFrame(np.random.rand(10, 2), columns=["a", "b"])
    >>> hn = 3
    >>> hankelizer = Hankelizer(hn)
    >>> hankelizer_old = H(hn)

    >>> hankelizer.learn_many(X_train)
    >>> X_t_new = hankelizer.transform_many(X_train)
    >>> X_t_old_ = []
    >>> for j, x in enumerate(X_train.to_dict(orient="records")):
    ...     hankelizer_old.learn_one(x)
    ...     X_t_old_.append(hankelizer_old.transform_one(x))
    >>> X_t_old = pd.DataFrame(X_t_old_)
    >>> X_t_new.equals(X_t_old)
    True

    Calling transform_many first produces consistent behavior
    >>> hankelizer = Hankelizer(hn)
    >>> X_t_first = hankelizer.transform_many(X_train)
    >>> X_t_first.equals(X_t_new)
    True
    """

    def __init__(
        self, w: int = 2, return_partial: bool | Literal["copy"] = "copy"
    ):
        super().__init__(w, return_partial)
        self.transform_track: list[dict] = []

    def learn_many(self, X: pd.DataFrame):
        self.transform_track = []
        for x in X.to_dict(orient="records"):
            self.learn_one(x)
            self.transform_track.append(self.transform_one(x))

    def transform_many(self, X: pd.DataFrame):
        # if transform_track is empty, it means that the transform is called first
        # so we need to learn the data first and reset the state
        if not self.transform_track:
            self.learn_many(X)
            self._window.clear()
        df = pd.DataFrame(self.transform_track)
        self.transform_track = []
        return df


def normalize(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


def hankel(
    X: np.ndarray | pd.DataFrame,
    hn: int,
    step: int = 1,
    return_partial: bool | Literal["copy"] = "copy",
) -> np.ndarray | pd.DataFrame:
    """Create a Hankel matrix from a given input array.

    Args:
        X (np.ndarray): The input array.
        hn (int): The number of columns in the Hankel matrix.
        step (int, optional): The step size for the delays. Defaults to 1.
        cut_rollover (bool, optional): Whether to cut the rollover part of the Hankel matrix. Defaults to True.

    Returns:
        np.ndarray: The Hankel matrix.

    TODO:
        - [ ] Add support for 2D arrays.

    Example:
    >>> X = np.array([1., 2., 3., 4., 5.])
    >>> hankel(X, 3)
    array([[1., 1., 1.],
           [1., 1., 2.],
           [1., 2., 3.],
           [2., 3., 4.],
           [3., 4., 5.]])
    >>> hankel(X, 3, return_partial=False)
    array([[1., 2., 3.],
           [2., 3., 4.],
           [3., 4., 5.]])
    >>> X = np.array([[1., 2., 3., 4., 5.], [9., 8., 7., 6., 5.]]).T
    >>> hankel(X, 3, return_partial=True)
    array([[nan, nan, nan, nan,  1.,  9.],
           [nan, nan,  1.,  9.,  2.,  8.],
           [ 1.,  9.,  2.,  8.,  3.,  7.],
           [ 2.,  8.,  3.,  7.,  4.,  6.],
           [ 3.,  7.,  4.,  6.,  5.,  5.]])
    >>> X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [9.0, 8.0, 7.0, 6.0, 5.0]]).T
    >>> hankel(X, 3, 2, return_partial=True)
    array([[nan, nan, nan, nan,  3.,  7.],
           [nan, nan,  2.,  8.,  4.,  6.],
           [ 1.,  9.,  3.,  7.,  5.,  5.],
           [ 2.,  8.,  4.,  6.,  1.,  9.],
           [ 3.,  7.,  5.,  5.,  2.,  8.]])
    """
    if hn <= 1:
        return X

    if isinstance(X, pd.DataFrame):
        feature_names_in_ = X.columns
        index_in_ = X.index
        X = X.values
    else:
        feature_names_in_ = None

    if len(X.shape) > 1:
        n = X.shape[1]
    else:
        n = 1

    hX = np.empty((X.shape[0], hn * n))
    # Roll forth so that the last hankel columns are the start of the array
    X = np.roll(X, hn - 1, axis=0)
    for i in range(0, hn * n, n):
        hX[:, i : i + n] = X if len(X.shape) > 1 else X.reshape(-1, 1)
        if return_partial == "copy" and i / n < hn - 1:
            hX[: hn - int(i / n) - 1, i : i + n] = hX[
                hn - int(i / n) - 1, i : i + n
            ]
        elif return_partial and i / n < hn - 1:
            hX[: hn - int(i / n) - 1, i : i + n] = np.nan
        X = np.roll(X, -step, axis=0)
    if not return_partial:
        hX = hX[hn - 1 :]
    if feature_names_in_ is not None:
        return pd.DataFrame(
            hX,
            columns=[f"{f}_{i}" for i in range(hn) for f in feature_names_in_],
            index=index_in_,
        )
    return hX
