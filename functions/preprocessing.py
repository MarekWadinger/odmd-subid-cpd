import numpy as np


def normalize(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


def hankel(
    X: np.ndarray,
    hn: int,
    cut_rollover: bool = True,
) -> np.ndarray:
    """Create a Hankel matrix from a given input array.

    Args:
        X (np.ndarray): The input array.
        hn (int): The number of columns in the Hankel matrix.
        cut_rollover (bool, optional): Whether to cut the rollover part of the Hankel matrix. Defaults to True.

    Returns:
        np.ndarray: The Hankel matrix.

    TODO:
        - [ ] Add support for 2D arrays.

    Example:
    >>> X = np.array([1., 2., 3., 4., 5.])
    >>> hankel(X, 3, cut_rollover=False)
    array([[1., 2., 3.],
           [2., 3., 4.],
           [3., 4., 5.],
           [4., 5., 1.],
           [5., 1., 2.]])
    >>> hankel(X, 3)
    array([[1., 2., 3.],
           [2., 3., 4.],
           [3., 4., 5.]])
    >>> X = np.array([[1., 2., 3., 4., 5.], [9., 8., 7., 6., 5.]]).T
    >>> hankel(X, 3, cut_rollover=False)
    array([[1., 9., 2., 8., 3., 7.],
           [2., 8., 3., 7., 4., 6.],
           [3., 7., 4., 6., 5., 5.],
           [4., 6., 5., 5., 1., 9.],
           [5., 5., 1., 9., 2., 8.]])
    """
    if len(X.shape) > 1:
        n = X.shape[1]
    else:
        n = 1
    if hn <= 1:
        return X
    hX = np.empty((X.shape[0], hn * n))
    for i in range(0, hn * n, n):
        hX[:, i : i + n] = X if len(X.shape) > 1 else X.reshape(-1, 1)
        X = np.roll(X, -1, axis=0)
    if cut_rollover:
        hX = hX[: -hn + 1]
    return hX
