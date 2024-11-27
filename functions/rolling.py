from typing import Any

import numpy as np
import pandas as pd
from river.utils.rolling import Rolling as R


def separate_args_kwargs(list_of_tuples):
    """Separate args and kwargs from a list of tuples.

    Imagine transposing a list of tuples. This function separates list of args and kwargs tuples and makes args and kwargs containing most compatible iterables.

    Examples:
        >>> separate_args_kwargs(
        ...     [((1, 2), {'x': 3, 'y': 4}), ((5, 6), {'x': 8, 'y': 9})])
        ([[1, 5], [2, 6]], {'x': [3, 8], 'y': [4, 9]})
        >>> separate_args_kwargs(
        ...     [((np.array([1, 2 ]),), {'x': np.array([3,  4]), 'y': np.array([5,  6])}),
        ...     ((np.array([ 2, 3]),), {'x': np.array([4, 5]), 'y': np.array([6 , 7])})])
        ([array([[1, 2],
            [2, 3]])], {'x': array([[3, 4],
            [4, 5]]), 'y': array([[5, 6],
            [6, 7]])})
        >>> separate_args_kwargs(
        ...     [(({'a': 1, 'b': 2},), {'x': {'a': 3, 'b': 4}, 'y': {'a': 5, 'b': 6}}),
        ...     (({'a': 7, 'b': 8},), {'x': {'a': 7, 'b': 8}, 'y': {'a': 7, 'b': 8}})])
        ([   a  b
        0  1  2
        1  7  8], {'x':    a  b
        0  3  4
        1  7  8, 'y':    a  b
        0  5  6
        1  7  8})
    """

    args_: list[Any] = []
    kwargs_: dict[str | int, Any] = {}
    args_types = []
    kwargs_types: dict[str | int, type] = {}
    # Extracting args
    for tpl in list_of_tuples:
        # Infer the types based on first tpl
        if not args_:
            args_types = [type(arg) for arg in tpl[0]]
            kwargs_types = {k: type(v) for k, v in tpl[1].items()}
        args, kwargs = tpl
        args_.append(list(args))  # Append args as a list

        # Extracting kwargs
        for key, value in kwargs.items():
            if key not in kwargs_:
                kwargs_[key] = [value]
            else:
                kwargs_[key].append(value)

    args_ = list(map(list, zip(*args_)))

    args__: list[np.ndarray | pd.DataFrame] = []
    for i, arg in enumerate(args_):
        if issubclass(args_types[i], np.ndarray):
            args__.append(np.array(arg))
        elif issubclass(args_types[i], dict | pd.Series):
            args__.append(pd.DataFrame(arg, index=range(len(arg))))
        else:
            args__.append(arg)

    kwargs__: dict[str | int, np.ndarray | pd.DataFrame] = {}
    for k, kwarg in kwargs_.items():
        if issubclass(kwargs_types[k], np.ndarray):
            kwargs__[k] = np.array(kwarg)
        elif issubclass(kwargs_types[k], dict | pd.Series):
            kwargs__[k] = pd.DataFrame(kwarg, index=range(len(kwarg)))
        else:
            kwargs__[k] = kwarg

    return args__, kwargs__


class Rolling(R):
    """A generic wrapper for performing rolling computations.

    This can be wrapped around any object which implements both an `update` and a `revert` method.
    Inputs to `update` are stored in a queue. Elements of the queue are popped when the window is
    full.

    Args:
        obj: An object that implements both an `update` method and a `rolling` method.
        window_size: Size of the window.

    Examples:
        For instance, here is how you can compute a rolling average over a window of size 3:

        >>> np.random.seed(0)
        >>> r = 1
        >>> m = 2
        >>> n = 5
        >>> X = pd.DataFrame(np.linalg.qr(np.random.randn(n, m))[0])
        >>> from river.decomposition import OnlineSVD

        >>> svd = Rolling(OnlineSVD(n_components=r, initialize=5, seed=0), window_size=5)
        >>> rsvd = Rolling(OnlineSVD(n_components=r, initialize=5, seed=0), window_size=5)

        >>> for x in X.to_dict(orient='records'):
        ...     rsvd.update(x=x)
        >>> svd.update_many(x=X)
        >>> svd.n_seen == rsvd.n_seen
        True
        >>> np.abs(svd.transform_one(x)[0]) == np.abs(rsvd.transform_one(x)[0])
        True
        >>> X = pd.DataFrame(np.linalg.qr(np.random.randn(2, m))[0])
        >>> for x in X.to_dict(orient='records'):
        ...     rsvd.update(x=x)
        >>> svd.update_many(x=X)
        >>> np.abs(svd.transform_one(x)[0]) == np.abs(rsvd.transform_one(x)[0])
        True
    """

    def __init__(self, obj, window_size):
        super().__init__(obj, window_size)

    def learn_one(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def update_many(self, *args, **kwargs):
        # First arg defines the number of samples to update
        n_update = (
            len(args[0]) if len(args) > 0 else len(list(kwargs.values())[0])
        )
        n_revert = len(self.window) + n_update - self.window_size
        if n_revert > 0:
            args_kwargs = [self.window[i] for i in range(n_revert)]
            args_old, kwargs_old = separate_args_kwargs(args_kwargs)
            if hasattr(self.obj, "revert_many"):
                self.obj.revert_many(*args_old, **kwargs_old)  # type: ignore
            else:
                # In this case revert should support multiple samples
                self.obj.revert(*args_old, **kwargs_old)
        if hasattr(self.obj, "update_many"):
            self.obj.update_many(*args, **kwargs)  # type: ignore
        else:
            # In this case update should support multiple samples
            self.obj.update(*args, **kwargs)

        # Check if all the lengths are the same
        # TODO: for args and kwargs with different length (should be always 1)
        #  we need to append it to every sample. This is the case of weights...
        args_ids = [
            i
            for i, arg in enumerate(args)
            if hasattr(arg, "__len__") and len(arg) == n_update
        ]
        kwargs_ids = [
            k
            for k, kwarg in kwargs.items()
            if hasattr(kwarg, "__len__") and len(kwarg) == n_update
        ]

        for i in range(n_update):
            sample = (
                tuple(
                    [
                        args[idx][i]
                        if not isinstance(args[idx], pd.DataFrame)
                        else args[idx].iloc[i]
                        for idx in args_ids
                    ]
                ),
                {
                    k: kwargs[k][i]
                    if not isinstance(kwargs[k], pd.DataFrame)
                    else kwargs[k].iloc[i]
                    for k in kwargs_ids
                },
            )
            self.window.append(sample)

    def learn_many(self, *args, **kwargs):
        self.update_many(*args, **kwargs)
