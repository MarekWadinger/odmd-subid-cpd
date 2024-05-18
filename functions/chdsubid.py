"""Change Detection based on Subspace Identification algorithm."""

from collections import deque

import numpy as np
import pandas as pd
from river.anomaly.base import AnomalyDetector
from river.base import MiniBatchTransformer, Transformer
from river.decomposition import OnlineDMD, OnlineDMDwC

from .rolling import Rolling


# # Default parameters
def get_default_rank(X):
    """Get default rank for the given data matrix

    Args:
        X (np.ndarray): Data matrix

    Returns:
        int: Default rank

    References:
        [1] Gavish, M., and Donoho L. D. (2014). The Optimal Hard Threshold for Singular Values is 4/sqrt(3). IEEE Transactions on Information Theory 60.8 (2014): 5040-5053. doi:[10.1109/TIT.2014.2323359](https://doi.org/10.1109/TIT.2014.2323359).
    """
    s = np.linalg.svd(X.T, compute_uv=False)
    beta = X.shape[1] / X.shape[0]
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    r = sum(s > omega * np.median(s))
    return r


def get_default_params(X, window_size: int, max_rank=10):
    """Get default parameters for the given dataset and window size
    Args:
        X (np.ndarray): Data matrix
        window_size (int): Window size. What kind of structural changes are we looking for?

    References:
        [2] Moskvina, V., & Zhigljavsky, A. (2003). An Algorithm Based on Singular Spectrum Analysis for Change-Point Detection. Communications in Statistics - Simulation and Computation, 32(2), 319-352. doi:[10.1081/SAC-120017494](https://doi.org/10.1081/SAC-120017494).
    """
    # If window_size is not very large, then take half
    hn = window_size // 2
    # Base size
    ref_size = window_size
    test_size = window_size
    # Optimal low-rank representation of signal with unknown noise variance
    if hn * X.shape[1] < 100:
        r = min(get_default_rank(X), max_rank)
    else:
        r = max_rank
    return hn, ref_size, test_size, r


class SubIDChangeDetector(AnomalyDetector):
    def __init__(
        self,
        subid: MiniBatchTransformer | Transformer | Rolling,
        ref_size: int,
        test_size: int | None = None,
        threshold: float = 0.25,
        time_lag: int = 0,
        grace_period: int = 0,
        learn_after_grace: bool = True,
    ):
        self.subid = subid
        self.threshold = threshold
        if ref_size == 0 and isinstance(subid, Rolling):
            ref_size = subid.window_size  # type: ignore
            # Since window_size is maxlen of deque in Rolling it may be None
            if ref_size is None:
                raise ValueError(
                    "window_size must be provided for Rolling subid"
                )
        self.ref_size = ref_size
        self.test_size = test_size if test_size is not None else ref_size
        self.time_lag = time_lag
        assert self.ref_size > 0
        assert self.test_size > 0
        assert self.test_size + self.time_lag >= 0
        self.learn_delay = self.time_lag + self.test_size
        # assert self.grace_period < self.test_size
        # TODO: basically grace period should be omitted and detection start once Transformer is fitted
        self.grace_period = grace_period
        self.learn_after_grace = learn_after_grace
        self.n_seen = 0

        self._score: float | None = None
        self._distances: tuple[complex, complex] | None = None
        self._drift_detected: bool | None = None

        self._X: deque[dict] = deque(
            maxlen=self.ref_size + self.time_lag + self.test_size
        )

    @property
    def distances(self) -> tuple[complex, complex]:
        if self._distances is None:
            # Do inference after grace period and enough data is available
            lenght_X = len(self._X)
            if (
                self.n_seen >= self.grace_period
                and lenght_X > self.learn_delay
            ):
                X = pd.DataFrame(self._X)
                X_p = self._transform_many(X)
                D_train = self._compute_distance(
                    X.iloc[: -self.learn_delay, :],
                    X_p.iloc[: -self.learn_delay, :],
                ) / (lenght_X - self.learn_delay)
                D_test = (
                    self._compute_distance(
                        X.iloc[-self.test_size :, :],
                        X_p.iloc[-self.test_size :, :],
                    )
                    / self.test_size
                )
                distances = (D_train, D_test)
            else:
                distances = (1.0 + 0.0j, 1.0 + 0.0j)
            self._distances = distances
        else:
            distances = self._distances
        return distances

    @property
    def drift_detected(self) -> bool:
        if self._drift_detected is None:
            drift_detected = self.score > self.threshold
            self._drift_detected = drift_detected
        else:
            drift_detected = self._drift_detected
        return drift_detected

    @property
    def score(self) -> float:
        if self._score is None:
            # Under some circumstances score < 0
            #  - lower test noise
            #  - running normalization
            #  - ...
            D_train, D_test = self.distances
            score_ = (D_test / D_train) - 1
            # TODO: explore interesting scoring option
            # score = D_train - D_test
            # TODO: figure out proper way of utilizing imaginary part of score
            if isinstance(score_, complex):
                score: float = score_.real + np.abs(score_.imag)
            # TODO: comment on score shawing
            score = max(score, 0.0)
            self._score = score
        else:
            score = self._score
        return score

    @property
    def _supervised(self):
        """Indicates whether or not the estimator is supervised or not.

        This is useful internally for determining if an estimator expects to be provided with a `y`
        value in it's `learn_one` method. For instance we use this in a pipeline to know whether or
        not we should pass `y` to an estimator or not.

        """
        return False

    def _compute_distance(self, X: pd.DataFrame, X_t: pd.DataFrame) -> complex:
        """Compute the distance between the data matrix and its transformation.

        This formulation computes a measure of how much information in the dataset represented by Y is preserved or retained when projected onto the space spanned by W. The difference between the covariance matrix of Y and the projected version is computed, and the sum of all elements in this difference matrix gives an overall measure of dissimilarity or distortion.

        Args:
            X: data matrix
            X_t: Transformed data matrix

        Returns:
            Distance between the data matrix and its transformation.
        """
        # Project the transformed data to the original space
        #  Similar scores are obtained combining this step with 2 norm and without projection and differencing covariances. Latter is less expensive
        # if hasattr(self.subid, "modes"):
        #     X_p = X_t @ self.subid.modes.T
        # elif hasattr(self.subid, "_U"):
        #     X_p = X_t @ self.subid._U.T
        # Opt 1: Using Frobenius norm
        # Q = float(
        #     np.linalg.norm(np.inner(X, X) - np.inner(X_p, X_p), ord="fro")
        # )
        # return Q
        # Opt 2: Using squared L1 norm (Kawahara 2007)
        XX = np.sum(X.values**2)
        # # XX = np.linalg.norm(X, 1)
        # # # Using following normalization changes the score baseline based on
        # # #  the proportion of ref and test size
        XX_std = 1  # np.sqrt(XX)
        XtXt = np.sum(X_t.values**2)
        # # XtXt = np.linalg.norm(X_t, 1)
        XtXt_std = 1  # np.sqrt(XtXt)
        return complex(XX / XX_std - XtXt / XtXt_std)
        # Opt 3: Using norm with projected data
        # Q = np.sum(np.linalg.norm(X.values - X_p.values, ord=1, axis=1))
        # return float(Q)

    def _reset_score(self):
        self._score = None
        self._distances = None
        self._drift_detected = None

    def _transform_many(self, X: pd.DataFrame) -> pd.DataFrame:
        if (
            isinstance(self.subid, MiniBatchTransformer)
            or not isinstance(self.subid, Transformer)
            and hasattr(self.subid, "transform_many")
        ):
            X_p = self.subid.transform_many(X)
        else:
            X_p = pd.DataFrame(
                [
                    self.subid.transform_one(x)
                    for x in X.to_dict(orient="records")
                ]
            )
        return X_p

    def learn_one(self, x: dict, **params) -> None:
        """Allias for update method for interoperability with Pipeline."""
        self.update(x, **params)

    def learn_many(self, X: pd.DataFrame, **params) -> None:
        n = len(X)
        # If buffer is too small, learn in chunks
        buffer_len = self.ref_size + self.time_lag + self.test_size
        if n > buffer_len:
            for X_part in [
                X[i : i + buffer_len] for i in range(0, X.shape[0], buffer_len)
            ]:
                self.learn_many(X_part, **params)
            return
        # This would discard samples beyond window size, but we make chunks
        self._X.extend(X.to_dict(orient="records"))

        X_ = pd.DataFrame(self._X)

        # Learn the model if data past the time lag and test size is availabe
        # If learn_after_grace is False learn only when grace period is not yet over
        if len(self._X) > self.learn_delay and (
            self.learn_after_grace or self.n_seen < self.grace_period
        ):
            if isinstance(self.subid, Rolling):
                self.subid.update_many(
                    X_.iloc[-self.learn_delay - n : -self.learn_delay],
                    **params,
                )
            elif isinstance(self.subid, MiniBatchTransformer):
                self.subid.learn_many(
                    X_.iloc[-self.learn_delay - n : -self.learn_delay],
                    **params,
                )
            else:
                for x in X_.iloc[
                    -self.learn_delay - n : -self.learn_delay
                ].to_dict(orient="records"):
                    self.subid.learn_one(x, **params)
        self.n_seen += n

    def predict_one(self, *args):
        return self._drift_detected

    def score_one(self, x: dict) -> float:
        # Temporarily add the new sample to the buffer
        self._X.append(x)

        self._reset_score()
        score = self.score

        # Preserve stateless behavior
        self._X.pop()
        return score

    def update(self, x: dict, **params) -> None:
        self._X.append(x)
        # Learn the model if data past the time lag and test size is availabe
        # If learn_after_grace is False learn only when grace period is not yet over
        if len(self._X) > self.learn_delay and (
            self.learn_after_grace or self.n_seen < self.grace_period
        ):
            if isinstance(self.subid, Rolling):
                self.subid.update(self._X[-self.learn_delay - 1], **params)
            else:
                self.subid.learn_one(self._X[-self.learn_delay - 1], **params)
        self.n_seen += 1


class DMDOptSubIDChangeDetector(SubIDChangeDetector):
    """Change-Point Detection on Subspace Identification with Online DMD.

    This class implements is optimized for the OnlineDMD and OnlineDMDwC classes,
    where computation of eigenvalues during transformation creates a bottleneck.
    It stores transformed data and only recomputes the transformation when the
    Koopman operator changes.

    The computation time is approx. 20% lower for the OnlineDMD (80 features).
    This has however, impact on the overall performance and adds positive trand
    in D_train - D_test score.

    Args:
        SubIDChangeDetector (_type_): _description_
    """

    def __init__(
        self,
        subid: OnlineDMD | OnlineDMDwC | Rolling,
        ref_size: int,
        test_size: int | None = None,
        threshold: float = 0.25,
        time_lag: int = 0,
        grace_period: int = 0,
        learn_after_grace: bool = True,
    ):
        super().__init__(
            subid=subid,
            ref_size=ref_size,
            test_size=test_size,
            threshold=threshold,
            time_lag=time_lag,
            grace_period=grace_period,
            learn_after_grace=learn_after_grace,
        )
        self.subid = subid  # Correct type hinting
        self._Xp: deque[dict] = deque(
            maxlen=self.ref_size + self.time_lag + self.test_size
        )

    def _transform_many(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(self.subid, Rolling):
            subid_: Transformer = self.subid.obj  # type: ignore
        else:
            subid_ = self.subid
        if (isinstance(subid_, OnlineDMD | OnlineDMDwC)) and subid_.A_allclose:
            self._Xp.append(subid_.transform_one(X.iloc[-1].to_dict()))
        else:
            X_p = super()._transform_many(X)
            self._Xp.extend(X_p.to_dict(orient="records"))

        return pd.DataFrame(self._Xp)
