"""Change Detection based on Subspace Identification algorithm."""

from collections import deque

import numpy as np
import pandas as pd
from river.anomaly.base import AnomalyDetector
from river.base import MiniBatchTransformer, Transformer
from river.decomposition import OnlineDMD, OnlineDMDwC
from river.utils.rolling import BaseRolling


class SubIDChangeDetector(AnomalyDetector):
    def __init__(
        self,
        subid: MiniBatchTransformer | Transformer | BaseRolling,
        ref_size: int,
        test_size: int | None = None,
        threshold: float = 0.1,
        time_lag: int = 0,
        grace_period: int = 0,
        learn_after_grace: bool = True,
    ):
        self.subid = subid
        self.threshold = threshold
        if ref_size == 0 and hasattr(subid, "window_size"):
            self.ref_size = subid.window_size  # type: ignore
        else:
            self.ref_size = ref_size
        self.test_size = test_size if test_size is not None else ref_size
        self.time_lag = time_lag
        assert self.ref_size > 0
        assert self.test_size > 0
        assert self.test_size + self.time_lag >= 0
        # assert self.grace_period < self.test_size
        # TODO: basically grace period should be omitted and detection start once Transformer is fitted
        self.grace_period = grace_period
        self.learn_after_grace = learn_after_grace
        self.n_seen = 0
        self.score: float = 0.0

        self._distances: tuple[float, float]
        self._drift_detected: bool = False

        self._X: deque[dict] = deque(
            maxlen=self.ref_size + self.time_lag + self.test_size
        )

    @property
    def _supervised(self):
        """Indicates whether or not the estimator is supervised or not.

        This is useful internally for determining if an estimator expects to be provided with a `y`
        value in it's `learn_one` method. For instance we use this in a pipeline to know whether or
        not we should pass `y` to an estimator or not.

        """
        return False

    def _compute_distance(self, X: pd.DataFrame, X_p: pd.DataFrame) -> float:
        """Compute the distance between the Hankel matrix and its transformation.

        This formulation computes a measure of how much information in the dataset represented by Y is preserved or retained when projected onto the space spanned by W. The difference between the covariance matrix of Y and the projected version is computed, and the sum of all elements in this difference matrix gives an overall measure of dissimilarity or distortion.

        Args:
            Y): Hankel matrix

        Returns:
            Distance between the Hankel matrix and its transformation.
        """
        XX = (X**2).sum().sum()
        # XX = np.linalg.norm(X, 1)
        # Using following normalization changes the score baseline based on
        #  the proportion of ref and test size
        XX_std = 1  # np.sqrt(XX)
        XpXp = (X_p**2).sum().sum()
        # XpXp = np.linalg.norm(X_p, 1)
        XpXp_std = 1  # np.sqrt(XpXp)

        return XX / XX_std - XpXp / XpXp_std

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

    def update(self, x: dict, **params) -> None:
        self._X.append(x)

        ref_delay = self.time_lag + self.test_size
        if len(self._X) > ref_delay and (
            self.learn_after_grace or self.n_seen < self.grace_period
        ):
            if isinstance(self.subid, BaseRolling):
                self.subid.update(self._X[-ref_delay - 1], **params)
            else:
                self.subid.learn_one(self._X[-ref_delay - 1], **params)
        self.n_seen += 1

        if (
            self.n_seen >= self.grace_period
            and len(self._X) == self.ref_size + ref_delay
        ):
            X = pd.DataFrame(self._X)
            X_p = self._transform_many(X)
            D_train: float = (
                self._compute_distance(
                    X.iloc[: self.ref_size, :],
                    X_p.iloc[: self.ref_size, :],
                )
                / self.ref_size
            )
            D_test = (
                self._compute_distance(
                    X.iloc[-self.test_size :, :],
                    X_p.iloc[-self.test_size :, :],
                )
                / self.test_size
            )
            # TODO: Figure out onder what circumstances the distance of train
            #  is higher than the distance of test (lower test noise?, running normalization, ...)
            self.distances = (D_train, D_test)
            self.score = (D_test / D_train) - 1
            # TODO: explore interesting scoring option
            # self.score = D_train - D_test
            # TODO: figure out proper way of utilizing imaginary part of score
            if isinstance(self.score, complex):
                self.score = self.score.real + np.abs(self.score.imag)
            # TODO: comment on score shawing
            self.score = max(self.score, 0.0)
            self._drift_detected = self.score > self.threshold
        else:
            self.score = 0.0
            self._drift_detected = False

    def score_one(self, *args):
        return self.score

    def predict_one(self, *args):
        return self._drift_detected

    def learn_one(self, x: dict, **params) -> None:
        """Allias for update method for interoperability with Pipeline."""
        self.update(x, **params)


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
        subid: OnlineDMD | OnlineDMDwC,
        ref_size: int,
        test_size: int | None = None,
        threshold: float = 0.1,
        time_lag: int = 0,
        grace_period: int = 0,
    ):
        super().__init__(
            subid=subid,
            ref_size=ref_size,
            test_size=test_size,
            threshold=threshold,
            time_lag=time_lag,
            grace_period=grace_period,
        )
        self.subid = subid  # Correct type hinting
        self._Xp: deque[dict] = deque(
            maxlen=self.ref_size + self.time_lag + self.test_size
        )

    def _transform_many(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(self.subid, BaseRolling):
            subid_: Transformer = self.subid.obj  # type: ignore
        else:
            subid_ = self.subid
        if (
            isinstance(subid_, (OnlineDMD, OnlineDMDwC))
        ) and subid_.A_allclose:
            self._Xp.append(subid_.transform_one(X.iloc[-1].to_dict()))
        else:
            X_p = super()._transform_many(X)
            self._Xp.extend(X_p.to_dict(orient="records"))

        return pd.DataFrame(self._Xp)
