from collections import deque

import numpy as np
import pandas as pd
from river.base import DriftDetector, MiniBatchTransformer, Transformer
from river.utils.rolling import BaseRolling


class SubIDDriftDetector(DriftDetector):
    def __init__(
        self,
        subid: MiniBatchTransformer | Transformer | BaseRolling,
        ref_size: int,
        test_size: int,
        threshold: float = 0.1,
        time_lag: int = 0,
        grace_period: int = 0,
    ):
        self.subid = subid
        self.threshold = threshold
        if ref_size == 0 and hasattr(subid, "window_size"):
            self.ref_size = subid.window_size  # type: ignore
        else:
            self.ref_size = ref_size
        self.test_size = test_size
        self.time_lag = time_lag
        assert self.ref_size > 0
        assert self.test_size > 0
        assert self.test_size + self.time_lag >= 0
        # assert self.grace_period < self.test_size
        # TODO: basically grace period should be omitted and detection start once Transformer is fitted
        self.grace_period = grace_period
        self.n_seen = 0
        self._drift_detected: bool
        self.score: float
        self._Y: deque[dict] = deque(
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

    def _compute_distance(self, Y: pd.DataFrame) -> float:
        """Compute the distance between the Hankel matrix and its transformation.

        This formulation computes a measure of how much information in the dataset represented by Y is preserved or retained when projected onto the space spanned by W. The difference between the covariance matrix of Y and the projected version is computed, and the sum of all elements in this difference matrix gives an overall measure of dissimilarity or distortion.

        Args:
            Y): Hankel matrix

        Returns:
            Distance between the Hankel matrix and its transformation.
        """
        if isinstance(self.subid, MiniBatchTransformer):
            Y_p = self.subid.transform_many(Y)
        else:
            Y_p = pd.DataFrame(
                [
                    self.subid.transform_one(x)
                    for x in Y.to_dict(orient="records")
                ]
            )
        YY = (Y**2).sum().sum()
        # YY = np.linalg.norm(Y, 1)
        YY_std = np.sqrt(YY)
        YpYp = (Y_p**2).sum().sum()
        # YpYp = np.linalg.norm(Y_p, 1)
        YpYp_std = np.sqrt(YpYp)

        D = YY / YY_std - YpYp / YpYp_std
        return float(D)

    def update(self, x: dict) -> None:
        self._Y.append(x)

        ref_delay = self.time_lag + self.test_size
        if len(self._Y) > ref_delay:
            if isinstance(self.subid, BaseRolling):
                self.subid.update(self._Y[-ref_delay - 1])
            else:
                self.subid.learn_one(self._Y[-ref_delay - 1])

        if (
            self.n_seen >= self.grace_period
            and len(self._Y) >= self.ref_size + ref_delay
        ):
            Y = pd.DataFrame(self._Y)
            D_train = (
                self._compute_distance(Y.iloc[: self.ref_size, :])
                / self.ref_size
            )
            D_test = (
                self._compute_distance(Y.iloc[-self.test_size :, :])
                / self.test_size
            )
            # TODO: Figure out onder what circumstances the distance of train
            #  is higher than the distance of test (lower test noise?, running normalization, ...)
            self.score = (D_test / D_train) - 1
            # TODO: comment on score shawing
            self.score = max(self.score, 0.0)
            self._drift_detected = self.score > self.threshold
        else:
            self.score = 0.0
            self._drift_detected = False

        self.n_seen += 1

    def learn_one(self, x: dict) -> None:
        """Allias for update method for interoperability with Pipeline."""
        self.update(x)
