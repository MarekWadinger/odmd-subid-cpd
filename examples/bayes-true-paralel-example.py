from __future__ import annotations

import pickle
import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from bayes_opt.bayesian_optimization import BayesianOptimization
from bayes_opt.util import UtilityFunction
from bayes_opt import SequentialDomainReductionTransformer
from scipy.optimize import rosen
from tqdm import tqdm

RANDOM_STATE = 42


def _closest_distance(point, points):
    return min(np.linalg.norm(point - p) for p in points if p is not point)


def optimize(
    func: Callable[..., float],
    num_iter: int,
    bounds: dict[str, tuple[float, float]],
    num_workers=0,
    with_dummies=False,
):
    init_samples = int(np.sqrt(num_iter))

    # bounds_transformer = SequentialDomainReductionTransformer(
    #     minimum_window=0.5
    # )

    optimizer = BayesianOptimization(
        f=None,
        pbounds=bounds,
        verbose=0,
        random_state=RANDOM_STATE,
        # bounds_transformer=bounds_transformer,
    )
    init_kappa = 10
    kappa_decay = (0.1 / init_kappa) ** (1 / (num_iter - init_samples))
    utility = UtilityFunction(
        kind="ucb",
        kappa=init_kappa,
        xi=0.0,
        kappa_decay=kappa_decay,
        kappa_decay_delay=init_samples,
    )

    init_queue = [optimizer.suggest(utility) for _ in range(init_samples)]
    result_queue = []
    # tbar = tqdm(total=num_iter, leave=False)
    while len(optimizer.res) < num_iter:
        sample = (
            init_queue.pop(0) if init_queue else optimizer.suggest(utility)
        )
        if with_dummies and num_workers > 1:
            optimizer.register_dummy(sample, -1)
        loss = func(list(sample.values())) * -1
        result_queue.append((sample, loss))
        if len(result_queue) >= num_workers:
            try:
                optimizer.register(*result_queue.pop(0))
                utility.update_params()
                # tbar.update()
            except KeyError:
                pass
    return optimizer.res


bounds = {"x": (-5., 5.), "y": (-5., 5.)}

all_times = {}
all_results = {}
workers_each = [1, 8, 24]
print(f"Simulating parallel optimization for {workers_each} workers, this can take some time.")
for num_workers in workers_each:
    print(f"\tChecking {num_workers} workers")
    results = []
    start = time.perf_counter()
    results = optimize(rosen, 150, bounds, num_workers, with_dummies=True)
    end = time.perf_counter()
    delta = end - start
    all_times[num_workers] = delta
    all_results[num_workers] = results
with open("results.pkl", "wb") as f:
    pickle.dump(all_results, f)
with open("results.pkl", "rb") as f:
    all_results = pickle.load(f)

fig, axs = plt.subplots(2, 2)
fig.set_figheight(8)
fig.set_figwidth(8)
axs = [item for sublist in axs for item in sublist]
for idx, (num_workers, results) in enumerate(all_results.items()):
    samples = [np.array(list(res["params"].values())) for res in results]
    axs[idx].scatter(*zip(*samples), s=1)
    axs[idx].set_title(f"{num_workers=}")
    avg_min_distance = np.mean(
        [_closest_distance(sample, samples) for sample in samples]
    )
    best_result = max([res["target"] for res in results])
    print(
        f"{num_workers=}, mean_min_distance={avg_min_distance:.3f}, {best_result=:.3f}, time={all_times[num_workers]:.3f}"
    )
fig.tight_layout()
fig.savefig("results.png")
