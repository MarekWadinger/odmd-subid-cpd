from multiprocessing import cpu_count

from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger

from ..functions.bayes_opt_parallel import BayesianOptimization

RANDOM_STATE = 42

if __name__ == "__main__":
    for _ in range(10):

        def mod_fun(x, y):
            """Function with unknown internals we wish to maximize.

            This is just serving as an example, however, for all intents and
            purposes think of the internals of this function, i.e.: the process
            which generates its outputs values, as unknown.
            """
            # time.sleep(random.uniform(0, 7))
            return -(x**2) - (y - 1) ** 2 + 1

        pbounds = {"x": (-4, 4), "y": (-3, 3)}
        n_jobs = cpu_count()
        optimizer = BayesianOptimization(
            f=mod_fun,
            pbounds=pbounds,
            constraints=None,
            random_state=RANDOM_STATE,
            verbose=1,
            allow_duplicate_points=True,
        )
        logger = JSONLogger(path="./.results/test.log")
        optimizer.subscribe(Events.OPTIMIZATION_END, logger)
        optimizer.maximize(n_jobs=n_jobs)
        print(optimizer.max)
