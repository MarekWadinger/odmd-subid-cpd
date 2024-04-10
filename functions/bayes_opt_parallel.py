import asyncio
import inspect
import threading
from multiprocessing import cpu_count

from bayes_opt import BayesianOptimization as BO
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from bayes_opt.util import UtilityFunction
from colorama import Fore

try:
    import json

    import requests
    import tornado.httpserver
    import tornado.ioloop
    from tornado.web import RequestHandler
except ImportError:
    raise ImportError(
        "In order to run this example you must have the libraries: "
        + "`tornado` and `requests` installed."
    )


class ColoramaIterator:
    def __init__(self):
        self.attributes = inspect.getmembers(Fore)
        self.uppercase_attributes = (
            attr for attr in self.attributes if attr[0].isupper()
        )
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            attr_name, attr_value = next(self.uppercase_attributes)

        except StopIteration:
            # Reset the iterator to loop back
            self.uppercase_attributes = (
                attr for attr in self.attributes if attr[0].isupper()
            )
            attr_name, attr_value = next(self.uppercase_attributes)
        return attr_name, attr_value


colorama_iterator = ColoramaIterator()
RANDOM_STATE = 42
iteration = 0


class BayesianOptimizationHandler(RequestHandler):
    """Basic functionality for NLP handlers."""

    def initialize(self, bo: BO, init_points, verbose=1) -> None:
        self._bo = bo
        self._bo._prime_queue(init_points=init_points)

        self._verbose = verbose

        self._uf = UtilityFunction(kind="ucb", kappa=3, xi=1)

    def post(self):
        global iteration
        """Deal with incoming requests."""
        body = tornado.escape.json_decode(self.request.body)

        try:
            self._bo.register(
                params=body["params"],
                target=body["target"],
            )
            if self._verbose > 1:
                print(
                    "BO has registered: {} points.".format(
                        len(self._bo.space)
                    ),
                    end="\n\n",
                )
        except KeyError:
            pass
        finally:
            suggested_params = self._bo.suggest(self._uf)
            iteration += 1
            if self._verbose > 1:
                print("Iter", iteration)

        self.write(json.dumps(suggested_params))


class BayesianOptimization(BO):
    def __init__(
        self,
        f,
        pbounds,
        constraints,
        random_state=None,
        verbose=2,
        allow_duplicate_points=True,
    ):
        super().__init__(
            f,
            pbounds,
            constraints,
            random_state,
            verbose,
            allow_duplicate_points=allow_duplicate_points,
        )
        self.f = f
        self.stop_event = threading.Event()
        self.results = []

    @property
    def max(self):
        _, params, target = self.results[0]
        return {"params": params, "target": target}

    def run_optimization_app(self, init_points):
        asyncio.set_event_loop(asyncio.new_event_loop())
        handlers = [
            (
                r"/bayesian_optimization",
                BayesianOptimizationHandler,
                {
                    "bo": super(),
                    "init_points": init_points,
                },
            ),
        ]
        server = tornado.httpserver.HTTPServer(
            tornado.web.Application(handlers)
        )
        server.listen(9009)
        tornado.ioloop.IOLoop.instance().start()

    def run_optimizer(self, config, n_iter):
        name = config["name"]
        colour = config["colour"]

        register_data = {}
        max_params = None
        max_target = None
        # for _ in range(2):
        while not self.stop_event.is_set():
            status = name + " wants to register: {}.\n".format(register_data)

            resp = requests.post(
                url="http://localhost:9009/bayesian_optimization",
                json=register_data,
            ).json()
            target = self.f(**resp)

            register_data = {
                "params": resp,
                "target": target,
            }

            if max_target is None or target > max_target:
                max_target = target
                max_params = resp

            status += name + " got {} as target.\n".format(target)
            status += name + " will register next: {}.\n".format(register_data)
            if self._verbose > 1:
                print(colour + status, end="\n")
            if iteration > n_iter:
                if self._verbose > 1:
                    print("Stopping all optimizers.")
                self.stop_event.set()

        if self._verbose > 1:
            print(colour + name + " is done!", end="\n\n")
        self.results.append((name, max_params, max_target))

    def maximize(
        self,
        init_points=5,
        n_iter=100,
        n_jobs=1,
    ):
        self.dispatch(Events.OPTIMIZATION_START)
        ioloop = tornado.ioloop.IOLoop.instance()
        optimizers_config = [
            {"name": f"optimizer {i}", "colour": next(colorama_iterator)[1]}
            for i in range(n_jobs)
        ]

        app_thread = threading.Thread(
            target=self.run_optimization_app,
            kwargs={"init_points": init_points},
        )
        app_thread.daemon = True
        app_thread.start()

        targets = (self.run_optimizer,) * (
            n_jobs if n_jobs < n_iter else n_iter
        )
        optimizer_threads = []
        for target in targets:
            optimizer_threads.append(
                threading.Thread(
                    target=target,
                    kwargs={
                        "config": optimizers_config.pop(),
                        "n_iter": n_iter - n_jobs,
                    },
                )
            )
            optimizer_threads[-1].daemon = True
            optimizer_threads[-1].start()

        for optimizer_thread in optimizer_threads:
            optimizer_thread.join()

        if self._verbose > 2:
            for result in self.results:
                print(
                    result[0], "found a maximum value of: {}".format(result[2])
                )

        ioloop.stop()
        self.dispatch(Events.OPTIMIZATION_END)
        self.results = sorted(
            self.results,
            key=lambda x: float("-inf") if x[2] is None else x[2],
            reverse=True,
        )


# if __name__ == "__main__":

#     def mod_fun(x, y):
#         """Function with unknown internals we wish to maximize.

#         This is just serving as an example, however, for all intents and
#         purposes think of the internals of this function, i.e.: the process
#         which generates its outputs values, as unknown.
#         """
#         # time.sleep(random.uniform(0, 7))
#         return -(x**2) - (y - 1) ** 2 + 1

#     pbounds = {"x": (-4, 4), "y": (-3, 3)}
#     n_jobs = cpu_count()
#     optimizer = BayesianOptimization(
#         f=mod_fun,
#         pbounds=pbounds,
#         constraints=None,
#         random_state=RANDOM_STATE,
#         verbose=1,
#         allow_duplicate_points=True,
#     )
#     logger = JSONLogger(path="./.results/test.log")
#     optimizer.subscribe(Events.OPTIMIZATION_END, logger)
#     optimizer.maximize(n_jobs=n_jobs)
#     print(optimizer.max)
