import asyncio
import inspect
import threading
from multiprocessing import cpu_count

from bayes_opt import BayesianOptimization as BO
from bayes_opt.event import Events
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

        if self._bo._bounds_transformer and iteration > 0:
            # The bounds transformer should only modify the bounds after
            # the init_points points (only for the true iterations)
            self._bo.set_bounds(
                self._bo._bounds_transformer.transform(self._bo._space)
            )

        if self._verbose > 2:
            print("Iteration:", iteration)

        self.write(json.dumps(suggested_params))


class BayesianOptimization(BO):
    def __init__(
        self,
        f,
        pbounds,
        constraints=None,
        random_state=None,
        verbose=2,
        bounds_transformer=None,
        allow_duplicate_points=True,
    ):
        super().__init__(
            f,
            pbounds,
            constraints,
            random_state,
            verbose,
            bounds_transformer,
            allow_duplicate_points,
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
                    "bo": self,
                    "init_points": init_points,
                    "verbose": self._verbose,
                },
            ),
        ]
        self.server = tornado.httpserver.HTTPServer(
            tornado.web.Application(handlers)  # type: ignore
        )
        self.server.listen(9009)
        tornado.ioloop.IOLoop.instance().start()

    def run_optimizer(self, config, n_iter):
        name = config["name"]
        colour = config["colour"]

        register_data: dict[str, dict | int] = {}
        max_params = None
        max_target = None
        # for _ in range(2):
        while not self.stop_event.is_set():
            status = name + f" wants to register: {register_data}.\n"

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

            status += name + f" got {target} as target.\n"
            status += name + f" will register next: {register_data}.\n"
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
        n_jobs=cpu_count(),
    ):
        # Reset iteration count for each initialization
        global iteration
        iteration = 0

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

        if self._verbose > 1:
            for result in self.results:
                print(result[0], f"found a maximum value of: {result[2]}")

        ioloop.stop()
        self.server.stop()
        self.dispatch(Events.OPTIMIZATION_END)
        self.results = sorted(
            self.results,
            key=lambda x: float("-inf") if x[2] is None else x[2],
            reverse=True,
        )
