import asyncio
import inspect
import threading
import time
from multiprocessing import cpu_count

from bayes_opt import BayesianOptimization as BO
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

VERBOSE = 0


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

    def initialize(self, bo: BO, n_iter=2) -> None:
        self._bo = bo
        self._bo._prime_queue(20)
        self.n_iter = n_iter

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
            if VERBOSE > 1:
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
            if VERBOSE > 2:
                print("Iter", iteration)

        self.write(json.dumps(suggested_params))


def run_optimization_app(bo):
    asyncio.set_event_loop(asyncio.new_event_loop())
    handlers = [
        (
            r"/bayesian_optimization",
            BayesianOptimizationHandler,
            {
                "bo": bo,
            },
        ),
    ]
    server = tornado.httpserver.HTTPServer(tornado.web.Application(handlers))
    server.listen(9009)
    tornado.ioloop.IOLoop.instance().start()


stop_event = threading.Event()


def run_optimizer(f, config, results, n_iter):
    global stop_event

    name = config["name"]
    colour = config["colour"]

    register_data = {}
    max_target = None
    # for _ in range(2):
    while not stop_event.is_set():
        status = name + " wants to register: {}.\n".format(register_data)

        resp = requests.post(
            url="http://localhost:9009/bayesian_optimization",
            json=register_data,
        ).json()
        target = f(**resp)

        register_data = {
            "params": resp,
            "target": target,
        }

        if max_target is None or target > max_target:
            max_target = target

        status += name + " got {} as target.\n".format(target)
        status += name + " will to register next: {}.\n".format(register_data)
        if VERBOSE > 1:
            print(colour + status, end="\n")
        if iteration > n_iter:
            if VERBOSE > 1:
                print("Stopping all optimizers.")
            stop_event.set()

    results.append((name, max_target))
    if VERBOSE > 1:
        print(colour + name + " is done!", end="\n\n")


def maximize(f, pbounds, n_jobs, n_iter=25):
    bo = BO(
        f,
        pbounds,
        random_state=RANDOM_STATE,
        allow_duplicate_points=True,
    )
    ioloop = tornado.ioloop.IOLoop.instance()
    optimizers_config = [
        {"name": f"optimizer {i}", "colour": next(colorama_iterator)[1]}
        for i in range(n_jobs)
    ]

    app_thread = threading.Thread(
        target=run_optimization_app,
        kwargs={"bo": bo},
    )
    app_thread.daemon = True
    app_thread.start()

    targets = (run_optimizer,) * n_jobs
    results = []
    optimizer_threads = []
    for target in targets:
        optimizer_threads.append(
            threading.Thread(
                target=target,
                kwargs={
                    "f": mod_fun,
                    "config": optimizers_config.pop(),
                    "results": results,
                    "n_iter": n_iter,
                },
            )
        )
        optimizer_threads[-1].daemon = True
        optimizer_threads[-1].start()

    for optimizer_thread in optimizer_threads:
        optimizer_thread.join()

    for result in results:
        if VERBOSE > 1:
            print(result[0], "found a maximum value of: {}".format(result[1]))

    ioloop.stop()
    return sorted(results, key=lambda x: x[1], reverse=True)[0][1]


if __name__ == "__main__":

    def mod_fun(x, y):
        """Function with unknown internals we wish to maximize.

        This is just serving as an example, however, for all intents and
        purposes think of the internals of this function, i.e.: the process
        which generates its outputs values, as unknown.
        """
        # time.sleep(random.uniform(0, 7))
        return -(x**2) - (y - 1) ** 2 + 1

    pbounds = {"x": (-4, 4), "y": (-3, 3)}
    n_jobs = 1  # cpu_count()
    start_time = time.time()
    results = maximize(mod_fun, pbounds, n_jobs)
    end_time = time.time()

    execution_time = end_time - start_time
    print("Execution time:", execution_time)
    print(results)
