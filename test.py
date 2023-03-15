import time
import ray
from ray import tune
from ray.air import session
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search import ConcurrencyLimiter

def evaluate(step, width, height):
    time.sleep(0.1)
    # 超参数是width和height
    return (0.1 + width * step / 100) ** (-1) + height * 0.1

def objective(config):
    for step in range(config["steps"]):
        score = evaluate(step, config["width"], config["height"])
        session.report({"iterations": step, "mean_loss": score})

algo = BayesOptSearch(utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0})
# 搜索时限制并发数为4
algo = ConcurrencyLimiter(algo, max_concurrent=4)

num_samples = 10

search_space = {
    "steps": 100,
    "width": tune.uniform(0, 20),
    "height": tune.uniform(-100, 100),
}

tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="mean_loss",
        mode="min",
        search_alg=algo,
        num_samples=num_samples,
    ),
    param_space=search_space,
)
results = tuner.fit()

print("Best hyperparameters found were: ", results.get_best_result().config)