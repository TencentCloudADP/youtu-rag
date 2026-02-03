from utu.config import ConfigLoader
from utu.eval.benchmarks import BaseBenchmark

config = ConfigLoader.load_eval_config("v00")
benchmark = BaseBenchmark(config)


async def test_benchmark_rollout():
    benchmark.preprocess()
    sample = benchmark.dataset.get_samples("init", limit=1)[0]
    res = await benchmark.rollout_one(sample)
    print(res.as_dict())
