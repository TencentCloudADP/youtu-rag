from utu.config import ConfigLoader
from utu.ww.ww_benchmark import WWBenchmark

config = ConfigLoader.load_eval_config("ww")
print(f"> config: {config}")
config.exp_id = "ww150_oss120_0808_base01"
benchmark = WWBenchmark(config)


async def test_rollout_one():
    benchmark.preprocess()
    sample = benchmark.dataset.get_samples(limit=1)[0]
    res = await benchmark.rollout_one(sample)
    # print(f"trace_id: {res.trace_id}. response: {res.response}")
    print(res)


async def test_judge_one():
    sample = benchmark.dataset.get_samples(stage="rollout", limit=1)[0]
    res = await benchmark.judge_one(sample)
    print(res)
