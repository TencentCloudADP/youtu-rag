import uuid

from utu.config import ConfigLoader
from utu.eval import DBDataManager

config = ConfigLoader.load_eval_config("ww")
config.exp_id = f"test_{uuid.uuid4()}"  # ensure unique
print(config)
db_manager = DBDataManager(config)


async def test_db_manager():
    data = db_manager.load()
    print(f"Load {len(data)} samples")
    data_state1 = db_manager.get_samples("init")
    print(f"Get {len(data_state1)} samples from `init` stage")
    data_state2 = db_manager.get_samples("rollout")
    print(f"Get {len(data_state2)} samples from `rollout` stage")
