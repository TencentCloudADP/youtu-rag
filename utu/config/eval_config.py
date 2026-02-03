from typing import Literal

from pydantic import Field

from ..utils import EnvUtils
from .agent_config import AgentConfig, ModelConfigs
from .base_config import ConfigBaseModel


class DataConfig(ConfigBaseModel):
    """Data config"""

    dataset: str  # WebWalkerQA | GAIA_validation | XBench | BrowseComp
    """Built-in dataset name or custom dataset path"""
    type: Literal["single", "mixed"] = "single"
    """Whether the dataset contains only single benchmark data or multiple benchmarks"""
    question_field: str = "question"
    """Question field name in the dataset"""
    gt_field: str = "answer"
    """Ground truth field name in the dataset"""


class EvalConfig(ConfigBaseModel):
    """Evaluation config"""

    exp_id: str = "default"
    """Experiment ID"""

    # data
    db_url: str = EnvUtils.get_env("UTU_DB_URL", "sqlite:///test.db")
    """Database URL"""
    data: DataConfig = None
    """Data config"""

    # rollout
    agent: AgentConfig | None = None
    """Agent config for rollout"""
    concurrency: int = 1
    """Rollout parallelism"""

    # judgement
    judge_model: ModelConfigs = Field(default_factory=ModelConfigs)
    """Judge model config"""
    judge_concurrency: int = 1
    """Judgement parallelism"""
    eval_method: str = None
    """Evaluation method"""
