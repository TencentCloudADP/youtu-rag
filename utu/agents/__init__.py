from ..config import AgentConfig
from .llm_agent import LLMAgent
from .orchestra_agent import OrchestraAgent
from .orchestrator_agent import OrchestratorAgent
from .parallel_orchestrator_agent import ParallelOrchestratorAgent
from .simple_agent import SimpleAgent
from .workforce_agent import WorkforceAgent


def get_agent(config: AgentConfig):
    if config.type == "simple":
        return SimpleAgent(config=config)
    elif config.type == "orchestra":
        return OrchestraAgent(config=config)
    elif config.type == "orchestra_react_sql":
        # Lazy import to avoid circular dependency
        from ..rag.rag_agents.orchestra_react_text2sql import OrchestraReactSqlAgent
        return OrchestraReactSqlAgent(config=config)
    elif config.type == "orchestrator":
        return OrchestratorAgent(config=config)
    elif config.type == "parallel_orchestrator":
        return ParallelOrchestratorAgent(config=config)
    elif config.type == "workforce":
        return WorkforceAgent(config=config)
    else:
        raise ValueError(f"Unknown agent type: {config.type}")


__all__ = [
    "SimpleAgent",
    "OrchestraAgent",
    "OrchestratorAgent",
    "ParallelOrchestratorAgent",
    "LLMAgent",
    "WorkforceAgent",
    "get_agent",
]
