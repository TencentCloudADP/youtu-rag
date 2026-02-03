# ruff: noqa

from utu.agents.workforce import AnswererAgent, AssignerAgent, ExecutorAgent, PlannerAgent, WorkspaceTaskRecorder
from utu.config import ConfigLoader

# overall_task = "It's May 2023, and I'm about to drive across the U.S. from California to Maine. I always recycle my water bottles at the end of a trip, and I drink 5 12-ounce water bottles for every 100 miles I travel, rounded to the nearest 100. Assuming I follow I-40 from Los Angeles to Cincinnati, then take I-90 from Cincinnati to Augusta, how many dollars will I get back according to Wikipedia?"
overall_task = "What's the weather like in Shanghai tomorrow?"
executor_agents_info = [
    {
        "name": "Web Search Agent",
        "description": "A web information search specialist that excels at finding relevant information through search tools (Google, Wikipedia, archived pages) and extracting webpage content for understanding. Focuses on information discovery and identifying authoritative sources.",
    }
]


async def test_workforce_subagents():
    config = ConfigLoader.load_agent_config("workforce")
    planner_agent = PlannerAgent(config=config)
    assigner_agent = AssignerAgent(config=config)
    executor_agent = ExecutorAgent(config=config.workforce_executor_agents["SearchAgent"], workforce_config=config)
    answerer_agent = AnswererAgent(config=config)

    recorder = WorkspaceTaskRecorder(overall_task=overall_task, executor_agent_kwargs_list=executor_agents_info)
    await planner_agent.plan_task(recorder)
    print(f"> plan_task: {recorder.task_plan}")

    print(f"> assigning task {recorder.get_next_task()}")
    next_task = await assigner_agent.assign_task(recorder)
    print(f"> assign_task: {next_task}")

    await executor_agent.execute_task(recorder=recorder, task=next_task)
    print(f"> task_result: {next_task.task_result}")

    final_answer = await answerer_agent.extract_final_answer(recorder)
    print(f"> final_answer: {final_answer}")
