import argparse
import asyncio

from utu.meta import ToolGenerator, ToolGeneratorDebugger
from utu.utils import DIR_ROOT, AgentsUtils, PrintUtils


async def do_gen():
    generator = ToolGenerator()
    task = await PrintUtils.async_print_input("Enter your tool requirements: ")
    task_recorder = generator.run_streamed(task)
    await AgentsUtils.print_stream_events(task_recorder.stream_events())
    PrintUtils.print_info(f"Generated tool config saved to {task_recorder.output_file}", color="green")

    if_debug = await PrintUtils.async_print_input("Do you want to debug the tool? (y/n): ")
    if if_debug.lower() == "y":
        await do_debug(task_recorder.name)


async def do_debug(tool_name: str):
    workspace_dir = DIR_ROOT / "configs/tools/generated" / tool_name
    assert workspace_dir.exists()
    generator = ToolGeneratorDebugger()
    recorder = generator.run_streamed(workspace_dir)
    await AgentsUtils.print_stream_events(recorder.stream_events())


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--tool_name", type=str, default=None)
    args = parser.parse_args()

    if args.debug:
        await do_debug(args.tool_name)
    else:
        await do_gen()


if __name__ == "__main__":
    asyncio.run(main())
