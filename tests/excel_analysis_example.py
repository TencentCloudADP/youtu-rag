"""
Excel分析Agent示例
演示如何使用OrchestratorAgent进行Excel表格分析
"""
import asyncio
from utu.agents import OrchestratorAgent
from utu.config import ConfigLoader
from utu.utils import AgentsUtils


async def main():
    """主函数"""
    # 加载配置
    config = ConfigLoader.load_agent_config("ragref/excel_analysis/base")
    
    # 创建Agent
    agent = OrchestratorAgent(config)
    
    # 用户问题示例
    user_input = """
    文件路径：/Users/felix/Downloads/ExcelAgent/case/cat_breeds_simple.xlsx
    分析猫品种数据，整理关键信息，生成网页看板
    """

    user_input = """
    报告路径：/private/tmp/utu/python_executor/20260302_195013_8c6828b2/猫品种数据分析报告.md
    根据markdown报告生成网页看板
    """
    
    # 运行Agent（流式输出）
    print("=" * 50)
    print("开始分析...")
    print("=" * 50)
    
    recorder = agent.run_streamed(user_input)
    
    # 流式输出过程
    await AgentsUtils.print_stream_events(recorder.stream_events())
    
    # 输出结果
    print("\n" + "=" * 50)
    print("分析完成！")
    print("=" * 50)
    print(f"\n最终输出：\n{recorder.final_output}")
    
    # 查看执行轨迹
    if hasattr(recorder, 'trajectories'):
        print("\n" + "=" * 50)
        print("执行轨迹：")
        print("=" * 50)
        for trajectory in recorder.trajectories:
            print(f"\n步骤: {trajectory.get('task', 'Unknown')}")
            print(f"结果: {str(trajectory.get('result', ''))[:200]}...")


if __name__ == "__main__":
    asyncio.run(main())
