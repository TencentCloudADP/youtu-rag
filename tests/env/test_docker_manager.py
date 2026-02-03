import asyncio

from utu.env.utils import DockerManager, MCPClient

docker_manager = DockerManager()


async def test_mcp():
    id = "test"
    res = await docker_manager.find_all(stop=True)
    print(res)
    container_info = await docker_manager.start_container(id)
    print(f"container_info: {container_info}")
    async with MCPClient.get_mcp_client(container_info["mcp_url"]) as client:
        res = await client.list_tools()
        print(res)
        res = await client.call_tool(
            "go_to_url", {"url": "https://github.com/modelcontextprotocol/python-sdk/issues/79"}
        )
        print(res)
    await docker_manager.stop_container(id)


async def test_concurrency():
    n = 20
    tasks = [docker_manager.start_container(f"test_{i}") for i in range(n)]
    res = await asyncio.gather(*tasks)
    print(res)
    stat = docker_manager.get_all_status()
    print(stat)
    docker_manager.cleanup()


async def test_find():
    res = await docker_manager.find_all()
    print(res)
    ids = [c["docker_id"] for c in res["found_containers"]]
    res = await docker_manager.stop_all_by_cid(ids)
    print(res)


if __name__ == "__main__":
    asyncio.run(test_find())
