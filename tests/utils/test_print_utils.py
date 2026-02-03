import asyncio

from utu.utils import PrintUtils


def test_print_input():
    res = PrintUtils.print_input("> ")
    print(res)


def test_async_print_input():
    async def test():
        res = await PrintUtils.async_print_input("> ")
        print(res)

    asyncio.run(test())
