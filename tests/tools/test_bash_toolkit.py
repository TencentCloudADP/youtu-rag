import pytest

from utu.config import ConfigLoader
from utu.tools import BashToolkit
from utu.utils import DIR_ROOT


@pytest.fixture
def bash_toolkit() -> BashToolkit:
    config = ConfigLoader.load_toolkit_config("bash")
    config.config["workspace_root"] = str(DIR_ROOT / "data" / "test_bash")
    return BashToolkit(config=config)


async def test_run_bash(bash_toolkit: BashToolkit):
    result = await bash_toolkit.run_bash("curl https://httpbin.org/get")
    print(result)
    result = await bash_toolkit.run_bash("wget https://www.gnu.org/software/wget/manual/wget.html -O wget.html")
    print(result)
