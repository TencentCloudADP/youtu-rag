import pytest

from utu.config import ConfigLoader
from utu.tools import FileEditToolkit
from utu.utils import DIR_ROOT

CONTENT = """line 1
line 2
line 3
"""

DIFF = """<<<<<<< SEARCH
line 1
=======
line 1 edited!
>>>>>>> REPLACE"""


@pytest.fixture
def file_edit_toolkit():
    config = ConfigLoader.load_toolkit_config("bash")
    config.config["work_dir"] = str(DIR_ROOT / "data" / "test_bash")
    return FileEditToolkit(config=config)


async def test_edit_file(file_edit_toolkit: FileEditToolkit):
    result = await file_edit_toolkit.write_file("test.txt", CONTENT)
    print(result)
    result = await file_edit_toolkit.edit_file("test.txt", DIFF)
    print(result)
