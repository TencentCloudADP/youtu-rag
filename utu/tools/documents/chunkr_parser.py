from ...utils import CACHE_DIR, EnvUtils, async_file_cache, get_logger

logger = get_logger(__name__)


class ChunkrParser:
    def __init__(self, config: dict) -> None:
        self.cache_dir = CACHE_DIR / "documents"
        EnvUtils.ensure_package("chunkr_ai")
        EnvUtils.assert_env("CHUNKR_API_KEY")
        from chunkr_ai import Chunkr
        from chunkr_ai.models import Configuration

        self.chunkr = Chunkr(api_key=config.get("CHUNKR_API_KEY"))
        self.chunkr.config = Configuration(
            high_resolution=config.get("high_resolution", True),
        )

    @async_file_cache(expire_time=None)
    async def parse(self, path: str) -> str:
        """Parse document to markdown with Chunkr.

        - ref: <https://docs.chunkr.ai/sdk/data-operations/create#supported-file-types>

        Args:
            md5 (str): md5 of the document.
        """
        task = await self.chunkr.upload(path)

        logger.info("  getting results...")
        markdown = task.markdown()
        return markdown
