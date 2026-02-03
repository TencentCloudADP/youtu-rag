"""
@smolagents/src/smolagents/default_tools.py
https://github.com/martin-majlis/Wikipedia-API
https://www.mediawiki.org/wiki/API:Main_page
"""

import calendar
import datetime

import requests

from ..config import ToolkitConfig
from .base import AsyncBaseToolkit, register_tool


class WikipediaSearchTool(AsyncBaseToolkit):
    """
    WikipediaSearchTool searches Wikipedia and returns a summary or full text of the given topic.

    Attributes:
        user_agent (str): A custom user-agent string to identify the project.
            This is required as per Wikipedia API policies, read more here:
            http://github.com/martin-majlis/Wikipedia-API/blob/master/README.rst
        language (str): The language in which to retrieve Wikipedia articles.
            http://meta.wikimedia.org/wiki/List_of_Wikipedias
        content_type (str): Defines the content to fetch.
            Can be "summary" for a short summary or "text" for the full article.
        extract_format (str): Defines the output format. Can be `"WIKI"` or `"HTML"`.
    """

    def __init__(self, config: ToolkitConfig | dict = None) -> None:
        super().__init__(config)
        try:
            import wikipediaapi
        except ImportError as e:
            raise ImportError(
                "You must install `wikipedia-api` to run this tool: for instance run `pip install wikipedia-api`"
            ) from e

        # Map string format to wikipediaapi.ExtractFormat
        extract_format_map = {
            "WIKI": wikipediaapi.ExtractFormat.WIKI,
            "HTML": wikipediaapi.ExtractFormat.HTML,
        }
        self.user_agent = self.config.config.get("user_agent", "uTu-agent")
        self.language = self.config.config.get("language", "en")
        self.content_type = self.config.config.get("content_type", "text")
        extract_format = self.config.config.get("extract_format", "WIKI")
        if extract_format not in extract_format_map:
            raise ValueError("Invalid extract_format. Choose between 'WIKI' or 'HTML'.")
        self.extract_format = extract_format_map[extract_format]

        self.wiki = wikipediaapi.Wikipedia(
            user_agent=self.user_agent, language=self.language, extract_format=self.extract_format
        )

    @register_tool
    async def wikipedia_search(self, query: str) -> str:
        """Searches Wikipedia and returns a summary or full text of the given topic, along with the page URL.

        Args:
            query (str): The topic to search on Wikipedia.
        """
        try:
            page = self.wiki.page(query)

            if not page.exists():
                return f"No Wikipedia page found for '{query}'. Try a different query."

            title = page.title
            url = page.fullurl

            if self.content_type == "summary":
                text = page.summary
            elif self.content_type == "text":
                text = page.text
            else:
                return "⚠️ Invalid `content_type`. Use either 'summary' or 'text'."

            return f"✅ **Wikipedia Page:** {title}\n\n**Content:** {text}\n\n🔗 **Read more:** {url}"

        except Exception as e:  # pylint: disable=broad-except
            return f"Error fetching Wikipedia summary: {str(e)}"

    @register_tool
    async def search_wikipedia_revisions(self, entity: str, year: int, month: int) -> str:
        """Get the revisions of a Wikipedia entity in a given month, return the revision url.

        Args:
            entity: the name of the Wikipedia entity, e.g. "Penguin"
            year: the year, e.g. 2022
            month: the month, e.g. 12
        """
        base_url = "https://en.wikipedia.org/w/api.php"

        start_date = datetime.datetime(year, month, 1)
        last_day = calendar.monthrange(year, month)[1]
        end_date = datetime.datetime(year, month, last_day, 23, 59, 59)
        start_iso = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "action": "query",
            "format": "json",
            "titles": entity,
            "prop": "revisions",
            "rvlimit": "max",
            "rvstart": start_iso,
            "rvend": end_iso,
            "rvdir": "newer",
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
        except requests.RequestException as e:
            return f"Request error: {e}"

        data = response.json()
        revisions_list = []
        pages = data.get("query", {}).get("pages", {})
        for _, page in pages.items():
            if "revisions" in page:
                for rev in page["revisions"]:
                    oldid = rev["revid"]
                    timestamp = rev["timestamp"]
                    # Construct the revision url
                    rev_url = f"https://en.wikipedia.org/w/index.php?title={entity}&oldid={oldid}"
                    revisions_list.append({"timestamp": timestamp, "oldid": oldid, "url": rev_url})
        return str(revisions_list)
