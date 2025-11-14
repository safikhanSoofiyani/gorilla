import os
import random
import time
from typing import Optional
from urllib.parse import urlparse

import html2text
import requests
from bs4 import BeautifulSoup

ERROR_TEMPLATES = [
    "503 Server Error: Service Unavailable for url: {url}",
    "429 Client Error: Too Many Requests for url: {url}",
    "403 Client Error: Forbidden for url: {url}",
    (
        "HTTPSConnectionPool(host='{host}', port=443): Max retries exceeded with url: {path} "
        "(Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x{id1:x}>, "
        "'Connection to {host} timed out. (connect timeout=5)'))"
    ),
    "HTTPSConnectionPool(host='{host}', port=443): Read timed out. (read timeout=5)",
    (
        "Max retries exceeded with url: {path} "
        "(Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x{id2:x}>: "
        "Failed to establish a new connection: [Errno -2] Name or service not known'))"
    ),
]


class WebSearchAPI:
    def __init__(self):
        self._api_description = (
            "This tool belongs to the Web Search API category. "
            "It provides functions to search the web and browse search results."
        )
        self.show_snippet = True
        # Random generators (kept for compatibility with your error simulation)
        self._random = random.Random(337)
        self._rng = random.Random(1053)

    def _load_scenario(self, initial_config: dict, long_context: bool = False):
        # We don't care about the long_context parameter here
        self.show_snippet = initial_config["show_snippet"]

    def search_engine_query(
        self,
        keywords: str,
        max_results: Optional[int] = 10,
        region: Optional[str] = "wt-wt",
    ) -> list:
        """
        This function queries the search engine for the provided keywords and region,
        using the Serper API under the hood.

        Args:
            keywords (str): The keywords to search for.
            max_results (int, optional): The maximum number of search results to return. Defaults to 10.
            region (str, optional): The region to search in. Defaults to "wt-wt".

        Returns:
            list: A list of search result dictionaries, each containing:
            - 'title' (str): The title of the search result.
            - 'href' (str): The URL of the search result.
            - 'body' (str): A brief description or snippet from the search result.

            Or, on error:
            - {'error': '<error message>'}
        """
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return {
                "error": "SERPER_API_KEY environment variable is not set. "
                         "Please set it to your Serper API key."
            }

        backoff = 2  # initial back-off in seconds

        # Serper basic payload
        payload = {"q": keywords}

        # Best-effort use of `region` as Serper's `location` parameter
        # (You can map your region codes to human-readable locations if you like.)
        if region and region != "wt-wt":
            payload["location"] = region

        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }

        # Infinite retry loop with exponential backoff for 429
        while True:
            try:
                response = requests.post(
                    "https://google.serper.dev/search",
                    headers=headers,
                    json=payload,
                    timeout=20,
                )
            except Exception as e:
                # Network / transport-level error
                error_block = (
                    "*" * 100
                    + f"\n❗️❗️ [WebSearchAPI] Error calling Serper API: {str(e)}. "
                      "This is not a rate-limit error, so it will not be retried."
                    + "*" * 100
                )
                print(error_block)
                return {"error": f"Error calling Serper API: {str(e)}"}

            # Handle HTTP 429 with exponential backoff
            if response.status_code == 429:
                wait_time = backoff + random.uniform(0, backoff)
                error_block = (
                    "*" * 100
                    + "\n❗️❗️ [WebSearchAPI] Received 429 from Serper API. "
                      "The number of requests sent using this API key exceeds your rate/budget. "
                      f"Retrying in {wait_time:.1f} seconds…"
                    + "\n" + "*" * 100
                )
                print(error_block)
                time.sleep(wait_time)
                backoff = min(backoff * 2, 120)  # cap the back-off
                continue

            # Any other non-success response: return as error, no retry
            if not response.ok:
                return {
                    "error": f"Serper API returned HTTP {response.status_code}: {response.text}"
                }

            # Successful response
            try:
                search_results = response.json()
            except ValueError as e:
                return {"error": f"Failed to parse JSON from Serper API: {str(e)}"}

            break

        # Serper returns organic results under the 'organic' key
        if "organic" not in search_results:
            return {
                "error": "Failed to retrieve the search results from server. Please try again later."
            }

        organic_results = search_results["organic"]

        # Convert to your existing format:
        #   [{'title': ..., 'href': ..., 'body': ...}, ...]
        results = []
        for result in organic_results[:max_results]:
            title = result.get("title", "")
            link = result.get("link", "")
            snippet = result.get("snippet", "")

            if self.show_snippet:
                results.append(
                    {
                        "title": title,
                        "href": link,
                        "body": snippet,
                    }
                )
            else:
                results.append(
                    {
                        "title": title,
                        "href": link,
                    }
                )

        return results

    def fetch_url_content(self, url: str, mode: str = "raw") -> str:
        """
        This function retrieves content from the provided URL and processes it based on the selected mode.

        Args:
            url (str): The URL to fetch content from. Must start with 'http://' or 'https://'.
            mode (str, optional): The mode to process the fetched content. Defaults to "raw".
                Supported modes are:
                    - "raw": Returns the raw HTML content.
                    - "markdown": Converts raw HTML content to Markdown format for better readability, using html2text.
                    - "truncate": Extracts and cleans text by removing scripts, styles, and extraneous whitespace.
        """
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {url}")

        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/112.0.0.0 Safari/537.36"
                ),
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Referer": "https://www.google.com/",
                "Sec-Fetch-Site": "same-origin",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-User": "?1",
                "Sec-Fetch-Dest": "document",
            }
            response = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
            response.raise_for_status()

            if mode == "raw":
                return {"content": response.text}

            elif mode == "markdown":
                converter = html2text.HTML2Text()
                markdown = converter.handle(response.text)
                return {"content": markdown}

            elif mode == "truncate":
                soup = BeautifulSoup(response.text, "html.parser")

                # Remove scripts and styles
                for script_or_style in soup(["script", "style"]):
                    script_or_style.extract()

                # Extract and clean text
                text = soup.get_text(separator="\n", strip=True)
                return {"content": text}
            else:
                raise ValueError(f"Unsupported mode: {mode}")

        except Exception as e:
            return {"error": f"An error occurred while fetching {url}: {str(e)}"}

    def _fake_requests_get_error_msg(self, url: str) -> str:
        """
        Return a realistic-looking requests/urllib3 error message.
        """
        parsed = urlparse(url)

        context = {
            "url": url,
            "host": parsed.hostname or "unknown",
            "path": parsed.path or "/",
            "id1": self._rng.randrange(0x10000000, 0xFFFFFFFF),
            "id2": self._rng.randrange(0x10000000, 0xFFFFFFFF),
        }

        template = self._rng.choice(ERROR_TEMPLATES)

        return template.format(**context)
