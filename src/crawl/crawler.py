"""
Lab 1 - Jazz Knowledge Graph: Wikipedia Crawler
================================================
Crawls Wikipedia Jazz pages respecting robots.txt, extracts clean text
using trafilatura, and stores results as JSONL.

Usage:
    python src/crawl/crawler.py

Output:
    data/crawler_output.jsonl
"""

import json
import logging
import time
import urllib.parse
import urllib.request
import urllib.robotparser
from pathlib import Path
from typing import Optional

import requests
import trafilatura

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "crawler_output.jsonl"

MIN_WORD_COUNT = 500
REQUEST_DELAY_SECONDS = 1.5        # polite crawl delay between requests
MAX_PAGES_TOTAL = 40               # hard ceiling — 40 pages is well above the ≥20 requirement
MAX_PAGES_PER_SEED_LIST = 15       # max artist/album links followed from a list page
USER_AGENT = "JazzKGBot/1.0 (academic research project; ESILV engineering school; https://en.wikipedia.org/wiki/Special:MyTalk)"
WIKIPEDIA_BASE = "https://en.wikipedia.org"

# Seed URLs: entry points for the crawl
SEED_URLS = [
    "https://en.wikipedia.org/wiki/List_of_jazz_musicians",
    "https://en.wikipedia.org/wiki/List_of_jazz_albums",
    "https://en.wikipedia.org/wiki/Miles_Davis",
    "https://en.wikipedia.org/wiki/John_Coltrane",
    "https://en.wikipedia.org/wiki/Duke_Ellington",
    "https://en.wikipedia.org/wiki/Louis_Armstrong",
    "https://en.wikipedia.org/wiki/Charlie_Parker",
    "https://en.wikipedia.org/wiki/Thelonious_Monk",
    "https://en.wikipedia.org/wiki/Bill_Evans",
    "https://en.wikipedia.org/wiki/Kind_of_Blue",
]

# Extra artist/album pages to ensure good coverage even if list pages are sparse
EXTRA_SEED_URLS = [
    "https://en.wikipedia.org/wiki/Herbie_Hancock",
    "https://en.wikipedia.org/wiki/Wayne_Shorter",
    "https://en.wikipedia.org/wiki/Dizzy_Gillespie",
    "https://en.wikipedia.org/wiki/Sonny_Rollins",
    "https://en.wikipedia.org/wiki/Art_Blakey",
    "https://en.wikipedia.org/wiki/Chet_Baker",
    "https://en.wikipedia.org/wiki/Dave_Brubeck",
    "https://en.wikipedia.org/wiki/Ornette_Coleman",
    "https://en.wikipedia.org/wiki/Charles_Mingus",
    "https://en.wikipedia.org/wiki/Blue_Note_Records",
    "https://en.wikipedia.org/wiki/Columbia_Records",
    "https://en.wikipedia.org/wiki/Prestige_Records",
    "https://en.wikipedia.org/wiki/Impulse!_Records",
    "https://en.wikipedia.org/wiki/A_Love_Supreme",
    "https://en.wikipedia.org/wiki/Time_Out_(Dave_Brubeck_Quartet_album)",
    "https://en.wikipedia.org/wiki/Giant_Steps_(album)",
    "https://en.wikipedia.org/wiki/Bitches_Brew",
    "https://en.wikipedia.org/wiki/Head_Hunters",
    "https://en.wikipedia.org/wiki/The_Black_Saint_and_the_Sinner_Lady",
    "https://en.wikipedia.org/wiki/New_Orleans_jazz",
    "https://en.wikipedia.org/wiki/Bebop",
    "https://en.wikipedia.org/wiki/Cool_jazz",
    "https://en.wikipedia.org/wiki/Free_jazz",
    "https://en.wikipedia.org/wiki/Hard_bop",
    "https://en.wikipedia.org/wiki/Modal_jazz",
    "https://en.wikipedia.org/wiki/Jazz_fusion",
    "https://en.wikipedia.org/wiki/Ella_Fitzgerald",
    "https://en.wikipedia.org/wiki/Billie_Holiday",
    "https://en.wikipedia.org/wiki/Nina_Simone",
    "https://en.wikipedia.org/wiki/Sarah_Vaughan",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("jazz_crawler")


# ---------------------------------------------------------------------------
# Robots.txt cache
# ---------------------------------------------------------------------------

class RobotsCache:
    """Cache robots.txt parsers per domain to avoid repeated fetches."""

    def __init__(self, user_agent: str) -> None:
        self._user_agent = user_agent
        self._parsers: dict[str, urllib.robotparser.RobotFileParser] = {}

    def is_allowed(self, url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        if domain not in self._parsers:
            rp = urllib.robotparser.RobotFileParser()
            robots_url = f"{domain}/robots.txt"
            try:
                # Fetch robots.txt with proper User-Agent header (required by Wikipedia)
                resp = requests.get(
                    robots_url,
                    headers={"User-Agent": self._user_agent},
                    timeout=10,
                )
                rp.parse(resp.text.splitlines())
                logger.debug("Loaded robots.txt from %s", robots_url)
            except Exception as exc:
                logger.warning("Could not fetch robots.txt for %s: %s", domain, exc)
                # If robots.txt is unreachable, allow by default (conservative)
                self._parsers[domain] = rp
                return True
            self._parsers[domain] = rp
        return self._parsers[domain].can_fetch(self._user_agent, url)


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def is_wikipedia_article(url: str) -> bool:
    """Return True if the URL looks like a standard Wikipedia article page."""
    parsed = urllib.parse.urlparse(url)
    if "wikipedia.org" not in parsed.netloc:
        return False
    path = parsed.path
    # Must be a /wiki/ path but not a special namespace
    if not path.startswith("/wiki/"):
        return False
    title = path[len("/wiki/"):]
    excluded_prefixes = (
        "Special:", "Talk:", "User:", "Wikipedia:", "File:", "Help:",
        "Category:", "Portal:", "Template:", "Draft:", "MediaWiki:",
        "Module:", "Book:", "TimedText:",
    )
    return not any(title.startswith(p) for p in excluded_prefixes)


def normalise_url(url: str) -> str:
    """Strip fragment identifiers and normalise to https."""
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.urlunparse(parsed._replace(fragment="", scheme="https"))


def extract_wiki_links(html: str, base_url: str) -> list[str]:
    """
    Extract /wiki/ links from raw HTML using simple string scanning.
    Avoids a full BeautifulSoup dependency for link extraction.
    """
    links: list[str] = []
    search = 'href="/wiki/'
    start = 0
    while True:
        idx = html.find(search, start)
        if idx == -1:
            break
        end = html.find('"', idx + len(search))
        if end == -1:
            break
        path = html[idx + len('href="'):end]
        full_url = WIKIPEDIA_BASE + path
        links.append(full_url)
        start = end
    return links


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def fetch_html(url: str, session: requests.Session) -> Optional[str]:
    """Fetch raw HTML for a URL, returning None on failure."""
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        logger.warning("HTTP error fetching %s: %s", url, exc)
        return None


def extract_text(html: str, url: str) -> Optional[str]:
    """Use trafilatura to extract clean article text from HTML."""
    try:
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        return text
    except Exception as exc:
        logger.warning("trafilatura extraction failed for %s: %s", url, exc)
        return None


def extract_title(html: str) -> str:
    """Extract page title from the <title> tag."""
    tag_open = html.find("<title>")
    tag_close = html.find("</title>")
    if tag_open == -1 or tag_close == -1:
        return ""
    raw = html[tag_open + 7: tag_close].strip()
    # Wikipedia titles look like "Miles Davis - Wikipedia"
    return raw.replace(" - Wikipedia", "").replace(" – Wikipedia", "").strip()


# ---------------------------------------------------------------------------
# Core crawler
# ---------------------------------------------------------------------------

class JazzCrawler:
    """Recursive Wikipedia crawler focused on jazz pages."""

    def __init__(self, output_path: Path = OUTPUT_PATH) -> None:
        self.output_path = output_path
        self.robots = RobotsCache(USER_AGENT)
        self.visited: set[str] = set()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._page_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Starting Jazz Wikipedia crawl. Output: %s", self.output_path)

        # Open output file fresh each run
        with open(self.output_path, "w", encoding="utf-8") as fout:
            self._fout = fout

            # Phase 1: crawl direct seed pages (artists, albums, sub-genres)
            direct_seeds = SEED_URLS[2:] + EXTRA_SEED_URLS  # skip list pages for now
            for url in direct_seeds:
                if self._page_count >= MAX_PAGES_TOTAL:
                    break
                self._crawl_page(url)
                time.sleep(REQUEST_DELAY_SECONDS)

            # Phase 2: expand from list pages
            list_seeds = SEED_URLS[:2]
            for list_url in list_seeds:
                if self._page_count >= MAX_PAGES_TOTAL:
                    break
                self._crawl_list_page(list_url)

        logger.info(
            "Crawl complete. %d pages stored in %s",
            self._page_count,
            self.output_path,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _crawl_page(self, url: str) -> bool:
        """
        Fetch, extract, and store a single Wikipedia page.
        Returns True if the page was successfully stored.
        """
        url = normalise_url(url)
        if url in self.visited:
            logger.debug("Already visited: %s", url)
            return False
        self.visited.add(url)

        if not is_wikipedia_article(url):
            logger.debug("Skipping non-article URL: %s", url)
            return False

        if not self.robots.is_allowed(url):
            logger.warning("robots.txt disallows: %s", url)
            return False

        logger.info("[%d] Fetching: %s", self._page_count + 1, url)
        html = fetch_html(url, self.session)
        if html is None:
            return False

        text = extract_text(html, url)
        if text is None:
            logger.debug("No text extracted from %s", url)
            return False

        word_count = len(text.split())
        if word_count < MIN_WORD_COUNT:
            logger.debug("Too short (%d words), skipping: %s", word_count, url)
            return False

        title = extract_title(html)
        record = {
            "url": url,
            "title": title,
            "text": text,
            "word_count": word_count,
        }
        self._fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fout.flush()
        self._page_count += 1
        logger.info("  Stored '%s' (%d words)", title, word_count)
        return True

    def _crawl_list_page(self, list_url: str) -> None:
        """
        Fetch a list page (e.g. List_of_jazz_musicians), extract article links,
        and crawl up to MAX_PAGES_PER_SEED_LIST of them.
        """
        list_url = normalise_url(list_url)
        if not self.robots.is_allowed(list_url):
            logger.warning("robots.txt disallows list page: %s", list_url)
            return

        logger.info("Expanding list page: %s", list_url)
        html = fetch_html(list_url, self.session)
        if html is None:
            return
        time.sleep(REQUEST_DELAY_SECONDS)

        links = extract_wiki_links(html, list_url)
        unique_links: list[str] = []
        seen_in_list: set[str] = set()
        for link in links:
            norm = normalise_url(link)
            if norm not in seen_in_list and norm not in self.visited:
                if is_wikipedia_article(norm):
                    seen_in_list.add(norm)
                    unique_links.append(norm)

        logger.info("  Found %d candidate links on list page", len(unique_links))
        crawled = 0
        for link in unique_links:
            if self._page_count >= MAX_PAGES_TOTAL:
                break
            if crawled >= MAX_PAGES_PER_SEED_LIST:
                break
            success = self._crawl_page(link)
            if success:
                crawled += 1
            time.sleep(REQUEST_DELAY_SECONDS)

        logger.info("  Crawled %d pages from list page", crawled)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    crawler = JazzCrawler(output_path=OUTPUT_PATH)
    crawler.run()

    # Summary statistics
    count = 0
    total_words = 0
    try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                count += 1
                total_words += record.get("word_count", 0)
    except FileNotFoundError:
        logger.error("Output file not found: %s", OUTPUT_PATH)
        return

    logger.info("--- Summary ---")
    logger.info("Total pages: %d", count)
    logger.info("Total words: %d", total_words)
    if count:
        logger.info("Average words per page: %.0f", total_words / count)


if __name__ == "__main__":
    main()
