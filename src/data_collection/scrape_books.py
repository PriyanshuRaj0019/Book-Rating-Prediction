from __future__ import annotations

import csv
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Tag
from requests import Response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
RAW_DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
OUTPUT_CSV_PATH: Final[Path] = RAW_DATA_DIR / "books.csv"
SOURCE_METADATA_PATH: Final[Path] = RAW_DATA_DIR / "source_metadata.json"

BASE_URL: Final[str] = "https://books.toscrape.com/"
CATALOGUE_URL: Final[str] = urljoin(BASE_URL, "catalogue/")
ROBOTS_URL: Final[str] = urljoin(BASE_URL, "robots.txt")

USER_AGENT: Final[str] = (
    "Mozilla/5.0 (compatible; PriyanshuRaj-MLProjectBot/1.0; "
    "+https://github.com/Priyanshu-INBT021231/DATA-SCIENCE_INBT021231)"
)

REQUEST_TIMEOUT_SECONDS: Final[int] = 20
REQUEST_DELAY_SECONDS: Final[float] = 1.0
MAX_PAGES: Final[int] = 50
EXPECTED_MIN_RECORDS: Final[int] = 500
EXPECTED_MAX_RECORDS: Final[int] = 1000

RATING_MAP: Final[dict[str, int]] = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}

CSV_COLUMNS: Final[list[str]] = [
    "title",
    "category",
    "price_gbp",
    "rating",
    "availability",
    "product_url",
]


@dataclass(frozen=True)
class BookRecord:
    title: str
    category: str
    price_gbp: float
    rating: int
    availability: str
    product_url: str


@dataclass(frozen=True)
class SourceMetadata:
    source_name: str
    base_url: str
    robots_url: str
    robots_txt_checked: bool
    terms_reviewed: bool
    allowed_for_scraping: bool
    collection_method: str
    requested_pages: int
    collected_records: int
    rate_limit_seconds: float
    limitations: list[str]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }
    )

    retry_strategy = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def is_scraping_allowed(user_agent: str, target_url: str) -> bool:
    parser = RobotFileParser()
    parser.set_url(ROBOTS_URL)

    try:
        parser.read()
    except Exception as exc:
        logging.warning(
            "Could not read robots.txt. Continuing only because this is an explicit scraping sandbox: %s",
            exc,
        )
        return True

    return parser.can_fetch(user_agent, target_url)


def fetch_html(session: requests.Session, url: str) -> str:
    response: Response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    if response.encoding is None:
        response.encoding = response.apparent_encoding

    return response.text


def build_listing_page_url(page_number: int) -> str:
    if page_number == 1:
        return BASE_URL

    return urljoin(CATALOGUE_URL, f"page-{page_number}.html")


def extract_rating(book_element: Tag) -> int:
    rating_element = book_element.select_one("p.star-rating")
    if rating_element is None:
        raise ValueError("Missing rating element")

    rating_classes = rating_element.get("class", [])
    rating_label = next((label for label in rating_classes if label in RATING_MAP), None)

    if rating_label is None:
        raise ValueError(f"Unknown rating classes: {rating_classes}")

    return RATING_MAP[rating_label]


def extract_price(book_element: Tag) -> float:
    price_element = book_element.select_one("p.price_color")
    if price_element is None:
        raise ValueError("Missing price element")

    price_text = price_element.get_text(strip=True)
    price_match = re.search(r"\d+(?:\.\d+)?", price_text)

    if price_match is None:
        raise ValueError(f"Could not parse price from text: {price_text!r}")

    return float(price_match.group())


def extract_title_and_url(book_element: Tag, page_url: str) -> tuple[str, str]:
    title_element = book_element.select_one("h3 a")
    if title_element is None:
        raise ValueError("Missing title element")

    title = title_element.get("title")
    href = title_element.get("href")

    if not isinstance(title, str) or not title.strip():
        raise ValueError("Missing book title")

    if not isinstance(href, str) or not href.strip():
        raise ValueError("Missing product URL")

    return title.strip(), urljoin(page_url, href)


def extract_availability(book_element: Tag) -> str:
    availability_element = book_element.select_one("p.instock.availability")
    if availability_element is None:
        raise ValueError("Missing availability element")

    return " ".join(availability_element.get_text(strip=True).split())


def extract_books_from_listing_page(html: str, page_url: str) -> list[BookRecord]:
    soup = BeautifulSoup(html, "lxml")
    category_element = soup.select_one(".breadcrumb li:nth-of-type(3) a")
    category = category_element.get_text(strip=True) if category_element else "Books"

    book_elements = soup.select("article.product_pod")
    records: list[BookRecord] = []

    for book_element in book_elements:
        title, product_url = extract_title_and_url(book_element, page_url)
        records.append(
            BookRecord(
                title=title,
                category=category,
                price_gbp=extract_price(book_element),
                rating=extract_rating(book_element),
                availability=extract_availability(book_element),
                product_url=product_url,
            )
        )

    return records


def scrape_books() -> list[BookRecord]:
    if not is_scraping_allowed(USER_AGENT, BASE_URL):
        raise PermissionError(f"robots.txt does not allow scraping: {BASE_URL}")

    session = build_session()
    records: list[BookRecord] = []

    for page_number in range(1, MAX_PAGES + 1):
        page_url = build_listing_page_url(page_number)
        logging.info("Fetching page %s: %s", page_number, page_url)

        html = fetch_html(session, page_url)
        page_records = extract_books_from_listing_page(html, page_url)

        if not page_records:
            logging.info("No records found on page %s. Stopping.", page_number)
            break

        records.extend(page_records)
        time.sleep(REQUEST_DELAY_SECONDS)

    if not EXPECTED_MIN_RECORDS <= len(records) <= EXPECTED_MAX_RECORDS:
        raise ValueError(
            f"Collected {len(records)} records. Expected between "
            f"{EXPECTED_MIN_RECORDS} and {EXPECTED_MAX_RECORDS}."
        )

    return records


def save_books_to_csv(records: list[BookRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def save_source_metadata(records_count: int, output_path: Path) -> None:
    metadata = SourceMetadata(
        source_name="Books to Scrape",
        base_url=BASE_URL,
        robots_url=ROBOTS_URL,
        robots_txt_checked=True,
        terms_reviewed=True,
        allowed_for_scraping=True,
        collection_method="BeautifulSoup HTML parsing with polite GET requests",
        requested_pages=MAX_PAGES,
        collected_records=records_count,
        rate_limit_seconds=REQUEST_DELAY_SECONDS,
        limitations=[
            "The website is a scraping sandbox, not a real commercial bookstore.",
            "Prices and ratings are randomly assigned and have no real business meaning.",
            "The dataset is suitable for educational ML classification practice only.",
            "Only public HTML pages were collected.",
            "No login, authentication, private user data, or restricted pages were accessed.",
        ],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(metadata), file, indent=2)


def main() -> None:
    configure_logging()
    records = scrape_books()
    save_books_to_csv(records, OUTPUT_CSV_PATH)
    save_source_metadata(len(records), SOURCE_METADATA_PATH)
    logging.info("Saved %s records to %s", len(records), OUTPUT_CSV_PATH)
    logging.info("Saved source metadata to %s", SOURCE_METADATA_PATH)


if __name__ == "__main__":
    main()
    