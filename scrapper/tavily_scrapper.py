
import os
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

# --------------------------------------------------
# Configuration
# --------------------------------------------------

APPROVED_DOMAINS = (
    "who.int",
    "cdc.gov",
    "nih.gov",
    "nice.org.uk",
)

# Exclude research-publication hosts from this initial
# patient-health guidance collection.
EXCLUDED_HOSTS = (
    "pmc.ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
)

USER_AGENT = "CareRoot/1.0 (Healthcare Research Bot)"
REQUEST_TIMEOUT = 15

MAX_SELECTED_CHARACTERS = 3000
MAX_SELECTED_PARAGRAPHS = 5
MIN_PARAGRAPH_CHARACTERS = 40
MIN_ARTICLE_CHARACTERS = 150

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "can", "do", "does", "for", "from", "how", "i",
    "in", "is", "it", "me", "of", "on", "or", "the",
    "to", "was", "what", "when", "which", "who", "why",
    "with", "you", "your",
}

SYNONYM_GROUPS = (
    {"cavity", "cavities", "decay", "caries"},
    {"tooth", "teeth", "dental"},
)

BOILERPLATE_PHRASES = {
    "on this page",
    "back to top",
    "skip to main content",
    "skip to content",
    "related resources",
    "related publications",
    "share this page",
    "print this page",
    "read more",
    "subscribe",
    "cookie settings",
}


# --------------------------------------------------
# URL and source validation
# --------------------------------------------------

def is_approved_url(url: str) -> bool:
    """Allow HTTPS URLs hosted on approved domains."""
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower().rstrip(".")

        return (
            parsed.scheme == "https"
            and any(
                hostname == domain or hostname.endswith("." + domain)
                for domain in APPROVED_DOMAINS
            )
        )
    except (TypeError, ValueError):
        return False


def is_suitable_guidance_url(url: str) -> bool:
    """Reject known research-publication hosts."""
    try:
        hostname = (urlparse(url).hostname or "").lower().rstrip(".")

        if not is_approved_url(url):
            return False

        for excluded_host in EXCLUDED_HOSTS:
            if (
                hostname == excluded_host
                or hostname.endswith("." + excluded_host)
            ):
                return False

        return True

    except (TypeError, ValueError):
        return False


# --------------------------------------------------
# Text cleaning
# --------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize whitespace without joining separate words."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()



def clean_paragraph(text: str) -> str:
    """Normalize whitespace and remove common webpage citation markers."""
    text = re.sub(r"\s+", " ", text).strip()

    # Remove superscript-style numeric citation markers at sentence ends.
    text = re.sub(r"(?<=[.!?])\s+\d{1,3}(?=\s|$)", "", text)

    return text.strip()



def is_boilerplate(text: str) -> bool:
    """Reject navigation text, references, and bibliography entries."""
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    normalized = normalized.rstrip(" .:")

    boilerplate_phrases = {
        "on this page",
        "back to top",
        "skip to main content",
        "skip to content",
        "related resources",
        "related publications",
        "share this page",
        "print this page",
        "read more",
        "subscribe",
        "cookie settings",
        "references",
        "references and resources",
        "bibliography",
        "acknowledgments",
        "acknowledgements",
    }

    if normalized in boilerplate_phrases:
        return True

    if normalized.startswith(("skip to ", "back to top")):
        return True

    # Reject paragraphs that are clearly bibliographic citations.
    if re.match(
        r"^(references?|bibliography|sources)\s*:",
        normalized,
    ):
        return True

    if re.search(r"https?://\S+", normalized) and re.search(
        r"\b(accessed|retrieved|available at|cited)\b",
        normalized,
    ):
        return True

    return False



# --------------------------------------------------
# Query relevance
# --------------------------------------------------

def normalize_word(word: str) -> str:
    """Normalize common plural forms and irregular dental terms."""
    word = word.lower()

    irregular_forms = {
        "teeth": "tooth",
        "cavities": "cavity",
        "children": "child",
        "people": "person",
    }

    if word in irregular_forms:
        return irregular_forms[word]

    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"

    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]

    return word


def expand_query_terms(query: str) -> set[str]:
    """Expand query keywords with a small, predefined synonym list."""
    words = {
        normalize_word(word)
        for word in re.findall(r"[a-zA-Z]+", query.lower())
        if word.lower() not in STOPWORDS
    }

    expanded = set(words)

    for group in SYNONYM_GROUPS:
        normalized_group = {normalize_word(word) for word in group}

        if words.intersection(normalized_group):
            expanded.update(normalized_group)

    return expanded


def calculate_relevance(
    paragraph: str,
    query_terms: set[str],
    title: str,
    heading: str,
) -> float:
    """Score a paragraph based on matching query terms."""
    paragraph_terms = {
        normalize_word(word)
        for word in re.findall(r"[a-zA-Z]+", paragraph.lower())
    }

    if not query_terms:
        return 0.0

    matched_terms = query_terms.intersection(paragraph_terms)

    # A passage must match at least one query term.
    if not matched_terms:
        return 0.0

    score = len(matched_terms) / len(query_terms)

    title_terms = {
        normalize_word(word)
        for word in re.findall(r"[a-zA-Z]+", title.lower())
    }

    heading_terms = {
        normalize_word(word)
        for word in re.findall(r"[a-zA-Z]+", heading.lower())
    }

    if query_terms.intersection(title_terms):
        score += 0.20

    if query_terms.intersection(heading_terms):
        score += 0.15

    return score


# --------------------------------------------------
# Extract readable paragraphs
# --------------------------------------------------

def extract_paragraph_candidates(
    soup: BeautifulSoup,
    query: str,
    title: str,
) -> list[dict]:
    """Extract relevant paragraphs with their section headings."""
    main = soup.find("article") or soup.find("main") or soup.body

    if main is None:
        return []

    for element in main.select(
        "script, style, nav, footer, header, aside, form, "
        "iframe, noscript, svg, button, "
        "[aria-hidden='true'], [role='navigation']"
    ):
        element.decompose()

    query_terms = expand_query_terms(query)

    candidates = []
    seen_paragraphs = set()
    current_heading = ""

    for element in main.find_all(
        ["h1", "h2", "h3", "h4", "p", "li", "blockquote"]
    ):
        if element.name in {"h1", "h2", "h3", "h4"}:
            current_heading = clean_paragraph(
                element.get_text(" ", strip=True)
            )
            continue

        # Avoid duplicate text when a list item contains paragraphs.
        if element.name == "li" and element.find("p"):
            continue

        paragraph = clean_paragraph(
            element.get_text(" ", strip=True)
        )

        if len(paragraph) < MIN_PARAGRAPH_CHARACTERS:
            continue

        if is_boilerplate(paragraph):
            continue

        paragraph_key = paragraph.casefold()

        if paragraph_key in seen_paragraphs:
            continue

        seen_paragraphs.add(paragraph_key)


        score = calculate_relevance(
            paragraph=paragraph,
            query_terms=query_terms,
            title=title,
            heading=current_heading,
        )

        # Prefer patient-health guidance sections.
        heading_lower = current_heading.lower()

        useful_sections = (
            "overview",
            "symptoms",
            "causes",
            "prevention",
            "treatment",
            "diagnosis",
            "risk factors",
            "when to see",
            "what you can do",
        )

        unwanted_sections = (
            "references",
            "bibliography",
            "acknowledgments",
            "acknowledgements",
            "citations",
            "myths and misconceptions",
        )

        if any(section in heading_lower for section in unwanted_sections):
            continue

        if any(section in heading_lower for section in useful_sections):
            score += 0.30


        if score <= 0:
            continue

        # Keep the heading with the passage so its meaning remains clear.
        if (
            current_heading
            and current_heading.casefold() not in paragraph.casefold()
        ):
            passage = f"{current_heading}: {paragraph}"
        else:
            passage = paragraph

        candidates.append({
            "text": passage,
            "score": score,
            "order": len(candidates),
        })

    return candidates


# --------------------------------------------------
# Select passages under strict limits
# --------------------------------------------------

def select_relevant_paragraphs(
    soup: BeautifulSoup,
    query: str,
    title: str,
) -> tuple[str, int]:
    """
    Select at most 3 complete passages and 1,000 characters.

    Passages that do not fit in the remaining budget are skipped,
    not cut off mid-word or mid-sentence.
    """
    candidates = extract_paragraph_candidates(
        soup=soup,
        query=query,
        title=title,
    )

    if not candidates:
        return "", 0

    # Rank by relevance, preferring earlier passages when scores tie.
    ranked = sorted(
        candidates,
        key=lambda item: (-item["score"], item["order"]),
    )

    selected = []
    used_characters = 0

    for candidate in ranked:
        if len(selected) >= MAX_SELECTED_PARAGRAPHS:
            break

        passage = candidate["text"]
        remaining = MAX_SELECTED_CHARACTERS - used_characters

        # Keep passages intact; do not slice a passage to fit.
        if len(passage) > remaining:
            continue

        selected.append({
            "text": passage,
            "order": candidate["order"],
        })

        used_characters += len(passage)

    # Restore original article order for readable output.
    selected.sort(key=lambda item: item["order"])

    selected_text = "\n\n".join(
        item["text"] for item in selected
    )

    return selected_text, len(selected)


# --------------------------------------------------
# Fetch and extract one article
# --------------------------------------------------

def extract_article(url: str, query: str) -> dict | None:
    """Fetch an approved guidance page and select relevant passages."""
    if not is_suitable_guidance_url(url):
        return None

    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        response.raise_for_status()

        # Validate the final URL after redirects.
        if not is_suitable_guidance_url(response.url):
            return None

        content_type = response.headers.get("Content-Type", "").lower()

        if "text/html" not in content_type:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        title = ""
        if soup.title:
            title = clean_paragraph(
                soup.title.get_text(" ", strip=True)
            )

        selected_text, selected_count = select_relevant_paragraphs(
            soup=soup,
            query=query,
            title=title,
        )

        if not selected_text:
            return None

        if len(selected_text) < MIN_ARTICLE_CHARACTERS:
            return None

        domain = urlparse(response.url).hostname or ""

        return {
            "text": selected_text,
            "metadata": {
                "source_url": response.url,
                "title": title or "Untitled healthcare guidance",
                "domain": domain,
                "document_type": "webpage",
                "search_query": query,
                "selected_paragraph_count": selected_count,
                "selected_characters": len(selected_text),
            },
        }

    except requests.RequestException:
        return None


# --------------------------------------------------
# Tavily search + extraction
# --------------------------------------------------

def search_and_scrape(
    query: str,
    max_results: int = 5,
) -> list[dict]:
    """Discover sources and return selected healthcare passages."""
    query = query.strip()

    if not query:
        raise ValueError("Search query cannot be empty.")

    if max_results < 1:
        raise ValueError("max_results must be at least 1.")

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY was not found in the .env file."
        )

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
        include_domains=list(APPROVED_DOMAINS),
    )

    articles = []
    seen_urls = set()

    for result in response.get("results", []):
        url = result.get("url", "")

        if not is_suitable_guidance_url(url):
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)

        article = extract_article(url=url, query=query)

        if article:
            articles.append(article)

    return articles


# --------------------------------------------------
# Standalone test
# --------------------------------------------------

if __name__ == "__main__":
    user_query = input(
        "Enter a healthcare topic to search: "
    ).strip()

    try:
        results = search_and_scrape(user_query)

        if not results:
            print(
                "\nNo suitable relevant passages were extracted "
                "from the approved healthcare sources."
            )
        else:
            for index, article in enumerate(results, start=1):
                print(f"\n{'=' * 60}")
                print(f"ARTICLE {index}")
                print("=" * 60)

                print("\nMETADATA:")
                print(article["metadata"])

                print("\nSELECTED RELEVANT PARAGRAPHS:")
                print(article["text"])

                print("\nCONTENT STATISTICS:")
                print(
                    f"Paragraphs: "
                    f"{article['metadata']['selected_paragraph_count']}"
                )
                print(
                    f"Characters: "
                    f"{article['metadata']['selected_characters']}"
                )

                # Defensive checks before any future ingestion.
                assert (
                    article["metadata"]["selected_paragraph_count"]
                    <= MAX_SELECTED_PARAGRAPHS
                )
                assert len(article["text"]) <= MAX_SELECTED_CHARACTERS

    except Exception as exc:
        print(f"\nScraper error: {exc}")
