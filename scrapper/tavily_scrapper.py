
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

APPROVED_DOMAINS = ("who.int", "cdc.gov", "nih.gov", "nice.org.uk", "medlineplus.gov", "fda.gov", "cancer.gov", "health.gov", "healthfinder.gov", "clinicaltrials.gov", "nhs.uk", "gov.uk", "gov.scot", "canada.ca", "phac-aspc.gc.ca", "healthlinkbc.ca", "health.gov.au", "healthdirect.gov.au", "betterhealth.vic.gov.au", "safetyandquality.gov.au", "mohfw.gov.in", "icmr.gov.in", "ncdc.mohfw.gov.in", "nhm.gov.in", "aiims.edu", "ema.europa.eu", "ecdc.europa.eu", "rki.de", "sign.ac.uk", "diabetes.org", "heart.org", "cancer.org", "aafp.org", "aap.org", "acog.org", "acc.org", "kidney.org", "cochrane.org")

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


def extract_clean_text(element) -> str:
    """Extract readable text while preserving boundaries between HTML nodes."""

    parts = []

    for text_node in element.find_all(string=True):
        text = str(text_node)

        # Ignore empty or whitespace-only nodes.
        if not text.strip():
            continue

        parts.append(text.strip())

    return clean_paragraph(" ".join(parts))


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
    """Score a passage using its text, title, and section heading."""
    paragraph_terms = {
        normalize_word(word)
        for word in re.findall(r"[a-zA-Z]+", paragraph.lower())
    }
    title_terms = {
        normalize_word(word)
        for word in re.findall(r"[a-zA-Z]+", title.lower())
    }
    heading_terms = {
        normalize_word(word)
        for word in re.findall(r"[a-zA-Z]+", heading.lower())
    }

    matched_terms = query_terms.intersection(paragraph_terms)
    title_matches = query_terms.intersection(title_terms)
    heading_matches = query_terms.intersection(heading_terms)

    # A short list item can be relevant because it belongs to a matching
    # section, even if the item itself does not repeat the query keywords.
    if not matched_terms and not title_matches and not heading_matches:
        return 0.0

    score = len(matched_terms) / len(query_terms)

    if title_matches:
        score += 0.20

    if heading_matches:
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
    """Extract complete content sections, keeping introductions and lists together."""
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
    seen_sections = set()

    current_heading = ""
    section_parts = []
    section_order = 0

    useful_sections = (
        "overview", "symptom", "cause", "prevention", "treatment",
        "diagnosis", "risk factor", "when to see", "what you can do",
    )
    unwanted_sections = (
        "references", "bibliography", "acknowledgments",
        "acknowledgements", "citations", "myths and misconceptions",
    )

    # Map common question wording to the corresponding section headings.
    requested_sections = {
        "symptom": ("symptom", "signs", "clinical presentation"),
        "cause": ("cause", "why it happens"),
        "treatment": ("treatment", "manage", "management", "therapy"),
        "prevention": ("prevention", "prevent", "reduce the risk"),
        "diagnosis": ("diagnosis", "diagnosed", "tests"),
        "risk": ("risk factor", "higher risk", "more likely"),
    }

    query_lower = query.lower()
    requested_facets = [
        facet
        for facet, phrases in requested_sections.items()
        if any(phrase in query_lower for phrase in phrases)
    ]

    def flush_section() -> None:
        """Create one candidate for the current heading and all its content."""
        nonlocal section_parts, section_order

        if not section_parts:
            return

        heading_lower = current_heading.lower()
        if any(section in heading_lower for section in unwanted_sections):
            section_parts = []
            return

        body = "\n".join(section_parts).strip()
        if not body:
            section_parts = []
            return

        score = calculate_relevance(
            paragraph=body,
            query_terms=query_terms,
            title=title,
            heading=current_heading,
        )

        # Section headings are useful context, but should not by themselves
        # make unrelated sections relevant.
        if any(section in heading_lower for section in useful_sections):
            score += 0.30

        # Explicitly prioritize sections that answer the requested facet.
        for facet in requested_facets:
            if any(
                phrase in heading_lower
                for phrase in requested_sections[facet]
            ):
                score += 1.25
                break

        if score > 0:
            passage = body
            if current_heading and current_heading.casefold() not in body.casefold():
                passage = f"{current_heading}: {body}"

            key = passage.casefold()
            if key not in seen_sections:
                seen_sections.add(key)
                candidates.append({
                    "text": passage,
                    "score": score,
                    "order": section_order,
                    "heading": current_heading,
                })
                section_order += 1

        section_parts = []

    for element in main.find_all(
        ["h1", "h2", "h3", "h4", "p", "li", "blockquote"]
    ):
        if element.name in {"h1", "h2", "h3", "h4"}:
            flush_section()
            current_heading = extract_clean_text(element)
            continue

        if element.name == "li":
            # Extract the current list item's own text. Nested list items are
            # visited separately, preventing parent/child duplication.
            item_copy = BeautifulSoup(str(element), "html.parser")
            item_node = item_copy.find("li")
            if item_node is None:
                continue

            for nested_list in item_node.find_all(["ul", "ol"]):
                nested_list.decompose()

            item_text = extract_clean_text(item_node)

            if len(item_text) >= 8 and not is_boilerplate(item_text):
                section_parts.append(f"• {item_text}")
            continue

        paragraph = extract_clean_text(element)

        if len(paragraph) >= MIN_PARAGRAPH_CHARACTERS:
            if not is_boilerplate(paragraph):
                section_parts.append(paragraph)

    flush_section()
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
    Select complete, relevant passages within the configured limits.

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

    # Rank complete sections by relevance, preferring earlier sections on ties.
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
