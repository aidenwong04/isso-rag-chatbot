import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

HEADING_TAGS = {"h2", "h3", "h4", "h5", "h6"}
BODY_TAGS = {"p", "ul", "ol", "dl", "li"}

TABLE_COMPONENT_CLASS = "paragraph--type--table"

# ISSO states a page's intended audience in its own breadcrumb, e.g.
# "Home > Employment > For Students > F-1 CPT" against
# "Home > Getting Started > For Scholars (Professors/Researchers) > ...".
# That matters because F-1 student, J-1 scholar and F-2/J-2 dependent rules
# differ on work authorization, travel and status, so a chunk retrieved for
# the wrong audience reads as a confident, correctly-cited, wrong answer.
# Taking the audience from the site's own navigation beats inferring it from
# URL patterns (which do not separate) or from embeddings (whose in-corpus
# and out-of-corpus score ranges overlap).
AUDIENCE_SEGMENT = re.compile(r"^For\s+([A-Za-z][A-Za-z\-]*)", re.IGNORECASE)

# Content that appears before the page's first h2 - a lead paragraph sitting
# outside any section - is collected here rather than dropped.
INTRO_LABEL = "(intro)"


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def get_heading_level(tag):
    # Two ways this site marks up a heading: a native <h2>-<h6>, or an
    # ARIA-only heading - a non-heading element (typically a <div>) carrying
    # role="heading" aria-level="N" instead of a real heading tag. Both are
    # treated as the same kind of thing; only the level number matters
    # downstream, not which markup produced it.
    if tag.name in HEADING_TAGS:
        return int(tag.name[1])
    if tag.get("role") == "heading":
        aria_level = tag.get("aria-level", "")
        if aria_level.isdigit():
            return int(aria_level)
    return None


def extract_breadcrumb(html_content):
    """The page's breadcrumb trail as a list of segments, outermost first.

    The breadcrumb lives outside #main-block, so this takes the whole
    document rather than the parsed content region.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    breadcrumb = soup.find(class_=re.compile("breadcrumb", re.IGNORECASE))
    if breadcrumb is None:
        return []

    items = [normalize(li.get_text(" ")) for li in breadcrumb.find_all("li")]
    if not items:
        items = [normalize(breadcrumb.get_text(" "))]
    return [item for item in items if item]


def audience_from_breadcrumb(segments):
    """The audience a breadcrumb names, or None when it names none.

    Returns None rather than assuming a default. A page whose breadcrumb is
    silent about audience has not been classified, and recording it as
    "students" would assert something the site never said - which is the
    exact failure this metadata exists to prevent. Roughly a third of pages
    reached by crawling are silent here, and they are not one group: some are
    cross-cutting utilities that belong in a student corpus, others are news
    and events that belong in no static index.
    """
    for segment in segments:
        match = AUDIENCE_SEGMENT.match(segment)
        if match:
            return match.group(1).lower()
    return None


def page_metadata(html_content):
    breadcrumb = extract_breadcrumb(html_content)
    return {
        "breadcrumb": breadcrumb,
        "audience": audience_from_breadcrumb(breadcrumb),
    }


def deobfuscate_emails(main):
    """Restore addresses hidden by Cloudflare's email-protection script.

    Cloudflare rewrites a mailto into <span class="__cf_email__"
    data-cfemail="HEX">[email&nbsp;protected]</span> and restores it with JS.
    Extracting text without decoding puts the literal "[email protected]" in
    the corpus, so the bot hands students a placeholder instead of an address
    - including the CBP Deferred Inspection contacts for a broken I-94.

    The hex is a XOR cipher: the first byte is the key, each remaining byte
    is one character of the address XORed with it.
    """
    for el in main.find_all(attrs={"data-cfemail": True}):
        hex_text = el["data-cfemail"]
        try:
            key = int(hex_text[:2], 16)
            address = "".join(
                chr(int(hex_text[i : i + 2], 16) ^ key)
                for i in range(2, len(hex_text), 2)
            )
        except ValueError:
            continue
        el.replace_with(address)


def is_table_data(tag):
    # Drupal's "table" component renders its data twice: a desktop copy held
    # as JSON in a rows="..." attribute (drawn by JS, so it has no parseable
    # HTML body) and a mobile fallback of one <dl> per row. The JSON is the
    # better source - it survives the fallback being dropped, and it keeps
    # the cell boundaries that flattening a <dl> to text throws away.
    return tag.has_attr("rows")


def is_heading_or_body(tag):
    return (
        get_heading_level(tag) is not None
        or tag.name in BODY_TAGS
        or is_table_data(tag)
    )


def parse_table_rows(tag):
    """Render a rows="...json..." attribute as one text line per table row.

    Cells arrive as HTML fragments ({"data": "<p>1.</p>", "class": [...]}),
    so each is parsed and flattened. Empty cells are dropped rather than
    left as blank columns - they're spacers, e.g. the empty corner cell that
    opens a header row.
    """
    try:
        rows = json.loads(tag["rows"])
    except (ValueError, TypeError):
        return []

    lines = []
    for row in rows:
        if not isinstance(row, list):
            continue
        cells = []
        for cell in row:
            html = cell.get("data", "") if isinstance(cell, dict) else str(cell)
            text = normalize(BeautifulSoup(html, "html.parser").get_text(" "))
            if text:
                cells.append(text)
        if cells:
            lines.append(" | ".join(cells))
    return lines


def superseded_fallback_rows(main):
    """The <dl> fallback rows made redundant by a sibling rows="..." attribute.

    Only within the same table component - a <dl> in a component that has no
    JSON copy is still the sole source for that data and is left alone.
    """
    superseded = set()
    for el in main.find_all(True):
        if TABLE_COMPONENT_CLASS not in el.get("class", []):
            continue
        if el.find(is_table_data) is None:
            continue
        superseded.update(id(dl) for dl in el.find_all("dl"))
    return superseded


def is_decorative_heading(tag):
    # sr-only headings (e.g. the breadcrumb's "You are here:") are the only
    # ones we can rule out purely by class - group-label headings sometimes
    # wrap real prose (e.g. "Visa Delays and Denials") and sometimes wrap
    # nothing but nav links (e.g. "On This Page"), so those are left in and
    # filtered out later if they end up with no captured body content.
    classes = tag.get("class", [])
    return "sr-only" in classes


def scrape_one_page(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    main = soup.find(id="main-block")
    if main is None:
        raise ValueError("could not find #main-block in page")

    deobfuscate_emails(main)

    elements = main.find_all(is_heading_or_body)
    superseded = superseded_fallback_rows(main)

    grouped_data = {}
    current_h2 = None

    for el in elements:
        level = get_heading_level(el)

        if level == 2:
            if is_decorative_heading(el):
                continue
            current_h2 = normalize(el.text)
            grouped_data.setdefault(current_h2, [])
            continue

        if current_h2 is None:
            current_h2 = INTRO_LABEL
            grouped_data.setdefault(current_h2, [])

        if level is not None:
            # nested under the current h2 as part of the same chunk. Tagged
            # by level ("h3"..."h6") rather than el.name so a native <h4> and
            # an ARIA role="heading" aria-level="4" dedupe against each other.
            heading_text = normalize(el.text)
            heading_tag = f"h{level}"
            items = grouped_data[current_h2]
            if items and items[-1]["tag"] == heading_tag and items[-1]["text"] == heading_text:
                # same heading text repeated back-to-back (e.g. an accordion
                # tab's own heading immediately followed by a nested
                # component's caption using the same label) - keep only the
                # first
                continue
            items.append({"tag": heading_tag, "text": heading_text})
            continue

        if is_table_data(el):
            for line in parse_table_rows(el):
                grouped_data[current_h2].append({"tag": "table", "text": f"- {line}"})
            continue

        if el.name == "dl" and id(el) in superseded:
            # same row, already taken from this component's JSON copy
            continue

        if el.name in {"ul", "ol"}:
            # handled item-by-item via their <li> children instead, so skip
            # the container itself to avoid double-counting its text
            continue

        if el.name == "dl":
            # Drupal's mobile-fallback rendering of a "table" component: each
            # <dl> is one row, holding two <dd>s (an index, then the row's
            # actual text) and no <dt>. Reached only when the component has
            # no rows="..." JSON copy to supersede it - flattening the <dd>s
            # to one string loses the cell boundaries, so it's the weaker of
            # the two sources and stays purely as a fallback.
            row_text = normalize(" ".join(dd.get_text(" ", strip=True) for dd in el.find_all("dd")))
            if row_text:
                grouped_data[current_h2].append({"tag": "dl", "text": f"- {row_text}"})
            continue

        if el.name == "li" and el.find_parent(["ul", "ol"]) is None:
            continue

        if el.name == "p" and el.find_parent("li") is not None:
            # already covered by the enclosing <li>
            continue

        text = normalize(el.text)
        if not text:
            continue
        if el.name == "li":
            text = f"- {text}"

        grouped_data[current_h2].append({"tag": el.name, "text": text})

    return {heading: items for heading, items in grouped_data.items() if items}


def print_grouped(grouped_data):
    for h2_title, items in grouped_data.items():
        print(f"★ H2: {h2_title}")
        for item in items:
            print(f"  - [{item['tag'].upper()}]: {item['text']}")


def flatten_to_chunks(grouped_data):
    # A chunk boundary is any heading (h2's dict key, or an h3-h6 item), not
    # just the top-level h2 - so "Eligibility" splits into one chunk per
    # sub-case ("Enrolled at Another School", "On OPT", ...) instead of one
    # chunk mixing all of them. Each chunk's heading is a breadcrumb of every
    # heading level above it, so the sub-chunk still carries its h2 context.
    chunks = []

    for h2_title, items in grouped_data.items():
        path = [h2_title]
        body = []

        def flush():
            if body:
                chunks.append({"heading": " > ".join(path), "text": "\n".join(body)})

        for item in items:
            tag = item["tag"]
            if tag in {"h3", "h4", "h5", "h6"}:
                flush()
                level = int(tag[1])
                # a same-or-shallower heading closes every deeper heading
                # that was open under it
                while len(path) > level - 2:
                    path.pop()
                path.append(item["text"])
                body = []
                continue
            body.append(item["text"])

        flush()

    return chunks


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


if __name__ == "__main__":
    raw_dir = DATA_DIR / "raw"
    out_dir = DATA_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    for html_path in sorted(raw_dir.glob("*.html")):
        html = html_path.read_text(encoding="utf-8")
        grouped = scrape_one_page(html)
        chunks = flatten_to_chunks(grouped)

        # Page-level, but written onto every chunk: retrieval works on chunks,
        # so a filter that has to look up the page first is a filter that gets
        # skipped.
        metadata = page_metadata(html)
        for chunk in chunks:
            chunk.update(metadata)

        out_path = out_dir / f"{html_path.stem}.json"
        out_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
        audience = metadata["audience"] or "unstated"
        print(
            f"{html_path.name}: {len(chunks)} chunks "
            f"[audience: {audience}] -> {out_path}"
        )
