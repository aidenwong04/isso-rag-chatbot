"""Detect when the ISSO corpus has drifted away from what isso.columbia.edu says.

Four checks, cheapest first, each one narrowing what the next has to look at:

  1. sitemap lastmod   - one request covers every page; tells us which pages
                         *claim* to have changed. Hearsay: a CMS timestamp can
                         move when nothing meaningful did, and can fail to move
                         when a shared block edited elsewhere changes the page.
                         So it selects candidates, it never clears a page.
  2. main-block hash   - authoritative about the bytes, but noisy at the page
                         level (nonces, cache-busted assets), so it is taken
                         over #main-block only and used purely as a trigger.
                         A false positive here costs one local re-parse.
  3. component markers - the set of Drupal `paragraph--type--*` values present.
                         The only check that looks at the input's *structure*
                         rather than the parser's output, so it is the only one
                         that can see a component type the parser has never met.
  4. coverage          - fraction of the page's prose blocks that survive into
                         chunks. The only check comparing raw against extracted
                         rather than extracted against extracted, which makes it
                         the only way to tell "ISSO deleted content" (raw and
                         extracted shrink together, coverage holds) apart from
                         "the parser went blind" (raw holds, extracted shrinks,
                         coverage craters). Those produce an identical chunk
                         diff and call for opposite responses.

Check 3 catches a component type that is new. Check 4 catches a known component
type that changed shape - the `paragraph--type--table` case, where the parser
reads a <dl> fallback that a theme upgrade could drop without renaming anything.
Neither check subsumes the other.

Read-only by default: reports and exits. Pass --apply to write refreshed HTML
into data/raw/ and update the baseline. Re-embedding is never automatic - it
costs API calls and it should follow a human reading this report.

Usage:
    python src/monitor.py                 # check pages whose lastmod moved
    python src/monitor.py --full          # ignore lastmod, check every page
    python src/monitor.py --apply         # also refresh data/raw/ and baseline
    python src/monitor.py --new-pages     # list sitemap URLs not in the corpus
"""

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
URL_MAP_PATH = DATA_DIR / "filename-to-url.json"
BASELINE_PATH = DATA_DIR / "state" / "baseline.json"

SITEMAP_URL = "https://isso.columbia.edu/sitemap.xml"
SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# robots.txt permits /content/ and sets no Crawl-delay, so this is politeness
# rather than compliance. Put a real contact address in CONTACT before running
# this unattended - an identifiable crawler is one a webmaster can ask about
# instead of block.
CONTACT = "hmw2156@columbia.edu"
USER_AGENT = f"isso-rag-chatbot/0.1 (student project; contact: {CONTACT})"
REQUEST_DELAY_SECONDS = 4.0

# Coverage is a ratio of noisy measurements, so it needs a tolerance band
# rather than an equality check. A drop this far below the stored baseline is
# treated as the parser losing ground, not as ISSO trimming a page.
COVERAGE_DROP_ALERT = 0.05
MIN_BLOCK_CHARS = 40


def load_parser():
    # one-page.py is not an importable module name (the hyphen makes it
    # invalid syntax for `import`), so it is loaded by path. Renaming it to
    # one_page.py would let this be a plain import.
    path = Path(__file__).resolve().parent / "one-page.py"
    spec = importlib.util.spec_from_file_location("one_page", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parser_module = load_parser()


def squash(text):
    """Compare on alphanumerics only.

    Whitespace and punctuation are exactly where two renderings of the same
    text disagree - get_text(" ") spaces out inline tag boundaries, so
    "<strong>yours</strong>." becomes "yours ." while the parser's own
    normalization does not. Comparing raw strings reports present text as
    missing.
    """
    return re.sub(r"[^a-z0-9]", "", text.lower())


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def get_main_block(html):
    main = BeautifulSoup(html, "html.parser").find(id="main-block")
    if main is None:
        raise ValueError("could not find #main-block in page")
    return main


def main_block_hash(html):
    """Fingerprint #main-block, not the whole document.

    Page chrome carries per-response noise - CSRF tokens, cache-busted asset
    URLs, build hashes - that would make a whole-page hash differ on every
    fetch and turn this check into a constant alarm.
    """
    main = get_main_block(html)
    for junk in main.find_all(["script", "style"]):
        junk.decompose()

    # Cloudflare re-encodes every protected address with a fresh random XOR
    # key on each response, so data-cfemail and the matching href fragment
    # differ on every fetch while the address behind them is unchanged.
    # Decoding first makes the fingerprint react to the address instead of
    # to the key.
    parser_module.deobfuscate_emails(main)
    for element in main.find_all(href=True):
        if "/cdn-cgi/l/email-protection" in element["href"]:
            del element["href"]

    # Whitespace between and around tags is serialization, not content: a page
    # saved through a browser is pretty-printed, while the same page fetched
    # over HTTP arrives as the server wrote it. Collapsing runs of whitespace
    # does not reconcile them, because the difference is a space against
    # nothing. Normalizing every text node does, and cannot hide a text change
    # since any node with actual characters survives.
    for node in list(main.find_all(string=True)):
        collapsed = re.sub(r"\s+", " ", str(node)).strip()
        if collapsed:
            node.replace_with(collapsed)
        else:
            node.extract()

    return hashlib.sha256(
        re.sub(r"\s+", " ", main.decode()).encode("utf-8")
    ).hexdigest()


def component_markers(html):
    markers = set()
    for element in get_main_block(html).find_all(True):
        for css_class in element.get("class", []):
            if css_class.startswith("paragraph--type--"):
                markers.add(css_class)
    return markers


def prose_blocks(main):
    blocks = []
    for element in main.find_all(["p", "li", "dd"]):
        if element.name == "p" and element.find_parent("li"):
            continue
        squashed = squash(element.get_text(" "))
        if len(squashed) >= MIN_BLOCK_CHARS:
            blocks.append(squashed)
    return blocks


def coverage(html, chunks):
    """Fraction of the page's prose blocks that reached the chunks.

    Blocks rather than characters: a character ratio double-counts the heading
    breadcrumb that flatten_to_chunks repeats into every chunk, which can push
    it above 100% and makes it meaningless as a retention measure.
    """
    main = get_main_block(html)
    for junk in main.find_all(["script", "style"]):
        junk.decompose()
    # Both sides of the comparison must go through the same normalization the
    # parser applies, or the difference between them is measured as content
    # loss. Skipping this counts every de-obfuscated address as a missing
    # block, because the raw side still reads "[email protected]".
    parser_module.deobfuscate_emails(main)
    blocks = prose_blocks(main)
    if not blocks:
        return 1.0, 0, 0, []

    haystack = squash(
        " ".join(f"{c['heading']} {c['text']}" for c in chunks)
    )
    missing = [b for b in blocks if b not in haystack]
    found = len(blocks) - len(missing)
    return found / len(blocks), found, len(blocks), missing


def parse_to_chunks(html):
    grouped = parser_module.scrape_one_page(html)
    return parser_module.flatten_to_chunks(grouped)


def fetch_sitemap_lastmod():
    root = ET.fromstring(fetch(SITEMAP_URL))
    entries = {}
    for url_element in root.findall("sm:url", SITEMAP_NS):
        loc = url_element.findtext("sm:loc", namespaces=SITEMAP_NS)
        if not loc:
            continue
        entries[loc.rstrip("/")] = url_element.findtext(
            "sm:lastmod", namespaces=SITEMAP_NS
        )
    return entries


def load_baseline():
    if BASELINE_PATH.exists():
        return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {"pages": {}}


def save_baseline(baseline):
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    baseline["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2), encoding="utf-8")


def known_component_markers(baseline):
    """The union of every marker ever recorded, not just this page's.

    A component type is only novel if it is new to the *project* - one that
    already appears on another page is one the parser has been exercised
    against, so flagging it again is noise.
    """
    seen = set()
    for page in baseline["pages"].values():
        seen.update(page.get("components", []))
    return seen


def check_page(stem, url, html, baseline, sitemap_lastmod):
    """Run checks 2-4 against freshly fetched HTML. Returns a report dict."""
    previous = baseline["pages"].get(stem, {})
    report = {"stem": stem, "url": url, "flags": [], "notes": []}

    new_hash = main_block_hash(html)
    report["hash"] = new_hash
    report["content_changed"] = (
        previous.get("hash") is not None and previous["hash"] != new_hash
    )

    markers = component_markers(html)
    report["components"] = sorted(markers)
    novel = markers - known_component_markers(baseline)
    if novel and previous:
        report["flags"].append("NEW_COMPONENT")
        report["notes"].append(
            "component types never seen before: " + ", ".join(sorted(novel))
        )

    chunks = parse_to_chunks(html)
    ratio, found, total, missing = coverage(html, chunks)
    report["coverage"] = round(ratio, 4)
    report["chunks"] = len(chunks)
    report["blocks"] = total

    baseline_coverage = previous.get("coverage")
    if baseline_coverage is not None and ratio < baseline_coverage - COVERAGE_DROP_ALERT:
        report["flags"].append("COVERAGE_DROP")
        report["notes"].append(
            f"coverage {ratio:.0%} vs baseline {baseline_coverage:.0%} "
            f"({total - found} of {total} blocks not reaching chunks) - "
            f"the source shrinking would keep this ratio flat, so suspect the parser"
        )
    report["missing_blocks"] = missing[:3]

    report["lastmod"] = sitemap_lastmod.get(url.rstrip("/"))
    if report["content_changed"] and previous.get("lastmod") == report["lastmod"]:
        report["flags"].append("LASTMOD_STALE")
        report["notes"].append(
            "content hash moved while sitemap lastmod did not - "
            "lastmod alone would have missed this page"
        )
    return report


def print_report(reports):
    width = max((len(r["stem"]) for r in reports), default=10) + 2
    print()
    print(f"{'page':<{width}}{'cover':>7}{'chunks':>8}  {'status'}")
    print("-" * (width + 30))
    for report in sorted(reports, key=lambda r: r["stem"]):
        if report["flags"]:
            status = " ".join(report["flags"])
        elif report["content_changed"]:
            status = "content changed"
        else:
            status = "ok"
        print(
            f"{report['stem']:<{width}}"
            f"{report['coverage']:>6.0%}"
            f"{report['chunks']:>8}  {status}"
        )

    for report in sorted(reports, key=lambda r: r["stem"]):
        if not report["notes"]:
            continue
        print(f"\n  {report['stem']}")
        for note in report["notes"]:
            print(f"    - {note}")
        for block in report["missing_blocks"]:
            print(f"    ~ not in chunks: {block[:110]}")

    flagged = [r for r in reports if r["flags"]]
    changed = [r for r in reports if r["content_changed"] and not r["flags"]]
    print()
    if flagged:
        print(
            f"{len(flagged)} page(s) need a human before re-embedding - "
            f"a flag means the diff may not mean what it looks like."
        )
    if changed:
        print(
            f"{len(changed)} page(s) changed cleanly and can be re-embedded: "
            f"python src/embed.py"
        )
    if not flagged and not changed:
        print("no changes detected.")


def report_new_pages(sitemap_lastmod, url_map):
    known = {url.rstrip("/") for url in url_map.values()}
    candidates = sorted(
        url
        for url in sitemap_lastmod
        if "/content/" in url and url not in known
    )
    print(f"\n{len(candidates)} /content/ URLs in the sitemap are not in the corpus.")
    print(
        "Discovery is solved; selection is not - which of these belong in a "
        "standing-guidance corpus is a judgment call, not a diff.\n"
    )
    for url in candidates[:40]:
        print(f"  {sitemap_lastmod.get(url) or '(no lastmod)':<22} {url}")
    if len(candidates) > 40:
        print(f"  ... and {len(candidates) - 40} more")


def seed_baseline(url_map, baseline):
    """Record the current local corpus as the last known-good state.

    No network and no lastmod - a seeded page has a content hash, a component
    vocabulary and a coverage figure, but the first real run still has to fetch
    it to learn its lastmod. That is deliberate: seeding asserts "what is on
    disk is correct", which is a claim about the parser, not about ISSO.
    """
    print("seeding baseline from data/raw/ (no network)...")
    for filename, url in sorted(url_map.items()):
        stem = Path(filename).stem
        raw_path = RAW_DIR / filename
        if not raw_path.exists():
            print(f"  SKIP {stem} - {raw_path} missing")
            continue

        html = raw_path.read_text(encoding="utf-8")
        chunks = parse_to_chunks(html)
        ratio, found, total, _ = coverage(html, chunks)
        baseline["pages"][stem] = {
            "url": url,
            "lastmod": None,
            "hash": main_block_hash(html),
            "coverage": round(ratio, 4),
            "chunks": len(chunks),
            "blocks": total,
            "components": sorted(component_markers(html)),
            "checked": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        print(f"  {stem:<50} coverage {ratio:>5.0%}  {len(chunks):>3} chunks")

    save_baseline(baseline)
    print(f"\nbaseline written -> {BASELINE_PATH}")
    print(f"known component types: {len(known_component_markers(baseline))}")


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--full",
        action="store_true",
        help="check every page regardless of lastmod (the backstop pass)",
    )
    argument_parser.add_argument(
        "--apply",
        action="store_true",
        help="write refreshed HTML to data/raw/ and update the baseline",
    )
    argument_parser.add_argument(
        "--new-pages",
        action="store_true",
        help="list sitemap URLs not represented in the corpus",
    )
    argument_parser.add_argument(
        "--seed",
        action="store_true",
        help="record a baseline from the local data/raw/ corpus, no network",
    )
    args = argument_parser.parse_args()

    url_map = json.loads(URL_MAP_PATH.read_text(encoding="utf-8"))
    baseline = load_baseline()

    if args.seed:
        seed_baseline(url_map, baseline)
        return

    print(f"fetching {SITEMAP_URL} ...")
    sitemap_lastmod = fetch_sitemap_lastmod()
    print(f"  {len(sitemap_lastmod)} URLs in sitemap")

    if args.new_pages:
        report_new_pages(sitemap_lastmod, url_map)
        return

    targets = []
    for filename, url in sorted(url_map.items()):
        stem = Path(filename).stem
        recorded = baseline["pages"].get(stem, {}).get("lastmod")
        current = sitemap_lastmod.get(url.rstrip("/"))
        if args.full or not baseline["pages"] or recorded != current:
            targets.append((stem, url))

    if not targets:
        print("\nno page's lastmod moved. Run --full periodically anyway: "
              "lastmod is a CMS claim, not an observation.")
        return

    print(f"\nchecking {len(targets)} page(s)...")
    reports = []
    for index, (stem, url) in enumerate(targets):
        if index:
            time.sleep(REQUEST_DELAY_SECONDS)
        print(f"  fetching {url}")
        try:
            html = fetch(url)
        except Exception as error:  # noqa: BLE001 - report and keep going
            print(f"    FAILED: {error}", file=sys.stderr)
            continue

        report = check_page(stem, url, html, baseline, sitemap_lastmod)
        reports.append(report)

        if args.apply:
            (RAW_DIR / f"{stem}.html").write_text(html, encoding="utf-8")
            baseline["pages"][stem] = {
                "url": url,
                "lastmod": report["lastmod"],
                "hash": report["hash"],
                "coverage": report["coverage"],
                "chunks": report["chunks"],
                "blocks": report["blocks"],
                "components": report["components"],
                "checked": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }

    print_report(reports)

    if args.apply:
        save_baseline(baseline)
        print(f"\nbaseline updated -> {BASELINE_PATH}")
        print("data/raw/ refreshed. Re-run src/one-page.py, then src/embed.py.")
    else:
        print("\n(read-only - pass --apply to refresh data/raw/ and the baseline)")


if __name__ == "__main__":
    main()
