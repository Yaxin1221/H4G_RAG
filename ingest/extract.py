"""HTML -> plain text, tuned for this site's Elementor markup.

Two things matter here and both are easy to get wrong:

1. We extract from the REST API's `content.rendered`, which excludes the site
   nav and footer. Scraping the rendered page instead would drag ~1,150 chars
   of identical chrome into every one of 800 pages (~240k tokens of pure noise)
   and make every chunk look like every other chunk.
2. These pages are directories of organisations. The phone numbers, emails and
   -- critically -- the hyperlink targets ARE the content. We fold `href`s into
   markdown links so the URLs survive into the model's context.
"""

from __future__ import annotations

import html
import re

_DROP = re.compile(r"(?is)<(script|style|noscript|svg)\b.*?</\1>")
_ANCHOR = re.compile(r"""(?is)<a\b[^>]*?href=["']([^"']+)["'][^>]*>(.*?)</a>""")
_BLOCK = re.compile(r"(?is)</?(p|div|li|ul|ol|h[1-6]|tr|table|br|section)\b[^>]*/?>")
_TAG = re.compile(r"(?s)<[^>]+>")
_WS_INLINE = re.compile(r"[ \t\xa0]+")
_WS_BLANK = re.compile(r"\n\s*\n+")


def _anchor_to_markdown(m: re.Match[str]) -> str:
    href = m.group(1).strip()
    label = _WS_INLINE.sub(" ", _TAG.sub("", m.group(2))).strip()
    if not label:
        return " "
    if href.startswith(("#", "javascript:")):
        return label
    href = href.replace("mailto:", "")
    # Markdown, not <angle brackets> -- the generic tag strip below would eat those.
    return f"[{label}]({href})"


def html_to_text(raw: str) -> str:
    """Flatten WordPress `content.rendered` into readable plain text."""
    s = _DROP.sub(" ", raw)
    s = _ANCHOR.sub(_anchor_to_markdown, s)
    s = _BLOCK.sub("\n", s)
    s = _TAG.sub(" ", s)
    s = html.unescape(s)
    s = _WS_INLINE.sub(" ", s)
    s = "\n".join(line.strip() for line in s.split("\n"))
    s = _WS_BLANK.sub("\n\n", s)
    return s.strip()


def clean_title(raw: str) -> str:
    return html.unescape(_TAG.sub("", raw)).strip()
