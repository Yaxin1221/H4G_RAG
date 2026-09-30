"""Country-name normalisation for the directory lookup.

The three per-country directories are keyed by the site's own spelling. Users
(and lawyers in a hurry) do not use the site's spelling: they write Burma, the
DRC, Holland, Turkey, Ivory Coast. Claude resolves most of this itself from the
country list in the system prompt, but the tool handler still normalises so a
near-miss resolves instead of 404-ing.
"""

from __future__ import annotations

import re
import unicodedata

# alias -> the site's canonical country name
ALIASES: dict[str, str] = {
    "burma": "Myanmar",
    "ivory coast": "Cote d'Ivoire",
    "cote divoire": "Cote d'Ivoire",
    "drc": "Democratic Republic of the Congo",
    "dr congo": "Democratic Republic of the Congo",
    "congo kinshasa": "Democratic Republic of the Congo",
    "congo brazzaville": "Republic of the Congo",
    "uae": "United Arab Emirates",
    "emirates": "United Arab Emirates",
    "usa": "United States",
    "us": "United States",
    "america": "United States",
    "uk": "United Kingdom",
    "britain": "United Kingdom",
    "great britain": "United Kingdom",
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "wales": "United Kingdom",
    "holland": "Netherlands",
    "czechia": "Czech Republic",
    "turkey": "Turkiye",
    "swaziland": "Eswatini",
    "macedonia": "North Macedonia",
    "cape verde": "Cabo Verde",
    "east timor": "Timor-Leste",
    "south korea": "Republic of Korea",
    "north korea": "Democratic People's Republic of Korea",
    "vatican": "Holy See",
    "russia": "Russian Federation",
    "syria": "Syrian Arab Republic",
    "iran": "Iran",
    "laos": "Lao People's Democratic Republic",
    "tanzania": "United Republic of Tanzania",
    "bolivia": "Bolivia",
    "venezuela": "Venezuela",
    "moldova": "Republic of Moldova",
    "palestine": "Palestine",
    "vietnam": "Viet Nam",
}


def normalise(name: str) -> str:
    """Fold a country name to a comparison key: ascii, lowercase, no punctuation."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("&", " and ")
    s = re.sub(r"\b(the|republic of|state of|kingdom of|islamic|people's|peoples)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def candidates(user_input: str) -> list[str]:
    """Keys to try, best first, for a user-supplied country name."""
    raw = user_input.strip()
    keys = [normalise(raw)]
    alias = ALIASES.get(normalise(raw))
    if alias:
        keys.append(normalise(alias))
    return list(dict.fromkeys(k for k in keys if k))
