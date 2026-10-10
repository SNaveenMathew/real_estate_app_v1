"""Place-name lexicon for entity domains whose stored labels look like ``"City-City--City, ST-ST Metro Area"``.

Why this exists: people say "Indianapolis", "Fort Worth" or "Portland, ME"; the Census stores
"Indianapolis-Carmel-Anderson, IN Metro Area", "Dallas-Fort Worth-Arlington, TX Metro Area" and
"Portland-South Portland, ME Metro Area".  The original ``prefix`` matcher only recognised a name when the user typed
the *entire* hyphenated label, so most multi-city metros could never be anchored to a live value.

An entity domain opts in with ``match_mode="components"``.  Nothing here is metro-specific: any column of
``"Name[-Name], ST"`` labels (counties, regions, markets) gets the same behaviour by declaring that one field.

How a stored label becomes lookup keys (all keys are accent-folded, lower-cased and token-normalized):

* tier 0   the whole label, with and without the trailing "Metro Area"/"Micro Area" and state list
* tier 1   the first (principal) city of the label
* tier 2   every other city of the label ("Fort Worth", "Carmel", "St. Paul")
* qualified  ``<city> <ST>`` and ``<city> <state name>`` for each state in the label ("portland me", "portland maine")

A request phrase that maps to several labels is *ambiguous* unless one candidate is strictly better by
(tier, metro-before-micro).  Ambiguity is reported, never guessed: callers ask the user which place they meant.
"""
from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import dataclass

from db.text_match import normalize

STATE_NAMES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "dc": "district of columbia", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland", "ma": "massachusetts",
    "mi": "michigan", "mn": "minnesota", "ms": "mississippi", "mo": "missouri", "mt": "montana",
    "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico",
    "ny": "new york", "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina", "sd": "south dakota",
    "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "pr": "puerto rico", "gu": "guam",
    "vi": "virgin islands",
}
# Spelling variants people use interchangeably; applied to BOTH the stored labels and the request.
_TOKEN_CANON = {"saint": "st", "fort": "ft", "mount": "mt"}
# One-word places that are also everyday words.  Matched only when qualified ("Mobile, AL"), never bare, so
# "mobile home" or "reading the report" cannot be mistaken for a metro area.
COMMON_WORD_PLACES = frozenset({"mobile", "reading", "bend", "normal", "orange"})

_SUFFIX = re.compile(r"\s+(?:metro|micro)(?:politan)?\s+(?:statistical\s+)?area\s*$", re.I)
_STATE_LIST = re.compile(r"^[A-Za-z]{2}(?:-[A-Za-z]{2})*$")
_NAME_ST_LABEL = re.compile(r"^[^,]+,\s*[A-Za-z]{2}(?:-[A-Za-z]{2})*(?:\s+(?:Metro|Micro)(?:politan)?\s+Area)?\s*$")
MAX_KEY_WORDS = 12


def fold(text: str) -> str:
    """Strip accents so "Cañon City" and "Canon City" are the same place."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def canon(text: str) -> str:
    """Normalize + canonical spelling variants ("saint"->"st", "fort"->"ft", "mount"->"mt")."""
    toks = normalize(fold(text)).split(" ")
    return " ".join(_TOKEN_CANON.get(t, t) for t in toks if t)


@dataclass(frozen=True)
class Parsed:
    label: str
    states: tuple[str, ...]
    kind: int                      # 0 metro, 1 micro, 2 other
    components: tuple[str, ...]


def parse(value: str) -> Parsed:
    v = str(value).strip()
    kind = 0 if re.search(r"\bmetro(?:politan)?\b", v, re.I) else 1 if re.search(r"\bmicro(?:politan)?\b", v, re.I) else 2
    body = _SUFFIX.sub("", v).strip()
    label, tail = (body.rsplit(",", 1) + [""])[:2] if "," in body else (body, "")
    tail = tail.strip()
    states = tuple(s.lower() for s in tail.split("-")) if _STATE_LIST.match(tail) else ()
    label = label.strip()
    # OMB titles use "--" between cities when a city name itself contains a hyphen ("Nashville-Davidson--Murfreesboro").
    parts = label.split("--") if "--" in label else re.split(r"[-/]", label)
    comps = tuple(p.strip() for p in parts if p.strip())
    return Parsed(label, states, kind, comps)


@dataclass(frozen=True)
class Match:
    phrase: str                    # canonical key that matched ("ft worth")
    display: str                   # the user's own words, normalized ("fort worth")
    start: int                     # token offsets in the request
    end: int
    values: tuple[str, ...]        # best candidate labels (more than one => ambiguous)
    tier: int
    ambiguous: bool


class Lexicon:
    __slots__ = ("keys", "strong", "max_words")

    def __init__(self):
        self.keys: dict[str, dict[str, tuple[int, int]]] = {}      # key -> {label: (tier, kind)}
        self.strong: set[str] = set()                              # multi-word or state-qualified keys
        self.max_words = 1

    def add(self, key: str, value: str, tier: int, kind: int, *, qualified: bool = False) -> None:
        if not key:
            return
        owners = self.keys.setdefault(key, {})
        cur = owners.get(value)
        if cur is None or (tier, kind) < cur:
            owners[value] = (tier, kind)
        n = key.count(" ") + 1
        self.max_words = max(self.max_words, min(n, MAX_KEY_WORDS))
        if qualified or n >= 2:
            self.strong.add(key)


def build(values) -> Lexicon:
    lex = Lexicon()
    for value in values:
        if value is None:
            continue
        value = str(value)
        p = parse(value)
        body = _SUFFIX.sub("", value).strip()
        for whole in (value, body, p.label):
            lex.add(canon(whole), value, 0, p.kind)
        for i, comp in enumerate(p.components):
            tier = 1 if i == 0 else 2                       # principal (first-named) city vs the others
            spellings = [comp]
            head = comp.split("-")[0].strip()
            if "-" in comp and len(head) >= 3:              # "Nashville-Davidson" is also just "Nashville"
                spellings.append(head)
            for spelling in spellings:
                key = canon(spelling)
                lex.add(key, value, tier, p.kind)
                for st in p.states:
                    lex.add(f"{key} {st}", value, tier, p.kind, qualified=True)
                    if st in STATE_NAMES:
                        lex.add(canon(f"{key} {STATE_NAMES[st]}"), value, tier, p.kind, qualified=True)
    return lex


def find(lex: Lexicon, request: str) -> list[Match]:
    """Greedy longest-first, non-overlapping matches of the lexicon in ``request`` (left to right in the result)."""
    tokens = canon(request).split(" ") if request else []
    plain = normalize(fold(request)).split(" ") if request else []
    taken = [False] * len(tokens)
    out: list[Match] = []
    for n in range(min(lex.max_words, len(tokens)), 0, -1):
        for i in range(len(tokens) - n + 1):
            if any(taken[i:i + n]):
                continue
            phrase = " ".join(tokens[i:i + n])
            owners = lex.keys.get(phrase)
            if not owners:
                continue
            if n == 1 and phrase in COMMON_WORD_PLACES and phrase not in lex.strong:
                continue
            best = min(rank for rank in owners.values())
            winners = tuple(sorted(v for v, rank in owners.items() if rank == best))
            for j in range(i, i + n):
                taken[j] = True
            out.append(Match(phrase, " ".join(plain[i:i + n]), i, i + n, winners, best[0], len(winners) > 1))
    out.sort(key=lambda m: m.start)
    return out


# ---------------------------------------------------------------------------
# Cache: one lexicon per (domain, live value set).  Keyed on the CONTENT of the values, so new data (a new load,
# an upload) is picked up automatically and a stale lexicon can never be served.
# ---------------------------------------------------------------------------
_CACHE: dict[tuple, Lexicon] = {}
_LOCK = threading.Lock()
_CACHE_LIMIT = 8


def get_lexicon(domain_name: str, values) -> Lexicon:
    key = (domain_name, len(values), hash(frozenset(values)))
    lex = _CACHE.get(key)
    if lex is None:
        lex = build(values)
        with _LOCK:
            if len(_CACHE) >= _CACHE_LIMIT:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[key] = lex
    return lex


def suggest_match_mode(values, sample: int = 500) -> str | None:
    """Used by the catalog lint: values shaped like ``"Name, ST"`` are better served by ``components``."""
    vals = [str(v) for v in list(values)[:sample] if v]
    if len(vals) >= 5 and sum(bool(_NAME_ST_LABEL.match(v)) for v in vals) / len(vals) >= 0.6:
        return "components"
    return None
