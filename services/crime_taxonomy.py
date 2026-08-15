"""
services/crime_taxonomy.py

Every city reports crime differently — Chicago has a clean `Primary Type` +
`FBI Code`, Philadelphia has `text_general_code` + `ucr_general`, Pittsburgh
just dumps every charge into one `OFFENSES` string ("2702 Aggravated
Assault. / 2709(a) Harassment."), Baltimore abbreviates ("AGG. ASSAULT"),
Buffalo has `Incident Type Primary`, and so on. None of that is worth
teaching the map about individually.

Instead, this module maps whatever free-text made it into a row (see
`services/crime_sources.py`) onto ONE standardized taxonomy, and attaches a
severity weight to each category. That weight — not a raw incident count —
is what powers the "Crime" map layer, so a block with five car break-ins
doesn't visually outweigh a block with one shooting.

Weighting approach
-------------------
The category set and relative ordering are loosely modeled on the FBI's
Uniform Crime Reporting hierarchy (violent crimes against a person score
highest, then property crimes involving force or real loss, then
quality-of-life/regulatory offenses) — but the actual numbers are a judgment
call about *visual severity*, not a legal classification, and there's no
canonical source of truth to test them against. Treat `weight` as a starting
point: it's a plain list below, easy to retune per category if your use case
wants e.g. property crime weighted closer to violent crime.

Classification approach
------------------------
`classify_crime_text()` does keyword/regex matching, not NLP — a pragmatic
choice given the input is a few free-text words from ~9 different source
formats, several already sampled and cross-checked here (Pittsburgh's real
export, notably, is riddled with things like "HARRASSMENT" (sic) and
abbreviations like "MTR VEH THEFT" — the patterns below were written against
actual sample rows, not assumed spellings).

It scans for EVERY category's keywords (not just the first match) and keeps
whichever matched category has the highest weight. This means classification
is insensitive to whether the input is a single "primary offense" field or a
Pittsburgh-style blob listing every charge on the incident — if a severe
keyword appears anywhere in the text, the incident is scored at that
severity, mirroring the same logic behind the FBI's Hierarchy Rule (and
Pittsburgh's own precomputed HIERARCHY field, which we don't use directly
since no other city has an equivalent — using the shared text classifier
everywhere keeps categories comparable across cities).

This is a best-effort heuristic, not a certified classifier. Ambiguous
compounds (e.g. "IDENTITY THEFT" containing the substring "THEFT") can land
in a neighboring category with a similar weight — acceptable slop for a map
layer, worth knowing about if you pipe `category` into something stricter.
"""
import re
from typing import NamedTuple, Optional


class CrimeCategory(NamedTuple):
    key: str
    label: str
    weight: float   # 1.0 (least severe) – 10.0 (most severe)


# Order here is presentation order only (e.g. for a legend) — classification
# checks every category and keeps the highest-weight match, so this list's
# order has no effect on how any given string gets classified.
CRIME_CATEGORIES: list[CrimeCategory] = [
    CrimeCategory("homicide",              "Homicide",                        10.0),
    CrimeCategory("rape_sexual_assault",   "Rape / Sexual Assault",            9.0),
    CrimeCategory("kidnapping_trafficking","Kidnapping / Trafficking",         8.5),
    CrimeCategory("robbery",               "Robbery",                          7.5),
    CrimeCategory("aggravated_assault",    "Aggravated Assault",               7.0),
    CrimeCategory("arson",                 "Arson",                            6.0),
    CrimeCategory("burglary",              "Burglary",                         5.0),
    CrimeCategory("weapons",               "Weapons Violation",                5.0),
    CrimeCategory("sex_offense_other",     "Other Sex Offense",                4.5),
    CrimeCategory("motor_vehicle_theft",   "Motor Vehicle Theft",              4.0),
    CrimeCategory("simple_assault",        "Simple Assault / Battery",         3.5),
    CrimeCategory("drug_offense",          "Drug / Narcotics",                 3.0),
    CrimeCategory("theft_larceny",         "Theft / Larceny",                  2.5),
    CrimeCategory("threats_harassment",    "Threats / Harassment / Stalking",  2.25),
    CrimeCategory("fraud_forgery",         "Fraud / Forgery / Embezzlement",   2.0),
    CrimeCategory("vandalism",             "Vandalism / Criminal Mischief",    1.5),
    CrimeCategory("dui",                   "DUI / Impaired Driving",           1.5),
    CrimeCategory("disorderly_quality_of_life", "Disorderly Conduct / Quality of Life", 1.0),
    CrimeCategory("other",                 "Other / Unclassified",             1.0),
]

CATEGORY_BY_KEY: dict[str, CrimeCategory] = {c.key: c for c in CRIME_CATEGORIES}
_OTHER = CATEGORY_BY_KEY["other"]


# Keyword/regex patterns per category. Case-insensitive by construction — the
# input is upper-cased before matching, so patterns are written upper-case.
# Prefer short fragments over full words where a source is known to abbreviate
# or misspell (e.g. "PUBLIC DRUNK" instead of "PUBLIC DRUNKENNESS", which
# shows up in real Pittsburgh data as "PUBLIC DRUNKENESS").
_PATTERNS: dict[str, list[str]] = {
    "homicide": [
        r"\bHOMICIDES?\b", r"\bMURDERS?\b", r"\bMANSLAUGHTER\b",
    ],
    "rape_sexual_assault": [
        r"\bRAPES?\b", r"SEXUAL ASSAULT", r"SEXUAL ABUSE", r"\bSODOMY\b",
        r"CRIM(?:INAL)?\s*SEXUAL",
    ],
    "kidnapping_trafficking": [
        r"\bKIDNAP", r"\bABDUCT", r"HUMAN TRAFFICKING", r"\bTRAFFICKING\b",
    ],
    "robbery": [
        r"\bROBBERY\b", r"\bROBBED\b", r"CARJACK",
    ],
    "aggravated_assault": [
        r"AGGRAVAT\w*\s*ASSAULT", r"\bAGG\.?\s*ASSAULT",
        r"ASSAULT[^.]{0,25}(FIREARM|WEAPON|DEADLY|GUN|KNIFE)",
        r"\bSHOOTING\b", r"SHOTS FIRED", r"\bSTABBING\b",
        r"DISCHARGE[^.]{0,15}FIREARM", r"RECKLESS\w*\s*ENDANGER",
    ],
    "arson": [
        r"\bARSON\b",
    ],
    "burglary": [
        r"\bBURGLAR", r"BREAKING\s*(AND|&)\s*ENTERING", r"\bB\s?&\s?E\b",
    ],
    "weapons": [
        r"WEAPONS?\s*(VIOLATION|OFFEN[CS]E)", r"ILLEGAL[^.]{0,15}(FIREARM|GUN)",
        r"CARRYING[^.]{0,20}CONCEALED", r"PROHIBITED\s*(WEAPON|FIREARM)",
        r"FELON[^.]{0,15}(FIREARM|WEAPON)",
    ],
    "sex_offense_other": [
        r"INDECENT EXPOSURE", r"PUBLIC INDECENCY", r"\bPROSTITUTION\b",
        r"\bLEWDNESS\b", r"\bOBSCENITY\b",
    ],
    "motor_vehicle_theft": [
        r"MOTOR\s*VEH(?:ICLE)?\s*THEFT", r"MTR\s*VEH(?:ICLE)?\s*THEFT",
        r"\bAUTO\s*THEFT\b", r"\bUUV\b",
        r"UNAUTHORIZED USE[^.]{0,15}VEHICLE", r"STOLEN\s*(AUTO|VEHICLE|CAR)\b",
    ],
    "simple_assault": [
        r"\bASSAULT\b", r"\bBATTERY\b",
    ],
    "drug_offense": [
        r"NARCOTIC", r"\bDRUG\b", r"CONTROLLED SUBSTANCE",
        r"POSSESS\w*[^.]{0,20}(COCAINE|HEROIN|MARIJUANA|METH|FENTANYL|NARCOTIC)",
    ],
    "theft_larceny": [
        r"\bTHEFTS?\b", r"\bLARCEN(?:Y|IES)\b", r"SHOPLIFT", r"STOLEN PROPERTY",
        r"\bPICKPOCKET",
    ],
    "threats_harassment": [
        r"HAR{1,2}ASSMENT", r"\bTHREAT", r"\bSTALKING\b", r"\bINTIMIDATION\b",
    ],
    "fraud_forgery": [
        r"\bFRAUD\b", r"\bFORGERY\b", r"EMBEZZLEMENT", r"COUNTERFEIT",
        r"IDENTITY THEFTS?", r"BAD CHECKS?", r"DECEPTIVE PRACTICE",
    ],
    "vandalism": [
        r"VANDALISM", r"CRIMINAL (MISCHIEF|DAMAGE)", r"\bGRAFFITI\b",
    ],
    "dui": [
        r"\bDUI\b", r"\bDWI\b", r"\bOVI\b", r"DRIVING UNDER[^.]{0,10}INFLUENCE",
        r"OPERATING[^.]{0,10}INFLUENCE",
    ],
    "disorderly_quality_of_life": [
        r"DISORDERLY", r"PUBLIC DRUNK", r"INTOXICAT", r"\bLOITERING\b", r"\bTRESPASS",
        r"\bCURFEW\b", r"\bNUISANCE\b", r"\bVAGRANCY\b", r"\bPROWLER\b",
        r"NOISE (COMPLAINT|VIOLATION)",
    ],
}

_COMPILED: dict[str, list[re.Pattern]] = {
    key: [re.compile(p) for p in pats] for key, pats in _PATTERNS.items()
}


def classify_crime_text(text: Optional[str]) -> tuple[str, str, float]:
    """
    Map a raw crime type/offense/description string to a standardized
    (category_key, category_label, severity_weight).

    Scans for every category's keywords and returns the HIGHEST-weight
    category among all matches (not the first match) — see module docstring
    for why. Returns the neutral 'other' bucket (weight 1.0) for empty input
    or text that matches nothing, never raises.
    """
    if not text:
        return _OTHER.key, _OTHER.label, _OTHER.weight
    t = str(text).upper()
    if not t.strip() or t.strip() in ("NAN", "NONE", "NULL"):
        return _OTHER.key, _OTHER.label, _OTHER.weight

    best: Optional[CrimeCategory] = None
    for key, regexes in _COMPILED.items():
        if any(rx.search(t) for rx in regexes):
            cat = CATEGORY_BY_KEY[key]
            if best is None or cat.weight > best.weight:
                best = cat

    if best is None:
        return _OTHER.key, _OTHER.label, _OTHER.weight
    return best.key, best.label, best.weight
