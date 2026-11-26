"""The simple column transformations a join may need - one registry, three consumers.

A join often only works after a small, value-preserving fix to a key: trimming stray spaces,
matching letter case, restoring leading zeros on a ZIP/tract, keeping one part of a composite
code.  Those fixes used to live only inside ``services/relationship_discovery.py`` (as
``transform_sql`` and friends), so they were invisible to the Data page and absent from the
catalog: a relationship stored a raw SQL expression and nothing said what it meant.

This module is the single definition of those transformations:

* the onboarding proposer tests them against the data,
* the Data page offers them (``options()``) and can re-test a choice,
* the catalog stores the structured result on the relationship (``relationship()`` /
  ``parse_expression()``), so a join plan can show and apply the same rule.

Nothing here decides a join: it only renders and describes a normalisation.  The SQL an
expression produces stays authoritative for query generation - the catalog's ``left_expr`` /
``right_expr`` are what the compiler qualifies and puts in ``ON``.
"""

from __future__ import annotations

import re

#: The transforms a person can choose.  ``identity`` must stay first (it is the no-op default).
SPECS: dict[str, dict] = {
    "identity": {
        "label": "No change",
        "description": "Compare the column exactly as stored.",
        "needs_width": False,
        "widths": (),
    },
    "trim": {
        "label": "Trim extra spaces",
        "description": "Remove leading and trailing spaces before comparing.",
        "needs_width": False,
        "widths": (),
    },
    "lower": {
        "label": "Lower case",
        "description": "Trim, then lower-case the source column (the target must already be lower-case).",
        "needs_width": False,
        "widths": (),
    },
    "upper": {
        "label": "Upper case",
        "description": "Trim, then upper-case the source column (the target must already be upper-case).",
        "needs_width": False,
        "widths": (),
    },
    "ci": {
        "label": "Ignore case (both sides)",
        "description": "Trim and upper-case BOTH columns, so letter case and stray spaces never block a match.",
        "needs_width": False,
        "widths": (),
    },
    "zfill": {
        "label": "Pad with leading zeros",
        "description": "Restore dropped leading zeros on a numeric-looking code (ZIP 5, county 5, tract 11, block group 12).",
        "needs_width": True,
        "widths": (2, 5, 11, 12, 15),
    },
    "left": {
        "label": "Keep first characters",
        "description": "Keep the first N characters of the source column (e.g. a 12-digit block group -> 11-digit tract).",
        "needs_width": True,
        "widths": (2, 5, 11, 12, 15),
    },
    "right": {
        "label": "Keep last characters",
        "description": "Keep the last N characters of the source column.",
        "needs_width": True,
        "widths": (2, 5, 11, 12, 15),
    },
    "digits": {
        "label": "Keep digits only",
        "description": "Strip spaces, dashes and other punctuation so only 0-9 characters are compared.",
        "needs_width": False,
        "widths": (),
    },
}

KINDS = tuple(SPECS)

#: Simplicity ranking used when two transforms match equally well: prefer the least invasive.
SIMPLICITY = {"identity": 0, "trim": 1, "zfill": 2, "left": 3, "right": 3, "upper": 4, "lower": 4, "digits": 5, "ci": 6}

_BARE_COLUMN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _column(text: str) -> str:
    """The bare column inside a parsed expression (quoted identifiers are unquoted)."""
    return text.strip().strip('"')


_TRIM = re.compile(r"^TRIM\((.+)\)$", re.I)
_CASED_TRIM = re.compile(r"^(LOWER|UPPER)\(TRIM\((.+)\)\)$", re.I)
_LPAD = re.compile(r"^LPAD\(TRIM\((.+)\),\s*([0-9]+),\s*'0'\)$", re.I)
_SLICE = re.compile(r"^(LEFT|RIGHT)\(TRIM\((.+)\),\s*([0-9]+)\)$", re.I)
_SLICE_PLAIN = re.compile(r"^(LEFT|RIGHT)\(([A-Za-z_][A-Za-z0-9_]*),\s*([0-9]+)\)$", re.I)
_DIGITS = re.compile(r"^REGEXP_REPLACE\(TRIM\((.+)\),\s*'\[\^0-9\]',\s*'',\s*'g'\)$", re.I)


def options() -> list[dict]:
    """Every transform a person may choose, as plain JSON for the Data page."""
    return [
        {
            "kind": kind,
            "label": spec["label"],
            "description": spec["description"],
            "needs_width": spec["needs_width"],
            "widths": list(spec["widths"]),
        }
        for kind, spec in SPECS.items()
    ]


def normalize(spec: dict | None) -> dict:
    """Validate a transform spec (``{"kind": ..., "width": ...}``) and keep only its real fields."""
    if not isinstance(spec, dict):
        raise ValueError("A transformation must be an object with a 'kind'.")
    kind = str(spec.get("kind") or "").strip().lower()
    if kind not in SPECS:
        raise ValueError(f"Unknown transformation '{kind}'.")
    out: dict = {"kind": kind}
    if SPECS[kind]["needs_width"]:
        try:
            width = int(spec.get("width") or 0)
        except (TypeError, ValueError):
            width = 0
        if width <= 0 or width > 32:
            raise ValueError(f"'{SPECS[kind]['label']}' needs a width between 1 and 32.")
        out["width"] = width
    return out


def key(spec: dict) -> str:
    spec = normalize(spec)
    return spec["kind"] + (str(spec["width"]) if spec.get("width") else "")


def sql(spec: dict, ref: str) -> str:
    """The source-side SQL for ``ref`` (a quoted column reference)."""
    kind = normalize(spec)["kind"]
    if kind == "identity":
        return ref
    if kind == "trim":
        return f"TRIM({ref})"
    if kind in ("upper", "ci"):
        return f"UPPER(TRIM({ref}))"
    if kind == "lower":
        return f"LOWER(TRIM({ref}))"
    if kind == "zfill":
        return f"LPAD(TRIM({ref}), {int(spec['width'])}, '0')"
    if kind == "left":
        return f"LEFT(TRIM({ref}), {int(spec['width'])})"
    if kind == "right":
        return f"RIGHT(TRIM({ref}), {int(spec['width'])})"
    if kind == "digits":
        return f"REGEXP_REPLACE(TRIM({ref}), '[^0-9]', '', 'g')"
    raise ValueError(f"Unknown transformation '{kind}'.")


def target_sql(spec: dict, ref: str) -> str:
    """The target-side SQL.  Only the case-insensitive rule normalises both sides."""
    return f"UPPER(TRIM({ref}))" if normalize(spec)["kind"] == "ci" else ref


def label(spec: dict) -> str:
    return SPECS[normalize(spec)["kind"]]["label"]


def rule_text(spec: dict, column: str) -> str:
    spec = normalize(spec)
    kind, width = spec["kind"], spec.get("width")
    return {
        "identity": f"Compare {column} as it is",
        "trim": f"Trim extra spaces from {column}",
        "lower": f"Trim {column} and convert it to lower case",
        "upper": f"Trim {column} and convert it to UPPER CASE",
        "ci": f"Match {column} ignoring extra spaces and letter case (both sides are normalized)",
        "zfill": f"Pad {column} with leading zeros to {width} digits",
        "left": f"Keep the first {width} characters of {column}",
        "right": f"Keep the last {width} characters of {column}",
        "digits": f"Strip everything except digits from {column}",
    }[kind]


def parse_expression(expr: str) -> dict | None:
    """Read one of this module's simple expressions back into ``{kind, width?, column}``.

    Returns ``None`` for anything more complex (a concatenation, a CASE, a derived column
    name), which is recorded as an opaque ``expression`` instead of being guessed at.
    """
    text = str(expr or "").strip()
    if not text:
        return None
    if _BARE_COLUMN.match(text):
        return {"kind": "identity", "column": _column(text)}
    m = _LPAD.match(text)
    if m:
        return {"kind": "zfill", "width": int(m.group(2)), "column": _column(m.group(1))}
    m = _SLICE_PLAIN.match(text)
    if m:
        return {
            "kind": "left" if m.group(1).upper() == "LEFT" else "right",
            "width": int(m.group(3)),
            "column": _column(m.group(2)),
        }
    m = _SLICE.match(text)
    if m:
        return {
            "kind": "left" if m.group(1).upper() == "LEFT" else "right",
            "width": int(m.group(3)),
            "column": _column(m.group(2)),
        }
    m = _CASED_TRIM.match(text)
    if m:
        return {
            "kind": "ci" if m.group(1).upper() == "UPPER" else "lower",
            "column": _column(m.group(2)),
        }
    m = _TRIM.match(text)
    if m:
        return {"kind": "trim", "column": _column(m.group(1))}
    m = _DIGITS.match(text)
    if m:
        return {"kind": "digits", "column": _column(m.group(1))}
    return None


def column_of(expr: str) -> str:
    """The source column an expression starts from (the expression itself when it is opaque)."""
    parsed = parse_expression(expr)
    return parsed["column"] if parsed else str(expr or "").strip()


def infer(expr: str) -> dict:
    """Structured metadata for one side of a relationship: a known rule, or an opaque expression."""
    text = str(expr or "").strip()
    parsed = parse_expression(text)
    if parsed:
        return {**parsed, "expr": text}
    return {"kind": "expression", "expr": text, "column": column_of(text)}


def relationship(left_expr: str, right_expr: str) -> dict:
    """Describe both sides of a join key with structured transformations.

    ``ci`` normalises the target too, so when the target expression is exactly the rule's
    target SQL the same spec is recorded on the right; everything else keeps its own parse.
    """
    left = infer(left_expr)
    right = infer(right_expr)
    if left["kind"] == "ci" and parse_expression(right_expr):
        parsed_right = parse_expression(right_expr) or {}
        if parsed_right.get("column") == left.get("column") and parsed_right.get("kind") in ("upper", "ci"):
            right = {**left, "expr": str(right_expr or "").strip()}
    rule = "" if left["kind"] == "expression" else rule_text(left, left.get("column") or "the key")
    return {"left": left, "right": right, "rule": rule}


def pair(spec: dict, source_column: str, target_column: str) -> dict:
    """Transform metadata for a proposal/relationship, with the columns each side starts from."""
    spec = normalize(spec)
    return {
        "left": {**spec, "column": source_column},
        "right": (
            {"kind": "ci", "column": target_column}
            if spec["kind"] == "ci"
            else {"kind": "identity", "column": target_column}
        ),
        "rule": rule_text(spec, source_column),
    }
