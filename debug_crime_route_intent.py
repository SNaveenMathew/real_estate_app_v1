import re

MESSAGE = "Find a bike route from 5624 Bryant St, Highland Park, PA 15206 to 5903 5th Ave, Shadyside, PA 15232 that avoids some of the crime prone areas in Pittsburgh."

m = re.search(r"\bfrom\s+(.+?)\s+\bto\s+(.+?)(?=\s+(?:that\s+)?(?:avoids?|avoiding|without)\b|\s+and\s+(?:avoids?|avoiding)\b|[.!?]?$)", MESSAGE, re.I)
assert m
print("start:", m.group(1))
print("end:", m.group(2))
