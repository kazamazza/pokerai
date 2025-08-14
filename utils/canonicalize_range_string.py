import re

_HAND_TOKEN_RE = re.compile(r'(?:[2-9TJQKA]{2}(?:[so])?\+?)')

def canonicalize_range_string(range_str: str) -> str:
    """
    Parse any messy range string into a canonical, comma-separated list of tokens.
    - Removes whitespace and odd separators
    - Inserts missing separators (e.g., '22+A6s+' → '22+,A6s+')
    - Deduplicates while preserving order
    """
    if not range_str:
        return ""

    # normalize separators, strip spaces
    s = re.sub(r'[\s;]+', '', str(range_str))

    # extract all valid tokens in order
    tokens = _HAND_TOKEN_RE.findall(s)

    # preserve order & dedupe
    seen = set()
    clean_tokens = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            clean_tokens.append(t)

    return ",".join(clean_tokens)