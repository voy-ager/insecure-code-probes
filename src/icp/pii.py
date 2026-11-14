"""Regex check of generated text for emails, IPv4, credential URLs and secret-shaped strings.

Gitleaks does not catch emails or names, so generation text is committed only if both pass.
"""

import math
import re

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
IPV4 = re.compile(r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\w.])")
URL_CREDS = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s/:@'\"]+:[^\s/@'\"]+@", re.IGNORECASE)
KEY_PREFIX = re.compile(r"\b(?:AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{36}|sk-[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,})")
PRIVATE_KEY = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
LONG_TOKEN = re.compile(r"[A-Za-z0-9+/_=-]{24,}")


def _entropy(s: str) -> float:
    return -sum(s.count(c) / len(s) * math.log2(s.count(c) / len(s)) for c in set(s))


def _secret_shaped(text: str) -> bool:
    # shortcut: entropy heuristic over 24+ char tokens; tune only if spot checks show noise
    return any(
        _entropy(t) > 4.0 and any(c.isdigit() for c in t) and any(c.isalpha() for c in t)
        for t in LONG_TOKEN.findall(text)
    )


def pii_flags(text: str) -> list[str]:
    """Names of the checks that match `text` (empty list = clean)."""
    checks = {
        "email": EMAIL.search(text),
        "ipv4": IPV4.search(text),
        "url_credentials": URL_CREDS.search(text),
        "key_prefix": KEY_PREFIX.search(text),
        "private_key": PRIVATE_KEY.search(text),
        "high_entropy": _secret_shaped(text),
    }
    return [k for k, v in checks.items() if v]
