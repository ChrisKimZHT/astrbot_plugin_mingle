"""Deterministic text redaction for newly stored chat records."""

import re

_PRIVATE_KEY = re.compile(
    r"-----BEGIN (?P<kind>(?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY)-----"
    r".*?-----END (?P=kind)-----",
    re.DOTALL,
)
_SECRET_FIELD = re.compile(
    r"(?<![A-Za-z0-9_])(?P<label>api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"client[_-]?secret|password|passwd|pwd|token|secret|密码|口令|密钥)"
    r"(?P<separator>[\"']?\s*[:=：]\s*)"
    r"(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;，；。}\[\]\"'<>`]+)",
    re.IGNORECASE,
)
_BEARER_TOKEN = re.compile(r"\bBearer[ \t]+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_API_KEY = re.compile(
    r"(?<![A-Za-z0-9_-])(?:sk-[A-Za-z0-9_-]{20,}|"
    r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"AKIA[A-Z0-9]{16})(?![A-Za-z0-9_-])"
)
_EMAIL = re.compile(
    r"(?<![A-Za-z0-9_.+-])[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)+"
    r"(?![A-Za-z0-9_-])"
)
_PHONE = re.compile(r"(?<![A-Za-z0-9_])(?:\+?86[- ]?)?1[3-9][0-9]{9}(?![A-Za-z0-9_])")
_IDENTITY_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_IDENTITY_CHECK_DIGITS = "10X98765432"
_IDENTITY = re.compile(
    r"(?<![A-Za-z0-9_])[1-9][0-9]{5}(?:18|19|20)[0-9]{2}"
    r"(?:0[1-9]|1[0-2])(?:0[1-9]|[12][0-9]|3[01])[0-9]{3}[0-9Xx](?![A-Za-z0-9_])"
)


def replace_identity(match: re.Match[str]) -> str:
    """Redact an identity number only if its check digit matches.

    Args:
        match: Candidate mainland Chinese identity number.

    Returns:
        A type marker for a valid checksum, otherwise the original number.
    """
    value = match.group(0)
    checksum = sum(
        int(digit) * weight for digit, weight in zip(value[:17], _IDENTITY_WEIGHTS)
    )
    if _IDENTITY_CHECK_DIGITS[checksum % 11] == value[-1].upper():
        return "[身份证号]"
    return value


def redact_text(text: str) -> str:
    """Replace recognized personal information and credentials with type markers.

    Args:
        text: Complete normalized chat body, before storage truncation.

    Returns:
        Redacted text. Matching is heuristic and does not inspect attachments.
    """
    # Redact credentials first so their contents cannot become partial markers.
    text = _PRIVATE_KEY.sub("[私钥]", text)
    text = _SECRET_FIELD.sub(r"\g<label>\g<separator>[凭据]", text)
    text = _BEARER_TOKEN.sub("Bearer [令牌]", text)
    text = _API_KEY.sub("[API密钥]", text)
    text = _EMAIL.sub("[邮箱]", text)
    text = _PHONE.sub("[手机号]", text)
    return _IDENTITY.sub(replace_identity, text)
