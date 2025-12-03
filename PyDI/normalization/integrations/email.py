"""
email-validator integration for email validation and normalization.

This module wraps the email-validator library to provide:
- Email syntax validation
- Email normalization (lowercase, remove dots from Gmail, etc.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from email_validator import validate_email as _validate_email, EmailNotValidError

logger = logging.getLogger(__name__)


@dataclass
class EmailInfo:
    """Information about a validated email."""

    original: str
    normalized: str
    local_part: str
    domain: str
    is_valid: bool
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "normalized": self.normalized,
            "local_part": self.local_part,
            "domain": self.domain,
            "is_valid": self.is_valid,
            "error": self.error,
        }


def validate_email(
    email: str,
    check_deliverability: bool = False,
) -> EmailInfo:
    """
    Validate an email address.

    Args:
        email: Email address to validate
        check_deliverability: If True, check if domain has MX records (slower)

    Returns:
        EmailInfo with validation result

    Examples:
        >>> info = validate_email("user@example.com")
        >>> info.is_valid
        True
        >>> info.normalized
        'user@example.com'
    """
    if not email or not isinstance(email, str):
        return EmailInfo(
            original=email or "",
            normalized="",
            local_part="",
            domain="",
            is_valid=False,
            error="Empty or invalid input",
        )

    email = email.strip()

    try:
        result = _validate_email(email, check_deliverability=check_deliverability)
        return EmailInfo(
            original=email,
            normalized=result.normalized,
            local_part=result.local_part,
            domain=result.domain,
            is_valid=True,
            error=None,
        )
    except EmailNotValidError as e:
        return EmailInfo(
            original=email,
            normalized="",
            local_part="",
            domain="",
            is_valid=False,
            error=str(e),
        )


def normalize_email(
    email: str,
    check_deliverability: bool = False,
) -> str | None:
    """
    Normalize an email address.

    Args:
        email: Email address to normalize
        check_deliverability: If True, check if domain has MX records

    Returns:
        Normalized email, or None if invalid

    Examples:
        >>> normalize_email("User@EXAMPLE.com")
        'user@example.com'
        >>> normalize_email("invalid")
        None
    """
    info = validate_email(email, check_deliverability)
    return info.normalized if info.is_valid else None


def is_valid_email(
    email: str,
    check_deliverability: bool = False,
) -> bool:
    """
    Check if an email address is valid.

    Args:
        email: Email address to check
        check_deliverability: If True, check if domain has MX records

    Returns:
        True if valid, False otherwise
    """
    return validate_email(email, check_deliverability).is_valid
