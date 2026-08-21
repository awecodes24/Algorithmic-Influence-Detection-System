"""
Account identifier anonymization utilities.

Phase 1 responsibility:
    Raw Reddit username
        ↓
    deterministic anonymized account ID

The raw username should never be stored in the research database.
"""

from __future__ import annotations

import hashlib


ANONYMIZED_ID_LENGTH = 16


def normalize_identifier(identifier: str) -> str:
    """
    Normalize a Reddit identifier before hashing.

    The normalization ensures that equivalent identifiers such as:
        "User123"
        " user123 "
        "USER123"

    produce the same anonymized ID.

    Parameters
    ----------
    identifier:
        Raw account identifier / username.

    Returns
    -------
    str
        Normalized identifier.

    Raises
    ------
    ValueError
        If the identifier is empty or not a string.
    """
    if not isinstance(identifier, str):
        raise ValueError("Account identifier must be a string.")

    normalized = identifier.strip().lower()

    if not normalized:
        raise ValueError("Account identifier cannot be empty.")

    return normalized


def anonymize(identifier: str) -> str:
    """
    Convert a raw account identifier into a deterministic anonymized ID.

    SHA-256 is used so that the same account always receives the same
    anonymized identifier during collection and later processing.

    Parameters
    ----------
    identifier:
        Raw Reddit username.

    Returns
    -------
    str
        Anonymized account ID.
    """
    normalized = normalize_identifier(identifier)

    digest = hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()

    return digest[:ANONYMIZED_ID_LENGTH]


def is_anonymized(identifier: str) -> bool:
    """
    Check whether a value has the expected anonymized-ID format.

    This is a lightweight format check, not a cryptographic proof.
    """
    if not isinstance(identifier, str):
        return False

    if len(identifier) != ANONYMIZED_ID_LENGTH:
        return False

    return all(
        character in "0123456789abcdef"
        for character in identifier.lower()
    )


def anonymize_optional(identifier: str | None) -> str | None:
    """
    Anonymize an optional identifier.

    Useful when Reddit returns a deleted/unavailable author.

    Returns None when no identifier is available.
    """
    if identifier is None:
        return None

    identifier = identifier.strip()

    if not identifier:
        return None

    return anonymize(identifier)


__all__ = [
    "ANONYMIZED_ID_LENGTH",
    "normalize_identifier",
    "anonymize",
    "anonymize_optional",
    "is_anonymized",
]