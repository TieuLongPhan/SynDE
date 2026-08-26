"""Structured, actionable exceptions for the SynDE public interface.

Every error raised by the user-facing predictor carries the offending input,
the domain rule that rejected it, and a concrete next step.  All exceptions
subclass :class:`ValueError` so existing ``except ValueError`` call sites keep
working unchanged.
"""

from __future__ import annotations

from typing import Any, Iterable

__all__ = [
    "SynDEError",
    "SynDEInputError",
    "SynDEDomainError",
    "format_domain_message",
]


class SynDEError(ValueError):
    """Base class for every SynDE error that reports a user-fixable problem."""


class SynDEInputError(SynDEError):
    """A structure could not be parsed into a normalized SynDE graph."""


class SynDEDomainError(SynDEError):
    """A parsed structure lies outside the active model's applicability domain.

    :param reason: Short sentence naming the violated domain rule.
    :type reason: str
    :param subject: Canonical SMILES or other identifier for the input.
    :type subject: str | None
    :param hint: Concrete remedial action offered to the caller.
    :type hint: str | None
    :param details: Extra machine-readable context attached to the failure.
    :type details: dict[str, Any] | None
    """

    def __init__(
        self,
        reason: str,
        *,
        subject: str | None = None,
        hint: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.reason = reason
        self.subject = subject
        self.hint = hint
        self.details = dict(details or {})
        super().__init__(format_domain_message(reason, subject=subject, hint=hint))


def format_domain_message(
    reason: str,
    *,
    subject: str | None = None,
    hint: str | None = None,
) -> str:
    """Compose a multi-line domain-error message.

    :param reason: Short sentence naming the violated domain rule.
    :type reason: str
    :param subject: Canonical SMILES or other identifier for the input.
    :type subject: str | None
    :param hint: Concrete remedial action offered to the caller.
    :type hint: str | None
    :return: Human-readable message beginning with the violated rule.
    :rtype: str
    """
    head = reason if subject is None else f"{reason} (input: {subject})"
    return head if hint is None else f"{head}\n  Hint: {hint}"


def describe_domain(
    elements: Iterable[str],
    charges: Iterable[int],
) -> str:
    """Summarize an energy model card's accepted chemistry in one line.

    :param elements: Element symbols the active artifact was fitted for.
    :type elements: Iterable[str]
    :param charges: Total formal charges the active artifact accepts.
    :type charges: Iterable[int]
    :return: Single-line description of the supported chemical domain.
    :rtype: str
    """
    element_text = " ".join(sorted(set(elements)))
    charge_text = ", ".join(str(value) for value in sorted(set(charges)))
    return (
        f"elements [{element_text}]; total formal charge [{charge_text}]; "
        "connected, closed-shell, non-isotopic structures"
    )
