"""Dependency-free terminal rendering primitives shared by the CLI and reprs.

The helpers here deliberately avoid third-party console libraries so that the
default install stays lean.  They emit plain ASCII plus optional ANSI colour,
and every function is pure so the same output can be asserted in tests.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Any, IO, Iterable, Sequence

__all__ = [
    "Palette",
    "supports_color",
    "terminal_width",
    "format_float",
    "truncate",
    "render_table",
    "render_fields",
    "render_bar",
    "render_rule",
    "indent_block",
]

_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}


class Palette:
    """Apply or suppress ANSI styling behind a single switch.

    :param enabled: Emit ANSI escape sequences when true.
    :type enabled: bool
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)

    def __call__(self, text: str, *styles: str) -> str:
        """Style one string with any number of named ANSI attributes.

        :param text: Text to wrap in escape sequences.
        :type text: str
        :param styles: Names drawn from the module's ANSI attribute table.
        :type styles: str
        :return: Styled text, or the input unchanged when styling is off.
        :rtype: str
        """
        if not self.enabled or not styles:
            return text
        codes = "".join(_ANSI.get(style, "") for style in styles)
        return f"{codes}{text}{_ANSI['reset']}" if codes else text

    @classmethod
    def automatic(cls, stream: IO[str] | None = None) -> "Palette":
        """Build a palette that respects the stream and the environment.

        :param stream: Output stream inspected for terminal attachment.
        :type stream: IO[str] | None
        :return: Palette enabled only for a colour-capable interactive stream.
        :rtype: Palette
        """
        return cls(supports_color(stream if stream is not None else sys.stdout))


def supports_color(stream: IO[str]) -> bool:
    """Report whether coloured output is appropriate for one stream.

    Honours the ``NO_COLOR`` and ``FORCE_COLOR`` conventions before falling
    back to interactive-terminal detection.

    :param stream: Output stream to inspect.
    :type stream: IO[str]
    :return: True when ANSI escapes should be written to the stream.
    :rtype: bool
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    if os.environ.get("TERM") == "dumb":
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def terminal_width(default: int = 100) -> int:
    """Return the rendering width for the active terminal.

    :param default: Width used when the terminal size cannot be determined.
    :type default: int
    :return: Column count clamped to a legible range.
    :rtype: int
    """
    try:
        columns = shutil.get_terminal_size((default, 24)).columns
    except (OSError, ValueError):
        columns = default
    return max(60, min(int(columns), 160))


def format_float(value: Any, precision: int = 4, *, signed: bool = False) -> str:
    """Format a number for column display, tolerating non-numeric input.

    :param value: Value to render.
    :type value: Any
    :param precision: Digits kept after the decimal point.
    :type precision: int
    :param signed: Always write an explicit ``+`` or ``-`` sign.
    :type signed: bool
    :return: Fixed-point text, or ``str(value)`` when not a real number.
    :rtype: str
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    specifier = "+" if signed else ""
    return f"{float(value):{specifier}.{precision}f}"


def truncate(text: str, width: int) -> str:
    """Shorten text to a maximum width using a trailing ellipsis.

    :param text: Text to shorten.
    :type text: str
    :param width: Maximum number of characters permitted.
    :type width: int
    :return: Text no longer than ``width`` characters.
    :rtype: str
    """
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    aligns: Sequence[str] | None = None,
    max_widths: Sequence[int] | None = None,
    palette: Palette | None = None,
    indent: str = "",
) -> str:
    """Render a fixed-width text table with a ruled header.

    :param headers: Column titles.
    :type headers: Sequence[str]
    :param rows: Row cells already converted to text.
    :type rows: Sequence[Sequence[str]]
    :param aligns: Per-column alignment, ``"l"`` or ``"r"``.
    :type aligns: Sequence[str] | None
    :param max_widths: Per-column truncation limits; zero means unlimited.
    :type max_widths: Sequence[int] | None
    :param palette: Styling switch applied to the header row.
    :type palette: Palette | None
    :param indent: Prefix written before every rendered line.
    :type indent: str
    :return: Multi-line table text without a trailing newline.
    :rtype: str
    """
    palette = palette or Palette(False)
    count = len(headers)
    aligns = list(aligns or ["l"] * count)
    limits = list(max_widths or [0] * count)
    cells = [
        [
            truncate(
                str(row[index]) if index < len(row) else "", limits[index] or 10**6
            )
            for index in range(count)
        ]
        for row in rows
    ]
    widths = [
        (
            max(len(str(headers[index])), *(len(row[index]) for row in cells))
            if cells
            else len(str(headers[index]))
        )
        for index in range(count)
    ]

    def line(values: Sequence[str]) -> str:
        parts = [
            (
                values[index].rjust(widths[index])
                if aligns[index] == "r"
                else values[index].ljust(widths[index])
            )
            for index in range(count)
        ]
        return indent + "  ".join(parts).rstrip()

    rule = (indent + "  ".join("-" * width for width in widths)).rstrip()
    out = [palette(line(list(headers)), "bold"), palette(rule, "dim")]
    out.extend(line(row) for row in cells)
    return "\n".join(out)


def render_fields(
    pairs: Iterable[tuple[str, str]],
    *,
    palette: Palette | None = None,
    indent: str = "",
) -> str:
    """Render aligned ``label  value`` lines.

    :param pairs: Label and value text pairs, rendered in order.
    :type pairs: Iterable[tuple[str, str]]
    :param palette: Styling switch applied to each label.
    :type palette: Palette | None
    :param indent: Prefix written before every rendered line.
    :type indent: str
    :return: Multi-line text without a trailing newline.
    :rtype: str
    """
    palette = palette or Palette(False)
    items = [(str(label), str(value)) for label, value in pairs]
    if not items:
        return ""
    width = max(len(label) for label, _ in items)
    return "\n".join(
        f"{indent}{palette(label.ljust(width), 'dim')}  {value}".rstrip()
        for label, value in items
    )


def render_bar(value: float, scale: float, width: int = 12) -> str:
    """Draw a signed magnitude bar that grows left or right from a centre.

    :param value: Signed magnitude to display.
    :type value: float
    :param scale: Absolute value mapped to a full half-width bar.
    :type scale: float
    :param width: Total character width, split evenly around the centre.
    :type width: int
    :return: Fixed-width bar text.
    :rtype: str
    """
    half = max(1, width // 2)
    if scale <= 0:
        return " " * (half * 2)
    filled = min(half, int(round(abs(value) / scale * half)))
    if value < 0:
        return " " * (half - filled) + "█" * filled + " " * half
    return " " * half + "█" * filled + " " * (half - filled)


def render_rule(width: int, *, palette: Palette | None = None, indent: str = "") -> str:
    """Render a horizontal rule.

    :param width: Rule length in characters.
    :type width: int
    :param palette: Styling switch applied to the rule.
    :type palette: Palette | None
    :param indent: Prefix written before the rule.
    :type indent: str
    :return: Single line of rule characters.
    :rtype: str
    """
    palette = palette or Palette(False)
    return indent + palette("─" * max(0, width), "dim")


def indent_block(text: str, prefix: str = "  ") -> str:
    """Prefix every line of a block of text.

    :param text: Block to indent.
    :type text: str
    :param prefix: Prefix applied to each line.
    :type prefix: str
    :return: Indented block without a trailing newline.
    :rtype: str
    """
    return "\n".join(prefix + line if line else line for line in text.splitlines())
