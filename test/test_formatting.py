from __future__ import annotations

import io

import pytest

from synde.formatting import (
    Palette,
    format_float,
    indent_block,
    render_bar,
    render_fields,
    render_rule,
    render_table,
    supports_color,
    terminal_width,
    truncate,
)


def test_palette_is_transparent_when_disabled() -> None:
    palette = Palette(False)
    assert palette("text", "bold", "red") == "text"


def test_palette_wraps_text_when_enabled() -> None:
    palette = Palette(True)
    styled = palette("text", "bold")
    assert styled.startswith("\033[1m")
    assert styled.endswith("\033[0m")
    assert "text" in styled


def test_supports_color_honours_no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    assert supports_color(io.StringIO()) is False


def test_supports_color_honours_force_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert supports_color(io.StringIO()) is True


def test_supports_color_is_false_for_non_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert supports_color(io.StringIO()) is False


def test_terminal_width_is_clamped() -> None:
    assert 60 <= terminal_width() <= 160


@pytest.mark.parametrize(
    "value,expected",
    [(1.5, "1.5000"), (-2.0, "-2.0000"), (3, "3.0000"), ("x", "x"), (True, "True")],
)
def test_format_float(value: object, expected: str) -> None:
    assert format_float(value) == expected


def test_format_float_signed_marks_positive_values() -> None:
    assert format_float(1.5, 2, signed=True) == "+1.50"
    assert format_float(-1.5, 2, signed=True) == "-1.50"


def test_truncate_adds_ellipsis_only_when_needed() -> None:
    assert truncate("abcdef", 10) == "abcdef"
    assert truncate("abcdef", 4) == "abc…"
    assert truncate("abcdef", 0) == ""


def test_render_table_aligns_and_rules_columns() -> None:
    text = render_table(
        ["name", "value"], [["a", "1"], ["bbb", "22"]], aligns=["l", "r"]
    )
    lines = text.splitlines()
    assert lines[0].startswith("name")
    assert set(lines[1]) <= {"-", " "}
    assert lines[2].endswith("1")
    assert lines[3].startswith("bbb")


def test_render_table_truncates_to_max_width() -> None:
    text = render_table(["term"], [["a" * 40]], max_widths=[10])
    assert "…" in text
    assert "a" * 40 not in text


def test_render_table_has_no_trailing_whitespace() -> None:
    text = render_table(["a", "b"], [["1", ""]])
    assert all(line == line.rstrip() for line in text.splitlines())


def test_render_fields_aligns_labels() -> None:
    text = render_fields([("short", "1"), ("much longer", "2")])
    first, second = text.splitlines()
    assert first.index("1") == second.index("2")


def test_render_fields_handles_empty_input() -> None:
    assert render_fields([]) == ""


def test_render_bar_is_fixed_width_and_signed() -> None:
    negative = render_bar(-1.0, 1.0, width=8)
    positive = render_bar(1.0, 1.0, width=8)
    assert len(negative) == len(positive) == 8
    assert negative.startswith("█")
    assert positive.endswith("█")
    assert render_bar(1.0, 0.0, width=8).strip() == ""


def test_render_rule_and_indent_block() -> None:
    assert render_rule(5) == "─" * 5
    assert indent_block("a\nb", "> ") == "> a\n> b"
