from __future__ import annotations

import pytest

from synde.errors import (
    SynDEDomainError,
    SynDEError,
    SynDEInputError,
    describe_domain,
    format_domain_message,
)
from synde.graph import GraphBuilder


def test_every_synde_error_stays_catchable_as_value_error() -> None:
    for error_type in (SynDEError, SynDEInputError, SynDEDomainError):
        assert issubclass(error_type, ValueError)


def test_format_domain_message_composes_subject_and_hint() -> None:
    message = format_domain_message("Rule violated.", subject="CCO", hint="Do this.")
    assert message.startswith("Rule violated. (input: CCO)")
    assert message.endswith("Hint: Do this.")


def test_format_domain_message_omits_absent_parts() -> None:
    assert format_domain_message("Rule violated.") == "Rule violated."


def test_domain_error_keeps_structured_context() -> None:
    error = SynDEDomainError(
        "Rule violated.", subject="CCO", hint="Do this.", details={"count": 2}
    )
    assert error.reason == "Rule violated."
    assert error.subject == "CCO"
    assert error.hint == "Do this."
    assert error.details == {"count": 2}


def test_describe_domain_summarizes_elements_and_charges() -> None:
    description = describe_domain(("C", "H", "O"), (0,))
    assert "[C H O]" in description
    assert "[0]" in description
    assert "closed-shell" in description


def test_invalid_smiles_reports_the_offending_string() -> None:
    with pytest.raises(SynDEInputError) as excinfo:
        GraphBuilder.from_smiles("C1CC")
    message = str(excinfo.value)
    assert "'C1CC'" in message
    assert "Hint:" in message


def test_invalid_reaction_smiles_explains_the_expected_form() -> None:
    with pytest.raises(SynDEInputError) as excinfo:
        GraphBuilder.reaction_states_from_smiles("not-a-reaction")
    assert "reactants>>products" in str(excinfo.value)


def test_isotope_rejection_names_the_mass_numbers() -> None:
    with pytest.raises(SynDEDomainError) as excinfo:
        GraphBuilder.from_smiles("[13CH4]")
    message = str(excinfo.value)
    assert "mass numbers [13]" in message
    assert "Hint:" in message
    assert excinfo.value.details["isotopes"] == [13]
