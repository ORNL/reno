"""Make sure index selection works on the interactive latex selector."""

import pytest

from reno.viz import ModelLatex


@pytest.mark.parametrize(
    "show_docs,start_name,stop_name,i,expected_name",
    [
        (False, None, None, 1, "faucet_volume"),
        (True, None, None, 1, "faucet_shutoff_time"),
        (False, "faucet_volume", "drain", 0, "faucet"),
        (False, "faucet_volume", "drain", 1, "drain"),
        (False, "faucet_volume", "drain", 2, None),
    ],
)
def test_equation_index(tub_model, show_docs, start_name, stop_name, i, expected_name):
    """Requesting an equation name via index with varying start/stop conditions and doc statuses should return the correct name."""
    latex = ModelLatex(tub_model, show_docs, start_name, stop_name)
    assert latex.find_equation_name_from_index(i) == expected_name


def test_lines_refname_reference(tub_model):
    """The equation lines aligned refname list should be correct."""
    latex = ModelLatex(tub_model, show_docs=True)
    assert latex._equation_lines_refname_reference() == [
        "faucet_shutoff_time",
        "faucet_shutoff_time",
        "faucet_volume",
        "faucet_volume",
        "faucet",
        "drain",
        "water_level",
        "water_level",
        "water_level",
        "final_water_level",
    ]
