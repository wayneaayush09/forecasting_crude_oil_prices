"""Leakage guard suite. Remaining stubs are xfail until implemented.
Implemented: no_future_rows, release_alignment -> tests/test_alignment.py"""
import pytest

pytestmark = pytest.mark.xfail(reason="stub: implement with pipeline", strict=False)


def test_scaler_fit_window():
    raise NotImplementedError


def test_decomposition_causal():
    raise NotImplementedError


def test_tool_as_of_filter():
    raise NotImplementedError


def test_sequence_after_split():
    raise NotImplementedError
