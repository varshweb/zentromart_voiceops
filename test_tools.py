"""
tests/test_tools.py — Unit tests for order lookup and ticket creation.
Run: python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.order_tools import lookup_order, create_ticket


def test_lookup_without_verification():
    """Unverified lookup must never return order data."""
    result = lookup_order("12345678")
    assert result.get("verification_required") is True
    assert "status" not in result, "Order data leaked before verification!"
    assert "items" not in result


def test_lookup_with_wrong_verification():
    """Wrong verification value must not reveal order data."""
    result = lookup_order("12345678", verification_value="0000")
    assert result.get("verified") is False or result.get("verification_required") is True
    assert "items" not in result


def test_create_ticket_success():
    """Ticket creation with a valid summary should return created=True."""
    result = create_ticket(
        summary="Customer reports damaged TV on order 12345678; human review required under damage policy.",
        order_id="12345678",
        reason="high_value_damage",
        intent="damage_claim",
    )
    assert result.get("created") is True
    assert result.get("ticket_id", "").startswith("TKT")


def test_create_ticket_short_summary():
    """Short summaries should fail with created=False."""
    result = create_ticket(summary="help")
    assert result.get("created") is False
    assert "error" in result
