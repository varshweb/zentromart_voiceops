"""
order_tools.py — Step 4: Thin wrappers around the dataset's tools/tools.py.

Adds path-agnostic import so the project works from any working directory.
All logic remains in the original tools.py — we never duplicate it.
"""
import sys
from pathlib import Path

from src.config import TOOLS_DIR

# Make the dataset's tools package importable
_tools_parent = str(TOOLS_DIR.parent)
if _tools_parent not in sys.path:
    sys.path.insert(0, _tools_parent)

from tools.tools import lookup_order as _lookup_order       # noqa: E402
from tools.tools import create_ticket as _create_ticket     # noqa: E402


def lookup_order(order_id: str, verification_value: str = None) -> dict:
    """
    Proxy to the dataset's lookup_order mock tool.

    Without verification_value: returns verification_required=True (no private data).
    With a valid 4-digit phone suffix or 6-digit delivery pincode: returns full order.
    With an invalid value: returns verified=False.
    """
    if verification_value is not None:
        return _lookup_order(str(order_id), verification_value=str(verification_value))
    return _lookup_order(str(order_id))


def create_ticket(
    summary: str,
    order_id: str = None,
    reason: str = None,
    intent: str = None,
) -> dict:
    """
    Proxy to the dataset's create_ticket mock tool.

    Returns {"created": True, "ticket_id": "TKT..."} on success.
    Returns {"created": False, "error": "..."} if summary is too short (<20 chars).
    ONLY claim success when created=True.
    """
    return _create_ticket(
        summary=summary,
        order_id=order_id,
        reason=reason,
        intent=intent,
    )
