"""
conversation_manager.py — Step 4: Multi-turn conversation state + tool routing.

Handles:
  - Conversation history (Gemini format)
  - Order verification state machine
  - Tool call detection and execution
  - RAG retrieval per turn
  - Final LLM response generation
"""
import re
import json
from dataclasses import dataclass, field
from typing import Optional

from src.config import AGENT_NAME, STORE_NAME, REFERENCE_DATE
from src.llm_agent import simple_respond
from src.rag_pipeline import retrieve, ensure_index
from src.order_tools import lookup_order, create_ticket

# ── Verification state ────────────────────────────────────────────────────────
MAX_VERIFY_ATTEMPTS = 1   # one retry allowed per the dataset rules


@dataclass
class VerificationState:
    order_id: str = ""
    attempts: int = 0
    verified: bool = False
    order_data: Optional[dict] = None


@dataclass
class ConversationState:
    history: list = field(default_factory=list)            # Gemini chat history
    verification: VerificationState = field(default_factory=VerificationState)
    last_tool_results: list = field(default_factory=list)  # tool call log
    last_evidence: list = field(default_factory=list)      # RAG chunks used


# ── Tool-call detection (LLM-guided with regex fallback) ─────────────────────
_ORDER_ID_RE  = re.compile(r"\b(\d{8})\b")
_PIN_RE       = re.compile(r"\b(\d{6})\b")
_PHONE4_RE    = re.compile(r"\b(\d{4})\b")
_TICKET_TRIGGERS = re.compile(
    r"\b(human|agent|escalat|handover|hand.over|complaint|not resolv|"
    r"speak to|talk to|connect me|raise.*ticket|create.*ticket)\b",
    re.IGNORECASE,
)
_VERIFY_TRIGGER = re.compile(
    r"\b(order|track|status|cancel|return|refund|item|deliver)\b",
    re.IGNORECASE,
)


def _detect_order_id(text: str) -> Optional[str]:
    # Match 8-digit IDs; also accept spaced forms like "8978 2837"
    clean = re.sub(r"\s", "", text)
    m = _ORDER_ID_RE.search(clean)
    return m.group(1) if m else None


def _detect_verification_value(text: str) -> Optional[str]:
    """Extract a 4-digit phone suffix or 6-digit pincode from user text."""
    pin = _PIN_RE.search(text)
    if pin:
        return pin.group(1)
    phone = _PHONE4_RE.search(text)
    if phone:
        return phone.group(1)
    return None


# ── Routing logic ─────────────────────────────────────────────────────────────
def _build_system_context(
    state: ConversationState,
    tool_notes: str = "",
    evidence: list[dict] | None = None,
) -> str:
    """Build the evidence + tool context block prepended to each turn."""
    parts = []

    if evidence:
        parts.append("## Retrieved policy / help documents:")
        for chunk in evidence:
            sid    = chunk.get("section_id", chunk.get("doc_id", "?"))
            status = chunk.get("status", "")
            parts.append(f"[{sid}] ({status})\n{chunk['text']}")
        parts.append("")

    if state.verification.verified and state.verification.order_data:
        od = state.verification.order_data
        parts.append(f"## Verified order data for order {od.get('order_id', '')}:")
        # Include key fields; avoid leaking phone
        safe_keys = [
            "order_id", "status", "payment_method", "delivery_type",
            "promised_date", "shipped_on", "delivered_on", "total_inr",
            "return_status", "return_refund_initiated_on",
            "return_refund_method", "return_refund_amount_inr",
            "cancellation_refund_initiated_on", "rto_refund_initiated_on",
            "damage_status", "damage_issue", "customer_first_name", "is_plus",
        ]
        od_lines = {k: od.get(k) for k in safe_keys if od.get(k) is not None}
        items = od.get("items", [])
        if items:
            item_desc = "; ".join(
                f"{it.get('title','?')} x{it.get('qty',1)}" for it in items
            )
            od_lines["items"] = item_desc
        parts.append(json.dumps(od_lines, ensure_ascii=False, indent=2))
        parts.append("")

    if tool_notes:
        parts.append(f"## Tool results:\n{tool_notes}\n")

    return "\n".join(parts)


# ── Main conversation turn ────────────────────────────────────────────────────
def chat_turn(
    user_message: str,
    state: ConversationState,
    use_rag: bool = True,
) -> tuple[str, ConversationState]:
    """
    Process one customer turn. Returns (reply_text, updated_state).

    Steps:
      1. Detect order IDs and verification values.
      2. Call lookup_order if appropriate.
      3. Detect escalation triggers → create_ticket.
      4. Retrieve RAG chunks.
      5. Build prompt and call LLM.
      6. Update history.
    """
    tool_notes = ""
    state.last_tool_results = []
    state.last_evidence = []

    # ── 1. Detect order ID ──────────────────────────────────────────────────
    detected_oid = _detect_order_id(user_message)
    if detected_oid and not state.verification.verified:
        if state.verification.order_id != detected_oid:
            # New order ID mentioned — reset verification
            state.verification = VerificationState(order_id=detected_oid)

    # ── 2. Order lookup & verification ───────────────────────────────────────
    if state.verification.order_id and not state.verification.verified:
        verification_value = _detect_verification_value(user_message)

        if verification_value and state.verification.attempts <= MAX_VERIFY_ATTEMPTS:
            state.verification.attempts += 1
            result = lookup_order(
                state.verification.order_id,
                verification_value=verification_value,
            )
            state.last_tool_results.append({
                "tool": "lookup_order",
                "order_id": state.verification.order_id,
                "verification_value": "****",      # never log the secret
                "result_keys": list(result.keys()),
            })

            if result.get("verified"):
                state.verification.verified = True
                state.verification.order_data = result
                tool_notes += (
                    f"lookup_order succeeded. Order {state.verification.order_id} verified.\n"
                )
            else:
                tool_notes += (
                    f"lookup_order: verification failed (attempt "
                    f"{state.verification.attempts}/{MAX_VERIFY_ATTEMPTS + 1}).\n"
                )

        elif _VERIFY_TRIGGER.search(user_message) and not verification_value:
            # Customer mentioned an order topic but gave no verification value
            if state.verification.order_id:
                # Do an unverified lookup just to confirm the order exists
                result = lookup_order(state.verification.order_id)
                state.last_tool_results.append({
                    "tool": "lookup_order",
                    "order_id": state.verification.order_id,
                    "unverified": True,
                })
                tool_notes += (
                    f"lookup_order: verification required for order "
                    f"{state.verification.order_id}. Asking customer.\n"
                )

    # ── 3. Ticket creation on escalation signals ─────────────────────────────
    if _TICKET_TRIGGERS.search(user_message) and state.verification.verified:
        summary = (
            f"Customer request for human handover. "
            f"Order: {state.verification.order_id}. "
            f"Issue: {user_message[:200].strip()}. "
            f"Verified caller, agent unable to resolve automatically."
        )
        ticket_result = create_ticket(
            summary=summary,
            order_id=state.verification.order_id,
            reason="customer_request",
            intent="escalation",
        )
        state.last_tool_results.append({"tool": "create_ticket", **ticket_result})
        if ticket_result.get("created"):
            tool_notes += (
                f"create_ticket succeeded. Ticket ID: {ticket_result['ticket_id']}.\n"
            )
        else:
            tool_notes += f"create_ticket failed: {ticket_result.get('error', 'unknown')}.\n"

    # ── 4. RAG retrieval ─────────────────────────────────────────────────────
    evidence = []
    if use_rag:
        try:
            ensure_index()
            evidence = retrieve(user_message)
            state.last_evidence = evidence
        except Exception as exc:
            tool_notes += f"[RAG retrieval error: {exc}]\n"

    # ── 5. Build full prompt and call LLM ────────────────────────────────────
    context_block = _build_system_context(state, tool_notes, evidence)
    augmented_message = (
        f"{context_block}\n## Customer message:\n{user_message}"
        if context_block.strip()
        else user_message
    )

    reply = simple_respond(
        user_message=augmented_message,
        history=state.history,
        context_chunks=None,   # evidence already in augmented_message
    )

    # ── 6. Update conversation history ───────────────────────────────────────
    state.history.append({"role": "user",  "parts": [user_message]})
    state.history.append({"role": "model", "parts": [reply]})

    return reply, state


def new_conversation() -> ConversationState:
    """Create a fresh conversation state."""
    return ConversationState()
