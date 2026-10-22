"""
evaluation.py — Step 7: Automated evaluation on practice data.

Evaluates:
  - questions.csv  → single-turn accuracy (Step 2 vs Step 3)
  - conversations.csv → multi-turn tool/verification accuracy

Usage:
  python -m src.evaluation --step 2          # LLM-only (no RAG)
  python -m src.evaluation --step 3          # RAG-enabled
  python -m src.evaluation --compare         # Both, then compare
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import PRACTICE_DIR, OUTPUTS_DIR, REFERENCE_DATE
from src.llm_agent import simple_respond
from src.rag_pipeline import retrieve, ensure_index
from src.conversation_manager import new_conversation, chat_turn

QUESTIONS_CSV    = PRACTICE_DIR / "questions.csv"
CONVERSATIONS_CSV= PRACTICE_DIR / "conversations.csv"


# ── Utility helpers ───────────────────────────────────────────────────────────
def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _check_must_include(reply: str, must_include: str) -> tuple[bool, list[str]]:
    """
    Return (passed, missing_facts).
    Semicolon-separated facts; slash-separated alternatives (any one is enough).
    """
    if not must_include or must_include.strip() == "":
        return True, []

    reply_norm = _normalize(reply)
    missing = []
    facts = [f.strip() for f in must_include.split(";") if f.strip()]

    for fact in facts:
        alternatives = [a.strip() for a in fact.split("/")]
        if not any(_normalize(alt) in reply_norm for alt in alternatives):
            missing.append(fact)

    return len(missing) == 0, missing


def _check_must_not(reply: str, must_not: str) -> tuple[bool, list[str]]:
    if not must_not or must_not.strip() == "":
        return True, []

    reply_norm = _normalize(reply)
    violations = []
    for phrase in [p.strip() for p in must_not.split(";") if p.strip()]:
        if _normalize(phrase) in reply_norm:
            violations.append(phrase)
    return len(violations) == 0, violations


def _evidence_match(retrieved_ids: list[str], gold: str) -> float:
    """Fraction of gold section IDs found in retrieved chunks."""
    if not gold or gold.strip() == "":
        return 1.0
    gold_ids = {g.strip() for g in re.split(r"[;,]", gold) if g.strip()}
    if not gold_ids:
        return 1.0
    hit = sum(1 for g in gold_ids if any(g in r for r in retrieved_ids))
    return hit / len(gold_ids)


# ── Single-question evaluation ────────────────────────────────────────────────
def evaluate_question(row: dict, use_rag: bool) -> dict:
    """Evaluate one row from questions.csv."""
    question = row.get("question", "")
    lang     = row.get("language", "en")

    evidence   = []
    if use_rag:
        evidence = retrieve(question)
    retrieved_ids = [c.get("section_id", c.get("doc_id", "")) for c in evidence]

    reply = simple_respond(user_message=question, context_chunks=evidence if use_rag else None)

    mi_pass, missing = _check_must_include(reply, row.get("must_include", ""))
    mn_pass, violations = _check_must_not(reply, row.get("must_not_include", ""))
    ev_score = _evidence_match(retrieved_ids, row.get("gold_evidence", ""))

    return {
        "id":            row.get("id", ""),
        "question":      question,
        "language":      lang,
        "expected_decision": row.get("expected_decision", ""),
        "reply":         reply,
        "must_include_pass": mi_pass,
        "missing_facts": missing,
        "must_not_pass": mn_pass,
        "violations":    violations,
        "evidence_recall": round(ev_score, 3),
        "retrieved_sections": retrieved_ids,
    }


# ── Questions batch evaluation ────────────────────────────────────────────────
def evaluate_questions(use_rag: bool = True, limit: int = None) -> list[dict]:
    """Run all (or first `limit`) practice questions."""
    if use_rag:
        ensure_index()

    df = pd.read_csv(QUESTIONS_CSV, dtype=str, keep_default_na=False)
    if limit:
        df = df.head(limit)

    results = []
    for _, row in df.iterrows():
        result = evaluate_question(row.to_dict(), use_rag=use_rag)
        results.append(result)
        status = "✓" if result["must_include_pass"] and result["must_not_pass"] else "✗"
        print(f"  [{status}] {result['id']} — {result['question'][:60]}")

    return results


# ── Conversation evaluation ───────────────────────────────────────────────────
def evaluate_conversations(use_rag: bool = True, limit_conv: int = None) -> list[dict]:
    """Evaluate multi-turn conversations from conversations.csv."""
    if use_rag:
        ensure_index()

    df = pd.read_csv(CONVERSATIONS_CSV, dtype=str, keep_default_na=False)
    conv_ids = df["conversation_id"].unique().tolist()
    if limit_conv:
        conv_ids = conv_ids[:limit_conv]

    all_results = []

    for conv_id in conv_ids:
        turns = df[df["conversation_id"] == conv_id].sort_values("turn")
        state = new_conversation()
        print(f"\n  [Conv {conv_id}] {len(turns)} turns")

        for _, row in turns.iterrows():
            msg  = row.get("customer_message", "")
            reply, state = chat_turn(msg, state, use_rag=use_rag)

            mi_pass, missing   = _check_must_include(reply, row.get("must_include", ""))
            mn_pass, violations= _check_must_not(reply, row.get("must_not_include", ""))
            retrieved_ids      = [c.get("section_id", "") for c in state.last_evidence]
            ev_score           = _evidence_match(retrieved_ids, row.get("gold_evidence", ""))
            tool_calls_made    = [t.get("tool", "") for t in state.last_tool_results]
            expected_tools     = row.get("expected_tool_calls", "")

            all_results.append({
                "conversation_id":    conv_id,
                "turn":               row.get("turn", ""),
                "customer_message":   msg,
                "reply":              reply,
                "expected_decision":  row.get("expected_decision", ""),
                "must_include_pass":  mi_pass,
                "missing_facts":      missing,
                "must_not_pass":      mn_pass,
                "violations":         violations,
                "evidence_recall":    round(ev_score, 3),
                "tool_calls_made":    tool_calls_made,
                "expected_tool_calls":expected_tools,
            })
            status = "✓" if mi_pass and mn_pass else "✗"
            print(f"    Turn {row.get('turn', '?')} [{status}]: {msg[:50]}")

    return all_results


# ── Summary statistics ────────────────────────────────────────────────────────
def summarize(results: list[dict], label: str) -> dict:
    if not results:
        return {}
    n       = len(results)
    mi_pass = sum(1 for r in results if r.get("must_include_pass", True))
    mn_pass = sum(1 for r in results if r.get("must_not_pass", True))
    ev_avg  = sum(r.get("evidence_recall", 0) for r in results) / n

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"  Questions:            {n}")
    print(f"  Must-include pass:    {mi_pass}/{n} ({100*mi_pass//n}%)")
    print(f"  Must-not pass:        {mn_pass}/{n} ({100*mn_pass//n}%)")
    print(f"  Avg evidence recall:  {ev_avg:.2%}")
    print(f"{'='*50}\n")

    return {
        "label": label, "n": n,
        "must_include_pct": round(100 * mi_pass / n, 1),
        "must_not_pct":     round(100 * mn_pass / n, 1),
        "avg_evidence_recall": round(ev_avg, 3),
    }


def save_results(results: list[dict], filename: str) -> Path:
    out_path = OUTPUTS_DIR / filename
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            for row in results:
                writer.writerow({k: str(v) for k, v in row.items()})
    print(f"  Saved: {out_path}")
    return out_path


# ── CLI entry ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoiceOps evaluation")
    parser.add_argument("--step", choices=["2", "3"], help="Evaluate Step 2 (LLM-only) or Step 3 (RAG)")
    parser.add_argument("--compare", action="store_true", help="Compare both steps")
    parser.add_argument("--conv", action="store_true", help="Evaluate conversations too")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    args = parser.parse_args()

    summaries = []

    if args.compare or args.step == "2":
        print("\n[Step 2] LLM-only evaluation...")
        r2 = evaluate_questions(use_rag=False, limit=args.limit)
        save_results(r2, "eval_step2_questions.csv")
        summaries.append(summarize(r2, "Step 2 — LLM only"))

    if args.compare or args.step == "3":
        print("\n[Step 3] RAG-enabled evaluation...")
        r3 = evaluate_questions(use_rag=True, limit=args.limit)
        save_results(r3, "eval_step3_questions.csv")
        summaries.append(summarize(r3, "Step 3 — RAG"))

    if args.conv:
        print("\n[Step 4] Conversation evaluation...")
        rc = evaluate_conversations(use_rag=True, limit_conv=args.limit)
        save_results(rc, "eval_conversations.csv")
        summarize(rc, "Step 4 — Conversations")

    if args.compare and len(summaries) == 2:
        s2, s3 = summaries
        print("\n=== COMPARISON ===")
        print(f"  Must-include:   {s2['must_include_pct']}% → {s3['must_include_pct']}% "
              f"(+{s3['must_include_pct']-s2['must_include_pct']:.1f}pp)")
        print(f"  Evidence recall: — → {s3['avg_evidence_recall']:.2%}")
