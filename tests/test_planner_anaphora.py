"""Planner anaphora resolution when conversation_context is present."""

from __future__ import annotations

from utils.query_planner import _fallback_plan, plan_query


def test_fallback_anaphora_stitches_prior_driver_topic():
    ctx = (
        "Driver: Hướng dẫn sấy kính lưng ghế\n"
        "Assistant: Press the heated seat button on the climate panel."
    )
    planned = _fallback_plan(
        "Chức năng đó bật bao lâu thì tắt?",
        conversation_context=ctx,
    )
    assert planned.intent == "procedure"
    # Wave4: whitelist focus entity (say kinh → rear window defroster), not full prior turn
    sq = planned.search_query.lower()
    assert "rear window" in sq or "defrost" in sq
    assert "bao lâu" in sq or "chức năng" in sq


def test_fallback_anaphora_skips_generic_focus():
    ctx = "Driver: Thông số xe này thế nào\nAssistant: Please specify the feature."
    planned = _fallback_plan(
        "nó hoạt động thế nào?",
        conversation_context=ctx,
    )
    # No whitelist hit → pass through unchanged
    assert planned.search_query == "nó hoạt động thế nào?"


def test_fallback_no_context_keeps_raw():
    planned = _fallback_plan("How to open the hood?")
    assert planned.search_query == "How to open the hood?"


def test_plan_query_accepts_conversation_context_kw():
    # Smoke: signature works with provider=none / fallback path
    planned = plan_query(
        "what was that number again?",
        language="en",
        conversation_context="Driver: Bronco wheel nut torque\nAssistant: 150 Nm",
    )
    assert planned.search_query
    assert isinstance(planned.search_query, str)
