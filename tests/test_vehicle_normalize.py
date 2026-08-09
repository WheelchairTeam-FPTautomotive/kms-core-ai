"""Unit tests for planner vehicle normalize + OEM prefer + extractive safety."""

from __future__ import annotations

from core.answer_generator import extractive_summary
from pipelines.solve_problem import _prefer_oem_among_gated
from utils.query_expand import expand_retrieval_queries, matched_oem_token_set
from utils.vehicle_meta import (
    normalize_query_vehicle,
    set_corpus_model_make_map,
)


def test_normalize_tucson_as_make_remaps_to_hyundai():
    assert normalize_query_vehicle("tucson", "") == ("hyundai", "tucson")
    assert normalize_query_vehicle("Tucson", None) == ("hyundai", "tucson")


def test_normalize_keeps_real_make_model():
    assert normalize_query_vehicle("hyundai", "tucson") == ("hyundai", "tucson")
    assert normalize_query_vehicle("ford", "bronco") == ("ford", "bronco")


def test_normalize_bronco_as_make_remaps():
    assert normalize_query_vehicle("bronco", "") == ("ford", "bronco")


def test_normalize_corpus_map_fills_unknown_model():
    set_corpus_model_make_map({"palette": "kia"})
    try:
        assert normalize_query_vehicle("palette", "") == ("kia", "palette")
        assert normalize_query_vehicle("", "palette") == ("kia", "palette")
        # Static override still wins for tucson
        assert normalize_query_vehicle("tucson", "") == ("hyundai", "tucson")
    finally:
        set_corpus_model_make_map({})


def test_epb_oem_expand_prepends_english():
    variants = expand_retrieval_queries(
        "Ioniq 5 kéo công tắc phanh tay điện tử EPB thế nào"
    )
    assert variants
    joined = " ".join(variants).lower()
    assert "electronic parking brake" in joined or "epb switch" in joined
    assert variants[0].lower() in {
        "electronic parking brake",
        "epb switch",
        "parking brake switch",
        "parking brake",
    }


def test_matched_oem_token_set_epb():
    toks = matched_oem_token_set("Ioniq EPB phanh tay")
    assert toks
    assert any("epb" in t or "parking" in t for t in toks)


def test_prefer_oem_epb_among_gated():
    cites = [
        {
            "document_name": "Ioniq",
            "section": "HV",
            "page": 1,
            "matched_text": "High-voltage battery introduction overview",
            "document_id": "a",
        },
        {
            "document_name": "Ioniq",
            "section": "Brake",
            "page": 2,
            "matched_text": "Pull the EPB switch to apply the electronic parking brake",
            "document_id": "b",
        },
    ]
    out = _prefer_oem_among_gated(cites, "Ioniq EPB phanh tay điện tử")
    assert out[0]["document_id"] == "b"
    assert len(out) == 2


def test_prefer_oem_defrost_among_gated():
    cites = [
        {
            "document_name": "Tucson",
            "section": "Climate intro",
            "page": 1,
            "matched_text": "Automatic climate control system overview",
            "document_id": "a",
        },
        {
            "document_name": "Tucson",
            "section": "Defrost",
            "page": 2,
            "matched_text": "Press the rear window defroster button to turn on",
            "document_id": "b",
        },
    ]
    out = _prefer_oem_among_gated(cites, "Tucson làm sao bật sấy kính sau")
    assert out[0]["document_id"] == "b"


def test_prefer_oem_does_not_invent_candidates():
    cites = [
        {
            "document_name": "Ioniq",
            "section": "HV",
            "page": 1,
            "matched_text": "High-voltage battery introduction",
            "document_id": "a",
        },
    ]
    out = _prefer_oem_among_gated(cites, "EPB switch")
    assert out[0]["document_id"] == "a"


def test_extractive_summary_prefers_warning_sentence():
    snippets = [
        "1. [Doc - Sec (Trang 1)]: Apply the brake normally. "
        "WARNING: Do not drive with the EPB engaged. "
        "Then release the switch."
    ]
    out = extractive_summary(snippets, max_sentences=2, language="en")
    assert "WARNING" in out or "warning" in out.lower()
