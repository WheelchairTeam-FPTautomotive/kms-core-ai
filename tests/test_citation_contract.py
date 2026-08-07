"""Citation contract for golden RAG successes (issue #10 DoD).

Every status=success RAG answer must include ≥1 citation with:
document_id, basename document_name, page, matched_text (snippet).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipelines.solve_problem import solve_automotive_query_live

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "test_queries" / "golden_set_s2.json"
REQUIRED_CITATION_FIELDS = ("document_id", "document_name", "page", "matched_text")


def _load_golden_success_cases() -> list[dict]:
    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))
    return [c for c in cases if isinstance(c, dict) and c.get("expected_status") == "success"]


def assert_citation_contract(result: dict) -> None:
    """Fail if a RAG success violates the traceability contract."""
    assert result["status"] == "success"
    citations = result.get("citations") or []
    assert len(citations) >= 1, f"success with empty citations: {result.get('query')!r}"
    for i, cite in enumerate(citations):
        for field in REQUIRED_CITATION_FIELDS:
            assert field in cite, f"citation[{i}] missing {field}: {cite}"
            assert cite[field] is not None and str(cite[field]).strip() != "", (
                f"citation[{i}] empty {field}: {cite}"
            )
        name = str(cite["document_name"])
        assert "/" not in name and "\\" not in name, (
            f"document_name must be basename only, got {name!r}"
        )


# --- START MODIFICATION ---
@pytest.mark.parametrize(
    "case",
    _load_golden_success_cases(),
    ids=lambda c: c.get("id", "unknown"),
)
@patch("core.answer_generator.generate_driver_answer")
@patch("pipelines.solve_problem.get_chroma_collection")
@patch("pipelines.solve_problem._merge_multi_query_hits")
@patch("pipelines.solve_problem.expand_retrieval_queries")
def test_golden_success_citation_contract_mocked(
    mock_expand,
    mock_merge,
    mock_get_collection,
    mock_generate,
    case: dict,
):
    """Mock retrieval + grounded LLM so every golden *success* case hits the contract."""
    query = case["query"]
    language = case.get("language") or "en"
    keywords = case.get("expected_doc_keywords") or ["manual"]
    doc_hint = str(keywords[0]).replace(" ", "_")

    mock_expand.return_value = [query]
    mock_merge.return_value = [
        (
            f"doc_{case.get('id', 'x')}",
            f"{doc_hint}.pdf",
            "Section 1",
            12,
            f"Grounded snippet mentioning {doc_hint} for query.",
            0.42,
        )
    ]
    mock_collection = MagicMock()
    mock_collection.metadata = {"hnsw:space": "l2"}
    mock_get_collection.return_value = mock_collection
    mock_generate.return_value = (
        f"GROUNDED: yes\nAccording to the manual, {doc_hint} procedure applies."
    )

    result = solve_automotive_query_live(query, language=language)
    assert_citation_contract(result)
    assert result["query"] == query


def test_success_empty_citations_fails_contract_helper():
    with pytest.raises(AssertionError, match="empty citations"):
        assert_citation_contract(
            {"status": "success", "query": "q", "answer": "a", "citations": []}
        )


def test_nested_path_document_name_fails_contract_helper():
    with pytest.raises(AssertionError, match="basename"):
        assert_citation_contract(
            {
                "status": "success",
                "query": "q",
                "answer": "a",
                "citations": [
                    {
                        "document_id": "abc",
                        "document_name": "year/adas/manual.pdf",
                        "page": 1,
                        "matched_text": "snippet",
                    }
                ],
            }
        )


@patch("core.answer_generator.generate_driver_answer")
@patch("pipelines.solve_problem.get_chroma_collection")
@patch("pipelines.solve_problem._merge_multi_query_hits")
@patch("pipelines.solve_problem.expand_retrieval_queries")
def test_live_success_keeps_required_fields(
    mock_expand, mock_merge, mock_get_collection, mock_generate
):
    mock_expand.return_value = ["How does HVAC work?"]
    mock_merge.return_value = [
        ("hash123", "Camry HVAC.pdf", "Climate", 7, "HVAC blower control…", 0.5)
    ]
    col = MagicMock()
    col.metadata = {"hnsw:space": "l2"}
    mock_get_collection.return_value = col
    mock_generate.return_value = "GROUNDED: yes\nHVAC is controlled via the climate panel."

    result = solve_automotive_query_live("How does HVAC work?", language="en")
    assert_citation_contract(result)
    cite = result["citations"][0]
    assert cite["document_id"] == "hash123"
    assert cite["document_name"] == "Camry HVAC.pdf"
    assert cite["page"] == 7
    assert "HVAC" in cite["matched_text"]


@patch("core.answer_generator.generate_driver_answer")
@patch("pipelines.solve_problem.get_chroma_collection")
@patch("pipelines.solve_problem._merge_multi_query_hits")
@patch("pipelines.solve_problem.expand_retrieval_queries")
def test_ungrounded_success_path_clears_citations(
    mock_expand, mock_merge, mock_get_collection, mock_generate
):
    """Honesty gate: soft-deny must not return success with ghost citations."""
    mock_expand.return_value = ["teleporter on Bronco"]
    mock_merge.return_value = [
        ("hash999", "Bronco.pdf", "Intro", 1, "Bronco overview", 0.4)
    ]
    col = MagicMock()
    col.metadata = {"hnsw:space": "l2"}
    mock_get_collection.return_value = col
    mock_generate.return_value = (
        "No matching information was found in the technical documents."
    )

    result = solve_automotive_query_live("teleporter on Bronco", language="en")
    assert result["status"] == "not_found"
    assert result["citations"] == []
# --- END MODIFICATION ---
