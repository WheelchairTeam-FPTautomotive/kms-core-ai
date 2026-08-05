"""Unit tests for citation honesty grounding detector (no LLM)."""

from core.grounding import is_ungrounded, looks_ungrounded, parse_grounded_answer


def test_parse_grounded_yes_strips_header():
    raw = "GROUNDED: yes\nCheck tire pressure when tires are cold."
    body, flag = parse_grounded_answer(raw)
    assert flag is True
    assert body == "Check tire pressure when tires are cold."
    assert "GROUNDED" not in body


def test_parse_grounded_no_strips_header():
    raw = "GROUNDED: no\nNo matching information was found in the technical documents."
    body, flag = parse_grounded_answer(raw)
    assert flag is False
    assert "GROUNDED" not in body
    assert "No matching" in body


def test_parse_grounded_only_line():
    body, flag = parse_grounded_answer("GROUNDED: no")
    assert flag is False
    assert body == ""


def test_parse_without_marker():
    body, flag = parse_grounded_answer("According to the Bronco manual, use Normal mode.")
    assert flag is None
    assert body.startswith("According")


def test_flag_no_alone_does_not_flip():
    # llama3.2:3b often labels valid answers GROUNDED: no — ignore flag alone
    assert is_ungrounded("Check tire pressure when tires are cold.", False) is False


def test_flag_no_empty_body_is_ungrounded():
    assert is_ungrounded("", False) is True


def test_soft_deny_body_overrides_grounded_yes():
    # Honesty: soft-deny phrasing still drops citations even if marker says yes
    answer = "No matching information was found in the technical documents."
    assert is_ungrounded(answer, True) is True


def test_phrase_en_insufficient_context():
    assert looks_ungrounded(
        "There is no mention of a teleporter in the glove box."
    )
    assert looks_ungrounded(
        "No matching information was found in the vehicle technical documents."
    )
    assert is_ungrounded(
        "The feature is not found in the manual for this vehicle.", None
    )


def test_phrase_vi_insufficient_context():
    assert looks_ungrounded("Không tìm thấy thông tin phù hợp trong tài liệu.")
    assert looks_ungrounded("Tài liệu không đề cập đến tính năng này.")
    assert is_ungrounded("Ngữ cảnh không đủ để trả lời.", None)


def test_safety_warning_en_not_false_positive():
    text = "Do not turn off the engine while driving."
    assert looks_ungrounded(text) is False
    assert is_ungrounded(text, None) is False


def test_safety_warning_vi_not_false_positive():
    text = "Không tắt động cơ khi xe đang chạy."
    assert looks_ungrounded(text) is False
    assert is_ungrounded(text, None) is False
