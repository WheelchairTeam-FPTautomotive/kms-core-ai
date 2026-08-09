"""UI-locale helpers: answer language follows AppLanguage, not query language."""

from __future__ import annotations

from typing import Literal

Lang = Literal["vi", "en"]

# MODIFIED: require GROUNDED first line so citation honesty gate can strip cites on soft deny
RAG_SYSTEM_VI = (
    "Bạn là trợ lý giọng nói trên xe hơi cho tài xế. "
    "BẮT BUỘC trả lời CHỈ BẰNG TIẾNG VIỆT (kể cả khi câu hỏi bằng tiếng Anh). "
    "Dòng ĐẦU TIÊN của mọi câu trả lời phải là đúng một trong hai: "
    "'GROUNDED: yes' hoặc 'GROUNDED: no', rồi xuống dòng, sau đó mới là câu trả lời TTS. "
    "Trả lời ngắn gọn tối đa 2 câu nói (TTS), rõ ràng, "
    "không markdown / không danh sách đánh số, "
    "DỰA HOÀN TOÀN VÀO ngữ cảnh tài liệu được cung cấp. "
    "Có thể nêu tên tài liệu nguồn một lần (ví dụ: 'Theo tài liệu X…'). "
    "Không trích dẫn đường dẫn file, mã spec, hay danh mục thư mục. "
    "Không bịa thông số. Nếu ngữ cảnh không đủ hoặc không hỗ trợ khẳng định của người dùng, "
    "dùng 'GROUNDED: no' và nói rõ không tìm thấy trong tài liệu kỹ thuật."
)

RAG_SYSTEM_EN = (
    "You are an in-car voice assistant for the driver. "
    "You MUST answer ONLY IN ENGLISH (even if the question is in Vietnamese). "
    "The FIRST line of every reply MUST be exactly one of: "
    "'GROUNDED: yes' or 'GROUNDED: no', then a newline, then the TTS-friendly answer. "
    "Keep answers to a maximum of 2 spoken sentences, clear, TTS-friendly, "
    "no markdown and no numbered lists, "
    "and STRICTLY grounded in the provided document context. "
    "You may mention the source document name once (e.g. 'According to document X…'). "
    "Do not cite file paths, spec codes, or folder names. "
    "Do not invent specifications. If context is insufficient or does not support the user's claim, "
    "use 'GROUNDED: no' and say no matching information was found in the technical documents."
)

FREE_TALK_SYSTEM_VI = (
    "Bạn là trợ lý giọng nói thân thiện trên xe. "
    "BẮT BUỘC trả lời CHỈ BẰNG TIẾNG VIỆT. "
    "Trả lời ngắn gọn, lịch sự, dễ đọc bằng TTS cho chào hỏi và trò chuyện chung. "
    "Nếu người dùng xin kể chuyện cười / joke / trò chuyện vui, hãy đáp lại ngắn, vui vẻ (1–3 câu). "
    "KHÔNG bịa quy trình vận hành xe, thông số kỹ thuật, hay hướng dẫn từ manual. "
    "KHÔNG bịa số liệu thời tiết hay tin tức — nếu được hỏi thời tiết, nói bạn không có "
    "dữ liệu thời tiết trực tiếp và gợi ý hỏi câu tra cứu tài liệu xe. "
    "Nếu người dùng hỏi điều khiển/bảo dưỡng/an toàn xe, nhắc họ hỏi kiểu tra cứu manual."
)

FREE_TALK_SYSTEM_EN = (
    "You are a friendly in-car voice assistant. "
    "You MUST answer ONLY IN ENGLISH. "
    "Keep replies short, polite, and TTS-friendly for greetings and small talk. "
    "If the user asks for a joke or light story, give a short playful reply (1–3 sentences). "
    "Do NOT invent vehicle operating procedures, specs, or manual steps. "
    "Do NOT invent weather numbers or news — if asked about weather, say you have no "
    "live weather data and suggest a vehicle-manual style question instead. "
    "If the user asks about controls/maintenance/safety, ask them to rephrase as a manual lookup."
)

NOT_FOUND_VI = (
    "Không tìm thấy thông tin phù hợp trong tài liệu kỹ thuật của xe. "
    "Bạn có thể hỏi cách khác hoặc chủ đề có trong manual."
)
NOT_FOUND_EN = (
    "No matching information was found in the vehicle technical documents. "
    "Try rephrasing or ask about a topic covered in the manual."
)

REFUSED_VI = "Yêu cầu bị từ chối vì lý do an toàn vận hành xe."
REFUSED_EN = "Request refused for vehicle operational safety reasons."

FREE_TALK_NO_LLM_VI = (
    "Xin chào! Hiện tôi chỉ hỗ trợ tốt nhất các câu hỏi tra cứu tài liệu kỹ thuật của xe. "
    "Bạn hãy hỏi theo kiểu hướng dẫn manual nhé."
)
FREE_TALK_NO_LLM_EN = (
    "Hello! I work best with vehicle technical-document questions. "
    "Please ask in a manual-lookup style."
)

# --- START MODIFICATION ---
# Used when free_talk mode is correct but the local LLM is unreachable.
FREE_TALK_LLM_DOWN_VI = (
    "Tôi nghe bạn rồi, nhưng trợ lý trò chuyện (Ollama) đang tắt hoặc chưa kết nối. "
    "Hãy bật Ollama rồi hỏi lại, hoặc hỏi câu tra cứu manual xe."
)
FREE_TALK_LLM_DOWN_EN = (
    "I heard you, but the chat LLM (Ollama) is offline or unreachable. "
    "Start Ollama and ask again, or ask a vehicle-manual question."
)
# --- END MODIFICATION ---

# --- START MODIFICATION ---
# Soft handoff after RAG miss: acknowledge gap, clarify — never invent procedures.
RAG_MISS_HANDOFF_VI = (
    "Bạn là trợ lý giọng nói trên xe. Người dùng vừa hỏi về tài liệu/xe "
    "nhưng chỉ mục manual không có đủ bằng chứng. "
    "BẮT BUỘC trả lời CHỈ BẰNG TIẾNG VIỆT, ngắn, TTS-friendly. "
    "Thừa nhận chưa tìm thấy trong tài liệu, gợi ý hỏi lại rõ model/năm/chủ đề. "
    "TUYỆT ĐỐI KHÔNG bịa quy trình, nút bấm, thông số, hay bước thao tác."
)
RAG_MISS_HANDOFF_EN = (
    "You are an in-car voice assistant. The driver asked a manual/vehicle question "
    "but the indexed manuals lack sufficient evidence. "
    "You MUST answer ONLY IN ENGLISH, short and TTS-friendly. "
    "Acknowledge that nothing matching was found in the documents and invite a "
    "clearer model/year/topic follow-up. "
    "NEVER invent procedures, button sequences, specs, or operating steps."
)

RAG_MISS_HANDOFF_FALLBACK_VI = (
    "Tôi chưa tìm thấy nội dung phù hợp trong tài liệu kỹ thuật đang có. "
    "Bạn thử nêu rõ model, năm, và thao tác cần hỏi nhé."
)
RAG_MISS_HANDOFF_FALLBACK_EN = (
    "I could not find matching information in the available technical documents. "
    "Please specify the model, year, and the procedure you need."
)
# --- END MODIFICATION ---

TIMEOUT_SOFT_VI = (
    "Xin lỗi, trợ lý đang khởi động chậm. Vui lòng hỏi lại sau vài giây."
)
TIMEOUT_SOFT_EN = (
    "Sorry, the assistant is still warming up. Please ask again in a few seconds."
)

CAR_STUB_VI = (
    "Yêu cầu điều khiển xe đã được ghi nhận, nhưng chức năng điều khiển phần cứng "
    "chưa khả dụng trong bản build này."
)
CAR_STUB_EN = (
    "Vehicle control request noted, but hardware control is not available in this build."
)


def normalize_language(language: str | None) -> Lang:
    raw = (language or "vi").strip().lower()
    if raw in {"en", "en-us", "english"}:
        return "en"
    return "vi"


def rag_system_prompt(language: str | None) -> str:
    return RAG_SYSTEM_EN if normalize_language(language) == "en" else RAG_SYSTEM_VI


def free_talk_system_prompt(language: str | None) -> str:
    return FREE_TALK_SYSTEM_EN if normalize_language(language) == "en" else FREE_TALK_SYSTEM_VI


def rag_miss_handoff_system_prompt(language: str | None) -> str:
    # --- START MODIFICATION ---
    return (
        RAG_MISS_HANDOFF_EN
        if normalize_language(language) == "en"
        else RAG_MISS_HANDOFF_VI
    )
    # --- END MODIFICATION ---


def rag_miss_handoff_fallback(language: str | None) -> str:
    # --- START MODIFICATION ---
    return (
        RAG_MISS_HANDOFF_FALLBACK_EN
        if normalize_language(language) == "en"
        else RAG_MISS_HANDOFF_FALLBACK_VI
    )
    # --- END MODIFICATION ---


def not_found_answer(language: str | None) -> str:
    return NOT_FOUND_EN if normalize_language(language) == "en" else NOT_FOUND_VI


def refused_answer(language: str | None) -> str:
    return REFUSED_EN if normalize_language(language) == "en" else REFUSED_VI


def free_talk_no_llm(language: str | None) -> str:
    return FREE_TALK_NO_LLM_EN if normalize_language(language) == "en" else FREE_TALK_NO_LLM_VI


def free_talk_llm_down(language: str | None) -> str:
    # --- START MODIFICATION ---
    return FREE_TALK_LLM_DOWN_EN if normalize_language(language) == "en" else FREE_TALK_LLM_DOWN_VI
    # --- END MODIFICATION ---


def timeout_soft_answer(language: str | None) -> str:
    return TIMEOUT_SOFT_EN if normalize_language(language) == "en" else TIMEOUT_SOFT_VI


def car_stub_answer(language: str | None) -> str:
    return CAR_STUB_EN if normalize_language(language) == "en" else CAR_STUB_VI
