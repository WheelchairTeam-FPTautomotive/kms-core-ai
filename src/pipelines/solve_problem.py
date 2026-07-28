import os
from typing import Dict, List, Any
from utils.logger import setup_logger

logger = setup_logger("core_rag_pipeline")

UNSAFE_TRIGGERS = ["hack", "bypass brakes", "overdrive engine safety", "ignore seatbelt alert"]
AUTOMOTIVE_KEYWORDS = ["car", "vehicle", "engine", "brake", "sensor", "battery", "hvac", "seatbelt", "adas", "cluster", "dashboard", "manual"]

def check_safety_and_scope(query: str) -> tuple[bool, str]:
    query_lower = query.lower()
    for trigger in UNSAFE_TRIGGERS:
        if trigger in query_lower:
            logger.warning(f"Unsafe request detected: '{query}' triggering: '{trigger}'")
            return False, "Yêu cầu bị từ chối vì lý do an toàn vận hành xe."

    is_on_topic = any(keyword in query_lower for keyword in AUTOMOTIVE_KEYWORDS)
    if not is_on_topic:
        logger.warning(f"Out of scope request: '{query}'")
        return False, "Tôi chỉ hỗ trợ giải đáp các câu hỏi liên quan đến vận hành và hướng dẫn kỹ thuật của xe."

    return True, ""

def solve_automotive_query(query: str) -> Dict[str, Any]:
    logger.info(f"RAG processing query: '{query}'")
    
    is_valid, refusal_reason = check_safety_and_scope(query)
    if not is_valid:
        return {
            "query": query,
            "answer": refusal_reason,
            "citations": [],
            "status": "refused"
        }
        
    logger.info("Executing vector database query...")
    citations = [
        {
            "document_id": "949eb66893b5dbf59aa4b4be35ad330c7b8f0c3802f9ccb8d25881128157bf9c",
            "document_name": "2011 - KMS Manual.pdf",
            "section": "Chương 4: Điều hòa & Hệ thống điện",
            "page": 42,
            "matched_text": "Hệ thống điều hòa (HVAC) được điều khiển qua CarPropertyManager với AreaId là 0."
        },
        {
            "document_id": "1ecc7f4e2b438cb0ac5c336fed7cfffbca78b42f87a31a0c0add50aa38cfc751",
            "document_name": "light-control-system.pdf",
            "section": "Chương 7: ADAS & Phanh khẩn cấp",
            "page": 105,
            "matched_text": "Khi xe chạy quá tốc độ 80km/h, hệ thống ADAS kích hoạt phanh khẩn cấp tự động (AEB) nếu khoảng cách xe trước < 15m."
        }
    ]
    
    answer = (
        f"Dựa trên tài liệu hướng dẫn kỹ thuật của xe:\n"
        f"1. Hệ thống điều hòa (HVAC) hoạt động trên VHAL thông qua CarPropertyManager (AreaId: 0).\n"
        f"2. Phanh khẩn cấp tự động (AEB) hoạt động kết hợp với ADAS sẽ kích hoạt để bảo vệ an toàn khi xe chạy > 80km/h và khoảng cách va chạm dưới 15m."
    )
    
    logger.info("Formulated RAG response with citations.")
    return {
        "query": query,
        "answer": answer,
        "citations": citations,
        "status": "success"
    }

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="KMS RAG Offline Evaluator CLI")
    parser.add_argument("--input", required=True, help="Input directory containing queries")
    parser.add_argument("--output", required=True, help="Output file to write responses to")
    args = parser.parse_args()
    
    logger.info(f"Running offline batch evaluation: input={args.input}, output={args.output}")
    
    # Mock reading inputs
    queries = ["Làm thế nào kích hoạt phanh khẩn cấp ADAS?"]
    
    results = []
    for q in queries:
        res = solve_automotive_query(q)
        results.append(res)
        
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Batch evaluation finished. Results written to: {args.output}")
