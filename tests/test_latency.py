import time

from pipelines.solve_problem import get_chroma_collection, solve_automotive_query_live


def test_solve_automotive_query_latency():
    """Verify that vector search queries execute well under the 200ms latency SLA."""
    # Warm up ONNX session before measuring latency
    get_chroma_collection()

    test_queries = [
        "Hệ thống điều hòa HVAC hoạt động như thế nào?",
        "Làm thế nào kích hoạt phanh khẩn cấp ADAS?",
        "Hướng dẫn kiểm tra cảm biến tốc độ xe",
        "Quy trình bảo trì ắc quy ô tô",
        "Hệ thống cảnh báo dây an toàn hoạt động ra sao?",
    ]

    latencies_ms: list[float] = []

    for q in test_queries:
        t0 = time.perf_counter()
        response = solve_automotive_query_live(q)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        latencies_ms.append(elapsed_ms)

        assert response["status"] in ["success", "refused"]
        if response["status"] == "success":
            assert "citations" in response
            assert len(response["citations"]) > 0

    avg_latency = sum(latencies_ms) / len(latencies_ms)
    max_latency = max(latencies_ms)

    print("\n[LATENCY BENCHMARK RESULT]")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Max Latency: {max_latency:.2f} ms")

    # Assert response SLA requirement (< 200ms average & max under 250ms allowance for OS test harness jitter)
    assert avg_latency < 200.0, (
        f"Average latency ({avg_latency:.2f}ms) exceeded 200ms SLA!"
    )
    assert max_latency < 250.0, (
        f"Max latency ({max_latency:.2f}ms) exceeded 250ms SLA limit!"
    )
