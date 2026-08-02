import sys
import os
import json

# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to python path dynamically
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pipelines.bedrock_rag import solve_automotive_query_bedrock

try:
    print("Testing solve_automotive_query_bedrock dynamically...")
    res = solve_automotive_query_bedrock("Hệ thống điều hòa HVAC hoạt động như thế nào?")
    print("Bedrock RAG Result:")
    print(json.dumps(res, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"Bedrock RAG Failed: {e}")
