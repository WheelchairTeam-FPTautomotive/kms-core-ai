import os
import re
import sys
import requests
import chromadb
from pathlib import Path

# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def detect_spacing_issues(text: str) -> list[str]:
    """Detects if there are sequences of 3 or more single-character words separated by a single space."""
    spaced_run_pattern = re.compile(r'\b[a-zA-Z0-9]\s[a-zA-Z0-9]\s[a-zA-Z0-9]\b')
    return spaced_run_pattern.findall(text)

def main():
    print("=== Connecting to ChromaDB to retrieve unique document samples ===")
    
    # Initialize ChromaDB client to query stored document metadata
    chroma_path = "data/chroma_db"
    if not os.path.exists(chroma_path):
        print(f"[ERROR] ChromaDB path does not exist at {chroma_path}")
        return
        
    client = chromadb.PersistentClient(path=chroma_path)
    try:
        collection = client.get_collection("automotive_manuals")
    except Exception as e:
        print(f"[ERROR] Could not load collection 'automotive_manuals': {e}")
        return

    # Get chunks from the database
    print("Fetching document metadata from ChromaDB...")
    db_data = collection.get(include=["metadatas", "documents"])
    
    metadatas = db_data.get("metadatas", [])
    documents = db_data.get("documents", [])
    
    if not metadatas:
        print("[ERROR] No documents found in database.")
        return

    # Group one sample document text per unique PDF file name
    unique_pdfs = {}
    for meta, doc in zip(metadatas, documents):
        doc_name = meta.get("document_name")
        if doc_name and doc_name not in unique_pdfs:
            # Clean up the doc text to use as a query seed
            # Take the first line or first 10 words that look like a technical query seed
            words = doc.split()
            if len(words) > 5:
                query_seed = " ".join(words[:6])
                # Clean up query seed punctuation
                query_seed = re.sub(r'[^\w\s]', '', query_seed).strip()
                if len(query_seed) > 10:
                    unique_pdfs[doc_name] = query_seed

    print(f"Found {len(unique_pdfs)} unique PDF manuals in ChromaDB.")
    print("Starting API query testing against http://localhost:8001/api/v1/search...")
    
    api_url = "http://localhost:8001/api/v1/search"
    results = []
    
    for idx, (doc_name, query_seed) in enumerate(unique_pdfs.items(), start=1):
        print(f"[{idx}/{len(unique_pdfs)}] Testing {doc_name} using query: '{query_seed}'")
        
        try:
            response = requests.post(api_url, json={"query": query_seed}, timeout=10)
            if response.status_code != 200:
                results.append({
                    "document": doc_name,
                    "query": query_seed,
                    "status": "HTTP_ERROR",
                    "reason": f"Status code {response.status_code}"
                })
                continue
                
            res_json = response.json()
            answer = res_json.get("answer", "")
            
            # Detect spacing issues
            issues = detect_spacing_issues(answer)
            
            # Also check citations for spacing issues
            citation_issues = []
            for cit in res_json.get("citations", []):
                cit_text = cit.get("matched_text", "")
                cit_issues = detect_spacing_issues(cit_text)
                if cit_issues:
                    citation_issues.extend(cit_issues)
            
            if issues or citation_issues:
                results.append({
                    "document": doc_name,
                    "query": query_seed,
                    "status": "FAIL",
                    "issues": list(set(issues + citation_issues)),
                    "preview": answer[:150].replace('\n', ' ')
                })
            else:
                results.append({
                    "document": doc_name,
                    "query": query_seed,
                    "status": "PASS",
                    "preview": answer[:150].replace('\n', ' ')
                })
                
        except Exception as e:
            results.append({
                "document": doc_name,
                "query": query_seed,
                "status": "ERROR",
                "reason": str(e)
            })

    # Summary report
    print("\n==================================================")
    print("                API TESTING REPORT                ")
    print("==================================================")
    
    passes = sum(1 for r in results if r["status"] == "PASS")
    fails = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] in ("ERROR", "HTTP_ERROR"))
    
    print(f"Total PDFs Checked:  {len(results)}")
    print(f"PASS (No spacing issues): {passes} / {len(results)}")
    print(f"FAIL (Spacing issues found): {fails} / {len(results)}")
    print(f"ERRORS (Connection/API errors): {errors} / {len(results)}")
    
    if fails > 0:
        print("\n--- Failed PDFs Details ---")
        for r in results:
            if r["status"] == "FAIL":
                print(f"❌ {r['document']}:")
                print(f"   Query:  '{r['query']}'")
                print(f"   Issues: {r['issues']}")
                print(f"   Answer Preview: {r['preview']}...")
    else:
        print("\n🎉 All unique PDF manuals returned 100% clean and readable API answers!")

if __name__ == "__main__":
    main()
