import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any

DATA_DIR = Path(__file__).resolve().parent / "data"

FILE_MAP = {
    "Physics": "physics.json",
    "Chemistry": "chemistry.json",
    "Mathematics": "math.json"
}

ARCHIVE_CACHE: Dict[str, List[Dict[str, Any]]] = {
    "Physics": [],
    "Chemistry": [],
    "Mathematics": []
}

def load_data():
    for subject, filename in FILE_MAP.items():
        file_path = DATA_DIR / filename
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    ARCHIVE_CACHE[subject] = json.load(f)
            except Exception as e:
                print(f"[RAG Load Warning] {subject}: {e}")

load_data()

def retrieve_relevant_jee_context(query: str, subject: str, top_k: int = 2) -> str:
    items = ARCHIVE_CACHE.get(subject, [])
    if not items:
        for s_items in ARCHIVE_CACHE.values():
            if s_items:
                items = s_items
                break

    if not items:
        return "No archival JEE reference available."

    query_tokens = set(re.findall(r'[a-zA-Z0-9_]+', query.lower()))
    
    scored_items = []
    for item in items:
        text = f"{item.get('topic', '')} {item.get('question', '')} {item.get('solution', '')}".lower()
        score = sum(1 for token in query_tokens if token in text)
        scored_items.append((score, item))

    scored_items.sort(key=lambda x: x[0], reverse=True)
    top_matches = [item for _, item in scored_items[:top_k]]

    formatted = []
    for i, item in enumerate(top_matches, 1):
        formatted.append(
            f"--- [Retrieved Archival JEE Reference #{i}] ---\n"
            f"Exam: {item.get('exam', '')}\n"
            f"Topic: {item.get('topic', '')}\n"
            f"Question: {item.get('question', '')}\n"
            f"Solution: {item.get('solution', '')}"
        )

    return "\n\n".join(formatted)