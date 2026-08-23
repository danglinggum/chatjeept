from __future__ import annotations

import json
import os
import re
import time
import sqlite3
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal
from collections import defaultdict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from rag import retrieve_relevant_jee_context

load_dotenv()

# =============================================================================
# Security & Auth Settings
# =============================================================================

ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "jeept2026")

REQUEST_RECORDS = defaultdict(list)
RATE_LIMIT_WINDOW = 60
MAX_REQUESTS_PER_WINDOW = 30

def check_rate_limit(ip_address: str):
    now = time.time()
    REQUEST_RECORDS[ip_address] = [t for t in REQUEST_RECORDS[ip_address] if now - t < RATE_LIMIT_WINDOW]
    if len(REQUEST_RECORDS[ip_address]) >= MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a moment before asking another question."
        )
    REQUEST_RECORDS[ip_address].append(now)

def verify_admin_key(x_admin_key: str | None = Header(default=None)):
    if not x_admin_key or x_admin_key != ADMIN_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized: Invalid Admin Secret Key")

# =============================================================================
# AI Client & Stable Models
# =============================================================================

SCENE_START_MARKER = "<<<3D_SCENE>>>"
SCENE_END_MARKER = "<<<END_3D_SCENE>>>"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY must be set in Environment Variables")

ai_client = genai.Client(api_key=GOOGLE_API_KEY)

MODELS_TO_TRY = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

# =============================================================================
# SQLite Analytics & Query Logging
# =============================================================================

DB_FILE = Path(__file__).resolve().parent / "chat_logs.db"

def init_db():
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ip_address TEXT NOT NULL,
            subject TEXT NOT NULL,
            tutor TEXT NOT NULL,
            mode TEXT NOT NULL,
            question TEXT NOT NULL,
            has_scene INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

init_db()

def log_user_query(ip_address: str, subject: str, tutor: str, mode: str, question: str, has_scene: bool):
    try:
        conn = sqlite3.connect(str(DB_FILE))
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """
            INSERT INTO query_logs (timestamp, ip_address, subject, tutor, mode, question, has_scene)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (now_str, ip_address, subject, tutor, mode, question, 1 if has_scene else 0)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Logging Error] {e}")

# =============================================================================
# Tutor Config & Pydantic Models
# =============================================================================

Subject = Literal["Physics", "Chemistry", "Mathematics"]
TutorMode = Literal["Socratic Hint", "Full Solution"]

TUTOR_CONFIG: dict[str, dict[str, str]] = {
    "Physics": {
        "name": "Rahul",
        "institution": "IIT Bombay",
        "specialty": "electromagnetism, mechanics, 3D vectors, force diagrams, fields",
    },
    "Chemistry": {
        "name": "Raj",
        "institution": "IIT Delhi",
        "specialty": "VSEPR molecular geometry, chemical bonding, stereochemistry, organic mechanisms",
    },
    "Mathematics": {
        "name": "Amit",
        "institution": "IIT Kanpur",
        "specialty": "3D coordinate geometry, vectors, planes, lines, calculus, algebra",
    },
}

Vector3 = tuple[float, float, float]

class SphereElement(BaseModel):
    model_config = ConfigDict(extra="ignore")
    kind: Literal["sphere"]
    position: Vector3
    radius: float = Field(default=0.35, gt=0, le=5)
    color: str = Field(default="#60a5fa", pattern=r"^#[0-9A-Fa-f]{6}$")
    label: str | None = Field(default=None, max_length=100)

class CylinderElement(BaseModel):
    model_config = ConfigDict(extra="ignore")
    kind: Literal["cylinder"]
    start: Vector3
    end: Vector3
    color: str = Field(default="#94a3b8", pattern=r"^#[0-9A-Fa-f]{6}$")

class ArrowElement(BaseModel):
    model_config = ConfigDict(extra="ignore")
    kind: Literal["arrow"]
    start: Vector3
    end: Vector3
    color: str = Field(default="#f97316", pattern=r"^#[0-9A-Fa-f]{6}$")
    label: str | None = Field(default=None, max_length=100)

SceneElement = Annotated[SphereElement | CylinderElement | ArrowElement, Field(discriminator="kind")]
scene_element_adapter = TypeAdapter(SceneElement)

class Scene(BaseModel):
    model_config = ConfigDict(extra="ignore")
    elements: list[SceneElement] = Field(default_factory=list, max_length=30)

class TutorResponse(BaseModel):
    explanation: str
    scene: Scene | None = None
    tutor: str
    subject: Subject
    mode: TutorMode

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=30000)

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=30000)
    subject: Subject = "Physics"
    mode: TutorMode = "Socratic Hint"
    history: list[ChatMessage] = Field(default_factory=list, max_length=40)

# =============================================================================
# FastAPI Setup
# =============================================================================

app = FastAPI(title="ChatJEEPT API", version="3.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_INSTRUCTION = f"""
You are one of the elite ChatJEEPT IIT-JEE Master Tutors.

GENERAL TEACHING RULES:
1. Remain strictly faithful to the active tutor persona (Rahul for Physics, Raj for Chemistry, Amit for Mathematics).
2. Utilize the provided 'RETRIEVED JEE ARCHIVAL REFERENCES' to model your analytical rigor, typical IIT traps, and exam-standard formulations.
3. Write the explanation in clear English using standard LaTeX ($...$ for inline, $$...$$ for display).
4. Never wrap the explanation in quotation marks or JSON.

REQUIRED OUTPUT FORMAT:
First write the complete explanation.
Then output on its own line:
{SCENE_START_MARKER}
If a 3D visualization is helpful, output valid JSON:
{{
  "elements": [
    {{"kind": "sphere", "position": [0, 0, 0], "radius": 0.35, "color": "#60a5fa", "label": "Mass"}},
    {{"kind": "cylinder", "start": [0, 0, 0], "end": [1, 0, 0], "color": "#94a3b8"}},
    {{"kind": "arrow", "start": [0, 0, 0], "end": [0, 2, 0], "color": "#f97316", "label": "Force F"}}
  ]
}}
If no 3D visualization is helpful, output:
null
Then output on its own line:
{SCENE_END_MARKER}
""".strip()

def build_prompt(request: ChatRequest) -> str:
    tutor = TUTOR_CONFIG[request.subject]
    try:
        rag_context = retrieve_relevant_jee_context(request.message, request.subject, top_k=2)
    except Exception:
        rag_context = "No archival reference available."

    history_payload = [{"role": item.role, "content": item.content} for item in request.history[-6:]]
    history_text = json.dumps(history_payload, ensure_ascii=False, indent=2)

    return f"""
ACTIVE TUTOR: Master {tutor["name"]} ({tutor["institution"]})
SUBJECT: {request.subject}
MODE: {request.mode}

RETRIEVED JEE ARCHIVAL REFERENCES:
{rag_context}

CONVERSATION HISTORY:
{history_text}

STUDENT QUESTION:
{request.message}

Answer as Master {tutor["name"]} in English adhering to the exact delimiter rules.
""".strip()

def parse_json_defensively(value: str) -> Any | None:
    if not value or value.strip().lower() == "null":
        return None
    cleaned = re.sub(r"^```(?:json)?\s*", "", value.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    s_idx = cleaned.find("{")
    e_idx = cleaned.rfind("}")
    if s_idx != -1 and e_idx > s_idx:
        cleaned = cleaned[s_idx:e_idx+1]
    try:
        return json.loads(cleaned)
    except Exception:
        return None

def extract_explanation_and_scene(raw_response: str) -> tuple[str, Scene | None]:
    text = raw_response.strip()
    if SCENE_START_MARKER not in text:
        return text, None

    parts = text.split(SCENE_START_MARKER)
    explanation = parts[0].strip()
    scene_part = parts[1].split(SCENE_END_MARKER)[0].strip() if SCENE_END_MARKER in parts[1] else parts[1].strip()

    parsed = parse_json_defensively(scene_part)
    scene = None
    if isinstance(parsed, dict) and "elements" in parsed:
        try:
            valid_elements = [scene_element_adapter.validate_python(el) for el in parsed["elements"]]
            scene = Scene(elements=valid_elements)
        except Exception:
            scene = None

    return explanation, scene

@app.get("/")
async def root():
    return {"status": "online", "service": "ChatJEEPT API", "timestamp": datetime.now().isoformat()}

def generate_ai_content_sync(model_name: str, prompt: str):
    return ai_client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
            max_output_tokens=3000,
        ),
    )

@app.post("/api/chat", response_model=TutorResponse)
async def chat(request: ChatRequest, req: Request) -> TutorResponse:
    client_ip = req.headers.get("x-forwarded-for", req.client.host if req.client else "127.0.0.1").split(",")[0].strip()
    check_rate_limit(client_ip)

    tutor = TUTOR_CONFIG[request.subject]
    prompt = build_prompt(request)
    last_error = None

    for model_name in MODELS_TO_TRY:
        try:
            response = await asyncio.to_thread(generate_ai_content_sync, model_name, prompt)
            raw_text = response.text or ""
            if raw_text.strip():
                explanation, scene = extract_explanation_and_scene(raw_text)
                
                log_user_query(
                    ip_address=client_ip,
                    subject=request.subject,
                    tutor=tutor["name"],
                    mode=request.mode,
                    question=request.message,
                    has_scene=scene is not None
                )

                return TutorResponse(
                    explanation=explanation,
                    scene=scene,
                    tutor=tutor["name"],
                    subject=request.subject,
                    mode=request.mode,
                )
        except Exception as exc:
            last_error = exc
            print(f"[Model Fallback] {model_name} error: {exc}")
            continue

    raise HTTPException(status_code=500, detail=f"AI generation failed: {str(last_error)}")

# =============================================================================
# Admin Endpoints (실시간 통계 & 로그 조회)
# =============================================================================

@app.get("/api/admin/stats", dependencies=[Depends(verify_admin_key)])
async def get_admin_stats():
    init_db()
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM query_logs")
    total_queries = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT ip_address) FROM query_logs")
    unique_visitors = cursor.fetchone()[0]

    today_str = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("SELECT COUNT(*) FROM query_logs WHERE timestamp LIKE ?", (f"{today_str}%",))
    today_queries = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM query_logs WHERE has_scene = 1")
    scenes_generated = cursor.fetchone()[0]

    conn.close()

    return {
        "total_queries": total_queries,
        "unique_visitors": unique_visitors,
        "today_queries": today_queries,
        "scenes_generated": scenes_generated,
    }

@app.get("/api/admin/queries", dependencies=[Depends(verify_admin_key)])
async def get_admin_queries(limit: int = 100):
    init_db()
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, timestamp, ip_address, subject, tutor, mode, question, has_scene
        FROM query_logs
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()

    logs = []
    for r in rows:
        logs.append({
            "id": r[0],
            "timestamp": r[1],
            "ip_address": r[2],
            "subject": r[3],
            "tutor": r[4],
            "mode": r[5],
            "question": r[6],
            "has_scene": bool(r[7]),
        })

    return {"logs": logs}

@app.post("/api/admin/seed-test-log", dependencies=[Depends(verify_admin_key)])
async def seed_test_log():
    log_user_query("127.0.0.1", "Physics", "Rahul", "Socratic Hint", "What is pure rolling condition on rough incline?", True)
    log_user_query("127.0.0.1", "Chemistry", "Raj", "Full Solution", "Explain PCl5 trigonal bipyramidal bond angles", True)
    return {"status": "success", "message": "Test logs created successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)