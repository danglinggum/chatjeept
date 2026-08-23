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
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

load_dotenv()

ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "jeept2026")
REQUEST_RECORDS = defaultdict(list)
RATE_LIMIT_WINDOW = 60
MAX_REQUESTS_PER_WINDOW = 30

def check_rate_limit(ip_address: str):
    now = time.time()
    REQUEST_RECORDS[ip_address] = [t for t in REQUEST_RECORDS[ip_address] if now - t < RATE_LIMIT_WINDOW]
    if len(REQUEST_RECORDS[ip_address]) >= MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(status_code=429, detail="요청이 너무 많습니다. 잠시 후 다시 시도해 주세요.")
    REQUEST_RECORDS[ip_address].append(now)

SCENE_START_MARKER = "<<<3D_SCENE>>>"
SCENE_END_MARKER = "<<<END_3D_SCENE>>>"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
ai_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None

MODELS_TO_TRY = ["gemini-2.0-flash", "gemini-1.5-flash"]
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
            "INSERT INTO query_logs (timestamp, ip_address, subject, tutor, mode, question, has_scene) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (now_str, ip_address, subject, tutor, mode, question, 1 if has_scene else 0)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Logging Error] {e}")

Subject = Literal["Physics", "Chemistry", "Mathematics"]
TutorMode = Literal["Socratic Hint", "Full Solution"]

TUTOR_CONFIG: dict[str, dict[str, str]] = {
    "Physics": {"name": "Rahul", "institution": "IIT Bombay"},
    "Chemistry": {"name": "Raj", "institution": "IIT Delhi"},
    "Mathematics": {"name": "Amit", "institution": "IIT Kanpur"},
}

Vector3 = tuple[float, float, float]

class SphereElement(BaseModel):
    model_config = ConfigDict(extra="ignore")
    kind: Literal["sphere"]
    position: Vector3
    radius: float = Field(default=0.35, gt=0, le=5)
    color: str = Field(default="#60a5fa")
    label: str | None = None

class CylinderElement(BaseModel):
    model_config = ConfigDict(extra="ignore")
    kind: Literal["cylinder"]
    start: Vector3
    end: Vector3
    color: str = Field(default="#94a3b8")

class ArrowElement(BaseModel):
    model_config = ConfigDict(extra="ignore")
    kind: Literal["arrow"]
    start: Vector3
    end: Vector3
    color: str = Field(default="#f97316")
    label: str | None = None

SceneElement = Annotated[SphereElement | CylinderElement | ArrowElement, Field(discriminator="kind")]
scene_element_adapter = TypeAdapter(SceneElement)

class Scene(BaseModel):
    model_config = ConfigDict(extra="ignore")
    elements: list[SceneElement] = Field(default_factory=list)

class TutorResponse(BaseModel):
    explanation: str
    scene: Scene | None = None
    tutor: str
    subject: Subject
    mode: TutorMode

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str
    subject: Subject = "Physics"
    mode: TutorMode = "Socratic Hint"
    history: list[ChatMessage] = Field(default_factory=list)

app = FastAPI(title="ChatJEEPT API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_INSTRUCTION = """
You are an elite IIT-JEE Master Tutor (Rahul: Physics, Raj: Chemistry, Amit: Mathematics).
Explain concepts step-by-step using LaTeX ($...$ and $$...$$).
If 3D spatial setup is helpful, end with:
<<<3D_SCENE>>>
{"elements": [{"kind": "sphere", "position": [0,0,0], "radius": 0.35, "color": "#60a5fa"}]}
<<<END_3D_SCENE>>>
If not needed, output null between delimiters.
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

def extract_explanation_and_scene(raw_text: str) -> tuple[str, Scene | None]:
    if SCENE_START_MARKER not in raw_text:
        return raw_text.strip(), None
    parts = raw_text.split(SCENE_START_MARKER)
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
    return {"status": "online", "service": "ChatJEEPT API v4.0"}

def generate_ai(model_name: str, prompt: str):
    return ai_client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION, temperature=0.2),
    )

async def process_chat(request: ChatRequest, req: Request) -> TutorResponse:
    if not ai_client:
        raise HTTPException(status_code=500, detail="Google API Key is not configured.")
    client_ip = req.headers.get("x-forwarded-for", req.client.host if req.client else "127.0.0.1").split(",")[0].strip()
    check_rate_limit(client_ip)

    tutor = TUTOR_CONFIG[request.subject]
    prompt = f"Subject: {request.subject}\nMode: {request.mode}\nQuestion: {request.message}"
    last_err = None

    for m in MODELS_TO_TRY:
        try:
            res = await asyncio.to_thread(generate_ai, m, prompt)
            raw = res.text or ""
            if raw.strip():
                exp, sc = extract_explanation_and_scene(raw)
                log_user_query(client_ip, request.subject, tutor["name"], request.mode, request.message, sc is not None)
                return TutorResponse(explanation=exp, scene=sc, tutor=tutor["name"], subject=request.subject, mode=request.mode)
        except Exception as e:
            last_err = e
            continue

    raise HTTPException(status_code=500, detail=f"AI 오류: {str(last_err)}")

@app.post("/api/chat")
@app.post("/api/chat/")
@app.post("/chat")
@app.post("/chat/")
async def chat_all(request: ChatRequest, req: Request):
    return await process_chat(request, req)

@app.get("/api/admin/stats")
@app.get("/admin/stats")
async def stats(x_admin_key: str | None = Header(default=None)):
    if x_admin_key != ADMIN_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")
    conn = sqlite3.connect(str(DB_FILE))
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM query_logs")
    t = c.fetchone()[0]
    c.execute("SELECT COUNT(DISTINCT ip_address) FROM query_logs")
    u = c.fetchone()[0]
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute("SELECT COUNT(*) FROM query_logs WHERE timestamp LIKE ?", (f"{today}%",))
    td = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM query_logs WHERE has_scene = 1")
    sc = c.fetchone()[0]
    conn.close()
    return {"total_queries": t, "unique_visitors": u, "today_queries": td, "scenes_generated": sc}

@app.get("/api/admin/queries")
@app.get("/admin/queries")
async def queries(limit: int = 100, x_admin_key: str | None = Header(default=None)):
    if x_admin_key != ADMIN_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")
    conn = sqlite3.connect(str(DB_FILE))
    c = conn.cursor()
    c.execute("SELECT id, timestamp, ip_address, subject, tutor, mode, question, has_scene FROM query_logs ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return {"logs": [{"id": r[0], "timestamp": r[1], "ip_address": r[2], "subject": r[3], "tutor": r[4], "mode": r[5], "question": r[6], "has_scene": bool(r[7])} for r in rows]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)