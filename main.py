from __future__ import annotations

import json
import os
import re
import time
import asyncio
from typing import Annotated, Any, Literal
from collections import defaultdict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

load_dotenv()

# =============================================================================
# Security: Rate Limiting
# =============================================================================

REQUEST_RECORDS = defaultdict(list)
RATE_LIMIT_WINDOW = 60
MAX_REQUESTS_PER_WINDOW = 30

def check_rate_limit(ip_address: str):
    now = time.time()
    REQUEST_RECORDS[ip_address] = [t for t in REQUEST_RECORDS[ip_address] if now - t < RATE_LIMIT_WINDOW]
    if len(REQUEST_RECORDS[ip_address]) >= MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment.")
    REQUEST_RECORDS[ip_address].append(now)

# =============================================================================
# Google GenAI Client
# =============================================================================

SCENE_START_MARKER = "<<<3D_SCENE>>>"
SCENE_END_MARKER = "<<<END_3D_SCENE>>>"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
ai_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None

MODELS_TO_TRY = ["gemini-3.6-flash"]

# =============================================================================
# Models & Configuration
# =============================================================================

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

# =============================================================================
# FastAPI Core App
# =============================================================================

app = FastAPI(title="ChatJEEPT API", version="8.2.0")

ALLOWED_ORIGINS = [
    "https://chatjeept-iota.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

SYSTEM_INSTRUCTION = """
You are an elite IIT-JEE Master Tutor (Rahul: Physics, Raj: Chemistry, Amit: Mathematics).
Provide rigorous, step-by-step mathematical and chemical derivations.

CRITICAL FORMATTING RULES:
1. Wrap ALL formulas, variables, and units cleanly in standard LaTeX: `$ ... $` for inline, `$$ ... $$` for block formulas.
2. NEVER nest dollar signs inside other dollar signs (e.g. do NOT write `$\\text{F}_{$$\\text{eq}}$`).

CRITICAL 3D VISUALIZATION DIRECTIVE:
Whenever geometry, molecules, vectors, skew lines, or orbitals are discussed, YOU MUST ALWAYS GENERATE A 3D SCENE AT THE VERY END.

MULTI-OBJECT / COMPARISON RULE:
If the user asks to compare or render TWO or MORE molecules/structures (e.g., SF4 and ClF3), render BOTH in the SAME scene by offsetting their X coordinates:
- Place Molecule 1 centered at x = -2.5
- Place Molecule 2 centered at x = +2.5

Output format:
[Your complete, exhaustive explanation and tables here]

<<<3D_SCENE>>>
{
  "elements": [
    {"kind": "sphere", "position": [-2.5, 0, 0], "radius": 0.4, "color": "#f59e0b", "label": "S (SF4)"},
    {"kind": "sphere", "position": [-2.5, 1.4, 0], "radius": 0.25, "color": "#10b981", "label": "F_ax"},
    {"kind": "cylinder", "start": [-2.5, 0, 0], "end": [-2.5, 1.4, 0], "color": "#94a3b8"},
    {"kind": "sphere", "position": [2.5, 0, 0], "radius": 0.4, "color": "#06b6d4", "label": "Cl (ClF3)"},
    {"kind": "sphere", "position": [2.5, 1.4, 0], "radius": 0.25, "color": "#10b981", "label": "F_ax"},
    {"kind": "cylinder", "start": [2.5, 0, 0], "end": [2.5, 1.4, 0], "color": "#94a3b8"}
  ]
}
<<<END_3D_SCENE>>>
""".strip()

def parse_json_defensively(value: str) -> Any | None:
    if not value or value.strip().lower() == "null":
        return None
    cleaned = value.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    s_idx = cleaned.find("{")
    e_idx = cleaned.rfind("}")
    if s_idx != -1 and e_idx > s_idx:
        cleaned = cleaned[s_idx:e_idx+1]
    # trailing comma 정리
    cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)
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
            if len(valid_elements) > 0:
                scene = Scene(elements=valid_elements)
        except Exception:
            scene = None
    return explanation, scene

def generate_ai(model_name: str, prompt: str):
    return ai_client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
            max_output_tokens=8192,
        ),
    )

@app.get("/")
@app.get("/health")
async def health():
    return {"status": "online", "service": "ChatJEEPT API v8.2"}

@app.post("/api/chat", response_model=TutorResponse)
@app.post("/api/chat/", response_model=TutorResponse)
@app.post("/chat", response_model=TutorResponse)
@app.post("/chat/", response_model=TutorResponse)
async def chat_endpoint(request: ChatRequest, req: Request):
    if not ai_client:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is not configured.")
    
    client_ip = req.headers.get("x-forwarded-for", req.client.host if req.client else "127.0.0.1").split(",")[0].strip()
    check_rate_limit(client_ip)

    tutor = TUTOR_CONFIG[request.subject]
    history_ctx = [{"role": h.role, "content": h.content} for h in request.history[-2:]]
    prompt = f"Subject: {request.subject}\nMode: {request.mode}\nHistory: {json.dumps(history_ctx)}\nQuestion: {request.message}"

    last_err = None
    for model_name in MODELS_TO_TRY:
        try:
            res = await asyncio.to_thread(generate_ai, model_name, prompt)
            raw = res.text or ""
            if raw.strip():
                exp, sc = extract_explanation_and_scene(raw)
                return TutorResponse(explanation=exp, scene=sc, tutor=tutor["name"], subject=request.subject, mode=request.mode)
        except Exception as e:
            last_err = e
            continue

    raise HTTPException(status_code=500, detail=f"AI generation failed: {str(last_err)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)