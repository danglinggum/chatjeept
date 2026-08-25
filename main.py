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
# Google GenAI Client & Resilient Model Pool
# =============================================================================

SCENE_START_MARKER = "<<<3D_SCENE>>>"
SCENE_END_MARKER = "<<<END_3D_SCENE>>>"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
ai_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None

MODELS_TO_TRY = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

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
# FastAPI Core App & Open CORS
# =============================================================================

app = FastAPI(title="ChatJEEPT API", version="8.9.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_INSTRUCTION = """
You are an elite IIT-JEE Master Tutor (Rahul: Physics, Raj: Chemistry, Amit: Mathematics).
Communicate STRICTLY in English with flawless mathematical precision.

CRITICAL MATHEMATICAL FORMATTING RULES:
1. Write all ordinary explanatory sentences in standard English markdown. NEVER put English words or sentences inside `$ ... $` or `$$ ... $$`.
2. Always put spaces around inline math variables:
   - CORRECT: "A particle of mass $m$ and positive charge $q$ is released at $t = 0$."
   - INCORRECT: "A particle of mass$m$and positive charge $$qis released...$$"
3. Put every standalone derivation or multi-line formula in BLOCK math using `$$` on its OWN separate line:
   $$m\\ddot{x} = q B_0 \\dot{y}$$
   $$m\\ddot{y} = q E_0 - q B_0 \\dot{x}$$
   $$m\\ddot{z} = 0$$
4. For systems of equations, always format cleanly using cases blocks:
   $$\\begin{cases} x(t) = \\dfrac{m E_0}{q B_0^2}(\\omega t - \\sin\\omega t) \\\\[8pt] y(t) = \\dfrac{m E_0}{q B_0^2}(1 - \\cos\\omega t) \\\\[8pt] z(t) = 0 \\end{cases}$$
5. NEVER concatenate multiple dollar signs back-to-back without spaces.

CRITICAL 3D VISUALIZATION DIRECTIVES:
1. NEVER describe the 3D scene in paragraph text.
2. ALWAYS append the actual 3D scene JSON at the end using `<<<3D_SCENE>>>` and `<<<END_3D_SCENE>>>`.
3. ONLY use allowed kinds: "sphere", "cylinder", "arrow".

Output format:
[Your complete, exhaustive explanation in English here]

<<<3D_SCENE>>>
{
  "elements": [ ... ]
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
    if isinstance(parsed, dict) and "elements" in parsed and isinstance(parsed["elements"], list):
        valid_elements = []
        for el in parsed["elements"]:
            try:
                valid_elements.append(scene_element_adapter.validate_python(el))
            except Exception:
                continue
        if len(valid_elements) > 0:
            scene = Scene(elements=valid_elements)
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
    return {"status": "online", "service": "ChatJEEPT API v8.9"}

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
        for _ in range(2):
            try:
                res = await asyncio.to_thread(generate_ai, model_name, prompt)
                raw = res.text or ""
                if raw.strip():
                    exp, sc = extract_explanation_and_scene(raw)
                    return TutorResponse(
                        explanation=exp,
                        scene=sc,
                        tutor=tutor["name"],
                        subject=request.subject,
                        mode=request.mode
                    )
            except Exception as e:
                last_err = e
                await asyncio.sleep(1.0)
                continue

    raise HTTPException(status_code=500, detail=f"AI generation failed: {str(last_err)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)