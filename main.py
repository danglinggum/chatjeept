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
# 1. Performance: In-Memory Response Caching
# =============================================================================

RESPONSE_CACHE: dict[str, dict[str, Any]] = {}
MAX_CACHE_SIZE = 300

def get_from_cache(key: str) -> dict[str, Any] | None:
    return RESPONSE_CACHE.get(key)

def set_to_cache(key: str, data: dict[str, Any]):
    if len(RESPONSE_CACHE) >= MAX_CACHE_SIZE:
        for k in list(RESPONSE_CACHE.keys())[:50]:
            RESPONSE_CACHE.pop(k, None)
    RESPONSE_CACHE[key] = data

# =============================================================================
# 2. Security: Rate Limiter
# =============================================================================

REQUEST_RECORDS = defaultdict(list)
RATE_LIMIT_WINDOW = 60
MAX_REQUESTS_PER_WINDOW = 40

def check_rate_limit(ip_address: str):
    now = time.time()
    REQUEST_RECORDS[ip_address] = [t for t in REQUEST_RECORDS[ip_address] if now - t < RATE_LIMIT_WINDOW]
    if len(REQUEST_RECORDS[ip_address]) >= MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a few seconds.")
    REQUEST_RECORDS[ip_address].append(now)

# =============================================================================
# 3. Google GenAI Client
# =============================================================================

SCENE_START_MARKER = "<<<3D_SCENE>>>"
SCENE_END_MARKER = "<<<END_3D_SCENE>>>"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
ai_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None

MODELS_TO_TRY = [
    "gemini-3.6-flash",
]

# =============================================================================
# 4. Data Models & Config
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
# 5. FastAPI App & Open CORS Policy
# =============================================================================

app = FastAPI(title="ChatJEEPT Turbo API", version="9.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_INSTRUCTION = """
You are an elite IIT-JEE Master Tutor (Rahul: Physics, Raj: Chemistry, Amit: Mathematics).
Deliver direct, mathematically rigorous solutions in English.

CRITICAL LENGTH & STRUCTURE RULES:
1. Provide clear, structured derivations without excessive conversational fluff so you NEVER run out of tokens.
2. ALWAYS ensure the response reaches completion and finishes with the full 3D SCENE JSON.

CRITICAL LATEX RULES:
1. Explain in markdown. NEVER put plain English words inside LaTeX tags.
2. Put standalone formulas in double dollar blocks:
   $$[\\text{Co}(\\text{en})_2\\text{Cl}_2]^+$$
3. For multi-line math, use aligned or cases blocks and ALWAYS close them properly:
   $$\\begin{aligned} \\hat{i}(\\vec{r}_1) &= \\vec{r}_2 \\\\ \\hat{i}(\\vec{r}_3) &= \\vec{r}_4 \\end{aligned}$$

CRITICAL 3D VISUALIZATION:
Always append the 3D scene JSON at the very end using the markers.
When comparing two molecules (like trans vs cis), place structure 1 at x = -2.5 and structure 2 at x = +2.5.
Only use kinds: "sphere", "cylinder", "arrow".

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
            temperature=0.1,
            max_output_tokens=8192,  # 잘림 방지를 위해 최대 토큰 상한 8192로 확장
        ),
    )

@app.get("/")
@app.get("/health")
async def health():
    return {"status": "online", "service": "ChatJEEPT Turbo API v9.3"}

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

    cache_key = f"{request.subject}:{request.mode}:{request.message.strip().lower()}"
    cached_payload = get_from_cache(cache_key)
    if cached_payload:
        return TutorResponse(**cached_payload)

    history_ctx = [{"role": h.role, "content": h.content} for h in request.history[-2:]]
    prompt = f"Subject: {request.subject}\nMode: {request.mode}\nContext: {json.dumps(history_ctx)}\nQuestion: {request.message}"

    last_err = None
    for model_name in MODELS_TO_TRY:
        for _ in range(2):
            try:
                res = await asyncio.to_thread(generate_ai, model_name, prompt)
                raw = res.text or ""
                if raw.strip():
                    exp, sc = extract_explanation_and_scene(raw)
                    result = {
                        "explanation": exp,
                        "scene": sc.model_dump() if sc else None,
                        "tutor": tutor["name"],
                        "subject": request.subject,
                        "mode": request.mode
                    }
                    set_to_cache(cache_key, result)
                    return TutorResponse(
                        explanation=exp,
                        scene=sc,
                        tutor=tutor["name"],
                        subject=request.subject,
                        mode=request.mode
                    )
            except Exception as e:
                last_err = e
                await asyncio.sleep(0.5)
                continue

    raise HTTPException(status_code=500, detail=f"AI generation failed: {str(last_err)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)