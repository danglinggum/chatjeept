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
# 1. Performance: In-Memory Response Caching (0.01s Instant Reply)
# =============================================================================

RESPONSE_CACHE: dict[str, dict[str, Any]] = {}
MAX_CACHE_SIZE = 300

def get_from_cache(key: str) -> dict[str, Any] | None:
    return RESPONSE_CACHE.get(key)

def set_to_cache(key: str, data: dict[str, Any]):
    if len(RESPONSE_CACHE) >= MAX_CACHE_SIZE:
        # 오래된 캐시 50개 자동 정리
        for k in list(RESPONSE_CACHE.keys())[:50]:
            RESPONSE_CACHE.pop(k, None)
    RESPONSE_CACHE[key] = data

# =============================================================================
# 2. Security: Lightweight Rate Limiter
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
# 3. Google GenAI Client & High-Speed Model Priority
# =============================================================================

SCENE_START_MARKER = "<<<3D_SCENE>>>"
SCENE_END_MARKER = "<<<END_3D_SCENE>>>"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
ai_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None

# 초고속 경량 모델 우선 호출 -> 실패 시 플래시 전환
MODELS_TO_TRY = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
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

app = FastAPI(title="ChatJEEPT Turbo API", version="9.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 토큰 수를 줄여 레이턴시를 최소화한 최적화 시스템 프롬프트
SYSTEM_INSTRUCTION = """
You are an elite IIT-JEE Master Tutor (Rahul: Physics, Raj: Chemistry, Amit: Mathematics).
Deliver direct, mathematically rigorous solutions in English without conversational preamble.

FORMATTING RULES:
1. Explain in markdown. NEVER put plain English prose inside LaTeX tags.
2. Space inline math cleanly: "A mass $m$ with velocity $v$ at $t = 0$."
3. Place major derivations on their own separate lines in double dollars:
   $$m\\ddot{x} = q B_0 \\dot{y}$$
4. Use standard cases block for piecewise formulas:
   $$\\begin{cases} x(t) = \\dfrac{m E_0}{q B_0^2}(\\omega t - \\sin\\omega t) \\\\[6pt] y(t) = \\dfrac{m E_0}{q B_0^2}(1 - \\cos\\omega t) \\end{cases}$$

3D VISUALIZATION:
Always append the 3D scene JSON at the very end using the delimiters. Do NOT write prose descriptions for the 3D model.
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
            temperature=0.1,         # 낮은 온도로 연산 속도 및 결정론적 수식 보장
            max_output_tokens=4096,  # 8192에서 4096으로 최적화하여 생성 속도 향상
        ),
    )

@app.get("/")
@app.get("/health")
async def health():
    return {"status": "online", "service": "ChatJEEPT Turbo API v9.0"}

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

    # 1. 인메모리 캐시 조회 (동일 질문 0.01초 즉시 응답)
    cache_key = f"{request.subject}:{request.mode}:{request.message.strip().lower()}"
    cached_payload = get_from_cache(cache_key)
    if cached_payload:
        return TutorResponse(**cached_payload)

    # 2. 토큰 절약을 위해 최근 1턴 히스토리만 전달
    history_ctx = [{"role": h.role, "content": h.content} for h in request.history[-2:]]
    prompt = f"Subject: {request.subject}\nMode: {request.mode}\nContext: {json.dumps(history_ctx)}\nQuestion: {request.message}"

    last_err = None
    for model_name in MODELS_TO_TRY:
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
            continue

    raise HTTPException(status_code=500, detail=f"AI generation failed: {str(last_err)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)