from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field


load_dotenv()


# =============================================================================
# Configuration
# =============================================================================

PROMPT_VERSION = "10.0"
SCENE_START_MARKER = "<<<3D_SCENE>>>"
SCENE_END_MARKER = "<<<END_3D_SCENE>>>"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
MODEL_NAMES = [
    name.strip()
    for name in os.getenv(
        "GEMINI_MODELS",
        "gemini-3.7-flash,gemini-3.6-flash",
    ).split(",")
    if name.strip()
]
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "16384"))

ai_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None


# =============================================================================
# Cache and rate limiting
# =============================================================================

RESPONSE_CACHE: dict[str, dict[str, Any]] = {}
MAX_CACHE_SIZE = 300

REQUEST_RECORDS: defaultdict[str, list[float]] = defaultdict(list)
RATE_LIMIT_WINDOW_SECONDS = 60
MAX_REQUESTS_PER_WINDOW = 40


def get_from_cache(key: str) -> dict[str, Any] | None:
    return RESPONSE_CACHE.get(key)


def set_to_cache(key: str, data: dict[str, Any]) -> None:
    if len(RESPONSE_CACHE) >= MAX_CACHE_SIZE:
        oldest_keys = list(RESPONSE_CACHE.keys())[:50]
        for old_key in oldest_keys:
            RESPONSE_CACHE.pop(old_key, None)

    RESPONSE_CACHE[key] = data


def check_rate_limit(ip_address: str) -> None:
    now = time.time()
    recent = [
        timestamp
        for timestamp in REQUEST_RECORDS[ip_address]
        if now - timestamp < RATE_LIMIT_WINDOW_SECONDS
    ]

    if len(recent) >= MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a few seconds.",
        )

    recent.append(now)
    REQUEST_RECORDS[ip_address] = recent


# =============================================================================
# API data models
# =============================================================================

Subject = Literal["Physics", "Chemistry", "Mathematics"]
TutorMode = Literal["Socratic Hint", "Full Solution"]
Vector3 = tuple[float, float, float]

TUTOR_CONFIG: dict[str, dict[str, str]] = {
    "Physics": {
        "name": "Rahul",
        "institution": "IIT Bombay",
        "specialty": "mechanics, electromagnetism, force diagrams, and 3D vectors",
        "data_file": "physics.json",
    },
    "Chemistry": {
        "name": "Raj",
        "institution": "IIT Delhi",
        "specialty": "VSEPR geometry, bonding, stereochemistry, and organic mechanisms",
        "data_file": "chemistry.json",
    },
    "Mathematics": {
        "name": "Amit",
        "institution": "IIT Kanpur",
        "specialty": "3D coordinate geometry, vectors, matrices, and calculus",
        "data_file": "math.json",
    },
}


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


SceneElement = Annotated[
    SphereElement | CylinderElement | ArrowElement,
    Field(discriminator="kind"),
]


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
    history: list[ChatMessage] = Field(default_factory=list, max_length=30)


# =============================================================================
# FastAPI application
# =============================================================================

app = FastAPI(
    title="ChatJEEPT Turbo API",
    version="10.0.0",
    redirect_slashes=False,
)

configured_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "https://chatjeept-iota.vercel.app,http://localhost:3000",
    ).split(",")
    if origin.strip()
]
allow_all_origins = "*" in configured_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all_origins else configured_origins,
    allow_credentials=not allow_all_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# =============================================================================
# Optional local data injection
# =============================================================================

def load_reference_data(subject: Subject) -> str:
    filename = TUTOR_CONFIG[subject]["data_file"]
    module_directory = Path(__file__).resolve().parent
    candidates = [
        module_directory / "data" / filename,
        Path.cwd() / "data" / filename,
    ]

    for path in candidates:
        if not path.is_file():
            continue

        try:
            raw = path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
            serialized = json.dumps(parsed, ensure_ascii=False, indent=2)
            return serialized[:100000]
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue

    return "No local reference file is available."


# =============================================================================
# Prompt
# =============================================================================

# This must be a raw string. A normal Python string would interpret sequences
# such as \t in \text, \b in \begin, and \f in \frac as control characters.
SYSTEM_INSTRUCTION = rf"""
You are ChatJEEPT, an elite IIT-JEE tutor.

The active tutor, subject, institution, teaching mode, conversation history,
and optional reference material are supplied with each question.

WRITTEN EXPLANATION

1. Write a complete, concise, rigorous explanation in English Markdown.
2. Use $...$ for inline LaTeX and $$...$$ for display LaTeX.
3. Put blank lines before and after every display-math block.
4. Close every LaTeX brace, delimiter, and environment before continuing.
5. Do not place ordinary explanatory paragraphs inside math delimiters.
6. Prefer ordinary Markdown lists over large aligned environments.
7. Keep the response under approximately 3,000 words so it reaches the scene.

TEACHING MODES

- Socratic Hint: guide with progressive questions and hints without immediately
  revealing the entire result.
- Full Solution: provide a complete step-by-step derivation and final result.

REQUIRED OUTPUT FORMAT

First output the complete Markdown/LaTeX explanation as ordinary text.
Then output this exact marker on its own line:

{SCENE_START_MARKER}

If a useful 3D scene exists, output one strict JSON object like this:

{{
  "elements": [
    {{
      "kind": "sphere",
      "position": [0, 0, 0],
      "radius": 0.35,
      "color": "#60a5fa",
      "label": "Point A"
    }},
    {{
      "kind": "cylinder",
      "start": [0, 0, 0],
      "end": [2, 0, 0],
      "color": "#94a3b8"
    }},
    {{
      "kind": "arrow",
      "start": [0, 0, 0],
      "end": [0, 2, 0],
      "color": "#f97316",
      "label": "Force F"
    }}
  ]
}}

If 3D does not materially help, output null instead of an object.
Finally output this exact marker on its own line:

{SCENE_END_MARKER}

SCENE RULES

1. The scene block is the only JSON in the response.
2. Use strict JSON: double quotes, no comments, no ellipsis, no trailing commas.
3. Do not wrap the JSON in Markdown code fences.
4. Use only sphere, cylinder, and arrow.
5. Use at most 30 elements and keep coordinates between -6 and 6.
6. Labels must be short plain text. Never put LaTeX or backslashes in labels.
7. For comparisons, separate structures along the x-axis.
8. The scene must agree with the written explanation.
""".strip()


def build_prompt(request: ChatRequest) -> str:
    tutor = TUTOR_CONFIG[request.subject]
    history_payload = [
        {"role": message.role, "content": message.content}
        for message in request.history[-12:]
    ]

    return f"""
ACTIVE TUTOR
Name: {tutor['name']}
Institution: {tutor['institution']}
Subject: {request.subject}
Specialty: {tutor['specialty']}

TEACHING MODE
{request.mode}

RECENT CONVERSATION
{json.dumps(history_payload, ensure_ascii=False, indent=2)}

OPTIONAL LOCAL REFERENCE
{load_reference_data(request.subject)}

CURRENT QUESTION
{request.message}

Answer the current question once. Do not repeat it. Follow the required output
format exactly and finish the scene block before ending the response.
""".strip()


# =============================================================================
# Defensive scene parsing
# =============================================================================

START_MARKER_PATTERN = re.compile(r"<<<\s*3D_SCENE\s*>>>", re.IGNORECASE)
END_MARKER_PATTERN = re.compile(r"<<<\s*END_3D_SCENE\s*>>>", re.IGNORECASE)


def escape_invalid_json_backslashes(value: str) -> str:
    valid_simple_escapes = {'"', "\\", "/", "b", "f", "n", "r", "t"}
    hexadecimal = set("0123456789abcdefABCDEF")
    result: list[str] = []
    index = 0

    while index < len(value):
        character = value[index]

        if character != "\\":
            result.append(character)
            index += 1
            continue

        if index + 1 >= len(value):
            result.append("\\\\")
            index += 1
            continue

        following = value[index + 1]

        if following in valid_simple_escapes:
            result.extend(["\\", following])
            index += 2
            continue

        if following == "u":
            digits = value[index + 2 : index + 6]
            if len(digits) == 4 and all(digit in hexadecimal for digit in digits):
                result.extend(["\\u", digits])
                index += 6
                continue

        result.append("\\\\")
        index += 1

    return "".join(result)


def isolate_json_object(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, count=1, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned, count=1)

    object_start = cleaned.find("{")
    object_end = cleaned.rfind("}")

    if object_start != -1 and object_end > object_start:
        return cleaned[object_start : object_end + 1]

    return cleaned


def parse_json_defensively(value: str) -> Any | None:
    cleaned = isolate_json_object(value)

    if not cleaned or cleaned.lower() == "null":
        return None

    repaired_backslashes = escape_invalid_json_backslashes(cleaned)
    attempts = [
        cleaned,
        repaired_backslashes,
        re.sub(r",\s*([}\]])", r"\1", cleaned),
        re.sub(r",\s*([}\]])", r"\1", repaired_backslashes),
    ]

    for attempt in attempts:
        try:
            return json.loads(attempt)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

    try:
        literal = ast.literal_eval(cleaned)
        return literal if isinstance(literal, (dict, list)) else None
    except (ValueError, SyntaxError):
        return None


def clean_color(value: Any, fallback: str) -> str:
    if isinstance(value, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", value):
        return value
    return fallback


def clean_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    cleaned = re.sub(r"[\x00-\x1f\x7f]", " ", value)
    cleaned = cleaned.replace("\\", "").replace('"', "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:100] or None


def clean_vector(value: Any) -> Vector3 | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None

    try:
        coordinates = tuple(float(coordinate) for coordinate in value)
    except (TypeError, ValueError):
        return None

    return tuple(max(-6.0, min(6.0, coordinate)) for coordinate in coordinates)  # type: ignore[return-value]


def validate_scene_payload(value: Any) -> Scene | None:
    if not isinstance(value, dict) or not isinstance(value.get("elements"), list):
        return None

    valid_elements: list[SceneElement] = []

    for raw_element in value["elements"][:30]:
        if not isinstance(raw_element, dict):
            continue

        kind = raw_element.get("kind")
        normalized: dict[str, Any] | None = None

        if kind == "sphere":
            position = clean_vector(raw_element.get("position"))
            if position is not None:
                radius_value = raw_element.get("radius", 0.35)
                try:
                    radius = max(0.05, min(5.0, float(radius_value)))
                except (TypeError, ValueError):
                    radius = 0.35

                normalized = {
                    "kind": "sphere",
                    "position": position,
                    "radius": radius,
                    "color": clean_color(raw_element.get("color"), "#60a5fa"),
                    "label": clean_label(raw_element.get("label")),
                }

        elif kind == "cylinder":
            start = clean_vector(raw_element.get("start"))
            end = clean_vector(raw_element.get("end"))
            if start is not None and end is not None and start != end:
                normalized = {
                    "kind": "cylinder",
                    "start": start,
                    "end": end,
                    "color": clean_color(raw_element.get("color"), "#94a3b8"),
                }

        elif kind == "arrow":
            start = clean_vector(raw_element.get("start"))
            end = clean_vector(raw_element.get("end"))
            if start is not None and end is not None and start != end:
                normalized = {
                    "kind": "arrow",
                    "start": start,
                    "end": end,
                    "color": clean_color(raw_element.get("color"), "#f97316"),
                    "label": clean_label(raw_element.get("label")),
                }

        if normalized is None:
            continue

        try:
            if kind == "sphere":
                valid_elements.append(SphereElement.model_validate(normalized))
            elif kind == "cylinder":
                valid_elements.append(CylinderElement.model_validate(normalized))
            elif kind == "arrow":
                valid_elements.append(ArrowElement.model_validate(normalized))
        except Exception:
            continue

    if not valid_elements:
        return None

    return Scene(elements=valid_elements)


def extract_explanation_and_scene(raw_text: str) -> tuple[str, Scene | None, str]:
    text = raw_text.strip()
    start_match = START_MARKER_PATTERN.search(text)

    if start_match:
        end_match = END_MARKER_PATTERN.search(text, start_match.end())
        scene_end = end_match.start() if end_match else len(text)
        scene_text = text[start_match.end() : scene_end].strip()

        explanation_parts = [text[: start_match.start()].strip()]
        if end_match:
            explanation_parts.append(text[end_match.end() :].strip())
        explanation = "\n\n".join(part for part in explanation_parts if part)

        if scene_text.lower() == "null":
            return explanation, None, "null"

        scene = validate_scene_payload(parse_json_defensively(scene_text))
        return explanation, scene, "valid" if scene else "invalid"

    # Recovery for a model that omitted markers but appended a final scene JSON.
    elements_key = text.rfind('"elements"')
    if elements_key != -1:
        object_start = text.rfind("{", 0, elements_key)
        if object_start != -1:
            scene = validate_scene_payload(parse_json_defensively(text[object_start:]))
            if scene:
                return text[:object_start].strip(), scene, "valid"

    return text, None, "missing"


# =============================================================================
# Gemini calls
# =============================================================================

async def generate_main_response(model_name: str, prompt: str):
    if ai_client is None:
        raise RuntimeError("Google GenAI client is not configured")

    return await ai_client.aio.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.15,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )


async def generate_scene_fallback(
    model_name: str,
    request: ChatRequest,
    explanation: str,
) -> Scene | None:
    if ai_client is None:
        return None

    scene_prompt = f"""
Create only a compact 3D scene for this IIT-JEE answer.

Subject: {request.subject}
Question: {request.message}
Explanation summary:
{explanation[:6000]}

Use sphere, cylinder, and arrow elements only. Use at most 30 elements. Keep
coordinates between -6 and 6. Labels must be short plain text with no LaTeX or
backslashes. If 3D would not help, return an object with an empty elements list.
""".strip()

    try:
        response = await ai_client.aio.models.generate_content(
            model=model_name,
            contents=scene_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Scene,
                temperature=0.1,
                max_output_tokens=4096,
            ),
        )

        if isinstance(response.parsed, Scene):
            return validate_scene_payload(response.parsed.model_dump())

        if response.parsed is not None:
            parsed_scene = validate_scene_payload(response.parsed)
            if parsed_scene:
                return parsed_scene

        if response.text:
            return validate_scene_payload(parse_json_defensively(response.text))
    except Exception:
        return None

    return None


def build_cache_key(request: ChatRequest) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "subject": request.subject,
        "mode": request.mode,
        "message": request.message.strip(),
        "history": [message.model_dump() for message in request.history[-8:]],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# =============================================================================
# Routes
# =============================================================================

@app.get("/")
@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "online",
        "service": "ChatJEEPT Turbo API v10.0",
        "models": MODEL_NAMES,
        "commit": os.getenv("RENDER_GIT_COMMIT", "local"),
    }


@app.post("/api/chat", response_model=TutorResponse)
@app.post("/api/chat/", response_model=TutorResponse, include_in_schema=False)
@app.post("/chat", response_model=TutorResponse, include_in_schema=False)
@app.post("/chat/", response_model=TutorResponse, include_in_schema=False)
async def chat_endpoint(request: ChatRequest, raw_request: Request) -> TutorResponse:
    if ai_client is None:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY or GEMINI_API_KEY is not configured.",
        )

    forwarded_for = raw_request.headers.get("x-forwarded-for")
    client_ip = (
        forwarded_for.split(",")[0].strip()
        if forwarded_for
        else raw_request.client.host if raw_request.client else "127.0.0.1"
    )
    check_rate_limit(client_ip)

    tutor = TUTOR_CONFIG[request.subject]
    cache_key = build_cache_key(request)
    cached_payload = get_from_cache(cache_key)

    if cached_payload:
        return TutorResponse.model_validate(cached_payload)

    prompt = build_prompt(request)
    last_error: Exception | None = None

    for model_name in MODEL_NAMES:
        for attempt in range(2):
            try:
                response = await generate_main_response(model_name, prompt)
                raw_text = response.text or ""

                if not raw_text.strip():
                    raise RuntimeError("Gemini returned an empty text response")

                explanation, scene, scene_status = extract_explanation_and_scene(raw_text)

                if not explanation.strip():
                    explanation = "The explanation could not be generated completely. Please retry."

                # If the main answer lost or malformed its scene block, make one
                # isolated structured-output retry. LaTeX remains outside this JSON.
                if scene is None and scene_status in {"missing", "invalid"}:
                    scene = await generate_scene_fallback(
                        model_name,
                        request,
                        explanation,
                    )

                result = TutorResponse(
                    explanation=explanation,
                    scene=scene,
                    tutor=tutor["name"],
                    subject=request.subject,
                    mode=request.mode,
                )

                set_to_cache(cache_key, result.model_dump(mode="json"))
                return result

            except Exception as error:
                last_error = error
                if attempt == 0:
                    await asyncio.sleep(0.5)

    raise HTTPException(
        status_code=502,
        detail=f"AI generation failed: {last_error}",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("ENVIRONMENT", "development").lower() == "development",
    )
