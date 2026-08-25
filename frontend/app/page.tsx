"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { Html, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";

type Subject = "Physics" | "Chemistry" | "Mathematics";
type TutorMode = "Socratic Hint" | "Full Solution";
type Vector3Tuple = [number, number, number];

interface SphereElement {
  kind: "sphere";
  position: Vector3Tuple;
  radius?: number;
  color?: string;
  label?: string | null;
}

interface CylinderElement {
  kind: "cylinder";
  start: Vector3Tuple;
  end: Vector3Tuple;
  color?: string;
}

interface ArrowElement {
  kind: "arrow";
  start: Vector3Tuple;
  end: Vector3Tuple;
  color?: string;
  label?: string | null;
}

type SceneElement = SphereElement | CylinderElement | ArrowElement;

interface Scene {
  elements: SceneElement[];
}

interface Message {
  role: "user" | "assistant";
  content: string;
  scene?: Scene | null;
  tutor?: string;
}

interface TutorApiResponse {
  explanation: string;
  scene: Scene | null;
  tutor: string;
  subject: Subject;
  mode: TutorMode;
}

const TUTOR_INFO: Record<
  Subject,
  { name: string; institute: string; spec: string }
> = {
  Physics: {
    name: "Rahul",
    institute: "IIT Bombay",
    spec: "Mechanics · Electromagnetism · 3D Vectors",
  },
  Chemistry: {
    name: "Raj",
    institute: "IIT Delhi",
    spec: "VSEPR Geometry · Chemical Bonding · Organic Mechanisms",
  },
  Mathematics: {
    name: "Amit",
    institute: "IIT Kanpur",
    spec: "3D Coordinate Geometry · Vectors · Matrices",
  },
};

const BACKEND_URL = (
  process.env.NEXT_PUBLIC_API_URL ?? "https://chatjeept.onrender.com"
)
  .trim()
  .replace(/^['"]|['"]$/g, "")
  .replace(/\/+$/, "");

const HISTORY_STORAGE_KEY = "chatjeept_history_v10";

function prepareMarkdown(text: string): string {
  if (!text) return "";

  // The backend already separates explanation and scene. Do not attempt to
  // delete, auto-close, or rewrite arbitrary LaTeX here. Only convert the two
  // standard LaTeX delimiters that remark-math does not parse by default.
  return text
    .replace(/\r\n/g, "\n")
    .replace(/\\\[([\s\S]*?)\\\]/g, (_match, math: string) => {
      return `\n\n$$\n${math.trim()}\n$$\n\n`;
    })
    .replace(/\\\(([\s\S]*?)\\\)/g, (_match, math: string) => {
      return `$${math.trim()}$`;
    });
}

function isVector3(value: unknown): value is Vector3Tuple {
  return (
    Array.isArray(value) &&
    value.length === 3 &&
    value.every((coordinate) => typeof coordinate === "number" && Number.isFinite(coordinate))
  );
}

function normalizeScene(value: unknown): Scene | null {
  if (!value || typeof value !== "object") return null;

  const rawElements = (value as { elements?: unknown }).elements;
  if (!Array.isArray(rawElements)) return null;

  const elements: SceneElement[] = [];

  for (const rawElement of rawElements.slice(0, 30)) {
    if (!rawElement || typeof rawElement !== "object") continue;

    const element = rawElement as Record<string, unknown>;
    const color =
      typeof element.color === "string" && /^#[0-9a-fA-F]{6}$/.test(element.color)
        ? element.color
        : undefined;
    const label = typeof element.label === "string" ? element.label.slice(0, 100) : null;

    if (element.kind === "sphere" && isVector3(element.position)) {
      elements.push({
        kind: "sphere",
        position: element.position,
        radius:
          typeof element.radius === "number" && element.radius > 0
            ? Math.min(element.radius, 5)
            : 0.35,
        color,
        label,
      });
      continue;
    }

    if (
      element.kind === "cylinder" &&
      isVector3(element.start) &&
      isVector3(element.end)
    ) {
      elements.push({
        kind: "cylinder",
        start: element.start,
        end: element.end,
        color,
      });
      continue;
    }

    if (element.kind === "arrow" && isVector3(element.start) && isVector3(element.end)) {
      elements.push({
        kind: "arrow",
        start: element.start,
        end: element.end,
        color,
        label,
      });
    }
  }

  return elements.length > 0 ? { elements } : null;
}

function SceneLabel({
  text,
  position,
}: {
  text: string;
  position: Vector3Tuple;
}) {
  return (
    <Html position={position} center distanceFactor={8} style={{ pointerEvents: "none" }}>
      <span className="whitespace-nowrap rounded-md border border-white/20 bg-slate-950/90 px-2 py-1 text-[11px] font-bold text-white shadow-xl">
        {text}
      </span>
    </Html>
  );
}

function CylinderSegment({
  start,
  end,
  color,
}: {
  start: Vector3Tuple;
  end: Vector3Tuple;
  color?: string;
}) {
  const transform = useMemo(() => {
    const p1 = new THREE.Vector3(...start);
    const p2 = new THREE.Vector3(...end);
    const difference = p2.clone().sub(p1);
    const length = difference.length();

    if (length < 0.001) return null;

    const midpoint = p1.clone().add(p2).multiplyScalar(0.5);
    const direction = difference.normalize();
    const quaternion = new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      direction,
    );

    return { length, midpoint, quaternion };
  }, [start, end]);

  if (!transform) return null;

  return (
    <mesh position={transform.midpoint} quaternion={transform.quaternion} castShadow receiveShadow>
      <cylinderGeometry args={[0.06, 0.06, transform.length, 20]} />
      <meshStandardMaterial color={color ?? "#94a3b8"} roughness={0.4} />
    </mesh>
  );
}

function VectorArrow({
  start,
  end,
  color,
  label,
}: {
  start: Vector3Tuple;
  end: Vector3Tuple;
  color?: string;
  label?: string | null;
}) {
  const geometry = useMemo(() => {
    const p1 = new THREE.Vector3(...start);
    const p2 = new THREE.Vector3(...end);
    const difference = p2.clone().sub(p1);
    const totalLength = difference.length();

    if (totalLength < 0.001) return null;

    const direction = difference.normalize();
    const coneLength = Math.min(0.45, Math.max(0.16, totalLength * 0.25));
    const shaftLength = Math.max(0.001, totalLength - coneLength);
    const quaternion = new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      direction,
    );
    const shaftMidpoint = p1.clone().addScaledVector(direction, shaftLength * 0.5);
    const coneMidpoint = p1
      .clone()
      .addScaledVector(direction, shaftLength + coneLength * 0.5);
    const labelPosition = p1.clone().add(p2).multiplyScalar(0.5);
    labelPosition.y += 0.25;

    return {
      coneLength,
      shaftLength,
      quaternion,
      shaftMidpoint,
      coneMidpoint,
      labelPosition: labelPosition.toArray() as Vector3Tuple,
    };
  }, [start, end]);

  if (!geometry) return null;

  return (
    <group>
      <mesh position={geometry.shaftMidpoint} quaternion={geometry.quaternion} castShadow>
        <cylinderGeometry args={[0.045, 0.045, geometry.shaftLength, 18]} />
        <meshStandardMaterial color={color ?? "#f97316"} />
      </mesh>
      <mesh position={geometry.coneMidpoint} quaternion={geometry.quaternion} castShadow>
        <coneGeometry args={[0.13, geometry.coneLength, 18]} />
        <meshStandardMaterial color={color ?? "#f97316"} />
      </mesh>
      {label ? <SceneLabel text={label} position={geometry.labelPosition} /> : null}
    </group>
  );
}

function getSceneView(scene: Scene) {
  const box = new THREE.Box3();
  let hasPoint = false;

  const expand = (point: Vector3Tuple) => {
    box.expandByPoint(new THREE.Vector3(...point));
    hasPoint = true;
  };

  for (const element of scene.elements) {
    if (element.kind === "sphere") {
      expand(element.position);
    } else {
      expand(element.start);
      expand(element.end);
    }
  }

  if (!hasPoint || box.isEmpty()) {
    return {
      center: [0, 0, 0] as Vector3Tuple,
      camera: [5, 4, 7] as Vector3Tuple,
      gridY: -2,
    };
  }

  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const span = Math.max(size.x, size.y, size.z, 2);
  const distance = Math.min(Math.max(span * 1.8, 5), 18);

  return {
    center: center.toArray() as Vector3Tuple,
    camera: [
      center.x + distance * 0.65,
      center.y + distance * 0.55,
      center.z + distance,
    ] as Vector3Tuple,
    gridY: box.min.y - 0.6,
  };
}

function SceneRenderer({ scene }: { scene: Scene }) {
  const view = useMemo(() => getSceneView(scene), [scene]);

  if (!scene.elements.length) return null;

  return (
    <div className="relative my-4 h-[340px] w-full overflow-hidden rounded-2xl border border-slate-800 bg-slate-950 shadow-inner">
      <div className="pointer-events-none absolute left-3 top-3 z-10 rounded-lg border border-slate-700/80 bg-slate-900/90 px-3 py-1 text-[11px] font-bold text-cyan-300 shadow">
        🎮 Interactive 3D View · Drag to rotate · Scroll to zoom
      </div>

      <Canvas shadows camera={{ position: view.camera, fov: 48, near: 0.1, far: 100 }}>
        <color attach="background" args={["#07101f"]} />
        <ambientLight intensity={1.35} />
        <directionalLight position={[8, 10, 7]} intensity={2} castShadow />
        <directionalLight position={[-6, 4, -5]} intensity={0.8} color="#60a5fa" />
        <OrbitControls
          makeDefault
          target={view.center}
          enableDamping
          dampingFactor={0.08}
          minDistance={2}
          maxDistance={30}
        />
        <gridHelper
          args={[14, 14, "#334155", "#1e293b"]}
          position={[view.center[0], view.gridY, view.center[2]]}
        />

        {scene.elements.map((element, index) => {
          if (element.kind === "sphere") {
            const radius = element.radius ?? 0.35;
            return (
              <group key={`sphere-${index}`}>
                <mesh position={element.position} castShadow receiveShadow>
                  <sphereGeometry args={[radius, 36, 36]} />
                  <meshStandardMaterial
                    color={element.color ?? "#60a5fa"}
                    roughness={0.3}
                    metalness={0.15}
                  />
                </mesh>
                {element.label ? (
                  <SceneLabel
                    text={element.label}
                    position={[
                      element.position[0],
                      element.position[1] + radius + 0.28,
                      element.position[2],
                    ]}
                  />
                ) : null}
              </group>
            );
          }

          if (element.kind === "cylinder") {
            return (
              <CylinderSegment
                key={`cylinder-${index}`}
                start={element.start}
                end={element.end}
                color={element.color}
              />
            );
          }

          return (
            <VectorArrow
              key={`arrow-${index}`}
              start={element.start}
              end={element.end}
              color={element.color}
              label={element.label}
            />
          );
        })}
      </Canvas>
    </div>
  );
}

export default function ChatJEEPTPage() {
  const [subject, setSubject] = useState<Subject>("Chemistry");
  const [mode, setMode] = useState<TutorMode>("Full Solution");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const saved = window.localStorage.getItem(HISTORY_STORAGE_KEY);

    if (saved) {
      try {
        const parsed = JSON.parse(saved) as unknown;
        if (Array.isArray(parsed)) {
          setMessages(parsed as Message[]);
        }
      } catch {
        window.localStorage.removeItem(HISTORY_STORAGE_KEY);
      }
    }

    setIsLoaded(true);
  }, []);

  useEffect(() => {
    if (!isLoaded) return;
    window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(messages));
  }, [messages, isLoaded]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const activeTutor = TUTOR_INFO[subject];

  const handleNewChat = () => {
    const shouldClear = window.confirm(
      "Are you sure you want to clear the conversation history and start a fresh session?",
    );

    if (shouldClear) {
      setMessages([]);
      window.localStorage.removeItem(HISTORY_STORAGE_KEY);
    }
  };

  const handleSend = async (event?: React.FormEvent) => {
    event?.preventDefault();

    const userText = input.trim();
    if (!userText || loading) return;

    // Send only previous turns as history. The current question is already in
    // the `message` field and must not be duplicated in `history`.
    const previousHistory = messages.slice(-20).map((message) => ({
      role: message.role,
      content: message.content,
    }));

    const userMessage: Message = { role: "user", content: userText, scene: null };
    setMessages((current) => [...current, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch(`${BACKEND_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userText,
          subject,
          mode,
          history: previousHistory,
        }),
      });

      const responseText = await response.text();
      let data: TutorApiResponse | { detail?: string };

      try {
        data = JSON.parse(responseText) as TutorApiResponse | { detail?: string };
      } catch {
        throw new Error(`Invalid server response (HTTP ${response.status})`);
      }

      if (!response.ok) {
        const detail =
          "detail" in data && typeof data.detail === "string"
            ? data.detail
            : `HTTP ${response.status}`;
        throw new Error(detail);
      }

      if (!("explanation" in data) || typeof data.explanation !== "string") {
        throw new Error("The server response does not contain an explanation.");
      }

      const assistantMessage: Message = {
        role: "assistant",
        content: data.explanation,
        scene: normalizeScene(data.scene),
        tutor: typeof data.tutor === "string" ? data.tutor : activeTutor.name,
      };

      setMessages((current) => [...current, assistantMessage]);
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : "Unknown connection error";
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: `⚠️ Error: ${errorMessage}`,
          tutor: activeTutor.name,
          scene: null,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col bg-slate-950 font-sans text-slate-100">
      <header className="sticky top-0 z-30 flex min-h-16 flex-wrap items-center justify-between gap-3 border-b border-slate-800 bg-slate-900/80 px-4 py-3 backdrop-blur md:px-6">
        <div className="flex items-center gap-3">
          <span className="bg-gradient-to-r from-cyan-400 to-indigo-400 bg-clip-text text-xl font-black text-transparent">
            ChatJEEPT
          </span>
          <span className="rounded-full border border-cyan-800 bg-cyan-950 px-2 py-0.5 text-[11px] font-bold text-cyan-300">
            3D AI Tutor
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-2 md:gap-3">
          <button
            type="button"
            onClick={handleNewChat}
            disabled={loading}
            className="rounded-xl border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-bold text-slate-300 transition hover:bg-slate-700 disabled:opacity-50"
          >
            ➕ New Chat
          </button>

          <select
            value={subject}
            disabled={loading}
            onChange={(event) => setSubject(event.target.value as Subject)}
            className="rounded-xl border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-bold text-slate-200 outline-none focus:border-cyan-500 disabled:opacity-50"
          >
            <option value="Physics">Physics — Rahul (IIT Bombay)</option>
            <option value="Chemistry">Chemistry — Raj (IIT Delhi)</option>
            <option value="Mathematics">Mathematics — Amit (IIT Kanpur)</option>
          </select>

          <div className="flex rounded-xl border border-slate-700 bg-slate-800 p-0.5">
            {(["Socratic Hint", "Full Solution"] as TutorMode[]).map((item) => (
              <button
                key={item}
                type="button"
                disabled={loading}
                onClick={() => setMode(item)}
                className={`rounded-lg px-3 py-1 text-xs font-bold transition ${
                  mode === item
                    ? "bg-cyan-500 text-slate-950 shadow"
                    : "text-slate-400 hover:text-white"
                } disabled:opacity-50`}
              >
                {item}
              </button>
            ))}
          </div>
        </div>
      </header>

      <div className="mx-auto w-full max-w-5xl flex-1 space-y-6 overflow-y-auto p-4 md:p-6">
        {messages.length === 0 ? (
          <div className="space-y-4 rounded-3xl border border-slate-800 bg-slate-900 p-6 shadow-xl md:p-8">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-2xl border border-cyan-500/40 bg-cyan-500/20 text-lg font-black text-cyan-400">
                {activeTutor.name[0]}
              </div>
              <div>
                <h2 className="text-base font-black text-white">
                  Master {activeTutor.name}{" "}
                  <span className="text-xs font-normal text-slate-400">
                    ({activeTutor.institute})
                  </span>
                </h2>
                <p className="font-mono text-xs text-cyan-400">{activeTutor.spec}</p>
              </div>
            </div>
            <p className="text-sm leading-relaxed text-slate-300">
              Hello! I am your IIT-JEE Master Tutor in <strong>{subject}</strong>. Ask any
              conceptual doubt, derivation, numerical problem, or JEE Advanced question.
            </p>
          </div>
        ) : null}

        {messages.map((message, index) => (
          <div
            key={`${message.role}-${index}`}
            className={`flex flex-col space-y-2 ${
              message.role === "user" ? "items-end" : "items-start"
            }`}
          >
            <div
              className={`max-w-[92%] rounded-3xl p-5 md:max-w-[88%] ${
                message.role === "user"
                  ? "rounded-br-none bg-cyan-500 font-medium text-slate-950 shadow-lg shadow-cyan-500/20"
                  : "rounded-bl-none border border-slate-800 bg-slate-900 text-slate-100 shadow-xl"
              }`}
            >
              {message.role === "assistant" ? (
                <div className="mb-2 font-mono text-xs font-bold text-cyan-400">
                  🎓 Master {message.tutor ?? activeTutor.name}
                </div>
              ) : null}

              {message.role === "user" ? (
                <div className="whitespace-pre-wrap text-sm">{message.content}</div>
              ) : (
                <div className="space-y-3 text-sm leading-relaxed text-slate-200 [&_.katex-display]:overflow-x-auto">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm, remarkMath]}
                    rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: false }]]}
                    components={{
                      table: ({ ...props }) => (
                        <div className="my-4 overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/80 shadow-md">
                          <table
                            className="w-full divide-y divide-slate-800 text-left text-xs text-slate-200"
                            {...props}
                          />
                        </div>
                      ),
                      thead: ({ ...props }) => (
                        <thead className="bg-slate-800/90 font-bold text-cyan-400" {...props} />
                      ),
                      th: ({ ...props }) => (
                        <th
                          className="border-b border-slate-700 px-4 py-2.5 font-bold"
                          {...props}
                        />
                      ),
                      td: ({ ...props }) => (
                        <td className="border-b border-slate-800/60 px-4 py-2 text-slate-300" {...props} />
                      ),
                      h1: ({ ...props }) => (
                        <h1 className="mb-2 mt-4 text-lg font-black text-cyan-400" {...props} />
                      ),
                      h2: ({ ...props }) => (
                        <h2 className="mb-2 mt-3 text-base font-bold text-cyan-300" {...props} />
                      ),
                      h3: ({ ...props }) => (
                        <h3 className="mb-1 mt-3 text-sm font-bold text-cyan-200" {...props} />
                      ),
                      p: ({ ...props }) => (
                        <p className="mb-2 text-sm leading-relaxed text-slate-200" {...props} />
                      ),
                      strong: ({ ...props }) => (
                        <strong className="font-bold tracking-wide text-white" {...props} />
                      ),
                      hr: ({ ...props }) => <hr className="my-4 border-slate-800" {...props} />,
                    }}
                  >
                    {prepareMarkdown(message.content)}
                  </ReactMarkdown>
                </div>
              )}

              {message.scene ? <SceneRenderer scene={message.scene} /> : null}
            </div>
          </div>
        ))}

        {loading ? (
          <div className="flex w-fit items-center gap-3 rounded-2xl border border-slate-800/80 bg-slate-900/60 px-4 py-3 font-mono text-xs text-cyan-400">
            <span className="animate-spin">⚙️</span>
            Calculating derivations and generating the 3D scene...
          </div>
        ) : null}

        <div ref={chatEndRef} />
      </div>

      <footer className="sticky bottom-0 z-30 border-t border-slate-800 bg-slate-900/90 p-4 backdrop-blur">
        <form onSubmit={handleSend} className="mx-auto flex max-w-5xl gap-3">
          <input
            type="text"
            value={input}
            disabled={loading}
            onChange={(event) => setInput(event.target.value)}
            placeholder={`Ask Master ${activeTutor.name} a ${subject} question...`}
            className="flex-1 rounded-2xl border border-slate-800 bg-slate-950 px-5 py-3.5 text-sm text-white shadow-inner outline-none placeholder:text-slate-500 focus:border-cyan-500 disabled:opacity-60"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="rounded-2xl bg-cyan-400 px-6 py-3.5 text-sm font-black text-slate-950 shadow-lg shadow-cyan-400/20 transition hover:bg-cyan-300 disabled:opacity-50"
          >
            Send
          </button>
        </form>
      </footer>
    </main>
  );
}
