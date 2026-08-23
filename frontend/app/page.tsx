"use client";

import React, { useState, useRef, useEffect } from "react";
import dynamic from "next/dynamic";

const ThreeCanvas = dynamic(() => import("@react-three/fiber").then((mod) => mod.Canvas), { ssr: false });
const OrbitControls = dynamic(() => import("@react-three/drei").then((mod) => mod.OrbitControls), { ssr: false });

type Subject = "Physics" | "Chemistry" | "Mathematics";
type TutorMode = "Socratic Hint" | "Full Solution";

interface SceneElement {
  kind: "sphere" | "cylinder" | "arrow";
  position?: [number, number, number];
  start?: [number, number, number];
  end?: [number, number, number];
  radius?: number;
  color?: string;
  label?: string;
}

interface Scene {
  elements: SceneElement[];
}

interface Message {
  role: "user" | "assistant";
  content: string;
  scene?: Scene | null;
  tutor?: string;
}

const TUTOR_INFO: Record<Subject, { name: string; institute: string; spec: string }> = {
  Physics: { name: "Rahul", institute: "IIT Bombay", spec: "Mechanics · E&M · 3D Vectors" },
  Chemistry: { name: "Raj", institute: "IIT Delhi", spec: "VSEPR Geometry · Bonding · Organic" },
  Mathematics: { name: "Amit", institute: "IIT Kanpur", spec: "3D Coordinate Geometry · Vectors" },
};

function resolveBackendUrl(): string {
  let raw = process.env.NEXT_PUBLIC_API_URL || "https://chatjeept.onrender.com";
  const mdMatch = raw.match(/\((https?:\/\/[^\)]+)\)/);
  if (mdMatch) raw = mdMatch[1];
  raw = raw.replace(/["'\[\]]/g, "").trim().replace(/\/+$/, "");
  if (!raw.startsWith("http")) return "https://chatjeept.onrender.com";
  return raw;
}

const BACKEND_URL = resolveBackendUrl();

// LaTeX 수식 및 마크다운 포맷팅 렌더러
function FormattedMessage({ content }: { content: string }) {
  const cleanText = content.split("<<<3D_SCENE>>>")[0].trim();
  const blocks = cleanText.split(/(\$\$[\s\S]*?\$\$)/g);

  return (
    <div className="space-y-3 leading-relaxed text-sm text-slate-200 font-sans">
      {blocks.map((block, idx) => {
        if (block.startsWith("$$") && block.endsWith("$$")) {
          const math = block.slice(2, -2).trim();
          return (
            <div
              key={idx}
              className="my-3 p-3.5 bg-slate-950/90 border border-cyan-800/40 rounded-2xl text-center font-mono text-cyan-300 text-sm overflow-x-auto shadow-inner"
            >
              {math}
            </div>
          );
        }

        return (
          <div key={idx} className="space-y-2">
            {block.split("\n").map((line, lIdx) => {
              if (!line.trim()) return <div key={lIdx} className="h-1" />;
              return (
                <div key={lIdx} className="leading-relaxed">
                  {renderInline(line)}
                </div>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}

function renderInline(text: string) {
  const parts = text.split(/(\$[^\$\n]+\$|\*\*[^\*]+\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith("$") && part.endsWith("$") && part.length > 2) {
      return (
        <span
          key={i}
          className="font-mono text-cyan-300 bg-slate-800/90 px-1.5 py-0.5 rounded text-[13px] mx-0.5 border border-cyan-900/40 font-semibold"
        >
          {part.slice(1, -1)}
        </span>
      );
    }
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return (
        <strong key={i} className="font-bold text-white tracking-wide">
          {part.slice(2, -2)}
        </strong>
      );
    }
    return part;
  });
}

function SceneRenderer({ scene }: { scene: Scene }) {
  return (
    <div className="w-full h-72 bg-slate-950 rounded-2xl overflow-hidden border border-slate-800 relative shadow-inner my-4">
      <div className="absolute top-3 left-3 z-10 bg-slate-900/80 border border-slate-700/60 px-3 py-1 rounded-lg text-[11px] font-mono font-bold text-cyan-300">
        🎮 Interactive 3D Viewport (Drag to Rotate / Scroll to Zoom)
      </div>
      <ThreeCanvas camera={{ position: [0, 2, 6], fov: 50 }}>
        <ambientLight intensity={1.2} />
        <pointLight position={[10, 10, 10]} intensity={1.5} />
        <directionalLight position={[-5, 5, 5]} intensity={0.8} />
        <OrbitControls makeDefault />
        <gridHelper args={[10, 10, "#334155", "#1e293b"]} position={[0, -1, 0]} />

        {scene.elements.map((el, idx) => {
          if (el.kind === "sphere" && el.position) {
            return (
              <mesh key={idx} position={el.position}>
                <sphereGeometry args={[el.radius || 0.35, 32, 32]} />
                <meshStandardMaterial color={el.color || "#60a5fa"} roughness={0.3} metalness={0.2} />
              </mesh>
            );
          }
          if (el.kind === "cylinder" && el.start && el.end) {
            const mid: [number, number, number] = [
              (el.start[0] + el.end[0]) / 2,
              (el.start[1] + el.end[1]) / 2,
              (el.start[2] + el.end[2]) / 2,
            ];
            return (
              <mesh key={idx} position={mid}>
                <cylinderGeometry args={[0.08, 0.08, 1.5, 16]} />
                <meshStandardMaterial color={el.color || "#94a3b8"} roughness={0.4} />
              </mesh>
            );
          }
          if (el.kind === "arrow" && el.start && el.end) {
            return (
              <mesh key={idx} position={el.start}>
                <sphereGeometry args={[0.15, 16, 16]} />
                <meshStandardMaterial color={el.color || "#f97316"} />
              </mesh>
            );
          }
          return null;
        })}
      </ThreeCanvas>
    </div>
  );
}

export default function ChatJEEPTPage() {
  const [subject, setSubject] = useState<Subject>("Physics");
  const [mode, setMode] = useState<TutorMode>("Socratic Hint");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const saved = localStorage.getItem("chatjeept_history");
    if (saved) {
      try {
        setMessages(JSON.parse(saved));
      } catch (e) {
        console.error(e);
      }
    }
    setIsLoaded(true);
  }, []);

  useEffect(() => {
    if (isLoaded) {
      localStorage.setItem("chatjeept_history", JSON.stringify(messages));
    }
  }, [messages, isLoaded]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const activeTutor = TUTOR_INFO[subject];

  const handleNewChat = () => {
    if (confirm("대화 기록을 비우고 새 대화를 시작할까요?")) {
      setMessages([]);
      localStorage.removeItem("chatjeept_history");
    }
  };

  const handleSend = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!input.trim() || loading) return;

    const userText = input.trim();
    setInput("");

    const newHistory: Message[] = [...messages, { role: "user", content: userText }];
    setMessages(newHistory);
    setLoading(true);

    const requestUrl = `${BACKEND_URL}/api/chat`;

    try {
      const res = await fetch(requestUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userText,
          subject: subject,
          mode: mode,
          history: newHistory.map((m) => ({ role: m.role, content: m.content })),
        }),
      });

      if (!res.ok) {
        let errDetail = `HTTP ${res.status}`;
        try {
          const errData = await res.json();
          if (errData.detail) errDetail = errData.detail;
        } catch (_) {}
        throw new Error(errDetail);
      }

      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.explanation,
          scene: data.scene,
          tutor: data.tutor,
        },
      ]);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `⚠️ 오류: ${err.message || "연결 오류가 발생했습니다."}`,
          tutor: activeTutor.name,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <header className="h-16 border-b border-slate-800 bg-slate-900/60 backdrop-blur px-6 flex items-center justify-between sticky top-0 z-30">
        <div className="flex items-center gap-3">
          <span className="text-xl font-black bg-gradient-to-r from-cyan-400 to-indigo-400 bg-clip-text text-transparent">
            ChatJEEPT
          </span>
          <span className="text-[11px] bg-cyan-950 text-cyan-300 border border-cyan-800 px-2 py-0.5 rounded-full font-bold">
            3D AI Tutor
          </span>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleNewChat}
            className="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 px-3 py-1.5 rounded-xl font-bold transition"
          >
            ➕ New Chat
          </button>

          <select
            value={subject}
            onChange={(e) => setSubject(e.target.value as Subject)}
            className="bg-slate-800 border border-slate-700 text-xs rounded-xl px-3 py-1.5 font-bold text-slate-200 focus:outline-none"
          >
            <option value="Physics">Physics — Rahul (IIT Bombay)</option>
            <option value="Chemistry">Chemistry — Raj (IIT Delhi)</option>
            <option value="Mathematics">Mathematics — Amit (IIT Kanpur)</option>
          </select>

          <div className="flex bg-slate-800 p-0.5 rounded-xl border border-slate-700">
            <button
              onClick={() => setMode("Socratic Hint")}
              className={`px-3 py-1 text-xs font-bold rounded-lg transition ${
                mode === "Socratic Hint" ? "bg-cyan-500 text-slate-950 shadow" : "text-slate-400 hover:text-white"
              }`}
            >
              Socratic Hint
            </button>
            <button
              onClick={() => setMode("Full Solution")}
              className={`px-3 py-1 text-xs font-bold rounded-lg transition ${
                mode === "Full Solution" ? "bg-cyan-500 text-slate-950 shadow" : "text-slate-400 hover:text-white"
              }`}
            >
              Full Solution
            </button>
          </div>
        </div>
      </header>

      <div className="flex-1 max-w-5xl w-full mx-auto p-4 md:p-6 overflow-y-auto space-y-6">
        {messages.length === 0 && (
          <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 md:p-8 space-y-4 shadow-xl">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-2xl bg-cyan-500/20 text-cyan-400 flex items-center justify-center font-black text-lg border border-cyan-500/40">
                {activeTutor.name[0]}
              </div>
              <div>
                <h2 className="text-base font-black text-white">
                  Master {activeTutor.name}{" "}
                  <span className="text-xs font-normal text-slate-400">({activeTutor.institute})</span>
                </h2>
                <p className="text-xs text-cyan-400 font-mono">{activeTutor.spec}</p>
              </div>
            </div>
            <p className="text-sm text-slate-300 leading-relaxed">
              Hello! I will guide you through rigorous <strong>{subject}</strong> concepts for IIT-JEE Advanced.
              Ask any theoretical doubt, numerical problem, or archival question.
            </p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex flex-col ${msg.role === "user" ? "items-end" : "items-start"} space-y-2`}
          >
            <div
              className={`max-w-[85%] rounded-3xl p-5 ${
                msg.role === "user"
                  ? "bg-cyan-500 text-slate-950 font-medium rounded-br-none shadow-lg shadow-cyan-500/20"
                  : "bg-slate-900 border border-slate-800 text-slate-100 rounded-bl-none shadow-xl"
              }`}
            >
              {msg.role === "assistant" && (
                <div className="text-xs font-mono font-bold text-cyan-400 mb-2">
                  🎓 Master {msg.tutor || activeTutor.name}
                </div>
              )}

              {msg.role === "user" ? (
                <div className="text-sm font-sans">{msg.content}</div>
              ) : (
                <FormattedMessage content={msg.content} />
              )}

              {msg.scene && msg.scene.elements && msg.scene.elements.length > 0 && (
                <SceneRenderer scene={msg.scene} />
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex items-center gap-3 text-xs font-mono text-cyan-400 bg-slate-900/60 border border-slate-800/80 px-4 py-3 rounded-2xl w-fit">
            <span className="animate-spin">⚙️</span> Analyzing question and generating 3D setup...
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      <footer className="border-t border-slate-800 bg-slate-900/80 backdrop-blur p-4 sticky bottom-0 z-30">
        <form onSubmit={handleSend} className="max-w-5xl mx-auto flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={`Ask Master ${activeTutor.name} a ${subject} question...`}
            className="flex-1 bg-slate-950 border border-slate-800 rounded-2xl px-5 py-3.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 font-sans shadow-inner"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="bg-cyan-400 hover:bg-cyan-300 disabled:opacity-50 text-slate-950 px-6 py-3.5 rounded-2xl text-sm font-black transition shadow-lg shadow-cyan-400/20"
          >
            Send
          </button>
        </form>
      </footer>
    </main>
  );
}