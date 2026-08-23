"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

interface StatsData {
  total_queries: number;
  unique_visitors: number;
  today_queries: number;
  scenes_generated: number;
}

interface QueryLog {
  id: number;
  timestamp: string;
  ip_address: string;
  subject: string;
  tutor: string;
  mode: string;
  question: string;
  has_scene: boolean;
}

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export default function AdminPage() {
  const [adminKey, setAdminKey] = useState<string>("jeept2026");
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [inputKey, setInputKey] = useState<string>("");

  const [stats, setStats] = useState<StatsData | null>(null);
  const [logs, setLogs] = useState<QueryLog[]>([]);
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string>("");
  const [search, setSearch] = useState("");
  const [filterSubject, setFilterSubject] = useState("All");

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputKey.trim()) return;
    setAdminKey(inputKey);
    fetchDataWithKey(inputKey);
  };

  const fetchDataWithKey = async (key: string) => {
    setLoading(true);
    setStatusMsg("인증 및 데이터 조회 중...");
    try {
      const headers = { "x-admin-key": key };
      const statsRes = await fetch(`${BACKEND_URL}/api/admin/stats`, { headers });
      const logsRes = await fetch(`${BACKEND_URL}/api/admin/queries?limit=200`, { headers });

      if (statsRes.status === 403 || logsRes.status === 403) {
        setIsAuthenticated(false);
        setStatusMsg("❌ 관리자 인증 암호가 일치하지 않습니다.");
        return;
      }

      if (!statsRes.ok || !logsRes.ok) {
        throw new Error(`HTTP ${statsRes.status}`);
      }

      const statsData = await statsRes.json();
      const logsData = await logsRes.json();

      setStats(statsData);
      setLogs(logsData.logs || []);
      setIsAuthenticated(true);
      setStatusMsg("🔒 관리자 인증 완료: 보안 연결됨");
    } catch (e: any) {
      setIsAuthenticated(false);
      setStatusMsg(`❌ 백엔드 연결 실패: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const createSampleLogs = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/admin/seed-test-log`, {
        method: "POST",
        headers: { "x-admin-key": adminKey }
      });
      if (res.ok) {
        fetchDataWithKey(adminKey);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDataWithKey(adminKey);
  }, []);

  const filteredLogs = logs.filter((log) => {
    const matchesSearch =
      log.question.toLowerCase().includes(search.toLowerCase()) ||
      log.ip_address.includes(search) ||
      log.tutor.toLowerCase().includes(search.toLowerCase());
    const matchesSubject = filterSubject === "All" || log.subject === filterSubject;
    return matchesSearch && matchesSubject;
  });

  if (!isAuthenticated) {
    return (
      <main className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center p-6 font-sans">
        <div className="bg-slate-900 border border-slate-800 p-8 rounded-3xl max-w-md w-full shadow-2xl space-y-6">
          <div className="text-center space-y-2">
            <div className="inline-flex p-3 rounded-2xl bg-cyan-950 text-cyan-400 border border-cyan-800 text-2xl font-black">
              🔒
            </div>
            <h1 className="text-xl font-black text-white">ChatJEEPT Admin Console</h1>
            <p className="text-xs text-slate-400">보안 콘솔 접속을 위해 관리자 암호를 입력해 주세요.</p>
          </div>

          <form onSubmit={handleLogin} className="space-y-4">
            <div>
              <label className="text-xs font-bold text-slate-400 block mb-1.5">Admin Secret Key</label>
              <input
                type="password"
                value={inputKey}
                onChange={(e) => setInputKey(e.target.value)}
                placeholder="기본 암호: jeept2026"
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-cyan-400 font-mono"
              />
            </div>
            {statusMsg && (
              <p className="text-xs text-red-400 font-medium">{statusMsg}</p>
            )}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-cyan-400 hover:bg-cyan-300 text-slate-950 font-black py-2.5 rounded-xl text-sm transition"
            >
              {loading ? "인증 확인 중..." : "인증 및 접속"}
            </button>
          </form>

          <div className="text-center">
            <Link href="/" className="text-xs text-slate-500 hover:text-slate-400">
              ← 메인 챗으로 돌아가기
            </Link>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 p-6 md:p-10 font-sans">
      <div className="max-w-7xl mx-auto space-y-8">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-6">
          <div>
            <div className="flex items-center gap-3">
              <span className="text-2xl font-black text-cyan-400">📊 ChatJEEPT</span>
              <span className="text-xs bg-emerald-950 text-emerald-300 border border-emerald-700 px-2.5 py-0.5 rounded-full font-mono font-bold">
                Protected Console
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Live Real-time Student Question Logs & User Analytics
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={createSampleLogs}
              disabled={loading}
              className="px-3 py-2 bg-indigo-600 hover:bg-indigo-500 border border-indigo-500 text-white rounded-xl text-xs font-bold transition shadow-lg shadow-indigo-600/30"
            >
              ➕ 테스트 샘플 로그 추가
            </button>
            <button
              onClick={() => fetchDataWithKey(adminKey)}
              disabled={loading}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-xl text-xs font-bold text-slate-200 transition"
            >
              🔄 새로고침
            </button>
            <Link
              href="/"
              className="px-4 py-2 bg-cyan-400 hover:bg-cyan-300 rounded-xl text-xs font-black text-slate-950 transition"
            >
              ← 메인 챗으로 가기
            </Link>
          </div>
        </div>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl">
            <div className="text-xs font-bold uppercase tracking-wider text-slate-400">Total Questions</div>
            <div className="text-3xl font-black text-white mt-2">{stats?.total_queries ?? 0}</div>
            <div className="text-[11px] text-slate-500 mt-1">전체 누적 질문 수</div>
          </div>

          <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl">
            <div className="text-xs font-bold uppercase tracking-wider text-cyan-400">Unique Users</div>
            <div className="text-3xl font-black text-cyan-400 mt-2">{stats?.unique_visitors ?? 0}</div>
            <div className="text-[11px] text-slate-500 mt-1">방문자 수 (고유 IP)</div>
          </div>

          <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl">
            <div className="text-xs font-bold uppercase tracking-wider text-emerald-400">Today's Queries</div>
            <div className="text-3xl font-black text-emerald-400 mt-2">{stats?.today_queries ?? 0}</div>
            <div className="text-[11px] text-slate-500 mt-1">오늘 들어온 질문 수</div>
          </div>

          <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl">
            <div className="text-xs font-bold uppercase tracking-wider text-indigo-400">3D Scenes Rendered</div>
            <div className="text-3xl font-black text-indigo-400 mt-2">{stats?.scenes_generated ?? 0}</div>
            <div className="text-[11px] text-slate-500 mt-1">3D 시각화 생성 횟수</div>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
          <div className="p-4 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3 bg-slate-950/40">
            <h2 className="text-sm font-bold text-white flex items-center gap-2">
              <span>💬</span> 실시간 유저 질문 기록표 ({filteredLogs.length}건)
            </h2>

            <div className="flex items-center gap-2">
              <select
                value={filterSubject}
                onChange={(e) => setFilterSubject(e.target.value)}
                className="bg-slate-800 border border-slate-700 text-xs rounded-xl px-3 py-1.5 font-semibold text-slate-200 focus:outline-none"
              >
                <option value="All">전체 과목</option>
                <option value="Physics">Physics</option>
                <option value="Chemistry">Chemistry</option>
                <option value="Mathematics">Mathematics</option>
              </select>

              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="질문 / IP / 튜터 검색..."
                className="bg-slate-800 border border-slate-700 text-xs rounded-xl px-3 py-1.5 text-white placeholder-slate-500 focus:outline-none w-60"
              />
            </div>
          </div>

          <div className="overflow-x-auto max-h-[500px]">
            <table className="w-full text-left text-xs border-collapse">
              <thead className="bg-slate-950 text-slate-400 uppercase tracking-wider font-bold border-b border-slate-800 sticky top-0">
                <tr>
                  <th className="py-3.5 px-4">시간</th>
                  <th className="py-3.5 px-4">접속 IP</th>
                  <th className="py-3.5 px-4">과목 & 튜터</th>
                  <th className="py-3.5 px-4">학습 모드</th>
                  <th className="py-3.5 px-4">학생 질문 내용</th>
                  <th className="py-3.5 px-4 text-center">3D 시각화</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {filteredLogs.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="py-12 text-center text-slate-500 font-sans">
                      <p className="text-sm font-bold text-slate-400 mb-1">기록된 로그가 없습니다.</p>
                      <p className="text-xs text-slate-600">상단의 '+ 테스트 샘플 로그 추가' 버튼을 눌러보세요.</p>
                    </td>
                  </tr>
                ) : (
                  filteredLogs.map((log) => (
                    <tr key={log.id} className="hover:bg-slate-800/50 transition">
                      <td className="py-3.5 px-4 text-slate-400 whitespace-nowrap">{log.timestamp}</td>
                      <td className="py-3.5 px-4 text-cyan-300 font-mono whitespace-nowrap">{log.ip_address}</td>
                      <td className="py-3.5 px-4 whitespace-nowrap">
                        <span
                          className={`font-bold ${
                            log.subject === "Physics"
                              ? "text-blue-400"
                              : log.subject === "Chemistry"
                              ? "text-emerald-400"
                              : "text-violet-400"
                          }`}
                        >
                          {log.subject}
                        </span>{" "}
                        <span className="text-slate-500">({log.tutor})</span>
                      </td>
                      <td className="py-3.5 px-4 whitespace-nowrap">
                        <span className="bg-slate-800 px-2 py-0.5 rounded text-[11px] text-slate-300 border border-slate-700">
                          {log.mode}
                        </span>
                      </td>
                      <td className="py-3.5 px-4 font-sans text-slate-100 max-w-md break-words font-medium">
                        {log.question}
                      </td>
                      <td className="py-3.5 px-4 text-center">
                        {log.has_scene ? (
                          <span className="text-[11px] bg-indigo-950 text-indigo-300 border border-indigo-700 px-2 py-0.5 rounded-full font-bold">
                            🎮 3D 생성됨
                          </span>
                        ) : (
                          <span className="text-slate-600">-</span>
                        )}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </main>
  );
}