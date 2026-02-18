import { useState, useEffect, useRef } from 'react';
import {
  Terminal as TerminalIcon, Zap, Search, Trash2,
  Save, FileText, History, RefreshCw, Paperclip, File,
  Edit, X, Copy, XCircle, Globe, Network, RotateCcw,
  Activity, GitBranch, Layout, ShieldCheck
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { toast } from 'sonner';
import ReactMarkdown from 'react-markdown';
import ForceGraph2D from 'react-force-graph-2d';

import { Toaster } from 'sonner';

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Sidebar } from "@/components/layout/sidebar";

// --- Types ---
interface HardwareStats {
  cpu_percent: number;
  memory_percent: number;
  memory_used_gb: number;
  memory_total_gb: number;
  disk_free_gb: number;
}

interface PlatformStatus {
  is_busy: boolean;
  current_task: string;
  progress: number;
  last_report: string | null;
  logs: string[];
}

interface GraphData {
  nodes: { id: string, label: string, type: string }[];
  links: { source: string, target: string, label: string }[];
}

interface HypothesisData {
  [key: string]: {
    content: string;
    confidence: number;
    status: string;
    evidence_count: number;
    created_at: string;
  };
}

interface ConsensusData {
  agreement_score: number;
  disputed_points: string[];
  consensus_at: string | null;
}

interface ThoughtBlock {
  agent: string;
  content: string;
}

export default function App() {
  const [topic, setTopic] = useState('');
  const [tier, setTier] = useState<'deep_apex' | 'ultra_light_fast'>('deep_apex');
  const [stats, setStats] = useState<HardwareStats | null>(null);
  const [status, setStatus] = useState<PlatformStatus | null>(null);
  const [reports, setReports] = useState<string[]>([]);
  const [viewingReport, setViewingReport] = useState<string | null>(null);
  const [reportContent, setReportContent] = useState<string>('');
  const [isEditing, setIsEditing] = useState(false);
  const [researchContext, setResearchContext] = useState<string>('');
  const [activeTab, setActiveTab] = useState<'feed' | 'graph' | 'hypotheses' | 'terminal'>('feed');
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
  const [hypotheses, setHypotheses] = useState<HypothesisData>({});
  const [consensusData, setConsensusData] = useState<ConsensusData>({ agreement_score: 0, disputed_points: [], consensus_at: null });
  // Set to null implies pending
  useEffect(() => {
    setConsensusData(prev => ({ ...prev, agreement_score: null as any }));
  }, []);
  const [thoughts, setThoughts] = useState<{ [key: string]: ThoughtBlock[] }>({});
  const [currentThoughtAgent, setCurrentThoughtAgent] = useState<string | null>(null);
  const [terminalOutput, setTerminalOutput] = useState<string[]>([]);
  const [activeModel, setActiveModel] = useState<string>('IDLE');
  const [sourceToggles, setSourceToggles] = useState({
    github: true,
    huggingface: true,
    arxiv: true,
    biorxiv: true,
    pubmed: true,
    web: true,
    linkedin: true,
    huggingface_daily: true
  });
  const [uploadedFiles, setUploadedFiles] = useState<{ name: string; path: string }[]>([]); // New State
  const logEndRef = useRef<HTMLDivElement>(null);
  const termEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const API_BASE = "http://localhost:8000/api";

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const hRes = await fetch(`${API_BASE}/hardware`);
        setStats(await hRes.json());
        const sRes = await fetch(`${API_BASE}/status`);
        const statusData = await sRes.json();
        setStatus(statusData);

        // Fetch granular events
        const eRes = await fetch(`${API_BASE}/events`);
        const events: string[] = await eRes.json();
        events.forEach(ev => {
          if (ev.startsWith("MODEL_ACTIVE:")) {
            setActiveModel(ev.split(":")[1]);
          } else if (ev.startsWith("SANDBOX_RESULT:")) {
            const parts = ev.split(":");
            const status = parts[1];
            const output = parts.slice(2).join(":");
            setTerminalOutput(prev => [...prev, `[SYSTEM] Sandbox Result: ${status}`, ...(output ? [output] : [])]);
          } else if (ev === "SANDBOX_START") {
            setTerminalOutput(prev => [...prev, `[SYSTEM] Initializing Sandbox...`]);
          } else if (ev.startsWith("THOUGHT_START:")) {
            setCurrentThoughtAgent(ev.split(":")[1]);
          } else if (ev.startsWith("THOUGHT:")) {
            const content = ev.split(":").slice(1).join(":");
            if (currentThoughtAgent) {
              setThoughts(prev => {
                const lastLog = statusData.logs?.[statusData.logs.length - 1] || "Process";
                const stepThoughts = [...(prev[lastLog] || [])];
                if (stepThoughts.length > 0 && stepThoughts[stepThoughts.length - 1].agent === currentThoughtAgent) {
                  stepThoughts[stepThoughts.length - 1].content += content;
                } else {
                  stepThoughts.push({ agent: currentThoughtAgent, content });
                }
                return { ...prev, [lastLog]: stepThoughts };
              });
            }
          } else if (ev.startsWith("THOUGHT_END:")) {
            setCurrentThoughtAgent(null);
          }
        });

        // Periodic deep fetches if busy
        if (statusData.is_busy) {
          const gRes = await fetch(`${API_BASE}/graph`);
          setGraphData(await gRes.json());
          const hyRes = await fetch(`${API_BASE}/hypotheses`);
          setHypotheses(await hyRes.json());
          const cRes = await fetch(`${API_BASE}/consensus`);
          setConsensusData(await cRes.json());
        }
      } catch (err) { }
    }, 1500);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (status?.is_busy) {
      logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [status?.logs]);

  useEffect(() => {
    const fetchReports = async () => {
      try {
        const res = await fetch(`${API_BASE}/reports`);
        setReports(await res.json());
      } catch (err) { }
    };
    fetchReports();
    const interval = setInterval(fetchReports, 15000); // 15 seconds
    return () => clearInterval(interval);
  }, []);

  const refreshReports = async () => {
    try {
      const res = await fetch(`${API_BASE}/reports`);
      setReports(await res.json());
    } catch (err) { }
  };

  const start_investigation = async () => {
    if (!topic || status?.is_busy) return;
    try {
      await fetch(`${API_BASE}/investigate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic,
          tier,
          context: researchContext,
          file_paths: uploadedFiles.map(f => f.path)
        })
      });
      setTopic('');
      setResearchContext('');
      setIsEditing(false);
    } catch (err) { }
  };

  const stopInvestigation = async () => {
    try {
      await fetch(`${API_BASE}/stop`, { method: 'POST' });
    } catch (err) { }
  };

  const deleteReport = async (filename: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      setReports(reports.filter(r => r !== filename));
      await fetch(`${API_BASE}/report/${filename}`, { method: 'DELETE' });
      if (viewingReport === filename) setViewingReport(null);
      setTimeout(refreshReports, 500);
    } catch (err) {
      refreshReports();
    }
  };

  const saveReport = async () => {
    if (!viewingReport) return;
    try {
      await fetch(`${API_BASE}/report/${viewingReport}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: reportContent })
      });
      setIsEditing(false);
      toast.success("Report saved successfully!");
    } catch (err) {
      toast.error("Failed to save report.");
    }
  };

  const openReport = async (filename: string) => {
    try {
      setViewingReport(filename);
      const res = await fetch(`${API_BASE}/report/${filename}`);
      const data = await res.json();
      setReportContent(data.content);
      setIsEditing(false);
    } catch (err) { }
  };

  const clearLogs = async () => {
    try {
      await fetch(`${API_BASE}/logs`, { method: 'DELETE' });
      setStatus(s => s ? { ...s, logs: [] } : s);
      toast.success("Logs cleared.");
    } catch (err) { }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success("Copied to clipboard!");
  };

  const onFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    const files = Array.from(e.target.files);

    for (const file of files) {
      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await fetch(`${API_BASE}/upload`, {
          method: "POST",
          body: formData,
        });
        const data = await res.json();
        setUploadedFiles(prev => [...prev, { name: data.filename, path: data.path }]);
        toast.success(`Attached: ${data.filename}`);
      } catch (err) {
        toast.error(`Failed to upload ${file.name}`);
      }
    }
  };

  const removeFile = (path: string) => {
    setUploadedFiles(prev => prev.filter(f => f.path !== path));
  };

  return (
    <div className="flex h-screen w-screen bg-background overflow-hidden text-foreground">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@100..900&family=JetBrains+Mono:wght@100..800&display=swap');

        body {
          font-family: 'Outfit', sans-serif;
        }

        .glass-card {
          background: rgba(255, 255, 255, 0.03);
          backdrop-filter: blur(12px);
          border: 1px solid rgba(255, 255, 255, 0.08);
          box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        .neo-gradient-header {
          background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        }

        .vibrant-glow {
          box-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
        }

        details {
          background: rgba(255, 255, 255, 0.02);
          backdrop-filter: blur(4px);
          border: 1px solid rgba(255, 255, 255, 0.05);
          border-left: 4px solid #6366f1;
          border-radius: 12px;
          margin: 1.5rem 0;
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }

        details[open] {
          background: rgba(255, 255, 255, 0.04);
          transform: translateY(-2px);
          box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        }

        summary {
          padding: 1.25rem;
          font-weight: 800;
          font-size: 0.75rem;
          text-transform: uppercase;
          letter-spacing: 0.15em;
          color: #94a3b8;
          cursor: pointer;
          list-style: none;
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        summary::before {
          content: '⚡';
          filter: grayscale(1);
          transition: all 0.3s;
        }

        details[open] summary::before {
          filter: grayscale(0);
          transform: rotate(180deg) scale(1.2);
        }

        .prose code {
          font-family: 'JetBrains Mono', monospace;
          background: rgba(99, 102, 241, 0.1);
          color: #818cf8;
          padding: 0.2em 0.4em;
          border-radius: 4px;
        }

        .source-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
          gap: 0.75rem;
        }

        .source-pill {
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.1);
          padding: 0.5rem;
          border-radius: 8px;
          font-size: 10px;
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
          transition: all 0.2s;
        }

        .source-pill:hover {
          background: rgba(99, 102, 241, 0.1);
          border-color: rgba(99, 102, 241, 0.3);
        }
      `}</style>
      <Toaster position="top-right" expand={false} richColors theme="dark" />
      <Sidebar stats={stats} status={status} />

      <main className="flex-1 flex flex-col min-w-0 bg-[#020617]">
        <header className="h-24 neo-gradient-header border-b border-white/5 flex items-center px-10 justify-between shrink-0 z-20">
          <div className="flex items-center gap-8 flex-1 max-w-6xl">
            <div className="flex bg-white/5 backdrop-blur-md p-1.5 rounded-2xl border border-white/10 shrink-0">
              <button
                onClick={() => setTier('deep_apex')}
                className={`flex items-center gap-2.5 px-5 py-2 rounded-xl text-[11px] font-black uppercase tracking-[0.1em] transition-all duration-300 ${tier === 'deep_apex'
                  ? 'bg-gradient-to-br from-indigo-500 to-purple-600 text-white shadow-[0_0_20px_rgba(99,102,241,0.4)] scale-105'
                  : 'text-slate-400 hover:text-white hover:bg-white/5'
                  }`}
              >
                <Zap className={`h-3.5 w-3.5 ${tier === 'deep_apex' ? 'fill-current' : ''}`} />
                Deep Apex
              </button>
              <button
                onClick={() => setTier('ultra_light_fast')}
                className={`flex items-center gap-2.5 px-5 py-2 rounded-xl text-[11px] font-black uppercase tracking-[0.1em] transition-all duration-300 ${tier === 'ultra_light_fast'
                  ? 'bg-gradient-to-br from-emerald-500 to-teal-600 text-white shadow-[0_0_20px_rgba(16,185,129,0.4)] scale-105'
                  : 'text-slate-400 hover:text-white hover:bg-white/5'
                  }`}
              >
                <Zap className={`h-3.5 w-3.5 ${tier === 'ultra_light_fast' ? 'fill-current' : ''}`} />
                Turbo Link
              </button>
            </div>

            <div className="flex flex-col flex-1 max-w-2xl gap-2">
              <div className="relative group">
                <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-2xl blur opacity-20 group-hover:opacity-40 transition duration-1000 group-hover:duration-200"></div>
                <div className="relative flex items-center bg-[#1e293b]/80 backdrop-blur-xl border border-white/10 rounded-xl px-4 h-12">
                  <Search className="h-4 w-4 text-indigo-400 mr-3" />
                  <input
                    value={topic}
                    onChange={(e) => setTopic(e.target.value)}
                    disabled={status?.is_busy}
                    placeholder="Initiate Universal Knowledge Synthesis..."
                    className="bg-transparent border-none outline-none text-white w-full text-sm font-medium placeholder:text-slate-500"
                    onKeyDown={(e) => e.key === 'Enter' && start_investigation()}
                  />
                </div>
              </div>
              <div className="flex gap-2 px-2">
                {Object.entries(sourceToggles).map(([key, val]) => (
                  <button
                    key={key}
                    onClick={() => setSourceToggles(prev => ({ ...prev, [key]: !val }))}
                    className={`px-2 py-1 rounded-md text-[9px] font-black uppercase tracking-tighter border transition-all ${val
                      ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300'
                      : 'bg-white/5 border-white/10 text-slate-500'
                      }`}
                  >
                    {key}
                  </button>
                ))}
              </div>

              {/* Attachments */}
              <div className="flex flex-wrap gap-2 px-2">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="flex items-center gap-2 px-3 py-1 bg-white/5 hover:bg-white/10 border border-white/10 rounded-md cursor-pointer transition-all group"
                >
                  <Paperclip className="h-3 w-3 text-slate-400 group-hover:text-white" />
                  <span className="text-[10px] font-bold text-slate-400 group-hover:text-white uppercase tracking-wider">Attach</span>
                </button>
                <input
                  type="file"
                  multiple
                  className="hidden"
                  ref={fileInputRef}
                  onChange={onFileUpload}
                />
                {uploadedFiles.map((file, i) => (
                  <div key={i} className="flex items-center gap-2 px-2 py-1 bg-indigo-500/20 border border-indigo-500/30 rounded-md">
                    <File className="h-3 w-3 text-indigo-400" />
                    <span className="text-[10px] text-indigo-200 truncate max-w-[100px]">{file.name}</span>
                    <X className="h-3 w-3 text-indigo-400/50 hover:text-white cursor-pointer" onClick={() => removeFile(file.path)} />
                  </div>
                ))}
              </div>
            </div>

            {status?.is_busy ? (
              <Button onClick={stopInvestigation} variant="destructive" className="px-6 h-12 font-black uppercase tracking-widest text-[10px] gap-2 vibrant-glow">
                <XCircle className="h-4 w-4" /> Abort
              </Button>
            ) : (
              <Button
                onClick={start_investigation}
                disabled={!topic}
                className="px-8 h-12 font-black uppercase tracking-widest text-[10px] bg-indigo-600 hover:bg-indigo-500 text-white border-b-4 border-indigo-800 active:border-b-0 transition-all vibrant-glow"
              >
                Execute Strategy
              </Button>
            )}
          </div>

          <div className="flex items-center gap-10">
            <div className="flex items-center gap-6 px-4 py-2 bg-white/5 rounded-2xl border border-white/10 backdrop-blur-xl">
              <div className="flex flex-col items-end">
                <span className="text-[9px] font-black text-indigo-400 uppercase tracking-widest">Ensemble Consensus</span>
                <span className="text-sm font-black text-white italic">
                  {consensusData.agreement_score !== null && consensusData.consensus_at ?
                    `${(consensusData.agreement_score * 100).toFixed(0)}%` :
                    "PENDING"}
                  <span className="text-[10px] text-slate-500 not-italic ml-1">AGREEMENT</span>
                </span>
              </div>
              <div className="h-8 w-px bg-white/10" />
              <div className="flex flex-col">
                <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">Active Model</span>
                <div className="flex items-center gap-2">
                  <Activity className="h-3 w-3 text-emerald-400 animate-pulse" />
                  <span className="text-sm font-black text-white truncate max-w-[120px]">{activeModel}</span>
                </div>
              </div>
            </div>
            <div className="flex flex-col items-end">
              <span className="text-[10px] font-black text-indigo-400 uppercase tracking-widest">Edison OS</span>
              <span className="text-sm font-bold text-white">v4.0 Ultra</span>
            </div>
          </div>
        </header>

        <nav className="h-14 bg-[#0f172a] border-b border-white/5 flex items-center px-10 gap-8 shrink-0">
          <button onClick={() => setActiveTab('feed')} className={`h-full flex items-center gap-2 px-2 border-b-2 transition-all text-xs font-bold uppercase tracking-widest ${activeTab === 'feed' ? 'border-indigo-500 text-indigo-400' : 'border-transparent text-slate-500 hover:text-slate-300'}`}>
            <Layout className="h-4 w-4" /> Strategic Feed
          </button>
          <button onClick={() => setActiveTab('graph')} className={`h-full flex items-center gap-2 px-2 border-b-2 transition-all text-xs font-bold uppercase tracking-widest ${activeTab === 'graph' ? 'border-indigo-500 text-indigo-400' : 'border-transparent text-slate-500 hover:text-slate-300'}`}>
            <Network className="h-4 w-4" /> Knowledge Graph
          </button>
          <button onClick={() => setActiveTab('hypotheses')} className={`h-full flex items-center gap-2 px-2 border-b-2 transition-all text-xs font-bold uppercase tracking-widest ${activeTab === 'hypotheses' ? 'border-indigo-500 text-indigo-400' : 'border-transparent text-slate-500 hover:text-slate-300'}`}>
            <GitBranch className="h-4 w-4" /> Hypotheses
          </button>
          <button onClick={() => setActiveTab('terminal')} className={`h-full flex items-center gap-2 px-2 border-b-2 transition-all text-xs font-bold uppercase tracking-widest ${activeTab === 'terminal' ? 'border-indigo-500 text-indigo-400' : 'border-transparent text-slate-500 hover:text-slate-300'}`}>
            <TerminalIcon className="h-4 w-4" /> Live Sandbox
          </button>
        </nav>

        <div className="flex-1 overflow-hidden relative">
          <AnimatePresence mode="wait">
            {activeTab === 'feed' && (
              <motion.div
                key="feed"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="h-full"
              >
                <ScrollArea className="h-full p-10 bg-[radial-gradient(circle_at_top_right,#1e1b4b,transparent),radial-gradient(circle_at_bottom_left,#0f172a,transparent)]">
                  <div className="max-w-7xl mx-auto space-y-10 pb-20">
                    <div className="grid grid-cols-1 lg:grid-cols-4 gap-10">
                      <div className="lg:col-span-3 space-y-8">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                          <div className="glass-card rounded-2xl p-6 border-indigo-500/20">
                            <div className="flex items-center justify-between mb-4">
                              <div className="p-3 bg-indigo-500/10 rounded-xl"><Network className="h-5 w-5 text-indigo-400" /></div>
                              <Badge variant="outline" className="border-indigo-500/30 text-indigo-300 text-[9px]">ACTIVE</Badge>
                            </div>
                            <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Source Convergence</h4>
                            <p className="text-xl font-bold text-white">Unified Bridge</p>
                          </div>
                          <div className="glass-card rounded-2xl p-6 border-purple-500/20">
                            <div className="flex items-center justify-between mb-4">
                              <div className="p-3 bg-purple-500/10 rounded-xl"><ShieldCheck className="h-5 w-5 text-purple-400" /></div>
                              <Badge variant="outline" className="border-purple-500/30 text-purple-300 text-[9px]">SOTA</Badge>
                            </div>
                            <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Validation Engine</h4>
                            <p className="text-xl font-bold text-white">Logic Auditor</p>
                          </div>
                          <div className="glass-card rounded-2xl p-6 border-emerald-500/20">
                            <div className="flex items-center justify-between mb-4">
                              <div className="p-3 bg-emerald-500/10 rounded-xl"><TerminalIcon className="h-5 w-5 text-emerald-400" /></div>
                              <Badge variant="outline" className="border-emerald-500/30 text-emerald-300 text-[9px]">LIVE</Badge>
                            </div>
                            <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Code Integrity</h4>
                            <p className="text-xl font-bold text-white">Secure Sandbox</p>
                          </div>
                        </div>

                        <Card className="glass-card border-none rounded-3xl overflow-hidden shadow-2xl">
                          <CardHeader className="border-b border-white/5 bg-white/5 py-5 px-8 flex flex-row items-center justify-between">
                            <div className="flex items-center gap-4">
                              <div className="h-2 w-2 rounded-full bg-indigo-500 animate-pulse" />
                              <CardTitle className="text-xs font-black uppercase tracking-[0.25em] text-slate-400">
                                Strategic Execution Feed
                              </CardTitle>
                            </div>
                            <div className="flex gap-3">
                              <Button variant="ghost" size="sm" onClick={clearLogs} className="h-8 text-[9px] uppercase font-black text-slate-500 hover:text-white">
                                Wipe Buffer
                              </Button>
                            </div>
                          </CardHeader>
                          <CardContent className="h-[600px] p-0 flex flex-col font-mono">
                            <ScrollArea className="flex-1 p-8">
                              <div className="space-y-5">
                                {!status?.logs || status.logs.length === 0 ? (
                                  <div className="h-[400px] flex flex-col items-center justify-center text-slate-500/20 gap-6">
                                    <Globe className="h-20 w-20 stroke-[0.5]" />
                                    <p className="text-[10px] font-black uppercase tracking-[0.4em] text-center max-w-sm leading-relaxed">System Idle. Awaiting Research Directives via Unified Source Bridge.</p>
                                  </div>
                                ) : (
                                  status.logs.map((log, i) => (
                                    <motion.div
                                      initial={{ opacity: 0, y: 10 }}
                                      animate={{ opacity: 1, y: 0 }}
                                      key={i}
                                      className="group flex flex-col gap-2 border-b border-white/5 pb-4 last:border-0"
                                    >
                                      <div className="flex gap-5 text-[13px]">
                                        <span className="text-indigo-500/50 font-black min-w-[2.5rem] tracking-tighter">PHASE_{i.toString().padStart(2, '0')}</span>
                                        <span className="text-slate-200 font-medium leading-relaxed tracking-wide">{log}</span>
                                      </div>

                                      {thoughts[log] && thoughts[log].map((t, ti) => (
                                        <details key={ti} className="ml-14 mt-2 mb-2 group/thought">
                                          <summary className="text-[10px] font-black text-indigo-400/60 uppercase tracking-widest cursor-pointer list-none hover:text-indigo-400 transition-colors">
                                            Expand Thinking Process ({t.agent})
                                          </summary>
                                          <div className="mt-4 p-4 bg-white/5 rounded-xl border border-white/5 text-[12px] text-slate-400 leading-relaxed font-sans italic">
                                            <div className="flex items-center gap-2 mb-2 text-indigo-300/80 not-italic">
                                              <Activity className="h-3 w-3" />
                                              <span className="font-black uppercase tracking-tighter">{t.agent} Rationale</span>
                                            </div>
                                            {t.content}
                                          </div>
                                        </details>
                                      ))}
                                    </motion.div>
                                  ))
                                )}
                                <div ref={logEndRef} />
                              </div>
                            </ScrollArea>
                            {status?.is_busy && (
                              <div className="p-10 bg-[#1e293b]/50 border-t border-white/10 backdrop-blur-3xl">
                                <div className="flex justify-between items-end mb-4">
                                  <div className="space-y-1">
                                    <p className="text-[10px] font-black text-indigo-400 uppercase tracking-widest">Active Protocol</p>
                                    <h3 className="text-xl font-bold text-white tracking-tight">{status.current_task}</h3>
                                  </div>
                                  <div className="text-right">
                                    <span className="text-3xl font-black text-white italic">{status.progress}<span className="text-indigo-500 text-lg ml-1">%</span></span>
                                  </div>
                                </div>
                                <div className="relative h-2.5 w-full bg-slate-800 rounded-full overflow-hidden border border-white/5 shadow-inner">
                                  <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${status.progress}%` }}
                                    className="absolute top-0 left-0 h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-indigo-500 vibrant-glow"
                                  />
                                </div>
                              </div>
                            )}
                          </CardContent>
                        </Card>
                      </div>

                      <div className="space-y-10">
                        <Card className="glass-card border-none rounded-3xl overflow-hidden h-full">
                          <CardHeader className="border-b border-white/5 bg-white/5 py-5">
                            <CardTitle className="text-xs font-black uppercase tracking-[0.2em] text-slate-400 flex items-center justify-between">
                              Discovery Vault
                              <RefreshCw onClick={refreshReports} className="h-3 w-3 cursor-pointer hover:rotate-180 transition-transform duration-500" />
                            </CardTitle>
                          </CardHeader>
                          <CardContent className="p-4 overflow-y-auto max-h-[700px]">
                            <div className="space-y-3">
                              {reports.map((report) => (
                                <div
                                  key={report}
                                  onClick={() => openReport(report)}
                                  className="w-full p-5 rounded-2xl bg-white/5 border border-white/5 hover:border-indigo-500/50 hover:bg-indigo-500/5 text-left transition-all duration-300 group relative overflow-hidden cursor-pointer"
                                >
                                  <div className="relative z-10 flex flex-col gap-1">
                                    <div className="text-[10px] text-indigo-400 font-black uppercase tracking-widest flex items-center gap-2">
                                      <History className="h-3 w-3" /> Synthesis Fragment
                                    </div>
                                    <div className="text-sm font-bold text-white group-hover:text-indigo-300 transition-colors truncate">
                                      {report.replace('.md', '').split('_').slice(2).join(' ')}
                                    </div>
                                  </div>
                                  <div className="absolute right-4 top-1/2 -translate-y-1/2 flex gap-2 opacity-0 group-hover:opacity-100 transition-all z-20">
                                    <button onClick={(e) => deleteReport(report, e)} className="p-2 bg-red-500/20 text-red-400 hover:bg-red-500 hover:text-white rounded-lg transition-all"><Trash2 className="h-3.5 w-3.5" /></button>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      </div>
                    </div>
                  </div>
                </ScrollArea>
              </motion.div>
            )}

            {activeTab === 'graph' && (
              <motion.div
                key="graph"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 1.05 }}
                className="h-full relative bg-[#020617]"
              >
                <div className="absolute top-10 left-10 z-10 space-y-2 pointer-events-none">
                  <h2 className="text-3xl font-black text-white tracking-tighter">Semantic Space</h2>
                  <p className="text-xs text-indigo-400/80 font-bold uppercase tracking-widest">Real-time Knowledge Connectivity</p>
                </div>
                <ForceGraph2D
                  graphData={graphData}
                  nodeAutoColorBy="type"
                  nodeLabel="label"
                  linkColor={() => 'rgba(99, 102, 241, 0.2)'}
                  nodeCanvasObject={(node: any, ctx, globalScale) => {
                    const label = node.label;
                    const fontSize = 12 / globalScale;
                    ctx.font = `${fontSize}px Outfit`;
                    const textWidth = ctx.measureText(label).width;

                    ctx.fillStyle = node.color || 'rgba(99, 102, 241, 0.8)';
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, 5 / globalScale, 0, 2 * Math.PI, false);
                    ctx.fill();

                    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
                    ctx.fillText(label, node.x - textWidth / 2, node.y + 10 / globalScale);
                  }}
                  backgroundColor="#020617"
                />
              </motion.div>
            )}

            {activeTab === 'hypotheses' && (
              <motion.div
                key="hypotheses"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="h-full p-10 overflow-y-auto bg-[#020617]"
              >
                <div className="max-w-4xl mx-auto space-y-12 pb-20">
                  <div className="space-y-2">
                    <h2 className="text-4xl font-black text-white tracking-tighter">Research Git</h2>
                    <p className="text-sm text-slate-500 font-medium leading-relaxed">Evolutionary history of core scientific hypotheses and evidence convergence.</p>
                  </div>
                  <div className="relative border-l-2 border-indigo-500/20 pl-10 space-y-10">
                    {Object.entries(hypotheses).map(([id, h], i) => (
                      <motion.div
                        key={id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.1 }}
                        className="relative"
                      >
                        <div className="absolute -left-[49px] top-2 h-4 w-4 rounded-full bg-indigo-500 border-4 border-[#020617] vibrant-glow" />
                        <Card className="glass-card border-white/5 rounded-2xl p-6 hover:border-indigo-500/30 transition-all">
                          <div className="flex justify-between items-start mb-4">
                            <Badge className={`${h.status === 'verified' ? 'bg-emerald-500/20 text-emerald-400' : h.status === 'rejected' ? 'bg-red-500/20 text-red-400' : 'bg-amber-500/20 text-amber-400'} border-none`}>
                              {h.status.toUpperCase()}
                            </Badge>
                            <span className="text-[10px] font-mono text-slate-500">{h.created_at}</span>
                          </div>
                          <h3 className="text-xl font-bold text-white mb-2">{h.content}</h3>
                          <div className="flex items-center gap-6 mt-6">
                            <div className="flex flex-col">
                              <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">Confidence</span>
                              <span className="text-lg font-black text-indigo-400">{(h.confidence * 100).toFixed(0)}%</span>
                            </div>
                            <div className="flex flex-col">
                              <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">Evidence Blocks</span>
                              <span className="text-lg font-black text-white">{h.evidence_count}</span>
                            </div>
                          </div>
                        </Card>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}

            {activeTab === 'terminal' && (
              <motion.div
                key="terminal"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="h-full p-10 flex flex-col bg-black"
              >
                <div className="flex items-center justify-between mb-6">
                  <div className="flex items-center gap-3">
                    <div className="h-3 w-3 rounded-full bg-red-500" />
                    <div className="h-3 w-3 rounded-full bg-amber-500" />
                    <div className="h-3 w-3 rounded-full bg-emerald-500" />
                    <span className="ml-4 text-xs font-mono text-emerald-500/80 tracking-widest">LIVE_SANDBOX@EDISON-V4.0_ULTRA</span>
                  </div>
                  <Badge variant="outline" className="border-emerald-500/30 text-emerald-400 text-[9px] uppercase font-black">Secure Kernel Active</Badge>
                </div>
                <ScrollArea className="flex-1 bg-black p-6 rounded-2xl border border-white/5 font-mono text-[13px] leading-relaxed">
                  <div className="space-y-2">
                    {terminalOutput.map((line, i) => (
                      <div key={i} className={`${line.includes('[SYSTEM]') ? 'text-indigo-400 font-bold' : 'text-slate-300'}`}>
                        <span className="text-slate-600 mr-3 opacity-50">[{new Date().toLocaleTimeString()}]</span>
                        {line}
                      </div>
                    ))}
                    {status?.is_busy && activeTab === 'terminal' && (
                      <motion.div
                        animate={{ opacity: [1, 0, 1] }}
                        transition={{ duration: 1, repeat: Infinity }}
                        className="inline-block h-4 w-2 bg-emerald-500 ml-1"
                      />
                    )}
                    <div ref={termEndRef} />
                  </div>
                </ScrollArea>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>

      <AnimatePresence>
        {viewingReport && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-12 bg-[#020617]/90 backdrop-blur-xl"
          >
            <motion.div
              initial={{ scale: 0.95, y: 20, opacity: 0 }} animate={{ scale: 1, y: 0, opacity: 1 }} exit={{ scale: 0.95, y: 20, opacity: 0 }}
              className="w-full max-w-6xl h-full bg-[#0f172a] shadow-[0_0_50px_rgba(0,0,0,0.5)] rounded-[2.5rem] border border-white/10 flex flex-col overflow-hidden"
            >
              <div className="p-8 border-b border-white/5 flex items-center justify-between bg-white/[0.02]">
                <div className="flex items-center gap-6">
                  <div className="h-14 w-14 rounded-2xl bg-indigo-500/10 text-indigo-400 flex items-center justify-center border border-indigo-500/20 shadow-inner">
                    <FileText className="h-8 w-8" />
                  </div>
                  <div>
                    <h3 className="text-2xl font-black text-white tracking-tight">Intelligence Synthesis</h3>
                    <div className="flex items-center gap-3 mt-1">
                      <Badge className="bg-indigo-500/20 text-indigo-300 border-none px-2 py-0 text-[9px] font-black">v4.0_CORE</Badge>
                      <span className="text-[10px] uppercase font-bold text-slate-500 tracking-[0.2em]">{viewingReport}</span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <Button variant="outline" className="border-white/10 text-slate-300 hover:bg-white/5 rounded-xl h-11 px-6 uppercase text-[10px] font-black tracking-widest" onClick={() => { setResearchContext(reportContent); setTopic(`Analyze findings from: ${viewingReport}`); setViewingReport(null); toast.info("Report context added to synthesis loop."); }}>
                    <RotateCcw className="h-4 w-4 mr-2" /> Inject Context
                  </Button>
                  <Button variant="outline" className="border-white/10 text-slate-300 hover:bg-white/5 rounded-xl h-11 px-6 uppercase text-[10px] font-black tracking-widest" onClick={() => copyToClipboard(reportContent)}>
                    <Copy className="h-4 w-4 mr-2" /> Copy Intel
                  </Button>
                  <Button className="bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl h-11 px-8 uppercase text-[10px] font-black tracking-widest vibrant-glow" onClick={() => isEditing ? saveReport() : setIsEditing(true)}>
                    {isEditing ? <Save className="h-4 w-4 mr-2" /> : <Edit className="h-4 w-4 mr-2" />}
                    {isEditing ? "Finalize" : "Refine"}
                  </Button>
                  <button onClick={() => setViewingReport(null)} className="h-11 w-11 flex items-center justify-center bg-white/5 hover:bg-red-500/20 text-slate-400 hover:text-red-400 rounded-xl transition-all ml-6">
                    <X className="h-6 w-6" />
                  </button>
                </div>
              </div>
              <ScrollArea className="flex-1 bg-transparent">
                <div className="p-16 md:p-24 max-w-4xl mx-auto">
                  {isEditing ? (
                    <textarea
                      value={reportContent}
                      onChange={(e) => setReportContent(e.target.value)}
                      className="w-full h-[700px] p-12 bg-black/20 border border-white/5 rounded-[2rem] font-mono text-[13px] outline-none resize-none text-slate-300 leading-relaxed shadow-inner"
                    />
                  ) : (
                    <article className="prose prose-slate prose-invert lg:prose-2xl max-w-none prose-headings:font-black prose-headings:tracking-tight prose-p:text-slate-400 prose-p:leading-relaxed prose-strong:text-indigo-400">
                      <ReactMarkdown>{reportContent}</ReactMarkdown>
                    </article>
                  )}
                </div>
              </ScrollArea>
              <div className="h-4 bg-gradient-to-t from-black/20 to-transparent"></div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
