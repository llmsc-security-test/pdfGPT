"use client";

import { useState, useRef, useEffect, useCallback, FormEvent } from "react";
import {
  FileText, Upload, Send, Trash2, Settings, Zap, Brain,
  ChevronDown, Loader2, FileSpreadsheet, File, X,
} from "lucide-react";
import { MODEL_LIST, type ModelOption } from "@/lib/models";
import type { DocumentChunk } from "@/lib/document-processor";

interface Message {
  role: "user" | "assistant";
  content: string;
}

const FREE_MODELS = MODEL_LIST.filter((m) => m.free);
const PAID_MODELS = MODEL_LIST.filter((m) => !m.free);

function fileIcon(name: string) {
  const ext = name.split(".").pop()?.toLowerCase();
  if (ext === "pdf") return <FileText className="w-4 h-4" />;
  if (["xlsx", "xls", "csv"].includes(ext || ""))
    return <FileSpreadsheet className="w-4 h-4" />;
  return <File className="w-4 h-4" />;
}

export default function Home() {
  const [chunks, setChunks] = useState<DocumentChunk[]>([]);
  const [docInfo, setDocInfo] = useState<{
    name: string;
    pages: number;
    chunkCount: number;
  } | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [selectedModel, setSelectedModel] = useState<ModelOption>(
    MODEL_LIST.find((m) => m.id === "openai:gpt-4o-mini") || FREE_MODELS[0]
  );
  const [apiKey, setApiKey] = useState("");
  const [mode, setMode] = useState<"rag" | "agentic">("rag");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [error, setError] = useState("");
  const chatEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleUpload = useCallback(async (file: globalThis.File) => {
    setUploading(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch("/api/upload", { method: "POST", body: formData });
      const data = await res.json();
      if (data.error) {
        setError(data.error);
        return;
      }
      setChunks(data.chunks);
      setDocInfo({
        name: data.name,
        pages: data.pageCount,
        chunkCount: data.chunkCount,
      });
      setMessages([]);
    } catch {
      setError("Failed to upload file");
    } finally {
      setUploading(false);
    }
  }, []);

  const handleSend = useCallback(
    async (e?: FormEvent) => {
      e?.preventDefault();
      if (!input.trim() || loading) return;
      if (chunks.length === 0) {
        setError("Upload a document first");
        return;
      }

      const question = input.trim();
      setInput("");
      setError("");
      setMessages((prev) => [...prev, { role: "user", content: question }]);
      setLoading(true);

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question,
            chunks,
            modelId: selectedModel.id,
            apiKey: apiKey || undefined,
            mode,
          }),
        });

        const contentType = res.headers.get("content-type") || "";
        if (contentType.includes("application/json")) {
          const errData = await res.json();
          throw new Error(errData.error || `Server error: ${res.status}`);
        }

        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }

        const reader = res.body?.getReader();
        if (!reader) throw new Error("No response stream");

        setMessages((prev) => [...prev, { role: "assistant", content: "" }]);
        const decoder = new TextDecoder();
        let fullText = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value);
          const lines = chunk.split("\n");
          for (const line of lines) {
            if (line.startsWith("0:")) {
              const textPart = line.slice(2);
              try {
                const parsed = JSON.parse(textPart);
                fullText += parsed;
                setMessages((prev) => {
                  const updated = [...prev];
                  updated[updated.length - 1] = {
                    role: "assistant",
                    content: fullText,
                  };
                  return updated;
                });
              } catch {
                // skip non-text chunks
              }
            }
          }
        }
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Failed to get response";
        setError(msg);
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: `Error: ${msg}` },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [input, loading, chunks, selectedModel, apiKey, mode]
  );

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside
        className={`${
          sidebarOpen ? "w-80" : "w-0"
        } transition-all duration-300 bg-gray-900 border-r border-gray-800 flex flex-col overflow-hidden shrink-0`}
      >
        <div className="p-4 border-b border-gray-800">
          <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            pdfGPT
          </h1>
          <p className="text-xs text-gray-500 mt-1">Smart Document Analysis</p>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-5">
          {/* File Upload */}
          <section>
            <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              Document
            </label>
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept=".pdf,.docx,.doc,.xlsx,.xls,.csv,.pptx,.ppt,.txt,.md"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleUpload(f);
              }}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              className="mt-2 w-full flex items-center gap-2 px-4 py-3 rounded-lg border-2 border-dashed border-gray-700 hover:border-blue-500 hover:bg-gray-800/50 transition-colors text-sm"
            >
              {uploading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Upload className="w-4 h-4" />
              )}
              <span>{uploading ? "Processing..." : "Upload file"}</span>
            </button>
            <p className="text-[10px] text-gray-600 mt-1">
              PDF, Word, Excel, PowerPoint, TXT
            </p>
            {docInfo && (
              <div className="mt-2 flex items-center gap-2 px-3 py-2 bg-gray-800 rounded-lg text-xs">
                {fileIcon(docInfo.name)}
                <div className="min-w-0 flex-1">
                  <p className="truncate font-medium">{docInfo.name}</p>
                  <p className="text-gray-500">
                    {docInfo.pages} pages &middot; {docInfo.chunkCount} chunks
                  </p>
                </div>
                <button
                  onClick={() => {
                    setChunks([]);
                    setDocInfo(null);
                    setMessages([]);
                  }}
                  className="text-gray-500 hover:text-red-400"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            )}
          </section>

          {/* Mode */}
          <section>
            <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              Analysis Mode
            </label>
            <div className="mt-2 grid grid-cols-2 gap-2">
              <button
                onClick={() => setMode("rag")}
                className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
                  mode === "rag"
                    ? "bg-blue-600 text-white"
                    : "bg-gray-800 text-gray-400 hover:bg-gray-700"
                }`}
              >
                <Zap className="w-3.5 h-3.5" /> RAG
              </button>
              <button
                onClick={() => setMode("agentic")}
                className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
                  mode === "agentic"
                    ? "bg-purple-600 text-white"
                    : "bg-gray-800 text-gray-400 hover:bg-gray-700"
                }`}
              >
                <Brain className="w-3.5 h-3.5" /> Agentic
              </button>
            </div>
            <p className="text-[10px] text-gray-600 mt-1">
              {mode === "rag"
                ? "Fast retrieval-augmented generation"
                : "Deep multi-step analysis with reasoning"}
            </p>
          </section>

          {/* Model */}
          <section>
            <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              Model
            </label>
            <div className="mt-2 relative">
              <select
                value={selectedModel.id}
                onChange={(e) => {
                  const m = MODEL_LIST.find((x) => x.id === e.target.value);
                  if (m) setSelectedModel(m);
                }}
                className="w-full appearance-none bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm pr-8 focus:outline-none focus:ring-1 focus:ring-blue-500"
              >
                <optgroup label="Free Models">
                  {FREE_MODELS.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name} ({m.provider})
                    </option>
                  ))}
                </optgroup>
                <optgroup label="Premium Models">
                  {PAID_MODELS.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name} ({m.provider})
                    </option>
                  ))}
                </optgroup>
              </select>
              <ChevronDown className="w-4 h-4 absolute right-2 top-2.5 text-gray-500 pointer-events-none" />
            </div>
            <p className="text-[10px] text-gray-600 mt-1">
              {selectedModel.description}
            </p>
          </section>

          {/* API Key */}
          {selectedModel.requiresKey && (
            <section>
              <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider flex items-center gap-1">
                <Settings className="w-3 h-3" /> API Key
              </label>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder={`Enter ${selectedModel.provider} API key`}
                className="mt-2 w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 placeholder:text-gray-600"
              />
              <p className="text-[10px] text-gray-600 mt-1">
                {selectedModel.free
                  ? "Free key - sign up at the provider website"
                  : "Paid - charges apply per usage"}
              </p>
            </section>
          )}
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="h-14 border-b border-gray-800 flex items-center px-4 gap-3 shrink-0">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-1.5 rounded-md hover:bg-gray-800 transition-colors"
          >
            <FileText className="w-5 h-5 text-gray-400" />
          </button>
          <span className="text-sm font-medium text-gray-300">
            {docInfo
              ? `${docInfo.name} - ${mode === "agentic" ? "Agentic Analysis" : "RAG"}`
              : "Upload a document to get started"}
          </span>
          {messages.length > 0 && (
            <button
              onClick={() => setMessages([])}
              className="ml-auto p-1.5 rounded-md hover:bg-gray-800 text-gray-500 hover:text-gray-300"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </header>

        {/* Chat area */}
        <div className="flex-1 overflow-y-auto px-4 py-6">
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center max-w-md">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center mx-auto mb-4">
                  <FileText className="w-8 h-8 text-white" />
                </div>
                <h2 className="text-lg font-semibold text-gray-200">
                  pdfGPT - Document Analysis
                </h2>
                <p className="text-sm text-gray-500 mt-2">
                  Upload a PDF, Word, Excel, or PowerPoint file and ask
                  questions about it. Choose between fast RAG or deep Agentic
                  analysis.
                </p>
                <div className="mt-6 grid grid-cols-2 gap-3 text-xs">
                  <div className="bg-gray-900 rounded-lg p-3 text-left">
                    <Zap className="w-4 h-4 text-blue-400 mb-1" />
                    <p className="font-medium text-gray-300">RAG Mode</p>
                    <p className="text-gray-600">
                      Quick retrieval and answer generation
                    </p>
                  </div>
                  <div className="bg-gray-900 rounded-lg p-3 text-left">
                    <Brain className="w-4 h-4 text-purple-400 mb-1" />
                    <p className="font-medium text-gray-300">Agentic Mode</p>
                    <p className="text-gray-600">
                      Multi-step reasoning and deep analysis
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto space-y-4">
              {messages.map((msg, i) => (
                <div
                  key={i}
                  className={`flex ${
                    msg.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
                      msg.role === "user"
                        ? "bg-blue-600 text-white"
                        : "bg-gray-800 text-gray-200"
                    }`}
                  >
                    {msg.content || (
                      <Loader2 className="w-4 h-4 animate-spin text-gray-500" />
                    )}
                  </div>
                </div>
              ))}
              <div ref={chatEndRef} />
            </div>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="mx-4 mb-2 px-3 py-2 bg-red-500/10 border border-red-500/20 rounded-lg text-xs text-red-400 flex items-center justify-between">
            <span>{error}</span>
            <button onClick={() => setError("")}>
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        {/* Input */}
        <form
          onSubmit={handleSend}
          className="border-t border-gray-800 p-4 shrink-0"
        >
          <div className="max-w-3xl mx-auto flex gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={
                chunks.length > 0
                  ? "Ask about your document..."
                  : "Upload a document first"
              }
              disabled={loading || chunks.length === 0}
              className="flex-1 bg-gray-900 border border-gray-700 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 disabled:opacity-50 placeholder:text-gray-600"
            />
            <button
              type="submit"
              disabled={loading || !input.trim() || chunks.length === 0}
              className="px-4 py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-800 disabled:text-gray-600 rounded-xl transition-colors"
            >
              {loading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </button>
          </div>
        </form>
      </main>
    </div>
  );
}
