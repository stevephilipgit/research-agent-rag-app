import { useState, useEffect, useRef, useCallback } from "react";
import { deleteDocument, fetchDocuments, fetchHistory, fetchLogs, streamQuery, subscribeToLogs, uploadFiles, clearSession } from "../api";
import Chat from "../components/Chat";
import LogsPanel from "../components/LogsPanel";
import Modal from "../components/Modal";
import Sidebar from "../components/Sidebar";
import Toast from "../components/Toast";
import Loader from "../components/Loader";
import { useToast } from "../hooks/useToast";

const ENABLE_CHAT_PERSIST = false;

function friendlyError(err, fallback) {
  const detail = err?.response?.data?.detail || err?.message || "";
  if (detail.toLowerCase().includes("timeout")) {
    return "The backend took too long to respond. Please try again.";
  }
  if (detail.toLowerCase().includes("no relevant")) {
    return "No relevant documents found.";
  }
  return fallback;
}

function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "dark");
  const [messages, setMessages] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [logs, setLogs] = useState([]);
  const [query, setQuery] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [serverStatus, setServerStatus] = useState("waking"); // "waking" | "ready" | "error"
  const [thinkingLabel, setThinkingLabel] = useState("Thinking.");
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [pendingResponse, setPendingResponse] = useState("");
  const [uploadState, setUploadState] = useState({ status: "", files: [] });
  const [selfHealingMode, setSelfHealingMode] = useState(false);
  const fileInputRef = useRef(null);
  const { toasts, removeToast, toast } = useToast();

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    localStorage.setItem("theme", theme);
  }, [theme]);

  useEffect(() => {
    const loadInitialState = async () => {
      try {
        const [historyData, documentData, logData] = await Promise.all([
          ENABLE_CHAT_PERSIST ? fetchHistory() : Promise.resolve({ messages: [] }),
          fetchDocuments(),
          fetchLogs(),
        ]);
        setMessages(historyData.messages || []);
        setDocuments(documentData.documents || []);
        setLogs(logData.logs || []);
      } catch (err) {
        setError(friendlyError(err, "Unable to load the app right now."));
      }
    };

    loadInitialState();
  }, []);

  useEffect(() => {
    const source = subscribeToLogs({
      onSnapshot: (entries) => setLogs(entries || []),
      onLog: (entry) => {
        setLogs((prev) => (prev.some((item) => item.id === entry.id) ? prev : [...prev, entry]));
      },
      onError: () => {},
    });

    const handleBeforeUnload = () => {
      // Optional: Fire and forget session cleanup
      // Navigator.sendBeacon is better for this but requires a specific endpoint design
      // For now we rely on the 2 hour TTL mainly, but we can try a fetch with keepalive
      const API_URL = import.meta.env.VITE_API_URL || "https://research-agent-rag-app-1.onrender.com";
      const sessionId = sessionStorage.getItem("rag_session_id");
      if (sessionId) {
        fetch(`${API_URL}/api/session`, {
          method: "DELETE",
          headers: { "X-Session-ID": sessionId },
          keepalive: true,
        });
      }
    };

    window.addEventListener("beforeunload", handleBeforeUnload);

    return () => {
      source.close();
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    const checkBackend = async () => {
      const BASE_URL = import.meta.env.VITE_API_URL || "https://research-agent-rag-app-1.onrender.com";
      try {
        await fetch(`${BASE_URL}/health`);
        if (!cancelled) setServerStatus("ready");
      } catch {
        if (!cancelled) {
          // Retry once after 10 seconds
          setTimeout(async () => {
            try {
              await fetch(`${BASE_URL}/health`);
              if (!cancelled) setServerStatus("ready");
            } catch {
              if (!cancelled) setServerStatus("error");
            }
          }, 10000);
        }
      }
    };

    checkBackend();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!loading) {
      setThinkingLabel("Thinking.");
      return undefined;
    }

    const states = ["Thinking.", "Thinking..", "Thinking..."];
    let index = 0;
    const timer = setInterval(() => {
      index = (index + 1) % states.length;
      setThinkingLabel(states[index]);
    }, 420);

    return () => clearInterval(timer);
  }, [loading]);

  const openPicker = () => fileInputRef.current?.click();

  const handleUpload = async (files) => {
    if (!files.length || uploading || loading) {
      return;
    }

    // Addition 1: Pre-flight checks
    for (const file of files) {
      // 10MB limit
      if (file.size > 10 * 1024 * 1024) {
        toast.error(
          "File Too Large",
          `"${file.name}" exceeds the 10MB limit. Please compress or split the file and try again.`,
          7000
        );
        return;
      }

      // Type check
      const allowed = [".pdf", ".txt", ".csv", ".docx"];
      const ext = "." + file.name.split(".").pop().toLowerCase();
      if (!allowed.includes(ext)) {
        toast.error(
          "Unsupported File Type",
          `"${file.name}" is not supported. Please upload a PDF, TXT, CSV, or DOCX file.`,
          7000
        );
        return;
      }
    }

    setError("");
    setUploading(true);
    setUploadState({
      status: "Uploading documents...",
      files: files.map((file) => file.name),
    });

    try {
      const data = await uploadFiles(files);
      setDocuments(data.documents || []);
      setLogs(data.logs || []);
      setUploadState({
        status: (data.uploaded_files || []).length ? "Upload complete" : "No new files uploaded",
        files: data.uploaded_files || files.map((file) => file.name),
      });

      if ((data.uploaded_files || []).length > 0) {
        toast.success(
          "Upload Successful",
          `${data.uploaded_files.length} file(s) indexed and ready for queries.`,
          4000
        );
      } else {
        toast.warning(
          "No New Files",
          "The uploaded files were duplicates or invalid.",
          3000
        );
      }
    } catch (err) {
      setUploadState({
        status: "Upload failed",
        files: files.map((file) => file.name),
      });

      const msg = err.message || "";
      if (msg.includes("429") || msg.includes("Too Many")) {
        toast.error(
          "Too Many Uploads",
          "You've reached the upload rate limit. Please wait a minute before uploading again.",
          8000
        );
      } else if (msg.includes("Maximum") && msg.includes("documents")) {
        toast.error(
          "Document Limit Reached",
          "You can upload a maximum of 5 documents per session. Please delete an existing document to upload a new one.",
          8000
        );
      } else if (msg.includes("413") || msg.includes("too large")) {
        toast.error(
          "File Too Large",
          "The file exceeds the maximum allowed size of 10MB. Please compress the file and try again.",
          7000
        );
      } else if (msg.includes("network") || msg.includes("fetch") || msg.includes("Failed to fetch")) {
        toast.error(
          "Connection Error",
          "Could not reach the server. Please check your internet connection or wait for the server to wake up.",
          8000
        );
      } else {
        toast.error(
          "Upload Failed",
          friendlyError(err, "An unexpected error occurred during upload. Please try again."),
          6000
        );
      }
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) {
      toast.warning(
        "Empty Question",
        "Please type a question before sending.",
        3000
      );
      return;
    }
    
    if (loading || uploading) {
      return;
    }

    setError("");
    setLoading(true);
    setPendingResponse("");
    setMessages((prev) => [...prev, { role: "user", content: trimmed }]);
    setQuery("");

    try {
      await streamQuery({
        query: trimmed,
        enableSelfHealing: selfHealingMode,
        onEvent: ({ type, data }) => {
          if (type === "meta" && data?.logs) {
            setLogs(data.logs);
          }

          if (type === "token") {
            setPendingResponse((prev) => `${prev}${data}`);
          }

          if (type === "self_healing_step") {
            // Update the last message with healing step info
            setMessages((prev) => {
              if (prev.length === 0) return prev;
              const updated = [...prev];
              const lastMsg = updated[updated.length - 1];
              if (!lastMsg.healing_steps) lastMsg.healing_steps = [];
              lastMsg.healing_steps.push(data);
              return updated;
            });
          }

          if (type === "done") {
            setMessages(data.messages || []);
            setLogs(data.logs || []);
            setPendingResponse("");
          }
        },
        onError: (msg) => {
          setPendingResponse("");
          toast.error(
            "Query Failed",
            msg || "Something went wrong. Please try again.",
            6000
          );
        },
      });
    } catch (err) {
      const msg = err.message || "";
      setPendingResponse("");
      
      if (msg.includes("429") || msg.includes("Too Many")) {
        toast.warning(
          "Slow Down!",
          "You're sending queries too quickly. Please wait a moment before asking another question.",
          6000
        );
      } else if (msg.includes("network") || msg.includes("fetch") || msg.includes("Failed to fetch")) {
        toast.error(
          "Connection Lost",
          "Lost connection to the server. The server may be waking up — please wait 30 seconds and try again.",
          8000
        );
      } else {
        toast.error(
          "Query Failed",
          friendlyError(err, "The assistant could not complete that request."),
          6000
        );
      }
    } finally {
      setLoading(false);
    }
  };

  const confirmDelete = async () => {
    if (!deleteTarget) {
      return;
    }

    setError("");
    try {
      const data = await deleteDocument(deleteTarget.doc_id);
      setDocuments(data.documents || []);
      setLogs(data.logs || []);
      toast.success(
        "Document Deleted",
        `"${deleteTarget.file_name}" has been removed successfully.`,
        3000
      );
    } catch (err) {
      toast.error(
        "Delete Failed",
        `Could not delete "${deleteTarget.file_name}". Please try again.`,
        5000
      );
    } finally {
      setDeleteTarget(null);
    }
  };

  const handleNewChat = async () => {
    // Show processing notification
    toast.warning(
      "Session Clearing",
      "Resetting chat and clearing documents...",
      2000
    );

    // Clear UI state immediately
    setMessages([]);
    setDocuments([]);
    setQuery("");
    setPendingResponse("");
    setError("");
    setServerStatus("ready");

    // Generate fresh session ID
    const newSessionId = crypto.randomUUID();
    
    // Clean up old session on backend BEFORE switching ID in storage
    try {
      await clearSession();
    } catch (e) {
      console.warn("Session cleanup failed:", e);
    }

    // Now switch to new session ID
    sessionStorage.setItem("rag_session_id", newSessionId);
    
    toast.success("New Chat Started", "Your session has been reset and documents cleared.", 3000);
  };

  return (
    <>
      <Loader status={serverStatus} />
      <Toast toasts={toasts} removeToast={removeToast} />
      <div className="app">
        <Sidebar documents={documents} uploadState={uploadState} onDeleteDocument={setDeleteTarget} />

        <main className="chat-column">
          <div className="chat-header">
            <div className="chat-header-row">
              <div>
                <h1>RAG Agent Assistant</h1>
                <p>Grounded answers from your uploaded files, retrieval, and agent workflow.</p>
                <div className="session-status">
                  <span className={`status-pill ${documents.length >= 5 ? 'at-limit' : ''}`}>
                    Documents: {documents.length} / 5
                  </span>
                </div>
              </div>
              <button
                type="button"
                className="new-chat-button"
                onClick={handleNewChat}
              >
                + New Chat
              </button>
              <button
                type="button"
                className={`self-healing-toggle ${selfHealingMode ? 'active' : ''}`}
                onClick={() => setSelfHealingMode(!selfHealingMode)}
                aria-label="Toggle self-healing mode"
                title="Toggle AI self-healing: enables iterative evaluation and retry"
              >
                {selfHealingMode ? "🤖 Self-Healing ON" : "🤖 Self-Healing OFF"}
              </button>
              <button
                type="button"
                className="theme-toggle"
                onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
                aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
              >
                {theme === "dark" ? "☀️ Light" : "🌙 Dark"}
              </button>
            </div>
          </div>

          <div className="chat-body">
            <Chat
              messages={messages}
              loading={loading}
              pendingResponse={pendingResponse}
              thinkingLabel={thinkingLabel}
            />
          </div>

          <form className="input-bar" onSubmit={handleSubmit}>
            <button
              className="composer-icon-button"
              type="button"
              onClick={openPicker}
              disabled={loading || uploading}
              aria-label="Upload files"
            >
              +
            </button>

            <input
              ref={fileInputRef}
              type="file"
              className="hidden-file-input"
              accept=".pdf,.txt,.csv,.docx"
              multiple
              onChange={(event) => handleUpload(Array.from(event.target.files || []))}
            />

            <textarea
              className="chat-input"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder={uploading ? "Uploading documents..." : "Message the assistant"}
              disabled={loading || uploading || serverStatus !== "ready"}
              rows={1}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
            />

            <button
              className="send-button"
              type="submit"
              disabled={loading || uploading || !query.trim() || serverStatus !== "ready"}
            >
              {uploading ? "Uploading..." : loading ? "Processing..." : "Send"}
            </button>
          </form>
        </main>

        <LogsPanel logs={logs} />
      </div>

      <Modal
        open={Boolean(deleteTarget)}
        title="Delete this document and its embeddings?"
        description={deleteTarget?.file_name || ""}
        cancelLabel="Cancel"
        confirmLabel="Confirm"
        onCancel={() => setDeleteTarget(null)}
        onConfirm={confirmDelete}
      />
    </>
  );
}

export default App;
