import { useEffect, useRef, useState } from "react";
import { deleteDocument, fetchDocuments, fetchHistory, fetchLogs, streamQuery, subscribeToLogs, uploadFiles } from "../api";
import Chat from "../components/Chat";
import LogsPanel from "../components/LogsPanel";
import Modal from "../components/Modal";
import Sidebar from "../components/Sidebar";

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
  const [thinkingLabel, setThinkingLabel] = useState("Thinking.");
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [sessionId, setSessionId] = useState(() => Date.now().toString());
  const [pendingResponse, setPendingResponse] = useState("");
  const [uploadState, setUploadState] = useState({ status: "", files: [] });
  const fileInputRef = useRef(null);

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

    return () => source.close();
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
    } catch (err) {
      setUploadState({
        status: "Upload failed",
        files: files.map((file) => file.name),
      });
      setError(friendlyError(err, "Upload failed. Please try again."));
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
    if (!trimmed || loading || uploading) {
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
        session_id: sessionId,
        onEvent: ({ type, data }) => {
          if (type === "meta" && data?.logs) {
            setLogs(data.logs);
          }

          if (type === "token") {
            setPendingResponse((prev) => `${prev}${data}`);
          }

          if (type === "done") {
            setMessages(data.messages || []);
            setLogs(data.logs || []);
            setPendingResponse("");
          }
        },
      });
    } catch (err) {
      setPendingResponse("");
      setError(friendlyError(err, "The assistant could not complete that request."));
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
    } catch (err) {
      setError(friendlyError(err, "Delete failed. Please try again."));
    } finally {
      setDeleteTarget(null);
    }
  };

  const handleNewChat = () => {
    setMessages([]);
    setPendingResponse("");
    setQuery("");
    setSessionId(Date.now().toString());
    setError("");
  };

  return (
    <>
      <div className="app">
        <Sidebar documents={documents} uploadState={uploadState} onDeleteDocument={setDeleteTarget} />

        <main className="chat-column">
          <div className="chat-header">
            <div className="chat-header-row">
              <div>
                <h1>RAG Agent Assistant</h1>
                <p>Grounded answers from your uploaded files, retrieval, and agent workflow.</p>
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

          {error ? <div className="error-banner">{error}</div> : null}
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
              disabled={loading || uploading}
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
              disabled={loading || uploading || !query.trim()}
            >
              {loading ? "..." : "Send"}
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
