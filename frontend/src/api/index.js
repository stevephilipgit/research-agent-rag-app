import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000",
});

const resolveApiBase = () => {
  const base = import.meta.env.VITE_API_URL;
  if (base) {
    return base.replace(/\/$/, "");
  }
  return "http://localhost:8000";
};

export const fetchHistory = async () => {
  const { data } = await api.get("/api/history");
  return data;
};

export const fetchDocuments = async () => {
  const { data } = await api.get("/api/documents");
  return data;
};

export const fetchLogs = async () => {
  const { data } = await api.get("/api/logs");
  return data;
};

export const sendQuery = async (query) => {
  const { data } = await api.post("/api/query", { query });
  return data;
};

export const streamQuery = async ({ query, session_id, onEvent }) => {
  const response = await fetch(`${resolveApiBase()}/api/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, session_id }),
  });

  if (!response.ok || !response.body) {
    throw new Error(`Streaming request failed with status ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";

    events.forEach((eventChunk) => {
      const lines = eventChunk.split("\n");
      const eventLine = lines.find((line) => line.startsWith("event: "));
      const dataLine = lines.find((line) => line.startsWith("data: "));
      if (!eventLine || !dataLine) {
        return;
      }

      const type = eventLine.replace("event: ", "").trim();
      const raw = dataLine.replace("data: ", "");
      let data;
      try {
        data = JSON.parse(raw);
      } catch {
        data = raw;
      }

      if (type === "error") {
        const message =
          (typeof data === "object" && data?.detail) ||
          "Something went wrong. Please try again.";
        throw new Error(message);
      }

      onEvent?.({ type, data });
    });
  }

  const tail = buffer.trim();
  if (tail) {
    const lines = tail.split("\n");
    const eventLine = lines.find((line) => line.startsWith("event: "));
    const dataLine = lines.find((line) => line.startsWith("data: "));
    if (eventLine && dataLine) {
      const type = eventLine.replace("event: ", "").trim();
      const raw = dataLine.replace("data: ", "");
      let data;
      try {
        data = JSON.parse(raw);
      } catch {
        data = raw;
      }

      if (type === "error") {
        const message =
          (typeof data === "object" && data?.detail) ||
          "Something went wrong. Please try again.";
        throw new Error(message);
      }

      onEvent?.({ type, data });
    }
  }
};

export const subscribeToLogs = ({ onSnapshot, onLog, onError }) => {
  const source = new EventSource(`${resolveApiBase()}/api/logs/stream`);

  source.addEventListener("snapshot", (event) => {
    onSnapshot?.(JSON.parse(event.data));
  });

  source.addEventListener("log", (event) => {
    onLog?.(JSON.parse(event.data));
  });

  source.addEventListener("error", (event) => {
    onError?.(event);
  });

  return source;
};

export const uploadFiles = async (files) => {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  const { data } = await api.post("/api/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
};

export const deleteDocument = async (docId) => {
  const { data } = await api.delete(`/api/documents/${docId}`);
  return data;
};
