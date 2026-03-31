import { useEffect, useRef } from "react";

const statusMap = {
  success: "success",
  failure: "error",
  in_progress: "running",
};

function LogsPanel({ logs }) {
  const logEndRef = useRef(null);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [logs]);

  return (
    <aside className="sidebar logs-sidebar">
      <div className="sidebar-header sticky-header">
        <div className="sidebar-title">Live Logs</div>
        <div className="sidebar-subtitle">Pipeline and agent trace</div>
      </div>

      <div className="sidebar-scroll">
        <section className="sidebar-card logs-card">
          {logs.length === 0 ? (
            <div className="empty-copy">Live system logs will appear here.</div>
          ) : (
            <div className="logs-stream">
              {logs.map((log) => (
                <article className="log-entry fade-in" key={log.id}>
                  <div className="log-head">
                    <span className={`log-dot ${statusMap[log.status] || "running"}`} />
                    <span className="log-time">[{log.time}]</span>
                    <span className="log-step">{log.step}</span>
                  </div>
                  <div className="log-detail">{log.detail || "No details provided."}</div>
                </article>
              ))}
              <div ref={logEndRef} />
            </div>
          )}
        </section>
      </div>
    </aside>
  );
}

export default LogsPanel;
