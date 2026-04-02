export default function Loader({ status }) {
  if (status === "ready") return null;

  return (
    <div style={{
      position: "fixed",
      inset: 0,
      background: "linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%)",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      zIndex: 99999,
      gap: "24px",
    }}>
      {/* Logo / Title */}
      <div style={{
        fontSize: "28px",
        fontWeight: "700",
        color: "#fff",
        letterSpacing: "1px",
        marginBottom: "8px",
      }}>
        🔍 Research Assistant
      </div>

      {/* Spinner */}
      <div style={{
        width: "48px",
        height: "48px",
        border: "4px solid rgba(255,255,255,0.1)",
        borderTop: "4px solid #6c63ff",
        borderRadius: "50%",
        animation: "spin 0.9s linear infinite",
      }} />

      {/* Status message */}
      <div style={{
        color: "rgba(255,255,255,0.6)",
        fontSize: "14px",
        textAlign: "center",
        maxWidth: "300px",
        lineHeight: "1.6",
      }}>
        {status === "waking"
          ? "Waking up the server...\nThis may take up to 60 seconds on first load."
          : "Unable to reach server. Please refresh the page."}
      </div>

      {/* Progress dots */}
      {status === "waking" && (
        <div style={{ display: "flex", gap: "8px", marginTop: "8px" }}>
          {[0, 1, 2].map(i => (
            <div key={i} style={{
              width: "8px",
              height: "8px",
              borderRadius: "50%",
              background: "#6c63ff",
              animation: `pulse 1.2s ease-in-out ${i * 0.4}s infinite`,
            }} />
          ))}
        </div>
      )}

      {/* Error state retry button */}
      {status === "error" && (
        <button
          onClick={() => window.location.reload()}
          style={{
            marginTop: "16px",
            padding: "10px 24px",
            background: "#6c63ff",
            color: "#fff",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer",
            fontSize: "14px",
            fontWeight: "600",
          }}
        >
          Retry
        </button>
      )}

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 0.3; transform: scale(0.8); }
          50% { opacity: 1; transform: scale(1.2); }
        }
      `}</style>
    </div>
  );
}
