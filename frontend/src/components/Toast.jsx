import { useEffect, useState } from "react";

const ICONS = {
  error: "❌",
  warning: "⚠️",
  success: "✅",
  info: "ℹ️",
};

const COLORS = {
  error: "#ff4d4f",
  warning: "#faad14",
  success: "#52c41a",
  info: "#1890ff",
};

export default function Toast({ toasts, removeToast }) {
  return (
    <div style={{
      position: "fixed",
      top: "20px",
      right: "20px",
      zIndex: 9999,
      display: "flex",
      flexDirection: "column",
      gap: "10px",
      maxWidth: "calc(100% - 40px)",
      pointerEvents: "none",
    }}>
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onClose={() => removeToast(toast.id)} />
      ))}
    </div>
  );
}

function ToastItem({ toast, onClose }) {
  useEffect(() => {
    const timer = setTimeout(onClose, toast.duration || 5000);
    return () => clearTimeout(timer);
  }, [onClose, toast.duration]);

  return (
    <div style={{
      background: "#fff",
      borderLeft: `4px solid ${COLORS[toast.type] || COLORS.info}`,
      borderRadius: "8px",
      boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
      padding: "12px 16px",
      display: "flex",
      alignItems: "flex-start",
      gap: "10px",
      animation: "slideIn 0.3s ease",
      pointerEvents: "all",
      minWidth: "280px",
      maxWidth: "100%",
    }}>
      <span style={{ fontSize: "18px", flexShrink: 0 }}>{ICONS[toast.type]}</span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontWeight: "600", fontSize: "14px", marginBottom: "4px", color: "#222", wordWrap: "break-word" }}>
          {toast.title}
        </div>
        <div style={{ fontSize: "13px", color: "#555", lineHeight: "1.4", wordWrap: "break-word" }}>
          {toast.message}
        </div>
      </div>
      <button onClick={onClose} style={{
        background: "none",
        border: "none",
        cursor: "pointer",
        fontSize: "16px",
        color: "#999",
        padding: "0",
        lineHeight: "1",
        flexShrink: 0,
        minWidth: "24px",
        height: "24px",
      }}>×</button>
    </div>
  );
}
