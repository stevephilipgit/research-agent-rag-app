import { useState, useCallback } from "react";

export function useToast() {
  const [toasts, setToasts] = useState([]);

  const addToast = useCallback((type, title, message, duration = 5000) => {
    const id = Date.now() + Math.random();
    setToasts(prev => [...prev, { id, type, title, message, duration }]);
  }, []);

  const removeToast = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  // Convenience methods
  const toast = {
    error: (title, message, duration) => addToast("error", title, message, duration),
    warning: (title, message, duration) => addToast("warning", title, message, duration),
    success: (title, message, duration) => addToast("success", title, message, duration),
    info: (title, message, duration) => addToast("info", title, message, duration),
  };

  return { toasts, removeToast, toast };
}
