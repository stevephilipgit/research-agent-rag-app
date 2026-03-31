function Modal({
  open,
  title,
  description,
  cancelLabel,
  confirmLabel,
  onCancel,
  onConfirm,
}) {
  if (!open) {
    return null;
  }

  return (
    <div className="modal-overlay" role="presentation" onClick={onCancel}>
      <div
        className="modal-shell"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-title">{title}</div>
        <div className="modal-text">{description}</div>
        <div className="modal-actions">
          <button type="button" className="modal-button secondary" onClick={onCancel}>
            {cancelLabel}
          </button>
          <button type="button" className="modal-button danger" onClick={onConfirm}>
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

export default Modal;
