function Sidebar({ documents, uploadState, onDeleteDocument }) {
  return (
    <aside className="sidebar documents-sidebar">
      <div className="sidebar-header sticky-header">
        <div className="sidebar-title">Documents</div>
        <div className="sidebar-subtitle">Uploaded knowledge base</div>
      </div>

      <div className="sidebar-scroll">
        <section className="sidebar-card upload-card">
          <div className="card-title">Latest Upload</div>
          {uploadState?.status ? (
            <>
              <div className="upload-badge">{uploadState.status}</div>
              <div className="upload-files">
                {uploadState.files?.map((file) => (
                  <div className="upload-file" key={file}>
                    {file}
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="empty-copy">Uploads will appear here after you add files from the composer.</div>
          )}
        </section>

        <section className="sidebar-card documents-card">
          <div className="card-title">Available Files</div>
          {documents.length === 0 ? (
            <div className="empty-copy">No documents uploaded yet.</div>
          ) : (
            <div className="document-stack">
              {documents.map((doc) => {
                const docId = doc.id || doc.doc_id;
                const fileName = doc.filename || doc.file_name || "Unknown File";
                return (
                <div className="document-row" key={docId}>
                  <div className="document-meta">
                    <div className="file-icon">DOC</div>
                    <div className="document-name">{fileName}</div>
                  </div>
                  <button
                    type="button"
                    className="delete-doc-button"
                    onClick={() => onDeleteDocument({ id: docId, filename: fileName })}
                    aria-label={`Delete ${fileName}`}
                  >
                    x
                  </button>
                </div>
              )})}
            </div>
          )}
        </section>
      </div>
    </aside>
  );
}

export default Sidebar;
