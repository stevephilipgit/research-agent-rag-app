import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function MarkdownMessage({ content }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        h1: ({ children }) => <h1 className="markdown-h1">{children}</h1>,
        h2: ({ children }) => <h2 className="markdown-h2">{children}</h2>,
        h3: ({ children }) => <h3 className="markdown-h3">{children}</h3>,
        p: ({ children }) => <p className="markdown-p">{children}</p>,
        ul: ({ children }) => <ul className="markdown-ul">{children}</ul>,
        ol: ({ children }) => <ol className="markdown-ol">{children}</ol>,
        li: ({ children }) => <li className="markdown-li">{children}</li>,
        code: ({ inline, children }) =>
          inline ? (
            <code className="markdown-inline-code">{children}</code>
          ) : (
            <code className="markdown-code-block">{children}</code>
          ),
        pre: ({ children }) => <pre className="markdown-pre">{children}</pre>,
        table: ({ children }) => <table className="markdown-table">{children}</table>,
        thead: ({ children }) => <thead>{children}</thead>,
        tbody: ({ children }) => <tbody>{children}</tbody>,
        tr: ({ children }) => <tr>{children}</tr>,
        th: ({ children }) => <th>{children}</th>,
        td: ({ children }) => <td>{children}</td>,
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

function Chat({ messages, loading, pendingResponse, thinkingLabel }) {
  const scrollRef = useRef(null);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, pendingResponse, loading]);

  return (
    <section className="chat-panel">
      <div className="messages-scroll" ref={scrollRef}>
        {messages.length === 0 ? (
          <div className="welcome-card">
            <div className="welcome-kicker">Ready</div>
            <h2>Ask about your documents, URLs, or research questions.</h2>
            <p>Upload files with the + button, then start a conversation in the center panel.</p>
          </div>
        ) : null}

        {messages.map((message, index) => (
          <article
            key={`${message.role}-${index}`}
            className={`message-row ${message.role === "user" ? "user" : "assistant"}`}
          >
            <div className={`message-card ${message.role === "user" ? "user" : "assistant"}`}>
              <div className="message-label">{message.role === "user" ? "You" : "Assistant"}</div>
              <div className="message-copy">
                <MarkdownMessage content={message.content} />
              </div>
              {message.role === "assistant" && message.citations?.length > 0 ? (
                <details className="message-footer-block collapsible-block">
                  <summary className="collapsible-summary">
                    <span className="summary-arrow">&gt;</span>
                    <span className="footer-title">Citations</span>
                  </summary>
                  <div className="footer-list">
                    {message.citations.map((citation, citationIndex) => (
                      <div className="footer-item" key={`${index}-citation-${citationIndex}`}>
                        {citation.type === "document"
                          ? `${citation.document || "unknown"} - page ${citation.page || "?"}`
                          : citation.label || "Source"}
                      </div>
                    ))}
                  </div>
                </details>
              ) : null}

              {message.role === "assistant" && message.steps?.length > 0 ? (
                <details className="message-footer-block collapsible-block">
                  <summary className="collapsible-summary">
                    <span className="summary-arrow">&gt;</span>
                    <span className="footer-title">Agent Steps</span>
                  </summary>
                  <div className="footer-list">
                    {message.steps.map((step, stepIndex) => (
                      <div className="footer-item" key={`${index}-step-${stepIndex}`}>
                        <strong>{step.label || "Step"}</strong>
                        {step.detail ? `: ${step.detail}` : ""}
                      </div>
                    ))}
                  </div>
                </details>
              ) : null}
            </div>
          </article>
        ))}

        {loading ? (
          <article className="message-row assistant">
            <div className="message-card assistant streaming-card">
              <div className="message-label">Assistant</div>
              <div className="message-copy">
                <MarkdownMessage content={pendingResponse || thinkingLabel} />
              </div>
            </div>
          </article>
        ) : null}

        <div ref={endRef} />
      </div>
    </section>
  );
}

export default Chat;
