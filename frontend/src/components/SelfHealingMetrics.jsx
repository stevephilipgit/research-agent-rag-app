import "./SelfHealingMetrics.css";

// Score confidence badge
function ScoreBadge({ score }) {
  if (score === null || score === undefined) return null;

  let confidence = "Low";
  let className = "score-badge low";

  if (score >= 0.75) {
    confidence = "High";
    className = "score-badge high";
  } else if (score >= 0.5) {
    confidence = "Medium";
    className = "score-badge medium";
  }

  return (
    <span className={className}>
      ✓ {confidence} ({(score * 100).toFixed(0)}%)
    </span>
  );
}

function SelfHealingMetrics({ evalScore, retryCount, selfHealingEnabled, healingSteps }) {
  if (!selfHealingEnabled) {
    return null;
  }

  const retryLabel =
    retryCount === 0 ? "First attempt" : retryCount === 1 ? "1 retry" : `${retryCount} retries`;

  return (
    <details className="metrics-footer-block collapsible-block">
      <summary className="collapsible-summary">
        <span className="summary-arrow">&gt;</span>
        <span className="footer-title">Self-Healing</span>
      </summary>
      <div className="footer-list metrics-list">
        <div className="metric-item">
          <strong>Response Quality</strong>
          <div className="metric-value">
            <ScoreBadge score={evalScore} />
          </div>
        </div>

        {retryCount > 0 && (
          <div className="metric-item">
            <strong>Attempts</strong>
            <div className="metric-value">Improved after {retryLabel}</div>
          </div>
        )}

        {healingSteps && healingSteps.length > 0 && (
          <div className="metric-item">
            <strong>Optimization Steps</strong>
            <div className="healing-steps">
              {healingSteps.map((step, idx) => (
                <div key={idx} className="healing-step">
                  <span className="step-number">Attempt {step.attempt}</span>
                  {step.score !== undefined && (
                    <span className="step-score">
                      Score: {(step.score * 100).toFixed(0)}%
                    </span>
                  )}
                  <span className={`step-status ${step.status}`}>{step.status}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </details>
  );
}

export default SelfHealingMetrics;
