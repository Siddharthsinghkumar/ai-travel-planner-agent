import { useMemo } from "react";
import type { TripPlan } from "../lib/types";
import type { Flight } from "../lib/types";

type StreamPaneProps = {
  tokens: string;
  finalText?: string;
  finalJson?: TripPlan | null;
  fallbackBestFlight?: Flight | null;
  fallbackWeather?: Record<string, unknown> | null;
  isStreaming: boolean;
  canCancel?: boolean;
  statusText?: string;
  onCancel: () => void;
};

export default function StreamPane({
  tokens,
  finalText = "",
  finalJson,
  fallbackBestFlight,
  fallbackWeather,
  isStreaming,
  canCancel = true,
  statusText,
  onCancel
}: StreamPaneProps) {
  const hasTokenText = typeof tokens === "string" && tokens.length > 0;
  const hasFinalText = typeof finalText === "string" && finalText.trim().length > 0;
  const hasFallbackSignal = Boolean(finalJson || fallbackBestFlight || fallbackWeather);
  const isIdle = !isStreaming && !hasTokenText && !hasFinalText && !hasFallbackSignal;
  const visibleText = hasTokenText ? tokens : hasFinalText ? finalText : "";
  const shouldShowNarrative = useMemo(() => hasTokenText || hasFinalText, [hasTokenText, hasFinalText]);

  return (
    <div
      className="stream-pane"
      aria-live="polite"
      aria-busy={isStreaming}
      data-testid="stream-pane"
    >
      {isStreaming && !hasTokenText ? (
        <div className="stream-pane__loading" aria-label="Loading stream">
          <p className="stream-pane__loading-title">Finding your best options…</p>
          <div className="shim-wrap">
            <div className="shim" style={{ width: "92%" }} />
            <div className="shim" style={{ width: "74%" }} />
          </div>
        </div>
      ) : null}

      {shouldShowNarrative ? (
        <div
          className={[
            "r-text llm-pane min-w-0 stream-pane__body",
            hasFinalText && !hasTokenText ? "stream-pane__body--final" : "",
          ]
            .filter(Boolean)
            .join(" ")}
          data-testid="stream-pane-body"
        >
          {visibleText}
          {isStreaming && hasTokenText && <span className="stream-caret" aria-hidden="true" />}
        </div>
      ) : null}

      {isStreaming && (
        <div className="stream-pane__controls">
          <div role="status" className="min-w-0 break-words stream-pane__status">
            {statusText || (hasTokenText ? "Building your trip summary..." : "Searching flights — results appear below as they arrive.")}
          </div>

          {canCancel && (
            <button
              onClick={onCancel}
              className="stream-pane__cancel"
              aria-label="Cancel streaming"
              data-testid="stream-cancel"
            >
              Stop
            </button>
          )}
        </div>
      )}

      {isIdle && (
        <div className="stream-empty">
          <div className="stream-empty__icon" aria-hidden="true">◉</div>
          <p className="stream-empty__title">Share your route to begin</p>
          <p className="stream-empty__description">You will get a best-flight callout, destination weather, and packing guidance in one view.</p>
        </div>
      )}
    </div>
  );
}
