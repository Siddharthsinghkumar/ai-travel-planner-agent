import { useEffect, useState } from "react";
import { formatPriceINR, formatTemperatureC } from "../lib/format";
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

function buildPackingTip(weather: Record<string, unknown> | null | undefined): string {
  if (!weather) return "Pack one light layer and keep essentials easy to reach.";

  const condition = String(weather.condition || "").toLowerCase();
  const temp = weather.temperature_c;
  const tempNum = typeof temp === "number" ? temp : Number(temp);

  if (condition.includes("rain")) return "Carry a compact umbrella and a light waterproof layer.";
  if (condition.includes("snow")) return "Pack warm layers, insulated shoes, and gloves.";
  if (Number.isFinite(tempNum) && tempNum >= 32) return "Pack breathable clothing and keep water handy.";
  if (Number.isFinite(tempNum) && tempNum <= 18) return "Bring a light jacket for cooler morning and evening hours.";
  return "Pack a light layer for changing temperatures through the day.";
}

function buildThinkingSummary(
  finalJson: TripPlan | null | undefined,
  fallbackBestFlight?: Flight | null,
  fallbackWeather?: Record<string, unknown> | null
) {
  const best = finalJson?.best_flight ?? fallbackBestFlight ?? null;
  if (!best) return null;

  const weather = (finalJson?.weather && typeof finalJson.weather === "object"
    ? (finalJson.weather as Record<string, unknown>)
    : (fallbackWeather ?? null));

  const weatherCondition = weather?.condition ? String(weather.condition) : "Weather details are being finalized";
  const weatherTemp = weather?.temperature_c;
  const weatherTempText = weatherTemp !== undefined && weatherTemp !== null ? formatTemperatureC(weatherTemp) : null;

  return {
    bestFlight: `Best overall option: ${best.airline} ${best.flight_no}, ${best.departure_time} → ${best.arrival_time}, ${formatPriceINR(best.price_inr)}.`,
    weather: weatherTempText
      ? `${weatherCondition} around ${weatherTempText}.`
      : `${weatherCondition}.`,
    packingTip: buildPackingTip(weather),
  };
}

export default function StreamPane({
  tokens,
  finalText = "",
  finalJson = null,
  fallbackBestFlight = null,
  fallbackWeather = null,
  isStreaming,
  canCancel = true,
  statusText,
  onCancel
}: StreamPaneProps) {
  const hasTokenText = typeof tokens === "string" && tokens.length > 0;
  const hasFinalText = typeof finalText === "string" && finalText.trim().length > 0;
  const summary = buildThinkingSummary(finalJson, fallbackBestFlight, fallbackWeather);
  const hasSummary = Boolean(summary);
  const [showStructuredSummary, setShowStructuredSummary] = useState(false);
  const [hideNarrative, setHideNarrative] = useState(false);
  const isIdle = !isStreaming && !hasTokenText && !hasFinalText && !hasSummary;
  const visibleText = hasTokenText ? tokens : hasFinalText ? finalText : "";
  const hasAnyText = hasTokenText || hasFinalText;
  const shouldShowStreamingText = hasAnyText && (isStreaming || !hasSummary || !hideNarrative);
  const shouldFadeNarrative = hasSummary && !isStreaming && showStructuredSummary;

  useEffect(() => {
    if (!hasSummary || isStreaming) {
      setShowStructuredSummary(false);
      setHideNarrative(false);
      return;
    }

    setHideNarrative(false);
    const showTimer = setTimeout(() => setShowStructuredSummary(true), 180);
    const hideTimer = setTimeout(() => setHideNarrative(true), 420);
    return () => {
      clearTimeout(showTimer);
      clearTimeout(hideTimer);
    };
  }, [hasSummary, isStreaming, finalJson]);

  return (
    <div
      className="stream-pane"
      aria-live="polite"
      aria-busy={isStreaming}
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

      {hasSummary && summary && showStructuredSummary && (
        <div className="thinking-summary thinking-summary--visible">
          <section className="thinking-summary__section thinking-summary__section--best">
            <h4 className="thinking-summary__title">Best flight</h4>
            <p className="thinking-summary__text">{summary.bestFlight}</p>
          </section>
          <section className="thinking-summary__section thinking-summary__section--weather">
            <h4 className="thinking-summary__title">Destination weather</h4>
            <p className="thinking-summary__text">{summary.weather}</p>
          </section>
          <section className="thinking-summary__section thinking-summary__section--packing">
            <h4 className="thinking-summary__title">Packing tip</h4>
            <p className="thinking-summary__text">{summary.packingTip}</p>
          </section>
        </div>
      )}

      {shouldShowStreamingText ? (
        <div
          className={[
            "r-text llm-pane min-w-0 stream-pane__body",
            hasFinalText && !hasTokenText ? "stream-pane__body--final" : "",
            shouldFadeNarrative ? "stream-pane__body--fade" : "",
          ]
            .filter(Boolean)
            .join(" ")}
        >
          {visibleText}
          {isStreaming && hasTokenText && <span className="stream-caret" aria-hidden="true" />}
        </div>
      ) : null}

      {isStreaming && (
        <div className="stream-pane__controls">
          <div role="status" className="min-w-0 break-words stream-pane__status">
            {statusText || (hasTokenText ? "Building your trip summary..." : "Checking flights and destination weather...")}
          </div>

          {canCancel && (
            <button
              onClick={onCancel}
              className="stream-pane__cancel"
              aria-label="Cancel streaming"
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
