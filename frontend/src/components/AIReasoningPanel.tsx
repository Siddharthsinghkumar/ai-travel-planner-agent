import type { TripPlan } from "../lib/types";

type Props = {
  finalJson: TripPlan | null;
  isStreaming: boolean;
  reasoningSteps?: string[];
};

function extractReasoning(finalJson: TripPlan | null): string[] {
  if (!finalJson) return [];

  const debugInfo = (finalJson.debug_info ?? {}) as Record<string, unknown>;
  const effectiveIntent = (debugInfo.effective_intent ?? {}) as Record<string, unknown>;
  const best = finalJson.best_flight;
  const weather = finalJson.weather && typeof finalJson.weather === "object"
    ? (finalJson.weather as Record<string, unknown>)
    : null;
  const legs = Array.isArray(finalJson.legs) ? finalJson.legs : [];

  const steps: string[] = [];

  if (best) {
    const nonstop = best.stops === 0 || String(best.stops).trim() === "0";
    const speedText = typeof best.duration_min === "number" ? `${best.duration_min} min total travel time` : "a stronger total travel time";
    steps.push(
      nonstop
        ? `We chose ${best.airline} ${best.flight_no} because it's non-stop with ${speedText}, which keeps the trip simpler for a similar fare band.`
        : `We chose ${best.airline} ${best.flight_no} because it balanced fare and timing better than the alternatives for this route.`
    );
  }

  if (weather) {
    const condition = weather.condition ? String(weather.condition).toLowerCase() : "";
    const temp = weather.temperature_c;
    const tempNum = typeof temp === "number" ? temp : Number(temp);
    if (condition.includes("rain")) {
      steps.push("Packing advice prioritizes light rain protection because the destination forecast shows wet conditions.");
    } else if (Number.isFinite(tempNum) && tempNum >= 32) {
      steps.push("Packing advice prioritizes breathable clothing and hydration because the destination forecast is hot.");
    } else if (condition || Number.isFinite(tempNum)) {
      steps.push("Packing advice is based on the destination forecast so comfort and weather risk are covered before booking.");
    }
  }

  if (!steps.length && effectiveIntent.trip_type) {
    steps.push(`This recommendation follows your ${String(effectiveIntent.trip_type).toLowerCase()} travel intent and route constraints.`);
  }

  if (!steps.length && legs.length > 0) {
    steps.push("This itinerary was split into route legs and each leg was optimized for practical timing and fare balance.");
  }

  return steps.slice(0, 2);
}

function normalizeLiveReasoning(step: string): string {
  const normalized = step.trim().toLowerCase();
  if (!normalized) return "";
  if (normalized.includes("weather") || normalized.includes("temperature") || normalized.includes("rain")) {
    return "Packing guidance is being tuned using the latest destination weather conditions.";
  }
  return "Route ranking is balancing travel time, stop count, and fare so the final pick stays practical.";
}

export default function AIReasoningPanel({ finalJson, isStreaming, reasoningSteps = [] }: Props) {
  const finalReasoning = extractReasoning(finalJson);
  const visibleReasoning = finalReasoning.length > 0
    ? finalReasoning
    : Array.from(new Set(reasoningSteps.map(normalizeLiveReasoning).filter(Boolean))).slice(0, 2);
  const showFooter = isStreaming || visibleReasoning.length === 0;

  return (
    <div className="reasoning-panel">
      <h3 className="reasoning-title">Selection evidence</h3>

      {visibleReasoning.length > 0 ? (
        <ol className="reasoning-list">
          {visibleReasoning.map((step, index) => (
            <li
              key={`${index}-${step.slice(0, 20)}`}
              className="reasoning-list__item"
            >
              {step}
            </li>
          ))}
        </ol>
      ) : (
        <div className="empty-state empty-state--reasoning">
          {isStreaming ? (
            <div className="reasoning-wait">
              <div className="shim" style={{ width: "78%" }} />
              <div className="shim" style={{ width: "64%" }} />
              <p className="empty-state__title">Ranking route trade-offs and timing fit...</p>
            </div>
          ) : (
            <p className="empty-state__title">Run a search to see why this option wins.</p>
          )}
        </div>
      )}

      {showFooter ? (
        <div className="reasoning-foot">
          {isStreaming ? "Evaluating top route trade-offs..." : "Reasoning updates appear as soon as results start."}
        </div>
      ) : null}
    </div>
  );
}
