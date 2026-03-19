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

  const steps: string[] = [];

  if (best) {
    steps.push(
      `Selected ${best.airline} ${best.flight_no} because it offered the best overall balance for this route.`
    );

    if (best.stops === 0 || String(best.stops).trim() === "0") {
      steps.push("Non-stop routing reduced transfer risk and kept the trip simpler.");
    } else {
      steps.push(`Itinerary tradeoff: ${best.stops} stop(s) in exchange for stronger overall value.`);
    }

    if (typeof best.duration_min === "number") {
      steps.push(`Total travel time of ${best.duration_min} minutes stayed within the strongest-ranked options.`);
    }
  }

  const filtersApplied = debugInfo.filters_applied;
  if (typeof filtersApplied === "string" && filtersApplied.trim() && filtersApplied !== "no specific filters") {
    steps.push(`Matched preferences considered: ${filtersApplied}.`);
  }

  const rankedCount = debugInfo.ranked_count;
  if (typeof rankedCount === "number" && rankedCount > 0) {
    steps.push(`Compared ${rankedCount} ranked option${rankedCount === 1 ? "" : "s"} before final selection.`);
  }

  if (weather) {
    const condition = weather.condition ? String(weather.condition).toLowerCase() : "";
    if (condition || weather.temperature_c !== undefined) {
      steps.push("Destination weather was checked to finalize comfort and packing guidance.");
    }
  }

  if (!steps.length && effectiveIntent.trip_type) {
    steps.push(`Trip intent aligned to ${String(effectiveIntent.trip_type).toLowerCase()} priorities.`);
  }

  const deduped: string[] = [];
  const seen = new Set<string>();
  for (const step of steps) {
    const normalized = step.trim().toLowerCase();
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    deduped.push(step.trim());
  }

  return deduped.slice(0, 4);
}

function isWeatherStep(step: string): boolean {
  const normalized = step.toLowerCase();
  return (
    normalized.includes("weather") ||
    normalized.includes("temperature") ||
    normalized.includes("rain") ||
    normalized.includes("snow") ||
    normalized.includes("humidity") ||
    normalized.includes("wind") ||
    normalized.includes("precip")
  );
}

function isSelectionStep(step: string): boolean {
  const normalized = step.toLowerCase();
  return (
    normalized.includes("selected") ||
    normalized.includes("best overall") ||
    normalized.includes("strongest overall")
  );
}

function isRoutingStep(step: string): boolean {
  const normalized = step.toLowerCase();
  return (
    normalized.includes("non-stop routing") ||
    normalized.includes("transfer risk") ||
    normalized.includes("trip simpler")
  );
}

function mergeReasoning(liveSteps: string[], finalSteps: string[]): string[] {
  const merged: string[] = [];
  const seen = new Set<string>();
  let weatherStepIncluded = false;
  let selectionStepIncluded = false;
  let routingStepIncluded = false;
  for (const step of [...liveSteps, ...finalSteps]) {
    const normalized = step.trim().toLowerCase();
    if (!normalized || seen.has(normalized)) continue;
    if (isWeatherStep(normalized)) {
      if (weatherStepIncluded) continue;
      weatherStepIncluded = true;
    }
    if (isSelectionStep(normalized)) {
      if (selectionStepIncluded) continue;
      selectionStepIncluded = true;
    }
    if (isRoutingStep(normalized)) {
      if (routingStepIncluded) continue;
      routingStepIncluded = true;
    }
    seen.add(normalized);
    merged.push(step.trim());
  }
  return merged.slice(0, 4);
}

export default function AIReasoningPanel({ finalJson, isStreaming, reasoningSteps = [] }: Props) {
  const finalReasoning = extractReasoning(finalJson);
  const visibleReasoning = mergeReasoning(reasoningSteps, finalReasoning);
  const showFooter = isStreaming || visibleReasoning.length === 0;

  return (
    <div className="reasoning-panel">
      <h3 className="reasoning-title">Why this recommendation</h3>

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
