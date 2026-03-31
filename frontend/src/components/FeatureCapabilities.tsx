import type { FeatureCapability } from "../lib/types";
import { IS_PREVIEW_UI } from "../lib/uiMode";

type Props = {
  items: FeatureCapability[];
};

function getPreviewDescription(item: FeatureCapability) {
  const map: Record<string, string> = {
    "flight-search": "Search and compare live flight options; response time varies by provider availability.",
    "weather-forecast": "Check destination weather guidance when forecast data is available.",
    streaming: "Follow live planning updates as recommendations are prepared.",
    "booking-handoff": "Open booking links when a provider handoff is available.",
    "booking-actions": "Use explicit chat phrases for basic confirm/cancel/hold actions.",
    "price-tracking-chat": "Start tracking via natural-language chat prompts; dedicated alert controls are still limited.",
    "full-itinerary": "Expanded day-by-day trip planning is on the way.",
    "price-alerts-ui": "Dedicated alert management is being designed.",
    "advanced-metrics": "Advanced per-flight insights are planned for a future release.",
    "rich-multi-city-streaming": "Basic two-leg via-stopover planning is available, with richer live leg-by-leg detail still in progress.",
    "mid-stream-failover": "More resilient live updates are being built.",
  };

  return map[item.id] || item.description;
}

function getStatusLabel(status: FeatureCapability["status"]) {
  if (status === "live") return "Live";
  if (status === "partial") return "Live (guided)";
  return "Coming soon";
}

function getStatusIcon(status: FeatureCapability["status"]) {
  if (status === "live") return "●";
  if (status === "partial") return "◐";
  return "○";
}

export default function FeatureCapabilities({ items }: Props) {
  const visibleItems = IS_PREVIEW_UI ? items : items.filter((item) => item.status !== "coming-soon");
  const comingSoonItems = items.filter((item) => item.status === "coming-soon");

  return (
    <section className="capabilities-section section-center">
      <div className="reveal">
        <p className="section-label">{IS_PREVIEW_UI ? "What you can do" : "Capabilities"}</p>
        <h2 className="section-title">{IS_PREVIEW_UI ? "Travel planning features" : "What this travel planner handles today"}</h2>
        <p className="section-sub">
          {IS_PREVIEW_UI
            ? "From search to shortlist, plan with live route intelligence and clear next steps."
            : "These cards reflect the current product surface, including chat-led capabilities and features still in progress."}
        </p>
      </div>

      <div className="features-grid">
        {visibleItems.map((item) => {
          return (
            <article
              key={item.id}
              className={`feat-card feat-card--${item.status} reveal`}
            >
              <div className={`cap-pill cap-pill--${item.status}`}>
                <span className="cap-pill__icon" aria-hidden="true">{getStatusIcon(item.status)}</span>
                {getStatusLabel(item.status)}
              </div>
              <h3 className="feat-title">{item.title}</h3>
              <p className="feat-desc">{getPreviewDescription(item)}</p>
            </article>
          );
        })}
      </div>

      {!IS_PREVIEW_UI && comingSoonItems.length > 0 && (
        <div className="coming-soon-row reveal">
          <div className="cs-label">What&apos;s coming</div>
          <div className="cs-chips">
            {comingSoonItems.map((item) => (
              <div key={item.id} className="cs-chip">
                {item.title} <span className="coming-soon-inline">Under development</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
