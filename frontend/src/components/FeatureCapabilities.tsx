import type { FeatureCapability } from "../lib/types";
import { IS_PREVIEW_UI } from "../lib/uiMode";

type Props = {
  items: FeatureCapability[];
};

function getPreviewDescription(item: FeatureCapability) {
  const map: Record<string, string> = {
    "flight-search": "Search and compare relevant flight options in seconds.",
    "weather-forecast": "Check destination weather guidance while planning your trip.",
    streaming: "Follow live planning updates as recommendations are prepared.",
    "booking-handoff": "Open booking links when a provider handoff is available.",
    "booking-actions": "Use chat prompts to update booking-related steps.",
    "price-tracking-chat": "Ask the planner to monitor pricing through chat guidance.",
    "full-itinerary": "Expanded day-by-day trip planning is on the way.",
    "price-alerts-ui": "Dedicated alert management is being designed.",
    "advanced-metrics": "Advanced per-flight insights are planned for a future release.",
    "rich-multi-city-streaming": "Richer multi-city planning views are coming soon.",
    "mid-stream-failover": "More resilient live updates are being built.",
  };

  return map[item.id] || item.description;
}

function getStatusLabel(status: FeatureCapability["status"]) {
  if (status === "live") return "Live";
  if (status === "partial") return "Partial";
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
        <p className="section-label">{IS_PREVIEW_UI ? "What you can do" : "What we offer"}</p>
        <h2 className="section-title">{IS_PREVIEW_UI ? "Travel planning features" : "Planner capabilities"}</h2>
        <p className="section-sub">
          {IS_PREVIEW_UI
            ? "From search to shortlist, plan with real-time insights and clear next steps."
            : "Built for clear planning: live essentials now, advanced capabilities rolling out."}
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
              <p className="feat-desc">{IS_PREVIEW_UI ? getPreviewDescription(item) : item.description}</p>
              {!IS_PREVIEW_UI && item.note && <p className="cap-note">{item.note}</p>}
            </article>
          );
        })}
      </div>

      {!IS_PREVIEW_UI && comingSoonItems.length > 0 && (
        <div className="coming-soon-row reveal">
          <div className="cs-label">Planned features</div>
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
