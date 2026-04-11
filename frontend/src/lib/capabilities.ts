import type { FeatureCapability } from "./types";

export const FEATURE_CAPABILITIES: FeatureCapability[] = [
  {
    id: "flight-search",
    title: "Real-time flight search",
    description:
      "Flight search and best-option ranking are available. Basic route controls are in the form; advanced constraints can be added in chat.",
    status: "partial",
  },
  {
    id: "weather-forecast",
    title: "Destination weather forecast",
    description:
      "Weather guidance is included for forecastable travel dates; farther dates may show limited placeholder guidance.",
    status: "partial",
  },
  {
    id: "streaming",
    title: "Streaming AI responses",
    description:
      "Live response updates are available, with non-stream retry for recoverable interruptions.",
    status: "partial",
  },
  {
    id: "booking-handoff",
    title: "Booking handoff links",
    description:
      "Booking handoff links appear for supported results when a booking URL is available.",
    status: "partial",
  },
  {
    id: "booking-actions",
    title: "Booking actions in chat",
    description:
      "Local hold/cancel actions are available through explicit natural-language prompts in the query box; real booking completion still happens on provider checkout pages.",
    status: "partial",
    note: "Chat-driven flow for now.",
  },
  {
    id: "price-tracking-chat",
    title: "Price tracking in chat",
    description:
      "Price tracking can be initiated with natural-language prompts in the query box; alert management remains limited and is not yet a dedicated guided UI flow.",
    status: "partial",
    note: "Setup is currently chat-led.",
  },
  {
    id: "full-itinerary",
    title: "Full itinerary generation",
    description: "Complete day-by-day itinerary planning is still in development.",
    status: "coming-soon",
    note: "Under development",
  },
  {
    id: "price-alerts-ui",
    title: "Dedicated price drop alerts",
    description: "A dedicated alert management experience is still in development.",
    status: "coming-soon",
    note: "Under development",
  },
  {
    id: "advanced-metrics",
    title: "Per-flight score and trend widgets",
    description: "Richer per-flight score, trend, and context indicators are still being refined.",
    status: "coming-soon",
    note: "Under development",
  },
  {
    id: "rich-multi-city-streaming",
    title: "Rich multi-city streaming",
    description: "Basic two-leg via-stopover itineraries are available now; richer per-leg live streaming is still under development.",
    status: "partial",
    note: "Included in limited form.",
  },
  {
    id: "mid-stream-failover",
    title: "Mid-stream provider failover",
    description: "Automatic mid-plan continuity across providers is still under development.",
    status: "coming-soon",
    note: "Under development",
  },
];
