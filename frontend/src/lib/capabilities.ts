import type { FeatureCapability } from "./types";

export const FEATURE_CAPABILITIES: FeatureCapability[] = [
  {
    id: "flight-search",
    title: "Real-time flight search",
    description: "Live flight search with ranking, filtering, and best-option selection.",
    status: "live",
  },
  {
    id: "weather-forecast",
    title: "Destination weather forecast",
    description:
      "Weather guidance is included for in-range travel dates; distant dates can return estimated fallback guidance.",
    status: "partial",
  },
  {
    id: "streaming",
    title: "Streaming AI responses",
    description:
      "Live response updates are available, with automatic continuity fallback if connection quality drops.",
    status: "partial",
  },
  {
    id: "booking-handoff",
    title: "Booking handoff links",
    description:
      "Booking handoff is available for supported results, with fallback links used when direct resolution is unavailable.",
    status: "partial",
  },
  {
    id: "booking-actions",
    title: "Booking actions in chat",
    description:
      "Action requests can be handled in chat language today; dedicated in-product controls are still evolving.",
    status: "partial",
    note: "Chat-driven flow for now.",
  },
  {
    id: "price-tracking-chat",
    title: "Price tracking in chat",
    description:
      "Price monitoring can be started in chat, while alert management UI remains limited.",
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
    description: "Multi-city live planning is available in limited form and is being expanded.",
    status: "coming-soon",
    note: "Under development",
  },
  {
    id: "mid-stream-failover",
    title: "Mid-stream provider failover",
    description: "Automatic mid-plan continuity across providers is still under development.",
    status: "coming-soon",
    note: "Under development",
  },
];
