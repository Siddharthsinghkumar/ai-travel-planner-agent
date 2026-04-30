import type { Flight } from "./types";

export function formatPriceINR(value: string | number | null | undefined): string {
  if (value === null || value === undefined) return "N/A";

  if (typeof value === "number" && Number.isFinite(value) && value > 0) {
    return `₹${value.toLocaleString("en-IN")}`;
  }

  const raw = String(value).trim();
  if (!raw) return "N/A";

  const lowered = raw.toLowerCase();
  if (lowered === "price unavailable" || lowered === "n/a" || lowered === "na" || lowered === "unknown") {
    return "N/A";
  }

  if (raw.includes("₹")) {
    return raw.replace(/₹+/g, "₹");
  }

  const numeric = Number(raw.replace(/[^\d.-]/g, ""));
  if (Number.isFinite(numeric) && numeric > 0) {
    return `₹${numeric.toLocaleString("en-IN")}`;
  }

  return raw;
}

export function formatFlightSummaryLine(flight: Flight): string {
  return `${flight.airline} ${flight.flight_no} ${flight.departure_time} → ${flight.arrival_time}, ${formatPriceINR(flight.price_inr)}.`;
}

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const cleaned = value.replace(/[^\d.-]/g, "").trim();
    if (!cleaned) return null;
    const numeric = Number(cleaned);
    if (Number.isFinite(numeric)) return numeric;
  }
  return null;
}

export function formatTemperatureC(value: unknown): string {
  const numeric = toNumber(value);
  if (numeric === null) return "N/A";
  return `${numeric.toLocaleString("en-IN", { minimumFractionDigits: 0, maximumFractionDigits: 1 })}°C`;
}

export function formatWeatherDate(value: unknown): string {
  if (!value) return "N/A";
  if (value instanceof Date && !Number.isNaN(value.getTime())) {
    return new Intl.DateTimeFormat("en-GB", { weekday: "short", day: "2-digit", month: "short" }).format(value);
  }

  if (typeof value === "string") {
    const ymd = value.trim().match(/^(\d{4})-(\d{2})-(\d{2})$/);
    if (ymd) {
      const year = Number(ymd[1]);
      const month = Number(ymd[2]);
      const day = Number(ymd[3]);
      const localDate = new Date(year, month - 1, day);
      if (!Number.isNaN(localDate.getTime())) {
        return new Intl.DateTimeFormat("en-GB", { weekday: "short", day: "2-digit", month: "short" }).format(localDate);
      }
    }
  }

  const parsed = new Date(String(value));
  if (Number.isNaN(parsed.getTime())) return String(value);
  return new Intl.DateTimeFormat("en-GB", { weekday: "short", day: "2-digit", month: "short" }).format(parsed);
}
