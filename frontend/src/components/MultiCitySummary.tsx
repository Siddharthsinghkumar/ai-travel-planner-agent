import { formatPriceINR, formatTemperatureC } from "../lib/format";
import type { MultiCityLeg } from "../lib/types";

type Props = {
  legs: MultiCityLeg[];
};

function toIata(value: unknown): string {
  if (typeof value !== "string") return "";
  const trimmed = value.trim().toUpperCase();
  return /^[A-Z]{3}$/.test(trimmed) ? trimmed : "";
}

function toText(value: unknown): string {
  if (typeof value !== "string") return "";
  return value.trim();
}

function endpointLabel(iata: string, city?: string, explicitLabel?: string): string {
  const label = toText(explicitLabel);
  if (label) return label;
  const cityText = toText(city);
  if (cityText && iata) return `${cityText} (${iata})`;
  return iata;
}

function routeLabel(leg: MultiCityLeg, index: number): string {
  const intent = (leg.debug_info?.intent ?? {}) as Record<string, unknown>;
  const routeLabels = (leg.debug_info?.route_labels ?? {}) as Record<string, unknown>;
  const origin = toIata(intent.origin_iata);
  const destination = toIata(intent.destination_iata);
  if (origin && destination) {
    return `${endpointLabel(origin, toText(routeLabels.origin_city), toText(routeLabels.origin_label))} -> ${endpointLabel(
      destination,
      toText(routeLabels.destination_city),
      toText(routeLabels.destination_label)
    )}`;
  }
  return `Leg ${index + 1}`;
}

export default function MultiCitySummary({ legs }: Props) {
  return (
    <div className="space-y-2">
      {legs.map((leg, index) => {
        const weather = leg.weather && typeof leg.weather === "object" ? leg.weather : null;
        const bestFlight = leg.best_flight;
        const weatherBits = weather
          ? [
              typeof weather.condition === "string" ? weather.condition : "",
              formatTemperatureC(weather.temperature_c),
              formatTemperatureC(weather.temp_min_c) !== "N/A" ? `low ${formatTemperatureC(weather.temp_min_c)}` : "",
              formatTemperatureC(weather.temp_max_c) !== "N/A" ? `high ${formatTemperatureC(weather.temp_max_c)}` : "",
            ].filter(Boolean)
          : [];

        return (
          <article key={index} className="flight-item">
            <p className="flight-item__meta">{routeLabel(leg, index)}</p>
            {bestFlight ? (
              <p className="flight-item__summary">
                {bestFlight.airline} {bestFlight.flight_no} · {bestFlight.departure_time} {"->"} {bestFlight.arrival_time} · {formatPriceINR(bestFlight.price_inr)}
              </p>
            ) : (
              <p className="flight-item__summary">No flight details available for this leg.</p>
            )}
            {weatherBits.length > 0 ? (
              <p className="flight-item__meta">Weather: {weatherBits.join(", ")}</p>
            ) : (
              <p className="flight-item__meta">Weather: N/A</p>
            )}
          </article>
        );
      })}
    </div>
  );
}
