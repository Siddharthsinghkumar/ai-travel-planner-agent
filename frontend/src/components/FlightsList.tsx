import FlightCard from "./FlightCard";
import type { Flight } from "../lib/types";

type FlightsListProps = {
  flights?: Flight[];
  bestFlight?: Flight;
  isLoading?: boolean;
};

function flightIdentity(flight: Flight) {
  return [
    flight.airline ?? "",
    flight.flight_no ?? "",
    flight.departure_time ?? "",
    flight.arrival_time ?? "",
    String(flight.price_inr ?? ""),
  ].join("|");
}

function isBestFlight(flight: Flight, bestFlight?: Flight) {
  const flightRecord = flight as Flight & Record<string, unknown>;
  if (flightRecord?.is_best || flightRecord?.ai_recommended || flightRecord?.best_flight) return true;
  if (!bestFlight) return false;

  return (
    flight?.flight_no === bestFlight?.flight_no &&
    flight?.airline === bestFlight?.airline &&
    flight?.departure_time === bestFlight?.departure_time
  );
}

function orderFlightsWithBestFirst(flights: Flight[], bestFlight?: Flight): Flight[] {
  if (flights.length <= 1) return flights;

  const bestIndex = flights.findIndex((flight) => isBestFlight(flight, bestFlight));
  if (bestIndex === 0) return flights;
  if (bestIndex > 0) {
    const best = flights[bestIndex];
    return [best, ...flights.slice(0, bestIndex), ...flights.slice(bestIndex + 1)];
  }

  if (!bestFlight) return flights;

  // Fallback: if best flight is provided separately, pin it first and deduplicate by identity.
  const combined = [bestFlight, ...flights];
  const seen = new Set<string>();
  const ordered: Flight[] = [];
  for (const flight of combined) {
    const key = flightIdentity(flight);
    if (seen.has(key)) continue;
    seen.add(key);
    ordered.push(flight);
  }
  return ordered;
}

export default function FlightsList({ flights, bestFlight, isLoading = false }: FlightsListProps) {
  if (isLoading) {
    return (
      <div className="flights-shimmer" aria-label="Loading flights">
        <div className="flight-skeleton-card">
          <div className="flight-skeleton-card__left">
            <div className="shim" style={{ width: "26px", height: "26px" }} />
            <div className="flight-skeleton-card__lines">
              <div className="shim" style={{ width: "68%" }} />
              <div className="shim" style={{ width: "88%", opacity: 0.75 }} />
            </div>
          </div>
          <div className="shim" style={{ width: "82px", height: "16px", marginBottom: 0 }} />
        </div>
        <div className="flight-skeleton-card flight-skeleton-card--muted">
          <div className="flight-skeleton-card__left">
            <div className="shim" style={{ width: "26px", height: "26px" }} />
            <div className="flight-skeleton-card__lines">
              <div className="shim" style={{ width: "60%" }} />
              <div className="shim" style={{ width: "84%", opacity: 0.65 }} />
            </div>
          </div>
          <div className="shim" style={{ width: "72px", height: "16px", marginBottom: 0, opacity: 0.6 }} />
        </div>
        <div className="flight-skeleton-card flight-skeleton-card--muted-2">
          <div className="flight-skeleton-card__left">
            <div className="shim" style={{ width: "26px", height: "26px" }} />
            <div className="flight-skeleton-card__lines">
              <div className="shim" style={{ width: "64%" }} />
              <div className="shim" style={{ width: "80%", opacity: 0.55 }} />
            </div>
          </div>
          <div className="shim" style={{ width: "68px", height: "16px", marginBottom: 0, opacity: 0.55 }} />
        </div>
      </div>
    );
  }

  if (!flights || flights.length === 0) {
    return (
      <div className="empty-state empty-state--flights empty-state--flights-compact">
        <p className="empty-state__title">Search to load ranked flight options.</p>
        <p className="empty-state__hint">Your strongest match will be highlighted first.</p>
      </div>
    );
  }

  const orderedFlights = orderFlightsWithBestFirst(flights, bestFlight);

  return (
    <div className="space-y-2 flights-stack">
      {orderedFlights.map((f, i) => (
        <div
          key={i}
          className="flights-stack__item"
          style={{ animationDelay: `${Math.min(i, 6) * 45}ms` }}
        >
          <FlightCard flight={f} isBest={isBestFlight(f, bestFlight)} />
        </div>
      ))}
    </div>
  );
}
