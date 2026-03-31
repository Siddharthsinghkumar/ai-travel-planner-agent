import { useState } from "react";
import { motion } from "framer-motion";
import type { Flight } from "../lib/types";
import { formatPriceINR } from "../lib/format";
import { resolveApiUrl } from "../lib/api";

export default function FlightCard({
  flight,
  isBest = false,
  rank,
  total,
}: {
  flight: Flight;
  isBest?: boolean;
  rank?: number;
  total?: number;
}) {
  const [copied, setCopied] = useState(false);
  const hasDate = typeof flight.date === "string" && flight.date.trim().length > 0;
  const hasEmissions = typeof flight.carbon_emissions_g === "number";
  const hasHandoff = typeof flight.handoff_url === "string" && flight.handoff_url.trim().length > 0;
  const resolvedHandoffUrl = hasHandoff ? resolveApiUrl(flight.handoff_url as string) : "";
  const stopLabel = Number(flight.stops) === 0 ? "Direct" : String(flight.stops);
  const routeInfo = flight.layover_info
    ? `Layover: ${flight.layover_info}`
    : Number(flight.stops) === 0
      ? "Non-stop routing"
      : "Connecting routing";
  const airlineCode = (flight.flight_no || flight.airline || "FL").split(" ")[0].slice(0, 3).toUpperCase();

  const priceText = formatPriceINR(flight.price_inr);

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    const details = `${flight.airline} ${flight.flight_no} | ${flight.departure_time} -> ${flight.arrival_time} | Price: ${priceText}`;
    navigator.clipboard.writeText(details);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <motion.div
      whileHover={{ y: -2 }}
      transition={{ type: "spring", stiffness: 300, damping: 24 }}
      className={`flight-item ${isBest ? "best-pick" : ""} ${
        isBest ? "flight-card--best" : ""
      }`}
    >
      <div className="airline-ico">{airlineCode}</div>
      <div className="fl-info">
        <div className="flight-item__topline">
          <span className={`flight-rank ${isBest ? "flight-rank--best" : ""}`}>
            {isBest ? "Top pick" : typeof rank === "number" ? `#${rank}` : "Candidate"}
            {typeof total === "number" && total > 1 ? ` of ${total}` : ""}
          </span>
          {isBest && (
            <span className="flight-reco-pill" aria-label="AI recommended flight">
              <span className="flight-reco-pill__star" aria-hidden="true">✦</span>
              AI recommended
            </span>
          )}
          <span className="flight-proof-note">
            {hasHandoff ? "Booking handoff ready" : "Handoff available on supported providers"}
          </span>
        </div>
        <p className="fl-route break-words">
          {flight.flight_no} · {flight.departure_time} → {flight.arrival_time}
        </p>
        <p className="fl-meta break-words">
          {flight.airline} · {routeInfo} · {flight.duration_min} min · {stopLabel}
          {hasEmissions ? ` · ${((flight.carbon_emissions_g as number) / 1000).toFixed(1)}kg CO2` : ""}
          {hasDate ? ` · ${flight.date}` : ""}
        </p>
        <p className="fl-meta break-words">
          Baggage: {flight.baggage || "Check airline"}
        </p>
        <div className="flight-card__actions">
          {hasHandoff && (
            <>
              <a
                href={resolvedHandoffUrl}
                target="_blank"
                rel="noreferrer"
                className="flight-card__link flight-card__link--primary"
                title="Secure booking handoff link"
              >
                Book now
              </a>
              <span className="fl-meta">Provider handoff opens a secure booking flow in a new tab.</span>
            </>
          )}
          <button
            onClick={handleCopy}
            className="flight-card__copy flight-card__copy--secondary"
            aria-label="Copy itinerary details"
          >
            <span aria-hidden="true" className="flight-card__copy-icon">⧉</span>
            {copied ? "Copied" : "Copy itinerary"}
          </button>
        </div>
      </div>
      {isBest && <div className="best-lbl">Best value</div>}
      <div className="fl-price">{priceText}</div>
    </motion.div>
  );
}
