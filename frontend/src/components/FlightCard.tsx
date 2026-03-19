import { useState } from "react";
import { motion } from "framer-motion";
import type { Flight } from "../lib/types";
import { formatPriceINR } from "../lib/format";

export default function FlightCard({
  flight,
  isBest = false
}: {
  flight: Flight;
  isBest?: boolean;
}) {
  const [copied, setCopied] = useState(false);
  const hasDate = typeof flight.date === "string" && flight.date.trim().length > 0;
  const hasEmissions = typeof flight.carbon_emissions_g === "number";
  const hasHandoff = typeof flight.handoff_url === "string" && flight.handoff_url.trim().length > 0;
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
            <a
              href={flight.handoff_url}
              target="_blank"
              rel="noreferrer"
              className="flight-card__link"
            >
              Open Booking
            </a>
          )}
          <button
            onClick={handleCopy}
            className="flight-card__copy"
            aria-label="Copy flight details"
          >
            <span aria-hidden="true" className="flight-card__copy-icon">⧉</span>
            {copied ? "Copied" : "Copy details"}
          </button>
        </div>
      </div>
      {isBest && <div className="best-lbl">Best value</div>}
      <div className="fl-price">{priceText}</div>
    </motion.div>
  );
}
