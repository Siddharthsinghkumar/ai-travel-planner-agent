import { useState } from "react";
import { motion } from "framer-motion";
import type { BookingResolveState, Flight } from "../lib/types";
import { formatPriceINR } from "../lib/format";
import { resolveApiUrl } from "../lib/api";

export default function FlightCard({
  flight,
  isBest = false,
  rank,
  total,
  onBook,
  bookingResolveState,
  bookBlockedReason,
  onHold,
  onTrack,
  actionDisabled = false,
}: {
  flight: Flight;
  isBest?: boolean;
  rank?: number;
  total?: number;
  onBook?: (flight: Flight) => Promise<{ handoff_url?: string | null; message?: string } | null | void>;
  bookingResolveState?: BookingResolveState;
  bookBlockedReason?: string;
  onHold?: (flight: Flight) => void;
  onTrack?: (flight: Flight) => void;
  actionDisabled?: boolean;
}) {
  const [copied, setCopied] = useState(false);
  const hasDate = typeof flight.date === "string" && flight.date.trim().length > 0;
  const hasEmissions = typeof flight.carbon_emissions_g === "number";
  const hasHandoff = typeof flight.handoff_url === "string" && flight.handoff_url.trim().length > 0;
  const hasBookingToken = typeof flight.booking_token === "string" && flight.booking_token.trim().length > 0;
  const resolvedHandoffUrl = hasHandoff ? resolveApiUrl(flight.handoff_url as string) : "";
  const handoffMeta = flight.booking_handoff;
  const bookingExitQuality = String(handoffMeta?.booking_exit_quality || "").toLowerCase();
  const sellerName = Array.isArray(flight.booking_sellers) && flight.booking_sellers.length > 0 ? flight.booking_sellers[0] : "";
  const resolvePermanentlyBlocked = bookingResolveState?.status === "failed" && bookingResolveState?.retryable === false;
  const canResolveOnBookClick = Boolean(onBook) && !hasHandoff && hasBookingToken && !resolvePermanentlyBlocked;
  // When streaming (actionDisabled) and booking_token not yet available, show a placeholder button
  // so the user sees "Book now" immediately rather than a permanent-looking "unavailable" message.
  const canShowPendingBookButton = Boolean(onBook) && !hasHandoff && !hasBookingToken && !resolvePermanentlyBlocked && actionDisabled;
  const isBookResolving = bookingResolveState?.status === "resolving";
  const bookError = bookingResolveState?.status === "failed" ? bookingResolveState?.message || null : null;
  const bookInfo = bookingResolveState?.status === "resolved" ? bookingResolveState?.message || null : null;
  const blockedCategory = bookingResolveState?.blocked_category ?? null;
  // Hold/Track should only appear after booking state is determined.
  // A state is "determined" when we have a resolved/failed resolve state,
  // or when the flight already has a handoff_url or a definitive booking_handoff status.
  const bookingStateDetermined =
    bookingResolveState?.status === "resolved" ||
    bookingResolveState?.status === "failed" ||
    hasHandoff ||
    bookingExitQuality === "booking_ready" ||
    bookingExitQuality === "unavailable" ||
    bookingExitQuality === "deferred";
  const bookingActionLabel =
    isBookResolving
      ? "Resolving..."
      : bookingResolveState?.status === "failed"
        ? bookingResolveState?.retryable
          ? "Retry"
          : blockedCategory === "allowlist_policy"
            ? "Policy blocked"
            : "Unavailable"
        : sellerName && hasHandoff
      ? `Book with ${sellerName}`
      : hasHandoff
        ? "Book now"
        : canResolveOnBookClick
          ? "Book now"
          : !hasBookingToken
            ? "No provider token"
            : "Unavailable";
  // Classify booking state into 3 user-friendly categories:
  // 1. Ready — checkout link available or can be fetched on click
  // 2. Unavailable — provider cannot book this row (permanent)
  // 3. Resolving — in-flight resolution
  const proofNote =
    isBookResolving
      ? "Resolving checkout link..."
      : bookingResolveState?.status === "resolved"
        ? "Ready to book"
        : bookingResolveState?.status === "failed"
          ? blockedCategory === "allowlist_policy"
            ? "Booking blocked by policy"
            : blockedCategory === "provider_key_exhausted"
              ? "Provider quota exhausted — try again later"
              : blockedCategory === "provider_unavailable" && !bookingResolveState?.retryable
                ? "Provider does not support booking"
                : bookingResolveState?.retryable
                  ? "Temporary issue — retry available"
                  : "Booking unavailable"
          : hasHandoff || bookingExitQuality === "booking_ready"
            ? "Ready to book"
            : bookingExitQuality === "deferred" && hasBookingToken
              ? "Click Book to fetch checkout link"
              : hasBookingToken
                ? "Click Book to check availability"
                : actionDisabled
                  ? "Booking details loading..."
                  : "No booking token from provider";
  const bookingUnavailableReason =
    resolvePermanentlyBlocked
      ? bookingResolveState?.message || "Provider does not support booking for this row."
      : !hasBookingToken
      ? "No booking token from provider for this flight option."
      : !onBook
        ? bookBlockedReason || "Booking requires route/date context — resubmit your search."
        : bookingExitQuality === "deferred"
        ? "Supported — click Book to fetch the checkout link from the provider."
        : "Provider handoff not yet available. Click Book to attempt resolution.";
  const marketedAs = Array.isArray(flight.marketed_as) ? flight.marketed_as.join(", ") : "";
  const separateTicketsWarning = Boolean(flight.separate_tickets);
  const stopLabel = Number(flight.stops) === 0 ? "Direct" : String(flight.stops);
  const routeInfo = flight.layover_info
    ? `Layover: ${flight.layover_info}`
    : Number(flight.stops) === 0
      ? "Non-stop routing"
      : "Connecting routing";
  const airlineCode = (flight.flight_no || flight.airline || "FL").split(" ")[0].slice(0, 3).toUpperCase();
  const airlineLogoUrl = typeof flight.airline_logo === "string" && flight.airline_logo.trim().length > 0
    ? flight.airline_logo.trim()
    : null;

  const priceText = formatPriceINR(flight.price_inr);
  const hasValidPrice = typeof flight.price_inr === "number" && flight.price_inr > 0 && !flight.price_unavailable;
  const isStreamingPhase = actionDisabled;
  const showPlaceholderPrice = !hasValidPrice && isStreamingPhase;

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    const details = `${flight.airline} ${flight.flight_no} | ${flight.departure_time} -> ${flight.arrival_time} | Price: ${priceText}`;
    navigator.clipboard.writeText(details);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleLazyBook = async () => {
    if (!onBook || !canResolveOnBookClick || isBookResolving) return;
    const pendingTab = window.open("about:blank", "_blank");
    const renderPendingTab = (title: string, html: string) => {
      if (!pendingTab || pendingTab.closed) return;
      try {
        pendingTab.opener = null;
        pendingTab.document.title = title;
        pendingTab.document.body.innerHTML = html;
      } catch {
        // Ignore cross-browser placeholder rendering differences.
      }
    };
    if (pendingTab) {
      renderPendingTab(
        "Resolving booking handoff",
        '<main style="font-family: ui-sans-serif, system-ui, sans-serif; padding: 24px; color: #1f2937;">' +
          '<h2 style="margin: 0 0 8px 0; font-size: 18px;">Resolving booking handoff</h2>' +
          '<p style="margin: 0;">Fetching a secure provider checkout URL. This tab will redirect automatically when ready.</p>' +
        "</main>"
      );
    }
    try {
      const response = await onBook(flight);
      const handoffUrl = typeof response?.handoff_url === "string" ? response.handoff_url.trim() : "";
      if (!handoffUrl) {
        throw new Error(response?.message || "Provider handoff is unavailable for this itinerary.");
      }
      const resolvedUrl = resolveApiUrl(handoffUrl);
      if (pendingTab && !pendingTab.closed) {
        pendingTab.location.replace(resolvedUrl);
      } else {
        window.location.assign(resolvedUrl);
      }
    } catch {
      // Close the pending tab on failure — inline resolve state already shows the error in the card.
      try { pendingTab?.close(); } catch { /* ignore */ }
    }
  };

  return (
    <motion.div
      whileHover={{ y: -2 }}
      transition={{ type: "spring", stiffness: 300, damping: 24 }}
      className={`flight-item ${isBest ? "best-pick" : ""} ${
        isBest ? "flight-card--best" : ""
      }`}
      data-testid={isBest ? "flight-card-best" : "flight-card"}
    >
      {airlineLogoUrl ? (
        <img className="airline-ico" src={airlineLogoUrl} alt={flight.airline} loading="lazy" />
      ) : (
        <div className="airline-ico">{airlineCode}</div>
      )}
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
            {proofNote}
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
        {marketedAs && <p className="fl-meta break-words">Marketed as: {marketedAs}</p>}
        {separateTicketsWarning && (
          <p className="fl-meta break-words">Separate tickets may apply. Re-check baggage and transfer rules.</p>
        )}
        <div className="flight-card__actions">
          {hasHandoff && (
            <>
              <a
                href={resolvedHandoffUrl}
                target="_blank"
                rel="noreferrer"
                className="flight-card__link flight-card__link--primary"
                title="Secure booking handoff link"
                data-testid="booking-link"
              >
                {bookingActionLabel}
              </a>
              {sellerName && (
                <span className="fl-meta" data-testid="booking-seller">
                  Seller: {sellerName}
                </span>
              )}
              <span className="fl-meta" data-testid="provider-handoff-hint">
                Provider handoff opens a secure booking flow in a new tab.
              </span>
            </>
          )}
          {!hasHandoff && canResolveOnBookClick && (
            <>
              <button
                type="button"
                className="flight-card__link flight-card__link--primary"
                onClick={handleLazyBook}
                disabled={actionDisabled || isBookResolving}
                data-testid="booking-resolve-button"
              >
                {bookingActionLabel}
              </button>
              <span className="fl-meta" data-testid="provider-handoff-hint">
                Resolves provider handoff only for this selected row.
              </span>
            </>
          )}
          {!hasHandoff && canShowPendingBookButton && (
            <button
              type="button"
              className="flight-card__link flight-card__link--primary"
              disabled={true}
              data-testid="booking-resolve-button-pending"
            >
              Book now
            </button>
          )}
          {!hasHandoff && !canResolveOnBookClick && !canShowPendingBookButton && (
            <span className="fl-meta" data-testid="checkout-unavailable-note">
              {bookingUnavailableReason}
            </span>
          )}
          {bookInfo && (
            <span className="fl-meta" data-testid="booking-resolve-info">
              {bookInfo}
            </span>
          )}
          {bookError && (
            <span className="fl-meta flight-card__booking-error" data-testid="booking-resolve-error">
              {bookError}
              {bookingResolveState?.retryable ? " You can retry this row." : ""}
            </span>
          )}
          {(onHold || onTrack) && bookingStateDetermined && (
            <div className="flight-card__actions-row">
              {onHold && (
                <button
                  onClick={() => onHold(flight)}
                  className="flight-card__link flight-card__link--secondary"
                  disabled={actionDisabled}
                  data-testid="action-hold"
                >
                  Hold
                </button>
              )}
              {onTrack && (
                <button
                  onClick={() => onTrack(flight)}
                  className="flight-card__link flight-card__link--secondary"
                  disabled={actionDisabled}
                  data-testid="action-track"
                >
                  Track price
                </button>
              )}
            </div>
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
      <div className={`fl-price ${!hasValidPrice ? "fl-price--unavailable" : ""}`}>
        {hasValidPrice ? priceText : showPlaceholderPrice ? "Loading price..." : "Price unavailable"}
      </div>
    </motion.div>
  );
}
