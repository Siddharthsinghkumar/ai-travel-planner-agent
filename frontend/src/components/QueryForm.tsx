import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import type { AskPayload, LLMMode } from "../lib/types";

type DevRoutingOverrides = {
  llm_mode?: LLMMode;
  cloud_provider?: string;
};

type Props = {
  onSubmit: (payload: AskPayload) => void;
  disabled: boolean;
  resultVersion?: number;
  onRecentQueriesChange?: (queries: string[]) => void;
  devRoutingOverrides?: DevRoutingOverrides | null;
  asyncMode?: boolean;
  onAsyncModeChange?: (next: boolean) => void;
};

export default function QueryForm({
  onSubmit,
  disabled,
  resultVersion = 0,
  onRecentQueriesChange,
  devRoutingOverrides = null,
  asyncMode = false,
  onAsyncModeChange,
}: Props) {
  const VIA_STOPOVER_INSTRUCTION_RE = /\b(via|stopover|stop over|through|connecting through|with stop in|stop in)\b/i;
  const [query, setQuery] = useState("Find cheap flight Delhi to Mumbai tomorrow");
  const [origin, setOrigin] = useState("");
  const [destination, setDestination] = useState("");
  const [stopover, setStopover] = useState("");
  const [date, setDate] = useState("");
  const [returnDate, setReturnDate] = useState("");
  const [directOnly, setDirectOnly] = useState(false);
  const [cabin, setCabin] = useState<"any" | "economy" | "premium" | "business" | "first">("any");
  const [baggagePref, setBaggagePref] = useState<"any" | "hand" | "checked">("any");
  const [tripType, setTripType] = useState<"one-way" | "round-trip" | "via-stopover">("one-way");
  const [tabChoiceEverExplicit, setTabChoiceEverExplicit] = useState(false);
  const [manualTabChangedSinceLastResult, setManualTabChangedSinceLastResult] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [recentQueries, setRecentQueries] = useState<string[]>(() => {
    if (typeof window === "undefined") return [];
    const saved = localStorage.getItem("recent_queries");
    if (!saved) return [];
    try {
      const parsed = JSON.parse(saved);
      return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === "string") : [];
    } catch {
      return [];
    }
  });

  const textRef = useRef<HTMLTextAreaElement>(null);

  const QUERY_MIN_HEIGHT = 52;
  const QUERY_MAX_HEIGHT = 280;
  const datePreview = date
    ? new Date(`${date}T00:00:00`).toLocaleDateString("en-GB")
    : "dd/mm/yyyy";
  const returnDatePreview = returnDate
    ? new Date(`${returnDate}T00:00:00`).toLocaleDateString("en-GB")
    : "dd/mm/yyyy";

  useEffect(() => {
    textRef.current?.focus();
  }, []);

  useEffect(() => {
    onRecentQueriesChange?.(recentQueries);
  }, [onRecentQueriesChange, recentQueries]);

  useEffect(() => {
    const handleSuggestion = (event: Event) => {
      const custom = event as CustomEvent<string>;
      if (typeof custom.detail === "string" && custom.detail.trim()) {
        setQuery(custom.detail.trim());
        textRef.current?.focus();
      }
    };

    window.addEventListener("travelyst:suggest", handleSuggestion as EventListener);
    return () => window.removeEventListener("travelyst:suggest", handleSuggestion as EventListener);
  }, []);

  useEffect(() => {
    const el = textRef.current;
    if (!el) return;

    el.style.height = "0px";
    const nextHeight = Math.min(Math.max(el.scrollHeight, QUERY_MIN_HEIGHT), QUERY_MAX_HEIGHT);
    el.style.height = `${nextHeight}px`;
    el.style.overflowY = el.scrollHeight > QUERY_MAX_HEIGHT ? "auto" : "hidden";
  }, [query]);

  useEffect(() => {
    if (!disabled) setIsSubmitting(false);
  }, [disabled]);

  useEffect(() => {
    setManualTabChangedSinceLastResult(false);
    setTabChoiceEverExplicit(false);
  }, [resultVersion]);

  function inferExplicitTripTypeFromQuery(queryText: string, stopoverText: string): "round-trip" | "via-stopover" | null {
    const q = queryText.toLowerCase();
    if (stopoverText.trim()) return "via-stopover";
    if (VIA_STOPOVER_INSTRUCTION_RE.test(q)) return "via-stopover";
    if (/\b(round[- ]?trip|return(?:ing)?|come back)\b/i.test(q)) return "round-trip";
    return null;
  }

  function handleSubmit(e?: FormEvent) {
    if (e) e.preventDefault();
    setIsSubmitting(true);
    setFormError(null);

    const payload: AskPayload = {};
    let finalQuery = query.trim();
    const queryDerivedTripType = inferExplicitTripTypeFromQuery(finalQuery, stopover);
    const hasCommittedResult = resultVersion > 0;

    let resolvedTripType: "one-way" | "round-trip" | "via-stopover";
    if (hasCommittedResult && manualTabChangedSinceLastResult) {
      resolvedTripType = tripType;
    } else if (queryDerivedTripType) {
      resolvedTripType = queryDerivedTripType;
    } else if (tabChoiceEverExplicit) {
      resolvedTripType = tripType;
    } else {
      resolvedTripType = "one-way";
    }

    if (tripType !== resolvedTripType) {
      setTripType(resolvedTripType);
    }

    if (resolvedTripType === "round-trip" && date.trim() && returnDate.trim() && returnDate < date) {
      setFormError("Return date must be the same day or after the departure date.");
      setIsSubmitting(false);
      return;
    }

    if (resolvedTripType === "via-stopover") {
      const stopoverText = stopover.trim();
      const hasViaInstruction = VIA_STOPOVER_INSTRUCTION_RE.test(finalQuery);
      if (!hasViaInstruction && !stopoverText) {
        setFormError("Enter a stopover city or IATA code to run a via-stopover search.");
        setIsSubmitting(false);
        return;
      }
      if (!hasViaInstruction && !finalQuery) {
        const originText = origin.trim() || "origin";
        const destinationText = destination.trim() || "destination";
        const dateText = date.trim() ? ` on ${date.trim()}` : "";
        finalQuery = `Flight ${originText} to ${destinationText} via ${stopoverText}${dateText}`;
      } else if (!hasViaInstruction && stopoverText) {
        finalQuery = `${finalQuery} via ${stopoverText}`.trim();
      }
    }

    if (resolvedTripType === "round-trip" && returnDate.trim()) {
      const hasReturnHint = /\b(return|returning|come back)\b/i.test(finalQuery);
      if (!hasReturnHint) {
        finalQuery = `${finalQuery} returning on ${returnDate.trim()}`.trim();
      }
      payload.return_date = returnDate.trim();
    }

    if (directOnly && !/\b(direct|nonstop|non-stop)\b/i.test(finalQuery)) {
      finalQuery = `${finalQuery} direct flights only`.trim();
    }
    if (cabin !== "any") {
      const cabinPhrase =
        cabin === "premium" ? "premium economy class" : `${cabin} class`;
      if (!new RegExp(`\\b${cabin.replace("-", " ")}\\b|\\bcabin\\b|\\bclass\\b`, "i").test(finalQuery)) {
        finalQuery = `${finalQuery} ${cabinPhrase}`.trim();
      }
      payload.cabin = cabin;
    }
    if (baggagePref !== "any") {
      const baggagePhrase = baggagePref === "hand" ? "cabin baggage only" : "checked baggage included";
      if (!/\b(baggage|luggage|carry-on|carry on)\b/i.test(finalQuery)) {
        finalQuery = `${finalQuery} ${baggagePhrase}`.trim();
      }
      payload.baggage_pref = baggagePref;
    }
    if (directOnly) {
      payload.direct_only = true;
    }

    if (finalQuery) {
      payload.user_query = finalQuery;

      const updatedRecent = [finalQuery, ...recentQueries.filter((q) => q !== finalQuery)].slice(0, 5);
      setRecentQueries(updatedRecent);
      localStorage.setItem("recent_queries", JSON.stringify(updatedRecent));
    }

    if (origin.trim()) payload.origin = origin.trim();
    if (destination.trim()) payload.destination = destination.trim();
    if (date.trim()) payload.date = date;
    payload.trip_type = resolvedTripType;

    if (devRoutingOverrides?.llm_mode) {
      payload.llm_mode = devRoutingOverrides.llm_mode;
      if (devRoutingOverrides.cloud_provider) {
        payload.cloud_provider = devRoutingOverrides.cloud_provider;
      }
    }

    onSubmit(payload);
  }

  return (
    <form onSubmit={handleSubmit} className="glass-form planner-form" data-testid="planner-form">
      <div className="trip-tabs">
        <button
          type="button"
          onClick={() => {
            setTripType("one-way");
            setTabChoiceEverExplicit(true);
            if (resultVersion > 0) setManualTabChangedSinceLastResult(true);
          }}
          className={`trip-tab ${tripType === "one-way" ? "active" : ""}`}
          data-testid="trip-tab-one-way"
        >
          One-way
        </button>
        <button
          type="button"
          onClick={() => {
            setTripType("round-trip");
            setTabChoiceEverExplicit(true);
            if (resultVersion > 0) setManualTabChangedSinceLastResult(true);
          }}
          className={`trip-tab ${tripType === "round-trip" ? "active" : ""}`}
          data-testid="trip-tab-round-trip"
        >
          Round-trip
        </button>
        <button
          type="button"
          onClick={() => {
            setTripType("via-stopover");
            setTabChoiceEverExplicit(true);
            if (resultVersion > 0) setManualTabChangedSinceLastResult(true);
          }}
          className={`trip-tab ${tripType === "via-stopover" ? "active" : ""}`}
          data-testid="trip-tab-via-stopover"
        >
          Via / Stopover
        </button>
      </div>

      <div className="nl-row">
        <span className="nl-icon" aria-hidden="true">↗</span>
        <div className="nl-main">
          <p className="planner-guidance">Primary input: describe your trip naturally</p>
          <textarea
            ref={textRef}
            placeholder={
              tripType === "via-stopover"
                ? "E.g. Flight Delhi to Goa via Mumbai tomorrow..."
                : tripType === "round-trip"
                  ? "E.g. Round-trip Delhi to Mumbai returning in 3 days..."
                  : "Find cheap flights Delhi to Mumbai tomorrow..."
            }
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            rows={1}
            className="nl-input"
            data-testid="query-input"
          />
        </div>
        <button type="submit" className="nl-send" disabled={disabled} aria-label="Submit query" data-testid="submit-query">
          <span className="nl-send__arrow" aria-hidden="true">→</span>
        </button>
      </div>

      <div className="planner-structured-head">
        <span className="planner-structured-label">Assistive fields</span>
        <span className="planner-structured-note">Optional quick controls. We merge these with your natural-language prompt.</span>
      </div>
      <div className="fields-row">
        <label className="field-group">
          <span className="field-label">Origin</span>
          <input
            placeholder="Delhi (DEL)"
            value={origin}
            onChange={(e) => setOrigin(e.target.value)}
            className="f-input"
            data-testid="input-origin"
          />
        </label>
        <label className="field-group">
          <span className="field-label">Destination</span>
          <input
            placeholder="Mumbai (BOM)"
            value={destination}
            onChange={(e) => setDestination(e.target.value)}
            className="f-input"
            data-testid="input-destination"
          />
        </label>
        <label className="field-group">
          <span className="field-label">Travel date</span>
          <div className="date-shell">
            <span className={`date-display ${date ? "date-display--value" : ""}`}>{datePreview}</span>
            <span className="date-icon" aria-hidden="true">📅</span>
            <input
              type="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              className="date-native f-date"
              lang="en-GB"
              data-testid="input-date"
            />
          </div>
        </label>
        {tripType === "round-trip" && (
          <label className="field-group">
            <span className="field-label">Return date</span>
            <div className="date-shell">
              <span className={`date-display ${returnDate ? "date-display--value" : ""}`}>{returnDatePreview}</span>
              <span className="date-icon" aria-hidden="true">↩</span>
              <input
                type="date"
                value={returnDate}
                onChange={(e) => setReturnDate(e.target.value)}
                className="date-native f-date"
                lang="en-GB"
                data-testid="input-return-date"
              />
            </div>
          </label>
        )}
        {tripType === "via-stopover" && (
          <label className="field-group">
            <span className="field-label">Stopover</span>
            <input
              placeholder="Mumbai (BOM)"
              value={stopover}
              onChange={(e) => setStopover(e.target.value)}
              className="f-input"
              data-testid="input-stopover"
            />
          </label>
        )}
      </div>
      <div className="fields-row">
        <label className="field-group">
          <span className="field-label">Cabin</span>
          <select
            value={cabin}
            onChange={(e) => setCabin(e.target.value as "any" | "economy" | "premium" | "business" | "first")}
            className="f-input"
            data-testid="select-cabin"
          >
            <option value="any">Any cabin</option>
            <option value="economy">Economy</option>
            <option value="premium">Premium economy</option>
            <option value="business">Business</option>
            <option value="first">First</option>
          </select>
        </label>
        <label className="field-group">
          <span className="field-label">Baggage</span>
          <select
            value={baggagePref}
            onChange={(e) => setBaggagePref(e.target.value as "any" | "hand" | "checked")}
            className="f-input"
            data-testid="select-baggage"
          >
            <option value="any">Any</option>
            <option value="hand">Cabin baggage only</option>
            <option value="checked">Checked bag</option>
          </select>
        </label>
        <label className="field-group">
          <span className="field-label">Direct only</span>
          <input
            type="checkbox"
            checked={directOnly}
            onChange={(e) => setDirectOnly(e.target.checked)}
            className="f-input"
            data-testid="toggle-direct-only"
          />
        </label>
      </div>
      {formError && <div className="notice notice--error notice--inline" data-testid="notice-error">{formError}</div>}

      <div className="card-footer min-w-0">
        <div className="async-toggle">
          <label className="inline-flex items-center gap-2 text-xs text-slate-400">
            <input
              type="checkbox"
              checked={asyncMode}
              onChange={(e) => onAsyncModeChange?.(e.target.checked)}
              disabled={disabled}
              data-testid="toggle-async"
            />
            Run in background (async job)
          </label>
          {asyncMode && (
            <span className="form-hint">
              Async jobs run in-process and clear on restart. Keep this tab open until results load.
            </span>
          )}
        </div>
        <button
          type="submit"
          disabled={disabled || isSubmitting}
          className="plan-btn"
        >
          <span className="plan-btn__content">
            {isSubmitting || disabled ? "Planning your trip..." : "Plan my trip →"}
          </span>
        </button>
      </div>
    </form>
  );
}
