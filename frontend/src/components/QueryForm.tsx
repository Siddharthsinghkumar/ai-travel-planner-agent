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
  onRecentQueriesChange?: (queries: string[]) => void;
  devRoutingOverrides?: DevRoutingOverrides | null;
};

export default function QueryForm({
  onSubmit,
  disabled,
  onRecentQueriesChange,
  devRoutingOverrides = null,
}: Props) {
  const [query, setQuery] = useState("Find cheap flight Delhi to Mumbai tomorrow");
  const [origin, setOrigin] = useState("");
  const [destination, setDestination] = useState("");
  const [date, setDate] = useState("");
  const [tripType, setTripType] = useState<"one-way" | "round-trip" | "via-stopover">("one-way");
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

  function handleSubmit(e?: FormEvent) {
    if (e) e.preventDefault();

    const payload: AskPayload = {};
    const finalQuery = query.trim();

    if (finalQuery) {
      payload.user_query = finalQuery;

      const updatedRecent = [finalQuery, ...recentQueries.filter((q) => q !== finalQuery)].slice(0, 5);
      setRecentQueries(updatedRecent);
      localStorage.setItem("recent_queries", JSON.stringify(updatedRecent));
    }

    if (origin.trim()) payload.origin = origin.trim();
    if (destination.trim()) payload.destination = destination.trim();
    if (date.trim()) payload.date = date;
    if (tripType) payload.trip_type = tripType;

    if (devRoutingOverrides?.llm_mode) {
      payload.llm_mode = devRoutingOverrides.llm_mode;
      if (devRoutingOverrides.cloud_provider) {
        payload.cloud_provider = devRoutingOverrides.cloud_provider;
      }
    }

    onSubmit(payload);
  }

  return (
    <form onSubmit={handleSubmit} className="glass-form planner-form">
      <div className="trip-tabs">
        <button
          type="button"
          onClick={() => setTripType("one-way")}
          className={`trip-tab ${tripType === "one-way" ? "active" : ""}`}
        >
          One-way
        </button>
        <button
          type="button"
          onClick={() => setTripType("round-trip")}
          className={`trip-tab ${tripType === "round-trip" ? "active" : ""}`}
        >
          Round-trip
        </button>
        <button
          type="button"
          onClick={() => setTripType("via-stopover")}
          className={`trip-tab ${tripType === "via-stopover" ? "active" : ""}`}
        >
          Via / Stopover
        </button>
      </div>

      <div className="nl-row">
        <span className="nl-icon" aria-hidden="true">↗</span>
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
        />
        <button type="submit" className="nl-send" disabled={disabled} aria-label="Submit query">
          →
        </button>
      </div>

      <div className="fields-row">
        <label className="field-group">
          <span className="field-label">Origin</span>
          <input
            placeholder="Delhi (DEL)"
            value={origin}
            onChange={(e) => setOrigin(e.target.value)}
            className="f-input"
          />
        </label>
        <label className="field-group">
          <span className="field-label">Destination</span>
          <input
            placeholder="Mumbai (BOM)"
            value={destination}
            onChange={(e) => setDestination(e.target.value)}
            className="f-input"
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
              className="date-native"
              lang="en-GB"
            />
          </div>
        </label>
      </div>

      <div className="card-footer min-w-0">
        <button
          type="submit"
          disabled={disabled}
          className="plan-btn"
        >
          Plan my trip →
        </button>
      </div>
    </form>
  );
}
