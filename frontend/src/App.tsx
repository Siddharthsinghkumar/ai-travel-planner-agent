import { useEffect, useRef, useState } from "react";
import { API_BASE } from "./lib/api";
import { useStreamingPlan } from "./hooks/useStreamingPlan";

import AuroraCanvas from "./components/AuroraCanvas";
import QueryForm from "./components/QueryForm";
import StreamPane from "./components/StreamPane";
import FlightsList from "./components/FlightsList";
import FeatureCapabilities from "./components/FeatureCapabilities";
import WeatherSummary from "./components/WeatherSummary";
import AIReasoningPanel from "./components/AIReasoningPanel";
import FlightsTicker from "./components/FlightsTicker";
import DevRoutingDrawer from "./components/DevRoutingDrawer";
import { IS_PREVIEW_UI } from "./lib/uiMode";
import { FEATURE_CAPABILITIES } from "./lib/capabilities";
import type { AskPayload, Flight, TripPlan, LLMMode, LLMOptionsResponse } from "./lib/types";
import { formatFlightSummaryLine, formatTemperatureC } from "./lib/format";

function stringifyReasoningCandidate(candidate: unknown): string {
  if (typeof candidate === "string" && candidate.trim()) return candidate.trim();
  if (Array.isArray(candidate)) {
    const lines = candidate
      .map((entry) => (typeof entry === "string" ? entry.trim() : String(entry)))
      .filter(Boolean);
    return lines.join("\n");
  }
  if (candidate && typeof candidate === "object") {
    const lines = Object.values(candidate as Record<string, unknown>)
      .map((entry) => (typeof entry === "string" ? entry.trim() : String(entry)))
      .filter(Boolean);
    return lines.join("\n");
  }
  return "";
}

function buildFallbackSummary(finalJson: TripPlan | null): string {
  if (!finalJson) return "";
  const bits: string[] = [];

  if (finalJson.best_flight) {
    const best = finalJson.best_flight;
    bits.push(`Top option: ${formatFlightSummaryLine(best)}`);
  }

  if (finalJson.weather && typeof finalJson.weather === "object") {
    const condition = finalJson.weather.condition;
    const temperature = finalJson.weather.temperature_c;
    const conditionText = typeof condition === "string" ? condition : "";
    const tempText = temperature !== undefined && temperature !== null ? formatTemperatureC(temperature) : "";
    if (conditionText || tempText) {
      bits.push(`Destination weather: ${[conditionText, tempText].filter(Boolean).join(", ")}.`);
    }
  }

  if (bits.length === 0) {
    bits.push("Trip results are ready. Review flights below for the best option.");
  }

  return bits.join(" ");
}

export default function App() {
  const [serverStatus, setServerStatus] = useState<"checking" | "online" | "offline">("checking");
  const [lastPayload, setLastPayload] = useState<AskPayload | null>(null);
  const [llmOptions, setLlmOptions] = useState<LLMOptionsResponse | null>(null);
  const [devDrawerOpen, setDevDrawerOpen] = useState(false);
  const [devLlmMode, setDevLlmMode] = useState<LLMMode>("ollama_first");
  const [devCloudProvider, setDevCloudProvider] = useState<string>("gemini");
  const autoScrolledRef = useRef(false);
  const liveRegionRef = useRef<HTMLElement | null>(null);
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

  const {
    tokens,
    finalJson,
    partialFlights,
    partialBestFlight,
    partialWeather,
    reasoningSteps,
    isStreaming,
    isFallback,
    error,
    start,
    cancel
  } = useStreamingPlan();
  const isDevMode = (() => {
    if (typeof window === "undefined") return false;
    const params = new URLSearchParams(window.location.search);
    return params.get("dev") === "true" || params.get("devmode") === "true";
  })();

  useEffect(() => {
    const checkServer = () => {
      fetch(`${API_BASE}/health`)
        .then((res) => (res.ok ? setServerStatus("online") : setServerStatus("offline")))
        .catch(() => setServerStatus("offline"));
    };

    checkServer();
    const intervalId = setInterval(checkServer, 5000);
    return () => clearInterval(intervalId);
  }, []);

  useEffect(() => {
    if (!isDevMode) {
      setLlmOptions(null);
      setDevDrawerOpen(false);
      return;
    }

    let cancelled = false;

    const fetchLlmOptions = async () => {
      try {
        const resp = await fetch(`${API_BASE}/llm/options`);
        if (!resp.ok) return;
        const data = (await resp.json()) as LLMOptionsResponse;
        if (!cancelled) {
          setLlmOptions(data);
        }
      } catch {
        // Keep defaults when endpoint is unavailable.
      }
    };

    fetchLlmOptions();
    return () => {
      cancelled = true;
    };
  }, [isDevMode]);

  useEffect(() => {
    if (!llmOptions) return;

    if (llmOptions.defaults?.llm_mode) {
      setDevLlmMode(llmOptions.defaults.llm_mode);
    }

    const providerChoices =
      llmOptions.usable_cloud_providers?.length
        ? llmOptions.usable_cloud_providers
        : llmOptions.cloud_providers?.length
          ? llmOptions.cloud_providers
          : ["gemini"];
    const effectiveProvider = llmOptions.effective_default_provider || llmOptions.defaults?.cloud_provider;
    if (effectiveProvider && providerChoices.includes(effectiveProvider)) {
      setDevCloudProvider(effectiveProvider);
    } else if (!providerChoices.includes(devCloudProvider)) {
      setDevCloudProvider(providerChoices[0]);
    }
  }, [llmOptions, devCloudProvider]);

  useEffect(() => {
    if (!isDevMode) return;

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === "d") {
        event.preventDefault();
        setDevDrawerOpen((prev) => !prev);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isDevMode]);

  useEffect(() => {
    const targets = document.querySelectorAll(".reveal");
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
            io.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12 }
    );

    targets.forEach((target) => io.observe(target));
    return () => io.disconnect();
  }, []);

  function handleSubmit(payload: AskPayload) {
    setLastPayload(payload);
    autoScrolledRef.current = false;
    start(payload);
  }

  const debugInfo = (finalJson?.debug_info ?? null) as TripPlan["debug_info"];
  const finalFlightsFromDebug = Array.isArray(debugInfo?.all_flights) ? (debugInfo.all_flights as Flight[]) : undefined;
  const finalFlights = finalFlightsFromDebug || (Array.isArray(finalJson?.all_flights) ? finalJson.all_flights : undefined);
  const flights = finalFlights || partialFlights || undefined;
  const bestFlight = finalJson?.best_flight || partialBestFlight || undefined;
  const hasFlights = Array.isArray(flights) && flights.length > 0;
  const finalRecord = (finalJson ?? null) as Record<string, unknown> | null;
  const reasoningFromDebug = stringifyReasoningCandidate(
    debugInfo?.agent_reasoning ??
      debugInfo?.reasoning ??
      finalRecord?.agent_reasoning ??
      finalRecord?.reasoning
  );
  const finalMessage =
    typeof finalRecord?.llm_response === "string"
      ? finalRecord.llm_response
      : typeof finalRecord?.message === "string"
        ? finalRecord.message
        : typeof finalRecord?.error === "string"
          ? finalRecord.error
          : reasoningFromDebug || buildFallbackSummary(finalJson);
  const finalWeatherData = finalJson?.weather && typeof finalJson.weather === "object" ? finalJson.weather : null;
  const weatherData = finalWeatherData || partialWeather;
  const destinationCode =
    typeof debugInfo?.intent?.destination_iata === "string" ? debugInfo.intent.destination_iata : undefined;
  const isBusy = isStreaming || isFallback;
  const hasTokenContent = tokens.length > 0;
  const hasReasoningContent =
    reasoningSteps.length > 0 || hasTokenContent || finalMessage.trim().length > 0 || reasoningFromDebug.trim().length > 0;
  const hasWeatherContent = Boolean(weatherData);
  const streamLabelTone = hasTokenContent ? "r-label--live" : isBusy ? "r-label--waiting" : "r-label--inactive";
  const weatherLabelTone = hasWeatherContent ? "r-label--live" : isBusy ? "r-label--waiting" : "r-label--inactive";
  const reasoningLabelTone = hasReasoningContent ? "r-label--live" : isBusy ? "r-label--waiting" : "r-label--inactive";
  const suggestions = [
    "Cheapest flight Delhi to Mumbai tomorrow",
    "Direct flights Bangalore to Goa this weekend under ₹5000",
    "Round-trip Delhi to Mumbai returning in 3 days",
    "Eco-friendly flights Mumbai to Bangalore tomorrow",
    "Flight Delhi to Goa via Mumbai tomorrow",
  ];
  const suggestionChips = Array.from(new Set([...recentQueries, ...suggestions])).slice(0, 10);
  const trustNames = ["IndiGo", "Air India", "SpiceJet", "Vistara", "Akasa", "AirAsia"];
  const statusText = IS_PREVIEW_UI
    ? serverStatus === "online"
      ? "● Live"
      : serverStatus === "offline"
        ? "● Limited"
        : "● Connecting"
    : serverStatus === "online"
      ? "Service available"
      : serverStatus === "offline"
        ? "Limited service"
        : "Connecting";
  const heroBadgeText = IS_PREVIEW_UI ? "AI-guided trip planning" : "Smart trip planning";
  const flightsCardClass = [
    "r-card",
    "results-card",
    !hasFlights && !isBusy ? "r-card--compact" : "",
    hasFlights ? "results-card--live" : "",
    isBusy && !hasFlights ? "results-card--loading" : "",
  ]
    .filter(Boolean)
    .join(" ");
  const hasLiveUpdate = tokens.length > 0 || hasFlights || Boolean(weatherData) || reasoningSteps.length > 0;

  useEffect(() => {
    if (!isBusy || !hasLiveUpdate || autoScrolledRef.current) return;
    if (typeof window === "undefined") return;

    if (window.scrollY > 220) return;

    liveRegionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    autoScrolledRef.current = true;
  }, [isBusy, hasLiveUpdate]);

  return (
    <div className="app-shell">
      <AuroraCanvas />

      <div className="page">
        <nav className="top-nav">
          <div className="nav-logo">
            <span className="logo-mark" aria-hidden="true"><span className="logo-glyph">T</span></span>
            Travelyst
          </div>

          <div className="nav-links" aria-label="Primary">
            <a href="#planner">Planner</a>
            <a href="#results">Results</a>
            <a href="#capabilities">Capabilities</a>
          </div>

          <div className="nav-right">
            <div
              className={`api-status ${serverStatus === "online" ? "api-status--online" : "api-status--offline"} ${IS_PREVIEW_UI ? "api-status--preview" : ""}`}
            >
              {statusText}
            </div>
            <a className="btn-primary" href="#planner">
              Start planning →
            </a>
          </div>
        </nav>

        <main className="app-main">
          <section id="planner" className="hero">
            <div className="hero-badge">
              <span className="badge-dot" aria-hidden="true" />
              {heroBadgeText}
            </div>
            <h1 className="hero-title">
              <span className="title-line-1">Your next trip,</span>
              <span className="title-line-2">planned instantly.</span>
            </h1>
            <p className="hero-sub">
              Tell us where you want to go. Get real flights, live weather, and a curated recommendation in one flow.
            </p>

            <section className="hero-grid">
              <div className="hero-left">
                <div className="search-card">
                  <QueryForm
                    onSubmit={handleSubmit}
                    disabled={isBusy}
                    onRecentQueriesChange={setRecentQueries}
                    devRoutingOverrides={
                      isDevMode
                        ? {
                            llm_mode: devLlmMode,
                            cloud_provider: devCloudProvider,
                          }
                        : null
                    }
                  />

                  {error && (
                    <div className="notice notice--error notice--inline">
                      <span className="min-w-0 break-words">We couldn&apos;t finish your plan. {error}</span>
                      {lastPayload && (
                        <button
                          onClick={() => start(lastPayload)}
                          className="notice__retry"
                        >
                          Try again
                        </button>
                      )}
                    </div>
                  )}

                </div>

                <div className="suggestions-row sugg-strip">
                  <div className="sugg-scroll">
                    {suggestionChips.map((item) => (
                      <button
                        key={item}
                        type="button"
                        className="s-chip history-chip"
                        title={item}
                        onClick={() => {
                          window.dispatchEvent(new CustomEvent<string>("travelyst:suggest", { detail: item }));
                        }}
                      >
                        <span className="s-chip__label">{item}</span>
                      </button>
                    ))}
                  </div>
                </div>

                <article
                  ref={(node) => { liveRegionRef.current = node; }}
                  className={[
                    "r-card",
                    "hero-stream-card",
                    hasTokenContent ? "hero-stream-card--live" : "",
                    isBusy && !hasTokenContent ? "hero-stream-card--loading" : "",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                >
                  <div className={`r-label ${streamLabelTone}`}>
                    <span className="r-dot" aria-hidden="true" />
                    AI thinking
                  </div>
                  <StreamPane
                    tokens={tokens}
                    finalText={finalMessage}
                    finalJson={finalJson}
                    fallbackBestFlight={bestFlight}
                    fallbackWeather={weatherData}
                    isStreaming={isBusy}
                    canCancel={isStreaming}
                    onCancel={cancel}
                  />
                </article>
              </div>

              <aside className="hero-right">
                <article
                  className={[
                    "r-card",
                    "support-card",
                    hasWeatherContent ? "support-card--live" : "",
                    isBusy && !hasWeatherContent ? "support-card--loading" : "",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                >
                  <div className={`r-label ${weatherLabelTone}`}>
                    <span className="r-dot" aria-hidden="true" />
                    Destination weather
                  </div>
                  <WeatherSummary weather={weatherData} destinationCode={destinationCode} isLoading={isBusy && !weatherData} />
                </article>

                <article
                  className={[
                    "r-card",
                    "support-card",
                    hasReasoningContent ? "support-card--live" : "",
                    isBusy && !hasReasoningContent ? "support-card--loading" : "",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                >
                  <div className={`r-label ${reasoningLabelTone}`}>
                    <span className="r-dot" aria-hidden="true" />
                    AI reasoning trace
                  </div>
                  <AIReasoningPanel finalJson={finalJson} isStreaming={isBusy} reasoningSteps={reasoningSteps} />
                </article>
              </aside>
            </section>

            <div id="results" className="result-wrap reveal">
              <article className={flightsCardClass}>
                <div className="r-label r-label--secondary">
                  <span className="r-dot" aria-hidden="true" />
                  Available flights
                </div>
                <FlightsList flights={flights} bestFlight={bestFlight} isLoading={isBusy && !hasFlights} />
              </article>
            </div>
          </section>

          <div className="trust-strip reveal">
            <FlightsTicker items={trustNames} speed={40} />
          </div>

          <section id="capabilities" className="capabilities-shell reveal">
            <FeatureCapabilities items={FEATURE_CAPABILITIES} />
          </section>
        </main>
        {isDevMode && (
          <DevRoutingDrawer
            isOpen={devDrawerOpen}
            llmOptions={llmOptions}
            llmMode={devLlmMode}
            cloudProvider={devCloudProvider}
            onModeChange={setDevLlmMode}
            onProviderChange={setDevCloudProvider}
            onClose={() => setDevDrawerOpen(false)}
          />
        )}
      </div>
    </div>
  );
}
