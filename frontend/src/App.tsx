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
import MultiCitySummary from "./components/MultiCitySummary";
import DevRoutingDrawer from "./components/DevRoutingDrawer";
import DebugDrawer from "./components/DebugDrawer";
import destinationCoastal from "./assets/photos/goa-beach.jpg";
import destinationBusiness from "./assets/photos/mumbai-skyline.jpg";
import destinationStopover from "./assets/photos/delhi-airport.jpg";
import { IS_PREVIEW_UI } from "./lib/uiMode";
import { FEATURE_CAPABILITIES } from "./lib/capabilities";
import type { AskPayload, Flight, TripPlan, MultiCityLeg, LLMMode, LLMOptionsResponse, ServerVersionMeta } from "./lib/types";
import { formatFlightSummaryLine, formatPriceINR, formatTemperatureC } from "./lib/format";

type ThemePreference = "system" | "dark" | "light";

const THEME_STORAGE_KEY = "travelyst_theme_preference";

function readThemePreference(): ThemePreference {
  if (typeof window === "undefined") return "system";
  const saved = window.localStorage.getItem(THEME_STORAGE_KEY);
  return saved === "dark" || saved === "light" || saved === "system" ? saved : "system";
}

function resolveTheme(preference: ThemePreference, prefersDark: boolean): "dark" | "light" {
  if (preference === "system") return prefersDark ? "dark" : "light";
  return preference;
}

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

function buildPackingTipFromWeather(weather: Record<string, unknown> | null): string {
  if (!weather) return "Pack one light layer for changing temperatures.";
  const condition = String(weather.condition || "").toLowerCase();
  const temp = weather.temperature_c;
  const tempNum = typeof temp === "number" ? temp : Number(temp);
  if (condition.includes("rain")) return "Carry a compact umbrella and a light waterproof layer.";
  if (Number.isFinite(tempNum) && tempNum >= 32) return "Pack breathable clothing and keep water handy.";
  if (Number.isFinite(tempNum) && tempNum <= 18) return "Carry a light jacket for cooler hours.";
  return "Pack one light layer for changing temperatures.";
}

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

function buildRouteSummary(
  debugInfo: TripPlan["debug_info"] | null,
  lastPayload: AskPayload | null
): string {
  const intent = (debugInfo?.intent ?? {}) as Record<string, unknown>;
  const routeLabels = (debugInfo?.route_labels ?? {}) as Record<string, unknown>;
  const origin = toIata(intent.origin_iata) || toIata(routeLabels.origin_iata) || toIata(lastPayload?.origin);
  const destination =
    toIata(intent.destination_iata) || toIata(routeLabels.destination_iata) || toIata(lastPayload?.destination);
  if (!origin || !destination) return "";
  const originText = endpointLabel(origin, toText(routeLabels.origin_city), toText(routeLabels.origin_label));
  const destinationText = endpointLabel(
    destination,
    toText(routeLabels.destination_city),
    toText(routeLabels.destination_label)
  );
  return `Route: ${originText} to ${destinationText}.`;
}

function buildMultiCityNarrative(legs: MultiCityLeg[]): string {
  if (!legs.length) return "";

  const sections = legs.map((leg, index) => {
    const intent = (leg.debug_info?.intent ?? {}) as Record<string, unknown>;
    const routeLabels = (leg.debug_info?.route_labels ?? {}) as Record<string, unknown>;
    const origin = toIata(intent.origin_iata);
    const destination = toIata(intent.destination_iata);
    const routeText =
      origin && destination
        ? `${endpointLabel(origin, toText(routeLabels.origin_city), toText(routeLabels.origin_label))} -> ${endpointLabel(
            destination,
            toText(routeLabels.destination_city),
            toText(routeLabels.destination_label)
          )}`
        : `Leg ${index + 1}`;

    const bestFlight = leg.best_flight;
    const weather = leg.weather && typeof leg.weather === "object" ? leg.weather : null;
    const weatherParts = weather
      ? [
          typeof weather.condition === "string" ? weather.condition : "",
          formatTemperatureC(weather.temperature_c),
          formatTemperatureC(weather.temp_min_c) !== "N/A" ? `low ${formatTemperatureC(weather.temp_min_c)}` : "",
          formatTemperatureC(weather.temp_max_c) !== "N/A" ? `high ${formatTemperatureC(weather.temp_max_c)}` : "",
        ].filter(Boolean)
      : [];

    const flightText = bestFlight
      ? `${bestFlight.airline} ${bestFlight.flight_no} ${bestFlight.departure_time} -> ${bestFlight.arrival_time} (${formatPriceINR(bestFlight.price_inr)})`
      : "Flight details unavailable";
    const weatherText = weatherParts.length ? weatherParts.join(", ") : "N/A";

    return `${routeText}\nFlight: ${flightText}\nWeather: ${weatherText}`;
  });

  return `Multi-city itinerary\n\n${sections.join("\n\n")}`;
}

export default function App() {
  const [serverStatus, setServerStatus] = useState<"checking" | "online" | "offline">("checking");
  const [lastPayload, setLastPayload] = useState<AskPayload | null>(null);
  const [resultVersion, setResultVersion] = useState(0);
  const [llmOptions, setLlmOptions] = useState<LLMOptionsResponse | null>(null);
  const [healthSnapshot, setHealthSnapshot] = useState<Record<string, unknown> | null>(null);
  const [serverVersion, setServerVersion] = useState<ServerVersionMeta | null>(null);
  const [devDrawerOpen, setDevDrawerOpen] = useState(false);
  const [devLlmMode, setDevLlmMode] = useState<LLMMode>("ollama_first");
  const [devCloudProvider, setDevCloudProvider] = useState<string>("gemini");
  const [themePreference, setThemePreference] = useState<ThemePreference>(() => readThemePreference());
  const [resolvedTheme, setResolvedTheme] = useState<"dark" | "light">(() => {
    if (typeof window === "undefined") return "dark";
    return resolveTheme(readThemePreference(), window.matchMedia("(prefers-color-scheme: dark)").matches);
  });
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
    responseMeta,
    rawStream,
    start,
    cancel
  } = useStreamingPlan();
  const isDevMode = (() => {
    if (typeof window === "undefined") return false;
    const params = new URLSearchParams(window.location.search);
    return params.get("dev") === "true" || params.get("devmode") === "true";
  })();

  useEffect(() => {
    if (typeof window === "undefined") return;
    const media = window.matchMedia("(prefers-color-scheme: dark)");

    const applyTheme = () => {
      const nextTheme = resolveTheme(themePreference, media.matches);
      setResolvedTheme(nextTheme);
      document.documentElement.setAttribute("data-theme", nextTheme);
      document.documentElement.style.colorScheme = nextTheme;
    };

    applyTheme();

    if (themePreference !== "system") return;

    const onSystemThemeChange = () => applyTheme();
    if (typeof media.addEventListener === "function") {
      media.addEventListener("change", onSystemThemeChange);
      return () => media.removeEventListener("change", onSystemThemeChange);
    }
    media.addListener(onSystemThemeChange);
    return () => media.removeListener(onSystemThemeChange);
  }, [themePreference]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (themePreference === "system") {
      window.localStorage.removeItem(THEME_STORAGE_KEY);
      return;
    }
    window.localStorage.setItem(THEME_STORAGE_KEY, themePreference);
  }, [themePreference]);

  useEffect(() => {
    const checkServer = () => {
      fetch(`${API_BASE}/health`)
        .then(async (res) => {
          const payload = await res.json().catch(() => null);
          if (payload && typeof payload === "object") {
            setHealthSnapshot(payload as Record<string, unknown>);
          }
          setServerStatus(res.ok ? "online" : "offline");
        })
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
    if (!isDevMode) {
      setServerVersion(null);
      return;
    }
    let cancelled = false;
    const fetchVersion = async () => {
      try {
        const resp = await fetch(`${API_BASE}/version`);
        if (!resp.ok) return;
        const data = (await resp.json()) as ServerVersionMeta;
        if (!cancelled) setServerVersion(data);
      } catch {
        // Keep empty on failure.
      }
    };
    fetchVersion();
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
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReducedMotion) {
      targets.forEach((target) => target.classList.add("visible"));
      return;
    }

    targets.forEach((target, index) => {
      const revealDelay = `${Math.min(index, 8) * 60}ms`;
      (target as HTMLElement).style.setProperty("--reveal-delay", revealDelay);
    });

    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
            io.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.16 }
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
  const routeSummary = buildRouteSummary(debugInfo, lastPayload);
  const finalFlightsFromDebug = Array.isArray(debugInfo?.all_flights) ? (debugInfo.all_flights as Flight[]) : undefined;
  const finalFlights = finalFlightsFromDebug || (Array.isArray(finalJson?.all_flights) ? finalJson.all_flights : undefined);
  const multiCityLegs = Array.isArray(finalJson?.legs) ? (finalJson.legs as MultiCityLeg[]) : [];
  const isMultiCity = Boolean(finalJson?.multicity) || multiCityLegs.length > 0;
  const multiCityFlights = multiCityLegs
    .map((leg) => leg.best_flight)
    .filter((flight): flight is Flight => Boolean(flight));
  const flights = finalFlights || (isMultiCity ? multiCityFlights : undefined) || partialFlights || undefined;
  const bestFlight = finalJson?.best_flight || (isMultiCity ? multiCityFlights[0] : undefined) || partialBestFlight || undefined;
  const hasFlights = Array.isArray(flights) && flights.length > 0;
  const flightsCount = Array.isArray(flights) ? flights.length : 0;
  const finalRecord = (finalJson ?? null) as Record<string, unknown> | null;
  const reasoningFromDebug = stringifyReasoningCandidate(
    debugInfo?.agent_reasoning ??
      debugInfo?.reasoning ??
      finalRecord?.agent_reasoning ??
      finalRecord?.reasoning
  );
  const multiCityNarrative = isMultiCity ? buildMultiCityNarrative(multiCityLegs) : "";
  const finalMessage =
    multiCityNarrative.trim().length > 0
      ? multiCityNarrative
      : typeof finalRecord?.llm_response === "string"
        ? finalRecord.llm_response
        : typeof finalRecord?.warning === "string"
          ? finalRecord.warning
        : typeof finalRecord?.message === "string"
          ? finalRecord.message
          : typeof finalRecord?.error === "string"
            ? finalRecord.error
            : reasoningFromDebug || buildFallbackSummary(finalJson);
  const enrichedFinalMessage =
    routeSummary && finalMessage && !finalMessage.toLowerCase().includes(routeSummary.toLowerCase())
      ? `${routeSummary} ${finalMessage}`.trim()
      : finalMessage;
  const finalWeatherData = finalJson?.weather && typeof finalJson.weather === "object" ? finalJson.weather : null;
  const weatherData = finalWeatherData || partialWeather;
  const returnTrip = finalJson?.return_trip || null;
  const returnTripFlight = returnTrip?.best_flight;
  const returnTripWeather = returnTrip?.weather && typeof returnTrip.weather === "object" ? returnTrip.weather : null;
  const returnTripWarnings = Array.isArray(returnTrip?.warnings) ? returnTrip.warnings.filter(Boolean) : [];
  const resultWarnings = Array.isArray(finalJson?.warnings) ? finalJson.warnings.filter(Boolean) : [];
  const hasConstraintWarnings = resultWarnings.length > 0 || returnTripWarnings.length > 0;
  const finalWeatherPresent = typeof finalJson?.weather_present === "boolean" ? finalJson.weather_present : undefined;
  const finalWeatherReason = typeof finalJson?.weather_reason === "string" ? finalJson.weather_reason : undefined;
  const routeLabels = (debugInfo?.route_labels ?? {}) as Record<string, unknown>;
  const destinationCode =
    (typeof debugInfo?.intent?.destination_iata === "string" ? debugInfo.intent.destination_iata : undefined) ||
    (typeof routeLabels.destination_iata === "string" ? routeLabels.destination_iata : undefined) ||
    (typeof weatherData?.location === "string" ? weatherData.location : undefined);
  const destinationLabel =
    typeof routeLabels.destination_label === "string"
      ? routeLabels.destination_label
      : typeof weatherData?.location_label === "string"
        ? weatherData.location_label
        : undefined;
  const isBusy = isStreaming || isFallback;
  const streamPaneTokens = isMultiCity && multiCityNarrative.trim().length > 0 ? "" : tokens;
  const hasTokenContent = streamPaneTokens.length > 0;
  const hasReasoningContent =
    reasoningSteps.length > 0 ||
    reasoningFromDebug.trim().length > 0 ||
    Boolean(finalJson?.best_flight) ||
    (isMultiCity && multiCityLegs.length > 0);
  const hasWeatherContent = Boolean(weatherData);
  const showWeatherPanel = hasWeatherContent;
  const showReasoningPanel = hasReasoningContent;
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
  const statusText =
    serverStatus === "online" ? "Service live" : serverStatus === "offline" ? "Service limited" : "Connecting";
  const showServiceStatus = serverStatus !== "online";
  const heroBadgeText = IS_PREVIEW_UI ? "AI-guided trip planning" : "Travel intelligence engine";
  const streamHeading = !isBusy && (hasTokenContent || finalMessage.trim().length > 0) ? "Trip brief" : "AI thinking";
  const highlightWeatherText = weatherData
    ? [typeof weatherData.condition === "string" ? weatherData.condition : "", formatTemperatureC(weatherData.temperature_c)]
        .filter(Boolean)
        .join(" · ")
    : "Weather updates appear once results load.";
  const highlights = bestFlight
    ? [
        {
          title: "Best Flight",
          text: `${bestFlight.airline} ${bestFlight.flight_no} · ${bestFlight.departure_time} → ${bestFlight.arrival_time} · ${formatPriceINR(bestFlight.price_inr)}`
        },
        {
          title: "Destination Weather",
          text: highlightWeatherText
        },
        {
          title: "Packing Tip",
          text: buildPackingTipFromWeather(weatherData)
        }
      ]
    : [];
  const showHighlights = !isBusy && !isMultiCity && highlights.length === 3;
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
  const partialOutcomeError =
    typeof error === "string" &&
    error.toLowerCase().includes("available flight/weather results are shown");
  const resultStatus = finalJson?.result_status || responseMeta?.result_status;
  const isDegradedResult = resultStatus === "degraded";
  const noFlightsFailure =
    responseMeta?.failure_reason === "no_flights" || responseMeta?.no_flights_reason === "no_flights";
  const degradedSummary =
    finalJson?.fallback_note ||
    finalJson?.degradation?.message ||
    responseMeta?.fallback_note ||
    responseMeta?.degradation_message ||
    "Some explanation details are unavailable right now, but the trip data shown is still usable.";
  const bestFlightHasHandoff = Boolean(
    typeof bestFlight?.handoff_url === "string" && bestFlight.handoff_url.trim().length > 0
  );
  const bestFlightStopSummary = bestFlight
    ? Number(bestFlight.stops) === 0
      ? "Non-stop route"
      : `${bestFlight.stops} stop${Number(bestFlight.stops) === 1 ? "" : "s"}`
    : "";
  const weatherProofSummary = weatherData
    ? [typeof weatherData.condition === "string" ? weatherData.condition : "", formatTemperatureC(weatherData.temperature_c)]
        .filter(Boolean)
        .join(" · ")
    : finalWeatherPresent === false
      ? "Forecast unavailable for selected date window"
      : "Weather context appears once flight results load";
  const reasoningProofSummary =
    reasoningFromDebug.split("\n").map((line) => line.trim()).find(Boolean) ||
    reasoningSteps.find((step) => step.trim().length > 0) ||
    "Reasoning trace appears when enough route evidence is available.";
  const proofStatusLabel =
    resultStatus === "degraded"
      ? "Partial but usable"
      : resultStatus === "error"
        ? "Needs retry"
        : hasFlights
          ? "Decision-ready"
          : isBusy
            ? "Analyzing"
            : "Awaiting query";
  const showProofSurface = Boolean(lastPayload || isBusy || hasFlights || finalJson || error);
  const routeRevealSteps = [
    {
      id: "intent",
      title: "Intent detected",
      description: "Your route, timing, and travel style are interpreted from the prompt.",
      active: Boolean(lastPayload),
    },
    {
      id: "route",
      title: "Route composed",
      description: "Candidate flights are collected and ranked for practicality and fare balance.",
      active: isBusy || hasFlights || Boolean(finalJson),
    },
    {
      id: "weather",
      title: "Weather layered in",
      description: "Destination conditions refine packing and comfort guidance.",
      active: hasWeatherContent || finalWeatherPresent === false || Boolean(finalWeatherReason),
    },
    {
      id: "itinerary",
      title: "Best itinerary selected",
      description: "A top option is surfaced with rationale and booking confidence signals.",
      active: Boolean(bestFlight) || Boolean(finalJson),
    },
  ];
  const curatedPanels = [
    {
      title: "Coastal escape",
      route: "Bengaluru → Goa",
      note: "Sunset arrival windows with clean non-stop leisure routing.",
      image: destinationCoastal,
      mood: "coastal",
    },
    {
      title: "Business corridor",
      route: "Delhi → Mumbai",
      note: "High-frequency corridor where schedule fit drives decision quality.",
      image: destinationBusiness,
      mood: "business",
    },
    {
      title: "Smart stopover",
      route: "Delhi → Goa via Mumbai",
      note: "Multi-leg continuity checks with transfer and timing confidence.",
      image: destinationStopover,
      mood: "stopover",
    },
  ];
  const heroTrustSignals = [
    "Live ranking across major carriers",
    "Weather context blended into every shortlist",
    "Secure handoff links when providers support booking",
  ];
  const activeRouteStepIndex = routeRevealSteps.reduce((lastActive, step, index) => (step.active ? index : lastActive), -1);
  const routeProgressPercent = ((activeRouteStepIndex + 1) / routeRevealSteps.length) * 100;
  const routeNarrativeStatus = isBusy
    ? "The planner is actively advancing through route composition and scoring."
    : activeRouteStepIndex >= routeRevealSteps.length - 1
      ? "Itinerary narrative is complete and ready for comparison below."
      : "Start planning to watch each stage progress in real time.";
  const themeStatusLabel =
    themePreference === "system"
      ? `Auto (${resolvedTheme === "dark" ? "Dark" : "Light"})`
      : themePreference === "dark"
        ? "Dark"
        : "Light";

  useEffect(() => {
    if (!isBusy && !error && finalJson) {
      setResultVersion((v) => v + 1);
    }
  }, [finalJson, isBusy, error]);

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
            <div className="theme-switch" role="group" aria-label="Theme mode">
              {(["system", "dark", "light"] as ThemePreference[]).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  className={`theme-switch__button ${themePreference === mode ? "theme-switch__button--active" : ""}`}
                  onClick={() => setThemePreference(mode)}
                >
                  {mode === "system" ? "Auto" : mode === "dark" ? "Dark" : "Light"}
                </button>
              ))}
            </div>
            {showServiceStatus && (
              <div
                className={`api-status ${serverStatus === "offline" ? "api-status--offline" : "api-status--online"} ${IS_PREVIEW_UI ? "api-status--preview" : ""}`}
                title={`Theme: ${themeStatusLabel}`}
              >
                {statusText}
              </div>
            )}
            <a className="btn-primary" href="#planner">
              Start planning →
            </a>
          </div>
        </nav>

        <main className="app-main">
          <div className="experience-shell">
          <section id="planner" className="hero experience-section experience-section--hero">
            <div className="hero-intro reveal">
              <div className="hero-badge">
                <span className="badge-dot" aria-hidden="true" />
                {heroBadgeText}
              </div>
              <h1 className="hero-title">
                <span className="title-line-1">Plan your next journey</span>
                <span className="title-line-2">with premium AI travel guidance.</span>
              </h1>
              <p className="hero-sub">
                Describe the trip in natural language and get ranked flights, weather-aware guidance, and booking-ready options in one polished decision flow.
              </p>
              <div className="hero-trust-row" aria-label="Trust indicators">
                {heroTrustSignals.map((signal) => (
                  <span key={signal} className="hero-trust-pill">{signal}</span>
                ))}
              </div>
            </div>

            <section className="hero-grid">
              <div className="hero-left">
                <div className="search-card">
                  <QueryForm
                    onSubmit={handleSubmit}
                    disabled={isBusy}
                    resultVersion={resultVersion}
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
                      <span className="min-w-0 break-words">
                        {partialOutcomeError
                          ? error
                          : noFlightsFailure
                            ? error
                            : `We couldn't finish your plan. ${error}`}
                      </span>
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
                  {!error && isDegradedResult && (
                    <div className="notice notice--inline">
                      <span className="min-w-0 break-words">
                        Partial result: {degradedSummary}
                      </span>
                    </div>
                  )}
                  {hasConstraintWarnings && !error && (
                    <div className="notice notice--inline">
                      <span className="min-w-0 break-words">
                        Constraint adjustments: {[...resultWarnings, ...returnTripWarnings].join(" ")}
                      </span>
                    </div>
                  )}

                </div>

                {showHighlights && (
                  <div className="highlights-row" aria-label="Trip highlights">
                    {highlights.map((highlight) => (
                      <article key={highlight.title} className="highlight-card">
                        <p className="highlight-card__title">{highlight.title}</p>
                        <p className="highlight-card__text">{highlight.text}</p>
                      </article>
                    ))}
                  </div>
                )}

                <div className="suggestions-row sugg-strip">
                  <div className="sugg-scroll">
                    {suggestionChips.map((item) => (
                      <button
                        key={item}
                        type="button"
                        className="s-chip history-chip"
                        title={item}
                        aria-label={item}
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
                    {streamHeading}
                  </div>
                  <StreamPane
                    tokens={streamPaneTokens}
                    finalText={enrichedFinalMessage}
                    finalJson={finalJson}
                    fallbackBestFlight={bestFlight}
                    fallbackWeather={weatherData}
                    isStreaming={isBusy}
                    canCancel={isStreaming}
                    onCancel={cancel}
                  />
                </article>

                {hasFlights && (
                  <a className="results-nudge" href="#results">
                    {flightsCount} {flightsCount === 1 ? "flight" : "flights"} found below ↓
                  </a>
                )}
              </div>

              {(showWeatherPanel || showReasoningPanel) && (
                <aside className="hero-right">
                {showWeatherPanel && (
                  <article
                    className={[
                      "r-card",
                      "support-card",
                      "support-card--reveal",
                      hasWeatherContent ? "support-card--live" : "",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                  >
                    <div className={`r-label r-label--sidebar ${weatherLabelTone}`}>
                      <span className="r-dot" aria-hidden="true" />
                      Destination weather
                    </div>
                    <WeatherSummary
                      weather={weatherData}
                      destinationCode={destinationCode}
                      destinationLabel={destinationLabel}
                      weatherPresent={finalWeatherPresent}
                      weatherReason={finalWeatherReason}
                      isLoading={false}
                    />
                  </article>
                )}

                {showReasoningPanel && (
                  <article
                    className={[
                      "r-card",
                      "support-card",
                      "support-card--reveal",
                      hasReasoningContent ? "support-card--live" : "",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                  >
                    <div className={`r-label r-label--sidebar ${reasoningLabelTone}`}>
                      <span className="r-dot" aria-hidden="true" />
                      AI reasoning trace
                    </div>
                    <AIReasoningPanel finalJson={finalJson} isStreaming={isBusy} reasoningSteps={reasoningSteps} />
                  </article>
                )}
                {isDevMode && (
                  <DebugDrawer
                    data={{
                      result_status: resultStatus,
                      response_meta: responseMeta,
                      health: healthSnapshot,
                      llm_options: llmOptions,
                    }}
                    rawStream={rawStream}
                    serverVersion={serverVersion}
                  />
                )}
                </aside>
              )}
            </section>
            <a href="#route-reveal" className="hero-scroll-cue">
              How the planner thinks ↓
            </a>
          </section>

          <div className="experience-divider reveal" aria-hidden="true" />

          <section id="route-reveal" className="experience-section experience-section--route reveal" aria-label="How the planner thinks">
            <div className="route-reveal-shell">
              <div className="route-reveal-intro">
                <div className="section-head route-reveal-intro__head">
                  <p className="section-label">How the planner thinks</p>
                  <h2 className="section-title">A guided four-step path from intent to booking confidence</h2>
                </div>
                <p className="route-reveal-intro__sub">
                  Follow how your travel intent becomes a ranked shortlist with weather context and booking confidence.
                </p>
                <div className="route-reveal-progress" aria-hidden="true">
                  <div className="route-reveal-progress__fill" style={{ width: `${Math.max(routeProgressPercent, 0)}%` }} />
                </div>
                <p className="route-reveal-intro__status">{routeNarrativeStatus}</p>
              </div>
              <div className="route-reveal-track">
                {routeRevealSteps.map((step, index) => (
                  <article
                    key={step.id}
                    className={`route-reveal-card ${step.active ? "route-reveal-card--active" : ""}`}
                    style={{ animationDelay: `${index * 80}ms` }}
                  >
                    <div className="route-reveal-card__rail" aria-hidden="true" />
                    <div className="route-reveal-card__body">
                      <div className="route-reveal-card__index">0{index + 1}</div>
                      <h3 className="route-reveal-card__title">{step.title}</h3>
                      <p className="route-reveal-card__desc">{step.description}</p>
                    </div>
                  </article>
                ))}
              </div>
            </div>
          </section>

          <div className="experience-divider reveal" aria-hidden="true" />

          <section className="experience-section experience-section--curation reveal" aria-label="Curated route moods">
            <div className="section-head">
              <p className="section-label">Curated lanes</p>
              <h2 className="section-title">Editorial route rhythms for every trip profile</h2>
            </div>
            <div className="curation-grid">
              {curatedPanels.map((panel, index) => (
                <article key={panel.title} className={`curation-card curation-card--${panel.mood} ${index === 0 ? "curation-card--featured" : ""}`}>
                  <div className="curation-card__media-wrap" aria-hidden="true">
                    <img src={panel.image} alt="" className="curation-card__media" loading="lazy" decoding="async" />
                  </div>
                  <div className="curation-card__veil" aria-hidden="true" />
                  <div className="curation-card__content">
                    <p className="curation-card__kicker">{panel.title}</p>
                    <h3 className="curation-card__route">{panel.route}</h3>
                    <p className="curation-card__note">{panel.note}</p>
                  </div>
                </article>
              ))}
            </div>
          </section>

          <div className="experience-divider reveal" aria-hidden="true" />

          <section className="experience-section experience-section--immersive reveal" aria-label="Immersive route map">
            <div className="immersive-shell">
              <div className="immersive-shell__header">
                <p className="section-label">Spatial route plane</p>
                <h2 className="section-title">A cinematic confidence map for route quality and handoff readiness</h2>
                <p className="immersive-shell__sub">
                  One immersive product moment, designed to preview route intelligence without distracting from planning or results.
                </p>
              </div>
              <div className="immersive-scene" aria-hidden="true">
                <div className="immersive-scene__halo immersive-scene__halo--top" />
                <div className="immersive-scene__halo immersive-scene__halo--bottom" />
                <div className="immersive-scene__plane" />
                <svg className="immersive-scene__routes" viewBox="0 0 720 320" preserveAspectRatio="none">
                  <defs>
                    <linearGradient id="routePrimary" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="rgba(124, 98, 255, 0.95)" />
                      <stop offset="100%" stopColor="rgba(90, 198, 255, 0.9)" />
                    </linearGradient>
                    <linearGradient id="routeSecondary" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="rgba(88, 233, 191, 0.9)" />
                      <stop offset="100%" stopColor="rgba(120, 164, 255, 0.82)" />
                    </linearGradient>
                  </defs>
                  <path className="immersive-route immersive-route--primary" d="M82 248 C 220 82, 366 92, 556 236" stroke="url(#routePrimary)" />
                  <path className="immersive-route immersive-route--secondary" d="M142 262 C 285 138, 406 126, 630 224" stroke="url(#routeSecondary)" />
                </svg>

                <div className="immersive-waypoint immersive-waypoint--left">
                  <span className="immersive-waypoint__label">DEL</span>
                </div>
                <div className="immersive-waypoint immersive-waypoint--mid">
                  <span className="immersive-waypoint__label">BOM</span>
                </div>
                <div className="immersive-waypoint immersive-waypoint--right">
                  <span className="immersive-waypoint__label">GOI</span>
                </div>

                <div className="immersive-chip immersive-chip--flight">Top fare trajectory · ₹5.4k band</div>
                <div className="immersive-chip immersive-chip--weather">Weather layer synced · warm coastal window</div>
                <div className="immersive-chip immersive-chip--handoff">Booking handoff confidence · provider-backed</div>

                <div className="immersive-scene__glow" />
              </div>
            </div>
          </section>

          <div className="experience-divider reveal" aria-hidden="true" />

          <section id="results" className="experience-section experience-section--proof">
            <div className="section-head reveal">
              <p className="section-label">Product proof</p>
              <h2 className="section-title">Choose faster with clear ranking, evidence, and booking confidence</h2>
            </div>
            {showProofSurface ? (
              <>
                <div className="proof-overview-grid reveal">
                  <article className="r-card proof-card proof-card--best">
                    <div className="proof-card__head">
                      <p className="proof-card__kicker">Top recommendation</p>
                      <span className={`proof-status proof-status--${proofStatusLabel.toLowerCase().replace(/\s+/g, "-")}`}>
                        {proofStatusLabel}
                      </span>
                    </div>
                    {bestFlight ? (
                      <>
                        <h3 className="proof-card__title">
                          {bestFlight.airline} {bestFlight.flight_no} · {formatPriceINR(bestFlight.price_inr)}
                        </h3>
                        <p className="proof-card__summary">
                          {bestFlight.departure_time} → {bestFlight.arrival_time} · {bestFlight.duration_min} min · {bestFlightStopSummary}
                        </p>
                        <div className="proof-chip-row">
                          <span className="proof-chip">{bestFlightHasHandoff ? "Provider handoff ready" : "Handoff depends on provider availability"}</span>
                          <span className="proof-chip">Weather: {weatherProofSummary}</span>
                        </div>
                      </>
                    ) : (
                      <p className="proof-card__summary">
                        {isBusy
                          ? "Building a recommendation from live route and weather evidence..."
                          : "Submit a route to generate ranked proof and booking readiness signals."}
                      </p>
                    )}
                  </article>

                  <article className="r-card proof-card proof-card--evidence">
                    <p className="proof-card__kicker">Evidence stack</p>
                    <ul className="proof-evidence-list">
                      <li className="proof-evidence-item">
                        <span className="proof-evidence-item__label">Ranked shortlist</span>
                        <span className="proof-evidence-item__value">
                          {hasFlights ? `${flightsCount} candidate${flightsCount === 1 ? "" : "s"} compared` : isBusy ? "Compiling live options" : "No shortlist yet"}
                        </span>
                      </li>
                      <li className="proof-evidence-item">
                        <span className="proof-evidence-item__label">Weather intelligence</span>
                        <span className="proof-evidence-item__value">{weatherProofSummary}</span>
                      </li>
                      <li className="proof-evidence-item">
                        <span className="proof-evidence-item__label">Selection rationale</span>
                        <span className="proof-evidence-item__value">{reasoningProofSummary}</span>
                      </li>
                      <li className="proof-evidence-item">
                        <span className="proof-evidence-item__label">Booking confidence</span>
                        <span className="proof-evidence-item__value">
                          {bestFlightHasHandoff ? "Secure handoff link available on recommended option" : "Link appears only when provider handoff is available"}
                        </span>
                      </li>
                    </ul>
                  </article>
                </div>
                <div className="result-wrap reveal">
                  {isMultiCity && multiCityLegs.length > 0 && (
                    <article className="r-card results-card">
                      <div className="r-label r-label--secondary">
                        <span className="r-dot" aria-hidden="true" />
                        Multi-city itinerary
                      </div>
                      <MultiCitySummary legs={multiCityLegs} />
                    </article>
                  )}
                  <article className={flightsCardClass}>
                    <div className="r-label r-label--secondary">
                      <span className="r-dot" aria-hidden="true" />
                      Ranked shortlist
                    </div>
                    <FlightsList flights={flights} bestFlight={bestFlight} isLoading={isBusy && !hasFlights} />
                  </article>
                  {returnTripFlight && (
                    <article className="r-card results-card">
                      <div className="r-label r-label--secondary">
                        <span className="r-dot" aria-hidden="true" />
                        Return leg snapshot
                      </div>
                      <p className="flight-item__summary">
                        {returnTripFlight.airline} {returnTripFlight.flight_no} · {returnTripFlight.departure_time} →{" "}
                        {returnTripFlight.arrival_time} · {formatPriceINR(returnTripFlight.price_inr)}
                      </p>
                      {returnTripWeather && (
                        <p className="flight-item__meta">
                          Weather:{" "}
                          {[
                            typeof returnTripWeather.condition === "string" ? returnTripWeather.condition : "",
                            formatTemperatureC(returnTripWeather.temperature_c),
                          ]
                            .filter(Boolean)
                            .join(", ")}
                        </p>
                      )}
                    </article>
                  )}
                </div>
              </>
            ) : (
              <article className="r-card proof-placeholder reveal">
                <p className="proof-card__kicker">Product proof</p>
                <h3 className="proof-card__title">Ranked proof appears right after your first query.</h3>
                <p className="proof-card__summary">
                  Submit a route to unlock top-pick rationale, weather intelligence, and booking confidence in one view.
                </p>
              </article>
            )}
          </section>

          {(showProofSurface || isBusy) && (
            <div className="trust-strip reveal">
              <p className="trust-strip__copy">Live itinerary confidence across high-traffic Indian carrier routes.</p>
              <FlightsTicker items={trustNames} speed={40} />
            </div>
          )}

          <section id="capabilities" className="capabilities-shell reveal experience-section experience-section--confidence">
            <FeatureCapabilities items={FEATURE_CAPABILITIES} />
          </section>
          </div>
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
