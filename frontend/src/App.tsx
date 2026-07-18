import { useEffect, useMemo, useRef, useState } from "react";
import {
  API_BASE,
  BOOKING_AUTH_TOKEN_CHANGED_EVENT,
  clearConfiguredAuthToken,
  getConfiguredAuthToken,
  getJson,
  postJson,
  resolveApiUrl,
  setConfiguredAuthToken,
} from "./lib/api";
import { useStreamingPlan } from "./hooks/useStreamingPlan";
import { useAsyncJob } from "./hooks/useAsyncJob";

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
import type {
  AskPayload,
  Flight,
  TripPlan,
  MultiCityLeg,
  LLMMode,
  LLMOptionsResponse,
  ServerVersionMeta,
  BookingActionResponse,
  BookingHandoffCapabilities,
  BookingResolveHandoffResponse,
  BookingResolveState,
  BookingRecord,
  PriceAlert,
  PriceTrackingStatus,
} from "./lib/types";
import { formatFlightSummaryLine, formatPriceINR, formatTemperatureC } from "./lib/format";

type ThemePreference = "system" | "dark" | "light";

const THEME_STORAGE_KEY = "travelyst_theme_preference";
let devLlmOptionsBootstrapPromise: Promise<LLMOptionsResponse | null> | null = null;
let devServerVersionBootstrapPromise: Promise<ServerVersionMeta | null> | null = null;

async function fetchDevLlmOptionsOnce(): Promise<LLMOptionsResponse | null> {
  if (!devLlmOptionsBootstrapPromise) {
    devLlmOptionsBootstrapPromise = fetch(`${API_BASE}/llm/options`, { cache: "no-store" })
      .then(async (resp) => (resp.ok ? ((await resp.json()) as LLMOptionsResponse) : null))
      .catch((err) => {
        devLlmOptionsBootstrapPromise = null;
        throw err;
      });
  }
  return devLlmOptionsBootstrapPromise;
}

async function fetchServerVersionOnce(): Promise<ServerVersionMeta | null> {
  if (!devServerVersionBootstrapPromise) {
    devServerVersionBootstrapPromise = fetch(`${API_BASE}/version`, { cache: "no-store" })
      .then(async (resp) => (resp.ok ? ((await resp.json()) as ServerVersionMeta) : null))
      .catch((err) => {
        devServerVersionBootstrapPromise = null;
        throw err;
      });
  }
  return devServerVersionBootstrapPromise;
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

function describeBookingMode(
  {
    capabilities,
    hasBookingAuth,
    bookingAuthMode,
  }: {
    capabilities: BookingHandoffCapabilities | null;
    hasBookingAuth: boolean;
    bookingAuthMode: "authenticated_token" | "local_dev_unauthed" | null;
  }
): string {
  if (capabilities?.auth_rejected) {
    if (capabilities.resolve_available_now && capabilities.auth_mode === "local_dev_unauthed") {
      return "Booking mode: configured bearer token was rejected; local-dev unauthenticated loopback mode is active.";
    }
    return "Booking mode: configured bearer token was rejected by backend auth.";
  }
  if (capabilities?.blocked_reason === "missing_token") {
    return "Booking mode: no bearer token configured.";
  }
  if (capabilities?.blocked_reason === "loopback_required_for_local_dev") {
    return "Booking mode: local-dev unauthenticated mode is enabled, but this request is not loopback-eligible.";
  }
  if (capabilities?.auth_mode === "local_dev_unauthed") {
    return "Booking mode: local-dev unauthenticated handoff resolution is active (loopback only).";
  }
  if (capabilities?.auth_mode === "authenticated_token") {
    return "Booking mode: authenticated bearer token is active.";
  }
  if (capabilities?.auth_mode === "auth_required") {
    return capabilities.message || "Booking mode: authentication is required for lazy handoff resolution.";
  }
  if (bookingAuthMode === "local_dev_unauthed") {
    return "Booking mode: local-dev unauthenticated handoff resolution (loopback only).";
  }
  if (hasBookingAuth || bookingAuthMode === "authenticated_token") {
    return "Booking mode: authenticated bearer token.";
  }
  return "Booking mode: no auth token detected. Lazy Book requires auth unless backend local-dev override is enabled.";
}

function isInvalidAuthErrorMessage(message: string): boolean {
  const lowered = String(message || "").toLowerCase();
  return lowered.includes("invalid authentication token") || lowered.includes("invalid authorization header");
}

function bookingFlightIdentity(flight: Flight): string {
  // Deliberately excludes booking_token: token is absent in partial (streaming) flights
  // but present in final flights. Using token in the key would produce different keys for
  // the same flight as it transitions from partial→final, losing any in-progress resolve state.
  // The booking_token is still included in the API request payload for backend coalescing.
  return [
    String(flight.flight_no || "").trim(),
    String(flight.airline || "").trim(),
    String(flight.departure_time || "").trim(),
    String(flight.arrival_time || "").trim(),
    String(flight.price_inr ?? "").trim(),
    String(flight.date || "").trim(),
  ].join("|");
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
  const [asyncMode, setAsyncMode] = useState(false);
  const [bookingItems, setBookingItems] = useState<BookingRecord[]>([]);
  const [bookingActionMessage, setBookingActionMessage] = useState<string | null>(null);
  const [bookingActionError, setBookingActionError] = useState<string | null>(null);
  const [bookingAuthMode, setBookingAuthMode] = useState<"authenticated_token" | "local_dev_unauthed" | null>(null);
  const [bookingAuthToken, setBookingAuthToken] = useState<string>(() => getConfiguredAuthToken());
  const [bookingHandoffCapabilities, setBookingHandoffCapabilities] = useState<BookingHandoffCapabilities | null>(null);
  const [bookingResolveStateByRow, setBookingResolveStateByRow] = useState<Record<string, BookingResolveState>>({});
  const [priceAlerts, setPriceAlerts] = useState<PriceAlert[]>([]);
  const [priceAlertError, setPriceAlertError] = useState<string | null>(null);
  const [priceTrackingStatus, setPriceTrackingStatus] = useState<PriceTrackingStatus | null>(null);
  const [isBookingActionBusy, setIsBookingActionBusy] = useState(false);
  const autoScrolledRef = useRef(false);
  const bookingResolveInflightRef = useRef<Map<string, Promise<BookingResolveHandoffResponse>>>(new Map());
  const healthPollInFlightRef = useRef(false);
  const liveRegionRef = useRef<HTMLElement | null>(null);
  const resultsSectionRef = useRef<HTMLElement | null>(null);
  const resultsAutoScrollVersionRef = useRef(-1);
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
  const [showAllInventory, setShowAllInventory] = useState(false);

  const {
    tokens,
    finalJson: streamFinalJson,
    partialFlights,
    partialTopFlights,
    partialBestFlight,
    partialWeather,
    reasoningSteps,
    isStreaming,
    isFallback,
    error,
    responseMeta,
    rawStream,
    start,
    cancel,
    reset,
    approvalRequired,
    approvalResult,
    respondToApproval,
  } = useStreamingPlan();
  const asyncJob = useAsyncJob();
  const isDevMode = (() => {
    if (typeof window === "undefined") return false;
    const params = new URLSearchParams(window.location.search);
    return params.get("dev") === "true" || params.get("devmode") === "true";
  })();
  const asyncJobActive = ["queued", "running"].includes(String(asyncJob.status));
  const asyncJobResult = asyncJob.job?.result ?? null;
  const finalJson = asyncJobResult || streamFinalJson;
  const asyncJobError =
    asyncJob.status === "error"
      ? (asyncJob.job?.error || asyncJob.error || "Async job failed.")
      : null;
  const activeError = asyncJobError || error;

  useEffect(() => {
    if (typeof window === "undefined") return;
    const media = window.matchMedia("(prefers-color-scheme: dark)");

    const applyTheme = () => {
      document.documentElement.setAttribute("data-theme", "dark");
      document.documentElement.style.colorScheme = "dark";
    };

    applyTheme();

    const onSystemThemeChange = () => applyTheme();
    if (typeof media.addEventListener === "function") {
      media.addEventListener("change", onSystemThemeChange);
      return () => media.removeEventListener("change", onSystemThemeChange);
    }
    media.addListener(onSystemThemeChange);
    return () => media.removeListener(onSystemThemeChange);
  }, []);

  useEffect(() => {
    let cancelled = false;
    let activeController: AbortController | null = null;

    const checkServer = () => {
      if (cancelled || healthPollInFlightRef.current) return;
      const controller = new AbortController();
      activeController = controller;
      healthPollInFlightRef.current = true;

      fetch(`${API_BASE}/health`, { signal: controller.signal, cache: "no-store" })
        .then(async (res) => {
          if (cancelled) return;
          const payload = await res.json().catch(() => null);
          if (payload && typeof payload === "object") {
            setHealthSnapshot(payload as Record<string, unknown>);
          }
          setServerStatus(res.ok ? "online" : "offline");
        })
        .catch(() => {
          if (!cancelled) setServerStatus("offline");
        })
        .finally(() => {
          if (activeController === controller) {
            activeController = null;
          }
          healthPollInFlightRef.current = false;
        });
    };

    checkServer();
    const intervalId = setInterval(checkServer, 5000);
    return () => {
      cancelled = true;
      clearInterval(intervalId);
      activeController?.abort();
      activeController = null;
      healthPollInFlightRef.current = false;
    };
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
        const data = await fetchDevLlmOptionsOnce();
        if (!cancelled && data) {
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
        const data = await fetchServerVersionOnce();
        if (!cancelled && data) setServerVersion(data);
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
      setDevCloudProvider((prev) => (providerChoices.includes(prev) ? prev : effectiveProvider));
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

  function handleSubmit(payload: AskPayload) {
    setLastPayload(payload);
    autoScrolledRef.current = false;
    setBookingActionMessage(null);
    setBookingActionError(null);
    setBookingAuthMode(null);
    setBookingResolveStateByRow({});
    bookingResolveInflightRef.current.clear();
    setShowAllInventory(false);
    setBookingItems([]);
    setPriceAlerts([]);
    void refreshBookingHandoffCapabilities();
    if (asyncMode) {
      reset();
      asyncJob.clearJob();
      asyncJob.startJob(payload);
      return;
    }
    asyncJob.clearJob();
    start(payload);
  }

  const debugInfo = (finalJson?.debug_info ?? null) as TripPlan["debug_info"];
  const routeSummary = buildRouteSummary(debugInfo, lastPayload);
  const finalTopFlightsFromDebug = Array.isArray(debugInfo?.top_flights) ? (debugInfo.top_flights as Flight[]) : undefined;
  const finalTopFlights = finalTopFlightsFromDebug || (Array.isArray(finalJson?.top_flights) ? finalJson.top_flights : undefined);
  const rankedShortlist = Array.isArray(finalTopFlights) && finalTopFlights.length > 0 ? finalTopFlights : undefined;
  const finalFlightsFromDebug = Array.isArray(debugInfo?.all_flights) ? (debugInfo.all_flights as Flight[]) : undefined;
  const finalFlights = finalFlightsFromDebug || (Array.isArray(finalJson?.all_flights) ? finalJson.all_flights : undefined);
  const inventoryFlights = Array.isArray(finalFlights) && finalFlights.length > 0 ? finalFlights : undefined;
  const partialRankedShortlist =
    Array.isArray(partialTopFlights) && partialTopFlights.length > 0 ? partialTopFlights : undefined;
  const multiCityLegs = Array.isArray(finalJson?.legs) ? (finalJson.legs as MultiCityLeg[]) : [];
  const isMultiCity = Boolean(finalJson?.multicity) || multiCityLegs.length > 0;
  const multiCityFlights = multiCityLegs
    .map((leg) => leg.best_flight)
    .filter((flight): flight is Flight => Boolean(flight));
  const flights =
    showAllInventory && inventoryFlights
      ? inventoryFlights
      : rankedShortlist ||
        inventoryFlights ||
        (isMultiCity ? multiCityFlights : undefined) ||
        partialRankedShortlist ||
        partialFlights ||
        undefined;
  const bestFlight = finalJson?.best_flight || (isMultiCity ? multiCityFlights[0] : undefined) || partialBestFlight || undefined;
  const hasFlights = Array.isArray(flights) && flights.length > 0;
  const flightsCount = Array.isArray(flights) ? flights.length : 0;
  const hasRankedShortlist = Boolean(rankedShortlist && rankedShortlist.length > 0);
  const flightCountsFromDebug = (debugInfo?.flight_counts ?? finalJson?.flight_counts ?? null) as Record<string, number> | null;
  const rawProviderCount = typeof flightCountsFromDebug?.raw_provider === "number" ? flightCountsFromDebug.raw_provider : 0;
  const postFilterCount = typeof flightCountsFromDebug?.post_filter === "number" ? flightCountsFromDebug.post_filter : 0;
  const truthfulInventoryCount = rawProviderCount > 0
    ? rawProviderCount
    : postFilterCount > 0
      ? postFilterCount
      : (Array.isArray(inventoryFlights) ? inventoryFlights.length : flightsCount);
  const inventoryCount = truthfulInventoryCount;
  const isStreamingOrPending = isStreaming || isFallback || ["queued", "running"].includes(String(asyncJob.status));
  const isStreamingPhase = isStreamingOrPending && !finalJson;
  const shortlistCountLabel = hasFlights
    ? isStreamingOrPending
      ? `Loading flights from provider... ${flightsCount} so far.`
      : hasRankedShortlist
        ? flightsCount < inventoryCount
          ? `Showing top ${flightsCount} ranked flight${flightsCount === 1 ? "" : "s"} of ${inventoryCount} from provider.`
          : `Showing all ${flightsCount} ranked flight${flightsCount === 1 ? "" : "s"} from provider.`
        : flightsCount < inventoryCount
          ? `Showing ${flightsCount} flight${flightsCount === 1 ? "" : "s"} of ${inventoryCount} from provider.`
          : `Showing all ${flightsCount} ranked flight${flightsCount === 1 ? "" : "s"} from provider.`
    : isStreamingOrPending
      ? "Searching provider inventory..."
      : "Ranked shortlist appears once flight inventory is available.";
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
  const returnTripDepartDate =
    typeof returnTrip?.search_date === "string" && returnTrip.search_date.trim().length > 0
      ? returnTrip.search_date
      : typeof debugInfo?.intent?.return_date === "string"
        ? debugInfo.intent.return_date
        : "";
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
  const intentRecord = (debugInfo?.intent ?? {}) as Record<string, unknown>;
  type BookingRouteContext = {
    origin: string;
    destination: string;
    depart_date: string;
    return_date?: string;
  };
  const buildBookingResolveRowKey = (flight: Flight, routeContext?: BookingRouteContext) => {
    const contextOrigin = (routeContext?.origin || actionOrigin || "").trim().toUpperCase();
    const contextDestination = (routeContext?.destination || actionDestination || "").trim().toUpperCase();
    const contextDepartDate = (routeContext?.depart_date || actionDepartDate || "").trim();
    const contextReturnDate = (routeContext?.return_date || actionReturnDate || "").trim();
    return [
      contextOrigin,
      contextDestination,
      contextDepartDate,
      contextReturnDate,
      bookingFlightIdentity(flight),
    ].join("::");
  };
  const isResolveFailureRetryable = ({
    blockedReason,
    blockedCategory,
    retryableFlag,
  }: {
    blockedReason?: string | null;
    blockedCategory?: string | null;
    retryableFlag?: boolean | null;
  }): boolean => {
    // Backend retryable flag is authoritative when present.
    if (typeof retryableFlag === "boolean") return retryableFlag;
    // Fallback heuristics only when backend didn't classify.
    const category = String(blockedCategory || "").trim().toLowerCase();
    const reason = String(blockedReason || "").trim().toLowerCase();
    if (category === "allowlist_policy") return false;
    if (reason === "booking_token_missing") return false;
    if (reason === "provider_domain_not_allowlisted") return false;
    return true;
  };
  const upsertBookingResolveState = (rowKey: string, next: BookingResolveState) => {
    setBookingResolveStateByRow((prev) => ({ ...prev, [rowKey]: next }));
  };
  const actionOrigin =
    toIata(intentRecord.origin_iata) ||
    toIata(routeLabels.origin_iata) ||
    toIata(lastPayload?.origin);
  const actionDestination =
    toIata(intentRecord.destination_iata) ||
    toIata(routeLabels.destination_iata) ||
    toIata(lastPayload?.destination);
  const actionDepartDate =
    typeof finalRecord?.search_date === "string"
      ? finalRecord.search_date
      : typeof intentRecord.date === "string"
        ? intentRecord.date
        : lastPayload?.date || "";
  const actionReturnDate =
    typeof intentRecord.return_date === "string" ? intentRecord.return_date : undefined;
  const canActionRouteContext = Boolean(actionOrigin && actionDestination && actionDepartDate);
  const returnRouteContext: BookingRouteContext | null =
    actionDestination && actionOrigin && returnTripDepartDate
      ? {
          origin: actionDestination,
          destination: actionOrigin,
          depart_date: returnTripDepartDate,
        }
      : null;
  const returnTripHasDirectHandoff = Boolean(
    typeof returnTripFlight?.handoff_url === "string" && returnTripFlight.handoff_url.trim().length > 0
  );
  const returnTripHasBookingToken = Boolean(
    typeof returnTripFlight?.booking_token === "string" && returnTripFlight.booking_token.trim().length > 0
  );
  const isBusy = isStreaming || isFallback || asyncJobActive;
  const activeBookingToken = bookingAuthToken.trim();
  const hasBookingAuth = activeBookingToken.length > 0;
  const capabilityResolveAvailable = Boolean(bookingHandoffCapabilities?.resolve_available_now);
  const capabilityMissingTokenWhileConfigured = Boolean(
    hasBookingAuth &&
      bookingHandoffCapabilities?.blocked_reason === "missing_token" &&
      !bookingHandoffCapabilities?.auth_rejected
  );
  const canResolveHandoffNow =
    capabilityResolveAvailable || capabilityMissingTokenWhileConfigured || (!bookingHandoffCapabilities && hasBookingAuth);
  const canBookingActions = canActionRouteContext && hasBookingAuth;
  const bookingResolveBlockedReason = (() => {
    if (!canActionRouteContext) {
      return "Booking handoff requires route/date context for this selection.";
    }
    if (canResolveHandoffNow) {
      return null;
    }
    if (bookingHandoffCapabilities?.blocked_reason === "invalid_token") {
      return "Configured token was rejected by backend auth. Update it to match AUTH_TOKEN / AUTH_BEARER_TOKENS.";
    }
    if (bookingHandoffCapabilities?.blocked_reason === "missing_token") {
      return "No bearer token was provided for booking handoff resolution. Set the same backend bearer token used by curl.";
    }
    return (
      bookingHandoffCapabilities?.message ||
      "Authentication required for booking handoff resolution. Configure a bearer token, or enable local-dev unauthenticated mode explicitly."
    );
  })();
  const returnTripResolveBlockedReason = !returnRouteContext
    ? "Missing return-leg route/date context for booking resolution."
    : canResolveHandoffNow
      ? null
      : bookingHandoffCapabilities?.message ||
        "Authentication required for return-leg booking handoff resolution.";
  const returnTripCanResolveOnClick = Boolean(
    returnTripFlight &&
      !returnTripHasDirectHandoff &&
      returnTripHasBookingToken &&
      returnRouteContext &&
      canResolveHandoffNow
  );
  const returnTripBookingHint = returnTripHasDirectHandoff
    ? "Return leg provider handoff is ready."
    : returnTripCanResolveOnClick
      ? "Return handoff is deferred and can be resolved on click."
      : returnTripHasBookingToken
        ? returnTripResolveBlockedReason || "Return handoff can be attempted on demand."
        : "Return booking handoff is unavailable for this row.";
  const outboundRouteContext: BookingRouteContext | undefined = canActionRouteContext
    ? {
        origin: actionOrigin,
        destination: actionDestination,
        depart_date: actionDepartDate,
        return_date: actionReturnDate,
      }
    : undefined;
  const hydratedFlightsForUi = useMemo(() => {
    if (!Array.isArray(flights) || flights.length === 0) return flights;
    return flights.map((flight) => {
      const rowKey = buildBookingResolveRowKey(flight, outboundRouteContext);
      const rowState = bookingResolveStateByRow[rowKey];
      if (!rowState) return flight;
      if (rowState.status === "resolved" && rowState.handoff_url) {
        return {
          ...flight,
          handoff_url: rowState.handoff_url,
          booking_handoff: {
            ...(flight.booking_handoff || {}),
            status: "booking_ready",
            booking_exit_quality: "booking_ready",
            reason: "resolved_booking_handoff_row_state",
            source: "booking_handoff_resolve",
          },
        } as Flight;
      }
      if (rowState.status === "failed") {
        return {
          ...flight,
          booking_handoff: {
            ...(flight.booking_handoff || {}),
            status: "unavailable",
            booking_exit_quality: "unavailable",
            reason: rowState.blocked_reason || "booking_handoff_unavailable",
          },
        } as Flight;
      }
      return flight;
    });
  }, [flights, bookingResolveStateByRow, outboundRouteContext]);
  const bookingActionDisabledHint = hasBookingAuth
    ? null
    : "Hold/Track actions require API auth. Set the same backend bearer token used by curl (or configure VITE_AUTH_TOKEN).";
  const bookingCapabilityBlockedReason = bookingHandoffCapabilities?.blocked_reason || null;
  const showBookingTokenSetupActions = Boolean(
    !canResolveHandoffNow &&
      (
        bookingCapabilityBlockedReason === "missing_token" ||
        bookingCapabilityBlockedReason === "invalid_token" ||
        (!bookingHandoffCapabilities && !hasBookingAuth)
      )
  );
  const bookingTokenSetupHint =
    bookingCapabilityBlockedReason === "invalid_token"
      ? "Configured token was rejected by backend auth. Update it to match AUTH_TOKEN / AUTH_BEARER_TOKENS."
      : hasBookingAuth && bookingCapabilityBlockedReason === "missing_token"
        ? "A booking token is stored in the browser, but backend still reports missing Authorization. Re-save token and retry."
      : "Set the same backend bearer token used by curl to enable browser Book requests.";
  const actionDisabled = isBookingActionBusy || isBusy;
  const bookingModeText = describeBookingMode({
    capabilities: bookingHandoffCapabilities,
    hasBookingAuth,
    bookingAuthMode,
  });
  const getFlightRowKey = (flight: Flight) => buildBookingResolveRowKey(flight, outboundRouteContext);

  const upsertBooking = (booking: BookingRecord) => {
    setBookingItems((prev) => {
      const next = prev.filter((item) => item.id !== booking.id);
      return [booking, ...next].slice(0, 20);
    });
  };

  const describeBooking = (booking: BookingRecord) => {
    const flightRecord =
      booking.flight && typeof booking.flight === "object" ? (booking.flight as Record<string, unknown>) : {};
    const airline = toText(flightRecord.airline);
    const flightNo = toText(flightRecord.flight_no);
    const origin = toIata(flightRecord.origin);
    const destination = toIata(flightRecord.destination);
    const departTime = toText(flightRecord.departure_time);
    const arrivalTime = toText(flightRecord.arrival_time);
    const dateText = toText(flightRecord.date);
    const priceValue = flightRecord.price_inr;
    const priceText = typeof priceValue === "number" ? formatPriceINR(priceValue) : "";
    const title = [airline, flightNo].filter(Boolean).join(" ").trim() || `Booking #${booking.id}`;
    const routeText = origin && destination ? `${origin} → ${destination}` : "Route unavailable";
    const summaryPieces = [routeText, dateText].filter(Boolean).join(" · ");
    const timeWindow = departTime && arrivalTime ? `${departTime} → ${arrivalTime}` : "";
    return {
      title,
      summary: [summaryPieces, timeWindow].filter(Boolean).join(" · "),
      priceText,
    };
  };

  // Scope booking items to the current query route so historical holds from
  // different routes/dates don't clutter the panel (#4/#5/#10/#6/#7).
  const currentRouteBookingItems = useMemo(() => {
    if (!actionOrigin && !actionDestination && !actionDepartDate) return bookingItems;
    let items = bookingItems.filter((booking) => {
      const f = booking.flight && typeof booking.flight === "object" ? (booking.flight as Record<string, unknown>) : {};
      const bOrigin = toIata(f.origin);
      const bDest = toIata(f.destination);
      const bDate = typeof f.date === "string" ? f.date.trim() : "";
      // If booking has no route info at all, keep it (legacy).
      if (!bOrigin && !bDest && !bDate) return true;
      const routeMatch =
        (!actionOrigin || bOrigin === actionOrigin) &&
        (!actionDestination || bDest === actionDestination);
      // If we have a depart date context, also require date match to exclude
      // historical HELD records from previous searches on the same route.
      const dateMatch = !actionDepartDate || !bDate || bDate === actionDepartDate;
      return routeMatch && dateMatch;
    });
    // Deduplicate: keep only the newest HELD record per flight identity
    // (airline + flight_no + date).  Older duplicates are hidden.
    const seen = new Map<string, BookingRecord>();
    for (const b of items) {
      const f = b.flight && typeof b.flight === "object" ? (b.flight as Record<string, unknown>) : {};
      const key = `${toText(f.airline)}::${toText(f.flight_no)}::${toText(f.date)}`;
      const existing = seen.get(key);
      if (!existing || b.id > existing.id) {
        seen.set(key, b);
      }
    }
    return Array.from(seen.values());
  }, [bookingItems, actionOrigin, actionDestination, actionDepartDate]);

  // Similarly scope price alerts to the current route AND date.
  const currentRoutePriceAlerts = useMemo(() => {
    if (!actionOrigin && !actionDestination && !actionDepartDate) return priceAlerts;
    return priceAlerts.filter((alert) => {
      const aOrigin = toIata(alert.origin);
      const aDest = toIata(alert.destination);
      const aDate = typeof alert.travel_date === "string" ? alert.travel_date.trim() : "";
      const routeMatch =
        (!actionOrigin || aOrigin === actionOrigin) &&
        (!actionDestination || aDest === actionDestination);
      const dateMatch = !actionDepartDate || !aDate || aDate === actionDepartDate;
      return routeMatch && dateMatch;
    });
  }, [priceAlerts, actionOrigin, actionDestination, actionDepartDate]);

  const buildBookingActionPayload = (flight: Flight) => {
    if (!actionOrigin || !actionDestination || !actionDepartDate) {
      return null;
    }
    const enrichedFlight = {
      ...flight,
      origin: actionOrigin,
      destination: actionDestination,
      date: actionDepartDate,
    };
    return {
      flight: enrichedFlight,
      origin: actionOrigin,
      destination: actionDestination,
      depart_date: actionDepartDate,
      return_date: actionReturnDate,
    };
  };

  const buildBookingResolvePayload = (flight: Flight, routeContext?: BookingRouteContext) => {
    const effectiveOrigin = routeContext?.origin || actionOrigin;
    const effectiveDestination = routeContext?.destination || actionDestination;
    const effectiveDepartDate = routeContext?.depart_date || actionDepartDate;
    const effectiveReturnDate = routeContext?.return_date ?? actionReturnDate;
    if (!effectiveOrigin || !effectiveDestination || !effectiveDepartDate) {
      return null;
    }
    return {
      flight,
      origin: effectiveOrigin,
      destination: effectiveDestination,
      depart_date: effectiveDepartDate,
      return_date: effectiveReturnDate,
      passengers: 1,
    };
  };
  const bookingAuthTokenForRequests = hasBookingAuth ? activeBookingToken : undefined;

  const resolveBookingForFlight = async (
    flight: Flight,
    routeContext?: BookingRouteContext
  ): Promise<BookingResolveHandoffResponse> => {
    const rowKey = buildBookingResolveRowKey(flight, routeContext);

    // Fast path: if the flight already has a resolved handoff_url from the
    // streaming response, skip the API call entirely and return the existing URL.
    const existingHandoffUrl = typeof flight.handoff_url === "string" ? flight.handoff_url.trim() : "";
    const existingMeta = flight.booking_handoff as Record<string, unknown> | undefined;
    const existingStatus = String(existingMeta?.status || "").toLowerCase();
    if (existingHandoffUrl && existingStatus === "booking_ready") {
      const normalizedUrl = resolveApiUrl(existingHandoffUrl);
      upsertBookingResolveState(rowKey, {
        status: "resolved",
        message: "Provider handoff URL already available from search results.",
        handoff_url: normalizedUrl,
        blocked_reason: null,
        blocked_category: null,
        retryable: false,
        updated_at: Date.now(),
      });
      return {
        action: "resolve_booking_handoff",
        success: true,
        handoff_url: normalizedUrl,
        booking_handoff: existingMeta || {},
        blocked_reason: null,
        blocked_category: null,
        retryable: false,
        message: "Provider handoff URL already available from search results.",
        auth_mode: bookingAuthMode || "authenticated_token",
        auth_required: false,
        owner_principal_id: null,
        best_flight: flight,
      };
    }

    const payload = buildBookingResolvePayload(flight, routeContext);
    if (!payload) {
      upsertBookingResolveState(rowKey, {
        status: "failed",
        message: "Missing route or date context — resubmit with a specific origin, destination, and date.",
        blocked_reason: "missing_route_context",
        blocked_category: "provider_unavailable",
        retryable: false,
        updated_at: Date.now(),
      });
      throw new Error("Missing route/date context required to resolve booking handoff.");
    }
    const existingInflight = bookingResolveInflightRef.current.get(rowKey);
    if (existingInflight) {
      return existingInflight;
    }
    setBookingActionError(null);
    setBookingActionMessage(null);
    upsertBookingResolveState(rowKey, {
      status: "resolving",
      message: "Resolving provider handoff for this row...",
      retryable: true,
      updated_at: Date.now(),
    });
    let resolveFailureClassified = false;
    const runResolve = async () => {
      let data: BookingResolveHandoffResponse;
      const capabilityAuthMode = bookingHandoffCapabilities?.resolve_auth_mode === "omit" ? "omit" : "auto";
      try {
        data = await postJson<BookingResolveHandoffResponse>("/booking/handoff/resolve", payload, {
          authMode: capabilityAuthMode,
          authToken: bookingAuthTokenForRequests,
          timeoutMs: 18000,
        });
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : "Booking handoff request failed.";
        let shouldRetryUnauthed = false;
        if (isInvalidAuthErrorMessage(message)) {
          shouldRetryUnauthed = Boolean(
            bookingHandoffCapabilities?.local_dev_unauth_available ||
              bookingHandoffCapabilities?.auth_mode === "local_dev_unauthed"
          );
          if (!shouldRetryUnauthed) {
            try {
              const fallbackCapabilities = await getJson<BookingHandoffCapabilities>(
                `/booking/handoff/capabilities?ts=${Date.now()}`,
                {
                  authMode: "omit",
                  cache: "no-store",
                  timeoutMs: 8000,
                }
              );
              setBookingHandoffCapabilities(fallbackCapabilities);
              if (
                fallbackCapabilities?.auth_mode === "authenticated_token" ||
                fallbackCapabilities?.auth_mode === "local_dev_unauthed"
              ) {
                setBookingAuthMode(fallbackCapabilities.auth_mode);
              }
              shouldRetryUnauthed = Boolean(
                fallbackCapabilities?.resolve_available_now &&
                  (
                    fallbackCapabilities?.auth_mode === "local_dev_unauthed" ||
                    fallbackCapabilities?.resolve_auth_mode === "omit"
                  )
              );
            } catch {
              shouldRetryUnauthed = false;
            }
          }
        }
        if (!shouldRetryUnauthed) {
          const lowered = message.toLowerCase();
          const timedOut = lowered.includes("abort") || lowered.includes("timeout");
          upsertBookingResolveState(rowKey, {
            status: "failed",
            message: timedOut
              ? "Booking resolve timed out. Retry this row."
              : message || "Booking handoff request failed.",
            blocked_reason: timedOut ? "resolve_request_timeout" : "resolve_request_failed",
            blocked_category: timedOut ? "request_exception" : "provider_unavailable",
            retryable: true,
            updated_at: Date.now(),
          });
          resolveFailureClassified = true;
          throw err;
        }
        data = await postJson<BookingResolveHandoffResponse>("/booking/handoff/resolve", payload, {
          authMode: "omit",
          authToken: bookingAuthTokenForRequests,
          timeoutMs: 18000,
        });
      }
      const authMode = data?.auth_mode;
      if (authMode === "authenticated_token" || authMode === "local_dev_unauthed") {
        setBookingAuthMode(authMode);
      }
      if (!data?.success || !data?.handoff_url) {
        const blockedReason = typeof data?.blocked_reason === "string" ? data.blocked_reason.trim() : "";
        const blockedCategory = typeof data?.blocked_category === "string" ? data.blocked_category.trim() : "";
        const retryable = isResolveFailureRetryable({
          blockedReason,
          blockedCategory,
          retryableFlag: data?.retryable,
        });
        const fallbackMessage =
          blockedReason === "booking_token_missing"
            ? "This flight row has no provider booking token."
            : "Provider handoff is unavailable for this itinerary.";
        const message = data?.message || fallbackMessage;
        upsertBookingResolveState(rowKey, {
          status: "failed",
          message,
          blocked_reason: blockedReason || "booking_handoff_unavailable",
          blocked_category: blockedCategory || "provider_unavailable",
          retryable,
          updated_at: Date.now(),
        });
        resolveFailureClassified = true;
        throw new Error(message);
      }
      const normalizedHandoffUrl = resolveApiUrl(data.handoff_url);
      upsertBookingResolveState(rowKey, {
        status: "resolved",
        message: data.message || "Provider handoff resolved.",
        handoff_url: normalizedHandoffUrl,
        blocked_reason: null,
        blocked_category: null,
        retryable: false,
        updated_at: Date.now(),
      });
      setBookingActionMessage(data.message || "Provider handoff resolved.");
      // Sync: patch any matching held record in local state so the booking panel shows
      // the resolved checkout link without needing a manual Refresh click.
      setBookingItems((prev) =>
        prev.map((item) => {
          if (item.handoff_url) return item; // already has a URL — don't overwrite
          const itemFlight = item.flight as (Flight & Record<string, unknown>) | undefined;
          if (!itemFlight) return item;
          const itemIdentity = bookingFlightIdentity(itemFlight as Flight);
          const resolvedIdentity = bookingFlightIdentity(flight);
          if (itemIdentity !== resolvedIdentity) return item;
          return {
            ...item,
            handoff_url: normalizedHandoffUrl,
            checkout_ready: true,
            checkout_status: "booking_ready",
          };
        })
      );
      return { ...data, handoff_url: normalizedHandoffUrl };
    };
    const resolvePromise = runResolve().finally(() => {
      bookingResolveInflightRef.current.delete(rowKey);
    });
    bookingResolveInflightRef.current.set(rowKey, resolvePromise);
    try {
      return await resolvePromise;
    } catch (err) {
      if (!resolveFailureClassified) {
        upsertBookingResolveState(rowKey, {
          status: "failed",
          message: err instanceof Error ? err.message : "Booking handoff failed.",
          blocked_reason: "booking_handoff_failed",
          blocked_category: "provider_unavailable",
          retryable: true,
          updated_at: Date.now(),
        });
      }
      throw err;
    }
  };

  const openResolvedHandoffInNewTab = async (
    resolveFn: () => Promise<BookingResolveHandoffResponse>,
    fallbackError: string
  ) => {
    const pendingTab = window.open("about:blank", "_blank");
    const renderPendingTab = (title: string, html: string) => {
      if (!pendingTab || pendingTab.closed) return;
      try {
        pendingTab.opener = null;
        pendingTab.document.title = title;
        pendingTab.document.body.innerHTML = html;
      } catch {
        // Ignore browser-specific rendering restrictions.
      }
    };
    if (pendingTab) {
      renderPendingTab(
        "Resolving booking handoff",
        '<main style="font-family: ui-sans-serif, system-ui, sans-serif; padding: 24px; color: #1f2937;">' +
          '<h2 style="margin: 0 0 8px 0; font-size: 18px;">Resolving booking handoff</h2>' +
          '<p style="margin: 0;">Fetching a secure checkout URL from the provider. This tab will redirect automatically when ready.</p>' +
        "</main>"
      );
    }
    try {
      const resolved = await resolveFn();
      const handoffUrl = typeof resolved?.handoff_url === "string" ? resolved.handoff_url.trim() : "";
      if (!handoffUrl) {
        throw new Error(resolved?.message || fallbackError);
      }
      const resolvedUrl = resolveApiUrl(handoffUrl);
      if (pendingTab && !pendingTab.closed) {
        pendingTab.location.replace(resolvedUrl);
      } else {
        window.location.assign(resolvedUrl);
      }
    } catch (err) {
      // Close the pending tab on failure so the user doesn't need to dismiss it manually.
      // The caller sets bookingActionError inline with the error message.
      try { pendingTab?.close(); } catch { /* ignore */ }
      throw err;
    }
  };

  const refreshBookingHandoffCapabilities = async (tokenOverride?: string) => {
    const requestToken =
      typeof tokenOverride === "string" ? tokenOverride.trim() : bookingAuthTokenForRequests;
    const capabilitiesPath = `/booking/handoff/capabilities?ts=${Date.now()}`;
    try {
      const data = await getJson<BookingHandoffCapabilities>(capabilitiesPath, {
        authToken: requestToken,
        cache: "no-store",
      });
      setBookingHandoffCapabilities(data);
      if (data?.auth_mode === "authenticated_token" || data?.auth_mode === "local_dev_unauthed") {
        setBookingAuthMode(data.auth_mode);
      } else {
        setBookingAuthMode(null);
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unable to load booking handoff capabilities.";
      if (isInvalidAuthErrorMessage(message)) {
        try {
          const fallbackData = await getJson<BookingHandoffCapabilities>(capabilitiesPath, {
            authMode: "omit",
            cache: "no-store",
          });
          setBookingHandoffCapabilities(fallbackData);
          if (fallbackData?.auth_mode === "authenticated_token" || fallbackData?.auth_mode === "local_dev_unauthed") {
            setBookingAuthMode(fallbackData.auth_mode);
          } else {
            setBookingAuthMode(null);
          }
          return;
        } catch {
          // fall back to explicit blocked state below
        }
        setBookingHandoffCapabilities({
          action: "booking_handoff_capabilities",
          resolve_available_now: false,
          auth_mode: "auth_required",
          resolve_auth_mode: "auto",
          auth_required: true,
          has_valid_token: false,
          token_present: Boolean(requestToken),
          auth_rejected: true,
          auth_error: "invalid_token",
          blocked_reason: "invalid_token",
          local_dev_unauth_configured: false,
          local_dev_unauth_enabled: false,
          loopback_request: false,
          loopback_eligible: false,
          local_dev_unauth_available: false,
          message,
        });
        return;
      }
      setBookingHandoffCapabilities(null);
    }
  };

  const synchronizeBookingAuthToken = () => {
    setBookingAuthToken(getConfiguredAuthToken());
  };

  const configureBookingAuthToken = async () => {
    if (typeof window === "undefined") return;
    const currentToken = getConfiguredAuthToken();
    const provided = window.prompt(
      "Paste the backend bearer token used by curl (token value only, no \"Bearer\" prefix).",
      currentToken
    );
    if (provided === null) return;
    const normalized = setConfiguredAuthToken(provided);
    setBookingAuthToken(normalized);
    setBookingActionError(null);
    setBookingActionMessage(
      normalized
        ? "Booking auth token saved in browser storage for lazy Book requests."
        : "Booking auth token cleared from browser storage."
    );
    await refreshBookingHandoffCapabilities(normalized);
  };

  const removeBookingAuthToken = async () => {
    clearConfiguredAuthToken();
    setBookingAuthToken("");
    setBookingActionError(null);
    setBookingActionMessage("Booking auth token cleared from browser storage.");
    await refreshBookingHandoffCapabilities("");
  };

  const refreshBookings = async () => {
    if (!hasBookingAuth) {
      setBookingItems([]);
      setBookingActionError(null);
      return;
    }
    try {
      const data = await getJson<{ items: BookingRecord[] }>(`/bookings?limit=50`, {
        authToken: bookingAuthTokenForRequests,
      });
      const items = data.items || [];
      setBookingItems(items);
      // Sync: if any held record has a persisted handoff_url, propagate it back into
      // the shortlist resolve state so the flight card "Book now" link stays in sync.
      if (outboundRouteContext && Array.isArray(flights) && flights.length > 0) {
        for (const booking of items) {
          if (!booking.handoff_url || booking.status !== "HELD") continue;
          const bookingFlight = booking.flight as (Flight & Record<string, unknown>) | undefined;
          if (!bookingFlight) continue;
          const identity = bookingFlightIdentity(bookingFlight as Flight);
          for (const flight of flights) {
            if (bookingFlightIdentity(flight) === identity) {
              const rowKey = buildBookingResolveRowKey(flight, outboundRouteContext);
              const current = bookingResolveStateByRow[rowKey];
              if (!current || (current.status !== "resolved" && current.status !== "resolving")) {
                upsertBookingResolveState(rowKey, {
                  status: "resolved",
                  message: "Provider handoff resolved (from held record).",
                  handoff_url: booking.handoff_url,
                  blocked_reason: null,
                  blocked_category: null,
                  retryable: false,
                  updated_at: Date.now(),
                });
              }
              break;
            }
          }
        }
      }
    } catch (err: unknown) {
      setBookingActionError(err instanceof Error ? err.message : "Failed to refresh bookings");
    }
  };

  const handleHoldFlight = async (flight: Flight) => {
    if (!hasBookingAuth) {
      setBookingActionError("Booking actions require API auth token.");
      return;
    }
    const payload = buildBookingActionPayload(flight);
    if (!payload) {
      setBookingActionError("Missing route/date for booking action.");
      return;
    }
    setIsBookingActionBusy(true);
    setBookingActionError(null);
    setBookingActionMessage(null);
    try {
      const data = await postJson<BookingActionResponse>("/booking/hold", payload, {
        authToken: bookingAuthTokenForRequests,
      });
      if (data?.booking) upsertBooking(data.booking);
      setBookingActionMessage(data?.message || "Flight held successfully.");
      // Sync: if hold resolved a handoff URL, update the shortlist row state so the card
      // immediately shows a "Book now" link instead of the deferred resolve button.
      const holdHandoffUrl = typeof data?.booking?.handoff_url === "string" ? data.booking.handoff_url.trim() : "";
      if (holdHandoffUrl && outboundRouteContext) {
        const rowKey = buildBookingResolveRowKey(flight, outboundRouteContext);
        upsertBookingResolveState(rowKey, {
          status: "resolved",
          message: data?.message || "Provider handoff resolved during hold.",
          handoff_url: resolveApiUrl(holdHandoffUrl),
          blocked_reason: null,
          blocked_category: null,
          retryable: false,
          updated_at: Date.now(),
        });
      }
    } catch (err: unknown) {
      setBookingActionError(err instanceof Error ? err.message : "Hold request failed");
    } finally {
      setIsBookingActionBusy(false);
    }
  };

  const handleTrackFlight = async (flight: Flight) => {
    if (!hasBookingAuth) {
      setBookingActionError("Booking actions require API auth token.");
      return;
    }
    const payload = buildBookingActionPayload(flight);
    if (!payload) {
      setBookingActionError("Missing route/date for tracking.");
      return;
    }
    setIsBookingActionBusy(true);
    setBookingActionError(null);
    setBookingActionMessage(null);
    try {
      const data = await postJson<BookingActionResponse>("/booking/track-price", payload, {
        authToken: bookingAuthTokenForRequests,
      });
      if (data?.booking) upsertBooking(data.booking);
      setBookingActionMessage(data?.message || "Price tracking activated.");
    } catch (err: unknown) {
      setBookingActionError(err instanceof Error ? err.message : "Track-price request failed");
    } finally {
      setIsBookingActionBusy(false);
    }
  };

  const handleCancelBooking = async (bookingId: number) => {
    setIsBookingActionBusy(true);
    setBookingActionError(null);
    setBookingActionMessage(null);
    try {
      const data = await postJson<BookingActionResponse>(
        "/booking/cancel",
        { booking_id: bookingId },
        { authToken: bookingAuthTokenForRequests }
      );
      setBookingActionMessage(data?.message || "Booking cancelled.");
      setBookingItems((prev) =>
        prev.map((item) => (item.id === bookingId ? { ...item, status: data.success ? "CANCELLED" : item.status } : item))
      );
    } catch (err: unknown) {
      setBookingActionError(err instanceof Error ? err.message : "Cancel request failed");
    } finally {
      setIsBookingActionBusy(false);
    }
  };

  const refreshAlerts = async () => {
    if (!hasBookingAuth) {
      setPriceAlerts([]);
      setPriceAlertError(null);
      return;
    }
    try {
      const data = await getJson<{ items: PriceAlert[] }>("/price-tracking/alerts", {
        authToken: bookingAuthTokenForRequests,
      });
      setPriceAlerts(data.items || []);
      setPriceAlertError(null);
    } catch (err: unknown) {
      setPriceAlertError(err instanceof Error ? err.message : "Failed to load alerts");
    }
  };

  const acknowledgeAlert = async (alertId: number) => {
    if (!hasBookingAuth) return;
    try {
      await postJson<{ acknowledged: boolean }>(
        `/price-tracking/alerts/${alertId}/ack`,
        {},
        { authToken: bookingAuthTokenForRequests }
      );
      setPriceAlerts((prev) => prev.filter((item) => item.alert_id !== alertId));
    } catch (err: unknown) {
      setPriceAlertError(err instanceof Error ? err.message : "Failed to acknowledge alert");
    }
  };

  useEffect(() => {
    if (typeof window === "undefined") return;
    const watchedKeys = new Set(["travelyst_auth_token", "AUTH_TOKEN", "auth_token"]);
    const onStorage = (event: StorageEvent) => {
      if (event.storageArea !== window.localStorage) return;
      if (event.key && !watchedKeys.has(event.key)) return;
      synchronizeBookingAuthToken();
    };
    const onTokenChanged = () => {
      synchronizeBookingAuthToken();
    };
    window.addEventListener("storage", onStorage);
    window.addEventListener(BOOKING_AUTH_TOKEN_CHANGED_EVENT, onTokenChanged);
    return () => {
      window.removeEventListener("storage", onStorage);
      window.removeEventListener(BOOKING_AUTH_TOKEN_CHANGED_EVENT, onTokenChanged);
    };
  }, []);

  // eslint-disable-next-line react-hooks/exhaustive-deps -- refreshBookingHandoffCapabilities is an inline function that changes every render; we intentionally fire on token/status changes only.
  useEffect(() => {
    let cancelled = false;
    const loadCapabilities = async () => {
      if (cancelled) return;
      await refreshBookingHandoffCapabilities();
    };
    loadCapabilities();
    return () => {
      cancelled = true;
    };
  }, [bookingAuthToken, serverStatus]);

  useEffect(() => {
    if (!hasBookingAuth) {
      setPriceTrackingStatus(null);
      return;
    }
    let cancelled = false;
    getJson<PriceTrackingStatus>("/price-tracking/status", {
      authToken: bookingAuthTokenForRequests,
    })
      .then((data) => { if (!cancelled) setPriceTrackingStatus(data); })
      .catch(() => { if (!cancelled) setPriceTrackingStatus(null); });
    return () => { cancelled = true; };
  }, [bookingAuthToken, bookingAuthTokenForRequests, hasBookingAuth]);
  useEffect(() => {
    if (!hasBookingAuth) {
      setBookingItems([]);
      setBookingActionError(null);
      return;
    }
    void refreshBookings();
  }, [bookingAuthToken]);

  // Auto-sync HELD record URLs back to shortlist when streaming/job finishes and flights settle.
  const flightsSettledRef = useRef(false);
  useEffect(() => {
    if (isBusy) {
      flightsSettledRef.current = false;
      return;
    }
    if (hasFlights && hasBookingAuth && !flightsSettledRef.current) {
      flightsSettledRef.current = true;
      refreshBookings();
    }
  }, [isBusy, hasFlights, hasBookingAuth]);

  useEffect(() => {
    if (hasBookingAuth && priceTrackingStatus?.enabled) {
      refreshAlerts();
    }
  }, [hasBookingAuth, priceTrackingStatus?.enabled]);
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
  const asyncStatusLabel = asyncJobActive
    ? `Async job ${String(asyncJob.status || "running")}: process-local queue`
    : undefined;
  const asyncJobNotice =
    asyncJob.job && !asyncJobActive
      ? `Async job ${String(asyncJob.status || "done")} ${asyncJob.status === "done" ? "completed" : "stopped"}.`
      : null;
  const highlightWeatherText = weatherData
    ? [typeof weatherData.condition === "string" ? weatherData.condition : "", formatTemperatureC(weatherData.temperature_c)]
        .filter(Boolean)
        .join(" · ")
    : "Weather updates appear once results load.";
  const bestFlightHasPrice = typeof bestFlight?.price_inr === "number" && bestFlight.price_inr > 0;
  const highlights = bestFlight
    ? [
        {
          title: "Best Flight",
          text: `${bestFlight.airline} ${bestFlight.flight_no} · ${bestFlight.departure_time} → ${bestFlight.arrival_time}${bestFlightHasPrice ? ` · ${formatPriceINR(bestFlight.price_inr)}` : ""}`
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
    typeof activeError === "string" &&
    activeError.toLowerCase().includes("available flight/weather results are shown");
  const resultStatus = finalJson?.result_status || responseMeta?.result_status;
  const isDegradedResult = resultStatus === "degraded";
  const noFlightsFailure =
    responseMeta?.failure_reason === "no_flights" || responseMeta?.no_flights_reason === "no_flights";
  const failureDomain = responseMeta?.failure_domain || finalJson?.failure_domain;
  const isProviderLimited = failureDomain === "upstream_provider" || failureDomain === "search_outcome";
  const isAppBroken = failureDomain === "internal_backend";
  const degradedSummary =
    finalJson?.fallback_note ||
    finalJson?.degradation?.message ||
    responseMeta?.fallback_note ||
    responseMeta?.degradation_message ||
    (isProviderLimited
      ? "The provider could not complete this search, but any results shown are still usable."
      : "Some explanation details are unavailable right now, but the trip data shown is still usable.");
  const bestFlightHasHandoff = Boolean(
    typeof bestFlight?.handoff_url === "string" && bestFlight.handoff_url.trim().length > 0
  );
  const bestFlightHandoffQuality = String(
    bestFlight?.booking_handoff?.booking_exit_quality || ""
  ).toLowerCase();
  const bestFlightHandoffLabel =
    bestFlightHandoffQuality === "booking_ready"
      ? "Booking-ready provider handoff"
      : bestFlightHandoffQuality === "deferred"
        ? "Booking deferred"
        : bestFlightHasHandoff
          ? "Provider handoff available"
          : "Booking unavailable from current artifacts";
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
          : isStreamingPhase
            ? "Gathering evidence"
            : isBusy
              ? "Analyzing"
              : "Awaiting query";
  const showProofSurface = Boolean(lastPayload || isBusy || hasFlights || finalJson || activeError);
  const resultsFirstMode = showProofSurface;
  const showExperienceNarrative = !resultsFirstMode;
  const showSearchEnhancers = !resultsFirstMode;
  const showBookingPanel =
    hasBookingAuth &&
    (currentRouteBookingItems.length > 0 || Boolean(bookingActionMessage) || Boolean(bookingActionError) || isBookingActionBusy);
  const showAlertsPanel =
    hasBookingAuth &&
    (currentRoutePriceAlerts.length > 0 || Boolean(priceAlertError) || Boolean(priceTrackingStatus));
  const trackingMeta = (priceTrackingStatus?.status ?? {}) as Record<string, unknown>;
  const trackingLastCompleted =
    typeof trackingMeta.last_completed_at === "string" ? trackingMeta.last_completed_at : "";
  const trackingLastError = typeof trackingMeta.last_error === "string" ? trackingMeta.last_error : "";
  const trackingLastAlerts =
    typeof trackingMeta.last_alert_count === "number" ? trackingMeta.last_alert_count : null;
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

  useEffect(() => {
    if (typeof window === "undefined") return;
    const targets = Array.from(document.querySelectorAll<HTMLElement>(".reveal"));
    if (!targets.length) return;

    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReducedMotion) {
      targets.forEach((target) => target.classList.add("visible"));
      return;
    }

    targets.forEach((target, index) => {
      const revealDelay = `${Math.min(index, 8) * 60}ms`;
      target.style.setProperty("--reveal-delay", revealDelay);
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

    targets.forEach((target) => {
      if (target.classList.contains("visible")) return;
      io.observe(target);
    });
    return () => io.disconnect();
  }, [resultVersion, showExperienceNarrative, showProofSurface]);

  useEffect(() => {
    if (!isBusy && !activeError && finalJson) {
      setResultVersion((v) => v + 1);
    }
  }, [finalJson, isBusy, activeError]);

  useEffect(() => {
    if (!showProofSurface || isBusy || !hasFlights) return;
    if (typeof window === "undefined") return;
    if (resultsAutoScrollVersionRef.current === resultVersion) return;
    resultsSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    resultsAutoScrollVersionRef.current = resultVersion;
  }, [hasFlights, isBusy, resultVersion, showProofSurface]);

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
            {showServiceStatus && (
              <div
                className={`api-status ${serverStatus === "offline" ? "api-status--offline" : "api-status--online"} ${IS_PREVIEW_UI ? "api-status--preview" : ""}`}
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
          <section
            id="planner"
            className={[
              "hero",
              "experience-section",
              "experience-section--hero",
              resultsFirstMode ? "hero--results-mode" : "",
            ]
              .filter(Boolean)
              .join(" ")}
          >
            <div className={["hero-intro", "reveal", resultsFirstMode ? "hero-intro--compact" : ""].filter(Boolean).join(" ")}>
              <div className="hero-badge">
                <span className="badge-dot" aria-hidden="true" />
                {resultsFirstMode ? "Refine and book" : heroBadgeText}
              </div>
              <h1 className={["hero-title", resultsFirstMode ? "hero-title--compact" : ""].filter(Boolean).join(" ")}>
                {resultsFirstMode ? (
                  <>
                    <span className="title-line-1">Your ranked flights are ready.</span>
                    <span className="title-line-2">Pick an option and open provider checkout.</span>
                  </>
                ) : (
                  <>
                    <span className="title-line-1">Plan your next journey</span>
                    <span className="title-line-2">with premium AI travel guidance.</span>
                  </>
                )}
              </h1>
              <p className={["hero-sub", resultsFirstMode ? "hero-sub--compact" : ""].filter(Boolean).join(" ")}>
                {resultsFirstMode
                  ? "The ranked shortlist is the primary surface below. Rows show immediate handoff, lazy resolve-on-click when enabled for this mode, or an explicit blocked/unavailable state."
                  : "Describe the trip in natural language and get ranked flights, weather-aware guidance, and booking-ready options in one polished decision flow."}
              </p>
              {!resultsFirstMode && (
                <div className="hero-trust-row" aria-label="Trust indicators">
                  {heroTrustSignals.map((signal) => (
                    <span key={signal} className="hero-trust-pill">{signal}</span>
                  ))}
                </div>
              )}
            </div>

            <section className="hero-grid">
              <div className="hero-left">
                <div className={["search-card", resultsFirstMode ? "search-card--compact" : ""].filter(Boolean).join(" ")}>
                  <QueryForm
                    onSubmit={handleSubmit}
                    disabled={isBusy}
                    resultVersion={resultVersion}
                    onRecentQueriesChange={setRecentQueries}
                    asyncMode={asyncMode}
                    onAsyncModeChange={setAsyncMode}
                    devRoutingOverrides={
                      isDevMode
                        ? {
                            llm_mode: devLlmMode,
                            cloud_provider: devCloudProvider,
                          }
                        : null
                    }
                  />

                  {activeError && (
                    <div className="notice notice--error notice--inline" data-testid="notice-error">
                      <span className="min-w-0 break-words">
                        {partialOutcomeError
                          ? activeError
                          : noFlightsFailure
                            ? activeError
                            : isProviderLimited
                              ? `Provider limitation: ${activeError}`
                              : isAppBroken
                                ? `Service error: ${activeError}`
                                : `We couldn't finish your plan. ${activeError}`}
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
                  {!activeError && isDegradedResult && (
                    <div className="notice notice--inline" data-testid="notice-inline">
                      <span className="min-w-0 break-words">
                        Partial result: {degradedSummary}
                      </span>
                    </div>
                  )}
                  {hasConstraintWarnings && !activeError && (
                    <div className="notice notice--inline" data-testid="notice-inline">
                      <span className="min-w-0 break-words">
                        Constraint adjustments: {[...resultWarnings, ...returnTripWarnings].join(" ")}
                      </span>
                    </div>
                  )}

                </div>

                {showSearchEnhancers && showHighlights && (
                  <div className="highlights-row" aria-label="Trip highlights">
                    {highlights.map((highlight) => (
                      <article key={highlight.title} className="highlight-card">
                        <p className="highlight-card__title">{highlight.title}</p>
                        <p className="highlight-card__text">{highlight.text}</p>
                      </article>
                    ))}
                  </div>
                )}

                {showSearchEnhancers && (
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
                )}

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
                    canCancel={isStreaming || asyncJobActive}
                    statusText={asyncStatusLabel}
                    onCancel={asyncJobActive ? asyncJob.cancelJob : cancel}
                  />
                </article>
                {asyncJobNotice && (
                  <div className={`notice notice--inline ${asyncJob.status === "error" ? "notice--error" : ""}`}>
                    <span className="min-w-0 break-words">
                      {asyncJobNotice}{" "}
                      {asyncJob.status === "done"
                        ? "Results are now ready below."
                        : asyncJob.status === "cancelled"
                          ? "You can submit a new query at any time."
                          : asyncJob.status === "error"
                            ? "Please retry the request."
                            : ""}
                    </span>
                  </div>
                )}

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
                    <AIReasoningPanel
                      finalJson={finalJson}
                      isStreaming={isBusy}
                      reasoningSteps={reasoningSteps}
                      approvalRequired={approvalRequired}
                      approvalResult={approvalResult}
                      onApprove={respondToApproval}
                    />
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
            {!resultsFirstMode && (
              <a href="#route-reveal" className="hero-scroll-cue">
                How the planner thinks ↓
              </a>
            )}
          </section>

          {showExperienceNarrative && (
            <>
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
            </>
          )}

          <section
            id="results"
            ref={(node) => {
              resultsSectionRef.current = node;
            }}
            className={[
              "experience-section",
              "experience-section--proof",
              resultsFirstMode ? "experience-section--proof-results-first" : "",
            ]
              .filter(Boolean)
              .join(" ")}
          >
            <div className="section-head reveal">
              <p className="section-label">{resultsFirstMode ? "Flight options" : "Product proof"}</p>
              <h2 className="section-title">
                {resultsFirstMode
                  ? "Scan flights quickly and book from the ranked shortlist"
                  : "Choose faster with clear ranking, evidence, and booking confidence"}
              </h2>
            </div>
            {showProofSurface ? (
              <>
                <div className="proof-overview-grid" data-testid="proof-overview">
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
                          {bestFlight.airline} {bestFlight.flight_no}{bestFlightHasPrice ? ` · ${formatPriceINR(bestFlight.price_inr)}` : ""}
                        </h3>
                        <p className="proof-card__summary">
                          {bestFlight.departure_time} → {bestFlight.arrival_time} · {bestFlight.duration_min} min · {bestFlightStopSummary}
                        </p>
                        <div className="proof-chip-row">
                          <span className="proof-chip">{bestFlightHandoffLabel}</span>
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
                    <ul className="proof-evidence-list" data-testid="proof-evidence">
                      <li className="proof-evidence-item">
                        <span className="proof-evidence-item__label">Ranked shortlist</span>
                        <span className="proof-evidence-item__value">
                          {hasFlights ? shortlistCountLabel : isBusy ? "Compiling live options" : "No shortlist yet"}
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
                          {bestFlightHandoffLabel}
                          </span>
                      </li>
                    </ul>
                  </article>
                </div>
                <div className="result-wrap" data-testid="result-wrap">
                  {isMultiCity && multiCityLegs.length > 0 && (
                    <article className="r-card results-card" data-testid="multicity-itinerary">
                      <div className="r-label r-label--secondary">
                        <span className="r-dot" aria-hidden="true" />
                        Multi-city itinerary
                      </div>
                      <MultiCitySummary legs={multiCityLegs} />
                    </article>
                  )}
                  <article className={flightsCardClass} data-testid="ranked-shortlist">
                    <div className="r-label r-label--secondary">
                      <span className="r-dot" aria-hidden="true" />
                      {showAllInventory ? "Full inventory" : "Ranked shortlist"}
                    </div>
                    <p className="results-callout">
                      {shortlistCountLabel} Book opens immediately when a provider handoff URL exists; otherwise Book resolves handoff lazily for that selected row only when current auth/setup mode allows it.
                    </p>
                    {inventoryFlights && inventoryFlights.length > (rankedShortlist?.length || 0) && (
                      <button
                        type="button"
                        className="inventory-toggle-btn"
                        onClick={() => setShowAllInventory((prev) => !prev)}
                        aria-label={showAllInventory ? "Show ranked shortlist only" : "Show all flights from provider"}
                      >
                        {showAllInventory ? `Show top ${rankedShortlist?.length || 0} ranked` : `Show all ${inventoryFlights.length} flights`}
                      </button>
                    )}
                    <p className="results-callout">{bookingModeText}</p>
                    {bookingResolveBlockedReason && (
                      <div className="notice notice--inline" data-testid="notice-inline">
                        <span className="min-w-0 break-words">{bookingResolveBlockedReason}</span>
                      </div>
                    )}
                    {showBookingTokenSetupActions && (
                      <div className="notice notice--inline" data-testid="notice-inline">
                        <span className="min-w-0 break-words">{bookingTokenSetupHint}</span>
                        <div className="booking-auth-token-actions">
                          <button
                            type="button"
                            className="booking-card__refresh"
                            onClick={configureBookingAuthToken}
                            disabled={actionDisabled}
                            data-testid="booking-auth-configure"
                          >
                            {hasBookingAuth ? "Update booking token" : "Set booking token"}
                          </button>
                          {hasBookingAuth && (
                            <button
                              type="button"
                              className="booking-card__refresh"
                              onClick={removeBookingAuthToken}
                              disabled={actionDisabled}
                              data-testid="booking-auth-clear"
                            >
                              Clear token
                            </button>
                          )}
                        </div>
                      </div>
                    )}
                    {!hasBookingAuth && (
                      <div className="notice notice--inline" data-testid="notice-inline">
                        <span className="min-w-0 break-words">{bookingActionDisabledHint}</span>
                      </div>
                    )}
                    <FlightsList
                      flights={hydratedFlightsForUi}
                      bestFlight={bestFlight}
                      isLoading={isBusy && !hasFlights}
                      onBook={canResolveHandoffNow ? (flight) => resolveBookingForFlight(flight) : undefined}
                      bookBlockedReason={bookingResolveBlockedReason || undefined}
                      flightKeyFor={getFlightRowKey}
                      bookingResolveStateByKey={bookingResolveStateByRow}
                      onHold={canBookingActions ? handleHoldFlight : undefined}
                      onTrack={canBookingActions ? handleTrackFlight : undefined}
                      actionDisabled={actionDisabled}
                    />
                  </article>
                  {returnTripFlight && (
                    <article className="r-card results-card" data-testid="return-leg">
                      <div className="r-label r-label--secondary">
                        <span className="r-dot" aria-hidden="true" />
                        Return leg snapshot
                      </div>
                      <p className="flight-item__summary">
                        {returnTripFlight.airline} {returnTripFlight.flight_no} · {returnTripFlight.departure_time} →{" "}
                        {returnTripFlight.arrival_time} · {formatPriceINR(returnTripFlight.price_inr)}
                      </p>
                      <p className="flight-item__meta">{returnTripBookingHint}</p>
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
                      <div className="flight-card__actions">
                        {returnTripHasDirectHandoff && (
                          <a
                            href={resolveApiUrl(returnTripFlight.handoff_url as string)}
                            target="_blank"
                            rel="noreferrer"
                            className="flight-card__link flight-card__link--primary"
                            data-testid="return-booking-link"
                          >
                            Book return now
                          </a>
                        )}
                        {!returnTripHasDirectHandoff && returnTripHasBookingToken && (
                          <>
                            {returnTripCanResolveOnClick ? (
                              <button
                                type="button"
                                className="flight-card__link flight-card__link--primary"
                                disabled={actionDisabled}
                                data-testid="return-booking-resolve"
                                onClick={async () => {
                                  if (!returnRouteContext) {
                                    setBookingActionError("Missing return-leg route/date context for booking resolution.");
                                    return;
                                  }
                                  setIsBookingActionBusy(true);
                                  setBookingActionError(null);
                                  try {
                                    await openResolvedHandoffInNewTab(
                                      () => resolveBookingForFlight(returnTripFlight, returnRouteContext),
                                      "Return-leg booking handoff failed."
                                    );
                                  } catch (err: unknown) {
                                    setBookingActionError(
                                      err instanceof Error ? err.message : "Return-leg booking handoff failed."
                                    );
                                  } finally {
                                    setIsBookingActionBusy(false);
                                  }
                                }}
                              >
                                {isBookingActionBusy ? "Resolving return booking..." : "Book return (resolve on click)"}
                              </button>
                            ) : (
                              <span className="fl-meta" data-testid="return-booking-blocked-note">
                                {returnTripResolveBlockedReason || "Return booking handoff is unavailable for this row."}
                              </span>
                            )}
                          </>
                        )}
                      </div>
                    </article>
                  )}
                  {(showBookingPanel || showAlertsPanel) && (
                    <div className="booking-grid">
                      <article className="r-card results-card booking-card" data-testid="booking-panel">
                        <div className="booking-card__head">
                          <div className="r-label r-label--secondary">
                            <span className="r-dot" aria-hidden="true" />
                            Booking actions
                          </div>
                          <button
                            type="button"
                            className="booking-card__refresh"
                            onClick={refreshBookings}
                            disabled={isBookingActionBusy}
                            data-testid="booking-refresh"
                          >
                            Refresh
                          </button>
                        </div>
                        <p className="booking-card__hint">
                          Hold or track a flight from the shortlist to create a local follow-up record. Complete checkout on provider sites via handoff links.
                        </p>
                        {bookingActionMessage && (
                          <div className="notice notice--inline" data-testid="notice-inline">
                            <span className="min-w-0 break-words">{bookingActionMessage}</span>
                          </div>
                        )}
                        {bookingActionError && (
                          <div className="notice notice--error notice--inline" data-testid="notice-error">
                            <span className="min-w-0 break-words">{bookingActionError}</span>
                          </div>
                        )}
                        {currentRouteBookingItems.length > 0 ? (
                          <div className="booking-list">
                            {currentRouteBookingItems.map((booking) => {
                              const details = describeBooking(booking);
                              const status = String(booking.status || "UNKNOWN").toUpperCase();
                              const statusLabel = status === "CONFIRMED" ? "LEGACY_CONFIRMED" : status;
                              const canCancel = status === "HELD" || status === "CONFIRMED";
                              const checkoutStatus = String(booking.checkout_status || "").toLowerCase();
                              const checkoutUnavailable = !booking.handoff_url && checkoutStatus && checkoutStatus !== "booking_ready";
                              return (
                                <div key={booking.id} className="booking-item">
                                  <div className="booking-item__main">
                                    <div className="booking-item__title-row">
                                      <p className="booking-item__title">{details.title}</p>
                                      <span className={`booking-status booking-status--${statusLabel.toLowerCase()}`}>
                                        {statusLabel}
                                      </span>
                                    </div>
                                    <p className="booking-item__summary">
                                      {details.summary || "Flight details stored for this booking."}
                                    </p>
                                    {details.priceText && (
                                      <p className="booking-item__price">{details.priceText}</p>
                                    )}
                                    {checkoutUnavailable && (
                                      <p className="booking-item__summary" data-testid="checkout-unavailable-note">
                                        Provider checkout link is currently unavailable for this held record.
                                      </p>
                                    )}
                                  </div>
                                  <div className="booking-item__actions">
                                    {booking.handoff_url && (
                                      <a
                                        href={booking.handoff_url}
                                        target="_blank"
                                        rel="noreferrer"
                                        className="booking-action booking-action--primary"
                                      >
                                        Open booking
                                      </a>
                                    )}
                                    {canCancel && (
                                      <button
                                        type="button"
                                        className="booking-action booking-action--ghost"
                                        onClick={() => handleCancelBooking(booking.id)}
                                        disabled={actionDisabled}
                                        data-testid="booking-cancel"
                                      >
                                        Cancel
                                      </button>
                                    )}
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        ) : (
                          <div className="empty-state empty-state--compact">
                            <p className="empty-state__title">No held bookings yet.</p>
                            <p className="empty-state__hint">Use Hold or Track on a flight card to start one.</p>
                          </div>
                        )}
                      </article>
                      <article className="r-card results-card booking-card" data-testid="tracking-panel">
                        <div className="booking-card__head">
                          <div className="r-label r-label--secondary">
                            <span className="r-dot" aria-hidden="true" />
                            Price tracking
                          </div>
                          <button
                            type="button"
                            className="booking-card__refresh"
                            onClick={refreshAlerts}
                            disabled={!priceTrackingStatus?.enabled}
                            data-testid="alerts-refresh"
                          >
                            Refresh
                          </button>
                        </div>
                        <p className="booking-card__hint">
                          {priceTrackingStatus?.enabled
                            ? "Tracking runs on the server. Alerts appear here when prices drop."
                            : "Price tracking is disabled in this environment."}
                        </p>
                        {trackingLastCompleted && (
                          <p className="booking-card__meta">
                            Last check: {trackingLastCompleted}
                            {trackingLastAlerts !== null ? ` · Alerts fired: ${trackingLastAlerts}` : ""}
                          </p>
                        )}
                        {trackingLastError && (
                          <div className="notice notice--error notice--inline" data-testid="notice-error">
                            <span className="min-w-0 break-words">Tracker error: {trackingLastError}</span>
                          </div>
                        )}
                        {priceAlertError && (
                          <div className="notice notice--error notice--inline" data-testid="notice-error">
                            <span className="min-w-0 break-words">{priceAlertError}</span>
                          </div>
                        )}
                        {currentRoutePriceAlerts.length > 0 ? (
                          <div className="alert-list">
                            {currentRoutePriceAlerts.map((alert) => (
                              <div key={alert.alert_id} className="alert-item">
                                <div className="alert-item__main">
                                  <p className="alert-item__title">
                                    {alert.origin} → {alert.destination} · {alert.travel_date}
                                  </p>
                                  <p className="alert-item__summary">
                                    Drop {alert.drop_pct}% · {formatPriceINR(alert.held_price_inr)} →{" "}
                                    {formatPriceINR(alert.new_price_inr)}
                                  </p>
                                </div>
                                <div className="alert-item__actions">
                                  {alert.new_handoff_url && (
                                    <a
                                      href={alert.new_handoff_url}
                                      target="_blank"
                                      rel="noreferrer"
                                      className="booking-action booking-action--primary"
                                    >
                                      View deal
                                    </a>
                                  )}
                                  <button
                                    type="button"
                                    className="booking-action booking-action--ghost"
                                    onClick={() => acknowledgeAlert(alert.alert_id)}
                                    data-testid="alert-ack"
                                  >
                                    Dismiss
                                  </button>
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div className="empty-state empty-state--compact">
                            <p className="empty-state__title">No active price alerts.</p>
                            <p className="empty-state__hint">Track a flight to start monitoring price drops.</p>
                          </div>
                        )}
                      </article>
                    </div>
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
