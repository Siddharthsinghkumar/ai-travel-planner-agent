// src/hooks/useStreamingPlan.tsx
import { useCallback, useRef, useState } from "react";
import { API_BASE, postJson } from "../lib/api";
import type { AskPayload, Flight, TripPlan } from "../lib/types";

type ResponseMeta = {
  result_status?: string;
  failure_reason?: string;
  failure_domain?: string;
  no_flights_reason?: string;
  fallback_note?: string;
  degradation_message?: string;
  flight_counts?: Record<string, number> | null;
};

function getErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === "string" && error.trim().length > 0) return error;
  return fallback;
}

function extractResponseErrorMessage(payload: unknown, fallback: string): string {
  if (payload && typeof payload === "object") {
    const data = payload as Record<string, unknown>;
    if (typeof data.detail === "string" && data.detail.trim()) return data.detail.trim();
    if (typeof data.error === "string" && data.error.trim()) return data.error.trim();
    if (typeof data.warning === "string" && data.warning.trim()) return data.warning.trim();
  }
  return fallback;
}

function extractResponseMeta(payload: unknown): ResponseMeta | null {
  if (!payload || typeof payload !== "object") return null;
  const data = payload as Record<string, unknown>;
  const degradation =
    data.degradation && typeof data.degradation === "object"
      ? (data.degradation as Record<string, unknown>)
      : null;
  const flightCounts =
    data.flight_counts && typeof data.flight_counts === "object"
      ? (data.flight_counts as Record<string, number>)
      : null;

  const meta: ResponseMeta = {
    result_status: typeof data.result_status === "string" ? data.result_status : undefined,
    failure_reason: typeof data.failure_reason === "string" ? data.failure_reason : undefined,
    failure_domain: typeof data.failure_domain === "string" ? data.failure_domain : undefined,
    no_flights_reason: typeof data.no_flights_reason === "string" ? data.no_flights_reason : undefined,
    fallback_note: typeof data.fallback_note === "string" ? data.fallback_note : undefined,
    degradation_message:
      typeof degradation?.message === "string"
        ? degradation.message
        : undefined,
    flight_counts: flightCounts,
  };

  if (
    !meta.result_status &&
    !meta.failure_reason &&
    !meta.fallback_note &&
    !meta.degradation_message &&
    !meta.no_flights_reason
  ) {
    return null;
  }
  return meta;
}

function formatStructuredError(payload: unknown, fallback: string): string {
  const base = extractResponseErrorMessage(payload, fallback);
  if (!payload || typeof payload !== "object") return base;
  const data = payload as Record<string, unknown>;
  const reason = typeof data.failure_reason === "string" ? data.failure_reason : "";
  const noFlightsReason = typeof data.no_flights_reason === "string" ? data.no_flights_reason : "";

  if (reason === "no_flights") {
    if (noFlightsReason === "filters_too_strict") {
      return "No matching flights found for current constraints. Try relaxing filters or changing date/route.";
    }
    return "No matching flights found for this route/date. Try a nearby date or a different route.";
  }
  return base;
}

export function useStreamingPlan() {
  const [tokens, setTokens] = useState("");
  const [finalJson, setFinalJson] = useState<TripPlan | null>(null);
  const [partialFlights, setPartialFlights] = useState<Flight[] | null>(null);
  const [partialTopFlights, setPartialTopFlights] = useState<Flight[] | null>(null);
  const [partialBestFlight, setPartialBestFlight] = useState<Flight | null>(null);
  const [partialWeather, setPartialWeather] = useState<Record<string, unknown> | null>(null);
  const [reasoningSteps, setReasoningSteps] = useState<string[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isFallback, setIsFallback] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [responseMeta, setResponseMeta] = useState<ResponseMeta | null>(null);
  const [rawStream, setRawStream] = useState("");
  const [approvalRequired, setApprovalRequired] = useState<{
    planId: string;
    action: string;
    message: string;
  } | null>(null);
  const [approvalResult, setApprovalResult] = useState<{
    approved: boolean;
    planId: string;
  } | null>(null);

  const controllerRef = useRef<AbortController | null>(null);
  const bufferRef = useRef("");
  const requestIdRef = useRef(0);
  const cancelledRequestIdRef = useRef<number | null>(null);
  const fallbackRequestIdRef = useRef<number | null>(null);

  const clearVisibleResultState = useCallback(() => {
    setTokens("");
    setFinalJson(null);
    setPartialFlights(null);
    setPartialTopFlights(null);
    setPartialBestFlight(null);
    setPartialWeather(null);
    setReasoningSteps([]);
    setRawStream("");
    setResponseMeta(null);
    setError(null);
    setIsFallback(false);
  }, []);

  const isRecoverableStreamError = useCallback((message: string) => {
    const m = message.toLowerCase();
    return (
      m.includes("temporarily unavailable") ||
      m.includes("streaming timed out") ||
      m.includes("stream initialization timed out") ||
      m.includes("interrupted before completion") ||
      m.includes("streaming unavailable") ||
      m.includes("circuit breaker open") ||
      m.includes("llm temporarily unavailable")
    );
  }, []);

  const cancel = useCallback(() => {
    if (controllerRef.current) {
      cancelledRequestIdRef.current = requestIdRef.current;
      controllerRef.current.abort();
    }
    clearVisibleResultState();
    setError("Request stopped before completion.");
    setIsFallback(false);
    setIsStreaming(false);
  }, [clearVisibleResultState]);

  const reset = useCallback(() => {
    if (controllerRef.current) {
      controllerRef.current.abort();
    }
    clearVisibleResultState();
    setError(null);
    setIsFallback(false);
    setIsStreaming(false);
  }, [clearVisibleResultState]);

  const runFallbackRequest = useCallback(async (requestId: number, payload: AskPayload) => {
    if (requestIdRef.current !== requestId) return;
    try {
      const resp = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (requestIdRef.current !== requestId) return;

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        setResponseMeta(extractResponseMeta(errData));
        throw new Error(formatStructuredError(errData, `HTTP Error ${resp.status}`));
      }

      const j = await resp.json();
      if (requestIdRef.current !== requestId) return;
      if (typeof j?.error === "string" && j.error.trim().length > 0) {
        throw new Error(j.error.trim());
      }
      if (typeof j?.warning === "string" && j.warning.trim().length > 0 && Boolean(j?.fallback)) {
        throw new Error(j.warning.trim());
      }
      const fallbackText =
        typeof j?.llm_response === "string"
          ? j.llm_response
          : typeof j?.message === "string"
            ? j.message
            : typeof j?.error === "string"
              ? j.error
              : "";
      setTokens(fallbackText.trim().length > 0 ? fallbackText : "");
      setFinalJson(j);
      setResponseMeta(extractResponseMeta(j));
      setIsFallback(false);
      setIsStreaming(false);
      console.info("[Analytics] Fallback completed successfully.");
    } catch (e: unknown) {
      if (requestIdRef.current !== requestId) return;
      setError(getErrorMessage(e, "We couldn't complete your plan. Please try again."));
      setIsFallback(false);
      setIsStreaming(false);
    }
  }, []);

  const triggerFallback = useCallback(async (requestId: number, payload: AskPayload) => {
    if (requestIdRef.current !== requestId) return;
    if (fallbackRequestIdRef.current === requestId) return;
    fallbackRequestIdRef.current = requestId;

    setIsFallback(true);
    await runFallbackRequest(requestId, payload);
  }, [runFallbackRequest]);

  const start = useCallback(
    async (payload: AskPayload) => {
      const requestId = ++requestIdRef.current;

      // Abort only the currently active request when a new one starts.
      if (controllerRef.current) {
        controllerRef.current.abort();
      }

      const controller = new AbortController();
      controllerRef.current = controller;

      cancelledRequestIdRef.current = null;
      fallbackRequestIdRef.current = null;

      const isActiveRequest = () => requestIdRef.current === requestId;

      clearVisibleResultState();
      setError(null);
      setIsFallback(false);
      setIsStreaming(true);
      bufferRef.current = "";

      const STREAM_SOFT_DELAY_MS = Number(import.meta.env.VITE_STREAM_SOFT_DELAY_MS) || 5000;
      const STREAM_HARD_NO_ACTIVITY_MS =
        Number(import.meta.env.VITE_STREAM_HARD_NO_ACTIVITY_MS) || Math.max(STREAM_SOFT_DELAY_MS * 3, 15000);

      let sawStreamActivity = false;
      let accumulatedTokens = "";
      let lastUpdate = Date.now();
      let softDelayTimer: ReturnType<typeof setTimeout> | null = null;
      let hardNoActivityTimer: ReturnType<typeof setTimeout> | null = null;
      let streamErrorMessage: string | null = null;
      let receivedFinalJson = false;
      let hasHydratedResultData = false;

      try {
        const response = await fetch(`${API_BASE}/ask?stream=true`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
          },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          const errMsg = extractResponseErrorMessage(errData, `HTTP Error ${response.status}`);
          throw new Error(errMsg);
        }

        if (!response.body) throw new Error("No stream body");

        console.info("[Analytics] Stream started for payload:", payload);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        const marker = "[DONE_JSON]";
        const frameSep = "\n\n";

        const parseFrame = (frame: string) => {
          const lines = frame.split("\n");
          let eventType: string | null = null;
          const dataLines: string[] = [];
          for (const line of lines) {
            if (line.startsWith("event:")) {
              eventType = line.slice(6).trim() || null;
              continue;
            }
            if (line.startsWith("data:")) {
              dataLines.push(line.startsWith("data: ") ? line.slice(6) : line.slice(5));
            }
          }
          return {
            eventType,
            data: dataLines.join("\n"),
          };
        };

        const appendReasoningStep = (step: string) => {
          if (!isActiveRequest()) return;
          const trimmed = step.trim();
          if (!trimmed) return;
          setReasoningSteps((prev) => {
            const exists = prev.some((item) => item.trim().toLowerCase() === trimmed.toLowerCase());
            if (exists) return prev;
            return [...prev, trimmed].slice(0, 8);
          });
        };

        const processStructuredEvent = (eventType: string, eventData: string) => {
          if (!isActiveRequest()) return;
          if (!eventData.trim()) return;

          if (eventType === "reasoning_step") {
            try {
              const parsed = JSON.parse(eventData);
              if (typeof parsed?.step === "string") {
                appendReasoningStep(parsed.step);
                return;
              }
            } catch {
              // Fall through to plain text handling.
            }
            appendReasoningStep(eventData);
            return;
          }

          if (eventType === "flights") {
            try {
              const parsed = JSON.parse(eventData) as Record<string, unknown>;
              if (Array.isArray(parsed?.top_flights)) {
                setPartialTopFlights(parsed.top_flights as Flight[]);
                if (parsed.top_flights.length > 0) {
                  hasHydratedResultData = true;
                }
              }
              if (Array.isArray(parsed?.all_flights)) {
                setPartialFlights(parsed.all_flights as Flight[]);
                if (parsed.all_flights.length > 0) {
                  hasHydratedResultData = true;
                }
              }
              if (parsed?.best_flight && typeof parsed.best_flight === "object") {
                setPartialBestFlight(parsed.best_flight as Flight);
                hasHydratedResultData = true;
              }
            } catch {
              // Ignore malformed structured frame and continue streaming.
            }
            return;
          }

          if (eventType === "weather") {
            try {
              const parsed = JSON.parse(eventData) as Record<string, unknown>;
              const payload =
                parsed?.weather && typeof parsed.weather === "object"
                  ? (parsed.weather as Record<string, unknown>)
                  : parsed;
              if (payload && typeof payload === "object") {
                setPartialWeather(payload);
                if (Object.keys(payload).length > 0) {
                  hasHydratedResultData = true;
                }
              }
            } catch {
              // Ignore malformed structured frame and continue streaming.
            }
          }

          if (eventType === "approval_required") {
            try {
              const parsed = JSON.parse(eventData) as Record<string, unknown>;
              if (typeof parsed?.plan_id === "string") {
                setApprovalRequired({
                  planId: parsed.plan_id as string,
                  action: typeof parsed.action === "string" ? parsed.action : "booking_handoff",
                  message: typeof parsed.message === "string" ? parsed.message : "Approval required before proceeding.",
                });
                setApprovalResult(null);
              }
            } catch {
              // Ignore malformed approval event.
            }
          }

          if (eventType === "approval_result") {
            try {
              const parsed = JSON.parse(eventData) as Record<string, unknown>;
              if (typeof parsed?.plan_id === "string" && typeof parsed.approved === "boolean") {
                setApprovalResult({
                  approved: parsed.approved as boolean,
                  planId: parsed.plan_id as string,
                });
                setApprovalRequired(null);
              }
            } catch {
              // Ignore malformed approval result.
            }
          }
        };

        const processData = async (eventType: string | null, data: string) => {
          if (!isActiveRequest()) return true;

          if (eventType && eventType !== "done") {
            sawStreamActivity = true;
            processStructuredEvent(eventType, data);
            return false;
          }

          if (!data) return false;
          sawStreamActivity = true;

          if (data.startsWith("[ERROR]")) {
            streamErrorMessage = data.replace("[ERROR]", "").trim() || "Streaming request failed.";

            if (isRecoverableStreamError(streamErrorMessage)) {
              if (hasHydratedResultData) {
                setError(
                  "Explanation generation timed out, but available flight/weather results are shown below."
                );
                setIsFallback(false);
                setIsStreaming(false);
                return true;
              }
              await triggerFallback(requestId, payload);
              return true;
            }

            if (!hasHydratedResultData) {
              clearVisibleResultState();
            }
            setError(streamErrorMessage || "We hit a connection issue. Please try again.");
            setIsStreaming(false);
            return true;
          }

          if (data.includes(marker)) {
            const [before, after] = data.split(marker);

            if (before) {
              accumulatedTokens += before;
              setTokens(accumulatedTokens);
            }

            if (!after.trim()) return false;

            try {
              const parsed = JSON.parse(after.trim());
              if (typeof parsed?.error === "string") {
                setResponseMeta(extractResponseMeta(parsed));
                const doneJsonError = parsed.error.trim();
                if (isRecoverableStreamError(doneJsonError)) {
                  if (hasHydratedResultData) {
                    setError(
                      "Explanation generation timed out, but available flight/weather results are shown below."
                    );
                    setIsFallback(false);
                    setIsStreaming(false);
                    return true;
                  }
                  await triggerFallback(requestId, payload);
                  return true;
                }
                if (!hasHydratedResultData) {
                  clearVisibleResultState();
                }
                setError(doneJsonError || "We couldn't complete your plan. Please try again.");
                setFinalJson(null);
                setIsStreaming(false);
                return true;
              }

              receivedFinalJson = true;
              setFinalJson(parsed);
              setResponseMeta(extractResponseMeta(parsed));
              setIsStreaming(false);
              return true;
            } catch {
              if (!hasHydratedResultData) {
                clearVisibleResultState();
              }
              setError("We couldn't complete your plan. Please try again.");
              setIsStreaming(false);
              return true;
            }
          }

          accumulatedTokens += data;

          if (Date.now() - lastUpdate > 50) {
            setTokens(accumulatedTokens);
            lastUpdate = Date.now();
          }

          return false;
        };

        softDelayTimer = setTimeout(() => {
          if (!sawStreamActivity && !receivedFinalJson) {
            // Soft-delay stage intentionally records internal elapsed time only.
          }
        }, STREAM_SOFT_DELAY_MS);

        hardNoActivityTimer = setTimeout(() => {
          if (!isActiveRequest()) return;
          if (!sawStreamActivity && !receivedFinalJson) {
            fallbackRequestIdRef.current = requestId;
            setIsFallback(true);
            controller.abort();
          }
        }, STREAM_HARD_NO_ACTIVITY_MS);

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          if (!isActiveRequest()) return;

          const chunk = decoder.decode(value, { stream: true });
          if (chunk.length > 0) {
            sawStreamActivity = true;
          }
          setRawStream((prev) => prev + chunk);

          bufferRef.current += chunk;

          let idx;
          while ((idx = bufferRef.current.indexOf(frameSep)) !== -1) {
            const frame = bufferRef.current.slice(0, idx);
            bufferRef.current = bufferRef.current.slice(idx + frameSep.length);
            const parsedFrame = parseFrame(frame);
            const shouldExit = await processData(parsedFrame.eventType, parsedFrame.data);
            if (shouldExit) return;
          }
        }

        if (bufferRef.current.trim().length > 0) {
          const trailingFrame = parseFrame(bufferRef.current);
          bufferRef.current = "";
          const shouldExit = await processData(trailingFrame.eventType, trailingFrame.data);
          if (shouldExit) return;
        }

        if (accumulatedTokens.trim().length > 0) {
          if (!isActiveRequest()) return;
          setTokens(accumulatedTokens);
        }

        if (!receivedFinalJson) {
          await triggerFallback(requestId, payload);
          return;
        }

        if (streamErrorMessage && isRecoverableStreamError(streamErrorMessage)) {
          await triggerFallback(requestId, payload);
          return;
        }
      } catch (err: unknown) {
        const isAbort = err instanceof DOMException && err.name === "AbortError";

        if (isAbort) {
          if (cancelledRequestIdRef.current === requestId) {
            console.info("Stream aborted intentionally");
            return;
          }

          if (fallbackRequestIdRef.current === requestId) {
            console.warn("[Analytics] Fallback triggered due to stream init timeout");
            await runFallbackRequest(requestId, payload);
            return;
          }

          // Stale request abort (e.g., replaced by a newer request) — ignore silently.
          return;
        }

        if (isRecoverableStreamError(getErrorMessage(err, ""))) {
          if (hasHydratedResultData) {
            setError("Explanation generation timed out, but available flight/weather results are shown below.");
            setIsFallback(false);
            setIsStreaming(false);
            return;
          }
          await triggerFallback(requestId, payload);
          return;
        }

        if (!isActiveRequest()) return;
        if (!hasHydratedResultData) {
          clearVisibleResultState();
        }
        setError(getErrorMessage(err, "We couldn't complete your plan. Please try again."));
      } finally {
        if (softDelayTimer) clearTimeout(softDelayTimer);
        if (hardNoActivityTimer) clearTimeout(hardNoActivityTimer);
        if (controllerRef.current === controller) {
          controllerRef.current = null;
        }
        if (isActiveRequest()) {
          setIsStreaming(false);
        }
      }
    },
    [clearVisibleResultState, isRecoverableStreamError, runFallbackRequest, triggerFallback]
  );

  const respondToApproval = useCallback(async (approved: boolean) => {
    if (!approvalRequired) return;
    try {
      await postJson(`/plan/${approvalRequired.planId}/approve`, { approved });
    } catch (err) {
      console.error("Approval request failed:", err);
    }
  }, [approvalRequired]);

  return {
    tokens,
    finalJson,
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
  };
}
