// src/hooks/useStreamingPlan.tsx
import { useCallback, useRef, useState } from "react";
import { API_BASE } from "../lib/api";
import type { AskPayload, Flight, TripPlan } from "../lib/types";

function getErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === "string" && error.trim().length > 0) return error;
  return fallback;
}

export function useStreamingPlan() {
  const [tokens, setTokens] = useState("");
  const [finalJson, setFinalJson] = useState<TripPlan | null>(null);
  const [partialFlights, setPartialFlights] = useState<Flight[] | null>(null);
  const [partialBestFlight, setPartialBestFlight] = useState<Flight | null>(null);
  const [partialWeather, setPartialWeather] = useState<Record<string, unknown> | null>(null);
  const [reasoningSteps, setReasoningSteps] = useState<string[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isFallback, setIsFallback] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rawStream, setRawStream] = useState("");

  const controllerRef = useRef<AbortController | null>(null);
  const bufferRef = useRef("");
  const requestIdRef = useRef(0);
  const cancelledRequestIdRef = useRef<number | null>(null);
  const fallbackRequestIdRef = useRef<number | null>(null);

  const isRecoverableStreamError = useCallback((message: string) => {
    const m = message.toLowerCase();
    return (
      m.includes("temporarily unavailable") ||
      m.includes("streaming timed out") ||
      m.includes("stream initialization timed out") ||
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
    setIsStreaming(false);
  }, []);

  const runFallbackRequest = useCallback(async (payload: AskPayload) => {
    try {
      const resp = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP Error ${resp.status}`);
      }

      const j = await resp.json();
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
      setIsFallback(false);
      setIsStreaming(false);
      console.info("[Analytics] Fallback completed successfully.");
    } catch (e: unknown) {
      setError("Connection issue — please try again.");
      setIsFallback(false);
      setIsStreaming(false);
    }
  }, []);

  const triggerFallback = useCallback(async (requestId: number, payload: AskPayload) => {
    if (fallbackRequestIdRef.current === requestId) return;
    fallbackRequestIdRef.current = requestId;

    setTokens("");
    setFinalJson(null);
    setPartialFlights(null);
    setPartialBestFlight(null);
    setPartialWeather(null);
    setReasoningSteps([]);
    setIsFallback(true);
    await runFallbackRequest(payload);
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

      setTokens("");
      setFinalJson(null);
      setPartialFlights(null);
      setPartialBestFlight(null);
      setPartialWeather(null);
      setReasoningSteps([]);
      setError(null);
      setIsFallback(false);
      setRawStream("");
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
          let errMsg = `HTTP Error ${response.status}`;
          if (errData.detail) {
            errMsg = typeof errData.detail === "string" ? errData.detail : JSON.stringify(errData.detail);
          }
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
          const trimmed = step.trim();
          if (!trimmed) return;
          setReasoningSteps((prev) => {
            const exists = prev.some((item) => item.trim().toLowerCase() === trimmed.toLowerCase());
            if (exists) return prev;
            return [...prev, trimmed].slice(0, 8);
          });
        };

        const processStructuredEvent = (eventType: string, eventData: string) => {
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
              if (Array.isArray(parsed?.all_flights)) {
                setPartialFlights(parsed.all_flights as Flight[]);
              }
              if (parsed?.best_flight && typeof parsed.best_flight === "object") {
                setPartialBestFlight(parsed.best_flight as Flight);
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
              }
            } catch {
              // Ignore malformed structured frame and continue streaming.
            }
          }
        };

        const processData = async (eventType: string | null, data: string) => {
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
              await triggerFallback(requestId, payload);
              return true;
            }

            setError("We hit a connection issue. Please try again.");
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
              if (typeof parsed?.error === "string" && isRecoverableStreamError(parsed.error)) {
                await triggerFallback(requestId, payload);
                return true;
              }

              receivedFinalJson = true;
              setFinalJson(parsed);
              setIsStreaming(false);
              return true;
            } catch {
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
          if (!sawStreamActivity && !receivedFinalJson) {
            fallbackRequestIdRef.current = requestId;
            setIsFallback(true);
            controller.abort();
          }
        }, STREAM_HARD_NO_ACTIVITY_MS);

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

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
            await runFallbackRequest(payload);
            return;
          }

          // Stale request abort (e.g., replaced by a newer request) — ignore silently.
          return;
        }

        if (isRecoverableStreamError(getErrorMessage(err, ""))) {
          await triggerFallback(requestId, payload);
          return;
        }

        setError("We couldn't complete your plan. Please try again.");
      } finally {
        if (softDelayTimer) clearTimeout(softDelayTimer);
        if (hardNoActivityTimer) clearTimeout(hardNoActivityTimer);
        if (controllerRef.current === controller) {
          controllerRef.current = null;
        }
        setIsStreaming(false);
      }
    },
    [isRecoverableStreamError, runFallbackRequest, triggerFallback]
  );

  return {
    tokens,
    finalJson,
    partialFlights,
    partialBestFlight,
    partialWeather,
    reasoningSteps,
    isStreaming,
    isFallback,
    error,
    rawStream,
    start,
    cancel,
  };
}
