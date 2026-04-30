import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE, getJson, postJson } from "../lib/api";
import type { AskPayload, AsyncJobEvent, AsyncJobState, AsyncJobStatus } from "../lib/types";

type AsyncJobHook = {
  job: AsyncJobState | null;
  status: AsyncJobStatus | "idle";
  error: string | null;
  startJob: (payload: AskPayload) => Promise<void>;
  cancelJob: () => Promise<void>;
  refreshJob: () => Promise<void>;
  clearJob: () => void;
};

const POLL_INTERVAL_MS = 2000;

export function useAsyncJob(): AsyncJobHook {
  const [job, setJob] = useState<AsyncJobState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);
  const eventsRef = useRef<EventSource | null>(null);
  const activeJobIdRef = useRef<string | null>(null);

  const status: AsyncJobStatus | "idle" = (job?.status as AsyncJobStatus) || "idle";

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const stopEvents = useCallback(() => {
    if (eventsRef.current) {
      eventsRef.current.close();
      eventsRef.current = null;
    }
  }, []);

  const applyEvent = useCallback((evt: AsyncJobEvent) => {
    setJob((prev) => {
      const status = typeof evt.status === "string" ? evt.status : prev?.status;
      const normalizedStatus =
        status === "closed" ? (prev?.status || undefined) : (status as AsyncJobStatus | undefined);
      const next: AsyncJobState = {
        ...(prev || {}),
        job_id: evt.job_id || prev?.job_id || activeJobIdRef.current || undefined,
        status: normalizedStatus,
      };
      if (typeof evt.message === "string") next.message = evt.message;
      if (typeof evt.error === "string") next.error = evt.error;
      if (evt.result && typeof evt.result === "object") next.result = evt.result;
      if (!next.result && evt.data && typeof evt.data === "object" && evt.event === "done") {
        next.result = evt.data as AsyncJobState["result"];
      }
      return next;
    });

    if (evt.event === "error") {
      setError(typeof evt.error === "string" ? evt.error : typeof evt.message === "string" ? evt.message : "Async job failed.");
    }
    if (["done", "error", "cancelled", "closed"].includes(String(evt.event)) || ["done", "error", "cancelled"].includes(String(evt.status))) {
      stopPolling();
      stopEvents();
    }
  }, [stopEvents, stopPolling]);

  const refreshJob = useCallback(async () => {
    const jobId = activeJobIdRef.current;
      if (!jobId) return;
      try {
        const data = await getJson<AsyncJobState>(`/jobs/${jobId}`);
        setJob(data);
        if (["done", "error", "cancelled"].includes(String(data?.status))) {
          stopPolling();
          stopEvents();
        }
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Failed to refresh job status");
        stopPolling();
        stopEvents();
      }
  }, [stopEvents, stopPolling]);

  const startEventStream = useCallback((jobId: string) => {
    if (typeof window === "undefined" || typeof EventSource === "undefined") return;
    stopEvents();
    const source = new EventSource(`${API_BASE}/jobs/${jobId}/events`);
    eventsRef.current = source;

    const handle = (event: MessageEvent) => {
      try {
        const parsed = JSON.parse(event.data) as AsyncJobEvent;
        applyEvent(parsed);
      } catch {
        // Keep polling fallback as source of truth when parsing fails.
      }
    };

    const eventNames = ["queued", "running", "token", "reasoning_step", "flights", "weather", "done", "error", "cancelled", "closed"];
    eventNames.forEach((name) => source.addEventListener(name, handle as EventListener));
    source.onmessage = handle;
    source.onerror = () => {
      // Keep polling fallback; close noisy stream on terminal transport failures.
      stopEvents();
    };
  }, [applyEvent, stopEvents]);

  const startPolling = useCallback((_jobId: string) => {
    stopPolling();
    pollRef.current = window.setInterval(() => {
      refreshJob();
    }, POLL_INTERVAL_MS);
  }, [refreshJob, stopPolling]);

  const startJob = useCallback(async (payload: AskPayload) => {
    setError(null);
    setJob({ status: "queued", job_id: undefined });
    try {
      const data = await postJson<{ job_id: string }>("/ask?async_job=true", payload);
      activeJobIdRef.current = data.job_id;
      setJob({ job_id: data.job_id, status: "queued" });
      startPolling(data.job_id);
      startEventStream(data.job_id);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to start async job");
      setJob(null);
      stopPolling();
      stopEvents();
    }
  }, [startEventStream, startPolling, stopEvents, stopPolling]);

  const cancelJob = useCallback(async () => {
    const jobId = activeJobIdRef.current;
    if (!jobId) return;
    try {
      const data = await postJson<{ job: AsyncJobState }>(`/jobs/${jobId}/cancel`, {});
      if (data?.job) {
        setJob(data.job);
      }
      stopPolling();
      stopEvents();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to cancel job");
    }
  }, [stopEvents, stopPolling]);

  const clearJob = useCallback(() => {
    activeJobIdRef.current = null;
    setJob(null);
    setError(null);
    stopPolling();
    stopEvents();
  }, [stopEvents, stopPolling]);

  useEffect(() => () => {
    stopPolling();
    stopEvents();
  }, [stopEvents, stopPolling]);

  return {
    job,
    status,
    error,
    startJob,
    cancelJob,
    refreshJob,
    clearJob,
  };
}
