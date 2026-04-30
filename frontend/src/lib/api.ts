//src/lib/api.ts
export const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const AUTH_STORAGE_KEY = "travelyst_auth_token";
const AUTH_LEGACY_STORAGE_KEYS = ["AUTH_TOKEN", "auth_token"];
const AUTH_TOKEN_STORAGE_KEYS = [AUTH_STORAGE_KEY, ...AUTH_LEGACY_STORAGE_KEYS];
const AUTH_TOKEN_FROM_ENV = String(import.meta.env.VITE_AUTH_TOKEN || "").trim();
export const BOOKING_AUTH_TOKEN_CHANGED_EVENT = "travelyst:booking-auth-token-changed";
type RequestAuthMode = "auto" | "omit";
type RequestOptions = {
  authMode?: RequestAuthMode;
  authToken?: string;
  cache?: RequestCache;
  timeoutMs?: number;
  signal?: AbortSignal;
};

function normalizeAuthToken(raw: string): string {
  const trimmed = String(raw || "").trim();
  if (!trimmed) return "";
  const unquoted =
    (trimmed.startsWith('"') && trimmed.endsWith('"')) || (trimmed.startsWith("'") && trimmed.endsWith("'"))
      ? trimmed.slice(1, -1).trim()
      : trimmed;
  const bearerPrefix = /^bearer\s+/i;
  return bearerPrefix.test(unquoted) ? unquoted.replace(bearerPrefix, "").trim() : unquoted;
}

let runtimeAuthToken = normalizeAuthToken(AUTH_TOKEN_FROM_ENV);

export function getConfiguredAuthToken(): string {
  if (typeof window !== "undefined" && window.localStorage) {
    for (const key of AUTH_TOKEN_STORAGE_KEYS) {
      const fromStorage = normalizeAuthToken(String(window.localStorage.getItem(key) || ""));
      if (!fromStorage) continue;
      if (key !== AUTH_STORAGE_KEY) {
        window.localStorage.setItem(AUTH_STORAGE_KEY, fromStorage);
        window.localStorage.removeItem(key);
      }
      runtimeAuthToken = fromStorage;
      return fromStorage;
    }
  }
  if (runtimeAuthToken) return runtimeAuthToken;
  runtimeAuthToken = normalizeAuthToken(AUTH_TOKEN_FROM_ENV);
  return runtimeAuthToken;
}

export function hasConfiguredAuthToken(): boolean {
  return getConfiguredAuthToken().length > 0;
}

export function setConfiguredAuthToken(rawToken: string): string {
  const normalized = normalizeAuthToken(rawToken);
  runtimeAuthToken = normalized;
  if (typeof window !== "undefined" && window.localStorage) {
    if (normalized) {
      window.localStorage.setItem(AUTH_STORAGE_KEY, normalized);
    } else {
      window.localStorage.removeItem(AUTH_STORAGE_KEY);
    }
    for (const legacyKey of AUTH_LEGACY_STORAGE_KEYS) {
      window.localStorage.removeItem(legacyKey);
    }
    window.dispatchEvent(new Event(BOOKING_AUTH_TOKEN_CHANGED_EVENT));
  }
  return normalized;
}

export function clearConfiguredAuthToken(): void {
  runtimeAuthToken = "";
  if (typeof window === "undefined" || !window.localStorage) return;
  for (const key of AUTH_TOKEN_STORAGE_KEYS) {
    window.localStorage.removeItem(key);
  }
  window.dispatchEvent(new Event(BOOKING_AUTH_TOKEN_CHANGED_EVENT));
}

function buildRequestHeaders(base: HeadersInit = {}, options?: RequestOptions): Headers {
  const headers = new Headers(base);
  const authMode = options?.authMode || "auto";
  if (authMode === "omit") return headers;
  const token = normalizeAuthToken(options?.authToken || getConfiguredAuthToken());
  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  return headers;
}

function extractErrorDetail(data: unknown, fallback: string): string {
  if (!data || typeof data !== "object") return fallback;
  const record = data as Record<string, unknown>;
  const detail = record.detail;
  if (typeof detail === "string" && detail.trim()) return detail;
  if (detail && typeof detail === "object") {
    const detailRecord = detail as Record<string, unknown>;
    if (typeof detailRecord.message === "string" && detailRecord.message.trim()) {
      return detailRecord.message;
    }
    if (typeof detailRecord.error === "string" && detailRecord.error.trim()) {
      return detailRecord.error;
    }
  }
  if (typeof record.message === "string" && record.message.trim()) return record.message;
  return fallback;
}

export function resolveApiUrl(url: string): string {
  const raw = (url || "").trim();
  if (!raw) return raw;
  try {
    return new URL(raw, API_BASE).toString();
  } catch {
    return raw;
  }
}

export async function postJson<T>(
  path: string,
  payload: unknown,
  options?: RequestOptions
): Promise<T> {
  const timeoutMs = typeof options?.timeoutMs === "number" ? Math.max(1, options.timeoutMs) : 0;
  const controller = timeoutMs > 0 ? new AbortController() : null;
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  const signal = controller?.signal || options?.signal;
  if (controller && timeoutMs > 0) {
    timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  }
  try {
    const resp = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: buildRequestHeaders({ "Content-Type": "application/json" }, options),
      body: JSON.stringify(payload),
      cache: options?.cache,
      signal,
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      const detail = extractErrorDetail(data, `HTTP Error ${resp.status}`);
      throw new Error(detail);
    }
    return data as T;
  } finally {
    if (timeoutId) clearTimeout(timeoutId);
  }
}

export async function getJson<T>(
  path: string,
  options?: RequestOptions
): Promise<T> {
  const timeoutMs = typeof options?.timeoutMs === "number" ? Math.max(1, options.timeoutMs) : 0;
  const controller = timeoutMs > 0 ? new AbortController() : null;
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  const signal = controller?.signal || options?.signal;
  if (controller && timeoutMs > 0) {
    timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  }
  try {
    const resp = await fetch(`${API_BASE}${path}`, {
      headers: buildRequestHeaders({}, options),
      cache: options?.cache,
      signal,
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      const detail = extractErrorDetail(data, `HTTP Error ${resp.status}`);
      throw new Error(detail);
    }
    return data as T;
  } finally {
    if (timeoutId) clearTimeout(timeoutId);
  }
}
