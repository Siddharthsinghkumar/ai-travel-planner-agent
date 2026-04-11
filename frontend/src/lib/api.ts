//src/lib/api.ts
export const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

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

export async function postJson<T>(path: string, payload: unknown): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const detail = extractErrorDetail(data, `HTTP Error ${resp.status}`);
    throw new Error(detail);
  }
  return data as T;
}

export async function getJson<T>(path: string): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const detail = extractErrorDetail(data, `HTTP Error ${resp.status}`);
    throw new Error(detail);
  }
  return data as T;
}
