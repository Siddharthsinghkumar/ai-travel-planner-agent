//src/lib/api.ts
export const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export function resolveApiUrl(url: string): string {
  const raw = (url || "").trim();
  if (!raw) return raw;
  try {
    return new URL(raw, API_BASE).toString();
  } catch {
    return raw;
  }
}
