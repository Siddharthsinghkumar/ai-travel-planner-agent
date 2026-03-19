function isLocalHost(hostname: string) {
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "0.0.0.0";
}

function detectPreviewMode() {
  const mode = import.meta.env.VITE_UI_MODE;
  if (mode === "preview") return true;
  if (mode === "dev") return false;

  if (import.meta.env.DEV) return false;
  if (typeof window === "undefined") return true;

  return !isLocalHost(window.location.hostname);
}

export const IS_PREVIEW_UI = detectPreviewMode();
