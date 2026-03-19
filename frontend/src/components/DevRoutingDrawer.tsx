import type { LLMMode, LLMOptionsResponse } from "../lib/types";

type DevRoutingDrawerProps = {
  isOpen: boolean;
  llmOptions: LLMOptionsResponse | null;
  llmMode: LLMMode;
  cloudProvider: string;
  onModeChange: (mode: LLMMode) => void;
  onProviderChange: (provider: string) => void;
  onClose: () => void;
};

const MODE_LABELS: Record<LLMMode, string> = {
  ollama_only: "Ollama only",
  cloud_only: "Cloud only",
  cloud_first: "Cloud first, local backup",
  ollama_first: "Local first, cloud backup",
};

function modeUsesCloud(mode: LLMMode): boolean {
  return mode === "cloud_only" || mode === "cloud_first" || mode === "ollama_first";
}

export default function DevRoutingDrawer({
  isOpen,
  llmOptions,
  llmMode,
  cloudProvider,
  onModeChange,
  onProviderChange,
  onClose,
}: DevRoutingDrawerProps) {
  if (!isOpen) return null;

  const modes = llmOptions?.llm_modes?.length
    ? llmOptions.llm_modes
    : (["ollama_only", "cloud_only", "cloud_first", "ollama_first"] as LLMMode[]);
  const providers = llmOptions?.usable_cloud_providers?.length
    ? llmOptions.usable_cloud_providers
    : llmOptions?.cloud_providers?.length
      ? llmOptions.cloud_providers
      : ["gemini"];
  const cloudEnabled = llmOptions?.cloud_enabled_by_config ?? true;
  const cloudUsable = llmOptions?.cloud_usable ?? (cloudEnabled && providers.length > 0);
  const providerDisabled = !modeUsesCloud(llmMode) || !cloudEnabled || !cloudUsable;

  return (
    <aside className="dev-drawer" aria-label="Developer routing controls">
      <div className="dev-drawer__head">
        <h3 className="dev-drawer__title">Developer routing</h3>
        <button type="button" className="dev-drawer__close" onClick={onClose} aria-label="Close developer controls">
          ×
        </button>
      </div>
      <p className="dev-drawer__hint">Hidden dev mode controls. Not visible in the consumer journey.</p>
      <label className="dev-drawer__field">
        <span>LLM mode</span>
        <select value={llmMode} onChange={(e) => onModeChange(e.target.value as LLMMode)} className="dev-drawer__select">
          {modes.map((mode) => (
            <option key={mode} value={mode}>
              {MODE_LABELS[mode] || mode}
            </option>
          ))}
        </select>
      </label>
      <label className="dev-drawer__field">
        <span>Cloud provider</span>
        <select
          value={cloudProvider}
          onChange={(e) => onProviderChange(e.target.value)}
          disabled={providerDisabled}
          className="dev-drawer__select"
        >
          {providers.map((provider) => (
            <option key={provider} value={provider}>
              {provider}
            </option>
          ))}
        </select>
      </label>
    </aside>
  );
}
