//src/lib/types.ts
export interface Flight {
  airline: string;
  flight_no: string;
  departure_time: string;
  arrival_time: string;
  price_inr: string | number;
  duration_min: number;
  stops: number | string;
  baggage: string;
  layover_info?: string;
  carbon_emissions_g?: number;
  date?: string;
  handoff_url?: string;
}

export interface TripDebugInfo {
  all_flights?: Flight[];
  agent_reasoning?: unknown;
  reasoning?: unknown;
  intent?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface TripPlan {
  best_flight?: Flight;
  all_flights?: Flight[];
  weather?: Record<string, unknown> | null;
  debug_info?: TripDebugInfo | null;
  llm_response?: string;
  error?: string;
}

export interface AskPayload {
  user_query?: string;
  origin?: string;
  destination?: string;
  date?: string;
  trip_type?: string;
  llm_mode?: LLMMode;
  cloud_provider?: string;
}

export interface ServerVersionMeta {
  git_commit?: string;
  file_mtime?: number;
}

export type CapabilityStatus =
  | "live"
  | "partial"
  | "coming-soon";

export interface FeatureCapability {
  id: string;
  title: string;
  description: string;
  status: CapabilityStatus;
  note?: string;
}

export type LLMMode = "ollama_only" | "cloud_only" | "cloud_first" | "ollama_first";

export interface LLMOptionsResponse {
  llm_modes: LLMMode[];
  cloud_providers: string[];
  defaults: {
    llm_mode: LLMMode;
    cloud_provider?: string;
  };
  provider_status?: Record<
    string,
    {
      configured: boolean;
      initialized: boolean;
      usable: boolean;
    }
  >;
  usable_cloud_providers?: string[];
  cloud_usable?: boolean;
  cloud_enabled_by_config?: boolean;
  provider_switch_enabled?: boolean;
  effective_default_provider?: string;
  effective_mode?: LLMMode;
  backend_availability?: {
    cloud: boolean;
    ollama: boolean;
  };
}
