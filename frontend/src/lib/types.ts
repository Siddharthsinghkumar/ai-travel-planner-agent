//src/lib/types.ts
export interface BookingHandoffMeta {
  url?: string | null;
  source?: string;
  reason?: string;
  status?: string;
  booking_exit_quality?: string;
  provider?: string;
  blocked_domain?: string;
  diagnostics?: Record<string, unknown>;
  selected_flight_rank?: number;
  round_trip?: {
    return_search_outcome?: string;
    return_search_reason?: string;
    return_handoff_status?: string;
    is_outbound_only_handoff?: boolean;
  };
}

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
  itinerary_type?: string;
  travel_class?: string;
  legroom?: string;
  marketed_as?: string[];
  separate_tickets?: boolean;
  local_prices?: unknown;
  baggage_prices?: unknown;
  booking_sellers?: string[];
  booking_handoff?: BookingHandoffMeta;
  booking_token?: string;
  airline_logo?: string;
  price_unavailable?: boolean;
}

export interface TripDebugInfo {
  all_flights?: Flight[];
  top_flights?: Flight[];
  agent_reasoning?: unknown;
  reasoning?: unknown;
  intent?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface MultiCityLeg {
  llm_response?: string | null;
  best_flight?: Flight;
  weather?: Record<string, unknown> | null;
  warnings?: string[] | null;
  weather_present?: boolean;
  weather_reason?: string | null;
  debug_info?: TripDebugInfo | null;
  search_date?: string;
}

export interface TripPlan {
  best_flight?: Flight;
  all_flights?: Flight[];
  top_flights?: Flight[];
  weather?: Record<string, unknown> | null;
  return_trip?: MultiCityLeg | null;
  warnings?: string[] | null;
  warning?: string;
  fallback?: boolean;
  debug_info?: TripDebugInfo | null;
  llm_response?: string;
  error?: string;
  weather_present?: boolean;
  weather_reason?: string | null;
  multicity?: boolean;
  legs?: MultiCityLeg[];
  result_status?: "success" | "degraded" | "error";
  fallback_note?: string;
  degradation?: {
    reason?: string;
    message?: string;
    provider?: string;
    [key: string]: unknown;
  } | null;
  failure_reason?: string;
  failure_domain?: string;
  no_flights_reason?: string;
  flight_counts?: Record<string, number> | null;
}

export type AsyncJobStatus = "queued" | "running" | "done" | "error" | "cancelled";

export interface AsyncJobContract {
  durability?: string;
  queue?: string;
  contract?: string;
}

export interface AsyncJobState {
  job_id?: string;
  status?: AsyncJobStatus;
  result?: TripPlan | null;
  error?: string | null;
  message?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  completed_at?: string | null;
  cancel_requested?: boolean;
  contract?: AsyncJobContract;
}

export interface AsyncJobEvent {
  event: string;
  job_id?: string;
  status?: AsyncJobStatus | "closed";
  sequence?: number;
  timestamp?: string;
  message?: string;
  data?: unknown;
  result?: TripPlan | null;
  error?: string | null;
}

export interface BookingRecord {
  id: number;
  status: string;
  handoff_url?: string | null;
  checkout_ready?: boolean;
  checkout_status?: string;
  hold_outcome?: string;
  booking_handoff?: Record<string, unknown> | null;
  expires_at?: string | null;
  created_at?: string | null;
  flight?: Flight | Record<string, unknown>;
  booking_token?: string | null;
  shareable_link?: string | null;
}

export interface BookingActionResponse {
  action: string;
  success: boolean;
  hold_created?: boolean;
  checkout_ready?: boolean;
  hold_outcome?: string;
  message?: string;
  error?: string;
  booking?: BookingRecord | null;
  best_flight?: Flight | Record<string, unknown>;
  monitoring_active?: boolean;
  booking_id?: number;
}

export interface BookingResolveHandoffResponse {
  action: "resolve_booking_handoff";
  success: boolean;
  handoff_url?: string | null;
  booking_handoff?: BookingHandoffMeta | Record<string, unknown> | null;
  blocked_reason?: string | null;
  blocked_category?: string | null;
  retryable?: boolean | null;
  message?: string;
  auth_mode?: "authenticated_token" | "local_dev_unauthed" | string;
  auth_required?: boolean;
  owner_principal_id?: string | null;
  best_flight?: Flight | Record<string, unknown>;
}

export interface BookingResolveState {
  status: "idle" | "resolving" | "resolved" | "failed";
  message?: string;
  handoff_url?: string | null;
  blocked_reason?: string | null;
  blocked_category?: string | null;
  retryable?: boolean;
  updated_at?: number;
}

export interface BookingHandoffCapabilities {
  action: "booking_handoff_capabilities";
  resolve_available_now: boolean;
  auth_mode: "authenticated_token" | "local_dev_unauthed" | "auth_required" | string;
  resolve_auth_mode?: "auto" | "omit" | string;
  auth_required: boolean;
  has_valid_token: boolean;
  token_present?: boolean;
  auth_rejected?: boolean;
  auth_error?: string | null;
  blocked_reason?: string | null;
  local_dev_unauth_configured: boolean;
  local_dev_unauth_enabled?: boolean;
  loopback_request: boolean;
  loopback_eligible?: boolean;
  local_dev_unauth_available: boolean;
  message?: string;
}

export interface PriceAlert {
  alert_id: number;
  booking_id: number;
  origin: string;
  destination: string;
  travel_date: string;
  held_price_inr: number;
  new_price_inr: number;
  drop_pct: number;
  new_handoff_url?: string | null;
  created_at: string;
}

export interface PriceTrackingStatus {
  enabled: boolean;
  status?: Record<string, unknown>;
  contract?: AsyncJobContract;
}

export interface AskPayload {
  user_query?: string;
  origin?: string;
  destination?: string;
  date?: string;
  return_date?: string;
  trip_type?: string;
  direct_only?: boolean;
  cabin?: "any" | "economy" | "premium" | "business" | "first";
  baggage_pref?: "any" | "hand" | "checked";
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
