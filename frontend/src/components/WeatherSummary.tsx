import { formatTemperatureC, formatWeatherDate } from "../lib/format";

type Props = {
  weather?: Record<string, unknown> | null;
  destinationCode?: string;
  destinationLabel?: string;
  weatherPresent?: boolean;
  weatherReason?: string;
  isLoading?: boolean;
};

function textOrEmpty(value: unknown): string {
  if (typeof value !== "string") return "";
  return value.trim();
}

function normalizeIata(code?: string): string {
  if (!code) return "";
  const normalized = code.trim().toUpperCase();
  return /^[A-Z]{3}$/.test(normalized) ? normalized : "";
}

function formatDestinationLabel(
  weather: Record<string, unknown> | null | undefined,
  code?: string,
  explicitLabel?: string
): string {
  const directLabel = textOrEmpty(explicitLabel);
  if (directLabel) return directLabel;

  const weatherLabel = textOrEmpty(weather?.location_label);
  if (weatherLabel) return weatherLabel;

  const weatherCity = textOrEmpty(weather?.location_city);
  const weatherCode = normalizeIata(textOrEmpty(weather?.location));
  if (weatherCity && weatherCode) return `${weatherCity} (${weatherCode})`;

  const normalized = normalizeIata(code);
  if (weatherCity && normalized) return `${weatherCity} (${normalized})`;
  if (normalized) return normalized;

  return "Destination";
}

function formatValue(value: unknown) {
  if (value === null || value === undefined || value === "") return "N/A";
  return String(value);
}

function weatherAvailabilityHint(reason?: string): string {
  const normalized = (reason || "").trim().toLowerCase();
  if (normalized === "forecast_horizon_exceeded") {
    return "Forecast data is unavailable for this date range yet.";
  }
  if (normalized === "api_failure") {
    return "Weather provider data is temporarily unavailable.";
  }
  return "Finalizing forecast details for this destination.";
}

export default function WeatherSummary({
  weather,
  destinationCode,
  destinationLabel,
  weatherPresent,
  weatherReason,
  isLoading = false,
}: Props) {
  const resolvedDestinationLabel = formatDestinationLabel(weather, destinationCode, destinationLabel);

  if (!weather || typeof weather !== "object") {
    return (
      <section
        className="weather-summary weather-summary--loading"
        aria-label={isLoading ? "Loading weather" : "Weather placeholder"}
        data-testid="weather-summary"
      >
        <div className="weather-summary__head">
          <h2 className="weather-summary__title">Weather Outlook</h2>
          <span key={resolvedDestinationLabel} className="best-lbl weather-summary__code">{resolvedDestinationLabel}</span>
        </div>
        <div className="weather-summary__grid weather-summary__grid--skeleton">
          <div className="weather-summary__tile-skeleton">
            <div className="shim" style={{ width: "48%", height: "10px" }} />
            <div className="shim" style={{ width: "72%", height: "12px", marginBottom: 0 }} />
          </div>
          <div className="weather-summary__tile-skeleton">
            <div className="shim" style={{ width: "46%", height: "10px" }} />
            <div className="shim" style={{ width: "70%", height: "12px", marginBottom: 0 }} />
          </div>
          <div className="weather-summary__tile-skeleton">
            <div className="shim" style={{ width: "44%", height: "10px" }} />
            <div className="shim" style={{ width: "65%", height: "12px", marginBottom: 0 }} />
          </div>
          <div className="weather-summary__tile-skeleton">
            <div className="shim" style={{ width: "56%", height: "10px" }} />
            <div className="shim" style={{ width: "74%", height: "12px", marginBottom: 0 }} />
          </div>
        </div>
        <p className="weather-summary__hint">
          {isLoading ? "Checking destination weather..." : "Add a route to preview destination weather."}
        </p>
      </section>
    );
  }

  const condition = formatValue(weather.condition);
  const tempText = formatTemperatureC(weather.temperature_c);
  const feelsLikeText = formatTemperatureC(weather.feels_like_c);
  const minText = formatTemperatureC(weather.temp_min_c);
  const maxText = formatTemperatureC(weather.temp_max_c);
  const precipText = formatValue(weather.precipitation_chance);
  const forecastDate = formatWeatherDate(weather.forecast_date);

  const hasWeather =
    condition !== "N/A" ||
    tempText !== "N/A" ||
    minText !== "N/A" ||
    maxText !== "N/A" ||
    feelsLikeText !== "N/A" ||
    precipText !== "N/A";

  if (!hasWeather) {
    const strictWeatherUnavailable = weatherPresent === false;
    return (
      <section className="weather-summary weather-summary--loading">
        <div className="weather-summary__head">
          <h2 className="weather-summary__title">Weather Outlook</h2>
          <span key={resolvedDestinationLabel} className="best-lbl weather-summary__code">{resolvedDestinationLabel}</span>
        </div>
        <p className="weather-summary__hint">
          {strictWeatherUnavailable ? weatherAvailabilityHint(weatherReason) : "Finalizing forecast details for this destination."}
        </p>
      </section>
    );
  }

  return (
    <section className="weather-summary weather-summary--ready" data-testid="weather-summary" data-weather-ready="true">
      <div className="weather-summary__head">
        <h2 className="weather-summary__title">Weather Outlook</h2>
        <span key={resolvedDestinationLabel} className="best-lbl weather-summary__code">{resolvedDestinationLabel}</span>
      </div>

      <div className="weather-summary__grid">
        <p className="weather-summary__item"><span>Condition</span>{condition}</p>
        <p className="weather-summary__item"><span>Temperature</span>{tempText}</p>
        <p className="weather-summary__item"><span>Feels Like</span>{feelsLikeText}</p>
        <p className="weather-summary__item"><span>Daily Low</span>{minText}</p>
        <p className="weather-summary__item"><span>Daily High</span>{maxText}</p>
        <p className="weather-summary__item"><span>Precipitation</span>{precipText === "N/A" ? "N/A" : `${precipText}%`}</p>
        <p className="weather-summary__item"><span>Forecast Date</span>{forecastDate}</p>
      </div>
    </section>
  );
}
