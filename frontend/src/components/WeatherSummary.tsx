import { formatTemperatureC, formatWeatherDate } from "../lib/format";

type Props = {
  weather?: Record<string, unknown> | null;
  destinationCode?: string;
  isLoading?: boolean;
};

function formatValue(value: unknown) {
  if (value === null || value === undefined || value === "") return "N/A";
  return String(value);
}

export default function WeatherSummary({ weather, destinationCode, isLoading = false }: Props) {
  if (!weather || typeof weather !== "object") {
    return (
      <section
        className="weather-summary weather-summary--loading"
        aria-label={isLoading ? "Loading weather" : "Weather placeholder"}
      >
        <div className="weather-summary__head">
          <h2 className="weather-summary__title">Weather Outlook</h2>
          <span key={destinationCode || "DEL"} className="best-lbl weather-summary__code">{destinationCode || "DEL"}</span>
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
    return (
      <section className="weather-summary weather-summary--loading">
        <div className="weather-summary__head">
          <h2 className="weather-summary__title">Weather Outlook</h2>
          <span key={destinationCode || "DEL"} className="best-lbl weather-summary__code">{destinationCode || "DEL"}</span>
        </div>
        <p className="weather-summary__hint">Finalizing forecast details for this destination.</p>
      </section>
    );
  }

  return (
    <section className="weather-summary weather-summary--ready">
      <div className="weather-summary__head">
        <h2 className="weather-summary__title">Weather Outlook</h2>
        {destinationCode && <span key={destinationCode} className="best-lbl weather-summary__code">{destinationCode}</span>}
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
