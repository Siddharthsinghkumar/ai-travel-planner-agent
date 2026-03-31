#!/usr/bin/env node
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { reviewFixtures } from "../review/review-fixtures.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const frontendDir = resolve(__dirname, "..");
const repoRoot = resolve(frontendDir, "..");
const indexCssPath = resolve(frontendDir, "src", "index.css");
const tickerCssPath = resolve(frontendDir, "src", "components", "flights-ticker.css");
const photoAssetsDir = resolve(frontendDir, "src", "assets", "photos");
const outputPath = resolve(repoRoot, "review-demo.html");

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatPriceINR(value) {
  const numeric = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(numeric)) return String(value);
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(numeric);
}

function formatTemp(value) {
  const numeric = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(numeric)) return "N/A";
  return `${Math.round(numeric * 10) / 10}°C`;
}

function toDataUrl(path) {
  const ext = extname(path).toLowerCase();
  const mime =
    ext === ".jpg" || ext === ".jpeg"
      ? "image/jpeg"
      : ext === ".png"
        ? "image/png"
        : ext === ".webp"
          ? "image/webp"
          : "application/octet-stream";
  const raw = readFileSync(path);
  return `data:${mime};base64,${raw.toString("base64")}`;
}

function statusLabel(status) {
  if (status === "live") return "Live";
  if (status === "partial") return "Live (guided)";
  return "Coming soon";
}

function statusIcon(status) {
  if (status === "live") return "●";
  if (status === "partial") return "◐";
  return "○";
}

function buildFlightRows(fixtures) {
  return fixtures.flights
    .map((flight, idx) => {
      const isBest = Boolean(flight.best);
      const stopLabel = Number(flight.stops) === 0 ? "Direct" : `${flight.stops}`;
      const routeInfo = flight.layover_info ? `Layover: ${flight.layover_info}` : "Non-stop routing";
      const airlineCode = String(flight.flight_no || flight.airline || "FL")
        .split(" ")[0]
        .slice(0, 3)
        .toUpperCase();
      return `
      <div class="flights-stack__item" style="animation-delay:${Math.min(idx, 6) * 45}ms">
        <article class="flight-item ${isBest ? "best-pick flight-card--best" : ""}">
          <div class="airline-ico">${escapeHtml(airlineCode)}</div>
          <div class="fl-info">
            <div class="flight-item__topline">
              <span class="flight-rank ${isBest ? "flight-rank--best" : ""}">${isBest ? "Top pick" : `#${idx + 1}`}</span>
              <span class="flight-proof-note">${isBest ? "Booking handoff ready" : "Handoff available on supported providers"}</span>
            </div>
            <p class="fl-route">${escapeHtml(flight.flight_no)} · ${escapeHtml(flight.departure_time)} → ${escapeHtml(flight.arrival_time)}</p>
            <p class="fl-meta">${escapeHtml(flight.airline)} · ${escapeHtml(routeInfo)} · ${escapeHtml(String(flight.duration_min))} min · ${escapeHtml(stopLabel)}</p>
            <p class="fl-meta">Baggage: ${escapeHtml(flight.baggage || "Check airline")}</p>
            <div class="flight-card__actions">
              <a href="#" class="flight-card__link flight-card__link--primary" aria-disabled="true">Book now</a>
              <span class="fl-meta">Provider handoff opens a secure booking flow in a new tab.</span>
            </div>
          </div>
          ${isBest ? '<div class="best-lbl">Best value</div>' : ""}
          <div class="fl-price">${escapeHtml(formatPriceINR(flight.price_inr))}</div>
        </article>
      </div>`;
    })
    .join("\n");
}

function buildWeatherPanel(fixtures) {
  const weather = fixtures.weather;
  return `
  <section class="weather-summary weather-summary--ready">
    <div class="weather-summary__head">
      <h3 class="weather-summary__title">Weather Outlook</h3>
      <span class="best-lbl weather-summary__code">${escapeHtml(weather.location_label)}</span>
    </div>
    <div class="weather-summary__grid">
      <p class="weather-summary__item"><span>Condition</span>${escapeHtml(weather.condition)}</p>
      <p class="weather-summary__item"><span>Temperature</span>${escapeHtml(formatTemp(weather.temperature_c))}</p>
      <p class="weather-summary__item"><span>Feels Like</span>${escapeHtml(formatTemp(weather.feels_like_c))}</p>
      <p class="weather-summary__item"><span>Daily Low</span>${escapeHtml(formatTemp(weather.temp_min_c))}</p>
      <p class="weather-summary__item"><span>Daily High</span>${escapeHtml(formatTemp(weather.temp_max_c))}</p>
      <p class="weather-summary__item"><span>Precipitation</span>${escapeHtml(String(weather.precipitation_chance))}%</p>
      <p class="weather-summary__item"><span>Forecast Date</span>${escapeHtml(weather.forecast_date)}</p>
    </div>
  </section>`;
}

function buildReasoningPanel(fixtures) {
  const items = fixtures.reasoning
    .map((step) => `<li class="reasoning-list__item">${escapeHtml(step)}</li>`)
    .join("\n");

  return `
  <div class="reasoning-panel">
    <h3 class="reasoning-title">Selection evidence</h3>
    <ol class="reasoning-list">${items}</ol>
  </div>`;
}

function buildRouteReveal(fixtures) {
  const steps = fixtures.route_reveal.steps || [];
  const activeIdx = steps.reduce((last, step, idx) => (step.active ? idx : last), -1);
  const progress = ((activeIdx + 1) / Math.max(steps.length, 1)) * 100;

  const stepMarkup = steps
    .map(
      (step, idx) => `
      <article class="route-reveal-card ${step.active ? "route-reveal-card--active" : ""}" style="animation-delay:${idx * 80}ms">
        <div class="route-reveal-card__rail" aria-hidden="true"></div>
        <div class="route-reveal-card__body">
          <div class="route-reveal-card__index">0${idx + 1}</div>
          <h3 class="route-reveal-card__title">${escapeHtml(step.title)}</h3>
          <p class="route-reveal-card__desc">${escapeHtml(step.description)}</p>
        </div>
      </article>`,
    )
    .join("\n");

  return `
  <section id="route-reveal" class="experience-section experience-section--route reveal visible" aria-label="How the planner thinks">
    <div class="route-reveal-shell">
      <div class="route-reveal-intro">
        <div class="section-head route-reveal-intro__head">
          <p class="section-label">${escapeHtml(fixtures.route_reveal.intro_label)}</p>
          <h2 class="section-title">${escapeHtml(fixtures.route_reveal.intro_title)}</h2>
        </div>
        <p class="route-reveal-intro__sub">${escapeHtml(fixtures.route_reveal.intro_sub)}</p>
        <div class="route-reveal-progress" aria-hidden="true">
          <div class="route-reveal-progress__fill" style="width:${Math.max(progress, 0)}%"></div>
        </div>
        <p class="route-reveal-intro__status">${escapeHtml(fixtures.route_reveal.narrative_status)}</p>
      </div>
      <div class="route-reveal-track">${stepMarkup}</div>
    </div>
  </section>`;
}

function buildCuratedSection(fixtures, curatedImageMap) {
  const cards = (fixtures.curated_panels || [])
    .map(
      (panel, idx) => `
      <article class="curation-card curation-card--${escapeHtml(panel.mood || "coastal")} ${idx === 0 ? "curation-card--featured" : ""}">
        <div class="curation-card__media-wrap" aria-hidden="true">
          <img src="${escapeHtml(curatedImageMap[panel.image_asset] || "")}" alt="" class="curation-card__media" loading="lazy" decoding="async" />
        </div>
        <div class="curation-card__veil" aria-hidden="true"></div>
        <div class="curation-card__content">
          <p class="curation-card__kicker">${escapeHtml(panel.title)}</p>
          <h3 class="curation-card__route">${escapeHtml(panel.route)}</h3>
          <p class="curation-card__note">${escapeHtml(panel.note)}</p>
        </div>
      </article>`,
    )
    .join("\n");

  return `
  <section class="experience-section experience-section--curation reveal visible" aria-label="Curated route moods">
    <div class="section-head">
      <p class="section-label">Curated lanes</p>
      <h2 class="section-title">Editorial route rhythms for every trip profile</h2>
    </div>
    <div class="curation-grid">${cards}</div>
  </section>`;
}

function buildImmersiveSection(fixtures) {
  const points = fixtures.immersive.waypoints || ["DEL", "BOM", "GOI"];
  const chips = fixtures.immersive.chips || [];
  return `
  <section class="experience-section experience-section--immersive reveal visible" aria-label="Immersive route map">
    <div class="immersive-shell">
      <div class="immersive-shell__header">
        <p class="section-label">${escapeHtml(fixtures.immersive.title_label)}</p>
        <h2 class="section-title">${escapeHtml(fixtures.immersive.title)}</h2>
        <p class="immersive-shell__sub">${escapeHtml(fixtures.immersive.sub)}</p>
      </div>
      <div class="immersive-scene" aria-hidden="true">
        <div class="immersive-scene__halo immersive-scene__halo--top"></div>
        <div class="immersive-scene__halo immersive-scene__halo--bottom"></div>
        <div class="immersive-scene__plane"></div>
        <svg class="immersive-scene__routes" viewBox="0 0 720 320" preserveAspectRatio="none">
          <defs>
            <linearGradient id="reviewRoutePrimary" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stop-color="rgba(124, 98, 255, 0.95)" />
              <stop offset="100%" stop-color="rgba(90, 198, 255, 0.9)" />
            </linearGradient>
            <linearGradient id="reviewRouteSecondary" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stop-color="rgba(88, 233, 191, 0.9)" />
              <stop offset="100%" stop-color="rgba(120, 164, 255, 0.82)" />
            </linearGradient>
          </defs>
          <path class="immersive-route immersive-route--primary" d="M82 248 C 220 82, 366 92, 556 236" stroke="url(#reviewRoutePrimary)" />
          <path class="immersive-route immersive-route--secondary" d="M142 262 C 285 138, 406 126, 630 224" stroke="url(#reviewRouteSecondary)" />
        </svg>

        <div class="immersive-waypoint immersive-waypoint--left"><span class="immersive-waypoint__label">${escapeHtml(points[0] || "DEL")}</span></div>
        <div class="immersive-waypoint immersive-waypoint--mid"><span class="immersive-waypoint__label">${escapeHtml(points[1] || "BOM")}</span></div>
        <div class="immersive-waypoint immersive-waypoint--right"><span class="immersive-waypoint__label">${escapeHtml(points[2] || "GOI")}</span></div>

        <div class="immersive-chip immersive-chip--flight">${escapeHtml(chips[0] || "Top fare trajectory")}</div>
        <div class="immersive-chip immersive-chip--weather">${escapeHtml(chips[1] || "Weather layer synced")}</div>
        <div class="immersive-chip immersive-chip--handoff">${escapeHtml(chips[2] || "Booking handoff confidence")}</div>

        <div class="immersive-scene__glow"></div>
      </div>
    </div>
  </section>`;
}

function buildTicker(fixtures) {
  const names = fixtures.trust_names || [];
  const items = names.map((name) => `<span class="ticker-item">${escapeHtml(name)}</span>`).join("\n");
  return `
  <div id="reviewTrustStrip" class="trust-strip reveal visible" hidden>
    <p class="trust-strip__copy">${escapeHtml(fixtures.trust_copy || "Live itinerary confidence across high-traffic Indian carrier routes.")}</p>
    <div class="ticker-wrap" aria-label="Carrier coverage ticker">
      <div class="ticker-track" id="reviewTickerTrack">
        <div class="ticker-group" id="reviewTickerA">${items}</div>
        <div class="ticker-group" id="reviewTickerB">${items}</div>
      </div>
    </div>
  </div>`;
}

function buildCapabilities(fixtures) {
  const visible = (fixtures.capabilities || []).filter((item) => item.status !== "coming-soon");
  const coming = (fixtures.capabilities || []).filter((item) => item.status === "coming-soon");

  const cards = visible
    .map(
      (item) => `
      <article class="feat-card feat-card--${escapeHtml(item.status)} reveal visible">
        <div class="cap-pill cap-pill--${escapeHtml(item.status)}">
          <span class="cap-pill__icon" aria-hidden="true">${statusIcon(item.status)}</span>
          ${escapeHtml(statusLabel(item.status))}
        </div>
        <h3 class="feat-title">${escapeHtml(item.title)}</h3>
        <p class="feat-desc">${escapeHtml(item.description)}</p>
      </article>`,
    )
    .join("\n");

  const comingMarkup = coming
    .map(
      (item) => `<div class="cs-chip">${escapeHtml(item.title)} <span class="coming-soon-inline">Under development</span></div>`,
    )
    .join("\n");

  return `
  <!--
  SECTION: Capabilities
  ORIGIN: live-app parity from frontend/src/components/FeatureCapabilities.tsx + frontend/src/index.css
  RUNTIME: no (static initial parity)
  NOTES: card coverage mirrors current live capability set
  -->
  <section id="capabilities" class="capabilities-shell reveal visible experience-section experience-section--confidence">
    <section class="capabilities-section section-center">
      <div class="reveal visible">
        <p class="section-label">Capabilities</p>
        <h2 class="section-title">What this travel planner handles today</h2>
        <p class="section-sub">These cards reflect the current product surface, including chat-led capabilities and features still in progress.</p>
      </div>

      <div class="features-grid">${cards}</div>

      <!--
      SECTION: What’s coming
      ORIGIN: live-app parity from frontend/src/components/FeatureCapabilities.tsx
      RUNTIME: no (static initial parity)
      NOTES: future capability chip block
      -->
      <div class="coming-soon-row reveal visible">
        <div class="cs-label">What&#39;s coming</div>
        <div class="cs-chips">${comingMarkup}</div>
      </div>
    </section>
  </section>`;
}

const rawCss = readFileSync(indexCssPath, "utf8");
const rawTickerCss = readFileSync(tickerCssPath, "utf8");
const cssWithoutTailwindDirectives = rawCss
  .split("\n")
  .filter((line) => !line.trim().startsWith("@tailwind"))
  .join("\n");

const fixtures = reviewFixtures;
const curatedImageMap = Object.fromEntries(
  (fixtures.curated_panels || [])
    .filter((panel) => typeof panel.image_asset === "string" && panel.image_asset.trim().length > 0)
    .map((panel) => {
      const fileName = panel.image_asset.trim();
      const assetPath = resolve(photoAssetsDir, fileName);
      return [fileName, toDataUrl(assetPath)];
    }),
);
const best = fixtures.best_flight;
const flightRows = buildFlightRows(fixtures);
const weatherPanel = buildWeatherPanel(fixtures);
const reasoningPanel = buildReasoningPanel(fixtures);
const routeReveal = buildRouteReveal(fixtures);
const curatedSection = buildCuratedSection(fixtures, curatedImageMap);
const immersiveSection = buildImmersiveSection(fixtures);
const tickerSection = buildTicker(fixtures);
const capabilitiesSection = buildCapabilities(fixtures);

const html = `<!doctype html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Travelyst Review Demo</title>
  <style>
${cssWithoutTailwindDirectives}

${rawTickerCss}

.review-dot {
  width: 6px;
  height: 6px;
  border-radius: 999px;
  background: var(--accent-2);
  box-shadow: 0 0 10px rgba(79, 163, 255, 0.55);
}

.review-hero-frame {
  width: min(1120px, 100%);
}

.review-reset {
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text-2);
  border-radius: 11px;
  padding: 9px 12px;
  font-size: 12px;
  white-space: nowrap;
  transition: all var(--motion-fast) ease;
}

.review-reset:hover {
  border-color: rgba(139, 100, 255, 0.42);
  color: var(--text-1);
  transform: translateY(-1px);
}

.review-watermark {
  position: fixed;
  right: 10px;
  bottom: 8px;
  z-index: 15;
  pointer-events: none;
  color: var(--text-3);
  font-size: 10px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  opacity: 0.55;
}

.reveal {
  opacity: 1;
  transform: none;
}

@media (max-width: 980px) {
  .review-hero-frame {
    width: 100%;
  }
}
  </style>
</head>
<body>
  <div class="app-shell">
    <!--
    SECTION: Aurora canvas
    ORIGIN: ported/adapted from ui_dummy_prototype.html + frontend/src/components/AuroraCanvas.tsx
    RUNTIME: yes
    NOTES: canvas-based wallpaper, active on load
    -->
    <div class="aurora-scene" aria-hidden="true">
      <canvas id="reviewAurora" class="aurora-canvas"></canvas>
      <div class="vignette"></div>
      <div class="grid-overlay"></div>
      <div class="grain"></div>
    </div>

    <div class="page">
      <!--
      SECTION: Top navigation
      ORIGIN: live-app parity from frontend/src/App.tsx + frontend/src/index.css
      RUNTIME: no (static initial parity)
      NOTES: review indicator intentionally de-emphasized
      -->
      <nav class="top-nav">
        <div class="nav-logo">
          <span class="logo-mark" aria-hidden="true"><span class="logo-glyph">T</span></span>
          Travelyst
        </div>

        <div class="nav-links" aria-label="Primary">
          <a href="#planner">Planner</a>
          <a href="#results">Results</a>
          <a href="#capabilities">Capabilities</a>
        </div>

        <div class="nav-right">
          <div class="theme-switch" role="group" aria-label="Theme mode">
            <button type="button" class="theme-switch__button" data-theme-pref="system">Auto</button>
            <button type="button" class="theme-switch__button theme-switch__button--active" data-theme-pref="dark">Dark</button>
            <button type="button" class="theme-switch__button" data-theme-pref="light">Light</button>
          </div>
          ${fixtures.status_text && !String(fixtures.status_text).toLowerCase().includes("live") ? `
          <div class="api-status api-status--offline">${escapeHtml(fixtures.status_text)}</div>
          ` : ""}
          <a class="btn-primary" href="#planner">Start planning →</a>
        </div>
      </nav>

      <main class="app-main">
        <div class="experience-shell">
          <!--
          SECTION: Hero shell
          ORIGIN: live-app parity from frontend/src/App.tsx + frontend/src/index.css
          RUNTIME: no (static initial parity)
          NOTES: intended to match live app first-load state
          -->
          <section id="planner" class="hero experience-section experience-section--hero">
            <div class="hero-intro reveal visible review-hero-frame">
              <div class="hero-badge"><span class="badge-dot" aria-hidden="true"></span>${escapeHtml(fixtures.hero_badge)}</div>
              <h1 class="hero-title">
                <span class="title-line-1">${escapeHtml(fixtures.hero_title_line_1)}</span>
                <span class="title-line-2">${escapeHtml(fixtures.hero_title_line_2)}</span>
              </h1>
              <p class="hero-sub">${escapeHtml(fixtures.hero_sub)}</p>
              <div class="hero-trust-row" aria-label="Trust indicators">
                ${fixtures.hero_trust_signals
                  .map((signal) => `<span class="hero-trust-pill">${escapeHtml(signal)}</span>`)
                  .join("\n")}
              </div>
            </div>

            <section class="hero-grid">
              <div class="hero-left">
                <div class="search-card">
                  <!--
                  SECTION: Hero / Planner form
                  ORIGIN: live-app parity from frontend/src/components/QueryForm.tsx
                  RUNTIME: yes (local fake demo flow on submit)
                  NOTES: initial values mirror live app first-load defaults
                  -->
                  <form id="reviewPlannerForm" class="glass-form planner-form">
                    <div class="trip-tabs">
                      <button type="button" class="trip-tab ${fixtures.planner.trip_type === "one-way" ? "active" : ""}" data-trip="one-way">One-way</button>
                      <button type="button" class="trip-tab ${fixtures.planner.trip_type === "round-trip" ? "active" : ""}" data-trip="round-trip">Round-trip</button>
                      <button type="button" class="trip-tab ${fixtures.planner.trip_type === "via-stopover" ? "active" : ""}" data-trip="via-stopover">Via / Stopover</button>
                    </div>
                    <div class="nl-row">
                      <span class="nl-icon" aria-hidden="true">↗</span>
                      <div class="nl-main">
                        <p class="planner-guidance">Primary input: describe your trip naturally</p>
                        <textarea id="reviewQueryInput" class="nl-input" rows="1">${escapeHtml(fixtures.planner.query)}</textarea>
                      </div>
                      <button type="submit" class="nl-send" aria-label="Submit query">
                        <span class="nl-send__arrow" aria-hidden="true">→</span>
                      </button>
                    </div>
                    <div class="planner-structured-head">
                      <span class="planner-structured-label">Assistive fields</span>
                      <span class="planner-structured-note">Optional quick controls. We merge these with your natural-language prompt.</span>
                    </div>
                    <div class="fields-row">
                      <label class="field-group"><span class="field-label">Origin</span><input id="reviewOriginInput" class="f-input" value="${escapeHtml(fixtures.planner.origin)}" /></label>
                      <label class="field-group"><span class="field-label">Destination</span><input id="reviewDestinationInput" class="f-input" value="${escapeHtml(fixtures.planner.destination)}" /></label>
                      <label class="field-group">
                        <span class="field-label">Travel date</span>
                        <div class="date-shell">
                          <span id="reviewDateDisplay" class="date-display ${fixtures.planner.date ? "date-display--value" : ""}">${escapeHtml(fixtures.planner.date || "dd/mm/yyyy")}</span>
                          <span class="date-icon" aria-hidden="true">📅</span>
                          <input id="reviewDateInput" class="date-native f-date" type="date" value="${escapeHtml(fixtures.planner.date)}" lang="en-GB" />
                        </div>
                      </label>
                    </div>
                    <div class="card-footer min-w-0">
                      <button id="reviewPlanButton" type="submit" class="plan-btn"><span class="plan-btn__content">Plan my trip →</span></button>
                      <button id="reviewResetButton" type="button" class="review-reset">Reset / Replay</button>
                    </div>
                  </form>
                </div>

                <!--
                SECTION: Suggestion chips
                ORIGIN: live-app parity from frontend/src/App.tsx + QueryForm suggest event behavior
                RUNTIME: yes (chip click updates planner query locally)
                NOTES: no backend calls
                -->
                <div class="suggestions-row sugg-strip">
                  <div class="sugg-scroll" id="reviewSuggestions">
                    ${fixtures.suggestion_chips
                      .map(
                        (item) =>
                          `<button type="button" class="s-chip history-chip" title="${escapeHtml(item)}" aria-label="${escapeHtml(item)}"><span class="s-chip__label">${escapeHtml(item)}</span></button>`,
                      )
                      .join("\n")}
                  </div>
                </div>

                <!--
                SECTION: Hero stream card
                ORIGIN: live-app parity from frontend/src/components/StreamPane.tsx
                RUNTIME: yes (local staged stream simulation)
                NOTES: starts in idle AI-thinking state
                -->
                <article id="reviewStreamCard" class="r-card hero-stream-card">
                  <div id="reviewStreamLabel" class="r-label r-label--inactive"><span class="r-dot" aria-hidden="true"></span>AI thinking</div>
                  <div id="reviewStreamPane" class="stream-pane" aria-live="polite" aria-busy="false"></div>
                </article>

                <a id="reviewResultsNudge" class="results-nudge" href="#results" hidden>${fixtures.flights.length} flights found below ↓</a>
              </div>

              <!--
              SECTION: Hero support cards
              ORIGIN: live-app parity from frontend/src/App.tsx + WeatherSummary + AIReasoningPanel
              RUNTIME: yes (shown after local planning interaction)
              NOTES: hidden on first load to match live initial state
              -->
              <aside id="reviewHeroRight" class="hero-right" hidden>
                <article id="reviewWeatherCard" class="r-card support-card support-card--reveal">
                  <div id="reviewWeatherLabel" class="r-label r-label--sidebar r-label--inactive"><span class="r-dot" aria-hidden="true"></span>Destination weather</div>
                  <div id="reviewWeatherPane"></div>
                </article>

                <article id="reviewReasoningCard" class="r-card support-card support-card--reveal">
                  <div id="reviewReasoningLabel" class="r-label r-label--sidebar r-label--inactive"><span class="r-dot" aria-hidden="true"></span>AI reasoning trace</div>
                  <div id="reviewReasoningPane"></div>
                </article>
              </aside>
            </section>

            <a href="#route-reveal" class="hero-scroll-cue">How the planner thinks ↓</a>
          </section>

          <div class="experience-divider reveal visible" aria-hidden="true"></div>

          <!--
          SECTION: Planner-thinking narrative
          ORIGIN: live-app parity from frontend/src/App.tsx
          RUNTIME: no (static initial parity)
          NOTES: narrative section preserved for parity flow
          -->
          ${routeReveal}

          <div class="experience-divider reveal visible" aria-hidden="true"></div>

          <!--
          SECTION: Curated lanes
          ORIGIN: live-app parity from frontend/src/App.tsx
          RUNTIME: no (static initial parity)
          NOTES: editorial section preserved for parity flow
          -->
          ${curatedSection}

          <div class="experience-divider reveal visible" aria-hidden="true"></div>

          <!--
          SECTION: Immersive route plane
          ORIGIN: live-app parity from frontend/src/App.tsx + frontend/src/index.css
          RUNTIME: no (static initial parity)
          NOTES: single immersive section retained
          -->
          ${immersiveSection}

          <div class="experience-divider reveal visible" aria-hidden="true"></div>

          <!--
          SECTION: Product proof + ranked results
          ORIGIN: live-app parity from frontend/src/App.tsx + FlightsList surface
          RUNTIME: yes (local fake demo hydrates proof/results)
          NOTES: initial state mirrors live awaiting-query posture
          -->
          <section id="results" class="experience-section experience-section--proof">
            <div class="section-head reveal visible">
              <p class="section-label">Product proof</p>
              <h2 class="section-title">Choose faster with clear ranking, evidence, and booking confidence</h2>
            </div>
            <article id="reviewProofPlaceholder" class="r-card proof-placeholder reveal visible">
              <p class="proof-card__kicker">Product proof</p>
              <h3 class="proof-card__title">Ranked proof appears right after your first query.</h3>
              <p class="proof-card__summary">Submit a route to unlock top-pick rationale, weather intelligence, and booking confidence in one view.</p>
            </article>
            <div id="reviewProofOverview" class="proof-overview-grid reveal visible" hidden>
              <article class="r-card proof-card proof-card--best">
                <div class="proof-card__head">
                  <p class="proof-card__kicker">Top recommendation</p>
                  <span id="reviewProofStatus" class="proof-status proof-status--awaiting-query">Awaiting query</span>
                </div>
                <h3 id="reviewProofTitle" class="proof-card__title">Submit a route to generate a recommendation.</h3>
                <p id="reviewProofSummary" class="proof-card__summary">The planner will rank options, layer destination weather, and show booking confidence cues.</p>
                <div class="proof-chip-row">
                  <span id="reviewProofBookingChip" class="proof-chip">Booking readiness pending</span>
                  <span id="reviewProofWeatherChip" class="proof-chip">Weather insight pending</span>
                </div>
              </article>

              <article class="r-card proof-card proof-card--evidence">
                <p class="proof-card__kicker">Evidence stack</p>
                <ul class="proof-evidence-list">
                  <li class="proof-evidence-item"><span class="proof-evidence-item__label">Ranked shortlist</span><span id="reviewProofShortlist" class="proof-evidence-item__value">No shortlist yet</span></li>
                  <li class="proof-evidence-item"><span class="proof-evidence-item__label">Weather intelligence</span><span id="reviewProofWeather" class="proof-evidence-item__value">No weather insight yet</span></li>
                  <li class="proof-evidence-item"><span class="proof-evidence-item__label">Selection rationale</span><span id="reviewProofReasoning" class="proof-evidence-item__value">Reasoning appears after planning starts</span></li>
                  <li class="proof-evidence-item"><span class="proof-evidence-item__label">Booking confidence</span><span id="reviewProofBooking" class="proof-evidence-item__value">Booking confidence appears with ranked results</span></li>
                </ul>
              </article>
            </div>
            <div id="reviewResultsWrap" class="result-wrap reveal visible" hidden>
              <article id="reviewFlightsCard" class="r-card results-card r-card--compact">
                <div class="r-label r-label--secondary"><span class="r-dot" aria-hidden="true"></span>Ranked shortlist</div>
                <div id="reviewFlightsPane"></div>
              </article>

              <article class="r-card results-card">
                <div class="r-label r-label--secondary"><span class="r-dot" aria-hidden="true"></span>Return leg snapshot</div>
                <p class="flight-item__summary">${escapeHtml(fixtures.return_leg.airline)} ${escapeHtml(fixtures.return_leg.flight_no)} · ${escapeHtml(fixtures.return_leg.departure_time)} → ${escapeHtml(fixtures.return_leg.arrival_time)} · ${escapeHtml(formatPriceINR(fixtures.return_leg.price_inr))}</p>
              </article>
            </div>
          </section>

          <!--
          SECTION: Carrier strip
          ORIGIN: live-app parity from frontend/src/components/FlightsTicker.tsx
          RUNTIME: yes (local ticker motion)
          NOTES: no network calls
          -->
          ${tickerSection}

          ${capabilitiesSection}
        </div>
      </main>
    </div>
  </div>
  <div class="review-watermark"><span class="review-dot" aria-hidden="true"></span> Offline review</div>

  <!--
  SECTION: Local demo runtime script
  ORIGIN: adapted from ui_dummy_prototype.html + frontend/src/App.tsx / QueryForm / AuroraCanvas
  RUNTIME: yes
  NOTES: offline-only interaction model; activated by planner submit; no backend calls
  -->
  <script>
    (function () {
      const root = document.documentElement;
      const buttons = Array.from(document.querySelectorAll('[data-theme-pref]'));
      const storageKey = 'travelyst_review_theme_preference';
      const fixtures = ${JSON.stringify(fixtures)};
      const finalWeatherHtml = ${JSON.stringify(weatherPanel)};
      const finalReasoningHtml = ${JSON.stringify(reasoningPanel)};
      const finalFlightsHtml = ${JSON.stringify('<div class="space-y-2 flights-stack">' + flightRows + '</div>')};
      let repaintAurora = null;

      function resolveTheme(pref, prefersDark) {
        if (pref === 'system') return prefersDark ? 'dark' : 'light';
        return pref;
      }

      function applyTheme(pref) {
        const resolved = resolveTheme(pref, window.matchMedia('(prefers-color-scheme: dark)').matches);
        root.setAttribute('data-theme', resolved);
        root.style.colorScheme = resolved;
        buttons.forEach((button) => {
          const active = button.getAttribute('data-theme-pref') === pref;
          button.classList.toggle('theme-switch__button--active', active);
        });
        if (typeof repaintAurora === 'function') repaintAurora();
        if (pref === 'system') {
          localStorage.removeItem(storageKey);
        } else {
          localStorage.setItem(storageKey, pref);
        }
      }

      const stored = localStorage.getItem(storageKey);
      const initialPref = stored === 'dark' || stored === 'light' ? stored : 'dark';
      applyTheme(initialPref);

      buttons.forEach((button) => {
        button.addEventListener('click', () => {
          const pref = button.getAttribute('data-theme-pref');
          if (pref === 'system' || pref === 'dark' || pref === 'light') applyTheme(pref);
        });
      });

      const media = window.matchMedia('(prefers-color-scheme: dark)');
      media.addEventListener('change', () => {
        const pref = localStorage.getItem(storageKey);
        if (!pref) applyTheme('system');
      });

      function escapeHtml(value) {
        return String(value)
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;')
          .replace(/'/g, '&#39;');
      }

      function formatPriceINR(value) {
        const num = Number(value);
        if (!Number.isFinite(num)) return String(value);
        return new Intl.NumberFormat('en-IN', {
          style: 'currency',
          currency: 'INR',
          maximumFractionDigits: 0,
        }).format(num);
      }

      function formatTemp(value) {
        const num = Number(value);
        if (!Number.isFinite(num)) return 'N/A';
        return String(Math.round(num * 10) / 10) + '°C';
      }

      function formatWeatherDate(value) {
        if (!value) return 'N/A';
        const date = new Date(String(value) + 'T00:00:00');
        if (Number.isNaN(date.getTime())) return String(value);
        return new Intl.DateTimeFormat('en-GB', { weekday: 'short', day: 'numeric', month: 'short' }).format(date);
      }

      function chunkText(text, chunkSize) {
        const words = String(text).split(/\s+/).filter(Boolean);
        const chunks = [];
        for (let i = 0; i < words.length; i += chunkSize) {
          chunks.push(words.slice(i, i + chunkSize).join(' ') + ' ');
        }
        return chunks;
      }

      const state = {
        tripType: fixtures.planner && fixtures.planner.trip_type ? fixtures.planner.trip_type : 'round-trip',
        query: fixtures.planner && fixtures.planner.query ? fixtures.planner.query : '',
        origin: fixtures.planner && fixtures.planner.origin ? fixtures.planner.origin : '',
        destination: fixtures.planner && fixtures.planner.destination ? fixtures.planner.destination : '',
        date: fixtures.planner && fixtures.planner.date ? fixtures.planner.date : '',
        stage: 'idle',
        isStreaming: false,
        isSubmitting: false,
        tokens: '',
        finalText: '',
        weather: null,
        reasoning: [],
        flights: [],
        bestFlight: null,
        timers: [],
      };

      const plannerForm = document.getElementById('reviewPlannerForm');
      const queryInput = document.getElementById('reviewQueryInput');
      const originInput = document.getElementById('reviewOriginInput');
      const destinationInput = document.getElementById('reviewDestinationInput');
      const dateInput = document.getElementById('reviewDateInput');
      const dateDisplay = document.getElementById('reviewDateDisplay');
      const planButton = document.getElementById('reviewPlanButton');
      const resetButton = document.getElementById('reviewResetButton');

      const streamCard = document.getElementById('reviewStreamCard');
      const streamLabel = document.getElementById('reviewStreamLabel');
      const streamPane = document.getElementById('reviewStreamPane');
      const weatherCard = document.getElementById('reviewWeatherCard');
      const weatherLabel = document.getElementById('reviewWeatherLabel');
      const weatherPane = document.getElementById('reviewWeatherPane');
      const reasoningCard = document.getElementById('reviewReasoningCard');
      const reasoningLabel = document.getElementById('reviewReasoningLabel');
      const reasoningPane = document.getElementById('reviewReasoningPane');
      const heroRight = document.getElementById('reviewHeroRight');
      const flightsCard = document.getElementById('reviewFlightsCard');
      const flightsPane = document.getElementById('reviewFlightsPane');
      const resultsNudge = document.getElementById('reviewResultsNudge');
      const trustStrip = document.getElementById('reviewTrustStrip');
      const proofPlaceholder = document.getElementById('reviewProofPlaceholder');
      const proofOverview = document.getElementById('reviewProofOverview');
      const resultsWrap = document.getElementById('reviewResultsWrap');

      const proofStatus = document.getElementById('reviewProofStatus');
      const proofTitle = document.getElementById('reviewProofTitle');
      const proofSummary = document.getElementById('reviewProofSummary');
      const proofBookingChip = document.getElementById('reviewProofBookingChip');
      const proofWeatherChip = document.getElementById('reviewProofWeatherChip');
      const proofShortlist = document.getElementById('reviewProofShortlist');
      const proofWeather = document.getElementById('reviewProofWeather');
      const proofReasoning = document.getElementById('reviewProofReasoning');
      const proofBooking = document.getElementById('reviewProofBooking');

      const streamChunks = chunkText(fixtures.stream_brief || '', 6);
      const fullReasoning = Array.isArray(fixtures.reasoning) ? fixtures.reasoning : [];
      const fullFlights = Array.isArray(fixtures.flights) ? fixtures.flights : [];
      const fullWeather = fixtures.weather || null;

      function setLabelTone(element, isSidebar, toneClass, labelText) {
        if (!element) return;
        let cls = 'r-label';
        if (isSidebar) cls += ' r-label--sidebar';
        if (toneClass) cls += ' ' + toneClass;
        element.className = cls;
        element.innerHTML = '<span class="r-dot" aria-hidden="true"></span>' + escapeHtml(labelText);
      }

      function setCardState(card, baseClass, tone) {
        if (!card) return;
        const classes = ['r-card', baseClass];
        if (baseClass === 'support-card') classes.push('support-card--reveal');
        if (tone === 'live') classes.push(baseClass + '--live');
        if (tone === 'loading') classes.push(baseClass + '--loading');
        card.className = classes.join(' ');
      }

      function renderStreamPane() {
        if (!streamPane) return;
        if (state.isStreaming && !state.tokens) {
          streamPane.innerHTML = '<div class="stream-pane__loading"><p class="stream-pane__loading-title">Finding your best options...</p><div class="shim-wrap"><div class="shim" style="width: 92%"></div><div class="shim" style="width: 74%"></div></div></div>';
          streamPane.setAttribute('aria-busy', 'true');
          return;
        }

        if (state.isStreaming && state.tokens) {
          streamPane.innerHTML = '<div class="r-text llm-pane min-w-0 stream-pane__body">' + escapeHtml(state.tokens) + '<span class="stream-caret" aria-hidden="true"></span></div><div class="stream-pane__controls"><div class="min-w-0 break-words stream-pane__status">Building your trip summary...</div><button id="reviewStopStreamButton" type="button" class="stream-pane__cancel" aria-label="Cancel streaming">Stop</button></div>';
          streamPane.setAttribute('aria-busy', 'true');
          const stopButton = document.getElementById('reviewStopStreamButton');
          if (stopButton) stopButton.addEventListener('click', resetDemo);
          return;
        }

        if (state.finalText) {
          streamPane.innerHTML = '<div class="r-text llm-pane min-w-0 stream-pane__body stream-pane__body--final">' + escapeHtml(state.finalText) + '</div>';
          streamPane.setAttribute('aria-busy', 'false');
          return;
        }

        streamPane.innerHTML = '<div class="stream-empty"><div class="stream-empty__icon" aria-hidden="true">◉</div><p class="stream-empty__title">Share your route to begin</p><p class="stream-empty__description">You will get a best-flight callout, destination weather, and packing guidance in one view.</p></div>';
        streamPane.setAttribute('aria-busy', 'false');
      }

      function renderWeatherPane() {
        if (!weatherPane) return;
        if (state.weather) {
          weatherPane.innerHTML = finalWeatherHtml;
          return;
        }

        if (state.isStreaming) {
          weatherPane.innerHTML = '<section class="weather-summary weather-summary--loading"><div class="weather-summary__head"><h3 class="weather-summary__title">Weather Outlook</h3><span class="best-lbl weather-summary__code">Checking...</span></div><div class="weather-summary__grid weather-summary__grid--skeleton"><div class="weather-summary__tile-skeleton"><div class="shim" style="width:48%;height:10px"></div><div class="shim" style="width:72%;height:12px;margin-bottom:0"></div></div><div class="weather-summary__tile-skeleton"><div class="shim" style="width:46%;height:10px"></div><div class="shim" style="width:70%;height:12px;margin-bottom:0"></div></div></div><p class="weather-summary__hint">Checking destination weather...</p></section>';
          return;
        }

        weatherPane.innerHTML = '<section class="weather-summary weather-summary--loading"><div class="weather-summary__head"><h3 class="weather-summary__title">Weather Outlook</h3><span class="best-lbl weather-summary__code">Awaiting route</span></div><p class="weather-summary__hint">Run the planner to populate destination weather guidance.</p></section>';
      }

      function renderReasoningPane() {
        if (!reasoningPane) return;
        if (state.reasoning.length > 0) {
          const items = state.reasoning
            .map((step) => '<li class="reasoning-list__item">' + escapeHtml(step) + '</li>')
            .join('');
          reasoningPane.innerHTML = '<div class="reasoning-panel"><h3 class="reasoning-title">Selection evidence</h3><ol class="reasoning-list">' + items + '</ol>' + (state.isStreaming ? '<div class="reasoning-foot">Evaluating top route trade-offs...</div>' : '') + '</div>';
          return;
        }

        reasoningPane.innerHTML = '<div class="reasoning-panel"><h3 class="reasoning-title">Selection evidence</h3><div class="empty-state empty-state--reasoning"><p class="empty-state__title">' + (state.isStreaming ? 'Ranking route trade-offs and timing fit...' : 'Run a search to see why this option wins.') + '</p></div></div>';
      }

      function renderFlightsPane() {
        if (!flightsPane) return;
        if (state.flights.length > 0) {
          flightsPane.innerHTML = finalFlightsHtml;
          return;
        }

        if (state.isStreaming) {
          flightsPane.innerHTML = '<div class="flights-shimmer"><div class="flight-skeleton-card"><div class="flight-skeleton-card__left"><div class="shim" style="width:26px;height:26px"></div><div class="flight-skeleton-card__lines"><div class="shim" style="width:68%"></div><div class="shim" style="width:88%;opacity:0.75"></div></div></div><div class="shim" style="width:82px;height:16px;margin-bottom:0"></div></div></div>';
          return;
        }

        flightsPane.innerHTML = '<div class="empty-state empty-state--flights empty-state--flights-compact"><p class="empty-state__title">Search to load ranked flight options.</p><p class="empty-state__hint">Your strongest match will be highlighted first.</p></div>';
      }

      function renderProof() {
        if (!proofStatus || !proofTitle || !proofSummary) return;

        if (state.bestFlight) {
          proofStatus.className = 'proof-status proof-status--decision-ready';
          proofStatus.textContent = 'Decision-ready';
          proofTitle.textContent = state.bestFlight.airline + ' ' + state.bestFlight.flight_no + ' · ' + formatPriceINR(state.bestFlight.price_inr);
          proofSummary.textContent = state.bestFlight.departure_time + ' → ' + state.bestFlight.arrival_time + ' · ' + String(state.bestFlight.duration_min) + ' min · ' + (Number(state.bestFlight.stops) === 0 ? 'Non-stop route' : String(state.bestFlight.stops) + ' stop route');
          proofBookingChip.textContent = 'Provider handoff ready';
          proofWeatherChip.textContent = 'Weather: ' + (fullWeather ? fullWeather.condition + ' · ' + formatTemp(fullWeather.temperature_c) : 'Unavailable');
          proofShortlist.textContent = String(fullFlights.length) + ' candidates compared';
          proofWeather.textContent = fullWeather ? fullWeather.condition + ' · ' + formatTemp(fullWeather.temperature_c) : 'No weather insight yet';
          proofReasoning.textContent = fullReasoning[0] || 'Reasoning details available';
          proofBooking.textContent = 'Secure handoff link available on recommended option';
          return;
        }

        if (state.isStreaming) {
          proofStatus.className = 'proof-status proof-status--analyzing';
          proofStatus.textContent = 'Analyzing';
          proofTitle.textContent = 'Building recommendation from live route signals...';
          proofSummary.textContent = 'Comparing flight practicality, fare quality, and weather fit.';
          proofBookingChip.textContent = 'Booking readiness pending';
          proofWeatherChip.textContent = state.weather ? 'Weather: ' + state.weather.condition + ' · ' + formatTemp(state.weather.temperature_c) : 'Weather insight pending';
          proofShortlist.textContent = state.flights.length > 0 ? String(state.flights.length) + ' candidates compared' : 'Compiling ranked shortlist';
          proofWeather.textContent = state.weather ? state.weather.condition + ' · ' + formatTemp(state.weather.temperature_c) : 'No weather insight yet';
          proofReasoning.textContent = state.reasoning[0] || 'Reasoning appears after planning starts';
          proofBooking.textContent = 'Booking confidence appears with ranked results';
          return;
        }

        proofStatus.className = 'proof-status proof-status--awaiting-query';
        proofStatus.textContent = 'Awaiting query';
        proofTitle.textContent = 'Submit a route to generate a recommendation.';
        proofSummary.textContent = 'The planner will rank options, layer destination weather, and show booking confidence cues.';
        proofBookingChip.textContent = 'Booking readiness pending';
        proofWeatherChip.textContent = 'Weather insight pending';
        proofShortlist.textContent = 'No shortlist yet';
        proofWeather.textContent = 'No weather insight yet';
        proofReasoning.textContent = 'Reasoning appears after planning starts';
        proofBooking.textContent = 'Booking confidence appears with ranked results';
      }

      function renderState() {
        const hasStream = state.tokens.length > 0 || state.finalText.length > 0;
        const hasWeather = Boolean(state.weather);
        const hasReasoning = state.reasoning.length > 0;
        const hasFlights = state.flights.length > 0;
        const showProofSurface = state.stage !== 'idle' || state.isStreaming || hasFlights || hasWeather || hasReasoning;

        setLabelTone(streamLabel, false, hasStream ? 'r-label--live' : state.isStreaming ? 'r-label--waiting' : 'r-label--inactive', hasStream ? 'Trip brief' : 'AI thinking');
        setLabelTone(weatherLabel, true, hasWeather ? 'r-label--live' : state.isStreaming ? 'r-label--waiting' : 'r-label--inactive', 'Destination weather');
        setLabelTone(reasoningLabel, true, hasReasoning ? 'r-label--live' : state.isStreaming ? 'r-label--waiting' : 'r-label--inactive', 'AI reasoning trace');

        setCardState(streamCard, 'hero-stream-card', hasStream ? 'live' : state.isStreaming ? 'loading' : null);
        setCardState(weatherCard, 'support-card', hasWeather ? 'live' : state.isStreaming ? 'loading' : null);
        setCardState(reasoningCard, 'support-card', hasReasoning ? 'live' : state.isStreaming ? 'loading' : null);
        setCardState(flightsCard, 'results-card', hasFlights ? 'live' : state.isStreaming ? 'loading' : null);
        if (flightsCard && !hasFlights && !state.isStreaming) flightsCard.classList.add('r-card--compact');
        if (heroRight) heroRight.hidden = !(hasWeather || hasReasoning);
        if (proofPlaceholder) proofPlaceholder.hidden = showProofSurface;
        if (proofOverview) proofOverview.hidden = !showProofSurface;
        if (resultsWrap) resultsWrap.hidden = !showProofSurface;
        if (trustStrip) trustStrip.hidden = !showProofSurface;

        renderStreamPane();
        renderWeatherPane();
        renderReasoningPane();
        renderFlightsPane();
        renderProof();

        if (resultsNudge) {
          if (hasFlights) {
            resultsNudge.hidden = false;
            resultsNudge.textContent = String(state.flights.length) + ' ' + (state.flights.length === 1 ? 'flight' : 'flights') + ' found below ↓';
          } else {
            resultsNudge.hidden = true;
          }
        }

        if (planButton) {
          planButton.innerHTML = '<span class="plan-btn__content">' + (state.isSubmitting || state.isStreaming ? 'Planning your trip...' : 'Plan my trip →') + '</span>';
        }
      }

      function clearTimers() {
        state.timers.forEach((id) => clearTimeout(id));
        state.timers = [];
      }

      function schedule(delay, fn) {
        const id = setTimeout(fn, delay);
        state.timers.push(id);
      }

      function resetDemo() {
        clearTimers();
        state.stage = 'idle';
        state.isStreaming = false;
        state.isSubmitting = false;
        state.tokens = '';
        state.finalText = '';
        state.weather = null;
        state.reasoning = [];
        state.flights = [];
        state.bestFlight = null;
        renderState();
      }

      function startDemo() {
        clearTimers();
        state.query = queryInput ? queryInput.value.trim() : state.query;
        state.origin = originInput ? originInput.value.trim() : state.origin;
        state.destination = destinationInput ? destinationInput.value.trim() : state.destination;
        state.date = dateInput ? dateInput.value : state.date;

        state.stage = 'loading';
        state.isStreaming = true;
        state.isSubmitting = true;
        state.tokens = '';
        state.finalText = '';
        state.weather = null;
        state.reasoning = [];
        state.flights = [];
        state.bestFlight = null;
        renderState();

        schedule(420, () => {
          state.stage = 'reasoning';
          state.reasoning = fullReasoning.slice(0, 1);
          renderState();
        });

        schedule(860, () => {
          state.stage = 'streaming';
          streamChunks.forEach((chunk, index) => {
            schedule(index * 180, () => {
              state.tokens += chunk;
              renderState();
            });
          });
        });

        schedule(1680, () => {
          state.weather = fullWeather;
          renderState();
        });

        schedule(2260, () => {
          state.flights = fullFlights.slice();
          state.bestFlight = fullFlights[0] || null;
          state.reasoning = fullReasoning.slice(0, 2);
          renderState();
        });

        schedule(3220, () => {
          state.stage = 'final';
          state.isStreaming = false;
          state.isSubmitting = false;
          state.finalText = fixtures.stream_brief || '';
          state.tokens = '';
          state.reasoning = fullReasoning.slice();
          renderState();
        });
      }

      function autoResizeTextarea() {
        if (!queryInput) return;
        queryInput.style.height = '0px';
        const nextHeight = Math.min(Math.max(queryInput.scrollHeight, 52), 280);
        queryInput.style.height = String(nextHeight) + 'px';
        queryInput.style.overflowY = queryInput.scrollHeight > 280 ? 'auto' : 'hidden';
      }

      function updateDateDisplay() {
        if (!dateDisplay || !dateInput) return;
        const value = dateInput.value;
        if (!value) {
          dateDisplay.textContent = 'dd/mm/yyyy';
          dateDisplay.classList.remove('date-display--value');
          return;
        }
        dateDisplay.textContent = formatWeatherDate(value);
        dateDisplay.classList.add('date-display--value');
      }

      function updateTripPlaceholder() {
        if (!queryInput) return;
        if (state.tripType === 'via-stopover') {
          queryInput.placeholder = 'E.g. Flight Delhi to Goa via Mumbai tomorrow...';
          return;
        }
        if (state.tripType === 'round-trip') {
          queryInput.placeholder = 'E.g. Round-trip Delhi to Mumbai returning in 3 days...';
          return;
        }
        queryInput.placeholder = 'Find cheap flights Delhi to Mumbai tomorrow...';
      }

      function setupSuggestions() {
        const chips = Array.from(document.querySelectorAll('#reviewSuggestions .history-chip'));
        chips.forEach((chip) => {
          chip.addEventListener('click', () => {
            if (!queryInput) return;
            const text = chip.getAttribute('title') || chip.textContent || '';
            queryInput.value = text.trim();
            queryInput.scrollTop = queryInput.scrollHeight;
            state.query = queryInput.value;
            autoResizeTextarea();
          });
        });
      }

      function setupTicker() {
        const track = document.getElementById('reviewTickerTrack');
        const groupA = document.getElementById('reviewTickerA');
        if (!track || !groupA) return;
        const update = () => {
          const width = groupA.getBoundingClientRect().width;
          if (!width) return;
          const speed = 40;
          const duration = Math.max(width / speed, 8);
          track.style.setProperty('--ticker-loop-distance', width + 'px');
          track.style.setProperty('--ticker-length', duration + 's');
        };
        update();
        if (document.fonts && document.fonts.ready) {
          document.fonts.ready.then(update).catch(() => undefined);
        }
        window.addEventListener('resize', update);
      }

      function drawAurora(canvas) {
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const W = canvas.width;
        const H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        const cx = W / 2;
        const cy = 0.5 * H + (W * W) / (2.4 * H);
        const R = cy - 0.35 * H;
        const REF = 1200;
        const sc = Math.max(R / REF, 0.25);
        const sinH = Math.min((W * 0.55) / R, 0.9998);
        const halfA = Math.asin(sinH);
        const sA = -Math.PI / 2 - halfA;
        const eA = -Math.PI / 2 + halfA;
        const RAW = [
          [420, 340, 'rgba(4,8,42,0.45)', 75],
          [330, 265, 'rgba(7,18,78,0.54)', 60],
          [248, 195, 'rgba(11,44,118,0.62)', 46],
          [175, 138, 'rgba(17,78,156,0.69)', 35],
          [112, 90, 'rgba(23,118,188,0.75)', 26],
          [60, 54, 'rgba(32,155,208,0.81)', 18],
          [22, 28, 'rgba(52,188,222,0.86)', 11],
          [7, 14, 'rgba(110,212,232,0.91)', 6],
          [2, 7, 'rgba(175,230,240,0.95)', 3],
          [0, 8, 'rgba(248,245,214,0.9)', 2],
          [-7, 13, 'rgba(255,238,170,0.97)', 4],
          [-22, 26, 'rgba(255,208,68,0.93)', 10],
          [-46, 40, 'rgba(255,160,22,0.90)', 17],
          [-80, 55, 'rgba(252,104,8,0.87)', 24],
          [-124, 70, 'rgba(222,56,7,0.83)', 32],
          [-176, 86, 'rgba(170,26,6,0.78)', 41],
          [-238, 104, 'rgba(110,12,4,0.70)', 50],
          [-308, 118, 'rgba(58,6,3,0.58)', 58],
          [-388, 128, 'rgba(25,3,2,0.42)', 66],
        ];

        ctx.save();
        ctx.globalCompositeOperation = 'screen';
        RAW.forEach(([rOff, lw, col, blur]) => {
          const r = R + rOff * sc;
          if (r < 1) return;
          ctx.save();
          ctx.filter = 'blur(' + String(Math.max(blur * Math.pow(sc, 0.55), 0.5).toFixed(1)) + 'px)';
          ctx.lineWidth = Math.max(lw * sc, 1.5);
          ctx.strokeStyle = col;
          ctx.lineCap = 'butt';
          ctx.beginPath();
          ctx.arc(cx, cy, r, sA, eA);
          ctx.stroke();
          ctx.restore();
        });
        ctx.restore();

        const pSolid = Math.max(R - 460 * sc, 10);
        const pFade = Math.max(R - 255 * sc, 20);
        if (pFade > pSolid) {
          const pGrad = ctx.createRadialGradient(cx, cy, pSolid, cx, cy, pFade);
          pGrad.addColorStop(0, 'rgba(7,6,15,1)');
          pGrad.addColorStop(0.6, 'rgba(7,6,15,0.85)');
          pGrad.addColorStop(1, 'rgba(7,6,15,0)');
          ctx.beginPath();
          ctx.arc(cx, cy, pFade, 0, Math.PI * 2);
          ctx.fillStyle = pGrad;
          ctx.fill();
        }

        const BG0 = 'rgba(7,6,15,0)';
        const BG1 = 'rgba(7,6,15,0.72)';
        const edgeWidth = W * 0.18;

        const lG = ctx.createLinearGradient(0, 0, edgeWidth, 0);
        lG.addColorStop(0, BG1);
        lG.addColorStop(0.22, BG1);
        lG.addColorStop(1, BG0);
        ctx.fillStyle = lG;
        ctx.fillRect(0, 0, edgeWidth, H);

        const rG = ctx.createLinearGradient(W - edgeWidth, 0, W, 0);
        rG.addColorStop(0, BG0);
        rG.addColorStop(0.78, BG1);
        rG.addColorStop(1, BG1);
        ctx.fillStyle = rG;
        ctx.fillRect(W - edgeWidth, 0, edgeWidth, H);

        const bG = ctx.createLinearGradient(0, H * 0.68, 0, H * 0.95);
        bG.addColorStop(0, BG0);
        bG.addColorStop(1, BG1);
        ctx.fillStyle = bG;
        ctx.fillRect(0, H * 0.68, W, H);

        const tG = ctx.createLinearGradient(0, 0, 0, H * 0.06);
        tG.addColorStop(0, BG1);
        tG.addColorStop(1, BG0);
        ctx.fillStyle = tG;
        ctx.fillRect(0, 0, W, H * 0.06);
      }

      function initAuroraCanvas() {
        const canvas = document.getElementById('reviewAurora');
        if (!canvas) return;
        if (canvas.dataset.auroraInit === '1') return;
        canvas.dataset.auroraInit = '1';
        let resizeTimer = null;
        let scrollRaf = 0;
        let rafHandle = null;
        const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

        const resize = () => {
          const dpr = Math.min(window.devicePixelRatio || 1, 2);
          canvas.width = Math.floor(window.innerWidth * dpr);
          canvas.height = Math.floor(window.innerHeight * dpr);
          canvas.style.width = String(window.innerWidth) + 'px';
          canvas.style.height = String(window.innerHeight) + 'px';
          drawAurora(canvas);
        };

        const onResize = () => {
          if (resizeTimer) clearTimeout(resizeTimer);
          resizeTimer = setTimeout(resize, 80);
        };

        const onScroll = () => {
          if (prefersReducedMotion) return;
          if (scrollRaf) return;
          scrollRaf = window.requestAnimationFrame(() => {
            const shift = Math.max(Math.min(window.scrollY * 0.04, 14), 0);
            canvas.style.transform = 'translate3d(0, ' + String(shift) + 'px, 0)';
            scrollRaf = 0;
          });
        };

        const onVisibilityChange = () => {
          if (document.visibilityState === 'visible') {
            resize();
          }
        };

        const onPageShow = () => {
          resize();
        };

        resize();
        rafHandle = window.requestAnimationFrame(() => resize());

        repaintAurora = resize;
        window.addEventListener('resize', onResize);
        window.addEventListener('scroll', onScroll, { passive: true });
        document.addEventListener('visibilitychange', onVisibilityChange);
        window.addEventListener('pageshow', onPageShow);
        window.addEventListener('beforeunload', () => {
          if (rafHandle) window.cancelAnimationFrame(rafHandle);
          if (scrollRaf) window.cancelAnimationFrame(scrollRaf);
          document.removeEventListener('visibilitychange', onVisibilityChange);
          window.removeEventListener('pageshow', onPageShow);
        });
      }

      if (plannerForm) {
        plannerForm.addEventListener('submit', (event) => {
          event.preventDefault();
          startDemo();
        });
      }

      if (queryInput) {
        queryInput.addEventListener('input', () => {
          state.query = queryInput.value;
          autoResizeTextarea();
        });
      }
      if (originInput) originInput.addEventListener('input', () => {
        state.origin = originInput.value;
      });
      if (destinationInput) destinationInput.addEventListener('input', () => {
        state.destination = destinationInput.value;
      });
      if (dateInput) dateInput.addEventListener('input', () => {
        state.date = dateInput.value;
        updateDateDisplay();
      });

      document.querySelectorAll('.trip-tab').forEach((button) => {
        button.addEventListener('click', () => {
          const trip = button.getAttribute('data-trip');
          state.tripType = trip || 'one-way';
          document.querySelectorAll('.trip-tab').forEach((tab) => tab.classList.remove('active'));
          button.classList.add('active');
          updateTripPlaceholder();
        });
      });

      if (resetButton) resetButton.addEventListener('click', resetDemo);
      setupSuggestions();
      setupTicker();
      initAuroraCanvas();
      updateTripPlaceholder();
      updateDateDisplay();
      autoResizeTextarea();
      resetDemo();
    })();
  </script>
</body>
</html>`;

mkdirSync(dirname(outputPath), { recursive: true });
writeFileSync(outputPath, html, "utf8");
console.log(`Wrote ${outputPath}`);
