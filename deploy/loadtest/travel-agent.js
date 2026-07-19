import http from "k6/http";
import { check, sleep } from "k6";
import { Rate, Trend } from "k6/metrics";
import { textSummary } from "https://jslib.k6.io/k6-summary/0.1.0/index.js";

const healthErrorRate = new Rate("health_errors");
const askErrorRate = new Rate("ask_errors");
const askDuration = new Trend("ask_duration", true);

const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";
const AUTH_TOKEN = __ENV.AUTH_TOKEN || "";

const headers = { "Content-Type": "application/json" };
if (AUTH_TOKEN) {
  headers["Authorization"] = `Bearer ${AUTH_TOKEN}`;
}

const scenarios = {
  ramping_load: {
    executor: "ramping-vus",
    startVUs: 0,
    stages: [
      { duration: "30s", target: 10 },
      { duration: "60s", target: 10 },
      { duration: "30s", target: 0 },
    ],
    gracefulRampDown: "10s",
  },
};

export const options = {
  scenarios,
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(95)<3000"],
    checks: ["rate>0.99"],
  },
};

const airportCodes = ["DEL", "BOM", "BLR", "MAA", "HYD", "CCU", "GOI", "PNQ", "LKO", "AMD"];
const tripTypes = ["Business", "Holiday", "one-way", "round-trip"];

function randomAirport() {
  return airportCodes[Math.floor(Math.random() * airportCodes.length)];
}

function randomTripType() {
  return tripTypes[Math.floor(Math.random() * tripTypes.length)];
}

function buildAskPayload() {
  const origin = randomAirport();
  let dest = randomAirport();
  while (dest === origin) dest = randomAirport();

  const today = new Date();
  const future = new Date(today);
  future.setDate(today.getDate() + 14 + Math.floor(Math.random() * 30));
  const dateStr = future.toISOString().slice(0, 10);

  return JSON.stringify({
    origin,
    destination: dest,
    date: dateStr,
    user_query: `${origin} to ${dest} with least travel time`,
    trip_type: randomTripType(),
  });
}

export default function () {
  const isAsk = Math.random() < 0.1;

  if (isAsk) {
    const payload = buildAskPayload();
    const res = http.post(`${BASE_URL}/ask`, payload, { headers, timeout: "60s" });
    askDuration.add(res.timings.duration);
    check(res, {
      "ask response ok": (r) => r.status === 200 || r.status === 202,
    }) || askErrorRate.add(1);
  } else {
    const res = http.get(`${BASE_URL}/health`, { headers, timeout: "5s" });
    check(res, {
      "health ok": (r) => r.status === 200,
    }) || healthErrorRate.add(1);
  }

  sleep(1);
}

export function handleSummary(data) {
  return {
    stdout: textSummary(data, { indent: " ", enableColors: true }),
  };
}
