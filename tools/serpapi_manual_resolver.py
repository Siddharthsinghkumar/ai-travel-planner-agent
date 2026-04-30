#!/usr/bin/env python3
"""
SerpApi manual booking resolver helper.

Implements the proven flow:
1) search (google_flights)
2) pick itinerary from best_flights/other_flights
3) use itinerary-level booking_token
4) fetch booking options on google_flights with booking_token + route context
5) read together.booking_request (url + post_data)
6) POST to booking_request.url
7) parse meta-refresh provider URL
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import html
import json
import re
import sys
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import httpx
from core.api_key_manager import key_manager


SERPAPI_ENDPOINT = "https://serpapi.com/search"
META_TAG_RE = re.compile(r"<meta[^>]*http-equiv\s*=\s*['\"]?\s*refresh\s*['\"]?[^>]*>", re.IGNORECASE)
TITLE_RE = re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def _pick_serpapi_key(cli_key: Optional[str]) -> Optional[str]:
    if cli_key and cli_key.strip():
        print(
            "warning: --api-key override is ignored; using key manager managed SerpAPI pool only.",
            file=sys.stderr,
        )

    async def _resolve() -> Optional[str]:
        with contextlib.suppress(Exception):
            await key_manager.load_env_keys()
        try:
            async with key_manager.reserve_key("serpapi") as (_idx, key):
                return key
        except Exception:
            return None

    return asyncio.run(_resolve())


def _canonical_url(value: Optional[str]) -> Optional[str]:
    if not value or not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    if candidate.startswith("//"):
        candidate = f"https:{candidate}"
    parsed = urllib.parse.urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return candidate


def _collect_itineraries(search_payload: Dict[str, Any]) -> List[Tuple[str, int, Dict[str, Any]]]:
    rows: List[Tuple[str, int, Dict[str, Any]]] = []
    for bucket in ("best_flights", "other_flights"):
        data = search_payload.get(bucket)
        if not isinstance(data, list):
            continue
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                rows.append((bucket, idx, item))
    return rows


def _extract_airline_code(itinerary: Dict[str, Any]) -> Optional[str]:
    flights = itinerary.get("flights")
    if not isinstance(flights, list) or not flights:
        return None
    first = flights[0] if isinstance(flights[0], dict) else {}
    flight_no = str(first.get("flight_number") or "").upper()
    match = re.search(r"\b([A-Z0-9]{2})\s*\d", flight_no)
    if match:
        return match.group(1)
    return None


def _extract_booking_request(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    queue: List[Any] = [payload]
    while queue:
        node = queue.pop(0)
        if isinstance(node, dict):
            direct = node.get("booking_request")
            if isinstance(direct, dict) and direct.get("url"):
                return direct
            together = node.get("together")
            if isinstance(together, dict):
                nested = together.get("booking_request")
                if isinstance(nested, dict) and nested.get("url"):
                    return nested
            for value in node.values():
                if isinstance(value, (dict, list)):
                    queue.append(value)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    queue.append(item)
    return None


def _extract_meta_refresh_url(html_text: str, *, base_url: str) -> Optional[str]:
    for tag_match in META_TAG_RE.finditer(html_text[:12000]):
        tag = str(tag_match.group(0) or "")
        content_match = re.search(r"content\s*=\s*(['\"])(.*?)\1", tag, re.IGNORECASE | re.DOTALL)
        if content_match:
            content_value = str(content_match.group(2) or "")
        else:
            fallback = re.search(r"content\s*=\s*([^>]+)", tag, re.IGNORECASE)
            content_value = str(fallback.group(1) or "") if fallback else ""
        content_url_match = re.search(r"url\s*=\s*", content_value, re.IGNORECASE)
        if not content_url_match:
            continue
        candidate_tail = str(content_value[content_url_match.end():] or "").strip()
        if not candidate_tail:
            continue
        if candidate_tail[0] in {"'", '"'}:
            quote = candidate_tail[0]
            closing = candidate_tail.find(quote, 1)
            raw = candidate_tail[1:closing] if closing > 0 else candidate_tail[1:]
        else:
            raw = candidate_tail.split()[0]
        raw = html.unescape(str(raw or "").strip().strip("'\""))
        absolute = urllib.parse.urljoin(base_url, raw)
        canonical = _canonical_url(absolute)
        if canonical:
            return canonical
    return None


def _extract_title(html_text: str) -> Optional[str]:
    match = TITLE_RE.search(html_text or "")
    if not match:
        return None
    return str(match.group(1) or "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve SerpApi booking_request POST into provider URL.")
    parser.add_argument("--origin", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD outbound date")
    parser.add_argument("--return-date", default=None)
    parser.add_argument("--trip-type", choices=["one-way", "round-trip"], default="one-way")
    parser.add_argument("--currency", default="INR")
    parser.add_argument("--hl", default="en")
    parser.add_argument("--gl", default="in")
    parser.add_argument("--deep-search", action="store_true")
    parser.add_argument("--itinerary-source", choices=["best_flights", "other_flights"], default="best_flights")
    parser.add_argument("--itinerary-index", type=int, default=0)
    parser.add_argument("--airline-contains", default=None)
    parser.add_argument("--include-airlines", default=None, help="Two-char airline code hint, e.g. 6E or AI")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    api_key = _pick_serpapi_key(args.api_key)
    if not api_key:
        print("error: no usable SerpApi key available through key manager.", file=sys.stderr)
        return 2

    search_params: Dict[str, Any] = {
        "engine": "google_flights",
        "departure_id": args.origin,
        "arrival_id": args.destination,
        "outbound_date": args.date,
        "type": "1" if (args.trip_type == "round-trip" or args.return_date) else "2",
        "currency": args.currency,
        "hl": args.hl,
        "gl": args.gl,
        "api_key": api_key,
    }
    if args.return_date:
        search_params["return_date"] = args.return_date
    if args.deep_search:
        search_params["deep_search"] = "true"

    with httpx.Client(timeout=25.0) as client:
        search_resp = client.get(SERPAPI_ENDPOINT, params=search_params)
        search_resp.raise_for_status()
        search_payload = search_resp.json()

        itineraries = _collect_itineraries(search_payload)
        if not itineraries:
            print(json.dumps({"status": "unavailable", "reason": "no_itineraries"}))
            return 1

        selected: Optional[Tuple[str, int, Dict[str, Any]]] = None
        for source, idx, row in itineraries:
            if source != args.itinerary_source:
                continue
            if idx != args.itinerary_index:
                continue
            if args.airline_contains:
                flights = row.get("flights")
                first = flights[0] if isinstance(flights, list) and flights and isinstance(flights[0], dict) else {}
                airline = str(first.get("airline") or "")
                if args.airline_contains.lower() not in airline.lower():
                    continue
            selected = (source, idx, row)
            break
        if selected is None:
            print(json.dumps({"status": "unavailable", "reason": "selected_itinerary_not_found"}))
            return 1

        source, index, itinerary = selected
        itinerary_token = str(itinerary.get("booking_token") or "").strip()
        if not itinerary_token:
            print(json.dumps({"status": "unavailable", "reason": "itinerary_booking_token_missing"}))
            return 1

        include_airlines = args.include_airlines or _extract_airline_code(itinerary)
        booking_params: Dict[str, Any] = {
            "engine": "google_flights",
            "booking_token": itinerary_token,
            "departure_id": args.origin,
            "arrival_id": args.destination,
            "outbound_date": args.date,
            "type": "1" if (args.trip_type == "round-trip" or args.return_date) else "2",
            "currency": args.currency,
            "hl": args.hl,
            "gl": args.gl,
            "api_key": api_key,
        }
        if args.return_date:
            booking_params["return_date"] = args.return_date
        if include_airlines:
            booking_params["include_airlines"] = include_airlines
        if args.deep_search:
            booking_params["deep_search"] = "true"

        booking_resp = client.get(SERPAPI_ENDPOINT, params=booking_params)
        booking_resp.raise_for_status()
        booking_payload = booking_resp.json()

        booking_request = _extract_booking_request(booking_payload)
        if not booking_request:
            print(
                json.dumps(
                    {
                        "status": "unavailable",
                        "reason": "booking_request_missing",
                        "selected_itinerary": {"source": source, "index": index},
                    }
                )
            )
            return 1

        request_url = _canonical_url(str(booking_request.get("url") or "").strip())
        post_data = booking_request.get("post_data")
        if not request_url or post_data in (None, "", {}, []):
            print(
                json.dumps(
                    {
                        "status": "unavailable",
                        "reason": "booking_request_incomplete",
                        "booking_request_has_url": bool(request_url),
                        "booking_request_has_post_data": post_data not in (None, "", {}, []),
                    }
                )
            )
            return 1

        if isinstance(post_data, str):
            post_kwargs: Dict[str, Any] = {"content": post_data}
        elif isinstance(post_data, (dict, list, tuple)):
            post_kwargs = {"data": post_data}
        else:
            post_kwargs = {"content": str(post_data)}

        resolver_resp = client.post(
            request_url,
            follow_redirects=True,
            timeout=25.0,
            **post_kwargs,
        )
        final_url = _canonical_url(str(resolver_resp.url))
        html_text = resolver_resp.text or ""
        meta_refresh_url = _extract_meta_refresh_url(html_text, base_url=final_url or request_url)
        title = _extract_title(html_text)
        response_path = "/tmp/serpapi_manual_resolver_last_response.html"
        with open(response_path, "w", encoding="utf-8") as f:
            f.write(html_text)

        result = {
            "status": "ok" if meta_refresh_url else "unavailable",
            "selected_itinerary": {"source": source, "index": index},
            "booking_token_present": bool(itinerary_token),
            "include_airlines": include_airlines,
            "booking_request_url": request_url,
            "resolver_http_status": resolver_resp.status_code,
            "resolver_final_url": final_url,
            "resolver_meta_refresh_url": meta_refresh_url,
            "resolver_title": title,
            "response_bytes": len(html_text.encode("utf-8")),
            "response_saved_to": response_path,
            "response_head_1200": html_text[:1200],
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if meta_refresh_url else 1


if __name__ == "__main__":
    raise SystemExit(main())
