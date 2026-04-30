#!/usr/bin/env python3
"""Generate frontend-consumed contract artifacts from FastAPI OpenAPI.

This is the authoritative backend->frontend contract sync pipeline for dev/operator
surfaces currently consumed by frontend code:
- GET /llm/options
- GET /version
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Set

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TS_OUT = ROOT / "frontend" / "src" / "lib" / "generated" / "contracts.ts"
JSON_OUT = ROOT / "frontend" / "src" / "lib" / "generated" / "openapi.frontend-contract.json"

TARGET_ENDPOINTS = {
    "/llm/options": ("get", "LLMOptionsResponse"),
    "/version": ("get", "ServerVersionMeta"),
}


def _load_openapi() -> Dict[str, Any]:
    from api.app import app

    return app.openapi()


def _ref_name(ref: str) -> str:
    return ref.rsplit("/", 1)[-1]


def _collect_refs(schema: Any, out: Set[str]) -> None:
    if isinstance(schema, dict):
        ref = schema.get("$ref")
        if isinstance(ref, str):
            out.add(_ref_name(ref))
        for value in schema.values():
            _collect_refs(value, out)
    elif isinstance(schema, list):
        for value in schema:
            _collect_refs(value, out)


def _is_identifier(name: str) -> bool:
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name))


def _format_prop(name: str) -> str:
    return name if _is_identifier(name) else json.dumps(name)


def _render_union(items: Iterable[str]) -> str:
    uniq: list[str] = []
    for item in items:
        if item not in uniq:
            uniq.append(item)
    return " | ".join(uniq) if uniq else "unknown"


def _render_ts_type(schema: Dict[str, Any], components: Dict[str, Any], indent: int = 0) -> str:
    if not schema:
        return "unknown"

    ref = schema.get("$ref")
    if isinstance(ref, str):
        return _ref_name(ref)

    if "enum" in schema and isinstance(schema["enum"], list):
        return _render_union(json.dumps(v) for v in schema["enum"])

    if "anyOf" in schema and isinstance(schema["anyOf"], list):
        variants = schema["anyOf"]
        non_null = [v for v in variants if not (isinstance(v, dict) and v.get("type") == "null")]
        rendered = [_render_ts_type(v, components, indent) for v in non_null]
        if len(non_null) != len(variants):
            rendered.append("null")
        return _render_union(rendered)

    if "oneOf" in schema and isinstance(schema["oneOf"], list):
        return _render_union(_render_ts_type(v, components, indent) for v in schema["oneOf"])

    stype = schema.get("type")

    if stype == "array":
        items = schema.get("items") or {}
        item_t = _render_ts_type(items, components, indent)
        if "\n" in item_t or " | " in item_t:
            return f"({item_t})[]"
        return f"{item_t}[]"

    if stype == "object" or "properties" in schema or "additionalProperties" in schema:
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        additional = schema.get("additionalProperties", None)
        pad = " " * indent
        inner = " " * (indent + 2)
        lines = ["{"]

        for pname in sorted(props.keys()):
            pschema = props[pname]
            ptype = _render_ts_type(pschema, components, indent + 2)
            optional = "" if pname in required else "?"
            if "\n" in ptype:
                ptype = f"({ptype})"
            lines.append(f"{inner}{_format_prop(pname)}{optional}: {ptype};")

        if additional is True:
            lines.append(f"{inner}[key: string]: unknown;")
        elif isinstance(additional, dict):
            atype = _render_ts_type(additional, components, indent + 2)
            if "\n" in atype:
                atype = f"({atype})"
            lines.append(f"{inner}[key: string]: {atype};")

        lines.append(f"{pad}}}")
        return "\n".join(lines)

    if stype == "string":
        return "string"
    if stype in {"integer", "number"}:
        return "number"
    if stype == "boolean":
        return "boolean"

    nullable = bool(schema.get("nullable"))
    base = "unknown"
    return f"{base} | null" if nullable else base


def build_artifacts() -> Dict[str, str]:
    openapi = _load_openapi()
    components = ((openapi.get("components") or {}).get("schemas") or {})

    endpoint_schemas: Dict[str, Dict[str, Any]] = {}
    needed_components: Set[str] = set()

    for path, (method, type_name) in TARGET_ENDPOINTS.items():
        try:
            schema = (
                openapi["paths"][path][method]["responses"]["200"]["content"]["application/json"]["schema"]
            )
        except KeyError as exc:
            raise RuntimeError(f"Missing OpenAPI schema for {method.upper()} {path}") from exc

        endpoint_schemas[type_name] = schema
        _collect_refs(schema, needed_components)

    queue = list(needed_components)
    seen = set(needed_components)
    while queue:
        name = queue.pop()
        schema = components.get(name)
        if not isinstance(schema, dict):
            continue
        refs: Set[str] = set()
        _collect_refs(schema, refs)
        for ref_name in refs:
            if ref_name not in seen:
                seen.add(ref_name)
                queue.append(ref_name)

    component_text_blocks = []
    for name in sorted(seen):
        schema = components.get(name)
        if not isinstance(schema, dict):
            continue
        rendered = _render_ts_type(schema, components, indent=0)
        component_text_blocks.append(f"export type {name} = {rendered};")

    endpoint_aliases = []
    for type_name in sorted(endpoint_schemas.keys()):
        schema = endpoint_schemas[type_name]
        rendered = _render_ts_type(schema, components, indent=0)
        endpoint_aliases.append(f"export type {type_name} = {rendered};")

    ts_lines = [
        "// AUTO-GENERATED by scripts/sync_frontend_contract.py",
        "// Source of truth: FastAPI OpenAPI schema (api/app.py response models).",
        "// Do not edit manually.",
        "",
    ]
    if component_text_blocks:
        ts_lines.extend(component_text_blocks)
        ts_lines.append("")
    ts_lines.extend(endpoint_aliases)
    ts_lines.append("")

    contract_payload = {
        "source": "fastapi_openapi",
        "endpoints": {
            path: {
                "method": method.upper(),
                "type_name": type_name,
                "schema": endpoint_schemas[type_name],
            }
            for path, (method, type_name) in TARGET_ENDPOINTS.items()
        },
        "components": {name: components[name] for name in sorted(seen) if name in components},
    }

    return {
        "ts": "\n".join(ts_lines),
        "json": json.dumps(contract_payload, ensure_ascii=False, indent=2) + "\n",
    }


def _ensure_written(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync frontend contract artifacts from backend OpenAPI")
    parser.add_argument("--check", action="store_true", help="Fail if generated artifacts are out of date")
    args = parser.parse_args()

    artifacts = build_artifacts()

    if args.check:
        current_ts = TS_OUT.read_text(encoding="utf-8") if TS_OUT.exists() else ""
        current_json = JSON_OUT.read_text(encoding="utf-8") if JSON_OUT.exists() else ""
        if current_ts != artifacts["ts"] or current_json != artifacts["json"]:
            print("Frontend contract artifacts are out of date. Run scripts/sync_frontend_contract.py")
            return 1
        print("Frontend contract artifacts are up to date.")
        return 0

    _ensure_written(TS_OUT, artifacts["ts"])
    _ensure_written(JSON_OUT, artifacts["json"])
    print(f"Wrote {TS_OUT}")
    print(f"Wrote {JSON_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
