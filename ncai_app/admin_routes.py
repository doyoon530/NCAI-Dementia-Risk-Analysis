import csv
import io
import uuid
from datetime import datetime, timedelta

from flask import jsonify, make_response, redirect, render_template, request

from . import runtime
from .common import (
    build_device_name,
    extract_client_ip_info,
    infer_browser,
    infer_device_type,
    infer_operating_system,
    normalize_text,
    safe_reverse_dns,
)
from .config import get_google_login_status
from .llm_service import (
    get_google_credentials_status,
    get_llm_provider_status,
    get_model_status,
)

visitor_event_store = runtime.visitor_event_store
visitor_snapshot_store = runtime.visitor_snapshot_store
visitor_hostname_cache = runtime.visitor_hostname_cache
visitor_lock = runtime.visitor_lock


def register_admin_routes(app, *, require_admin_access):
    def should_track_request(path: str) -> bool:
        if not path:
            return False
        if path.startswith("/static/"):
            return False
        if path == "/favicon.ico":
            return False
        return True

    def resolve_hostname(ip_address: str) -> str:
        ip = normalize_text(ip_address)
        if not ip or ip == "unknown":
            return ""

        with visitor_lock:
            cached = visitor_hostname_cache.get(ip)
        if cached is not None:
            return cached

        hostname = safe_reverse_dns(ip)
        with visitor_lock:
            if len(visitor_hostname_cache) >= runtime._HOSTNAME_CACHE_MAX:
                try:
                    visitor_hostname_cache.pop(next(iter(visitor_hostname_cache)))
                except StopIteration:
                    pass
            visitor_hostname_cache[ip] = hostname
        return hostname

    def normalize_client_telemetry(raw_payload: dict | None) -> dict:
        payload = raw_payload or {}
        browser_brands = payload.get("brands") or []

        if isinstance(browser_brands, list):
            brands = [
                normalize_text(item.get("brand", ""))
                for item in browser_brands
                if isinstance(item, dict)
            ]
        else:
            brands = []

        return {
            "platform": normalize_text(payload.get("platform", "")),
            "platform_version": normalize_text(payload.get("platformVersion", "")),
            "model": normalize_text(payload.get("model", "")),
            "language": normalize_text(payload.get("language", "")),
            "languages": [
                normalize_text(item)
                for item in (payload.get("languages") or [])
                if normalize_text(item)
            ],
            "timezone": normalize_text(payload.get("timezone", "")),
            "screen": normalize_text(payload.get("screen", "")),
            "viewport": normalize_text(payload.get("viewport", "")),
            "device_memory": payload.get("deviceMemory"),
            "hardware_concurrency": payload.get("hardwareConcurrency"),
            "max_touch_points": payload.get("maxTouchPoints"),
            "is_mobile": bool(payload.get("isMobile")),
            "connection_type": normalize_text(payload.get("connectionType", "")),
            "effective_type": normalize_text(payload.get("effectiveType", "")),
            "referrer": normalize_text(payload.get("referrer", "")),
            "page_url": normalize_text(payload.get("pageUrl", "")),
            "user_agent": normalize_text(payload.get("userAgent", "")),
            "brands": [brand for brand in brands if brand],
        }

    def get_request_session_id() -> str:
        query_session_id = normalize_text(request.args.get("session_id", ""))
        body = request.get_json(silent=True) or {}
        body_session_id = normalize_text(body.get("session_id", ""))
        return query_session_id or body_session_id

    def build_request_visitor_context() -> dict:
        ip_info = extract_client_ip_info(request)
        user_agent = normalize_text(request.headers.get("User-Agent", ""))
        visitor_id = normalize_text(request.headers.get("X-Visitor-Id", ""))
        session_id = get_request_session_id()
        snapshot_key = visitor_id or uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{ip_info['ip']}|{user_agent[:120]}",
        ).hex[:16]

        with visitor_lock:
            snapshot = dict(visitor_snapshot_store.get(snapshot_key, {}))
            telemetry = dict(snapshot.get("telemetry", {}))

        browser = infer_browser(user_agent or telemetry.get("user_agent", ""))
        operating_system = infer_operating_system(
            user_agent or telemetry.get("user_agent", ""),
            platform_hint=telemetry.get("platform", ""),
        )
        device_type = infer_device_type(
            user_agent or telemetry.get("user_agent", ""),
            is_mobile_hint=telemetry.get("is_mobile"),
            max_touch_points=telemetry.get("max_touch_points"),
        )
        hostname = resolve_hostname(ip_info["ip"])
        model = normalize_text(telemetry.get("model", ""))
        device_name = build_device_name(
            browser=browser,
            operating_system=operating_system,
            hostname=hostname,
            model=model,
        )

        return {
            "visitor_id": snapshot_key,
            "session_id": session_id or snapshot.get("session_id", ""),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "path": request.path,
            "method": request.method,
            "ip": ip_info["ip"],
            "ip_source": ip_info["source"],
            "remote_addr": ip_info["remote_addr"],
            "forwarded_chain": ip_info["forwarded_chain"],
            "hostname": hostname,
            "browser": browser,
            "operating_system": operating_system,
            "device_type": device_type,
            "device_name": device_name,
            "user_agent": user_agent,
            "cf_ip_country": normalize_text(request.headers.get("CF-IPCountry", "")),
            "cf_ray": normalize_text(request.headers.get("CF-Ray", "")),
            "telemetry": telemetry,
            "referrer": telemetry.get("referrer", ""),
            "page_url": telemetry.get("page_url", ""),
            "language": telemetry.get("language", ""),
            "screen": telemetry.get("screen", ""),
            "viewport": telemetry.get("viewport", ""),
            "timezone": telemetry.get("timezone", ""),
            "connection_type": telemetry.get("connection_type", ""),
            "effective_type": telemetry.get("effective_type", ""),
        }

    def record_visitor_event(context: dict) -> None:
        snapshot_key = context["visitor_id"]
        timestamp = context["timestamp"]
        path = context["path"]

        with visitor_lock:
            existing = dict(visitor_snapshot_store.get(snapshot_key, {}))
            path_history = list(existing.get("recent_paths", []))
            path_history.append(path)
            path_history = path_history[-8:]
            visit_count = int(existing.get("visit_count", 0)) + 1

            snapshot = {
                **existing,
                "visitor_id": snapshot_key,
                "session_id": context.get("session_id", "") or existing.get("session_id", ""),
                "first_seen": existing.get("first_seen", timestamp),
                "last_seen": timestamp,
                "visit_count": visit_count,
                "last_path": path,
                "recent_paths": path_history,
                "ip": context["ip"],
                "ip_source": context["ip_source"],
                "remote_addr": context["remote_addr"],
                "forwarded_chain": context["forwarded_chain"],
                "hostname": context["hostname"],
                "browser": context["browser"],
                "operating_system": context["operating_system"],
                "device_type": context["device_type"],
                "device_name": context["device_name"],
                "user_agent": context["user_agent"],
                "cf_ip_country": context["cf_ip_country"],
                "cf_ray": context["cf_ray"],
                "telemetry": context.get("telemetry", existing.get("telemetry", {})),
                "referrer": context.get("referrer", existing.get("referrer", "")),
                "page_url": context.get("page_url", existing.get("page_url", "")),
                "language": context.get("language", existing.get("language", "")),
                "screen": context.get("screen", existing.get("screen", "")),
                "viewport": context.get("viewport", existing.get("viewport", "")),
                "timezone": context.get("timezone", existing.get("timezone", "")),
                "connection_type": context.get(
                    "connection_type", existing.get("connection_type", "")
                ),
                "effective_type": context.get(
                    "effective_type", existing.get("effective_type", "")
                ),
            }
            visitor_snapshot_store[snapshot_key] = snapshot
            visitor_event_store.append(
                {
                    "timestamp": timestamp,
                    "visitor_id": snapshot_key,
                    "session_id": snapshot["session_id"],
                    "method": context["method"],
                    "path": path,
                    "ip": context["ip"],
                    "ip_source": context["ip_source"],
                    "hostname": context["hostname"],
                    "browser": context["browser"],
                    "operating_system": context["operating_system"],
                    "device_type": context["device_type"],
                    "device_name": context["device_name"],
                    "language": context.get("language", ""),
                    "screen": context.get("screen", ""),
                    "viewport": context.get("viewport", ""),
                    "timezone": context.get("timezone", ""),
                    "connection_type": context.get("connection_type", ""),
                    "effective_type": context.get("effective_type", ""),
                    "referrer": context.get("referrer", ""),
                    "page_url": context.get("page_url", ""),
                    "cf_ip_country": context["cf_ip_country"],
                    "cf_ray": context["cf_ray"],
                    "user_agent": context["user_agent"],
                }
            )

        app.logger.info(
            "VISITOR %s %s %s ip=%s source=%s device=%s browser=%s os=%s host=%s session=%s",
            timestamp,
            context["method"],
            path,
            context["ip"],
            context["ip_source"],
            context["device_name"],
            context["browser"],
            context["operating_system"],
            context["hostname"] or "-",
            context.get("session_id", "") or "-",
        )

    def is_visitor_online(last_seen: str, threshold_minutes: int = 3) -> bool:
        normalized_last_seen = normalize_text(last_seen)
        if not normalized_last_seen:
            return False

        try:
            last_seen_at = datetime.fromisoformat(normalized_last_seen)
        except ValueError:
            return False

        return last_seen_at >= datetime.now() - timedelta(minutes=threshold_minutes)

    def build_server_status_snapshot() -> dict:
        model_status = get_model_status()
        credentials_status = get_google_credentials_status()
        provider_status = get_llm_provider_status()
        google_login_status = get_google_login_status()
        llm_ready = provider_status["local"]["ready"] or provider_status["api"]["ready"]
        ready = llm_ready and credentials_status["configured"]

        active_visitor_count = 0
        with visitor_lock:
            for snapshot in visitor_snapshot_store.values():
                if is_visitor_online(snapshot.get("last_seen", "")):
                    active_visitor_count += 1

        return {
            "ready": ready,
            "status": "READY" if ready else "NOT READY",
            "default_provider": provider_status.get("default", "local"),
            "local_ready": bool(provider_status.get("local", {}).get("ready")),
            "api_ready": bool(provider_status.get("api", {}).get("ready")),
            "model_exists": bool(model_status.get("exists")),
            "google_credentials": bool(credentials_status.get("configured")),
            "google_login_enabled": bool(google_login_status.get("enabled")),
            "active_visitor_count": active_visitor_count,
            "tracked_sessions": len(runtime.conversation_store),
        }

    def build_visitors_csv_response(payload: dict):
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "visitor_id",
                "device_name",
                "ip",
                "hostname",
                "browser",
                "operating_system",
                "device_type",
                "session_id",
                "first_seen",
                "last_seen",
                "visit_count",
                "is_online",
                "page_url",
                "referrer",
                "recent_paths",
            ]
        )

        for visitor in payload.get("visitors", []):
            writer.writerow(
                [
                    visitor.get("visitor_id", ""),
                    visitor.get("device_name", ""),
                    visitor.get("ip", ""),
                    visitor.get("hostname", ""),
                    visitor.get("browser", ""),
                    visitor.get("operating_system", ""),
                    visitor.get("device_type", ""),
                    visitor.get("session_id", ""),
                    visitor.get("first_seen", ""),
                    visitor.get("last_seen", ""),
                    visitor.get("visit_count", 0),
                    "yes" if visitor.get("is_online") else "no",
                    visitor.get("page_url", ""),
                    visitor.get("referrer", ""),
                    " | ".join(visitor.get("recent_paths", []) or []),
                ]
            )

        response = make_response(buffer.getvalue())
        response.headers["Content-Type"] = "text/csv; charset=utf-8"
        response.headers["Content-Disposition"] = (
            'attachment; filename="dr-jinu-visitors.csv"'
        )
        return response

    def build_visitors_payload(limit: int) -> dict:
        with visitor_lock:
            visitors = sorted(
                visitor_snapshot_store.values(),
                key=lambda item: item.get("last_seen", ""),
                reverse=True,
            )
            recent_events = list(visitor_event_store)[-limit:][::-1]

        enriched_visitors = []
        for visitor in visitors[:limit]:
            visitor_id = visitor.get("visitor_id", "")
            visitor_events = [
                event
                for event in recent_events
                if event.get("visitor_id", "") == visitor_id
            ]
            if len(visitor_events) < 8:
                visitor_events = [
                    event
                    for event in reversed(visitor_event_store)
                    if event.get("visitor_id", "") == visitor_id
                ][:8]

            enriched_visitors.append(
                {
                    **visitor,
                    "is_online": is_visitor_online(visitor.get("last_seen", "")),
                    "recent_event_timeline": list(reversed(visitor_events)),
                }
            )

        active_visitor_count = sum(
            1 for visitor in enriched_visitors if visitor.get("is_online", False)
        )

        return {
            "status": "ok",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "total_known_visitors": len(visitors),
            "total_recent_events": len(recent_events),
            "active_visitor_count": active_visitor_count,
            "visitors": enriched_visitors,
            "recent_events": recent_events,
            "server_status": build_server_status_snapshot(),
        }

    @app.before_request
    def track_visitor_request():
        if not should_track_request(request.path):
            return None
        record_visitor_event(build_request_visitor_context())
        return None

    @app.route("/client-telemetry", methods=["POST"])
    def client_telemetry():
        payload = request.get_json(silent=True) or {}
        raw_visitor_id = normalize_text(
            request.headers.get("X-Visitor-Id", "") or payload.get("visitor_id", "")
        )
        if not raw_visitor_id:
            return jsonify({"error": "visitor_id가 없습니다."}), 400

        telemetry = normalize_client_telemetry(payload)
        ip_info = extract_client_ip_info(request)
        request_ip = ip_info["ip"]
        user_agent = telemetry.get("user_agent", "")
        normalized_ua = normalize_text(user_agent)
        ip_ua_key = (request_ip, normalized_ua)

        with visitor_lock:
            existing_id = runtime.visitor_ip_ua_index.get(ip_ua_key)
            merge_key = existing_id if existing_id and existing_id != raw_visitor_id else ""
            existing = dict(
                visitor_snapshot_store.get(raw_visitor_id, {})
                or visitor_snapshot_store.get(merge_key, {})
            )
            hostname = existing.get("hostname", "")
            browser = infer_browser(telemetry.get("user_agent", ""))
            operating_system = infer_operating_system(
                telemetry.get("user_agent", ""),
                platform_hint=telemetry.get("platform", ""),
            )
            device_type = infer_device_type(
                telemetry.get("user_agent", ""),
                is_mobile_hint=telemetry.get("is_mobile"),
                max_touch_points=telemetry.get("max_touch_points"),
            )
            device_name = build_device_name(
                browser=browser,
                operating_system=operating_system,
                hostname=hostname,
                model=telemetry.get("model", ""),
            )

            visitor_snapshot_store[raw_visitor_id] = {
                **existing,
                "visitor_id": raw_visitor_id,
                "session_id": normalize_text(payload.get("session_id", ""))
                or existing.get("session_id", ""),
                "ip": existing.get("ip", request_ip),
                "ip_source": existing.get("ip_source", ip_info["source"]),
                "remote_addr": existing.get("remote_addr", ip_info["remote_addr"]),
                "forwarded_chain": existing.get(
                    "forwarded_chain", ip_info["forwarded_chain"]
                ),
                "hostname": hostname,
                "telemetry": telemetry,
                "browser": browser
                if browser != "Unknown Browser"
                else existing.get("browser", browser),
                "operating_system": operating_system
                if operating_system != "Unknown OS"
                else existing.get("operating_system", operating_system),
                "device_type": device_type or existing.get("device_type", ""),
                "device_name": device_name or existing.get("device_name", ""),
                "language": telemetry.get("language", "")
                or existing.get("language", ""),
                "screen": telemetry.get("screen", "") or existing.get("screen", ""),
                "viewport": telemetry.get("viewport", "")
                or existing.get("viewport", ""),
                "timezone": telemetry.get("timezone", "")
                or existing.get("timezone", ""),
                "connection_type": telemetry.get("connection_type", "")
                or existing.get("connection_type", ""),
                "effective_type": telemetry.get("effective_type", "")
                or existing.get("effective_type", ""),
                "page_url": telemetry.get("page_url", "")
                or existing.get("page_url", ""),
                "referrer": telemetry.get("referrer", "")
                or existing.get("referrer", ""),
            }

            runtime.visitor_ip_ua_index[ip_ua_key] = raw_visitor_id

            if merge_key and merge_key in visitor_snapshot_store:
                old_snapshot = visitor_snapshot_store[merge_key]
                old_raw_ua = old_snapshot.get("telemetry", {}).get("user_agent", "")
                old_key = (old_snapshot.get("ip", ""), normalize_text(old_raw_ua))
                runtime.visitor_ip_ua_index.pop(old_key, None)
                visitor_snapshot_store.pop(merge_key, None)
                for event in list(visitor_event_store):
                    if event.get("visitor_id") == merge_key:
                        event["visitor_id"] = raw_visitor_id

        app.logger.info(
            "VISITOR-CLIENT visitor=%s device=%s browser=%s os=%s screen=%s viewport=%s tz=%s",
            raw_visitor_id,
            visitor_snapshot_store[raw_visitor_id].get("device_name", "-"),
            visitor_snapshot_store[raw_visitor_id].get("browser", "-"),
            visitor_snapshot_store[raw_visitor_id].get("operating_system", "-"),
            visitor_snapshot_store[raw_visitor_id].get("screen", "-"),
            visitor_snapshot_store[raw_visitor_id].get("viewport", "-"),
            visitor_snapshot_store[raw_visitor_id].get("timezone", "-"),
        )

        return jsonify(
            {
                "status": "ok",
                "visitor_id": raw_visitor_id,
                "device_name": visitor_snapshot_store[raw_visitor_id].get(
                    "device_name", ""
                ),
            }
        )

    @app.route("/admin/visitors", methods=["GET"])
    def admin_visitors():
        denied = require_admin_access()
        if denied:
            return denied

        try:
            limit = int(request.args.get("limit", 25) or 25)
        except (TypeError, ValueError):
            limit = 25
        limit = max(1, min(limit, 200))

        payload = build_visitors_payload(limit)
        response_format = normalize_text(request.args.get("format", "")).lower()

        if response_format == "json":
            return jsonify(payload)
        if response_format == "csv":
            return build_visitors_csv_response(payload)

        return render_template("admin_visitors.html", payload=payload, limit=limit)

    @app.route("/admin", methods=["GET"])
    def admin_root():
        denied = require_admin_access()
        if denied:
            return denied
        return redirect("/admin/visitors")
