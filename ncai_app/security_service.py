import ipaddress
import time
from typing import Any

from flask import jsonify

from .common import normalize_text


def get_security_request_ip(request: Any) -> str:
    remote_addr = normalize_text(getattr(request, "remote_addr", "") or "")
    cf_connecting_ip = normalize_text(request.headers.get("CF-Connecting-IP", ""))

    try:
        remote_is_loopback = bool(remote_addr) and ipaddress.ip_address(
            remote_addr
        ).is_loopback
    except ValueError:
        remote_is_loopback = False

    if remote_is_loopback and cf_connecting_ip:
        return cf_connecting_ip
    return remote_addr or "unknown"


def is_direct_loopback_request(request: Any) -> bool:
    remote_addr = normalize_text(getattr(request, "remote_addr", "") or "")
    try:
        return (
            bool(remote_addr)
            and ipaddress.ip_address(remote_addr).is_loopback
            and not normalize_text(request.headers.get("CF-Connecting-IP", ""))
            and not normalize_text(request.headers.get("X-Forwarded-For", ""))
        )
    except ValueError:
        return False


def get_bearer_or_header_token(request: Any, header_name: str) -> str:
    header_value = normalize_text(request.headers.get(header_name, ""))
    if header_value:
        return header_value

    authorization = normalize_text(request.headers.get("Authorization", ""))
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()

    return ""


def enforce_rate_limit(
    *,
    store: dict[tuple[str, str], list[float]],
    request: Any,
    bucket: str,
    limit: int,
    window_seconds: int,
):
    now = time.time()
    key = (bucket, get_security_request_ip(request))
    entries = [
        timestamp
        for timestamp in store.get(key, [])
        if now - timestamp < window_seconds
    ]

    if len(entries) >= limit:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "rate_limited",
                    "message": "요청이 너무 많습니다. 잠시 후 다시 시도해 주세요.",
                }
            ),
            429,
        )

    entries.append(now)
    store[key] = entries
    return None
