import hmac
from datetime import datetime

from flask import Flask, jsonify, redirect, request, session

from .admin_routes import register_admin_routes
from .analysis_routes import register_analysis_routes
from .auth_routes import register_auth_routes
from .auth_service import ensure_auth_db
from .common import normalize_text
from .config import (
    get_admin_emails,
    get_admin_token,
    get_admin_user_ids,
    get_android_api_key,
    get_google_login_status,
)
from .history_service import (
    get_analysis_generation,
)
from .llm_service import (
    get_google_credentials_status,
    get_llm_provider_status,
    get_model_status,
)
from .security_service import (
    get_bearer_or_header_token,
    is_direct_loopback_request,
)


_rate_limit_store: dict[tuple[str, str], list[float]] = {}


def register_routes(app: Flask) -> None:
    ensure_auth_db()

    def is_authenticated() -> bool:
        return bool(session.get("authenticated"))

    def require_login_redirect():
        if is_authenticated():
            return None
        return redirect("/login")

    def is_admin_session() -> bool:
        if not is_authenticated():
            return False

        admin_user_ids = get_admin_user_ids()
        admin_emails = get_admin_emails()
        operator_id = normalize_text(session.get("operator_id", "")).lower()
        operator_email = normalize_text(session.get("operator_email", "")).lower()

        return bool(
            (operator_id and operator_id in admin_user_ids)
            or (operator_email and operator_email in admin_emails)
        )

    def require_admin_access():
        if is_direct_loopback_request(request) or is_admin_session():
            return None

        admin_token = get_admin_token()
        provided_token = get_bearer_or_header_token(request, "X-Admin-Token")
        if admin_token and hmac.compare_digest(provided_token, admin_token):
            return None

        return jsonify({"error": "admin access denied"}), 403

    def require_api_access():
        api_key = get_android_api_key()
        provided_key = get_bearer_or_header_token(request, "X-API-Key")
        if api_key and hmac.compare_digest(provided_key, api_key):
            return None

        return jsonify({"error": "valid X-API-Key is required"}), 401

    def set_authenticated_session(user: dict) -> None:
        session["authenticated"] = True
        session["operator"] = user.get("email") or user.get("user_id", "")
        session["operator_id"] = user.get("user_id", "")
        session["operator_name"] = user.get("display_name", "")
        session["operator_email"] = user.get("email", "")
        session["operator_avatar_url"] = user.get("avatar_url", "")
        session["operator_provider"] = user.get("auth_provider", "local")

    def build_stale_generation_response(session_id: str, status_code: int = 409):
        return (
            jsonify(
                {
                    "stale": True,
                    "error": "초기화 이전 분석 요청이어서 현재 세션에는 반영되지 않습니다.",
                    "session_id": session_id,
                    "analysis_generation": get_analysis_generation(session_id),
                }
            ),
            status_code,
        )

    register_admin_routes(app, require_admin_access=require_admin_access)

    register_auth_routes(
        app,
        rate_limit_store=_rate_limit_store,
        is_authenticated=is_authenticated,
        require_login_redirect=require_login_redirect,
        set_authenticated_session=set_authenticated_session,
    )

    @app.route("/health", methods=["GET"])
    def health():
        model_status = get_model_status()
        credentials_status = get_google_credentials_status()
        google_login_status = get_google_login_status()
        provider_status = get_llm_provider_status()
        llm_ready = provider_status["local"]["ready"] or provider_status["api"]["ready"]
        ready = llm_ready and credentials_status["configured"]

        return jsonify(
            {
                "status": "ok" if ready else "degraded",
                "service": "ncai-dementia-risk-monitor",
                "time": datetime.now().isoformat(),
                "ready": ready,
                "model": model_status,
                "google_credentials": credentials_status,
                "google_login": google_login_status,
                "llm_provider": provider_status,
            }
        )

    register_analysis_routes(
        app,
        rate_limit_store=_rate_limit_store,
        require_api_access=require_api_access,
        build_stale_generation_response=build_stale_generation_response,
    )
