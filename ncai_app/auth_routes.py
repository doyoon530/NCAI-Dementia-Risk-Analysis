from flask import jsonify, make_response, redirect, render_template, request, session
from google.auth.transport import requests as google_auth_requests
from google.oauth2 import id_token as google_id_token

from .auth_service import (
    authenticate_user,
    create_or_update_google_user,
    create_user,
    validate_display_name,
    validate_password,
    validate_user_id,
)
from .common import normalize_text
from .config import (
    get_google_login_status,
    get_google_oauth_client_id,
    get_login_rate_limit,
    get_rate_limit_window_seconds,
)
from .security_service import enforce_rate_limit


def register_auth_routes(
    app,
    *,
    rate_limit_store,
    is_authenticated,
    require_login_redirect,
    set_authenticated_session,
):
    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "GET":
            if is_authenticated():
                return redirect("/")

            response = make_response(render_template("login.html"))
            response.headers["Cache-Control"] = (
                "no-store, no-cache, must-revalidate, max-age=0"
            )
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response

        payload = request.get_json(silent=True) or {}
        user_id = normalize_text(payload.get("user_id", ""))
        password = normalize_text(payload.get("password", ""))

        limited = enforce_rate_limit(
            store=rate_limit_store,
            request=request,
            bucket="login",
            limit=get_login_rate_limit(),
            window_seconds=get_rate_limit_window_seconds(),
        )
        if limited:
            return limited

        if not user_id or not password:
            return (
                jsonify(
                    {
                        "ok": False,
                        "message": "사용자 ID와 비밀번호를 모두 입력해 주세요.",
                    }
                ),
                400,
            )

        user = authenticate_user(user_id, password)
        if user is None:
            return (
                jsonify(
                    {
                        "ok": False,
                        "message": "사용자 ID 또는 비밀번호가 올바르지 않습니다.",
                    }
                ),
                401,
            )

        set_authenticated_session(user)
        session.permanent = True
        return jsonify(
            {
                "ok": True,
                "redirect": "/",
                "operator": session.get("operator", ""),
                "display_name": user["display_name"],
            }
        )

    @app.route("/auth/config", methods=["GET"])
    def auth_config():
        google_login_status = get_google_login_status()
        return jsonify(
            {
                "google_login_enabled": bool(google_login_status["enabled"]),
                "google_client_id": google_login_status["client_id"] or "",
            }
        )

    @app.route("/auth/google", methods=["POST"])
    def auth_google():
        limited = enforce_rate_limit(
            store=rate_limit_store,
            request=request,
            bucket="auth-google",
            limit=get_login_rate_limit(),
            window_seconds=get_rate_limit_window_seconds(),
        )
        if limited:
            return limited

        google_client_id = get_google_oauth_client_id()
        if not google_client_id:
            return (
                jsonify(
                    {
                        "ok": False,
                        "message": "Google 로그인 설정이 아직 완료되지 않았습니다.",
                    }
                ),
                503,
            )

        payload = request.get_json(silent=True) or {}
        credential = normalize_text(payload.get("credential", ""))
        if not credential:
            return (
                jsonify(
                    {"ok": False, "message": "Google 인증 토큰이 비어 있습니다."}
                ),
                400,
            )

        try:
            token_info = google_id_token.verify_oauth2_token(
                credential,
                google_auth_requests.Request(),
                google_client_id,
            )
        except ValueError:
            return (
                jsonify(
                    {
                        "ok": False,
                        "message": "Google 로그인 검증에 실패했습니다. 다시 시도해 주세요.",
                    }
                ),
                401,
            )

        if not token_info.get("email_verified", False):
            return (
                jsonify(
                    {
                        "ok": False,
                        "message": "이메일 인증이 완료된 Google 계정만 사용할 수 있습니다.",
                    }
                ),
                403,
            )

        google_sub = normalize_text(token_info.get("sub", ""))
        google_email = normalize_text(token_info.get("email", ""))
        if not google_sub or not google_email:
            return (
                jsonify(
                    {
                        "ok": False,
                        "message": "Google 계정 정보를 확인할 수 없습니다.",
                    }
                ),
                400,
            )

        user = create_or_update_google_user(
            google_sub=google_sub,
            email=google_email,
            display_name=normalize_text(token_info.get("name", "")) or "Google User",
            avatar_url=normalize_text(token_info.get("picture", "")),
        )
        set_authenticated_session(user)
        session.permanent = True

        return jsonify(
            {
                "ok": True,
                "redirect": "/",
                "operator": session.get("operator", ""),
                "display_name": user["display_name"],
                "provider": user["auth_provider"],
            }
        )

    @app.route("/signup", methods=["POST"])
    def signup():
        limited = enforce_rate_limit(
            store=rate_limit_store,
            request=request,
            bucket="signup",
            limit=get_login_rate_limit(),
            window_seconds=get_rate_limit_window_seconds(),
        )
        if limited:
            return limited

        payload = request.get_json(silent=True) or {}
        display_name = normalize_text(payload.get("display_name", ""))
        user_id = normalize_text(payload.get("user_id", ""))
        password = normalize_text(payload.get("password", ""))
        confirm_password = normalize_text(payload.get("confirm_password", ""))

        if not display_name or not user_id or not password or not confirm_password:
            return (
                jsonify(
                    {
                        "ok": False,
                        "message": "사용자명, 사용자 ID, 비밀번호, 비밀번호 확인을 모두 입력해 주세요.",
                    }
                ),
                400,
            )

        try:
            normalized_display_name = validate_display_name(display_name)
            normalized_user_id = validate_user_id(user_id)
            normalized_password = validate_password(password, confirm_password)
        except ValueError as error:
            return jsonify({"ok": False, "message": str(error)}), 400

        try:
            user = create_user(
                normalized_user_id,
                normalized_display_name,
                normalized_password,
            )
        except ValueError as error:
            return jsonify({"ok": False, "message": str(error)}), 409

        return jsonify(
            {
                "ok": True,
                "message": "회원가입이 완료되었습니다. 로그인해 주세요.",
                "user": user,
            }
        )

    @app.route("/logout", methods=["POST"])
    def logout():
        session.pop("authenticated", None)
        session.pop("operator", None)
        session.pop("operator_id", None)
        session.pop("operator_name", None)
        session.pop("operator_email", None)
        session.pop("operator_avatar_url", None)
        session.pop("operator_provider", None)
        return jsonify({"ok": True, "redirect": "/login"})

    @app.route("/")
    def index():
        login_redirect = require_login_redirect()
        if login_redirect is not None:
            return login_redirect

        return render_template(
            "index.html",
            operator_name=session.get("operator_name", ""),
            operator_id=session.get("operator_id", "") or session.get("operator", ""),
        )

    @app.route("/team")
    def team():
        return render_template("team.html")
