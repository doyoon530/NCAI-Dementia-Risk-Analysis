import hashlib
import os
import secrets
import socket

from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix

from ncai_app.config import (
    UPLOAD_DIR,
    get_server_host,
    get_server_port,
    get_waitress_threads,
)
from ncai_app.routes import register_routes

try:
    from waitress import serve
except ImportError:
    serve = None


def load_local_env() -> None:
    for filename in (".env.local", ".env"):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        if not os.path.exists(path):
            continue

        with open(path, encoding="utf-8-sig") as env_file:
            for line in env_file:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
app.config["SECRET_KEY"] = os.getenv("NCAI_SECRET_KEY") or secrets.token_hex(32)
app.config["JSON_AS_ASCII"] = False
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
secure_cookie = os.getenv("SESSION_COOKIE_SECURE", "").strip().lower()
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = (
    secure_cookie in {"1", "true", "yes", "on"}
    or bool(os.getenv("CLOUDFLARE_PUBLIC_HOSTNAME", "").strip())
)

os.makedirs(UPLOAD_DIR, exist_ok=True)
register_routes(app)


def _compute_static_ver(filename: str) -> str:
    path = os.path.join(app.static_folder, filename)
    try:
        mtime = str(os.path.getmtime(path)).encode()
        return hashlib.md5(mtime).hexdigest()[:8]
    except OSError:
        return "0"


_static_ver_cache: dict[str, str] = {}


def _static_ver(filename: str) -> str:
    if filename not in _static_ver_cache:
        _static_ver_cache[filename] = _compute_static_ver(filename)
    return _static_ver_cache[filename]


@app.context_processor
def inject_static_ver():
    return {"static_ver": _static_ver}


def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"


def print_server_urls(host: str, port: int) -> None:
    local_ip = get_local_ip()

    print("=" * 60)
    print("NCAI server is starting")
    print(f"Local URL:   http://127.0.0.1:{port}")
    print(f"LAN URL:     http://{local_ip}:{port}")
    if host not in {"0.0.0.0", "::"}:
        print(f"Bind Host:   {host}")
    else:
        print("Bind Host:   0.0.0.0 (all network interfaces)")
    print("=" * 60)


def run_server() -> None:
    host = get_server_host()
    port = get_server_port()

    print_server_urls(host, port)

    if serve is not None:
        serve(app, host=host, port=port, threads=get_waitress_threads())
        return

    print("waitress is not installed. Falling back to Flask development server.")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    run_server()
