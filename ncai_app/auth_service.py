import os
import re
import sqlite3
from datetime import datetime

from werkzeug.security import check_password_hash, generate_password_hash

from .config import AUTH_DB_PATH, DATA_DIR

USER_ID_PATTERN = re.compile(r"^[a-z][a-z0-9._-]{3,19}$")
PASSWORD_ALLOWED_PATTERN = re.compile(r"^[A-Za-z0-9!@#$%^&*()_+\-=\[\]{};':\",.<>/?`~\\|]+$")


def ensure_auth_db() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with sqlite3.connect(AUTH_DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL DEFAULT '',
                auth_provider TEXT NOT NULL DEFAULT 'local',
                email TEXT,
                google_sub TEXT,
                avatar_url TEXT,
                created_at TEXT NOT NULL,
                last_login_at TEXT
            )
            """
        )

        existing_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(users)").fetchall()
        }

        required_columns = {
            "auth_provider": "ALTER TABLE users ADD COLUMN auth_provider TEXT NOT NULL DEFAULT 'local'",
            "email": "ALTER TABLE users ADD COLUMN email TEXT",
            "google_sub": "ALTER TABLE users ADD COLUMN google_sub TEXT",
            "avatar_url": "ALTER TABLE users ADD COLUMN avatar_url TEXT",
            "last_login_at": "ALTER TABLE users ADD COLUMN last_login_at TEXT",
        }

        for column, statement in required_columns.items():
            if column not in existing_columns:
                connection.execute(statement)

        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_sub
            ON users(google_sub)
            WHERE google_sub IS NOT NULL AND google_sub != ''
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_users_email
            ON users(email)
            """
        )
        connection.commit()


def get_auth_connection() -> sqlite3.Connection:
    ensure_auth_db()
    connection = sqlite3.connect(AUTH_DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _normalize_provider(provider: str) -> str:
    normalized = (provider or "local").strip().lower()
    if normalized not in {"local", "google", "hybrid"}:
        return "local"
    return normalized


_USER_COLUMNS = (
    "id, user_id, display_name, password_hash, auth_provider, "
    "email, google_sub, avatar_url, created_at, last_login_at"
)


def _serialize_user(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None

    user = dict(row)
    return {
        "id": user["id"],
        "user_id": user["user_id"],
        "display_name": user["display_name"],
        "password_hash": user.get("password_hash", ""),
        "auth_provider": _normalize_provider(user.get("auth_provider", "local")),
        "email": user.get("email", "") or "",
        "google_sub": user.get("google_sub", "") or "",
        "avatar_url": user.get("avatar_url", "") or "",
        "created_at": user["created_at"],
        "last_login_at": user.get("last_login_at", "") or "",
    }


def _touch_last_login(connection: sqlite3.Connection, user_id: int) -> None:
    connection.execute(
        """
        UPDATE users
        SET last_login_at = ?
        WHERE id = ?
        """,
        (_now_iso(), user_id),
    )


def _build_google_user_id(email: str, google_sub: str) -> str:
    normalized_email = (email or "").strip().lower()
    if normalized_email:
        return normalized_email
    return f"google_{google_sub[-12:]}"


def _determine_google_provider(password_hash: str) -> str:
    return "hybrid" if (password_hash or "").strip() else "google"


def validate_display_name(display_name: str) -> str:
    normalized_display_name = " ".join((display_name or "").strip().split())
    if len(normalized_display_name) < 2 or len(normalized_display_name) > 20:
        raise ValueError("사용자명은 2자 이상 20자 이하로 입력해 주세요.")
    return normalized_display_name


def validate_user_id(user_id: str) -> str:
    normalized_user_id = (user_id or "").strip().lower()
    if not USER_ID_PATTERN.fullmatch(normalized_user_id):
        raise ValueError(
            "사용자 ID는 영문 소문자로 시작하는 4~20자여야 하며, 소문자/숫자/._- 만 사용할 수 있습니다."
        )
    return normalized_user_id


def validate_password(password: str, confirm_password: str | None = None) -> str:
    normalized_password = password or ""
    if len(normalized_password) < 8 or len(normalized_password) > 32:
        raise ValueError("비밀번호는 8자 이상 32자 이하로 입력해 주세요.")
    if not re.search(r"[A-Za-z]", normalized_password):
        raise ValueError("비밀번호에는 영문자가 최소 1개 이상 포함되어야 합니다.")
    if not re.search(r"\d", normalized_password):
        raise ValueError("비밀번호에는 숫자가 최소 1개 이상 포함되어야 합니다.")
    if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\",.<>/?`~\\|]", normalized_password):
        raise ValueError("비밀번호에는 특수문자가 최소 1개 이상 포함되어야 합니다.")
    if not PASSWORD_ALLOWED_PATTERN.fullmatch(normalized_password):
        raise ValueError("비밀번호에는 영문자, 숫자, 특수문자만 사용할 수 있습니다.")
    if confirm_password is not None and normalized_password != (confirm_password or ""):
        raise ValueError("비밀번호 확인이 일치하지 않습니다.")
    return normalized_password


def create_user(user_id: str, display_name: str, password: str) -> dict:
    normalized_user_id = validate_user_id(user_id)
    normalized_display_name = validate_display_name(display_name)
    normalized_password = validate_password(password)
    password_hash = generate_password_hash(normalized_password)
    created_at = _now_iso()

    try:
        with get_auth_connection() as connection:
            connection.execute(
                """
                INSERT INTO users (
                    user_id,
                    display_name,
                    password_hash,
                    auth_provider,
                    created_at,
                    last_login_at
                )
                VALUES (?, ?, ?, 'local', ?, ?)
                """,
                (
                    normalized_user_id,
                    normalized_display_name,
                    password_hash,
                    created_at,
                    created_at,
                ),
            )
            connection.commit()
    except sqlite3.IntegrityError as error:
        raise ValueError("이미 존재하는 사용자 ID입니다.") from error

    return {
        "user_id": normalized_user_id,
        "display_name": normalized_display_name,
        "auth_provider": "local",
        "email": "",
        "avatar_url": "",
        "created_at": created_at,
        "last_login_at": created_at,
    }


def get_user_by_user_id(user_id: str) -> dict | None:
    with get_auth_connection() as connection:
        row = connection.execute(
            f"SELECT {_USER_COLUMNS} FROM users WHERE user_id = ?",
            (user_id.strip(),),
        ).fetchone()
    return _serialize_user(row)


def get_user_by_google_sub(google_sub: str) -> dict | None:
    with get_auth_connection() as connection:
        row = connection.execute(
            f"SELECT {_USER_COLUMNS} FROM users WHERE google_sub = ?",
            (google_sub.strip(),),
        ).fetchone()
    return _serialize_user(row)


def authenticate_user(user_id: str, password: str) -> dict | None:
    user = get_user_by_user_id(user_id)
    if user is None:
        return None

    if not check_password_hash(user["password_hash"], password):
        return None

    with get_auth_connection() as connection:
        _touch_last_login(connection, user["id"])
        connection.commit()

    user["last_login_at"] = _now_iso()
    return user


def create_or_update_google_user(
    *,
    google_sub: str,
    email: str,
    display_name: str,
    avatar_url: str = "",
) -> dict:
    normalized_google_sub = google_sub.strip()
    normalized_email = email.strip().lower()
    normalized_display_name = display_name.strip() or "Google User"
    normalized_avatar_url = avatar_url.strip()
    now = _now_iso()

    with get_auth_connection() as connection:
        row = connection.execute(
            f"SELECT {_USER_COLUMNS} FROM users WHERE google_sub = ?",
            (normalized_google_sub,),
        ).fetchone()

        if row is None and normalized_email:
            row = connection.execute(
                f"SELECT {_USER_COLUMNS} FROM users WHERE email = ? OR user_id = ? ORDER BY id ASC LIMIT 1",
                (normalized_email, normalized_email),
            ).fetchone()

        if row is not None:
            existing = dict(row)
            updated_provider = _determine_google_provider(
                existing.get("password_hash", "")
            )
            connection.execute(
                """
                UPDATE users
                SET display_name = ?,
                    email = ?,
                    google_sub = ?,
                    avatar_url = ?,
                    auth_provider = ?,
                    last_login_at = ?
                WHERE id = ?
                """,
                (
                    normalized_display_name,
                    normalized_email or existing.get("email", ""),
                    normalized_google_sub,
                    normalized_avatar_url,
                    updated_provider,
                    now,
                    existing["id"],
                ),
            )
            connection.commit()

            return {
                "id": existing["id"],
                "user_id": existing["user_id"],
                "display_name": normalized_display_name,
                "password_hash": existing.get("password_hash", ""),
                "auth_provider": updated_provider,
                "email": normalized_email or existing.get("email", ""),
                "google_sub": normalized_google_sub,
                "avatar_url": normalized_avatar_url,
                "created_at": existing["created_at"],
                "last_login_at": now,
            }

        generated_user_id = _build_google_user_id(normalized_email, normalized_google_sub)
        candidate_user_id = generated_user_id
        sequence = 1

        while True:
            exists = connection.execute(
                "SELECT 1 FROM users WHERE user_id = ?",
                (candidate_user_id,),
            ).fetchone()
            if exists is None:
                break
            sequence += 1
            candidate_user_id = f"{generated_user_id}-{sequence}"

        connection.execute(
            """
            INSERT INTO users (
                user_id,
                display_name,
                password_hash,
                auth_provider,
                email,
                google_sub,
                avatar_url,
                created_at,
                last_login_at
            )
            VALUES (?, ?, '', 'google', ?, ?, ?, ?, ?)
            """,
            (
                candidate_user_id,
                normalized_display_name,
                normalized_email,
                normalized_google_sub,
                normalized_avatar_url,
                now,
                now,
            ),
        )
        user_id = connection.execute("SELECT last_insert_rowid()").fetchone()[0]
        connection.commit()

    return {
        "id": user_id,
        "user_id": candidate_user_id,
        "display_name": normalized_display_name,
        "password_hash": "",
        "auth_provider": "google",
        "email": normalized_email,
        "google_sub": normalized_google_sub,
        "avatar_url": normalized_avatar_url,
        "created_at": now,
        "last_login_at": now,
    }


def list_users() -> list[dict]:
    with get_auth_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                user_id,
                display_name,
                auth_provider,
                email,
                google_sub,
                avatar_url,
                created_at,
                last_login_at
            FROM users
            ORDER BY created_at ASC, id ASC
            """
        ).fetchall()

    return [_serialize_user(row) for row in rows]
