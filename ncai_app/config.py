import os

from langchain_core.prompts import ChatPromptTemplate


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DATA_DIR = os.path.join(BASE_DIR, "data")
AUTH_DB_PATH = os.path.join(DATA_DIR, "dr_jinu_auth.db")
DEFAULT_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "EXAONE-3.5-7.8B-Instruct-Q8_0.gguf"
)
DEFAULT_GOOGLE_KEY_PATH = os.path.join(
    BASE_DIR, "stt-bot-489913-807430be631b.json"
)
DEFAULT_API_LLM_BASE_URL = "https://api.openai.com/v1"
SUPPORTED_LLM_PROVIDERS = {"local", "api"}

MAX_HISTORY_TURNS = 12
MAX_SCORE_HISTORY = 30
MAX_ANALYSIS_RETRY = 3
RECENT_WINDOW = 5
ANALYSIS_CONTEXT_TURNS = 3
REPETITION_CONTEXT_TURNS = 4
REPETITION_SCORE_OPTIONS = [0, 8, 15, 20, 25]
REPETITION_HEURISTIC_SKIP_THRESHOLD = 20

RECALL_WORDS = [
    "사과",
    "버스",
    "바다",
    "연필",
    "시계",
    "나무",
    "기차",
    "고양이",
    "책상",
    "우산",
    "커피",
    "구름",
    "노트",
    "학교",
    "안경",
]

ANSWER_STOP_SEQUENCES = [
    "\n질문:",
    "\n사용자 질문:",
    "\n사용자 발화:",
    "\n사용자:",
    "\nUser:",
    "\nHuman:",
]

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
당신은 치매 케어 대화형 보조 AI입니다.

규칙:
- 사용자에게 바로 보여줄 최종 답변만 작성합니다.
- 프롬프트 설명, 역할 설명, 메타 문구를 쓰지 않습니다.
- "질문:", "답변:" 같은 접두어를 붙이지 않습니다.
- 근거가 부족하면 지어내지 말고 짧고 공손하게 확인합니다.
- 답변은 1~3문장으로 간결하게 작성합니다.
- 말투는 자연스럽고 따뜻하게 유지합니다.
""",
        ),
        (
            "human",
            """
사용자 발화:
{question}

위 발화를 바탕으로 사용자에게 바로 보여줄 최종 답변만 작성하세요.
""",
        ),
    ]
)

ANALYSIS_SCORING_GUIDE = """
분석 항목은 총 4가지입니다.

1. 질문 반복
2. 기억 혼란
3. 시간/상황 혼란
4. 문장 비논리성

점수 기준

[질문반복점수] 0~25
0 : 반복 없음
8 : 같은 의미의 질문이지만 반복 여부가 애매함
15 : 같은 질문이 1회 반복됨
20 : 같은 질문이 2회 이상 반복됨
25 : 직전 답변 이후에도 거의 동일한 질문을 계속 반복함

[기억혼란점수] 0~25
0 : 기억 관련 이상 신호 없음
8 : "기억이 안 난다", "모르겠다" 같은 표현이 1회 나타남
15 : 최근 정보 회상에 어려움이 드러남
20 : 기억 혼란 표현이 여러 번 반복됨
25 : 방금 제시한 정보도 유지하지 못하는 수준이 드러남

[시간혼란점수] 0~30
0 : 시간/상황 혼란 없음
10 : 날짜, 요일, 시간 중 일부를 헷갈림
18 : 현재 일정이나 상황을 바로 떠올리지 못함
24 : 시간 정보와 상황 정보가 반복적으로 흔들림
30 : 현재 시점이나 맥락을 거의 파악하지 못함

[문장비논리점수] 0~20
0 : 문장 흐름이 자연스럽고 의미가 분명함
5 : 다소 모호하지만 전체 의미는 전달됨
10 : 문장 연결이 불안정함
15 : 주제가 자주 이탈하거나 연결성이 약함
20 : 문장 전개가 매우 약하고 비논리성이 강함

최종 점수는 네 항목 점수의 합입니다.

최종 판단 기준
0~19 : 정상
20~39 : 의심
40 이상 : 판단 어려움
"""

ANALYSIS_OUTPUT_RULES = """
중요 규칙

- 반드시 아래 형식만 출력합니다.
- 질문하거나 조언하지 않습니다.
- 각 점수는 정수 하나만 작성합니다.
- 근거는 2문장 이상 작성합니다.
- 최종 점수는 세부 점수 합과 일치해야 합니다.

출력 형식

판단:
최종점수:
질문반복점수:
기억혼란점수:
시간혼란점수:
문장비논리점수:
근거:
"""

analysis_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
당신은 대화 기반 인지 위험 신호 분석기입니다.

지침:
- 질문하지 않습니다.
- 조언하지 않습니다.
- 현재 질문만 보지 말고 최근 대화 맥락도 함께 봅니다.
- 점수는 과도하게 높이지 말고, 근거가 약하면 보수적으로 판단합니다.

{ANALYSIS_SCORING_GUIDE}

{ANALYSIS_OUTPUT_RULES}
""",
        ),
        (
            "human",
            """
최근 대화 맥락:
{conversation_context}

현재 질문:
{question}

반드시 지정된 형식만 출력하세요.
""",
        ),
    ]
)

analysis_retry_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
당신은 대화 기반 인지 위험 신호 분석기입니다.

이전 응답은 형식이 잘못되었습니다.
이번에는 오직 지정된 형식만 사용하세요.

{ANALYSIS_SCORING_GUIDE}

{ANALYSIS_OUTPUT_RULES}
""",
        ),
        (
            "human",
            """
최근 대화 맥락:
{conversation_context}

현재 질문:
{question}

이전 응답:
{previous_response}

반드시 지정된 형식만 다시 출력하세요.
""",
        ),
    ]
)

repetition_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
당신은 질문 반복 여부만 판정하는 보조 분석기입니다.

규칙:
- 오직 질문 반복만 봅니다.
- 기억 혼란, 시간 혼란, 문장 비논리성은 평가하지 않습니다.
- 표현이 조금 달라도 같은 요청이면 반복으로 볼 수 있습니다.
- 같은 주제라도 다른 요청이면 반복으로 보지 않습니다.

질문반복점수는 아래 값 중 하나만 사용합니다.
0, 8, 15, 20, 25

출력 형식:
질문반복점수:
근거:
""",
        ),
        (
            "human",
            """
최근 사용자 질문과 직후 답변:
{recent_user_questions}

현재 질문:
{question}

반드시 지정된 형식만 출력하세요.
""",
        ),
    ]
)

ROLE_ANALYSIS_META = {
    "repetition": {
        "title": "질문 반복",
        "score_label": "질문반복점수",
        "max_score": 25,
        "focus_rule": "최근 질문과 현재 질문의 의미가 같은지에만 집중하세요.",
    },
    "memory": {
        "title": "기억 혼란",
        "score_label": "기억혼란점수",
        "max_score": 25,
        "focus_rule": "기억이 나지 않는다는 표현이나 방금 나온 정보를 유지하지 못하는지에만 집중하세요.",
    },
    "time_confusion": {
        "title": "시간/상황 혼란",
        "score_label": "시간혼란점수",
        "max_score": 30,
        "focus_rule": "시간, 일정, 현재 상황을 헷갈리는 표현만 평가하세요.",
    },
    "incoherence": {
        "title": "문장 비논리성",
        "score_label": "문장비논리점수",
        "max_score": 20,
        "focus_rule": "문장 연결이 불안정하거나 논리 흐름이 깨지는 부분만 평가하세요.",
    },
}

ROLE_ANALYSIS_ORDER = ["repetition", "memory", "time_confusion", "incoherence"]


def normalize_role_key(role_key: str) -> str:
    normalized = str(role_key or "").strip().lower()
    aliases = {
        "repetition": "repetition",
        "memory": "memory",
        "time_confusion": "time_confusion",
        "time": "time_confusion",
        "incoherence": "incoherence",
    }
    return aliases.get(normalized, "")


def build_role_prompt(role_key: str) -> ChatPromptTemplate:
    meta = ROLE_ANALYSIS_META[role_key]
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
당신은 대화 기반 인지 위험 신호 분석기입니다.

이번에는 오직 {meta["title"]}만 평가하세요.
다른 항목 점수나 최종 판단은 쓰지 않습니다.

규칙:
- 질문하거나 조언하지 않습니다.
- {meta["focus_rule"]}
- 근거는 반드시 2문장 이상 작성합니다.

출력 형식:
{meta["score_label"]}:
근거:
""",
            ),
            (
                "human",
                """
최근 대화 맥락:
{conversation_context}

현재 질문:
{question}

반드시 지정된 형식만 출력하세요.
""",
            ),
        ]
    )


def build_role_retry_prompt(role_key: str) -> ChatPromptTemplate:
    meta = ROLE_ANALYSIS_META[role_key]
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
당신은 대화 기반 인지 위험 신호 분석기입니다.

이번에는 오직 {meta["title"]}만 평가하세요.
이전 응답은 형식이 잘못되었습니다.

출력 형식:
{meta["score_label"]}:
근거:
""",
            ),
            (
                "human",
                """
최근 대화 맥락:
{conversation_context}

현재 질문:
{question}

이전 응답:
{previous_response}

반드시 지정된 형식만 다시 출력하세요.
""",
            ),
        ]
    )


ROLE_ANALYSIS_PROMPTS = {
    role_key: build_role_prompt(role_key) for role_key in ROLE_ANALYSIS_META
}

ROLE_ANALYSIS_RETRY_PROMPTS = {
    role_key: build_role_retry_prompt(role_key)
    for role_key in ROLE_ANALYSIS_META
}


def normalize_llm_provider(provider: str | None) -> str:
    normalized = str(provider or "").strip().lower()
    return "api" if normalized == "api" else "local"


def get_model_path() -> str:
    return os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)


def get_default_llm_provider() -> str:
    return normalize_llm_provider(os.getenv("LLM_PROVIDER", "local"))


def get_api_llm_base_url() -> str:
    return os.getenv("API_LLM_BASE_URL", DEFAULT_API_LLM_BASE_URL).strip().rstrip("/")


def get_api_llm_api_key() -> str:
    return os.getenv("API_LLM_API_KEY", "").strip()


def get_api_llm_answer_model() -> str:
    return os.getenv("API_LLM_ANSWER_MODEL", "").strip()


def get_api_llm_analysis_model() -> str:
    return os.getenv("API_LLM_ANALYSIS_MODEL", "").strip() or get_api_llm_answer_model()


def get_positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default

    try:
        parsed = int(raw_value)
        if parsed < 1:
            raise ValueError
        return parsed
    except ValueError:
        print(f"[config] Invalid {name}={raw_value!r}; using default {default}")
        return default


def get_api_llm_timeout() -> int:
    return get_positive_int_env("API_LLM_TIMEOUT", 60)


def is_api_llm_configured() -> bool:
    return bool(
        get_api_llm_api_key()
        and get_api_llm_answer_model()
        and get_api_llm_analysis_model()
    )


def get_analysis_n_ctx() -> int:
    return get_positive_int_env("ANALYSIS_N_CTX", 8192)


def get_analysis_max_tokens() -> int:
    return get_positive_int_env("ANALYSIS_MAX_TOKENS", 384)


def get_analysis_n_batch() -> int:
    n_ctx = get_analysis_n_ctx()
    requested = get_positive_int_env("ANALYSIS_N_BATCH", 512)
    return max(64, min(requested, n_ctx))


def get_server_host() -> str:
    return os.getenv("HOST", "0.0.0.0")


def get_server_port() -> int:
    return get_positive_int_env("PORT", 5000)


def get_waitress_threads() -> int:
    return get_positive_int_env("WAITRESS_THREADS", 8)


def get_google_oauth_client_id() -> str:
    return (
        os.getenv("GOOGLE_OAUTH_CLIENT_ID", "")
        or os.getenv("GOOGLE_CLIENT_ID", "")
    ).strip()


def get_google_login_status() -> dict:
    client_id = get_google_oauth_client_id()
    return {"enabled": bool(client_id), "client_id": client_id}


def get_admin_token() -> str:
    return os.getenv("ADMIN_TOKEN", "").strip()


def get_android_api_key() -> str:
    return os.getenv("ANDROID_API_KEY", "").strip()


def get_rate_limit_window_seconds() -> int:
    return get_positive_int_env("RATE_LIMIT_WINDOW_SECONDS", 60)


def get_login_rate_limit() -> int:
    return get_positive_int_env("LOGIN_RATE_LIMIT", 12)


def get_api_rate_limit() -> int:
    return get_positive_int_env("API_RATE_LIMIT", 30)


def get_ffmpeg_timeout_seconds() -> int:
    return get_positive_int_env("FFMPEG_TIMEOUT_SECONDS", 30)


def get_allowed_audio_extensions() -> set[str]:
    raw_value = os.getenv(
        "ALLOWED_AUDIO_EXTENSIONS",
        "wav,webm,m4a,mp3,ogg",
    ).strip()
    return {
        item.strip().lower().lstrip(".")
        for item in raw_value.split(",")
        if item.strip()
    }


def _get_csv_env_set(name: str) -> set[str]:
    raw_value = os.getenv(name, "").strip()
    return {
        item.strip().lower()
        for item in raw_value.split(",")
        if item.strip()
    }


def get_admin_user_ids() -> set[str]:
    return _get_csv_env_set("ADMIN_USER_IDS")


def get_admin_emails() -> set[str]:
    return _get_csv_env_set("ADMIN_EMAILS")
