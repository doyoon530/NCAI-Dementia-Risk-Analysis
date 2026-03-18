import os

from langchain_core.prompts import ChatPromptTemplate


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DEFAULT_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "EXAONE-3.5-7.8B-Instruct-Q8_0.gguf"
)
DEFAULT_GOOGLE_KEY_PATH = os.path.join(BASE_DIR, "stt-bot-489913-807430be631b.json")
DEFAULT_API_LLM_BASE_URL = "https://api.openai.com/v1"
SUPPORTED_LLM_PROVIDERS = {"local", "api"}

MAX_HISTORY_TURNS = 12
MAX_SCORE_HISTORY = 30
MAX_ANALYSIS_RETRY = 3
RECENT_WINDOW = 5
ANALYSIS_CONTEXT_TURNS = 3
REPETITION_CONTEXT_TURNS = 4
REPETITION_SCORE_OPTIONS = [0, 8, 15, 20, 25]

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
    "수건",
    "안경",
]

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
당신은 사용자의 질문에 짧고 직접적으로 답변하는 AI입니다.

규칙:
- 반드시 한국어로만 답변하십시오.
- 질문에 없는 사실을 지어내지 마십시오.
- 상황, 장소, 시간, 인물, 사건을 임의로 추가하지 마십시오.
- 소설처럼 서술하지 마십시오.
- 사용자가 물은 내용에만 직접 답하십시오.
- 질문만으로 확실히 알 수 없는 내용은 추측하지 말고 확인이 필요하다고 답하십시오.
- 정보가 부족하면 부족하다고 짧게 설명하십시오.
- 답변은 2~4문장 이내로 간결하게 작성하십시오.
- 'AI:' 같은 접두어를 절대 붙이지 마십시오.
""",
        ),
        (
            "human",
            """
질문: {question}
""",
        ),
    ]
)

ANALYSIS_SCORING_GUIDE = """
분석 항목 (총 4개)

1. 같은 질문 또는 의미가 같은 질문의 반복
2. 기억 혼란 또는 기억 상실 표현
3. 시간 또는 상황 인식 혼란
4. 문장 구조의 비논리성 또는 담화 응집성 저하

점수 산정 방식

각 항목을 아래 기준으로 평가하십시오.

[1] 질문반복점수 (0~25점)
0 : 반복 없음
8 : 표현이 일부 반복되지만 의미 반복은 아님
15 : 같은 의미의 질문이 1회 반복
20 : 같은 질문이 거의 동일하게 2회 이상 반복
25 : 답을 들은 직후에도 같은 질문을 반복하는 수준

[2] 기억혼란점수 (0~25점)
0 : 기억 혼란 없음
8 : "뭐였지", "기억이 안 나네" 같은 약한 표현 1회
15 : 기억 공백 또는 기억 혼란이 분명히 나타남
20 : 기억 혼란 표현이 여러 번 나타남
25 : 방금 들은 정보조차 기억하지 못하는 표현

[3] 시간혼란점수 (0~30점)
0 : 혼란 없음
10 : 날짜/요일 등을 잠깐 헷갈림
18 : 현재 시간 또는 상황을 잘못 인식
24 : 시간과 상황을 동시에 혼동
30 : 현재 연도 / 현재 상황에 대한 명백한 혼란

[4] 문장비논리점수 (0~20점)
0 : 문장 구조 자연스러움
5 : 약간 모호하지만 의미 이해 가능
10 : 문장 구조가 흔들려 해석이 불안정
15 : 주제가 자주 이탈하거나 연결이 약함
20 : 의미 응집성이 매우 낮고 비논리성이 강함

최종 점수 계산
의심점수 = 질문반복점수 + 기억혼란점수 + 시간혼란점수 + 문장비논리점수

총점 범위는 0~100입니다.

판단 기준
0~19 → 정상
20~39 → 약한 의심
40~59 → 중간 의심
60~79 → 높은 의심
80~100 → 매우 높은 의심

판단 규칙
- 0~19점이면 판단: 정상
- 20점 이상이면 판단: 의심
- 분석 근거가 부족하면 판단: 판단 어려움
"""

ANALYSIS_OUTPUT_RULES = """
중요 규칙

- 반드시 한국어로 작성하십시오.
- 질문에 대한 답변을 절대 하지 마십시오.
- 반드시 아래 형식만 출력하십시오.
- 다른 문장은 절대 출력하지 마십시오.
- 근거에는 실제 분석 내용만 작성하십시오.
- '한 문장 이상', '두 문장 이상', '작성하십시오' 같은 지시문을 그대로 출력하면 실패입니다.
- 근거는 반드시 실제 분석 문장 2문장 이상 작성하십시오.
- 의심점수는 반드시 0~100 사이 정수 하나만 작성하십시오.
- 각 세부 점수도 반드시 정수 하나만 작성하십시오.
- 의심점수는 네 개 세부 점수의 합과 같아야 합니다.

출력 형식

판단:
의심점수:
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
당신은 대화 기반 치매 징후 분석기입니다.

절대 질문에 답변하지 마십시오.
조언하지 마십시오.
설명하지 마십시오.

가능하면 현재 질문뿐 아니라 대화 맥락에서 나타나는 반복 질문이나 기억 혼란 패턴도 함께 고려하여 판단하십시오.
질문반복점수는 최근 대화에서 같은 의미의 질문이 다시 제시되었는지, 특히 직전 답변 직후 비슷한 질문이 반복되었는지를 우선 확인하여 평가하십시오.
단일 질문만으로 강한 판단을 내리지 말고 언어적 특징이 명확할 때만 높은 점수를 부여하십시오.

사용자의 질문에 나타난 언어적 특징을 분석하여 치매 의심 징후 점수를 계산하십시오.

{ANALYSIS_SCORING_GUIDE}

{ANALYSIS_OUTPUT_RULES}
""",
        ),
        (
            "human",
            """
최근 대화 맥락:
{conversation_context}

현재 질문: {question}

반드시 아래 형식만 출력하세요.

판단:
의심점수:
질문반복점수:
기억혼란점수:
시간혼란점수:
문장비논리점수:
근거:
""",
        ),
    ]
)

analysis_retry_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
당신은 대화 기반 치매 징후 분석기입니다.

질문에 답변하지 마십시오.
조언하지 마십시오.
설명하지 마십시오.

이전 응답은 형식이 잘못되었습니다.
점수 기준에 따라 다시 분석하고 형식을 정확히 맞추십시오.

가능하면 현재 질문뿐 아니라 대화 맥락에서 나타나는 반복 질문이나 기억 혼란 패턴도 함께 고려하여 판단하십시오.
질문반복점수는 최근 대화에서 같은 의미의 질문이 다시 제시되었는지, 특히 직전 답변 직후 비슷한 질문이 반복되었는지를 우선 확인하여 평가하십시오.
단일 질문만으로 강한 판단을 내리지 말고 언어적 특징이 명확할 때만 높은 점수를 부여하십시오.

{ANALYSIS_SCORING_GUIDE}

{ANALYSIS_OUTPUT_RULES}
""",
        ),
        (
            "human",
            """
최근 대화 맥락:
{conversation_context}

현재 질문: {question}

이전 응답:
{previous_response}

위 응답은 형식이 불완전했습니다.
같은 오류를 반복하지 말고 다시 작성하십시오.

판단:
의심점수:
질문반복점수:
기억혼란점수:
시간혼란점수:
문장비논리점수:
근거:
""",
        ),
    ]
)

repetition_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
당신은 사용자의 현재 질문이 최근 사용자 질문의 반복인지 판정하는 보조 분석기입니다.

오직 질문 반복 여부만 판단하십시오.
기억 혼란, 시간 혼란, 문장 비논리성은 여기서 평가하지 마십시오.

판단 규칙:
- 단어 순서, 조사, 어미가 조금 달라도 핵심 요청이 같으면 같은 의미의 질문일 수 있습니다.
- "기억이 안 난다", "몇 시지", "몇 시일까"처럼 표현 위치만 달라도 핵심 질문이 같으면 반복으로 판단할 수 있습니다.
- 같은 주제만 공유하고 실제 요청이 다르면 반복으로 보지 마십시오.
- 최근 AI 답변 직후 같은 의미 질문이 다시 나오면 더 높은 점수를 줄 수 있습니다.

질문반복점수는 반드시 아래 다섯 값 중 하나만 사용하십시오:
0, 8, 15, 20, 25

출력 형식:
질문반복점수:
반복대상:
근거:
""",
        ),
        (
            "human",
            """
최근 사용자 질문과 당시 AI 답변:
{recent_user_questions}

현재 질문:
{question}

반드시 형식만 출력하세요.
""",
        ),
    ]
)

feature_analysis_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
당신은 대화 기반 인지 위험 징후 분석기입니다.

오직 기억 혼란, 시간/상황 혼란, 문장 비논리성만 평가하십시오.
질문 반복 여부는 이미 별도 분석기가 담당하므로 여기서 점수화하지 마십시오.

중요 규칙:
- 질문에 답변하지 마십시오.
- 조언하지 마십시오.
- 같은 질문이 반복되더라도 반복 자체를 이유로 기억혼란점수, 시간혼란점수, 문장비논리점수를 올리지 마십시오.
- 오직 현재 질문과 대화 맥락에서 실제로 드러난 기억 회상 어려움, 시간·상황 혼란, 문장 논리성만 평가하십시오.
- 근거는 반드시 한국어 문장 2문장 이상으로 작성하십시오.
- 형식 외 다른 문장을 출력하면 실패입니다.

점수 범위:
- 기억혼란점수: 0~25
- 시간혼란점수: 0~30
- 문장비논리점수: 0~20

출력 형식:
기억혼란점수:
시간혼란점수:
문장비논리점수:
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

반드시 위 형식만 출력하세요.
""",
        ),
    ]
)

feature_analysis_retry_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
당신은 대화 기반 인지 위험 징후 분석기입니다.

오직 기억 혼란, 시간/상황 혼란, 문장 비논리성만 평가하십시오.
질문 반복 여부는 이미 별도 분석기가 담당하므로 여기서 점수화하지 마십시오.

이전 응답은 형식이 잘못되었습니다.
형식을 정확히 맞추고, 반복 질문이라는 이유만으로 다른 점수를 자동으로 올리지 마십시오.

출력 형식:
기억혼란점수:
시간혼란점수:
문장비논리점수:
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

반드시 위 형식만 다시 출력하세요.
""",
        ),
    ]
)

ROLE_ANALYSIS_META = {
    "memory": {
        "title": "기억 혼란",
        "score_label": "기억혼란점수",
        "max_score": 25,
        "focus_rule": "기억이 안 난다, 떠오르지 않는다, 방금 들은 내용을 바로 회상하지 못한다는 식의 기억 회상 어려움만 평가하십시오.",
    },
    "time_confusion": {
        "title": "시간/상황 혼란",
        "score_label": "시간혼란점수",
        "max_score": 30,
        "focus_rule": "언제, 몇 시, 오늘 일정, 현재 상황 같은 시간·상황 정보를 혼동하거나 바로 떠올리지 못하는 표현만 평가하십시오.",
    },
    "incoherence": {
        "title": "문장 비논리성",
        "score_label": "문장비논리점수",
        "max_score": 20,
        "focus_rule": "문장 전개가 비약적이거나 앞뒤 논리가 약한 경우만 평가하십시오.",
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
당신은 대화 기반 인지 위험 징후 분석기입니다.

오직 {meta["title"]}만 평가하십시오.
질문 반복, 다른 역할 점수, 전체 판단은 여기서 평가하지 마십시오.

평가 규칙:
- 질문에 답변하지 마십시오.
- 조언하지 마십시오.
- {meta["focus_rule"]}
- 다른 특징이 보여도 {meta["title"]}과 직접 관련이 없으면 점수에 반영하지 마십시오.
- 근거는 반드시 한국어 문장 2문장 이상으로 작성하십시오.

점수 범위:
- {meta["score_label"]}: 0~{meta["max_score"]}

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

반드시 위 형식만 출력하세요.
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
당신은 대화 기반 인지 위험 징후 분석기입니다.

오직 {meta["title"]}만 평가하십시오.
이전 응답은 형식이 잘못되었습니다.

중요:
- 질문 반복이나 다른 점수는 고려하지 마십시오.
- 형식 외 다른 문장을 출력하지 마십시오.

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

반드시 위 형식만 다시 출력하세요.
""",
            ),
        ]
    )


ROLE_ANALYSIS_PROMPTS = {
    role_key: build_role_prompt(role_key) for role_key in ROLE_ANALYSIS_META
}

ROLE_ANALYSIS_RETRY_PROMPTS = {
    role_key: build_role_retry_prompt(role_key) for role_key in ROLE_ANALYSIS_META
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
    return int(os.getenv("PORT", "5000"))


def get_waitress_threads() -> int:
    return int(os.getenv("WAITRESS_THREADS", "8"))
