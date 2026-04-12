from collections import deque
from threading import Lock


conversation_store = {}
score_store = {}
recall_store = {}
turn_store = {}
session_generation_store = {}

answer_chain = None
analysis_chain = None
analysis_retry_chain = None
analysis_repetition_chain = None
analysis_llm_instance = None
role_analysis_chains = {}
role_analysis_retry_chains = {}
speech_client = None
temp_google_credentials_path = None
analysis_runtime_cache = {}
# Maps session_id -> turn_count at the time of last repair.
# repair_session_analysis_history is called 3+ times per turn (average, recent_avg, trend).
# This cache makes the 2nd+ call within the same turn count a no-op.
repair_cache: dict[str, int] = {}
analysis_llm_lock = Lock()
visitor_lock = Lock()
visitor_event_store = deque(maxlen=300)
visitor_snapshot_store = {}
visitor_hostname_cache: dict[str, str] = {}  # capped at _HOSTNAME_CACHE_MAX entries by routes.py
_HOSTNAME_CACHE_MAX = 500
# Reverse index: (ip, normalized_ua) -> visitor_id  — avoids O(n) scan on every telemetry update
visitor_ip_ua_index: dict[tuple, str] = {}
# job_id -> job state dict for async analysis progress tracking
job_store: dict = {}
