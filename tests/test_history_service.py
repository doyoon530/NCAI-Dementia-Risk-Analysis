"""
Tests for history_service.py — focuses on:
  1. deque auto-truncation of bounded stores
  2. repair_session_analysis_history in-place mutation of deque elements
  3. repair_cache optimization (skip redundant O(n) repairs)

Run with:
    python -m pytest tests/test_history_service.py -v
  or directly:
    python tests/test_history_service.py
"""
import sys
import os
import types

# ---------------------------------------------------------------------------
# Minimal stubs so we can import ncai_app without the real dependencies
# ---------------------------------------------------------------------------

def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _setup_stubs():
    class FakeCPT:
        @staticmethod
        def from_messages(*a, **k):
            return FakeCPT()

    class FakeLLMChain:
        def __init__(self, *a, **k):
            pass

    class FakeLlamaCpp:
        def __init__(self, *a, **k):
            pass

    _stub("flask", jsonify=lambda x: x, request=None, session=None,
          make_response=None, redirect=None, render_template=None)

    lcp = _stub("langchain_core.prompts", ChatPromptTemplate=FakeCPT)
    lc = _stub("langchain_core")
    lc.prompts = lcp

    lcc_chains = _stub("langchain_classic.chains", LLMChain=FakeLLMChain)
    lcc = _stub("langchain_classic")
    lcc.chains = lcc_chains

    lcm_llms = _stub("langchain_community.llms", LlamaCpp=FakeLlamaCpp)
    lcm = _stub("langchain_community")
    lcm.llms = lcm_llms

    gcsp = _stub("google.cloud.speech", SpeechClient=None,
                 RecognitionAudio=None, RecognitionConfig=None)
    gcs = _stub("google.cloud")
    gcs.speech = gcsp
    _stub("google")
    _stub("google.auth")
    _stub("google.auth.transport")
    _stub("google.auth.transport.requests", Request=None)
    _stub("google.oauth2")
    _stub("google.oauth2.id_token", verify_oauth2_token=None)

    _stub("werkzeug")
    _stub("werkzeug.utils", secure_filename=lambda x: x)


_setup_stubs()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from collections import deque  # noqa: E402
from ncai_app import runtime  # noqa: E402
from ncai_app.config import MAX_SCORE_HISTORY, MAX_HISTORY_TURNS  # noqa: E402
from ncai_app.history_service import (  # noqa: E402
    repair_session_analysis_history,
    add_turn_history,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_turn(score: int, score_included: bool = True) -> dict:
    return {
        "turn_id": "t",
        "time": "00:00:00",
        "user_text": "테스트 질문",
        "answer": "AI 답변",
        "judgment": "정상",
        "score": score,
        "reason": f"반복:0 기억:0 시간혼란:0 비논리:{score} 합계:{score}",
        "feature_scores": {
            "repetition": 0,
            "memory": 0,
            "time_confusion": 0,
            "incoherence": score,
        },
        "follow_up_messages": [],
        "score_included": score_included,
        "excluded_reason": "",
        "llm_provider": "api",
    }


def _reset(sid: str) -> None:
    runtime.turn_store[sid] = deque(maxlen=MAX_SCORE_HISTORY)
    runtime.score_store[sid] = deque(maxlen=MAX_SCORE_HISTORY)
    runtime.conversation_store[sid] = deque(maxlen=MAX_HISTORY_TURNS * 2)
    runtime.repair_cache.pop(sid, None)
    runtime.analysis_runtime_cache.pop(sid, None)


# ---------------------------------------------------------------------------
# 1. deque auto-truncation
# ---------------------------------------------------------------------------

class TestDequeAutoTruncation:
    def test_turn_store_capped_at_max(self):
        sid = "dt-turn"
        _reset(sid)
        for i in range(MAX_SCORE_HISTORY + 10):
            runtime.turn_store[sid].append(_make_turn(i))
        assert len(runtime.turn_store[sid]) == MAX_SCORE_HISTORY

    def test_score_store_capped_at_max(self):
        sid = "dt-score"
        _reset(sid)
        for i in range(MAX_SCORE_HISTORY + 10):
            runtime.score_store[sid].append({"score": i, "time": "x"})
        assert len(runtime.score_store[sid]) == MAX_SCORE_HISTORY

    def test_conversation_store_capped(self):
        sid = "dt-conv"
        _reset(sid)
        cap = MAX_HISTORY_TURNS * 2
        for i in range(cap + 5):
            runtime.conversation_store[sid].append({"role": "user", "content": str(i)})
        assert len(runtime.conversation_store[sid]) == cap

    def test_deque_retains_newest_entries(self):
        """Overflow should drop OLDEST entries and keep the newest MAX."""
        sid = "dt-order"
        _reset(sid)
        total = MAX_SCORE_HISTORY + 5
        for i in range(total):
            runtime.turn_store[sid].append(_make_turn(i))
        turns = list(runtime.turn_store[sid])
        assert turns[0]["score"] == 5          # oldest kept
        assert turns[-1]["score"] == total - 1  # newest


# ---------------------------------------------------------------------------
# 2. repair_session_analysis_history — in-place deque mutation
# ---------------------------------------------------------------------------

class TestRepairInPlace:
    def test_same_object_identity(self):
        """repair must mutate the dict in the deque, not replace it."""
        sid = "rip-identity"
        _reset(sid)
        turn = _make_turn(10, True)
        runtime.turn_store[sid].append(turn)

        repair_session_analysis_history(sid)

        repaired = list(runtime.turn_store[sid])[0]
        assert repaired is turn, "Expected in-place mutation but got a different object"

    def test_risk_level_added(self):
        sid = "rip-risk"
        _reset(sid)
        runtime.turn_store[sid].append(_make_turn(10, True))
        repair_session_analysis_history(sid)
        assert "risk_level" in list(runtime.turn_store[sid])[0]

    def test_excluded_turn_not_in_score_store(self):
        sid = "rip-excl"
        _reset(sid)
        runtime.turn_store[sid].append(_make_turn(50, score_included=False))
        repair_session_analysis_history(sid)
        assert len(runtime.score_store[sid]) == 0

    def test_included_turn_in_score_store(self):
        sid = "rip-incl"
        _reset(sid)
        runtime.turn_store[sid].append(_make_turn(20, True))
        repair_session_analysis_history(sid)
        assert len(runtime.score_store[sid]) == 1
        assert list(runtime.score_store[sid])[0]["score"] == 20

    def test_score_store_remains_deque_after_repair(self):
        """repair reassigns score_store — must be deque, not list."""
        sid = "rip-type"
        _reset(sid)
        runtime.turn_store[sid].append(_make_turn(5, True))
        repair_session_analysis_history(sid)
        store = runtime.score_store[sid]
        assert isinstance(store, deque), f"Expected deque, got {type(store)}"
        assert store.maxlen == MAX_SCORE_HISTORY


# ---------------------------------------------------------------------------
# 3. repair_cache — skip duplicate repairs within same turn count
# ---------------------------------------------------------------------------

class TestRepairCache:
    def test_cache_populated_after_first_repair(self):
        sid = "rc-populate"
        _reset(sid)
        runtime.turn_store[sid].append(_make_turn(10, True))
        repair_session_analysis_history(sid)
        assert runtime.repair_cache.get(sid) == 1

    def test_second_repair_skipped_on_same_turn_count(self):
        """With no new turns, the second repair call must be a no-op."""
        sid = "rc-skip"
        _reset(sid)
        runtime.turn_store[sid].append(_make_turn(10, True))
        repair_session_analysis_history(sid)          # first call — runs repair

        # Insert a sentinel that a real repair would overwrite
        runtime.score_store[sid].append({"score": 999, "time": "sentinel"})
        repair_session_analysis_history(sid)          # second call — must be skipped

        assert list(runtime.score_store[sid])[-1]["score"] == 999, (
            "repair ran again when it should have been skipped via cache"
        )

    def test_cache_invalidated_by_add_turn_history(self):
        """add_turn_history must clear repair_cache so the next read re-runs repair."""
        sid = "rc-inval"
        _reset(sid)
        runtime.repair_cache[sid] = 99  # pre-seed with a stale value

        add_turn_history(
            sid, "질문", "답변", "정상", 5, "근거",
            {"repetition": 0, "memory": 0, "time_confusion": 0, "incoherence": 5},
        )

        cached = runtime.repair_cache.get(sid, "MISSING")
        assert cached != 99, (
            f"repair_cache was not invalidated after add_turn_history (got {cached!r})"
        )


# ---------------------------------------------------------------------------
# Standalone runner (no pytest required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    suites = [
        TestDequeAutoTruncation,
        TestRepairInPlace,
        TestRepairCache,
    ]

    passed = failed = 0
    for suite_cls in suites:
        print(f"\n=== {suite_cls.__name__} ===")
        suite = suite_cls()
        for name in [n for n in dir(suite_cls) if n.startswith("test_")]:
            try:
                getattr(suite, name)()
                print(f"  PASS  {name}")
                passed += 1
            except Exception:
                print(f"  FAIL  {name}")
                traceback.print_exc()
                failed += 1

    print(f"\nResult: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
