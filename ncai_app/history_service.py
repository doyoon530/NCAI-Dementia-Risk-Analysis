import random
import uuid
from datetime import datetime

from flask import jsonify, request

from . import runtime
from .common import clamp_score, clamp_subscore, normalize_text
from .config import (
    ANALYSIS_CONTEXT_TURNS,
    MAX_HISTORY_TURNS,
    MAX_SCORE_HISTORY,
    RECALL_WORDS,
    RECENT_WINDOW,
    REPETITION_CONTEXT_TURNS,
    normalize_llm_provider,
)


def get_or_create_session_id() -> str:
    session_id = request.headers.get("X-Session-Id") or request.args.get("session_id")

    if not session_id:
        session_id = str(uuid.uuid4())

    runtime.conversation_store.setdefault(session_id, [])
    runtime.score_store.setdefault(session_id, [])
    runtime.turn_store.setdefault(session_id, [])
    runtime.recall_store.setdefault(
        session_id,
        {
            "status": "idle",
            "target_word": "",
            "prompt": "",
            "last_result": "없음",
            "introduced_turn": 0,
        },
    )
    return session_id


def add_to_history(session_id: str, role: str, content: str) -> None:
    runtime.conversation_store.setdefault(session_id, [])
    runtime.conversation_store[session_id].append({"role": role, "content": content})

    max_messages = MAX_HISTORY_TURNS * 2
    if len(runtime.conversation_store[session_id]) > max_messages:
        runtime.conversation_store[session_id] = runtime.conversation_store[session_id][
            -max_messages:
        ]


def build_analysis_context_from_turns(
    turns, max_turns: int = ANALYSIS_CONTEXT_TURNS
) -> str:
    if not turns:
        return "이전 대화 없음"

    recent_turns = turns[-max_turns:]
    context_lines = []

    for index, turn in enumerate(recent_turns, start=1):
        user_text = normalize_text(turn.get("user_text", ""))
        answer_text = normalize_text(turn.get("answer", ""))

        if not user_text and not answer_text:
            continue

        context_lines.append(f"[이전 대화 {index}]")
        if user_text:
            context_lines.append(f"사용자: {user_text}")
        if answer_text:
            context_lines.append(f"AI: {answer_text}")

    return "\n".join(context_lines) if context_lines else "이전 대화 없음"


def build_analysis_context(
    session_id: str | None, max_turns: int = ANALYSIS_CONTEXT_TURNS
) -> str:
    if not session_id:
        return "이전 대화 없음"

    turns = runtime.turn_store.get(session_id, [])
    return build_analysis_context_from_turns(turns, max_turns=max_turns)


def get_recent_user_turns(turns, max_turns: int = REPETITION_CONTEXT_TURNS):
    if not turns:
        return []

    recent_turns = []
    for turn in turns:
        user_text = normalize_text(turn.get("user_text", ""))
        if not user_text:
            continue

        recent_turns.append(
            {"user_text": user_text, "answer": normalize_text(turn.get("answer", ""))}
        )

    return recent_turns[-max_turns:] if len(recent_turns) > max_turns else recent_turns


def get_analysis_runtime_state(session_id: str | None) -> dict:
    if not session_id:
        return {
            "analysis_context": "이전 대화 없음",
            "previous_turns": [],
            "turn_count": 0,
        }

    turns = runtime.turn_store.get(session_id, [])
    turn_count = len(turns)
    cached = runtime.analysis_runtime_cache.get(session_id)
    if cached and cached.get("turn_count") == turn_count:
        return cached

    runtime_state = {
        "analysis_context": build_analysis_context_from_turns(turns),
        "previous_turns": get_recent_user_turns(turns),
        "turn_count": turn_count,
    }
    runtime.analysis_runtime_cache[session_id] = runtime_state
    return runtime_state


def add_turn_history(
    session_id: str,
    user_text: str,
    answer: str,
    judgment: str,
    score: int,
    reason: str,
    feature_scores: dict,
    follow_up_messages=None,
    score_included: bool = True,
    excluded_reason: str = "",
    llm_provider: str = "local",
) -> dict:
    runtime.turn_store.setdefault(session_id, [])

    turn = {
        "turn_id": str(uuid.uuid4()),
        "time": datetime.now().strftime("%H:%M:%S"),
        "user_text": user_text,
        "answer": answer,
        "judgment": judgment,
        "score": clamp_score(score),
        "reason": reason,
        "feature_scores": {
            "repetition": int(feature_scores.get("repetition", 0)),
            "memory": int(feature_scores.get("memory", 0)),
            "time_confusion": int(feature_scores.get("time_confusion", 0)),
            "incoherence": int(feature_scores.get("incoherence", 0)),
        },
        "follow_up_messages": follow_up_messages or [],
        "score_included": bool(score_included),
        "excluded_reason": str(excluded_reason or ""),
        "llm_provider": normalize_llm_provider(llm_provider),
    }

    runtime.turn_store[session_id].append(turn)
    if len(runtime.turn_store[session_id]) > MAX_SCORE_HISTORY:
        runtime.turn_store[session_id] = runtime.turn_store[session_id][-MAX_SCORE_HISTORY:]

    turn["average_score"] = get_average_score(session_id)
    turn["recent_average_score"] = get_recent_average_score(session_id)

    if turn["score_included"]:
        turn["risk_level"] = get_risk_level_from_score(turn["score"])
        turn["trend"] = get_score_trend(session_id)
        turn["confidence"] = calculate_confidence_from_feature_scores(
            turn["feature_scores"], turn["score"]
        )
    else:
        turn["risk_level"] = "반영 제외"
        turn["trend"] = "반영 제외"
        turn["confidence"] = 0

    return turn


def calculate_trend_from_score_values(scores, window: int = RECENT_WINDOW) -> str:
    if len(scores) < 2:
        return "데이터 부족"

    recent = scores[-window:]
    if len(recent) < 2:
        return "안정"

    diff = recent[-1] - recent[0]
    if diff >= 10:
        return "상승"
    if diff <= -10:
        return "하락"
    return "안정"


def repair_session_analysis_history(session_id: str) -> None:
    from .analysis_service import (
        infer_judgment_from_score,
        normalize_reason_text,
        parse_analysis_scores,
    )

    turns = runtime.turn_store.get(session_id, [])
    if not turns:
        return

    existing_scores = runtime.score_store.get(session_id, [])
    running_scores = []

    for index, turn in enumerate(turns):
        feature_scores = turn.get("feature_scores") or {}
        repaired_feature_scores = {
            "repetition": clamp_subscore(int(feature_scores.get("repetition", 0)), 25),
            "memory": clamp_subscore(int(feature_scores.get("memory", 0)), 25),
            "time_confusion": clamp_subscore(
                int(feature_scores.get("time_confusion", 0)), 30
            ),
            "incoherence": clamp_subscore(
                int(feature_scores.get("incoherence", 0)), 20
            ),
        }

        current_score = clamp_score(int(turn.get("score", 0)))
        current_subtotal = sum(repaired_feature_scores.values())
        parsed_from_reason = parse_analysis_scores(turn.get("reason", ""))
        score_included = turn.get("score_included")
        if score_included is None:
            score_included = should_include_analysis_score(
                turn.get("judgment", ""), current_score, repaired_feature_scores
            )
        score_included = bool(score_included)

        if (
            score_included
            and parsed_from_reason["total"] > 0
            and (current_score == 0 or current_subtotal == 0)
        ):
            repaired_feature_scores = {
                "repetition": parsed_from_reason["repetition"],
                "memory": parsed_from_reason["memory"],
                "time_confusion": parsed_from_reason["time_confusion"],
                "incoherence": parsed_from_reason["incoherence"],
            }
            current_score = parsed_from_reason["total"]
        elif (
            score_included
            and current_subtotal > 0
            and current_score != clamp_score(current_subtotal)
        ):
            current_score = clamp_score(current_subtotal)

        judgment = str(turn.get("judgment", "")).strip()
        if judgment not in {"정상", "의심", "판단 어려움"}:
            judgment = infer_judgment_from_score(current_score)
        if score_included and (
            (judgment == "정상" and current_score >= 20)
            or (judgment == "의심" and current_score < 20)
        ):
            judgment = infer_judgment_from_score(current_score)

        turn["feature_scores"] = repaired_feature_scores
        turn["score"] = current_score if score_included else 0
        turn["judgment"] = judgment
        turn["score_included"] = score_included
        turn["excluded_reason"] = str(
            turn.get("excluded_reason")
            or build_score_exclusion_reason(
                judgment, turn["score"], turn.get("reason", ""), repaired_feature_scores
            )
        )
        turn["reason"] = normalize_reason_text(
            turn.get("reason", ""), repaired_feature_scores
        )

        if score_included:
            turn["confidence"] = calculate_confidence_from_feature_scores(
                repaired_feature_scores, current_score
            )
            turn["risk_level"] = get_risk_level_from_score(current_score)
            running_scores.append(current_score)
            turn["trend"] = calculate_trend_from_score_values(running_scores)
        else:
            turn["confidence"] = 0
            turn["risk_level"] = "반영 제외"
            turn["trend"] = "반영 제외"

        if running_scores:
            turn["average_score"] = round(sum(running_scores) / len(running_scores), 1)
            recent_scores = running_scores[-RECENT_WINDOW:]
            turn["recent_average_score"] = round(
                sum(recent_scores) / len(recent_scores), 1
            )
        else:
            turn["average_score"] = 0.0
            turn["recent_average_score"] = 0.0

    repaired_score_history = []
    for index, turn in enumerate(turns):
        if not turn.get("score_included", True):
            continue

        if index < len(existing_scores):
            time_value = existing_scores[index].get("time", turn.get("time", ""))
        else:
            time_value = turn.get("time", "")

        repaired_score_history.append(
            {"score": clamp_score(int(turn.get("score", 0))), "time": time_value}
        )

    runtime.score_store[session_id] = repaired_score_history[-MAX_SCORE_HISTORY:]


def get_turn_history(session_id: str):
    repair_session_analysis_history(session_id)
    return runtime.turn_store.get(session_id, [])


def get_user_turn_count(session_id: str) -> int:
    history = runtime.conversation_store.get(session_id, [])
    return sum(1 for item in history if item["role"] == "user")


def add_score_history(session_id: str, score: int) -> None:
    runtime.score_store.setdefault(session_id, [])
    runtime.score_store[session_id].append(
        {"score": clamp_score(score), "time": datetime.now().strftime("%H:%M:%S")}
    )

    if len(runtime.score_store[session_id]) > MAX_SCORE_HISTORY:
        runtime.score_store[session_id] = runtime.score_store[session_id][
            -MAX_SCORE_HISTORY:
        ]


def get_score_history(session_id: str):
    repair_session_analysis_history(session_id)
    return runtime.score_store.get(session_id, [])


def get_average_score(session_id: str) -> float:
    history = get_score_history(session_id)
    if not history:
        return 0.0

    avg = sum(item["score"] for item in history) / len(history)
    return round(avg, 1)


def get_recent_average_score(session_id: str, window: int = RECENT_WINDOW) -> float:
    history = get_score_history(session_id)
    if not history:
        return 0.0

    recent = history[-window:]
    avg = sum(item["score"] for item in recent) / len(recent)
    return round(avg, 1)


def calculate_confidence_from_feature_scores(
    feature_scores: dict, total_score: int
) -> int:
    repetition = int(feature_scores.get("repetition", 0))
    memory = int(feature_scores.get("memory", 0))
    time_confusion = int(feature_scores.get("time_confusion", 0))
    incoherence = int(feature_scores.get("incoherence", 0))

    confidence = 55
    if memory > 0:
        confidence += 8
    if time_confusion > 0:
        confidence += 8
    if repetition > 0:
        confidence += 6
    if incoherence > 0:
        confidence += 6
    if total_score >= 40:
        confidence += 8
    if total_score >= 60:
        confidence += 4

    return max(0, min(95, confidence))


def has_meaningful_feature_scores(feature_scores: dict) -> bool:
    if not isinstance(feature_scores, dict):
        return False

    return any(
        int(feature_scores.get(key, 0)) > 0
        for key in ["repetition", "memory", "time_confusion", "incoherence"]
    )


def should_include_analysis_score(
    judgment: str, score: int, feature_scores: dict
) -> bool:
    normalized_judgment = str(judgment or "").strip()
    normalized_score = clamp_score(int(score or 0))

    if (
        normalized_judgment == "판단 어려움"
        and normalized_score == 0
        and not has_meaningful_feature_scores(feature_scores)
    ):
        return False

    return True


def build_score_exclusion_reason(
    judgment: str, score: int, reason: str, feature_scores: dict
) -> str:
    if should_include_analysis_score(judgment, score, feature_scores):
        return ""

    normalized_reason = str(reason or "").strip()
    if "너무 짧아" in normalized_reason or "입력이 필요" in normalized_reason:
        return "입력이 너무 짧아 이번 대화는 점수 통계에서 제외했습니다."
    if "음성 인식 결과" in normalized_reason:
        return "음성 인식 결과가 없어 이번 대화는 점수 통계에서 제외했습니다."
    if "입력된 대화가 없습니다" in normalized_reason:
        return "분석할 대화가 없어 이번 대화는 점수 통계에서 제외했습니다."
    if "문제가 발생" in normalized_reason or "오류" in normalized_reason:
        return "분석 중 오류가 발생해 이번 대화는 점수 통계에서 제외했습니다."

    return "분석 결과가 불안정하여 이번 대화는 점수 통계에서 제외했습니다."


def get_risk_level_from_score(score: float) -> str:
    if score < 20:
        return "Normal"
    if score < 40:
        return "Low Risk"
    if score < 60:
        return "Moderate Risk"
    if score < 80:
        return "High Risk"
    return "Very High Risk"


def get_score_trend(session_id: str, window: int = RECENT_WINDOW) -> str:
    history = get_score_history(session_id)
    return calculate_trend_from_score_values([item["score"] for item in history], window)


def get_recall_state(session_id: str) -> dict:
    return runtime.recall_store.setdefault(
        session_id,
        {
            "status": "idle",
            "target_word": "",
            "prompt": "",
            "last_result": "없음",
            "introduced_turn": 0,
        },
    )


def evaluate_recall_answer(session_id: str, user_input: str) -> str:
    state = get_recall_state(session_id)
    if state["status"] != "ask":
        return ""

    target_word = state["target_word"].strip()
    normalized_input = normalize_text(user_input)

    if target_word and target_word in normalized_input:
        feedback = f"Recall Test 결과: 정답입니다. 기억 단어는 '{target_word}'였습니다."
        state["last_result"] = "정답"
    else:
        feedback = f"Recall Test 결과: 오답입니다. 기억 단어는 '{target_word}'였습니다."
        state["last_result"] = "오답"

    state["status"] = "idle"
    state["prompt"] = ""
    state["target_word"] = ""
    state["introduced_turn"] = 0
    return feedback


def maybe_advance_recall_test(session_id: str) -> str:
    state = get_recall_state(session_id)
    user_turn_count = get_user_turn_count(session_id)

    if state["status"] == "idle" and user_turn_count >= 3 and user_turn_count % 3 == 0:
        target_word = random.choice(RECALL_WORDS)
        state["status"] = "memorize"
        state["target_word"] = target_word
        state["introduced_turn"] = user_turn_count
        state["prompt"] = (
            f"Recall Test: 지금 제시하는 기억 단어는 '{target_word}'입니다. "
            "다음 대화에서 이 단어를 기억해 보세요."
        )
        return state["prompt"]

    if state["status"] == "memorize" and user_turn_count == state["introduced_turn"] + 1:
        state["status"] = "ask"
        state["prompt"] = "Recall Test: 제가 조금 전에 제시한 기억 단어가 무엇이었나요?"
        return state["prompt"]

    return ""


def serialize_recall_state(session_id: str) -> dict:
    state = get_recall_state(session_id)
    return {
        "status": state["status"],
        "prompt": state["prompt"],
        "last_result": state["last_result"],
        "target_word": state["target_word"] if state["status"] == "memorize" else "",
    }


def build_chat_response(
    session_id: str,
    user_speech: str,
    sys_response: str,
    answer: str,
    judgment: str,
    score: int,
    reason: str,
    feature_scores: dict,
    follow_up_messages=None,
    turn=None,
    score_included: bool = True,
    excluded_reason: str = "",
    llm_provider: str = "local",
):
    recent_average_score = get_recent_average_score(session_id)
    risk_level = get_risk_level_from_score(recent_average_score)
    trend = get_score_trend(session_id)

    return jsonify(
        {
            "session_id": session_id,
            "user_speech": user_speech,
            "sys_response": sys_response,
            "answer": answer,
            "follow_up_messages": follow_up_messages or [],
            "judgment": judgment,
            "score": clamp_score(score),
            "reason": reason,
            "feature_scores": feature_scores,
            "llm_provider": normalize_llm_provider(llm_provider),
            "score_included": bool(score_included),
            "excluded_reason": str(excluded_reason or ""),
            "average_score": get_average_score(session_id),
            "recent_average_score": recent_average_score,
            "risk_level": risk_level,
            "trend": trend,
            "score_history": get_score_history(session_id),
            "turn_history": get_turn_history(session_id),
            "turn": turn,
            "recall": serialize_recall_state(session_id),
        }
    )


def finalize_analysis_response(
    session_id: str,
    user_input: str,
    answer_text: str,
    fields: dict,
    llm_provider: str = "local",
):
    from .analysis_service import build_full_text

    recall_feedback = evaluate_recall_answer(session_id, user_input)
    add_to_history(session_id, "user", user_input)
    follow_up_messages = []

    if recall_feedback:
        follow_up_messages.append(recall_feedback)

    recall_prompt = maybe_advance_recall_test(session_id)
    if recall_prompt:
        follow_up_messages.append(recall_prompt)

    full_text = build_full_text(answer_text, fields)
    history_full_text = full_text
    if follow_up_messages:
        history_full_text = f"{history_full_text}\n\n" + "\n\n".join(follow_up_messages)

    add_to_history(session_id, "assistant", history_full_text)
    if fields.get("score_included", True):
        add_score_history(session_id, fields["score"])

    turn = add_turn_history(
        session_id=session_id,
        user_text=user_input,
        answer=answer_text,
        judgment=fields["judgment"],
        score=fields["score"],
        reason=fields["reason"],
        feature_scores=fields["feature_scores"],
        follow_up_messages=follow_up_messages,
        score_included=fields.get("score_included", True),
        excluded_reason=fields.get("excluded_reason", ""),
        llm_provider=llm_provider,
    )
    runtime.analysis_runtime_cache.pop(session_id, None)

    return build_chat_response(
        session_id=session_id,
        user_speech=user_input,
        sys_response=full_text,
        answer=answer_text,
        judgment=fields["judgment"],
        score=fields["score"],
        reason=fields["reason"],
        feature_scores=fields["feature_scores"],
        follow_up_messages=follow_up_messages,
        turn=turn,
        score_included=fields.get("score_included", True),
        excluded_reason=fields.get("excluded_reason", ""),
        llm_provider=llm_provider,
    )
