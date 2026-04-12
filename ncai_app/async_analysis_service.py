import logging
from datetime import datetime, timezone

from . import runtime
from .analysis_service import get_response_from_llama
from .history_service import (
    add_score_history,
    add_to_history,
    add_turn_history,
    build_analysis_detail_payload,
    build_empty_analysis_detail_payload,
    evaluate_recall_answer,
    maybe_advance_recall_test,
)
from .llm_service import transcribe_audio_file


_JOB_STORE_MAX = 200
logger = logging.getLogger(__name__)


def update_job(job_id: str, **kwargs) -> None:
    job = runtime.job_store.get(job_id)
    if job is None:
        return

    job.update(kwargs)
    job["updated_at"] = datetime.now(timezone.utc).isoformat()

    if kwargs.get("status") in {"completed", "failed"} and len(runtime.job_store) > _JOB_STORE_MAX:
        completed_ids = [
            existing_job_id
            for existing_job_id, existing_job in runtime.job_store.items()
            if existing_job.get("status") in {"completed", "failed"}
            and existing_job_id != job_id
        ]
        for stale_job_id in completed_ids[: len(runtime.job_store) - _JOB_STORE_MAX]:
            runtime.job_store.pop(stale_job_id, None)


def _build_empty_audio_result(session_id: str, provider: str) -> dict:
    return build_empty_analysis_detail_payload(
        session_id,
        llm_provider=provider,
        reason="음성 인식 결과가 없어 분석을 진행할 수 없습니다.",
        excluded_reason="음성 인식 결과가 없어 이번 분석은 점수 통계에서 제외되었습니다.",
    )


def run_analysis_job(
    job_id: str,
    file_path: str,
    session_id: str,
    provider: str,
) -> None:
    try:
        update_job(job_id, status="running", stage="stt", progress=5, message="음성 인식 중입니다.")

        user_input = transcribe_audio_file(file_path)
        update_job(
            job_id,
            stage="stt",
            progress=20,
            message="음성 인식이 완료되었습니다.",
            partial_text=user_input or "",
        )

        if not user_input:
            update_job(
                job_id,
                status="completed",
                stage="completed",
                progress=100,
                message="분석이 완료되었습니다.",
                result=_build_empty_audio_result(session_id, provider),
            )
            return

        recall_feedback = evaluate_recall_answer(session_id, user_input)

        def progress_callback(stage: str, progress: int, message: str) -> None:
            update_job(job_id, stage=stage, progress=progress, message=message)

        llm_result = get_response_from_llama(
            user_input,
            session_id=session_id,
            provider=provider,
            progress_callback=progress_callback,
        )

        update_job(job_id, stage="finalize", progress=90, message="결과를 정리하는 중입니다.")

        if recall_feedback:
            llm_result["answer"] = f"{llm_result['answer']}\n\n{recall_feedback}"

        add_to_history(session_id, "user", user_input)
        add_to_history(session_id, "assistant", llm_result["full_text"])
        if llm_result.get("score_included", True):
            add_score_history(session_id, llm_result["score"])

        follow_up_messages = []
        recall_prompt = maybe_advance_recall_test(session_id)
        if recall_prompt:
            llm_result["answer"] = f"{llm_result['answer']}\n\n{recall_prompt}"
            llm_result["full_text"] = f"{llm_result['full_text']}\n\n{recall_prompt}"
            follow_up_messages.append(recall_prompt)

        turn = add_turn_history(
            session_id=session_id,
            user_text=user_input,
            answer=llm_result["answer"],
            judgment=llm_result["judgment"],
            score=llm_result["score"],
            reason=llm_result["reason"],
            feature_scores=llm_result["feature_scores"],
            follow_up_messages=follow_up_messages,
            score_included=llm_result.get("score_included", True),
            excluded_reason=llm_result.get("excluded_reason", ""),
            llm_provider=provider,
        )

        result = build_analysis_detail_payload(
            session_id,
            stt_result=user_input,
            answer=llm_result["answer"],
            judgment=llm_result["judgment"],
            reason=llm_result["reason"],
            score_total=llm_result["score"],
            feature_scores=llm_result.get("feature_scores", {}),
            llm_provider=provider,
            turn=turn,
            score_included=llm_result.get("score_included", True),
            excluded_reason=llm_result.get("excluded_reason", ""),
        )

        update_job(
            job_id,
            status="completed",
            stage="completed",
            progress=100,
            message="분석이 완료되었습니다.",
            result=result,
        )
    except Exception:
        logger.exception("analysis job %s failed", job_id)
        update_job(
            job_id,
            status="failed",
            message="분석 중 오류가 발생했습니다.",
            error_message="분석 중 오류가 발생했습니다.",
        )
