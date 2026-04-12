import json
import threading
import time
import uuid
from datetime import datetime, timezone

from flask import Response, jsonify, request

from . import runtime
from .analysis_service import (
    build_fields_from_role_results,
    generate_analysis_result,
    generate_answer_result,
    generate_role_analysis_result,
    get_response_from_llama,
    normalize_role_results_payload,
)
from .async_analysis_service import run_analysis_job
from .audio_upload_service import save_audio_upload
from .common import normalize_text
from .config import (
    UPLOAD_DIR,
    get_allowed_audio_extensions,
    get_api_rate_limit,
    get_rate_limit_window_seconds,
    normalize_role_key,
)
from .history_service import (
    add_score_history,
    add_to_history,
    add_turn_history,
    bump_analysis_generation,
    build_analysis_detail_payload,
    build_analysis_metrics_payload,
    build_chat_response,
    build_empty_analysis_detail_payload,
    evaluate_recall_answer,
    finalize_analysis_response,
    get_analysis_generation,
    get_or_create_session_id,
    get_requested_analysis_generation,
    is_current_analysis_generation,
    maybe_advance_recall_test,
    reset_session,
)
from .llm_service import get_requested_llm_provider, transcribe_audio_file
from .security_service import enforce_rate_limit


def register_analysis_routes(
    app,
    *,
    rate_limit_store,
    require_api_access,
    build_stale_generation_response,
):
    def build_empty_chat_response(session_id: str, reason: str, excluded_reason: str):
        return build_chat_response(
            session_id=session_id,
            user_speech="",
            sys_response=(
                "판단: 판단 어려움\n"
                "최종점수: 반영 제외\n"
                "질문반복점수: 0\n"
                "기억혼란점수: 0\n"
                "시간혼란점수: 0\n"
                "문장비논리점수: 0\n"
                f"근거: {reason}"
            ),
            answer="",
            judgment="판단 어려움",
            score=0,
            reason=reason,
            feature_scores={
                "repetition": 0,
                "memory": 0,
                "time_confusion": 0,
                "incoherence": 0,
            },
            score_included=False,
            excluded_reason=excluded_reason,
        )

    def build_empty_stt_analysis_response(session_id: str, provider: str):
        return jsonify(
            build_empty_analysis_detail_payload(
                session_id,
                llm_provider=provider,
                reason="음성 인식 결과가 없어 분석을 진행할 수 없습니다.",
                excluded_reason="음성 인식 결과가 없어 이번 분석은 점수 통계에서 제외되었습니다.",
            )
        )

    @app.route("/transcribe-audio", methods=["POST"])
    def transcribe_audio():
        denied = require_api_access()
        if denied:
            return denied

        limited = enforce_rate_limit(
            store=rate_limit_store,
            request=request,
            bucket="stt",
            limit=get_api_rate_limit(),
            window_seconds=get_rate_limit_window_seconds(),
        )
        if limited:
            return limited

        session_id = get_or_create_session_id()

        try:
            requested_generation = get_requested_analysis_generation()
            if not is_current_analysis_generation(session_id, requested_generation):
                return build_stale_generation_response(session_id)

            if "audio" not in request.files:
                return jsonify({"error": "오디오 파일이 없습니다."}), 400

            audio_file = request.files["audio"]
            if audio_file.filename == "":
                return jsonify({"error": "오디오 파일이 없습니다."}), 400

            file_path = save_audio_upload(
                audio_file, UPLOAD_DIR, get_allowed_audio_extensions()
            )
            if not file_path:
                return jsonify({"error": "unsupported audio format"}), 400

            user_input = transcribe_audio_file(file_path)

            if not is_current_analysis_generation(session_id, requested_generation):
                return build_stale_generation_response(session_id)

            return jsonify(
                {
                    "session_id": session_id,
                    "analysis_generation": get_analysis_generation(session_id),
                    "user_speech": user_input,
                }
            )
        except Exception:
            app.logger.exception("STT 처리 오류")
            return jsonify({"error": "음성 인식 중 문제가 발생했습니다."}), 500

    @app.route("/generate-answer", methods=["POST"])
    def generate_answer():
        denied = require_api_access()
        if denied:
            return denied

        session_id = get_or_create_session_id()

        try:
            data = request.get_json(silent=True) or {}
            user_input = normalize_text(data.get("message", ""))
            llm_provider = get_requested_llm_provider(data)
            requested_generation = get_requested_analysis_generation(data)

            if not user_input:
                return jsonify({"error": "분석할 텍스트가 없습니다."}), 400
            if not is_current_analysis_generation(session_id, requested_generation):
                return build_stale_generation_response(session_id)

            result = generate_answer_result(user_input, provider=llm_provider)

            if not is_current_analysis_generation(session_id, requested_generation):
                return build_stale_generation_response(session_id)

            return jsonify(
                {
                    "session_id": session_id,
                    "analysis_generation": get_analysis_generation(session_id),
                    "user_speech": user_input,
                    "answer": result.get("answer", ""),
                    "is_answer_only": True,
                    "llm_provider": llm_provider,
                }
            )
        except Exception:
            app.logger.exception("답변 생성 오류")
            return jsonify({"error": "답변 생성 중 문제가 발생했습니다."}), 500

    @app.route("/analyze-role", methods=["POST"])
    def analyze_role():
        denied = require_api_access()
        if denied:
            return denied

        session_id = get_or_create_session_id()

        try:
            data = request.get_json(silent=True) or {}
            user_input = normalize_text(data.get("message", ""))
            role_key = normalize_role_key(data.get("role", ""))
            llm_provider = get_requested_llm_provider(data)
            requested_generation = get_requested_analysis_generation(data)

            if not user_input:
                return jsonify({"error": "분석할 텍스트가 없습니다."}), 400
            if not role_key:
                return jsonify({"error": "유효한 분석 역할이 필요합니다."}), 400
            if not is_current_analysis_generation(session_id, requested_generation):
                return build_stale_generation_response(session_id)

            result = generate_role_analysis_result(
                user_input,
                role_key=role_key,
                session_id=session_id,
                provider=llm_provider,
            )

            if not is_current_analysis_generation(session_id, requested_generation):
                return build_stale_generation_response(session_id)

            return jsonify(
                {
                    "session_id": session_id,
                    "analysis_generation": get_analysis_generation(session_id),
                    "role": role_key,
                    "subscore": result.get("subscore", 0),
                    "reason": result.get("reason", ""),
                    "llm_provider": llm_provider,
                }
            )
        except Exception:
            app.logger.exception("역할 분석 오류")
            return jsonify({"error": "역할 분석 중 문제가 발생했습니다."}), 500

    @app.route("/finalize-analysis", methods=["POST"])
    def finalize_analysis():
        denied = require_api_access()
        if denied:
            return denied

        session_id = get_or_create_session_id()

        try:
            data = request.get_json(silent=True) or {}
            user_input = normalize_text(data.get("message", ""))
            answer_text = normalize_text(data.get("answer", ""))
            llm_provider = get_requested_llm_provider(data)
            requested_generation = get_requested_analysis_generation(data)
            role_results_payload = data.get("role_results")

            if not user_input:
                return jsonify({"error": "분석할 텍스트가 없습니다."}), 400
            if not answer_text:
                return jsonify({"error": "최종 답변이 없습니다."}), 400
            if not is_current_analysis_generation(session_id, requested_generation):
                return build_stale_generation_response(session_id)

            normalized_role_results = normalize_role_results_payload(role_results_payload)
            if normalized_role_results:
                fields = build_fields_from_role_results(normalized_role_results)
            else:
                fields = generate_analysis_result(
                    user_input,
                    session_id=session_id,
                    provider=llm_provider,
                )

            if not is_current_analysis_generation(session_id, requested_generation):
                return build_stale_generation_response(session_id)

            return finalize_analysis_response(
                session_id=session_id,
                user_input=user_input,
                answer_text=answer_text,
                fields=fields,
                llm_provider=llm_provider,
            )
        except Exception:
            app.logger.exception("분석 결과 확정 오류")
            return jsonify({"error": "분석 결과를 정리하는 중 문제가 발생했습니다."}), 500

    @app.route("/analyze-text", methods=["POST"])
    def analyze_text():
        denied = require_api_access()
        if denied:
            return denied

        session_id = get_or_create_session_id()

        try:
            data = request.get_json(silent=True) or {}
            user_input = normalize_text(data.get("message", ""))
            llm_provider = get_requested_llm_provider(data)
            requested_generation = get_requested_analysis_generation(data)
            precomputed_answer = normalize_text(data.get("answer", ""))

            if not user_input:
                return jsonify({"error": "분석할 텍스트가 없습니다."}), 400
            if not is_current_analysis_generation(session_id, requested_generation):
                return build_stale_generation_response(session_id)

            answer_result = (
                {"answer": precomputed_answer}
                if precomputed_answer
                else generate_answer_result(user_input, provider=llm_provider)
            )
            fields = generate_analysis_result(
                user_input,
                session_id=session_id,
                provider=llm_provider,
            )

            if not is_current_analysis_generation(session_id, requested_generation):
                return build_stale_generation_response(session_id)

            return finalize_analysis_response(
                session_id=session_id,
                user_input=user_input,
                answer_text=answer_result.get("answer", ""),
                fields=fields,
                llm_provider=llm_provider,
            )
        except Exception:
            app.logger.exception("텍스트 분석 오류")
            return jsonify({"error": "텍스트 분석 중 문제가 발생했습니다."}), 500

    @app.route("/chat", methods=["POST"])
    def chat():
        denied = require_api_access()
        if denied:
            return denied

        limited = enforce_rate_limit(
            store=rate_limit_store,
            request=request,
            bucket="chat",
            limit=get_api_rate_limit(),
            window_seconds=get_rate_limit_window_seconds(),
        )
        if limited:
            return limited

        session_id = get_or_create_session_id()
        user_input = ""
        llm_provider = get_requested_llm_provider()

        try:
            if "audio" in request.files:
                audio_file = request.files["audio"]
                if audio_file.filename == "":
                    return jsonify({"error": "오디오 파일이 없습니다."}), 400

                file_path = save_audio_upload(
                    audio_file, UPLOAD_DIR, get_allowed_audio_extensions()
                )
                if not file_path:
                    return jsonify({"error": "unsupported audio format"}), 400

                user_input = transcribe_audio_file(file_path)
                if not user_input:
                    return build_empty_chat_response(
                        session_id,
                        "음성 인식 결과가 없어 분석을 진행할 수 없습니다. 다시 녹음해 주세요.",
                        "음성 인식 결과가 없어 이번 분석은 점수 통계에서 제외되었습니다.",
                    )
            else:
                data = request.get_json(silent=True) or {}
                user_input = normalize_text(data.get("message", ""))
                llm_provider = get_requested_llm_provider(data)

                if not user_input:
                    return build_empty_chat_response(
                        session_id,
                        "입력된 텍스트가 없어 분석을 진행할 수 없습니다.",
                        "분석할 텍스트가 없어 이번 분석은 점수 통계에서 제외되었습니다.",
                    )

            recall_feedback = evaluate_recall_answer(session_id, user_input)
            result = get_response_from_llama(
                user_input,
                session_id=session_id,
                provider=llm_provider,
            )

            if recall_feedback:
                result["answer"] = f"{result['answer']}\n\n{recall_feedback}"

            add_to_history(session_id, "user", user_input)
            add_to_history(session_id, "assistant", result["full_text"])
            if result.get("score_included", True):
                add_score_history(session_id, result["score"])

            follow_up_messages = []
            recall_prompt = maybe_advance_recall_test(session_id)
            if recall_prompt:
                result["answer"] = f"{result['answer']}\n\n{recall_prompt}"
                result["full_text"] = f"{result['full_text']}\n\n{recall_prompt}"
                follow_up_messages.append(recall_prompt)

            turn = add_turn_history(
                session_id=session_id,
                user_text=user_input,
                answer=result["answer"],
                judgment=result["judgment"],
                score=result["score"],
                reason=result["reason"],
                feature_scores=result["feature_scores"],
                follow_up_messages=follow_up_messages,
                score_included=result.get("score_included", True),
                excluded_reason=result.get("excluded_reason", ""),
                llm_provider=llm_provider,
            )

            return build_chat_response(
                session_id=session_id,
                user_speech=user_input,
                sys_response=result["full_text"],
                answer=result["answer"],
                judgment=result["judgment"],
                score=result["score"],
                reason=result["reason"],
                feature_scores=result["feature_scores"],
                follow_up_messages=follow_up_messages,
                turn=turn,
                score_included=result.get("score_included", True),
                excluded_reason=result.get("excluded_reason", ""),
                llm_provider=llm_provider,
            )
        except Exception:
            app.logger.exception("채팅 처리 오류")
            return jsonify({"error": "서버 처리 중 문제가 발생했습니다."}), 500

    @app.route("/api/analysis/start", methods=["POST"])
    def api_analysis_start():
        denied = require_api_access()
        if denied:
            return denied

        limited = enforce_rate_limit(
            store=rate_limit_store,
            request=request,
            bucket="analysis-start",
            limit=get_api_rate_limit(),
            window_seconds=get_rate_limit_window_seconds(),
        )
        if limited:
            return limited

        session_id = get_or_create_session_id()
        llm_provider = get_requested_llm_provider()

        if "audio" not in request.files or request.files["audio"].filename == "":
            return jsonify({"error": "audio file is required"}), 400

        audio_file = request.files["audio"]
        file_path = save_audio_upload(
            audio_file, UPLOAD_DIR, get_allowed_audio_extensions()
        )
        if not file_path:
            return jsonify({"error": "unsupported audio format"}), 400

        job_id = str(uuid.uuid4())
        runtime.job_store[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "stage": "stt",
            "progress": 0,
            "message": "분석을 준비하는 중입니다.",
            "partial_text": "",
            "error_message": "",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "result": None,
        }

        thread = threading.Thread(
            target=run_analysis_job,
            args=(job_id, file_path, session_id, llm_provider),
            daemon=True,
        )
        thread.start()

        return jsonify({"job_id": job_id})

    @app.route("/api/analysis/progress/<job_id>", methods=["GET"])
    def api_analysis_progress(job_id):
        denied = require_api_access()
        if denied:
            return denied

        def generate():
            last_payload = None
            deadline = time.time() + 180
            while time.time() < deadline:
                job = runtime.job_store.get(job_id)
                if job is None:
                    yield f"data: {json.dumps({'error': 'job not found'})}\n\n"
                    return

                payload = json.dumps(
                    {
                        "job_id": job["job_id"],
                        "status": job["status"],
                        "stage": job["stage"],
                        "progress": job["progress"],
                        "message": job["message"],
                        "partial_text": job.get("partial_text", ""),
                        "error_message": job.get("error_message", ""),
                        "updated_at": job["updated_at"],
                        "result": job["result"] if job["status"] == "completed" else None,
                    }
                )

                if payload != last_payload:
                    last_payload = payload
                    yield f"data: {payload}\n\n"

                if job["status"] in ("completed", "failed"):
                    return

                time.sleep(0.3)

            yield f"data: {json.dumps({'error': 'timeout', 'status': 'failed'})}\n\n"

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.route("/api/analysis/status/<job_id>", methods=["GET"])
    def api_analysis_status(job_id):
        denied = require_api_access()
        if denied:
            return denied

        job = runtime.job_store.get(job_id)
        if job is None:
            return jsonify({"error": "job not found"}), 404

        return jsonify(
            {
                "job_id": job["job_id"],
                "status": job["status"],
                "stage": job["stage"],
                "progress": job["progress"],
                "message": job["message"],
                "partial_text": job.get("partial_text", ""),
                "error_message": job.get("error_message", ""),
                "updated_at": job["updated_at"],
                "result": job["result"] if job["status"] == "completed" else None,
            }
        )

    @app.route("/api/stt-analyze", methods=["POST"])
    def api_stt_analyze():
        denied = require_api_access()
        if denied:
            return denied

        limited = enforce_rate_limit(
            store=rate_limit_store,
            request=request,
            bucket="stt-analyze",
            limit=get_api_rate_limit(),
            window_seconds=get_rate_limit_window_seconds(),
        )
        if limited:
            return limited

        session_id = get_or_create_session_id()
        llm_provider = get_requested_llm_provider()

        try:
            if "audio" not in request.files:
                return jsonify({"error": "audio file is required"}), 400

            audio_file = request.files["audio"]
            if audio_file.filename == "":
                return jsonify({"error": "audio file is required"}), 400

            file_path = save_audio_upload(
                audio_file, UPLOAD_DIR, get_allowed_audio_extensions()
            )
            if not file_path:
                return jsonify({"error": "unsupported audio format"}), 400

            user_input = transcribe_audio_file(file_path)
            if not user_input:
                return build_empty_stt_analysis_response(session_id, llm_provider)

            recall_feedback = evaluate_recall_answer(session_id, user_input)
            result = get_response_from_llama(
                user_input,
                session_id=session_id,
                provider=llm_provider,
            )

            if recall_feedback:
                result["answer"] = f"{result['answer']}\n\n{recall_feedback}"

            add_to_history(session_id, "user", user_input)
            add_to_history(session_id, "assistant", result["full_text"])
            if result.get("score_included", True):
                add_score_history(session_id, result["score"])

            follow_up_messages = []
            recall_prompt = maybe_advance_recall_test(session_id)
            if recall_prompt:
                result["answer"] = f"{result['answer']}\n\n{recall_prompt}"
                result["full_text"] = f"{result['full_text']}\n\n{recall_prompt}"
                follow_up_messages.append(recall_prompt)

            turn = add_turn_history(
                session_id=session_id,
                user_text=user_input,
                answer=result["answer"],
                judgment=result["judgment"],
                score=result["score"],
                reason=result["reason"],
                feature_scores=result["feature_scores"],
                follow_up_messages=follow_up_messages,
                score_included=result.get("score_included", True),
                excluded_reason=result.get("excluded_reason", ""),
                llm_provider=llm_provider,
            )

            return jsonify(
                build_analysis_detail_payload(
                    session_id,
                    stt_result=user_input,
                    answer=result["answer"],
                    judgment=result["judgment"],
                    reason=result["reason"],
                    score_total=result["score"],
                    feature_scores=result.get("feature_scores", {}),
                    llm_provider=llm_provider,
                    turn=turn,
                    score_included=result.get("score_included", True),
                    excluded_reason=result.get("excluded_reason", ""),
                )
            )
        except Exception:
            app.logger.exception("STT analyze API error")
            return jsonify({"error": "stt analyze failed"}), 500

    @app.route("/score-history", methods=["GET"])
    def score_history():
        denied = require_api_access()
        if denied:
            return denied

        session_id = get_or_create_session_id()
        return jsonify(
            {
                "session_id": session_id,
                "analysis_generation": get_analysis_generation(session_id),
                **build_analysis_metrics_payload(session_id),
            }
        )

    @app.route("/reset-history", methods=["POST"])
    def reset_history():
        denied = require_api_access()
        if denied:
            return denied

        session_id = get_or_create_session_id()
        next_generation = bump_analysis_generation(session_id)
        reset_session(session_id)

        return jsonify(
            {
                "session_id": session_id,
                "analysis_generation": next_generation,
                "message": "기록이 초기화되었습니다.",
                **build_analysis_metrics_payload(session_id),
            }
        )
