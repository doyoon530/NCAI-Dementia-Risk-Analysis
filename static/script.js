let mediaRecorder = null;
let audioChunks = [];

let sessionId = localStorage.getItem("session_id") || null;

let scoreHistory = [];
let scoreChart = null;
let gaugeChart = null;
let radarChart = null;
let analysisRevealTimeout = null;

const startButton = document.getElementById("startRecord");
const stopButton = document.getElementById("stopRecord");
const resetButton = document.getElementById("resetHistory");
const chatWindow = document.getElementById("chatWindow");
const recordingIndicator = document.getElementById("recordingIndicator");
const aiThinking = document.getElementById("aiThinking");
const systemStateText = document.getElementById("systemStateText");

const avgScoreEl = document.getElementById("avgScore");
const recentAvgScoreEl = document.getElementById("recentAvgScore");
const latestScoreEl = document.getElementById("latestScore");
const gaugeScoreEl = document.getElementById("gaugeScore");
const trendTextEl = document.getElementById("trendText");

const analysisJudgmentEl = document.getElementById("analysisJudgment");
const analysisScoreEl = document.getElementById("analysisScore");
const analysisRiskLevelEl = document.getElementById("analysisRiskLevel");
const analysisTrendEl = document.getElementById("analysisTrend");
const analysisReasonEl = document.getElementById("analysisReason");

const featureRepetitionValueEl = document.getElementById("featureRepetitionValue");
const featureMemoryValueEl = document.getElementById("featureMemoryValue");
const featureTimeValueEl = document.getElementById("featureTimeValue");
const featureIncoherenceValueEl = document.getElementById("featureIncoherenceValue");

const featureRepetitionBarEl = document.getElementById("featureRepetitionBar");
const featureMemoryBarEl = document.getElementById("featureMemoryBar");
const featureTimeBarEl = document.getElementById("featureTimeBar");
const featureIncoherenceBarEl = document.getElementById("featureIncoherenceBar");

const confidenceScoreEl = document.getElementById("confidenceScore");

const recallStatusEl = document.getElementById("recallStatus");
const recallLastResultEl = document.getElementById("recallLastResult");
const recallPromptEl = document.getElementById("recallPrompt");

const warningPopup = document.getElementById("warningPopup");
const warningPopupText = document.getElementById("warningPopupText");
const closeWarningPopupButton = document.getElementById("closeWarningPopup");

document.addEventListener("DOMContentLoaded", async () => {
    bindEvents();
    await loadScoreHistory();
});

function bindEvents() {
    if (startButton) startButton.onclick = startRecording;
    if (stopButton) stopButton.onclick = stopRecording;
    if (resetButton) resetButton.onclick = resetHistory;
    if (closeWarningPopupButton) closeWarningPopupButton.onclick = hideWarningPopup;
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = function (event) {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async function () {
            setRecordingState(false);

            try {
                setSystemState("음성 인식 중...");
                setAnalysisThinking(true);
                setAnalysisLoadingState(true);

                const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
                const formData = new FormData();
                formData.append("audio", audioBlob, "recording.wav");

                const transcribeUrl = sessionId
                    ? `/transcribe-audio?session_id=${encodeURIComponent(sessionId)}`
                    : "/transcribe-audio";

                const sttResponse = await fetch(transcribeUrl, {
                    method: "POST",
                    body: formData
                });

                const sttData = await sttResponse.json();

                if (sttData.error) {
                    appendChatMessage("system", sttData.error);
                    setSystemState("오류 발생");
                    setAnalysisThinking(false);
                    setAnalysisLoadingState(false);
                    return;
                }

                if (sttData.session_id) {
                    sessionId = sttData.session_id;
                    localStorage.setItem("session_id", sessionId);
                }

                const recognizedText = normalizeText(sttData.user_speech || "");

                if (!recognizedText) {
                    appendChatMessage("system", "음성 인식 결과가 없습니다. 다시 녹음해 주세요.");
                    setSystemState("음성 인식 실패");
                    setAnalysisThinking(false);
                    setAnalysisLoadingState(false);
                    return;
                }

                appendChatMessage("user", recognizedText);
                appendLoadingMessage("답변 생성 중...");
                setSystemState("답변 생성 중...");

                const analyzeUrl = sessionId
                    ? `/analyze-text?session_id=${encodeURIComponent(sessionId)}`
                    : "/analyze-text";

                const analyzeResponse = await fetch(analyzeUrl, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        message: recognizedText
                    })
                });

                const data = await analyzeResponse.json();

                if (data.error) {
                    removeLoadingMessage();
                    appendChatMessage("system", data.error);
                    setSystemState("오류 발생");
                    setAnalysisThinking(false);
                    setAnalysisLoadingState(false);
                    return;
                }

                if (data.session_id) {
                    sessionId = data.session_id;
                    localStorage.setItem("session_id", sessionId);
                }

                removeLoadingMessage();
                appendChatMessage("system", data.answer || data.sys_response || "");
                setSystemState("치매 분석 반영 중...");

                scoreHistory = Array.isArray(data.score_history) ? data.score_history : [];

                updateAnalysisCard(data);
                updateFeatureBreakdown(data.feature_scores || {});
                updateRecallCard(data.recall || {});
                renderAll(data);
                revealAnalysisWithCountUp(data);

                setSystemState("분석 완료");

                if ((data.score ?? 0) >= 60 || (data.recent_average_score ?? 0) >= 60) {
                    showWarningPopup(
                        `현재 점수 ${data.score ?? 0}점, 최근 5회 평균 ${data.recent_average_score ?? 0}점으로 위험 구간에 해당합니다.`
                    );
                }
            } catch (error) {
                console.error(error);
                removeLoadingMessage();
                appendChatMessage("system", "오류가 발생했습니다. 다시 시도해주세요.");
                setSystemState("오류 발생");
                setAnalysisThinking(false);
                setAnalysisLoadingState(false);
            } finally {
                audioChunks = [];
            }
        };

        mediaRecorder.start();
        setRecordingState(true);
    } catch (error) {
        console.error(error);
        alert("마이크 접근에 실패했습니다.");
    }
}

function stopRecording() {
    if (!mediaRecorder) {
        return;
    }

    mediaRecorder.stop();
}

async function resetHistory() {
    try {
        const url = sessionId
            ? `/reset-history?session_id=${encodeURIComponent(sessionId)}`
            : "/reset-history";

        const response = await fetch(url, {
            method: "POST"
        });

        const data = await response.json();

        if (data.session_id) {
            sessionId = data.session_id;
            localStorage.setItem("session_id", sessionId);
        }

        scoreHistory = [];
        if (chatWindow) {
            chatWindow.innerHTML = "";
        }

        resetAnalysisCard();
        updateFeatureBreakdown({});
        updateRecallCard(data.recall || {});
        updateConfidence({}, 0);
        renderAll(data);

        setSystemState("기록 초기화 완료");
    } catch (error) {
        console.error(error);
        alert("기록 초기화 중 오류가 발생했습니다.");
    }
}

function setRecordingState(isRecording) {
    if (isRecording) {
        if (startButton) startButton.disabled = true;
        if (stopButton) stopButton.disabled = false;
        if (recordingIndicator) recordingIndicator.classList.remove("hidden");
        setSystemState("녹음 중");
    } else {
        if (startButton) startButton.disabled = false;
        if (stopButton) stopButton.disabled = true;
        if (recordingIndicator) recordingIndicator.classList.add("hidden");
    }
}

function setAnalysisThinking(isThinking) {
    if (!aiThinking) {
        return;
    }

    if (isThinking) {
        aiThinking.classList.remove("hidden");
    } else {
        aiThinking.classList.add("hidden");
    }
}

function setSystemState(text) {
    if (systemStateText) {
        systemStateText.innerText = text;
    }
}

function getAnalysisCards() {
    return Array.from(document.querySelectorAll(".analysis-card"));
}

function setSkeletonLoading(isLoading) {
    const cards = getAnalysisCards();

    cards.forEach((card) => {
        if (isLoading) {
            card.classList.add("is-skeleton");
        } else {
            card.classList.remove("is-skeleton");
        }
    });
}

function setAnalysisLoadingState(isLoading) {
    const targets = [
        analysisScoreEl,
        analysisJudgmentEl,
        analysisRiskLevelEl,
        analysisTrendEl,
        analysisReasonEl,
        confidenceScoreEl,
        featureRepetitionValueEl,
        featureMemoryValueEl,
        featureTimeValueEl,
        featureIncoherenceValueEl,
        recallStatusEl,
        recallLastResultEl,
        recallPromptEl
    ];

    targets.forEach((el) => {
        if (!el) return;
        el.style.opacity = isLoading ? "0.55" : "1";
        el.style.transition = "opacity 0.2s ease";
    });

    setSkeletonLoading(isLoading);
}

function appendChatMessage(type, text) {
    if (!chatWindow) {
        return;
    }

    const message = document.createElement("div");
    message.classList.add("message", "message-enter");

    if (type === "user") {
        message.classList.add("user-message");
    } else {
        message.classList.add("system-message");
    }

    message.innerText = text;
    chatWindow.appendChild(message);
    scrollChatToBottom();
}

function appendLoadingMessage(text = "답변 생성 중...") {
    if (!chatWindow) {
        return;
    }

    removeLoadingMessage();

    const message = document.createElement("div");
    message.classList.add("message", "system-message", "message-enter");
    message.id = "loadingMessage";
    message.innerText = text;
    chatWindow.appendChild(message);
    scrollChatToBottom();
}

function removeLoadingMessage() {
    const loading = document.getElementById("loadingMessage");
    if (loading) {
        loading.remove();
    }
}

function scrollChatToBottom() {
    if (!chatWindow) {
        return;
    }
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function loadScoreHistory() {
    try {
        const url = sessionId
            ? `/score-history?session_id=${encodeURIComponent(sessionId)}`
            : "/score-history";

        const response = await fetch(url);
        const data = await response.json();

        if (data.session_id) {
            sessionId = data.session_id;
            localStorage.setItem("session_id", sessionId);
        }

        scoreHistory = Array.isArray(data.score_history) ? data.score_history : [];
        updateRecallCard(data.recall || {});
        renderAll(data);
        updateConfidence({}, 0);
    } catch (error) {
        console.error("점수 기록 로딩 실패:", error);
        renderAll({
            average_score: 0,
            recent_average_score: 0,
            risk_level: "Normal",
            trend: "데이터 부족",
            score_history: []
        });
    }
}

function updateAnalysisCard(data) {
    if (analysisJudgmentEl) analysisJudgmentEl.innerText = data.judgment || "없음";
    if (analysisRiskLevelEl) analysisRiskLevelEl.innerText = data.risk_level || "Normal";
    if (analysisTrendEl) analysisTrendEl.innerText = data.trend || "데이터 부족";
    if (analysisReasonEl) analysisReasonEl.innerText = data.reason || "분석 근거가 없습니다.";
}

function resetAnalysisCard() {
    if (analysisJudgmentEl) analysisJudgmentEl.innerText = "없음";
    if (analysisScoreEl) analysisScoreEl.innerText = "0";
    if (analysisRiskLevelEl) analysisRiskLevelEl.innerText = "Normal";
    if (analysisTrendEl) analysisTrendEl.innerText = "데이터 부족";
    if (analysisReasonEl) analysisReasonEl.innerText = "아직 분석 결과가 없습니다.";
}

function updateFeatureBreakdown(featureScores) {
    const repetition = Number(featureScores.repetition ?? 0);
    const memory = Number(featureScores.memory ?? 0);
    const timeConfusion = Number(featureScores.time_confusion ?? 0);
    const incoherence = Number(featureScores.incoherence ?? 0);

    if (featureRepetitionValueEl) featureRepetitionValueEl.innerText = repetition;
    if (featureMemoryValueEl) featureMemoryValueEl.innerText = memory;
    if (featureTimeValueEl) featureTimeValueEl.innerText = timeConfusion;
    if (featureIncoherenceValueEl) featureIncoherenceValueEl.innerText = incoherence;

    if (featureRepetitionBarEl) featureRepetitionBarEl.style.width = `${(repetition / 25) * 100}%`;
    if (featureMemoryBarEl) featureMemoryBarEl.style.width = `${(memory / 25) * 100}%`;
    if (featureTimeBarEl) featureTimeBarEl.style.width = `${(timeConfusion / 30) * 100}%`;
    if (featureIncoherenceBarEl) featureIncoherenceBarEl.style.width = `${(incoherence / 20) * 100}%`;

    updateRadarChart(repetition, memory, timeConfusion, incoherence);
}

function updateRecallCard(recall) {
    const statusMap = {
        idle: "대기",
        memorize: "단어 제시",
        ask: "회상 질문"
    };

    if (recallStatusEl) recallStatusEl.innerText = statusMap[recall.status] || "대기";
    if (recallLastResultEl) recallLastResultEl.innerText = recall.last_result || "없음";

    if (recallPromptEl) {
        if (recall.prompt) {
            recallPromptEl.innerText = recall.prompt;
        } else {
            recallPromptEl.innerText = "아직 진행 중인 기억 테스트가 없습니다.";
        }
    }
}

function calculateConfidenceValue(featureScores, totalScore) {
    const repetition = Number(featureScores.repetition ?? 0);
    const memory = Number(featureScores.memory ?? 0);
    const timeConfusion = Number(featureScores.time_confusion ?? 0);
    const incoherence = Number(featureScores.incoherence ?? 0);

    let confidence = 55;

    if (memory > 0) confidence += 8;
    if (timeConfusion > 0) confidence += 8;
    if (repetition > 0) confidence += 6;
    if (incoherence > 0) confidence += 6;
    if (totalScore >= 40) confidence += 8;
    if (totalScore >= 60) confidence += 4;

    return Math.max(0, Math.min(95, confidence));
}

function updateConfidence(featureScores, totalScore) {
    const confidence = calculateConfidenceValue(featureScores, totalScore);

    if (confidenceScoreEl) {
        animateNumber(
            confidenceScoreEl,
            extractNumber(confidenceScoreEl.innerText),
            confidence,
            750,
            true
        );
    }
}

function revealSummaryNumbers(data) {
    const averageScore = Number(data.average_score ?? 0);
    const recentAverageScore = Number(data.recent_average_score ?? averageScore);
    const latestScore = scoreHistory.length > 0
        ? scoreHistory[scoreHistory.length - 1].score
        : 0;
    const confidenceValue = calculateConfidenceValue(data.feature_scores || {}, data.score ?? 0);

    if (avgScoreEl) {
        animateNumber(avgScoreEl, extractNumber(avgScoreEl.innerText), averageScore, 700, false, 1);
    }
    if (recentAvgScoreEl) {
        animateNumber(recentAvgScoreEl, extractNumber(recentAvgScoreEl.innerText), recentAverageScore, 700, false, 1);
    }
    if (latestScoreEl) {
        animateNumber(latestScoreEl, extractNumber(latestScoreEl.innerText), latestScore, 700, false);
    }
    if (gaugeScoreEl) {
        animateNumber(gaugeScoreEl, extractNumber(gaugeScoreEl.innerText), Math.round(recentAverageScore), 700, false);
    }
    if (analysisScoreEl) {
        animateNumber(analysisScoreEl, extractNumber(analysisScoreEl.innerText), Number(data.score ?? 0), 750, false);
    }
    if (confidenceScoreEl) {
        animateNumber(confidenceScoreEl, extractNumber(confidenceScoreEl.innerText), confidenceValue, 750, true);
    }
}

function revealAnalysisWithCountUp(data) {
    if (analysisRevealTimeout) {
        clearTimeout(analysisRevealTimeout);
    }

    analysisRevealTimeout = setTimeout(() => {
        setAnalysisThinking(false);
        setAnalysisLoadingState(false);
        revealSummaryNumbers(data);
    }, 140);
}

function renderAll(data) {
    const averageScore = Number(data.average_score ?? 0);
    const recentAverageScore = Number(data.recent_average_score ?? averageScore);
    const latestScore = scoreHistory.length > 0
        ? scoreHistory[scoreHistory.length - 1].score
        : 0;

    updateSummary(averageScore, recentAverageScore, latestScore, data.trend || "데이터 부족");
    updateStatusCard(recentAverageScore);
    updateLineChart(recentAverageScore);
    updateGaugeChart(recentAverageScore);
}

function updateSummary(averageScore, recentAverageScore, latestScore, trend) {
    if (trendTextEl) {
        trendTextEl.innerText = trend;
    }
}

function getRiskInfo(score) {
    if (score < 20) {
        return {
            text: "정상",
            desc: "안정적인 상태입니다.",
            cssClass: "risk-safe",
            color: "#22c55e"
        };
    }

    if (score < 40) {
        return {
            text: "낮은 위험",
            desc: "경미한 변화가 보입니다.",
            cssClass: "risk-low",
            color: "#60a5fa"
        };
    }

    if (score < 60) {
        return {
            text: "주의",
            desc: "지속 관찰이 필요합니다.",
            cssClass: "risk-warning",
            color: "#f59e0b"
        };
    }

    if (score < 80) {
        return {
            text: "위험",
            desc: "상당한 위험 신호가 있습니다.",
            cssClass: "risk-high",
            color: "#ef4444"
        };
    }

    return {
        text: "매우 위험",
        desc: "즉각적인 관찰이 필요합니다.",
        cssClass: "risk-critical",
        color: "#991b1b"
    };
}

function updateStatusCard(recentAverageScore) {
    const statusCard = document.getElementById("statusCard");
    const riskText = document.getElementById("riskText");
    const riskDescription = document.getElementById("riskDescription");

    if (!statusCard || !riskText || !riskDescription) {
        return;
    }

    const risk = getRiskInfo(recentAverageScore);

    statusCard.classList.remove("risk-safe", "risk-low", "risk-warning", "risk-high", "risk-critical");
    statusCard.classList.add(risk.cssClass);

    riskText.innerText = risk.text;
    riskDescription.innerText = risk.desc;
}

function buildThresholdDataset(value, label) {
    return {
        label: label,
        data: scoreHistory.map(() => value),
        borderColor: value === 30 ? "rgba(245, 158, 11, 0.55)" : "rgba(239, 68, 68, 0.55)",
        borderWidth: 1,
        borderDash: [6, 6],
        pointRadius: 0,
        fill: false
    };
}

function updateLineChart(recentAverageScore) {
    const canvas = document.getElementById("scoreChart");
    if (!canvas) {
        return;
    }

    const ctx = canvas.getContext("2d");

    const labels = scoreHistory.map((item) => item.time);
    const scores = scoreHistory.map((item) => item.score);
    const risk = getRiskInfo(recentAverageScore);

    if (scoreChart) {
        scoreChart.destroy();
    }

    scoreChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "치매 의심 점수",
                    data: scores,
                    borderColor: risk.color,
                    backgroundColor: risk.color,
                    borderWidth: 3,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    tension: 0.35,
                    fill: false
                },
                buildThresholdDataset(30, "주의 기준"),
                buildThresholdDataset(60, "위험 기준")
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 900,
                easing: "easeOutQuart"
            },
            plugins: {
                legend: {
                    labels: {
                        color: "#334155"
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: "#64748b"
                    },
                    grid: {
                        color: "rgba(148, 163, 184, 0.12)"
                    }
                },
                y: {
                    min: 0,
                    max: 100,
                    ticks: {
                        stepSize: 20,
                        color: "#64748b"
                    },
                    grid: {
                        color: "rgba(148, 163, 184, 0.12)"
                    }
                }
            }
        }
    });
}

function updateGaugeChart(recentAverageScore) {
    const canvas = document.getElementById("gaugeChart");
    if (!canvas) {
        return;
    }

    const ctx = canvas.getContext("2d");
    const safeScore = Math.max(0, Math.min(100, recentAverageScore));
    const risk = getRiskInfo(safeScore);

    if (gaugeChart) {
        gaugeChart.destroy();
    }

    gaugeChart = new Chart(ctx, {
        type: "doughnut",
        data: {
            datasets: [
                {
                    data: [safeScore, 100 - safeScore],
                    backgroundColor: [risk.color, "rgba(255, 255, 255, 0.35)"],
                    borderWidth: 0,
                    circumference: 180,
                    rotation: 270,
                    cutout: "76%"
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                animateRotate: true,
                duration: 900
            },
            plugins: {
                tooltip: {
                    enabled: false
                },
                legend: {
                    display: false
                }
            }
        }
    });
}

function updateRadarChart(repetition, memory, timeConfusion, incoherence) {
    const canvas = document.getElementById("radarChart");
    if (!canvas) {
        return;
    }

    const ctx = canvas.getContext("2d");

    if (radarChart) {
        radarChart.destroy();
    }

    radarChart = new Chart(ctx, {
        type: "radar",
        data: {
            labels: ["질문 반복", "기억 혼란", "시간 혼란", "문장 비논리성"],
            datasets: [
                {
                    label: "언어 특징 점수",
                    data: [repetition, memory, timeConfusion, incoherence],
                    borderColor: "#3b82f6",
                    backgroundColor: "rgba(96, 165, 250, 0.18)",
                    borderWidth: 2,
                    pointBackgroundColor: "#2563eb"
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    min: 0,
                    max: 30,
                    ticks: {
                        backdropColor: "transparent",
                        color: "#64748b"
                    },
                    grid: {
                        color: "rgba(148, 163, 184, 0.18)"
                    },
                    angleLines: {
                        color: "rgba(148, 163, 184, 0.18)"
                    },
                    pointLabels: {
                        color: "#334155",
                        font: {
                            size: 12
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: "#334155"
                    }
                }
            }
        }
    });
}

function animateNumber(element, start, end, duration = 700, isPercent = false, fixed = 0) {
    if (!element) {
        return;
    }

    let startTime = null;

    function update(currentTime) {
        if (!startTime) {
            startTime = currentTime;
        }

        const progress = Math.min((currentTime - startTime) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const value = start + (end - start) * eased;

        if (fixed > 0) {
            element.innerText = `${value.toFixed(fixed)}${isPercent ? "%" : ""}`;
        } else {
            element.innerText = `${Math.round(value)}${isPercent ? "%" : ""}`;
        }

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

function extractNumber(text) {
    const numeric = parseFloat(String(text).replace(/[^0-9.]/g, ""));
    return Number.isNaN(numeric) ? 0 : numeric;
}

function normalizeText(text) {
    return String(text || "").replace(/\s+/g, " ").trim();
}

function showWarningPopup(message) {
    if (!warningPopup || !warningPopupText) {
        return;
    }

    warningPopupText.innerText = message;
    warningPopup.classList.remove("hidden");
}

function hideWarningPopup() {
    if (!warningPopup) {
        return;
    }

    warningPopup.classList.add("hidden");
}