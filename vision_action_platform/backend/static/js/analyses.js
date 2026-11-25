/*
 * 사용자 분석 화면
 *  ① 입력 방식(업로드 / IP Webcam) 선택
 *  ② "AI 분석 시작" → POST /api/analyses/  → 작업 ID 바로 받음 (분석은 Celery가 뒤에서)
 *  ③ 작업 ID로 GET /api/analyses/{id}/status/ 를 주기적으로 호출 (폴링)
 *  ④ 완료되면 GET /api/analyses/{id}/ 로 사람별 결과 표시 + 이력 새로고침
 */
(() => {
  const config = JSON.parse(document.getElementById("page-config").textContent);
  const LABELS = config.actionLabels;
  const POLL_MS = 1500;
  const STATUS_TEXT = { PENDING: "대기", STARTED: "처리 중", SUCCESS: "완료", FAILURE: "실패" };
  const INPUT_TEXT = { upload: "영상 업로드", camera: "IP Webcam" };

  const $ = (sel) => document.querySelector(sel);
  const form = $("#analysis-form");
  let inputType = "upload";
  let pollTimer = null;

  // Django CSRF 토큰 (Session 인증으로 POST 할 때 필요)
  const csrf = () => form.querySelector("[name=csrfmiddlewaretoken]").value;

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const actionText = (a) => (LABELS[a] ? `${LABELS[a]} <span class="muted">(${a})</span>` : esc(a));

  // ── ① 탭 전환 ─────────────────────────────
  document.querySelectorAll(".tabs button").forEach((btn) => {
    btn.addEventListener("click", () => {
      inputType = btn.dataset.type;
      document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("on", b === btn));
      document.querySelectorAll(".pane").forEach((p) => p.classList.toggle("on", p.dataset.pane === inputType));
      if (inputType === "camera") startPreview();
      else stopPreview();
    });
  });

  // ── IP Webcam 실시간 미리보기 ──────────────
  // Django 카메라 중계(/analyses/camera/<n>/stream/)를 <img>로 받음 (MJPEG).
  // 카메라 연결은 Django가 하나만 열어서 미리보기·Worker 녹화가 같이 씀 → DroidCam "Busy" 없음
  const previewImg = $("#preview-img");
  let previewTimer = null;
  let previewLive = false;

  const cameraIndex = () => $("#camera_url").selectedOptions[0]?.dataset.index;

  function startPreview() {
    const idx = cameraIndex();
    if (!idx) {
      $("#preview-msg").textContent = "등록된 카메라가 없습니다.";
      return;
    }
    previewLive = false;
    $("#preview").classList.remove("live");
    $("#preview-msg").textContent = "카메라 연결 중…";
    previewImg.src = `/analyses/camera/${idx}/stream/?t=${Date.now()}`;
    clearInterval(previewTimer);
    checkPreview();
    previewTimer = setInterval(checkPreview, 2000);
  }

  function stopPreview() {
    clearInterval(previewTimer);
    previewImg.removeAttribute("src"); // 연결 끊기 → 아무도 안 보면 Django가 카메라 연결 해제
    $("#preview").classList.remove("live");
  }

  async function checkPreview() {
    const idx = cameraIndex();
    if (!idx || inputType !== "camera") return;
    try {
      const st = await (await fetch(`/analyses/camera/${idx}/status/`)).json();
      const wasLive = previewLive;
      previewLive = st.connected;
      $("#preview").classList.toggle("live", st.connected);
      $("#preview-msg").textContent = st.connected ? "" : st.error || "카메라 연결 중…";
      $("#preview-state").textContent = st.connected && st.size
        ? `🟢 연결됨 · ${st.size[0]}×${st.size[1]} · 회전 ${st.rotation}°` : "🔴 연결 안 됨";
      if (st.size) $("#preview").style.aspectRatio = `${st.size[0]} / ${st.size[1]}`; // 세로 화면이면 세로로
      if (st.connected && !wasLive) previewImg.src = `/analyses/camera/${idx}/stream/?t=${Date.now()}`; // 다시 연결되면 새로 받기
    } catch (ex) {
      $("#preview-state").textContent = "🔴 서버 응답 없음";
    }
  }

  $("#camera_url").addEventListener("change", () => inputType === "camera" && startPreview());

  // 화면 돌리기: 휴대폰을 세워 찍으면 DroidCam 화면이 옆으로 누움 → 서버(중계)에서 회전 → 녹화에도 적용
  $("#rotate-btn").addEventListener("click", async () => {
    const idx = cameraIndex();
    if (!idx) return;
    const res = await fetch(`/analyses/camera/${idx}/rotate/`, { method: "POST", headers: { "X-CSRFToken": csrf() } });
    const body = await res.json().catch(() => ({}));
    if (res.ok) {
      $("#preview-state").textContent = `🟢 회전 ${body.rotation}° 적용`;
    } else {
      $("#preview-state").textContent = `⚠ ${body.detail || "화면을 돌리지 못했습니다."}`;
    }
  });

  // 분석이 끝나면 결과 동작을 미리보기 위에 크게 표시 (사람이 여러 명이면 모두)
  function showPreviewResult(job) {
    const box = $("#preview-result");
    if (job.input_type !== "camera") return;
    if (job.status === "FAILURE") {
      box.innerHTML = `⚠ 실패: ${esc(job.error_message).slice(0, 80)}`;
    } else if (!job.predictions.length) {
      box.innerHTML = `결과: ${esc(job.result_message || "사람을 찾지 못했습니다")}`;
    } else {
      box.innerHTML = job.predictions.map((p) =>
        `#${p.track_id} <b>${LABELS[p.action_label] || esc(p.action_label)}</b> ${p.confidence.toFixed(2)}` +
        (p.is_uncertain ? ' <span class="unsure">판단 불확실</span>' : "")).join("<br>");
    }
    box.hidden = false;
    clearTimeout(showPreviewResult.timer);
    showPreviewResult.timer = setTimeout(() => (box.hidden = true), 15000); // 15초 뒤 숨김
  }
  window.addEventListener("pagehide", stopPreview);

  // ── ② 분석 시작 ───────────────────────────
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const err = $("#form-error");
    err.hidden = true;

    const data = new FormData();
    data.append("input_type", inputType);
    data.append("title", $("#title").value.trim());
    data.append("expected_action", $("#expected_action").value);
    if (inputType === "upload") {
      const file = $("#video_file").files[0];
      if (!file) return showFormError("영상 파일을 선택하세요.");
      data.append("video_file", file);
    } else {
      if (!$("#camera_url").value) return showFormError("사용할 수 있는 카메라가 없습니다.");
      data.append("camera_url", $("#camera_url").value);
    }

    $("#start-btn").disabled = true;
    $("#start-btn").textContent = inputType === "upload" ? "업로드 중…" : "요청 중…";
    try {
      const res = await fetch("/api/analyses/", { method: "POST", body: data, headers: { "X-CSRFToken": csrf() } });
      const body = await res.json();
      if (!res.ok) return showFormError(flattenErrors(body));
      startTracking(body); // 작업 ID를 바로 받음 → 상태 추적 시작
      form.reset();
    } catch (ex) {
      showFormError("서버에 연결할 수 없습니다.");
    } finally {
      $("#start-btn").disabled = false;
      $("#start-btn").textContent = "▶ AI 분석 시작";
    }
  });

  function showFormError(msg) {
    const err = $("#form-error");
    err.textContent = msg;
    err.hidden = false;
  }

  function flattenErrors(body) {
    if (typeof body === "string") return body;
    return Object.entries(body).map(([k, v]) => `${k === "non_field_errors" || k === "detail" ? "" : k + ": "}${[].concat(v).join(" ")}`).join(" / ");
  }

  // ── ③ 상태 폴링 ───────────────────────────
  function startTracking(job) {
    clearTimeout(pollTimer);
    $("#preview-result").hidden = true; // 이전 결과 지우기
    $("#current-empty").hidden = true;
    $("#current-body").hidden = false;
    $("#current-title").textContent = `#${job.id} ${job.title || "(제목 없음)"}`;
    $("#current-meta").textContent = `· ${INPUT_TEXT[job.input_type]}`;
    $("#current-result").innerHTML = "";
    $("#current-error").hidden = true;
    $("#current-link").hidden = true;
    $("#current-note").textContent = job.input_type === "camera" ? "녹화 준비 중… ● REC 가 뜨면 동작하세요." : "";
    renderSteps(job.status);
    poll(job.id, job.input_type);
  }

  // IP Webcam: Worker가 첫 화면을 받아 녹화를 시작하면(result_message="카메라 수집 중")
  // 오른쪽 카운트다운 + 미리보기 위 ● REC 표시 → 그동안 카메라 앞에서 동작
  const CAPTURING = "카메라 수집 중";
  const PREPARING = "카메라 준비 중";
  let countdownTimer = null;

  // 준비 시간: 버튼을 누르고 촬영 위치로 이동 → "준비! N" 뒤에 ● REC
  function startPrepare() {
    clearInterval(countdownTimer);
    const end = Date.now() + config.prepareSeconds * 1000;
    const tick = () => {
      const left = Math.ceil((end - Date.now()) / 1000);
      const text = left > 0 ? `준비! ${left}` : "곧 녹화 시작";
      $("#current-note").innerHTML = `<b style="font-size:1.3rem;color:var(--warn)">🚶 촬영 위치로! ${left > 0 ? left + "초 뒤 녹화" : "곧 녹화"}</b>`;
      $("#rec-badge").hidden = false;
      $("#rec-badge").classList.add("prep");
      $("#rec-left").textContent = text;
    };
    tick();
    countdownTimer = setInterval(tick, 250);
  }
  function setRecording(on, left = "") {
    $("#rec-badge").classList.remove("prep");
    $("#rec-badge").hidden = !on;
    $("#rec-left").textContent = left;
    $("#preview").classList.toggle("recording", on);
  }
  function startCountdown() {
    clearInterval(countdownTimer);
    const end = Date.now() + config.captureSeconds * 1000;
    const note = $("#current-note");
    const tick = () => {
      const left = Math.ceil((end - Date.now()) / 1000);
      if (left > 0) {
        note.innerHTML = `<b style="font-size:1.4rem;color:var(--bad)">📸 지금 동작하세요! ${left}초</b>`;
        setRecording(true, `${left}초`);
      } else {
        note.textContent = "수집 끝 — AI가 분석 중입니다…";
        setRecording(false);
        clearInterval(countdownTimer);
      }
    };
    tick();
    countdownTimer = setInterval(tick, 250);
  }

  async function poll(id, inputType, counting = false, preparing = false) {
    try {
      const st = await (await fetch(`/api/analyses/${id}/status/`)).json();
      renderSteps(st.status);
      if (inputType === "camera" && st.result_message === PREPARING && !preparing) {
        startPrepare();
        preparing = true;
      }
      if (inputType === "camera" && st.result_message === CAPTURING && !counting) {
        startCountdown();
        counting = true;
      }
      if (st.is_finished) {
        clearInterval(countdownTimer);
        setRecording(false);
        return showResult(id);
      }
    } catch (ex) {
      /* 일시적인 네트워크 오류 → 다음 주기에 다시 */
    }
    // 카메라는 수집 시작을 놓치지 않게 더 자주 확인
    pollTimer = setTimeout(() => poll(id, inputType, counting, preparing), inputType === "camera" ? 400 : POLL_MS);
  }

  function renderSteps(status) {
    const order = ["PENDING", "STARTED", "DONE"];
    const now = status === "SUCCESS" || status === "FAILURE" ? "DONE" : status;
    document.querySelectorAll("#steps span").forEach((el) => {
      const i = order.indexOf(el.dataset.step), j = order.indexOf(now);
      el.className = i < j ? "done" : i === j ? (status === "FAILURE" ? "fail" : status === "SUCCESS" ? "done" : "now") : "";
      if (el.dataset.step === "DONE") el.textContent = status === "FAILURE" ? "실패" : "완료";
    });
  }

  // ── ④ 결과 표시 ───────────────────────────
  async function showResult(id) {
    const job = await (await fetch(`/api/analyses/${id}/`)).json();
    $("#current-note").textContent = job.processing_time_sec != null ? `처리 시간 ${job.processing_time_sec}초` : "";
    $("#current-link").hidden = false;
    $("#current-link a").href = `/analyses/${id}/`;
    showPreviewResult(job);
    if (job.status === "FAILURE") {
      $("#current-error").textContent = `⚠ ${job.error_message}`;
      $("#current-error").hidden = false;
    } else {
      $("#current-result").innerHTML = predictionsTable(job);
    }
    refreshHistory();
  }

  function confBar(p) {
    const pct = Math.round(p.confidence * 100);
    return `<div class="conf"><div class="bar"><i class="${p.is_uncertain ? "low" : ""}" style="width:${pct}%"></i></div>${p.confidence.toFixed(2)}</div>`;
  }

  function predictionsTable(job) {
    if (!job.predictions.length) return `<p class="muted">${esc(job.result_message || "예측 결과가 없습니다.")}</p>`;
    const others = (p) => (p.top3 || []).slice(1)
      .map((t) => `${LABELS[t.action_label] || esc(t.action_label)} ${t.confidence.toFixed(2)}`).join(" · ");
    const rows = job.predictions.map((p) => `
      <tr>
        <td>#${p.track_id}</td>
        <td>${actionText(p.action_label)}${p.is_uncertain ? ' <span class="badge uncertain">판단 불확실</span>' : ""}
          ${others(p) ? `<div class="muted">다음 후보: ${others(p)}</div>` : ""}
          ${p.rule_note ? `<div class="muted">🔧 보정: ${esc(p.rule_note)}</div>` : ""}</td>
        <td>${confBar(p)}</td>
        <td class="num muted">${p.valid_frames}</td>
      </tr>`).join("");
    const expected = job.expected_action
      ? `<p class="muted">수행하려는 동작: ${actionText(job.expected_action)} ${job.predictions.some((p) => p.action_label === job.expected_action) ? "✅ 일치" : "❌ 불일치"}</p>`
      : "";
    return `${expected}
      <table><thead><tr><th>Track ID</th><th>예측 동작</th><th>Confidence</th><th>유효 프레임</th></tr></thead><tbody>${rows}</tbody></table>
      <p class="muted">Confidence ${config.uncertainThreshold} 미만은 "판단 불확실" — 17개 동작에 없는 동작이나 새로운 환경은 낮게 나올 수 있습니다.</p>`;
  }

  // ── 이력 새로고침 ─────────────────────────
  async function refreshHistory() {
    const jobs = (await (await fetch("/api/analyses/")).json()).slice(0, 10);
    $("#history").innerHTML = jobs.length
      ? jobs.map((j) => `
        <tr>
          <td>${j.id}</td>
          <td>${esc(j.title || "(제목 없음)")}</td>
          <td>${INPUT_TEXT[j.input_type]}</td>
          <td><span class="badge s-${j.status}">${STATUS_TEXT[j.status]}</span></td>
          <td>${j.predictions.length
            ? j.predictions.map((p) => `<div>#${p.track_id} ${LABELS[p.action_label] || esc(p.action_label)} <span class="muted">${p.confidence.toFixed(2)}</span>${p.is_uncertain ? ' <span class="badge uncertain">불확실</span>' : ""}</div>`).join("")
            : `<span class="muted">${esc(j.status === "FAILURE" ? j.error_message.slice(0, 40) : j.result_message || "-")}</span>`}</td>
          <td class="muted">${new Date(j.created_at).toLocaleString("ko-KR", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" })}</td>
          <td><a href="/analyses/${j.id}/">상세</a></td>
        </tr>`).join("")
      : '<tr><td colspan="7" class="muted">아직 분석 기록이 없습니다.</td></tr>';
  }

  // 새로고침했을 때 진행 중인 작업이 있으면 이어서 추적 (T06)
  (async () => {
    const jobs = await (await fetch("/api/analyses/")).json();
    const running = jobs.find((j) => !j.is_finished);
    if (running) startTracking(running);
  })();
})();
