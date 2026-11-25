/*
 * 관리자 대시보드 — Django가 ORM으로 집계해 넘겨준 값(json_script)으로 Chart.js 차트 2개를 그립니다 (17개 동작).
 * (이 파일은 숫자를 만들지 않고 그리기만 함 → 예시 숫자 없음)
 */
(() => {
  const data = JSON.parse(document.getElementById("chart-data").textContent);
  if (typeof Chart === "undefined") {
    document.querySelectorAll(".chart-box").forEach((box) => {
      box.innerHTML = '<p class="muted">차트 라이브러리(Chart.js)를 불러오지 못했습니다. 인터넷 연결을 확인하세요.</p>';
    });
    return;
  }
  Chart.defaults.font.family = 'system-ui, "Malgun Gothic", sans-serif';

  // ① 17개 동작별 예측 건수 — 확실 / 불확실 누적 가로 막대 (동작 이름이 길어서 가로)
  new Chart(document.getElementById("action-chart"), {
    type: "bar",
    data: {
      labels: data.action.labels.map((l) => l.split("\n")[0]),
      datasets: [
        { label: "확실", data: data.action.certain, backgroundColor: "#2f5bea" },
        { label: "판단 불확실", data: data.action.uncertain, backgroundColor: "#f0b44c" },
      ],
    },
    options: {
      indexAxis: "y",
      maintainAspectRatio: false,
      scales: { x: { stacked: true, beginAtZero: true, ticks: { precision: 0 } }, y: { stacked: true } },
      plugins: {
        tooltip: { mode: "index", callbacks: { title: (items) => data.action.labels[items[0].dataIndex].replace("\n", " ") } },
      },
    },
  });

  // ② 작업 상태별 건수 — 원형
  const colors = { PENDING: "#94a3b8", STARTED: "#2f5bea", SUCCESS: "#1f8f55", FAILURE: "#c0392b" };
  new Chart(document.getElementById("status-chart"), {
    type: "doughnut",
    data: {
      labels: data.status.labels.map((l, i) => `${l} (${data.status.keys[i]})`),
      datasets: [{ data: data.status.counts, backgroundColor: data.status.keys.map((k) => colors[k]) }],
    },
    options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom" } } },
  });
})();
