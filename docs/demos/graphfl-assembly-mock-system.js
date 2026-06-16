(() => {
  "use strict";

  const STORAGE_KEY = "graphfl-demo-runs-v2";
  const MAX_RECORDS = 20;
  const STAGES = ["draft", "submitted", "queued", "running", "collecting", "saved", "completed"];
  const state = {
    root: null,
    records: [],
    active: null,
    timers: [],
    els: {},
    storageAvailable: true,
  };

  function node(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text != null) el.textContent = text;
    return el;
  }

  function clear(el) {
    if (el) el.textContent = "";
  }

  function clip(value, limit = 160) {
    const text = value == null ? "" : String(value);
    return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
  }

  function isoNow() {
    return new Date().toISOString();
  }

  function id(prefix) {
    const stamp = Date.now().toString(36);
    const rand = Math.random().toString(36).slice(2, 7);
    return `${prefix}_${stamp}_${rand}`;
  }

  function getRunButton() {
    return document.getElementById("btn-run");
  }

  function setRunButton(running) {
    const btn = getRunButton();
    if (!btn) return;
    if (running) {
      btn.disabled = true;
      btn.textContent = "Mock 실행 중…";
      return;
    }
    const submission = window.GraphFLDemo?.getSubmission?.();
    btn.disabled = !submission?.ready;
    btn.textContent = "▶ 실험 시작 (데모)";
  }

  function setRunNote(text) {
    const note = document.getElementById("run-note");
    if (note) note.textContent = text;
  }

  function readRecords() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch (_) {
      state.storageAvailable = false;
      return [];
    }
  }

  function writeRecords(records) {
    state.records = records.slice(0, MAX_RECORDS);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state.records));
      state.storageAvailable = true;
    } catch (_) {
      state.storageAvailable = false;
    }
  }

  function latestCompleted() {
    return state.records.find((record) => record.status === "completed" && Array.isArray(record.rows));
  }

  function schedule(token, delay, fn) {
    const timer = setTimeout(() => {
      if (!state.active || state.active.token !== token) return;
      fn();
    }, delay);
    state.timers.push(timer);
  }

  function clearTimers() {
    state.timers.forEach((timer) => clearTimeout(timer));
    state.timers = [];
  }

  function addLog(text) {
    if (!state.els.log) return;
    const p = node("p", "", `[${new Date().toLocaleTimeString()}] ${text}`);
    state.els.log.appendChild(p);
    state.els.log.scrollTop = state.els.log.scrollHeight;
  }

  function setProgress(percent) {
    if (state.els.progressBar) {
      state.els.progressBar.style.width = `${Math.max(0, Math.min(100, percent))}%`;
    }
    if (state.els.progressTrack) {
      state.els.progressTrack.setAttribute("aria-valuenow", String(Math.max(0, Math.min(100, percent))));
    }
  }

  function setField(name, text) {
    const target = state.els[name];
    if (target) target.textContent = text;
  }

  function renderArtifacts(artifacts) {
    const box = state.els.artifacts;
    if (!box) return;
    clear(box);
    if (!artifacts.length) {
      box.appendChild(node("p", "", "artifact 대기 중"));
      return;
    }
    artifacts.forEach((artifact) => {
      box.appendChild(
        node(
          "p",
          "",
          `${artifact.status} · ${artifact.type} · ${artifact.path}`
        )
      );
    });
  }

  function makeCard(title, key, initialText) {
    const card = node("div", "mock-card");
    card.appendChild(node("div", "mock-card-title", title));
    const value = node("div", "mock-value", initialText);
    state.els[key] = value;
    card.appendChild(value);
    return card;
  }

  function makeStatusRow(label, key, value) {
    const row = node("div", "mock-row");
    row.appendChild(node("span", "", label));
    const strong = node("strong", "", value);
    state.els[key] = strong;
    row.appendChild(strong);
    return row;
  }

  function renderTable(title, columns, rows) {
    const wrap = node("div", "mock-table-wrap");
    const heading = node("h3", "", title);
    const table = node("table", "mock-mini-table");
    const thead = node("thead");
    const headRow = node("tr");
    columns.forEach((col) => headRow.appendChild(node("th", "", col.label)));
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = node("tbody");
    if (!rows.length) {
      const tr = node("tr");
      const td = node("td", "", "기록 없음");
      td.colSpan = columns.length;
      tr.appendChild(td);
      tbody.appendChild(tr);
    } else {
      rows.forEach((row) => {
        const tr = node("tr");
        columns.forEach((col) => tr.appendChild(node("td", "", clip(col.value(row)))));
        tbody.appendChild(tr);
      });
    }
    table.appendChild(tbody);
    wrap.appendChild(heading);
    wrap.appendChild(table);
    return wrap;
  }

  function flattenResults() {
    return state.records.flatMap((record) =>
      (record.rows || []).map((row) => ({
        job_id: record.job_id,
        case_id: row.id,
        setting: row.setting,
        acc: row.acc,
        gap: row.real_control_gap,
      }))
    );
  }

  function flattenArtifacts() {
    return state.records.flatMap((record) =>
      (record.artifacts || []).map((artifact) => ({
        job_id: record.job_id,
        config_source: record.config_source || "",
        ...artifact,
      }))
    );
  }

  function renderDb() {
    const db = state.els.db;
    if (!db) return;
    clear(db);
    const experiments = state.records.map((record) => ({
      experiment_id: record.experiment_id,
      track: record.track,
      component: record.authoring?.component_name || "",
      status: record.status,
      created_at: record.created_at,
      config_path: record.config_path,
      config_source: record.config_source,
      config_saved_path: record.config_saved_path,
    }));
    const jobs = state.records.map((record) => ({
      job_id: record.job_id,
      experiment_id: record.experiment_id,
      status: record.status,
      stage: record.stage,
      progress: `${record.progress || 0}%`,
    }));
    const results = flattenResults();
    const artifacts = flattenArtifacts();

    db.appendChild(
      renderTable(
        "experiments",
        [
          { label: "experiment_id", value: (r) => r.experiment_id },
          { label: "track", value: (r) => r.track },
          { label: "component", value: (r) => r.component },
          { label: "status", value: (r) => r.status },
          { label: "config_source", value: (r) => r.config_source },
          { label: "config_path", value: (r) => r.config_path },
        ],
        experiments
      )
    );
    db.appendChild(
      renderTable(
        "jobs",
        [
          { label: "job_id", value: (r) => r.job_id },
          { label: "experiment_id", value: (r) => r.experiment_id },
          { label: "stage", value: (r) => r.stage },
          { label: "progress", value: (r) => r.progress },
        ],
        jobs
      )
    );
    db.appendChild(
      renderTable(
        "results",
        [
          { label: "job_id", value: (r) => r.job_id },
          { label: "case", value: (r) => r.case_id },
          { label: "acc", value: (r) => Number(r.acc).toFixed(3) },
          { label: "gap", value: (r) => Number(r.gap).toFixed(3) },
        ],
        results
      )
    );
    db.appendChild(
      renderTable(
        "artifacts",
        [
          { label: "job_id", value: (r) => r.job_id },
          { label: "source", value: (r) => r.config_source },
          { label: "type", value: (r) => r.type },
          { label: "status", value: (r) => r.status },
          { label: "path", value: (r) => r.path },
        ],
        artifacts
      )
    );

    const storageText = state.storageAvailable
      ? `${STORAGE_KEY} · ${state.records.length} run`
      : "localStorage 사용 불가 · 화면 메모리만 유지";
    setField("dbCard", storageText);
  }

  function updateActiveStage(stage, progress, message) {
    if (!state.active) return;
    state.active.stage = stage;
    state.active.progress = progress;
    setProgress(progress);
    setField("stage", stage);
    setField("progress", `${progress}%`);
    if (message) addLog(message);
  }

  function setResultStale(isStale) {
    const wrap = document.getElementById("results-wrap");
    const note = document.getElementById("results-stale-note");
    if (!wrap) return;
    wrap.classList.toggle("is-stale", isStale);
    if (note) {
      note.hidden = !isStale;
      if (isStale) {
        note.textContent = "현재 조립과 다른 Mock DB 결과입니다. 다시 실행하면 현재 조립 기준으로 갱신됩니다.";
      }
    }
  }

  function refreshResultFreshness(submission) {
    const wrap = document.getElementById("results-wrap");
    if (!wrap?.classList.contains("show")) {
      setResultStale(false);
      return;
    }
    const renderedSignature = wrap.dataset.resultSignature || latestCompleted()?.signature || "";
    setResultStale(
      !!(
        renderedSignature &&
        submission?.signature &&
        submission.signature !== renderedSignature
      )
    );
  }

  function revealMockPanel() {
    try {
      state.root?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    } catch (_) {
      state.root?.scrollIntoView();
    }
  }

  function classifyRow(row, cfg) {
    const setting = String(row.setting || "").toLowerCase();
    if (setting.includes("baseline") || row.id === "base") return "baseline";
    if (cfg?.track === "cora" && Number(row.real_control_gap) >= 0.025) return "graph 효과 강함";
    if (Number(row.real_control_gap) <= 0.01) return "control도 비슷함";
    if (setting.includes("diagnostic")) return "diagnostic 변화";
    return "setting-dependent";
  }

  function enrichRows(rows, cfg) {
    return rows.map((row) => ({
      ...row,
      interpretation_type: row.interpretation_type || classifyRow(row, cfg),
    }));
  }

  function artifactBase(submission) {
    const cfg = submission.cfg || {};
    const track = cfg.track || "unknown";
    const dataset = cfg.dataset || "dataset";
    const safeDataset = String(dataset).replace(/[^\w.-]+/g, "_").slice(0, 60) || "dataset";
    return `mock-db/${track}/${safeDataset}/${submission.configType || "single"}`;
  }

  function configSource(submission) {
    const configType = String(submission?.configType || "");
    return submission?.doc?.source || (configType.startsWith("existing") ? "existing_config" : "generated_config");
  }

  function configArtifacts(submission) {
    const source = configSource(submission);
    if (source === "existing_config") {
      return [
        {
          artifact_id: id("artifact"),
          type: "config_reference",
          path: submission.configPath || "",
          status: "linked",
        },
      ];
    }
    const items = [
      {
        artifact_id: id("artifact"),
        type: "generated_config",
        path: submission.configPath || "",
        status: "created",
      },
    ];
    const jobs = Array.isArray(submission.doc?.jobs) ? submission.doc.jobs : [];
    jobs.forEach((job) => {
      if (!job?.config_path) return;
      items.push({
        artifact_id: id("artifact"),
        type: "generated_per_run_config",
        path: job.config_path,
        status: "created",
      });
    });
    return items;
  }

  function makeArtifacts(submission) {
    const base = artifactBase(submission);
    const cfg = submission.cfg || {};
    const common = [
      ...configArtifacts(submission),
      { artifact_id: id("artifact"), type: "server_log", path: `${base}/server.log`, status: "created" },
      { artifact_id: id("artifact"), type: "result_rows", path: `${base}/result_rows.json`, status: "created" },
      { artifact_id: id("artifact"), type: "module_traces", path: `${base}/module_traces.jsonl`, status: "created" },
    ];
    if (submission.authoring?.component_name) {
      common.push({
        artifact_id: id("artifact"),
        type: "component_validation_report",
        path: `${base}/component_validation_report.json`,
        status: "linked",
      });
    }
    if (cfg.track === "cora") {
      if (submission.configType === "cora-single") {
        return [
          ...common,
          { artifact_id: id("artifact"), type: "cora_single_result", path: `${base}/result_cora_single.json`, status: "created" },
          { artifact_id: id("artifact"), type: "round_metrics", path: `${base}/round_metrics.csv`, status: "created" },
        ];
      }
      return [
        ...common,
        { artifact_id: id("artifact"), type: "graph_ablation", path: `${base}/run_graph_ablation_result.json`, status: "created" },
        { artifact_id: id("artifact"), type: "spectral_report", path: `${base}/spectral_decomposition.csv`, status: "created" },
      ];
    }
    return [
      ...common,
      { artifact_id: id("artifact"), type: "round_metrics", path: `${base}/round_metrics.csv`, status: "created" },
      { artifact_id: id("artifact"), type: "client_metrics", path: `${base}/client_metrics.csv`, status: "created" },
      { artifact_id: id("artifact"), type: "counterfactual", path: `${base}/counterfactual_metrics.csv`, status: "created" },
    ];
  }

  function makeRecord(submission, active, status, rows, artifacts) {
    const now = isoNow();
    const cfg = submission.cfg || {};
    return {
      schema_version: 2,
      experiment_id: active.experiment_id,
      job_id: active.job_id,
      created_at: active.created_at,
      completed_at: status === "completed" ? now : null,
      updated_at: now,
      status,
      stage: active.stage,
      progress: active.progress,
      track: cfg.track || "",
      config_path: submission.configPath || "",
      config_source: configSource(submission),
      config_saved_path: configSource(submission) === "generated_config" ? submission.configPath || "" : "",
      command: submission.command || "",
      config_type: submission.configType || "",
      config: submission.doc || null,
      authoring: submission.authoring || null,
      design: submission.design || null,
      caseMode: submission.rowBundle?.caseMode || "",
      signature: submission.signature || "",
      rows,
      artifacts,
    };
  }

  function saveTerminalRecord(status, rows, artifacts) {
    if (!state.active) return null;
    const record = makeRecord(state.active.submission, state.active, status, rows, artifacts);
    const remaining = state.records.filter((item) => item.job_id !== record.job_id);
    writeRecords([record, ...remaining]);
    renderDb();
    return record;
  }

  function cancelActive(reason) {
    if (!state.active) return;
    const { token } = state.active;
    clearTimers();
    if (state.active.token !== token) return;
    updateActiveStage("stale", state.active.progress || 0, reason || "active mock job cancelled");
    const artifacts = (state.active.artifacts || []).map((artifact) => ({
      ...artifact,
      status: "stale",
    }));
    saveTerminalRecord("stale", [], artifacts);
    state.active = null;
    setRunButton(false);
    setRunNote("조립 변경으로 active mock job을 stale 처리했습니다. 다시 제출할 수 있습니다.");
  }

  function completeActive() {
    if (!state.active) return;
    const submission = state.active.submission;
    const rows = enrichRows(submission.rowBundle?.rows || [], submission.cfg);
    const artifacts = (state.active.artifacts || makeArtifacts(submission)).map((artifact) => ({
      ...artifact,
      status: artifact.status === "linked" ? "linked" : "saved",
    }));
    updateActiveStage("completed", 100, "DB 저장 완료 · latest completed record 조회");
    renderArtifacts(artifacts);
    const record = saveTerminalRecord("completed", rows, artifacts);
    state.active = null;
    setRunButton(false);
    setRunNote("완료. Mock DB latest completed run에서 결과표를 렌더링했습니다.");
    if (record) window.GraphFLDemo?.renderResultsTable?.(record);
  }

  function start(submission) {
    if (!submission?.ready || !submission.doc) {
      setRunNote("Mock 제출 불가: runReadiness와 config preview가 모두 준비되어야 합니다.");
      return;
    }
    if (state.active) cancelActive("새 제출로 이전 active mock job stale 처리");
    clearTimers();

    const token = id("token");
    const active = {
      token,
      experiment_id: id("exp"),
      job_id: id("job"),
      created_at: isoNow(),
      stage: "draft",
      progress: 0,
      submission,
      artifacts: [],
    };
    state.active = active;
    setRunButton(true);
    revealMockPanel();
    clear(state.els.log);
    renderArtifacts([]);
    setField("apiCard", "draft");
    setField("serverCard", "idle");
    setField("dbCard", `${STORAGE_KEY} · writing pending`);
    setField("job", active.job_id);
    updateActiveStage("draft", 0, "draft 생성");

    schedule(token, 120, () => {
      setField("apiCard", `POST /api/experiments · ${active.experiment_id}`);
      updateActiveStage("submitted", 6, "Mock API 제출 완료");
    });
    schedule(token, 360, () => {
      setField("serverCard", `queue: ${active.job_id}`);
      updateActiveStage("queued", 14, "Mock Server queue 등록");
    });

    const totalRounds = Math.max(1, Number(submission.cfg?.rounds) || 1);
    const ticks = Math.min(12, Math.max(4, totalRounds));
    for (let i = 1; i <= ticks; i += 1) {
      schedule(token, 560 + i * 170, () => {
        const currentRound = Math.ceil((i / ticks) * totalRounds);
        const progress = Math.min(82, 16 + Math.round((i / ticks) * 64));
        setField("serverCard", `running · round ${currentRound}/${totalRounds}`);
        setField("job", `${active.job_id} · round ${currentRound}/${totalRounds}`);
        updateActiveStage("running", progress, `round ${currentRound}/${totalRounds} progress event`);
      });
    }

    const collectAt = 760 + ticks * 170;
    schedule(token, collectAt, () => {
      active.artifacts = makeArtifacts(submission);
      setField("serverCard", "collecting artifacts");
      updateActiveStage("collecting", 88, "metrics/artifacts 수집");
      renderArtifacts(active.artifacts);
    });
    schedule(token, collectAt + 260, () => {
      updateActiveStage("saved", 96, "config/results Mock DB 저장 트랜잭션");
      setField("dbCard", `${STORAGE_KEY} · saving`);
    });
    schedule(token, collectAt + 520, completeActive);
  }

  function clearDb() {
    clearTimers();
    state.active = null;
    state.records = [];
    try {
      localStorage.removeItem(STORAGE_KEY);
      state.storageAvailable = true;
    } catch (_) {
      state.storageAvailable = false;
    }
    setProgress(0);
    setField("apiCard", "draft");
    setField("serverCard", "idle");
    setField("stage", "draft");
    setField("progress", "0%");
    setField("job", "-");
    clear(state.els.log);
    renderArtifacts([]);
    renderDb();
    document.getElementById("results-wrap")?.classList.remove("show");
    setResultStale(false);
    setRunButton(false);
    setRunNote("Mock DB를 비웠습니다. 조립이 준비되면 다시 제출할 수 있습니다.");
  }

  function onAssemblyChanged(event) {
    const submission = event.detail;
    if (state.active && submission?.signature !== state.active.submission.signature) {
      cancelActive("조립 변경 감지 · active mock job stale 처리");
      refreshResultFreshness(submission);
      return;
    }
    if (!state.active) {
      setField("apiCard", submission?.ready ? "draft ready" : "draft incomplete");
      setField("stage", submission?.ready ? "draft" : "draft incomplete");
      refreshResultFreshness(submission);
    }
  }

  function buildUi() {
    const root = state.root;
    if (!root) return;
    clear(root);

    const head = node("div", "mock-head");
    head.appendChild(node("h2", "", "Mock FL execution"));
    head.appendChild(node("span", "mock-badge", "Actual validation과 분리"));
    root.appendChild(head);

    const grid = node("div", "mock-grid");
    grid.appendChild(makeCard("Mock API", "apiCard", "draft incomplete"));
    grid.appendChild(makeCard("Mock Server", "serverCard", "idle"));
    grid.appendChild(makeCard("Mock DB", "dbCard", `${STORAGE_KEY} · ${state.records.length} run`));
    root.appendChild(grid);

    const progress = node("div", "mock-progress");
    progress.setAttribute("role", "progressbar");
    progress.setAttribute("aria-label", "Mock job progress");
    progress.setAttribute("aria-valuemin", "0");
    progress.setAttribute("aria-valuemax", "100");
    progress.setAttribute("aria-valuenow", "0");
    state.els.progressTrack = progress;
    state.els.progressBar = node("div", "mock-progress-bar");
    progress.appendChild(state.els.progressBar);
    root.appendChild(progress);
    root.appendChild(makeStatusRow("stage", "stage", "draft"));
    root.appendChild(makeStatusRow("progress", "progress", "0%"));
    root.appendChild(makeStatusRow("job", "job", "-"));

    state.els.log = node("div", "mock-log");
    state.els.log.setAttribute("aria-live", "polite");
    state.els.artifacts = node("div", "mock-artifacts");
    root.appendChild(state.els.log);
    root.appendChild(state.els.artifacts);
    renderArtifacts([]);

    const actions = node("div", "mock-actions");
    const clearButton = node("button", "", "Mock DB Clear");
    clearButton.type = "button";
    clearButton.addEventListener("click", clearDb);
    actions.appendChild(clearButton);
    root.appendChild(actions);

    const dbWrap = node("div", "mock-db");
    dbWrap.appendChild(node("h3", "", "Mock DB Viewer"));
    state.els.db = node("div", "mock-db-grid");
    dbWrap.appendChild(state.els.db);
    root.appendChild(dbWrap);
  }

  function restoreLatestResult() {
    const latest = latestCompleted();
    if (latest) {
      window.GraphFLDemo?.renderResultsTable?.(latest);
      setRunNote("새로고침 후 Mock DB latest completed run을 복원했습니다.");
    }
  }

  function init() {
    state.root = document.getElementById("mock-system-root");
    if (!state.root) return;
    state.records = readRecords();
    buildUi();
    renderDb();
    restoreLatestResult();
    window.addEventListener("graphfl:assembly-changed", onAssemblyChanged);
    const submission = window.GraphFLDemo?.getSubmission?.();
    if (submission) onAssemblyChanged({ detail: submission });
  }

  window.GraphFLMockSystem = {
    start,
    cancel: cancelActive,
    refresh: renderDb,
    clear: clearDb,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
