    const IMPL = [
      "authoring: graph_source + graph_mode + aggregation_target + correction_family",
      "graph_source: update, ema_update, classifier_head_update, weight, layerwise_update, …",
      "graph_mode/control: knn, rbf_knn, random, shuffled, uniform, identity, graph_free, …",
      "aggregation_target: graph_filtered_update, graph_filtered_ema_update, graph_filtered_weight",
      "evidence: accuracy/loss, real-control gap, graph-free gap, alignment, DI/N_eff, LOO",
      "확장: register_graph_source / graph builder / GraphFLDesign / graph_method",
      "실행: run_vision_suite, run_graph_ablation, run_experiment --track (config JSON)",
    ];

    const TRACK_RULES = {
      vision: {
        datasets: new Set(["fashionmnist", "mnist", "cifar10"]),
        partition: new Set(["dirichlet", "iid"]),
      },
      cora: {
        datasets: new Set(["cora"]),
        partition: new Set([]),
      },
    };

    const ENV_SLOT_IDS = ["config_name", "track", "dataset"];
    const ENV_SLOT_ORDER = ["config_name", "track", "dataset"];
    const MAX_ENV_COLUMNS = 12;
    function effectiveRunner(track) {
      if (track === "vision") return "suite";
      if (track === "cora") return "single";
      return null;
    }

    function usesColumnAssembly() {
      return currentTrack() === "vision" && baseRowReady();
    }
    const ENV_CUSTOM_VALUES = new Set(["custom_dataset", "custom_config", "custom_partition", "custom_variant"]);
    const ENV_CUSTOM_DEFAULTS = {
      dataset: "my_dataset",
      config_name: "my_config",
      partition: "my_partition",
      sweep_variants: "ours_knn_k2",
    };
    const COMPARE_LIFE_KINDS = new Set([
      "client",
      "relation",
      "topology",
      "aggregation",
      "correction",
      "diagnostic",
      "method",
      "baseline",
      "extend",
    ]);
    const COMPARE_REQUIRED_AXES = [
      "graph_source",
      "graph_mode",
      "aggregation_target",
    ];
    const CONTROL_GRAPH_MODES = ["random", "shuffled", "uniform", "identity"];
    const BLOCK_KIND = {
      ENV: "env",
      BASELINE: "baseline",
      CLIENT: "client",
      RELATION: "relation",
      TOPOLOGY: "topology",
      AGGREGATION: "aggregation",
      CORRECTION: "correction",
      DIAGNOSTIC: "diagnostic",
      PRESET: "preset",
    };
    const LIFECYCLE_STAGES = [
      { key: "graph_source", kind: BLOCK_KIND.CLIENT, label: "graph_source", allowedKinds: [BLOCK_KIND.CLIENT], colorClass: "next-client", required: true },
      { key: "relation_or_graph_mode", kind: BLOCK_KIND.RELATION, label: "relation cue / graph_mode", allowedKinds: [BLOCK_KIND.RELATION, BLOCK_KIND.TOPOLOGY], colorClass: "next-relation", required: true },
      { key: "aggregation_target", kind: BLOCK_KIND.AGGREGATION, label: "aggregation_target", allowedKinds: [BLOCK_KIND.AGGREGATION], colorClass: "next-aggregation", required: true },
      { key: "correction", kind: BLOCK_KIND.CORRECTION, label: "선택: control/correction", allowedKinds: [BLOCK_KIND.CORRECTION], colorClass: "next-control", required: false },
    ];
    const COLOR_CLASS_BY_STAGE = {
      track: "next-track",
      dataset: "next-dataset",
      partition: "next-partition",
      baseline: "next-baseline",
      client: "next-client",
      relation: "next-relation",
      topology: "next-topology",
      aggregation: "next-aggregation",
      correction: "next-control",
      diagnostic: "next-diagnostic",
    };
    const GRAPH_FREE_VARIANT_TOKEN = {
      norm_clip: "ours_graphfree_normclip",
      contribution_cap: "ours_graphfree_cap",
      dominance_reweight: "ours_graphfree_reweight",
    };
    const BASELINE_METHOD_VALUES = new Set([
      "fedavg",
      "fedavgm",
      "graph_smooth",
      "dominance_aware",
    ]);
    const PRESET_BUNDLES = {
      default_similarity_knn: [
        { kind: "client", value: "update", title: "update", sub: "graph_source" },
        { kind: "relation", value: "rbf", title: "rbf", sub: "relation cue" },
        { kind: "topology", value: "rbf_knn", title: "rbf_knn", sub: "graph_mode" },
        {
          kind: "aggregation",
          value: "graph_filtered_update",
          title: "graph_filtered_update",
          sub: "aggregation_target",
        },
        { kind: "correction", value: "real_graph", title: "real_graph", sub: "correction" },
      ],
      pfedgraph: [
        { kind: "client", value: "update", title: "update", sub: "graph_source" },
        { kind: "relation", value: "cosine", title: "cosine", sub: "relation cue" },
        { kind: "topology", value: "pfedgraph_qp", title: "pfedgraph_qp", sub: "graph_mode" },
        {
          kind: "aggregation",
          value: "graph_filtered_update",
          title: "graph_filtered_update",
          sub: "aggregation_target",
        },
        { kind: "correction", value: "real_graph", title: "real_graph", sub: "correction" },
      ],
    };
    const EXTEND_AXIS_CLASS = {
      name_graph_source: "extend-client",
      name_relation: "extend-relation",
      name_graph_mode: "extend-topology",
      name_aggregation: "extend-aggregation",
      name_correction: "extend-control",
      name_method: "extend-method",
    };

    function isEnvCustomData(data) {
      return !!(data && (data.envCustom || ENV_CUSTOM_VALUES.has(data.value)));
    }

    function envCustomPlaceholder(slotId) {
      if (slotId === "config_name") return "custom_config";
      if (slotId === "partition") return "custom_partition";
      if (slotId === "baseline") return "custom_variant";
      if (slotId === "sweep_variants") return "custom_variant";
      return "custom_dataset";
    }

    function envCustomDefaultName(slotId) {
      return ENV_CUSTOM_DEFAULTS[slotId] || envCustomPlaceholder(slotId);
    }

    function resolvedEnvSlotValue(slot) {
      if (!slot) return null;
      if (slot.compareKind === "assembly" && slot.assembly) {
        return compareEntryVariantToken(slot);
      }
      if (slot.envCustom) {
        const name = String(slot.customName || "").trim();
        return name || null;
      }
      return slot.value || null;
    }

    function snapshotLifeAssembly() {
      const parts = columnLifecycleParts(envColumns[0]);
      return {
        ...parts,
        diagnostic:
          (envSlots.diagnostics || []).map((b) => b.value).join(",") ||
          getLife("diagnostic")?.value ||
          null,
      };
    }

    function compareAssemblyComplete(asm) {
      if (!asm) return false;
      return COMPARE_REQUIRED_AXES.every((k) => asm[k]);
    }

    function assemblyLabel(asm) {
      if (!asm) return "조립 …";
      const bits = [
        asm.graph_source && `src:${asm.graph_source}`,
        asm.relation && `rel:${asm.relation}`,
        asm.graph_mode && `mode:${asm.graph_mode}`,
        asm.aggregation_target && `agg:${asm.aggregation_target}`,
        asm.correction_family && `corr:${asm.correction_family}`,
        asm.correction_family === "control_graph" &&
          asm.control_graph_mode &&
          `ctl:${asm.control_graph_mode}`,
        asm.diagnostic && `diag:${asm.diagnostic}`,
      ].filter(Boolean);
      return bits.length ? bits.join(" · ") : "조립 …";
    }

    function assemblyAxisHtml(asm) {
      if (!asm) return "";
      const rows = [
        ["graph_source", asm.graph_source],
        ["relation", asm.relation],
        ["graph_mode", asm.graph_mode],
        ["aggregation_target", asm.aggregation_target],
        ["correction_family", asm.correction_family],
        ["control_graph_mode", asm.control_graph_mode],
        ["diagnostic", asm.diagnostic],
      ]
        .filter(([, v]) => v)
        .map(([k, v]) => `${k}: ${v}`);
      return rows.join("<br/>");
    }

    function buildVariantToken(col) {
      if (!col) return null;
      if (col.caseMode === "baseline" && col.preset) {
        const v = col.preset.method || col.preset.value;
        if (v && v !== "custom_variant") return v;
        if (col.preset.compareKind === "registry") return resolvedEnvSlotValue(col.preset);
        return col.preset.value || null;
      }
      if (col.caseMode !== "assembly") return null;
      const p = columnLifecycleParts(col);
      const k = getEnvNums().knn_k || 2;

      if (p.correction_family === "graph_free") {
        const mode = col.graphFreeMode || "dominance_reweight";
        return GRAPH_FREE_VARIANT_TOKEN[mode] || null;
      }
      if (p.correction_family === "clustering_only") return `ours_cluster_only_k${k}`;
      if (p.correction_family === "control_graph") {
        const mode = p.control_graph_mode || "random";
        return `ours_${mode}_control_k${k}`;
      }
      if (p.correction_family === "real_graph") return `ours_real_graph_k${k}`;

      if (
        p.graph_source === "update" &&
        p.relation === "cosine" &&
        p.graph_mode === "knn" &&
        p.aggregation_target === "graph_filtered_update"
      ) {
        return `ours_default_graph_k${k}`;
      }
      if (
        p.graph_source === "update" &&
        p.graph_mode === "rbf_knn" &&
        p.aggregation_target === "graph_filtered_update"
      ) {
        return `ours_rbf_knn_k${k}`;
      }
      if (p.graph_source === "update" && p.graph_mode === "knn" && p.aggregation_target === "update") {
        return `ours_knn_k${k}`;
      }
      if (p.graph_mode === "dense") return "ours_dense";
      if (p.graph_mode === "mutual_knn") return `ours_mutual_knn_k${k}`;
      if (p.graph_mode === "magnitude") return "ours_magnitude";
      if (p.graph_mode === "rbf") return "ours_rbf";
      return null;
    }

    function comparisonCaseForColumn(col, index) {
      const token = buildVariantToken(col);
      const displayLabel = columnVariantToken(col);
      const part = resolvedEnvSlotValue(col?.partition);
      const assembly = col?.caseMode === "assembly" ? columnAssembly(col) : null;
      return {
        index: index + 1,
        run_id: `run_${index + 1}`,
        case_mode: col?.caseMode || "incomplete",
        label:
          col?.caseMode === "baseline"
            ? displayLabel || col?.preset?.title || "baseline"
            : assembly
              ? assemblyLabel(assembly)
              : "incomplete",
        variant_token: token,
        partition: part || null,
        dirichlet_alpha: usesDirichletAlpha(part) ? col?.dirichletAlpha : null,
        assembly,
      };
    }

    function assemblyToVariantToken(asm) {
      if (!asm) return null;
      const stub = {
        caseMode: "assembly",
        lifeStack: [],
        controlGraphMode: asm.control_graph_mode,
        graphFreeMode: asm.graph_free_mode,
      };
      [
        ["client", asm.graph_source],
        ["relation", asm.relation],
        ["topology", asm.graph_mode],
        ["aggregation", asm.aggregation_target],
        ["correction", asm.correction_family],
        ["diagnostic", asm.diagnostic],
      ].forEach(([kind, value]) => {
        if (value) stub.lifeStack.push({ kind, value, title: value });
      });
      return buildVariantToken(stub);
    }

    function compareEntryVariantToken(entry) {
      if (!entry) return null;
      if (entry.compareKind === "assembly") {
        return assemblyToVariantToken(entry.assembly) || entry.title || "assembled";
      }
      if (entry.compareKind === "baseline") return entry.value || "fedavg";
      if (entry.compareKind === "registry") return resolvedEnvSlotValue(entry);
      return entry.value || null;
    }

    function compareSignature(entry) {
      if (entry?.compareKind === "assembly") return JSON.stringify(entry.assembly || {});
      return compareEntryVariantToken(entry) || "";
    }

    function normalizeCompareEntry(entry) {
      const out = { ...entry };
      if (out.value === "fedavg" && !out.compareKind) out.compareKind = "baseline";
      if (out.envCustom || out.value === "custom_variant") {
        out.compareKind = "registry";
        out.envCustom = true;
        const name = String(out.customName || out.value || "").trim();
        if (name && name !== "custom_variant") {
          out.customName = name;
          out.value = name;
          out.title = name;
        }
      }
      if (out.compareKind === "assembly" && out.assembly) {
        out.title = assemblyLabel(out.assembly);
        out.value = assemblyToVariantToken(out.assembly) || out.title;
        out.sub = out.sub || "lifecycle 조립";
      }
      return out;
    }

    function pushCompareEntry(entry) {
      const normalized = normalizeCompareEntry({
        ...entry,
        id: entry.id || "e" + ++idSeq,
        envSlot: "sweep_variants",
        kind: "env",
        target: "env",
      });
      const sig = compareSignature(normalized);
      const store = envSweepSlots.sweep_variants;
      if (store.some((b) => compareSignature(b) === sig)) return false;
      store.push(normalized);
      renderEnvSlots();
      applySweepStackUI();
      syncUI();
      return true;
    }

    function captureCompareFromLifeStack() {
      const asm = snapshotLifeAssembly();
      if (!compareAssemblyComplete(asm)) {
        const slot = document.querySelector('.env-h-slot[data-env-slot="sweep_variants"]');
        flashReject(slot);
        return;
      }
      pushCompareEntry({
        compareKind: "assembly",
        assembly: asm,
        title: assemblyLabel(asm),
        sub: "아래 조립 스냅샷",
      });
    }

    function addFedavgCompare() {
      pushCompareEntry({
        compareKind: "baseline",
        value: "fedavg",
        title: "fedavg",
        sub: "베이스라인",
      });
    }

    function isLifeCompareBlock(data) {
      if (!data || data.kind === "env" || data.envSlot) return false;
      if (data.kind === "extend") return !!EXTEND_AXIS[data.value];
      return COMPARE_LIFE_KINDS.has(data.kind);
    }

    function lifeBlockAxisValue(data) {
      const axis = lifeAxisKey(data);
      if (data.kind === "extend") {
        const name = String(data.customName || "").trim();
        return name || null;
      }
      return data.value || null;
    }

    function addCompareFromLifeBlock(data) {
      const axis = lifeAxisKey(data);
      const val = lifeBlockAxisValue(data);
      if (!axis || !val) return;
      const store = envSweepSlots.sweep_variants;
      const last = store[store.length - 1];
      const patch = { [axis]: val };
      if (last?.compareKind === "assembly" && !last.assembly?.[axis]) {
        last.assembly = { ...(last.assembly || {}), ...patch };
        normalizeCompareEntry(last);
        renderEnvSlots();
        syncUI();
        return;
      }
      const asm = {
        graph_source: null,
        relation: null,
        graph_mode: null,
        aggregation_target: null,
        correction_family: null,
        control_graph_mode: null,
        diagnostic: null,
        ...patch,
      };
      pushCompareEntry({
        compareKind: "assembly",
        assembly: asm,
        title: assemblyLabel(asm),
        sub: "부품 1개 — 더 끌어 쌓기",
      });
    }

    function envSlotPending(slot) {
      return !!(slot && slot.envCustom && !String(slot.customName || "").trim());
    }

    function currentTrack() {
      return envSlots.track?.value || null;
    }

    function createEnvColumn() {
      return {
        id: "col" + ++idSeq,
        partition: null,
        dirichletAlpha: 0.03,
        controlGraphMode: "random",
        caseMode: null,
        preset: null,
        lifeStack: [],
        lastStemHint: "",
      };
    }

    function lifecycleKindFromData(d) {
      if (!d) return null;
      if (d.kind === "extend") {
        const axis = EXTEND_AXIS[d.value];
        if (axis === "graph_source") return "client";
        if (axis === "relation") return "relation";
        if (axis === "graph_mode") return "topology";
        if (axis === "aggregation_target") return "aggregation";
        if (axis === "correction_family") return "correction";
        return null;
      }
      if (["client", "relation", "topology", "aggregation", "correction", "diagnostic"].includes(d.kind)) {
        return d.kind;
      }
      return null;
    }

    function columnLifecycleTrail(col) {
      if (!col) return "";
      if (col.caseMode === "baseline" && col.preset) {
        return compareEntryVariantToken(col.preset) || col.preset.title || "baseline";
      }
      return LIFECYCLE_STAGES.map((s) => {
        const b = getColumnLife(col, s.kind);
        return b?.title || b?.value;
      })
        .filter(Boolean)
        .join(" → ");
    }

    function columnGraphCoreComplete(col) {
      if (!col) return false;
      const hasGraphSource = !!getColumnLife(col, BLOCK_KIND.CLIENT);
      const hasGraphMode = !!getColumnLife(col, BLOCK_KIND.TOPOLOGY);
      const hasAggregation = !!getColumnLife(col, BLOCK_KIND.AGGREGATION);
      return !!(hasGraphSource && hasGraphMode && hasAggregation);
    }

    function getColumnNextStage(col) {
      if (!col || col.caseMode === "baseline") return null;
      if (!getColumnLife(col, BLOCK_KIND.CLIENT)) {
        return { ...LIFECYCLE_STAGES[0] };
      }
      const hasRelation = !!getColumnLife(col, BLOCK_KIND.RELATION);
      const hasTopology = !!getColumnLife(col, BLOCK_KIND.TOPOLOGY);
      if (!hasRelation && !hasTopology) {
        return { ...LIFECYCLE_STAGES[1] };
      }
      if (hasRelation && !hasTopology) {
        return { key: "graph_mode", kind: BLOCK_KIND.TOPOLOGY, label: "graph_mode / topology", allowedKinds: [BLOCK_KIND.TOPOLOGY], colorClass: "next-topology", required: true };
      }
      if (!getColumnLife(col, BLOCK_KIND.AGGREGATION)) {
        return { ...LIFECYCLE_STAGES[2] };
      }
      const hasCorrection = !!getColumnLife(col, BLOCK_KIND.CORRECTION);
      if (!hasCorrection) {
        return { ...LIFECYCLE_STAGES[3] };
      }
      return null;
    }

    function columnNextLifecycleStage(col) {
      return getColumnNextStage(col);
    }

    function lifecycleInstallCheck(col, d) {
      if (!col) return { ok: false, reason: "열 없음" };
      if (col.caseMode === "baseline") return { ok: false, reason: "baseline 열에는 Graph-FL 부품을 추가할 수 없습니다" };
      if (!columnPartitionReady(col)) return { ok: false, reason: "먼저 파티션을 선택하세요" };
      const kind = lifecycleKindFromData(d);
      if (!kind) return { ok: false, reason: "Graph-FL lifecycle 부품만 놓을 수 있습니다" };
      if (getColumnLife(col, kind) && kind !== BLOCK_KIND.CORRECTION) {
        return { ok: false, reason: `${kind} 블록은 이미 있습니다` };
      }
      const next = getColumnNextStage(col);
      if (!next) return { ok: false, reason: "Graph-FL lifecycle 조립 완료" };

      const allowed = next.allowedKinds || [next.kind];
      if (!allowed.includes(kind)) {
        const stageLabel = LIFECYCLE_STAGES.find((s) => s.kind === kind || s.allowedKinds?.includes(kind))?.label || kind;
        return { ok: false, reason: `${stageLabel}는 현재 단계(${next.label})에 놓을 수 없습니다` };
      }
      if (kind === BLOCK_KIND.CORRECTION) {
        if (!columnGraphCoreComplete(col)) return { ok: false, reason: "correction/control은 aggregation 이후에 붙일 수 있습니다" };
        if (getColumnLife(col, BLOCK_KIND.CORRECTION)) return { ok: false, reason: "correction/control은 이미 있습니다" };
      }
      if (kind === BLOCK_KIND.DIAGNOSTIC) {
        return { ok: false, reason: "diagnostic은 조립 블록이 아니라 실행 옵션입니다" };
      }
      return { ok: true };
    }

    function canInstallBlock(data, focusCol) {
      return blockCanInstall(data, focusCol);
    }

    function installBlock(data, ctx) {
      return tryInstallBlock(data, ctx);
    }

    function isBaselineBlock(d) {
      if (!d) return false;
      if (d.kind === "baseline") return true;
      if (d.envSlot === "baseline" || d.envSlot === "sweep_variants") return true;
      if (BASELINE_METHOD_VALUES.has(d.value)) return true;
      if (d.value === "custom_variant" || d.envCustom) return true;
      return false;
    }


    function nextKindClass(kind) {
      if (!kind) return "";
      return COLOR_CLASS_BY_STAGE[kind === "correction" ? "correction" : kind] || "";
    }

    function clearNextClasses(el) {
      if (!el) return;
      el.classList.remove(
        "next-track",
        "next-dataset",
        "next-partition",
        "next-baseline",
        "next-client",
        "next-relation",
        "next-topology",
        "next-aggregation",
        "next-correction",
        "next-control",
        "next-diagnostic"
      );
    }

    function applyNextClass(el, kind) {
      if (!el) return;
      clearNextClasses(el);
      const cls = nextKindClass(kind);
      if (cls) el.classList.add(cls);
    }

    function stepNextKind(step) {
      if (!step) return "";
      if (step.phase === "track") return "track";
      if (step.phase === "dataset") return "dataset";
      if (step.phase === "partition") return "partition";
      if (step.phase === "column_stem") return "baseline";
      if (step.phase === "graph") return step.nextStage?.kind || "diagnostic";
      return "";
    }

    function appendStemLine(host, className, text, optionalText) {
      const line = document.createElement("div");
      line.className = className;
      const rawText = String(text || "");
      const safeText = rawText.includes("correction/control") || rawText.includes("control/correction")
        ? "선택 단계 · control/correction 또는 바로 실행 가능"
        : rawText.includes("diagnostic")
          ? "실행 가능 · 검증은 실행 옵션에서 선택"
          : text || "";
      const suppressOptional =
        rawText.includes("diagnostic") ||
        rawText.includes("control/correction") ||
        rawText.includes("correction/control");
      line.appendChild(document.createTextNode(safeText));
      if (optionalText && !suppressOptional) {
        const optional = document.createElement("span");
        optional.className = "stem-optional";
        const rawOptional = String(optionalText);
        optional.textContent =
          rawOptional.includes("diagnostic") ||
          rawOptional.includes("control/correction") ||
          rawOptional.includes("correction/control")
            ? " (optional)"
            : optionalText;
        line.appendChild(optional);
      }
      host.appendChild(line);
    }

    function updateColumnStemHelper(col) {
      const el = document.querySelector(`.column-stem-helper[data-column-id="${col.id}"]`);
      if (!el) return;
      const trail = columnLifecycleTrail(col);
      const next = getColumnNextStage(col);
      el.classList.toggle("is-baseline-done", col.caseMode === "baseline");
      el.classList.toggle("is-runnable-assembly", col.caseMode === "assembly" && columnAssemblyRunComplete(col));
      applyNextClass(el, col.caseMode === "baseline" ? "baseline" : !col.caseMode ? "baseline" : next?.kind || "aggregation");
      el.textContent = "";
      if (col.caseMode === "baseline") {
        appendStemLine(el, "stem-trail", trail || "baseline");
        appendStemLine(el, "stem-next", "완료 · baseline 단일 줄기 종료");
      } else if (!col.caseMode) {
        appendStemLine(el, "stem-next", "다음: baseline 파란 블록 또는 graph_source 하늘색 블록");
        if (col.lastStemHint) {
          appendStemLine(el, "stem-warn", col.lastStemHint);
        }
      } else {
        if (trail) {
          appendStemLine(el, "stem-trail", trail);
        }
        if (next) {
          const optional = next.key === "correction"
            ? " (control/correction은 선택, 바로 실행 가능)"
            : "";
          appendStemLine(el, "stem-next", `다음 색깔: ${next.label}`, optional);
        } else if (columnAssemblyAnalysisComplete(col)) {
          appendStemLine(el, "stem-next", "Graph-FL 분석 완료 · 실행 옵션에서 검증 선택");
        } else if (columnAssemblyRunComplete(col)) {
          appendStemLine(el, "stem-next", "실행 가능 · 필요하면 실행 옵션에서 검증 선택");
        }
        if (col.lastStemHint) {
          appendStemLine(el, "stem-warn", col.lastStemHint);
        }
      }
    }

    function updateAllColumnStemHelpers() {
      envColumns.forEach(updateColumnStemHelper);
    }

    function inferColumnCaseMode(col) {
      if (!col || col.caseMode) return;
      if (col.lifeStack.length) col.caseMode = "assembly";
      else if (col.preset) col.caseMode = "baseline";
    }

    function columnPartitionReady(col) {
      if (!col) return false;
      if (currentTrack() !== "vision") return true;
      return !!resolvedEnvSlotValue(col.partition);
    }

    function columnStemOpen(col) {
      if (!col) return false;
      return columnPartitionReady(col) && !col.caseMode;
    }

    function columnAssemblyUnlocked(colId) {
      const col = getColumnById(colId);
      if (!baseRowReady() || !col) return false;
      if (!columnPartitionReady(col)) return false;
      return col.caseMode === "assembly";
    }

    function baseRowReady() {
      return !!(envSlots.track && resolvedEnvSlotValue(envSlots.dataset));
    }

    function existingConfigLocked() {
      const cfg = buildEnvConfig();
      const selection = resolveConfigSelection(cfg);
      const selectedTrack = cfg.track || selection.track;
      return !!(cfg.config_name && selection.existing && selectedTrack && selection.validation.ok);
    }

    function configInputReadyForAssembly() {
      const cfg = buildEnvConfig();
      const selection = resolveConfigSelection(cfg);
      return !!(cfg.config_name && selection.validation.ok);
    }

    function blockAllowedWhileExistingLocked(data) {
      return data?.envSlot === "config_name" || data?.value === "custom_config";
    }

    function currentAssemblyStep() {
      const cfg = buildEnvConfig();
      const configSelection = resolveConfigSelection(cfg);
      if (!resolvedEnvSlotValue(envSlots.config_name) || !configSelection.validation.ok) return { phase: "config_name" };
      if (configSelection.existing && (cfg.track || configSelection.track)) return { phase: "done" };
      if (!envSlots.track) return { phase: "track" };
      if (!resolvedEnvSlotValue(envSlots.dataset)) return { phase: "dataset" };
      if (!baseRowReady()) return { phase: "dataset" };
      if (currentTrack() === "cora") return { phase: "done" };
      syncEnvColumnsArray();
      const n = getTargetColumnCount();
      const track = currentTrack();
      for (let i = 0; i < n; i++) {
        const col = envColumns[i];
        if (!col) continue;
        if (track === "vision" && !resolvedEnvSlotValue(col.partition)) {
          return { phase: "partition", colIndex: i, colId: col.id };
        }
        inferColumnCaseMode(col);
        if (!col.caseMode) {
          return { phase: "column_stem", colIndex: i, colId: col.id };
        }
        if (col.caseMode === "assembly" && !columnAssemblyComplete(col)) {
          const next = getColumnNextStage(col);
          return { phase: "graph", colIndex: i, colId: col.id, nextStage: next };
        }
      }
      return { phase: "done" };
    }

    function stepBannerText(step) {
      const n = step.colIndex != null ? step.colIndex + 1 : 1;
      if (step.phase === "track") {
        return "② 트랙 — 새 JSON 저장 흐름에서 Vision 또는 Cora를 고르기";
      }
      if (step.phase === "dataset") {
        return "③ 데이터 — 새 JSON에 저장할 데이터셋을 고르기 · 트랙은 언제든 다시 바꿀 수 있음";
      }
      if (step.phase === "config_name") {
        return "① config JSON — 기존 configs/...json을 쓰거나 새 JSON 이름을 정하기";
      }
      if (step.phase === "partition") {
        return `비교 열 ${n} — 파티션 블록을 이 열에 붙이기 (Dirichlet이면 블록 안 α)`;
      }
      if (step.phase === "column_stem") {
        return `비교 열 ${n} — baseline 한 블록(완료) 또는 graph_source로 Graph-FL 줄기 시작`;
      }
      if (step.phase === "graph") {
        const next = step.nextStage?.label || "Graph-FL block";
        const optional = step.nextStage?.kind === "correction" ? " · 선택 사항" : "";
        return `비교 열 ${n} — Graph-FL 줄기 · 다음 색깔 블록: ${next}${optional}`;
      }
      return "조립 완료 — 각 단계는 접어서 다시 열고 바꿀 수 있음";
    }

    function isLifeStackBlock(data) {
      if (!data) return false;
      if (data.kind === "method") return false;
      if (data.target === "life" && data.kind !== "env") return true;
      return ["client", "relation", "topology", "aggregation", "correction", "extend"].includes(
        data.kind
      );
    }

    function columnGraphUnlocked(colId) {
      return columnAssemblyUnlocked(colId);
    }

    function targetColumnForBlock(data, preferredColId) {
      if (preferredColId) return getColumnById(preferredColId);
      const step = currentAssemblyStep();
      if (step.colId) return getColumnById(step.colId);
      if (data.envSlot === "partition") {
        return envColumns.find((c) => !resolvedEnvSlotValue(c.partition)) || envColumns[0];
      }
      if (isBaselineBlock(data)) {
        return envColumns.find((c) => columnStemOpen(c)) || envColumns[0];
      }
      if (isLifeStackBlock(data)) {
        return (
          envColumns.find((c) => c.caseMode === "assembly" && getColumnNextStage(c)) ||
          envColumns.find((c) => columnStemOpen(c)) ||
          envColumns[0]
        );
      }
      return envColumns[0];
    }

    function tryInstallBlock(data, ctx = {}) {
      const d = normalizeEnvDrop({ ...data, target: data.target || (data.kind === "env" ? "env" : "life") });
      const colId = ctx.colId || d.envColumn || null;
      const slotId = ctx.slotId || d.envSlot || null;
      const onStack = ctx.slotEl?.classList?.contains("column-life-stack");

      if (existingConfigLocked() && !blockAllowedWhileExistingLocked(d)) {
        flashReject(ctx?.slotEl);
        return false;
      }
      if (!configInputReadyForAssembly() && !blockAllowedWhileExistingLocked(d)) {
        flashReject(ctx?.slotEl);
        return false;
      }

      if (d.kind === "diagnostic" && d.target === "env" && (d.envSlot === "diagnostics" || slotId === "diagnostics") && !colId && !onStack) {
        if (!baseRowReady()) {
          flashReject(ctx?.slotEl);
          return false;
        }
        addEnvStackBlock("diagnostics", { ...d, kind: "diagnostic", target: "env", envSlot: "diagnostics" });
        return true;
      }

      const directEnvSlot =
        d.envSlot === "track" || d.value === "vision" || d.value === "cora"
          ? "track"
          : d.envSlot === "dataset" || d.value === "custom_dataset"
            ? "dataset"
            : d.envSlot === "config_name" || d.value === "custom_config"
              ? "config_name"
              : null;

      if (directEnvSlot === "track") {
        if (!envValueAllowed("track", d.value, null, d)) {
          flashReject(ctx?.slotEl);
          return false;
        }
        d.envSlot = "track";
        setEnvSlot("track", d);
        if (d.value === "cora") {
          setEnvSlot("dataset", {
            id: "e" + ++idSeq,
            target: "env",
            envSlot: "dataset",
            kind: "env",
            value: "cora",
            title: "Cora",
            sub: "Planetoid",
          });
        }
        return true;
      }
      if (directEnvSlot === "dataset") {
        if (!envSlotUnlocked("dataset") || !envValueAllowed("dataset", d.value, currentTrack(), d)) {
          flashReject(ctx?.slotEl);
          return false;
        }
        d.envSlot = "dataset";
        setEnvSlot("dataset", d);
        return true;
      }
      if (directEnvSlot === "config_name") {
        if (!envSlotUnlocked("config_name") || !envValueAllowed("config_name", d.value, currentTrack(), d)) {
          flashReject(ctx?.slotEl);
          return false;
        }
        d.envSlot = "config_name";
        setEnvSlot("config_name", d);
        return true;
      }

      if (
        colId &&
        (isBaselineBlock(d) ||
          ctx.slotEl?.dataset?.envSlot === "column_stem" ||
          ctx.slotEl?.classList?.contains("env-case-stem-slot"))
      ) {
        const col = getColumnById(colId);
        if (!col || !columnPartitionReady(col)) {
          if (col) col.lastStemHint = "먼저 파티션";
          flashReject(ctx?.slotEl);
          updateColumnStemHelper(col);
          return false;
        }
        if (isLifeStackBlock(d) && !isBaselineBlock(d)) {
          if (!col.caseMode) {
            col.caseMode = "assembly";
            col.preset = null;
            renderCasesChain();
          }
          const stackEl = document.querySelector(`.column-life-stack[data-column-id="${col.id}"]`);
          return tryInstallBlock(d, { colId: col.id, slotEl: stackEl || ctx.slotEl });
        }
        if (isBaselineBlock(d)) {
          if (col.caseMode === "assembly") {
            col.lastStemHint = "Graph-FL 열에는 baseline 불가";
            flashReject(ctx?.slotEl);
            updateColumnStemHelper(col);
            return false;
          }
          setColumnBaseline(col.id, d);
          col.lastStemHint = "";
          return true;
        }
        if (d.kind === "preset" && PRESET_BUNDLES[d.value]) {
          installColumnPreset(col, d.value);
          return true;
        }
      }

      if (onStack || (isLifeStackBlock(d) && colId) || ctx.slotEl?.classList?.contains("column-life-stack")) {
        const col = targetColumnForBlock(d, colId);
        if (!col) {
          flashReject(ctx?.slotEl);
          return false;
        }
        if (isLifeStackBlock(d) && !col.caseMode) {
          col.caseMode = "assembly";
          col.preset = null;
          renderCasesChain();
          const stackEl = document.querySelector(`.column-life-stack[data-column-id="${col.id}"]`);
          return tryInstallBlock(d, { colId: col.id, slotEl: stackEl });
        }
        if (isBaselineBlock(d)) {
          if (!columnPartitionReady(col)) {
            col.lastStemHint = "먼저 파티션";
            flashReject(ctx?.slotEl);
            updateColumnStemHelper(col);
            return false;
          }
          if (col.caseMode === "assembly") {
            col.lastStemHint = "Graph-FL 열에는 baseline 불가";
            flashReject(ctx?.slotEl);
            updateColumnStemHelper(col);
            return false;
          }
          setColumnBaseline(col.id, d);
          col.lastStemHint = "";
          return true;
        }
        if (col.caseMode === "baseline") {
          col.lastStemHint = "baseline 열은 완료됨";
          flashReject(ctx?.slotEl);
          updateColumnStemHelper(col);
          return false;
        }
        if (d.kind === "preset" && PRESET_BUNDLES[d.value]) {
          if (!columnPartitionReady(col)) {
            col.lastStemHint = "먼저 파티션";
            flashReject(ctx?.slotEl);
            updateColumnStemHelper(col);
            return false;
          }
          installColumnPreset(col, d.value);
          return true;
        }
        if (d.kind === "diagnostic") {
          const check = lifecycleInstallCheck(col, d);
          if (!check.ok) {
            col.lastStemHint = check.reason;
            flashReject(ctx?.slotEl);
            updateColumnStemHelper(col);
            return false;
          }
          col.caseMode = "assembly";
          col.preset = null;
          const stackEl =
            ctx.slotEl?.classList?.contains("column-life-stack")
              ? ctx.slotEl
              : document.querySelector(`.column-life-stack[data-column-id="${col.id}"]`);
          if (!stackEl) {
            renderCasesChain();
            return tryInstallBlock(d, { colId: col.id, slotEl: document.querySelector(`.column-life-stack[data-column-id="${col.id}"]`) });
          }
          replaceKind(col.lifeStack, stackEl, { ...d, target: "life", envColumn: col.id }, "life", col.id);
          col.lastStemHint = "";
          return true;
        }
        const check = lifecycleInstallCheck(col, d);
        if (!check.ok) {
          col.lastStemHint = check.reason;
          flashReject(ctx?.slotEl);
          updateColumnStemHelper(col);
          return false;
        }
        col.lastStemHint = "";
        col.caseMode = "assembly";
        col.preset = null;
        const stackEl =
          ctx.slotEl?.classList?.contains("column-life-stack")
            ? ctx.slotEl
            : document.querySelector(`.column-life-stack[data-column-id="${col.id}"]`);
        replaceKind(col.lifeStack, stackEl, { ...d, target: "life", envColumn: col.id }, "life", col.id);
        return true;
      }

      if (colId && slotId === "partition") {
        if (!envSlotUnlocked("partition", colId)) {
          flashReject(ctx?.slotEl);
          return false;
        }
        setColumnPartition(colId, { ...d, envSlot: "partition" });
        return true;
      }
      const explicitEnvSlot =
        d.envSlot &&
        d.envSlot !== "partition" &&
        d.envSlot !== "column_stem"
          ? d.envSlot
          : null;
      const inferredEnvSlot =
        (d.value === "vision" || d.value === "cora" ? "track" : null) ||
        (d.envSlot === "partition" || d.value === "dirichlet" || d.value === "iid" ? "partition" : null) ||
        (d.envSlot === "config_name" || d.value === "custom_config" ? "config_name" : null) ||
        (d.envSlot === "dataset" || d.value === "custom_dataset" ? "dataset" : null);
      const envSlot =
        explicitEnvSlot ||
        inferredEnvSlot ||
        slotId;

      if (envSlot === "track") {
        if (!envValueAllowed("track", d.value, null, d)) {
          flashReject(ctx?.slotEl);
          return false;
        }
        d.envSlot = "track";
        setEnvSlot("track", d);
        return true;
      }
      if (envSlot === "dataset") {
        if (!envSlotUnlocked("dataset") || !envValueAllowed("dataset", d.value, currentTrack(), d)) {
          flashReject(ctx?.slotEl);
          return false;
        }
        d.envSlot = "dataset";
        setEnvSlot("dataset", d);
        return true;
      }
      if (envSlot === "config_name") {
        if (!envSlotUnlocked("config_name") || !envValueAllowed("config_name", d.value, currentTrack(), d)) {
          flashReject(ctx?.slotEl);
          return false;
        }
        d.envSlot = "config_name";
        setEnvSlot("config_name", d);
        return true;
      }
      if (isBaselineBlock(d) && !colId) {
        const col = targetColumnForBlock(d, null);
        if (!col || !columnStemOpen(col)) {
          if (col) col.lastStemHint = col.caseMode ? "열 분기 완료" : "먼저 파티션";
          flashReject(ctx?.slotEl);
          updateColumnStemHelper(col);
          return false;
        }
        setColumnBaseline(col.id, d);
        return true;
      }
      if (isLifeStackBlock(d) && !colId) {
        const col = targetColumnForBlock(d, null);
        if (!col) {
          flashReject(ctx?.slotEl);
          return false;
        }
        return tryInstallBlock({ ...d, envColumn: col.id }, { colId: col.id, slotEl: ctx?.slotEl });
      }
      if (envSlot === "diagnostics" || (d.kind === "diagnostic" && d.target === "env")) {
        if (!baseRowReady()) {
          flashReject(ctx?.slotEl);
          return false;
        }
        addEnvStackBlock("diagnostics", { ...d, kind: "diagnostic", target: "env", envSlot: "diagnostics" });
        return true;
      }
      if (envSlot === "partition" || d.envSlot === "partition") {
        const col = targetColumnForBlock(d, colId);
        if (!col || !envSlotUnlocked("partition", col.id)) {
          flashReject(ctx?.slotEl);
          return false;
        }
        setColumnPartition(col.id, { ...d, envSlot: "partition" });
        return true;
      }

      flashReject(ctx?.slotEl);
      return false;
    }

    function blockCanInstall(data, focusCol) {
      const d = normalizeEnvDrop({ ...data, target: data.target || (data.kind === "env" ? "env" : "life") });
      if (existingConfigLocked() && !blockAllowedWhileExistingLocked(d)) return false;
      if (!configInputReadyForAssembly() && !blockAllowedWhileExistingLocked(d)) return false;
      const col = focusCol || targetColumnForBlock(d, d.envColumn);
      if (d.kind === "preset" && PRESET_BUNDLES[d.value]) {
        return col && columnPartitionReady(col) && col.caseMode !== "baseline";
      }
      if (d.kind === "diagnostic" && d.target === "life") {
        return !!(col && columnPartitionReady(col) && col.caseMode !== "baseline" && lifecycleInstallCheck(col, d).ok);
      }
      if (isLifeStackBlock(d) && d.target !== "env") {
        if (!col || !columnPartitionReady(col)) return false;
        if (col.caseMode === "baseline") return false;
        return lifecycleInstallCheck(col, d).ok;
      }
      if (isBaselineBlock(d)) {
        return col && columnStemOpen(col);
      }
      if (d.envSlot === "partition" || d.value === "dirichlet" || d.value === "iid") {
        return col && envSlotUnlocked("partition", col.id);
      }
      if (d.envSlot === "track" || d.value === "vision" || d.value === "cora") {
        return envValueAllowed("track", d.value, null, d);
      }
      if (d.envSlot === "dataset" || d.value === "custom_dataset") {
        return envSlotUnlocked("dataset") && envValueAllowed("dataset", d.value, currentTrack(), d);
      }
      if (d.envSlot === "config_name" || d.value === "custom_config") {
        return envSlotUnlocked("config_name") && envValueAllowed("config_name", d.value, currentTrack(), d);
      }
      if (d.envSlot === "diagnostics" || (d.kind === "diagnostic" && d.target === "env")) {
        return baseRowReady();
      }
      return false;
    }

    function getTargetColumnCount() {
      const track = currentTrack();
      if (track === "cora") return 1;
      const n = parseInt(String(envColumnCount), 10);
      return Math.max(1, Math.min(MAX_ENV_COLUMNS, Number.isNaN(n) ? 1 : n));
    }

    function syncEnvColumnsArray() {
      const n = getTargetColumnCount();
      while (envColumns.length < n) envColumns.push(createEnvColumn());
      while (envColumns.length > n) envColumns.pop();
    }

    function setEnvColumnCount(n) {
      const track = currentTrack();
      const max = track === "cora" ? 1 : MAX_ENV_COLUMNS;
      envColumnCount = Math.max(1, Math.min(max, parseInt(String(n), 10) || 1));
      const inp = document.getElementById("input-col-count");
      if (inp) inp.value = String(envColumnCount);
      syncEnvColumnsArray();
      renderCasesChain();
      setupEnvSlotDrops();
      setupColumnLifeDrops();
      syncUI();
    }

    function getColumnById(colId) {
      return envColumns.find((c) => c.id === colId) || null;
    }

    function getColumnLife(col, kind) {
      if (!col) return null;
      const items = col.lifeStack.filter((b) => b.kind === kind);
      return items.length ? items[items.length - 1] : null;
    }

    function columnLifecycleParts(col) {
      if (!col) {
        return {
          graph_source: null,
          relation: null,
          graph_mode: null,
          aggregation_target: null,
          correction_family: null,
          control_graph_mode: null,
          diagnostic: null,
        };
      }
      const corr = getColumnLife(col, "correction");
      const corrFam = corr?.value ?? null;
      return {
        graph_source: getColumnLife(col, "client")?.value ?? null,
        relation: getColumnLife(col, "relation")?.value ?? null,
        graph_mode: getColumnLife(col, "topology")?.value ?? null,
        aggregation_target: getColumnLife(col, "aggregation")?.value ?? null,
        correction_family: corrFam,
        control_graph_mode:
          corrFam === "control_graph" ? col.controlGraphMode || "random" : null,
        diagnostic: getColumnLife(col, "diagnostic")?.value ?? null,
      };
    }

    function columnAssembly(col) {
      const parts = columnLifecycleParts(col);
      if (parts.graph_source || parts.graph_mode || parts.aggregation_target) return parts;
      if (col.preset?.compareKind === "assembly" && col.preset.assembly) return { ...col.preset.assembly };
      return parts;
    }

    function columnVariantToken(col) {
      const token = buildVariantToken(col);
      if (token) return token;
      if (col.preset) return compareEntryVariantToken(col.preset) || resolvedEnvSlotValue(col.preset);
      return assemblyLabel(columnAssembly(col));
    }

    function columnAssemblyRunComplete(col) {
      return !!(col && col.caseMode === "assembly" && columnGraphCoreComplete(col));
    }

    function columnAssemblyAnalysisComplete(col) {
      return !!(columnAssemblyRunComplete(col) && getColumnLife(col, BLOCK_KIND.DIAGNOSTIC));
    }

    function columnAssemblyComplete(col) {
      return columnAssemblyRunComplete(col);
    }

    function columnHasGraph(col) {
      if (!col) return false;
      if (col.caseMode === "baseline") return !!col.preset;
      return columnAssemblyRunComplete(col);
    }

    function columnCaseGraphReady(col) {
      if (!col) return false;
      if (!columnPartitionReady(col)) return false;
      inferColumnCaseMode(col);
      if (!col.caseMode) return false;
      if (col.caseMode === "baseline") return !!col.preset;
      return columnAssemblyRunComplete(col);
    }

    function resetEnvColumns(nextCount = 1) {
      envColumns.length = 0;
      envColumnCount = Math.max(1, Math.min(MAX_ENV_COLUMNS, parseInt(String(nextCount), 10) || 1));
      const inp = document.getElementById("input-col-count");
      if (inp) inp.value = String(envColumnCount);
    }

    const CASE_SLOT_PLACEHOLDER = {
      partition: "← 파티션",
      column_stem: "← baseline 또는 graph_source",
    };

    function mountCaseSlotPlaceholder(slotEl, slotId) {
      slotEl.querySelectorAll(".workspace-block").forEach((n) => n.remove());
      const label = CASE_SLOT_PLACEHOLDER[slotId] || "←";
      const phBlock = document.createElement("div");
      phBlock.className = "scratch-block stack column env-case-placeholder workspace-block";
      phBlock.innerHTML = `<div class="body"><span class="env-h-ph">${label}</span></div>`;
      slotEl.appendChild(phBlock);
    }

    function makePartitionBlock(col, data) {
      const el = makeNode({ ...data, kind: "column" }, "column");
      const partVal = resolvedEnvSlotValue(data);
      if (!usesDirichletAlpha(partVal)) return el;
      el.classList.add("env-partition-dirichlet");
      const body = el.querySelector(".body");
      if (!body) return el;
      const alphaWrap = document.createElement("label");
      alphaWrap.className = "env-partition-alpha";
      alphaWrap.innerHTML = `α <input type="number" class="col-alpha-input" min="0.01" max="1" step="0.01" value="${col.dirichletAlpha}" />`;
      body.appendChild(alphaWrap);
      const alphaInput = alphaWrap.querySelector(".col-alpha-input");
      alphaInput?.addEventListener("input", (e) => {
        const next = parseFloat(e.target.value);
        if (Number.isFinite(next) && next > 0) col.dirichletAlpha = next;
        renderPreview();
        updateRunNote();
      });
      alphaInput?.addEventListener("change", () => syncUI());
      return el;
    }

    function renderColumnSlot(col, slotId, storeKey) {
      const slotEl = document.querySelector(
        `.env-h-slot[data-env-column="${col.id}"][data-env-slot="${slotId}"]`
      );
      if (!slotEl) return;
      slotEl.querySelectorAll(".workspace-block").forEach((n) => n.remove());
      const data = col[storeKey];
      if (data) {
        const cmp =
          storeKey === "partition"
            ? makePartitionBlock(col, data)
            : storeKey === "method"
              ? makeNode({ ...data, kind: "method" }, "column")
              : storeKey === "preset" && (data.compareKind === "assembly" || data.compareKind === "baseline")
                ? (() => {
                    const n = makeCompareNode({ ...data, kind: "column", envSlot: "preset" }, "env");
                    n.classList.add("column");
                    return n;
                  })()
                : makeNode({ ...data, kind: "column" }, "column");
        slotEl.appendChild(cmp);
        slotEl.classList.add("filled");
      } else {
        slotEl.classList.remove("filled");
        mountCaseSlotPlaceholder(slotEl, slotId);
      }
    }

    function makeCorrectionNode(col, data) {
      const el = makeNode({ ...data, kind: "correction" }, "life");
      if (data.value !== "control_graph") return el;
      el.classList.add("env-correction-control-graph");
      const body = el.querySelector(".body");
      if (!body) return el;
      const modeWrap = document.createElement("div");
      modeWrap.className = "env-control-graph-mode";
      CONTROL_GRAPH_MODES.forEach((mode) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = mode;
        btn.classList.toggle("is-active", (col.controlGraphMode || "random") === mode);
        btn.addEventListener("mousedown", (e) => e.stopPropagation());
        btn.addEventListener("click", (e) => {
          e.stopPropagation();
          col.controlGraphMode = mode;
          modeWrap.querySelectorAll("button").forEach((b) => b.classList.remove("is-active"));
          btn.classList.add("is-active");
          syncUI();
        });
      });
      body.appendChild(modeWrap);
      return el;
    }

    function renderColumnLifeStack(col) {
      const stackEl = document.querySelector(`.column-life-stack[data-column-id="${col.id}"]`);
      if (!stackEl) return;
      const socket = stackEl.querySelector(".column-life-socket");
      stackEl.querySelectorAll(".workspace-block").forEach((n) => n.remove());
      col.lifeStack.forEach((data) => {
        if (data.kind === "correction") {
          stackEl.appendChild(makeCorrectionNode(col, data));
        } else {
          stackEl.appendChild(makeNode(data, "life"));
        }
      });
      if (socket) {
        const next = getColumnNextStage(col);
        socket.classList.toggle("hidden", !next);
        clearNextClasses(socket);
        if (next) {
          applyNextClass(socket, next.kind);
          const hint = socket.querySelector(".socket-hint");
          if (hint) hint.textContent = `다음: ${next.label}`;
        } else {
          const hint = socket.querySelector(".socket-hint");
          if (hint) hint.textContent = "Graph-FL complete";
        }
        stackEl.appendChild(socket);
      }
      const inCaseStack = !!stackEl.closest(".env-case-stack");
      col.lifeStack.forEach((_, i, arr) => {
        const nodes = stackEl.querySelectorAll(".workspace-block:not(.column-life-socket)");
        const el = nodes[i];
        if (!el) return;
        el.style.position = "relative";
        el.style.zIndex = String(i + 1);
        el.style.marginTop = i === 0 && inCaseStack ? "0" : i === 0 ? "-10px" : "-10px";
      });
    }

    function renderCasesChain() {
      const chain = document.getElementById("env-cases-chain");
      if (!chain) return;
      syncEnvColumnsArray();
      const n = getTargetColumnCount();
      const track = currentTrack();
      chain.innerHTML = "";
      for (let i = 0; i < n; i++) {
        const col = envColumns[i];
        const showPart = track === "vision";

        const colWrap = document.createElement("div");
        colWrap.className = "env-case-column";
        colWrap.dataset.columnId = col.id;

        const runStem = document.createElement("div");
        runStem.className = "run-stem";
        runStem.dataset.columnId = col.id;
        runStem.innerHTML = `<div class="run-stem-cap" aria-hidden="true"></div>`;

        const runBody = document.createElement("div");
        runBody.className = "run-stem-body";
        runBody.dataset.columnId = col.id;

        const head = document.createElement("div");
        head.className = "env-cases-head";
        head.textContent = `Run ${i + 1}`;
        runBody.appendChild(head);

        const stack = document.createElement("div");
        stack.className = "env-case-stack";
        stack.dataset.columnId = col.id;

        const socket = document.createElement("div");
        socket.className = "scratch-block hat env env-case-socket workspace-block";
        socket.setAttribute("aria-hidden", "true");
        socket.innerHTML = `<div class="body"></div>`;
        stack.appendChild(socket);

        if (showPart) {
          const partSlot = document.createElement("div");
          partSlot.className = "env-h-slot";
          partSlot.dataset.envSlot = "partition";
          partSlot.dataset.envColumn = col.id;
          stack.appendChild(partSlot);
          mountCaseSlotPlaceholder(partSlot, "partition");
        }

        inferColumnCaseMode(col);

        if (!col.caseMode) {
          const stemSlot = document.createElement("div");
          stemSlot.className = "env-h-slot env-case-stem-slot";
          stemSlot.dataset.envSlot = "column_stem";
          stemSlot.dataset.envColumn = col.id;
          stack.appendChild(stemSlot);
          mountCaseSlotPlaceholder(stemSlot, "column_stem");
        } else if (col.caseMode === "baseline" && col.preset) {
          const baseBlk = makeCompareNode({ ...col.preset, kind: "baseline" }, "env");
          baseBlk.classList.add("env-case-baseline-block");
          stack.appendChild(baseBlk);
        } else if (col.caseMode === "assembly") {
          const lifeWrap = document.createElement("div");
          lifeWrap.className = "script-stack column-life-stack";
          lifeWrap.dataset.columnId = col.id;
          lifeWrap.dataset.accept = "life";
          const next = getColumnNextStage(col);
          const nextKind = next?.kind || "aggregation";
          const hint = next ? `다음: ${next.label}` : "Graph-FL complete";
          lifeWrap.innerHTML = `<div class="column-life-socket scratch-block stack client workspace-block"><div class="body"><span class="socket-hint">${hint}</span></div></div>`;
          const socketNode = lifeWrap.querySelector(".column-life-socket");
          applyNextClass(socketNode, nextKind);
          stack.appendChild(lifeWrap);
        }

        const helper = document.createElement("div");
        helper.className = "column-stem-helper";
        helper.dataset.columnId = col.id;

        runBody.appendChild(stack);
        runBody.appendChild(helper);
        runStem.appendChild(runBody);
        colWrap.appendChild(runStem);
        chain.appendChild(colWrap);

        if (showPart) renderColumnSlot(col, "partition", "partition");
        if (!col.caseMode) renderColumnSlot(col, "column_stem", "column_stem");
        if (col.caseMode === "assembly") renderColumnLifeStack(col);
        updateColumnStemHelper(col);
      }
    }

    function expandFoldSection(section) {
      if (!section) return;
      section.classList.remove("is-collapsed");
      const head = section.querySelector(":scope > .fold-head");
      if (head) head.setAttribute("aria-expanded", "true");
    }

    function applyComparePaletteSections() {
      const show = baseRowReady() && currentTrack() === "vision";
      document.getElementById("fold-palette-column")?.classList.toggle("palette-block-hidden", !show);
      document.getElementById("fold-palette-baseline")?.classList.toggle("palette-block-hidden", !show);
      document.getElementById("fold-palette-life")?.classList.toggle("palette-block-hidden", !show);
      document.getElementById("fold-palette-preset")?.classList.toggle("palette-block-hidden", !show);
    }

    function applyCasesZoneUI() {
      const show = baseRowReady();
      const zone = document.getElementById("env-cases-zone");
      if (zone) zone.classList.toggle("is-dimmed", !show);
      const casesPiece = document.getElementById("piece-cases");
      const track = currentTrack();
      const hideCases = track === "cora";
      if (casesPiece) {
        casesPiece.classList.toggle("palette-block-hidden", hideCases);
        casesPiece.classList.toggle("is-locked", hideCases);
      }
      if (track === "cora" && envColumnCount !== 1) envColumnCount = 1;
      const inp = document.getElementById("input-col-count");
      if (inp) inp.value = String(getTargetColumnCount());
      applyComparePaletteSections();
      syncEnvColumnsArray();
      renderCasesChain();
    }

    function setColumnPartition(colId, data) {
      const col = getColumnById(colId);
      if (!col) return;
      const entry = normalizeEnvDrop({
        ...data,
        id: data.id && data.move ? data.id : "e" + ++idSeq,
        envSlot: "partition",
        kind: "column",
        target: "env",
      });
      delete entry.move;
      delete entry.from;
      col.partition = entry;
      renderCasesChain();
      applyEnvChainGating();
      syncUI();
    }

    function setColumnBaseline(colId, data) {
      const col = getColumnById(colId);
      if (!col) return;
      let entry = normalizeEnvDrop({
        ...data,
        id: data.id && data.move ? data.id : "e" + ++idSeq,
        envSlot: "baseline",
        kind: "baseline",
        target: "env",
      });
      delete entry.move;
      delete entry.from;
      if (BASELINE_METHOD_VALUES.has(entry.value)) {
        entry = {
          compareKind: "baseline",
          value: entry.value,
          title: entry.title || entry.value,
          sub: entry.sub || `baseline · --method ${entry.value}`,
          id: entry.id,
          kind: "baseline",
          method: entry.value,
        };
      } else if (entry.value === "custom_variant" || entry.envCustom) {
        entry = normalizeCompareEntry({
          compareKind: "registry",
          value: entry.value,
          title: entry.title,
          sub: entry.sub || "repo variant",
          id: entry.id,
          kind: "baseline",
        });
      } else {
        entry = normalizeCompareEntry({
          ...entry,
          compareKind: entry.compareKind || "baseline",
          kind: "baseline",
        });
      }
      col.caseMode = "baseline";
      col.lifeStack = [];
      col.preset = entry;
      col.lastStemHint = "";
      renderCasesChain();
      applyEnvChainGating();
      syncUI();
    }

    function installColumnPreset(col, presetId) {
      const bundle = PRESET_BUNDLES[presetId];
      if (!col || !bundle) return;
      col.caseMode = "assembly";
      col.preset = null;
      col.lifeStack = bundle.map((b) => ({
        ...b,
        id: "b" + ++idSeq,
        target: "life",
        envColumn: col.id,
      }));
      col.lastStemHint = "";
      renderCasesChain();
      applyEnvChainGating();
      syncUI();
    }

    function setupColumnCountControls() {
      const minus = document.getElementById("btn-col-minus");
      const plus = document.getElementById("btn-col-plus");
      const inp = document.getElementById("input-col-count");
      if (!minus || minus.dataset.bound === "1") return;
      minus.dataset.bound = "1";
      plus.dataset.bound = "1";
      inp.dataset.bound = "1";
      minus.addEventListener("click", () => setEnvColumnCount(envColumnCount - 1));
      plus.addEventListener("click", () => setEnvColumnCount(envColumnCount + 1));
      inp.addEventListener("change", () => setEnvColumnCount(inp.value));
      inp.addEventListener("input", () => setEnvColumnCount(inp.value));
    }

    function setupColumnLifeDrops() {
      /* handled by setupWorkspaceDropDelegation */
    }

    const SWEEP_SLOT_IDS = [
      "partition",
      "sweep_variants",
      "sweep_client_counts",
      "sweep_knn_ks",
      "sweep_templates",
    ];
    const SWEEP_ROWS = [{ slot: "sweep_variants", label: "baseline" }];

    function isPartitionStackMode() {
      return usesColumnAssembly();
    }

    function isEnvStackSlot(slotId) {
      if (!slotId) return false;
      if (slotId === "partition") return isPartitionStackMode();
      return String(slotId).startsWith("sweep_");
    }

    function readSweepAlphaNums() {
      return [...document.querySelectorAll("#env-alpha-num-list input[type='number']")]
        .map((inp) => parseFloat(inp.value))
        .filter((n) => !Number.isNaN(n));
    }

    function collectAlphaNumRowsFromDOM() {
      const list = document.getElementById("env-alpha-num-list");
      if (!list) return [];
      return [...list.querySelectorAll(".env-alpha-num-row")].map((row) => {
        const inp = row.querySelector("input[type='number']");
        const raw = inp ? parseFloat(inp.value) : NaN;
        return Number.isNaN(raw) ? null : raw;
      });
    }

    function renderAlphaNumList(minRows) {
      const list = document.getElementById("env-alpha-num-list");
      if (!list) return;
      let vals = collectAlphaNumRowsFromDOM();
      if (!vals.length && minRows > 0) vals = Array.from({ length: minRows }, () => null);
      list.innerHTML = "";
      vals.forEach((v) => {
        const row = document.createElement("div");
        row.className = "env-alpha-num-row";
        const showDel = vals.length > 1;
        const valAttr = v != null && !Number.isNaN(v) ? ` value="${v}"` : "";
        row.innerHTML = `<label>α <input type="number" min="0.01" max="1" step="0.01"${valAttr} placeholder="0.03" /></label>${
          showDel ? '<button type="button" class="env-alpha-del" title="이 α 삭제">×</button>' : ""
        }`;
        list.appendChild(row);
      });
      list.querySelectorAll("input[type='number']").forEach((inp) => {
        inp.addEventListener("input", () => syncUI());
      });
      list.querySelectorAll(".env-alpha-del").forEach((btn) => {
        btn.addEventListener("click", () => {
          const row = btn.closest(".env-alpha-num-row");
          if (!row) return;
          row.remove();
          if (!list.querySelector(".env-alpha-num-row")) renderAlphaNumList(1);
          else renderAlphaNumList(0);
          syncUI();
        });
      });
    }

    function setupAlphaNumControls() {
      const addBtn = document.getElementById("btn-add-alpha");
      if (!addBtn || addBtn.dataset.bound === "1") return;
      addBtn.dataset.bound = "1";
      addBtn.addEventListener("click", () => {
        const list = document.getElementById("env-alpha-num-list");
        if (!list) return;
        const row = document.createElement("div");
        row.className = "env-alpha-num-row";
        row.innerHTML =
          '<label>α <input type="number" min="0.01" max="1" step="0.01" placeholder="0.03" /></label><button type="button" class="env-alpha-del" title="이 α 삭제">×</button>';
        list.appendChild(row);
        renderAlphaNumList(0);
        row.querySelector("input")?.focus();
        syncUI();
      });
    }

    function isSweepSlot(slotId) {
      return isEnvStackSlot(slotId);
    }

    function getPartitionBlocks() {
      return envColumns.map((c) => c.partition).filter(Boolean);
    }

    function partitionValues() {
      return getPartitionBlocks()
        .map((b) => resolvedEnvSlotValue(b))
        .filter(Boolean);
    }

    function hasDirichletInPartitions() {
      return partitionValues().includes("dirichlet");
    }

    const STACK_SOCKET_HTML =
      '<div class="env-stack-socket"><span class="env-stack-plus">+</span><span class="env-stack-hint">더 쌓기</span></div>';

    function ensureStackSocket(slotEl) {
      if (!slotEl || slotEl.dataset.multi !== "true") return;
      if (!slotEl.querySelector(".env-stack-socket")) {
        slotEl.insertAdjacentHTML("beforeend", STACK_SOCKET_HTML);
      }
    }

    function sweepSlotValues(slotId) {
      if (slotId === "partition") return partitionValues();
      const items = envSweepSlots[slotId] || [];
      return items.map((b) => resolvedEnvSlotValue(b)).filter(Boolean);
    }

    function envStackStore(slotId) {
      if (slotId === "partition") return envSweepSlots.partition;
      if (slotId === "diagnostics") return envSlots.diagnostics;
      return envSweepSlots[slotId];
    }

    function makeSweepEntry(slotId, value, title, sub) {
      return {
        id: "e" + ++idSeq,
        envSlot: slotId,
        kind: "env",
        target: "env",
        value: String(value),
        title: title || String(value),
        sub: sub || "",
        envCustom: false,
      };
    }

    function clearAllSweepSlots() {
      SWEEP_SLOT_IDS.forEach((id) => {
        envSweepSlots[id] = [];
      });
      const alphaList = document.getElementById("env-alpha-num-list");
      if (alphaList) alphaList.innerHTML = "";
    }

    function readSweepFromUI() {
      if (!usesColumnAssembly()) return null;
      const cols = envColumns.slice(0, getTargetColumnCount());
      const comparison_cases = cols.map((c, i) => comparisonCaseForColumn(c, i));
      const variants = [...new Set(comparison_cases.map((c) => c.variant_token).filter(Boolean))];
      const compare_assemblies = cols
        .map((c) => ({
          label: assemblyLabel(columnAssembly(c)),
          variant_token: columnVariantToken(c),
          ...columnAssembly(c),
        }))
        .filter((a) => a.graph_source || a.graph_mode || a.aggregation_target);
      const partitions = [...new Set(partitionValues())];
      const dirichlet_alphas = [
        ...new Set(
          cols
            .filter((c) => usesDirichletAlpha(resolvedEnvSlotValue(c.partition)))
            .map((c) => c.dirichletAlpha)
        ),
      ];
      return {
        partitions,
        dirichlet_alphas,
        variants,
        comparison_cases,
        compare_assemblies,
      };
    }

    function buildInlineSweepPieces(runner) {
      const host = document.getElementById("env-inline-sweep");
      if (!host) return;
      host.innerHTML = "";
      const rows = SWEEP_ROWS[runner] || [];
      rows.forEach((row) => {
        const el = document.createElement("div");
        el.className = "env-chain-piece env-sweep-axis";
        el.dataset.sweepAxis = row.slot;
        el.dataset.sweepRunner = row.runner;
        const compareToolbar =
          row.slot === "sweep_variants"
            ? `<div class="env-compare-toolbar">
                 <button type="button" class="env-compare-capture">+ 조립 캡처</button>
                 <button type="button" class="env-compare-fedavg">+ fedavg</button>
               </div>`
            : "";
        const comparePh =
          row.slot === "sweep_variants"
            ? "← fedavg · repo 토큰 · Graph-FL 부품"
            : "← 끌어다 놓기";
        el.innerHTML = `
          <div class="env-chain-piece-label">${row.label}</div>
          <div class="env-chain-piece-body">
            ${compareToolbar}
            <div class="env-chain-slot-row env-chain-slot-stack">
              <div class="env-h-slot env-stack-slot" data-env-slot="${row.slot}" data-multi="true" data-layout="stack">
                <span class="env-h-ph">${comparePh}</span>
                ${STACK_SOCKET_HTML}
              </div>
            </div>
          </div>`;
        host.appendChild(el);
        if (row.slot === "sweep_variants") {
          el.querySelector(".env-compare-capture")?.addEventListener("click", captureCompareFromLifeStack);
          el.querySelector(".env-compare-fedavg")?.addEventListener("click", addFedavgCompare);
        }
      });
      setupEnvSlotDrops();
      renderEnvSlots();
    }

    function configurePartitionSlotMode() {
      const stack = isPartitionStackMode();
      const partSlot = document.querySelector('.env-h-slot[data-env-slot="partition"]');
      const partRow = document.getElementById("partition-slot-row");
      if (partSlot) {
        partSlot.dataset.multi = stack ? "true" : "false";
        partSlot.classList.toggle("env-stack-slot", stack);
        ensureStackSocket(partSlot);
      }
      if (partRow) partRow.classList.toggle("env-chain-slot-stack", stack);
      updatePartitionAlphaUI();
    }

    function applySweepStackUI() {
      applyCasesZoneUI();
      const show = usesColumnAssembly();
      document.querySelectorAll(".sweep-palette-suite, .palette-vision-compare .scratch-block").forEach((el) => {
        el.classList.toggle("palette-block-hidden", !show);
      });
    }

    function envSlotUnlocked(slotId, colId) {
      if (existingConfigLocked()) {
        return !colId && slotId === "config_name";
      }
      if (!configInputReadyForAssembly()) {
        return !colId && slotId === "config_name";
      }
      if (colId) {
        if (!baseRowReady()) return false;
        if (slotId === "partition") return currentTrack() === "vision";
        if (slotId === "column_stem") {
          const col = getColumnById(colId);
          return columnStemOpen(col);
        }
        return false;
      }
      const track = currentTrack();
      if (slotId === "track") return true;
      if (slotId === "dataset") return !!envSlots.track;
      if (slotId === "config_name") return true;
      if (slotId === "diagnostics") return baseRowReady();
      return false;
    }

    function clearEnvDownstream(fromSlotId) {
      const idx = ENV_SLOT_ORDER.indexOf(fromSlotId);
      if (idx >= 0) {
        for (let i = idx + 1; i < ENV_SLOT_ORDER.length; i++) {
          if (fromSlotId === "dataset" && ENV_SLOT_ORDER[i] === "config_name") continue;
          envSlots[ENV_SLOT_ORDER[i]] = null;
        }
      }
      if (fromSlotId === "track" || fromSlotId === "dataset") {
        const nextColumnCount =
          fromSlotId === "dataset" && currentTrack() === "vision" ? getTargetColumnCount() : 1;
        resetEnvColumns(nextColumnCount);
        const chain = document.getElementById("env-cases-chain");
        if (chain) chain.innerHTML = "";
      }
    }

    function envValueAllowed(slotId, value, track, data) {
      if (!value || !String(value).trim()) return false;
      const activeTrack = track || envSlots.track?.value || null;
      if (data && (data.envCustom || ENV_CUSTOM_VALUES.has(data.value))) {
        if (slotId === "partition" && activeTrack === "cora") return false;
        if (slotId === "dataset" && !activeTrack) return false;
        return true;
      }
      if (slotId === "track") return value === "vision" || value === "cora";
      if (slotId === "dataset") {
        if (!activeTrack || !TRACK_RULES[activeTrack]) return false;
        return TRACK_RULES[activeTrack].datasets.has(value);
      }
      if (slotId === "partition" || slotId === "column_stem") {
        return activeTrack === "vision" || slotId !== "partition";
      }
      return true;
    }

    function normalizeEnvDrop(data) {
      const out = { ...data };
      const slotId =
        out.value === "custom_variant"
          ? "sweep_variants"
          : out.envSlot ||
        (out.value === "custom_config"
          ? "config_name"
          : out.value === "custom_partition"
            ? "partition"
            : "dataset");

      if (ENV_CUSTOM_VALUES.has(out.value)) {
        out.envCustom = true;
        const name = String(out.customName || "").trim();
        out.customName = name;
        out.value = name || envCustomPlaceholder(slotId);
        out.title = name || out.title || slotId;
        if (!out.sub) out.sub = slotId === "partition" ? "--partition" : slotId === "config_name" ? "--config" : "--dataset";
        return out;
      }

      if (out.envCustom) {
        const name = String(out.customName || "").trim();
        out.customName = name;
        out.value = name || envCustomPlaceholder(slotId);
        out.title = name || out.title || slotId;
        return out;
      }

      out.envCustom = false;
      delete out.customName;
      return out;
    }

    function usesDirichletAlpha(partition) {
      return partition === "dirichlet";
    }

    function updatePartitionAlphaUI() {
      const numWrap = document.getElementById("env-alpha-wrap");
      const stackWrap = document.getElementById("env-alpha-stack-wrap");
      const track = currentTrack();
      const stack = isPartitionStackMode();
      const showStack = track === "vision" && stack && hasDirichletInPartitions();
      const showNum = track === "vision" && !stack && hasDirichletInPartitions();
      if (numWrap) numWrap.classList.toggle("palette-block-hidden", !showNum);
      if (stackWrap) {
        stackWrap.classList.toggle("palette-block-hidden", !showStack);
        if (showStack && !document.querySelector("#env-alpha-num-list .env-alpha-num-row")) {
          renderAlphaNumList(1);
        }
      }
    }

    function flashReject(slotEl) {
      if (!slotEl) return;
      slotEl.classList.add("reject-flash");
      setTimeout(() => slotEl.classList.remove("reject-flash"), 500);
    }

    function pruneInvalidEnvSlots(track) {
      if (!track) {
        clearEnvDownstream("track");
        renderEnvSlots();
        return;
      }
      if (envSlots.dataset && !envValueAllowed("dataset", envSlots.dataset.value, track, envSlots.dataset)) {
        envSlots.dataset = null;
        clearEnvDownstream("dataset");
      }
      renderEnvSlots();
    }

    function applyRunnerContext() {
      applySweepStackUI();
    }

    function applyStepHighlights() {
      const step = currentAssemblyStep();
      const existingLocked = existingConfigLocked();
      const envPanel = document.getElementById("env-panel");
      if (envPanel) {
        envPanel.dataset.stepPhase = step.phase || "";
        envPanel.dataset.existingLocked = existingLocked ? "true" : "false";
      }
      const banner = document.getElementById("env-step-banner");
      if (banner) {
        banner.textContent = stepBannerText(step);
        applyNextClass(banner, stepNextKind(step));
      }

      document.querySelectorAll("#env-base-row .env-chain-piece").forEach((piece) => {
        const id = piece.dataset.piece;
        const active =
          (step.phase === "track" && id === "track") ||
          (step.phase === "config_name" && id === "config_name") ||
          (step.phase === "dataset" && (id === "dataset" || id === "scale")) ||
          (step.phase === "track" && id === "cases");
        const wait =
          (step.phase === "track" && id === "config_name") ||
          (step.phase === "dataset" && (id === "config_name" || id === "track")) ||
          ((step.phase === "partition" ||
            step.phase === "column_stem" ||
            step.phase === "graph") &&
            id !== "scale" &&
            id !== "cases" &&
            id !== "diagnostics");
        piece.classList.toggle("step-active", active);
        piece.classList.toggle("step-wait", wait && !active);
        const slot = piece.querySelector(".env-h-slot");
        if (slot) {
          const unlocked = envSlotUnlocked(id);
          slot.classList.toggle("is-locked", !unlocked);
          slot.classList.toggle("step-active", active);
        }
        const lockedByStep = !["scale", "cases", "diagnostics"].includes(id) && !envSlotUnlocked(id);
        piece.classList.toggle("is-locked", (existingLocked && id !== "config_name") || lockedByStep);
        if (id === "cases" && currentTrack() === "cora") {
          piece.classList.add("is-locked");
        }
      });

      document.querySelectorAll(".env-case-stack").forEach((colEl) => {
        const colId = colEl.dataset.columnId;
        const partActive = step.phase === "partition" && step.colId === colId;
        const stemActive = step.phase === "column_stem" && step.colId === colId;
        const graphActive = step.phase === "graph" && step.colId === colId;
        const col = getColumnById(colId);
        colEl.querySelectorAll(".env-h-slot, .column-life-stack").forEach((el) => {
          el.classList.remove("step-active", "is-locked");
          clearNextClasses(el);
          const slotKind = el.dataset.envSlot || (el.classList.contains("column-life-stack") ? "graph" : "");
          const unlocked =
            slotKind === "partition"
              ? envSlotUnlocked("partition", colId)
              : slotKind === "column_stem"
                ? envSlotUnlocked("column_stem", colId)
                : slotKind === "graph" || el.classList.contains("column-life-stack")
                  ? columnGraphUnlocked(colId) || columnStemOpen(col)
                  : false;
          el.classList.toggle("is-locked", !unlocked);
          if (partActive && el.dataset.envSlot === "partition") {
            el.classList.add("step-active");
            applyNextClass(el, "partition");
          }
          if (stemActive && el.dataset.envSlot === "column_stem") {
            el.classList.add("step-active");
            applyNextClass(el, "baseline");
          }
          if (graphActive && col?.caseMode === "assembly" && el.classList.contains("column-life-stack")) {
            el.classList.add("step-active");
            applyNextClass(el, step.nextStage?.kind || "diagnostic");
          }
        });
      });
    }

    function applyPaletteHighlight() {
      const step = currentAssemblyStep();
      const focusCol = step.colId ? getColumnById(step.colId) : null;
      document.querySelectorAll(".palette .scratch-block[draggable]").forEach((block) => {
        const p = payload(block);
        let can = blockCanInstall(p, focusCol);
        if (step.phase === "graph" && p.kind === "diagnostic" && p.target === "env") can = false;
        clearNextClasses(block);
        if (can) {
          const lifeKind = lifecycleKindFromData(p);
          const highlightKind =
            step.phase === "column_stem" && isBaselineBlock(p)
              ? "baseline"
              : step.phase === "graph" && lifeKind
                ? lifeKind
                : stepNextKind(step);
          applyNextClass(block, highlightKind);
        }
        block.classList.toggle("palette-dimmed", !can);
        block.classList.toggle("palette-highlight", can);
        block.classList.toggle("palette-clickable", can);
      });
    }

    function applyPaletteFolds() {
      const step = currentAssemblyStep();
      const track = currentTrack();
      if (step.phase === "config_name") {
        expandFoldSection(document.getElementById("fold-palette-env"));
        expandFoldSection(document.getElementById("fold-palette-config"));
      }
      if (step.phase === "track") {
        expandFoldSection(document.getElementById("fold-palette-env"));
        expandFoldSection(document.getElementById("fold-palette-track"));
      }
      if (step.phase === "dataset") {
        expandFoldSection(document.getElementById("fold-palette-env"));
        expandFoldSection(document.getElementById("fold-palette-dataset"));
      }
      if (step.phase === "partition" && track === "vision") {
        expandFoldSection(document.getElementById("fold-palette-column"));
        expandFoldSection(document.getElementById("fold-palette-partition"));
      }
      if (step.phase === "column_stem") {
        expandFoldSection(document.getElementById("fold-palette-baseline"));
        expandFoldSection(document.getElementById("fold-palette-life"));
      }
      if (step.phase === "graph") {
        expandFoldSection(document.getElementById("fold-palette-life"));
        const nextKind = step.nextStage?.kind;
        const lifeSubs = Array.from(document.querySelectorAll("#fold-palette-life .cat-sub"));
        const idx = { client: 0, relation: 1, topology: 2, aggregation: 3, correction: 4, diagnostic: 5 }[nextKind];
        if (idx != null && lifeSubs[idx]) expandFoldSection(lifeSubs[idx]);
      }
    }

    function applyEnvChainGating() {
      applyRunnerContext();
      applyCasesZoneUI();
      applyStepHighlights();
      applyPaletteFolds();
      applyPaletteHighlight();
      updateAllColumnStemHelpers();
    }

    function applyTrackContext() {
      const track = currentTrack();
      const existingLocked = existingConfigLocked();
      document.querySelector(".palette")?.classList.toggle("existing-config-locked", existingLocked);
      document.querySelectorAll(".palette [data-tracks]").forEach((el) => {
        const allowed = (el.dataset.tracks || "").split(",").map((s) => s.trim());
        let show;
        if (!track) {
          show = !el.dataset.tracks || el.dataset.envSlot === "track" || el.dataset.envSlot === "config_name";
        } else {
          show = allowed.includes(track);
        }
        el.classList.toggle("palette-block-hidden", !show);
      });
      pruneInvalidEnvSlots(track);
      updatePartitionAlphaUI();
      applyEnvChainGating();
    }
    const envSlots = {
      track: null,
      dataset: null,
      diagnostics: [],
    };
    const envSweepSlots = {
      partition: [],
      sweep_variants: [],
      sweep_client_counts: [],
      sweep_knn_ks: [],
      sweep_templates: [],
    };
    let envColumnCount = 1;
    const envColumns = [];
    let idSeq = 0;
    let configUseMode = "existing";
    let configModeTouched = false;

    const stackLife = null;
    const workspace = document.getElementById("workspace");
    const btnRun = document.getElementById("btn-run");
    const numAlpha = document.getElementById("num-alpha");
    const numClients = document.getElementById("num-clients");
    const numRounds = document.getElementById("num-rounds");

    const DIAGNOSTIC_META = {
      preflight: { title: "preflight", sub: "diagnostic_suite_preflight" },
      suite_rows: { title: "suite_rows", sub: "vision_suite_*" },
      loo: { title: "LOO", sub: "--loo-enabled true" },
      evidence: { title: "evidence", sub: "result_evidence_bundle.py" },
      counterfactual: { title: "counterfactual", sub: "control-family compare" },
    };

    function makeDiagnosticEntry(value) {
      const meta = DIAGNOSTIC_META[value] || { title: value, sub: "diagnostic" };
      return {
        id: `diag-${value}`,
        target: "env",
        envSlot: "diagnostics",
        kind: "diagnostic",
        value,
        title: meta.title,
        sub: meta.sub,
      };
    }

    function suiteRowsAutoEnabled() {
      if (currentTrack() !== "vision") return false;
      return getTargetColumnCount() > 1;
    }

    function selectedDiagnosticValuesFromControls() {
      if (existingConfigLocked()) return [];
      const values = ["preflight"];
      if (suiteRowsAutoEnabled()) values.push("suite_rows");
      document.querySelectorAll("[data-diagnostic-option]").forEach((input) => {
        if (input.checked) values.push(input.dataset.diagnosticOption);
      });
      return Array.from(new Set(values.filter(Boolean)));
    }

    function syncDiagnosticOptionsFromControls() {
      const values = selectedDiagnosticValuesFromControls();
      const old = new Map((envSlots.diagnostics || []).map((entry) => [entry.value, entry]));
      envSlots.diagnostics = values.map((value) => old.get(value) || makeDiagnosticEntry(value));
    }

    function updateDiagnosticOptionsUI() {
      const locked = existingConfigLocked();
      const wrap = document.getElementById("diagnostic-options");
      if (wrap) wrap.classList.toggle("is-locked", locked);
      document.querySelectorAll("[data-diagnostic-option]").forEach((input) => {
        input.disabled = locked;
      });
      const suite = document.getElementById("diag-auto-suite");
      if (suite) suite.classList.toggle("is-active", suiteRowsAutoEnabled() && !locked);
      const preflight = document.getElementById("diag-auto-preflight");
      if (preflight) preflight.classList.toggle("is-active", !locked);
    }

    function setupDiagnosticOptions() {
      document.querySelectorAll("[data-diagnostic-option]").forEach((input) => {
        if (input.dataset.diagnosticBound === "1") return;
        input.dataset.diagnosticBound = "1";
        input.addEventListener("change", () => {
          syncDiagnosticOptionsFromControls();
          syncUI();
        });
      });
    }

    function getEnvNums() {
      const stackAlphas = readSweepAlphaNums();
      const alpha =
        stackAlphas.length > 0
          ? stackAlphas[0]
          : parseFloat(numAlpha?.value || "0.1");
      return {
        dirichlet_alpha: Number.isNaN(alpha) ? 0.1 : alpha,
        num_clients: parseInt(numClients?.value || "5", 10) || 5,
        rounds: parseInt(numRounds?.value || "3", 10) || 3,
        knn_k: 2,
        graph_filter_strength: 1.0,
        train_subset_size: 1000,
        test_subset_size: 300,
        out_dir: "experiments_current/generated_graphfl",
      };
    }

    function commonDiagnosticsEnabled() {
      return (envSlots.diagnostics || []).length > 0;
    }

    function buildCommonEnvArgs() {
      const nums = getEnvNums();
      const ds = resolvedEnvSlotValue(envSlots.dataset);
      const firstCol = envColumns[0];
      const part = firstCol ? resolvedEnvSlotValue(firstCol.partition) : null;
      const args = {
        dataset: ds,
        model: currentTrack() === "cora" ? "gcn" : "mlp",
        num_clients: nums.num_clients,
        rounds: nums.rounds,
        local_epochs: 1,
        batch_size: 64,
        train_subset_size: nums.train_subset_size,
        test_subset_size: nums.test_subset_size,
        seeds: [42],
        knn_k: nums.knn_k,
        graph_filter_strength: nums.graph_filter_strength,
        diagnostics_enable: commonDiagnosticsEnabled() || envColumns.some((c) => getColumnLife(c, "diagnostic")),
        loo_enabled: hasDiagnostic("loo") || envColumns.some((c) => getColumnLife(c, "diagnostic")?.value === "loo"),
        out_dir: nums.out_dir,
      };
      if (part) args.partition = part;
      if (usesDirichletAlpha(args.partition) && firstCol) {
        args.dirichlet_alpha = firstCol.dirichletAlpha ?? nums.dirichlet_alpha;
      }
      return args;
    }

    function buildSingleRunArgs(col, envBase) {
      const args = {
        engine: "app",
        ...envBase,
        seed: Array.isArray(envBase.seeds) ? envBase.seeds[0] || 42 : 42,
        run_tag: `generated_graphfl_col_${col.id}`,
        out_dir: `${envBase.out_dir}_single`,
      };
      delete args.seeds;
      const part = resolvedEnvSlotValue(col.partition);
      if (part) {
        args.partition = part;
        if (usesDirichletAlpha(part)) {
          args.dirichlet_alpha = col.dirichletAlpha ?? args.dirichlet_alpha;
        } else {
          delete args.dirichlet_alpha;
        }
      }
      if (col.caseMode === "baseline" && col.preset) {
        args.method = col.preset.method || col.preset.value || "fedavg";
        delete args.graph_source;
        delete args.graph_mode;
        delete args.aggregation_target;
        delete args.correction_family;
        delete args.control_graph_mode;
        delete args.knn_k;
        delete args.graph_filter_strength;
        return args;
      }
      const p = columnLifecycleParts(col);
      args.method = "ours";
      if (p.graph_source) args.graph_source = p.graph_source;
      if (p.graph_mode) args.graph_mode = p.graph_mode;
      if (p.aggregation_target) args.aggregation_target = p.aggregation_target;
      if (p.correction_family) args.correction_family = p.correction_family;
      if (p.control_graph_mode) args.control_graph_mode = p.control_graph_mode;
      if (
        p.graph_source === "update" &&
        p.relation === "cosine" &&
        p.graph_mode === "knn" &&
        p.aggregation_target === "graph_filtered_update" &&
        !p.correction_family
      ) {
        args.graph_method = "default_similarity_knn";
      }
      return args;
    }

    function buildSingleRunConfig() {
      const n = getTargetColumnCount();
      const col = envColumns[0];
      if (!col || !columnCaseGraphReady(col)) return null;
      const args = buildSingleRunArgs(col, buildCommonEnvArgs());
      return {
        description: "Generated by Graph-FL Assembly UI",
        args,
      };
    }

    function safeRepoToken(value, fallback) {
      const raw = String(value || fallback || "graphfl_demo").trim().replace(/\.json$/i, "");
      const normalized = raw.replace(/[^\w.-]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 80);
      return normalized || fallback || "graphfl_demo";
    }

    function buildCoraGraphAblationConfig(cfg) {
      const nums = getEnvNums();
      const suiteTag = safeRepoToken(cfg?.config_name, "graphfl_cora_demo");
      const diagnosticValues = (envSlots.diagnostics || []).map((b) => b.value).filter(Boolean);
      return {
        ...(diagnosticValues.length ? { demo_meta: { diagnostics: diagnosticValues } } : {}),
        description: "Generated by Graph-FL Assembly UI · Cora graph ablation",
        args: {
          num_clients: nums.num_clients,
          rounds: nums.rounds,
          local_epochs: 1,
          seeds: [42],
          partition: "iid",
          variants: ["fedavg", "ours_knn", "ours_random"],
          knn_k: nums.knn_k,
          warmup_rounds: 0,
          diagnostic_only: true,
          out_dir: `experiments_current/${suiteTag}`,
          suite_tag: suiteTag,
        },
      };
    }

    function buildSuiteConfig() {
      const n = getTargetColumnCount();
      const cols = envColumns.slice(0, n);
      if (cols.some((c) => !columnCaseGraphReady(c))) return null;
      if (cols.length < 2) return null;
      const partitions = [...new Set(cols.map((c) => resolvedEnvSlotValue(c.partition)).filter(Boolean))];
      if (partitions.length !== 1) return null;
      if (usesDirichletAlpha(partitions[0])) {
        const alphas = [...new Set(cols.map((c) => String(c.dirichletAlpha ?? getEnvNums().dirichlet_alpha)))];
        if (alphas.length !== 1) return null;
      }
      const colTokens = cols.map((c) => buildVariantToken(c));
      if (colTokens.some((token) => !token)) return null;
      const tokens = [...new Set(colTokens)];
      if (!tokens.length || tokens.length !== colTokens.length) return null;
      const env = buildCommonEnvArgs();
      const comparisonCases = cols.map((c, i) => comparisonCaseForColumn(c, i));
      return {
        description: "Generated by Graph-FL Assembly UI",
        comparison_cases: comparisonCases,
        args: {
          ...env,
          partition: partitions[0],
          ...(usesDirichletAlpha(partitions[0]) ? { dirichlet_alpha: cols[0].dirichletAlpha ?? env.dirichlet_alpha } : {}),
          variants: tokens,
          out_dir: `${env.out_dir}_suite`,
        },
      };
    }

    function describeVisionBatchFallback() {
      const n = getTargetColumnCount();
      const cols = envColumns.slice(0, n);
      if (cols.some((c) => !columnCaseGraphReady(c))) {
        return "모든 비교 열이 완성되면 실행 경로가 생성됩니다.";
      }
      const partitions = [...new Set(cols.map((c) => resolvedEnvSlotValue(c.partition)).filter(Boolean))];
      if (partitions.length > 1) {
        return "partition이 달라 batch로 열마다 별도 job을 제출합니다.";
      }
      if (partitions.length === 1 && usesDirichletAlpha(partitions[0])) {
        const alphas = [...new Set(cols.map((c) => String(c.dirichletAlpha ?? getEnvNums().dirichlet_alpha)))];
        if (alphas.length > 1) {
          return "Dirichlet α가 달라 batch로 열마다 별도 job을 제출합니다.";
        }
      }
      const tokens = cols.map((c) => buildVariantToken(c));
      if (tokens.some((token) => !token)) {
        return "suite token이 없는 Graph-FL 조합이라 batch로 별도 job을 제출합니다.";
      }
      if (new Set(tokens).size !== tokens.length) {
        return "같은 suite token이 반복되어 batch로 열마다 별도 job을 제출합니다.";
      }
      return "이 조합은 batch로 열마다 별도 job을 제출합니다.";
    }

    function columnConfigPath(configPath, index) {
      const path = String(configPath || "configs/vision/smoke/generated.json");
      const dot = path.toLowerCase().endsWith(".json") ? path.length - 5 : path.length;
      return `${path.slice(0, dot)}_col${index + 1}.json`;
    }

    function buildPerColumnSingleConfigs() {
      const n = getTargetColumnCount();
      const cols = envColumns.slice(0, n);
      if (cols.some((c) => !columnCaseGraphReady(c))) return [];
      return cols.map((col, i) => ({
        description: `Generated by Graph-FL Assembly UI · column ${i + 1}`,
        comparison_case: comparisonCaseForColumn(col, i),
        args: buildSingleRunArgs(col, buildCommonEnvArgs()),
      }));
    }

    function buildVisionBatchConfig(configPath) {
      const configs = buildPerColumnSingleConfigs();
      if (!configs.length) return null;
      return {
        description:
          "Generated by Graph-FL Assembly UI · Mock API batch. Each job maps to run_vision_experiment.py because this assembly cannot be represented as one run_vision_suite.py config.",
        jobs: configs.map((config, i) => ({
          job_index: i + 1,
          entrypoint: "run_vision_experiment.py",
          config_path: columnConfigPath(configPath, i),
          command: `python run_vision_experiment.py --config ${columnConfigPath(configPath, i)}`,
          comparison_case: config.comparison_case,
          config,
        })),
      };
    }

    function buildBatchRunCommand(configPath) {
      const configs = buildPerColumnSingleConfigs();
      if (!configs.length) return "# 열 조립을 완료하면 실행 명령이 표시됩니다";
      return configs
        .map((_, i) => `python run_vision_experiment.py --config ${columnConfigPath(configPath, i)}`)
        .join("\n");
    }

    function buildRunCommand(configType, configPath, track) {
      if (track === "cora" || configType === "cora-graph-ablation" || configType === "existing-cora-graph-ablation") {
        return `python run_graph_ablation.py --config ${configPath}`;
      }
      if (configType === "existing-vision-single") {
        return `python run_vision_experiment.py --config ${configPath}`;
      }
      if (configType === "existing-vision-client-count-sweep") {
        return `python run_vision_client_count_sweep.py --config ${configPath}`;
      }
      if (configType === "existing-vision-stress-grid") {
        return `python run_vision_stress_grid.py --config ${configPath}`;
      }
      if (configType === "suite" || configType === "existing-vision-suite") {
        return `python run_vision_suite.py --config ${configPath}`;
      }
      if (configType === "batch") {
        return buildBatchRunCommand(configPath);
      }
      return `python run_vision_experiment.py --config ${configPath}`;
    }

    function downloadJson(filename, object) {
      const blob = new Blob([JSON.stringify(object, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    }

    function renderAssemblySummary() {
      const env = buildCommonEnvArgs();
      const cfg = buildEnvConfig();
      const selection = resolveConfigSelection(cfg);
      const n = getTargetColumnCount();
      if (selection.existing) {
        const lines = [
          "기존 config JSON 직접 실행:",
          `  path=${selection.path}`,
          `  entrypoint=${entrypointForConfigType(selection.configType || inferExistingConfigType(selection.path, cfg.track || selection.track), cfg.track || selection.track)}`,
          "",
          "화면 조립값은 이 파일에 덮어쓰지 않습니다.",
          "새 JSON을 만들려면 configs/...에 없는 새 이름을 입력하고 환경·비교 열을 조립하세요.",
        ];
        const el = document.getElementById("assembly-summary-out");
        if (el) el.textContent = lines.join("\n");
        return;
      }
      if (currentTrack() === "cora") {
        const lines = [
          "공통 실행환경:",
          `  dataset=${env.dataset} model=${env.model} N=${env.num_clients} R=${env.rounds}`,
          "",
          "Cora graph ablation:",
          "  variants=fedavg, ours_knn, ours_random",
          `  partition=iid knn_k=${env.knn_k} diagnostic_only=true`,
          "  결과: suite summary, result rows, graph ablation artifacts",
        ];
        const el = document.getElementById("assembly-summary-out");
        if (el) el.textContent = lines.join("\n");
        return;
      }
      const lines = [
        "공통 실행환경:",
        `  dataset=${env.dataset} model=${env.model} N=${env.num_clients} R=${env.rounds}`,
        `  partition=${env.partition}${env.dirichlet_alpha != null ? ` alpha=${env.dirichlet_alpha}` : ""}`,
        `  diagnostics_enable=${env.diagnostics_enable} loo_enabled=${env.loo_enabled}`,
        "",
        "비교 열:",
      ];
      envColumns.slice(0, n).forEach((col, i) => {
        const token = buildVariantToken(col);
        const status = col.caseMode === "baseline" && col.preset
          ? "baseline-complete"
          : columnAssemblyRunComplete(col)
            ? getColumnLife(col, BLOCK_KIND.CORRECTION)
              ? "graphfl-complete + correction"
              : "graphfl-complete"
            : "incomplete";
        if (col.caseMode === "baseline") {
          lines.push(`  열 ${i + 1}: baseline · method=${col.preset?.value || "?"} · ${status}`);
        } else if (col.caseMode === "assembly") {
          lines.push(`  열 ${i + 1}: Graph-FL · ${status} · ${columnLifecycleTrail(col)}`);
        } else {
          lines.push(`  열 ${i + 1}: (미완)`);
        }
        if (token) lines.push(`    suite variant token: ${token}`);
        else if (col.caseMode === "assembly") lines.push("    suite variant: (표현 불가 — single-run만)");
      });
      const el = document.getElementById("assembly-summary-out");
      if (el) el.textContent = lines.join("\n");
    }

    function renderPreview() {
      const n = getTargetColumnCount();
      const warnEl = document.getElementById("config-warn");
      const cmdEl = document.getElementById("run-command-out");
      const jsonEl = document.getElementById("config-json-out");
      const perRunEl = document.getElementById("per-run-configs-out");
      const cfgForPath = buildEnvConfig();
      const configSelection = resolveConfigSelection(cfgForPath);
      const configPathUsable = !!configSelection.path && (
        configSelection.existing ? !!(cfgForPath.track || configSelection.track) : !!cfgForPath.track
      );
      const selectedPath = configPathUsable ? configSelection.path : "";
      const previewTrack = cfgForPath.track || configSelection.track;
      const singlePath = selectedPath || "";
      const suitePath = selectedPath || "";

      renderAssemblySummary();

      let configType = configSelection.existing
        ? configSelection.configType || inferExistingConfigType(selectedPath, previewTrack)
        : cfgForPath.track === "cora"
          ? "cora-graph-ablation"
          : "single";
      let doc = selectedPath
        ? configSelection.existing
          ? buildExistingConfigReference(configSelection, cfgForPath)
          : cfgForPath.track === "cora"
            ? buildCoraGraphAblationConfig(cfgForPath)
            : buildSingleRunConfig()
        : null;
      let configPath = singlePath;
      let warn = selectedPath ? "" : "config JSON 이름 블록을 조립해야 실행 경로가 생성됩니다.";
      if (configSelection.existing) {
        warn =
          "기존 repo config JSON을 그대로 사용합니다. 현재 화면 조립값은 이 파일에 병합하지 않습니다.";
      } else if (selectedPath) {
        warn =
          "새 JSON 저장 모드 · 현재 조립을 config_path에 저장한 뒤 --config로 실행합니다.";
      }

      if (!configSelection.existing && cfgForPath.track !== "cora" && n >= 2) {
        const suiteDoc = selectedPath ? buildSuiteConfig() : null;
        if (suiteDoc) {
          configType = "suite";
          doc = suiteDoc;
          configPath = suitePath;
        } else {
          configType = "batch";
          const fallback = describeVisionBatchFallback();
          warn = warn ? `${warn}\n${fallback}` : fallback;
          doc = selectedPath ? buildVisionBatchConfig(selectedPath) : null;
        }
      }
      if (doc && !configSelection.existing) {
        doc = withGeneratedConfigMeta(doc, configPath, configType);
      }

      const cmd = doc
        ? buildRunCommand(configType, configPath, previewTrack)
        : "# 열 조립을 완료하면 실행 명령이 표시됩니다";

      if (cmdEl) cmdEl.textContent = cmd;
      document.getElementById("cli-out").textContent = cmd;

      if (jsonEl) {
        if (doc) {
          jsonEl.textContent = JSON.stringify(doc, null, 2);
        } else if (configType === "per-column" && selectedPath) {
          const list = selectedPath ? buildPerColumnSingleConfigs() : [];
          jsonEl.textContent = JSON.stringify(list, null, 2);
        } else if (!selectedPath) {
          jsonEl.textContent = "# config JSON 이름 블록을 조립하면 JSON과 실행 명령이 생성됩니다.";
        } else {
          jsonEl.textContent = "# 아직 생성할 수 없습니다 (환경·열 조립 필요)";
        }
      }

      const perRunTitle = document.getElementById("per-run-title");
      const showPerRun = !!doc && (configType === "batch" || configType === "per-column") && !!selectedPath;
      if (perRunTitle) perRunTitle.style.display = showPerRun ? "block" : "none";
      if (perRunEl) {
        perRunEl.style.display = showPerRun ? "block" : "none";
        if (showPerRun) {
          perRunEl.textContent = JSON.stringify(
            buildPerColumnSingleConfigs().map((config, i) => ({
              config_path: columnConfigPath(selectedPath, i),
              command: `python run_vision_experiment.py --config ${columnConfigPath(selectedPath, i)}`,
              config,
            })),
            null,
            2
          );
        }
      }

      if (warnEl) {
        warnEl.textContent = warn;
        warnEl.style.display = warn ? "block" : "none";
      }

      const pathEl = document.getElementById("config-path");
      if (pathEl) pathEl.textContent = selectedPath ? configPath : "";

      window.__graphflPreview = { configType, doc, singlePath, suitePath, configPath, cmd };
      const currentDownload = document.getElementById("btn-dl-single");
      if (currentDownload) {
        currentDownload.disabled = !doc || configSelection.existing;
        currentDownload.title =
          configSelection.existing
            ? "기존 repo JSON은 그대로 --config로 사용합니다. 화면에서 파일 본문을 다시 다운로드하지 않습니다."
            : doc
              ? ""
              : "생성된 JSON이 있을 때 다운로드할 수 있습니다.";
      }
      const perRunDownload = document.getElementById("btn-dl-suite");
      if (perRunDownload) {
        const perRunAvailable =
          !!doc && !configSelection.existing && cfgForPath.track === "vision" && buildPerColumnSingleConfigs().length > 0;
        perRunDownload.disabled = !perRunAvailable;
        perRunDownload.title = perRunAvailable ? "" : "Vision per-run config 목록이 있을 때 다운로드할 수 있습니다.";
      }

      return renderRunArtifacts();
    }

    const EXTEND_PLACEHOLDERS = {
      name_graph_source: "my_source",
      name_relation: "my_relation",
      name_graph_mode: "my_mode",
      name_aggregation: "my_agg",
      name_correction: "my_correction",
      name_method: "my_method",
    };

    const PY_IDENTIFIER_RE = /^[A-Za-z_][A-Za-z0-9_]{0,63}$/;
    const REPO_TOKEN_RE = /^[A-Za-z0-9_.-]{1,80}$/;
    const CONFIG_PATH_RE = /^[A-Za-z0-9_./\\-]{1,160}$/;

    function configNameHasPath(rawName) {
      return /[\\/]/.test(String(rawName || "").trim());
    }

    function inferTrackFromConfigPath(path) {
      const normalized = String(path || "").trim().replace(/\\/g, "/").toLowerCase();
      if (normalized.startsWith("configs/cora/")) return "cora";
      if (normalized.startsWith("configs/vision/") || normalized.startsWith("configs/general/")) return "vision";
      return null;
    }

    function normalizeConfigPath(path) {
      return String(path || "").trim().replace(/\\/g, "/").toLowerCase();
    }

    const EXISTING_VISION_SINGLE_CONFIGS = new Set([
      "configs/vision/smoke/default_similarity_knn.json",
    ]);
    const EXISTING_VISION_STRESS_GRID_CONFIGS = new Set([
      "configs/vision/smoke/semantic_ema_weight.json",
    ]);

    function inferExistingConfigType(path, track) {
      const normalized = normalizeConfigPath(path);
      const selectedTrack = track || inferTrackFromConfigPath(normalized);
      if (selectedTrack === "cora" || normalized.startsWith("configs/cora/")) {
        return "existing-cora-graph-ablation";
      }
      if (EXISTING_VISION_SINGLE_CONFIGS.has(normalized)) {
        return "existing-vision-single";
      }
      if (
        normalized.includes("/sweeps/client_count/") ||
        normalized.includes("/client_count_sweep/")
      ) {
        return "existing-vision-client-count-sweep";
      }
      if (
        EXISTING_VISION_STRESS_GRID_CONFIGS.has(normalized) ||
        normalized.includes("/stress/")
      ) {
        return "existing-vision-stress-grid";
      }
      return "existing-vision-suite";
    }

    function entrypointForConfigType(configType, track) {
      if (track === "cora" || configType === "cora-graph-ablation" || configType === "existing-cora-graph-ablation") return "run_graph_ablation.py";
      if (configType === "existing-vision-single") return "run_vision_experiment.py";
      if (configType === "existing-vision-client-count-sweep") return "run_vision_client_count_sweep.py";
      if (configType === "existing-vision-stress-grid") return "run_vision_stress_grid.py";
      if (configType === "batch") return "run_vision_experiment.py";
      if (configType === "suite" || configType === "existing-vision-suite") return "run_vision_suite.py";
      return track === "cora" ? "run_graph_ablation.py" : "run_vision_experiment.py";
    }

    function configPathForSelection(rawName, track) {
      const selected = String(rawName || "").trim();
      if (!selected) return "";
      const normalized = selected.replace(/\\/g, "/");
      const base = normalized.endsWith(".json") ? normalized : `${normalized}.json`;
      if (base.includes("/")) return base;
      return track === "cora"
        ? `configs/cora/ablations/graph/${base}`
        : `configs/vision/smoke/${base}`;
    }

    function validateConfigPathName(rawName, forcedTrack = null) {
      const raw = String(rawName || "").trim();
      if (!raw) return { ok: true, message: "" };
      const normalized = raw.replace(/\\/g, "/");
      const hasPath = normalized.includes("/");
      const track = forcedTrack || currentTrack();
      if (
        !CONFIG_PATH_RE.test(raw) ||
        normalized.includes("//") ||
        normalized.startsWith("/") ||
        normalized.split("/").includes("..") ||
        /^[A-Za-z]:/.test(raw)
      ) {
        return {
          ok: false,
          message: "config JSON은 repo 내부 상대 경로만 가능하며 ../, 절대 경로, 따옴표는 사용할 수 없습니다.",
        };
      }
      if (
        hasPath &&
        track === "vision" &&
        !normalized.startsWith("configs/vision/") &&
        !normalized.startsWith("configs/general/")
      ) {
        return { ok: false, message: "Vision config 경로는 configs/vision/ 또는 legacy configs/general/ 아래여야 합니다." };
      }
      if (hasPath && track === "cora" && !normalized.startsWith("configs/cora/")) {
        return { ok: false, message: "Cora config 경로는 configs/cora/ 아래여야 합니다." };
      }
      if (hasPath && !track && !normalized.startsWith("configs/")) {
        return { ok: false, message: "config 경로를 직접 쓰려면 configs/ 아래 상대 경로를 사용하세요." };
      }
      return { ok: true, message: "" };
    }

    function validateCustomName(value, slotId, name) {
      const raw = String(name || "").trim();
      if (!raw) return { ok: true, message: "" };
      if (value && String(value).startsWith("name_")) {
        return PY_IDENTIFIER_RE.test(raw)
          ? { ok: true, message: "" }
          : { ok: false, message: "Python 식별자는 영문/숫자/_만 가능하고 숫자로 시작할 수 없습니다." };
      }
      if (slotId === "sweep_variants") {
        return REPO_TOKEN_RE.test(raw)
          ? { ok: true, message: "" }
          : { ok: false, message: "repo variant token은 영문/숫자/._-만 권장합니다." };
      }
      if (slotId === "config_name") {
        return validateConfigPathName(raw);
      }
      if (slotId === "dataset" || slotId === "partition") {
        return REPO_TOKEN_RE.test(raw)
          ? { ok: true, message: "" }
          : { ok: false, message: "데이터셋/파티션 이름은 영문/숫자/._-만 권장합니다." };
      }
      return { ok: true, message: "" };
    }

    function looksLikeRepoConfigPath(rawName) {
      const raw = String(rawName || "").trim().replace(/\\/g, "/");
      return raw.startsWith("configs/") && !raw.endsWith("/");
    }

    function resolveConfigSelection(cfg) {
      const selected = String(cfg?.config_name || "").trim();
      if (!selected) {
        return {
          path: "",
          track: cfg?.track || null,
          existing: false,
          generated: false,
          validation: { ok: true, message: "" },
        };
      }
      const cfgTrack = cfg?.track || null;
      const validation = validateConfigPathName(selected, cfgTrack);
      if (!validation.ok) {
        return { path: "", track: cfgTrack, existing: false, generated: false, validation };
      }
      const path = configPathForSelection(selected, cfgTrack);
      const inferredTrack = configNameHasPath(selected) ? inferTrackFromConfigPath(path) : null;
      const mode = cfg?.config_mode || configUseMode;
      const existing = mode === "existing";
      const selectedTrack = cfgTrack || inferredTrack;
      return {
        path,
        track: selectedTrack,
        mode,
        existing,
        generated: !existing,
        configType: existing ? inferExistingConfigType(path, selectedTrack) : null,
        validation,
      };
    }

    function buildExistingConfigReference(selection, cfg) {
      const selectedTrack = cfg?.track || selection.track;
      const configType = selection.configType || inferExistingConfigType(selection.path, selectedTrack);
      const command = buildRunCommand(configType, selection.path, selectedTrack);
      return {
        source: "existing_config",
        config_path: selection.path,
        config_type: configType,
        entrypoint: entrypointForConfigType(configType, selectedTrack),
        command,
      };
    }

    function withGeneratedConfigMeta(doc, configPath, configType) {
      if (!doc) return doc;
      return {
        source: "generated_config",
        config_path: configPath,
        config_type: configType,
        save_policy: "save current visual assembly as this JSON before mock submit",
        ...doc,
      };
    }

    function setInputValidation(input, validation) {
      if (!input) return;
      input.classList.toggle("is-invalid", !validation.ok);
      input.setAttribute("aria-invalid", validation.ok ? "false" : "true");
      input.title = validation.message || "";
    }

    function makeConfigNameEntry(name) {
      const raw = String(name || "").trim();
      return normalizeEnvDrop({
        id: envSlots.config_name?.id || "e" + ++idSeq,
        target: "env",
        envSlot: "config_name",
        kind: "env",
        value: raw || envCustomPlaceholder("config_name"),
        title: raw || "config JSON",
        sub: "--config",
        envCustom: true,
        customName: raw,
      });
    }

    function makeTrackEntry(track, inferredFromConfig = false) {
      return normalizeEnvDrop({
        id: envSlots.track?.value === track ? envSlots.track.id : "e" + ++idSeq,
        target: "env",
        envSlot: "track",
        kind: "env",
        value: track,
        title: track === "cora" ? "Cora" : "Vision",
        sub: inferredFromConfig ? "from config JSON" : track === "cora" ? "run_graph_ablation" : "run_vision_suite",
        inferredFromConfig,
      });
    }

    function setTrackFromExistingConfigPath(path) {
      const inferredTrack = inferTrackFromConfigPath(path);
      if (!inferredTrack || currentTrack() === inferredTrack) return;
      envSlots.track = makeTrackEntry(inferredTrack, true);
      clearEnvDownstream("track");
    }

    function clearVisualAssemblyForExistingConfig() {
      envSlots.dataset = null;
      envSlots.diagnostics = [];
      Object.keys(envSweepSlots).forEach((slotId) => {
        envSweepSlots[slotId] = [];
      });
      resetEnvColumns(1);
      const chain = document.getElementById("env-cases-chain");
      if (chain) chain.innerHTML = "";
    }

    function setDirectConfigName(rawName) {
      const raw = String(rawName || "").trim();
      if (!configModeTouched) {
        configUseMode = looksLikeRepoConfigPath(raw) ? "existing" : "generate";
      }
      if (!raw) {
        envSlots.config_name = null;
        if (envSlots.track?.inferredFromConfig) {
          envSlots.track = null;
          clearEnvDownstream("track");
        }
      } else {
        envSlots.config_name = makeConfigNameEntry(raw);
        if (configNameHasPath(raw)) {
          setTrackFromExistingConfigPath(configPathForSelection(raw, currentTrack()));
        }
        if (configUseMode === "existing" && (currentTrack() || configNameHasPath(raw))) {
          clearVisualAssemblyForExistingConfig();
        }
      }
      renderEnvSlots();
      applyEnvChainGating();
      syncUI();
    }

    function setConfigUseMode(mode, touched = true) {
      configUseMode = mode === "existing" ? "existing" : "generate";
      if (touched) configModeTouched = true;
      const raw = String(resolvedEnvSlotValue(envSlots.config_name) || "").trim();
      if (configUseMode === "existing" && configNameHasPath(raw)) {
        setTrackFromExistingConfigPath(configPathForSelection(raw, currentTrack()));
        clearVisualAssemblyForExistingConfig();
      } else if (configUseMode === "existing" && raw && currentTrack()) {
        clearVisualAssemblyForExistingConfig();
      } else if (configUseMode === "generate" && configNameHasPath(raw)) {
        setTrackFromExistingConfigPath(configPathForSelection(raw, currentTrack()));
        if (envSlots.track?.inferredFromConfig) {
          envSlots.track = {
            ...envSlots.track,
            inferredFromConfig: false,
            sub: envSlots.track.value === "cora" ? "run_graph_ablation" : "run_vision_suite",
          };
        }
      } else if (configUseMode === "generate" && envSlots.track?.inferredFromConfig) {
        envSlots.track = {
          ...envSlots.track,
          inferredFromConfig: false,
          sub: envSlots.track.value === "cora" ? "run_graph_ablation" : "run_vision_suite",
        };
      }
      renderEnvSlots();
      applyEnvChainGating();
      syncUI();
    }

    function chooseConfigQuickPick(path, mode) {
      configUseMode = mode === "existing" ? "existing" : "generate";
      configModeTouched = true;
      const input = document.getElementById("config-direct-input");
      if (input) input.value = path;
      setDirectConfigName(path);
    }

    function existingConfigKindLabel(configType) {
      if (configType === "existing-cora-graph-ablation") return "Cora ablation";
      if (configType === "existing-vision-single") return "Vision single";
      if (configType === "existing-vision-client-count-sweep") return "Client sweep";
      if (configType === "existing-vision-stress-grid") return "Stress grid";
      return "Vision suite";
    }

    function updateConfigQuickPickState(raw) {
      const selected = String(raw || "").trim().replace(/\\/g, "/").toLowerCase();
      document.querySelectorAll("[data-config-pick]").forEach((btn) => {
        const mode = btn.dataset.configMode || "existing";
        const value = String(btn.dataset.configPick || "").trim().replace(/\\/g, "/").toLowerCase();
        btn.classList.toggle("is-hidden", mode !== configUseMode);
        btn.classList.toggle("is-active", mode === configUseMode && value === selected);
      });
    }

    function updateConfigDirectInputState() {
      const input = document.getElementById("config-direct-input");
      const state = document.getElementById("config-direct-state");
      if (!input || !state) return;
      const existingBtn = document.getElementById("config-mode-existing");
      const generateBtn = document.getElementById("config-mode-generate");

      const currentValue = String(resolvedEnvSlotValue(envSlots.config_name) || "");
      if (document.activeElement !== input) input.value = currentValue;
      input.placeholder =
        configUseMode === "existing"
          ? "configs/vision/smoke/extension.json"
          : "my_graphfl_demo";
      input.disabled = false;
      [existingBtn, generateBtn].forEach((btn) => {
        if (!btn) return;
        btn.disabled = false;
        btn.classList.toggle(
          "is-active",
          (btn === existingBtn && configUseMode === "existing") ||
            (btn === generateBtn && configUseMode === "generate")
        );
        btn.setAttribute(
          "aria-pressed",
          ((btn === existingBtn && configUseMode === "existing") ||
            (btn === generateBtn && configUseMode === "generate")).toString()
        );
      });

      const raw = String(input.value || "").trim();
      const cfg = buildEnvConfig();
      const selection = resolveConfigSelection({ ...cfg, config_name: raw, config_mode: configUseMode });
      let validation = selection.validation;
      updateConfigQuickPickState(raw);
      if (raw && configUseMode === "existing" && !selection.track) {
        validation = {
          ok: false,
          message: "기존 JSON은 configs/vision/... 또는 configs/cora/... 경로로 지정하거나 트랙을 먼저 선택하세요.",
        };
      }
      setInputValidation(input, validation);

      state.className = "";
      if (!raw) {
        state.textContent = "1번: 기존 configs/...json 경로 또는 새 JSON 이름";
        state.classList.add("warn");
        state.textContent = configUseMode === "existing" ? "repo JSON 선택" : "새 JSON 이름";
        return;
      }
      if (!validation.ok) {
        state.textContent = validation.message;
        state.classList.add("bad");
        return;
      }
      if (selection.mode === "existing") {
        state.textContent = `기존 JSON 확정: ${selection.path}`;
      } else if (raw.includes("/") || raw.includes("\\")) {
        state.textContent = currentTrack()
          ? `실행 시 현재 조립을 새 JSON으로 저장: ${selection.path}`
          : "새 JSON 저장: 다음으로 트랙/데이터/비교열 선택";
      } else {
        state.textContent = currentTrack()
          ? `실행 시 현재 조립을 새 JSON으로 저장: ${selection.path}`
          : "새 JSON 저장: 다음으로 트랙 선택";
      }
      if (selection.mode === "existing") {
        const entry = entrypointForConfigType(selection.configType, selection.track);
        state.textContent = `${existingConfigKindLabel(selection.configType)} · ${entry} · locked`;
      } else {
        state.textContent = currentTrack() ? `save · ${selection.path}` : "save · pick track";
      }
      state.textContent =
        selection.mode === "existing"
          ? `${existingConfigKindLabel(selection.configType)} | ${entrypointForConfigType(selection.configType, selection.track)} | locked`
          : currentTrack()
            ? `save | ${selection.path}`
            : "save | pick track";
      state.classList.add(selection.mode === "generate" && !currentTrack() ? "warn" : "ok");
    }

    function safePythonIdentifier(name, fallback) {
      const raw = String(name || fallback || "my_name").trim();
      if (PY_IDENTIFIER_RE.test(raw)) return { name: raw, warning: "" };
      let normalized = raw.replace(/[^A-Za-z0-9_]/g, "_").replace(/^[^A-Za-z_]+/, "");
      if (!normalized) normalized = fallback || "my_name";
      if (!/^[A-Za-z_]/.test(normalized)) normalized = `x_${normalized}`;
      normalized = normalized.slice(0, 64);
      return {
        name: normalized,
        warning: `# 경고: '${raw}'는 repo/Python 식별자로 안전하지 않아 '${normalized}' 예시로 정규화했습니다.\n`,
      };
    }

    function collectCustomValidationIssues() {
      const issues = [];
      const visit = (entry, label) => {
        if (!entry) return;
        const name = String(entry.customName || "").trim();
        if (!name) return;
        const validation = validateCustomName(entry.value, entry.envSlot, name);
        if (!validation.ok) issues.push(`${label}: ${validation.message}`);
      };

      Object.entries(envSlots).forEach(([slotId, value]) => {
        if (Array.isArray(value)) value.forEach((entry, i) => visit({ ...entry, envSlot: entry.envSlot || slotId }, `${slotId} ${i + 1}`));
        else visit({ ...value, envSlot: value?.envSlot || slotId }, slotId);
      });
      Object.entries(envSweepSlots).forEach(([slotId, values]) => {
        (values || []).forEach((entry, i) => visit({ ...entry, envSlot: entry.envSlot || slotId }, `${slotId} ${i + 1}`));
      });
      envColumns.forEach((col, i) => {
        visit({ ...col.partition, envSlot: col.partition?.envSlot || "partition" }, `열 ${i + 1} 파티션`);
        visit({ ...col.preset, envSlot: col.preset?.envSlot || "sweep_variants" }, `열 ${i + 1} preset`);
        (col.lifeStack || []).forEach((entry, j) => visit(entry, `열 ${i + 1} 확장 ${j + 1}`));
      });
      return issues;
    }

    function extendSnippet(value, customName) {
      const safe = value && String(value).startsWith("name_")
        ? safePythonIdentifier(customName, EXTEND_PLACEHOLDERS[value] || "my_name")
        : { name: customName || EXTEND_PLACEHOLDERS[value] || "my_name", warning: "" };
      const n = safe.name;
      if (value === "name_graph_source")
        return `${safe.warning}@register_graph_source('${n}')\ndef encode_${n}(ctx): ...`;
      if (value === "name_relation") return `${safe.warning}@register_relation('${n}')\ndef build_${n}(ctx): ...`;
      if (value === "name_graph_mode")
        return `${safe.warning}@register_graph_builder('${n}')\ndef build_${n}(ctx): ...`;
      if (value === "name_aggregation") return `${safe.warning}# register aggregation_target: ${n}`;
      if (value === "name_correction") return `${safe.warning}# register correction_family: ${n}`;
      if (value === "name_method") return `${safe.warning}# register method: ${n}`;
      return `# ${value}: ${n}`;
    }

    function payload(el) {
      const input = el.querySelector(".extend-input");
      const value = el.dataset.value;
      const envSlot = el.dataset.envSlot || null;
      const envCustom = ENV_CUSTOM_VALUES.has(value);
      const base = {
        target: el.dataset.target,
        envSlot,
        kind: el.dataset.kind,
        value,
        title: el.dataset.title,
        sub: el.dataset.sub,
      };
      if (!envCustom) return base;
      const customName = input ? input.value.trim() : "";
      return { ...base, envCustom: true, customName };
    }

    function bindExtendInput(el, entry) {
      const input = el.querySelector(".extend-input");
      if (!input) return;
      setInputValidation(input, validateCustomName(entry.value, entry.envSlot, input.value.trim()));
      input.addEventListener("mousedown", (e) => e.stopPropagation());
      input.addEventListener("click", (e) => e.stopPropagation());
      input.addEventListener("input", () => {
        entry.customName = input.value.trim();
        setInputValidation(input, validateCustomName(entry.value, entry.envSlot, entry.customName));
        syncUI();
      });
    }

    function makeDeleteButton() {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "del";
      btn.title = "삭제";
      btn.textContent = "×";
      return btn;
    }

    function makeExtendTextInput(placeholder, value) {
      const input = document.createElement("input");
      input.type = "text";
      input.className = "extend-input";
      input.placeholder = placeholder;
      input.value = value || "";
      input.autocomplete = "off";
      input.spellcheck = false;
      return input;
    }

    function makeExtendNode(data, stackName) {
      const el = document.createElement("div");
      const ph = EXTEND_PLACEHOLDERS[data.value] || "my_name";
      const name = data.customName || "";
      const axisTitle = data.axisTitle || data.title;
      const axisCls = EXTEND_AXIS_CLASS[data.value] || "";
      el.className = `scratch-block stack extend workspace-block${axisCls ? ` ${axisCls}` : ""}`;
      el.dataset.kind = "extend";
      el.dataset.value = data.value || "";
      el.dataset.title = axisTitle;
      el.dataset.sub = data.sub || "";
      el.dataset.id = data.id;
      const del = makeDeleteButton();
      const body = document.createElement("div");
      body.className = "body extend-body";
      const tag = document.createElement("span");
      tag.className = "extend-tag";
      tag.textContent = `${axisTitle} · 새 이름`;
      const input = makeExtendTextInput(ph, name);
      const sub = document.createElement("span");
      sub.className = "sub";
      sub.textContent = data.sub || "";
      body.appendChild(tag);
      body.appendChild(input);
      body.appendChild(sub);
      el.appendChild(del);
      el.appendChild(body);
      const entry = { ...data, axisTitle, customName: name };
      del.onclick = (e) => {
        e.stopPropagation();
        removeBlock(data.id, stackName);
      };
      el.draggable = true;
      el.addEventListener("dragstart", (e) => {
        if (e.target.classList.contains("extend-input")) {
          e.preventDefault();
          return;
        }
        const input = el.querySelector(".extend-input");
        e.dataTransfer.setData(
          "application/json",
          JSON.stringify({ ...entry, customName: input.value.trim(), from: stackName, move: true })
        );
        el.style.opacity = "0.4";
      });
      el.addEventListener("dragend", () => { el.style.opacity = "1"; });
      bindExtendInput(el, entry);
      return el;
    }

    function makeEnvCustomNode(data, stackName) {
      const el = document.createElement("div");
      const slotId =
        data.envSlot ||
        (data.value === "custom_config"
          ? "config_name"
          : data.value === "custom_partition"
            ? "partition"
            : "dataset");
      const ph = envCustomDefaultName(slotId);
      const raw = data.value || envCustomPlaceholder(slotId);
      let name = ENV_CUSTOM_VALUES.has(raw)
        ? String(data.customName || "").trim()
        : String(data.customName || raw).trim();
      const tag =
        slotId === "partition"
          ? "파티션 · 이름"
          : slotId === "config_name"
            ? "config JSON · 이름"
          : slotId === "sweep_variants"
            ? "variant · repo"
            : "데이터셋 · 이름";
      el.className = `scratch-block stack env env-h env-custom-block workspace-block`;
      el.dataset.kind = "env";
      el.dataset.value = data.value || "";
      el.dataset.title = data.title;
      el.dataset.sub = data.sub || "";
      el.dataset.id = data.id;
      const del = makeDeleteButton();
      const body = document.createElement("div");
      body.className = "body extend-body";
      const tagEl = document.createElement("span");
      tagEl.className = "extend-tag";
      tagEl.textContent = tag;
      const input = makeExtendTextInput(ph, name);
      const sub = document.createElement("span");
      sub.className = "sub";
      sub.textContent = data.sub || "";
      body.appendChild(tagEl);
      body.appendChild(input);
      body.appendChild(sub);
      el.appendChild(del);
      el.appendChild(body);
      const entry = {
        ...data,
        envSlot: slotId,
        envCustom: true,
        customName: name,
        value: name || envCustomPlaceholder(slotId),
        title: name || (slotId === "partition" ? "partition" : slotId === "config_name" ? "config JSON" : "dataset"),
      };
      del.onclick = (e) => {
        e.stopPropagation();
        removeBlock(data.id, stackName);
      };
      bindEnvCustomInput(el, entry, stackName);
      el.draggable = true;
      el.addEventListener("dragstart", (e) => {
        if (e.target.classList.contains("extend-input")) {
          e.preventDefault();
          return;
        }
        const input = el.querySelector(".extend-input");
        const payload = {
          ...entry,
          customName: input.value.trim(),
          envSlot: data.envSlot,
          target: "env",
          from: stackName,
          move: true,
        };
        e.dataTransfer.setData("application/json", JSON.stringify(payload));
        el.style.opacity = "0.4";
      });
      el.addEventListener("dragend", () => { el.style.opacity = "1"; });
      return el;
    }

    function bindEnvCustomInput(el, entry, stackName) {
      const input = el.querySelector(".extend-input");
      if (!input) return;
      setInputValidation(input, validateCustomName(entry.value, entry.envSlot, input.value.trim()));
      input.addEventListener("mousedown", (e) => e.stopPropagation());
      input.addEventListener("click", (e) => e.stopPropagation());
      input.addEventListener("input", () => {
        const name = input.value.trim();
        const slotId = entry.envSlot;
        setInputValidation(input, validateCustomName(entry.value, slotId, name));
        entry.customName = name;
        if (name) {
          entry.value = name;
          entry.title = name;
        } else {
          entry.value = envCustomPlaceholder(slotId);
          entry.title = slotId === "partition" ? "partition" : slotId === "config_name" ? "config JSON" : "dataset";
          entry.customName = "";
        }
        if (stackName === "env") {
          if (slotId && envSlots[slotId]?.id === entry.id) {
            envSlots[slotId] = { ...envSlots[slotId], ...entry, envCustom: true };
            if (slotId === "config_name") {
              if (!configModeTouched) {
                configUseMode = looksLikeRepoConfigPath(name) ? "existing" : "generate";
              }
              if (name && configUseMode === "existing" && configNameHasPath(name)) {
                setTrackFromExistingConfigPath(configPathForSelection(name, currentTrack()));
                clearVisualAssemblyForExistingConfig();
              } else if (name && configUseMode === "existing" && currentTrack()) {
                clearVisualAssemblyForExistingConfig();
              } else if (!name && envSlots.track?.inferredFromConfig) {
                envSlots.track = null;
                clearEnvDownstream("track");
              }
            }
          }
        }
        syncUI();
      });
    }

    function makeCompareNode(data, stackName) {
      const el = document.createElement("div");
      const isBaseline = data.compareKind === "baseline";
      el.className = `scratch-block stack env env-h env-compare-card workspace-block${isBaseline ? " baseline" : ""}`;
      el.dataset.kind = "env";
      el.dataset.value = data.value || "";
      el.dataset.title = data.title;
      el.dataset.sub = data.sub || "";
      el.dataset.id = data.id;
      const title = data.title || compareEntryVariantToken(data) || "?";
      const del = makeDeleteButton();
      const body = document.createElement("div");
      body.className = "body";
      body.appendChild(document.createTextNode(title));
      if (data.compareKind === "assembly") {
        const axes = document.createElement("span");
        axes.className = "compare-axes";
        [
          ["graph_source", data.assembly?.graph_source],
          ["relation", data.assembly?.relation],
          ["graph_mode", data.assembly?.graph_mode],
          ["aggregation_target", data.assembly?.aggregation_target],
          ["correction_family", data.assembly?.correction_family],
          ["control_graph_mode", data.assembly?.control_graph_mode],
          ["diagnostic", data.assembly?.diagnostic],
        ]
          .filter(([, v]) => v)
          .forEach(([k, v], i) => {
            if (i > 0) axes.appendChild(document.createElement("br"));
            axes.appendChild(document.createTextNode(`${k}: ${v}`));
          });
        body.appendChild(axes);
      } else {
        const sub = document.createElement("span");
        sub.className = "sub";
        sub.textContent = data.sub || "";
        body.appendChild(sub);
      }
      el.appendChild(del);
      el.appendChild(body);
      del.onclick = (e) => {
        e.stopPropagation();
        removeBlock(data.id, stackName);
      };
      el.draggable = true;
      el.addEventListener("dragstart", (e) => {
        e.dataTransfer.setData(
          "application/json",
          JSON.stringify({ ...data, from: stackName, move: true, envSlot: "sweep_variants" })
        );
        el.style.opacity = "0.4";
      });
      el.addEventListener("dragend", () => {
        el.style.opacity = "1";
      });
      return el;
    }

    function makeNode(data, stackName) {
      if (stackName === "env" && data.envSlot === "sweep_variants" && data.compareKind) {
        return makeCompareNode(data, stackName);
      }
      if (data.kind === "extend") return makeExtendNode(data, stackName);
      if (stackName === "env" && isEnvCustomData(data)) return makeEnvCustomNode(data, stackName);
      const el = document.createElement("div");
      const cls = data.kind === "hat" ? "hat" : "stack";
      const envH = stackName === "env" ? " env-h" : "";
      el.className = `scratch-block ${cls} ${data.kind}${envH} workspace-block`;
      el.dataset.kind = data.kind;
      el.dataset.value = data.value || "";
      el.dataset.title = data.title;
      el.dataset.sub = data.sub || "";
      el.dataset.id = data.id;
      if (stackName !== "life" || data.kind !== "hat") {
        const del = makeDeleteButton();
        const body = document.createElement("div");
        body.className = "body";
        body.appendChild(document.createTextNode(data.title || ""));
        const sub = document.createElement("span");
        sub.className = "sub";
        sub.textContent = data.sub || "";
        body.appendChild(sub);
        el.appendChild(del);
        el.appendChild(body);
        del.onclick = (e) => {
          e.stopPropagation();
          const colId = el.closest(".column-life-stack")?.dataset.columnId;
          removeBlock(data.id, stackName, colId);
        };
        el.draggable = true;
        el.addEventListener("dragstart", (e) => {
          e.dataTransfer.setData("application/json", JSON.stringify({ ...data, from: stackName, move: true }));
          el.style.opacity = "0.4";
        });
        el.addEventListener("dragend", () => { el.style.opacity = "1"; });
      } else {
        const body = document.createElement("div");
        body.className = "body";
        body.appendChild(document.createTextNode(data.title || ""));
        const sub = document.createElement("span");
        sub.className = "sub";
        sub.textContent = data.sub || "";
        body.appendChild(sub);
        el.appendChild(body);
      }
      return el;
    }

    function removeBlock(id, stackName, colId) {
      if (stackName === "env" || stackName === "life" || stackName === "column") {
        ENV_SLOT_IDS.forEach((slotId) => {
          if (envSlots[slotId]?.id === id) envSlots[slotId] = null;
        });
        const di = envSlots.diagnostics.findIndex((b) => b.id === id);
        if (di >= 0) envSlots.diagnostics.splice(di, 1);
        envColumns.forEach((col) => {
          if (col.partition?.id === id) col.partition = null;
          if (col.preset?.id === id) {
            col.preset = null;
            col.caseMode = null;
          }
        });
        renderEnvSlots();
        renderCasesChain();
        setupEnvSlotDrops();
        applySweepStackUI();
      } else {
        const col = colId ? getColumnById(colId) : envColumns.find((c) => c.lifeStack.some((b) => b.id === id));
        if (!col) return;
        const i = col.lifeStack.findIndex((b) => b.id === id);
        if (i >= 0) col.lifeStack.splice(i, 1);
        const node = document.querySelector(
          `.column-life-stack[data-column-id="${col.id}"] [data-id="${id}"]`
        );
        if (node) node.remove();
        renderColumnLifeStack(col);
      }
      syncUI();
    }

    function relayoutLifeStackZ() {
      envColumns.forEach((col) => renderColumnLifeStack(col));
    }

    function renderEnvSlots() {
      document.querySelectorAll("#env-base-row .env-h-slot").forEach((slotEl) => {
        const slotId = slotEl.dataset.envSlot;
        const ph = slotEl.querySelector(".env-h-ph");
        const multi = slotEl.dataset.multi === "true";
        slotEl.querySelectorAll(".workspace-block").forEach((n) => n.remove());
        if (slotEl.dataset.layout === "stack") ensureStackSocket(slotEl);
        if (multi && slotId === "diagnostics") {
          const store = envSlots.diagnostics || [];
          slotEl.classList.toggle("filled", store.length > 0);
          store.forEach((data) => {
            slotEl.appendChild(makeNode({ ...data, kind: "diagnostic", target: "env", envSlot: "diagnostics" }, "env"));
          });
        } else {
          const data = envSlots[slotId];
          slotEl.classList.toggle("filled", !!data);
          if (data) {
            slotEl.appendChild(makeNode({ ...data, kind: data.kind || "env" }, "env"));
          }
        }
        if (ph) slotEl.insertBefore(ph, slotEl.firstChild);
      });
    }

    function addEnvStackBlock(slotId, data) {
      const id = data.id && data.move ? data.id : "e" + ++idSeq;
      let entry = normalizeEnvDrop({
        ...data,
        id,
        envSlot: slotId,
        kind: slotId === "diagnostics" ? "diagnostic" : "env",
        target: "env",
      });
      delete entry.move;
      delete entry.from;
      if (slotId === "sweep_variants") entry = normalizeCompareEntry(entry);
      const sig =
        slotId === "sweep_variants" ? compareSignature(entry) : resolvedEnvSlotValue(entry);
      const store = envStackStore(slotId);
      if (!store) return;
      if (data.move) {
        SWEEP_SLOT_IDS.forEach((sid) => {
          const s = envStackStore(sid);
          if (!s) return;
          const idx = s.findIndex((b) => b.id === id);
          if (idx >= 0) s.splice(idx, 1);
        });
        const di = envSlots.diagnostics.findIndex((b) => b.id === id);
        if (di >= 0) envSlots.diagnostics.splice(di, 1);
      } else {
        const dup = store.some((b) =>
          slotId === "sweep_variants"
            ? compareSignature(b) === sig
            : slotId === "diagnostics"
              ? b.value === entry.value
              : resolvedEnvSlotValue(b) === sig
        );
        if (dup) return;
      }
      store.push(entry);
      renderEnvSlots();
      if (slotId === "partition") updatePartitionAlphaUI();
      applySweepStackUI();
      syncUI();
    }

    function acceptEnvDrop(data, targetSlotId, colId) {
      if (colId) {
        if (targetSlotId === "column_stem") {
          return isBaselineBlock(data) ? columnStemOpen(getColumnById(colId)) : blockCanInstall(data, getColumnById(colId));
        }
        if (targetSlotId === "partition") {
          return envSlotUnlocked("partition", colId);
        }
        return false;
      }
      const isEnv = data.kind === "env" || data.target === "env" || !!data.envSlot;
      if (!isEnv) return false;
      if (!envSlotUnlocked(targetSlotId)) return false;
      if (data.move) {
        data.envSlot = targetSlotId;
        return true;
      }
      const slot = data.envSlot || targetSlotId;
      if (slot && slot !== targetSlotId && slot !== "sweep_variants" && slot !== "diagnostics") {
        return false;
      }
      if (targetSlotId === "diagnostics") {
        data.envSlot = targetSlotId;
        return true;
      }
        if (targetSlotId === "column_stem" && colId) {
        data.envSlot = "column_stem";
        return envSlotUnlocked("column_stem", colId);
      }
      data.envSlot = targetSlotId;
      return true;
    }

    function handleWorkspaceDrop(e, slotEl) {
      const raw = e.dataTransfer.getData("application/json");
      if (!raw) return;
      const data = JSON.parse(raw);
      installBlock(data, {
        colId: slotEl.dataset.envColumn || null,
        slotId: slotEl.dataset.envSlot,
        slotEl,
      });
    }

    function handleColumnStackDrop(e, stackEl) {
      const raw = e.dataTransfer.getData("application/json");
      if (!raw) return;
      const data = JSON.parse(raw);
      installBlock(data, { colId: stackEl.dataset.columnId, slotEl: stackEl });
    }

    function setEnvSlot(slotId, data) {
      const prevTrack = envSlots.track?.value;
      const prevDataset = resolvedEnvSlotValue(envSlots.dataset);
      const id = data.id && data.move ? data.id : "e" + ++idSeq;
      const entry = normalizeEnvDrop({
        ...data,
        id,
        envSlot: slotId,
        kind: "env",
        target: "env",
      });
      delete entry.move;
      delete entry.from;
      envSlots[slotId] = entry;
      if (slotId === "config_name") {
        const configName = resolvedEnvSlotValue(entry);
        if (!configModeTouched) {
          configUseMode = looksLikeRepoConfigPath(configName) ? "existing" : "generate";
        }
        if (configUseMode === "existing" && configNameHasPath(configName)) {
          setTrackFromExistingConfigPath(configPathForSelection(configName, currentTrack()));
          clearVisualAssemblyForExistingConfig();
        } else if (configUseMode === "existing" && configName && currentTrack()) {
          clearVisualAssemblyForExistingConfig();
        }
      }
      if (slotId === "track" && entry.value !== prevTrack) {
        clearEnvDownstream("track");
      } else if (slotId === "dataset") {
        const nextDs = resolvedEnvSlotValue(entry);
        if (nextDs !== prevDataset) clearEnvDownstream("dataset");
        if (entry.value === "cora") envColumnCount = 1;
      }
      renderEnvSlots();
      if (entry.envCustom) {
        requestAnimationFrame(() => {
          document
            .querySelector(`#env-base-row .env-h-slot[data-env-slot="${slotId}"] .extend-input`)
            ?.focus();
        });
      }
      if (slotId === "track") applyTrackContext();
      else applyEnvChainGating();
      syncUI();
    }

    const EXTEND_AXIS = {
      name_graph_source: "graph_source",
      name_relation: "relation",
      name_graph_mode: "graph_mode",
      name_aggregation: "aggregation_target",
      name_correction: "correction_family",
      name_method: "method",
    };
    const KIND_AXIS = {
      client: "graph_source",
      relation: "relation",
      topology: "graph_mode",
      aggregation: "aggregation_target",
      correction: "correction_family",
      method: "method",
      baseline: "method",
      diagnostic: "diagnostic",
    };

    function lifeAxisKey(block) {
      if (!block) return "";
      if (block.kind === "extend") return EXTEND_AXIS[block.value] || block.value;
      return KIND_AXIS[block.kind] || block.kind;
    }

    function replaceKind(stack, stackEl, data, stackName, colId) {
      const multi = new Set(["diagnostic", "custom"]);
      let same;
      if (stackName === "env") {
        same = [];
      } else if (multi.has(data.kind)) {
        same = [];
      } else {
        const axis = lifeAxisKey(data);
        same = stack.filter((b) => lifeAxisKey(b) === axis);
      }
      same.forEach((b) => removeBlock(b.id, stackName, colId));
      if (data.move) {
        removeBlock(data.id, data.from === "env" ? "env" : "life", colId);
      }
      const id = data.move ? data.id : "b" + ++idSeq;
      const entry = {
        ...data,
        id,
        axisTitle: data.axisTitle || data.title,
        customName: data.customName || "",
      };
      stack.push(entry);
      if (stackEl) {
        const col = colId ? getColumnById(colId) : null;
        if (col) renderColumnLifeStack(col);
        else stackEl.appendChild(makeNode(entry, stackName));
      }
      if (stackName === "life") relayoutLifeStackZ();
      syncUI();
    }

    function addBlock(data) {
      tryInstallBlock(data, {
        colId: data.envColumn,
        slotId: data.envSlot,
      });
    }

    function getLife(kind) {
      return getColumnLife(envColumns[0], kind);
    }

    function getMethodPart(col) {
      const c = col || envColumns[0];
      if (!c) return { value: "ours" };
      if (c.caseMode === "baseline" && c.preset) {
        const v =
          c.preset.method ||
          compareEntryVariantToken(c.preset) ||
          c.preset.value ||
          "fedavg";
        return { value: v };
      }
      if (c.caseMode === "assembly") return { value: "ours" };
      return { value: "ours" };
    }

    function hasDiagnostic(...values) {
      if ((envSlots.diagnostics || []).some((b) => values.includes(b.value))) return true;
      return envColumns.some((col) =>
        col.lifeStack.some((b) => b.kind === "diagnostic" && values.includes(b.value))
      );
    }

    function envFilled() {
      const base = ENV_SLOT_IDS.every((id) => !!envSlots[id]);
      const cols = envColumns.some(
        (c) => c.partition || c.preset || c.lifeStack.length > 0
      );
      return base || cols;
    }

    function visionPartitionReady(cfg) {
      if (cfg.track !== "vision") return true;
      const n = getTargetColumnCount();
      if (!n) return false;
      return envColumns.slice(0, n).every((c) => resolvedEnvSlotValue(c.partition));
    }

    function buildEnvConfig() {
      const nums = getEnvNums();
      const cfg = { ...nums };
      if (envSlots.track) cfg.track = envSlots.track.value;
      const ds = resolvedEnvSlotValue(envSlots.dataset);
      if (ds) cfg.dataset = ds;
      const configName = resolvedEnvSlotValue(envSlots.config_name);
      if (configName) cfg.config_name = configName;
      cfg.config_mode = configUseMode;
      if (!cfg.track && cfg.config_name && configNameHasPath(cfg.config_name)) {
        const inferredTrack = inferTrackFromConfigPath(configPathForSelection(cfg.config_name, null));
        if (inferredTrack) cfg.track = inferredTrack;
      }
      const track = cfg.track;
      cfg.runner = effectiveRunner(track);
      const parts = partitionValues();
      if (parts.length) cfg.partition = parts[0];
      if (envColumns[0]?.partition && usesDirichletAlpha(resolvedEnvSlotValue(envColumns[0].partition))) {
        cfg.dirichlet_alpha = envColumns[0].dirichletAlpha;
      }
      const sweep = readSweepFromUI();
      if (sweep) cfg.sweep = sweep;
      cfg.column_count = getTargetColumnCount();
      return cfg;
    }

    function envChainComplete() {
      const cfg = buildEnvConfig();
      if (!cfg.track || !cfg.dataset) return false;
      return visionPartitionReady(cfg);
    }

    function appendPartitionCliFlags(cfg, flagLines) {
      const part = cfg.partition || "dirichlet";
      if (part === "iid") {
        flagLines.push("--partition iid");
        return;
      }
      if (part === "dirichlet") {
        flagLines.push(`--partition dirichlet --dirichlet-alpha ${cfg.dirichlet_alpha}`);
        return;
      }
      flagLines.push(`--partition ${part}`);
      if (usesDirichletAlpha(part)) {
        flagLines.push(`--dirichlet-alpha ${cfg.dirichlet_alpha}`);
      }
    }

    function lifecycleParts(col) {
      return columnLifecycleParts(col || envColumns[0]);
    }

    function runReadiness() {
      const cfg = buildEnvConfig();
      const configSelection = resolveConfigSelection(cfg);
      const missing = [];
      if (!cfg.config_name) missing.push("config JSON 이름");
      if (cfg.config_name && !configSelection.validation.ok) missing.push(configSelection.validation.message);
      const selectedTrack = cfg.track || configSelection.track;
      const directExistingConfig = !!(
        cfg.config_name &&
        configSelection.existing &&
        selectedTrack &&
        configSelection.validation.ok
      );
      const unresolvedExistingConfig = !!(
        cfg.config_name &&
        configSelection.existing &&
        !selectedTrack &&
        configSelection.validation.ok
      );
      if (!directExistingConfig) {
        if (!cfg.track) missing.push("트랙");
      } else if (!cfg.track && selectedTrack) {
        cfg.track = selectedTrack;
        cfg.runner = effectiveRunner(selectedTrack);
      }
      if (unresolvedExistingConfig) {
        missing.push("기존 JSON 경로(configs/vision 또는 configs/cora)");
      }
      if (!directExistingConfig && !unresolvedExistingConfig) {
        if (!cfg.dataset) missing.push("데이터셋");
        if (cfg.track === "vision" && !visionPartitionReady(cfg)) missing.push("각 열 파티션");
      }
      const n = getTargetColumnCount();
      if (!directExistingConfig && !unresolvedExistingConfig && cfg.track && cfg.track !== "cora") {
        envColumns.slice(0, n).forEach((col, i) => {
          if (!columnCaseGraphReady(col)) {
            missing.push(`열 ${i + 1}: fedavg(완성) 또는 Graph-FL 조립`);
          }
        });
      }
      collectCustomValidationIssues().forEach((issue) => missing.push(issue));
      const parts = lifecycleParts(envColumns[0]);
      return { ready: missing.length === 0, missing, cfg, parts };
    }

    function suggestConfigPath(cfg) {
      return resolveConfigSelection(cfg).path;
    }

    function compactObject(obj) {
      return Object.fromEntries(
        Object.entries(obj).filter(([, value]) => value !== null && value !== undefined && value !== "")
      );
    }

    function buildConfigDocument(cfg, parts) {
      const method = getMethodPart()?.value || "ours";
      const diagnostics = (envSlots.diagnostics || []).map((b) => b.value).filter(Boolean);
      const extensions = envColumns
        .flatMap((c) => c.lifeStack)
        .filter((b) => b.kind === "custom" || b.kind === "extend")
        .map((b) => ({
          register: b.value,
          label: b.axisTitle || b.title,
          name: b.customName || EXTEND_PLACEHOLDERS[b.value] || b.sub || b.title,
        }));

      const baseArgs = {
        track: cfg.track || null,
        dataset: cfg.dataset || null,
        model: cfg.track === "cora" ? "gcn" : cfg.track === "vision" ? "mlp" : null,
        num_clients: cfg.num_clients,
        rounds: cfg.rounds,
        local_epochs: 1,
        batch_size: 64,
        seeds: [42],
        partition: cfg.partition || null,
        runner: cfg.runner || null,
        method,
        graph_source: parts.graph_source,
        graph_mode: parts.graph_mode,
        aggregation_target: parts.aggregation_target,
        correction_family: parts.correction_family,
      };
      if (usesDirichletAlpha(cfg.partition)) baseArgs.dirichlet_alpha = cfg.dirichlet_alpha;
      if (diagnostics.length) baseArgs.diagnostics = diagnostics;
      if (hasDiagnostic("loo")) baseArgs.loo_enabled = true;
      if (parts.control_graph_mode) baseArgs.control_graph_mode = parts.control_graph_mode;

      const doc = {
        description:
          "조립 스냅샷 (repo에는 configs/vision/smoke/… JSON으로 저장 후 --config 실행)",
        args: compactObject(baseArgs),
      };

      const sweep = cfg.sweep;
      if (cfg.track === "vision" && sweep) {
        const parts = sweep.partitions || (cfg.partition ? [cfg.partition] : []);
        if (parts.length) doc.args.partition = parts[0];
        doc.sweep = {
          partitions: parts,
          dirichlet_alphas: sweep.dirichlet_alphas || [cfg.dirichlet_alpha],
          variants: sweep.variants || [],
          compare_assemblies: sweep.compare_assemblies || [],
        };
        doc.args.variants = doc.sweep.variants;
        if (doc.sweep.compare_assemblies.length) {
          doc.args.compare_assemblies = doc.sweep.compare_assemblies;
        }
        if (parts.includes("dirichlet")) {
          doc.args.dirichlet_alphas = doc.sweep.dirichlet_alphas;
          doc.args.dirichlet_alpha = doc.sweep.dirichlet_alphas[0];
        }
      }

      if (extensions.length) doc.extensions = extensions;
      return doc;
    }

    function buildCli(cfg, parts, configPath) {
      const isCora = cfg.track === "cora";
      let entry = "";
      if (cfg.track === "vision") entry = "python run_vision_suite.py";
      else if (isCora) entry = "python run_graph_ablation.py";
      else entry = "python run_vision_experiment.py";

      const useConfigFirst = cfg.track === "vision" || isCora;
      const lines = [];

      if (useConfigFirst) {
        lines.push(`${entry} \\`);
        lines.push(`  --config ${configPath}`);
        if (hasDiagnostic("preflight")) {
          lines.push("");
          lines.push("# 실행 전 점검 (diagnostic · preflight)");
          lines.push("python scripts/checks/diagnostic_suite_preflight.py");
        }
        if (hasDiagnostic("evidence")) {
          lines.push("");
          lines.push("# 결과 계약/evidence bundle 점검");
          lines.push("python scripts/checks/result_evidence_bundle.py <result.json> --kind single-run");
        }
        lines.push("");
        lines.push("# 동일 설정을 플래그로만 실행할 때 (발표 비교용):");
      }

      const flagLines = [];
      if (!useConfigFirst) flagLines.push(entry);
      if (cfg.dataset && !isCora) flagLines.push(`--dataset ${cfg.dataset}`);
      if (!isCora) appendPartitionCliFlags(cfg, flagLines);
      flagLines.push(`--num-clients ${cfg.num_clients}`, `--rounds ${cfg.rounds}`);
      flagLines.push(`--method ${getMethodPart()?.value || "ours"}`);
      if (parts.graph_source) flagLines.push(`--graph-source ${parts.graph_source}`);
      if (parts.graph_mode) flagLines.push(`--graph-mode ${parts.graph_mode}`);
      if (parts.aggregation_target) flagLines.push(`--aggregation-target ${parts.aggregation_target}`);
      if (parts.correction_family) flagLines.push(`--correction-family ${parts.correction_family}`);
      if (parts.control_graph_mode) flagLines.push(`--control-graph-mode ${parts.control_graph_mode}`);
      if (hasDiagnostic("loo")) flagLines.push("--loo-enabled true");
      const plugin = envColumns[0]?.lifeStack.find((b) => b.value === "graph_plugin");
      if (plugin?.sub) flagLines.push(`--graph-plugin ${plugin.sub}`);

      for (let i = 0; i < flagLines.length; i++) {
        const sep = i === 0 ? " " : " \\\n  ";
        lines.push(`${sep}${flagLines[i]}`);
      }
      return lines.join("");
    }

    function renderRunArtifacts() {
      const { ready, missing, cfg } = runReadiness();
      const statusEl = document.getElementById("run-status");
      const hintEl = document.getElementById("run-output-hint");
      if (statusEl) {
        statusEl.textContent = ready ? "실행 가능" : "부품 부족";
        statusEl.className = "run-status " + (ready ? "ok" : "wait");
      }
      if (hintEl) {
        const selection = resolveConfigSelection(cfg);
        const trackLabel = cfg.track === "cora" ? "Cora" : "Vision";
        const modeLabel = cfg.track === "cora"
          ? "graph ablation"
          : selection.existing
            ? "기존 config JSON"
            : `열 ${cfg.column_count || getTargetColumnCount()}개`;
        hintEl.textContent = ready
          ? `${trackLabel} · ${modeLabel} · repo --config JSON`
          : `필요: ${missing.join(" · ")}`;
      }
      return ready;
    }

    function installContextFromStep() {
      const step = currentAssemblyStep();
      const ctx = {};
      if (step.phase === "track") {
        ctx.slotId = "track";
        ctx.slotEl = document.querySelector('#env-base-row .env-h-slot[data-env-slot="track"]');
        return ctx;
      }
      if (step.phase === "dataset") {
        ctx.slotId = "dataset";
        ctx.slotEl = document.querySelector('#env-base-row .env-h-slot[data-env-slot="dataset"]');
        return ctx;
      }
      if (step.phase === "config_name") {
        ctx.slotId = "config_name";
        ctx.slotEl = document.querySelector('#env-base-row .env-h-slot[data-env-slot="config_name"]');
        return ctx;
      }
      if (!step.colId) return ctx;
      ctx.colId = step.colId;
      if (step.phase === "partition") {
        ctx.slotId = "partition";
        ctx.slotEl = document.querySelector(
          `.env-h-slot[data-env-column="${step.colId}"][data-env-slot="partition"]`
        );
      } else if (step.phase === "column_stem") {
        ctx.slotId = "column_stem";
        ctx.slotEl = document.querySelector(
          `.env-h-slot[data-env-column="${step.colId}"][data-env-slot="column_stem"]`
        );
      } else if (step.phase === "graph") {
        ctx.slotEl = document.querySelector(`.column-life-stack[data-column-id="${step.colId}"]`);
      }
      return ctx;
    }

    function lifecycleComplete() {
      const p = columnLifecycleParts(envColumns[0]);
      return COMPARE_REQUIRED_AXES.every((k) => p[k]);
    }

    function getDemoRunSubmission() {
      const readiness = runReadiness();
      const preview = window.__graphflPreview || {};
      const ready = !!(readiness.ready && preview.doc);
      let rowBundle = null;
      if (ready) {
        rowBundle = buildDemoRows();
      }
      const signature = JSON.stringify({
        cfg: readiness.cfg,
        command: preview.cmd || "",
        configPath: preview.configPath || "",
        configType: preview.configType || "",
      });
      return {
        ready,
        readiness,
        cfg: readiness.cfg,
        parts: readiness.parts,
        preview,
        doc: preview.doc || null,
        command: preview.cmd || "",
        configPath: preview.configPath || "",
        configType: preview.configType || "",
        rowBundle,
        signature,
      };
    }

    function canRunExperiment() {
      return getDemoRunSubmission().ready;
    }

    function updateRunNote() {
      const note = document.getElementById("run-note");
      if (!note) return;
      const { missing } = runReadiness();
      if (canRunExperiment()) {
        note.textContent = "Mock API 제출 가능 · 실제 CLI는 실행하지 않음.";
        return;
      }
      const need = [];
      if (!resolvedEnvSlotValue(envSlots.config_name)) need.push("config JSON");
      else if (!configInputReadyForAssembly()) need.push("올바른 config JSON");
      else if (!envSlots.track) need.push("트랙");
      else if (!resolvedEnvSlotValue(envSlots.dataset)) need.push("데이터");
      else if (currentTrack() === "vision" && !visionPartitionReady(buildEnvConfig())) need.push("열 파티션");
      note.textContent = need.length
        ? `비어 있는 칸: ${need.join(" · ")}`
        : missing.length
          ? `Mock 제출 전 필요: ${missing.join(" · ")}`
          : "주황 환경 또는 Graph-FL 부품을 추가하세요.";
    }

    function caseModeLabel(cfg) {
      const n = cfg.column_count || getTargetColumnCount();
      if (cfg.track === "cora") return "Cora graph ablation";
      return n <= 1 ? "열 1개 (= 단일 run)" : `열 ${n}개 (다변량)`;
    }

    function fmtPct(x) {
      return (x * 100).toFixed(1) + "%";
    }

    function buildDemoRows() {
      const cfg = buildEnvConfig();
      const p0 = columnLifecycleParts(envColumns[0]);
      const ds = cfg.dataset || "(데이터 없음)";
      const alphaBit = usesDirichletAlpha(cfg.partition) ? ` · Dirichletα=${cfg.dirichlet_alpha}` : "";
      const envPrefix = `${cfg.track || "?"} · ${ds}${alphaBit} · N=${cfg.num_clients} · R=${cfg.rounds}`;
      const previewType = window.__graphflPreview?.configType || "";

      const interpretations = {
        ours_main:
          "real graph가 matched control·graph-free control보다 낫고 alignment·DI/N_eff가 같이 움직이면 graph-specific effect가 남았다고 해석할 수 있음.",
        fedavg_base:
          "베이스라인: 그래프 없는 기준선. 이후 real/control/graph-free 행을 같은 환경에서 비교해야 attribution이 가능함.",
        alpha_low:
          "Dirichlet α↓ → Non-IID↑ → N_eff↓. 관측 smoothness(H_G)도 바뀔 수 있으나, 그건 진단값이지 이 α 입력과 동일 개념이 아님.",
        alpha_high:
          "Dirichlet α↑ → IID에 가까움. graph 이득이 작아지면 topology 기여가 약하다는 신호.",
        shuffled:
          "real≈shuffled·gap≈0이면 관측된 이득이 topology가 아니라 dominance/스케일 confounder일 가능성.",
        suite_note:
          "suite 한 번 실행 → run_id별 행과 evidence bundle이 함께 남음. 정확도, gap, mechanism metric을 한 결과 묶음에서 비교.",
      };

      if (previewType === "existing") {
        const path = window.__graphflPreview?.configPath || cfg.config_name || "configs/...";
        const isCora = cfg.track === "cora";
        const rows = isCora
          ? [
              {
                id: "fedavg",
                setting: `기존 config 직접 실행 · ${path} · fedavg`,
                acc: 0.684,
                alignment: 0.38,
                filter_gain: 0,
                real_control_gap: 0,
                interp: "기존 Cora graph ablation JSON의 기준선 행. 화면 조립값은 결과에 섞지 않음.",
              },
              {
                id: "ours_knn",
                setting: `기존 config 직접 실행 · ${path} · ours_knn`,
                acc: 0.731,
                alignment: 0.57,
                filter_gain: 0.09,
                real_control_gap: 0.047,
                interp: "기존 JSON에 정의된 graph ablation variant를 Mock DB 결과로 저장한 예시.",
              },
              {
                id: "ours_random",
                setting: `기존 config 직접 실행 · ${path} · ours_random`,
                acc: 0.697,
                alignment: 0.41,
                filter_gain: 0.02,
                real_control_gap: 0.013,
                interp: "matched random control까지 같은 기존 config run에서 비교된다는 흐름을 보여줌.",
              },
            ]
          : [
              {
                id: "fedavg",
                setting: `기존 config 직접 실행 · ${path} · fedavg`,
                acc: 0.681,
                alignment: 0.42,
                filter_gain: 0,
                real_control_gap: 0,
                interp: "repo에 이미 있는 Vision config를 그대로 --config로 실행한 기준선 예시.",
              },
              {
                id: "ours_graph",
                setting: `기존 config 직접 실행 · ${path} · graph variant`,
                acc: 0.748,
                alignment: 0.59,
                filter_gain: 0.08,
                real_control_gap: 0.041,
                interp: "config 내부 variants/sweep 정의를 실행한 뒤 Mock DB latest completed record로 렌더링.",
              },
              {
                id: "matched_control",
                setting: `기존 config 직접 실행 · ${path} · matched control`,
                acc: 0.711,
                alignment: 0.47,
                filter_gain: 0.02,
                real_control_gap: 0.008,
                interp: "기존 JSON이 가진 대조군 결과까지 같은 결과 묶음에서 비교한다는 발표용 행.",
              },
            ];
        return {
          rows,
          isSuite: true,
          cfg,
          caseMode: "기존 config JSON 직접 실행",
        };
      }

      if (cfg.track === "cora") {
        return {
          rows: [
            {
              id: "fedavg",
              setting: `${envPrefix} · graph_ablation · fedavg`,
              acc: 0.684,
              alignment: 0.38,
              filter_gain: 0,
              real_control_gap: 0,
              interp: "Cora GCN 기준선. graph ablation에서는 ours_* 변형의 delta 기준점으로 저장됨.",
            },
            {
              id: "ours_knn",
              setting: `${envPrefix} · graph_ablation · ours_knn · k=${cfg.knn_k}`,
              acc: 0.731,
              alignment: 0.57,
              filter_gain: 0.09,
              real_control_gap: 0.047,
              interp: "KNN graph 변형. FedAvg 대비 delta와 spectral metric이 함께 저장되어 graph-specific 효과를 설명함.",
            },
            {
              id: "ours_random",
              setting: `${envPrefix} · graph_ablation · ours_random · matched edges`,
              acc: 0.697,
              alignment: 0.41,
              filter_gain: 0.02,
              real_control_gap: 0.013,
              interp: "matched random control. real graph와 가까우면 topology 효과가 약하고, 차이가 크면 graph 구성의 근거가 강해짐.",
            },
          ],
          isSuite: true,
          cfg,
          caseMode: caseModeLabel(cfg),
        };
      }

      const rows = [];
      const isSweep = cfg.track === "vision" && previewType === "suite" && !!cfg.sweep;

      if (isSweep && cfg.sweep) {
        const s = cfg.sweep;
        const cases = s.comparison_cases?.length
          ? s.comparison_cases
          : (s.variants || ["fedavg"]).map((v, i) => ({
              index: i + 1,
              run_id: `run_${i + 1}`,
              label: v,
              variant_token: v,
              partition: cfg.partition || "dirichlet",
              dirichlet_alpha: cfg.dirichlet_alpha,
            }));
        cases.forEach((cmp, vi) => {
          const label = cmp.label || cmp.variant_token || `case_${vi + 1}`;
          const isFed =
            String(cmp.variant_token || "").toLowerCase() === "fedavg" ||
            String(label || "").toLowerCase() === "fedavg";
          const a = cmp.dirichlet_alpha;
          const p = cmp.partition || cfg.partition || "dirichlet";
          const accOurs = 0.72 + (a != null ? a * 0.8 : 0.1) - vi * 0.02;
          rows.push({
            id: cmp.run_id || `suite_${vi + 1}`,
            setting: `${envPrefix} · Run ${cmp.index || vi + 1} · ${p}${a != null ? ` · alpha=${a}` : ""} · ${label}`,
            acc: isFed ? 0.68 : accOurs,
            alignment: isFed ? 0.42 : 0.55 + (a || 0.03) * 1.2,
            filter_gain: isFed ? 0 : 0.08 + (a || 0.03) * 0.15,
            real_control_gap: isFed ? 0 : 0.04,
            interp: isFed
              ? interpretations.fedavg_base
              : a != null && a <= 0.03
                ? interpretations.alpha_low
                : interpretations.alpha_high,
          });
        });
      } else {
        const nCols = getTargetColumnCount();
        envColumns.slice(0, nCols).forEach((col, ci) => {
          const method = getMethodPart(col)?.value || "ours";
          const part = currentTrack() === "vision" ? resolvedEnvSlotValue(col.partition) : "";
          const rowAlphaBit = usesDirichletAlpha(part) ? ` · Dirichletα=${col.dirichletAlpha}` : "";
          const rowEnvPrefix = `${cfg.track || "?"} · ${ds}${rowAlphaBit} · N=${cfg.num_clients} · R=${cfg.rounds}`;
          const colPart = part ? ` · ${part}` : "";
          const colLifeOk = col.caseMode === "baseline" || columnAssemblyComplete(col);
          const partialNote = colLifeOk ? "" : " (lifecycle 미완성 — 발표용 단일 행만 표시)";
          const label =
            col.caseMode === "baseline"
              ? columnVariantToken(col) || method
              : colLifeOk
                ? assemblyLabel(columnAssembly(col))
                : "lifecycle …";
          const runBit = nCols > 1 ? ` · Run ${ci + 1}` : "";
          const baseSetting = `${rowEnvPrefix}${runBit}${colPart} · ${method}`;
          const settingLabel =
            col.caseMode === "baseline" && String(label).toLowerCase() === String(method).toLowerCase()
              ? baseSetting
              : `${baseSetting} · ${label}`;
          rows.push({
            id: `run_${ci + 1}`,
            setting: settingLabel,
            acc: method === "ours" ? 0.76 - ci * 0.02 : 0.69,
            alignment: method === "ours" ? 0.62 : 0.44,
            filter_gain: method === "ours" ? 0.11 : 0,
            real_control_gap: method === "ours" ? 0.05 : 0,
            interp:
              (method === "ours" ? interpretations.ours_main : interpretations.fedavg_base) +
              partialNote +
              (nCols > 1 ? ` · 열 ${ci + 1}` : " · 열을 늘리면 run이 늘어남."),
          });
        });
        if (
          rows.length === 1 &&
          p0.correction_family === "control_graph" &&
          p0.control_graph_mode === "shuffled"
        ) {
          const method0 = getMethodPart(envColumns[0])?.value || "ours";
          rows.push({
            id: "run_control",
            setting: `${envPrefix} · ${method0} · control_graph · shuffled`,
            acc: 0.74,
            alignment: 0.48,
            filter_gain: 0.02,
            real_control_gap: 0.005,
            interp: interpretations.shuffled,
          });
        }
      }

      if (hasDiagnostic("evidence")) {
        rows.forEach((r) => {
          r.interp += " Evidence 번들로 재현·제출 가능.";
        });
      }

      return {
        rows,
        isSuite: isSweep,
        cfg,
        caseMode: previewType === "batch" ? "Mock batch · per-run single configs" : caseModeLabel(cfg),
      };
    }

    function renderResultsTable(record) {
      const built = record && Array.isArray(record.rows)
        ? {
            rows: record.rows,
            caseMode: record.caseMode || record.case_mode || "Mock DB 조회",
          }
        : buildDemoRows();
      const { rows, caseMode } = built;
      const tbody = document.getElementById("results-body");
      if (!tbody) return;
      tbody.textContent = "";

      const addCell = (tr, text, className) => {
        const td = document.createElement("td");
        if (className) td.className = className;
        td.textContent = text == null ? "" : String(text);
        tr.appendChild(td);
      };
      const fixed = (value, digits) => {
        const n = Number(value);
        return Number.isFinite(n) ? n.toFixed(digits) : "";
      };

      rows.forEach((r) => {
        const tr = document.createElement("tr");
        addCell(tr, r.id, "case-id");
        addCell(tr, r.setting);
        addCell(tr, Number.isFinite(Number(r.acc)) ? fmtPct(Number(r.acc)) : "", "metric-val");
        addCell(tr, fixed(r.alignment, 2), "metric-val");
        addCell(tr, fixed(r.filter_gain, 2), "metric-val");
        addCell(tr, fixed(r.real_control_gap, 3), "metric-val");
        addCell(
          tr,
          r.interpretation_type ? `[${r.interpretation_type}] ${r.interp || ""}` : r.interp,
          "interp"
        );
        tbody.appendChild(tr);
      });

      const cap = document.getElementById("results-caption");
      if (cap) {
        cap.textContent = record
          ? `[${caseMode}] · ${rows.length}행 · Mock DB latest completed run에서 조회.`
          : `[${caseMode}] · ${rows.length}행 (데모). 해석 열은 발표용 예시 문장.`;
      }

      const wrap = document.getElementById("results-wrap");
      if (wrap) {
        wrap.dataset.resultSignature = record?.signature || "";
        wrap.classList.remove("is-stale");
        wrap.classList.add("show");
      }
      const staleNote = document.getElementById("results-stale-note");
      if (staleNote) staleNote.hidden = true;
    }

    function renderImplList() {
      const list = document.getElementById("impl-list");
      if (!list) return;
      list.textContent = "";
      IMPL.forEach((text) => {
        const li = document.createElement("li");
        li.textContent = text;
        list.appendChild(li);
      });
    }

    function runExperimentDemo() {
      const submission = getDemoRunSubmission();
      if (!submission.ready) {
        updateRunNote();
        return;
      }
      if (window.GraphFLMockSystem?.start) {
        window.GraphFLMockSystem.start(submission);
        return;
      }
      btnRun.disabled = true;
      btnRun.textContent = "Mock 실행 중…";
      document.getElementById("run-note").textContent = "Mock: suite 집계 → 표 생성 (실제 CLI 미실행)";
      setTimeout(() => {
        renderResultsTable({
          rows: submission.rowBundle.rows,
          caseMode: submission.rowBundle.caseMode,
        });
        btnRun.disabled = false;
        btnRun.textContent = "▶ 실험 시작 (데모)";
        document.getElementById("run-note").textContent =
          "완료. Mock 결과표는 청중에게 run 의미를 설명할 때 사용합니다.";
      }, 650);
    }

    function setChecklistItemText(li, text) {
      li.textContent = "";
      const dot = document.createElement("span");
      dot.className = "dot";
      li.appendChild(dot);
      li.appendChild(document.createTextNode(text));
    }

    function syncUI() {
      applyTrackContext();
      updatePartitionAlphaUI();
      updateConfigDirectInputState();
      syncDiagnosticOptionsFromControls();
      updateDiagnosticOptionsUI();
      const cfg = buildEnvConfig();
      const configSelection = resolveConfigSelection(cfg);
      const directExistingConfig = existingConfigLocked();
      [numAlpha, numClients, numRounds, document.getElementById("input-col-count")].forEach((el) => {
        if (el) el.disabled = directExistingConfig;
      });
      [document.getElementById("btn-col-minus"), document.getElementById("btn-col-plus")].forEach((el) => {
        if (el) el.disabled = directExistingConfig;
      });
      const has = envFilled();
      workspace.classList.toggle("has-content", has);

      const envChk = document.getElementById("env-chk");
      envChk.innerHTML = "";
      const slotLabels = {
        track: "트랙",
        dataset: "데이터셋",
        config_name: "config JSON",
        partition: "파티션",
        nums: "N · R (수치)",
        cases: "비교 열",
      };
      {
        const li = document.createElement("li");
        const showA =
          currentTrack() === "vision" &&
          envColumns.some((c) => usesDirichletAlpha(resolvedEnvSlotValue(c.partition)));
        li.className = showA ? "ok" : "";
        setChecklistItemText(li, "Dirichlet α (Non-IID)");
        envChk.appendChild(li);
      }
      ENV_SLOT_IDS.forEach((id) => {
        const slot = envSlots[id];
        const filled = Array.isArray(slot) ? slot.length > 0 : !!slot;
        const unlocked = envSlotUnlocked(id);
        const existingCovered = directExistingConfig && (id === "track" || id === "dataset");
        const required = id === "config_name" || (!directExistingConfig && (id === "track" || id === "dataset"));
        const ok = existingCovered || (required && filled && !envSlotPending(slot) && unlocked);
        const li = document.createElement("li");
        if (existingCovered) li.className = "ok";
        else if (!unlocked && required) li.className = "";
        else if (ok) li.className = "ok";
        else if (filled) li.className = "partial";
        else li.className = "";
        const pending = filled && envSlotPending(slot) ? " (이름 입력)" : "";
        const lock = !existingCovered && !unlocked && required ? " · 잠김" : "";
        const label = existingCovered
          ? id === "track"
            ? `트랙 (${cfg.track === "cora" ? "Cora" : "Vision"} · 기존 JSON 경로)`
            : "데이터셋 (기존 JSON 내부값)"
          : slotLabels[id];
        setChecklistItemText(li, `${label}${pending}${lock}`);
        envChk.appendChild(li);
      });
      {
        const n = getTargetColumnCount();
        const isCoraTrack = currentTrack() === "cora";
        const allCols =
          directExistingConfig || isCoraTrack || envColumns.slice(0, n).every((c) => columnCaseGraphReady(c));
        const li = document.createElement("li");
        li.className = allCols && (baseRowReady() || directExistingConfig) ? "ok" : "";
        setChecklistItemText(
          li,
          directExistingConfig
            ? "비교 열 (기존 JSON 내부값)"
            : isCoraTrack
              ? "Cora graph ablation"
              : "비교 열 (baseline 또는 Graph-FL)"
        );
        envChk.appendChild(li);
      }
      const sweep = cfg.sweep;
      if (currentTrack() === "vision" && sweep) {
        const li = document.createElement("li");
        let sweepOk = !!(sweep.variants?.length && sweep.partitions?.length);
        if (sweep.partitions?.includes("dirichlet")) {
          sweepOk = sweepOk && !!sweep.dirichlet_alphas?.length;
        }
        li.className = sweepOk ? "ok" : "partial";
        const ncol = cfg.column_count || getTargetColumnCount();
        setChecklistItemText(li, `비교 열 ${ncol}개 · partition∈${(sweep.partitions || []).join(",")}`);
        envChk.appendChild(li);
      }
      {
        const li = document.createElement("li");
        li.className = directExistingConfig || baseRowReady() ? "ok" : "";
        setChecklistItemText(li, directExistingConfig ? "N · R (기존 JSON 내부값)" : slotLabels.nums);
        envChk.appendChild(li);
      }
      {
        const li = document.createElement("li");
        const nDiag = (envSlots.diagnostics || []).length;
        li.className = nDiag ? "ok" : "";
        setChecklistItemText(li, `실행 옵션${nDiag ? ` (${nDiag}개)` : " (기본값)"}`);
        envChk.appendChild(li);
      }

      document.getElementById("env-out").textContent = JSON.stringify(cfg, null, 2);

      const req = [
        ["client", "graph_source"],
        ["topology", "graph_mode"],
        ["aggregation", "aggregation_target"],
      ];
      const lifeChk = document.getElementById("life-chk");
      lifeChk.innerHTML = "";
      req.forEach(([k, label]) => {
        const li = document.createElement("li");
        const ok = envColumns.slice(0, getTargetColumnCount()).every((col) => {
          if (col.caseMode === "baseline") return true;
          const kind =
            k === "client" ? "client" : k === "topology" ? "topology" : "aggregation";
          return !!getColumnLife(col, kind) || !!col.preset;
        });
        li.className = ok ? "ok" : "";
        setChecklistItemText(li, `${label} (열당)`);
        lifeChk.appendChild(li);
      });

      const partsDoc = {
        columns: envColumns.slice(0, getTargetColumnCount()).map((col, i) => ({
          index: i + 1,
          case_mode: col.caseMode,
          partition: resolvedEnvSlotValue(col.partition),
          dirichlet_alpha: col.dirichletAlpha,
          preset: col.preset ? columnVariantToken(col) : null,
          ...columnLifecycleParts(col),
        })),
      };
      document.getElementById("parts-out").textContent = JSON.stringify(partsDoc, null, 2);

      const runReady = renderPreview();
      const complete = runReady;

      const envAssembly = document.querySelector(".env-chain-assembly");
      const anyLife = envColumns.some((c) => c.lifeStack.length > 0);
      if (envAssembly) envAssembly.classList.toggle("has-life-blocks", anyLife);
      relayoutLifeStackZ();

      const submission = getDemoRunSubmission();
      btnRun.disabled = !submission.ready;
      updateRunNote();
      window.dispatchEvent(new CustomEvent("graphfl:assembly-changed", { detail: submission }));
      if (!submission.ready) {
        const wrap = document.getElementById("results-wrap");
        wrap?.classList.remove("show", "is-stale");
        const staleNote = document.getElementById("results-stale-note");
        if (staleNote) staleNote.hidden = true;
      }

      const extensions = envColumns
        .flatMap((c) => c.lifeStack)
        .filter((b) => b.kind === "extend" || b.kind === "custom");
      const box = document.getElementById("custom-box");
      if (extensions.length) {
        box.classList.add("show");
        box.textContent = extensions
          .map((b) => extendSnippet(b.value, b.customName))
          .join("\n\n");
      } else {
        box.classList.remove("show");
      }

      renderImplList();
    }

    function setupPalette() {
      document.querySelectorAll(".palette .scratch-block[draggable]").forEach((block) => {
        block.querySelectorAll(".extend-input, .env-custom-block .extend-input").forEach((input) => {
          const validatePaletteInput = () => {
            setInputValidation(
              input,
              validateCustomName(block.dataset.value, block.dataset.envSlot || null, input.value.trim())
            );
          };
          validatePaletteInput();
          input.addEventListener("mousedown", (e) => e.stopPropagation());
          input.addEventListener("dragstart", (e) => e.preventDefault());
          input.addEventListener("click", (e) => e.stopPropagation());
          input.addEventListener("input", validatePaletteInput);
          input.addEventListener("keydown", (e) => {
            if (e.key !== "Enter") return;
            e.preventDefault();
            e.stopPropagation();
            const data = payload(block);
            const ok = installBlock(data, installContextFromStep());
            if (!ok) {
              applyEnvChainGating();
              syncUI();
            }
          });
        });
        block.addEventListener("dragstart", (e) => {
          if (e.target.classList.contains("extend-input")) {
            e.preventDefault();
            return;
          }
          e.dataTransfer.setData("application/json", JSON.stringify(payload(block)));
          e.dataTransfer.effectAllowed = "copy";
        });
        block.addEventListener("click", (e) => {
          if (e.target.classList.contains("extend-input")) return;
          const data = payload(block);
          const ok = installBlock(data, installContextFromStep());
          if (!ok) {
            applyEnvChainGating();
            syncUI();
          }
        });
      });
    }

    function setupWorkspaceDropDelegation() {
      const root = document.getElementById("env-panel") || document.getElementById("env-unified");
      if (!root || root.dataset.dropDelegate === "1") return;
      root.dataset.dropDelegate = "1";
      root.addEventListener("dragover", (e) => {
        const slot = e.target.closest(".env-h-slot:not(.is-locked)");
        const stack = e.target.closest(".column-life-stack");
        if (!slot && !stack) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = "copy";
        if (slot) slot.classList.add("drag-over");
        if (stack) stack.classList.add("drag-over");
      });
      root.addEventListener("dragleave", (e) => {
        const slot = e.target.closest(".env-h-slot");
        const stack = e.target.closest(".column-life-stack");
        if (slot && (!e.relatedTarget || !slot.contains(e.relatedTarget))) {
          slot.classList.remove("drag-over");
        }
        if (stack && (!e.relatedTarget || !stack.contains(e.relatedTarget))) {
          stack.classList.remove("drag-over");
        }
      });
      root.addEventListener("drop", (e) => {
        const slot = e.target.closest(".env-h-slot");
        const stack = e.target.closest(".column-life-stack");
        if (slot) {
          e.preventDefault();
          e.stopPropagation();
          slot.classList.remove("drag-over");
          handleWorkspaceDrop(e, slot);
          return;
        }
        if (stack) {
          e.preventDefault();
          e.stopPropagation();
          stack.classList.remove("drag-over");
          handleColumnStackDrop(e, stack);
        }
      });
    }

    function setupEnvSlotDrops() {
      setupWorkspaceDropDelegation();
    }

    function clearEnv() {
      envSlots.track = null;
      envSlots.dataset = null;
      envSlots.diagnostics = [];
      resetEnvColumns();
      const chain = document.getElementById("env-cases-chain");
      if (chain) chain.innerHTML = "";
      const zone = document.getElementById("env-cases-zone");
      if (zone) zone.classList.add("is-dimmed");
      renderEnvSlots();
      applySweepStackUI();
      applyEnvChainGating();
    }

    function setupPreviewResize() {
      const col = document.getElementById("preview-col");
      const handle = document.getElementById("preview-resizer");
      const main = document.getElementById("main-layout");
      if (!col || !handle || !main) return;

      let dragging = false;

      const setWidth = (px) => {
        const max = Math.min(main.getBoundingClientRect().width * 0.65, window.innerWidth * 0.7);
        const w = Math.round(Math.max(260, Math.min(px, max)));
        col.style.setProperty("--preview-w", w + "px");
        try {
          localStorage.setItem("graphfl-demo-preview-w", String(w));
        } catch (_) {}
      };

      let saved = NaN;
      try {
        saved = parseInt(localStorage.getItem("graphfl-demo-preview-w") || "", 10);
      } catch (_) {}
      if (saved >= 260) setWidth(saved);

      handle.addEventListener("mousedown", (e) => {
        dragging = true;
        handle.classList.add("active");
        document.body.classList.add("resizing-preview");
        e.preventDefault();
      });
      window.addEventListener("mousemove", (e) => {
        if (!dragging) return;
        const rect = main.getBoundingClientRect();
        setWidth(rect.right - e.clientX);
      });
      const stop = () => {
        if (!dragging) return;
        dragging = false;
        handle.classList.remove("active");
        document.body.classList.remove("resizing-preview");
      };
      window.addEventListener("mouseup", stop);
      window.addEventListener("blur", stop);
    }

    function setupFoldSections() {
      document.querySelectorAll(".fold-section").forEach((section) => {
        const head = section.querySelector(":scope > .fold-head");
        if (!head || head.dataset.foldBound === "1") return;
        head.dataset.foldBound = "1";
        const open = section.dataset.foldDefault !== "collapsed";
        section.classList.toggle("is-collapsed", !open);
        head.setAttribute("aria-expanded", open ? "true" : "false");
        head.addEventListener("click", () => {
          const collapsed = section.classList.toggle("is-collapsed");
          head.setAttribute("aria-expanded", collapsed ? "false" : "true");
        });
      });
    }

    function setFoldHeadLabel(head, label) {
      if (!head || !label) return;
      const badges = Array.from(head.children).filter((child) => child.classList.contains("cap-badge"));
      Array.from(head.childNodes).forEach((node) => {
        if (node.nodeType === Node.TEXT_NODE) node.remove();
      });
      const chevron = head.querySelector(".fold-chevron");
      if (chevron) {
        chevron.insertAdjacentText("afterend", ` ${label} `);
      } else {
        head.insertBefore(document.createTextNode(`${label} `), head.firstChild);
      }
      badges.forEach((badge) => head.appendChild(badge));
    }

    function sectionCapability(section) {
      const head = section?.querySelector(":scope > .fold-head");
      if (!section || !head) return null;
      if (section.id === "fold-palette-preset") return "preset";
      if (section.id === "fold-palette-diagnostic" || head.classList.contains("diagnostic")) return "fixed";
      return section.querySelector(":scope > .fold-body .env-custom-block, :scope > .fold-body .name-extend-block")
        ? "custom"
        : "fixed";
    }

    function addCapabilityBadge(head, capability) {
      if (!head || !capability) return;
      let badge = Array.from(head.children).find((child) => child.classList.contains("cap-badge"));
      if (!badge) {
        badge = document.createElement("span");
        head.appendChild(badge);
      }
      badge.className = `cap-badge ${capability}`;
      badge.textContent = capability;
      badge.title =
        capability === "custom"
          ? "custom value can be entered"
          : capability === "preset"
            ? "prebuilt shortcut"
            : "fixed choices only";
    }

    function setupPaletteCapabilityCues() {
      setFoldHeadLabel(document.querySelector("#fold-palette-diagnostic > .fold-head"), "run checks");
      setFoldHeadLabel(document.querySelector("#fold-palette-life .cat-sub > .fold-head.diagnostic"), "evidence addon");
      document.getElementById("fold-palette-diagnostic")?.classList.add("palette-diagnostic-section");
      document
        .querySelector("#fold-palette-life .cat-sub > .fold-head.diagnostic")
        ?.closest(".cat-sub")
        ?.classList.add("palette-diagnostic-section");
      document
        .querySelectorAll(".palette .cat-sub.fold-section, #fold-palette-baseline, #fold-palette-preset")
        .forEach((section) => {
          const head = section.querySelector(":scope > .fold-head");
          addCapabilityBadge(head, sectionCapability(section));
        });
    }

    function setupPreviewActions() {
      const downloadName = (suffix = "") => {
        const preview = window.__graphflPreview || {};
        const name = String(preview.configPath || buildEnvConfig().config_name || "").trim();
        if (!name) return "";
        const leaf = name.split(/[\\/]/).filter(Boolean).pop() || name;
        const base = leaf.endsWith(".json") ? leaf.slice(0, -5) : leaf;
        return suffix ? `${base}-${suffix}.json` : `${base}.json`;
      };
      document.getElementById("btn-dl-single")?.addEventListener("click", () => {
        const doc = window.__graphflPreview?.doc || null;
        const filename = downloadName();
        if (doc && filename) downloadJson(filename, doc);
      });
      document.getElementById("btn-dl-suite")?.addEventListener("click", () => {
        const configs = buildPerColumnSingleConfigs().map((config, i) => ({
          config_path: columnConfigPath(window.__graphflPreview?.configPath || "", i),
          command: `python run_vision_experiment.py --config ${columnConfigPath(window.__graphflPreview?.configPath || "", i)}`,
          config,
        }));
        const filename = downloadName("per-run-list");
        if (configs.length && filename) downloadJson(filename, configs);
      });
      document.getElementById("btn-copy-cmd")?.addEventListener("click", () => {
        const cmd = document.getElementById("run-command-out")?.textContent || "";
        if (cmd && navigator.clipboard?.writeText) {
          navigator.clipboard.writeText(cmd);
        }
      });
    }

    function setupConfigDirectInput() {
      const input = document.getElementById("config-direct-input");
      const existingBtn = document.getElementById("config-mode-existing");
      const generateBtn = document.getElementById("config-mode-generate");
      input?.addEventListener("input", () => {
        setDirectConfigName(input.value);
      });
      input?.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          setDirectConfigName(input.value);
          input.blur();
        }
      });
      input?.addEventListener("blur", () => {
        setDirectConfigName(input.value);
      });
      existingBtn?.addEventListener("click", () => setConfigUseMode("existing"));
      generateBtn?.addEventListener("click", () => setConfigUseMode("generate"));
      document.querySelectorAll("[data-config-pick]").forEach((btn) => {
        btn.addEventListener("click", () => {
          chooseConfigQuickPick(btn.dataset.configPick || "", btn.dataset.configMode || "existing");
        });
      });
    }

    setupPalette();
    setupFoldSections();
    setupPaletteCapabilityCues();
    setupConfigDirectInput();
    setupDiagnosticOptions();
    setupAlphaNumControls();
    setupWorkspaceDropDelegation();
    setupColumnCountControls();
    setupPreviewResize();
    setupPreviewActions();
    relayoutLifeStackZ();
    [numAlpha, numClients, numRounds].forEach((el) => {
      el?.addEventListener("input", syncUI);
    });
    btnRun?.addEventListener("click", runExperimentDemo);
    renderImplList();
    document
      .querySelectorAll('#env-base-row .env-h-slot[data-layout="stack"]')
      .forEach((el) => ensureStackSocket(el));
    window.GraphFLDemo = {
      getSubmission: getDemoRunSubmission,
      renderResultsTable,
      buildDemoRows,
      runReadiness,
      updateRunNote,
    };
    syncUI();
