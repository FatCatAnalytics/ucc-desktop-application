import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { EngineClient, native } from "../api";
import type { Bucket, CsvSource, Health, LogLine, ModelsStatus, RunSnapshot, SourcePreview, SourceSpec, StageState, Summary } from "../types";
import { Bars, DataTable, LoadFailed, OutputsTab, ReviewTab, describeError } from "./tables";

/**
 * The pipeline page. One horizontal flow:
 *
 *   Ingest ─► Direct match ─► Fuzzy match ─► Categorise ┬► Fuzzy · high probability (auto)
 *                                                       ├► Review (human decides)
 *                                                       └► Unmatched (left as-is)       ─► Outputs
 *
 * With no run selected the Ingest node takes files (drag-and-drop or Browse) and
 * the panel below holds the run options. With a run selected, the nodes light up
 * with live status / timings / counts and clicking one opens its detail below.
 */

interface Props {
  client: EngineClient;
  health: Health | null;
  run: RunSnapshot | null;
  onStarted: (run: RunSnapshot) => void;
  onUpdate: (run: RunSnapshot) => void;
  onNewRun: () => void;
}

type Slot = "ucc" | "master" | "universe";
type NodeId = "ingest" | "direct" | "fuzzy" | "fuzzy_high" | "review" | "unmatched" | "outputs";
type NodeStatus = "idle" | "pending" | "running" | "done" | "failed";

const SLOTS: Array<{ slot: Slot; field: keyof Omit<CsvSource, "type">; title: string; hint: string; required: boolean }> = [
  { slot: "ucc", field: "ucc_path", title: "UCC filings", hint: "latest ucc.csv", required: true },
  { slot: "master", field: "master_path", title: "Enriched mapping", hint: "latest Master Mapping.csv", required: true },
  { slot: "universe", field: "universe_path", title: "Tracked banks", hint: "Bank Name + Firm ID (optional)", required: false },
];

/** Which engine stages roll up into which pipeline node (clustering stages are never shown). */
const NODE_STAGES: Record<"ingest" | "direct" | "fuzzy" | "outputs", string[]> = {
  ingest: ["ucc_raw_data", "master_mapping_data", "normalized_master_data", "our_universe_data"],
  direct: ["exact_match_raw_results", "exact_match_normalized_results"],
  fuzzy: ["master_embeddings_cache", "fuzzy_match_results", "advanced_embedding_results"],
  outputs: ["universe_enriched_results", "updated_master_data", "merged_raw_with_firm_id", "lead_bank_designation", "pipeline_outputs"],
};

const LS_KEY = "ucc.lastSource";
const LS_OVERRIDES = "ucc.overrides";

const emptyCsv: CsvSource = { type: "csv", ucc_path: "", master_path: "", universe_path: "" };

function basename(p: string) {
  return p.split(/[\\/]/).pop() ?? p;
}

/** Where a dropped file most likely belongs, from its name. */
function guessSlot(path: string): Slot {
  const n = basename(path).toLowerCase();
  if (/master|mapping|enrich/.test(n)) return "master";
  if (/universe|tracked|bank|firm/.test(n)) return "universe";
  return "ucc";
}

function nodeStatus(stages: StageState[], names: string[]): NodeStatus {
  const mine = stages.filter((s) => names.includes(s.name));
  if (mine.length === 0) return "idle";
  if (mine.some((s) => s.status === "failed")) return "failed";
  if (mine.some((s) => s.status === "running")) return "running";
  if (mine.every((s) => s.status === "done" || s.status === "skipped")) return "done";
  if (mine.some((s) => s.status === "done")) return "running";
  return "pending";
}

function nodeSeconds(stages: StageState[], names: string[]): number | null {
  const mine = stages.filter((s) => names.includes(s.name) && s.seconds != null);
  return mine.length ? mine.reduce((a, s) => a + (s.seconds ?? 0), 0) : null;
}

// ============================================================================
export function PipelineView({ client, health, run, onStarted, onUpdate, onNewRun }: Props) {
  // ---- inputs (only editable when no run is selected) ----------------------
  const stored = useMemo(() => {
    try {
      const s = JSON.parse(localStorage.getItem(LS_KEY) ?? "null") as SourceSpec | null;
      return s?.type === "csv" ? s : null;
    } catch {
      return null;
    }
  }, []);
  const [csv, setCsv] = useState<CsvSource>(stored ?? emptyCsv);
  const [preview, setPreview] = useState<SourcePreview | null>(null);
  const [name, setName] = useState("");
  const [sample, setSample] = useState(0);
  const [advanced, setAdvanced] = useState(false);
  const [overrides, setOverrides] = useState<Record<string, string>>(() => {
    try {
      return (JSON.parse(localStorage.getItem(LS_OVERRIDES) ?? "{}") as Record<string, string>) ?? {};
    } catch {
      return {};
    }
  });
  const [defaults, setDefaults] = useState<Record<string, unknown>>({});
  const [busy, setBusy] = useState<"" | "start">("");
  const [error, setError] = useState<string | null>(null);

  const editable = run === null;
  const shownSource: CsvSource = run && run.source.type === "csv" ? run.source : csv;

  useEffect(() => {
    client.configDefaults().then((d) => setDefaults(d.defaults)).catch(() => {});
  }, [client]);

  const updateOverride = (k: string, v: string) =>
    setOverrides((prev) => {
      const next = { ...prev, [k]: v };
      localStorage.setItem(LS_OVERRIDES, JSON.stringify(next));
      return next;
    });
  const activeOverrides = () =>
    Object.fromEntries(Object.entries(overrides).filter(([, v]) => v.trim() !== "").map(([k, v]) => [k, v.trim()]));

  const setSlot = useCallback((slot: Slot, path: string) => {
    const field = SLOTS.find((s) => s.slot === slot)!.field;
    setCsv((c) => ({ ...c, [field]: path }));
    setPreview(null);
  }, []);

  /** Dropped files -> slots. A file dropped on a zone goes there; the rest by name. */
  const assignDropped = useCallback(
    (paths: string[], target: Slot | null) => {
      const files = paths.filter((p) => /\.(csv|tsv|txt)$/i.test(p));
      if (files.length === 0) {
        setError("Only CSV / TSV / TXT files can be mapped.");
        return;
      }
      setError(null);
      setCsv((prev) => {
        const next = { ...prev };
        const taken = new Set<Slot>();
        const put = (slot: Slot, p: string) => {
          next[SLOTS.find((s) => s.slot === slot)!.field] = p;
          taken.add(slot);
        };
        const rest = [...files];
        if (target) put(target, rest.shift()!);
        for (const p of rest) {
          const g = guessSlot(p);
          if (!taken.has(g)) put(g, p);
          else {
            const free = SLOTS.map((s) => s.slot).find((s) => !taken.has(s) && !next[SLOTS.find((x) => x.slot === s)!.field]);
            if (free) put(free, p);
          }
        }
        return next;
      });
      setPreview(null);
    },
    [],
  );

  const hoverSlot = useNativeDragDrop(editable, assignDropped);

  const pick = async (slot: Slot) => {
    const title = `Choose ${SLOTS.find((s) => s.slot === slot)!.title} CSV`;
    const p = await native.pickFile(title);
    if (p) setSlot(slot, p);
  };

  const ready = !!(csv.ucc_path && csv.master_path);

  // Preview the inputs (columns + first rows) whenever they change, and for a selected run.
  const previewSpec: SourceSpec | null = run ? run.source : (csv.ucc_path || csv.master_path || csv.universe_path) ? csv : null;
  const previewKey = previewSpec ? JSON.stringify(previewSpec) : "";
  useEffect(() => {
    if (!previewSpec) { setPreview(null); return; }
    let alive = true;
    const t = setTimeout(() => {
      client.previewSource(previewSpec)
        .then((p) => { if (alive) setPreview(p); })
        .catch((e) => { if (alive) setError(describeError(e)); });
    }, 350);
    return () => { alive = false; clearTimeout(t); };
  }, [client, previewKey]); // eslint-disable-line react-hooks/exhaustive-deps
  const doStart = async () => {
    setBusy("start");
    setError(null);
    try {
      const started = await client.startRun({ source: csv, config: activeOverrides(), sample_rows: sample, name });
      localStorage.setItem(LS_KEY, JSON.stringify(csv));
      onStarted(started);
    } catch (e) {
      setError(describeError(e));
    } finally {
      setBusy("");
    }
  };

  // ---- run state -----------------------------------------------------------
  const stages = run?.stages ?? [];
  const active = run?.status === "running" || run?.status === "queued";
  const [summary, setSummary] = useState<Summary | null>(run?.summary ?? null);
  useEffect(() => {
    setSummary(run?.summary ?? null);
    if (run && !run.summary && run.status === "succeeded") client.summary(run.id).then(setSummary).catch(() => {});
  }, [client, run?.id, run?.summary, run?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  // Poll the selected run while it is active (status + logs).
  const [logs, setLogs] = useState<LogLine[]>([]);
  const lastSeq = useRef(0);
  useEffect(() => {
    setLogs([]);
    lastSeq.current = 0;
  }, [run?.id]);
  useEffect(() => {
    if (!run) return;
    let alive = true;
    const tick = async () => {
      try {
        const [snap, lg] = await Promise.all([client.getRun(run.id), client.logs(run.id, lastSeq.current)]);
        if (!alive) return;
        onUpdate(snap);
        if (lg.lines.length) {
          lastSeq.current = lg.last_seq;
          setLogs((prev) => [...prev, ...lg.lines].slice(-3000));
        }
      } catch {
        /* transient */
      }
    };
    tick();
    const t = setInterval(tick, active ? 1000 : 10000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [client, run?.id, active, onUpdate]); // eslint-disable-line react-hooks/exhaustive-deps

  const st = {
    ingest: run ? nodeStatus(stages, NODE_STAGES.ingest) : "idle",
    direct: run ? nodeStatus(stages, NODE_STAGES.direct) : "idle",
    fuzzy: run ? nodeStatus(stages, NODE_STAGES.fuzzy) : "idle",
    outputs: run ? nodeStatus(stages, NODE_STAGES.outputs) : "idle",
  } satisfies Record<string, NodeStatus>;
  const finished = run?.status === "succeeded";
  const bucketStatus: NodeStatus = !run ? "idle" : finished ? "done" : st.fuzzy === "done" ? "running" : run.status === "failed" ? "failed" : "pending";
  const buckets = summary?.buckets;

  // ---- which detail panel is open ------------------------------------------
  const [panel, setPanel] = useState<NodeId>("ingest");
  const [userPicked, setUserPicked] = useState(false);
  useEffect(() => {
    // Follow the pipeline while it runs; land on Review when it finishes.
    setUserPicked(false);
    if (!run) setPanel("ingest");
    else if (run.status === "succeeded") setPanel("review");
    else setPanel("direct");
  }, [run?.id, run?.status]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!run || !active || userPicked) return;
    if (st.fuzzy === "running") setPanel("fuzzy");
    else if (st.direct === "running") setPanel("direct");
    else if (st.outputs === "running") setPanel("outputs");
    else if (st.ingest === "running") setPanel("ingest");
  }, [run, active, userPicked, st.ingest, st.direct, st.fuzzy, st.outputs]);
  const choose = (id: NodeId) => {
    setUserPicked(true);
    setPanel(id);
  };

  const pct = run ? Math.round(run.progress * 100) : 0;
  const total = summary?.total_records ?? 0;
  const share = (n?: number) => (n != null && total ? `${((n / total) * 100).toFixed(1)}%` : undefined);

  return (
    <div className="page pipeline-page">
      <header className="page-head row between">
        <div>
          <h1>{run ? run.name : "New mapping run"}</h1>
          <p className="muted">
            {run
              ? `${run.status.toUpperCase()}${active ? ` · ${pct}%` : ""}${run.input_rows.ucc ? ` · ${run.input_rows.ucc.toLocaleString()} UCC rows` : ""}${run.input_rows.master ? ` · ${run.input_rows.master.toLocaleString()} mapping rows` : ""}${run.sample_rows ? ` · sample of ${run.sample_rows.toLocaleString()}` : ""}${summary ? ` · ${summary.total_seconds}s` : ""}`
              : "Drop the latest UCC file and the current enriched mapping onto the first step, then start the pipeline."}
          </p>
        </div>
        <div className="row gap">
          {run && active && (
            <button className="btn btn-danger" onClick={() => client.cancelRun(run.id).then(onUpdate)}>Cancel after current stage</button>
          )}
          {run && !active && (
            <button className="btn" onClick={() => { setCsv(run.source.type === "csv" ? run.source : csv); onNewRun(); }}>
              New run with these inputs
            </button>
          )}
          {!run && (
            <button className="btn btn-primary" disabled={!ready || busy !== "" || !!health?.active_run} onClick={doStart}>
              {busy === "start" ? "Starting…" : "▶ Run pipeline"}
            </button>
          )}
        </div>
      </header>

      {run && active && <div className="progress"><div style={{ width: `${pct}%` }} /></div>}
      {run?.error && <div className="alert">{run.error}</div>}

      {/* ------------------------------------------------------------ flow */}
      <div className="flow">
        <FlowNode id="ingest" title="1 · Ingest" status={st.ingest} selected={panel === "ingest"} onClick={() => choose("ingest")}
          seconds={nodeSeconds(stages, NODE_STAGES.ingest)}>
          <div className="dropzones">
            {SLOTS.map((s) => (
              <DropZone key={s.slot} slot={s.slot} title={s.title} hint={s.hint} required={s.required}
                path={shownSource[s.field]} hovering={hoverSlot === s.slot} editable={editable}
                status={preview?.inputs[s.slot]} onBrowse={() => pick(s.slot)} onClear={() => setSlot(s.slot, "")}
                onPath={(p) => setSlot(s.slot, p)} />
            ))}
          </div>
        </FlowNode>

        <Arrow />

        <FlowNode id="direct" title="2 · Direct match" status={st.direct} selected={panel === "direct"} onClick={() => choose("direct")}
          seconds={nodeSeconds(stages, NODE_STAGES.direct)} count={buckets?.direct} sub={share(buckets?.direct)}>
          <p className="node-desc">Exact on the filed name, then on the normalised name (suffixes, roles, abbreviations stripped).</p>
        </FlowNode>

        <Arrow />

        <FlowNode id="fuzzy" title="3 · Fuzzy match" status={st.fuzzy} selected={panel === "fuzzy"} onClick={() => choose("fuzzy")}
          seconds={nodeSeconds(stages, NODE_STAGES.fuzzy)}
          count={buckets ? buckets.fuzzy_high + buckets.review : undefined} sub={buckets ? "candidates scored" : undefined}>
          <p className="node-desc">Token-level string similarity, then embedding + multi-signal scoring for what's left. Every candidate gets a probability.</p>
        </FlowNode>

        <Arrow />

        <div className="fanout">
          <FlowNode id="fuzzy_high" tone="good" title="High probability" status={bucketStatus} selected={panel === "fuzzy_high"}
            onClick={() => choose("fuzzy_high")} count={buckets?.fuzzy_high} sub={share(buckets?.fuzzy_high)}>
            <p className="node-desc">Classified <em>fuzzy match · high probability</em>. Accepted automatically and added to the updated mapping.</p>
          </FlowNode>
          <FlowNode id="review" tone="warn" title="Review" status={bucketStatus} selected={panel === "review"}
            onClick={() => choose("review")} count={buckets?.review} sub={share(buckets?.review)}>
            <p className="node-desc">Plausible but not certain. Grouped for a person to approve, correct or reject.</p>
          </FlowNode>
          <FlowNode id="unmatched" tone="muted" title="Unmatched" status={bucketStatus} selected={panel === "unmatched"}
            onClick={() => choose("unmatched")} count={buckets?.unmatched} sub={share(buckets?.unmatched)}>
            <p className="node-desc">Probability too low. Left as filed — exported, never added to the mapping.</p>
          </FlowNode>
        </div>

        <Arrow />

        <FlowNode id="outputs" title="4 · Outputs" status={st.outputs} selected={panel === "outputs"} onClick={() => choose("outputs")}
          seconds={nodeSeconds(stages, NODE_STAGES.outputs)}
          count={summary?.updated_master?.rows} sub={summary?.updated_master ? "rows in updated mapping" : undefined}>
          <p className="node-desc">Updated mapping, Firm-ID enriched filings, lead-bank file. Promote the mapping when you're happy.</p>
        </FlowNode>
      </div>

      {error && <div className="alert">{error}</div>}

      {/* ----------------------------------------------------------- panel */}
      <section className="card panel">
        {panel === "ingest" && (
          editable ? (
            <IngestPanel client={client} preview={preview} name={name} setName={setName}
              sample={sample} setSample={setSample} advanced={advanced} setAdvanced={setAdvanced}
              overrides={overrides} defaults={defaults} updateOverride={updateOverride} activeOverrides={activeOverrides} />
          ) : (
            <>
              <InputPreviews preview={preview} />
              <StagePanel title="Ingest" stages={stages.filter((s) => NODE_STAGES.ingest.includes(s.name))} logs={logs} run={run!} />
            </>
          )
        )}
        {panel === "direct" && run && (
          <StagePanel title="Direct match" stages={stages.filter((s) => NODE_STAGES.direct.includes(s.name))} logs={logs} run={run}
            methods={summary ? pickMethods(summary.by_match_method, /^Exact Match/) : undefined} total={total} />
        )}
        {panel === "fuzzy" && run && (
          <StagePanel title="Fuzzy match" stages={stages.filter((s) => NODE_STAGES.fuzzy.includes(s.name))} logs={logs} run={run}
            methods={summary ? pickMethods(summary.by_match_method, /Fuzzy|Advanced/) : undefined} total={total}
            bands={summary?.review_bands} />
        )}
        {panel === "fuzzy_high" && run && (
          <BucketPanel run={run} client={client} bucket="fuzzy_high" title="Fuzzy match · high probability"
            blurb="Above the auto threshold (fuzzy ≥ fuzzy_auto_threshold, embedding ≥ embed_auto_threshold). These are added to the updated mapping without review — spot-check before promoting." />
        )}
        {panel === "review" && run && (
          finished ? <ReviewTab run={run} client={client} /> : <p className="muted">The review queue is available once the run finishes.</p>
        )}
        {panel === "unmatched" && run && (
          <BucketPanel run={run} client={client} bucket="unmatched" title="Unmatched"
            blurb="No candidate above the floor (fuzzy_min_threshold / embed_medium_threshold). Exported as new entities; nothing is written to the mapping." />
        )}
        {panel === "outputs" && run && <OutputsPanel run={run} client={client} summary={summary} stages={stages} logs={logs} />}
        {panel !== "ingest" && !run && <p className="muted">Start a run to see this step.</p>}
      </section>
    </div>
  );
}

// ============================================================================
// Native drag-and-drop (Tauri). HTML5 drop events carry no file paths and are
// suppressed by the webview's own handler, so we listen to Tauri's event and
// hit-test the cursor position against the drop zones ourselves.
function useNativeDragDrop(enabled: boolean, onDrop: (paths: string[], slot: Slot | null) => void): Slot | null {
  const [hover, setHover] = useState<Slot | null>(null);
  const cb = useRef(onDrop);
  cb.current = onDrop;
  useEffect(() => {
    if (!native.inTauri || !enabled) return;
    let unlisten: (() => void) | undefined;
    let disposed = false;
    const zoneAt = (pos: { x: number; y: number }): Slot | null => {
      const scale = window.devicePixelRatio || 1;
      const el = document.elementFromPoint(pos.x / scale, pos.y / scale);
      const z = el?.closest<HTMLElement>("[data-slot]");
      return (z?.dataset.slot as Slot | undefined) ?? null;
    };
    import("@tauri-apps/api/webview").then(({ getCurrentWebview }) =>
      getCurrentWebview().onDragDropEvent((event) => {
        const p = event.payload;
        if (p.type === "enter" || p.type === "over") setHover(zoneAt(p.position) ?? ("ucc" as Slot));
        else if (p.type === "drop") {
          setHover(null);
          cb.current(p.paths, zoneAt(p.position));
        } else setHover(null);
      }),
    ).then((u) => {
      if (disposed) u();
      else unlisten = u;
    }).catch(() => {});
    return () => {
      disposed = true;
      unlisten?.();
    };
  }, [enabled]);
  return hover;
}

// ============================================================================
function FlowNode({ id, title, status, selected, onClick, seconds, count, sub, tone, children }: {
  id: NodeId; title: string; status: NodeStatus; selected: boolean; onClick: () => void;
  seconds?: number | null; count?: number; sub?: string; tone?: "good" | "warn" | "muted"; children?: React.ReactNode;
}) {
  return (
    <button type="button" className={`node ${status} ${selected ? "selected" : ""} ${tone ?? ""}`} data-node={id} onClick={onClick}>
      <div className="node-head">
        <span className={`dot ${status === "done" ? "done" : status === "running" ? "running" : status === "failed" ? "failed" : ""}`} />
        <span className="node-title">{title}</span>
        <span className="muted small node-time">{seconds != null ? `${seconds.toFixed(1)}s` : status === "running" ? "running…" : ""}</span>
      </div>
      {count != null && (
        <div className="node-count">
          <span className="node-num">{count.toLocaleString()}</span>
          {sub && <span className="muted small"> {sub}</span>}
        </div>
      )}
      {children}
    </button>
  );
}

function Arrow() {
  return <div className="arrow" aria-hidden><span /></div>;
}

function DropZone({ slot, title, hint, required, path, hovering, editable, status, onBrowse, onClear, onPath }: {
  slot: Slot; title: string; hint: string; required: boolean; path: string; hovering: boolean; editable: boolean;
  status?: SourcePreview["inputs"]["ucc"]; onBrowse: () => void; onClear: () => void; onPath: (p: string) => void;
}) {
  const ok = status && status.present && !status.error && (status.missing_columns?.length ?? 0) === 0;
  const bad = status && status.present && (status.error || (status.missing_columns?.length ?? 0) > 0);
  return (
    <div className={`dropzone ${path ? "filled" : ""} ${hovering ? "hover" : ""} ${editable ? "" : "readonly"} ${bad ? "bad" : ok ? "ok" : ""}`}
      data-slot={editable ? slot : undefined} onClick={(e) => { e.stopPropagation(); if (editable && !path) onBrowse(); }}>
      <div className="dz-title">
        {title}{required ? "" : <span className="muted"> · optional</span>}
      </div>
      {path ? (<>
        <div className="dz-file" title={path}>
          <span className="dz-name">{basename(path)}</span>
          {editable && (
            <span className="dz-actions">
              <button type="button" className="link small" onClick={(e) => { e.stopPropagation(); onBrowse(); }}>change</button>
              <button type="button" className="link small" onClick={(e) => { e.stopPropagation(); onClear(); }}>×</button>
            </span>
          )}
        </div>
        {(status?.rows != null || bad) && (
          <div className="dz-meta small">
            {status?.rows != null && <span className="muted">{status.rows.toLocaleString()} rows</span>}
            {bad && <span className="bad"> · {status?.error ?? `missing ${status?.missing_columns?.join(", ")}`}</span>}
          </div>
        )}
      </>) : native.inTauri || !editable ? (
        <div className="dz-empty muted small">drop {hint} here, or click to browse</div>
      ) : (
        <input className="dz-input" placeholder={`path to ${hint}`} onClick={(e) => e.stopPropagation()}
          onKeyDown={(e) => { if (e.key === "Enter") onPath((e.target as HTMLInputElement).value); }}
          onBlur={(e) => { if (e.target.value) onPath(e.target.value); }} />
      )}
    </div>
  );
}

// ============================================================================
function IngestPanel(p: {
  client: EngineClient; preview: SourcePreview | null;
  name: string; setName: (v: string) => void; sample: number; setSample: (v: number) => void;
  advanced: boolean; setAdvanced: (v: boolean) => void;
  overrides: Record<string, string>; defaults: Record<string, unknown>; updateOverride: (k: string, v: string) => void;
  activeOverrides: () => Record<string, string>;
}) {
  const thresholds: Array<[key: string, hint: string]> = [
    ["fuzzy_auto_threshold", "Fuzzy score (0–100) at or above which a match is high probability (auto)."],
    ["fuzzy_review_threshold", "Fuzzy score at or above which a match goes to review."],
    ["fuzzy_min_threshold", "Floor: below this a fuzzy candidate is dropped (unmatched)."],
    ["embed_auto_threshold", "Embedding composite (0–1) for high probability (auto)."],
    ["embed_high_threshold", "Embedding composite for review · high."],
    ["embed_medium_threshold", "Floor for review · medium; below it is unmatched."],
  ];
  const modelSettings: Array<[key: string, hint: string]> = [
    ["model_dir", "Read the models in place from this folder instead of copying (local path or share). Overrides the network folder."],
    ["model_cache_dir", "The application's local model folder (where the copy lives / where downloads go)."],
    ["cpu_threads", "BLAS threads per process (0 = default)."],
  ];
  return (
    <>
      <InputPreviews preview={p.preview} />

      <div className="row gap wrap">
        <span className="grow" />
        <label className="field inline-field">
          <span>Run name</span>
          <input value={p.name} onChange={(e) => p.setName(e.target.value)} placeholder="e.g. October refresh" />
        </label>
        <label className="field inline-field">
          <span>Quick test · random sample (0 = all rows)</span>
          <input type="number" min={0} step={1000} value={p.sample} onChange={(e) => p.setSample(Number(e.target.value) || 0)} />
        </label>
      </div>

      <ModelsCard client={p.client} share={p.overrides.model_share_dir ?? ""} setShare={(v) => p.updateOverride("model_share_dir", v)}
        overrides={p.activeOverrides()} />

      <button className="link" onClick={() => p.setAdvanced(!p.advanced)}>{p.advanced ? "Hide" : "Show"} probability thresholds & model settings</button>
      {p.advanced && (
        <>
          <h3 className="subhead">Probability thresholds</h3>
          <div className="grid3">
            {thresholds.map(([k, hint]) => (
              <label className="field" key={k}>
                <span>{k}</span>
                <input value={p.overrides[k] ?? ""} placeholder={String(p.defaults[k] ?? "")} onChange={(e) => p.updateOverride(k, e.target.value)} />
                <small className="muted">{hint}</small>
              </label>
            ))}
          </div>
          <h3 className="subhead">Models & performance</h3>
          <div className="grid3">
            {modelSettings.map(([k, hint]) => (
              <label className="field" key={k}>
                <span>{k}</span>
                <input value={p.overrides[k] ?? ""} placeholder={String(p.defaults[k] ?? "")} onChange={(e) => p.updateOverride(k, e.target.value)} />
                <small className="muted">{hint}</small>
              </label>
            ))}
          </div>
        </>
      )}
    </>
  );
}

// ============================================================================
/** What came in: one card per input file with its columns and first rows. */
function InputPreviews({ preview }: { preview: SourcePreview | null }) {
  const [open, setOpen] = useState<Slot | null>("ucc");
  const required: Record<Slot, string[]> = {
    ucc: ["DEBT_STATE", "SECP_STATE", "SECP_NAME"],
    master: ["DEBT_STATE", "SECP_STATE", "SECP_NAME", "SECP_NAME_MAPPED", "Firm ID"],
    universe: ["Bank Name", "Firm ID"],
  };
  const present = SLOTS.filter((s) => preview?.inputs[s.slot]?.present);
  if (!preview || present.length === 0) {
    return <p className="muted small" style={{ margin: 0 }}>No input files yet — drop them on the Ingest step to see their columns and first rows here.</p>;
  }
  const mb = (n?: number) => (n == null ? "" : n > 1_048_576 ? `${(n / 1_048_576).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`);
  return (
    <div className="input-previews">
      {present.map((s) => {
        const x = preview.inputs[s.slot];
        const missing = x.missing_columns ?? [];
        const ok = !x.error && missing.length === 0;
        const isOpen = open === s.slot;
        const cols = x.columns ?? [];
        return (
          <section key={s.slot} className={`input-card ${ok ? "ok" : "bad"} ${isOpen ? "open" : ""}`}>
            <div className="input-head" onClick={() => setOpen(isOpen ? null : s.slot)}>
              <span className="bank-caret">{isOpen ? "▾" : "▸"}</span>
              <span className="input-title">{s.title}</span>
              <span className="input-file" title={x.path}>{x.path ? basename(x.path) : ""}</span>
              <span className="muted small input-meta">
                {x.rows != null ? `${x.rows.toLocaleString()} rows` : ""}{x.size_bytes ? ` · ${mb(x.size_bytes)}` : ""}{cols.length ? ` · ${cols.length} columns` : ""}
              </span>
              <span className={`pill ${ok ? "approved" : "rejected"}`}>
                {x.error ? "cannot read" : missing.length ? `missing ${missing.join(", ")}` : "ready"}
              </span>
            </div>
            {isOpen && (
              x.error ? <p className="small bad" style={{ margin: "0 12px 10px" }}>{x.error}</p> : (
                <div className="table-wrap inset">
                  <table className="table compact preview">
                    <thead>
                      <tr>{cols.map((c) => <th key={c} className={required[s.slot].includes(c) ? "req" : ""} title={required[s.slot].includes(c) ? "used by the pipeline" : "carried through, not used for matching"}>{c}</th>)}</tr>
                    </thead>
                    <tbody>
                      {(x.preview ?? []).map((row, i) => (
                        <tr key={i}>{cols.map((c) => <td key={c} title={row[c]}>{row[c]}</td>)}</tr>
                      ))}
                    </tbody>
                  </table>
                  <div className="muted small" style={{ padding: "6px 10px" }}>
                    First {x.preview?.length ?? 0} of {x.rows?.toLocaleString()} rows · <strong>bold</strong> columns are the ones the pipeline uses
                    {x.path ? <> · <span className="mono">{x.path}</span></> : null}
                  </div>
                </div>
              )
            )}
          </section>
        );
      })}
    </div>
  );
}

// ============================================================================
/** The two matching models live on a network share and are *copied* onto this
 *  PC's application model folder (once; re-copied only when the share has newer
 *  files). The pipeline then loads them from the local copy, offline. */
function ModelsCard({ client, share, setShare, overrides }: {
  client: EngineClient; share: string; setShare: (v: string) => void; overrides: Record<string, string>;
}) {
  const [status, setStatus] = useState<ModelsStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => {
      client.modelsStatus({ model_share_dir: share, model_cache_dir: overrides.model_cache_dir })
        .then((s) => { setStatus(s); setErr(null); })
        .catch((e) => setErr(describeError(e)));
    }, 300);
    return () => clearTimeout(t);
  }, [client, share, overrides.model_cache_dir, tick]);

  const copy = async () => {
    setBusy(true);
    setMsg(null);
    setErr(null);
    try {
      const r = await client.syncModels(overrides);
      setMsg(r.status === "updated"
        ? `Copied ${r.copied} file(s) (${(r.bytes / 1e6).toFixed(0)} MB) in ${r.seconds}s → ${r.local}`
        : r.status === "up_to_date" ? `Local models are up to date with the network folder.`
        : `Network folder not reachable — using the local copy in ${r.local}.`);
      setTick((t) => t + 1);
    } catch (e) {
      setErr(describeError(e));
    } finally {
      setBusy(false);
    }
  };
  const download = async () => {
    setBusy(true);
    setMsg(null);
    setErr(null);
    try {
      const r = await client.prefetchModels(overrides);
      setMsg(`Models ready in ${r.location} · ${r.embedding} · ${r.reranker}`);
      setTick((t) => t + 1);
    } catch (e) {
      setErr(describeError(e));
    } finally {
      setBusy(false);
    }
  };

  const localOk = (status?.local_models.length ?? 0) >= 2;
  const shareOk = !!status?.share_reachable;
  return (
    <div className="models-card">
      <div className="row between wrap">
        <div>
          <h4>Matching models</h4>
          <p className="small muted" style={{ maxWidth: 720, margin: 0 }}>
            The two models (name embedder + reranker, ~150 MB) are copied from the network folder into this application's model
            folder the first time they are needed, and again only when the network copy is newer. Runs always use the local copy.
          </p>
        </div>
        <span className={`pill ${localOk ? "approved" : "rejected"}`}>{localOk ? "models on this PC" : "models missing on this PC"}</span>
      </div>
      <div className="row gap">
        <label className="field grow">
          <span>Network model folder (UNC path or mapped drive)</span>
          <input value={share} placeholder={"\\\\fileserver\\apps\\ucc-models"} onChange={(e) => setShare(e.target.value)} />
        </label>
        <button className="btn" disabled={busy || !share} onClick={copy} title="Copy anything missing or newer from the network folder now">
          {busy ? "Copying…" : "Copy models from network"}
        </button>
        {!share && (
          <button className="btn" disabled={busy} onClick={download} title="No network folder: download the models from the internet into the local folder">
            {busy ? "Downloading…" : "Download models"}
          </button>
        )}
      </div>
      {status && (
        <p className="small muted" style={{ margin: 0 }}>
          {share ? (shareOk ? `Network folder reachable · ${status.share_models.length} model(s)` : "Network folder not reachable from this PC") : "No network folder set"}
          {" · "}local folder {status.local}: {status.local_models.length ? status.local_models.map((m) => m.replace(/^models--/, "")).join(", ") : "empty"}
          {status.last_synced ? ` · last copied ${status.last_synced.replace("T", " ")}` : ""}
        </p>
      )}
      {msg && <p className="small" style={{ margin: 0 }}>{msg}</p>}
      {err && <p className="small bad" style={{ margin: 0 }}>{err}</p>}
    </div>
  );
}

function pickMethods(by: Record<string, number>, re: RegExp): Record<string, number> {
  return Object.fromEntries(Object.entries(by).filter(([k]) => re.test(k)));
}

function StagePanel({ title, stages, logs, run, methods, total, bands }: {
  title: string; stages: StageState[]; logs: LogLine[]; run: RunSnapshot;
  methods?: Record<string, number>; total?: number; bands?: Record<string, number>;
}) {
  const names = new Set(stages.map((s) => s.name));
  const mine = logs.filter((l) => names.has(l.stage) && l.level !== "DEBUG");
  const [showAll, setShowAll] = useState(false);
  const shown = showAll ? logs.filter((l) => l.level !== "DEBUG") : mine;
  return (
    <div className="split">
      <div>
        <h4>{title} · steps</h4>
        {stages.map((s) => (
          <div key={s.name} className={`stage ${s.status}`}>
            <span className={`dot ${s.status}`} />
            <span className="stage-label">{s.label}</span>
            <span className="muted small">
              {s.status === "done" && s.seconds != null ? `${s.seconds.toFixed(1)}s` : ""}
              {s.rows != null ? ` · ${s.rows.toLocaleString()} rows` : ""}
              {s.status === "running" ? "running…" : s.status === "skipped" ? "skipped" : ""}
            </span>
          </div>
        ))}
        {methods && Object.keys(methods).length > 0 && (
          <>
            <h4 style={{ marginTop: 16 }}>Matched by method</h4>
            <Bars data={methods} total={total ?? 0} />
          </>
        )}
        {bands && (
          <>
            <h4 style={{ marginTop: 16 }}>Probability bands sent to review</h4>
            <Bars data={bands} total={total ?? 0} />
          </>
        )}
      </div>
      <div className="logs">
        <div className="row between">
          <h4>Log</h4>
          <label className="small"><input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} /> whole run</label>
        </div>
        <LogBox lines={shown} active={run.status === "running" || run.status === "queued"} />
      </div>
    </div>
  );
}

function LogBox({ lines, active }: { lines: LogLine[]; active: boolean }) {
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (active && box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [lines, active]);
  return (
    <div className="logbox" ref={box}>
      {lines.map((l) => (
        <div key={l.seq} className={`log ${l.level.toLowerCase()}`}>
          <span className="muted">{l.ts.slice(11, 19)}</span> <span className="tag">{l.stage}</span> {l.message}
        </div>
      ))}
      {lines.length === 0 && <p className="muted small">{active ? "Waiting for output…" : "No log lines for this step."}</p>}
    </div>
  );
}

// ============================================================================
function BucketPanel({ run, client, bucket, title, blurb }: { run: RunSnapshot; client: EngineClient; bucket: Bucket; title: string; blurb: string }) {
  const [search, setSearch] = useState("");
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState<(import("../types").Page & { mapped_total?: number }) | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  const [banks, setBanks] = useState<import("../types").Bank[]>([]);
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [show, setShow] = useState<"pending" | "mapped" | "all">("pending");
  const limit = 100;
  const finished = run.status === "succeeded";
  const manual = bucket === "unmatched";
  useEffect(() => {
    if (!finished) return;
    const t = setTimeout(() => {
      client.rows(run.id, { bucket, search, offset, limit, decision: manual && show !== "all" ? show : undefined })
        .then((p) => { setPage(p); setLoadError(null); })
        .catch((e) => { setPage(null); setLoadError(describeError(e)); });
    }, 200);
    return () => clearTimeout(t);
  }, [client, run.id, bucket, search, offset, tick, finished]);
  useEffect(() => {
    if (finished && manual) client.banks(run.id).then(setBanks).catch(() => setBanks([]));
  }, [client, run.id, finished, manual]);
  if (!finished) return <p className="muted">Available once the run finishes.</p>;

  const mapTo = async (key: string, bank: string, allStates: boolean) => {
    setBusy(true);
    setMsg(null);
    try {
      const r = await client.setDecisions(run.id, [{ key, decision: "approved", mapped_name: bank, all_states: allStates }]);
      const inh = r.inherited.approved;
      setMsg(`Mapped → ${bank}${inh ? ` · ${inh} more filer(s) with the same name inherited it` : ""}. It goes into the updated mapping with “Add approved”.`);
      setTick((t) => t + 1);
    } catch (e) {
      setMsg(describeError(e));
    } finally {
      setBusy(false);
    }
  };
  const undo = async (key: string) => {
    setBusy(true);
    try {
      await client.setDecisions(run.id, [{ key, decision: null }]);
      setTick((t) => t + 1);
    } catch (e) {
      setMsg(describeError(e));
    } finally {
      setBusy(false);
    }
  };
  const apply = async () => {
    setBusy(true);
    setMsg(null);
    try {
      const r = await client.applyDecisions(run.id);
      setMsg(`Added ${r.added.toLocaleString()} row(s) to the updated mapping (${(r.by_hand ?? 0).toLocaleString()} mapped by hand, ${(r.added - (r.by_hand ?? 0)).toLocaleString()} approved in review) → ${r.file.split(/[\\/]/).pop()} (${r.rows.toLocaleString()} rows). They are listed under "mapped by hand" / "approved in review" on the Outputs step; promote there to make it the master.`);
      setTick((t) => t + 1);
    } catch (e) {
      setMsg(describeError(e));
    } finally {
      setBusy(false);
    }
  };

  const cols = manual
    ? ["SECP_NAME", "DEBT_STATE", "SECP_STATE", "SECP_NAME_NORMALIZED"]
    : ["SECP_NAME", "SECP_NAME_MAPPED", "matched_master_raw", "DEBT_STATE", "SECP_STATE", "match_method", "match_score", "confidence", "Firm_ID"];
  const rows = page?.rows ?? [];
  const counts = page?.decision_counts;
  const mappedHere = page?.mapped_total ?? 0;
  return (
    <>
      <div className="row between wrap">
        <div>
          <h4>{title}</h4>
          <p className="small muted" style={{ maxWidth: 720 }}>{blurb}</p>
        </div>
        <div className="row gap">
          {manual && (
            <div className="seg small">
              {([["pending", "To map"], ["mapped", `Mapped by hand${mappedHere ? ` (${mappedHere.toLocaleString()})` : ""}`], ["all", "All"]] as const).map(([v, l]) => (
                <button key={v} className={show === v ? "on" : ""} onClick={() => { setOffset(0); setShow(v); }}>{l}</button>
              ))}
            </div>
          )}
          <input style={{ maxWidth: 360 }} placeholder="Search secured party / normalised name…" value={search}
            onChange={(e) => { setOffset(0); setSearch(e.target.value); }} />
        </div>
      </div>
      {manual && (
        <div className="row between wrap decision-bar">
          <span className="small muted">
            Know the bank? Pick it from the tracked banks and <em>Map</em>. Tick <em>all states</em> when the filer name is that bank everywhere
            (JPMORGAN CHASE BANK); leave it off for generic names that differ by state (PINNACLE BANK). Mapped filers are remembered for later runs
            and go into the updated mapping with the button on the right.
          </span>
          <button className="btn btn-primary" disabled={busy || !counts?.unapplied_approved} onClick={apply}>
            {`Add ${(counts?.unapplied_approved ?? 0).toLocaleString()} approved / mapped to updated mapping`}
          </button>
        </div>
      )}
      {msg && <p className="small">{msg}</p>}
      {loadError && !page
        ? <p className="muted"><LoadFailed error={loadError} onRetry={() => setTick((t) => t + 1)} /></p>
        : manual ? (
          <>
            <datalist id="bank-list">{banks.map((b) => <option key={b.name} value={b.name}>{b.firm_id != null ? `Firm ID ${b.firm_id}` : ""}</option>)}</datalist>
            <div className="table-wrap">
              <table className="table review manual">
                <thead><tr><th>Secured party (as filed)</th><th>States</th><th>Normalised</th><th>Map to bank</th><th>Status</th></tr></thead>
                <tbody>
                  {rows.map((r) => <ManualRow key={String(r.key)} row={r} run={run} busy={busy} banks={banks} onMap={mapTo} onUndo={undo} />)}
                  {page && rows.length === 0 && <tr><td colSpan={5} className="muted">No rows.</td></tr>}
                  {!page && <tr><td colSpan={5} className="muted">Loading…</td></tr>}
                </tbody>
              </table>
            </div>
            {page && (
              <div className="row between small">
                <span className="muted">{page.total.toLocaleString()} filers · showing {Math.min(offset + 1, page.total)}–{Math.min(offset + limit, page.total)} · sorted by number of filings</span>
                <div className="row gap">
                  <button className="btn" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>‹ Prev</button>
                  <button className="btn" disabled={offset + limit >= page.total} onClick={() => setOffset(offset + limit)}>Next ›</button>
                </div>
              </div>
            )}
          </>
        ) : <DataTable page={page} cols={cols} offset={offset} limit={limit} setOffset={setOffset} />}
    </>
  );
}

function ManualRow({ row: r, run, busy, banks, onMap, onUndo }: {
  row: Record<string, unknown>; run: RunSnapshot; busy: boolean; banks: import("../types").Bank[];
  onMap: (key: string, bank: string, allStates: boolean) => void; onUndo: (key: string) => void;
}) {
  const [bank, setBank] = useState("");
  const [allStates, setAllStates] = useState(true);
  const str = (v: unknown) => (v == null ? "" : String(v));
  const key = str(r.key);
  const decided = !!r.decision;
  const known = banks.find((b) => b.name.toLowerCase() === bank.trim().toLowerCase());
  const filings = typeof r.filings === "number" ? r.filings : 1;
  const inherited = str(r.decision_source).startsWith("inherited:");
  return (
    <tr className={decided ? "decided approved" : ""}>
      <td title={str(r.SECP_NAME)}><div className="cell-main">{str(r.SECP_NAME)}{filings > 1 ? <span className="muted small"> ×{filings}</span> : null}</div></td>
      <td className="muted small">{str(r.DEBT_STATE)} → {str(r.SECP_STATE)}</td>
      <td className="small muted" title={str(r.SECP_NAME_NORMALIZED)}><div className="cell-main">{str(r.SECP_NAME_NORMALIZED)}</div></td>
      <td>
        {decided ? (
          <strong>{str(r.decided_mapped_name)}</strong>
        ) : (
          <div className="map-ctl">
            <input className="inline bank-pick" list="bank-list" placeholder="type a bank…" value={bank} disabled={busy}
              onChange={(e) => setBank(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && known) onMap(key, known.name, allStates); }} />
            <label className="map-all" title="Also map this filer name wherever it is unmatched, now and in later runs">
              <input type="checkbox" checked={allStates} onChange={(e) => setAllStates(e.target.checked)} /><span>all states</span>
            </label>
            <button className="btn ok" disabled={busy || !known} title={bank && !known ? "Pick a bank from the list" : undefined}
              onClick={() => known && onMap(key, known.name, allStates)}>Map</button>
          </div>
        )}
      </td>
      <td>
        {decided ? (
          <div className="row gap">
            <span className="pill approved" title={inherited ? `inherited from ${str(r.decision_source).slice(10)}` : r.decided_in_run && r.decided_in_run !== run.id ? `mapped in run ${str(r.decided_in_run)}` : undefined}>
              mapped{inherited ? " · inherited" : r.decided_in_run && r.decided_in_run !== run.id ? " · earlier run" : ""}{r.decided_firm_id != null ? ` · Firm ID ${str(r.decided_firm_id)}` : ""}
            </span>
            <button className="link small" disabled={busy} onClick={() => onUndo(key)}>undo</button>
          </div>
        ) : <span className="muted small">unmatched</span>}
      </td>
    </tr>
  );
}

// ============================================================================
/** The updated mapping with provenance: what came in vs what this run / reviewers added. */
function MasterProvenance({ run, client }: { run: RunSnapshot; client: EngineClient }) {
  const [source, setSource] = useState<string>("added");
  const [search, setSearch] = useState("");
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState<(import("../types").Page & { by_source: Record<string, number> }) | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [open, setOpen] = useState(true);
  const limit = 100;
  useEffect(() => {
    if (!open) return;
    const t = setTimeout(() => {
      client.masterRows(run.id, { source, search, offset, limit })
        .then((p) => { setPage(p); setLoadError(null); })
        .catch((e) => { setPage(null); setLoadError(describeError(e)); });
    }, 200);
    return () => clearTimeout(t);
    // re-read when "Add approved / mapped" or Promote wrote a new file for this run
  }, [client, run.id, source, search, offset, open, run.summary?.updated_master?.file]);
  const label = (k: string) =>
    k === "original_master" ? "came in as the mapping"
    : k === "mapped_by_hand" ? "mapped by hand"
    : k === "reviewed_approved" ? "approved in review"
    : k.startsWith("pipeline_") ? `pipeline · ${k.slice(9)}` : k;
  const by = page?.by_source ?? {};
  const added = Object.entries(by).filter(([k]) => k !== "original_master").sort((a, b) => b[1] - a[1]);
  return (
    <div className="provenance">
      <div className="row gap wrap">
        <button className="link" onClick={() => setOpen((o) => !o)}>{open ? "Hide" : "Show"} what came in vs what was added</button>
      </div>
      {open && (
        <>
          <div className="seg small" style={{ marginTop: 8 }}>
            <button className={source === "original" ? "on" : ""} onClick={() => { setOffset(0); setSource("original"); }}>
              Came in {by.original_master != null ? `(${by.original_master.toLocaleString()})` : ""}
            </button>
            <button className={source === "added" ? "on" : ""} onClick={() => { setOffset(0); setSource("added"); }}>
              Added {added.length ? `(${added.reduce((a, [, v]) => a + v, 0).toLocaleString()})` : ""}
            </button>
            {added.map(([k, v]) => (
              <button key={k} className={source === k ? "on" : ""} onClick={() => { setOffset(0); setSource(k); }} title={k}>
                {label(k)} ({v.toLocaleString()})
              </button>
            ))}
          </div>
          <div className="row gap" style={{ marginTop: 8 }}>
            <input style={{ maxWidth: 360 }} placeholder="Search filer / bank…" value={search} onChange={(e) => { setOffset(0); setSearch(e.target.value); }} />
          </div>
          {loadError && !page
            ? <p className="muted"><LoadFailed error={loadError} onRetry={() => setOpen(true)} /></p>
            : <DataTable page={page} cols={["SECP_NAME", "SECP_NAME_MAPPED", "Firm ID", "DEBT_STATE", "SECP_STATE", "_source"]} offset={offset} limit={limit} setOffset={setOffset} />}
        </>
      )}
    </div>
  );
}

function OutputsPanel({ run, client, summary, stages, logs }: {
  run: RunSnapshot; client: EngineClient; summary: Summary | null; stages: StageState[]; logs: LogLine[];
}) {
  const [dest, setDest] = useState(run.source.type === "csv" ? run.source.master_path : "");
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [unapplied, setUnapplied] = useState(0);
  useEffect(() => {
    if (run.status === "succeeded") client.decisions(run.id).then((d) => setUnapplied(d.counts.unapplied_approved)).catch(() => {});
  }, [client, run.id, run.status]);

  if (run.status !== "succeeded") {
    return <StagePanel title="Outputs" stages={stages.filter((s) => NODE_STAGES.outputs.includes(s.name))} logs={logs} run={run} />;
  }
  const promote = async () => {
    if (!dest) return;
    const extra = unapplied ? `\n\n${unapplied} approved review row(s) not yet in the file will be added first.` : "";
    if (!confirm(`Overwrite\n${dest}\nwith this run's updated mapping? A timestamped backup is kept.${extra}`)) return;
    setBusy(true);
    setMsg(null);
    try {
      const r = await client.promoteMaster(run.id, dest);
      setMsg(`Promoted → ${r.destination}${r.backup ? ` (backup: ${r.backup})` : ""}`);
    } catch (e) {
      setMsg(describeError(e));
    } finally {
      setBusy(false);
    }
  };
  const um = summary?.updated_master;
  return (
    <>
      <div>
        <h4>Updated mapping</h4>
        {um ? (
          <p>
            {um.rows.toLocaleString()} rows = <strong>{um.original_rows.toLocaleString()} came in</strong> as the mapping +{" "}
            <strong>{um.added_rows.toLocaleString()} added by the pipeline</strong> (direct + fuzzy · high probability)
            {um.by_source?.reviewed_approved ? <> + <strong>{um.by_source.reviewed_approved.toLocaleString()} approved in review</strong></> : null}
            {um.by_source?.mapped_by_hand ? <> + <strong>{um.by_source.mapped_by_hand.toLocaleString()} mapped by hand</strong></> : null}
            {um.approved_rows && !um.by_source ? <> + <strong>{um.approved_rows.toLocaleString()} approved or mapped by hand</strong></> : null}
            . The existing mapping always wins on conflicts.
          </p>
        ) : <p className="muted">Not produced.</p>}
        {um && <MasterProvenance run={run} client={client} />}
        <div className="row gap">
          <input className="grow" value={dest} onChange={(e) => setDest(e.target.value)} placeholder="Destination mapping CSV" />
          {native.inTauri && (
            <button className="btn" onClick={async () => {
              const p = await native.pickSaveFile("Save updated mapping as…", "Master Mapping.csv");
              if (p) setDest(p);
            }}>Choose…</button>
          )}
          <button className="btn btn-primary" disabled={!dest || busy || !um} onClick={promote}>
            {busy ? "Promoting…" : "Promote as new mapping"}
          </button>
        </div>
        {unapplied ? <p className="small muted">{unapplied} approved review row(s) are not in the file yet — promoting adds them.</p> : null}
        {msg && <p className="small">{msg}</p>}
      </div>
      {summary && (
        <div>
          <h4>Step timings</h4>
          <Bars data={Object.fromEntries(Object.entries(summary.stage_seconds)
            .filter(([k]) => !/^(clustered|merged_cluster|suggested_mapping)_results$/.test(k))
            .map(([k, v]) => [stages.find((s) => s.name === k)?.label ?? k, v]))} total={summary.total_seconds} unit="s" />
        </div>
      )}
      <OutputsTab run={run} client={client} />
    </>
  );
}
