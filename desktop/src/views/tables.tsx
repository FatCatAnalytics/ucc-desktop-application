import { useEffect, useState } from "react";
import { EngineClient, native } from "../api";
import type { Decision, DecisionCounts, OutputFile, Page, ReviewGroup, RunSnapshot } from "../types";

/** Shared tables and widgets used by the pipeline page and the decisions page. */

const BANDS = ["REVIEW_HIGH", "REVIEW_MEDIUM", "REVIEW_LOW"];

export function Stat({ label, value, sub, tone }: { label: string; value: number; sub?: string; tone?: "good" | "warn" }) {
  return (
    <div className={`stat ${tone ?? ""}`}>
      <div className="stat-value">{value.toLocaleString()}</div>
      <div className="stat-label">{label}</div>
      {sub && <div className="muted small">{sub}</div>}
    </div>
  );
}

export function Bars({ data, total, unit = "" }: { data: Record<string, number>; total: number; unit?: string }) {
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...entries.map(([, v]) => v));
  return (
    <div className="bars">
      {entries.map(([k, v]) => (
        <div key={k} className="bar-row">
          <span className="bar-label" title={k}>{k}</span>
          <div className="bar"><div style={{ width: `${(v / max) * 100}%` }} /></div>
          <span className="bar-val">
            {unit === "s" ? `${v.toFixed(1)}s` : v.toLocaleString()}
            {total && unit !== "s" ? <span className="muted"> {((v / total) * 100).toFixed(1)}%</span> : null}
          </span>
        </div>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------- review
type DecisionFilter = "" | "pending" | "approved" | "rejected";
type ReviewMode = "bank" | "rows";
type DecideFn = (items: { key: string; decision: Decision | null; mapped_name?: string | null }[]) => Promise<void>;

const str = (v: unknown) => (v == null ? "" : String(v));
const num = (v: unknown) => (typeof v === "number" ? v.toFixed(3) : str(v));

export function ReviewTab({ run, client }: { run: RunSnapshot; client: EngineClient }) {
  const [mode, setMode] = useState<ReviewMode>("bank");
  const [bands, setBands] = useState<string[]>(BANDS);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<DecisionFilter>("pending");
  const [page, setPage] = useState<Page | null>(null);
  const [groups, setGroups] = useState<ReviewGroup[] | null>(null);
  const [counts, setCounts] = useState<DecisionCounts | null>(null);
  const [offset, setOffset] = useState(0);
  const [tick, setTick] = useState(0);
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const limit = 100;

  // rows mode: one page of the flat queue
  useEffect(() => {
    if (mode !== "rows") return;
    const t = setTimeout(() => {
      client.review(run.id, { confidence: bands, search, decision: filter, offset, limit })
        .then((p) => { setPage(p); if (p.decision_counts) setCounts(p.decision_counts); setLoadError(null); })
        .catch((e) => { setPage(null); setLoadError(describeError(e)); });
    }, 200);
    return () => clearTimeout(t);
  }, [client, run.id, mode, bands, search, filter, offset, tick]);
  // bank mode: the roll-up (+ the counts, from a 1-row page)
  useEffect(() => {
    if (mode !== "bank") return;
    const t = setTimeout(() => {
      Promise.all([client.reviewGroups(run.id, { decision: filter, search }), client.review(run.id, { limit: 1 })])
        .then(([g, p]) => { setGroups(g); if (p.decision_counts) setCounts(p.decision_counts); setLoadError(null); })
        .catch((e) => { setGroups(null); setLoadError(describeError(e)); });
    }, 200);
    return () => clearTimeout(t);
  }, [client, run.id, mode, search, filter, tick]);

  const refresh = () => setTick((t) => t + 1);

  const decide: DecideFn = async (items) => {
    setBusy("decide");
    setMsg(null);
    try {
      const r = await client.setDecisions(run.id, items);
      setEdits((e) => { const n = { ...e }; items.forEach((i) => delete n[i.key]); return n; });
      const inh = r.inherited.approved + r.inherited.rejected;
      if (inh) setMsg(`${r.changed} decided; ${inh} more inherited the same decision (same filer name in another state pair or spelling).`);
      refresh();
    } catch (e) {
      setMsg(describeError(e));
    } finally {
      setBusy("");
    }
  };

  const apply = async () => {
    setBusy("apply");
    setMsg(null);
    try {
      const r = await client.applyDecisions(run.id);
      setMsg(`Added ${r.added.toLocaleString()} approved row(s) to the updated mapping → ${r.file.split(/[\\/]/).pop()} (${r.rows.toLocaleString()} rows).`);
      refresh();
    } catch (e) {
      setMsg(describeError(e));
    } finally {
      setBusy("");
    }
  };

  /** Approve every pending row currently suggesting this bank. */
  const approveBank = async (bank: string) => {
    setBusy("decide");
    setMsg(null);
    try {
      const p = await client.review(run.id, { mapped: bank, decision: "pending", limit: 1000 });
      const items = p.rows.map((r) => ({ key: str(r.key), decision: "approved" as Decision }));
      if (items.length) {
        const r = await client.setDecisions(run.id, items);
        const inh = r.inherited.approved + r.inherited.rejected;
        setMsg(`Approved ${items.length} as ${bank}${inh ? `; ${inh} more inherited` : ""}.`);
      }
      refresh();
    } catch (e) {
      setMsg(describeError(e));
    } finally {
      setBusy("");
    }
  };

  const rows = page?.rows ?? [];
  const undecidedShown = rows.filter((r) => !r.decision);

  return (
    <>
      <div className="row gap wrap">
        <div className="seg small">
          <button className={mode === "bank" ? "on" : ""} onClick={() => setMode("bank")}>By bank</button>
          <button className={mode === "rows" ? "on" : ""} onClick={() => setMode("rows")}>By row</button>
        </div>
        <div className="seg small">
          {([["pending", "To decide"], ["approved", "Approved"], ["rejected", "Rejected"], ["", "All"]] as [DecisionFilter, string][]).map(([v, l]) => (
            <button key={v} className={filter === v ? "on" : ""} onClick={() => { setOffset(0); setFilter(v); }}>
              {l}{counts && v ? ` (${counts[v].toLocaleString()})` : ""}
            </button>
          ))}
        </div>
        {mode === "rows" && BANDS.map((b) => (
          <label key={b} className="chip">
            <input type="checkbox" checked={bands.includes(b)}
              onChange={(e) => { setOffset(0); setBands(e.target.checked ? [...bands, b] : bands.filter((x) => x !== b)); }} />
            {b.replace("REVIEW_", "")}
          </label>
        ))}
        <input className="grow" placeholder="Search secured party / bank…" value={search}
          onChange={(e) => { setOffset(0); setSearch(e.target.value); }} />
      </div>

      <div className="row between wrap decision-bar">
        <span className="small muted">
          {mode === "bank"
            ? <>Each block is one suggested bank. Scan the filer names, untick nothing — <em>Approve all</em> takes the block, or reject the odd one out first. A decision on a filer name also covers the same name in other state pairs and spellings, and is remembered in every later run.</>
            : <>Approve a suggestion to add <em>filer → bank (Firm ID)</em> to the updated mapping. Edit the bank name first if the suggestion is close but not right. Decisions are remembered across runs and propagate to the same filer name elsewhere.</>}
        </span>
        <div className="row gap">
          {mode === "rows" && (
            <button className="btn" disabled={busy !== "" || undecidedShown.length === 0}
              onClick={() => decide(undecidedShown.map((r) => ({ key: str(r.key), decision: "approved" as Decision, mapped_name: edits[str(r.key)] })))}>
              Approve all shown ({undecidedShown.length})
            </button>
          )}
          <button className="btn btn-primary" disabled={busy !== "" || !counts?.unapplied_approved} onClick={apply}>
            {busy === "apply" ? "Adding…" : `Add ${(counts?.unapplied_approved ?? 0).toLocaleString()} approved to updated mapping`}
          </button>
        </div>
      </div>
      {msg && <p className="small">{msg}</p>}
      {loadError && (mode === "bank" ? !groups : !page) && <p className="muted"><LoadFailed error={loadError} onRetry={refresh} /></p>}

      {mode === "bank" ? (
        groups === null ? (!loadError && <p className="muted">Loading…</p>) : groups.length === 0 ? (
          <p className="muted">Nothing {filter === "pending" ? "to decide" : "here"}.</p>
        ) : (
          <div className="bank-groups">
            {groups.map((g) => (
              <BankGroup key={g.mapped_name || "(no suggestion)"} run={run} client={client} group={g} filter={filter} search={search}
                busy={busy} edits={edits} setEdits={setEdits} decide={decide} approveBank={approveBank} tick={tick}
                open={groups.length <= 3} />
            ))}
          </div>
        )
      ) : (
        <>
          <ReviewRows run={run} rows={rows} loading={!page} loadError={loadError} onRetry={refresh}
            busy={busy} edits={edits} setEdits={setEdits} decide={decide} />
          {page && (
            <div className="row between small">
              <span className="muted">{page.total.toLocaleString()} rows, showing {Math.min(offset + 1, page.total)}–{Math.min(offset + limit, page.total)}</span>
              <div className="row gap">
                <button className="btn" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>‹ Prev</button>
                <button className="btn" disabled={offset + limit >= page.total} onClick={() => setOffset(offset + limit)}>Next ›</button>
              </div>
            </div>
          )}
        </>
      )}
    </>
  );
}

function BankGroup({ run, client, group: g, filter, search, busy, edits, setEdits, decide, approveBank, tick, open: initiallyOpen }: {
  run: RunSnapshot; client: EngineClient; group: ReviewGroup; filter: DecisionFilter; search: string; busy: string;
  edits: Record<string, string>; setEdits: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  decide: DecideFn; approveBank: (bank: string) => void; tick: number; open: boolean;
}) {
  const [open, setOpen] = useState(initiallyOpen);
  const [page, setPage] = useState<Page | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  useEffect(() => {
    if (!open) return;
    client.review(run.id, { mapped: g.mapped_name, decision: filter, search, limit: 500 })
      .then((p) => { setPage(p); setLoadError(null); })
      .catch((e) => { setPage(null); setLoadError(describeError(e)); });
  }, [client, run.id, g.mapped_name, filter, search, open, tick]);
  const methods = Object.entries(g.methods).map(([k, v]) => `${k.replace(" (Cross-Encoder Rerank)", " (reranked)")} ${v}`).join(", ");
  return (
    <section className={`bank-group ${open ? "open" : ""}`}>
      <div className="bank-head" onClick={() => setOpen((o) => !o)}>
        <span className="bank-caret">{open ? "▾" : "▸"}</span>
        <span className="bank-name">{g.mapped_name || <em className="muted">no suggestion</em>}</span>
        <span className="muted small">{g.firm_id != null ? `Firm ID ${g.firm_id}` : "no Firm ID"}</span>
        <span className="bank-stats small">
          <strong>{g.keys}</strong> filer{g.keys === 1 ? "" : "s"}, {g.filings.toLocaleString()} filing{g.filings === 1 ? "" : "s"}
          <span className="muted">. {methods}. Scores {num(g.min_score)}–{num(g.max_score)}</span>
        </span>
        {g.pending > 0 && (
          <button className="btn ok" disabled={busy !== ""} onClick={(e) => { e.stopPropagation(); approveBank(g.mapped_name); }}>
            Approve all {g.pending}
          </button>
        )}
      </div>
      {open && (
        <ReviewRows run={run} rows={page?.rows ?? []} loading={!page} loadError={loadError} onRetry={() => setOpen(true)}
          busy={busy} edits={edits} setEdits={setEdits} decide={decide} compact />
      )}
    </section>
  );
}

function ReviewRows({ run, rows, loading, loadError, onRetry, busy, edits, setEdits, decide, compact }: {
  run: RunSnapshot; rows: Record<string, unknown>[]; loading: boolean; loadError: string | null; onRetry: () => void; busy: string;
  edits: Record<string, string>; setEdits: React.Dispatch<React.SetStateAction<Record<string, string>>>; decide: DecideFn; compact?: boolean;
}) {
  return (
    <div className={`table-wrap ${compact ? "inset" : ""}`}>
      <table className="table review">
        <thead>
          <tr>
            <th>Secured party (as filed)</th><th>States</th><th>Suggested mapping</th><th>Method / score</th><th>Firm ID</th><th>Decision</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const key = str(r.key);
            const decision = r.decision as Decision | null;
            const suggested = str(r.SECP_NAME_MAPPED);
            const value = edits[key] ?? str(r.decided_mapped_name) ?? "";
            const inherited = str(r.decision_source).startsWith("inherited:");
            const filings = typeof r.filings === "number" ? r.filings : 1;
            return (
              <tr key={key} className={decision ? `decided ${decision}` : ""}>
                <td title={str(r.SECP_NAME)}>
                  <div className="cell-main">{str(r.SECP_NAME)}{filings > 1 ? <span className="muted small"> ×{filings}</span> : null}</div>
                  {r.matched_master_raw ? <div className="muted small">master: {str(r.matched_master_raw)}</div> : null}
                </td>
                <td className="muted small">{str(r.DEBT_STATE)} → {str(r.SECP_STATE)}</td>
                <td>
                  <input className="inline" value={value || suggested} placeholder={suggested}
                    disabled={decision === "approved" || busy !== ""}
                    onChange={(e) => { const v = e.target.value; setEdits((prev) => ({ ...prev, [key]: v })); }} />
                  {r.decided_mapped_name ? <div className="small muted">suggested: {suggested}</div> : null}
                  {r.previously_rejected ? (
                    <div className="small warn" title="A different suggestion for this filer was rejected before; this one is new.">
                      previously rejected: {str(r.previously_rejected)}
                    </div>
                  ) : null}
                </td>
                <td className="small">
                  <div>{str(r.match_method).replace(" (Cross-Encoder Rerank)", " (reranked)")}</div>
                  <div className="muted">{num(r.match_score)} {str(r.confidence).replace("REVIEW_", "").toLowerCase()}</div>
                </td>
                <td className="small">
                  {str(decision ? r.decided_firm_id : r.Firm_ID) || (
                    <span className="muted" title={decision === "approved" ? "Corrected name not found in the tracked banks – Firm ID left blank" : undefined}>—</span>
                  )}
                </td>
                <td>
                  {decision ? (
                    <div className="row gap">
                      <span className={`pill ${decision}`}
                        title={inherited ? `inherited from ${str(r.decision_source).slice(10)}` : r.decided_in_run && r.decided_in_run !== run.id ? `decided in run ${str(r.decided_in_run)}` : undefined}>
                        {decision}{inherited ? " (inherited)" : r.decided_in_run && r.decided_in_run !== run.id ? " (earlier run)" : ""}
                      </span>
                      <button className="link small" disabled={busy !== ""} onClick={() => decide([{ key, decision: null }])}>undo</button>
                    </div>
                  ) : (
                    <div className="row gap">
                      <button className="btn ok" disabled={busy !== ""}
                        onClick={() => decide([{ key, decision: "approved", mapped_name: edits[key] }])}>Approve</button>
                      <button className="btn" disabled={busy !== ""} onClick={() => decide([{ key, decision: "rejected" }])}>Reject</button>
                    </div>
                  )}
                </td>
              </tr>
            );
          })}
          {!loading && rows.length === 0 && <tr><td colSpan={6} className="muted">No rows.</td></tr>}
          {loading && (
            <tr><td colSpan={6} className="muted">
              {loadError ? <LoadFailed error={loadError} onRetry={onRetry} /> : "Loading…"}
            </td></tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

// ----------------------------------------------------------------- clusters
export function describeError(e: unknown): string {
  const m = e instanceof Error ? e.message : String(e);
  return /failed to fetch|load failed|networkerror|connection refused/i.test(m)
    ? "The mapping engine is not reachable. If you started it yourself, check it is still running; otherwise restart the app."
    : m;
}

export function LoadFailed({ error, onRetry }: { error: string; onRetry: () => void }) {
  return (
    <span className="load-failed">
      <span className="warn">Could not load: {error}</span>{" "}
      <button className="link small" onClick={onRetry}>Retry</button>
    </span>
  );
}

export function DataTable({ page, cols, offset, limit, setOffset }: { page: Page | null; cols: string[]; offset: number; limit: number; setOffset: (n: number) => void }) {
  if (!page) return <p className="muted">Loading…</p>;
  const shown = cols.filter((c) => !page.columns || page.columns.includes(c));
  const fmt = (v: unknown) => (typeof v === "number" && !Number.isInteger(v) ? v.toFixed(3) : v == null ? "" : String(v));
  return (
    <>
      <div className="table-wrap">
        <table className="table">
          <thead><tr>{shown.map((c) => <th key={c}>{c}</th>)}</tr></thead>
          <tbody>
            {page.rows.map((r, i) => (
              <tr key={i}>{shown.map((c) => <td key={c} title={fmt(r[c])}>{fmt(r[c])}</td>)}</tr>
            ))}
            {page.rows.length === 0 && <tr><td colSpan={shown.length} className="muted">No rows.</td></tr>}
          </tbody>
        </table>
      </div>
      <div className="row between small">
        <span className="muted">{page.total.toLocaleString()} rows, showing {offset + 1}–{Math.min(offset + limit, page.total)}</span>
        <div className="row gap">
          <button className="btn" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>‹ Prev</button>
          <button className="btn" disabled={offset + limit >= page.total} onClick={() => setOffset(offset + limit)}>Next ›</button>
        </div>
      </div>
    </>
  );
}

// ------------------------------------------------------------------ outputs
export function OutputsTab({ run, client }: { run: RunSnapshot; client: EngineClient }) {
  const [files, setFiles] = useState<OutputFile[]>([]);
  useEffect(() => { client.outputs(run.id).then(setFiles).catch(() => setFiles([])); }, [client, run.id]);
  const kb = (n: number) => (n > 1_048_576 ? `${(n / 1_048_576).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`);
  return (
    <section className="card">
      <div className="row between">
        <h4>{run.output_dir}</h4>
        {native.inTauri && <button className="btn" onClick={() => native.open(run.output_dir)}>Open folder</button>}
      </div>
      <table className="table compact">
        <thead><tr><th>File</th><th>Size</th><th>Modified</th><th></th></tr></thead>
        <tbody>
          {files.map((f) => (
            <tr key={f.name}>
              <td>{f.name}</td>
              <td>{kb(f.size_bytes)}</td>
              <td className="muted small">{f.modified.replace("T", " ")}</td>
              <td className="row gap">
                {native.inTauri ? (
                  <>
                    <button className="link" onClick={() => native.open(f.path)}>Open</button>
                    <button className="link" onClick={() => native.reveal(f.path)}>Reveal</button>
                  </>
                ) : (
                  <a className="link" href={`${client.base}/runs/${run.id}/outputs/${encodeURIComponent(f.name)}`}>Download</a>
                )}
              </td>
            </tr>
          ))}
          {files.length === 0 && <tr><td colSpan={4} className="muted">No files.</td></tr>}
        </tbody>
      </table>
    </section>
  );
}
