import { useEffect, useState } from "react";
import { EngineClient, native } from "../api";
import type { DecisionRecord, DecisionsPage, Health, RunSnapshot } from "../types";
import { LoadFailed, describeError } from "./tables";

interface Props {
  client: EngineClient;
  health: Health | null;
  runs: RunSnapshot[];
  onOpenRun: (id: string) => void;
}

type Filter = "" | "approved" | "rejected";
const LIMIT = 100;

/** The decisions database: every approve / reject ever made, across all runs. */
export function DecisionsView({ client, health, runs, onOpenRun }: Props) {
  const [filter, setFilter] = useState<Filter>("");
  const [search, setSearch] = useState("");
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState<DecisionsPage | null>(null);
  const [tick, setTick] = useState(0);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    const t = setTimeout(() => {
      client.allDecisions({ decision: filter, search, offset, limit: LIMIT })
        .then((p) => { setPage(p); setLoadError(null); })
        .catch((e) => { setPage(null); setLoadError(describeError(e)); });
    }, 200);
    return () => clearTimeout(t);
  }, [client, filter, search, offset, tick]);

  const runName = (id: string | null) => (id && runs.find((r) => r.id === id)?.name) || id || "—";
  const counts = page?.counts;

  const undo = async (rows: DecisionRecord[]) => {
    if (rows.length > 1 && !confirm(`Clear ${rows.length} decision(s)? They will come back as "to decide" in their runs.`)) return;
    setBusy(true);
    setMsg(null);
    try {
      const r = await client.clearDecisions(rows.map((d) => d.key));
      setMsg(`Cleared ${r.cleared} decision(s).`);
      setTick((t) => t + 1);
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const exportCsv = async () => {
    if (!native.inTauri) {
      window.open(client.decisionsExportUrl(), "_blank");
      return;
    }
    setBusy(true);
    setMsg(null);
    try {
      const dest = await native.pickSaveFile("Export decisions as…", "review_decisions.csv");
      if (!dest) return;
      const r = await client.exportDecisions(dest);
      setMsg(`Exported ${r.rows.toLocaleString()} decision(s) → ${r.path}`);
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const rows = page?.rows ?? [];
  const when = (iso: string) => {
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? iso.slice(0, 16) : d.toLocaleString(undefined, { dateStyle: "short", timeStyle: "short" });
  };

  return (
    <div className="page">
      <header className="page-head">
        <h1>Decisions</h1>
        <p className="muted">
          Every suggestion you approved or rejected, across all runs. Decided filers are skipped in future review queues; approved
          mappings are added to every later updated master until they are in the promoted Master Mapping.
          {health?.decisions_db ? <> Stored in <code>{health.decisions_db}</code>.</> : null}
        </p>
      </header>

      <div className="row gap wrap">
        <div className="seg small">
          {([["", "All"], ["approved", "Approved"], ["rejected", "Rejected"]] as [Filter, string][]).map(([v, l]) => (
            <button key={v} className={filter === v ? "on" : ""} onClick={() => { setOffset(0); setFilter(v); }}>
              {l}{counts ? ` (${(v ? counts[v] : counts.total).toLocaleString()})` : ""}
            </button>
          ))}
        </div>
        <input className="grow" placeholder="Search secured party / mapped name…" value={search}
          onChange={(e) => { setOffset(0); setSearch(e.target.value); }} />
        <button className="btn" disabled={busy || !counts?.total} onClick={exportCsv}>Export CSV</button>
        {native.inTauri && health?.decisions_db && (
          <button className="btn" onClick={() => native.reveal(health.decisions_db)}>Reveal database</button>
        )}
      </div>
      {msg && <p className="small">{msg}</p>}

      <div className="table-wrap">
        <table className="table decisions">
          <thead>
            <tr>
              <th>Decided</th><th>Secured party (as filed)</th><th>States</th><th>Decision</th><th>Mapped name</th><th>Firm ID</th><th>Method / score</th><th>Run</th><th></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((d) => (
              <tr key={d.key} className={`decided ${d.decision}`}>
                <td className="small muted" title={d.decided_at}>{when(d.decided_at)}</td>
                <td title={d.key}><div className="cell-main">{d.secp_name}</div></td>
                <td className="muted small">{d.debt_state ?? "—"} → {d.secp_state ?? "—"}</td>
                <td><span className={`pill ${d.decision}`}>{d.decision}</span></td>
                <td>
                  <div className="cell-main">{d.mapped_name ?? d.suggested ?? <span className="muted">—</span>}</div>
                  {d.mapped_name && d.suggested && d.mapped_name !== d.suggested
                    ? <div className="small muted">suggested: {d.suggested}</div> : null}
                </td>
                <td className="small">{d.firm_id ?? <span className="muted">—</span>}</td>
                <td className="small">
                  <div>{d.match_method ?? ""}</div>
                  <div className="muted">{d.match_score != null ? d.match_score.toFixed(3) : ""}{d.confidence ? ` ${d.confidence.replace("REVIEW_", "").toLowerCase()}` : ""}</div>
                </td>
                <td className="small">
                  {d.run_id && runs.some((r) => r.id === d.run_id)
                    ? <button className="link small cell-main" title={runName(d.run_id)} onClick={() => onOpenRun(d.run_id!)}>{runName(d.run_id)}</button>
                    : <span className="muted">{runName(d.run_id)}</span>}
                </td>
                <td><button className="link small" disabled={busy} onClick={() => undo([d])}>undo</button></td>
              </tr>
            ))}
            {page && rows.length === 0 && <tr><td colSpan={9} className="muted">No decisions yet. Approve or reject suggestions in a run's Review queue.</td></tr>}
            {!page && (
              <tr><td colSpan={9} className="muted">
                {loadError ? <LoadFailed error={loadError} onRetry={() => setTick((t) => t + 1)} /> : "Loading…"}
              </td></tr>
            )}
          </tbody>
        </table>
      </div>
      {page && page.total > 0 && (
        <div className="row between small">
          <span className="muted">{page.total.toLocaleString()} decisions, showing {Math.min(offset + 1, page.total)}–{Math.min(offset + LIMIT, page.total)}</span>
          <div className="row gap">
            <button className="btn" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - LIMIT))}>‹ Prev</button>
            <button className="btn" disabled={offset + LIMIT >= page.total} onClick={() => setOffset(offset + LIMIT)}>Next ›</button>
          </div>
        </div>
      )}
    </div>
  );
}
