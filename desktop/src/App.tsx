import { useCallback, useEffect, useMemo, useState } from "react";
import { EngineClient, resolveEngineUrl } from "./api";
import type { Health, RunSnapshot } from "./types";
import { PipelineView } from "./views/PipelineView";
import { DecisionsView } from "./views/DecisionsView";

type View = "pipeline" | "decisions";

export default function App() {
  const [client, setClient] = useState<EngineClient | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);
  const [runs, setRuns] = useState<RunSnapshot[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [view, setView] = useState<View>("pipeline");

  // Boot: find the engine, then poll health + run list.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const url = await resolveEngineUrl();
        const c = new EngineClient(url);
        for (let i = 0; i < 100; i++) {
          try {
            const h = await c.health();
            if (cancelled) return;
            setHealth(h);
            break;
          } catch {
            await new Promise((r) => setTimeout(r, 300));
          }
        }
        if (!cancelled) setClient(c);
      } catch (e) {
        if (!cancelled) setBootError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const refreshRuns = useCallback(async () => {
    if (!client) return;
    try {
      const [list, h] = await Promise.all([client.listRuns(), client.health()]);
      setRuns(list);
      setHealth(h);
    } catch {
      /* transient */
    }
  }, [client]);

  useEffect(() => {
    if (!client) return;
    refreshRuns();
    const t = setInterval(refreshRuns, 3000);
    return () => clearInterval(t);
  }, [client, refreshRuns]);

  const selectedRun = useMemo(() => runs.find((r) => r.id === selected) ?? null, [runs, selected]);

  const onStarted = (run: RunSnapshot) => {
    setRuns((prev) => [run, ...prev]);
    setSelected(run.id);
    setView("pipeline");
  };
  const onUpdate = useCallback((r: RunSnapshot) => setRuns((prev) => prev.map((x) => (x.id === r.id ? r : x))), []);
  const onDelete = async (r: RunSnapshot) => {
    if (!client) return;
    if (!confirm(`Delete run "${r.name}"?\n\nIts output files are removed. Review decisions are kept (they live in the decisions database, not the run).`)) return;
    try {
      await client.deleteRun(r.id);
      setRuns((prev) => prev.filter((x) => x.id !== r.id));
      if (selected === r.id) setSelected(null);
    } catch (e) {
      alert(e instanceof Error ? e.message : String(e));
    }
  };

  if (bootError) {
    return (
      <div className="boot boot-error">
        <h1>Mapping engine unavailable</h1>
        <pre>{bootError}</pre>
        <p>
          In development, start the engine yourself (<code>ucc-engine serve --port 8765</code>) and launch the app
          with <code>UCC_ENGINE_URL=http://127.0.0.1:8765</code>, or point <code>UCC_ENGINE_CMD</code> at the
          <code> ucc-engine</code> executable.
        </p>
      </div>
    );
  }
  if (!client) {
    return (
      <div className="boot">
        <div className="spinner" />
        <p>Starting the mapping engine…</p>
      </div>
    );
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">UCC</span>
          <span className="brand-text">Mapping pipeline</span>
        </div>
        <button className={`btn wide ${!selected && view === "pipeline" ? "btn-primary" : ""}`}
          onClick={() => { setSelected(null); setView("pipeline"); }}>
          + New run
        </button>
        <div className="runs-list">
          {runs.length === 0 && <p className="muted small">No runs yet.</p>}
          {runs.map((r) => (
            <div
              key={r.id}
              role="button"
              tabIndex={0}
              className={`run-item ${r.id === selected && view === "pipeline" ? "active" : ""}`}
              onClick={() => { setSelected(r.id); setView("pipeline"); }}
              onKeyDown={(e) => { if (e.key === "Enter") { setSelected(r.id); setView("pipeline"); } }}
            >
              <span className={`dot ${r.status}`} />
              <span className="run-name">{r.name}</span>
              <span className="muted small">
                {r.status === "running" ? `${Math.round(r.progress * 100)}%` : r.status}
                {r.sample_rows ? ` · sample ${r.sample_rows.toLocaleString()}` : ""}
                {r.summary?.buckets ? ` · ${r.summary.buckets.review.toLocaleString()} to review` : ""}
              </span>
              {r.status !== "running" && r.status !== "queued" && (
                <button type="button" className="run-delete" title="Delete this run" aria-label={`Delete run ${r.name}`}
                  onClick={(e) => { e.stopPropagation(); onDelete(r); }}>×</button>
              )}
            </div>
          ))}
        </div>
        <button className={`nav-item ${view === "decisions" ? "active" : ""}`} onClick={() => setView("decisions")}>
          <span>Decisions</span>
          <span className="muted small">
            {health?.decisions ? `${health.decisions.approved.toLocaleString()} approved · ${health.decisions.rejected.toLocaleString()} rejected` : "—"}
          </span>
        </button>
        <footer className="sidebar-footer muted small">
          engine v{health?.version} · {health?.backends.fastembed ? "fastembed" : "sentence-transformers"} · local CPU
        </footer>
      </aside>

      <main className="content">
        {view === "pipeline" && (
          <PipelineView client={client} health={health} run={selectedRun} onStarted={onStarted} onUpdate={onUpdate}
            onNewRun={() => setSelected(null)} />
        )}
        {view === "decisions" && (
          <DecisionsView client={client} health={health} runs={runs} onOpenRun={(id) => { setSelected(id); setView("pipeline"); }} />
        )}
      </main>
    </div>
  );
}
