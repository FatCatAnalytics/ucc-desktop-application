import { invoke } from "@tauri-apps/api/core";
import type {
  ApplyResult, Bank, FinalizeResult, Bucket, DecisionCounts, DecisionItem, DecisionsPage, Health, LogLine, ModelsStatus, ModelsSync, OutputFile, Page, ReviewGroup, RunSnapshot,
  SourcePreview, SourceSpec, StoreCounts, Summary,
} from "./types";

const inTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

interface EngineInfo {
  url: string | null;
  error: string | null;
  source: string | null;
}

/** Resolve the engine base URL: Tauri sidecar port, or VITE_ENGINE_URL when running plain `vite`. */
export async function resolveEngineUrl(timeoutMs = 60_000): Promise<string> {
  const fallback = import.meta.env.VITE_ENGINE_URL as string | undefined;
  if (!inTauri) {
    if (fallback) return fallback.replace(/\/$/, "");
    throw new Error("Not running inside Tauri and VITE_ENGINE_URL is not set.");
  }
  const started = Date.now();
  let lastError: string | null = null;
  while (Date.now() - started < timeoutMs) {
    const info = await invoke<EngineInfo>("engine_url");
    if (info.url) return info.url;
    if (info.error) lastError = info.error;
    if (lastError && !lastError.startsWith("Engine exited")) break;
    await new Promise((r) => setTimeout(r, 400));
  }
  throw new Error(lastError ?? "Timed out waiting for the mapping engine to start.");
}

export class EngineClient {
  constructor(public readonly base: string) {}

  private async req<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await fetch(this.base + path, {
      ...init,
      headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
    });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
      } catch {
        /* keep statusText */
      }
      throw new Error(`${res.status} ${detail}`);
    }
    return (await res.json()) as T;
  }

  health = () => this.req<Health>("/health");
  configDefaults = () => this.req<{ defaults: Record<string, unknown>; schema: unknown[] }>("/config/defaults");
  prefetchModels = (overrides: Record<string, unknown> = {}) =>
    this.req<Record<string, string>>("/models/prefetch", { method: "POST", body: JSON.stringify(overrides) });

  modelsStatus = (q: { model_share_dir?: string; model_cache_dir?: string }) => {
    const p = new URLSearchParams();
    if (q.model_share_dir) p.set("model_share_dir", q.model_share_dir);
    if (q.model_cache_dir) p.set("model_cache_dir", q.model_cache_dir);
    return this.req<ModelsStatus>(`/models/status?${p}`);
  };
  syncModels = (overrides: Record<string, unknown> = {}) =>
    this.req<ModelsSync>("/models/sync", { method: "POST", body: JSON.stringify(overrides) });

  previewSource = (source: SourceSpec) =>
    this.req<SourcePreview>("/sources/preview", { method: "POST", body: JSON.stringify({ source }) });

  listRuns = () => this.req<RunSnapshot[]>("/runs");
  getRun = (id: string) => this.req<RunSnapshot>(`/runs/${id}`);
  deleteRun = (id: string) => this.req<{ deleted: boolean }>(`/runs/${id}`, { method: "DELETE" });
  cancelRun = (id: string) => this.req<RunSnapshot>(`/runs/${id}/cancel`, { method: "POST" });
  startRun = (body: { source: SourceSpec; config: Record<string, unknown>; sample_rows: number; name: string }) =>
    this.req<RunSnapshot>("/runs", { method: "POST", body: JSON.stringify(body) });

  logs = (id: string, after = 0) =>
    this.req<{ lines: LogLine[]; last_seq: number; status: string }>(`/runs/${id}/logs?after=${after}&limit=1000`);
  summary = (id: string) => this.req<Summary>(`/runs/${id}/summary`);
  outputs = (id: string) => this.req<OutputFile[]>(`/runs/${id}/outputs`);

  review = (id: string, q: { confidence?: string[]; search?: string; method?: string; decision?: string; mapped?: string; offset?: number; limit?: number }) => {
    const p = new URLSearchParams();
    (q.confidence ?? []).forEach((c) => p.append("confidence", c));
    if (q.search) p.set("search", q.search);
    if (q.method) p.set("method", q.method);
    if (q.decision) p.set("decision", q.decision);
    if (q.mapped != null) p.set("mapped", q.mapped);
    p.set("offset", String(q.offset ?? 0));
    p.set("limit", String(q.limit ?? 100));
    return this.req<Page>(`/runs/${id}/review?${p}`);
  };

  /** Read-only page of one result bucket. */
  rows = (id: string, q: { bucket: Bucket; search?: string; decision?: "pending" | "mapped"; offset?: number; limit?: number }) => {
    const p = new URLSearchParams();
    p.set("bucket", q.bucket);
    if (q.search) p.set("search", q.search);
    if (q.decision) p.set("decision", q.decision);
    p.set("offset", String(q.offset ?? 0));
    p.set("limit", String(q.limit ?? 100));
    return this.req<Page & { mapped_total?: number }>(`/runs/${id}/rows?${p}`);
  };

  clusters = (id: string, q: { search?: string; min_size?: number; offset?: number; limit?: number }) => {
    const p = new URLSearchParams();
    if (q.search) p.set("search", q.search);
    p.set("min_size", String(q.min_size ?? 1));
    p.set("offset", String(q.offset ?? 0));
    p.set("limit", String(q.limit ?? 100));
    return this.req<Page>(`/runs/${id}/clusters?${p}`);
  };

  /** Banks a filer can be mapped to by hand (tracked banks + the master's banks). */
  banks = (id: string) => this.req<Bank[]>(`/runs/${id}/banks`);
  /** The updated master with provenance: what came in vs what was added. */
  masterRows = (id: string, q: { source?: string; search?: string; offset?: number; limit?: number }) => {
    const p = new URLSearchParams();
    p.set("source", q.source ?? "added");
    if (q.search) p.set("search", q.search);
    p.set("offset", String(q.offset ?? 0));
    p.set("limit", String(q.limit ?? 100));
    return this.req<Page & { by_source: Record<string, number> }>(`/runs/${id}/master?${p}`);
  };
  /** The review queue rolled up by suggested bank, biggest first. */
  reviewGroups = (id: string, q: { decision?: string; search?: string }) => {
    const p = new URLSearchParams();
    p.set("decision", q.decision ?? "pending");
    if (q.search) p.set("search", q.search);
    return this.req<ReviewGroup[]>(`/runs/${id}/review/groups?${p}`);
  };
  propagateDecisions = (id: string) =>
    this.req<{ approved: number; rejected: number; candidates: number; counts: DecisionCounts }>(`/runs/${id}/propagate-decisions`, { method: "POST" });

  decisions = (id: string) => this.req<{ counts: DecisionCounts }>(`/runs/${id}/decisions`);
  setDecisions = (id: string, items: DecisionItem[]) =>
    this.req<{ changed: number; inherited: { approved: number; rejected: number }; counts: DecisionCounts }>(`/runs/${id}/decisions`, { method: "PUT", body: JSON.stringify({ items }) });
  applyDecisions = (id: string) => this.req<ApplyResult>(`/runs/${id}/apply-decisions`, { method: "POST" });

  // The decisions database (every run)
  allDecisions = (q: { decision?: string; search?: string; run_id?: string; offset?: number; limit?: number }) => {
    const p = new URLSearchParams();
    if (q.decision) p.set("decision", q.decision);
    if (q.search) p.set("search", q.search);
    if (q.run_id) p.set("run_id", q.run_id);
    p.set("offset", String(q.offset ?? 0));
    p.set("limit", String(q.limit ?? 100));
    return this.req<DecisionsPage>(`/decisions?${p}`);
  };
  clearDecisions = (keys: string[]) =>
    this.req<{ cleared: number; counts: StoreCounts }>("/decisions/clear", { method: "POST", body: JSON.stringify({ keys }) });
  decisionsExportUrl = () => `${this.base}/decisions/export`;
  exportDecisions = (destination: string) =>
    this.req<{ path: string; rows: number }>("/decisions/export", { method: "POST", body: JSON.stringify({ destination }) });

  finalize = (id: string, period: string, folder: string) =>
    this.req<FinalizeResult>(`/runs/${id}/finalize`, { method: "POST", body: JSON.stringify({ period, folder }) });
  promoteMaster = (id: string, destination: string) =>
    this.req<{ source: string; destination: string; backup: string | null }>(`/runs/${id}/promote-master`, {
      method: "POST",
      body: JSON.stringify({ destination }),
    });
}

// ---- native helpers (Rust commands); no-ops outside Tauri --------------------
export const native = {
  pickFile: (title: string) => (inTauri ? invoke<string | null>("pick_file", { title }) : Promise.resolve(null)),
  pickSaveFile: (title: string, defaultName: string) =>
    inTauri ? invoke<string | null>("pick_save_file", { title, defaultName }) : Promise.resolve(null),
  pickFolder: (title: string) => (inTauri ? invoke<string | null>("pick_folder", { title }) : Promise.resolve(null)),
  reveal: (path: string) => (inTauri ? invoke<void>("reveal_path", { path }) : Promise.resolve()),
  open: (path: string) => (inTauri ? invoke<void>("open_path", { path }) : Promise.resolve()),
  inTauri,
};
