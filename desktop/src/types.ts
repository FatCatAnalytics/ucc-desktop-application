export type CsvSource = { type: "csv"; ucc_path: string; master_path: string; universe_path: string };
export type SqlSource = { type: "sql"; url: string; ucc_query: string; master_query: string; universe_query: string };
export type SourceSpec = CsvSource | SqlSource;

export type StageStatus = "pending" | "running" | "done" | "failed" | "skipped";
export interface StageState {
  name: string;
  label: string;
  group: string;
  status: StageStatus;
  started_at: string | null;
  finished_at: string | null;
  seconds: number | null;
  rows: number | null;
}

export type RunStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled" | "unknown";

export interface Summary {
  total_records: number;
  auto_mapped: number;
  review_needed: number;
  new_entities: number;
  no_master_state_pair: number;
  by_match_method: Record<string, number>;
  by_confidence: Record<string, number>;
  clusters: number;
  avg_cluster_size: number;
  records_with_firm_id: number;
  unique_firm_ids: number;
  updated_master?: { rows: number; original_rows: number; added_rows: number; approved_rows?: number; by_source?: Record<string, number>; file: string | null };
  lead_bank?: { rows: number; columns: string[] };
  /** The four pipeline buckets every result row lands in (see engine `bucket_series`). */
  buckets?: Record<Bucket, number>;
  review_bands?: Record<string, number>;
  stage_seconds: Record<string, number>;
  total_seconds: number;
}

export type Bucket = "direct" | "fuzzy_high" | "review" | "unmatched";

export interface RunSnapshot {
  id: string;
  name: string;
  status: RunStatus;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  progress: number;
  current_stage: string | null;
  sample_rows: number;
  input_rows: Record<string, number>;
  source: SourceSpec;
  run_dir: string;
  output_dir: string;
  stages: StageState[];
  summary: Summary | null;
  config?: Record<string, unknown>;
}

export interface LogLine {
  seq: number;
  ts: string;
  level: string;
  stage: string;
  message: string;
}

export interface Page {
  total: number;
  offset: number;
  limit: number;
  columns?: string[];
  rows: Record<string, unknown>[];
  decision_counts?: DecisionCounts;
}

export type Decision = "approved" | "rejected";

/** Decision status of one run's review rows (+ what the master still lacks, + the global store totals). */
export interface DecisionCounts {
  approved: number;
  rejected: number;
  pending: number;
  total: number;
  unapplied_approved: number;
  store: StoreCounts;
}

export interface StoreCounts {
  approved: number;
  rejected: number;
  total: number;
}

/** One row of the decisions database. */
export interface DecisionRecord {
  key: string;
  debt_state: string | null;
  secp_state: string | null;
  secp_name: string;
  decision: Decision;
  suggested: string | null;
  mapped_name: string | null;
  firm_id: number | null;
  match_method: string | null;
  match_score: number | null;
  confidence: string | null;
  run_id: string | null;
  decided_at: string;
}

export interface DecisionsPage {
  total: number;
  offset: number;
  limit: number;
  rows: DecisionRecord[];
  counts: StoreCounts;
}

export interface DecisionItem {
  key: string;
  decision: Decision | null;
  mapped_name?: string | null;
  /** manual mapping of an unmatched row: also apply to this filer name wherever it is unmatched */
  all_states?: boolean;
}

export interface Bank {
  name: string;
  firm_id: number | string | null;
  source: "universe" | "master";
}

export interface ApplyResult {
  approved: number;
  added: number;
  file: string;
  rows: number;
  counts: DecisionCounts;
}

export interface OutputFile {
  name: string;
  path: string;
  size_bytes: number;
  modified: string;
}

export interface InputPreview {
  present: boolean;
  path?: string;
  error?: string;
  size_bytes?: number;
  rows?: number;
  columns?: string[];
  missing_columns?: string[];
  preview?: Record<string, string>[];
}

export interface SourcePreview {
  type: "csv" | "sql";
  url?: string;
  inputs: Record<"ucc" | "master" | "universe", InputPreview>;
}

export interface Health {
  status: string;
  version: string;
  pid: number;
  data_dir: string;
  decisions_db: string;
  decisions: StoreCounts;
  backends: Record<string, boolean>;
  active_run: string | null;
}

/** GET /models/status */
export interface ModelsStatus {
  share: string;
  share_reachable: boolean;
  share_models: string[];
  local: string;
  local_models: string[];
  last_synced: string | null;
  last_synced_from: string | null;
}

/** POST /models/sync */
export interface ModelsSync {
  share: string;
  local: string;
  models: string[];
  copied: number;
  bytes: number;
  seconds: number;
  status: "updated" | "up_to_date" | "share_unreachable";
}

/** GET /runs/{id}/review/groups — one row per suggested bank. */
export interface ReviewGroup {
  mapped_name: string;
  firm_id: number | string | null;
  keys: number;
  filings: number;
  pending: number;
  approved: number;
  rejected: number;
  methods: Record<string, number>;
  min_score: number | null;
  max_score: number | null;
}
