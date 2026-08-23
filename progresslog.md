# Project Progress Log

**Project:** Detection of Coordinated Influence Amplification on Social Media
Using Behavioural, Content and Network Analysis

**Team:** Abhinash Kumar Yadav · Abhisek Karki · Bibek Oli · Kaushal Adhikari


---

## 1. Summary — Where We Are Right Now

| Phase | Folder(s) | Status |
|---|---|---|
| Phase 1 — Project Setup & Configuration | `src/` (`config.py`) | ✅ Done |
| Phase 2 — Database Design & Schema | `src/db.py`, `src/benchmarks/legacy/` | ✅ Done, now migratable |
| Phase 3 — Benchmark Validation (Cresci-2017) | `src/benchmarks/cresci/`, `src/benchmarks/legacy/`, `src/models/isolationforest.py` | ✅ Core result — **AUC-ROC ≈ 0.84**; expanded analyses added |
| Phase 4 — Reddit Data Collection | `src/collector.py`, `src/utils/`, `src/tools/`, `src/tools/dev/` | 🔄 In progress — collector substantially hardened |
| Phase 5 — Feature Engineering & Detection Models | `src/pipeline/reddit_preprocessor.py`, `src/models/` | ✅ Written — 4 original signals + 1 new (temporal) |
| Phase 6 — Evidence Fusion & Composite Scoring | `src/models/` (pairs/community), `src/pipeline/composite_score.py`, `src/run_pipeline.py` | ✅ Written — new layer beyond original proposal |
| Phase 7 — Visualization / Dashboard | `src/dashboard.py` | ✅ Written |
| Phase 8 — Diagnostics & QA Tooling | `src/tools/` | ✅ Written — new this pass |

**Headline result so far:** the behavioral anomaly detector (Isolation
Forest) correctly separates bots from humans in the Cresci-2017
benchmark with an **AUC-ROC ≈ 0.84**, meeting the ≥0.80 target. Since
that result, the project has moved from "benchmark validated, Reddit
pipeline partially built" to **every phase of the pipeline having real
code behind it** — HDBSCAN, cosine similarity, temporal coordination,
NetworkX, evidence fusion, community detection, the composite score, and
the dashboard are all implemented. No folder in `src/` is at "not
started" anymore.

---

## 2. Phase 1 — Project Setup & Configuration

**Status: ✅ Done**

### 2.1 — 📁 `src/`
`config.py` is the central configuration file. It defines:
- `DATA_DIR`, `OUTPUTS`, `DB_PATH` (Reddit) and `BENCHMARK_DB_PATH`
  (Cresci-2017) as separate constants — the fix for the DB-collision bug
  found earlier in the project.
- `REDDIT_COLLECTION` — subreddits, search terms, and collection targets
  (`target_posts: 8000`, `target_accounts: 1500`).
- `ISOLATION_FOREST`, `HDBSCAN_PARAMS`, `COSINE_THRESHOLD`, `WEIGHTS`,
  `TIERS` — tunable parameters for every downstream model.
- `validate_config()` / `ensure_runtime_ready()` — checks scoring weights
  sum to 1.0 and model parameters are in valid ranges before anything
  runs.

---

## 3. Phase 2 — Database Design & Schema

**Status: ✅ Done, actively evolving**

### 3.1 — 📁 `src/`
`db.py` defines every table the pipeline touches: `posts`, `comments`,
`accounts`, `account_activity`, `features`, `edges`,
`content_similarity`, `temporal_similarity`, `account_pairs`,
`communities`, `scores`, `predictions`, `coordination_events`,
`dataset_metadata`, `model_metrics`, `experiments`, and a new
`collection_runs` table (fetched/inserted/duplicate/rejected counts per
collection run, plus `status`/`error_message`).

**New this pass — `_migrate_schema()`:** SQLite's
`CREATE TABLE IF NOT EXISTS` doesn't add columns to an existing table.
This function checks `PRAGMA table_info(...)` per table and idempotently
adds anything missing (`scores.temporal_score`,
`scores.network_score_topic_scoped`, `scores.evidence_status`,
`scores.confidence_level`, `scores.assessment`,
`account_pairs.network_volume_score` / `_reciprocity_score` /
`_concentration_score`, three new `coordination_events` columns), so
existing on-disk databases evolve safely instead of needing a manual
rebuild every time a new signal is added.

### 3.2 — 📁 `src/benchmarks/legacy/`
`benchmark_database.py` — the separate schema for `benchmark.db`
(Twitter-style Cresci-2017 columns), preserved from the original
benchmark/Reddit database split.

---

## 4. Phase 3 — Benchmark Validation (Cresci-2017)

**Status: ✅ Core result — AUC-ROC ≈ 0.84; benchmark tooling substantially expanded**

### 4.1 — 📁 `src/benchmarks/cresci/` (current, active package)
- `database.py`, `features.py`, `final_features.py` — schema, feature
  computation, and the finalized feature set behind the verified AUC.
- `split.py` — a real train/validation/test split, persisted in the
  database.
- `temporal.py` — temporal-similarity-style features
  (`mean_temporal_similarity`, `high_coordination_count`,
  `temporal_coordination_score`) computed for the **benchmark** dataset
  too — the same idea used later on real Reddit data (Phase 5).
- `ablation.py` — a RandomForest-based feature ablation study, to see
  which features actually drive the result.
- `category_evaluation.py` — performance broken out by Cresci-2017's
  actual bot categories (`fake_followers`, `social_spambots_1/2/3`,
  `traditional_spambots_1-4`) rather than one pooled number.
- `isolation_forest_evaluation.py` — a fuller evaluation harness
  (`Pipeline`, `SimpleImputer`, ROC-curve export, `joblib` model
  persistence) than the original script.
- `train.py` — training entry point.

### 4.2 — 📁 `src/benchmarks/legacy/`
`benchmark_loader.py`, `benchmark_preprocessor.py`, `validate.py`,
`cresci_train.py`, `cresci_evaluate.py`, `cresci_temporal.py`,
`cresci_temporal_merge.py`, `cresci_compare_models.py`.

### 4.3 — 📁 `src/models/`
`isolationforest.py` — the benchmark-facing Isolation Forest script,
imports `BENCHMARK_DB_PATH` specifically.

`ablation.py`, `category_evaluation.py`, and the expanded
`isolation_forest_evaluation.py` extend the benchmark analysis beyond the
single pooled AUC number.

---

## 5. Phase 4 — Reddit Data Collection

**Status: 🔄 In progress — collector substantially hardened this pass**

### 5.1 — 📁 `src/`
`collector.py` (76KB) is where most of this phase's work sits:
- **Anonymization fix:** `STORE_RAW_USERNAME` flag removed entirely —
  `accounts.username` is now unconditionally `NULL`. Raw usernames
  (needed only for account-history mode) now go to a separate,
  local-only, git-ignored mapping file that nothing else in the pipeline
  reads.
- `comments.post_id` read directly from `item.get("postId")`, confirmed
  against live output (retained from an earlier fix).
- Comment `subreddit` now read from `"category"` (comments don't carry
  `parsedCommunityName` the way posts do) — posts and comments now agree
  on subreddit formatting.
- Self-reply exclusion fixed — the old `author`/`parentAuthor` check
  compared a field that doesn't exist in the actor's output; replaced
  with an id→account-id map resolved from the batch + DB.
- `posts.sentiment` / `comments.sentiment` now actually computed (VADER,
  English-only).
- `backfill_enrichment()` (new) — recomputes topic/language/sentiment for
  rows already in the DB from before this rewrite, from stored text
  only, no re-scraping.
- `collect_user_profiles()` / `save_user_profiles()` — populates
  `created_utc`/`comment_karma`/`link_karma`, with the batching/delay
  mitigation for Reddit's anti-bot defenses.

### 5.2 — 📁 `src/utils/`
`anonymization.py` (new) — the SHA-256 hashing logic pulled out of
`collector.py` into its own reusable module.

### 5.3 — 📁 `src/tools/`
`merge_databases.py` (new) — merges teammates' independently-collected
databases, raw tables only (`posts`, `comments`, `accounts`,
`dataset_metadata`); derived tables are recomputed separately.
`reset_db.py`, `checkschema.py`, `debug_check.py` — DB reset/introspection
utilities.

### 5.4 — 📁 `src/tools/dev/`
`test_profile_retry.py` — test script for the profile-scrape
batching/delay logic.

**Still to reach:** the proposal's target of 5,000 posts / 1,000 accounts,
or `config.py`'s target of 8,000 posts / 1,500 accounts.

---

## 6. Phase 5 — Feature Engineering & Detection Models

**Status: ✅ Written for all four original signals, plus one new one**

### 6.1 — 📁 `src/pipeline/`
`reddit_preprocessor.py` — computes the 15 Reddit behavioral features
(`age_days`, `posts_per_day`, `comments_per_day`, `comment_ratio`,
`karma_score`, `avg_score`, `subreddit_count`, `active_days`,
`hour_entropy`, `duplicate_ratio`, `avg_post_interval`,
`avg_comment_interval`, `night_activity_ratio`, `burstiness_score`,
`engagement_rate`) into the `features` table.

### 6.2 — 📁 `src/models/`
- **`reddit_isolation_forest.py`** — anomaly detection on real
  (unlabeled) Reddit data; ranked anomaly scores, no ground truth so no
  AUC is possible here the way it is for the benchmark.
- **`reddit_hdbscan.py`** *(new)* — coordination clustering on 7
  behavioral features. Sparse accounts are **not** forced into clusters:
  below-threshold accounts get `coord_score = NULL` /
  `INSUFFICIENT_DATA`, distinct from analyzed-but-noise
  (`coord_score = 0.0`, `NO_CLUSTER`) and analyzed-and-clustered
  (`coord_score > 0`, `ANALYZED`).
- **`reddit_cosine_similarity.py`** *(new)* — TF-IDF +
  `NearestNeighbors(cosine)` near-duplicate detection, filtered for
  minimum length/word count and Reddit's `[deleted]`/`[removed]`
  placeholders. Writes `content_similarity` pairs and per-account
  `scores.dup_score`.
- **`reddit_content_coordination.py`** *(new, second implementation)* —
  a stricter two-tier content-similarity detector, writing to the
  **same** `content_similarity` table as the script above (open item,
  see Phase 6 notes).
- **`reddit_temporal_coordination.py`** *(new, not in the original
  4-signal plan)* — detects repeated independent activity bursts shared
  by the same account pair across multiple, separated bursts.
- **`reddit_networkx.py`** *(new)* — PageRank over Reddit's reply
  structure, run twice (whole-activity + topic-scoped) into separate
  score columns.
- **`content_diagnostics.py`, `content_threshold_sweep.py`,
  `content_candidate_inspector.py`** *(new, diagnostic-only)* — check
  whether `COSINE_THRESHOLD` is actually supported by the dataset.
- **`inspect_account.py`** *(new)* — CLI dump of everything known about
  one account.

---

## 7. Phase 6 — Evidence Fusion & Composite Scoring

**Status: ✅ Written — new layer beyond the original proposal's scope**

### 7.1 — 📁 `src/models/`
- **`build_account_pairs.py`** *(new)* — fuses content, temporal, and
  network evidence per account pair. Network-only evidence stays
  conservative (capped at 0.40 final score), content-only evidence is
  the strongest single-source signal, and independent evidence sources
  get an explicit convergence bonus (2 sources: ×1.10 + 0.15; 3 sources:
  ×1.20 + 0.25) instead of a plain average.
- **`community_detection.py`** *(new)* — Louvain community detection on
  the fused `account_pairs` evidence graph, not on raw interaction edges.
- **`community_validator.py`** *(new)* — classifies each community's
  evidence strength and composition without claiming to prove intent.
- **`update_coordination_evidence.py`** *(new, deliberately standalone)*
  — a narrower, pair-only evidence summary, explicitly excluded from the
  automated pipeline because it was found to disagree with
  `composite_score.py`'s fuller answer depending on run order.

### 7.2 — 📁 `src/pipeline/`
**`composite_score.py`** *(restructured)* — now a 5-signal weighted
score (anomaly 30%, coordination 25%, temporal 20%, duplication 15%,
network 10%). Missing signals aren't treated as 0 — weights are
renormalized and a `SIGNAL_COVERAGE_FACTORS` confidence penalty applies.
Also computes `evidence_status` (insufficient_data → strong_support) and
a human-readable `assessment` label (likely_organic →
likely_coordinated_influence).

### 7.3 — 📁 `src/`
**`run_pipeline.py`** *(new)* — orchestrates all 9 steps in dependency
order (feature engineering → isolation forest → HDBSCAN → cosine
similarity → content coordination → temporal coordination → networkx →
account pairs → composite score), with in-line reasoning for why
`update_coordination_evidence.py` is excluded.

---

## 8. Phase 7 — Visualization / Dashboard

**Status: ✅ Written — was "Not started" as of the previous update**

### 8.1 — 📁 `src/`
**`dashboard.py`** (~48KB) — Streamlit app, "InfluenceWatch Nepal," 9
tabs: **Overview, Isolation Forest, HDBSCAN, Cosine Similarity,
NetworkX, Coordination Evidence, Investigate, Cresci Benchmark, Data
Coverage**. Reads weights/tiers directly from `composite_score.py` and
`config.py` rather than hardcoding its own copies. Every number is read
from the live database or a saved benchmark output file by design.

**Run command:** `streamlit run src/dashboard.py`, intended after
`python -m src.run_pipeline` has populated the scoring tables.

---

## 9. Phase 8 — Diagnostics & QA Tooling

**Status: ✅ New this pass**

### 9.1 — 📁 `src/tools/`
- **`verify_setup.py`** — checks required packages import and config/DB
  paths resolve before anything else runs.
- **`score_diagnostics.py`** — composite-score distributions, coverage,
  tier/assessment breakdowns, consistency checks. Read-only.
- **`pair_diagnostics.py`** — inspects `account_pairs` output.
- **`evidence_inspector.py`** — `python -m src.tools.evidence_inspector
  ACCOUNT_ID` — explains exactly why an account got its score.
- **`checkschema.py`, `debug_check.py`** — low-level schema/DB-path
  introspection.
- **`reset_db.py`** — wipes `influence.db` for a clean rebuild.
- **`merge_databases.py`** — see Phase 4.

### 9.2 — 📁 `src/tools/dev/`
**`test_profile_retry.py`** — scratch/manual test script, not part of
the production pipeline.

---

## 10. Key Lessons From This Pass

- **Schema drift is now handled by infrastructure, not memory** —
  `_migrate_schema()` (Phase 2) means new columns no longer require a
  manual `ALTER TABLE` or full DB rebuild.
- **A "safe" default can silently break an unrelated feature** —
  `STORE_RAW_USERNAME=false` was the correct anonymization default, but
  it also happened to make account-history mode a silent no-op. The fix
  was recognizing two features shared one field with incompatible
  requirements, not changing the default.
- **A second implementation of the same signal needs a deliberate
  decision, not silent coexistence** — `reddit_cosine_similarity.py` and
  `reddit_content_coordination.py` (Phase 5) both write to
  `content_similarity`; this should be resolved before final results are
  reported.
- **A stale/unused config constant is a landmine even when nothing is
  broken yet** — `config.py`'s `WEIGHTS` (Phase 1) vs.
  `composite_score.py`'s `COMPOSITE_WEIGHTS` (Phase 6) — the same class
  of issue that caused real damage earlier in the project when the DB
  path constants diverged.
- **Independent evidence sources should be rewarded for converging, not
  just averaged** — the explicit convergence bonus in
  `build_account_pairs.py` (Phase 6) is a defensible, citable
  methodological choice worth writing up explicitly.

---

## 11. Next Steps

1. **Run the full pipeline end-to-end** — `python -m src.run_pipeline`
   against `data/influence.db`, then use `src.tools.verify_setup` /
   `score_diagnostics.py` / `pair_diagnostics.py` to sanity-check output.
2. **Resolve the two content-similarity implementations** (Phase 5).
3. **Reconcile the weight/contamination mismatches** (Phase 1) between
   `config.py` and the scripts that should be using it.
4. **Scale up Reddit collection** (Phase 4) toward the proposal's
   targets — the collector is meaningfully more robust now than at the
   last update.
5. **Run the new benchmark analyses** (Phase 3) — `ablation.py`,
   `category_evaluation.py`, `isolation_forest_evaluation.py` — and
   capture real output.
6. **Run the full pipeline on a real, at-scale dataset** (Phases 5–7)
   and capture cluster counts, coordination events, PageRank rankings,
   account-pair evidence, communities, and composite scores for the
   final report.
7. **Decide on missing-feature handling** for accounts where profile
   data never resolves (`age_days`/`karma_score`) — flagged previously,
   no imputation logic found in the current `reddit_preprocessor.py` or
   `reddit_isolation_forest.py`.
8. **Write up the deliberate scope additions** (temporal coordination,
   community detection, evidence fusion — Phase 6) as defensible
   extensions beyond the proposal's four named algorithms, the same way
   the Reddit pivot was documented as a deliberate design decision.