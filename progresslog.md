# Project Progress Log

**Project:** Detection of Coordinated Influence Amplification on Social Media
Using Behavioural, Content and Network Analysis

**Team:** Abhinash Kumar Yadav · Abhisek Karki · Bibek Oli · Kaushal Adhikari

**Last updated:** July 2026

---

## 1. Summary — Where We Are Right Now

| Stage | Status |
|---|---|
| Project structure + environment setup | ✅ Done |
| Benchmark database schema + Reddit database schema (now separated) | ✅ Done |
| Cresci-2017 benchmark data loaded | ✅ Done |
| Benchmark feature engineering / preprocessing | ✅ Done |
| Isolation Forest on benchmark data (anomaly detection) | ✅ Done — **AUC-ROC ≈ 0.84, meets target (≥0.80)** |
| Reddit-specific schema + collector (`db.py` / `collector.py`) | ✅ Done, adopted as team standard |
| Reddit data collection (posts/comments/profiles) | 🔄 In progress — small test batches validated, full-scale run not yet started |
| Reddit feature engineering (`reddit_preprocessor.py`) | ✅ Written, not yet run at scale |
| Isolation Forest on real Reddit data | ✅ Written (`reddit_isolation_forest.py`), not yet run at scale |
| HDBSCAN (coordination clustering) | 🔜 Not started |
| Cosine Similarity (content duplication) | 🔜 Not started |
| NetworkX (influence graph / PageRank) | 🔜 Not started |
| Composite Influence Score | 🔜 Not started |
| Streamlit dashboard | 🔜 Not started |

**Headline result so far:** our behavioral anomaly detector (Isolation Forest)
correctly separates bots from humans in the Cresci-2017 benchmark with an
**AUC-ROC of ≈0.84**, meeting the ≥0.80 target set in our proposal
(Chapter 5, Expected Output) and in line with the baseline reported in
Varol et al. [2]. This validates the pipeline logic; it does not yet say
anything about real Nepal Reddit data, which is the actual point of the
project and is now the active focus.

Since the last update, the project underwent a deliberate, documented pivot
from platform-neutral to **Reddit-specific** data collection (see Section 4),
a teammate-built Reddit schema was adopted as the team standard (Section 3),
and two significant bugs in real Reddit data collection were found, root-caused,
and fixed with verified before/after evidence (Section 7).

---

## 2. Project Structure (current)

```
nepal_influence_detector/
├── data/
│   ├── raw/                    # future scraped data goes here
│   ├── processed/              # cleaned data
│   ├── benchmark/
│   │   └── cresci-2017/        # benchmark dataset (users.csv + tweets.csv per category)
│   ├── benchmark.db            # SQLite DB — Cresci-2017 ONLY (Twitter-style schema)
│   └── influence.db            # SQLite DB — real Reddit data ONLY (db.py schema)
├── src/
│   ├── config.py                     # paths + model parameters. DB_PATH (Reddit) and
│   │                                  # BENCHMARK_DB_PATH (Cresci-2017) are separate constants.
│   ├── db.py                         # Reddit schema + init_db() — team-standard schema (adopted)
│   ├── collector.py                  # Reddit collection via Apify (posts/comments/profiles)
│   ├── reddit_preprocessor.py        # computes all 15 Reddit features into `features` table
│   ├── benchmark_database.py         # Cresci-2017 schema + init — renamed from database.py
│   ├── benchmark_loader.py           # loads Cresci-2017 CSVs — renamed from dataloader.py
│   ├── benchmark_preprocessor.py     # computes benchmark features — renamed from preprocessor.py
│   └── models/
│       ├── isolationforest.py            # benchmark anomaly detection (done, AUC ≈0.84)
│       └── reddit_isolation_forest.py    # real-data anomaly detection (written, not yet run at scale)
├── dashboard/                   # Streamlit app (not started yet)
├── outputs/
│   ├── reports/
│   └── graphs/
└── venv/
```

**Why two separate databases:** `benchmark.db` (Cresci-2017, Twitter-style
columns like `follower_ratio`, `log_followers`) and `influence.db` (real
Reddit data, `db.py`'s schema) used to share a single file and a single
`DB_PATH`. This caused a real bug — see Section 7.1 — where the Reddit
schema silently overwrote the benchmark schema's `features` table. They are
now fully separated at the `config.py` level (`DB_PATH` vs
`BENCHMARK_DB_PATH`), and every benchmark script imports the latter
specifically. This is the single most important structural fix from this
update — **never let a script default back to importing `DB_PATH` if it's
meant to run against the benchmark data**, and vice versa.

---

## 3. Database Schemas

### 3.1 — Benchmark schema (`benchmark_database.py` → `benchmark.db`)

Five tables: `accounts`, `posts`, `interactions`, `features`, `results`.
Twitter/Cresci-style columns throughout (`follower_ratio`, `log_followers`,
`favourites_ratio`, etc. — see Section 5 for the full feature list and why
they don't transfer to Reddit).

### 3.2 — Reddit schema (`db.py` → `influence.db`) — **adopted team standard**

Built by a teammate in parallel with the benchmark work, and adopted as the
team standard for all real Reddit data after evaluation. Considerably more
detailed than the benchmark schema, since it's designed to support the full
pipeline (clustering, network analysis, composite scoring) rather than just
anomaly detection.

Tables: `posts`, `comments`, `accounts` (SHA-256 anonymized IDs, see below),
`account_activity`, `features` (15 Reddit-specific behavioral features —
see Section 5.2), `edges`, `content_similarity`, `temporal_similarity`,
`account_pairs`, `communities`, `scores`, `predictions`,
`coordination_events`, `dataset_metadata`, `model_metrics`, `experiments`.
Well-indexed throughout.

**Anonymization:** `accounts.id` is a SHA-256-derived pseudonym and is what
every other table joins on; it is always populated. `accounts.username` is
left `NULL` by default (`STORE_RAW_USERNAME=false` in `.env`) — storing the
raw username next to the hash would make the "anonymized" claim in the
proposal (Sec 3.4.1) untrue, since the mapping would be trivially reversible
by anyone with DB access. Only ever set `STORE_RAW_USERNAME=true` for local
manual verification, and never commit, share, or expose `influence.db` (or
any export built from it) while it's on.

---

## 4. The Reddit Pivot — What Changed and Why

The project moved from a "platform-neutral" design to **Reddit-specific**,
deliberately, after ruling out the alternatives:

- **YouTube** — no follower/following concept, shallow interaction graphs;
  not enough structure for meaningful network analysis.
- **Twitter/X** — API access restrictions blocked live collection entirely.

This is a documented, defensible design decision, not scope creep — it's
why the benchmark (Cresci-2017, Twitter-style) and the real-world pipeline
(Reddit) now have genuinely different feature sets (see Section 5).

### Collection tool: Apify

Reddit's official API is inaccessible, so collection uses Apify's
`trudax/reddit-scraper-lite` actor via `collector.py`. Key operational
learnings:

- `includeMediaLinks: True` must be set in the actor input, or Apify uses a
  fast RSS path that silently omits `upVotes`, `upVoteRatio`, and
  `numberOfComments` entirely.
- Reddit's own bot defenses (403/429 responses, navigation timeouts) are
  the main obstacle to collection reliability, not our code — see Section 7.2.

### Topics chosen for data collection

Need genuine discourse volume/controversy, since HDBSCAN/cosine-similarity
need shared topical context to find a coordination signal at all (see
Section 9 on what we're actually detecting):

1. Balen Shah government's first 100 days (active political debate)
2. 2025 social media ban / youth protests aftermath (thematically ideal —
   literally about platform manipulation; 75+ deaths, 2,000+ injuries)
3. Constitutional amendments / federalism debate
4. FATF grey list / economic reform (lower-heat "control" topic for comparison)

Subreddits currently in use: `Nepal`, `NepaliPolitics`, `nepalinews`,
`NepalSocial`, `SouthAsia`, `Kathmandu`.

---

## 5. Feature Engineering — Benchmark vs. Reddit

### 5.1 — Benchmark features (Twitter-style, Cresci-2017 only)

| Feature | What it measures | Notes |
|---|---|---|
| `posts_per_day` | total_posts / account_age_days | Universal concept |
| `account_age_days` | days since account creation | Universal concept |
| `is_empty_account` | 0 followers AND 0 posts | Universal concept |
| `log_posts` | log(1 + total_posts) | Universal concept |
| `follower_ratio` | followers / following | Twitter-style only |
| `followers_per_day` | followers / account_age_days | Twitter-style only |
| `log_followers` / `log_following` | log-scaled counts | Twitter-style only |
| `favourites_ratio` | favourites_count / total_posts | Twitter only |
| `listed_ratio` | listed_count / follower_count | Twitter only |
| `log_favourites` | log(1 + favourites_count) | Twitter only |

**Why log-scaling:** raw follower/post counts are heavily skewed. Without
it, Isolation Forest ends up detecting "who has huge numbers" rather than
genuinely unusual *behavioral patterns*. `log1p(x) = log(1 + x)` compresses
extreme outliers so differences at the normal end of the scale stay visible.

**Why StandardScaler on top of log-scaling:** log-scaling fixes skew
*within* a feature; StandardScaler (mean=0, std=1) fixes the fact that
different features live on totally different numeric scales — without it,
whichever feature has the biggest raw numbers dominates every distance
calculation regardless of how meaningful it actually is.

**What moved the benchmark AUC from 0.7361 → 0.84:** adding
`favourites_ratio`, `listed_ratio`, `log_favourites` — engagement features
measuring how much an account *consumes* content (likes/favourites given)
versus how much it *produces* (posts). Real accounts tend to like more than
they post; automated accounts post constantly but rarely show engagement
behavior.

### 5.2 — Reddit features (`reddit_preprocessor.py`, 15 total)

Reddit has no "following" concept and doesn't expose per-user upvote
history via what's scrapable, so none of the Twitter-only features above
transfer. `reddit_preprocessor.py` computes a different, Reddit-appropriate
set instead: `age_days`, `posts_per_day`, `comments_per_day`,
`comment_ratio`, `karma_score`, `avg_score`, `subreddit_count`,
`active_days`, `hour_entropy`, `duplicate_ratio`, `avg_post_interval`,
`avg_comment_interval`, `night_activity_ratio`, `burstiness_score`,
`engagement_rate`.

Notable computations:
- **`burstiness_score`** uses the Goh & Barabási formula:
  `B = (σ_τ - m_τ)/(σ_τ + m_τ)`, requiring ≥3 timestamps.
- **`duplicate_ratio`** measures self-duplication via `content_hash`.
- **`karma_score`** and **`age_days`** depend on profile-page data, which is
  the feature pair most affected by the Bug 2 collection gap (Section 7.2).

### 5.3 — `reddit_isolation_forest.py`

Runs Isolation Forest on real (unlabeled) Reddit data. No `is_bot` ground
truth exists for real data, so this only outputs anomaly scores plus a
ranked list of top suspicious accounts (deliberately chosen over
auto-tiering). `contamination=0.1` is flagged as an assumption requiring
manual spot-check validation, not a measured value — worth re-examining
once a real collection run is in hand, since the true anomalous fraction in
Nepal political Reddit discourse is unknown and may not match Cresci-2017's
76% bot rate at all.

---

## 6. Isolation Forest (Benchmark) — Result and What It Means

**Result: AUC-ROC ≈ 0.84** ✅ (target was ≥ 0.80)

### What AUC-ROC means here
It measures how well the model's anomaly scores *rank* bots above humans,
across every possible decision threshold — not just at one fixed cutoff.
0.5 = random guessing, 1.0 = perfect separation.

### Class imbalance flips the raw result
Cresci-2017 is 76% bots, 24% humans. Isolation Forest is unsupervised — it
flags whichever pattern is statistically *rare*, which here initially meant
flagging **humans** as the anomaly (backwards for our purposes). Raw AUC
came out as 0.2639 until we recognized this and tested both directions:

```
AUC (anomaly=bot):    0.2639 (or 0.1614, seen in a later re-run — direction-dependent)
AUC (anomaly=human):  0.7361 (or 0.8386 after the engagement features)  ← the real signal
```

The code now automatically checks both directions and picks whichever
actually separates the classes correctly. This will not be an issue on real
Reddit data, where bots/coordinated accounts are expected to be a minority
— but it's important to check for on any new dataset, never assume the raw
output is oriented as expected.

### A known remaining weakness
Even at AUC ≈0.84, a naive 0.5 classification threshold still
misclassifies most individual humans as bots (recall on the Human class as
low as 0.09 in one run), even though the overall *ranking* (AUC) is good.
AUC-ROC and threshold-based accuracy/recall are different things — this is
a threshold-calibration problem for later (e.g. precision-recall-curve
tuning), not a sign the model doesn't work. Worth listing explicitly as a
limitation in the final report, and worth being careful not to over-quote
the confusion-matrix numbers alongside AUC without this caveat attached.

---

## 7. Bugs Found and Fixed This Update

### 7.1 — Benchmark and Reddit databases silently colliding

**Symptom:** `isolationforest.py` (benchmark) failed with
`sqlite3.OperationalError: no such column: follower_ratio`, then later
`no such column: favourites_ratio`.

**Root cause:** `config.py` had a single `DB_PATH`, and an older,
independently-hardcoded `DB_PATH` inside `benchmark_database.py` (formerly
`database.py`) happened to resolve to the exact same physical file
(`data/influence.db`) as the Reddit pipeline's `db.py`. Both pipelines were
always pointing at the same file; it just hadn't caused visible damage
until the Reddit schema's `features` table (via `db.py`'s `init_db()`)
effectively replaced the benchmark schema's `features` table, since two
differently-shaped tables can't coexist under the same name.

**Fix:** Added a dedicated `BENCHMARK_DB_PATH` constant in `config.py`,
pointing at a new, separate `data/benchmark.db`. Every benchmark script
(`benchmark_database.py`, `benchmark_loader.py`, `benchmark_preprocessor.py`,
`models/isolationforest.py`) now imports `BENCHMARK_DB_PATH` specifically
instead of the shared `DB_PATH`. Along the way, also caught and fixed: a
leftover reference to the old bare `DB_PATH` name inside
`benchmark_database.py`'s print statement, and a copy-paste duplicate
column-definition bug (`favourites_count`/`listed_count` accidentally
declared twice in the same `CREATE TABLE` statement) introduced while
adding the three missing engagement columns.

**Verified fixed:** rebuilt `benchmark.db` from scratch (14,368 accounts,
6.6M+ tweets, all 11 benchmark features computed) and reproduced
AUC-ROC = 0.8386 — consistent with the original 0.8397 within normal
`IsolationForest` run-to-run variance. `influence.db` (Reddit data)
confirmed untouched throughout.

**Lesson for the team:** a shared `DB_PATH` constant is only safe if every
script that touches the database actually imports it from the same place.
A script that independently recomputes its own path (even if it resolves
to the same file today) is a landmine — it'll keep working right up until
something else changes the schema at that path, with no warning.

### 7.2a — `comments.post_id` was 0% populated

**Symptom:** every comment lost its link back to its parent post — even
top-level comments, which should link directly.

**Root cause (initially assumed, later disproven):** an earlier version of
`collector.py` assumed the actor's comment items exposed no direct
post-reference field (`postId`/`linkId`), based on a reading of the actor's
*documented* schema, and instead tried to resolve `post_id` by walking a
`parentId` chain up to a `t3_`-prefixed (post) id via `resolve_post_id()`.

**Actual root cause, found by adding a debug print of raw comment items and
checking real Apify output directly:** `item.get("postId")` **is** present
on every comment item, already correctly `t3_`-prefixed, for both top-level
comments (confirmed) and nested replies (confirmed — `postId` correctly
points to the thread root regardless of nesting depth). The documented
schema assumption didn't match observed behavior.

**Fix:** `post_id = item.get("postId")` directly, replacing the
`resolve_post_id()` chain-walk. Confirmed via direct DB query
(`SELECT id, post_id, parent_id FROM comments`) that new comments collected
after the fix have `post_id` populated correctly; old rows from before the
fix remain `NULL` (expected — `INSERT OR IGNORE` never retroactively
repairs existing rows; a handful of pre-fix rows were left as-is rather
than manually patched, since the volume was trivial).

**Lesson for the team:** trust real observed data over a data source's
*documented* schema when the two disagree — this is the second time this
exact lesson has come up on this project (see also Section 5.2's
Romanized-Nepali language/sentiment caveats, and Appendix A generally).
When something looks structurally broken, add a debug print of the raw
item and look at it directly before assuming the fix requires complex
logic.

### 7.2b — Account profile data (`created_utc`, `comment_karma`,
`link_karma`) only populated for ~11% of accounts

**Symptom:** `collect_user_profiles()` was failing/timing out on most
individual profile-page fetches, leaving `age_days` and `karma_score` (2 of
15 Reddit features) missing for the large majority of accounts.

**Root cause:** Reddit's own anti-bot defenses (403/429 responses,
60-second navigation timeouts) actively and aggressively block
profile-page scraping specifically — much more so than subreddit-listing
scraping (posts/comments), which came through comparatively reliably
throughout. This is adversarial behavior on Reddit's side, not a bug in our
code, and is not something we can expect to fully eliminate.

**Mitigation applied:** reduced `collect_user_profiles()`'s `batch_size`
from 90 to 15, added a 30-second delay between batches, and bumped
`maxRequestRetries` from 2 to 3. Smaller, spaced-out batches give Apify's
own retry mechanism room to recover from transient 403/429s before giving
up, rather than hammering Reddit with 90 near-simultaneous requests.

**Verified improved, across three separate test runs at increasing scale:**

| Run | Batches | Result |
|---|---|---|
| Original (pre-fix) | — | ~11% of accounts had profile data |
| Test 1 (10 usernames) | 1 | 8/10 updated |
| Test 2 (26 usernames) | 2 | 20/26 updated in-run; delay confirmed firing (`Waiting 30s before next batch...`) |
| Cumulative across all runs | — | **42/49 accounts (≈86%) with profile data** |

**Not fully solved, and shouldn't be expected to be:** even with this
mitigation, some fraction of profile fetches will likely still fail at
full scale (~1,000 accounts), since Reddit's blocking isn't perfectly
deterministic and may behave differently under sustained higher volume
than in these test batches. Decision needed before running
`reddit_preprocessor.py` at scale: how should `age_days`/`karma_score` be
handled for accounts where profile data never resolves (e.g. median
imputation, or excluding those two features from the composite score for
affected accounts only) — this is a design decision to make explicitly,
not something to leave implicit.

**Lesson for the team:** when a third-party scraping target is actively
adversarial (rate limiting, bot detection), the fix is request *pattern*
(smaller batches, spacing, retries) more than request *count* (raising
retry limits alone did little in earlier tests without also shrinking
batch size). Also worth noting for the final report as a documented,
expected methodological limitation rather than something to hide or be
surprised by later.

---

## 8. Key Lessons From Debugging (cumulative, so the team doesn't repeat them)

- **A shared config constant is only safe if every script actually imports
  it from the same place** — see Section 7.1. A script that independently
  recomputes its own path is a landmine.
- **Always run the relevant `*_database.py`/`db.py` init before loading
  data.** If a table doesn't exist yet, inserts can fail and get silently
  swallowed by a broad `except: continue` — the script won't crash, it'll
  just report `0 rows loaded` with no explanation.
- **`INSERT OR IGNORE` silently skips existing rows.** If a bug is fixed in
  how a column is computed but old data isn't wiped first, `INSERT OR
  IGNORE` keeps the old (broken) values forever, because the primary key
  already exists. Rebuild from a clean DB whenever a schema or
  loader/collector bug has been fixed and old rows might be stale.
- **A missing `return` statement returns `None` silently** — caused
  `safe_str()` to corrupt every string field it touched for a long stretch
  of debugging, no exception ever thrown.
- **Match column names exactly between `CREATE TABLE`, `INSERT`, and
  `SELECT`.** Most benchmark-era bugs were simple mismatches between these
  three places, not deep logic bugs.
- **Trust real observed data over a data source's documented schema when
  they disagree** — see Section 7.2a. When something is silently wrong
  (a column is always 0/NULL, or always the same value), check the raw
  source data directly with a throwaway debug print *before* assuming the
  bug is in the pipeline's logic.
- **When a third-party scraper is being actively rate-limited/blocked,
  fix the request pattern, not just the retry count** — see Section 7.2b.
- **Watch for copy-paste duplication bugs** — both a duplicated
  `compute_features()` function (benchmark era) and a duplicated pair of
  column definitions inside one `CREATE TABLE` statement (this update)
  came from the same underlying habit of pasting new code near similarly-
  named existing code instead of directly adjacent to what's being
  extended.

---

## 9. A Note on What We're Actually Detecting

Worth the whole team internalizing this, since it shapes data collection:
we are not measuring "is this topic popular" — we're measuring whether the
*appearance* of consensus around a topic is organic (many independent
people) or manufactured (a coordinated group posing as many). The topic is
just the shared context that makes clustering and duplication scores
meaningful; the accounts and their synchronized behavior are the actual
thing being detected. This is why data collection is topic-focused
(3–5 specific discussions/events, Section 4) rather than random — random
posts give HDBSCAN and cosine similarity nothing meaningful to compare
against.

---

## 10. Next Steps

1. **Scale up Reddit collection** toward the proposal's stated targets
   (5,000 posts / 1,000 unique accounts, Sec 3.4.1). Expect this to take
   real wall-clock time given the batch delays in profile collection —
   plan to kick off and let it run rather than expecting a quick turnaround.
2. **Decide on missing-feature handling** for accounts where profile data
   never resolves (Section 7.2b) before running `reddit_preprocessor.py`
   at scale.
3. **Run `reddit_preprocessor.py` and `reddit_isolation_forest.py`** at
   scale once collection is further along; sanity-check the
   `contamination=0.1` assumption against a manual spot-check of top
   flagged accounts.
4. **HDBSCAN clustering** — group accounts with synchronized posting
   behavior into coordination clusters (Coordination Score, Eq. 4.5).
5. **Cosine Similarity** — TF-IDF content comparison across posts within a
   topic, to detect near-duplicate campaigns.
6. **NetworkX / PageRank** — build the interaction graph and compute
   network influence scores (also populates `edges`/`communities`).
7. **Composite Influence Score** — combine all four signals per the
   proposal's weighting (40% anomaly, 40% coordination, 10% duplication,
   10% network).
8. **Streamlit dashboard** — reads only from the `scores`/`predictions`
   tables; no model computation happens in the dashboard itself.

---

## Appendix A — Detailed Debugging Log (Benchmark Era)

Kept for reference in case similar issues resurface in the Reddit pipeline.

### A.1 — File naming mismatches
Early files were saved as `dataloader.py` and `verifysetup.py` instead of
the conventional `data_loader.py` / `verify_setup.py`. Not itself a bug,
but caused confusion when running commands from memory/instructions that
assumed underscores.

### A.2 — Wrong working directory
Scripts failed with "No such file or directory" when run from the wrong
folder, including config imports (`ModuleNotFoundError: No module named
'config'`) when run from the project root instead of `src/`. Always `cd`
into the correct folder and confirm with `pwd` before running any script,
or adjust `sys.path` explicitly.

### A.3 — Cresci-2017 nested folder structure
The dataset unpacks as a **double-nested** structure:
`cresci-2017/genuine_accounts.csv/genuine_accounts.csv/users.csv` (the
`.csv` suffix is actually a folder name, repeated twice). `find_folder()`
checks the double-nested path first, then falls back to single-nested.

### A.4 — Encoding errors reading tweets.csv
`UnicodeDecodeError` on rows containing non-UTF-8 characters. Fixed by
reading with `encoding='latin-1'` instead of default UTF-8, with row-level
try/except so one bad row doesn't kill the whole load.

### A.5 — `accounts` table accidentally never created
A copy-paste error had a `CREATE TABLE` block commented as "ACCOUNTS
table" but which actually created a table named `features` instead. Caught
by explicitly listing table names via `sqlite_master`.

### A.6 — `safe_str()` missing a return statement
```python
def safe_str(value, default=""):
    try:
        if pd.isna(value):
            return default
    except (ValueError, TypeError):
        return default
    # no return here — silently returned None for every valid value
```
Silently corrupted `created_at`, `username`, and `lang` for all 14,368
accounts, no exception ever thrown. Caught by comparing raw pandas values
against what ended up in the database.

### A.7 — `INSERT OR IGNORE` masking stale data after bug fixes
Old broken rows persisted after loader fixes because `account_id` already
existed as a primary key. Fixed each time via a full DB rebuild, not just
re-running the loader on top of old data.

### A.8 — Duplicated function definitions merging incorrectly
`preprocessor.py` once contained two full copies of `compute_features()`
pasted one after another, producing confusing partial output from both
blocks in the same run. Fixed by deleting the file and pasting one clean
version.

### A.9 — Isolation Forest direction inversion due to class imbalance
Covered in Section 6 — a below-random AUC (0.2639) turned out to indicate
correct separation in the opposite direction from assumed, due to bots
being the majority class in Cresci-2017.

### A.10 — Schema drift between CREATE TABLE, INSERT, and SELECT
Recurring theme: new columns added to Python code (`INSERT`/`SELECT`
statements) before the corresponding schema change was actually applied to
the live `.db` file. SQLite does not auto-migrate schemas.

### A.11 — Two independent databases sharing one file path (this update)
See Section 7.1 in full. The most significant structural bug found this
update — worth its own top-level section rather than folding into the
appendix, given how much time it cost and how easily it could recur if
`config.py`'s two DB-path constants aren't kept genuinely separate in every
future script.

### A.12 — `collector.py`'s documented-schema assumption for `post_id`
was wrong (this update)
See Section 7.2a in full. A case where a *reasonable, documented*
assumption (checked against the actor's published schema) still turned out
to be false in practice — reinforcing the lesson that raw data should be
checked directly whenever something looks structurally broken, even when
there's a plausible-sounding explanation already on hand.