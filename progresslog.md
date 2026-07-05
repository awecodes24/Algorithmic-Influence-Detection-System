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
| SQLite database schema | ✅ Done |
| Cresci-2017 benchmark data loaded | ✅ Done |
| Feature engineering / preprocessing | ✅ Done |
| Isolation Forest (anomaly detection) | ✅ Done — **AUC-ROC 0.8397, meets target (≥0.80)** |
| HDBSCAN (coordination clustering) | 🔜 Not started |
| Cosine Similarity (content duplication) | 🔜 Not started |
| NetworkX (influence graph / PageRank) | 🔜 Not started |
| Composite Influence Score | 🔜 Not started |
| Streamlit dashboard | 🔜 Not started |
| Real-world data collection (Reddit/Apify) | 🔜 Not started |

**Headline result so far:** our behavioral anomaly detector (Isolation Forest)
correctly separates bots from humans in the Cresci-2017 benchmark with an
**AUC-ROC of 0.8397**, which meets the ≥0.80 target set in our proposal
(Chapter 5, Expected Output) and is in line with the baseline reported in
Varol et al. [2].

---

## 2. Project Structure

```
nepal_influence_detector/
├── data/
│   ├── raw/                 # future scraped data (Reddit/Apify) goes here
│   ├── processed/           # cleaned data
│   ├── benchmark/
│   │   └── cresci-2017/     # benchmark dataset (users.csv + tweets.csv per category)
│   └── influence.db         # SQLite database — the single source of truth
├── src/
│   ├── config.py            # paths + model parameters (weights, thresholds)
│   ├── database.py          # creates all 5 tables (run this first, always)
│   ├── reset_db.py          # drops all tables cleanly (use when schema is stale)
│   ├── debug_check.py       # quick sanity check: tables + sample rows
│   ├── dataloader.py        # loads Cresci-2017 CSVs into the database
│   ├── preprocessor.py      # computes universal behavioral features
│   └── models/
│       └── isolationforest.py   # anomaly detection model (done)
├── dashboard/                # Streamlit app (not started yet)
├── outputs/
│   ├── reports/
│   └── graphs/
└── venv/
```

---

## 3. Database Schema (as it stands)

Five tables inside `data/influence.db`:

- **accounts** — one row per account (platform, username, follower/following
  counts, created_at, etc.)
- **posts** — one row per post/tweet (content, engagement counts, timestamps)
- **interactions** — reply/mention/retweet edges *(empty until we build the
  NetworkX step — this is expected, not a bug)*
- **features** — computed behavioral features per account, feeds the models
- **results** — model outputs per account (anomaly score, cluster id,
  final influence score) *(only `anomaly_score` is populated so far)*

**Important:** `database.py` must be run before `dataloader.py`, every time
the database is reset. If a table is missing, every insert into it fails
silently (caught by a broad `except: continue`), so an empty table after a
run usually means `database.py` wasn't run first, not a loader bug.

---

## 4. Data We're Using

### Benchmark (for validating the pipeline)

**Cresci-2017** — chosen because it is the most widely cited bot detection
benchmark in the literature we reviewed, and is small enough to run locally.

| Dataset | Accounts | Label |
|---|---|---|
| genuine_accounts | 3,474 | Human |
| fake_followers | 3,351 | Bot |
| social_spambots_1/2/3 | 991 / 3,457 / 464 | Bot |
| traditional_spambots_1/2/3/4 | 1,000 / 100 / 403 / 1,128 | Bot |
| **Total** | **14,368 accounts**, 6.6M+ tweets | |

Each dataset folder contains `users.csv` (account-level data) and
`tweets.csv` (post-level data, where available).

**Note on dataset age:** Cresci-2017 predates this project by several years.
We are using it because (a) it's still the standard benchmark cited by
current bot-detection papers, and (b) our features are *behavioral*
(posting frequency, follower ratios, account age patterns) rather than
content-based, so they remain valid signals regardless of dataset year. Our
real-world validation will come from 2026 Nepal Reddit data, which is what
makes the project current — Cresci-2017 only proves the pipeline works.

We deliberately did **not** use TwiBot-22 (the newer, larger benchmark) —
it's 7GB+, ~1 million accounts, and requires graph neural networks to use
properly. It needs lab-grade hardware, not a student laptop. If we want a
second, more recent benchmark later, **Midterm-2018** or **Covid-2020**
(same IU Bot Repository) are much more practical alternatives.

### Real-world data (planned, not started)

Reddit via PRAW/Apify. YouTube was considered and rejected — its comment
graph is too shallow for meaningful PageRank, and it lacks a "following"
mechanic entirely, unlike Reddit which at least has karma and subreddit
activity we can work with.

---

## 5. Feature Engineering — What We Compute and Why

All features are computed in `preprocessor.py` from raw account/post data.

| Feature | What it measures | Cross-platform? |
|---|---|---|
| `posts_per_day` | total_posts / account_age_days | ✅ Universal |
| `account_age_days` | days since account creation | ✅ Universal |
| `is_empty_account` | 0 followers AND 0 posts | ✅ Universal |
| `log_posts` | log(1 + total_posts) | ✅ Universal |
| `follower_ratio` | followers / following | ⚠️ Twitter-style only |
| `followers_per_day` | followers / account_age_days | ⚠️ Twitter-style only |
| `log_followers` / `log_following` | log-scaled counts | ⚠️ Twitter-style only |
| `favourites_ratio` | favourites_count / total_posts | ⚠️ Twitter only |
| `listed_ratio` | listed_count / follower_count | ⚠️ Twitter only |
| `log_favourites` | log(1 + favourites_count) | ⚠️ Twitter only |

**Why log-scaling:** raw follower/post counts are heavily skewed (a handful
of viral accounts vs. thousands of ordinary ones). Without log-scaling, a
model like Isolation Forest ends up only detecting "who has huge numbers"
rather than genuinely unusual *behavioral patterns*, which is what we
actually want to detect. `log1p(x) = log(1 + x)` compresses the extreme
outliers so differences at the normal end of the scale become visible again.

**Why StandardScaler on top of log-scaling:** these solve two different
problems. Log-scaling fixes skew *within* a single feature. StandardScaler
(mean=0, std=1) fixes the fact that different features live on totally
different numeric scales (`account_age_days` can be in the thousands,
`is_empty_account` is just 0 or 1) — without it, whichever feature happens
to have the biggest raw numbers would dominate every distance calculation,
regardless of how meaningful it actually is.

**Important limitation to flag for the team:** the Twitter-only features
above (marked ⚠️) will not exist for Reddit data — Reddit has no
"following" concept and likes/upvotes given by a user aren't publicly
exposed via the API. When we build the Reddit loader, these columns will
either be left at their default (0.0) or replaced with Reddit-specific
equivalents (e.g. `karma_ratio`, `karma_per_day`). The four "Universal"
features above are the ones guaranteed to carry real signal on any platform.
This is a deliberate design decision, not an oversight — worth stating
explicitly in the final report as a limitation/future-work item.

---

## 6. Isolation Forest — Result and What It Means

**Result: AUC-ROC = 0.8397** ✅ (target was ≥ 0.80)

### What AUC-ROC means here
It measures how well the model's anomaly scores *rank* bots above humans,
across every possible decision threshold — not just at one fixed cutoff.
0.5 = random guessing, 1.0 = perfect separation.

### A real problem we hit and had to understand: class imbalance flips the result

Cresci-2017 is **76% bots, 24% humans**. Isolation Forest is unsupervised —
it doesn't know what "bot" means, it just flags whichever pattern is
statistically *rare*. Since bots are the majority in this dataset, the
model's raw output initially flagged **humans** as the anomaly (the rare
group), which is backwards for our purposes. Our first raw AUC came out as
0.2639 — worse than random — until we recognized this and tested the
score in both directions:

```
AUC (anomaly=bot):    0.2639
AUC (anomaly=human):  0.7361   ← this is the "real" signal, just inverted
```

The code now automatically checks both directions and picks whichever one
actually separates the classes correctly. **This will not be an issue on
real-world data**, where bots are expected to be the minority — but it's an
important thing to understand and check for on any new dataset, not just
assume the raw output is oriented the way you expect.

### What moved the score from 0.7361 → 0.8397
Adding three engagement-based features: `favourites_ratio`, `listed_ratio`,
`log_favourites`. These measure how much an account *consumes* content
(likes/favourites given) versus how much it *produces* (posts). Real
accounts tend to like more than they post; automated accounts post
constantly but rarely show engagement behavior. This one addition alone
closed the gap to the target.

### A known remaining weakness
Even at AUC 0.8397, our current 0.5 classification threshold still
misclassifies most individual humans as bots (91% false-positive rate on
the human class specifically), even though the overall *ranking* is good.
AUC-ROC and threshold-based accuracy are different things — this is a
threshold-calibration problem to solve later (e.g. via
precision-recall-curve tuning), not a sign the model doesn't work. Worth
listing as a limitation/future-work item in the final report.

---

## 7. Key Lessons From Debugging (so the team doesn't repeat them)

- **Always run `database.py` before `dataloader.py`.** If a table doesn't
  exist yet, every insert into it fails and gets silently swallowed by a
  broad `except: continue` — the script won't crash, it'll just report
  `0 rows loaded` with no explanation.
- **`INSERT OR IGNORE` silently skips existing rows.** If you fix a bug in
  how a column is computed but don't wipe old data first, `INSERT OR IGNORE`
  will keep the old (broken) values forever, because the `account_id`
  already exists. Use `reset_db.py` whenever a schema or loader bug has
  been fixed, before re-running the pipeline.
- **A missing `return` statement returns `None` silently.** This caused our
  `safe_str()` helper to return `None` for every valid value for a long
  stretch of debugging — no error was thrown, it just silently corrupted
  every string field it touched.
- **Match column names exactly between the CREATE TABLE, the INSERT
  statement, and the SELECT query that reads it back.** Most of the errors
  we hit were simple mismatches between these three places (e.g.
  `follower_ratio` vs `follower_following_ratio`), not deep logic bugs.
- When something is silently wrong (e.g. a column is always 0, or always
  the same value), check the raw source data directly with a throwaway
  pandas script *before* assuming the bug is in the pipeline logic.

---

## 8. Next Steps

1. **HDBSCAN clustering** — group accounts with synchronized posting
   behavior into coordination clusters (this produces the Coordination
   Score from Equation 4.5 in the proposal).
2. **Cosine Similarity** — TF-IDF content comparison across posts within a
   topic, to detect near-duplicate campaigns.
3. **NetworkX / PageRank** — build the interaction graph and compute
   network influence scores (this is also what will finally populate the
   `interactions` table).
4. **Composite Influence Score** — combine all four signals per the
   weighting in the proposal (40% anomaly, 40% coordination, 10%
   duplication, 10% network).
5. **Reddit/Apify data collection** — topic-focused, not random (see
   discussion below), to give HDBSCAN and cosine similarity meaningful
   shared context to compare against.
6. **Streamlit dashboard** — reads only from the `results` table; no model
   computation happens in the dashboard itself.

---

## 9. A Note on What We're Actually Detecting

Worth the whole team internalizing this, since it shapes how we collect
data later: we are not measuring "is this topic popular" — we're measuring
whether the *appearance* of consensus around a topic is organic (many
independent people) or manufactured (a coordinated group posing as many).
The topic is just the shared context that makes clustering and duplication
scores meaningful; the accounts and their synchronized behavior are the
actual thing being detected. This is why data collection needs to be
topic-focused (3–5 specific discussions/events) rather than random —
random posts give HDBSCAN and cosine similarity nothing meaningful to
compare against.

---

## Appendix A — Detailed Debugging Log

This section is a more complete record of specific bugs encountered while
building the pipeline, kept for reference in case similar issues resurface.

### A.1 — File naming mismatches
Early on, files were saved as `dataloader.py` and `verifysetup.py` instead
of the conventional `data_loader.py` / `verify_setup.py`. Not itself a bug,
but caused confusion when running commands from memory/instructions that
assumed underscores. **Lesson:** keep filenames consistent with whatever
you actually import elsewhere.

### A.2 — Wrong working directory
`python src/data_loader.py` failed with "No such file or directory" when
run from the wrong folder. Always `cd` into the project root
(`nepal_influence_detector/`) and confirm with `pwd` before running any
script.

### A.3 — Cresci-2017 nested folder structure
The dataset unpacks as a **double-nested** structure:
```
cresci-2017/genuine_accounts.csv/genuine_accounts.csv/users.csv
```
(the `.csv` suffix is actually a folder name, repeated twice). A
`find_folder()` helper checks the double-nested path first, then falls
back to a single-nested path.

### A.4 — Encoding errors reading tweets.csv
`UnicodeDecodeError` on certain rows containing non-UTF-8 characters
(including one literal PHP error message that got exported into a CSV cell
by mistake, upstream in the original dataset). Fixed by reading with
`encoding='latin-1'` instead of the default UTF-8, and wrapping row-level
parsing in try/except so one bad row doesn't kill the whole load.

### A.5 — `accounts` table accidentally never created
A copy-paste error in `database.py` had a `CREATE TABLE` block *commented
as* "ACCOUNTS table" but which actually created a table named `features`
instead — meaning the real `accounts` table definition was missing
entirely, and `features` had two conflicting definitions fighting each
other. Every subsequent step failed until this was caught by explicitly
listing table names via `sqlite_master`.

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
This silently corrupted `created_at`, `username`, and `lang` for all
14,368 accounts, with no exception ever thrown (Python functions return
`None` by default when no `return` is hit). Caught by directly comparing
values read via raw pandas against what ended up in the database.

### A.7 — `INSERT OR IGNORE` masking stale data after bug fixes
Several times, a bug was fixed in the loader but old broken rows (e.g. with
`created_at = NULL` or `favourites_count = 0`) remained in the database
because `account_id` already existed as a primary key, and `OR IGNORE`
silently skips inserts on conflict. The fix each time was a full reset via
`reset_db.py` → `database.py` → `dataloader.py`, not just re-running the
loader on top of old data.

### A.8 — Duplicated function definitions merging incorrectly
`preprocessor.py` at one point contained two full copies of
`compute_features()` pasted one after another. Python's real behavior in
this situation is subtler than "the second one wins" — code after the
first function's body (but before a second `def` block at the same
indentation) can end up interpreted as still being inside the first
function, producing confusing partial output from both blocks in the same
run. Fixed by deleting the file contents entirely and pasting one clean
version, rather than patching in place.

### A.9 — Isolation Forest direction inversion due to class imbalance
Covered in Section 6 above — the first real (non-bug) modeling issue,
where a below-random AUC (0.2639) turned out to indicate the model was
correctly separating the classes, just in the opposite direction from what
we assumed, due to bots being the majority class in this specific
benchmark.

### A.10 — Schema drift between CREATE TABLE, INSERT, and SELECT
Recurring theme throughout: new columns (e.g. `favourites_ratio`,
`is_bot`, `log_following`) were added to Python code (INSERT statements,
SELECT queries) before the corresponding `ALTER TABLE ... ADD COLUMN`
was actually run against the live database file. SQLite does not
auto-migrate schemas — every new column needs an explicit `ALTER TABLE`
run once against the existing `.db` file, or a full `reset_db.py` +
`database.py` rebuild.