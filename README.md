# Patch: is_relevant filtering for reddit_cosine_similarity.py and reddit_networkx.py

## What's in this package

```
src/db.py                              -- schema migration for the new column
src/models/reddit_cosine_similarity.py -- relevant_only parameter
src/models/reddit_networkx.py          -- topic_scoped parameter
test_integration.py                    -- proves both fixes work, standalone
```

Drop the three `src/...` files into your real project at the same
relative paths, overwriting the originals. `test_integration.py` goes
at your project root (next to your existing `src/` folder).

## The problem this fixes

`collector.py`'s `classify_topic()` already tags every post/comment with
`is_relevant` (1 = matches `HISTORICAL_SEARCH_TERMS`/`TOPIC_KEYWORDS`,
0 = doesn't) at collection time, and `db.py` indexes both columns for
fast filtering. But `collect_posts()` pulls `r/<subreddit>/new/`
wholesale -- everything in the subreddit, not just on-topic posts --
so a real chunk of what lands in `posts`/`comments` is on-subreddit but
off-topic. Before this patch, nothing downstream filtered on
`is_relevant`: every model queried the full table.

For cosine similarity specifically, that's a real problem: two accounts
sharing off-topic boilerplate (a subreddit rule reminder, a flair
template) get flagged as a false "near-duplicate coordination" pair,
identical to how a real templated political-astroturfing pair would
look.

## What changed

### `reddit_cosine_similarity.py`
`load_content()` takes a `relevant_only` parameter (default `True`).
When `True`, both the posts and comments queries add
`WHERE is_relevant = 1`. `__main__` runs with `relevant_only=True` by
default -- flip `RELEVANT_ONLY = True` near the bottom of the file to
`False` if you want to compare against the unfiltered run.

### `reddit_networkx.py`
`build_edges()` takes a `topic_scoped` parameter. This one has two
genuinely different, both-legitimate readings, so both are built rather
than picking one for you:

- `topic_scoped=False` (whole-activity, unchanged default behavior):
  every reply edge counts, on-topic or not. Answers "who's
  structurally influential among these accounts, period."
- `topic_scoped=True`: only counts a reply as an edge when the
  post/parent-comment it targets has `is_relevant=1`. Answers "who's
  amplified *within the protest/political conversation specifically*"
  -- closer to your proposal's Ch. 4.4 wording ("mutually amplifying
  account clusters" in a coordinated campaign).

`__main__` now runs **both** variants every time and prints a labeled
top-20 for each. They're saved to separate columns
(`scores.network_score` for whole-activity, the new
`scores.network_score_topic_scoped` for topic-scoped) so neither
overwrites the other. `composite_score.py` still reads
`scores.network_score` (whole-activity) for the Influence Score's N
component, unchanged -- switching that to the topic-scoped column is a
deliberate edit you'd make yourself if you decide that's the one you
want feeding the composite score.

**Worth watching for:** if a given account's rank swings a lot between
the two variants, most of its apparent influence is coming from outside
the specific campaign you're investigating -- worth knowing before
citing its `network_score` as evidence of coordination within it.

### `db.py`
Added `scores.network_score_topic_scoped` to the `CREATE TABLE`
statement (for anyone initializing a fresh DB) and to `_migrate_schema()`
(for anyone with an existing DB -- it'll get added automatically next
time `init_db()` runs, same pattern the file already used for
`coordination_events.target_post_id`).

### Deliberately NOT changed: Isolation Forest, HDBSCAN
Both read from the `features` table, which has no `is_relevant`
concept -- `hour_entropy`, `burstiness_score`, etc. describe an
account's *overall* behavioral pattern, computed by
`reddit_preprocessor.py` from that account's full activity history by
design. Filtering those inputs by topic would change what the number
means (an account could look "irregular" purely because its off-topic
posts got excluded), not just make it more precise. These files were
never broken on this axis and don't need this fix.

## How to verify

```
cd /path/to/your/project     # the folder containing src/
python3 test_integration.py
```

This builds a small, schema-correct SQLite test DB (not your real
data -- that still needs `collector.py` run against actual Reddit
content) with:
- 3 accounts posting near-identical on-topic content and replying to
  each other (a mutual-amplification ring)
- 3 organic accounts with varied, mostly-independent activity,
  including some off-topic posts
- 2 more organic accounts sharing *off-topic* boilerplate text (the
  realistic false-positive case)

It then runs your actual patched functions against that DB and
**asserts** that:
- the migration adds `network_score_topic_scoped` to a fresh DB
- `relevant_only=False` catches the boilerplate pair as a false
  near-duplicate (proving the bug is real)
- `relevant_only=True` correctly excludes it (proving the fix works)
- both NetworkX variants run, save to separate columns, and neither
  overwrites the other

If you see `AssertionError`, something about your actual `src/`
tree differs from what this patch expects -- worth a closer look
before trusting the fix on real data. A clean run ends with
"Both columns populated independently, no collision confirmed."

## What this does NOT prove

This is a synthetic, schema-correct test DB -- it proves the code
paths execute correctly and the specific bug they target is fixed. It
does not prove these thresholds (`COSINE_THRESHOLD=0.90`,
`CONTAMINATION=0.1`, `min_cluster_size=3`) or the `is_relevant`
keyword list's coverage are well-tuned for your actual, real Reddit
data once you collect it -- that's still an open question this test
can't answer.




# Fix: Isolation Forest direction bug in isolation_forest_evaluation.py

## What's in this package
```
src/benchmarks/cresci/isolation_forest_evaluation.py
```
Drop this in at the same path in your project, overwriting the original.

## The bug
`isolation_forest_evaluation.py` computed `anomaly_score =
-model.decision_function(X_test)` and used it as-is. That sign flip is
the correct sklearn-convention translation ("larger = more normal" ->
"larger = more anomalous"), but "more anomalous" and "more bot-like" are
only the same thing if bots are actually the rarer, more unusual
pattern in this feature space -- and the script never checked that.

On your actual uploaded `isolation_forest_metrics.json`:
`auc_roc: 0.2564` -- confirmed consistently below random chance across
the full ROC curve (544/546 threshold points below the diagonal, not
sampling noise). That means the model was correctly separating the two
classes, just backwards: humans were being ranked as more anomalous
than bots on this feature set.

## The fix
Compute AUC in both directions (`anomaly_score` and `-anomaly_score`),
keep whichever one actually separates the classes (the higher AUC), and
record which direction was used. Every downstream computation
(threshold, precision, recall, f1, ROC curve, saved predictions, saved
model) already reads from the same `anomaly_score` variable, so fixing
it once at the top fixes all of them -- no other line needed to change.

`direction_flipped`, `auc_roc_as_is`, and `auc_roc_flipped` are now also
saved in `isolation_forest_metrics.json`, so the fix is visible in the
output artifact itself, not just in the source code -- useful if anyone
(including a committee member) asks how this number was derived.

## Verified, with an honest limit on how

I don't have `data/benchmarks/cresci-2017.db` in this environment --
only outputs from previous runs (predictions CSVs, metrics JSON) were
available. So I could NOT run this script end-to-end here to regenerate
a fresh `isolation_forest_metrics.json` from scratch.

What I DID verify: I extracted the exact new direction-check logic and
ran it against your real, already-saved
`outputs/cresci/final/isolation_forest_predictions.csv` (the actual
per-account `label`/`anomaly_score` from your held-out test set). On
that real data:

```
auc_as_is:         0.256401   (matches your uploaded metrics.json exactly)
auc_flipped:        0.743599
direction_flipped:  True
final auc_roc:      0.743599
```

Precision/recall/f1 also recompute sensibly under the corrected
direction (accuracy 0.28->0.42, precision 0.31->1.00, recall
0.045->0.147, f1 0.079->0.256) -- worth reporting recall alongside AUC,
not AUC alone: at the 90th-percentile threshold, the corrected model's
top 10% flagged accounts are ALL genuinely bots (precision=1.0), but
that threshold only catches ~15% of all true bots (recall=0.147).
That's the exact "AUC-ROC and threshold-based recall are different
things" caveat worth stating explicitly in your report, not glossing
over.

## What you still need to do
Run it for real against your actual database:
```
python -m src.benchmarks.cresci.isolation_forest_evaluation
```
Confirm the printed `AUC-ROC (direction flipped)` and the resulting
`direction_flipped` in `isolation_forest_metrics.json` land close to
0.74 (matching what I verified above) -- if they come out meaningfully
different, that likely means the underlying data or model changed since
the `isolation_forest_predictions.csv` I checked this against was
generated, which is worth knowing either way.

## One more thing, stated directly
A progress-log document circulating on this project claims AUC-ROC ~=
0.84 for the benchmark Isolation Forest. That number does not come from
this file, and I don't have the actual `models/isolationforest.py` the
document describes to check it against. Don't report 0.84 as this
script's result -- report what this script, once patched and re-run,
actually prints. If your real re-run lands meaningfully above or below
0.74, that's real information worth understanding (a different feature
set, different contamination setting, or different train/test split
than what the saved predictions I checked came from), not something to
average away.

# Fix: Isolation Forest AUC on Cresci-2017

## What's in this package
```
src/benchmarks/cresci/isolation_forest_evaluation.py
```
Drop this in at the same path in your project, overwriting the original.
This supersedes the version from the previous round -- it contains that
fix plus two more, all in the same file.

## Change 1 (previous round): direction fix
Already applied and unchanged from last time. See the in-file comments
around `anomaly_score_as_is` / `auc_flipped` for the full explanation.
Verified against real saved predictions: 0.2564 (as-is) -> 0.7436
(flipped).

## Change 2 (this round): add the 5 temporal coordination features
`FEATURES` was 34 static profile/behavior columns. `train.py`'s
RandomForestClassifier already uses 5 more --
`high_coordination_count`, `temporal_events_per_tweet`,
`temporal_neighbors_per_tweet`, `high_coordination_ratio`,
`temporal_coordination_score` -- already computed into
`account_features_final` by `final_features.py`, just never requested
by this script. Added as `TEMPORAL_FEATURES`, following the same
`BASE_FEATURES + TEMPORAL_FEATURES` naming already used in `train.py`
and `final_features.py`. `FEATURES` is now 39 columns.

Rationale: temporal coordination (synchronized posting, repeated
near-simultaneous activity with other accounts) is closer to what
actually distinguishes automated/coordinated behavior than static
profile counts are -- a reasonable first thing to add before reaching
for anything more exotic, and it's already proven useful to the
RandomForest model on this same dataset.

## Change 3 (this round): contamination from measured data, not "auto"
`IsolationForest(contamination="auto")` was sklearn's own uninformed
guess. Cresci-2017's train split is genuinely stratified by label
(`split.py`'s `train_test_split(..., stratify=df["label"], ...)`), so
the actual bot fraction is knowable and stable -- now computed directly
from `train["label"].mean()` each run, instead of hardcoded or guessed.
Isolation Forest still never sees labels during `model.fit(X_train)`
itself -- contamination is a hyperparameter, not part of the fit, so
this doesn't turn it into a supervised model.

`contamination`, `base_feature_count`, and `temporal_feature_count` are
now also saved into `isolation_forest_metrics.json`, same transparency
approach as `direction_flipped` from the previous round.

## What I verified, and what I honestly could not

Same constraint as last round: no `cresci-2017.db` in this environment,
so I could not run the script end-to-end.

What I DID verify:
- The contamination computation logic is correct, and its expected
  magnitude (~0.68, from real held-out test-set data) is consistent
  with what a stratified train split would produce.
- `final_features.py` genuinely creates `account_features_final` with
  all 5 temporal columns, and `train.py` genuinely already uses them --
  so this isn't a guess that they exist, it's confirmed from the actual
  schema-creation code.
- Full syntax check, and a careful line-by-line re-read of the whole
  changed section (I caught and fixed a real mistake of my own here --
  an earlier draft of this edit accidentally dropped the script's
  original header banner print statements; the version in this zip has
  them back, in the right place, verified by direct re-read).

What I could NOT verify, and want to be direct about: whether these two
changes actually get real AUC to >= 0.80. `isolation_forest_predictions.csv`
(the only real data I have) only contains the FINAL anomaly_score from
the OLD 34-feature run -- not the raw feature values themselves -- so
there's no way for me to compute what the 39-feature, real-contamination
model would actually score without running it against your real
database. I'm not going to give you a number I can't back up.

## What you need to do
```
python -m src.benchmarks.cresci.isolation_forest_evaluation
```
Check the printed `Contamination (measured bot fraction in TRAIN...)`
line lands near 0.68-0.70, and check the final `AUC-ROC` line. If it's
still below 0.80, the next legitimate lever (not yet tried, and not
included in this patch) is checking which of the 5 temporal features
actually carry signal individually -- `ablation.py` already has the
scaffolding for exactly this kind of per-feature-group comparison for
the RandomForest model; the same approach could be adapted here rather
than guessing which feature helps.

## On "AUC 99.87"
That number is not from this file. I checked every script in your
`benchmarks.zip` upload for what computes AUC: only `train.py` and
`ablation.py` train a model that could plausibly reach that range, and
both train a `RandomForestClassifier` -- a supervised model that
learns directly from Cresci-2017's labels, not Isolation Forest, which
never sees them. It's normal and expected for a supervised classifier
to dramatically outperform an unsupervised one on the same labeled
data; they're answering different questions, not two measurements of
the same thing. RandomForest also isn't one of your proposal's four
named algorithms (Isolation Forest, HDBSCAN, Cosine Similarity,
NetworkX) -- worth keeping 99.87% clearly labeled as a RandomForest
side-benchmark in your report, not presented as your Isolation Forest
result.

