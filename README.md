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
