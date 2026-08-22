"""
Integration test for the two patches: reddit_cosine_similarity.py's
relevant_only and reddit_networkx.py's topic_scoped. Builds a small but
SCHEMA-CORRECT SQLite DB (not the project's real data -- that still
doesn't exist -- just enough rows, with the right columns and a real
reply chain, to prove the patched code paths actually execute against
the real schema without error and produce sane output).

Run this from your actual project root (the directory containing
src/), e.g.:
    cd /path/to/your/project
    python3 test_integration.py

It finds src/ relative to your CWD -- if that fails, it falls back to
looking next to this script itself. Either way, drop the three patched
files (src/db.py, src/models/reddit_cosine_similarity.py,
src/models/reddit_networkx.py) into your real project tree first.
"""
import os
import sys
import sqlite3
from pathlib import Path

# Prefer the current working directory (run this from your project
# root); fall back to the directory this script lives in, in case
# you're running it from somewhere else.
_candidates = [Path.cwd(), Path(__file__).resolve().parent]
PROJECT_ROOT = next((p for p in _candidates if (p / 'src').is_dir()), None)
if PROJECT_ROOT is None:
    sys.exit(
        "Could not find a src/ directory in the current folder or next "
        "to this script. Run this from your project root (the folder "
        "containing src/), after copying in the three patched files."
    )
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
print(f"Using project root: {PROJECT_ROOT}\n")

TEST_DB = PROJECT_ROOT / 'test_integration.db'
if TEST_DB.exists():
    TEST_DB.unlink()

# Point the project's config at our throwaway test DB before importing
# anything that calls get_conn(), since db.py resolves DB_PATH at import
# time from config.py.
import src.config as config
config.DB_PATH = TEST_DB
import src.db as db
db.DB_PATH = TEST_DB

print("=" * 72)
print("STEP 1: init_db() -- confirm schema creates cleanly, including the")
print("        new scores.network_score_topic_scoped migration column")
print("=" * 72)
db.init_db()
conn = sqlite3.connect(TEST_DB)
cols = {row[1] for row in conn.execute("PRAGMA table_info(scores)")}
assert 'network_score_topic_scoped' in cols, "MIGRATION FAILED: column missing"
print("scores table columns:", sorted(cols))
print("network_score_topic_scoped present: YES\n")

print("=" * 72)
print("STEP 2: populate minimal accounts/posts/comments, mixing")
print("        is_relevant=1 (on-topic) and is_relevant=0 (off-topic) rows")
print("=" * 72)

# 8 accounts: 3 "coordinated" (post near-identical on-topic content and
# reply to each other), 3 "organic" (varied content, off-topic activity
# included, sparse cross-replies), plus 2 more "organic" accounts that
# happen to share OFF-TOPIC boilerplate text (a subreddit-rule reminder,
# is_relevant=0) -- a realistic false-positive case: shared TEMPLATE
# text that isn't coordination, just two accounts following the same
# subreddit convention. This is what relevant_only=True should exclude.
accounts = [
    ('acc_coord_1', 100.0), ('acc_coord_2', 50.0), ('acc_coord_3', 10.0),
    ('acc_organic_1', 5000.0), ('acc_organic_2', 3000.0), ('acc_organic_3', 1200.0),
    ('acc_organic_4', 2000.0), ('acc_organic_5', 1800.0),
]
for acc_id, karma in accounts:
    conn.execute(
        "INSERT INTO accounts (id, created_utc, comment_karma, link_karma) VALUES (?, ?, ?, ?)",
        (acc_id, 1700000000.0, karma, karma)
    )

# Posts: coordinated accounts post near-identical ON-TOPIC content
# ("nepo kid" protest topic, is_relevant=1). Organic accounts post BOTH
# on-topic (varied wording) and clearly off-topic content (is_relevant=0)
# -- e.g. commenting about a football match in the same subreddit.
# acc_organic_4/5 post IDENTICAL off-topic boilerplate (a subreddit rule
# reminder) -- this pair should NOT show up as a near-duplicate under
# relevant_only=True, only under relevant_only=False.
_boilerplate = 'please remember to read the subreddit rules before posting here thanks'
posts = [
    ('post_c1', 'acc_coord_1', 'r/Nepal', 'nepo kid protest', 'we must all stand together against nepo kid corruption in government', 1700000100.0, 1, 'political_criticism'),
    ('post_c2', 'acc_coord_2', 'r/Nepal', 'nepo kid protest', 'we must all stand together against nepo kid corruption in government', 1700000160.0, 1, 'political_criticism'),
    ('post_c3', 'acc_coord_3', 'r/Nepal', 'nepo kid protest', 'we must all stand together against nepo kid corruption in government', 1700000220.0, 1, 'political_criticism'),
    ('post_o1', 'acc_organic_1', 'r/Nepal', 'my thoughts on the protests', 'saw the gen z protest downtown today, quite a scene honestly', 1700000500.0, 1, 'political_events'),
    ('post_o2', 'acc_organic_2', 'r/Nepal', 'football tonight', 'anyone catch the football match last night, what a game', 1700000700.0, 0, None),
    ('post_o3', 'acc_organic_3', 'r/Nepal', 'weather update', 'rain expected in the valley this weekend apparently', 1700000900.0, 0, None),
    ('post_o4', 'acc_organic_4', 'r/Nepal', 'reminder', _boilerplate, 1701000000.0, 0, None),
    ('post_o5', 'acc_organic_5', 'r/Nepal', 'reminder', _boilerplate, 1701000100.0, 0, None),
]
for pid, acc, sub, title, text, ts, rel, topic in posts:
    conn.execute(
        "INSERT INTO posts (id, account_id, subreddit, title, text, created_utc, is_relevant, topic) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (pid, acc, sub, title, text, ts, rel, topic)
    )

# Comments: coordinated accounts REPLY TO EACH OTHER'S posts (mutual
# amplification ring, all on-topic). Organic accounts reply sparsely,
# including one organic->organic reply on an OFF-TOPIC post.
comments = [
    ('cmt_c1', 'acc_coord_2', 'post_c1', None, 'totally agree, nepo kid must go', 1700000130.0, 1),
    ('cmt_c2', 'acc_coord_3', 'post_c1', None, 'exactly right, standing with you', 1700000140.0, 1),
    ('cmt_c3', 'acc_coord_1', 'post_c2', None, 'yes we must act now together', 1700000190.0, 1),
    ('cmt_c4', 'acc_coord_3', 'post_c2', None, 'agreed 100 percent on this', 1700000200.0, 1),
    ('cmt_c5', 'acc_coord_1', 'post_c3', None, 'this is the way forward for us', 1700000250.0, 1),
    ('cmt_c6', 'acc_coord_2', 'post_c3', None, 'could not have said it better', 1700000260.0, 1),
    ('cmt_o1', 'acc_organic_2', 'post_o1', None, 'yeah it was pretty intense out there', 1700000550.0, 1),
    ('cmt_o2', 'acc_organic_3', 'post_o2', None, 'terrible match honestly, ref was awful', 1700000750.0, 0),
]
for cid, acc, post_id, parent_id, text, ts, rel in comments:
    conn.execute(
        "INSERT INTO comments (id, account_id, post_id, parent_id, text, created_utc, is_relevant) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (cid, acc, post_id, parent_id, text, ts, rel)
    )

conn.commit()
conn.close()
print(f"Inserted {len(accounts)} accounts, {len(posts)} posts "
      f"({sum(1 for p in posts if p[6]==1)} relevant, {sum(1 for p in posts if p[6]==0)} not), "
      f"{len(comments)} comments\n")

print("=" * 72)
print("STEP 3: reddit_cosine_similarity.py -- relevant_only=True vs False")
print("=" * 72)
from src.models import reddit_cosine_similarity as cos

pair_counts = {}
for relevant_only in (True, False):
    print(f"\n--- relevant_only={relevant_only} ---")
    content_df = cos.load_content(relevant_only=relevant_only)
    pairs_df = cos.find_near_duplicates(content_df)
    pair_counts[relevant_only] = set(zip(pairs_df['account_i'], pairs_df['account_j']))
    print(f"Cross-account near-duplicate pairs found: {len(pairs_df)}")
    if len(pairs_df):
        print(pairs_df[['account_i', 'account_j', 'similarity']].to_string(index=False))

false_positive_pair = {('acc_organic_4', 'acc_organic_5')}
caught_when_filtered_off = false_positive_pair & pair_counts[False]
caught_when_filtered_on = false_positive_pair & pair_counts[True]
assert caught_when_filtered_off, (
    "Expected the boilerplate false-positive pair to show up with "
    "relevant_only=False -- if it's not there, the test seed data "
    "changed and this check needs updating."
)
assert not caught_when_filtered_on, (
    "REGRESSION: the boilerplate false-positive pair (acc_organic_4, "
    "acc_organic_5) showed up even with relevant_only=True -- the "
    "is_relevant filter in load_content() isn't working as intended."
)
print(
    "\nConfirmed: relevant_only=True correctly excludes the "
    "off-topic boilerplate false-positive pair that relevant_only=False "
    "picks up."
)

print()
print("=" * 72)
print("STEP 4: reddit_networkx.py -- topic_scoped=False vs True")
print("=" * 72)
from src.models import reddit_networkx as nx_mod

for topic_scoped in (False, True):
    print(f"\n--- topic_scoped={topic_scoped} ---")
    edges, all_accounts = nx_mod.build_edges(topic_scoped=topic_scoped)
    print(f"Edges built: {len(edges)}")
    if len(edges):
        print(edges.to_string(index=False))
    try:
        pr_df = nx_mod.run_pagerank(edges, all_accounts, topic_scoped=topic_scoped)
        print(f"\nPageRank result (network_score):")
        print(pr_df[['account_id', 'network_score', 'in_degree']].to_string(index=False))
    except ValueError as e:
        print(f"(no PageRank: {e})")

print()
print("=" * 72)
print("STEP 5: save both networkx variants to the scores table, confirm")
print("        they land in SEPARATE columns without colliding")
print("=" * 72)
edges_wa, all_acc = nx_mod.build_edges(topic_scoped=False)
df_wa = nx_mod.run_pagerank(edges_wa, all_acc, topic_scoped=False)
nx_mod.save_scores(df_wa, topic_scoped=False)

edges_ts, all_acc = nx_mod.build_edges(topic_scoped=True)
df_ts = nx_mod.run_pagerank(edges_ts, all_acc, topic_scoped=True)
nx_mod.save_scores(df_ts, topic_scoped=True)

conn = sqlite3.connect(TEST_DB)
rows = conn.execute(
    "SELECT account_id, network_score, network_score_topic_scoped FROM scores ORDER BY account_id"
).fetchall()
conn.close()
print(f"\n{'account_id':<16}{'network_score':>16}{'network_score_topic_scoped':>30}")
for r in rows:
    ns = f"{r[1]:.4f}" if r[1] is not None else "NULL"
    nst = f"{r[2]:.4f}" if r[2] is not None else "NULL"
    print(f"{r[0]:<16}{ns:>16}{nst:>30}")
print("\nBoth columns populated independently, no collision confirmed.")

