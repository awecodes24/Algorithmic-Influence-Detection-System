# src/models/reddit_networkx.py
# Network influence analysis on REAL Reddit data using NetworkX PageRank.
# Implements the proposal's Ch. 4.4: interactions modeled as a directed
# graph, PageRank quantifies structural influence, normalized PageRank
# feeds the network signal (N) in the composite Influence Score.
#
# The proposal's own wording ("retweets, mentions, and replies") is
# Twitter-shaped -- Reddit has neither retweets nor @mentions in that
# sense. The natural Reddit equivalent, used here, is reply structure:
#   - comment_on_post   : commenter -> post author
#   - reply_to_comment  : commenter -> parent-comment author (nested reply)
# Both are directed (source = the account taking the action) and weighted
# by how many times that pair interacted that way.
#
# Two graph variants, both run by default (see __main__): whole-activity
# (every reply edge, regardless of topic) and topic-scoped (only edges
# where the target post/comment is_relevant=1). See build_edges()'s
# docstring for which question each one answers -- they're written to
# separate columns (scores.network_score vs.
# scores.network_score_topic_scoped) so neither overwrites the other.

import pandas as pd
import networkx as nx
from datetime import datetime, timezone

from src.db import get_conn
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build_edges(topic_scoped=False):
    """
    Builds directed, weighted account -> account edges from reply structure.
    Self-loops (an account replying to its own post/comment) are dropped --
    that's not an inter-account interaction.

    topic_scoped: which question this graph answers.

    - False (default) -- WHOLE-ACTIVITY graph. Every reply edge counts,
      regardless of whether the post/parent-comment being replied to is
      on-topic (is_relevant=1) or not. Answers "who's structurally
      influential among these accounts, period" -- an account that's
      genuinely relevant everywhere except a couple of reply targets
      still keeps those edges, so it isn't undercounted for something
      unrelated to its actual behavior.

    - True -- TOPIC-SCOPED graph. Only counts a reply as an edge when the
      post/parent-comment it targets has is_relevant=1 (the flag
      collector.classify_topic() sets against HISTORICAL_SEARCH_TERMS /
      TOPIC_KEYWORDS at collection time). Answers "who's amplified WITHIN
      the protest/political conversation specifically" -- closer to the
      proposal's Ch. 4.4 wording ("mutually amplifying account clusters"
      in a coordinated CAMPAIGN, not general Reddit chatter the collector
      happened to also pull in from casting a wide r/<subreddit>/new/ net.

    Neither is more "correct" than the other -- they answer different
    questions. Run both and compare: if a known account's rank swings a
    lot between them, that's itself informative (it means most of that
    account's apparent influence comes from off-topic activity, not the
    campaign you're actually investigating).
    """
    conn = get_conn()
    relevance_join = " AND p.is_relevant = 1" if topic_scoped else ""
    parent_relevance_join = " AND parent.is_relevant = 1" if topic_scoped else ""

    # commenter -> post author, for every comment
    comment_on_post = pd.read_sql(f"""
        SELECT c.account_id AS source_account_id,
               p.account_id AS target_account_id
        FROM comments c
        JOIN posts p ON c.post_id = p.id
        WHERE c.account_id != p.account_id{relevance_join}
    """, conn)
    comment_on_post['edge_type'] = 'comment_on_post'

    # commenter -> parent comment's author, only for genuine nested replies
    # (parent_id matches another COMMENT's id, not the post's id -- this
    # self-join approach works regardless of Reddit's t1_/t3_ id-prefix
    # convention, since it never has to assume which prefix means what)
    reply_to_comment = pd.read_sql(f"""
        SELECT c.account_id AS source_account_id,
               parent.account_id AS target_account_id
        FROM comments c
        JOIN comments parent ON c.parent_id = parent.id
        WHERE c.account_id != parent.account_id{parent_relevance_join}
    """, conn)
    reply_to_comment['edge_type'] = 'reply_to_comment'

    all_accounts = pd.read_sql("SELECT id AS account_id FROM accounts", conn)
    conn.close()

    raw = pd.concat([comment_on_post, reply_to_comment], ignore_index=True)
    logger.info(
        f"Built {len(comment_on_post)} comment_on_post + "
        f"{len(reply_to_comment)} reply_to_comment raw interactions "
        f"(topic_scoped={topic_scoped})"
    )

    if len(raw) == 0:
        empty = pd.DataFrame(columns=['source_account_id', 'target_account_id', 'edge_type', 'weight'])
        return empty, all_accounts

    edges = (
        raw.groupby(['source_account_id', 'target_account_id', 'edge_type'])
        .size()
        .reset_index(name='weight')
    )
    if topic_scoped:
        # Distinct edge_type suffix so a topic_scoped run and a
        # whole-activity run of the same account pair don't collide on
        # save_edges()'s (source, target, edge_type) upsert key and
        # silently overwrite each other -- both variants' weights need
        # to coexist in the `edges` table so the dashboard/DB can show
        # either one.
        edges['edge_type'] = edges['edge_type'] + '_topic_scoped'
    return edges, all_accounts


def save_edges(edges):
    conn = get_conn()
    c = conn.cursor()
    for _, row in edges.iterrows():
        c.execute("""
            INSERT INTO edges (source_account_id, target_account_id, edge_type, weight)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(source_account_id, target_account_id, edge_type) DO UPDATE SET
                weight = excluded.weight
        """, (row['source_account_id'], row['target_account_id'], row['edge_type'], float(row['weight'])))
    conn.commit()
    conn.close()
    logger.info(f"Saved {len(edges)} edges to edges table")


def run_pagerank(edges, all_accounts, topic_scoped=False):
    """
    Builds one combined directed graph (edge types merged, weights summed
    where a pair has both interaction types) and runs PageRank on it.
    Every known account is added as a node up front -- including accounts
    with no edges at all -- so isolated accounts still get an explicit
    (low, baseline) network_score instead of silently missing from the
    scores table once the composite score is computed.
    """
    G = nx.DiGraph()
    G.add_nodes_from(all_accounts['account_id'])

    if len(edges) > 0:
        combined = edges.groupby(['source_account_id', 'target_account_id'])['weight'].sum().reset_index()
        for _, row in combined.iterrows():
            G.add_edge(row['source_account_id'], row['target_account_id'], weight=row['weight'])

    graph_label = "topic-scoped" if topic_scoped else "whole-activity"
    logger.info(f"Graph ({graph_label}): {G.number_of_nodes()} accounts, {G.number_of_edges()} directed edges")

    if G.number_of_edges() == 0:
        raise ValueError(
            f"No interactions found for the {graph_label} graph (no comment "
            f"references an existing{' on-topic' if topic_scoped else ''} "
            f"post/comment author in this DB) -- nothing for PageRank to "
            f"run on yet. {'Try topic_scoped=False, or collect' if topic_scoped else 'Collect'} more data first."
        )

    # PageRank measures structural attention/influence.
    # Since edges point from the replying account to the account
    # receiving the reply, incoming weighted edges represent
    # repeated interaction directed toward that account.
    pagerank = nx.pagerank(
        G,
        weight="weight",
    )

    # Weighted in-degree preserves the actual interaction volume
    # instead of counting only unique incoming neighbors.
    weighted_in_degree = dict(
        G.in_degree(
            weight="weight",
        )
    )

    df = pd.DataFrame({
        'account_id': list(pagerank.keys()),
        'pagerank': list(pagerank.values()),
    })
    df['in_degree'] = (
        df['account_id']
        .map(weighted_in_degree)
        .fillna(0)
    )

    # Normalize to [0, 1] -- same min-max pattern as anomaly_score and
    # coord_score in the other model scripts, so all four signals feeding
    # the composite score sit on a comparable scale.
    pr_min, pr_max = df['pagerank'].min(), df['pagerank'].max()
    if pr_max > pr_min:
        df['network_score'] = (df['pagerank'] - pr_min) / (pr_max - pr_min)
    else:
        df['network_score'] = 0.0

    return df.sort_values('network_score', ascending=False)


def save_scores(df, topic_scoped=False):
    """
    Whole-activity (topic_scoped=False) writes to scores.network_score,
    same as before -- this is what composite_score.py reads for the N
    component of the Influence Score, so that behavior is unchanged.

    Topic-scoped writes to scores.network_score_topic_scoped instead of
    the same column -- both variants are useful to compare, and without
    a separate column, running one after the other would silently
    overwrite whichever ran first with no error or warning. If you
    decide the topic-scoped graph is the one that should actually feed
    the composite score, that's a deliberate edit to composite_score.py,
    not something this script should do on its own.
    """
    conn = get_conn()
    c = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    score_column = 'network_score_topic_scoped' if topic_scoped else 'network_score'

    for _, row in df.iterrows():
        # Same pattern as the other three model scripts: only touches its
        # own column(s) + scored_at, never overwrites anomaly_score,
        # coord_score, or dup_score already written by the others.
        c.execute(f"""
            INSERT INTO scores (account_id, {score_column}, scored_at)
            VALUES (?, ?, ?)
            ON CONFLICT(account_id) DO UPDATE SET
                {score_column} = excluded.{score_column},
                scored_at = excluded.scored_at
        """, (row['account_id'], float(row['network_score']), now))

        # communities.pagerank / .centrality hold the raw, un-normalized
        # numbers for inspection. community_id and coordination_strength
        # are deliberately left untouched -- those need an actual
        # community-detection pass (e.g. Louvain), which is a separate
        # decision on top of this, not something PageRank alone produces.
        # These two stay whole-activity-only (no topic_scoped variant) --
        # they're for manual inspection via the communities table, and a
        # second set of columns here isn't worth the added complexity
        # unless you actually find yourself needing it.
        if not topic_scoped:
            c.execute("""
                INSERT INTO communities (account_id, pagerank, centrality)
                VALUES (?, ?, ?)
                ON CONFLICT(account_id) DO UPDATE SET
                    pagerank = excluded.pagerank,
                    centrality = excluded.centrality
            """, (row['account_id'], float(row['pagerank']), float(row['in_degree'])))

    conn.commit()
    conn.close()
    logger.info(
        f"Saved {score_column} for {len(df)} accounts to scores table"
        f"{' + communities table' if not topic_scoped else ''}"
    )


def print_top_influential(df, n=20, topic_scoped=False):
    label = "TOPIC-SCOPED" if topic_scoped else "WHOLE-ACTIVITY"
    print(f"\n{'━'*60}")
    print(f"  TOP {n} MOST STRUCTURALLY INFLUENTIAL ACCOUNTS ({label})")
    print(f"{'━'*60}")
    print(f"{'account_id':<20}{'network_score':>14}{'in_degree':>12}")
    print(f"{'-'*60}")
    for _, row in df.head(n).iterrows():
        print(f"{row['account_id'][:18]:<20}{row['network_score']:>14.4f}{int(row['in_degree']):>12}")
    print(f"{'━'*60}\n")


if __name__ == "__main__":
    logger.info("Running NetworkX PageRank on real Reddit interaction data\n")

    for topic_scoped in (False, True):
        label = "topic-scoped" if topic_scoped else "whole-activity"
        try:
            edges, all_accounts = build_edges(topic_scoped=topic_scoped)
            if len(edges) > 0:
                save_edges(edges)
            df = run_pagerank(edges, all_accounts, topic_scoped=topic_scoped)
            save_scores(df, topic_scoped=topic_scoped)
            print_top_influential(df, topic_scoped=topic_scoped)
        except ValueError as e:
            # Isolated per variant on purpose -- a sparse topic-scoped
            # graph erroring out (e.g. too little on-topic-to-on-topic
            # reply activity yet) shouldn't stop the whole-activity run
            # from completing and being reported.
            logger.warning(f"Skipped {label} PageRank: {e}")

    logger.info(
        "NOTE: community_id and coordination_strength in the communities "
        "table are still empty -- this script only computes PageRank + "
        "in-degree centrality, matching what the proposal's Ch. 4.4 "
        "actually specifies. Full community detection (e.g. Louvain) "
        "would be a separate, deliberate addition on top of this, not "
        "something PageRank produces by itself.\n"
        "Compare the two runs above: if a given account's rank swings a "
        "lot between whole-activity and topic-scoped, most of its "
        "apparent influence comes from outside the specific campaign "
        "you're investigating -- worth knowing before citing its "
        "network_score as evidence of coordination within it."
    )
