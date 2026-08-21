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

import pandas as pd
import networkx as nx
from datetime import datetime, timezone

from src.db import get_conn
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build_edges():
    """
    Builds directed, weighted account -> account edges from reply structure.
    Self-loops (an account replying to its own post/comment) are dropped --
    that's not an inter-account interaction.
    """
    conn = get_conn()

    # commenter -> post author, for every comment
    comment_on_post = pd.read_sql("""
        SELECT c.account_id AS source_account_id,
               p.account_id AS target_account_id
        FROM comments c
        JOIN posts p ON c.post_id = p.id
        WHERE c.account_id != p.account_id
    """, conn)
    comment_on_post['edge_type'] = 'comment_on_post'

    # commenter -> parent comment's author, only for genuine nested replies
    # (parent_id matches another COMMENT's id, not the post's id -- this
    # self-join approach works regardless of Reddit's t1_/t3_ id-prefix
    # convention, since it never has to assume which prefix means what)
    reply_to_comment = pd.read_sql("""
        SELECT c.account_id AS source_account_id,
               parent.account_id AS target_account_id
        FROM comments c
        JOIN comments parent ON c.parent_id = parent.id
        WHERE c.account_id != parent.account_id
    """, conn)
    reply_to_comment['edge_type'] = 'reply_to_comment'

    all_accounts = pd.read_sql("SELECT id AS account_id FROM accounts", conn)
    conn.close()

    raw = pd.concat([comment_on_post, reply_to_comment], ignore_index=True)
    logger.info(
        f"Built {len(comment_on_post)} comment_on_post + "
        f"{len(reply_to_comment)} reply_to_comment raw interactions"
    )

    if len(raw) == 0:
        empty = pd.DataFrame(columns=['source_account_id', 'target_account_id', 'edge_type', 'weight'])
        return empty, all_accounts

    edges = (
        raw.groupby(['source_account_id', 'target_account_id', 'edge_type'])
        .size()
        .reset_index(name='weight')
    )
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


def run_pagerank(edges, all_accounts):
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

    logger.info(f"Graph: {G.number_of_nodes()} accounts, {G.number_of_edges()} directed edges")

    if G.number_of_edges() == 0:
        raise ValueError(
            "No interactions found (no comment references an existing "
            "post/comment author in this DB) -- nothing for PageRank to "
            "run on yet. Collect more data first."
        )

    pagerank = nx.pagerank(G, weight='weight')
    in_degree = dict(G.in_degree())

    df = pd.DataFrame({
        'account_id': list(pagerank.keys()),
        'pagerank': list(pagerank.values()),
    })
    df['in_degree'] = df['account_id'].map(in_degree).fillna(0)

    # Normalize to [0, 1] -- same min-max pattern as anomaly_score and
    # coord_score in the other model scripts, so all four signals feeding
    # the composite score sit on a comparable scale.
    pr_min, pr_max = df['pagerank'].min(), df['pagerank'].max()
    if pr_max > pr_min:
        df['network_score'] = (df['pagerank'] - pr_min) / (pr_max - pr_min)
    else:
        df['network_score'] = 0.0

    return df.sort_values('network_score', ascending=False)


def save_scores(df):
    conn = get_conn()
    c = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()

    for _, row in df.iterrows():
        # Same pattern as the other three model scripts: only touches its
        # own column(s) + scored_at, never overwrites anomaly_score,
        # coord_score, or dup_score already written by the others.
        c.execute("""
            INSERT INTO scores (account_id, network_score, scored_at)
            VALUES (?, ?, ?)
            ON CONFLICT(account_id) DO UPDATE SET
                network_score = excluded.network_score,
                scored_at = excluded.scored_at
        """, (row['account_id'], float(row['network_score']), now))

        # communities.pagerank / .centrality hold the raw, un-normalized
        # numbers for inspection. community_id and coordination_strength
        # are deliberately left untouched -- those need an actual
        # community-detection pass (e.g. Louvain), which is a separate
        # decision on top of this, not something PageRank alone produces.
        c.execute("""
            INSERT INTO communities (account_id, pagerank, centrality)
            VALUES (?, ?, ?)
            ON CONFLICT(account_id) DO UPDATE SET
                pagerank = excluded.pagerank,
                centrality = excluded.centrality
        """, (row['account_id'], float(row['pagerank']), float(row['in_degree'])))

    conn.commit()
    conn.close()
    logger.info(f"Saved network_score for {len(df)} accounts to scores + communities tables")


def print_top_influential(df, n=20):
    print(f"\n{'━'*60}")
    print(f"  TOP {n} MOST STRUCTURALLY INFLUENTIAL ACCOUNTS")
    print(f"{'━'*60}")
    print(f"{'account_id':<20}{'network_score':>14}{'in_degree':>12}")
    print(f"{'-'*60}")
    for _, row in df.head(n).iterrows():
        print(f"{row['account_id'][:18]:<20}{row['network_score']:>14.4f}{int(row['in_degree']):>12}")
    print(f"{'━'*60}\n")


if __name__ == "__main__":
    logger.info("Running NetworkX PageRank on real Reddit interaction data\n")

    edges, all_accounts = build_edges()
    if len(edges) > 0:
        save_edges(edges)
    df = run_pagerank(edges, all_accounts)
    save_scores(df)
    print_top_influential(df)

    logger.info(
        "NOTE: community_id and coordination_strength in the communities "
        "table are still empty -- this script only computes PageRank + "
        "in-degree centrality, matching what the proposal's Ch. 4.4 "
        "actually specifies. Full community detection (e.g. Louvain) "
        "would be a separate, deliberate addition on top of this, not "
        "something PageRank produces by itself."
    )
