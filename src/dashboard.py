# src/dashboard.py
# Streamlit dashboard for the Influence Score pipeline -- the proposal's
# Ch. 5 / Sec 3.3.2 requirement: interactive network graphs, cluster
# visualizations, and exportable score reports.
#
# Run with: streamlit run src/dashboard.py   (from the project root)
# Needs composite_score.py to have run at least once, or influence_score/
# tier will be empty -- the dashboard will still load, just with those
# panels showing a "run the pipeline first" message instead of erroring.

import sys
import os
sys.path.append(os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

try:
    from src.db import get_conn
except ModuleNotFoundError:
    from db import get_conn

try:
    from src.config import TIERS
except ModuleNotFoundError:
    from config import TIERS


st.set_page_config(page_title="Coordinated Influence Detection", layout="wide")

TIER_COLORS = {
    'organic': '#2ca02c',
    'suspicious': '#ff9800',
    'coordinated': '#d62728',
}


@st.cache_data(ttl=60)
def load_scores():
    """Cached for 60s so dragging a slider doesn't re-hit SQLite on every
    rerun -- if you just ran a new model script, wait a minute or restart
    the app to see fresh numbers."""
    conn = get_conn()
    df = pd.read_sql("""
        SELECT s.account_id, s.anomaly_score, s.coord_score, s.dup_score,
               s.network_score, s.influence_score, s.tier, s.cluster_id,
               f.posts_per_day, f.hour_entropy, f.subreddit_count
        FROM scores s
        LEFT JOIN features f ON s.account_id = f.account_id
    """, conn)
    conn.close()
    return df


@st.cache_data(ttl=60)
def load_edges(account_ids):
    """account_ids must be a tuple, not a list/set -- st.cache_data needs
    a hashable argument to know when to reuse a cached result."""
    if not account_ids:
        return pd.DataFrame(columns=['source_account_id', 'target_account_id', 'weight'])
    conn = get_conn()
    placeholders = ','.join(['?'] * len(account_ids))
    df = pd.read_sql(f"""
        SELECT source_account_id, target_account_id, SUM(weight) AS weight
        FROM edges
        WHERE source_account_id IN ({placeholders})
           OR target_account_id IN ({placeholders})
        GROUP BY source_account_id, target_account_id
    """, conn, params=list(account_ids) * 2)
    conn.close()
    return df


@st.cache_data(ttl=60)
def load_amplified_content():
    """
    Near-duplicate content pairs (reddit_cosine_similarity.py) where BOTH
    accounts also landed in the same HDBSCAN cluster (reddit_hdbscan.py)
    -- accounts that behave in sync AND posted the same/near-identical
    text. That intersection of two independent signals is a much
    stronger "coordinated amplification" case than either alone.
    """
    conn = get_conn()
    events = pd.read_sql("""
        SELECT ce.source_account_id, ce.target_account_id,
               ce.source_post_id, ce.target_post_id,
               ce.similarity, ce.event_time,
               s1.cluster_id AS cluster_id
        FROM coordination_events ce
        JOIN scores s1 ON ce.source_account_id = s1.account_id
        JOIN scores s2 ON ce.target_account_id = s2.account_id
        WHERE ce.event_type = 'near_duplicate_content'
          AND s1.cluster_id IS NOT NULL
          AND s1.cluster_id != -1
          AND s1.cluster_id = s2.cluster_id
    """, conn)

    if events.empty:
        conn.close()
        return events

    # source_post_id/target_post_id can each be either a post or a
    # comment -- rather than tracking which is which everywhere upstream,
    # just look the text up in both tables.
    content_ids = tuple(set(events['source_post_id']) | set(events['target_post_id']))
    placeholders = ','.join(['?'] * len(content_ids))
    text_lookup = pd.read_sql(f"""
        SELECT id, COALESCE(title, '') || ' ' || COALESCE(text, '') AS text FROM posts WHERE id IN ({placeholders})
        UNION ALL
        SELECT id, text FROM comments WHERE id IN ({placeholders})
    """, conn, params=list(content_ids) * 2)
    conn.close()

    text_map = dict(zip(text_lookup['id'], text_lookup['text']))
    events['source_text'] = events['source_post_id'].map(text_map)
    events['target_text'] = events['target_post_id'].map(text_map)
    return events.sort_values('similarity', ascending=False)


@st.cache_data(ttl=60)
def load_cluster_topics():
    """
    Topic breakdown per HDBSCAN cluster, built from posts.topic /
    comments.topic (collector.py's keyword-based classify_topic() --
    see TOPIC_KEYWORDS there for the exact label set and what each one
    catches). This is deliberately broader than load_amplified_content()
    above: it counts EVERY post/comment from a clustered account, not
    just the subset that also happened to near-duplicate-match another
    account's text. "What is this coordinated cluster talking about" and
    "which specific posts are near-duplicates of each other" are
    different questions -- this answers the first one.

    is_relevant=1 filters out rows classify_topic() couldn't match to
    any keyword (topic IS NULL) -- an "off-topic / unclassified" bucket
    isn't useful to chart alongside real topic labels, but its size is
    still worth knowing, so unmatched_count is returned separately
    rather than silently dropped.
    """
    conn = get_conn()
    topic_counts = pd.read_sql("""
        SELECT s.cluster_id, content.topic, COUNT(*) AS item_count
        FROM scores s
        JOIN (
            SELECT account_id, topic, is_relevant FROM posts
            UNION ALL
            SELECT account_id, topic, is_relevant FROM comments
        ) content ON s.account_id = content.account_id
        WHERE s.cluster_id IS NOT NULL
          AND s.cluster_id != -1
          AND content.is_relevant = 1
          AND content.topic IS NOT NULL
        GROUP BY s.cluster_id, content.topic
    """, conn)

    unmatched_counts = pd.read_sql("""
        SELECT s.cluster_id, COUNT(*) AS unmatched_count
        FROM scores s
        JOIN (
            SELECT account_id, topic, is_relevant FROM posts
            UNION ALL
            SELECT account_id, topic, is_relevant FROM comments
        ) content ON s.account_id = content.account_id
        WHERE s.cluster_id IS NOT NULL
          AND s.cluster_id != -1
          AND (content.is_relevant = 0 OR content.topic IS NULL)
        GROUP BY s.cluster_id
    """, conn)
    conn.close()

    return topic_counts, unmatched_counts


def render_overview(df):
    st.subheader("Overview")

    if df.empty:
        st.info("No accounts in the `scores` table yet -- run the four model scripts first.")
        return

    scored = df[df['influence_score'].notna()]
    cols = st.columns(4)
    cols[0].metric("Accounts scored", len(scored))
    for i, tier_name in enumerate(TIERS):
        cols[i + 1].metric(tier_name.capitalize(), int((scored['tier'] == tier_name).sum()))

    if scored.empty:
        st.warning("Accounts exist, but influence_score/tier are empty -- run composite_score.py.")
        return

    fig = px.histogram(
        scored, x='influence_score', color='tier',
        color_discrete_map=TIER_COLORS, nbins=30,
        labels={'influence_score': 'Influence Score'},
    )
    fig.update_layout(bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)


def render_accounts_table(df):
    st.subheader("Accounts")

    scored = df[df['influence_score'].notna()].copy()
    if scored.empty:
        st.info("No composite scores yet -- run composite_score.py.")
        return

    tier_filter = st.multiselect("Filter by tier", options=list(TIERS), default=list(TIERS))
    min_score = st.slider("Minimum influence score", 0, 100, 0)

    filtered = scored[
        scored['tier'].isin(tier_filter) & (scored['influence_score'] >= min_score)
    ].sort_values('influence_score', ascending=False)

    display_cols = ['account_id', 'influence_score', 'tier', 'anomaly_score',
                     'coord_score', 'dup_score', 'network_score']
    st.dataframe(filtered[display_cols], use_container_width=True, height=400)

    st.download_button(
        "Download filtered results as CSV",
        data=filtered[display_cols].to_csv(index=False),
        file_name="influence_scores.csv",
        mime="text/csv",
    )


def render_network(df):
    st.subheader("Network graph")

    scored = df[df['influence_score'].notna()].copy()
    if scored.empty:
        st.info("No composite scores yet -- run composite_score.py.")
        return

    top_n = st.slider("Show top N accounts by influence score", 5, 100, 30)
    top_accounts = scored.sort_values('influence_score', ascending=False).head(top_n)
    account_ids = tuple(top_accounts['account_id'])

    edges = load_edges(account_ids)
    id_set = set(account_ids)
    edges = edges[
        edges['source_account_id'].isin(id_set) & edges['target_account_id'].isin(id_set)
    ]

    if edges.empty:
        st.info("No interactions among the selected accounts -- try a larger N.")
        return

    G = nx.DiGraph()
    for _, row in top_accounts.iterrows():
        G.add_node(row['account_id'], influence=row['influence_score'], tier=row['tier'])
    for _, row in edges.iterrows():
        if row['source_account_id'] in G and row['target_account_id'] in G:
            G.add_edge(row['source_account_id'], row['target_account_id'], weight=row['weight'])

    pos = nx.spring_layout(G, seed=42, k=0.6)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                             line=dict(width=1, color='#999'), hoverinfo='none')

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_color = [TIER_COLORS.get(G.nodes[n]['tier'], '#888') for n in G.nodes()]
    node_size = [8 + 25 * G.nodes[n]['influence'] / 100 for n in G.nodes()]
    node_text = [f"{n}<br>score: {G.nodes[n]['influence']:.1f}<br>tier: {G.nodes[n]['tier']}" for n in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
        marker=dict(size=node_size, color=node_color, line=dict(width=1, color='#333')),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False, height=600,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Node size = influence score. Color = tier. Edge = reply interaction (Sec 4.4).")


def render_clusters(df):
    st.subheader("Coordination clusters (HDBSCAN)")

    have_cluster = df[df['cluster_id'].notna() & (df['cluster_id'] != -1)]
    if have_cluster.empty:
        st.info("No clusters found yet -- run reddit_hdbscan.py first.")
        return

    plot_df = df[df['posts_per_day'].notna() & df['hour_entropy'].notna()].copy()
    plot_df['cluster_label'] = plot_df['cluster_id'].apply(
        lambda c: 'noise' if pd.isna(c) or c == -1 else f"cluster {int(c)}"
    )

    fig = px.scatter(
        plot_df, x='hour_entropy', y='posts_per_day', color='cluster_label',
        hover_data=['account_id', 'influence_score'],
        labels={'hour_entropy': 'Hour entropy (posting-time spread)', 'posts_per_day': 'Posts per day'},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Each point is an account. Tight, low-entropy clusters are the ones worth checking first.")


def render_amplified_content():
    st.subheader("Content amplified by coordinated groups")

    topic_counts, unmatched_counts = load_cluster_topics()

    # ---- Cross-cluster topic overview -----------------------------------
    # Answers "which subjects are getting the most coordinated-cluster
    # activity, across every cluster" -- a dashboard-wide summary, shown
    # before the per-cluster drilldown below.
    st.markdown("#### Topics by coordinated activity (all clusters)")
    st.caption(
        "Every post/comment from an account in a non-noise HDBSCAN cluster "
        "(Sec 4.4), grouped by keyword-classified topic (collector.py's "
        "classify_topic()). This counts ALL clustered-account content, not "
        "just the near-duplicate pairs below -- a broader 'what is this "
        "coordinated activity actually about' view."
    )

    if topic_counts.empty:
        st.info(
            "No topic-classified content from clustered accounts yet -- "
            "this needs reddit_hdbscan.py to have run, and at least some "
            "posts/comments from clustered accounts to match a keyword in "
            "collector.py's TOPIC_KEYWORDS."
        )
    else:
        overview = topic_counts.groupby('topic')['item_count'].sum().reset_index()
        overview = overview.sort_values('item_count', ascending=True)
        fig_overview = px.bar(
            overview, x='item_count', y='topic', orientation='h',
            labels={'item_count': 'Posts + comments (clustered accounts)', 'topic': ''},
        )
        fig_overview.update_layout(height=max(250, 40 * len(overview)))
        st.plotly_chart(fig_overview, use_container_width=True)

        total_unmatched = int(unmatched_counts['unmatched_count'].sum()) if not unmatched_counts.empty else 0
        if total_unmatched:
            st.caption(
                f"Additionally, {total_unmatched} posts/comments from clustered "
                f"accounts didn't match any topic keyword or were marked "
                f"not relevant -- not shown above. See TOPIC_KEYWORDS in "
                f"collector.py to extend the keyword list."
            )

    st.divider()

    # ---- Per-cluster drilldown --------------------------------------------
    st.markdown("#### Drill into one cluster")

    all_cluster_ids = sorted(topic_counts['cluster_id'].unique()) if not topic_counts.empty else []
    events = load_amplified_content()
    dup_cluster_ids = set(events['cluster_id'].unique()) if not events.empty else set()

    if not all_cluster_ids:
        st.info("No coordinated clusters with classified content to drill into yet.")
        return

    selected = st.selectbox(
        "Cluster", options=all_cluster_ids,
        format_func=lambda cl: (
            f"Cluster {int(cl)} "
            f"({(events['cluster_id'] == cl).sum() if cl in dup_cluster_ids else 0} amplified pairs, "
            f"{int(topic_counts[topic_counts['cluster_id'] == cl]['item_count'].sum())} classified posts/comments)"
        ),
    )

    cluster_topics = topic_counts[topic_counts['cluster_id'] == selected].sort_values('item_count', ascending=False)
    if not cluster_topics.empty:
        top_topic_row = cluster_topics.iloc[0]
        cols = st.columns(2)
        cols[0].metric("Top topic in this cluster", top_topic_row['topic'].replace('_', ' '))
        cols[1].metric("Posts/comments on that topic", int(top_topic_row['item_count']))

        fig_cluster = px.bar(
            cluster_topics, x='item_count', y='topic', orientation='h',
            labels={'item_count': 'Posts + comments', 'topic': ''},
        )
        fig_cluster.update_layout(height=max(200, 40 * len(cluster_topics)))
        st.plotly_chart(fig_cluster, use_container_width=True)

    if selected not in dup_cluster_ids:
        st.caption(
            "This cluster has no near-duplicate text pairs yet (needs "
            "reddit_cosine_similarity.py to find matching content), so no "
            "amplified-pairs list below -- the topic breakdown above still "
            "reflects everything this cluster has posted."
        )
        return

    cluster_events = events[events['cluster_id'] == selected]
    st.metric("Near-duplicate pairs in this cluster", len(cluster_events))

    for _, row in cluster_events.iterrows():
        source_text = row['source_text'] if pd.notna(row['source_text']) else "_(text unavailable)_"
        target_text = row['target_text'] if pd.notna(row['target_text']) else "_(text unavailable)_"
        label = (f"{row['source_account_id'][:10]}\u2026 \u2194 "
                 f"{row['target_account_id'][:10]}\u2026  "
                 f"(similarity {row['similarity']:.3f})")
        with st.expander(label):
            cols = st.columns(2)
            cols[0].markdown(f"**{row['source_account_id'][:12]}**")
            cols[0].write(source_text)
            cols[1].markdown(f"**{row['target_account_id'][:12]}**")
            cols[1].write(target_text)

    st.download_button(
        "Download this cluster's amplified content as CSV",
        data=cluster_events[[
            'source_account_id', 'target_account_id', 'similarity',
            'source_text', 'target_text'
        ]].to_csv(index=False),
        file_name=f"amplified_content_cluster_{int(selected)}.csv",
        mime="text/csv",
    )


def main():
    st.title("Coordinated Influence Amplification Detection")
    st.caption("Nepal social media monitoring dashboard")

    df = load_scores()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Overview", "Accounts", "Network graph", "Clusters", "Amplified content"]
    )
    with tab1:
        render_overview(df)
    with tab2:
        render_accounts_table(df)
    with tab3:
        render_network(df)
    with tab4:
        render_clusters(df)
    with tab5:
        render_amplified_content()


if __name__ == "__main__":
    main()