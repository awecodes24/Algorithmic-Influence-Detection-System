"""InfluenceWatch Nepal — evidence dashboard for the AID System project.

Run: streamlit run src/dashboard.py

Every score is shown with its component signals attached; nothing here
is labelled "coordinated" without the anomaly, cluster, duplication,
and network evidence sitting right next to it.
"""

import os
import sys
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

try:
    from sklearn.ensemble import IsolationForest
except Exception:
    IsolationForest = None

pio.templates.default = "plotly_dark"

try:
    from src.config import COSINE_THRESHOLD, TIERS, WEIGHTS
    from src.db import get_conn
except ModuleNotFoundError:
    from config import COSINE_THRESHOLD, TIERS, WEIGHTS
    from db import get_conn


st.set_page_config(
    page_title="InfluenceWatch Nepal",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- theme --------------------------------------

TIER_COLORS = {
    "organic": "#5FBE8D",
    "suspicious": "#E3A548",
    "coordinated": "#E56A65",
}

# Muted, non-semantic palette for arbitrary cluster IDs — kept clear of
# green/red so it never gets confused with the tier colors above.
CLUSTER_PALETTE = [
    "#7FA8D4", "#D4B073", "#A896D4", "#7FC2C0",
    "#C79470", "#8FA0D9", "#C285B0", "#9AACB8",
]

PLOT_BG = "#171B21"
PLOT_PAPER = "#171B21"
PLOT_GRID = "#303641"
PLOT_TEXT = "#ECEDF0"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --paper: #0F1216;
        --surface: #171B21;
        --ink: #ECEDF0;
        --muted: #98A2B0;
        --line: #2B303A;
        --signal: #7AA2F7;
        --organic: #5FBE8D;
        --suspicious: #E3A548;
        --coordinated: #E56A65;
    }

    html, body, .stApp,
    [data-testid="stMarkdownContainer"], [data-testid="stText"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: var(--ink);
    }
    .stApp { background: var(--paper); }
    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; }

    h1, h2, h3 {
        font-family: 'Newsreader', Georgia, serif;
        font-weight: 600;
        letter-spacing: -0.01em;
        color: var(--ink);
    }

    .mono { font-family: 'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace; }

    .masthead {
        border-top: 1px solid var(--ink);
        border-bottom: 1px solid var(--line);
        padding: 1.1rem 0 1.15rem;
        margin-bottom: 1.5rem;
    }
    .masthead .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: .72rem;
        letter-spacing: .14em;
        text-transform: uppercase;
        color: var(--muted);
    }
    .masthead h1 { font-size: 2.05rem; margin: .3rem 0 .4rem; }
    .masthead p { color: var(--muted); font-size: .98rem; max-width: 620px; margin: 0; }

    .legend-row { margin: .3rem 0 1.1rem; font-size: .84rem; }
    .legend-row .dot { margin-left: 1.1rem; }
    .legend-row .dot:first-child { margin-left: 0; }

    .dot {
        display: inline-block; width: 8px; height: 8px; border-radius: 50%;
        margin-right: 6px; position: relative; top: -1px;
    }

    .callout {
        border-left: 3px solid var(--line);
        background: var(--surface);
        padding: .7rem 1rem;
        border-radius: 3px;
        margin: .7rem 0;
        color: var(--ink);
    }
    .callout .tag {
        display: block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: .68rem;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .3rem;
        color: var(--muted);
    }
    .callout.note { border-color: var(--signal); }
    .callout.note .tag { color: var(--signal); }
    .callout.limit { border-color: var(--suspicious); }
    .callout.limit .tag { color: var(--suspicious); }
    .callout.good { border-color: var(--organic); }
    .callout.good .tag { color: var(--organic); }

    [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; color: var(--ink); }
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif; text-transform: uppercase;
        letter-spacing: .06em; font-size: .72rem; color: var(--muted);
    }
    section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }
    .stTabs [data-baseweb="tab"] { font-family: 'Inter', sans-serif; font-weight: 500; color: var(--muted); }
    .stTabs [aria-selected="true"] { color: var(--ink) !important; }

    /* Large, high-contrast visualization cards */
    [data-testid="stPlotlyChart"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 10px 10px 6px 10px;
        box-shadow: 0 8px 24px rgba(0,0,0,.22);
        overflow: hidden;
    }

    .viz-title {
        font-family: 'Inter', sans-serif;
        font-size: .78rem;
        font-weight: 600;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: var(--muted);
        margin: .4rem 0 .55rem;
    }

    .network-note {
        background: #12161C;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: .7rem .9rem;
        margin-top: .55rem;
        color: var(--muted);
    }

    .if-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: .8rem 1rem;
        margin: .35rem 0 .9rem;
        box-shadow: 0 8px 24px rgba(0,0,0,.18);
    }
    .if-title {
        font-family: 'Inter', sans-serif;
        font-size: .82rem;
        font-weight: 600;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: .35rem;
    }

    .cluster-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: .4rem;
        box-shadow: 0 8px 24px rgba(0,0,0,.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- data layer ---------------------------------

@st.cache_data(ttl=30)
def read_df(query: str, params: tuple = ()) -> pd.DataFrame:
    try:
        conn = get_conn()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_scores() -> pd.DataFrame:
    return read_df(
        """
        SELECT
            s.account_id, s.anomaly_score, s.coord_score, s.dup_score,
            s.network_score, s.influence_score, s.tier, s.cluster_id,
            s.scored_at,
            f.age_days, f.posts_per_day, f.comments_per_day,
            f.comment_ratio, f.hour_entropy, f.subreddit_count,
            f.active_days, f.duplicate_ratio, f.avg_post_interval,
            f.avg_comment_interval, f.night_activity_ratio,
            f.burstiness_score, f.engagement_rate
        FROM scores s
        LEFT JOIN features f ON s.account_id = f.account_id
        """
    )


@st.cache_data(ttl=30)
def load_isolation_forest_data() -> pd.DataFrame:
    """Load behavioural features plus persisted anomaly values."""
    return read_df(
        """
        SELECT
            f.account_id,
            f.age_days, f.posts_per_day, f.comments_per_day,
            f.comment_ratio, f.karma_score, f.avg_score,
            f.subreddit_count, f.active_days, f.hour_entropy,
            f.duplicate_ratio, f.avg_post_interval, f.avg_comment_interval,
            f.night_activity_ratio, f.burstiness_score, f.engagement_rate,
            s.anomaly_score, s.influence_score, s.tier
        FROM features f
        LEFT JOIN scores s ON f.account_id = s.account_id
        """
    )


@st.cache_data(ttl=30)
def load_collection_stats() -> dict:
    counts = {}
    for key, table in [("posts", "posts"), ("comments", "comments"), ("accounts", "accounts")]:
        df = read_df(f"SELECT COUNT(*) AS n FROM {table}")
        counts[key] = int(df.iloc[0]["n"]) if not df.empty else 0

    subreddit_df = read_df("SELECT COUNT(DISTINCT subreddit) AS n FROM posts WHERE subreddit IS NOT NULL")
    topic_df = read_df(
        """
        SELECT topic, COUNT(*) AS n FROM (
            SELECT topic FROM posts WHERE topic IS NOT NULL
            UNION ALL
            SELECT topic FROM comments WHERE topic IS NOT NULL
        ) GROUP BY topic ORDER BY n DESC
        """
    )
    counts["subreddits"] = int(subreddit_df.iloc[0]["n"]) if not subreddit_df.empty else 0
    return {**counts, "topics": topic_df}


@st.cache_data(ttl=30)
def load_edges() -> pd.DataFrame:
    return read_df(
        """
        SELECT source_account_id, target_account_id,
               edge_type, SUM(weight) AS weight
        FROM edges
        GROUP BY source_account_id, target_account_id, edge_type
        """
    )


@st.cache_data(ttl=30)
def load_clusters() -> pd.DataFrame:
    return read_df(
        """
        SELECT cluster_id, COUNT(*) AS accounts,
               AVG(influence_score) AS avg_influence,
               MAX(influence_score) AS max_influence
        FROM scores
        WHERE cluster_id IS NOT NULL AND cluster_id != -1
        GROUP BY cluster_id
        ORDER BY avg_influence DESC
        """
    )


@st.cache_data(ttl=30)
def load_amplified_events() -> pd.DataFrame:
    return read_df(
        """
        SELECT ce.source_account_id, ce.target_account_id,
               ce.source_post_id, ce.target_post_id,
               ce.similarity, ce.event_time,
               s1.cluster_id
        FROM coordination_events ce
        LEFT JOIN scores s1 ON ce.source_account_id = s1.account_id
        LEFT JOIN scores s2 ON ce.target_account_id = s2.account_id
        WHERE ce.event_type = 'near_duplicate_content'
          AND ce.source_account_id IS NOT NULL
          AND ce.target_account_id IS NOT NULL
        ORDER BY ce.similarity DESC
        """
    )


@st.cache_data(ttl=30)
def load_temporal_pairs() -> pd.DataFrame:
    return read_df(
        """
        SELECT source_account_id, target_account_id,
               similarity, avg_time_diff
        FROM temporal_similarity
        ORDER BY similarity DESC
        """
    )


@st.cache_data(ttl=30)
def load_pair_scores() -> pd.DataFrame:
    return read_df(
        """
        SELECT source_account_id, target_account_id,
               content_score, temporal_score, network_score,
               final_score, coordination_type
        FROM account_pairs
        ORDER BY final_score DESC
        """
    )


@st.cache_data(ttl=30)
def load_metrics() -> pd.DataFrame:
    return read_df(
        """
        SELECT model_name, accuracy, precision, recall, f1, roc_auc, created_at
        FROM model_metrics
        ORDER BY created_at DESC
        """
    )


@st.cache_data(ttl=30)
def load_dataset_metadata() -> pd.DataFrame:
    return read_df(
        """
        SELECT collection_date, subreddits, total_posts,
               total_comments, total_accounts, notes
        FROM dataset_metadata
        ORDER BY id DESC
        """
    )


@st.cache_data(ttl=30)
def load_content_lookup(ids: tuple) -> dict:
    if not ids:
        return {}
    placeholders = ",".join(["?"] * len(ids))
    post_df = read_df(
        f"SELECT id, COALESCE(title,'') || ' ' || COALESCE(text,'') AS text FROM posts WHERE id IN ({placeholders})",
        ids,
    )
    comment_df = read_df(
        f"SELECT id, COALESCE(text,'') AS text FROM comments WHERE id IN ({placeholders})",
        ids,
    )
    both = pd.concat([post_df, comment_df], ignore_index=True)
    if both.empty:
        return {}
    return dict(zip(both["id"], both["text"]))


# ----------------------------- helpers ------------------------------------

def score_label(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "Not scored"
    value = float(value)
    if value <= TIERS["organic"][1]:
        return "Organic"
    if value <= TIERS["suspicious"][1]:
        return "Suspicious"
    return "Coordinated"


def evidence_strength(row: pd.Series) -> str:
    components = [
        row.get("anomaly_score"),
        row.get("coord_score"),
        row.get("dup_score"),
        row.get("network_score"),
    ]
    valid = [float(x) for x in components if pd.notna(x)]
    if len(valid) < 2:
        return "Limited evidence"
    strong = sum(v >= 0.70 for v in valid)
    if strong >= 3:
        return "Multi-signal evidence"
    if strong == 2:
        return "Converging evidence"
    return "Single-dominant signal"


def safe_metric(df: pd.DataFrame, col: str, agg: str = "count", default=0):
    if df.empty or col not in df.columns:
        return default
    s = df[col].dropna()
    if s.empty:
        return default
    return getattr(s, agg)()


def tier_counts(df: pd.DataFrame) -> pd.Series:
    if df.empty or "tier" not in df.columns:
        return pd.Series(dtype="int64")
    return df["tier"].value_counts().reindex(TIERS.keys(), fill_value=0)


def callout(kind: str, tag: str, text: str):
    """A single, reusable evidence-note style — replaces one-off HTML boxes."""
    st.markdown(
        f"<div class='callout {kind}'><span class='tag'>{tag}</span>{text}</div>",
        unsafe_allow_html=True,
    )


def tier_legend():
    dots = "".join(
        f"<span class='dot' style='background:{TIER_COLORS[t]}'></span>{t.capitalize()}"
        for t in TIERS.keys()
    )
    st.markdown(f"<div class='legend-row mono'>{dots}</div>", unsafe_allow_html=True)


def page_title():
    st.markdown(
        """
        <div class='masthead'>
          <span class='eyebrow'>Reddit dataset · Nepal · AID System</span>
          <h1>InfluenceWatch Nepal</h1>
          <p>Behavioural, content, clustering and network signals combined into one
          evidence trail for reviewing possible coordinated amplification.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )



def style_plot(fig, height=560, title=None, show_legend=True):
    """Apply a consistent, large, readable dashboard visualization style."""
    fig.update_layout(
        height=height,
        autosize=True,
        paper_bgcolor=PLOT_PAPER,
        plot_bgcolor=PLOT_BG,
        font=dict(
            family="Inter, Arial, sans-serif",
            size=14,
            color=PLOT_TEXT,
        ),
        title=dict(
            text=title or "",
            x=0.02,
            xanchor="left",
            font=dict(size=18, color=PLOT_TEXT),
        ),
        margin=dict(l=55, r=35, t=55 if title else 25, b=55),
        showlegend=show_legend,
        legend=dict(
            bgcolor="rgba(15,18,22,.78)",
            bordercolor=PLOT_GRID,
            borderwidth=1,
            font=dict(size=12),
        ),
        hoverlabel=dict(
            bgcolor="#0F1216",
            bordercolor="#4B5563",
            font=dict(size=13, color=PLOT_TEXT),
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=PLOT_GRID,
        gridwidth=1,
        zeroline=False,
        tickfont=dict(size=12, color="#B7C0CC"),
        title_font=dict(size=13, color="#D7DCE3"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=PLOT_GRID,
        gridwidth=1,
        zeroline=False,
        tickfont=dict(size=12, color="#B7C0CC"),
        title_font=dict(size=13, color="#D7DCE3"),
    )
    return fig


def _normalize_anomaly(values: pd.Series) -> pd.Series:
    """Map values to 0-1 so larger values mean more anomalous."""
    s = pd.to_numeric(values, errors="coerce")
    if s.notna().sum() == 0:
        return s
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def prepare_isolation_forest_view() -> tuple[pd.DataFrame, str]:
    """Prepare the data used by the dedicated Isolation Forest dashboard."""
    df = load_isolation_forest_data().copy()
    if df.empty:
        return df, "No feature records are available."

    feature_cols = [
        c for c in [
            "age_days", "posts_per_day", "comments_per_day", "comment_ratio",
            "karma_score", "avg_score", "subreddit_count", "active_days",
            "hour_entropy", "duplicate_ratio", "avg_post_interval",
            "avg_comment_interval", "night_activity_ratio",
            "burstiness_score", "engagement_rate",
        ]
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any()
    ]

    if len(feature_cols) < 2:
        return df, "Not enough numeric behavioural features are available."

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([float("inf"), float("-inf")], pd.NA)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    stored = pd.to_numeric(df["anomaly_score"], errors="coerce") if "anomaly_score" in df else pd.Series(dtype=float)

    # Prefer the persisted project score so the visualization matches the
    # existing scoring pipeline. Only fall back to a dashboard computation
    # when no persisted anomaly scores exist.
    if stored.notna().sum() >= 2:
        df["if_anomaly"] = _normalize_anomaly(stored)
        source = "Using persisted Isolation Forest anomaly_score values from the scoring pipeline."
    elif IsolationForest is not None and len(df) >= 10:
        model = IsolationForest(
            n_estimators=250,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X)
        decision = pd.Series(model.decision_function(X), index=df.index)
        df["if_anomaly"] = _normalize_anomaly(-decision)
        df["if_prediction"] = model.predict(X)
        source = (
            "No persisted anomaly scores were found; this visualization is "
            "computed from the stored engineered behavioural features."
        )
    else:
        return df, "Isolation Forest cannot be rendered yet because the model is unavailable or too few accounts have features."

    df["if_anomaly"] = pd.to_numeric(df["if_anomaly"], errors="coerce")
    return df, source


def render_isolation_forest(df_scores: pd.DataFrame):
    """Render a large, dedicated Isolation Forest anomaly diagram."""
    st.subheader("Isolation Forest — behavioural anomaly detection")
    st.caption(
        "Each point is an account. Higher anomaly values indicate greater behavioural isolation from the rest of the population."
    )

    if_df, source = prepare_isolation_forest_view()
    if if_df.empty or "if_anomaly" not in if_df.columns:
        st.warning(source)
        return

    st.markdown(
        f"<div class='if-card'><div class='if-title'>Model source</div>{source}</div>",
        unsafe_allow_html=True,
    )

    candidate_pairs = [
        ("posts_per_day", "comments_per_day"),
        ("burstiness_score", "night_activity_ratio"),
        ("hour_entropy", "duplicate_ratio"),
        ("engagement_rate", "comment_ratio"),
    ]
    available_pairs = [
        (x, y) for x, y in candidate_pairs
        if x in if_df.columns and y in if_df.columns
        and pd.to_numeric(if_df[x], errors="coerce").notna().any()
        and pd.to_numeric(if_df[y], errors="coerce").notna().any()
    ]

    if not available_pairs:
        st.warning("No compatible behavioural feature pair is available for the Isolation Forest diagram.")
        return

    pair_labels = [
        f"{x.replace('_', ' ').title()} × {y.replace('_', ' ').title()}"
        for x, y in available_pairs
    ]
    choice = st.selectbox("Behavioural dimensions", pair_labels, index=0)
    x_col, y_col = available_pairs[pair_labels.index(choice)]

    keep = ["account_id", x_col, y_col, "if_anomaly"]
    if "tier" in if_df.columns:
        keep.append("tier")
    plot_df = if_df[keep].copy()

    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_col, y_col, "if_anomaly"]).copy()

    if plot_df.empty:
        st.warning("No complete records are available for the selected dimensions.")
        return

    plot_df["anomaly_level"] = pd.cut(
        plot_df["if_anomaly"],
        bins=[-0.001, 0.50, 0.75, 1.001],
        labels=["Normal", "Elevated", "Anomalous"],
    )
    plot_df["account_label"] = plot_df["account_id"].astype(str).str[:12]
    plot_df["point_size"] = 10 + plot_df["if_anomaly"] * 22

    left, right = st.columns([1.75, 1])

    with left:
        st.markdown("<div class='viz-title'>Isolation Forest anomaly map</div>", unsafe_allow_html=True)

        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color="if_anomaly",
            color_continuous_scale=["#5FBE8D", "#E3A548", "#E56A65"],
            size="point_size",
            hover_name="account_label",
            hover_data={
                x_col: ":.3f",
                y_col: ":.3f",
                "if_anomaly": ":.3f",
                "anomaly_level": True,
                "account_id": True,
            },
            labels={
                x_col: x_col.replace("_", " ").title(),
                y_col: y_col.replace("_", " ").title(),
                "if_anomaly": "Anomaly score",
            },
        )
        fig.update_traces(
            marker=dict(opacity=0.86, line=dict(width=1.4, color="#0B0E12"))
        )
        fig = style_plot(
            fig,
            height=760,
            title="Isolation Forest behavioural outlier space",
            show_legend=False,
        )
        fig.update_coloraxes(
            colorbar=dict(
                title=dict(text="Anomaly", font=dict(size=13, color="#D7DCE3")),
                tickfont=dict(size=12, color="#D7DCE3"),
            )
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False},
        )

    with right:
        st.markdown("<div class='viz-title'>Most isolated accounts</div>", unsafe_allow_html=True)
        top = plot_df.sort_values("if_anomaly", ascending=False).head(15).copy()
        top["account"] = top["account_id"].astype(str).str[:18]
        top = top.sort_values("if_anomaly", ascending=True)

        bar = px.bar(
            top,
            x="if_anomaly",
            y="account",
            orientation="h",
            labels={"if_anomaly": "Anomaly score", "account": "Account"},
        )
        bar = style_plot(
            bar,
            height=760,
            title="Top anomaly scores",
            show_legend=False,
        )
        st.plotly_chart(bar, use_container_width=True)

    p90 = float(plot_df["if_anomaly"].quantile(0.90))
    p95 = float(plot_df["if_anomaly"].quantile(0.95))

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Accounts visualized", f"{len(plot_df):,}")
    q2.metric("Median anomaly", f"{plot_df['if_anomaly'].median():.3f}")
    q3.metric("90th percentile", f"{p90:.3f}")
    q4.metric("95th percentile", f"{p95:.3f}")

    st.markdown("### Isolation Forest interpretation")
    callout(
        "note",
        "How to read it",
        "Isolation Forest highlights points that are easier to isolate in the engineered behavioural feature space. "
        "A high anomaly score is an investigation signal, not proof that an account is automated, malicious, or coordinated.",
    )

    if "anomaly_score" in df_scores.columns:
        persisted = df_scores[
            [c for c in ["account_id", "anomaly_score", "influence_score", "tier", "cluster_id"]
             if c in df_scores.columns]
        ].copy()
        persisted["anomaly_score"] = pd.to_numeric(
            persisted["anomaly_score"], errors="coerce"
        )
        persisted = (
            persisted.dropna(subset=["anomaly_score"])
            .sort_values("anomaly_score", ascending=False)
            .head(20)
        )
        if not persisted.empty:
            st.markdown("#### Highest persisted anomaly scores")
            st.dataframe(
                persisted,
                hide_index=True,
                use_container_width=True,
                height=430,
            )


def networkx_cluster_graph(cluster_df, score_df, cluster_id, max_nodes=80):
    """Build a clear NetworkX subgraph for one HDBSCAN cluster."""
    members = score_df[
        score_df["cluster_id"].notna()
        & (score_df["cluster_id"].astype(str) == str(cluster_id))
    ].copy()

    if members.empty:
        return None, None

    members = members.sort_values("influence_score", ascending=False).head(max_nodes)
    ids = set(members["account_id"].tolist())

    edges = load_edges()
    if edges.empty:
        return None, members

    e = edges[
        edges["source_account_id"].isin(ids)
        & edges["target_account_id"].isin(ids)
    ].copy()

    G = nx.Graph()
    for _, r in members.iterrows():
        G.add_node(
            r["account_id"],
            score=float(r["influence_score"]) if pd.notna(r["influence_score"]) else 0.0,
            tier=str(r["tier"]),
        )

    for _, r in e.iterrows():
        u, v = r["source_account_id"], r["target_account_id"]
        if u != v:
            G.add_edge(u, v, weight=float(r["weight"]))

    if len(G) == 0:
        return None, members

    # Larger k + more iterations separates dense clusters and makes labels legible.
    k = max(0.8, 2.2 / max(len(G.nodes) ** 0.35, 1))
    pos = nx.spring_layout(
        G,
        seed=42,
        k=k,
        iterations=180,
        scale=3.0,
        weight="weight",
    )
    return (G, pos), members


# ----------------------------- sections -----------------------------------

def render_sidebar(df: pd.DataFrame, stats: dict):
    with st.sidebar:
        st.header("Controls")
        st.caption("Filter to investigate evidence — tier is a starting point, not a verdict.")

        tier_options = list(TIERS.keys())
        selected_tiers = st.multiselect(
            "Tier", tier_options,
            default=tier_options,
            format_func=lambda x: x.capitalize(),
        )
        min_score = st.slider("Minimum Influence Score", 0, 100, 0)
        max_rows = st.slider("Rows per table", 10, 250, 50)

        st.divider()
        st.subheader("Collection coverage")
        posts = stats.get("posts", 0)
        accounts = stats.get("accounts", 0)
        comments = stats.get("comments", 0)
        post_target = 5000
        account_target = 1000
        st.progress(min(posts / post_target, 1.0), text=f"Posts — {posts:,} / {post_target:,}")
        st.progress(min(accounts / account_target, 1.0), text=f"Accounts — {accounts:,} / {account_target:,}")
        st.metric("Comments", f"{comments:,}")

        st.divider()
        st.subheader("Detection settings")
        st.markdown(
            f"<span class='mono'>cosine threshold — {COSINE_THRESHOLD:.2f}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<span class='mono'>"
            + " · ".join(
                [
                    f"A {WEIGHTS.get('anomaly', 0):.0%}",
                    f"C {WEIGHTS.get('coordination', 0):.0%}",
                    f"D {WEIGHTS.get('duplication', 0):.0%}",
                    f"N {WEIGHTS.get('network', 0):.0%}",
                ]
            )
            + "</span>",
            unsafe_allow_html=True,
        )
        st.caption("A · anomaly   C · coordination   D · duplication   N · network")

        return selected_tiers, min_score, max_rows


def render_overview(df: pd.DataFrame, stats: dict):
    st.subheader("Overview")

    if df.empty:
        st.warning("No scores are currently available. Run the analysis pipeline first.")
        return

    scored = df[df["influence_score"].notna()].copy()
    counts = tier_counts(scored)

    c = st.columns(5)
    c[0].metric("Posts", f"{stats['posts']:,}")
    c[1].metric("Comments", f"{stats['comments']:,}")
    c[2].metric("Accounts", f"{stats['accounts']:,}")
    c[3].metric("Scored accounts", f"{len(scored):,}")
    c[4].metric("Non-noise clusters", f"{len(load_clusters()):,}")

    if stats["posts"] < 5000:
        callout(
            "limit", "Collection",
            f"{stats['posts']:,} of the proposal's 5,000-post target collected. "
            "Treat coverage as partial until this closes.",
        )
    else:
        callout("good", "Collection", "Post-collection target reached — 5,000+ posts in the database.")

    st.markdown("### Influence distribution")
    if scored.empty:
        st.info("The score table exists but has no populated Influence Scores yet.")
        return

    tier_legend()

    col1, col2 = st.columns([1.4, 1])
    with col1:
        fig = px.histogram(
            scored,
            x="influence_score",
            color="tier",
            nbins=30,
            color_discrete_map=TIER_COLORS,
            category_orders={"tier": list(TIERS.keys())},
            labels={"influence_score": "Influence Score", "tier": "Tier"},
        )
        fig = style_plot(fig, height=560, title="Influence score distribution", show_legend=True)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        tier_df = counts.rename_axis("tier").reset_index(name="accounts")
        fig2 = px.pie(
            tier_df, names="tier", values="accounts",
            color="tier", color_discrete_map=TIER_COLORS,
            hole=0.58,
        )
        fig2 = style_plot(fig2, height=560, title="Accounts by influence tier", show_legend=True)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Reading the score")
    callout(
        "note", "Score",
        "The Influence Score weighs evidence — it isn't a probability or proof of coordination. "
        "Read it alongside the anomaly, cluster, content and network signals behind it.",
    )


def render_accounts(df: pd.DataFrame, selected_tiers, min_score: int, max_rows: int):
    st.subheader("Account investigation")
    scored = df[df["influence_score"].notna()].copy()
    if scored.empty:
        st.info("No composite scores are available yet.")
        return

    filtered = scored[
        scored["tier"].isin(selected_tiers)
        & (scored["influence_score"] >= min_score)
    ].copy()
    filtered["evidence"] = filtered.apply(evidence_strength, axis=1)
    filtered = filtered.sort_values("influence_score", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Accounts matching filters", len(filtered))
    c2.metric("Highest score", f"{filtered['influence_score'].max():.1f}" if not filtered.empty else "-")
    c3.metric("Median score", f"{filtered['influence_score'].median():.1f}" if not filtered.empty else "-")

    show = filtered.head(max_rows).copy()
    cols = [
        "account_id", "influence_score", "tier", "evidence",
        "anomaly_score", "coord_score", "dup_score", "network_score",
        "cluster_id", "posts_per_day", "hour_entropy", "duplicate_ratio",
    ]
    cols = [c for c in cols if c in show.columns]
    st.dataframe(show[cols], use_container_width=True, height=430)

    st.download_button(
        "Export filtered accounts (CSV)",
        data=show[cols].to_csv(index=False),
        file_name="account_investigation.csv",
        mime="text/csv",
    )

    st.markdown("### Evidence for this account")
    account_ids = list(filtered["account_id"].head(500))
    if not account_ids:
        st.info("No accounts match the current filters.")
        return

    selected = st.selectbox("Select an account", account_ids)
    row = filtered[filtered["account_id"] == selected].iloc[0]

    st.markdown(
        f"<span class='dot' style='background:{TIER_COLORS.get(row['tier'], '#98A2B0')}'></span>"
        f"<span class='mono'>{str(row['tier']).capitalize()} · {row['evidence']}</span>",
        unsafe_allow_html=True,
    )

    a, b, c, d = st.columns(4)
    a.metric("Influence", f"{row['influence_score']:.1f}")
    b.metric("Anomaly", f"{row['anomaly_score']:.2f}" if pd.notna(row['anomaly_score']) else "-")
    c.metric("Coordination", f"{row['coord_score']:.2f}" if pd.notna(row['coord_score']) else "-")
    d.metric("Network", f"{row['network_score']:.2f}" if pd.notna(row['network_score']) else "-")

    evidence_df = pd.DataFrame({
        "Signal": ["Behavioural anomaly", "Coordination cluster", "Content duplication", "Network structure"],
        "Normalized value": [row.get("anomaly_score"), row.get("coord_score"), row.get("dup_score"), row.get("network_score")],
        "Weight": [WEIGHTS.get("anomaly"), WEIGHTS.get("coordination"), WEIGHTS.get("duplication"), WEIGHTS.get("network")],
    })
    evidence_df["Contribution"] = evidence_df["Normalized value"] * evidence_df["Weight"] * 100
    st.dataframe(evidence_df, hide_index=True, use_container_width=True)

    st.caption("A score built from several converging signals carries more weight than one driven by a single component.")


def render_network(df: pd.DataFrame):
    st.subheader("Interaction network")
    scored = df[df["influence_score"].notna()].copy()
    edges = load_edges()

    if scored.empty or edges.empty:
        st.info("Network data are not available yet. Run the NetworkX stage first.")
        return

    top_n = st.slider(
        "Top accounts by Influence Score",
        10, 150, 50,
        help="More accounts reveal more structure, but can make the graph denser."
    )
    top = scored.sort_values("influence_score", ascending=False).head(top_n)
    ids = set(top["account_id"])
    e = edges[
        edges["source_account_id"].isin(ids)
        & edges["target_account_id"].isin(ids)
    ].copy()

    if e.empty:
        st.info("No edges connect the selected high-scoring accounts. Increase the account count.")
        return

    G = nx.DiGraph()
    for _, r in top.iterrows():
        G.add_node(
            r["account_id"],
            score=float(r["influence_score"]),
            tier=str(r["tier"]),
        )
    for _, r in e.iterrows():
        G.add_edge(
            r["source_account_id"],
            r["target_account_id"],
            weight=float(r["weight"]),
        )

    # NetworkX layout: increased spacing and iterations for a cleaner large graph.
    k = max(0.9, 2.8 / max(len(G.nodes) ** 0.35, 1))
    pos = nx.spring_layout(
        G,
        seed=42,
        k=k,
        iterations=220,
        scale=4.0,
        weight="weight",
    )

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1.15, color="#667085"),
        hoverinfo="none",
        opacity=0.60,
    )

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_scores = [G.nodes[n]["score"] for n in G.nodes()]
    node_colors = [
        TIER_COLORS.get(G.nodes[n]["tier"], "#98A2B0")
        for n in G.nodes()
    ]

    # Larger node range, with a visible outline against the dark background.
    node_sizes = [
        max(16, min(48, 16 + s * 0.42))
        for s in node_scores
    ]

    labels = [
        f"<b>{n}</b><br>"
        f"Influence: {G.nodes[n]['score']:.1f}<br>"
        f"Tier: {G.nodes[n]['tier']}<br>"
        f"Degree: {G.degree[n]}"
        for n in G.nodes()
    ]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[str(n)[:12] for n in G.nodes()],
        textposition="top center",
        textfont=dict(size=10, color="#E8EBF0"),
        hovertext=labels,
        hoverinfo="text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            opacity=0.96,
            line=dict(width=2.2, color="#0B0E12"),
        ),
    )

    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        height=900,
        autosize=True,
        paper_bgcolor="#12161C",
        plot_bgcolor="#12161C",
        font=dict(family="Inter, Arial, sans-serif", size=14, color="#ECEDF0"),
        showlegend=False,
        margin=dict(l=20, r=20, t=25, b=20),
        hoverlabel=dict(
            bgcolor="#0F1216",
            bordercolor="#667085",
            font=dict(size=14, color="#ECEDF0"),
        ),
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            fixedrange=False,
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            fixedrange=False,
            scaleanchor="x",
            scaleratio=1,
        ),
    )

    st.markdown(
        "<div class='viz-title'>NetworkX force-directed interaction graph</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "scrollZoom": True,
            "displaylogo": False,
        },
    )

    type_df = (
        e.groupby("edge_type", dropna=False)["weight"]
        .sum()
        .reset_index()
        .sort_values("weight", ascending=False)
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Accounts shown", len(G.nodes))
    c2.metric("Interactions shown", len(G.edges))
    c3.metric("Network density", f"{nx.density(G.to_undirected()):.3f}")

    if not type_df.empty:
        st.markdown("### Interaction types")
        edge_fig = px.bar(
            type_df,
            x="edge_type",
            y="weight",
            labels={"edge_type": "Interaction type", "weight": "Total weight"},
        )
        edge_fig = style_plot(
            edge_fig,
            height=480,
            title="Interaction weight by edge type",
            show_legend=False,
        )
        st.plotly_chart(edge_fig, use_container_width=True)

        st.dataframe(type_df, hide_index=True, use_container_width=True)

    st.markdown(
        "<div class='network-note'>"
        "<b>Interpretation:</b> Edges represent Reddit reply/comment interactions. "
        "Node size reflects Influence Score and node colour reflects tier. "
        "Network structure is evidence for investigation, not proof of intent or coordination."
        "</div>",
        unsafe_allow_html=True,
    )


def render_clusters(df: pd.DataFrame):
    st.subheader("Coordination clusters")
    clusters = load_clusters()
    temporal = load_temporal_pairs()
    pairs = load_pair_scores()

    if clusters.empty:
        st.info("No non-noise HDBSCAN clusters are currently stored.")
    else:
        cluster_df = df[df["cluster_id"].notna()].copy()
        cluster_df["cluster_label"] = cluster_df["cluster_id"].astype(str)

        left, right = st.columns([1.65, 1])

        with left:
            st.markdown(
                "<div class='viz-title'>HDBSCAN behavioural cluster map</div>",
                unsafe_allow_html=True,
            )
            fig = px.scatter(
                cluster_df,
                x="hour_entropy",
                y="posts_per_day",
                color="cluster_label",
                color_discrete_sequence=CLUSTER_PALETTE,
                hover_data=[
                    "account_id",
                    "influence_score",
                    "coord_score",
                    "duplicate_ratio",
                ],
                labels={
                    "cluster_label": "Cluster",
                    "hour_entropy": "Posting-hour entropy",
                    "posts_per_day": "Posts/day",
                },
            )
            fig.update_traces(
                marker=dict(size=14, opacity=0.88, line=dict(width=1.2, color="#0B0E12"))
            )
            fig = style_plot(
                fig,
                height=680,
                title="Behavioural separation of detected clusters",
                show_legend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown("#### Cluster summary")
            st.dataframe(
                clusters,
                hide_index=True,
                use_container_width=True,
                height=680,
            )

        # Actual NetworkX graph for a selected cluster.
        cluster_ids = clusters["cluster_id"].astype(str).tolist()
        selected_cluster = st.selectbox(
            "Inspect NetworkX graph for cluster",
            cluster_ids,
            help="Select a stored HDBSCAN cluster to see its internal interaction structure.",
        )

        result, members = networkx_cluster_graph(
            clusters,
            df,
            selected_cluster,
            max_nodes=80,
        )

        if result is None:
            st.info(
                "This cluster has no stored internal interaction edges. "
                "The HDBSCAN cluster still exists, but there is no NetworkX graph to render."
            )
        else:
            G, pos = result

            edge_x, edge_y = [], []
            for u, v in G.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=1.4, color="#667085"),
                opacity=0.65,
                hoverinfo="none",
            )

            node_x = [pos[n][0] for n in G.nodes()]
            node_y = [pos[n][1] for n in G.nodes()]
            scores = [G.nodes[n]["score"] for n in G.nodes()]
            colors = [
                TIER_COLORS.get(G.nodes[n]["tier"], "#98A2B0")
                for n in G.nodes()
            ]

            sizes = [max(20, min(52, 20 + s * 0.42)) for s in scores]
            labels = [
                f"<b>{n}</b><br>"
                f"Influence: {G.nodes[n]['score']:.1f}<br>"
                f"Tier: {G.nodes[n]['tier']}<br>"
                f"Connections: {G.degree[n]}"
                for n in G.nodes()
            ]

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=[str(n)[:12] for n in G.nodes()],
                textposition="top center",
                textfont=dict(size=10, color="#E8EBF0"),
                hovertext=labels,
                hoverinfo="text",
                marker=dict(
                    size=sizes,
                    color=colors,
                    opacity=0.97,
                    line=dict(width=2.4, color="#080A0D"),
                ),
            )

            cluster_fig = go.Figure([edge_trace, node_trace])
            cluster_fig.update_layout(
                height=900,
                autosize=True,
                paper_bgcolor="#12161C",
                plot_bgcolor="#12161C",
                font=dict(
                    family="Inter, Arial, sans-serif",
                    size=14,
                    color="#ECEDF0",
                ),
                showlegend=False,
                margin=dict(l=20, r=20, t=55, b=20),
                title=dict(
                    text=f"NetworkX structure — Cluster {selected_cluster}",
                    x=0.02,
                    xanchor="left",
                    font=dict(size=19, color="#ECEDF0"),
                ),
                hoverlabel=dict(
                    bgcolor="#0F1216",
                    bordercolor="#667085",
                    font=dict(size=14, color="#ECEDF0"),
                ),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    scaleanchor="x",
                    scaleratio=1,
                ),
            )

            st.markdown(
                "<div class='viz-title'>NetworkX internal cluster structure</div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                cluster_fig,
                use_container_width=True,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                },
            )

            gc1, gc2, gc3 = st.columns(3)
            gc1.metric("Cluster accounts", len(G.nodes))
            gc2.metric("Internal interactions", len(G.edges))
            gc3.metric(
                "Avg. degree",
                f"{sum(dict(G.degree()).values()) / max(len(G.nodes), 1):.1f}",
            )

    st.markdown("### Coordination evidence")
    c1, c2, c3 = st.columns(3)
    c1.metric("Stored temporal pairs", len(temporal))
    c2.metric("Stored account pairs", len(pairs))
    c3.metric("Near-duplicate coordination events", len(load_amplified_events()))

    if not temporal.empty:
        st.markdown("#### Strongest temporal similarities")
        st.dataframe(
            temporal.head(25),
            hide_index=True,
            use_container_width=True,
            height=480,
        )
    elif pairs.empty:
        callout(
            "limit", "Pending",
            "Temporal and pairwise coordination tables are wired up but not yet populated for this dataset.",
        )


def render_content():
    st.subheader("Content & duplication")
    events = load_amplified_events()
    stats = load_collection_stats()

    topic_df = stats.get("topics", pd.DataFrame())
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("### Topics in the dataset")
        if topic_df.empty:
            st.info("No topic counts are available.")
        else:
            fig = px.bar(
                topic_df.sort_values("n"), x="n", y="topic", orientation="h",
                labels={"n": "Posts + comments", "topic": "Topic"},
            )
            fig = style_plot(fig, height=600, title="Topics in the dataset", show_legend=False)
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("### Near-duplicate events")
        st.metric("Detected pairs", len(events))
        st.metric("Threshold", f"{COSINE_THRESHOLD:.2f}")
        if not events.empty:
            dist = events["similarity"].describe()
            st.dataframe(dist.to_frame("similarity").round(3), use_container_width=True)

    if events.empty:
        st.info("No near-duplicate coordination events are stored yet.")
        return

    events = events.copy()
    ids = tuple(set(events["source_post_id"].dropna()) | set(events["target_post_id"].dropna()))
    text_map = load_content_lookup(ids)
    events["source_text"] = events["source_post_id"].map(text_map)
    events["target_text"] = events["target_post_id"].map(text_map)

    st.markdown("### Highest-similarity content pairs")
    for _, row in events.head(20).iterrows():
        label = f"{str(row['source_account_id'])[:12]}… ↔ {str(row['target_account_id'])[:12]}…  ·  similarity {row['similarity']:.3f}"
        with st.expander(label):
            a, b = st.columns(2)
            a.markdown(f"**Source:** `{row['source_account_id']}`")
            a.write(row.get("source_text") or "Text unavailable")
            b.markdown(f"**Target:** `{row['target_account_id']}`")
            b.write(row.get("target_text") or "Text unavailable")

    st.download_button(
        "Export duplicate pairs (CSV)",
        data=events.to_csv(index=False),
        file_name="near_duplicate_evidence.csv",
        mime="text/csv",
    )


def render_benchmark():
    st.subheader("Benchmark & validation")
    st.caption("Only stored evaluation runs are shown — an empty table means the run hasn't happened yet, not that it scored zero.")
    metrics = load_metrics()

    if metrics.empty:
        callout(
            "limit", "Pending",
            "No benchmark runs stored yet. The Cresci-2017 / TwiBot-22 evaluation path is wired up and waiting on a run.",
        )
    else:
        show = metrics.copy()
        for c in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            if c in show.columns:
                show[c] = (show[c] * 100).round(2)
        st.dataframe(show, hide_index=True, use_container_width=True)

        melted = show.melt(
            id_vars=["model_name"],
            value_vars=[c for c in ["accuracy", "precision", "recall", "f1", "roc_auc"] if c in show.columns],
            var_name="metric", value_name="percent",
        )
        fig = px.bar(
            melted, x="model_name", y="percent", color="metric", barmode="group",
            labels={"percent": "Metric (%)", "model_name": "Model"},
        )
        fig = style_plot(fig, height=600, title="Topics in the dataset", show_legend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Proposal validation targets")
    target_df = pd.DataFrame({
        "Component": [
            "Isolation Forest",
            "HDBSCAN",
            "TF-IDF + cosine",
            "Composite Influence Score",
        ],
        "Proposal target": [
            "ROC-AUC ≥ 0.80",
            "Cluster purity ≥ 0.75",
            "Precision ≥ 0.85 @ similarity 0.90",
            "Accuracy ≥ 0.78",
        ],
        "Status": [
            "Held-out evaluation not yet run",
            "Cluster purity not yet validated",
            "Threshold/precision sweep not yet run",
            "End-to-end evaluation not yet run",
        ],
    })
    st.dataframe(target_df, hide_index=True, use_container_width=True)

    callout("note", "Targets", "These are proposal targets, not results — mark one achieved only after a reproducible evaluation run.")


def render_data_quality():
    st.subheader("Data & scope")
    stats = load_collection_stats()
    meta = load_dataset_metadata()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Posts", f"{stats['posts']:,}")
    c2.metric("Comments", f"{stats['comments']:,}")
    c3.metric("Accounts", f"{stats['accounts']:,}")
    c4.metric("Subreddits", f"{stats['subreddits']:,}")

    st.markdown("### Collection metadata")
    if meta.empty:
        st.info("No collection metadata records are stored yet.")
    else:
        st.dataframe(meta.head(10), hide_index=True, use_container_width=True)

    st.markdown("### Scope")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("**Can identify**")
        st.caption("Unusual account behaviour, dense behavioural clusters, near-duplicate content, structurally influential accounts.")
    with s2:
        st.markdown("**Can support**")
        st.caption("Investigating possible coordinated amplification when independent signals converge.")
    with s3:
        st.markdown("**Can't prove alone**")
        st.caption("That an account is a bot, a cluster is malicious, or a pattern was intentionally coordinated.")

    st.markdown("### Next steps")
    st.markdown(
        "- TwiBot-22 and additional benchmark loaders\n"
        "- Held-out evaluation run\n"
        "- Temporal pair similarity\n"
        "- Ablation and threshold-sensitivity testing\n"
        "- Production deployment"
    )


# ----------------------------- app ----------------------------------------

def main():
    page_title()
    stats = load_collection_stats()
    df = load_scores()
    selected_tiers, min_score, max_rows = render_sidebar(df, stats)

    tabs = st.tabs([
        "Overview",
        "Isolation Forest",
        "Accounts",
        "Network",
        "Clusters",
        "Content",
        "Benchmark",
        "Data & scope",
    ])

    with tabs[0]:
        render_overview(df, stats)
    with tabs[1]:
        render_isolation_forest(df)
    with tabs[2]:
        render_accounts(df, selected_tiers, min_score, max_rows)
    with tabs[3]:
        render_network(df)
    with tabs[4]:
        render_clusters(df)
    with tabs[5]:
        render_content()
    with tabs[6]:
        render_benchmark()
    with tabs[7]:
        render_data_quality()


if __name__ == "__main__":
    main()