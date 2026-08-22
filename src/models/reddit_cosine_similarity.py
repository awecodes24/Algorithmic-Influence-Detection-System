# src/models/reddit_cosine_similarity.py
# TF-IDF + cosine similarity near-duplicate content detection on REAL
# Reddit data. Implements the proposal's Eq. 4.6-4.8 (Ch. 4.3).
#
# Two outputs, matching the two places the schema already reserves for this:
#   - content_similarity: one row per ACCOUNT PAIR that shares at least one
#     near-duplicate item, holding the highest similarity found between them
#   - scores.dup_score: one value per ACCOUNT -- the fraction of that
#     account's own posts+comments that duplicate something posted by a
#     DIFFERENT account. This is what feeds the composite Influence Score.

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from src.db import get_conn
from src.config import COSINE_THRESHOLD

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# Below this many characters, TF-IDF cosine similarity is unreliable --
# short text ("lol", "this", "same") trivially matches unrelated short
# text and produces meaningless "near-duplicates". 20 is a rough floor,
# not a tuned value -- revisit if the output below looks like junk.
MIN_TEXT_LENGTH = 20

# Very short texts can produce misleadingly high TF-IDF similarity.
# A match must contain enough words to be considered meaningful
# coordination evidence.
MIN_WORDS_FOR_COORDINATION = 8

# Reddit's own placeholders for removed/deleted content -- identical
# across thousands of unrelated accounts by construction, and would
# otherwise dominate the results as false "coordination".
PLACEHOLDER_TEXT = {"[deleted]", "[removed]", ""}


def load_content(relevant_only=True):
    """
    Pulls post titles+text and comment text as one combined corpus of
    "content items", each tagged with the account that authored it.
    Posts and comments are treated the same way -- coordinated
    near-duplicate content can show up in either.

    relevant_only: when True (default), only pulls rows where
    is_relevant=1 -- the flag collector.classify_topic() already sets
    on every row against HISTORICAL_SEARCH_TERMS/TOPIC_KEYWORDS at
    collection time. collect_posts() pulls r/<subreddit>/new/ wholesale,
    so a chunk of what lands in `posts`/`comments` is on-subreddit but
    off-topic; comparing those rows for near-duplicates just adds noise
    two unrelated accounts could trivially "match" on (a subreddit rule,
    a flair template) and never reflects real coordination. is_relevant
    is stored and indexed (db.py: idx_post_relevance/idx_comment_relevance)
    but nothing here filtered on it until now.

    Set relevant_only=False to compare against the unfiltered run --
    classify_topic()'s own keyword list is admittedly incomplete for
    Romanized Nepali (see its docstring), so on a small dataset this
    filter could cut out real on-topic rows it simply didn't catch.
    Run both and compare pair counts before trusting either alone.
    """
    conn = get_conn()
    relevance_clause = " WHERE is_relevant = 1" if relevant_only else ""
    posts = pd.read_sql(
        "SELECT id, account_id, created_utc, "
        "COALESCE(title, '') || ' ' || COALESCE(text, '') AS content "
        f"FROM posts{relevance_clause}", conn
    )
    comments = pd.read_sql(
        f"SELECT id, account_id, created_utc, text AS content FROM comments{relevance_clause}", conn
    )
    conn.close()

    posts['source'] = 'post'
    comments['source'] = 'comment'
    df = pd.concat([posts, comments], ignore_index=True)

    df['content'] = df['content'].fillna('').str.strip()
    before = len(df)
    df = df[
        (df['content'].str.len() >= MIN_TEXT_LENGTH) &
        (~df['content'].isin(PLACEHOLDER_TEXT))
    ].reset_index(drop=True)
    logger.info(
        f"Loaded {before} content items (relevant_only={relevant_only}), "
        f"kept {len(df)} after dropping short/empty/deleted text "
        f"(min length={MIN_TEXT_LENGTH} chars)"
    )
    return df


def find_near_duplicates(df):
    """
    Vectorizes all content with TF-IDF, then uses NearestNeighbors with
    cosine distance to find every pair above COSINE_THRESHOLD. This
    avoids ever building a full N x N similarity matrix, which gets
    slow and memory-heavy once N reaches a few thousand items -- the
    "efficient content comparison" the proposal calls for in Sec 1.6,
    as opposed to comparing every pair with a Python loop.

    stop_words is left off deliberately: content here is a mix of
    English, Nepali, and Romanized Nepali (per collector.py's langdetect
    tagging), and sklearn's built-in stopword list only covers English --
    applying it would strip English filler words but leave Nepali ones
    untouched, an inconsistency that's safer to avoid than to have.
    """
    if len(df) < 2:
        raise ValueError(f"Only {len(df)} content items after filtering -- nothing to compare.")

    logger.info(f"Vectorizing {len(df)} content items with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=20000)
    X = vectorizer.fit_transform(df['content'])

    # radius is a cosine DISTANCE, not a similarity -- NearestNeighbors
    # wants 1 - similarity_threshold here.
    radius = 1 - COSINE_THRESHOLD
    logger.info(f"Searching for pairs with cosine similarity >= {COSINE_THRESHOLD}...")
    nn = NearestNeighbors(metric='cosine', radius=radius)
    nn.fit(X)
    distances, indices = nn.radius_neighbors(X)

    pairs = []
    seen = set()
    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        for dist, j in zip(dists, idxs):
            if i == j:
                continue
            pair_key = (min(i, j), max(i, j))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            acc_i, acc_j = df.at[i, 'account_id'], df.at[j, 'account_id']

            if acc_i == acc_j:
                # Same account duplicating its own content is spam, not
                # coordination BETWEEN accounts.
                continue


            # ---------------------------------------------------------------
            # Reject very short matches.
            #
            # Short generic comments can have high TF-IDF cosine similarity
            # without representing coordinated behavior. Require both pieces
            # of content to contain enough words before treating the pair as
            # meaningful duplicate-content evidence.
            # ---------------------------------------------------------------
            text_i = df.at[i, 'content']
            text_j = df.at[j, 'content']

            word_count_i = len(str(text_i).split())
            word_count_j = len(str(text_j).split())

            if (
                word_count_i < MIN_WORDS_FOR_COORDINATION
                or word_count_j < MIN_WORDS_FOR_COORDINATION
            ):
                continue

            pairs.append({
                'item_i': i,
                'item_j': j,

                'account_i': acc_i,
                'account_j': acc_j,

                'content_id_i': df.at[i, 'id'],
                'content_id_j': df.at[j, 'id'],

                'content_type_i': df.at[i, 'source'],
                'content_type_j': df.at[j, 'source'],

                'created_utc_i': df.at[i, 'created_utc'],
                'created_utc_j': df.at[j, 'created_utc'],

                'similarity': 1 - dist
            })

    logger.info(f"Found {len(pairs)} cross-account near-duplicate content pairs")
    return pd.DataFrame(pairs, columns=[
        'item_i',
        'item_j',

        'account_i',
        'account_j',

        'content_id_i',
        'content_id_j',

        'content_type_i',
        'content_type_j',

        'created_utc_i',
        'created_utc_j',

        'similarity'
    ])


def compute_account_pair_scores(pairs_df):
    """content_similarity: one row per account pair, highest similarity found."""
    if len(pairs_df) == 0:
        return pd.DataFrame(columns=['source_account_id', 'target_account_id', 'similarity'])

    pairs_df = pairs_df.copy()
    pairs_df['source_account_id'] = pairs_df[['account_i', 'account_j']].min(axis=1)
    pairs_df['target_account_id'] = pairs_df[['account_i', 'account_j']].max(axis=1)

    return (
        pairs_df.groupby(['source_account_id', 'target_account_id'])['similarity']
        .max()
        .reset_index()
    )


def compute_dup_scores(df, pairs_df):
    """
    scores.dup_score: fraction of each account's OWN content items that
    are a near-duplicate of some OTHER account's content.
    """
    content_counts = df.groupby('account_id').size().rename('total_items')

    if len(pairs_df) == 0:
        # NOT just "no data" -- pd.Series(dtype='int64', name='dup_items')
        # with no index arg gets a nameless empty RangeIndex. concat()
        # below aligns by index, and when one side's index is named
        # 'account_id' (from content_counts) and the other's isn't, pandas
        # drops the name entirely instead of picking one, so the later
        # reset_index() produces a column called 'index' instead of
        # 'account_id' and save_results() dies with KeyError: 'account_id'.
        # An explicitly-named empty index keeps it consistent with the
        # non-empty branch below, where groupby('account_id') already
        # names it correctly.
        dup_counts = pd.Series(dtype='int64', name='dup_items', index=pd.Index([], name='account_id'))
    else:
        dup_item_indices = pd.concat([
            pairs_df[['item_i', 'account_i']].rename(columns={'item_i': 'item', 'account_i': 'account_id'}),
            pairs_df[['item_j', 'account_j']].rename(columns={'item_j': 'item', 'account_j': 'account_id'}),
        ]).drop_duplicates(subset=['item', 'account_id'])
        dup_counts = dup_item_indices.groupby('account_id').size().rename('dup_items')

    result = pd.concat([content_counts, dup_counts], axis=1).fillna(0)
    result['dup_score'] = (result['dup_items'] / result['total_items']).clip(0, 1)
    return result[['dup_score']].reset_index()


def save_results(account_pairs, dup_scores):
    conn = get_conn()
    c = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    
    # Remove previous TF-IDF similarity results so stale matches
    # from earlier thresholds or filtering settings do not remain.
    c.execute("""
        DELETE FROM content_similarity
        WHERE method = 'tfidf_cosine'
    """)

    for _, row in account_pairs.iterrows():
        c.execute("""
            INSERT INTO content_similarity (source_account_id, target_account_id, similarity, method)
            VALUES (?, ?, ?, 'tfidf_cosine')
            ON CONFLICT(source_account_id, target_account_id) DO UPDATE SET
                similarity = excluded.similarity,
                method = excluded.method
        """, (row['source_account_id'], row['target_account_id'], float(row['similarity'])))

    for _, row in dup_scores.iterrows():
        # Same pattern as the other two model scripts: only touches
        # dup_score + scored_at, never overwrites anomaly_score or
        # coord_score if either already ran for this account.
        c.execute("""
            INSERT INTO scores (account_id, dup_score, scored_at)
            VALUES (?, ?, ?)
            ON CONFLICT(account_id) DO UPDATE SET
                dup_score = excluded.dup_score,
                scored_at = excluded.scored_at
        """, (row['account_id'], float(row['dup_score']), now))

    conn.commit()
    conn.close()
    logger.info(
        f"Saved {len(account_pairs)} account-pair rows to content_similarity, "
        f"{len(dup_scores)} dup_score values to scores"
    )


def save_coordination_events(pairs_df):
    """
    Save one row per near-duplicate content pair.

    source_post_id and target_post_id are retained for backward
    compatibility with the existing database schema, but the actual
    content type is explicitly stored in:

        source_content_type
        target_content_type

    Therefore the dashboard/evidence inspector can correctly retrieve
    matched posts or comments.
    """

    conn = get_conn()
    c = conn.cursor()

    # Remove previous cosine-similarity events before inserting the
    # current run. This prevents duplicate evidence accumulation.
    c.execute("""
        DELETE FROM coordination_events
        WHERE event_type = 'near_duplicate_content'
    """)

    if pairs_df.empty:
        conn.commit()
        conn.close()

        logger.info(
            "No near-duplicate coordination events to save."
        )
        return

    for _, row in pairs_df.iterrows():

        event_time = None

        if (
            pd.notna(row['created_utc_i'])
            and pd.notna(row['created_utc_j'])
        ):
            event_time = float(
                max(
                    row['created_utc_i'],
                    row['created_utc_j']
                )
            )

        c.execute("""
            INSERT INTO coordination_events (

                source_account_id,
                target_account_id,

                source_post_id,
                target_post_id,

                source_content_type,
                target_content_type,

                event_type,
                similarity,
                event_time

            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (

            row['account_i'],
            row['account_j'],

            row['content_id_i'],
            row['content_id_j'],

            row['content_type_i'],
            row['content_type_j'],

            'near_duplicate_content',

            float(row['similarity']),

            event_time
        ))

    conn.commit()
    conn.close()

    logger.info(
        f"Saved {len(pairs_df)} near-duplicate "
        f"coordination events"
    )

def print_top_duplicated(dup_scores, n=20):
    print(f"\n{'━'*50}")
    print(f"  TOP {n} ACCOUNTS BY DUPLICATION SCORE")
    print(f"{'━'*50}")
    top = dup_scores.sort_values('dup_score', ascending=False).head(n)
    for _, row in top.iterrows():
        print(f"{row['account_id'][:18]:<20}{row['dup_score']:>10.3f}")
    print(f"{'━'*50}\n")


if __name__ == "__main__":
    # Flip to False to compare against the unfiltered run -- see
    # load_content()'s docstring for why you'd want to check both before
    # trusting either one on a dataset you haven't sized yet.
    RELEVANT_ONLY = False

    logger.info("Running TF-IDF cosine similarity on real Reddit content\n")

    df = load_content(relevant_only=RELEVANT_ONLY)
    pairs_df = find_near_duplicates(df)
    account_pairs = compute_account_pair_scores(pairs_df)
    dup_scores = compute_dup_scores(df, pairs_df)
    save_results(account_pairs, dup_scores)
    save_coordination_events(pairs_df)
    print_top_duplicated(dup_scores)

    logger.info(
        f"NOTE: MIN_TEXT_LENGTH={MIN_TEXT_LENGTH} and "
        f"COSINE_THRESHOLD={COSINE_THRESHOLD} are starting points, "
        "not tuned values. If the top accounts above look like false "
        "positives, raise one or both; if a campaign you already know "
        "about isn't showing up, lower them."
    )