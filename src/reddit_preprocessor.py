# src/reddit_preprocessor.py
# Computes behavioral features for Reddit accounts from raw posts/comments
# Run this AFTER collector.py has collected data into db.py's schema

import numpy as np
from datetime import datetime, timezone
from collections import defaultdict

try:
    from src.db import get_conn
except ModuleNotFoundError:
    from db import get_conn

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


NIGHT_HOURS = set(range(0, 6))  # 00:00–05:59 counts as "night" — adjust if needed


def compute_burstiness(timestamps):
    """
    Goh & Barabási burstiness parameter:
        B = (sigma_tau - mean_tau) / (sigma_tau + mean_tau)

    B ~ -1  ->  very regular/periodic posting (bot-like)
    B ~  0  ->  random/Poisson-like posting (organic baseline)
    B ~ +1  ->  bursty, clustered activity with long gaps between

    Needs at least 3 timestamps (2 gaps) to be meaningful.
    """
    if len(timestamps) < 3:
        return None

    ts = sorted(timestamps)
    gaps = np.diff(ts)

    mean_gap = gaps.mean()
    std_gap = gaps.std()

    if (std_gap + mean_gap) == 0:
        return 0.0

    return round(float((std_gap - mean_gap) / (std_gap + mean_gap)), 4)


def compute_hour_entropy(timestamps):
    """
    Shannon entropy of posting-hour distribution (0-23).
    Low entropy  -> posts concentrated in a few hours (bot-like schedule)
    High entropy -> posts spread naturally across the day (organic)
    Normalized to [0, 1] by dividing by max possible entropy (log2(24)).
    """
    if not timestamps:
        return None

    hours = [datetime.fromtimestamp(t, tz=timezone.utc).hour for t in timestamps]
    counts = np.bincount(hours, minlength=24)
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # avoid log(0)

    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(24)

    return round(float(entropy / max_entropy), 4)


def compute_night_activity_ratio(timestamps):
    if not timestamps:
        return 0.0
    hours = [datetime.fromtimestamp(t, tz=timezone.utc).hour for t in timestamps]
    night_count = sum(1 for h in hours if h in NIGHT_HOURS)
    return round(night_count / len(hours), 4)


def compute_avg_interval(timestamps):
    if len(timestamps) < 2:
        return None
    ts = sorted(timestamps)
    gaps = np.diff(ts)
    return round(float(gaps.mean()), 2)  # seconds


def compute_duplicate_ratio(content_hashes):
    """
    Fraction of an account's posts/comments that share a content_hash
    with at least one other item from the SAME account (self-duplication —
    e.g. copy-pasting the same message repeatedly). Cross-account
    duplication is a separate, later step (cosine similarity /
    content_similarity table) — this is just a cheap per-account signal.
    """
    if not content_hashes:
        return 0.0

    hashes = [h for h in content_hashes if h]
    if not hashes:
        return 0.0

    counts = defaultdict(int)
    for h in hashes:
        counts[h] += 1

    duplicated = sum(c for c in counts.values() if c > 1)
    return round(duplicated / len(hashes), 4)


def compute_features_for_account(account_id, posts, comments, karma_score):
    """
    posts / comments: lists of dicts with keys 'created_utc', 'score',
    'content_hash', 'subreddit'
    """
    all_items = posts + comments
    all_timestamps = [p['created_utc'] for p in all_items if p['created_utc']]
    post_timestamps = [p['created_utc'] for p in posts if p['created_utc']]
    comment_timestamps = [c['created_utc'] for c in comments if c['created_utc']]

    if not all_timestamps:
        return None  # nothing to compute from

    now = datetime.now(timezone.utc).timestamp()
    first_seen = min(all_timestamps)
    age_days = max((now - first_seen) / 86400, 1)

    n_posts = len(posts)
    n_comments = len(comments)
    total = n_posts + n_comments

    subreddits = {p['subreddit'] for p in all_items if p.get('subreddit')}
    active_days = len({
        datetime.fromtimestamp(t, tz=timezone.utc).date()
        for t in all_timestamps
    })

    scores = [p['score'] for p in all_items if p.get('score') is not None]
    avg_score = round(float(np.mean(scores)), 4) if scores else 0.0

    content_hashes = [p.get('content_hash') for p in all_items]

    features = {
        'account_id': account_id,
        'age_days': round(age_days, 2),
        'posts_per_day': round(n_posts / age_days, 4),
        'comments_per_day': round(n_comments / age_days, 4),
        'comment_ratio': round(n_comments / total, 4) if total else 0.0,
        'karma_score': karma_score if karma_score is not None else 0.0,
        'avg_score': avg_score,
        'subreddit_count': len(subreddits),
        'active_days': active_days,
        'hour_entropy': compute_hour_entropy(all_timestamps),
        'duplicate_ratio': compute_duplicate_ratio(content_hashes),
        'avg_post_interval': compute_avg_interval(post_timestamps),
        'avg_comment_interval': compute_avg_interval(comment_timestamps),
        'night_activity_ratio': compute_night_activity_ratio(all_timestamps),
        'burstiness_score': compute_burstiness(all_timestamps),
        'engagement_rate': round(sum(scores) / total, 4) if total and scores else 0.0,
        'computed_at': datetime.now(timezone.utc).isoformat()
    }

    return features


def compute_all_features():
    conn = get_conn()
    c = conn.cursor()

    # karma_score isn't a real accounts column in db.py -- it's computed
    # below from comment_karma + link_karma instead.
    c.execute("SELECT id, comment_karma, link_karma FROM accounts")
    accounts = c.fetchall()

    logger.info(f"Computing features for {len(accounts)} accounts...")

    saved = 0
    skipped = 0

    for account in accounts:
        account_id = account['id']
        comment_karma = account['comment_karma'] or 0
        link_karma = account['link_karma'] or 0
        karma_score = comment_karma + link_karma

        c.execute(
            "SELECT created_utc, score, content_hash, subreddit FROM posts WHERE account_id=?",
            (account_id,)
        )
        posts = [dict(row) for row in c.fetchall()]

        c.execute(
            "SELECT created_utc, score, content_hash, subreddit FROM comments WHERE account_id=?",
            (account_id,)
        )
        comments = [dict(row) for row in c.fetchall()]

        features = compute_features_for_account(account_id, posts, comments, karma_score)

        if features is None:
            skipped += 1
            continue

        c.execute("""
            INSERT OR REPLACE INTO features (
                account_id, age_days, posts_per_day, comments_per_day,
                comment_ratio, karma_score, avg_score, subreddit_count,
                active_days, hour_entropy, duplicate_ratio,
                avg_post_interval, avg_comment_interval,
                night_activity_ratio, burstiness_score, engagement_rate,
                computed_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            features['account_id'], features['age_days'], features['posts_per_day'],
            features['comments_per_day'], features['comment_ratio'], features['karma_score'],
            features['avg_score'], features['subreddit_count'], features['active_days'],
            features['hour_entropy'], features['duplicate_ratio'],
            features['avg_post_interval'], features['avg_comment_interval'],
            features['night_activity_ratio'], features['burstiness_score'],
            features['engagement_rate'], features['computed_at']
        ))
        saved += 1

    conn.commit()
    conn.close()

    logger.info(f"Features computed for {saved} accounts, skipped {skipped} (no activity)")
    return saved, skipped


if __name__ == "__main__":
    compute_all_features()