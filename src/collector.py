"""
src/collector.py

Collect Reddit data using Apify and save it into SQLite.
Compatible with the db.py schema.

Required in .env:
    APIFY_API_TOKEN=...              (required)
    STORE_RAW_USERNAME=false         (optional, default false -- see note below)

Changes from the original version, and why:

  * `includeMediaLinks` is now set in the actor input. Per Apify's own docs
    for trudax/reddit-scraper-lite: without it, the actor uses a fast RSS
    path that OMITS upVotes, upVoteRatio and numberOfComments entirely.
    Both were already being read (item.get("upVotes", 0) etc.) -- without
    this flag they were silently defaulting to 0 on every item.

* comments.post_id is read directly from item.get("postId"), which the
  actor DOES populate on comment items (confirmed against live output on
  YYYY-MM-DD) -- already correctly t3_-prefixed, so no parent-chain walk
  is needed. An earlier version of this file assumed postId/linkId were
  absent based on the actor's documented schema; that assumption didn't
  match observed behavior once real debug output was checked.


  * raw usernames are no longer written to the accounts table by default.
    Previously `username` was stored in plaintext right next to the
    SHA-256 `id` in the same row, which makes every account trivially
    reversible for anyone with DB access -- not what "anonymized" in the
    proposal (Sec 3.4.1) describes. See STORE_RAW_USERNAME below.

  * collect_user_profiles() / save_user_profiles() are new: the original
    scrape never populated accounts.created_utc / comment_karma /
    link_karma, which blocked the age_days and karma_score features.

  * language detection (langdetect) is wired in, matching the tools list
    in Sec 3.3.1 -- it wasn't used anywhere in the original file.

  * to_unix() now logs when it falls back to "now" instead of failing
    silently, so a bad batch of timestamps doesn't quietly corrupt every
    temporal feature downstream.

  * the topic keyword list has a small seed of common Romanized Nepali
    political terms. This is NOT a comprehensive Nepali-language list --
    treat it as a starting point, not authoritative coverage. The most
    reliable way to improve it is to sample rows where is_relevant=0 and
    language != 'en' from your own data and add whatever you see getting
    missed.

Further changes made in this pass -- closing the gap between what db.py's
posts/comments tables define and what was actually being written, and
fixing two bugs found while re-checking every field against Apify's own
published example output for trudax/reddit-scraper-lite (the JSON examples
on the actor's Store page, not assumed field names):

  * posts.sentiment / comments.sentiment were columns that nothing ever
    wrote to -- every row had NULL sentiment regardless of content. A
    VADER (vaderSentiment) compound score is now computed for both, the
    same lightweight, no-model-download approach as langdetect for
    language. VADER's lexicon is English-only -- see analyze_sentiment()'s
    docstring, same caveat as detect_language() has for Romanized Nepali.

  * posts.edited / comments.edited were also never written (silently
    resting on the schema's DEFAULT 0 for every row). extract_edited()
    now checks for the field defensively -- but to be transparent: as of
    this rewrite, trudax/reddit-scraper-lite's *documented* output schema
    does not expose an edited flag under any name, for either posts or
    comments, so this will in practice still be 0 for every row today.
    It's wired in so nothing needs to change here if the actor adds the
    field later.

  * comment self-replies were meant to be excluded (see the note in the
    comment branch below) by comparing item.get("author") to
    item.get("parentAuthor") -- but neither field exists anywhere in this
    actor's documented output (only "username" is present; there is no
    "parentAuthor" at all). That comparison was always None == None and
    never actually skipped anything. Fixed by resolving the parent's
    author from an id -> anonymized-account-id map built from the current
    batch plus what's already in the DB (see author_map in save_items),
    so the comparison is real and still never touches a raw username.

  * subreddit resolution for comments was falling through to
    communityName (e.g. "r/nepal") instead of a clean name, because
    parsedCommunityName -- which posts do carry -- does not exist on
    comment items at all; comments instead expose the clean name under
    "category". Posts were fine, but comments were silently getting a
    different, inconsistently-formatted subreddit value than posts from
    the same subreddit, which breaks anything that groups/joins on it.
    normalize_subreddit() now checks "category" too and strips any
    leading "r/" regardless of which field ends up used, so posts and
    comments always agree.

  * backfill_enrichment() is new: recomputes topic/language/sentiment for
    rows already sitting in the DB from *before* this rewrite, purely
    from the text already stored (no Apify calls, no cost). INSERT OR
    IGNORE means re-running collect_posts() never fixes old rows -- see
    progresslog.md Sec 7 -- and re-scraping everything just to backfill
    a column you already have the text for is a waste of Apify credits.
    Run with `python collector.py --backfill`.
"""

import os
import re
import sys
import hashlib
import logging
import time
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv

try:
    from langdetect import detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = 0  # deterministic results across runs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _sentiment_analyzer = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except ImportError:
    _sentiment_analyzer = None
    VADER_AVAILABLE = False

try:
    from src.db import get_conn, init_db
except ModuleNotFoundError:
    from db import get_conn, init_db


#################################################
# CONFIG
#################################################

load_dotenv()

APIFY_TOKEN = os.getenv("APIFY_API_TOKEN")
POSTS_ACTOR = "trudax/reddit-scraper-lite"

# If True, the raw Reddit username is stored alongside the hashed account_id
# in the accounts table, for local manual verification only. Default is
# False so what's actually stored matches the "anonymized" language in the
# proposal. NEVER enable this if influence.db (or any export/dashboard built
# from it) will be shared, committed, or shown to anyone outside the team.
STORE_RAW_USERNAME = os.getenv("STORE_RAW_USERNAME", "false").lower() == "true"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

_client = None


def get_client():
    """
    Lazily builds the Apify client, so importing this module (e.g. to reuse
    classify_topic() or anonymize() elsewhere) doesn't require an API key
    to already be set.
    """
    global _client
    if _client is None:
        if not APIFY_TOKEN:
            raise ValueError("APIFY_API_TOKEN not found in environment variables.")
        from apify_client import ApifyClient
        _client = ApifyClient(APIFY_TOKEN)
    return _client


#################################################
# HELPERS
#################################################

def anonymize(username: str) -> str:
    return hashlib.sha256(username.encode("utf-8")).hexdigest()[:16]


def get_content_hash(text: str):
    if not text:
        return None
    text = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

   
def to_unix(value, context=""):
    """
    Converts an epoch number or ISO8601 string to a unix timestamp.
    Falls back to "now" only when parsing truly fails, and logs it --
    the original version failed silently, which could let a batch of bad
    timestamps quietly corrupt every temporal feature (avg_post_interval,
    hour_entropy, burstiness_score, ...) without leaving a trace.
    """
    if value is None:
        return datetime.now(timezone.utc).timestamp()

    if isinstance(value, (int, float)):
        return float(value)

    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except Exception:
        logger.warning(f"Could not parse timestamp {value!r} ({context}); using now()")
        return datetime.now(timezone.utc).timestamp()


_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def detect_language(text):
    """
    Returns an ISO 639-1 language code, or None if the text is too short
    or detection fails.

    IMPORTANT, confirmed by testing during this rewrite: langdetect
    reliably identifies Devanagari-script Nepali ('ne'), which is why
    that case is short-circuited below rather than left to langdetect's
    statistical guess (which occasionally confuses Devanagari Nepali with
    Hindi). But on ROMANIZED Nepali -- "sarkar le bhrastachar roknu
    parcha" -- langdetect is unreliable and will confidently return the
    wrong language entirely (in testing: Swahili, Somali, and Turkish for
    different Romanized Nepali samples, never Nepali or English). Given
    the target subreddits will have real amounts of Romanized Nepali,
    treat `language` as reliable for English and Devanagari-script
    Nepali/Hindi only. Don't use it to filter Romanized content out of
    analysis -- it'll be tagged with whatever language langdetect
    happened to guess, not correctly excluded or included. Building a
    proper Romanized-Nepali detector is a reasonable next investment
    (e.g. a small hand-labeled sample + a simple classifier) but isn't
    attempted here.
    """
    if not text or len(text.strip()) < 3:
        return None

    if _DEVANAGARI_RE.search(text):
        return "ne"

    if not LANGDETECT_AVAILABLE:
        return None

    try:
        return detect(text)
    except LangDetectException:
        return None


def analyze_sentiment(text):
    """
    Returns a VADER compound sentiment score in [-1, 1] (-1 most negative,
    +1 most positive), or None if the text is too short or vaderSentiment
    isn't installed. Populates posts.sentiment / comments.sentiment, which
    the schema defined but nothing was ever writing to.

    VADER is rule/lexicon-based (no model download, fast enough to run on
    every row at collection time, tuned for short informal text -- caps
    lock, punctuation emphasis, emoji, negation, etc. -- which fits Reddit
    posts/comments better than a tool built for formal text would).

    IMPORTANT, same caveat as detect_language() above: VADER's lexicon is
    English-only. Confirmed by testing during this rewrite: on Romanized
    Nepali and on Devanagari Nepali it doesn't error, it just doesn't
    recognize any sentiment-bearing words and quietly returns 0.0 --
    which looks identical to "genuinely neutral text", not "no signal
    available". Treat this column as reliable only where language == 'en';
    a 0.0 next to a non-'en' row means "not measured", not "neutral".
    """
    if not text or len(text.strip()) < 3 or not VADER_AVAILABLE:
        return None
    return _sentiment_analyzer.polarity_scores(text)["compound"]


def normalize_username(item):
    """
    Returns the lowercased, stripped username for a raw Apify item, or
    None if it's missing, [deleted], or [removed]. Pulled out into its
    own function so the pre-pass that builds author_map/parent_map (see
    save_items) and the main insert pass can't quietly disagree about
    which items count as having a real author.
    """
    username = item.get("username") or item.get("author")
    if not username:
        return None
    username = username.strip()
    if username.lower() in ("[deleted]", "[removed]"):
        return None
    return username.lower()


def normalize_subreddit(item):
    """
    Returns a clean, lowercase subreddit name with no "r/" prefix, from
    whichever field the item actually carries it under.

    Confirmed against Apify's own published example output for
    trudax/reddit-scraper-lite: post items carry the clean name under
    parsedCommunityName (e.g. "HonkaiStarRail"). Comment items do NOT --
    parsedCommunityName never appears on a comment item -- they carry it
    under "category" instead. communityName exists on both but always
    keeps the "r/" prefix (e.g. "r/nepal"). Previously, comments fell
    straight through to communityName, so posts and comments from the
    same subreddit ended up with different stored values ("nepal" vs
    "r/nepal") -- breaking anything that groups/joins on it.
    Stripping a leading "r/" here regardless of which field is used means
    that can't happen even if Apify changes which field is populated.
    """
    raw = (
        item.get("parsedCommunityName")
        or item.get("category")
        or item.get("communityName")
        or item.get("subreddit")
        or ""
    )
    return re.sub(r"^r/", "", raw.strip(), flags=re.IGNORECASE).lower()


def extract_edited(item):
    """
    Returns 1 if the item is flagged as edited, else 0, for posts.edited /
    comments.edited.

    Transparency note: as of this rewrite, trudax/reddit-scraper-lite's
    *documented* output schema (checked against the actor's own published
    example post/comment JSON) does not expose an edited flag under any
    name, for either posts or comments. Checking a couple of plausible
    key names below is forward-compatible defensiveness in case the actor
    adds this later, not a confirmed field -- in practice this will
    currently still be 0 for every row, same end result as before, but
    now because the data genuinely isn't offered rather than because the
    column was silently skipped.
    """
    return 1 if item.get("edited", item.get("isEdited")) else 0


def resolve_post_id(parent_id, parent_map, max_hops=50):
    """
    comments.post_id needs to point at a *post* id (t3_ prefix). This
    actor's comment items only give us parentId, which for a top-level
    comment IS the post id, but for a nested reply is another comment's
    id (t1_ prefix) -- so walk up the chain via parent_map until hitting
    a t3_ id, or run out of chain.

    parent_map: dict of {comment_fullname: its_parent_fullname}, built
    from both the current batch and whatever's already in the DB, so
    chains spanning multiple collection runs still resolve.
    """
    current = parent_id
    hops = 0
    while current and not current.startswith("t3_") and hops < max_hops:
        current = parent_map.get(current)
        hops += 1
    return current if current and current.startswith("t3_") else None


def update_account_stats(cursor, account_id):
    cursor.execute(
        """
        UPDATE accounts
        SET
            total_posts = (SELECT COUNT(*) FROM posts WHERE account_id=?),
            total_comments = (SELECT COUNT(*) FROM comments WHERE account_id=?)
        WHERE id=?
        """,
        (account_id, account_id, account_id)
    )


#################################################
# TOPIC CLASSIFICATION
#################################################

# NOTE: this is a keyword-only classifier -- fast, but it only ever catches
# what's in these lists. A handful of common Romanized Nepali political
# terms are seeded below as a starting point (sarkar/government,
# pradhanmantri/PM, mantri/minister, rajniti(k)/politic(al), chunab/
# election, neta/leader, bhrastachar/corruption, andolan/movement-protest).
# This is NOT comprehensive -- Romanized Nepali has no single standard
# spelling (e.g. "andolan" vs "aandolan"), so even these will miss variants.
# Expand this based on your own data: pull a sample of rows where
# is_relevant=0 and language != 'en', read them, and add what's missing.

TOPIC_KEYWORDS = {

    "government": [
        "government", "minister", "prime minister", "pm", "president",
        "cabinet", "official", "administration", "authority",
        "sarkar", "pradhanmantri", "mantri"
    ],

    "politics": [
        "politics", "political", "party", "election", "vote", "campaign",
        "leader", "opposition", "coalition", "democracy",
        "rajniti", "rajnitik", "chunab", "neta"
    ],

    "government_policy": [
        "policy", "decision", "law", "bill", "constitution", "budget",
        "tax", "reform", "development", "regulation"
    ],

    "political_criticism": [
        "corruption", "scandal", "failure", "failed", "incompetent",
        "corrupt", "protest", "resign", "accountability", "criticism",
        "controversy", "mismanagement",
        "bhrastachar", "andolan"
    ],

    "political_entities": [
        "rsp", "r.s.p", "rastriya swatantra party", "rabi lamichhane",
        "rabi", "balen", "balendra shah", "maoist", "uml",
        "nepali congress", "congress", "nc"
    ],

    "political_events": [
        "gen z", "genz", "protest", "movement", "revolution", "corruption",
        "social media ban", "government change"
    ]
}


def classify_topic(text):
    if not text:
        return None, 0.0, 0

    text = text.lower()
    best_topic = None
    best_score = 0

    for topic, words in TOPIC_KEYWORDS.items():
        score = sum(
            1 for word in words
            if re.search(rf"\b{re.escape(word)}\b", text)
        )
        if score > best_score:
            best_score = score
            best_topic = topic

    if best_score < 1:
        return None, 0.0, 0

    return best_topic, float(best_score), 1


#################################################
# APIFY COLLECTION -- POSTS & COMMENTS
#################################################

def collect_posts(subreddits, max_posts=100, max_comments=20):
    start_urls = [{"url": f"https://www.reddit.com/r/{s}/new/"} for s in subreddits]

    run_input = {
        "startUrls": start_urls,
        "maxItems": max_posts,
        "maxPostCount": max_posts,
        "maxComments": max_comments,
        "maxRequestRetries": 2,
        # Without this the actor uses its fast RSS path, which omits
        # upVotes / upVoteRatio / numberOfComments entirely (confirmed
        # against the actor's own docs) -- see module docstring.
        "includeMediaLinks": True,
        "proxy": {"useApifyProxy": True}
    }

    logger.info("Starting Apify actor (posts/comments)...")
    client = get_client()

    try:
        run = client.actor(POSTS_ACTOR).call(
            run_input=run_input,
            run_timeout=timedelta(seconds=1800)
        )

        if run is None:
            logger.error("Actor run returned no result.")
            return [], None

        if run.status != "SUCCEEDED":
            logger.warning(
                f"Run finished with status={run.status!r} -- "
                f"dataset may be incomplete."
            )

        dataset_id = run.default_dataset_id
        items = list(client.dataset(dataset_id).iterate_items())
        logger.info(f"Collected {len(items)} items (run status: {run.status}).")
        return items, dataset_id

    except KeyboardInterrupt:
        logger.warning("Collection interrupted.")
        return [], None

    except Exception as e:
        logger.exception(f"Collection failed: {e}")
        return [], None



def collect_user_profiles(usernames, batch_size=15, delay_between_batches=30):
    """
    Fetches account-level profile data (karma, account creation date) for
    a list of RAW (non-anonymized) Reddit usernames, using the same actor
    pointed at each user's profile URL. skipUserPosts=True keeps this
    cheap and avoids re-collecting posts/comments already captured by
    collect_posts().

    batch_size kept small (15, down from 90) and delay_between_batches
    added (30s) because profile-page fetches get blocked/rate-limited by
    Reddit far more aggressively than subreddit-listing scraping does --
    confirmed across multiple test runs where large batches saw 403/429
    responses on most requests. Smaller, spaced-out batches trade total
    runtime for a meaningfully higher per-account success rate.

    Apify bills per result item ("pay per result"), so this adds roughly
    one extra billed item per unique account on top of the posts/comments
    run -- worth a small test batch (2-3 usernames) first to confirm the
    actual number of billed results before running it on ~1,000 accounts.

    usernames should be the RAW usernames still held in memory from this
    same run (see __main__) -- never persisted to disk except via the
    STORE_RAW_USERNAME debug flag.
    """
    if not usernames:
        return [], None

    all_items = []
    last_dataset_id = None
    client = get_client()

    num_batches = (len(usernames) + batch_size - 1) // batch_size

    for batch_num, i in enumerate(range(0, len(usernames), batch_size), start=1):
        batch = usernames[i:i + batch_size]
        start_urls = [{"url": f"https://www.reddit.com/user/{u}/"} for u in batch]

        run_input = {
            "startUrls": start_urls,
            "skipUserPosts": True,
            "maxRequestRetries": 3,
            "proxy": {"useApifyProxy": True}
        }

        logger.info(f"Fetching profile batch {batch_num}/{num_batches} ({len(batch)} profiles)...")

        try:
            run = client.actor(POSTS_ACTOR).call(
                run_input=run_input,
                run_timeout=timedelta(seconds=900)
            )

            if run is None:
                logger.error("User-profile run returned no result.")
                continue

            if run.status != "SUCCEEDED":
                logger.warning(f"User-profile run status={run.status!r}.")

            dataset_id = run.default_dataset_id
            last_dataset_id = dataset_id
            items = list(client.dataset(dataset_id).iterate_items())
            all_items.extend(items)
            logger.info(f"Batch {batch_num}/{num_batches}: got {len(items)} profile items.")

        except Exception as e:
            logger.exception(f"User-profile collection failed for batch at {i}: {e}")
            continue

        # Space batches out so we're not hammering Reddit back-to-back --
        # skip the sleep after the last batch.
        if batch_num < num_batches:
            logger.info(f"Waiting {delay_between_batches}s before next batch...")
            time.sleep(delay_between_batches)

    logger.info(f"Collected {len(all_items)} user profile items.")
    return all_items, last_dataset_id
    """
    Fetches account-level profile data (karma, account creation date) for
    a list of RAW (non-anonymized) Reddit usernames, using the same actor
    pointed at each user's profile URL. skipUserPosts=True keeps this
    cheap and avoids re-collecting posts/comments already captured by
    collect_posts().

    Apify bills per result item ("pay per result"), so this adds roughly
    one extra billed item per unique account on top of the posts/comments
    run -- worth a small test batch (2-3 usernames) first to confirm the
    actual number of billed results before running it on ~1,000 accounts.

    usernames should be the RAW usernames still held in memory from this
    same run (see __main__) -- never persisted to disk except via the
    STORE_RAW_USERNAME debug flag.
    """
    if not usernames:
        return [], None

    all_items = []
    last_dataset_id = None
    client = get_client()

    for i in range(0, len(usernames), batch_size):
        batch = usernames[i:i + batch_size]
        start_urls = [{"url": f"https://www.reddit.com/user/{u}/"} for u in batch]

        run_input = {
            "startUrls": start_urls,
            "skipUserPosts": True,
            "maxRequestRetries": 2,
            "proxy": {"useApifyProxy": True}
        }

        logger.info(f"Fetching {len(batch)} user profiles...")

        try:
            run = client.actor(POSTS_ACTOR).call(
                run_input=run_input,
                run_timeout=timedelta(seconds=900)
            )

            if run is None:
                logger.error("User-profile run returned no result.")
                continue

            if run.status != "SUCCEEDED":
                logger.warning(f"User-profile run status={run.status!r}.")

            dataset_id = run.default_dataset_id
            last_dataset_id = dataset_id
            items = list(client.dataset(dataset_id).iterate_items())
            all_items.extend(items)

        except Exception as e:
            logger.exception(f"User-profile collection failed for batch at {i}: {e}")
            continue

    logger.info(f"Collected {len(all_items)} user profile items.")
    return all_items, last_dataset_id


#################################################
# DATABASE SAVE -- POSTS & COMMENTS
#################################################

def save_items(items, subreddits=None, dataset_id=None):
    if not items:
        logger.warning("No items to save.")
        return {"posts": 0, "comments": 0, "users": 0, "usernames": []}

    conn = get_conn()
    c = conn.cursor()

    scraped_at = datetime.now(timezone.utc).timestamp()

    users = set()
    post_count = 0
    comment_count = 0

    try:
        # Pre-pass: build two lookup maps before touching any INSERT --
        #
        #   parent_map {comment_fullname: its_parent_fullname}
        #     lets resolve_post_id() walk a nested reply up to its root
        #     post, even across multiple collection runs.
        #
        #   author_map {item_fullname: anonymized_account_id}
        #     lets the comment branch below detect self-replies (a user
        #     replying to their own post/comment) by comparing anonymized
        #     ids -- never raw usernames. See module docstring for why
        #     the old author/parentAuthor comparison never worked.
        #
        # Both are seeded from what's already in the DB (so chains and
        # self-reply checks spanning multiple runs still resolve), then
        # filled in from the current batch, since Apify doesn't guarantee
        # a parent appears before its children in the same dataset.
        parent_map = {}
        author_map = {}

        c.execute("SELECT id, parent_id, account_id FROM comments")
        for row in c.fetchall():
            if row["parent_id"]:
                parent_map[row["id"]] = row["parent_id"]
            author_map[row["id"]] = row["account_id"]

        c.execute("SELECT id, account_id FROM posts")
        for row in c.fetchall():
            author_map[row["id"]] = row["account_id"]

        for item in items:
            fid = item.get("id")
            data_type = (item.get("dataType") or "").lower()
            if not fid or data_type not in ("post", "comment"):
                continue

            username_key = normalize_username(item)
            if username_key:
                author_map[fid] = anonymize(username_key)

            if data_type == "comment" and item.get("parentId"):
                parent_map[fid] = item["parentId"]
                
        # ── TEMPORARY DEBUG — remove after checking ──────────────────────────────
        debug_shown = 0
        for item in items:
            data_type = (item.get("dataType") or "").lower()
            if data_type == "comment" and debug_shown < 5:
                print(f"\n--- RAW COMMENT ITEM {debug_shown} ---")
                print("id:", item.get("id"))
                print("parentId:", item.get("parentId"))
                print("linkId:", item.get("linkId"))
                print("postId:", item.get("postId"))
                print("All keys:", list(item.keys()))
                debug_shown += 1
# ──────────────────────────────────────────────────────────────────────────

        for item in items:

            if not item.get("id"):
                continue

            data_type = (item.get("dataType") or "").lower()

            if data_type not in ("post", "comment"):
                continue

            username_key = normalize_username(item)
            if not username_key:
                continue

            anon_id = anonymize(username_key)

            created_ts = to_unix(item.get("createdAt"), context=f"item:{item.get('id')}")

            users.add(username_key)

            subreddit = normalize_subreddit(item)
            edited = extract_edited(item)

            #################################################
            # POSTS
            #################################################

            if data_type == "post":

                title = item.get("title", "")
                body = item.get("body", "") or item.get("text", "")
                text = f"{title} {body}".strip()

                if text.lower() in ("[deleted]", "[removed]"):
                    continue

                if not text:
                    continue

                topic, topic_score, is_relevant = classify_topic(text)
                language = detect_language(text)
                sentiment = analyze_sentiment(text)

                c.execute(
                    """
                    INSERT OR IGNORE INTO posts
                    (
                        id, account_id, subreddit, title, text, content_hash,
                        created_utc, scraped_at, score, num_comments, permalink,
                        edited, topic, topic_score, sentiment, language, is_relevant
                    )
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        item.get("id"),
                        anon_id,
                        subreddit,
                        title,
                        text,
                        get_content_hash(text),
                        created_ts,
                        scraped_at,
                        item.get("upVotes", 0),
                        item.get("numberOfComments", 0),
                        item.get("url", ""),
                        edited,
                        topic,
                        topic_score,
                        sentiment,
                        language,
                        is_relevant
                    )
                )

                if c.rowcount:
                    post_count += 1

                    c.execute(
                        """
                        INSERT OR IGNORE INTO account_activity
                        (account_id, activity_type, subreddit, created_utc)
                        VALUES (?,?,?,?)
                        """,
                        (anon_id, "post", subreddit, created_ts)
                    )
                    if (post_count + comment_count) % 500 == 0:
                        conn.commit()

            #################################################
            # COMMENTS
            #################################################

            elif data_type == "comment":

                parent_id = item.get("parentId")

                # Self-replies (a user replying to their own post/comment)
                # aren't an inter-account coordination signal, so they're
                # excluded here -- this also means they don't count toward
                # comments_per_day / engagement features, which is a
                # deliberate trade-off worth noting in your methodology.
                # Resolved via author_map (built above), comparing
                # anonymized ids -- see module docstring.
                if parent_id and author_map.get(parent_id) == anon_id:
                    continue

                body = item.get("body", "").strip()

                if body.lower() in ("[deleted]", "[removed]"):
                    continue

                if not body:
                    continue

                topic, topic_score, is_relevant = classify_topic(body)
                language = detect_language(body)
                sentiment = analyze_sentiment(body)

                post_id = item.get("postId")

                c.execute(
                    """
                    INSERT OR IGNORE INTO comments
                    (
                        id, account_id, post_id, parent_id, subreddit, text,
                        content_hash, created_utc, scraped_at, score, edited,
                        topic, topic_score, sentiment, language, is_relevant
                    )
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        item.get("id"),
                        anon_id,
                        post_id,
                        parent_id,
                        subreddit,
                        body,
                        get_content_hash(body),
                        created_ts,
                        scraped_at,
                        item.get("upVotes", 0),
                        edited,
                        topic,
                        topic_score,
                        sentiment,
                        language,
                        is_relevant
                    )
                )

                if c.rowcount:
                    comment_count += 1

                    c.execute(
                        """
                        INSERT OR IGNORE INTO account_activity
                        (account_id, activity_type, subreddit, created_utc)
                        VALUES (?,?,?,?)
                        """,
                        (anon_id, "comment", subreddit, created_ts)
                    )
                    if (post_count + comment_count) % 500 == 0:
                        conn.commit()

        #################################################
        # UPDATE ACCOUNTS
        #################################################

        for username in users:
            anon_id = anonymize(username)

            c.execute(
                """
                INSERT OR IGNORE INTO accounts (id, username)
                VALUES (?,?)
                """,
                (anon_id, username if STORE_RAW_USERNAME else None)
            )

            update_account_stats(c, anon_id)

        #################################################
        # DATASET METADATA
        #################################################

        c.execute(
            """
            INSERT INTO dataset_metadata
            (collection_date, subreddits, total_posts, total_comments, total_accounts, notes)
            VALUES (?,?,?,?,?,?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                ",".join(subreddits or []),
                post_count,
                comment_count,
                len(users),
                f"Actor={POSTS_ACTOR};Mode=posts_comments;Dataset={dataset_id or 'unknown'}"
            )
        )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

    logger.info(f"Posts inserted: {post_count}")
    logger.info(f"Comments inserted: {comment_count}")
    logger.info(f"Unique users: {len(users)}")
    logger.info(f"Total records: {post_count + comment_count}")

    return {
        "posts": post_count,
        "comments": comment_count,
        "users": len(users),
        # Raw usernames, kept only in memory for this run, so
        # collect_user_profiles() can be called without ever reading a
        # plaintext username back out of the database.
        "usernames": sorted(users)
    }


#################################################
# DATABASE SAVE -- USER PROFILES
#################################################

def save_user_profiles(items):
    """
    Updates accounts.created_utc / comment_karma / link_karma from
    "user"-dataType items returned by collect_user_profiles().

    Reddit's own API calls these "link karma" / "comment karma"; this
    actor's user items expose them as postKarma / commentKarma.
    """
    if not items:
        return {"profiles_updated": 0}

    conn = get_conn()
    c = conn.cursor()
    updated = 0

    try:
        for item in items:
            if (item.get("dataType") or "").lower() != "user":
                continue

            username = item.get("username")
            if not username:
                continue

            anon_id = anonymize(username.strip().lower())
            created_ts = to_unix(item.get("createdAt"), context=f"user:{anon_id}")

            c.execute(
                """
                UPDATE accounts
                SET created_utc = ?, comment_karma = ?, link_karma = ?
                WHERE id = ?
                """,
                (
                    created_ts,
                    item.get("commentKarma"),
                    item.get("postKarma"),
                    anon_id
                )
            )
            if c.rowcount:
                updated += 1

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

    logger.info(f"Updated profile data for {updated} accounts.")
    return {"profiles_updated": updated}


#################################################
# BACKFILL -- ENRICHMENT FOR ALREADY-COLLECTED ROWS
#################################################

def backfill_enrichment(batch_size=500):
    """
    Recomputes topic / topic_score / is_relevant / language / sentiment
    for rows already in posts/comments, purely from the text already
    stored in the DB -- no Apify calls, no cost. For rows collected
    before this rewrite added sentiment (or before a topic-keyword
    change), those columns are otherwise stuck at their old / NULL values
    forever, because INSERT OR IGNORE never touches a row that already
    exists (see progresslog.md Sec 7).

    Does NOT touch `edited` -- that comes from the raw Apify item, which
    isn't stored anywhere in the DB, so it can only be filled by
    re-collecting, not backfilled from what's already saved.
    """
    conn = get_conn()
    c = conn.cursor()
    updated = {"posts": 0, "comments": 0}

    try:
        for table in ("posts", "comments"):
            c.execute(f"SELECT id, text FROM {table}")
            rows = c.fetchall()

            for i, row in enumerate(rows, start=1):
                text = row["text"]
                topic, topic_score, is_relevant = classify_topic(text)
                language = detect_language(text)
                sentiment = analyze_sentiment(text)

                c.execute(
                    f"""
                    UPDATE {table}
                    SET topic=?, topic_score=?, is_relevant=?, language=?, sentiment=?
                    WHERE id=?
                    """,
                    (topic, topic_score, is_relevant, language, sentiment, row["id"])
                )
                updated[table] += 1

                if i % batch_size == 0:
                    conn.commit()
                    logger.info(f"Backfilled {i}/{len(rows)} {table}...")

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

    logger.info(f"Backfill complete: {updated}")
    return updated


#################################################
# PROGRESS REPORTING
#################################################

def report_progress(target_posts=5000, target_accounts=1000):
    """
    Prints cumulative totals across ALL collection runs so far -- a single
    run's max_posts cap won't reach the proposal's stated minimum
    (5,000 posts / 1,000 unique accounts, Sec 3.4.1) on its own.
    """
    conn = get_conn()
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM posts")
    total_posts = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM comments")
    total_comments = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM accounts")
    total_accounts = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM accounts WHERE created_utc IS NOT NULL")
    profiled_accounts = c.fetchone()[0]

    conn.close()

    logger.info(
        f"Progress: {total_posts}/{target_posts} posts | "
        f"{total_accounts}/{target_accounts} accounts "
        f"({profiled_accounts} with profile data) | "
        f"{total_comments} comments total"
    )


#################################################
# MAIN
#################################################

if __name__ == "__main__":

    init_db()

    if "--backfill" in sys.argv:
        logger.info("Running enrichment backfill on existing rows (no Apify calls)...")
        backfill_enrichment()
        report_progress()
        sys.exit(0)

    SUBREDDITS = [
        "Nepal",
        "NepaliPolitics",
        "nepalinews",
        "NepalSocial",
        "SouthAsia",
        "Kathmandu"
    ]

    # MAX_POSTS = 500
    # MAX_COMMENTS = 100
    MAX_POSTS = 10
    MAX_COMMENTS = 30

    logger.info("Starting Reddit collection...")

    items, dataset_id = collect_posts(
        subreddits=SUBREDDITS,
        max_posts=MAX_POSTS,
        max_comments=MAX_COMMENTS
    )

    if items:
        stats = save_items(items, SUBREDDITS, dataset_id)
        logger.info({k: v for k, v in stats.items() if k != "usernames"})

        # Backfill account-level profile data (karma, account age) for
        # every username seen in this batch, while the raw usernames are
        # still in memory -- see save_items()'s "usernames" and the
        # module docstring for why this doesn't touch the DB in plaintext.
        raw_usernames = stats.get("usernames", [])
        if raw_usernames:
            profile_items, profile_dataset_id = collect_user_profiles(raw_usernames)
            if profile_items:
                profile_stats = save_user_profiles(profile_items)
                logger.info(profile_stats)
    else:
        logger.warning("No data collected.")

    report_progress()
    logger.info("Collection completed successfully.")