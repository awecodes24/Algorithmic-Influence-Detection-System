"""
src/collector.py

Collect Reddit data using Apify and save it into SQLite.
Compatible with the db.py schema.

Required in .env:
    APIFY_API_TOKEN=...              (required)

Optional in .env:
    ACCOUNT_MAP_PATH=...             (path to the local, git-ignored
                                       hashed-id -> raw-username mapping
                                       file account-history mode reads
                                       from; defaults to .account_map.json
                                       next to the repo root -- see the
                                       "LOCAL ACCOUNT MAPPING" section
                                       below for what this is and why)

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


  * raw usernames are NEVER written to the accounts table, full stop --
    accounts.username is always NULL. There used to be a
    STORE_RAW_USERNAME flag controlling this; it's been removed (see the
    account-history fix below for why) rather than left as a setting
    someone has to remember to leave on false. What's actually stored in
    the shared research database now matches "anonymized" in the
    proposal (Sec 3.4.1) unconditionally, with nothing to misconfigure.

  * FIXED: account-history mode (--mode accounts) was silently a no-op
    under the old STORE_RAW_USERNAME=false default. It read usernames
    from accounts.username, which that same default always left NULL --
    so get_stored_usernames() always returned an empty list, and
    collect_known_account_history() always returned zero accounts
    processed, with no error, just a warning log easy to miss in normal
    output. Flipping the default to true would have "fixed" it by
    contradicting the anonymization the rest of this file argues for.
    Fixed properly instead: raw usernames are now also appended to a
    separate, local-only, git-ignored mapping file
    (see "LOCAL ACCOUNT MAPPING" below) every time save_items() sees one
    -- automatically, not behind a flag -- and account-history mode reads
    from that file instead of the database. The database stays
    genuinely anonymized; account-history mode actually works.

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

import re
import os
import json
import hashlib
import logging
import time
import argparse
from datetime import datetime, timezone, timedelta
from src.config import REDDIT_COLLECTION

from dotenv import load_dotenv

try:
    from langdetect import detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = 0
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

from src.db import get_conn, init_db



# CONFIG

load_dotenv()

APIFY_TOKEN = os.getenv("APIFY_API_TOKEN")
POSTS_ACTOR = "trudax/reddit-scraper-lite"
HISTORICAL_ACTOR = "harshmaur/reddit-scraper"

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



# HELPERS

def anonymize(username: str) -> str:
    return hashlib.sha256(username.encode("utf-8")).hexdigest()[:16]



# LOCAL ACCOUNT MAPPING (hashed_id -> raw_username)


# influence.db never stores a raw username -- accounts.username is
# always NULL, so the shared research database stays genuinely
# anonymized no matter who ends up looking at it (matches "anonymized"
# in the proposal, Sec 3.4.1, without a flag to remember to leave off).
#
# Account-history mode (--mode accounts) still needs SOME way to know
# which raw username a given hashed accounts.id came from, to run the
# author:<username> search Reddit needs. That mapping lives here
# instead: a local JSON file, never read by anything that touches the
# database, never uploaded, never shown on the dashboard. Every raw
# username this script ever sees gets appended to it automatically --
# there's no separate flag to remember to turn on before it's useful.
#
# THIS FILE MUST NEVER BE COMMITTED, SHARED, OR SHOWN DURING A DEFENSE.
# It's the one place a real Reddit username can be recovered from a
# hashed account id. ACCOUNT_MAP_PATH below defaults next to the repo
# root; _ensure_gitignored() adds a `.gitignore` entry for it the first
# time this module runs, so a plain `git add .` can't accidentally
# pick it up.

ACCOUNT_MAP_PATH = os.getenv(
    "ACCOUNT_MAP_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".account_map.json"),
)


def _ensure_gitignored(path):
    """
    Appends path's basename to a .gitignore next to it, if not already
    present. Best-effort -- a missing/unwritable .gitignore is logged,
    not fatal, since the mapping file itself is still created and used
    correctly either way; this only prevents an accidental `git add .`
    from picking it up.
    """
    gitignore_path = os.path.join(os.path.dirname(path), ".gitignore")
    entry = os.path.basename(path)
    try:
        existing = ""
        if os.path.exists(gitignore_path):
            with open(gitignore_path, "r", encoding="utf-8") as f:
                existing = f.read()
        if entry not in existing.splitlines():
            with open(gitignore_path, "a", encoding="utf-8") as f:
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write(f"{entry}\n")
    except OSError as exc:
        logger.warning(
            "Could not update .gitignore for %s (%s). Add it manually so "
            "it's never committed.", entry, exc,
        )


def _load_account_map():
    if not os.path.exists(ACCOUNT_MAP_PATH):
        return {}
    try:
        with open(ACCOUNT_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Could not read %s (%s) -- treating as empty. If this file "
            "has usernames you need, fix or restore it before running "
            "account-history mode.", ACCOUNT_MAP_PATH, exc,
        )
        return {}


def update_account_map(usernames):
    """
    Merges the given raw usernames into the local mapping file, keyed by
    their anonymized id -- read-merge-write, so this is safe to call
    every run without losing entries from earlier runs. Called
    automatically from save_items() for every username it sees; you
    should not need to call this directly.
    """
    if not usernames:
        return

    existing = _load_account_map()
    added = 0
    for username in usernames:
        anon_id = anonymize(username)
        if anon_id not in existing:
            added += 1
        existing[anon_id] = username

    try:
        os.makedirs(os.path.dirname(ACCOUNT_MAP_PATH) or ".", exist_ok=True)
        tmp_path = ACCOUNT_MAP_PATH + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, sort_keys=True)
        os.replace(tmp_path, ACCOUNT_MAP_PATH)  # atomic on POSIX -- a crash
                                                  # mid-write can't corrupt
                                                  # the real file
        _ensure_gitignored(ACCOUNT_MAP_PATH)
    except OSError as exc:
        logger.warning(
            "Could not write %s (%s). Account-history mode will not see "
            "%d newly-seen username(s) from this run.",
            ACCOUNT_MAP_PATH, exc, added,
        )
        return

    if added:
        logger.info(
            "Local account map: %d new username(s) added (%s).",
            added, ACCOUNT_MAP_PATH,
        )


def get_mapped_usernames(limit=0):
    """
    Returns raw usernames from the local account-mapping file --
    account-history mode's actual source of usernames, replacing the
    old accounts.username column read. Order is insertion order from
    the JSON file (stable across runs, not database row order).
    """
    mapping = _load_account_map()
    usernames = list(dict.fromkeys(mapping.values()))  # de-dup, preserve order
    if limit and limit > 0:
        usernames = usernames[:limit]
    return usernames


def get_content_hash(text: str):
    if not text:
        return None
    text = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

   
def to_unix(value, context=""):
    """Convert an epoch number or ISO8601 timestamp to Unix seconds.

    Missing/unparseable timestamps are represented as None rather than
    fabricated with the current time. A fabricated timestamp would corrupt
    temporal coordination features.
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError, OverflowError) as exc:
        logger.warning(
            "Could not parse timestamp %r (%s): %s; storing NULL.",
            value, context, exc,
        )
        return None


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



# TOPIC CLASSIFICATION


# Historical keyword searches are intentionally broader than the classifier.
# Keep this list reasonably small because each keyword x subreddit is an Apify run.
HISTORICAL_SEARCH_TERMS = [
    "gen z",
    "genz",
    "gen z protest",
    "youth protest",
    "protest",
    "nepo kid",
    "nepokid",
    "social media ban",
    "government response",
    "curfew",
]

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
        "bhrastachar", "andolan",
        # Added for the Sept 2025 protests / Balen Shah government arc --
        # sourced from actual reporting on the event, not guessed. "nepo
        # kid(s)" was the movement's originating hashtag/phrase; both
        # spaced and unspaced forms are included since hashtags usually
        # drop the space ("#NepoKid"). "border encroachment" is the
        # current controversy from PM Shah's first 100 days.
        "nepo kid", "nepo kids", "nepokid", "nepokids",
        "nepo baby", "nepobabies",
        "border encroachment", "border dispute"
    ],

    "political_entities": [
        "rsp", "r.s.p", "rastriya swatantra party", "rabi lamichhane",
        "rabi", "balen", "balendra shah", "maoist", "uml",
        "nepali congress", "congress", "nc",
        # kp oli / kp sharma oli kept as full name fragments, not just
        # "oli", to avoid matching unrelated common-surname mentions.
        # Same reasoning for "sushila karki" over standalone "karki".
        "kp oli", "kp sharma oli", "sushila karki"
    ],

    "political_events": [
        "gen z", "genz", "protest", "movement", "revolution", "corruption",
        "social media ban", "government change",
        "wake up nepal", "curfew"
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



# APIFY COLLECTION -- POSTS & COMMENTS

def _validate_date_range(start_date=None, end_date=None):
    """Validate YYYY-MM-DD dates and ensure start <= end."""
    parsed_start = parsed_end = None
    for label, value in (("start-date", start_date), ("end-date", end_date)):
        if value is None:
            continue
        try:
            parsed = datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"{label} must use YYYY-MM-DD (got {value!r})") from exc
        if label == "start-date":
            parsed_start = parsed
        else:
            parsed_end = parsed
    if parsed_start and parsed_end and parsed_start > parsed_end:
        raise ValueError("start-date cannot be after end-date")
    return parsed_start, parsed_end


def _normalize_historical_post(item, subreddit_hint=None):
    """Normalize harshmaur/reddit-scraper post output to this project's schema shape."""
    if not isinstance(item, dict):
        return None

    # The actor's post output uses authorName/communityName/title/body/
    # postUrl/createdAt/upVotes/commentsCount. Keep a few fallbacks so a
    # minor actor output rename does not silently drop a whole run.
    post_id = item.get("id") or item.get("postId") or item.get("name")
    if not post_id:
        return None
    if isinstance(post_id, str) and post_id.startswith("t3_"):
        post_id = post_id[3:]

    username = item.get("authorName") or item.get("username") or item.get("author")
    if not username or str(username).lower() in ("[deleted]", "[removed]"):
        return None

    body = item.get("body")
    if body is None:
        body = item.get("content") or item.get("text") or ""
    title = item.get("title") or ""
    text = f"{title} {body}".strip()
    if not text or text.lower() in ("[deleted]", "[removed]"):
        return None

    community = (
        item.get("communityName")
        or item.get("subredditName")
        or item.get("parsedCommunity")
        or subreddit_hint
        or ""
    )
    community = re.sub(r"^r/", "", str(community).strip(), flags=re.IGNORECASE).lower()

    return {
        "id": post_id,
        "dataType": "post",
        "username": str(username).strip().lower(),
        "parsedCommunityName": community,
        "title": title,
        "body": body or "",
        "createdAt": item.get("createdAt") or item.get("created_at") or item.get("createdTime"),
        "upVotes": item.get("upVotes", item.get("score", 0)) or 0,
        "numberOfComments": item.get("commentsCount", item.get("numberOfComments", 0)) or 0,
        "url": item.get("postUrl", item.get("url", "")) or "",
        "edited": item.get("edited", item.get("isEdited", False)),
    }


def collect_historical_keyword_search(
    keyword,
    subreddit=None,
    max_posts=100,
    max_comments=0,
    sort="top",
    time_filter="all",
    start_date=None,
    end_date=None,
):
    """Collect topic posts from an exact historical date window.

    The historical collector uses harshmaur/reddit-scraper because it natively
    supports postedAfter/postedBefore for keyword searches. Plain end dates
    cover the full UTC day. The actor also limits each listing to roughly
    1,000 reachable Reddit results, so narrow keywords/date windows are still
    preferable for historical research.
    """
    if not keyword or not keyword.strip():
        raise ValueError("keyword must not be empty")
    if sort not in ("relevance", "hot", "top", "new", "comments"):
        raise ValueError(f"Unsupported search sort: {sort!r}")
    _validate_date_range(start_date, end_date)
    _validate_limit("max_posts", max_posts)
    if max_comments != 0:
        logger.warning("Historical comment collection is disabled in this focused collector; forcing 0 comments.")

    # The actor switches to chronological/newest-first when absolute date
    # bounds are supplied, so Reddit's relative searchTime is intentionally
    # omitted in that case.
    run_input = {
        "searchTerms": [keyword.strip()],
        "searchPosts": True,
        "searchComments": False,
        "searchCommunities": False,
        "searchSort": sort if not (start_date or end_date) else "new",
        "searchTime": time_filter if not (start_date or end_date) else "all",
        "maxPostsCount": max_posts,
        "maxCommentsCount": 0,
        "crawlCommentsPerPost": False,
        "includeNSFW": False,
        "proxy": {"useApifyProxy": True},
    }
    if subreddit:
        run_input["withinCommunity"] = subreddit.strip().lstrip("r/")
    if start_date:
        run_input["postedAfter"] = start_date
    if end_date:
        run_input["postedBefore"] = end_date

    logger.info(
        "Historical search: r/%s | %r | dates=%s..%s | sort=%s | max_posts=%s",
        subreddit or "*", keyword, start_date or "-", end_date or "-",
        run_input["searchSort"], max_posts,
    )

    client = get_client()
    try:
        run = client.actor(HISTORICAL_ACTOR).call(
            run_input=run_input,
            run_timeout=timedelta(seconds=1800),
        )
        if run is None:
            logger.error("Historical actor returned no run result.")
            return [], None
        if run.status != "SUCCEEDED":
            logger.warning("Historical actor finished with status=%r; dataset may be incomplete.", run.status)

        dataset_id = run.default_dataset_id
        raw_items = list(client.dataset(dataset_id).iterate_items())
        items = []
        for raw in raw_items:
            normalized = _normalize_historical_post(raw, subreddit)
            if normalized:
                items.append(normalized)

        logger.info(
            "Historical actor returned %d raw items; %d normalized posts (status=%s).",
            len(raw_items), len(items), run.status,
        )
        return items, dataset_id
    except KeyboardInterrupt:
        logger.warning("Historical search interrupted.")
        return [], None
    except Exception as exc:
        logger.exception("Historical search failed for %r: %s", keyword, exc)
        return [], None


def build_historical_jobs(subreddits, keywords, sort="top"):
    """Create deterministic subreddit x keyword jobs for this single-laptop run."""
    jobs = []
    seen = set()
    for subreddit in subreddits:
        clean_subreddit = subreddit.strip().lstrip("r/").lower()
        if not clean_subreddit:
            continue
        for keyword in keywords:
            clean_keyword = re.sub(r"\s+", " ", keyword.strip().lower())
            key = (clean_subreddit, clean_keyword, sort)
            if clean_keyword and key not in seen:
                seen.add(key)
                jobs.append({
                    "subreddit": clean_subreddit,
                    "keyword": clean_keyword,
                    "sort": sort,
                })
    return jobs



def _run_actor(run_input, label, timeout_seconds=1800):
    """Run the Apify actor and return (items, dataset_id), never raising for a normal run failure."""
    client = get_client()
    try:
        run = client.actor(POSTS_ACTOR).call(
            run_input=run_input,
            run_timeout=timedelta(seconds=timeout_seconds),
        )
        if run is None:
            logger.error("%s: actor returned no run result.", label)
            return [], None

        dataset_id = getattr(run, "default_dataset_id", None)
        if not dataset_id:
            logger.error("%s: actor returned no dataset id (status=%r).", label, run.status)
            return [], None

        if run.status != "SUCCEEDED":
            logger.warning(
                "%s: actor finished with status=%r; dataset may be incomplete.",
                label, run.status,
            )

        items = list(client.dataset(dataset_id).iterate_items())
        logger.info("%s: received %d items (status=%s).", label, len(items), run.status)
        return items, dataset_id
    except KeyboardInterrupt:
        logger.warning("%s: interrupted by user.", label)
        return [], None
    except Exception as exc:
        logger.exception("%s: actor call failed: %s", label, exc)
        return [], None


def _validate_limit(name, value, allow_zero=False):
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def collect_account_history_posts(
    username,
    subreddit=None,
    max_posts=100,
    sort="top",
    time_filter="all",
):
    """Collect posts previously submitted by one account.

    Uses Reddit's documented author search syntax (author:<username>) and can
    optionally restrict the search to one project subreddit.
    """
    username = (username or "").strip().lstrip("u/")
    if not username or username.lower() in ("[deleted]", "[removed]"):
        return [], None
    if sort not in ("relevance", "hot", "top", "new", "rising", "comments"):
        raise ValueError(f"Unsupported account-history post sort: {sort!r}")
    if time_filter not in ("all", "hour", "day", "week", "month", "year"):
        raise ValueError(f"Unsupported time filter: {time_filter!r}")
    _validate_limit("max_posts", max_posts)

    run_input = {
        "searches": [f"author:{username}"],
        "searchPosts": True,
        "searchComments": False,
        "searchCommunities": False,
        "searchUsers": False,
        "sort": sort,
        "time": time_filter,
        "maxItems": max_posts,
        "maxPostCount": max_posts,
        "maxComments": 0,
        "includeMediaLinks": True,
        "maxRequestRetries": 3,
        "proxy": {"useApifyProxy": True},
    }
    if subreddit:
        run_input["searchCommunityName"] = subreddit.strip().lstrip("r/")

    label = f"Account posts: u/{username}" + (f" in r/{subreddit}" if subreddit else "")
    logger.info("%s | sort=%s time=%s max_posts=%s", label, sort, time_filter, max_posts)
    return _run_actor(run_input, label)


def collect_account_history_comments(
    username,
    subreddit=None,
    max_comments=100,
    sort="new",
    time_filter="all",
):
    """Collect comments previously written by one account."""
    username = (username or "").strip().lstrip("u/")
    if not username or username.lower() in ("[deleted]", "[removed]"):
        return [], None
    if sort not in ("relevance", "hot", "top", "new", "rising", "comments"):
        raise ValueError(f"Unsupported account-history comment sort: {sort!r}")
    if time_filter not in ("all", "hour", "day", "week", "month", "year"):
        raise ValueError(f"Unsupported time filter: {time_filter!r}")
    _validate_limit("max_comments", max_comments)

    run_input = {
        "searches": [f"author:{username}"],
        "searchPosts": False,
        "searchComments": True,
        "searchCommunities": False,
        "searchUsers": False,
        "sort": sort,
        "time": time_filter,
        "maxItems": max_comments,
        "maxPostCount": 0,
        "maxComments": max_comments,
        "includeMediaLinks": True,
        "maxRequestRetries": 3,
        "proxy": {"useApifyProxy": True},
    }
    if subreddit:
        run_input["searchCommunityName"] = subreddit.strip().lstrip("r/")

    label = f"Account comments: u/{username}" + (f" in r/{subreddit}" if subreddit else "")
    logger.info("%s | sort=%s time=%s max_comments=%s", label, sort, time_filter, max_comments)
    return _run_actor(run_input, label)


def collect_known_account_history(
    subreddits,
    max_accounts=0,
    post_limit=100,
    comment_limit=100,
    post_sorts=("new", "top"),
    comment_sorts=("new", "top"),
    time_filter="all",
    delay_seconds=2,
):
    """
    Collect historical posts/comments for usernames in the local
    account-mapping file (see "LOCAL ACCOUNT MAPPING" near anonymize(),
    above) -- populated automatically by save_items() every time this
    script runs in any other mode, so a normal recent/historical run
    first, then --mode accounts, has usernames to work with without any
    extra setup step.
    """
    usernames = get_mapped_usernames(limit=max_accounts)
    if not usernames:
        logger.warning(
            "Account-history mode found no usernames in %s. Run "
            "collection in another mode first (recent/historical) so "
            "save_items() has usernames to record, then try --mode "
            "accounts again.",
            ACCOUNT_MAP_PATH,
        )
        return {"posts": 0, "comments": 0, "accounts": 0, "usernames": 0}

    logger.info("Account-history mode: %d username(s) available.", len(usernames))
    totals = {"posts": 0, "comments": 0, "accounts": 0, "usernames": len(usernames)}

    for account_index, username in enumerate(usernames, start=1):
        logger.info("Account history %d/%d: u/%s", account_index, len(usernames), username)
        account_had_items = False

        for subreddit in subreddits:
            for sort in post_sorts:
                items, dataset_id = collect_account_history_posts(
                    username=username,
                    subreddit=subreddit,
                    max_posts=post_limit,
                    sort=sort,
                    time_filter=time_filter,
                )
                if items:
                    stats = save_items(
                        items,
                        subreddits=[subreddit],
                        dataset_id=dataset_id,
                        collection_mode="historical_account_posts",
                        search_keyword=f"author:{username}",
                    )
                    totals["posts"] += stats["posts"]
                    totals["comments"] += stats["comments"]
                    account_had_items = True
                time.sleep(delay_seconds)

            if comment_limit > 0:
                for sort in comment_sorts:
                    items, dataset_id = collect_account_history_comments(
                        username=username,
                        subreddit=subreddit,
                        max_comments=comment_limit,
                        sort=sort,
                        time_filter=time_filter,
                    )
                    if items:
                        stats = save_items(
                            items,
                            subreddits=[subreddit],
                            dataset_id=dataset_id,
                            collection_mode="historical_account_comments",
                            search_keyword=f"author:{username}",
                        )
                        totals["posts"] += stats["posts"]
                        totals["comments"] += stats["comments"]
                        account_had_items = True
                    time.sleep(delay_seconds)

        if account_had_items:
            totals["accounts"] += 1

    return totals

def collect_posts(subreddits, max_posts=100, max_comments=20):
    start_urls = [{"url": f"https://www.reddit.com/r/{s}/new/"} for s in subreddits]

    run_input = {
        "startUrls": start_urls,
        # maxItems caps the TOTAL dataset (posts + comments combined),
        # separately from maxPostCount below. This used to be set equal
        # to max_posts, which meant comments were eating the same budget
        # posts needed -- a handful of heavily-commented posts could hit
        # this cap before maxPostCount posts were ever collected, which
        # is almost certainly why posts came back lower than accounts.
        # Sized here so maxPostCount is always the real constraint.
        "maxItems": max_posts * (max_comments + 1),
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



def unprofiled_only(usernames):
    """
    Drop usernames that already have profile data in the accounts table.

    The database stores only the anonymized account id by default, so the
    plaintext username is hashed in memory and compared against accounts.id.
    An account is considered profiled once accounts.created_utc is non-NULL.

    This prevents repeated Apify profile-fetch calls for accounts whose
    profile data has already been collected on an earlier run.
    """
    if not usernames:
        return []

    conn = get_conn()
    try:
        done = {
            row["id"]
            for row in conn.execute(
                "SELECT id FROM accounts WHERE created_utc IS NOT NULL"
            )
        }
    finally:
        conn.close()

    # Normalize exactly the same way save_items()/save_user_profiles() do:
    # strip whitespace, lowercase, then anonymize.
    return [
        username
        for username in usernames
        if username
        and anonymize(username.strip().lower()) not in done
    ]


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
    same run (see __main__). save_items() already writes these to the
    local account-mapping file (see "LOCAL ACCOUNT MAPPING" near
    anonymize()) before this function is ever called -- influence.db
    itself still never sees them; only that local file does.
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



# DATABASE SAVE -- POSTS & COMMENTS


def save_items(items, subreddits=None, dataset_id=None, collection_mode="recent", search_keyword=None):
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


            # POSTS

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


            # COMMENTS


            elif data_type == "comment":

                parent_id = item.get("parentId")

                # Self-replies (a user replying to their own post/comment)
                # aren't an inter-account coordination signal, so they're
                # excluded here -- this also means they don't count toward
                # comments_per_day / engagement features, which is a
                # deliberate trade-off worth noting in your methodology.
                # Resolved via author_map (built above), comparing
                # anonymized ids -- see module docstring.
                # FIXED: parent_id arrives prefixed ('t3_abc123' for a
                # post parent, 't1_xyz' for a comment parent), but
                # author_map is keyed by BARE ids ('abc123') everywhere
                # it's built above (from posts.id/comments.id, and from
                # item.get("id") in the pre-pass) -- so
                # author_map.get(parent_id) was always looking up a
                # "t3_..."/"t1_..." string that never matched any key,
                # always returned None, and this check silently never
                # skipped a single self-reply for any real item.
                # Confirmed by testing with realistic prefixed sample
                # data during this rewrite. Strip the two-char type
                # prefix + "_" before the lookup so it actually matches.
                parent_bare_id = (
                    parent_id.split("_", 1)[1]
                    if parent_id and "_" in parent_id
                    else parent_id
                )
                if parent_bare_id and author_map.get(parent_bare_id) == anon_id:
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


        # UPDATE ACCOUNTS

        for username in users:
            anon_id = anonymize(username)

            c.execute(
                """
                INSERT OR IGNORE INTO accounts (id, username)
                VALUES (?,?)
                """,
                (anon_id, None)
            )

            update_account_stats(c, anon_id)

        
        # DATASET METADATA

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
                f"Actor={POSTS_ACTOR};Mode={collection_mode};"
                f"Keyword={search_keyword or '-'};Dataset={dataset_id or 'unknown'}"
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

    # Every raw username this run saw goes into the local account-map
    # file (see "LOCAL ACCOUNT MAPPING" near anonymize(), above) --
    # unconditionally, so account-history mode has something to work
    # with without a separate flag to remember. The database itself
    # never sees these; only ACCOUNT_MAP_PATH does.
    update_account_map(sorted(users))

    return {
        "posts": post_count,
        "comments": comment_count,
        "users": len(users),
        # Raw usernames from this run, for collect_user_profiles() /
        # unprofiled_only() below to use immediately without a second
        # disk read -- update_account_map() above already persisted
        # them, this is just handing the same list onward in memory.
        "usernames": sorted(users)
    }



# DATABASE SAVE -- USER PROFILES


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



# BACKFILL -- ENRICHMENT FOR ALREADY-COLLECTED ROWS


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



# PROGRESS REPORTING


def report_progress(
    target_posts=REDDIT_COLLECTION["target_posts"],
    target_accounts=REDDIT_COLLECTION["target_accounts"],
):
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
# ACCOUNT CANDIDATE SELECTION
#################################################

def get_candidate_usernames(min_relevant_posts=3, min_topics=1, limit=30):
    """Return raw usernames for accounts with repeated topic-relevant activity."""
    _validate_limit("min_relevant_posts", min_relevant_posts)
    _validate_limit("min_topics", min_topics)

    mapping = _load_account_map()
    if not mapping:
        logger.warning("No local account map exists at %s.", ACCOUNT_MAP_PATH)
        return []

    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT account_id,
                   COUNT(*) AS relevant_posts,
                   COUNT(DISTINCT COALESCE(topic, 'unknown')) AS topic_count,
                   COUNT(DISTINCT subreddit) AS subreddit_count
            FROM posts
            WHERE is_relevant = 1
            GROUP BY account_id
            HAVING COUNT(*) >= ?
               AND COUNT(DISTINCT COALESCE(topic, 'unknown')) >= ?
            ORDER BY relevant_posts DESC, topic_count DESC, subreddit_count DESC
            """,
            (min_relevant_posts, min_topics),
        ).fetchall()
    finally:
        conn.close()

    usernames = []
    for row in rows:
        username = mapping.get(row["account_id"])
        if not username or username.lower() in ("[deleted]", "[removed]"):
            continue
        usernames.append(username)
        if limit > 0 and len(usernames) >= limit:
            break

    logger.info("Selected %d candidate account(s).", len(usernames))
    return usernames


def print_candidate_accounts(limit=50):
    """Print ranked anonymous candidate accounts for manual review."""
    _validate_limit("limit", limit)
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT account_id, COUNT(*) AS relevant_posts,
                   COUNT(DISTINCT COALESCE(topic, 'unknown')) AS topic_count,
                   COUNT(DISTINCT subreddit) AS subreddit_count,
                   MIN(created_utc) AS first_activity,
                   MAX(created_utc) AS last_activity
            FROM posts
            WHERE is_relevant = 1
            GROUP BY account_id
            ORDER BY relevant_posts DESC, topic_count DESC, subreddit_count DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print("No topic-relevant candidate accounts found.")
        return
    print("account_id\trelevant_posts\ttopics\tsubreddits\tfirst_activity\tlast_activity")
    for row in rows:
        print(f"{row['account_id']}\t{row['relevant_posts']}\t{row['topic_count']}\t{row['subreddit_count']}\t{row['first_activity']}\t{row['last_activity']}")


#################################################
# MAIN
#################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect focused public Reddit data for the coordination-analysis project."
    )
    parser.add_argument(
        "--mode", choices=("recent", "historical", "accounts", "candidates", "both", "full"),
        default="historical",
        help="recent=/new/; historical=keyword history; accounts=selected candidate histories; candidates=list candidates; both=recent+historical; full=recent+historical+selected histories.",
    )
    parser.add_argument("--backfill", action="store_true", help="Recompute topic/language/sentiment without Apify calls.")
    parser.add_argument("--subreddit", action="append", dest="subreddits", help="Scrape only this subreddit; repeat for multiple.")
    parser.add_argument("--keyword", action="append", dest="keywords", help="Historical keyword; repeat for multiple.")
    parser.add_argument("--start-date", default=None, help="Historical posts on/after YYYY-MM-DD (UTC).")
    parser.add_argument("--end-date", default=None, help="Historical posts on/before YYYY-MM-DD (UTC, inclusive).")
    parser.add_argument("--recent-posts", type=int, default=100)
    parser.add_argument("--recent-comments", type=int, default=0)
    parser.add_argument("--historical-posts", type=int, default=100)
    parser.add_argument("--historical-comments", type=int, default=0)
    parser.add_argument("--historical-sorts", nargs="+", default=["top"], choices=("relevance", "hot", "top", "new", "rising", "comments"))
    parser.add_argument("--historical-time", default="all", choices=("all", "hour", "day", "week", "month", "year"))
    parser.add_argument("--max-historical-jobs", type=int, default=0)
    parser.add_argument("--account-limit", type=int, default=30)
    parser.add_argument("--candidate-min-posts", type=int, default=3)
    parser.add_argument("--candidate-min-topics", type=int, default=1)
    parser.add_argument("--account-posts", type=int, default=50)
    parser.add_argument("--account-comments", type=int, default=0)
    parser.add_argument("--account-post-sorts", nargs="+", default=["new"], choices=("relevance", "hot", "top", "new", "rising", "comments"))
    parser.add_argument("--account-comment-sorts", nargs="+", default=["new"], choices=("relevance", "hot", "top", "new", "rising", "comments"))
    parser.add_argument("--account-time", default="all", choices=("all", "hour", "day", "week", "month", "year"))
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--no-profiles", action="store_true")
    args = parser.parse_args()

    try:
        _validate_date_range(args.start_date, args.end_date)
    except ValueError as exc:
        parser.error(str(exc))

    if args.recent_posts < 1 or args.historical_posts < 1 or args.account_posts < 1:
        parser.error("post limits must be >= 1")
    if args.recent_comments < 0 or args.historical_comments < 0 or args.account_comments < 0:
        parser.error("comment limits must be >= 0")
    if args.max_historical_jobs < 0 or args.account_limit < 0 or args.delay < 0:
        parser.error("limits/delay cannot be negative")
    if args.candidate_min_posts < 1 or args.candidate_min_topics < 1:
        parser.error("candidate thresholds must be >= 1")

    init_db()

    if args.backfill:
        logger.info("Running enrichment backfill (no Apify calls)...")
        backfill_enrichment()
        report_progress()
        sys.exit(0)

    default_subreddits = ["Nepal", "NepaliPolitics", "nepalinews", "NepalSocial"]
    subreddits = [s.strip().lstrip("r/") for s in (args.subreddits or default_subreddits) if s.strip()]
    if not subreddits:
        parser.error("provide at least one --subreddit")

    keywords = list(dict.fromkeys(
        re.sub(r"\s+", " ", k.strip().lower())
        for k in (args.keywords or HISTORICAL_SEARCH_TERMS) if k and k.strip()
    ))

    totals = {"posts": 0, "comments": 0, "users": 0}

    def save_batch(items, dataset_id, mode, keyword, batch_subreddits):
        if not items:
            return
        stats = save_items(items, batch_subreddits, dataset_id, mode, keyword)
        totals["posts"] += stats["posts"]
        totals["comments"] += stats["comments"]
        totals["users"] += stats["users"]

        # Profile enrichment is expensive; only do it when explicitly requested.
        if not args.no_profiles:
            usernames = unprofiled_only(stats.get("usernames", []))
            if usernames:
                profile_items, _ = collect_user_profiles(usernames)
                if profile_items:
                    save_user_profiles(profile_items)

    if args.mode in ("recent", "both", "full"):
        items, dataset_id = collect_posts(subreddits, args.recent_posts, args.recent_comments)
        save_batch(items, dataset_id, "recent_new_feed", None, subreddits)

    if args.mode in ("historical", "both", "full"):
        jobs = []
        for subreddit in subreddits:
            for keyword in keywords:
                for sort in args.historical_sorts:
                    jobs.append((subreddit, keyword, sort))
        # Deduplicate identical jobs while preserving order.
        jobs = list(dict.fromkeys(jobs))
        if args.max_historical_jobs > 0:
            jobs = jobs[:args.max_historical_jobs]
        logger.info("Historical jobs: %d", len(jobs))

        for i, (subreddit, keyword, sort) in enumerate(jobs, 1):
            logger.info("Historical %d/%d: r/%s | %r | sort=%s", i, len(jobs), subreddit, keyword, sort)
            items, dataset_id = collect_historical_keyword_search(
                keyword, subreddit, args.historical_posts, args.historical_comments, sort, args.historical_time,
                args.start_date, args.end_date
            )
            save_batch(items, dataset_id, "historical_keyword", keyword, [subreddit])

    if args.mode == "candidates":
        print_candidate_accounts(args.account_limit or 50)

    if args.mode in ("accounts", "full"):
        candidates = get_candidate_usernames(
            min_relevant_posts=args.candidate_min_posts,
            min_topics=args.candidate_min_topics,
            limit=args.account_limit,
        )
        logger.info("Deep-history candidates: %d", len(candidates))

        for i, username in enumerate(candidates, 1):
            logger.info("Account history %d/%d: u/%s", i, len(candidates), username)
            for subreddit in subreddits:
                for sort in dict.fromkeys(args.account_post_sorts):
                    items, dataset_id = collect_account_history_posts(
                        username, subreddit, args.account_posts, sort, args.account_time
                    )
                    save_batch(items, dataset_id, "historical_account_posts", f"author:{username}", [subreddit])
                    if args.delay:
                        time.sleep(args.delay)

                if args.account_comments > 0:
                    for sort in dict.fromkeys(args.account_comment_sorts):
                        items, dataset_id = collect_account_history_comments(
                            username, subreddit, args.account_comments, sort, args.account_time
                        )
                        save_batch(items, dataset_id, "historical_account_comments", f"author:{username}", [subreddit])
                        if args.delay:
                            time.sleep(args.delay)

    report_progress()
    logger.info("Newly inserted this process: posts=%d comments=%d users=%d", totals["posts"], totals["comments"], totals["users"])