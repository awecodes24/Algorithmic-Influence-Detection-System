import pandas as pd
import sqlite3
import os
import uuid
from datetime import datetime
from config import DB_PATH, BENCHMARK

CRESCI_PATH = os.path.join(BENCHMARK, 'cresci-2017')


# benchmark dataset haru
CRESCI_DATASETS = {
    "genuine_accounts.csv": 0,
    "fake_followers.csv": 1,
    "social_spambots_1.csv": 1,
    "social_spambots_2.csv": 1,
    "social_spambots_3.csv": 1,
    "traditional_spambots_1.csv": 1,
    "traditional_spambots_2.csv": 1,
    "traditional_spambots_3.csv": 1,
    "traditional_spambots_4.csv": 1,
}


# database sanga connect garne
def get_connection():
    return sqlite3.connect(DB_PATH)


# int ma convert garne value lai with try catch handling
def safe_int(value, default=0):
    try:
        return int(float(value))

    except (ValueError, TypeError):
        return default


# float ma convert garne value lai with try catch handling


def safe_float(value, default=0.0):
    try:
        return float(value)

    except (ValueError, TypeError):
        return default


# str ma convert garne value lai with try catch handling


def safe_str(value, default=""):
    try:
        if pd.isna(value):
            return default
    except (ValueError, TypeError):
        return default


def find_folder(base_path, folder_name):
    """
     Cresci-2017 has nested folders like:
    social_spambots_1.csv/social_spambots_1.csv/users.csv
    This function finds the correct inner path.
    """
    # for nested path
    nested = os.path.join(base_path, folder_name)
    if os.path.exists(nested):
        return nested

    # for direct path
    direct = os.path.join(base_path, folder_name)
    if os.path.exists(direct):
        return direct

    return None


def load_users(cursor, folder_path, dataset_name, is_bot):
    """_loads users.csv from one dataset folder into accounts + features tables._

    Args:
        cursor (_type_): _description_
        folder_path (_type_): _description_
        dataset_name (_type_): _description_
        is_bot (bool): _description_
    """

    users_path = os.path.join(folder_path, "users,csv")

    if not os.path.exists(users_path):
        print(f"users.csv not found in {dataset_name}")
        return 0

    try:
        df = pd.read_csv(users_path, low_memory=False)
    except Exception as e:
        print(f"could not read users.csv: {e}")
        return 0

    print(f" users.csv -> {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    loaded = 0
    for _, row in df.iterrows():
        try:
            # account id
            raw_id = row.get("id", row.get("Id", None))
            if pd.isna(raw_id) or raw_id is None:
                account_id = f"cresci_{uuid.uuid4().hex[:8]}"
            else:
                account_id = f"cresci_{int(float(raw_id))}"

            # map colums - haldles variations in naming
            username = safe_str(row["screen_name"])
            created_at = safe_str(row["created_at"])
            followers = safe_int(row["followers_count"])
            following = safe_int(row["friends_count"])
            total_posts = safe_int(row["statuses_count"])
            favourites = safe_int(row["favourites_count"])
            listed = safe_int(row["listed_count"])
            lang = safe_str(row["lang"], default="unknown")
            is_verified = safe_int(row["verified"])

            # insert into accounts
            cursor.execute(
                """
            INSERT OR IGNORE INTO accounts(
                account_id, platform , username, create_at, follower_count, following_count , total_posts, is_verified, language , collected_at ) values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            )""",
                (
                    account_id,
                    "twitter",
                    username,
                    created_at,
                    followers,
                    following,
                    total_posts,
                    is_verified,
                    lang,
                    datetime.now().isoformat(),
                ),
            )

            # insert label into features
            cursor.execute(
                """
                           INSERT OR IGNORE INTO features (
                               account_idm platform, is_bot, favourites_count, listed_count
                           ) VALUES(?, ?, ?, ?, ?)
                           """,
                (account_id, "twitter", is_bot, favourites, listed),
            )
            loaded += 1

        except Exception:
            continue
        return loaded
    
    

def load_tweets(cursor, folder_path, dataset_name):
    """ Load tweets.csv using exact Cresci-2017 column names.
    Confirmed columns from actual file:
    id, text, user_id, retweet_count, favorite_count,
    num_hashtags, num_urls, num_mentions, created_at

    Args:
        cursor (_type_): _description_
        folder_path (_type_): _description_
        dataset_name (_type_): _description_
    """

    tweets_path = os.path.join(folder_path, 'tweets.csv')
    if not os.path.exists(tweets_path):
        print("tweets.csv not found")
        return 0
    
    try:
        df = pd.read_csv(tweets_path, low_memory=False)
    except Exception as e:
        print(f"Read error: {e}")
        return 0
    print(f"tweets.csv ->{len(df):>6} rows")
    
    loaded = 0
    
    for _, row in df.iterrows():
        try:
            # ── Post ID ──────────────────────────────────────────────────────
            raw_id = row.get('id', None)
            if pd.isna(raw_id) or raw_id is None:
                post_id = f"tweet_{uuid.uuid4().hex[:8]}"
            else:
                post_id = f"tweet_{int(float(raw_id))}"

            # ── Link to account ──────────────────────────────────────────────
            raw_user_id = row.get('user_id', None)
            if pd.isna(raw_user_id) or raw_user_id is None:
                continue
            account_id = f"cresci_{int(float(raw_user_id))}"

            # ── Exact column mapping ─────────────────────────────────────────
            content      = safe_str(row['text'])
            posted_at    = safe_str(row['created_at'])
            likes        = safe_int(row['favorite_count'])
            shares       = safe_int(row['retweet_count'])
            num_hashtags = safe_int(row['num_hashtags'])
            num_urls     = safe_int(row['num_urls'])
            num_mentions = safe_int(row['num_mentions'])

            # Compute basic engagement rate ────────────────────────────────
            engagement = safe_float(likes + shares)

            # Extract hashtags as simple count string ──────────────────────
            hashtags = f'{{"count": {num_hashtags}}}'

            #  Insert into posts ────────────────────────────────────────────
            cursor.execute('''
                INSERT OR IGNORE INTO posts (
                    post_id, account_id, platform, content,
                    topic_label, hashtags, posted_at,
                    likes, shares, comments_count,
                    engagement_rate, collected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                post_id, account_id, 'twitter', content,
                'benchmark', hashtags, posted_at,
                likes, shares, 0,
                engagement, datetime.now().isoformat()
            ))

            loaded += 1

        except Exception:
            continue

    return loaded


def load_all_datasets():
    conn = get_connection()
    cursor = conn.cursor()
    
    total_users = 0
    total_tweets = 0
    
    for folder_name, is_bot in CRESCI_DATASETS.items():
        
        label = "HUMAN" if is_bot == 0 else 'BOT'
        folder_path = find_folder(CRESCI_PATH, folder_name)
           
        print(f"\n {folder_name} [{label}]")
        if folder_path is None:
            print(f"\n Folder not found , skipping")
            continue
        print(f" path: {folder_path}")
        
        users_loaded  = load_users(cursor, folder_path, folder_name, is_bot)
        tweets_loaded = load_tweets(cursor, folder_path, folder_name)

        
        total_users+= users_loaded
        total_tweets+= tweets_loaded
        
        print(f" Users: {users_loaded} | Tweets: {tweets_loaded}")
        conn.commit()
        
    conn.close()
    
    print(f"""
          
    -----------------------------------
    cresci -2017 load Complete
    Total accounts : {total_users: >8}
    Total tweets : {total_tweets: >8}
    ----------------------------------
          
          
          
          """)
        
    
def verify_load():
    conn = get_connection()
    cursor = conn.cursor()
    
    print("\n Database row counts:")
    for table in ['accounts', 'posts','features']:
        cursor.execute(f"SELECT COUNT (*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f" {table:<20} {count:>20} rows")
    print("\n Bot vs Human breakdown:")
    cursor.execute("""
            SELECT is_bot , COUNT(*)
            FROM features
            GROUP BY is_bot
            
                   
                   
                   """)   
    for row in cursor.fetchall():
        label = "HUMAN" if row[0] == 0 else 'Bot'
        print(f" {label: <12} {row[0]:<20}"
              f"followers={row[1]:<8}"
              f"posts={row[2]:<8}"
              f"lang={row[3]}")
        
    conn.close()
    
if __name__ == "__main__":
    print("Starting Cresci-2017 data load...\n")
    load_all_datasets()
    verify_load()
    
    

