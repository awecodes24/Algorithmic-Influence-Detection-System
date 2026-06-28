import pandas as pd
import sqlite3 
import os
import uuid
from datetime import datetime
from config import DB_PATH, BENCHMARK

# benchmark dataset haru
CRESCI_DATASETS = {
    'genuine_accounts.csv':       0,
    'fake_followers.csv':         1,
    'social_spambots_1.csv':      1,
    'social_spambots_2.csv':      1,
    'social_spambots_3.csv':      1,
    'traditional_spambots_1.csv': 1,
    'traditional_spambots_2.csv': 1,
    'traditional_spambots_3.csv': 1,
    'traditional_spambots_4.csv': 1,
}

# database sanga connect garne
def get_connection():
    return sqlite3.connect(DB_PATH)


# int ma convert garne value lai with try catch handling
def safe_int(value , default = 0):
    try:
        return int(float(value))
    
    except (ValueError, TypeError):
        return default

# float ma convert garne value lai with try catch handling

def safe_float(value , default = 0.0):
    try:
        return float(value)
    
    except(ValueError, TypeError):
        return default

# str ma convert garne value lai with try catch handling

def safe_str(value, default = ''):
    try:
        if pd.isna(value):
            return default
    except(ValueError, TypeError):
        return default
    

def load_users(cursor , folder_path, dataset_name , is_bot):
    """_loads users.csv from one dataset folder into accounts + features tables._

    Args:
        cursor (_type_): _description_
        folder_path (_type_): _description_
        dataset_name (_type_): _description_
        is_bot (bool): _description_
    """
    
    users_path = os.path.join(folder_path, 'users,csv')
    
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
            #account id
            raw_id = row.get('id', row.get('Id', None))
            if pd.isna(raw_id) or raw_id is None:
                account_id = f"cresci_{uuid.uuid4().hex[:8]}"
            else:
                account_id = f"cresci_{int(float(raw_id))}"
                
                