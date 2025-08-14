import requests
import time
import json
import os
import gzip
from pymongo import MongoClient

# Répertoire de sortie des fichiers compressés
OUTPUT_DIR = "steam_reviews"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Connexion MongoDB (service docker-compose = mongo)
client = MongoClient("mongodb://localhost:27017/")
db = client["steamdb"]

def fetch_reviews_for_game(app_id, delay=2):
    url = f"https://store.steampowered.com/appreviews/{app_id}"
    params = {
        "json": 1,
        "filter": "updated",
        "review": "all",
        "language": "all",
        "day_range": "365",
        "review_type": "all",
        "purchase_type": "all",
        "num_per_page": 100,
        "cursor": "*"
    }

    all_reviews = []
    print(f"\n📦 Dumping reviews for App ID: {app_id}")

    while True:
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
        except Exception as e:
            print(f"❌ Error for app_id {app_id}: {e}")
            break

        reviews = data.get("reviews", [])
        if not reviews:
            break

        all_reviews.extend(reviews)
        print(f"  → {len(reviews)} reviews fetched (total: {len(all_reviews)})")

        cursor = data.get("cursor")
        if not cursor or len(reviews) < 100:
            break

        params["cursor"] = cursor
        time.sleep(delay)

    # Sauvegarde gzip
    if all_reviews:
        output_path = os.path.join(OUTPUT_DIR, f"{app_id}.json.gz")
        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            json.dump(all_reviews, f, ensure_ascii=False)
        print(f"✅ Saved compressed: {output_path}")
    else:
        print(f"⚠️ No reviews for app_id {app_id}")

def insert_reviews_into_mongo(app_id):
    collection = db[f"reviews_{app_id}"]
    input_path = os.path.join(OUTPUT_DIR, f"{app_id}.json.gz")

    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return

    with gzip.open(input_path, "rt", encoding="utf-8") as f:
        reviews = json.load(f)
        if reviews:
            collection.insert_many(reviews)
            print(f"✅ Inserted {len(reviews)} reviews into MongoDB (collection: reviews_{app_id})")
        else:
            print(f"⚠️ No reviews to insert for app_id {app_id}")

# Liste des jeux à traiter
popular_app_ids = [
    1250410
]

# Exécution complète
for app_id in popular_app_ids:
    fetch_reviews_for_game(app_id)
    insert_reviews_into_mongo(app_id)
