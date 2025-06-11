import json
import csv
from datetime import datetime

def convert_unix_to_datetime(ts):
    return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') if ts else ""

def convert_reviews_json_to_csv(json_file, csv_file):
    with open(json_file, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    with open(csv_file, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        # En-têtes CSV
        writer.writerow([
            "review_id",
            "author_steamid",
            "review",
            "voted_up",
            "votes_up",
            "votes_funny",
            "weighted_vote_score",
            "comment_count",
            "created_at",
            "updated_at",
            "language"
        ])

        for review in reviews:
            created = convert_unix_to_datetime(review.get("timestamp_created"))
            updated = convert_unix_to_datetime(review.get("timestamp_updated"))
            writer.writerow([
                review.get("recommendationid"),
                review.get("author", {}).get("steamid"),
                review.get("review"),
                review.get("voted_up"),
                review.get("votes_up"),
                review.get("votes_funny"),
                review.get("weighted_vote_score"),
                review.get("comment_count"),
                created,
                updated,
                review.get("language")
            ])

    print(f"✅ Fichier CSV avec dates lisibles créé : {csv_file}")

# Exemple d'utilisation
convert_reviews_json_to_csv("steam_reviews.json", "steam_reviews.csv")
