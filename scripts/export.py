import requests
import time
import json

def fetch_all_reviews(app_id, output_file="steam_reviews.json"):
    base_url = "https://store.steampowered.com/appreviews/{}"
    params = {
        "json": 1,
        "filter": "recent",
        "weighted_vote_score": "all",
        "language": "all",
        "review_type": "all",
        "purchase_type": "all",
        "num_per_page": 100,
        "cursor": "*"
    }

    all_reviews = []
    total_fetched = 0

    print(f"Fetching reviews for app_id={app_id}...")

    while True:
        response = requests.get(base_url.format(app_id), params=params)
        data = response.json()

        if "reviews" not in data:
            print("No reviews found or error in response.")
            break

        reviews = data["reviews"]
        all_reviews.extend(reviews)
        total_fetched += len(reviews)
        print(f"Fetched {len(reviews)} reviews. Total so far: {total_fetched}")

        if not data.get("success") or len(reviews) == 0:
            break

        # Préparer la requête suivante
        params["cursor"] = data["cursor"]
        time.sleep(0.03)  # Respecter le délai

    # Sauvegarde
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_reviews, f, ensure_ascii=False, indent=2)

    print(f"✅ Done! Total reviews fetched: {total_fetched}. Saved to {output_file}")

# Exemple : Elden Ring (app_id = 1245620)
fetch_all_reviews(app_id=1172710)

