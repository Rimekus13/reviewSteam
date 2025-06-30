import pandas as pd
import streamlit as st
import pymongo
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# --- Ajout de style CSS personnalisé avec couleurs dominantes claires ---
from PIL import Image
import base64
import io
from collections import Counter

@st.cache_data(show_spinner=False)
def get_light_dominant_colors(app_id, num_colors=5):
    img_url = f"https://cdn.akamai.steamstatic.com/steam/apps/{app_id}/header.jpg"
    try:
        response = requests.get(img_url)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image = image.resize((50, 50))
        pixels = list(image.getdata())
        # Filtrer uniquement les couleurs claires
        light_pixels = [p for p in pixels if sum(p) > 400]  # sum(R,G,B) > 400 = couleurs claires
        common_colors = Counter(light_pixels).most_common(num_colors)
        return [f"rgb{color[0]}" for color in common_colors]
    except:
        return ["#f0f0f0", "#e0e0e0", "#ffffff"]

# --- Fonction pour récupérer le nom d'un jeu à partir de son app_id ---
@st.cache_data(show_spinner=False)
def get_game_name(app_id):
    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
    try:
        res = requests.get(url, timeout=5)
        data = res.json()
        return data[str(app_id)]["data"]["name"]
    except:
        return f"App {app_id}"

# Connexion à MongoDB Docker
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["steamdb"]

# 🔁 Récupération des collections et association nom/app_id
collections = db.list_collection_names()
collection_ids = [col.replace("reviews_", "") for col in collections if col.startswith("reviews_")]
mapping = {app_id: get_game_name(app_id) for app_id in collection_ids}

# 🎮 Sélection du jeu via son nom
selected_app_id = st.sidebar.selectbox(
    "Choisir un jeu Steam",
    options=sorted(mapping),
    format_func=lambda app_id: mapping[app_id]
)
selected_collection = f"reviews_{selected_app_id}"

# 📸 Image du jeu + couleurs dominantes
img_url = f"https://cdn.akamai.steamstatic.com/steam/apps/{selected_app_id}/header.jpg"
st.title(f"Avis Steam – {mapping[selected_app_id]}")
st.image(img_url, use_column_width=True)

# 🎨 Appliquer un fond dégradé basé sur des couleurs dominantes claires
colors = get_light_dominant_colors(selected_app_id)
if len(colors) >= 3:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(135deg, {colors[0]}, {colors[1]}, {colors[2]});
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Chargement des données depuis Mongo
reviews = list(db[selected_collection].find())
df = pd.DataFrame(reviews)

# Nettoyage du texte
@st.cache_data
def clean_text(text):
    if not isinstance(text, str):
        return ""
    import re
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["cleaned_review"] = df["review"].apply(clean_text)

# Affichage de base
st.write(f"Total d'avis : {len(df)}")

# 🔎 Filtrage interactif
langues = df["language"].dropna().unique().tolist()
langue_filtrée = st.sidebar.multiselect("Langue(s)", langues, default=langues)
positif = st.sidebar.checkbox("Avis positifs seulement", value=False)

filtres = (df["language"].isin(langue_filtrée))
if positif:
    filtres &= (df["voted_up"] == True)

df_filtered = df[filtres]
st.write(f"Avis filtrés : {len(df_filtered)}")

# WordCloud
if st.checkbox("Afficher le WordCloud"):
    text = " ".join(df_filtered["cleaned_review"].dropna())
    if text:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.warning("Aucun texte disponible pour générer le nuage de mots.")

# Histogramme des votes
if "voted_up" in df_filtered:
    st.subheader("Répartition des votes")
    sns.set_style("whitegrid")
    fig, ax = plt.subplots()
    sns.countplot(x="voted_up", data=df_filtered, palette="Set2", ax=ax)
    ax.set_xticklabels(["Négatif", "Positif"])
    st.pyplot(fig)

# Afficher quelques avis
st.subheader("Aperçu des avis")
st.dataframe(df_filtered[["author", "review", "voted_up", "language"]].head(20))
