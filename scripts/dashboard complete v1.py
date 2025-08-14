# app_dashboard_steam.py
# Dashboard Streamlit "lisible + luxe" avec explications intégrées
# + Bouton "Rafraîchir les données" pour recalcul global (option purge cache)

import streamlit as st
import pandas as pd
import numpy as np
import pymongo
import matplotlib.pyplot as plt
import seaborn as sns
import json, io, requests
from collections import Counter
from datetime import datetime
from PIL import Image
from wordcloud import WordCloud
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# ============ CONFIG & STYLE ============
st.set_page_config(page_title="Avis Steam – Insights", layout="wide")

PRIMARY_BG = "#0f1115"          # fond sobre
CARD_BG    = "rgba(255,255,255,0.06)"
TEXT       = "#E8EAED"
ACCENT     = "#8AB4F8"
GOOD       = "#34A853"
WARN       = "#F9AB00"
BAD        = "#EA4335"
BORDER     = "#2b2f36"

# CSS : fond, cards, titres
st.markdown(f"""
<style>
:root {{
  --text:{TEXT}; --accent:{ACCENT}; --border:{BORDER};
  --good:{GOOD}; --warn:{WARN}; --bad:{BAD};
}}
html, body, .stApp {{
  background: radial-gradient(1200px 600px at 10% 10%, #171b22, #0b0d11 60%, #090b0e);
  color: var(--text);
}}
.block {{
  background:{CARD_BG};
  border:1px solid var(--border);
  border-radius:16px; padding:18px 18px 10px 18px;
  backdrop-filter: blur(6px);
}}
.metric {{
  background:{CARD_BG};
  border:1px solid var(--border);
  border-radius:14px; padding:16px;
  text-align:center;
}}
.metric .value {{ font-size: 28px; font-weight:700; }}
.metric .label {{ font-size: 13px; opacity:0.8; }}
.caption {{
  font-size: 13px; opacity:.85; margin-top:6px;
}}
h1, h2, h3 {{ color: var(--text); }}
hr {{ border: none; border-top:1px solid var(--border); margin: 8px 0 16px 0; }}
</style>
""", unsafe_allow_html=True)

sns.set_style("whitegrid")

# ============ UTILS ============
@st.cache_resource(show_spinner=False)
def get_db():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    return client["steamdb"]

@st.cache_data(show_spinner=False)
def get_game_name(app_id):
    try:
        url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
        res = requests.get(url, timeout=5)
        data = res.json()
        return data[str(app_id)]["data"]["name"]
    except:
        return f"App {app_id}"

@st.cache_data(show_spinner=False)
def get_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()

@st.cache_data(show_spinner=False)
def clean_text_series(series: pd.Series) -> pd.Series:
    import re
    s = series.fillna("").astype(str).str.lower()
    s = s.str.replace(r"http\S+", " ", regex=True)
    s = s.str.replace(r"[^a-zA-ZÀ-ÖØ-öø-ÿ0-9\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

@st.cache_data(show_spinner=False)
def get_colors_from_header(app_id, num_colors=3):
    img_url = f"https://cdn.akamai.steamstatic.com/steam/apps/{app_id}/header.jpg"
    try:
        r = requests.get(img_url, timeout=5)
        im = Image.open(io.BytesIO(r.content)).convert("RGB").resize((50,50))
        px = list(im.getdata())
        light = [p for p in px if sum(p) > 350] or px
        common = Counter(light).most_common(num_colors)
        return [f"rgb{c[0]}" for c in common]
    except:
        return [ACCENT, GOOD, BAD]

def apply_bg_gradient(colors):
    if len(colors) >= 3:
        st.markdown(f"""
        <style>
        .stApp {{
            background: radial-gradient(1100px 500px at 5% 10%, {colors[0]}22, transparent 50%),
                        radial-gradient(900px 450px at 90% 20%, {colors[1]}22, transparent 50%),
                        radial-gradient(800px 400px at 50% 90%, {colors[2]}22, transparent 50%),
                        linear-gradient(180deg, #0d0f14, #090b0e);
        }}
        </style>
        """, unsafe_allow_html=True)

# ============ DATA ============
db = get_db()
cols = [c for c in db.list_collection_names() if c.startswith("reviews_")]
app_ids = [c.replace("reviews_", "") for c in cols]
if not app_ids:
    st.error("Aucune collection 'reviews_<app_id>' dans MongoDB.")
    st.stop()

names = {app_id: get_game_name(app_id) for app_id in app_ids}

left, right = st.columns([2,1])
with left:
    selected_app = st.selectbox("🎮 Choisir un jeu", options=sorted(app_ids), format_func=lambda a: names[a])
with right:
    theme_mode = st.radio("🎨 Mode", ["Sombre", "Clair"], horizontal=True, index=0)
    if theme_mode == "Clair":
        st.write("")  # placeholder simple

colors = get_colors_from_header(selected_app)
apply_bg_gradient(colors)

img_url = f"https://cdn.akamai.steamstatic.com/steam/apps/{selected_app}/header.jpg"
st.markdown(f"## {names[selected_app]}")
st.image(img_url, use_column_width=True)

@st.cache_data(show_spinner=False)
def load_df(collection):
    docs = list(db[collection].find())
    if not docs: return pd.DataFrame()
    df = pd.DataFrame(docs)

    # standardisation minimale
    if "review" in df.columns:
        df["review_text"] = df["review"]
    elif "review_text" not in df.columns:
        df["review_text"] = ""

    if "language" not in df.columns:
        df["language"] = "unknown"

    # voted_up par défaut NaN
    if "voted_up" not in df.columns:
        df["voted_up"] = np.nan

    # playtime
    if "author" in df.columns:
        try:
            pt = df["author"].apply(lambda x: x.get("playtime_forever") if isinstance(x, dict) else None)
            df["playtime_hours"] = (pt.fillna(0)/60).round(2)
        except: pass

    # dates
    if "timestamp_created" in df.columns:
        df["review_date"] = pd.to_datetime(df["timestamp_created"], unit="s", errors="coerce")
    elif "created" in df.columns:
        df["review_date"] = pd.to_datetime(df["created"], errors="coerce")
    else:
        df["review_date"] = pd.to_datetime(df.get("review_date"), errors="coerce")

    df["cleaned_review"] = clean_text_series(df["review_text"])
    return df

df = load_df(f"reviews_{selected_app}")
if df.empty:
    st.warning("Aucune donnée disponible pour ce jeu.")
    st.stop()

# ============ FILTRES ============
langs = sorted(df["language"].dropna().unique().tolist())
c1, c2, c3, c4 = st.columns([1.2,1,1,1])
with c1:
    chosen_langs = st.multiselect("🌐 Langues", options=langs, default=langs,
                                  help="Filtrez les avis par langue.")
with c2:
    only_positive = st.checkbox("👍 Avis positifs uniquement", value=False,
                                help="Filtre sur voted_up == True (si disponible).")
with c3:
    dmin = st.date_input("📅 Depuis", value=(df["review_date"].min().date() if df["review_date"].notna().any() else datetime(2024,1,1).date()))
with c4:
    dmax = st.date_input("📅 Jusqu’à", value=(df["review_date"].max().date() if df["review_date"].notna().any() else datetime.now().date()))

# --- BOUTON RAFRAÎCHIR ---
rf_col1, rf_col2, rf_col3 = st.columns([1, 1, 2])
with rf_col1:
    refresh = st.button("🔄 Rafraîchir les données", help="Force le recalcul des KPI et graphiques selon les filtres actuels.")
with rf_col2:
    hard_refresh = st.checkbox("Purger le cache", value=False, help="Vide les caches (chargement/traitements) avant de recalculer.")
if refresh:
    if hard_refresh:
        st.cache_data.clear()       # purge les fonctions @st.cache_data
        # st.cache_resource.clear() # décommente si besoin de purger les ressources persistantes
    st.experimental_rerun()         # relance l’app avec les filtres courants

# Filtre principal basé sur les widgets
mask = df["language"].isin(chosen_langs)
if only_positive and "voted_up" in df.columns:
    mask &= (df["voted_up"] == True)
if df["review_date"].notna().any():
    mask &= (df["review_date"].dt.date >= dmin) & (df["review_date"].dt.date <= dmax)

df_f = df[mask].copy()

# ============ SENTIMENT ============
sia = get_vader()
if "sentiment" not in df_f.columns:
    df_f["sentiment"] = df_f["cleaned_review"].apply(lambda t: sia.polarity_scores(t)["compound"])

# ============ KPI STRIP ============
st.markdown("<div class='block'>", unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown("<div class='metric'>", unsafe_allow_html=True)
    st.markdown("<div class='label'>Satisfaction moyenne</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='value'>{df_f['sentiment'].mean():.2f}</div>", unsafe_allow_html=True)
    st.caption("Entre -1 (très négatif) et +1 (très positif).")
    st.markdown("</div>", unsafe_allow_html=True)
with k2:
    pos = (df_f["sentiment"] > 0.05).mean()*100
    neu = ((df_f["sentiment"] >= -0.05) & (df_f["sentiment"] <= 0.05)).mean()*100
    neg = (df_f["sentiment"] < -0.05).mean()*100
    st.markdown("<div class='metric'>", unsafe_allow_html=True)
    st.markdown("<div class='label'>% Pos / Neu / Neg</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='value'>{pos:.0f}% / {neu:.0f}% / {neg:.0f}%</div>", unsafe_allow_html=True)
    st.caption("Équilibre global des avis.")
    st.markdown("</div>", unsafe_allow_html=True)
with k3:
    avg_len = df_f["cleaned_review"].str.split().apply(len).replace(0, np.nan).mean()
    st.markdown("<div class='metric'>", unsafe_allow_html=True)
    st.markdown("<div class='label'>Mots / avis (moy.)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='value'>{(avg_len or 0):.1f}</div>", unsafe_allow_html=True)
    st.caption("Plus c’est élevé, plus l’avis est détaillé.")
    st.markdown("</div>", unsafe_allow_html=True)
with k4:
    st.markdown("<div class='metric'>", unsafe_allow_html=True)
    st.markdown("<div class='label'>Langues distinctes</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='value'>{df_f['language'].nunique()}</div>", unsafe_allow_html=True)
    st.caption("Diversité linguistique des retours.")
    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.write("")  # spacing

# ============ TABS ============
tab_syn, tab_sent, tab_topics, tab_seg, tab_updates, tab_price, tab_table = st.tabs(
    ["📌 Synthèse", "🙂 Sentiment", "🧩 Thèmes", "👥 Segments", "🛠️ Mises à jour", "💶 Prix & Sentiment", "🔎 Explorateur"]
)

# ====== SYNTHÈSE ======
with tab_syn:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    colA, colB = st.columns([2,1])
    with colA:
        if df_f["review_date"].notna().any():
            ts = df_f.dropna(subset=["review_date"]).copy()
            ts["date"] = ts["review_date"].dt.to_period("W").dt.start_time
            series = ts.groupby("date")["sentiment"].mean()
            fig, ax = plt.subplots(figsize=(8,3))
            series.plot(ax=ax, color=ACCENT)
            ax.set_title("Évolution hebdomadaire de la satisfaction")
            ax.set_xlabel(""); ax.set_ylabel("Score (-1 à +1)")
            st.pyplot(fig)
            st.caption("Chaque point = moyenne des avis sur la semaine. Les hausses/baisses reflètent l’impact d’événements (mise à jour, promo, bug…).")

    # Thèmes rapides
    with colB:
        default_theme_dict = {
            "performances":["lag","fps","performance","stuttering","freeze","latence"],
            "gameplay":["gameplay","controls","mécaniques","mechanics","control"],
            "graphismes":["graphics","graphismes","art","textures"],
            "multijoueur":["multiplayer","coop","serveur","server"],
            "bugs":["bug","crash","error","issue","glitch"],
            "contenu":["content","dlc","missions","maps","map"]
        }
        theme_dict = default_theme_dict
        def contains_any(text, kws): 
            t = text.lower(); return any(k.lower() in t for k in kws)
        rows = []
        for th, kws in theme_dict.items():
            rows.append((th, df_f["cleaned_review"].apply(lambda x: contains_any(x, kws)).mean()*100))
        freq_df = pd.DataFrame(rows, columns=["Thème","Fréquence (%)"]).sort_values("Fréquence (%)", ascending=False)
        fig, ax = plt.subplots(figsize=(4,3))
        sns.barplot(data=freq_df.head(5), y="Thème", x="Fréquence (%)", ax=ax, color="#5E81AC")
        ax.set_xlim(0,100)
        st.pyplot(fig)
        st.caption("Part des avis qui mentionnent chaque thème. Plus la barre est grande, plus le sujet est souvent évoqué.")
    st.markdown("</div>", unsafe_allow_html=True)

# ====== SENTIMENT ======
with tab_sent:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.6,1])
    with c1:
        if df_f["review_date"].notna().any():
            ts = df_f.dropna(subset=["review_date"]).copy()
            ts["date"] = ts["review_date"].dt.to_period("W").dt.start_time
            series = ts.groupby("date")["sentiment"].mean()
            fig, ax = plt.subplots(figsize=(7,3))
            series.plot(ax=ax, color=ACCENT)
            ax.set_title("Tendance du sentiment")
            ax.set_ylabel("Score (-1 à +1)")
            st.pyplot(fig)
            st.caption("Suivi du ressenti dans le temps pour repérer les périodes de hausse/baisse de satisfaction.")
    with c2:
        fig, ax = plt.subplots(figsize=(4,3))
        sns.barplot(x=["Positif","Neutre","Négatif"], y=[pos,neu,neg], ax=ax, palette=[GOOD,"#9AA0A6",BAD])
        ax.set_ylabel("%")
        st.pyplot(fig)
        st.caption("Répartition des avis. Indique si la perception est globalement favorable ou partagée.")
    st.markdown("</div>", unsafe_allow_html=True)

# ====== THÈMES ======
with tab_topics:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    c1, c2 = st.columns([1,1])
    with c1:
        st.dataframe(freq_df, use_container_width=True)
        st.caption("Pourcentage d’avis qui contiennent les mots-clés du thème.")
    with c2:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(data=freq_df, x="Fréquence (%)", y="Thème", ax=ax, color="#81A1C1")
        ax.set_xlim(0,100)
        st.pyplot(fig)
        st.caption("Aide à prioriser : problèmes récurrents à corriger, points forts à valoriser.")
    if st.checkbox("Afficher le nuage de mots"):
        text = " ".join(df_f["cleaned_review"].dropna())
        if text.strip():
            wc = WordCloud(width=1000, height=400, background_color="white").generate(text)
            fig, ax = plt.subplots(figsize=(10,4))
            ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
            st.pyplot(fig)
            st.caption("Plus un mot est gros, plus il est cité.")
    st.markdown("</div>", unsafe_allow_html=True)

# ====== SEGMENTS ======
with tab_seg:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    def segment(h):
        try: h = float(h)
        except: return "Inconnu"
        if h < 10: return "Occasionnel (<10h)"
        if h < 100: return "Régulier (10–100h)"
        return "Intensif (≥100h)"
    if "playtime_hours" in df_f.columns:
        df_f["player_segment"] = df_f["playtime_hours"].apply(segment)
        seg = df_f.groupby("player_segment").agg(
            avis=("review_text","count"),
            sentiment_moy=("sentiment","mean"),
            pos_pct=("sentiment", lambda s: (s>0.05).mean()*100),
            neg_pct=("sentiment", lambda s: (s<-0.05).mean()*100)
        ).reset_index().sort_values("avis", ascending=False)
        st.dataframe(seg, use_container_width=True)
        fig, ax = plt.subplots(figsize=(6,3))
        sns.barplot(data=seg, x="player_segment", y="sentiment_moy", ax=ax, color="#88C0D0")
        ax.set_title("Sentiment moyen par segment"); ax.set_xlabel("")
        st.pyplot(fig)
        st.caption("Compare la satisfaction selon l’expérience (heures jouées). Utile pour des actions ciblées par profil.")
    else:
        st.info("Aucun 'playtime_hours' détecté. Normalisez depuis author.playtime_forever si possible.")
    st.markdown("</div>", unsafe_allow_html=True)

# ====== MISES À JOUR ======
with tab_updates:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    if df_f["review_date"].notna().any():
        pivot = st.date_input("Date de mise à jour (pivot)", value=df_f["review_date"].min().date())
        before = df_f[df_f["review_date"].dt.date < pivot]["sentiment"].mean()
        after  = df_f[df_f["review_date"].dt.date >= pivot]["sentiment"].mean()
        var = (after - before) if (pd.notna(before) and pd.notna(after)) else np.nan
        m1, m2, m3 = st.columns(3)
        m1.metric("Avant", f"{before:.2f}" if pd.notna(before) else "n/a")
        m2.metric("Après", f"{after:.2f}" if pd.notna(after) else "n/a")
        m3.metric("Variation", f"{var:+.2f}" if pd.notna(var) else "n/a")
        # boxplot
        tdf = df_f.dropna(subset=["review_date"]).copy()
        tdf["period"] = np.where(tdf["review_date"].dt.date < pivot, "Avant", "Après")
        fig, ax = plt.subplots(figsize=(6,3))
        sns.boxplot(data=tdf, x="period", y="sentiment", ax=ax, palette=[WARN, ACCENT])
        st.pyplot(fig)
        st.caption("Si le score augmente après la date, la mise à jour est perçue positivement. Sinon, des points restent à corriger.")
    else:
        st.info("Pas de dates disponibles pour mesurer l’impact d’une mise à jour.")
    st.markdown("</div>", unsafe_allow_html=True)

# ====== PRIX & SENTIMENT (OPTIONNEL) ======
with tab_price:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.caption("Importer un CSV avec colonnes 'date' (YYYY-MM-DD) et 'price' (€).")
    price_file = st.file_uploader("Importer un CSV de prix (optionnel)", type=["csv"])
    if price_file is not None and df_f["review_date"].notna().any():
        price_df = pd.read_csv(price_file)
        # normaliser colonnes
        for c in price_df.columns:
            if c.lower() == "date": price_df.rename(columns={c:"date"}, inplace=True)
            if c.lower() == "price": price_df.rename(columns={c:"price"}, inplace=True)
        if set(["date","price"]).issubset(price_df.columns):
            price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce").dt.date
            daily = df_f.dropna(subset=["review_date"]).copy()
            daily["date"] = daily["review_date"].dt.date
            sent = daily.groupby("date")["sentiment"].mean().reset_index()
            merged = pd.merge(sent, price_df[["date","price"]], on="date", how="inner").sort_values("date")
            if not merged.empty:
                corr = merged["sentiment"].corr(merged["price"])
                st.write(f"Corrélation prix–sentiment : **{corr:.3f}** (Pearson)")
                ds = merged["sentiment"].pct_change()
                dp = merged["price"].pct_change().replace(0, np.nan)
                elasticity = (ds/dp).replace([np.inf,-np.inf], np.nan).dropna().mean()
                st.write(f"Élasticité moyenne (Δsentiment/Δprix) : **{elasticity:.3f}**")
                c1, c2 = st.columns(2)
                with c1:
                    fig, ax = plt.subplots(figsize=(6,3))
                    ax.plot(merged["date"], merged["sentiment"], color=ACCENT)
                    ax.set_title("Sentiment moyen (jour)")
                    ax.tick_params(axis='x', rotation=45)
                    st.pyplot(fig)
                with c2:
                    fig, ax = plt.subplots(figsize=(6,3))
                    ax.plot(merged["date"], merged["price"], color=GOOD)
                    ax.set_title("Prix (jour)")
                    ax.tick_params(axis='x', rotation=45)
                    st.pyplot(fig)
                st.caption("Vérifie si le ressenti suit les variations de prix (corrélation/élasticité).")
            else:
                st.info("Pas d'intersection de dates entre avis et prix.")
        else:
            st.warning("CSV attendu avec colonnes 'date' et 'price'.")
    else:
        st.caption("Ajoutez un CSV pour activer l’analyse prix–sentiment.")
    st.markdown("</div>", unsafe_allow_html=True)

# ====== TABLEAU EXPLORATEUR ======
with tab_table:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.subheader("Explorateur d’avis (filtrés)")
    view_cols = ["review_date","language","sentiment","playtime_hours","review_text"]
    view_cols = [c for c in view_cols if c in df_f.columns]
    st.dataframe(df_f[view_cols].sort_values(by=view_cols[0]).head(300), use_container_width=True)
    csv = df_f[view_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Exporter les avis filtrés (CSV)", data=csv, file_name="avis_filtrés.csv", mime="text/csv")
    st.caption("Modifiez les filtres puis cliquez sur « Rafraîchir les données » pour mettre à jour l’ensemble des indicateurs et graphiques.")
    st.markdown("</div>", unsafe_allow_html=True)
