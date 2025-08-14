# dashboard.py
# Dashboard Streamlit (mode clair) – analyses avancées et schémas par catégories
# - Filtres persistants & clamp
# - Recherche mots-clés (ET/OU) + surlignage
# - Bouton Rafraîchir (st.rerun)
# - Zéro warning Seaborn (hue/palette)
# - Onglets d’analyses enrichies

import streamlit as st
import pandas as pd
import numpy as np
import pymongo
import matplotlib.pyplot as plt
import seaborn as sns
import re, requests
from datetime import datetime, date
from collections import Counter
from wordcloud import WordCloud
from PIL import Image
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# =================== CONFIG UI ===================
st.set_page_config(page_title="Avis Steam – Insights", layout="wide")

PRIMARY = "#2563eb"
GOOD    = "#16a34a"
WARN    = "#f59e0b"
BAD     = "#dc2626"
GREY    = "#9CA3AF"
BORDER  = "#e5e7eb"
CARD_BG = "#ffffff"

sns.set_style("whitegrid")

st.markdown(f"""
<style>
.block {{
  background:{CARD_BG};
  border:1px solid {BORDER};
  border-radius:14px; padding:16px 16px 12px 16px;
  margin-bottom: 12px;
}}
.metric {{
  background:{CARD_BG};
  border:1px solid {BORDER};
  border-radius:12px; padding:14px;
  text-align:center;
}}
.metric .value {{ font-size: 26px; font-weight:700; }}
.metric .label {{ font-size: 13px; opacity:0.85; }}
.caption {{ font-size: 13px; opacity:.9; margin-top:6px; }}
.small {{ font-size: 12px; opacity:.9; }}
</style>
""", unsafe_allow_html=True)

# =================== UTILS ===================
@st.cache_resource(show_spinner=False)
def get_db():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    return client["steamdb"]

@st.cache_data(show_spinner=False)
def get_game_name(app_id: str) -> str:
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
    s = series.fillna("").astype(str).str.lower()
    s = s.str.replace(r"http\S+", " ", regex=True)
    s = s.str.replace(r"[^a-zA-ZÀ-ÖØ-öø-ÿ0-9\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def clamp(d, lo, hi):
    if d < lo: return lo
    if d > hi: return hi
    return d

def compute_sentiment(sia, text: str) -> float:
    return sia.polarity_scores(text)["compound"] if isinstance(text, str) else 0.0

# =================== DATA LOAD ===================
db = get_db()
collections = [c for c in db.list_collection_names() if c.startswith("reviews_")]
app_ids = [c.replace("reviews_", "") for c in collections]
if not app_ids:
    st.error("Aucune collection 'reviews_<app_id>' dans MongoDB.")
    st.stop()

names = {app_id: get_game_name(app_id) for app_id in app_ids}
selected_app = st.selectbox("🎮 Choisir un jeu", options=sorted(app_ids), format_func=lambda a: names[a])

img_url = f"https://cdn.akamai.steamstatic.com/steam/apps/{selected_app}/header.jpg"
st.markdown(f"## {names[selected_app]}")
st.image(img_url, use_container_width=True)

@st.cache_data(show_spinner=False)
def load_df(collection):
    docs = list(db[collection].find())
    if not docs: return pd.DataFrame()
    df = pd.DataFrame(docs)

    # Standardisation
    if "review" in df.columns:
        df["review_text"] = df["review"]
    elif "review_text" not in df.columns:
        df["review_text"] = ""

    if "language" not in df.columns:
        df["language"] = "unknown"

    if "voted_up" not in df.columns:
        df["voted_up"] = np.nan

    if "author" in df.columns:
        try:
            pt = df["author"].apply(lambda x: x.get("playtime_forever") if isinstance(x, dict) else None)
            df["playtime_hours"] = (pd.to_numeric(pt, errors="coerce").fillna(0)/60).round(2)
        except:
            pass

    # Dates
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

# =================== DATES PERSISTANTES + CLAMP ===================
if df["review_date"].notna().any():
    global_min = df["review_date"].min().date()
    global_max = df["review_date"].max().date()
else:
    global_min = date(2024, 1, 1)
    global_max = datetime.now().date()

# Réinit si changement de jeu
if st.session_state.get("current_app") != selected_app:
    st.session_state.current_app = selected_app
    st.session_state.date_min = global_min
    st.session_state.date_max = global_max

# Défaut + clamp
st.session_state.date_min = clamp(st.session_state.get("date_min", global_min), global_min, global_max)
st.session_state.date_max = clamp(st.session_state.get("date_max", global_max), global_min, global_max)
if st.session_state.date_min > st.session_state.date_max:
    st.session_state.date_min, st.session_state.date_max = st.session_state.date_max, st.session_state.date_min

# =================== FILTRES ===================
langs = sorted(df["language"].dropna().unique().tolist())
c1, c2, c3, c4, c5 = st.columns([1.3,1,1,1,1])
with c1:
    chosen_langs = st.multiselect("🌐 Langues", options=langs, default=langs,
                                  help="Filtre les avis par langue.", key="langs_key")
with c2:
    only_positive = st.checkbox("👍 Avis positifs uniquement", value=False,
                                help="Filtre sur voted_up == True (si disponible).", key="pos_only_key")
with c3:
    dmin = st.date_input("📅 Depuis", value=st.session_state.date_min, min_value=global_min, max_value=global_max, key="date_min_input")
with c4:
    dmax = st.date_input("📅 Jusqu’à", value=st.session_state.date_max, min_value=global_min, max_value=global_max, key="date_max_input")
with c5:
    refresh = st.button("🔄 Rafraîchir", help="Met à jour tous les indicateurs et graphiques selon les filtres sélectionnés.")
    hard_refresh = st.checkbox("Purger le cache", value=False, help="Vide le cache avant recalcul.", key="purge_cache")

# MàJ session
st.session_state.date_min = clamp(dmin, global_min, global_max)
st.session_state.date_max = clamp(dmax, global_min, global_max)
if st.session_state.date_min > st.session_state.date_max:
    st.session_state.date_min, st.session_state.date_max = st.session_state.date_max, st.session_state.date_min

# ==== Recherche mots-clés (ET/OU) ====
kw1, kw2 = st.columns([2,1])
with kw1:
    keywords_raw = st.text_input("🔎 Mots-clés (séparés par des virgules)", placeholder="ex : bug, freeze, crash")
with kw2:
    match_all = st.checkbox("Contient tous (ET logique)", value=False, help="Coché : tous les mots ; Décoché : au moins un.")

st.caption("💡 Modifiez les filtres, saisissez des mots-clés si besoin, puis cliquez sur **Rafraîchir**.")

if refresh:
    if hard_refresh:
        st.cache_data.clear()
    st.rerun()

# =================== APPLICATION DES FILTRES ===================
mask = pd.Series(True, index=df.index)
if len(chosen_langs) > 0:
    mask &= df["language"].isin(chosen_langs)
if "voted_up" in df.columns and st.session_state.pos_only_key:
    mask &= (df["voted_up"] == True)
if df["review_date"].notna().any():
    dates_ok = (df["review_date"].dt.date >= st.session_state.date_min) & (df["review_date"].dt.date <= st.session_state.date_max)
    mask &= dates_ok
df_f = df[mask].copy()

# Mots-clés
if keywords_raw and keywords_raw.strip() and not df_f.empty:
    kws = [k.strip().lower() for k in keywords_raw.split(",") if k.strip()]
    if kws:
        if match_all:
            lookaheads = "".join([rf"(?=.*\b{re.escape(k)}\b)" for k in kws])
            pattern = lookaheads + r".*"
        else:
            pattern = r"\b(" + "|".join([re.escape(k) for k in kws]) + r")\b"
        df_f = df_f[df_f["cleaned_review"].str.contains(pattern, regex=True, na=False)]

st.caption(f"🗓️ Période : **{st.session_state.date_min} → {st.session_state.date_max}** | Avis filtrés : **{len(df_f):,}**")

# =================== CALCULS ===================
sia = get_vader()
if not df_f.empty:
    df_f["sentiment"] = df_f["cleaned_review"].apply(lambda t: compute_sentiment(sia, t))
else:
    df_f["sentiment"] = []

pos = (df_f["sentiment"] > 0.05).mean()*100 if len(df_f) else 0.0
neu = ((df_f["sentiment"] >= -0.05) & (df_f["sentiment"] <= 0.05)).mean()*100 if len(df_f) else 0.0
neg = (df_f["sentiment"] < -0.05).mean()*100 if len(df_f) else 0.0
avg_len = df_f["cleaned_review"].str.split().apply(len).replace(0, np.nan).mean() if len(df_f) else 0.0

# Thèmes (règles simples)
theme_dict = {
    "performances":["lag","fps","performance","stuttering","freeze","latence"],
    "gameplay":["gameplay","controls","mécaniques","mechanics","control"],
    "graphismes":["graphics","graphismes","art","textures"],
    "multijoueur":["multiplayer","coop","serveur","server"],
    "bugs":["bug","crash","error","issue","glitch"],
    "contenu":["content","dlc","missions","maps","map"]
}
def contains_any(text, kws):
    t = str(text).lower()
    return any(k.lower() in t for k in kws)

rows = []
for th, kws in theme_dict.items():
    freq = df_f["cleaned_review"].apply(lambda x: contains_any(x, kws)).mean()*100 if len(df_f) else 0.0
    rows.append((th, freq))
freq_df = pd.DataFrame(rows, columns=["Thème","Fréquence (%)"]).sort_values("Fréquence (%)", ascending=False)

# =================== KPI ===================
st.markdown("<div class='block'>", unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown("<div class='metric'>", unsafe_allow_html=True)
    st.markdown("<div class='label'>Satisfaction moyenne</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='value'>{(df_f['sentiment'].mean() if len(df_f) else 0):.2f}</div>", unsafe_allow_html=True)
    st.caption("Échelle : -1 (négatif) → +1 (positif).")
    st.markdown("</div>", unsafe_allow_html=True)
with k2:
    st.markdown("<div class='metric'>", unsafe_allow_html=True)
    st.markdown("<div class='label'>% Pos / Neu / Neg</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='value'>{pos:.0f}% / {neu:.0f}% / {neg:.0f}%</div>", unsafe_allow_html=True)
    st.caption("Répartition des avis sur la période filtrée.")
    st.markdown("</div>", unsafe_allow_html=True)
with k3:
    st.markdown("<div class='metric'>", unsafe_allow_html=True)
    st.markdown("<div class='label'>Mots / avis (moy.)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='value'>{(avg_len or 0):.1f}</div>", unsafe_allow_html=True)
    st.caption("Indicateur d’engagement (avis plus détaillés).")
    st.markdown("</div>", unsafe_allow_html=True)
with k4:
    st.markdown("<div class='metric'>", unsafe_allow_html=True)
    st.markdown("<div class='label'>Langues distinctes</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='value'>{df_f['language'].nunique() if len(df_f) else 0}</div>", unsafe_allow_html=True)
    st.caption("Diversité linguistique des retours.")
    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# =================== TABS ===================
tabs = st.tabs([
    "📌 Synthèse",
    "🙂 Sentiment",
    "🧩 Thèmes",
    "🌍 Langues",
    "⏱️ Heures de jeu",
    "✍️ Longueur & lisibilité",
    "🔗 Cooccurrences",
    "⚠️ Anomalies",
    "✅ Qualité des données",
    "🔎 Explorateur"
])
(tab_syn, tab_sent, tab_topics, tab_lang, tab_play, tab_len, tab_cooc, tab_anom, tab_qual, tab_table) = tabs

# ===== Synthèse =====
with tab_syn:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.info("**Objectif :** vue d’ensemble – tendance du ressenti et thèmes dominants (période filtrée).")
    cA, cB = st.columns([2,1])
    with cA:
        if df_f["review_date"].notna().any() and len(df_f):
            ts = df_f.dropna(subset=["review_date"]).copy()
            ts["date"] = ts["review_date"].dt.to_period("W").dt.start_time
            series = ts.groupby("date")["sentiment"].mean()
            fig, ax = plt.subplots(figsize=(8,3))
            ax.plot(series.index, series.values, color=PRIMARY)
            ax.set_title("Évolution hebdomadaire de la satisfaction")
            ax.set_xlabel(""); ax.set_ylabel("Score (-1 à +1)")
            st.pyplot(fig)
            st.caption(f"Période : {st.session_state.date_min} → {st.session_state.date_max}.")
        else:
            st.info("Pas de dates exploitables dans la sélection courante.")
    with cB:
        fig, ax = plt.subplots(figsize=(4,3))
        sns.barplot(data=freq_df.head(5), y="Thème", x="Fréquence (%)", ax=ax, color="#60a5fa")
        ax.set_xlim(0,100)
        st.pyplot(fig)
        st.caption("Part des avis mentionnant ces thèmes (règles simples).")
    st.markdown("</div>", unsafe_allow_html=True)

# ===== Sentiment =====
with tab_sent:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.info("**Objectif :** perception globale (répartition et tendance).")
    c1, c2, c3 = st.columns([1.4,1,1])
    with c1:
        if df_f["review_date"].notna().any() and len(df_f):
            ts = df_f.dropna(subset=["review_date"]).copy()
            ts["date"] = ts["review_date"].dt.to_period("W").dt.start_time
            series = ts.groupby("date")["sentiment"].mean().rolling(3).mean()
            fig, ax = plt.subplots(figsize=(7,3))
            ax.plot(series.index, series.values, color=PRIMARY)
            ax.set_title("Sentiment hebdo (moyenne mobile x3)")
            ax.set_ylabel("Score (-1 à +1)")
            st.pyplot(fig)
            st.caption("Lissage pour repérer les tendances.")
        else:
            st.info("Pas de dates exploitables.")
    with c2:
        dist_df = pd.DataFrame({"cat":["Positif","Neutre","Négatif"], "val":[pos,neu,neg]})
        fig, ax = plt.subplots(figsize=(4,3))
        sns.barplot(data=dist_df, x="cat", y="val", hue="cat",
                    palette={"Positif":GOOD,"Neutre":GREY,"Négatif":BAD}, legend=False, ax=ax)
        ax.set_ylabel("%"); ax.set_xlabel("")
        st.pyplot(fig)
        st.caption("Répartition sur la période filtrée.")
    with c3:
        if len(df_f):
            fig, ax = plt.subplots(figsize=(4,3))
            sns.histplot(df_f["sentiment"], bins=30, kde=True, ax=ax, color="#93c5fd")
            ax.set_title("Distribution des scores")
            st.pyplot(fig)
            st.caption("Comprendre l’étalement des ressentis.")
    st.markdown("</div>", unsafe_allow_html=True)

# ===== Thèmes =====
with tab_topics:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.info("**Objectif :** prioriser les sujets – drill‑down par thème.")
    c1, c2 = st.columns([1,1])
    with c1:
        st.dataframe(freq_df, use_container_width=True)
    with c2:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(data=freq_df, x="Fréquence (%)", y="Thème", ax=ax, color="#93c5fd")
        ax.set_xlim(0,100)
        st.pyplot(fig)
    # Drill-down thème
    theme_choice = st.selectbox("🔍 Analyser un thème en détail", options=list(theme_dict.keys()))
    sub = df_f[df_f["cleaned_review"].apply(lambda t: contains_any(t, theme_dict[theme_choice]))].copy()
    st.caption(f"Avis correspondant au thème **{theme_choice}** : {len(sub):,}")
    if not sub.empty:
        # Top mots du thème
        STOP = set("""the and for you are with that this was but not have your plus tres très les des une que pour est pas qui dans sur par au aux du de la le un et ou mais donc car game jeu games steam play jouer joue joué""".split())
        words = [w for t in sub["cleaned_review"].tolist() for w in str(t).split() if len(w)>2 and w not in STOP]
        freq = Counter(words).most_common(30)
        topw = pd.DataFrame(freq, columns=["Mot","Fréquence"])
        st.dataframe(topw, use_container_width=True)
        # Sentiment du thème
        fig, ax = plt.subplots(figsize=(6,3))
        sns.histplot(sub["sentiment"], bins=30, kde=True, ax=ax, color="#60a5fa")
        ax.set_title(f"Distribution des sentiments – {theme_choice}")
        st.pyplot(fig)
        st.caption("Aide à juger si le thème est plutôt perçu positivement ou négativement.")
    # WordCloud global (option)
    if st.checkbox("Afficher le nuage de mots (global)"):
        text = " ".join(df_f["cleaned_review"].dropna())
        if text.strip():
            wc = WordCloud(width=1000, height=400, background_color="white").generate(text)
            fig, ax = plt.subplots(figsize=(10,4))
            ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
            st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ===== Langues =====
with tab_lang:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.info("**Objectif :** mesurer le ressenti par langue (qualité localisation, communication).")
    if len(df_f):
        lang_agg = df_f.groupby("language").agg(
            avis=("review_text","count"),
            sentiment_moy=("sentiment","mean")
        ).reset_index().sort_values("avis", ascending=False)
        c1, c2 = st.columns([1,1])
        with c1:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.barplot(data=lang_agg.head(12), x="avis", y="language", ax=ax, color="#93c5fd")
            ax.set_title("Volume d'avis par langue (Top 12)")
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.barplot(data=lang_agg.head(12), x="sentiment_moy", y="language", ax=ax, color="#60a5fa")
            ax.set_title("Sentiment moyen par langue (Top 12)")
            st.pyplot(fig)
        st.caption("Comparer volume et satisfaction pour adapter support/community management.")
    else:
        st.info("Aucune donnée filtrée.")
    st.markdown("</div>", unsafe_allow_html=True)

# ===== Heures de jeu =====
with tab_play:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.info("**Objectif :** comprendre les différences de perception selon l’expérience (heures jouées).")
    def segment(h):
        try: h = float(h)
        except: return "Inconnu"
        if h < 10: return "Occasionnel (<10h)"
        if h < 100: return "Régulier (10–100h)"
        return "Intensif (≥100h)"
    if "playtime_hours" in df_f.columns and len(df_f):
        df_f["player_segment"] = df_f["playtime_hours"].apply(segment)
        seg = df_f.groupby("player_segment").agg(
            avis=("review_text","count"),
            sentiment_moy=("sentiment","mean")
        ).reset_index().sort_values("avis", ascending=False)
        c1, c2 = st.columns([1,1])
        with c1:
            st.dataframe(seg, use_container_width=True)
        with c2:
            fig, ax = plt.subplots(figsize=(6,3))
            sns.barplot(data=seg, x="player_segment", y="sentiment_moy", ax=ax, color="#60a5fa")
            ax.set_title("Sentiment moyen par segment"); ax.set_xlabel("")
            st.pyplot(fig)
        # Scatter sentiment vs heures
        fig, ax = plt.subplots(figsize=(6,3))
        sns.scatterplot(data=df_f, x="playtime_hours", y="sentiment", ax=ax, color="#2563eb", alpha=0.4)
        ax.set_title("Sentiment vs Heures jouées"); ax.set_xlabel("Heures")
        st.pyplot(fig)
        st.caption("Permet d’identifier si les joueurs expérimentés sont plus critiques ou plus satisfaits.")
    else:
        st.info("Heures de jeu non disponibles.")
    st.markdown("</div>", unsafe_allow_html=True)

# ===== Longueur & lisibilité =====
with tab_len:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.info("**Objectif :** relier la longueur des avis au sentiment (engagement vs satisfaction).")
    if len(df_f):
        df_f["word_count"] = df_f["cleaned_review"].str.split().apply(len)
        c1, c2 = st.columns([1,1])
        with c1:
            fig, ax = plt.subplots(figsize=(6,3))
            sns.histplot(df_f["word_count"], bins=40, ax=ax, color="#93c5fd")
            ax.set_title("Distribution de la longueur des avis")
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(6,3))
            sns.scatterplot(data=df_f, x="word_count", y="sentiment", ax=ax, color="#2563eb", alpha=0.4)
            ax.set_title("Sentiment vs longueur d’avis"); ax.set_xlabel("Mots/avis")
            st.pyplot(fig)
        st.caption("Regarder si les avis détaillés sont plutôt plus positifs ou négatifs.")
    else:
        st.info("Aucune donnée filtrée.")
    st.markdown("</div>", unsafe_allow_html=True)

# ===== Cooccurrences =====
with tab_cooc:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.info("**Objectif :** voir quels thèmes sont souvent mentionnés ensemble (cartographie des problèmes).")
    if len(df_f):
        themes = list(theme_dict.keys())
        M = pd.DataFrame(0, index=themes, columns=themes, dtype=int)
        for _, row in df_f.iterrows():
            text = row.get("cleaned_review","")
            present = [th for th,kws in theme_dict.items() if contains_any(text, kws)]
            for i in range(len(present)):
                for j in range(i, len(present)):
                    M.loc[present[i], present[j]] += 1
                    if i != j:
                        M.loc[present[j], present[i]] += 1
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(M, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Cooccurrences de thèmes")
        st.pyplot(fig)
        st.caption("Ex. Bugs + Performances → priorité aux correctifs et optimisation.")
    else:
        st.info("Aucune donnée filtrée.")
    st.markdown("</div>", unsafe_allow_html=True)

# ===== Anomalies =====
with tab_anom:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.info("**Objectif :** détecter des semaines atypiques (sauts de satisfaction ou volume inhabituel).")
    if df_f["review_date"].notna().any() and len(df_f):
        ts = df_f.dropna(subset=["review_date"]).copy()
        ts["week"] = ts["review_date"].dt.to_period("W").dt.start_time
        agg = ts.groupby("week").agg(sent=("sentiment","mean"), n=("review_text","count")).reset_index()
        # z-score simple
        agg["z_sent"] = (agg["sent"] - agg["sent"].mean())/agg["sent"].std(ddof=0)
        agg["z_n"]    = (agg["n"] - agg["n"].mean())/agg["n"].std(ddof=0)
        anomalies = agg[(agg["z_sent"].abs()>2) | (agg["z_n"].abs()>2)]
        c1, c2 = st.columns([1,1])
        with c1:
            fig, ax = plt.subplots(figsize=(6,3))
            ax.plot(agg["week"], agg["sent"], color=PRIMARY)
            ax.scatter(anomalies["week"], anomalies["sent"], color=BAD)
            ax.set_title("Sentiment hebdo & anomalies")
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(6,3))
            ax.plot(agg["week"], agg["n"], color=PRIMARY)
            ax.scatter(anomalies["week"], anomalies["n"], color=BAD)
            ax.set_title("Volume d’avis hebdo & anomalies")
            st.pyplot(fig)
        st.caption("Points rouges = semaines atypiques (à investiguer : patch, promo, bad buzz…).")
        if not anomalies.empty:
            st.dataframe(anomalies[["week","sent","n","z_sent","z_n"]], use_container_width=True)
    else:
        st.info("Pas de dates exploitables.")
    st.markdown("</div>", unsafe_allow_html=True)

# ===== Qualité des données =====
with tab_qual:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.info("**Objectif :** vérifier complétude, duplicats, formats – gage d’analyses fiables.")
    if len(df_f):
        cols = ["review_text","language","review_date"]
        comp = {c: f"{df_f[c].notna().mean()*100:.1f}%" if c in df_f.columns else "absent" for c in cols}
        st.write("**Complétude (présence de valeurs non nulles)** :", comp)
        # duplicats basés sur texte + date
        dup_rate = (df_f.duplicated(subset=["review_text","review_date"]).mean()*100) if {"review_text","review_date"}.issubset(df_f.columns) else 0.0
        st.write(f"**Duplicats estimés** : {dup_rate:.1f}%")
        # top reviewers (si id/author dispo)
        if "author" in df_f.columns:
            try:
                auth = df_f["author"].apply(lambda x: x.get("steamid") if isinstance(x, dict) else None)
                top = auth.value_counts().head(10).reset_index()
                top.columns = ["steamid","avis"]
                st.dataframe(top, use_container_width=True)
                st.caption("Vérifier les contributions massives (spam ou fans très actifs).")
            except:
                pass
    else:
        st.info("Aucune donnée filtrée.")
    st.markdown("</div>", unsafe_allow_html=True)

# ===== Explorateur =====
with tab_table:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.info("**Objectif :** lire des avis concrets correspondant aux filtres – avec surlignage des mots‑clés.")
    view_cols = ["review_date","language","sentiment","playtime_hours","review_text"]
    view_cols = [c for c in view_cols if c in df_f.columns]
    def highlight_keywords(text, kws):
        if not kws or not isinstance(text, str): return text
        pattern = r"(" + "|".join([re.escape(k) for k in kws]) + r")"
        return re.sub(pattern, r"<mark>\1</mark>", text, flags=re.IGNORECASE)
    if view_cols and len(df_f):
        df_view = df_f[view_cols].copy()
        if keywords_raw and keywords_raw.strip():
            kws = [k.strip() for k in keywords_raw.split(",") if k.strip()]
            if kws:
                df_view["review_text"] = df_view["review_text"].astype(str).apply(lambda t: highlight_keywords(t, kws))
                st.write("Les mots-clés sont **surlignés** dans les avis (jaune).")
        st.write(df_view.to_html(escape=False, index=False), unsafe_allow_html=True)
        csv = df_f[["review_date","language","sentiment","playtime_hours","review_text"]].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Exporter les avis filtrés (CSV)", data=csv, file_name="avis_filtrés.csv", mime="text/csv")
    else:
        st.info("Aucune donnée à afficher avec les filtres actuels.")
    st.markdown("</div>", unsafe_allow_html=True)

# ============== Aide mots les plus cités (global) ==============
with st.expander("🔝 Mots les plus cités (période + filtres actifs)"):
    if len(df_f):
        STOP = set("""the and for you are with that this was but not have your plus tres très les des une que pour est pas qui dans sur par au aux du de la le un et ou mais donc car game jeu games steam play jouer joue joué""".split())
        text_all = " ".join(df_f["cleaned_review"].dropna())
        words = [w for w in text_all.split() if len(w) > 2 and w not in STOP]
        freq = Counter(words).most_common(50)
        top_words_df = pd.DataFrame(freq, columns=["Mot", "Fréquence"])
        st.dataframe(top_words_df, use_container_width=True)
        st.caption("Astuce : copiez des mots d’ici vers le champ de recherche ci‑dessus.")
    else:
        st.info("Aucun mot significatif avec les filtres actuels.")
