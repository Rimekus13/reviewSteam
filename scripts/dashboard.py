# dashboard.py
# Dashboard Streamlit - Avis Steam
# UX compact, explications sous chaque graph, filtres globaux + filtres locaux par onglet (sans filtre de langue dans les onglets),
# "Top mots" visible dans Synthèse, dates non chevauchées, onglet Thèmes enrichi.
# + FIX: clés uniques pour tous les sliders (et quelques inputs) pour éviter StreamlitDuplicateElementId.

import streamlit as st
import pandas as pd
import numpy as np
import pymongo
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import requests, re
from datetime import datetime, date, timedelta
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.util import ngrams
from nltk.corpus import stopwords

# ================= UI / THEME =================
st.set_page_config(page_title="Avis Steam – Insights", layout="wide")

PRIMARY = "#2563eb"; GOOD = "#16a34a"; WARN = "#f59e0b"; BAD = "#dc2626"; GREY = "#9CA3AF"; BORDER = "#e5e7eb"
sns.set_style("whitegrid")

st.markdown(f"""
<style>
.card {{ background:#fff; border:1px solid {BORDER}; border-radius:12px; padding:12px 14px 10px; margin-bottom:12px; }}
.kpi {{ background:#fff; border:1px solid {BORDER}; border-radius:10px; padding:10px; text-align:center; }}
.kpi .v {{ font-size:20px; font-weight:700; line-height:1; }}
.kpi .l {{ font-size:12px; opacity:.9; }}
.small {{ font-size:12px; opacity:.9; }}
h3 {{ margin:0 0 6px 0; }}
</style>
""", unsafe_allow_html=True)

# ================= HELPERS =================
DEFAULT_FIGSIZE_WIDE = (5.0, 2.2); DEFAULT_FIGSIZE_TALL = (6.0, 3.0)
def title(ax, txt): ax.set_title(txt, fontsize=11, pad=6)
def compact_time_axis(ax, minticks=3, maxticks=6, rotate=0):
    loc = mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
    ax.xaxis.set_major_locator(loc); ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.tick_params(axis="x", labelrotation=rotate); ax.tick_params(labelsize=9)
def clamp(d, lo, hi):
    if d < lo: return lo
    if d > hi: return hi
    return d

@st.cache_resource(show_spinner=False)
def get_db():
    return pymongo.MongoClient("mongodb://localhost:27017/")["steamdb"]

@st.cache_data(show_spinner=False)
def get_game_name(app_id: str) -> str:
    try:
        url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
        res = requests.get(url, timeout=5)
        return res.json()[str(app_id)]["data"]["name"]
    except:
        return f"App {app_id}"

@st.cache_data(show_spinner=False)
def get_vader():
    try: nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError: nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()

@st.cache_data(show_spinner=False)
def clean_text_series(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.lower()
    s = s.str.replace(r"http\S+", " ", regex=True)
    s = s.str.replace(r"[^a-zA-ZÀ-ÖØ-öø-ÿ0-9\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def compute_sentiment(sia, text: str) -> float:
    return sia.polarity_scores(text)["compound"] if isinstance(text, str) else 0.0

# ---- Helpers analytiques Thèmes ----
@st.cache_data(show_spinner=False)
def ensure_nltk():
    try: nltk.data.find("corpora/stopwords")
    except LookupError: nltk.download("stopwords")
    try: nltk.data.find("tokenizers/punkt")
    except LookupError: nltk.download("punkt")
    return True

def tokenize(text): return [w for w in re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ0-9]+", str(text).lower())]

@st.cache_data(show_spinner=False)
def get_stop_set():
    ensure_nltk()
    sw = set()
    for lg in ("english","french"):
        try: sw |= set(stopwords.words(lg))
        except: pass
    sw |= set("""game games play played playing steam je tu il elle nous vous ils elles plus tres très les des une un le la de du dans sur par pour est pas que avec sans ou mais donc car bien mal bug bugs crash crashe""".split())
    return sw

def top_unigrams_bigrams(texts, n_top=15):
    sw = get_stop_set(); toks=[]
    for t in texts: toks += [w for w in tokenize(t) if len(w)>2 and w not in sw]
    uni = Counter(toks).most_common(n_top)
    big = Counter([" ".join(bg) for bg in ngrams(toks, 2) if all(len(w)>2 and w not in sw for w in bg)]).most_common(n_top)
    return uni, big

def pick_examples(df_sub, n=3):
    pos_examples = df_sub.sort_values("sentiment", ascending=False).head(n)["review_text"].astype(str).tolist()
    neg_examples = df_sub.sort_values("sentiment", ascending=True).head(n)["review_text"].astype(str).tolist()
    def cut(s, L=220):
        s = re.sub(r"\s+", " ", s).strip()
        return s if len(s)<=L else s[:L-1]+"…"
    return [cut(x) for x in pos_examples], [cut(x) for x in neg_examples]

# ================= LOAD DATA =================
db = get_db()
collections = [c for c in db.list_collection_names() if c.startswith("reviews_")]
app_ids = [c.replace("reviews_", "") for c in collections]
if not app_ids:
    st.error("Aucune collection 'reviews_<app_id>' dans MongoDB."); st.stop()

names = {app_id: get_game_name(app_id) for app_id in app_ids}
selected_app = st.selectbox("🎮 Jeu", options=sorted(app_ids), format_func=lambda a: names[a])

st.markdown(f"### {names[selected_app]}")
st.image(f"https://cdn.akamai.steamstatic.com/steam/apps/{selected_app}/header.jpg", use_container_width=True)

@st.cache_data(show_spinner=False)
def load_df(collection):
    docs = list(db[collection].find())
    if not docs: return pd.DataFrame()
    df = pd.DataFrame(docs)
    df["review_text"] = df["review"] if "review" in df.columns else df.get("review_text","")
    if "language" not in df.columns: df["language"] = "unknown"
    if "voted_up" not in df.columns: df["voted_up"] = np.nan

    if "author" in df.columns:
        try:
            pt = df["author"].apply(lambda x: x.get("playtime_forever") if isinstance(x, dict) else None)
            df["playtime_hours"] = (pd.to_numeric(pt, errors="coerce").fillna(0)/60).round(2)
        except: pass

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
    st.warning("Aucune donnée disponible pour ce jeu."); st.stop()

# ================= PLAGE DATES =================
if df["review_date"].notna().any():
    global_min = df["review_date"].min().date(); global_max = df["review_date"].max().date()
else:
    global_min = date(2024, 1, 1); global_max = datetime.now().date()

if st.session_state.get("current_app") != selected_app:
    st.session_state.current_app = selected_app
    st.session_state.date_min = global_min; st.session_state.date_max = global_max

st.session_state.date_min = clamp(st.session_state.get("date_min", global_min), global_min, global_max)
st.session_state.date_max = clamp(st.session_state.get("date_max", global_max), global_min, global_max)
if st.session_state.date_min > st.session_state.date_max:
    st.session_state.date_min, st.session_state.date_max = st.session_state.date_max, st.session_state.date_min

# ================= FILTRES GLOBAUX =================
with st.container():
    st.markdown("<div class='card'><h3>Filtres</h3>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns([1.3, 1, 1, 1, 1.2, 1.5])
    langs = sorted(df["language"].dropna().unique().tolist())
    with c1:
        chosen_langs = st.multiselect("🌐 Langues", options=langs, default=langs, key="global_langs")
    with c2:
        only_positive = st.checkbox("👍 Positifs", value=False, help="Filtre voted_up=True (si dispo).", key="global_pos_only")
    with c3:
        dmin = st.date_input("📅 Depuis", value=st.session_state.date_min, min_value=global_min, max_value=global_max, key="global_date_min")
    with c4:
        dmax = st.date_input("📅 Jusqu’à", value=st.session_state.date_max, min_value=global_min, max_value=global_max, key="global_date_max")
    with c5:
        st.caption("Période rapide"); b1, b2, b3 = st.columns(3)
        if b1.button("7 j", key="quick_7"):  st.session_state.date_min = max(global_min, global_max - timedelta(days=6));  st.session_state.date_max = global_max; st.rerun()
        if b2.button("30 j", key="quick_30"): st.session_state.date_min = max(global_min, global_max - timedelta(days=29)); st.session_state.date_max = global_max; st.rerun()
        if b3.button("90 j", key="quick_90"): st.session_state.date_min = max(global_min, global_max - timedelta(days=89)); st.session_state.date_max = global_max; st.rerun()
    with c6:
        keywords_raw = st.text_input("🔎 Mots‑clés (ex: bug, crash)", key="global_keywords")
        match_all   = st.checkbox("ET logique (tous)", value=False, key="global_keywords_all")

    dA, dB = st.columns([1,1])
    with dA: hard_refresh = st.checkbox("Purger le cache", value=False, help="Vide le cache avant recalcul.", key="global_hard_refresh")
    with dB:
        if st.button("🔄 Rafraîchir", key="global_refresh"):
            if hard_refresh: st.cache_data.clear()
            st.session_state.date_min = clamp(dmin, global_min, global_max)
            st.session_state.date_max = clamp(dmax, global_min, global_max)
            if st.session_state.date_min > st.session_state.date_max:
                st.session_state.date_min, st.session_state.date_max = st.session_state.date_max, st.session_state.date_min
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ================= APPLICATION FILTRES GLOBAUX =================
mask = pd.Series(True, index=df.index)
if len(chosen_langs) > 0: mask &= df["language"].isin(chosen_langs)
if "voted_up" in df.columns and st.session_state.global_pos_only: mask &= (df["voted_up"] == True)
if df["review_date"].notna().any():
    mask &= (df["review_date"].dt.date >= st.session_state.date_min) & (df["review_date"].dt.date <= st.session_state.date_max)
df_f = df[mask].copy()

# Mots-clés globaux
if keywords_raw and keywords_raw.strip() and not df_f.empty:
    kws = [k.strip().lower() for k in keywords_raw.split(",") if k.strip()]
    if kws:
        if match_all:
            lookaheads = "".join([rf"(?=.*\b{re.escape(k)}\b)" for k in kws]); pattern = lookaheads + r".*"
        else:
            pattern = r"\b(" + "|".join([re.escape(k) for k in kws]) + r")\b"
        df_f = df_f[df_f["cleaned_review"].str.contains(pattern, regex=True, na=False)]

st.caption(f"🗓️ Période : **{st.session_state.date_min} → {st.session_state.date_max}** • Avis filtrés : **{len(df_f):,}**")

# ================= CALCULS BASE =================
sia = get_vader()
if not df_f.empty: df_f["sentiment"] = df_f["cleaned_review"].apply(lambda t: compute_sentiment(sia, t))
else: df_f["sentiment"] = []

pos = (df_f["sentiment"] > 0.05).mean()*100 if len(df_f) else 0.0
neu = ((df_f["sentiment"] >= -0.05) & (df_f["sentiment"] <= 0.05)).mean()*100 if len(df_f) else 0.0
neg = (df_f["sentiment"] < -0.05).mean()*100 if len(df_f) else 0.0
avg_len = df_f["cleaned_review"].str.split().apply(len).replace(0, np.nan).mean() if len(df_f) else 0.0

theme_dict = {
    "performances":["lag","fps","performance","stuttering","freeze","latence"],
    "gameplay":["gameplay","controls","mécaniques","mechanics","control"],
    "graphismes":["graphics","graphismes","art","textures"],
    "multijoueur":["multiplayer","coop","serveur","server"],
    "bugs":["bug","crash","error","issue","glitch"],
    "contenu":["content","dlc","missions","maps","map"]
}
def contains_any(text, kws):
    t = str(text).lower(); return any(k.lower() in t for k in kws)

rows = []
for th, kws in theme_dict.items():
    freq = df_f["cleaned_review"].apply(lambda x: contains_any(x, kws)).mean()*100 if len(df_f) else 0.0
    rows.append((th, freq))
freq_df = pd.DataFrame(rows, columns=["Thème","Fréquence (%)"]).sort_values("Fréquence (%)", ascending=False)

# ================= TABS =================
tabs = st.tabs([
    "📌 Synthèse","🙂 Sentiment","🧩 Thèmes","🌍 Langues","⏱️ Heures de jeu",
    "✍️ Longueur & lisibilité","🔗 Cooccurrences","⚠️ Anomalies","✅ Qualité données","🔎 Explorateur","🛠️ Mises à jour"
])
(tab_syn, tab_sent, tab_topics, tab_lang, tab_play, tab_len, tab_cooc, tab_anom, tab_qual, tab_table, tab_updates) = tabs

# --------- SYNTHÈSE ----------
with tab_syn:
    st.markdown("<div class='card'><h3>Vue d’ensemble</h3>", unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    with k1:
        st.markdown("<div class='kpi'><div class='l'>Satisfaction moyenne</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='v'>{(df_f['sentiment'].mean() if len(df_f) else 0):.2f}</div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown("<div class='kpi'><div class='l'>% Pos / Neu / Neg</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='v'>{pos:.0f}% / {neu:.0f}% / {neg:.0f}%</div></div>", unsafe_allow_html=True)
    with k3:
        st.markdown("<div class='kpi'><div class='l'>Mots / avis (moy.)</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='v'>{(avg_len or 0):.1f}</div></div>", unsafe_allow_html=True)
    with k4:
        st.markdown("<div class='kpi'><div class='l'>Langues distinctes</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='v'>{df_f['language'].nunique() if len(df_f) else 0}</div></div>", unsafe_allow_html=True)

    # filtres locaux (avec keys uniques)
    l1,l2 = st.columns([1,1])
    with l1: syn_min_sent = st.slider("Seuil min. sentiment", -1.0, 1.0, -1.0, 0.05, key="syn_min_sent_slider")
    with l2: syn_max_sent = st.slider("Seuil max. sentiment", -1.0, 1.0, 1.0, 0.05, key="syn_max_sent_slider")
    df_syn = df_f[(df_f["sentiment"]>=syn_min_sent) & (df_f["sentiment"]<=syn_max_sent)]

    A,B,C = st.columns([1.5,1,1])
    with A:
        if df_syn["review_date"].notna().any() and len(df_syn):
            ts = df_syn.dropna(subset=["review_date"]).copy()
            ts["date"] = ts["review_date"].dt.to_period("W").dt.start_time
            series = ts.groupby("date")["sentiment"].mean()
            fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_WIDE)
            ax.plot(series.index, series.values, color=PRIMARY, linewidth=2)
            compact_time_axis(ax,3,5); title(ax,"Sentiment hebdo")
            ax.set_xlabel(""); ax.set_ylabel("Score")
            st.pyplot(fig, use_container_width=True)
            st.markdown("<div class='small'>Évolution lissée par semaine sur la période filtrée.</div>", unsafe_allow_html=True)
        else: st.info("Pas de dates exploitables.")
    with B:
        fig, ax = plt.subplots(figsize=(4,2.2))
        sns.barplot(y=freq_df["Thème"].head(5), x=freq_df["Fréquence (%)"].head(5), color="#60a5fa", ax=ax)
        title(ax,"Thèmes dominants"); st.pyplot(fig, use_container_width=True)
        st.markdown("<div class='small'>Part d’avis citant explicitement chaque thème.</div>", unsafe_allow_html=True)
    with C:
        cats=["Positif","Neutre","Négatif"]; vals=[
            (df_syn["sentiment"]>0.05).mean()*100 if len(df_syn) else 0,
            ((df_syn["sentiment"]>=-0.05)&(df_syn["sentiment"]<=0.05)).mean()*100 if len(df_syn) else 0,
            (df_syn["sentiment"]<-0.05).mean()*100 if len(df_syn) else 0
        ]
        fig, ax = plt.subplots(figsize=(4,2.2))
        ax.bar(cats, vals, color=[GOOD,GREY,BAD]); ax.set_ylim(0,100); title(ax,"Répartition des avis")
        st.pyplot(fig, use_container_width=True)
        st.markdown("<div class='small'>Équilibre global des ressentis sur la période.</div>", unsafe_allow_html=True)

    # Top mots (visible direct)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    c_top_left, c_top_right = st.columns([1,1])
    with c_top_left:
        if len(df_syn):
            STOP = set("""the and for you are with that this was but not have your plus tres très les des une que pour est pas qui dans sur par au aux du de la le un et ou mais donc car game jeu games steam play jouer joue joué""".split())
            text_all=" ".join(df_syn["cleaned_review"].dropna()); words=[w for w in text_all.split() if len(w)>2 and w not in STOP]
            top_words_df=pd.DataFrame(Counter(words).most_common(15), columns=["Mot","Fréquence"])
            st.markdown("**Top mots (période + filtres)**"); st.dataframe(top_words_df, use_container_width=True, height=250)
            st.markdown("<div class='small'>Servez‑vous de ces mots pour filtrer plus finement.</div>", unsafe_allow_html=True)
        else: st.info("Aucun mot significatif.")
    with c_top_right:
        st.markdown("**Filtres actifs**")
        st.markdown(f"- Période : **{st.session_state.date_min} → {st.session_state.date_max}**")
        if keywords_raw and keywords_raw.strip():
            st.markdown(f"- Mots‑clés : **{keywords_raw}** {'(ET)' if match_all else '(OU)'}")
        else: st.markdown("- Mots‑clés : *(aucun)*")
        st.markdown("<div class='small'>Rappel utile pour les lecteurs non analystes.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --------- SENTIMENT ----------
with tab_sent:
    st.markdown("<div class='card'><h3>Perception globale</h3>", unsafe_allow_html=True)
    s1,s2 = st.columns([1,1])
    with s1: sent_min = st.slider("Seuil min. sentiment", -1.0, 1.0, -1.0, 0.05, key="sent_min_slider")
    with s2: sent_max = st.slider("Seuil max. sentiment", -1.0, 1.0, 1.0, 0.05, key="sent_max_slider")
    df_sent = df_f[(df_f["sentiment"]>=sent_min) & (df_f["sentiment"]<=sent_max)]

    c1,c2,c3 = st.columns([1.4,1,1])
    with c1:
        if df_sent["review_date"].notna().any() and len(df_sent):
            ts=df_sent.dropna(subset=["review_date"]).copy()
            ts["date"]=ts["review_date"].dt.to_period("W").dt.start_time
            series=ts.groupby("date")["sentiment"].mean().rolling(3).mean()
            fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_WIDE)
            ax.plot(series.index, series.values, color=PRIMARY)
            compact_time_axis(ax,3,6); title(ax,"Tendance (MM x3)")
            ax.set_ylabel("Score"); ax.set_xlabel("")
            st.pyplot(fig, use_container_width=True)
            st.markdown("<div class='small'>Lissage léger pour mieux lire la tendance.</div>", unsafe_allow_html=True)
        else: st.info("Pas de dates exploitables.")
    with c2:
        dist_df=pd.DataFrame({"cat":["Positif","Neutre","Négatif"],"val":[
            (df_sent["sentiment"]>0.05).mean()*100 if len(df_sent) else 0,
            ((df_sent["sentiment"]>=-0.05)&(df_sent["sentiment"]<=0.05)).mean()*100 if len(df_sent) else 0,
            (df_sent["sentiment"]<-0.05).mean()*100 if len(df_sent) else 0]})
        fig, ax = plt.subplots(figsize=(4,2.2))
        sns.barplot(data=dist_df, x="cat", y="val", hue="cat",
                    palette={"Positif":GOOD,"Neutre":GREY,"Négatif":BAD}, legend=False, ax=ax)
        title(ax,"Répartition"); ax.set_ylabel("%"); ax.set_xlabel("")
        st.pyplot(fig, use_container_width=True)
        st.markdown("<div class='small'>Équilibre global des ressentis.</div>", unsafe_allow_html=True)
    with c3:
        if len(df_sent):
            fig, ax = plt.subplots(figsize=(4,2.2))
            sns.histplot(df_sent["sentiment"], bins=30, kde=True, ax=ax, color="#93c5fd")
            title(ax,"Distribution des scores"); st.pyplot(fig, use_container_width=True)
            st.markdown("<div class='small'>Neutre ou polarisé ?</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --------- THEMES (enrichi) ----------
with tab_topics:
    st.markdown("<div class='card'><h3>Analyse par thème</h3>", unsafe_allow_html=True)

    tc1, tc2, tc3 = st.columns([1,1,1])
    with tc1: theme_choice = st.selectbox("Thème ciblé", options=list(theme_dict.keys()), key="topics_theme_choice")
    with tc2: tmin = st.slider("Sentiment min.", -1.0, 1.0, -1.0, 0.05, key="topic_min_slider")
    with tc3: tmax = st.slider("Sentiment max.", -1.0, 1.0, 1.0, 0.05, key="topic_max_slider")

    df_theme = df_f[df_f["cleaned_review"].apply(lambda t: contains_any(t, theme_dict[theme_choice]))].copy()
    df_theme = df_theme[(df_theme["sentiment"]>=tmin) & (df_theme["sentiment"]<=tmax)]

    if df_theme.empty:
        st.info("Aucun avis pour ce thème avec les filtres actuels."); st.markdown("</div>", unsafe_allow_html=True)
    else:
        k1,k2,k3,k4 = st.columns(4)
        with k1:
            st.markdown("<div class='kpi'><div class='l'>Avis (thème)</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='v'>{len(df_theme):,}</div></div>", unsafe_allow_html=True)
        with k2:
            st.markdown("<div class='kpi'><div class='l'>Sentiment moyen</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='v'>{df_theme['sentiment'].mean():.2f}</div></div>", unsafe_allow_html=True)
        with k3:
            p=(df_theme["sentiment"]>0.05).mean()*100; n=((df_theme["sentiment"]>=-0.05)&(df_theme["sentiment"]<=0.05)).mean()*100; g=(df_theme["sentiment"]<-0.05).mean()*100
            st.markdown("<div class='kpi'><div class='l'>% Pos / Neu / Neg</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='v'>{p:.0f}% / {n:.0f}% / {g:.0f}%</div></div>", unsafe_allow_html=True)
        with k4:
            st.markdown("<div class='kpi'><div class='l'>Mots / avis (moy.)</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='v'>{df_theme['cleaned_review'].str.split().apply(len).replace(0,np.nan).mean():.1f}</div></div>", unsafe_allow_html=True)

        r1,r2 = st.columns([1.4, 1])
        with r1:
            if df_theme["review_date"].notna().any():
                ts = df_theme.dropna(subset=["review_date"]).copy()
                ts["date"] = ts["review_date"].dt.to_period("W").dt.start_time
                series = ts.groupby("date")["sentiment"].mean()
                fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_WIDE)
                ax.plot(series.index, series.values, color=PRIMARY, linewidth=2)
                compact_time_axis(ax,3,6); title(ax, f"Tendance hebdo – {theme_choice}")
                ax.set_xlabel(""); ax.set_ylabel("Score")
                st.pyplot(fig, use_container_width=True)
                st.markdown("<div class='small'>Évolution du ressenti ; utile pour mesurer l’effet des patchs.</div>", unsafe_allow_html=True)
            else: st.info("Pas de dates exploitables pour la tendance de ce thème.")
        with r2:
            fig, ax = plt.subplots(figsize=(4.2,2.2))
            sns.histplot(df_theme["sentiment"], bins=24, kde=True, ax=ax, color="#60a5fa")
            title(ax,"Distribution des scores"); st.pyplot(fig, use_container_width=True)
            st.markdown("<div class='small'>Retour plutôt mixte ou polarisé ?</div>", unsafe_allow_html=True)

        uni, big = top_unigrams_bigrams(df_theme["cleaned_review"].dropna().tolist(), n_top=15)
        c1,c2 = st.columns([1,1])
        with c1:
            st.markdown("**Top mots – thème**")
            st.dataframe(pd.DataFrame(uni, columns=["Mot","Fréquence"]), use_container_width=True, height=220)
            st.markdown("<div class='small'>Vocabulaire le plus associé à ce thème.</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("**Top expressions (bigrams)**")
            st.dataframe(pd.DataFrame(big, columns=["Bigramme","Fréquence"]), use_container_width=True, height=220)
            st.markdown("<div class='small'>Expressions fréquentes (ex. “crash serveur”, “drop fps”).</div>", unsafe_allow_html=True)

        pos_ex, neg_ex = pick_examples(df_theme, n=3)
        e1,e2 = st.columns([1,1])
        with e1:
            st.markdown("**Exemples positifs (résumé)**")
            for t in pos_ex: st.write(f"• {t}")
            st.markdown("<div class='small'>Ce que les joueurs apprécient sur ce thème.</div>", unsafe_allow_html=True)
        with e2:
            st.markdown("**Exemples négatifs (résumé)**")
            for t in neg_ex: st.write(f"• {t}")
            st.markdown("<div class='small'>Douleurs récurrentes à prioriser.</div>", unsafe_allow_html=True)

        if st.checkbox("Afficher le nuage de mots (thème)", key="topics_wordcloud"):
            text=" ".join(df_theme["cleaned_review"].dropna().tolist())
            if text.strip():
                wc=WordCloud(width=900, height=300, background_color="white").generate(text)
                fig, ax = plt.subplots(figsize=(6.5,2.4)); ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
                st.pyplot(fig, use_container_width=True)
                st.markdown("<div class='small'>Nuage indicatif ; se référer aux tableaux pour le détail.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# --------- LANGUES ----------
with tab_lang:
    st.markdown("<div class='card'><h3>Langues</h3>", unsafe_allow_html=True)
    if len(df_f):
        lang_agg = df_f.groupby("language").agg(avis=("review_text","count"), sentiment_moy=("sentiment","mean")).reset_index().sort_values("avis", ascending=False)
        top = lang_agg.head(10)
        c1,c2 = st.columns([1,1])
        with c1:
            fig, ax = plt.subplots(figsize=(5,2.2))
            ax.barh(top["language"], top["avis"], color="#93c5fd"); ax.invert_yaxis()
            title(ax,"Volume (Top 10)"); st.pyplot(fig, use_container_width=True)
            st.markdown("<div class='small'>Langues les plus actives.</div>", unsafe_allow_html=True)
        with c2:
            fig, ax = plt.subplots(figsize=(5,2.2))
            ax.barh(top["language"], top["sentiment_moy"], color="#60a5fa"); ax.invert_yaxis()
            title(ax,"Sentiment moyen (Top 10)"); st.pyplot(fig, use_container_width=True)
            st.markdown("<div class='small'>Comparer satisfaction par langue.</div>", unsafe_allow_html=True)
    else: st.info("Aucune donnée filtrée.")
    st.markdown("</div>", unsafe_allow_html=True)

# --------- HEURES DE JEU ----------
def segment_playtime(h):
    try: h=float(h)
    except: return "Inconnu"
    if h<10: return "Occasionnel (<10h)"
    if h<100: return "Régulier (10–100h)"
    return "Intensif (≥100h)"

with tab_play:
    st.markdown("<div class='card'><h3>Profils par heures de jeu</h3>", unsafe_allow_html=True)
    if "playtime_hours" in df_f.columns and len(df_f):
        min_h=float(np.nanmin(df_f["playtime_hours"])) if len(df_f) else 0.0
        max_h=float(np.nanmax(df_f["playtime_hours"])) if len(df_f) else 0.0
        pmin,pmax=st.slider("Plage d'heures de jeu", float(min_h), float(max_h), (float(min_h), float(max_h)), 1.0, key="playtime_range_slider")
        df_p=df_f[(df_f["playtime_hours"]>=pmin)&(df_f["playtime_hours"]<=pmax)].copy()
        df_p["player_segment"]=df_p["playtime_hours"].apply(segment_playtime)

        seg=df_p.groupby("player_segment").agg(avis=("review_text","count"), sentiment_moy=("sentiment","mean")).reset_index().sort_values("avis", ascending=False)
        c1,c2=st.columns([1,1])
        with c1: st.dataframe(seg, use_container_width=True, height=220)
        with c2:
            fig, ax = plt.subplots(figsize=(5,2.2))
            ax.bar(seg["player_segment"], seg["sentiment_moy"], color=PRIMARY); ax.tick_params(axis='x', rotation=15)
            title(ax,"Sentiment par segment"); st.pyplot(fig, use_container_width=True)
            st.markdown("<div class='small'>Cibler des actions par profils (onboarding vs end‑game).</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10,2.2))
        ax.scatter(df_p["playtime_hours"], df_p["sentiment"], s=8, alpha=0.35, color=PRIMARY)
        title(ax,"Sentiment vs Heures"); ax.set_xlabel("Heures"); ax.set_ylabel("Score")
        st.pyplot(fig, use_container_width=True)
        st.markdown("<div class='small'>Relation expérience ↔ ressenti.</div>", unsafe_allow_html=True)
    else: st.info("Heures de jeu non disponibles.")
    st.markdown("</div>", unsafe_allow_html=True)

# --------- LONGUEUR ----------
with tab_len:
    st.markdown("<div class='card'><h3>Longueur des avis & ressenti</h3>", unsafe_allow_html=True)
    if len(df_f):
        df_f["word_count"]=df_f["cleaned_review"].str.split().apply(len)
        wc_max=int(np.nanpercentile(df_f["word_count"],99)) if len(df_f) else 500
        wc_cut=st.slider("Max mots/avis (trim)", 20, max(40, wc_max), wc_max, key="wc_cut_slider")
        df_l=df_f[df_f["word_count"]<=wc_cut]
        c1,c2=st.columns([1,1])
        with c1:
            fig, ax = plt.subplots(figsize=(5,2.2))
            ax.hist(df_l["word_count"], bins=40, color="#93c5fd")
            title(ax,"Distribution longueur d’avis"); st.pyplot(fig, use_container_width=True)
            st.markdown("<div class='small'>Répartition des tailles d’avis (trim 99e pct).</div>", unsafe_allow_html=True)
        with c2:
            fig, ax = plt.subplots(figsize=(5,2.2))
            ax.scatter(df_l["word_count"], df_l["sentiment"], s=8, alpha=0.35, color=PRIMARY)
            title(ax,"Sentiment vs longueur"); ax.set_xlabel("Mots/avis"); ax.set_ylabel("Score")
            st.pyplot(fig, use_container_width=True)
            st.markdown("<div class='small'>Les avis longs sont-ils plus critiques ?</div>", unsafe_allow_html=True)
    else: st.info("Aucune donnée filtrée.")
    st.markdown("</div>", unsafe_allow_html=True)

# --------- COOCCURRENCES ----------
with tab_cooc:
    st.markdown("<div class='card'><h3>Co‑mentions de thèmes</h3>", unsafe_allow_html=True)
    themes_all=list(theme_dict.keys()); chosen_themes=st.multiselect("Thèmes à cartographier", options=themes_all, default=themes_all, key="cooc_themes")
    if len(df_f) and chosen_themes:
        M=pd.DataFrame(0, index=chosen_themes, columns=chosen_themes, dtype=int)
        for _,row in df_f.iterrows():
            text=row.get("cleaned_review",""); present=[th for th in chosen_themes if contains_any(text, theme_dict[th])]
            for i in range(len(present)):
                for j in range(i, len(present)):
                    M.loc[present[i],present[j]]+=1
                    if i!=j: M.loc[present[j],present[i]]+=1
        fig, ax = plt.subplots(figsize=(6,3.0))
        sns.heatmap(M, cmap="Blues", cbar=False, ax=ax, annot=False)
        title(ax,"Carte des cooccurrences"); st.pyplot(fig, use_container_width=True)
        st.markdown("<div class='small'>Repère les paquets de sujets récurrents (ex. Bugs + Performances).</div>", unsafe_allow_html=True)
    else: st.info("Aucune donnée/thème sélectionné.")
    st.markdown("</div>", unsafe_allow_html=True)

# --------- ANOMALIES ----------
with tab_anom:
    st.markdown("<div class='card'><h3>Anomalies hebdomadaires</h3>", unsafe_allow_html=True)
    if df_f["review_date"].notna().any() and len(df_f):
        z_thr=st.slider("Seuil z-score (anomalie)", 1.0, 3.0, 2.0, 0.1, key="anom_z_slider")
        ts=df_f.dropna(subset=["review_date"]).copy()
        ts["week"]=ts["review_date"].dt.to_period("W").dt.start_time
        agg=ts.groupby("week").agg(sent=("sentiment","mean"), n=("review_text","count")).reset_index()
        if len(agg)>=3:
            agg["z_sent"]=(agg["sent"]-agg["sent"].mean())/agg["sent"].std(ddof=0)
            agg["z_n"]=(agg["n"]-agg["n"].mean())/agg["n"].std(ddof=0)
            anomalies=agg[(agg["z_sent"].abs()>z_thr)|(agg["z_n"].abs()>z_thr)]
        else: anomalies=pd.DataFrame(columns=agg.columns)
        c1,c2=st.columns([1,1])
        with c1:
            fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_WIDE)
            ax.plot(agg["week"], agg["sent"], color=PRIMARY)
            if not anomalies.empty: ax.scatter(anomalies["week"], anomalies["sent"], color=BAD, s=20)
            compact_time_axis(ax,3,6); title(ax,"Satisfaction (hebdo)")
            st.pyplot(fig, use_container_width=True)
        with c2:
            fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_WIDE)
            ax.plot(agg["week"], agg["n"], color=PRIMARY)
            if not anomalies.empty: ax.scatter(anomalies["week"], anomalies["n"], color=BAD, s=20)
            compact_time_axis(ax,3,6); title(ax,"Volume d’avis (hebdo)")
            st.pyplot(fig, use_container_width=True)
        st.markdown("<div class='small'>Points rouges = semaines atypiques (patch, promo, bad buzz…).</div>", unsafe_allow_html=True)
        if not anomalies.empty: st.dataframe(anomalies, use_container_width=True, height=220)
    else: st.info("Pas de dates exploitables.")
    st.markdown("</div>", unsafe_allow_html=True)

# --------- QUALITÉ ----------
with tab_qual:
    st.markdown("<div class='card'><h3>Qualité des données</h3>", unsafe_allow_html=True)
    if len(df_f):
        cols=["review_text","language","review_date"]
        comp={c:f"{df_f[c].notna().mean()*100:.1f}%" if c in df_f.columns else "absent" for c in cols}
        st.write("**Complétude (non-nulles)** :", comp)
        dup_rate=(df_f.duplicated(subset=["review_text","review_date"]).mean()*100) if {"review_text","review_date"}.issubset(df_f.columns) else 0.0
        st.write(f"**Duplicats estimés** : {dup_rate:.1f}%")
        st.markdown("<div class='small'>Une bonne complétude et peu de doublons = indicateurs fiables.</div>", unsafe_allow_html=True)
    else: st.info("Aucune donnée filtrée.")
    st.markdown("</div>", unsafe_allow_html=True)

# --------- EXPLORATEUR ----------
with tab_table:
    st.markdown("<div class='card'><h3>Explorateur d’avis</h3>", unsafe_allow_html=True)
    view_cols=[c for c in ["review_date","language","sentiment","playtime_hours","review_text"] if c in df_f.columns]
    e1,e2=st.columns([2,1])
    with e1: local_kw=st.text_input("Filtrer par mots (explorateur)", value="", key="explore_kw")
    with e2: smin,smax=st.slider("Plage de sentiment", -1.0, 1.0, (-1.0,1.0), 0.05, key="explore_sent_range")
    df_view_base=df_f[(df_f["sentiment"]>=smin)&(df_f["sentiment"]<=smax)].copy()
    if local_kw.strip():
        pattern=r"\b(" + "|".join([re.escape(k.strip()) for k in local_kw.split(",") if k.strip()]) + r")\b"
        df_view_base=df_view_base[df_view_base["cleaned_review"].str.contains(pattern, regex=True, na=False)]
    def highlight_keywords(text,kws):
        if not kws or not isinstance(text,str): return text
        pattern=r"(" + "|".join([re.escape(k) for k in kws]) + r")"; return re.sub(pattern,r"<mark>\1</mark>", text, flags=re.IGNORECASE)
    if view_cols and len(df_view_base):
        df_view=df_view_base[view_cols].copy()
        if local_kw.strip():
            kws=[k.strip() for k in local_kw.split(",") if k.strip()]
            df_view["review_text"]=df_view["review_text"].astype(str).apply(lambda t: highlight_keywords(t,kws))
            st.write("Les mots saisis sont **surlignés** dans les avis.")
        st.write(df_view.to_html(escape=False, index=False), unsafe_allow_html=True)
        csv=df_view_base[view_cols].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export CSV (avis filtrés)", data=csv, file_name="avis_filtrés.csv", mime="text/csv", key="explore_download")
    else: st.info("Aucune donnée à afficher.")
    st.markdown("</div>", unsafe_allow_html=True)

# --------- MISES À JOUR ----------
with tab_updates:
    st.markdown("<div class='card'><h3>Impact d’une mise à jour</h3>", unsafe_allow_html=True)
    if df_f["review_date"].notna().any() and len(df_f):
        df_f_min=max(st.session_state.date_min, df_f["review_date"].min().date())
        df_f_max=min(st.session_state.date_max, df_f["review_date"].max().date())
        pivot=st.date_input("Date pivot", value=df_f_min, min_value=df_f_min, max_value=df_f_max, key="pivot_input")
        before=df_f[df_f["review_date"].dt.date < pivot]["sentiment"].mean()
        after =df_f[df_f["review_date"].dt.date >= pivot]["sentiment"].mean()
        var=(after-before) if (pd.notna(before) and pd.notna(after)) else np.nan
        c1,c2,c3=st.columns(3)
        c1.markdown(f"<div class='kpi'><div class='l'>Avant</div><div class='v'>{before if pd.notna(before) else 0:.2f}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='kpi'><div class='l'>Après</div><div class='v'>{after if pd.notna(after) else 0:.2f}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='kpi'><div class='l'>Variation</div><div class='v'>{var if pd.notna(var) else 0:+.2f}</div></div>", unsafe_allow_html=True)
        tdf=df_f.dropna(subset=["review_date"]).copy(); tdf["period"]=np.where(tdf["review_date"].dt.date < pivot, "Avant", "Après")
        fig, ax = plt.subplots(figsize=(5,2.2))
        sns.boxplot(data=tdf, x="period", y="sentiment", hue="period",
                    palette={"Avant":"#fde68a","Après":"#93c5fd"}, legend=False, ax=ax)
        title(ax,"Avant / Après (sentiment)"); ax.set_xlabel("")
        st.pyplot(fig, use_container_width=True)
        st.markdown("<div class='small'>Comparer rapidement l’effet d’un patch ou d’une mise à jour majeure.</div>", unsafe_allow_html=True)
    else: st.info("Pas de dates exploitables.")
    st.markdown("</div>", unsafe_allow_html=True)
