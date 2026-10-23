# -----------------------------------------------------------------------------
# 1. Imports
# -----------------------------------------------------------------------------
import re
from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from nltk.stem import PorterStemmer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# 2. App configuration and shared constants
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Climate Change Twitter Analytics", page_icon="🌍", layout="wide")
px.defaults.template = "plotly_white"

LABEL_MAP = {-1: "Anti", 0: "Neutral", 1: "Pro", 2: "News"}
LABEL_ORDER = ["Pro", "Anti", "Neutral", "News"]
PALETTE = {"Pro": "#2f8f5b", "Anti": "#dc143c", "Neutral": "#dca91e", "News": "#4b87b8"}
TOPIC_COLORS = ["#f29b50", "#19a8b7", "#e75d8b", "#9cb94a", "#9064c7"]
stemmer = PorterStemmer()
stop_words = set(ENGLISH_STOP_WORDS) - {"no", "nor", "not", "never"}


# -----------------------------------------------------------------------------
# 3. Text preprocessing
# -----------------------------------------------------------------------------
def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\brt\b", " ", text)
    text = text.replace("#", "")
    words = re.sub(r"[^a-z\s]", " ", text).split()
    return " ".join(stemmer.stem(w) for w in words if w not in stop_words and len(w) > 2)


# -----------------------------------------------------------------------------
# 4. Dataset preparation, model training, evaluation, and topic modelling
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def analyze_csv(file_bytes):
    df = pd.read_csv(BytesIO(file_bytes))
    needed = {"sentiment", "message"}
    if not needed.issubset(df.columns):
        raise ValueError("The CSV must contain 'sentiment' and 'message' columns.")
    df = df.dropna(subset=["sentiment", "message"]).copy()
    df["category"] = pd.to_numeric(df["sentiment"], errors="coerce").map(LABEL_MAP)
    df = df.dropna(subset=["category"]).copy()

    # Remove messages whose copies carry conflicting source labels, then deduplicate.
    conflict_messages = df.groupby("message")["category"].nunique()
    conflict_messages = conflict_messages[conflict_messages > 1].index
    conflicts_removed = len(conflict_messages)
    df = df[~df["message"].isin(conflict_messages)].drop_duplicates("message").copy()
    df["clean_message"] = df["message"].map(clean_tweet)
    df = df[df["clean_message"].str.strip().ne("")].copy()
    if len(df) < 100:
        raise ValueError("Too few rows remain after cleaning. Please check the dataset.")

    # Split labeled examples; keep label proportions similar in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"], df["category"], test_size=0.2, random_state=42,
        stratify=df["category"]
    )
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    cm = confusion_matrix(y_test, predictions, labels=LABEL_ORDER)

    # LDA is a separate, unsupervised analysis of discussion themes.
    count_vectorizer = CountVectorizer(
        max_df=0.90, min_df=10, max_features=3000, ngram_range=(1, 2),
        stop_words=["climat", "chang", "amp"]
    )
    X_counts = count_vectorizer.fit_transform(df["clean_message"])
    lda = LatentDirichletAllocation(
        n_components=5, random_state=42, max_iter=5,
        learning_method="online", batch_size=1024
    )
    doc_topics = lda.fit_transform(X_counts)
    df["topic"] = doc_topics.argmax(axis=1) + 1
    terms = count_vectorizer.get_feature_names_out()
    topic_words = {}
    for i, weights in enumerate(lda.components_):
        top_indices = weights.argsort()[-10:][::-1]
        topic_words[i + 1] = [terms[j] for j in top_indices]
    return df, accuracy, macro_f1, cm, topic_words, conflicts_removed


# -----------------------------------------------------------------------------
# 5. Dashboard styling and project introduction
# -----------------------------------------------------------------------------
st.markdown("""
<style>
.stApp {background:#f4f7fb;color:#17233a}
[data-testid="stMetric"] {background:#ffffff;border:1px solid #e5ebf3;padding:16px;border-radius:14px;color:#17233a !important}
[data-testid="stMetric"] *, [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {color:#17233a !important; opacity:1 !important}
[data-testid="stAlert"] {background:#e8f5e9;border:1px solid #a5d6a7}
[data-testid="stAlert"] * {color:#1b5e20 !important; opacity:1 !important}
.hero {padding:25px 30px;border-radius:18px;background:linear-gradient(120deg,#102a43,#087f83);color:white;margin-bottom:20px}
.hero p {color:#d7e9ef;margin-bottom:0}
</style>
<div class="hero"><div>TEXT &amp; SOCIAL MEDIA ANALYTICS · MICRO PROJECT</div>
<h1>Climate Change Opinions on Twitter</h1>
<p>Stance Classification and Discussion Theme Analysis</p>
<p><b>Project question:</b> What themes appear in climate-change tweets, and how do the dataset labels vary across those themes?</p></div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. Upload dataset and handle input/errors
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Start analysis")
    uploaded = st.file_uploader("Upload Kaggle CSV", type=["csv"])
    st.caption("Select twitter_sentiment_data.csv. The analysis runs locally on this computer.")

if uploaded is None:
    st.info("Upload the Kaggle dataset CSV file to get started.")
    st.markdown("**Dataset:** `twitter_sentiment_data.csv` — columns `sentiment`, `message`, `tweetid`.")
    st.stop()

try:
    with st.spinner("Cleaning tweets and training models. The first run may take 1–2 minutes..."):
        df, accuracy, macro_f1, cm, topic_words, conflicts_removed = analyze_csv(uploaded.getvalue())
except Exception as exc:
    st.error(f"The analysis could not be started: {exc}")
    st.stop()

# -----------------------------------------------------------------------------
# 7. Summary metrics
# -----------------------------------------------------------------------------
st.success(f"Analysis complete · {len(df):,} cleaned tweets · {conflicts_removed} conflicting messages removed")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Classifier accuracy", f"{accuracy:.1%}")
k2.metric("Macro F1-score", f"{macro_f1:.1%}")
k3.metric("Tweets analyzed", f"{len(df):,}")
k4.metric("LDA discussion themes", "5")

# -----------------------------------------------------------------------------
# 8. Dataset label and LDA topic distributions
# -----------------------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Dataset label distribution")
    label_counts = df["category"].value_counts().reindex(LABEL_ORDER, fill_value=0).rename_axis("Label").reset_index(name="Tweets")
    fig = px.pie(label_counts, names="Label", values="Tweets", hole=0,
                 color="Label", color_discrete_map=PALETTE)
    fig.update_traces(textposition="outside", texttemplate="%{percent:.1%}",
                      hovertemplate="%{label}: %{value:,} tweets (%{percent})<extra></extra>")
    fig.update_layout(height=370, margin=dict(t=10, b=20, l=15, r=15), legend_title="Dataset label")
    st.plotly_chart(fig, use_container_width=True, theme=None)
with right:
    st.subheader("Tweet share by LDA topic")
    topic_counts = df["topic"].value_counts().sort_index().rename_axis("Topic").reset_index(name="Tweets")
    topic_counts["Topic"] = topic_counts["Topic"].map(lambda x: f"Topic {x}")
    fig = px.pie(topic_counts, names="Topic", values="Tweets", hole=.58,
                 color="Topic", color_discrete_sequence=TOPIC_COLORS)
    fig.update_traces(textposition="outside", texttemplate="%{percent:.1%}",
                      hovertemplate="%{label}: %{value:,} tweets (%{percent})<extra></extra>")
    fig.update_layout(height=370, margin=dict(t=10, b=20, l=15, r=15), legend_title="LDA topic")
    st.plotly_chart(fig, use_container_width=True, theme=None)

# -----------------------------------------------------------------------------
# 9. Compare dataset labels within each LDA topic
# -----------------------------------------------------------------------------
st.subheader("Dataset label mix within each discussion theme")
mix = pd.crosstab(df["topic"], df["category"], normalize="index").reindex(columns=LABEL_ORDER, fill_value=0).mul(100).round(1)
mix.index = [f"Topic {i}" for i in mix.index]
mix_long = mix.rename_axis("Topic").reset_index().melt(id_vars="Topic", var_name="Label", value_name="Share (%)")
fig = px.bar(mix_long, x="Topic", y="Share (%)", color="Label", barmode="stack",
             text= "Share (%)", color_discrete_map=PALETTE, category_orders={"Label": LABEL_ORDER})
fig.update_traces(texttemplate="%{text:.0f}%", textposition="inside",
                  hovertemplate="%{x}<br>%{fullData.name}: %{y:.1f}%<extra></extra>")
fig.update_layout(height=430, yaxis_range=[0, 100], legend_title="Dataset label", margin=dict(t=15, b=15))
st.plotly_chart(fig, use_container_width=True, theme=None)

# -----------------------------------------------------------------------------
# 10. Topic sizes and classifier evaluation chart
# -----------------------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Number of tweets in each topic")
    fig = px.bar(topic_counts, x="Topic", y="Tweets", text="Tweets", color="Topic", color_discrete_sequence=TOPIC_COLORS)
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(showlegend=False, height=360, margin=dict(t=15, b=15), yaxis_title="Tweets assigned to topic")
    st.plotly_chart(fig, use_container_width=True, theme=None)
with right:
    st.subheader("Classifier confusion matrix")
    fig = px.imshow(cm, x=LABEL_ORDER, y=LABEL_ORDER, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predicted label", y="Actual label", color="Tweets"), aspect="auto")
    fig.update_layout(height=360, margin=dict(t=15, b=15))
    st.plotly_chart(fig, use_container_width=True, theme=None)

# -----------------------------------------------------------------------------
# 11. Interpret topic keywords and explain findings
# -----------------------------------------------------------------------------
st.subheader("What the five LDA topics discuss")
st.caption("LDA topic numbers are model-generated groups, not stance classes. Names/meaning are interpreted from frequent words.")
topic_titles = ["Global warming, causes & weather", "Climate policy & science debate",
                "Belief and skepticism", "Political debate about climate policy",
                "Global climate action & leadership"]
cols = st.columns(5)
for idx, col in enumerate(cols, start=1):
    with col:
        st.markdown(f"**Topic {idx}: {topic_titles[idx-1]}**")
        st.caption(", ".join(topic_words[idx]))

st.subheader("How to read the results")
st.markdown(f"""
- **Classification:** TF-IDF + Logistic Regression predicts the dataset-provided label (Pro, Anti, Neutral, or News). Accuracy is **{accuracy:.1%}** and macro-F1 is **{macro_f1:.1%}**.
- **Topic analysis:** LDA groups tweets by words that co-occur. It does **not** predict stance; the stacked chart compares supplied labels inside each discovered topic.
- **Important:** “News” is an informational label, not an opinion stance. Findings describe this dataset and its existing labels, not all public opinion.
""")
st.caption("Source: edqian, Twitter Climate Change Sentiment Dataset (Kaggle).")

