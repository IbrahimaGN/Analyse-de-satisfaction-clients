# Importation des bibliothèques
import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud

# Initialisation des téléchargements NLTK
nltk.download("punkt")
nltk.download("stopwords")

# Configuration de la page
st.set_page_config(
    page_title="Analyse de Satisfaction des Avis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Charger les données
file_path = "Amazon_Reviews.csv"
df = pd.read_csv(file_path, encoding="utf-8", engine="python", on_bad_lines="skip", delimiter=",")
df = df.dropna(subset=["Review Text"])  # Supprimer les lignes sans avis

# Calculer la colonne "Sentiment" une fois pour toutes
def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity

df["Sentiment"] = df["Review Text"].apply(analyze_sentiment)

# Nettoyage des dates
df["Review Date"] = pd.to_datetime(df["Review Date"], errors="coerce")
df = df.dropna(subset=["Review Date"])  # Supprimer les lignes sans date valide


# Sidebar pour le menu
menu = st.sidebar.radio(
    "Menu",
    options=[
        "Aperçu des données",
        "Analyse des mots fréquents",
        "Distribution des notes",
        "Analyse des sentiments",
        "Nuage de mots",
        "Évolution des sentiments",
    ],
)

# Prétraitement des avis
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalpha() and word not in stop_words]


# Menu : Aperçu des données
if menu == "Aperçu des données":
    st.header("📄 Aperçu des Données")
    st.dataframe(df.head())
    st.write("Résumé des informations :")
    st.text(df.info())

# Menu : Analyse des mots fréquents
elif menu == "Analyse des mots fréquents":
    st.header("🔍 Analyse des Mots Fréquents")
    review_texts = df["Review Text"]
    all_tokens = [token for text in review_texts for token in preprocess_text(text)]
    most_common_words = Counter(all_tokens).most_common(20)
    most_common_df = pd.DataFrame(most_common_words, columns=["Word", "Frequency"])
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x="Frequency", y="Word", data=most_common_df.sort_values(by="Frequency", ascending=False), ax=ax)
    ax.set_title("Top 20 des mots les plus fréquents")
    st.pyplot(fig)

# Menu : Distribution des notes
elif menu == "Distribution des notes":
    st.header("📊 Distribution des Notes")
    df["Rating"] = pd.to_numeric(df["Rating"].str.extract("(\d+)")[0], errors="coerce")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["Rating"], bins=5, kde=True, ax=ax)
    ax.set_title("Distribution des notes")
    st.pyplot(fig)

# Menu : Analyse des sentiments
elif menu == "Analyse des sentiments":
    st.header("😊 Analyse des Sentiments")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["Sentiment"], kde=True, bins=30, color="blue", ax=ax)
    ax.set_title("Distribution des sentiments")
    st.pyplot(fig)

    # Catégorisation des sentiments
    def categorize_sentiment(polarity):
        if polarity > 0:
            return "Positif"
        elif polarity < 0:
            return "Négatif"
        else:
            return "Neutre"

    df["Sentiment_Category"] = df["Sentiment"].apply(categorize_sentiment)
    st.subheader("Répartition des Sentiments")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x="Sentiment_Category", data=df, ax=ax)
    ax.set_title("Répartition des catégories de sentiments")
    st.pyplot(fig)

# Menu : Nuage de mots
elif menu == "Nuage de mots":
    st.header("☁️ Nuage de Mots")
    review_texts = df["Review Text"]
    all_tokens = [token for text in review_texts for token in preprocess_text(text)]
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(all_tokens))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Nuage de mots des avis", fontsize=16)
    st.pyplot(fig)

# Menu : Évolution des sentiments
elif menu == "Évolution des sentiments":
    st.header("📈 Évolution des Sentiments")
    df["Review Date"] = pd.to_datetime(df["Review Date"], errors="coerce")
    df["Month"] = df["Review Date"].dt.to_period("M")
    monthly_sentiments = df.groupby("Month")["Sentiment"].mean()
    st.line_chart(monthly_sentiments)
