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

# Styles CSS pour personnalisation
st.markdown(
    """
    <style>
    body {
        background-color: #f9f9f9;
        color: #333333;
    }
    .sidebar .sidebar-content {
        background-color: #f4f4f4;
    }
    h1 {
        color: #003566;
    }
    .stButton>button {
        background-color: #003566;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #00509e;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Charger les données
file_path = "Amazon_Reviews.csv"
df = pd.read_csv(file_path, encoding="utf-8", engine="python", on_bad_lines="skip", delimiter=",")

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

# Fonction de prétraitement
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalpha() and word not in stop_words]


# Si Aperçu des données
if menu == "Aperçu des données":
    st.header("📄 Aperçu des Données")
    st.write("Voici un aperçu des données chargées :")
    st.dataframe(df.head(20))

# Si Analyse des mots fréquents
elif menu == "Analyse des mots fréquents":
    st.header("🔍 Analyse des Mots Fréquents")
    review_texts = df["Review Text"].dropna()
    all_tokens = [token for text in review_texts for token in preprocess_text(text)]
    most_common_words = Counter(all_tokens).most_common(20)
    most_common_df = pd.DataFrame(most_common_words, columns=["Word", "Frequency"])
    st.subheader("Top 20 des mots les plus fréquents")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x="Frequency", y="Word", data=most_common_df.sort_values(by="Frequency", ascending=False), ax=ax)
    ax.set_title("Mots les plus fréquents")
    st.pyplot(fig)

# Si Distribution des notes
elif menu == "Distribution des notes":
    st.header("📊 Distribution des Notes")
    df["Rating"] = pd.to_numeric(df["Rating"].str.extract("(\d+)")[0], errors="coerce")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["Rating"], bins=5, kde=True, ax=ax)
    ax.set_title("Distribution des notes")
    st.pyplot(fig)

# Analyse des sentiments avec des couleurs pour l'histogramme
elif menu == "Analyse des sentiments":
    st.header("😊 Analyse des Sentiments")
    df = df.dropna(subset=["Review Text"])
    df["Sentiment"] = df["Review Text"].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Histogramme avec des couleurs basées sur la polarité
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(
        df["Sentiment"],
        kde=True,
        bins=30,
        palette="coolwarm",  # Palette de couleurs dégradées
        ax=ax
    )
    ax.set_title("Distribution des sentiments", fontsize=16)
    ax.set_xlabel("Polarité", fontsize=14)
    ax.set_ylabel("Fréquence", fontsize=14)
    st.pyplot(fig)

    # Catégorisation des sentiments avec des couleurs
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

    # Compte des catégories avec des couleurs
    custom_palette = {
        "Positif": "green",
        "Négatif": "red",
        "Neutre": "gray"
    }
    sns.countplot(
        x="Sentiment_Category",
        data=df,
        palette=custom_palette,  # Palette personnalisée
        ax=ax
    )

    ax.set_title("Répartition des catégories de sentiments", fontsize=16)
    ax.set_xlabel("Catégorie", fontsize=14)
    ax.set_ylabel("Nombre d'avis", fontsize=14)
    st.pyplot(fig)


# Si Nuage de mots
elif menu == "Nuage de mots":
    st.header("☁️ Nuage de Mots")
    review_texts = df["Review Text"].dropna()
    all_tokens = [token for text in review_texts for token in preprocess_text(text)]
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(all_tokens))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Nuage de mots des avis", fontsize=16)
    st.pyplot(fig)


# Si Évolution des sentiments
elif menu == "Évolution des sentiments":
    # Charger le fichier analysé
    file_analyzed_path = "Amazon_Reviews_Analyzed.csv"

    try:
        df = pd.read_csv(file_analyzed_path, encoding="utf-8", engine="python", on_bad_lines="skip", delimiter=",")
        # st.success(f"Fichier {file_analyzed_path} chargé avec succès.")
    except FileNotFoundError:
        st.error(f"Le fichier {file_analyzed_path} est introuvable. Vérifiez que le fichier analysé existe dans le répertoire.")
        st.stop()  # Arrête l'exécution si le fichier est introuvable
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
        st.stop()  # Arrête l'exécution en cas d'erreur

    st.header("📈 Évolution des Sentiments")
    # Conversion des dates et agrégation mensuelle
    df["Review Date"] = pd.to_datetime(df["Review Date"], errors="coerce")
    df["Month"] = df["Review Date"].dt.to_period("M")
    monthly_sentiments = df.groupby("Month")["Sentiment"].mean()

    # Création du graphique avec matplotlib pour personnalisation
    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_sentiments.plot(kind="line", marker="o", ax=ax, color="blue")
    ax.set_title("Évolution des Sentiments au Fil du Temps", fontsize=16)
    ax.set_xlabel("Mois", fontsize=14)
    ax.set_ylabel("Sentiment Moyen", fontsize=14)
    ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=45, fontsize=12)  # Rotation des labels pour plus de lisibilité
    st.pyplot(fig)

