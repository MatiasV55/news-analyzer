import os
import re
import json
import argparse
from collections import Counter
from typing import List, Dict, Tuple

import requests
import pandas as pd
import nltk
from dotenv import load_dotenv
from nltk import bigrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer

load_dotenv()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

train_data = pd.read_csv('train_data.csv')


def extract_features(words) -> Dict[str, bool]:
    return {bigram: True for bigram in bigrams(words)}


train_dataset = [(extract_features(word_tokenize(text.lower())), label)
                 for (text, label) in zip(train_data['texto'], train_data['categoria'])]
classifier = NaiveBayesClassifier.train(train_dataset)


def get_latest_news(country='ar', page_size=100) -> List[Dict]:
    """
    Obtiene las últimas noticias más recientes de un país a través de la API de NewsAPI.

    :param country: El código del país para las noticias.
    :param page_size: La cantidad de noticias a obtener.
    :return: Una lista de diccionarios con las noticias más recientes.
    """
    api_key = os.getenv('NEWS_API_KEY')
    if not api_key:
        raise ValueError("API key not found")

    url = 'https://newsapi.org/v2/top-headlines'
    params = {
        'apiKey': api_key,
        'country': country,
        'pageSize': page_size
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data['articles']
    except:
        raise RuntimeError("Error in request to NewsAPI.")


def get_google_news_ar() -> List[Dict]:
    """
    Obtiene las noticias principales de Google News Argentina a través de la API de NewsAPI.

    :return: Una lista de diccionarios con las noticias más recientes de Google News Argentina.
    """
    api_key = os.getenv('NEWS_API_KEY')
    if not api_key:
        raise ValueError("API key not found")
    url = f'https://newsapi.org/v2/top-headlines?sources=google-news-ar&apiKey={api_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['articles']
    except:
        raise RuntimeError("Error in request to Google News Argentina.")


def categorize_article(title: str, content: str, description: str) -> str:
    """
    Categoriza un artículo usando un clasificador de Naive Bayes.

    :param title: Título del artículo.
    :param content: Contenido del artículo.
    :param description: Descripción del artículo.
    :return: Categoría del artículo.
    """
    try:
        combined_text = (title or "") + " " + (content or "") + \
            " " + (description or "")
        features = extract_features(word_tokenize(combined_text.lower()))
        return classifier.classify(features)
    except:
        raise RuntimeError("Error during categorizing article")


def sentiment_analysis(title: str) -> str:
    """
    Realiza un análisis de sentimiento sobre un título de noticia.

    :param title: Título de la noticia.
    :return: Su respectivo sentimiento.
    """
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(title)
    sentiment = 'neutral'
    if score['compound'] >= 0.05:
        sentiment = 'positive'
    elif score['compound'] <= -0.05:
        sentiment = 'negative'
    return sentiment


def analyze_articles(articles, language="spanish") -> Tuple:
    """
    Analiza una lista de artículos para extraer información relevante,
    identificar palabras clave y categorizar los artículos en temas generales.

    :param articles: Lista de artículos donde cada artículo es un diccionario.
    :return: Una tupla con:
             - Información relevante de los artículos
             - Palabras clave más comunes
             - Categorías de los artículos
    """
    articles_info = []
    all_titles = []
    all_contents = []
    categories = []
    source_frequency = Counter()
    for article in articles:
        title = article['title']
        author = article.get('author', 'Desconocido')
        published_at = article['publishedAt']
        source = article['source']['name']
        content = article.get('content', '')

        category = categorize_article(title, content, article['description'])
        sentiment = sentiment_analysis(title)

        articles_info.append({
            'title': title,
            'author': author,
            'publishedAt': published_at,
            'source': source,
            'content': content,
            'category': category,
            'title_sentiment_analysis': sentiment
        })
        all_titles.append(title)
        all_contents.append(content)
        categories.append(category)
        source_frequency[source] += 1

    word_freq = Counter()
    stop_words = set(stopwords.words(language))
    for text in all_titles + all_contents:
        if text:
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            tokens = word_tokenize(text)
            clean_tokens = [word for word in tokens if word not in stop_words]
            word_freq.update(clean_tokens)

    most_common_keywords = word_freq.most_common(10)
    category_distribution = Counter(categories)
    return articles_info, most_common_keywords, categories, source_frequency, category_distribution


def generate_report(articles_info, most_common_keywords, source_frequency, category_distribution):
    """
    Genera un informe en formato JSON sobre el análisis realizado.

    :param articles_info: Información sobre los artículos analizados.
    :param most_common_keywords: Palabras clave más comunes.
    :param source_frequency: Frecuencia de publicación por fuente.
    :param category_distribution: Distribución de artículos por categoría.
    """
    category_counts = Counter(category_distribution)
    most_common_categories = category_counts.most_common(5)
    report = {
        'total_articles': len(articles_info),
        'most_common_keywords': most_common_keywords,
        'most_common_categories': {category: count for category, count in most_common_categories},
        'source_frequency': dict(source_frequency),
        'category_distribution': dict(category_distribution),
        'articles_info': articles_info
    }

    with open('report.json', 'w', encoding='utf-8') as json_file:
        json.dump(report, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--country', type=str, default='ar')
    args = parser.parse_args()

    if args.country not in ['ar', 'us']:
        raise ValueError(
            "The --country argument must be 'ar' for Argentina or 'us' for the United States.")

    try:
        news = get_latest_news(country=args.country, page_size=100)
    except RuntimeError as re:
        print(f"RuntimeError: {re}")
        exit(1)

    if not news:
        print(f"No recent articles found in '{args.country}'. Using support from Google News Argentina.")
        news = get_google_news_ar()

    try:
        if args.country == "us":
            articles_info, most_common_keywords, categories, source_frequency, category_distribution = analyze_articles(
                news, "english")
        else:
            articles_info, most_common_keywords, categories, source_frequency, category_distribution = analyze_articles(
                news)

        generate_report(articles_info, most_common_keywords,
                        source_frequency, category_distribution)
    except Exception as e:
        print(f"Error during report generation: {e}")
