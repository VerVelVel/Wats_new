import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
from transformers import pipeline
from datasets import Dataset
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import requests

# Загрузка данных и предобработка
def preprocess_data(df):
    df = df.dropna(subset=['Content'])
    stop_words = set(stopwords.words('english')).union(set(stopwords.words('russian')))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        text = text.lower()
        text = re.sub('<.*?>', '', text)
        text = re.sub(r'[^\w\s,]', '', text)
        text = ''.join([c for c in text if c not in string.punctuation])
        text = re.sub(r'\d+', '<NUM>', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '<URL>', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '<USER>', text)
        text = ' '.join([word for word in text.split() if word not in stop_words])
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df.loc[:, 'clean_text'] = df['Content'].apply(preprocess_text)
    df.loc[:, 'message_channel'] = df['clean_text'] + ' ' + df['Source']
    return df

# Инициализация модели для zero-shot классификации с кэшированием
# @lru_cache(maxsize=1)
# def init_classifier():
#     # Проверка доступности CUDA до инициализации модели
#     device = 0 if torch.cuda.is_available() else -1
#     if device == -1:
#         print("CUDA is not available. Using CPU instead.")
#     classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0", device=device)
#     return classifier


API_URL = "https://api-inference.huggingface.co/models/MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def classify_text(df, classifier):
    candidate_labels = ["технологии", "политика", "спорт", "экономика", "развлечения", "здоровье", "образование", "мода", "происшествия"]

    def classify_row(row):
        payload = {
            "inputs": row['message_channel'],
            "parameters": {"candidate_labels": candidate_labels}
        }
        result = query(payload)
        highest_score_index = result['scores'].index(max(result['scores']))
        most_likely_label = result['labels'][highest_score_index]
        return most_likely_label
    
    with ThreadPoolExecutor() as executor:
        df['Category'] = list(executor.map(classify_row, df.to_dict('records')))

    return df

