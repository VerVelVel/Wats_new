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
import time

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
    df['message_channel'] = df['clean_text'] + ' ' + df['Source'].copy()
    return df


API_URL = "https://api-inference.huggingface.co/models/MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
headers = {"Authorization": ""}

def query(payload, max_retries=3, retry_delay=5):
    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Request failed with status code {response.status_code}. Retrying...")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}. Retrying...")
        
        retries += 1
        time.sleep(retry_delay)
    
    return {"error": f"Failed to get response after {max_retries} retries"}


def classify_text(df):
    candidate_labels = ["технологии", "политика", "спорт", "экономика", "развлечения", "здоровье", "образование", "мода", "происшествия"]

    def classify_row(row):
        payload = {
            "inputs": row['message_channel'],
            "parameters": {"candidate_labels": candidate_labels}
        }
        result = query(payload)

        if 'scores' not in result:
            print("Result without scores:", result)
            return "error"

        highest_score_index = result['scores'].index(max(result['scores']))
        most_likely_label = result['labels'][highest_score_index]
        return most_likely_label

    with ThreadPoolExecutor() as executor:
        results = []
        for result in executor.map(classify_row, df.to_dict('records')):
            results.append(result)
            time.sleep(1)  # Добавление задержки в 1 секунду между запросами
        df['Category'] = results

    return df

# # # Инициализация модели для zero-shot классификации с кэшированием
# @lru_cache(maxsize=1)
# def init_classifier():
#     # Проверка доступности CUDA до инициализации модели
#     device = 0 if torch.cuda.is_available() else -1
#     if device == -1:
#         print("CUDA is not available. Using CPU instead.")
#     classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0", device=device)
#     return classifier

# # Функция для классификации текста
# def classify_text_1(df, classifier):
#     candidate_labels = ["технологии", "политика", "спорт", "экономика", "развлечения", "здоровье", "образование", "мода", "происшествия"]
    
#     def classify_row(row):
#         result = classifier(row['message_channel'], candidate_labels)
#         highest_score_index = result['scores'].index(max(result['scores']))
#         most_likely_label = result['labels'][highest_score_index]
#         return most_likely_label

#     with ThreadPoolExecutor() as executor:
#         df['Category'] = list(executor.map(classify_row, df.to_dict('records')))

#     return df

# def query(payload, max_retries=5, retry_delay=5):
#     retries = 0
#     while retries < max_retries:
#         try:
#             response = requests.post(API_URL, headers=headers, json=payload)
#             if response.status_code == 200:
#                 return response.json()
#             else:
#                 print(f"Request failed with status code {response.status_code}. Retrying...")
#         except requests.exceptions.RequestException as e:
#             print(f"Request failed: {str(e)}. Retrying...")
        
#         retries += 1
#         time.sleep(retry_delay)
    
#     return {"error": f"Failed to get response after {max_retries} retries"}

# def classify_text(df):
#     candidate_labels = ["технологии", "политика", "спорт", "экономика", "развлечения", "здоровье", "образование", "мода", "происшествия"]

#     def classify_row(row):
#         payload = {
#             "inputs": row['message_channel'],
#             "parameters": {"candidate_labels": candidate_labels}
#         }
#         result = query(payload)

#         if 'scores' not in result:
#             print("Result without scores:", result)
#             return "error"

#         highest_score_index = result['scores'].index(max(result['scores']))
#         most_likely_label = result['labels'][highest_score_index]
#         return most_likely_label

#     with ThreadPoolExecutor() as executor:
#         results = []
#         for result in executor.map(classify_row, df.to_dict('records')):
#             if result == "error":
#                 print("Switching to classify_text_1 due to repeated errors")
#                 classifier = init_classifier()
#                 return classify_text_1(df, classifier)
#             results.append(result)
#             time.sleep(1)  # Добавление задержки в 1 секунду между запросами
#         df['Category'] = results

#     return df

# @lru_cache(maxsize=1)
# def init_classifier():
#     device = 0 if torch.cuda.is_available() else -1
#     if device == -1:
#         print("CUDA is not available. Using CPU instead.")
#     classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0", device=device)
#     return classifier

# def classify_text_1(df, classifier):
#     candidate_labels = ["технологии", "политика", "спорт", "экономика", "развлечения", "здоровье", "образование", "мода", "происшествия"]
    
#     def classify_row(row):
#         result = classifier(row['message_channel'], candidate_labels)
#         highest_score_index = result['scores'].index(max(result['scores']))
#         most_likely_label = result['labels'][highest_score_index]
#         return most_likely_label

#     with ThreadPoolExecutor() as executor:
#         df['Category'] = list(executor.map(classify_row, df.to_dict('records')))

#     return df