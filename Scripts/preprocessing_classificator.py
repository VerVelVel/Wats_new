import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pickle
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from torch.nn.functional import softmax
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from functools import lru_cache
import time
import aiohttp
import requests
import asyncio
import math

# Предобработка текста
def preprocess_data(df):
    start_time = time.time()  
    df = df.dropna(subset=['Content'])
    stop_words = set(stopwords.words('english')).union(set(stopwords.words('russian')))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        text = text.lower()
        text = re.sub('<.*?>', '', text)
        text = re.sub(r'[^\w\s,]', '', text)
        text = ''.join([c for c in text if c not in string.punctuation])
        # text = re.sub(r'\d+', '<NUM>', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '<URL>', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '<USER>', text)
        text = ' '.join([word for word in text.split() if word not in stop_words])
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['clean_text'] = df['Content'].apply(preprocess_text)
    df['message_channel'] = df['clean_text'] + ' ' + df['Source']
    end_time = time.time()  # Засекаем время окончания выполнения функции
    print(f"TIME Preprocessing took {end_time - start_time} seconds") 
    return df


API_URL = "https://api-inference.huggingface.co/models/MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
headers = {"Authorization": "Bearer "}

candidate_labels = ["технологии", "политика", "спорт", "экономика", "развлечения", "здоровье", "образование", "мода", "инциденты и катастрофы"]

@lru_cache(maxsize=1)
def init_classifier():
    device = 0 if torch.cuda.is_available() else -1
    if device == -1:
        print("CUDA is not available. Using CPU instead.")
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0", device=device)
    return classifier

async def query_async(session, payload, retry_attempts=3, retry_delay=5, extended_delay=35):
    for attempt in range(retry_attempts):
        try:
            async with session.post(API_URL, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                elif response.status == 503:
                    error_message = await response.json()
                    if "Model" in error_message["error"] and "is currently loading" in error_message["error"]:
                        print(f"Service Unavailable. Model is loading. Estimated time: {error_message['estimated_time']} seconds. Retrying... Attempt {attempt + 1}/{retry_attempts}")
                        await asyncio.sleep(extended_delay)
                    else:
                        print(f"Service Unavailable. Status code: 503. Error: {error_message}. Retrying... Attempt {attempt + 1}/{retry_attempts}")
                else:
                    print(f"Request failed with status code {response.status}. Retrying... Attempt {attempt + 1}/{retry_attempts}")
        except aiohttp.ClientError as e:
            print(f"Request failed: {str(e)}. Retrying... Attempt {attempt + 1}/{retry_attempts}")
        
        await asyncio.sleep(retry_delay)
    
    return {"error": f"Failed to get response after {retry_attempts} retries"}

async def classify_row(session, row):
    payload = {
        "inputs": row['message_channel'],
        "parameters": {"candidate_labels": candidate_labels}
    }
    try:
        result = await query_async(session, payload)
        if 'scores' not in result:
            print("Result without scores:", result)
            return classify_locally(row['message_channel'])
        highest_score_index = result['scores'].index(max(result['scores']))
        most_likely_label = result['labels'][highest_score_index]
        return most_likely_label
    except Exception as e:
        print(f"Failed to classify remotely: {str(e)}. Classifying locally instead...")
        return classify_locally(row['message_channel'])

def classify_locally(text):
    classifier = init_classifier()
    result = classifier(text, candidate_labels)
    highest_score_index = result['scores'].index(max(result['scores']))
    most_likely_label = result['labels'][highest_score_index]
    return most_likely_label

def chunk_data(df, batch_size):
    num_chunks = math.ceil(len(df) / batch_size)
    return (df[i * batch_size:(i + 1) * batch_size] for i in range(num_chunks))

# Обновленная функция classify_text_async с обработкой данных батчами
async def classify_text_async(df, batch_size=24):
    async with aiohttp.ClientSession() as session:
        results = []

        for chunk in chunk_data(df, batch_size):
            tasks = [classify_row(session, row) for row in chunk.to_dict('records')]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

        df['Category'] = results

    return df

###С загрузкой модели из трансформеров
# # # Инициализация модели для zero-shot классификации с кэшированием и ThreadPoolExecutor
# @lru_cache(maxsize=1)
# def init_classifier():
#     # Проверка доступности CUDA до инициализации модели
#     device = 0 if torch.cuda.is_available() else -1
#     if device == -1:
#         print("CUDA is not available. Using CPU instead.")
#     classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0", device=device)
#     return classifier

# # Функция для классификации текста
# def classify_text(df, classifier):
#     start_clas_time = time.time()
#     candidate_labels = ["технологии", "политика", "спорт", "экономика", "развлечения", "здоровье", "образование", "мода", "инциденты и катастрофы"]

#     def classify_batch(batch):
#         messages = [row['message_channel'] for row in batch]
#         results = classifier(messages, candidate_labels)
#         return [result['labels'][result['scores'].index(max(result['scores']))] for result in results]

#     # Adjust batch size based on your system's capabilities and data characteristics
#     batch_size = 16  # Experiment with different batch sizes
#     num_batches = (len(df) + batch_size - 1) // batch_size

#     classified_results = []
#     with ThreadPoolExecutor() as executor:
#         futures = []
#         for i in range(num_batches):
#             start_idx = i * batch_size
#             end_idx = min((i + 1) * batch_size, len(df))
#             batch = df.iloc[start_idx:end_idx].to_dict('records')
#             futures.append(executor.submit(classify_batch, batch))

#         for future in as_completed(futures):
#             batch_results = future.result()
#             classified_results.extend(batch_results)

#     df['Category'] = classified_results
#     end_clf_time = time.time()
#     print(f"TIME Classification took {end_clf_time - start_clas_time} seconds")
#     return df