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

# Загрузка модели и токенизатора из pkl файла
# scripts_dir = os.path.dirname(os.path.abspath(__file__))  # Получаем текущую директорию скрипта
# filepath = os.path.join(scripts_dir, "model_tokenizer.pkl")
# with open(filepath, "rb") as f:
#     model, tokenizer = pickle.load(f)

# # Перемещение модели на GPU, если доступно
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

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

# # # Инициализация модели для zero-shot классификации с кэшированием
@lru_cache(maxsize=1)
def init_classifier():
    # Проверка доступности CUDA до инициализации модели
    device = 0 if torch.cuda.is_available() else -1
    if device == -1:
        print("CUDA is not available. Using CPU instead.")
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0", device=device)
    return classifier

# Функция для классификации текста
def classify_text(df, classifier):
    start_clas_time = time.time()
    candidate_labels = ["технологии", "политика", "спорт", "экономика", "развлечения", "здоровье", "образование", "мода", "инциденты и катастрофы"]

    def classify_batch(batch):
        messages = [row['message_channel'] for row in batch]
        results = classifier(messages, candidate_labels)
        return [result['labels'][result['scores'].index(max(result['scores']))] for result in results]

    # Adjust batch size based on your system's capabilities and data characteristics
    batch_size = 16  # Experiment with different batch sizes
    num_batches = (len(df) + batch_size - 1) // batch_size

    classified_results = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))
            batch = df.iloc[start_idx:end_idx].to_dict('records')
            futures.append(executor.submit(classify_batch, batch))

        for future in as_completed(futures):
            batch_results = future.result()
            classified_results.extend(batch_results)

    df['Category'] = classified_results
    end_clf_time = time.time()
    print(f"TIME Classification took {end_clf_time - start_clas_time} seconds")
    return df
# # Классификация текста
# def classify_text(df):
#     candidate_labels = ["технологии", "политика", "спорт", "экономика", "развлечения", "здоровье", "образование", "мода", "инциденты и катастрофы"]

#     def classify_row(row):
#         inputs = tokenizer(row['message_channel'], return_tensors="pt", padding=True, truncation=True).to(device)
#         with torch.no_grad():
#             outputs = model(**inputs)
#             scores = softmax(outputs.logits, dim=1).cpu().numpy().flatten()
#             highest_score_index = scores.argmax()
#             most_likely_label = candidate_labels[highest_score_index]
#             return most_likely_label

#     with ThreadPoolExecutor() as executor:
#         results = list(executor.map(classify_row, df.to_dict('records')))
#     df['Category'] = results
#     return df

# async def query_async(session, payload, max_retries=3):
#     for attempt in range(max_retries):
#         try:
#             async with session.post(API_URL, headers=headers, json=payload) as response:
#                 result = await response.json()
#                 if 'scores' in result:
#                     return result
#                 else:
#                     print(f"Result without scores on attempt {attempt + 1} of {max_retries}")
#         except aiohttp.ClientError as e:
#             print(f"ClientError: {e}. Attempt {attempt + 1} of {max_retries}")
#         except Exception as e:
#             print(f"Unexpected error: {e}. Attempt {attempt + 1} of {max_retries}")
#         await asyncio.sleep(2 ** attempt)  # Exponential backoff

#     return {"error": f"Failed to get response after {max_retries} retries"}

# async def classify_row_async(session, row, candidate_labels, max_retries=3):
#     payload = {
#         "inputs": row['message_channel'],
#         "parameters": {"candidate_labels": candidate_labels}
#     }
#     result = await query_async(session, payload, max_retries)

#     if 'scores' not in result:
#         print("Result without scores:", result)
#         return "error"

#     highest_score_index = result['scores'].index(max(result['scores']))
#     most_likely_label = result['labels'][highest_score_index]
#     return most_likely_label

# async def classify_text_async(df):
#     candidate_labels = ["технологии", "политика", "спорт", "экономика", "развлечения", "здоровье", "образование", "мода", "инциденты и катастрофы"]

#     async with aiohttp.ClientSession() as session:
#         tasks = [classify_row_async(session, row, candidate_labels) for row in df.to_dict('records')]
#         results = []
#         for future in asyncio.as_completed(tasks):
#             result = await future
#             results.append(result)
#             await asyncio.sleep(0.5)  # Добавление небольшой задержки между запросами
#         df['Category'] = results

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