import torch
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time
import asyncio
from sentence_transformers import SentenceTransformer, util

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

#КЛАССИФИКАЦИЯ
##С загрузкой модели из трансформеров
# # Инициализация модели для zero-shot классификации с кэшированием и ThreadPoolExecutor
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

#РАНЖИРОВАНИЕ
@lru_cache(maxsize=1)
def init_ranking_model():
    # Проверка доступности CUDA до инициализации модели
    device = 0 if torch.cuda.is_available() else -1
    if device == -1:
        print("CUDA is not available. Using CPU instead.")
    ranking_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device=device)
    return ranking_model

async def init_models_async():
    start_init_time = time.time()
    classifier = init_classifier()  # Инициализация модели классификации
    ranking_model = init_ranking_model()  # Инициализация модели ранжирования
    end_init_time = time.time()
    print(f"TIME Init models took {end_init_time - start_init_time} seconds")   
    return classifier, ranking_model


async def most_common_meaningful_text_async(df, category_col, processed_text_col, original_text_col, ranking_model):
    start_rang_time = time.time()
    results = []

    async def process_category(category):
        processed_texts = df[df[category_col] == category][processed_text_col].tolist()
        original_texts = df[df[category_col] == category][original_text_col].tolist()

        # Используем ThreadPoolExecutor для выполнения CPU-bound задачи в отдельном потоке
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(None, lambda: ranking_model.encode(processed_texts, convert_to_tensor=True))

        # Подсчет средних косинусных сходств
        similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)
        similarity_sum = similarity_matrix.sum(dim=1)

        # Нахождение текста с наибольшей суммой сходств
        most_common_idx = similarity_sum.argmax().item()
        most_common_text = original_texts[most_common_idx]
        results.append({"category": category, "most_common_text": most_common_text})

    tasks = [process_category(category) for category in df[category_col].unique()]
    await asyncio.gather(*tasks)

    end_rang_time = time.time()
    print(f"TIME Ranging took {end_rang_time - start_rang_time} seconds")   
    return pd.DataFrame(results)


# САММАРИЗАЦИЯ
model_name = "SiberiaSoft/SiberianFredT5-instructor"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_summary(text):

    text = re.sub(r'@\w+', '', text)

    # Подготовка входных данных
    prompt = 'Выдели главные мысли в новости:'
    data = tokenizer('<SC6>' + prompt + text + '\nОтвет: <extra_id_0>', return_tensors="pt")
    data = {k: v.to(model.device) for k, v in data.items()}

    # Генерация текста
    output_ids = model.generate(
        **data,
        do_sample=True,
        temperature=0.2,
        max_new_tokens=512,
        top_p=0.95,
        top_k=5,
        repetition_penalty=1.03,
        no_repeat_ngram_size=4
    )[0]


    out = tokenizer.decode(output_ids.tolist())
    out = out.replace("<s>", "").replace("</s>", "").replace("<pad>", "").replace("<extra_id_0>", "").strip()
    return out

MAX_ARTICLES_PER_CATEGORY = 10  # Максимальное количество статей для саммаризации в каждой категории

# Асинхронная функция для генерации саммари
async def generate_summaries(df):
    loop = asyncio.get_event_loop()
    start_sum_time = time.time()
    selected_indices = []
    categories = df['Category'].unique()
    for category in categories:
        # Фильтруем данные по текущей категории и ограничиваем количество строк
        category_indices = df[df['Category'] == category].head(MAX_ARTICLES_PER_CATEGORY).index
        selected_indices.extend(category_indices)
    
    # Используем отобранные индексы для создания нового DataFrame с ограниченными данными
    filtered_df = df.loc[selected_indices]
    with ThreadPoolExecutor() as executor:
        summaries = await asyncio.gather(*[loop.run_in_executor(executor, generate_summary, text) for text in filtered_df['Content']])
    filtered_df['summary'] = summaries
    end_sum_time = time.time()
    print(f"TIME Summarization took {end_sum_time - start_sum_time} seconds")
    return filtered_df


# API_URL = "https://api-inference.huggingface.co/models/MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
# headers = {"Authorization": "Bearer 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"}

# candidate_labels = ["технологии", "политика", "спорт", "экономика", "развлечения", "здоровье", "образование", "мода", "инциденты и катастрофы"]

# BATCH_SIZE = 36

# async def preload_model():
#     """Send a dummy request to preload the model."""
#     dummy_payload = {
#         "inputs": "",
#         "parameters": {"candidate_labels": candidate_labels}
#     }
#     try:
#         response = await asyncio.to_thread(
#             requests.post, API_URL, headers=headers, json=dummy_payload, timeout=30
#         )
#         response.raise_for_status()
#         print("Model preloaded successfully.")
#     except requests.RequestException as e:
#         print(f"Failed to preload model: {str(e)}")

# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# def query(payload):
#     try:
#         response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
#         response.raise_for_status()
#         time.sleep(40)  # Задержка в 40 секунд после успешного запроса
#         return response.json()
#     except requests.RequestException as e:
#         print(f"Request failed: {str(e)}")
#         raise

# @lru_cache(maxsize=1000)
# def cached_query(payload_str):
#     payload = eval(payload_str)
#     return query(payload)

# def classify_batch(batch):
#     payload = {
#         "inputs": batch['message_channel'].tolist(),
#         "parameters": {"candidate_labels": candidate_labels}
#     }
#     payload_str = str(payload)
#     result = cached_query(payload_str)
    
#     if not isinstance(result, list):
#         print(f"Unexpected result format: {result}")
#         return [None] * len(batch)
    
#     categories = []
#     for item in result:
#         if 'scores' not in item or 'labels' not in item:
#             categories.append(None)
#         else:
#             highest_score_index = item['scores'].index(max(item['scores']))
#             categories.append(item['labels'][highest_score_index])
    
#     return categories

# def classify_text(df):
#     start_time = time.time()
#     results = []
    
#     total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    
#     for i in range(0, len(df), BATCH_SIZE):
#         batch = df.iloc[i:i+BATCH_SIZE]
#         batch_results = classify_batch(batch)
#         results.extend(batch_results)
        
#         print(f"Processed batch {i//BATCH_SIZE + 1}/{total_batches}")
    
#     df['Category'] = results
#     end_time = time.time()
#     print(f"Classification took {end_time - start_time:.2f} seconds")
#     return df

# async def classify_text_async(df):
#     start_time = time.time()
#     results = []
    
#     total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    
#     for i in range(0, len(df), BATCH_SIZE):
#         batch = df.iloc[i:i+BATCH_SIZE]
#         batch_results = await asyncio.to_thread(classify_batch, batch)
#         results.extend(batch_results)
        
#         print(f"Processed batch {i//BATCH_SIZE + 1}/{total_batches}")
    
#     df['Category'] = results
#     end_time = time.time()
#     print(f"Classification took {end_time - start_time:.2f} seconds")
#     return df
