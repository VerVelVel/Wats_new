#скачать один раз
nltk.download('stopwords')
nltk.download('wordnet')

import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
from transformers import pipeline
from datasets import Dataset


# df = pd.read #добавить файл с данными после парсинга
# Загрузка данных и предобработка
def preprocess_data(df):

    df = df.dropna(subset=['Message Text'])

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

    df['clean_text'] = df['Message Text'].apply(preprocess_text)
    df['message_channel'] = df['clean_text'] + ' ' + df['Channel Name']

    return df

df = preprocess_data(df)

# Инициализация модели для zero-shot классификации
def init_classifier():
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device=0 if torch.cuda.is_available() else -1)
    return classifier

classifier = init_classifier()
dataset = Dataset.from_pandas(df)
# Возможные категории для классификации
candidate_labels = ["технологии", "политика", "спорт", "экономика", "развлечения", "здоровье", "образование", "мода", "происшествия"]

# Функция для классификации текста
def classify_text(example):
    result = classifier(example['message_channel'], candidate_labels)
    # Извлечение наиболее вероятного класса
    highest_score_index = result['scores'].index(max(result['scores']))
    most_likely_label = result['labels'][highest_score_index]
    example['classification'] = most_likely_label
    return example

classified_dataset = dataset.map(classify_text)

# Преобразование обратно в DataFrame для удобства работы
classified_df = classified_dataset.to_pandas()
