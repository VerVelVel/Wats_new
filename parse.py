# async_parse_telegram.py
import asyncio
from telethon import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from datetime import datetime, timedelta, timezone
import csv
import os

async def parse_telegram_messages():
    airflow_home = os.getenv('AIRFLOW_HOME', os.path.expanduser('~/airflow'))
    api_id = 29936300
    api_hash = 'fde88dcaee764c22dab01c68a8a3c347'
    phone = "+79313528188"

    client = TelegramClient('session_name', api_id, api_hash)
    await client.start()

    # Список с именами каналов, которые вы хотите парсить
    channel_usernames = ['nytimes', 'timiliasov', 'rian_ru', 'bbcrussian', 'bbbreaking', 'kstati_e', 'banksta', 'economika',
                         'suverenka', 'Match_TV', 'sportazarto', 'olympic_russia', 'UFCRussia', 'matchpremier', 'sportsru']

    # Определяем период, за который будем парсить сообщения (последние 24 часа)
    since_date = datetime.now(timezone.utc) - timedelta(days=3)

    # Сохраняем данные о сообщениях в список
    all_messages = []

    for channel_username in channel_usernames:
        # Получаем информацию о канале
        channel = await client.get_entity(channel_username)

        limit = 100
        offset_id = 0
        while True:
            history = await client(GetHistoryRequest(
                peer=channel,
                offset_id=offset_id,
                offset_date=None,
                add_offset=0,
                limit=limit,
                max_id=0,
                min_id=0,
                hash=0
            ))

            if not history.messages:
                break

            messages = history.messages
            for message in messages:
                # Приводим message.date к UTC (aware datetime)
                message_date_utc = message.date.astimezone(timezone.utc)

                # Проверяем, попадает ли сообщение в интервал последних 24 часов
                if message_date_utc > since_date:
                    # Извлекаем информацию о сообщении
                    channel_name = channel_username
                    post_date = message.date
                    message_text = message.message if message.message else "NaN"
                    message_id = f"{channel_username}_{message.id}"  # Уникальный идентификатор для каждого сообщения

                    # Добавляем данные о сообщении в список
                    all_messages.append([channel_name, post_date, message_text, message_id])

            offset_id = messages[len(messages) - 1].id

            if len(all_messages) >= limit:
                break

    # Сохраняем данные в CSV файл
    csv_filename = f"{airflow_home}/test_parsing_preprocessing.csv"  # Укажите правильный путь к файлу
    with open(csv_filename, "w", encoding="UTF-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Channel Name", "Post Date", "Message Text", "Message ID"])
        writer.writerows(all_messages)

    print(f"Сохранено {len(all_messages)} сообщений за последний день в файл {csv_filename}")

    await client.disconnect()
