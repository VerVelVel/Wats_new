# parser.py

import logging
from datetime import datetime, timedelta, timezone
from telethon import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import InputPeerChannel
import csv
import asyncio
import os

# Настройки клиента Telethon
api_id = 
api_hash = 
phone_number = 

# Создание клиента Telethon
client = TelegramClient('session_name', api_id, api_hash)

# Настройка логирования
logger = logging.getLogger(__name__)

# Функция для сохранения данных в CSV
def save_to_csv(data, filename='output.csv'):
    scripts_dir = os.path.dirname(os.path.abspath(__file__))  # Получаем текущую директорию скрипта
    filepath = os.path.join(scripts_dir, filename)
    logger.info(f"Сохранение данных в файл {filename}")
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Title', 'Content', 'Published At', 'Source', 'URL', 'Category'])  # Заголовки столбцов
            writer.writerows(data)
        logger.info(f"Данные успешно сохранены в файл {filename}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении данных в файл: {e}")

# Функция для парсинга
async def parse_channels(links, period_days):
    await client.start(phone=phone_number)
    target_date = datetime.now(timezone.utc) - timedelta(days=period_days)
    all_news = []

    async with client:
        for url in links:
            url = url.strip()
            try:
                channel = await client.get_entity(url)
                logger.info(f"Получен канал: {url}")
            except Exception as e:
                logger.error(f"Необработанная ошибка при получении канала {url}: {e}")
                continue

            if not hasattr(channel, 'broadcast') or not channel.broadcast:
                logger.warning(f"{url} не является телеграм-каналом. Пропускаем.")
                continue

            logger.info(f"Начинаем парсинг сообщений с {url}")

            offset_id = 0
            limit = 100

            while True:
                try:
                    history = await client(GetHistoryRequest(
                        peer=InputPeerChannel(channel.id, channel.access_hash),
                        offset_id=offset_id,
                        offset_date=None,
                        add_offset=0,
                        limit=limit,
                        max_id=0,
                        min_id=0,
                        hash=0
                    ))
                except telethon.errors.RPCError as e:
                    logger.error(f"Ошибка RPC при работе с Telethon: {e}")
                    break
                except telethon.errors.rpcerrorlist.FloodWaitError as e:
                    logger.error(f"Слишком много запросов при получении истории сообщений с канала {url}. Ожидание {e.seconds} секунд перед повторной попыткой.")
                    await asyncio.sleep(e.seconds)
                    continue
                except Exception as e:
                    logger.error(f"Необработанная ошибка при получении истории сообщений с канала {url}: {e}")
                    break

                if not history.messages:
                    logger.info(f"Нет больше сообщений для парсинга с канала {url}")
                    break

                messages = history.messages
                for message in messages:
                    logger.info(f"Обработка сообщения ID {message.id} с канала {url}")
                    if message.date.replace(tzinfo=timezone.utc) >= target_date:
                        title = message.message.split('\n')[0] if message.message else 'No Title'
                        content = message.message
                        published_at = message.date.replace(tzinfo=timezone.utc).isoformat()
                        source = channel.title
                        url_record = f"https://t.me/{channel.username}/{message.id}" if channel.username else "URL not available"
                        category = ''  # категория может быть заполнена позже
                        all_news.append((title, content, published_at, source, url_record, category))
                    else:
                        logger.info(f"Сообщение ID {message.id} с канала {url} старше указанного периода")
                        break

                offset_id = messages[-1].id
                if messages[-1].date.replace(tzinfo=timezone.utc) < target_date:
                    logger.info(f"Сообщения на канале {url} старше указанного периода. Завершение парсинга.")
                    break

                # Добавляем задержку между запросами
                await asyncio.sleep(1)
                logger.info(f"Обработано {len(messages)} сообщений с канала {url}, продолжаем парсинг...")

            # Добавляем задержку между обработкой разных каналов
            await asyncio.sleep(5)
            logger.info(f"Завершен парсинг сообщений с канала {url}")

    if all_news:
        logger.info(f"Всего собрано {len(all_news)} сообщений из всех каналов")
        save_to_csv(all_news)
        return True
    else:
        logger.info("Парсинг завершен. Нет данных для сохранения.")
        return False