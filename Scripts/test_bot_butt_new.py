import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.contrib.middlewares.logging import LoggingMiddleware
import asyncio
import os
from aiogram.utils import executor
import pandas as pd
from parser_1 import parse_channels 
from preprocessing_classificator_new import preprocess_data, classify_text, init_models_async, most_common_meaningful_text_async, generate_summaries
from aiogram.utils.exceptions import BotBlocked

# from preprocessing_classificator import preprocess_data, classify_text_async

# Настройки бота aiogram
bot_token = '7466597015:AAEsbquKsmzi6Sr_YtNv3SCkeM3uukm1FT4'
bot = Bot(token=bot_token)
dp = Dispatcher(bot)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Команды для кнопок
start_keyboard = types.ReplyKeyboardMarkup(keyboard=[[types.KeyboardButton(text="/start")]], resize_keyboard=True)

parse_keyboard = types.ReplyKeyboardMarkup(keyboard=[[types.KeyboardButton(text="Начнем!")]], resize_keyboard=True)

period_keyboard = types.ReplyKeyboardMarkup(keyboard=[[types.KeyboardButton(text="Выбрать период")],
                                                      [types.KeyboardButton(text="Ввести конкретную дату")],
                                                      [types.KeyboardButton(text="К вводу каналов")]], resize_keyboard=True)

period_options_keyboard = types.ReplyKeyboardMarkup(keyboard=[[types.KeyboardButton(text="1 день")],
                                                              [types.KeyboardButton(text="3 дня")], [types.KeyboardButton(text="Назад")]], resize_keyboard=True)

#Генерация клавиатуры для категорий
def generate_category_keyboard(categories):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    for category in categories:
        keyboard.add(types.KeyboardButton(category))
    return keyboard

# Состояния для FSM
class ParseStates:
    WAITING_FOR_LINKS = 1
    WAITING_FOR_PERIOD = 2
    WAITING_FOR_DATE = 3
    CHOOSING_CATEGORY = 4
    SHOWING_NEWS = 5

# Переменная для хранения данных пользователя
user_data = {}

# Команда /start
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Привет! Я здесь, чтобы помочь тебе быть в курсе всех новостей 🤓. Давай начинать!", reply_markup=parse_keyboard)
    logger.info("Пользователь начал работу с ботом.")

# Команда "поехали!"
@dp.message_handler(lambda message: message.text == "Начнем!")
async def parse_command(message: types.Message):
    await message.answer("Введите ссылки на Telegram-каналы, новости из которых вы хотите узнать, разделенные запятой:", reply_markup=types.ReplyKeyboardRemove())
    user_data[message.from_user.id] = {'state': ParseStates.WAITING_FOR_LINKS}
    logger.info(f"Пользователь {message.from_user.id} ввел команду /parse")

# Обработка введенных ссылок
@dp.message_handler(lambda message: user_data.get(message.from_user.id, {}).get('state') == ParseStates.WAITING_FOR_LINKS)
async def handle_links(message: types.Message):
    user_data[message.from_user.id]['links'] = message.text
    user_data[message.from_user.id]['state'] = ParseStates.WAITING_FOR_PERIOD
    await message.answer("Выберите период поиска новостей или введите конкретную дату:", reply_markup=period_keyboard)
    logger.info(f"Пользователь {message.from_user.id} ввел ссылки: {message.text}")

# Обработка выбора периода
@dp.message_handler(lambda message: message.text in ["Выбрать период", "Ввести конкретную дату", "1 день", "3 дня", "Назад", "К вводу каналов"])
async def handle_period_choice(message: types.Message):
    user_id = message.from_user.id
    logger.info(f"Пользователь {user_id} выбрал опцию: {message.text}")
    logger.info(f"Текущее состояние пользователя: {user_data.get(user_id, {}).get('state')}")

    if message.text == "Выбрать период":
        await message.answer("Выберите период для парсинга:", reply_markup=period_options_keyboard)
    elif message.text == "Ввести конкретную дату":
        await message.answer("Введите дату в формате ГГГГ-ММ-ДД:")
        user_data[user_id]['state'] = ParseStates.WAITING_FOR_DATE
    elif message.text in ["1 день", "3 дня"]:
        period_days = 3 if message.text == "3 дня" else 1
        links = user_data[user_id]['links'].split(',')
        await start_parsing(message, links, period_days=period_days)
    elif message.text == "Назад":
        await message.answer("Выберите период поиска новостей или введите конкретную дату:", reply_markup=period_keyboard)
    elif message.text == "К вводу каналов":
        user_data[user_id]['state'] = ParseStates.WAITING_FOR_LINKS
        await message.answer("Введите ссылки на Telegram-каналы, новости из которых вы хотите узнать, разделенные запятой:",
                             reply_markup=types.ReplyKeyboardRemove())

# Обработка введенной конкретной даты
@dp.message_handler(lambda message: user_data.get(message.from_user.id, {}).get('state') == ParseStates.WAITING_FOR_DATE)
async def handle_date(message: types.Message):
    user_id = message.from_user.id
    date = message.text
    try:
        specific_date = pd.to_datetime(date)  # Проверка корректности даты
    except ValueError:
        await message.answer("Неверный формат даты. Попробуйте снова.")
        return
    
    links = user_data[user_id]['links'].split(',')
    await start_parsing(message, links, specific_date=specific_date)
    # user_data[user_id]['state'] = None  # Очищаем состояние после парсинга
    logger.info(f"Парсинг завершен для пользователя {user_id}")

async def start_parsing(message: types.Message, links: list, period_days=None, specific_date=None):
    user_id = message.from_user.id

    await message.answer("Начинаем парсинг...", reply_markup=types.ReplyKeyboardRemove())
    logger.info(f"Начинаем парсинг для пользователя {user_id} с периодом {period_days} дней и ссылками: {links}")
    
    # Инициализация моделей и парсинг параллельно
    parsing_task = asyncio.create_task(parse_channels(links, period_days=period_days, specific_date=specific_date))
    init_models_task = asyncio.create_task(init_models_async())

    # Ожидание завершения задач
    success = await parsing_task
    classifier, ranking_model = await init_models_task

    ## Только для локальной модели
    if success:
        await message.answer("Парсинг завершен. Начинаем классификацию...")

        # Классификация
        scripts_dir = os.path.dirname(os.path.abspath(__file__))  # Получаем текущую директорию скрипта
        filepath = os.path.join(scripts_dir, 'output.csv')
        classified_filepath = os.path.join(scripts_dir, 'classified_output.csv')
        ranked_filepath = os.path.join(scripts_dir, 'ranked_output.csv')
        summary_filepath = os.path.join(scripts_dir, 'summary_output.csv')

        if os.path.exists(classified_filepath):
            df = pd.read_csv(classified_filepath)
            await message.answer(f"Классификация уже выполнена ранее. Данные загружены из файла {classified_filepath}.")
            logger.info(f"Классификация уже выполнена ранее. Данные загружены из файла {classified_filepath}.")
        else:
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df = preprocess_data(df)
                df = classify_text(df, classifier)

                # Сохранение данных с классификацией
                df.to_csv(classified_filepath, index=False)
                await message.answer(f"Классификация завершена. Данные сохранены в файле {classified_filepath}.")
                logger.info(f"Классификация завершена. Данные сохранены в файле {classified_filepath}.")
            else:
                await message.answer(f"Ошибка: файл {filepath} не найден.")
                logger.error(f"Ошибка: файл {filepath} не найден.")
                return

        # Ранжирование и суммаризация параллельно
        ranking_task = most_common_meaningful_text_async(df, 'Category', 'clean_text', 'Content', ranking_model)
        summary_task = generate_summaries(df)
        ranked_df, summary_df = await asyncio.gather(ranking_task, summary_task)
        
        # Сохранение данных с ранжированием и суммаризацией
        ranked_df.to_csv(os.path.join(scripts_dir, 'ranked_output.csv'), index=False)
        summary_df.to_csv(os.path.join(scripts_dir, 'summary_output.csv'), index=False)
        
        await message.answer(f"Ранжирование и суммаризация завершены. Данные сохранены в файле {filepath}.")
        logger.info(f"Ранжирование и суммаризация завершены. Данные сохранены в файле {filepath}.")

        # Отправка кнопок с категориями пользователю
        await message.answer("Выберите категорию новостей:", reply_markup=generate_category_keyboard(df['Category'].unique()))
        user_data[user_id]['state'] = ParseStates.CHOOSING_CATEGORY

    else:
        await message.answer("Парсинг завершен. Нет данных для сохранения.")
        logger.info("Парсинг завершен. Нет данных для сохранения.")

@dp.message_handler(lambda message: user_data.get(message.from_user.id, {}).get('state') == ParseStates.CHOOSING_CATEGORY)
async def choose_category(message: types.Message):
    user_id = message.from_user.id
    chosen_category = message.text

    # Проверяем, что категория существует
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    ranked_filepath = os.path.join(scripts_dir, 'ranked_output.csv')
    summary_filepath = os.path.join(scripts_dir, 'summary_output.csv') # Добавлено
    if os.path.exists(ranked_filepath):
        most_common_meaningful_df = pd.read_csv(ranked_filepath)
        summary_df = pd.read_csv(summary_filepath) # Добавлено
        if chosen_category in most_common_meaningful_df['category'].values:
            news_text = most_common_meaningful_df[most_common_meaningful_df['category'] == chosen_category]['most_common_text'].values[0]
            summary_text = summary_df[summary_df['Category'] == chosen_category]['summary'].values[0] # Добавлено
            await message.answer(f"Наиболее значимая новость в категории '{chosen_category}':\n\n{news_text}\n\nСаммари:\n{summary_text}") # Изменено
            await ParseStates.SHOWING_NEWS.set()
            user_data.pop(user_id, None)  # Очистка данных пользователя после завершения процесса
        else:
            await message.answer(f"Категория '{chosen_category}' не найдена. Попробуйте снова.")
    else:
        await message.answer(f"Ошибка: файл {ranked_filepath} не найден.")
        logger.error(f"Ошибка: файл {ranked_filepath} не найден.")

# Запуск бота
if __name__ == "__main__":
    from aiogram import executor
    executor.start_polling(dp, skip_updates=True)


# #Только для API - aсинхронно
#     if success:
#         await message.answer("Парсинг завершен. Начинаем классификацию...")

#         # Классификация
#         scripts_dir = os.path.dirname(os.path.abspath(__file__))  # Получаем текущую директорию скрипта
#         filepath = os.path.join(scripts_dir, 'output.csv')
#         if os.path.exists(filepath):
#             df = pd.read_csv(filepath)
#             df = preprocess_data(df)
#             classified_df = await classify_text_async(df)  # Используем асинхронную классификацию

#             # Сохранение данных с классификацией
#             classified_df.to_csv(filepath, index=False)
#             await message.answer(f"Классификация завершена. Данные сохранены в файле {filepath}.")
#             logger.info(f"Классификация завершена. Данные сохранены в файле {filepath}.")
#         else:
#             await message.answer(f"Ошибка: файл {filepath} не найден.")
#             logger.error(f"Ошибка: файл {filepath} не найден.")
#     else:
#         await message.answer("Парсинг завершен. Нет данных для сохранения.")

#         logger.info("Парсинг завершен. Нет данных для сохранения.")
#     user_data.pop(message.from_user.id, None)  # Очистка данных пользователя

#Только для API - синхронно
    # if success:
    #     await message.answer("Парсинг завершен. Начинаем классификацию...")

    #     # Классификация
    #     scripts_dir = os.path.dirname(os.path.abspath(__file__))  # Получаем текущую директорию скрипта
    #     filepath = os.path.join(scripts_dir, 'output.csv')
    #     if os.path.exists(filepath):
    #         df = pd.read_csv(filepath)
    #         df = preprocess_data(df)
    #         classified_df = classify_text_async(df)

    #         # Сохранение данных с классификацией
    #         classified_df.to_csv(filepath, index=False)
    #         await message.answer("Классификация завершена. Данные сохранены в файле output.csv.")
    #     else:
    #         await message.answer("Ошибка: файл output.csv не найден.")
    # else:
    #     await message.answer("Парсинг завершен. Нет данных для сохранения.")
    
    # user_data.pop(message.from_user.id, None)  # Очистка данных пользователя