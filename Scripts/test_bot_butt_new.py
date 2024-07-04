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

# Настройки бота aiogram
bot_token = '7388106883:AAGznNWkQqs3dxBb90BXT5OaOS3ln_dD2ZU'
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

# Генерация клавиатуры для категорий
def generate_category_keyboard(categories):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    for category in categories:
        keyboard.add(types.KeyboardButton(category))
    keyboard.add(types.KeyboardButton(text="К вводу каналов"))
    return keyboard


# Генерация клавиатуры для выбора действия после выбора категории
action_keyboard = types.ReplyKeyboardMarkup(keyboard=[
    [types.KeyboardButton(text="Показать саммари")],
    [types.KeyboardButton(text="Вывести тренды")], 
    [types.KeyboardButton(text="К выбору категории")]
], resize_keyboard=True)


# Состояния для FSM
class ParseStates:
    WAITING_FOR_LINKS = 1
    WAITING_FOR_PERIOD = 2
    WAITING_FOR_DATE = 3
    CHOOSING_CATEGORY = 4
    CHOOSING_ACTION = 5
    SHOWING_NEWS = 6

# Переменная для хранения данных пользователя
user_data = {}



# Команда /start
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.answer_sticker('CAACAgQAAxkBAAEGoM1mhZx-75Z6L_im4Q7slV1DrftqjgACWAADzjkIDRhMYBsy9QjTNQQ')
    await message.reply("Привет, друг! Я твой помощник в мире новостей 📰\nДавай начнем и узнаем, что происходит в мире!", reply_markup=parse_keyboard)
    logger.info("Пользователь начал работу с ботом.")


# Команда "Начнем!"
@dp.message_handler(lambda message: message.text == "Начнем!")
async def parse_command(message: types.Message):
    await message.answer("Отлично! Введи ссылки на Telegram-каналы, новости из которых ты хочешь узнать, разделенные запятой:", reply_markup=types.ReplyKeyboardRemove())
    await message.answer_sticker('CAACAgQAAxkBAAEGoUpmhanDXAtk4cVRcoWJCoXs0jt4XQACzAADzjkIDd9nfGV-RLlkNQQ')
    user_data[message.from_user.id] = {'state': ParseStates.WAITING_FOR_LINKS}
    logger.info(f"Пользователь {message.from_user.id} ввел команду Начнем!")

# Обработка введенных ссылок
@dp.message_handler(lambda message: user_data.get(message.from_user.id, {}).get('state') == ParseStates.WAITING_FOR_LINKS)
async def handle_links(message: types.Message):
    user_data[message.from_user.id]['links'] = message.text
    user_data[message.from_user.id]['state'] = ParseStates.WAITING_FOR_PERIOD
    await message.answer("Отлично! Теперь выбери период поиска новостей или введи конкретную дату:", reply_markup=period_keyboard)
    await message.answer_sticker('CAACAgQAAxkBAAEGoVhmhaxYgpNmGLirqGucA5u_tm_b3wACcAADzjkIDZMJAAG9MCuf2zUE')
    logger.info(f"Пользователь {message.from_user.id} ввел ссылки: {message.text}")

# Обработка выбора периода
@dp.message_handler(lambda message: message.text in ["Выбрать период", "Ввести конкретную дату", "1 день", "3 дня", "Назад", "К вводу каналов"])
async def handle_period_choice(message: types.Message):
    user_id = message.from_user.id
    logger.info(f"Пользователь {user_id} выбрал опцию: {message.text}")
    logger.info(f"Текущее состояние пользователя: {user_data.get(user_id, {}).get('state')}")

    if message.text == "Выбрать период":
        await message.answer("Выбери период для парсинга:", reply_markup=period_options_keyboard)
    elif message.text == "Ввести конкретную дату":
        await message.answer("Введи дату в формате ГГГГ-ММ-ДД:")
        user_data[user_id]['state'] = ParseStates.WAITING_FOR_DATE
    elif message.text in ["1 день", "3 дня"]:
        period_days = 3 if message.text == "3 дня" else 1
        links = user_data[user_id]['links'].split(',')
        await start_parsing(message, links, period_days=period_days)
    elif message.text == "Назад":
        await message.answer("Выбери период поиска новостей или введи конкретную дату:", reply_markup=period_keyboard)
    elif message.text == "К вводу каналов":
        user_data[user_id]['state'] = ParseStates.WAITING_FOR_LINKS
        await message.answer("Введи ссылки на Telegram-каналы, новости из которых ты хочешь узнать, разделенные запятой:", reply_markup=types.ReplyKeyboardRemove())

# Обработка введенной конкретной даты
@dp.message_handler(lambda message: user_data.get(message.from_user.id, {}).get('state') == ParseStates.WAITING_FOR_DATE)
async def handle_date(message: types.Message):
    user_id = message.from_user.id
    date = message.text
    try:
        specific_date = pd.to_datetime(date)  # Проверка корректности даты
    except ValueError:
        await message.answer("Неверный формат даты. Попробуй снова.")
        await message.answer_sticker('CAACAgQAAxkBAAEGoV9mhayYLmqlqsGk3edHEpPr3dqlZQACygADzjkIDSDvUXySrKaQNQQ')
        return
    
    links = user_data[user_id]['links'].split(',')
    await start_parsing(message, links, specific_date=specific_date)
    logger.info(f"Парсинг завершен для пользователя {user_id}")

async def start_parsing(message: types.Message, links: list, period_days=None, specific_date=None):
    user_id = message.from_user.id

    await message.answer("Начинаю парсинг... Это может занять некоторое время. Подождёшь немного? 🕒", reply_markup=types.ReplyKeyboardRemove())
    await message.answer_sticker('CAACAgQAAxkBAAEGoWFmhazOoLDDgga7HMD8SIRCeL41_wACsAADzjkIDRihrUgMN9XlNQQ')
    logger.info(f"Начинаем парсинг для пользователя {user_id} с периодом {period_days} дней и ссылками: {links}")

    # Инициализация моделей и парсинг параллельно
    parsing_task = asyncio.create_task(parse_channels(links, period_days=period_days, specific_date=specific_date))
    init_models_task = asyncio.create_task(init_models_async())

    # Ожидание завершения задач
    success = await parsing_task
    classifier, ranking_model = await init_models_task

    if success:
        await message.answer("Парсинг завершен. Начинаю классификацию...")
        await message.answer_sticker('CAACAgQAAxkBAAEGoWNmha1O2YZ-c38k7RyPqsiIPZz1wgACcgADzjkIDZ4tlzE34tf6NQQ')

        # Классификация
        scripts_dir = os.path.dirname(os.path.abspath(__file__))  # Получаем текущую директорию скрипта
        filepath = os.path.join(scripts_dir, 'output.csv')
        classified_filepath = os.path.join(scripts_dir, 'classified_output.csv')
        ranked_filepath = os.path.join(scripts_dir, 'ranked_output.csv')
        summary_filepath = os.path.join(scripts_dir, 'summary_output.csv')

        if os.path.exists(classified_filepath):
            df = pd.read_csv(classified_filepath)
            await message.answer(f"Классификация уже выполнена ранее. Данные загружены из файла.")
            logger.info(f"Классификация уже выполнена ранее. Данные загружены из файла {classified_filepath}.")
        else:
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df = preprocess_data(df)
                df = classify_text(df, classifier)

                # Сохранение данных с классификацией
                df.to_csv(filepath, index=False)   # Заменить обратно на filepath
                await message.answer(f"Классификация завершена. Данные сохранены.")
                logger.info(f"Классификация завершена. Данные сохранены в файле {filepath}.")
            else:
                await message.answer(f"Ошибка: файл {filepath} не найден.")
                logger.error(f"Ошибка: файл {filepath} не найден.")
                return

        # Ранжирование и саммаризация параллельно
        ranking_task = most_common_meaningful_text_async(df, 'Category', 'clean_text', 'Content', ranking_model)
        summary_task = generate_summaries(df)
        ranked_df, summary_df = await asyncio.gather(ranking_task, summary_task)
        
        # Сохранение данных с ранжированием и саммаризацией
        ranked_df.to_csv(ranked_filepath, index=False)
        summary_df.to_csv(summary_filepath, index=False)
        
        await message.answer(f"Ранжирование и саммаризация завершены. Данные сохранены в файле {filepath}.")
        logger.info(f"Ранжирование и суммаризация завершены. Данные сохранены в файле {filepath}.")

        # Отправка кнопок с категориями пользователю
        await message.answer("Выбери категорию новостей:", reply_markup=generate_category_keyboard(df['Category'].unique()))
        await message.answer_sticker('CAACAgQAAxkBAAEGoWdmha2ZaHVkoel6Scwdvz1-ChIYQAACVwADzjkIDSf1_eb9ekmGNQQ')
        user_data[user_id]['state'] = ParseStates.CHOOSING_CATEGORY

    else:
        await message.answer("Парсинг завершен. Нет данных для сохранения.")
        logger.info("Парсинг завершен. Нет данных для сохранения.")

# Обработка выбора категории
@dp.message_handler(lambda message: user_data.get(message.from_user.id, {}).get('state') == ParseStates.CHOOSING_CATEGORY)
async def choose_category(message: types.Message):
    user_id = message.from_user.id
    chosen_category = message.text

    if chosen_category == "К вводу каналов":
        user_data[user_id]['state'] = ParseStates.WAITING_FOR_LINKS
        await message.answer("Введи ссылки на Telegram-каналы, новости из которых ты хочешь узнать, разделенные запятой:", reply_markup=types.ReplyKeyboardRemove())
    else:    
        # Проверяем, что категория существует
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        ranked_filepath = os.path.join(scripts_dir, 'ranked_output.csv')
        summary_filepath = os.path.join(scripts_dir, 'summary_output.csv')

        if os.path.exists(ranked_filepath) and os.path.exists(summary_filepath):
            most_common_meaningful_df = pd.read_csv(ranked_filepath)
            summary_df = pd.read_csv(summary_filepath)

            if chosen_category in most_common_meaningful_df['category'].values:
                await message.answer(f"Категория '{chosen_category}' выбрана. Что хочешь сделать дальше?", reply_markup=action_keyboard)
                user_data[user_id]['chosen_category'] = chosen_category
                user_data[user_id]['state'] = ParseStates.CHOOSING_ACTION
            else:
                await message.answer(f"Категория '{chosen_category}' не найдена. Попробуй снова.")
        else:
            await message.answer(f"Ошибка: файлы {ranked_filepath} или {summary_filepath} не найдены.")
            logger.error(f"Ошибка: файлы {ranked_filepath} или {summary_filepath} не найдены.")

# Обработка выбора действия после выбора категории
@dp.message_handler(lambda message: user_data.get(message.from_user.id, {}).get('state') == ParseStates.CHOOSING_ACTION)
async def handle_action_choice(message: types.Message):
    user_id = message.from_user.id
    chosen_action = message.text.lower()
    scripts_dir = os.path.dirname(os.path.abspath(__file__))

    chosen_category = user_data.get(user_id, {}).get('chosen_category', None)

    if chosen_action == "показать саммари":
        summary_df = pd.read_csv(os.path.join(scripts_dir, 'summary_output.csv'))

        # Фильтруем DataFrame по выбранной категории
        summaries = summary_df[summary_df['Category'] == chosen_category]

        # Собираем все суммаризированные тексты и ссылки в одно сообщение
        response_message = ""
        for index, row in summaries.iterrows():
            summary_text = row['summary']
            original_url = row['URL']

            response_message += f"**Саммари**:\n{summary_text}\n\nСсылка на пост: {original_url}\n\n"
        await message.answer(response_message, parse_mode='Markdown')

        # После показа саммари возвращаем пользователя к выбору действия
        await message.answer("Что хочешь сделать дальше?", reply_markup=action_keyboard)
        user_data[user_id]['state'] = ParseStates.CHOOSING_ACTION

    elif chosen_action == "вывести тренды":
        ranked_df = pd.read_csv(os.path.join(scripts_dir, 'ranked_output.csv'))
        news_text = ranked_df[ranked_df['category'] == chosen_category]['most_common_text'].values[0]
        await message.answer(f"Наиболее значимая новость в категории '{chosen_category}':\n\n{news_text}")

        # После показа трендов также возвращаем пользователя к выбору действия
        await message.answer("Что хочешь сделать дальше?", reply_markup=action_keyboard)
        user_data[user_id]['state'] = ParseStates.CHOOSING_ACTION

    elif chosen_action == "к выбору категории":
        # Отправка кнопок с категориями пользователю
        user_data[user_id]['state'] = ParseStates.CHOOSING_CATEGORY
        df = pd.read_csv(os.path.join(scripts_dir, 'ranked_output.csv')) 
        await message.answer("Выбери категорию новостей:", reply_markup=generate_category_keyboard(df['category'].unique()))
        user_data[user_id]['state'] = ParseStates.CHOOSING_CATEGORY

    else:
        await message.answer("Извини, я не понял твоего выбора. Пожалуйста, выбери действие из списка.")

    
# Запуск бота
if __name__ == "__main__":
    from aiogram import executor
    executor.start_polling(dp, skip_updates=True)
