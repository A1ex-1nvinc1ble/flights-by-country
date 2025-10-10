import os
import requests
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

# --- Конфигурация ---
# Загружаем ключи из переменных окружения
FLIGHT_API_KEY = os.getenv("FLIGHT_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Конфигурируем Gemini API
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Инициализируем FastAPI приложение
app = FastAPI()

# --- Модели данных ---
class UserQuery(BaseModel):
    airport: str
    question: str

# --- Логика приложения ---

@app.post("/api/ask")
async def handle_question(query: UserQuery):
    print(f"Получен запрос для аэропорта {query.airport} с вопросом: '{query.question}'")

    if not FLIGHT_API_KEY or not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Один или несколько API ключей не настроены на сервере.")

    # 1. Получаем данные о прибывающих рейсах с FlightAPI.io
    flight_api_url = f"https://flightapi.io/api/v1/airport/{query.airport}/arrivals?key={FLIGHT_API_KEY}"
    try:
        response = requests.get(flight_api_url)
        if response.status_code != 200:
            print(f"Ошибка от FlightAPI: {response.status_code} {response.text}")
            raise HTTPException(status_code=502, detail=f"Не удалось получить данные о рейсах. Код: {response.status_code}")
        flight_data = response.json()
    except requests.RequestException as e:
        print(f"Ошибка сети при запросе к FlightAPI: {e}")
        raise HTTPException(status_code=503, detail="Ошибка сети при запросе к сервису полетов.")

    # 2. Формируем запрос к Gemini
    # Сериализуем данные в строку JSON для передачи в промпт
    flight_data_str = json.dumps(flight_data, indent=2, ensure_ascii=False)
    
    prompt = f"""
    Ты — ассистент по анализу авиаперелетов. Твоя задача — отвечать на вопросы пользователя, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленных JSON-данных о рейсах. Будь кратким и точным. Не придумывай информацию.

    Вот данные о прибывающих рейсах в аэропорт {query.airport} в формате JSON:
    '''json
    {flight_data_str}
    '''

    Вопрос пользователя: "{query.question}"
    """

    # 3. Отправляем запрос в Gemini и получаем ответ
    try:
        model = genai.GenerativeModel('gemini-pro')
        gemini_response = model.generate_content(prompt)
        answer = gemini_response.text
    except Exception as e:
        print(f"Ошибка от Gemini API: {e}")
        # Gemini может блокировать запросы, если сочтет их небезопасными
        if 'block_reason' in str(e):
             answer = "Не удалось сгенерировать ответ из-за ограничений безопасности контента. Попробуйте переформулировать вопрос."
        else:
            raise HTTPException(status_code=500, detail="Произошла ошибка при обработке вопроса LLM.")

    # 4. Возвращаем ответ фронтенду
    return {"answer": answer}

@app.get("/")
def read_root():
    return {"status": "Backend is running with Gemini"}