import os
import requests
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from fastapi.responses import FileResponse

# --- Конфигурация ---
FLIGHT_API_KEY = os.getenv("FLIGHT_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Инициализируем FastAPI приложение
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Конфигурируем Gemini API
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- Модели данных ---
class UserQuery(BaseModel):
    airport: str
    question: str

# --- Логика приложения ---

@app.get("/")
async def serve_frontend():
    """Обслуживаем фронтенд"""
    return FileResponse("public/index.html")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "framework": "FastAPI"}

@app.post("/api/ask")
async def handle_question(query: UserQuery):
    print(f"Получен запрос для аэропорта {query.airport} с вопросом: '{query.question}'")

    if not FLIGHT_API_KEY:
        raise HTTPException(status_code=500, detail="FLIGHT_API_KEY не настроен")
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY не настроен")

    # 1. Сначала используем демо-данные для тестирования
    demo_data = {
        "DXB": [
            {"flightNumber": "EK202", "departure": {"city": "London", "country": "UK"}, "airline": "Emirates"},
            {"flightNumber": "EK412", "departure": {"city": "Mumbai", "country": "India"}, "airline": "Emirates"}
        ],
        "LHR": [
            {"flightNumber": "BA123", "departure": {"city": "New York", "country": "USA"}, "airline": "British Airways"},
            {"flightNumber": "LH456", "departure": {"city": "Frankfurt", "country": "Germany"}, "airline": "Lufthansa"}
        ]
    }
    
    flight_data = {"flights": demo_data.get(query.airport, [])}
    
    # 2. Формируем запрос к Gemini
    flight_data_str = json.dumps(flight_data, indent=2, ensure_ascii=False)
    
    prompt = f"""
    Ты — ассистент по анализу авиаперелетов. Отвечай на вопросы ИСКЛЮЧИТЕЛЬНО на основе предоставленных данных.

    Данные о рейсах в аэропорт {query.airport}:
    {flight_data_str}

    Вопрос: "{query.question}"

    Инструкции:
    - Отвечай кратко и точно
    - Используй только предоставленные данные
    - Если информации нет, скажи об этом
    - Отвечай на русском языке

    Ответ:
    """

    try:
        # 3. Отправляем запрос в Gemini
        model = genai.GenerativeModel('gemini-pro')
        gemini_response = model.generate_content(prompt)
        answer = gemini_response.text
        
        return {
            "answer": answer,
            "debug": {
                "flights_count": len(flight_data["flights"]),
                "airport": query.airport
            }
        }
        
    except Exception as e:
        print(f"Ошибка от Gemini API: {e}")
        # Fallback ответ
        fallback_answer = f"На основе данных для аэропорта {query.airport}, найдено {len(flight_data['flights'])} рейсов. AI сервис временно недоступен."
        return {"answer": fallback_answer}

# Для Vercel
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)