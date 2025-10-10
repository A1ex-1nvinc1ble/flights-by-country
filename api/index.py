import os
import requests
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --- Конфигурация ---
FLIGHT_API_KEY = os.getenv("FLIGHT_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Инициализируем FastAPI приложение
app = FastAPI(title="Flights by Country", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Модели данных ---
class UserQuery(BaseModel):
    airport: str
    question: str

# --- Функция для получения данных о рейсах ---
async def get_flight_data(airport):
    """Получаем данные о рейсах - сначала пробуем реальный API, потом демо-данные"""
    
    # Если есть реальный API ключ, пробуем получить реальные данные
    if FLIGHT_API_KEY and FLIGHT_API_KEY != "your_flight_api_key_here":
        try:
            print(f"Пытаемся получить реальные данные для {airport}")
            real_data = await get_real_flight_data(airport)
            if real_data and real_data.get("flights"):
                print(f"Успешно получили реальные данные: {len(real_data['flights'])} рейсов")
                return real_data
            else:
                print("Реальные данные не получены, используем демо-данные")
        except Exception as e:
            print(f"Ошибка при получении реальных данных: {e}")
    
    # Если реальные данные не получены, используем демо-данные
    return get_demo_flight_data(airport)

async def get_real_flight_data(airport):
    """Получаем реальные данные из FlightAPI"""
    try:
        # Пробуем разные endpoints FlightAPI
        endpoints = [
            f"https://api.flightapi.io/schedule/{airport}/arrivals",
            f"https://api.flightapi.io/arrivals/{airport}",
            f"https://api.flightapi.io/schedule/arrivals/{airport}",
            f"https://flightapi.io/api/v1/airport/{airport}/arrivals"
        ]
        
        for endpoint in endpoints:
            try:
                print(f"Пробуем endpoint: {endpoint}")
                response = requests.get(
                    endpoint,
                    params={
                        "mode": "live",
                        "date": "2024-01-29",  # Можно сделать динамическим
                        "api_key": FLIGHT_API_KEY
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # Преобразуем данные в наш формат
                    if data.get("schedules"):
                        flights = []
                        for schedule in data["schedules"]:
                            flight_info = {
                                "flightNumber": schedule.get("flightNumber", "Unknown"),
                                "departure": {
                                    "city": schedule.get("departure", {}).get("city", "Unknown"),
                                    "country": schedule.get("departure", {}).get("country", "Unknown"),
                                    "iata": schedule.get("departure", {}).get("iata", "Unknown")
                                },
                                "airline": {
                                    "name": schedule.get("airline", {}).get("name", "Unknown")
                                },
                                "status": schedule.get("status", "Unknown")
                            }
                            flights.append(flight_info)
                        
                        return {"flights": flights}
                    elif data.get("flights"):
                        return data
                        
            except Exception as e:
                print(f"Endpoint {endpoint} не сработал: {e}")
                continue
                
        return None
        
    except Exception as e:
        print(f"Общая ошибка при получении реальных данных: {e}")
        return None


# --- Демо данные (надежные и богатые) ---
def get_demo_flight_data(airport):
    demo_flights = {
        "DXB": [
            {
                "flightNumber": "EK202",
                "departure": {"city": "London", "country": "UK", "iata": "LHR"},
                "airline": {"name": "Emirates"},
                "status": "Landed",
                "arrival": {"scheduledTime": "2024-01-29T14:30:00Z"}
            },
            {
                "flightNumber": "EK412", 
                "departure": {"city": "Mumbai", "country": "India", "iata": "BOM"},
                "airline": {"name": "Emirates"},
                "status": "Scheduled",
                "arrival": {"scheduledTime": "2024-01-29T18:45:00Z"}
            },
            {
                "flightNumber": "QF841",
                "departure": {"city": "Sydney", "country": "Australia", "iata": "SYD"},
                "airline": {"name": "Qantas"},
                "status": "In Flight",
                "arrival": {"scheduledTime": "2024-01-29T22:15:00Z"}
            },
            {
                "flightNumber": "EK721",
                "departure": {"city": "New York", "country": "USA", "iata": "JFK"},
                "airline": {"name": "Emirates"},
                "status": "Scheduled",
                "arrival": {"scheduledTime": "2024-01-30T06:30:00Z"}
            }
        ],
        "LHR": [
            {
                "flightNumber": "BA123",
                "departure": {"city": "New York", "country": "USA", "iata": "JFK"},
                "airline": {"name": "British Airways"},
                "status": "Landed",
                "arrival": {"scheduledTime": "2024-01-29T11:20:00Z"}
            },
            {
                "flightNumber": "LH456",
                "departure": {"city": "Frankfurt", "country": "Germany", "iata": "FRA"},
                "airline": {"name": "Lufthansa"},
                "status": "Scheduled",
                "arrival": {"scheduledTime": "2024-01-29T15:40:00Z"}
            },
            {
                "flightNumber": "AF789",
                "departure": {"city": "Paris", "country": "France", "iata": "CDG"},
                "airline": {"name": "Air France"},
                "status": "In Flight",
                "arrival": {"scheduledTime": "2024-01-29T16:25:00Z"}
            },
            {
                "flightNumber": "AA123",
                "departure": {"city": "Chicago", "country": "USA", "iata": "ORD"},
                "airline": {"name": "American Airlines"},
                "status": "Landed",
                "arrival": {"scheduledTime": "2024-01-29T13:15:00Z"}
            },
            {
                "flightNumber": "EK003",
                "departure": {"city": "Dubai", "country": "UAE", "iata": "DXB"},
                "airline": {"name": "Emirates"},
                "status": "Scheduled",
                "arrival": {"scheduledTime": "2024-01-29T20:10:00Z"}
            }
        ],
        "CDG": [
            {
                "flightNumber": "AF256",
                "departure": {"city": "New York", "country": "USA", "iata": "JFK"},
                "airline": {"name": "Air France"},
                "status": "Landed"
            },
            {
                "flightNumber": "LH987",
                "departure": {"city": "Frankfurt", "country": "Germany", "iata": "FRA"},
                "airline": {"name": "Lufthansa"},
                "status": "Scheduled"
            }
        ],
        "SIN": [
            {
                "flightNumber": "SQ321",
                "departure": {"city": "London", "country": "UK", "iata": "LHR"},
                "airline": {"name": "Singapore Airlines"},
                "status": "Landed"
            },
            {
                "flightNumber": "CX650",
                "departure": {"city": "Hong Kong", "country": "China", "iata": "HKG"},
                "airline": {"name": "Cathay Pacific"},
                "status": "In Flight"
            }
        ],
        "HKG": [
            {
                "flightNumber": "CX251",
                "departure": {"city": "London", "country": "UK", "iata": "LHR"},
                "airline": {"name": "Cathay Pacific"},
                "status": "Scheduled"
            },
            {
                "flightNumber": "SQ891",
                "departure": {"city": "Singapore", "country": "Singapore", "iata": "SIN"},
                "airline": {"name": "Singapore Airlines"},
                "status": "Landed"
            }
        ],
        "AMS": [
            {
                "flightNumber": "KL1001",
                "departure": {"city": "New York", "country": "USA", "iata": "JFK"},
                "airline": {"name": "KLM"},
                "status": "Landed"
            },
            {
                "flightNumber": "DL143",
                "departure": {"city": "Atlanta", "country": "USA", "iata": "ATL"},
                "airline": {"name": "Delta Air Lines"},
                "status": "Scheduled"
            }
        ]
    }
    return {"flights": demo_flights.get(airport, [])}

# --- DeepSeek API функция ---
async def ask_deepseek(flight_data, question, airport):
    """Запрос к DeepSeek API"""
    if not DEEPSEEK_API_KEY:
        return "DeepSeek API ключ не настроен. Используется базовый ответ."
    
    prompt = f"""
Ты — эксперт по анализу авиаданных. Отвечай на вопросы пользователя на основе предоставленных данных о рейсах.

ДАННЫЕ О РЕЙСАХ (аэропорт {airport}):
{json.dumps(flight_data, indent=2, ensure_ascii=False)}

ВОПРОС ПОЛЬЗОВАТЕЛЯ: "{question}"

ВАЖНЫЕ ИНСТРУКЦИИ:
- Отвечай ТОЛЬКО на основе предоставленных данных
- Если информации нет в данных, честно говори "Не могу найти эту информацию"
- Будь кратким и точным
- Считай количество рейсов если спрашивают о числах
- Перечисляй страны/города если спрашивают о них
- Отвечай на русском языке

ОТВЕТ:
"""
    
    try:
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.3
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Ошибка DeepSeek API: {response.status_code}"
            
    except Exception as e:
        return f"Ошибка соединения с DeepSeek: {str(e)}"

# --- Роуты ---
@app.get("/")
async def serve_frontend():
    return FileResponse("public/index.html")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "api": "DeepSeek", "framework": "FastAPI"}

@app.post("/api/ask")
async def handle_question(query: UserQuery):
    # Получаем демо-данные о рейсах
    flight_data = await get_flight_data(query.airport)
    
    # Получаем ответ от DeepSeek
    answer = await ask_deepseek(flight_data, query.question, query.airport)
    
    return {
        "answer": answer,
        "debug": {
            "flights_count": len(flight_data["flights"]),
            "airport": query.airport,
            "ai_provider": "DeepSeek"
        }
    }

# Для Vercel
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
