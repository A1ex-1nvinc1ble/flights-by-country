import os
import requests
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime, timedelta

# --- Конфигурация ---
AVIATIONSTACK_KEY = os.getenv("AVIATIONSTACK_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Инициализируем FastAPI приложение
app = FastAPI()

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

# --- AviationStack API функция ---
async def get_aviationstack_data(airport):
    """Получаем реальные данные о рейсах из AviationStack API"""
    if not AVIATIONSTACK_KEY:
        print("AviationStack API ключ не настроен")
        return None
    
    try:
        # Формируем дату для запроса (сегодня)
        today = datetime.now().strftime("%Y-%m-%d")
        
        url = "http://api.aviationstack.com/v1/flights"
        params = {
            "access_key": AVIATIONSTACK_KEY,
            "arr_iata": airport,
            "flight_status": "scheduled,active,landed",
            "limit": 50  # Лимит для бесплатного плана
        }
        
        print(f"Запрос к AviationStack для аэропорта {airport}")
        
        response = requests.get(url, params=params, timeout=15)
        
        print(f"AviationStack статус: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Проверяем наличие данных
            if "data" in data and data["data"]:
                flights = []
                
                for flight in data["data"]:
                    # Парсим информацию о рейсе
                    flight_info = {
                        "flightNumber": flight.get("flight", {}).get("iata", "Unknown"),
                        "departure": {
                            "city": flight.get("departure", {}).get("timezone", "").split("/")[-1].replace("_", " ").title(),
                            "country": flight.get("departure", {}).get("country", "Unknown"),
                            "iata": flight.get("departure", {}).get("iata", "Unknown")
                        },
                        "airline": {
                            "name": flight.get("airline", {}).get("name", "Unknown")
                        },
                        "status": flight.get("flight_status", "unknown").title(),
                        "arrival": {
                            "scheduled": flight.get("arrival", {}).get("scheduled", ""),
                            "estimated": flight.get("arrival", {}).get("estimated", "")
                        }
                    }
                    
                    # Если город неизвестен, используем название аэропорта
                    if flight_info["departure"]["city"] in ["", "Unknown"]:
                        flight_info["departure"]["city"] = flight.get("departure", {}).get("airport", "Unknown")
                    
                    flights.append(flight_info)
                
                print(f"Успешно получено {len(flights)} рейсов из AviationStack")
                return {"flights": flights}
            else:
                print("AviationStack вернул пустые данные")
                return None
                
        elif response.status_code == 401:
            print("Ошибка 401: Неверный AviationStack API ключ")
            return None
        elif response.status_code == 429:
            print("Ошибка 429: Превышен лимит запросов к AviationStack")
            return None
        else:
            print(f"Ошибка AviationStack {response.status_code}: {response.text[:200]}")
            return None
            
    except requests.exceptions.Timeout:
        print("Таймаут при запросе к AviationStack")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Ошибка соединения с AviationStack: {e}")
        return None
    except Exception as e:
        print(f"Неожиданная ошибка AviationStack: {e}")
        return None

# --- Демо данные (обогащенные) ---
def get_demo_flight_data(airport):
    """Качественные демо-данные для всех аэропортов"""
    demo_flights = {
        "DXB": [
            {
                "flightNumber": "EK202",
                "departure": {"city": "London", "country": "UK", "iata": "LHR"},
                "airline": {"name": "Emirates"},
                "status": "Landed",
                "arrival": {"scheduled": "2024-01-29T14:30:00Z"}
            },
            {
                "flightNumber": "EK412", 
                "departure": {"city": "Mumbai", "country": "India", "iata": "BOM"},
                "airline": {"name": "Emirates"},
                "status": "Scheduled",
                "arrival": {"scheduled": "2024-01-29T18:45:00Z"}
            },
            {
                "flightNumber": "QF841",
                "departure": {"city": "Sydney", "country": "Australia", "iata": "SYD"},
                "airline": {"name": "Qantas"},
                "status": "In Flight",
                "arrival": {"scheduled": "2024-01-29T22:15:00Z"}
            },
            {
                "flightNumber": "EK721",
                "departure": {"city": "New York", "country": "USA", "iata": "JFK"},
                "airline": {"name": "Emirates"},
                "status": "Scheduled",
                "arrival": {"scheduled": "2024-01-30T06:30:00Z"}
            },
            {
                "flightNumber": "FZ893",
                "departure": {"city": "Sharjah", "country": "UAE", "iata": "SHJ"},
                "airline": {"name": "Flydubai"},
                "status": "Landed",
                "arrival": {"scheduled": "2024-01-29T12:20:00Z"}
            }
        ],
        "LHR": [
            {
                "flightNumber": "BA123",
                "departure": {"city": "New York", "country": "USA", "iata": "JFK"},
                "airline": {"name": "British Airways"},
                "status": "Landed",
                "arrival": {"scheduled": "2024-01-29T11:20:00Z"}
            },
            {
                "flightNumber": "LH456",
                "departure": {"city": "Frankfurt", "country": "Germany", "iata": "FRA"},
                "airline": {"name": "Lufthansa"},
                "status": "Scheduled",
                "arrival": {"scheduled": "2024-01-29T15:40:00Z"}
            },
            {
                "flightNumber": "AF789",
                "departure": {"city": "Paris", "country": "France", "iata": "CDG"},
                "airline": {"name": "Air France"},
                "status": "In Flight",
                "arrival": {"scheduled": "2024-01-29T16:25:00Z"}
            },
            {
                "flightNumber": "AA123",
                "departure": {"city": "Chicago", "country": "USA", "iata": "ORD"},
                "airline": {"name": "American Airlines"},
                "status": "Landed",
                "arrival": {"scheduled": "2024-01-29T13:15:00Z"}
            },
            {
                "flightNumber": "EK003",
                "departure": {"city": "Dubai", "country": "UAE", "iata": "DXB"},
                "airline": {"name": "Emirates"},
                "status": "Scheduled",
                "arrival": {"scheduled": "2024-01-29T20:10:00Z"}
            }
        ],
        "CDG": [
            {
                "flightNumber": "AF256",
                "departure": {"city": "New York", "country": "USA", "iata": "JFK"},
                "airline": {"name": "Air France"},
                "status": "Landed",
                "arrival": {"scheduled": "2024-01-29T09:30:00Z"}
            },
            {
                "flightNumber": "LH987",
                "departure": {"city": "Frankfurt", "country": "Germany", "iata": "FRA"},
                "airline": {"name": "Lufthansa"},
                "status": "Scheduled",
                "arrival": {"scheduled": "2024-01-29T14:20:00Z"}
            },
            {
                "flightNumber": "UA945",
                "departure": {"city": "Washington", "country": "USA", "iata": "IAD"},
                "airline": {"name": "United Airlines"},
                "status": "In Flight",
                "arrival": {"scheduled": "2024-01-29T17:45:00Z"}
            }
        ],
        "SIN": [
            {
                "flightNumber": "SQ321",
                "departure": {"city": "London", "country": "UK", "iata": "LHR"},
                "airline": {"name": "Singapore Airlines"},
                "status": "Landed",
                "arrival": {"scheduled": "2024-01-29T18:30:00Z"}
            },
            {
                "flightNumber": "CX650",
                "departure": {"city": "Hong Kong", "country": "China", "iata": "HKG"},
                "airline": {"name": "Cathay Pacific"},
                "status": "In Flight",
                "arrival": {"scheduled": "2024-01-29T21:15:00Z"}
            },
            {
                "flightNumber": "QF81",
                "departure": {"city": "Sydney", "country": "Australia", "iata": "SYD"},
                "airline": {"name": "Qantas"},
                "status": "Scheduled",
                "arrival": {"scheduled": "2024-01-30T05:40:00Z"}
            }
        ],
        "HKG": [
            {
                "flightNumber": "CX251",
                "departure": {"city": "London", "country": "UK", "iata": "LHR"},
                "airline": {"name": "Cathay Pacific"},
                "status": "Scheduled",
                "arrival": {"scheduled": "2024-01-29T16:50:00Z"}
            },
            {
                "flightNumber": "SQ891",
                "departure": {"city": "Singapore", "country": "Singapore", "iata": "SIN"},
                "airline": {"name": "Singapore Airlines"},
                "status": "Landed",
                "arrival": {"scheduled": "2024-01-29T14:25:00Z"}
            },
            {
                "flightNumber": "KA450",
                "departure": {"city": "Taipei", "country": "Taiwan", "iata": "TPE"},
                "airline": {"name": "Cathay Dragon"},
                "status": "In Flight",
                "arrival": {"scheduled": "2024-01-29T19:30:00Z"}
            }
        ],
        "AMS": [
            {
                "flightNumber": "KL1001",
                "departure": {"city": "New York", "country": "USA", "iata": "JFK"},
                "airline": {"name": "KLM"},
                "status": "Landed",
                "arrival": {"scheduled": "2024-01-29T08:45:00Z"}
            },
            {
                "flightNumber": "DL143",
                "departure": {"city": "Atlanta", "country": "USA", "iata": "ATL"},
                "airline": {"name": "Delta Air Lines"},
                "status": "Scheduled",
                "arrival": {"scheduled": "2024-01-29T12:30:00Z"}
            },
            {
                "flightNumber": "BA428",
                "departure": {"city": "London", "country": "UK", "iata": "LHR"},
                "airline": {"name": "British Airways"},
                "status": "In Flight",
                "arrival": {"scheduled": "2024-01-29T10:15:00Z"}
            }
        ]
    }
    return {"flights": demo_flights.get(airport, [])}

# --- Основная функция получения данных ---
async def get_flight_data(airport):
    """Получаем данные о рейсах - сначала AviationStack, потом демо-данные"""
    
    # Пробуем получить реальные данные из AviationStack
    if AVIATIONSTACK_KEY and AVIATIONSTACK_KEY != "your_aviationstack_key_here":
        print(f"Пытаемся получить реальные данные для {airport} через AviationStack")
        real_data = await get_aviationstack_data(airport)
        if real_data and real_data.get("flights"):
            print(f"Успешно получили реальные данные: {len(real_data['flights'])} рейсов")
            return real_data
        else:
            print("AviationStack не вернул данные, используем демо-данные")
    
    # Используем демо-данные
    return get_demo_flight_data(airport)

# --- Groq API функция (оставляем твою работающую версию) ---
async def ask_groq(flight_data, question, airport):
    """Запрос к Groq API"""
    if not GROQ_API_KEY:
        return "Groq API ключ не настроен. Используется базовый ответ."
    
    # Актуальные модели Groq
    available_models = [
        "llama-3.1-8b-instant",
        "llama-3.2-1b-preview", 
        "llama-3.2-3b-preview",
        "llama-3.1-70b-versatile",
        "mixtral-8x7b-32768",
    ]
    
    # Создаем краткое текстовое описание данных
    flights = flight_data.get("flights", [])
    flight_summary = f"Аэропорт: {airport}\n"
    flight_summary += f"Всего рейсов: {len(flights)}\n"
    
    if flights:
        countries = list(set([f["departure"]["country"] for f in flights]))
        flight_summary += f"Страны отправления: {', '.join(countries)}\n"
        
        cities = list(set([f["departure"]["city"] for f in flights]))
        flight_summary += f"Города отправления: {', '.join(cities)}\n"
        
        airlines = list(set([f["airline"]["name"] for f in flights]))
        flight_summary += f"Авиакомпании: {', '.join(airlines)}\n"
        
        flight_summary += "Примеры рейсов:\n"
        for i, flight in enumerate(flights[:5]):
            flight_summary += f"- {flight['flightNumber']}: {flight['departure']['city']} → {airport} ({flight['airline']['name']})\n"
    else:
        flight_summary += "Нет данных о рейсах\n"

    prompt = {
        "role": "user",
        "content": f"""Ты — ассистент для анализа авиарейсов. Ответь на вопрос пользователя на основе этих данных:

ДАННЫЕ:
{flight_summary}

ВОПРОС: {question}

ИНСТРУКЦИИ:
- Отвечай ТОЛЬКО на основе предоставленных данных
- Будь кратким и точным (1-3 предложения)
- Если информации нет, скажи "Не могу найти эту информацию в данных"
- Отвечай на русском языке
- Не придумывай информацию

ОТВЕТ:"""
    }
    
    # Пробуем модели по очереди
    for model in available_models:
        try:
            print(f"Пробуем модель: {model}")
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [prompt],
                    "max_tokens": 500,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "stream": False
                },
                timeout=10
            )
            
            print(f"Модель {model} - статус: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                print(f"Успешно использована модель: {model}")
                return answer
            else:
                error_detail = response.text
                print(f"Модель {model} ошибка {response.status_code}: {error_detail}")
                continue
                
        except requests.exceptions.Timeout:
            print(f"Модель {model} - таймаут")
            continue
        except Exception as e:
            print(f"Модель {model} - ошибка: {str(e)}")
            continue
    
    return "Все модели Groq API временно недоступны."

# --- Простой анализатор как fallback ---
def simple_analyzer(flight_data, question, airport):
    """Простой анализатор вопросов без использования внешнего API"""
    flights = flight_data.get("flights", [])
    question_lower = question.lower()
    
    if "сколько" in question_lower and ("рейс" in question_lower or "рейсов" in question_lower):
        count = len(flights)
        return f"В аэропорту {airport} прибывает {count} рейсов."
    
    elif "стран" in question_lower:
        countries = list(set([f["departure"]["country"] for f in flights]))
        if countries:
            return f"Рейсы прибывают из следующих стран: {', '.join(countries)}."
        else:
            return "Информация о странах отсутствует в данных."
    
    elif "авиакомпани" in question_lower or "компани" in question_lower:
        airlines = list(set([f["airline"]["name"] for f in flights]))
        if airlines:
            return f"Рейсы выполняются следующими авиакомпаниями: {', '.join(airlines)}."
        else:
            return "Информация об авиакомпаниях отсутствует в данных."
    
    elif "город" in question_lower:
        cities = list(set([f["departure"]["city"] for f in flights]))
        if cities:
            return f"Рейсы прибывают из следующих городов: {', '.join(cities)}."
        else:
            return "Информация о городах отсутствует в данных."
    
    elif "статус" in question_lower:
        statuses = {}
        for f in flights:
            status = f.get("status", "Unknown")
            statuses[status] = statuses.get(status, 0) + 1
        
        status_text = ", ".join([f"{status}: {count}" for status, count in statuses.items()])
        return f"Статусы рейсов: {status_text}."
    
    elif "откуда" in question_lower or "из каких" in question_lower:
        countries = list(set([f["departure"]["country"] for f in flights]))
        cities = list(set([f["departure"]["city"] for f in flights]))
        
        if countries and cities:
            return f"Рейсы прибывают из городов: {', '.join(cities)}. Страны: {', '.join(countries)}."
        elif countries:
            return f"Рейсы прибывают из стран: {', '.join(countries)}."
        else:
            return "Информация о происхождении рейсов отсутствует."
    
    else:
        count = len(flights)
        example_questions = [
            "Сколько всего рейсов?",
            "Из каких стран рейсы?",
            "Какие авиакомпании?",
            "Из каких городов рейсы?",
            "Какие статусы рейсов?"
        ]
        examples = "\n- " + "\n- ".join(example_questions)
        return f"По аэропорту {airport} есть информация о {count} рейсах. Вы можете спросить:{examples}"

# --- Роуты ---
@app.get("/")
async def serve_frontend():
    return FileResponse("public/index.html")

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy", 
        "apis": {
            "aviationstack": "configured" if AVIATIONSTACK_KEY and AVIATIONSTACK_KEY != "your_aviationstack_key_here" else "not configured",
            "groq": "configured" if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here" else "not configured"
        }
    }

@app.get("/api/test-aviationstack")
async def test_aviationstack():
    """Тестовый endpoint для проверки AviationStack"""
    airport = "LHR"
    
    if not AVIATIONSTACK_KEY or AVIATIONSTACK_KEY == "your_aviationstack_key_here":
        return {"error": "AviationStack ключ не настроен"}
    
    try:
        result = await get_aviationstack_data(airport)
        
        if result:
            return {
                "status": "success",
                "flights_count": len(result.get("flights", [])),
                "sample_flight": result.get("flights", [])[0] if result.get("flights") else None,
                "message": "AviationStack работает корректно"
            }
        else:
            return {
                "status": "error", 
                "message": "Не удалось получить данные от AviationStack"
            }
            
    except Exception as e:
        return {"status": "error", "message": f"Исключение: {str(e)}"}

@app.post("/api/ask")
async def handle_question(query: UserQuery):
    # Получаем данные о рейсах
    flight_data = await get_flight_data(query.airport)
    
    # Определяем источник данных для отладки
    data_source = "AviationStack" if AVIATIONSTACK_KEY and AVIATIONSTACK_KEY != "your_aviationstack_key_here" else "Demo Data"
    
    # Пробуем Groq API, если настроен
    if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
        try:
            answer = await ask_groq(flight_data, query.question, query.airport)
            ai_provider = "Groq API"
            
            # Если в ответе есть ошибка, используем fallback
            if any(keyword in answer.lower() for keyword in ["ошибка", "error", "недоступен"]):
                answer = simple_analyzer(flight_data, query.question, query.airport)
                ai_provider = "Simple Analyzer (Groq API временно недоступен)"
                
        except Exception as e:
            print(f"Groq failed: {e}")
            answer = simple_analyzer(flight_data, query.question, query.airport)
            ai_provider = "Simple Analyzer (Исключение в Groq API)"
    else:
        # Используем простой анализатор
        answer = simple_analyzer(flight_data, query.question, query.airport)
        ai_provider = "Simple Analyzer"
    
    return {
        "answer": answer,
        "debug": {
            "flights_count": len(flight_data["flights"]),
            "airport": query.airport,
            "ai_provider": ai_provider,
            "data_source": data_source
        }
    }

# Для Vercel
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
