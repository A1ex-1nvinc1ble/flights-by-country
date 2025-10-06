from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import requests
import json
from flight_service import FlightService
from ai_service import AIService

app = Flask(__name__)
CORS(app)  # Разрешаем запросы с фронтенда

# Инициализируем службы
flight_service = FlightService()
ai_service = AIService()

@app.route('/')
def serve_frontend():
    """Отдаем фронтенд"""
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Статические файлы фронтенда"""
    return send_from_directory('../frontend', path)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка здоровья API"""
    return jsonify({"status": "healthy", "framework": "Flask"})

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Основной endpoint для вопросов"""
    try:
        # Получаем данные из запроса
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        airport = data.get('airport')
        question = data.get('question')
        
        if not airport or not question:
            return jsonify({"error": "Airport and question are required"}), 400
        
        print(f"Processing request: {airport} - {question}")
        
        # Получаем данные о рейсах
        flight_data = flight_service.get_flight_data(airport)
        
        # Получаем ответ от AI
        answer = ai_service.ask_question(flight_data, question, airport)
        
        # Формируем ответ
        response = {
            "answer": answer,
            "debug": {
                "flights_count": len(flight_data.get("schedules", [])),
                "airport": airport,
                "framework": "Flask"
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in ask_question: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# Для Vercel нужно указать app
if __name__ == '__main__':
    app.run(debug=True)
else:
    # Для production на Vercel
    application = app