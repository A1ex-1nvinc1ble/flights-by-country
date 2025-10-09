from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import requests
import json

app = Flask(__name__)
CORS(app)

class FlightService:
    def get_flight_data(self, airport):
        """Демо данные - надежно работают"""
        demo_flights = {
            "DXB": [
                {"flightNumber": "EK202", "departure": {"city": "London", "country": "UK"}, "airline": "Emirates"},
                {"flightNumber": "EK412", "departure": {"city": "Mumbai", "country": "India"}, "airline": "Emirates"},
            ],
            "LHR": [
                {"flightNumber": "BA123", "departure": {"city": "New York", "country": "USA"}, "airline": "British Airways"},
                {"flightNumber": "LH456", "departure": {"city": "Frankfurt", "country": "Germany"}, "airline": "Lufthansa"},
            ]
        }
        return {"schedules": demo_flights.get(airport, [])}

class AIService:
    def ask_question(self, flight_data, question, airport):
        """Упрощенный AI - всегда работает"""
        flights = flight_data.get("schedules", [])
        
        if "how many" in question.lower():
            return f"There are {len(flights)} flights arriving at {airport}."
        
        if "country" in question.lower():
            countries = list(set([f["departure"]["country"] for f in flights]))
            return f"Flights are arriving from: {', '.join(countries)}"
        
        if "airline" in question.lower():
            airlines = list(set([f["airline"] for f in flights]))
            return f"Airlines: {', '.join(airlines)}"
        
        return f"I found {len(flights)} flights for {airport}. Ask me about countries, airlines, or flight counts."

flight_service = FlightService()
ai_service = AIService()

@app.route('/')
def serve_frontend():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('../frontend', path)

@app.route('/api/health')
def health_check():
    return jsonify({"status": "healthy", "message": "Flask API is working!"})

@app.route('/api/test')
def test_api():
    return jsonify({"message": "Test endpoint works!", "data": [1, 2, 3]})

@app.route('/api/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        airport = data.get('airport', 'LHR')
        question = data.get('question', 'How many flights?')
        
        flight_data = flight_service.get_flight_data(airport)
        answer = ai_service.ask_question(flight_data, question, airport)
        
        return jsonify({
            "answer": answer,
            "debug": {
                "airport": airport,
                "flights_count": len(flight_data.get("schedules", [])),
                "status": "success"
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Для локальной разработки
if __name__ == '__main__':
    app.run(debug=True)