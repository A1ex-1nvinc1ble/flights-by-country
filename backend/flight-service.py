import requests
import os

class FlightService:
    def __init__(self):
        self.api_key = os.getenv("FLIGHT_API_KEY")
    
    def get_flight_data(self, airport):
        """Получаем данные о рейсах или используем демо данные"""
        try:
            if self.api_key:
                # Пробуем реальное API
                endpoints = [
                    f"https://api.flightapi.io/schedule/{airport}/arrivals",
                    f"https://api.flightapi.io/arrivals/{airport}",
                ]
                
                for endpoint in endpoints:
                    try:
                        response = requests.get(
                            endpoint,
                            params={
                                "mode": "live",
                                "date": "2024-01-29",
                                "api_key": self.api_key
                            },
                            timeout=10
                        )
                        if response.status_code == 200:
                            data = response.json()
                            if data.get('schedules'):
                                print(f"Successfully got flight data from {endpoint}")
                                return data
                    except Exception as e:
                        print(f"Flight API endpoint failed {endpoint}: {e}")
                        continue
            
            # Если API не сработало, используем демо данные
            return self.get_demo_data(airport)
            
        except Exception as e:
            print(f"Flight service error: {e}")
            return self.get_demo_data(airport)
    
    def get_demo_data(self, airport):
        """Демо данные для тестирования"""
        demo_flights = {
            "DXB": [
                {
                    "flightNumber": "EK202", 
                    "departure": {"city": "London", "country": "UK", "iata": "LHR"}, 
                    "airline": {"name": "Emirates"},
                    "status": "Scheduled"
                },
                {
                    "flightNumber": "EK412", 
                    "departure": {"city": "Mumbai", "country": "India", "iata": "BOM"}, 
                    "airline": {"name": "Emirates"},
                    "status": "Landed"
                },
                {
                    "flightNumber": "QF841", 
                    "departure": {"city": "Sydney", "country": "Australia", "iata": "SYD"}, 
                    "airline": {"name": "Qantas"},
                    "status": "Scheduled"
                }
            ],
            "LHR": [
                {
                    "flightNumber": "BA123", 
                    "departure": {"city": "New York", "country": "USA", "iata": "JFK"}, 
                    "airline": {"name": "British Airways"},
                    "status": "Landed"
                },
                {
                    "flightNumber": "LH456", 
                    "departure": {"city": "Frankfurt", "country": "Germany", "iata": "FRA"}, 
                    "airline": {"name": "Lufthansa"},
                    "status": "Scheduled"
                },
                {
                    "flightNumber": "AF789", 
                    "departure": {"city": "Paris", "country": "France", "iata": "CDG"}, 
                    "airline": {"name": "Air France"},
                    "status": "Scheduled"
                },
                {
                    "flightNumber": "AA123", 
                    "departure": {"city": "Chicago", "country": "USA", "iata": "ORD"}, 
                    "airline": {"name": "American Airlines"},
                    "status": "Landed"
                }
            ],
            "CDG": [
                {
                    "flightNumber": "AF100", 
                    "departure": {"city": "Tokyo", "country": "Japan", "iata": "NRT"}, 
                    "airline": {"name": "Air France"},
                    "status": "Scheduled"
                },
                {
                    "flightNumber": "KL123", 
                    "departure": {"city": "Amsterdam", "country": "Netherlands", "iata": "AMS"}, 
                    "airline": {"name": "KLM"},
                    "status": "Landed"
                }
            ]
        }
        
        return {"schedules": demo_flights.get(airport, [])}