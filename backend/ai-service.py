import requests
import json
import os

class AIService:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
    
    def ask_question(self, flight_data, question, airport):
        """Задаем вопрос AI на основе данных о рейсах"""
        try:
            if not self.api_key:
                return "AI service is not configured. Please check API keys."
            
            prompt = f"""
You are an expert flight data analyst. Answer the user's question based ONLY on the provided flight data.

AIRPORT: {airport}
FLIGHT DATA: {json.dumps(flight_data, indent=2)}

USER QUESTION: {question}

IMPORTANT INSTRUCTIONS:
- Answer based ONLY on the data provided above
- If the data doesn't contain the information, say "I cannot find that information in the current flight data"
- Be concise and factual
- If counting flights, provide the count
- If listing countries/cities, provide the list
- Do not make up or hallucinate information

ANSWER:
"""
            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.2
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                error_msg = f"AI service error: {response.status_code}"
                if response.status_code == 401:
                    error_msg += " - Invalid API key"
                return error_msg
                
        except requests.exceptions.Timeout:
            return "AI service timeout. Please try again."
        except Exception as e:
            return f"AI service unavailable. Error: {str(e)}"