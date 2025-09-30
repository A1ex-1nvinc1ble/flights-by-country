// api/ask/index.js
import axios from 'axios';

// Используем переменные окружения
const FLIGHT_API_KEY = process.env.FLIGHT_API_KEY;
const DEEPSEEK_API_KEY = process.env.OPENAI_API_KEY; // Переименуем для ясности

export default async function handler(req, res) {
  // Разрешаем только POST-запросы
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method Not Allowed' });
  }

  const { airport, question } = req.body;

  // Базовая валидация
  if (!airport || !question) {
    return res.status(400).json({ message: 'Airport and question are required.' });
  }

  try {
    console.log(`Fetching data for airport: ${airport}`);

    // 1. Получаем данные о расписании из FlightAPI
    const flightApiUrl = `https://api.flightapi.io/schedule/${airport}/arrivals`;
    const flightApiParams = {
      mode: 'live',
      date: new Date().toISOString().split('T')[0],
      api_key: FLIGHT_API_KEY
    };

    console.log('Requesting FlightAPI with key:', FLIGHT_API_KEY ? 'Key exists' : 'Key missing');

    const flightResponse = await axios.get(flightApiUrl, { params: flightApiParams });

    // Проверяем успешность запроса
    if (flightResponse.status !== 200 || !flightResponse.data) {
      throw new Error(`Flight API error: ${flightResponse.status}`);
    }

    const flightData = flightResponse.data;
    console.log(`Received ${flightData?.schedules?.length || 0} flights.`);

    // 2. Подготавливаем промпт для LLM
    const prompt = `
You are an expert flight data analyst. Your task is to answer questions based solely on the provided flight schedule data.

CONTEXT DATA (Flight Arrivals for airport ${airport}):
${JSON.stringify(flightData, null, 2)}

USER QUESTION: "${question}"

IMPORTANT INSTRUCTIONS:
- Answer the question based ONLY on the data provided above.
- If the data does not contain information needed to answer the question, clearly state "I cannot find that information in the current flight data."
- Be concise and factual. Do not hallucinate or make up information.
- If counting, list the count and optionally a few examples if relevant.
- If asking about cities/countries, list them.

ANSWER:
`;

    console.log('Sending request to DeepSeek API...');

    // 3. Запрос к DeepSeek API
    const deepSeekResponse = await axios.post(
      'https://api.deepseek.com/chat/completions',
      {
        model: 'deepseek-chat',
        messages: [
          {
            role: 'user',
            content: prompt
          }
        ],
        max_tokens: 500,
        temperature: 0.2,
        stream: false
      },
      {
        headers: {
          'Authorization': `Bearer ${DEEPSEEK_API_KEY}`,
          'Content-Type': 'application/json'
        }
      }
    );

    const llmAnswer = deepSeekResponse.data.choices[0].message.content;
    console.log('Received answer from DeepSeek');

    // 4. Отправляем ответ обратно на фронтенд
    res.status(200).json({ answer: llmAnswer });

  } catch (error) {
    console.error('Error in API handler:', error.message);
    
    // Детальная информация об ошибке
    if (error.response) {
      console.error('API Response error:', error.response.status, error.response.data);
    }

    let errorMessage = 'Failed to process your question.';
    if (error.response?.status === 401) {
      errorMessage = 'API key is invalid or missing.';
    } else if (error.code === 'ENOTFOUND') {
      errorMessage = 'Network error. Please check your connection.';
    } else if (error.message.includes('Flight API')) {
      errorMessage = 'Flight data service is temporarily unavailable.';
    }

    res.status(500).json({ answer: `Error: ${errorMessage}` });
  }
}