import { useState } from 'react';
import axios from 'axios';

function App() {
  // Состояния для элементов формы и ответа
  const [selectedAirport, setSelectedAirport] = useState('DXB');
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Список аэропортов из задания
  const airports = [
    { code: 'DXB', name: 'Dubai International' },
    { code: 'LHR', name: 'London Heathrow' },
    { code: 'CDG', name: 'Paris Charles de Gaulle' },
    { code: 'SIN', name: 'Singapore Changi' },
    { code: 'HKG', name: 'Hong Kong International' },
    { code: 'AMS', name: 'Amsterdam Schiphol' }
  ];

  // Функция для обработки отправки формы
  const handleSubmit = async (e) => {
    e.preventDefault(); // Предотвращаем перезагрузку страницы
    if (!question.trim()) return; // Игнорируем пустые вопросы

    setIsLoading(true);
    setAnswer(''); // Очищаем предыдущий ответ

    try {
      // Отправляем запрос на наш бэкенд (API route Vercel)
      const response = await axios.post('/api/ask', {
        airport: selectedAirport,
        question: question
      });
      setAnswer(response.data.answer);
    } catch (error) {
      console.error('Error fetching the answer:', error);
      setAnswer('Sorry, an error occurred while processing your question. Please check if the server is running and API keys are configured.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-cyan-50 to-blue-100 py-12 px-4">
      <div className="max-w-2xl mx-auto bg-white rounded-xl shadow-md p-8">
        <h1 className="text-3xl font-bold text-center text-gray-800 mb-2">✈️ Flights by Country</h1>
        <p className="text-center text-gray-600 mb-8">Explore flight data for major airports worldwide.</p>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Dropdown для выбора аэропорта */}
          <div>
            <label htmlFor="airport" className="block text-sm font-medium text-gray-700 mb-1">
              Select Airport
            </label>
            <select
              id="airport"
              value={selectedAirport}
              onChange={(e) => setSelectedAirport(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-lg shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              {airports.map((airport) => (
                <option key={airport.code} value={airport.code}>
                  {airport.code} - {airport.name}
                </option>
              ))}
            </select>
          </div>

          {/* Поле для ввода вопроса */}
          <div>
            <label htmlFor="question" className="block text-sm font-medium text-gray-700 mb-1">
              Your Question
            </label>
            <input
              type="text"
              id="question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g., How many flights arrived from Germany?"
              className="w-full p-3 border border-gray-300 rounded-lg shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              disabled={isLoading}
            />
          </div>

          {/* Кнопка отправки */}
          <button
            type="submit"
            disabled={isLoading || !question.trim()}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white font-semibold py-3 px-4 rounded-lg transition duration-200"
          >
            {isLoading ? 'Thinking...' : 'Ask Question'}
          </button>
        </form>

        {/* Отображение ответа */}
        {answer && (
          <div className="mt-8 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <h2 className="text-lg font-semibold text-gray-800 mb-2">Answer:</h2>
            <p className="text-gray-700 whitespace-pre-wrap">{answer}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;