// Flights by Country - Frontend JavaScript

class FlightsApp {
    constructor() {
        this.init();
    }

    init() {
        this.bindEvents();
        console.log('Flights by Country app initialized');
    }

    bindEvents() {
        const form = document.getElementById('questionForm');
        if (form) {
            form.addEventListener('submit', (e) => this.handleSubmit(e));
        }

        // Enter key support
        const questionInput = document.getElementById('question');
        if (questionInput) {
            questionInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.handleSubmit(e);
                }
            });
        }
    }

    async handleSubmit(e) {
        e.preventDefault();
        
        const airport = document.getElementById('airport').value;
        const question = document.getElementById('question').value;
        const submitBtn = document.getElementById('submitBtn');
        const answerDiv = document.getElementById('answer');
        const answerText = document.getElementById('answerText');
        
        if (!question.trim()) {
            this.showError('Please enter a question');
            return;
        }
        
        // Show loading state
        this.setLoadingState(true, submitBtn);
        this.hideAnswer(answerDiv);
        
        try {
            const response = await this.sendQuestion(airport, question);
            
            if (response.ok) {
                const data = await response.json();
                this.showAnswer(data.answer, answerText, answerDiv);
            } else {
                const errorData = await response.json();
                this.showError(errorData.error || 'Unknown server error', answerText, answerDiv);
            }
            
        } catch (error) {
            console.error('Network error:', error);
            this.showError('Network error: ' + error.message, answerText, answerDiv);
        } finally {
            this.setLoadingState(false, submitBtn);
        }
    }

    async sendQuestion(airport, question) {
        return await fetch('/api/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                airport: airport,
                question: question
            })
        });
    }

    setLoadingState(isLoading, button) {
        if (isLoading) {
            button.innerHTML = '<div class="loading-spinner"></div>Thinking...';
            button.disabled = true;
            button.classList.add('opacity-50');
        } else {
            button.innerHTML = 'Ask Question';
            button.disabled = false;
            button.classList.remove('opacity-50');
        }
    }

    showAnswer(answer, answerElement, answerContainer) {
        answerElement.textContent = answer;
        answerContainer.classList.remove('hidden');
        answerContainer.classList.add('fade-in');
    }

    showError(message, answerElement, answerContainer) {
        answerElement.textContent = `Error: ${message}`;
        answerContainer.classList.remove('hidden');
        answerContainer.classList.add('fade-in');
        answerContainer.classList.add('bg-red-50', 'border-red-200');
    }

    hideAnswer(answerContainer) {
        answerContainer.classList.add('hidden');
        answerContainer.classList.remove('bg-red-50', 'border-red-200');
        answerContainer.classList.remove('fade-in');
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new FlightsApp();
});

// Utility functions
const utils = {
    formatFlightNumber: (number) => {
        return number.replace(/([A-Z]+)(\d+)/, '$1 $2');
    },
    
    capitalize: (str) => {
        return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
    },
    
    debounce: (func, wait) => {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
};